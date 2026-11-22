"""
Certified run and composite directories are read-only, and a certified file that has other
names is never rewritten.

A certified file is locked `a-w`, which stops every in-place write through any of its names
(a hard link from a test's temporary directory included) and a write that reaches the file
through `open`, `truncate` or a dataframe writer. It does not stop `os.replace` onto the file,
which succeeds, arrives unlocked and leaves every other name of the old inode with the old
bytes. So a rename-in is only safe for a file with one name, and `modify` refuses any other.

The directory is locked too, against deleting or renaming over a file by hand. The one
sanctioned way to add or replace a file in a certified directory is `modify`, which holds the
shared pull-and-modify lock, unlocks the directory for the duration, and fails if anything but
the named target changed. A SIGKILL inside that window leaves the directory unlocked until a
standing check names it.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import socket
import stat
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

LOCK_NAME = ".certified.lock"
LOCK_ENV = "LMOPT_CERTIFIED_LOCK"
RUN_ROOT_NAME = "run_experiment"

_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
# rsync writes `.<name>.<six random characters>` beside a file it is replacing
_RSYNC_TEMPORARY = re.compile(r"^\..+\.[A-Za-z0-9]{6}$")
_POLL_SECONDS = 0.05
_HELD: set[tuple[str, int]] = set()
_HELD_GUARD = threading.Lock()


class CertifiedError(Exception):
    """Base class for a refusal by this module."""


class LockUnresolvable(CertifiedError):
    """No lock file can be derived for the directory."""


class LockHeld(CertifiedError):
    """The lock is held by another process and the caller chose not to wait."""


class ModifyRefused(CertifiedError):
    """A modification was refused before the window opened."""

    def __init__(self, message: str, other_paths: Optional[list[Path]] = None):
        super().__init__(message)
        self.other_paths = other_paths or []


class ModifyViolated(CertifiedError):
    """Something other than the target changed inside the window."""


@dataclass(frozen=True)
class OtherNames:
    """
    the names of one inode other than the queried path
    :param paths: other names found under the root, sorted (list of Path)
    :param outside_root: names the root does not contain, `st_nlink` minus every name found
        including the queried path itself; a count, because names outside the root cannot be
        listed (int)
    :param root: the directory that was walked (Path)
    """

    paths: list[Path] = field(default_factory=list)
    outside_root: int = 0
    root: Optional[Path] = None

    @property
    def count(self) -> int:
        """
        every other name, inside the root or outside it
        :returns: the total (int)
        """
        return len(self.paths) + self.outside_root


def _clear_write(path: Path) -> bool:
    """
    remove every write bit from one path, following nothing
    :param path: a regular file or a directory (Path)
    :returns: True when the mode changed (bool)
    """
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    if not mode & _WRITE_BITS:
        return False
    os.chmod(path, mode & ~_WRITE_BITS)
    return True


def _regular_files(directory: Path) -> Iterator[Path]:
    """
    every regular file under a directory, links not followed
    :param directory: the directory to walk (Path)
    :returns: the file paths (Iterator[Path])
    """
    for dirpath, _dirs, files in os.walk(directory, followlinks=False):
        for name in files:
            path = Path(dirpath) / name
            if stat.S_ISREG(os.lstat(path).st_mode):
                yield path


def lock_files(directory: str | Path) -> int:
    """
    mark every regular file under a directory unwritable. A symlink is skipped, never
    followed: `chmod` on a link would change its target
    :param directory: the run or composite directory (str or Path)
    :returns: how many files changed mode (int)
    """
    return sum(_clear_write(path) for path in _regular_files(Path(directory)))


def lock_paths(paths: Iterable[str | Path]) -> int:
    """
    mark the named regular files unwritable and touch nothing else. A symlink or a directory is
    skipped, never followed
    :param paths: the files to lock, for example the ones a pull just wrote (iterable of str or Path)
    :returns: how many files changed mode (int)
    """
    changed = 0
    for raw in paths:
        path = Path(raw)
        if path.is_symlink() or not path.is_file():
            continue
        changed += _clear_write(path)
    return changed


def lock_dir(directory: str | Path) -> int:
    """
    mark a directory and every directory under it unwritable, deepest first
    :param directory: the run or composite directory (str or Path)
    :returns: how many directories changed mode (int)
    """
    directory = Path(directory)
    below = [
        Path(dirpath) / name
        for dirpath, dirs, _files in os.walk(directory, followlinks=False)
        for name in dirs
        if not os.path.islink(Path(dirpath) / name)
    ]
    below.sort(key=lambda p: len(p.parts), reverse=True)
    return sum(_clear_write(path) for path in [*below, directory])


def _require_real_dir(directory: str | Path) -> Path:
    """
    refuse a symlink, which `chmod` would follow, and anything that is not a directory
    :param directory: the path to check (str or Path)
    :returns: the path (Path)
    :raises CertifiedError: when it is not a real directory
    """
    directory = Path(directory)
    if os.path.islink(directory) or not directory.is_dir():
        raise CertifiedError(f"{directory} is not a real directory")
    return directory


def unlock_dir(directory: str | Path) -> int:
    """
    give the owner write access to one certified directory, so it can be moved to another
    parent (POSIX needs write on a moved directory to update its `..`). Only the directory
    itself changes: its files stay locked, and `lock_dir` seals it again at the new place
    :param directory: the certified directory (str or Path)
    :returns: 1 when the mode changed, 0 when the owner could already write (int)
    :raises CertifiedError: when the path is a symlink or not a directory
    """
    directory = _require_real_dir(directory)
    mode = stat.S_IMODE(os.lstat(directory).st_mode)
    if mode & stat.S_IWUSR:
        return 0
    os.chmod(directory, mode | stat.S_IWUSR)
    return 1


def is_locked(path: str | Path) -> bool:
    """
    whether a file or directory has no write bit set for anyone
    :param path: a file or directory (str or Path)
    :returns: True when unwritable by every class (bool)
    """
    return not stat.S_IMODE(os.lstat(path).st_mode) & _WRITE_BITS


def writable_entries(directory: str | Path) -> list[Path]:
    """
    every regular file or directory under a directory, itself included, that has a write bit
    :param directory: a run or composite directory (str or Path)
    :returns: the writable paths, sorted (list of Path)
    """
    directory = Path(directory)
    found = [directory] if not is_locked(directory) else []
    for dirpath, dirs, files in os.walk(directory, followlinks=False):
        for name in [*dirs, *files]:
            path = Path(dirpath) / name
            mode = os.lstat(path).st_mode
            if (stat.S_ISREG(mode) or stat.S_ISDIR(mode)) and not is_locked(path):
                found.append(path)
    return sorted(found)


def _root_for(path: Path) -> Optional[Path]:
    """
    the nearest `run_experiment` ancestor of a path's realpath
    :param path: any path (Path)
    :returns: that ancestor, or None when the realpath has none (Optional[Path])
    """
    for parent in Path(os.path.realpath(path)).parents:
        if parent.name == RUN_ROOT_NAME:
            return parent
    return None


def _override() -> Optional[Path]:
    """
    the explicit lock file location, when one is set
    :returns: an absolute path, or None when unset or empty (Optional[Path])
    :raises LockUnresolvable: when the override is a relative path, which would resolve against
        whatever directory the process happens to be in
    """
    raw = os.environ.get(LOCK_ENV)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise LockUnresolvable(f"{LOCK_ENV} must be an absolute path, got {raw!r}")
    return path


def _lock_location(directory: str | Path) -> tuple[Path, Path]:
    """
    where the lock file lives and which directory the name search walks. The nearest
    `run_experiment` ancestor of the realpath decides both. The realpath passes through a
    worktree's per-child symlink to the primary checkout's tree, which is why a path written
    literally would be a different file in every worktree. Only a directory with no such
    ancestor (a test's temporary directory) takes the explicit override, and with neither it
    refuses: there is no fallback to the live data tree
    :param directory: a run or composite directory (str or Path)
    :returns: the lock file and the root to walk (tuple of Path)
    :raises LockUnresolvable: when there is no ancestor and no usable override
    """
    root = _root_for(Path(directory))
    if root is not None:
        return root / LOCK_NAME, root
    override = _override()
    if override is None:
        raise LockUnresolvable(
            f"{directory} has no {RUN_ROOT_NAME} ancestor on its realpath and {LOCK_ENV} is "
            "not set; refusing to guess a lock file"
        )
    return override, override.parent


def lock_path_for(directory: str | Path) -> Path:
    """
    the one lock file every holder of a directory's tree takes
    :param directory: a run or composite directory (str or Path)
    :returns: the lock file path (Path)
    :raises LockUnresolvable: when no lock file can be derived
    """
    return _lock_location(directory)[0]


def other_names(path: str | Path) -> OtherNames:
    """
    every other name of a file's inode under the lock root, and how many names lie outside it.
    The walk follows links and guards against loops. It is a whole-tree walk, meant for a
    refusal path, not a loop
    :param path: a regular file (str or Path)
    :returns: the other names found and the count outside the root (OtherNames)
    :raises LockUnresolvable: when no root can be derived
    """
    path = Path(path)
    info = os.stat(path)
    key = (info.st_dev, info.st_ino)
    me = Path(os.path.realpath(path))
    root = _lock_location(path)[1]
    if info.st_nlink == 1:
        return OtherNames(paths=[], outside_root=0, root=root)
    seen_dirs: set[tuple[int, int]] = set()
    found: set[Path] = set()
    for dirpath, dirs, files in os.walk(root, followlinks=True):
        here = os.stat(dirpath)
        marker = (here.st_dev, here.st_ino)
        if marker in seen_dirs:
            dirs[:] = []
            continue
        seen_dirs.add(marker)
        for name in files:
            candidate = Path(dirpath) / name
            try:
                other = os.stat(candidate)
            except OSError:
                continue
            if (other.st_dev, other.st_ino) == key:
                found.add(candidate)
    # one name per real location: a directory reached through a symlink would otherwise list
    # the same file twice, and the queried file itself is one name, not an "other"
    by_real = {Path(os.path.realpath(p)): p for p in found}
    by_real.pop(me, None)
    others = sorted(by_real.values())
    outside = max(0, info.st_nlink - (len(others) + 1))
    return OtherNames(paths=others, outside_root=outside, root=root)


def _holder_text(holder: str, purpose: str) -> str:
    """
    the line written into the lock file while it is held
    :param holder: who holds it (str)
    :param purpose: what for (str)
    :returns: one tab-separated line (str)
    """
    stamp = datetime.now(timezone.utc).isoformat()
    return f"{holder}\tpid={os.getpid()}\thost={socket.gethostname()}\t{purpose}\t{stamp}\n"


def _read_holder(path: Path) -> str:
    """
    who the lock file says holds it
    :param path: the lock file (Path)
    :returns: the recorded text, or a placeholder when none can be read (str)
    """
    try:
        text = path.read_text().strip()
    except OSError:
        return "an unknown holder"
    return text or "an unknown holder"


@contextlib.contextmanager
def certified_lock(
    directory: str | Path,
    holder: str,
    purpose: str,
    *,
    timeout: Optional[float] = None,
) -> Iterator[Path]:
    """
    hold the pull-and-modify lock for a directory's tree, exclusively. It waits for another
    holder by default and prints who holds it; a `timeout` makes the wait a refusal instead.
    The holder's name and purpose are written into the lock file while it is held. A process
    that already holds the lock must not take it again: the second `flock` is a different open
    file and waits for the first forever
    :param directory: a run or composite directory (str or Path)
    :param holder: who is taking it, written into the lock file (str)
    :param purpose: what for, written beside the holder (str)
    :param timeout: seconds to wait before raising, None to wait without limit (float or None)
    :returns: the lock file path, for the duration of the block (Iterator[Path])
    :raises LockUnresolvable: when no lock file can be derived
    :raises LockHeld: when `timeout` elapsed with the lock still held
    """
    lock = lock_path_for(directory)
    if not lock.parent.is_dir():
        raise LockUnresolvable(
            f"the directory for the lock file does not exist: {lock.parent}"
        )
    key = (str(lock), threading.get_ident())
    with _HELD_GUARD:
        if key in _HELD:
            raise CertifiedError(
                f"{lock} is already held by this thread; taking it again would wait on itself"
            )
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o664)
    except OSError as exc:
        raise LockUnresolvable(f"cannot open the lock file {lock}: {exc}") from exc
    held = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"waiting on {lock}, held by {_read_holder(lock)}",
                file=sys.stderr,
                flush=True,
            )
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                try:
                    if deadline is None:
                        fcntl.flock(fd, fcntl.LOCK_EX)
                    else:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise LockHeld(
                            f"{lock} is held by {_read_holder(lock)}"
                        ) from None
                    time.sleep(_POLL_SECONDS)
        held = True
        with _HELD_GUARD:
            _HELD.add(key)
        os.ftruncate(fd, 0)
        os.write(fd, _holder_text(holder, purpose).encode())
        yield lock
    finally:
        if held:
            with _HELD_GUARD:
                _HELD.discard(key)
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
        os.close(fd)


def _snapshot(directory: Path) -> dict[str, tuple[int, int, int, int]]:
    """
    what each entry of a directory looks like without following links
    :param directory: the directory (Path)
    :returns: name to (inode, mode, size, mtime_ns) (dict)
    """
    shot = {}
    for entry in os.scandir(directory):
        info = entry.stat(follow_symlinks=False)
        shot[entry.name] = (
            info.st_ino,
            stat.S_IMODE(info.st_mode),
            info.st_size,
            info.st_mtime_ns,
        )
    return shot


def _check_target_name(target: str) -> None:
    """
    refuse a target that is not one plain file name
    :param target: the proposed name (str)
    :raises ModifyRefused: when it has a separator, is empty, or is `.` or `..`
    """
    if (
        not target
        or target in (".", "..")
        or "/" in target
        or "\x00" in target
        or os.path.basename(target) != target
    ):
        raise ModifyRefused(f"target {target!r} is not a plain file name")


def _refuse_if_shared(path: Path) -> None:
    """
    refuse to replace an existing file that is not a plain regular file or that has other names
    :param path: the file about to be created or replaced (Path)
    :raises ModifyRefused: for a symlink or non-regular entry, or a file with other names (the
        exception lists the names under the root and counts those outside it)
    """
    if not os.path.lexists(path):
        return
    if os.path.islink(path) or not path.is_file():
        raise ModifyRefused(f"{path} is not a regular file")
    names = other_names(path)
    if names.count:
        listed = ", ".join(str(p) for p in names.paths) or "none under the root"
        raise ModifyRefused(
            f"{path} has {names.count} other name(s) ({listed}; "
            f"{names.outside_root} outside {names.root}): rewriting it would leave "
            "them with the old bytes. Recompose instead",
            other_paths=names.paths,
        )


def refuse_shared(directory: str | Path, names: Iterable[str]) -> None:
    """
    refuse, before the caller's first write, when any file it is about to create or replace
    already exists with other names. A writer that puts several files into one directory calls
    this once up front so a refusal on a later file cannot leave an earlier file written;
    `modify` repeats the check for its own target inside the lock
    :param directory: the certified run or composite directory (str or Path)
    :param names: every file name the caller will create or replace, in any order (iterable of str)
    :raises ModifyRefused: for a bad name, or the first file found with other names
    """
    directory = Path(directory)
    for name in names:
        _check_target_name(name)
        _refuse_if_shared(directory / name)


@contextlib.contextmanager
def modify(
    directory: str | Path,
    target: str,
    *,
    holder: str = "certified.modify",
    purpose: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Iterator[Path]:
    """
    the one sanctioned way to add or replace a file in a certified directory. Takes the shared
    lock, refuses before touching anything when an rsync temporary is present or when the
    existing target has other names (rewriting it would leave those names with the old bytes),
    snapshots the directory, unlocks it, and yields the target path for the caller to create
    or rename into. On exit, in a `finally`: a `<target>.tmp` left behind is removed, the target
    is locked, and the directory gets back the mode it had. On a clean body the directory must
    differ from the snapshot by the target only (a removed file, a new file or a mode-only
    change to another entry fails the call)
    :param directory: the certified run or composite directory (str or Path)
    :param target: the one file name that may be created or replaced (str)
    :param holder: who is modifying, written into the lock file (str)
    :param purpose: what for, defaulting to the target name (str or None)
    :param timeout: seconds to wait for the lock before raising, None to wait (float or None)
    :returns: the path of the target inside the directory (Iterator[Path])
    :raises ModifyRefused: for a bad target name, an rsync temporary, or an existing target with
        other names (the exception lists them)
    :raises ModifyViolated: when anything but the target changed inside the window
    :raises LockHeld: when `timeout` elapsed with the lock still held
    """
    directory = Path(directory)
    _check_target_name(target)
    if not directory.is_dir():
        raise ModifyRefused(f"{directory} is not a directory")
    with certified_lock(
        directory, holder, purpose or f"modify {target}", timeout=timeout
    ):
        temporaries = sorted(
            entry.name
            for entry in os.scandir(directory)
            if _RSYNC_TEMPORARY.match(entry.name)
        )
        if temporaries:
            raise ModifyRefused(
                f"{directory} holds rsync temporar{'y' if len(temporaries) == 1 else 'ies'} "
                f"{temporaries}; a transfer is in progress or died, so nothing is modified"
            )
        path = directory / target
        _refuse_if_shared(path)
        before = _snapshot(directory)
        original_mode = stat.S_IMODE(os.stat(directory).st_mode)
        os.chmod(directory, original_mode | stat.S_IWUSR)
        failed = True
        try:
            yield path
            failed = False
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(directory / f"{target}.tmp")
            if os.path.lexists(path) and not os.path.islink(path):
                _clear_write(path)
            os.chmod(directory, original_mode)
        if not failed:
            _verify_only_target_changed(directory, target, before)


def _verify_only_target_changed(
    directory: Path, target: str, before: dict[str, tuple[int, int, int, int]]
) -> None:
    """
    fail when anything but the target differs from the snapshot taken at the window's start
    :param directory: the certified directory (Path)
    :param target: the one name allowed to change (str)
    :param before: the snapshot (dict)
    :raises ModifyViolated: naming every entry that changed
    """
    after = _snapshot(directory)
    allowed = {target, f"{target}.tmp"}
    changed = sorted(
        name
        for name in set(before) | set(after)
        if name not in allowed and before.get(name) != after.get(name)
    )
    if changed:
        raise ModifyViolated(
            f"{directory}: besides {target}, these entries changed inside the window: {changed}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    """
    the command line the shell tools use: find the one lock file, or bracket the move of a
    certified directory. `unlock-dir` and `lock-dir` run under the shared lock, with a holder
    and a purpose written into it
    :param argv: arguments, None for sys.argv[1:] (Optional[list])
    :returns: 0 on success (the lock path printed for `lock-path`), 2 on a usage error or a
        directory that cannot be changed, 3 when no lock file can be derived (int)
    """
    import argparse

    parser = argparse.ArgumentParser(description="certified-directory helpers")
    sub = parser.add_subparsers(dest="command", required=True)
    where = sub.add_parser("lock-path", help="print the lock file for a directory")
    where.add_argument("directory")
    for name, text in (
        ("unlock-dir", "make a certified directory movable, before a `mv`"),
        ("lock-dir", "seal a directory again, at its new place after a `mv`"),
    ):
        step = sub.add_parser(name, help=text)
        step.add_argument("directory")
        step.add_argument("--holder", default=f"certified.{name}")
        step.add_argument("--purpose", default=name)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    try:
        if args.command == "lock-path":
            print(lock_path_for(args.directory))
        else:
            _require_real_dir(args.directory)
            with certified_lock(args.directory, args.holder, args.purpose):
                if args.command == "unlock-dir":
                    changed = unlock_dir(args.directory)
                else:
                    changed = lock_dir(args.directory)
            print(f"{args.command}: {changed} directory mode(s) changed")
    except LockUnresolvable as exc:
        print(f"certified: {exc}", file=sys.stderr)
        return 3
    except (CertifiedError, OSError) as exc:
        print(f"certified: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
