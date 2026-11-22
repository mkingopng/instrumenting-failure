"""The provenance keys every analysis `run_meta.json` carries.

One helper, one key set, every frozen analysis entrypoint. Without it, an
export that records no git sha cannot say which tree produced it, and drift in
a library version is invisible: `robustness.csv`'s bootstrap bounds can move
by up to 0.0215 between two runs with no point estimate change and nothing in
the artifacts to attribute it to.

The keys are identical across entrypoints and live in the same `run_meta.json`
the artifact beside them is keyed to, never in a sidecar or a global file: a
provenance record a reader has to go and find elsewhere is one they will read
for the wrong run.

`bootstrap_seed` and `n_resamples` are `None` where the entrypoint draws no
bootstrap. `None` here means "not applicable to this artifact", which is a
different statement from an absent key (that would mean "this entrypoint has
not been taught to record it") -- so every entrypoint writes both keys.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

#: The repo this package lives in: `__file__`, never the cwd, so a run from any
#: directory records the tree it actually imported.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: The paths whose uncommitted state `git_diff_sha256` identifies. It answers "which code ran",
#: so it covers what executes or configures a run (`src`, `scripts`, `configs`, `tests`,
#: `pyproject.toml`, `uv.lock`, `Makefile`) and nothing else. The analysis output directories are
#: excluded on purpose: they are tracked, so regenerating one arm's outputs would otherwise move
#: the digest recorded beside every other arm made from identical code. Untracked files inside
#: these paths are hashed too, by name and content, because `git diff HEAD` cannot see them.
DIFF_CODE_PATHS = (
    "src",
    "scripts",
    "configs",
    "tests",
    "pyproject.toml",
    "uv.lock",
    "Makefile",
)

#: The exact key set every analysis `run_meta.json` must carry. A standing test
#: per entrypoint asserts its record is a superset of this tuple.
ENVIRONMENT_KEYS = (
    "git_sha",
    "git_dirty",
    "git_diff_sha256",
    "bootstrap_seed",
    "n_resamples",
    "numpy_version",
    "scipy_version",
    "polars_version",
    "python_version",
)


def git_sha() -> tuple[str, str]:
    """`(sha, note)`. The sha is the string "unknown" and the note says why."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "unknown", f"git rev-parse HEAD failed: {exc}"
    return completed.stdout.strip(), "git rev-parse HEAD"


def _pathspec_excluding(
    out_dir: Path | str | None, roots: tuple[str, ...] = (".",)
) -> tuple[list[str], str]:
    """
    the git pathspec that measures `roots` except the run's own output
    :param out_dir: where the artifacts are being written, or None to exclude nothing (Path | str | None)
    :param roots: the paths to measure, the whole tree by default (tuple[str, ...])
    :returns: the pathspec list and a note saying what it excludes (tuple[list[str], str])
    """
    pathspec = list(roots)
    excluded = "no path excluded"
    if out_dir is not None:
        try:
            rel_out = Path(out_dir).resolve().relative_to(REPO_ROOT.resolve())
        except ValueError:
            excluded = "no path excluded (the out dir is outside the repo)"
        else:
            if rel_out == Path("."):
                # `:(exclude).` would exclude the whole repo and report every
                # tree as clean, which is the silent fallback this exists to
                # prevent. Measure the whole tree and say so.
                excluded = "no path excluded (the out dir IS the repo root)"
            elif roots != (".",) and not any(
                rel_out == Path(root) or Path(root) in rel_out.parents for root in roots
            ):
                excluded = "the out directory is outside the measured paths"
            else:
                pathspec.append(f":(exclude){rel_out.as_posix()}")
                excluded = f"excluding {rel_out.as_posix()} (the out directory)"
    return pathspec, excluded


def git_dirty(out_dir: Path | str | None) -> tuple[bool | None, str]:
    """
    was the tree that produced these artifacts modified? `git_sha` alone cannot
    say: a run from a dirty working tree stamps the sha of the last commit, not
    of the code that ran. Everything under `out_dir` is excluded, because the
    artifacts being written are the run's own expected change and would make
    every run look dirty. `(None, note)` when git could not be run; the note
    always says what was measured or why it was not (no silent fallbacks)
    :param out_dir: where the artifacts are being written, or None for the whole tree (Path | str | None)
    :returns: `(dirty, note)` (tuple[bool | None, str])
    """
    pathspec, excluded = _pathspec_excluding(out_dir)
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--", *pathspec],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git status --porcelain failed: {exc}"
    return bool(completed.stdout.strip()), f"git status --porcelain, {excluded}"


def _file_sha256(path: Path) -> bytes:
    """
    the sha256 of a file's content, read in chunks so a large untracked file is never held in memory
    :param path: the file (Path)
    :returns: the 32-byte digest (bytes)
    :raises OSError: when the file cannot be read, including a broken symlink or a directory
    """
    return bytes.fromhex(sha256_hex(path))


def _untracked_fingerprint(path: Path) -> tuple[bytes, bool]:
    """
    fingerprint one untracked path without ever following a link or opening anything that is not
    a regular file, so a link to a device or a FIFO cannot block the run
    :param path: the path git listed as untracked (Path)
    :returns: the bytes to hash, and whether the path's content was covered (tuple[bytes, bool])
    """
    try:
        info = os.lstat(path)
    except OSError:
        return b"missing", False
    if stat.S_ISLNK(info.st_mode):
        try:
            return b"symlink\0" + os.fsencode(os.readlink(path)), True
        except OSError:
            return b"symlink-unreadable", False
    if not stat.S_ISREG(info.st_mode):
        return b"not-a-regular-file", False
    executable = b"x" if info.st_mode & 0o111 else b"-"
    try:
        return b"file\0" + executable + b"\0" + _file_sha256(path), True
    except OSError:
        return b"unreadable", False


def git_diff_sha256(out_dir: Path | str | None) -> tuple[str | None, str]:
    """
    the sha256 that identifies the uncommitted code of the tree that ran: `git diff HEAD --binary`
    over `DIFF_CODE_PATHS`, then each untracked path inside those paths, sorted by name, so two
    artifacts stamped with the same `git_sha` can be told apart by which uncommitted change
    produced them. Regenerating an analysis directory does not move it, and a new script does.
    The diff is asked for in one fixed form (full object names, three lines of context, quoted
    paths, no colour, no renames, no external or textconv drivers, fixed path prefixes), so the
    git settings that change the text of a diff do not change the digest.
    An untracked path is fingerprinted, not read blindly: a regular file by its executable bit
    and content, a symlink by its target text (as a tracked one appears in a diff), and anything
    else, including a nested repository, by name only and counted in the note as not covered.
    Which untracked files count depends only on the tree's own `.gitignore` files, never on the
    machine's global ignore file or `.git/info/exclude`, so one tree hashes the same everywhere.
    A tree with no change and no untracked path in the measured paths hashes the empty diff.
    Everything under `out_dir` is excluded from both the diff and the untracked scan. A file new
    since HEAD is fingerprinted like an untracked one whether or not it was `git add`ed, so the
    index state is not part of the digest. A nested repository, such as a submodule, is still
    hashed by name only and counted as not covered. `(None, note)` when git could not be run,
    naming the step that failed
    :param out_dir: where the artifacts are being written, or None (Path | str | None)
    :returns: `(hex digest, note)` (tuple[str | None, str])
    """
    pathspec, excluded = _pathspec_excluding(out_dir, DIFF_CODE_PATHS)
    try:
        added_list = subprocess.run(
            [
                "git",
                "diff",
                "HEAD",
                "--name-only",
                "-z",
                "--no-renames",
                "--diff-filter=A",
                "--",
                *pathspec,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=120,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git diff HEAD failed (the newly added scan): {exc}"
    # A file new since HEAD hashes the same whether or not it was `git add`ed: an added one leaves the
    # diff and joins the untracked fingerprints below. The scan compares HEAD with the working tree,
    # not with the index, so a file staged and then deleted from disk is not listed at all.
    added = sorted(name for name in added_list.stdout.split(b"\0") if name)
    diff_pathspec = pathspec + [
        f":(exclude,literal){os.fsdecode(name)}" for name in added
    ]
    try:
        diff = subprocess.run(
            [
                "git",
                "-c",
                "core.quotePath=true",
                "diff",
                "HEAD",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                "--no-renames",
                "-U3",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "--",
                *diff_pathspec,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=120,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git diff HEAD failed: {exc}"
    try:
        untracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-per-directory=.gitignore",
                "-z",
                "--",
                *pathspec,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=120,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git ls-files (the untracked scan) failed: {exc}"
    digest = hashlib.sha256(diff.stdout)
    untracked_names = [name for name in untracked.stdout.split(b"\0") if name]
    names = sorted(set(untracked_names) | set(added))
    uncovered = 0
    for name in names:
        fingerprint, covered = _untracked_fingerprint(REPO_ROOT / os.fsdecode(name))
        digest.update(b"\0untracked\0" + name + b"\0" + fingerprint)
        uncovered += not covered
    plural = "" if len(untracked_names) == 1 else "s"
    newly = f" and {len(added)} newly added (hashed as untracked)" if added else ""
    note = (
        f"sha256 of git diff HEAD --binary over {', '.join(DIFF_CODE_PATHS)}, plus "
        f"{len(untracked_names)} untracked file{plural}{newly} in those paths by content, "
        f"executable bit and name, {excluded}"
    )
    if uncovered:
        note += (
            f", and {uncovered} untracked path{'' if uncovered == 1 else 's'} could not be "
            "covered and hashed by name only"
        )
    return digest.hexdigest(), note


def _version(module_name: str) -> str | None:
    """The installed version of `module_name`, or None if it is not importable.

    `None`, not "unknown": scipy is an optional dependency of some entrypoints,
    and a version string invented for a module that is not there would be a
    fabricated record.
    """
    import importlib

    try:
        return getattr(importlib.import_module(module_name), "__version__", None)
    except ImportError:
        return None


def environment_record(
    out_dir: Path | str | None,
    *,
    bootstrap_seed: int | None = None,
    n_resamples: int | None = None,
) -> dict:
    """
    the `ENVIRONMENT_KEYS` block for one artifact's `run_meta.json`, plus the
    notes that say how the git facts were obtained
    :param out_dir: where the artifacts are being written; excluded from the dirty check and the diff hash (see `git_dirty`), `None` measures the whole tree (Path | str | None)
    :param bootstrap_seed: the seed the artifact's bootstrap used, or None where the entrypoint draws none (int | None)
    :param n_resamples: the replicate count, or None likewise (int | None)
    :returns: a dict carrying every key in `ENVIRONMENT_KEYS` (dict)
    """
    sha, sha_note = git_sha()
    dirty, dirty_note = git_dirty(out_dir)
    diff_sha, diff_note = git_diff_sha256(out_dir)
    return {
        "git_sha": sha,
        "git_sha_note": sha_note,
        "git_dirty": dirty,
        "git_dirty_note": dirty_note,
        "git_diff_sha256": diff_sha,
        "git_diff_sha256_note": diff_note,
        "bootstrap_seed": bootstrap_seed,
        "n_resamples": n_resamples,
        "numpy_version": _version("numpy"),
        "scipy_version": _version("scipy"),
        "polars_version": _version("polars"),
        "python_version": sys.version.split()[0],
    }


#: Where every entrypoint's environment record lives inside a `run_meta.json`.
#: A map keyed by entrypoint, because one output directory can hold artifacts
#: written by three different scripts (the arm export dir holds
#: `properties.csv`, `robustness.csv` and `per_infoset.csv`) with different
#: seeds and replicate counts. Flat keys would have the last writer overwrite
#: the others' bootstrap parameters with its own, which is a false record in
#: the one file a reader trusts.
PROVENANCE_RECORDS_KEY = "provenance_records"


def merge_run_meta(
    path: Path | str,
    entrypoint: str,
    record: dict,
    *,
    extra: dict | None = None,
) -> dict:
    """
    Write `record` into `path`'s `provenance_records[entrypoint]`, in place.

    Reads the existing `run_meta.json` when there is one and preserves every
    other key, so three entrypoints writing into one arm export directory each
    record their own provenance without erasing the others'. An unreadable or
    non-object file is replaced (and the fact recorded under
    `provenance_records_note`), never merged into blindly.

    Returns the merged mapping that was written.
    :param path: the `run_meta.json` to write.
    :param entrypoint: the script's name, e.g. "scripts/analyse_stats.py".
    :param record: an `environment_record` result.
    :param extra: entrypoint-specific keys stored beside it in the same block
        (`alpha`, the CFR cache dir, ...). A key the record already carries is refused.
    :raises ValueError: when an `extra` key collides with a record key.
    :raises TypeError: when an `extra` key is not a string.
    :raises PermissionError: when the existing file is not writable.
    """
    import json

    path = Path(path)
    meta: dict = {}
    note = None
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
        except Exception as exc:  # a finding, never fatal
            note = f"previous {path.name} was unreadable and was replaced: {exc}"
        else:
            if isinstance(loaded, dict):
                meta = loaded
            else:
                note = (
                    f"previous {path.name} parsed as {type(loaded).__name__}, "
                    "not an object, and was replaced"
                )
    extra = dict(extra or {})
    non_string = sorted(repr(k) for k in extra if not isinstance(k, str))
    if non_string:
        raise TypeError(
            f"merge_run_meta({entrypoint!r}): extra keys must be strings, got {non_string}"
        )
    colliding = sorted(set(extra) & set(record))
    if colliding:
        raise ValueError(
            f"merge_run_meta({entrypoint!r}): extra keys {colliding} would replace the "
            "recorded values of the same name; rename them in the caller's extra"
        )
    records = meta.get(PROVENANCE_RECORDS_KEY)
    if not isinstance(records, dict):
        records = {}
    records[entrypoint] = {**record, **extra}
    meta[PROVENANCE_RECORDS_KEY] = records
    if note:
        meta["provenance_records_note"] = note
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(
            f"{path} is not writable; a locked file is changed through its own unlock path, "
            "never replaced"
        )
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(meta, indent=2) + "\n")
        if path.exists():
            # The replacement keeps the old file's permission bits, not the umask's.
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return meta


#: The `status` of a record whose pickle was hashed.
RECORDED = "recorded"
#: The `status` prefix of a record that could not be made. Never "unchanged" and never an empty hash.
NOT_RECORDED = "not recorded"


def sha256_hex(path: str | Path) -> str:
    """
    :param path: a file (str | Path)
    :returns: the hex sha256 of its bytes, read in chunks (str)
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_record(
    game: Any, game_name: str, iterations: int, cache_dir: str | Path
) -> dict:
    """
    the identity of the CFR reference pickle an analysis loaded for one game
    :param game: the loaded OpenSpiel game, as handed to `load_or_solve_cfr` (pyspiel.Game)
    :param game_name: the registry key of the game (str)
    :param iterations: the CFR iteration count that is part of the cache key (int)
    :param cache_dir: the directory holding the cached policies (str | Path)
    :returns: `game`, `iterations`, `solver`, `blob` (the path), `sha256` and `status`; when the
        pickle cannot be found or read, `sha256` is None and `status` begins with "not recorded"
        and says why (dict)
    """
    from lmopt.games import policy_cache

    components = policy_cache.cache_components(
        game,
        iters=iterations,
        solver="cfr+",
        code_hash=policy_cache._combined_sha256(
            policy_cache._CFR_SOLVER_FILE, policy_cache._CFR_ORCHESTRATION_FILE
        ),
    )
    blob_path, _meta_path = policy_cache.paths(cache_dir, "cfr", components)
    record = {
        "game": game_name,
        "iterations": iterations,
        "solver": "cfr+",
        "blob": os.path.basename(blob_path),
        "sha256": None,
        "status": f"{NOT_RECORDED}: no file at the path the cache key names",
    }
    if not os.path.exists(blob_path):
        return record
    try:
        record["sha256"] = sha256_hex(blob_path)
    except OSError as exc:
        record["status"] = f"{NOT_RECORDED}: {exc}"
        return record
    record["status"] = RECORDED
    return record
