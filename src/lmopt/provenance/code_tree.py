"""
The code a collection run executed, named by git tree object ids, recorded in its `run_meta.json`.

A commit sha names a whole repository, so two checkouts at different shas can still hold the same
code, and a checkout at one sha can hold different code when a file is modified or untracked. The
tree object id of each code directory names that directory's content exactly, so a run that records
the three ids carries its own code provenance and a later reader needs no ops log.

The ids are blind to an untracked file by construction, so the record also carries every modified,
staged or untracked path under the same directories.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The directories whose content is the code a run executes, in the order the shell side lists them.
CODE_TREE_PATHS = ("src", "scripts", "configs")

REPO_ROOT = Path(__file__).resolve().parents[3]

#: What the cluster job exports after its start-of-job check passed: the three tree ids joined as
#: `src=<id>+scripts=<id>+configs=<id>` and the commit the check saw. A sweep inside the job records
#: these rather than asking git again.
ENV_IDS = "LMOPT_CODE_TREE_IDS"
ENV_HEAD = "LMOPT_CODE_TREE_HEAD"
SOURCE_JOB_CHECK = "job_check"
SOURCE_COMPUTED = "computed"
#: git attempts per call and seconds per attempt; only a timeout is retried, a git that exits
#: non-zero is a verdict and is raised at once.
GIT_ATTEMPTS = 3
GIT_TIMEOUT_S = 60
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class CodeTreeError(RuntimeError):
    """git could not name the code tree, so the run's code provenance cannot be recorded"""


def _git(args: list[str], repo_root: Path) -> str:
    """
    the standard output of one git command in the checkout, retried when git times out
    :param args: the git arguments after `git` (list of str)
    :param repo_root: the checkout (Path)
    :returns: stdout, unstripped (str)
    :raises CodeTreeError: when git cannot be run, exits non-zero, or times out on every attempt
    """
    last: Exception | None = None
    for _ in range(GIT_ATTEMPTS):
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_S,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            last = exc
            continue
        except (OSError, subprocess.SubprocessError) as exc:
            raise CodeTreeError(
                f"git {' '.join(args)} failed in {repo_root}: {exc}"
            ) from exc
        return completed.stdout
    raise CodeTreeError(
        f"git {' '.join(args)} failed in {repo_root}: {last} on each of {GIT_ATTEMPTS} attempts"
    ) from last


def code_tree_record(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """
    the tree ids and the dirty paths of the checkout's code directories
    :param repo_root: the checkout to name (Path or str)
    :returns: `{"ids": {"src": ..., "scripts": ..., "configs": ...}, "dirty": [...]}` where
        `dirty` lists each modified, staged or untracked path as `git status --porcelain` prints
        it and an empty list says the code directories are clean (dict)
    :raises CodeTreeError: when git cannot name a tree or cannot report status; a record that
        could not be made is never written as an empty one
    """
    root = Path(repo_root)
    ids = {
        path: _git(["rev-parse", "--verify", f"HEAD:{path}"], root).strip()
        for path in CODE_TREE_PATHS
    }
    status = _git(
        ["status", "--porcelain", "--untracked-files=all", "--", *CODE_TREE_PATHS], root
    )
    return {"ids": ids, "dirty": [line for line in status.splitlines() if line]}


def parse_env_ids(value: str) -> dict[str, str]:
    """
    the three tree ids a job exported
    :param value: `src=<id>+scripts=<id>+configs=<id>` (str)
    :returns: `{"src": ..., "scripts": ..., "configs": ...}` (dict of str to str)
    :raises CodeTreeError: for anything but exactly the three paths, each with a 40-hex id
    """
    parsed: dict[str, str] = {}
    for part in value.split("+"):
        path, sep, ident = part.partition("=")
        if not sep or path in parsed or not _HEX40.fullmatch(ident):
            raise CodeTreeError(
                f"{ENV_IDS} is not three `path=<40-hex id>` parts: {value!r}"
            )
        parsed[path] = ident
    if set(parsed) != set(CODE_TREE_PATHS):
        raise CodeTreeError(
            f"{ENV_IDS} must name exactly {list(CODE_TREE_PATHS)}, names {sorted(parsed)}"
        )
    return {path: parsed[path] for path in CODE_TREE_PATHS}


def head_sha(repo_root: Path | str = REPO_ROOT) -> str:
    """
    the commit the checkout's HEAD names, read from the files in `.git` with no subprocess
    :param repo_root: the checkout (Path or str)
    :returns: the 40-hex commit id (str)
    :raises CodeTreeError: when `.git/HEAD` or the ref it names cannot be read as a commit id
    """
    root = Path(repo_root)
    try:
        dot = root / ".git"
        gitdir = dot
        if dot.is_file():
            pointer = dot.read_text().split(":", 1)[1].strip()
            gitdir = (root / pointer).resolve()
        head = (gitdir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            sha = head
        else:
            ref = head[len("ref: ") :]
            common = gitdir
            if (gitdir / "commondir").is_file():
                common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve()
            sha = ""
            for base in (gitdir, common):
                if (base / ref).is_file():
                    sha = (base / ref).read_text().strip()
                    break
            if not sha and (common / "packed-refs").is_file():
                for line in (common / "packed-refs").read_text().splitlines():
                    if line.endswith(" " + ref) and not line.startswith("#"):
                        sha = line.split(" ", 1)[0]
                        break
    except (OSError, IndexError) as exc:
        raise CodeTreeError(f"cannot read HEAD of {root}: {exc}") from exc
    if not _HEX40.match(sha):
        raise CodeTreeError(f"HEAD of {root} does not name a commit: {sha!r}")
    return sha


def recorded_code_tree(
    repo_root: Path | str, environ: Mapping[str, str], in_job: bool
) -> tuple[dict[str, Any], str]:
    """
    the code tree record a sweep writes, and where it came from
    Inside a cluster job the job already named the tree once, at its start and before any GPU
    time, so the sweep records that and asks git nothing; it only checks, by reading files, that
    HEAD is still the commit the job saw, and refuses when it moved. Anywhere else (a local
    run, a test, a job queued before the export existed) the record is computed with a bounded
    retry. The exported variables are read only when `in_job` says the run is a job: a stale
    variable in a local shell is never recorded as a check that did not happen
    :param repo_root: the checkout (Path or str)
    :param environ: the process environment (mapping of str to str)
    :param in_job: whether `lmopt.logging.job` read this run as a cluster job (bool)
    :returns: (the record, "job_check" or "computed"); on the job_check path `dirty` is empty
        because the job's check refused a dirty checkout at its start, which is not a claim that
        nothing changed since: only HEAD is re-read (tuple)
    :raises CodeTreeError: when only one of the two variables is set, an id is malformed, HEAD
        moved since the job's check, or git cannot name the tree
    """
    ids_value = environ.get(ENV_IDS)
    head_value = environ.get(ENV_HEAD)
    if in_job and (ids_value or head_value):
        if not (ids_value and head_value):
            raise CodeTreeError(
                f"only one of {ENV_IDS} and {ENV_HEAD} is set; the job exports both or neither"
            )
        ids = parse_env_ids(ids_value)
        now = head_sha(repo_root)
        if now != head_value:
            raise CodeTreeError(
                f"HEAD moved during the job: the start-of-job check saw {head_value}, "
                f"the checkout is now at {now}"
            )
        return {"ids": ids, "dirty": []}, SOURCE_JOB_CHECK
    return code_tree_record(repo_root), SOURCE_COMPUTED
