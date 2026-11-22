"""The ops event log: what was done to a run, one JSON line per event.

`ops_events.jsonl` sits in the experiment root beside the run directories (the directory
that holds `recycle-v7-m0` and its siblings). It records events of A RUN: a job submitted,
a submission refused, a job killed, a job resubmitted, a run directory pulled from the cluster,
an arm stamped or refused at the stamp. Each carries a UTC time, the exit status of what was
done, the job id where one exists, (kind, model, index), and who acted. It is append-only
and never rewritten. Gate processes and their markers are not events of a run and stay out
of it.

Appends hold an exclusive lock for the whole write and use a single `write`, so two
processes cannot interleave within a line. The reader is strict: a line that is not a JSON
object carrying the required keys, or that names an unknown event, is refused with its
line number, never skipped.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

OPS_EVENTS_FILENAME = "ops_events.jsonl"

# Variables that tell git which repository to use; dropped when asking where a checkout's
# repository is.
_GIT_LOCATION_VARIABLES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
)

# A git that hangs (a lock, a stalled network filesystem) must not hang the run that is
# logging; past this it is treated like any other git failure.
_GIT_TIMEOUT_S = 30

EVENT_SUBMIT = "submit"
EVENT_RESUBMIT = "resubmit"
EVENT_REFUSE = "refuse"
EVENT_KILL = "kill"
EVENT_PULL = "pull"
EVENT_STAMP = "stamp"
EVENTS = (
    EVENT_SUBMIT,
    EVENT_RESUBMIT,
    EVENT_REFUSE,
    EVENT_KILL,
    EVENT_PULL,
    EVENT_STAMP,
)

REQUIRED_KEYS = (
    "utc",
    "event",
    "status",
    "actor",
    "session_id",
    "job_id",
    "kind",
    "model",
    "index",
)
OPTIONAL_TEXT_KEYS = (
    "detail",
    "reason",
    "job_output_path",
    "source",
    "dest",
    "job_id_source",
    "moved_aside",
)

# Where a recorded job id was read from: qsub's own output, the job's echo line, or the PBS
# epilogue (which gives the number alone, without the server suffix).
JOB_ID_SOURCES = ("qsub", "echo", "epilogue")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_WRITE_FAILED = 3


def ops_events_path(experiment_root: str | Path) -> Path:
    """
    where the log lives for an experiment root
    :param experiment_root: the directory beside the run directories (str or Path)
    :returns: the log's path (Path)
    """
    return Path(experiment_root) / OPS_EVENTS_FILENAME


_JOB_NUMBER = re.compile(r"([0-9]+)(?:\..+)?")


def job_number(job_id: str | None) -> str | None:
    """
    the one normal form of a scheduler job id: the number alone. PBS gives the same job as
    `9425945.pbs.cluster.example` (qsub, `$PBS_JOBID`, the job's echo) and as `9425945`
    (the epilogue's `Job Id:` line), so anything that compares or keys on a job id uses this
    :param job_id: an id in either form (str or None)
    :returns: the number as text, or None for None or an id that does not start with one (str
        or None)
    """
    if job_id is None:
        return None
    matched = _JOB_NUMBER.fullmatch(job_id)
    return matched.group(1) if matched else None


def canonical_root(root: str | Path) -> Path:
    """
    the one experiment root every worktree of a repository shares. A `data/run_experiment`
    inside a git checkout resolves, through the git common directory, to the main checkout's
    `data/run_experiment`, so a record written from any worktree is the same file, with no
    per-worktree setup. A directory that is not `<checkout>/data/run_experiment` of a git
    checkout, or whose common directory is not a `.git` directory, is returned unchanged:
    that is the rule for fixtures and for a root chosen on purpose
    :param root: a candidate experiment root (str or Path)
    :returns: the shared root (Path)
    """
    root = Path(root)
    if root.name != "run_experiment" or root.parent.name != "data":
        return root
    checkout = root.parent.parent
    # A GIT_DIR (or its relatives) in the environment, as a git hook or gate wrapper sets,
    # would point git at another repository whatever the checkout is.
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_VARIABLES}
    try:
        lines = subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=_GIT_TIMEOUT_S,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if (checkout / ".git").exists():
            print(
                f"ops_events: could not resolve the shared root through git for "
                f"{checkout} ({exc}); using {root}, so this checkout's log is its own file",
                file=sys.stderr,
            )
        return root
    # A git too old for --path-format echoes the argument; only an absolute last line is a path.
    common = lines[-1] if lines else ""
    if not common.startswith("/") or Path(common).name != ".git":
        return root
    return Path(common).parent / "data" / "run_experiment"


def experiment_root(checkout: str | Path) -> Path:
    """
    where a checkout's writers record events: `OPS_EVENTS_ROOT` when set (an explicit choice,
    used as given), else the repository's shared root for that checkout
    :param checkout: the repo root the writer runs from (str or Path)
    :returns: the experiment root (Path)
    """
    override = os.environ.get("OPS_EVENTS_ROOT")
    if override:
        return Path(override)
    return canonical_root(Path(checkout) / "data" / "run_experiment")


def _is_text_or_none(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _validate_record(record: dict) -> None:
    """
    refuse a record the reader would refuse
    :param record: the event as it would be written (dict)
    :raises ValueError: naming what is wrong
    """
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(f"event is missing {missing}")
    if record["event"] not in EVENTS:
        raise ValueError(f"unknown event {record['event']!r}; expected one of {EVENTS}")
    status = record["status"]
    if isinstance(status, bool) or not isinstance(status, int):
        raise ValueError(f"status must be an integer exit status, got {status!r}")
    index = record["index"]
    if index is not None and (isinstance(index, bool) or not isinstance(index, int)):
        raise ValueError(f"index must be an integer or None, got {index!r}")
    if not isinstance(record["utc"], str):
        raise ValueError("utc must be a string")
    try:
        datetime.fromisoformat(record["utc"])
    except ValueError as exc:
        raise ValueError(f"utc is not an ISO timestamp: {record['utc']!r}") from exc
    for key in ("actor", "session_id", "job_id", "kind", "model", *OPTIONAL_TEXT_KEYS):
        if not _is_text_or_none(record.get(key)):
            raise ValueError(f"{key} must be text or None, got {record.get(key)!r}")
    source = record.get("job_id_source")
    if source is not None and source not in JOB_ID_SOURCES:
        raise ValueError(
            f"job_id_source must be one of {JOB_ID_SOURCES} or None, got {source!r}"
        )


def _same_run(
    record: dict, kind: Optional[str], model: Optional[str], index: Optional[int]
) -> bool:
    return (record.get("kind"), record.get("model"), record.get("index")) == (
        kind,
        model,
        index,
    )


def _earlier_accepted_submissions(
    path: Path, kind: Optional[str], model: Optional[str], index: Optional[int]
) -> tuple[int, int]:
    """
    count the earlier submissions of one run that qsub accepted, and the lines that could not
    be read. Used only to label a submit that follows another as a resubmit; a line that
    cannot be read is counted and reported on the new event, never ignored silently
    :param path: the log (Path)
    :param kind: the run's kind (Optional[str])
    :param model: the run's model (Optional[str])
    :param index: the run's index (Optional[int])
    :returns: (earlier accepted submissions, unreadable lines) (tuple of int)
    """
    accepted = unreadable = 0
    if not path.exists():
        return 0, 0
    with open(path, encoding="utf-8") as handle:
        for text in handle:
            if not text.strip():
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                unreadable += 1
                continue
            if not isinstance(record, dict):
                unreadable += 1
                continue
            if (
                record.get("event") in (EVENT_SUBMIT, EVENT_RESUBMIT)
                and record.get("status") == 0
                and _same_run(record, kind, model, index)
            ):
                accepted += 1
    return accepted, unreadable


def _ends_with_newline(path: Path) -> bool:
    """
    whether the log's last byte is a newline
    :param path: the log (Path)
    :returns: True for an empty file or one ending in a newline (bool)
    """
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return True
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def require_actor(actor: Optional[str] = None) -> str:
    """
    the name an ops event is written under: the argument, else OPS_ACTOR; never invented
    :param actor: an explicit actor, None to read OPS_ACTOR (Optional[str])
    :returns: the actor's name (str)
    :raises ValueError: when neither names someone
    """
    who = actor if actor is not None else os.environ.get("OPS_ACTOR")
    if who is None or not who.strip():
        raise ValueError(
            "no actor: pass actor or export OPS_ACTOR=<your name>; "
            "an ops event is never written without one"
        )
    return who


def append_event(
    experiment_root: str | Path,
    event: str,
    status: int,
    *,
    kind: Optional[str] = None,
    model: Optional[str] = None,
    index: Optional[int] = None,
    job_id: Optional[str] = None,
    actor: Optional[str] = None,
    session_id: Optional[str] = None,
    **fields: Any,
) -> dict:
    """
    append one event under an exclusive lock, in a single write
    A `submit` for a run that already has an accepted submission is recorded as a `resubmit`,
    so the log says it without a reader having to work it out. If earlier lines could not be
    read, the new event carries `unreadable_lines` with their count
    :param experiment_root: the directory beside the run directories (str or Path)
    :param event: one of EVENTS (str)
    :param status: the exit status of what was done (int)
    :param kind: the run's JOBS kind (Optional[str])
    :param model: the run's model (Optional[str])
    :param index: the run's index (Optional[int])
    :param job_id: the scheduler's job id, as the scheduler gave it (Optional[str])
    :param actor: who acted; defaults to the environment's OPS_ACTOR; a row is never written
        without one (Optional[str])
    :param session_id: an id for the shell session that acted; defaults to the session-id environment
        variable read below (Optional[str])
    :param fields: any of detail, reason, job_output_path, source, dest (text or None)
    :returns: the record that was written (dict)
    :raises ValueError: when neither actor nor OPS_ACTOR names someone, or the record would not
        pass the reader
    :raises OSError: when the log cannot be written; nothing is swallowed
    """
    if not str(experiment_root).strip():
        raise ValueError(
            "the experiment root is empty; refusing to write into the working directory"
        )
    who = require_actor(actor)
    path = ops_events_path(experiment_root)
    unknown = sorted(set(fields) - set(OPTIONAL_TEXT_KEYS))
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; allowed {OPTIONAL_TEXT_KEYS}")
    record: dict[str, Any] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "status": status,
        "actor": who,
        "session_id": (
            session_id
            if session_id is not None
            else os.environ.get("OPS_SESSION_ID")
        ),
        "job_id": job_id,
        "kind": kind,
        "model": model,
        "index": index,
    }
    record.update({key: value for key, value in fields.items() if value is not None})
    _validate_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if event == EVENT_SUBMIT:
            earlier, unreadable = _earlier_accepted_submissions(
                path, kind, model, index
            )
            if earlier:
                record["event"] = EVENT_RESUBMIT
            if unreadable:
                record["unreadable_lines"] = unreadable
        line = json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n"
        payload = line.encode("ascii")
        # A writer killed mid-line leaves a line with no newline. Starting this event on a
        # fresh line keeps it intact: the torn line stays its own malformed line, which the
        # reader refuses with its number, instead of swallowing a good event.
        if os.fstat(fd).st_size and not _ends_with_newline(path):
            payload = b"\n" + payload
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
    finally:
        os.close(fd)
    return record


def read_events(path: str | Path) -> list[dict]:
    """
    read the whole log, refusing anything malformed
    :param path: the log (str or Path)
    :returns: the events in file order; an absent file is an empty list (list of dict)
    :raises ValueError: naming the file and the line number of the first malformed line
    """
    path = Path(path)
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as handle:
        for number, text in enumerate(handle, start=1):
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}: line {number} is valid JSON but not an object"
                )
            try:
                _validate_record(record)
            except ValueError as exc:
                raise ValueError(f"{path}: line {number}: {exc}") from exc
            events.append(record)
    return events


def events_for_runs(
    events: list[dict],
    kinds: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    indices: Optional[list[int]] = None,
) -> list[dict]:
    """
    the events of the runs a selection names; a filter left None matches everything
    :param events: events as read (list of dict)
    :param kinds: JOBS kinds (Optional[list])
    :param models: models (Optional[list])
    :param indices: run indices (Optional[list])
    :returns: the matching events, in order (list of dict)
    """
    return [
        record
        for record in events
        if (not kinds or record["kind"] in kinds)
        and (not models or record["model"] in models)
        and (not indices or record["index"] in indices)
    ]


def format_event(record: dict) -> str:
    """
    one event as a line of text
    :param record: an event (dict)
    :returns: time, event, status, run, job id, actor and the detail or reason (str)
    """
    run = "-"
    if record["kind"] is not None:
        run = f"{record['kind']} {record['model']}" + (
            f" i{record['index']}" if record["index"] is not None else ""
        )
    parts = [
        record["utc"],
        f"{record['event']:<8}",
        f"status {record['status']}",
        run,
        f"job {record['job_id'] or '-'}",
        f"by {record['actor'] or '-'}",
    ]
    if record.get("job_id_source"):
        parts[-2] += f" ({record['job_id_source']})"
    note = record.get("reason") or record.get("detail")
    if note:
        parts.append(f"| {note}")
    if record.get("moved_aside"):
        parts.append(f"| earlier attempt moved to {record['moved_aside']}")
    return "  ".join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    """
    the command line the shell writers call: `append` one event
    :param argv: command-line arguments, None for sys.argv[1:] (Optional[list])
    :returns: 0 on an event written, 2 on a usage error, 3 when the log could not be written
        (int)
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    where = sub.add_parser("root", help="print the shared experiment root")
    where.add_argument("--checkout", help="the repo root the writer runs from")
    where.add_argument("--path", help="a candidate data/run_experiment directory")
    sub.add_parser(
        "require-actor",
        help="exit 2 unless OPS_ACTOR names someone; the shell writers call this before acting",
    )
    add = sub.add_parser("append", help="append one event")
    add.add_argument("--root", required=True, help="the experiment root")
    add.add_argument("--event", required=True, choices=EVENTS)
    add.add_argument("--status", required=True, type=int)
    add.add_argument("--kind")
    add.add_argument("--model")
    add.add_argument("--index", type=int)
    add.add_argument("--job-id")
    add.add_argument("--actor")
    for name in OPTIONAL_TEXT_KEYS:
        add.add_argument(f"--{name.replace('_', '-')}")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_OK
    if args.command == "require-actor":
        try:
            require_actor()
        except ValueError as exc:
            print(f"ops_events: refused: {exc}", file=sys.stderr)
            return EXIT_USAGE
        return EXIT_OK
    if args.command == "root":
        if bool(args.checkout) == bool(args.path):
            print(
                "ops_events: give exactly one of --checkout and --path", file=sys.stderr
            )
            return EXIT_USAGE
        resolved = (
            experiment_root(args.checkout)
            if args.checkout
            else canonical_root(args.path)
        )
        print(resolved)
        return EXIT_OK
    try:
        append_event(
            args.root,
            args.event,
            args.status,
            kind=args.kind,
            model=args.model,
            index=args.index,
            job_id=args.job_id,
            actor=args.actor,
            **{name: getattr(args, name) for name in OPTIONAL_TEXT_KEYS},
        )
    except ValueError as exc:
        print(f"ops_events: refused: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(
            f"ops_events: could not write {ops_events_path(args.root)}: {exc}",
            file=sys.stderr,
        )
        return EXIT_WRITE_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
