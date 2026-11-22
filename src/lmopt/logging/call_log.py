"""Per-call log: one JSON line per API call, beside the shard's Parquet.

The log is JSON lines, never Parquet: dozens of readers glob ``*.parquet`` in a run
directory and treat every match as a match log, so a second Parquet there would be scored
as decisions.

It is an append-only record, not an atomic artefact: it carries its final name from the
first call, each line is flushed as the call returns, and nothing is renamed or truncated.
Whether it is complete is answered inside the file. The first line of an attempt is a
``shard_started`` event, each call is a ``call`` line, and a successful flush ends the
attempt with a ``shard_flushed`` event naming how many calls it wrote. A resubmitted shard
appends a new attempt under its own ``run_id``; a killed attempt's lines stay, told apart
by ``run_id``. Each line is flushed but not fsynced: it survives a process kill (the
wall-kill case) and not a node crash.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

CALL_LOG_EXTENSION = ".calls.jsonl"

EVENT_STARTED = "shard_started"
EVENT_CALL = "call"
EVENT_FLUSHED = "shard_flushed"

ABSENT = "absent"
COMPLETE = "complete"
INCOMPLETE = "incomplete"


def call_log_path(parquet_path: str) -> str:
    """
    the call log's path for a shard
    :param parquet_path: the shard's Parquet path (str)
    :returns: the shard path with its extension replaced by `.calls.jsonl` (str)
    """
    stem, _ = os.path.splitext(parquet_path)
    return stem + CALL_LOG_EXTENSION


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CallLog:
    """
    appends lines for one shard attempt. Nothing is created until the first call is logged,
    so a shard with no LM agent leaves no file. Opened for append and never truncated.
    """

    def __init__(self, path: str) -> None:
        """
        :param path: the log's path (str)
        """
        self.path = path
        self.run_id: Optional[str] = None
        self._handle: Optional[Any] = None
        self._n_calls = 0
        self._finalized = False

    def attach(self, run_id: Optional[str]) -> None:
        """
        name the run every following line belongs to
        :param run_id: the shard's run id (Optional[str])
        """
        self.run_id = run_id

    def _write(self, record: dict) -> None:
        # ASCII-escaped, so a completion holding a lone surrogate (which UTF-8 cannot encode)
        # is written rather than killing the shard mid-run.
        line = json.dumps(
            {"run_id": self.run_id, **record}, ensure_ascii=True, allow_nan=False
        )
        assert self._handle is not None
        self._handle.write(line + "\n")
        self._handle.flush()

    def append(self, record: dict) -> None:
        """
        write one call line and flush it, so a killed process keeps every completed call.
        The first call of an attempt is preceded by its `shard_started` line
        :param record: the call's fields, JSON-serialisable (dict)
        :raises ValueError: when no run id was attached, or the attempt was already flushed
        """
        if self.run_id is None:
            raise ValueError(
                "call log has no run_id attached; a line could not be joined to a row"
            )
        if self._finalized:
            raise ValueError(f"call log {self.path} already ended this attempt")
        if self._handle is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8")
            self._write({"event": EVENT_STARTED, "started_utc": _now()})
        self._write({"event": EVENT_CALL, **record})
        self._n_calls += 1

    def finalize(self) -> None:
        """
        end the attempt with its `shard_flushed` line; a no-op when nothing was logged, and
        safe to call twice. Called after the shard's Parquet is written, so the footer's
        presence says the shard reached its flush
        """
        if self._handle is None or self._finalized:
            return
        self._write(
            {"event": EVENT_FLUSHED, "n_calls": self._n_calls, "flushed_utc": _now()}
        )
        self._handle.close()
        self._handle = None
        self._finalized = True


@dataclass
class CallLogReading:
    """
    what a shard's call log holds for one run id
    :param calls: this run's call lines, in order (list of dict)
    :param earlier_attempts: line counts of other run ids in the same file (dict)
    :param flushed_calls: the `n_calls` of this run's `shard_flushed` line, None without one
    :param torn_line: the 1-based number of a final line that is not valid JSON, else None
    """

    calls: list[dict] = field(default_factory=list)
    earlier_attempts: dict[str, int] = field(default_factory=dict)
    flushed_calls: Optional[int] = None
    torn_line: Optional[int] = None

    @property
    def complete(self) -> bool:
        """
        :returns: True when this run flushed and its footer's count equals its call lines (bool)
        """
        return self.flushed_calls is not None and self.flushed_calls == len(self.calls)


def read_call_log(path: str, run_id: str) -> CallLogReading:
    """
    read the lines of one run out of a call log. Lines of other run ids are counted as
    earlier attempts and not returned, so a killed attempt cannot double-join. A torn final
    line (a write cut short by a kill) is reported and the rest stays readable
    :param path: the log (str)
    :param run_id: the shard attempt to read, the `run_id` its Parquet rows carry (str)
    :returns: the reading (CallLogReading)
    :raises ValueError: naming the line, when a line other than the last is not valid JSON
    """
    with open(path, encoding="utf-8") as handle:
        raw = handle.read().split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    reading = CallLogReading()
    for number, text in enumerate(raw, start=1):
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            if number == len(raw):
                reading.torn_line = number
                continue
            raise ValueError(f"{path}: line {number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"{path}: line {number} is valid JSON but not an object: {text[:60]!r}"
            )
        if record.get("run_id") != run_id:
            other = str(record.get("run_id"))
            reading.earlier_attempts[other] = reading.earlier_attempts.get(other, 0) + 1
        elif record.get("event") == EVENT_CALL:
            reading.calls.append(record)
        elif record.get("event") == EVENT_FLUSHED:
            reading.flushed_calls = record.get("n_calls")
        elif record.get("event") != EVENT_STARTED:
            raise ValueError(
                f"{path}: line {number} carries this shard's run_id but the unknown event "
                f"{record.get('event')!r}; it would silently drop out of the completeness count"
            )
    return reading


def call_log_status(parquet_path: str, run_id: str) -> tuple[str, str]:
    """
    whether a shard's call log is complete. A shard collected before the log existed has
    none, and that is `absent`, not incomplete
    :param parquet_path: the shard's Parquet (str)
    :param run_id: the `run_id` its rows carry (str)
    :returns: (ABSENT, COMPLETE or INCOMPLETE, a reason naming what is wrong, empty otherwise)
    """
    path = call_log_path(parquet_path)
    if not os.path.exists(path):
        return ABSENT, ""
    reading = read_call_log(path, run_id)
    if reading.complete and reading.torn_line is None:
        return COMPLETE, ""
    if reading.flushed_calls is None:
        why = f"run {run_id} has no shard_flushed line"
    elif reading.flushed_calls != len(reading.calls):
        why = (
            f"run {run_id} flushed {reading.flushed_calls} calls but {len(reading.calls)} "
            "call lines carry its run_id"
        )
    else:
        why = f"line {reading.torn_line} is torn"
    return INCOMPLETE, why


def orphan_call_logs(directory: str) -> list[str]:
    """
    the call logs in a run directory that have no shard beside them. A log is named after
    its shard, so a log without one means the shard was killed before it flushed; the
    shard's cell is not collected, and the log is the only trace of what it did
    :param directory: a run directory (str)
    :returns: the orphan log file names, sorted (list of str)
    """
    orphans = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(CALL_LOG_EXTENSION):
            parquet = name[: -len(CALL_LOG_EXTENSION)] + ".parquet"
            if not os.path.exists(os.path.join(directory, parquet)):
                orphans.append(name)
    return orphans
