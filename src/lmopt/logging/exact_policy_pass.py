"""
The per-arm record of whether the exact-policy pass ran, and its one reader.

`scripts/run_experiment.py` writes `exact_policy_pass.json` beside every arm's shards, including an
arm whose config left the pass off, so that "the pass did not run" is a recorded fact and not an
inference from a missing csv or a missing log line. The stamp and the first-job check both read it
here, so the three states an arm can be in have one definition: no record (an arm collected before
the record existed, which says nothing about the pass), a record that cannot be read as the three
known keys (an error, never read as no record), and a valid record.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

FILENAME = "exact_policy_pass.json"
_KEYS = frozenset({"enabled", "rc", "csv_rows"})


class PassRecordError(ValueError):
    """The record is present but is not a readable `{enabled, rc, csv_rows}` object"""


@dataclass(frozen=True)
class PassRecord:
    """
    what an arm recorded about its exact-policy pass
    :param enabled: whether the arm's config enabled the pass (bool)
    :param rc: the pass's exit status, None when it did not run (int or None)
    :param csv_rows: data rows in `exact_policy.csv`, None when the file was absent (int or None)
    """

    enabled: bool
    rc: Optional[int]
    csv_rows: Optional[int]


def _int_or_none(value: object, key: str) -> Optional[int]:
    """
    :param value: a decoded JSON value (object)
    :param key: the key it came from, for the message (str)
    :returns: the value when it is an int that is not a bool, None when it is null (int or None)
    :raises PassRecordError: for anything else, a bool and a string included
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PassRecordError(f"{key} must be an integer or null, got {value!r}")
    return value


def read_record(run_dir: Path | str) -> Optional[PassRecord]:
    """
    the arm's pass record
    :param run_dir: the arm's run directory (Path or str)
    :returns: the record, or None when no `exact_policy_pass.json` exists, which means the arm
        predates the record and is not evidence that the pass failed or never ran (PassRecord or None)
    :raises PassRecordError: when the file exists but cannot be read, is not a JSON object, holds
        other keys than enabled, rc and csv_rows, or a value of the wrong type
    """
    path = Path(run_dir) / FILENAME
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PassRecordError(f"{FILENAME} cannot be read ({exc})") from exc
    if not isinstance(raw, dict):
        raise PassRecordError(f"{FILENAME} is not a JSON object")
    if set(raw) != _KEYS:
        raise PassRecordError(
            f"{FILENAME} must hold exactly {sorted(_KEYS)}, holds {sorted(raw)}"
        )
    if not isinstance(raw["enabled"], bool):
        raise PassRecordError(f"enabled must be true or false, got {raw['enabled']!r}")
    return PassRecord(
        enabled=raw["enabled"],
        rc=_int_or_none(raw["rc"], "rc"),
        csv_rows=_int_or_none(raw["csv_rows"], "csv_rows"),
    )


def count_csv_rows(path: Path | str) -> Optional[int]:
    """
    the data rows of an `exact_policy.csv`, counted with a CSV parser (a quoted field may hold a
    newline, so a physical line count is wrong)
    :param path: the csv file (Path or str)
    :returns: the number of rows after the header, or None when the file does not exist (int or None)
    """
    path = Path(path)
    if not path.is_file():
        return None
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)
