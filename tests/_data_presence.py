"""
Presence checks for tests that read real collected data, which is gitignored and not on every
checkout. A test must skip when the data it needs is not there, and "the directory exists" is not
that condition: a directory left empty (or holding other files) passes `exists()` and the test then
fails on `found 0`, so one commit's failure set depended on the shape of each tree. These helpers
ask for the files the test reads.
"""

from __future__ import annotations

from pathlib import Path


def has_files(directory, pattern: str = "*.parquet", recursive: bool = False) -> bool:
    """
    whether a directory exists and holds at least one file matching a pattern
    :param directory: the directory a test reads real data from (str or Path)
    :param pattern: the file-name pattern the test reads (str)
    :param recursive: whether to look in subdirectories as well (bool)
    :returns: True only when a matching file is present; False for a missing path, a path that is
        not a directory and an empty directory alike (bool)
    """
    path = Path(directory)
    if not path.is_dir():
        return False
    matches = path.rglob(pattern) if recursive else path.glob(pattern)
    return any(p.is_file() for p in matches)
