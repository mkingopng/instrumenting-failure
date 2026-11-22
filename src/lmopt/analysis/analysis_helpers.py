"""Small helpers shared by the analysis modules.

Only functions that were byte-identical in more than one module live here; a helper
that shares a name but differs in signature or rule stays with its module.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, TypeGuard

import polars as pl


def clean_number(value: Any) -> float | None:
    """
    NaN and None both become None: a row never reports a non-number
    :param value: a number, a numeric string, NaN or None (any)
    :returns: the value as a float, or None for None and NaN (float or None)
    """
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def is_finite_number(x: Any) -> TypeGuard[float]:
    """
    True for a finite int or float, never a bool, so a row never reads a NaN or a flag as a number
    :param x: any value
    :returns: True only for a finite float or int, never a bool (bool)
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def count_hands(frame: pl.DataFrame, keys: Iterable[str]) -> int:
    """
    the number of distinct hands in a frame
    :param frame: decision rows (pl.DataFrame)
    :param keys: the columns that identify a hand; those absent from the frame are ignored (iterable of str)
    :returns: the count of distinct key tuples, 0 when none of the keys is a column (int)
    """
    present = [c for c in keys if c in frame.columns]
    return frame.select(present).unique().height if present else 0
