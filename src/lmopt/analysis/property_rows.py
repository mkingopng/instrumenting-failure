"""
The one accessor for "the LM's rows" of a properties table.

`properties.csv` holds a row for the LM and, since the control opponent's own properties are
scored, a row for that opponent under the same (game, property, opponent) key. A reader that
keys on those three columns without selecting the player reads the opponent's value as the
LM's, or lets an opponent row overwrite an LM row. Every reader that wants the LM's rows takes
them through this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import polars as pl

LM_PLAYER = "LM"

_Row = TypeVar("_Row", bound=Mapping)


def lm_rows(rows: Iterable[_Row]) -> list[_Row]:
    """
    the LM's rows of a properties table, the opponent's own rows dropped
    :param rows: properties rows, each with a `player` key (iterable of mappings)
    :returns: the rows whose `player` is the LM, in their original order (list)
    :raises KeyError: when a row has no `player` key, because a table with no player column
        cannot say whose rows they are and must not read as all the LM's
    """
    return [r for r in rows if r["player"] == LM_PLAYER]


def lm_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """
    the LM's rows of a properties frame, the opponent's own rows dropped
    :param frame: a properties table with a `player` column (pl.DataFrame)
    :returns: the rows whose `player` is the LM, in their original order (pl.DataFrame)
    :raises pl.exceptions.ColumnNotFoundError: when the frame has no `player` column
    """
    import polars as pl  # function-local: `lm_rows` serves modules that never load polars

    return frame.filter(pl.col("player") == LM_PLAYER)
