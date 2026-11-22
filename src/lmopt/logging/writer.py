"""Parquet writer for per-node NodeRecord instances."""

from __future__ import annotations

import dataclasses
import os

import polars as pl

from lmopt.logging.call_log import CallLog, call_log_path
from lmopt.logging.schema import NodeRecord, PARQUET_SCHEMA


class MatchWriter:
    """Accumulates NodeRecord instances and writes a Parquet file on flush(), then publishes the
    shard's per-call log."""

    def __init__(self, output_path: str) -> None:
        # Refused here, before a shard plays its first hand, rather than at flush(), which
        # is the end of the shard: a record field the schema does not name would otherwise
        # cost the whole run.
        missing = sorted(
            f.name
            for f in dataclasses.fields(NodeRecord)
            if f.name not in PARQUET_SCHEMA
        )
        if missing:
            raise ValueError(
                f"NodeRecord field(s) {missing} are not in PARQUET_SCHEMA "
                "(lmopt.logging.schema); add each to the schema or remove it from the record"
            )
        self._output_path = output_path
        self._records: list[NodeRecord] = []
        # Owned here so the log is named after, and published with, the shard it belongs to.
        self.call_log = CallLog(call_log_path(output_path))

    def add(self, record: NodeRecord) -> None:
        self._records.append(record)

    def flush(self) -> pl.DataFrame:
        """Write accumulated records to Parquet and return the DataFrame."""
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)), exist_ok=True)
        rows = [dataclasses.asdict(r) for r in self._records]
        df = _build_dataframe(rows)
        # Written beside the final path and moved into place, so a reader or a pull never sees
        # a half-written shard. Dot-prefixed and ending ".tmp" is a form no "*.parquet" glob
        # matches.
        directory, name = os.path.split(os.path.abspath(self._output_path))
        tmp = os.path.join(directory, "." + name + ".tmp")
        try:
            df.write_parquet(tmp)
            os.replace(tmp, self._output_path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        self.call_log.finalize()
        return df

    def __len__(self) -> int:
        return len(self._records)


def _build_dataframe(rows: list[dict]) -> pl.DataFrame:
    """Build a polars DataFrame with the canonical schema.

    A row key the schema does not name is REFUSED, naming it: the final select below
    would otherwise drop it with no error, so a field added to NodeRecord without its
    schema entry would vanish from every shard.

    The declared PARQUET_SCHEMA governs construction, not just a post-hoc cast:
    polars' default infer_schema_length=100 locks a column dtype from the first
    100 rows, so >100 leading Nones in a nullable Utf8 column (e.g.
    elicited_belief under a base model that rarely parses) followed by a real
    string would raise ComputeError and kill the shard. An unknown row key is refused above rather than dropped; the cast
    loop still backfills any schema column absent from the rows.
    """
    if not rows:
        return pl.DataFrame(schema=PARQUET_SCHEMA)

    present = set(rows[0])
    unknown = sorted(present - set(PARQUET_SCHEMA))
    if unknown:
        raise ValueError(
            f"rows carry key(s) the Parquet schema does not name: {unknown}. Writing them "
            "would drop them silently; add each to PARQUET_SCHEMA (lmopt.logging.schema) "
            "or remove it from the record."
        )
    df = pl.DataFrame(
        rows, schema={c: t for c, t in PARQUET_SCHEMA.items() if c in present}
    )

    # Cast each column to the declared type, handling nulls in List columns.
    casts = []
    for col, dtype in PARQUET_SCHEMA.items():
        if col in df.columns:
            casts.append(pl.col(col).cast(dtype))
        else:
            casts.append(pl.lit(None).cast(dtype).alias(col))

    df = df.with_columns(casts)
    return df.select(list(PARQUET_SCHEMA.keys()))


def read_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)
