"""Schema-fingerprint shard loader with divergence reporting.

Provides two entry points for loading and concatenating Parquet shards:

- `load_shards(files, ...)`: reads files from paths, fingerprints schemas,
  reports divergence to stderr, and returns a diagonal_relaxed concat.

- `load_frames(frames, ...)`: same behavior for already-read DataFrames.

Both fingerprint schemas as sorted (column, dtype) pairs and produce a loud
report to stderr when >1 unique fingerprint is detected, listing the groups,
file counts, and symmetric-difference columns. The default behavior (strict=False)
is to report but not raise, because column-additive schema evolution is the
sanctioned migration path. Pass
strict=True to opt into raising ValueError on divergence.
"""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from polars._typing import ConcatMethod


def load_shards(
    files: Sequence[str | Path],
    *,
    strict: bool = False,
) -> pl.DataFrame:
    """
    Read Parquet files, fingerprint schemas, report divergence, and concat.
    :param files: Paths to Parquet files to read.
    :param strict: If True, raise ValueError on schema divergence.
    :returns: Concatenated DataFrame using how="diagonal_relaxed".
    :raises ValueError: If strict=True and schemas diverge.
    """
    if not files:
        return pl.DataFrame()

    dfs = [pl.read_parquet(f) for f in files]
    names = [str(f) for f in files]
    return load_frames(dfs, strict=strict, names=names)


def load_frames(
    frames: Sequence[pl.DataFrame],
    *,
    strict: bool = False,
    names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """
    Fingerprint already-read DataFrames, report divergence, and concat.
    :param frames: DataFrames to fingerprint and concat.
    :param strict: If True, raise ValueError on schema divergence.
    :param names: Optional names for frames (used in error messages).
    :returns: Concatenated DataFrame using how="diagonal_relaxed".
    :raises ValueError: If strict=True and schemas diverge.
    """
    if not frames:
        return pl.DataFrame()

    if names is None:
        names = [f"frame_{i}" for i in range(len(frames))]
    elif len(names) != len(frames):
        raise ValueError(
            f"load_frames: {len(names)} names for {len(frames)} frames; a skipped frame must "
            f"drop its name too, or the divergence report names the wrong files"
        )

    fingerprints = [_fingerprint(df) for df in frames]
    unique_fps = list(dict.fromkeys(fingerprints))

    if len(unique_fps) > 1:
        _report_divergence(fingerprints, names, unique_fps)
        if strict:
            all_cols: dict[str, list[Any]] = {}
            for fp in unique_fps:
                for col, dtype in fp:
                    if col not in all_cols:
                        all_cols[col] = []
                    all_cols[col].append(dtype)

            diff_cols = [
                col
                for col, dtypes in all_cols.items()
                if len(set(map(str, dtypes))) > 1 or len(dtypes) < len(unique_fps)
            ]
            diff_cols = sorted(diff_cols)

            files_msg = f" Files: {names[0]}, {names[1]}" if len(names) >= 2 else ""
            raise ValueError(
                f"Schema divergence detected. Columns: {', '.join(diff_cols)}.{files_msg}"
            )

    return pl.concat(frames, how="diagonal_relaxed")


def concat_strict(
    frames: Sequence[pl.DataFrame],
    *,
    how: "ConcatMethod" = "vertical",
    what: str = "these shards",
) -> pl.DataFrame:
    """
    concatenate frames that must share one column set, and fail with the likely cause and the
    fix when they do not. The reader stays strict on purpose: a column present in only some
    shards would either drop out or read as null for the rest, and either way hide which rows
    carry it (which shards salvaged a belief, for one). Frames with the same columns go to
    `pl.concat` unchanged, so a dtype mismatch still fails as it did
    :param frames: the frames to stack, one per shard (Sequence[pl.DataFrame])
    :param how: the `pl.concat` strategy for frames that share their columns (str)
    :param what: a short name for the frames, used in the message (str)
    :returns: the stacked frame (pl.DataFrame)
    :raises ValueError: when the frames do not all carry the same column names
    """
    frames = list(frames)
    column_sets = [frozenset(frame.columns) for frame in frames]
    if len(set(column_sets)) > 1:
        shared = frozenset.intersection(*column_sets)
        uneven = sorted(frozenset.union(*column_sets) - shared)
        raise ValueError(
            f"{what}: the shards do not share one column set. Columns missing from at least "
            f"one shard: {uneven}; {len(shared)} columns are in every shard. The usual cause is "
            "an arm whose shards were reparsed in separate passes, so only some carry the "
            "salvage columns (belief_salvage_kind, belief_salvage_attempt), or shards "
            "collected under an older schema left beside newer ones. Fix it at the source: "
            "reparse every shard of the arm together, which gives "
            "each shard the same columns, or re-run the whole cell after a schema change."
        )
    return pl.concat(frames, how=how)


def _fingerprint(df: pl.DataFrame) -> tuple[tuple[str, Any], ...]:
    """Return sorted (column, dtype) pairs as schema fingerprint."""
    return tuple(sorted((name, dtype) for name, dtype in df.schema.items()))


def _report_divergence(
    fingerprints: list[tuple[tuple[str, Any], ...]],
    names: Sequence[str],
    unique_fps: list[tuple[tuple[str, Any], ...]],
) -> None:
    """Print loud schema divergence report to stderr."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("SCHEMA DIVERGENCE DETECTED", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    for i, fp in enumerate(unique_fps):
        count = fingerprints.count(fp)
        files_with_fp = [n for n, f in zip(names, fingerprints, strict=True) if f == fp]
        print(f"\nGroup {i + 1}: {count} file(s)", file=sys.stderr)
        print(f"  Schema: {dict(fp)}", file=sys.stderr)
        print(
            f"  Files: {files_with_fp[:3]}"
            + (
                f" ... and {len(files_with_fp) - 3} more"
                if len(files_with_fp) > 3
                else ""
            ),
            file=sys.stderr,
        )

    all_cols: dict[str, list[Any]] = {}
    for fp in unique_fps:
        for col, dtype in fp:
            if col not in all_cols:
                all_cols[col] = []
            all_cols[col].append(dtype)

    diff_cols = [
        col for col, dtypes in all_cols.items() if len(set(map(str, dtypes))) > 1
    ]
    if diff_cols:
        print(f"\nColumns with type differences: {diff_cols}", file=sys.stderr)

    only_in_some = [
        col for col, dtypes in all_cols.items() if len(dtypes) < len(unique_fps)
    ]
    if only_in_some:
        print(f"Columns in only some schemas: {only_in_some}", file=sys.stderr)

    print("=" * 70 + "\n", file=sys.stderr)
