"""
Config-driven arm discovery for the contrast suites.

One loader feeds all three contrast suites (instruct vs think, memory rungs,
seat contrasts). The roster lives in `configs/contrast_arms.yaml`, never in
code: which arms exist is data, and a shipped roster that names an arm nobody
collected must fail loudly rather than quietly render an empty panel.

Two behaviours are deliberately re-implemented here rather than imported,
because `src/` never imports from `scripts/`:

- `_load_arm_parquet`:
  glob `*.parquet` in one named arm directory (never a
  multi-arm glob, which spans preserved archive dirs), concatenate with
  `how="diagonal_relaxed"` so a mixed shard set unions columns with nulls
  instead of raising ShapeError.
- `_ensure_additive_columns` mirrors the same script's `_ensure_additive_columns`:
  the additive columns are backfilled as all-null Utf8 on
  shards that predate them, with a printed note per column.

No silent fallbacks: a missing arm directory, an empty shard set,
an uncited exclusion, or an arm spanning two prompt schema versions is returned
as a problem string naming the arm, and that arm is not in the returned dict.
Callers surface the problem list; they never see a placeholder frame.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

from lmopt.analysis.analysis_helpers import clean_number
from lmopt.analysis.condition import DIAGNOSTIC_KEY, is_diagnostic
from lmopt.analysis.condition import parse_seat_mode_field
from lmopt.prompting.memory import MEMORY_RUNGS

# Roles an arm may take in the contrast suites.
ROLES = ("think", "instruct", "memory")

# Large columns no contrast suite reads, left on disk when an arm is loaded. A suite that came to
# need one gets polars' ColumnNotFoundError at the access, not a silent null.
UNREAD_COLUMNS = ("belief_prompt", "belief_raw_completion")

ADDITIVE_COLUMNS = ("reasoning_trace", "finish_reason", "belief_finish_reason")

# How many arms' frames `StreamedContrastArm` keeps resident after use.
FRAME_CACHE_SLOTS = 2

_TOP_LEVEL_KEYS = frozenset({"arms", "memory_arms", "notes"})
_REQUIRED_TOP_LEVEL_KEYS = ("arms", "memory_arms")
_ARM_KEYS = frozenset({"name", "path", "role", "epoch", "paired_with", "exclusions"})
_MEMORY_ARM_KEYS = _ARM_KEYS | {"rung"}
_REQUIRED_ARM_KEYS = ("name", "path", "role", "epoch")
_EXCLUSION_KEYS = frozenset({"game", "reason", "citation"})


@dataclass(frozen=True)
class Exclusion:
    """
    One game excluded from one arm, with the record that justifies it.
    `citation` is mandatory (enforced in `discover_contrast_arms`): an
    uncited exclusion silently deletes data from a contrast.
    """

    game: str
    reason: str
    citation: str


@dataclass
class ContrastArm:
    """
    One loaded arm: config metadata plus the concatenated shard frame.
    Not frozen because `df` is a polars DataFrame (unhashable); treat every
    field as read-only.
    """

    name: str
    path: Path
    role: str
    epoch: str
    df: pl.DataFrame
    seat_mode: str | None
    paired_with: str | None
    exclusions: tuple[Exclusion, ...]
    prompt_spec_hashes: tuple[str, ...]
    schema_version: int | None
    rung: str | None = None


class StreamedContrastArm(ContrastArm):
    """
    An arm discovered from disk whose frame is read when a suite asks for it, not kept.

    Holding every arm's frame at once makes the full roster peak at about 57 GiB: one composite is
    2.4 GB in memory and the roster has twelve. A suite touches one or two arms at a time, so `df`
    reads the arm's shards on access and keeps the last `FRAME_CACHE_SLOTS` frames. A caller that
    holds a frame in a local variable keeps it alive past the cache.
    :param fields: every `ContrastArm` field except `df`, which is never stored; assigning `df`
        raises AttributeError
    """

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    @property
    def df(self) -> pl.DataFrame:
        return _frame_of(self.path)

    @df.setter
    def df(self, value: pl.DataFrame) -> None:
        raise AttributeError(
            "a streamed arm reads its frame from disk; `df` cannot be assigned"
        )

    def __repr__(self) -> str:
        return (
            f"StreamedContrastArm(name={self.name!r}, path={self.path!r}, "
            f"role={self.role!r}, epoch={self.epoch!r}, rung={self.rung!r})"
        )


_frame_cache: "OrderedDict[tuple, pl.DataFrame]" = OrderedDict()


def _frame_of(arm_dir: Path) -> pl.DataFrame:
    """
    the arm's frame, read from its shards unless one of the last `FRAME_CACHE_SLOTS` reads is
    still resident. The key carries every shard's name, size and modification time, so a shard
    rewritten under the same directory is read again and never served stale
    :param arm_dir: the arm's directory (Path)
    :returns: the arm's concatenated frame with the additive columns present (pl.DataFrame)
    """
    files = sorted(arm_dir.glob("*.parquet"))
    key = (
        str(arm_dir),
        tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files),
    )
    cached = _frame_cache.get(key)
    if cached is not None:
        _frame_cache.move_to_end(key)
        return cached
    df = _load_arm_parquet(arm_dir)
    if df is None:
        raise ValueError(f"no parquet shards with rows under {arm_dir} on re-read")
    df = _ensure_additive_columns(df, quiet=True)
    _frame_cache[key] = df
    while len(_frame_cache) > FRAME_CACHE_SLOTS:
        _frame_cache.popitem(last=False)
    return df


def load_contrast_config(path: str | Path) -> dict:
    """
    Parse and validate the contrast-arm roster YAML.

    Structural defects are `ValueError`s naming the offending key or arm: the
    config is authored by hand, and a typo that silently drops an arm from a
    contrast is worse than a crash. Data defects (a path that does not exist,
    an uncited exclusion) are not checked here; they belong to
    `discover_contrast_arms`, which reports them as problem strings.
    :param path: Path to the YAML roster.
    :returns: The parsed mapping, with `arms`, `memory_arms` and `notes` keys.
    """
    import yaml

    path = Path(path)
    parsed = yaml.safe_load(path.read_text())
    if not isinstance(parsed, Mapping):
        raise ValueError(
            f"{path}: contrast config must be a mapping, got {type(parsed).__name__}"
        )

    for key in parsed:
        if key not in _TOP_LEVEL_KEYS:
            raise ValueError(
                f"{path}: unknown top-level key {key!r}; "
                f"expected one of {sorted(_TOP_LEVEL_KEYS)}"
            )
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in parsed:
            raise ValueError(f"{path}: required top-level key {key!r} is missing")
        if not isinstance(parsed[key], list):
            raise ValueError(
                f"{path}: top-level key {key!r} must be a list, "
                f"got {type(parsed[key]).__name__}"
            )

    config = {
        "arms": parsed["arms"],
        "memory_arms": parsed["memory_arms"],
        "notes": parsed.get("notes") or [],
    }
    if not isinstance(config["notes"], list):
        raise ValueError(f"{path}: top-level key 'notes' must be a list")

    seen: set[str] = set()
    for entry in config["arms"]:
        _validate_arm_entry(entry, path, allowed=_ARM_KEYS, is_memory=False, seen=seen)
    for entry in config["memory_arms"]:
        _validate_arm_entry(
            entry, path, allowed=_MEMORY_ARM_KEYS, is_memory=True, seen=seen
        )
    return config


def _validate_arm_entry(
    entry: Any,
    path: Path,
    *,
    allowed: frozenset[str] | set[str],
    is_memory: bool,
    seen: set[str],
) -> None:
    """
    Validate one roster entry; raise ValueError naming the defect.
    :param entry: one item from the YAML `arms` or `memory_arms` list (any type, checked here)
    :param path: the YAML file, named in every error message (Path)
    :param allowed: the keys an entry of this kind may carry (set of str)
    :param is_memory: whether the entry sits under `memory_arms`, which requires role `memory`
        and a `rung` in `MEMORY_RUNGS` (bool)
    :param seen: arm names already validated; this entry's name is added, and a repeat raises
        (set of str, mutated)
    :returns: None
    """
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"{path}: each arm entry must be a mapping, got {type(entry).__name__}"
        )

    name = entry.get("name", "<unnamed>")

    for key in entry:
        if key not in allowed:
            raise ValueError(
                f"{path}: arm {name!r} has unknown key {key!r}; "
                f"expected one of {sorted(allowed)}"
            )

    for key in _REQUIRED_ARM_KEYS:
        if not entry.get(key):
            raise ValueError(f"{path}: arm {name!r} is missing required key {key!r}")

    if entry["role"] not in ROLES:
        raise ValueError(
            f"{path}: arm {name!r} has unknown role {entry['role']!r}; "
            f"expected one of {list(ROLES)}"
        )

    if name in seen:
        raise ValueError(f"{path}: duplicate arm name {name!r}")
    seen.add(name)

    if is_memory:
        if entry["role"] != "memory":
            raise ValueError(
                f"{path}: memory_arms entry {name!r} has role {entry['role']!r}; "
                "memory_arms entries must have role 'memory'"
            )
        rung = entry.get("rung")
        if rung not in MEMORY_RUNGS:
            raise ValueError(
                f"{path}: memory arm {name!r} has rung {rung!r}; "
                f"expected one of {list(MEMORY_RUNGS)}"
            )
    elif entry["role"] == "memory":
        raise ValueError(
            f"{path}: arm {name!r} has role 'memory' but is listed under 'arms'; "
            "memory arms belong under 'memory_arms' (they need a rung)"
        )

    for exc in entry.get("exclusions") or []:
        if not isinstance(exc, Mapping):
            raise ValueError(
                f"{path}: arm {name!r} has a non-mapping exclusion entry "
                f"({type(exc).__name__})"
            )
        for key in exc:
            if key not in _EXCLUSION_KEYS:
                raise ValueError(
                    f"{path}: arm {name!r} exclusion has unknown key {key!r}; "
                    f"expected one of {sorted(_EXCLUSION_KEYS)}"
                )
        if not exc.get("game"):
            raise ValueError(f"{path}: arm {name!r} has an exclusion without a game")


def _load_arm_parquet(arm_dir: Path) -> pl.DataFrame | None:
    """Load and concatenate every shard parquet in one arm directory.

    Returns None when the directory, the files, or the rows
    are absent.
    """
    parquet_files = sorted(arm_dir.glob("*.parquet"))
    if not parquet_files:
        return None
    # diagonal_relaxed, not vertical_relaxed: a mixed shard set (older shards
    # without the additive columns beside newer shards with them) must union
    # columns with nulls, not raise ShapeError.
    combined = pl.concat(
        [
            pl.read_parquet(
                pf,
                columns=[
                    c for c in pl.read_parquet_schema(pf) if c not in UNREAD_COLUMNS
                ],
            )
            for pf in parquet_files
        ],
        how="diagonal_relaxed",
    )
    return combined if len(combined) else None


def _ensure_additive_columns(df: pl.DataFrame, quiet: bool = False) -> pl.DataFrame:
    """Backfill the additive columns as all-null Utf8 when absent.

    Prints a note per column: an
    arm collected before these columns existed has no traces, and the suites must read that as absence rather
    than crash on a missing column. `quiet` is for a re-read of an arm discovery already
    announced.
    """
    missing = [c for c in ADDITIVE_COLUMNS if c not in df.columns]
    for col in missing:
        if not quiet:
            print(
                f"NOTE: column {col!r} absent from these shards (an arm older than the column); "
                f"treating as all-null."
            )
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
    return df


def _refuse_diagnostic(arm_dir: Path) -> None:
    """
    refuse an arm whose run_meta.json marks it diagnostic, or cannot be read to tell
    The marker is keyed on presence: `resolved_config.diagnostic` in a collection, or the
    `condition` block of a scored arm directory. An unparseable run_meta.json refuses too, naming
    the file; an arm with no run_meta.json loads.
    :param arm_dir: the arm's collection directory (Path)
    :returns: None
    :raises ValueError: naming the file and the reason the arm is refused
    """
    meta_path = arm_dir / "run_meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{meta_path} does not parse ({exc}), so whether the arm is marked diagnostic "
            "cannot be read; a contrast arm must state it is not"
        ) from exc
    resolved = meta.get("resolved_config") if isinstance(meta, Mapping) else None
    if (isinstance(resolved, Mapping) and DIAGNOSTIC_KEY in resolved) or is_diagnostic(
        meta
    ):
        raise ValueError(
            f"{meta_path} marks the arm diagnostic, so it is never a contrast arm"
        )


def _read_seat_mode(arm_dir: Path) -> str | None:
    """Read `resolved_config.seat_mode` from the arm's run_meta.json.

    The resolved config is the one the run used; the top-level key records the requested value.
    `parse_seat_mode_field` refuses any shape other than a str, None or per-game dict of str; the
    seat contrasts' own refusal of a per-game dict is applied below, to the checked value.
    """
    meta_path = arm_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    resolved = meta.get("resolved_config")
    if not isinstance(resolved, Mapping):
        return None
    seat_mode = parse_seat_mode_field(resolved.get("seat_mode"), context=str(arm_dir))
    if isinstance(seat_mode, Mapping):
        # A game-disjoint composite records one seat mode per game; `str()` of it would read as a
        # rotating mode and silently misclassify a per-game-fixed arm.
        raise ValueError(
            f"run_meta records a per-game seat_mode {dict(seat_mode)!r}, but the "
            "seat contrasts read one seat mode per arm"
        )
    return seat_mode


def build_row(
    fields: tuple[str, ...],
    values: dict[str, Any],
    *,
    derived: frozenset[str] = frozenset(),
    required: tuple[str, ...] = (),
    cleaned: frozenset[str] = frozenset(),
) -> dict:
    """
    one output row with exactly `fields`, in order, shared by the contrast suites' row builders
    Fields not passed are None; a suite fills its `derived` fields (computed, never passed in)
    after the call.
    :param fields: the row's keys in output order (tuple of str)
    :param values: the caller's keyword values (dict)
    :param derived: keys in `fields` that the suite computes, which a caller may not pass (frozenset)
    :param required: keys every caller must pass (tuple of str)
    :param cleaned: numeric keys whose NaN is normalised to None (frozenset)
    :returns: the row dict (dict)
    :raises TypeError: on a key the row does not have, a derived key passed in, or a missing
        required key
    """
    unknown = values.keys() - (set(fields) - derived)
    missing = [k for k in required if k not in values]
    if unknown or missing:
        raise TypeError(f"row fields: unexpected {sorted(unknown)}, missing {missing}")
    row = {f: values.get(f) for f in fields}
    for field in cleaned & row.keys():
        row[field] = clean_number(row[field])
    return row


def rows_with(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    the rows whose `column` is non-null
    :param df: any frame carrying `column` (pl.DataFrame)
    :param column: the column that must be non-null (str)
    :returns: the filtered frame (pl.DataFrame)
    """
    return df.filter(pl.col(column).is_not_null())


def lm_decision_rows(df: pl.DataFrame) -> pl.DataFrame:
    """LM decision rows: `prompt` non-null (schema.py's documented rule)."""
    return rows_with(df, "prompt")


def discover_contrast_arms(
    config: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, ContrastArm], list[str]]:
    """
    Load every arm the config names, reporting refusals rather than hiding them.
    :param config: A mapping as returned by `load_contrast_config`.
    :param repo_root: Root the entries' `path` values are relative to.
    :returns: `(arms, problems)`. `arms` maps arm name to `ContrastArm` for every arm
        that loaded cleanly. `problems` is a list of human-readable strings,
        each naming the arm it refers to; a named arm is never also in `arms`.
    """
    arms: dict[str, ContrastArm] = {}
    problems: list[str] = []
    entries: Iterable[Mapping[str, Any]] = [
        *config.get("arms", []),
        *config.get("memory_arms", []),
    ]

    for entry in entries:
        name = entry["name"]
        arm_dir = Path(repo_root) / entry["path"]

        if not arm_dir.exists():
            problems.append(f"arm {name!r}: configured path does not exist: {arm_dir}")
            continue

        try:
            _refuse_diagnostic(arm_dir)
        except ValueError as exc:
            problems.append(f"arm {name!r}: {exc}")
            continue

        exclusions, exclusion_problems = _parse_exclusions(name, entry)
        if exclusion_problems:
            problems.extend(exclusion_problems)
            continue

        df = _load_arm_parquet(arm_dir)
        if df is None:
            problems.append(
                f"arm {name!r}: no parquet shards with rows under {arm_dir}"
            )
            continue
        if "prompt" not in df.columns:
            problems.append(
                f"arm {name!r}: shards under {arm_dir} have no 'prompt' column; "
                "this is not a decision-node log"
            )
            continue
        df = _ensure_additive_columns(df)

        lm = lm_decision_rows(df)
        hashes = tuple(str(v) for v in _unique_values(lm, "prompt_spec_hash"))

        versions = _unique_values(lm, "prompt_schema_version")
        if len(versions) > 1:
            problems.append(
                f"arm {name!r}: shards under {arm_dir} span "
                f"{len(versions)} prompt_schema_version values ({versions}); "
                "one arm must be one prompt epoch"
            )
            continue

        try:
            seat_mode = _read_seat_mode(arm_dir)
        except ValueError as exc:
            problems.append(f"arm {name!r}: {exc}")
            continue

        # The frame is read to check the arm and then let go; a suite reads it again when it asks
        # for it, so the roster never sits in memory at once.
        del df, lm
        arms[name] = StreamedContrastArm(
            name=name,
            path=arm_dir,
            role=entry["role"],
            epoch=entry["epoch"],
            seat_mode=seat_mode,
            paired_with=entry.get("paired_with"),
            exclusions=exclusions,
            prompt_spec_hashes=hashes,
            schema_version=int(versions[0]) if versions else None,
            rung=entry.get("rung"),
        )

    return arms, problems


def _parse_exclusions(
    name: str, entry: Mapping[str, Any]
) -> tuple[tuple[Exclusion, ...], list[str]]:
    """Parse an entry's exclusions; an uncited one is a problem, not a warning."""
    parsed: list[Exclusion] = []
    problems: list[str] = []
    for exc in entry.get("exclusions") or []:
        citation = exc.get("citation")
        if not citation:
            problems.append(
                f"arm {name!r}: exclusion of game {exc.get('game')!r} has no "
                "citation; an uncited exclusion silently deletes data"
            )
            continue
        parsed.append(
            Exclusion(game=exc["game"], reason=exc.get("reason", ""), citation=citation)
        )
    return tuple(parsed), problems


def _unique_values(lm: pl.DataFrame, column: str) -> list:
    """Sorted unique non-null values of `column` on LM rows, or []."""
    if column not in lm.columns:
        return []
    return sorted(lm[column].drop_nulls().unique().to_list())


def comparable(a: ContrastArm, b: ContrastArm) -> bool:
    """True iff the two arms share one known prompt epoch.

    Both hash sets must be non-empty and equal. An arm with no
    `prompt_spec_hash` column (schema v2) has an unknown epoch, and unknown is
    never a match, not even against another unknown: cross-epoch contrasts are
    never pooled.
    """
    return bool(a.prompt_spec_hashes) and a.prompt_spec_hashes == b.prompt_spec_hashes


def excluded_games(arm: ContrastArm) -> set[str]:
    """The set of game names excluded from this arm by config."""
    return {e.game for e in arm.exclusions}


def filter_excluded(arm: ContrastArm) -> pl.DataFrame:
    """`arm.df` with the rows of every excluded game dropped."""
    excluded = excluded_games(arm)
    if not excluded:
        return arm.df
    return arm.df.filter(~pl.col("game").is_in(sorted(excluded)))
