"""
The condition an analysis output analysed, recorded in the analysis output itself.

An arm directory under `analysis/` holds one condition of one model: a memory level, a
hand count, a probe role. None of that was ever written down beside the scored tables, so
every consumer either guessed it or followed `data_dir` back into the collection directory.
Following the pointer is wrong for a consumer that reads analysis output only, and guessing
is worse, so the writer records the condition once and every reader reads that block.

Nothing here derives a value. Each field is copied from a field the collection run_meta
already carries, except the prompt schema version, which only the logged rows carry and is
copied from them. A field that collection never recorded stays None rather than being
filled in from a convention.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

#: The key the block lives under in an analysis arm's `run_meta.json`.
CONDITION_KEY = "condition"
#: The fields every block must carry. A block missing any of these is not a condition:
#: `condition_of` returns None for it, and every reader then treats the output as not
#: stating what it analysed.
REQUIRED_CONDITION_FIELDS = (
    "model_id",
    "memory_rung",
    "n_hands",
    "n_seeds",
    "n_pieces",
    "probe_type",
    "source",
)
#: The fields a block may lack and still be a condition, each named with the reason it may.
#: A field joins this tuple by a deliberate decision, never by being the newest addition:
#: what `condition_of` requires must not drift to whatever the oldest artefact contained.
#:   temperature          recorded from the prompt version 7 collections onward; blocks
#:                        written before the writer recorded it lack the key, and the
#:                        collection they describe was not re-read to fill it.
#:   prompt_schema_version  the same: read from the logged rows by the writer, so a block
#:                        written before the writer read them lacks the key.
#: An absent optional field reads as None, which is the true value (the block does not say),
#: never a default.
OPTIONAL_CONDITION_FIELDS = ("temperature", "prompt_schema_version")
#: Every field of the block, in the order it is written.
CONDITION_FIELDS = REQUIRED_CONDITION_FIELDS + OPTIONAL_CONDITION_FIELDS
#: The key a diagnostic arm's block carries. It is written only when the collection config set
#: it, so an absent key is a scoreboard arm by construction and the refusal is keyed on the
#: key's presence, never on a value read with a default.
DIAGNOSTIC_KEY = "diagnostic"
#: The `probe_type` of a run with no probe, as `lmopt.logging.schema` defines it.
BASE_PROBE_TYPE = "none"
#: What `source` says when the writer is the scoring entrypoint itself.
SOURCE_ANALYSE_STATS = "analyse_stats"
#: What a backfilled block's `source` starts with. Only such a block may be rewritten by a
#: later backfill: a block the scoring run itself wrote is the run's own statement about
#: what it analysed, and no later pass may restate it.
SOURCE_BACKFILL_PREFIX = "backfilled from"


class ConditionUnavailable(ValueError):
    """
    Raised when a collection run_meta cannot be read, so no block can be written.

    An exception rather than a partly-filled block: a block that exists is taken by every
    reader as the recorded truth, and one filled from a missing file would be a provenance
    lie no later reader could detect.
    """


def condition_block(
    collection_meta: Mapping,
    source: str,
    prompt_schema_versions: Sequence | None = None,
) -> dict:
    """
    the condition block, copied field by field out of a collection run_meta and the rows
    :param collection_meta: the parsed `run_meta.json` of the collection directory the
        analysis output was scored from
    :param source: what wrote the block, either SOURCE_ANALYSE_STATS or a sentence naming
        the file a backfill read
    :param prompt_schema_versions: the distinct `prompt_schema_version` values the arm's LM
        rows carry, or None when the caller did not read the rows. The collection run_meta
        does not hold this field; only the logged rows do.
    :returns: a dict with exactly CONDITION_FIELDS; a value the collection run never
        recorded is None, never a default, except `probe_type`, which falls back to
        BASE_PROBE_TYPE when the config has no probe block. `n_seeds` is `resolved_config.n_seeds` verbatim,
        the seed count of one piece; `n_pieces` is the length of the collection run_meta's
        `composed_from` (1 when that key is absent or is not a list, which is what an
        uncomposed run is). `n_pieces` is a piece count, not a replicate count, and no
        reader may multiply `n_seeds * n_pieces` to get one: a game-disjoint composite
        (one off-path run per game sharing a single seed) has
        `n_pieces > distinct seeds`, so that product overcounts. The correct replicate
        count is the number of distinct seed values actually present for the game/cell
        being read, exactly as `probe_scores.py::_replicates` already computes it (from
        the logged `seed` column of that game's own rows, never from a piece count) --
        any future consumer needing a replicate count should do the same, not read this
        field for it. `temperature` is `resolved_config.temperature` verbatim.
        `prompt_schema_version` is the one distinct value in `prompt_schema_versions`, None
        when the caller passed none or the rows carry none.
    :raises ConditionUnavailable: when the rows carry more than one distinct prompt schema
        version. The certification gate already refuses such an arm, so this only fires when
        that gate was bypassed, and a block naming one of several versions would be a
        provenance lie no later reader could detect. Also when the collection config records
        `diagnostic` as anything but the boolean true: the marker is a statement that the arm
        is never a scoreboard arm, and a value that says otherwise has no meaning to refuse on.
    """
    versions = sorted({v for v in (prompt_schema_versions or ()) if v is not None})
    if len(versions) > 1:
        raise ConditionUnavailable(
            f"cannot record the condition: the rows carry {len(versions)} prompt schema "
            f"versions {versions}, and a condition names exactly one"
        )
    pieces = collection_meta.get("composed_from")
    resolved = collection_meta.get("resolved_config") or {}
    probe = resolved.get("probe") or {}
    marker = {}
    if DIAGNOSTIC_KEY in resolved:
        if resolved[DIAGNOSTIC_KEY] is not True:
            raise ConditionUnavailable(
                f"cannot record the condition: the collection config records "
                f"{DIAGNOSTIC_KEY}={resolved[DIAGNOSTIC_KEY]!r}, and the only value the "
                "marker takes is true"
            )
        marker = {DIAGNOSTIC_KEY: True}
    return {
        "model_id": collection_meta.get("model_id") or resolved.get("model"),
        "memory_rung": resolved.get("memory_rung"),
        "n_hands": collection_meta.get("n_hands", resolved.get("n_hands")),
        "n_seeds": resolved.get("n_seeds"),
        "n_pieces": len(pieces) if isinstance(pieces, list) else 1,
        "probe_type": probe.get("type", BASE_PROBE_TYPE),
        "source": source,
        "temperature": resolved.get("temperature"),
        "prompt_schema_version": versions[0] if versions else None,
        **marker,
    }


def load_collection_meta(data_dir, repo_root=None) -> dict:
    """
    read the collection run_meta an analysis output was scored from
    :param data_dir: the `data_dir` an analysis run_meta names, absolute or repo-relative
    :param repo_root: the root a relative `data_dir` resolves against, or None for the cwd
    :returns: the parsed run_meta
    :raises ConditionUnavailable: when the directory or the file is missing or unreadable
    """
    path = Path(data_dir)
    if not path.is_absolute() and repo_root is not None:
        path = Path(repo_root) / path
    meta_path = path / "run_meta.json"
    try:
        loaded = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        raise ConditionUnavailable(
            f"cannot record the condition: {meta_path} is missing or unreadable ({exc})"
        ) from exc
    if not isinstance(loaded, dict):
        raise ConditionUnavailable(
            f"cannot record the condition: {meta_path} is not an object"
        )
    return loaded


def condition_of(analysis_meta: Mapping | None) -> dict | None:
    """
    the condition an analysis arm records, for a reader that opens analysis output only
    :param analysis_meta: the parsed `run_meta.json` of a scored arm directory, or None
    :returns: the block when the arm records one with every required field present, with each
        absent optional field reported as None; None otherwise. None means the output does
        not say what it analysed, which is an instrument defect and never an invitation to
        assume a memory level
    """
    if not isinstance(analysis_meta, Mapping):
        return None
    block = analysis_meta.get(CONDITION_KEY)
    if not isinstance(block, Mapping):
        return None
    if any(field not in block for field in REQUIRED_CONDITION_FIELDS):
        return None
    return {**{field: None for field in OPTIONAL_CONDITION_FIELDS}, **block}


def condition_epoch(block: Mapping | None) -> tuple:
    """
    the collection epoch a condition belongs to, for telling two runs of one model at one
    memory level apart
    :param block: a condition block as `condition_of` returns it, or None
    :returns: (temperature, prompt_schema_version); two blocks of one model and memory level
        with different epochs are different conditions, not two candidates for one
    """
    if not isinstance(block, Mapping):
        return (None, None)
    return (block.get("temperature"), block.get("prompt_schema_version"))


def is_backfilled(block: Mapping | None) -> bool:
    """
    :param block: a condition block, or None
    :returns: True only when the block was written by a backfill and may therefore be
        rewritten by a later one; a block the scoring run wrote is never rewritten
    """
    if not isinstance(block, Mapping):
        return False
    return str(block.get("source") or "").startswith(SOURCE_BACKFILL_PREFIX)


def is_diagnostic(analysis_meta: Mapping | None) -> bool:
    """
    whether a scored arm directory marks itself as a diagnostic arm
    :param analysis_meta: the parsed `run_meta.json` of a scored arm directory, or None
    :returns: True when the arm's condition block carries the diagnostic key, whatever its
        value; False when the file, the block or the key is absent, which is what every arm
        scored before the marker existed looks like. Keyed on the key's presence so that no
        value read with a default can stand in for either answer
    """
    if not isinstance(analysis_meta, Mapping):
        return False
    block = analysis_meta.get(CONDITION_KEY)
    return isinstance(block, Mapping) and DIAGNOSTIC_KEY in block


# --- Shape checks for resolved_config's seat_mode and drive_openings ------------------------
# The one place `resolved_config.seat_mode` and `resolved_config.drive_openings` have their
# shape checked, so a field whose type changes fails loudly here rather than silently at
# whichever of the six readers and writers happens to touch it next.
#
# Both fields have exactly two legitimate shapes: a scalar (`str` for seat_mode, `list[int]`
# for drive_openings) on a single-piece run, or a per-game `Mapping[str, <that same scalar
# type>]` on a game-disjoint off-path composite (`compose_arm_pieces.PER_GAME_FIELDS`). Which
# shape a given consumer may accept is a policy that differs by consumer -- the seat contrasts
# refuse a per-game dict, P6's scoring path preserves one -- so this module does not encode a
# policy. It refuses only a third shape, one that is neither of the two legitimate ones, and
# returns the value's content unchanged otherwise: a scalar stays a scalar, a dict's or list's
# content stays the same content (returned as a fresh copy, never a value rebuilt from a
# different shape), so no caller here mistakes a shape check for a coercion.


class ResolvedConfigShapeError(ValueError):
    """
    Raised when a resolved_config field is present with a shape that is neither of the two
    this module declares legitimate: a flat scalar, or a per-game record of that scalar.
    """


def _type_name(value) -> str:
    """
    :param value: any value
    :returns: its type's plain name, for a refusal message (str)
    """
    return type(value).__name__


def parse_seat_mode_field(value, *, context: str) -> str | dict[str, str] | None:
    """
    the shape-checked value of resolved_config["seat_mode"], never coerced from a wrong shape
    :param value: the raw value read from (or about to be written to) a run_meta.json
        resolved_config block, or None
    :param context: identifies the arm or run this value belongs to, for a refusal message
    :returns: value unchanged, once confirmed to be one of: None; a str; or a Mapping every
        one of whose values is a str (a per-game record on a game-disjoint composite)
    :raises ResolvedConfigShapeError: value is any other shape, for example a list, an int,
        or a Mapping holding a non-str value
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        bad = {g: v for g, v in value.items() if not isinstance(v, str)}
        if bad:
            raise ResolvedConfigShapeError(
                f"{context}: resolved_config['seat_mode'] is a per-game record but "
                f"{bad!r} is not a str; a per-game seat_mode must map each game to a str"
            )
        return dict(value)
    raise ResolvedConfigShapeError(
        f"{context}: resolved_config['seat_mode'] is a {_type_name(value)} ({value!r}); "
        f"expected a str, a per-game dict of str, or None"
    )


def require_seat_mode_field(
    resolved: Mapping, *, context: str
) -> str | dict[str, str] | None:
    """
    the seat mode a run_meta.json resolved_config records, refusing a block that records none
    An absent key is never read as a default: a run that actually played another seat mode would
    be scored, paired or identified as `fixed0` with nothing said, and a falsely identified arm
    can void a P6 anchor against a value nobody recorded. A key recorded as null is a recorded
    value and is returned as the shape check returns it
    :param resolved: the resolved_config block of a run_meta.json (Mapping)
    :param context: identifies the arm or run this block belongs to, for a refusal message (str)
    :returns: the recorded value, shape-checked (str, per-game dict of str, or None)
    :raises ResolvedConfigShapeError: the key is absent, or the recorded value has a wrong shape
    """
    if "seat_mode" not in resolved:
        raise ResolvedConfigShapeError(
            f"{context}: resolved_config records no seat_mode; an absent key is not read as "
            f"'fixed0', because that would assert a seat mode nobody recorded"
        )
    return parse_seat_mode_field(resolved["seat_mode"], context=context)


def parse_drive_openings_field(
    value, *, context: str
) -> list[int] | dict[str, list[int]] | None:
    """
    the shape-checked value of resolved_config["drive_openings"], never coerced from a wrong
    shape
    :param value: the raw value, or None
    :param context: identifies the arm or run this value belongs to, for a refusal message
    :returns: value unchanged, once confirmed to be one of: None; a list of int; or a Mapping
        every one of whose values is a list of int (a per-game record)
    :raises ResolvedConfigShapeError: value is any other shape, for example a bare string
        (which `list()` would silently shred character by character), a tuple, a dict of
        bare strings, or a dict whose values are not lists of int
    """
    if value is None:
        return value
    if isinstance(value, Mapping):
        bad = {g: v for g, v in value.items() if not _is_int_list(v)}
        if bad:
            raise ResolvedConfigShapeError(
                f"{context}: resolved_config['drive_openings'] is a per-game record but "
                f"{bad!r} is not a list of int; a per-game drive_openings must map each "
                f"game to a list of int"
            )
        return dict(value)
    if _is_int_list(value):
        return list(value)
    raise ResolvedConfigShapeError(
        f"{context}: resolved_config['drive_openings'] is a {_type_name(value)} ({value!r}); "
        f"expected a list of int, a per-game dict of list of int, or None"
    )


def _is_int_list(value) -> bool:
    """
    whether value is a legitimate flat drive_openings: a real list of plain int
    :param value: any value
    :returns: True only for a `list` (never a `str`, a `tuple`, or another sequence type) of
        exactly `int` (never `bool`, which is an `int` subclass but never a legal opening
        index in this codebase)
    """
    return isinstance(value, list) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    )
