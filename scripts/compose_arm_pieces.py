#!/usr/bin/env python3
"""Compose N certified, stamped one-seed-per-job PIECE arms into one composite arm.

Why this exists. Collection runs one seed per job, and `run_sweep.py`'s
`_shard_out_path` appends a `_seed{seed_idx}` suffix to a shard's filename
only when `n_seeds > 1`. At `n_seeds=1` every index of one model writes
byte-identical filenames (e.g. `kuhn_lm_vs_cfr_ref.parquet` in every one of
ten index directories), so a naive union of ten directories overwrites nine
of every ten shards, and `scripts/analyse_stats.py::cell_of`'s regex requires
that suffix to parse a shard at all -- meaning a single un-composed piece is
unreadable by `load_cells`, not only multiple pieces together.
`scripts/recycle_arm_evidence.py` accepts multiple `arm_dir` arguments but
scores them as separate arms (worst exit code wins), never composing them.

This script composes a caller-specified list of PIECE directories -- each
already a complete, independently certified+stamped arm in its own right --
into one new composite directory that `recycle_arm_evidence.py` and, via that
stamp, every downstream script can consume as a single arm. Pieces are never
modified, moved, or deleted: every shard and sidecar is hard-linked into the
composite under a renamed basename, so the piece directories remain complete,
independently-inspectable provenance after composition.

Renaming rule: `<game>_lm_vs_<opp>.parquet` becomes
`<game>_lm_vs_<opp>_seed{TRUE_SEED}.parquet`, where TRUE_SEED is read from the
piece's own Parquet rows' `seed` column (never the filename, never `seed_idx`
-- `seed_idx` is not persisted anywhere on disk: it exists only in
`run_sweep.py`'s in-process `built` list for console printing, never in
`PARQUET_SCHEMA`, never in `shard_outcomes.json`), cross-checked against the
piece's own declared `run_meta.json["resolved_config"]["seed"]`. Every
sidecar beside a shard (`os.path.splitext(shard)[0] + <any suffix>` --
`posterior_fallback_sidecar_path`/`driven_discard_sidecar_path` in
`src/lmopt/orchestration/shard.py` both derive their path this same way) is renamed in lockstep, generically, not by a hardcoded list of known
suffixes.

Piece-level artefacts (a file in the piece directory that is not a shard, not a shard's
sidecar and not `run_meta.json`/`shard_outcomes.json`, which are composed): every one is
hard-linked into the composite as `<stem>.seed{S}<ext>` (`.seed{S}_{n}` for the n-th piece
sharing seed S), whatever its name, so a file class added later is carried without a code
change. `PIECE_LEVEL_MERGES` names the classes that are also merged into one composite file:
`exact_policy.csv`, concatenated in (seed, path) order with every row unchanged. A
directory in a piece refuses rather than being skipped, and `acceptance.json` is only ever
carried under its tagged name, so a composite is unstamped until certified itself.

Classification. The composite's `run_meta.json` is built from named classes
(`lmopt/logging/run_meta_classes.py`), never copied from piece 0: an ARM fact must be recorded with
one value on every piece, or omitted on every piece (an omitted key and a recorded null differ);
a PIECE fact is dropped from the top level and carried in full in that piece's `pieces[]` entry
(`piece_facts`), with `piece_fact_spread` stating which environment facts differ across pieces and
how many values each takes; a COMPUTED fact is derived here (`provenance` from every piece's own
stamp, tallies summed). A value in no class refuses the piece, naming it. `git_sha` is an arm fact:
`--allow-mixed-sha` composes pieces from more than one commit, sets the composite's `git_sha` to
null, lists the shas in `git_shas` and sets `mixed_git_sha`. `MUST_MATCH_FIELDS` stays the
base-versus-probe pairing set that `probe_scores` reads; the within-arm set is `ARM_AGREEMENT_FIELDS`.
A file named for a shard with no shard beside it refuses the piece.

Every piece and the output directory are caller-supplied paths, and no rung name is
hardcoded here. The one thing read from outside the arguments is the roster
`scripts/ladder_state.py` reads (`roster.JOBS` and `MODELS`), because the number of
pieces an arm should have is a fact about the roster, not about the operator's memory: a
piece whose path ends in a declared run's `rel_dir` (compared by path components) is that
run. The pieces must be the whole arm (`shape: arm`) or, for a kind collected as one
directory per game, every game of one run index (`shape: index`, a whole run); anything
else is refused unless `--allow-partial`, which composes it as `shape: partial`. Pieces at
paths the roster does not declare cannot have a count derived: they are refused unless
`--expected-count N` (recorded as `count_source: operator`) or `--allow-partial` is given.
The composite's `run_meta.json` carries `shape`, `n_pieces`, `n_expected` and
`count_source`. Importing the roster needs `lmopt`, so the script does not run in a
polars-only environment, and it carries the cross-checkout import guard.

Exit codes (distinct per outcome; a success signal is not the work):

  0  COMPOSED     the composite directory was written.
  2  usage        argparse rejected the invocation, or fewer than 1 --piece.
  3  REFUSED      a validation check failed; nothing was written to --out.
                  A piece lacks a stamped acceptance.json; the piece count
                  disagrees with --expected-count; the pieces are neither the
                  whole arm nor one whole run index of the roster (and
                  --allow-partial was not given), or no count can be derived
                  for pieces at undeclared paths; one cell's shards carry
                  different column sets across pieces; a must-match field
                  disagrees between two pieces; a piece's rows disagree with
                  each other or with its own run_meta on seed; two pieces
                  resolve to the same seed and share a game (a post-rekey
                  basename collision) -- same-seed, game-disjoint pieces
                  (the off-path shape) compose without refusing; a piece
                  directory holds an entry that is not a regular file;
                  `exact_policy.csv` is in some pieces and not others, has
                  differing headers, or carries different value columns for one
                  key in two pieces.
  4  OPERATIONAL  --out already exists and is not empty (refuse to guess
                  whether it is safe to write into), or a piece directory
                  could not be read at all.

Usage:
  scripts/compose_arm_pieces.py \\
      --piece data/run_experiment/recycle-v6-m1/i0/qwen2.5-14b-instruct \\
      --piece data/run_experiment/recycle-v6-m1/i1/qwen2.5-14b-instruct \\
      ... \\
      --out data/run_experiment/recycle-v6-m1/composite-qwen2.5-14b-instruct
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import sys
from pathlib import Path, PurePath

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmopt._import_guard import assert_lmopt_is_local
from lmopt.logging.call_log import (
    ABSENT,
    COMPLETE,
    call_log_status,
    orphan_call_logs,
)

assert_lmopt_is_local(__file__)

from lmopt.analysis.provenance import KEEPABLE, THROWAWAY, arm_provenance
from lmopt.logging import certified

from lmopt.logging.run_meta_classes import (
    ARM,
    ARM_GATED,
    MISSING,
    PIECE_ENV,
    PIECE_IDENTITY,
    agreement_value,
    distinct_values,
    get_path,
    paths_of,
    walk,
)
from scripts.ladder_state import RunKey, expected_runs, indexed_kinds

EXIT_COMPOSED = 0
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_OPERATIONAL = 4

ACCEPTANCE_FILENAME = "acceptance.json"
RUN_META_FILENAME = "run_meta.json"
SHARD_OUTCOMES_FILENAME = "shard_outcomes.json"

# Fields that must agree, exactly, across every piece being composed -- a
# disagreement here means the pieces are not safe to treat as one arm.
# Dotted paths resolve via `_get_path`. A full pairwise flatten-diff across
# every present index of all three v6 M1 models found the differing-field set
# is closed and never includes anything outside `PER_PIECE_DROP_FIELDS` below.
# An allowlist, not a blanket dict-equality minus a denylist, because that
# closed set was verified against real data, not assumed.
MUST_MATCH_FIELDS: tuple[str, ...] = (
    "model_id",
    "prompt_spec_hash",
    "resolved_config.model",
    "resolved_config.served_weights_revision",
    "resolved_config.memory_rung",
    "resolved_config.temperature",
    "resolved_config.n_hands",
    "resolved_config.n_seeds",
    "resolved_config.backend",
    "resolved_config.seat_mode",
    "environment.pyspiel_version",
    "environment.numpy_version",
    # The run-level top-k logprobs setting changes the request an action call sends, so a
    # base arm and its probe arm, or two pieces of one arm, must agree on it. See
    # ABSENT_MEANS_NONE_FIELDS for how an arm collected before the key existed compares.
    "resolved_config.action_top_logprobs",
)

# `ABSENT_MEANS_NONE_FIELDS` (imported above) names the one field for which an absent key and a
# recorded None are the same value; see `lmopt.logging.run_meta_classes`.

# Composite-only must-match fields, deliberately not added to MUST_MATCH_FIELDS:
# probe_scores.py iterates that tuple to compare a base arm against a probe arm
# (`_mismatches`), where a `probe` disagreement is the expected, checked-elsewhere
# shape of the comparison, not a defect -- adding it there would make every such
# comparison refuse. Here, composing pieces into one arm, a `probe`/`cfr_iters`
# disagreement means the pieces are not the same arm at all (a base piece beside
# a probe piece, or two probe pieces solved at different CFR depths) and the
# composite would otherwise silently inherit piece 0's value.
COMPOSITE_MUST_MATCH_FIELDS: tuple[str, ...] = (
    "resolved_config.probe",
    "resolved_config.cfr_iters",
)

# Fields that are legitimately PER game on a game-disjoint composite: an off-path run is one
# run per (game, index), and each game's launch config names its own forced opening and seat
# mode. Composition records each as a scalar when every game agrees and as `{game: value}`
# when they differ, and refuses when two pieces name one game with different values. Both are
# read per game downstream (`sentinel_anchors.anchor_for`, `arm_reading.read_arm`), so a
# consumer needs the per-game value, not a label. `resolved_config.seat_mode` stays in
# MUST_MATCH_FIELDS because `probe_scores.py` compares that tuple between a base and a paired
# probe arm; `check_must_match` skips the entries listed here.
PER_GAME_FIELDS: tuple[str, ...] = (
    "resolved_config.drive_openings",
    "resolved_config.seat_mode",
)

# The job keys a piece's `pieces[]` entry carries. A piece collected before they existed has
# neither key, and its entry then omits them: an absent key is not written as null, which would
# claim a recorded "no job".
PIECE_JOB_KEYS: tuple[str, ...] = ("job", "previous_jobs")

# Every piece fact, identity and environment: dropped from the composite's top level and carried,
# in full, in that piece's `pieces[]` entry (`piece_facts`). Derived from the classification in
# `lmopt.logging.run_meta_classes`, never listed here a second time. `posterior_fallback_counts`
# and the other COMPUTED keys are derived by compose instead.
PER_PIECE_DROP_FIELDS: tuple[str, ...] = paths_of(PIECE_IDENTITY, PIECE_ENV)

# The within-arm agreement set: every ARM fact, which pieces of one arm must all record with one
# value, or all omit. A superset of MUST_MATCH_FIELDS (the base-versus-probe pairing set, read by
# `probe_scores._mismatches`); used only by compose.
ARM_AGREEMENT_FIELDS: tuple[str, ...] = paths_of(ARM)

# The ARM facts the two existing tuples above do not already check, so each set has exactly one
# check and removing a name from one is seen by a test.
EXTRA_ARM_FIELDS: tuple[str, ...] = tuple(
    path
    for path in ARM_AGREEMENT_FIELDS
    if path not in MUST_MATCH_FIELDS and path not in COMPOSITE_MUST_MATCH_FIELDS
)

# The key a composite carries when `--allow-mixed-sha` composes pieces from more than one commit.
MIXED_SHA_FLAG = "--allow-mixed-sha"
#: The ARM facts that name the code a piece ran; `MIXED_SHA_FLAG` lets pieces disagree on them.
MIXED_SHA_FIELDS = ("git_sha", "code_tree")

# A name ending in one of these is a file whose write was never completed: every atomic writer in
# the repo builds its temporary name with ".tmp",
# and ".partial" is the older spelling.
IN_FLIGHT_SUFFIXES: tuple[str, ...] = (".tmp", ".partial")

# Files compose writes itself for the composite, so a piece's own copy is never carried.
SYNTHESISED_PIECE_FILES: frozenset[str] = frozenset(
    {RUN_META_FILENAME, SHARD_OUTCOMES_FILENAME}
)

# The piece-level files a piece may hold beyond its shards and their sidecars. A name outside this
# set is refused, never carried: a stray ordinary file would ride into the composite under a piece
# tag and read as an artefact of the arm. The set is the enumeration of what every real v7 piece
# holds (acceptance.json and serving_manifest.json in all of them, exact_policy.csv where the
# exact pass ran, katana_job.out where the job log was pulled, and exact_policy_pass.json, the
# record of whether the pass ran, from the arms collected after it was introduced), so it refuses
# nothing that exists.
# A new legitimate artefact is added here on purpose, in the same change that produces it.
PIECE_LEVEL_FILES: frozenset[str] = frozenset(
    {
        "acceptance.json",
        "serving_manifest.json",
        "exact_policy.csv",
        "exact_policy_pass.json",
        "katana_job.out",
    }
)

# `exact_policy.csv` merge: rows are keyed by these columns, and every other column named in
# `EXACT_POLICY_VALUE_COLUMNS` must be byte-identical for one key across the pieces that reach
# it. The remaining columns (`logged_*`, `population_*`, `base_url`, `run_timestamp`, ...)
# describe the piece's own run and are carried per row, unmerged.
EXACT_POLICY_KEY_COLUMNS: tuple[str, ...] = (
    "game",
    "game_params",
    "info_set",
    "action",
)
EXACT_POLICY_VALUE_COLUMNS: tuple[str, ...] = (
    "action_text",
    "raw_logprob",
    "prob",
    "n_tokens",
    "error",
    "error_kind",
    "coverage_mass",
    "exact_modal_action",
)


SHAPE_ARM = "arm"
SHAPE_INDEX = "index"
SHAPE_PARTIAL = "partial"
COUNT_ROSTER = "roster"
COUNT_OPERATOR = "operator"


class ComposeRefused(Exception):
    """A validation check failed; nothing is written to --out."""


def _get_path(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def load_piece(piece_dir: Path) -> dict:
    """Load and minimally validate one piece directory's manifests.

    Raises `ComposeRefused` if the piece has no stamped `acceptance.json` (the
    landing order is: certify+stamp each piece first, exactly as any other
    arm, then compose only already-stamped pieces) or has zero Parquet
    shards. `run_meta.json`/`shard_outcomes.json` being unreadable is treated
    as OPERATIONAL by the caller (`main`), not REFUSED: a missing manifest is
    a different kind of problem than pieces that disagree.
    """
    run_meta_path = piece_dir / RUN_META_FILENAME
    run_meta = json.loads(run_meta_path.read_text())

    acceptance_path = piece_dir / ACCEPTANCE_FILENAME
    if not acceptance_path.exists():
        raise ComposeRefused(
            f"{piece_dir}: no stamped {ACCEPTANCE_FILENAME} -- certify and "
            "stamp this piece (recycle_arm_evidence.py --stamp) before "
            "composing it"
        )

    shards = sorted(piece_dir.glob("*.parquet"))
    if not shards:
        raise ComposeRefused(f"{piece_dir}: zero Parquet shards on disk")

    outcomes = json.loads((piece_dir / SHARD_OUTCOMES_FILENAME).read_text())

    return {
        "dir": piece_dir,
        "run_meta": run_meta,
        "outcomes": outcomes,
        "shards": shards,
    }


def resolve_piece_seed(piece: dict) -> int:
    """The TRUE seed this piece's rows carry, cross-checked against run_meta.

    Reads every shard's `seed` column. Refuses if a shard carries more than
    one distinct seed value, if two shards in the same piece disagree, or if
    the agreed value disagrees with `run_meta.resolved_config.seed` -- a
    piece whose rows don't match its own declared seed is not safe to compose
    under any name.
    """
    piece_dir = piece["dir"]
    seeds_seen: set[int] = set()
    for shard in piece["shards"]:
        distinct = pl.read_parquet(shard, columns=["seed"])["seed"].unique().to_list()
        if len(distinct) != 1:
            raise ComposeRefused(
                f"{shard}: {len(distinct)} distinct seed value(s) in one "
                f"shard ({distinct}), expected exactly 1"
            )
        seeds_seen.add(distinct[0])
    if len(seeds_seen) != 1:
        raise ComposeRefused(
            f"{piece_dir}: shards disagree with each other on seed: "
            f"{sorted(seeds_seen)}"
        )
    row_seed = next(iter(seeds_seen))
    declared_seed = _get_path(piece["run_meta"], "resolved_config.seed")
    if declared_seed != row_seed:
        raise ComposeRefused(
            f"{piece_dir}: row seed {row_seed} != "
            f"run_meta.resolved_config.seed {declared_seed!r}"
        )
    return row_seed


def _shown(piece: dict, path: str) -> str:
    """
    :param piece: a loaded piece (dict)
    :param path: a declared path (str)
    :returns: what the piece records there for a refusal message: the value, or "absent"
    """
    value = get_path(piece["run_meta"], path)
    return "absent" if value is MISSING else repr(value)


def _disagreement(path: str, base: dict, other: dict) -> str:
    """
    :param path: the field the two pieces disagree on (str)
    :param base: the piece the others are compared with (dict)
    :param other: the piece that differs (dict)
    :returns: one line naming the field and both pieces' values (str)
    """
    return f"{path}: {base['dir']} has {_shown(base, path)}, {other['dir']} has {_shown(other, path)}"


def _check_agreement(pieces: list[dict], fields) -> None:
    """
    refuse unless every field is recorded with one value on every piece, or omitted on every
    piece; an omitted key and a recorded null are different values, except for the field in
    ABSENT_MEANS_NONE_FIELDS
    :param pieces: loaded pieces (list of dict)
    :param fields: declared paths to check (iterable of str)
    :raises ComposeRefused: naming every field and every piece that differs from the first, in
        one refusal, so several disagreeing facts cost one round trip and not one each
    """
    base = pieces[0]
    problems = [
        _disagreement(field, base, other)
        for field in fields
        for other in pieces[1:]
        if agreement_value(other["run_meta"], field)
        != agreement_value(base["run_meta"], field)
    ]
    if problems:
        raise ComposeRefused(
            "; ".join(problems) + " -- pieces are not safe to compose as one arm"
        )


# Correspondence-CHECK: piece-must-match
def check_must_match(pieces: list[dict]) -> None:
    """Refuse (naming the field and the two disagreeing pieces) unless every
    MUST_MATCH_FIELDS entry is recorded identically on all pieces, or omitted on all of them.
    """
    _check_agreement(pieces, (f for f in MUST_MATCH_FIELDS if f not in PER_GAME_FIELDS))


def check_composite_must_match(pieces: list[dict]) -> None:
    """
    refuse unless every COMPOSITE_MUST_MATCH_FIELDS entry agrees across all pieces
    Separate from `check_must_match`/`MUST_MATCH_FIELDS`: those fields are also read by
    `probe_scores.py` to compare a base arm against a probe arm, where a `probe`
    disagreement is the expected shape being checked, not a defect. Here it means the
    pieces are not the same arm.
    :param pieces: loaded pieces, as `check_must_match` receives them (list of dict)
    :raises ComposeRefused: naming the field and the two disagreeing pieces
    """
    base = pieces[0]
    _check_agreement(pieces, COMPOSITE_MUST_MATCH_FIELDS)
    # Above M0 the memory block is part of every prompt after the first hand, so pieces
    # collected under two memory wordings are two conditions. At M0 nothing is rendered, and
    # pieces from before the digest existed must still compose with pieces from after.
    if _get_path(base["run_meta"], "resolved_config.memory_rung") not in (None, "M0"):
        from lmopt.prompting.memory import MEMORY_RENDER_HASH_KEY

        recorded = {
            piece["dir"]: piece["run_meta"].get(MEMORY_RENDER_HASH_KEY)
            for piece in pieces
        }
        if None in recorded.values() or len(set(recorded.values())) > 1:
            raise ComposeRefused(
                f"{MEMORY_RENDER_HASH_KEY} per piece {recorded!r} -- above M0 every piece must "
                "record one and the same memory wording"
            )


def check_classification(pieces: list[dict]) -> None:
    """
    refuse a piece whose `run_meta.json` records a value that no class names
    There is no default class: an arm fact is compared, a piece fact is carried per piece, a
    computed fact is derived, and a key in none of them stops the composition until someone
    classifies it (`lmopt.logging.run_meta_classes.DECLARED`).
    :param pieces: loaded pieces (list of dict)
    :raises ComposeRefused: naming the piece and every unclassified path
    """
    for piece in pieces:
        _found, unknown = walk(piece["run_meta"])
        if unknown:
            raise ComposeRefused(
                f"{piece['dir']}: run_meta.json records {', '.join(unknown)}, which no class "
                "names (arm fact, piece fact or computed fact); classify it in "
                "lmopt/logging/run_meta_classes.py before composing"
            )


def check_arm_agreement(pieces: list[dict], *, allow_mixed_sha: bool = False) -> None:
    """
    refuse unless every ARM fact the two must-match tuples do not already check agrees
    Each is recorded with one value on every piece, or omitted on every piece. `git_sha` is one
    of them: pieces from two commits are not one arm, unless `allow_mixed_sha` says to compose
    them anyway (the composite then records that, see `_write_composite_run_meta`).
    :param pieces: loaded pieces (list of dict)
    :param allow_mixed_sha: whether pieces that disagree on `git_sha` may compose (bool)
    :raises ComposeRefused: naming the field and the first two pieces that differ, and for a
        `git_sha` disagreement the flag that would compose it
    """
    fields = tuple(
        f for f in EXTRA_ARM_FIELDS if not (allow_mixed_sha and f in MIXED_SHA_FIELDS)
    )
    try:
        _check_agreement(pieces, fields)
    except ComposeRefused as exc:
        if str(exc).startswith(tuple(f"{f}:" for f in MIXED_SHA_FIELDS)):
            raise ComposeRefused(
                f"{exc}; a mixed-commit arm composes only with {MIXED_SHA_FLAG}, which sets the "
                "composite's git_sha to null and records every sha"
            ) from exc
        raise


def check_count_presence(pieces: list[dict], key: str) -> None:
    """
    refuse when a computed tally is recorded on some pieces and omitted on others
    :param pieces: loaded pieces (list of dict)
    :param key: a top-level run_meta key compose sums (str)
    :raises ComposeRefused: naming the key and the pieces on each side
    """
    have = [str(p["dir"]) for p in pieces if key in p["run_meta"]]
    if have and len(have) != len(pieces):
        lack = [str(p["dir"]) for p in pieces if key not in p["run_meta"]]
        raise ComposeRefused(
            f"{key} is recorded on {have} and omitted on {lack}; a composite summing it would "
            "cover a subset of the arm without saying so"
        )


def check_orphan_sidecars(pieces: list[dict]) -> None:
    """
    refuse a piece holding a shard-stem file with no shard beside it
    One rule for every sidecar class (call logs, timing sidecars, any added later): a file named
    for a shard (`<game>_lm_vs_<opponent>...`) that no shard of the piece owns is the trace of a
    shard that started and never flushed, so the piece is refused and the file named.
    :param pieces: loaded pieces (list of dict)
    :raises ComposeRefused: naming the file
    """
    for piece in pieces:
        owned = set(piece["shards"])
        for shard in piece["shards"]:
            owned.update(_sidecar_paths(piece["dir"], shard))
        for path in sorted(piece["dir"].iterdir()):
            if path in owned or "_lm_vs_" not in path.name or not path.is_file():
                continue
            raise ComposeRefused(
                f"{path}: a file named for a shard with no shard beside it (a shard that "
                "started and never flushed); the piece is not safe to compose"
            )


def check_no_interrupted_writes(pieces: list[dict]) -> None:
    """
    refuse a piece holding a file whose write never completed
    The shape rule, a computation over the name and not a list of known artefacts: a name ending
    in `IN_FLIGHT_SUFFIXES` is an unfinished write whatever artefact it was for, and a
    dot-prefixed name is hidden state that is not an artefact. Precedence is stated here and
    pinned by a test: a shard and its own sidecars (matched by the shard's stem) belong to the
    piece before the dot rule is applied, so only the suffix rule reaches them; every other
    dot-prefixed name is refused. A shard-named stray is also caught by `check_orphan_sidecars`;
    this check is the one that reaches a stray named for no shard (`.exact_policy.csv.tmp`).
    :param pieces: loaded pieces (list of dict)
    :raises ComposeRefused: naming the file and saying a write was interrupted
    """
    for piece in pieces:
        owned: set[Path] = set(piece["shards"])
        for shard in piece["shards"]:
            owned.update(_sidecar_paths(piece["dir"], shard))
        for path in sorted(piece["dir"].iterdir()):
            unfinished = path.name.endswith(IN_FLIGHT_SUFFIXES)
            hidden = path.name.startswith(".") and path not in owned
            if unfinished or hidden:
                raise ComposeRefused(
                    f"{path}: a write was interrupted or this is hidden state (a name ending "
                    f"{list(IN_FLIGHT_SUFFIXES)} or starting with a dot); compose will not carry "
                    "it into a composite. Remove it only once you know what was being written"
                )


def _piece_games(piece: dict) -> list[str] | None:
    """
    the games one piece's run_meta says it collected
    :param piece: a loaded piece (dict)
    :returns: `resolved_config.games`, or None when the piece records no game list (list of
        str or None)
    """
    return (piece["run_meta"].get("resolved_config") or {}).get("games") or None


def _per_game_values(
    pieces: list[dict], field: str
) -> dict[str, tuple[Path, object]] | None:
    """
    `{game: (the first piece's dir naming it, that piece's value of `field`)}`, refusing on the
    first two pieces that name one game with differing values
    Shared by the upfront refusal check and the composite-value computation, so the two can
    never see a different notion of which pieces disagree.
    :param pieces: loaded pieces, in the order `check_must_match` received them (list of dict)
    :param field: a PER_GAME_FIELDS entry (str)
    :returns: one entry per distinct game key, or None when any piece records no game list,
        so the field cannot be keyed by game at all (dict of str to (Path, Any), or None)
    :raises ComposeRefused: naming the game and the two disagreeing pieces
    """
    named = [games for games in map(_piece_games, pieces) if games]
    if len(named) != len(pieces):
        return None
    per_game: dict[str, tuple[Path, object]] = {}
    for piece, games in zip(pieces, named, strict=True):
        value = _get_path(piece["run_meta"], field)
        for game in games:
            if game in per_game:
                prior_dir, prior_value = per_game[game]
                if prior_value != value:
                    raise ComposeRefused(
                        f"{field} for {game!r}: {prior_dir} has {prior_value!r}, "
                        f"{piece['dir']} has {value!r} -- pieces are not safe to compose "
                        "as one arm"
                    )
            else:
                per_game[game] = (piece["dir"], value)
    return per_game


def check_composite_per_game_fields(pieces: list[dict]) -> None:
    """
    refuse when two pieces name one game with a differing value of a PER_GAME_FIELDS entry
    A game-disjoint composite's pieces may legitimately differ on these fields, one game per
    piece, so this is a per-game check rather than `check_must_match`'s whole-value one. When
    a piece records no game list the fields cannot be keyed by game, and they fall back to the
    whole-value comparison `check_must_match` applies to every other field. Called upfront,
    before any shard is linked, so a conflict leaves `--out` unwritten like every other
    refusal.
    :param pieces: loaded pieces, as `check_must_match` receives them (list of dict)
    :raises ComposeRefused: naming the field, the game where there is one, and the two pieces
    """
    base = pieces[0]
    for field in PER_GAME_FIELDS:
        if _per_game_values(pieces, field) is not None:
            continue
        base_value = _get_path(base["run_meta"], field)
        for other in pieces[1:]:
            other_value = _get_path(other["run_meta"], field)
            if other_value != base_value:
                raise ComposeRefused(
                    f"{field}: {base['dir']} has {base_value!r}, {other['dir']} has "
                    f"{other_value!r} -- pieces are not safe to compose as one arm"
                )


def _composite_per_game_value(pieces: list[dict], field: str) -> tuple[bool, object]:
    """
    the composite's value of one PER_GAME_FIELDS entry, keyed by game where pieces differ
    Assumes `check_composite_per_game_fields` already ran and did not raise. When every game's
    pieces agree on one value, that flat scalar is returned unchanged, so no existing
    composite's `run_meta.json` shape moves; only a genuine per-game difference upgrades the
    field to a `{game: value}` mapping. The first element is False when the field cannot be
    keyed by game (a piece records no game list), which tells the caller to leave the value
    it inherited from piece 0 alone -- the check above already required every piece to agree
    on it. It is not the same as a value that is None, which is a real, recorded value.
    :param pieces: loaded pieces, in the order `check_must_match` received them (list of dict)
    :param field: a PER_GAME_FIELDS entry (str)
    :returns: `(False, None)` when the field cannot be keyed by game; otherwise `(True, value)`,
        value a scalar when every game's pieces agree (possibly None itself) and
        `{game: value}` when they differ (tuple of bool and (Any | dict[str, Any] | None))
    """
    per_game = _per_game_values(pieces, field)
    if per_game is None:
        return False, None
    values = [v for _, v in per_game.values()]
    # Values may be lists (unhashable), so uniqueness is a linear scan against the first.
    if all(v == values[0] for v in values[1:]):
        return True, values[0]
    return True, {game: value for game, (_, value) in sorted(per_game.items())}


def _sidecar_paths(piece_dir: Path, shard_path: Path) -> list[Path]:
    """Every file beside `shard_path` derived from its own stem, any suffix.

    Generic by construction: `posterior_fallback_sidecar_path` and
    `driven_discard_sidecar_path` (`src/lmopt/orchestration/shard.py`) both
    derive as `splitext(out_path)[0] + suffix`, so any file matching that
    shape is found and renamed in lockstep -- not just the one sidecar type
    this rung happens to have (M1 has no driven-discard sidecars; a future
    driven arm composed with this tool must not silently strand them under
    stale names). When the piece also holds a shard whose stem extends this one's (a base shard and
    its variant shard), that shard and its sidecars belong to the longer stem and are not returned.
    """
    prefix = shard_path.stem + "."
    # A file belongs to the shard with the LONGEST stem that prefixes it: a base shard `X` and a
    # variant shard `X.relabel_v1` share the prefix `X.`, and the variant shard and its sidecars
    # must not be claimed by `X`.
    longer_stems = [
        other.stem + "."
        for other in piece_dir.glob("*.parquet")
        if other != shard_path and len(other.stem) > len(shard_path.stem)
    ]
    return sorted(
        p
        for p in piece_dir.iterdir()
        if p.name.startswith(prefix)
        and p != shard_path
        and not any(p.name.startswith(longer) for longer in longer_stems)
    )


def _piece_level_files(piece: dict) -> list[Path]:
    """
    every file in a piece directory that belongs to the piece as a whole
    :param piece: a loaded piece (dict from `load_piece`)
    :returns: the regular files in the piece directory that are not a shard, not a shard's
        sidecar and not a file compose writes itself (`SYNTHESISED_PIECE_FILES`), in name order
    :raises ComposeRefused: when the directory holds an entry that is not a regular file, since
        this tool cannot say how to carry it, or a regular file whose name is not in
        `PIECE_LEVEL_FILES`
    """
    piece_dir = piece["dir"]
    shard_side = set(piece["shards"])
    for shard in piece["shards"]:
        shard_side.update(_sidecar_paths(piece_dir, shard))
    found = []
    for path in sorted(piece_dir.iterdir()):
        if path in shard_side or path.name in SYNTHESISED_PIECE_FILES:
            continue
        if not path.is_file():
            raise ComposeRefused(
                f"{path}: a piece directory holds an entry that is not a regular file; "
                "compose cannot carry it and will not skip it silently"
            )
        if path.name not in PIECE_LEVEL_FILES:
            raise ComposeRefused(
                f"{path}: a piece directory holds a file compose does not know "
                f"({path.name!r}); the piece-level files it carries are "
                f"{sorted(PIECE_LEVEL_FILES)}. A stray file would be carried into the composite "
                "as if it were an artefact of the arm; remove it, or add it to "
                "PIECE_LEVEL_FILES if it is a real artefact"
            )
        found.append(path)
    return found


def _merged_file_label(name: str) -> str:
    """
    how a merged composite file is named in a failure message
    :param name: the merged file's name
    :returns: the label standing where a source path would
    """
    return f"the merge of every piece's {name}"


def _piece_tags(pieces: list[dict], seeds_by_piece: dict[Path, int]) -> dict[Path, str]:
    """
    the label that keeps one piece's piece-level files apart from every other piece's
    :param pieces: loaded pieces
    :param seeds_by_piece: each piece's true seed
    :returns: {piece dir: `seed{S}`, or `seed{S}_{n}` for the n-th piece (n from 2, in path
        order) that shares seed S with an earlier one}
    """
    tags: dict[Path, str] = {}
    seen: dict[int, int] = {}
    for piece in sorted(
        pieces, key=lambda item: (seeds_by_piece[item["dir"]], str(item["dir"]))
    ):
        seed = seeds_by_piece[piece["dir"]]
        seen[seed] = seen.get(seed, 0) + 1
        tags[piece["dir"]] = (
            f"seed{seed}" if seen[seed] == 1 else f"seed{seed}_{seen[seed]}"
        )
    return tags


def merge_exact_policy_csv(sources: list[Path]) -> str:
    """
    concatenate the pieces' `exact_policy.csv` into the text of the composite's one file
    :param sources: each piece's file, in the order their rows are to appear
    :returns: the merged CSV text, the shared header once and every piece's rows unchanged
    :raises ComposeRefused: when the headers differ, a key column is absent, or one
        (game, game_params, info_set, action) key carries a different
        `EXACT_POLICY_VALUE_COLUMNS` value in two pieces. Rows are not deduplicated: the
        `logged_*` and `population_*` columns describe the piece a row came from.
    """
    header: list[str] | None = None
    first_seen: dict[tuple, tuple[Path, dict]] = {}
    rows: list[dict] = []
    for source in sources:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            these = list(reader.fieldnames or [])
            if header is None:
                header = these
                missing = [
                    c
                    for c in EXACT_POLICY_KEY_COLUMNS + EXACT_POLICY_VALUE_COLUMNS
                    if c not in header
                ]
                if missing:
                    raise ComposeRefused(
                        f"{source}: exact_policy.csv lacks column(s) {missing}"
                    )
            elif these != header:
                raise ComposeRefused(
                    f"{source}: exact_policy.csv header differs from {sources[0]}'s; "
                    "the pieces were not produced by the same pass"
                )
            for row in reader:
                if None in row or None in row.values():
                    raise ComposeRefused(
                        f"{source}: exact_policy.csv row {reader.line_num} has a different "
                        "number of fields than its header"
                    )
                key = tuple(row[c] for c in EXACT_POLICY_KEY_COLUMNS)
                earlier = first_seen.get(key)
                if earlier is None:
                    first_seen[key] = (source, row)
                else:
                    for column in EXACT_POLICY_VALUE_COLUMNS:
                        if earlier[1][column] != row[column]:
                            raise ComposeRefused(
                                f"exact_policy.csv key {key}: column {column!r} is "
                                f"{earlier[1][column]!r} in {earlier[0]} but {row[column]!r} "
                                f"in {source}; the pieces disagree, so no single composite "
                                "row can stand for both"
                            )
                rows.append(row)
    assert header is not None
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


# Piece-level files that are merged into one composite file as well as carried per piece,
# keyed by the file name. A piece-level file not named here is only carried.
PIECE_LEVEL_MERGES = {"exact_policy.csv": merge_exact_policy_csv}


def _sum_counts(pieces: list[dict], key: str) -> dict[str, dict[str, int]]:
    """
    the per-game tallies of every piece, added up
    :param pieces: loaded pieces (list of dict)
    :param key: a top-level run_meta key holding `{game: {name: int}}` (str)
    :returns: the sums; a count is summed and never inherited from one piece (dict)
    """
    summed: dict[str, dict[str, int]] = {}
    for piece in pieces:
        for game, gcounts in (piece["run_meta"].get(key) or {}).items():
            dest = summed.setdefault(game, {})
            for name, value in gcounts.items():
                dest[name] = dest.get(name, 0) + value
    return summed


def _piece_entry(piece: dict, seed: int) -> dict:
    """
    one `pieces[]` entry: the facts compose has always listed, and every piece fact in full
    :param piece: a loaded piece (dict)
    :param seed: the piece's true seed (int)
    :returns: the entry (dict). `piece_facts` maps each piece-fact path the piece records to its
        value, so nothing a piece fact says is discarded; a path the piece omits is omitted.
    """
    meta = piece["run_meta"]
    return {
        "piece_dir": str(piece["dir"]),
        "seed": seed,
        "git_sha": meta.get("git_sha"),
        "created_utc": meta.get("created_utc"),
        "finished_utc": meta.get("finished_utc"),
        **{k: meta[k] for k in PIECE_JOB_KEYS if k in meta},
        "serving_gpu_name": _get_path(meta, "serving.gpu_name"),
        "serving_gpu_memory_total_mib": _get_path(meta, "serving.gpu_memory_total_mib"),
        "acceptance_path": str(piece["dir"] / ACCEPTANCE_FILENAME),
        "piece_facts": {
            path: copy.deepcopy(value)
            for path in PER_PIECE_DROP_FIELDS
            if (value := get_path(meta, path)) is not MISSING
        },
    }


def _write_composite_run_meta(
    pieces: list[dict],
    seeds_by_piece: dict[Path, int],
    out_dir: Path,
    completeness: dict,
    *,
    allow_mixed_sha: bool = False,
) -> None:
    """
    write the composite's `run_meta.json`, built from the classes and never from piece 0
    ARM facts are copied from the first piece (every piece agrees, or the check refused), piece
    facts are carried only in `pieces[]`, and COMPUTED facts are derived here. A key piece 0
    happens to carry cannot reach the composite unless some class names it.
    :param pieces: loaded pieces (list of dict)
    :param seeds_by_piece: each piece's true seed (dict)
    :param out_dir: the composite directory (Path)
    :param completeness: the completeness record (`shape`, `n_pieces`, `n_expected`,
        `count_source`) (dict)
    :param allow_mixed_sha: whether pieces from more than one commit were allowed (bool)
    """
    first = pieces[0]["run_meta"]
    base: dict = {}
    # PER_GAME facts start from piece 0 too: when no piece names its games they cannot be keyed by
    # game, `check_composite_per_game_fields` has already required the pieces to agree, and a
    # recorded null must survive as a present key.
    for path in (*ARM_AGREEMENT_FIELDS, *PER_GAME_FIELDS):
        value = get_path(first, path)
        if value is not MISSING:
            _set_path(base, path, copy.deepcopy(value))
    # An ARM_GATED fact is carried only when every piece records the same value, so piece 0's
    # value never stands for pieces that differ (an M0 piece from before the digest existed
    # beside one from after).
    for path in paths_of(ARM_GATED):
        if (
            len({agreement_value(m, path) for m in (p["run_meta"] for p in pieces)})
            == 1
        ):
            value = get_path(first, path)
            if value is not MISSING:
                _set_path(base, path, copy.deepcopy(value))
    base.update(completeness)

    shas = sorted(
        {p["run_meta"]["git_sha"] for p in pieces if "git_sha" in p["run_meta"]},
        key=str,
    )
    if allow_mixed_sha and len(shas) > 1:
        base["git_sha"] = None
        base["git_shas"] = shas
        base["mixed_git_sha"] = True
    # The code a piece ran is the same kind of fact as its sha: a mixed-commit composite carries
    # every distinct record (null for a piece that recorded none) and no single one.
    trees = sorted(
        {json.dumps(p["run_meta"].get("code_tree"), sort_keys=True) for p in pieces}
    )
    if allow_mixed_sha and len(trees) > 1:
        base.pop("code_tree", None)
        base["code_trees"] = [json.loads(t) for t in trees]

    # Per-game row tally: summed across pieces, not a config value subject to the agreement check.
    base["posterior_fallback_counts"] = _sum_counts(pieces, "posterior_fallback_counts")
    if any("driven_discard_counts" in p["run_meta"] for p in pieces):
        base["driven_discard_counts"] = _sum_counts(pieces, "driven_discard_counts")

    # A composite is keepable if and only if every piece is. Each piece is evaluated through
    # `arm_provenance` (its own acceptance.json), never copied from piece 0.
    verdicts = [arm_provenance(piece["dir"]).value for piece in pieces]
    n_keepable = sum(1 for v in verdicts if v == KEEPABLE)
    base["provenance"] = KEEPABLE if n_keepable == len(pieces) else THROWAWAY
    base["provenance_basis"] = {
        "derived_from": "pieces",
        "n_keepable": n_keepable,
        "n_pieces": len(pieces),
    }

    # The games the composed arm was asked to play are every piece's games together, so a
    # per-game piece cannot make the composite look like it covered one game only. If any
    # piece records no game list the composite's is unknown, and it is left unrecorded rather
    # than copied from piece 0.
    piece_games = [
        (piece["run_meta"].get("resolved_config") or {}).get("games")
        for piece in pieces
    ]
    named_games = [games for games in piece_games if games]
    if len(named_games) == len(piece_games):
        _set_path(base, "resolved_config.games", sorted(set().union(*named_games)))

    # Each PER_GAME_FIELDS value stands only when every game's pieces agree with it. `found`
    # (not `value is None`) decides whether to overwrite: every real driven piece names its
    # games and records `drive_openings: null` explicitly, so the correct composite value is
    # `None`, present, not a missing key.
    for field in PER_GAME_FIELDS:
        found, value = _composite_per_game_value(pieces, field)
        if found:
            _set_path(base, field, value)

    # The environment piece facts whose values are not all equal across the pieces, with how many
    # distinct values each takes. Always present; empty when the arm was uniform.
    metas = [piece["run_meta"] for piece in pieces]
    base["piece_fact_spread"] = {
        path: len(values)
        for path in paths_of(PIECE_ENV)
        if len(values := distinct_values(metas, path)) > 1
    }

    base["pieces"] = [
        _piece_entry(piece, seeds_by_piece[piece["dir"]]) for piece in pieces
    ]
    base["composed_from"] = [str(piece["dir"]) for piece in pieces]

    (out_dir / RUN_META_FILENAME).write_text(json.dumps(base, indent=2, sort_keys=True))


def _write_composite_shard_outcomes(
    pieces: list[dict], seeds_by_piece: dict[Path, int], out_dir: Path
) -> None:
    shards_out: list[dict] = []
    n_requested = n_completed = n_failed = 0
    for piece in pieces:
        seed = seeds_by_piece[piece["dir"]]
        outcomes = piece["outcomes"]
        n_requested += outcomes.get("n_requested", 0)
        n_completed += outcomes.get("n_completed", 0)
        n_failed += outcomes.get("n_failed", 0)
        for shard_entry in outcomes.get("shards", []):
            entry = dict(shard_entry)
            old_out_path = entry.get("out_path")
            if old_out_path:
                stem, ext = os.path.splitext(os.path.basename(old_out_path))
                entry["out_path"] = str(out_dir / f"{stem}_seed{seed}{ext}")
            shards_out.append(entry)
    payload = {
        "n_requested": n_requested,
        "n_completed": n_completed,
        "n_failed": n_failed,
        "shards": shards_out,
    }
    (out_dir / SHARD_OUTCOMES_FILENAME).write_text(json.dumps(payload, indent=2))


def _piece_shard_games(piece: dict) -> set[str]:
    """
    the game each of a piece's own shard filenames names, read from disk not config
    Shard basenames are `<game>_lm_vs_<opponent>[.variant].parquet` (module docstring); the
    game is everything before the fixed `_lm_vs_` separator. Read from the actual shard
    list rather than `resolved_config.games` so a config that lies about what a piece
    collected cannot mask a real rename collision.
    :param piece: a loaded piece, as `check_must_match` receives it (dict)
    :returns: the distinct game names this piece's shards belong to (set of str)
    """
    return {shard.name.split("_lm_vs_", 1)[0] for shard in piece["shards"]}


def check_duplicate_seeds(pieces: list[dict], seeds_by_piece: dict[Path, int]) -> None:
    """
    refuse a same-seed pair of pieces only when their shard games overlap
    Off-path collection is one run per (game, index), so composing one index's five
    per-game directories together is a real, needed shape -- all five legitimately share a
    seed, and none collide, because the rename target is `<game>_lm_vs_<opp>_seed{seed}
    {ext}` and their `<game>` prefixes differ. A same-seed pair sharing a game would
    collide (two pieces would rename to the identical basename, and `os.link` would fail
    or silently pick whichever ran first), so that pair alone must still refuse.
    :param pieces: loaded pieces, as `check_must_match` receives them (list of dict)
    :param seeds_by_piece: each piece's resolved true seed, keyed by piece dir (dict of
        Path to int)
    :raises ComposeRefused: naming the seed, the game and the two colliding pieces
    """
    seen_by_seed: dict[int, dict[str, Path]] = {}
    for piece in pieces:
        seed = seeds_by_piece[piece["dir"]]
        seen_games = seen_by_seed.setdefault(seed, {})
        for game in _piece_shard_games(piece):
            if game in seen_games:
                raise ComposeRefused(
                    f"duplicate seed {seed}: {seen_games[game]} and {piece['dir']} both "
                    f"have {game!r} shards -- their renamed basenames would collide"
                )
            seen_games[game] = piece["dir"]


def check_call_logs(pieces: list[dict]) -> None:
    """
    refuse a piece whose per-call logs do not match its shards. A log with no shard beside it
    is a killed shard's trace (compose's sidecar rule matches by shard stem, so an orphan
    would otherwise be invisible), and a shard whose log is not complete for the run its
    rows carry did not reach its flush the way the log says it should. A shard with no log
    at all is fine: every shard collected before the log existed has none
    :param pieces: loaded pieces (list of dict)
    :raises ComposeRefused: naming the piece, the file and what is wrong
    """
    for piece in pieces:
        orphans = orphan_call_logs(str(piece["dir"]))
        if orphans:
            raise ComposeRefused(
                f"{piece['dir']}: call log(s) with no shard beside them: {', '.join(orphans)}; "
                "a shard was killed before it flushed. Re-run it, or remove the log if the "
                "shard is knowingly dropped"
            )
        for shard in piece["shards"]:
            run_ids = sorted(
                pl.read_parquet(shard, columns=["run_id"])["run_id"].unique().to_list()
            )
            for run_id in run_ids:
                status, why = call_log_status(str(shard), run_id)
                if status not in (ABSENT, COMPLETE):
                    raise ComposeRefused(f"{shard}: its call log is incomplete: {why}")


def check_column_sets(pieces: list[dict]) -> None:
    """
    refuse pieces whose shards of one cell carry different columns. A cell is a shard
    filename that occurs in two or more pieces; a shard present in only some pieces is
    legitimate (per-game pieces) and is not compared. Names only, read from each file's
    schema without loading rows; dtypes are not compared. A guard: it fills, casts and
    drops nothing
    :param pieces: loaded pieces (list of dict)
    :raises ComposeRefused: naming the shard, the pieces and the columns that differ
    """
    by_name: dict[str, list[tuple[Path, frozenset[str]]]] = {}
    for piece in pieces:
        for shard in piece["shards"]:
            columns = frozenset(pl.read_parquet_schema(shard))
            by_name.setdefault(shard.name, []).append((piece["dir"], columns))
    for name, entries in sorted(by_name.items()):
        first_dir, first_columns = entries[0]
        for other_dir, other_columns in entries[1:]:
            if other_columns != first_columns:
                raise ComposeRefused(
                    f"{name}: pieces do not share one column set. {other_dir} has "
                    f"{sorted(other_columns - first_columns)} that {first_dir} lacks, and "
                    f"{first_dir} has {sorted(first_columns - other_columns)} that "
                    f"{other_dir} lacks. Shards collected under different row schemas "
                    "must not share a composite; re-run the whole cell under one schema"
                )


def _roster_arm(
    piece_dirs: list[Path],
) -> tuple[str, list[RunKey], list[RunKey]] | None:
    """
    find the roster arm a set of piece directories belongs to. The roster is the one
    `ladder_state.py` reads (`roster.JOBS` and `MODELS`); nothing is copied. A piece
    matches a run when the run's `rel_dir` is the tail of the piece's path, compared by
    path components, under both the path as given and its symlink-resolved form
    :param piece_dirs: the directories passed as pieces (list of Path)
    :returns: None when no piece matches any declared run (the expected count cannot be
        derived), else (arm label, every run of the arm, the runs the pieces matched)
    :raises ComposeRefused: when a piece matches ambiguously, only some pieces match, the
        pieces span more than one (kind, model), a run is given twice, or the roster cannot
        be read
    """
    try:
        runs = expected_runs(indexed_kinds())
    except ValueError as exc:
        raise ComposeRefused(f"the roster could not be read: {exc}") from exc
    matched: list[RunKey | None] = []
    for piece_dir in piece_dirs:
        forms = {PurePath(os.path.abspath(piece_dir)), PurePath(piece_dir.resolve())}
        hits = {
            run
            for form in forms
            for run in runs
            if form.parts[-len(PurePath(run.rel_dir).parts) :]
            == PurePath(run.rel_dir).parts
        }
        if len(hits) > 1:
            raise ComposeRefused(
                f"{piece_dir} matches more than one roster run: {sorted(h.rel_dir for h in hits)}"
            )
        matched.append(next(iter(hits)) if hits else None)
    if all(run is None for run in matched):
        return None
    unmatched = [str(d) for d, run in zip(piece_dirs, matched, strict=True) if run is None]
    if unmatched:
        raise ComposeRefused(
            f"{unmatched} are not runs the roster declares, but other pieces are; "
            "pieces from different layouts cannot be composed as one arm"
        )
    arms = {(run.kind, run.model) for run in matched if run is not None}
    if len(arms) != 1:
        raise ComposeRefused(
            f"pieces span more than one (kind, model) arm: {sorted(arms)}"
        )
    if len(set(matched)) != len(matched):
        raise ComposeRefused("the same roster run was given as a piece more than once")
    kind, model = next(iter(arms))
    arm_runs = [run for run in runs if (run.kind, run.model) == (kind, model)]
    return f"{kind} {model}", arm_runs, [run for run in matched if run is not None]


def _run_label(run: RunKey) -> str:
    """
    name a roster run for a refusal message
    :param run: the run (RunKey)
    :returns: the index, then the game for per-game kinds (str)
    """
    return f"i{run.index}" + (f" {run.game}" if run.game else "")


def check_piece_count(
    piece_dirs: list[Path], *, expected_count: int | None, allow_partial: bool
) -> dict:
    """
    refuse a composite whose pieces are not what the roster says an arm, or one run, is, and
    record the shape that was established. The expected count is derived from the roster,
    never from the operator's number. Two shapes are complete: `arm` (every run the roster
    declares for the (kind, model)) and, for kinds collected as one directory per game,
    `index` (every game of exactly one run index, which is a whole run). Anything else is
    refused unless `allow_partial`, which composes it as `partial`
    :param piece_dirs: the directories passed as pieces (list of Path)
    :param expected_count: the operator's `--expected-count`, or None (int or None)
    :param allow_partial: compose an incomplete set anyway and say so (bool)
    :returns: the completeness record for the composite run_meta: `shape` ("arm", "index"
        or "partial"), `n_pieces`, `n_expected` (the count the shape implies, None when it
        cannot be derived) and `count_source` ("roster", "operator", or None when nothing
        established a count) (dict)
    :raises ComposeRefused: when the pieces fit neither complete shape and `allow_partial`
        is not set, when the operator's count disagrees with the shape the roster
        implies, or when no count can be derived and neither `expected_count` nor
        `allow_partial` was given
    """
    n_pieces = len(piece_dirs)
    arm = _roster_arm(piece_dirs)
    if arm is None:
        if expected_count is not None:
            return {
                "shape": SHAPE_ARM,
                "n_pieces": n_pieces,
                "n_expected": expected_count,
                "count_source": COUNT_OPERATOR,
            }
        if allow_partial:
            return {
                "shape": SHAPE_PARTIAL,
                "n_pieces": n_pieces,
                "n_expected": None,
                "count_source": None,
            }
        raise ComposeRefused(
            "the roster declares no arm these pieces belong to, so how many pieces the arm "
            "should have cannot be derived. Pass --expected-count N if the pieces were "
            "relocated, or --allow-partial to compose anyway and record that completeness "
            "was not verified"
        )
    label, arm_runs, matched = arm
    have = set(matched)
    per_game = any(run.game is not None for run in arm_runs)
    shape, n_expected = SHAPE_PARTIAL, len(arm_runs)
    if have == set(arm_runs):
        shape = SHAPE_ARM
    elif per_game and len({run.index for run in have}) == 1:
        index = next(iter(have)).index
        one_run = {run for run in arm_runs if run.index == index}
        if have == one_run:
            shape, n_expected = SHAPE_INDEX, len(one_run)
    if shape == SHAPE_PARTIAL:
        if not allow_partial:
            missing = [run for run in arm_runs if run not in have]
            shown = ", ".join(_run_label(run) for run in missing[:12])
            more = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
            wanted = (
                f"neither the whole arm ({len(arm_runs)}) nor every game of one run index"
                if per_game
                else f"not the whole arm ({len(arm_runs)})"
            )
            raise ComposeRefused(
                f"{n_pieces} pieces for {label} are {wanted}; missing {shown}{more}. A short arm scores "
                "without error and its interval widens honestly, but nothing else would say "
                "that a cell rests on fewer seeds than its neighbours. Pass --allow-partial "
                "to compose it anyway"
            )
    elif expected_count is not None and expected_count != n_expected:
        raise ComposeRefused(
            f"--expected-count {expected_count} disagrees with the roster, which implies "
            f"{n_expected} pieces for this {shape} of {label}"
        )
    return {
        "shape": shape,
        "n_pieces": n_pieces,
        "n_expected": n_expected,
        "count_source": COUNT_ROSTER,
    }


def compose(
    pieces: list[dict],
    out_dir: Path,
    *,
    expected_count: int | None = None,
    allow_partial: bool = False,
    allow_mixed_sha: bool = False,
) -> None:
    """Validate, then hard-link every piece's shards+sidecars into `out_dir`.

    Nothing is written until every check has passed -- `check_must_match` and
    every `resolve_piece_seed` call happen before the first `os.link`.
    """
    completeness = check_piece_count(
        [piece["dir"] for piece in pieces],
        expected_count=expected_count,
        allow_partial=allow_partial,
    )
    check_column_sets(pieces)
    check_call_logs(pieces)
    check_orphan_sidecars(pieces)
    check_no_interrupted_writes(pieces)
    check_classification(pieces)
    check_must_match(pieces)
    check_composite_must_match(pieces)
    check_arm_agreement(pieces, allow_mixed_sha=allow_mixed_sha)
    check_count_presence(pieces, "driven_discard_counts")
    check_composite_per_game_fields(pieces)

    seeds_by_piece = {piece["dir"]: resolve_piece_seed(piece) for piece in pieces}
    check_duplicate_seeds(pieces, seeds_by_piece)

    tags = _piece_tags(pieces, seeds_by_piece)
    ordered = sorted(
        pieces, key=lambda item: (seeds_by_piece[item["dir"]], str(item["dir"]))
    )
    piece_files = {piece["dir"]: _piece_level_files(piece) for piece in pieces}
    merged: dict[str, str] = {}
    for name, merge in PIECE_LEVEL_MERGES.items():
        holders = [
            piece
            for piece in ordered
            if any(path.name == name for path in piece_files[piece["dir"]])
        ]
        if not holders:
            continue
        lacking = [piece["dir"] for piece in ordered if piece not in holders]
        if lacking:
            raise ComposeRefused(
                f"{name} is present in {len(holders)} of {len(pieces)} pieces and missing from "
                f"{[str(d) for d in lacking]}; a composite holding it would cover a subset of "
                "the arm without saying so"
            )
        merged[name] = merge([piece["dir"] / name for piece in holders])

    planned: list[tuple[Path, Path]] = []
    for piece in ordered:
        for path in piece_files[piece["dir"]]:
            stem, ext = os.path.splitext(path.name)
            planned.append((path, out_dir / f"{stem}.{tags[piece['dir']]}{ext}"))
    for piece in pieces:
        seed = seeds_by_piece[piece["dir"]]
        for shard in piece["shards"]:
            stem, ext = os.path.splitext(shard.name)
            dest_name = f"{stem}_seed{seed}{ext}"
            planned.append((shard, out_dir / dest_name))
            for sidecar in _sidecar_paths(piece["dir"], shard):
                # Not `sidecar.name.split(".", 1)`: that splits on the
                # First dot in the whole filename, which mangles a shard
                # stem that itself contains a dot (a probe/variant arm's
                # `_arm_path_component`, e.g. `kuhn_lm_vs_bayesian.relabel_
                # v1.parquet` -- `_shard_out_path` inserts the component
                # this way). `_sidecar_paths` already guarantees `sidecar.
                # name` starts with exactly `stem + "."`, so slicing off
                # that known-length prefix is the correct, and only
                # correct, way to recover the suffix.
                sidecar_suffix = sidecar.name[len(stem) + 1 :]
                planned.append(
                    (sidecar, out_dir / f"{stem}_seed{seed}.{sidecar_suffix}")
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    linked: list[Path] = []
    src: Path | str
    try:
        for src, dst in planned:
            os.link(src, dst)
            linked.append(dst)
        for name, text in merged.items():
            src, dst = _merged_file_label(name), out_dir / name
            linked.append(dst)
            dst.write_text(text, encoding="utf-8", newline="")
    except OSError as exc:
        # Never leave a partially-linked --out behind: on the very next
        # invocation it would trip the "already exists and is not empty"
        # OPERATIONAL guard, permanently blocking retry until a human
        # deletes it by hand. Best-effort cleanup, then re-raise with the
        # failing pair named -- the caller (main) reports this OPERATIONAL,
        # not REFUSED: nothing here was a validation judgement.
        for path in linked:
            path.unlink(missing_ok=True)
        try:
            out_dir.rmdir()
        except OSError:
            pass
        raise OSError(f"linking {src} -> {dst} failed: {exc}") from exc

    _write_composite_run_meta(
        pieces, seeds_by_piece, out_dir, completeness, allow_mixed_sha=allow_mixed_sha
    )
    _write_composite_shard_outcomes(pieces, seeds_by_piece, out_dir)
    # certification: reached only after every file is written, so a failed compose seals nothing
    certified.lock_files(out_dir)
    certified.lock_dir(out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--piece",
        action="append",
        required=True,
        dest="pieces",
        type=Path,
        metavar="DIR",
        help="A certified, stamped piece arm directory. Repeatable.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="The composite arm directory to create. Must not already exist non-empty.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        metavar="N",
        help="Refuse (exit 3) unless exactly N --piece arguments were given. For pieces the "
        "roster does not declare it also stands in for the roster's count, recorded as "
        "count_source operator.",
    )
    parser.add_argument(
        MIXED_SHA_FLAG,
        action="store_true",
        dest="allow_mixed_sha",
        help="Compose pieces collected from more than one git commit anyway. The composite's "
        "git_sha is then null, git_shas lists every sha and mixed_git_sha is true.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compose even when the pieces are neither the whole arm nor every game of one "
        "run index, and record shape partial, n_pieces and n_expected in the "
        "composite's run_meta.json.",
    )
    args = parser.parse_args(argv)

    if not args.pieces:
        print("compose_arm_pieces: at least one --piece is required", file=sys.stderr)
        return EXIT_USAGE

    if args.expected_count is not None and len(args.pieces) != args.expected_count:
        print(
            f"compose_arm_pieces: REFUSED -- got {len(args.pieces)} --piece "
            f"argument(s), --expected-count {args.expected_count}",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    if args.out.exists():
        if not args.out.is_dir():
            print(
                f"compose_arm_pieces: OPERATIONAL -- {args.out} exists and "
                "is not a directory",
                file=sys.stderr,
            )
            return EXIT_OPERATIONAL
        if any(args.out.iterdir()):
            print(
                f"compose_arm_pieces: OPERATIONAL -- {args.out} already exists "
                "and is not empty",
                file=sys.stderr,
            )
            return EXIT_OPERATIONAL

    try:
        pieces = [load_piece(p) for p in args.pieces]
    except ComposeRefused as exc:
        print(f"compose_arm_pieces: REFUSED -- {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (OSError, json.JSONDecodeError) as exc:
        print(f"compose_arm_pieces: OPERATIONAL -- {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

    try:
        compose(
            pieces,
            args.out,
            expected_count=args.expected_count,
            allow_partial=args.allow_partial,
            allow_mixed_sha=args.allow_mixed_sha,
        )
    except ComposeRefused as exc:
        print(f"compose_arm_pieces: REFUSED -- {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except OSError as exc:
        print(f"compose_arm_pieces: OPERATIONAL -- {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

    print(f"compose_arm_pieces: COMPOSED {len(pieces)} piece(s) into {args.out}")
    return EXIT_COMPOSED


if __name__ == "__main__":
    sys.exit(main())
