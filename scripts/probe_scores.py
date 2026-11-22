#!/usr/bin/env python
"""
Probe scoring: run the P1, P4 and P6 scorers on explicit arm directories and write the per-game
`probe_scores.csv` the scoreboard reads.

Usage:
    .venv/bin/python scripts/probe_scores.py --base DIR [--relabel DIR] [--payoff DIR] \
        [--driven DIR] [--null DIR] --out ANALYSIS_ARM_DIR

Each arm is verified by `arm_reading.read_arm` (run_meta, complete shard outcomes, provenance
against the rows, drive check). On top of that every probe arm must carry the role of its flag and
agree with the base arm on the project's must-match fields. Nothing is solved: the P6 reference is
loaded with `require_hit=True`, so a cold cache reads NOT_COMPUTED for that game.

Above M0 the base arm is cut to its first N hands per seed, N the hand count the paired probe arm
recorded, because a memory prompt carries the hands before it; every paired row then opens a
clause saying its distance includes the divergence of the two arms' memory, and `--null` (the
canonical config rerun over the same hands) adds that rerun's own distance as the floor of it.

Each P6 row carries two bases side by side. The sampled distance is judged against the
anchor measured on its own game, read from `configs/anchors/p6_sentinel_anchors.json`, plus
the replicate interval over the arm's seeds that the interval-excluding rule compares. A
game with no anchor, or one whose anchor this arm's provenance voids, carries no verdict and
records the reason; it never borrows another game's threshold. The distribution distance
comes from the arm's own `exact_policy.csv` and is judged against the analytic uniform
baseline computed over the same info-sets with the same weights, so nothing is collected for
it; an arm with no exact-policy pass says so and keeps its sampled headline.

Exit codes:
    0  written
    1  an arm was refused (unreadable, incomplete, wrong role, rows contradicting its meta)
    2  a usage error (argparse)
    3  the arms do not belong together, a game key is not a board game, or the base memory level
       does not satisfy the condition of a supplied probe arm's property
    4  the output file exists and --allow-overwrite was not given
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import polars as pl

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmopt.analysis import probe_scores as ps
from lmopt.analysis.arm_collapse import (
    FingerprintUnavailable,
    constant_action,
)
from lmopt.analysis.scoreboard import scoreboard as sb
from lmopt.analysis.scoreboard import scoreboard_sources as sources
from lmopt.analysis.sentinel_anchors import (
    P6_ESTIMAND,
    P6_VOIDING_PROVENANCE_FIELDS,
    anchor_for,
    basis_refusal,
    replicate_stats,
)
from lmopt.games.families import GAME_FAMILIES
from lmopt.games.loaders import load_game
from lmopt.logging.run_meta_classes import (
    ABSENT_MEANS_NONE_FIELDS,
    MISSING,
    canonical,
    get_path,
)
from lmopt.scoring import property_paired
from lmopt.scoring.property_paired import score_paired_exact
from lmopt.analysis import scoring_vocabulary as vocabulary
from lmopt.scoring.property6 import (
    DEFAULT_TREMBLE_EPS,
    offpath_reference,
    score_property6,
    score_property6_exact,
)
from scripts import compose_arm_pieces
from lmopt.analysis import arm_reading

EXIT_OK = 0
EXIT_ARM_REFUSED = 1
EXIT_MISMATCH = 3
EXIT_EXISTS = 4
OUTPUT_NAME = "probe_scores.csv"
#: The exact-policy pass's own output, written into the arm's directory by
#: `scripts/exact_policy_pass.py`. A driven arm that has one is scored on both bases; one
#: that has none keeps its sampled headline and records the absence.
EXACT_POLICY_NAME = "exact_policy.csv"

#: The one opponent the off-path run faces, and so the opponent key every P6 anchor is
#: filed under.
OFFPATH_OPPONENT = "cfr_ref"
#: The committed off-path sentinel anchor artefact, one anchor per game.
#: `--p6-sentinel-anchors` overrides it; a path that
#: does not exist means no anchor stands behind any P6 row, and every one says so rather
#: than publishing a verdict against a threshold measured somewhere else.
DEFAULT_P6_SENTINEL_ANCHORS = "configs/anchors/p6_sentinel_anchors.json"

#: Fields excluded from the base-versus-probe comparison, with the reason each is not a mismatch:
#: a probe arm hashes the variant spec it renders, and hand and seed counts may differ between a
#: full base and a one-seed probe job. Their values are recorded in the file.
NOT_COMPARED = frozenset(
    {
        "prompt_spec_hash",
        "resolved_config.n_hands",
        "resolved_config.n_seeds",
    }
)
#: Additionally not compared for the driven arm, which is a different manipulation of a different
#: subgame rather than a paired rerun.
NOT_COMPARED_FOR_DRIVEN = frozenset(
    {"resolved_config.backend", "resolved_config.seat_mode"}
)
#: Additionally not compared for the null rerun, which carries the diagnostic marker the base arm
#: does not: that marker is what keeps it off the scoreboard, and the rerun is paired with the
#: base arm all the same.
NOT_COMPARED_FOR_NULL = frozenset({"resolved_config.diagnostic"})
#: The one sentinel for an omitted key, shared with compose's must-match reader, so that an omitted key
#: and a recorded null differ in both places for the same reason.
_ABSENT = MISSING
RUNG_FIELD = "resolved_config.memory_rung"

#: The scorer for each paired role.
PAIRED_SCORERS = {
    sources.ROLE_RELABEL: property_paired.score_property1,
    sources.ROLE_PAYOFF: property_paired.score_property4,
}
#: The role a null rerun is compared under; it is read as a base arm and has no flag of its own.
NULL_ROLE = "null"
FLAGS = {
    sources.ROLE_RELABEL: "relabel",
    sources.ROLE_PAYOFF: "payoff",
    sources.ROLE_DRIVEN: "driven",
}


class Refusal(Exception):
    """An input the script will not score; carries the exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _get(meta: dict, path: str):
    """
    :param meta: a run_meta dict
    :param path: a dotted path such as `resolved_config.model`
    :returns: the value at the path, or a sentinel when any step is absent
    """
    return get_path(meta, path)


def _read_meta(path: Path):
    """
    :param path: an arm directory
    :returns: its parsed run_meta.json, or None when it cannot be read
    """
    try:
        meta = json.loads((path / "run_meta.json").read_text())
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _families(games) -> frozenset[str]:
    """
    :param games: game keys
    :returns: their families
    """
    return frozenset(GAME_FAMILIES.get(g, g) for g in games)


def _rung(value):
    """
    :param value: a recorded `memory_rung`, None, or the absent sentinel
    :returns: the rung, where a null or absent one counts as M0
    """
    return sources.M0 if value is None or value is _ABSENT else value


def _configured(meta: dict, arm) -> frozenset[str] | None:
    """
    the games an arm was configured for, or None when that cannot be known
    A composite deep-copies its first piece's run_meta and does not merge the pieces' game lists,
    so a composite's own list covers only piece 0; the union over every piece is used instead, and
    an unreadable piece or a piece with no list makes the answer unknown. Otherwise the list
    `read_arm` already parsed is used.
    :param meta: the arm's run_meta
    :param arm: the verified Arm
    :returns: the game families, or None when unknown
    """
    pieces = meta.get("composed_from")
    if pieces:
        union: set[str] = set()
        for piece in pieces:
            path = Path(piece)
            piece_meta = _read_meta(path if path.is_absolute() else REPO_ROOT / path)
            games = ((piece_meta or {}).get("resolved_config") or {}).get("games")
            if not games:
                return None
            union.update(games)
        return _families(union)
    return _families(arm.configured_games) if arm.configured_games else None


def _load(path: str, role: str, flag: str):
    """
    read and verify one arm
    :param path: the arm directory
    :param role: the role it must carry
    :param flag: the flag it was given under, for messages
    :returns: (Arm, run_meta dict, ArmInfo)
    :raises Refusal: when the arm is unreadable, has the wrong role, or plays a game outside the
        five board games
    """
    arm, problems = arm_reading.read_arm(path)
    if problems or arm is None:
        raise Refusal(
            EXIT_ARM_REFUSED, "; ".join(problems) or f"{path}: arm not readable"
        )
    if arm.role != role:
        raise Refusal(
            EXIT_ARM_REFUSED,
            f"--{flag} names a {arm.role!r} arm ({path}), expected {role!r}",
        )
    if role == sources.ROLE_DRIVEN and arm.drive_openings is None:
        raise Refusal(
            EXIT_ARM_REFUSED,
            f"--{flag} names an off-path arm ({path}) whose run_meta records no drive_openings, "
            f"so the forced-opening claim was never checked",
        )
    strangers = sorted(set(arm.games()) - set(sb.SCOREBOARD_GAMES))
    if strangers:
        raise Refusal(
            EXIT_MISMATCH,
            f"{path}: game key(s) {strangers} are not board games; depth variants are not scored",
        )
    meta = _read_meta(Path(path)) or {}
    recorded = _rung((meta.get("resolved_config") or {}).get("memory_rung", _ABSENT))
    if "memory_rung" in arm.df.columns:
        seen = set(arm.df["memory_rung"].drop_nulls().unique().to_list())
        if seen and seen != {recorded}:
            raise Refusal(
                EXIT_MISMATCH,
                f"{path}: run_meta records memory_rung {recorded!r} but its rows carry "
                f"{sorted(seen)}",
            )
    info = ps.ArmInfo(
        directory=sb.short_data_dir(os.path.abspath(path)),
        role=arm.role,
        families_present=_families(arm.games()),
        configured=_configured(meta, arm),
        model_id=meta.get("model_id"),
        memory_rung=recorded,
        prompt_spec_hash=meta.get("prompt_spec_hash"),
        n_rows=arm.n_rows,
    )
    if info.configured is not None and not info.families_present <= info.configured:
        extra = sorted(info.families_present - info.configured)
        raise Refusal(
            EXIT_MISMATCH,
            f"{path}: rows for {extra} but run_meta lists the games {sorted(info.configured)}",
        )
    return arm, meta, info


def _mismatches(base_arm, base_meta: dict, arm, meta: dict, role: str) -> list[str]:
    """
    :param base_arm: the base Arm
    :param base_meta: the base arm's run_meta
    :param arm: a probe Arm
    :param meta: that arm's run_meta
    :param role: the probe arm's role
    :returns: a message per must-match field on which the arms disagree; a field recorded in one
        and absent in the other counts
    """
    skip = set(NOT_COMPARED)
    if role == sources.ROLE_DRIVEN:
        skip |= NOT_COMPARED_FOR_DRIVEN
    if role == NULL_ROLE:
        skip |= NOT_COMPARED_FOR_NULL
    found = []
    if base_arm.actor != arm.actor:
        found.append(f"scored actor {base_arm.actor!r} vs {arm.actor!r}")
    for field in compose_arm_pieces.MUST_MATCH_FIELDS:
        if field in skip:
            continue
        a, b = _get(base_meta, field), _get(meta, field)
        if field == RUNG_FIELD:
            a, b = _rung(a), _rung(b)
        if field in ABSENT_MEANS_NONE_FIELDS:
            a = None if a is _ABSENT else a
            b = None if b is _ABSENT else b
        if canonical(a) != canonical(b):
            show = lambda v: "absent" if v is _ABSENT else repr(v)
            found.append(f"{field}: base {show(a)} vs {show(b)}")
    return found


def _run_provenance(arm, meta: dict) -> dict:
    """
    the provenance of the driven arm that an off-path anchor is checked against
    Only what the arm actually records goes in. A field absent here is a field the anchor
    lookup voids by name, which is the point: an anchor measured under conditions this run
    cannot state is an anchor for something else.
    `arm.seat_mode`/`arm.drive_openings` are shape-checked once, at `read_arm`'s own
    construction of `arm` -- the only place an `Arm` is built. This function trusts that
    guarantee rather than re-implementing its own copy of the same isinstance/coercion
    logic: passing either field through unchanged is correct for both the flat and the
    per-game shape, since `read_arm` has already refused anything else.
    :param arm: the driven Arm
    :param meta: that arm's run_meta
    :returns: the fields `sentinel_anchors.P6_VOIDING_PROVENANCE_FIELDS` names, minus any
        the arm does not record
    """
    recorded = {
        "cfr_iters": arm.cfr_iters,
        "seat_mode": arm.seat_mode,
        "drive_openings": arm.drive_openings,
        "tremble_eps": DEFAULT_TREMBLE_EPS,
        "estimand": P6_ESTIMAND,
        **{k: v for k, v in (meta.get("environment") or {}).items()},
    }
    return {k: v for k, v in recorded.items() if v is not None}


def _replicates(frame, score, value_key: str) -> dict | None:
    """
    the per-seed replicate statistics of one game's off-path distance, on either basis
    The seed is the replicate, so each seed is scored on its own and the statistics are
    taken over those values; scoring every seed's rows in one call would give an
    info-set-count-weighted average across seeds instead, with no interval at all.
    :param frame: that game's rows of the driven arm (pl.DataFrame)
    :param score: a scorer over one seed's rows, returning a P6 output dict (Callable)
    :param value_key: which distance of that output is the replicate value (str)
    :returns: `replicate_stats` output plus `n_distinct` and `spread` of the same per-seed
        values, or None when the frame records no seed
    """
    if "seed" not in frame.columns:
        return None
    values = []
    for seed in sorted(frame["seed"].drop_nulls().unique().to_list()):
        out = score(frame.filter(pl.col("seed") == seed))
        if out["n_offpath_infosets"] and math.isfinite(out[value_key]):
            values.append(out[value_key])
    return {**replicate_stats(values), **ps.replicate_dispersion(values)}


def _paired_replicates(scorer, base_df, probe_df, actor) -> dict:
    """
    the per-seed replicate statistics of a paired distance, one per game
    The seed is the replicate, so the scorer runs once per seed both arms share, pairing
    each probe seed against the same seed of the base arm; a game whose seed pairs no
    info-set contributes no value for that seed. Each seed's own `n_infosets_paired` and
    `paired_coverage` are collected alongside its distance, since a seed's own pairing (base
    and probe compared only within that seed) is a different paired set from the pooled
    call's cross-seed union pairing, and the two need their own coverage figures.
    :param scorer: `score_property1` or `score_property4` (Callable)
    :param base_df: the base arm's rows (pl.DataFrame)
    :param probe_df: the probe arm's rows (pl.DataFrame)
    :param actor: the scored actor, as the pooled call passes it (str | None)
    :returns: {game: `replicate_stats` output plus `n_distinct`, `spread`,
        `n_infosets_seed_mean` and `paired_coverage_seed_mean`, the plain means over the
        contributing seeds of each seed's own `n_infosets_paired`/`paired_coverage`}; empty
        when an arm records no seed or the two share none
    """
    if "seed" not in base_df.columns or "seed" not in probe_df.columns:
        return {}
    shared = sorted(
        set(base_df["seed"].drop_nulls().unique().to_list())
        & set(probe_df["seed"].drop_nulls().unique().to_list())
    )
    values: dict[str, list] = {}
    n_infosets: dict[str, list] = {}
    coverage: dict[str, list] = {}
    for seed in shared:
        out = scorer(
            base_df.filter(pl.col("seed") == seed),
            probe_df.filter(pl.col("seed") == seed),
            actor=actor,
        )
        for row in out["per_game"]:
            if row["n_infosets_paired"] and math.isfinite(row["tv_mean"]):
                values.setdefault(row["game"], []).append(row["tv_mean"])
                n_infosets.setdefault(row["game"], []).append(row["n_infosets_paired"])
                coverage.setdefault(row["game"], []).append(row["paired_coverage"])
    return {
        game: {
            **replicate_stats(v),
            **ps.replicate_dispersion(v),
            "n_infosets_seed_mean": sum(n_infosets[game]) / len(n_infosets[game]),
            "paired_coverage_seed_mean": sum(coverage[game]) / len(coverage[game]),
        }
        for game, v in values.items()
    }


def _exact_policy(arm, info: ps.ArmInfo):
    """
    the driven arm's own exact-policy pass, or the reason there is none to use
    The pass is a separate post-hoc step: a driven arm collected without it keeps its
    sampled headline and says so, which is a fact about this arm's collection and not a
    fault in the scoring. The loader's own cross-model refusal is passed through for the
    same reason, since a different model's distributions are not this arm's policy.
    :param arm: the driven Arm, whose directory the pass would have been written into
    :param info: its ArmInfo, read for the model the arm records
    :returns: (policy, coverage_mass, reason): the `{(game, info_set): {action: prob}}`
        dict and the `{(game, info_set): coverage mass}` map, or (None, {}, reason)
    """
    path = Path(arm.path) / EXACT_POLICY_NAME
    if not path.is_file():
        return None, {}, ps.EXACT_PASS_ABSENT
    load = vocabulary.load_exact_policy(path, run_model_id=info.model_id)
    if load.policy is None:
        return (
            None,
            {},
            (f"the exact-policy pass on this arm was refused ({load.refusal_reason})"),
        )
    frame = pl.read_csv(
        path,
        schema_overrides={"game": pl.Utf8, "info_set": pl.Utf8},
        infer_schema_length=None,
    )
    mass: dict[tuple[str, str], float] = {}
    if "coverage_mass" in frame.columns:
        for game, info_set, value in (
            frame.select("game", "info_set", "coverage_mass").unique().rows()
        ):
            if value is not None:
                mass[(game, info_set)] = float(value)
    return load.policy, mass, ""


def _paired_exact(base_arm, base_info, arm, info, base_df, probe_df, actor) -> dict:
    """
    the distribution basis of every game of one paired property, from the two arms' own passes
    Both arms need a usable pass: an arm with none leaves the property on its sampled headline, which
    each cell states, and the loader's own cross-model refusal is passed through the same way.
    :param base_arm: the base Arm, whose directory holds its pass
    :param base_info: its ArmInfo
    :param arm: the probe Arm
    :param info: its ArmInfo
    :param base_df: the base arm's rows (pl.DataFrame)
    :param probe_df: the probe arm's rows (pl.DataFrame)
    :param actor: the scored actor, as the sampled call passes it (str | None)
    :returns: {game: `ps.PairedExact`} for every registry family the scorer reports, each carrying the
        same reason when no pass can be used
    """
    # The pass keeps one prompt per info-set, which is one arm-wide policy only at M0; a paired exact
    # value above it would present one arbitrary memory-conditioned prompt as the arm's policy.
    above = [
        f"{label} arm records memory_rung {i.memory_rung!r}"
        for label, i in (("base", base_info), ("probe", info))
        if i.memory_rung not in (None, "M0")
    ]
    if above:
        reason = "the distribution basis is M0 only: " + " and ".join(above)
        games = {
            r["game"]
            for r in property_paired.score_paired(base_df, probe_df, actor=actor)[
                "per_game"
            ]
        }
        return {game: ps.PairedExact(reason=reason) for game in games}
    base_policy, base_mass, base_reason = _exact_policy(base_arm, base_info)
    probe_policy, probe_mass, probe_reason = _exact_policy(arm, info)
    reason = ""
    if base_policy is None:
        reason = f"base arm: {base_reason}"
    elif probe_policy is None:
        reason = f"probe arm: {probe_reason}"
    else:
        empty = [
            label
            for label, policy in (("base", base_policy), ("probe", probe_policy))
            if not policy
        ]
        if empty:
            reason = (
                " and ".join(f"{label} arm" for label in empty)
                + ": the exact-policy pass ran and holds no info-sets"
            )
    pooled = (
        None
        if reason
        else score_paired_exact(
            base_policy, probe_policy, base_df, probe_df, actor=actor
        )
    )
    if pooled is None:
        games = {
            r["game"]
            for r in property_paired.score_paired(base_df, probe_df, actor=actor)[
                "per_game"
            ]
        }
        return {game: ps.PairedExact(reason=reason) for game in games}
    values: dict[str, list] = {}
    if "seed" in base_df.columns and "seed" in probe_df.columns:
        shared = sorted(
            set(base_df["seed"].drop_nulls().unique().to_list())
            & set(probe_df["seed"].drop_nulls().unique().to_list())
        )
        for seed in shared:
            seed_out = score_paired_exact(
                base_policy,
                probe_policy,
                base_df.filter(pl.col("seed") == seed),
                probe_df.filter(pl.col("seed") == seed),
                actor=actor,
            )
            for row in seed_out["per_game_exact"]:
                if row["n_infosets_exact"] and math.isfinite(row["tv_mean_exact"]):
                    values.setdefault(row["game"], []).append(row["tv_mean_exact"])
    masses_by_game: dict[str, list] = {}
    for game, tvs in pooled["tv_exact_by_infoset"].items():
        for info_set in tvs:
            for table in (base_mass, probe_mass):
                if (game, info_set) in table:
                    masses_by_game.setdefault(game, []).append(table[(game, info_set)])
    result = {}
    for row in pooled["per_game_exact"]:
        frames_masses = [
            m for key in row["frame_games"] for m in masses_by_game.get(key, [])
        ]
        result[row["game"]] = ps.PairedExact(
            row=row,
            stats=(
                {
                    **replicate_stats(values[row["game"]]),
                    **ps.replicate_dispersion(values[row["game"]]),
                }
                if row["game"] in values
                else None
            ),
            coverage_mass=(
                (sum(frames_masses) / len(frames_masses)) if frames_masses else None
            ),
        )
    return result


def _exact_basis(
    frame, game: str, game_obj, reference, arm, exact_policy, mass, reason
):
    """
    the distribution basis of one game's off-path cell
    :param frame: that game's rows of the driven arm (pl.DataFrame)
    :param game: the game key (str)
    :param game_obj: the OpenSpiel game (pyspiel.Game)
    :param reference: the trembled reference from `offpath_reference`
    :param arm: the driven Arm
    :param exact_policy: the arm's exact-policy dict, or None when it has no usable pass
    :param mass: `{(game, info_set): coverage mass}` from the same pass (dict)
    :param reason: why there is no pass to use, empty when there is one (str)
    :returns: the `ps.ExactBasis` for the cell
    """
    if exact_policy is None:
        return ps.ExactBasis(reason=reason)
    out = score_property6_exact(
        exact_policy, frame, reference, game=game_obj, actor=arm.actor
    )
    covered = [(game, info_set) for info_set in out["tv_by_infoset"]]
    masses = [mass[key] for key in covered if key in mass]
    return ps.ExactBasis(
        out=out,
        stats=_replicates(
            frame,
            lambda seed_frame: score_property6_exact(
                exact_policy, seed_frame, reference, game=game_obj, actor=arm.actor
            ),
            "tv_mean_visit_weighted",
        ),
        coverage_mass=(sum(masses) / len(masses)) if masses else None,
    )


def _score_driven(
    arm,
    info: ps.ArmInfo,
    base_info: ps.ArmInfo,
    prop_name: str,
    meta: dict,
    artefact,
    memory_note: str = "",
) -> list[dict]:
    """
    score the driven arm one game at a time, on both bases
    The sampled distance is judged against that game's own collected anchor and the
    distribution distance against the analytic baseline computed beside it; the two never
    cross, so an artefact that does not declare the sampled basis withholds the sampled
    verdict alone and leaves the distribution verdict untouched.
    :param arm: the driven Arm
    :param info: its ArmInfo
    :param base_info: the base ArmInfo the file is bound to
    :param prop_name: the P6 headline row name
    :param meta: the driven arm's run_meta, read for the collecting box's library versions
    :param artefact: the off-path anchor artefact, or None when none was loaded
    :param memory_note: the clause a measured row carries about the memory its prompts held, or ""
        at M0
    :returns: one row per board game
    """
    provenance = _run_provenance(arm, meta)
    refused = None if artefact is None else basis_refusal(artefact)
    exact_policy, mass, exact_reason = _exact_policy(arm, info)
    result = []
    for game in sb.SCOREBOARD_GAMES:
        anchor: dict[str, Any] | str | None
        if artefact is None:
            anchor = None
        elif refused is not None:
            anchor = refused
        else:
            anchor = anchor_for(
                artefact,
                game,
                OFFPATH_OPPONENT,
                provenance,
                P6_VOIDING_PROVENANCE_FIELDS,
            )
        frame = arm.df.filter(pl.col("game") == game)
        if frame.is_empty():
            result.append(
                ps.driven_row(
                    prop_name,
                    game,
                    None,
                    None,
                    info,
                    base_info,
                    arm.cfr_iters,
                    DEFAULT_TREMBLE_EPS,
                )
            )
            continue
        out, error, stats, exact, constant = None, None, None, None, None
        constant_error = None
        try:
            game_obj = load_game(game)
            reference = offpath_reference(
                game_obj,
                arm.cfr_iters,
                cache_dir=arm.cfr_cache_dir,
                require_hit=True,
            )
            out = score_property6(frame, reference, game=game_obj, actor=arm.actor)
            stats = _replicates(
                frame,
                lambda seed_frame, g=game_obj, r=reference: score_property6(
                    seed_frame, r, game=g, actor=arm.actor
                ),
                "tv_mean",
            )
            exact = _exact_basis(
                frame, game, game_obj, reference, arm, exact_policy, mass, exact_reason
            )
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
        if error is None:
            try:
                constant = constant_action(frame, arm.actor)
            except FingerprintUnavailable as exc:
                constant_error = str(exc)
        result.append(
            ps.driven_row(
                prop_name,
                game,
                out,
                error,
                info,
                base_info,
                arm.cfr_iters,
                DEFAULT_TREMBLE_EPS,
                stats=stats,
                anchor=anchor,
                exact=exact,
                constant=constant,
                constant_error=constant_error,
                memory_note=memory_note,
            )
        )
    return result


def _pair_base(base_df, probe_df, n_hands: int, flag: str):
    """
    the base rows a probe arm pairs with above M0: the probe arm's own seeds, first hands only
    A memory prompt carries the hands before it, so a base arm that played more seeds and far
    more hands than its probe pairs with the probe only over the same seeds and the hands the
    probe played, never over another seed's trajectory.
    :param base_df: the base arm's rows (pl.DataFrame)
    :param probe_df: the probe arm's rows (pl.DataFrame)
    :param n_hands: how many hands per seed the probe arm recorded playing (int)
    :param flag: the flag the probe arm was given under, for messages
    :returns: the base rows of the probe's seeds whose `hand_idx` is below `n_hands` (pl.DataFrame)
    :raises Refusal: when either frame lacks `seed` or the base lacks `hand_idx`, or a probe seed
        has no base seed
    """
    for label, df in (("base", base_df), (flag, probe_df)):
        if "seed" not in df.columns:
            raise Refusal(
                EXIT_MISMATCH,
                f"the {label} arm's rows carry no seed column, so its trajectories cannot be paired",
            )
    if "hand_idx" not in base_df.columns:
        raise Refusal(
            EXIT_MISMATCH,
            "the base arm's rows carry no hand_idx column, so its first hands cannot be paired",
        )
    probe_seeds = set(probe_df["seed"].drop_nulls().unique().to_list())
    if not probe_seeds:
        raise Refusal(
            EXIT_MISMATCH,
            f"--{flag} records no seed on any row, so no base trajectory can be paired with it",
        )
    missing = sorted(probe_seeds - set(base_df["seed"].drop_nulls().unique().to_list()))
    if missing:
        raise Refusal(
            EXIT_MISMATCH,
            f"--{flag} plays seed(s) {missing} that the base arm does not, so those "
            f"trajectories have no base arm to pair with",
        )
    return base_df.filter(
        pl.col("seed").is_in(sorted(probe_seeds)) & (pl.col("hand_idx") < n_hands)
    )


def _probe_hands(meta: dict, flag: str) -> int:
    """
    how many hands per seed a probe arm recorded playing
    :param meta: the probe arm's run_meta
    :param flag: the flag the arm was given under, for messages
    :returns: the hand count (int)
    :raises Refusal: when the run_meta records no positive integer hand count
    """
    n = _get(meta, "resolved_config.n_hands")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise Refusal(
            EXIT_MISMATCH,
            f"--{flag}: run_meta records n_hands {n!r}, so the base arm's first hands cannot be "
            f"chosen",
        )
    return n


def _memory_note(n_hands: int, null_note: str) -> str:
    """
    the clause a paired row at M1 or M2 carries about what its distance measures
    :param n_hands: the hands per seed the base arm was cut to (int)
    :param null_note: the null control's words for the game, or "" when none was supplied (str)
    :returns: one clause with no semicolon, opening with the protected memory prefix (str)
    """
    return (
        f"{sources.MEMORY_NOTE_PREFIX} above M0 the distance includes the divergence of the two "
        f"arms' memory, with the base arm cut to its first {n_hands} hands per seed, "
        + (null_note or "no null control was supplied so that divergence is unmeasured")
    )


def _driven_note() -> str:
    """
    the clause an off-path row above M0 carries about the memory its prompts held
    :returns: one clause with no semicolon, opening with the protected memory prefix (str)
    """
    return (
        f"{sources.MEMORY_NOTE_PREFIX} above M0 the arm's prompts carry the hands before them, "
        f"which record the agent's own replies and not the forced openings, so the distance is "
        f"off-path play with that memory"
    )


def _null_notes(base_df, null_arm, actor, n_hands: int) -> dict:
    """
    the null control's distance per game: the canonical arm rerun, paired with the base arm's
    same seeds and first hands exactly as a probe arm is
    :param base_df: the base arm's rows, whole (pl.DataFrame)
    :param null_arm: the rerun Arm
    :param actor: the scored actor (str | None)
    :param n_hands: the hands per seed the rerun played (int)
    :returns: {game: sentence} (dict)
    """
    base_df = _pair_base(base_df, null_arm.df, n_hands, "null")
    out = property_paired.score_paired(base_df, null_arm.df, actor=actor)
    notes = {}
    for row in out["per_game"]:
        if row["n_infosets_paired"] and math.isfinite(row["tv_mean"]):
            notes[row["game"]] = (
                f"the null control, the same configuration rerun over the same {n_hands} hands, "
                f"moves it by {row['tv_mean']:.4f} over {row['n_infosets_paired']} info-sets"
            )
    return notes


def parse_args(argv=None):
    """
    :param argv: argument list, or None for sys.argv
    :returns: the parsed namespace
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, help="The base (M0) arm directory.")
    parser.add_argument("--relabel", help="The relabel arm directory (P1).")
    parser.add_argument("--payoff", help="The payoff-scale arm directory (P4).")
    parser.add_argument("--driven", help="The off-path arm directory (P6).")
    parser.add_argument(
        "--null",
        help="The canonical rerun directory at the base arm's memory level, whose distance "
        "from the base arm's first hands is printed beside every P1 and P4 row there.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="The scored arm's analysis directory; probe_scores.csv is written there.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace an existing probe_scores.csv.",
    )
    parser.add_argument(
        "--p6-sentinel-anchors",
        default=DEFAULT_P6_SENTINEL_ANCHORS,
        help="Path to the off-path sentinel anchor artefact (one "
        "uniform-random off-path distance per game). A game with no anchor "
        "there, or whose anchor this arm's own provenance voids, "
        "carries no P6 verdict and states why.",
    )
    return parser.parse_args(argv)


def _read_anchors(path: str):
    """
    the off-path anchor artefact, or None when there is none to read
    :param path: the artefact path named on the command line
    :returns: the parsed artefact, or None when the file is absent or unreadable, in which
        case every P6 row says so instead of borrowing a threshold
    """
    try:
        artefact = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return artefact if isinstance(artefact, dict) else None


def main(argv=None) -> int:
    """
    verify the arms, run the scorers and write probe_scores.csv
    :param argv: argument list, or None for sys.argv
    :returns: the exit code documented in the module docstring
    """
    args = parse_args(argv)
    try:
        heads = sb.headline_rows(vocabulary.DECISION_RULE)
        sources.check_probe_sources(vocabulary.PROPERTY_STATUS, heads)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_MISMATCH
    role_of = sources.probe_roles(heads)
    name_of = {role: name for name, role in role_of.items()}
    order = [
        name
        for number in sorted(sources.PROPERTY_SOURCES)
        for name in heads[number]
        if name in role_of
    ]
    target = Path(args.out) / OUTPUT_NAME
    if target.exists() and not args.allow_overwrite:
        print(f"{target} exists; pass --allow-overwrite to replace it", file=sys.stderr)
        return EXIT_EXISTS

    try:
        base_arm, base_meta, base_info = _load(args.base, "base", "base")
        out_meta = _read_meta(Path(args.out))
        out_data_dir = (out_meta or {}).get("data_dir")
        if out_data_dir and sb.short_data_dir(out_data_dir) != base_info.directory:
            raise Refusal(
                EXIT_MISMATCH,
                f"--out {args.out} belongs to {sb.short_data_dir(out_data_dir)!r}, not to the "
                f"base arm {base_info.directory!r}",
            )
        supplied = {}
        for role, flag in FLAGS.items():
            path = getattr(args, flag)
            if path:
                supplied[role] = _load(path, role, flag)
        problems = []
        for role, (arm, meta, _info) in supplied.items():
            problems += [
                f"--{FLAGS[role]}: {m}"
                for m in _mismatches(base_arm, base_meta, arm, meta, role)
            ]
        if problems:
            raise Refusal(
                EXIT_MISMATCH, "the arms do not belong together: " + "; ".join(problems)
            )
        for role in supplied:
            entry = next(
                e for e in sources.PROPERTY_SOURCES.values() if e.probe_role == role
            )
            if not sources.memory_level_ok(
                entry.pairable_levels, base_info.memory_rung
            ):
                raise Refusal(
                    EXIT_MISMATCH,
                    f"{name_of[role]} needs a base arm at "
                    f"{'/'.join(entry.pairable_levels)}, this one records memory_rung "
                    f"{base_info.memory_rung!r}",
                )
        above_m0 = base_info.memory_rung != sources.M0
        cut = {}
        paired_base = {}
        notes = {}
        if above_m0:
            for role in PAIRED_SCORERS:
                if role in supplied:
                    cut[role] = _probe_hands(supplied[role][1], FLAGS[role])
                    paired_base[role] = _pair_base(
                        base_arm.df, supplied[role][0].df, cut[role], FLAGS[role]
                    )
        if args.null:
            if not above_m0:
                raise Refusal(
                    EXIT_MISMATCH,
                    "--null is the floor of the memory divergence and applies above M0 only",
                )
            null_arm, null_meta, _null_info = _load(args.null, "base", "null")
            problems = _mismatches(base_arm, base_meta, null_arm, null_meta, NULL_ROLE)
            if problems:
                raise Refusal(
                    EXIT_MISMATCH,
                    "--null does not belong with the base arm: " + "; ".join(problems),
                )
            null_hands = _probe_hands(null_meta, "null")
            notes = _null_notes(base_arm.df, null_arm, base_arm.actor, null_hands)
    except Refusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return refusal.code

    rows = []
    for role, scorer in PAIRED_SCORERS.items():
        name = name_of[role]
        if role not in supplied:
            rows += ps.unsupplied_rows(name, role, base_info)
            continue
        arm, _meta, info = supplied[role]
        base_df = paired_base.get(role, base_arm.df)
        try:
            out = scorer(base_df, arm.df, actor=base_arm.actor)
            stats = _paired_replicates(scorer, base_df, arm.df, base_arm.actor)
        except ValueError as exc:
            rows += ps.failed_rows(name, str(exc), base_info, info)
            continue
        rows += ps.paired_rows(
            name,
            out,
            base_info,
            info,
            role=role,
            stats_by_game=stats,
            exact_by_game=_paired_exact(
                base_arm, base_info, arm, info, base_df, arm.df, base_arm.actor
            ),
            memory_note=(
                {
                    g: _memory_note(cut[role], notes.get(g, ""))
                    for g in sb.SCOREBOARD_GAMES
                }
                if role in cut
                else None
            ),
        )
    driven_name = name_of[sources.ROLE_DRIVEN]
    if sources.ROLE_DRIVEN in supplied:
        arm, driven_meta, info = supplied[sources.ROLE_DRIVEN]
        rows += _score_driven(
            arm,
            info,
            base_info,
            driven_name,
            driven_meta,
            _read_anchors(args.p6_sentinel_anchors),
            _driven_note() if above_m0 else "",
        )
    else:
        rows += ps.unsupplied_rows(driven_name, sources.ROLE_DRIVEN, base_info)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(ps.serialise(rows, order).encode("utf-8"))
    os.replace(tmp, target)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(
        f"wrote {target}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
