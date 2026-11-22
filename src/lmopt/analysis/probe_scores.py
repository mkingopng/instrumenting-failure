"""
Rows of `probe_scores.csv`: one per (game, property) for the three properties whose scorers need a
probe arm (P1 relabel, P4 payoff scale, P6 off-path).

The functions here turn what the scorers returned into rows and never score anything themselves.
A row reads MEASURABLE only for a finite value backed by paired or off-path info-sets. It reads
AWAITING_DATA only when it is provably a collection gap: no rows for the game in an arm whose
configured game set is known and excludes it. Everything else with no number reads NOT_COMPUTED
with the reason. The paired scorer returns `not-scored` with its counts when both arms have rows
and nothing paired, and `awaiting-data` only when an arm has no rows it can score; a status the
scorer does not define raises here instead of being guessed at.

An off-path row carries two bases. The sampled distance is judged against the collected
sentinel anchor for its own game; the distribution distance, taken from the arm's exact-policy
pass, is judged against the analytic uniform baseline computed beside it. Neither verdict is
ever issued against the other's threshold, and a row with no exact-policy pass states that
rather than leaving the distribution columns unexplained.

A paired row (P1 or P4) carries the distribution basis too, taken from the two arms' own exact-policy
passes: the distance is the total variation between the arms' exact distributions at each paired
info-set, over the pairing, visit counts and coverage the sampled scorer computes. It carries no baseline
and no verdict on that basis, because the verdict of those two properties is the decision table's, on the
sampled scale, at the threshold that table holds; the row says so in `exact_reason`, and the board reads
that text as its own state, never as awaiting data. An arm with no pass leaves the row on its sampled
headline and names which arm it is.

`value`/`value_exact` is the mean of the per-seed values a row's own replicate interval is
formed over (the centre of its own `[value_ci_lo, value_ci_hi]`/`[ci_low_exact,
ci_high_exact]`), and `value_pooled`/`value_pooled_exact` carries the pooled distance the
scorer returns beside it; the two coincide when fewer than two seeds contribute. For P1 and P4
this means `value` is the within-seed paired estimate (base and probe compared only at
info-sets both reached in the same seed) while `value_pooled` is the cross-seed union pairing
(base and probe compared after pooling every seed's rows together first). `n_infosets` and
`paired_coverage` describe that same union pairing throughout on P1 and P4 rows; `paired_coverage`
is None on a P6 driven row, which has no base/probe pairing to describe. `n_infosets_seed_mean` and
`paired_coverage_seed_mean` are the mean, over the seeds that contributed a paired value, of
each seed's own `n_infosets_paired`/`paired_coverage` from its own within-seed pairing, so a
reader can tell which paired set a coverage number is describing.

Caveats written here contain no semicolon: the scoreboard shows a caveat up to its first one.
"""

import csv
import dataclasses
import io
from collections.abc import Mapping, Sequence

from lmopt.analysis.analysis_helpers import is_finite_number
from lmopt.analysis.arm_collapse import ConstantAction
from lmopt.analysis.scoreboard.scoreboard import (
    PAIRED_EXACT_NOTE,
    ZERO_BASELINE_REASON,
    arm_identity,
    replicate_clause,
)
from lmopt.analysis.scoreboard.scoreboard_sources import (
    SCOREBOARD_GAMES,
    STATUS_AWAITING_DATA,
    STATUS_MEASURABLE,
    STATUS_NOT_COMPUTED,
    STATUS_RECORDED_ABSENCE,
)
from lmopt.analysis.scoring_vocabulary import (
    VERDICT_CI_UNAVAILABLE,
    VERDICT_FAILS,
    VERDICT_INDETERMINATE,
    VERDICT_MEETS,
)
from lmopt.scoring.property_paired import (
    STATUS_AWAITING_DATA as SCORER_AWAITING_DATA,
    STATUS_MEASURED as SCORER_MEASURED,
    STATUS_NOT_SCORED as SCORER_NOT_SCORED,
    STATUS_RECORDED_ABSENCE as SCORER_RECORDED_ABSENCE,
)

COLUMNS = (
    "game",
    "property",
    "value",
    "value_pooled",
    "status",
    "caveat",
    "n_infosets",
    "n_infosets_seed_mean",
    "paired_coverage",
    "paired_coverage_seed_mean",
    "dropped_rate",
    "n_dropped",
    "n_total",
    "base_dropped_rate",
    "probe_dropped_rate",
    "base_dir",
    "probe_dir",
    "base_n_rows",
    "model_id",
    "memory_rung",
    "base_prompt_spec_hash",
    "probe_prompt_spec_hash",
    "cfr_iters",
    "tremble_eps",
    "value_ci_lo",
    "value_ci_hi",
    "n_replicates",
    "anchor_mean",
    "anchor_ci_lo",
    "anchor_ci_hi",
    "verdict",
    "verdict_reason",
    "value_exact",
    "value_pooled_exact",
    "ci_low_exact",
    "ci_high_exact",
    "baseline_exact",
    "verdict_exact",
    "exact_coverage",
    "exact_coverage_mass",
    "exact_reason",
    "n_replicates_distinct",
    "replicate_sd",
    "replicate_spread",
    "n_replicates_exact",
    "n_replicates_distinct_exact",
    "replicate_sd_exact",
    "replicate_spread_exact",
)
VALUE_DECIMALS = 6

#: The verdict an off-path row can carry, against the anchor measured on its own game. The
#: words are the project's own (`scoring_vocabulary.VERDICTS`), imported and not restated,
#: because two spellings of one verdict is how a figure's colour map and a table come to
#: disagree.
OFFPATH_VERDICTS = (
    VERDICT_MEETS,
    VERDICT_FAILS,
    VERDICT_INDETERMINATE,
    VERDICT_CI_UNAVAILABLE,
)
#: The sentence every non-withheld off-path row carries beside its number: the distance is
#: measured from one reference mixture, so it is not the value a model loses by playing it.
DISTANCE_NOTE = (
    "distance from the equilibrium reference's mixture, which is not the value lost "
    "against this opponent"
)
#: What a driven arm with no exact-policy pass reads on the distribution basis. It is
#: neither BROKEN nor awaiting a collection: the arm keeps its sampled headline, and the
#: pass is a separate post-hoc step that has not been run on it.
EXACT_PASS_ABSENT = "no exact-policy pass on this arm"


@dataclasses.dataclass(frozen=True)
class ArmInfo:
    """
    what the rows need to know about one arm
    :param directory: the arm directory, as its last two path components; informational only,
        never the identity `base_dir`/`probe_dir` bind on (see `arm_identity`)
    :param role: the arm's recorded role (base, relabel, payoff_scale, offpath_drive, or any
        role added later)
    :param families_present: the game families the arm has rows for
    :param configured: the game families the arm was configured for, or None when that is unknown
    :param model_id: the recorded model identity
    :param memory_rung: the recorded memory rung
    :param prompt_spec_hash: the recorded prompt spec hash
    :param n_rows: the arm's row count
    """

    directory: str
    role: str
    families_present: frozenset[str]
    configured: frozenset[str] | None
    model_id: str | None
    memory_rung: str | None
    prompt_spec_hash: str | None
    n_rows: int


def _round(x):
    """
    :param x: a number, or None
    :returns: the number rounded for the file, or None when it is missing or not finite
    """
    return round(float(x), VALUE_DECIMALS) if is_finite_number(x) else None


def replicate_dispersion(values: Sequence[float]) -> dict:
    """
    how much the per-seed values of one cell differ, from the values themselves
    Distinctness is counted on the raw floats the interval was formed over, never on the
    rounded values the file writes, so two values that differ only past the sixth decimal are
    still two.
    :param values: the finite per-seed values the interval was formed over (Sequence[float])
    :returns: {"n_distinct": how many distinct values, "spread": the largest minus the
        smallest}; both None for no values, so a cell with no per-seed data is never read as
        one repeated value
    """
    if not values:
        return {"n_distinct": None, "spread": None}
    return {"n_distinct": len(set(values)), "spread": max(values) - min(values)}


def _diagnostic_fields(stats: Mapping | None, suffix: str = "") -> dict:
    """
    :param stats: a cell's per-seed statistics with the dispersion fields, or None
    :param suffix: "" for the sampled basis, "_exact" for the distribution basis
    :returns: that basis's distinct count, sample standard deviation (n-1)
        and range as CSV columns; every one None when the cell has no statistics
    """
    stats = stats or {}
    return {
        f"n_replicates_distinct{suffix}": stats.get("n_distinct"),
        f"replicate_sd{suffix}": _round(stats.get("sd")),
        f"replicate_spread{suffix}": _round(stats.get("spread")),
    }


def _zero_variance(n, distinct) -> str:
    """
    :param n: how many per-seed values the interval was formed over, or None
    :param distinct: how many distinct raw values those were, or None
    :returns: the sentence naming a zero-width interval's reason when every seed scored one
        value, else an empty string
    """
    return replicate_clause(n, distinct, 0.0) if distinct == 1 else ""


def _constant_clause(
    constant: ConstantAction | None,
    stats: Mapping | None,
    constant_error: str | None = None,
) -> str:
    """
    :param constant: the one action the driven arm played at every off-path info-set, or None
    :param stats: the cell's per-seed statistics, or None
    :param constant_error: why the constant-action check failed, or None
    :returns: the sentence saying the check failed when it did, so a missing diagnostic is never
        read as a policy that varies; else the sentence naming the constant action, which leads
        the caveat because the scoreboard shows only a caveat's first clause and says the interval
        is zero-width for that reason when it is; an empty string when the arm played more than
        one action
    """
    if constant_error is not None:
        return f"constant-action check failed: {_first_line(constant_error)}"
    if constant is None:
        return ""
    stats = stats or {}
    lo, hi = stats.get("ci_low"), stats.get("ci_high")
    zero = is_finite_number(lo) and is_finite_number(hi) and lo == hi
    tail = (
        "so the zero-width interval is one policy repeated, not seeds agreeing"
        if zero
        else "so the interval reflects which info-sets each seed reached, not a choice"
    )
    return (
        f"the driven policy plays one constant action ({constant.words()}) at all "
        f"{constant.n_infosets} off-path info-sets, {tail}"
    )


def _flat(text) -> str:
    """
    :param text: scorer text that may contain semicolons
    :returns: the text with each semicolon written as a comma, since the scoreboard shows a caveat
        only up to its first semicolon and would otherwise lose the tail
    """
    return str(text).replace(";", ",")


def _first_line(text) -> str:
    """
    :param text: an error message
    :returns: its first line, stripped, with semicolons flattened
    """
    return _flat(str(text).strip().split("\n")[0].strip())


def _provenance(
    base: ArmInfo, probe: ArmInfo | None, cfr_iters=None, tremble_eps=None
) -> dict:
    """
    :param base: the base arm
    :param probe: the probe arm, or None
    :param cfr_iters: the reference's iteration count, for P6
    :param tremble_eps: the reference's tremble mass, for P6
    :returns: the provenance columns
    """
    return {
        "base_dir": arm_identity(base.model_id, base.memory_rung, base.role),
        "probe_dir": (
            arm_identity(probe.model_id, probe.memory_rung, probe.role)
            if probe
            else None
        ),
        "base_n_rows": base.n_rows,
        "model_id": base.model_id,
        "memory_rung": base.memory_rung,
        "base_prompt_spec_hash": base.prompt_spec_hash,
        "probe_prompt_spec_hash": probe.prompt_spec_hash if probe else None,
        "cfr_iters": cfr_iters,
        "tremble_eps": tremble_eps,
    }


def _row(game, prop_name, status, caveat="", **fields) -> dict:
    """
    :param game: the game key
    :param prop_name: the property row name
    :param status: one of the STATUS_ words
    :param caveat: a semicolon-free reason
    :param fields: any other column values
    :returns: a full row with every column present
    """
    row = dict.fromkeys(COLUMNS)
    row.update(game=game, property=prop_name, status=status, caveat=caveat)
    row.update(fields)
    return row


def _arm_rate(out: Mapping, frame_keys: Sequence[str], arm: str):
    """
    :param out: a paired scorer's output
    :param frame_keys: the frame game keys pooled into one family row
    :param arm: "base" or "probe"
    :returns: that arm's own dropped rate over those keys, or None when it had no LM row
    """
    details = [
        out["dropped_detail"][k][arm] for k in frame_keys if k in out["dropped_detail"]
    ]
    total = sum(d["n_total"] for d in details)
    return sum(d["n_dropped"] for d in details) / total if total else None


def _missing_status(
    game: str, lacking: Sequence[tuple[str, ArmInfo]]
) -> tuple[str, str]:
    """
    the status of a game with no paired info-set that some arm has no rows for
    :param game: the game family
    :param lacking: (label, arm) for each arm with no rows for the game
    :returns: (AWAITING_DATA, reason) only when every such arm's configured set is known and
        excludes the game, otherwise (NOT_COMPUTED, reason)
    :raises ValueError: when `lacking` is empty, since no arm leaves no basis for a verdict
    """
    if not lacking:
        raise ValueError(
            f"_missing_status({game!r}) was given no lacking arm; call it only for a game "
            "that some arm has no rows for"
        )
    for label, arm in lacking:
        if arm.configured is None:
            return STATUS_NOT_COMPUTED, (
                f"cannot tell a collection gap from a lost piece: the {label} arm records no "
                f"complete list of the games it was configured for"
            )
        if game in arm.configured:
            return STATUS_NOT_COMPUTED, (
                f"the {label} arm was configured for {game} but wrote no rows"
            )
    names = " and ".join(label for label, _arm in lacking)
    return STATUS_AWAITING_DATA, (
        f"no {game} rows in the {names} arm, which was not configured for it"
    )


def _paired_caveat(stats: Mapping | None) -> str:
    """
    :param stats: a paired cell's per-seed statistics, or None when no seed is shared
    :returns: why the cell's interval is absent or has no width, or an empty string when it
        has a width of its own
    """
    stats = stats or {}
    n = stats.get("n") or 0
    if n < 2:
        return (
            f"no replicate interval: {n} seed{'' if n == 1 else 's'} shared by both arms "
            f"with a paired distance"
        )
    return _zero_variance(n, stats.get("n_distinct"))


@dataclasses.dataclass(frozen=True)
class PairedExact:
    """
    the distribution-basis side of one paired (P1 or P4) cell, beside its sampled side
    :param row: the game's row of `score_paired_exact`'s `per_game_exact`, or None when the cell has
        no distribution basis
    :param stats: the per-seed replicate statistics of the exact distance, or None
    :param reason: why the cell has no distribution basis, empty when it has one
    :param coverage_mass: the mean coverage mass of the two arms' passes over the info-sets scored,
        or None when the passes record none
    """

    row: Mapping | None = None
    stats: Mapping | None = None
    reason: str = ""
    coverage_mass: float | None = None


def _paired_exact_fields(exact: PairedExact | None) -> dict:
    """
    the distribution-basis columns of one paired row
    A paired row has no baseline and no verdict on this basis: the verdict of P1 and P4 is the
    decision table's, on the sampled scale, so the reason carries `PAIRED_EXACT_NOTE` when a value is
    present and the board reads that as its own state.
    :param exact: the cell's distribution basis, or None when the caller computed none, which leaves
        every column empty as it was before this basis existed
    :returns: the exact columns; `value_exact` is the mean of the per-seed distances (the interval's
        own centre) when at least two seeds contribute one, else the pooled unweighted mean, which is
        the estimator the sampled `value` uses; `value_pooled_exact` always carries the pooled figure
    """
    if exact is None:
        return {}
    row = exact.row
    if row is None or not row["n_infosets_exact"]:
        covered = (
            f"the exact-policy passes cover none of the {row['n_infosets_paired']} paired info-sets"
            if row is not None
            else None
        )
        return dict(
            value_exact=None,
            value_pooled_exact=None,
            ci_low_exact=None,
            ci_high_exact=None,
            baseline_exact=None,
            verdict_exact="",
            exact_coverage=_round(row["exact_coverage"]) if row is not None else None,
            exact_coverage_mass=None,
            exact_reason=_flat(exact.reason or covered or EXACT_PASS_ABSENT),
            n_replicates_exact=None,
            **_diagnostic_fields(None, "_exact"),
        )
    stats = exact.stats or {}
    pooled = _round(row["tv_mean_exact"])
    has_interval = (stats.get("n") or 0) >= 2
    return dict(
        value_exact=_round(stats.get("mean")) if has_interval else pooled,
        value_pooled_exact=pooled,
        ci_low_exact=_round(stats.get("ci_low")),
        ci_high_exact=_round(stats.get("ci_high")),
        baseline_exact=None,
        verdict_exact="",
        exact_coverage=_round(row["exact_coverage"]),
        exact_coverage_mass=_round(exact.coverage_mass),
        exact_reason=PAIRED_EXACT_NOTE,
        n_replicates_exact=stats.get("n"),
        **_diagnostic_fields(exact.stats, "_exact"),
    )


def paired_rows(
    prop_name: str,
    out: Mapping,
    base: ArmInfo,
    probe: ArmInfo,
    games: Sequence[str] = SCOREBOARD_GAMES,
    role: str = "probe",
    stats_by_game: Mapping[str, Mapping] | None = None,
    exact_by_game: Mapping[str, PairedExact] | None = None,
    memory_note: Mapping[str, str] | None = None,
) -> list[dict]:
    """
    the rows of one paired property (P1 or P4) from the scorer's output
    The value is the mean of the per-seed within-seed paired distances in `stats_by_game`
    (the centre of the row's own interval) when at least two seeds contribute one; `value_pooled`
    always carries the scorer's own cross-seed union distance. A row that could not form an
    interval says why in its caveat, and its value falls back to the pooled figure.
    :param prop_name: the headline row name
    :param out: `score_property1` or `score_property4` output
    :param base: the base arm
    :param probe: the probe arm
    :param games: the games to emit, in order
    :param role: the probe arm's role, used in reasons
    :param stats_by_game: {game: per-seed statistics with the dispersion fields, plus
        `n_infosets_seed_mean` and `paired_coverage_seed_mean`} over the seeds both arms
        share, or None when none were computed
    :param exact_by_game: {game: the cell's distribution basis}, or None when no basis was computed,
        which leaves the exact columns empty on every row
    :param memory_note: {game: a sentence appended to that game's measured caveat}, or None for an
        M0 pair, where the prompt carries no memory and there is nothing to add
    :returns: one row per game
    """
    prov = _provenance(base, probe)
    per_game = {r["game"]: r for r in out["per_game"]}
    total = sum(r["n_infosets_paired"] for r in out["per_game"])

    def whole(reason):
        return [_row(g, prop_name, STATUS_NOT_COMPUTED, reason, **prov) for g in games]

    if total != out["n_infosets_paired"]:
        return whole(
            f"the scorer paired {out['n_infosets_paired']} info-sets but the per-game rows "
            f"account for {total}, so a game key outside the registry was dropped"
        )
    if not out["arms_differ"]:
        return whole(
            "the two arms carry the same variant, so the distance is an identity and not a "
            "measurement"
        )
    result = []
    for game in games:
        src = per_game.get(game)
        if src is None:
            result.append(
                _row(
                    game,
                    prop_name,
                    STATUS_NOT_COMPUTED,
                    "the scorer returned no row for the game",
                    **prov,
                )
            )
            continue
        frames = src["frame_games"]
        fields = dict(
            prov,
            n_infosets=src["n_infosets_paired"],
            paired_coverage=_round(src["paired_coverage"]),
            dropped_rate=_round(src["dropped_rate"]),
            n_dropped=src["n_dropped"],
            n_total=src["n_total"],
            base_dropped_rate=_round(_arm_rate(out, frames, "base")),
            probe_dropped_rate=_round(_arm_rate(out, frames, "probe")),
        )
        scorer_status = src["status"]
        if scorer_status == SCORER_RECORDED_ABSENCE:
            result.append(
                _row(
                    game,
                    prop_name,
                    STATUS_RECORDED_ABSENCE,
                    _flat(src["reason"] or ""),
                    **fields,
                )
            )
        elif scorer_status == SCORER_MEASURED:
            if is_finite_number(src["tv_mean"]) and src["n_infosets_paired"]:
                stats = (stats_by_game or {}).get(game)
                pooled = _round(src["tv_mean"])
                seed_mean = (stats or {}).get("mean")
                has_interval = bool(stats and (stats.get("n") or 0) >= 2)
                value = _round(seed_mean) if has_interval else pooled
                fields.update(
                    value_ci_lo=_round((stats or {}).get("ci_low")),
                    value_ci_hi=_round((stats or {}).get("ci_high")),
                    n_replicates=(stats or {}).get("n"),
                    n_infosets_seed_mean=_round(
                        (stats or {}).get("n_infosets_seed_mean")
                    ),
                    paired_coverage_seed_mean=_round(
                        (stats or {}).get("paired_coverage_seed_mean")
                    ),
                    **_diagnostic_fields(stats),
                )
                if exact_by_game is not None:
                    fields.update(
                        _paired_exact_fields(
                            exact_by_game.get(game)
                            or PairedExact(
                                reason="the scorer returned no distribution basis for the game"
                            )
                        )
                    )
                result.append(
                    _row(
                        game,
                        prop_name,
                        STATUS_MEASURABLE,
                        "; ".join(
                            part
                            for part in (
                                _paired_caveat(stats),
                                memory_note[game] if memory_note else "",
                            )
                            if part
                        ),
                        value=value,
                        value_pooled=pooled,
                        **fields,
                    )
                )
            else:
                result.append(
                    _row(
                        game,
                        prop_name,
                        STATUS_NOT_COMPUTED,
                        "the scorer reported a measurement without a finite distance over paired "
                        "info-sets",
                        **fields,
                    )
                )
        elif scorer_status == SCORER_AWAITING_DATA:
            lacking = [
                (label, arm)
                for label, arm in (("base", base), (role, probe))
                if game not in arm.families_present
            ]
            if lacking:
                status, caveat = _missing_status(game, lacking)
            else:
                status, caveat = STATUS_NOT_COMPUTED, (
                    f"an arm has rows for {game} but the scorer found no LM rows from the "
                    f"scored actor to pair on the boards the base arm played, so the cause is "
                    f"a different actor, rows that are not LM rows, or another board"
                )
            result.append(_row(game, prop_name, status, caveat, **fields))
        elif scorer_status == SCORER_NOT_SCORED:
            result.append(
                _row(
                    game,
                    prop_name,
                    STATUS_NOT_COMPUTED,
                    _flat(src["reason"] or ""),
                    **fields,
                )
            )
        else:
            raise ValueError(
                f"the paired scorer returned the unknown status {scorer_status!r} for {game}"
            )
    return result


def offpath_verdict(
    stats: Mapping | None, anchor: Mapping | str | None
) -> tuple[str, str]:
    """
    the verdict of one off-path cell against the anchor measured on its own game
    The rule is the signed one: the cell fails unless its distance is below the
    uniform-random sentinel's, interval-excluding. The LM's replicate interval must lie
    entirely below the anchor's replicate interval to pass, so an overlap reads
    indeterminate and never a pass, and a cell with no interval publishes no verdict at
    all rather than a point verdict the design cannot back.
    :param stats: the LM cell's replicate statistics, as
        `lmopt.analysis.sentinel_anchors.replicate_stats` returns them, or None when the
        cell produced no replicate at all (Mapping | None)
    :param anchor: that game's anchor, or the string reason it cannot be used, or None
        when no anchor artefact was loaded (Mapping | str | None)
    :returns: (verdict, reason); the verdict is one of `OFFPATH_VERDICTS`, or an empty
        string when no anchor stands behind the cell, in which case the reason names why
        (tuple[str, str])
    """
    if anchor is None:
        return "", "no off-path sentinel anchor artefact was loaded for this run"
    if isinstance(anchor, str):
        return "", anchor
    if not (
        is_finite_number(anchor.get("ci_low"))
        and is_finite_number(anchor.get("ci_high"))
    ):
        return "", (
            f"the anchor for this game has no replicate interval (n={anchor.get('n')}), so "
            f"the interval-excluding rule cannot be applied"
        )
    if stats is None or not (
        is_finite_number(stats.get("ci_low")) and is_finite_number(stats.get("ci_high"))
    ):
        n = (stats or {}).get("n")
        return VERDICT_CI_UNAVAILABLE, (
            f"no replicate interval is formable over this cell's seeds (n={n}), so the "
            f"point verdict is withheld rather than published without the interval check "
            f"the rule requires"
        )
    lo, hi = stats["ci_low"], stats["ci_high"]
    a_lo, a_hi = anchor["ci_low"], anchor["ci_high"]
    band = f"[{lo:.6g}, {hi:.6g}] against the anchor's [{a_lo:.6g}, {a_hi:.6g}]"
    if hi < a_lo:
        return (
            VERDICT_MEETS,
            f"the cell's replicate interval {band} lies entirely below it",
        )
    if lo >= a_hi:
        return (
            VERDICT_FAILS,
            f"the cell's replicate interval {band} lies at or above it",
        )
    return VERDICT_INDETERMINATE, (
        f"the cell's replicate interval {band} overlaps it, so the design cannot resolve "
        f"this verdict at the current replicate count"
    )


@dataclasses.dataclass(frozen=True)
class ExactBasis:
    """
    the distribution-basis side of one off-path cell, beside its sampled side
    :param out: `score_property6_exact` output for the game, or None when the cell has
        no distribution basis
    :param stats: the per-seed replicate statistics of the exact distance, or None
    :param reason: why the cell has no distribution basis, empty when it has one
    :param coverage_mass: the mean coverage mass of the exact-policy pass over the
        info-sets it scored, or None when the pass records none
    """

    out: Mapping | None = None
    stats: Mapping | None = None
    reason: str = ""
    coverage_mass: float | None = None


def exact_verdict(stats: Mapping | None, baseline) -> tuple[str, str]:
    """
    the verdict of one off-path cell's distribution basis against its analytic baseline
    The baseline is a point, not an interval: a uniform-random agent's exact policy is
    uniform over the legal actions, so its distance from the reference is computed rather
    than collected and carries no sampling error. The LM's own replicate interval still
    has to clear it, so an interval straddling the baseline reads indeterminate.
    :param stats: the cell's replicate statistics over the per-seed exact distances, as
        `lmopt.analysis.sentinel_anchors.replicate_stats` returns them, or None when the
        cell produced no replicate (Mapping | None)
    :param baseline: the analytic uniform baseline over the same info-sets and weights
    :returns: (verdict, reason); the verdict is one of `OFFPATH_VERDICTS`, or an empty
        string when no baseline stands behind the cell, in which case the reason names
        why (tuple[str, str])
    """
    if not is_finite_number(baseline):
        return "", (
            "no analytic baseline is defined for this cell, so the distribution basis "
            "has nothing to judge the distance against"
        )
    if _round(baseline) == 0.0:
        return "", ZERO_BASELINE_REASON
    if stats is None or not (
        is_finite_number(stats.get("ci_low")) and is_finite_number(stats.get("ci_high"))
    ):
        n = (stats or {}).get("n")
        return VERDICT_CI_UNAVAILABLE, (
            f"no replicate interval is formable over this cell's seeds (n={n}), so the "
            f"point verdict is withheld rather than published without the interval check "
            f"the rule requires"
        )
    lo, hi = stats["ci_low"], stats["ci_high"]
    band = f"[{lo:.6g}, {hi:.6g}] against the analytic baseline {baseline:.6g}"
    if hi < baseline:
        return (
            VERDICT_MEETS,
            f"the cell's replicate interval {band} lies entirely below it",
        )
    if lo >= baseline:
        return (
            VERDICT_FAILS,
            f"the cell's replicate interval {band} lies at or above it",
        )
    return VERDICT_INDETERMINATE, (
        f"the cell's replicate interval {band} straddles it, so the design cannot resolve "
        f"this verdict at the current replicate count"
    )


def zero_baseline(exact: ExactBasis | None) -> bool:
    """
    whether a cell's analytic uniform baseline is zero as the file records it
    :param exact: the cell's distribution basis, or None when it has none (ExactBasis | None)
    :returns: True when the basis scored at least one info-set and carries a finite baseline that
        rounds to 0.0, the exact value `baseline_exact` holds in `probe_scores.csv`; False for no
        basis, nothing scored, or no baseline (bool)
    """
    if exact is None or exact.out is None:
        return False
    out = exact.out
    scored = out["n_offpath_infosets"] > 0 and out["n_rows_scored"] > 0
    baseline = out.get("baseline")
    return scored and is_finite_number(baseline) and _round(baseline) == 0.0


def withholding_violations(rows: Sequence[Mapping]) -> list[str]:
    """
    the standing check on a written `probe_scores.csv`: an off-path cell has the zero-baseline
    reason on both bases, with no verdict beside it, if and only if its recorded baseline is
    zero and the distribution basis scored. A cell with a non-zero baseline must therefore never
    carry the reason, so a falsely withheld cell and an approved one cannot look alike. A row
    with no baseline (no distribution basis) is outside the check unless it carries the reason, and other empty verdicts
    (no anchor artefact, nothing scored) keep their own reasons
    :param rows: the rows of one file, as dicts with numbers parsed (Sequence[Mapping])
    :returns: one sentence per violation, empty when the file holds (list[str])
    """
    problems = []
    for row in rows:
        if not str(row.get("property", "")).startswith("P6"):
            continue
        baseline = row.get("baseline_exact")
        where = f"{row.get('game')} {row.get('property')}"
        if not is_finite_number(baseline):
            if ZERO_BASELINE_REASON in (
                row.get("verdict_reason"),
                row.get("exact_reason"),
            ):
                problems.append(
                    f"{where}: carries the zero-baseline reason but records no baseline"
                )
            continue
        zero = is_finite_number(row.get("value_exact")) and baseline == 0.0
        sampled = row.get("verdict_reason") == ZERO_BASELINE_REASON
        exact = row.get("exact_reason") == ZERO_BASELINE_REASON
        if sampled != exact:
            problems.append(f"{where}: only one basis carries the zero-baseline reason")
        elif (sampled and exact) != zero:
            problems.append(
                f"{where}: baseline_exact is {baseline} but the zero-baseline reason is "
                f"{'present' if sampled else 'absent'}"
            )
        if (sampled or exact) and (row.get("verdict") or row.get("verdict_exact")):
            problems.append(
                f"{where}: a verdict stands beside the zero-baseline reason"
            )
    return problems


def _exact_fields(exact: ExactBasis | None) -> dict:
    """
    the distribution-basis columns of one off-path row
    :param exact: the cell's distribution basis, or None when the caller computed none
    :returns: the seven value columns, the verdict and the reason; every value column is
        None when there is no distribution basis, and the reason then states why rather
        than leaving the absence unexplained. `value_exact` is the mean of the per-seed
        distances (the interval's own centre) when at least two seeds contribute one, else
        the pooled `tv_mean_visit_weighted`; `value_pooled_exact` always carries that pooled
        figure.
    """
    basis = exact or ExactBasis(reason=EXACT_PASS_ABSENT)
    out = basis.out
    if out is None:
        return dict(
            value_exact=None,
            value_pooled_exact=None,
            ci_low_exact=None,
            ci_high_exact=None,
            baseline_exact=None,
            verdict_exact="",
            exact_coverage=None,
            exact_coverage_mass=None,
            exact_reason=_flat(basis.reason or EXACT_PASS_ABSENT),
            n_replicates_exact=None,
            **_diagnostic_fields(None, "_exact"),
        )
    value = out["tv_mean_visit_weighted"]
    scored = out["n_offpath_infosets"] > 0 and out["n_rows_scored"] > 0
    verdict, reason = exact_verdict(basis.stats, out["baseline"])
    pooled = _round(value) if scored else None
    stats = basis.stats or {}
    has_interval = scored and (stats.get("n") or 0) >= 2
    value_exact = _round(stats.get("mean")) if has_interval else pooled
    return dict(
        value_exact=value_exact,
        value_pooled_exact=pooled,
        ci_low_exact=_round((basis.stats or {}).get("ci_low")),
        ci_high_exact=_round((basis.stats or {}).get("ci_high")),
        baseline_exact=_round(out["baseline"]),
        verdict_exact=verdict if scored else "",
        exact_coverage=_round(out["coverage"]),
        exact_coverage_mass=_round(basis.coverage_mass),
        n_replicates_exact=(basis.stats or {}).get("n"),
        **_diagnostic_fields(basis.stats, "_exact"),
        exact_reason=_flat(
            reason
            if scored
            else (
                f"the exact-policy pass covers none of the {out['n_infosets_reached']} "
                f"driven info-sets this arm reached"
            )
        ),
    )


def driven_row(
    prop_name: str,
    game: str,
    out: Mapping | None,
    error: str | None,
    arm: ArmInfo,
    base: ArmInfo,
    cfr_iters,
    tremble_eps,
    stats: Mapping | None = None,
    anchor: Mapping | str | None = None,
    exact: ExactBasis | None = None,
    constant: ConstantAction | None = None,
    constant_error: str | None = None,
    memory_note: str = "",
) -> dict:
    """
    the row of one game of P6: the sampled distance against that game's anchor, and the
    distribution distance against its analytic baseline, side by side
    :param prop_name: the headline row name
    :param game: the game key
    :param out: `score_property6` output for that game, or None when the arm has no frame for it
    :param error: the scorer's or the reference loader's error message, or None
    :param arm: the driven arm
    :param base: the base arm the CSV is bound to
    :param cfr_iters: the reference's iteration count
    :param tremble_eps: the reference's tremble mass
    :param stats: the cell's per-seed replicate statistics, or None when none were computed.
        The row's `value` is this statistic's mean (the interval's own centre) when it
        carries at least two seeds, else the scorer's pooled `tv_mean`; `value_pooled`
        always carries that pooled figure.
    :param anchor: that game's anchor, the string reason it cannot be used, or None when no
        artefact was loaded
    :param exact: the cell's distribution basis, or None when the arm has no exact-policy
        pass, in which case the row keeps its sampled headline and states the absence
    :param constant: the one action the driven arm played at every off-path info-set of this
        game, or None when it played more than one; the verdict is kept and the caveat leads
        with the action
    :param constant_error: why the constant-action check could not run, or None; the measured
        value and verdict stay and the caveat leads with the failure, since a failing diagnostic
        must not destroy a measured value
    :param memory_note: a clause appended to a measured row's caveat, "" for an M0 arm
    :returns: the row
    """
    prov = _provenance(base, arm, cfr_iters, tremble_eps)
    if error is not None:
        return _row(game, prop_name, STATUS_NOT_COMPUTED, _first_line(error), **prov)
    if out is None:
        status, caveat = _missing_status(game, [("driven", arm)])
        return _row(game, prop_name, status, caveat, **prov)
    detail = out.get("dropped_detail") or {}
    verdict, reason = offpath_verdict(stats, anchor)
    if zero_baseline(exact):
        verdict, reason = "", ZERO_BASELINE_REASON
    anchor_fields = anchor if isinstance(anchor, Mapping) else {}
    fields = dict(
        prov,
        n_infosets=out["n_offpath_infosets"],
        dropped_rate=_round(out["dropped_rate"]),
        n_dropped=detail.get("n_dropped"),
        n_total=detail.get("n_total"),
        value_ci_lo=_round((stats or {}).get("ci_low")),
        value_ci_hi=_round((stats or {}).get("ci_high")),
        n_replicates=(stats or {}).get("n"),
        **_diagnostic_fields(stats),
        anchor_mean=_round(anchor_fields.get("mean")),
        anchor_ci_lo=_round(anchor_fields.get("ci_low")),
        anchor_ci_hi=_round(anchor_fields.get("ci_high")),
        verdict=verdict,
        verdict_reason=_flat(reason),
        **_exact_fields(exact),
    )
    if (
        is_finite_number(out["tv_mean"])
        and out["n_offpath_infosets"] > 0
        and out["n_rows_scored"] > 0
    ):
        stats = stats or {}
        pooled = _round(out["tv_mean"])
        value = _round(stats.get("mean")) if (stats.get("n") or 0) >= 2 else pooled
        return _row(
            game,
            prop_name,
            STATUS_MEASURABLE,
            "; ".join(
                clause
                for clause in (
                    _constant_clause(constant, stats, constant_error),
                    _zero_variance(stats.get("n"), stats.get("n_distinct")),
                    "" if zero_baseline(exact) else DISTANCE_NOTE,
                    memory_note,
                )
                if clause
            ),
            value=value,
            value_pooled=pooled,
            **fields,
        )
    dropped = (
        f"{fields['n_dropped']} of {fields['n_total']} LM rows"
        if fields["n_dropped"] is not None and fields["n_total"] is not None
        else "dropped counts not recorded"
    )
    return _row(
        game,
        prop_name,
        STATUS_NOT_COMPUTED,
        f"no off-path info-set was scored (fallback rows dropped: {dropped})",
        **fields,
    )


def failed_rows(
    prop_name: str,
    message: str,
    base: ArmInfo,
    probe: ArmInfo,
    games: Sequence[str] = SCOREBOARD_GAMES,
) -> list[dict]:
    """
    the rows of a property whose scorer raised for the whole arm pair
    :param prop_name: the headline row name
    :param message: the scorer's error message
    :param base: the base arm
    :param probe: the probe arm
    :param games: the games to emit, in order
    :returns: one NOT_COMPUTED row per game carrying the first line of the message
    """
    prov = _provenance(base, probe)
    return [
        _row(g, prop_name, STATUS_NOT_COMPUTED, _first_line(message), **prov)
        for g in games
    ]


def unsupplied_rows(
    prop_name: str,
    role: str,
    base: ArmInfo,
    games: Sequence[str] = SCOREBOARD_GAMES,
) -> list[dict]:
    """
    the rows of a property whose probe arm was not supplied
    :param prop_name: the headline row name
    :param role: the missing arm's role
    :param base: the base arm the CSV is bound to
    :param games: the games to emit, in order
    :returns: one NOT_COMPUTED row per game
    """
    prov = _provenance(base, None)
    return [
        _row(g, prop_name, STATUS_NOT_COMPUTED, f"no {role} arm was supplied", **prov)
        for g in games
    ]


def _csv_text(value) -> str:
    """
    :param value: a column value
    :returns: its text in the file; a missing value is an empty field
    """
    if value is None:
        return ""
    return repr(value) if isinstance(value, float) else str(value)


def serialise(rows: Sequence[Mapping], property_order: Sequence[str]) -> str:
    """
    the file text for a set of rows
    :param rows: rows built by this module
    :param property_order: the property row names in the order they are written
    :returns: CSV text with LF endings, rows ordered by property then board game order
    """
    prop_index = {name: i for i, name in enumerate(property_order)}
    game_index = {game: i for i, game in enumerate(SCOREBOARD_GAMES)}
    ordered = sorted(
        rows,
        key=lambda r: (
            prop_index[r["property"]],
            game_index.get(r["game"], len(game_index)),
        ),
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in ordered:
        writer.writerow([_csv_text(row.get(c)) for c in COLUMNS])
    return buffer.getvalue()
