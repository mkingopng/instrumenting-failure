"""Property 5: Calibration - ECE between stated LM beliefs and true posteriors.

ECE (Expected Calibration Error) measures how well the LM's stated card
probabilities match the actual Bayesian posteriors. Score of 0 = perfect
calibration; higher = overconfident or underconfident beliefs.
"""

from __future__ import annotations

import json
import math

import polars as pl

from lmopt.games.families import GAME_FAMILIES
from lmopt.scoring._actor import (
    LEGACY_BELIEF,
    actor_label,
    actor_rows,
    belief_drop_report,
    refuse_foreign_probe_type,
    require_actor_column,
)
from lmopt.scoring._belief_alignment import (
    assert_true_support_within_stated,
)
from lmopt.scoring.darkhex_belief import (
    aligned_cells,
    opponent_stone_count,
    stone_count_cell_probability,
    uninformed_cell_probability,
)
from lmopt.scoring.leduc_rank import leduc_num_ranks, marginalise_to_rank

#: The columns the Dark Hex stone-count baseline needs beyond the two belief
#: columns. A frame without them still scores the family's headline margin; the
#: stricter baseline is omitted with a recorded reason rather than approximated.
_DARKHEX_STONE_BASELINE_COLUMNS = ("info_set", "player_to_act")


def _mean_abs(flat_errors: list[float]) -> float:
    """Mean of a flat list of values, or NaN if empty.

    Despite the name (both `calibration_error` below
    and `score_property5`'s per-game `per_pair_error` computation call it
    on a list of already-absolute-valued |stated - true| errors, so every
    element is >= 0 at those call sites), this function itself does not
    take an `abs()` of anything - it is a plain "sum / count, or NaN if
    empty" mean. That genericity is load-bearing, not incidental:
    `global_uniform_baseline_per_pair`/`global_per_pair_margin`
    reuse it on a list of per-game margins (`per_pair_error -
    uniform_baseline_per_pair`), which are signed (negative when the LM
    beats the uninformed baseline) - passing a negative value here is
    correct, not a misuse of an "absolute errors" helper despite the name.

    The single definition of "the per-pair calibration metric" - both
    `calibration_error` (below) and `score_property5`'s per-game loop call
    this for their (non-negative) use, so there is exactly one place that
    averaging happens, and `score_property5` reuses the scalar (s, t) pairs
    its loop already computed rather than retaining a second parallel list of
    parsed (stated_dict, true_dict) row pairs.
    """
    if not flat_errors:
        return float("nan")
    return sum(flat_errors) / len(flat_errors)


def calibration_error(pairs: list[tuple[dict[str, float], dict[str, float]]]) -> float:
    """
    Per-pair calibration error: mean absolute |stated - true| computed
    over every (card, row) pair before any averaging.

    Contrast with `score_property5`'s bin-based ECE, which bins by stated
    probability then compares bin-mean stated to bin-mean true and takes
    the absolute value only after that averaging (see the bin loop in that
    function's body). That order cancels whenever a bin's signed errors
    happen to sum to (near) zero. The textbook case is a uniform stated
    belief (bin-mean true collapses to exactly the value the bin was keyed
    on: liarsdice ECE 0.0 on n=50,532 despite per-pair mean |error| 0.1597).
    But it is not the only route:
    the landed Kuhn arm has zero uniform-stated rows, yet 4,045 of its
    24,045 calibration rows split into equal counts of over- and
    under-estimates whose signed errors sum to 0.000000000, so its ECE is
    also 0.0 by the same bin-then-average-then-abs mechanism (see
    tests/test_scorer_property5.py::
    TestPerPairCalibrationErrorProductionPath::
    test_kuhn_arm_level_ece_stays_zero_but_per_pair_error_reveals_binning_cancellation).
    Differencing per pair, before any averaging, cannot cancel either way.

    Follows the same zero-pad-to-`stated` convention as `score_property5`'s
    main loop (see that function's per-game loop comment): only `stated`'s
    keys are iterated, so a `true` key absent from `stated` is silently
    invisible here too. Callers are responsible for any family-specific
    alignment (leduc rank marginalisation, the goofspiel narrowing guard,
    `_belief_alignment.assert_true_support_within_stated`) before calling
    this function - `score_property5`'s per-game loop does that alignment
    first and passes already-aligned (stated, true) pairs in.
    :param pairs: list of (stated_belief, true_posterior) dicts, one pair per
        decision-node row, both keyed by the same card/rank identifiers.
    :returns: Mean absolute per-(card, row) error, or NaN if `pairs` flattens to
        no card-level entries at all (e.g. an empty list, or every stated
        dict being empty).
    """
    flat_errors: list[float] = []
    for stated, true in pairs:
        for card_key in stated:
            s = float(stated[card_key])
            t = float(true.get(card_key, 0.0))
            flat_errors.append(abs(s - t))
    return _mean_abs(flat_errors)


def _belief_dropped(
    df: pl.DataFrame, actor: str | None
) -> tuple[dict[str, float | None], dict[str, dict]]:
    """
    the belief-side dropped rate and its counts for every scored game
    The population is the scored actor's own LM rows, so a game whose belief
    attempts all failed still reports a rate although it never reaches the
    calibration loop. Every family with a belief row carries an entry, Dark Hex
    included: its belief is scored per cell rather than over a simplex, but the
    rows still succeed or fail in the same way.
    :param df: full match DataFrame (pl.DataFrame)
    :param actor: the `actor` column value being scored, or None for the legacy population (str | None)
        :returns: (dropped_rate, dropped_detail), each keyed by game; the rate is None where no belief was attempted
    """
    legacy = pl.col("prompt").is_not_null() if "prompt" in df.columns else pl.lit(True)
    lm_rows = actor_rows(df, actor, legacy=legacy)
    games = sorted(set(lm_rows["game"].drop_nulls().to_list()))
    reports = {
        g: belief_drop_report(lm_rows.filter(pl.col("game") == g)) for g in games
    }
    return (
        {g: r.rate for g, r in reports.items()},
        {g: r.detail() for g, r in reports.items()},
    )


def score_property5(
    df: pl.DataFrame, n_bins: int = 10, actor: str | None = None
) -> dict:
    """
    compute ECE between an agent's elicited_belief and the true_posterior

    With `actor=None`, only rows where both belief columns are non-null
    contribute. That filter excludes non-LM rows only incidentally: no
    principled opponent elicits a belief, so their `elicited_belief` is
    null. It is not an actor filter, and a belief-carrying opponent would be
    pooled into the LM's ECE. Naming an `actor` makes the exclusion stated
    instead of incidental.

    The action call's `fallback_used` is never read here: a belief is the
    model's own separate output, so a row whose action fell back keeps its valid
    belief in the estimate. The dropped rate counts the belief-side rows that
    were already excluded (null or unparseable beliefs).
    :param df: full match DataFrame (pl.DataFrame)
    :param n_bins: number of equal-width probability bins for the reliability diagram (int)
    :param actor: the `actor` column value to score, or `None` for the legacy
        both-belief-columns population, the only one a legacy arm can express
        since it has no `actor` column (str | None)
    :returns: dict with keys
        "ece": {game: float} - bin-based ECE per game (retained: the
            reliability-diagram bins in "bins" are consumed downstream).
        "global_ece": float - mean bin-based ECE over all games.
        "bins": {game: list[dict]} - per-bin data for reliability diagrams.
        "n_pairs": {game: int} - number of (card, row) pairs that contributed.
        "per_pair_error": {game: float} - mean absolute |stated - true|
            computed per (card, row) pair before any binning/averaging (see
            `calibration_error`). Reported alongside "ece", not instead of
            it: the two can diverge sharply when a bin's signed errors
            cancel - kuhn is 0.0 / 0.056074, see
            `calibration_error`'s docstring.
        "global_per_pair_error": float - mean per_pair_error over all games,
            mirroring "global_ece".
        "uniform_baseline_per_pair": {game: float} - the same per_pair_error
            metric, recomputed with each row's `elicited_belief` replaced by
            the uninformed stater's belief, holding `true_posterior` fixed.
            On a simplex family that is the uniform distribution over the
            row's own stated keys (`u = 1.0 / len(stated)`); on the Dark Hex
            family, whose belief is one independent Bernoulli per cell rather
            than a simplex, it is 0.5 at every cell, since a cell is occupied
            or it is not and there is no key count to divide by. This
            is what the pre-registered "beat-uninformed" P5 rule
            compares the LM against: a
            per-pair-error mean that depends on the LM only through which
            keys it was asked about, never through its stated values.
        "global_uniform_baseline_per_pair": float - mean of
            "uniform_baseline_per_pair" over all games, mirroring
            "global_per_pair_error".
        "per_pair_margin": {game: float} - `per_pair_error[game] -
            uniform_baseline_per_pair[game]`. Positive means the LM is
            worse than an uninformed uniform stater; the pre-registered rule
            is `margin > 0.0 => FAIL`.
        "global_per_pair_margin": float - mean of "per_pair_margin" over all
            games (equivalently, and confirmed to agree to every printed
            digit on the landed arm: "global_per_pair_error" -
            "global_uniform_baseline_per_pair", since both per-game dicts
            share an identical non-NaN game set by construction - see the
            loop comment below).

            The trap this guards against: the baseline is
            computed here, inside this per-game loop, from the same
            already-aligned `(stated, true)`-producing pass the observed
            metric uses - never from raw `true_posterior` keys re-parsed
            separately. Leduc's true_posterior is rank-marginalised to 3
            keys before comparison (`marginalise_to_rank`, see the loop
            below) and goofspiel's stated support is zero-padded over the
            full deck; a baseline computed from raw posterior keys silently
            uses a different key space than the observed metric and produces
            a wrong number that looks plausible: raw posterior keys give
            goofspiel 0.289170 / leduc 0.117171 against the correct (this
            implementation's) 0.304632 / 0.172784 - wrong by 5 and 6
            percentage points. Do not "simplify" this to a separate pass
            over raw keys.
        "darkhex_stone_baseline_per_pair": {game: float} - the Dark Hex
            family's second, stricter baseline, never the headline: the same
            per_pair_error metric against a stater who knows only how many
            stones the opponent has placed and spreads them evenly over the
            unknown cells (`darkhex_belief.stone_count_cell_probability`).
            Absent for every other family, and absent for a Dark Hex game
            whose frame lacks `info_set`/`player_to_act` (the stone count
            cannot be derived without them, and an approximation would be a
            silent fallback); the reason is then recorded in
            "darkhex_stone_baseline_unavailable".
        "darkhex_stone_baseline_margin": {game: float} -
            `per_pair_error[game] - darkhex_stone_baseline_per_pair[game]`.
            Reported beside the stricter baseline so it can be read the same
            way "per_pair_margin" is read, and never in its place.
        "darkhex_stone_baseline_unavailable": {game: str} - why the stricter
            baseline is missing for a Dark Hex game. Deliberately its own key
            and not "not_applicable": the headline margin for that game is
            still published, so a consumer reading "not_applicable" as "this
            game was withheld" must not meet this reason there.
        "key_mismatch_rows": {game: {"n_rows": int, "reason": str}} - Dark Hex
            rows whose stated cell set and true-posterior cell set differ.
            They are scored, over the intersection, and counted here so the
            partial comparison is visible. A game with no mismatch is ABSENT,
            matching the absence convention the other reason dicts use.
        "unscorable_rows": {game: {"n_rows": int, "reason": str}} - Dark Hex
            rows that entered the loop and reached no pair at all, because the
            stated and true cell sets do not intersect. Counted, never
            defaulted, imputed or silently dropped (no silent
            fallbacks). A game with none is ABSENT.
        "not_applicable": {game: str} - a family or a game excluded from "ece"
            and "per_pair_error" together, with an explicit recorded reason.
            Mirrors the shape of `score_property7`'s own `not_applicable`
            docstring entry. The goofspiel support violation is the only
            producer; the Dark Hex family is scored with its own per-cell
            estimand (see "uniform_baseline_per_pair" and
            "darkhex_stone_baseline_per_pair").
        "actor_filter": str - which population produced every number above,
            `"actor=<name>"` or `"legacy-belief-columns"`. A filter that is
            not reported is a silent fallback.
        "dropped_rate": {game: float | None} - the belief-side rate: LM rows
            whose belief attempt produced no usable belief (`belief_failed`,
            null `elicited_belief`, or JSON that failed to parse in the pair
            loop) over LM rows where a belief was attempted. `None` for a game
            with no belief attempts. Every family with a belief row has an
            entry, Dark Hex included.
        "dropped_detail": {game: dict} - the counts behind each rate
            (`DropReport.detail()`).
    """
    not_applicable: dict[str, str] = {}
    stone_baseline_per_game: dict[str, float] = {}
    stone_margin_per_game: dict[str, float] = {}
    stone_unavailable: dict[str, str] = {}
    key_mismatch_rows: dict[str, dict] = {}
    unscorable_rows: dict[str, dict] = {}

    require_actor_column(df, actor, "score_property5")
    refuse_foreign_probe_type(df, "score_property5")
    label = actor_label(actor, LEGACY_BELIEF)
    calibration_rows = actor_rows(
        df,
        actor,
        legacy=pl.col("elicited_belief").is_not_null()
        & pl.col("true_posterior").is_not_null(),
    )
    if actor is not None:
        # A named actor selects whose rows; the pair still needs both halves to
        # exist, so keep the null guard as a conjunct rather than replacing it.
        calibration_rows = calibration_rows.filter(
            pl.col("elicited_belief").is_not_null()
            & pl.col("true_posterior").is_not_null()
        )
    if calibration_rows.is_empty():
        empty_rate, empty_detail = _belief_dropped(df, actor)
        return {
            "ece": {},
            "global_ece": float("nan"),
            "bins": {},
            "n_pairs": {},
            "per_pair_error": {},
            "global_per_pair_error": float("nan"),
            "uniform_baseline_per_pair": {},
            "global_uniform_baseline_per_pair": float("nan"),
            "per_pair_margin": {},
            "global_per_pair_margin": float("nan"),
            "darkhex_stone_baseline_per_pair": {},
            "darkhex_stone_baseline_margin": {},
            "darkhex_stone_baseline_unavailable": {},
            "key_mismatch_rows": {},
            "unscorable_rows": {},
            "not_applicable": not_applicable,
            "actor_filter": label,
            "dropped_rate": empty_rate,
            "dropped_detail": empty_detail,
        }

    ece_per_game: dict[str, float] = {}
    bins_per_game: dict[str, list[dict]] = {}
    n_pairs: dict[str, int] = {}
    per_pair_error_per_game: dict[str, float] = {}
    uniform_baseline_per_pair_per_game: dict[str, float] = {}
    per_pair_margin_per_game: dict[str, float] = {}

    for game_name in calibration_rows["game"].unique().to_list():
        game_df = calibration_rows.filter(pl.col("game") == game_name)
        # .get with fallback-to-self, matching property7.py: offline scorers
        # must tolerate foreign game values from synthetic fixtures.
        family = GAME_FAMILIES.get(game_name, game_name)

        # Dark Hex belief is per-cell occupancy (one independent Bernoulli per
        # unseen cell), so the observed quantity is the mean per-cell absolute
        # error and the uninformed stater says 0.5 at every cell rather than
        # `1 / len(stated)`. Everything else -- the pair collection, the
        # binning, the margin, the global aggregation -- is the shared path,
        # so the family enters "ece"/"per_pair_error"/"n_pairs"/"bins"
        # together like any other.
        darkhex_stone_ok = family == "darkhex" and all(
            col in game_df.columns for col in _DARKHEX_STONE_BASELINE_COLUMNS
        )
        if family == "darkhex" and not darkhex_stone_ok:
            stone_unavailable[game_name] = (
                "the opponent's placed-stone count is read off the observer's "
                f"own view, which needs columns "
                f"{list(_DARKHEX_STONE_BASELINE_COLUMNS)}; this frame carries "
                f"{sorted(game_df.columns)}"
            )
        stone_pairs: list[tuple[float, float]] = []
        n_key_mismatch = 0
        n_unscorable = 0
        # Computed once per game_name (never
        # per-row), since it requires constructing a pyspiel.Game - real
        # rank count for "leduc"-family rows, so leduc_deep's own numRanks
        # (not the canonical 3) drives the marginalisation below. Native
        # "leduc" has no numRanks parameter, so leduc_num_ranks falls back
        # to 3 there.
        leduc_n_ranks = leduc_num_ranks(game_name) if family == "leduc" else None

        # Collect all (stated_prob, true_prob) pairs across all cards and
        # rows. `calibration_error` is not called here: it would rebuild
        # and re-flatten the same scalar pairs a second time from a
        # parallel (stated_dict, true_dict) list, which is bit-identical to
        # reusing `pairs` directly below and retains ~200k parsed dicts
        # across the arm for no behavioural difference. Both this loop and `calibration_error` route through the one
        # shared `_mean_abs`, so there is still exactly one definition of
        # the metric.
        #
        # This loop's parse/align/pair shape duplicates
        # `_belief_alignment.game_row_tvds` (used by `property7.py`) rather
        # than calling it -- deliberate, not accidental. `game_row_tvds`
        # returns one TVD scalar per row (`{"info_set": str, "tvd": float}`);
        # this loop needs the raw `(stated_value, true_value)` scalar pairs
        # per card, for ECE binning -- a different output shape, not a
        # superset/subset of `game_row_tvds`'s. Reshaping `game_row_tvds` to
        # also emit per-card pairs (or wrapping it and re-deriving pairs from
        # its already-aggregated TVD) would touch a helper `property7.py`
        # depends on, on scoring-critical code. What could be
        # shared without a shape mismatch is only the family-dispatch
        # alignment sub-step (leduc marginalise / true-support guard) both
        # loops repeat, so this duplication is confined to what genuinely
        # cannot be shared.
        pairs: list[tuple[float, float]] = []
        # Per-game accumulators for the goofspiel
        # support guard below, mirroring property7.py's game_row_tvds -- a
        # violating row is counted, not raised immediately, so the whole
        # game (not just the first bad row) can be discarded loudly via
        # not_applicable after the full per-row scan, instead of a bare
        # ValueError aborting score_property5 for every other game too.
        n_goofspiel_violations = 0
        goofspiel_offending_keys: set = set()
        # Baseline pairs for the "beat-uninformed" P5 rule:
        # `(u, t)` where `u = 1.0 / len(stated)` (the row's own elicited key
        # count) and `t` is the same already-aligned true value `pairs`
        # uses. Built inside this exact loop, from the exact same iteration
        # over `stated`'s keys, so `baseline_pairs` and `pairs` stay
        # index-aligned in length by construction - never re-derived from a
        # separately re-parsed key space (see the module docstring's "THE
        # TRAP" note). A row whose `stated` dict is empty contributes no
        # pairs to either list (`for card_key in stated` never executes),
        # so the two lists cannot diverge in length.
        baseline_pairs: list[tuple[float, float]] = []
        for row in game_df.iter_rows(named=True):
            try:
                stated = json.loads(row["elicited_belief"])
                true = json.loads(row["true_posterior"])
            except (json.JSONDecodeError, TypeError):
                continue
            if family == "leduc":
                # Leduc's true_posterior is logged over 6 card indices, but
                # the stated belief is a 3-key rank simplex. Iterating only
                # the stated keys against the raw 6-key posterior silently
                # discards 55.48% of posterior mass per row (measured on
                # data/run_experiment/qwen2.5-14b-instruct) -
                # marginalise the posterior to rank space first so both
                # sides share the same keys. See leduc_rank.py.
                assert leduc_n_ranks is not None  # set whenever family == "leduc"
                true = marginalise_to_rank(true, n_ranks=leduc_n_ranks)
            elif family == "goofspiel":
                # Guards against a silent mass drop on this path. Mirrors
                # property7.py's goofspiel guard exactly: both call the one
                # shared `_belief_alignment` helper.
                # 0 of 899,894 real goofspiel calibration rows (180 files)
                # trip this; it guards against future elicitation
                # narrowing, not a live bug. That stops being true the moment
                # goofspiel.py's parse_belief stops padding densely over the
                # full deck - re-check this figure if that ever changes.
                try:
                    assert_true_support_within_stated(
                        stated, true, scorer_name="score_property5"
                    )
                except ValueError:
                    # Counted and skipped, not raised: the rest of this
                    # game's rows are still scanned so the count and key
                    # union below are exact.
                    n_goofspiel_violations += 1
                    goofspiel_offending_keys |= set(true.keys()) - set(stated.keys())
                    continue
            # No `assert_true_support_within_stated` guard for kuhn/liarsdice
            # (structurally unfireable on landed data: 0/378,498 kuhn rows and
            # 0/458,240 liarsdice rows): an ungated scorability failure on a
            # non-goofspiel family is deliberately published as census, not
            # dropped to `not_applicable` (`scripts/analyse_stats.py`,
            # `test_i3_an_ungated_scorability_failure_is_published_as_
            # census_not_dropped`) -- and dropping it is exactly what adding
            # the guard would do.
            # For goofspiel this loop is the
            # zero-padding-to-stated convention `property7.py`'s goofspiel
            # branch states explicitly (stated is dense over the full deck
            # by construction; true's narrower support is legitimate
            # deduction, not an instrument gap - see that block's comment
            # and tests/test_goofspiel_posterior.py::
            # test_missing_posterior_keys_are_deducibly_unplayable_by_p1).
            # Iterating only `stated`'s keys already achieves this. The guard
            # above keeps this scorer symmetric with `property7.py`: an
            # out-of-support `true` key is caught here too, instead of being
            # silently invisible.
            # `u` is this ROW's uniform-stater value, the
            # same value for every card_key in this row, since it depends
            # only on `len(stated)`, never on stated's actual probabilities.
            # Computed once per row, outside the card loop, purely as a
            # micro-optimisation; it is never read unless the loop below
            # executes at least once (an empty `stated` skips both).
            if family == "darkhex":
                # Scored over the intersection of the stated and true cell
                # sets: a cell only one side names is not a disagreement to
                # measure, it is a row the two sides do not describe the same
                # way, and zero-padding either side would invent a belief or a
                # posterior. The partial comparison is counted, and a row with
                # no shared cell at all is counted and left unscored rather
                # than defaulted (no silent fallbacks).
                cells = aligned_cells(stated, true)
                if set(stated) != set(true):
                    n_key_mismatch += 1
                if not cells:
                    n_unscorable += 1
                    continue
                u = uninformed_cell_probability()
                v = (
                    stone_count_cell_probability(
                        opponent_stone_count(
                            row["info_set"], int(row["player_to_act"])
                        ),
                        len(cells),
                    )
                    if darkhex_stone_ok
                    else None
                )
                for cell_key in cells:
                    s = float(stated[cell_key])
                    t = float(true[cell_key])
                    pairs.append((s, t))
                    baseline_pairs.append((u, t))
                    if v is not None:
                        stone_pairs.append((v, t))
                continue
            u = 1.0 / len(stated) if stated else 0.0
            for card_key in stated:
                s = float(stated[card_key])
                t = float(true.get(card_key, 0.0))
                pairs.append((s, t))
                baseline_pairs.append((u, t))

        if n_goofspiel_violations:
            # Whole game discarded, not partially scored from
            # whatever clean rows survived -- mirrors property7.py's
            # GoofspielSupportViolation degrade exactly. `pairs`/
            # `baseline_pairs` collected from this game's clean rows (if
            # any) are dropped along with it.
            #
            # This is wasted work (rows scanned before the violating row are
            # thrown away), accepted rather than restructured to
            # short-circuit, because it deliberately mirrors
            # `_belief_alignment.game_row_tvds`'s own documented
            # whole-game-discard design (`_belief_alignment.py`'s
            # `GoofspielSupportViolation` docstring) and its cost is bounded
            # by the guard's own measured rate: 0 of 899,894 goofspiel rows
            # trip it.
            not_applicable[game_name] = (
                f"goofspiel: true_posterior key(s) "
                f"{sorted(goofspiel_offending_keys)} absent from elicited "
                f"belief on {n_goofspiel_violations} row(s) - investigate "
                "before trusting this game's P5"
            )
            continue

        if n_key_mismatch:
            key_mismatch_rows[game_name] = {
                "n_rows": n_key_mismatch,
                "reason": (
                    "the elicited belief and the true posterior name different "
                    "cells; these rows are scored over the cells both name, "
                    "never over a zero-padded union"
                ),
            }
        if n_unscorable:
            unscorable_rows[game_name] = {
                "n_rows": n_unscorable,
                "reason": (
                    "the elicited belief and the true posterior share no cell, "
                    "so there is nothing to compare; counted and left unscored "
                    "rather than defaulted or imputed"
                ),
            }

        if not pairs:
            ece_per_game[game_name] = float("nan")
            bins_per_game[game_name] = []
            n_pairs[game_name] = 0
            per_pair_error_per_game[game_name] = float("nan")
            uniform_baseline_per_pair_per_game[game_name] = float("nan")
            per_pair_margin_per_game[game_name] = float("nan")
            continue

        n_pairs[game_name] = len(pairs)
        per_pair_error_per_game[game_name] = _mean_abs([abs(s - t) for s, t in pairs])
        # Reuses `_mean_abs` - the one shared averaging definition - rather
        # than writing a second one; `baseline_pairs` is index-aligned with
        # `pairs` by construction (built in the same loop above), so this is
        # the per-pair-error metric computed against the uniform baseline
        # instead of the LM's stated belief, nothing else different.
        uniform_baseline_per_pair_per_game[game_name] = _mean_abs(
            [abs(u - t) for u, t in baseline_pairs]
        )
        per_pair_margin_per_game[game_name] = (
            per_pair_error_per_game[game_name]
            - uniform_baseline_per_pair_per_game[game_name]
        )
        if stone_pairs:
            # Built in the same row loop as `pairs`, over the same aligned
            # cells, for the same reason the uniform baseline is: a baseline
            # computed from a separately re-derived key space is a different
            # number that looks plausible.
            stone_baseline_per_game[game_name] = _mean_abs(
                [abs(v - t) for v, t in stone_pairs]
            )
            stone_margin_per_game[game_name] = (
                per_pair_error_per_game[game_name] - stone_baseline_per_game[game_name]
            )

        # Bin by stated probability.
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        bin_stated: list[list[float]] = [[] for _ in range(n_bins)]
        bin_true: list[list[float]] = [[] for _ in range(n_bins)]

        for s, t in pairs:
            if not 0.0 <= s <= 1.0:
                raise ValueError(
                    f"score_property5: {game_name!r} has a stated probability {s!r} outside "
                    f"[0, 1]; a value there has no calibration bin, so fix the belief parse "
                    f"that admitted it"
                )
            b = min(int(s * n_bins), n_bins - 1)
            bin_stated[b].append(s)
            bin_true[b].append(t)

        ece = 0.0
        total = len(pairs)
        bin_data = []
        for b in range(n_bins):
            count = len(bin_stated[b])
            if count == 0:
                bin_data.append(
                    {
                        "bin_lower": bin_edges[b],
                        "bin_upper": bin_edges[b + 1],
                        "mean_stated": float("nan"),
                        "mean_true": float("nan"),
                        "count": 0,
                    }
                )
                continue
            mean_s = sum(bin_stated[b]) / count
            mean_t = sum(bin_true[b]) / count
            ece += (count / total) * abs(mean_s - mean_t)
            bin_data.append(
                {
                    "bin_lower": bin_edges[b],
                    "bin_upper": bin_edges[b + 1],
                    "mean_stated": mean_s,
                    "mean_true": mean_t,
                    "count": count,
                }
            )

        ece_per_game[game_name] = ece
        bins_per_game[game_name] = bin_data

    valid = [v for v in ece_per_game.values() if not math.isnan(v)]
    global_ece = sum(valid) / len(valid) if valid else float("nan")

    valid_pp = [v for v in per_pair_error_per_game.values() if not math.isnan(v)]
    global_per_pair_error = sum(valid_pp) / len(valid_pp) if valid_pp else float("nan")

    # Reuses `_mean_abs` for the global aggregation too; see
    # its docstring for why calling it on signed margins is correct, not a
    # misuse of an "absolute errors" helper.
    valid_ub = [
        v for v in uniform_baseline_per_pair_per_game.values() if not math.isnan(v)
    ]
    global_uniform_baseline_per_pair = _mean_abs(valid_ub)

    valid_margin = [v for v in per_pair_margin_per_game.values() if not math.isnan(v)]
    global_per_pair_margin = _mean_abs(valid_margin)

    dropped_rate, dropped_detail = _belief_dropped(df, actor)
    return {
        "ece": ece_per_game,
        "global_ece": global_ece,
        "bins": bins_per_game,
        "n_pairs": n_pairs,
        "per_pair_error": per_pair_error_per_game,
        "global_per_pair_error": global_per_pair_error,
        "uniform_baseline_per_pair": uniform_baseline_per_pair_per_game,
        "global_uniform_baseline_per_pair": global_uniform_baseline_per_pair,
        "per_pair_margin": per_pair_margin_per_game,
        "global_per_pair_margin": global_per_pair_margin,
        "darkhex_stone_baseline_per_pair": stone_baseline_per_game,
        "darkhex_stone_baseline_margin": stone_margin_per_game,
        "darkhex_stone_baseline_unavailable": stone_unavailable,
        "key_mismatch_rows": key_mismatch_rows,
        "unscorable_rows": unscorable_rows,
        "not_applicable": not_applicable,
        "actor_filter": label,
        "dropped_rate": dropped_rate,
        "dropped_detail": dropped_detail,
    }


