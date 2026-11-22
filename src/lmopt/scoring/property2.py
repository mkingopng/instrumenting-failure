"""Property 2: Adaptivity - best-response gap vs the empirical opponent.

Measures how much the LM leaves on the table compared to the best response
against the specific opponent it faced. A gap of 0 means the LM was already
playing optimally against that opponent; a positive gap means an adaptive
player could have done better.

Secondary: monotone reward trend (is the LM getting better over hands?).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import polars as pl

# Shared duck-typing guard for property2's BR-gap tree walk - see
# lmopt.scoring._reference for why this lives in its own module (one copy,
# not three).
from lmopt.scoring._actor import (
    LEGACY_PROMPT,
    actor_label,
    actor_rows,
    drop_fallback_hands,
    environment_rows,
    refuse_foreign_probe_type,
    require_actor_column,
)
from lmopt.scoring._reference import is_partial_reference as _is_partial_reference

#: The run configuration P2 is waiting for, as reachable data.
#: P2's scorer, its trajectory binning, the memory accumulation and
#: the per-seat payoff hook are all landed; what no collected arm carries is an
#: agent that can condition on earlier hands, so a `br_gap` measured on them is a
#: gap against an opponent the agent could not have adapted to. The recorded
#: reason is satisfied by naming the arm rather than by a number.
#:
#: A constant rather than a string in the acceptance table, for the reason the
#: admissibility-gate rule states: the script computes and reports, it does not
#: know.
P2_AWAITING_CONFIG = "memory arm (ROADMAP step 6 post-reorder, step 7 pre-reorder)"


def hand_bin_index(hand_idx: int, span: tuple[int, int], bins: int) -> int:
    """
    the bin one hand falls in when a span of hand indices is cut into equal-width bins, in exact
    integer arithmetic: floor of (hand - first) * bins / span length. Float forms of the same
    division round a boundary hand into the wrong bin for widths that are not exact in binary
    (100 hands in 30 bins puts hand 10 in bin 2 under a float floor and in bin 3 here), so every
    caller that cuts P2's trajectory uses this one function.
    :param hand_idx: the hand index (int)
    :param span: (first hand index, last hand index) of the cut, both included (tuple[int, int])
    :param bins: the number of bins, at least 1 (int)
    :returns: the bin, clipped to [0, bins - 1] so a hand outside the span lands in an end bin (int)
    :raises ValueError: when `bins` is below 1 or the span is empty (`hi` below `lo`)
    """
    lo, hi = span
    if bins < 1 or hi < lo:
        raise ValueError(f"cannot cut span {span} into {bins} bins")
    return min(bins - 1, max(0, ((hand_idx - lo) * bins) // (hi - lo + 1)))


def score_property2(
    df: pl.DataFrame,
    games: Optional[dict] = None,
    cfr_policies: Optional[dict] = None,
    n_hand_bins: int = 0,
    actor: Optional[str] = None,
    hand_span: Optional[tuple[int, int]] = None,
) -> dict:
    """Compute the best-response gap against empirical opponent policies.

    For each matchup (game × opponent), reconstruct the empirical opponent
    policy from the rows of everyone the scored agent faced, then compute the
    best-response value at the seat that agent sat in. The gap =
    br_value - mean_reward.

    :param df: full match DataFrame (all games, all matchups).
    :param games: {game_name: pyspiel.Game} - needed for BR computation. If None,
        only the reward-trend secondary score is computed.
    :param cfr_policies: {game_name: policy} - used only to detect a partial
        depth-reference policy (via `is_partial_reference`) so the BR walk
        can be skipped for that game. If None, the guard cannot fire and
        BR is attempted unconditionally (safe for the five canonical
        games, which always have a full reference).
    :param n_hand_bins: number of equal-width hand-index bins for the P2
        trajectory. **0 (the default) disables it and leaves every other key
        byte-identical**, so this parameter cannot change any number already
        published. P2 is defined as `delta_t -> 0` as evidence accumulates
        -- a trajectory in `t` -- but the pooled `br_gap` below is one
        scalar per cell with `lm_mean` averaged over every hand, so it
        carries no `delta_t`. Bins group on
        `hand_idx`, pooling seeds at comparable `t`; they do not slice the
        (seed, hand_idx)-sorted row order, which would concatenate seeds
        and manufacture a trend out of between-seed differences.
    :param actor: the `actor` column value of the agent under test. `None` (the
        default) keeps the legacy populations, `prompt` non-null for the
        scored agent and `prompt` null for the environment, which is the
        only thing a legacy arm can express.
        Naming an actor scores that agent, and the environment becomes
        every row that is not that actor rather than every unprompted row.
    :param hand_span: (first hand index, last hand index) the trajectory bins
        are cut over. `None` (the default) cuts over the hands the cell
        nominally holds, that is the (seed, hand_idx) hands of the scored
        agent before fallback hands are dropped, so a late run of drops
        cannot move the bin edges. A per-shard caller passes the pooled span
        of its cell so every shard is cut on the same edges as the pooled
        cell. Ignored when `n_hand_bins` is 0. A span with its last hand
        before its first raises ValueError, and hands outside the span leave
        the trajectory unscored with that reason (tuple[int, int] | None)

    :returns: dict with keys (each nested ``{matchup_id: {game: value}}`` - P2 is
    opponent-keyed but must be computed PER game, never mixed across games):
        "br_gap": {matchup_id: {game: float}} - expected value gap.
        "br_value": {matchup_id: {game: float}} - BR value vs empirical opponent.
        "lm_mean_reward": {matchup_id: {game: float}} - actual mean reward of
            the scored agent (the key keeps its published name; with
            `actor=None` it is the LM's, as before).
        "on_support_fraction": {matchup_id: {game: float}} - fraction of the
            opponent info-sets the best-response walk queried that the
            empirical policy table actually had data for; the rest fell back
            to uniform. No silent fallbacks: br_value is a best
            response to a hybrid (empirical-where-covered, uniform
            elsewhere) policy, and this number is how much of the hybrid is
            uniform. nan wherever br_value is nan (same cell, same reason).
        "br_error": {matchup_id: {game: str}} - populated only when
            `_compute_br_value_and_coverage` raised for that cell (e.g. a
            malformed logged info-set string). No silent
            fallbacks: br_value/on_support_fraction are nan in this case too,
            but that nan must never be read as "measured, no signal" without
            checking this dict first -- a bare nan cannot distinguish "the
            walk ran and found nothing" from "the walk crashed".
        "br_seat": {matchup_id: {game: {str(seat): n_hands}}} - the seats the
            scored agent actually occupied in that cell, and how many hands at
            each. `br_value` is the hand-weighted mean of the per-seat
            best-response values, so this dict is the weighting, reported
            rather than assumed: a `seat_mode="random"` cell shows both seats,
            a fixed-seat cell shows one and the mean is that seat's value
            exactly (weight 1.0). Present for exactly the cells that have a
            "br_value" entry, which includes cells whose BR walk crashed (nan
            value, reason in "br_error") and excludes cells skipped by the
            partial-reference guard, since those `continue` before any BR
            value exists and weights for an absent value serve no consumer.
        "global_mean_gap": float - mean BR gap over all (matchup, game) cells.
        "not_applicable": {game: str} - BR-gap skipped with an explicit reason
            (e.g. a PARTIAL depth-reference policy: a full-tree best-response
            walk is not defined against off-support gaps, and is intractable
            at this depth). No score is populated in "br_gap"/"br_value" for
            that (matchup, game) cell when this fires.
        "dropped_rate": {matchup_id: {game: float | None}} - the fraction of
            the scored agent's hands removed from every estimate above because
            they held one of its harness-fallback rows (a substituted legal
            move is not the model's answer). The unit is hands, not rows:
            `lm_mean_reward` is one reward per hand, so a hand with a fallback
            row is dropped whole. `None` where the cell has no prompted
            hands, so nothing could be dropped (e.g. `actor="bayesian"`,
            whose rows carry no prompt and are never dropped). A cell with no
            rows for the scored agent at all is absent. A cell whose every
            hand was dropped reads 1.0 and has no entry in the estimate
            dicts above.
        "dropped_detail": {matchup_id: {game: dict}} - the counts behind
            "dropped_rate" (`DropReport.detail()`): unit, n_dropped, n_total,
            n_fallback_rows, n_lm_rows, fallback_row_rate, n_infosets and
            n_infosets_fully_dropped. `fallback_row_rate` is a rate over rows
            and differs from "dropped_rate", which is over hands.
        "actor_filter": str - which population produced every number above,
            `"actor=<name>"` or `"legacy-prompt-null"`. A filter that is not
            reported is a silent fallback.
        "lm_mean_trajectory": {matchup_id: {game: [float | None]}} - the
            scored agent's mean reward per hand-index bin, `None` for a bin
            that kept no hand. Present only for a cell whose first and last
            bins each kept at least one hand; a hollow bin between them leaves
            the series standing, because the delta reads only the two ends.
        "trajectory_bin_n_hands": {matchup_id: {game: [int]}} - nominal hands
            per bin, before fallback hands are dropped.
        "trajectory_bin_n_kept": {matchup_id: {game: [int]}} - hands per bin the
            means were taken over. Present only when `n_hand_bins > 0`.
        "trajectory_bin_dropped_rate": {matchup_id: {game: [float | None]}} -
            per-bin share of nominal hands dropped, `None` for a bin with no
            nominal hands. Present only when `n_hand_bins > 0`.
        "trajectory_not_applicable": {matchup_id: {game: str}} - why a cell has
            no trajectory: no `hand_idx` column, fewer hand indices than bins,
            hands outside a given `hand_span`, or an END bin holding no kept
            hand. An end bin with nothing in it leaves the series unscored and
            never re-cut over the surviving hands, which would put a different
            window under the same licence.

    Each (matchup_id, game) cell is computed from rows filtered by
    both matchup_id and game, so a game's reward/tree never bleeds into another's.
    Hands are grouped by (seed, hand_idx) when a seed column is present, so
    multi-seed data is not collapsed onto a single reward per hand_idx.
    """
    require_actor_column(df, actor, "score_property2")
    if n_hand_bins > 0 and hand_span is not None and hand_span[0] > hand_span[1]:
        raise ValueError(f"hand_span {hand_span} ends before it starts")
    refuse_foreign_probe_type(df, "score_property2")
    actor_filter = actor_label(actor, LEGACY_PROMPT)
    br_gap: dict[str, dict[str, float]] = {}
    br_seat: dict[str, dict[str, dict[str, int]]] = {}
    br_value_out: dict[str, dict[str, float]] = {}
    on_support_out: dict[str, dict[str, float]] = {}
    br_error: dict[str, dict[str, str]] = {}
    lm_mean_reward: dict[str, dict[str, float]] = {}
    not_applicable: dict[str, str] = {}
    dropped_rate: dict[str, dict[str, Optional[float]]] = {}
    dropped_detail: dict[str, dict[str, dict]] = {}
    # P2 trajectory. Populated only when n_hand_bins > 0.
    br_gap_trajectory: dict[str, dict[str, list[float | None]]] = {}
    lm_mean_trajectory: dict[str, dict[str, list[float | None]]] = {}
    trajectory_bin_n_hands: dict[str, dict[str, list[int]]] = {}
    trajectory_bin_n_kept: dict[str, dict[str, list[int]]] = {}
    trajectory_bin_dropped_rate: dict[str, dict[str, list[Optional[float]]]] = {}
    trajectory_not_applicable: dict[str, dict[str, str]] = {}

    # Group hands by (seed, hand_idx) so seeds are not collapsed; fall back to
    # hand_idx alone for synthetic frames with no seed column.
    if "seed" in df.columns and "hand_idx" not in df.columns:
        raise ValueError(
            "score_property2: the frame has a seed column but no hand_idx column, so its "
            "hands cannot be told apart within a seed; score a frame that logs hand_idx"
        )
    hand_keys: list[str] = [c for c in ("seed", "hand_idx") if c in df.columns] or [
        "hand_idx"
    ]

    # maintain_order=True: without it,
    # `.unique()`'s default order is process-dependent (PYTHONHASHSEED-
    # affected), so
    # Which matchup a multi-matchup cell reports (via property_table's
    # `mid = next(iter(p2["br_gap"]))`) would vary run to run. This is a
    # research instrument with FP-determinism audit requirements; a scorer
    # must not have a process-dependent selection in it.
    for matchup_id in df["matchup_id"].unique(maintain_order=True).to_list():
        mm = df.filter(pl.col("matchup_id") == matchup_id)
        # Same process-dependence class as the
        # matchup_id loop above. Confined to the
        # FP summation order of `global_mean_gap` on whole-frame calls
        # (last-ULP) -- `property_table` never reaches this loop with more
        # than one game per cell, since its `df` is already sliced to a
        # single (game, opponent) pair before `score_property2` is called.
        for game_name in mm["game"].unique(maintain_order=True).to_list():
            matchup_df = mm.filter(pl.col("game") == game_name)

            # --- Scored agent's mean reward (this game, this opponent only) ---
            lm_rows = actor_rows(
                matchup_df, actor, legacy=pl.col("prompt").is_not_null()
            )
            if lm_rows.is_empty():
                continue
            # The unit is the hand: `lm_mean_reward` is one reward per hand, so
            # a hand holding a fallback row leaves the estimate whole. The
            # drop restricts itself to rows with a prompt, which is what keeps
            # a Bayesian actor's own `fallback_used` rows in. The opponent
            # policy below is built from `matchup_df`, not from these kept
            # rows: what the opponent did at an info-set does not depend on
            # how the LM reached it.
            nominal_hands = lm_rows.select(hand_keys).unique()
            report = drop_fallback_hands(lm_rows, hand_keys)
            dropped_rate.setdefault(matchup_id, {})[game_name] = report.rate
            dropped_detail.setdefault(matchup_id, {})[game_name] = report.detail()
            lm_rows = report.kept
            if lm_rows.is_empty():
                continue
            # The seat the scored agent occupied in a hand is `player_to_act`
            # on its own rows: an agent has a decision row only at the seat it
            # sat in, which also makes this correct on legacy arms with no seat
            # column at all. Not `lm_seat`, which names the seat AGENT0
            # occupies (logging/schema.py) and is therefore the OPPONENT's seat
            # whenever the scored agent is agent1.
            hand_agg = [pl.col("reward").first()]
            if "player_to_act" in lm_rows.columns:
                hand_agg.append(pl.col("player_to_act").first().alias("_seat"))
            lm_hands = lm_rows.group_by(hand_keys).agg(hand_agg).sort(hand_keys)
            rewards = lm_hands["reward"].to_list()
            seat_hands: dict[int, int] = {}
            seats_per_hand = (
                [int(s) for s in lm_hands["_seat"].to_list()]
                if "_seat" in lm_hands.columns
                else [0] * len(lm_hands)
            )
            for _s in seats_per_hand:
                seat_hands[_s] = seat_hands.get(_s, 0) + 1
            lm_mean = float(np.mean(rewards)) if rewards else float("nan")
            lm_mean_reward.setdefault(matchup_id, {})[game_name] = lm_mean

            # --- P2 trajectory: per-hand-index-bin LM mean ---
            # Grouped on `hand_idx`, not sliced from the (seed, hand_idx)-sorted
            # `rewards` list above: positional slicing would put whole seeds in
            # separate bins and read between-seed differences as a learning
            # curve. A per-hand-order correlation over `rewards` as sorted here
            # would have exactly that defect (its index runs across
            # concatenated seeds), which is why no such quantity is computed
            # from this list at all -- a reward trend worth having is one per
            # seed, since the seed is the replication unit, and nothing in this
            # file builds that.
            bin_means: list[float | None] = []
            if n_hand_bins > 0:
                if "hand_idx" not in lm_hands.columns:
                    trajectory_not_applicable.setdefault(matchup_id, {})[
                        game_name
                    ] = "no hand_idx column: hands cannot be ordered in t"
                elif nominal_hands["hand_idx"].n_unique() < n_hand_bins:
                    # Degenerate-cell rule: name it; do not hand back a
                    # shaped-but-meaningless series.
                    trajectory_not_applicable.setdefault(matchup_id, {})[game_name] = (
                        f"{nominal_hands['hand_idx'].n_unique()} distinct hand indices is "
                        f"fewer than the {n_hand_bins} bins requested"
                    )
                else:
                    # The edges come from the hands the cell nominally holds,
                    # before the fallback drop, through the one shared bin
                    # function; the kept hands are placed on those edges.
                    nominal_list = nominal_hands["hand_idx"].to_list()
                    span = (
                        hand_span
                        if hand_span is not None
                        else (int(min(nominal_list)), int(max(nominal_list)))
                    )
                    outside = [h for h in nominal_list if not span[0] <= h <= span[1]]
                    if outside:
                        trajectory_not_applicable.setdefault(matchup_id, {})[
                            game_name
                        ] = (
                            f"{len(outside)} of {len(nominal_list)} hands lie outside the span "
                            f"{span} the bins were asked to cover"
                        )
                    else:
                        binned = lm_hands.with_columns(
                            pl.Series(
                                "_bin",
                                [
                                    hand_bin_index(h, span, n_hand_bins)
                                    for h in lm_hands["hand_idx"].to_list()
                                ],
                                dtype=pl.Int64,
                            )
                        )
                        agg = (
                            binned.group_by("_bin")
                            .agg(
                                [
                                    pl.col("reward").mean().alias("_m"),
                                    pl.len().alias("_n"),
                                ]
                            )
                            .sort("_bin")
                        )
                        by_bin = {
                            r["_bin"]: (r["_m"], r["_n"])
                            for r in agg.iter_rows(named=True)
                        }
                        nominal_counts = [0] * n_hand_bins
                        for h in nominal_list:
                            nominal_counts[hand_bin_index(h, span, n_hand_bins)] += 1
                        kept_counts = [
                            int(by_bin.get(b, (None, 0))[1]) for b in range(n_hand_bins)
                        ]
                        trajectory_bin_n_hands.setdefault(matchup_id, {})[
                            game_name
                        ] = nominal_counts
                        trajectory_bin_n_kept.setdefault(matchup_id, {})[
                            game_name
                        ] = kept_counts
                        trajectory_bin_dropped_rate.setdefault(matchup_id, {})[
                            game_name
                        ] = [
                            1.0 - k / n if n else None
                            for k, n in zip(kept_counts, nominal_counts, strict=True)
                        ]
                        # Only the two ends decide whether the series stands:
                        # the delta is the first bin's mean minus the last
                        # bin's, so a hollow bin between them costs one
                        # reported mean and nothing else. An empty END bin
                        # leaves the delta undefined, and the series is not
                        # re-cut over the surviving hands, which would put a
                        # different window under the same licence.
                        empty_ends = sorted(
                            {b for b in (0, n_hand_bins - 1) if kept_counts[b] == 0}
                        )
                        if empty_ends:
                            trajectory_not_applicable.setdefault(matchup_id, {})[
                                game_name
                            ] = (
                                f"end bins {empty_ends} (0-based, of {n_hand_bins}) hold no kept "
                                f"hands, out of nominal {nominal_counts}: "
                                + (
                                    "every hand in them was dropped as a fallback"
                                    if all(nominal_counts[b] > 0 for b in empty_ends)
                                    else "some hold no hands at all"
                                )
                            )
                        else:
                            bin_means = [
                                float(by_bin[b][0]) if kept_counts[b] else None
                                for b in range(n_hand_bins)
                            ]
                            lm_mean_trajectory.setdefault(matchup_id, {})[
                                game_name
                            ] = bin_means

            # --- Best-response gap (this game's tree, this game's opponent rows) ---
            game_obj = games.get(game_name) if games else None
            cfr_pol = cfr_policies.get(game_name) if cfr_policies else None
            if cfr_pol is not None and _is_partial_reference(cfr_pol):
                # No silent fallbacks: record a reason, never NaN (a
                # skip signal elsewhere in this codebase) and never a bare
                # `continue` - a full-tree best-response is not defined against
                # a policy with off-support gaps, and the tree is intractable
                # at this depth. Mirrors property7.py/property8.py exactly:
                # `_compute_br_value`
                # is an unbounded recursive full-tree walk over `games[game_name]`
                # that never consults a reference and never returns for
                # darkhex_3x3 (measured: still running after 90s under
                # faulthandler.dump_traceback_later - it does not raise, so the
                # broad except below never fires).
                not_applicable.setdefault(
                    game_name,
                    (
                        f"partial reference ({cfr_pol.n_infosets} info-sets): a full-tree "
                        "best-response is not defined against a policy with off-support "
                        "gaps, and the tree is intractable at this depth"
                    ),
                )
                continue

            br_val = float("nan")
            on_support = float("nan")
            gap = float("nan")
            if game_obj is not None:
                emp_policy = _build_empirical_policy(
                    environment_rows(
                        matchup_df, actor, legacy=pl.col("prompt").is_null()
                    )
                )
                if emp_policy:
                    try:
                        # One scalar per cell, over an agent that may have sat
                        # at both seats: the hand-weighted mean of the per-seat
                        # best-response values. With a single seat the weight is
                        # exactly 1.0, so a fixed-seat cell is bit-identical to
                        # a single per-seat call. `br_seat` reports
                        # the weighting (see this function's docstring).
                        total_hands = sum(seat_hands.values())
                        br_val = 0.0
                        on_support = 0.0
                        for _seat, _n in sorted(seat_hands.items()):
                            _v, _c = _compute_br_value_and_coverage(
                                game_obj, emp_policy, seat=_seat
                            )
                            _w = _n / total_hands
                            br_val += _w * _v
                            on_support += _w * _c
                    except Exception as e:
                        # No silent fallbacks: a bare `except: br_val = nan`
                        # here is exactly the shape the rule forbids. The
                        # reason must survive so a downstream reader
                        # (analyse_stats.py's `emit`) can tell "the BR walk
                        # crashed" apart from "measured, no data" (both look
                        # like a bare nan without this). Recorded per (matchup,
                        # game) cell, not per game, because a malformed
                        # opponent row (e.g. an info_set string OpenSpiel's
                        # `information_state_string` never produces) is a
                        # property of this matchup's logged data, not of the
                        # game itself -- unlike the partial-reference guard
                        # above, whose reason is the same for every matchup.
                        br_val = float("nan")
                        on_support = float("nan")
                        br_error.setdefault(matchup_id, {})[
                            game_name
                        ] = f"{type(e).__name__}: {e}"
                    if not math.isnan(br_val) and not math.isnan(lm_mean):
                        gap = br_val - lm_mean
            br_value_out.setdefault(matchup_id, {})[game_name] = br_val
            on_support_out.setdefault(matchup_id, {})[game_name] = on_support
            br_gap.setdefault(matchup_id, {})[game_name] = gap
            br_seat.setdefault(matchup_id, {})[game_name] = {
                str(_seat): _n for _seat, _n in sorted(seat_hands.items())
            }

            # `br_val` is pinned across bins, never recomputed per window.
            # Per-window opponent coverage collapses (leduc 135 pooled
            # info-sets -> 56-72 per window) and `_build_empirical_policy`
            # pads off-support with uniform, so a per-window br_value would
            # best-respond to a different hybrid target each window. Only the
            # pinned form answers P2's question, "does the gap to a fixed
            # target shrink" -- and the core-path design fixes the opponent,
            # so pinning is what the instrument's own design already asserts.
            if bin_means:
                br_gap_trajectory.setdefault(matchup_id, {})[game_name] = [
                    None if m is None else br_val - m for m in bin_means
                ]

    valid_gaps = [
        v for inner in br_gap.values() for v in inner.values() if not math.isnan(v)
    ]
    global_mean = float(np.mean(valid_gaps)) if valid_gaps else float("nan")

    result = {
        "br_gap": br_gap,
        "br_value": br_value_out,
        "br_seat": br_seat,
        "on_support_fraction": on_support_out,
        "br_error": br_error,
        "actor_filter": actor_filter,
        "lm_mean_reward": lm_mean_reward,
        "global_mean_gap": global_mean,
        "not_applicable": not_applicable,
        "dropped_rate": dropped_rate,
        "dropped_detail": dropped_detail,
        # P2 trajectory. Empty dicts when n_hand_bins == 0, which
        # is the default -- every key above is unchanged in that case.
        "br_gap_trajectory": br_gap_trajectory,
        "lm_mean_trajectory": lm_mean_trajectory,
        "trajectory_bin_n_hands": trajectory_bin_n_hands,
        "trajectory_not_applicable": trajectory_not_applicable,
    }
    if n_hand_bins > 0:
        # Present only when the trajectory is requested, so the default call
        # keeps exactly the keys it always returned.
        result["trajectory_bin_n_kept"] = trajectory_bin_n_kept
        result["trajectory_bin_dropped_rate"] = trajectory_bin_dropped_rate
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_empirical_policy(
    opp_rows: pl.DataFrame,
) -> dict[tuple[int, str], dict[int, float]]:
    """Build a tabular policy from logged opponent rows.

    Returns {(player, info_set_str): {action: probability}}.
    """
    if opp_rows.is_empty():
        return {}

    policy: dict[tuple[int, str], dict[int, int]] = {}
    for row in opp_rows.iter_rows(named=True):
        player = int(row["player_to_act"])
        info_str = row["info_set"]
        action = int(row["parsed_action"])
        key = (player, info_str)
        if key not in policy:
            policy[key] = {}
        policy[key][action] = policy[key].get(action, 0) + 1

    # Normalise counts to probabilities.
    result: dict[tuple[int, str], dict[int, float]] = {}
    for key, counts in policy.items():
        total = sum(counts.values())
        result[key] = {a: c / total for a, c in counts.items()}
    return result


class _EmpiricalPolicy:
    """OpenSpiel-compatible policy wrapper around an empirical frequency table.

    Also tracks, per (player, info_set) key queried, whether the table had
    data for it ("on support") or the query fell back to the uniform
    distribution ("off support"). No silent fallbacks: the
    uniform fill on unseen info-sets is itself a fallback, so its rate must
    be visible in the output rather than silently absorbed into br_value.
    See `on_support_fraction()`.

    This instance is only ever queried for the opponent
    (non-`seat`) player during a `BestResponsePolicy` walk -- `transitions()`
    only calls `self._policy.action_probabilities(state)` on a turn where
    `state.current_player() != self._player_id`, never the BR player's own
    turn. The one place this could break is a genuine simultaneous-move
    node, where OpenSpiel's `joint_action_probabilities_aux`
    (`open_spiel/python/policy.py`) queries `action_probabilities(state,
    player)` for every player including the BR player, which would inject
    `seat`'s own keys into `_seen` and understate coverage. This holds
    only because every simultaneous-move game this scorer sees (goofspiel)
    is wrapped in `turn_based_simultaneous_game(...)` before it reaches here
    (`lmopt.games.loaders.load_goofspiel`), which removes true simultaneous
    nodes from the tree. If a genuinely simultaneous game is ever scored
    through this path, `_seen`/`on_support_fraction` need a `player == seat`
    guard that does not exist yet.
    """

    def __init__(self, table: dict[tuple[int, str], dict[int, float]]) -> None:
        self._table = table
        self._seen: set[tuple[int, str]] = set()
        self._on_support: set[tuple[int, str]] = set()

    def action_probabilities(
        self, state, player_id: Optional[int] = None
    ) -> dict[int, float]:
        player = state.current_player() if player_id is None else player_id
        info_str = state.information_state_string(player)
        key = (player, info_str)
        self._seen.add(key)
        legal = state.legal_actions()
        probs = self._table.get(key)
        if probs is not None:
            # Only return probabilities for legal actions; normalise in case
            # of mismatch.
            filtered = {a: probs.get(a, 0.0) for a in legal}
            total = sum(filtered.values())
            if total > 0.0:
                self._on_support.add(key)
                return {a: v / total for a, v in filtered.items()}
        p = 1.0 / len(legal)
        return {a: p for a in legal}

    def on_support_fraction(self) -> float:
        """Fraction of queried (player, info_set) keys the table actually covered.

        "Queried" undersells what `_seen` actually is:
        `BestResponsePolicy.__init__` (via `info_sets()`/`decision_nodes()`)
        recurses over every transition with no reach-probability pruning
        (`best_response.py`'s `decision_nodes`, no `p > 0` filter), and this
        class returns a key for every legal action at every state it is
        asked about. So by construction `len(self._seen)` equals the total
        number of distinct opponent (player != seat) info-sets in the whole
        game tree, not merely the ones this particular walk happened to
        touch -- verified: kuhn 6/6, goofspiel_4 81/81,
        liarsdice 12288/12288 (`get_all_states.get_all_states` count of
        distinct `information_state_string(1)` values, vs `len(_seen)` after
        a full walk, exact match on all three). `on_support_fraction` is
        therefore literally "distinct logged opponent info-sets ÷ total
        opponent info-sets in the tree" -- the same denominator behind
        coverage figures such as 0.2% (liarsdice) / 83% (kuhn).

        Deliberately unweighted (not reach-probability- or
        counterfactual-reach-weighted): a reach-weighted version would have
        to weight by the hybrid policy's own probabilities, which are partly
        the uniform fallback this fraction exists to expose -- weighting the
        contamination by itself. The unweighted count treats "we never saw
        this opponent info-set" as equally worth reporting regardless of how
        reachable it is under the (already-contaminated) hybrid policy.

        nan when nothing was queried (e.g. an empty empirical table never
        reached, which `score_property2` already guards against upstream via
        `if emp_policy:`).
        """
        if not self._seen:
            return float("nan")
        return len(self._on_support) / len(self._seen)


def _compute_br_value_and_coverage(
    game, emp_policy_table: dict, seat: int
) -> tuple[float, float]:
    """Best-response value for `seat` against `emp_policy_table`, plus the
    fraction of opponent info-sets encountered along the way that the table
    actually covered (vs the uniform fallback). The walk is exhaustive, so
    this denominator is every opponent info-set in the game tree, not a
    sample of it -- see `_EmpiricalPolicy.on_support_fraction`'s docstring.

    Imported, never hand-rolled: this is OpenSpiel's own
    `BestResponsePolicy`, which pools counterfactual-reach-weighted value
    over every state that shares an information set (`infosets` dict built
    in its `__init__`, `best_response_action`'s `max` taken over that pooled
    sum) -- a hand-rolled walk taking `max` per state would be
    equivalent to solving with the opponent's card revealed.
    """
    from open_spiel.python.algorithms.best_response import BestResponsePolicy

    emp_pol = _EmpiricalPolicy(emp_policy_table)
    br = BestResponsePolicy(game, seat, emp_pol)
    value = float(br.value(game.new_initial_state()))
    return value, emp_pol.on_support_fraction()


def _compute_br_value(game, emp_policy_table: dict, seat: int) -> float:
    """Compute the expected value for seat playing best-response against
    emp_policy_table. Thin wrapper over `_compute_br_value_and_coverage` that
    drops the coverage fraction. Tests only -- no
    production caller reads coverage-blind br_value; `score_property2` calls
    `_compute_br_value_and_coverage` directly. Kept because
    `tests/test_scorer_property2.py` (`test_gap_zero_when_lm_plays_br`,
    `TestPartialReferenceGuard`'s hang-proof) call it directly and do not
    need the coverage half.
    """
    return _compute_br_value_and_coverage(game, emp_policy_table, seat)[0]


# Expose _EmpiricalPolicy for tests.
__all__ = [
    "P2_AWAITING_CONFIG",
    "score_property2",
    "_EmpiricalPolicy",
    "_compute_br_value",
    "_compute_br_value_and_coverage",
]
