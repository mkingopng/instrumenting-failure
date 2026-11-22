"""
Property 8: Value of Information (Dark Hex) - offline scorer.

Scores whether the LM captures the value available from *information-gathering*
(probing contested cells), against the CFR+ reference. Pure function of the logged
Parquet + CFR reference + board-posterior (scoring is offline); it
uses the LM's logged action, so it needs no belief elicitation (that is P8b).

Definitions (the 2x2-baseline operationalisation - a rigorous Howard form on a
larger board needs a 3x3 reference):

  μ(w|I)      reach-weighted worlds consistent with P0 info-state I (darkhex_posterior)
  V_cfr(I)    = Σ_w μ(w) · value(w under CFR+ both players)          [imperfect-info optimum]
  V_pi(I)     = Σ_w μ(w) · value(w with P0 best-responding, knowing w) [perfect-info optimum]
  VoI*(I)     = V_pi(I) − V_cfr(I) ≥ 0     [EVPI: the value of resolving all uncertainty here]
  V(a|I)      = Σ_w μ(w) · value(w after P0 plays a, then CFR+)       [info-aware value of a]
  P8_shortfall(I) = (max_a V(a|I) − V(a_LM|I)) / max(VoI*(I), ε)      [↓ better; 0 captured all]

Aggregated over P0 nodes with VoI*(I) > τ (nodes where information actually
matters); the count of such nodes (`n_informative`) is reported so a cell with
no informative population reads as "no signal", not as a pass. **2x2 Dark Hex
is itself measured NON-degenerate** (`tests/test_
property8.py`: 25 of 33 P0 info-states have VoI* > 0), so `n_informative=0`
for one arm is a fact about that ARM's own logged rows, never a fact about the
board, and must not be read as "the board has no signal" -- `n_board_
informative` reports the board's own count over the same
reference so a caller can tell the two apart.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import polars as pl

# Shared duck-typing guard, used here and in property7.py, lives in
# lmopt.scoring._reference so a loosening can't fix one copy and miss the
# other. Aliased to a private name for this module's internal call sites.
from lmopt.scoring._actor import (
    LEGACY_PROMPT,
    actor_label,
    actor_rows,
    drop_fallback_rows,
    refuse_foreign_probe_type,
    require_actor_column,
)
from lmopt.scoring._reference import is_partial_reference as _is_partial_reference


def _br_value(state, cfr_pol, seat: int = 0) -> float:
    """Expected return to ``seat`` best-responding (knowing this world) vs a CFR+
    opponent. seat maximises; the opponent plays cfr_pol."""
    if state.is_terminal():
        return state.returns()[seat]
    acting = state.current_player()
    legal = state.legal_actions()
    if acting == seat:
        return max(_br_value(state.child(a), cfr_pol, seat) for a in legal)
    probs = cfr_pol.action_probabilities(state, acting)
    ev = 0.0
    for a in legal:
        p = probs.get(a, 0.0)
        if p > 0.0:
            ev += p * _br_value(state.child(a), cfr_pol, seat)
    return ev


def _voi_and_action_values(
    worlds_for_info: list, cfr_pol, seat: int = 0
) -> tuple[dict, float]:
    """Return ({action: V(a|I)}, VoI*(I)) for one info-state's worlds.

    V(a|I) is the reach-weighted CFR+ continuation after `seat` plays a; VoI* is
    the EVPI = V_pi − V_cfr over the same worlds.

    Every value is read at `seat`'s own index. Dark Hex is zero-sum,
    so `returns()[0] == -returns()[1]`: scoring a seat-1 node at index 0 negates
    the whole action-value map and turns the best action into the worst one.
    `seat=0` is the default, which is what a legacy (seat-0-only) arm needs.
    """
    from open_spiel.python.algorithms.expected_game_score import policy_value

    total = sum(w for _, w in worlds_for_info)
    if total <= 0.0:
        return {}, 0.0

    legal = worlds_for_info[0][0].legal_actions()
    v_a: dict[int, float] = {a: 0.0 for a in legal}
    v_cfr = 0.0
    v_pi = 0.0
    for state, reach in worlds_for_info:
        if reach <= 0.0:
            continue
        w = reach / total
        v_cfr += w * policy_value(state, [cfr_pol, cfr_pol])[seat]
        v_pi += w * _br_value(state, cfr_pol, seat=seat)
        for a in legal:
            v_a[a] += w * policy_value(state.child(a), [cfr_pol, cfr_pol])[seat]
    voi = max(0.0, v_pi - v_cfr)
    return v_a, voi


def reference_visit_weights(game, policy, seat: int) -> dict:
    """``{info_str(seat): expected visits per game under CFR+ self-play}``.

    Not the same weighting as ``gather_worlds``, and the difference is the point.
    ``gather_worlds`` accumulates the OPPONENT's reach only, because the
    observer's own reach cancels in the posterior normalisation; that is right
    for mu(w|I) and wrong for "how much of the reference's own play does this
    info-set account for", since an info-set the reference itself would never
    choose to enter still carries full opponent reach there. This walk
    multiplies both players' CFR+ probabilities, so the weight is the expected
    number of times CFR+ vs CFR+ play visits that info-set in one game.

    It weights the REFERENCE's occupancy, never the agent's. The two come apart
    hard: at darkhex 2x2 seat 0, the 12 informative
    info-sets carry ~1e-6 of this mass while being 42.9 % of the qwen `cfr_ref`
    cell's logged LM rows, because an information-optimal player avoids ever
    needing to probe. `score_property8` reports the ratio so the caveat layer
    can name that gap from measurement instead of asserting it.
    """
    weights: dict[str, float] = {}

    def _walk(state, reach: float) -> None:
        if state.is_terminal() or reach <= 0.0:
            return
        player = state.current_player()
        probs = policy.action_probabilities(state, player)
        if player == seat:
            info = state.information_state_string(seat)
            weights[info] = weights.get(info, 0.0) + reach
        for a in state.legal_actions():
            p = probs.get(a, 0.0)
            if p > 0.0:
                _walk(state.child(a), reach * p)

    _walk(game.new_initial_state(), 1.0)
    return weights


def score_property8(
    df: pl.DataFrame,
    cfr_policies: Optional[dict] = None,
    games: Optional[dict] = None,
    voi_threshold: float = 1e-2,
    epsilon: float = 1e-6,
    actor: Optional[str] = None,
) -> dict:
    """
    compute Dark Hex P8 (value of information) over the scored agent's nodes
    The scored LM's own harness-fallback rows are dropped per game before any
    node is scored, so a substituted move is never read as the model's choice.
    Rows lost to a worlds-map miss are a different loss and stay in
    "discarded_rows".
    :param df: full match DataFrame (pl.DataFrame)
    :param cfr_policies: `{game: full CFR+ policy}` for the darkhex-family games (dict | None)
    :param games: `{game: OpenSpiel game}` matching `cfr_policies` (dict | None)
    :param voi_threshold: tau in the spec's "aggregate over nodes with VoI*(I) >
            tau". Set to 1e-2. The measured basis, not a preference: any value in
            [1e-4, 0.1] selects an identical population, at both boards and
            both seats. At darkhex 2x2 seat 0, the 33
            info-sets are 8 exact zeros, 13 CFR+ residuals between 6.7e-7 and
            2.0e-6, 4 at 0.6667 and 8 at 1.0, and nothing lands in (1e-4, 0.1]
            at either board or either seat, so 1e-2 sits in an empty band.

            The wider band [1e-4, 0.5] holds on the 2x2 board only. On
            darkhex_2x3 (`p8_reconciliation.json` tau_sweep), seat 1 goes from
            328 info-sets at tau <= 0.1 to 312 at
            0.25 and 52 at 0.5, and seat 0 goes 1,178 to 1,146 at 0.5.
            Do not widen tau past 0.1 for a 2x3 arm on the strength of a 2x2
            measurement.

            tau = 0 is the one bad choice and is not a neutral one:
            `max(VoI*, epsilon)` divides an O(1) numerator by 1e-6 at the
            residual nodes, and qwen `cfr_ref` reads mean shortfall 333,666.5 at
            tau=0 against 0.0 at tau=1e-4. (float)
    :param epsilon: floor on the VoI* denominator and tolerance for "attains the
        best action value" (float)
    :param actor: the `actor` column value of the agent under test, or `None` for
            the legacy prompt-null population (the only one a legacy arm can
            express). Either way the nodes are taken at both seats: filtering
            `player_to_act == 0` would make a `seat_mode="random"` arm
            lose about half its P8 rows with no count and no reason.
            On a legacy arm this changes nothing: every one of the
            1,259,089 landed legacy LM rows has `player_to_act == 0`. (str | None)
    :returns: dict with keys
        "p8_shortfall": {game: float}      mean P8_shortfall over informative nodes
        "capture_rate": {game: float}      fraction of informative nodes where the
                                           agent's action attains ~max_a V(a|I)
        "n_informative": {game: int}       nodes with VoI* > threshold (the ones scored)
        "n_board_informative": {game: int} the REFERENCE's own count of
                                           informative info-states (VoI* > threshold) over
                                           every info-state in the worlds map built for the
                                           seat(s) this arm occupied -- not just the ones
                                           this arm's own rows actually visited. Lets a
                                           caller distinguish "this arm never reached any
                                           informative node" (n_informative=0,
                                           n_board_informative>0 -- an arm-specific fact)
                                           from "this board/seat has no informative nodes at
                                           all at this threshold" (n_board_informative=0 --
                                           a board fact). Present only when at least one row
                                           for this game entered the scoring loop (i.e. some
                                           seat's worlds map was built); absent when the
                                           game/opponent cell had no rows at all.
        "n_nodes": {game: int}             total darkhex nodes considered, over
                                           every seat the agent sat in
        "n_visited_infosets": {game: int}  distinct (seat, info-state) pairs those nodes
                                           reached in the reference's worlds map
        "mean_voi": {game: float}          mean VoI* over all nodes (diagnostic)
        "available": bool
        "not_applicable": {game: str}      skipped with an explicit reason: no CFR
                                           reference/game object supplied at all,
                                           no reference available yet for this
                                           specific darkhex-family game, or a
                                           PARTIAL depth-reference policy. Only
                                           darkhex-family games ever get an entry
                                           here -- a game whose family isn't Dark
                                           Hex is a different state the caller
                                           determines on its own (GAME_FAMILIES),
                                           not this scorer's to report. No score
                                           is populated for that game elsewhere.
        "discarded_rows": {game: {...}}    rows that entered the scorer and never
                                           reached `n_nodes`, counted PER seat with
                                           a reason (no silent
                                           fallbacks). A game with nothing dropped
                                           is ABSENT, so a clean arm grows no
                                           spurious caveat. Per seat because the
                                           lookup key space is per seat: a seat-0
                                           row misses against a different map from
                                           a seat-1 row, and one pooled count
                                           cannot tell those apart.
        "excluded_rows": {game: {...}}     nodes that did reach `n_nodes` and were
                                           then excluded from the estimand, by
                                           reason. The spec excludes nodes with no
                                           available VoI because they do not test
                                           the property; this names them instead of
                                           leaving them as an `n_nodes` minus
                                           `n_informative` subtraction. The keys are
                                           `voi_at_or_below_threshold`,
                                           `unreachable_info_set` and
                                           `action_absent_from_value_dict` (the
                                           row's action has no entry in V(a|I)).
        "n_null_action": {game: int}       informative rows whose parsed_action is
                                           null, counted and left out of the
                                           estimand; every scored game has an entry.
        "n_shortfall_gt1": {game: int}     scored nodes whose shortfall exceeds 1.
                                           The spec reads P8_shortfall as a
                                           fraction of available VoI ("1 = captured
                                           none"), but nothing bounds the numerator
                                           by the denominator, and the landed
                                           arms reach 2.0. A
                                           non-zero count here means the published
                                           mean is not a fraction; it is reported
                                           rather than clipped, because clipping
                                           would hide the divergence the number is
                                           evidence of.
        "reference_visit_share": {game: float}
                                           the share of the CFR+ reference's own
                                           expected visit mass, at the seats the
                                           agent occupied, carried by the
                                           info-sets the agent was actually
                                           scored at. Near 0 means the yardstick
                                           ("what would CFR+ do from here") is
                                           being evaluated almost entirely at
                                           positions CFR+ would never reach.
                                           ~1e-6 on the
                                           landed 2x2 arms, against those same
                                           info-sets being 42.9 % of the agent's
                                           own decision nodes. Absent when the
                                           game produced no informative node.
        "voi_threshold": float             the tau these numbers were computed at,
                                           echoed for provenance (1e-2 by
                                           default, see the argument).
        "actor_filter": str                which population produced the numbers,
                                           `"actor=<name>"` or `"legacy-prompt-null"`.
        "dropped_rate": {game: float | None}
                                           the scored LM's fallback rows over its
                                           rows for that game, dropped before
                                           scoring. Separate from
                                           "discarded_rows". `None` when the game
                                           has no LM row. Empty for a game the
                                           scorer did not reach (no reference,
                                           partial reference, not Dark Hex).
        "dropped_detail": {game: dict}     the counts behind each rate
                                           (`DropReport.detail()`).
    """
    from lmopt.games.families import GAME_FAMILIES
    from lmopt.scoring.darkhex_posterior import gather_worlds

    require_actor_column(df, actor, "score_property8")
    refuse_foreign_probe_type(df, "score_property8")
    # `player_to_act` is read unguarded in the scoring loop below, where the row
    # is a plain dict from `iter_rows(named=True)`, so its absence would surface as a
    # bare `KeyError: 'player_to_act'` with no mention of the scorer, the frame
    # or the arm. It is not optional here: the seat is
    # the estimand's owner, and P8 has no seat-0 default to fall back to the way
    # P2 does. Named here, once, rather than defended per row.
    if "player_to_act" not in df.columns:
        raise ValueError(
            "score_property8: frame is missing column 'player_to_act'. P8's "
            "estimand belongs to the ACTING seat (not `lm_seat`, which names "
            "agent0's seat), so there is no seat to score without it."
        )
    # Heterogeneous result mapping: per-game dicts, a bool, a float and a str; its keys are the
    # ones the docstring lists.
    out: dict[str, Any] = {
        "p8_shortfall": {},
        "capture_rate": {},
        "n_informative": {},
        "n_board_informative": {},
        "n_nodes": {},
        "n_visited_infosets": {},
        "mean_voi": {},
        "available": False,
        "not_applicable": {},
        "discarded_rows": {},
        "excluded_rows": {},
        "n_shortfall_gt1": {},
        "reference_visit_share": {},
        "voi_threshold": voi_threshold,
        "actor_filter": actor_label(actor, LEGACY_PROMPT),
        "dropped_rate": {},
        "dropped_detail": {},
        "n_null_action": {},
    }
    if cfr_policies is None or games is None:
        # No CFR reference/game object was supplied to the scorer at all --
        # distinct from "this game's family isn't Dark Hex" (which the
        # caller (analyse_stats.py) can determine on its own via
        # GAME_FAMILIES and needs no reason from here) and from a PARTIAL
        # reference (`_is_partial_reference`, recorded below). Recorded for
        # every darkhex-family game actually present in `df` so a caller
        # consulting `not_applicable` sees "no reference supplied" rather
        # than a game-applicability-shaped message ("P8 is defined for Dark
        # Hex only") on a real Dark Hex cell whose only problem is a missing
        # artefact.
        # Worded "for this game", not "for this run": through property_table
        # (the caller that writes properties.csv), this scorer is called once
        # per cell with a pre-filtered singleton dict, so a reader seeing this
        # message beside other rows from the same run that do carry real
        # CFR-referenced scores must not read it as "the whole run lacked
        # references".
        for game_name in df["game"].unique().to_list():
            if GAME_FAMILIES.get(game_name, game_name) == "darkhex":
                out["not_applicable"][
                    game_name
                ] = "no CFR reference or game object was supplied for this game"
        return out

    for game_name in df["game"].unique().to_list():
        # .get with fallback-to-self (not the raising game_family()): offline
        # scorers must tolerate foreign game values in the Parquet (e.g.
        # synthetic test fixtures); registry keys are covered - the map
        # covers every GAMES key and
        # run_match validates game_key at write time.
        family = GAME_FAMILIES.get(game_name, game_name)
        if family != "darkhex":
            continue
        game_obj = games.get(game_name)
        cfr_pol = cfr_policies.get(game_name)
        if game_obj is None or cfr_pol is None:
            # This specific darkhex-family game has no entry in `games`/
            # `cfr_policies` -- e.g. `darkhex_3x3`, which has no full
            # reference. Distinct from the
            # dicts being absent entirely (handled above) and from a
            # PARTIAL reference (below): this game has no reference at all
            # yet, which is a missing-artefact state, not a game-
            # applicability one.
            out["not_applicable"][
                game_name
            ] = f"no CFR reference available for {game_name} yet"
            continue

        if _is_partial_reference(cfr_pol):
            # No silent fallbacks: record a reason, never NaN (a
            # skip signal elsewhere in this codebase) and never a bare
            # `continue` - a full-tree best-response is not defined against a
            # policy with off-support gaps, and the tree is intractable at
            # this depth.
            out["not_applicable"][game_name] = (
                f"partial reference ({cfr_pol.n_infosets} info-sets): a full-tree "
                "best-response is not defined against a policy with off-support "
                "gaps, and the tree is intractable at this depth"
            )
            continue

        # A worlds map PER seat, built lazily: a fixed-seat arm must not pay for
        # the seat it never occupied. At darkhex 2x3, one
        # `gather_worlds` call each: seat 0 is 0.177 s over 2,053 info-states
        # and 11,881 worlds, seat 1 is 0.213 s over 5,193 info-states and
        # 38,428 worlds (the seats are not symmetric in Dark Hex, P1 moves
        # second into a partly-revealed board). Cached per (game, seat) here and
        # never per row. `gather_worlds` is seat-general and its posteriors were
        # verified by full-reach Bayes to 1.3e-15.
        worlds_by_seat: dict[int, dict] = {}
        # Cache per (seat, info) (V(a|I), VoI*) - independent of the action
        # taken, but not of the seat: the values are read at the acting seat's
        # index and Dark Hex is zero-sum.
        cache: dict[tuple[int, str], tuple[dict, float]] = {}

        report = drop_fallback_rows(
            actor_rows(
                df.filter(pl.col("game") == game_name),
                actor,
                legacy=pl.col("prompt").is_not_null(),
            )
        )
        shortfalls: list[float] = []
        captures: list[int] = []
        vois: list[float] = []
        n_nodes = 0
        # Every row that leaves the loop early leaves through one of these,
        # never through a bare `continue`.
        missed_by_seat: dict[int, dict] = {}
        n_unreachable = 0
        n_low_voi = 0
        n_action_absent = 0
        n_null_action = 0
        # The (seat, info) pairs a shortfall was actually recorded at, for
        # `reference_visit_share`. A set, not a count: the share is over
        # distinct info-sets weighted by the REFERENCE's occupancy, and the
        # agent's own row multiplicity must not leak into it.
        scored_infos: set[tuple[int, str]] = set()
        # The distinct (seat, info) pairs this arm's rows reached in the reference's worlds map,
        # whatever their value of information: an arm that plays one path visits few, which is
        # what a zero informative count on a non-degenerate board comes down to.
        visited_infos: set[tuple[int, str]] = set()
        for row in report.kept.iter_rows(named=True):
            info = row["info_set"]
            # The acting seat, which is the seat the estimand belongs to. Not
            # `lm_seat`: that names the seat AGENT0 occupies (logging/schema.py)
            # and is the opponent's seat whenever the scored agent is agent1.
            seat = int(row["player_to_act"])
            if seat not in worlds_by_seat:
                worlds_by_seat[seat] = gather_worlds(game_obj, cfr_pol, seat=seat)
            wl = worlds_by_seat[seat].get(info)
            if not wl:
                miss = missed_by_seat.setdefault(
                    seat, {"n_rows": 0, "info_sets": set()}
                )
                miss["n_rows"] += 1
                miss["info_sets"].add(info)
                continue
            n_nodes += 1
            visited_infos.add((seat, info))
            if (seat, info) not in cache:
                cache[(seat, info)] = _voi_and_action_values(wl, cfr_pol, seat=seat)
            v_a, voi = cache[(seat, info)]
            vois.append(voi)
            if not v_a:
                # Total opponent reach 0 over every world of this info-state:
                # the reference never plays into it, so there is no posterior to
                # weight the continuation values by. Distinct from "no VoI here".
                n_unreachable += 1
                continue
            if voi <= voi_threshold:
                n_low_voi += 1
                continue
            if row["parsed_action"] is None:
                n_null_action += 1
                continue
            a_lm = int(row["parsed_action"])
            if a_lm not in v_a:
                # The model's action has no value to compare: scoring it as the worst
                # action would turn a row this scorer cannot read into a confident miss.
                n_action_absent += 1
                continue
            best = max(v_a.values())
            v_lm = v_a[a_lm]
            shortfalls.append((best - v_lm) / max(voi, epsilon))
            captures.append(int(best - v_lm <= epsilon))
            scored_infos.add((seat, info))

        if missed_by_seat:
            out["discarded_rows"][game_name] = {
                "n_rows": sum(m["n_rows"] for m in missed_by_seat.values()),
                "by_seat": {
                    str(s): {
                        "n_rows": m["n_rows"],
                        "distinct_info_sets": len(m["info_sets"]),
                        "worlds_map_info_sets": len(worlds_by_seat[s]),
                    }
                    for s, m in sorted(missed_by_seat.items())
                },
                "reason": (
                    "the row's `info_set` is absent from the worlds map built "
                    "for its acting seat, so no posterior over hidden boards "
                    "exists for it and V(a|I) is undefined. The map enumerates "
                    "the info-states of the CONFIGURED game, so a miss means "
                    "the row was logged on a different board than the reference "
                    "was solved for, or its info-state string was written by a "
                    "different OpenSpiel version. Counted per seat because the "
                    "key space is per seat."
                ),
            }
        excluded: dict[str, dict] = {}
        if n_low_voi:
            excluded["voi_at_or_below_threshold"] = {
                "n_rows": n_low_voi,
                "reason": (
                    f"VoI*(I) <= voi_threshold={voi_threshold}: information is "
                    "worth nothing at this node, so it does not test the "
                    "property. The definition excludes these deliberately, and the exclusion is "
                    "logged so it is not read as a pass."
                ),
            }
        if n_action_absent:
            excluded["action_absent_from_value_dict"] = {
                "n_rows": n_action_absent,
                "reason": (
                    "the row's parsed action is not a key of V(a|I), the value map built "
                    "over the info-state's legal actions, so there is no value to compare "
                    "it with. It is excluded and counted rather than scored as the worst "
                    "action, which would read an unreadable row as a confident miss."
                ),
            }
        if n_unreachable:
            excluded["unreachable_info_set"] = {
                "n_rows": n_unreachable,
                "reason": (
                    "total opponent reach is 0 over every world of this "
                    "info-state, so the posterior mu(w|I) is empty. The scorer "
                    "does not fabricate a uniform "
                    "(darkhex_posterior.py: no silent fallbacks)."
                ),
            }
        if excluded:
            out["excluded_rows"][game_name] = excluded

        out["n_null_action"][game_name] = n_null_action
        out["dropped_rate"][game_name] = report.rate
        out["dropped_detail"][game_name] = report.detail()
        out["p8_shortfall"][game_name] = (
            float(np.mean(shortfalls)) if shortfalls else float("nan")
        )
        out["capture_rate"][game_name] = (
            float(np.mean(captures)) if captures else float("nan")
        )
        out["n_informative"][game_name] = len(shortfalls)
        out["n_nodes"][game_name] = n_nodes
        out["n_visited_infosets"][game_name] = len(visited_infos)
        out["mean_voi"][game_name] = float(np.mean(vois)) if vois else float("nan")
        out["n_shortfall_gt1"][game_name] = sum(
            1 for s in shortfalls if s > 1.0 + epsilon
        )
        if worlds_by_seat:
            # The board's own informative-node count, over
            # every info-state in the worlds map for every seat THIS ARM
            # occupied -- not just the ones this arm's own rows visited.
            # Reuses `cache` for any (seat, info) already scored above;
            # computes and caches the rest. Deliberately scoped to occupied
            # seats only (never "both seats regardless"): seats are not
            # symmetric in Dark Hex (module docstring above), so a seat-1-
            # only arm must get seat-1's own count, not seat-0's.
            n_board_informative = 0
            for seat_k, wl_map in worlds_by_seat.items():
                for info_k, wl_k in wl_map.items():
                    if (seat_k, info_k) not in cache:
                        cache[(seat_k, info_k)] = _voi_and_action_values(
                            wl_k, cfr_pol, seat=seat_k
                        )
                    _, voi_k = cache[(seat_k, info_k)]
                    if voi_k > voi_threshold:
                        n_board_informative += 1
            out["n_board_informative"][game_name] = n_board_informative
        if scored_infos:
            # One extra tree walk per OCCUPIED seat, built here rather than
            # beside the worlds map so a game that produced no informative node
            # never pays for it. About 0.19-0.23 s per seat
            # at darkhex 2x3, the same order as `gather_worlds` itself.
            visits = {
                s: reference_visit_weights(game_obj, cfr_pol, s)
                for s in {seat for seat, _ in scored_infos}
            }
            total = sum(sum(w.values()) for w in visits.values())
            scored_mass = sum(
                visits[seat].get(info, 0.0) for seat, info in scored_infos
            )
            out["reference_visit_share"][game_name] = (
                float(scored_mass / total) if total > 0.0 else float("nan")
            )
        out["available"] = True

    return out
