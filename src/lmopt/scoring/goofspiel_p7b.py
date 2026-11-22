"""Goofspiel P7b - action-vs-own-belief under a CFR+ continuation.

Status: live. The non-degenerate continuation is wired - see
``goofspiel_br_set`` below, which best-responds to the stated belief under the
CFR+ continuation over reach-weighted worlds. Consumed in production by
``scoring/property7.py`` (a function-scoped import).

This does not reuse the Kuhn/Leduc BR-to-own-belief machinery - that path is
hard-coded to a private-card deal (``my_card = history[seat]``, two-card
counterfactual states) which Goofspiel has no analogue of. Goofspiel needs a
genuinely simultaneous best-response.

Given the model's stated belief ``q`` over the opponent's current bid, the
simultaneous expected value of bidding ``b0`` for the prize ``v`` this round is

    EV(b0 | q) = Σ_b q[b] · ( immediate(b0, b, v) + V_cont(b0, b) )

P7b = fraction of nodes where the executed bid equals ``argmax_{b0} EV(b0 | q)``.

Critical ESTIMAND NOTE - the continuation value is not optional here.
With ``V_cont ≡ 0`` (purely myopic) the immediate outcome is monotone in the
bid: a higher card weakly beats a lower one against *every* opponent bid, so the
best response is always the highest legal card regardless of the belief.
Myopic P7b is therefore degenerate - it measures "does the model always dump its
highest card," not action-vs-belief consistency. The continuation (the
opportunity cost of spending a high card now versus saving it for a high future
prize) is what makes the best response belief-dependent and the metric
meaningful. The continuation must be non-trivial before any Goofspiel P7b number
is reported - see ``goofspiel_br_set`` below for the CFR+-continuation
implementation wired into production.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Non-degenerate P7b: CFR+-continuation best response over reach-weighted worlds
# ---------------------------------------------------------------------------
#
# A purely myopic continuation (V_cont = 0) is degenerate (highest card always
# wins, belief-independent). The real Goofspiel P7b uses the CFR+ continuation value of the
# state reached after both players bid. The accumulated + immediate terms are the
# same across the candidate bids b0 (P0's outcome history is observed under
# imp_info, and immediate(b0,b1) is folded into the resolved state's returns), so
# the best response reduces to
#
#     BR(q) = argmax_{b0}  Σ_w (reach_w / W) · Σ_{b1 ∈ P1hand(w)} q[b1]
#                                              · V_cfr(state after b0 then b1 in w)
#
# where the worlds w are the P0-info-consistent concrete states, reach-weighted by
# the opponent's CFR+ reach (same weighting as the true posterior), and V_cfr is
# the CFR+ reference value to P0 (both players play the reference thereafter).

_BR_TIE_TOL = 1e-9


def gather_p0_worlds(game, opp_policy) -> dict:
    """One tree walk -> ``{p0_info_str: [(state, opp_reach), ...]}``.

    The concrete states consistent with each P0 info-state, weighted by the
    opponent's CFR+ reach - the world set the P7b continuation averages over.
    States are independent copies (serialize/deserialize) so callers may advance
    them freely.
    """
    worlds: dict[str, list[tuple[object, float]]] = {}

    def _walk(state, opp_reach: float) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                _walk(state.child(a), opp_reach)
            return
        player = state.current_player()
        if player == 0:
            info0 = state.information_state_string(0)
            worlds.setdefault(info0, []).append(
                (game.deserialize_state(state.serialize()), opp_reach)
            )
            for a in state.legal_actions():
                _walk(state.child(a), opp_reach)
        else:
            probs = opp_policy.action_probabilities(state, 1)
            for a in state.legal_actions():
                _walk(state.child(a), opp_reach * probs.get(a, 0.0))

    _walk(game.new_initial_state(), 1.0)
    return worlds


def gather_p1_prefix_worlds(game, opp_policy) -> dict:
    """One tree walk -> ``{p1_info_str: [(prefix_state, prefix_reach), ...]}``.

    The seat-1 analogue of :func:`gather_p0_worlds`. ``prefix_state`` is a
    P0-to-act state, immediately before the current round's P0 bid (P0 is "the
    opponent" from seat 1's point of view); ``prefix_reach`` is the reach
    weight accumulated getting to that prefix from P0's prior-round bids only
    - it never includes the probability of the current round's bid, which is
    what ``prefix_state.legal_actions()`` enumerates for the caller to weight
    separately (e.g. against a stated belief or the opponent policy itself).

    Keyed by ``information_state_string(1)`` of the resulting P1 node rather
    than by P0's own info-state: by ``imp_info`` (every P0 bid choice from a
    given prefix leads to a P1 node with the identical info-state, since P1
    does not observe P0's simultaneous bid), any one child's info-string names
    the group correctly, and a single prefix can be reached along more than
    one path so a given key can collect several prefix states.

    States are independent copies (serialize/deserialize), matching
    :func:`gather_p0_worlds`, so callers may advance them freely.
    """
    worlds: dict[str, list[tuple[object, float]]] = {}

    def _walk(state, opp_reach: float) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                _walk(state.child(a), opp_reach)
            return
        player = state.current_player()
        if player == 0:
            # Prefix node: P0 about to bid this round. Record with the reach
            # carried into this node (prior rounds only) - imp_info means any
            # child names the group's p1_info correctly.
            child = state.child(state.legal_actions()[0])
            assert child.current_player() == 1, (
                "gather_p1_prefix_worlds assumes every P0 bid this round "
                "lands on a P1 node (imp_info); a game config where a round "
                "can end or skip P1 here would key entries on the wrong "
                "info-string instead of failing loudly."
            )
            p1_info = child.information_state_string(1)
            worlds.setdefault(p1_info, []).append(
                (game.deserialize_state(state.serialize()), opp_reach)
            )
            # Descend, folding in this round's P0-bid probability so that
            # deeper (later-round) prefixes see it as part of their own
            # "prior rounds" reach - never as part of this prefix's reach.
            probs = opp_policy.action_probabilities(state, 0)
            for a in state.legal_actions():
                _walk(state.child(a), opp_reach * probs.get(a, 0.0))
        else:
            # player == 1: P1's own move closes the round; its reach cancels
            # (same reasoning as gather_p0_worlds's observer-reach cancelling).
            for a in state.legal_actions():
                _walk(state.child(a), opp_reach)

    _walk(game.new_initial_state(), 1.0)
    return worlds


def goofspiel_br_set(
    worlds_for_info: list, cfr_pol, belief: dict[int, float]
) -> list[int]:
    """
    Best-response bid set at a P0 info-state given stated bid-belief ``belief``.
    :param worlds_for_info: ``[(state, opp_reach)]`` for the target P0 info-state
        (from :func:`gather_p0_worlds`).
    :param cfr_pol: CFR+ reference policy (``action_probabilities(state, player)``).
    :param belief: ``{bid_action: probability}``, the model's stated distribution
        over the opponent's current bid.
    :returns: The list of P0 bid actions attaining the max reach-/belief-weighted CFR+
        continuation value (ties credited). Empty if the info-state is unreachable.
    """
    from open_spiel.python.algorithms.expected_game_score import policy_value

    if not worlds_for_info:
        return []
    total_reach = sum(w for _, w in worlds_for_info)
    if total_reach <= 0.0:
        return []

    legal_b0 = worlds_for_info[0][0].legal_actions()  # P0 hand: same across worlds
    evs: dict[int, float] = {b0: 0.0 for b0 in legal_b0}

    for state, reach in worlds_for_info:
        if reach <= 0.0:
            continue
        w = reach / total_reach
        for b0 in legal_b0:
            after_b0 = state.child(b0)  # P1 to act
            for b1 in after_b0.legal_actions():
                q = belief.get(b1, 0.0)
                if q <= 0.0:
                    continue
                after_b1 = after_b0.child(b1)
                evs[b0] += w * q * policy_value(after_b1, [cfr_pol, cfr_pol])[0]

    best = max(evs.values())
    return [b0 for b0, e in evs.items() if e >= best - _BR_TIE_TOL]


def goofspiel_br_set_seat1(
    worlds_for_info: list, cfr_pol, belief: dict[int, float]
) -> list[int]:
    """
    Best-response bid set at a P1 info-state given a stated belief ``belief``
    over the opponent's already-made, hidden current-round bid.

    The seat-1 analogue of :func:`goofspiel_br_set`. Seat 1 acts strictly after
    seat 0 within a round, so seat 1's current-round decision is not
    simultaneous with seat 0's in the way seat 0's own decision is: by the time
    seat 1 chooses ``b1``, seat 0's ``b0`` is already fixed (just hidden from
    seat 1). The model's stated belief ``q`` over that already-made bid replaces
    the opponent's own current-round bid probability - it is not an extra factor
    alongside it, which would double-count it (see
    :func:`gather_p1_prefix_worlds`'s docstring).

        EV(b1 | q) = Σ_prefix (prefix_reach / Z) · Σ_b0 q[b0] · V_1(prefix.child(b0).child(b1))
        BR(q) = argmax_b1 EV(b1 | q)

    where ``V_1(state) = policy_value(state, [cfr_pol, cfr_pol])[1]`` - index 1
    (seat 1's own value), never index 0 (see :func:`goofspiel_br_set`'s ``[0]``
    for the seat-0 mirror; under ``point_difference`` these are exact negations
    of each other, so reading the wrong index flips every BR).
    :param worlds_for_info: ``[(prefix_state, prefix_reach)]`` for the target P1
        info-state (from :func:`gather_p1_prefix_worlds`). ``prefix_state``
        is P0-to-act, immediately before the current round's P0 bid;
        ``prefix_reach`` excludes that bid's probability by construction.
    :param cfr_pol: CFR+ reference policy (``action_probabilities(state, player)``).
    :param belief: ``{bid_action: probability}``, the model's stated distribution
        over the opponent's (P0's) already-made current-round bid ``b0``.
    :returns: The list of P1 bid actions (``b1``) attaining the max reach-/belief-
        weighted CFR+ continuation value (ties credited). Empty if the
        info-state is unreachable.
    """
    from open_spiel.python.algorithms.expected_game_score import policy_value

    if not worlds_for_info:
        return []
    total_reach = sum(w for _, w in worlds_for_info)
    if total_reach <= 0.0:
        return []

    # P1's legal current-round bids (P1 hand) - same regardless of which b0
    # child we probe (imp_info: verified against a real fixture during a
    # trace, not merely assumed), so any (prefix, b0) pair names the group.
    probe_prefix = worlds_for_info[0][0]
    legal_b1 = probe_prefix.child(probe_prefix.legal_actions()[0]).legal_actions()
    evs: dict[int, float] = {b1: 0.0 for b1 in legal_b1}

    for prefix_state, reach in worlds_for_info:
        if reach <= 0.0:
            continue
        w = reach / total_reach
        for b0 in prefix_state.legal_actions():
            q = belief.get(b0, 0.0)
            if q <= 0.0:
                continue
            after_b0 = prefix_state.child(b0)  # P1 to act, b0 now fixed (hidden)
            for b1 in legal_b1:
                after_b1 = after_b0.child(b1)
                evs[b1] += w * q * policy_value(after_b1, [cfr_pol, cfr_pol])[1]

    best = max(evs.values())
    return [b1 for b1, e in evs.items() if e >= best - _BR_TIE_TOL]
