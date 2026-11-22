"""Reach-weighted true posterior for Goofspiel P7 - live.

Status: enabled. ``GOOFSPIEL_P7_ENABLED`` is True, so this module is wired
into the match runner (true_posterior for P0 bid nodes) and analyse (P7a/P7b).

Estimand: at a P0 decision node the model predicts the opponent's
*current-round bid* distribution ``q[b]``. The true reference is the predictive
distribution over P1's current bid, marginalised over the worlds consistent with
P0's information:

    true_posterior[b] = Σ_s w(s) · π_opp(b | I₁ at s),   w(s) ∝ opp_reach(s)

where the sum is over P0 decision states ``s`` sharing P0's info-state, and
``opp_reach(s)`` is the product of P1's CFR+ action probabilities along P1's past
moves to ``s`` (chance reach = 1).

Preconditions (load-bearing - see test_loaders.py):
  * ``points_order=descending`` ⇒ no chance player ⇒ chance reach = 1 exactly.
  * perfect-recall info-states ⇒ P0's own reach cancels in the normalisation,
    so it is correct to weight by opponent reach alone.

Bayesian content is round-dependent: from round 2 the outcomes constrain the
consistent worlds (genuine inference); at round 1 the distribution is P1's CFR+
opening mix (pure prediction, no updating).
"""

from __future__ import annotations

import pyspiel

# Gate: Goofspiel P7 is live in the pipeline.
GOOFSPIEL_P7_ENABLED = True

# Hard cap: the exact tree walk is enumerable only for small decks.
# N=4 -> 576 leaves, N=5 -> 14,400. N=6 -> ~500k; refuse rather than hang.
_MAX_NUM_CARDS = 5


def _check_seat(seat: int) -> int:
    """Validate an observing seat. Raises on anything but 0 or 1."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    return seat


def goofspiel_all_posteriors(
    game: pyspiel.Game,
    opp_policy,
    seat: int = 0,
    num_cards: int = 4,
) -> dict[str, dict[int, float]]:
    """All of ``seat``'s reach-weighted bid posteriors in a single tree walk.

    Returns ``{info_str(seat): {bid_action: probability}}``: at each of the
    observing seat's decision nodes, the distribution over the OPPONENT's
    current-round bid given that seat's information, marginalised over the
    consistent worlds and weighted by the opponent's reach. Walks the tree once
    (a per-node walk would repeat it for every call, which is wasteful when the match
    runner needs the posterior at every node).

    The two seats read the same estimand off different sides of the round.
    The wrapped game serialises each simultaneous round as player 0 then player
    1, so:

    * at a seat-0 node the opponent has not yet committed this round. Its bid
      distribution is read off its policy at the child node - the predictive
      form.
    * at a seat-1 node the opponent has committed this round; the world names its
      bid, and seat 1 does not observe it (at a P1 node
      ``information_state_string(1)`` is identical across every P0 bid). The
      contribution is therefore a point mass on that bid, and the reach weighting
      over the consistent worlds is what performs the marginalisation.

    These are the same object, not an approximation of one by the other: the
    world reach factorises as ``prefix_reach * pi_opp(bid | prefix)``, so summing
    the reach-weighted point masses reproduces the reach-weighted mixture of the
    opponent's policies. ``tests/test_goofspiel_posterior.py`` computes each seat
    in the other's formulation and asserts they agree.

    A ``seat`` outside {0, 1} raises (no silent fallbacks): a
    defaulted seat would score the wrong player's information in silence.
    """
    opponent = 1 - _check_seat(seat)
    deck_size = max(game.num_distinct_actions(), num_cards)
    if deck_size > _MAX_NUM_CARDS:
        raise ValueError(
            f"goofspiel_all_posteriors: deck size {deck_size} exceeds the exact "
            f"enumeration cap {_MAX_NUM_CARDS}; the tree is too large to walk."
        )

    # {info_str(seat): [(opp_reach, opponent_current_bid_dist), ...]}
    contribs: dict[str, list[tuple[float, dict[int, float]]]] = {}

    def _walk(state: pyspiel.State, opp_reach: float, opp_pending: int | None) -> None:
        """``opp_pending`` is the action the opponent took at its most recent
        node, i.e. its bid for the round now in progress (None before it has
        moved at all). Only seat 1 reads it; for seat 0 the opponent's most
        recent node belongs to the previous round and the value is unused.
        """
        if state.is_terminal():
            return
        if state.is_chance_node():
            # descending order removes chance; reach = 1 if any chance survives.
            for action, _prob in state.chance_outcomes():
                _walk(state.child(action), opp_reach, opp_pending)
            return
        player = state.current_player()
        if player == seat:
            info = state.information_state_string(seat)
            if seat == 0:
                # The opponent moves next: read its mix at the resulting node.
                # imp_info => the opponent's info-state (and so its policy) does
                # not depend on which action the observer takes here.
                child = state.child(state.legal_actions()[0])
                if child.current_player() == opponent:
                    dist = opp_policy.action_probabilities(child, opponent)
                    contribs.setdefault(info, []).append((opp_reach, dict(dist)))
            elif opp_pending is not None:
                # The opponent already bid this round; this world names it.
                contribs.setdefault(info, []).append((opp_reach, {opp_pending: 1.0}))
            for action in state.legal_actions():
                # The observer's own reach cancels, and its move closes the
                # round, so no opponent bid is pending in the child.
                _walk(state.child(action), opp_reach, None)
        else:  # the opponent: accumulate its reach and record its bid
            probs = opp_policy.action_probabilities(state, opponent)
            for action in state.legal_actions():
                _walk(
                    state.child(action),
                    opp_reach * probs.get(action, 0.0),
                    action,
                )

    _walk(game.new_initial_state(), 1.0, None)

    result: dict[str, dict[int, float]] = {}
    for info, parts in contribs.items():
        total = sum(w for w, _ in parts)
        if total <= 0.0:
            continue
        posterior: dict[int, float] = {}
        for w, dist in parts:
            weight = w / total
            for card, p in dist.items():
                posterior[card] = posterior.get(card, 0.0) + weight * p
        result[info] = posterior
    return result
