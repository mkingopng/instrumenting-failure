"""Standalone true-posterior computation, Leduc-aware."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pyspiel

from lmopt.games.loaders import independent_deal


class PosteriorResult(NamedTuple):
    """A posterior and whether it was COMPUTED or fell back to uniform.

    The uniform fallback below is silent, and on an
    off-path-driven arm it is systematically reachable: a driven opening is
    chosen precisely because the reference reaches it with (near) zero
    probability, so every hypothesis's likelihood can be exactly zero. A
    fallback that reaches the `true_posterior` column unmarked would make P5
    (ECE) and P7a (TVD) measure elicited-vs-uniform while reporting
    elicited-vs-Bayes.
    """

    posterior: np.ndarray
    fell_back: bool


def compute_true_posterior(
    game: pyspiel.Game,
    state: pyspiel.State,
    seat: int,
    opponent_policy,
) -> np.ndarray:
    """Exact Bayesian posterior P(opp_card | state) under the known opponent policy.

    The array alone, for every caller that does not need to know whether the
    uniform fallback fired. :func:`compute_true_posterior_with_provenance` is
    the same computation with that one extra fact; this wrapper keeps the
    plain-array return type for callers that need only the posterior.
    """
    return compute_true_posterior_with_provenance(
        game, state, seat, opponent_policy
    ).posterior


def compute_true_posterior_with_provenance(
    game: pyspiel.Game,
    state: pyspiel.State,
    seat: int,
    opponent_policy,
) -> PosteriorResult:
    """The posterior, plus whether it is a posterior at all.

    Works for multi-round games (e.g. Leduc) where a public community card is
    revealed mid-game via a second chance node. Public cards are excluded from
    the hypothesis space and applied to counterfactual states without multiplying
    the likelihood.

    Returns an array of shape (num_cards,) with own-card and public-card entries
    = 0.0, and ``fell_back=True`` when all likelihoods were zero and the array
    is a uniform over the remaining cards rather than a posterior. The array is
    unchanged in either case: what a caller does with the flag is the caller's
    decision, and the match runner's is in ``harness/match_runner.py`` (log None
    on a driven hand).
    """
    init = game.new_initial_state()
    num_cards = len(init.chance_outcomes())

    full_history = state.history()
    my_card = full_history[seat]
    opp_seat = 1 - seat

    post_deal = _post_deal_history(game, state)
    public_cards = {action for (is_chance, action) in post_deal if is_chance}

    # Without replacement (Kuhn/Leduc): exclude own card. Independent (Liar's
    # Dice): the opponent's die can equal yours, so own value stays in the set.
    indep = independent_deal(game)
    possible = [
        c for c in range(num_cards) if c not in public_cards and (indep or c != my_card)
    ]
    likelihoods = np.array(
        [
            _likelihood(game, seat, my_card, hyp, post_deal, opp_seat, opponent_policy)
            for hyp in possible
        ],
        dtype=float,
    )

    total = likelihoods.sum()
    fell_back = total <= 0.0
    if fell_back:
        likelihoods = np.ones(len(possible), dtype=float)
        total = float(len(possible))

    posterior = likelihoods / total
    result = np.zeros(num_cards, dtype=float)
    for i, card in enumerate(possible):
        result[card] = posterior[i]
    return PosteriorResult(result, fell_back)


def _post_deal_history(
    game: pyspiel.Game,
    state: pyspiel.State,
) -> list[tuple[bool, int]]:
    """Return (is_chance, action) pairs for everything after the initial private deal.

    Advances past the leading chance nodes (initial card deal), then classifies
    each subsequent action as a public chance event (is_chance=True, e.g. Leduc
    community card) or a player action (is_chance=False).
    """
    full_history = state.history()
    s = game.new_initial_state()

    idx = 0
    while idx < len(full_history) and s.is_chance_node():
        s.apply_action(full_history[idx])
        idx += 1

    result: list[tuple[bool, int]] = []
    for h in full_history[idx:]:
        if s.is_terminal():
            break
        result.append((s.is_chance_node(), h))
        s.apply_action(h)
    return result


def _likelihood(
    game: pyspiel.Game,
    seat: int,
    my_card: int,
    hyp_opp_card: int,
    post_deal: list[tuple[bool, int]],
    opp_seat: int,
    opp_policy,
) -> float:
    """P(observed_actions | my_card, opp_has hyp_opp_card) under opp_policy.

    Public chance events (is_chance=True, e.g. community card) are applied to
    the counterfactual state without multiplying the likelihood.
    """
    s = game.new_initial_state()
    if seat == 0:
        s.apply_action(my_card)
        s.apply_action(hyp_opp_card)
    else:
        s.apply_action(hyp_opp_card)
        s.apply_action(my_card)

    likelihood = 1.0
    for is_chance, action in post_deal:
        if s.is_terminal():
            break
        if is_chance:
            s.apply_action(action)
            continue
        acting = s.current_player()
        if acting == opp_seat:
            probs = opp_policy.action_probabilities(s, acting)
            likelihood *= probs.get(action, 0.0)
            if likelihood == 0.0:
                return 0.0
        s.apply_action(action)

    return likelihood
