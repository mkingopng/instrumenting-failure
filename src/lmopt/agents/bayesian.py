"""Bayesian agent - exact posterior + best response."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyspiel

from lmopt.agents.base import SeatAware
from lmopt.games.belief_br import per_state_max_action_values, pooled_action_values
from lmopt.games.loaders import independent_deal

#: How the agent values its own later moves. "best_response" is the pooled best response
#: and the default. "as_collected_v7" chooses each later move with the opponent's card
#: revealed: it is not a best response, and exists only to reproduce the agent's play in
#: data collected with it.
_CONTINUATIONS = {
    "best_response": pooled_action_values,
    "as_collected_v7": per_state_max_action_values,
}


class BayesianAgent(SeatAware):
    """Computes an exact Bayesian posterior over the opponent's hidden card and
    best-responds against that posterior.

    The opponent's policy must be known at construction time (Policy-like object
    with action_probabilities(state, player_id) -> dict[int, float]): a fixed,
    non-adapting policy (CFR, Uniform, or classical). The working reference is
    CFR+ - ``opponent_policy`` is the ``solve_cfr(..., plus=True)`` average
    policy, so the Bayesian likelihood model is generated from the CFR+
    reference. On-path CFR and CFR+ agree, so this coupling is exact where it
    matters; see
    ``tests/test_bayesian_posterior.py::TestBayesianLikelihoodIsCFRPlus``.

    This agent is the headline discriminator and must implement exact
    posterior + best response - it must not be stubbed.

    Key implementation detail: all EV computations use *counterfactual states*
    built from a fresh game episode with the hypothetical opponent card.  This
    ensures that opponent policy lookups use the correct information state
    (which includes the opponent's card in Kuhn poker) rather than the actual
    state where the opponent's card may differ from the hypothesis.

    Its own later moves are valued by ``lmopt.games.belief_br.pooled_action_values``:
    each later information set takes one action, chosen over every history it cannot
    tell apart. That is the same operator the P7b scorer uses.
    """

    backend = "cpu"
    requires_model = False

    def __init__(
        self, opponent_policy, seed: int = 0, continuation: str = "best_response"
    ) -> None:
        """
        a Bayesian agent best-responding to its posterior under a fixed opponent model
        :param opponent_policy: the opponent's policy, `action_probabilities(state, player)`
        :param seed: the agent's random seed (int)
        :param continuation: "best_response" (default) or "as_collected_v7", the legacy
            per-state rule that is not a best response (str)
        :raises ValueError: on any other continuation
        """
        if continuation not in _CONTINUATIONS:
            raise ValueError(
                f"continuation must be one of {sorted(_CONTINUATIONS)}, got {continuation!r}"
            )
        self._continuation = continuation
        self._opp_policy = opponent_policy
        self._rng = np.random.default_rng(seed)
        self._game: Optional[pyspiel.Game] = None
        self._seat: Optional[int] = None
        self._num_cards: int = 0
        self._hand_count: int = 0
        # True on nodes where the agent could not compute a posterior and fell
        # back to CFR argmax (e.g. Goofspiel, no fixed private deal), and on nodes
        # where every hypothesis gave the observed actions zero probability so the
        # posterior is uniform by substitution. Logged per node so the degradation
        # is not silent (no silent fallbacks).
        self.last_fallback_used: bool = False

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    def _bound_game(self) -> pyspiel.Game:
        """
        :returns: the game `reset` bound (pyspiel.Game)
        :raises RuntimeError: when `reset` has not run
        """
        if self._game is None:
            raise RuntimeError("BayesianAgent used before reset() bound a game")
        return self._game

    def _bound_seat(self) -> int:
        """
        :returns: the seat `reset` bound (int)
        :raises RuntimeError: when `reset` has not run
        """
        if self._seat is None:
            raise RuntimeError("BayesianAgent used before reset() bound a seat")
        return self._seat

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        self._game = game
        self._seat = seat
        self._hand_count = 0
        init = game.new_initial_state()
        # Bayesian posterior requires an initial private-card chance deal.
        # Games without one (e.g. Goofspiel) fall back to CFR argmax in act().
        self._bayesian_supported = init.is_chance_node()
        self._num_cards = len(init.chance_outcomes()) if self._bayesian_supported else 0
        # Deal model: without replacement (Kuhn/Leduc - opponent cannot hold your
        # card) vs independent (Liar's Dice - opponent can roll your face). Drives
        # the hypothesis set in _compute_posterior. Detected from deck structure,
        # not a game-name fork (config over forks).
        self._independent_deal = (
            independent_deal(game) if self._bayesian_supported else False
        )

    def act(self, state: pyspiel.State) -> int:
        if not self._bayesian_supported:
            # Goofspiel has no fixed private cards; fall back to CFR argmax.
            # Flagged so the Parquet records that no posterior was computed here.
            self.last_fallback_used = True
            probs = self._opp_policy.action_probabilities(state, state.current_player())
            legal = state.legal_actions()
            best = max((probs.get(a, 0.0) for a in legal), default=0.0)
            for a in legal:
                if probs.get(a, 0.0) >= best:
                    return a
            return legal[0]
        self.last_fallback_used = False
        posterior = self._compute_posterior(state)
        return self._best_respond(state, posterior)

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        if not self._bayesian_supported:
            return None
        return self._compute_posterior(state)

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        self._hand_count += 1

    # ------------------------------------------------------------------
    # Posterior computation
    # ------------------------------------------------------------------

    def _post_deal_history(self, state: pyspiel.State) -> list[tuple[bool, int]]:
        """Classify post-deal history into (is_chance, action) pairs.

        Skips the initial private-card chance deal; everything after is
        either a player action (is_chance=False) or a public chance event
        such as the Leduc community card (is_chance=True).
        """
        full_history = state.history()
        s = self._bound_game().new_initial_state()

        # Advance past the initial dealing (all leading chance nodes).
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

    def _compute_posterior(self, state: pyspiel.State) -> np.ndarray:
        """Return P(opponent_card | public_history), shape (num_cards,).

        Own-card entry is 0.0. When every hypothesis gives the observed actions zero
        probability the posterior is uniform over the hypotheses and
        `last_fallback_used` is set, so the row reads as a substitution. The
        uniform prior at the first node is the ordinary path (every likelihood is
        1.0) and does not set it. Works for multi-round games (e.g. Leduc) where a public community card
        is revealed mid-game via a second chance node.
        """
        full_history = state.history()
        opp_seat = 1 - self._bound_seat()
        my_card = full_history[self._seat]

        post_deal = self._post_deal_history(state)

        # Cards publicly revealed after initial deal (e.g., Leduc community card).
        public_cards = {action for (is_chance, action) in post_deal if is_chance}

        # Possible opponent cards. Without replacement (Kuhn/Leduc): exclude own
        # card - the opponent cannot hold it. Independent (Liar's Dice): the
        # opponent's die can equal yours, so own value stays in the hypothesis set.
        possible = [
            c
            for c in range(self._num_cards)
            if c not in public_cards and (self._independent_deal or c != my_card)
        ]

        likelihoods = np.array(
            [self._likelihood(my_card, hyp, post_deal, opp_seat) for hyp in possible],
            dtype=float,
        )

        total = likelihoods.sum()
        if total <= 0.0:
            self.last_fallback_used = True
            likelihoods = np.ones(len(possible), dtype=float)
            total = float(len(possible))

        posterior_possible = likelihoods / total

        result = np.zeros(self._num_cards, dtype=float)
        for i, card in enumerate(possible):
            result[card] = posterior_possible[i]
        return result

    def _likelihood(
        self,
        my_card: int,
        hyp_opp_card: int,
        post_deal: list[tuple[bool, int]],
        opp_seat: int,
    ) -> float:
        """P(player_actions | my_card, opp_has hyp_opp_card) under opp_policy.

        post_deal is a list of (is_chance, action) pairs classifying history
        after the initial private-card deal. Public chance events (is_chance=True,
        e.g. community card) are applied to the counterfactual state without
        modifying the likelihood.
        """
        s = self._make_cf_state(my_card, hyp_opp_card, [])

        likelihood = 1.0
        for is_chance, action in post_deal:
            if s.is_terminal():
                break
            if is_chance:
                # Public chance event (e.g. community card): apply, don't multiply.
                s.apply_action(action)
                continue
            acting = s.current_player()
            if acting == opp_seat:
                probs = self._opp_policy.action_probabilities(s, acting)
                likelihood *= probs.get(action, 0.0)
                if likelihood == 0.0:
                    return 0.0
            s.apply_action(action)

        return likelihood

    # ------------------------------------------------------------------
    # Best response
    # ------------------------------------------------------------------

    def _best_respond(self, state: pyspiel.State, posterior: np.ndarray) -> int:
        """
        the first legal action attaining the best value under the posterior
        :param state: the current state (pyspiel.State)
        :param posterior: P(opponent card | history) (np.ndarray)
        :returns: the chosen action (int)
        """
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        seat = self._bound_seat()
        values = _CONTINUATIONS[self._continuation](
            state, seat, posterior, self._opp_policy
        )
        return max(legal, key=values.__getitem__)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_cf_state(
        self, my_card: int, opp_card: int, actions: list[int]
    ) -> pyspiel.State:
        """Build a counterfactual state from a fresh game with the given deal."""
        s = self._bound_game().new_initial_state()
        if self._seat == 0:
            s.apply_action(my_card)
            s.apply_action(opp_card)
        else:
            s.apply_action(opp_card)
            s.apply_action(my_card)
        for a in actions:
            if s.is_terminal():
                break
            s.apply_action(a)
        return s
