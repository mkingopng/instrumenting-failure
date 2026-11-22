#
"""Classical Kuhn-poker heuristic agents and the UniformPolicy adapter."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyspiel

from lmopt.agents.base import SeatAware


# Policy adapter

class UniformPolicy:
    """Returns a uniform distribution over legal actions at any state.

    Used as the opponent_policy for BayesianAgent when facing a RandomAgent,
    so the posterior computation stays uniform across opponent types.
    """

    def action_probabilities(
        self, state: pyspiel.State, player_id: int
    ) -> dict[int, float]:
        """
        :param state:
        :param player_id:
        :returns:
        """
        legal = state.legal_actions(player_id)
        p = 1.0 / len(legal)
        return {a: p for a in legal}


# Heuristic agents

class AlwaysBetAgent(SeatAware):
    """
    Bets (action 1) whenever legal; otherwise checks/folds.
    Overly aggressive lower anchor - exposes heterogeneity vs the equilibrium.
    """
    backend = "cpu"
    requires_model = False

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        """
        :param game:
        :param seat:
        :param opponent_id:
        :returns:
        """
        self._seat = seat

    def act(self, state: pyspiel.State) -> int:
        """
        :param state:
        :returns:
        """
        legal = state.legal_actions()
        if 1 in legal:
            return 1
        if 0 in legal:
            return 0
        return legal[0]

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        """
        :param state:
        :returns:
        """
        return None

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        pass


class ThresholdAgent(SeatAware):
    """Bets iff own card ≥ threshold; otherwise checks/folds.

    Default threshold=2 (King-only betting). The card is read from the first
    character of the information-state string, which OpenSpiel's kuhn_poker
    always places there.
    """

    backend = "cpu"
    requires_model = False

    def __init__(self, threshold: int = 2) -> None:
        self._threshold = threshold

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        self._seat = seat

    def act(self, state: pyspiel.State) -> int:
        info_str = state.information_state_string(state.current_player())
        own_card = int(info_str[0])
        legal = state.legal_actions()
        if own_card >= self._threshold and 1 in legal:
            return 1
        return 0

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        return None

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        pass
