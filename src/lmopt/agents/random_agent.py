"""Uniform-random agent, absolute lower anchor."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyspiel

from lmopt.agents.base import SeatAware


class RandomAgent(SeatAware):
    backend = "cpu"
    requires_model = False

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        self._seat = seat

    def act(self, state: pyspiel.State) -> int:
        legal = state.legal_actions()
        return int(self._rng.choice(legal))

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        return None

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        pass


# todo: this looks like it can be better
