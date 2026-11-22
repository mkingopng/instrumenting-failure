"""
The prescribed belief-reporting agent: uniform-random actions and a belief mixed toward uniform.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pyspiel

from lmopt.agents.base import SeatAware
from lmopt.elicitation.posterior import compute_true_posterior


def uniform_over_other_cards(true_posterior: np.ndarray, own_card: int) -> np.ndarray:
    """
    the uniform distribution over every card except the observer's own
    :param true_posterior: the exact posterior, one entry per card (np.ndarray)
    :param own_card: the observer's card index (int)
    :returns: an array shaped like the posterior with mass 1/(n-1) off the own card (np.ndarray)
    """
    out = np.full(len(true_posterior), 1.0 / (len(true_posterior) - 1))
    out[own_card] = 0.0
    return out


class PrescribedReporter(SeatAware):
    """
    plays a uniformly random legal action and states `(1 - lam) * true + lam * uniform`

    The stated belief departs from the true posterior by exactly `lam * (uniform - true)`, so the
    per-row total variation, per-pair error and uninformed-baseline margin it earns are known
    functions of the information state and `lam`. It sets `last_elicited_belief` the way the
    language-model agent does, and nothing else the scorers read.

    :param oracle_policy: the reference policy the logged true posterior is computed under (object)
    :param lam: the weight on the uniform belief, in [0, 1] (float)
    :param seed: seeds the action choice (int)
    """

    backend = "cpu"
    requires_model = False

    def __init__(self, oracle_policy, lam: float, seed: int = 0) -> None:
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam!r}")
        self._oracle = oracle_policy
        self._lam = lam
        self._rng = np.random.default_rng(seed)
        self._game: Optional[pyspiel.Game] = None
        self.last_elicited_belief: Optional[str] = None

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        self._game = game
        self._seat = seat

    def act(self, state: pyspiel.State) -> int:
        seat = state.current_player()
        true = compute_true_posterior(self._game, state, seat, self._oracle)
        own = state.history()[seat]
        stated = (1.0 - self._lam) * true + self._lam * uniform_over_other_cards(
            true, own
        )
        self.last_elicited_belief = json.dumps(
            {str(i): float(p) for i, p in enumerate(stated)}
        )
        return int(self._rng.choice(state.legal_actions()))

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        return None

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        pass
