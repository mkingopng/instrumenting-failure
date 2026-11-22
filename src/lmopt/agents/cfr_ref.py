"""CFR reference agent - wraps a pre-solved OpenSpiel TabularPolicy."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyspiel
from open_spiel.python.policy import Policy

from lmopt.agents.base import SeatAware


class CFRRefAgent(SeatAware):
    """Samples actions from the pre-solved CFR average policy.

    The policy is solved externally (via lmopt.games.cfr.solve_cfr) and passed
    in - this agent never runs CFR itself (imported CFR, never hand-rolled).
    """

    backend = "cpu"
    requires_model = False
    # `act()` below always calls `self._rng.choice(actions, p=weights)`, no
    # branch skips it -- overrides `SeatAware`'s `True` default.
    deterministic_action = False

    def __init__(self, policy: Policy, seed: int = 0) -> None:
        self._policy = policy
        self._rng = np.random.default_rng(seed)
        self.last_fallback_used = False

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        self._seat = seat

    def _probs_and_flag(self, state: pyspiel.State) -> tuple[dict[int, float], bool]:
        """Shared core for ``act`` and ``policy_at``: one place that decides
        whether ``self._policy`` is a partial ``DepthReference`` (has ``lookup``)
        or a fully-enumerated ``TabularPolicy``, so the two call sites cannot
        drift apart on how off-support is detected."""
        lookup = getattr(self._policy, "lookup", None)
        if lookup is not None:  # DepthReference: partial, may be off-support
            probs, off_support = lookup(state)
        else:
            probs = self._policy.action_probabilities(state, state.current_player())
            off_support = False
        return {int(a): float(p) for a, p in probs.items()}, off_support

    def act(self, state: pyspiel.State) -> int:
        probs, off_support = self._probs_and_flag(state)
        # A uniform draw at an info-set the reference never visited is a fallback:
        # record it (no silent fallbacks). Always assigned,
        # so the flag can never be a stale True from an earlier node.
        self.last_fallback_used = off_support
        actions = list(probs.keys())
        weights = [probs[a] for a in actions]
        return int(self._rng.choice(actions, p=weights))

    def policy_at(self, state: pyspiel.State) -> tuple[dict[int, float], bool]:
        """Return ``(cfr distribution, off_support)`` at this state.

        Called by the match runner on every decision node to populate
        cfr_action / cfr_policy / cfr_ref_off_support, regardless of which agent
        is acting. ``off_support`` is always False for a fully-enumerated
        TabularPolicy; only a partial depth reference can set it.
        """
        return self._probs_and_flag(state)

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        return None

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        pass
