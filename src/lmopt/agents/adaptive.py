"""Adaptive reference agent: an empirical best responder with the LM's information.

The P2 (adaptivity) ceiling. The agent models its opponent from what the harness lets an agent see
across hands, the public action sequence of every finished hand and its own payoff, never the
opponent's private card, which is the same record the LM's cross-hand memory holds
(``lmopt.prompting.memory.HandMemory``). The model is a table of the opponent's action counts keyed by
the opponent's seat and the prefix of player actions it had seen when it moved; chance events are not
in the key, so a public chance event (Leduc's community card, Goofspiel's prize) is marginalised. The
table is smoothed towards the CFR+ reference with a pseudo-count, so the agent starts at the
equilibrium and moves towards the opponent's observed behaviour as hands accumulate.

Every ``update_every`` hands the agent recomputes OpenSpiel's ``BestResponsePolicy`` against that model
for the seat it is about to play and plays its argmax; nothing in ``act`` reads an RNG, so
``deterministic_action`` holds. The posterior over the opponent's hidden state is implicit in the
best-response walk, which pools counterfactual reach over each information set.

Scoring is offline and unchanged: an arm collected with ``agent0: adaptive`` is scored by
``score_property2`` with ``actor="adaptive"`` like any other arm, and its shrinkage delta per
(game, opponent) cell is the ceiling the LM arms are read against.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import pyspiel

from lmopt.agents.base import SeatAware

#: Opponent-model key: the opponent's seat and the (player, action) prefix it had seen.
PrefixKey = tuple[int, tuple[tuple[int, int], ...]]


def action_prefix(game: pyspiel.Game, state: pyspiel.State) -> tuple[tuple[int, int], ...]:
    """
    the (player, action) pairs played so far, chance nodes skipped, as ``_build_action_history`` builds them
    :param game: the game (pyspiel.Game)
    :param state: a state of it (pyspiel.State)
    :returns: the prefix (tuple)
    """
    s = game.new_initial_state()
    out: list[tuple[int, int]] = []
    for action in state.history():
        if s.is_chance_node():
            s.apply_action(action)
            continue
        out.append((s.current_player(), action))
        s.apply_action(action)
    return tuple(out)


class PrefixOpponentModel:
    """
    an OpenSpiel-compatible policy for the opponent: observed action counts at each public prefix, smoothed
    towards a prior policy with ``pseudo_count`` prior weight
    """

    def __init__(self, game: pyspiel.Game, prior, pseudo_count: float = 1.0) -> None:
        """
        :param game: the game (pyspiel.Game)
        :param prior: the prior policy, ``action_probabilities(state, player)`` (CFR+ reference)
        :param pseudo_count: weight of the prior in the smoothed estimate (float)
        :raises ValueError: on a non-positive pseudo-count, which would leave an unseen prefix with no distribution
        """
        if pseudo_count <= 0.0:
            raise ValueError(f"pseudo_count must be positive, got {pseudo_count}")
        self._game = game
        self._prior = prior
        self._pseudo = float(pseudo_count)
        self._counts: dict[PrefixKey, Counter] = {}
        self.n_observed = 0

    def observe(self, opp_seat: int, history: list[tuple[int, int]]) -> None:
        """
        record the opponent's decisions of one finished hand
        :param opp_seat: the seat the opponent held in that hand (int)
        :param history: the hand's (action, player) pairs, chance skipped (list)
        :returns: None
        """
        prefix: list[tuple[int, int]] = []
        for action, player in history:
            if player == opp_seat:
                key = (opp_seat, tuple(prefix))
                self._counts.setdefault(key, Counter())[int(action)] += 1
                self.n_observed += 1
            prefix.append((player, action))

    def action_probabilities(self, state: pyspiel.State, player_id: Optional[int] = None) -> dict[int, float]:
        """
        the smoothed empirical distribution over the legal actions at ``state`` for the opponent
        :param state: an opponent decision node (pyspiel.State)
        :param player_id: the player to act, default the state's current player (int | None)
        :returns: ``{action: probability}`` over the legal actions (dict)
        """
        player = state.current_player() if player_id is None else player_id
        legal = state.legal_actions()
        prior = self._prior.action_probabilities(state, player)
        p0 = np.array([prior.get(a, 0.0) for a in legal], dtype=float)
        if p0.sum() <= 0.0:
            p0 = np.ones(len(legal)) / len(legal)
        else:
            p0 = p0 / p0.sum()
        counts = self._counts.get((player, action_prefix(self._game, state)))
        n = np.array([counts.get(a, 0) if counts else 0 for a in legal], dtype=float)
        p = (n + self._pseudo * p0) / (n.sum() + self._pseudo)
        return {a: float(x) for a, x in zip(legal, p, strict=True)}


class AdaptiveAgent(SeatAware):
    """
    best response to a prefix-keyed empirical model of the opponent, recomputed every ``update_every`` hands
    """

    backend = "cpu"
    requires_model = False
    deterministic_action = True

    def __init__(self, prior_policy, update_every: int = 50, pseudo_count: float = 1.0, seed: int = 0,
                 memory_rung: str = "M1") -> None:
        """
        :param prior_policy: the CFR+ reference, the opponent model before any observation and its smoothing prior
        :param update_every: hands between best-response recomputations (int)
        :param pseudo_count: prior weight in the opponent model (float)
        :param seed: recorded for provenance; ``act`` reads no RNG (int)
        :param memory_rung: the memory level stamped on the agent's rows; the agent conditions on every earlier
            hand, so it is never M0, and the analysis reads P2 at M1 and M2 only (str)
        :raises ValueError: on ``update_every`` below 1 or an M0 rung
        """
        if update_every < 1:
            raise ValueError(f"update_every must be at least 1, got {update_every}")
        if memory_rung == "M0":
            raise ValueError("an adaptive agent conditions on earlier hands; its rows cannot carry M0")
        self.memory_rung = memory_rung
        self._prior = prior_policy
        self._update_every = int(update_every)
        self._pseudo = pseudo_count
        self._seed = seed
        self._game: Optional[pyspiel.Game] = None
        self._model: Optional[PrefixOpponentModel] = None
        self._br: dict[int, object] = {}
        self._br_hand: dict[int, int] = {}
        self._hand_count = 0
        self.last_fallback_used = False

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        self._game = game
        self._seat = seat
        self._model = PrefixOpponentModel(game, self._prior, self._pseudo)
        self._br = {}
        self._br_hand = {}
        self._hand_count = 0

    def _bound(self) -> tuple[pyspiel.Game, PrefixOpponentModel]:
        """
        :returns: the game and model ``reset`` bound (tuple)
        :raises RuntimeError: when ``reset`` has not run
        """
        if self._game is None or self._model is None:
            raise RuntimeError("AdaptiveAgent used before reset() bound a game")
        return self._game, self._model

    def _policy_for(self, seat: int):
        """
        the current best response for ``seat``, recomputed when ``update_every`` hands have passed since the last solve
        :param seat: the seat to play (int)
        :returns: the OpenSpiel best-response policy
        """
        from open_spiel.python.algorithms.best_response import BestResponsePolicy

        game, model = self._bound()
        if seat not in self._br or self._hand_count - self._br_hand[seat] >= self._update_every:
            self._br[seat] = BestResponsePolicy(game, seat, model)
            self._br_hand[seat] = self._hand_count
        return self._br[seat]

    def act(self, state: pyspiel.State) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        seat = state.current_player()
        br = self._policy_for(seat)
        action = int(br.best_response_action(state.information_state_string(seat)))
        if action not in legal:
            raise RuntimeError(f"best response returned illegal action {action} at {state.information_state_string(seat)!r}")
        return action

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        return None

    def observe_hand_end(self, history: list[tuple[int, int]], own_payoff: float | None = None) -> None:
        _, model = self._bound()
        if self._seat is None:
            raise RuntimeError("AdaptiveAgent.observe_hand_end before begin_hand bound a seat")
        model.observe(1 - self._seat, history)
        self._hand_count += 1

    @property
    def hands_observed(self) -> int:
        """
        :returns: hands recorded so far (int)
        """
        return self._hand_count
