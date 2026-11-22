"""Extensive-Form Fictitious Play (XFP) self-play agent, then frozen.

XFP (Heinrich, Lanctot & Silver, 2015, ICML) is the tabular algorithm that
computes average policies via best-response iteration. NFSP (Heinrich & Silver,
2016) approximates the same update with neural networks for large-state games;
for the small games in this study (Kuhn 12 info-states, Leduc ~80, Goofspiel 4)
the exact tabular form is both faster and more reliable. Both carry the same
theoretical guarantee: convergence to Nash Equilibrium in two-player zero-sum
extensive-form games.

The frozen average policy is a stochastic mixed strategy, enabling Nash
equilibria in imperfect-information games unlike a deterministic greedy policy.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyspiel

from lmopt.agents.base import SeatAware
from lmopt.games.policy_cache import DEFAULT_RL_DIR

#: This module's own file: hashed into the RL cache key's ``code_hash``
#: component, the same whole-file-hash reasoning
#: ``policy_cache._CFR_SOLVER_FILE`` uses for CFR; see that constant's
#: docstring. A change here (the XFP training loop, the average-policy
#: extraction) must never let a cached policy trained under the OLD code be
#: assigned to a run using the new code, with the same key.
_RL_TRAIN_FILE = Path(__file__)

_EXPLOITABILITY_BARS = {
    "kuhn_poker": 0.10,
    "leduc_poker": 0.50,
    "turn_based_simultaneous_game": 1.0,  # wrapped Goofspiel
}


class RLSelfPlayAgent(SeatAware):
    """Frozen stochastic average policy from XFP self-play.

    Training (XFP iterations) runs once in __init__. After that the policy is
    immutable. The policy is stochastic, enabling Nash equilibrium representation.
    """

    backend = "cpu"
    requires_model = False

    def __init__(
        self,
        game: pyspiel.Game,
        num_iterations: int = 1_000,
        seed: int = 42,
        check_exploitability: bool = True,
    ) -> None:
        self._game = game
        self._seat: int = 0
        # {(player, info_str): {action: probability}}
        self._policy: dict[tuple[int, str], dict[int, float]] = {}
        self._rng = np.random.default_rng(seed)

        self._train(game, num_iterations)

        if check_exploitability:
            self._log_exploitability(game)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, game: pyspiel.Game, num_iterations: int) -> None:
        from open_spiel.python.algorithms.fictitious_play import XFPSolver

        solver = XFPSolver(game)
        for _ in range(num_iterations):
            solver.iteration()

        # average_policy_tables() returns [{info_str: {action: prob}}, ...]
        # indexed by player; convert to our keyed dict.
        tables = solver.average_policy_tables()
        self._policy = {}
        for player, table in enumerate(tables):
            for info_str, probs in table.items():
                self._policy[(player, info_str)] = {
                    int(a): float(p) for a, p in probs.items()
                }

        # Keep solver reference for the exploitability check (avoids re-building).
        self._solver = solver

    def _log_exploitability(self, game: pyspiel.Game) -> None:
        try:
            from open_spiel.python.algorithms.exploitability import exploitability

            expl = exploitability(game, self._solver.average_policy())
            short = game.get_type().short_name
            bar = _EXPLOITABILITY_BARS.get(short, 1.0)
            status = "OK" if expl <= bar else f"WARNING: exceeds bar ({bar})"
            print(f"  [RL] Exploitability: {expl:.4f} {status}")
        except Exception as e:
            print(f"  [RL] Exploitability check skipped: {e}")

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        self._seat = seat

    def act(self, state: pyspiel.State) -> int:
        player = state.current_player()
        info_str = state.information_state_string(player)
        legal = state.legal_actions()

        probs_dict = self._policy.get((player, info_str))
        if probs_dict is None:
            return legal[0]

        actions = [a for a in probs_dict if a in legal]
        if not actions:
            return legal[0]

        probs = np.array([probs_dict[a] for a in actions], dtype=float)
        total = probs.sum()
        probs = probs / total if total > 0 else np.ones(len(actions)) / len(actions)
        return int(self._rng.choice(actions, p=probs))

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        return None

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._policy, f)

    @classmethod
    def load(cls, game: pyspiel.Game, path: str | Path) -> "RLSelfPlayAgent":
        """Load a previously trained frozen policy without retraining."""
        obj = object.__new__(cls)
        obj._game = game
        obj._seat = 0
        obj._rng = np.random.default_rng(0)
        with open(path, "rb") as f:
            obj._policy = pickle.load(f)
        return obj

    @classmethod
    def load_or_train(
        cls,
        game: pyspiel.Game,
        *,
        num_iterations: int = 1_000,
        seed: int = 42,
        cache_dir: str | Path = DEFAULT_RL_DIR,
        check_exploitability: bool = False,
        require_hit: bool = False,
    ) -> "RLSelfPlayAgent":
        """Load the frozen XFP policy from cache, or train-and-persist on a miss.

        This makes the "RL agent is *frozen* after training" [INVARIANT] literally
        true: training (the XFP self-play in ``__init__``) runs at most once per
        ``(str(game), num_iterations, seed, pyspiel version)`` key, then every
        subsequent run loads the pickled average policy without retraining.

        Keyed like the CFR cache (see :mod:`lmopt.games.policy_cache`) on
        ``(str(game), num_iterations, solver, pyspiel/numpy version, code_hash)``
        but WITHOUT the seed: XFP's average-policy tabulation is seed-independent (verified —
        identical policy across seeds; ``_train`` never touches ``self._rng``, the
        seed only drives ``act()`` sampling). Omitting the seed means any play seed
        reuses the one trained policy instead of forcing a needless ~110-min
        Liar's Dice retrain, and ``precompute_references`` need not share a seed
        with ``run_sweep``. The requested ``seed`` is still applied to the loaded
        agent's play RNG.

        Args:
            game: the OpenSpiel game to train/play on.
            num_iterations: XFP iterations (part of the cache key).
            seed: RNG seed for action sampling (NOT part of the cache key).
            cache_dir: directory for the ``.pkl`` policy + ``.json`` sidecar.
            check_exploitability: only meaningful on a miss (fresh train); loaded
                policies skip the check (the solver is not retained).
            require_hit: when True, raise ``RuntimeError`` on a cache miss
                instead of training. M7 parallel runs set this so a cold cache
                fails loudly up front rather than letting N workers retrain
                concurrently (design.md "no silent fallbacks" [INVARIANT]);
                warm the cache first with ``scripts/precompute_references.py``.
        """
        from lmopt.games.policy_cache import (
            cache_components,
            paths,
            meta_matches,
            _write_meta,
            _file_sha256,
        )

        comps = cache_components(
            game, iters=num_iterations, solver="xfp",
            code_hash=_file_sha256(_RL_TRAIN_FILE),
        )
        blob_path, meta_path = paths(cache_dir, "rl", comps)

        if os.path.exists(blob_path) and meta_matches(meta_path, comps):
            try:
                agent = cls.load(game, blob_path)
                if not isinstance(agent._policy, dict):
                    raise TypeError(
                        f"cached RL policy is {type(agent._policy).__name__}, not dict"
                    )
                agent._rng = np.random.default_rng(seed)
                return agent
            except (pickle.UnpicklingError, EOFError, ValueError, OSError,
                    TypeError, AttributeError) as exc:
                # Corrupt/foreign blob: re-train rather than crash the sweep
                # (mirrors the CFR cache), and say so ("no silent fallbacks").
                print(
                    f"[policy_cache] WARNING: cached RL blob {blob_path} corrupt "
                    f"({exc!r}); re-training.",
                    file=sys.stderr,
                )

        if require_hit:
            raise RuntimeError(
                f"RL cache miss for {comps} with require_hit=True — warm the "
                "cache with scripts/precompute_references.py before a parallel run"
            )

        # --- Miss / corrupt: train fresh (XFP self-play) and persist ---
        agent = cls(
            game,
            num_iterations=num_iterations,
            seed=seed,
            check_exploitability=check_exploitability,
        )
        os.makedirs(cache_dir, exist_ok=True)
        agent.save(blob_path)
        _write_meta(meta_path, comps)
        return agent


# ---------------------------------------------------------------------------
# Thin policy wrapper for exploitability check (used when loading from file)
# ---------------------------------------------------------------------------

class _XFPPolicy:
    """Minimal OpenSpiel Policy interface wrapping the XFP tabular dict."""

    def __init__(
        self,
        game: pyspiel.Game,
        policy: dict[tuple[int, str], dict[int, float]],
    ) -> None:
        self._game = game
        self._policy = policy

    def action_probabilities(
        self, state: pyspiel.State, player_id: Optional[int] = None
    ) -> dict[int, float]:
        """

        Args:
        	state:
        	player_id:

        Returns:

        """
        player = state.current_player() if player_id is None else player_id
        info_str = state.information_state_string(player)
        legal = state.legal_actions()
        probs_dict = self._policy.get((player, info_str))
        if probs_dict is None:
            p = 1.0 / len(legal)
            return {a: p for a in legal}
        return {a: probs_dict.get(a, 0.0) for a in legal}


# todo: this can definitely be better