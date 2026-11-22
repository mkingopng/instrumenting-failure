"""Thin wrappers over OpenSpiel's CFR solver and exploitability metrics.

design.md [INVARIANT]: the CFR/CFR+ policy is *imported and validated*, never
reimplemented. This module only orchestrates OpenSpiel's own solver and exposes
its convergence metrics, so the equilibrium reference and the scoring yardstick
come from the same validated source.
"""

from __future__ import annotations

import pyspiel
from open_spiel.python.algorithms import cfr, exploitability
from open_spiel.python.policy import Policy


def solve_cfr(game: pyspiel.Game, iterations: int, plus: bool = True) -> Policy:
    """Run CFR(+) to convergence and return the average policy.

    Args:
        game: a sequential (turn-based) OpenSpiel game. Simultaneous-move games
            must be wrapped via ``turn_based_simultaneous_game`` first (see
            :func:`lmopt.games.loaders.load_goofspiel`).
        iterations: number of CFR iterations to run.
        plus: use ``CFRPlusSolver`` (faster convergence) when True, else the
            vanilla ``CFRSolver``.

    Returns:
        The average policy, which converges to a Nash equilibrium for two-player
        zero-sum games.
    """
    solver_cls = cfr.CFRPlusSolver if plus else cfr.CFRSolver
    solver = solver_cls(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    return solver.average_policy()


def nash_conv(game: pyspiel.Game, policy: Policy) -> float:
    """Total gain available to players by best-responding (0 at equilibrium).

    Preferred convergence metric (defined for n-player games). See OpenSpiel
    ``exploitability.nash_conv``.
    """
    return exploitability.nash_conv(game, policy)


#: The production tremble mass. Small on purpose: the reference must be the
#: equilibrium wherever the equilibrium is defined, and only differ from it by
#: the minimum that makes every info-set reachable. A test that wants the blend
#: VISIBLE passes a large eps explicitly (M9 Task 10, controller addendum §3:
#: at this default the per-info-set distance from plain CFR is at most
#: eps*(1-1/n), so a pin written at it would be resolving sub-1e-4 differences).
DEFAULT_TREMBLE_EPS = 1e-4


def tremble_policy(
    game: pyspiel.Game,
    base: Policy,
    tremble_eps: float = DEFAULT_TREMBLE_EPS,
) -> Policy:
    """Blend an ALREADY-SOLVED policy with the uniform policy.

    Each info-set action probability becomes ``(1 - tremble_eps) * base_prob +
    tremble_eps / n``, so every legal action keeps probability >=
    ``tremble_eps / n`` and every node has strictly positive reach.

    Separated from the solve (M9 Task 10) so a caller can pay for CFR ONCE:
    the base is a ``TabularPolicy`` and goes through ``policy_cache``'s
    persistent cache, while the object returned here does not (see below), and
    the blend itself is cheap arithmetic on the base's output. The
    never-reimplement-CFR [INVARIANT] binds the solve, which is still
    OpenSpiel's; this function does no solving.

    **This object is deliberately not cacheable and not picklable.** It is an
    instance of a class defined inside this function, closing over ``base``, so
    it cannot cross a process boundary (the A3 Task 17 fork-deadlock work found
    spawn blocked by exactly this shape) and it is not a ``TabularPolicy``, so
    ``policy_cache``'s array-persistence mechanism does not apply to it. Both
    are fine because its only consumer, P6 scoring, is offline and
    single-process [INVARIANT scoring is offline]; anything that needs to hand
    a reference to a worker must hand over the BASE and blend on the far side.
    """
    player_ids = list(range(game.num_players()))

    class _TremblePolicy(Policy):
        def __init__(self):
            super().__init__(game, player_ids)

        def action_probabilities(self, state, player_id=None):  # type: ignore[override]
            base_probs = base.action_probabilities(state, player_id)
            n = len(base_probs)
            if n == 0:
                return {}
            u = tremble_eps / n
            return {a: (1.0 - tremble_eps) * p + u for a, p in base_probs.items()}

    return _TremblePolicy()


def solve_cfr_tremble(
    game: pyspiel.Game,
    iterations: int,
    tremble_eps: float = DEFAULT_TREMBLE_EPS,
    plus: bool = True,
) -> Policy:
    """CFR(+) with a uniform-tremble perturbation, giving a unique, reproducible
    off-path reference for P6.

    A thin wrapper: solve, then :func:`tremble_policy`. The two exist
    separately so that a caller with a cached base policy can blend without
    re-solving (M9 Task 10); this signature is unchanged, so every existing
    caller and pin is untouched.

    Required for P6 (subgame perfection) off-path scoring.  Without positive
    reach, the equilibrium strategy is not uniquely determined at zero-reach
    nodes, making "distance from CFR" partially a function of solver run.  CFR+
    is deterministic and this blend is a pure function of its average policy, so
    the trembled policy is a single, reproducible reference at every info-set
    (see ``test_cfr_tremble_deterministic_unique``).  Note: this selects a
    *well-defined fixed reference for scoring*; it is not a claim to have computed
    the theoretical sequential equilibrium of the perturbed game.

    Args:
        game: a sequential (turn-based) OpenSpiel game.
        iterations: CFR(+) iterations to run before blending.
        tremble_eps: uniform mass added across all legal actions (default 1e-4).
        plus: use CFRPlusSolver when True.

    Returns:
        A Policy where every legal action has probability >= tremble_eps / n_legal.
    """
    return tremble_policy(
        game, solve_cfr(game, iterations, plus=plus), tremble_eps=tremble_eps
    )


def exploitability_value(game: pyspiel.Game, policy: Policy) -> float:
    """Per-player exploitability = nash_conv / num_players.

    Defined for two-player constant-sum games; equals nash_conv/2 there.
    Falls back to nash_conv / num_players for general-sum games (e.g. Goofspiel
    with returns_type=total_points) where OpenSpiel's exploitability() rejects the game.
    """
    try:
        return exploitability.exploitability(game, policy)
    except ValueError:
        return exploitability.nash_conv(game, policy) / game.num_players()
