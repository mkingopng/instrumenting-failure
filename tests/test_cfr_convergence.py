"""CFR converges to the equilibrium reference for all three games.

Asserts exploitability -> 0 within a per-game iteration budget, and that the
Kuhn average-policy game value matches the known analytic equilibrium (-1/18).
This is the scientific validity gate: the optimum reference must be provably
correct before any agent or LM is scored against it.
"""

import pytest

pytest.importorskip("pyspiel")

from open_spiel.python.algorithms import expected_game_score

from lmopt.games import load_goofspiel, load_kuhn, load_leduc
from lmopt.games.cfr import (
    exploitability_value,
    solve_cfr,
    solve_cfr_tremble,
)

KUHN_GAME_VALUE = -1.0 / 18.0

# (key, loader, iterations, exploitability tolerance). Budgets kept modest so the
# default test run stays fast; CFR+ converges well within them.
# Leduc's exploitability_value call takes about 29s (a full
# best-response walk over Leduc's tree at 300 CFR+ iterations); kuhn and
# goofspiel are both well under 2s. Only leduc is marked slow, via
# pytest.param -- the other two parametrizations are cheap and stay in the
# inner loop.
CONVERGENCE_SPECS = [
    ("kuhn", load_kuhn, 200, 1e-2),
    pytest.param("leduc", load_leduc, 300, 1e-1, marks=pytest.mark.slow),
    ("goofspiel", load_goofspiel, 300, 1e-1),
]


@pytest.mark.parametrize("name,loader,iters,tol", CONVERGENCE_SPECS)
def test_cfr_exploitability_converges(name, loader, iters, tol):
    game = loader()
    avg_policy = solve_cfr(game, iters, plus=True)
    expl = exploitability_value(game, avg_policy)
    assert expl >= 0.0
    assert expl < tol, f"{name}: exploitability {expl:.6f} not below tol {tol}"


def test_cfr_tremble_positive_reach():
    """Every info-set action must have probability >= tremble_eps / n_actions."""
    game = load_kuhn()
    tremble_eps = 1e-3
    policy = solve_cfr_tremble(game, iterations=200, tremble_eps=tremble_eps)
    state = game.new_initial_state()

    def _visit(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for action, _ in s.chance_outcomes():
                _visit(s.child(action))
            return
        probs = policy.action_probabilities(s)
        n = len(probs)
        floor = tremble_eps / n
        for a, p in probs.items():
            assert p >= floor - 1e-10, f"action {a}: prob {p} below floor {floor}"
        assert abs(sum(probs.values()) - 1.0) < 1e-9, "probs do not sum to 1"
        for action in s.legal_actions():
            _visit(s.child(action))

    _visit(state)


def test_cfr_tremble_exploitability_bounded():
    """Trembled policy remains reasonably close to equilibrium."""
    game = load_kuhn()
    policy = solve_cfr_tremble(game, iterations=200, tremble_eps=1e-4)
    expl = exploitability_value(game, policy)
    # With tremble_eps=1e-4, exploitability rises slightly above pure CFR+ but stays small.
    assert expl < 0.05


def test_cfr_tremble_deterministic_reproducible():
    """Two solves with identical args give identical policies at every info-set.

    This is the property P6 (subgame perfection) relies on: a single, reproducible
    off-path reference. CFR+ is deterministic and the tremble blend is a pure
    function of the average policy, so the trembled reference must be unique. If
    it were not, "distance from CFR" off-path would partly measure solver run, not
    a fact about the equilibrium.
    """
    game = load_kuhn()
    p1 = solve_cfr_tremble(game, iterations=200, tremble_eps=1e-4)
    p2 = solve_cfr_tremble(game, iterations=200, tremble_eps=1e-4)

    def _visit(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for action, _ in s.chance_outcomes():
                _visit(s.child(action))
            return
        d1 = p1.action_probabilities(s)
        d2 = p2.action_probabilities(s)
        assert d1.keys() == d2.keys()
        for a in d1:
            assert d1[a] == pytest.approx(d2[a], abs=1e-12), (
                f"info-set {s.information_state_string()} action {a}: "
                f"{d1[a]} != {d2[a]}"
            )
        for action in s.legal_actions():
            _visit(s.child(action))

    _visit(game.new_initial_state())


def test_kuhn_game_value_matches_analytic_equilibrium():
    """Average policy is in the one-parameter equilibrium family, so we validate
    the family's invariant - the game value to player 0 - rather than the
    (alpha-dependent) per-action probabilities."""
    game = load_kuhn()
    avg_policy = solve_cfr(game, 500, plus=True)
    value_p0 = expected_game_score.policy_value(
        game.new_initial_state(), [avg_policy, avg_policy]
    )[0]
    assert value_p0 == pytest.approx(KUHN_GAME_VALUE, abs=1e-3)
