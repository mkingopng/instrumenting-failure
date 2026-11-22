"""Correct Liar's Dice Bayesian posterior (independent dice).

Dice are independent - the opponent can roll the same face as you - so the
posterior support must include the agent's own die value. Excluding the own
face (`c != my_card`) would silently assign the opponent's true die
probability 0 whenever it matched. These tests pin the behaviour for both the
standalone `compute_true_posterior` and the `BayesianAgent` itself.
"""

import pytest

pytest.importorskip("pyspiel")

from lmopt.agents.bayesian import BayesianAgent
from lmopt.elicitation.posterior import compute_true_posterior
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_liarsdice

# `game`/`policy` below are module-scoped and shared by all 3 tests in this
# file. `solve_cfr(game, 200)` on Liar's Dice is the ~268s setup cost pytest attributes to whichever test runs first. Marking only that
# test `slow` would just move the 268s to the next test that requests `policy`
# -- so the whole module is marked, not one test.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def game():
    return load_liarsdice()


@pytest.fixture(scope="module")
def policy(game):
    return solve_cfr(game, 200)


def _state_both_dice_zero(game):
    """A P1 decision state where both players rolled die face 0 (own == opp)."""
    s = game.new_initial_state()
    s.apply_action(0)  # P0 die face 0
    s.apply_action(0)  # P1 die face 0
    while not s.is_terminal() and s.current_player() != 1:
        if s.is_chance_node():
            s.apply_action(s.legal_actions()[0])
            continue
        s.apply_action(s.legal_actions()[0])
    return s


def test_opponent_die_can_equal_own_true_posterior(game, policy):
    """compute_true_posterior assigns the opponent's matching die NON-zero mass."""
    s = _state_both_dice_zero(game)
    post = compute_true_posterior(game, s, seat=1, opponent_policy=policy)
    assert len(post) == 6
    assert post[0] > 0.0, "P(opp die == own die 0) must be > 0 for independent dice"
    assert abs(sum(post) - 1.0) < 1e-6
    assert all(p >= -1e-12 for p in post)


def test_bayesian_agent_belief_includes_own_face(game, policy):
    """The BayesianAgent's own belief() also keeps the own face in support."""
    agent = BayesianAgent(opponent_policy=policy)
    agent.reset(game, seat=1, opponent_id="cfr")
    assert agent._independent_deal is True

    s = _state_both_dice_zero(game)
    belief = agent.belief(s)
    assert belief is not None
    assert belief[0] > 0.0
    assert abs(sum(belief) - 1.0) < 1e-6


def test_bayesian_act_runs_and_no_fallback(game, policy):
    """act() best-responds on the real posterior - not a flagged fallback."""
    agent = BayesianAgent(opponent_policy=policy)
    agent.reset(game, seat=1, opponent_id="cfr")
    s = _state_both_dice_zero(game)
    action = agent.act(s)
    assert action in s.legal_actions()
    assert agent.last_fallback_used is False
