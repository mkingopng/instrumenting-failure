"""Tests for compute_true_posterior, including agreement with BayesianAgent."""

from __future__ import annotations

import numpy as np
import pytest

from lmopt.agents.bayesian import BayesianAgent
from lmopt.elicitation.posterior import compute_true_posterior
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_kuhn


@pytest.fixture(scope="module")
def game():
    return load_kuhn()


@pytest.fixture(scope="module")
def cfr_policy(game):
    return solve_cfr(game, 1000)


def _state(game, *actions):
    s = game.new_initial_state()
    for a in actions:
        s.apply_action(a)
    return s


class TestComputeTruePosteriorShape:
    def test_shape_is_num_cards(self, game, cfr_policy):
        s = _state(game, 0, 1)  # Jack vs Queen, no actions yet
        p = compute_true_posterior(game, s, seat=0, opponent_policy=cfr_policy)
        assert p.shape == (3,)

    def test_sums_to_one(self, game, cfr_policy):
        s = _state(game, 0, 1)
        p = compute_true_posterior(game, s, seat=0, opponent_policy=cfr_policy)
        assert abs(p.sum() - 1.0) < 1e-6

    def test_own_card_zeroed_seat0(self, game, cfr_policy):
        s = _state(game, 0, 1)  # P0 has Jack (0)
        p = compute_true_posterior(game, s, seat=0, opponent_policy=cfr_policy)
        assert p[0] == pytest.approx(0.0)

    def test_own_card_zeroed_seat1(self, game, cfr_policy):
        s = _state(game, 0, 1)  # P1 has Queen (1)
        p = compute_true_posterior(game, s, seat=1, opponent_policy=cfr_policy)
        assert p[1] == pytest.approx(0.0)


class TestComputeTruePosteriorValues:
    def test_first_node_approximately_uniform(self, game, cfr_policy):
        """With no action history, prior is uniform over the two unseen cards."""
        s = _state(game, 0, 1)  # P0=Jack, no actions
        p = compute_true_posterior(game, s, seat=0, opponent_policy=cfr_policy)
        # P1 could be Queen(1) or King(2) - approximately equal likelihood
        assert p[0] == pytest.approx(0.0)
        assert abs(p[1] - p[2]) < 0.05

    def test_queen_facing_bet(self, game, cfr_policy):
        """P0=Jack bets, P1=Queen faces bet. Posterior over Jack vs King for P1."""
        # P0 has Jack(0), P1 has Queen(1), P0 bets (action 1)
        s = _state(game, 0, 1, 1)
        p = compute_true_posterior(game, s, seat=1, opponent_policy=cfr_policy)
        assert p[1] == pytest.approx(0.0)
        # From plan: P(Jack | bet) ≈ 0.249, P(King | bet) ≈ 0.751
        assert abs(p[0] - 0.249) < 0.02
        assert abs(p[2] - 0.751) < 0.02

    def test_jack_after_check_bet(self, game, cfr_policy):
        """P0=Jack checks, P1 bets - P0 faces bet. Posterior should be near King-certain."""
        # P0=Jack(0), P1=King(2), P0 checks(0), P1 bets(1)
        s = _state(game, 0, 2, 0, 1)
        p = compute_true_posterior(game, s, seat=0, opponent_policy=cfr_policy)
        assert p[0] == pytest.approx(0.0)
        # From plan: posterior[2] (King) ≈ 1.0 after check-bet
        assert p[2] > 0.90


class TestComputeTruePosteriorMatchesBayesian:
    """compute_true_posterior must exactly agree with BayesianAgent._compute_posterior."""

    def test_first_node(self, game, cfr_policy):
        s = _state(game, 0, 1)
        tp = compute_true_posterior(game, s, seat=0, opponent_policy=cfr_policy)
        ba = BayesianAgent(opponent_policy=cfr_policy, seed=0)
        ba.reset(game, seat=0, opponent_id="cfr")
        bp = ba._compute_posterior(s)
        np.testing.assert_allclose(tp, bp, atol=1e-8)

    def test_facing_bet(self, game, cfr_policy):
        s = _state(game, 0, 1, 1)
        tp = compute_true_posterior(game, s, seat=1, opponent_policy=cfr_policy)
        ba = BayesianAgent(opponent_policy=cfr_policy, seed=0)
        ba.reset(game, seat=1, opponent_id="cfr")
        bp = ba._compute_posterior(s)
        np.testing.assert_allclose(tp, bp, atol=1e-8)

    def test_check_bet(self, game, cfr_policy):
        s = _state(game, 0, 2, 0, 1)
        tp = compute_true_posterior(game, s, seat=0, opponent_policy=cfr_policy)
        ba = BayesianAgent(opponent_policy=cfr_policy, seed=0)
        ba.reset(game, seat=0, opponent_id="cfr")
        bp = ba._compute_posterior(s)
        np.testing.assert_allclose(tp, bp, atol=1e-8)
