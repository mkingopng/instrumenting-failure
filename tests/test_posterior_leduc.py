"""Tests for compute_true_posterior with Leduc Poker (multi-round, community card)."""

from __future__ import annotations

import math

import pytest

pyspiel = pytest.importorskip("pyspiel")

from lmopt.elicitation.posterior import compute_true_posterior
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import GAMES


@pytest.fixture(scope="module")
def leduc_setup():
    game = GAMES["leduc"]()
    cfr_pol = solve_cfr(game, 200)
    return game, cfr_pol


def _reach_round1_state(game, p0_card: int, p1_card: int):
    """Deal p0_card/p1_card, P0 checks - P1 to act in round 1."""
    s = game.new_initial_state()
    s.apply_action(p0_card)
    s.apply_action(p1_card)
    s.apply_action(1)  # P0 checks
    return s


def _reach_round2_state(game, p0_card: int, p1_card: int, community: int):
    """Both check round 1, community card dealt - P0 to act in round 2."""
    s = game.new_initial_state()
    s.apply_action(p0_card)
    s.apply_action(p1_card)
    s.apply_action(1)  # P0 checks
    s.apply_action(1)  # P1 checks
    s.apply_action(community)  # community card
    return s


# `leduc_setup` above is module-scoped and shared by
# every test in TestRound1 and TestRound2CommunityCard (9 tests; 20s setup
# cost attributed to whichever runs first). TestKuhnBackwardCompat below
# builds its own game/policy locally and does not request this fixture, so it
# is left unmarked (class-level `pytestmark`, not module-level, so it does not
# leak onto TestKuhnBackwardCompat). Marking only one of the 9 fixture-sharing
# tests would just move the 20s to the next one, so both classes are marked
# in full.
class TestRound1:
    pytestmark = pytest.mark.slow

    def test_posterior_sums_to_one(self, leduc_setup):
        game, cfr = leduc_setup
        state = _reach_round1_state(game, p0_card=0, p1_card=2)
        post = compute_true_posterior(game, state, seat=1, opponent_policy=cfr)
        assert math.isclose(post.sum(), 1.0, abs_tol=1e-9)

    def test_own_card_is_zero(self, leduc_setup):
        game, cfr = leduc_setup
        state = _reach_round1_state(game, p0_card=0, p1_card=2)
        post = compute_true_posterior(game, state, seat=1, opponent_policy=cfr)
        assert post[2] == 0.0  # P1 holds card 2; must be zero

    def test_no_public_cards_yet_all_others_nonzero(self, leduc_setup):
        """Round 1: no community card yet, all cards except own should be in support."""
        game, cfr = leduc_setup
        state = _reach_round1_state(game, p0_card=0, p1_card=2)
        post = compute_true_posterior(game, state, seat=1, opponent_policy=cfr)
        # Cards 0,1,3,4,5 (all except own card 2) should have positive probability.
        for c in [0, 1, 3, 4, 5]:
            assert post[c] > 0.0, f"card {c} should be reachable but got {post[c]}"

    def test_shape_is_num_cards(self, leduc_setup):
        game, cfr = leduc_setup
        state = _reach_round1_state(game, p0_card=0, p1_card=2)
        post = compute_true_posterior(game, state, seat=1, opponent_policy=cfr)
        assert post.shape == (6,)


class TestRound2CommunityCard:
    pytestmark = pytest.mark.slow

    def test_community_card_excluded(self, leduc_setup):
        """After community card is revealed, that card must have probability 0."""
        game, cfr = leduc_setup
        # P0=card0, P1=card2, community=card4
        state = _reach_round2_state(game, p0_card=0, p1_card=2, community=4)
        post = compute_true_posterior(game, state, seat=0, opponent_policy=cfr)
        assert post[4] == 0.0, f"Community card 4 should be excluded but got {post[4]}"

    def test_own_card_is_zero(self, leduc_setup):
        game, cfr = leduc_setup
        state = _reach_round2_state(game, p0_card=0, p1_card=2, community=4)
        post = compute_true_posterior(game, state, seat=0, opponent_policy=cfr)
        assert post[0] == 0.0

    def test_posterior_sums_to_one(self, leduc_setup):
        game, cfr = leduc_setup
        state = _reach_round2_state(game, p0_card=0, p1_card=2, community=4)
        post = compute_true_posterior(game, state, seat=0, opponent_policy=cfr)
        assert math.isclose(post.sum(), 1.0, abs_tol=1e-9)

    def test_support_excludes_own_and_community(self, leduc_setup):
        """Cards 1,2,3,5 should all have positive probability; 0 and 4 should not."""
        game, cfr = leduc_setup
        state = _reach_round2_state(game, p0_card=0, p1_card=2, community=4)
        post = compute_true_posterior(game, state, seat=0, opponent_policy=cfr)
        for c in [1, 2, 3, 5]:
            assert post[c] > 0.0, f"card {c} should be reachable but got {post[c]}"
        for c in [0, 4]:
            assert post[c] == 0.0, f"card {c} should be excluded but got {post[c]}"

    def test_different_community_cards_give_different_posteriors(self, leduc_setup):
        """Community card identity affects the posterior support."""
        game, cfr = leduc_setup
        s_c4 = _reach_round2_state(game, p0_card=0, p1_card=2, community=4)
        s_c5 = _reach_round2_state(game, p0_card=0, p1_card=2, community=5)
        post4 = compute_true_posterior(game, s_c4, seat=0, opponent_policy=cfr)
        post5 = compute_true_posterior(game, s_c5, seat=0, opponent_policy=cfr)
        # Different community cards -> different excluded cards -> different posteriors.
        assert not (post4 == post5).all()
        assert post4[4] == 0.0
        assert post5[5] == 0.0


class TestKuhnBackwardCompat:
    """Ensure the fix doesn't break the original Kuhn path."""

    def test_kuhn_posterior_sums_to_one(self):
        from lmopt.games.loaders import load_kuhn

        game = load_kuhn()
        cfr = solve_cfr(game, 100)
        s = game.new_initial_state()
        s.apply_action(0)  # P0 card 0 (Jack)
        s.apply_action(2)  # P1 card 2 (King)
        # P1 to act (P0 checked)
        s.apply_action(0)  # P0 checks
        post = compute_true_posterior(game, s, seat=1, opponent_policy=cfr)
        assert math.isclose(post.sum(), 1.0, abs_tol=1e-9)
        assert post[2] == 0.0  # own card

    def test_kuhn_posterior_only_two_possible_cards(self):
        from lmopt.games.loaders import load_kuhn

        game = load_kuhn()
        cfr = solve_cfr(game, 100)
        s = game.new_initial_state()
        s.apply_action(0)  # P0 = Jack
        s.apply_action(1)  # P1 = Queen
        s.apply_action(0)  # P0 checks
        post = compute_true_posterior(game, s, seat=1, opponent_policy=cfr)
        # P1 holds card 1 (Queen). Possible: 0 (Jack) and 2 (King) only.
        assert post[1] == 0.0
        assert post[0] > 0.0
        assert post[2] > 0.0
        assert math.isclose(post.sum(), 1.0, abs_tol=1e-9)
