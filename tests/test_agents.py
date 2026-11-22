"""Unit tests for the non-LM agent implementations."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspiel")

import numpy as np

from lmopt.agents.base import Agent
from lmopt.agents.bayesian import BayesianAgent
from lmopt.agents.cfr_ref import CFRRefAgent
from lmopt.agents.classical import AlwaysBetAgent, ThresholdAgent
from lmopt.agents.random_agent import RandomAgent
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_kuhn


@pytest.fixture(scope="module")
def kuhn_game():
    return load_kuhn()


@pytest.fixture(scope="module")
def cfr_policy(kuhn_game):
    return solve_cfr(kuhn_game, 500)


@pytest.fixture(scope="module")
def all_decision_states(kuhn_game):
    """Return a sample of decision states by playing random rollouts."""
    states = []
    rng = np.random.default_rng(0)
    for _ in range(10):
        s = kuhn_game.new_initial_state()
        while not s.is_terminal():
            if s.is_chance_node():
                actions, probs = zip(*s.chance_outcomes(), strict=False)
                s.apply_action(int(rng.choice(list(actions), p=list(probs))))
            else:
                states.append(s.clone())
                s.apply_action(int(rng.choice(s.legal_actions())))
    return states[:5]  # keep first 5 decision states


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------


class TestRandomAgent:
    def test_returns_legal_action(self, kuhn_game, all_decision_states):
        agent = RandomAgent(seed=0)
        agent.reset(kuhn_game, seat=0, opponent_id="cfr_ref")
        for state in all_decision_states:
            if state.current_player() == 0:
                action = agent.act(state)
                assert action in state.legal_actions()

    def test_belief_is_none(self, kuhn_game, all_decision_states):
        agent = RandomAgent(seed=0)
        agent.reset(kuhn_game, seat=0, opponent_id="cfr_ref")
        for state in all_decision_states:
            assert agent.belief(state) is None

    def test_protocol_conformance(self):
        """
        The agent has every attribute the harness's `Agent` protocol names. This checks presence, not
        signatures; the act/belief tests around it check behaviour.
        """
        assert isinstance(RandomAgent(seed=0), Agent)


# ---------------------------------------------------------------------------
# CFRRefAgent
# ---------------------------------------------------------------------------


class TestCFRRefAgent:
    def test_returns_legal_action(self, kuhn_game, cfr_policy, all_decision_states):
        agent = CFRRefAgent(cfr_policy, seed=0)
        agent.reset(kuhn_game, seat=0, opponent_id="random")
        for state in all_decision_states:
            if state.current_player() == 0:
                action = agent.act(state)
                assert action in state.legal_actions()

    def test_policy_at_sums_to_one(self, kuhn_game, cfr_policy, all_decision_states):
        agent = CFRRefAgent(cfr_policy, seed=0)
        for state in all_decision_states:
            probs, off_support = agent.policy_at(state)
            assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_policy_at_keys_are_legal(self, kuhn_game, cfr_policy, all_decision_states):
        agent = CFRRefAgent(cfr_policy, seed=0)
        for state in all_decision_states:
            probs, off_support = agent.policy_at(state)
            legal = set(state.legal_actions())
            assert set(probs.keys()) == legal

    def test_policy_at_off_support_false_for_full_policy(
        self, kuhn_game, cfr_policy, all_decision_states
    ):
        """A fully-enumerated TabularPolicy (CFR+) never reports off-support."""
        agent = CFRRefAgent(cfr_policy, seed=0)
        for state in all_decision_states:
            _, off_support = agent.policy_at(state)
            assert off_support is False

    def test_belief_is_none(self, kuhn_game, cfr_policy, all_decision_states):
        agent = CFRRefAgent(cfr_policy, seed=0)
        for state in all_decision_states:
            assert agent.belief(state) is None

    def test_act_flags_last_fallback_used_on_partial_reference(
        self, kuhn_game, tmp_path
    ):
        """act() must not silently sample the DepthReference's library-uniform
        default. Reuses test_depth_reference's truncated-table helper so the
        walk necessarily crosses on-support and off-support nodes."""
        import pickle

        from tests.test_depth_reference import _truncated_kuhn_payload
        from lmopt.games.depth_reference import load_depth_reference

        payload = _truncated_kuhn_payload(kuhn_game, keep=2)
        path = tmp_path / "trunc.pkl"
        path.write_bytes(pickle.dumps(payload))
        ref = load_depth_reference(path, kuhn_game, mode="uniform")

        agent = CFRRefAgent(ref, seed=0)
        assert agent.last_fallback_used is False  # set in __init__, before any act()

        st = kuhn_game.new_initial_state()
        while st.is_chance_node():
            st.apply_action(st.legal_actions()[0])

        seen_on = seen_off = False
        while not st.is_terminal():
            _, off_support = ref.lookup(st)
            action = agent.act(st)
            assert agent.last_fallback_used == off_support
            seen_on = seen_on or not off_support
            seen_off = seen_off or off_support
            st.apply_action(action)
        assert seen_on and seen_off, "walk must cross both on- and off-support nodes"

    def test_protocol_conformance(self, cfr_policy):
        """
        The agent has every attribute the harness's `Agent` protocol names. This checks presence, not
        signatures; the act/belief tests around it check behaviour.
        """
        assert isinstance(CFRRefAgent(cfr_policy, seed=0), Agent)


# ---------------------------------------------------------------------------
# AlwaysBetAgent
# ---------------------------------------------------------------------------


class TestAlwaysBetAgent:
    def test_returns_legal_action(self, kuhn_game, all_decision_states):
        agent = AlwaysBetAgent()
        agent.reset(kuhn_game, seat=0, opponent_id="cfr_ref")
        for state in all_decision_states:
            action = agent.act(state)
            assert action in state.legal_actions()

    def test_bets_when_legal(self, kuhn_game):
        # Manually construct a state where betting is legal.
        s = kuhn_game.new_initial_state()
        s.apply_action(0)  # p0 = Jack
        s.apply_action(1)  # p1 = Queen
        # First action for p0: legal = [0 (check), 1 (bet)]
        agent = AlwaysBetAgent()
        assert agent.act(s) == 1

    def test_protocol_conformance(self):
        """
        The agent has every attribute the harness's `Agent` protocol names. This checks presence, not
        signatures; the act/belief tests around it check behaviour.
        """
        assert isinstance(AlwaysBetAgent(), Agent)


# ---------------------------------------------------------------------------
# ThresholdAgent
# ---------------------------------------------------------------------------


class TestThresholdAgent:
    def test_returns_legal_action(self, kuhn_game, all_decision_states):
        agent = ThresholdAgent(threshold=2)
        agent.reset(kuhn_game, seat=0, opponent_id="cfr_ref")
        for state in all_decision_states:
            action = agent.act(state)
            assert action in state.legal_actions()

    def test_bets_only_on_king(self, kuhn_game):
        agent = ThresholdAgent(threshold=2)
        # Jack (card 0) - should pass
        s_jack = kuhn_game.new_initial_state()
        s_jack.apply_action(0)  # p0 = Jack
        s_jack.apply_action(1)  # p1 = Queen
        assert agent.act(s_jack) == 0  # check/pass

        # King (card 2) - should bet
        s_king = kuhn_game.new_initial_state()
        s_king.apply_action(2)  # p0 = King
        s_king.apply_action(0)  # p1 = Jack
        assert agent.act(s_king) == 1  # bet

    def test_protocol_conformance(self):
        """
        The agent has every attribute the harness's `Agent` protocol names. This checks presence, not
        signatures; the act/belief tests around it check behaviour.
        """
        assert isinstance(ThresholdAgent(), Agent)


# ---------------------------------------------------------------------------
# BayesianAgent
# ---------------------------------------------------------------------------


class TestBayesianAgent:
    def test_returns_legal_action(self, kuhn_game, cfr_policy, all_decision_states):
        agent = BayesianAgent(opponent_policy=cfr_policy, seed=0)
        agent.reset(kuhn_game, seat=0, opponent_id="cfr_ref")
        for state in all_decision_states:
            if state.current_player() == 0:
                action = agent.act(state)
                assert action in state.legal_actions()

    def test_belief_shape(self, kuhn_game, cfr_policy, all_decision_states):
        agent = BayesianAgent(opponent_policy=cfr_policy, seed=0)
        agent.reset(kuhn_game, seat=1, opponent_id="cfr_ref")
        for state in all_decision_states:
            if state.current_player() == 1:
                b = agent.belief(state)
                assert b is not None
                assert b.shape == (3,)

    def test_belief_sums_to_one(self, kuhn_game, cfr_policy, all_decision_states):
        agent = BayesianAgent(opponent_policy=cfr_policy, seed=0)
        agent.reset(kuhn_game, seat=1, opponent_id="cfr_ref")
        for state in all_decision_states:
            if state.current_player() == 1:
                b = agent.belief(state)
                assert abs(b.sum() - 1.0) < 1e-6

    def test_own_card_entry_is_zero(self, kuhn_game, cfr_policy):
        agent = BayesianAgent(opponent_policy=cfr_policy, seed=0)
        agent.reset(kuhn_game, seat=0, opponent_id="cfr_ref")
        # p0 = Jack (card 0)
        s = kuhn_game.new_initial_state()
        s.apply_action(0)  # p0 = Jack
        s.apply_action(1)  # p1 = Queen
        b = agent.belief(s)
        assert b[0] == pytest.approx(0.0, abs=1e-9)

    def test_protocol_conformance(self, cfr_policy):
        """
        The agent has every attribute the harness's `Agent` protocol names. This checks presence, not
        signatures; the act/belief tests around it check behaviour.
        """
        assert isinstance(BayesianAgent(opponent_policy=cfr_policy), Agent)
