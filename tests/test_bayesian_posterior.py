"""Verify BayesianAgent posterior values against hand-computed CFR-based expectations."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspiel")

from lmopt.agents.bayesian import BayesianAgent
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_goofspiel, load_kuhn

TOL = 0.01


@pytest.fixture(scope="module")
def game():
    return load_kuhn()


@pytest.fixture(scope="module")
def cfr_policy(game):
    return solve_cfr(game, 1000)


class TestBayesianPosterior:
    def test_queen_facing_bet(self, game, cfr_policy):
        """P1=Queen facing P0's bet: Jack ≈ 0.249, King ≈ 0.751."""
        agent = BayesianAgent(opponent_policy=cfr_policy)
        agent.reset(game, seat=1, opponent_id="cfr_ref")

        s = game.new_initial_state()
        s.apply_action(0)  # p0 = Jack
        s.apply_action(1)  # p1 = Queen
        s.apply_action(1)  # p0 bets  (info_set for p1: '1b')

        posterior = agent._compute_posterior(s)

        assert posterior[0] == pytest.approx(
            0.249, abs=TOL
        ), f"Jack posterior: {posterior[0]}"
        assert posterior[1] == pytest.approx(
            0.0, abs=1e-9
        ), "Own card (Queen) must be 0"
        assert posterior[2] == pytest.approx(
            0.751, abs=TOL
        ), f"King posterior: {posterior[2]}"

    def test_jack_after_check_bet(self, game, cfr_policy):
        """P0=Jack after P0-check P1-bet: King ≈ 1.0 (Queen virtually impossible)."""
        agent = BayesianAgent(opponent_policy=cfr_policy)
        agent.reset(game, seat=0, opponent_id="cfr_ref")

        s = game.new_initial_state()
        s.apply_action(0)  # p0 = Jack
        s.apply_action(2)  # p1 = King
        s.apply_action(0)  # p0 checks
        s.apply_action(1)  # p1 bets  (info_set for p0: '0pb')

        posterior = agent._compute_posterior(s)

        assert posterior[0] == pytest.approx(0.0, abs=1e-9), "Own card (Jack) must be 0"
        assert posterior[2] == pytest.approx(
            1.0, abs=TOL
        ), f"King posterior: {posterior[2]}"

    def test_first_node_uniform_prior(self, game, cfr_policy):
        """At the first decision node (no actions observed), posterior is uniform over 2 remaining cards."""
        agent = BayesianAgent(opponent_policy=cfr_policy)
        agent.reset(game, seat=0, opponent_id="cfr_ref")

        s = game.new_initial_state()
        s.apply_action(1)  # p0 = Queen
        s.apply_action(0)  # p1 = Jack  (but p0 doesn't know this)

        posterior = agent._compute_posterior(s)

        # p0 holds Queen (card 1); remaining cards are Jack (0) and King (2).
        assert posterior[1] == pytest.approx(0.0, abs=1e-9), "Own card must be 0"
        assert posterior[0] == pytest.approx(
            0.5, abs=TOL
        ), f"Jack posterior: {posterior[0]}"
        assert posterior[2] == pytest.approx(
            0.5, abs=TOL
        ), f"King posterior: {posterior[2]}"


class TestBayesianLikelihoodIsCFRPlus:
    """The Bayesian likelihood model is the CFR+ reference.

    The agent's posterior is built from the opponent policy passed at
    construction. The working reference is CFR+ (solve_cfr defaults to
    plus=True). These tests make that coupling explicit so it cannot silently
    regress to vanilla CFR.
    """

    def _queen_facing_bet_posterior(self, game, policy):
        agent = BayesianAgent(opponent_policy=policy)
        agent.reset(game, seat=1, opponent_id="cfr_ref")
        s = game.new_initial_state()
        s.apply_action(0)  # p0 = Jack
        s.apply_action(1)  # p1 = Queen
        s.apply_action(1)  # p0 bets -> p1 info_set '1b'
        return agent._compute_posterior(s)

    def test_solve_cfr_default_kwarg_is_plus(self):
        """The coupling is pinned at the source: solve_cfr's `plus` default is
        True. On-path CFR and CFR+ agree, so a posterior comparison alone could
        not catch a regression to plus=False; this asserts the config directly."""
        import inspect

        from lmopt.games.cfr import solve_cfr as _solve_cfr

        assert inspect.signature(_solve_cfr).parameters["plus"].default is True

    def test_solve_cfr_default_is_plus(self, game):
        """solve_cfr() with no `plus` arg yields the CFR+ policy (same posterior
        as the explicit plus=True call).

        Smoke test through the default call path. On-path CFR and CFR+ agree, so
        a regression of the default to plus=False leaves this green; that
        mutation is killed by the signature test above, not here."""
        default_pol = solve_cfr(game, 1000)
        plus_pol = solve_cfr(game, 1000, plus=True)
        p_default = self._queen_facing_bet_posterior(game, default_pol)
        p_plus = self._queen_facing_bet_posterior(game, plus_pol)
        assert p_default == pytest.approx(p_plus, abs=1e-9)

    def test_likelihood_matches_cfr_plus_value(self, game):
        """Posterior under the CFR+ reference matches the hand-computed CFR+
        value at the Queen-facing-bet info-set (Jack≈0.249, King≈0.751)."""
        plus_pol = solve_cfr(game, 1000, plus=True)
        post = self._queen_facing_bet_posterior(game, plus_pol)
        assert post[0] == pytest.approx(0.249, abs=TOL)
        assert post[1] == pytest.approx(0.0, abs=1e-9)
        assert post[2] == pytest.approx(0.751, abs=TOL)


class TestGoofspielFallbackFlagged:
    """The Bayesian's Goofspiel CFR-argmax fallback is not silent.

    Goofspiel has no fixed private-card deal, so the agent cannot compute a
    posterior and falls back to CFR argmax. That degradation must be logged
    (no silent fallbacks) via last_fallback_used, and belief() must
    return None rather than a fabricated posterior.
    """

    def test_goofspiel_fallback_is_flagged(self):
        game = load_goofspiel(num_cards=4)
        policy = solve_cfr(game, 300)
        agent = BayesianAgent(opponent_policy=policy)
        agent.reset(game, seat=1, opponent_id="cfr_ref")

        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.legal_actions()[0])

        action = agent.act(state)
        assert action in state.legal_actions()
        assert agent.last_fallback_used is True
        assert agent.belief(state) is None


class _NeverBets:
    """An opponent policy that gives a bet probability zero at every information state."""

    def action_probabilities(self, state, player):
        return {0: 1.0, 1: 0.0}


class TestZeroLikelihoodSubstitutionIsFlagged:
    """
    When every hypothesis gives the observed actions zero probability the posterior is
    replaced by a uniform one, and the row must say so. The uniform prior at the first node
    is the ordinary path and must not say so.
    """

    def test_an_observed_action_at_probability_zero_sets_the_flag(self, game):
        """
        Seat 1 holds the Queen and sees a bet from an opponent policy that never bets: both
        hypotheses have likelihood 0, the posterior is uniform over them, and the flag is set.
        """
        agent = BayesianAgent(opponent_policy=_NeverBets())
        agent.reset(game, seat=1, opponent_id="never_bets")
        s = game.new_initial_state()
        for a in (0, 1, 1):  # p0 Jack, p1 Queen, p0 bets
            s.apply_action(a)

        action = agent.act(s)

        assert action in s.legal_actions()
        assert agent.last_fallback_used is True
        posterior = agent._compute_posterior(s)
        assert posterior[0] == pytest.approx(0.5) and posterior[2] == pytest.approx(0.5)
        assert posterior[1] == 0.0

    def test_the_first_node_does_not_set_the_flag(self, game):
        """
        With no opponent action observed every likelihood is 1.0, the posterior is the
        uniform prior, and nothing was substituted.
        """
        agent = BayesianAgent(opponent_policy=_NeverBets())
        agent.reset(game, seat=0, opponent_id="never_bets")
        s = game.new_initial_state()
        s.apply_action(1)
        s.apply_action(0)

        agent.act(s)

        assert agent.last_fallback_used is False

    def test_the_next_ordinary_node_clears_the_flag(self, game):
        """
        The flag is per node: a node after a flagged one reads False again.
        """
        agent = BayesianAgent(opponent_policy=_NeverBets())
        agent.reset(game, seat=1, opponent_id="never_bets")
        flagged = game.new_initial_state()
        for a in (0, 1, 1):
            flagged.apply_action(a)
        agent.act(flagged)
        assert agent.last_fallback_used is True

        ordinary = game.new_initial_state()
        for a in (0, 1, 0):  # p0 Jack, p1 Queen, p0 checks, which the policy allows
            ordinary.apply_action(a)
        agent.act(ordinary)

        assert agent.last_fallback_used is False

    def test_the_logged_rows_carry_the_flag(self, game, tmp_path):
        """
        Through the production entrypoint: a Bayesian agent whose opponent model never bets,
        playing an opponent that always bets, logs `fallback_used` True on the nodes that
        follow a bet and False on its first-move nodes.
        """
        from lmopt.agents.classical import AlwaysBetAgent
        from lmopt.agents.cfr_ref import CFRRefAgent
        from lmopt.harness.match_runner import run_match
        from lmopt.logging.writer import MatchWriter

        reference = CFRRefAgent(solve_cfr(game, 50), seed=0)
        bayesian = BayesianAgent(opponent_policy=_NeverBets(), seed=2)
        writer = MatchWriter(str(tmp_path / "match.parquet"))
        records = run_match(
            game,
            bayesian,
            AlwaysBetAgent(),
            reference,
            n_hands=40,
            seed=7,
            writer=writer,
        )

        mine = [r for r in records if r.player_to_act == 0]
        after_a_bet = [r for r in mine if r.info_set.endswith("pb")]
        first_move = [r for r in mine if len(r.info_set) == 1]
        assert after_a_bet and first_move
        assert all(r.fallback_used for r in after_a_bet)
        assert not any(r.fallback_used for r in first_move)
