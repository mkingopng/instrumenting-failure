"""
The coverage check's own machinery: the planted truth is right, independent of the rows, and the
verdict can fail. Fast; the full coverage runs are in test_coverage_kuhn.py.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

pytest.importorskip("pyspiel")

import pyspiel

from lmopt.elicitation.posterior import compute_true_posterior
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_kuhn

from tests.coverage_check import coverage, truth
from tests.coverage_check.coverage import Coverage, summarise, wilson_lower
from tests.coverage_check.truth import enumerate_nodes, planted_p5_margin, planted_p7a


@pytest.fixture(scope="module")
def kuhn():
    game = load_kuhn()
    return game, solve_cfr(game, 300)


@pytest.fixture(scope="module")
def nodes(kuhn):
    return enumerate_nodes(*kuhn)


def _replay(game, history) -> pyspiel.State:
    state = game.new_initial_state()
    for action in history:
        state.apply_action(action)
    return state


class TestThePlantedTruthIsExact:
    def test_the_tree_posterior_equals_the_production_posterior_on_every_information_state(
        self, kuhn, nodes
    ):
        """
        The posterior the truth uses is derived from joint reach over the tree, and the one the
        runner logs is derived by replaying hypotheses. They must agree on all six information
        states, to rounding.
        """
        game, policy = kuhn
        assert len(nodes) == 6
        for node in nodes:
            production = compute_true_posterior(
                game, _replay(game, node.history), 0, policy
            )
            assert np.allclose(node.posterior, production, atol=1e-12), node.info_state

    def test_the_visit_weights_are_the_hand_computed_probabilities(self, kuhn, nodes):
        """
        Each hand reaches the observer's first node once. It reaches its second node when it
        passes (one half, it plays uniformly) and the opponent then bets, which the reference
        policy decides per card; the expected total is summed here straight from the policy over
        the six deals, with no tree walk.
        """
        game, policy = kuhn
        first = [n for n in nodes if len(n.history) == 2]
        second = [n for n in nodes if len(n.history) == 4]
        assert sum(n.weight for n in first) == pytest.approx(1.0)
        expected = 0.0
        for observer_card in range(3):
            for opponent_card in range(3):
                if observer_card == opponent_card:
                    continue
                after_pass = _replay(game, (observer_card, opponent_card, 0))
                bets = policy.action_probabilities(after_pass)[1]
                expected += (1 / 6) * 0.5 * bets
        assert sum(n.weight for n in second) == pytest.approx(expected)

    def test_a_truthful_reporter_has_zero_error_and_a_negative_margin(self, nodes):
        """
        With lam = 0 the stated belief is the true posterior: no total variation, and the margin
        is minus the uninformed stater's error.
        """
        assert planted_p7a(nodes, 0.0) == pytest.approx(0.0, abs=1e-15)
        assert planted_p5_margin(nodes, 0.0) < 0.0

    def test_the_error_grows_with_the_departure(self, nodes):
        values = [planted_p7a(nodes, lam) for lam in (0.0, 0.25, 0.5, 1.0)]
        assert values == sorted(values) and len(set(values)) == 4

    def test_the_truth_functions_take_no_sampled_data(self):
        """
        Independence from the rows is the whole point of the design: the truth is a function of the
        tree-derived nodes and the planted parameter, and of nothing else. No frame, no shard, no
        row can be passed in.
        """
        for fn in (planted_p7a, planted_p5_margin):
            assert list(inspect.signature(fn).parameters) == ["nodes", "lam"]
        assert "polars" not in inspect.getsource(truth)
        assert "parquet" not in inspect.getsource(truth).lower()


class TestTheVerdictCanFail:
    """The verdict is a function of intervals and a truth, so it can be driven with made-up ones."""

    def _arms(self, truth_value, n=400, half=0.5, shift=0.0, seed=1):
        """
        arm points with a fixed true spread (0.5 / 2.262, so a 0.5 half-width is the right size),
        and intervals of the half-width asked for
        """
        rng = np.random.default_rng(seed)
        points = rng.normal(truth_value + shift, 0.5 / 2.262, n)
        return (
            list(points),
            list(points - half),
            list(points + half),
        )

    def test_well_sized_unbiased_intervals_are_covered(self):
        points, los, his = self._arms(0.0)
        result = summarise(0.0, points, los, his, 2.262)
        assert result.verdict() == []

    def test_intervals_half_the_right_size_fail_on_coverage_and_width(self):
        points, los, his = self._arms(0.0, half=0.2)
        result = summarise(0.0, points, los, his, 2.262)
        assert any("coverage" in f for f in result.verdict())
        assert any("width" in f for f in result.verdict())
        assert result.width_ratio < 0.5

    def test_a_biased_point_fails_on_bias_and_coverage(self):
        points, los, his = self._arms(0.0, shift=0.5)
        result = summarise(0.0, points, los, his, 2.262)
        assert any("bias" in f for f in result.verdict())

    def test_intervals_that_always_contain_the_truth_are_not_informative(self):
        """
        Coverage above the ceiling means the interval says nothing: here every interval is far
        wider than the spread of the points it is built around.
        """
        rng = np.random.default_rng(3)
        points = list(rng.normal(0.0, 0.01, 400))
        los = [p - 50.0 for p in points]
        his = [p + 50.0 for p in points]
        result = summarise(0.0, points, los, his, 2.262)
        assert result.rate == 1.0
        assert any("coverage" in f for f in result.verdict())

    def test_too_few_arms_cannot_clear_the_floor(self):
        """
        Nineteen of twenty arms covering is a rate of 0.95, but twenty arms cannot show 0.90
        coverage: the verdict fails on the Wilson lower bound rather than passing on a sample too
        small to tell.
        """
        rng = np.random.default_rng(5)
        points = list(rng.normal(0.0, 0.5 / 2.262, 20))
        los = [-0.5] * 19 + [0.4]
        his = [0.5] * 19 + [1.4]
        result = summarise(0.0, points, los, his, 2.262)
        assert result.rate == 0.95 and result.wilson_lower < 0.90
        assert any("coverage" in f for f in result.verdict())

    def test_wilson_lower_matches_a_known_value(self):
        assert wilson_lower(95, 100) == pytest.approx(0.8882, abs=1e-3)
        assert wilson_lower(100, 100) == pytest.approx(0.9630, abs=1e-3)

    def test_the_constants_are_the_agreed_thresholds(self):
        """
        The thresholds are a Wilson lower bound of 0.90, a ceiling of 0.995 and a width factor of 2.
        """
        assert (
            coverage.COVERAGE_FLOOR,
            coverage.COVERAGE_CEILING,
            coverage.WIDTH_FACTOR,
        ) == (
            0.90,
            0.995,
            2.0,
        )


def test_a_smoke_run_through_the_production_scorers_returns_finite_summaries(
    kuhn, nodes
):
    """
    Through the real entrypoint: arms written by the production runner and scored by the
    production cell function come back as finite summaries. No verdict is asserted: six arms say
    nothing about coverage.
    """
    game, policy = kuhn
    truths = {
        "P7a_belief_error": planted_p7a(nodes, 0.3),
        "P5_calibration_per_pair_margin": planted_p5_margin(nodes, 0.3),
    }
    result = coverage.run_property_coverage(
        game,
        "kuhn",
        policy,
        0.3,
        n_arms=6,
        n_seeds=5,
        n_hands=60,
        base_seed=500,
        truths=truths,
    )
    assert set(result) == set(truths)
    for summary in result.values():
        assert isinstance(summary, Coverage)
        assert summary.n_arms == 6
        assert math.isfinite(summary.bias_z) and math.isfinite(summary.width_ratio)


class TestDegenerateInputsAreRefused:
    """A summary over too few or identical arms names its cause; it does not divide by zero."""

    def test_a_single_arm_is_refused(self):
        with pytest.raises(ValueError, match="at least 2 arms"):
            summarise(0.0, [1.0], [0.9], [1.1], 2.262)

    def test_no_arm_is_refused(self):
        with pytest.raises(ValueError, match="at least 2 arms"):
            summarise(0.0, [], [], [], 2.262)

    def test_arms_that_all_return_the_same_point_are_refused(self):
        with pytest.raises(ValueError, match="same point"):
            summarise(0.5, [1.0] * 5, [0.9] * 5, [1.1] * 5, 2.262)

    def test_the_generic_runner_refuses_an_arm_with_no_interval(self, monkeypatch):
        import tests.coverage_check.coverage as cov

        calls = {"n": 0}

        def arm(task):
            calls["n"] += 1
            if calls["n"] == 1:
                return {}
            return {"p7a": (0.1 + 0.01 * calls["n"], 0.0, 0.3)}

        monkeypatch.setattr(cov, "_arm_task", arm)
        with pytest.raises(ValueError, match="1 of 4 arms had no formable interval"):
            cov.run_property_coverage(
                None,
                "kuhn",
                None,
                0.5,
                n_arms=4,
                n_seeds=3,
                n_hands=10,
                base_seed=0,
                truths={"p7a": 0.1},
                workers=1,
            )

    def test_a_summary_over_fewer_arms_than_were_simulated_is_refused(self):
        from tests.coverage_check.coverage import require_all_arms

        require_all_arms("p7a", 5, 5)
        with pytest.raises(ValueError, match="2 of 5 arms had no formable interval"):
            require_all_arms("p7a", 3, 5)
