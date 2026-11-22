"""
The semi-synthetic coverage check on Kuhn poker: P7a (mean total variation between the stated and
the true posterior) and the P5 per-pair margin. The planted answers are exact functions of the
tree; the intervals are the production cell function's. See tests/coverage_check/__init__.py.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import polars as pl
import pytest

pytest.importorskip("pyspiel")

from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_kuhn
from lmopt.scoring import property5, property7, replicate
from lmopt.scoring.replicate import t_crit_95

from tests.coverage_check import arm as arm_module
from tests.coverage_check.arm import simulate_arm
from tests.coverage_check.coverage import (
    collect_arms,
    run_property_coverage,
    summarise,
)
from tests.coverage_check.truth import enumerate_nodes, planted_p5_margin, planted_p7a

LAM = 0.3
P7A = "P7a_belief_error"
P5 = "P5_calibration_per_pair_margin"
WORKERS = int(os.environ.get("COVERAGE_WORKERS", "4"))


@pytest.fixture(scope="module")
def kuhn():
    game = load_kuhn()
    return game, solve_cfr(game, 300)


@pytest.fixture(scope="module")
def truths(kuhn):
    nodes = enumerate_nodes(*kuhn)
    return {P7A: planted_p7a(nodes, LAM), P5: planted_p5_margin(nodes, LAM)}


def _run(kuhn, truths, **kwargs):
    game, policy = kuhn
    settings = dict(
        lam=LAM, n_arms=60, n_seeds=10, n_hands=200, base_seed=7_000, truths=truths
    )
    settings.update(kwargs)
    return run_property_coverage(game, "kuhn", policy, **settings)


def test_the_real_scorers_converge_on_the_planted_values(kuhn, truths):
    """
    Ten shards of 2000 hands scored by the production per-shard scorers have a mean within five
    standard errors of the planted truth, so the harness and the estimand definitions agree.
    """
    game, policy = kuhn
    shards = simulate_arm(game, "kuhn", policy, LAM, list(range(300, 310)), 2000)
    p7 = [
        property7.score_property7(
            df, cfr_policies={"kuhn": policy}, games={"kuhn": game}
        )["tvd_mean"]["kuhn"]
        for _, df in shards
    ]
    p5 = [property5.score_property5(df)["per_pair_margin"]["kuhn"] for _, df in shards]
    for values, key in ((p7, P7A), (p5, P5)):
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        assert abs(mean - truths[key]) < 5 * se, (key, mean, truths[key], se)


@pytest.mark.slow
@pytest.mark.coverage_run
def test_kuhn_p7a_and_the_p5_margin_are_covered(kuhn, truths):
    """
    Five hundred simulated arms of ten seeds and 800 hands: both published intervals bracket the
    planted truth at the nominal rate, unbiased and the right width.
    """
    result = _run(
        kuhn, truths, n_arms=500, n_hands=800, base_seed=100_000, workers=WORKERS
    )
    for key, summary in result.items():
        assert summary.verdict() == [], (key, summary)


@pytest.mark.slow
@pytest.mark.coverage_run
class TestEachMutationIsCaught:
    """Each mutation runs in this process (patches do not reach a pool of spawned workers)."""

    def test_halving_the_standard_error_fails_both_properties(
        self, kuhn, truths, monkeypatch
    ):
        real = replicate.replicate_ci

        def halved(values, level=0.95):
            out = real(values, level)
            if not math.isnan(out["lo"]):
                out = dict(
                    out,
                    lo=out["point"] - (out["point"] - out["lo"]) / 2,
                    hi=out["point"] + (out["hi"] - out["point"]) / 2,
                )
            return out

        monkeypatch.setattr(replicate, "replicate_ci", halved)
        result = _run(kuhn, truths)
        for key in (P7A, P5):
            assert result[key].verdict(), (key, result[key])
            assert result[key].rate < 0.85

    def test_a_normal_critical_value_at_five_seeds_fails(
        self, kuhn, truths, monkeypatch
    ):
        real = replicate.replicate_ci

        def z_interval(values, level=0.95):
            out = real(values, level)
            if not math.isnan(out["lo"]):
                scale = 1.96 / t_crit_95(out["n_replicates"] - 1)
                out = dict(
                    out,
                    lo=out["point"] - (out["point"] - out["lo"]) * scale,
                    hi=out["point"] + (out["hi"] - out["point"]) * scale,
                )
            return out

        monkeypatch.setattr(replicate, "replicate_ci", z_interval)
        result = _run(kuhn, truths, n_arms=200, n_seeds=5, n_hands=200)
        for key in (P7A, P5):
            assert result[key].verdict(), (key, result[key])

    def test_scoring_against_the_wrong_posterior_fails(self, kuhn, truths, monkeypatch):
        real = arm_module.simulate_shard
        uniform = json.dumps({"0": 1 / 3, "1": 1 / 3, "2": 1 / 3})

        def wrong_posterior(*args, **kwargs):
            frame = real(*args, **kwargs)
            return frame.with_columns(
                pl.when(pl.col("elicited_belief").is_not_null())
                .then(pl.lit(uniform))
                .otherwise(None)
                .alias("true_posterior")
            )

        monkeypatch.setattr(arm_module, "simulate_shard", wrong_posterior)
        result = _run(kuhn, truths)
        assert result[P7A].verdict() and result[P5].verdict()

    def test_a_rescaled_margin_fails_p5_and_leaves_p7a_unbiased(
        self, kuhn, truths, monkeypatch
    ):
        real = property5.score_property5

        def tripled(df, *args, **kwargs):
            out = real(df, *args, **kwargs)
            out["per_pair_margin"] = {
                g: 3 * v for g, v in out["per_pair_margin"].items()
            }
            return out

        monkeypatch.setattr(property5, "score_property5", tripled)
        result = _run(kuhn, truths)
        assert result[P5].verdict() and result[P5].rate < 0.5
        assert abs(result[P7A].bias_z) < 3.0

    def test_a_truth_shifted_by_five_standard_errors_fails(self, kuhn, truths):
        """
        The planted truth is not secretly tied to the sampled rows: the unmutated intervals fail
        against a truth moved five standard errors (expected coverage under one percent), so the truth
        is a free-standing number.
        """
        game, policy = kuhn
        arms = collect_arms(
            game, "kuhn", policy, LAM, 60, 10, 200, 9_000, (P7A,), workers=WORKERS
        )
        points, los, his = arms[P7A]
        sd = float(np.std(points, ddof=1))
        shifted = summarise(truths[P7A] + 5 * sd, points, los, his, t_crit_95(9))
        assert shifted.rate < 0.1 and shifted.verdict()

    def test_the_tautological_control_cannot_fail(self, kuhn, truths, monkeypatch):
        """
        The design this one replaces: take the truth from the arm's own rows, so "the point is
        inside its own interval". Under the halved-standard-error mutation the real truth fails
        while the control still reads 100 percent, which is exactly why that design was refused.
        """
        real = replicate.replicate_ci

        def halved(values, level=0.95):
            out = real(values, level)
            if not math.isnan(out["lo"]):
                out = dict(
                    out,
                    lo=out["point"] - (out["point"] - out["lo"]) / 2,
                    hi=out["point"] + (out["hi"] - out["point"]) / 2,
                )
            return out

        monkeypatch.setattr(replicate, "replicate_ci", halved)
        game, policy = kuhn
        points, los, his = collect_arms(
            game, "kuhn", policy, LAM, 60, 10, 200, 11_000, (P7A,)
        )[P7A]
        control = sum(lo <= p <= hi for p, lo, hi in zip(points, los, his, strict=True))
        real_truth = summarise(truths[P7A], points, los, his, t_crit_95(9))
        assert control == len(points)
        assert real_truth.verdict() and real_truth.rate < 0.85
