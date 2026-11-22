"""
The semi-synthetic coverage check for P8 (value of information) on the DEGENERATE 2x2 Dark Hex
board. dark_hex(board_size=2) has four cells and a tree of a few hundred nodes; a pass here is
evidence about the interval arithmetic of the P8 capture-rate statistic on that tiny tree and is
not evidence about real Dark Hex. The planted answer is an exact function of the tree, the solved
reference and one planted parameter; the intervals are the production cell function's.
"""

from __future__ import annotations

import inspect
import math
import os

import numpy as np
import pytest

pytest.importorskip("pyspiel")

from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_darkhex
from lmopt.scoring import replicate
from lmopt.scoring.darkhex_posterior import gather_worlds
from lmopt.scoring.property8 import _voi_and_action_values, score_property8

from tests.coverage_check import p8
from tests.coverage_check.p8 import (
    DEGENERATE_BOARD_NOTE,
    acquisition_policy,
    collect_p8_arms,
    enumerate_p8_nodes,
    expected_scored_rows_per_hand,
    p8_coverage,
    planted_p8_capture,
    simulate_p8_arm,
)

SKILL = 0.5
GAME_KEY = "darkhex"
WORKERS = min(int(os.environ.get("COVERAGE_WORKERS", "4")), 4)


@pytest.fixture(scope="module")
def dh():
    game = load_darkhex()
    return game, solve_cfr(game, 300)


@pytest.fixture(scope="module")
def nodes(dh):
    return enumerate_p8_nodes(*dh, SKILL)


@pytest.fixture(scope="module")
def truth(nodes):
    return planted_p8_capture(nodes)


def _run(dh, truth_value, **kwargs):
    game, policy = dh
    settings = dict(
        skill=SKILL, n_arms=60, n_seeds=10, n_hands=800, base_seed=7_000, workers=1
    )
    settings.update(kwargs)
    arms = collect_p8_arms(game, GAME_KEY, policy, **settings)
    return p8_coverage(arms, truth_value, settings["n_seeds"]), arms


class TestThePlantedTruthIsExact:
    """DEGENERATE board: every check here is on dark_hex(board_size=2)."""

    def test_the_value_tables_equal_the_production_scorer_on_every_information_state(
        self, dh
    ):
        """
        The tables the agent and the truth use are written from the tree and the P8 definition.
        The scorer derives V(a|I) and the value of information
        from `gather_worlds` and `policy_value`. All 33 seat-0 states must agree to rounding.
        """
        game, policy = dh
        tables = p8._value_tables(game, policy)
        worlds = gather_worlds(game, policy, seat=0)
        assert len(tables) == len(worlds) == 33
        for info, (values, voi) in tables.items():
            scorer_values, scorer_voi = _voi_and_action_values(
                worlds[info], policy, seat=0
            )
            assert voi == pytest.approx(scorer_voi, abs=1e-12), info
            for action, value in values.items():
                assert value == pytest.approx(scorer_values[action], abs=1e-12), (
                    info,
                    action,
                )

    def test_the_constants_equal_the_scorer_defaults(self):
        """
        The truth hard-codes the informativeness threshold and the capture tolerance; a change to
        either scorer default must turn this red and not silently move the estimand.
        """
        defaults = inspect.signature(score_property8).parameters
        assert defaults["voi_threshold"].default == p8.VOI_THRESHOLD
        assert defaults["epsilon"].default == p8.EPSILON

    def test_the_first_decision_is_reached_once_per_hand(self, nodes):
        """
        The empty-board state is the seat's first node and every hand passes through it once.
        """
        root = [n for n in nodes if n.info_state.endswith("\n0\n")]
        assert len(root) == 1 and root[0].weight == pytest.approx(1.0)

    def test_capture_probability_is_a_mixture_of_deliberate_and_random_play(
        self, nodes
    ):
        """
        At every state the chance of a near-best action is skill plus the random share of them.
        """
        for n in nodes:
            expected = SKILL + (1.0 - SKILL) * len(n.best) / n.n_legal
            assert n.capture_prob == pytest.approx(expected)
        assert 0.0 < p8.planted_p8_capture(nodes) < 1.0

    def test_the_planted_value_grows_with_skill(self, dh):
        values = [
            planted_p8_capture(enumerate_p8_nodes(*dh, s))
            for s in (0.0, 0.25, 0.5, 0.75)
        ]
        assert values == sorted(values) and len(set(values)) == 4

    def test_an_agent_that_never_reaches_an_informative_state_has_no_planted_value(
        self, dh
    ):
        """
        A perfect player (skill 1) never enters a state where information is worth anything on
        this board, so the statistic has no scored row and the truth refuses to invent one.
        """
        with pytest.raises(ValueError, match="no informative state"):
            planted_p8_capture(enumerate_p8_nodes(*dh, 1.0))

    def test_the_acquisition_policy_is_a_distribution(self):
        probs = acquisition_policy((1, 3), [0, 1, 2, 3], 0.6)
        assert sum(probs.values()) == pytest.approx(1.0)
        assert probs[1] == probs[3] > probs[0] == probs[2]

    def test_the_truth_functions_take_no_sampled_data(self):
        """
        Independence from the rows is the whole point of the design: the truth is a function of
        the tree-derived nodes, and the tree-walk of the game, the reference and the planted
        parameter. No frame, no shard, no row can be passed in.
        """
        assert list(inspect.signature(planted_p8_capture).parameters) == ["nodes"]
        assert list(inspect.signature(expected_scored_rows_per_hand).parameters) == [
            "nodes"
        ]
        assert list(inspect.signature(enumerate_p8_nodes).parameters) == [
            "game",
            "reference_policy",
            "skill",
        ]
        for fn in (
            p8._value_tables,
            p8._best_actions,
            p8._cfr_value,
            p8._knowing_value,
        ):
            assert not {"df", "frame", "rows", "shard"} & set(
                inspect.signature(fn).parameters
            )


class TestTheRealScorerConvergesOnThePlantedValue:
    def test_pooled_shards_match_the_planted_capture_rate_and_row_count(
        self, dh, nodes, truth
    ):
        """
        Eight shards of 3000 hands through the production runner and scorer: the mean per-shard
        capture rate is within five standard errors of the planted value (five sigma, so a
        correct truth fails by chance about once in 1.7 million runs), and the mean scored-row
        count matches the exact expectation the same way. DEGENERATE board: 2x2 only.
        """
        game, policy = dh
        shards = simulate_p8_arm(
            game, GAME_KEY, policy, SKILL, list(range(300, 308)), 3000
        )
        scored = [
            score_property8(df, cfr_policies={GAME_KEY: policy}, games={GAME_KEY: game})
            for _, df in shards
        ]
        rates = [s["capture_rate"][GAME_KEY] for s in scored]
        rows = [s["n_informative"][GAME_KEY] for s in scored]
        for values, expected in (
            (rates, truth),
            (rows, expected_scored_rows_per_hand(nodes) * 3000),
        ):
            se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            assert abs(float(np.mean(values)) - expected) < 5 * se, (
                np.mean(values),
                expected,
                se,
            )
        wrong = planted_p8_capture(enumerate_p8_nodes(game, policy, 0.4))
        se = float(np.std(rates, ddof=1) / math.sqrt(len(rates)))
        assert abs(float(np.mean(rates)) - wrong) > 5 * se

    def test_a_perfect_agent_leaves_every_interval_unformable(self, dh):
        """
        At skill 1 no shard has a scored row, `capture_rate` is NaN in every shard, and the
        production cell function forms no interval. The check reports that and not a pass.
        DEGENERATE board: this is the 2x2 tree, where a perfect player never enters an
        informative state.
        """
        game, policy = dh
        arms = collect_p8_arms(
            game,
            GAME_KEY,
            policy,
            skill=1.0,
            n_arms=2,
            n_seeds=3,
            n_hands=60,
            base_seed=500,
        )
        assert arms == [{}, {}]
        coverage, unformable = p8_coverage(arms, 0.5, 3)
        assert coverage is None and unformable == 2


@pytest.mark.slow
@pytest.mark.coverage_run
def test_p8_capture_rate_is_covered_on_the_degenerate_2x2_board(dh, truth):
    """
    Five hundred simulated arms of ten seeds and 800 hands, scored by the production cell
    function: the published P8 capture-rate interval brackets the planted value at the nominal
    rate, unbiased and the right width. DEGENERATE board: this is dark_hex(board_size=2); a pass
    is not evidence about real Dark Hex.
    """
    (coverage, unformable), arms = _run(
        dh, truth, n_arms=500, base_seed=100_000, workers=WORKERS
    )
    print("P8 COVERAGE", DEGENERATE_BOARD_NOTE, truth, coverage, unformable)
    assert unformable == 0, (unformable, DEGENERATE_BOARD_NOTE)
    assert all(a["n_replicates"] == 10 for a in arms)
    assert min(a["n_distinct"] for a in arms) >= 8
    assert coverage.verdict() == [], (coverage, DEGENERATE_BOARD_NOTE)


@pytest.mark.slow
@pytest.mark.coverage_run
class TestEachMutationIsCaught:
    """Each mutation runs in this process (patches do not reach a pool of spawned workers)."""

    def test_halving_the_standard_error_fails(self, dh, truth, monkeypatch):
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
        (coverage, _), _ = _run(dh, truth)
        print("P8 HALVED SE", DEGENERATE_BOARD_NOTE, coverage, coverage.verdict())
        assert coverage.verdict(), coverage
        assert coverage.rate < 0.85

    def test_a_corrupted_planted_acquisition_probability_fails(self, dh):
        game, policy = dh
        planted = planted_p8_capture(enumerate_p8_nodes(game, policy, SKILL))
        (coverage, _), _ = _run(dh, planted, skill=0.6)
        print("P8 CORRUPT SKILL", DEGENERATE_BOARD_NOTE, coverage, coverage.verdict())
        assert any("bias" in f or "coverage" in f for f in coverage.verdict())
        assert coverage.rate < 0.5
