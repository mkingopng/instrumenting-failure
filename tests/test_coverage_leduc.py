"""
The Leduc coverage check: the rank-space planted truth is exact, the real scorers reach it, and the
published replicate intervals bracket it at the nominal rate. The fast tests run in the inner loop;
the full coverage run is marked slow.
"""

from __future__ import annotations

import json
import math
import tempfile

import numpy as np
import pytest

pytest.importorskip("pyspiel")

import pyspiel

from lmopt.elicitation.posterior import compute_true_posterior
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_leduc
from lmopt.scoring.leduc_rank import marginalise_to_rank
from lmopt.scoring.property5 import score_property5
from lmopt.scoring.property7 import score_property7

from tests.coverage_check import arm
from tests.coverage_check.leduc import (
    GAME_KEY,
    P5_KEY,
    P7A_KEY,
    LeducReporter,
    enumerate_rank_nodes,
    holdable_ranks,
    leduc_truths,
    planted_leduc_p5_margin,
    planted_leduc_p7a,
    run_leduc_coverage,
    stated_rank_belief,
)

LAM = 0.3


@pytest.fixture(scope="module")
def leduc():
    game = load_leduc()
    return game, solve_cfr(game, 100)


@pytest.fixture(scope="module")
def nodes(leduc):
    return enumerate_rank_nodes(*leduc)


def _replay(game, history) -> pyspiel.State:
    state = game.new_initial_state()
    for action in history:
        state.apply_action(action)
    return state


def _production_rank_posterior(game, policy, history) -> np.ndarray:
    card = compute_true_posterior(game, _replay(game, history), 0, policy)
    rank = marginalise_to_rank({str(i): float(p) for i, p in enumerate(card)})
    return np.array([rank[str(r)] for r in range(3)])


class TestThePlantedTruthIsExact:
    def test_the_tree_rank_posterior_equals_the_production_one_on_every_information_state(
        self, leduc, nodes
    ):
        """
        The truth marginalises its own tree-derived card posterior with its own reshape; the runner
        logs a replayed-hypothesis posterior that the scorers marginalise with `marginalise_to_rank`.
        They must agree to rounding on every information state, in card space and in rank space.
        """
        game, policy = leduc
        assert len(nodes) == 468
        for node in nodes:
            state = _replay(game, node.history)
            card = compute_true_posterior(game, state, 0, policy)
            assert np.allclose(node.card_posterior, card, atol=1e-12), node.info_state
            rank = _production_rank_posterior(game, policy, node.history)
            assert np.allclose(node.rank_posterior, rank, atol=1e-12), node.info_state

    def test_the_posterior_excludes_the_own_and_the_public_card(self, leduc, nodes):
        """
        The hypothesis set excludes the observer's card and, in round two, the public card, so the
        posterior is zero on both and the holdable mask drops a rank whose copies are both gone.
        """
        game, _ = leduc
        later = [n for n in nodes if "Round 2" in n.info_state]
        assert later
        for node in nodes:
            public = [
                a
                for i, a in enumerate(node.history)
                if i >= 2 and _is_chance(game, node.history, i)
            ]
            for card in [node.history[0], *public]:
                assert node.card_posterior[card] == 0.0
            mask = holdable_ranks(game, node.history, 0)
            assert node.rank_posterior[~mask].sum() == 0.0
        assert any(not n.holdable.all() for n in later)

    def test_the_visit_weights_are_a_distribution_over_first_nodes(self, nodes):
        """
        The six round-one first nodes (one per own card) carry weight 1 between them, and every
        weight is positive.
        """
        first = [n for n in nodes if len(n.history) == 2]
        assert sum(n.weight for n in first) == pytest.approx(1.0)
        assert all(n.weight > 0.0 for n in nodes)

    def test_the_visit_weights_after_a_three_action_move_are_the_hand_computed_probabilities(
        self, leduc, nodes
    ):
        """
        The observer checks (one half of two legal actions), the opponent raises (the reference
        policy decides), the observer then faces three legal actions and calls (one third), and the
        public card is dealt. The total weight of the observer's round-two first nodes reached that
        way is summed here straight from the policy over the thirty deals, with no tree walk.
        """
        game, policy = leduc
        reached = [
            n for n in nodes if len(n.history) == 6 and n.history[2:5] == (1, 2, 1)
        ]
        assert reached
        expected = 0.0
        for observer_card in range(6):
            for opponent_card in range(6):
                if observer_card == opponent_card:
                    continue
                after_check = _replay(game, (observer_card, opponent_card, 1))
                raises = policy.action_probabilities(after_check)[2]
                faced = _replay(game, (observer_card, opponent_card, 1, 2))
                assert len(faced.legal_actions()) == 3
                expected += (1 / 30) * 0.5 * raises * (1 / 3)
        assert sum(n.weight for n in reached) == pytest.approx(expected)

    def test_a_truthful_reporter_has_zero_error_and_a_negative_margin(self, nodes):
        """
        With lam = 0 the stated belief is the true rank posterior.
        """
        assert planted_leduc_p7a(nodes, 0.0) == pytest.approx(0.0, abs=1e-15)
        assert planted_leduc_p5_margin(nodes, 0.0) < 0.0

    def test_the_error_grows_with_the_departure(self, nodes):
        values = [planted_leduc_p7a(nodes, lam) for lam in (0.0, 0.25, 0.5, 1.0)]
        assert values == sorted(values) and len(set(values)) == 4

    def test_the_stated_belief_is_a_three_key_simplex_with_holdable_support(
        self, nodes
    ):
        """
        The stated belief the reporter plants sums to one over three rank keys and puts no mass on
        a rank the opponent cannot hold.
        """
        for node in nodes:
            stated = stated_rank_belief(node.rank_posterior, node.holdable, LAM)
            assert stated.shape == (3,)
            assert stated.sum() == pytest.approx(1.0)
            assert stated[~node.holdable].sum() == 0.0

    def test_the_reporter_states_what_the_truth_plants_on_every_information_state(
        self, leduc, nodes
    ):
        """
        Through the real reporter entrypoint: the JSON it sets on `last_elicited_belief` has the
        three rank keys the leduc parser returns and equals the planted stated belief on all 468
        states.
        """
        game, policy = leduc
        reporter = LeducReporter(policy, LAM, seed=1)
        reporter.reset(game, 0, "cfr_ref")
        for node in nodes:
            reporter.act(_replay(game, node.history))
            stated = json.loads(reporter.last_elicited_belief)
            assert list(stated) == ["0", "1", "2"]
            expected = stated_rank_belief(node.rank_posterior, node.holdable, LAM)
            assert np.allclose([stated[k] for k in stated], expected, atol=1e-12)

    def test_the_truth_functions_take_no_sampled_data(self):
        """
        Independence from the rows is the design: the truth is a function of tree-derived nodes and
        the planted parameter only.
        """
        import inspect

        for fn in (planted_leduc_p7a, planted_leduc_p5_margin):
            assert list(inspect.signature(fn).parameters) == ["nodes", "lam"]


def _is_chance(game, history, index) -> bool:
    return _replay(game, history[:index]).is_chance_node()


def _score_shards(game, policy, seeds, n_hands, monkeypatch):
    monkeypatch.setattr(arm, "PrescribedReporter", LeducReporter)
    p7, p5 = [], []
    with tempfile.TemporaryDirectory() as directory:
        for seed in seeds:
            shard = arm.simulate_shard(
                game, GAME_KEY, policy, LAM, seed, n_hands, directory
            )
            p7.append(
                score_property7(
                    shard, cfr_policies={GAME_KEY: policy}, games={GAME_KEY: game}
                )["tvd_mean"][GAME_KEY]
            )
            p5.append(score_property5(shard)["per_pair_margin"][GAME_KEY])
    return np.array(p7), np.array(p5)


def test_the_real_scorers_reach_the_planted_values_on_large_samples(
    leduc, nodes, monkeypatch
):
    """
    Six shards of 4000 hands each, scored by the production scorers, average to the planted P7a
    and P5 margin. The tolerance is five standard errors of the shard mean, the standard error
    taken from the spread of the six shard values (the seeds are fixed, so the outcome is
    deterministic); the per-hand noise is far larger than any plausible bias, so a scorer that
    mis-pools or mis-marginalises is several tolerances away.
    """
    game, policy = leduc
    p7, p5 = _score_shards(game, policy, range(9100, 9106), 4000, monkeypatch)
    truths = leduc_truths(nodes, LAM)
    for values, key in ((p7, P7A_KEY), (p5, P5_KEY)):
        se = values.std(ddof=1) / math.sqrt(len(values))
        assert abs(values.mean() - truths[key]) < 5 * se, (
            key,
            values.mean(),
            truths[key],
            se,
        )
        assert se < 0.005


@pytest.mark.slow
@pytest.mark.coverage_run
def test_the_published_intervals_cover_the_planted_leduc_truth(leduc, nodes):
    """
    Five hundred simulated arms of ten shards and 800 hands, scored by the production cell
    function, bracket the planted P7a and P5 margin at the agreed rate, without bias and with
    intervals of the right width.
    """
    game, policy = leduc
    result = run_leduc_coverage(
        game,
        policy,
        LAM,
        n_arms=500,
        n_seeds=10,
        n_hands=800,
        base_seed=20_000,
        truths=leduc_truths(nodes, LAM),
        workers=4,
    )
    for key, summary in result.items():
        print(key, summary)
        assert summary.verdict() == [], (key, summary)


@pytest.mark.slow
@pytest.mark.coverage_run
class TestEachMutationIsCaught:
    """Each mutation runs in this process; patches do not reach a pool of spawned workers."""

    def _run(self, leduc, nodes, **kwargs):
        game, policy = leduc
        settings = dict(n_arms=60, n_seeds=10, n_hands=200, base_seed=40_000, workers=1)
        settings.update(kwargs)
        return run_leduc_coverage(
            game, policy, LAM, truths=leduc_truths(nodes, LAM), **settings
        )

    def test_halving_the_standard_error_fails_both_properties(
        self, leduc, nodes, monkeypatch
    ):
        from lmopt.scoring import replicate

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
        result = self._run(leduc, nodes)
        for key in (P7A_KEY, P5_KEY):
            assert result[key].verdict(), (key, result[key])
            assert result[key].rate < 0.85

    def test_scoring_against_the_wrong_posterior_fails_both_properties(
        self, leduc, nodes, monkeypatch
    ):
        import polars as pl

        real = arm.simulate_shard
        uniform = json.dumps({str(i): 1 / 6 for i in range(6)})

        def wrong_posterior(*args, **kwargs):
            frame = real(*args, **kwargs)
            return frame.with_columns(
                pl.when(pl.col("true_posterior").is_not_null())
                .then(pl.lit(uniform))
                .otherwise(pl.col("true_posterior"))
                .alias("true_posterior")
            )

        monkeypatch.setattr(arm, "simulate_shard", wrong_posterior)
        result = self._run(leduc, nodes)
        assert result[P7A_KEY].verdict() and result[P5_KEY].verdict()

    def test_a_rescaled_margin_fails_p5_and_leaves_p7a_unbiased(
        self, leduc, nodes, monkeypatch
    ):
        from lmopt.scoring import property5

        real = property5.score_property5

        def tripled(df, *args, **kwargs):
            out = real(df, *args, **kwargs)
            out["per_pair_margin"] = {
                g: 3 * v for g, v in out["per_pair_margin"].items()
            }
            return out

        monkeypatch.setattr(property5, "score_property5", tripled)
        result = self._run(leduc, nodes)
        assert result[P5_KEY].verdict() and result[P5_KEY].rate < 0.5
        assert abs(result[P7A_KEY].bias_z) < 3.0


class TestDegenerateLeducInputsAreRefused:
    """An empty enumeration, or arms that drop out, are named; they are not a division or a smaller sample."""

    def test_pooling_over_no_nodes_is_refused(self):
        from tests.coverage_check.leduc import _pooled

        with pytest.raises(ValueError, match="no weight to pool over"):
            _pooled([], lambda node: 1.0)

    def test_an_arm_with_no_interval_stops_the_run(self, monkeypatch):
        import tests.coverage_check.leduc as ld

        calls = {"n": 0}

        def arm(task):
            calls["n"] += 1
            if calls["n"] == 1:
                return {}
            return {"p7a": (0.1 + 0.01 * calls["n"], 0.0, 0.3)}

        monkeypatch.setattr(ld, "_leduc_arm_task", arm)
        with pytest.raises(ValueError, match="1 of 4 arms had no formable interval"):
            ld.run_leduc_coverage(
                None,
                None,
                0.5,
                n_arms=4,
                n_seeds=3,
                n_hands=10,
                base_seed=0,
                truths={"p7a": 0.1},
                workers=1,
            )
