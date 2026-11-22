"""The paired probe scorer: P1 and P4 from base+probe arms.

The estimand is a policy difference between two arms that differ only in
rendered prompt bytes, so every test here builds two frames that differ in
exactly one stated way and asserts what the scorer reports about that
difference. The join key is (game, canonical info-set), which the probe
variants leave untouched: relabelling renames what the prompt offers, never
what OpenSpiel calls the information state.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from lmopt.prompting.probe_variants import (
    PAYOFF_SCALE_VARIANT_NAME,
    RELABEL_VARIANT_NAME,
    variant_caveat,
)
from lmopt.scoring.property_paired import (
    score_paired,
    score_property1,
    score_property4,
)


def _arm(rows, *, prompt_variant="base", probe_type="none", actor="lm"):
    """A minimal arm frame: one row per decision node, as the writer emits."""
    return pl.DataFrame(
        [
            {
                "game": game,
                "info_set": info_set,
                "parsed_action": action,
                "actor": actor,
                "prompt": "p",
                "prompt_variant": prompt_variant,
                "probe_type": probe_type,
            }
            for game, info_set, action in rows
        ]
    )


def _rows(game, info_set, counts):
    """`counts` maps action -> how many rows chose it at this info-set."""
    out = []
    for action, n in counts.items():
        out.extend([(game, info_set, action)] * n)
    return out


class TestNullPair:
    def test_scoring_an_arm_against_itself_gives_tv_mean_exactly_zero(self):
        """P-null-pair.

        Not reddened by pooling the rows per game: an
        arm pooled against itself is still identical, so the pooling defect
        needs its own fixture, which is TestDetectsShift's."""
        arm = _arm(_rows("kuhn", "0", {0: 8, 1: 2}) + _rows("kuhn", "1", {0: 3, 1: 7}))
        out = score_paired(arm, arm, actor="lm")
        assert out["tv_mean"] == 0.0
        assert out["tv_by_infoset"]["kuhn"] == {"0": 0.0, "1": 0.0}

    def test_a_deliberately_divergent_pair_reports_a_positive_tv_mean(self):
        """Falsify-the-falsifier for the zero above: the same call on a pair
        that does differ must report a positive mean, so the exact-zero test
        cannot be passing because the scorer always returns zero."""
        base = _arm(_rows("kuhn", "0", {0: 10}))
        probe = _arm(
            _rows("kuhn", "0", {1: 10}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert out["tv_mean"] == pytest.approx(1.0)


class TestDetectsShift:
    def test_a_shift_of_zero_point_four_is_reported_at_that_info_set(self):
        """P-detects-shift. The fixture is built so pooling the rows of a game
        hides the shift: info-set "0" moves 0.4 of its mass from action 0 to
        action 1 and info-set "1" moves 0.4 the other way, so the game-level
        pooled distribution is identical between the arms."""
        base = _arm(_rows("kuhn", "0", {0: 10}) + _rows("kuhn", "1", {1: 10}))
        probe = _arm(
            _rows("kuhn", "0", {0: 6, 1: 4}) + _rows("kuhn", "1", {1: 6, 0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert out["tv_by_infoset"]["kuhn"]["0"] == pytest.approx(0.4, abs=1e-12)
        assert out["tv_by_infoset"]["kuhn"]["1"] == pytest.approx(0.4, abs=1e-12)

    def test_an_action_seen_in_only_one_arm_still_counts_toward_the_distance(self):
        """The union of actions, not the intersection."""
        base = _arm(_rows("kuhn", "0", {0: 10}))
        probe = _arm(
            _rows("kuhn", "0", {0: 5, 1: 5}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert out["tv_by_infoset"]["kuhn"]["0"] == pytest.approx(0.5)


class TestCoverageIsHonest:
    def test_a_missing_info_set_lowers_coverage_and_is_named(self):
        """P-coverage-honest."""
        base = _arm(_rows("kuhn", "0", {0: 4}) + _rows("kuhn", "1", {0: 4}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert out["paired_coverage"] == pytest.approx(0.5)
        assert out["missing_infosets"]["kuhn"] == ["1"]

    def test_an_info_set_only_the_probe_arm_reached_is_not_scored(self):
        """Coverage is measured against the base arm's info-sets, and a
        probe-only info-set has no base distribution to be compared with."""
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4}) + _rows("kuhn", "9", {1: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert set(out["tv_by_infoset"]["kuhn"]) == {"0"}
        assert out["paired_coverage"] == pytest.approx(1.0)


class TestPopulationIsStated:
    def test_the_actor_filter_is_reported(self):
        arm = _arm(_rows("kuhn", "0", {0: 2}))
        assert score_paired(arm, arm, actor="lm")["actor_filter"] == "actor=lm"

    def test_only_the_named_actors_rows_are_scored(self):
        base = _arm(_rows("kuhn", "0", {0: 4}))
        opponent = _arm(_rows("kuhn", "0", {1: 4}), actor="cfr_ref")
        base = pl.concat([base, opponent])
        probe = _arm(
            _rows("kuhn", "0", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert out["tv_mean"] == 0.0

    def test_a_frame_without_the_variant_columns_is_refused(self):
        """No silent fallbacks: an arm that carries no
        prompt_variant cannot be identified."""
        arm = _arm(_rows("kuhn", "0", {0: 2})).drop("prompt_variant")
        with pytest.raises(ValueError, match="prompt_variant"):
            score_paired(arm, arm, actor="lm")

    def test_an_arm_that_mixes_two_variants_is_refused(self):
        """One arm is one rendering."""
        arm = pl.concat(
            [
                _arm(_rows("kuhn", "0", {0: 2})),
                _arm(
                    _rows("kuhn", "0", {0: 2}),
                    prompt_variant=RELABEL_VARIANT_NAME,
                    probe_type="relabel",
                ),
            ]
        )
        with pytest.raises(ValueError, match="one prompt_variant"):
            score_paired(arm, arm, actor="lm")


class TestPropertyWrappers:
    def _pair(self, probe_type, variant):
        base = _arm(_rows("kuhn", "0", {0: 6, 1: 4}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4, 1: 6}),
            prompt_variant=variant,
            probe_type=probe_type,
        )
        return base, probe

    def test_property1_scores_a_relabel_pair(self):
        base, probe = self._pair("relabel", RELABEL_VARIANT_NAME)
        out = score_property1(base, probe, actor="lm")
        assert out["tv_by_infoset"]["kuhn"]["0"] == pytest.approx(0.2)
        assert out["probe_type"] == "relabel"

    def test_property1_refuses_a_payoff_scale_probe_arm(self):
        """The wrapper names the property's own probe type.

        The match string is the probe_type clause specifically, not the bare
        word "relabel": under the mutation the variant assertion still fires
        with a message containing "relabel_v1", so a looser pattern would pass
        under its own named mutation."""
        base, probe = self._pair("payoff_scale", PAYOFF_SCALE_VARIANT_NAME)
        with pytest.raises(
            ValueError, match="probe_type 'payoff_scale', expected 'relabel'"
        ):
            score_property1(base, probe, actor="lm")

    def test_property1_refuses_a_base_arm_that_carries_a_variant_label(self):
        """P-end-to-end's mutation, at the assertion it lands on: if the runner
        stamped the variant on the base arm's rows too, the pair no longer
        differs in the one way the estimand requires."""
        base = _arm(
            _rows("kuhn", "0", {0: 6, 1: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        probe = _arm(
            _rows("kuhn", "0", {0: 4, 1: 6}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        with pytest.raises(ValueError, match="base"):
            score_property1(base, probe, actor="lm")

    def test_property4_scores_a_payoff_scale_pair(self):
        base, probe = self._pair("payoff_scale", PAYOFF_SCALE_VARIANT_NAME)
        out = score_property4(base, probe, actor="lm")
        assert out["probe_type"] == "payoff_scale"
        assert out["probe_variant"] == PAYOFF_SCALE_VARIANT_NAME

    def test_property4_refuses_a_relabel_probe_arm(self):
        base, probe = self._pair("relabel", RELABEL_VARIANT_NAME)
        with pytest.raises(ValueError, match="payoff_scale"):
            score_property4(base, probe, actor="lm")


class TestPerGameRowsCarryTheRegistrysAbsences:
    """The coverage table must consume data, not prose."""

    def test_property1_emits_a_row_for_every_registered_game_not_just_the_played_one(
        self,
    ):
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        rows = {
            r["game"]: r for r in score_property1(base, probe, actor="lm")["per_game"]
        }
        assert set(rows) == {"kuhn", "leduc", "goofspiel", "liarsdice", "darkhex"}
        goof = rows["goofspiel"]
        assert goof["available"] is True
        assert goof["status"] == "awaiting-data"
        assert goof["reason"] is None

    def test_a_staged_registry_absence_still_reaches_the_per_game_row(
        self, monkeypatch
    ):
        """The recorded-absence branch is live code with no live absence to
        reach it: prompt version 7 gave every (game, probe) pair a variant, so
        the branch is exercised by putting one absence back rather than by
        finding one. Staging it is the point -- a status the table can still
        emit must keep a test, or the first withdrawn variant lands on an
        untested path.
        """
        import lmopt.prompting.probe_variants as pv

        monkeypatch.setitem(
            pv._SPECS_BY_PROBE,
            "relabel",
            {g: s for g, s in pv.RELABEL_SPECS.items() if g != "goofspiel"},
        )
        monkeypatch.setitem(
            pv.VARIANT_ABSENT_REASONS,
            "goofspiel",
            {
                "relabel": pv.VariantAbsence(
                    game="goofspiel",
                    probe_type="relabel",
                    reason="SENTINEL-REASON",
                    would_need="SENTINEL-WOULD-NEED",
                )
            },
        )
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        rows = {
            r["game"]: r for r in score_property1(base, probe, actor="lm")["per_game"]
        }
        goof = rows["goofspiel"]
        assert goof["status"] == "recorded-absence"
        assert goof["available"] is False
        assert goof["reason"] == "SENTINEL-REASON"
        assert goof["would_need"] == "SENTINEL-WOULD-NEED"

    def test_property4_now_covers_all_five_games_with_one_measured(self):
        """P4 was 1 of 5 games; prompt version 7 states a payoff in every game,
        so it is 5 of 5 and the four uncollected ones read `awaiting-data`, not
        `recorded-absence`."""
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4}),
            prompt_variant=PAYOFF_SCALE_VARIANT_NAME,
            probe_type="payoff_scale",
        )
        rows = {
            r["game"]: r for r in score_property4(base, probe, actor="lm")["per_game"]
        }
        assert not {g for g, r in rows.items() if r["status"] == "recorded-absence"}
        assert {g for g, r in rows.items() if r["status"] == "awaiting-data"} == {
            "leduc",
            "goofspiel",
            "liarsdice",
            "darkhex",
        }
        assert rows["kuhn"]["status"] == "measured"
        # Each row must carry whatever the registry records for its pair and nothing
        # else: no caveat for Kuhn, whose prompt states every chip magnitude it
        # renders, and Dark Hex's weakest-probe caveat for Dark Hex. Written against
        # the registry rather than against literals so that re-registering or
        # withdrawing a caveat does not silently stop this checking the propagation.
        for game in ("kuhn", "darkhex"):
            registered = variant_caveat(game, "payoff_scale")
            assert rows[game]["caveat"] == (registered.caveat if registered else None)
        assert variant_caveat("darkhex", "payoff_scale") is not None

    def test_a_game_with_a_variant_but_no_rows_reads_awaiting_data(self):
        """A variant that exists and was not collected is a different state
        from one that cannot exist."""
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        rows = {
            r["game"]: r for r in score_property1(base, probe, actor="lm")["per_game"]
        }
        assert rows["leduc"]["status"] == "awaiting-data"
        assert rows["leduc"]["available"] is True
        assert rows["leduc"]["reason"] is None

    def test_a_depth_variant_lands_on_its_familys_row_as_measured(self):
        """A depth variant is the family it belongs to, not an unknown game.

        The registry is family-keyed (one relabel scheme serves every dark hex
        board) while the `game` column carries the registry key actually
        played, so `darkhex_3x3` rows must be reported on the `darkhex` row.
        Synthetic `game` column values only; no real game is loaded.
        """
        base = _arm(
            _rows("darkhex_3x3", "a", {0: 4}) + _rows("darkhex_3x3", "b", {0: 4})
        )
        probe = _arm(
            _rows("darkhex_3x3", "a", {0: 2, 1: 2}) + _rows("darkhex_3x3", "b", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        rows = {
            r["game"]: r for r in score_property1(base, probe, actor="lm")["per_game"]
        }
        assert set(rows) == {"kuhn", "leduc", "goofspiel", "liarsdice", "darkhex"}
        dark = rows["darkhex"]
        assert dark["status"] == "measured"
        assert dark["n_infosets_paired"] == 2
        assert dark["frame_games"] == ["darkhex_3x3"]
        # Pooled over the family's raw distances (0.5 and 0.0), not averaged
        # per board: mutation -- mean the per-board means -> unchanged here on
        # one board, which is why the two-board test below exists.
        assert dark["tv_mean"] == pytest.approx(0.25)
        assert dark["paired_coverage"] == pytest.approx(1.0)

    def test_two_boards_of_one_family_pool_by_infoset_not_by_board(self):
        """The two darkhex-family registry keys
        `darkhex` and `darkhex_3x3` are used here as the "two boards",
        synthetic `game` column values only, no real game loaded.
        """
        base = _arm(
            _rows("darkhex", "a", {0: 4})
            + _rows("darkhex", "b", {0: 4})
            + _rows("darkhex", "c", {0: 4})
            + _rows("darkhex", "d", {0: 4})
            + _rows("darkhex_3x3", "z", {0: 4})
        )
        probe = _arm(
            _rows("darkhex", "a", {0: 4})
            + _rows("darkhex", "b", {0: 4})
            + _rows("darkhex", "c", {0: 4})
            + _rows("darkhex", "d", {0: 4})
            + _rows("darkhex_3x3", "z", {0: 2, 1: 2}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        dark = {
            r["game"]: r for r in score_property1(base, probe, actor="lm")["per_game"]
        }["darkhex"]
        assert dark["frame_games"] == ["darkhex", "darkhex_3x3"]
        assert dark["n_infosets_paired"] == 5
        assert dark["tv_mean"] == pytest.approx(0.1)

    def test_per_game_measured_rows_carry_their_own_tv_and_coverage(self):
        base = _arm(_rows("kuhn", "0", {0: 10}) + _rows("kuhn", "1", {0: 10}))
        probe = _arm(
            _rows("kuhn", "0", {0: 5, 1: 5}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        row = {
            r["game"]: r for r in score_property1(base, probe, actor="lm")["per_game"]
        }["kuhn"]
        assert row["tv_mean"] == pytest.approx(0.5)
        assert row["paired_coverage"] == pytest.approx(0.5)
        assert row["missing_infosets"] == ["1"]

    def test_per_game_measured_rows_carry_their_own_visit_counts(self):
        """A per-game `tv_mean` without its n cannot be
        told from sampling noise, the same reasoning `_visit_summary` already
        applies at the property level (docstring, `score_paired`). Only "0" is
        paired here (10 base / 10 probe rows); "1" is base-only (5 rows) and
        must not leak into the paired count.
        """
        base = _arm(_rows("kuhn", "0", {0: 10}) + _rows("kuhn", "1", {0: 5}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4, 1: 6}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        row = {
            r["game"]: r for r in score_property1(base, probe, actor="lm")["per_game"]
        }["kuhn"]
        assert row["n_visits_base"] == 10
        assert row["n_visits_probe"] == 10

    def test_per_game_rows_with_no_frame_carry_no_visit_count(self):
        """A game the arm never played gets `None`, not `0` -- the same
        "no rows here" vs "rows here and nothing paired" distinction
        `tv_mean`/`paired_coverage` already draw (comment above their
        assignment)."""
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "0", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        rows = {
            r["game"]: r for r in score_property1(base, probe, actor="lm")["per_game"]
        }
        assert rows["goofspiel"]["n_visits_base"] is None
        assert rows["goofspiel"]["n_visits_probe"] is None


class TestEmptyPairIsNotAZero:
    def test_no_paired_info_sets_gives_nan_not_zero(self):
        """A pair with nothing in common has no measurement, and 0.0 would
        read as "the policy did not move"."""
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "9", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert math.isnan(out["tv_mean"])
        assert out["paired_coverage"] == 0.0


class TestTheJoinKeySurvivesTheRelabel:
    """The join is on OpenSpiel info-set strings; the variant changes only the
    rendered prompt. Asserted against the real renderer, not by argument."""

    def test_relabelling_kuhn_changes_prompts_and_no_info_set_string(self):
        pytest.importorskip("pyspiel")
        from lmopt.games.loaders import load_kuhn
        from lmopt.prompting.fidelity import enumerate_infosets
        from lmopt.prompting.probe_variants import RELABEL_SPECS

        game = load_kuhn()
        canonical = enumerate_infosets(game, player=0)
        variant = enumerate_infosets(game, player=0, spec=RELABEL_SPECS["kuhn"])
        assert set(canonical) == set(variant)
        assert canonical != variant


class TestVisitCountsAndWeighting:
    """TV between two multinomials is upward-biased at finite
    n, so a per-info-set number carries no meaning without its sample size: a
    perfectly invariant pair at 20-40 visits per info-set reads a positive
    tv_mean purely from sampling noise. The counts make the null band
    computable offline; the unweighted mean stays the headline because the
    estimand treats each info-set as one comparison unit.
    """

    def test_per_info_set_visit_counts_are_reported(self):
        base = _arm(_rows("kuhn", "0", {0: 7, 1: 3}))
        probe = _arm(
            _rows("kuhn", "0", {0: 2}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert out["visits_by_infoset"]["kuhn"]["0"] == {"n_base": 10, "n_probe": 2}

    def test_visit_weighting_moves_the_mean_when_visits_are_unequal(self):
        """The secondary key must actually be a different estimator."""
        base = _arm(_rows("kuhn", "0", {0: 100}) + _rows("kuhn", "1", {0: 2}))
        probe = _arm(
            _rows("kuhn", "0", {0: 100}) + _rows("kuhn", "1", {1: 2}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert out["tv_mean"] == pytest.approx(0.5)
        assert out["tv_mean_visit_weighted"] == pytest.approx(4 / 204)
        assert out["tv_mean_visit_weighted"] < out["tv_mean"]

    def test_visit_weighted_mean_is_nan_when_nothing_paired(self):
        """Same rule as tv_mean: an absent measurement is not a zero."""
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "9", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        assert math.isnan(
            score_paired(base, probe, actor="lm")["tv_mean_visit_weighted"]
        )


class TestArmsDifferIsStated:
    def test_a_self_pair_is_flagged(self):
        """`score_paired` deliberately allows scoring an arm against
        itself (the exact zero is this module's falsification anchor), so a
        direct caller that skips the P1/P4 wrappers could read that 0.0 as
        "the probe moved nothing"."""
        arm = _arm(_rows("kuhn", "0", {0: 4}))
        assert score_paired(arm, arm, actor="lm")["arms_differ"] is False

    def test_a_real_pair_is_flagged_as_differing(self):
        base = _arm(_rows("kuhn", "0", {0: 4}))
        probe = _arm(
            _rows("kuhn", "0", {1: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        assert score_paired(base, probe, actor="lm")["arms_differ"] is True


class TestNullActionsAreNotCounted:
    def test_a_null_parsed_action_is_excluded_from_the_distribution(self):
        """A null action is not a choice, and counting it would put mass on a
        None key and shift every frequency at that info-set. Unreachable
        (`parsed_action` is in REQUIRED_COLUMNS), cheap insurance
        against a future nullable path."""
        base = pl.concat(
            [
                _arm(_rows("kuhn", "0", {0: 4})),
                _arm([("kuhn", "0", None)]),
            ]
        )
        probe = _arm(
            _rows("kuhn", "0", {0: 4}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")
        assert out["tv_by_infoset"]["kuhn"]["0"] == 0.0
        assert out["visits_by_infoset"]["kuhn"]["0"]["n_base"] == 4


class TestEmpiricalNullDropCounting:
    """_empirical counts null drops so an all-null info-set
    is distinguishable from an unvisited one.
    """

    def test_all_null_info_set_reports_null_count_and_is_not_unvisited(self):
        """An info-set whose rows are present but all-null `parsed_action`
        reports `n_null_dropped == row count` and is not reported as unvisited.
        """
        from lmopt.scoring.property_paired import _empirical

        rows = pl.DataFrame(
            {
                "parsed_action": [None, None, None, None],
            }
        )
        dist, n, n_null = _empirical(rows)
        assert dist == {}
        assert n == 0
        assert n_null == 4  # The all-null count is reported

    def test_genuinely_absent_info_set_still_reports_as_unvisited(self):
        """A genuinely absent info-set (zero rows) is still distinguishable.
        """
        from lmopt.scoring.property_paired import _empirical

        rows = pl.DataFrame({"parsed_action": []}, schema={"parsed_action": pl.Int64})
        dist, n, n_null = _empirical(rows)
        assert dist == {}
        assert n == 0
        assert n_null == 0  # Zero rows, zero nulls

    def test_mixed_info_set_reports_both_counts(self):
        """An info-set with both real and null actions reports both.
        """
        from lmopt.scoring.property_paired import _empirical

        rows = pl.DataFrame(
            {
                "parsed_action": [0, 1, None, 0, None, 1, 1],
            }
        )
        dist, n, n_null = _empirical(rows)
        assert n == 5  # 5 real actions
        assert n_null == 2  # 2 nulls
        assert n + n_null == 7  # Total rows
        assert dist[0] == 2 / 5
        assert dist[1] == 3 / 5


class TestNullDropsReachTheCaller:
    """The counted drop is only worth counting if it reaches a
    reader. `_empirical`'s third element is exercised above in isolation; these
    two drive the real `score_paired` entrypoint, where the count is
    unpacked.
    """

    def test_null_drops_are_summed_per_game_and_pooled_onto_the_per_game_row(self):
        """A null-action row at a paired info-set is reported per game and on the
        family row, on both sides of the pair, and is not in `n_base`/`n_probe`.
        """
        base = pl.concat(
            [
                _arm(_rows("kuhn", "0", {0: 4})),
                _arm([("kuhn", "0", None)]),
            ]
        )
        probe = pl.concat(
            [
                _arm(
                    _rows("kuhn", "0", {0: 4}),
                    prompt_variant=RELABEL_VARIANT_NAME,
                    probe_type="relabel",
                ),
                _arm(
                    [("kuhn", "0", None), ("kuhn", "0", None)],
                    prompt_variant=RELABEL_VARIANT_NAME,
                    probe_type="relabel",
                ),
            ]
        )
        out = score_paired(base, probe, actor="lm")

        assert out["n_null_dropped_by_game"]["kuhn"] == {"n_base": 1, "n_probe": 2}
        # The point of the pair: the visible sample sizes exclude exactly those
        # rows, so the two numbers together say "4 of 5" and "4 of 6".
        assert out["visits_by_infoset"]["kuhn"]["0"] == {"n_base": 4, "n_probe": 4}

        kuhn_row = next(r for r in out["per_game"] if r["game"] == "kuhn")
        assert kuhn_row["n_null_dropped_base"] == 1
        assert kuhn_row["n_null_dropped_probe"] == 2

    def test_a_family_with_no_frame_reports_none_not_zero(self):
        """A game the arm never played reports `None`, keeping "no rows here"
        distinct from "rows here and nothing dropped" -- the same rule the
        neighbouring `n_visits_*` keys follow.
        """
        base = _arm(_rows("kuhn", "0", {0: 2}))
        probe = _arm(
            _rows("kuhn", "0", {0: 2}),
            prompt_variant=RELABEL_VARIANT_NAME,
            probe_type="relabel",
        )
        out = score_paired(base, probe, actor="lm")

        leduc = next(r for r in out["per_game"] if r["game"] == "leduc")
        assert leduc["n_visits_base"] is None, "guard the premise of this test"
        assert leduc["n_null_dropped_base"] is None
        assert leduc["n_null_dropped_probe"] is None
        kuhn_row = next(r for r in out["per_game"] if r["game"] == "kuhn")
        assert kuhn_row["n_null_dropped_base"] == 0
        assert kuhn_row["n_null_dropped_probe"] == 0


class TestGamesOutsideThePairAreNotSilent:
    def test_a_game_only_the_probe_arm_played_is_recorded(self):
        """
        a game present in the probe arm and absent from the base arm is named in
        ``probe_only_games`` rather than dropped with no count; the paired game is unchanged
        """
        base = _arm(_rows("kuhn", "0", {0: 10}))
        probe_rows = _rows("kuhn", "0", {0: 5, 1: 5}) + _rows("leduc", "x", {0: 3})
        probe = _arm(probe_rows, prompt_variant=RELABEL_VARIANT_NAME, probe_type="relabel")
        out = score_paired(base, probe, actor="lm")
        assert out["probe_only_games"] == ["leduc"]
        assert out["tv_by_infoset"]["kuhn"]["0"] == pytest.approx(0.5)
