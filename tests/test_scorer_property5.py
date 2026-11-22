"""Tests for property5 scorer (Calibration - ECE)."""

from __future__ import annotations

import json
import math

import polars as pl
import pytest

from lmopt.scoring.property5 import (
    calibration_error,
    score_property5,
)


def _row(game: str, stated: dict, true: dict) -> dict:
    return {
        "game": game,
        "elicited_belief": json.dumps({str(k): v for k, v in stated.items()}),
        "true_posterior": json.dumps({str(k): v for k, v in true.items()}),
    }


class TestProperty5:
    def test_perfect_calibration_ece_zero(self):
        """Stated = true -> ECE = 0."""
        # 100 rows with stated = true = 50/50 split over 2 cards.
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(100)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert result["ece"]["kuhn"] == pytest.approx(0.0, abs=1e-9)

    def test_maximally_wrong_ece_high(self):
        """Stated = 1.0 for a card, true = 0.0 -> high ECE."""
        rows = [
            _row("kuhn", {"0": 1.0, "1": 0.0}, {"0": 0.0, "1": 1.0}) for _ in range(100)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        ece = result["ece"]["kuhn"]
        assert ece > 0.3  # Should be near 0.5

    def test_no_null_rows_skipped(self):
        """Rows with null elicited_belief or true_posterior are excluded."""
        rows = [
            {
                "game": "kuhn",
                "elicited_belief": None,
                "true_posterior": json.dumps({"0": 1.0}),
            },
            {
                "game": "kuhn",
                "elicited_belief": json.dumps({"0": 1.0}),
                "true_posterior": None,
            },
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}),
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert not math.isnan(result["ece"]["kuhn"])
        assert result["n_pairs"]["kuhn"] == 2  # Only 1 valid row × 2 cards

    def test_multiple_games(self):
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(20)
        ] + [
            _row("leduc", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(20)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert "kuhn" in result["ece"]
        assert "leduc" in result["ece"]
        assert not math.isnan(result["global_ece"])
        # "global_per_pair_error" mirrors "global_ece" (property5.py's
        # return dict).
        assert not math.isnan(result["global_per_pair_error"])

    def test_global_per_pair_error_is_the_mean_of_the_per_game_values(self):
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(5)
        ] + [
            _row(
                "leduc",
                {"0": 1.0, "1": 0.0, "2": 0.0},
                {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 1.0},
            )
            for _ in range(5)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert result["per_pair_error"]["kuhn"] == pytest.approx(0.0, abs=1e-9)
        assert result["per_pair_error"]["leduc"] == pytest.approx(2 / 3, rel=1e-9)
        assert result["global_per_pair_error"] == pytest.approx(1 / 3, rel=1e-9)

    def test_empty_df_returns_nan(self):
        df = pl.DataFrame(
            {"game": [], "elicited_belief": [], "true_posterior": []},
            schema={
                "game": pl.Utf8,
                "elicited_belief": pl.Utf8,
                "true_posterior": pl.Utf8,
            },
        )
        result = score_property5(df)
        assert math.isnan(result["global_ece"])
        assert math.isnan(result["global_per_pair_error"])
        # The empty-df early return must populate the two baseline/margin
        # global keys with NaN too, not KeyError -- a caller reading
        # "global_per_pair_margin" off an empty-arm result must get the
        # same "no data" signal as its siblings, not a crash.
        assert math.isnan(result["global_uniform_baseline_per_pair"])
        assert math.isnan(result["global_per_pair_margin"])
        assert result["uniform_baseline_per_pair"] == {}
        assert result["per_pair_margin"] == {}

    def test_leduc_marginalises_true_posterior_before_comparing(self):
        """Leduc's true_posterior is over 6 card indices; the stated belief
        is a 3-key rank simplex. Iterating stated keys against the raw
        6-key posterior (`true.get(card_key, 0.0)`) silently discards the
        mass at indices 3-5 (55.48% of the mass on the landed arm, see
        tests/test_leduc_rank.py).

        Here: stated puts all its mass on rank "0" (Jack); the posterior
        splits 0.5/0.5 across the two Jack indices ("0","1") and 0.0
        elsewhere. Without marginalisation, `true.get("0")` == 0.5 !=
        stated's 1.0 -> ECE > 0. Rank-marginalised, true rank "0" == 1.0 ==
        stated -> ECE == 0.
        """
        rows = [
            _row(
                "leduc",
                {"0": 1.0, "1": 0.0, "2": 0.0},
                {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0},
            )
            for _ in range(20)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert result["ece"]["leduc"] == pytest.approx(0.0, abs=1e-9)

    def test_kuhn_true_posterior_is_not_marginalised(self):
        """Kuhn's posterior is already 1:1 with its stated keys - the
        leduc-only marginalisation gate must not fire for it."""
        rows = [
            _row("kuhn", {"0": 1.0, "1": 0.0}, {"0": 0.0, "1": 1.0}) for _ in range(20)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert result["ece"]["kuhn"] > 0.3

    def test_leduc_ece_on_landed_arm_fixture_matches_the_corrected_value(self):
        """Pins the P5 ECE and the per-pair metric against the committed
        landed-arm fixture. Derived directly against the real parquet
        shards, by calling the real `score_property5`:

            uv run python - <<'EOF'
            ... score_property5(df) on data/run_experiment/qwen2.5-14b-instruct ...
            EOF
            -> ece['leduc'] = 0.15803516240979426
            -> per_pair_error['leduc'] = 0.23610688755435663

        The committed 142-pair fixture (tests/fixtures/
        leduc_belief_posterior_pairs.csv) reproduces both values bit-exactly
        - verified by loading the fixture and calling the real, unmodified
        `score_property5`. Mutation this pins: removing
        `_mean_abs`/the `per_pair_error` computation from the per-game loop
        would KeyError on `result["per_pair_error"]`
        rather than merely returning a different number, so this assertion
        is live against that regression too, not just the ece one.
        """
        from tests.test_leduc_rank import _load_landed_leduc_fixture

        df = _load_landed_leduc_fixture()
        assert df.height == 53145
        result = score_property5(df)
        assert result["ece"]["leduc"] == pytest.approx(0.15803516240979426, rel=1e-9)
        assert result["per_pair_error"]["leduc"] == pytest.approx(
            0.23610688755435663, rel=1e-9
        )
        # Pins the "beat-uninformed" baseline/margin against the same real
        # fixture, measured by calling the real `score_property5` directly on
        # this fixture (agrees to every printed digit with an independent
        # whole-arm run: 0.172784 / 0.063323).
        #
        # The trap this guards: leduc's true_posterior is over 6 raw card
        # indices; a baseline computed by iterating `stated`'s 3 rank keys
        # against the raw (un-marginalised) posterior would silently read
        # 0.0 mass at index "2" (rank 2's real mass lives at raw index "5")
        # instead of `marginalise_to_rank`'s correct 1.0 -- giving
        # baseline=1/3=0.333333, margin=1/3=0.333333, neither of which
        # matches the values pinned here.
        assert result["uniform_baseline_per_pair"]["leduc"] == pytest.approx(
            0.17278411126300552, rel=1e-9
        )
        assert result["per_pair_margin"]["leduc"] == pytest.approx(
            0.06332277629135111, rel=1e-9
        )

    def test_bin_data_present(self):
        rows = [
            _row(
                "kuhn",
                {"0": float(i) / 10, "1": 1 - float(i) / 10},
                {"0": float(i) / 10, "1": 1 - float(i) / 10},
            )
            for i in range(10)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert "bins" in result
        assert isinstance(result["bins"]["kuhn"], list)


class TestCalibrationErrorUnit:
    """These pin the unit-level `calibration_error` function, not the arm -
    see TestPerPairCalibrationErrorProductionPath below for the arm-level
    pin, which shows Kuhn's arm-level ECE of 0.0 is a binning-cancellation
    artefact."""

    def test_uniform_stated_belief_does_not_report_perfect_calibration(self):
        """A flat stated belief against a peaked posterior must not score 0.0."""
        stated = {str(i): 1 / 6 for i in range(6)}
        true = {"0": 0.42, "1": 0.25, "2": 0.12, "3": 0.12, "4": 0.09, "5": 0.0}
        score = calibration_error([(stated, true)])
        assert score > 0.1, f"binning cancelled the error: got {score}"

    def test_kuhn_exact_posterior_still_scores_zero(self):
        """Guards the real result: kuhn states the exact posterior on 83.2% of rows."""
        d = {"0": 0.5, "1": 0.5}
        assert calibration_error([(d, d)]) == pytest.approx(0.0)


class TestPerPairCalibrationErrorProductionPath:
    """Reaches the per-pair metric through `score_property5`, the entrypoint
    `scripts/analyse_stats.py` actually
    call - `calibration_error` alone (above) never runs through that path.
    """

    def test_kuhn_arm_level_ece_stays_zero_but_per_pair_error_reveals_binning_cancellation(
        self,
    ):
        """Kuhn's P5 ECE of 0.0 is a binning-cancellation artefact, just not
        via the *uniform-stated* route - it is the same bin-then-average-
        then-abs ordering (property5.py's bin loop) cancelling equal and
        opposite signed errors instead.

        Measured directly against the real landed arm
        (data/run_experiment/qwen2.5-14b-instruct/kuhn_*.parquet, 20
        shards, 24,045 calibration rows):

            uv run python - <<'EOF'
            import polars as pl, glob
            from collections import Counter
            files = glob.glob("data/run_experiment/qwen2.5-14b-instruct/kuhn_*.parquet")
            df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
            rows = df.filter(pl.col("elicited_belief").is_not_null()
                              & pl.col("true_posterior").is_not_null())
            pairs = Counter((r["elicited_belief"], r["true_posterior"])
                             for r in rows.iter_rows(named=True))
            for (eb, tp), c in pairs.items(): print(c, eb, tp)
            EOF
            -> exactly 4 distinct (elicited_belief, true_posterior) pairs,
               counts 6436 / 6908 / 6656 / 4045, summing to 24045 (the real
               arm's full calibration-row count for kuhn)

        Reproduced below as those 4 distinct rows at their real counts -
        not a synthetic shape - so `score_property5` sees exactly the
        distribution the landed arm has.
        """
        exact_pairs = [
            (6436, {"0": 0.5, "1": 0.0, "2": 0.5}, {"0": 0.5, "1": 0.0, "2": 0.5}),
            (6908, {"0": 0.5, "1": 0.5, "2": 0.0}, {"0": 0.5, "1": 0.5, "2": 0.0}),
            (6656, {"0": 0.0, "1": 0.5, "2": 0.5}, {"0": 0.0, "1": 0.5, "2": 0.5}),
            (
                4045,
                {"0": 0.0, "1": 0.5, "2": 0.5},
                {"0": 0.0, "1": 9.989940130288799e-06, "2": 0.9999900100598696},
            ),
        ]
        rows = []
        for count, stated, true in exact_pairs:
            rows.extend([_row("kuhn", stated, true)] * count)
        df = pl.DataFrame(rows)
        assert df.height == 24045

        result = score_property5(df)

        # Bin-based ECE stays 0.0 for the bin view; the per-pair metric
        # below reveals the cancellation.
        assert result["ece"]["kuhn"] == pytest.approx(0.0, abs=1e-9)

        # Calling the real score_property5 on the 4-pair reconstruction above
        # (and independently, on the concatenated real parquet shards) both
        # give 0.05607429377395641.
        assert result["per_pair_error"]["kuhn"] == pytest.approx(
            0.05607429377395641, rel=1e-9
        )

    def test_per_pair_error_key_present_for_every_scored_game(self):
        """`score_property5`'s per-game loop must always populate
        `per_pair_error` alongside `ece`, not just for kuhn.

        With stated == true for both games, a stub that hardcoded
        `per_pair_error[game] = 0.0` for every game (or that copied `ece`'s
        value verbatim) would pass identically -- a branch-independent
        expectation. Leduc here has stated != true, so
        its expected `per_pair_error` is a specific nonzero float that
        neither a 0.0 stub nor a copy-of-`ece` stub produces (leduc's `ece`
        is 0.0 too on this fixture, by construction -- see the assertion
        below -- so copying `ece` would give 0.0, still wrong).
        """
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(5)
        ] + [
            _row(
                "leduc",
                {"0": 1.0, "1": 0.0, "2": 0.0},
                {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 1.0},
            )
            for _ in range(5)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert set(result["per_pair_error"]) == set(result["ece"])
        assert result["per_pair_error"]["kuhn"] == pytest.approx(0.0, abs=1e-9)
        # leduc: stated all-mass-on-rank-0 vs true all-mass-on-index-5 (rank
        # 2 after marginalisation) -- rank 0 off by 1.0, rank 2 off by 1.0,
        # rank 1 off by 0.0, mean = 2/3.
        assert result["per_pair_error"]["leduc"] == pytest.approx(2 / 3, rel=1e-9)
        assert not math.isnan(result["ece"]["leduc"])

        # Extends this test's invariant to the baseline/margin per-game
        # dicts: darkhex must drop out of all four dicts together, so a
        # future edit cannot let one dict keep a game while another drops it.
        # Mutation this catches beyond the assertion above: populating
        # "uniform_baseline_per_pair"/"per_pair_margin" for a game not in
        # "ece" (or vice versa) -- e.g. reverting the darkhex `continue` to
        # sit after only some of the four dicts are written.
        assert set(result["uniform_baseline_per_pair"]) == set(result["ece"])
        assert set(result["per_pair_margin"]) == set(result["ece"])

        # kuhn: stated == true == uniform 0.5/0.5, so the uniform-stater
        # baseline is also 0.5/0.5 == stated exactly here -- baseline error
        # 0.0, margin 0.0 - 0.0 = 0.0 (a genuine, non-hardcoded computation
        # that happens to land on the A4 tie boundary; the real-data A4
        # test below is the one that pins the boundary's decision-rule
        # BEHAVIOUR, not this value-correctness check).
        assert result["uniform_baseline_per_pair"]["kuhn"] == pytest.approx(
            0.0, abs=1e-9
        )
        assert result["per_pair_margin"]["kuhn"] == pytest.approx(0.0, abs=1e-9)
        # leduc: baseline pairs are (1/3, true) for each of the 3 rank
        # keys, true (marginalised) = {0: 0.0, 1: 0.0, 2: 1.0} -- errors
        # 1/3, 1/3, 2/3, mean 4/9. margin = 2/3 - 4/9 = 2/9. A stub that
        # used raw (un-marginalised) posterior keys instead would compute
        # baseline against true={"0":0,"1":0,"2":0,"3":0,"4":0,"5":1.0}. get
        # ("2",0.0)=0.0 for all three rank keys (index "2" ISN'T where rank
        # 2's mass lives post-marginalisation, index "5" is) giving
        # baseline=1/3, margin=1/3 -- both distinctly wrong.
        assert result["uniform_baseline_per_pair"]["leduc"] == pytest.approx(
            4 / 9, rel=1e-9
        )
        assert result["per_pair_margin"]["leduc"] == pytest.approx(2 / 9, rel=1e-9)

    def test_uniform_stated_belief_through_score_property5_diverges_from_bin_ece(self):
        """Drives the headline cancellation route (uniform stated
        belief vs peaked posterior) through the production entrypoint
        `score_property5`, not just `calibration_error` directly
        (`TestCalibrationErrorUnit` above only ever calls the unit
        function).

        Liarsdice illustration: stated flat 1/6, true
        `{0.42, 0.25, 0.12, 0.12, 0.09, 0}`.
        """
        stated = {str(i): 1 / 6 for i in range(6)}
        true = {"0": 0.42, "1": 0.25, "2": 0.12, "3": 0.12, "4": 0.09, "5": 0.0}
        rows = [_row("liarsdice", stated, true) for _ in range(50)]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert result["ece"]["liarsdice"] == pytest.approx(0.0, abs=1e-9)
        assert result["per_pair_error"]["liarsdice"] > 0.1

    def test_all_rows_malformed_json_yields_nan_not_zero(self):
        """Every row's `elicited_belief`/`true_posterior` is non-null (so
        it passes the not-null filter) but is not valid JSON, so every row
        hits the `except (JSONDecodeError, TypeError): continue` clause and
        `pairs` stays empty for this game.
        """
        rows = [
            {
                "game": "kuhn",
                "elicited_belief": "{not valid json",
                "true_posterior": "{also not valid json",
            }
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert math.isnan(result["ece"]["kuhn"])
        assert math.isnan(result["per_pair_error"]["kuhn"])
        assert result["n_pairs"]["kuhn"] == 0
        # Same "if not pairs" branch populates the baseline/margin keys too -- must be NaN, not KeyError, not 0.0.
        assert math.isnan(result["uniform_baseline_per_pair"]["kuhn"])
        assert math.isnan(result["per_pair_margin"]["kuhn"])

    def test_empty_stated_dict_row_contributes_to_neither_pairs_nor_baseline(self):
        """The documented "empty `stated`" behaviour: a
        row whose `elicited_belief` parses to `{}` makes `for card_key in
        stated` a no-op, so it contributes zero entries to both `pairs` and
        `baseline_pairs` -- the two lists stay index-aligned in length by
        construction (they can never diverge, since both are appended to
        inside the same `for card_key in stated` loop body). Confirmed
        here by mixing one such row into an otherwise-normal fixture: the
        empty-belief row must be invisible to every metric, not counted as
        a "row with no cards" that skews `n_pairs` or introduces a spurious
        baseline entry.
        """
        rows = [
            _row("kuhn", {}, {"0": 0.5, "1": 0.5}),
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}),
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        # Only the second (non-empty-stated) row contributes: 2 pairs.
        assert result["n_pairs"]["kuhn"] == 2
        assert result["per_pair_error"]["kuhn"] == pytest.approx(0.0, abs=1e-9)
        assert result["uniform_baseline_per_pair"]["kuhn"] == pytest.approx(
            0.0, abs=1e-9
        )
        assert result["per_pair_margin"]["kuhn"] == pytest.approx(0.0, abs=1e-9)

    def test_goofspiel_true_key_absent_from_stated_is_named_not_applicable(self):
        """A violating row does not abort score_property5 entirely -- it
        confines the failure to this game, loudly, via not_applicable,
        mirroring property7.py's own confined degrade exactly (both discard
        the whole game, never a partial score built from surviving clean
        rows).
        """
        rows = [_row("goofspiel", {"0": 0.5, "1": 0.5}, {"0": 0.4, "1": 0.4, "2": 0.2})]
        df = pl.DataFrame(rows)
        result = score_property5(df)  # must not raise
        assert "goofspiel" not in result["ece"]
        assert "goofspiel" not in result["per_pair_error"]
        assert "goofspiel" in result["not_applicable"]
        assert "2" in result["not_applicable"]["goofspiel"]

    def test_goofspiel_violation_does_not_abort_other_games_in_the_same_call(self):
        """The blast-radius regression the confinement exists to prevent:
        a goofspiel violation must not cost kuhn's already-computed ECE.
        """
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(5)
        ] + [_row("goofspiel", {"0": 0.5, "1": 0.5}, {"0": 0.4, "1": 0.4, "2": 0.2})]
        df = pl.DataFrame(rows)
        result = score_property5(df)  # must not raise
        assert result["ece"]["kuhn"] == pytest.approx(0.0, abs=1e-9)
        assert "goofspiel" not in result["ece"]
        assert "goofspiel" in result["not_applicable"]

    def test_goofspiel_guard_is_the_one_shared_helper_not_a_third_copy(self):
        """This goofspiel key-subset check lives once, in
        `_belief_alignment.assert_true_support_within_stated`, and both
        `property7.py` and `property5.py` call it. "Mirrors the guard"
        is not enough of a guarantee on its own -- a future edit could
        re-duplicate the check inline in one module while leaving the
        shared helper in place, unused, in the other. This test asserts
        object identity: `property5.assert_true_support_within_stated` must be the same
        function object as the one defined in `_belief_alignment`, not an
        independently-defined function that merely behaves alike today.
        """
        import lmopt.scoring.property5 as p5
        from lmopt.scoring._belief_alignment import assert_true_support_within_stated

        assert p5.assert_true_support_within_stated is assert_true_support_within_stated


class TestDarkHexScoredWithPerCellEstimand:
    """Per-cell occupancy is not a probability simplex, so an
    ECE-of-a-distribution is undefined for the dark hex family. It is scored
    with a per-cell estimand instead of a simplex one: the observed quantity is the
    same pooled mean |stated - true| every family reports as
    `per_pair_error`, and the baseline it is measured against is a stater who
    says 0.5 at every cell rather than `1 / len(stated)`.

    These tests pin the family's presence in each per-game dict, not the estimand's own arithmetic.
    """

    def test_darkhex_enters_every_per_game_dict_with_no_recorded_reason(self):
        rows = [
            _row("darkhex", {"a1": 0.5, "b1": 0.5}, {"a1": 0.5, "b1": 0.5})
            for _ in range(10)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert "darkhex" in result["ece"]
        assert "darkhex" in result["per_pair_error"]
        assert result["n_pairs"]["darkhex"] == 20
        # The two baseline dicts are populated inside
        # the same per-game loop, so they must carry the family too.
        assert "darkhex" in result["uniform_baseline_per_pair"]
        assert "darkhex" in result["per_pair_margin"]
        assert "darkhex" not in result["not_applicable"]

    def test_darkhex_depth_variant_also_scored(self):
        """`darkhex_3x3` etc. must resolve through GAME_FAMILIES, not a raw
        `== "darkhex"` string compare (families.py's own provenance note:
        a depth variant would silently fall through that check). Synthetic
        `game` column value only; no game is loaded and nothing is solved.
        """
        rows = [
            _row("darkhex_3x3", {"a1": 0.25, "b1": 0.25}, {"a1": 0.5, "b1": 0.5})
            for _ in range(10)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert "darkhex_3x3" in result["ece"]
        assert "darkhex_3x3" not in result["not_applicable"]
        # 0.5-per-cell against a 0.5 posterior is exactly 0; the simplex
        # baseline would be 1/2 per key here and also 0, so the discriminating
        # quantity is the observed error against that baseline.
        assert result["uniform_baseline_per_pair"]["darkhex_3x3"] == pytest.approx(0.0)
        assert result["per_pair_margin"]["darkhex_3x3"] == pytest.approx(0.25)

    def test_darkhex_enters_ece_and_per_pair_error_together(self):
        """The pinned `set(per_pair_error) == set(ece)` invariant
        (`test_per_pair_error_key_present_for_every_scored_game`) must hold
        with darkhex present alongside a normally-scored game: the family
        enters both dicts, not one.
        """
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(5)
        ] + [
            _row("darkhex", {"a1": 0.5, "b1": 0.5}, {"a1": 0.5, "b1": 0.5})
            for _ in range(5)
        ]
        df = pl.DataFrame(rows)
        result = score_property5(df)
        assert set(result["per_pair_error"]) == {"kuhn", "darkhex"}
        assert set(result["ece"]) == {"kuhn", "darkhex"}
        assert set(result["uniform_baseline_per_pair"]) == {"kuhn", "darkhex"}
        assert set(result["per_pair_margin"]) == {"kuhn", "darkhex"}

    def test_empty_df_reports_no_reason_and_no_score(self):
        """A darkhex-only frame whose beliefs are all null takes the
        empty-rows early return: no score, and no not-applicable reason
        either, because the family is not the thing that is missing.
        """
        df = pl.DataFrame(
            {"game": ["darkhex"], "elicited_belief": [None], "true_posterior": [None]},
        )
        result = score_property5(df)
        assert result["ece"] == {}
        assert result["not_applicable"] == {}


class TestActorParameterisation:
    """P5's explicit actor filter.

    P5's legacy population filter is `elicited_belief.is_not_null() &
    true_posterior.is_not_null()`, which excludes non-LM
    rows only incidentally: `match_runner._get_lm_metadata` reads
    `elicited_belief` off the agent with a `getattr(..., None)` default, so
    every non-LM agent happens to log null. It is not a seat or actor filter,
    it holds only while exactly one agent in a matchup elicits beliefs, and no
    column is consulted that could separate two belief-carrying agents.
    These tests state the exclusion.
    """

    @staticmethod
    def _belief_row(actor, stated, true, prompt=None):
        row = _row("kuhn", stated, true)
        row["actor"] = actor
        row["prompt"] = prompt
        return row

    def test_a_foreign_agents_belief_row_is_excluded_when_an_actor_is_named(self):
        """A NON-LM row carrying a non-null `elicited_belief` and a non-null
        `true_posterior`.
        """
        lm = [
            self._belief_row(
                "model-x", {"0": 0.8, "1": 0.2}, {"0": 1.0, "1": 0.0}, prompt="p"
            )
            for _ in range(2)
        ]
        foreign = [
            self._belief_row("bayesian", {"0": 0.0, "1": 1.0}, {"0": 1.0, "1": 0.0})
        ]
        df = pl.DataFrame(lm + foreign)

        out = score_property5(df, actor="model-x")
        assert out["actor_filter"] == "actor=model-x"
        assert out["n_pairs"]["kuhn"] == 4
        assert out["per_pair_error"]["kuhn"] == pytest.approx(0.2, abs=1e-12)

        pooled = score_property5(df)
        assert pooled["n_pairs"]["kuhn"] == 6
        assert pooled["per_pair_error"]["kuhn"] == pytest.approx(
            (4 * 0.2 + 2 * 1.0) / 6, abs=1e-12
        )

    def test_legacy_frame_pools_as_before_and_says_so(self):
        """A legacy arm has no `actor` column, so the pooled two-column
        filter is all there is; it must still produce the pooled number and
        label itself (no silent fallbacks).
        """
        rows = [
            _row("kuhn", {"0": 0.8, "1": 0.2}, {"0": 1.0, "1": 0.0}) for _ in range(2)
        ]
        df = pl.DataFrame(rows)
        assert "actor" not in df.columns
        out = score_property5(df)
        assert out["actor_filter"] == "legacy-belief-columns"
        assert out["per_pair_error"]["kuhn"] == pytest.approx(0.2, abs=1e-12)

    def test_naming_an_actor_a_legacy_frame_cannot_identify_raises(self):
        """Same rule as property3's: a named actor that the frame
        cannot attribute must raise, not silently return the pooled
        population under that agent's name.
        """
        df = pl.DataFrame([_row("kuhn", {"0": 1.0}, {"0": 1.0})])
        with pytest.raises(ValueError, match="actor"):
            score_property5(df, actor="model-x")


class TestForeignProbeType:
    """Without a probe_type guard in the plain (non-paired) scorers, a frame
    assembled from a directory that also held a probe arm's rows (reachable
    via an unfiltered `--data` glob) would be silently
    pooled into the calibration estimate.
    """

    def test_refuses_a_frame_holding_a_foreign_probe_type(self):
        base = pl.DataFrame([_row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5})])
        mixed = pl.concat(
            [
                base.with_columns(pl.lit("none").alias("probe_type")),
                base.with_columns(pl.lit("offpath_drive").alias("probe_type")),
            ]
        )
        with pytest.raises(ValueError, match="offpath_drive"):
            score_property5(mixed)

    def test_accepts_an_all_base_frame(self):
        df = pl.DataFrame(
            [_row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5})]
        ).with_columns(pl.lit("none").alias("probe_type"))
        score_property5(df)  # must not raise


# ---------------------------------------------------------------------------
# P5 k-sample integration
# ---------------------------------------------------------------------------




@pytest.mark.parametrize("stated", [{"0": -0.5, "1": 1.5}, {"0": 1.5, "1": -0.5}])
def test_a_stated_value_outside_zero_one_is_refused_before_binning(stated):
    """
    a stated probability outside [0, 1] is refused, rather than a negative value wrapping to a
    bin counted from the end through Python's negative indexing
    """
    df = pl.DataFrame([_row("kuhn", stated, {"0": 0.5, "1": 0.5})])
    with pytest.raises(ValueError, match="outside"):
        score_property5(df)
