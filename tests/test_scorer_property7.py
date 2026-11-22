"""Tests for property7 scorer (Bayesian Consistency - TVD + action-vs-belief)."""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest

pyspiel = pytest.importorskip("pyspiel")

from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_goofspiel, load_kuhn, load_leduc
from lmopt.scoring import property7
from lmopt.scoring.goofspiel_posterior import goofspiel_all_posteriors
from lmopt.scoring.property7 import (
    _build_infoset_map,
    _leduc_dealt_cards,
    _parse_belief_array,
    score_property7,
)


def _row(
    game: str,
    stated: dict,
    true: dict,
    prompt: str | None = None,
    player_to_act: int = 0,
    info_set: str = "0",
    parsed_action: int = 0,
) -> dict:
    return {
        "game": game,
        "elicited_belief": json.dumps({str(k): v for k, v in stated.items()}),
        "true_posterior": json.dumps({str(k): v for k, v in true.items()}),
        "prompt": prompt,
        "player_to_act": player_to_act,
        "info_set": info_set,
        "parsed_action": parsed_action,
    }


class TestProperty7a:
    def test_tvd_zero_when_identical(self):
        """Stated = true -> TVD = 0."""
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(20)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        assert result["tvd_mean"]["kuhn"] == pytest.approx(0.0, abs=1e-9)

    def test_tvd_max_when_opposite(self):
        """Stated puts all mass on card 0, true puts all on card 1 -> TVD = 1.0."""
        rows = [
            _row("kuhn", {"0": 1.0, "1": 0.0}, {"0": 0.0, "1": 1.0}) for _ in range(20)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        tvd = result["tvd_mean"]["kuhn"]
        assert tvd == pytest.approx(1.0, abs=1e-9)

    def test_tvd_symmetric_example(self):
        """50/50 vs 0/100 -> TVD = 0.5."""
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.0, "1": 1.0}) for _ in range(20)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        tvd = result["tvd_mean"]["kuhn"]
        assert tvd == pytest.approx(0.5, abs=1e-9)

    def test_rows_without_both_columns_skipped(self):
        rows = [
            {
                "game": "kuhn",
                "elicited_belief": None,
                "true_posterior": json.dumps({"0": 1.0}),
                "prompt": None,
                "player_to_act": 0,
                "info_set": "0",
                "parsed_action": 0,
            },
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}),
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        # Only 1 valid row; TVD = 0.
        assert result["tvd_mean"]["kuhn"] == pytest.approx(0.0, abs=1e-9)

    def test_multiple_games(self):
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(10)
        ] + [
            _row("leduc", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(10)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        assert "kuhn" in result["tvd_mean"]
        assert "leduc" in result["tvd_mean"]
        assert not math.isnan(result["tvd_global_mean"])


class TestLeducRankMarginalisation:
    """Leduc's true_posterior is over 6 card indices; the stated belief is a
    3-key rank simplex. Comparing them raw unions the key sets and inflates
    TVD - see leduc_rank.py and tests/test_leduc_rank.py for the 0.5858-vs-0.3542
    regression on a real arm. These are the small, hand-checkable unit cases.
    """

    def test_tvd_zero_when_stated_matches_rank_marginalised_posterior(self):
        # true_posterior all on index 0 and 1 (rank 0, Jack) == stated all
        # on rank "0". Without marginalisation, the key-union would compare
        # stated={"0":1.0} against true={"0":0.5,"1":0.5} over keys {"0","1"}
        # and see a (nonexistent) disagreement at key "1".
        rows = [
            _row(
                "leduc",
                {"0": 1.0, "1": 0.0, "2": 0.0},
                {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0},
            )
            for _ in range(10)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        assert result["tvd_mean"]["leduc"] == pytest.approx(0.0, abs=1e-9)

    def test_tvd_nonzero_survives_marginalisation(self):
        # stated puts all mass on rank 2 (King); true_posterior puts all
        # mass on rank 0 (Jack, indices 0-1) -> TVD = 1.0 either way.
        rows = [
            _row(
                "leduc",
                {"0": 0.0, "1": 0.0, "2": 1.0},
                {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0},
            )
            for _ in range(10)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        assert result["tvd_mean"]["leduc"] == pytest.approx(1.0, abs=1e-9)

    def test_non_leduc_family_is_not_marginalised(self):
        """Kuhn's posterior is already 1:1 with its stated keys (3 distinct
        cards, no rank duplication) - marginalising it would be wrong, so
        the family gate must not fire for it."""
        rows = [
            _row("kuhn", {"0": 1.0, "1": 0.0}, {"0": 0.0, "1": 1.0}) for _ in range(10)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        assert result["tvd_mean"]["kuhn"] == pytest.approx(1.0, abs=1e-9)


class TestGoofspielZeroPadding:
    """Goofspiel's true_posterior support omits deck cards P1 cannot
    hold in any world consistent with the P0 info-state - legitimate
    deduction, not an instrument gap (see tests/test_goofspiel_posterior.py::
    test_missing_posterior_keys_are_deducibly_unplayable_by_p1, which proves
    this on a real CFR+-solved game). A raw key-union zero-pads the
    goofspiel posterior correctly only by coincidence, while being wrong for
    leduc (rank-vs-index mismatch, handled by ``marginalise_to_rank``). This
    class asserts the goofspiel alignment is an explicit step that does not
    depend on the union-of-keys behaviour below.
    """

    def test_narrow_posterior_zero_padded_explicitly(self):
        """CHARACTERISATION only - no mutation of the finished code makes
        this fail. Dense stated belief over the full 4-card deck
        (goofspiel's elicited belief is always dense -
        prompting/goofspiel.py:132-134); true_posterior has a narrower
        support (cards 2, 3 deducibly zero - mirrors a real round>=2 P0 node
        per test_goofspiel_posterior.py). Hand-verified as a worked example,
        but it passes identically if the goofspiel
        padding `elif` branch in `score_property7` is deleted entirely: with
        the default union `_combine_keys` and `true`'s keys always a subset
        of `stated`'s (guaranteed by the invariant check below), union
        with an unpadded narrower `true` produces the exact same defaulted
        array as union with the pre-padded version, because
        `.get(k, 0.0)` behaves identically either way. The actual regression
        guard for this change is
        `test_zero_padding_survives_intersection_instead_of_union` below -
        only a change to `_combine_keys` itself (or a branch deletion, as
        demonstrated by
        `test_padding_reachable_via_real_infoset_and_survives_deletion`) can
        expose the difference.
        """
        stated = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
        true = {"0": 0.5, "1": 0.5}
        rows = [_row("goofspiel", stated, true) for _ in range(5)]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        # By hand, true zero-padded to {0,1,2,3}: {0:0.5, 1:0.5, 2:0, 3:0}.
        expected = 0.5 * (0.25 + 0.25 + 0.25 + 0.25)
        assert result["tvd_mean"]["goofspiel"] == pytest.approx(expected, abs=1e-9)

    def test_zero_padding_survives_intersection_instead_of_union(self, monkeypatch):
        """The goofspiel alignment must survive a change to property7.py's
        union-of-keys, because it does not depend on it. Demonstrated concretely: monkeypatch the
        module's key-combination step from union to intersection and assert
        the goofspiel TVD is unchanged - it can only be unchanged if the
        explicit zero-padding already made true's key set a subset of
        stated's before this step runs.
        """
        import lmopt.scoring.property7 as p7mod

        stated = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
        true = {"0": 0.5, "1": 0.5}
        rows = [_row("goofspiel", stated, true) for _ in range(5)]
        df = pl.DataFrame(rows)

        result_union = score_property7(df)

        monkeypatch.setattr(
            p7mod,
            "_combine_keys",
            lambda s, t: sorted(set(s.keys()) & set(t.keys())),
        )
        result_intersection = score_property7(df)

        assert result_intersection["tvd_mean"]["goofspiel"] == pytest.approx(
            result_union["tvd_mean"]["goofspiel"], abs=1e-9
        )
        # Sanity: the two results are not trivially both-nan / both-empty.
        assert not math.isnan(result_union["tvd_mean"]["goofspiel"])

    def test_real_games_and_cfr_policies_args_do_not_crash_goofspiel_path(self):
        """CRASH smoke-CHECK only. `_row`'s default `info_set="0"`
        is not a real goofspiel P0 info-state string, so
        `_score_7b_goofspiel`'s `worlds.get("0", [])` returns `[]` and 7b
        short-circuits without touching any state-dependent code - passing
        real `games`/`cfr_policies` here proves only that `score_property7`
        does not raise when they are supplied, not that anything reachable
        through them was exercised. See
        `test_padding_reachable_via_real_infoset_and_survives_deletion`
        below for the genuine production-call-shape regression test.
        """
        game = load_goofspiel(num_cards=4)
        cfr_pol = solve_cfr(game, 10)  # small: only need a valid policy object

        stated = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
        true = {"0": 0.5, "1": 0.5}  # cards 2, 3 deducibly zero
        rows = [_row("goofspiel", stated, true) for _ in range(5)]
        df = pl.DataFrame(rows)

        result = score_property7(
            df,
            cfr_policies={"goofspiel": cfr_pol},
            games={"goofspiel": game},
        )
        expected = 0.5 * (0.25 + 0.25 + 0.25 + 0.25)
        assert result["tvd_mean"]["goofspiel"] == pytest.approx(expected, abs=1e-9)
        # Goofspiel's coverage is filled whenever 7b runs, as every game's is; these rows
        # never reach 7b, so, as on the generic path, no coverage entry is written for them. `test_padding_reachable_via_real_infoset_and_survives_deletion`
        # asserts the filled value on a row that does reach it.
        assert "goofspiel" in result["action_belief_agreement"]

    def test_padding_reachable_via_real_infoset_and_survives_deletion(
        self, monkeypatch
    ):
        """The genuine production-call-shape regression test.
        Uses a real P0 info-state string (found by walking an actual
        CFR+-solved goofspiel game with `gather_p0_worlds`), so
        `_score_7b_goofspiel` genuinely runs against real game state - the
        `action_belief_agreement` assertion below is not nan, proving 7b was
        reached, not short-circuited as in the smoke-check above.
        """
        from lmopt.scoring.goofspiel_p7b import gather_p0_worlds

        game = load_goofspiel(num_cards=4)
        cfr_pol = solve_cfr(game, 50)
        worlds = gather_p0_worlds(game, cfr_pol)
        posteriors = goofspiel_all_posteriors(game, cfr_pol, seat=0)

        info_str = None
        legal_action = None
        for info, states_reach in worlds.items():
            post = posteriors.get(info, {})
            if post and len(post) < 4:
                info_str = info
                legal_action = states_reach[0][0].legal_actions()[0]
                break
        assert (
            info_str is not None
        ), "expected a real narrow-support P0 info-state for this fixture"

        stated = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
        true = {"0": 0.5, "1": 0.5}
        row = _row(
            "goofspiel",
            stated,
            true,
            prompt="p",
            player_to_act=0,
            info_set=info_str,
            parsed_action=legal_action,
        )
        df = pl.DataFrame([row])

        result_union = score_property7(
            df,
            cfr_policies={"goofspiel": cfr_pol},
            games={"goofspiel": game},
        )
        # 7b genuinely ran against this real info-state (not short-circuited
        # to nan by an empty `worlds.get(info_str, [])`).
        assert not math.isnan(
            result_union["action_belief_agreement"].get("goofspiel", float("nan"))
        ), "7b must actually run against a real P0 info-state, not short-circuit"
        # The one row reached the estimand, so goofspiel's 7b coverage is 1.0, and a row
        # that cannot reach it reads 0.0 (see the crash smoke-check above).
        assert result_union["action_belief_coverage"]["goofspiel"] == 1.0

        import lmopt.scoring.property7 as p7mod

        monkeypatch.setattr(
            p7mod,
            "_combine_keys",
            lambda s, t: sorted(set(s.keys()) & set(t.keys())),
        )
        result_intersection = score_property7(
            df,
            cfr_policies={"goofspiel": cfr_pol},
            games={"goofspiel": game},
        )
        assert result_intersection["tvd_mean"]["goofspiel"] == pytest.approx(
            result_union["tvd_mean"]["goofspiel"], abs=1e-9
        ), (
            "intersection must equal union when the padding branch "
            "pre-widens true's keys to stated's; a divergence here means "
            "the padding branch was removed or broken and stated's mass on "
            "the missing cards is being silently dropped"
        )

    def test_non_goofspiel_family_is_not_padded(self):
        """A fixture that discriminates: if the
        goofspiel-only padding wrongly fired for kuhn, it would clip `true`
        to `stated`'s single key "0" - since `true` has no "0" key, padding
        would zero it (`true.get("0", 0.0) == 0.0`), giving TVD 0.5 instead
        of the correct 1.0 that the (unpadded) union path computes over
        keys {"0", "1"}.
        """
        rows = [_row("kuhn", {"0": 1.0}, {"1": 1.0}) for _ in range(10)]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        assert result["tvd_mean"]["kuhn"] == pytest.approx(1.0, abs=1e-9)


class TestGoofspielPaddingInvariant:
    """The explicit zero-padding above assumes
    `stated`'s keys are a superset of `true`'s (goofspiel's elicited belief
    is dense over the full deck; prompting/goofspiel.py:132-134). A raw
    key-union can only ever gain keys, never lose posterior mass; the
    explicit padding can, silently, the moment that assumption breaks - and
    always in the direction of understating TVD (dropped `true` mass reads
    as "the model was more right than it was"). The no-silent-fallbacks
    rule forbids computing around that without surfacing it. This
    class asserts the violation is a loud failure, not a quietly-smaller
    number.
    """

    def test_true_key_absent_from_stated_is_named_not_applicable(self):
        """A violating row does not abort score_property7
        entirely -- it confines the failure to this game, loudly, via
        not_applicable, rather than a silent score.
        """
        stated = {"0": 1.0}  # narrower than a real goofspiel elicited belief
        true = {"0": 0.5, "1": 0.5}  # has a key ("1") absent from stated
        rows = [_row("goofspiel", stated, true)]
        df = pl.DataFrame(rows)
        result = score_property7(df)  # must not raise
        assert "goofspiel" not in result["tvd_mean"]
        assert "goofspiel" in result["not_applicable"]
        assert "1" in result["not_applicable"]["goofspiel"]

    def test_goofspiel_violation_does_not_abort_other_games_in_the_same_call(self):
        """The blast-radius regression this confinement exists to prevent:
        a goofspiel violation must not cost kuhn's already-computed P7a.
        """
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) for _ in range(5)
        ] + [_row("goofspiel", {"0": 1.0}, {"0": 0.5, "1": 0.5})]
        df = pl.DataFrame(rows)
        result = score_property7(df)  # must not raise
        assert result["tvd_mean"]["kuhn"] == pytest.approx(0.0, abs=1e-9)
        assert "goofspiel" not in result["tvd_mean"]
        assert "goofspiel" in result["not_applicable"]

    def test_true_keys_subset_of_stated_does_not_raise(self):
        """Negative control: the ordinary case (true's keys ⊆ stated's, the
        density anchor guarantees this in production) must not raise - the
        guard should fire only on the violation, not on every goofspiel row.
        """
        stated = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
        true = {"0": 0.5, "1": 0.5}
        rows = [_row("goofspiel", stated, true)]
        df = pl.DataFrame(rows)
        result = score_property7(df)  # must not raise
        assert not math.isnan(result["tvd_mean"]["goofspiel"])

    def test_goofspiel_7a_violation_does_not_discard_7b(self):
        """A 7a-only GoofspielSupportViolation must not `continue` past 7b
        for the same game, because 7b's inputs
        (game_lm_belief/game_obj/cfr_pol) are built independently of
        game_row_tvds/`records`. Real goofspiel rows do not trip this guard
        (0 of 899,894 rows across 180 files), so this test's violation is a
        synthetic construction -- the only way to exercise this path.
        """
        game = load_goofspiel(num_cards=4)
        cfr_pol = solve_cfr(game, 10)
        stated = {"0": 1.0}  # narrower than a real goofspiel elicited belief
        true = {"0": 0.5, "1": 0.5}  # has a key ("1") absent from stated
        rows = [
            _row(
                "goofspiel",
                stated,
                true,
                prompt="p",
                player_to_act=0,
                info_set="0",
                parsed_action=0,
            )
            for _ in range(5)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(
            df,
            cfr_policies={"goofspiel": cfr_pol},
            games={"goofspiel": game},
        )
        assert (
            "goofspiel" in result["not_applicable"]
        ), "7a violation must still be recorded"
        assert (
            "goofspiel" not in result["tvd_mean"]
        ), "7a itself stays discarded -- only 7b's fate changes"
        assert (
            "goofspiel" in result["action_belief_agreement"]
            or "goofspiel" in result["discarded_rows"]
        ), "7b must still be attempted for this game despite the 7a violation"


class TestParseBeliefArrayLeducExpansion:
    """Writing `arr[int(k)] = v` for k in {"0","1","2"} (the rank keys)
    would put all mass on card indices 0-2 and leave 3-5 (half the deck) at
    zero - P7b would then best-respond to a belief with no mass on half the
    opponent's possible hands.

    An unconditional 50/50 split per rank is also wrong: it can assign mass
    to a card the opponent provably cannot hold (the acting player's own
    card, or the revealed board card in round 2), which crashes
    `_br_to_belief`'s counterfactual with an uncaught OpenSpiel SpielError.
    These tests build
    real Leduc states (not the game tree traversal - round-1 and round-2
    nodes are constructed directly via `apply_action` so the exact excluded
    set is known) and assert the availability-aware behaviour.
    """

    def test_leduc_family_excludes_the_acting_players_own_card(self):
        # P0 dealt index 0 (rank 0), P1 dealt index 3 (rank 1); P0 to act
        # in round 1. Own card (index 0) must get zero mass; its rank-0
        # partner (index 1) absorbs all of rank 0's stated mass.
        game = load_leduc()
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(3)
        assert state.current_player() == 0 and not state.is_terminal()

        stated = {"0": 0.2, "1": 0.3, "2": 0.5}
        arr = _parse_belief_array(stated, game, family="leduc", state=state, seat=0)
        assert arr is not None
        assert arr.shape == (6,)
        assert arr[0] == pytest.approx(0.0), "own card must get zero mass"
        assert arr[1] == pytest.approx(
            0.2
        ), "rank 0's mass goes entirely to the live copy"
        assert arr[2] == pytest.approx(0.15)
        assert arr[3] == pytest.approx(0.15)
        assert arr[4] == pytest.approx(0.25)
        assert arr[5] == pytest.approx(0.25)
        assert arr.sum() == pytest.approx(1.0)

    def test_leduc_family_excludes_public_board_card_in_round_2_and_renormalises(self):
        # Same deal as above, both players check round 1, board reveals
        # index 1 - same rank (0) as P0's own card (index 0). Both copies
        # of rank 0 are now excluded; stated mass on rank "0" is dropped
        # and the remainder renormalised over ranks 1-2.
        game = load_leduc()
        state = game.new_initial_state()
        state.apply_action(0)  # P0 card, index 0 (rank 0)
        state.apply_action(3)  # P1 card, index 3 (rank 1)
        state.apply_action(1)  # P0 call/check
        state.apply_action(1)  # P1 call/check
        state.apply_action(1)  # board card, index 1 (rank 0)
        assert state.current_player() == 0 and not state.is_terminal()
        assert _leduc_dealt_cards(state) == [0, 3, 1]

        stated = {"0": 0.4, "1": 0.3, "2": 0.3}
        arr = _parse_belief_array(stated, game, family="leduc", state=state, seat=0)
        assert arr is not None
        assert arr[0] == pytest.approx(0.0)
        assert arr[1] == pytest.approx(0.0), "both rank-0 copies are excluded"
        # Remaining 0.6 of stated mass renormalised to sum to 1.0.
        assert arr[2] == pytest.approx(0.25)
        assert arr[3] == pytest.approx(0.25)
        assert arr[4] == pytest.approx(0.25)
        assert arr[5] == pytest.approx(0.25)
        assert arr.sum() == pytest.approx(1.0)

    def test_leduc_family_returns_none_when_all_mass_is_on_an_impossible_rank(self):
        """No silent fallback: if the model's entire stated
        belief is on a rank that is now provably impossible, there is no
        belief left to best-respond to - return None (skipped by the
        caller), never a fabricated uniform prior."""
        game = load_leduc()
        state = game.new_initial_state()
        state.apply_action(0)
        state.apply_action(3)
        state.apply_action(1)
        state.apply_action(1)
        state.apply_action(1)  # board card index 1, same rank as P0's own card

        stated = {"0": 1.0, "1": 0.0, "2": 0.0}
        arr = _parse_belief_array(stated, game, family="leduc", state=state, seat=0)
        assert arr is None

    def test_leduc_family_without_state_or_seat_returns_none(self):
        """Defensive guard: the leduc path cannot determine availability
        without the state, so it must not silently fall back to the
        (unsafe) unconditional split."""
        game = load_leduc()
        stated = {"0": 0.2, "1": 0.3, "2": 0.5}
        assert _parse_belief_array(stated, game, family="leduc") is None
        assert _parse_belief_array(stated, game, family="leduc", seat=0) is None

    def test_non_leduc_family_is_unchanged(self):
        """Kuhn's stated belief keys already are card indices - the family
        gate must leave this path alone."""
        game = load_kuhn()
        stated = {"1": 0.5, "2": 0.5}
        arr = _parse_belief_array(stated, game, family="kuhn")
        assert arr is not None
        assert arr[0] == pytest.approx(0.0)
        assert arr[1] == pytest.approx(0.5)
        assert arr[2] == pytest.approx(0.5)

    def test_family_none_default_is_unchanged(self):
        """Callers that don't pass family (none exist, but the default must
        be safe) get the literal-index behaviour."""
        game = load_kuhn()
        stated = {"1": 0.5, "2": 0.5}
        arr = _parse_belief_array(stated, game)
        assert arr[1] == pytest.approx(0.5)
        assert arr[2] == pytest.approx(0.5)


class TestLeducBRDoesNotDoubleDealACard:
    """An end-to-end call of `score_property7` with `games`/`cfr_policies`
    for Leduc. Tests that skip 7b or exercise only Kuhn (where the stated
    belief already omits the acting player's own card by construction)
    cannot catch an even-split `expand_rank_to_indices`, which crashes on
    most real Leduc rows. This test calls the real end-to-end path with a
    Leduc row whose own card collides with an even split, and must not
    raise `pyspiel.SpielError`.
    """

    def test_7b_leduc_does_not_crash_when_belief_would_double_deal_own_card(self):
        game = load_leduc()
        cfr_pol = solve_cfr(
            game, 10
        )  # small: only need a valid policy object, not convergence

        infoset_map = _build_infoset_map(game)
        target = None
        for (player, info_str), state in infoset_map.items():
            dealt = _leduc_dealt_cards(state)
            if dealt[player] == 1:  # acting player's own card is index 1 (rank 0)
                target = (player, info_str, state)
                break
        assert target is not None, "expected some decision node with own card index 1"
        player, info_str, state = target
        parsed_action = state.legal_actions()[0]

        # Stating any mass on rank "0" is exactly the case an even split
        # mishandles: it puts half of that mass on index 1, the acting
        # player's own card, which `_make_cf_state` then tries to deal a
        # second time (pyspiel.SpielError: "deck_[move] != kInvalidCard",
        # as `_make_cf_state(seat=0, my_card=1, opp_card=1, ...)` raises).
        stated = {"0": 1.0, "1": 0.0, "2": 0.0}
        true_posterior = {str(i): 1 / 6 for i in range(6)}
        rows = [
            {
                "game": "leduc",
                "elicited_belief": json.dumps(stated),
                "true_posterior": json.dumps(true_posterior),
                "prompt": "p",
                "player_to_act": player,
                "info_set": info_str,
                "parsed_action": parsed_action,
            }
        ]
        df = pl.DataFrame(rows)

        # Must not raise. This exact scenario raises pyspiel.SpielError
        # against an even-split version.
        result = score_property7(
            df, cfr_policies={"leduc": cfr_pol}, games={"leduc": game}
        )
        agr = result["action_belief_agreement"]["leduc"]
        assert 0.0 <= agr <= 1.0


class TestProperty7b:
    """7b requires game objects; tests use Kuhn poker."""

    @pytest.fixture(scope="class")
    def kuhn_setup(self):
        game = load_kuhn()
        cfr_pol = solve_cfr(game, 200)
        return game, cfr_pol

    def test_7b_not_computed_without_games(self):
        rows = [
            _row(
                "kuhn",
                {"0": 0.5, "1": 0.5},
                {"0": 0.5, "1": 0.5},
                prompt="p",
                player_to_act=0,
                info_set="0",
                parsed_action=0,
            )
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        assert not result["7b_available"]

    def test_7b_available_when_games_provided(self, kuhn_setup):
        game, cfr_pol = kuhn_setup
        # Kuhn: P0 holds Jack (card 0), info_set = "0", P0 checks (action 0).
        rows = [
            _row(
                "kuhn",
                {"1": 0.5, "2": 0.5},
                {"1": 0.5, "2": 0.5},
                prompt="p",
                player_to_act=0,
                info_set="0",
                parsed_action=0,
            )
        ]
        df = pl.DataFrame(rows)
        result = score_property7(
            df,
            cfr_policies={"kuhn": cfr_pol},
            games={"kuhn": game},
        )
        assert result["7b_available"]
        # Game-level agreement should be a valid fraction (0–1).
        agr = result["action_belief_agreement"].get("kuhn", float("nan"))
        assert not math.isnan(agr)
        assert 0.0 <= agr <= 1.0
        # Pin the other direction of the partial-reference guard: a
        # full CFR+ TabularPolicy must not be misidentified as partial (it has
        # neither `lookup` nor `n_infosets`), so 7b must not silently disable
        # itself for the canonical games.
        assert "kuhn" not in result.get("not_applicable", {})

    def test_7b_partial_reference_is_refused_with_a_reason(self, tmp_path, kuhn_setup):
        """A PARTIAL depth-reference policy must not be handed
        to the full-tree best-response walk - the off-support gaps make a
        full-tree BR undefined, and (at 3x3 scale) the tree is intractable."""
        import pickle

        from lmopt.games.depth_reference import load_depth_reference
        from tests.test_depth_reference import _truncated_kuhn_payload

        game, _ = kuhn_setup
        payload = _truncated_kuhn_payload(game, keep=2)
        path = tmp_path / "trunc.pkl"
        path.write_bytes(pickle.dumps(payload))
        partial_ref = load_depth_reference(path, game)

        rows = [
            _row(
                "kuhn",
                {"1": 0.5, "2": 0.5},
                {"1": 0.5, "2": 0.5},
                prompt="p",
                player_to_act=0,
                info_set="0",
                parsed_action=0,
            )
        ]
        df = pl.DataFrame(rows)
        result = score_property7(
            df,
            cfr_policies={"kuhn": partial_ref},
            games={"kuhn": game},
        )
        assert "kuhn" in result["not_applicable"]
        assert result["not_applicable"]["kuhn"], "reason string must be non-empty"
        assert (
            "kuhn" not in result["action_belief_agreement"]
        ), "a partial reference must not produce a 7b score"

    def test_7b_per_infoset_present_when_games_provided(self, kuhn_setup):
        game, cfr_pol = kuhn_setup
        # "0" = P0 holds Jack, first to act.
        # "0pb" = P0 holds Jack, checked, P1 bet, P0 acts again.
        # Both are valid P0 decision nodes in OpenSpiel's Kuhn tree.
        rows = [
            _row(
                "kuhn",
                {"1": 0.5, "2": 0.5},
                {"1": 0.5, "2": 0.5},
                prompt="p",
                player_to_act=0,
                info_set="0",
                parsed_action=0,
            ),
            _row(
                "kuhn",
                {"1": 0.5, "2": 0.5},
                {"1": 0.5, "2": 0.5},
                prompt="p",
                player_to_act=0,
                info_set="0pb",
                parsed_action=0,
            ),
        ]
        df = pl.DataFrame(rows)
        result = score_property7(
            df,
            cfr_policies={"kuhn": cfr_pol},
            games={"kuhn": game},
        )
        assert "action_belief_per_infoset" in result
        per_is = result["action_belief_per_infoset"].get("kuhn", {})
        assert isinstance(per_is, dict)
        # Both info-sets must be independently tracked.
        assert "0" in per_is, "info-set '0' not found in per-infoset output"
        assert "0pb" in per_is, "info-set '0pb' not found in per-infoset output"
        for val in per_is.values():
            assert 0.0 <= val <= 1.0

    def test_7b_per_infoset_not_present_without_games(self):
        rows = [
            _row(
                "kuhn",
                {"1": 0.5, "2": 0.5},
                {"1": 0.5, "2": 0.5},
                prompt="p",
                player_to_act=0,
                info_set="0",
                parsed_action=0,
            )
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)
        # Key is present but empty when 7b is not computed.
        assert "action_belief_per_infoset" in result
        assert result["action_belief_per_infoset"] == {}


class TestDarkhexNullBeliefShard:
    """A real darkhex shard can carry beliefs that are all NULL. The family is
    scored, so the shard's own emptiness is the only thing to say about it,
    and nothing may invent a reason about the family from a cell that simply
    carries no belief."""

    def _null_belief_darkhex_df(self):
        return pl.DataFrame(
            {
                "game": ["darkhex_3x3"] * 3,
                "matchup_id": ["lm_vs_random"] * 3,
                "elicited_belief": [None, None, None],
                "true_posterior": [None, None, None],
                "prompt": ["p", None, "p"],
                "player_to_act": [0, 1, 0],
                "info_set": ["a", "b", "c"],
                "parsed_action": [0, 1, 2],
            },
            schema_overrides={
                "elicited_belief": pl.Utf8,
                "true_posterior": pl.Utf8,
                "prompt": pl.Utf8,
            },
        )

    def test_no_family_reason_is_invented_for_a_null_belief_shard(self):
        result = score_property7(self._null_belief_darkhex_df())
        assert "darkhex_3x3" not in result["not_applicable"]

    def test_a_null_belief_shard_populates_no_subscore(self):
        """The converse direction of the consumer rule: no belief row means no
        7a number either, scored family or not.
        """
        result = score_property7(self._null_belief_darkhex_df())
        assert "darkhex_3x3" not in result["tvd_mean"]
        assert "darkhex_3x3" not in result["action_belief_agreement"]


class TestActorParameterisation:
    """Explicit belief-actor selection on P7a.

    Without an actor filter, `belief_rows` (`property7.py:76`) pools both
    actors: it selects on the two belief columns and excludes non-LM rows only
    incidentally, because `match_runner._get_lm_metadata` defaults a missing
    `elicited_belief` to None. That exclusion is a coincidence of the agent
    roster, not a filter.
    """

    @staticmethod
    def _belief_row(actor, stated, true, prompt=None):
        row = _row("kuhn", stated, true, prompt=prompt)
        row["actor"] = actor
        return row

    def test_a_foreign_agents_belief_row_is_excluded_from_7a(self):
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

        out = score_property7(df, actor="model-x")
        assert out["actor_filter"] == "actor=model-x"
        assert out["n_belief_rows"]["kuhn"] == 2
        assert out["tvd_mean"]["kuhn"] == pytest.approx(0.2, abs=1e-12)

        pooled = score_property7(df)
        assert pooled["n_belief_rows"]["kuhn"] == 3
        assert pooled["tvd_mean"]["kuhn"] == pytest.approx(
            (0.2 + 0.2 + 1.0) / 3, abs=1e-12
        )

    def test_legacy_frame_pools_as_before_and_says_so(self):
        """Legacy frame for P7: no `actor` column, so both populations keep their
        column-based filters and the fallback is labelled.
        """
        df = pl.DataFrame([_row("kuhn", {"0": 0.8, "1": 0.2}, {"0": 1.0, "1": 0.0})])
        assert "actor" not in df.columns
        out = score_property7(df)
        assert out["actor_filter"] == "legacy-belief-columns-and-prompt-null"
        assert out["tvd_mean"]["kuhn"] == pytest.approx(0.2, abs=1e-12)

    def test_naming_an_actor_a_legacy_frame_cannot_identify_raises(self):
        df = pl.DataFrame([_row("kuhn", {"0": 1.0}, {"0": 1.0})])
        with pytest.raises(ValueError, match="actor"):
            score_property7(df, actor="model-x")


class TestGoofspielSeatOneWiredAndDegenerateBeliefGuard:
    """Seat-1 rows are wired into the same unified P7b agreement figure via
    `gather_p1_prefix_worlds` + `goofspiel_br_set_seat1`. A second,
    independent guard in the same loop: a degenerate or off-support
    `elicited_belief` (no positive mass on any currently-legal opponent
    action) is counted and discarded rather than silently credited as a
    match.
    """

    def _real_p0_info(self):
        from lmopt.scoring.goofspiel_p7b import gather_p0_worlds

        game = load_goofspiel(num_cards=4)
        cfr_pol = solve_cfr(game, 50)
        worlds = gather_p0_worlds(game, cfr_pol)
        info, states_reach = next(iter(worlds.items()))
        return game, cfr_pol, info, states_reach

    def _real_p1_info(self, game, cfr_pol):
        from lmopt.scoring.goofspiel_p7b import gather_p1_prefix_worlds

        worlds = gather_p1_prefix_worlds(game, cfr_pol)
        info, states_reach = next(iter(worlds.items()))
        return info, states_reach

    def test_degenerate_belief_returns_full_action_set_not_empty(self):
        """Characterises the behaviour the degenerate-belief guard exists for:
        `goofspiel_br_set` on an empty or off-support belief
        returns the full legal action set, not an empty one, because every
        candidate action's EV ties at 0.0 when no belief mass lands on any
        of its opponent-action terms. This is why a bare `if not br:
        continue` guard cannot detect it, and why the guard must inspect
        the belief's support directly rather than trust `br`'s emptiness.
        """
        from lmopt.scoring.goofspiel_p7b import goofspiel_br_set

        game, cfr_pol, p0_info, p0_states_reach = self._real_p0_info()
        legal_b0 = set(p0_states_reach[0][0].legal_actions())

        assert set(goofspiel_br_set(p0_states_reach, cfr_pol, {})) == legal_b0
        assert set(goofspiel_br_set(p0_states_reach, cfr_pol, {999: 1.0})) == legal_b0

    def test_seat1_row_contributes_via_direct_call(self):
        """Mechanism anchor: `_score_7b_goofspiel` itself scores a
        seat-1 row using `goofspiel_br_set_seat1`, rather than counting it
        under `n_wrong_seat` and skipping it.
        """
        from lmopt.scoring.goofspiel_p7b import goofspiel_br_set_seat1
        from lmopt.scoring.property7 import _score_7b_goofspiel

        game, cfr_pol, _p0_info, _p0_states_reach = self._real_p0_info()
        p1_info, p1_states_reach = self._real_p1_info(game, cfr_pol)

        dense = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
        dense_int = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        br1 = goofspiel_br_set_seat1(p1_states_reach, cfr_pol, dense_int)
        assert br1, "fixture must have a defined BR set for this p1 info-state"

        row = _row(
            "goofspiel",
            dense,
            {"0": 0.5, "1": 0.5},
            prompt="p",
            player_to_act=1,
            info_set=p1_info,
            parsed_action=br1[0],
        )
        df = pl.DataFrame([row])
        agreement, per_infoset, n_discarded, _ties, _n_scored = _score_7b_goofspiel(
            df, game, cfr_pol
        )

        assert n_discarded == 0
        assert agreement == pytest.approx(1.0, abs=1e-9)
        assert per_infoset.get(p1_info) == pytest.approx(1.0, abs=1e-9)

    def test_seat0_degenerate_belief_discarded_not_counted(self):
        from lmopt.scoring.property7 import _score_7b_goofspiel

        game, cfr_pol, p0_info, p0_states_reach = self._real_p0_info()
        played = p0_states_reach[0][0].legal_actions()[0]

        row = _row(
            "goofspiel",
            {"0": 0.0, "1": 0.0, "999": 1.0},
            {"0": 0.5, "1": 0.5},
            prompt="p",
            player_to_act=0,
            info_set=p0_info,
            parsed_action=played,
        )
        df = pl.DataFrame([row])
        agreement, per_infoset, n_discarded, _ties, _n_scored = _score_7b_goofspiel(
            df, game, cfr_pol
        )

        assert n_discarded == 1
        assert math.isnan(agreement)
        assert per_infoset == {}

    def test_seat1_degenerate_belief_discarded_not_counted(self):
        """Symmetric seat-1 case of the guard above. Paired with a real
        seat-1 hit row in the same call: this is the differentiator against
        a seat-based discard - under such a discard both seat-1 rows would
        be counted as `n_wrong_seat` (2) and
        neither would reach `matches`/`total`, so `agreement` would stay nan
        rather than resolve to the hit row's 1.0. A version of this test
        with only the degenerate row cannot tell "discarded because the
        belief is degenerate" apart from "discarded because it's seat 1",
        since both produce identical output for a single isolated row.
        """
        from lmopt.scoring.goofspiel_p7b import goofspiel_br_set_seat1
        from lmopt.scoring.property7 import _score_7b_goofspiel

        game, cfr_pol, _p0_info, _p0_states_reach = self._real_p0_info()
        p1_info, p1_states_reach = self._real_p1_info(game, cfr_pol)
        probe = p1_states_reach[0][0]
        legal_b1 = probe.child(probe.legal_actions()[0]).legal_actions()

        dense_int = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        br1 = goofspiel_br_set_seat1(p1_states_reach, cfr_pol, dense_int)
        assert br1, "fixture must have a defined BR set for this p1 info-state"

        rows = [
            _row(
                "goofspiel",
                {"999": 1.0},
                {"0": 0.5, "1": 0.5},
                prompt="p",
                player_to_act=1,
                info_set=p1_info,
                parsed_action=legal_b1[0],
            ),
            _row(
                "goofspiel",
                {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
                {"0": 0.5, "1": 0.5},
                prompt="p",
                player_to_act=1,
                info_set=p1_info,
                parsed_action=br1[0],
            ),
        ]
        df = pl.DataFrame(rows)
        agreement, per_infoset, n_discarded, _ties, _n_scored = _score_7b_goofspiel(
            df, game, cfr_pol
        )

        assert n_discarded == 1
        assert agreement == pytest.approx(1.0, abs=1e-9)
        assert per_infoset.get(p1_info) == pytest.approx(1.0, abs=1e-9)

    def test_all_zero_mass_on_legal_actions_is_also_degenerate(self):
        """A belief dict that is non-empty and entirely on-support, but
        every listed probability is 0.0, must still be caught: a naive
        `if belief:` truthiness check (the obvious but insufficient guard)
        would treat this dict as "has content" and let it through.
        """
        from lmopt.scoring.property7 import _score_7b_goofspiel

        game, cfr_pol, p0_info, p0_states_reach = self._real_p0_info()
        legal_b0 = p0_states_reach[0][0].legal_actions()
        played = legal_b0[0]
        zero_mass = {str(a): 0.0 for a in legal_b0}

        row = _row(
            "goofspiel",
            zero_mass,
            {"0": 0.5, "1": 0.5},
            prompt="p",
            player_to_act=0,
            info_set=p0_info,
            parsed_action=played,
        )
        df = pl.DataFrame([row])
        agreement, per_infoset, n_discarded, _ties, _n_scored = _score_7b_goofspiel(
            df, game, cfr_pol
        )

        assert n_discarded == 1
        assert math.isnan(agreement)

    def test_probe_vs_union_divergence_is_not_wrongly_discarded_seat0(self):
        """A single-world probe of the opponent's legal actions is a strict
        subset of what `goofspiel_br_set` actually consults across every
        world sharing an info-state - different histories reaching the same
        P0 info-state can have consumed different P1 cards, so P1's
        remaining legal actions genuinely differ by world. Searches the real
        fixture for a P0 info-state where this provably diverges (40/81 P0
        info-states in this exact fixture), rather than hardcoding one
        info-string - which info-states diverge depends on `solve_cfr`'s
        tree, not this test. Every fixture in the other tests in this class
        picks `next(iter(worlds.items()))`/`entries[0]`, so probe and union
        agree on all of them by construction - none of them could catch
        this defect; this test is specifically constructed so probe and
        union disagree.
        """
        from lmopt.scoring.goofspiel_p7b import gather_p0_worlds, goofspiel_br_set
        from lmopt.scoring.property7 import (
            _goofspiel_legal_opponent_actions,
            _score_7b_goofspiel,
        )

        game = load_goofspiel(num_cards=4)
        cfr_pol = solve_cfr(game, 50)
        worlds0 = gather_p0_worlds(game, cfr_pol)

        found = None
        for info, entries in worlds0.items():
            if len(entries) < 2:
                continue
            probe_state = entries[0][0]
            b0 = probe_state.legal_actions()[0]
            probe = set(probe_state.child(b0).legal_actions())
            union = _goofspiel_legal_opponent_actions(entries, 0)
            extra = union - probe
            if extra:
                found = (info, entries, sorted(extra)[0])
                break
        assert found is not None, (
            "expected at least one P0 info-state in this real fixture where "
            "a single-world probe of legal opponent actions differs from "
            "the true union (40/81 measured)"
        )
        info, entries, extra_action = found

        belief = {extra_action: 1.0}
        br = goofspiel_br_set(entries, cfr_pol, belief)
        assert br, "fixture's BR set for this belief must be non-empty"

        row = _row(
            "goofspiel",
            {str(extra_action): 1.0},
            {"0": 0.5, "1": 0.5},
            prompt="p",
            player_to_act=0,
            info_set=info,
            parsed_action=br[0],
        )
        df = pl.DataFrame([row])
        agreement, per_infoset, n_discarded, _ties, _n_scored = _score_7b_goofspiel(
            df, game, cfr_pol
        )

        assert n_discarded == 0, (
            "a belief with real, union-reachable support must not be "
            "discarded as degenerate just because it lands outside a "
            "single arbitrary world's legal-action set"
        )
        assert agreement == pytest.approx(1.0, abs=1e-9)

    def test_probe_vs_union_divergence_is_not_wrongly_discarded_seat1(self):
        """Seat-1 mirror of the test above: `goofspiel_br_set_seat1` also
        recomputes the opponent's (P0's) legal actions PER prefix world
        (`prefix_state.legal_actions()` inside its own
        `for prefix_state, reach in worlds_for_info` loop), so the same
        single-world-probe defect applies symmetrically. 40/81
        P1 info-states diverge in this exact fixture.
        """
        from lmopt.scoring.goofspiel_p7b import (
            gather_p1_prefix_worlds,
            goofspiel_br_set_seat1,
        )
        from lmopt.scoring.property7 import (
            _goofspiel_legal_opponent_actions,
            _score_7b_goofspiel,
        )

        game = load_goofspiel(num_cards=4)
        cfr_pol = solve_cfr(game, 50)
        worlds1 = gather_p1_prefix_worlds(game, cfr_pol)

        found = None
        for info, entries in worlds1.items():
            if len(entries) < 2:
                continue
            probe = set(entries[0][0].legal_actions())
            union = _goofspiel_legal_opponent_actions(entries, 1)
            extra = union - probe
            if extra:
                found = (info, entries, sorted(extra)[0])
                break
        assert found is not None, (
            "expected at least one P1 info-state in this real fixture where "
            "a single-world probe of legal opponent actions differs from "
            "the true union (40/81 measured)"
        )
        info, entries, extra_action = found

        belief = {extra_action: 1.0}
        br = goofspiel_br_set_seat1(entries, cfr_pol, belief)
        assert br, "fixture's BR set for this belief must be non-empty"

        row = _row(
            "goofspiel",
            {str(extra_action): 1.0},
            {"0": 0.5, "1": 0.5},
            prompt="p",
            player_to_act=1,
            info_set=info,
            parsed_action=br[0],
        )
        df = pl.DataFrame([row])
        agreement, per_infoset, n_discarded, _ties, _n_scored = _score_7b_goofspiel(
            df, game, cfr_pol
        )

        assert n_discarded == 0
        assert agreement == pytest.approx(1.0, abs=1e-9)

    def test_score_property7_unifies_both_seats_and_reports_degenerate_discard(self):
        """Reachability anchor: the same behaviour verified above through
        the direct call is present via the real production entrypoint
        `score_property7` - both seats' real-belief rows are hits, folded
        into one unified `action_belief_agreement["goofspiel"]`, and the
        degenerate row is named in `discarded_rows` under the belief-support
        reason, not a seat-mismatch reason.
        """
        from lmopt.scoring.goofspiel_p7b import goofspiel_br_set, goofspiel_br_set_seat1

        game, cfr_pol, p0_info, p0_states_reach = self._real_p0_info()
        p1_info, p1_states_reach = self._real_p1_info(game, cfr_pol)

        dense = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
        dense_int = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        br0 = goofspiel_br_set(p0_states_reach, cfr_pol, dense_int)
        br1 = goofspiel_br_set_seat1(p1_states_reach, cfr_pol, dense_int)

        rows = [
            _row(
                "goofspiel",
                dense,
                {"0": 0.5, "1": 0.5},
                prompt="p",
                player_to_act=0,
                info_set=p0_info,
                parsed_action=br0[0],
            ),
            _row(
                "goofspiel",
                dense,
                {"0": 0.5, "1": 0.5},
                prompt="p",
                player_to_act=1,
                info_set=p1_info,
                parsed_action=br1[0],
            ),
            _row(
                "goofspiel",
                {"0": 0.0, "1": 0.0, "999": 1.0},
                {"0": 0.5, "1": 0.5},
                prompt="p",
                player_to_act=0,
                info_set=p0_info,
                parsed_action=p0_states_reach[0][0].legal_actions()[0],
            ),
        ]
        out = score_property7(
            pl.DataFrame(rows),
            cfr_policies={"goofspiel": cfr_pol},
            games={"goofspiel": game},
        )
        assert out["action_belief_agreement"]["goofspiel"] == pytest.approx(
            1.0, abs=1e-9
        ), "both real BR rows hit; the degenerate row must not drag this down"
        dropped = out["discarded_rows"]["goofspiel"]
        assert dropped["n_rows"] == 1
        assert "belief" in dropped["reason"].lower()
        # No seat-mismatch reason text: "seat 0 only" is its distinguishing
        # phrase (goofspiel_br_set_seat1 legitimately appears in the belief
        # reason, so a bare "seat" substring check would be a false positive).
        assert "seat 0 only" not in dropped["reason"].lower()

    def test_score_property7_records_no_drop_when_nothing_degenerate(self):
        """The converse: with real-support beliefs at both seats and no
        degenerate row, `discarded_rows` must not name goofspiel at all - a
        fixed-seat or seat-random arm with no defect gets no spurious
        caveat.
        """
        from lmopt.scoring.goofspiel_p7b import goofspiel_br_set, goofspiel_br_set_seat1

        game, cfr_pol, p0_info, p0_states_reach = self._real_p0_info()
        p1_info, p1_states_reach = self._real_p1_info(game, cfr_pol)

        dense = {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
        dense_int = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        br0 = goofspiel_br_set(p0_states_reach, cfr_pol, dense_int)
        br1 = goofspiel_br_set_seat1(p1_states_reach, cfr_pol, dense_int)

        rows = [
            _row(
                "goofspiel",
                dense,
                {"0": 0.5, "1": 0.5},
                prompt="p",
                player_to_act=0,
                info_set=p0_info,
                parsed_action=br0[0],
            ),
            _row(
                "goofspiel",
                dense,
                {"0": 0.5, "1": 0.5},
                prompt="p",
                player_to_act=1,
                info_set=p1_info,
                parsed_action=br1[0],
            ),
        ]
        out = score_property7(
            pl.DataFrame(rows),
            cfr_policies={"goofspiel": cfr_pol},
            games={"goofspiel": game},
        )
        assert "goofspiel" not in out["discarded_rows"]
        assert not math.isnan(out["action_belief_agreement"]["goofspiel"])


class TestPerInfoSetTVD:
    """Per-info-set mean of 7a TVDs, pooling exactly the same per-row TVDs
    the existing 7a mean uses (same alignment: leduc rank-marginalisation,
    goofspiel zero-padding, dark hex family skip).
    """

    def test_per_infoset_means_differ_from_pooled_when_infosets_diverge(self):
        """Two info-sets with different per-row TVDs yield per-info-set means
        distinct from the pooled `tvd_mean`.
        """
        rows = [
            _row("kuhn", {"0": 0.8, "1": 0.2}, {"0": 1.0, "1": 0.0}, info_set="info_A")
            for _ in range(5)
        ] + [
            _row("kuhn", {"0": 0.4, "1": 0.6}, {"0": 1.0, "1": 0.0}, info_set="info_B")
            for _ in range(5)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)

        # Pooled mean is (0.2*5 + 0.6*5) / 10 = 0.4
        assert result["tvd_mean"]["kuhn"] == pytest.approx(0.4, abs=1e-12)

        # Per-info-set means are distinct
        per_is = result["tvd_per_infoset"]["kuhn"]
        assert per_is["info_A"] == pytest.approx(0.2, abs=1e-12)
        assert per_is["info_B"] == pytest.approx(0.6, abs=1e-12)

    def test_visit_weighted_mean_of_per_infoset_reproduces_pooled(self):
        """The visit-weighted mean of `tvd_per_infoset` values reproduces
        `tvd_mean` per game.
        """
        # Leduc fixture: true posterior over 6 card indices, stated belief
        # over 3 ranks. The alignment (rank-marginalisation) happens before
        # the per-row TVD is computed. Recomputing TVD naively would give
        # different values.
        rows = [
            _row(
                "leduc",
                {"0": 0.5, "1": 0.3, "2": 0.2},
                {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0, "5": 0.0},
                info_set="leduc_info_1",
            )
            for _ in range(3)
        ] + [
            _row(
                "leduc",
                {"0": 0.4, "1": 0.4, "2": 0.2},
                {"0": 0.0, "1": 0.0, "2": 0.5, "3": 0.0, "4": 0.5, "5": 0.0},
                info_set="leduc_info_2",
            )
            for _ in range(7)
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)

        pooled = result["tvd_mean"]["leduc"]
        per_is = result["tvd_per_infoset"]["leduc"]
        n_belief = result["n_belief_rows"]["leduc"]

        # Compute visit-weighted mean
        weighted_sum = 0.0
        for info_set, tvd_val in per_is.items():
            # Count rows for this info-set
            visits = sum(1 for r in rows if r["info_set"] == info_set)
            weighted_sum += tvd_val * visits

        reconstructed = weighted_sum / n_belief
        assert reconstructed == pytest.approx(pooled, abs=1e-12)

    def test_dark_hex_reaches_the_per_infoset_dict_like_any_other_family(self):
        """Dark hex 7a is the mean per-cell absolute error, pooled into the
        same `tvd_mean`/`tvd_per_infoset` pair every other family uses, with
        no recorded reason.
        """
        rows = [
            _row(
                "darkhex",
                {"0": 0.5, "1": 0.5},
                {"0": 0.5, "1": 0.5},
                info_set="hex_info",
            )
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)

        assert result["tvd_mean"]["darkhex"] == pytest.approx(0.0)
        assert result["tvd_per_infoset"]["darkhex"]["hex_info"] == pytest.approx(0.0)
        assert "darkhex" not in result["not_applicable"]


class TestTask2RefactorPin:
    """score_property7 consumes game_row_tvds, byte-identical.

    Running the 7a inner loop (property7.py:188-277) through the shared
    helper _belief_alignment.game_row_tvds must not change any number that
    reaches a published figure. These tests pin the baseline output.
    """


    def test_unparseable_json_counted_not_silent(self):
        """        An unparseable belief is a counted, visible category, not a silent
        `continue`.
        """
        rows = [
            _row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}),
            {
                "game": "kuhn",
                "elicited_belief": "not valid json{",
                "true_posterior": json.dumps({"0": 0.5, "1": 0.5}),
                "prompt": None,
                "player_to_act": 0,
                "info_set": "0",
                "parsed_action": 0,
            },
        ]
        df = pl.DataFrame(rows)
        result = score_property7(df)

        assert "skipped_unparseable_json" in result
        assert result["skipped_unparseable_json"]["kuhn"] == 1


class TestForeignProbeType:
    """Without a probe_type guard on the plain (non-paired) scorers, a frame
    assembled from a directory that also holds a probe arm's rows (reachable
    via an unfiltered `--data` glob) would be silently
    pooled into the TVD/belief-consistency estimate.
    """

    def test_refuses_a_frame_holding_a_foreign_probe_type(self):
        base = pl.DataFrame([_row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5})])
        mixed = pl.concat(
            [
                base.with_columns(pl.lit("none").alias("probe_type")),
                base.with_columns(pl.lit("relabel").alias("probe_type")),
            ]
        )
        with pytest.raises(ValueError, match="relabel"):
            score_property7(mixed)

    def test_accepts_an_all_base_frame(self):
        df = pl.DataFrame(
            [_row("kuhn", {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5})]
        ).with_columns(pl.lit("none").alias("probe_type"))
        score_property7(df)  # must not raise


# ---------------------------------------------------------------------------
# P7 k-sample integration
# ---------------------------------------------------------------------------


class TestSevenBTies:
    """
    7b credits every action that attains the best value under the stated belief, and counts
    the rows where that leaves the choice wide open or where the belief names no best
    response at all.
    """

    @pytest.fixture(scope="class")
    def kuhn_setup(self):
        game = load_kuhn()
        return game, solve_cfr(game, 200)

    def _state(self, game, info_set="0", seat=0):
        from lmopt.scoring.property7 import _build_infoset_map

        return _build_infoset_map(game)[(seat, info_set)]

    def test_a_belief_with_no_positive_mass_names_no_best_response(self, kuhn_setup):
        """
        An all-zero stated belief cannot pick out a best action, so the row is reported as
        having no mass instead of being credited to the lowest-index legal action.
        """
        from lmopt.scoring.property7 import BR_NO_MASS, _br_set_to_belief

        game, cfr_pol = kuhn_setup
        br, status = _br_set_to_belief(self._state(game), 0, np.zeros(3), cfr_pol)
        assert (br, status) == (None, BR_NO_MASS)

    def test_every_action_attaining_the_best_value_is_a_best_response(
        self,
        kuhn_setup,
        monkeypatch,
    ):
        """
        With every continuation worth the same, every legal action attains the best value,
        so the best-response set is the whole legal set rather than its first member.
        """
        from lmopt.scoring import property7
        from lmopt.scoring.property7 import BR_OK, _br_set_to_belief

        game, cfr_pol = kuhn_setup
        monkeypatch.setattr(
            property7,
            "pooled_action_values",
            lambda state, *a, **k: {x: 0.25 for x in state.legal_actions()},
        )
        state = self._state(game)
        belief = np.zeros(3)
        belief[1] = belief[2] = 0.5
        br, status = _br_set_to_belief(state, 0, belief, cfr_pol)
        assert status == BR_OK
        assert br == set(state.legal_actions()) and len(br) > 1

    def test_the_scorer_reports_the_all_tie_share_and_credits_every_action(
        self,
        kuhn_setup,
        monkeypatch,
    ):
        """
        Through the real entrypoint with every continuation worth the same: both rows are
        hits whatever they played, and the all-tie share says the agreement is trivial.
        """
        from lmopt.scoring import property7

        game, cfr_pol = kuhn_setup
        monkeypatch.setattr(
            property7,
            "pooled_action_values",
            lambda state, *a, **k: {x: 0.25 for x in state.legal_actions()},
        )
        rows = [
            _row(
                "kuhn",
                {"1": 0.5, "2": 0.5},
                {"1": 0.5, "2": 0.5},
                prompt="p",
                player_to_act=0,
                info_set="0",
                parsed_action=action,
            )
            for action in (0, 1)
        ]
        out = property7.score_property7(
            pl.DataFrame(rows), cfr_policies={"kuhn": cfr_pol}, games={"kuhn": game}
        )
        assert out["action_belief_agreement"]["kuhn"] == pytest.approx(1.0)
        assert out["action_belief_all_tie_share"]["kuhn"] == pytest.approx(1.0)
        assert out["action_belief_no_mass_rows"]["kuhn"] == 0

    def test_massless_rows_are_counted_and_never_scored(self, kuhn_setup):
        """
        Through the real entrypoint: rows whose stated belief puts mass on no card at all
        leave 7b with nothing to score, and the count of them is reported.
        """
        from lmopt.scoring import property7

        game, cfr_pol = kuhn_setup
        rows = [
            _row(
                "kuhn",
                {"1": 0.0, "2": 0.0},
                {"1": 0.5, "2": 0.5},
                prompt="p",
                player_to_act=0,
                info_set="0",
                parsed_action=action,
            )
            for action in (0, 1)
        ]
        out = property7.score_property7(
            pl.DataFrame(rows), cfr_policies={"kuhn": cfr_pol}, games={"kuhn": game}
        )
        assert out["action_belief_no_mass_rows"]["kuhn"] == 2
        assert math.isnan(out["action_belief_agreement"]["kuhn"])
        assert math.isnan(out["action_belief_all_tie_share"]["kuhn"])


class TestCardGame7bExitsAreCounted:
    def test_each_unscored_row_is_counted_with_its_reason(self):
        """
        A card-game 7b row that leaves the loop unscored is counted into `discarded_rows` with
        its reason: a belief that is not JSON, an info set with no state in the game tree, and a
        belief that cannot be mapped onto card indices (the path through the blanket except).
        One row of each, beside two scoreable rows, so the counts are 1, 1, 1.
        """
        game = load_kuhn()
        cfr_pol = solve_cfr(game, 50)
        good = _row("kuhn", {"1": 0.5, "2": 0.5}, {"1": 0.5, "2": 0.5}, prompt="p")
        not_json = dict(good, elicited_belief="not json at all")
        no_state = dict(good, info_set="no_such_info_set")
        not_mapped = _row("kuhn", {"7": 1.0}, {"1": 0.5, "2": 0.5}, prompt="p")
        df = pl.DataFrame([good, good, not_json, no_state, not_mapped])
        result = score_property7(
            df, cfr_policies={"kuhn": cfr_pol}, games={"kuhn": game}
        )
        entry = result["discarded_rows"]["kuhn"]
        assert entry["n_rows"] == 3
        for reason in (
            "1 elicited_belief is not JSON",
            "1 info_set has no representative state in the game tree",
            "1 the belief could not be mapped onto card indices",
        ):
            assert reason in entry["reason"], entry
        assert not math.isnan(result["action_belief_agreement"]["kuhn"])

    def test_a_cell_with_every_row_scored_reports_no_discards(self):
        game = load_kuhn()
        cfr_pol = solve_cfr(game, 50)
        good = _row("kuhn", {"1": 0.5, "2": 0.5}, {"1": 0.5, "2": 0.5}, prompt="p")
        result = score_property7(
            pl.DataFrame([good, good]),
            cfr_policies={"kuhn": cfr_pol},
            games={"kuhn": game},
        )
        assert "kuhn" not in result["discarded_rows"]


class TestCardGame7bRowGenerator:
    """
    the one Kuhn and Leduc 7b row generator that the sampled and the exact scorer both read
    """

    @staticmethod
    def _rows():
        good = _row("kuhn", {"1": 0.5, "2": 0.5}, {"1": 0.5, "2": 0.5}, prompt="p")
        return {
            "good": good,
            "not_json": dict(good, elicited_belief="not json at all"),
            "null_action": dict(good, parsed_action=None),
            "no_state": dict(good, info_set="no_such_info_set"),
            "not_mapped": _row("kuhn", {"7": 1.0}, {"1": 0.5, "2": 0.5}, prompt="p"),
            "no_mass": _row(
                "kuhn", {"1": 0.0, "2": 0.0}, {"1": 0.5, "2": 0.5}, prompt="p"
            ),
        }

    def test_every_row_yields_a_scored_tuple_or_its_named_exit(self):
        """
        One row per way to leave the loop yields exactly that reason, and the scoreable row a
        tuple.
        """
        game = load_kuhn()
        cfr_pol = solve_cfr(game, 50)
        rows = self._rows()
        out = list(
            property7._card_game_scored_rows(
                pl.DataFrame(list(rows.values())), game, "kuhn", cfr_pol
            )
        )
        reasons = [str(i) for i in out if isinstance(i, property7._CardGameExit)]
        assert reasons == [
            property7._EXIT_BELIEF_NOT_JSON,
            property7._EXIT_NULL_ACTION,
            property7._EXIT_NO_STATE,
            property7._EXIT_NOT_MAPPED,
            property7._EXIT_NO_MASS,
        ]
        assert out[0][0] == "0" and out[0][1] == 0

    def test_the_exact_pass_does_not_need_a_parsed_action(self):
        """
        With `needs_action=False` a null-action row is scored, not exited.
        """
        game = load_kuhn()
        cfr_pol = solve_cfr(game, 50)
        df = pl.DataFrame([self._rows()["null_action"]])
        (item,) = property7._card_game_scored_rows(
            df, game, "kuhn", cfr_pol, needs_action=False
        )
        assert not isinstance(item, property7._CardGameExit)
        assert item[1] is None

    def test_br_cache_computes_each_distinct_set_once(self, monkeypatch):
        """
        Two passes over the same rows with one cache call `_br_set_to_belief` once.
        """
        game = load_kuhn()
        cfr_pol = solve_cfr(game, 50)
        df = pl.DataFrame([self._rows()["good"]])
        calls = []
        real = property7._br_set_to_belief

        def counting(*args):
            calls.append(1)
            return real(*args)

        monkeypatch.setattr(property7, "_br_set_to_belief", counting)
        cache: dict = {}
        for _ in range(2):
            list(
                property7._card_game_scored_rows(
                    df, game, "kuhn", cfr_pol, br_cache=cache
                )
            )
        assert len(calls) == 1 and len(cache) == 1

    def test_sampled_and_exact_both_read_the_generator(self, monkeypatch):
        """
        Both scorers' Kuhn rows come from the generator, and the exact pass counts the no-mass
        exit that the sampled pass counts apart.
        """
        game = load_kuhn()
        cfr_pol = solve_cfr(game, 50)
        rows = self._rows()
        df = pl.DataFrame([rows["good"], rows["no_mass"]]).with_columns(
            pl.lit("M0").alias("memory_rung"), pl.lit(False).alias("fallback_used")
        )
        seen = []
        real = property7._card_game_scored_rows

        def spy(*args, **kwargs):
            seen.append(kwargs.get("needs_action", True))
            return real(*args, **kwargs)

        monkeypatch.setattr(property7, "_card_game_scored_rows", spy)
        sampled = score_property7(
            df, cfr_policies={"kuhn": cfr_pol}, games={"kuhn": game}
        )
        exact = property7.score_property7_exact(
            df,
            {("kuhn", "0"): {0: 0.5, 1: 0.5}},
            cfr_policies={"kuhn": cfr_pol},
            games={"kuhn": game},
            include_sampled=False,
        )
        assert seen == [True, False]
        assert sampled["action_belief_no_mass_rows"]["kuhn"] == 1
        assert exact["discarded_exact"]["kuhn"] == {property7._EXIT_NO_MASS: 1}


class TestSilentSevenBExits:
    """
    7b rows that would otherwise leave the scorer with no count, or crash it, are counted in
    their own keys and leave every other number as it is
    """

    @pytest.fixture(scope="class")
    def kuhn_setup(self):
        game = load_kuhn()
        return game, solve_cfr(game, 200)

    def _kuhn_row(self, **overrides):
        row = _row("kuhn", {"1": 0.5, "2": 0.5}, {"1": 0.5, "2": 0.5}, prompt="p")
        row.update(overrides)
        return row

    def test_a_missing_reference_is_recorded_in_a_new_key_and_not_in_not_applicable(self):
        """
        a game with no reference policy keeps its NaN agreement, is not added to
        ``not_applicable``, and is named in ``action_belief_reference_missing``
        """
        result = score_property7(
            pl.DataFrame([self._kuhn_row()]), cfr_policies={}, games={"kuhn": object()}
        )
        assert math.isnan(result["action_belief_agreement"]["kuhn"])
        assert "kuhn" not in result["not_applicable"]
        assert "reference" in result["action_belief_reference_missing"]["kuhn"]

    def test_a_null_parsed_action_is_counted_not_a_crash(self, kuhn_setup):
        """
        a kept row whose parsed_action is null is counted as a named exit in
        ``action_belief_unscored_rows`` and the scored rows' agreement is unchanged
        """
        game, cfr_pol = kuhn_setup
        kw = {"cfr_policies": {"kuhn": cfr_pol}, "games": {"kuhn": game}}
        good = self._kuhn_row()
        bad = pl.DataFrame([good, self._kuhn_row(parsed_action=None)])
        result = score_property7(bad, **kw)
        alone = score_property7(pl.DataFrame([good]), **kw)
        assert result["action_belief_unscored_rows"]["kuhn"] == {"parsed_action is null": 1}
        assert result["action_belief_agreement"] == alone["action_belief_agreement"]
        assert "kuhn" not in result["discarded_rows"]

    def test_a_no_legal_action_status_is_counted(self, kuhn_setup, monkeypatch):
        """
        a row whose best-response status is BR_NO_LEGAL is counted in
        ``action_belief_unscored_rows``, not dropped with only a lower coverage to show it
        """
        from lmopt.scoring import property7 as p7

        game, cfr_pol = kuhn_setup
        monkeypatch.setattr(
            p7, "_br_set_to_belief", lambda *a, **k: (None, p7.BR_NO_LEGAL)
        )
        result = score_property7(
            pl.DataFrame([self._kuhn_row()]),
            cfr_policies={"kuhn": cfr_pol},
            games={"kuhn": game},
        )
        assert result["action_belief_unscored_rows"]["kuhn"] == {
            "the info-state has no legal action": 1
        }
        assert result["action_belief_coverage"]["kuhn"] == 0.0

    def test_parse_belief_array_lets_an_unexpected_error_propagate(self, monkeypatch):
        """
        an error the belief mapping does not handle propagates rather than reading as
        "the belief could not be mapped"
        """
        from lmopt.scoring import property7 as p7

        def boom(*a, **k):
            raise RuntimeError("a programming error")

        monkeypatch.setattr(p7, "_leduc_excluded_indices", boom)
        game = load_leduc()
        state = game.new_initial_state()
        with pytest.raises(RuntimeError, match="programming error"):
            p7._parse_belief_array({"0": 1.0}, game, "leduc", state=state, seat=0)

    @pytest.mark.parametrize("raw", [None, [0.5, 0.5], 0.5])
    def test_parse_belief_array_refuses_a_belief_that_is_not_an_object(self, raw):
        """
        a stated belief that parsed to a non-object reads as unmapped (None), as it did under
        the broad except
        """
        assert _parse_belief_array(raw, load_kuhn(), "kuhn") is None


class TestGoofspielSevenBExits:
    """
    goofspiel 7b rows the generator cannot score are yielded as named sentinels rather than
    dropped with a bare ``continue``, counted in their own key, and leave agreement and
    coverage unchanged
    """

    @pytest.fixture(scope="class")
    def goof(self):
        from lmopt.scoring.goofspiel_p7b import gather_p0_worlds

        game = load_goofspiel(num_cards=4)
        cfr_pol = solve_cfr(game, 50)
        info, states_reach = next(iter(gather_p0_worlds(game, cfr_pol).items()))
        return game, cfr_pol, info, states_reach[0][0].legal_actions()[0]

    def _row(self, info, played, belief='{"0": 0.5, "1": 0.5}', **overrides):
        row = _row("goofspiel", {}, {"0": 0.5, "1": 0.5}, prompt="p", info_set=info, parsed_action=played)
        row["elicited_belief"] = belief
        row.update(overrides)
        return row

    def test_each_silent_exit_is_counted_and_the_scored_rows_are_unchanged(self, goof):
        """
        a bad-JSON belief, a non-object belief, an unreachable info-state and a null action are
        each counted under their own name in ``action_belief_unscored_rows``; the agreement and
        coverage numerators are those of the good row alone
        """
        game, cfr_pol, info, played = goof
        kw = {"cfr_policies": {"goofspiel": cfr_pol}, "games": {"goofspiel": game}}
        good = self._row(info, played)
        rows = [
            good,
            self._row(info, played, belief="{bad"),
            # No posterior, so 7a never reads these two; 7b reads them on the stated belief alone.
            self._row(info, played, belief="[0.5, 0.5]", true_posterior=None),
            self._row(info, played, belief="null", true_posterior=None),
            self._row("no such info-state", played),
            self._row(info, None),
        ]
        result = score_property7(pl.DataFrame(rows), **kw)
        alone = score_property7(pl.DataFrame([good]), **kw)
        assert result["action_belief_unscored_rows"]["goofspiel"] == {
            "elicited_belief is not JSON": 1,
            "elicited_belief is not a JSON object": 2,
            "info_set is unreachable in the reference's worlds": 1,
            "parsed_action is null": 1,
        }
        assert result["action_belief_agreement"] == alone["action_belief_agreement"]
        assert result["action_belief_coverage"]["goofspiel"] == pytest.approx(
            alone["action_belief_coverage"]["goofspiel"] / len(rows)
        )

    def test_an_empty_best_response_is_counted(self, goof, monkeypatch):
        """
        a row whose best-response set comes back empty is counted, not dropped
        """
        from lmopt.scoring import goofspiel_p7b

        game, cfr_pol, info, played = goof
        monkeypatch.setattr(goofspiel_p7b, "goofspiel_br_set", lambda *a, **k: set())
        result = score_property7(
            pl.DataFrame([self._row(info, played)]),
            cfr_policies={"goofspiel": cfr_pol},
            games={"goofspiel": game},
        )
        assert result["action_belief_unscored_rows"]["goofspiel"] == {
            "the best-response set is empty": 1
        }
