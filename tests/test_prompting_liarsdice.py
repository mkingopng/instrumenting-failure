"""Tests for the Liar's Dice prompt builder and response parsers."""

from __future__ import annotations

import json

import pytest

from lmopt.prompting.liarsdice import (
    action_labels,
    action_suffix,
    belief_suffix,
    build_prompt,
    parse_action,
    parse_belief,
    parse_own_card,
    _CHALLENGE_ACTION,
)

# Info-state fixtures.
_FREE = "4"  # own die=4, no bids
_AFTER_ONE = "4 1-3"  # own die=4, one bid: 1×3
_AFTER_TWO = "4 1-3 2-5"  # own die=4, bids: 1×3 then 2×5


class TestParseOwnCard:
    def test_bare_die(self):
        assert parse_own_card("4") == 4

    def test_with_bid_history(self):
        assert parse_own_card("4 1-3") == 4

    def test_die_one(self):
        assert parse_own_card("1 2-6") == 1

    def test_die_six(self):
        assert parse_own_card("6") == 6


class TestBuildPrompt:
    def test_returns_non_empty(self):
        prompt = build_prompt(_FREE, 0)
        assert isinstance(prompt, str) and len(prompt) > 50

    def test_contains_own_die(self):
        prompt = build_prompt(_FREE, 0)
        assert "4" in prompt

    def test_contains_die_word(self):
        prompt = build_prompt(_FREE, 0)
        assert "die" in prompt.lower() or "dice" in prompt.lower()

    def test_no_bids_shows_no_history(self):
        prompt = build_prompt(_FREE, 0)
        assert "No bids yet" in prompt

    def test_bid_history_shown(self):
        prompt = build_prompt(_AFTER_ONE, 0)
        assert "1" in prompt and "3" in prompt

    def test_player_number(self):
        assert "Player 0" in build_prompt(_FREE, 0)
        assert "Player 1" in build_prompt(_FREE, 1)


class TestActionSuffix:
    def test_free_no_challenge(self):
        sfx = action_suffix(_FREE)
        assert "CHALLENGE" not in sfx

    def test_free_contains_opening_bid(self):
        sfx = action_suffix(_FREE)
        assert "1 1" in sfx  # lowest legal bid

    def test_free_contains_highest_bid(self):
        sfx = action_suffix(_FREE)
        assert "2 6" in sfx  # highest legal bid

    def test_after_bid_has_challenge(self):
        sfx = action_suffix(_AFTER_ONE)
        assert "CHALLENGE" in sfx

    def test_after_bid_excludes_lower_bids(self):
        # After 1×3 (action 2), bids 0,1,2 are illegal.
        sfx = action_suffix(_AFTER_ONE)
        assert "1 1" not in sfx
        assert "1 2" not in sfx
        assert "1 3" not in sfx  # exactly equal bid is also illegal

    def test_after_bid_includes_higher_bids(self):
        sfx = action_suffix(_AFTER_ONE)
        assert "1 4" in sfx  # next face up
        assert "2 6" in sfx  # highest

    def test_after_max_bid_only_challenge(self):
        # After 2×6 (action 11, the max), only CHALLENGE remains.
        sfx = action_suffix("3 2-6")
        assert "CHALLENGE" in sfx
        # No bid options higher than 2×6 exist.
        assert "1 1" not in sfx


class TestParseAction:
    def test_challenge_keyword(self):
        assert parse_action("CHALLENGE", _AFTER_ONE) == _CHALLENGE_ACTION

    def test_challenge_lowercase(self):
        assert parse_action("CHALLENGE", _AFTER_ONE) == _CHALLENGE_ACTION

    def test_liar_keyword(self):
        assert parse_action("LIAR", _AFTER_ONE) == _CHALLENGE_ACTION

    def test_bid_two_five(self):
        # "2 5" -> count=2, face=5 -> (2-1)*6 + (5-1) = 10
        assert parse_action("2 5", _FREE) == 10

    def test_bid_one_three(self):
        # "1 3" -> count=1, face=3 -> (1-1)*6 + (3-1) = 2
        assert parse_action("1 3", _FREE) == 2

    def test_bid_one_one(self):
        assert parse_action("1 1", _FREE) == 0

    def test_bid_two_six(self):
        assert parse_action("2 6", _FREE) == 11

    def test_unparseable_returns_none(self):
        assert parse_action("xyz", _FREE) is None

    def test_empty_returns_none(self):
        assert parse_action("", _FREE) is None

    def test_case_insensitive_challenge(self):
        assert parse_action("Challenge please", _AFTER_ONE) == _CHALLENGE_ACTION

    def test_bare_digit_action(self):
        """The bare-digit fallback is
        legality-gated (`a in labels`, liarsdice.py) -- it does not accept
        any syntactically-valid action number regardless of whether that
        action is actually offered at this info-set. Action 12
        (CHALLENGE) is not legal at `_FREE` (no bids yet -- action_labels()
        excludes CHALLENGE until there is something to challenge), so
        `parse_action("12", _FREE)` is None; the digit-12-means-CHALLENGE
        mapping is checked at `_AFTER_ONE`, where a bid
        exists and CHALLENGE genuinely is legal.
        Bare digit 0 is checked at `_FREE`: bid 1x1 (action 0) is the LOWEST
        legal opening bid, so it is legal there."""
        # Bare digit 0 -> bid 1×1 (legal at _FREE: no bids yet, so 1x1 is offered)
        assert parse_action("0", _FREE) == 0
        # Bare digit 12 -> CHALLENGE (legal only once a bid exists to challenge)
        assert parse_action("12", _AFTER_ONE) == 12

    def test_exemplar_literal_does_not_leak_a_bogus_bid_at_max_bid_info_set(self):
        """Action parser hardening -- the specific
        defect for liarsdice's |A|=1 case, with its own
        dedicated test: `self_parse_report`'s standing invariant
        (test_prompt_fidelity.py) deliberately excludes |A|=1 info-sets
        (fidelity.py's `SelfParseReport` docstring explains why -- an echo
        at |A|=1 trivially reproduces the only legal answer, which is
        unavoidable, not a defect), so it structurally cannot catch a
        regression in the other fix this info-set needed: the fixed
        exemplar phrase such as "e.g. '2 5'" in action_suffix's trailing
        sentence would decode to a real (but illegal, since only
        CHALLENGE remains) bid action, regardless of the |A|=1 exclusion.
        This test is the only regression guard for that legality gate.

        Deliberately checked against the exemplar fragment alone, not the
        full `action_suffix` text: the full suffix at this info-set also
        contains the (correct, expected) word "CHALLENGE", so
        `parse_action(action_suffix(max_bid_info), max_bid_info)` legitimately
        returns 12 via the challenge branch before the bid regex is even
        reached -- exactly the |A|=1 self-parse case `self_parse_report`
        deliberately excludes (echoing a single-label suffix trivially
        reproduces the one legal answer). Isolating the exemplar phrase is
        what proves the legality gate specifically, independent of that
        already-documented, unavoidable case.
        """
        max_bid_info = (
            "4 2-6"  # own die 4, one bid: 2x6 (the maximum) -> only CHALLENGE legal
        )
        assert action_labels(max_bid_info) == {12: "CHALLENGE"}
        sfx = action_suffix(max_bid_info)
        # Prompt version 7: at the maximum bid the suffix offers CHALLENGE alone and
        # carries no worked bid at all (version 6 rendered "e.g. '2 5'" here, an
        # instruction to make an impossible bid).
        assert "only legal response right now is: CHALLENGE" in sfx
        assert "for example" not in sfx
        exemplar_fragment = (
            "for example '2 5' claims that at least 2 of the two dice show 5"
        )
        assert parse_action(exemplar_fragment, max_bid_info) is None


class TestBeliefSuffix:
    def test_not_none(self):
        assert belief_suffix(_FREE) is not None

    def test_contains_json_example(self):
        sfx = belief_suffix(_FREE)
        assert "{" in sfx and "}" in sfx

    def test_mentions_own_die(self):
        sfx = belief_suffix(_FREE)
        assert "4" in sfx  # own die value


class TestParseBelief:
    # Input is face-keyed (1–6, as prompted); canonical output is 0-indexed die
    # Action keys (0–5, face f -> key f-1) to align with true_posterior.
    def test_valid_six_key_json(self):
        raw = '{"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.1, "6": 0.1}'
        result = parse_belief(raw, _FREE)
        assert result is not None
        d = json.loads(result)
        assert set(d.keys()) == {"0", "1", "2", "3", "4", "5"}
        assert abs(sum(d.values()) - 1.0) < 1e-6

    def test_normalises_to_one(self):
        raw = '{"1": 2, "2": 2, "3": 2, "4": 2, "5": 1, "6": 1}'
        result = parse_belief(raw, _FREE)
        assert result is not None
        d = json.loads(result)
        assert abs(sum(d.values()) - 1.0) < 1e-6

    def test_all_six_faces_present(self):
        raw = '{"1": 0.5, "3": 0.5}'
        result = parse_belief(raw, _FREE)
        assert result is not None
        d = json.loads(result)
        assert set(d.keys()) == {"0", "1", "2", "3", "4", "5"}

    def test_all_eight_faces_present_for_liarsdice_1d8(self):
        """The module-level `parse_belief` imported at the
        top of this file is `make_fns()`'s default `dice_sides=6` closure --
        it says nothing about `liarsdice_1d8`'s own `dice_sides=8` closure,
        which this test covers. `property5.py`'s
        reasoning (that the kuhn/liarsdice `assert_true_support_
        within_stated` guard can never fire because both `elicited_belief`
        and `true_posterior` are always built over the identical full range)
        rests on this same unconditional-padding property for
        `liarsdice_1d8` too, verified here at the real production
        parametrisation rather than assumed from the `dice_sides=6` case.
        """
        from lmopt.prompting.liarsdice import make_fns

        fns = make_fns(dice_sides=8)
        raw = '{"1": 0.5, "3": 0.5}'  # states only 2 of 8 faces
        result = fns["parse_belief"](raw, _FREE)
        assert result is not None
        d = json.loads(result)
        assert set(d.keys()) == {"0", "1", "2", "3", "4", "5", "6", "7"}
        assert abs(sum(d.values()) - 1.0) < 1e-6

    def test_face_maps_to_zero_indexed_action(self):
        # Face "1" -> action key "0"; all mass on face 1 lands on key "0".
        raw = '{"1": 1.0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}'
        result = parse_belief(raw, _FREE)
        assert result is not None
        d = json.loads(result)
        assert d["0"] == pytest.approx(1.0)
        assert d["5"] == pytest.approx(0.0)  # face 6 -> key 5

    def test_unparseable_returns_none(self):
        assert parse_belief("I don't know", _FREE) is None

    def test_invalid_json_returns_none(self):
        assert parse_belief("{bad json}", _FREE) is None

    def test_all_keys_unmappable_returns_none_not_renormalised(self):
        """REQUIRED sub-behaviour check, sibling of the
        kuhn/leduc/goofspiel tests of the same name: `parse_belief` must
        return None when every key fails `int()`, rather than renormalising
        over an empty partial parse. This is what makes LIARSDICE_SPEC's
        placeholder-keyed exemplar actually echo-safe (a real-keyed exemplar
        gave an echo rate of 0.5607 for pythia-1.4b on liarsdice)."""
        raw = (
            '{"<face_A>": 0.17, "<face_B>": 0.17, "<face_C>": 0.17, '
            '"<face_D>": 0.17, "<face_E>": 0.16, "<face_F>": 0.16}'
        )
        assert parse_belief(raw, _FREE) is None

    def test_no_zeroing_of_own_die(self):
        # Unlike card games, own die does not constrain opponent - no face forced to 0.
        raw = '{"1": 0.17, "2": 0.17, "3": 0.17, "4": 0.17, "5": 0.16, "6": 0.16}'
        result = parse_belief(raw, _FREE)  # own die=4
        assert result is not None
        d = json.loads(result)
        assert d["3"] > 0.0  # face 4 -> action key "3"; not zeroed despite own die 4


@pytest.mark.parametrize("dice_sides", [0, 10, 12])
def test_make_fns_refuses_dice_sides_outside_one_to_nine_with_a_value_error(dice_sides):
    """
    the dice_sides bound is a ValueError, which survives `python -O`, not an assert.
    """
    from lmopt.prompting.liarsdice import make_fns

    with pytest.raises(ValueError, match="dice_sides"):
        make_fns(None, dice_sides)
