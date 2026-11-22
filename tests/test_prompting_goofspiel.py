"""Tests for the Goofspiel NL prompt builder and response parsers."""

from __future__ import annotations

import json


from lmopt.prompting.goofspiel import (
    action_suffix,
    belief_suffix,
    build_prompt,
    parse_action,
    parse_belief,
    parse_own_card,
)

# Goofspiel info-state strings (num_cards=4, player 0's view):
# Round 1, no prior bids, point card 4 is up
_G_START = (
    "P0 hand: 1 2 3 4\n"
    "P0 action sequence:\n"
    "Point card sequence: 4\n"
    "Win sequence:\n"
    "Points: 0 0"
)

# Round 2, P0 bid card 3 in round 1 (won that round), point card 3 is up
_G_ROUND2 = (
    "P0 hand: 1 2 4\n"
    "P0 action sequence: 3\n"
    "Point card sequence: 4 3\n"
    "Win sequence: 0\n"
    "Points: 4 0"
)

# Player 1's view (to check it works regardless of player index)
_G_P1_VIEW = (
    "P1 hand: 2 3 4\n"
    "P1 action sequence: 1\n"
    "Point card sequence: 4 3\n"
    "Win sequence: 1\n"
    "Points: 0 4"
)


class TestParseOwnCard:
    def test_returns_none(self):
        # Goofspiel has no fixed private card
        assert parse_own_card(_G_START) is None
        assert parse_own_card(_G_ROUND2) is None


class TestBuildPrompt:
    def test_returns_non_empty(self):
        prompt = build_prompt(_G_START, 0)
        assert isinstance(prompt, str) and len(prompt) > 50

    def test_lists_all_hand_cards_at_start(self):
        prompt = build_prompt(_G_START, 0)
        assert "1" in prompt and "2" in prompt and "3" in prompt and "4" in prompt

    def test_mentions_point_card(self):
        prompt = build_prompt(_G_START, 0)
        assert "4" in prompt  # current point card is 4

    def test_names_current_prize_not_next(self):
        # Regression for the off-by-one prize bug: the current prize is the last
        # card in the point sequence, not _next_point_card.
        p = build_prompt(_G_START, 0)  # seq "4" -> prize 4
        assert "bid for the 4-point card" in p
        assert "bid for the 3-point card" not in p
        p2 = build_prompt(_G_ROUND2, 0)  # seq "4 3" -> prize 3
        assert "bid for the 3-point card" in p2
        assert "bid for the 2-point card" not in p2

    def test_player_number(self):
        prompt = build_prompt(_G_START, 0)
        assert "Player 0" in prompt

    def test_round2_fewer_cards(self):
        prompt = build_prompt(_G_ROUND2, 0)
        # Card 3 was bid, so hand is now 1 2 4
        assert "3" not in prompt.split("hand")[0] or "1" in prompt

    def test_p1_view(self):
        prompt = build_prompt(_G_P1_VIEW, 1)
        assert "Player 1" in prompt


class TestActionSuffix:
    def test_lists_available_cards(self):
        sfx = action_suffix(_G_START)
        assert "1" in sfx and "2" in sfx and "3" in sfx and "4" in sfx

    def test_round2_missing_bid_card(self):
        sfx = action_suffix(_G_ROUND2)
        # Card 3 used, remaining: 1 2 4
        assert "3" not in sfx
        assert "1" in sfx and "2" in sfx and "4" in sfx


class TestParseAction:
    def test_card_to_action_index(self):
        # Hand has cards 1 2 3 4 -> card 2 = action 1
        assert parse_action("2", _G_START) == 1
        assert parse_action("4", _G_START) == 3
        assert parse_action("1", _G_START) == 0

    def test_only_cards_in_hand_valid(self):
        # Hand is 1 2 4 (card 3 used); bidding 3 should return None
        assert parse_action("3", _G_ROUND2) is None

    def test_text_with_card_number(self):
        # Agent says "I'll bid 3" - parse out the 3
        assert parse_action("I'll bid 3", _G_START) == 2

    def test_ambiguous_large_number_ignored(self):
        # 5 is not in the hand -> None
        assert parse_action("5", _G_START) is None

    def test_unparseable_returns_none(self):
        assert parse_action("abc", _G_START) is None
        assert parse_action("", _G_START) is None


class TestBeliefSuffix:
    def test_elicits_opponent_bid_distribution(self):
        # Belief elicitation is enabled and mentions the next prize.
        s = belief_suffix(_G_START)
        assert s is not None
        assert "opponent" in s.lower()
        assert "4-point card" in s  # next prize at round 1
        assert "1" in s and "4" in s  # cards 1-4


class TestParseBelief:
    def test_parses_and_rekeys_card_to_action_index(self):
        # LM answers in card values 1-4; parse emits 0-indexed action keys (card-1)
        # so the belief aligns with true_posterior.
        out = parse_belief('{"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4}', _G_START)
        assert out is not None
        d = json.loads(out)
        assert set(d.keys()) == {"0", "1", "2", "3"}  # action indices
        assert abs(d["0"] - 0.1) < 1e-9 and abs(d["3"] - 0.4) < 1e-9
        assert abs(sum(d.values()) - 1.0) < 1e-9

    def test_normalises_unnormalised_input(self):
        out = parse_belief('{"1": 1, "2": 1, "3": 1, "4": 1}', _G_START)
        d = json.loads(out)
        assert all(abs(v - 0.25) < 1e-9 for v in d.values())

    def test_missing_cards_default_zero(self):
        out = parse_belief('{"3": 0.5, "4": 0.5}', _G_ROUND2)
        d = json.loads(out)
        assert abs(d["0"]) < 1e-9 and abs(d["1"]) < 1e-9  # cards 1,2 absent -> 0
        assert abs(d["2"] - 0.5) < 1e-9 and abs(d["3"] - 0.5) < 1e-9

    def test_unparseable_returns_none(self):
        assert parse_belief("no json here", _G_START) is None
        assert parse_belief('{"9": 1.0}', _G_START) is None  # out-of-range only

    def test_all_keys_unmappable_returns_none_not_renormalised(self):
        """REQUIRED sub-behaviour check, sibling of the
        kuhn/leduc tests of the same name: `parse_belief` must return None
        when every key fails `int()` (not merely out-of-range, which
        `test_unparseable_returns_none` above already covers with `"9"`),
        rather than renormalising over an empty partial parse. This is what
        makes GOOFSPIEL_SPEC's placeholder-keyed exemplar
        (`<card_A>`/`<card_B>`) actually echo-safe."""
        assert parse_belief('{"<card_A>": 0.25, "<card_B>": 0.25}', _G_START) is None


class TestRefusesMalformedInfoState:
    def test_build_prompt_refuses_an_empty_point_card_sequence(self):
        """
        an info state with no point card up renders no "None-point card" prompt; it refuses.
        """
        import pytest

        info = _G_START.replace("Point card sequence: 4", "Point card sequence:")
        with pytest.raises(ValueError, match="point card"):
            build_prompt(info, 0)

    def test_make_fns_refuses_a_deck_above_nine_cards(self):
        """
        a ten-card deck refuses, because the canonical parser reads single digits and could
        never parse card 10.
        """
        import pytest

        from lmopt.prompting.goofspiel import make_fns

        with pytest.raises(ValueError, match="num_cards"):
            make_fns(10)

    def test_points_refuse_a_missing_points_line(self):
        """
        an info state with no Points line refuses instead of rendering both scores as 0.
        """
        import pytest

        from lmopt.prompting.goofspiel import _parse_points

        with pytest.raises(ValueError, match="Points"):
            _parse_points(_G_START.replace("Points: 0 0", ""))

    def test_hand_refuses_a_missing_hand_line_and_reads_an_empty_hand_as_empty(self):
        """
        an info state with no hand line refuses; a hand line with no cards is an empty hand.
        """
        import pytest

        from lmopt.prompting.goofspiel import _parse_hand

        with pytest.raises(ValueError, match="hand"):
            _parse_hand(_G_START.replace("P0 hand: 1 2 3 4\n", ""))
        assert _parse_hand(_G_START.replace("1 2 3 4", "")) == []
