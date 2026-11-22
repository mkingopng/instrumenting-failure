"""Tests for the Kuhn poker NL prompt builder and response parsers."""

from __future__ import annotations

import json
import re

import pytest

from lmopt.prompting.kuhn import (
    action_suffix,
    belief_suffix,
    build_prompt,
    parse_action,
    parse_belief,
)

# Canonical Kuhn info sets and their properties:
#   "0"   - Jack, P0 first to act            -> check/bet
#   "1b"  - Queen, facing P0's bet           -> fold/call
#   "0p"  - Jack (as P1), after P0 checked   -> check/bet
#   "0pb" - Jack, facing P1's bet after own check -> fold/call
#   "2"   - King, first to act               -> check/bet
#   "2p"  - King (as P1), after P0 checked   -> check/bet


class TestBuildPrompt:
    def test_contains_card_name(self):
        assert "Jack" in build_prompt("0", 0)
        assert "Queen" in build_prompt("1b", 1)
        assert "King" in build_prompt("2", 0)

    def test_contains_player_number(self):
        assert "Player 0" in build_prompt("0", 0)
        assert "Player 1" in build_prompt("1b", 1)

    def test_facing_bet_mentions_fold_call(self):
        prompt = build_prompt("1b", 1)
        assert "FOLD" in prompt or "CALL" in prompt

    def test_free_action_mentions_check_bet(self):
        prompt = build_prompt("0", 0)
        assert "CHECK" in prompt or "BET" in prompt


class TestActionSuffix:
    def test_free_action_suffix(self):
        sfx = action_suffix("0")
        assert "CHECK" in sfx and "BET" in sfx

    def test_facing_bet_suffix(self):
        sfx = action_suffix("1b")
        assert "FOLD" in sfx and "CALL" in sfx


class TestParseAction:
    # Free-action states (no trailing 'b')
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("CHECK", 0),
            ("CHECK", 0),
            ("PASS", 0),
            ("BET", 1),
            ("BET", 1),
            ("I will CHECK here", 0),
            ("My choice is BET", 1),
            ("0", 0),
            ("1", 1),
        ],
    )
    def test_free_action(self, text, expected):
        assert parse_action(text, "0") == expected

    # Facing-bet states (trailing 'b')
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("FOLD", 0),
            ("FOLD", 0),
            ("CALL", 1),
            ("CALL", 1),
            ("I will FOLD", 0),
            ("I CALL", 1),
            ("0", 0),
            ("1", 1),
        ],
    )
    def test_facing_bet(self, text, expected):
        assert parse_action(text, "1b") == expected

    def test_unparseable_returns_none(self):
        assert parse_action("xyz random words", "0") is None
        assert parse_action("", "1b") is None

    def test_does_not_confuse_fold_with_bet(self):
        # "FOLD" should parse to 0 when facing bet, not 1
        assert parse_action("FOLD", "1b") == 0
        # "BET" should parse to 1 when not facing bet
        assert parse_action("BET", "0") == 1


class TestBeliefSuffix:
    def test_own_card_name_stated_but_exemplar_keys_are_placeholders(self):
        """Echo-safety: the JSON exemplar's keys are unmappable placeholder
        tokens (`<other_card_A>`/`<other_card_B>`, KUHN_SPEC.belief_exemplar,
        spec.py), not the two real remaining card names, so an echo of this
        text cannot parse as a valid belief. "Jack"/"King" appearing as
        exemplar keys is exactly the property that must be ABSENT. The model
        still learns all three card names from build_prompt's rules prose,
        concatenated ahead of this suffix in every live prompt
        (`LMVllmAgent.act` sends `base + bel_sfx` together) -- belief_suffix
        itself never needs to re-state them; only an exemplar naming them as
        JSON keys is a defect.

        What this guards: the own card's name is stated (per-info-set
        content) and the exemplar block's keys are the placeholder tokens,
        never real card names.

        CARDINALITY pin: `all(k.startswith("<") for k in example)` alone is
        a null-tolerant aggregate -- `all()` over an empty iterable is
        `True`, so an exemplar that mutated to `{}` (e.g. an
        accidentally-blanked `belief_exemplar`) would pass despite having no
        placeholder content at all. `len(example) == 2` (kuhn's exemplar
        always names exactly the two non-own cards) closes that gap:
        mutating `KUHN_SPEC.belief_exemplar` to `"{}"` reddens exactly that
        assertion (the `all(...)` line stays green under the same mutation).
        """
        sfx = belief_suffix(1)  # own card = Queen
        assert "Your card is Queen" in sfx
        example_match = re.search(r"\{[^{}]*\}", sfx)
        assert example_match is not None
        example = json.loads(example_match.group())
        assert len(example) == 2, f"expected 2 exemplar keys, got {list(example)}"
        assert "Jack" not in example and "King" not in example
        assert all(
            k.startswith("<") for k in example
        ), f"expected placeholder-only exemplar keys, got {list(example)}"

    def test_contains_json_example(self):
        sfx = belief_suffix(0)
        assert "{" in sfx and "}" in sfx


class TestParseBelief:
    def test_card_name_keys(self):
        result = parse_belief('{"Jack": 0.3, "King": 0.7}', 1)
        assert result is not None
        d = json.loads(result)
        assert abs(sum(d.values()) - 1.0) < 1e-6
        assert d["1"] == pytest.approx(0.0)  # own card (Queen) = 0

    def test_numeric_keys(self):
        result = parse_belief('{"0": 0.4, "2": 0.6}', 1)
        assert result is not None
        d = json.loads(result)
        assert abs(sum(d.values()) - 1.0) < 1e-6

    def test_normalises_to_one(self):
        result = parse_belief('{"Jack": 2, "King": 6}', 1)
        assert result is not None
        d = json.loads(result)
        assert abs(sum(d.values()) - 1.0) < 1e-6

    def test_own_card_zeroed(self):
        result = parse_belief('{"Jack": 0.3, "Queen": 0.2, "King": 0.5}', 1)
        assert result is not None
        d = json.loads(result)
        assert d["1"] == pytest.approx(0.0)

    def test_all_three_keys_present(self):
        result = parse_belief('{"Jack": 0.5, "King": 0.5}', 1)
        assert result is not None
        d = json.loads(result)
        assert set(d.keys()) == {"0", "1", "2"}

    def test_unparseable_returns_none(self):
        assert parse_belief("I don't know", 0) is None
        assert parse_belief("no json here", 1) is None
        assert parse_belief("{invalid}", 0) is None

    def test_all_keys_unmappable_returns_none_not_renormalised(self):
        """REQUIRED sub-behaviour check: placeholder-keyed belief exemplars
        are echo-safe only if `parse_belief` returns None when every key
        is unmappable, rather than silently skipping the unmapped keys and
        renormalising over an empty (or degenerate) partial parse.
        """
        assert parse_belief('{"<foo>": 0.5, "<bar>": 0.5}', 0) is None

    def test_embedded_in_text(self):
        text = (
            'I think the probabilities are {"Jack": 0.6, "King": 0.4} based on history.'
        )
        result = parse_belief(text, 1)
        assert result is not None


class _ParamGame:
    """a stand-in pyspiel game exposing only the two calls the dispatch reads"""

    def __init__(self, short_name: str, params: dict):
        self._short_name = short_name
        self._params = params

    def get_type(self):
        """
        the game type, carrying only short_name
        :returns: an object with a short_name attribute (object)
        """
        return type("T", (), {"short_name": self._short_name})()

    def get_parameters(self) -> dict:
        """
        the game parameters
        :returns: the parameter dict (dict)
        """
        return dict(self._params)


@pytest.mark.parametrize(
    "short_name, params, missing",
    [
        ("liars_dice", {"numdice": 1}, "dice_sides"),
        ("universal_poker", {"numRanks": 13}, "numRounds"),
    ],
)
def test_dispatch_refuses_a_game_missing_its_dimension_parameter(
    short_name, params, missing
):
    """
    a game whose parameters omit the dimension the dispatch reads refuses rather than
    defaulting to the canonical value.
    """
    from lmopt.prompting import get_game_fns_for_game

    with pytest.raises(KeyError, match=missing):
        get_game_fns_for_game(_ParamGame(short_name, params))
