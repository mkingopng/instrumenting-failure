"""Tests for the Leduc poker NL prompt builder and response parsers."""

from __future__ import annotations

import json
import re

import pytest

from lmopt.prompting.leduc import (
    action_suffix,
    belief_suffix,
    build_prompt,
    parse_action,
    parse_action_aliased,
    parse_belief,
    parse_own_card,
)

# Canonical Leduc info-state strings (letter-code format, used in unit tests):
# Round 1, Player 0 to act first (pot=2, money=19 19, no actions yet)
_R1_P0_FIRST = "[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 19 19][Round1: ][Round2: ]"
# Round 1, Player 1 facing a bet (Player 0 bet 'b')
_R1_P1_FACING_BET = "[Observer: 1][Private: 1][Round 1][Player: 1][Pot: 4][Money: 18 19][Round1: b][Round2: ]"
# Round 1, Player 0 facing a raise (sequence 'br' = bet-raise)
_R1_P0_FACING_RAISE = "[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 6][Money: 18 17][Round1: br][Round2: ]"
# Round 2, Player 0 first to act. Private idx 2 -> rank 1 = Queen. Public idx 0 -> rank 0 = Jack.
_R2_P0_FIRST = "[Observer: 0][Private: 2][Round 2][Player: 0][Pot: 4][Money: 18 18][Public: 0][Round1: pp][Round2: ]"
# Round 1, Player 0 with a high card index (4 = King, rank 2) -- regression fixture
# for a raw-index lookup into the 3-entry CARD_NAMES dict, which renders idx 3/4/5 as "Unknown".
_R1_P0_KING_HIGH_IDX = "[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 19 19][Round1: ][Round2: ]"
# Round 2: private idx 0 (Jack, rank 0), public idx 1 (the second Jack, also rank 0) --
# a legitimate deal (two distinct physical cards of the same rank), unlike
# _INT_R2_P1_FACING_BET below (Private and Public share the same raw index, which no
# real deal produces). Discriminates the public-card rank lookup: a raw-index
# lookup CARD_NAMES.get(1) = "Queen", which is wrong -- idx 1 is a Jack (rank 0).
_R2_PUBLIC_SECOND_JACK = "[Observer: 0][Private: 0][Round 2][Player: 0][Pot: 4][Money: 18 18][Public: 1][Round1: pp][Round2: ]"
# Malformed info-state with no Private field at all -- regression fixture for the
# no-silent-fallbacks rule (build_prompt/belief_suffix must raise, not default to "Jack").
_MALFORMED_NO_PRIVATE = (
    "[Observer: 0][Round 1][Player: 0][Pot: 2][Money: 19 19][Round1: ][Round2: ]"
)

# Integer-code format: actual OpenSpiel info-state strings from the game engine.
# Action codes: 0=Fold, 1=Call/Check, 2=Bet/Raise (space-separated in sequence fields).
# P0 free bet (empty Round1 sequence - identical to letter format)
_INT_R1_P0_FIRST = "[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]"
# P0 bet (2), P1 raised (2) -> P0 facing raise, only Fold/Call legal (max bets reached)
_INT_R1_P0_FACING_RAISE_NO_RERAISE = "[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 8][Money: 97 95][Round1: 2 2][Round2: ]"
# P0 checked (1), P1 bet (2) -> P0 facing one bet, can fold/call/raise
_INT_R1_P0_FACING_BET_CAN_RAISE = "[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 6][Money: 98 96][Round1: 1 2][Round2: ]"
# Round 2, P1 facing a bet from P0 - regression fixture for _parse_round bug.
# Round 1 ended with both players checking (1 1); Round 2: P0 bet (2), P1 to act.
_INT_R2_P1_FACING_BET = "[Observer: 1][Private: 3][Round 2][Player: 1][Pot: 6][Money: 95 99][Public: 3][Round1: 1 1][Round2: 2]"
# Round 2, P0 first to act, Round 1 ended 1-1 (check-check) -- regression fixture
# for the cross-round betting-history defect. Round2's own sequence is
# empty ("[Round2: ]"), matching the real OpenSpiel format
# (`pyspiel.load_game("leduc_poker")` at a round-2-start node always emits both
# "Round1"/"Round2" fields, the latter present-but-empty) -- not the shorter
# "[Round1: 1 1][Round 2]" form, which omits the Round2 field entirely. The
# difference does not matter: `_parse_field` returns None for a genuinely
# absent field and "" for a present-but-empty one, and both are normalised to
# "" before use, so the two forms exercise the identical code path.
_INT_R2_P0_FIRST_AFTER_R1_HISTORY = "[Observer: 0][Private: 0][Round 2][Player: 0][Pot: 4][Money: 97 97][Public: 2][Round1: 1 1][Round2: ]"
# Round 2, P1 facing a bet, where Round 1 closed having already used its two
# raises ("2 2 1" = bet, raise, call) -- discriminates a raise-availability
# regression that _facing_bet alone cannot: Round 1's sequence is not
# "facing" once closed (its last token is a call), so a bug that reads
# facing-ness from the wrong round would not show up here, but a bug that
# derives _raise_available from Round 1's raise count instead of Round 2's
# own (fresh) sequence "2" would wrongly suppress RAISE in Round 2.
_INT_R2_P1_FACING_BET_AFTER_R1_TWO_RAISES = "[Observer: 1][Private: 3][Round 2][Player: 1][Pot: 12][Money: 93 95][Public: 4][Round1: 2 2 1][Round2: 2]"


class TestParseOwnCard:
    def test_extracts_private_field(self):
        assert parse_own_card(_R1_P0_FIRST) == 2
        assert parse_own_card(_R1_P1_FACING_BET) == 1
        assert parse_own_card(_R1_P0_FACING_RAISE) == 0

    def test_round2(self):
        assert parse_own_card(_R2_P0_FIRST) == 2


class TestBuildPrompt:
    def test_returns_non_empty(self):
        prompt = build_prompt(_R1_P0_FIRST, 0)
        assert isinstance(prompt, str) and len(prompt) > 100

    def test_states_own_card_by_rank(self):
        assert "private card: Queen" in build_prompt(
            _R1_P0_FIRST, 0
        )  # idx 2 -> rank 1 = Queen
        assert "private card: Jack" in build_prompt(
            _R1_P1_FACING_BET, 1
        )  # idx 1 -> rank 0 = Jack
        assert "private card: Jack" in build_prompt(
            _R1_P0_FACING_RAISE, 0
        )  # idx 0 -> rank 0 = Jack

    def test_high_card_index_is_not_unknown(self):
        prompt = build_prompt(_R1_P0_KING_HIGH_IDX, 0)
        assert "private card: King" in prompt
        assert "Unknown" not in prompt

    def test_community_card_named_by_rank(self):
        prompt = build_prompt(_R2_PUBLIC_SECOND_JACK, 0)
        assert "community card: Jack" in prompt

    def test_player_number(self):
        assert "Player 0" in build_prompt(_R1_P0_FIRST, 0)
        assert "Player 1" in build_prompt(_R1_P1_FACING_BET, 1)

    def test_round1_mentions_no_community_card(self):
        prompt = build_prompt(_R1_P0_FIRST, 0)
        assert "community" not in prompt.lower() or "Round 1" in prompt

    def test_round2_mentions_community_card(self):
        prompt = build_prompt(_R2_P0_FIRST, 0)
        assert "community card: Jack" in prompt  # public idx 0 -> rank 0 = Jack

    def test_facing_bet_mentions_fold_call(self):
        prompt = build_prompt(_R1_P1_FACING_BET, 1)
        assert "FOLD" in prompt or "CALL" in prompt

    def test_free_action_mentions_check_bet(self):
        prompt = build_prompt(_R1_P0_FIRST, 0)
        assert "CHECK" in prompt or "BET" in prompt

    def test_unparseable_private_field_raises(self):
        with pytest.raises(ValueError):
            build_prompt(_MALFORMED_NO_PRIVATE, 0)


class TestBuildPromptCrossRoundHistory:
    """build_prompt describes every round's sequence, not only the current
    round's: a current-round-only render hides at a Round-2 node a full
    Round-1 betting sequence that the info-set string plainly carries (19,211
    of 57,408 real rows / 33.5% have that shape)."""

    def test_round2_renders_round1_actions(self):
        prompt = build_prompt(_INT_R2_P0_FIRST_AFTER_R1_HISTORY, 0)
        assert prompt.count("checked/called") == 2  # Round-1 "1 1": both players

    def test_round2_no_history_yet_does_not_use_the_stale_unqualified_phrase(self):
        prompt = build_prompt(_INT_R2_P0_FIRST_AFTER_R1_HISTORY, 0)
        assert "No actions yet this round" not in prompt

    def test_round2_facing_bet_also_renders_round1_actions(self):
        """Same defect on the _INT_R2_P1_FACING_BET regression
        fixture (Round1: "1 1", Round2: "2")."""
        prompt = build_prompt(_INT_R2_P1_FACING_BET, 1)
        assert prompt.count("checked/called") == 2  # Round-1 "1 1"
        assert "bet/raised" in prompt  # Round-2 "2" (P0's bet) still shown

    def test_round1_history_rendering_is_unaffected(self):
        """Round-1 nodes have only one round to describe; the cross-round
        fix must not change their rendering."""
        prompt = build_prompt(_R1_P0_FIRST, 0)
        history_line = next(
            ln for ln in prompt.splitlines() if ln.startswith("- History")
        )
        assert history_line == "- History: No actions yet this round."

    def test_round2_action_legality_is_unaffected(self):
        """The cross-round history render must not change which actions are
        offered -- action legality still comes only from
        `_current_round_seq(info_str, current_round)` (Round 2's own
        sequence "2"), never from Round 1's closed sequence. Round 1 here
        ("2 2 1") already used both its raises, but that must not leak into
        Round 2's fresh raise count."""
        prompt = build_prompt(_INT_R2_P1_FACING_BET_AFTER_R1_TWO_RAISES, 1)
        assert "FOLD" in prompt and "CALL" in prompt and "RAISE" in prompt


class TestActionSuffix:
    def test_free_action(self):
        sfx = action_suffix(_R1_P0_FIRST)
        assert "CHECK" in sfx and "BET" in sfx

    def test_facing_bet_with_raise_available(self):
        sfx = action_suffix(_R1_P1_FACING_BET)
        assert "FOLD" in sfx and "CALL" in sfx and "RAISE" in sfx

    def test_facing_bet_suffix_discourages_bet(self):
        # Hardened suffix must tell the model not to say BET.
        sfx = action_suffix(_R1_P1_FACING_BET)
        assert "do NOT say BET" in sfx

    def test_facing_raise_suffix_discourages_bet(self):
        sfx = action_suffix(_R1_P0_FACING_RAISE)
        assert "do NOT say BET" in sfx

    def test_facing_raise_no_more_raise(self):
        # 'br' = bet then raise -> next player cannot raise again
        sfx = action_suffix(_R1_P0_FACING_RAISE)
        assert "FOLD" in sfx and "CALL" in sfx
        assert "RAISE" not in sfx

    # --- Integer-code format (actual OpenSpiel sequences) ---

    def test_int_free_node_suffix(self):
        # Empty sequence: free bet node, prompt should offer CHECK or BET.
        sfx = action_suffix(_INT_R1_P0_FIRST)
        assert "CHECK" in sfx and "BET" in sfx

    def test_int_facing_bet_can_raise_suffix(self):
        # "1 2": P0 checked, P1 bet -> P0 facing one bet with raise still available.
        sfx = action_suffix(_INT_R1_P0_FACING_BET_CAN_RAISE)
        assert "FOLD" in sfx and "CALL" in sfx and "RAISE" in sfx
        assert "do NOT say BET" in sfx

    def test_int_facing_raise_no_reraise_suffix(self):
        # "2 2": max bets reached -> only FOLD or CALL legal.
        sfx = action_suffix(_INT_R1_P0_FACING_RAISE_NO_RERAISE)
        assert "FOLD" in sfx and "CALL" in sfx
        assert "RAISE" not in sfx
        assert "do NOT say BET" in sfx

    def test_int_round2_facing_bet_suffix(self):
        # Regression: a "[Round: N]" (colon) pattern returns 1 for all states
        # because OpenSpiel emits "[Round N]" (space).
        # At round-2 facing-bet nodes, the suffix must say FOLD/CALL/RAISE, not BET.
        sfx = action_suffix(_INT_R2_P1_FACING_BET)
        assert "FOLD" in sfx and "CALL" in sfx and "RAISE" in sfx
        assert "do NOT say BET" in sfx


class TestParseAction:
    def test_free_action_check(self):
        assert parse_action("CHECK", _R1_P0_FIRST) == 1
        assert parse_action("CHECK", _R1_P0_FIRST) == 1
        assert parse_action("PASS", _R1_P0_FIRST) == 1

    def test_free_action_bet(self):
        assert parse_action("BET", _R1_P0_FIRST) == 2
        assert parse_action("BET", _R1_P0_FIRST) == 2

    def test_facing_bet_fold(self):
        assert parse_action("FOLD", _R1_P1_FACING_BET) == 0

    def test_facing_bet_call(self):
        assert parse_action("CALL", _R1_P1_FACING_BET) == 1

    def test_facing_bet_raise(self):
        assert parse_action("RAISE", _R1_P1_FACING_BET) == 2

    def test_facing_bet_bet_aliased_to_raise(self):
        # Model outputs "BET" at facing-bet nodes; must parse as RAISE (action 2).
        assert parse_action("BET", _R1_P1_FACING_BET) == 2
        assert parse_action("BET", _R1_P0_FACING_RAISE) == 2

    def test_free_action_bet_still_works(self):
        # "BET" at a free node (not facing) should still parse as action 2.
        assert parse_action("BET", _R1_P0_FIRST) == 2

    # --- Integer-code format (actual OpenSpiel sequences) ---

    def test_int_free_bet_parses(self):
        # Empty sequence: "BET" -> action 2 (bet).
        assert parse_action("BET", _INT_R1_P0_FIRST) == 2
        assert parse_action("CHECK", _INT_R1_P0_FIRST) == 1

    def test_int_facing_bet_fold_call_raise(self):
        # "1 2": facing one bet, raise still available.
        assert parse_action("FOLD", _INT_R1_P0_FACING_BET_CAN_RAISE) == 0
        assert parse_action("CALL", _INT_R1_P0_FACING_BET_CAN_RAISE) == 1
        assert parse_action("RAISE", _INT_R1_P0_FACING_BET_CAN_RAISE) == 2
        assert parse_action("BET", _INT_R1_P0_FACING_BET_CAN_RAISE) == 2

    def test_int_facing_raise_fold_call(self):
        # "2 2": max bets - facing=True, can_raise=False. FOLD or CALL only.
        assert parse_action("FOLD", _INT_R1_P0_FACING_RAISE_NO_RERAISE) == 0
        assert parse_action("CALL", _INT_R1_P0_FACING_RAISE_NO_RERAISE) == 1
        # "BET" at a no-raise node still hits the BET alias -> returns 2.
        # Whether action 2 is legal is enforced by the harness, not the parser.
        assert parse_action("BET", _INT_R1_P0_FACING_RAISE_NO_RERAISE) == 2

    def test_int_round2_facing_bet_parse(self):
        # Regression: CALL must parse as 1 at a round-2 facing-bet node.
        # If _parse_round returned 1 here, _facing_bet would check round-1 seq
        # "1 1" (not facing) instead of "2" and CALL would parse as None.
        assert parse_action("CALL", _INT_R2_P1_FACING_BET) == 1
        assert parse_action("FOLD", _INT_R2_P1_FACING_BET) == 0
        assert parse_action("RAISE", _INT_R2_P1_FACING_BET) == 2
        assert parse_action("BET", _INT_R2_P1_FACING_BET) == 2  # alias still works

    def test_bare_digit_fallback(self):
        assert parse_action("0", _R1_P1_FACING_BET) == 0
        assert parse_action("1", _R1_P1_FACING_BET) == 1
        assert parse_action("2", _R1_P0_FIRST) == 2

    def test_unparseable_returns_none(self):
        assert parse_action("xyz", _R1_P0_FIRST) is None
        assert parse_action("", _R1_P1_FACING_BET) is None

    def test_case_insensitive(self):
        assert parse_action("FOLD", _R1_P1_FACING_BET) == 0
        assert parse_action("CALL", _R1_P1_FACING_BET) == 1


class TestBeliefSuffix:
    def test_not_none(self):
        sfx = belief_suffix(_R1_P0_FIRST)
        assert sfx is not None and isinstance(sfx, str)

    def test_contains_card_names(self):
        """CHARACTERISATION: `belief_suffix` names all
        three ranks in the deck-composition prose unconditionally (iterates
        `range(3)` regardless of any card's count), so this stays green under
        a 2-key JSON-example mutation or a deleted-deck-prose mutation alike
        -- verified directly, neither reddens it, because the surviving half
        of the text still carries all three names either way. The genuine
        regression guards for those two mutations already exist and do
        redden: `test_json_example_has_three_placeholder_keys` (JSON example
        key set) and `test_states_per_rank_deck_composition` (deck-prose
        counts). This test is not a further guard beyond those two; it is a
        basic sanity check that the suffix mentions all three card names
        somewhere."""
        sfx = belief_suffix(_R1_P0_FIRST)  # private idx 2 -> rank 1 = Queen
        assert "Jack" in sfx and "Queen" in sfx and "King" in sfx

    def test_contains_json_example(self):
        sfx = belief_suffix(_R1_P0_FIRST)
        assert "{" in sfx and "}" in sfx

    def test_json_example_has_three_placeholder_keys(self):
        """Echo safety: the JSON example's keys are unmappable placeholder
        tokens (`<rank_A>`/`<rank_B>`/`<rank_C>`, LEDUC_SPEC.belief_exemplar,
        spec.py), not the three real rank names, so an echo of
        belief_suffix's text cannot parse as a valid belief (real rank-name
        keys would let it parse straight through `_CARD_FROM_NAME`). There
        are still exactly 3 keys (not 2, not 4) -- mutation: editing
        `LEDUC_SPEC.belief_exemplar` to drop a key reddens this; editing it
        to real rank names reddens the second assertion.
        """
        sfx = belief_suffix(_R1_P0_FIRST)
        example_match = re.search(r"\{[^{}]*\}", sfx)
        assert example_match is not None
        example = json.loads(example_match.group())
        assert len(example) == 3, f"expected 3 keys, got {list(example)}"
        assert not (
            {"Jack", "Queen", "King"} & set(example)
        ), f"exemplar keys must not be real rank names: {list(example)}"

    def test_states_per_rank_deck_composition(self):
        """Re-derived worked example: a Queen holder (idx 2, rank 1) with no
        public card faces 5 unseen cards -- 2 Jacks, 1 Queen (the second copy),
        2 Kings -- true marginals 0.4/0.2/0.4."""
        sfx = belief_suffix(_R1_P0_FIRST)
        assert "2 Jacks" in sfx
        assert "1 Queen" in sfx
        assert "2 Kings" in sfx

    def test_worked_example_is_illustrative_not_the_deck_prior(self):
        """The JSON example must not equal _R1_P0_FIRST's true deck fractions
        (2 Jacks + 1 Queen + 2 Kings of 5 unseen -> 0.4/0.2/0.4): at 6 of 468
        P0 info-sets (Round 1, first to act, no evidence yet) a computed
        "example" would be the correct answer, so a model copying it would
        score perfectly without reasoning.

        The exemplar's keys are placeholder tokens
        (`<rank_A>`/`<rank_B>`/`<rank_C>`), so the check is on the values (in
        the same A/B/C = Jack/Queen/King position order
        LEDUC_SPEC.belief_exemplar's comment documents): the numbers are a
        fixed illustration, not the real per-info-set deck prior.
        "Never equals the true deck-conditional prior" holds by
        construction (placeholder keys are not a distribution over real keys
        at all, so no permutation of them can coincide with any real prior)
        rather than merely by the specific chosen values -- both properties
        are asserted below so this stays a real regression guard, not just a
        by-construction tautology: mutation, setting the values to the true
        deck fractions (0.4, 0.2, 0.4) while keeping placeholder keys, still
        reddens the values-list assertion.
        """
        sfx = belief_suffix(_R1_P0_FIRST)  # true deck fractions here: 0.4/0.2/0.4
        example_match = re.search(r"\{[^{}]*\}", sfx)
        example = json.loads(example_match.group())
        assert not (
            {"Jack", "Queen", "King"} & set(example)
        ), f"exemplar keys must not be real rank names: {list(example)}"
        values = list(example.values())
        assert values == [0.5, 0.3, 0.2], f"expected [0.5, 0.3, 0.2], got {values}"
        # Documentation, not an independent check: [0.5, 0.3, 0.2] != (0.4, 0.2, 0.4) is already IMPLIED by the
        # equality assert immediately above -- kept as an explicit statement
        # of the property this test exists to name ("the illustration is not
        # the true deck prior"), not because it can ever fail independently
        # of that assert.
        assert tuple(values) != (0.4, 0.2, 0.4)
        assert "illustrative" in sfx.lower() or "not a suggested answer" in sfx.lower()

    def test_deck_composition_accounts_for_public_card_in_round2(self):
        """Round 2 removes a 4th card (the public card) from the unseen pool.
        _R2_P0_FIRST: private idx 2 (Queen), public idx 0 (Jack) -> 4 unseen:
        1 Jack, 1 Queen, 2 Kings."""
        sfx = belief_suffix(_R2_P0_FIRST)
        assert "1 Jack" in sfx
        assert "1 Queen" in sfx
        assert "2 Kings" in sfx

    def test_unparseable_private_field_raises(self):
        with pytest.raises(ValueError):
            belief_suffix(_MALFORMED_NO_PRIVATE)


class TestParseBelief:
    def test_card_name_keys(self):
        """CHARACTERISATION: this does not guard against zeroing the own
        rank. Introducing `probs[own_card] = 0.0` (own_card=1, the raw
        Private index) on this fixture zeroes key "1",
        which the JSON input ('{"Jack": 0.3, "King": 0.7}') never mentions and
        which therefore already defaults to 0.0 -- the mutation is a no-op
        here and the test stays green under it. The real regression guard for
        that defect is `test_own_rank_not_zeroed` below (own=Queen, mentioned
        in the input at a genuinely nonzero 0.2), which does redden under the
        same mutation. This test instead documents the basic rank-keyed
        output contract: own_card index=1 maps to rank 0 (Jack) via `_rank`,
        and the returned dict always carries all three rank keys "0"/"1"/"2"
        (`_CARD_FROM_NAME`-mediated name->index mapping, not itself under
        test by any mutation named here)."""
        result = parse_belief(
            '{"Jack": 0.3, "King": 0.7}', _R1_P1_FACING_BET
        )  # own=Jack (idx 1, rank 0)
        assert result is not None
        d = json.loads(result)
        assert abs(sum(d.values()) - 1.0) < 1e-6
        assert d["0"] == pytest.approx(0.3)  # own rank (Jack), not zeroed
        assert d["1"] == pytest.approx(0.0)  # Queen never mentioned in input -> 0.0
        assert d["2"] == pytest.approx(0.7)

    def test_normalises_to_one(self):
        result = parse_belief('{"Jack": 2, "King": 6}', _R1_P1_FACING_BET)
        assert result is not None
        d = json.loads(result)
        assert abs(sum(d.values()) - 1.0) < 1e-6

    def test_own_rank_not_zeroed(self):
        result = parse_belief(
            '{"Jack": 0.5, "Queen": 0.2, "King": 0.3}', _R1_P0_FIRST
        )  # own=Queen (idx 2)
        assert result is not None
        d = json.loads(result)
        assert d["1"] == pytest.approx(0.2)  # own rank (Queen), not zeroed

    def test_all_zero_probabilities_falls_back_to_uniform_thirds(self):
        result = parse_belief(
            '{"Jack": 0, "Queen": 0, "King": 0}', _R1_P0_FACING_RAISE
        )  # own=Jack (idx 0)
        assert result is not None
        d = json.loads(result)
        assert d["0"] == pytest.approx(1 / 3)
        assert d["1"] == pytest.approx(1 / 3)
        assert d["2"] == pytest.approx(1 / 3)

    def test_all_three_keys_present(self):
        result = parse_belief('{"Jack": 0.5, "King": 0.5}', _R1_P1_FACING_BET)
        assert result is not None
        d = json.loads(result)
        assert set(d.keys()) == {"0", "1", "2"}

    def test_unparseable_returns_none(self):
        assert parse_belief("I don't know", _R1_P0_FIRST) is None
        assert parse_belief("{invalid}", _R1_P0_FIRST) is None

    def test_all_keys_unmappable_returns_none_not_renormalised(self):
        """REQUIRED sub-behaviour check, sibling of the
        kuhn test of the same name: `parse_belief` must return None when
        every key is unmappable (neither a real rank name nor a digit),
        rather than renormalising over an empty/degenerate partial parse.
        This is what makes LEDUC_SPEC's placeholder-keyed exemplar
        (`<rank_A>`/`<rank_B>`/`<rank_C>`) actually echo-safe."""
        assert (
            parse_belief(
                '{"<rank_A>": 0.5, "<rank_B>": 0.3, "<rank_C>": 0.2}', _R1_P0_FIRST
            )
            is None
        )


class TestParseActionAliased:
    """The BET->RAISE alias is flagged so the normalisation is auditable
    (no silent fallbacks)."""

    def test_bet_at_facing_node_is_aliased(self):
        # Facing a bet, model says "BET" -> parse_action returns 2 (RAISE) via alias.
        assert parse_action("BET", _INT_R1_P0_FACING_BET_CAN_RAISE) == 2
        assert parse_action_aliased("BET", _INT_R1_P0_FACING_BET_CAN_RAISE) is True

    def test_raise_at_facing_node_not_aliased(self):
        # Canonical token "RAISE" -> action 2, no alias.
        assert parse_action("RAISE", _INT_R1_P0_FACING_BET_CAN_RAISE) == 2
        assert parse_action_aliased("RAISE", _INT_R1_P0_FACING_BET_CAN_RAISE) is False

    def test_call_at_facing_node_not_aliased(self):
        assert parse_action("CALL", _INT_R1_P0_FACING_BET_CAN_RAISE) == 1
        assert parse_action_aliased("CALL", _INT_R1_P0_FACING_BET_CAN_RAISE) is False

    def test_bet_at_free_node_not_aliased(self):
        # Free node: "BET" is the canonical token (action 2), not an alias.
        assert parse_action("BET", _INT_R1_P0_FIRST) == 2
        assert parse_action_aliased("BET", _INT_R1_P0_FIRST) is False

    def test_bet_at_round2_facing_node_is_aliased(self):
        # Round-2 facing-bet node (regression fixture): BET -> aliased.
        assert parse_action("BET", _INT_R2_P1_FACING_BET) == 2
        assert parse_action_aliased("BET", _INT_R2_P1_FACING_BET) is True


class TestRefusesMissingStakes:
    _INFO = "[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]"

    @pytest.mark.parametrize(
        "info",
        [
            _INFO.replace("[Pot: 2]", ""),
            _INFO.replace("[Pot: 2]", "[Pot: x]"),
            _INFO.replace("[Money: 99 99]", ""),
            _INFO.replace("[Money: 99 99]", "[Money: 99]"),
            _INFO.replace("[Money: 99 99]", "[Money: 99 x]"),
        ],
    )
    def test_build_prompt_refuses_an_unparseable_pot_or_money_field(self, info):
        """
        a missing or malformed Pot or Money field refuses instead of rendering 0 chips.
        """
        from lmopt.prompting.leduc import build_prompt as leduc_build_prompt

        with pytest.raises(ValueError, match="Pot|Money"):
            leduc_build_prompt(info, 0)
