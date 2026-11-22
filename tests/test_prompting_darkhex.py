"""Tests for the Dark Hex prompt builder and response parsers."""

from __future__ import annotations

import json
import re

from lmopt.prompting.darkhex import (
    action_suffix,
    belief_suffix,
    build_prompt,
    parse_action,
    parse_belief,
    parse_own_card,
)

# ---------------------------------------------------------------------------
# Info-state fixtures (based on empirically verified encoding)
# ---------------------------------------------------------------------------

# Empty board - P0 perspective, no stones placed
_EMPTY = "..\n..\n0\n"

# P0 has placed at A1 (action 0, top-left); P0's view: 'x' = own stone
_P0_PLAYED_A1 = "x.\n..\n1\n0,0"

# P0 has A1, P1 has B2 (P1's stone hidden from P0), P0 moves next
# P0 still sees '.' at B2 because it's hidden
_P0_A1_P1_B2_P0VIEW = "x.\n..\n2\n0,0 1,3"

# Board where P0 discovers P1's stone at B1 after a collision: 'o' at (0,1)
_P0_DISCOVERS_B1 = "xo\n..\n2\n0,0 0,1"


# ---------------------------------------------------------------------------
# the board and legal moves, read back off the production prompt
# ---------------------------------------------------------------------------

_STONE_OF_CELL = {"[ ]": ".", "[Y]": "x", "[E]": "o"}
_ACTION_OF_LABEL = {"A1": 0, "B1": 1, "A2": 2, "B2": 3}


def _rendered_board(info_state: str) -> list[list[str]]:
    """
    the board the production prompt shows for an info-state, in the info-state's own symbols
    :param info_state: a 2x2 Dark Hex info-state string (str)
    :returns: rows of '.', 'x' (own stone) and 'o' (discovered enemy) (list of list of str)
    """
    rows = re.findall(
        r"^\s+[12]\s+(\[.\])\s+(\[.\])$", build_prompt(info_state, 0), re.M
    )
    return [[_STONE_OF_CELL[cell] for cell in row] for row in rows]


def _rendered_legal(info_state: str) -> list[int]:
    """
    the action indices the production prompt lists under "Available moves"
    :param info_state: a 2x2 Dark Hex info-state string (str)
    :returns: the listed actions, in listed order (list of int)
    """
    line = re.search(r"^Available moves: (.*)$", build_prompt(info_state, 0), re.M)
    assert line, "the prompt has no Available moves line"
    return [
        _ACTION_OF_LABEL[label]
        for label in re.findall(r"\b([AB][12]) \(", line.group(1))
    ]


class TestRenderedBoard:

    def test_empty_board(self):
        assert _rendered_board(_EMPTY) == [[".", "."], [".", "."]]

    def test_p0_own_stone(self):
        board = _rendered_board(_P0_PLAYED_A1)
        assert board[0][0] == "x"  # own stone
        assert board[0][1] == "."  # unknown

    def test_discovered_opponent(self):
        board = _rendered_board(_P0_DISCOVERS_B1)
        assert board[0][0] == "x"  # own
        assert board[0][1] == "o"  # discovered enemy

    def test_hidden_opponent_shows_dot(self):
        # P1 has B2 but it's invisible to P0
        board = _rendered_board(_P0_A1_P1_B2_P0VIEW)
        assert board[1][1] == "."


class TestRenderedLegalMoves:

    def test_all_unknown_is_all_legal(self):
        assert _rendered_legal("..\n..\n0\n") == [0, 1, 2, 3]

    def test_own_stone_excluded(self):
        # 'x' at (0,0) -- action 0 not legal
        assert _rendered_legal("x.\n..\n1\n0,0") == [1, 2, 3]

    def test_discovered_opp_excluded(self):
        # 'o' at (0,1) -- action 1 not legal
        assert _rendered_legal(".o\n..\n1\n0,1") == [0, 2, 3]

    def test_full_board_minus_one_is_one_legal(self):
        assert _rendered_legal("xo\nx.\n3\n0,0 0,1 0,2") == [3]

    def test_legal_sorted(self):
        assert _rendered_legal("..\nx.\n1\n0,2") == [0, 1, 3]


# ---------------------------------------------------------------------------
# parse_own_card - always None for Dark Hex
# ---------------------------------------------------------------------------


class TestParseOwnCard:
    def test_empty_board(self):
        assert parse_own_card(_EMPTY) is None

    def test_non_empty_board(self):
        assert parse_own_card(_P0_PLAYED_A1) is None


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_non_empty(self):
        p = build_prompt(_EMPTY, 0)
        assert isinstance(p, str) and len(p) > 50

    def test_p0_goal(self):
        p = build_prompt(_EMPTY, 0)
        assert "top" in p.lower() and "bottom" in p.lower()

    def test_p1_goal(self):
        p = build_prompt(_EMPTY, 1)
        assert "left" in p.lower() and "right" in p.lower()

    def test_p0_color(self):
        p = build_prompt(_EMPTY, 0)
        assert "Black" in p

    def test_p1_color(self):
        p = build_prompt(_EMPTY, 1)
        assert "White" in p

    def test_own_stone_shown_as_Y(self):
        p = build_prompt(_P0_PLAYED_A1, 0)
        assert "[Y]" in p

    def test_discovered_enemy_shown_as_E(self):
        p = build_prompt(_P0_DISCOVERS_B1, 0)
        assert "[E]" in p

    def test_unknown_shown_as_blank(self):
        p = build_prompt(_EMPTY, 0)
        assert "[ ]" in p

    def test_available_moves_listed(self):
        # After P0 plays A1, available moves are B1, A2, B2
        p = build_prompt(_P0_PLAYED_A1, 0)
        assert "B1" in p
        assert "A2" in p
        assert "B2" in p
        assert "A1" not in p.split("Available moves:")[1]


# ---------------------------------------------------------------------------
# action_suffix
# ---------------------------------------------------------------------------


class TestActionSuffix:
    def test_returns_string(self):
        assert isinstance(action_suffix(_EMPTY), str)

    def test_lists_all_labels_when_empty(self):
        sfx = action_suffix(_EMPTY)
        for label in ["A1", "B1", "A2", "B2"]:
            assert label in sfx

    def test_excludes_played_cell(self):
        sfx = action_suffix(_P0_PLAYED_A1)
        # "right now: B1, A2, B2." -- A1 must not appear in the choices list
        choices_part = sfx.split("right now:")[1].split(".")[0]
        assert "A1" not in choices_part
        assert "B1" in choices_part

    def test_ends_with_the_shared_reply_instruction(self):
        """
        every game closes its action call with the same sentence.
        """
        sfx = action_suffix(_EMPTY)
        assert sfx.endswith(
            "Reply with exactly one of those, copied exactly, and nothing else. "
            "No explanation and no reasoning."
        )


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------


class TestParseAction:
    def test_label_a1(self):
        assert parse_action("A1", _EMPTY) == 0

    def test_label_b1(self):
        assert parse_action("B1", _EMPTY) == 1

    def test_label_a2(self):
        assert parse_action("A2", _EMPTY) == 2

    def test_label_b2(self):
        assert parse_action("B2", _EMPTY) == 3

    def test_label_case_insensitive(self):
        assert parse_action("b2", _EMPTY) == 3
        assert parse_action("a1", _EMPTY) == 0

    def test_label_in_sentence(self):
        assert parse_action("I choose A1 as my move.", _EMPTY) == 0

    def test_verbose_top_left(self):
        assert parse_action("I'll play top-left", _EMPTY) == 0

    def test_verbose_bottom_right(self):
        assert parse_action("bottom right", _EMPTY) == 3

    def test_verbose_top_right(self):
        assert parse_action("top right please", _EMPTY) == 1

    def test_verbose_bottom_left(self):
        assert parse_action("bottom-left", _EMPTY) == 2

    def test_coordinate_pair(self):
        assert parse_action("0,0", _EMPTY) == 0
        assert parse_action("1,1", _EMPTY) == 3

    def test_bare_integer(self):
        assert parse_action("2", _EMPTY) == 2

    def test_none_on_garbage(self):
        assert parse_action("xyzzy", _EMPTY) is None

    def test_none_on_empty_string(self):
        assert parse_action("", _EMPTY) is None

    def test_does_not_filter_illegal_actions(self):
        # parse_action parses without filtering - match_runner validates separately
        assert (
            parse_action("A1", _P0_PLAYED_A1) == 0
        )  # A1 is "illegal" but parser returns 0


# ---------------------------------------------------------------------------
# parse_action_vocab
# ---------------------------------------------------------------------------


class TestParseActionVocab:
    """parse_action_vocab mirrors parse_action's branch order exactly and returns
    which vocabulary resolved the input: "label", "verbose", "coords", "digit", or None.

    Named mutations that redden these tests:
    - Swap the label and verbose branch order in parse_action_vocab (order drift)
    - Return "label" unconditionally (ignores which vocabulary actually matched)
    """

    def test_label_vocabulary(self):
        """Short labels like A1, B2 resolve via the label vocabulary."""
        from lmopt.prompting.darkhex import make_fns

        fns = make_fns(2, 2)
        parse_vocab = fns["parse_action_vocab"]
        parse_act = fns["parse_action"]

        for text in ["A1", "I choose A1", "B2", "b1"]:
            vocab = parse_vocab(text, _EMPTY)
            action = parse_act(text, _EMPTY)
            assert vocab == "label", f"{text} should resolve via label vocab"
            assert action is not None, f"{text} must parse to an action"

    def test_verbose_vocabulary(self):
        """Corner names like top-left, bottom right resolve via verbose vocabulary."""
        from lmopt.prompting.darkhex import make_fns

        fns = make_fns(2, 2)
        parse_vocab = fns["parse_action_vocab"]
        parse_act = fns["parse_action"]

        for text in ["top-left", "topleft", "top left", "bottom-right", "bottom right"]:
            vocab = parse_vocab(text, _EMPTY)
            action = parse_act(text, _EMPTY)
            assert vocab == "verbose", f"{text} should resolve via verbose vocab"
            assert action is not None, f"{text} must parse to an action"

    def test_coords_vocabulary(self):
        """Coordinate pairs like 0,0 or 1, 1 resolve via coords vocabulary."""
        from lmopt.prompting.darkhex import make_fns

        fns = make_fns(2, 2)
        parse_vocab = fns["parse_action_vocab"]
        parse_act = fns["parse_action"]

        for text in ["0, 0", "1,1", "0,1"]:
            vocab = parse_vocab(text, _EMPTY)
            action = parse_act(text, _EMPTY)
            assert vocab == "coords", f"{text} should resolve via coords vocab"
            assert action is not None, f"{text} must parse to an action"

    def test_digit_vocabulary(self):
        """Bare word-bounded digits like 0, 2, 3 resolve via digit vocabulary."""
        from lmopt.prompting.darkhex import make_fns

        fns = make_fns(2, 2)
        parse_vocab = fns["parse_action_vocab"]
        parse_act = fns["parse_action"]

        for text in ["0", "2", "3", "I choose 1"]:
            vocab = parse_vocab(text, _EMPTY)
            action = parse_act(text, _EMPTY)
            assert vocab == "digit", f"{text} should resolve via digit vocab"
            assert action is not None, f"{text} must parse to an action"

    def test_multiple_labels_returns_none(self):
        """Text containing multiple labels is rejected by both parsers."""
        from lmopt.prompting.darkhex import make_fns

        fns = make_fns(2, 2)
        parse_vocab = fns["parse_action_vocab"]
        parse_act = fns["parse_action"]

        text = "A1, B1"
        vocab = parse_vocab(text, _EMPTY)
        action = parse_act(text, _EMPTY)
        assert vocab is None, "multiple labels should return None"
        assert action is None, "parse_action must also return None"

    def test_unparseable_returns_none(self):
        """Unparseable text returns None from both parsers."""
        from lmopt.prompting.darkhex import make_fns

        fns = make_fns(2, 2)
        parse_vocab = fns["parse_action_vocab"]
        parse_act = fns["parse_action"]

        for text in ["xyzzy", "invalid", ""]:
            vocab = parse_vocab(text, _EMPTY)
            action = parse_act(text, _EMPTY)
            assert vocab is None, f"{text} should return None"
            assert action is None, "parse_action must also return None"

    def test_parsers_agree_on_each_vocabulary(self):
        """parse_action_vocab returns tag X only when parse_action succeeds on the
        corresponding vocabulary. The two closures cannot disagree."""
        from lmopt.prompting.darkhex import make_fns

        fns = make_fns(2, 2)
        parse_vocab = fns["parse_action_vocab"]
        parse_act = fns["parse_action"]

        test_cases = [
            ("A1", "label", 0),
            ("B2", "label", 3),
            ("top-left", "verbose", 0),
            ("bottom-right", "verbose", 3),
            ("0, 0", "coords", 0),
            ("1,1", "coords", 3),
            ("0", "digit", 0),
            ("3", "digit", 3),
            ("A1, B1", None, None),
            ("invalid", None, None),
        ]

        for text, expected_vocab, expected_action in test_cases:
            vocab = parse_vocab(text, _EMPTY)
            action = parse_act(text, _EMPTY)
            assert vocab == expected_vocab, f"{text}: vocab mismatch"
            assert action == expected_action, f"{text}: action mismatch"

    def test_coords_miss_fallthrough_to_digit(self):
        """"9, 3" matches the coordinate-pair regex, but coords_action
        lookup misses (row 9 doesn't exist on 2x2 board), so parse_action falls
        through and resolves bare digit 3. parse_action_vocab must also return "digit",
        not "coords"."""
        from lmopt.prompting.darkhex import make_fns

        fns = make_fns(2, 2)
        parse_vocab = fns["parse_action_vocab"]
        parse_act = fns["parse_action"]

        action = parse_act("9, 3", _EMPTY)
        vocab = parse_vocab("9, 3", _EMPTY)

        assert action == 3, "Coords miss on row 9 must fallthrough to bare digit 3"
        assert (
            vocab == "digit"
        ), f"parse_action_vocab must also return 'digit', got {vocab}"


# ---------------------------------------------------------------------------
# belief_suffix / parse_belief - per-cell opponent-occupancy (P8 board belief)
# ---------------------------------------------------------------------------


class TestBelief:
    def test_belief_suffix_lists_unknown_cells(self):
        s = belief_suffix(_EMPTY)
        assert s is not None
        assert "opponent" in s.lower()
        # all four cells unknown at the empty board
        assert "A1" in s and "B1" in s and "A2" in s and "B2" in s

    def test_parse_belief_by_label_to_action_index(self):
        # Labels A1=0, B1=1, A2=2, B2=3; emit keyed by action index.
        out = parse_belief('{"A1": 0.1, "B1": 0.2, "A2": 0.3, "B2": 0.9}', _EMPTY)
        d = json.loads(out)
        assert set(d.keys()) == {"0", "1", "2", "3"}
        assert abs(d["0"] - 0.1) < 1e-9 and abs(d["3"] - 0.9) < 1e-9
        # Not normalised - independent Bernoullis (sum can exceed 1)
        assert sum(d.values()) > 1.0

    def test_parse_belief_clamps_and_defaults_missing(self):
        out = parse_belief('{"A1": 1.5}', _EMPTY)  # clamp 1.5->1.0; others->0
        d = json.loads(out)
        assert abs(d["0"] - 1.0) < 1e-9
        assert abs(d["1"]) < 1e-9 and abs(d["2"]) < 1e-9 and abs(d["3"]) < 1e-9

    def test_parse_belief_none_on_garbage(self):
        assert parse_belief("no json here", _EMPTY) is None

    def test_all_keys_unmappable_returns_none_not_renormalised(self):
        """REQUIRED sub-behaviour check, sibling of the
        kuhn/leduc/goofspiel/liarsdice tests of the same name: `parse_belief`
        must return None when every key maps through neither
        `label_action.get` nor the `int()` fallback, rather than
        renormalising over an empty partial parse. This is what makes
        DARKHEX_SPEC's placeholder-keyed exemplar (`<cell_A>`/`<cell_B>`)
        actually echo-safe."""
        assert parse_belief('{"<cell_A>": 0.5, "<cell_B>": 0.5}', _EMPTY) is None


# ---------------------------------------------------------------------------
# GamePromptFns registration
# ---------------------------------------------------------------------------


def test_get_game_fns_darkhex():
    from lmopt.prompting import get_game_fns

    fns = get_game_fns("darkhex")
    assert callable(fns.build_prompt)
    assert callable(fns.parse_action)


def test_get_game_fns_dark_hex():
    from lmopt.prompting import get_game_fns

    fns = get_game_fns("dark_hex")
    assert callable(fns.build_prompt)
    assert callable(fns.parse_action)
