"""Tests for memory rendering module.
"""

from __future__ import annotations

import pytest

from lmopt.prompting.memory import MemoryTerms
from lmopt.prompting.memory import (
    MEMORY_CHAR_BUDGET,
    MEMORY_RUNGS,
    HandMemory,
    render_memory,
    truncate_memory_block,
)

# A canonical poker-style terms set for tests that do not depend on a game's words.
_TERMS = MemoryTerms(
    "hand", "chips", label_of=lambda info_str, action: f"ACTION{action}"
)


class TestEmptyRenderRule:
    """The empty-render rule: M0 or empty memories with no scratchpad returns ""."""

    def test_m0_always_returns_empty(self):
        """M0 always renders empty string regardless of memories.
        """
        memories = [HandMemory(decisions=(("info1", 0), ("info2", 1)), payoff=1.0)]
        result = render_memory(memories, "M0", k=5, scratchpad=None, terms=_TERMS)
        assert result == ""

    def test_empty_memories_returns_empty_for_any_rung(self):
        """Empty memories with no scratchpad returns "" for any rung.
        """
        for rung in MEMORY_RUNGS:
            result = render_memory([], rung, k=5, scratchpad=None, terms=_TERMS)
            assert result == "", f"rung {rung} should return empty on empty input"

    def test_empty_memories_with_scratchpad_none_returns_empty(self):
        """Empty memories and scratchpad=None returns "" even for M3.
        """
        result = render_memory([], "M3", k=5, scratchpad=None, terms=_TERMS)
        assert result == ""


class TestM1Rendering:
    """M1 renders cross-hand summary line."""

    def test_m1_single_hand_summary(self):
        """M1 renders summary line with hand count, cumulative and last payoff.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=1.5)]
        result = render_memory(memories, "M1", k=5, scratchpad=None, terms=_TERMS)
        assert (
            result
            == "\n\nYour match so far, against the same opponent in every hand:\n- 1 hand played. Your total: +1.5 chips. Last hand: +1.5 chips."
        )

    def test_m1_multiple_hands_cumulative(self):
        """M1 sums payoffs across hands correctly.
        """
        memories = [
            HandMemory(decisions=(("info", 0),), payoff=1.0),
            HandMemory(decisions=(("info", 1),), payoff=-0.5),
            HandMemory(decisions=(("info", 2),), payoff=2.5),
        ]
        result = render_memory(memories, "M1", k=5, scratchpad=None, terms=_TERMS)
        # 1.0 + (-0.5) + 2.5 = 3.0
        assert (
            result
            == "\n\nYour match so far, against the same opponent in every hand:\n- 3 hands played. Your total: +3 chips. Last hand: +2.5 chips."
        )

    def test_m1_negative_payoff_format(self):
        """M1 formats negative payoffs with sign.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=-2.0)]
        result = render_memory(memories, "M1", k=5, scratchpad=None, terms=_TERMS)
        assert "-2" in result
        assert "Last hand: -2 chips." in result


class TestM2Rendering:
    """M2 renders M1 line plus transcript of last k hands."""

    def test_m2_includes_m1_line(self):
        """M2 starts with M1's summary line.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=1.0)]
        result = render_memory(memories, "M2", k=5, scratchpad=None, terms=_TERMS)
        assert result.startswith(
            "\n\nYour match so far, against the same opponent in every hand:\n- 1 hand played."
        )

    def test_m2_renders_single_hand_transcript(self):
        """M2 renders hand transcript with info_str and action verbatim.
        """
        memories = [
            HandMemory(
                decisions=(("Player 0 cards: J", 0), ("Player 0 cards: JK", 1)),
                payoff=1.0,
            )
        ]
        result = render_memory(memories, "M2", k=5, scratchpad=None, terms=_TERMS)
        # M2 transcript lines have leading \n, creating empty line when split
        assert "- 1 hand played." in result
        assert "- Hand 1 (your result +1 chip): " in result
        assert "Player 0 cards: J -> you chose ACTION0" in result
        assert "Player 0 cards: JK -> you chose ACTION1" in result

    def test_m2_window_respects_k(self):
        """M2 renders exactly min(k, n) hands, dropping oldest first.
        """
        memories = [
            HandMemory(decisions=(("hand1", 0),), payoff=1.0),
            HandMemory(decisions=(("hand2", 0),), payoff=2.0),
            HandMemory(decisions=(("hand3", 0),), payoff=3.0),
            HandMemory(decisions=(("hand4", 0),), payoff=4.0),
            HandMemory(decisions=(("hand5", 0),), payoff=5.0),
            HandMemory(decisions=(("hand6", 0),), payoff=6.0),
            HandMemory(decisions=(("hand7", 0),), payoff=7.0),
        ]
        result = render_memory(memories, "M2", k=3, scratchpad=None, terms=_TERMS)
        # Should only show hands 5, 6, 7 (last 3)
        assert "hand1" not in result
        assert "hand4" not in result
        assert "hand5" in result
        assert "hand6" in result
        assert "hand7" in result
        # Count hand lines
        hand_lines = [
            line for line in result.split("\n") if line.strip().startswith("- Hand")
        ]
        assert len(hand_lines) == 3

    def test_m2_hands_in_play_order(self):
        """M2 renders hands in chronological order, not reverse.
        """
        memories = [
            HandMemory(decisions=(("first", 0),), payoff=1.0),
            HandMemory(decisions=(("second", 0),), payoff=2.0),
            HandMemory(decisions=(("third", 0),), payoff=3.0),
        ]
        result = render_memory(memories, "M2", k=5, scratchpad=None, terms=_TERMS)
        first_idx = result.index("first")
        second_idx = result.index("second")
        third_idx = result.index("third")
        assert first_idx < second_idx < third_idx

    def test_m2_decisions_joined_with_semicolon(self):
        """M2 joins multiple decisions within a hand with semicolon-space.
        """
        memories = [
            HandMemory(
                decisions=(("state_a", 0), ("state_b", 1), ("state_c", 2)), payoff=1.0
            )
        ]
        result = render_memory(memories, "M2", k=5, scratchpad=None, terms=_TERMS)
        # Expect: "state_a -> you chose action 0; state_b -> ..."
        assert "ACTION0; " in result
        assert "ACTION1; " in result
        assert "ACTION2" in result


class TestM3Rendering:
    """M3 renders M1 line plus scratchpad."""

    def test_m3_includes_m1_line(self):
        """M3 starts with M1's summary line.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=1.0)]
        result = render_memory(memories, "M3", k=5, scratchpad="notes", terms=_TERMS)
        assert result.startswith(
            "\n\nYour match so far, against the same opponent in every hand:\n- 1 hand played."
        )

    def test_m3_renders_scratchpad_verbatim(self):
        """M3 includes scratchpad text exactly as provided.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=1.0)]
        scratchpad = "Opponent seems aggressive.\nPlays high cards early."
        result = render_memory(memories, "M3", k=5, scratchpad=scratchpad, terms=_TERMS)
        assert "- Your private notes on the opponent so far:" in result
        assert "Opponent seems aggressive." in result
        assert "Plays high cards early." in result

    def test_m3_scratchpad_none_with_memories_renders_m1_only(self):
        """M3 with memories but scratchpad=None renders only M1 line.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=1.0)]
        result = render_memory(memories, "M3", k=5, scratchpad=None, terms=_TERMS)
        assert (
            result
            == "\n\nYour match so far, against the same opponent in every hand:\n- 1 hand played. Your total: +1 chip. Last hand: +1 chip."
        )
        assert "private notes" not in result


class TestDeterminism:
    """Rendering must be byte-deterministic."""

    def test_render_determinism(self):
        """Two calls with same input produce byte-identical output.
        """
        memories = [
            HandMemory(decisions=(("info1", 0), ("info2", 1)), payoff=1.0),
            HandMemory(decisions=(("info3", 2),), payoff=-0.5),
        ]
        result1 = render_memory(memories, "M2", k=5, scratchpad=None, terms=_TERMS)
        result2 = render_memory(memories, "M2", k=5, scratchpad=None, terms=_TERMS)
        assert result1 == result2


class TestTruncation:
    """truncate_memory_block enforces char budget with line-boundary cuts."""

    def test_truncation_within_budget_unchanged(self):
        """Block within budget returns unchanged with flag False.
        """
        block = "- Match so far: 5 hands\n- Hand 1: x\n"
        result, truncated = truncate_memory_block(block, budget=1000)
        assert result == block
        assert truncated is False

    def test_truncation_over_budget_drops_lines(self):
        """Block over budget returns truncated with flag True.
        """
        # Build a block that exceeds budget
        lines = [f"- Hand {i}: " + "x" * 100 + "\n" for i in range(20)]
        block = "".join(lines)
        assert len(block) > 500

        result, truncated = truncate_memory_block(block, budget=500)
        assert len(result) <= 500
        assert truncated is True

    def test_truncation_preserves_line_boundaries(self):
        """Truncation cuts at line boundaries, never mid-line.
        """
        lines = [f"- Hand {i}: " + "x" * 100 + "\n" for i in range(10)]
        block = "".join(lines)

        result, _truncated = truncate_memory_block(block, budget=300)
        # Result should end with newline (complete line)
        assert result == "" or result.endswith("\n")
        # Should not have incomplete lines
        assert not result.endswith("x")

    def test_truncation_drops_oldest_first(self):
        """Truncation drops oldest lines first, keeping newest.
        """
        lines = [
            "- Match so far: 5 hands\n",
            "- Hand 1: oldest\n",
            "- Hand 2: middle\n",
            "- Hand 3: newest\n",
        ]
        block = "".join(lines)

        result, _truncated = truncate_memory_block(block, budget=60)
        # Should keep newest content
        assert "newest" in result or "middle" in result
        # Should drop oldest if needed
        if len(result) < 100:
            assert "oldest" not in result

    def test_m2_truncation_drops_oldest_transcript_lines(self):
        """M2 truncation drops OLDEST transcript lines, newest survives.
        """
        # Build M2 block via production render (not hand-assembled strings)
        memories = [
            HandMemory(decisions=(("OLDEST_HAND", 0),), payoff=1.0),
            HandMemory(decisions=(("MIDDLE_HAND", 1),), payoff=2.0),
            HandMemory(decisions=(("MIDDLE2_HAND", 2),), payoff=3.0),
            HandMemory(decisions=(("MIDDLE3_HAND", 3),), payoff=4.0),
            HandMemory(decisions=(("NEWEST_HAND", 4),), payoff=5.0),
        ]
        block = render_memory(memories, "M2", k=5, scratchpad=None, terms=_TERMS)

        # Set budget to force dropping some transcript lines
        result, truncated = truncate_memory_block(block, budget=200)

        assert truncated is True
        assert len(result) <= 200
        # M1 summary must survive
        assert "- 5 hands played." in result
        # Newest hand should survive
        assert "NEWEST_HAND" in result
        # Oldest hand should be dropped
        assert "OLDEST_HAND" not in result

    def test_m3_truncation_drops_scratchpad_tail(self):
        """M3 truncation drops scratchpad TAIL, opening of notes survives.
        """
        # Build M3 block via production render (not hand-assembled strings)
        memories = [
            HandMemory(decisions=(("info1", 0),), payoff=2.0),
            HandMemory(decisions=(("info2", 1),), payoff=1.0),
            HandMemory(decisions=(("info3", 2),), payoff=2.0),
        ]
        scratchpad = (
            "OPENING_NOTE here.\n"
            "MIDDLE_NOTE here.\n"
            "LATER_NOTE here.\n"
            "TAIL_NOTE here.\n"
        )
        block = render_memory(memories, "M3", k=5, scratchpad=scratchpad, terms=_TERMS)

        # A budget that ends just before the third note forces dropping tail lines only
        budget = block.index("LATER_NOTE")
        result, truncated = truncate_memory_block(block, budget=budget)

        assert truncated is True
        assert len(result) <= budget
        # M1 summary and notes header must survive
        assert "- 3 hands played." in result
        assert "private notes" in result
        # Opening note should survive (beginning of scratchpad)
        assert "OPENING_NOTE" in result
        # Tail note should be dropped
        assert "TAIL_NOTE" not in result

    def test_m3_single_line_scratchpad_character_truncates(self):
        """M3 with single-line scratchpad over budget character-truncates instead of dropping.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=1.0)]
        # Single long scratchpad line
        scratchpad = "A" * 500
        block = render_memory(memories, "M3", k=5, scratchpad=scratchpad, terms=_TERMS)

        # Tight budget that forces character truncation
        result, truncated = truncate_memory_block(block, budget=100)

        assert truncated is True
        assert len(result) <= 100, f"Block over budget: {len(result)} > 100"
        # M1 line or notes header should be present (not a dangling header)
        assert "Your match so far" in result or "private notes" in result


class TestUnknownRung:
    """Unknown rung raises ValueError, no silent fallback."""

    def test_unknown_rung_raises_valueerror(self):
        """Unknown rung value raises ValueError with message.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=1.0)]
        with pytest.raises(ValueError) as exc_info:
            render_memory(memories, "M99", k=5, scratchpad=None, terms=_TERMS)

        # Error should name the bad value and valid set
        assert "M99" in str(exc_info.value)
        assert "M0" in str(exc_info.value) or "MEMORY_RUNGS" in str(exc_info.value)


class TestConstants:
    """Module exports expected constants."""

    def test_memory_rungs_tuple(self):
        """MEMORY_RUNGS is the exact 4-tuple of valid rungs.
        """
        assert MEMORY_RUNGS == ("M0", "M1", "M2", "M3")

    def test_memory_char_budget_value(self):
        """MEMORY_CHAR_BUDGET is 6000 per the headroom argument.
        """
        assert MEMORY_CHAR_BUDGET == 6000


class TestHeadroomAssertion:
    """MEMORY_CHAR_BUDGET leaves base prompts representable within serving window."""

    def test_budget_plus_longest_prompt_fits_serving_window(self):
        """Budget + longest prompt + decode margin fits 4096-token window.

        Arithmetic: serving window is 4096 tokens. Working ratio ~4 chars/token
        (approximation) gives 16384 chars. Reserve 2048 chars for decode (512 tokens
        at the working ratio). Assert longest_prompt + MEMORY_CHAR_BUDGET <= 14336.
        """
        pytest.importorskip("pyspiel")
        from lmopt.games.loaders import load_game
        from lmopt.prompting import get_game_fns_for_game
        from lmopt.prompting.fidelity import enumerate_infosets

        # Enumerate kuhn, leduc, goofspiel fully (small trees).
        # Sample liarsdice and darkhex (large trees, same sampling as render_pin test).
        games_to_measure = ["kuhn", "leduc", "goofspiel"]

        max_prompt_chars = 0
        for game_name in games_to_measure:
            game = load_game(game_name)
            fns = get_game_fns_for_game(game)
            for seat in (0, 1):
                rendered = enumerate_infosets(game, seat)
                for info_str in rendered:
                    # Measure action channel: base + action_suffix
                    action_prompt = rendered[info_str]
                    max_prompt_chars = max(max_prompt_chars, len(action_prompt))

                    # Measure belief channel: base + belief_suffix
                    belief_sfx = fns.belief_suffix(info_str)
                    if belief_sfx is not None:
                        base = fns.build_prompt(info_str, seat)
                        belief_prompt = base + belief_sfx
                        max_prompt_chars = max(max_prompt_chars, len(belief_prompt))

        # Sample the two large games (liarsdice, darkhex) at a few states.
        # darkhex_3x3: sampling only 10 states via bounded BFS, never a full
        # enumeration, so board size carries no tractability cost here.
        large_games = [
            ("liarsdice", 10),  # Sample 10 states
            ("darkhex_3x3", 10),
        ]
        for game_name, sample_count in large_games:
            game = load_game(game_name)
            fns = get_game_fns_for_game(game)
            state = game.new_initial_state()
            sampled = 0
            states_to_sample = [state.clone()]

            while states_to_sample and sampled < sample_count:
                current = states_to_sample.pop(0)
                if current.is_terminal():
                    continue

                player = current.current_player()
                if player >= 0:
                    info_str = current.information_state_string(player)
                    # Measure both channels
                    base = fns.build_prompt(info_str, player)
                    action_sfx = fns.action_suffix(info_str)
                    action_prompt = base + action_sfx
                    max_prompt_chars = max(max_prompt_chars, len(action_prompt))

                    belief_sfx = fns.belief_suffix(info_str)
                    if belief_sfx is not None:
                        belief_prompt = base + belief_sfx
                        max_prompt_chars = max(max_prompt_chars, len(belief_prompt))

                    sampled += 1

                # Expand one child for continued sampling
                legal = current.legal_actions()
                if legal:
                    child = current.child(legal[0])
                    states_to_sample.append(child)

        # Serving window: 4096 tokens * ~4 chars/token = 16384 chars
        # Reserve 2048 chars for decode (512 tokens at working ratio)
        # Headroom check: longest_prompt + MEMORY_CHAR_BUDGET + decode_margin <= 16384
        SERVING_WINDOW_CHARS = 16384
        DECODE_MARGIN_CHARS = 2048
        available_for_prompt_and_memory = SERVING_WINDOW_CHARS - DECODE_MARGIN_CHARS

        assert (
            max_prompt_chars + MEMORY_CHAR_BUDGET <= available_for_prompt_and_memory
        ), (
            f"Budget headroom violated: longest prompt {max_prompt_chars} chars + "
            f"MEMORY_CHAR_BUDGET {MEMORY_CHAR_BUDGET} = {max_prompt_chars + MEMORY_CHAR_BUDGET} "
            f"exceeds available {available_for_prompt_and_memory} chars "
            f"(serving window {SERVING_WINDOW_CHARS} - decode margin {DECODE_MARGIN_CHARS})"
        )


class TestTheTranscriptWindowMustBePositive:
    @pytest.mark.parametrize("k", [0, -1])
    def test_an_m2_window_below_one_is_refused(self, k):
        """
        `memories[-0:]` is the whole history, so a window of 0 would render every hand the agent
        had played. Refused for M2, even with no memory yet, so the configuration error surfaces on
        hand 1.
        """
        memories = [
            HandMemory(decisions=(("info", 0),), payoff=1.0),
            HandMemory(decisions=(("info", 1),), payoff=-1.0),
        ]
        with pytest.raises(ValueError, match="at least 1"):
            render_memory(memories, "M2", k=k, scratchpad=None, terms=_TERMS)
        with pytest.raises(ValueError, match="at least 1"):
            render_memory([], "M2", k=k, scratchpad=None, terms=_TERMS)

    def test_the_window_does_not_constrain_the_other_rungs(self):
        """
        k is the M2 window only; M0 and M1 ignore it, so a k of 0 there is not an error.
        """
        memories = [HandMemory(decisions=(("info", 0),), payoff=1.0)]
        assert render_memory(memories, "M0", k=0, scratchpad=None, terms=_TERMS) == ""
        assert "1 hand played" in render_memory(
            memories, "M1", k=0, scratchpad=None, terms=_TERMS
        )
