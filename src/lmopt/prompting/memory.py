"""Memory rendering for LM agents across hands.

Four memory rungs (M0-M3) vary what context the agent carries across hands within
a match. Rendering is pure: a function from (memories, rung, params, terms) to string,
with no side effects and byte-deterministic output.

The block names what it is and speaks the game's own words: every hand of a match is
played against the same opponent, a play is called whatever the rules call it (a hand of
poker, a game of Goofspiel), and a payoff is shown in the units the rules state. The
engine's return is not always in those units, so each game supplies the factor that
converts it (``MemoryTerms.payoff_factor``).

The wording is data: ``memory.json`` in the live prompt set holds every template this module
renders, as ``str.format`` templates with named fields, and this module supplies the values
(counts, amounts, labels, the flattened state text). ``memory_render_hash`` pins the rendered
result, so a change to a template moves the digest exactly as a change here would.

Character budget headroom argument: the serving window is per model, not one
figure -- Qwen and Mistral are served at 8192 tokens, OLMo at 4096
(``scripts/collection_job.py``'s ``MODELS`` table is the live source for all
three). The longest enumerated base prompt+suffix across the five games is
measured in the test suite, and the budget is chosen to leave those prompts
representable. The working ratio is ~4 chars per token (an approximation,
recorded as such, not a tight bound). MEMORY_CHAR_BUDGET = 6000 leaves
headroom for the base prompt, memory block, and completion within the
smaller, 4096-token OLMo window, and is correspondingly more conservative
for the two models served at 8192 tokens.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

from lmopt.prompts import LIVE_SET, load_set

MEMORY_RUNGS = ("M0", "M1", "M2", "M3")
MEMORY_CHAR_BUDGET = 6000

#: The memory wording templates of the live prompt set, by key.
_WORDING = load_set(LIVE_SET).memory
NOTES_HEADER = _WORDING["notes_header"]


@dataclass(frozen=True)
class HandMemory:
    """One completed hand, as this agent observed it.

    `decisions` holds `(info_str, action)` for each decision this agent itself
    made, in play order. `payoff` is its own terminal return, supplied by the
    harness through `observe_hand_end` because it is not recoverable from the
    agent's own stream: the terminal information-state
    string leaves the payoff ambiguous in kuhn (5 of 12 classes) and liarsdice
    (6,300 of 24,570), and the agent's last pre-terminal info_str never sees
    the showdown at all.

    Frozen because a rendered memory must be a faithful record of what was
    observed; a mutable one invites post-hoc edits that no test would catch.
    """

    decisions: tuple[tuple[str, int], ...]
    payoff: Optional[float]


@dataclass(frozen=True)
class MemoryTerms:
    """
    the words and units one game's memory block is written in
    :param unit: what one play is called in the game's rules, "hand" or "game" (str)
    :param currency: what is won and lost, in the plural, "chips" or "points" (str)
    :param payoff_factor: multiplies the engine's return into the units the rules state;
        Goofspiel's engine returns half the point difference its rules define, so its
        factor is 2, and a payoff-scale variant multiplies in its display scale (float)
    :param label_of: maps (information state string, action) to the reply label the model
        was offered at that decision, so the M2 transcript names a past move the way the
        prompt did; None only for terms no M2 block is rendered with
        (Callable[[str, int], str] | None)
    :param state_view: rewrites an information state string into the units the prompt states
        its payoff magnitudes in, for the M2 transcript's replayed states; None for every
        canonical game, whose replayed state is the engine's own text
        (Callable[[str], str] | None)
    """

    unit: str
    currency: str
    payoff_factor: float = 1.0
    label_of: Optional[Callable[[str, int], str]] = field(default=None, compare=False)
    state_view: Optional[Callable[[str], str]] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """
        refuse terms that would render a malformed block
        :raises ValueError: on an empty word or a non-positive factor
        """
        if not self.unit or not self.currency.endswith("s"):
            raise ValueError(
                f"MemoryTerms needs a unit and a plural currency, got {self!r}"
            )
        if not self.payoff_factor > 0:
            raise ValueError(
                f"MemoryTerms payoff_factor must be positive, got {self!r}"
            )

    def scaled(self, display_scale: float) -> "MemoryTerms":
        """
        these terms with the payoff factor multiplied by a prompt variant's display scale,
        so a payoff-scale run's memory shows the magnitudes its prompt shows
        :param display_scale: the variant spec's ``display_scale``, 1 for canonical (float)
        :returns: the scaled terms (MemoryTerms)
        """
        return replace(self, payoff_factor=self.payoff_factor * display_scale)

    def with_labels(self, action_labels: Callable[[str], dict]) -> "MemoryTerms":
        """
        these terms with the game's reply labels attached, for the M2 transcript
        :param action_labels: the game's ``action_labels``, info string to {action: label} (Callable)
        :returns: the terms with ``label_of`` set (MemoryTerms)
        """
        return replace(
            self, label_of=lambda info_str, action: action_labels(info_str)[action]
        )

    def with_state_view(self, state_view: Callable[[str], str]) -> "MemoryTerms":
        """
        these terms with the game's state rewriting attached, for a payoff-scale variant
        :param state_view: info state string to the string the transcript replays (Callable)
        :returns: the terms with ``state_view`` set (MemoryTerms)
        """
        return replace(self, state_view=state_view)


def _amount(value: float, terms: MemoryTerms) -> str:
    """
    one payoff in the game's own units, signed, e.g. "-2 chips" or "+1 point"
    :param value: the engine's return (float)
    :param terms: the game's memory terms (MemoryTerms)
    :returns: the rendered amount (str)
    """
    shown = value * terms.payoff_factor
    word = terms.currency[:-1] if abs(shown) == 1 else terms.currency
    return _WORDING["amount"].format(amount=shown, word=word)


def _render_summary(memories: Sequence[HandMemory], terms: MemoryTerms) -> str:
    """
    the section common to M1, M2 and M3: what the block is, then the count, the total
    and the last result
    :param memories: completed hands in play order (Sequence[HandMemory])
    :param terms: the game's memory terms (MemoryTerms)
    :returns: the section with a leading blank line and no trailing newline, or "" when
        memories is empty, which is part of the empty-render rule (str)
    """
    if not memories:
        return ""
    n = len(memories)
    total = sum(m.payoff for m in memories if m.payoff is not None)
    last = memories[-1].payoff if memories[-1].payoff is not None else 0.0
    plays = terms.unit if n == 1 else f"{terms.unit}s"
    return _WORDING["header"].format(unit=terms.unit) + _WORDING["summary"].format(
        n=n,
        plays=plays,
        total=_amount(total, terms),
        unit=terms.unit,
        last=_amount(last, terms),
    )


def render_memory(
    memories: Sequence[HandMemory],
    rung: str,
    k: int,
    scratchpad: str | None,
    terms: MemoryTerms,
) -> str:
    """
    Render the memory block for insertion into the LM prompt.
    Empty-render rule: M0, or any rung with empty memories and scratchpad None, returns exactly
    "". This makes hand-1 prompts byte-identical across rungs.
    :param memories: completed hands observed by this agent, in play order (Sequence[HandMemory])
    :param rung: memory rung identifier: M0, M1, M2 or M3 (str)
    :param k: transcript window size for M2, how many recent hands to include (int)
    :param scratchpad: the agent's private notes for M3, None if not yet written (str | None)
    :param terms: the game's words and payoff units (MemoryTerms)
    :returns: the rendered memory string, or "" when the empty-render rule applies (str)
    :raises ValueError: if rung is not in MEMORY_RUNGS, or rung is M2 and k is below 1
    """
    if rung not in MEMORY_RUNGS:
        raise ValueError(f"Unknown memory rung: {rung!r}. Valid rungs: {MEMORY_RUNGS}")
    if rung == "M2" and k < 1:
        # `memories[-0:]` is the whole history, so a window of 0 would render every hand.
        raise ValueError(f"M2 transcript window k must be at least 1, got {k!r}")

    if rung == "M0":
        return ""
    if not memories and scratchpad is None:
        return ""

    summary = _render_summary(memories, terms)
    if rung == "M1":
        return summary

    if rung == "M2":
        if not memories:
            return summary
        window = memories[-k:] if len(memories) > k else memories
        # Numbering is 1-indexed and relative to the full history.
        start_idx = len(memories) - len(window) + 1
        if terms.label_of is None:
            raise ValueError(
                "an M2 transcript needs the game's reply labels (MemoryTerms.label_of)"
            )
        label = terms.unit.capitalize()
        transcript_lines = []
        for idx, hand in enumerate(window, start=start_idx):
            payoff = hand.payoff if hand.payoff is not None else 0.0
            decisions_str = decision_text(hand.decisions, terms)
            transcript_lines.append(
                _WORDING["transcript_line"].format(
                    label=label,
                    idx=idx,
                    payoff=_amount(payoff, terms),
                    decisions=decisions_str,
                )
            )
        return summary + "".join(transcript_lines)

    if rung == "M3":
        if scratchpad is None:
            return summary
        return summary + _WORDING["notes"].format(
            notes_header=NOTES_HEADER, scratchpad=scratchpad
        )

    # Unreachable: rung validated at entry
    raise AssertionError(f"Unhandled rung: {rung}")  # pragma: no cover


def decision_text(decisions: Sequence[tuple[str, int]], terms: MemoryTerms) -> str:
    """
    one past hand's decisions as the M2 transcript writes them, on one line: each state as
    the engine describes it, its line breaks flattened, then the reply the model gave
    :param decisions: (information state string, action) pairs of one hand, in play order (Sequence)
    :param terms: the game's memory terms, whose ``label_of`` names each reply (MemoryTerms)
    :returns: the decisions joined with "; " (str)
    :raises ValueError: when the terms carry no reply labels
    """
    label_of = terms.label_of
    if label_of is None:
        raise ValueError(
            "an M2 transcript needs the game's reply labels (MemoryTerms.label_of)"
        )
    view = terms.state_view or (lambda state: state)
    return _WORDING["decision_join"].join(
        _WORDING["decision"].format(
            state=_WORDING["state_line_join"].join(
                part.strip() for part in view(info_str).splitlines() if part.strip()
            ),
            label=label_of(info_str, action),
        )
        for info_str, action in decisions
    )


def truncate_memory_block(block: str, budget: int) -> tuple[str, bool]:
    """
    Truncate a memory block to fit a character budget at line boundaries.
    The summary section (every line up to and including the first "- " line) always
    survives. After it, an M2 block drops whole transcript lines, oldest first, and an M3
    block drops scratchpad lines from the tail, until the block fits. Only when the summary
    alone exceeds the budget is it cut mid-line.
    :param block: the rendered memory block (str)
    :param budget: maximum character count (int)
    :returns: (truncated_block, was_truncated): the block, possibly shortened, and a flag saying whether truncation occurred (tuple[str, bool])
    """
    if len(block) <= budget:
        return block, False

    lines = block.split("\n")
    head_end = next(
        (i for i, line in enumerate(lines) if line.startswith("- ")), len(lines) - 1
    )
    header, rest = lines[: head_end + 1], lines[head_end + 1 :]
    is_m3 = NOTES_HEADER in rest

    def joined(tail: list[str]) -> str:
        return "\n".join(header + tail)

    if is_m3:
        # Keep the notes header; drop the scratchpad from its tail.
        while len(rest) > 1 and len(joined(rest)) > budget:
            rest = rest[:-1]
    else:
        # Drop transcript lines, oldest first.
        while rest and len(joined(rest)) > budget:
            rest = rest[1:]

    result = joined(rest)
    if len(result) > budget:
        result = result[:budget]
    return result, True


#: The run_meta key a run records `memory_render_hash()` under; every writer and reader imports it.
MEMORY_RENDER_HASH_KEY = "memory_render_hash"

#: The games the digest renders in their own terms: the canonical five the ladder collects.
_HASH_GAMES = ("kuhn", "leduc", "goofspiel", "liarsdice", "darkhex")
_HASH_PAYOFFS = (1.0, -2.0, None, 0.5, 3.0, -1.0, 13.0)


def fixture_memories(
    info: str, legal: Sequence[int], payoffs: Sequence[Optional[float]]
) -> list[HandMemory]:
    """
    a fixed memory of one decision per hand at one information state, cycling through the
    legal actions, with the given payoffs; the construction the wording digest and the readable
    rendered pins share, so both exercise the same block
    :param info: the information state string every hand's decision is recorded at (str)
    :param legal: the actions legal there, in the order to cycle through (Sequence[int])
    :param payoffs: one payoff per hand, None for a hand with no recorded payoff (Sequence)
    :returns: the hands in play order (list[HandMemory])
    """
    return [
        HandMemory(decisions=((info, legal[n % len(legal)]),), payoff=payoff)
        for n, payoff in enumerate(payoffs)
    ]


def memory_render_hash() -> str:
    """
    a digest of the memory wording as each canonical game renders it, so a run can record
    which wording it was collected under and a pilot gate can refuse a pilot collected under
    another. The prompt spec hash does not cover this module, and every M0 and hand-1 prompt
    is identical across wordings, so without this digest two M1 wordings are
    indistinguishable in the record. Each game renders a fixed memory built from its own first
    decision state and reply labels, through its own ``memory_terms``, at every rung and once
    truncated, so a change to a game's words, payoff factor or labels moves the digest as
    surely as a change to this module's text. Where the block sits in the prompt is decided by
    the agent, not here, and is outside the digest
    :returns: the sha256 hex digest (str)
    """
    from lmopt.games.loaders import load_game
    from lmopt.prompting import get_game_fns_for_game

    parts = [f"budget={MEMORY_CHAR_BUDGET}"]
    for key in _HASH_GAMES:
        game = load_game(key)
        fns = get_game_fns_for_game(game)
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        player = state.current_player()
        info = state.information_state_string(player)
        terms = fns.memory_terms
        if terms is None:
            raise ValueError(f"{key} carries no memory terms")
        legal = sorted(fns.action_labels(info))
        memories = fixture_memories(info, legal, _HASH_PAYOFFS)
        for rung in MEMORY_RUNGS:
            for scratchpad in (None, "note one\nnote two"):
                parts.append(render_memory(memories, rung, 5, scratchpad, terms))
        parts.append(
            truncate_memory_block(render_memory(memories, "M2", 5, None, terms), 160)[0]
        )
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
