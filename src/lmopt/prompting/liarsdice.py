# src/lmopt/prompting/liarsdice.py
"""Liar's Dice: info-state string -> NL prompt, action/belief parsing.

Liar's Dice info-state format (OpenSpiel, player-N view):
  "<own_die_face> [<count>-<face> ...]"

  - First token: own die value (1–6).
  - Subsequent tokens: bid history, each as "count-face"
    (e.g. "1-3" = at least one die shows 3, "2-5" = at least two show 5).
  - No tokens after own die = no bids yet.

  Examples:
    "4"         -> own die=4, no bids
    "4 1-3"     -> own die=4, one bid: 1×3
    "4 1-3 2-5" -> own die=4, bids: 1×3, then 2×5

Action encoding (numdice=1, dice_sides=6, players=2):
  action = (count - 1) * DICE_SIDES + (face - 1)   for bids (0–11)
  action = 12                                        for CHALLENGE

Parametric on prompt content: ``make_fns(spec)`` builds the
prompt/parse closures for one ``GamePromptSpec``, and the module-level names
are that factory applied to this module's own ``_SPEC``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from lmopt.prompting.memory import MemoryTerms
from lmopt.prompting.spec import LIARSDICE_SPEC as _SPEC
from lmopt.prompting.spec import (
    GamePromptSpec,
    finite_float,
    has_multiple_labels,
    in_display_order,
    labels_in_display_order,
    legal_response_suffix,
    load_belief_object,
    quoted_keys,
)

_NUMDICE = 1  # per player
_DICE_SIDES = 6
_NUM_PLAYERS = 2
_MAX_COUNT = _NUMDICE * _NUM_PLAYERS  # 2 total dice
_CHALLENGE_ACTION = _MAX_COUNT * _DICE_SIDES  # 12

# `_SPEC` and `_VOCAB` are bound once, when this module is imported: `_SPEC` to the
# `LIARSDICE_SPEC` object that `spec` held at that moment, and `_VOCAB` to that
# object's `action_vocabulary` dict. `build_prompt` reads this module's `_SPEC`, so
# patching `spec.LIARSDICE_SPEC` afterwards changes neither it nor `_VOCAB`. A test
# that needs different spec content patches `liarsdice._SPEC` and `liarsdice._VOCAB`
# instead. The default instance built by `make_fns` reads both attributes at call
# time (see `_content`).
_VOCAB = _SPEC.action_vocabulary


def _word_pattern(label: str) -> str:
    """A `\\bLABEL\\b` regex pattern for one vocabulary literal."""
    return r"\b" + re.escape(label) + r"\b"


def _parse_own_die(info_str: str) -> int:
    """Extract own die face (1–6) from the info-state string."""
    return int(info_str.split()[0])


def _parse_bid_history(info_str: str) -> list[tuple[int, int]]:
    """Return list of (count, face) pairs from the bid history tokens."""
    tokens = info_str.split()[1:]
    bids = []
    for tok in tokens:
        count_str, face_str = tok.split("-")
        bids.append((int(count_str), int(face_str)))
    return bids


def parse_own_card(info_str: str) -> Optional[int]:
    """Return own die face (1–6). Named parse_own_card for interface compatibility."""
    try:
        return _parse_own_die(info_str)
    except (ValueError, IndexError):
        return None


# What the memory block calls a play and what is won or lost, in the rules' own words.
MEMORY_TERMS = MemoryTerms("hand", "points")


def make_fns(
    spec: Optional[GamePromptSpec] = None, dice_sides: int = _DICE_SIDES
) -> dict:
    """
    Build the Liar's Dice prompt and parse closures for one prompt spec and one dice depth.
    The bid labels ("2 5" and friends) are generated from `dice_sides` and the module-level
    `_MAX_COUNT`, the same split darkhex and goofspiel make between generated label
    enumerations and constant prose. The only vocabulary token this game carries is CHALLENGE.
    The number of dice per player stays fixed at the module-level `_NUMDICE` (1): a
    two-dice game is blocked by a known posterior defect, and no registered variant varies it.
    :param spec: the ``GamePromptSpec`` whose instruction prose, action vocabulary and belief
        question these closures render, or None (the default, used for the module-level
        instance) to read this module's ``_SPEC`` and ``_VOCAB`` at call time, which keeps the
        monkeypatch contract described above `_VOCAB`. A spec passed explicitly is bound here,
        so a later patch of the module attributes does not reach it. It stays the first
        positional parameter because ``lmopt.prompting.__init__``'s ``_SPEC_FACTORIES``
        dispatch calls every spec-aware game's
        factory as ``make_fns(spec)`` (Optional[GamePromptSpec])
    :param dice_sides: number of faces per die; the default is the canonical 6-sided game (int)
    :returns: a dict of the ``GamePromptFns`` fields, the same dict-not-dataclass convention as
        ``darkhex.make_fns`` and ``goofspiel.make_fns`` (dict)
    :raises ValueError: when ``dice_sides`` is outside 1..9
    """
    if not 1 <= dice_sides <= 9:
        raise ValueError(
            f"dice_sides={dice_sides} is outside 1..9: parse_action's COUNT/FACE class is "
            f"single-digit, so a larger die needs a different regex"
        )
    challenge_action = _MAX_COUNT * dice_sides

    def _content() -> tuple[GamePromptSpec, Mapping[str, Any]]:
        """
        the prompt spec and action vocabulary this call renders with
        :returns: the explicit spec and its vocabulary if one was passed to `make_fns`, else this
            module's ``_SPEC`` and ``_VOCAB`` read now (tuple[GamePromptSpec, dict])
        """
        explicit = spec
        if explicit is None:
            return _SPEC, _VOCAB
        return explicit, explicit.action_vocabulary

    def _encode(count: int, face: int) -> int:
        return (count - 1) * dice_sides + (face - 1)

    def build_prompt(info_str: str, current_player: int) -> str:
        """Return the base NL prompt for a Liar's Dice decision node."""
        sp, vocab = _content()
        own_die = _parse_own_die(info_str)
        bids = _parse_bid_history(info_str)

        if bids:
            history_desc = ", ".join(f"{c} {f}" for c, f in bids)
            last_count, last_face = bids[-1]
            at_max_bid = _encode(last_count, last_face) == _encode(
                _MAX_COUNT, dice_sides
            )
            situation_key = (
                "situation_facing_max_bid" if at_max_bid else "situation_facing_bid"
            )
            situation = sp.instruction[situation_key].format(
                last_count=last_count, last_face=last_face, CHALLENGE=vocab["CHALLENGE"]
            )
        else:
            history_desc = "No bids yet."
            situation = "You go first. You must make the opening bid."

        return f"""{sp.instruction['rules']}

Current state:
- You are Player {current_player}.
- Your die shows: {own_die}
- Bid history: {history_desc}
- {situation}"""

    def action_suffix(info_str: str) -> str:
        """Suffix appended to build_prompt for the action call."""
        sp, vocab = _content()
        bids = _parse_bid_history(info_str)

        # Compute legal bids: all (count, face) combinations with action > last bid action.
        last_action = _encode(*bids[-1]) if bids else -1
        legal_bids = []
        for count in range(1, _MAX_COUNT + 1):
            for face in range(1, dice_sides + 1):
                a = _encode(count, face)
                if a > last_action:
                    legal_bids.append(f"{count} {face}")

        labels = list(legal_bids)
        if bids:
            labels.append(vocab["CHALLENGE"])
        # the example note below reads the lowest canonical bid, so it is built from
        # `legal_bids`, which the reordering never touches
        displayed = in_display_order(sp, labels)
        if not legal_bids:
            # the previous bid is the maximum: CHALLENGE is the only legal reply
            return legal_response_suffix(displayed)
        low_count, low_face = legal_bids[0].split()
        wild = (
            ""
            if int(low_face) == dice_sides
            else f", counting any {dice_sides} as a {low_face}"
        )
        note = (
            f" A bid is written COUNT FACE: for example '{legal_bids[0]}' claims that at "
            f"least {low_count} of the two dice show {low_face}{wild}."
        )
        return legal_response_suffix(displayed, note=note)

    def action_labels(info_str: str) -> dict[int, str]:
        """
        map each legal action at this info-set to the exact literal `action_suffix` offers for it
        The legal-bid loop and the CHALLENGE gating (`if bids:`) mirror `action_suffix` and read
        the same data (`_encode`, `_parse_bid_history`) rather than re-deriving the branch from
        rendered text. It reads the same vocabulary as `action_suffix` and `parse_action`
        (`_content`), so the CHALLENGE literal cannot drift between them under an injected
        variant spec.
        :param info_str: the OpenSpiel info-state string, "<own die> [<count>-<face> ...]" (str)
        :returns: legal action index -> the literal offered for it, "COUNT FACE" for a bid and the
            CHALLENGE token for a challenge (dict[int, str])
        """
        sp, vocab = _content()
        bids = _parse_bid_history(info_str)
        last_action = _encode(*bids[-1]) if bids else -1
        labels: dict[int, str] = {}
        for count in range(1, _MAX_COUNT + 1):
            for face in range(1, dice_sides + 1):
                a = _encode(count, face)
                if a > last_action:
                    labels[a] = f"{count} {face}"
        if bids:
            labels[challenge_action] = vocab["CHALLENGE"]
        return labels_in_display_order(sp, labels)

    def belief_suffix(info_str: str) -> str:
        """Suffix appended to build_prompt for the sibling belief call."""
        sp, _vocab = _content()
        own_die = _parse_own_die(info_str)
        return sp.belief_question.format(
            own_die=own_die,
            belief_exemplar=sp.belief_exemplar,
            belief_keys=quoted_keys([str(f) for f in range(1, dice_sides + 1)]),
        )

    def parse_action(text: str, info_str: str) -> Optional[int]:
        """
        parse an action index (0..challenge_action) from model text, or return None
        Two checks apply. A completion naming two or more of the bid labels `action_labels`
        offers at this info-set, for example an echo of `action_suffix`, which lists every legal
        bid plus CHALLENGE, is rejected before any other matching (see
        `spec.has_multiple_labels`); multi-word labels such as "1 4" match as whole phrases.
        Then the COUNT FACE regex and the bare-digit fallback accept only an action that is
        offered here (`a in labels`), not any text that merely matches the pattern. Without that
        gate, a bid mentioned in the text but not legal at this node, for example any bid at the
        maximum-bid node where only CHALLENGE is legal, would decode to a bogus action number.
        A CHALLENGE mention is not gated: it decodes to `challenge_action` wherever it appears,
        including the opening node where challenging is not legal, so the caller must check the
        result against the legal actions.
        :param text: the model's completion (str)
        :param info_str: the OpenSpiel info-state string for this node (str)
        :returns: the action index, or None if the text is unparseable; a bid or bare digit is
            returned only if it is legal here, a CHALLENGE without any legality check (Optional[int])
        """
        _sp, vocab = _content()
        t = text.strip().upper()
        labels = action_labels(info_str)
        if has_multiple_labels(t, labels.values()):
            return None

        # Challenge
        challenge_pattern = "|".join(
            _word_pattern(lab)
            for lab in [vocab["CHALLENGE"], *vocab["parse_synonyms"]["CHALLENGE"]]
        )
        if re.search(challenge_pattern, t):
            return challenge_action

        # Bid: "COUNT FACE" pattern (e.g. "2 5", "1 3") -- only if actually legal
        # here. Count/face ranges are generated from `_MAX_COUNT`/`dice_sides`
        # (single-digit character classes -- see the `dice_sides` check
        # above), not hardcoded, so this generalizes past the canonical 1-6.
        m = re.search(rf"\b([1-{_MAX_COUNT}])\s+([1-{dice_sides}])\b", t)
        if m:
            count, face = int(m.group(1)), int(m.group(2))
            a = _encode(count, face)
            if a in labels:
                return a

        # Bare action digit 0–12 -- only if actually legal here.
        m = re.search(r"\b(\d+)\b", t)
        if m:
            a = int(m.group(1))
            if a in labels:
                return a

        return None

    def parse_belief(text: str, info_str: str) -> Optional[str]:
        """Parse a belief distribution over opponent die values (1..dice_sides).

        Returns JSON string '{"1": p1, ..., "<dice_sides>": p_n}' normalised to
        sum 1.0. Dice are independent so no face is excluded. Returns None if
        unparseable. `dice_sides` is this closure's own parameter: a
        module-level ``_DICE_SIDES=6`` bound at import time would silently
        mis-bound every non-canonical variant's belief parsing to the wrong
        face count.
        """
        raw = load_belief_object(text)
        if raw is None:
            return None

        probs: dict[int, float] = {}
        for k, v in raw.items():
            try:
                face = int(str(k).strip())
            except ValueError:
                continue
            if 1 <= face <= dice_sides:
                number = finite_float(v)
                if number is not None:
                    probs[face] = number

        if not probs:
            return None

        total = sum(probs.values())
        if total <= 0.0:
            probs = {f: 1.0 / dice_sides for f in range(1, dice_sides + 1)}
            total = 1.0

        # Emit 0-indexed die-action keys ("0".."dice_sides-1") so the belief
        # aligns with true_posterior, which match_runner indexes by OpenSpiel
        # die action (action a = shown face a+1). The LM is prompted in face
        # values 1..dice_sides; we map face f -> action key str(f-1).
        # Misaligning these would corrupt every liarsdice P5/P7 TVD by one face.
        normalised = {f: probs.get(f, 0.0) / total for f in range(1, dice_sides + 1)}
        return json.dumps(
            {str(f - 1): float(normalised[f]) for f in range(1, dice_sides + 1)}
        )

    return {
        "build_prompt": build_prompt,
        "action_suffix": action_suffix,
        "action_labels": action_labels,
        "belief_suffix": belief_suffix,
        "parse_action": parse_action,
        "parse_belief": parse_belief,
        "parse_own_card": parse_own_card,
        "memory_terms": MEMORY_TERMS.scaled(
            1 if spec is None else spec.memory_payoff_scale
        ).with_labels(action_labels),
    }


# Canonical default instance -- module-level names keep their pre-factory
# signatures and behaviour, so existing imports and tests are untouched
# (mirrors darkhex.py/goofspiel.py's `_DEFAULT_FNS` block).
_DEFAULT_FNS = make_fns()
build_prompt = _DEFAULT_FNS["build_prompt"]
action_suffix = _DEFAULT_FNS["action_suffix"]
action_labels = _DEFAULT_FNS["action_labels"]
belief_suffix = _DEFAULT_FNS["belief_suffix"]
parse_action = _DEFAULT_FNS["parse_action"]
parse_belief = _DEFAULT_FNS["parse_belief"]
