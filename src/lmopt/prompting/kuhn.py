"""Kuhn poker: info-state string -> NL prompt, action/belief parsing.

Parametric on prompt content: ``make_fns(spec)`` builds the
prompt/parse closures for one ``GamePromptSpec``, and the module-level names
are that factory applied to this module's own ``_SPEC``. Darkhex/goofspiel's
factories parameterise on board/deck shape; those axes are orthogonal and Kuhn
has no shape axis, so the spec is this game's only factory parameter.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from lmopt.prompting.memory import MemoryTerms
from lmopt.prompting.spec import KUHN_SPEC as _SPEC
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

CARD_NAMES = {0: "Jack", 1: "Queen", 2: "King"}
_CARD_FROM_NAME = {v.upper(): k for k, v in CARD_NAMES.items()}

# Both `_SPEC` (above) and `_VOCAB` are import-time bindings -- `_SPEC` binds
# to the `KUHN_SPEC` object the `spec` module's global pointed at when this
# module was imported, and `_VOCAB` binds to that object's
# `.action_vocabulary` dict at the same moment. Monkeypatching
# `spec.KUHN_SPEC` after import changes neither `build_prompt` nor `_VOCAB` --
# `build_prompt` reads this module's `_SPEC` name, not `spec.KUHN_SPEC`, so
# re-pointing the latter is invisible to both. A test wanting to mutate spec
# content must patch the module-level `_SPEC` and/or `_VOCAB` attributes of
# This game module (e.g. `kuhn._SPEC`, as the production-entrypoint test in
# tests/test_prompt_spec.py does), never `spec.KUHN_SPEC`. That contract
# holds for the factory too: the default instance's closures read these
# two module attributes at CALL time (see `make_fns`'s `_content`), exactly as
# the pre-factory module-level functions did.
_VOCAB = _SPEC.action_vocabulary


def _word_pattern(label: str) -> str:
    """A `\\bLABEL\\b` regex pattern for one vocabulary literal."""
    return r"\b" + re.escape(label) + r"\b"


# ---------------------------------------------------------------------------
# Spec-independent info-set parsing (shared by every spec instance)
# ---------------------------------------------------------------------------


def _facing_bet(info_str: str) -> bool:
    """True if the current player is responding to a bet."""
    history = info_str[1:]
    return bool(history) and history[-1] == "b"


def _describe_history(info_str: str) -> str:
    history = info_str[1:]
    if not history:
        return "No actions yet."
    parts = []
    p = 0  # Player 0 always acts first
    for ch in history:
        verb = "checked" if ch == "p" else "bet"
        parts.append(f"Player {p} {verb}")
        p = 1 - p
    return ". ".join(parts) + "."


def parse_own_card(info_str: str) -> int:
    """Extract own card index from Kuhn info-state (first character)."""
    return int(info_str[0])


def parse_belief(text: str, own_card_or_info_str) -> Optional[str]:
    """Parse a belief distribution from model text.

    Accepts own_card as int (legacy) or info_str as str (multi-game API).
    Returns a JSON string '{"0": p, "1": q, "2": r}' normalised to sum 1.0,
    with own_card entry = 0.  Returns None if the response is unparseable.

    Spec-independent: the belief key space is Kuhn's card names,
    not the action vocabulary, so this needs no per-spec instance and stays a
    plain module function shared by every ``make_fns`` instance.
    """
    own_card = (
        own_card_or_info_str
        if isinstance(own_card_or_info_str, int)
        else parse_own_card(own_card_or_info_str)
    )
    raw = load_belief_object(text)
    if raw is None:
        return None

    probs: dict[int, float] = {}
    for k, v in raw.items():
        k_str = str(k).strip().upper()
        if k_str in _CARD_FROM_NAME:
            idx = _CARD_FROM_NAME[k_str]
        elif k_str.isascii() and k_str.isdigit():
            idx = int(k_str)
        else:
            continue
        if 0 <= idx <= 2:
            number = finite_float(v)
            if number is not None:
                probs[idx] = number

    if not probs:
        return None

    probs[own_card] = 0.0
    total = sum(probs.values())
    if total <= 0.0:
        remaining = [c for c in range(3) if c != own_card]
        probs = {own_card: 0.0, remaining[0]: 0.5, remaining[1]: 0.5}
    else:
        probs = {k: v / total for k, v in probs.items()}

    # Ensure all three entries are present
    result = {str(c): float(probs.get(c, 0.0)) for c in range(3)}
    return json.dumps(result)


# ---------------------------------------------------------------------------
# The spec-parametric factory
# ---------------------------------------------------------------------------

# What the memory block calls a play and what is won or lost, in the rules' own words.
MEMORY_TERMS = MemoryTerms("hand", "chips")


def make_fns(spec: Optional[GamePromptSpec] = None) -> dict:
    """
    Build the Kuhn prompt and parse closures for one prompt spec.
    :param spec: the ``GamePromptSpec`` whose instruction prose, action vocabulary and belief
        question these closures render, or None (the default, used for the module-level
        instance) to read this module's ``_SPEC`` and ``_VOCAB`` at call time, which keeps the
        monkeypatch contract described above: patching ``kuhn._SPEC`` still changes what a live
        match renders. A spec passed explicitly is bound here, so a later patch of the module
        attributes does not reach it (Optional[GamePromptSpec])
    :returns: a dict of the ``GamePromptFns`` fields, the same shape as ``darkhex.make_fns`` and
        ``goofspiel.make_fns``; it is a dict rather than a ``GamePromptFns`` because that
        dataclass lives in ``lmopt.prompting.__init__``, which imports this module (dict)
    """

    def _content() -> tuple[GamePromptSpec, Mapping[str, Any]]:
        """(spec, action vocabulary) for this call -- see `spec` above."""
        if spec is None:
            return _SPEC, _VOCAB
        return spec, spec.action_vocabulary

    def build_prompt(info_str: str, current_player: int) -> str:
        """Return the base NL prompt for a Kuhn decision node (no action/belief suffix)."""
        sp, vocab = _content()
        own_card = int(info_str[0])
        card_name = CARD_NAMES[own_card]
        history_desc = _describe_history(info_str)
        facing = _facing_bet(info_str)

        if facing:
            situation = "You are facing a bet and must respond."
            options_desc = sp.instruction["options_facing_bet"].format(
                FOLD=vocab["FOLD"], CALL=vocab["CALL"]
            )
        elif not info_str[1:]:
            situation = "You act first."
            options_desc = sp.instruction["options_first"].format(
                CHECK=vocab["CHECK"], BET=vocab["BET"]
            )
        else:
            situation = "Your opponent checked. You act."
            options_desc = sp.instruction["options_after_check"].format(
                CHECK=vocab["CHECK"], BET=vocab["BET"]
            )

        # `instruction["rules"]` is now the shared template spec.py holds for
        # both kuhn.py (here, at the fixed canonical 3-card values) and
        # kuhn_deep.py (at its chosen numRanks) -- `.format()` fills it at
        # the fixed canonical values, so the rendered bytes stay
        # byte-identical to the pre-templating literal (a regression test
        # pins this).
        rules = sp.instruction["rules"].format(
            num_ranks="Three",
            rank_range_desc="Jack (lowest), Queen, King (highest)",
            n_spare="The third",
            spare_plural=" is",
        )
        return (
            f"{rules}\n\n"
            f"Current state:\n"
            f"- You are Player {current_player}.\n"
            f"- Your card: {card_name}\n"
            f"- History: {history_desc}\n"
            f"- {situation}\n\n"
            f"Your legal actions: {options_desc}"
        )

    def action_suffix(info_str: str) -> str:
        """Suffix appended to build_prompt for the action call."""
        sp, vocab = _content()
        if _facing_bet(info_str):
            return legal_response_suffix(
                in_display_order(sp, [vocab["FOLD"], vocab["CALL"]])
            )
        return legal_response_suffix(
            in_display_order(sp, [vocab["CHECK"], vocab["BET"]])
        )

    def action_labels(info_str: str) -> dict[int, str]:
        """Legal action -> the exact literal action_suffix offers for it.

        Mirrors action_suffix's own branching exactly:
        this is data the prompting layer already has, not text to be re-derived
        by parsing action_suffix's rendered string. Never call this to change
        what action_suffix renders -- it must stay byte-identical (see
        tests/test_prompt_fidelity.py's action-label agreement check, which
        verifies every label here appears verbatim in action_suffix's output
        over the full tree of every registered variant). Both this and
        action_suffix read the same vocabulary (`_content`; before
        that, the module-level `_VOCAB` directly), so the literal text cannot
        drift between the two representations -- including under an injected
        variant spec, since both read the same closure.
        """
        sp, vocab = _content()
        if _facing_bet(info_str):
            return labels_in_display_order(sp, {0: vocab["FOLD"], 1: vocab["CALL"]})
        return labels_in_display_order(sp, {0: vocab["CHECK"], 1: vocab["BET"]})

    def _belief_suffix_for_card(own_card: int) -> str:
        """Internal: build belief suffix given own card index.

        `belief_exemplar` is now a fixed literal string with
        placeholder (unmappable) keys (spec.py) -- it no longer varies with
        `own_card` (the pre-fix version built a per-card `example` naming the
        two real "remaining" card names, which is exactly what made an echo of
        belief_suffix's text self-parse: see spec.py's KUHN_SPEC comment). No
        rendering needed here any more; interpolated directly.
        """
        sp, _vocab = _content()
        own_name = CARD_NAMES[own_card]
        # belief_question now also templates
        # {num_other_cards} (spec.py); kuhn.py fills it at the fixed
        # canonical value "two" (kuhn_deep.py fills it at `num_ranks - 1`).
        return sp.belief_question.format(
            own_name=own_name,
            belief_exemplar=sp.belief_exemplar,
            num_other_cards="two",
            belief_keys=quoted_keys(
                [CARD_NAMES[c] for c in sorted(CARD_NAMES) if c != own_card]
            ),
        )

    def belief_suffix(info_str_or_card) -> str:
        """Suffix appended to build_prompt for the sibling belief call.

        Accepts either an info_state string (new multi-game API) or an int (legacy).
        """
        if isinstance(info_str_or_card, int):
            return _belief_suffix_for_card(info_str_or_card)
        return _belief_suffix_for_card(parse_own_card(info_str_or_card))

    def parse_action(text: str, info_str: str) -> Optional[int]:
        """Parse a legal action (0 or 1) from model text. Returns None if unparseable.

        Action parser hardening: a
        completion naming both legal labels at this info-set (e.g. an echo of
        action_suffix, which names both "CHECK or BET"/"FOLD or CALL") is
        rejected before any branch-specific matching -- see
        `spec.has_multiple_labels`'s docstring for why this does not affect a
        bare single label or a single label embedded in short prose.
        """
        _sp, vocab = _content()
        t = text.strip().upper()
        if has_multiple_labels(t, action_labels(info_str).values()):
            return None
        facing = _facing_bet(info_str)

        if facing:
            if re.search(_word_pattern(vocab["FOLD"]), t):
                return 0
            if re.search(_word_pattern(vocab["CALL"]), t):
                return 1
        else:
            check_pattern = "|".join(
                _word_pattern(lab)
                for lab in [vocab["CHECK"], *vocab["parse_synonyms"]["CHECK"]]
            )
            if re.search(check_pattern, t):
                return 0
            if re.search(_word_pattern(vocab["BET"]), t):
                return 1

        # Fallback: bare digit
        m = re.search(r"\b([01])\b", t)
        if m:
            return int(m.group(1))

        return None

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
