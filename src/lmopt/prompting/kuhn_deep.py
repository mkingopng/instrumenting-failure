"""Kuhn-shaped universal_poker: info-state string -> NL prompt, action/belief
parsing. Depth-axis sibling of kuhn.py -- same
1-round, no-board-card, single-private-card shape as native kuhn_poker, but
numRanks-parametric via universal_poker.

info-state format (differs from native kuhn_poker's bare
"<card><history>" string):
  [Round 0][Player: P][Pot: N][Money: A B][Private: RS][Public: ][Sequences: seq]
Private RS is a poker-notation token: rank char + suit char (numSuits=1 here, so
suit is constant and not shown to the model). Round is always 0 (single round).

Action IDS (confirmed by running the game; they are not plain 0/1 the way
native kuhn_poker's are). universal_poker's limit-betting action ids are not
context-independent:

  - Not facing a bet (first-to-act, or after an opponent check): legal_actions()
    is [1, 2] -- 1="Call" (the check-equivalent: matching an outstanding bet of
    0), 2="Bet".
  - Facing a bet: legal_actions() is [0, 1] -- 0="Fold", 1="Call".

So action id 1 means "Call" in both branches but is the CHECK-equivalent
label in the first and the actual call in the second; id 0 only exists in the
facing-bet branch (Fold); id 2 only exists in the not-facing-bet branch (Bet).
This module's action_labels/parse_action return these real ids (branch-
dependent), never a hardcoded {0, 1} -- returning the wrong id would make
run_match's ``legal_mask[parsed_action]``/``state.apply_action(parsed_action)``
either raise or silently apply the wrong action.

Sequence characters observed at this game's config (limit betting, maxRaises=1,
1 round): 'c' (check/call), 'r' (bet/raise -- 'b' never appears), 'f' (fold,
terminal so never mid-sequence).
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
    legal_response_suffix,
    load_belief_object,
    quoted_keys,
    require_canonical_option_order,
)

_VOCAB = _SPEC.action_vocabulary

# Standard poker rank ordering; universal_poker allocates the LOWEST numRanks
# characters from this string for a numRanks-card deck.
_RANK_CHARS = "23456789TJQKA"


def _word_pattern(label: str) -> str:
    return r"\b" + re.escape(label) + r"\b"


def _parse_field(info_str: str, field: str) -> Optional[str]:
    m = re.search(r"\[" + re.escape(field) + r":\s*([^\]]*)\]", info_str)
    return m.group(1).strip() if m else None


def parse_own_card(info_str: str) -> Optional[str]:
    """Extract own card's rank character (not an index -- universal_poker's deal
    is card-token-keyed, not integer-index-keyed, unlike native kuhn_poker)."""
    val = _parse_field(info_str, "Private")
    if val is None or len(val) < 2:
        return None
    return val[0]


def _facing_bet(seq: str) -> bool:
    seq = seq.strip()
    return bool(seq) and seq[-1] in ("b", "r")


def _describe_history(seq: str) -> str:
    seq = seq.strip()
    if not seq:
        return "No actions yet."
    # Mirrors leduc.py's own combined "checked/called"/"bet/raised" verbs
    # (leduc's action codes are equally context-dependent -- 0/1/2 there,
    # c/r here): a single sequence character means different concrete actions
    # depending on the info-set it appears at, so the description states both.
    letter_mapping = {"c": "checked/called", "r": "bet/raised", "f": "folded"}
    parts = [f"Player {i % 2} {letter_mapping.get(ch, ch)}" for i, ch in enumerate(seq)]
    return ". ".join(parts) + "."


# What the memory block calls a play and what is won or lost, in the rules' own words.
MEMORY_TERMS = MemoryTerms("hand", "chips")


def make_fns(num_ranks: int, spec: Optional[GamePromptSpec] = None) -> dict:
    """
    Build the kuhn_deep prompt and parse closures for one deck size.
    :param num_ranks: deck size, this game's depth dial, mirroring ``goofspiel.make_fns(num_cards)`` (int)
    :param spec: an optional ``GamePromptSpec`` override, the probe injection point, mirroring
        ``kuhn.make_fns`` and ``darkhex.make_fns``; ``None`` uses the shared, templated
        ``KUHN_SPEC`` (Optional[GamePromptSpec])
    :returns: a dict of the ``GamePromptFns`` fields (dict)
    :raises ValueError: when ``spec`` asks for a non-canonical option order, which this factory
        does not render
    """
    require_canonical_option_order(spec, "kuhn_deep.make_fns")

    def _content() -> tuple[GamePromptSpec, Mapping[str, Any]]:
        sp = spec if spec is not None else _SPEC
        return sp, sp.action_vocabulary

    def _rank_label(rank_char: str) -> str:
        """Generic depth-variant label (goofspiel/darkhex precedent: numbers,
        not invented names) -- the rank char itself is the label; a model reads
        poker notation directly (e.g. "T" = Ten) without translation."""
        return rank_char

    def build_prompt(info_str: str, current_player: int) -> str:
        sp, vocab = _content()
        own_rank = parse_own_card(info_str)
        if own_rank is None:
            raise ValueError(
                f"build_prompt: info-state string has no parseable Private field: {info_str!r}"
            )
        seq = _parse_field(info_str, "Sequences") or ""
        facing = _facing_bet(seq)
        history_desc = _describe_history(seq)

        rules = sp.instruction["rules"].format(
            num_ranks=str(num_ranks),
            rank_range_desc=(
                f"ranked {_RANK_CHARS[0]} (lowest) through "
                f"{_RANK_CHARS[num_ranks - 1]} (highest)"
            ),
            n_spare=f"The other {num_ranks - 2}" if num_ranks > 3 else "The third",
            # KUHN_SPEC's template (spec.py) is fixed literal text
            # "{n_spare} card{spare_plural} not in play." -- "card" itself
            # never pluralizes, so the "s" for the plural noun has to travel
            # inside spare_plural too ("s are", not " are"), or a num_ranks>3
            # render reads "The other 11 card are not in play." (missing the
            # plural "s" on the noun); the templating tests only
            # cover the canonical (num_ranks=3) values.
            spare_plural="s are" if num_ranks > 3 else " is",
        )

        if facing:
            situation = "You are facing a bet and must respond."
            options_desc = sp.instruction["options_facing_bet"].format(
                FOLD=vocab["FOLD"], CALL=vocab["CALL"]
            )
        elif not seq:
            situation = "You act first."
            options_desc = sp.instruction["options_first"].format(
                CHECK=vocab["CHECK"], BET=vocab["BET"]
            )
        else:
            situation = "Your opponent checked. You act."
            options_desc = sp.instruction["options_after_check"].format(
                CHECK=vocab["CHECK"], BET=vocab["BET"]
            )

        return (
            f"{rules}\n\n"
            f"Current state:\n"
            f"- You are Player {current_player}.\n"
            f"- Your card: {_rank_label(own_rank)}\n"
            f"- History: {history_desc}\n"
            f"- {situation}\n\n"
            f"Your legal actions: {options_desc}"
        )

    def action_suffix(info_str: str) -> str:
        _sp, vocab = _content()
        seq = _parse_field(info_str, "Sequences") or ""
        if _facing_bet(seq):
            return legal_response_suffix([vocab["FOLD"], vocab["CALL"]])
        return legal_response_suffix([vocab["CHECK"], vocab["BET"]])

    def action_labels(info_str: str) -> dict[int, str]:
        """Legal action -> the exact literal action_suffix offers for it.

        Keys are the real universal_poker action ids at this branch (module
        docstring): {0: FOLD, 1: CALL} when facing a bet, {1: CHECK, 2: BET}
        otherwise -- never a hardcoded {0, 1}, which would disagree with
        ``state.legal_actions()`` and break run_match's legal-mask indexing.
        """
        _sp, vocab = _content()
        seq = _parse_field(info_str, "Sequences") or ""
        if _facing_bet(seq):
            return {0: vocab["FOLD"], 1: vocab["CALL"]}
        return {1: vocab["CHECK"], 2: vocab["BET"]}

    def belief_suffix(info_str: str) -> str:
        sp, _vocab = _content()
        own_rank = parse_own_card(info_str)
        if own_rank is None:
            raise ValueError(
                f"belief_suffix: info-state string has no parseable Private field: {info_str!r}"
            )
        return sp.belief_question.format(
            own_name=_rank_label(own_rank),
            belief_exemplar=sp.belief_exemplar,
            num_other_cards=str(num_ranks - 1),
            belief_keys=quoted_keys(
                [_rank_label(r) for r in range(num_ranks) if r != own_rank]
            ),
        )

    def parse_action(text: str, info_str: str) -> Optional[int]:
        """Parse a legal action from model text, returning the real
        universal_poker action id for the branch this info-set is in (see
        module docstring) -- never a hardcoded 0/1."""
        _sp, vocab = _content()
        t = text.strip().upper()
        if has_multiple_labels(t, action_labels(info_str).values()):
            return None
        seq = _parse_field(info_str, "Sequences") or ""
        facing = _facing_bet(seq)

        if facing:
            if re.search(_word_pattern(vocab["FOLD"]), t):
                return 0
            if re.search(_word_pattern(vocab["CALL"]), t):
                return 1
            m = re.search(r"\b([01])\b", t)
            if m:
                return int(m.group(1))
        else:
            check_pattern = "|".join(
                _word_pattern(lab)
                for lab in [vocab["CHECK"], *vocab["parse_synonyms"]["CHECK"]]
            )
            if re.search(check_pattern, t):
                return 1
            if re.search(_word_pattern(vocab["BET"]), t):
                return 2
            m = re.search(r"\b([12])\b", t)
            if m:
                return int(m.group(1))

        return None

    def parse_belief(text: str, info_str: str) -> Optional[str]:
        """Parse a belief distribution from model text.

        The model sees and answers in rank-character space (`build_prompt`/
        `belief_suffix` render "T"/"J"/"Q"/"K"/"A" etc., never an index), but
        the returned JSON is index-keyed ("0".."{num_ranks-1}"), matching
        `harness/match_runner.py`'s `true_posterior` encoding
        (`{str(i): p for i, p in enumerate(tp)}` over universal_poker's
        chance-outcome action ids) -- confirmed 1:1 with `_RANK_CHARS`'s
        position by construction (chance action 0 deals rank char
        `_RANK_CHARS[0]`, action 1 deals `_RANK_CHARS[1]`, etc., verified by
        walking every chance outcome directly).

        Rank-character keys in the result would be compared by `property5.py`/
        `property7.py` (routed to `kuhn`'s default belief-alignment path via
        `GAME_FAMILIES["kuhn_deep"] = "kuhn"`) directly against
        `true_posterior`'s index keys with no translation -- a perfect
        elicited belief would score TVD 0.4167 against itself, the same
        key-space-mismatch defect class `kuhn.py`'s own
        `parse_belief` (index-keyed result, name-or-digit-accepting input) and
        `goofspiel.py`'s action-key belief already exist to avoid.
        """
        own_rank = parse_own_card(info_str)
        if own_rank is None:
            return None
        raw = load_belief_object(text)
        if raw is None:
            return None

        all_ranks = list(_RANK_CHARS[:num_ranks])
        probs: dict[str, float] = {}
        for k, v in raw.items():
            k_str = str(k).strip()
            if k_str in all_ranks:
                number = finite_float(v)
                if number is not None:
                    probs[k_str] = number
        if not probs:
            return None

        probs[own_rank] = 0.0
        total = sum(probs.values())
        if total <= 0.0:
            remaining = [r for r in all_ranks if r != own_rank]
            share = 1.0 / len(remaining)
            probs = {own_rank: 0.0, **{r: share for r in remaining}}
        else:
            probs = {k: v / total for k, v in probs.items()}

        # Translate rank-char keys -> index-string keys (see docstring) only
        # at the final output boundary, so every upstream branch above still
        # reasons in rank-char space (own_rank, all_ranks).
        result = {
            str(_RANK_CHARS.index(r)): float(probs.get(r, 0.0)) for r in all_ranks
        }
        return json.dumps(result)

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
