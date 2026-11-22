"""Leduc-shaped universal_poker: info-state string -> NL prompt, action/belief
parsing. Depth-axis sibling of leduc.py -- same
2-round, 1-public-card, 2-suit shape as native leduc_poker, but
numRanks-parametric via universal_poker.

info-state format (confirmed; differs from native leduc_poker's
bracket format, and from kuhn_deep.py's single-round shape):
  [Round R][Player: P][Pot: N][Money: A B][Private: RS][Public: XY][Sequences: seq]
Round is 0-indexed (0 = round 1, 1 = round 2) -- differs from native
leduc_poker, whose `_parse_round` assumes 1-indexed rounds; do not reuse that
off-by-one assumption. Private/Public are poker-notation tokens (rank char +
suit char); numSuits=2 here, matching native Leduc's two-suits-per-rank deck,
so suit is present in the token but never decision-relevant (confirmed via
scoring/leduc_rank.py's existing rank = card_index // 2 invariant, which this
module's belief output relies on).

Sequences uses a pipe separator between rounds once round 2 begins (e.g.
"cc|c" = round-0 sequence "cc", round-1 sequence "c"; before the pipe appears,
the whole field is round 0's sequence) -- confirmed by walking a real 2-round
hand. Letters observed: 'c' (check/call), 'r' (bet/raise -- 'b' never
appears, same finding as kuhn_deep.py).

Action IDS (confirmed by running the game; three branches unlike
kuhn_deep's two -- Leduc's maxRaises=2 allows one raise after the opening
bet, matching native leduc_poker's own "check, bet, raise -- then no more
raises" rule):
  - Not facing a bet (first-to-act in the round): legal_actions() = [1, 2] --
    1 = "Call" (the check-equivalent), 2 = "Bet".
  - Facing a bet, a raise still available (fewer than 2 wagers so far this
    round -- opening bet counts as the first): legal_actions() = [0, 1, 2] --
    0 = Fold, 1 = Call, 2 = "Bet" (universal_poker's action_to_string calls
    this "Bet" even when facing a bet; this module relabels it RAISE to
    match LEDUC_SPEC's vocabulary, mirroring native leduc.py's own
    FOLD/CALL/RAISE facing-bet vocabulary -- the model is never shown "BET"
    at a facing-bet node, matching leduc.py's action_suffix precedent).
  - Facing the final bet (2 wagers already made this round, maxRaises hit):
    legal_actions() = [0, 1] -- 0 = Fold, 1 = Call, no raise offered.
This module's action_labels/parse_action return these real, branch-dependent
ids, never a hardcoded {0, 1, 2} triple.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from lmopt.prompting.memory import MemoryTerms
from lmopt.prompting.spec import LEDUC_SPEC as _SPEC
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
# characters from this string for a numRanks-rank deck (same table
# kuhn_deep.py uses).
_RANK_CHARS = "23456789TJQKA"


def _word_pattern(label: str) -> str:
    return r"\b" + re.escape(label) + r"\b"


def _parse_field(info_str: str, field: str) -> Optional[str]:
    m = re.search(r"\[" + re.escape(field) + r":\s*([^\]]*)\]", info_str)
    return m.group(1).strip() if m else None


def parse_own_card(info_str: str) -> Optional[str]:
    """Extract own card's rank character (not the index -- universal_poker's
    deal is card-token-keyed; suit is discarded, matching leduc_rank.py's
    rank = card_index // 2 invariant)."""
    val = _parse_field(info_str, "Private")
    if val is None or len(val) < 2:
        return None
    return val[0]


def _parse_round(info_str: str) -> int:
    """
    the 0-indexed round from the info state's Round field
    :param info_str: the info-state string (str)
    :returns: the round (int)
    :raises ValueError: when the Round field is missing
    """
    m = re.search(r"\[Round (\d+)\]", info_str)
    if m is None:
        raise ValueError(
            f"info-state string has no parseable Round field: {info_str!r}"
        )
    return int(m.group(1))


def _parse_public_card(info_str: str) -> Optional[str]:
    """RANK character of the revealed community card, or None before round 1."""
    val = _parse_field(info_str, "Public")
    if val is None or len(val) < 2:
        return None
    return val[0]


def _parse_pot(info_str: str) -> int:
    """
    the pot from the info state's Pot field
    :param info_str: the info-state string (str)
    :returns: the pot (int)
    :raises ValueError: when the Pot field is missing or not a whole number
    """
    val = _parse_field(info_str, "Pot")
    if not val or not val.isdigit():
        raise ValueError(f"info-state string has no parseable Pot field: {info_str!r}")
    return int(val)


def _current_round_seq(info_str: str, current_round: int) -> str:
    """The current round's betting sequence.

    The Sequences field is pipe-separated once round 1 (0-indexed) begins
    (confirmed: "cc|c" for round-0 seq "cc", round-1 seq "c"); before the
    pipe appears, the whole field is round 0's sequence. Splitting
    unconditionally on '|' and indexing by current_round handles both: at
    round 0 with no pipe yet, split gives a 1-element list and index 0 is
    the whole (possibly empty) field.
    :param info_str: the info-state string (str)
    :param current_round: the 0-indexed round (int)
    :returns: the current round's sequence, "" when it has not begun (str)
    :raises ValueError: when the Sequences field is absent or has no part for the round
    """
    val = _parse_field(info_str, "Sequences")
    if val is None:
        raise ValueError(f"info-state string has no Sequences field: {info_str!r}")
    parts = val.split("|")
    if current_round >= len(parts):
        raise ValueError(
            f"Sequences field has {len(parts)} part(s), none for round {current_round}: {info_str!r}"
        )
    return parts[current_round]


def _facing_bet(seq: str) -> bool:
    seq = seq.strip()
    return bool(seq) and seq[-1] in ("b", "r")


def _raise_available(seq: str) -> bool:
    """True if fewer than 2 total wagers have been made this round (the
    opening bet counts as the first; maxRaises=2 allows exactly one raise
    after it, confirmed empirically -- facing 'rr' offers no id 2)."""
    letter_bets = seq.count("b") + seq.count("r")
    return letter_bets < 2


def _describe_history(seq: str) -> str:
    seq = seq.strip()
    if not seq:
        return "No actions yet this round."
    letter_mapping = {"c": "checked/called", "r": "bet/raised", "f": "folded"}
    parts = [f"Player {i % 2} {letter_mapping.get(ch, ch)}" for i, ch in enumerate(seq)]
    return ". ".join(parts) + "."


def _describe_full_history(info_str: str, current_round: int) -> str:
    """Betting history for every round up to and including current_round,
    mirroring leduc.py's _describe_full_history -- a Round-1 (0-indexed:
    the second round) node must not hide Round 0's sequence."""
    if current_round <= 0:
        return _describe_history(_current_round_seq(info_str, current_round))
    val = _parse_field(info_str, "Sequences") or ""
    parts_raw = val.split("|")
    parts = []
    for r in range(current_round + 1):
        seq = parts_raw[r] if r < len(parts_raw) else ""
        if seq:
            parts.append(f"Round {r + 1}: {_describe_history(seq)}")
        else:
            parts.append(f"Round {r + 1}: no actions yet.")
    return " ".join(parts)


# What the memory block calls a play and what is won or lost, in the rules' own words.
MEMORY_TERMS = MemoryTerms("hand", "chips")


def make_fns(num_ranks: int, spec: Optional[GamePromptSpec] = None) -> dict:
    """
    Build the leduc_deep prompt and parse closures for one deck size.
    :param num_ranks: deck size, this game's depth dial, mirroring ``kuhn_deep.make_fns(num_ranks)``
        and ``goofspiel.make_fns(num_cards)``. The number of suits stays fixed at 2, native
        Leduc's own value; only the rank count varies (int)
    :param spec: an optional ``GamePromptSpec`` override, the probe injection point, mirroring
        ``leduc.make_fns`` and ``kuhn_deep.make_fns``; ``None`` uses the shared, templated
        ``LEDUC_SPEC`` (Optional[GamePromptSpec])
    :returns: a dict of the ``GamePromptFns`` fields (dict)
    :raises ValueError: when ``spec`` asks for a non-canonical option order, which this factory
        does not render, or when ``num_ranks`` is outside 2..13
    """
    if not 2 <= num_ranks <= len(_RANK_CHARS):
        raise ValueError(
            f"leduc_deep num_ranks={num_ranks} is outside 2..{len(_RANK_CHARS)}, the ranks "
            f"the rank alphabet {_RANK_CHARS!r} can name"
        )
    require_canonical_option_order(spec, "leduc_deep.make_fns")

    def _content() -> tuple[GamePromptSpec, Mapping[str, Any]]:
        sp = spec if spec is not None else _SPEC
        return sp, sp.action_vocabulary

    def _rank_label(rank_char: str) -> str:
        """Generic depth-variant label (goofspiel/darkhex/kuhn_deep
        precedent): the rank char itself is the label."""
        return rank_char

    def build_prompt(info_str: str, current_player: int) -> str:
        sp, vocab = _content()
        own_rank = parse_own_card(info_str)
        if own_rank is None:
            raise ValueError(
                f"build_prompt: info-state string has no parseable Private field: {info_str!r}"
            )
        current_round = _parse_round(info_str)
        public_rank = _parse_public_card(info_str) if current_round >= 1 else None
        # See leduc.py's build_prompt: the pot is a payoff magnitude with no
        # template phrase to whitelist, so it scales through display_scale.
        pot = _parse_pot(info_str) * sp.display_scale
        seq = _current_round_seq(info_str, current_round)
        facing = _facing_bet(seq)
        history_desc = _describe_full_history(info_str, current_round)
        can_raise = _raise_available(seq)

        rules = sp.instruction["rules"].format(
            num_cards=str(num_ranks * 2),
            rank_list_desc=(
                f"ranked {_RANK_CHARS[0]} (lowest) through "
                f"{_RANK_CHARS[num_ranks - 1]} (highest)"
            ),
        )

        round_desc = f"Round {current_round + 1}"
        if current_round >= 1 and public_rank is not None:
            round_desc += f" (community card: {_rank_label(public_rank)})"

        if facing:
            if can_raise:
                situation = "You are facing a bet and must respond."
                options_desc = sp.instruction["options_facing_raise"].format(
                    FOLD=vocab["FOLD"], CALL=vocab["CALL"], RAISE=vocab["RAISE"]
                )
            else:
                situation = "You are facing the final bet (no more raises allowed)."
                options_desc = sp.instruction["options_facing_final"].format(
                    FOLD=vocab["FOLD"], CALL=vocab["CALL"]
                )
        else:
            situation = "It is your turn to act first this round."
            options_desc = sp.instruction["options_first"].format(
                CHECK=vocab["CHECK"], BET=vocab["BET"]
            )

        return (
            f"{rules}\n\n"
            f"Current state:\n"
            f"- You are Player {current_player}.\n"
            f"- Your private card: {_rank_label(own_rank)}\n"
            f"- {round_desc}\n"
            f"- Pot: {pot} chips.\n"
            f"- History: {history_desc}\n"
            f"- {situation}\n\n"
            f"Your legal actions: {options_desc}"
        )

    def action_suffix(info_str: str) -> str:
        _sp, vocab = _content()
        current_round = _parse_round(info_str)
        seq = _current_round_seq(info_str, current_round)
        facing = _facing_bet(seq)
        can_raise = _raise_available(seq)

        if facing:
            not_bet = f" (do NOT say {vocab['BET']})"
            if can_raise:
                return legal_response_suffix(
                    [vocab["FOLD"], vocab["CALL"], vocab["RAISE"]], note=not_bet
                )
            return legal_response_suffix([vocab["FOLD"], vocab["CALL"]], note=not_bet)
        return legal_response_suffix([vocab["CHECK"], vocab["BET"]])

    def action_labels(info_str: str) -> dict[int, str]:
        """Legal action -> the exact literal action_suffix offers for it.

        Keys are the real universal_poker action ids at this branch (module
        docstring): {0: FOLD, 1: CALL, 2: RAISE} facing a bet with a raise
        available, {0: FOLD, 1: CALL} facing the final bet, {1: CHECK,
        2: BET} otherwise -- never a hardcoded triple. RAISE, never BET, is
        offered at a facing-bet node (universal_poker's own action_to_string
        says "Bet" there regardless; this is the relabel that matches
        LEDUC_SPEC's vocabulary and action_suffix's "do not say BET" text).
        """
        _sp, vocab = _content()
        current_round = _parse_round(info_str)
        seq = _current_round_seq(info_str, current_round)
        if _facing_bet(seq):
            if _raise_available(seq):
                return {0: vocab["FOLD"], 1: vocab["CALL"], 2: vocab["RAISE"]}
            return {0: vocab["FOLD"], 1: vocab["CALL"]}
        return {1: vocab["CHECK"], 2: vocab["BET"]}

    def belief_suffix(info_str: str) -> str:
        sp, _vocab = _content()
        own_rank = parse_own_card(info_str)
        if own_rank is None:
            raise ValueError(
                f"belief_suffix: info-state string has no parseable Private field: {info_str!r}"
            )
        current_round = _parse_round(info_str)
        public_rank = _parse_public_card(info_str) if current_round >= 1 else None

        all_ranks = list(_RANK_CHARS[:num_ranks])
        counts = {r: 2 for r in all_ranks}
        counts[own_rank] -= 1
        if public_rank is not None:
            counts[public_rank] -= 1
        total_unseen = sum(counts.values())

        deck_desc = "; ".join(
            f"{counts[r]} {_rank_label(r)}{'s' if counts[r] != 1 else ''}"
            for r in all_ranks
        )

        return sp.belief_question.format(
            own_name=_rank_label(own_rank),
            total_unseen=total_unseen,
            deck_desc=deck_desc,
            belief_exemplar=sp.belief_exemplar,
            num_ranks_word=str(num_ranks),
            belief_keys=quoted_keys([_rank_label(r) for r in all_ranks]),
        )

    def parse_action(text: str, info_str: str) -> Optional[int]:
        """Parse a legal action, returning the real universal_poker action id
        for the branch this info-set is in (see module docstring) -- never a
        hardcoded 0/1/2."""
        _sp, vocab = _content()
        t = text.strip().upper()
        if has_multiple_labels(t, action_labels(info_str).values()):
            return None
        current_round = _parse_round(info_str)
        seq = _current_round_seq(info_str, current_round)
        facing = _facing_bet(seq)
        can_raise = _raise_available(seq)

        if facing:
            if re.search(_word_pattern(vocab["FOLD"]), t):
                return 0
            if re.search(_word_pattern(vocab["CALL"]), t):
                return 1
            if can_raise and re.search(_word_pattern(vocab["RAISE"]), t):
                return 2
            if can_raise and re.search(_word_pattern(vocab["BET"]), t):
                return 2  # BET ~ RAISE when facing a bet; model conflates the two
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

    def parse_action_aliased(text: str, info_str: str) -> bool:
        """True iff parse_action resolved this text via the BET->RAISE alias
        (mirrors leduc.py's own alias/logging contract exactly)."""
        _sp, vocab = _content()
        t = text.strip().upper()
        current_round = _parse_round(info_str)
        seq = _current_round_seq(info_str, current_round)
        if not (_facing_bet(seq) and _raise_available(seq)):
            return False
        if (
            re.search(_word_pattern(vocab["FOLD"]), t)
            or re.search(_word_pattern(vocab["CALL"]), t)
            or re.search(_word_pattern(vocab["RAISE"]), t)
        ):
            return False
        return bool(re.search(_word_pattern(vocab["BET"]), t))

    def parse_belief(text: str, info_str: str) -> Optional[str]:
        """Parse a belief distribution from model text.

        Returns a rank-space dict (keys "0".."num_ranks-1"), matching
        leduc.py's own parse_belief convention -- not the raw 2*num_ranks
        card-index space true_posterior is logged in. The bridge between the
        two lives in scoring/leduc_rank.py (marginalise_to_rank/
        expand_rank_to_indices, num_ranks-parametric for exactly this
        reason), not here: parse_belief's contract is "what
        rank does the model believe", same as the canonical game.
        """
        own_rank = parse_own_card(info_str)
        if own_rank is None:
            return None
        raw = load_belief_object(text)
        if raw is None:
            return None

        all_ranks = list(_RANK_CHARS[:num_ranks])
        rank_index = {r: i for i, r in enumerate(all_ranks)}
        probs: dict[int, float] = {}
        for k, v in raw.items():
            k_str = str(k).strip()
            if k_str in rank_index:
                number = finite_float(v)
                if number is not None:
                    probs[rank_index[k_str]] = number
        if not probs:
            return None

        total = sum(probs.values())
        if total <= 0.0:
            share = 1.0 / num_ranks
            probs = {i: share for i in range(num_ranks)}
        else:
            probs = {k: v / total for k, v in probs.items()}

        result = {str(i): float(probs.get(i, 0.0)) for i in range(num_ranks)}
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
        "parse_action_aliased": parse_action_aliased,
    }
