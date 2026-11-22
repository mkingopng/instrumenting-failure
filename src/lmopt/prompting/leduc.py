"""Leduc poker: info-state string -> NL prompt, action/belief parsing.

Leduc info-state format (OpenSpiel, player-N view):
  [Observer: N][Private: C][Round R][Player: P][Pot: K][Money: A B][Round1: <seq>][Round2: <seq>]
  [Public: C] appears only in Round 2.

Action codes: 0=Fold, 1=Call/Check, 2=Raise/Bet

Sequence format - OpenSpiel emits space-separated integer codes:
  e.g. "[Round1: 2 2]" = first player bet (2), second player raised (2).
Letter-code sequences (b/r/p/c/f) appear only in test fixtures; both
formats are supported so tests remain readable without faking the game.

Parametric on prompt content: ``make_fns(spec)`` builds the
prompt/parse closures for one ``GamePromptSpec``, and the module-level names
are that factory applied to this module's own ``_SPEC``.
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
    in_display_order,
    labels_in_display_order,
    legal_response_suffix,
    load_belief_object,
    quoted_keys,
)

# Leduc deals six card indices (0-5): two of each rank, rank = index // 2 --
# {0,1}=Jack, {2,3}=Queen, {4,5}=King. Leduc is suit-symmetric (see
# scoring/leduc_rank.py's module docstring, which pins showdown outcome to
# floor(index / 2) against real OpenSpiel returns), so rank is the complete
# decision-relevant content of a card and the prompt is built in rank space.
# CARD_NAMES is keyed by rank (0, 1, 2), never by the raw 0-5 card index --
# callers must convert via `_rank(idx)` before indexing it.
CARD_NAMES = {0: "Jack", 1: "Queen", 2: "King"}
_CARD_FROM_NAME = {v.upper(): k for k, v in CARD_NAMES.items()}

# Both `_SPEC` (above) and `_VOCAB` are import-time bindings -- `_SPEC` binds
# to the `LEDUC_SPEC` object the `spec` module's global pointed at when this
# module was imported, and `_VOCAB` binds to that object's
# `.action_vocabulary` dict at the same moment. Monkeypatching
# `spec.LEDUC_SPEC` after import changes neither `build_prompt` nor `_VOCAB`
# -- `build_prompt` reads this module's `_SPEC` name, not `spec.LEDUC_SPEC`,
# so re-pointing the latter is invisible to both. A test wanting to mutate
# spec content must patch the module-level `_SPEC` and/or `_VOCAB`
# attributes of this game module (e.g. `leduc._SPEC`, as the
# production-entrypoint test in tests/test_prompt_spec.py does for kuhn),
# never `spec.LEDUC_SPEC`. That contract holds for the factory too: the
# default instance's closures read these module attributes at CALL time
# (see `make_fns`'s `_content`).
_VOCAB = _SPEC.action_vocabulary


def _word_pattern(label: str) -> str:
    """A `\\bLABEL\\b` regex pattern for one vocabulary literal."""
    return r"\b" + re.escape(label) + r"\b"


def _rank(card_idx: int) -> int:
    """Map an OpenSpiel Leduc card index (0-5) to its rank (0=Jack, 1=Queen, 2=King)."""
    return card_idx // 2


def _parse_field(info_str: str, field: str) -> Optional[str]:
    """Extract value from '[FieldName: value]' bracket."""
    m = re.search(r"\[" + re.escape(field) + r":\s*([^\]]*)\]", info_str)
    return m.group(1).strip() if m else None


def _scale_state(info_str: str, scale: int) -> str:
    """
    the info state with its pot and both stacks multiplied by a payoff-scale variant's factor,
    which is the scale the variant's prompt states them in
    :param info_str: the info-state string (str)
    :param scale: the variant's display scale (int)
    :returns: the string with the Pot and Money fields rescaled and nothing else changed (str)
    :raises ValueError: when the Pot or Money field is missing
    """
    pot = _parse_pot(info_str) * scale
    mine, theirs = (m * scale for m in _parse_money(info_str))
    scaled = re.sub(r"\[Pot:\s*[^\]]*\]", f"[Pot: {pot}]", info_str, count=1)
    return re.sub(
        r"\[Money:\s*[^\]]*\]", f"[Money: {mine} {theirs}]", scaled, count=1
    )


def parse_own_card(info_str: str) -> Optional[int]:
    """Extract own card index from the Private field."""
    val = _parse_field(info_str, "Private")
    if val is not None and val.isdigit():
        return int(val)
    return None


def _parse_round(info_str: str) -> int:
    # OpenSpiel format is "[Round N]" (space, no colon) - not parseable by _parse_field.
    m = re.search(r"\[Round\s+(\d+)\]", info_str)
    return int(m.group(1)) if m else 1


def _parse_public_card(info_str: str) -> Optional[int]:
    val = _parse_field(info_str, "Public")
    if val is not None and val.isdigit():
        return int(val)
    return None


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


def _parse_money(info_str: str) -> tuple[int, int]:
    """
    both seats' stacks from the info state's Money field
    :param info_str: the info-state string (str)
    :returns: seat 0's and seat 1's stacks (tuple[int, int])
    :raises ValueError: when the Money field is missing or is not two whole numbers
    """
    val = _parse_field(info_str, "Money")
    parts = val.split() if val else []
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"info-state string has no parseable Money field: {info_str!r}"
        )
    return int(parts[0]), int(parts[1])


def _current_round_seq(info_str: str, current_round: int) -> str:
    """Return the action sequence for the current round (may be empty)."""
    field = f"Round{current_round}"
    val = _parse_field(info_str, field)
    return val if val is not None else ""


def _facing_bet(seq: str) -> bool:
    """True if the last action in the round sequence was a bet or raise (unanswered)."""
    seq = seq.strip()
    if not seq:
        return False
    # Letter-code format (test fixtures): last char is 'b' (bet) or 'r' (raise).
    if seq[-1] in ("b", "r"):
        return True
    # Integer-code format (actual OpenSpiel): last space-separated token is '2'.
    return seq.split()[-1] == "2"


def _raise_available(seq: str) -> bool:
    """True if another raise is still allowed (< 2 bets/raises in the round so far)."""
    # Letter-code format
    letter_bets = seq.count("b") + seq.count("r")
    if letter_bets:
        return letter_bets < 2
    # Integer-code format: count '2' tokens
    return seq.split().count("2") < 2


def _describe_history(seq: str) -> str:
    seq = seq.strip()
    if not seq:
        return "No actions yet this round."
    tokens = seq.split()
    # Integer-code format: all tokens are single digits 0/1/2
    if all(t in ("0", "1", "2") for t in tokens):
        int_mapping = {"0": "folded", "1": "checked/called", "2": "bet/raised"}
        parts = [f"Player {i % 2} {int_mapping[t]}" for i, t in enumerate(tokens)]
        return ". ".join(parts) + "."
    # Letter-code format
    letter_mapping = {
        "p": "checked",
        "b": "bet",
        "c": "called",
        "r": "raised",
        "f": "folded",
    }
    parts = []
    for i, ch in enumerate(seq):
        verb = letter_mapping.get(ch, ch)
        parts.append(f"Player {i % 2} {verb}")
    return ". ".join(parts) + "."


def _describe_full_history(info_str: str, current_round: int) -> str:
    """Describe the betting history for every round up to and including
    `current_round`, not just the current one.

    Rendering only `_describe_history(_current_round_seq(info_str,
    current_round))` -- the current round's sequence alone -- would hide
    Round 1's sequence at a Round-2 node even though the info-set string
    carries it (e.g. "[Round1: 1 1][Round 2]" would render "No actions yet
    this round" although Round 1 had two actions; 19,211/57,408 collected
    rows, 33.5%, had that shape). Deliberately a sibling of
    `_current_round_seq`, not a
    change to it: `_current_round_seq` also drives `_facing_bet`/
    `_raise_available` for every action-legality caller (`action_suffix`,
    `action_labels`, `parse_action`, `parse_action_aliased`), and those must
    keep reading only the current round's sequence -- a player is never
    facing a bet from a round that has already closed.

    Round 2's own empty segment is worded "Round 2: no actions yet." rather
    than reusing `_describe_history("")`'s "No actions yet this round." --
    once a prompt legitimately describes more than one round (current_round
    >= 2), "this round" singular is ambiguous, and rows that hid a real
    Round-1 sequence used exactly that unqualified phrase at a Round-2 node,
    so this branch deliberately does not reproduce it.
    Scoped precisely: a genuinely empty Round 1 still renders "No actions
    yet this round." verbatim, via the `current_round <= 1` branch below --
    there is only one round to describe there, the phrase is unambiguous and
    correct, and Round 1 never has a hidden earlier round to drop. Do not
    overstate this as "the phrase never appears" -- it legitimately still
    does, at every genuinely-first-round-1 P0 info-set.
    """
    if current_round <= 1:
        # Use the parameter, not a hardcoded literal. `_parse_round` does not guarantee
        # >= 1 in general -- it returns `int(m.group(1))` verbatim for any
        # matched "[Round N]", so a literal "[Round 0]" would return 0; the
        # >= 1 guarantee holds only via its no-match default (never observed
        # from real OpenSpiel Leduc, which starts numbering rounds at 1: 0 of
        # 468 P0 info-sets have current_round == 0, single caller into this
        # function is `build_prompt`). So at current_round == 0 this branch is
        # Not value-identical to a hardcoded-`1` version (that would read the
        # `Round1` field; this reads `Round0`, which is absent -> "") --
        # unreachable in practice, not equal in general.
        return _describe_history(_current_round_seq(info_str, current_round))
    parts = []
    for r in range(1, current_round + 1):
        seq = _parse_field(info_str, f"Round{r}")
        seq = (seq or "").strip()
        if seq:
            parts.append(f"Round {r}: {_describe_history(seq)}")
        else:
            parts.append(f"Round {r}: no actions yet.")
    return " ".join(parts)


_ILLUSTRATIVE_BELIEF_EXAMPLE = _SPEC.belief_exemplar
"""Fixed FORMAT illustration for belief_suffix's JSON example -- deliberately
NOT computed from any info-set's actual deck counts. Computing `example` from
the SAME per-info-set `counts`/`total_unseen` the deck-composition prose
states would make the "example" the true answer at every info-set -- most
visibly at the 6 of 468 P0 info-sets where Round 1 is empty (the acting
player is first to act, so no evidence has arrived yet and the true posterior
is exactly the deck-conditional prior), where a model that copies the example
verbatim would score perfectly on P5/P7a
without reasoning, which confounds the very belief-elicitation measurement
this suffix exists to support. This constant is fixed across ALL info-sets so
it cannot coincide with a real deck-conditional prior: every ACTUAL prior over
the whole P0 tree is a permutation of exactly one of three fraction-triples
(round 1: {0.2, 0.4, 0.4}; round 2, public card a different rank: {0.25, 0.25,
0.5}; round 2, public card the acting player's own rank -- both copies of that
rank are then visible: {0.0, 0.5, 0.5}), none of which contain 0.3 or 0.5-and-
0.3-together, so {0.5, 0.3, 0.2} cannot match any of them under any
permutation. Verified directly over the full tree (not asserted from this
reasoning alone): comparing this literal example against each of the 468
P0 info-sets' actual (counts[r]/total_unseen) triples finds zero matches.
"""


# What the memory block calls a play and what is won or lost, in the rules' own words.
MEMORY_TERMS = MemoryTerms("hand", "chips")


def _memory_terms(spec: Optional[GamePromptSpec], action_labels) -> MemoryTerms:
    """
    the memory terms of one prompt spec: payoffs in the variant's units and, for a variant
    that scales the pot and stacks its prompt states, the replayed states scaled to match
    :param spec: the prompt spec, or None for the canonical game (Optional[GamePromptSpec])
    :param action_labels: the game's reply labeller (Callable)
    :returns: the terms (MemoryTerms)
    """
    terms = MEMORY_TERMS.scaled(
        1 if spec is None else spec.memory_payoff_scale
    ).with_labels(action_labels)
    if spec is not None and spec.display_scale != 1:
        scale = spec.display_scale
        terms = terms.with_state_view(lambda state: _scale_state(state, scale))
    return terms


def make_fns(spec: Optional[GamePromptSpec] = None) -> dict:
    """
    Build the Leduc prompt and parse closures for one prompt spec.
    :param spec: the ``GamePromptSpec`` whose instruction prose, action vocabulary, belief
        question and belief exemplar these closures render, or None (the default, used for the
        module-level instance) to read this module's ``_SPEC``, ``_VOCAB`` and
        ``_ILLUSTRATIVE_BELIEF_EXAMPLE`` at call time, which keeps the monkeypatch contract
        documented at ``_VOCAB``. A spec passed explicitly is bound here, so a later patch of
        the module attributes does not reach it (Optional[GamePromptSpec])
    :returns: a dict of the ``GamePromptFns`` fields, including the optional
        ``parse_action_aliased`` (Leduc is the only game with an alias path), with the same
        dict-not-dataclass convention as ``darkhex.make_fns`` and ``goofspiel.make_fns`` (dict)
    """

    def _content() -> tuple[GamePromptSpec, Mapping[str, Any]]:
        """(spec, action vocabulary) for this call -- see `spec` above."""
        if spec is None:
            return _SPEC, _VOCAB
        return spec, spec.action_vocabulary

    def _belief_exemplar():
        """The belief question's fixed FORMAT illustration for this call."""
        return _ILLUSTRATIVE_BELIEF_EXAMPLE if spec is None else spec.belief_exemplar

    def build_prompt(info_str: str, current_player: int) -> str:
        """Return the base NL prompt for a Leduc decision node.

        No silent fallbacks: an info-state string with no parseable Private
        field is a malformed/non-Leduc info-state -- a bug upstream in the
        caller, not a normal "model produced bad text" case -- so this raises
        rather than defaulting to card index 0 ("Jack"), as
        `parse_own_card(info_str) or 0` would silently do (doubly wrong, since
        `or 0` also swallows the legitimate value 0).
        """
        sp, vocab = _content()
        own_card_idx = parse_own_card(info_str)
        if own_card_idx is None:
            raise ValueError(
                f"build_prompt: info-state string has no parseable Private field: {info_str!r}"
            )
        own_name = CARD_NAMES.get(_rank(own_card_idx), "Unknown")
        current_round = _parse_round(info_str)
        public_card_idx = _parse_public_card(info_str)
        # The pot and both stacks are payoff magnitudes parsed out of the
        # info-state string, so no template phrase exists for the payoff-scale
        # whitelist to move; they scale through the spec's display_scale
        # instead, which is 1 for every canonical spec.
        pot = _parse_pot(info_str) * sp.display_scale
        money = _parse_money(info_str)
        my_chips = money[current_player] * sp.display_scale
        opp_chips = money[1 - current_player] * sp.display_scale
        seq = _current_round_seq(info_str, current_round)
        facing = _facing_bet(seq)
        history_desc = _describe_full_history(info_str, current_round)
        can_raise = _raise_available(seq)

        round_desc = f"Round {current_round}"
        if current_round == 2 and public_card_idx is not None:
            pub_name = CARD_NAMES.get(_rank(public_card_idx), "Unknown")
            round_desc += f" (community card: {pub_name})"

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

        # instruction["rules"] is the shared template spec.py holds for
        # both leduc.py (here, at the fixed canonical 6-card values) and
        # leduc_deep.py (at its chosen deck size) -- .format() fills it at
        # the fixed canonical values, so the rendered bytes are fixed (a
        # regression test pins them).
        rules = sp.instruction["rules"].format(
            num_cards="Six", rank_list_desc="Jack, Queen, King"
        )
        return (
            f"{rules}\n\n"
            f"Current state:\n"
            f"- You are Player {current_player}.\n"
            f"- Your private card: {own_name}\n"
            f"- {round_desc}\n"
            f"- Pot: {pot} chips. Your chips: {my_chips}. Opponent chips: {opp_chips}.\n"
            f"- History: {history_desc}\n"
            f"- {situation}\n\n"
            f"Your legal actions: {options_desc}"
        )

    def action_suffix(info_str: str) -> str:
        """Suffix appended to build_prompt for the action call."""
        sp, vocab = _content()
        current_round = _parse_round(info_str)
        seq = _current_round_seq(info_str, current_round)
        facing = _facing_bet(seq)
        can_raise = _raise_available(seq)

        if facing:
            not_bet = f" (do NOT say {vocab['BET']})"
            if can_raise:
                return legal_response_suffix(
                    in_display_order(
                        sp, [vocab["FOLD"], vocab["CALL"], vocab["RAISE"]]
                    ),
                    note=not_bet,
                )
            return legal_response_suffix(
                in_display_order(sp, [vocab["FOLD"], vocab["CALL"]]), note=not_bet
            )
        return legal_response_suffix(
            in_display_order(sp, [vocab["CHECK"], vocab["BET"]])
        )

    def action_labels(info_str: str) -> dict[int, str]:
        """Legal action -> the exact literal action_suffix offers for it.

        Mirrors action_suffix's own branching exactly:
        data the prompting layer already has, not text to be re-derived by
        parsing action_suffix's rendered string. Deliberately never returns
        "BET" for the facing-bet RAISE action even though parse_action accepts
        it as a logged alias -- the prompt's own vocabulary explicitly says
        "do not say BET" there, so "BET" is never a literal action_suffix
        offers. Reads the same vocabulary (`_content`) as
        action_suffix and parse_action, so the literal text cannot drift
        between the three representations, including under an injected
        variant spec.
        """
        sp, vocab = _content()
        current_round = _parse_round(info_str)
        seq = _current_round_seq(info_str, current_round)
        if _facing_bet(seq):
            if _raise_available(seq):
                return labels_in_display_order(
                    sp, {0: vocab["FOLD"], 1: vocab["CALL"], 2: vocab["RAISE"]}
                )
            return labels_in_display_order(sp, {0: vocab["FOLD"], 1: vocab["CALL"]})
        return labels_in_display_order(sp, {1: vocab["CHECK"], 2: vocab["BET"]})

    def belief_suffix(info_str: str) -> str:
        """Suffix appended to build_prompt for the sibling belief call.

        No silent fallbacks: see build_prompt's docstring
        -- an unparseable Private field raises rather than silently naming "Jack".

        States the per-rank deck composition (how many unseen Jacks/Queens/Kings
        remain, given the acting player's own card and any revealed public card)
        because the question is unanswerable from the label alone: Leduc deals
        cards uniformly at random, so the true rank marginal on the opponent's
        card is exactly proportional to these remaining counts. This prose
        states all three ranks, including the holder's own rank -- the opponent
        may hold its second copy, so it is not zeroed.

        The JSON example is a fixed format illustration (`_ILLUSTRATIVE_BELIEF_EXAMPLE`),
        not computed from this info-set's counts: using the real per-info-set
        deck fractions as the "example" would make it, at 6 of 468 P0
        info-sets (Round 1, first to act -- no evidence has arrived yet), the
        exact correct answer, letting a model that merely copies the example
        score perfectly without reasoning. The prompt explicitly labels it a
        format illustration, not a suggestion.
        """
        sp, _vocab = _content()
        own_card_idx = parse_own_card(info_str)
        if own_card_idx is None:
            raise ValueError(
                f"belief_suffix: info-state string has no parseable Private field: {info_str!r}"
            )
        own_rank = _rank(own_card_idx)
        own_name = CARD_NAMES.get(own_rank, "Unknown")

        current_round = _parse_round(info_str)
        public_card_idx = _parse_public_card(info_str) if current_round == 2 else None

        counts = {r: 2 for r in range(3)}
        counts[own_rank] -= 1
        if public_card_idx is not None:
            counts[_rank(public_card_idx)] -= 1
        total_unseen = sum(counts.values())

        deck_desc = "; ".join(
            f"{counts[r]} {CARD_NAMES[r]}{'s' if counts[r] != 1 else ''}"
            for r in range(3)
        )

        return sp.belief_question.format(
            own_name=own_name,
            total_unseen=total_unseen,
            deck_desc=deck_desc,
            belief_exemplar=_belief_exemplar(),
            num_ranks_word="three",
            belief_keys=quoted_keys([CARD_NAMES[r] for r in range(3)]),
        )

    def parse_action(text: str, info_str: str) -> Optional[int]:
        """Parse a legal action (0, 1, or 2) from model text. Returns None if unparseable.

        Action parser hardening: a
        completion naming 2+ of the labels action_labels() offers at this
        info-set (e.g. an echo of action_suffix, which names every offered
        word) is rejected before any branch-specific matching -- see
        `spec.has_multiple_labels`'s docstring. Checked against action_labels()
        (never including "BET" at a facing-bet node, since that word is offered
        nowhere by the prompt there -- action_suffix explicitly says "do not say
        BET"), not the raw BET->RAISE alias literal below, so the alias path
        itself is untouched: a completion saying only "BET" at a facing-bet node
        still has just 1 label match (via the alias branch, not action_labels)
        and still parses via `parse_action_aliased`'s exact mechanism.
        """
        _sp, vocab = _content()
        t = text.strip().upper()
        current_round = _parse_round(info_str)
        seq = _current_round_seq(info_str, current_round)
        if has_multiple_labels(t, action_labels(info_str).values()):
            return None
        facing = _facing_bet(seq)

        if facing:
            if re.search(_word_pattern(vocab["FOLD"]), t):
                return 0
            if re.search(_word_pattern(vocab["CALL"]), t):
                return 1
            if re.search(_word_pattern(vocab["RAISE"]), t):
                return 2
            if re.search(_word_pattern(vocab["BET"]), t):
                return 2  # BET ≈ RAISE when facing a bet; model conflates the two
        else:
            check_pattern = "|".join(
                _word_pattern(lab)
                for lab in [vocab["CHECK"], *vocab["parse_synonyms"]["CHECK"]]
            )
            if re.search(check_pattern, t):
                return 1
            if re.search(_word_pattern(vocab["BET"]), t):
                return 2

        # Fallback: bare digit 0/1/2
        m = re.search(r"\b([012])\b", t)
        if m:
            return int(m.group(1))

        return None

    def parse_action_aliased(text: str, info_str: str) -> bool:
        """True iff parse_action resolved this text via the BET->RAISE alias.

        The alias fires only at a facing-bet node where the model said "BET" and did
        not say FOLD/CALL/RAISE - parse_action then returns 2 (RAISE) because in
        OpenSpiel Leduc "bet" and "raise" are the same action index. The executed
        action is correct, but the model's literal vocabulary ("BET") diverges from
        the canonical token ("RAISE"). This is logged via the ``bet_aliased`` Parquet
        column so the silent normalisation is auditable (no silent
        fallbacks). Mirrors the facing branch of parse_action exactly so
        the two cannot drift.
        """
        _sp, vocab = _content()
        t = text.strip().upper()
        current_round = _parse_round(info_str)
        seq = _current_round_seq(info_str, current_round)
        if not _facing_bet(seq):
            return False
        if (
            re.search(_word_pattern(vocab["FOLD"]), t)
            or re.search(_word_pattern(vocab["CALL"]), t)
            or re.search(_word_pattern(vocab["RAISE"]), t)
        ):
            return False
        return bool(re.search(_word_pattern(vocab["BET"]), t))

    return {
        "build_prompt": build_prompt,
        "action_suffix": action_suffix,
        "action_labels": action_labels,
        "belief_suffix": belief_suffix,
        "parse_action": parse_action,
        "parse_belief": parse_belief,
        "parse_own_card": parse_own_card,
        "memory_terms": _memory_terms(spec, action_labels),
        "parse_action_aliased": parse_action_aliased,
    }


def parse_belief(text: str, info_str: str) -> Optional[str]:
    """Parse a belief distribution from model text.

    Returns JSON string '{"0": p, "1": q, "2": r}' (rank-keyed 3-simplex)
    normalised to sum 1.0. Returns None if unparseable.

    No rank is ever zeroed. Forcing `probs[own_card] = 0.0` with the raw
    0-5 Private card index as if it were a rank KEY would be wrong for
    own_card in {1, 2} (zeroing the wrong rank: 1 -> Queen when card 1 is a
    Jack; 2 -> King when card 2 is a Queen) and a silent no-op for own_card
    in {3, 4, 5} (writing outside range(3)). Zeroing the own rank is also
    wrong *in principle*: the opponent can hold the second copy of the
    acting player's own rank, so the true marginal on the holder's own rank
    is never exactly zero. `leduc_rank.py`'s `expand_rank_to_indices`
    excludes only the acting player's own card index and any revealed public
    card, never the whole rank, so nothing downstream depends on zeroing.
    """
    own_card = parse_own_card(info_str)
    if own_card is None:
        return None

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

    total = sum(probs.values())
    if total <= 0.0:
        probs = {c: 1.0 / 3.0 for c in range(3)}
    else:
        probs = {k: v / total for k, v in probs.items()}

    result = {str(c): float(probs.get(c, 0.0)) for c in range(3)}
    return json.dumps(result)


# Canonical default instance -- module-level names are the factory applied to
# this module's own spec, so callers can import them directly (mirrors
# darkhex.py/goofspiel.py's `_DEFAULT_FNS` block).
_DEFAULT_FNS = make_fns()
build_prompt = _DEFAULT_FNS["build_prompt"]
action_suffix = _DEFAULT_FNS["action_suffix"]
action_labels = _DEFAULT_FNS["action_labels"]
belief_suffix = _DEFAULT_FNS["belief_suffix"]
parse_action = _DEFAULT_FNS["parse_action"]
parse_action_aliased = _DEFAULT_FNS["parse_action_aliased"]
