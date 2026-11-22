"""Goofspiel: info-state string -> NL prompt, action/belief parsing (parametric;
default instance is 4-card, via make_fns(num_cards)).

Goofspiel info-state format (OpenSpiel, player-P view; num_cards=4 shown, sizes
derive from make_fns):
  P{P} hand: 1 2 3 4          <- cards still in hand
  P{P} action sequence:        <- own bids so far, as action indices (card - 1)
  Point card sequence: 4 3     <- point cards revealed (descending, from loaders.py config)
  Win sequence: 0 1            <- who won each round (0=P0, 1=P1, -1=tie)
  Points: 4 3                  <- cumulative scores

Action indices (num_cards=4 shown): 0=bid card 1, 1=bid card 2, 2=bid card 3,
3=bid card 4. In general, for a make_fns(num_cards) instance, action a bids
card a+1, for a in 0..num_cards-1. Only actions for cards still in hand are legal.

Belief elicitation: the model
predicts the opponent's current-round bid distribution q[b] over cards 1-num_cards
(1-4 shown above). The hidden state is the opponent's simultaneous bid, inferred from
round outcomes (imp_info hides the bids), not a fixed private card. parse_belief
emits 0-indexed bid-action keys so the belief aligns with true_posterior
(``goofspiel_all_posteriors``), which is keyed by OpenSpiel bid action (action a = card a+1).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from lmopt.prompting.memory import MemoryTerms
from lmopt.prompting.spec import GOOFSPIEL_SPEC as _SPEC
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

# What the memory block calls a play and its payoff. The engine returns half the point
# difference the rules define (``returns_type=point_difference``), so the factor is 2.
MEMORY_TERMS = MemoryTerms("game", "points", 2.0)


def _memory_terms(spec: Optional[GamePromptSpec], action_labels) -> MemoryTerms:
    """
    the memory terms of one prompt spec: payoffs in the variant's units and, for a variant
    that scales the prizes and scores its prompt states, the replayed states scaled to match
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


def make_fns(num_cards: int, spec: Optional[GamePromptSpec] = None) -> dict:
    """
    build the Goofspiel prompt/parse closures for one deck size and one prompt spec. Rules
    text, parse ranges, and belief key space all derive from num_cards; points_order=
    descending is pinned by loaders.py, so the prize sequence line is always
    "num_cards, ..., 1". Belief keys are 0-indexed bid actions 0..num_cards-1 (action a =
    card a+1), normalised, matching goofspiel_all_posteriors. The constant wording lives in
    spec.py, one source for all deck sizes; only the dimension- and info-set-dependent
    values are filled in here.

    Two spec fields separate the two things a probe run may vary, and keeping them
    separate is the whole design. ``action_vocabulary["card_labels"]`` renames the BID
    cards, which are action identities: it reaches the hand line, the legal replies, the
    action labels, the round history and the belief keys. ``display_scale`` multiplies the
    prize values and the scores, which are payoffs: the prize sequence in the rules, the
    current point card, the "N-point card" phrase and both score numbers. Neither touches
    the other's surface, so P1's relabel run and P4's payoff-scale run vary disjoint parts
    of the prompt
    :param num_cards: deck size for this instance (int)
    :param spec: the prompt spec these closures render, or None for this module's
        canonical ``GOOFSPIEL_SPEC`` (Optional[GamePromptSpec])
    :returns: the ``GamePromptFns`` fields as a dict (dict)
    :raises ValueError: when ``num_cards`` is outside 1..9, which the single-digit parser reads
    """
    if not 1 <= num_cards <= 9:
        raise ValueError(
            f"goofspiel num_cards={num_cards} is outside 1..9: the canonical parser reads a "
            f"bid as one digit, so card 10 and above could never parse; widen parse_action "
            f"before registering a larger deck"
        )
    sp = _SPEC if spec is None else spec
    card_labels = sp.action_vocabulary.get("card_labels")
    if card_labels is not None:
        missing = [c for c in range(1, num_cards + 1) if c not in card_labels]
        if missing:
            raise ValueError(
                f"goofspiel card_labels covers {sorted(card_labels)} but this deck "
                f"needs labels for {missing}; a deck larger than the label alphabet "
                f"would render an unlabelled card rather than fail"
            )

    def label(card: int) -> str:
        """
        the literal this arm renders for one bid card: its number word under a relabel
        variant, its digit canonically. One function serves the hand, the legal replies, the
        round history and the belief keys, so a card reads the same everywhere in an arm
        :param card: a card face, 1..num_cards (int)
        :returns: the rendered literal (str)
        """
        return card_labels[card] if card_labels else str(card)

    scale = sp.display_scale
    first_card, last_card = label(1), label(num_cards)
    descending = ", ".join(str(c * scale) for c in range(num_cards, 0, -1))
    rules = sp.instruction["rules"].format(
        num_cards=num_cards,
        descending=descending,
        first_card=first_card,
        last_card=last_card,
    )

    def build_prompt(info_str: str, current_player: int) -> str:
        hand = _parse_hand(info_str)
        action_seq = _parse_action_seq(info_str)
        revealed = _parse_point_sequence(info_str)
        win_seq = _parse_win_sequence(info_str)
        points = _parse_points(info_str)
        my_points = points[current_player]
        opp_points = points[1 - current_player]
        current_card = _current_point_card(revealed)
        if current_card is None:
            raise ValueError(
                f"build_prompt: info-state string has no point card up for bid, so there is "
                f"nothing to choose a card for: {info_str!r}"
            )
        prize = current_card * scale
        history_desc = _describe_history(action_seq, win_seq, label, current_player)
        hand_str = (
            " ".join(label(c) for c in in_display_order(sp, sorted(hand)))
            if hand
            else "(none)"
        )
        point_card_desc = (
            f"The current point card up for bid is: {prize}"
            if current_card
            else "No more cards."
        )
        return (
            f"{rules}\n\n"
            f"Current state:\n"
            f"- You are Player {current_player}.\n"
            f"- Your remaining cards: {hand_str}\n"
            f"- {point_card_desc}\n"
            f"- Your score: {my_points * scale}. Opponent score: {opp_points * scale}.\n"
            f"- Round history: {history_desc}\n\n"
            f"Choose one card to bid for the {prize}-point card."
        )

    def action_suffix(info_str: str) -> str:
        hand = _parse_hand(info_str)
        hand_str = ", ".join(label(c) for c in sorted(hand)) if hand else "none"
        # Appended on the same line as the hand-listing sentence, not a new
        # line -- see liarsdice.py's action_suffix comment: the echo-adversary
        # wrapper quotes the
        # longest line on the assumption it names the legal labels, and a
        # separate trailing line lets this clause outgrow it and steal the
        # exemplar.
        del hand_str
        return legal_response_suffix(
            [label(c) for c in in_display_order(sp, sorted(hand))]
        )

    def action_labels(info_str: str) -> dict[int, str]:
        """Legal action -> the exact literal action_suffix offers for it.
        Card c in hand -> action c-1, label
        ``label(c)`` -- mirrors action_suffix's own hand_str construction, not
        a re-parse of it, so a relabel variant's words reach both together."""
        hand = _parse_hand(info_str)
        return labels_in_display_order(sp, {c - 1: label(c) for c in sorted(hand)})

    def belief_suffix(info_str: str) -> str:
        """`belief_exemplar` is a PER-slot template (spec.py) with a
        placeholder key -- it does not enumerate a dense `"1": p, "2": p,
        ...` distribution over real card numbers 1..num_cards, which would let
        an echo of belief_suffix's text self-parse (every key mapped via
        `int()`). This closure fills exactly `num_cards`
        slots at `value = 1/num_cards` each, so the illustration's key count
        and value sum match "a probability distribution over cards
        1-{num_cards}" (the question's own stated semantics) for this
        variant -- a fixed 2-key literal regardless of `num_cards` would
        break that sum-to-1 property for every variant except num_cards=2
        (goofspiel_5's example would sum to 0.5, not 1.0). The KEY stays the
        placeholder token `<card_{i}>` (never a real card digit) at every
        slot, so echo-safety is unaffected by per-variant cardinality. The illustrative weights are non-uniform
        and descending, normalized to sum to 1."""
        current_card = _current_point_card(_parse_point_sequence(info_str))
        prize = f" for the {current_card * scale}-point card" if current_card else ""
        # Descending normalized weights: weight[i] = (num_cards + 1 - i)
        # Sum = num_cards * (num_cards + 1) / 2, so normalized value[i] = weight[i] / sum
        total_weight = num_cards * (num_cards + 1) / 2.0
        exemplar = ", ".join(
            sp.belief_exemplar.format(i=i, value=(num_cards + 1 - i) / total_weight)
            for i in range(1, num_cards + 1)
        )
        return sp.belief_question.format(
            prize=prize,
            num_cards=num_cards,
            belief_exemplar=exemplar,
            first_card=first_card,
            last_card=last_card,
            belief_keys=quoted_keys([label(i) for i in range(1, num_cards + 1)]),
        )

    def parse_action(text: str, info_str: str) -> Optional[int]:
        """Action parser hardening: a completion naming 2+ of the in-hand
        card digits action_labels() offers -- e.g. an echo of action_suffix,
        which lists the whole hand -- is rejected before the (already
        legality-gated) digit scan below. See `spec.has_multiple_labels`'s
        docstring.

        No liarsdice-style |A|=1 "last round, 1 card in hand" exception is
        needed here: over the full P0 tree of both registered goofspiel
        variants,
        `len(action_labels(info))` is never 1 -- goofspiel_4's histogram is
        `{2: 70, 3: 10, 4: 1}` (minimum 2), goofspiel_5's likewise never
        drops to 1. Goofspiel's P0 player never actually decides at true
        `|hand| == 1` (the wrapped `turn_based_simultaneous_game`'s round
        structure ends P0's decisions one card early), so this game needs
        no |A|=1 exception -- unlike liarsdice, `self_parse_report`'s
        exclusion never fires for either goofspiel variant
        (`n_action_checked == n_infosets` for both, confirmed in
        `test_prompt_fidelity.py`).
        """
        hand = _parse_hand(info_str)
        if has_multiple_labels(text, action_labels(info_str).values()):
            return None
        if card_labels is not None:
            # Word-bounded and legality-gated exactly as the digit path is, and
            # digits are not accepted: parser permissiveness is a property of
            # the arm (see probe_variants._RELABEL_SYNONYMS' comment), so the
            # relabel arm accepts one vocabulary just as the base arm does.
            upper = text.strip().upper()
            for card in sorted(hand):
                if re.search(r"\b" + re.escape(label(card).upper()) + r"\b", upper):
                    return card - 1
            return None
        # Strict word-bounded single digit (see darkhex parse_action note):
        # verified 0-diff against the pre-parametric \b([1-4])\b acceptance
        # set - letter-adjacent digits ("card4") stay unparseable.
        for m in re.finditer(r"\b(\d)\b", text.strip()):
            card = int(m.group(1))
            if 1 <= card <= num_cards and card in hand:
                return card - 1
        return None

    _LABEL_LOOKUP = (
        {} if card_labels is None else {v.upper(): k for k, v in card_labels.items()}
    )

    def _belief_key_to_card(key) -> Optional[int]:
        """
        one belief-object key read back as a card face value, or None when it names no card.
        Under a relabel variant the accepted keys are the word labels, matched
        case-insensitively; canonically they are the card digits
        :param key: a key from the model's parsed belief object (Any)
        :returns: the card's face value, or None (Optional[int])
        """
        token = str(key).strip()
        if card_labels is not None:
            return _LABEL_LOOKUP.get(token.upper())
        try:
            return int(token)
        except ValueError:
            return None

    def parse_belief(text: str, info_str: str) -> Optional[str]:
        raw = load_belief_object(text)
        if raw is None:
            return None
        probs: dict[int, float] = {}
        for k, v in raw.items():
            card = _belief_key_to_card(k)
            if card is None:
                continue
            if 1 <= card <= num_cards:
                number = finite_float(v)
                if number is not None:
                    probs[card] = number
        if not probs:
            return None
        total = sum(probs.values())
        if total <= 0.0:
            return None
        return json.dumps(
            {str(c - 1): probs.get(c, 0.0) / total for c in range(1, num_cards + 1)}
        )

    return {
        "build_prompt": build_prompt,
        "action_suffix": action_suffix,
        "action_labels": action_labels,
        "belief_suffix": belief_suffix,
        "parse_action": parse_action,
        "parse_belief": parse_belief,
        "parse_own_card": parse_own_card,
        "memory_terms": _memory_terms(spec, action_labels),
    }


def _parse_hand(info_str: str) -> list[int]:
    """
    the acting player's remaining cards; a hand line with no cards is an empty hand
    :param info_str: the info-state string for the acting player (str)
    :returns: the card faces in the hand (list[int])
    :raises ValueError: when the info state has no hand line
    """
    m = re.search(r"P\d+\s+hand:\s*([\d ]*)", info_str)
    if not m:
        raise ValueError(f"info-state string has no hand line: {info_str!r}")
    return [int(x) for x in m.group(1).split() if x.isdigit()]


def _parse_action_seq(info_str: str) -> list[int]:
    """
    the acting player's own bids so far, as card faces. OpenSpiel writes this sequence as
    bid action indices, where index ``a`` is card ``a + 1``; the hand line and the legal
    replies are card faces, so the history must be too or the prompt reports a bid of card
    4 as "you bid 3" while card 3 is still in the hand
    :param info_str: the info-state string for the acting player (str)
    :returns: one card face per round already played (list[int])
    """
    m = re.search(r"P\d+\s+action sequence:\s*([\d ]*)", info_str)
    if not m:
        return []
    return [int(x) + 1 for x in m.group(1).split() if x.isdigit()]


def _scale_state(info_str: str, scale: int) -> str:
    """
    the info state with its point cards and both seats' points multiplied by a payoff-scale
    variant's factor, which is the scale the variant's prompt states them in; the hand, the
    action sequence and the win sequence name cards and winners, not payoffs, and stay as the
    engine wrote them
    :param info_str: the info-state string (str)
    :param scale: the variant's display scale (int)
    :returns: the string with the Point card sequence and Points lines rescaled (str)
    :raises ValueError: when either line is missing, because a state replayed with one of them
        unscaled would state two scales in one prompt
    """

    def times(match: "re.Match[str]") -> str:
        return match.group(1) + re.sub(
            r"\d+", lambda n: str(int(n.group()) * scale), match.group(2)
        )

    scaled = info_str
    for line in ("Point card sequence:", "Points:"):
        scaled, found = re.subn(
            "(" + re.escape(line) + r")([^\n]*)", times, scaled, count=1
        )
        if not found:
            raise ValueError(f"info-state string has no {line!r} line: {info_str!r}")
    return scaled


def _parse_point_sequence(info_str: str) -> list[int]:
    """Extract point cards revealed so far."""
    m = re.search(r"Point card sequence:\s*([\d ]*)", info_str)
    if not m:
        return []
    return [int(x) for x in m.group(1).split() if x.isdigit()]


def _parse_win_sequence(info_str: str) -> list[str]:
    """Extract win sequence as list of strings ('0', '1', '-1')."""
    m = re.search(r"Win sequence:\s*([-\d ]*)", info_str)
    if not m:
        return []
    return [x for x in m.group(1).split() if x]


def _parse_points(info_str: str) -> tuple[int, int]:
    """
    both seats' points so far
    :param info_str: the info-state string for the acting player (str)
    :returns: seat 0's and seat 1's points (tuple[int, int])
    :raises ValueError: when the info state has no parseable Points line
    """
    m = re.search(r"Points:\s*(\d+)\s+(\d+)", info_str)
    if not m:
        raise ValueError(
            f"info-state string has no parseable Points line: {info_str!r}"
        )
    return int(m.group(1)), int(m.group(2))


def _current_point_card(revealed: list[int]) -> Optional[int]:
    """The point card being contested at this decision node.

    OpenSpiel reveals the round's point card (a chance event) *before* the
    players bid, so it is the last entry of the info-state's point-card sequence
    (verified: the round-1 P0 node already shows "Point card sequence: 4", card 4
    being contested). Returns None only when the sequence is empty.
    """
    return revealed[-1] if revealed else None


def _describe_history(action_seq, win_seq, label=str, current_player: int = 0) -> str:
    """
    the acting player's own bidding history as prose. The bid is an action identity, so it
    is rendered through the arm's label function and never scaled. The engine names each
    round's winner by player ID and writes a tie as a negative id, so the outcome is
    resolved against the acting player's seat: read as seat 0 from seat 1, every "you won"
    would be the opponent's win
    :param action_seq: own bids so far, as card face values (list[int])
    :param win_seq: the winner's player id per round, negative for a tie (list[str])
    :param label: the arm's card-label function, ``str`` canonically (Callable[[int], str])
    :param current_player: the seat the prompt is written for (int)
    :returns: one sentence per played round, or the no-rounds line (str)
    """
    if not action_seq:
        return "No rounds played yet."
    parts = []
    for i, (bid, result) in enumerate(zip(action_seq, win_seq, strict=True)):
        winner = int(result)
        if winner < 0:
            outcome = "tie, nobody won the point card"
        elif winner == current_player:
            outcome = "you won"
        else:
            outcome = "opponent won"
        parts.append(f"Round {i+1}: you bid {label(bid)} → {outcome}")
    return "; ".join(parts) + "."


def parse_own_card(info_str: str) -> None:
    """No fixed private card in Goofspiel."""
    return None


_DEFAULT_FNS = make_fns(4)
build_prompt = _DEFAULT_FNS["build_prompt"]
action_suffix = _DEFAULT_FNS["action_suffix"]
action_labels = _DEFAULT_FNS["action_labels"]
belief_suffix = _DEFAULT_FNS["belief_suffix"]
parse_action = _DEFAULT_FNS["parse_action"]
parse_belief = _DEFAULT_FNS["parse_belief"]
