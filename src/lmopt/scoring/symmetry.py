"""
Card symmetry of a reference policy, read from the reference itself.

Two logged info-sets that differ only by which of two interchangeable cards they hold are
one decision of the game seen twice, so an average over logged info-sets that counts both
counts that decision twice. This module finds which card ids are interchangeable and maps an
info-set string to its class.

Nothing here knows how many suits a game has. A game names only the fields of its info-set
string that hold a card id (`CARD_FIELDS`); two card ids are interchangeable when swapping
them in every info-set of the reference lands on an info-set the reference also holds, with
an identical policy. A reference that is not symmetric under a swap does not merge those
cards, so an asymmetric reference collapses nothing.
"""

import re
from typing import Mapping

#: The info-set string fields that hold a card id, by OpenSpiel short name. A game with no
#: entry has no card classes, and every info-set is its own class.
CARD_FIELDS: dict[str, tuple[str, ...]] = {"leduc_poker": ("Private", "Public")}

#: The largest difference between two probabilities still called the same policy.
POLICY_TOLERANCE = 1e-9


def _field_pattern(fields: tuple[str, ...]) -> re.Pattern:
    """
    :param fields: the info-set field names that hold a card id (tuple[str, ...])
    :returns: a pattern whose groups are the field's opening text and its card id (re.Pattern)
    """
    return re.compile(r"(\[(?:" + "|".join(map(re.escape, fields)) + r"): )(\d+)(\])")


def _card_ids(reference: Mapping, pattern: re.Pattern) -> list[int]:
    """
    :param reference: {(info_set, player): {action: probability}} (Mapping)
    :param pattern: the card field pattern (re.Pattern)
    :returns: every card id any info-set of the reference shows, sorted (list[int])
    """
    return sorted(
        {int(m.group(2)) for key in reference for m in pattern.finditer(key[0])}
    )


def _relabelled(info_set: str, pattern: re.Pattern, mapping: Mapping[int, int]) -> str:
    """
    :param info_set: an info-set string (str)
    :param pattern: the card field pattern (re.Pattern)
    :param mapping: card id to the card id it is replaced by; a missing id stays (Mapping)
    :returns: the string with each card field's id replaced (str)
    """
    return pattern.sub(
        lambda m: m.group(1)
        + str(mapping.get(int(m.group(2)), int(m.group(2))))
        + m.group(3),
        info_set,
    )


def _same_policy(left: Mapping, right: Mapping) -> bool:
    """
    :param left: {action: probability} (Mapping)
    :param right: {action: probability} (Mapping)
    :returns: True when both name the same actions with probabilities within POLICY_TOLERANCE
        (bool)
    """
    return left.keys() == right.keys() and all(
        abs(left[a] - right[a]) <= POLICY_TOLERANCE for a in left
    )


def _swap_is_symmetry(
    reference: Mapping, pattern: re.Pattern, first: int, second: int
) -> bool:
    """
    :param reference: {(info_set, player): {action: probability}} (Mapping)
    :param pattern: the card field pattern (re.Pattern)
    :param first: one card id (int)
    :param second: another card id (int)
    :returns: True when, for every info-set of the reference, the info-set with the two ids
        swapped is also in the reference and carries the same policy; a swapped info-set the
        reference does not hold counts against the swap (bool)
    """
    swap = {first: second, second: first}
    for (info_set, player), policy in reference.items():
        swapped = _relabelled(info_set, pattern, swap)
        if swapped == info_set:
            continue
        other = reference.get((swapped, player))
        if other is None or not _same_policy(policy, other):
            return False
    return True


def card_classes(reference: Mapping, game_short_name: str) -> dict[int, int]:
    """
    which card ids the reference treats as interchangeable
    :param reference: {(info_set, player): {action: probability}} for every decision node,
        as `reference_policy_map` returns it (Mapping)
    :param game_short_name: the OpenSpiel short name of the game (str)
    :returns: card id to the smallest card id of its class; empty when the game names no card
        fields or the reference shows no card (dict[int, int])
    """
    fields = CARD_FIELDS.get(game_short_name)
    if not fields:
        return {}
    pattern = _field_pattern(fields)
    cards = _card_ids(reference, pattern)
    parent = {card: card for card in cards}

    def find(card: int) -> int:
        while parent[card] != card:
            parent[card] = parent[parent[card]]
            card = parent[card]
        return card

    for i, first in enumerate(cards):
        for second in cards[i + 1 :]:
            if find(first) != find(second) and _swap_is_symmetry(
                reference, pattern, first, second
            ):
                parent[find(second)] = find(first)
    return {card: min(c for c in cards if find(c) == find(card)) for card in cards}


def class_key(info_set: str, classes: Mapping[int, int], game_short_name: str) -> str:
    """
    :param info_set: a logged info-set string (str)
    :param classes: card id to its class, from `card_classes` (Mapping)
    :param game_short_name: the OpenSpiel short name of the game (str)
    :returns: the info-set string with every card id replaced by its class; the string
        itself when the game names no card fields or `classes` is empty (str)
    """
    fields = CARD_FIELDS.get(game_short_name)
    if not fields or not classes:
        return info_set
    return _relabelled(info_set, _field_pattern(fields), classes)
