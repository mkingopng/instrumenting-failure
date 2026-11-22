"""Shared game-tree traversal utilities for scoring modules that need a
representative pyspiel.State per (player, info_set_string) key.

Extracted from property7.py's 7b (action-vs-belief) computation, which
needs a real State to call best-response machinery against; property3.py's
regret metric (S1 restatement) needs the identical traversal.
"""

from __future__ import annotations

from typing import Any  # pyspiel is untyped, so a State is Any


def build_infoset_map(game) -> dict[tuple[int, str], Any]:
    """Traverse the full game tree once; map (player, info_set_str) to one
    representative State reached at that info-set.

    Deserializes a fresh State per key (`game.deserialize_state(
    state.serialize())`) rather than holding the live traversal State,
    since a live State's identity is bound to its position in this one
    traversal, not reusable independently by a caller.
    """
    result: dict[tuple[int, str], Any] = {}

    def traverse(state) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                traverse(state.child(a))
            return
        player = state.current_player()
        key = (player, state.information_state_string(player))
        if key not in result:
            result[key] = game.deserialize_state(state.serialize())
        for a in state.legal_actions():
            traverse(state.child(a))

    traverse(game.new_initial_state())
    return result
