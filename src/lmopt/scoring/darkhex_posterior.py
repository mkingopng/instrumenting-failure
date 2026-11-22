"""Dark Hex board-posterior μ(w|I) - the load-bearing P8 dependency.

At a decision node of the observing seat with info-state ``I``, the hidden state
is the opponent's true board (its placed stones, which the observer sees only
where it has probed). ``μ(w|I)`` is the reach-weighted distribution over the
concrete worlds (full game states) consistent with ``I`` under the opponent's
CFR+ policy - the same reach-weighting used for the Goofspiel bid posterior,
since Dark Hex has no chance nodes and the only hidden randomness is the
opponent's move choices.

Every function here takes a ``seat``: the seat whose information the posterior
conditions on. The opponent is ``1 - seat``, and the observer's own reach cancels
in the normalisation (perfect recall), which is what makes weighting by opponent
reach alone correct at either seat. The two seats are genuinely
different populations, not relabellings of one another: on the 2x2 board,
33 seat-0 info-sets against 101 seat-1 info-sets, sharing none.

Three products:
- ``gather_worlds`` -> ``{info_str(seat): [(state, opp_reach), ...]}`` (shared
  with the P8 scorer, which needs the concrete worlds for continuation values).
- ``gather_p0_worlds`` -> the ``seat=0`` delegate, kept as a name because it is
  what ``property8.py`` and the P8 tests were written against.
- ``cell_occupancy_posterior`` -> ``{cell_action: P(opponent stone at cell)}`` -
  the per-cell marginal used as the Dark Hex ``true_posterior`` for belief scoring.

Silent-wrong-posterior mitigation (the failure mode that bit Liar's Dice):
- The walk-accumulated opponent reach is cross-checked against an independent
  recomputation by replaying the world's opponent-move history (the oracle lives in
  ``tests/test_darkhex_posterior.py``); a test asserts they agree to 1e-9 at both
  seats, with a positive control that a wrong (first-action) reach fails the check.
- Invariants: the world weights normalise to 1; cell marginals ∈ [0, 1]; the
  posterior is never fabricated - an unreachable info-state returns ``{}`` (the
  caller flags it), never a silent uniform.
- A ``seat`` outside {0, 1} raises. Defaulting it would score the wrong seat's
  population in silence, which is precisely the defect this module exists to
  make impossible (no silent fallbacks).
"""

from __future__ import annotations

import pyspiel

from lmopt.games.loaders import darkhex_board_dims as _board_dims

# The mark each player's own stones carry in any player's rendered view of the
# board: player 0 is black ('x'), player 1 is white ('o'). A discovered opponent
# stone is drawn with the OPPONENT's mark, so reading player p's own mark out of
# player p's own view yields exactly p's placed stones. Across the full 2x2
# and 2x3 trees and 300 random 3x3 playouts, dark_hex's
# information-state strings contain only '.', 'x' and 'o' - the edge-connection
# marks Hex uses ('y', 'z', 'p', 'q', ...) never appear.
_OWN_STONE_MARK = ("x", "o")


def _check_seat(seat: int) -> int:
    """Validate an observing seat. Raises on anything but 0 or 1."""
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    return seat


def opponent_stone_cells(
    state: pyspiel.State, game: pyspiel.Game, seat: int = 0
) -> set[int]:
    """Cell action-indices where ``seat``'s opponent has a placed stone here.

    Read from ``information_state_string(1 - seat)`` - the opponent's own stones
    are its own mark in its own view, and it sees all of them. Cell (r, c) ->
    action index ``r * num_cols + c``.
    """
    opponent = 1 - _check_seat(seat)
    mark = _OWN_STONE_MARK[opponent]
    nrows, ncols = _board_dims(game)
    lines = state.information_state_string(opponent).split("\n")
    cells: set[int] = set()
    for r in range(nrows):
        if r >= len(lines):
            break
        row = lines[r]
        for c in range(ncols):
            if c < len(row) and row[c] == mark:
                cells.add(r * ncols + c)
    return cells


def gather_worlds(game: pyspiel.Game, opp_policy, seat: int = 0) -> dict:
    """One tree walk -> ``{info_str(seat): [(state, opp_reach), ...]}``.

    The concrete worlds consistent with each of ``seat``'s info-states, weighted
    by the opponent's CFR+ reach (the product of ``1 - seat``'s action
    probabilities along its own moves). Dark Hex has no chance nodes, so chance
    reach is 1 throughout, and the observer's own reach cancels in the
    normalisation, so it is not accumulated.

    The walk is keyed on the OBSERVER's decision nodes, which is what makes the
    two seats different partitions rather than two labellings of one partition:
    Dark Hex does not alternate strictly (a collision returns the move to the
    same player), so the seats do not even see the same number of nodes.
    """
    opponent = 1 - _check_seat(seat)
    worlds: dict[str, list[tuple[object, float]]] = {}

    def _walk(state: pyspiel.State, opp_reach: float) -> None:
        if state.is_terminal():
            return
        player = state.current_player()
        if player == seat:
            info = state.information_state_string(seat)
            worlds.setdefault(info, []).append(
                (game.deserialize_state(state.serialize()), opp_reach)
            )
            for a in state.legal_actions():
                _walk(state.child(a), opp_reach)  # observer reach cancels
        else:
            probs = opp_policy.action_probabilities(state, opponent)
            for a in state.legal_actions():
                _walk(state.child(a), opp_reach * probs.get(a, 0.0))

    _walk(game.new_initial_state(), 1.0)
    return worlds


def gather_p0_worlds(game: pyspiel.Game, opp_policy) -> dict:
    """The ``seat=0`` delegate of :func:`gather_worlds`, kept as a name.

    ``property8.py`` and the P8 tests were written against this name; it is a
    call, not a second implementation, so the two cannot drift.
    """
    return gather_worlds(game, opp_policy, seat=0)


def cell_occupancy_posterior(
    worlds_for_info: list, game: pyspiel.Game, seat: int = 0
) -> dict:
    """Per-cell opponent-occupancy marginal at one of ``seat``'s info-states.

    Returns ``{cell_action: P(opponent has a stone at cell)}`` over the cells the
    observer has not already discovered (only unknown cells carry belief; a
    discovered cell is certain and excluded). Empty dict if the info-state is
    unreachable (total reach 0) - never a fabricated uniform.
    """
    _check_seat(seat)
    total = sum(w for _, w in worlds_for_info)
    if total <= 0.0:
        return {}
    # Cells the observer already knows about (its own + discovered) are excluded:
    # they carry no belief. Unknown cells = the observer's legal actions at any
    # consistent world, which is a function of the info-state alone.
    unknown = set(worlds_for_info[0][0].legal_actions())
    marg: dict[int, float] = {c: 0.0 for c in unknown}
    for state, reach in worlds_for_info:
        if reach <= 0.0:
            continue
        opp_cells = opponent_stone_cells(state, game, seat=seat)
        for c in unknown:
            if c in opp_cells:
                marg[c] += reach / total
    return marg
