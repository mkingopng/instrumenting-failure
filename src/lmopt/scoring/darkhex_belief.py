"""Dark Hex belief scoring: the per-cell primitives P5 and P7 need.

Dark Hex belief is per-cell opponent occupancy -- one independent Bernoulli per
cell the observer cannot see -- not a probability simplex, so the ECE and TVD
forms the card games use do not transfer. The estimands here replace them:

- per-cell absolute error, ``mean_k |b_k - p_k|``, is both P5's observed
  quantity and P7a's belief error (the total-variation distance of a single
  Bernoulli is exactly ``|b_k - p_k|``, so the per-cell mean is the mean TVD
  over the cells).
- two baselines for P5: an uninformed stater who says 0.5 at every cell, and a
  stricter stater who knows only how many stones the opponent has placed and
  spreads them evenly over the unknown cells.
- for P7b, a distribution over the opponent boards consistent with an
  info-state, induced by the stated per-cell probabilities under an
  independence assumption, spread across each board's concrete worlds by the
  opponent reach the enumeration carries, and the best actions under it.

The independence assumption is the caveat every number here carries. The
true occupancy marginals are not independent (the opponent has placed a fixed
number of stones, so the cells are negatively correlated), and a stated belief
is elicited per cell with no joint. Turning per-cell probabilities into a board
distribution therefore adds an assumption the elicitation never made; it is the
only way to best-respond to a per-cell belief at all, and it is named rather
than hidden.

Nothing here reads or writes a file (scoring is offline); the two
world-level functions take already-gathered worlds from
``darkhex_posterior.gather_worlds`` and an already-solved policy.
"""

from __future__ import annotations

#: The marks a rendered Dark Hex view uses for each player's own stones, the
#: same convention ``darkhex_posterior`` reads the opponent's stones with.
_OWN_STONE_MARK = ("x", "o")
#: Everything a board line of a rendered view may contain. The information
#: state string is ``<board lines>\n<moves made>\n<own action sequence>``, and
#: the moves-made line is the first that is not drawn from this alphabet, so
#: the board can be split off without knowing the board's dimensions.
_BOARD_CHARS = frozenset(".xo")


def aligned_cells(stated: dict, true: dict) -> list:
    """
    the cells both the stated belief and the true posterior speak about
    Scoring runs over the intersection so a stated belief that omits a cell, or
    invents one, is never compared against a fabricated 0.0 on either side; the
    caller counts the mismatch and the empty-intersection case rather than
    letting either pass silently.
    :param stated: the row's elicited belief, {cell_action_index: probability} (dict)
    :param true: the row's true posterior, the same key space (dict)
    :returns: the sorted list of keys present in both, possibly empty
    """
    return sorted(set(stated) & set(true), key=str)


def per_cell_abs_error(stated: dict, true: dict, cells: list) -> float:
    """
    mean over `cells` of |stated - true|, one Bernoulli's total-variation
    distance averaged over the cells
    :param stated: the row's elicited belief (dict)
    :param true: the row's true posterior (dict)
    :param cells: the keys to score, normally `aligned_cells`' output (list)
    :returns: the mean absolute per-cell error, or NaN when `cells` is empty
    """
    if not cells:
        return float("nan")
    return sum(abs(float(stated[c]) - float(true[c])) for c in cells) / len(cells)


def uninformed_cell_probability() -> float:
    """
    the per-cell probability an uninformed stater gives, the Dark Hex analogue
    of the card games' uniform-over-stated-keys baseline
    A cell is occupied or it is not, and a stater who knows nothing about the
    board says 0.5 to each; unlike a simplex there is no key count to divide by,
    which is why P5's baseline for this family is a constant rather than
    `1 / len(stated)`.
    :returns: 0.5
    """
    return 0.5


def stone_count_cell_probability(n_opponent_stones: int, n_unknown_cells: int) -> float:
    """
    the per-cell probability of a stater who knows only the opponent's stone
    count and spreads it evenly over the cells it cannot see
    Stricter than the uninformed 0.5 baseline: it already encodes the board's
    own arithmetic, so beating it is evidence the model read the position rather
    than the rules.
    :param n_opponent_stones: stones the opponent has placed so far (int)
    :param n_unknown_cells: cells the observer cannot see, the belief's key count (int)
    :returns: stones per unknown cell, or 0.0 when there are no unknown cells
    """
    if n_unknown_cells <= 0:
        return 0.0
    return n_opponent_stones / n_unknown_cells


def board_view_lines(info_set: str) -> list:
    """
    the rendered board rows at the head of a Dark Hex information state string
    The string is ``<board lines>\\n<moves made>\\n<own action sequence>``; the
    board rows are exactly the leading lines drawn from '.', 'x' and 'o', so the
    split needs no board dimensions and works on any board size.
    :param info_set: the row's logged `info_set` (str)
    :returns: the list of board row strings, possibly empty
    """
    lines = []
    for line in info_set.split("\n"):
        if line and set(line) <= _BOARD_CHARS:
            lines.append(line)
            continue
        break
    return lines


def opponent_stone_count(info_set: str, seat: int) -> int:
    """
    how many stones the opponent has placed, read off the observer's own view

    Turn structure, not arithmetic on the move counter: a Dark Hex turn ends
    only on a successful placement (a collision returns the move to the same
    player), so the completed turns alternate strictly from player 0. When the
    observer is to act, player 0 and player 1 have therefore completed equal
    numbers of turns at seat 0, and player 0 is one ahead at seat 1. Each
    completed turn placed exactly one stone, so the opponent's count is the
    observer's own visible stone count, plus one at seat 1. The move counter in
    the information state string cannot be used instead: it counts collisions
    too, and the observer never sees the opponent's collisions -- on the
    `board_size=2` board, `moves_made - own_attempts` overstates the
    opponent's stones on 64 of the 134 information states, while this rule is
    exact on all 134.

    :param info_set: the row's logged `info_set` (str)
    :param seat: the acting seat whose view `info_set` is, 0 or 1 (int)
    :returns: the opponent's placed-stone count
    :raises ValueError: when `seat` is not 0 or 1
    """
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    own_mark = _OWN_STONE_MARK[seat]
    own_stones = sum(line.count(own_mark) for line in board_view_lines(info_set))
    return own_stones + (1 if seat == 1 else 0)


def world_weights(
    stated: dict, world_occupancies: list, world_reaches: list, cells: list
) -> list:
    """
    the stated belief's distribution over concrete worlds, under independence

    The belief is over boards, so the mass is assigned per board and not per
    world. A board's unnormalised mass is the product over `cells` of ``b_k``
    where the opponent holds cell ``k`` and ``1 - b_k`` where it does not, and
    the masses are normalised over the distinct boards. A board that several
    move orders reach is still one board: giving it the product once per world
    would multiply its mass by how many histories happen to produce it, which
    is an artefact of enumerating histories and not something the model stated.

    Within a board the mass is split across that board's worlds in proportion
    to the opponent reach the world enumeration already carries, because the
    model said nothing about move order and the truth is the only non-arbitrary
    thing left to fill the residual with. A board whose worlds all carry zero
    reach is split evenly, there being nothing to weight by.

    :param stated: the row's elicited belief (dict)
    :param world_occupancies: one entry per world, the set of cell keys the
        opponent holds there, in the same key type `stated` uses (list)
    :param world_reaches: the opponent reach of each world, index-aligned with
        `world_occupancies` (list)
    :param cells: the cells the product runs over; a board is identified by
        which of these it holds, since that is all the belief distinguishes
        (list)
    :returns: the normalised weights, aligned index-for-index with
        `world_occupancies`; an empty list when every board's mass is zero, so
        the caller counts the row rather than scoring it against a fabricated
        prior
    """
    cell_set = set(cells)
    boards = [frozenset(set(occupied) & cell_set) for occupied in world_occupancies]
    worlds_of: dict = {}
    for index, board in enumerate(boards):
        worlds_of.setdefault(board, []).append(index)

    mass = {}
    for board in worlds_of:
        product = 1.0
        for c in cells:
            b = float(stated[c])
            product *= b if c in board else (1.0 - b)
        mass[board] = product
    total = sum(mass.values())
    if total <= 0.0:
        return []

    weights = [0.0] * len(world_occupancies)
    for board, indices in worlds_of.items():
        board_mass = mass[board] / total
        reaches = [max(0.0, float(world_reaches[i])) for i in indices]
        reach_total = sum(reaches)
        for i, reach in zip(indices, reaches, strict=True):
            weights[i] = (
                board_mass * (reach / reach_total)
                if reach_total > 0.0
                else board_mass / len(indices)
            )
    return weights


def best_actions(action_values: dict, tolerance: float) -> set:
    """
    the actions attaining the maximum value, within `tolerance`
    A set, not a single argmax: Dark Hex ties genuinely, and crediting only one
    of several equal-value actions would score a correct choice as wrong.
    :param action_values: {action: expected continuation value} (dict)
    :param tolerance: the slack property 8 uses for "attains the best value" (float)
    :returns: the set of actions within `tolerance` of the best value, empty when
        `action_values` is empty
    """
    if not action_values:
        return set()
    best = max(action_values.values())
    return {a for a, v in action_values.items() if best - v <= tolerance}


def belief_action_values(
    world_action_values: list, weights: list, legal_actions: list
) -> dict:
    """
    expected continuation value per action under a stated belief's world weights
    :param world_action_values: one {action: value} map per world, index-aligned
        with `weights` (list)
    :param weights: the normalised world weights from `world_weights` (list)
    :param legal_actions: the actions to score, the same at every world of one
        information state (list)
    :returns: {action: sum over worlds of weight * value}, empty when there are
        no weights
    """
    if not weights:
        return {}
    values = {a: 0.0 for a in legal_actions}
    for per_world, weight in zip(world_action_values, weights, strict=True):
        if weight <= 0.0:
            continue
        for a in legal_actions:
            values[a] += weight * per_world[a]
    return values
