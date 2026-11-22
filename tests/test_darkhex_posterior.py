"""Dark Hex board-posterior: correctness + silent-wrong-posterior mitigation.

The board-posterior is load-bearing (shared by the P8 scorer and, later, the Dark
Hex Bayesian agent), so it gets the same dual-path validation that caught the
Liar's Dice posterior bug: the walk-accumulated opponent reach is cross-checked
against an independent history-replay (``opp_reach_from_history``, kept in this file
as a test oracle, not in the scorer), with a positive control proving the check
is not vacuous, plus no-fabrication invariants.
"""

from __future__ import annotations

import pytest

pyspiel = pytest.importorskip("pyspiel")

from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_darkhex
from lmopt.scoring import darkhex_posterior
from lmopt.scoring.darkhex_posterior import (
    _check_seat,
    cell_occupancy_posterior,
    gather_p0_worlds,
    gather_worlds,
    opponent_stone_cells,
)


def opp_reach_from_history(game, world, opp_policy, seat=0):
    """
    independent recomputation of the opponent's CFR+ reach to a world, by replaying its action history
    :param game: the Dark Hex game (pyspiel.Game)
    :param world: a concrete state whose history is replayed from the root (pyspiel.State)
    :param opp_policy: policy with ``action_probabilities(state, player)`` (TabularPolicy)
    :param seat: the observing seat; the opponent is ``1 - seat`` (int)
    :returns: the product of the opponent's action probabilities at its own decisions (float)

    Kept here, not in the scorer, as the oracle for the dual-path check against the reach that
    ``gather_worlds`` accumulates in its tree walk.
    """
    opponent = 1 - _check_seat(seat)
    s = game.new_initial_state()
    reach = 1.0
    for a in world.history():
        if s.is_terminal():
            break
        if s.current_player() == opponent:
            reach *= opp_policy.action_probabilities(s, opponent).get(a, 0.0)
        s.apply_action(a)
    return reach


@pytest.fixture(scope="module")
def dh_game_policy():
    game = load_darkhex()
    return game, solve_cfr(game, 300)


def test_opponent_stone_cells_reads_p1_stones(dh_game_policy):
    game, _ = dh_game_policy
    s = game.new_initial_state()
    s.apply_action(0)  # P0 -> (0,0)
    s.apply_action(3)  # P1 -> (1,1) == cell action 3
    # From any world, P1's true stone is cell 3.
    assert opponent_stone_cells(s, game) == {3}


def test_dual_path_reach_matches(dh_game_policy):
    """
    The walk-accumulated opponent reach equals an independent history replay.
    """
    game, policy = dh_game_policy
    worlds = gather_p0_worlds(game, policy)
    assert worlds
    checked = 0
    for wlist in worlds.values():
        for state, reach in wlist:
            indep = opp_reach_from_history(game, state, policy)
            assert abs(reach - indep) < 1e-9
            checked += 1
    assert checked > 20  # non-trivial coverage (2x2 has ~49 P0 worlds)


def test_positive_control_wrong_reach_is_caught(dh_game_policy):
    """A deliberately wrong opponent reach must differ from the true CFR+ reach
    somewhere - proving the dual-path check discriminates (not vacuous). NB: on
    2x2 the CFR+ P1 policy happens to be uniform (P1 is the doomed player, so it is
    indifferent), so a uniform 'wrong' policy would not differ - we use a
    deterministic first-action policy, which does."""
    game, policy = dh_game_policy

    class FirstAction:  # puts all mass on the first legal action - clearly != CFR+
        def action_probabilities(self, state, player_id=None):
            legal = state.legal_actions()
            return {a: (1.0 if a == legal[0] else 0.0) for a in legal}

    worlds = gather_p0_worlds(game, policy)
    differs = False
    for wlist in worlds.values():
        for state, reach in wlist:
            if abs(opp_reach_from_history(game, state, FirstAction()) - reach) > 1e-6:
                differs = True
                break
        if differs:
            break
    assert differs, "dual-path check would be vacuous if the wrong reach matched CFR+"


def test_marginals_in_unit_interval_and_normalised_worlds(dh_game_policy):
    game, policy = dh_game_policy
    worlds = gather_p0_worlds(game, policy)
    for wlist in list(worlds.values())[:40]:
        total = sum(w for _, w in wlist)
        if total <= 0:
            continue
        marg = cell_occupancy_posterior(wlist, game)
        for c, p in marg.items():
            assert -1e-9 <= p <= 1.0 + 1e-9  # per-cell probability


def test_no_fabrication_on_unreachable(dh_game_policy):
    """Unreachable info-state (empty/zero-reach worlds) -> {} (never a uniform)."""
    game, _ = dh_game_policy
    assert cell_occupancy_posterior([], game) == {}
    s = game.new_initial_state()
    assert cell_occupancy_posterior([(s, 0.0)], game) == {}


def test_board_dims_square_default():
    """Direct assertion for the plain square case (otherwise covered only
    indirectly, via test_opponent_stone_cells_reads_p1_stones depending on a
    correct ncols). load_darkhex() defaults to board_size=2, i.e. a 2x2 board."""
    game = load_darkhex()
    assert darkhex_posterior._board_dims(game) == (2, 2)


def test_board_dims_raises_on_paramless_game_instead_of_defaulting_to_3x3():
    """A dark-hex game with neither dims form must raise, not silently score 3x3.

    With p.get("board_size", 3), a malformed game would score as 3x3 with no
    error - a silent fallback.
    """
    pytest.importorskip("pyspiel")

    class _NoParams:
        def get_parameters(self):
            return {}

    with pytest.raises(KeyError):
        darkhex_posterior._board_dims(_NoParams())


def test_board_dims_prefers_num_rows_cols_over_misleading_board_size():
    """dark_hex(num_rows=3,num_cols=4) also reports a misleading board_size=3.

    This test's whole point is a NON-square board (num_rows != num_cols),
    which a square board (darkhex_3x3, the only other registry variant)
    cannot exercise -- `board_size` would then coincidentally equal both
    num_rows and num_cols, hiding exactly the defect this test catches.
    Built directly via `load_darkhex(num_rows=3, num_cols=4)` (not a GAMES
    registry key, no new registry surface) rather than through the registry,
    since 3x4 is not itself a registered variant and does not need to be
    one. `_board_dims` reads only `get_parameters()`, no tree walk, so board
    size carries no cost here either way.
    """
    pytest.importorskip("pyspiel")
    game = load_darkhex(num_rows=3, num_cols=4)
    assert darkhex_posterior._board_dims(game) == (3, 4)


# ---------------------------------------------------------------------------
# The seat-general walk
# ---------------------------------------------------------------------------


def _brute_force_occupancy_maps(game, policy, seat):
    """Independent ``{info_str(seat): {cell: P(opponent stone at cell)}}`` on 2x2.

    Written from scratch here, importing nothing from ``darkhex_posterior``, so
    a seat error in that module surfaces as a disagreement rather than as a
    shared bug. Two quantities are re-derived independently:

    * the opponent's reach, accumulated down the walk as the product of the
      opponent's policy probabilities at its own nodes (the module accumulates
      the same product, but a seat error there points it at the wrong player);
    * the opponent's placed stones, replayed on a board this function keeps
      itself: an action lands a stone only on an empty cell, and a collision
      leaves the board unchanged and returns the move to the same player
      (on 2x2: P1 attempting P0's cell stays
      `current_player() == 1`). The module instead parses the opponent's
      information-state rendering, which is a different route to the same set.

    The unknown-cell key set is taken from OpenSpiel's own ``legal_actions()``
    at the observer's node, which is a property of the state and not of the
    module under test.
    """
    opponent = 1 - seat
    groups: dict[str, list[tuple[float, set[int], tuple[int, ...]]]] = {}

    def _walk(state, reach, board):
        if state.is_terminal():
            return
        player = state.current_player()
        if player == seat:
            groups.setdefault(state.information_state_string(seat), []).append(
                (
                    reach,
                    {c for c, owner in board.items() if owner == opponent},
                    tuple(state.legal_actions()),
                )
            )
        probs = (
            policy.action_probabilities(state, opponent) if player == opponent else None
        )
        for a in state.legal_actions():
            next_board = dict(board)
            next_board.setdefault(a, player)
            _walk(
                state.child(a),
                reach if probs is None else reach * probs.get(a, 0.0),
                next_board,
            )

    _walk(game.new_initial_state(), 1.0, {})

    out: dict[str, dict[int, float]] = {}
    for info, parts in groups.items():
        legal_sets = {p[2] for p in parts}
        assert len(legal_sets) == 1, (
            f"info-set {info!r} groups states with different legal actions; "
            "the unknown-cell key set would be ambiguous"
        )
        total = sum(r for r, _, _ in parts)
        if total <= 0.0:
            continue
        unknown = parts[0][2]
        marg = {c: 0.0 for c in unknown}
        for reach, opp_cells, _ in parts:
            if reach <= 0.0:
                continue
            for c in unknown:
                if c in opp_cells:
                    marg[c] += reach / total
        out[info] = marg
    return out


def _module_occupancy_maps(game, policy, seat):
    """The production route: ``gather_worlds`` -> ``cell_occupancy_posterior``."""
    worlds = gather_worlds(game, policy, seat=seat)
    maps = {
        info: cell_occupancy_posterior(wl, game, seat=seat)
        for info, wl in worlds.items()
    }
    # An unreachable info-set yields {} (never a fabricated uniform); the brute
    # force drops those, so drop them here too before comparing.
    return {info: m for info, m in maps.items() if m}


def test_seat_populations_differ_so_relabelling_cannot_produce_seat_one(
    dh_game_policy,
):
    """Non-vacuity for the P-seat1-correct mutation below.
    """
    game, policy = dh_game_policy
    seat0 = gather_worlds(game, policy, seat=0)
    seat1 = gather_worlds(game, policy, seat=1)
    assert len(seat0) == 33
    assert len(seat1) == 101
    assert not (set(seat0) & set(seat1))


def test_seat1_occupancy_matches_independent_brute_force(dh_game_policy):
    """P-seat1-correct: every seat-1 info-set's occupancy marginal is right.

    Checked over all 101 seat-1 info-sets, not one: which info-set a single
    spot check lands on depends on dict traversal order and on the CFR
    iteration count.
    """
    game, policy = dh_game_policy
    expected = _brute_force_occupancy_maps(game, policy, seat=1)
    got = _module_occupancy_maps(game, policy, seat=1)

    assert set(got) == set(expected)
    assert len(expected) > 50, "brute force must cover the seat-1 population"
    nonzero = 0
    for info, marg in expected.items():
        assert set(got[info]) == set(marg), f"cell key set differs at {info!r}"
        for cell, p in marg.items():
            assert got[info][cell] == pytest.approx(p, abs=1e-12)
            nonzero += int(p > 1e-9)
    assert nonzero > 0, "all-zero marginals everywhere would make this vacuous"


def test_seat0_occupancy_matches_the_same_brute_force(dh_game_policy):
    """Calibration control for the brute force above.

    If the brute force disagreed with the module at seat 0 - the population
    every collected arm was scored on - then a seat-1 agreement would be
    evidence about nothing.
    """
    game, policy = dh_game_policy
    expected = _brute_force_occupancy_maps(game, policy, seat=0)
    got = _module_occupancy_maps(game, policy, seat=0)
    assert set(got) == set(expected)
    for info, marg in expected.items():
        assert set(got[info]) == set(marg)
        for cell, p in marg.items():
            assert got[info][cell] == pytest.approx(p, abs=1e-12)


def _full_reach_occupancy_maps(game, policy, seat):
    """The same marginals with no cancellation assumption anywhere.

    ``gather_worlds`` weights a world by the opponent's reach alone, and
    ``_brute_force_occupancy_maps`` above does the same, so the two share a
    premise rather than testing it: that the OBSERVER's own reach is constant
    across the worlds of one info-set (perfect recall) and therefore divides out
    of the normalisation. This function drops that premise and weights each world
    by its full reach under the policy profile, the product of both players'
    behavioural probabilities along the path. Everything else is as before: the
    opponent's stones come from a board this function replays itself.

    On the 2x2 board, CFR+ 300 iterations: all 33 seat-0
    and all 101 seat-1 info-sets carry strictly positive full reach, so nothing
    is skipped and the coverage assertions below are exact counts, not floors.
    """
    opponent = 1 - seat
    groups: dict[str, list[tuple[float, set[int], tuple[int, ...]]]] = {}

    def _walk(state, reach, board):
        if state.is_terminal():
            return
        player = state.current_player()
        if player == seat:
            groups.setdefault(state.information_state_string(seat), []).append(
                (
                    reach,
                    {c for c, owner in board.items() if owner == opponent},
                    tuple(state.legal_actions()),
                )
            )
        # Both players' probabilities, the observer's included.
        probs = policy.action_probabilities(state, player)
        for a in state.legal_actions():
            next_board = dict(board)
            next_board.setdefault(a, player)
            _walk(state.child(a), reach * probs.get(a, 0.0), next_board)

    _walk(game.new_initial_state(), 1.0, {})

    out: dict[str, dict[int, float]] = {}
    for info, parts in groups.items():
        total = sum(r for r, _, _ in parts)
        if total <= 0.0:
            continue
        unknown = parts[0][2]
        marg = {c: 0.0 for c in unknown}
        for reach, opp_cells, _ in parts:
            if reach <= 0.0:
                continue
            for c in unknown:
                if c in opp_cells:
                    marg[c] += reach / total
        out[info] = marg
    return out


@pytest.mark.parametrize("seat,n_infosets", [(0, 33), (1, 101)])
def test_full_reach_posterior_agrees_at_both_seats(dh_game_policy, seat, n_infosets):
    """The standing gate for the observer's-reach-cancels premise.

    Weighting by opponent reach alone is only correct because the observer's own
    reach is identical across the worlds of one of its info-sets and divides out.
    That is a claim about Dark Hex's information structure, and this test makes
    it something the suite would notice breaking. Full-reach Bayes, weighting every world by both players'
    probabilities, is the same posterior if and only if the premise holds.

    What this adds over `_brute_force_occupancy_maps`: that sibling drops the
    observer's reach exactly as the module does, so it inherits the premise and
    cannot certify it. If the premise were false, the module and the sibling
    would be wrong together and agree. This test is the one that would notice.
    """
    game, policy = dh_game_policy
    expected = _full_reach_occupancy_maps(game, policy, seat=seat)
    got = _module_occupancy_maps(game, policy, seat=seat)

    assert len(expected) == n_infosets, (
        "every info-set must be reachable under the full profile, else the "
        "comparison silently skips the ones that are not"
    )
    assert set(got) == set(expected)
    nonzero = 0
    for info, marg in expected.items():
        assert set(got[info]) == set(marg), f"cell key set differs at {info!r}"
        for cell, p in marg.items():
            assert got[info][cell] == pytest.approx(p, abs=1e-12)
            nonzero += int(p > 1e-9)
    assert nonzero > 0, "all-zero marginals everywhere would make this vacuous"


def test_gather_p0_worlds_is_exactly_the_seat0_delegate(dh_game_policy):
    """P-delegate: the kept name is the seat-0 call, not a parallel walk.
    """
    game, policy = dh_game_policy
    legacy = gather_p0_worlds(game, policy)
    general = gather_worlds(game, policy, seat=0)
    assert list(legacy) == list(general)
    for info in legacy:
        a, b = legacy[info], general[info]
        assert len(a) == len(b)
        for (sa, ra), (sb, rb) in zip(a, b, strict=False):
            assert sa.history() == sb.history()
            assert ra == rb


def test_dual_path_reach_matches_at_seat1(dh_game_policy):
    """The seat-1 half of the silent-wrong-posterior mitigation.

    `opp_reach_from_history` replays the history multiplying the OPPONENT's
    probabilities; at seat 1 the opponent is player 0.
    """
    game, policy = dh_game_policy
    worlds = gather_worlds(game, policy, seat=1)
    checked = 0
    for wlist in worlds.values():
        for state, reach in wlist:
            indep = opp_reach_from_history(game, state, policy, seat=1)
            assert abs(reach - indep) < 1e-9
            checked += 1
    assert checked > 100  # 2x2 has 101 seat-1 info-sets, so many more worlds


def test_opponent_stone_cells_follows_the_seat(dh_game_policy):
    """At seat 1 the hidden stones are player 0's, not player 1's.
    """
    game, _ = dh_game_policy
    s = game.new_initial_state()
    s.apply_action(0)  # P0 -> cell 0
    s.apply_action(3)  # P1 -> cell 3
    assert opponent_stone_cells(s, game, seat=0) == {3}
    assert opponent_stone_cells(s, game, seat=1) == {0}


def test_unknown_seat_raises_everywhere(dh_game_policy):
    """No silent fallbacks: a bad seat must not quietly score seat 0.
    """
    game, policy = dh_game_policy
    with pytest.raises(ValueError, match="seat"):
        gather_worlds(game, policy, seat=2)
    with pytest.raises(ValueError, match="seat"):
        opponent_stone_cells(game.new_initial_state(), game, seat=-1)
    with pytest.raises(ValueError, match="seat"):
        cell_occupancy_posterior([], game, seat=2)
    with pytest.raises(ValueError, match="seat"):
        opp_reach_from_history(game, game.new_initial_state(), policy, seat=2)
