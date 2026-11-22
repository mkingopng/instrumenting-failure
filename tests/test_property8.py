"""Property 8 (Dark Hex value of information) - scorer + measured 2x2 signal.

2x2 Dark Hex has rich VoI signal, despite looking degenerate: 25 of 33 P0
info-states have VoI* > 0 (mean 0.32, max 1.0). The forced win is *achieved
through probing*, so information is valuable at intermediate nodes. P8 is
therefore measurable at 2x2; larger boards (3x3+) belong to the depth axis.
"""

from __future__ import annotations

import polars as pl
import pytest

pyspiel = pytest.importorskip("pyspiel")

from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_darkhex
from lmopt.scoring.darkhex_posterior import gather_p0_worlds
from lmopt.scoring.property8 import (
    _voi_and_action_values,
    score_property8,
)


@pytest.fixture(scope="module")
def dh():
    game = load_darkhex()
    return game, solve_cfr(game, 300)


def test_voi_has_signal_at_2x2(dh):
    """Measured: 2x2 has real VoI (not degenerate) - many nodes with VoI* > 0."""
    game, policy = dh
    worlds = gather_p0_worlds(game, policy)
    vois = [_voi_and_action_values(wl, policy)[1] for wl in worlds.values()]
    informative = sum(1 for v in vois if v > 1e-6)
    assert informative >= 10  # rich signal (measured ~25/33)
    assert max(vois) > 0.9  # a node where info is worth ~a full win
    assert all(v >= -1e-9 for v in vois)  # EVPI is non-negative


def _find_spread_node(worlds, policy, seat: int = 0):
    """An info-state of `seat` where information is worth something (VoI* >
    0.1) and the actions genuinely differ in value, so a wrong action is
    visibly wrong. `seat` reaches `_voi_and_action_values`' own seat argument:
    the continuation values must be read at the acting seat's
    index, since Dark Hex is zero-sum and index 0 is the negation of index 1.
    """
    for info, wl in worlds.items():
        va, voi = _voi_and_action_values(wl, policy, seat=seat)
        if voi > 0.1 and va and (max(va.values()) - min(va.values())) > 0.1:
            return info, va, voi
    return None


def _seat_action_values(worlds_for_info, policy, seat: int) -> dict:
    """V(a|I) at `seat`'s own payoff index, computed independently of
    `_voi_and_action_values`.

    An oracle, deliberately duplicating only the reach-weighting definition
    of V(a|I) over OpenSpiel's own `policy_value`. It
    exists because a test that picks its expected action with the function under
    test cannot see a wrong payoff index: the error cancels on both sides.
    """
    from open_spiel.python.algorithms.expected_game_score import policy_value

    total = sum(w for _, w in worlds_for_info)
    legal = worlds_for_info[0][0].legal_actions()
    values = {a: 0.0 for a in legal}
    for state, reach in worlds_for_info:
        if reach <= 0.0:
            continue
        w = reach / total
        for a in legal:
            values[a] += w * policy_value(state.child(a), [policy, policy])[seat]
    return values


def test_scorer_scores_shortfall_and_capture(dh):
    game, policy = dh
    worlds = gather_p0_worlds(game, policy)
    target = _find_spread_node(worlds, policy)
    assert target is not None
    info, va, _ = target
    best_a = max(va, key=va.get)
    worst_a = min(va, key=va.get)

    def _row(action):
        return {
            "game": "darkhex",
            "prompt": "p",
            "player_to_act": 0,
            "info_set": info,
            "parsed_action": action,
        }

    refs = dict(cfr_policies={"darkhex": policy}, games={"darkhex": game})

    # Worst action at a high-VoI node -> positive P8 shortfall.
    s = score_property8(pl.DataFrame([_row(worst_a)]), **refs)
    assert s["available"] and s["n_informative"]["darkhex"] == 1
    assert s["p8_shortfall"]["darkhex"] > 0.0
    assert s["capture_rate"]["darkhex"] == 0.0

    # Best action -> zero shortfall, full capture.
    s2 = score_property8(pl.DataFrame([_row(best_a)]), **refs)
    assert abs(s2["p8_shortfall"]["darkhex"]) < 1e-9
    assert s2["capture_rate"]["darkhex"] == 1.0
    # Pin the other direction of the partial-reference guard: a full
    # CFR+ TabularPolicy must not be misidentified as partial (it has neither
    # `lookup` nor `n_infosets`), so P8 must not silently disable itself for
    # the canonical games.
    assert "darkhex" not in s2.get("not_applicable", {})


def test_scorer_unavailable_without_refs(dh):
    """`available=False` alone collapses three distinct causes, so a caller
    such as `analyse_stats.py` could otherwise say "P8 is defined for Dark
    Hex only" even when the real problem is a missing reference, not a
    game-applicability question. This case is
    the "no CFR reference/game object supplied to the scorer at all" cause
    specifically -- a real Dark Hex row, no `cfr_policies`/`games` at all.
    """
    df = pl.DataFrame(
        [
            {
                "game": "darkhex",
                "prompt": "p",
                "player_to_act": 0,
                "info_set": "x",
                "parsed_action": 0,
            }
        ]
    )
    result = score_property8(df)
    assert result["available"] is False
    assert "darkhex" in result["not_applicable"], (
        "a real Dark Hex row with no reference supplied at all must carry a "
        "specific reason, not silently drop out of not_applicable"
    )
    assert (
        "no CFR reference or game object was supplied"
        in result["not_applicable"]["darkhex"]
    )


def test_scorer_missing_reference_for_this_specific_game_is_refused_with_a_reason():
    """The second distinct `available=False` cause: the
    `cfr_policies`/`games` dicts are real and non-empty, but neither one has
    an entry for this darkhex-family game -- the shape of `darkhex_3x3`,
    which has no reference. Distinct from both
    `test_scorer_unavailable_without_refs` (dicts absent entirely) and
    `test_p8_partial_reference_is_refused_with_a_reason` (a reference exists
    but is partial) -- a caller reading only `available` cannot tell any of
    these three apart without this dict.
    """
    df = pl.DataFrame(
        [
            {
                "game": "darkhex",
                "prompt": "p",
                "player_to_act": 0,
                "info_set": "x",
                "parsed_action": 0,
            }
        ]
    )
    # Real, non-empty dicts, but keyed under a different game than the one
    # being scored -- "darkhex" itself is absent from both, mirroring
    # darkhex_3x3 before its own reference exists while some other game's
    # reference is present in the same scoring run. The placeholder values
    # are never touched (the missing-key check fires first), so their
    # actual type doesn't matter.
    result = score_property8(
        df, cfr_policies={"kuhn": object()}, games={"kuhn": object()}
    )
    assert result["available"] is False
    assert "darkhex" in result["not_applicable"]
    assert (
        "no CFR reference available for darkhex yet"
        == result["not_applicable"]["darkhex"]
    )


def test_p8_partial_reference_is_refused_with_a_reason(tmp_path, dh):
    """A PARTIAL depth-reference policy must not be handed
    to gather_p0_worlds's full-tree walk - the off-support gaps make VoI
    undefined, and (at 3x3 scale) the tree is intractable."""
    import pickle

    from lmopt.games.depth_reference import load_depth_reference
    from tests.test_depth_reference import _truncated_kuhn_payload

    game, _ = dh
    payload = _truncated_kuhn_payload(game, keep=2)
    path = tmp_path / "trunc.pkl"
    path.write_bytes(pickle.dumps(payload))
    partial_ref = load_depth_reference(path, game)

    row = {
        "game": "darkhex",
        "prompt": "p",
        "player_to_act": 0,
        "info_set": "irrelevant",
        "parsed_action": 0,
    }
    df = pl.DataFrame([row])
    result = score_property8(
        df, cfr_policies={"darkhex": partial_ref}, games={"darkhex": game}
    )
    assert "darkhex" in result["not_applicable"]
    assert result["not_applicable"]["darkhex"], "reason string must be non-empty"
    assert (
        "darkhex" not in result["p8_shortfall"]
    ), "a partial reference must not produce a P8 score"


class TestBothSeatsReachTheEstimand:
    """P8 must score rows from both seats.

    Building one worlds map at `seat=0` and keeping only `player_to_act == 0`
    rows would make a `seat_mode="random"` arm lose about half its P8 rows
    with no count, no reason string and no null. On the 2x2 board: 33 seat-0
    info-sets over 49 worlds against 101 seat-1 info-sets over 176 worlds,
    sharing none, each walk ~0.001 s.
    """

    @staticmethod
    def _row(info, action, player_to_act, lm_seat=None):
        row = {
            "game": "darkhex",
            "prompt": "p",
            "player_to_act": player_to_act,
            "info_set": info,
            "parsed_action": action,
        }
        if lm_seat is not None:
            row["lm_seat"] = lm_seat
        return row

    def test_rows_from_both_seats_are_scored(self, dh):
        """P8-both-seats. Two hands, `lm_seat` 0 on one and 1 on the other.
        """
        game, policy = dh
        from lmopt.scoring.darkhex_posterior import gather_worlds

        info0, va0, _ = _find_spread_node(
            gather_worlds(game, policy, seat=0), policy, seat=0
        )
        info1, va1, _ = _find_spread_node(
            gather_worlds(game, policy, seat=1), policy, seat=1
        )
        rows = [
            self._row(info0, max(va0, key=va0.get), player_to_act=0, lm_seat=0),
            self._row(info1, max(va1, key=va1.get), player_to_act=1, lm_seat=1),
        ]
        out = score_property8(
            pl.DataFrame(rows),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["available"]
        assert out["n_nodes"]["darkhex"] == 2
        assert out["n_informative"]["darkhex"] == 2
        # Both rows played their own seat's best action.
        assert out["capture_rate"]["darkhex"] == 1.0

    def test_seat_one_values_are_read_at_the_seat_one_index(self, dh):
        """Dark Hex is zero-sum, so `state.returns()[0] == -returns()[1]`: a
        seat-1 row scored at index 0 has every V(a|I) negated, which turns the
        best action into the worst one.

        The expected best action comes from `_seat_action_values`, an
        independent oracle over OpenSpiel's own `policy_value`, not from
        `_voi_and_action_values`. Deriving it from the function under mutation
        makes this test pass under its own mutation: the wrong index cancels
        on both sides.
        """
        game, policy = dh
        from lmopt.scoring.darkhex_posterior import gather_worlds

        worlds1 = gather_worlds(game, policy, seat=1)
        info1, _, _ = _find_spread_node(worlds1, policy, seat=1)
        oracle = _seat_action_values(worlds1[info1], policy, seat=1)
        best = max(oracle, key=oracle.get)
        assert (
            oracle[best] > min(oracle.values()) + 0.1
        ), "fixture needs a genuine spread or the sign flip is invisible"
        # The zero-sum negation itself, at the mechanism: index 1 is minus
        # index 0 over the same worlds, so the two maps cannot be interchanged.
        at_seat_0 = _seat_action_values(worlds1[info1], policy, seat=0)
        assert oracle[best] == pytest.approx(-at_seat_0[best], abs=1e-12)
        assert oracle[best] != pytest.approx(at_seat_0[best], abs=1e-9)

        out = score_property8(
            pl.DataFrame([self._row(info1, best, player_to_act=1, lm_seat=1)]),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["n_informative"]["darkhex"] == 1
        assert out["capture_rate"]["darkhex"] == 1.0
        assert abs(out["p8_shortfall"]["darkhex"]) < 1e-9

    def test_a_worlds_map_is_built_only_for_the_seats_present(self, dh, monkeypatch):
        """A frame with no seat-1 row must not pay for the seat-1 walk: at 2x3
        that walk is 5,193 info-sets over 38,428 worlds, and the precompute measured ~0.53 s/match under
        random seats against ~0.21 s fixed.
        """
        from lmopt.scoring import darkhex_posterior

        game, policy = dh
        info0, va0, _ = _find_spread_node(
            darkhex_posterior.gather_worlds(game, policy, seat=0), policy, seat=0
        )
        info1, va1, _ = _find_spread_node(
            darkhex_posterior.gather_worlds(game, policy, seat=1), policy, seat=1
        )

        seats: list[int] = []
        real = darkhex_posterior.gather_worlds

        def _spy(g, pol, seat=0):
            seats.append(seat)
            return real(g, pol, seat=seat)

        monkeypatch.setattr(darkhex_posterior, "gather_worlds", _spy)
        refs = dict(cfr_policies={"darkhex": policy}, games={"darkhex": game})

        score_property8(
            pl.DataFrame([self._row(info0, max(va0, key=va0.get), 0, lm_seat=0)]),
            **refs,
        )
        assert seats == [0]
        seats.clear()
        score_property8(
            pl.DataFrame([self._row(info1, max(va1, key=va1.get), 1, lm_seat=1)]),
            **refs,
        )
        assert seats == [1], "the seat built must be the OCCUPIED seat, not a literal 0"
        seats.clear()
        score_property8(
            pl.DataFrame(
                [
                    self._row(info0, max(va0, key=va0.get), 0, lm_seat=0),
                    self._row(info1, max(va1, key=va1.get), 1, lm_seat=1),
                ]
            ),
            **refs,
        )
        assert sorted(seats) == [0, 1], "a mixed frame needs both maps, once each"


class TestEveryDroppedRowIsCountedWithAReason:
    """Every row P8 drops is counted with a reason.

    An uncounted `if not wl: continue` in `property8.py` would drop rows
    against a different key space per seat: on darkhex 2x3, a seat-0 row
    misses against 2,053 info-states and a seat-1 row against 5,193. A single uncounted `n_nodes` cannot tell "the
    agent played few Dark Hex nodes" from "half its nodes were unreachable in
    the map we built".
    """

    @staticmethod
    def _refs(dh):
        game, policy = dh
        return (
            game,
            policy,
            dict(cfr_policies={"darkhex": policy}, games={"darkhex": game}),
        )

    @staticmethod
    def _row(info, action, seat):
        return {
            "game": "darkhex",
            "prompt": "p",
            "player_to_act": seat,
            "info_set": info,
            "parsed_action": action,
        }

    def test_a_worlds_map_miss_is_counted_per_seat_with_a_reason(self, dh):
        """Two rows whose `info_set` is in neither seat's map, one per seat.
        """
        game, policy, refs = self._refs(dh)
        from lmopt.scoring.darkhex_posterior import gather_worlds

        bogus = "NOT-AN-INFO-STATE"
        assert bogus not in gather_worlds(game, policy, seat=0)
        assert bogus not in gather_worlds(game, policy, seat=1)

        out = score_property8(
            pl.DataFrame(
                [self._row(bogus, 0, 0), self._row(bogus, 0, 1), self._row(bogus, 0, 1)]
            ),
            **refs,
        )
        d = out["discarded_rows"]["darkhex"]
        assert d["n_rows"] == 3
        assert (
            out["n_nodes"]["darkhex"] == 0
        ), "a dropped row must not also be counted as a scored node"
        assert d["by_seat"]["0"]["n_rows"] == 1
        assert d["by_seat"]["1"]["n_rows"] == 2
        assert d["by_seat"]["0"]["distinct_info_sets"] == 1
        # The per-seat key spaces are different sizes, which is the whole reason
        # the count is per seat; on 2x2: 33 against 101.
        assert (
            d["by_seat"]["0"]["worlds_map_info_sets"]
            != d["by_seat"]["1"]["worlds_map_info_sets"]
        )
        assert "worlds map" in d["reason"] and "seat" in d["reason"]

    def test_a_frame_with_no_miss_carries_no_discarded_entry(self, dh):
        """The negative control for the counter, matching `score_property7`'s
        convention: absent, not a zero.
        """
        game, policy, refs = self._refs(dh)
        from lmopt.scoring.darkhex_posterior import gather_worlds

        info0, va0, _ = _find_spread_node(
            gather_worlds(game, policy, seat=0), policy, seat=0
        )
        out = score_property8(
            pl.DataFrame([self._row(info0, max(va0, key=va0.get), 0)]), **refs
        )
        assert out["discarded_rows"] == {}
        assert out["n_nodes"]["darkhex"] == 1

    def test_a_null_parsed_action_is_counted_in_n_null_action(self, dh):
        """
        an informative row whose parsed_action is null is counted in ``n_null_action`` rather
        than crashing the scorer, and the scored row beside it is scored as alone
        """
        game, policy, refs = self._refs(dh)
        from lmopt.scoring.darkhex_posterior import gather_worlds

        info0, va0, _ = _find_spread_node(
            gather_worlds(game, policy, seat=0), policy, seat=0
        )
        good = self._row(info0, max(va0, key=va0.get), 0)
        out = score_property8(pl.DataFrame([good, dict(good, parsed_action=None)]), **refs)
        alone = score_property8(pl.DataFrame([good]), **refs)
        assert out["n_null_action"] == {"darkhex": 1}
        assert out["p8_shortfall"] == alone["p8_shortfall"]
        assert out["n_informative"] == alone["n_informative"]

    def test_a_no_voi_node_is_excluded_by_name_not_by_subtraction(self, dh):
        """A node with VoI* <= tau enters `n_nodes` and leaves the estimand.
        """
        game, policy, refs = self._refs(dh)
        from lmopt.scoring.darkhex_posterior import gather_worlds

        worlds = gather_worlds(game, policy, seat=0)
        flat = [
            i
            for i, wl in worlds.items()
            if _voi_and_action_values(wl, policy, seat=0)[1] <= 1e-2
        ]
        assert flat, "2x2 must have at least one zero-VoI seat-0 info-state"
        legal = worlds[flat[0]][0][0].legal_actions()
        out = score_property8(pl.DataFrame([self._row(flat[0], legal[0], 0)]), **refs)
        assert out["n_nodes"]["darkhex"] == 1
        assert out["n_informative"]["darkhex"] == 0
        excl = out["excluded_rows"]["darkhex"]["voi_at_or_below_threshold"]
        assert excl["n_rows"] == 1
        assert "voi_threshold" in excl["reason"]
        assert (
            "unreachable_info_set" not in out["excluded_rows"]["darkhex"]
        ), "a zero-VoI node is reachable; the two exclusions are not the same"

    def test_an_unreachable_info_set_is_a_different_exclusion(self, dh, monkeypatch):
        """Total opponent reach 0 => empty posterior => no V(a|I) at all. This
        is not "information is worth nothing here"; it is "there is nothing to
        weight by", and merging them would report an absent posterior as a
        measured zero.

        No 2x2 info-state has zero reach under CFR+ (0 of 33 at seat 0 and 0
        of 101 at seat 1), so the condition is injected by
        zeroing one info-state's reaches in the map the scorer receives. The
        states are real; only the weights are forced.
        """
        from lmopt.scoring import darkhex_posterior

        game, policy, refs = self._refs(dh)
        real = darkhex_posterior.gather_worlds
        info0, va0, _ = _find_spread_node(real(game, policy, seat=0), policy, seat=0)

        def _zeroed(g, pol, seat=0):
            wl = real(g, pol, seat=seat)
            if seat == 0:
                wl = dict(wl)
                wl[info0] = [(st, 0.0) for st, _ in wl[info0]]
            return wl

        monkeypatch.setattr(darkhex_posterior, "gather_worlds", _zeroed)
        out = score_property8(
            pl.DataFrame([self._row(info0, max(va0, key=va0.get), 0)]), **refs
        )
        assert (
            out["n_nodes"]["darkhex"] == 1
        ), "the row is not a worlds-map MISS: the key is present"
        assert out["discarded_rows"] == {}
        excl = out["excluded_rows"]["darkhex"]
        assert excl["unreachable_info_set"]["n_rows"] == 1
        assert "voi_at_or_below_threshold" not in excl

    def test_an_action_absent_from_the_value_dict_is_excluded_and_counted_not_scored_worst(
        self, dh
    ):
        """A row whose `parsed_action` is not a key of V(a|I) has no value to compare:
        a `v_a.get(a, min(v_a.values()))` fallback would score it as the worst action,
        which turns a row the scorer cannot read into a confident miss.
        """
        from lmopt.scoring.darkhex_posterior import gather_worlds

        game, policy, refs = self._refs(dh)
        info0, va0, _ = _find_spread_node(
            gather_worlds(game, policy, seat=0), policy, seat=0
        )
        absent = max(va0) + 100
        assert absent not in va0
        out = score_property8(pl.DataFrame([self._row(info0, absent, 0)]), **refs)
        assert out["n_nodes"]["darkhex"] == 1
        assert out["n_informative"]["darkhex"] == 0
        assert out["capture_rate"]["darkhex"] != out["capture_rate"]["darkhex"]  # NaN
        excl = out["excluded_rows"]["darkhex"]["action_absent_from_value_dict"]
        assert excl["n_rows"] == 1
        assert "not a key of V(a|I)" in excl["reason"]
        assert "voi_at_or_below_threshold" not in out["excluded_rows"]["darkhex"]

        both = score_property8(
            pl.DataFrame(
                [
                    self._row(info0, absent, 0),
                    self._row(info0, max(va0, key=va0.get), 0),
                ]
            ),
            **refs,
        )
        assert both["n_informative"]["darkhex"] == 1
        assert both["capture_rate"]["darkhex"] == 1.0
        assert (
            both["excluded_rows"]["darkhex"]["action_absent_from_value_dict"]["n_rows"]
            == 1
        )


class TestShortfallIsNotBoundedByOne:
    """A measured divergence from the P8 definition's own reading.

    The P8 definition calls `P8_shortfall` "the fraction of the available value
    of information the LM failed to capture (0 = captured the
    informationally-optimal action, 1 = captured none)". Nothing bounds the
    numerator (an action-value gap on a +/-1 payoff scale, so up to 2.0) by the
    denominator (EVPI). On real arms, 8 of 16 cells carry shortfall > 1 and
    olmo-2 reaches a cell mean of 1.75.

    The scorer reports the violation instead of clipping it. Clipping would make
    the published mean look like a fraction while destroying the evidence that
    it is not one; whether to clip is an analysis decision, not the scorer's.
    """

    def test_a_worst_action_at_a_high_voi_node_scores_above_one(self, dh):
        """        The expected value comes from `_seat_action_values`, the independent
        oracle, so a wrong payoff index cannot cancel on both sides.
        """
        game, policy = dh
        from lmopt.scoring.darkhex_posterior import gather_worlds

        worlds = gather_worlds(game, policy, seat=0)
        target = None
        for info, wl in worlds.items():
            va, voi = _voi_and_action_values(wl, policy, seat=0)
            if va and voi > 1e-2 and (max(va.values()) - min(va.values())) / voi > 1.2:
                target = (info, wl, voi)
                break
        assert target is not None, "2x2 must expose a node where the gap exceeds EVPI"
        info, wl, voi = target
        oracle = _seat_action_values(wl, policy, seat=0)
        worst = min(oracle, key=oracle.get)
        expected = (max(oracle.values()) - oracle[worst]) / voi
        assert expected > 1.0, "the fixture must violate the spec's stated range"

        out = score_property8(
            pl.DataFrame(
                [
                    {
                        "game": "darkhex",
                        "prompt": "p",
                        "player_to_act": 0,
                        "info_set": info,
                        "parsed_action": worst,
                    }
                ]
            ),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["p8_shortfall"]["darkhex"] == pytest.approx(expected, abs=1e-9)
        assert out["p8_shortfall"]["darkhex"] > 1.0
        assert out["n_shortfall_gt1"]["darkhex"] == 1

    def test_the_threshold_the_numbers_were_computed_at_is_reported(self, dh):
        """tau is provisional, so a reader must be able to see which value produced a given table without re-deriving it
        from the call site.
        """
        game, policy = dh
        df = pl.DataFrame(
            [
                {
                    "game": "darkhex",
                    "prompt": "p",
                    "player_to_act": 0,
                    "info_set": "irrelevant",
                    "parsed_action": 0,
                }
            ]
        )
        refs = dict(cfr_policies={"darkhex": policy}, games={"darkhex": game})
        assert score_property8(df, **refs)["voi_threshold"] == 1e-2
        assert score_property8(df, voi_threshold=0.25, **refs)["voi_threshold"] == 0.25


class TestReferenceVisitShare:
    """P8 reports how often the reference itself visits the scored nodes.

    P8's caveat must name, from measurement, that the yardstick is read where
    the reference does not play, so the scorer reports that number.
    """

    def test_the_share_weights_the_reference_not_the_posterior(self, dh):
        """`reference_visit_weights` multiplies both players' CFR+
        probabilities; `gather_worlds` accumulates the opponent's alone,
        because the observer's own reach cancels in the posterior. Using the
        posterior's weighting here would answer a different question and
        answer it wrongly by orders of magnitude.
        """
        from lmopt.scoring.darkhex_posterior import gather_worlds
        from lmopt.scoring.property8 import reference_visit_weights

        game, policy = dh
        worlds = gather_worlds(game, policy, seat=0)
        info0, va0, _ = _find_spread_node(worlds, policy, seat=0)

        visits = reference_visit_weights(game, policy, seat=0)
        assert set(visits) <= set(
            worlds
        ), "the visit walk must key on the same info-state strings"
        share = visits.get(info0, 0.0) / sum(visits.values())
        assert 0.0 < share < 1e-4

        out = score_property8(
            pl.DataFrame(
                [
                    {
                        "game": "darkhex",
                        "prompt": "p",
                        "player_to_act": 0,
                        "info_set": info0,
                        "parsed_action": max(va0, key=va0.get),
                    }
                ]
            ),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["reference_visit_share"]["darkhex"] == pytest.approx(share)
        # The agent spent 100 % of its decision nodes here, the reference
        # essentially none: that gap is what the caveat layer compares.
        assert out["n_informative"]["darkhex"] == out["n_nodes"]["darkhex"] == 1

    def test_a_game_with_no_informative_node_pays_for_no_visit_walk(self, dh):
        """A key that is absent, not a NaN, and no tree walk behind it.
        """
        from lmopt.scoring import property8 as p8mod
        from lmopt.scoring.darkhex_posterior import gather_worlds

        game, policy = dh
        worlds = gather_worlds(game, policy, seat=0)
        flat = [
            i
            for i, wl in worlds.items()
            if _voi_and_action_values(wl, policy, seat=0)[1] <= 1e-2
        ]
        assert flat
        legal = worlds[flat[0]][0][0].legal_actions()

        calls: list[int] = []
        real = p8mod.reference_visit_weights

        def _spy(g, pol, seat):
            calls.append(seat)
            return real(g, pol, seat)

        p8mod.reference_visit_weights = _spy
        try:
            out = score_property8(
                pl.DataFrame(
                    [
                        {
                            "game": "darkhex",
                            "prompt": "p",
                            "player_to_act": 0,
                            "info_set": flat[0],
                            "parsed_action": legal[0],
                        }
                    ]
                ),
                cfr_policies={"darkhex": policy},
                games={"darkhex": game},
            )
        finally:
            p8mod.reference_visit_weights = real
        assert calls == []
        assert "darkhex" not in out["reference_visit_share"]


class TestActorParameterisation:
    def test_a_named_opponent_is_scored_over_its_own_rows(self, dh):
        """On P8 the population is an `actor` predicate.
        """
        game, policy = dh
        from lmopt.scoring.darkhex_posterior import gather_worlds

        info0, va0, _ = _find_spread_node(
            gather_worlds(game, policy, seat=0), policy, seat=0
        )
        rows = [
            {
                "game": "darkhex",
                "prompt": None,
                "actor": "bayesian",
                "player_to_act": 0,
                "lm_seat": 0,
                "info_set": info0,
                "parsed_action": max(va0, key=va0.get),
            }
        ]
        out = score_property8(
            pl.DataFrame(rows),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
            actor="bayesian",
        )
        assert out["actor_filter"] == "actor=bayesian"
        assert out["n_nodes"]["darkhex"] == 1
        assert out["capture_rate"]["darkhex"] == 1.0

    def test_a_frame_without_player_to_act_is_named_not_a_bare_keyerror(self, dh):
        """`player_to_act` is the seat the estimand belongs to; read unguarded
        out of an `iter_rows(named=True)` dict, a frame lacking it would raise
        `KeyError: 'player_to_act'` naming neither the scorer nor the frame.
        P2's sibling read degrades to a silent
        seat-0 default on the same defective frame; P8 must not, because it has
        no seat-0 estimand to fall back to.
        """
        game, policy = dh
        from lmopt.scoring.darkhex_posterior import gather_worlds

        info0, va0, _ = _find_spread_node(
            gather_worlds(game, policy, seat=0), policy, seat=0
        )
        df = pl.DataFrame(
            [
                {
                    "game": "darkhex",
                    "prompt": "p",
                    "info_set": info0,
                    "parsed_action": max(va0, key=va0.get),
                }
            ]
        )
        assert "player_to_act" not in df.columns
        with pytest.raises(ValueError, match="player_to_act"):
            score_property8(
                df, cfr_policies={"darkhex": policy}, games={"darkhex": game}
            )

    def test_legacy_frame_scores_byte_identically_and_says_so(self, dh):
        """P-legacy for P8: no `actor` and no `lm_seat` column, LM at seat 0
        on every row, so the numbers match the prompt-null reading.
        """
        game, policy = dh
        from lmopt.scoring.darkhex_posterior import gather_worlds

        info0, va0, _ = _find_spread_node(
            gather_worlds(game, policy, seat=0), policy, seat=0
        )
        df = pl.DataFrame(
            [
                {
                    "game": "darkhex",
                    "prompt": "p",
                    "player_to_act": 0,
                    "info_set": info0,
                    "parsed_action": max(va0, key=va0.get),
                }
            ]
        )
        assert "actor" not in df.columns and "lm_seat" not in df.columns
        out = score_property8(
            df, cfr_policies={"darkhex": policy}, games={"darkhex": game}
        )
        assert out["actor_filter"] == "legacy-prompt-null"
        assert out["n_nodes"]["darkhex"] == 1
        assert abs(out["p8_shortfall"]["darkhex"]) < 1e-9

    def test_naming_an_actor_a_legacy_frame_cannot_identify_raises(self, dh):
        game, policy = dh
        df = pl.DataFrame(
            [
                {
                    "game": "darkhex",
                    "prompt": "p",
                    "player_to_act": 0,
                    "info_set": "irrelevant",
                    "parsed_action": 0,
                }
            ]
        )
        with pytest.raises(ValueError, match="actor"):
            score_property8(
                df,
                cfr_policies={"darkhex": policy},
                games={"darkhex": game},
                actor="bayesian",
            )


class TestForeignProbeType:
    """Without a probe_type guard, a frame assembled from a directory that also
    holds a probe arm's rows (reachable via an unfiltered
    `--data` glob) would be silently pooled into the VoI estimate.
    """

    def _row(self, probe_type):
        return {
            "game": "darkhex",
            "prompt": "p",
            "player_to_act": 0,
            "info_set": "irrelevant",
            "parsed_action": 0,
            "probe_type": probe_type,
        }

    def test_refuses_a_frame_holding_a_foreign_probe_type(self):
        df = pl.DataFrame([self._row("none"), self._row("payoff_scale")])
        with pytest.raises(ValueError, match="payoff_scale"):
            score_property8(df)

    def test_accepts_an_all_base_frame(self):
        df = pl.DataFrame([self._row("none")])
        score_property8(df)  # must not raise ("available" False is fine)


class TestNBoardInformative:
    """An arm can read `n_informative=0` on every darkhex opponent (as
    `M1-olmo-2-1124-13b-instruct`'s properties.csv does), which alone is
    indistinguishable from `score_property8`'s own module-docstring reading
    of "an all-degenerate board". 2x2 Dark Hex is measured not degenerate
    (`test_voi_has_signal_at_2x2` above), and other arms score real, nonzero
    `n_informative` on the same board -- such a zero is a fact about that
    arm's own logged rows, not the board. `n_board_informative`
    reports the reference's own informative-node count (over every info-state
    in the worlds map for the seat(s) the arm occupied, not just the ones
    visited) so a caller can tell the two apart.
    """

    def _low_voi_info(self, worlds, policy, seat: int):
        """A real info-state on the live 2x2 board with VoI* at or below the
        default threshold (1e-2) -- not invented, not mocked. Seat 0 has 8
        exact-zero-VoI info-states out of 33
        (property8.py's own `score_property8` docstring), so one must exist.
        """
        for info, wl in worlds.items():
            _, voi = _voi_and_action_values(wl, policy, seat=seat)
            if voi <= 1e-2:
                return info
        raise AssertionError(f"seat {seat} has no low-VoI info-state on this board")

    def test_reports_the_boards_own_count_even_when_the_arm_never_visits_it(self, dh):
        """A single row at a real, LOW-VoI seat-0 info-state: this arm's own
        `n_informative` is 0, but the board itself has real signal
        (`test_voi_has_signal_at_2x2`'s own >= 10 bound, reused here so both
        tests stay consistent with the same measured board).
        """
        from lmopt.scoring.darkhex_posterior import gather_worlds

        game, policy = dh
        worlds0 = gather_worlds(game, policy, seat=0)
        info0 = self._low_voi_info(worlds0, policy, seat=0)
        out = score_property8(
            pl.DataFrame(
                [
                    {
                        "game": "darkhex",
                        "prompt": "p",
                        "player_to_act": 0,
                        "info_set": info0,
                        "parsed_action": 0,
                    }
                ]
            ),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["n_informative"]["darkhex"] == 0
        assert out["n_board_informative"]["darkhex"] >= 10

    def test_counts_the_distinct_info_states_the_rows_reached_and_says_so(self, dh):
        """Three rows at two real, low-VoI seat-0 info-states: the arm reached 2 distinct
        info-states, none informative, and the zero-informative caveat carries that count, so
        the cell says why the arm scored no informative node.
        """
        from lmopt.scoring.darkhex_posterior import gather_worlds

        from scripts.analyse_stats import _p8_zero_informative_clause

        game, policy = dh
        worlds0 = gather_worlds(game, policy, seat=0)
        low = [
            info
            for info, wl in worlds0.items()
            if _voi_and_action_values(wl, policy, seat=0)[1] <= 1e-2
        ][:2]
        assert len(low) == 2
        rows = [
            {
                "game": "darkhex",
                "prompt": "p",
                "player_to_act": 0,
                "info_set": info,
                "parsed_action": 0,
            }
            for info in (low[0], low[0], low[1])
        ]
        out = score_property8(
            pl.DataFrame(rows),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["n_informative"]["darkhex"] == 0
        assert out["n_visited_infosets"]["darkhex"] == 2
        text = _p8_zero_informative_clause(out, "darkhex")
        assert "reached only 2 distinct info-state(s), none of them informative" in text

    def test_is_seat_scoped_not_a_fixed_seat_zero_default(self, dh):
        """An arm that occupies only seat 1 must get seat 1's own board
        count, not seat 0's -- seats are not symmetric in Dark Hex
        (`gather_worlds`'s own docstring; seat 0 has 12 informative
        info-states of 33, seat 1 has 2 of 101).
        """
        from lmopt.scoring.darkhex_posterior import gather_worlds

        game, policy = dh
        worlds1 = gather_worlds(game, policy, seat=1)
        info1 = self._low_voi_info(worlds1, policy, seat=1)
        out = score_property8(
            pl.DataFrame(
                [
                    {
                        "game": "darkhex",
                        "prompt": "p",
                        "player_to_act": 1,
                        "info_set": info1,
                        "parsed_action": 0,
                    }
                ]
            ),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["n_informative"]["darkhex"] == 0
        assert out["n_board_informative"]["darkhex"] == 2

    def test_is_zero_on_a_genuinely_degenerate_reference(self, dh, monkeypatch):
        """The isolation-coverage counterpart to the two tests above: a
        mutation that always reports some nonzero `n_board_informative`
        (e.g. hardcoding a placeholder) would pass those two by luck but
        must be caught here, where the true board-level count is forced to
        zero everywhere via a monkeypatched `_voi_and_action_values`
        (defined in this module, not imported -- `monkeypatch.setattr` on
        the module object still works, module-global lookup at call time).
        """
        import lmopt.scoring.property8 as property8_module

        game, policy = dh

        def _zero_voi(worlds_for_info, cfr_pol, seat=0):
            legal = worlds_for_info[0][0].legal_actions()
            return {a: 0.0 for a in legal}, 0.0

        monkeypatch.setattr(property8_module, "_voi_and_action_values", _zero_voi)
        out = score_property8(
            pl.DataFrame(
                [
                    {
                        "game": "darkhex",
                        "prompt": "p",
                        "player_to_act": 0,
                        "info_set": "irrelevant-since-voi-is-mocked",
                        "parsed_action": 0,
                    }
                ]
            ),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["n_informative"]["darkhex"] == 0
        assert out["n_board_informative"]["darkhex"] == 0

    def test_all_miss_cell_reports_no_data_not_arm_specific(self, dh):
        """A row whose `info_set` does not exist in the
        reference's worlds map (e.g. logged on a different board/OpenSpiel
        version) is a genuine 100%-miss cell -- `n_nodes=0`, the miss
        recorded in `discarded_rows`, zero real decisions scored. But
        `worlds_by_seat` (the source `n_board_informative` walks) is built
        for a seat as soon as any row claims it, before that row's
        `info_set` is checked against the map -- so this cell still carries
        the seat's real, nonzero `n_board_informative`. A caveat-clause
        helper that checks `n_board_informative` before `n_nodes` reports
        this exact cell as "ARM-specific ... not board degeneracy" -- a
        confident, false claim that real decisions were made and avoided
        informative nodes, when in fact none were scored at all.

        Through the real production path end to end: `score_property8` on a
        genuine lookup-miss row, then `analyse_stats._p8_zero_informative_
        clause` on its real output (not a hand-built dict)."""
        from scripts.analyse_stats import _p8_zero_informative_clause

        game, policy = dh
        out = score_property8(
            pl.DataFrame(
                [
                    {
                        "game": "darkhex",
                        "prompt": "p",
                        "player_to_act": 0,
                        "info_set": "this-info-set-does-not-exist-on-2x2",
                        "parsed_action": 0,
                    }
                ]
            ),
            cfr_policies={"darkhex": policy},
            games={"darkhex": game},
        )
        assert out["n_nodes"]["darkhex"] == 0, "fixture must be a genuine miss"
        assert out["n_informative"]["darkhex"] == 0
        assert out["n_board_informative"]["darkhex"] > 0, (
            "fixture must reproduce the real hazard: a nonzero board count "
            "alongside zero real scored rows"
        )
        assert (
            "darkhex" in out["discarded_rows"]
        ), "the miss must be recorded (no silent fallbacks)"

        text = _p8_zero_informative_clause(out, "darkhex")
        assert text is not None
        assert "arm-specific" not in text.lower()
        assert "informative info-state" not in text.lower()
        assert "n_nodes=0" in text.lower()
