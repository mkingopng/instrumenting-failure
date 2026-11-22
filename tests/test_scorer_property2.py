"""Tests for property2 scorer (Adaptivity - best-response gap)."""

from __future__ import annotations

import math
import subprocess
import sys

import polars as pl
import pytest

pyspiel = pytest.importorskip("pyspiel")

from lmopt.games.cfr import nash_conv, solve_cfr
from lmopt.games.loaders import load_kuhn
from lmopt.scoring.property2 import (
    _build_empirical_policy,
    _compute_br_value,
    _compute_br_value_and_coverage,
    score_property2,
)


@pytest.fixture(scope="module")
def kuhn_cfr():
    """500 CFR+ iterations, shared by every test in this module.

    `test_p2_br_value_matches_analytic_kuhn_equilibrium`
    ties its tolerance to `nash_conv(game, policy)`, recomputed at test time --
    not a literal. So lowering this fixture's iteration count loosens that
    tolerance (nash_conv grows as CFR+ under-converges), it does not tighten
    it or make the test more brittle. The actual residual fragility is
    `_kuhn_df_from_policy`'s `n_samples` (rounding noise from approximating
    real-valued probabilities as integer row counts), independent of this
    fixture: at `n_samples=2000` the deviation from -1/18 is 0.0001389
    against a tolerance (this fixture's own nash_conv) of
    0.00034539369948721754 -- a 2.49x margin, not a large one.
    """
    game = load_kuhn()
    cfr_pol = solve_cfr(game, 500)
    return game, cfr_pol


def _lm_row(
    matchup_id: str,
    game: str,
    hand_idx: int,
    reward: float,
    fallback_used: bool = False,
) -> dict:
    return {
        "matchup_id": matchup_id,
        "game": game,
        "hand_idx": hand_idx,
        "reward": reward,
        "prompt": "some prompt",
        "player_to_act": 0,
        "info_set": "0",
        "parsed_action": 0,
        "fallback_used": fallback_used,
    }


def _opp_row(
    matchup_id: str, game: str, hand_idx: int, action: int, info_set: str = "1pb"
) -> dict:
    return {
        "matchup_id": matchup_id,
        "game": game,
        "hand_idx": hand_idx,
        "reward": 0.0,
        "prompt": None,
        "player_to_act": 1,
        "info_set": info_set,
        "parsed_action": action,
    }


class TestBRGap:
    def test_gap_structure_without_games(self):
        """Without games dict, gap is nan but other keys present."""
        rows = [_lm_row("lm_vs_cfr", "kuhn", i, 0.0) for i in range(5)]
        df = pl.DataFrame(rows)
        result = score_property2(df)
        assert math.isnan(result["br_gap"]["lm_vs_cfr"]["kuhn"])
        assert not math.isnan(result["lm_mean_reward"]["lm_vs_cfr"]["kuhn"])

    def test_gap_nonnegative_against_cfr_opponent(self, kuhn_cfr):
        """BR value against any policy ≥ LM mean reward (by definition of BR)."""
        game, cfr_pol = kuhn_cfr
        # LM rows: 10 hands, reward varies
        lm_rows = [
            _lm_row("lm_vs_cfr", "kuhn", i, float((-1) ** i) * 0.5) for i in range(10)
        ]
        # Opp rows: opponent always checks (action 0)
        opp_rows = [_opp_row("lm_vs_cfr", "kuhn", i, 0) for i in range(10)]
        df = pl.DataFrame(lm_rows + opp_rows)
        result = score_property2(df, games={"kuhn": game})
        gap = result["br_gap"]["lm_vs_cfr"]["kuhn"]
        # BR is at least as good as any fixed policy.
        assert not math.isnan(gap)

    def test_gap_zero_when_lm_plays_br(self, kuhn_cfr):
        """If the empirical opponent is uniform and the LM earns BR value, gap ≈ 0."""
        game, _ = kuhn_cfr
        # Build empirical opponent: always checks (action 0) - uniform-ish.
        emp = {(1, "0pb"): {0: 1.0}, (1, "1pb"): {0: 1.0}, (1, "2pb"): {0: 1.0}}
        br_val = _compute_br_value(game, emp, seat=0)
        # LM earns exactly br_val per hand.
        lm_rows = [_lm_row("lm_vs_check", "kuhn", i, br_val) for i in range(10)]
        opp_rows = [_opp_row("lm_vs_check", "kuhn", i, 0) for i in range(10)]
        df = pl.DataFrame(lm_rows + opp_rows)
        result = score_property2(df, games={"kuhn": game})
        # Gap should be approximately 0 (br_val - br_val).
        assert abs(result["br_gap"]["lm_vs_check"]["kuhn"]) < 0.01


class TestBRGapTrajectory:
    """P2 is defined as `delta_t -> 0` as evidence accumulates -- a
    trajectory in t. A single scalar per (matchup_id, game) with `lm_mean`
    pooled over every hand has no `delta_t`, so the property could not be
    scored from it even with a memory-equipped agent.

    The estimator is pinned, not per-window: `br_value` is computed once against
    the pooled empirical opponent and reused for every bin. Per-window opponent
    info-set coverage collapses (leduc 135 pooled -> 56-72 per window), and
    `_build_empirical_policy` falls back to uniform off-support, so a
    per-window `br_value` would be a best response to a different hybrid
    target in each window. Only the pinned form answers P2's actual question,
    "does the gap to a fixed target shrink". It is also what the core-path
    invariant implies, the opponent being fixed.
    """

    def test_trajectory_bins_by_hand_index_not_by_row_order(self):
        """Bins must pool seeds at comparable t, not concatenate them.
        """
        rows = []
        for seed, reward in ((0, 1.0), (1, -1.0)):
            for i in range(20):
                r = _lm_row("lm_vs_cfr", "kuhn", i, reward)
                r["seed"] = seed
                rows.append(r)
        result = score_property2(pl.DataFrame(rows), n_hand_bins=4)
        traj = result["lm_mean_trajectory"]["lm_vs_cfr"]["kuhn"]
        assert len(traj) == 4
        # Every bin pools both seeds -> mean 0.0 everywhere, no trend.
        for value in traj:
            assert abs(value) < 1e-9, f"bin means {traj} show a seed-order artefact"

    def test_trajectory_exposes_improvement_the_pooled_scalar_hides(self, kuhn_cfr):
        """A cell whose LM improves must show a shrinking gap across bins.
        """
        game, _ = kuhn_cfr
        # Reward climbs with hand index: -1.0 early, +1.0 late.
        lm_rows = []
        for i in range(40):
            lm_rows.append(_lm_row("lm_vs_cfr", "kuhn", i, -1.0 + 2.0 * (i / 39.0)))
        opp_rows = [_opp_row("lm_vs_cfr", "kuhn", i, 0) for i in range(40)]
        df = pl.DataFrame(lm_rows + opp_rows)
        result = score_property2(df, games={"kuhn": game}, n_hand_bins=4)
        gaps = result["br_gap_trajectory"]["lm_vs_cfr"]["kuhn"]
        assert len(gaps) == 4
        assert all(not math.isnan(g) for g in gaps)
        # delta_t must be strictly decreasing as the LM improves.
        assert gaps[0] > gaps[1] > gaps[2] > gaps[3], gaps

    def test_br_value_is_pinned_so_only_lm_mean_moves_across_bins(self, kuhn_cfr):
        """Each bin's gap must equal the one pooled br_value minus that bin's mean.
        """
        game, _ = kuhn_cfr
        lm_rows = [_lm_row("lm_vs_cfr", "kuhn", i, float((-1) ** i)) for i in range(40)]
        opp_rows = [_opp_row("lm_vs_cfr", "kuhn", i, i % 2) for i in range(40)]
        df = pl.DataFrame(lm_rows + opp_rows)
        result = score_property2(df, games={"kuhn": game}, n_hand_bins=4)
        br_val = result["br_value"]["lm_vs_cfr"]["kuhn"]
        gaps = result["br_gap_trajectory"]["lm_vs_cfr"]["kuhn"]
        means = result["lm_mean_trajectory"]["lm_vs_cfr"]["kuhn"]
        assert not math.isnan(br_val)
        for gap, mean in zip(gaps, means, strict=False):
            assert abs((gap + mean) - br_val) < 1e-9

    def test_degenerate_cell_is_named_not_given_a_flat_trajectory(self):
        """A cell with too few hands to bin must be named, never silently binned.
        """
        rows = [_lm_row("lm_vs_cfr", "kuhn", i, 1.0) for i in range(3)]
        result = score_property2(pl.DataFrame(rows), n_hand_bins=10)
        assert "lm_vs_cfr" not in result.get("br_gap_trajectory", {})
        reason = result["trajectory_not_applicable"]["lm_vs_cfr"]["kuhn"]
        assert "3" in reason and "10" in reason


class TestNoCrossGameMixing:
    """Regression: P2 must be computed per (opponent, game), never mixed across games.

    Looping over the game-agnostic matchup_id and taking matchup_df["game"][0]
    would mix all games' rewards into one number and build the empirical
    opponent policy from mixed-game info-sets against one game's tree. Output
    is {matchup_id: {game: value}}.
    """

    def test_reward_not_mixed_across_games(self):
        # One matchup_id spanning two games with distinct per-game rewards.
        rows = [_lm_row("lm_vs_cfr", "kuhn", i, +1.0) for i in range(10)]
        rows += [_lm_row("lm_vs_cfr", "leduc", i, -3.0) for i in range(10)]
        result = score_property2(pl.DataFrame(rows))  # games=None -> reward path only
        lmr = result["lm_mean_reward"]["lm_vs_cfr"]
        assert isinstance(lmr, dict), "lm_mean_reward must be nested per game"
        assert abs(lmr["kuhn"] - 1.0) < 1e-9  # Not the mixed mean (-1.0)
        assert abs(lmr["leduc"] + 3.0) < 1e-9  # Not the mixed mean (-1.0)

    def test_seed_not_collapsed(self):
        # Same hand_idx across two seeds with opposite rewards: the mean must use
        # Both seeds (10 hands), not collapse by hand_idx and keep one seed.
        rows = [dict(_lm_row("m", "kuhn", i, +1.0), seed=0) for i in range(5)]
        rows += [dict(_lm_row("m", "kuhn", i, -1.0), seed=1) for i in range(5)]
        result = score_property2(pl.DataFrame(rows))
        assert abs(result["lm_mean_reward"]["m"]["kuhn"]) < 1e-9  # mean over 10 hands

    def test_br_uses_per_game_tree(self, kuhn_cfr):
        # One matchup spanning kuhn + a decoy game; only kuhn has a tree supplied.
        game, _ = kuhn_cfr
        rows = [_lm_row("lm_vs_check", "kuhn", i, 0.0) for i in range(6)]
        rows += [_opp_row("lm_vs_check", "kuhn", i, 0) for i in range(6)]
        rows += [_lm_row("lm_vs_check", "leduc", i, 0.0) for i in range(6)]
        rows += [_opp_row("lm_vs_check", "leduc", i, 0) for i in range(6)]
        result = score_property2(pl.DataFrame(rows), games={"kuhn": game})
        # kuhn cell has a real BR value; leduc cell is nan (no tree) - never mixed.
        assert not math.isnan(result["br_gap"]["lm_vs_check"]["kuhn"])
        assert math.isnan(result["br_gap"]["lm_vs_check"]["leduc"])


class TestEmpiricalPolicy:
    def test_normalises_counts(self):
        """Action counts are normalised to probabilities."""
        rows = [
            {
                "player_to_act": 0,
                "info_set": "A",
                "parsed_action": 0,
                "game": "kuhn",
                "matchup_id": "x",
                "hand_idx": 0,
                "reward": 0.0,
                "prompt": None,
            },
            {
                "player_to_act": 0,
                "info_set": "A",
                "parsed_action": 1,
                "game": "kuhn",
                "matchup_id": "x",
                "hand_idx": 1,
                "reward": 0.0,
                "prompt": None,
            },
        ]
        from lmopt.scoring.property2 import _build_empirical_policy

        policy = _build_empirical_policy(pl.DataFrame(rows))
        probs = policy[(0, "A")]
        assert abs(probs[0] - 0.5) < 1e-9
        assert abs(probs[1] - 0.5) < 1e-9


def _kuhn_df_from_policy(
    game, policy, matchup_id: str = "lm_vs_cfr", n_samples: int = 2000
) -> pl.DataFrame:
    """Build a df whose opponent (player 1) rows reproduce `policy`'s
    action-probabilities at every player-1 info-set in `game`, via row
    counts proportional to those probabilities -- `score_property2` only
    ever sees row counts (`_build_empirical_policy` normalises them), never
    real-valued probabilities, so this is how a policy is smuggled through
    the actual Parquet-shaped interface rather than injected directly.

    Also adds one LM row (`_lm_row`) so `score_property2` does not skip the
    (matchup, game) cell via its `if lm_rows.is_empty(): continue` guard
    (property2.py) -- br_value/on_support_fraction are computed from the
    opponent rows only, so its content is irrelevant to what this measures.
    """
    from open_spiel.python.algorithms import get_all_states

    all_states = get_all_states.get_all_states(
        game, include_terminals=False, include_chance_states=False
    )
    p1_infosets: dict[str, dict[int, float]] = {}
    for state in all_states.values():
        if state.current_player() == 1:
            info = state.information_state_string(1)
            p1_infosets[info] = policy.action_probabilities(state)

    rows = [_lm_row(matchup_id, "kuhn", -1, 0.0)]
    hid = 0
    for info, probs in p1_infosets.items():
        for action, p in probs.items():
            for _ in range(round(p * n_samples)):
                rows.append(
                    _opp_row(matchup_id, "kuhn", hid, int(action), info_set=info)
                )
                hid += 1
    return pl.DataFrame(rows, infer_schema_length=None)


class TestImportedBestResponse:
    """Imported CFR, never hand-rolled -- applied to P2's BR walk. Taking
    `max` per state rather than per INFO SET is equivalent to solving with
    the opponent's card revealed. On this class's own fixture (`kuhn_cfr` at
    500 CFR+ iterations, `_kuhn_df_from_policy`'s N=2000 row-count sampling)
    the per-state walk returns +0.3331667 and the imported best response
    -0.0554167. The analytic Nash value is -1/18 = -0.0555556; a different
    configuration (2000 iterations, N=200000 samples, no rounding noise, not
    this fixture) gives -0.0554883 -- cited here only to show the two
    numbers are the same mechanism at different sample sizes, not as this
    test's own tolerance.
    """

    def test_p2_br_value_matches_analytic_kuhn_equilibrium(self, kuhn_cfr):
        """Must run through score_property2 with a df, else it tests OpenSpiel
        directly and says nothing about whether score_property2 actually
        reaches the imported best-response path. Tolerance is nash_conv(game, policy) for
        this policy, recomputed here rather than hardcoded, because a
        converged-but-not-exact CFR+ policy's true best-response value can
        differ from the analytic optimum by up to its own exploitability.
        """
        game, policy = kuhn_cfr
        nc = nash_conv(game, policy)
        df = _kuhn_df_from_policy(game, policy)
        result = score_property2(
            df, games={"kuhn": game}, cfr_policies={"kuhn": policy}
        )
        br_val = result["br_value"]["lm_vs_cfr"]["kuhn"]
        assert abs(br_val - (-1 / 18)) <= nc, (
            f"br_value={br_val} vs analytic -1/18={-1 / 18:.6f}, "
            f"tolerance(nash_conv)={nc:.6f}"
        )

    def test_p2_reports_opponent_infoset_coverage(self, kuhn_cfr):
        """Low coverage (0.2% on liarsdice) is invisible without this field:
        `_EmpiricalPolicy` fills every unseen info-set with uniform, so
        br_value silently becomes a best response to a hybrid
        (mostly-uniform) policy with nothing in the output to say so. Pinned here on kuhn -- liarsdice's tree is far
        larger, but this is the identical mechanism at a size a unit test can
        afford: full coverage (all 6 player-1 kuhn info-sets logged) must
        read differently from partial coverage (2 of 6), and both must be
        genuine fractions, not a constant.
        """
        game, policy = kuhn_cfr
        full_df = _kuhn_df_from_policy(game, policy)
        full_result = score_property2(
            full_df, games={"kuhn": game}, cfr_policies={"kuhn": policy}
        )
        full_cov = full_result["on_support_fraction"]["lm_vs_cfr"]["kuhn"]
        assert 0.0 <= full_cov <= 1.0
        assert full_cov > 0.99, f"all 6 player-1 kuhn info-sets were logged: {full_cov}"

        partial_rows = [
            _lm_row("lm_vs_partial", "kuhn", -1, 0.0),
            _opp_row("lm_vs_partial", "kuhn", 0, 0, info_set="0p"),
            _opp_row("lm_vs_partial", "kuhn", 1, 0, info_set="1p"),
        ]
        partial_df = pl.DataFrame(partial_rows)
        partial_result = score_property2(
            partial_df, games={"kuhn": game}, cfr_policies={"kuhn": policy}
        )
        partial_cov = partial_result["on_support_fraction"]["lm_vs_partial"]["kuhn"]
        assert 0.0 <= partial_cov <= 1.0
        assert partial_cov < full_cov, (
            f"partial ({partial_cov}) must read lower than full ({full_cov})"
        )


class TestBRErrorSurfaced:
    """A bare `except Exception: br_val = nan; on_support = nan` in
    `score_property2` would discard the exception with no counter and no
    reason string -- the forbidden silent-fallback shape, and the defect
    class that once let an uncaught SpielError hit 75.9% of rows while 31
    unit tests stayed green. `score_property2` must still catch the
    exception (one matchup's bad data must not crash the whole run) but must
    record why in `br_error`, per (matchup_id, game) cell.
    """

    def test_raising_br_walk_is_recorded_not_silently_nanned(
        self, kuhn_cfr, monkeypatch
    ):
        """Must run through score_property2, not by calling
        `_compute_br_value_and_coverage` directly, so this pins the catching
        code's own behaviour, not just the function it wraps.
        """
        game, policy = kuhn_cfr
        import lmopt.scoring.property2 as property2_mod

        def _raise(*a, **kw):
            raise RuntimeError("synthetic BR failure")

        monkeypatch.setattr(property2_mod, "_compute_br_value_and_coverage", _raise)

        rows = [
            _lm_row("m_err", "kuhn", 0, 0.0),
            _opp_row("m_err", "kuhn", 0, 0, info_set="0p"),
        ]
        df = pl.DataFrame(rows)
        result = score_property2(
            df, games={"kuhn": game}, cfr_policies={"kuhn": policy}
        )

        assert math.isnan(result["br_value"]["m_err"]["kuhn"])
        assert math.isnan(result["on_support_fraction"]["m_err"]["kuhn"])
        err = result["br_error"]["m_err"]["kuhn"]
        assert err, "a raised BR walk must leave a non-empty reason, not a silent nan"
        assert "synthetic BR failure" in err


class TestPartialReferenceGuard:
    """Without a partial-reference guard in score_property2,
    `_compute_br_value` - an unbounded recursive full-tree walk over
    `games[game_name]` that never consults a reference - runs
    unconditionally whenever any opponent (non-LM) rows exist. It does
    not raise, so it hangs forever rather than being caught by a broad
    `except Exception`. Mirrors property7.py/property8.py's
    `is_partial_reference` guard tests exactly, but property2 additionally
    needs the "would actually hang" proof: P7b/P8's own tests never showed the
    hazard is real, only that the guard fires.
    """


    def test_p2_full_reference_is_not_misidentified_as_partial(self, kuhn_cfr):
        """Pin the other direction of the guard: a full CFR+ TabularPolicy has
        neither `lookup` nor `n_infosets` (only `state_lookup`), so P2 must
        not silently disable itself for the five canonical games."""
        game, cfr_pol = kuhn_cfr
        rows = [_lm_row("m1", "kuhn", i, 0.0) for i in range(5)]
        rows += [_opp_row("m1", "kuhn", i, 0) for i in range(5)]
        df = pl.DataFrame(rows)
        result = score_property2(
            df, games={"kuhn": game}, cfr_policies={"kuhn": cfr_pol}
        )
        assert "kuhn" not in result.get("not_applicable", {})
        assert not math.isnan(result["br_gap"]["m1"]["kuhn"])

    @pytest.mark.slow  # 20s call -- a deliberate
    # subprocess.TimeoutExpired wait, not amortisable. Takes no fixture (only
    # `self`), so it does not share `kuhn_cfr`
    # with the other tests in this class; marking it alone is isolated.
    def test_p2_compute_br_value_hangs_on_real_darkhex_3x3_without_a_guard(self):
        """THE hazard proof: `_compute_br_value` called directly - i.e. exactly
        what an unguarded `score_property2` would do -
        against the real darkhex_3x3 game tree does not return within a
        generous bound.
        `timeout 20 python -c "..._compute_br_value(load_darkhex_3x3(), {}, seat=0)"`
        exits 124 (still running when `timeout` killed it), with a
        `faulthandler.dump_traceback_later(90)` trace sitting in property2.py's
        `ev_from`. Run in a SUBPROCESS with its own timeout
        so this proof of the hazard cannot itself hang the test suite.
        """
        script = (
            "from lmopt.games.loaders import load_darkhex_3x3\n"
            "from lmopt.scoring.property2 import _compute_br_value\n"
            "_compute_br_value(load_darkhex_3x3(), {}, seat=0)\n"
            "print('COMPLETED')\n"
        )
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                [sys.executable, "-c", script],
                timeout=20,
                capture_output=True,
                check=True,
            )



# --- Seat- and actor-aware P2 -----------------------------------------------
#
# Kuhn's information-state strings, from walking `load_kuhn()`: player 0 sees {"0","1","2","0pb","1pb","2pb"} and player 1
# sees {"0p","0b","1p","1b","2p","2b"}. The `_opp_row` default
# ("1pb") is a player-0 string, which is why the tests above run
# at `on_support_fraction == 0.0`; the fixtures below use real strings for the
# seat they claim, so coverage is 1.0 and the seat assertions bite.
_KUHN_P0_INFO = ["0", "1", "2", "0pb", "1pb", "2pb"]
_KUHN_P1_INFO = ["0p", "1p", "2p", "0b", "1b", "2b"]
# The deterministic seat-0 environment used below: pass on a low card, bet on
# the high one. Value of a best response to it, via
# `_compute_br_value_and_coverage`: as seat 1 it is 1/3 at coverage 1.0; as
# seat 0 it is 0.5 at coverage 0.0 (a seat-0 walk queries player-1 info-sets,
# which this table does not contain at all).
_SEAT0_ENV_ACTION = {"0": 0, "1": 0, "2": 1, "0pb": 0, "1pb": 0, "2pb": 1}
# 1/3, the best-response value three of the tests below happen to share -- two
# at seat 1 against `_SEAT0_ENV_ACTION` and one at seat 0 against a different
# (player-1) environment. It is a shared numeric literal, not a seat-1
# quantity (the legacy-frame test asserts it beside `br_seat == {"0": 6}`),
# and the seat each use belongs to is stated at the use.
_BR_ONE_THIRD = 0.3333333333333333
# Six per-hand rewards drawn from Kuhn's actual payoff set {-2,-1,1,2}. These
# fixtures assert nothing about reward; varied rather than constant so the
# fixture reads as a realistic hand series.
_HAND_REWARDS = [1.0, -1.0, 2.0, -2.0, 1.0, -1.0]


def _row(
    matchup_id,
    game,
    hand_idx,
    *,
    player_to_act,
    info_set,
    action,
    reward=0.0,
    prompt=None,
    actor=None,
    lm_seat=None,
):
    row = {
        "matchup_id": matchup_id,
        "game": game,
        "hand_idx": hand_idx,
        "reward": reward,
        "prompt": prompt,
        "player_to_act": player_to_act,
        "info_set": info_set,
        "parsed_action": action,
    }
    if actor is not None:
        row["actor"] = actor
    if lm_seat is not None:
        row["lm_seat"] = lm_seat
    return row


class TestSeatAwareBestResponse:
    """The trap: moving P2's row filter without moving its BR seat yields
    opponent reward minus a seat-0 best-response value: a mismatched number
    that looks plausible.
    """

    def test_an_agent_that_sat_seat_one_is_best_responded_as_seat_one(self, kuhn_cfr):
        """P-seat-consistency. The agent under test occupies seat 1 in every
        hand, so its best-response value must be the seat-1 one against the
        logged seat-0 environment.
        """
        game, _ = kuhn_cfr
        rows = [
            _row(
                "m",
                "kuhn",
                h,
                player_to_act=1,
                info_set=_KUHN_P1_INFO[h],
                action=0,
                reward=_HAND_REWARDS[h],
                prompt="p",
                lm_seat=1,
            )
            for h in range(6)
        ] + [
            _row(
                "m",
                "kuhn",
                h,
                player_to_act=0,
                info_set=info,
                action=_SEAT0_ENV_ACTION[info],
            )
            for h, info in enumerate(_KUHN_P0_INFO)
        ]
        result = score_property2(pl.DataFrame(rows), games={"kuhn": game})
        assert result["br_value"]["m"]["kuhn"] == pytest.approx(
            _BR_ONE_THIRD, abs=1e-12
        )
        assert result["on_support_fraction"]["m"]["kuhn"] == 1.0
        assert result["br_seat"]["m"]["kuhn"] == {"1": 6}

    def test_the_environment_is_the_other_actor_not_the_null_prompt_rows(
        self, kuhn_cfr
    ):
        """Actor-awareness on the environment side. Scoring the bayesian opponent, the
        environment is the LM (the other actor), not "every row with a null
        prompt" -- which on this frame is the bayesian's own rows.
        """
        game, _ = kuhn_cfr
        rows = [
            _row(
                "m",
                "kuhn",
                h,
                player_to_act=1,
                info_set=_KUHN_P1_INFO[h],
                action=0,
                reward=_HAND_REWARDS[h],
                actor="bayesian",
                lm_seat=0,
            )
            for h in range(6)
        ] + [
            _row(
                "m",
                "kuhn",
                h,
                player_to_act=0,
                info_set=info,
                action=_SEAT0_ENV_ACTION[info],
                prompt="p",
                actor="model-x",
                lm_seat=0,
            )
            for h, info in enumerate(_KUHN_P0_INFO)
        ]
        result = score_property2(
            pl.DataFrame(rows), games={"kuhn": game}, actor="bayesian"
        )
        assert result["actor_filter"] == "actor=bayesian"
        assert result["br_value"]["m"]["kuhn"] == pytest.approx(
            _BR_ONE_THIRD, abs=1e-12
        )
        assert result["on_support_fraction"]["m"]["kuhn"] == 1.0
        assert result["br_seat"]["m"]["kuhn"] == {"1": 6}

    def test_a_mixed_seat_cell_is_weighted_over_both_seats(self, kuhn_cfr):
        """A `seat_mode="random"` cell has the agent under test at both
        seats, so one scalar per cell can only be the hand-weighted mean of
        the per-seat gaps. Checked against `_compute_br_value_and_coverage`
        directly, which is the seat-general primitive the scorer threads
        rather than rewrites.

        The split is deliberately asymmetric, 3 hands at seat 0 against 1 at
        seat 1. At 2-vs-2 the hand weights are both 0.5 and an unweighted
        mean over seats returns the identical number, so the weighting this
        test exists to pin would not be pinned at all.
        """
        game, _ = kuhn_cfr
        env = [
            _row(
                "m",
                "kuhn",
                100 + h,
                player_to_act=0,
                info_set=info,
                action=_SEAT0_ENV_ACTION[info],
                actor="opp",
            )
            for h, info in enumerate(_KUHN_P0_INFO)
        ] + [
            _row(
                "m",
                "kuhn",
                200 + h,
                player_to_act=1,
                info_set=info,
                action=0,
                actor="opp",
            )
            for h, info in enumerate(_KUHN_P1_INFO)
        ]
        at_seat_0 = [
            _row(
                "m",
                "kuhn",
                h,
                player_to_act=0,
                info_set="0",
                action=0,
                reward=_HAND_REWARDS[h],
                prompt="p",
                actor="model-x",
                lm_seat=0,
            )
            for h in range(3)
        ]
        at_seat_1 = [
            _row(
                "m",
                "kuhn",
                10,
                player_to_act=1,
                info_set="0p",
                action=0,
                reward=_HAND_REWARDS[3],
                prompt="p",
                actor="model-x",
                lm_seat=1,
            )
        ]
        df = pl.DataFrame(env + at_seat_0 + at_seat_1)
        result = score_property2(df, games={"kuhn": game}, actor="model-x")

        emp = _build_empirical_policy(df.filter(pl.col("actor") == "opp"))
        v0, c0 = _compute_br_value_and_coverage(game, emp, seat=0)
        v1, c1 = _compute_br_value_and_coverage(game, emp, seat=1)
        assert v0 != pytest.approx(v1, abs=1e-9), (
            "fixture must have genuinely different per-seat BR values or the "
            "weighting assertion below cannot fail"
        )
        assert result["br_seat"]["m"]["kuhn"] == {"0": 3, "1": 1}
        assert result["br_value"]["m"]["kuhn"] == pytest.approx(
            0.75 * v0 + 0.25 * v1, abs=1e-12
        )
        assert result["br_value"]["m"]["kuhn"] != pytest.approx(
            0.5 * v0 + 0.5 * v1, abs=1e-9
        ), "an unweighted mean over seats must NOT reproduce this number"
        assert result["on_support_fraction"]["m"]["kuhn"] == pytest.approx(
            0.75 * c0 + 0.25 * c1, abs=1e-12
        )

    def test_legacy_frame_scores_byte_identically_and_says_so(self, kuhn_cfr):
        """P-legacy for P2: an arm collected before the `actor`/`lm_seat`
        columns existed has the LM at seat 0 on 100 % of rows (1,259,089 LM
        rows, `player_to_act` distribution `{0: 1259089}`), so the seat
        derived from the scored rows is 0 and every number is unchanged by
        seat-awareness. The pinned 1/3 is the seat-unaware scorer's value.
        """
        game, _ = kuhn_cfr
        rows = [
            _row(
                "m",
                "kuhn",
                h,
                player_to_act=0,
                info_set="0",
                action=0,
                reward=_HAND_REWARDS[h],
                prompt="p",
            )
            for h in range(6)
        ] + [
            _row("m", "kuhn", h, player_to_act=1, info_set=info, action=a)
            for h, (info, a) in enumerate(zip(_KUHN_P1_INFO, [0, 0, 0, 1, 1, 1], strict=False))
        ]
        df = pl.DataFrame(rows)
        assert "actor" not in df.columns and "lm_seat" not in df.columns
        result = score_property2(df, games={"kuhn": game})
        assert result["actor_filter"] == "legacy-prompt-null"
        assert result["br_value"]["m"]["kuhn"] == _BR_ONE_THIRD
        assert result["on_support_fraction"]["m"]["kuhn"] == 1.0
        assert result["br_seat"]["m"]["kuhn"] == {"0": 6}


class TestForeignProbeType:
    """Without a probe_type guard in the plain (non-paired) scorers, a frame
    assembled from a directory that also holds a probe arm's rows (reachable
    via an unfiltered `--data` glob) is silently pooled
    into the BR-gap/reward estimate.
    """

    def _base_frame(self):
        return pl.DataFrame(
            [
                _lm_row("lm_vs_check", "kuhn", 0, 1.0),
                _opp_row("lm_vs_check", "kuhn", 0, 0),
            ]
        )

    def test_refuses_a_frame_holding_a_foreign_probe_type(self):
        base = self._base_frame()
        mixed = pl.concat(
            [
                base.with_columns(pl.lit("none").alias("probe_type")),
                base.with_columns(pl.lit("relabel").alias("probe_type")),
            ]
        )
        with pytest.raises(ValueError, match="relabel"):
            score_property2(mixed)

    def test_accepts_an_all_base_frame(self):
        df = self._base_frame().with_columns(pl.lit("none").alias("probe_type"))
        score_property2(df)  # must not raise


# ---------------------------------------------------------------------------
# P2 k-sample integration
# ---------------------------------------------------------------------------


def test_a_frame_with_seed_but_no_hand_idx_is_refused():
    """
    a frame that carries ``seed`` but no ``hand_idx`` is refused, rather than grouped by seed
    alone so that every seed collapses into one hand
    """
    rows = [
        dict(_lm_row("m", "kuhn", i, 1.0), seed=s) for s in (0, 1) for i in range(3)
    ]
    df = pl.DataFrame(rows).drop("hand_idx")
    with pytest.raises(ValueError, match="hand_idx"):
        score_property2(df)
