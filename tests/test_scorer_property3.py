"""Tests for property3 scorer (Randomisation - regret vs. CFR equilibrium value).

`score_property3` scores the regret of the action taken relative to the best
action at that info-set, under OpenSpiel's own `BestResponsePolicy.q_value`
against a solved CFR reference (not KL(LM action distribution || CFR Nash)).
Most of this file's fixtures use Kuhn's own real info-sets and a real solved
CFR+ policy (`kuhn_cfr` below) rather than synthetic info-sets or per-row
`cfr_policy` JSON -- `score_property3` ignores the `cfr_policy` column
entirely and loads/solves its own whole-game reference internally.
"""

from __future__ import annotations

import json
import math

import polars as pl
import pytest

pyspiel = pytest.importorskip("pyspiel")

from lmopt.games import policy_cache
from lmopt.games.cfr import nash_conv, solve_cfr
from lmopt.games.loaders import load_kuhn
from lmopt.scoring._tree import build_infoset_map
from lmopt.scoring.property3 import (
    KLScore,
    _compute_regret_table,
    score_property3,
    score_property3_exact,
)


@pytest.fixture(scope="module")
def kuhn_cfr():
    """5000 CFR+ iterations.

    Tighter than `test_scorer_property2.py`'s own 500-iteration `kuhn_cfr`
    fixture: that module's checks are against a single pooled best-response
    value at nash_conv-scaled tolerance (a coarse statistic); this module's
    `test_p3_regret_is_zero_for_every_equilibrium_support_action` checks
    per-info-set, per-action indifference directly, which needs tighter
    convergence to clear the same style of tolerance.

    At 500 iterations (nash_conv 3.45e-4), player
    1's "1b" info-set's two equilibrium-support actions' pooled Q values
    differ by ~0.0059 -- outside a `max(nash_conv, 1e-3)` tolerance. At 5000
    iterations (nash_conv 5.5e-5, solve time ~3.2s), every one of Kuhn's 12
    info-sets (both seats) clears `abs(diff) <= 1e-3` with a support
    threshold of 0.01 -- see `test_p3_regret_is_zero_for_every_equilibrium_
    support_action`'s own tolerance choice.
    """
    game = load_kuhn()
    cfr_pol = solve_cfr(game, 5000)
    return game, cfr_pol


@pytest.fixture()
def scoped_cfr_cache(tmp_path, monkeypatch):
    """Redirect `score_property3`'s internal CFR cache to a throwaway dir.

    `score_property3(df, actor=None)` has no `cache_dir` parameter -- its
    call signature is pinned across every existing consumer, so it cannot
    take one. `score_property3` reads
    `policy_cache.DEFAULT_CFR_DIR` as a module attribute at call time (a
    local `from lmopt.games import policy_cache` inside the function body,
    not a `from ... import DEFAULT_CFR_DIR` bound at import time), so
    monkeypatching the attribute is the only lever available -- the same
    "never write into the real project data/cfr_policies/ from a test"
    discipline `tests/test_property6.py`'s own `tmp_path`-cache-dir fixtures
    already establish for `load_or_solve_cfr`/`offpath_reference`, applied
    here via monkeypatch since the parameter itself isn't exposed.
    """
    monkeypatch.setattr(policy_cache, "DEFAULT_CFR_DIR", str(tmp_path))
    return tmp_path


def _queen_open_info(game):
    """Player 0's real Kuhn opening info-set holding the Queen, plus its
    Pass/Bet action ids from pyspiel's own `action_to_string` (never
    hardcoded/guessed by number).

    Kuhn's textbook equilibrium (Kuhn 1950): the JACK (weakest card) is the
    one that bluffs -- bets with probability alpha in [0, 1/3] -- while the
    QUEEN (middle card) never bets at the opening; betting Queen is a
    strictly dominated pure deviation. Against this module's own `kuhn_cfr`
    fixture (5000 CFR+ iterations): pooled Q at this info-set is
    Pass=-0.33323, Bet=-0.49999, a real ~1/6 gap, not solver noise (CFR+'s
    own residual floor on Bet's probability here is ~1e-6, the solver's
    residual-probability floor -- irrelevant to this value-based gap, which
    is why regret is used rather than a probability/KL-based statistic).
    """
    s = game.new_initial_state()
    s.apply_action(1)  # Queen dealt to player 0.
    s.apply_action(0)  # (Arbitrary) Jack dealt to player 1 -- the info-set
    # string and legal action ids are properties of the GAME/player-0's own
    # hand, not of the opponent's, so any other opponent card gives the same
    # info_str/action ids here.
    info_str = s.information_state_string(0)
    pass_action = next(
        a for a in s.legal_actions() if s.action_to_string(0, a) == "Pass"
    )
    bet_action = next(a for a in s.legal_actions() if s.action_to_string(0, a) == "Bet")
    return info_str, pass_action, bet_action


def _p3_fixture(
    game, action, on_support_rows: int = 3, off_support_rows: int = 0
) -> pl.DataFrame:
    """Rows at Kuhn's real Queen-opening info-set (see `_queen_open_info`),
    split between on-/off-support, against a real info-set/action -- the
    estimand ignores the per-row `cfr_policy` JSON column entirely, so a
    synthetic info-set does not produce a meaningful score.
    """
    info_str, pass_a, bet_a = _queen_open_info(game)
    act = pass_a if action == "pass" else bet_a
    rows = []
    for _ in range(on_support_rows):
        rows.append(
            {
                "game": "kuhn",
                "info_set": info_str,
                "parsed_action": act,
                "prompt": "some prompt",
                "cfr_ref_off_support": False,
            }
        )
    for _ in range(off_support_rows):
        rows.append(
            {
                "game": "kuhn",
                "info_set": info_str,
                "parsed_action": act,
                "prompt": "some prompt",
                "cfr_ref_off_support": True,
            }
        )
    return pl.DataFrame(rows)


class TestComputeRegretTable:
    """Direct tests of `_compute_regret_table`, the regret primitive -- no `score_property3`/CFR-cache involved, since a
    pre-solved `cfr_policy` is passed straight in.
    """

    def test_p3_regret_is_zero_for_every_equilibrium_support_action(self, kuhn_cfr):
        game, policy = kuhn_cfr
        nc = nash_conv(game, policy)
        tol = max(nc, 1e-3)
        # A support threshold well above CFR+'s own residual-probability
        # floor (~2.28e-8): a genuinely-zero-support
        # action never reads as exactly 0.0 under finite CFR+ iterations, so
        # a threshold this far above the floor is needed to select only the
        # Real equilibrium support, not solver residue.
        support_threshold = 0.01
        info_map = build_infoset_map(game)
        for lm_seat in (0, 1):
            table = _compute_regret_table(game, policy, lm_seat)
            for (player, info_str), state in info_map.items():
                if player != lm_seat:
                    continue
                max_q, q_values = table[info_str]
                probs = policy.action_probabilities(state)
                for action, prob in probs.items():
                    if prob > support_threshold:
                        assert q_values[action] == pytest.approx(max_q, abs=tol), (
                            f"seat={lm_seat} info_set={info_str!r} action={action} "
                            f"prob={prob}"
                        )

    def test_p3_regret_is_strictly_negative_for_a_known_suboptimal_action(
        self, kuhn_cfr
    ):
        """Kuhn's own known equilibrium: player 0 holding the QUEEN (the
        middle card, not the Jack) at the opening node should never bet
        under CFR+'s solved strategy -- betting Queen is a strictly
        dominated pure deviation (the Jack is the card that mixes/bluffs in
        equilibrium; see `_queen_open_info`'s docstring). Concretized
        against this fixture's own solved policy via pyspiel's own
        `action_to_string`, not hardcoded by guessing which action id is
        "Bet".
        """
        game, policy = kuhn_cfr
        info_str, pass_action, bet_action = _queen_open_info(game)
        table = _compute_regret_table(game, policy, lm_seat=0)
        max_q, q_values = table[info_str]
        assert q_values[pass_action] == pytest.approx(max_q, abs=1e-3)
        regret = q_values[bet_action] - max_q
        assert regret < -0.1, f"expected a real, structural regret; got {regret}"


def test_mixed_seat_cell_uses_player_to_act_not_lm_seat(kuhn_cfr, scoped_cfr_cache):
    """Regression guard for scoring by `player_to_act`, not `lm_seat` (per
    `logging/schema.py`: `lm_seat` names AGENT0's seat, not the scored
    actor's). Every other fixture in this file either omits the
    `player_to_act`/`lm_seat` columns entirely or sets them consistently, so
    switching `score_property3`'s seat derivation to
    `game_df["lm_seat"][0]` would not redden any of them. This fixture has
    two real seats and a `lm_seat` column that disagrees with `player_to_act`
    on the seat-1 row (a real production case: under `seat_mode="random"`,
    `lm_seat` -- AGENT0's seat -- need not equal the scored actor's seat on
    every hand).
    """
    game, _ = kuhn_cfr
    queen_info, queen_pass, _ = _queen_open_info(game)
    # Real player-1 info-set: King dealt to player 1, facing a check from
    # player 0 (Queen) -- action ids from pyspiel's own `action_to_string`,
    # never guessed.
    s = game.new_initial_state()
    s.apply_action(1)  # Queen to player 0
    s.apply_action(2)  # King to player 1
    pass0 = next(a for a in s.legal_actions() if s.action_to_string(0, a) == "Pass")
    s.apply_action(pass0)
    king_facing_check_info = s.information_state_string(1)
    bet1 = next(a for a in s.legal_actions() if s.action_to_string(1, a) == "Bet")

    rows = pl.DataFrame(
        [
            # Seat 0, Queen's opening, Pass (equilibrium-optimal, regret ~0).
            # lm_seat correctly agrees with player_to_act here.
            {
                "game": "kuhn",
                "info_set": queen_info,
                "parsed_action": queen_pass,
                "prompt": "p",
                "player_to_act": 0,
                "lm_seat": 0,
            },
            # Seat 1, King facing a check, Bet (equilibrium-optimal, regret ~0).
            # lm_seat is deliberately wrong (0, not 1): the row this test exists
            # to catch a regression on.
            {
                "game": "kuhn",
                "info_set": king_facing_check_info,
                "parsed_action": bet1,
                "prompt": "p",
                "player_to_act": 1,
                "lm_seat": 0,
            },
        ]
    )
    out = score_property3(rows)
    assert out["per_info_set"]["kuhn"].get(queen_info) == pytest.approx(0.0, abs=1e-3)
    assert out["per_info_set"]["kuhn"].get(king_facing_check_info) == pytest.approx(
        0.0, abs=1e-3
    ), (
        "the seat-1 row must be scored via player_to_act, not silently "
        "dropped by a lm_seat-keyed lookup"
    )
    assert out["mean_kl"]["kuhn"].on_support_fraction == 1.0


def _kuhn_lm_df_from_policy(game, policy, n_samples: int = 200) -> pl.DataFrame:
    """Build a df whose LM (player 0) rows reproduce `policy`'s
    action-probabilities at every player-0 info-set, via row counts
    proportional to those probabilities -- mirrors
    `test_scorer_property2.py`'s own `_kuhn_df_from_policy` pattern
    (opponent rows there, LM rows here), so `score_property3` only ever
    sees row counts, never real-valued probabilities.
    """
    from open_spiel.python.algorithms import get_all_states

    all_states = get_all_states.get_all_states(
        game, include_terminals=False, include_chance_states=False
    )
    p0_infosets: dict[str, dict[int, float]] = {}
    for state in all_states.values():
        if state.current_player() == 0:
            info = state.information_state_string(0)
            p0_infosets[info] = policy.action_probabilities(state)

    rows = []
    for info, probs in p0_infosets.items():
        for action, p in probs.items():
            for _ in range(round(p * n_samples)):
                rows.append(
                    {
                        "game": "kuhn",
                        "info_set": info,
                        "parsed_action": int(action),
                        "prompt": "some prompt",
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def test_score_property3_returns_regret_not_kl_shape(kuhn_cfr, scoped_cfr_cache):
    """    `regret <= 1e-9` alone is tautological given `_compute_regret_table`'s
    own definition (`q_values[action] - max(...)` can never exceed 0
    regardless of whether the underlying Q-values are computed correctly,
    e.g. even without pooling) -- it does not actually discriminate "regret"
    from "some other max-relative quantity", only from a formula that could
    go positive (KL). `_kuhn_lm_df_from_policy`'s pure policy-sampled
    rows also mostly land on support (near-zero regret by construction), so
    without an explicit off-policy row this test could pass vacuously on a
    fixture that never exercises a real negative value at all. Mixed in
    below: extra rows at Kuhn's known-suboptimal Queen-Bet action
    (`_queen_open_info`, regret ~-1/6 -- see
    `TestComputeRegretTable`'s own known-suboptimal-action test), so this
    test also asserts a genuinely negative value exists, not just an
    absence of positive ones.
    """
    game, policy = kuhn_cfr
    df = _kuhn_lm_df_from_policy(game, policy)
    queen_info, _pass_action, bet_action = _queen_open_info(game)
    # `_kuhn_lm_df_from_policy` already samples ~200 on-policy (near-zero
    # regret) rows at this same info-set (Pass, at the solved policy's own
    # ~1.0 probability there), and `per_info_set` aggregates by info_str --
    # so a small number of off-policy rows would be diluted into the mean
    # rather than producing a clearly negative value. 5000 rows dominates
    # that mix (weighted mean ~= 5000/5200 * (-1/6) =~ -0.16), comfortably
    # past the `< -0.1` threshold below.
    off_policy_rows = pl.DataFrame(
        [
            {
                "game": "kuhn",
                "info_set": queen_info,
                "parsed_action": bet_action,
                "prompt": "some prompt",
            }
            for _ in range(5000)
        ]
    )
    df = pl.concat([df, off_policy_rows], how="diagonal_relaxed")
    result = score_property3(df, actor=None)
    assert result["per_info_set"]["kuhn"], "fixture produced no scored info-sets"
    regrets = [
        regret
        for game_regrets in result["per_info_set"].values()
        for regret in game_regrets.values()
    ]
    for regret in regrets:
        assert regret <= 1e-9
    assert any(r < -0.1 for r in regrets), (
        "fixture must exercise a genuinely negative regret, not just avoid "
        f"positive ones (all regrets: {regrets})"
    )


def test_mean_kl_is_klscore_carrying_coverage(kuhn_cfr, scoped_cfr_cache):
    game, _ = kuhn_cfr
    df = _p3_fixture(game, "pass", off_support_rows=0)
    out = score_property3(df)
    score = out["mean_kl"]["kuhn"]
    assert isinstance(score, KLScore)
    assert score.on_support_fraction == 1.0
    # Every row plays Pass (the equilibrium-optimal action at Queen's
    # opening): mean regret must be (approximately) 0 -- regret is <= 0
    # always under this estimand.
    assert score.value == pytest.approx(0.0, abs=1e-6)


def test_off_support_rows_are_excluded_from_kl_but_counted_in_coverage(
    kuhn_cfr, scoped_cfr_cache
):
    """The survivorship-bias guard: off-support rows must not silently vanish."""
    game, _ = kuhn_cfr
    df = _p3_fixture(game, "pass", off_support_rows=1, on_support_rows=3)
    score = score_property3(df)["mean_kl"]["kuhn"]
    assert score.on_support_fraction == pytest.approx(0.75)


def test_klscore_raises_on_arithmetic():
    """Unmigrated readers must fail, not silently degrade."""
    s = KLScore(1.0, 0.5)
    with pytest.raises(TypeError):
        _ = s / 6.0
    with pytest.raises(TypeError):
        _ = min(s, 1.0)


def test_global_mean_kl_excludes_partially_covered_games_explicitly(
    kuhn_cfr, scoped_cfr_cache
):
    game, _ = kuhn_cfr
    out = score_property3(
        _p3_fixture(game, "pass", off_support_rows=1, on_support_rows=3)
    )
    assert "global_mean_kl_partial_games" in out
    assert "kuhn" in out["global_mean_kl_partial_games"]
    # The actual exclusion, not just the naming: with only one (partially
    # covered) game present, global_mean_kl must be NaN -- `fully_covered`
    # (the dict fed to the average) is computed independently of
    # `partial_games`, so a regression that replaces it with an "any
    # non-nan value" check would leave the two asserts above green while
    # silently including a partially-covered game's regret in the global
    # average.
    assert math.isnan(out["global_mean_kl"])


# score_property3_exact -- the exact-distribution KL diagnostic, reported
# Beside the primary regret metric (property3.py's own docstring). The tests
# cover the LM-row filter, off-support exclusion and KLScore, with the
# matched/unmatched bookkeeping specific to this scorer.


def _lm_row_for_exact(
    game: str,
    info_set: str,
    cfr_policy: dict[int, float],
    cfr_ref_off_support: bool = False,
) -> dict:
    return {
        "game": game,
        "info_set": info_set,
        "cfr_policy": json.dumps({str(k): v for k, v in cfr_policy.items()}),
        "prompt": "some prompt",
        "cfr_ref_off_support": cfr_ref_off_support,
    }


class TestScoreProperty3Exact:
    def test_zero_kl_when_exact_matches_cfr(self):
        """Zero KL, using the exact distribution directly (no Laplace smoothing) instead of empirical
        counts. This is also the one fixture in this class where every
        visited info-set is matched (matched_fraction == 1.0, n_unmatched
        == 0) -- `test_unmatched_info_set_...` below covers the mixed
        case and `test_no_matches_gives_...` covers the zero-matched case,
        so this is the third, distinct point on that spectrum.
        """
        cfr = {0: 0.5, 1: 0.5}
        rows = [_lm_row_for_exact("kuhn", "infoA", cfr) for _ in range(3)]
        df = pl.DataFrame(rows)
        exact_policy = {("kuhn", "infoA"): {0: 0.5, 1: 0.5}}
        out = score_property3_exact(df, exact_policy)
        assert out["per_info_set"]["kuhn"]["infoA"] == pytest.approx(0.0, abs=1e-9)
        assert out["n_matched"]["kuhn"] == 1
        assert out["n_unmatched"]["kuhn"] == 0
        assert out["matched_fraction"]["kuhn"] == pytest.approx(1.0)

    def test_nonzero_kl_when_exact_disagrees_with_cfr(self):
        """Nonzero KL when the exact distribution disagrees with CFR.
        """
        cfr = {0: 0.5, 1: 0.5}
        rows = [_lm_row_for_exact("kuhn", "infoA", cfr)]
        df = pl.DataFrame(rows)
        exact_policy = {("kuhn", "infoA"): {0: 0.9, 1: 0.1}}
        out = score_property3_exact(df, exact_policy)
        assert out["per_info_set"]["kuhn"]["infoA"] > 0.0

    def test_unmatched_info_set_skipped_and_counted_not_silently_dropped(self):
        """The signature requirement: an info-set with no `exact_policy` entry must be SKIPPED (absent from
        "per_info_set") and counted (present in "n_unmatched"), never
        silently vanish from both.
        """
        cfr = {0: 0.5, 1: 0.5}
        rows = [
            _lm_row_for_exact("kuhn", "infoA", cfr),
            _lm_row_for_exact("kuhn", "infoB", cfr),
        ]
        df = pl.DataFrame(rows)
        exact_policy = {("kuhn", "infoA"): {0: 0.5, 1: 0.5}}  # infoB unmatched
        out = score_property3_exact(df, exact_policy)
        assert "infoA" in out["per_info_set"]["kuhn"]
        assert "infoB" not in out["per_info_set"]["kuhn"]
        assert out["n_matched"]["kuhn"] == 1
        assert out["n_unmatched"]["kuhn"] == 1
        assert out["matched_fraction"]["kuhn"] == pytest.approx(0.5)

    def test_no_matches_gives_matched_fraction_zero_not_nan(self):
        """An empty `exact_policy` dict is a legitimate ("nothing matched")
        answer, not a "no data" one -- `matched_fraction` must read 0.0,
        not NaN, when the population is non-empty but nothing joined.
        """
        cfr = {0: 0.5, 1: 0.5}
        rows = [_lm_row_for_exact("kuhn", "infoA", cfr)]
        df = pl.DataFrame(rows)
        out = score_property3_exact(df, {})
        assert out["matched_fraction"]["kuhn"] == 0.0
        assert out["n_matched"]["kuhn"] == 0
        assert out["n_unmatched"]["kuhn"] == 1
        assert math.isnan(out["mean_kl"]["kuhn"].value)

    def test_kl_goes_infinite_when_cfr_assigns_zero_to_a_positive_exact_action(self):
        """CFR assigns exactly 0.0 to an action the exact distribution puts
        positive mass on -- `kl` must go to +inf, exactly mirroring
        score_property3's own `pc <= 0.0 -> kl = inf` branch, never silently
        clipped or skipped.
        """
        cfr = {0: 1.0, 1: 0.0}
        rows = [_lm_row_for_exact("kuhn", "infoA", cfr)]
        df = pl.DataFrame(rows)
        exact_policy = {("kuhn", "infoA"): {0: 0.5, 1: 0.5}}
        out = score_property3_exact(df, exact_policy)
        assert math.isinf(out["per_info_set"]["kuhn"]["infoA"])

    def test_empty_lm_rows_returns_nan_and_empty_matched_dicts(self):
        """Empty LM rows return NaN, plus the three matched/unmatched dicts
        empty.
        """
        rows = [
            {
                "game": "kuhn",
                "info_set": "infoA",
                "prompt": None,
                "cfr_policy": json.dumps({"0": 1.0}),
            }
        ]
        df = pl.DataFrame(rows)
        out = score_property3_exact(df, {})
        assert math.isnan(out["global_mean_kl"])
        assert out["matched_fraction"] == {}
        assert out["n_matched"] == {}
        assert out["n_unmatched"] == {}

    def test_mean_kl_is_klscore_carrying_coverage(self):
        """Mirrors test_mean_kl_is_klscore_carrying_coverage for
        score_property3 -- score_property3_exact must return the same
        KLScore type, not a bare float, so every KLScore-typed consumer
        can read
        either scorer's output identically.
        """
        cfr = {0: 0.5, 1: 0.5}
        rows = [_lm_row_for_exact("kuhn", "infoA", cfr)]
        df = pl.DataFrame(rows)
        exact_policy = {("kuhn", "infoA"): {0: 0.5, 1: 0.5}}
        out = score_property3_exact(df, exact_policy)
        score = out["mean_kl"]["kuhn"]
        assert isinstance(score, KLScore)
        assert score.on_support_fraction == 1.0

    def test_off_support_rows_excluded_from_matching_but_counted_in_coverage(self):
        """Mirrors score_property3's own off-support exclusion test:
        off-support rows must not silently vanish from
        `on_support_fraction`, and must not count towards
        `matched_fraction`'s denominator either -- an off-support info-set
        is excluded from `game_df` before the matched/unmatched loop even
        runs.
        """
        cfr = {0: 0.5, 1: 0.5}
        rows = [
            _lm_row_for_exact("kuhn", "infoA", cfr, cfr_ref_off_support=False)
            for _ in range(3)
        ] + [_lm_row_for_exact("kuhn", "infoB", cfr, cfr_ref_off_support=True)]
        df = pl.DataFrame(rows)
        exact_policy = {("kuhn", "infoA"): {0: 0.5, 1: 0.5}}
        out = score_property3_exact(df, exact_policy)
        assert out["mean_kl"]["kuhn"].on_support_fraction == pytest.approx(0.75)
        # infoB is off-support, so it never reaches the matched/unmatched
        # count at all -- denominator is 1 (infoA only), not 2.
        assert out["n_matched"]["kuhn"] + out["n_unmatched"]["kuhn"] == 1


class TestActorParameterisation:
    """The population a scorer means is "the rows of the agent under test",
    which is an `actor` predicate, not `prompt.is_not_null()`. Opponent rows
    carry `parsed_action` on every row, so the same logs score a principled
    agent on the same yardstick once the filter is parameterised.

    The estimand ignores the per-row `cfr_policy` column and needs a real
    info-set to produce a meaningful number. Two actors are scored at Kuhn's
    real Queen-opening info-set (`_queen_open_info`): one always plays the
    equilibrium-optimal action (Pass, regret == 0 exactly, since it is the
    argmax the regret table computed from), the other always plays the
    known-suboptimal one (Bet, regret clearly negative), so actor selection
    produces genuinely different numbers.
    """

    @staticmethod
    def _row(actor, prompt, action, info_set):
        return {
            "game": "kuhn",
            "info_set": info_set,
            "parsed_action": action,
            "prompt": prompt,
            "actor": actor,
        }

    def test_a_named_opponent_is_scored_over_exactly_its_own_rows(
        self, kuhn_cfr, scoped_cfr_cache
    ):
        """P-opponent-scorable.
        """
        game, _ = kuhn_cfr
        info_str, pass_a, bet_a = _queen_open_info(game)
        lm = [self._row("model-x", "p", pass_a, info_str) for _ in range(100)]
        bayesian = [self._row("bayesian", None, bet_a, info_str) for _ in range(100)]
        df = pl.DataFrame(lm + bayesian)

        out = score_property3(df, actor="bayesian")
        assert out["actor_filter"] == "actor=bayesian"
        assert out["per_info_set"]["kuhn"][info_str] < -0.1

        # The two populations are genuinely different numbers, so the filter
        # is selecting and not merely relabelling.
        lm_out = score_property3(df, actor="model-x")
        assert lm_out["per_info_set"]["kuhn"][info_str] == pytest.approx(0.0, abs=1e-6)

    def test_legacy_frame_without_an_actor_column_is_unchanged_and_labelled(
        self, kuhn_cfr, scoped_cfr_cache
    ):
        """P-legacy: an arm with no `actor` column must score by the
        prompt-null population, with the fallback labelled rather than silent
        (no silent fallbacks).
        """
        game, _ = kuhn_cfr
        info_str, pass_a, _ = _queen_open_info(game)
        rows = [
            {
                "game": "kuhn",
                "info_set": info_str,
                "parsed_action": pass_a,
                "prompt": "p",
            }
            for _ in range(100)
        ]
        df = pl.DataFrame(rows)
        assert "actor" not in df.columns
        out = score_property3(df)
        assert out["per_info_set"]["kuhn"][info_str] == pytest.approx(0.0, abs=1e-6)
        assert out["actor_filter"] == "legacy-prompt-null"

    def test_naming_an_actor_a_legacy_frame_cannot_identify_raises(self):
        """No silent fallbacks, at the one place the legacy fallback rule
        would break it: `actor="bayesian"` on a frame with no
        `actor` column cannot be honoured, and answering with the
        prompt-null population would publish the LM's regret under the
        opponent's name.
        """
        rows = [
            {
                "game": "kuhn",
                "info_set": "infoA",
                "parsed_action": 0,
                "prompt": "p",
            }
        ]
        with pytest.raises(ValueError, match="actor"):
            score_property3(pl.DataFrame(rows), actor="bayesian")

    def test_exact_variant_takes_the_same_parameter(self):
        """Two anchors: `score_property3_exact` is a second public entry
        point over the same population (`property3.py`), and a fix
        applied to only one of the two filters is a defect.
        """
        bayesian = [
            {
                "game": "kuhn",
                "info_set": "infoA",
                "parsed_action": 0,
                "cfr_policy": json.dumps({"0": 0.5, "1": 0.5}),
                "prompt": None,
                "actor": "bayesian",
            }
            for _ in range(4)
        ]
        df = pl.DataFrame(bayesian)
        exact = {("kuhn", "infoA"): {0: 1.0, 1: 0.0}}
        out = score_property3_exact(df, exact, actor="bayesian")
        assert out["actor_filter"] == "actor=bayesian"
        assert out["n_matched"]["kuhn"] == 1
        # KL(exact=deterministic 0 || cfr=50/50) = ln 2.
        assert out["per_info_set"]["kuhn"]["infoA"] == pytest.approx(
            math.log(2.0), abs=1e-12
        )

    def test_exact_variant_labels_the_legacy_fallback_too(self):
        """The label is what makes the fallback non-silent, so it must reach
        both entry points' output dicts.
        """
        rows = [_lm_row_for_exact("kuhn", "infoA", {0: 0.5, 1: 0.5})]
        out = score_property3_exact(pl.DataFrame(rows), {})
        assert out["actor_filter"] == "legacy-prompt-null"


class TestForeignProbeType:
    """The plain (non-paired) scorers need a probe_type guard: without one, a
    frame assembled from a directory that also holds a probe arm's rows
    (reachable via an unfiltered `--data` glob) is
    silently averaged in as base-arm evidence.
    """

    def _row(self, probe_type):
        return {
            "game": "kuhn",
            "info_set": "infoA",
            "parsed_action": 0,
            "prompt": "some prompt",
            "probe_type": probe_type,
        }

    def test_score_property3_refuses_a_frame_holding_a_foreign_probe_type(self):
        """Raises before any CFR loading (`refuse_foreign_probe_type` fires
        right after `require_actor_column`), so this needs no
        `scoped_cfr_cache`.
        """
        df = pl.DataFrame([self._row("none"), self._row("relabel")])
        with pytest.raises(ValueError, match="relabel"):
            score_property3(df)

    def test_score_property3_accepts_an_all_base_frame(self, scoped_cfr_cache):
        """Reaches the per-game loop (probe_type check passes), which
        loads/solves a real Kuhn CFR reference internally -- needs
        `scoped_cfr_cache` so that solve's cache write lands in a throwaway
        dir, not the real project `data/cfr_policies/`. The fake "infoA"
        info-set matches nothing in the real regret table, so every row is
        skipped and the game reads NaN -- this test only asserts "must not
        raise".
        """
        df = pl.DataFrame([self._row("none"), self._row("none")])
        score_property3(df)  # must not raise

    def test_score_property3_exact_refuses_a_frame_holding_a_foreign_probe_type(self):
        rows = [
            {**_lm_row_for_exact("kuhn", "infoA", {0: 1.0}), "probe_type": "none"},
            {
                **_lm_row_for_exact("kuhn", "infoA", {0: 1.0}),
                "probe_type": "payoff_scale",
            },
        ]
        with pytest.raises(ValueError, match="payoff_scale"):
            score_property3_exact(pl.DataFrame(rows), {("kuhn", "infoA"): {0: 1.0}})


def test_null_parsed_action_rows_are_counted_in_n_null_action(kuhn_cfr, scoped_cfr_cache):
    """
    a kept row whose parsed_action is null is counted in ``n_null_action`` rather than leaving
    the regret mean with no trace, and the mean itself is unchanged by it
    """
    game, _ = kuhn_cfr
    df = _p3_fixture(game, "pass", on_support_rows=3)
    null_row = df.head(1).with_columns(pl.lit(None, dtype=df["parsed_action"].dtype).alias("parsed_action"))
    out = score_property3(pl.concat([df, null_row, null_row]))
    assert out["n_null_action"] == {"kuhn": 2}
    assert out["mean_kl"]["kuhn"].value == score_property3(df)["mean_kl"]["kuhn"].value
