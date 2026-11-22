"""P6 (off-path rationality): the tremble-referenced scorer and the opening
selector.

The estimand is a distance between the agent's realised policy inside a driven
subgame and a reference that is defined there, which plain CFR is not: at a
zero-reach info-set the average policy is whatever the solver's regrets happened
to leave, so "distance from CFR" would be partly a function of the run. The
tremble blend is what makes the reference unique off the equilibrium path, and
that is the thing these tests pin.
"""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("pyspiel")

from lmopt.agents.cfr_ref import CFRRefAgent
from lmopt.agents.random_agent import RandomAgent
from lmopt.games.cfr import solve_cfr, solve_cfr_tremble, tremble_policy
from lmopt.games.loaders import load_kuhn
from lmopt.games.policy_cache import load_or_solve_cfr
from lmopt.harness.match_runner import run_match
from lmopt.logging.writer import MatchWriter
from lmopt.scoring.property6 import (
    OFFPATH_PROBE_TYPE,
    offpath_reference,
    reference_policy_map,
    score_property6,
)

ITERS = 200


@pytest.fixture(scope="module")
def game():
    return load_kuhn()


@pytest.fixture(scope="module")
def uniform(game):
    from open_spiel.python.policy import UniformRandomPolicy

    return UniformRandomPolicy(game)


@pytest.fixture(scope="module")
def leduc():
    from lmopt.games.loaders import load_leduc

    return load_leduc()


def _rows(*specs, actor="random") -> pl.DataFrame:
    """A driven arm's frame: (info_set, player, [actions]) per info-set."""
    records = []
    for info_set, player, actions in specs:
        for action in actions:
            records.append(
                {
                    "game": "kuhn",
                    "info_set": info_set,
                    "player_to_act": player,
                    "parsed_action": action,
                    "actor": actor,
                    "prompt": "p",
                    "probe_type": OFFPATH_PROBE_TYPE,
                    "prompt_variant": "base",
                }
            )
    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# The reference: cached base, blend applied at load
# ---------------------------------------------------------------------------


def test_the_blend_of_a_solved_base_equals_the_solve_and_blend_wrapper(game):
    """`solve_cfr_tremble` keeps its signature as a thin wrapper over
    `tremble_policy`, so the two routes cannot produce different references.
    """
    base = solve_cfr(game, ITERS)
    blended = tremble_policy(game, base, tremble_eps=0.25)
    wrapped = solve_cfr_tremble(game, ITERS, tremble_eps=0.25)

    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)
    assert blended.action_probabilities(state) == wrapped.action_probabilities(state)


def test_the_reference_caches_the_base_policy_rather_than_the_blend(game, tmp_path):
    """`_TremblePolicy` is a function-local class, so
    it neither pickles nor fits `load_or_solve_cfr`'s TabularPolicy cache. The
    blend is cheap arithmetic on a solved base, so the base is what gets
    cached and the blend is applied at load.
    """
    offpath_reference(game, ITERS, cache_dir=str(tmp_path))
    # The base is on disk under the ordinary CFR key: a warm cache serves it.
    load_or_solve_cfr(game, ITERS, cache_dir=str(tmp_path), require_hit=True)


# ---------------------------------------------------------------------------
# score_property6
# ---------------------------------------------------------------------------


def test_the_distance_at_an_info_set_is_half_the_l1_from_the_reference(game, uniform):
    """A deterministic empirical policy against a uniform reference is TV 0.5
    on a two-action info-set; matching the reference is 0.0.
    """
    df = _rows(("0pb", 0, [0, 0, 0, 0]), ("1pb", 0, [0, 1]))
    out = score_property6(df, uniform, game=game, actor="random")

    assert out["tv_by_infoset"]["0pb"] == pytest.approx(0.5)
    assert out["tv_by_infoset"]["1pb"] == pytest.approx(0.0)
    assert out["n_offpath_infosets"] == 2
    assert out["tv_mean"] == pytest.approx(0.25)


def test_every_distance_carries_its_visit_count(game, uniform):
    """TV between two distributions is upward-biased at finite n, so a per-info-set distance
    without its sample size cannot be told from sampling noise.
    """
    df = _rows(("0pb", 0, [0, 0, 0, 0]), ("1pb", 0, [0, 1]))
    out = score_property6(df, uniform, game=game, actor="random")

    assert out["visits_by_infoset"] == {"0pb": 4, "1pb": 2}


def test_null_action_rows_are_counted_out_loud_not_only_excluded(game, uniform):
    """A row at a scored info-set whose `parsed_action` is null
    is excluded from `n_rows_scored` (it is not a choice) and reported under
    `n_null_dropped`, so the shrunken denominator is visible.
    """
    df = pl.concat(
        [
            _rows(("0pb", 0, [0, 0, 0, 0]), ("1pb", 0, [0, 1])),
            _rows(("0pb", 0, [None, None])),
        ]
    )
    out = score_property6(df, uniform, game=game, actor="random")

    assert out["n_rows_scored"] == 6, "the nulls are not scored"
    assert out["n_null_dropped"] == 2, "and they are not invisible either"
    # The distribution itself is untouched by the nulls: this is an accounting
    # fix, not a change of estimand.
    assert out["visits_by_infoset"] == {"0pb": 4, "1pb": 2}
    assert out["tv_mean"] == pytest.approx(0.25)


def test_the_visit_weighted_mean_moves_when_the_visits_are_unequal(game, uniform):
    """The secondary estimator, emitted beside the unweighted mean.
    """
    df = _rows(("0pb", 0, [0, 0, 0, 0]), ("1pb", 0, [0, 1]))
    out = score_property6(df, uniform, game=game, actor="random")

    assert out["tv_mean"] == pytest.approx(0.25)
    assert out["tv_mean_visit_weighted"] == pytest.approx((0.5 * 4) / 6)


def test_a_frame_holding_undriven_rows_is_refused(game, uniform):
    """P6's estimand is the policy inside the driven subgame.
    """
    df = _rows(("0pb", 0, [0, 1]))
    mixed = pl.concat([df, df.with_columns(pl.lit("none").alias("probe_type"))])
    with pytest.raises(ValueError, match="offpath_drive"):
        score_property6(mixed, uniform, game=game, actor="random")


def test_an_info_set_the_reference_never_reaches_is_refused(game, uniform):
    """The reference map is a full walk of the game the caller named, so an
    info-set it does not contain means the frame and the reference are not
    about the same game.
    """
    df = _rows(("not-an-info-set", 0, [0, 1]))
    with pytest.raises(ValueError, match="not-an-info-set"):
        score_property6(df, uniform, game=game, actor="random")


def test_a_frame_with_no_scored_rows_reports_nan_not_zero(game, uniform):
    df = _rows(("0pb", 0, [0, 1]))
    out = score_property6(df, uniform, game=game, actor="nobody")

    assert out["n_offpath_infosets"] == 0
    assert out["tv_mean"] != out["tv_mean"]  # NaN
    assert out["tv_mean_visit_weighted"] != out["tv_mean_visit_weighted"]


def test_the_scored_population_is_reported(game, uniform):
    """A filter that is not reported is a silent fallback.
    """
    df = _rows(("0pb", 0, [0, 1]))
    assert (
        score_property6(df, uniform, game=game, actor="random")["actor_filter"]
        == "actor=random"
    )
    assert score_property6(df, uniform, game=game)["actor_filter"] == (
        "legacy-prompt-null"
    )


# ---------------------------------------------------------------------------
# P-scorer-integration: the tremble reference is what the scorer measures
# against, through the real entrypoints
# ---------------------------------------------------------------------------


def test_the_tremble_reference_moves_the_score_away_from_plain_cfr(game, tmp_path):
    """P-scorer-integration's mutation anchor, with the tolerance trap
    handled: the blend moves each action
    probability by at most `tremble_eps`, so at the production default (1e-4)
    a tremble-vs-plain pin would be resolving sub-1e-4 differences and would
    flake. `tremble_eps=0.5` is therefore a test parameter: it makes the same
    mechanism visible at order 0.1, where the assertion is robust. The last
    assertion pins the trap itself, so the reason for the large eps stays
    checkable.
    """
    df = _rows(("0", 0, [0, 0, 0, 0]))
    plain = load_or_solve_cfr(game, ITERS, cache_dir=str(tmp_path))
    trembled = offpath_reference(game, ITERS, tremble_eps=0.5, cache_dir=str(tmp_path))
    defaulted = offpath_reference(game, ITERS, cache_dir=str(tmp_path))

    tv_plain = score_property6(df, plain, game=game, actor="random")["tv_mean"]
    tv_trembled = score_property6(df, trembled, game=game, actor="random")["tv_mean"]
    tv_default = score_property6(df, defaulted, game=game, actor="random")["tv_mean"]

    assert tv_trembled > tv_plain + 0.05
    # The trap, pinned: at the production default the same comparison is
    # sub-1e-4, which is why the divergence pin above does not use it.
    assert abs(tv_default - tv_plain) < 1e-3


def test_a_driven_arm_scores_end_to_end_against_the_real_reference(game, tmp_path):
    """The reachability anchor: a match driven through `run_match`, written to
    Parquet, read back and scored against a real `offpath_reference`.
    """
    out = tmp_path / "driven.parquet"
    policy = solve_cfr(game, ITERS)
    run_match(
        game,
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        CFRRefAgent(policy, seed=0),
        n_hands=12,
        seed=42,
        writer=MatchWriter(str(out)),
        drive_openings=[1],
        probe_type=OFFPATH_PROBE_TYPE,
    )

    df = pl.read_parquet(out)
    reference = offpath_reference(game, ITERS, cache_dir=str(tmp_path))
    scored = score_property6(df, reference, game=game, actor="random")

    assert scored["n_offpath_infosets"] >= 1
    assert 0.0 <= scored["tv_mean"] <= 1.0
    assert all(k.endswith("b") for k in scored["tv_by_infoset"])
    assert scored["game"] == "kuhn"


def test_the_reference_map_is_keyed_on_the_acting_player(game, uniform):
    """Two players can in principle render the same information-state string,
    and scoring one against the other's reference would be a silent mismatch.
    """
    ref_map = reference_policy_map(game, uniform)
    assert ("0pb", 0) in ref_map
    assert ("0b", 1) in ref_map
    assert all(isinstance(k, tuple) and len(k) == 2 for k in ref_map)


def test_a_null_probe_type_is_not_read_as_driven(game, uniform):
    df = _rows(("0pb", 0, [0, 1])).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("probe_type")
    )
    with pytest.raises(ValueError, match="null counts as foreign"):
        score_property6(df, uniform, game=game, actor="random")


def test_a_game_that_differs_from_the_frames_game_label_is_refused(
    game, leduc, uniform
):
    """Rows labelled leduc scored against a Kuhn game raise instead of scoring.
    """
    frame = _rows(("0", 0, [0, 1])).with_columns(pl.lit("leduc").alias("game"))

    with pytest.raises(ValueError, match="labelled game 'leduc'"):
        score_property6(frame, uniform, game=game)


def test_a_matching_game_is_scored_and_an_unregistered_label_is_not_checked(
    game, uniform
):
    """The registered label that matches, and a label the registry lacks, both still score.
    """
    matching = score_property6(_rows(("0", 0, [0, 1])), uniform, game=game)
    synthetic = score_property6(
        _rows(("0", 0, [0, 1])).with_columns(pl.lit("toygame").alias("game")),
        uniform,
        game=game,
    )

    assert matching["n_rows_scored"] == synthetic["n_rows_scored"] == 2
