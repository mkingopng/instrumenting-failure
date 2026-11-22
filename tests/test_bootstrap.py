"""Tests for the statistical framework (bootstrap CIs + visit counts)."""

from __future__ import annotations


import numpy as np
import polars as pl
import pytest

from lmopt.scoring.bootstrap import (
    bootstrap_ci,
    per_infoset_visit_counts,
    resample_hands,
)


def _hand_df(game: str, rewards: list[float]) -> list[dict]:
    return [
        {
            "matchup_id": "m",
            "game": game,
            "seed": 0,
            "hand_idx": i,
            "info_set": "0",
            "prompt": "p",
            "reward": r,
        }
        for i, r in enumerate(rewards)
    ]


def _mean_reward_per_game(df: pl.DataFrame) -> dict:
    hands = df.group_by(["game", "seed", "hand_idx"]).agg(pl.col("reward").first())
    out = {}
    for game in hands["game"].unique().to_list():
        vals = hands.filter(pl.col("game") == game)["reward"].to_list()
        out[game] = float(np.mean(vals)) if vals else float("nan")
    return out


def test_constant_statistic_zero_width_ci():
    df = pl.DataFrame(_hand_df("kuhn", [0.0] * 10))
    out = bootstrap_ci(df, lambda d: {"kuhn": 5.0}, n_resamples=50, seed=1)
    assert out["kuhn"]["point"] == 5.0
    assert out["kuhn"]["lo"] == 5.0 and out["kuhn"]["hi"] == 5.0
    assert out["kuhn"]["se"] == 0.0


def test_ci_brackets_point_and_has_width():
    rng = np.random.default_rng(0)
    rewards = list(rng.normal(1.0, 1.0, size=200))
    df = pl.DataFrame(_hand_df("kuhn", rewards))
    out = bootstrap_ci(df, _mean_reward_per_game, n_resamples=300, seed=7)
    r = out["kuhn"]
    assert abs(r["point"] - float(np.mean(rewards))) < 1e-9
    assert r["lo"] < r["point"] < r["hi"]  # point inside the interval
    assert r["hi"] - r["lo"] > 0.0  # nonzero width
    assert r["lo"] < 1.0 < r["hi"]  # brackets the true mean 1.0


def test_resample_hands_determinism_independent_of_input_row_order():
    """Calling resample_hands twice with the same seed on frames holding
    identical content in different row orders yields identical sampled hand
    multisets per game and identical output row order.

    Full cross-process determinism (hashlib vs hash, sorted iteration, final
    sort) is proven by N=5 bit-identical regeneration rather than within-process
    mutations.
    """
    # Three games with different hand counts: kuhn=3, leduc=5, goofspiel=2.
    # Different counts ensure unsorted game iteration reallocates rng streams.
    rows_a = (
        _hand_df("kuhn", [0.0] * 3)
        + _hand_df("leduc", [1.0] * 5)
        + _hand_df("goofspiel", [2.0] * 2)
    )
    # Reverse order for df_b.
    rows_b = (
        _hand_df("goofspiel", [2.0] * 2)
        + _hand_df("leduc", [1.0] * 5)
        + _hand_df("kuhn", [0.0] * 3)
    )
    df_a = pl.DataFrame(rows_a)
    df_b = pl.DataFrame(rows_b)

    # Resample with the same seed.
    r_a = resample_hands(df_a, seed=42)
    r_b = resample_hands(df_b, seed=42)

    # Per-game sampled hand multisets must be identical.
    hand_cols = ["matchup_id", "game", "seed", "hand_idx"]
    for game in ["kuhn", "leduc", "goofspiel"]:
        hands_a = r_a.filter(pl.col("game") == game).select(hand_cols).sort(hand_cols)
        hands_b = r_b.filter(pl.col("game") == game).select(hand_cols).sort(hand_cols)
        assert hands_a.equals(hands_b), f"{game}: sampled hand multisets differ"

    # Output row order must also be identical (enables stable float reductions).
    # This is the exact order assertion an unsorted final frame breaks.
    assert r_a.equals(r_b), "output frames differ in content or row order"


def _multi_node_hand_rows(game: str, n_hands: int, n_nodes: int) -> list[dict]:
    """Hands with several decision-node rows each, as a real logged frame has."""
    return [
        {
            "matchup_id": "m",
            "game": game,
            "seed": 0,
            "hand_idx": h,
            "node_idx": k,
            "player_to_act": k % 2,
            "info_set": f"{game}-{h}-{k}",
            "prompt": "p",
            "reward": float(h),
        }
        for h in range(n_hands)
        for k in range(n_nodes)
    ]


def test_within_hand_row_order_is_fully_specified_not_left_to_input_order():
    """A hand's own rows come out in ``node_idx`` order whatever order they
    arrived in, so the downstream order-sensitive float reductions see one
    canonical frame.
    """
    rows = _multi_node_hand_rows("kuhn", n_hands=4, n_nodes=3)
    df_a = pl.DataFrame(rows)
    # Same content, node rows reversed inside every hand.
    df_b = pl.DataFrame(list(reversed(rows)))

    r_a = resample_hands(df_a, seed=11)
    r_b = resample_hands(df_b, seed=11)

    assert r_a.equals(r_b), "within-hand row order still depends on input order"

    # And the order is the canonical one, not merely a shared arbitrary one.
    for frame in (r_a, r_b):
        pairs = list(
            zip(frame["hand_idx"].to_list(), frame["node_idx"].to_list(), strict=False)
        )
        assert len(pairs) == 12  # 4 sampled hand-slots x 3 node rows
        # matchup_id/game/seed are constant here, so (hand_idx, node_idx)
        # ascending is the canonical order. Duplicated hands interleave
        # (h,0),(h,0),(h,1),... which is still ascending.
        assert pairs == sorted(pairs), f"non-canonical node order: {pairs}"


def test_resample_preserves_both_games():
    rows = _hand_df("kuhn", [0.0] * 8) + _hand_df("leduc", [1.0] * 4)
    df = pl.DataFrame(rows)
    rdf = resample_hands(df, seed=3)
    games = set(rdf["game"].unique().to_list())
    assert games == {"kuhn", "leduc"}  # stratified: neither game dropped
    # Each game keeps its number of sampled hand-slots (duplicates allowed).
    for game, n_hands in (("kuhn", 8), ("leduc", 4)):
        got = rdf.filter(pl.col("game") == game).select(["seed", "hand_idx"]).height
        assert got == n_hands  # one row per hand here (single node per hand)


def test_visit_counts_flags_under_visited():
    # kuhn info-set "0" visited 40x, "1" visited 5x; min_visits=30 flags "1".
    rows = [
        {
            "matchup_id": "m",
            "game": "kuhn",
            "seed": 0,
            "hand_idx": i,
            "info_set": "0",
            "prompt": "p",
            "reward": 0.0,
        }
        for i in range(40)
    ] + [
        {
            "matchup_id": "m",
            "game": "kuhn",
            "seed": 0,
            "hand_idx": 100 + i,
            "info_set": "1",
            "prompt": "p",
            "reward": 0.0,
        }
        for i in range(5)
    ]
    df = pl.DataFrame(rows)
    vc = per_infoset_visit_counts(df, min_visits=30)
    assert vc["kuhn"]["counts"]["0"] == 40
    assert vc["kuhn"]["counts"]["1"] == 5
    assert vc["kuhn"]["under_visited"] == ["1"]
    assert vc["kuhn"]["n_under"] == 1


def test_visit_counts_lm_only_excludes_opponent_rows():
    rows = [
        {
            "matchup_id": "m",
            "game": "kuhn",
            "seed": 0,
            "hand_idx": i,
            "info_set": "0",
            "prompt": "p",
            "reward": 0.0,
        }
        for i in range(5)
    ] + [
        {
            "matchup_id": "m",
            "game": "kuhn",
            "seed": 0,
            "hand_idx": i,
            "info_set": "opp",
            "prompt": None,
            "reward": 0.0,
        }
        for i in range(50)
    ]
    df = pl.DataFrame(rows)
    vc = per_infoset_visit_counts(df, min_visits=1, lm_only=True)
    assert "opp" not in vc["kuhn"]["counts"]  # opponent rows excluded
    assert vc["kuhn"]["counts"]["0"] == 5


def test_replicate_ci_by_key_widens_on_between_shard_disagreement():
    """THE positive case. Two frames with comparable, but not identical,
    pooled rows must get very different property CIs when their between-shard
    structure differs. The
    pooled composition is 5 zeros/6 ones (agree, 11 rows) vs 6 zeros/4 ones
    (disagree, 10 rows): close, not an identical multiset.

    One agreeing shard is perturbed to 3 hands ([0.0, 1.0, 1.0], mean 2/3)
    while the other four stay [0.0, 1.0] (mean 0.5), so the agreeing CI has a
    strictly positive baseline width. A fully symmetric agree fixture (every
    shard identical) gives every shard mean equal and a zero-width interval;
    the `> 5 *` comparison then collapses to "width > 0" and passes without
    exercising the between-shard estimator at all."""
    import polars as pl
    from lmopt.scoring.bootstrap import replicate_ci_by_key

    def stat(d):
        return {
            g: float(d.filter(pl.col("game") == g)["x"].mean())
            for g in d["game"].unique().to_list()
        }

    def frame(per_shard):
        rows = []
        for s, xs in enumerate(per_shard):
            for h, x in enumerate(xs):
                rows.append({"game": "g", "seed": s, "hand_idx": h, "x": x})
        return pl.DataFrame(rows)

    agree = frame([[0.0, 1.0, 1.0]] + [[0.0, 1.0]] * 4)
    dis = frame([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    ra = replicate_ci_by_key(agree, stat)["g"]
    rd = replicate_ci_by_key(dis, stat)["g"]
    wa = ra["hi"] - ra["lo"]
    wd = rd["hi"] - rd["lo"]
    assert wa > 0
    assert wd > 5 * wa
    assert ra["method"] == "t_replicate"


def test_replicate_ci_by_key_null_matches_analytic_width():
    """NULL that must stay quiet. Independent shards from one distribution
    must give a width near the analytic 2*t(4)*sd/sqrt(5)."""
    import numpy as np
    import polars as pl
    from lmopt.scoring.bootstrap import replicate_ci_by_key

    rng = np.random.default_rng(7)
    per_shard = [rng.normal(0.0, 1.0, size=200).tolist() for _ in range(5)]
    rows = [
        {"game": "g", "seed": s, "hand_idx": h, "x": x}
        for s, xs in enumerate(per_shard)
        for h, x in enumerate(xs)
    ]
    df = pl.DataFrame(rows)

    def stat(d):
        return {"g": float(d["x"].mean())}

    r = replicate_ci_by_key(df, stat)["g"]
    means = [float(np.mean(s)) for s in per_shard]
    sd = float(np.std(means, ddof=1))
    expected = 2 * 2.776 * sd / np.sqrt(5)
    assert (r["hi"] - r["lo"]) == pytest.approx(expected, rel=1e-6)


def test_missing_seed_column_raises_rather_than_silently_pooling():
    import polars as pl
    import pytest as _pt
    from lmopt.scoring.bootstrap import replicate_ci_by_key

    df = pl.DataFrame({"game": ["g"], "hand_idx": [0], "x": [1.0]})
    with _pt.raises(ValueError, match="seed"):
        replicate_ci_by_key(df, lambda d: {"g": 1.0})
