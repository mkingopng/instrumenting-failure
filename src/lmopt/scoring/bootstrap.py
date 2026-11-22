"""Statistical framework for the property scores.

Three offline tools over the logged Parquet (scoring is offline):

1. ``bootstrap_ci`` - hand-level nonparametric bootstrap confidence intervals for
   any per-game scalar statistic. Resampling is at the hand level (the unit of
   independent replication in a match), stratified by game so each game keeps its
   sample size. A "hand" is a unique ``(matchup_id, game, seed, hand_idx)`` - so
   resampling a hand pulls all of its decision-node rows together, preserving the
   within-hand correlation the naive per-row bootstrap would break. This remains
   correct for any within-shard question; it is not the right unit for a
   between-run uncertainty statement (see ``replicate_ci_by_key`` below).

2. ``replicate_ci_by_key``: shard-replicate Student-t confidence intervals for
   the same kind of per-key scalar statistic. The shard is the
   unit of independent replication in this harness, not the hand; a stateful
   server can correlate hands within one shard. This is the path the published
   property CIs (P2, P3, P5, P7a) use.

3. ``per_infoset_visit_counts`` - per-(game, info-set) visit counts with a
   min-visit flag, to verify each information set is seen often enough for its
   per-info-set property estimates to be trustworthy.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np
import polars as pl

# Columns that jointly identify one independent hand. Only those present in the
# frame are used (synthetic test frames may omit seed/matchup_id).
_HAND_COLS = ["matchup_id", "game", "seed", "hand_idx"]

# Within-hand ordering columns, most significant first. All three are in
# schema.REQUIRED_COLUMNS (never null in a valid record) and the first,
# ``node_idx``, is 0-indexed within the hand, so on a real logged frame
# _HAND_COLS + node_idx already identifies a row uniquely. Only those present
# are used, for the same reason as _HAND_COLS.
_WITHIN_HAND_COLS = ["node_idx", "player_to_act", "info_set"]


def _hand_cols(df: pl.DataFrame) -> list[str]:
    cols = [c for c in _HAND_COLS if c in df.columns]
    if "hand_idx" not in cols:
        raise ValueError("bootstrap requires a 'hand_idx' column")
    if "game" not in cols:
        raise ValueError("bootstrap requires a 'game' column")
    return cols


def resample_hands(df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Return a hand-level bootstrap resample of ``df``, stratified by game.

    Each game's hands are sampled with replacement to the same count, and all
    rows of a sampled hand are duplicated together (so within-hand structure is
    preserved).

    Determinism contract: same (df-content, seed) yields identical output frame
    including row order, regardless of input row order or polars invocation.

    Scope of that contract, stated exactly: the output is
    sorted on the hand keys plus ``node_idx``, ``player_to_act`` and
    ``info_set`` where present, with ``maintain_order=True``. On a real logged
    frame the hand keys plus ``node_idx`` are already unique, so the order is
    total and input order cannot reach it. On a synthetic frame that omits
    ``node_idx`` several rows can tie on every sort key; those keep their join
    order, which does track input order. Give such a frame a ``node_idx`` (or
    any distinct within-hand key) to get the unconditional guarantee.

    The per-game seed derivation, stated so it is re-checkable: ``game_seed = seed + int(sha256(game)[:8], 16)``, i.e. the base
    seed plus the leading 32 bits of the game key's sha256 digest, read as an
    unsigned integer in ``[0, 2**32)``. sha256 rather than ``hash()`` because
    Python's string hash is salted per process, which would make the draws
    irreproducible across runs; a truncated cryptographic digest is used purely
    as a fixed, portable spreading function, not for any security property.

    Two distinct collisions are possible, and the second is the larger of the
    two, so both are stated rather than only the obvious one:

    * Within one resample two games collide only if their 32-bit prefixes are
      equal, which would give them identical draw sequences at equal ``n``. Over
      the ``k = 9`` game keys the registry names the birthday bound is
      ``C(9,2) / 2**32 = 8.4e-9``, and the actual prefixes are all distinct,
      with a minimum pairwise gap of 25,235,064.
    * Across resamples ``bootstrap_ci`` calls this with ``seed + i + 1``, so
      game A at resample i shares a seed with game B at resample j whenever the
      prefix difference equals ``j - i``. Over 1000 resamples that is bounded by
      ``C(9,2) * 1000 / 2**32 = 8.4e-6``. The consequence of such a hit is one
      pair of games drawing the same index sequence in two different replicates,
      which perturbs a percentile bound in the last digits, not a wrong estimate.

    Falsification: add a game key whose 32-bit prefix repeats an existing one, or
    raise ``n_resamples`` far above 1000, and the second bound grows linearly;
    re-measure the prefixes rather than trusting the figures above.
    """
    hand_cols = _hand_cols(df)
    # Sort before unique: makes the result independent of input row order.
    hands = df.select(hand_cols).sort(hand_cols).unique(maintain_order=True)

    picks = []
    # Iterate games in sorted order: stable across invocations.
    for game in sorted(hands["game"].unique().to_list()):
        gh = hands.filter(pl.col("game") == game)
        n = gh.height
        if n == 0:
            continue
        # Per-game seeded RNG: roster change in one game cannot shift another's draws.
        # Use deterministic hash (not Python's randomized hash()) for cross-process stability.
        game_hash = int(hashlib.sha256(game.encode()).hexdigest()[:8], 16)
        game_seed = seed + game_hash
        game_rng = np.random.default_rng(game_seed)
        idx = game_rng.integers(0, n, size=n)
        picks.append(gh.select(pl.all().gather(pl.Series(idx))))
    sampled = pl.concat(picks)
    # Duplicate hands -> duplicated row sets via the join.
    #
    # The final sort must be fully specified, because order-sensitive float
    # reductions run downstream. Sorting on ``hand_cols`` alone would leave the
    # order of a hand's own rows unspecified twice over: polars' sort is not
    # stable unless asked (equal keys may be permuted between two invocations
    # of the identical call), and a duplicated hand contributes several
    # identical key blocks. Both are closed here: sort on the hand keys
    # plus every within-hand ordering column present, then ``maintain_order``
    # for the residue.
    order_cols = hand_cols + [c for c in _WITHIN_HAND_COLS if c in df.columns]
    joined = sampled.join(df, on=hand_cols, how="left")
    return joined.sort(order_cols, maintain_order=True)


def bootstrap_ci(
    df: pl.DataFrame,
    stat_fn: Callable[[pl.DataFrame], dict],
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Percentile bootstrap CIs for a per-key scalar statistic.
    :param df: full logged frame.
    :param stat_fn: maps a frame to ``{key: float}`` (key is typically a game name,
        or ``"game"`` for a global scalar). Called on the real frame for the
        point estimate and on each resample.
    :param n_resamples: bootstrap replicates (target: 1000).
    :param level: central CI mass (0.95 -> 2.5%/97.5% percentiles).
    :param seed: base RNG seed; resample ``i`` uses ``seed + i + 1``.
    :returns: ``{key: {"point", "lo", "hi", "se", "n"}}`` where ``n`` is the number of
        non-NaN replicate values that contributed.
    """
    point = stat_fn(df)
    keys = list(point.keys())
    samples: dict[str, list[float]] = {k: [] for k in keys}

    for i in range(n_resamples):
        rdf = resample_hands(df, seed + i + 1)
        s = stat_fn(rdf)
        for k in keys:
            v = s.get(k, float("nan"))
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                samples[k].append(float(v))

    alpha = (1.0 - level) / 2.0
    out: dict[str, dict] = {}
    for k in keys:
        arr = np.array(samples[k], dtype=float)
        pv = point[k]
        pv = float(pv) if isinstance(pv, (int, float)) else float("nan")
        if arr.size == 0:
            out[k] = {
                "point": pv,
                "lo": float("nan"),
                "hi": float("nan"),
                "se": float("nan"),
                "n": 0,
            }
        else:
            out[k] = {
                "point": pv,
                "lo": float(np.percentile(arr, 100.0 * alpha)),
                "hi": float(np.percentile(arr, 100.0 * (1.0 - alpha))),
                "se": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "n": int(arr.size),
            }
    return out


def replicate_ci_by_key(
    df: pl.DataFrame,
    stat_fn: Callable[[pl.DataFrame], dict],
    level: float = 0.95,
) -> dict:
    """
    Shard-replicate CIs for a per-key scalar statistic.

    Computes ``stat_fn`` independently on each shard, then forms a Student-t
    interval over those replicate values. The shard is the unit of independent
    replication (run_sweep.py builds one ``MatchupSpec`` per (game, opponent,
    seed) shard, whose ``seed`` becomes ``run_match``'s ``seed`` in
    match_runner.py); the hand is not, because a stateful server can correlate hands within a shard.

    This does not replace ``bootstrap_ci``. The hand-level bootstrap remains
    correct for any within-shard question; it is simply not the right unit for
    a between-run uncertainty statement.
    :param df: full logged frame, with a ``seed`` column identifying shards.
    :param stat_fn: maps a frame to ``{key: float}``. Called once on the full
        pooled frame (for ``point_pooled``) and once per shard (to build
        the replicate values ``point``/``lo``/``hi`` are derived from).
    :param level: passed through to ``replicate_ci`` (0.95 only; see there).
    :returns: ``{key: {"point", "point_pooled", "lo", "hi", "se", "n",
        "n_replicates", "method", "note"}}``.
        ``point`` is the mean of the per-shard replicate values, i.e.
        ``replicate_ci``'s own ``"point"``: the same centre the Student-t
        interval in ``lo``/``hi`` is built around, so ``lo <= point <= hi``
        holds by construction.
        ``point_pooled`` is ``stat_fn`` evaluated once on the full pooled
        frame (all shards concatenated). Kept as an auditable sibling, not
        reported as ``point``, because the two can differ for any estimator
        whose value depends on the row-count composition it sees, not only a
        sample-size-dependent one: a plain visit-weighted mean already
        differs between "mean of per-shard means" and "mean over pooled
        rows" whenever shard sizes/compositions are uneven (a KL estimator
        with a Laplace-smoothing denominator ``n + eps*K`` makes this worse;
        property3's regret-based estimator has no such denominator but
        still exhibits the same general divergence via uneven composition). Reporting ``point_pooled`` as ``point`` while
        ``lo``/``hi`` are built from the (differently centred) replicate
        values breaks containment (e.g. under a KL estimator on a synthetic
        fixture, leduc P3: pooled point 0.353957 outside the degenerate
        replicate interval [0.349627, 0.349627]).
        ``n`` duplicates ``n_replicates`` for shape-compatibility with
        ``bootstrap_ci``'s return dict, where the same key counted resamples;
        here it counts usable shard replicates instead. No consumer reads
        ``"n"`` directly (both scripts and the W&B mirror pass the whole dict
        through).
    :raises ValueError: if the frame carries no ``seed`` column. Pooling silently
        would return a hand-level answer under a replicate label.
    """
    from lmopt.scoring.replicate import replicate_ci

    if "seed" not in df.columns:
        raise ValueError(
            "replicate_ci_by_key requires a 'seed' column to identify shards; "
            "refusing to pool silently"
        )
    pooled = stat_fn(df)
    keys = list(pooled.keys())
    per_key: dict[str, list[float]] = {k: [] for k in keys}

    for s in sorted(df["seed"].unique().to_list()):
        sdf = df.filter(pl.col("seed") == s)
        if sdf.is_empty():
            continue
        vals = stat_fn(sdf)
        for k in keys:
            v = vals.get(k, float("nan"))
            per_key[k].append(float(v) if isinstance(v, (int, float)) else float("nan"))

    out: dict[str, dict] = {}
    for k in keys:
        r = replicate_ci(per_key[k], level=level)
        pv_pooled = pooled[k]
        out[k] = {
            "point": r["point"],
            "point_pooled": (
                float(pv_pooled)
                if isinstance(pv_pooled, (int, float))
                else float("nan")
            ),
            "lo": r["lo"],
            "hi": r["hi"],
            "se": r["se"],
            "n": r["n_replicates"],
            "n_replicates": r["n_replicates"],
            "method": r["method"],
            "note": r["suppressed_reason"],
        }
    return out


def per_infoset_visit_counts(
    df: pl.DataFrame,
    min_visits: int = 30,
    lm_only: bool = True,
) -> dict:
    """
    Per-(game, info-set) visit counts, with under-visited sets flagged.
    :param df: full logged frame.
    :param min_visits: threshold below which an info-set is flagged under-visited.
    :param lm_only: count only the player-under-test decision nodes (rows with a
        non-null ``prompt``), which is what the LM property scorers estimate
        over. Set False to count all decision nodes.
    :returns: ``{game: {"counts": {info_set: n}, "under_visited": [...],
        "n_infosets": int, "n_under": int, "min_visits": int}}``.
    """
    d = df.filter(pl.col("prompt").is_not_null()) if lm_only else df
    if d.is_empty():
        return {}
    grouped = d.group_by(["game", "info_set"]).agg(pl.len().alias("n"))
    result: dict[str, dict] = {}
    for game in grouped["game"].unique().to_list():
        gc = grouped.filter(pl.col("game") == game)
        counts = {row["info_set"]: int(row["n"]) for row in gc.iter_rows(named=True)}
        under = sorted(is_ for is_, n in counts.items() if n < min_visits)
        result[game] = {
            "counts": counts,
            "under_visited": under,
            "n_infosets": len(counts),
            "n_under": len(under),
            "min_visits": min_visits,
        }
    return result
