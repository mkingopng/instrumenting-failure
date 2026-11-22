"""Shard-replicate intervals.

The unit of independent replication in this harness is the shard, not the hand.
`scripts/run_sweep.py` assigns shard ``i`` the matchup seed ``base_seed + i``,
and `src/lmopt/harness/match_runner.py` constructs one ``np.random.default_rng``
per shard which every chance node draws from. Distinct seeds therefore give
independent chance streams, so the shards are genuine replicates while the hands
inside one shard are not guaranteed to be
(a stateful vLLM server can correlate them; 20,639 OLMo rows have shown
temperature-0 nondeterminism).

This estimator rests on the independence premise above. Streams are spawned
via ``SeedSequence`` (`orchestration/shard.py`'s ``derive_stream_seeds``), which
makes it hold. Under an older seeding -- the chance stream seeded with the
matchup integer itself while the opponent's action sampling took
``matchup_seed + 1``, with sweep seeds spaced by 1 -- shard *S*'s opponent
stream is byte-identical to shard *S+1*'s deal stream (4 of 5 seeds), so the
shards are not independent along the very axis this interval treats as
independent, which biases coverage in an unknown direction: dependence between
replicates makes a t interval over them anti-conservative in the usual case.
**Intervals computed over the three early arms collected under that seeding
carry this caveat and should not be quoted without it**; those arms are
trials, not results.

The estimator computes the statistic independently per shard and forms a
Student-t interval over those replicate values. At 5 shards the t correction is
the difference between 2.776 and 1.96, so it is not optional.

Why not a cluster bootstrap: with 5 clusters a percentile bootstrap draws from
at most 5**5 distinct resamples and is known to under-cover badly. The t
interval on replicate values is the textbook estimator for a designed
experiment with a small number of independent replicates.

Why no scipy: scipy is importable in the dev venv but is not a declared
dependency of this project (it arrives transitively via the cvxpy dev extra),
so library code must not import it.
"""

from __future__ import annotations

import math
from typing import Sequence

# Two-sided 95% Student-t critical values, dof 1..30. Standard published table.
_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

_Z_CRIT_95 = 1.959964


def t_crit_95(dof: int) -> float:
    """
    Two-sided 95% critical value at ``dof`` degrees of freedom.
    Falls back to the normal quantile beyond dof 30, where the difference is
    under 5% and shrinking.
    :param dof: degrees of freedom
    :returns: `t_crit_95`
    """
    if dof < 1:
        return float("nan")
    return _T_CRIT_95.get(dof, _Z_CRIT_95)


def replicate_ci(values: Sequence[float], level: float = 0.95) -> dict:
    """
    Student-t interval over per-shard replicate values.
    :param values: one value of the statistic per shard. NaN entries are dropped
        and the surviving count is reported as ``n_replicates``.
    :param level: only 0.95 is supported; anything else raises, because the t
        table is tabulated at 95% only and silently returning a 95%
        interval under another label would be a silent fallback.
    :returns: ``{"point", "lo", "hi", "se", "n_replicates", "method",
        "suppressed_reason"}``. When fewer than 2 replicates survive, ``lo``
        and ``hi`` are NaN, ``method`` is ``"suppressed"`` and
        ``suppressed_reason`` says why. A constant set of replicates yields a
        genuine zero-width interval, which is information, not absence.
    """
    if level != 0.95:
        raise ValueError(
            f"replicate_ci supports level=0.95 only, got {level!r}; the t table "
            "is tabulated at 95% and returning another width silently would be "
            "a silent fallback"
        )

    finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(finite)
    nan = float("nan")

    if n == 0:
        return {
            "point": nan,
            "lo": nan,
            "hi": nan,
            "se": nan,
            "n_replicates": 0,
            "method": "suppressed",
            "suppressed_reason": "no finite replicate values",
        }
    point = sum(finite) / n
    if n < 2:
        return {
            "point": point,
            "lo": nan,
            "hi": nan,
            "se": nan,
            "n_replicates": n,
            "method": "suppressed",
            "suppressed_reason": (
                f"{n} usable replicate shard(s); a between-shard interval "
                "needs at least 2"
            ),
        }

    var = sum((v - point) ** 2 for v in finite) / (n - 1)
    se = math.sqrt(var / n)
    half = t_crit_95(n - 1) * se
    return {
        "point": point,
        "lo": point - half,
        "hi": point + half,
        "se": se,
        "n_replicates": n,
        "method": "t_replicate",
        "suppressed_reason": "",
    }


def effective_replicates(series_per_shard) -> dict:
    """
    Count distinct shards, for reporting only: never use this to drop
    replicates from an interval.

    A shard that reproduces another exactly carries no additional independent
    information, but the seed, not the value it produced, is the unit of
    replication -- `scripts/analyse_stats.py`'s `_cell_replicate_cis` forms
    every replicate interval over every shard, with no filtering or
    deduplication by whether two shards' values happen to agree, and this
    function's output must not be used to change that. Its remaining callers
    (`outcome_table`'s `n_effective_shards` column and
    `lmopt.scoring.robustness.seed_variance_budget`) consume it purely to
    report how many of a cell's shards are byte-identical, not to gate or
    filter anything downstream. Comparison is exact, never rounded: two
    shards that differ in the last bit are genuinely different draws.

    :param series_per_shard: one sequence of per-hand values per shard
    :returns: dict with keys ``n_nominal`` (shard count), ``n_effective``
        (count of distinct shards), ``duplicate_groups`` (sorted index
        lists, one per group with more than one member), and
        ``representative_indices`` (one index per distinct group --
        singleton or duplicate -- sorted ascending; each group's index is
        deterministically its lowest member, since `groups` is built by
        iterating `series_per_shard` in order via `enumerate`)
    """
    keys = [tuple(float(v) for v in s) for s in series_per_shard]
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    dupes = [idxs for idxs in groups.values() if len(idxs) > 1]
    return {
        "n_nominal": len(keys),
        "n_effective": len(groups),
        "duplicate_groups": sorted(dupes),
        "representative_indices": sorted(idxs[0] for idxs in groups.values()),
    }
