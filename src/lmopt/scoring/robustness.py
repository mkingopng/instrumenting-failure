"""Effect sizes and robustness statistics.

Pure offline functions over logged frames and existing artifact rows:
scoring is offline.
"""

import dataclasses
import math
from typing import Sequence, cast

import numpy as np
import polars as pl
from scipy.stats import norm

from lmopt.scoring.bootstrap import bootstrap_ci
from lmopt.scoring.replicate import effective_replicates, replicate_ci


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Hedges' g: small-sample corrected standardized mean difference.

    Hedges' g = Cohen's d times the small-sample correction factor J,
    where J = 1 - 3/(4*df - 1) and df = n_a + n_b - 2.
    :param a: first sample (hand-level rewards).
    :param b: second sample (hand-level rewards).
    :returns: Hedges' g effect size (positive when mean(a) > mean(b)). Returns NaN
        for n<2 or zero pooled SD with unequal means (undefined effect size,
        mirroring the scorers' NaN-for-empty convention).
    """
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    n_a = len(a_arr)
    n_b = len(b_arr)

    if n_a < 2 or n_b < 2:
        return float("nan")

    mean_a = np.mean(a_arr)
    mean_b = np.mean(b_arr)
    var_a = np.var(a_arr, ddof=1)
    var_b = np.var(b_arr, ddof=1)

    # Pooled variance with ddof=1
    df = n_a + n_b - 2
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / df
    pooled_sd = math.sqrt(pooled_var)

    if pooled_sd == 0:
        return 0.0 if mean_a == mean_b else float("nan")

    # Cohen's d
    cohens_d = (mean_a - mean_b) / pooled_sd

    # Small-sample correction
    j = 1.0 - (3.0 / (4.0 * df - 1.0))

    return cohens_d * j


def _cliffs_delta_naive(a: Sequence[float], b: Sequence[float]) -> float:
    """Naive O(n_a * n_b) double-loop implementation (frozen reference).

    Kept as the reference implementation for equivalence testing. Production
    code uses the vectorized cliffs_delta().
    """
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    n_a = len(a_arr)
    n_b = len(b_arr)

    if n_a == 0 or n_b == 0:
        return float("nan")

    n_greater = 0
    n_less = 0

    for a_val in a_arr:
        for b_val in b_arr:
            if a_val > b_val:
                n_greater += 1
            elif a_val < b_val:
                n_less += 1

    total_pairs = n_a * n_b
    return (n_greater - n_less) / total_pairs


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cliff's delta: tie-robust rank-based effect size (vectorized).

    Cliff's delta = P(a>b) - P(a<b) over all cross pairs.
    Range: [-1, 1] where 0 indicates no difference.

    Vectorized implementation using numpy sort + searchsorted:
    - Sort b ascending
    - For each a_val: count b < a_val and b > a_val via binary search
    - Handles ties correctly (neither > nor <)
    :param a: first sample (hand-level rewards).
    :param b: second sample (hand-level rewards).
    :returns: Cliff's delta (positive when a tends to be greater than b). Returns
        NaN for empty samples.
    """
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    n_a = len(a_arr)
    n_b = len(b_arr)

    if n_a == 0 or n_b == 0:
        return float("nan")

    # Sort b for binary search
    b_sorted = np.sort(b_arr)

    # For each a_val, count how many b values are less than or greater than it
    n_greater = 0  # count of pairs where a > b
    n_less = 0  # count of pairs where a < b

    for a_val in a_arr:
        # Count b < a_val: all b values to the left of a_val's insert position
        # searchsorted with side='left' finds the first position where a_val could be inserted
        # to maintain sorted order, so all elements before that position are < a_val
        # When b < a_val, that means a_val > b, so this contributes to n_greater
        idx_left = np.searchsorted(b_sorted, a_val, side="left")
        count_b_less_than_a = idx_left  # b values strictly < a_val

        # Count b > a_val: all b values to the right of a_val's rightmost position
        # searchsorted with side='right' finds the position after all elements == a_val
        # When b > a_val, that means a_val < b, so this contributes to n_less
        idx_right = np.searchsorted(b_sorted, a_val, side="right")
        count_b_greater_than_a = n_b - idx_right  # b values strictly > a_val

        n_greater += count_b_less_than_a  # a > b
        n_less += count_b_greater_than_a  # a < b

    total_pairs = n_a * n_b
    return (n_greater - n_less) / total_pairs


def effect_sizes_ci(
    df: pl.DataFrame,
    group_col: str,
    value_col: str,
    *,
    group_a=None,
    group_b=None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """
    Bootstrap CIs for effect sizes between two groups.
    :param df: logged frame with group and value columns (must have hand_id for
        bootstrap resampling).
    :param group_col: column name identifying the two groups.
    :param value_col: column name with hand-level rewards.
    :param group_a: first group value (if both are None, uses first alphabetically;
        naming only one of the two raises).
    :param group_b: second group value (if both are None, uses second alphabetically).
    :param n_resamples: bootstrap replicates (default 1000).
    :param seed: RNG seed for bootstrap.
    :returns: List of dicts, one per metric (hedges_g, cliffs_delta), each with:
        - metric: "hedges_g" or "cliffs_delta"
        - value: point estimate (positive when group_a > group_b)
        - ci_lo: lower CI bound
        - ci_hi: upper CI bound
        - n_a: sample size of group_a
        - n_b: sample size of group_b
        - group_a: the first group identifier
        - group_b: the second group identifier
    """
    groups = df[group_col].unique().sort()
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")

    # Explicit group ordering: caller specifies both, or neither for the alphabetical order
    if (group_a is None) != (group_b is None):
        raise ValueError(
            f"effect_sizes_ci: name both groups or neither, got group_a={group_a!r} and "
            f"group_b={group_b!r}; one named group cannot fix the effect's sign"
        )
    if group_a is None:
        group_a, group_b = groups[0], groups[1]
    else:
        # Validate provided groups exist in data
        if group_a not in groups or group_b not in groups:
            raise ValueError(
                f"Provided groups {group_a}, {group_b} not found in data: {groups}"
            )

    # Get sample sizes via polars count
    n_a = df.filter(pl.col(group_col) == group_a).height
    n_b = df.filter(pl.col(group_col) == group_b).height

    # Define stat_fn wrappers for bootstrap_ci
    def hedges_stat_fn(frame: pl.DataFrame) -> dict:
        """
        :param frame:
        :returns:
        """
        a_vals = frame.filter(pl.col(group_col) == group_a)[value_col].to_list()
        b_vals = frame.filter(pl.col(group_col) == group_b)[value_col].to_list()
        g = hedges_g(a_vals, b_vals)
        return {"hedges_g": g}

    def cliffs_stat_fn(frame: pl.DataFrame) -> dict:
        """
        :param frame:
        :returns:
        """
        a_vals = frame.filter(pl.col(group_col) == group_a)[value_col].to_list()
        b_vals = frame.filter(pl.col(group_col) == group_b)[value_col].to_list()
        delta = cliffs_delta(a_vals, b_vals)
        return {"cliffs_delta": delta}

    # Compute CIs via bootstrap_ci
    hedges_ci = bootstrap_ci(
        df, hedges_stat_fn, n_resamples=n_resamples, level=0.95, seed=seed
    )
    cliffs_ci = bootstrap_ci(
        df, cliffs_stat_fn, n_resamples=n_resamples, level=0.95, seed=seed
    )

    results = []

    # Hedges' g result
    h = hedges_ci["hedges_g"]
    results.append(
        {
            "metric": "hedges_g",
            "value": h["point"],
            "ci_lo": h["lo"],
            "ci_hi": h["hi"],
            "n_a": n_a,
            "n_b": n_b,
            "group_a": group_a,
            "group_b": group_b,
        }
    )

    # Cliff's delta result
    c = cliffs_ci["cliffs_delta"]
    results.append(
        {
            "metric": "cliffs_delta",
            "value": c["point"],
            "ci_lo": c["lo"],
            "ci_hi": c["hi"],
            "n_a": n_a,
            "n_b": n_b,
            "group_a": group_a,
            "group_b": group_b,
        }
    )

    return results


def holm_adjusted_levels(n_tests: int, alpha: float = 0.05) -> list[float]:
    """
    Holm-Bonferroni step-down confidence levels for multiple comparisons.

    Returns per-rank confidence levels in descending order (most significant
    first). The k-th most significant test (k=1..n) receives level
    1 - alpha/(n - k + 1).

    Example:
        For n=4, alpha=0.05: [0.9875, 0.98333, 0.975, 0.95]
        (corresponds to alpha/4, alpha/3, alpha/2, alpha/1)
    :param n_tests: number of tests in the family.
    :param alpha: family-wise error rate (default 0.05).
    :returns: List of n_tests confidence levels in descending order. The first
        element (most significant test) gets the strictest level
        1 - alpha/n_tests; the last gets the unadjusted level 1 - alpha.
    """
    levels = []
    for k in range(1, n_tests + 1):
        # k-th most significant test gets alpha/(n - k + 1)
        alpha_k = alpha / (n_tests - k + 1)
        level_k = 1.0 - alpha_k
        levels.append(level_k)
    return levels


@dataclasses.dataclass
class _FamilyRow:
    """one in-family decision row: its CI, the normal-approximation se and the ordering statistic"""

    row_index: int
    ci_lo: float
    ci_hi: float
    se: float = 0.0
    z_dist: float = 0.0
    adjusted_level: float = 0.0


def _series_mean(series: pl.Series) -> float:
    """
    the mean of a float series
    :param series: a numeric series (pl.Series)
    :returns: the mean; the cast is because polars types a scalar as a wide union, though a float
        column's mean is a float or None (float)
    """
    return float(cast(float, series.mean()))


def corrected_verdicts(decision_rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """
    Apply Holm-Bonferroni correction to per-arm decision family verdicts.

    Re-evaluates the existing straddle rule at adjusted per-rank CI levels
    across the per-arm test family. The in-family population is rows with
    numeric CIs and verdict in {LM_MEETS_CRITERION, LM_FAILS_CRITERION,
    CI_INDETERMINATE}. Out-of-family rows (NOT_MEASURED, INADMISSIBLE,
    CI_UNAVAILABLE, or missing CIs) pass through with in_family=False.

    Ordering statistic: z-distance of point from threshold, |point - threshold| / se,
    largest (most significant) first. The se is derived from the level-0.95 CI
    via normal approximation: se = (ci_hi - ci_lo) / (2 * z_0.975).

    Adjusted CIs use the same normal approximation: for adjusted level L,
    z_L = norm.ppf((1 + L)/2), adjusted_CI = point +/- z_L * se.

    The straddle rule (reproduced from scripts/analyse_stats.py:2994-3024):
    - If lo_failed != hi_failed (CI endpoints disagree on failure): CI_INDETERMINATE
    - Else if lo_failed != point_failed (point and CI disagree): CI_INDETERMINATE
    - Else: LM_FAILS_CRITERION if failed, LM_MEETS_CRITERION otherwise
    :param decision_rows: list of decision.csv row dicts (must have: property,
        direction, fail_threshold, observed_mean_admissible, ci_lo, ci_hi,
        verdict).
    :param alpha: family-wise error rate (default 0.05).
    :returns: List of dicts, one per input row, with all input fields plus:
        - verdict_corrected: re-evaluated verdict at adjusted level
        - adjusted_level: the confidence level assigned to this test (or None
        if out-of-family)
        - in_family: bool, whether this row received correction
        - method: "normal-approx from level-0.95 CI" for in-family rows
    """
    # Direction comparators (from scripts/analyse_stats.py:1209-1214)
    # Accepts both short codes (gt/gte/lt) and verbose forms
    _DIRECTION_COMPARATORS = {
        "value > threshold means FAIL": lambda obs, thr: obs > thr,
        "value >= threshold means FAIL": lambda obs, thr: obs >= thr,
        "value < threshold means FAIL": lambda obs, thr: obs < thr,
        "gt": lambda obs, thr: obs > thr,
        "gte": lambda obs, thr: obs >= thr,
        "lt": lambda obs, thr: obs < thr,
        "less": lambda obs, thr: obs < thr,  # legacy/test alias
        "greater": lambda obs, thr: obs > thr,  # legacy/test alias
    }

    # Identify in-family rows: numeric CIs and verdict in decision set
    in_family_data: list[_FamilyRow] = []

    for i, row in enumerate(decision_rows):
        ci_lo_str = row.get("ci_lo", "")
        ci_hi_str = row.get("ci_hi", "")
        verdict = row.get("verdict", "")

        # Check if CIs are numeric (explicit None/empty check to handle 0.0)
        try:
            ci_lo = (
                float(ci_lo_str)
                if (ci_lo_str is not None and ci_lo_str != "")
                else None
            )
            ci_hi = (
                float(ci_hi_str)
                if (ci_hi_str is not None and ci_hi_str != "")
                else None
            )
        except (ValueError, TypeError):
            ci_lo = ci_hi = None

        is_decision_verdict = verdict in (
            "LM_MEETS_CRITERION",
            "LM_FAILS_CRITERION",
            "CI_INDETERMINATE",
        )

        if ci_lo is not None and ci_hi is not None and is_decision_verdict:
            in_family_data.append(_FamilyRow(row_index=i, ci_lo=ci_lo, ci_hi=ci_hi))

    # Compute ordering statistic (z-distance) for in-family rows
    z_0_975 = norm.ppf(0.975)  # 1.96 for two-tailed 95% CI

    for data in in_family_data:
        i = data.row_index
        row = decision_rows[i]
        ci_lo = data.ci_lo
        ci_hi = data.ci_hi

        # Parse fields (explicit None/empty check to handle 0.0)
        observed_str = row.get("observed_mean_admissible", "")
        threshold_str = row.get("fail_threshold", "")

        try:
            observed = (
                float(observed_str)
                if (observed_str is not None and observed_str != "")
                else None
            )
            threshold = (
                float(threshold_str)
                if (threshold_str is not None and threshold_str != "")
                else None
            )
        except (ValueError, TypeError):
            observed = threshold = None

        # Compute se via normal approximation
        se = (ci_hi - ci_lo) / (2.0 * z_0_975)

        # z-distance (ordering statistic)
        if observed is not None and threshold is not None and se > 0:
            z_dist = abs(observed - threshold) / se
        else:
            z_dist = 0.0  # fallback for edge cases

        data.se = se
        data.z_dist = z_dist

    # Sort by z-distance descending (most significant first)
    in_family_data.sort(key=lambda d: d.z_dist, reverse=True)

    # Assign adjusted levels
    n_family = len(in_family_data)
    if n_family > 0:
        adjusted_levels = holm_adjusted_levels(n_family, alpha=alpha)
        for k, data in enumerate(in_family_data):
            data.adjusted_level = adjusted_levels[k]
    else:
        adjusted_levels = []

    # Build output rows
    results = []
    for i, row in enumerate(decision_rows):
        output_row = dict(row)  # copy all input fields

        # Check if this row is in-family
        in_family_entry = next((d for d in in_family_data if d.row_index == i), None)

        if in_family_entry is None:
            # Out-of-family: passthrough
            output_row["verdict_corrected"] = row.get("verdict", "")
            output_row["adjusted_level"] = None
            output_row["in_family"] = False
            output_row["method"] = None
        else:
            # In-family: re-evaluate at adjusted level
            output_row["in_family"] = True
            output_row["method"] = "normal-approx from level-0.95 CI"
            output_row["adjusted_level"] = in_family_entry.adjusted_level

            # Parse fields for straddle rule (explicit None/empty check to handle 0.0)
            direction = row.get("direction", "")
            threshold_str = row.get("fail_threshold", "")
            observed_str = row.get("observed_mean_admissible", "")

            try:
                threshold = (
                    float(threshold_str)
                    if (threshold_str is not None and threshold_str != "")
                    else None
                )
                observed = (
                    float(observed_str)
                    if (observed_str is not None and observed_str != "")
                    else None
                )
            except (ValueError, TypeError):
                threshold = observed = None

            ci_lo = in_family_entry.ci_lo
            ci_hi = in_family_entry.ci_hi
            se = in_family_entry.se
            adj_level = in_family_entry.adjusted_level

            # Apply straddle rule with adjusted CI
            if threshold is None or observed is None:
                # Fallback: cannot re-evaluate
                output_row["verdict_corrected"] = row.get("verdict", "")
            elif direction not in _DIRECTION_COMPARATORS:
                # Loud failure on unknown direction
                raise ValueError(f"Unknown direction: {direction}")
            else:
                # Compute adjusted CI, only where there is an observed mean to centre it on
                z_adj = norm.ppf((1.0 + adj_level) / 2.0)
                adj_ci_lo = observed - z_adj * se
                adj_ci_hi = observed + z_adj * se
                comparator = _DIRECTION_COMPARATORS[direction]
                # Straddle rule (from analyse_stats.py:2993-3024)
                point_failed = comparator(observed, threshold)
                lo_failed = comparator(adj_ci_lo, threshold)
                hi_failed = comparator(adj_ci_hi, threshold)

                if lo_failed != hi_failed:
                    # CI straddles threshold
                    output_row["verdict_corrected"] = "CI_INDETERMINATE"
                elif lo_failed != point_failed:
                    # Point and CI disagree
                    output_row["verdict_corrected"] = "CI_INDETERMINATE"
                else:
                    # Agree
                    output_row["verdict_corrected"] = (
                        "LM_FAILS_CRITERION" if point_failed else "LM_MEETS_CRITERION"
                    )

        results.append(output_row)

    return results


def seed_variance_budget(
    df: pl.DataFrame, n_resamples: int = 1000, seed: int = 0
) -> list[dict]:
    """
    Seed-variance decomposition: within-pairing vs between-seed uncertainty.

    The P2 variance budget for any arm frame. For each
    (game, opponent) cell, computes:
    - within_ci: hand-level bootstrap CI (within one policy pairing)
    - between_ci: seed-level replicate CI (between policy pairings)
    - n_effective: the replicate count, which is the seed count (the seed is the unit of
      replication); `n_distinct` reports how many seeds carry a distinct reward series, for
      reporting only and never as a gate
    :param df: logged frame with decision rows (must have: game, opponent, seed,
        hand_idx, reward, prompt columns).
    :param n_resamples: bootstrap replicates for within_ci (default 1000).
    :param seed: RNG seed for bootstrap (default 0).
    :returns: List of dicts, one per (game, opponent), each with:
        - game: game key
        - opponent: opponent key
        - within_ci_lo, within_ci_hi: hand-level bootstrap CI bounds
        - between_ci_lo, between_ci_hi: seed-level replicate CI bounds
        - n_seeds: number of seeds in this cell
        - n_hands: hands per seed (raises if seeds disagree)
        - n_effective: the replicate count, equal to n_seeds
        - n_distinct: seeds with a distinct per-hand reward series (reporting only)
        - within_share: (within_ci_hi - within_ci_lo) / total_width (%)
        - between_share: (between_ci_hi - between_ci_lo) / total_width (%)
        - reason: None for a scored pair; for a pair with no LM rows, why it holds
          no interval (its CIs and shares are NaN and its counts 0)
    """

    # Extract hand-level rewards (one row per (seed, hand_idx))
    def _hand_level_lm_rewards(cell_df: pl.DataFrame) -> pl.DataFrame:
        """One row per (seed, hand_idx) LM hand, matching S2's definition.

        S2 line 175-193: filters to LM rows (prompt non-null), groups by
        (seed, hand_idx), takes first reward per hand.
        """
        lm = cell_df.filter(pl.col("prompt").is_not_null())
        if lm.is_empty():
            return (
                lm.select(["seed", "hand_idx", "reward", "game"])
                if "seed" in lm.columns
                else pl.DataFrame(
                    schema={
                        "seed": pl.Int64,
                        "hand_idx": pl.Int64,
                        "reward": pl.Float64,
                        "game": pl.Utf8,
                    }
                )
            )
        # Capture game value before grouping
        game_val = lm["game"][0] if "game" in lm.columns and lm.height else "cell"
        hand_cols = ["seed", "hand_idx"] if "seed" in lm.columns else ["hand_idx"]
        return (
            lm.group_by(hand_cols)
            .agg(pl.col("reward").first())
            .sort(hand_cols)
            .with_columns(pl.lit(game_val).alias("game"))
        )

    # Within-policy hand CI (S2 line 236-255)
    def _within_ci(hands: pl.DataFrame) -> dict:
        """Hand-level bootstrap CI, matching S2's within_policy_hand_ci."""

        def stat_fn(d: pl.DataFrame) -> dict:
            return {"value": _series_mean(d["reward"]) if d.height else float("nan")}

        return bootstrap_ci(hands, stat_fn, n_resamples=n_resamples, seed=seed)["value"]

    # Between-policy seed CI (S2 line 258-280)
    def _between_ci(hands: pl.DataFrame) -> dict:
        """Seed-replicate CI, matching S2's between_policy_seed_ci."""
        if "seed" not in hands.columns or hands.is_empty():
            return {
                "point": float("nan"),
                "lo": float("nan"),
                "hi": float("nan"),
                "se": float("nan"),
                "n_replicates": 0,
            }

        # Compute per-seed means
        per_seed_means = []
        for s in sorted(hands["seed"].unique().to_list()):
            seed_df = hands.filter(pl.col("seed") == s)
            per_seed_means.append(
                _series_mean(seed_df["reward"]) if seed_df.height else float("nan")
            )

        # replicate_ci over per-seed means
        return replicate_ci(per_seed_means, level=0.95)

    # Effective replicates (S2 line 283-303)
    def _effective_replicates(hands: pl.DataFrame) -> dict:
        """Effective independent replicates, matching S2's cell_effective_replicates."""
        if hands.is_empty() or "seed" not in hands.columns:
            return {"n_nominal": 0, "n_effective": 0, "duplicate_groups": []}
        series_per_shard = []
        for s in sorted(hands["seed"].unique().to_list()):
            sd = hands.filter(pl.col("seed") == s).sort("hand_idx")
            series_per_shard.append(sd["reward"].to_list())
        return effective_replicates(series_per_shard)

    # Hands per seed (S2 line 306-345)
    def _n_hands_per_seed(hands: pl.DataFrame) -> int:
        """Single hand count all seeds share, matching S2's n_hands_per_seed."""
        if hands.is_empty():
            return 0
        if "seed" not in hands.columns:
            return hands.height
        per_seed = hands.group_by("seed").agg(pl.len().alias("n")).sort("seed")
        counts = per_seed["n"].to_list()
        if len(set(counts)) > 1:
            detail = dict(zip(per_seed["seed"].to_list(), counts, strict=True))
            raise ValueError(f"seeds do not share one hand count: {detail}")
        return counts[0]

    # Group by (game, opponent)
    if "game" not in df.columns or "opponent" not in df.columns:
        raise ValueError("df must have 'game' and 'opponent' columns")

    results = []
    for (game, opponent), cell_df in df.group_by(["game", "opponent"]):
        hands = _hand_level_lm_rewards(cell_df)
        if hands.is_empty():
            results.append(
                {
                    "game": game,
                    "opponent": opponent,
                    "within_ci_lo": float("nan"),
                    "within_ci_hi": float("nan"),
                    "between_ci_lo": float("nan"),
                    "between_ci_hi": float("nan"),
                    "n_seeds": 0,
                    "n_hands": 0,
                    "n_effective": 0,
                    "n_distinct": 0,
                    "within_share": float("nan"),
                    "between_share": float("nan"),
                    "reason": "no LM rows (prompt non-null) for this (game, opponent) pair, "
                    "so there are no hands to bootstrap",
                }
            )
            continue

        within = _within_ci(hands)
        between = _between_ci(hands)
        eff = _effective_replicates(hands)

        n_seeds = hands["seed"].n_unique() if "seed" in hands.columns else 1
        n_hands = _n_hands_per_seed(hands)

        # Compute shares
        within_width = within["hi"] - within["lo"]
        between_width = between["hi"] - between["lo"]
        total_width = within_width + between_width

        if total_width > 0:
            within_share = (within_width / total_width) * 100.0
            between_share = (between_width / total_width) * 100.0
        else:
            within_share = between_share = float("nan")

        results.append(
            {
                "game": game,
                "opponent": opponent,
                "within_ci_lo": within["lo"],
                "within_ci_hi": within["hi"],
                "between_ci_lo": between["lo"],
                "between_ci_hi": between["hi"],
                "n_seeds": n_seeds,
                "n_hands": n_hands,
                "n_effective": n_seeds,
                "n_distinct": eff["n_effective"],
                "within_share": within_share,
                "between_share": between_share,
                "reason": None,
            }
        )

    return results
