"""
Coverage statistics over repeated simulated arms, and the verdict on them.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import dataclass

#: the nominal level of the intervals under test
LEVEL = 0.95
#: wilson lower bound on the observed coverage must clear this
COVERAGE_FLOOR = 0.90
#: observed coverage above this means the intervals are not informative
COVERAGE_CEILING = 0.995
#: the median half-width must be within this factor of t * sd(arm points)
WIDTH_FACTOR = 2.0


@dataclass(frozen=True)
class Coverage:
    """
    what R simulated arms say about one property's interval
    :param n_arms: arms with a formable interval (int)
    :param covered: arms whose interval contains the planted truth (int)
    :param wilson_lower: lower 95% bound on the coverage rate (float)
    :param bias_z: mean (point - truth) / se over arms, scaled by sqrt(n) (float)
    :param width_ratio: median half-width over its expected size from the spread of the points (float)
    """

    n_arms: int
    covered: int
    wilson_lower: float
    bias_z: float
    width_ratio: float

    @property
    def rate(self) -> float:
        return self.covered / self.n_arms

    def verdict(self) -> list[str]:
        """
        which of the three conditions fail
        :returns: an empty list when the property is covered (list of str)
        """
        failures = []
        if self.wilson_lower < COVERAGE_FLOOR or self.rate > COVERAGE_CEILING:
            failures.append(
                f"coverage {self.rate:.3f} (wilson lower {self.wilson_lower:.3f}, "
                f"floor {COVERAGE_FLOOR}, ceiling {COVERAGE_CEILING})"
            )
        if abs(self.bias_z) > 3.0:
            failures.append(f"bias z {self.bias_z:+.2f} beyond 3")
        if not 1.0 / WIDTH_FACTOR <= self.width_ratio <= WIDTH_FACTOR:
            failures.append(
                f"width ratio {self.width_ratio:.2f} outside 1/{WIDTH_FACTOR}..{WIDTH_FACTOR}"
            )
        return failures


def wilson_lower(covered: int, n: int, z: float = 1.96) -> float:
    """
    the lower end of the Wilson score interval for a proportion
    :param covered: successes (int)
    :param n: trials (int)
    :param z: the normal quantile (float)
    :returns: the lower bound (float)
    """
    p = covered / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n)


def summarise(
    truth: float, points: list[float], los: list[float], his: list[float], t_crit: float
) -> Coverage:
    """
    coverage, bias and width of R intervals against one planted truth
    :param truth: the planted answer, independent of every sampled row (float)
    :param points: each arm's point estimate (list of float)
    :param los: each arm's interval lower end (list of float)
    :param his: each arm's interval upper end (list of float)
    :param t_crit: the critical value the intervals were built with (float)
    :returns: the summary (Coverage)
    """
    n = len(points)
    if n < 2:
        raise ValueError(
            f"summarise needs at least 2 arms to estimate the spread of the points, got {n}"
        )
    covered = sum(1 for lo, hi in zip(los, his, strict=True) if lo <= truth <= hi)
    mean_point = sum(points) / n
    sd_points = math.sqrt(sum((p - mean_point) ** 2 for p in points) / (n - 1))
    if sd_points == 0.0:
        raise ValueError(
            "every arm returned the same point (sd 0), so bias_z and width_ratio are undefined; "
            "a coverage check needs arms that differ"
        )
    half_widths = sorted((hi - lo) / 2 for lo, hi in zip(los, his, strict=True))
    median_half = half_widths[n // 2]
    expected_half = t_crit * sd_points
    return Coverage(
        n_arms=n,
        covered=covered,
        wilson_lower=wilson_lower(covered, n),
        bias_z=(mean_point - truth) / (sd_points / math.sqrt(n)),
        width_ratio=median_half / expected_half,
    )


def require_all_arms(prop: str, formed: int, n_arms: int) -> None:
    """
    refuse a coverage summary computed over fewer arms than were simulated
    :param prop: the property key (str)
    :param formed: how many arms produced an interval for it (int)
    :param n_arms: how many arms were simulated (int)
    :raises ValueError: when any arm had no formable interval, since coverage over the rest is biased toward the arms that did
    """
    if formed != n_arms:
        raise ValueError(
            f"{prop}: {n_arms - formed} of {n_arms} arms had no formable interval, so coverage over "
            "the rest would be computed on a subset the data selected"
        )


def _arm_task(task: tuple) -> dict[str, tuple[float, float, float]]:
    """
    one simulated arm scored by the production cell scorer; a top-level function so a pool can run it
    :param task: (game, game_key, reference_policy, lam, seeds, n_hands, properties) (tuple)
    :returns: `{property: (point, lo, hi)}` for every property with a formable interval (dict)
    """
    os.environ.setdefault("TQDM_DISABLE", "1")
    from scripts.analyse_stats import _cell_replicate_cis
    from tests.coverage_check.arm import simulate_arm

    game, game_key, reference_policy, lam, seeds, n_hands, properties = task
    shards = simulate_arm(game, game_key, reference_policy, lam, seeds, n_hands)
    means: dict = {}
    cis = _cell_replicate_cis(
        shards,
        game_key,
        "cfr_ref",
        {game_key: game},
        {game_key: reference_policy},
        means=means,
    )
    return {
        prop: (means[prop], cis[prop][0], cis[prop][1])
        for prop in properties
        if cis[prop][0] is not None
    }


def collect_arms(
    game,
    game_key: str,
    reference_policy,
    lam: float,
    n_arms: int,
    n_seeds: int,
    n_hands: int,
    base_seed: int,
    properties: tuple[str, ...],
    workers: int = 1,
) -> dict[str, tuple[list[float], list[float], list[float]]]:
    """
    simulate arms and return each property's point, lower and upper interval end per arm

    Every interval comes from `analyse_stats._cell_replicate_cis`, the function `property_table`
    calls, on shards written by the production match runner.
    :param game: the pyspiel game (pyspiel.Game)
    :param game_key: the registry key (str)
    :param reference_policy: the solved reference (object)
    :param lam: the weight on the uniform belief (float)
    :param n_arms: simulated arms (int)
    :param n_seeds: shards per arm (int)
    :param n_hands: hands per shard (int)
    :param base_seed: the first shard seed; arms use disjoint seed ranges (int)
    :param properties: the property keys to collect (tuple of str)
    :param workers: processes to spread the arms over; patches applied in this process reach the
        arms only when this is 1 (int)
    :returns: `{property: (points, los, his)}`, one entry per arm with a formable interval (dict)
    """
    tasks = [
        (
            game,
            game_key,
            reference_policy,
            lam,
            [base_seed + arm * n_seeds + i for i in range(n_seeds)],
            n_hands,
            properties,
        )
        for arm in range(n_arms)
    ]
    if workers > 1:
        # spawn, not fork: a forked child inherits the parent's polars thread pool and deadlocks
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            results = list(pool.map(_arm_task, tasks, chunksize=1))
    else:
        results = [_arm_task(t) for t in tasks]
    return {
        prop: (
            [r[prop][0] for r in results if prop in r],
            [r[prop][1] for r in results if prop in r],
            [r[prop][2] for r in results if prop in r],
        )
        for prop in properties
    }


def run_property_coverage(
    game,
    game_key: str,
    reference_policy,
    lam: float,
    n_arms: int,
    n_seeds: int,
    n_hands: int,
    base_seed: int,
    truths: dict[str, float],
    workers: int = 1,
) -> dict[str, Coverage]:
    """
    simulate arms and measure how often each property's published interval brackets its truth
    :param game: the pyspiel game (pyspiel.Game)
    :param game_key: the registry key (str)
    :param reference_policy: the solved reference (object)
    :param lam: the weight on the uniform belief (float)
    :param n_arms: simulated arms (int)
    :param n_seeds: shards per arm (int)
    :param n_hands: hands per shard (int)
    :param base_seed: the first shard seed (int)
    :param truths: planted truth per property key, computed without any sampled row (dict)
    :param workers: processes to spread the arms over (int)
    :returns: the coverage summary per property key in `truths` (dict)
    """
    from lmopt.scoring.replicate import t_crit_95

    arms = collect_arms(
        game,
        game_key,
        reference_policy,
        lam,
        n_arms,
        n_seeds,
        n_hands,
        base_seed,
        tuple(truths),
        workers,
    )
    for prop in truths:
        require_all_arms(prop, len(arms[prop][0]), n_arms)
    return {
        prop: summarise(truths[prop], *arms[prop], t_crit_95(n_seeds - 1))
        for prop in truths
    }
