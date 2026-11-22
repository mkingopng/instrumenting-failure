"""
Simulated arms: k seeds of the prescribed reporter, written and read back as real shards.
"""

from __future__ import annotations

import os
import tempfile

import polars as pl

from lmopt.agents.cfr_ref import CFRRefAgent
from lmopt.harness.match_runner import run_match
from lmopt.logging.writer import MatchWriter

from tests.coverage_check.reporter import PrescribedReporter


def simulate_shard(
    game,
    game_key: str,
    reference_policy,
    lam: float,
    seed: int,
    n_hands: int,
    directory: str,
) -> pl.DataFrame:
    """
    play one shard through the production match runner and read its Parquet back
    :param game: the pyspiel game (pyspiel.Game)
    :param game_key: the registry key written to the `game` column (str)
    :param reference_policy: the solved reference, used as the opponent and as the oracle (object)
    :param lam: the weight on the uniform belief (float)
    :param seed: the shard's seed, for the deal, the opponent and the reporter (int)
    :param n_hands: hands to play (int)
    :param directory: where the shard is written (str)
    :returns: the shard as the scorers read it (pl.DataFrame)
    """
    path = os.path.join(directory, f"{game_key}_reporter_vs_cfr_ref_seed{seed}.parquet")
    run_match(
        game,
        PrescribedReporter(reference_policy, lam, seed=seed),
        CFRRefAgent(reference_policy, seed=seed + 10_000),
        CFRRefAgent(reference_policy, seed=0),
        n_hands=n_hands,
        seed=seed,
        writer=MatchWriter(path),
        oracle_policy=reference_policy,
        game_key=game_key,
    )
    return pl.read_parquet(path)


def simulate_arm(
    game, game_key: str, reference_policy, lam: float, seeds: list[int], n_hands: int
) -> list[tuple[int, pl.DataFrame]]:
    """
    one arm: a shard per seed, as the `(seed, frame)` pairs the cell scorers take
    :param game: the pyspiel game (pyspiel.Game)
    :param game_key: the registry key (str)
    :param reference_policy: the solved reference (object)
    :param lam: the weight on the uniform belief (float)
    :param seeds: one seed per shard (list of int)
    :param n_hands: hands per shard (int)
    :returns: `(seed, frame)` per shard (list of tuple)
    """
    with tempfile.TemporaryDirectory() as directory:
        return [
            (
                s,
                simulate_shard(
                    game, game_key, reference_policy, lam, s, n_hands, directory
                ),
            )
            for s in seeds
        ]
