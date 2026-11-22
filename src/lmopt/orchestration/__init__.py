"""Match distribution for the LM Optimality Tournament harness.

Ray actors/tasks and backend resource tagging live here. The
unit of distribution is the matchup shard: one (game, opponent, seed) triple
played for N hands, writing exactly one Parquet file.
"""

from lmopt.orchestration.shard import MatchupSpec, run_matchup_shard

__all__ = ["MatchupSpec", "run_matchup_shard"]
