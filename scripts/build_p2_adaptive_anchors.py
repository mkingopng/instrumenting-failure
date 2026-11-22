#!/usr/bin/env python
"""Build the P2 adaptive-reference anchor artefact from a collected adaptive sweep.

Reads the Parquet shards `scripts/run_sweep.py --config configs/p2_adaptive_reference.yaml`
writes (the adaptive reference as agent0, `lmopt.agents.adaptive`, against every opponent,
`n_seeds` seeds per (game, opponent) cell), scores each (game, opponent, seed) shard on its own with
`score_property2(df, actor="adaptive", n_hand_bins=<bins>, hand_span=<the cell's hands>)`, takes the
shrinkage delta the way `scripts/analyse_stats.py` takes it for an LM arm (`_trajectory_delta`: the
first bin's mean reward minus the last bin's; negative is a gap that shrank), and writes the replicate
statistics over seeds to `configs/anchors/p2_adaptive_anchors.json`. That per-cell delta is the P2
ceiling the LM arms are read against (the paper's Fig. 6).

Why a builder and not `analyse_stats.py`: the analysis script selects the scored agent's rows by a
non-null prompt at several sites, so a scripted agent0 scores as nothing there; the P3 and P6 sentinel
anchors take this same route (`scripts/build_p3_sentinel_anchors.py`). The scorer is the unchanged
`score_property2`, so the estimand is the LM arms' estimand (scoring is offline).

Usage:
    .venv/bin/python scripts/build_p2_adaptive_anchors.py \\
        --run-dir data/p2_adaptive_reference --config configs/p2_adaptive_reference.yaml [--bins 2]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lmopt.agents.adaptive import AdaptiveAgent  # noqa: E402
from lmopt.analysis.sentinel_anchors import replicate_stats  # noqa: E402
from lmopt.games.loaders import GAMES  # noqa: E402
from lmopt.games.policy_cache import load_or_solve_cfr  # noqa: E402
from lmopt.harness.match_runner import _agent_id  # noqa: E402
from lmopt.scoring.property2 import score_property2  # noqa: E402

SHARD = re.compile(r"^(?P<game>[a-z0-9_]+?)_lm_vs_(?P<opp>[a-z_]+?)_seed(?P<idx>\d+)\.parquet$")
#: hands per bin the P2 trajectory is licensed at (two bins of 1000 hands for a 2000-hand arm)
DEFAULT_BINS = 2
#: a finer trajectory recorded beside the licensed one, to show where in the run the reference adapts (100-hand bins)
FINE_BINS = 20


def trajectory_delta(means) -> float:
    """
    the P2 shrinkage delta of one trajectory, as `analyse_stats._trajectory_delta` defines it
    :param means: mean reward per bin, first bin first (list | None)
    :returns: first minus last, nan when absent or an endpoint is empty (float)
    """
    if not means or len(means) < 2:
        return float("nan")
    early, late = means[0], means[-1]
    if early is None or late is None or math.isnan(early) or math.isnan(late):
        return float("nan")
    return early - late


def main() -> int:
    """
    score every shard and write the artefact
    :returns: 0 on success, 2 on a missing run dir, 3 on a cell short of its seeds
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS)
    ap.add_argument("--out", type=Path, default=REPO / "configs" / "anchors" / "p2_adaptive_anchors.json")
    a = ap.parse_args()
    if not a.run_dir.is_dir():
        print(f"run dir {a.run_dir} does not exist", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(a.config.read_text())
    n_hands, n_seeds = int(cfg["n_hands"]), int(cfg["n_seeds"])
    actor = _agent_id(AdaptiveAgent(prior_policy=None))
    games, policies = {}, {}
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    for f in sorted(a.run_dir.glob("*.parquet")):
        m = SHARD.match(f.name)
        if not m:
            continue
        game, opp, idx = m["game"], m["opp"], int(m["idx"])
        if game not in games:
            games[game] = GAMES[game]()
            policies[game] = load_or_solve_cfr(games[game], int(cfg["cfr_iters"]), plus=True,
                                               cache_dir=cfg["cfr_cache_dir"], require_hit=True)
        df = pl.read_parquet(f)
        out = score_property2(df, games={game: games[game]}, cfr_policies={game: policies[game]},
                              n_hand_bins=a.bins, actor=actor, hand_span=(0, n_hands - 1))
        mid = next(iter(out["br_gap"]), None)
        traj = out["lm_mean_trajectory"].get(mid, {}).get(game) if mid is not None else None
        delta = trajectory_delta(traj)
        fine = score_property2(df, games={game: games[game]}, cfr_policies={game: policies[game]},
                               n_hand_bins=FINE_BINS, actor=actor, hand_span=(0, n_hands - 1))
        fmid = next(iter(fine["br_gap"]), None)
        fine_traj = fine["lm_mean_trajectory"].get(fmid, {}).get(game) if fmid is not None else None
        gap = out["br_gap"][mid][game] if mid is not None else float("nan")
        cells.setdefault((game, opp), {})[idx] = {
            "delta": delta, "gap": gap, "trajectory": traj, "fine_trajectory": fine_traj,
            "coverage": out["on_support_fraction"].get(mid, {}).get(game) if mid is not None else None,
            "not_applicable": out["trajectory_not_applicable"].get(mid, {}).get(game, "") if mid is not None else "no matchup",
        }
        print(f"{game:10s} {opp:11s} seed{idx:<3d} delta={delta:+.4f} gap={gap:+.4f} traj={[round(x, 3) if x is not None else None for x in (traj or [])]}", flush=True)
    anchors: dict[str, dict] = {}
    short = []
    for (game, opp), by_seed in sorted(cells.items()):
        if len(by_seed) < n_seeds:
            short.append(f"{game}/{opp}: {len(by_seed)} of {n_seeds} seeds")
        vals = [by_seed[i]["delta"] for i in sorted(by_seed) if not math.isnan(by_seed[i]["delta"])]
        st = replicate_stats(vals)
        fines = [by_seed[i]["fine_trajectory"] for i in sorted(by_seed) if by_seed[i]["fine_trajectory"]]
        fine_mean = [sum(t[b] for t in fines if t[b] is not None) / max(1, sum(1 for t in fines if t[b] is not None))
                     for b in range(FINE_BINS)] if fines else None
        anchors.setdefault(game, {})[opp] = {
            **st,
            "fine_bins": FINE_BINS,
            "fine_trajectory_mean": fine_mean,
            "per_seed": vals,
            "seeds": [int(cfg["seed"]) + i for i in sorted(by_seed)],
            "gap_per_seed": [by_seed[i]["gap"] for i in sorted(by_seed)],
            "coverage_per_seed": [by_seed[i]["coverage"] for i in sorted(by_seed)],
        }
    if short:
        print("cells short of their seeds: " + "; ".join(short), file=sys.stderr)
        return 3
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    artefact = {
        "anchors": anchors,
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(a.config.relative_to(REPO)) if a.config.is_absolute() else str(a.config),
            "run_dir": str(a.run_dir),
            "git_sha": sha,
            "estimand": "P2_adaptivity_delta: first-bin mean reward minus last-bin mean reward of the adaptive reference, per seed; negative is a gap that shrank",
            "bins": a.bins,
            "fine_bins": FINE_BINS,
            "n_hands": n_hands,
            "n_seeds": n_seeds,
            "actor": actor,
            "agent": "lmopt.agents.adaptive.AdaptiveAgent (public-prefix opponent model, CFR+ prior, pseudo_count 1.0, best response recomputed every 50 hands)",
            "memory_rung": cfg.get("memory_rung"),
        },
    }
    a.out.write_text(json.dumps(artefact, indent=2) + "\n")
    print(f"wrote {sum(len(v) for v in anchors.values())} cells to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
