#!/usr/bin/env python
"""Build the P3 sentinel anchor artefact from a collected sentinel run.

Reads the Parquet shards `scripts/run_sweep.py --config configs/p3_sentinel.yaml`
writes (a uniform-random agent0 against every opponent, `n_seeds` seeds per
(game, opponent) cell), scores each (game, opponent, seed) independently with
`lmopt.scoring.property3.score_property3(df, actor="random")`, and writes the
replicate statistics `lmopt.analysis.sentinel_anchors.build_artefact` computes
to `configs/anchors/p3_sentinel_anchors.json`.

This script is the only place that touches the filesystem for this artefact;
`sentinel_anchors.py`'s functions are pure (scoring is offline).

Usage:
    .venv/bin/python scripts/build_p3_sentinel_anchors.py \\
        --run-dir data/p3_sentinel --config configs/p3_sentinel.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from lmopt._import_guard import assert_lmopt_is_local
from lmopt.analysis.condition import (
    ResolvedConfigShapeError,
    parse_seat_mode_field,
)
from lmopt.analysis.scoring_vocabulary import parse_shard_name
from lmopt.analysis.sentinel_anchors import build_artefact
from lmopt.provenance.versions import LIBRARY_VERSION_KEYS
from lmopt.scoring.property3 import score_property3

assert_lmopt_is_local(__file__)

#: The sentinel run directory named on the command line does not exist.
RUN_DIR_MISSING_EXIT_CODE = 2
#: A (game, opponent) cell the run's own config declares is missing at least
#: one of the run's seeds -- either no shard at all, or a shard whose `seed`
#: column never carries an expected value.
MISSING_SEED_EXIT_CODE = 3
#: A (game, opponent, seed) scored to a non-finite regret, or to reference
#: coverage below 1.0 -- a sentinel anchor built from either would silently
#: describe a different measurement than "random play's regret".
SCORE_INVALID_EXIT_CODE = 4
#: The sentinel run's own `run_meta.json` carries no `environment` block, so
#: the OpenSpiel and numpy versions it was COLLECTED under are unrecorded.
#: They are not taken from this machine instead: the artefact's provenance is
#: what every scored run is later compared against, and a version read here
#: would describe whichever box happened to build the artefact. A run
#: collected before `write_run_meta` recorded the block must be re-collected.
RUN_ENVIRONMENT_MISSING_EXIT_CODE = 5
#: The sentinel run's `resolved_config.seat_mode` is present but not the shape
#: this script expects: a bare scalar. A P3 sentinel run is one run, never a
#: game-disjoint composite, so a per-game dict here -- or any other
#: unexpected shape -- is itself wrong, not a value to write into the
#: committed artefact unchecked.
RUN_PROVENANCE_SHAPE_EXIT_CODE = 6


def parse_args(argv=None) -> argparse.Namespace:
    """
    the command-line arguments this script accepts
    :param argv: argument list, `None` reads `sys.argv` (list[str] | None)
    :returns: parsed arguments (argparse.Namespace)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="directory the sentinel sweep wrote its shards and run_meta.json to",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="the YAML config the sentinel run was launched from, recorded in "
        "the artefact's provenance so a reader can find the exact run again",
    )
    parser.add_argument(
        "--out",
        default="configs/anchors/p3_sentinel_anchors.json",
        help="path the anchor artefact is written to",
    )
    return parser.parse_args(argv)


def _discover_shards(run_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """
    every shard Parquet in `run_dir`, grouped by the (game, opponent) its
    filename names
    :param run_dir: the sentinel run's own output directory (Path)
    :returns: `{(game, opponent): [path, ...]}`, one entry per file matching
        `parse_shard_name` with no arm component; a non-matching file is silently excluded, the same
        convention `run_sweep.py`'s own readers use for a file that is not
        one of its shards (dict[tuple[str, str], list[Path]])
    """
    by_cell: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(run_dir.glob("*.parquet")):
        parsed = parse_shard_name(path.name)
        if parsed is None or parsed.arm is not None:
            continue
        key = (parsed.game, parsed.opponent)
        by_cell.setdefault(key, []).append(path)
    return by_cell


def main(argv=None) -> int:
    """
    build and write the P3 sentinel anchor artefact
    :param argv: argument list, `None` reads `sys.argv` (list[str] | None)
    :returns: one of the module-level exit codes, 0 on success (int)
    """
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ABORT: sentinel run dir not found: {run_dir}", file=sys.stderr)
        return RUN_DIR_MISSING_EXIT_CODE

    run_meta_path = run_dir / "run_meta.json"
    if not run_meta_path.is_file():
        print(
            f"ABORT: {run_meta_path} not found; not a run_sweep.py output dir",
            file=sys.stderr,
        )
        return RUN_DIR_MISSING_EXIT_CODE

    run_meta = json.loads(run_meta_path.read_text())
    # Read from the RUN, never from this process. The artefact's provenance is
    # the thing every later run is compared against, so a version sampled here
    # would assert that the collecting box matched the box that built the
    # artefact, which nobody measured.
    environment = run_meta.get("environment") or {}
    unrecorded = [k for k in LIBRARY_VERSION_KEYS if k not in environment]
    if unrecorded:
        print(
            f"ABORT: {run_meta_path} records no {', '.join(unrecorded)} under "
            "'environment', so the versions this run was COLLECTED under are "
            "unknown; re-collect the sentinel run with a harness that records "
            "them rather than stamping this machine's versions on the anchors",
            file=sys.stderr,
        )
        return RUN_ENVIRONMENT_MISSING_EXIT_CODE

    resolved = run_meta["resolved_config"]
    games: list[str] = resolved["games"]
    opponents: list[str] = resolved["opponents"]
    n_seeds: int = resolved["n_seeds"]
    base_seed: int = resolved["seed"]
    expected_seeds = [base_seed + i for i in range(n_seeds)]

    by_cell = _discover_shards(run_dir)

    missing: list[str] = []
    for game in games:
        for opponent in opponents:
            shards = by_cell.get((game, opponent))
            if not shards:
                missing.append(f"({game}, {opponent}): no shard files found")
                continue
            cell_df = pl.concat([pl.read_parquet(p) for p in shards])
            seeds_present = set(cell_df["seed"].unique().to_list())
            absent = [s for s in expected_seeds if s not in seeds_present]
            if absent:
                missing.append(f"({game}, {opponent}): missing seed(s) {absent}")
    if missing:
        print(
            "ABORT: sentinel run is missing seeds for one or more cells:",
            file=sys.stderr,
        )
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        return MISSING_SEED_EXIT_CODE

    per_seed_by_cell: dict[tuple[str, str], dict[str, list]] = {}
    invalid: list[str] = []
    for game in games:
        for opponent in opponents:
            cell_df = pl.concat([pl.read_parquet(p) for p in by_cell[(game, opponent)]])
            seeds: list[int] = []
            values: list[float] = []
            for seed in expected_seeds:
                seed_df = cell_df.filter(pl.col("seed") == seed)
                result = score_property3(seed_df, actor="random")
                score = result["mean_kl"].get(game)
                if score is None:
                    invalid.append(
                        f"(game={game}, opponent={opponent}, seed={seed}): "
                        "score_property3 returned no entry for this game"
                    )
                    continue
                if not math.isfinite(score.value):
                    invalid.append(
                        f"(game={game}, opponent={opponent}, seed={seed}): "
                        f"regret is not finite ({score.value})"
                    )
                    continue
                if score.on_support_fraction < 1.0:
                    invalid.append(
                        f"(game={game}, opponent={opponent}, seed={seed}): "
                        f"reference coverage {score.on_support_fraction} < 1.0"
                    )
                    continue
                seeds.append(seed)
                values.append(score.value)
            per_seed_by_cell[(game, opponent)] = {"seeds": seeds, "values": values}
    if invalid:
        print(
            "ABORT: one or more (game, opponent, seed) cells scored invalid:",
            file=sys.stderr,
        )
        for line in invalid:
            print(f"  {line}", file=sys.stderr)
        return SCORE_INVALID_EXIT_CODE

    # Shape-checked before it is written into the committed artefact; a
    # per-game dict, an absent value, or any other unexpected shape is
    # refused by name rather than baked into a long-lived JSON every future
    # run is checked against. `.get()` does not raise on a missing key, so
    # absence is refused explicitly here rather than becoming a `None` in
    # the artefact.
    try:
        seat_mode = parse_seat_mode_field(
            resolved.get("seat_mode"), context=str(run_meta_path)
        )
        if seat_mode is None:
            raise ResolvedConfigShapeError(
                f"{run_meta_path}: resolved_config records no seat_mode, so this "
                "artefact's provenance cannot state one"
            )
        if isinstance(seat_mode, dict):
            raise ResolvedConfigShapeError(
                f"{run_meta_path}: resolved_config['seat_mode'] is a per-game record "
                f"{seat_mode!r}, but a P3 sentinel run is one run, never a game-disjoint "
                "composite, and this artefact's provenance records one seat mode"
            )
    except ResolvedConfigShapeError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return RUN_PROVENANCE_SHAPE_EXIT_CODE

    provenance = {
        "git_sha": run_meta.get("git_sha"),
        "config_path": args.config,
        "run_dir": str(run_dir),
        "n_hands": resolved["n_hands"],
        "seeds": expected_seeds,
        "cfr_iters": resolved["cfr_iters"],
        "rl_iterations": resolved["rl_iterations"],
        "seat_mode": seat_mode,
        **{k: environment[k] for k in LIBRARY_VERSION_KEYS},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    artefact = build_artefact(per_seed_by_cell, provenance)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artefact, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out_path} ({len(games)} games x {len(opponents)} opponents "
        f"= {len(per_seed_by_cell)} anchors, {n_seeds} seeds each)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
