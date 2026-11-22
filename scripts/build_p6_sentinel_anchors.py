#!/usr/bin/env python
"""Build the P6 sentinel anchor artefact from the collected off-path sentinel runs.

Reads the Parquet shards `scripts/run_sweep.py --config configs/p6_sentinel/<game>.yaml`
writes (a uniform-random agent0 driven into that game's off-path subgame against
`cfr_ref`, `n_seeds` seeds per game), scores each (game, seed) independently with
`lmopt.scoring.property6.score_property6(df, actor="random")` against the trembled
CFR reference, and writes the replicate statistics
`lmopt.analysis.sentinel_anchors.build_artefact` computes to
`configs/anchors/p6_sentinel_anchors.json`.

One run directory per game, because a sweep carries one `drive_openings` list and
one `seat_mode` and both differ by game. The opening and the seat are read from
each run's own `run_meta.json`, never from the job script that launched it: the
artefact must record the subgame that was actually measured, and a test asserts the
five configs still agree with `scripts/collection_job.py`'s `OFFPATH_SWEEPS`.

The reference is loaded with `require_hit=True`. This script never solves.

This script is the only place that touches the filesystem for this artefact;
`sentinel_anchors.py`'s functions are pure (scoring is offline).

Usage:
    .venv/bin/python scripts/build_p6_sentinel_anchors.py \\
        --run-root data/p6_sentinel --config-dir configs/p6_sentinel
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from lmopt._import_guard import assert_lmopt_is_local
from lmopt.analysis.condition import (
    ResolvedConfigShapeError,
    parse_drive_openings_field,
    parse_seat_mode_field,
)
from lmopt.analysis.scoring_vocabulary import parse_shard_name
from lmopt.analysis.sentinel_anchors import P6_ESTIMAND, SAMPLED_BASIS, build_artefact
from lmopt.provenance.versions import LIBRARY_VERSION_KEYS
from lmopt.scoring.property6 import (
    DEFAULT_TREMBLE_EPS,
    offpath_reference,
    score_property6,
)

assert_lmopt_is_local(__file__)

#: The one opponent the off-path run faces, so the one opponent key the anchors
#: carry. The artefact is keyed (game, opponent) anyway, so a second opponent can
#: be added later without a schema change.
OPPONENT = "cfr_ref"

#: A run directory named on the command line does not exist, or holds no
#: `run_meta.json`.
RUN_DIR_MISSING_EXIT_CODE = 2
#: A game the run root covers is missing at least one of its run's seeds --
#: either no shard at all, or a shard whose `seed` column never carries an
#: expected value.
MISSING_SEED_EXIT_CODE = 3
#: A (game, seed) scored to a non-finite total variation, or to no off-path
#: info-set at all -- an anchor built from either would silently describe a
#: different measurement than "random play's off-path distance".
SCORE_INVALID_EXIT_CODE = 4
#: A run's own `run_meta.json` carries no `environment` block, so the OpenSpiel
#: and numpy versions it was COLLECTED under are unrecorded. They are not taken
#: from this machine instead: the artefact's provenance is what every scored run
#: is later compared against, and a version read here would describe whichever
#: box happened to build the artefact.
RUN_ENVIRONMENT_MISSING_EXIT_CODE = 5
#: A run's own `run_meta.json` records no `drive_openings` or no `seat_mode`, so
#: which subgame the distance was measured inside is unknown. Those two fields
#: select the subgame, and an anchor that cannot name its subgame is an anchor
#: for nothing in particular.
RUN_SUBGAME_MISSING_EXIT_CODE = 6


def parse_args(argv=None) -> argparse.Namespace:
    """
    the command-line arguments this script accepts
    :param argv: argument list, `None` reads `sys.argv` (list[str] | None)
    :returns: parsed arguments (argparse.Namespace)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        required=True,
        help="directory holding one subdirectory per game, each one an off-path "
        "sentinel sweep's own output directory",
    )
    parser.add_argument(
        "--config-dir",
        required=True,
        help="the directory of YAML configs the sentinel runs were launched from, "
        "recorded per game in the artefact's provenance so a reader can find "
        "the exact run again",
    )
    parser.add_argument(
        "--out",
        default="configs/anchors/p6_sentinel_anchors.json",
        help="path the anchor artefact is written to",
    )
    parser.add_argument(
        "--cfr-cache-dir",
        default="data/cfr_policies",
        help="cache directory the CFR reference is loaded from; it is never solved",
    )
    return parser.parse_args(argv)


def _discover_shards(run_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """
    every off-path shard Parquet in `run_dir`, grouped by the (game, opponent)
    its filename names
    :param run_dir: one game's sentinel run output directory (Path)
    :returns: `{(game, opponent): [path, ...]}`, one entry per file matching
        `parse_shard_name` with an arm component, which an off-path sweep always writes; a non-matching file is silently excluded, the same
        convention `run_sweep.py`'s own readers use for a file that is not one
        of its shards (dict[tuple[str, str], list[Path]])
    """
    by_cell: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(run_dir.glob("*.parquet")):
        parsed = parse_shard_name(path.name)
        if parsed is None or parsed.arm is None:
            continue
        key = (parsed.game, parsed.opponent)
        by_cell.setdefault(key, []).append(path)
    return by_cell


def _read_run(run_dir: Path) -> tuple[dict | None, str | None]:
    """
    one game's run metadata, or why it cannot be used
    :param run_dir: the game's sentinel run output directory (Path)
    :returns: (run_meta, None) when the directory holds a readable
        `run_meta.json`, else (None, reason) (tuple[dict | None, str | None])
    """
    if not run_dir.is_dir():
        return None, f"run dir not found: {run_dir}"
    run_meta_path = run_dir / "run_meta.json"
    if not run_meta_path.is_file():
        return None, f"{run_meta_path} not found; not a run_sweep.py output dir"
    return json.loads(run_meta_path.read_text()), None


def main(argv=None) -> int:
    """
    build and write the P6 sentinel anchor artefact
    :param argv: argument list, `None` reads `sys.argv` (list[str] | None)
    :returns: one of the module-level exit codes, 0 on success (int)
    """
    from lmopt.games.loaders import load_game

    args = parse_args(argv)
    run_root = Path(args.run_root)
    if not run_root.is_dir():
        print(f"ABORT: sentinel run root not found: {run_root}", file=sys.stderr)
        return RUN_DIR_MISSING_EXIT_CODE

    games = sorted(p.name for p in run_root.iterdir() if p.is_dir())
    if not games:
        print(f"ABORT: {run_root} holds no per-game run directory", file=sys.stderr)
        return RUN_DIR_MISSING_EXIT_CODE

    metas: dict[str, dict] = {}
    unreadable: list[str] = []
    for game in games:
        run_meta, problem = _read_run(run_root / game)
        if run_meta is None:
            unreadable.append(problem or f"{run_root / game}: no run metadata")
            continue
        metas[game] = run_meta
    if unreadable:
        print(
            "ABORT: one or more sentinel run directories are unreadable:",
            file=sys.stderr,
        )
        for line in unreadable:
            print(f"  {line}", file=sys.stderr)
        return RUN_DIR_MISSING_EXIT_CODE

    # Read from the runs, never from this process. The artefact's provenance is
    # the thing every later run is compared against, so a version sampled here
    # would assert that the collecting box matched the box that built the
    # artefact, which nobody measured.
    unrecorded: list[str] = []
    for game, run_meta in metas.items():
        environment = run_meta.get("environment") or {}
        absent = [k for k in LIBRARY_VERSION_KEYS if k not in environment]
        if absent:
            unrecorded.append(f"{game}: no {', '.join(absent)} under 'environment'")
    if unrecorded:
        print(
            "ABORT: the versions these runs were COLLECTED under are unknown, so "
            "re-collect them with a harness that records the environment block "
            "rather than stamping this machine's versions on the anchors:",
            file=sys.stderr,
        )
        for line in unrecorded:
            print(f"  {line}", file=sys.stderr)
        return RUN_ENVIRONMENT_MISSING_EXIT_CODE

    # One artefact carries one run-level provenance block, so the five runs must
    # agree on the fields that block holds. They disagree only when the games
    # were collected on different boxes or against different references, and an
    # anchor set whose provenance describes one of those runs would silently be
    # read as describing all five.
    shared = {
        field: {
            game: (
                meta["resolved_config"].get(field)
                if field == "cfr_iters"
                else meta["environment"][field]
            )
            for game, meta in metas.items()
        }
        for field in ("cfr_iters", *LIBRARY_VERSION_KEYS)
    }
    disagreeing = [
        f"{field}: " + ", ".join(f"{g}={v!r}" for g, v in sorted(by_game.items()))
        for field, by_game in shared.items()
        if len(set(map(repr, by_game.values()))) > 1
    ]
    if disagreeing:
        print(
            "ABORT: the sentinel runs disagree on run-level provenance, so one "
            "provenance block cannot describe them all:",
            file=sys.stderr,
        )
        for line in disagreeing:
            print(f"  {line}", file=sys.stderr)
        return RUN_ENVIRONMENT_MISSING_EXIT_CODE

    # Each game's own single run is never a game-disjoint composite, so
    # `drive_openings`/`seat_mode` here are shape-checked and (unlike a
    # composite reader) a per-game dict is itself the wrong shape -- refused
    # the same way a bare string or any other unexpected shape is, rather
    # than `list()`-shredded into the committed artefact
    # (`per_seed_by_cell[...]`, below) with no error.
    no_subgame: list[str] = []
    subgame_drive_openings: dict[str, list[int]] = {}
    subgame_seat_mode: dict[str, str] = {}
    for game, run_meta in metas.items():
        resolved = run_meta["resolved_config"]
        try:
            drive = parse_drive_openings_field(
                resolved.get("drive_openings"), context=game
            )
            seat_mode = parse_seat_mode_field(resolved.get("seat_mode"), context=game)
        except ResolvedConfigShapeError as exc:
            no_subgame.append(str(exc))
            continue
        if isinstance(drive, dict) or isinstance(seat_mode, dict):
            no_subgame.append(
                f"{game}: resolved_config records a per-game drive_openings/seat_mode, but "
                "this script reads one game's own single run at a time, never a composite"
            )
            continue
        if not drive:
            no_subgame.append(f"{game}: run_meta records no drive_openings")
        if not seat_mode:
            no_subgame.append(f"{game}: run_meta records no seat_mode")
        if drive and seat_mode:
            subgame_drive_openings[game] = drive
            subgame_seat_mode[game] = seat_mode
    if no_subgame:
        print(
            "ABORT: a sentinel run cannot name the subgame it measured:",
            file=sys.stderr,
        )
        for line in no_subgame:
            print(f"  {line}", file=sys.stderr)
        return RUN_SUBGAME_MISSING_EXIT_CODE

    expected_seeds: dict[str, list[int]] = {}
    shards: dict[str, list[Path]] = {}
    missing: list[str] = []
    for game, run_meta in metas.items():
        resolved = run_meta["resolved_config"]
        base_seed, n_seeds = resolved["seed"], resolved["n_seeds"]
        expected_seeds[game] = [base_seed + i for i in range(n_seeds)]
        by_cell = _discover_shards(run_root / game)
        cell = by_cell.get((game, OPPONENT))
        if not cell:
            missing.append(f"({game}, {OPPONENT}): no shard files found")
            continue
        shards[game] = cell
        present = set(
            pl.concat([pl.read_parquet(p) for p in cell])["seed"].unique().to_list()
        )
        absent_seeds = [s for s in expected_seeds[game] if s not in present]
        if absent_seeds:
            missing.append(f"({game}, {OPPONENT}): missing seed(s) {absent_seeds}")
    if missing:
        print(
            "ABORT: the sentinel runs are missing seeds for one or more games:",
            file=sys.stderr,
        )
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        return MISSING_SEED_EXIT_CODE

    per_seed_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    invalid: list[str] = []
    for game in sorted(metas):
        resolved = metas[game]["resolved_config"]
        game_obj = load_game(game)
        reference = offpath_reference(
            game_obj,
            resolved["cfr_iters"],
            cache_dir=args.cfr_cache_dir,
            require_hit=True,
        )
        cell_df = pl.concat([pl.read_parquet(p) for p in shards[game]])
        seeds: list[int] = []
        values: list[float] = []
        for seed in expected_seeds[game]:
            out = score_property6(
                cell_df.filter(pl.col("seed") == seed),
                reference,
                game=game_obj,
                actor="random",
            )
            if not out["n_offpath_infosets"]:
                invalid.append(
                    f"(game={game}, seed={seed}): no off-path info-set was scored"
                )
                continue
            if math.isnan(out["tv_mean"]):
                invalid.append(f"(game={game}, seed={seed}): tv_mean is NaN")
                continue
            seeds.append(seed)
            values.append(out["tv_mean"])
        per_seed_by_cell[(game, OPPONENT)] = {
            "seeds": seeds,
            "values": values,
            # Per game, not per run: the opening and the seat select the subgame
            # the distance was taken inside, and they differ by game, so
            # `anchor_for` reads them off the cell rather than off one run-level
            # value that would have to stand for all five. Already shape-checked
            # above: these are the validated values, not a fresh re-read of
            # `resolved`, so a bad shape aborted before this write is ever reached.
            "drive_openings": subgame_drive_openings[game],
            "seat_mode": subgame_seat_mode[game],
            "config_path": str(Path(args.config_dir) / f"{game}.yaml"),
            "run_dir": str(run_root / game),
            "git_sha": metas[game].get("git_sha"),
        }
    if invalid:
        print("ABORT: one or more (game, seed) cells scored invalid:", file=sys.stderr)
        for line in invalid:
            print(f"  {line}", file=sys.stderr)
        return SCORE_INVALID_EXIT_CODE

    first = metas[sorted(metas)[0]]
    provenance = {
        "config_dir": args.config_dir,
        "run_root": str(run_root),
        "games": sorted(metas),
        "opponent": OPPONENT,
        "n_hands": {g: metas[g]["resolved_config"]["n_hands"] for g in sorted(metas)},
        "seeds": {g: expected_seeds[g] for g in sorted(metas)},
        "cfr_iters": first["resolved_config"]["cfr_iters"],
        "tremble_eps": DEFAULT_TREMBLE_EPS,
        "estimand": P6_ESTIMAND,
        # What these anchors measure, stated on the artefact: they are distances
        # between sampled actions and the reference. The distribution basis needs no
        # collected anchor at all, and the consumer refuses an artefact that does not
        # say which of the two it holds.
        "basis": SAMPLED_BASIS,
        **{k: first["environment"][k] for k in LIBRARY_VERSION_KEYS},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    artefact = build_artefact(per_seed_by_cell, provenance)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artefact, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out_path} ({len(per_seed_by_cell)} anchors, one per game against "
        f"{OPPONENT})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
