#!/usr/bin/env python
"""
Write each scored arm's per-game collapse record beside its `properties.csv`, so the
scoreboard can say when two arms are one observed policy.

Usage:
    .venv/bin/python scripts/arm_collapse_pass.py --analysis-root analysis/recycle-v6-m1
    .venv/bin/python scripts/arm_collapse_pass.py --analysis-root analysis/recycle-v8-all \\
        --offpath-root data/run_experiment/recycle-v8-probes/composite-offpath/seeds42-46

Two arms can play an identical policy on a game, and every table built from their analysis
output then reads as two models independently agreeing when the truth is one policy observed
twice. Detecting that needs the raw shards, which the scoreboard never opens, so this pass
opens them once and leaves a small file the scoreboard reads.

With `--offpath-root` the pass also reads the driven off-path runs, which are what a P6 cell scores,
and writes `collapse_offpath.csv` beside the `collapse.csv` of every arm that has a
`probe_scores.csv`: the same fingerprint and grouping over the driven play, the one action an arm
played at every off-path info-set when it played only one, and the root that was read. Without the
flag no such file is written and the scoreboard says the off-path collapse was not checked.

The fingerprint and the grouping come from `lmopt.analysis.arm_collapse`, the one place the
project computes them; nothing is reimplemented here.

Exit codes:
    0  a collapse file written for every arm
    1  an arm was refused, or its shards could not be read
    3  no arms found in the root
    6  the analysis root is missing or unreadable
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import polars as pl

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmopt.analysis.arm_collapse import (
    FingerprintUnavailable,
    constant_action,
    detect_collapse,
    first_listed_action_share,
    policy_fingerprint,
)
from lmopt.analysis.cross_arm import discover_arm_artifacts
from lmopt.analysis.scoreboard.scoreboard import (
    COLLAPSE_COLUMNS,
    COLLAPSE_FILE,
    COLLAPSE_OFFPATH_COLUMNS,
    COLLAPSE_OFFPATH_FILE,
)
from lmopt.analysis.scoreboard.scoreboard_sources import SCOREBOARD_GAMES

EXIT_OK = 0
EXIT_ARM_PROBLEM = 1
EXIT_NO_ARMS = 3
EXIT_INPUT = 6


def data_dir_of(arm_dir: Path) -> Path:
    """
    :param arm_dir: a scored arm directory
    :returns: the collection directory its run_meta.json names, resolved against the repo
        root when the recorded path is relative
    :raises ValueError: when run_meta.json is missing, unreadable, names no data_dir or
        names a directory that does not exist
    """
    path = arm_dir / "run_meta.json"
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"run_meta.json unreadable ({exc})") from exc
    data_dir = meta.get("data_dir") if isinstance(meta, dict) else None
    if not data_dir:
        raise ValueError("run_meta.json names no data_dir")
    resolved = Path(data_dir)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.is_dir():
        raise ValueError(f"data_dir {data_dir!r} resolves to no directory ({resolved})")
    return resolved


def game_frame(source: Path, game: str):
    """
    :param source: an arm's collection directory
    :param game: a game key
    :returns: every shard of that game as one frame, or None when the arm has no shard for it
    """
    files = sorted(glob.glob(str(source / f"{game}_*.parquet")))
    if not files:
        return None
    return pl.concat(
        [pl.scan_parquet(f) for f in files], how="diagonal_relaxed"
    ).collect()


def measure(sources: dict) -> tuple[dict, list[str]]:
    """
    every arm's fingerprint and first-listed share, per game
    :param sources: {arm name: its collection directory}
    :returns: ({(arm, game): {"fingerprint", "first_listed_share", "n_rows"}}, notes), where
        a note names each (arm, game) left out and why, since an arm silently dropped from
        the check would read as distinct
    """
    measured, notes = {}, []
    for game in SCOREBOARD_GAMES:
        for arm, source in sorted(sources.items()):
            frame = game_frame(source, game)
            if frame is None:
                notes.append(f"{arm} {game}: no shard")
                continue
            try:
                digest = policy_fingerprint(frame)
                share, n_rows = first_listed_action_share(frame)
            except FingerprintUnavailable as exc:
                notes.append(f"{arm} {game}: {exc}")
                continue
            measured[(arm, game)] = {
                "fingerprint": digest,
                "first_listed_share": share,
                "n_rows": n_rows,
            }
    return measured, notes


def collapse_rows(arm: str, measured: dict, arms: list) -> list[dict]:
    """
    one arm's rows of the collapse file
    :param arm: the arm name
    :param measured: the whole fleet's measurements, keyed by (arm, game)
    :param arms: every arm name in the root
    :returns: one row per game the arm was measured on, naming the other arms that share its
        fingerprint on that game
    """
    rows = []
    for game in SCOREBOARD_GAMES:
        mine = measured.get((arm, game))
        if mine is None:
            continue
        prints = {
            other: measured[(other, game)]["fingerprint"]
            for other in arms
            if (other, game) in measured
        }
        others = [
            name
            for group in detect_collapse(prints)
            if arm in group
            for name in group
            if name != arm
        ]
        rows.append(
            {
                "game": game,
                "fingerprint": mine["fingerprint"],
                "collapsed_with": ",".join(sorted(others)),
                "first_listed_share": repr(round(mine["first_listed_share"], 6)),
                "n_rows": mine["n_rows"],
            }
        )
    return rows


def write_collapse(arm_dir: Path, rows: list) -> None:
    """
    :param arm_dir: a scored arm directory
    :param rows: that arm's collapse rows
    :returns: None
    """
    with open(arm_dir / COLLAPSE_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(COLLAPSE_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def offpath_sources(arms: dict, root: Path) -> dict:
    """
    the driven off-path directory of every arm that has a probe_scores.csv
    :param arms: {arm name: its discovered artifacts}
    :param root: a directory holding one subdirectory per model's driven composite
    :returns: {arm name: that model's driven directory}
    :raises ValueError: when the root is not a directory, an arm with a probe_scores.csv
        matches no subdirectory or more than one, since a guess would attach one model's driven
        play to another, or two arms resolve to the same subdirectory, which would report an
        arm as collapsed with its own sibling
    """
    if not root.is_dir():
        raise ValueError(f"--offpath-root {root} is not a directory")
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    sources = {}
    for name in sorted(arms):
        if not (arms[name].path / "probe_scores.csv").exists():
            continue
        matches = [
            p for p in subdirs if name == p.name or name.startswith(p.name + "-m")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{name} has a probe_scores.csv and {len(matches)} matching subdirectories "
                f"under {root} ({[p.name for p in matches]})"
            )
        sources[name] = matches[0]
    shared = sorted(
        {a for a, d in sources.items() for b, e in sources.items() if a != b and d == e}
    )
    if shared:
        raise ValueError(
            f"arms {shared} resolve to one driven directory, so each would be reported as "
            f"collapsed with the other; give each arm its own driven composite"
        )
    return sources


def measure_offpath(sources: dict) -> tuple[dict, list[str]]:
    """
    every probed arm's driven fingerprint and constant action, per game
    :param sources: {arm name: its driven off-path directory}
    :returns: ({(arm, game): {"fingerprint", "constant", "n_rows"}}, notes), where a note names
        each (arm, game) left out and why
    """
    measured, notes = {}, []
    for game in SCOREBOARD_GAMES:
        for arm, source in sorted(sources.items()):
            frame = game_frame(source, game)
            if frame is None:
                notes.append(f"{arm} {game}: no driven shard")
                continue
            try:
                digest = policy_fingerprint(frame)
                constant = constant_action(frame)
            except FingerprintUnavailable as exc:
                notes.append(f"{arm} {game}: {exc}")
                continue
            measured[(arm, game)] = {
                "fingerprint": digest,
                "constant": constant,
                "n_rows": frame.height,
            }
    return measured, notes


def offpath_rows(arm: str, measured: dict, arms: list, root: Path) -> list[dict]:
    """
    one arm's rows of the off-path collapse file
    :param arm: the arm name
    :param measured: every probed arm's driven measurements, keyed by (arm, game)
    :param arms: every probed arm name
    :param root: the off-path root that was read, recorded in every row
    :returns: one row per game the arm was measured on
    """
    rows = []
    for game in SCOREBOARD_GAMES:
        mine = measured.get((arm, game))
        if mine is None:
            continue
        prints = {
            other: measured[(other, game)]["fingerprint"]
            for other in arms
            if (other, game) in measured
        }
        others = [
            name
            for group in detect_collapse(prints)
            if arm in group
            for name in group
            if name != arm
        ]
        constant = mine["constant"]
        rows.append(
            {
                "game": game,
                "fingerprint": mine["fingerprint"],
                "collapsed_with": ",".join(sorted(others)),
                "constant_action": constant.words() if constant else "",
                "n_rows": mine["n_rows"],
                "n_infosets": constant.n_infosets if constant else "",
                "offpath_root": str(root),
            }
        )
    return rows


def write_offpath(arm_dir: Path, rows: list) -> None:
    """
    :param arm_dir: a scored arm directory
    :param rows: that arm's off-path collapse rows
    :returns: None
    """
    with open(
        arm_dir / COLLAPSE_OFFPATH_FILE, "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(COLLAPSE_OFFPATH_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    """
    :param argv: argument list, or None for sys.argv
    :returns: the parsed namespace
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--analysis-root",
        required=True,
        help="Directory of scored arm directories, one per model.",
    )
    parser.add_argument(
        "--offpath-root",
        help="Directory of driven off-path composites, one subdirectory per model; also "
        "writes collapse_offpath.csv for every arm that has a probe_scores.csv.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    measure every arm's play and write its collapse file
    :param argv: argument list, or None for sys.argv
    :returns: the exit code documented in the module docstring
    """
    args = parse_args(argv)
    try:
        arms, problems = discover_arm_artifacts(args.analysis_root)
    except ValueError as exc:
        print(f"analysis root: {exc}", file=sys.stderr)
        return EXIT_INPUT
    for problem in problems:
        print(f"REFUSED: {problem}", file=sys.stderr)
    if problems:
        return EXIT_ARM_PROBLEM
    if not arms:
        print(f"no arms found in {args.analysis_root}", file=sys.stderr)
        return EXIT_NO_ARMS
    sources = {}
    for name in sorted(arms):
        try:
            sources[name] = data_dir_of(arms[name].path)
        except ValueError as exc:
            print(f"REFUSED: {name}: {exc}", file=sys.stderr)
            return EXIT_ARM_PROBLEM
    measured, notes = measure(sources)
    for note in notes:
        print(f"NOT MEASURED: {note}", file=sys.stderr)
    for name in sorted(arms):
        rows = collapse_rows(name, measured, sorted(arms))
        write_collapse(arms[name].path, rows)
        collapsed = [r["game"] for r in rows if r["collapsed_with"]]
        print(
            f"{name}: {len(rows)} game(s) measured, {len(collapsed)} collapsed "
            f"{collapsed}"
        )
    if args.offpath_root:
        root = Path(args.offpath_root)
        if not root.is_absolute():
            root = REPO_ROOT / root
        try:
            off_sources = offpath_sources(arms, root)
        except ValueError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return EXIT_ARM_PROBLEM
        off_measured, off_notes = measure_offpath(off_sources)
        for note in off_notes:
            print(f"NOT MEASURED (off-path): {note}", file=sys.stderr)
        for name in sorted(off_sources):
            rows = offpath_rows(name, off_measured, sorted(off_sources), root)
            write_offpath(arms[name].path, rows)
            constant = [r["game"] for r in rows if r["constant_action"]]
            shared = [r["game"] for r in rows if r["collapsed_with"]]
            print(
                f"{name}: off-path {len(rows)} game(s) measured, constant action on "
                f"{constant}, shared with another arm on {shared}"
            )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
