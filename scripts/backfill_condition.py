#!/usr/bin/env python
"""
Add the `condition` block to analysis output written before the scoring entrypoint
recorded one.

Usage:
    .venv/bin/python scripts/backfill_condition.py --analysis-root analysis/recycle-v6-m1

The block says which condition an arm's tables analysed: model, memory level, hand count,
seed count and probe role, each copied from the collection run_meta that the analysis
run_meta's own `data_dir` names. It cannot state the prompt schema version, which only the
logged rows carry, so a backfilled block leaves that field None. Only the arms scored before
the writer existed need this;
an arm that already records a block is left exactly as it is, because overwriting one would
let a backfill quietly restate what the scoring run itself recorded. `--update-backfilled`
lifts that for blocks this pass wrote before, so a field added later reaches them; a block
whose `source` is the scoring entrypoint is never rewritten, flag or no flag.

Exit codes:
    0  every arm now records a condition (written here, or already present)
    1  an arm was refused: its collection run_meta is missing or unreadable, or its analysis
       run_meta is not a JSON object or holds a condition block that cannot be read and this
       pass may not rewrite
    3  no arms found in the root
    6  the analysis root is missing or unreadable
"""

import argparse
import json
import sys
from pathlib import Path

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmopt.analysis.condition import (
    CONDITION_KEY,
    ConditionUnavailable,
    condition_block,
    condition_of,
    is_backfilled,
    load_collection_meta,
)
from lmopt.analysis.cross_arm import discover_arm_artifacts

EXIT_OK = 0
EXIT_ARM_PROBLEM = 1
EXIT_NO_ARMS = 3
EXIT_INPUT = 6


def backfill(arm_dir: Path, repo_root: Path, update: bool = False) -> str:
    """
    write one arm's condition block, or say why nothing was written
    :param arm_dir: a scored arm directory
    :param repo_root: the root a relative `data_dir` resolves against
    :param update: whether a block this same pass wrote before may be rewritten; a block
        the scoring entrypoint wrote is left alone whatever this says
    :returns: a one-line report of what happened
    :raises ConditionUnavailable: when the arm names no data_dir, or its collection
        run_meta is missing or unreadable, or the arm's own run_meta is not a JSON object, or
        it holds a condition block that cannot be read and this pass may not rewrite
    """
    meta_path = arm_dir / "run_meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        raise ConditionUnavailable(
            f"{meta_path} is missing or unreadable ({exc})"
        ) from exc
    if not isinstance(meta, dict):
        raise ConditionUnavailable(f"{meta_path} is not a JSON object")
    existing = meta.get(CONDITION_KEY)
    rewritable = update and is_backfilled(existing)
    if condition_of(meta) is not None and not rewritable:
        return f"{arm_dir.name}: already records a condition, left alone"
    if existing is not None and not rewritable:
        remedy = (
            "re-run with --update-backfilled"
            if is_backfilled(existing)
            else "re-score the arm, since a block the scoring run wrote is never rewritten here"
        )
        source = existing.get("source") if isinstance(existing, dict) else None
        raise ConditionUnavailable(
            f"{meta_path} records a condition block (source {source!r}) that cannot be "
            f"read (not an object, or missing a required field), and this pass may not "
            f"rewrite it; {remedy}"
        )
    data_dir = meta.get("data_dir")
    if not data_dir:
        raise ConditionUnavailable(f"{meta_path} names no data_dir")
    collection = load_collection_meta(data_dir, repo_root=repo_root)
    source = f"backfilled from {data_dir}/run_meta.json"
    meta[CONDITION_KEY] = condition_block(collection, source)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    block = meta[CONDITION_KEY]
    return (
        f"{arm_dir.name}: wrote condition "
        f"{block['memory_rung']} {block['n_hands']} hands "
        f"probe {block['probe_type']}"
    )


def parse_args(argv=None):
    """
    :param argv: argument list, or None for sys.argv
    :returns: the parsed namespace
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--analysis-root",
        required=True,
        help="Directory of scored arm directories, one per condition.",
    )
    parser.add_argument(
        "--update-backfilled",
        action="store_true",
        help="Rewrite a block this pass wrote before, so a new field "
        "reaches it. A block the scoring run wrote is never touched.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    backfill every arm under a root
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
    for name in sorted(arms):
        try:
            print(backfill(arms[name].path, REPO_ROOT, args.update_backfilled))
        except ConditionUnavailable as exc:
            print(f"REFUSED: {name}: {exc}", file=sys.stderr)
            return EXIT_ARM_PROBLEM
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
