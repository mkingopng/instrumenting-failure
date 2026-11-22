#!/usr/bin/env python
"""
Write or verify the committed manifest of a scoreboard root.

Usage:
    .venv/bin/python scripts/board_manifest.py --root analysis/<root> --out configs/board_manifests/<root>.json
    .venv/bin/python scripts/board_manifest.py --root analysis/<root> --verify configs/board_manifests/<root>.json

The manifest names every file the scoreboard reads from the root, by path and full sha256 (CRLF read
as LF), so a copy of a root that is not in git can be checked against the repository. Writing is
deterministic: the same root gives the same bytes. `scripts/scoreboard.py --manifest` runs the same
check before it renders.

Exit codes:
    0   manifest written, or the root matches it
    6   the root or the manifest is missing or unreadable
    10  the root differs from the manifest (each difference is printed)
"""

import argparse
import sys
from pathlib import Path

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmopt.analysis.scoreboard import scoreboard_sources as bm

EXIT_OK = 0
EXIT_INPUT = 6
EXIT_DIFFERS = 10


def parse_args(argv=None):
    """
    :param argv: argument list, or None for sys.argv
    :returns: the parsed arguments (argparse.Namespace)
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", required=True, help="The root of scored arm directories."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="Write the manifest of the root to this path.")
    mode.add_argument(
        "--verify", help="Check the root against this committed manifest."
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    write a root's manifest, or check a root against one
    :param argv: argument list, or None for sys.argv
    :returns: the exit code documented in the module docstring
    """
    args = parse_args(argv)
    try:
        if args.out:
            text = bm.dumps(bm.build(args.root))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text)
            print(f"wrote {args.out} ({len(bm.scan(args.root))} files)")
            return EXIT_OK
        problems = bm.verify(args.root, bm.load_manifest(args.verify))
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_INPUT
    for problem in problems:
        print(f"MANIFEST: {problem}", file=sys.stderr)
    return EXIT_DIFFERS if problems else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
