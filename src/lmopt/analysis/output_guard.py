"""Shared refuse-on-overwrite guard for analysis entrypoints' `--out` directories.

Not scoped to git-tracked content: generated output sits uncommitted between generation and
review, so a tracked-only check would miss a same-day re-run into real prior output.
"""

from __future__ import annotations

import sys
from pathlib import Path


def refuse_non_empty_out_dir(
    out_dir: Path, allow_overwrite: bool
) -> bool:
    """True if the caller should refuse (the refusal message is already
    printed to stderr); False if the caller should proceed.

    Refuses when `out_dir` exists and is not a directory, whatever
    `allow_overwrite` says, and when it exists and is non-empty, unless
    `allow_overwrite` is set. The caller is responsible for acting on the return value (a
    `return 1` in the three current call sites) -- this function never exits
    or raises, matching each site's own existing exit-code contract, which
    differs slightly across the three (return 1 vs sys.exit) and is not this
    function's business to unify.
    """
    if out_dir.exists() and not out_dir.is_dir():
        print(
            f"ERROR: --out {out_dir} exists and is not a directory. Refusing to write "
            f"output there; pass a directory path.",
            file=sys.stderr,
        )
        return True
    if out_dir.exists() and any(out_dir.iterdir()) and not allow_overwrite:
        print(
            f"ERROR: --out {out_dir} already exists and is non-empty. "
            f"Refusing to overwrite -- this may silently destroy a prior "
            f"run's output at the same path. "
            f"Pass --allow-overwrite to proceed anyway.",
            file=sys.stderr,
        )
        return True
    return False
