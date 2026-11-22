"""Tripwire: assert that ``lmopt`` resolved inside the checkout being worked in.

Why this exists: when a git worktree's ``.venv`` is a symlink to the primary
checkout's ``.venv``, and ``uv run`` syncs the project environment on every
invocation, ``make test`` in one worktree rewrites the shared
``.venv/lib/python3.12/site-packages/__editable__.lm_optimality_tournament-0.1.0.pth``
to point at that worktree's ``src``. A run started afterwards in a different
worktree then imports the other worktree's source. There is no error, no
warning, and ``git status`` shows nothing, because ``.venv/`` is not tracked.
A contaminated run's output can even match its clean re-derivation, which is
luck about which files the other branch touched.

``scripts/__init__.py`` documents the sibling defect for ``scripts.*``; the
editable ``.pth`` does not pin ``src/lmopt/*`` per checkout when the venv is
shared.

Everything else in this repo avoids the problem (a per-worktree venv, a pinned
``PYTHONPATH``). This module is the only thing that detects
it, so it is deliberately a hard failure at every call site rather than a
warning: a warning on stderr scrolls past unnoticed in a multi-hour run, and a
contaminated run that reports green is a success signal that is not the work.
No silent fallbacks.

Where this module lives, and why it still fires from the wrong tree. It ships
inside the package whose identity it checks, which looks self-referential but
is not, because the comparison is anchored on the CALLER's ``__file__`` rather
than on anything ``lmopt`` knows about itself. Under contamination, the foreign
tree's copy of this module is the one that loads; it then compares the foreign
``lmopt`` directory against the caller's own repository root and fires. The
alternative homes are all worse: ``tests/`` is not importable from ``scripts/``,
a repo-root module is not on ``sys.path`` when a script runs as
``python scripts/foo.py``, and ``scripts/`` resolves across checkouts under the
ambient ``PYTHONPATH`` (the very hazard ``scripts/__init__.py`` documents).
``lmopt`` is the one name all three call sites already import successfully. The
residual gap is honest and small: against a checkout predating this module the
import raises ``ImportError`` instead of the tailored message, which is still
loud, and the ``ImportError`` text names the foreign path itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = [
    "ForeignCheckoutError",
    "assert_lmopt_is_local",
    "check_lmopt_is_local",
    "find_repo_root",
]

#: Directory of the ``lmopt`` package that actually got imported. Resolved once,
#: at import time, so a later ``chdir`` cannot change the answer.
IMPORTED_LMOPT_DIR = Path(__file__).resolve().parent

#: Walk-up marker for the repository root. There is exactly one ``pyproject.toml``
#: in this repo, so the marker is unambiguous.
_ROOT_MARKER = "pyproject.toml"


class ForeignCheckoutError(RuntimeError):
    """``lmopt`` resolved outside the checkout the caller is running from."""


def find_repo_root(anchor: str | Path) -> Path:
    """Return the repository root containing ``anchor``.

    ``anchor`` is the caller's own ``__file__``. Resolving it is load bearing:
    worktrees and symlinked venvs are precisely the confusing case, and an
    unresolved path can compare unequal to the identical resolved one.
    """
    start = Path(anchor).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    raise ForeignCheckoutError(
        f"cannot locate the repository root: no {_ROOT_MARKER} in {start} or any parent. "
        f"assert_lmopt_is_local() must be given the calling file's own __file__."
    )


def _offending_pth_hint() -> str:
    """Name the concrete ``.pth`` file to fix, or describe where to look for it."""
    pattern = "lib/python*/site-packages/__editable__*.pth"
    found = sorted(str(p) for p in Path(sys.prefix).glob(pattern))
    if found:
        return "\n".join(f"    {p}" for p in found)
    return (
        f"    {Path(sys.prefix) / pattern}  (not found; check the venv actually in use)"
    )


def check_lmopt_is_local(
    anchor: str | Path, *, lmopt_dir: str | Path | None = None
) -> str | None:
    """Return an explanatory message if ``lmopt`` is foreign, else ``None``.

    Split out from :func:`assert_lmopt_is_local` so the message can be asserted on
    without catching an exception, and so ``lmopt_dir`` can be driven with a
    deliberately foreign path in tests.
    """
    repo_root = find_repo_root(anchor)
    actual = Path(lmopt_dir if lmopt_dir is not None else IMPORTED_LMOPT_DIR).resolve()
    expected = (repo_root / "src" / "lmopt").resolve()
    if actual == expected:
        return None
    return (
        "CROSS-CHECKOUT IMPORT: `import lmopt` resolved to a DIFFERENT checkout "
        "than the one you are running from.\n"
        f"  running from : {repo_root}\n"
        f"  expected     : {expected}\n"
        f"  actually got : {actual}\n"
        "This is silent by construction: there is no error, no output difference, "
        "and `git status` shows nothing, because .venv/ is not tracked. Any result "
        "produced now describes the OTHER branch's code.\n"
        "The offending editable-install pointer is:\n"
        f"{_offending_pth_hint()}\n"
        "Fix it one of two ways, in order of preference:\n"
        f"  (a) give this checkout its OWN venv, never a symlink: `cd {repo_root} && "
        "make install`;\n"
        f"  (b) pin the interpreter's path explicitly: PYTHONPATH={repo_root}/src:{repo_root} "
        "<python> ...  , which precedes site-packages on sys.path and so outranks the .pth."
    )


def assert_lmopt_is_local(
    anchor: str | Path, *, lmopt_dir: str | Path | None = None
) -> None:
    """Raise :class:`ForeignCheckoutError` if ``lmopt`` came from another checkout.

    Call as ``assert_lmopt_is_local(__file__)`` from an entrypoint. Hard failure is
    deliberate at every call site; see this module's docstring.
    """
    message = check_lmopt_is_local(anchor, lmopt_dir=lmopt_dir)
    if message is not None:
        raise ForeignCheckoutError(message)
