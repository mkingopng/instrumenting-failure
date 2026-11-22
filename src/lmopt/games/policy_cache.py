"""Cache-once-freeze for the CFR reference and (via helpers) the RL policy.

design.md §9 M6 "FIRST TASK": solve/train each game's policy **once**, persist it,
and load it thereafter — solving/training only on a cache miss. This turns a
~25-min-to-2-hour per-run setup (Liar's Dice CFR+ ~23 min, XFP ~110 min, re-solved
on every ``run_sweep``/``analyse`` invocation today) into a one-time cost, and
makes the "RL agent is frozen after training" [INVARIANT] literally true.

**What is cached, and why it is exact.** For CFR we persist OpenSpiel's own
``TabularPolicy.action_probability_array`` — the array produced by
``policy.tabular_policy_from_callable(game, cfr_average_policy)``, whose
``state_lookup`` enumerates *every* info-set of the game by construction. On load
we rebuild a fresh ``TabularPolicy(game)`` (which regenerates the identical
``state_lookup`` deterministically from the game) and assign the saved array. This
reproduces ``action_probabilities`` **byte-identically at every info-set including
off-path / zero-reach nodes** (verified: full-tree walk, max|Δ|=0.0, zero missing
keys on Leduc / wrapped Goofspiel / Dark Hex). That off-path fidelity matters
because the Bayesian likelihood (``bayesian.py`` ``_ev_from``/``_likelihood``), the
P7b EV rollout (``property7.py``), and ``compute_true_posterior``
(``elicitation/posterior.py``) all recurse the full subtree.

**Cache key.** ``(str(game), iterations, solver, seed?, pyspiel version, code_hash)``.
``str(game)`` is the fully-expanded canonical parameter string
(``...goofspiel(...num_cards=4...)`` ≠ ``...num_cards=5...``), NOT
``get_type().short_name`` (which collapses all Goofspiel sizes and would serve an
M6 small-game policy to an M7 large-game solve). ``pyspiel.__version__`` pins the
solver build, so an OpenSpiel upgrade forces a re-solve. ``code_hash`` is the
SHA-256 of the whole file (or files) that decide what gets solved or trained --
``rl_selfplay.py`` for RL, computed by :func:`_file_sha256`; ``cfr.py`` AND this
module's own file for CFR (:func:`_combined_sha256`), since CFR's cache
orchestration -- whether a blob is a hit, how ``iterations``/``plus`` reach
:func:`lmopt.games.cfr.solve_cfr` -- lives in :func:`load_or_solve_cfr` below,
not in ``cfr.py`` alone. Each caller computes and passes its own ``code_hash``
(never defaulted, so a caller cannot forget it). Whole-file, not just the
solving function: a harness change landing on a shared checkout mid-sweep (two
piecewise Katana jobs on the same filesystem, one before and one after) must
never serve a policy solved under the OTHER code with the same key. Hashing
whole files means an unrelated docstring edit also forces a re-solve; that is a
deliberate, cheap false positive next to the alternative of silently serving a
stale reference, and narrowing the hash to "just the solving function" was
considered and rejected because it invites an argument, function by function,
about which one counts.

**Invariants.** "Imported CFR, never hand-rolled" is untouched — OpenSpiel still
solves on a miss via :func:`lmopt.games.cfr.solve_cfr`; the cache only persists its
solved average policy. ``scripts/validate_cfr.py`` (the scientific-validity gate)
is NOT fed by this cache; it keeps solving fresh as an independent ground truth.

**Known bound (not fixed here, by design).** Run-time load trusts the version pin
plus the CI fidelity test (``tests/test_policy_cache_cfr.py``); it does not
re-verify exploitability on every load. A blob corrupted on disk *after* CI, or
produced on a different OpenSpiel build that reports the same ``__version__``
string, would load without detection. The shape-assert catches truncation and the
``str(game)`` key prevents cross-game reuse; the residual risk is judged
proportionate for a single-machine harness.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pyspiel

DEFAULT_CFR_DIR = "data/cfr_policies"
DEFAULT_RL_DIR = "data/rl_policies"

#: The solver wrapper: a sibling of this file, not an importable path, so
#: hashing it never triggers the lazy ``lmopt.games.cfr`` import (see
#: :func:`load_or_solve_cfr`) at module load time.
_CFR_SOLVER_FILE = Path(__file__).with_name("cfr.py")

#: This module's own file. CFR's cache ORCHESTRATION -- whether a blob on disk
#: is treated as a hit, and exactly how ``iterations``/``plus`` reach
#: :func:`lmopt.games.cfr.solve_cfr` -- lives in :func:`load_or_solve_cfr`
#: below, in THIS file, not in ``cfr.py``. A CFR ``code_hash`` built from
#: ``_CFR_SOLVER_FILE`` alone would miss a change here (e.g. how the two
#: arguments are threaded through on a miss), so the CFR key hashes both
#: files together (:func:`_combined_sha256`). RL needs no second file: its
#: orchestration (``RLSelfPlayAgent.load_or_train``) already lives inside
#: ``rl_selfplay.py``, the one file ``_RL_TRAIN_FILE`` hashes.
_CFR_ORCHESTRATION_FILE = Path(__file__)


def game_key(game: pyspiel.Game) -> str:
    """Canonical, parameter-expanded cache key for a game (``str(game)``)."""
    return str(game)


def _file_sha256(path: str | Path) -> str:
    """
    hex SHA-256 of a file's current bytes, for the cache key's ``code_hash``
    component
    :param path: the file whose bytes determine the cache-key component (str | Path)
    :returns: 64-hex-char digest (str)
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _combined_sha256(*paths: str | Path) -> str:
    """
    hex SHA-256 over the concatenated per-file digests of several files, for a
    ``code_hash`` that must change if ANY of them does
    :param paths: the files whose bytes jointly determine the cache-key component (str | Path)
    :returns: 64-hex-char digest (str)
    """
    return hashlib.sha256(
        ":".join(_file_sha256(p) for p in paths).encode("utf-8")
    ).hexdigest()


def _pyspiel_version() -> str:
    """
    the OpenSpiel build the cache key is stamped with
    A key whose version component is a placeholder would drop the version out of the key, and a
    sidecar from another solver build could then satisfy a live lookup; so a build that does not
    state its version is refused rather than stamped.
    :returns: ``pyspiel.__version__`` (str)
    :raises RuntimeError: when the build has no ``__version__`` or it is not a non-empty string
    """
    version = getattr(pyspiel, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError(
            "pyspiel exposes no usable __version__, so the policy cache key cannot name the "
            "OpenSpiel build that solved a policy; refusing to stamp a placeholder"
        )
    return version


def cache_components(
    game: pyspiel.Game,
    *,
    iters: int,
    solver: str,
    code_hash: str,
    seed: Optional[int] = None,
) -> dict:
    """The full key dict persisted in the sidecar and compared on load.

    Every field that determines the solved/trained policy is included, so any
    change (game parameters, iteration count, solver, OpenSpiel/numpy build, or
    the code that does the solving/training) yields a distinct key and forces a
    re-solve rather than serving a stale reference. ``seed`` is included only
    when a caller passes it; the XFP RL policy is seed-independent (the seed
    drives play-time sampling, not the average-policy tabulation), so
    ``RLSelfPlayAgent.load_or_train`` deliberately omits it — see that method.
    ``code_hash`` has no default: a caller must compute it with
    :func:`_file_sha256` (or :func:`_combined_sha256` over more than one file)
    from the file(s) that actually decide what it solves or trains, so a call
    site can never fall back to an old key shape by omission.
    :param game: the game to solve or train on (pyspiel.Game)
    :param iters: solver/training iteration count (int)
    :param solver: the solver name, e.g. "cfr+" or "xfp" (str)
    :param code_hash: SHA-256 of the file that decides what gets solved/trained (str)
    :param seed: RNG seed, only for a caller whose result depends on it (int | None)
    :returns: the full key dict (dict)
    """
    comps = {
        "game": game_key(game),
        "iters": int(iters),
        "solver": solver,
        "pyspiel_version": _pyspiel_version(),
        "numpy_version": np.__version__,
        "code_hash": code_hash,
    }
    if seed is not None:
        comps["seed"] = int(seed)
    return comps


def _slug(text: str, maxlen: int = 40) -> str:
    """Readable, filesystem-safe fragment of a game short-name for the filename."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return s[:maxlen] or "game"


def paths(cache_dir: str | Path, kind: str, components: dict) -> tuple[str, str]:
    """Return ``(blob_path, meta_path)`` for a key.

    The filename carries a readable prefix (kind + game short-name + iters) plus a
    stable 12-hex-char hash of the full components, so distinct keys never collide
    and the file is human-identifiable in ``data/``.
    """
    short = _slug(components["game"].split("(")[0])
    digest = hashlib.sha256(
        json.dumps(components, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    stem = f"{kind}_{short}_iters{components['iters']}_{digest}"
    base = os.path.join(str(cache_dir), stem)
    return base + ".pkl", base + ".json"


def meta_matches(meta_path: str, components: dict) -> bool:
    """True iff the on-disk sidecar equals ``components`` exactly.

    This is the Note-1 load contract: a persisted array is only ever paired with a
    game whose ``str(game)`` (and iters/solver/seed/version) match the sidecar, so
    a mismatched-but-same-shape array can never be assigned as a silent permutation.
    """
    try:
        with open(meta_path) as f:
            on_disk = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return on_disk == components


def _atomic_write_bytes(path: str, write_fn) -> None:
    """Write via ``write_fn(file_obj)`` to a per-writer-unique temp file
    beside ``path``, then publish with one atomic ``os.replace``.

    A concurrent reader of ``path`` only ever observes the complete prior
    file or the complete new file, never a partial/spliced one: ``path``
    itself is never opened for writing, only the freshly-created temp file
    is, and only the final rename touches ``path``. The temp name is
    PER-WRITER-UNIQUE (``tempfile.mkstemp``, not a fixed suffix -- see
    ``docs/working_docs/bugs.md`` 2026-09-09 for the defect a fixed name
    produces under real concurrency): two concurrent writers to the same
    ``path`` never share a temp file, so neither can truncate the other's
    in-flight write. A writer killed between the temp file's creation and
    the final replace leaves only an orphaned temp file, never a partial
    ``path``.
    """
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_meta(meta_path: str, components: dict) -> None:
    text = json.dumps(components, indent=2, sort_keys=True)
    _atomic_write_bytes(meta_path, lambda f: f.write(text.encode("utf-8")))


def announce_cache_state(
    game_name: str, iters: int, cache_dir: str, require_hit: bool = False
) -> None:
    """Print whether the CFR reference will be LOADED or SOLVED (M8 B4c).

    docs/working_docs/bugs.md 2026-07-11 names the defect as a solve happening *silently*:
    :func:`load_or_solve_cfr`'s miss branch goes straight to ``solve_cfr`` with no
    output, and the callers' own "Loading/solving CFR" wording was byte-identical
    for a warm hit and a ~20-minute cold solve. This resolves the disjunction
    BEFORE the call, using the same public helpers ``load_or_solve_cfr`` itself
    uses to decide, so the announcement cannot drift from the real hit condition.

    Lives here rather than in a script because four call sites consume it
    (parity_check, analyse, analyse_stats, run_sweep) and it wraps three
    functions from this module; a script importing another script for a library
    concern is the layering inversion this avoids.

    PARTIAL FIX: this makes the cost visible, it does not avoid it. Warming
    remains the operator's job (``scripts/precompute_references.py``).

    :param game_name: the registry key of the game (str)
    :param iters: the CFR iteration count that is part of the cache key (int)
    :param cache_dir: the directory holding the cached policies (str)
    :param require_hit: whether the caller will refuse to solve on a miss, so the miss
        message says the next call raises rather than that a solve is starting (bool)
    :returns: None
    """
    from lmopt.games.loaders import load_game

    comps = cache_components(
        load_game(game_name), iters=iters, solver="cfr+",
        code_hash=_combined_sha256(_CFR_SOLVER_FILE, _CFR_ORCHESTRATION_FILE),
    )
    blob_path, meta_path = paths(cache_dir, "cfr", comps)
    if os.path.exists(blob_path) and meta_matches(meta_path, comps):
        print(f"  CFR cache HIT for {game_name} ({iters} iters) — loading.", flush=True)
    elif require_hit:
        print(
            f"  CFR cache MISS for {game_name} ({iters} iters) — nothing is solved "
            f"because a cache hit is required, so the next step raises. Warm it first "
            f"with scripts/precompute_references.py --games {game_name}",
            flush=True,
        )
    else:
        print(
            f"  CFR cache MISS for {game_name} ({iters} iters) — solving now. "
            f"A dark-hex rectangle costs ~20 min; warm it first with "
            f"scripts/precompute_references.py --games {game_name}",
            flush=True,
        )


def load_or_solve_cfr(
    game: pyspiel.Game,
    iterations: int,
    *,
    plus: bool = True,
    cache_dir: str | Path = DEFAULT_CFR_DIR,
    require_hit: bool = False,
):
    """Load a cached CFR(+) policy, or solve-and-persist on a miss.

    Returns an OpenSpiel ``TabularPolicy`` whose ``action_probabilities(state,
    player)`` are identical to a fresh :func:`solve_cfr` at every info-set. Drop-in
    for the CFRRef agent, the Bayesian likelihood, and the P3/P5/P7 yardstick — all
    of which consume only ``action_probabilities``.

    Args:
        game: a sequential (turn-based) OpenSpiel game.
        iterations: CFR iterations (part of the cache key).
        plus: CFR+ when True (``solver="cfr+"``), else vanilla CFR.
        cache_dir: directory for the ``.pkl`` blob + ``.json`` sidecar.
        require_hit: when True, raise ``RuntimeError`` on a cache miss instead
            of solving. M7 parallel runs set this so a cold cache fails loudly
            up front rather than letting N workers re-solve concurrently
            (design.md "no silent fallbacks" [INVARIANT]); warm the cache
            first with ``scripts/precompute_references.py``.
    """
    from open_spiel.python.policy import TabularPolicy, tabular_policy_from_callable

    comps = cache_components(
        game, iters=iterations, solver="cfr+" if plus else "cfr",
        code_hash=_combined_sha256(_CFR_SOLVER_FILE, _CFR_ORCHESTRATION_FILE),
    )
    blob_path, meta_path = paths(cache_dir, "cfr", comps)

    if os.path.exists(blob_path) and meta_matches(meta_path, comps):
        try:
            with open(blob_path, "rb") as f:
                array = pickle.load(f)
            tab = TabularPolicy(game)
            if (isinstance(array, np.ndarray)
                    and array.shape == tab.action_probability_array.shape):
                tab.action_probability_array = array
                return tab
            # Wrong type / shape => corrupt or foreign blob. Re-solve rather than
            # assign a bad array, and SAY SO (design.md "no silent fallbacks").
            print(
                f"[policy_cache] WARNING: cached CFR blob {blob_path} unusable "
                f"(type={type(array).__name__}, shape mismatch); re-solving.",
                file=sys.stderr,
            )
        except (pickle.UnpicklingError, EOFError, ValueError, OSError,
                AttributeError, TypeError) as exc:
            print(
                f"[policy_cache] WARNING: cached CFR blob {blob_path} corrupt "
                f"({exc!r}); re-solving.",
                file=sys.stderr,
            )

    if require_hit:
        raise RuntimeError(
            f"CFR cache miss for {comps} with require_hit=True — warm the "
            "cache with scripts/precompute_references.py before a parallel run"
        )

    # --- Miss: solve fresh via OpenSpiel and persist ---
    from lmopt.games.cfr import solve_cfr

    avg = solve_cfr(game, iterations, plus=plus)
    tab = tabular_policy_from_callable(game, avg)
    os.makedirs(cache_dir, exist_ok=True)
    _atomic_write_bytes(
        blob_path, lambda f: pickle.dump(tab.action_probability_array, f)
    )
    _write_meta(meta_path, comps)
    return tab