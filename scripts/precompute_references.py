#!/usr/bin/env python
"""Cache-once-freeze entrypoint: solve/train every game's CFR + RL reference once.

Pay the expensive solve/train once, up front, so the
sweep and analysis just load. Running this before a sweep lets the ~23-min Liar's
Dice CFR+ (and the XFP training) happen deliberately (e.g. overnight) rather than
silently mid-run. On a warm cache every game reports a hit and returns instantly.

Usage:
    # Warm the five canonical games at the sweep's iteration counts:
    uv run python scripts/precompute_references.py --cfr-iters 1000 --rl-iters 1000

    # Only some games:
    uv run python scripts/precompute_references.py --games kuhn leduc --cfr-iters 200

    # Verify a warm cache without solving or training anything (exit 0 when every
    # reference hits, exit 3 when any misses, every miss listed):
    .venv/bin/python scripts/precompute_references.py --check-only --games kuhn

    # The same check, reading games, iteration counts and cache directories from
    # a run config (RL is checked only when ``rl`` is in its ``opponents``; keys
    # such as ``__PORT__`` elsewhere in a rendered config are ignored):
    .venv/bin/python scripts/precompute_references.py --check-only --config configs/run_experiment_recycle_v8_m1.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

from lmopt.agents.rl_selfplay import RLSelfPlayAgent, _RL_TRAIN_FILE
from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

from lmopt.games.loaders import CANONICAL_GAMES, GAMES
from lmopt.games.policy_cache import (
    _CFR_ORCHESTRATION_FILE,
    _CFR_SOLVER_FILE,
    DEFAULT_CFR_DIR,
    DEFAULT_RL_DIR,
    _combined_sha256,
    _file_sha256,
    cache_components,
    load_or_solve_cfr,
    meta_matches,
    paths,
)


def _cfr_is_cached(game, iters: int, cache_dir: str) -> bool:
    comps = cache_components(
        game,
        iters=iters,
        solver="cfr+",
        code_hash=_combined_sha256(_CFR_SOLVER_FILE, _CFR_ORCHESTRATION_FILE),
    )
    blob, meta = paths(cache_dir, "cfr", comps)
    return os.path.exists(blob) and meta_matches(meta, comps)


def _rl_is_cached(game, iters: int, cache_dir: str) -> bool:
    # XFP policy is seed-independent, so the RL cache key omits the seed.
    comps = cache_components(
        game, iters=iters, solver="xfp", code_hash=_file_sha256(_RL_TRAIN_FILE)
    )
    blob, meta = paths(cache_dir, "rl", comps)
    return os.path.exists(blob) and meta_matches(meta, comps)


#: Exit status of ``--check-only`` when any reference is missing or unwarmable.
EXIT_COLD_CACHE = 3


def _load_config(path: str) -> dict:
    """
    read the reference-relevant keys of a run config with plain ``yaml.safe_load``;
    every other key, including strings holding placeholders such as ``__PORT__``,
    is ignored
    :param path: path to a run-experiment YAML file (rendered or template)
    :returns: dict with ``games``, ``cfr_iters``, ``cfr_cache_dir`` and
        ``check_rl``, plus ``rl_iters`` and ``rl_cache_dir`` when ``check_rl``
    :raises ValueError: when the file is not a mapping, or a needed key is
        missing or has the wrong type
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    def need(key: str, kind: type):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, kind):
            raise ValueError(
                f"{path}: key {key!r} must be {kind.__name__}, got {value!r}"
            )
        return value

    games = need("games", list)
    if not games or not all(isinstance(g, str) for g in games):
        raise ValueError(f"{path}: key 'games' must be a non-empty list of names")
    unknown = [g for g in games if g not in GAMES]
    if unknown:
        raise ValueError(f"{path}: unknown games {unknown}; known: {list(GAMES)}")
    out = {
        "games": games,
        "cfr_iters": need("cfr_iters", int),
        "cfr_cache_dir": need("cfr_cache_dir", str),
        "check_rl": "rl" in need("opponents", list),
    }
    if out["check_rl"]:
        out["rl_iters"] = need("rl_iterations", int)
        out["rl_cache_dir"] = need("rl_cache_dir", str)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warm the CFR + RL policy caches.")
    p.add_argument(
        "--games",
        nargs="+",
        default=None,
        choices=list(GAMES),
        help="Games to warm (default: the five canonical games; "
        "depth variants like goofspiel_5 must be named explicitly "
        "- at --cfr-iters 1000 a depth variant costs ~20 min). "
        "darkhex_3x3 has no full CFR+ reference and cannot be "
        "warmed this way; see lmopt.games.depth_reference.",
    )
    p.add_argument(
        "--cfr-iters", type=int, default=None, help="CFR+ iterations (default 1000)."
    )
    p.add_argument(
        "--rl-iters", type=int, default=None, help="XFP iterations (default 1000)."
    )
    p.add_argument("--seed", type=int, default=42, help="RL training seed.")
    p.add_argument(
        "--cfr-cache-dir",
        default=None,
        help=f"CFR cache directory (default {DEFAULT_CFR_DIR}).",
    )
    p.add_argument(
        "--rl-cache-dir",
        default=None,
        help=f"RL cache directory (default {DEFAULT_RL_DIR}).",
    )
    p.add_argument("--skip-rl", action="store_true", help="CFR only.")
    p.add_argument("--skip-cfr", action="store_true", help="RL only.")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Solve and train nothing: load every reference with "
        "require_hit, list each miss, exit 3 if any missed and "
        "0 if all hit.",
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="With --check-only: read games, cfr_iters, rl_iterations, "
        "cfr_cache_dir and rl_cache_dir from this run config, "
        "checking RL only when 'rl' is in its opponents. "
        "Exclusive with --games/--cfr-iters/--rl-iters/"
        "--cfr-cache-dir/--rl-cache-dir.",
    )
    args = p.parse_args()

    if args.config is not None:
        if not args.check_only:
            p.error("--config is only supported together with --check-only")
        explicit = [
            flag
            for flag, value in (
                ("--games", args.games),
                ("--cfr-iters", args.cfr_iters),
                ("--rl-iters", args.rl_iters),
                ("--cfr-cache-dir", args.cfr_cache_dir),
                ("--rl-cache-dir", args.rl_cache_dir),
            )
            if value is not None
        ]
        if explicit:
            p.error(f"--config already supplies {', '.join(explicit)}")
        try:
            cfg = _load_config(args.config)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            p.error(str(exc))
        args.games = cfg["games"]
        args.cfr_iters = cfg["cfr_iters"]
        args.cfr_cache_dir = cfg["cfr_cache_dir"]
        args.rl_iters = cfg.get("rl_iters")
        args.rl_cache_dir = cfg.get("rl_cache_dir")
        args.skip_rl = args.skip_rl or not cfg["check_rl"]
        return args

    if args.games is None:
        args.games = list(CANONICAL_GAMES)
    if args.cfr_iters is None:
        args.cfr_iters = 1000
    if args.rl_iters is None:
        args.rl_iters = 1000
    if args.cfr_cache_dir is None:
        args.cfr_cache_dir = DEFAULT_CFR_DIR
    if args.rl_cache_dir is None:
        args.rl_cache_dir = DEFAULT_RL_DIR
    return args


def _check_only(args: argparse.Namespace) -> int:
    """
    report every missing or unwarmable reference without solving or training
    :param args: parsed arguments with ``games``, iteration counts, cache
        directories, ``seed`` and the ``skip_cfr``/``skip_rl`` switches
    :returns: ``EXIT_COLD_CACHE`` if any reference is missing or unwarmable, else 0
    """
    from lmopt.games.depth_reference import DEPTH_REFERENCES

    print(f"Checking references (no solving) for: {', '.join(args.games)}")
    if not args.skip_cfr:
        print(f"  CFR+ iters={args.cfr_iters}  cache_dir={args.cfr_cache_dir}")
    if not args.skip_rl:
        print(f"  XFP iters={args.rl_iters}  cache_dir={args.rl_cache_dir}")

    problems = 0
    for name in args.games:
        if name in DEPTH_REFERENCES:
            print(
                f"UNWARMABLE {name}: no full CFR+ or XFP reference can exist "
                f"(its reference is the partial MCCFR capture at "
                f"{DEPTH_REFERENCES[name]}), so this cache cannot be warmed"
            )
            problems += 1
            continue
        game = GAMES[name]()
        if not args.skip_cfr:
            try:
                load_or_solve_cfr(
                    game,
                    args.cfr_iters,
                    plus=True,
                    cache_dir=args.cfr_cache_dir,
                    require_hit=True,
                )
            except RuntimeError:
                print(
                    f"MISS {name} CFR iters={args.cfr_iters} "
                    f"cache_dir={args.cfr_cache_dir}"
                )
                problems += 1
            else:
                print(f"HIT  {name} CFR iters={args.cfr_iters}")
        if not args.skip_rl:
            try:
                RLSelfPlayAgent.load_or_train(
                    game,
                    num_iterations=args.rl_iters,
                    seed=args.seed,
                    cache_dir=args.rl_cache_dir,
                    require_hit=True,
                )
            except RuntimeError:
                print(
                    f"MISS {name} RL iters={args.rl_iters} "
                    f"cache_dir={args.rl_cache_dir}"
                )
                problems += 1
            else:
                print(f"HIT  {name} RL iters={args.rl_iters}")

    if problems:
        print(f"COLD: {problems} reference(s) missing or unwarmable")
        return EXIT_COLD_CACHE
    print("WARM: every reference hit")
    return 0


def main() -> int:
    args = parse_args()
    if args.check_only:
        return _check_only(args)
    print(f"Warming references for: {', '.join(args.games)}")
    print(f"  CFR+ iters={args.cfr_iters}  XFP iters={args.rl_iters} seed={args.seed}")
    print(f"  cfr_cache_dir={args.cfr_cache_dir}  rl_cache_dir={args.rl_cache_dir}")
    print()

    for name in args.games:
        from lmopt.games.depth_reference import DEPTH_REFERENCES

        if name in DEPTH_REFERENCES:
            # Refuse unconditionally - neither --skip-cfr nor --skip-rl makes
            # this safe. CFR+ (TabularPolicy) MemoryErrors at this size
            # (6 GB cap, 17.6 s), and XFP is no safer:
            # open_spiel's fictitious_play.XFPSolver computes each iteration's
            # best response via best_response.py's
            # get_all_states.get_all_states(game, depth_limit=-1, ...) - the
            # same full-tree enumeration. "--skip-cfr to warm only the RL
            # policy" is therefore no escape hatch: it hits the identical
            # MemoryError via XFP.
            sys.exit(
                f"{name} has no full CFR+ reference and no XFP RL reference, "
                "and cannot have either (both require full-tree enumeration, "
                "which OOMs at this size). Its reference is the partial MCCFR "
                f"capture at {DEPTH_REFERENCES[name]}; there is nothing to warm."
            )
        game = GAMES[name]()
        print(f"=== {name.upper()} ===")

        if not args.skip_cfr:
            hit = _cfr_is_cached(game, args.cfr_iters, args.cfr_cache_dir)
            t0 = time.monotonic()
            load_or_solve_cfr(
                game, args.cfr_iters, plus=True, cache_dir=args.cfr_cache_dir
            )
            dt = time.monotonic() - t0
            print(f"  CFR+  {'HIT ' if hit else 'MISS'} ({dt:6.1f}s)")

        if not args.skip_rl:
            hit = _rl_is_cached(game, args.rl_iters, args.rl_cache_dir)
            t0 = time.monotonic()
            RLSelfPlayAgent.load_or_train(
                game,
                num_iterations=args.rl_iters,
                seed=args.seed,
                cache_dir=args.rl_cache_dir,
                check_exploitability=not hit,
            )
            dt = time.monotonic() - t0
            print(f"  XFP   {'HIT ' if hit else 'MISS'} ({dt:6.1f}s)")
        print()

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
