"""
Probe-arm fixtures, written the way `run_sweep` writes an arm: shards, `run_meta.json` through the
production `write_run_meta`, and a complete `shard_outcomes.json`.

Shared by the probe-scoring tests. Nothing here scores anything.
"""

import json
from pathlib import Path

import polars as pl

from lmopt.games.loaders import load_game
from lmopt.games.policy_cache import load_or_solve_cfr
from lmopt.logging.run_meta import write_run_meta

ACTOR = "random"
LM_MODEL = "test-model"
CFR_ITERS = 20


def rows(
    info_sets,
    *,
    game="kuhn",
    probe_type="none",
    prompt_variant="base",
    actor=ACTOR,
    player=None,
    fallback=False,
    memory_rung=None,
    seed=42,
):
    """
    LM-side rows with a logged reference policy, as the writer emits them
    :param info_sets: {info_set: [parsed actions]}, one row per action
    :param game: the `game` column value
    :param probe_type: the arm's probe type
    :param prompt_variant: the arm's prompt variant
    :param actor: the `actor` column value
    :param player: pins the acting seat, or None to alternate
    :param fallback: True marks every row a harness fallback, or a list of flags per row
    :param memory_rung: the `memory_rung` column value, or None for a pre-rung row
    :param seed: the `seed` column value, which is the replicate identity every per-seed
        statistic is grouped by
    :returns: a pl.DataFrame
    """
    records = []
    i = 0
    for info_set, actions in info_sets.items():
        for action in actions:
            seat = i % 2 if player is None else player
            flag = (
                fallback[i] if isinstance(fallback, (list, tuple)) else bool(fallback)
            )
            records.append(
                {
                    "game": game,
                    "info_set": info_set,
                    "player_to_act": seat,
                    "lm_seat": seat,
                    "parsed_action": action,
                    "actor": actor,
                    "prompt": "p",
                    "prompt_variant": prompt_variant,
                    "probe_type": probe_type,
                    "cfr_policy": '{"0": 0.5, "1": 0.5}',
                    "cfr_ref_off_support": False,
                    "elicited_belief": None,
                    "true_posterior": None,
                    "reward": 0.0,
                    "hand_idx": i,
                    "matchup_id": "random_vs_cfrref",
                    "seed": seed,
                    "fallback_used": flag,
                    "memory_rung": memory_rung,
                }
            )
            i += 1
    return pl.DataFrame(records)


def warm_cfr_cache(cache_dir, game_key="kuhn"):
    """
    solve and persist a small CFR reference so a P6 reference load hits the cache
    :param cache_dir: the cache directory
    :param game_key: the registry game key
    :returns: the cache directory as a str
    """
    load_or_solve_cfr(load_game(game_key), CFR_ITERS, cache_dir=str(cache_dir))
    return str(cache_dir)


def write_arm(
    path,
    frames,
    *,
    cfr_cache_dir,
    probe_type="none",
    prompt_variant="base",
    drive_openings=None,
    games="frames",
    agent0="random",
    memory_rung=None,
    temperature=0.0,
    seat_mode="random",
    backend="test",
    served_weights_revision=None,
    model_id=None,
    prompt_spec_hash=None,
    n_hands=4,
    n_seeds=1,
    composed_from=None,
    omit=(),
    model=LM_MODEL,
    diagnostic=False,
):
    """
    one arm directory: shards, run_meta.json and a complete shard_outcomes.json
    :param path: the arm directory
    :param frames: {shard name: pl.DataFrame}
    :param cfr_cache_dir: the reference cache the arm's run_meta records
    :param probe_type: the arm's probe type
    :param prompt_variant: the arm's prompt variant
    :param drive_openings: forced openings for a driven arm, or None
    :param games: the games the arm was configured for: "frames" for the games it wrote, a list,
        or None to record no game list at all
    :param agent0: "random" or "lm"; an "lm" arm records the model and its rows carry it as actor
    :param memory_rung: recorded `memory_rung`, or None for a pre-rung arm
    :param temperature: recorded decoding temperature
    :param seat_mode: recorded seat mode
    :param backend: recorded backend
    :param served_weights_revision: recorded revision, or None
    :param model_id: top-level `model_id`, or None to omit
    :param prompt_spec_hash: top-level hash, or None to omit
    :param n_hands: recorded hand count
    :param n_seeds: recorded seed count
    :param composed_from: piece directories to record as `composed_from`, or None
    :param omit: resolved_config keys to leave out of run_meta entirely (absent, not null)
    :param diagnostic: True records the diagnostic marker a null rerun's template carries
    :param model: the model an "lm" arm records
    :returns: the arm directory as a Path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.write_parquet(path / f"{name}.parquet")
    resolved = {
        "opponents": ["cfr_ref"],
        "agent0": agent0,
        "seat_mode": seat_mode,
        "n_hands": n_hands,
        "n_seeds": n_seeds,
        "seed": 42,
        "cfr_iters": CFR_ITERS,
        "cfr_cache_dir": str(cfr_cache_dir),
        "probe": {"type": probe_type, "prompt_variant": prompt_variant},
        "drive_openings": drive_openings,
        "out_dir": str(path),
        "memory_rung": memory_rung,
        "temperature": temperature,
        "backend": backend,
        "served_weights_revision": served_weights_revision,
    }
    if diagnostic:
        resolved["diagnostic"] = True
    for key in omit:
        resolved.pop(key, None)
    if agent0 == "lm":
        resolved["model"] = model
    if games == "frames":
        resolved["games"] = sorted(frames)
    elif games is not None:
        resolved["games"] = sorted(games)
    write_run_meta(
        str(path),
        resolved_config=resolved,
        max_workers=1,
        seed_policy={},
        git_sha="test",
        runner="serial",
        model_id=model_id,
        n_hands=n_hands,
        prompt_spec_hash=prompt_spec_hash,
    )
    if composed_from is not None:
        meta_path = path / "run_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["composed_from"] = [str(p) for p in composed_from]
        meta_path.write_text(json.dumps(meta))
    with open(path / "shard_outcomes.json", "w") as fh:
        json.dump(
            {
                "n_requested": len(frames),
                "n_completed": len(frames),
                "n_failed": 0,
                "shards": [],
            },
            fh,
        )
    return path
