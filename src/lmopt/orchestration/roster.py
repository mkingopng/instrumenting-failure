"""Collection roster: the model, job-kind, size and walltime-estimate tables.

One job is one seed of one model of one kind on one H200. This module holds the
tables (``MODELS``, ``JOBS``, the rates and the estimate helpers) that
``scripts/collection_job.py`` renders into job plans and that the ladder readers
consume, so a library caller never imports a script.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import TypedDict, Unpack

from lmopt.prompting.probe_variants import (
    GAME_KEYS,
    variant_spec,
)

# The resource request every job carries. WALLTIME is the fallback; each job's plan
# derives its own walltime.
SELECT = "select=1:ncpus=8:mem=64gb:ngpus=1:gpu_h200=True"
WALLTIME = "2:00:00"

BASE_SEED = 42
# Real kinds take an array index: seed = BASE_SEED + index.
INDICES_BASE = 10
INDICES_PROBE = 5
# Gate pilots: 20 hands x 5 seeds at the base seed, no index.
PILOT_N_HANDS = 20
PILOT_N_SEEDS = 5

PLACEHOLDER = re.compile(r"__[A-Z0-9_]+__")

# ---------------------------------------------------------------------------
# Wall-clock estimate of one job.
#
# All cells (game x opponent) of a seed run concurrently and the hands within a
# cell run in sequence, so a sweep lasts as long as its slowest cell's chain:
#   chain(game) = CHAIN_MIN_M0[model][game] * max(1.0, LEVEL_RATIO[model][rung][game])
#                 * n_hands * n_seeds / CHAIN_HANDS
#   est = ceil(STARTUP_MIN + max over the job's games of chain(game))
# The off-path job runs its five one-game sweeps one after another, so its chain
# is the sum over its sweeps instead of the max. Seeds inside one job (the pilots'
# five) are counted as if they ran in sequence, an upper bound.
#
#   STARTUP_MIN   10  estimated: job start, reference load, vLLM load, teardown
#   CHAIN_MIN_M0      measured on H200, v7 M0, all 10 indices: minutes per 1000
#                     hands of the slowest opponent cell of each game, from row
#                     timestamps (span / distinct hand_idx), mean over indices
#   LEVEL_RATIO       measured on the v7 gate pilots (20 hands x 5 seeds): per
#                     hand, the slowest cell's seconds at M1 or M2 over M0, max over
#                     opponents of the per-seed mean; only an inference as a production
#                     ratio, so every M1 and M2 estimate is inferred and carries
#                     pending_pilot until an index-0 job measures it. A ratio below
#                     1.0 is pilot noise, and the estimate uses 1.0 instead.
#
# Falsification condition: an index-0 job whose measured wall time exceeds its
# estimate by more than 10% means the ratio or the chain is wrong for that (model,
# level); refit from that job before submitting the other indices. The chains are
# means over indices, so an index landing up to about 5% above its estimate is
# ordinary spread (v7 M0: Qwen 173-186 min against 185, OLMo 76-87 against 86).
#
# The walltime requested is the estimate x WALLTIME_MARGIN, rounded up to the
# next WALLTIME_STEP_MIN minutes, and plan() refuses a job whose walltime would
# exceed WALLTIME_CAP_MIN (the 12-hour ceiling).
# ---------------------------------------------------------------------------
STARTUP_MIN = 10
CHAIN_HANDS = 1000
WALLTIME_MARGIN = 1.25
WALLTIME_STEP_MIN = 15
WALLTIME_CAP_MIN = 12 * 60
CHAIN_MIN_M0: dict[str, dict[str, float]] = {
    "qwen": {
        "kuhn": 4.5,
        "leduc": 138.0,
        "goofspiel": 174.5,
        "liarsdice": 56.4,
        "darkhex": 43.7,
    },
    "olmo": {
        "kuhn": 6.4,
        "leduc": 12.5,
        "goofspiel": 75.2,
        "liarsdice": 21.1,
        "darkhex": 15.9,
    },
    "mistral": {
        "kuhn": 4.7,
        "leduc": 15.9,
        "goofspiel": 92.4,
        "liarsdice": 75.0,
        "darkhex": 11.7,
    },
}
LEVEL_RATIO: dict[str, dict[str, dict[str, float]]] = {
    "qwen": {
        "M1": {
            "kuhn": 1.67,
            "leduc": 1.51,
            "goofspiel": 1.34,
            "liarsdice": 0.82,
            "darkhex": 1.66,
        },
        "M2": {
            "kuhn": 2.04,
            "leduc": 1.61,
            "goofspiel": 1.58,
            "liarsdice": 2.00,
            "darkhex": 1.74,
        },
    },
    "olmo": {
        "M1": {
            "kuhn": 1.61,
            "leduc": 1.44,
            "goofspiel": 1.35,
            "liarsdice": 1.35,
            "darkhex": 1.35,
        },
        "M2": {
            "kuhn": 2.67,
            "leduc": 3.95,
            "goofspiel": 0.76,
            "liarsdice": 2.56,
            "darkhex": 1.85,
        },
    },
    "mistral": {
        "M1": {
            "kuhn": 4.82,
            "leduc": 1.40,
            "goofspiel": 1.03,
            "liarsdice": 1.14,
            "darkhex": 1.16,
        },
        "M2": {
            "kuhn": 11.88,
            "leduc": 2.08,
            "goofspiel": 1.30,
            "liarsdice": 1.51,
            "darkhex": 2.86,
        },
    },
}
# What an in-job exact-policy pass costs one sweep that runs it, in minutes. Estimated from the
# gap between the last shard and exact_policy.csv on the 30 collected experiment 7 M0 pieces
# (by file mtimes: 0.8 to 1.2 minutes per five-game arm), doubled for the single-game
# off-path sweeps and the probe kinds, which have not run the pass. Index 0 of the two probe
# kinds measures it. Experiment 7's estimates carry no allowance.
EXACT_PASS_MIN_PER_SWEEP_V8 = 2.0


@dataclass(frozen=True)
class Rates:
    """
    the wall-clock tables one experiment's estimates read
    :param chain_min_m0: minutes per 1000 hands of the slowest opponent cell, per model and game (dict)
    :param level_ratio: per-hand slowdown of M1 and M2 over M0, per model, level and game (dict)
    :param measured_rungs: memory levels whose chain is a production measurement, not an inference (tuple of str)
    :param exact_pass_min_per_sweep: minutes one in-job exact-policy pass adds to a sweep that runs it (float)
    """

    chain_min_m0: dict[str, dict[str, float]]
    level_ratio: dict[str, dict[str, dict[str, float]]]
    measured_rungs: tuple[str, ...]
    exact_pass_min_per_sweep: float


# Experiment 7's tables are the module-level ones above and are what every default reads, so an
# experiment 7 estimate cannot move when experiment 8's rates are refit.
EXPERIMENT_7_RATES = Rates(CHAIN_MIN_M0, LEVEL_RATIO, ("M0",), 0.0)
# Experiment 8 starts from experiment 7's figures and measures none of them yet: every estimate
# is inferred and pending its pilot, and the nine pilots and the index-0 jobs refit it.
EXPERIMENT_8_RATES = Rates(
    copy.deepcopy(CHAIN_MIN_M0),
    copy.deepcopy(LEVEL_RATIO),
    (),
    EXACT_PASS_MIN_PER_SWEEP_V8,
)

# P2's adaptivity delta needs >= 1000 hands per endpoint bin and there are two bins.
P2_MIN_HANDS = 2000
# The experiment a kind belongs to, and the environment variable that lets `plan` act on an
# experiment 7 kind now that experiment 8 is the one being collected. Only the exact value "1"
# counts: a stray "0" or "yes" must not read as consent.
EXPERIMENT_7 = 7
EXPERIMENT_8 = 8
EXPERIMENT_7_OVERRIDE = "LMOPT_ALLOW_EXPERIMENT_7"


@dataclass(frozen=True)
class ModelSpec:
    """
    one served model: its hub id, pinned weights revision, output key and served window
    :param model_id: hub id passed to vllm serve (str)
    :param revision: weights revision pinned on the serve line (str)
    :param short: lowercase last path segment of the id, the out_dir key (str)
    :param max_model_len: window passed to vllm serve as --max-model-len, at most the model's trained context (int)
    """

    model_id: str
    revision: str
    short: str
    max_model_len: int


MODELS: dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        "Qwen/Qwen2.5-14B-Instruct",
        # The Hub's `main` sha for this repo, and the only snapshot in the
        # local HF cache and in the cluster's HF_HOME
        # (/path/to/scratch/hf_cache, which the job serves from). The sha
        # 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 404s on the Hub API and
        # is not in any reachable HF cache.
        "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "qwen2.5-14b-instruct",
        8192,
    ),
    "olmo": ModelSpec(
        "allenai/OLMo-2-1124-13B-Instruct",
        "3a5c85baefbb1896a54d56fe2e76c0395627ddf4",
        "olmo-2-1124-13b-instruct",
        4096,
    ),
    "mistral": ModelSpec(
        "mistralai/Mistral-Nemo-Instruct-2407",
        "04d8a90549d23fc6bd7f642064003592df51e9b3",
        "mistral-nemo-instruct-2407",
        8192,
    ),
}


@dataclass(frozen=True)
class OffpathSweep:
    """
    one off-path drive sweep: a single game driven into a low-probability subgame
    :param game: game key (str)
    :param drive_openings: scripted opening action indices (tuple of int)
    :param seat_mode: which seat the LM plays (str)
    """

    game: str
    drive_openings: tuple[int, ...]
    seat_mode: str


# The probe type whose job runs one sweep per game, one after another.
OFFPATH_PROBE = "offpath_drive"

OFFPATH_SWEEPS: tuple[OffpathSweep, ...] = (
    OffpathSweep("kuhn", (1,), "fixed1"),
    OffpathSweep("leduc", (1, 1, 1, 2, 2), "fixed1"),
    OffpathSweep("goofspiel", (1, 2, 0), "fixed1"),
    OffpathSweep("liarsdice", (1, 9), "fixed0"),
    OffpathSweep("darkhex", (0, 0), "fixed1"),
)


@dataclass(frozen=True)
class JobSpec:
    """
    one job kind: template, memory level, shape, per-model sizes and paths
    :param template: template config path relative to the repo root (str)
    :param rung: memory level of the arm, M0, M1 or M2 (str)
    :param seeded: True when the job takes an array index (bool)
    :param n_indices: number of array indices, None for pilot kinds (int or None)
    :param n_seeds: seeds run inside one job (int)
    :param n_hands: hands per cell for each model (dict of str to int)
    :param basis: measured or inferred, per model (dict of str to str)
    :param est_min: estimated wall-clock minutes, per model (dict of str to int)
    :param pending_pilot: True where the estimate waits on an index-0 job to measure it (dict of str to bool)
    :param p2_licensed: whether the size reaches P2_MIN_HANDS, None for a kind that does not feed P2 (dict of str to bool or None)
    :param out_root: out_root format string, fields index and game (str)
    :param pilot_dir: pilot directory format string, field short (str)
    :param games: games the kind is defined over (tuple of str)
    :param model_games: games one job of each model covers, equal to games for every model (dict of str to tuple of str)
    :param probe: probe type of a probe kind, else None (str or None)
    :param gate: True for pilot kinds, which end with pilot_gate (bool)
    :param experiment: 7 or 8, the experiment the kind belongs to (int)
    :param label: the name a status table prints for the kind, naming the experiment for experiment 8 (str)
    :param exact_pass: True when the kind's template runs the in-job exact-policy pass, which the estimate prices (bool)
    :param requires_exact_pass: True for an experiment 8 probe kind that is refused while its rendered config has the pass off (bool)
    :param action_top_logprobs: per model, the top-k the arm requests on its action call, None until set (dict of str to int or None)
    :param diagnostic: True for a kind that is never a scoreboard arm; it is declared in the roster so ladder_state and compose read it, and left out of the default kind set (bool)
    """

    template: str
    rung: str
    seeded: bool
    n_indices: int | None
    n_seeds: int
    n_hands: dict[str, int]
    basis: dict[str, str]
    est_min: dict[str, int]
    pending_pilot: dict[str, bool]
    p2_licensed: dict[str, bool | None]
    out_root: str
    pilot_dir: str
    games: tuple[str, ...]
    model_games: dict[str, tuple[str, ...]]
    probe: str | None
    gate: bool
    experiment: int
    label: str
    exact_pass: bool
    requires_exact_pass: bool
    action_top_logprobs: dict[str, int | None]
    diagnostic: bool


def _probe_games(probe_type: str) -> tuple[str, ...]:
    """
    the games the prompt-variant registry covers for a probe type
    :param probe_type: relabel or payoff_scale (str)
    :returns: covered game keys in registry order (tuple of str)
    """
    return tuple(g for g in GAME_KEYS if variant_spec(g, probe_type) is not None)


def _chain_min(
    model: str,
    rung: str,
    game: str,
    n_hands: int,
    n_seeds: int,
    rates: Rates = EXPERIMENT_7_RATES,
) -> float:
    """
    minutes of the slowest cell's sequential chain for one game of one job
    :param model: model key (str)
    :param rung: memory level, M0, M1 or M2 (str)
    :param game: game key (str)
    :param n_hands: hands per cell (int)
    :param n_seeds: seeds inside the job, counted as sequential (int)
    :param rates: the tables to read, experiment 7's unless the caller names others (Rates)
    :returns: minutes, unrounded (float)
    """
    ratio = 1.0 if rung == "M0" else max(1.0, rates.level_ratio[model][rung][game])
    return rates.chain_min_m0[model][game] * ratio * n_hands * n_seeds / CHAIN_HANDS


def _estimate(
    model: str,
    rung: str,
    games: tuple[str, ...],
    n_hands: int,
    n_seeds: int,
    sequential: bool,
    rates: Rates = EXPERIMENT_7_RATES,
    exact_pass: bool = False,
) -> int:
    """
    estimated wall-clock minutes of one job, rounded up: start-up plus the slowest
    chain when the games' cells run concurrently, or plus the sum of the chains when
    the games run as sweeps one after another, plus the exact-pass allowance once per sweep
    that runs the pass
    :param model: model key (str)
    :param rung: memory level (str)
    :param games: games the job covers (tuple of str)
    :param n_hands: hands per cell (int)
    :param n_seeds: seeds inside the job (int)
    :param sequential: True when each game is its own sweep run in order, as off-path is (bool)
    :param rates: the tables to read, experiment 7's unless the caller names others (Rates)
    :param exact_pass: True when the job's sweeps run the in-job exact-policy pass (bool)
    :returns: minutes (int)
    """
    chains = [_chain_min(model, rung, g, n_hands, n_seeds, rates) for g in games]
    sweeps = len(games) if sequential else 1
    allowance = rates.exact_pass_min_per_sweep * sweeps if exact_pass else 0.0
    return math.ceil(
        STARTUP_MIN + (sum(chains) if sequential else max(chains)) + allowance
    )


def walltime_for(est_min: int) -> str:
    """
    the PBS walltime to request for an estimate: est_min x WALLTIME_MARGIN rounded up
    to the next WALLTIME_STEP_MIN minutes, refused above WALLTIME_CAP_MIN
    :param est_min: estimated wall-clock minutes (int)
    :returns: walltime as H:MM:SS (str); raises ValueError above the cap
    """
    minutes = (
        math.ceil(est_min * WALLTIME_MARGIN / WALLTIME_STEP_MIN) * WALLTIME_STEP_MIN
    )
    if minutes > WALLTIME_CAP_MIN:
        raise ValueError(
            f"estimated {est_min} min needs a walltime of {minutes} min "
            f"(x{WALLTIME_MARGIN}, rounded up to {WALLTIME_STEP_MIN} min), above the "
            f"{WALLTIME_CAP_MIN}-min ceiling; split the job or refit the estimate"
        )
    return f"{minutes // 60}:{minutes % 60:02d}:00"


def _job(
    kind: str,
    template: str,
    rung: str,
    n_hands: dict[str, int],
    *,
    n_seeds: int,
    n_indices: int | None,
    out_root: str,
    pilot_dir: str,
    games: tuple[str, ...],
    probe: str | None = None,
    gate: bool = False,
    experiment: int = EXPERIMENT_7,
    label: str | None = None,
    exact_pass: bool = False,
    requires_exact_pass: bool = False,
    rates: Rates = EXPERIMENT_7_RATES,
    action_top_logprobs: dict[str, int | None] | None = None,
    diagnostic: bool = False,
) -> JobSpec:
    """
    build a JobSpec, deriving basis, estimate, pending_pilot and the P2 flag from the rate tables
    :param kind: job kind key (str)
    :param template: template config path (str)
    :param rung: memory level (str)
    :param n_hands: hands per cell for each model (dict of str to int)
    :param n_seeds: seeds inside one job (int)
    :param n_indices: array index count or None (int or None)
    :param out_root: out_root format string (str)
    :param pilot_dir: pilot directory format string (str)
    :param games: games one job covers (tuple of str)
    :param probe: probe type or None (str or None)
    :param gate: True for pilot kinds (bool)
    :param experiment: 7 or 8 (int)
    :param label: the status-table name, the kind's own name when None (str or None)
    :param exact_pass: True when the template runs the in-job exact-policy pass (bool)
    :param requires_exact_pass: True when the kind is refused while its rendered config has the pass off (bool)
    :param rates: the tables the estimate reads (Rates)
    :param action_top_logprobs: per-model top-k for the action call, None for every model when None (dict or None)
    :param diagnostic: True for a kind that is never a scoreboard arm (bool)
    :returns: the job spec (JobSpec)
    :raises ValueError: when a top-k is not None or a positive integer
    """
    topk: dict[str, int | None] = (
        {m: None for m in MODELS}
        if action_top_logprobs is None
        else dict(action_top_logprobs)
    )
    if set(topk) != set(MODELS) or any(
        v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1)
        for v in topk.values()
    ):
        raise ValueError(
            f"kind {kind!r}: action_top_logprobs must name every model with None or a "
            f"positive integer, got {topk!r}"
        )
    basis = "measured" if rung in rates.measured_rungs else "inferred"
    return JobSpec(
        template=template,
        rung=rung,
        seeded=n_indices is not None,
        n_indices=n_indices,
        n_seeds=n_seeds,
        n_hands=dict(n_hands),
        basis={m: basis for m in MODELS},
        est_min={
            m: _estimate(
                m,
                rung,
                games,
                n_hands[m],
                n_seeds,
                probe == OFFPATH_PROBE,
                rates,
                exact_pass,
            )
            for m in MODELS
        },
        pending_pilot={m: basis == "inferred" for m in MODELS},
        p2_licensed={
            m: (
                (n_hands[m] >= P2_MIN_HANDS)
                if rung != "M0" and not gate and probe is None
                else None
            )
            for m in MODELS
        },
        out_root=out_root,
        pilot_dir=pilot_dir,
        games=games,
        model_games={m: games for m in MODELS},
        probe=probe,
        gate=gate,
        experiment=experiment,
        label=kind if label is None else label,
        exact_pass=exact_pass,
        requires_exact_pass=requires_exact_pass,
        action_top_logprobs=topk,
        diagnostic=diagnostic,
    )


_FULL_GAMES = GAME_KEYS
_M0_PILOT = "data/pilot/recycle-v7-m0/{short}"
# The M1 and M2 pilots write to fresh directories: the earlier ones hold pilots
# collected on an older memory wording, and a pilot refuses a directory that
# already holds parquet.
_M1_PILOT = "data/pilot/recycle-v7-m1b/{short}"
_M2_PILOT = "data/pilot/recycle-v7-m2b/{short}"
_PROBE_OUT = "data/run_experiment/recycle-v7-probes"
# Experiment 8's own roots. The gate pilots take no `b` suffix: their directories are fresh.
_V8_M0_PILOT = "data/pilot/recycle-v8-m0/{short}"
_V8_M1_PILOT = "data/pilot/recycle-v8-m1/{short}"
_V8_M2_PILOT = "data/pilot/recycle-v8-m2/{short}"
_V8_PROBE_OUT = "data/run_experiment/recycle-v8-probes"


def _same(n: int) -> dict[str, int]:
    """
    one hand count for every model
    :param n: hands per cell (int)
    :returns: model key to n (dict of str to int)
    """
    return {m: n for m in MODELS}


class _PilotSpec(TypedDict, total=False):
    """the keyword fields `_job` takes that a pilot kind may set"""

    probe: str | None
    experiment: int
    label: str | None
    exact_pass: bool
    requires_exact_pass: bool
    rates: Rates
    action_top_logprobs: dict[str, int | None] | None
    diagnostic: bool


def _pilot_kind(
    name: str,
    template: str,
    rung: str,
    pilot_dir: str,
    **spec: Unpack[_PilotSpec],
) -> JobSpec:
    """
    the gate-pilot kind of one memory level: 20 hands x 5 seeds under data/pilot
    :param name: the kind's key, which the status label and the job name carry (str)
    :param template: template config path relative to the repo root (str)
    :param rung: M0, M1 or M2 (str)
    :param pilot_dir: pilot directory format string, whose parent is the pilot's out_root (str)
    :param spec: the experiment, label, rates and exact-pass fields `_job` takes (dict)
    :returns: the job spec (JobSpec)
    """
    return _job(
        name,
        template,
        rung,
        _same(PILOT_N_HANDS),
        n_seeds=PILOT_N_SEEDS,
        n_indices=None,
        out_root=pilot_dir.rsplit("/", 1)[0],
        pilot_dir=pilot_dir,
        games=_FULL_GAMES,
        gate=True,
        **spec,
    )


_EXPERIMENT_7_JOBS: dict[str, JobSpec] = {
    "m0": _job(
        "m0",
        "configs/run_experiment_recycle_v7_m0.yaml",
        "M0",
        _same(1000),
        n_seeds=1,
        n_indices=INDICES_BASE,
        out_root="data/run_experiment/recycle-v7-m0/i{index}",
        pilot_dir=_M0_PILOT,
        games=_FULL_GAMES,
        label="M0",
        exact_pass=True,
    ),
    "m1": _job(
        "m1",
        "configs/run_experiment_recycle_v7_m1.yaml",
        "M1",
        _same(P2_MIN_HANDS),
        n_seeds=1,
        n_indices=INDICES_BASE,
        out_root="data/run_experiment/recycle-v7-m1/i{index}",
        pilot_dir=_M1_PILOT,
        games=_FULL_GAMES,
        label="M1",
    ),
    "m2": _job(
        "m2",
        "configs/run_experiment_recycle_v7_m2.yaml",
        "M2",
        _same(P2_MIN_HANDS),
        n_seeds=1,
        n_indices=INDICES_BASE,
        out_root="data/run_experiment/recycle-v7-m2/i{index}",
        pilot_dir=_M2_PILOT,
        games=_FULL_GAMES,
        label="M2",
    ),
    "relabel": _job(
        "relabel",
        "configs/run_experiment_recycle_v7_relabel.yaml",
        "M0",
        _same(100),
        n_seeds=1,
        n_indices=INDICES_PROBE,
        out_root=f"{_PROBE_OUT}/relabel/i{{index}}",
        pilot_dir=_M0_PILOT,
        games=_probe_games("relabel"),
        probe="relabel",
    ),
    "payoff_scale": _job(
        "payoff_scale",
        "configs/run_experiment_recycle_v7_payoff_scale.yaml",
        "M0",
        _same(100),
        n_seeds=1,
        n_indices=INDICES_PROBE,
        out_root=f"{_PROBE_OUT}/payoff_scale/i{{index}}",
        pilot_dir=_M0_PILOT,
        games=_probe_games("payoff_scale"),
        probe="payoff_scale",
    ),
    "offpath": _job(
        "offpath",
        "configs/run_experiment_recycle_v7_offpath.yaml",
        "M0",
        _same(60),
        n_seeds=1,
        n_indices=INDICES_PROBE,
        out_root=f"{_PROBE_OUT}/offpath-{{game}}/i{{index}}",
        pilot_dir=_M0_PILOT,
        games=tuple(s.game for s in OFFPATH_SWEEPS),
        probe=OFFPATH_PROBE,
    ),
    **{
        f"{name}_{rung.lower()}": _job(
            f"{name}_{rung.lower()}",
            f"configs/run_experiment_recycle_v7_{name}.yaml",
            rung,
            _same(hands),
            n_seeds=1,
            n_indices=INDICES_PROBE,
            out_root=f"{_PROBE_OUT}-{rung.lower()}/{root}/i{{index}}",
            pilot_dir=pilot_dir,
            games=games,
            probe=probe,
            label=f"{rung} {name}",
        )
        for name, hands, root, games, probe in (
            ("relabel", 100, "relabel", _probe_games("relabel"), "relabel"),
            (
                "payoff_scale",
                100,
                "payoff_scale",
                _probe_games("payoff_scale"),
                "payoff_scale",
            ),
            (
                "offpath",
                60,
                "offpath-{game}",
                tuple(sweep.game for sweep in OFFPATH_SWEEPS),
                OFFPATH_PROBE,
            ),
        )
        for rung, pilot_dir in (("M1", _M1_PILOT), ("M2", _M2_PILOT))
    },
    **{
        f"{rung.lower()}_null": _job(
            f"{rung.lower()}_null",
            f"configs/run_experiment_recycle_v7_{rung.lower()}_null.yaml",
            rung,
            _same(100),
            n_seeds=1,
            n_indices=1,
            out_root=f"data/run_experiment/recycle-v7-null-{rung.lower()}/i{{index}}",
            pilot_dir=pilot_dir,
            games=_FULL_GAMES,
            label=f"{rung} null",
            diagnostic=True,
        )
        for rung, pilot_dir in (("M1", _M1_PILOT), ("M2", _M2_PILOT))
    },
    "pilot_m0": _pilot_kind(
        "pilot_m0",
        "configs/run_experiment_recycle_v7_m0.yaml",
        "M0",
        _M0_PILOT,
        exact_pass=True,
    ),
    "pilot_m1": _pilot_kind(
        "pilot_m1", "configs/run_experiment_recycle_v7_m1.yaml", "M1", _M1_PILOT
    ),
    "pilot_m2": _pilot_kind(
        "pilot_m2", "configs/run_experiment_recycle_v7_m2.yaml", "M2", _M2_PILOT
    ),
}


def _v8_kind(
    counterpart: str,
    template: str,
    out_root: str,
    pilot_dir: str,
    *,
    label: str,
    exact_pass: bool,
    requires_exact_pass: bool = False,
) -> JobSpec:
    """
    an experiment 8 kind. Its hands, seeds, indices, games, memory level and probe type are read
    from its experiment 7 counterpart, so they cannot drift; it names its own template, roots, label
    and rates
    :param counterpart: the experiment 7 kind it mirrors (str)
    :param template: the experiment 8 template path (str)
    :param out_root: out_root format string under a recycle-v8 root (str)
    :param pilot_dir: pilot directory format string under a recycle-v8 root (str)
    :param label: the status-table name, which names the experiment (str)
    :param exact_pass: True when the template runs the in-job exact-policy pass (bool)
    :param requires_exact_pass: True for a probe kind refused while its pass is off (bool)
    :returns: the job spec (JobSpec)
    """
    base = _EXPERIMENT_7_JOBS[counterpart]
    return _job(
        f"v8_{counterpart}",
        template,
        base.rung,
        base.n_hands,
        n_seeds=base.n_seeds,
        n_indices=base.n_indices,
        out_root=out_root,
        pilot_dir=pilot_dir,
        games=base.games,
        probe=base.probe,
        gate=base.gate,
        experiment=EXPERIMENT_8,
        label=label,
        exact_pass=exact_pass,
        requires_exact_pass=requires_exact_pass,
        rates=EXPERIMENT_8_RATES,
        diagnostic=base.diagnostic,
    )


def _v8_pilot(rung_key: str, pilot_dir: str, *, exact_pass: bool) -> JobSpec:
    """
    an experiment 8 gate-pilot kind: 20 hands x 5 seeds at its own template's temperature
    :param rung_key: m0, m1 or m2 (str)
    :param pilot_dir: pilot directory format string under data/pilot/recycle-v8-* (str)
    :param exact_pass: True when the template runs the in-job exact-policy pass (bool)
    :returns: the job spec (JobSpec)
    """
    return _pilot_kind(
        f"v8_pilot_{rung_key}",
        f"configs/run_experiment_recycle_v8_{rung_key}.yaml",
        rung_key.upper(),
        pilot_dir,
        experiment=EXPERIMENT_8,
        label=f"v8 pilot {rung_key.upper()}",
        exact_pass=exact_pass,
        rates=EXPERIMENT_8_RATES,
    )


_EXPERIMENT_8_JOBS: dict[str, JobSpec] = {
    "v8_m0": _v8_kind(
        "m0",
        "configs/run_experiment_recycle_v8_m0.yaml",
        "data/run_experiment/recycle-v8-m0/i{index}",
        _V8_M0_PILOT,
        label="v8 M0",
        exact_pass=True,
    ),
    "v8_m1": _v8_kind(
        "m1",
        "configs/run_experiment_recycle_v8_m1.yaml",
        "data/run_experiment/recycle-v8-m1/i{index}",
        _V8_M1_PILOT,
        label="v8 M1",
        exact_pass=False,
    ),
    "v8_m2": _v8_kind(
        "m2",
        "configs/run_experiment_recycle_v8_m2.yaml",
        "data/run_experiment/recycle-v8-m2/i{index}",
        _V8_M2_PILOT,
        label="v8 M2",
        exact_pass=False,
    ),
    "v8_relabel": _v8_kind(
        "relabel",
        "configs/run_experiment_recycle_v8_relabel.yaml",
        f"{_V8_PROBE_OUT}/relabel/i{{index}}",
        _V8_M0_PILOT,
        label="v8 relabel",
        exact_pass=True,
        requires_exact_pass=True,
    ),
    "v8_payoff_scale": _v8_kind(
        "payoff_scale",
        "configs/run_experiment_recycle_v8_payoff_scale.yaml",
        f"{_V8_PROBE_OUT}/payoff_scale/i{{index}}",
        _V8_M0_PILOT,
        label="v8 payoff_scale",
        exact_pass=True,
        requires_exact_pass=True,
    ),
    "v8_offpath": _v8_kind(
        "offpath",
        "configs/run_experiment_recycle_v8_offpath.yaml",
        f"{_V8_PROBE_OUT}/offpath-{{game}}/i{{index}}",
        _V8_M0_PILOT,
        label="v8 offpath",
        exact_pass=True,
        requires_exact_pass=True,
    ),
    **{
        f"v8_{name}_{rung.lower()}": _v8_kind(
            f"{name}_{rung.lower()}",
            f"configs/run_experiment_recycle_v8_{name}.yaml",
            f"{_V8_PROBE_OUT}-{rung.lower()}/{root}/i{{index}}",
            pilot_dir,
            label=f"v8 {rung} {name}",
            exact_pass=False,
        )
        for name, root in (
            ("relabel", "relabel"),
            ("payoff_scale", "payoff_scale"),
            ("offpath", "offpath-{game}"),
        )
        for rung, pilot_dir in (("M1", _V8_M1_PILOT), ("M2", _V8_M2_PILOT))
    },
    **{
        f"v8_{rung.lower()}_null": _v8_kind(
            f"{rung.lower()}_null",
            f"configs/run_experiment_recycle_v8_{rung.lower()}_null.yaml",
            f"data/run_experiment/recycle-v8-null-{rung.lower()}/i{{index}}",
            pilot_dir,
            label=f"v8 {rung} null",
            exact_pass=False,
        )
        for rung, pilot_dir in (("M1", _V8_M1_PILOT), ("M2", _V8_M2_PILOT))
    },
    "v8_pilot_m0": _v8_pilot("m0", _V8_M0_PILOT, exact_pass=True),
    "v8_pilot_m1": _v8_pilot("m1", _V8_M1_PILOT, exact_pass=False),
    "v8_pilot_m2": _v8_pilot("m2", _V8_M2_PILOT, exact_pass=False),
}

JOBS: dict[str, JobSpec] = {**_EXPERIMENT_7_JOBS, **_EXPERIMENT_8_JOBS}
