#!/usr/bin/env python
"""Collection job plans: which model, kind, seed and sizes one job runs.

One job is one seed of one model of one kind on one H200. The model, job, size and path
tables, and the wall-clock and walltime estimates (at most 12 hours), live in
`lmopt.orchestration.roster`. This script renders the templated run_experiment config
for a job, so the plan a person reads locally is the one the job script runs.

Usage:
    .venv/bin/python scripts/collection_job.py plan --model qwen --job m1 --index 3
    .venv/bin/python scripts/collection_job.py render --model qwen --job m1 --index 3 --port 8123 --out "$TMPDIR"

``plan`` prints the plan as JSON. ``render`` writes one config file per sweep
into ``--out`` and prints ``KEY=value`` shell assignments (single-quoted) that
the job script sources.

Experiment 8 is nine kinds of its own (``v8_m0`` ... ``v8_pilot_m2``) in the same table. ``plan``
refuses an experiment 7 kind unless ``LMOPT_ALLOW_EXPERIMENT_7=1`` is set in that one command's
environment, which keeps a mistyped ``m0`` for ``v8_m0`` from reaching experiment 7's names.
``render`` never refuses on that ground.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

REPO = Path(__file__).resolve().parent.parent

from lmopt.orchestration.roster import (  # noqa: F401
    BASE_SEED,
    CHAIN_HANDS,
    CHAIN_MIN_M0,
    EXPERIMENT_7,
    EXPERIMENT_7_OVERRIDE,
    EXPERIMENT_7_RATES,
    EXPERIMENT_8,
    EXPERIMENT_8_RATES,
    JOBS,
    JobSpec,
    LEVEL_RATIO,
    MODELS,
    OFFPATH_PROBE,
    OFFPATH_SWEEPS,
    P2_MIN_HANDS,
    PLACEHOLDER,
    SELECT,
    STARTUP_MIN,
    WALLTIME,
    _job,
    walltime_for,
)


@dataclass(frozen=True)
class Plan:
    """
    everything one job runs, resolved
    :param model: model key (str)
    :param job: job kind key (str)
    :param index: array index or None (int or None)
    :param model_id: hub id (str)
    :param revision: pinned weights revision (str)
    :param max_model_len: served window passed to vllm serve (int)
    :param short: out_dir key of the model (str)
    :param seed: seed of this job (int)
    :param n_hands: hands per cell (int)
    :param n_seeds: seeds inside the job (int)
    :param out_roots: one out_root per sweep (list of str)
    :param pilot_dir: pilot directory the arm is gated on, or written for a pilot kind (str)
    :param configs: rendered config YAML text, one per sweep (list of str)
    :param est_min: estimated wall-clock minutes (int)
    :param basis: measured or inferred (str)
    :param pending_pilot: True where the estimate waits on an index-0 job to measure it (bool)
    :param p2_licensed: whether the size reaches P2_MIN_HANDS, None where the kind does not feed P2 (bool or None)
    :param select: PBS select string (str)
    :param walltime: PBS walltime derived from est_min, passed on the qsub line (str)
    :param qsub_name: PBS job name (str)
    :param sweep_argv: one command per sweep (list of list of str)
    :param gate_argv: pilot_gate command for pilot kinds, else None (list of str or None)
    """

    model: str
    job: str
    index: int | None
    model_id: str
    revision: str
    max_model_len: int
    short: str
    seed: int
    n_hands: int
    n_seeds: int
    out_roots: list[str]
    pilot_dir: str
    configs: list[str]
    est_min: int
    basis: str
    pending_pilot: bool
    p2_licensed: bool | None
    select: str
    walltime: str
    qsub_name: str
    sweep_argv: list[list[str]]
    gate_argv: list[str] | None


def config_name(k: int) -> str:
    """
    file name of the k-th rendered sweep config
    :param k: sweep position (int)
    :returns: file name (str)
    """
    return f"config_{k}.yaml"


def _check_index(spec: JobSpec, job: str, index: int | None) -> None:
    """
    refuse a missing, extra or out-of-range array index
    :param spec: the job spec (JobSpec)
    :param job: job kind key (str)
    :param index: requested index or None (int or None)
    :returns: None; raises ValueError on a bad index
    """
    if spec.n_indices is None:
        if index is not None:
            raise ValueError(f"job {job!r} takes no index, got {index}")
        return
    if index is None:
        raise ValueError(f"job {job!r} needs an index in 0-{spec.n_indices - 1}")
    if not 0 <= index < spec.n_indices:
        raise ValueError(
            f"index {index} out of range for job {job!r}: 0-{spec.n_indices - 1}"
        )


def _substitute(text: str, mapping: dict[str, str]) -> str:
    """
    replace placeholders outside comment lines, then refuse any left over
    :param text: template text (str)
    :param mapping: placeholder to value (dict of str to str); a value that is
        itself a placeholder is left in place on purpose and not refused
    :returns: rendered text (str)
    """
    kept = {v for v in mapping.values() if PLACEHOLDER.fullmatch(v)}
    out = []
    for line in text.splitlines(keepends=True):
        if not line.lstrip().startswith("#"):
            for key, value in mapping.items():
                line = line.replace(key, value)
            left = [t for t in PLACEHOLDER.findall(line) if t not in kept]
            if left:
                raise ValueError(
                    f"unsubstituted placeholder(s) {left} in {line.strip()!r}"
                )
        out.append(line)
    return "".join(out)


def _flow_list(items) -> str:
    """
    a YAML flow list of the items
    :param items: values to list (iterable)
    :returns: text such as [a, b] (str)
    """
    return "[" + ", ".join(str(i) for i in items) + "]"


def _top_k_text(value: int | None) -> str:
    """
    the YAML scalar a template's action_top_logprobs line takes
    :param value: the model's top-k, or None when the model has none set (int or None)
    :returns: null, or the integer's digits (str)
    """
    return "null" if value is None else str(value)


def plan(
    model: str,
    job: str,
    index: int | None = None,
    port: str = "__PORT__",
    config_dir: str = "$TMPDIR",
) -> Plan:
    """
    resolve one job into its configs, output roots, commands and estimate
    :param model: model key, qwen, olmo or mistral (str)
    :param job: job kind key from JOBS (str)
    :param index: array index for seeded kinds, None for pilot kinds (int or None)
    :param port: vLLM port written into the configs (str)
    :param config_dir: directory the sweep commands read the configs from (str)
    :returns: the resolved plan (Plan); raises ValueError on an unknown model or job,
        a bad index, or a walltime above the 12-hour ceiling
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODELS)}")
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; expected one of {sorted(JOBS)}")
    spec = JOBS[job]
    _check_index(spec, job, index)
    m = MODELS[model]
    seed = BASE_SEED + (index or 0)
    n_hands = spec.n_hands[model]
    games = spec.model_games[model]
    est_min = spec.est_min[model]
    try:
        walltime = walltime_for(est_min)
    except ValueError as exc:
        raise ValueError(f"model {model!r} job {job!r}: {exc}") from None
    pilot_dir = spec.pilot_dir.format(short=m.short)
    template = (REPO / spec.template).read_text()
    common = {
        "__MODEL_ID__": m.model_id,
        "__REVISION__": m.revision,
        "__PORT__": port,
        "__SEED__": str(seed),
        "__N_HANDS__": str(n_hands),
        "__N_SEEDS__": str(spec.n_seeds),
        "__PILOT_DIR__": pilot_dir,
        "__MEMORY_RUNG__": spec.rung,
        "__ACTION_TOP_LOGPROBS__": _top_k_text(spec.action_top_logprobs[model]),
        "__EXACT_POLICY_ENABLED__": "true" if spec.exact_pass else "false",
    }
    per_sweep: list[dict[str, str]] = []
    out_roots: list[str] = []
    if spec.probe == OFFPATH_PROBE:
        for sweep in OFFPATH_SWEEPS:
            root = spec.out_root.format(index=index, game=sweep.game)
            out_roots.append(root)
            per_sweep.append(
                {
                    "__OUT_ROOT__": root,
                    "__GAME__": sweep.game,
                    "__DRIVE_OPENINGS__": _flow_list(sweep.drive_openings),
                    "__SEAT_MODE__": sweep.seat_mode,
                }
            )
    else:
        root = spec.out_root.format(index=index)
        out_roots.append(root)
        per_sweep.append({"__OUT_ROOT__": root, "__GAMES__": _flow_list(games)})
    configs = [_substitute(template, {**common, **extra}) for extra in per_sweep]
    skip = ["--skip-pilot-check"] if spec.gate else []
    # An off-path job runs one sweep per game, so a resubmit for one game meets the games
    # that already landed: those skip with success instead of refusing with exit 1. Experiment 7
    # is closed and its renders are pinned, so only later experiments carry the flag.
    if spec.probe == OFFPATH_PROBE and spec.experiment != EXPERIMENT_7:
        skip.append("--skip-collected")
    sweep_argv = [
        [
            "python",
            "scripts/run_experiment.py",
            "--config",
            f"{config_dir}/{config_name(k)}",
            *skip,
        ]
        for k in range(len(configs))
    ]
    gate_argv = None
    if spec.gate:
        gate_argv = [
            "python",
            "scripts/pilot_gate.py",
            "--config",
            f"{config_dir}/{config_name(0)}",
            "--model",
            m.model_id,
            "--out-dir",
            pilot_dir,
            "--skip-sweep",
        ]
    name = f"v7-{job}-{model}" + ("" if index is None else f"-i{index}")
    return Plan(
        model=model,
        job=job,
        index=index,
        model_id=m.model_id,
        revision=m.revision,
        max_model_len=m.max_model_len,
        short=m.short,
        seed=seed,
        n_hands=n_hands,
        n_seeds=spec.n_seeds,
        out_roots=out_roots,
        pilot_dir=pilot_dir,
        configs=configs,
        est_min=est_min,
        basis=spec.basis[model],
        pending_pilot=spec.pending_pilot[model],
        p2_licensed=spec.p2_licensed[model],
        select=SELECT,
        walltime=walltime,
        qsub_name=name.replace("_", "-"),
        sweep_argv=sweep_argv,
        gate_argv=gate_argv,
    )


def refuse_experiment_7(job: str, environ: dict[str, str] | None = None) -> None:
    """
    refuse an experiment 7 kind unless the operator named the override for this command
    A valid kind typed for the wrong experiment (`m0` for `v8_m0`) would otherwise pass `plan`
    and let the preflight move experiment 7's logs aside, so the refusal sits in the one call the
    submit makes on the local machine before anything touches the cluster. `render` does not call it:
    the cluster's preflight and the job call `render`, and a job already submitted must still render
    :param job: job kind key (str)
    :param environ: the environment to read, os.environ when None (dict or None)
    :returns: None; raises ValueError for an experiment 7 kind with the override unset
    """
    env = os.environ if environ is None else environ
    spec = JOBS.get(job)
    if spec is None or spec.experiment != EXPERIMENT_7:
        return
    if env.get(EXPERIMENT_7_OVERRIDE) == "1":
        return
    eight = sorted(k for k, v in JOBS.items() if v.experiment == EXPERIMENT_8)
    raise ValueError(
        f"job {job!r} is an experiment 7 kind and experiment 7 is complete. Experiment 8's "
        f"kinds are {eight}. To act on experiment 7 anyway, set {EXPERIMENT_7_OVERRIDE}=1 in the "
        f"environment of this one command"
    )


def _exact_pass_enabled(config_text: str) -> bool:
    """
    whether a rendered config turns the in-job exact-policy pass on
    :param config_text: one sweep's rendered YAML (str)
    :returns: True only when exact_policy.enabled is the boolean true (bool); an absent block or
        key reads as off, which is what the runner does with it
    """
    block = yaml.safe_load(config_text).get("exact_policy") or {}
    return block.get("enabled") is True


def checked_plan(
    model: str,
    job: str,
    index: int | None = None,
    port: str = "__PORT__",
    config_dir: str = "$TMPDIR",
) -> Plan:
    """
    `plan`, refused for an experiment 8 probe kind whose rendered config has the exact pass off
    The CLI's `plan` and `render` call this; the pure `plan` the tests call does not, so a test can
    still render a kind that cannot be submitted
    :param model: model key (str)
    :param job: job kind key from JOBS (str)
    :param index: array index or None (int or None)
    :param port: vLLM port written into the configs (str)
    :param config_dir: directory the sweep commands read the configs from (str)
    :returns: the resolved plan (Plan); raises ValueError as `plan` does, and when the kind
        requires the exact pass and a rendered sweep config has it off
    """
    resolved = plan(model, job, index, port=port, config_dir=config_dir)
    if JOBS[job].requires_exact_pass and not all(
        _exact_pass_enabled(text) for text in resolved.configs
    ):
        raise ValueError(
            f"job {job!r} requires the in-job exact-policy pass and its rendered config has "
            f"exact_policy.enabled off. It cannot be planned or submitted until the template turns "
            f"the pass on"
        )
    return resolved


def _sq(value: object) -> str:
    """
    single-quote a value for a shell assignment
    :param value: the value (any, rendered with str)
    :returns: the single-quoted text (str)
    """
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def render(model: str, job: str, index: int | None, port: str, out: Path) -> str:
    """
    write one config file per sweep into out and return the shell assignments
    :param model: model key (str)
    :param job: job kind key (str)
    :param index: array index or None (int or None)
    :param port: vLLM port written into the configs (str)
    :param out: directory the configs are written to (Path)
    :returns: KEY='value' lines, one per line (str)
    """
    out = Path(out)
    p = checked_plan(model, job, index, port=port, config_dir=str(out))
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        ("MODEL_ID", p.model_id),
        ("REVISION", p.revision),
        ("MAX_MODEL_LEN", p.max_model_len),
        ("SEED", p.seed),
        ("N_HANDS", p.n_hands),
        ("N_SEEDS", p.n_seeds),
        ("EST_MIN", p.est_min),
        ("WALLTIME", p.walltime),
        ("QSUB_NAME", p.qsub_name),
        ("PILOT_DIR", p.pilot_dir),
        ("GATE", 1 if p.gate_argv is not None else 0),
        ("N_SWEEPS", len(p.configs)),
    ]
    for k, text in enumerate(p.configs):
        path = out / config_name(k)
        path.write_text(text)
        lines += [
            (f"CONFIG_{k}", path),
            (f"OUT_ROOT_{k}", p.out_roots[k]),
            (f"SWEEP_CMD_{k}", shlex.join(p.sweep_argv[k])),
        ]
    if p.gate_argv is not None:
        lines.append(("GATE_CMD", shlex.join(p.gate_argv)))
    return "\n".join(f"{key}={_sq(value)}" for key, value in lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    parse the plan and render subcommands
    :param argv: arguments without the program name (list of str or None)
    :returns: parsed arguments (argparse.Namespace)
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "render"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", required=True, choices=sorted(MODELS))
        sp.add_argument("--job", required=True, choices=sorted(JOBS))
        sp.add_argument("--index", type=int, default=None)
        if name == "plan":
            sp.add_argument("--port", default="__PORT__")
        else:
            sp.add_argument("--port", required=True)
            sp.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    command line entry: print a plan as JSON or render configs and print assignments
    :param argv: arguments without the program name, None for sys.argv (list of str or None)
    :returns: process exit code, 0 on success and 2 on a refused request (int)
    """
    args = _parse_args(argv)
    try:
        if args.cmd == "plan":
            refuse_experiment_7(args.job)
            resolved = checked_plan(args.model, args.job, args.index, port=args.port)
            print(json.dumps(dataclasses.asdict(resolved), indent=2))
            if resolved.p2_licensed is False:
                print(
                    f"collection_job: P2 is not licensed at {resolved.n_hands} hands per cell "
                    f"(needs {P2_MIN_HANDS})",
                    file=sys.stderr,
                )
        else:
            sys.stdout.write(
                render(args.model, args.job, args.index, args.port, Path(args.out))
            )
    except ValueError as exc:
        print(f"collection_job: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
