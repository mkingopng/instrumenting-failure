#!/usr/bin/env python
"""Full split-topology experiment orchestrator.

Topology: this desktop runs the harness, every non-LM agent, and the solvers
(in-process CPU work inside shard workers); the serving box serves ALL LM
inference (``backend: dgx-spark-vllm`` recorded per row).
Serving is managed outside this repo - this script
only verifies-and-consumes endpoints.

Per enabled model in the roster config:
  0. PILOT GATE - run ``scripts/pilot_gate.py --skip-sweep`` against this
     arm's pilot directory and refuse to launch unless it exits 0. A launch
     precondition, not a courtesy: see the PILOT_GATE_REFUSAL_EXIT_CODE block
     below. Runs first, before preflight and before any sweep subprocess,
     because it is offline, cheap, and needs no endpoint - an operator with
     no pilot learns so immediately rather than after the serving box comes up.
  1. PREFLIGHT - GET <vllm_base_url>/models and assert the expected model id
     is among the served ids. On failure: exit non-zero, printing the
     serving-side docker launch hint. No silent skip (no silent
     fallbacks).
  2. Run ``uv run python scripts/run_sweep.py --config <this same config>
     --model <id> --vllm-base-url <url> --out-dir <out_root>/<model_short>
     --backend <backend>`` - run_sweep's own CLI overrides win over its
     --config YAML, it ignores this config's orchestrator-only keys
     (models:, out_root:) harmlessly, and its run_meta.json records the
     resolved config per run (the provenance source of truth - no generated
     intermediate config file).
  3. Run ``uv run python scripts/diagnose.py <dir>/`` (non-zero propagates).
  4. If ``exact_policy.enabled`` (config over forks; code default
     FALSE when the key is absent, same asymmetry as ``serving.auto_swap`` -
     see that block's comment in ``main()``): run ``uv run python
     scripts/exact_policy_pass.py --data-dir <dir> --base-url <url> --model
     <id> --out <dir>/exact_policy.csv`` against this
     arm's own served endpoint, so each arm gets its own artefact rather than
     one hand-run pass being borrowed by every arm's analysis. A failure here
     is the one deliberate exception to "any subprocess failure aborts the
     whole experiment loudly" below - reported loudly (stderr, tracked, and
     re-surfaced in the final summary) but does not abort: see the inline
     comment at the call site for the reasoning (recoverable diagnostic vs.
     an unrecoverable arm). "Does not abort" is not "exits 0", though -
     see Exit codes, below.

Any other subprocess failure aborts the whole experiment loudly (no
continue-on-error) - with one further exception: a
sweep exiting ``scripts/run_sweep.py``'s SHARD_FAILURE_EXIT_CODE (5) ran to
completion and lost shards. That arm is marked PARTIAL, its diagnose and
exact-policy steps are SKIPPED (both report over whatever Parquet is
present), the loop continues to the next arm, and the experiment ends on
PARTIAL_ARM_EXIT_CODE. A sweep that died any other way still aborts
everything: that is a tool failure and the next arm will probably die the
same way. Analysis is offline: the
analyse_stats.py commands are printed, not run, and a PARTIAL arm gets no
such command printed.

Exit codes (the contract a caller - a cron wrapper, a CI step, an operator's
shell - actually depends on):
  0   - either (a) every enabled arm's sweep+diagnose succeeded, and (if
        exact_policy.enabled) every arm's exact-policy pass succeeded too;
        or (b) ``--dry-run`` was passed - no preflight, no subprocess, no
        arm actually ran at all. ``--dry-run``'s 0 is "the plan printed
        cleanly", not "the experiment completed"; do not conflate the two
        readings of a 0 from this script.
  EXACT_POLICY_FAILURE_EXIT_CODE (3) - every enabled arm's sweep+diagnose
        succeeded (no data lost, nothing aborted early) but at least one
        arm's exact-policy pass failed; see stderr for which model(s) and
        the exact command to re-run by hand. The experiment is otherwise
        complete - this is not "the experiment died".
  PILOT_GATE_REFUSAL_EXIT_CODE (4) - an arm's pilot gate fired: the pilot
        exists, was evaluated, and the arm is unusable. Nothing was
        launched for that model. Distinct in kind from 1 below, which
        covers every case where no verdict exists (no pilot directory, or
        the gate itself broke) - "the arm is measured-bad" and "nothing has
        been measured" call for different operator actions: fix the
        model/config vs. run a pilot.
  PARTIAL_ARM_EXIT_CODE (6) - the loop reached the END (no arm aborted it)
        but at least one arm's sweep LOST SHARDS: it ran to completion,
        recorded the failures in that arm's shard_outcomes.json, and exited
        5. Those arms are named in the final summary, were not diagnosed,
        exact-policy-passed or given an analyse command, and must not be
        read as finished. Every other arm completed normally and its data is
        untouched. Outranks 3 when both occur (missing data beats a missing
        diagnostic); both are printed regardless.
  other non-zero - a config/preflight error before any arm ran (typically
        1, via ``sys.exit``/an explicit ``return 1``), or a sweep/diagnose
        subprocess's own exit code passed through unchanged (``return rc``)
        - the experiment did not complete for at least this arm; distinct
        in kind from the 3 case above (died vs. completed-with-a-diagnostic-
        gap). NOTE this is not a hard numeric guarantee: ``return rc`` is
        an arbitrary passthrough of whatever the sweep/diagnose subprocess
        itself returned, so a literal collision with 3 is possible in
        principle if some future sweep/diagnose failure mode ever returned
        exactly 3 - both ``scripts/run_sweep.py`` and
        ``scripts/diagnose.py`` only ever return 0 or 1, so this is
        unreached in practice, not impossible by construction.
  1, via the signal path - SIGTERM or SIGINT arrived while
        ``stop_on_teardown`` was
        true in the config. Only under that config does ``main`` install
        handlers; they raise ``_TeardownSignal``, whose ``except`` prints an
        ``ABORT:`` line to stderr and returns 1, and the shared ``finally``
        then restores the previous handlers and runs the one teardown call.
        Under any other config Python's default signal behaviour is
        untouched, so this 1 never appears. It is a caller-visible
        distinction: a 1 here means an operator or supervisor killed the
        run and the serving unit was stopped cleanly, not a config or
        preflight error. Only the stderr ``ABORT:`` line separates the two,
        so a wrapper that reads the code alone cannot tell them apart.
        Whichever arm step is
        running when the
        signal arrives is itself torn down first: the child receives the
        forwarded SIGTERM, or already has the group-delivered SIGINT, and
        gets a 15 s grace window to run its own cleanup before it is killed
        (``_run_step``/``_teardown_child``). A further signal inside that
        window is absorbed rather than allowed to abandon the child, so a
        double Ctrl-C returns this same 1 with the child dead, not orphaned.
        Exit codes are unchanged by this; only the child's fate is.

  Residual (documented, not fixed - a real limitation of returning a
  single code from a MULTI-ARM loop): if arm N's exact-policy pass fails
  and a later arm N+1's sweep or diagnose then also fails, the function
  returns at arm N+1's failure with that subprocess's own ``rc`` - the
  earlier arm's ``3``-worthy exact-policy failure is silently absorbed
  into "the experiment did not complete", and the final-summary WARNING
  block (which only prints after the per-model loop finishes) never runs,
  so arm N's diagnostic gap is invisible to both the exit code and the
  summary in this specific ordering. This is correct per this section's
  own "did not complete" wording - arm N+1 genuinely did not complete,
  and that is a strictly worse outcome the exit code is right to
  prioritise - but it means EXACT_POLICY_FAILURE_EXIT_CODE is not a
  complete audit of every exact-policy failure across a multi-arm run;
  only stderr, read in full, is.

Usage:
    set -a; . ./.env; set +a          # exports WANDB_API_KEY
    uv run python scripts/run_experiment.py --config configs/run_experiment.yaml
    uv run python scripts/run_experiment.py --config configs/run_experiment.yaml --dry-run
    uv run python scripts/run_experiment.py --config configs/run_experiment.yaml \
        --models Qwen/Qwen2.5-14B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

# This module's own code does not call urllib.request.urlopen directly
# (preflight() delegates to ServingController.current()). The import is kept
# so `urllib.request` stays a bound attribute of this module: a caller that
# monkeypatches `mod.urllib.request.urlopen` reaches control.py's call too,
# since `urllib.request` is a singleton module, but only if `mod.urllib`
# resolves at all.
from typing import Any, cast

import yaml

# The swap client this orchestrator consumes.
# lmopt.serving.control has zero pyspiel dependency, so importing it here
# does not change which tests in this file need pyspiel (only the
# run_sweep-round-trip contract test does, via its own importorskip).
from lmopt.serving.chat_template import (
    ChatTemplateUnavailable,
    chat_template_pin_key,
    resolve_chat_template_hash,
)

# ServingError is re-exported: tests raise it through this module (mod.ServingError).
from lmopt.serving.control import ServingController
from lmopt.serving.preserved import PreservedServing, PreservedServingError
from lmopt.serving.injob import InJobServing, InJobServingError
from lmopt.provenance import parse_revision_flags
from lmopt.provenance.weights_cache import resolve_prelaunch_revision
from lmopt.logging.exact_policy_pass import FILENAME as _EXACT_POLICY_PASS_FILENAME
from lmopt.logging.exact_policy_pass import count_csv_rows

# Cross-checkout import tripwire. Hard failure, and at
# import so it lands before preflight and before any arm starts: this script
# commits the box to hours of GPU work whose every row records a git sha, and a
# run whose `lmopt` came from another branch records a sha that does not describe
# the code that produced it. That is a provenance breach, not a
# nuisance, and it is invisible in the output.
from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)


def arm_already_finished(out_dir: str, cfg: dict, exact_policy_enabled: bool) -> bool:
    """
    whether an arm's out_dir already holds this arm, finished: every configured cell collected, read
    by the same function `scripts/ladder_state.py` reports the ladder with, so a skip here and a
    COLLECTED there cannot disagree; collected under the games, opponents and hand count this config
    asks for; and its exact-policy pass record written, which happens only after diagnose, with the
    pass returning 0 when this config enables it. An arm that predates the record is not finished
    by this test, so it is refused as before. ladder_state is imported here rather than at module
    top because the repo root is put on sys.path only for this call
    :param out_dir: the arm's run directory (str)
    :param cfg: the loaded experiment config (dict)
    :param exact_policy_enabled: whether this config enables the exact-policy pass (bool)
    :returns: True only when the arm is finished on every count above (bool)
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from pathlib import Path

    from lmopt.logging.exact_policy_pass import PassRecordError, read_record
    from scripts import ladder_state

    if ladder_state.run_status(Path(out_dir)).status != ladder_state.COLLECTED:
        return False
    try:
        with open(os.path.join(out_dir, "run_meta.json")) as fh:
            resolved = json.load(fh)["resolved_config"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    for key in ("games", "opponents"):
        if set(resolved.get(key) or ()) != set(cfg.get(key) or ()):
            return False
    if resolved.get("n_hands") != cfg.get("n_hands"):
        return False
    try:
        record = read_record(out_dir)
    except PassRecordError:
        return False
    if record is None:
        return False
    if exact_policy_enabled:
        return record.enabled and record.rc == 0
    return True


class PreflightError(RuntimeError):
    """Endpoint does not serve the expected model - fail loudly, never skip."""


# "Do not abort the cycle" and "exit 0" are two different decisions. Exiting
# 0 when the exact-policy pass failed on every arm would be a success signal
# with no corresponding work done, placed in the verification layer, exactly
# where an unattended multi-day cycle's only downstream signal is the exit
# code, not the scrollback.
#
# A distinct code, not 0 and not reused from the abort paths above (which
# either `return rc`, an unpredictable passthrough of the sweep/diagnose
# subprocess's own exit code, or `return 1`/`sys.exit(...)` for a config/
# preflight error before any arm ran) -- so a caller can tell "the
# experiment died (no arm completed, or one aborted mid-run)" apart from
# "the experiment completed (every arm's sweep+diagnose succeeded, no data
# lost) but at least one arm's exact-policy pass needs a manual re-run".
# Collapsing those to the same code would defeat the point of distinguishing
# them at all.
EXACT_POLICY_FAILURE_EXIT_CODE = 3

# The pilot gate is a precondition of this script rather than a step an
# operator has to remember: a pilot whose shards silently drop a requested
# game (the gate reports e.g. ``[vacuity] goofspiel: 0.0000 vs threshold
# 100.0000``) must refuse the arm before hours of GPU work start. This is the
# code that says an arm was refused by it.
#
# A distinct code, for the same reason EXACT_POLICY_FAILURE_EXIT_CODE is
# (and pilot_gate.py's own GATE_FAILURE_EXIT_CODE, whose module docstring
# draws the identical line): "the arm is bad" and "the tool broke" must
# never share a code. 4, not 2, because this script's codes are its own
# contract - a caller sees 4 and knows the refusal came from this script's
# precondition, not from a passed-through subprocess rc.
PILOT_GATE_REFUSAL_EXIT_CODE = 4

# scripts/pilot_gate.py's GATE_FAILURE_EXIT_CODE, mirrored rather than
# imported: `scripts` is only importable as a package when the repo root is
# on sys.path, which is not guaranteed for a script invoked by path (and
# tests load this module by file path, not as scripts.run_experiment).
_PILOT_GATE_FAILURE_EXIT_CODE = 2

# An arm that lost SHARDS must not discard every arm behind it.
#
# `scripts/run_sweep.py` records a failed shard, keeps going, writes a
# shard_outcomes.json naming every requested shard and its status, and exits
# 5 - so "this arm is incomplete" is distinguishable from "the tooling is
# broken", and the two must not share an outcome. The precedent for the
# continue is in this file, at the exact-policy pass below: "aborting the
# whole experiment here would throw away every remaining arm's
# sweep+diagnose too".
#
# A distinct code, and specifically not run_sweep's own 5 passed through.
# 5 is already published in this module's docstring under "other non-zero -
# ... the experiment did not complete for at least this arm"; returning it
# from the END of a completed loop would silently give one number two
# meanings depending on which module produced it, which is the confusion the
# 3 and 4 codes exist to prevent. It is also still reachable as a genuine
# passthrough: `scripts/diagnose.py` exiting 5 is an unknown failure of a
# different tool, not a shard loss. 6 is the
# first free value in the namespace this script shares with pilot_gate's 2,
# EXACT_POLICY_FAILURE's 3, PILOT_GATE_REFUSAL's 4 and run_sweep's 5.
#
# It outranks EXACT_POLICY_FAILURE_EXIT_CODE at the final return, because
# one code out of a multi-arm loop has to rank outcomes and missing data
# outranks a missing diagnostic: an exact-policy failure leaves the arm's
# Parquet untouched and is re-runnable by hand, a partial arm's rows do not
# exist. Same direction as run_sweep's own rule that 5 outranks its
# schema-violation exit 1. Both are always printed; the code ranks them, it
# does not hide either.
PARTIAL_ARM_EXIT_CODE = 6

# scripts/run_sweep.py's SHARD_FAILURE_EXIT_CODE and SHARD_OUTCOMES_FILENAME,
# mirrored rather than imported, for the reason given at
# _PILOT_GATE_FAILURE_EXIT_CODE above (`scripts` is not importable as a
# package for a script invoked by path) and for a second reason that case
# does not have: loading run_sweep.py pulls
# lmopt.games.loaders and therefore `pyspiel` (692 modules). This module is
# deliberately pyspiel-free, and pyspiel
# installs on Linux only, so importing run_sweep would make
# this script unimportable on macOS to save mirroring two
# values.
_SWEEP_PARTIAL_EXIT_CODE = 5
_SHARD_OUTCOMES_FILENAME = "shard_outcomes.json"


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full split-topology experiment: per enabled model, "
        "preflight the serving endpoint, then run_sweep + diagnose."
    )
    p.add_argument(
        "--config",
        required=True,
        help="Roster-driven experiment YAML (configs/run_experiment.yaml)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-model plan and resolved commands without "
        "executing anything (no network, no subprocesses, no writes).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of the ENABLED roster to run (model ids). "
        "Naming a disabled/unknown model fails loudly - enable it "
        "in the config first.",
    )
    p.add_argument(
        "--allow-existing",
        action="store_true",
        help="Proceed even if a model's out_dir already contains "
        "*.parquet files (default: exit loudly - stale-data "
        "pollution guard).",
    )
    p.add_argument(
        "--skip-collected",
        action="store_true",
        help="Skip, loudly and with success, an arm whose out_dir already "
        "holds it finished: COLLECTED as scripts/ladder_state.py reads it, "
        "under this config's games, opponents and n_hands, with an "
        "exact_policy_pass.json that shows the pass returned 0 when this "
        "config enables it. Any other out_dir holding parquet is refused.",
    )
    p.add_argument(
        "--pilot-dir",
        default=None,
        help="Pilot data directory for the launch precondition. "
        "Overrides the roster entry's own 'pilot_dir' key and "
        "the derived default (data/pilot/<model_short>). Valid "
        "only when exactly one model is selected - one arm's "
        "pilot must never gate another's.",
    )
    p.add_argument(
        "--skip-pilot-check",
        action="store_true",
        help="Launch WITHOUT a passing pilot gate. Loud, and "
        "RECORDED in this run's output (per arm and in the "
        "final summary) so a reader can tell a skipped gate "
        "from a passed one. A skipped gate is not a passed "
        "gate.",
    )
    args = p.parse_args(argv)
    if args.skip_collected and args.allow_existing:
        p.error(
            "--skip-collected and --allow-existing contradict each other: one skips a "
            "finished arm, the other writes over an unfinished one"
        )
    return args


def model_short(model_id: str) -> str:
    """'Qwen/Qwen2.5-14B-Instruct' -> 'qwen2.5-14b-instruct' (out_root subdir key)."""
    return model_id.rsplit("/", 1)[-1].lower()


def enabled_models(cfg: dict, subset: list[str] | None = None) -> list[dict]:
    """Validated enabled roster entries, optionally narrowed to an id subset.

    Fails loudly (exit non-zero) on: an enabled entry missing ``id`` or
    ``vllm_base_url``; two enabled ids deriving the same model_short (their
    out_dirs would silently overwrite each other); a subset id that is not an
    ENABLED roster entry (running a disabled model behind the config's back,
    or silently dropping a typo, would both be silent-fallback behaviour).
    """
    roster = cfg.get("models") or []
    enabled: list[dict] = []
    for idx, m in enumerate(roster):
        if not m.get("enabled"):
            continue
        missing = [k for k in ("id", "vllm_base_url") if not m.get(k)]
        if missing:
            sys.exit(
                f"config error: models[{idx}] ({m.get('id', '<no id>')!r}) is "
                f"enabled but missing {missing}"
            )
        enabled.append(m)

    by_short: dict[str, str] = {}
    for m in enabled:
        short = model_short(m["id"])
        if short in by_short:
            sys.exit(
                f"config error: model_short collision - {by_short[short]!r} and "
                f"{m['id']!r} both derive out_dir key {short!r}; their outputs "
                "would overwrite each other."
            )
        by_short[short] = m["id"]

    if subset is None:
        return enabled
    by_id = {m["id"]: m for m in enabled}
    unknown = [mid for mid in subset if mid not in by_id]
    if unknown:
        sys.exit(
            f"--models: {unknown} not in the enabled roster.\n"
            f"Enabled models: {sorted(by_id)}\n"
            "Enable the entry in the config first (no silent skip)."
        )
    return [by_id[mid] for mid in subset]


def pilot_dir_for(entry: dict, override: str | None = None) -> str:
    """Which directory holds this arm's pilot, in precedence order.

    1. ``--pilot-dir`` - the operator's explicit one-off (main() refuses it
       when more than one model is selected).
    2. the roster entry's own ``pilot_dir`` key - config over forks, and
       the one that makes this precondition usable at all: pilot directory
       names (e.g. mistral-nemo, olmo2-13b, qwen14b-mw80) need not match
       ``model_short()``. A derive-only precondition would refuse such arms
       and push the operator straight to --skip-pilot-check, defeating
       itself.
    3. ``data/pilot/<model_short>`` - the naming convention for new pilots.

    Deliberately not a search over data/pilot/ for a plausible match: a
    guessed directory is exactly the cross-arm misattribution
    pilot_gate.py's own ``_verify_pilot_provenance`` exists to catch, and a
    precondition that guesses its own evidence is not a precondition.
    """
    if override:
        return override
    configured = entry.get("pilot_dir")
    if configured:
        return configured
    return os.path.join("data", "pilot", model_short(entry["id"]))


def pilot_gate_command(
    config_path: str, model_id: str, pilot_dir: str, *, skip_sweep: bool = True
) -> list[str]:
    """``scripts/pilot_gate.py`` invocation for one arm's launch precondition.

    ``--config`` is this experiment's config, so the gate resolves the same
    games/n_seeds shape the sweep is about to run - its non-vacuity
    criterion is "every requested game has >= 100 LM rows", and which games
    were requested is a property of this config, not of the gate.

    ``--skip-sweep`` (default) evaluates the pilot that already exists.
    Without it the gate would run a fresh GPU pilot on every launch, which
    is both the cost this precondition exists to avoid and an immediate
    refusal from pilot_gate.py's own non-empty --out-dir guard. The keyword
    is exposed so the "run a pilot first" hint printed on an absent
    directory is built by this same function rather than hand-assembled
    into a string that can drift from what the gate actually accepts.

    The first argv element is `sys.executable`, for the same reason as
    `sweep_command` below: no `uv` on the cluster's path. Gate-pilot-kind jobs
    pass `--skip-pilot-check`, so main() never reaches the `_run_step(p_cmd)`
    call that runs this command for them, but every real collection launch
    (m0/m1/m2/relabel/payoff_scale/offpath) does reach it, on the step after
    preflight, before the sweep.
    """
    cmd = [
        sys.executable,
        "scripts/pilot_gate.py",
        "--config",
        config_path,
        "--model",
        model_id,
        "--out-dir",
        pilot_dir,
    ]
    if skip_sweep:
        cmd.append("--skip-sweep")
    return cmd


def _docker_hint(model_id: str) -> str:
    return (
        "Launch it on the serving box first:\n"
        f"  docker run ... vllm/vllm-openai@<digest> {model_id} "
        "--dtype bfloat16 --max-model-len <max_model_len of the model in configs/models.yaml> "
        "--gpu-memory-utilization 0.35 --enforce-eager"
    )


def preflight(base_url: str, model_id: str, timeout: float = 30.0) -> list[str]:
    """Assert <base_url>/models serves model_id; return the served-id list.

    Raises PreflightError (message carries the launch hint) on an
    unreachable endpoint, an unparseable/unexpected response shape, or a
    wrong served model - never skips. Membership is exact-id: a superstring
    variant (e.g. pythia-12b-deduped when pythia-12b is expected) fails.

    Delegates the actual /v1/models check to
    ServingController.current() (lmopt.serving.control) rather than keeping
    a second, independent HTTP-parsing implementation - this function's own
    docstring is what control.py's ServingController.current() cites back
    (see that method's docstring) as the line it mirrors, so the two must
    not drift apart. `ssh_host=""` below is not a real host: current() is
    HTTP-only and never shells out, so no ssh_host is needed for this call -
    only swap_to()/resolve() (used by main()'s auto-swap path, below) touch
    the box over ssh.

    `current()` returns None only for an unreachable endpoint (never raises
    for that case) and raises ServingControlError for an
    answering-but-unparseable one - this function folds both into one
    PreflightError carrying the docker launch hint, and a wrong-but-parseable
    served model is a separate PreflightError naming what was actually
    served. `timeout` is kept for signature compatibility but is not threaded
    through to the HTTP call: ServingController.current() fixes its own
    internal timeout (control.py's `_HTTP_TIMEOUT_S`) and exposes no override
    knob. No caller in this repo passes a non-default `timeout`, but a caller
    relying on a custom timeout here would silently not get it.
    """
    url = base_url.rstrip("/") + "/models"
    controller = ServingController(ssh_host="", base_url=base_url)
    try:
        served_model = controller.current()
    except Exception as exc:
        # current()'s own except clauses (in lmopt.serving.control) catch
        # urllib.error.HTTPError/URLError and OSError, but not
        # http.client.HTTPException subclasses (e.g. IncompleteRead) raised
        # by resp.read() mid-response - exactly what a box mid-swap/restart
        # looks like: a truncated response. The broad catch here, at the
        # run_experiment.py boundary, keeps preflight()'s contract ("never a
        # raw exception, always a PreflightError with the launch hint")
        # without touching control.py. Deliberately `except Exception` (not
        # just `ServingControlError`): a narrower catch would let a raw
        # IncompleteRead escape with no launch hint.
        raise PreflightError(
            f"PREFLIGHT FAILED for {model_id}: cannot reach or parse {url} "
            f"({exc!r}).\n" + _docker_hint(model_id)
        ) from exc
    if served_model is None:
        raise PreflightError(
            f"PREFLIGHT FAILED for {model_id}: cannot reach or parse {url} "
            f"(endpoint unreachable).\n" + _docker_hint(model_id)
        )
    served = list(served_model.served_ids)
    if model_id not in served:
        raise PreflightError(
            f"PREFLIGHT FAILED for {model_id}: {url} serves {served!r} - "
            f"expected {model_id!r} among them.\n" + _docker_hint(model_id)
        )
    return served


def sweep_command(
    config_path: str,
    entry: dict,
    out_dir: str,
    backend: str,
    manifest_path: str | None = None,
    served_weights_revision: str | None = None,
    tokenizer_revision: str | None = None,
    git_sha_override: str | None = None,
    expected_chat_template_hash: str | None = None,
    action_max_tokens: int | None = None,
    belief_max_tokens: int | None = None,
) -> list[str]:
    """run_sweep invocation: CLI overrides win over the shared --config YAML.

    The first argv element is `sys.executable`, not `["uv", "run",
    "python", ...]`: the cluster's job environment has no `uv` on path, so a
    `uv run` shape here makes a launched sweep fail with
    `FileNotFoundError: 'uv'` right after vLLM has come up, before any
    hand is played. `sys.executable` is the same venv python this very
    process (run_experiment.py) is running under, so it needs no path
    lookup at all; it is also the interpreter a cluster job script already
    runs every other step with (`source .venv/bin/activate`, then bare
    `python`). Putting `uv` on path inside the job script instead was
    considered and rejected: it would add a second way to reach the same
    interpreter, install a tool this process never otherwise needs, and
    leave the identical failure live for any caller of run_experiment.py
    not launched through a job script's activation step.
    `pilot_gate_command`, `diagnose_command` and `exact_policy_pass_command`
    use `sys.executable` for the identical reason: of the command
    builders in this module, these four are executed by a real cluster job
    rather than merely printed (see each one's own docstring).
    `analyse_stats_command` is print-only and stays `uv`-wrapped.

    --backend is passed explicitly from the experiment config: run_sweep's
    own default is cuda-3090-vllm, and a silently defaulted backend would
    corrupt the per-row provenance pin.

    manifest_path: when this model's endpoint was
    swapped or resolved without a swap by main() (below), the box-only serving fields (image_ref,
    vllm_args, container, unit) it read back are written to a manifest file
    and its path is appended here as --serving-manifest, so run_sweep.py's
    run_meta serving block carries them alongside what it resolves itself
    over HTTP. None (default - a config with no ssh_host configured, and
    every standalone run_sweep invocation) omits the flag entirely;
    run_sweep.py then leaves those box-only fields explicitly absent
    rather than fabricating them (no silent fallbacks).

    served_weights_revision:
    the HF weights revision main() resolved for this arm off the serving
    box's cache (_resolve_served_weights_revision below), appended as
    --served-weights-revision so run_sweep stamps it on every LM row.
    None omits the flag - never a fabricated value (no silent
    fallbacks); run_sweep's own startup gate is what decides that a real
    sweep without one must abort.

    tokenizer_revision: the HF
    tokenizer revision main() parsed from a pbs-injob arm's actually-
    observed launch argv (parse_revision_flags over resolved.vllm_args),
    appended as --tokenizer-revision so run_sweep stamps it on every LM
    row. None omits the flag (the common case: no distinct tokenizer
    revision was ever observed) - run_sweep applies no startup gate to
    this one, since there is no declared config field it must agree with.

    git_sha_override: the model_revision main() read off an existing shard in
    out_dir when this call is a --allow-existing resume of an arm that
    already has landed shards (_existing_shard_git_sha below), appended as
    --git-sha-override so the resumed run_sweep.py invocation reuses that
    sha instead of resolving a fresh one at its own startup -- without this,
    a resume launched after a HEAD move would stamp the same arm's later
    shards with a different model_revision than its earlier ones, so a
    per-row sha would name a tree that never executed those rows. None
    (default -- a fresh arm, or --allow-existing with no detectable prior
    sha) omits the flag, and run_sweep.py resolves fresh.

    expected_chat_template_hash: the per-model pin main() looked up in
    configs/models.yaml (_expected_chat_template_hash below), appended as
    --expected-chat-template-hash so run_sweep can compare it against the
    hash it resolves from the live endpoint at startup (mismatch aborts
    there). None omits the flag - an unpinned model must reach run_sweep
    with no pin, so run_sweep's own loud NOT-pinned line fires; never a
    fabricated or stringified-None pin (no silent fallbacks).

    action_max_tokens / belief_max_tokens: per-model
    decode budgets from configs/models.yaml's fullsize_models entries, resolved
    by _model_decode_budgets and appended as CLI flags when present. None omits
    the flags; run_sweep's overrides
    dict feeds them to the agent when set, else the agent's constructor
    defaults apply (ACTION_MAX_TOKENS / BELIEF_MAX_TOKENS in lm_vllm.py).
    """
    cmd = [
        sys.executable,
        "scripts/run_sweep.py",
        "--config",
        config_path,
        "--model",
        entry["id"],
        "--vllm-base-url",
        entry["vllm_base_url"],
        "--out-dir",
        out_dir,
        "--backend",
        backend,
    ]
    if manifest_path:
        cmd += ["--serving-manifest", manifest_path]
    if served_weights_revision:
        cmd += ["--served-weights-revision", served_weights_revision]
    if tokenizer_revision:
        cmd += ["--tokenizer-revision", tokenizer_revision]
    if git_sha_override:
        cmd += ["--git-sha-override", git_sha_override]
    if expected_chat_template_hash:
        cmd += ["--expected-chat-template-hash", expected_chat_template_hash]
    if action_max_tokens is not None:
        cmd += ["--action-max-tokens", str(action_max_tokens)]
    if belief_max_tokens is not None:
        cmd += ["--belief-max-tokens", str(belief_max_tokens)]
    return cmd


def _existing_shard_git_sha(out_dir: str) -> str | None:
    """The `model_revision` an existing shard in `out_dir` already carries,
    for a `--allow-existing` resume of an arm this process did not start.

    Read here so the resumed run_sweep.py invocation can be told to reuse
    it (via --git-sha-override) instead of resolving a fresh sha at its own
    startup, which would otherwise let the two invocations' shards carry
    different model_revision values across a HEAD move.

    Returns None whenever no reuse-worthy value exists: `out_dir` absent,
    no `.parquet` files in it, none readable/have the column, or the shards
    -- checked across every file, never just the first -- do not all agree
    on exactly one value. Never a fabricated sha (no silent
    fallbacks): a cross-shard disagreement is precisely the corruption this
    helper exists to guard against re-widening, not a case to silently
    resolve by picking one shard's answer. Checking only the first shard
    would miss a split that already happened across prior shards and
    silently anoint the first shard's value as ground truth. A fresh arm (no existing shards) always returns
    None here, and `sweep_command`'s own None-omits-the-flag contract then
    lets run_sweep.py resolve its own sha exactly as it always has. A
    shard that cannot be read, or lacks the column, returns None, not an abort: this helper only ever
    widens what run_sweep.py would otherwise do on its own (reuse a
    known-good sha instead of resolving fresh) -- it must never be the
    reason a resume that would otherwise succeed now fails, but it must
    also never let one clean-looking shard mask a real split among others.
    """
    if not os.path.isdir(out_dir):
        return None
    parquet_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".parquet"))
    if not parquet_files:
        return None
    import polars as pl

    revisions: set[str] = set()
    for fname in parquet_files:
        try:
            df = pl.read_parquet(
                os.path.join(out_dir, fname), columns=["model_revision"]
            )
        except (OSError, pl.exceptions.PolarsError):
            return None
        revisions.update(df["model_revision"].drop_nulls().unique().to_list())
    return next(iter(revisions)) if len(revisions) == 1 else None


# The per-model registry carrying the expected_chat_template_hash pins.
# Script-relative, not cwd-relative: main() is run
# from arbitrary working directories in tests, and the registry is a shipped
# repo file, not a run artifact.
_MODELS_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "models.yaml"
)


def _expected_chat_template_hash(
    model_id: str, chat_template_kwargs: dict | None = None
) -> str | None:
    """Look up the expected_chat_template_hash pin for this model and kwargs.

    When chat_template_kwargs is None, looks up the bare model_id pin. When
    set, builds a variant key via chat_template_pin_key() and looks up only
    that variant - never falls back to the bare pin. A kwargs run with no
    variant pin returns None (the gradual-rollout state; run_sweep prints its
    loud NOT-pinned line and adds the value at first on-box serve).

    Returns None when the pin is absent (gradual rollout). Raises on an
    unreadable/unparseable registry (a broken checkout, not an unpinned
    model); main() turns that into an arm-level ABORT (no silent
    fallbacks).
    """
    with open(_MODELS_YAML) as f:
        registry = yaml.safe_load(f) or {}
    pins = registry.get("expected_chat_template_hash") or {}
    key = chat_template_pin_key(model_id, chat_template_kwargs)
    return pins.get(key) or None


def _model_decode_budgets(model_id: str) -> tuple[int | None, int | None]:
    """Look up per-model action_max_tokens and belief_max_tokens from models.yaml.

    Returns (action_max_tokens, belief_max_tokens), with None for absent keys.
    Reads from the fullsize_models section when present.
    Raises on an unreadable/unparseable registry.
    """
    with open(_MODELS_YAML) as f:
        registry = yaml.safe_load(f) or {}

    # Read from fullsize_models section if present (the trio carries budget values)
    fullsize_models = registry.get("fullsize_models") or {}
    model_config = fullsize_models.get(model_id) or {}

    action_max_tokens = model_config.get("action_max_tokens")
    belief_max_tokens = model_config.get("belief_max_tokens")

    return action_max_tokens, belief_max_tokens


# Providers that perform no lifecycle actions: the endpoint is started by
# something else (a person, a cloud launch, a PBS job script) and the harness
# only reads back what is serving. Neither can auto_swap or stop_on_teardown.
LIFECYCLE_FREE_PROVIDERS = frozenset({"preserved", "pbs-injob"})


def _resolve_served_weights_revision(model_id: str, ssh_host: str | None) -> str:
    """Resolve the HF weights revision the serving box will serve for
    `model_id`. run_experiment - the layer that owns serving.ssh_host -
    resolves; run_sweep only consumes.

    Delegates to `lmopt.provenance.weights_cache.resolve_prelaunch_revision`,
    the same evidence + condition set the weights-provenance capture uses (one
    implementation, no fork). Raises on any failure -
    including a missing ssh_host, since without box access nothing here can
    see the serving cache; main() turns that into an arm-level ABORT rather
    than launching a sweep that run_sweep's own startup gate would refuse
    moments later with a less specific message.
    """
    if not ssh_host:
        raise RuntimeError(
            f"cannot resolve the served-weights revision for {model_id!r}: "
            "serving.ssh_host is not configured, and the revision lives in "
            "the serving box's HF cache. Set serving.ssh_host, or launch "
            "scripts/run_sweep.py directly with --served-weights-revision."
        )
    return resolve_prelaunch_revision(model_id, host=ssh_host)


# Correspondence-CHECK: serving-manifest-identity
def _write_serving_manifest(out_dir: str, resolved, *, producer: str) -> str:
    """Write <out_dir>/serving_manifest.json - the BOX-only serving fields
    (image_ref, vllm_args, container, unit) read back as a byproduct of
    main() confirming what is served this arm (split by what each layer can
    actually see). run_sweep.py
    resolves model_id/max_model_len/chat_template_hash itself, directly
    over HTTP - this manifest deliberately omits them so there is never a
    second, possibly-drifting source of truth for a field HTTP can already
    see.

    Fields carry ResolvedServing's own "not observed" semantics verbatim
    (None when the box-side lookup could not verify them, e.g. an unlabelled
    hand-started container) -- never re-interpreted or defaulted here.
    `vllm_args` is null with `vllm_args_unobserved_reason` beside it when the
    launch argv was not observed, and a list otherwise; it is never `[]`.

    Also carries `model_id`/`base_url` as an identity stamp - never as a
    resolved value run_sweep.py should use (it resolves both itself, over
    HTTP). This lets run_sweep.py refuse loudly if it is ever handed a
    stale/mismatched manifest - a cross-model misattribution - rather than
    silently attributing one model's box-side provenance to a different
    model's run.

    producer (required keyword, no default - a silent default would
    mislabel where the box fields came from): "swap" when this manifest is
    a byproduct of `swap_to()` (the except-PreflightError branch),
    "resolve" when it is a byproduct of the no-swap `resolve()` call (the
    provenance-symmetry branch, for an arm that preflights clean and never
    swaps). Threaded through by run_sweep.py's `_resolve_serving_block`
    into the run_meta serving block's `box_fields_source` field, so a
    no-swap arm's record never asserts that a swap produced its box fields.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "model_id": resolved.model_id,
        "base_url": resolved.base_url,
        "producer": producer,
        "image_ref": resolved.image_ref,
        "vllm_args": (
            list(resolved.vllm_args) if resolved.vllm_args is not None else None
        ),
        "container": resolved.container,
        "unit": resolved.unit,
        # PBS lane: box-only in exactly the sense this
        # manifest exists for -- HTTP can see max_model_len, it cannot see a
        # vLLM version or a driver version. None on every provider that has a
        # container to digest instead; see src/lmopt/serving/env_fingerprint.py
        # for why a job-scheduler backend records this rather than image_ref.
        "serving_env": resolved.serving_env,
        "serving_env_hash": resolved.serving_env_hash,
        # GPU-card provenance: carried through
        # exactly like the fields above, from whichever provider's
        # resolve() populated them (or left them explicitly None).
        "gpu_name": resolved.gpu_name,
        "gpu_memory_total_mib": resolved.gpu_memory_total_mib,
    }
    if resolved.vllm_args is None:
        manifest["vllm_args_unobserved_reason"] = resolved.vllm_args_unobserved_reason
    path = os.path.join(out_dir, "serving_manifest.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, path)
    return path


def diagnose_command(out_dir: str) -> list[str]:
    """`scripts/diagnose.py` invocation, run unconditionally by `main()`
    after every successful sweep (`_run_step(d_cmd)`), so this is a third
    builder a real cluster job executes rather than prints - for the same
    reason as `sweep_command` and `pilot_gate_command`:
    `sys.executable` needs no path lookup, and `uv` is not on the cluster's.
    """
    return [sys.executable, "scripts/diagnose.py", out_dir + "/"]


#: The per-arm record of whether the exact-policy pass ran, written for every arm including one whose
#: config left the pass off, so "the pass did not run" is a recorded fact and never an inference from
#: a missing `exact_policy.csv` or a missing log line. Its reader is `lmopt.logging.exact_policy_pass`.
EXACT_POLICY_PASS_FILENAME = _EXACT_POLICY_PASS_FILENAME


def exact_policy_csv_rows(out_dir: str) -> int | None:
    """
    the number of data rows in the arm's `exact_policy.csv`
    :param out_dir: the arm's output directory (str)
    :returns: the row count excluding the header, or None when the file is absent (int or None)
    """
    return count_csv_rows(os.path.join(out_dir, "exact_policy.csv"))


def write_exact_policy_pass_record(out_dir: str, enabled: bool, rc: int | None) -> str:
    """
    write `<out_dir>/exact_policy_pass.json`: whether the arm's config enabled the pass, the exit
    status of the pass when it ran, and the number of rows its csv holds. A pass that did not run
    (disabled, or its arm lost shards and skipped it) records `rc` null; a csv that is absent
    records `csv_rows` null. The three are never defaulted into one another. Written beside the
    final name and moved into place, so a pull never sees half a file
    :param out_dir: the arm's output directory (str)
    :param enabled: whether `exact_policy.enabled` was true for this arm (bool)
    :param rc: the exit status of `scripts/exact_policy_pass.py`, None when it did not run (int or None)
    :returns: the path written (str)
    """
    record = {
        "enabled": bool(enabled),
        "rc": rc,
        "csv_rows": exact_policy_csv_rows(out_dir),
    }
    path = os.path.join(out_dir, EXACT_POLICY_PASS_FILENAME)
    tmp = os.path.join(out_dir, "." + EXACT_POLICY_PASS_FILENAME + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2)
    os.replace(tmp, path)
    return path


def exact_policy_pass_command(out_dir: str, base_url: str, model_id: str) -> list[str]:
    """`scripts/exact_policy_pass.py` invocation for one arm. Reads
    `out_dir`'s own just-collected parquet shards, renders each distinct
    (game, game_params, info_set)'s prompt through this arm's served model
    (`base_url`/`model_id` - the same values `sweep_command` used, so the
    pass always hits the endpoint that actually produced the data it is
    scoring), and writes `<out_dir>/exact_policy.csv` - the default path
    `analyse_stats.py`'s own `--exact-policy` flag would need pointed at
    explicitly, since that script's own default resolves against its
    `--out` (a separate analysis directory), not the sweep's `out_dir`.

    No `--limit`: this is the real per-arm pass (roughly 574 calls on the
    reference arm - cheap, but not free), never a smoke subset. Default
    `--timeout` (60s) is left alone, matching every other invocation in
    this module's own usage examples.

    The first argv element is `sys.executable`, a fourth builder actually
    executed by a real cluster job (via `_run_step(e_cmd)`, whenever
    `exact_policy.enabled` is true in the arm's config) rather than merely
    printed - same reason as the three builders above.
    """
    return [
        sys.executable,
        "scripts/exact_policy_pass.py",
        "--data-dir",
        out_dir,
        "--base-url",
        base_url,
        "--model",
        model_id,
        "--out",
        os.path.join(out_dir, "exact_policy.csv"),
    ]


_DEFAULT_OUT_ROOT = "data/run_experiment"


def _analysis_out_subdir(out_dir: str) -> str:
    """The <slug> path component that re-keys figures/analysis output
    on out_root, so two different epochs never collide on the same bare
    model short name.

    Empty string when out_root is the bare default, on purpose: this preserves
    the unscoped `analysis/run_experiment/<model>/exact_policy.csv`
    path of arms that predate the out_root-scoping convention, and
    `analyse_stats.py`'s own docstring examples for
    the still-common default case.

    Otherwise, a collision-free slug of the full out_root value. Existing
    '-' characters are escaped to '--' before '/' is replaced with '-', so
    a nested out_root and an unrelated flat one that merely contains a
    literal dash can never slug to the same string -- e.g. a hypothetical
    "data/run_experiment/recycle-v5-p2-M1/seedB" slugs to
    "recycle--v5--p2--M1-seedB", distinct from the real, already-shipped
    "data/run_experiment/recycle-v5-p2-M1-seedB" slugging to
    "recycle--v5--p2--M1--seedB" (a naive '/'->'-' replace alone is not
    injective once the source alphabet already contains '-', which real
    `out_root` values in `configs/*.yaml` do).

    Relies on `main()`'s own `out_root` read point having already
    refused an unsubstituted template placeholder (`__OUT_ROOT__` etc.)
    before any caller reaches here -- this function does not re-check that.
    """
    out_root = os.path.dirname(out_dir.rstrip("/"))
    if out_root == _DEFAULT_OUT_ROOT:
        return ""
    return out_root.rstrip("/").replace("-", "--").replace("/", "-")


def _analysis_out_path(base: str, out_dir: str, short: str) -> str:
    slug = _analysis_out_subdir(out_dir)
    return f"{base}/{slug}/{short}" if slug else f"{base}/{short}"


def analyse_stats_command(out_dir: str, short: str) -> list[str]:
    """`scripts/analyse_stats.py` invocation naming `--exact-policy` explicitly.

    `exact_policy_pass_command` (above) writes this arm's artefact to
    `<out_dir>/exact_policy.csv`, the sweep's own directory (`--data`,
    where `run_meta.json` also lives). `analyse_stats.py`'s
    `--exact-policy` default resolves against its `--out` instead (a
    separate analysis directory, `analysis/run_experiment/<short>`), so the
    two directories are never the same one and the flag has to be passed.
    The command is printed in the final summary.
    """
    return [
        "uv",
        "run",
        "python",
        "scripts/analyse_stats.py",
        "--data",
        out_dir,
        "--out",
        _analysis_out_path("analysis/run_experiment", out_dir, short),
        "--exact-policy",
        os.path.join(out_dir, "exact_policy.csv"),
    ]


class _TeardownSignal(Exception):
    """Raised from the SIGTERM/SIGINT handler below so the process's one
    teardown call site - the `finally` around main()'s arm loop - runs
    regardless of whether the run is aborting via a `return` or a signal."""


# The signum the handler last saw. A module-level cell, not a closure inside
# main(), because `_run_step` (below) needs it to know which signal to forward
# to the running child: the exception it catches carries the number in its
# message only, and the two origins call for different treatment.
_teardown_signum: int | None = None

# How long a step child gets to finish its own cleanup after the forwarded
# signal before it is killed. run_sweep.py's `_pool_signal_guard`
# (src/lmopt/orchestration/local.py) has to shut down a ProcessPoolExecutor
# and terminate every captured worker inside this window.
_STEP_TEARDOWN_GRACE_S = 15.0


def _handle_teardown_signal(signum, _frame):
    global _teardown_signum
    _teardown_signum = signum
    raise _TeardownSignal(f"received signal {signum}")


def _wait_absorbing_interrupts(child, timeout: float | None) -> bool:
    """`child.wait(timeout)` that a further signal cannot cut short. Returns
    True if the child exited, False if the timeout ran out first.

    Double-Ctrl-C is the operator reflex, and a second signal during teardown
    raises out of `wait()` in the middle of the very cleanup that guarantees
    the child dies. Letting it through abandons a live, unreaped child, which
    is the whole defect this module's teardown path exists to close, so it is
    absorbed here and the remaining budget is re-waited. Bounded twice over:
    by `timeout` while the child is being asked politely, and by SIGKILL
    afterwards (`timeout=None` is only ever used on an already-killed child).
    """
    remaining = timeout
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            child.wait(remaining)
            return True
        except subprocess.TimeoutExpired:
            return False
        except (_TeardownSignal, KeyboardInterrupt):
            if deadline is None:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False


def _teardown_child(
    child, signum: int | None, grace_s: float = _STEP_TEARDOWN_GRACE_S
) -> None:
    """Give a running step child a chance to run its own signal handler, then
    kill it if it will not go. Returns only once the child is dead, whatever
    arrives in the meantime.

    `signum` is the signal this process received, and it decides whether the
    child is signalled at all. SIGTERM was delivered to this process only, so
    it is forwarded. SIGINT was not: a Ctrl-C goes to the whole foreground
    process group, and the child is in this process group (`_run_step`
    deliberately does not `start_new_session`), so it already has the signal
    and is already running its cleanup, so sending a second one races that
    cleanup for no gain.
    """
    if signum != signal.SIGINT:
        child.send_signal(signal.SIGTERM)
    if _wait_absorbing_interrupts(child, grace_s):
        return
    print(
        f"WARNING: step child {getattr(child, 'pid', '?')} did not exit "
        f"within {grace_s:g}s of the teardown signal, so killing it. Its own "
        "children (a sweep's pool workers) may need checking by hand.",
        file=sys.stderr,
    )
    child.kill()
    _wait_absorbing_interrupts(child, None)


def _run_step(cmd: list[str]) -> int:
    """Dispatch one arm step (pilot gate, sweep, diagnose, exact-policy) and
    return its exit code, forwarding a teardown signal to it rather than
    SIGKILLing it.

    `subprocess.run` cannot be used here, and that is the whole point of this
    function. `_handle_teardown_signal` raises a plain Python exception from
    inside an OS signal handler, so it surfaces wherever the main thread
    happens to be blocked; if that is inside `subprocess.run`, CPython's own
    `run()` catches it in a bare `except:` and calls `process.kill()`, i.e.
    SIGKILL, uncatchable, before re-raising. The child then never runs its own handler,
    so run_sweep.py's `_pool_signal_guard` cannot fire and its pool workers
    orphan.

    `Popen.wait()` has no such handler, so the exception arrives here, where
    the child can be signalled properly. No `start_new_session`: the child
    stays in this process group so an interactive Ctrl-C keeps reaching it
    directly. The exception is re-raised after the child is reaped, so main()'s
    `except _TeardownSignal` / `finally` teardown path is unchanged.

    The invariant, and the reason the `except` is `BaseException` rather than
    `_TeardownSignal`: no exit path from this function may leave the child
    alive. `_TeardownSignal` is only raised when `stop_on_teardown` installed
    our handlers; under every other config Python's default disposition turns
    SIGINT into `KeyboardInterrupt`, and `kill -INT` on the parent alone does
    not reach the child at all, so narrowing the `except` would orphan it. A
    `KeyboardInterrupt` is a SIGINT, so it is reported to `_teardown_child` as
    one: in the normal interactive case the whole process group already has
    that signal.
    """
    child = subprocess.Popen(cmd)
    try:
        return child.wait()
    except BaseException as exc:
        origin = (
            signal.SIGINT if isinstance(exc, KeyboardInterrupt) else _teardown_signum
        )
        _teardown_child(child, origin)
        raise


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    # backend is load-bearing per-row provenance - no silent default
    # (run_sweep alone would default it to cuda-3090-vllm).
    backend = cfg.get("backend")
    if not backend:
        sys.exit(
            "config error: 'backend' is required (per-row provenance pin "
            "- e.g. 'dgx-spark-vllm'); no silent default."
        )

    # serving.auto_swap decides (config over forks) whether a preflight
    # failure aborts, or triggers a swap-then-verify. [default asymmetry,
    # deliberate]: an absent 'serving' key, or an absent 'auto_swap' sub-key,
    # means auto_swap defaults to FALSE. Shipped configs that never set this
    # block must not silently acquire the power to stop/start a GPU-serving
    # systemd unit on a remote box merely because this key is read. Only
    # configs/run_experiment.yaml opts in explicitly.
    serving_cfg = cfg.get("serving") or {}
    # serving.provider dispatch. Default "spark-systemd"
    # (ServingController behaviour when the key is absent).
    # "preserved" means no lifecycle actions (no ssh/systemctl): endpoint already up,
    # manifest from config + HTTP. Unknown provider value is a loud abort (no
    # silent fallbacks).
    # "pbs-injob" (OpenPBS lane) is also lifecycle-free, but
    # unlike preserved it shares a machine with the server: the PBS job script
    # starts vLLM inside the allocation and the harness reads back the real
    # argv and the serving environment. See src/lmopt/serving/injob.py.
    provider = serving_cfg.get("provider", "spark-systemd")
    if provider not in ("spark-systemd", "preserved", "pbs-injob"):
        sys.exit(
            f"config error: serving.provider={provider!r} is unknown "
            "(supported: 'spark-systemd', 'preserved', 'pbs-injob')"
        )
    auto_swap = bool(serving_cfg.get("auto_swap", False))
    ssh_host = serving_cfg.get("ssh_host")
    unit_template = serving_cfg.get("unit_template", "vllm-roster@")
    swap_timeout_s = float(serving_cfg.get("timeout_s", 900.0))
    # serving.stop_on_teardown - read exactly like
    # auto_swap directly above, same default asymmetry: absent key or
    # absent 'serving' block means FALSE, and only configs/run_experiment.yaml
    # opts in. Without a teardown path, a killed arm's vLLM server keeps
    # holding GPU memory on the serving box indefinitely. When true, the most-recently-served model's unit is
    # stopped at normal completion, on every abort ``return``, and on
    # SIGTERM/SIGINT (handlers installed below, only when this key is true).
    stop_on_teardown = bool(serving_cfg.get("stop_on_teardown", False))

    # Fail at time zero, not mid-experiment: an auto_swap:true config with no
    # ssh_host cannot swap anything, and discovering that only after the
    # first model's preflight fails (possibly after other models already
    # ran) would waste exactly the unattended-run time this feature exists
    # to save.
    if auto_swap and not ssh_host:
        sys.exit(
            "config error: serving.auto_swap is true but serving.ssh_host is "
            "not set - cannot swap without a target box."
        )
    # Same fail-at-time-zero shape: a stop_on_teardown:true config with no
    # ssh_host cannot stop anything either, and discovering that only at
    # teardown - after the sweep already ran - defeats the point of a
    # config-time check.
    if stop_on_teardown and not ssh_host:
        sys.exit(
            "config error: serving.stop_on_teardown is true but "
            "serving.ssh_host is not set - cannot stop a serving unit "
            "without a target box."
        )
    # The preserved provider conflicts with auto_swap and
    # stop_on_teardown (preserved performs no lifecycle actions). Fail at time
    # zero, same shape as the ssh_host guards above.
    # Written as a SET rather than repeated equality: pbs-injob is
    # the second lifecycle-free provider, and a third would otherwise need both
    # guards edited again, with a missed edit silently permitting auto_swap.
    if provider in LIFECYCLE_FREE_PROVIDERS and auto_swap:
        sys.exit(
            f"config error: serving.provider={provider} cannot auto_swap "
            f"({provider} arms perform no lifecycle actions; the endpoint "
            "must already be serving the expected model)."
        )
    if provider in LIFECYCLE_FREE_PROVIDERS and stop_on_teardown:
        sys.exit(
            f"config error: serving.provider={provider} cannot stop_on_teardown "
            f"({provider} arms perform no lifecycle actions)."
        )

    # exact_policy.enabled decides (config over forks) whether each arm runs
    # its own scripts/exact_policy_pass.py after sweep + diagnose. [default
    # Asymmetry, deliberate - same shape as serving.auto_swap above]: an
    # absent 'exact_policy' key, or an absent 'enabled' sub-key, means this
    # defaults to FALSE, the conservative code default; configs opt in
    # explicitly, for the same reason serving.auto_swap does - no other
    # shipped config should silently start making ~574 extra network calls
    # against a live endpoint merely because this key is read.
    exact_policy_cfg = cfg.get("exact_policy") or {}
    exact_policy_enabled = bool(exact_policy_cfg.get("enabled", False))

    models = enabled_models(cfg, args.models)
    # Fail at time zero, like the backend and ssh_host guards above: one
    # directory cannot be the pilot for several arms. pilot_gate.py's own
    # --skip-sweep provenance check would refuse the mismatch, but only
    # After the first arm had already been judged against the wrong pilot.
    if args.pilot_dir and len(models) > 1:
        sys.exit(
            f"config error: --pilot-dir names ONE directory but "
            f"{len(models)} models are selected "
            f"({[m['id'] for m in models]}) - one arm's pilot must never "
            "gate another's. Narrow with --models, or give each roster "
            "entry its own 'pilot_dir' key."
        )
    if not models:
        print(
            "No enabled models in the config - nothing to run "
            "(enable roster entries as the serving setup serves them)."
        )
        return 1

    # Fail at time zero, not inside a subprocess mid-experiment: an online W&B
    # run without WANDB_API_KEY would abort the first sweep only after the
    # (expensive) preflight + cache warmup.
    wandb_cfg = cfg.get("wandb") or {}
    if (
        not args.dry_run
        and wandb_cfg.get("enabled")
        and wandb_cfg.get("mode") == "online"
        and "WANDB_API_KEY" not in os.environ
    ):
        sys.exit(
            "wandb.mode is 'online' but WANDB_API_KEY is not exported. "
            "Source it first:\n  set -a; . ./.env; set +a"
        )

    out_root = cfg.get("out_root", "data/run_experiment")
    # Refuse an unsubstituted template placeholder before any write.
    # `configs/run_experiment_recycle_v*_*.yaml` ship `out_root: __OUT_ROOT__`
    # (and analogous __SEED__/__PORT__ placeholders elsewhere in the same
    # files) -- substituted only by `scripts/collection_job.py render`, which
    # must run before those configs are used. An invocation that skips the render reads the
    # Literal placeholder string here, and out_root flows straight into
    # `out_dir = os.path.join(out_root, short)` a few lines below, then into
    # `sweep_command` -- which actually writes real experiment data -- long
    # before either analyse-side command builder ever runs (they only fire
    # in a later, separate summary loop, after every arm's sweep has
    # already completed). Nothing else refuses this first write; the
    # existing `--allow-existing`-gated guard further down only catches a
    # Second write into an already-populated out_dir. Pattern-matched
    # (`__[A-Z0-9_]+__`), not hardcoded to this one instance, since __SEED__/
    # __PORT__ show the same "template token, substituted only by a wrapper
    # script" shape could recur under a different name.
    if re.fullmatch(r"__[A-Z0-9_]+__", out_root):
        sys.exit(
            f"REFUSING: out_root is the literal unsubstituted placeholder "
            f"{out_root!r}. This config is a template that "
            f"scripts/collection_job.py render fills in, not a config for run_experiment.py "
            f"directly. Running with this value would write real experiment "
            f"data under a directory literally named {out_root!r}."
        )
    n_files = (
        len(cfg.get("games", []))
        * len(cfg.get("opponents", []))
        * cfg.get("n_seeds", 1)
    )
    print(
        f"Experiment: {len(models)} model(s) × {len(cfg.get('games', []))} games × "
        f"{len(cfg.get('opponents', []))} opponents × {cfg.get('n_seeds', 1)} seeds "
        f"(N={cfg.get('n_hands')}, backend={backend}) "
        f"→ {n_files} parquet files per model"
    )
    print(f"Output root: {out_root}/")
    print()

    ran: list[tuple[str, str]] = []
    # Arms whose sweep ran to completion but LOST shards. Kept out
    # of `ran` on purpose (see the append site) and reported on their own.
    partial_arms: list[tuple[str, str]] = []
    # Arms skipped because their out_dir already holds a collected run (--skip-collected).
    skipped_collected: list[tuple[str, str]] = []
    exact_policy_failures: list[str] = []
    pilot_skipped: list[str] = []

    # The id/base_url of the model this run last confirmed
    # served (a clean preflight, or the re-verify immediately after a
    # swap) - never the roster's last entry, which may not be the one
    # actually served if the loop aborts before reaching it. `None` means
    # nothing was ever confirmed served, so teardown has nothing to stop.
    last_served_model_id: str | None = None
    last_served_base_url: str | None = None

    def _stop_served_unit() -> None:
        """Stop the most recently served unit's systemd instance.

        No-op unless `stop_on_teardown` is true and something was actually
        confirmed served - an abort before any preflight succeeded (a bad
        config, an unreachable box on arm 1) has nothing to stop. No silent
        fallbacks: prints loudly what is being stopped and why
        before attempting it. A stop() failure is reported to stderr and
        never changes the exit code this run would otherwise return - by
        the time this runs (in `finally`, below), the run's own outcome (0,
        an abort code, a partial code) is already decided.
        """
        if not stop_on_teardown or last_served_model_id is None:
            return
        print(
            f"\nserving.stop_on_teardown is true - stopping the unit "
            f"serving {last_served_model_id} at {last_served_base_url} "
            "(a killed arm's vLLM server keeps holding GPU memory on the "
            "serving box with no teardown path to catch it)."
        )
        try:
            ServingController(
                ssh_host=cast(
                    str, ssh_host
                ),  # stop_on_teardown requires ssh_host (config guard)
                base_url=last_served_base_url or "",
                unit_template=unit_template,
            ).stop(last_served_model_id)
        except Exception as exc:
            print(
                f"WARNING: teardown stop() FAILED for {last_served_model_id}: "
                f"{exc!r} - the unit may still be running; stop it by hand "
                f"(systemctl --user stop <unit> on {ssh_host}).",
                file=sys.stderr,
            )

    # `_TeardownSignal` and `_handle_teardown_signal` are module-level:
    # `_run_step` needs both, and the handler has to record the signum
    # somewhere that outlives its own frame.
    #
    # Installed only when stop_on_teardown is true (config over forks):
    # every other config keeps Python's default SIGTERM/SIGINT
    # behaviour unchanged. Previous handlers are restored in `finally` below
    # so this function's signal footprint never outlives the call.
    _prev_signal_handlers: dict[signal.Signals, Any] = {}
    if stop_on_teardown:
        global _teardown_signum
        _teardown_signum = None  # never inherit a previous call's signal
        for _sig in (signal.SIGTERM, signal.SIGINT):
            _prev_signal_handlers[_sig] = signal.signal(_sig, _handle_teardown_signal)

    try:
        for entry in models:
            mid = entry["id"]
            short = model_short(mid)
            out_dir = os.path.join(out_root, short)
            base_url = entry["vllm_base_url"]
            pilot_dir = pilot_dir_for(entry, args.pilot_dir)
            p_cmd = pilot_gate_command(args.config, mid, pilot_dir)
            s_cmd = sweep_command(args.config, entry, out_dir, backend)
            d_cmd = diagnose_command(out_dir)
            e_cmd = exact_policy_pass_command(out_dir, base_url, mid)

            print(f"=== {mid} ===")
            print(f"  out_dir:   {out_dir}")
            print(f"  preflight: GET {base_url.rstrip('/')}/models  (expect {mid!r})")
            if args.skip_pilot_check:
                print("  pilot gate: SKIPPED via --skip-pilot-check")
            else:
                print(f"  pilot gate: {' '.join(p_cmd)}")
            print(f"  run_sweep: {' '.join(s_cmd)}")
            print(f"  diagnose:  {' '.join(d_cmd)}")
            if exact_policy_enabled:
                print(f"  exact_policy: {' '.join(e_cmd)}")
            else:
                print(
                    "  exact_policy: SKIPPED (exact_policy.enabled is not true in this config)"
                )

            if args.dry_run:
                print()
                continue

            # Stale-data pollution guard: refuse to mix a new run into an out_dir
            # that already holds parquet files (override with --allow-existing).
            if os.path.isdir(out_dir):
                existing = [f for f in os.listdir(out_dir) if f.endswith(".parquet")]
                if (
                    existing
                    and args.skip_collected
                    and arm_already_finished(out_dir, cfg, exact_policy_enabled)
                ):
                    print(
                        f"SKIPPING {mid}: {out_dir} already holds this arm, finished "
                        "(--skip-collected); nothing is rerun or overwritten."
                    )
                    skipped_collected.append((mid, out_dir))
                    continue
                if existing and not args.allow_existing:
                    sys.exit(
                        f"REFUSING to run {mid}: {out_dir} already contains "
                        f"{len(existing)} parquet file(s). Move/clean it, or pass "
                        "--allow-existing to proceed anyway."
                    )

            # The launch precondition - see PILOT_GATE_REFUSAL_EXIT_CODE
            # above. Placed here, before
            # preflight and before every sweep subprocess, because it is offline
            # and endpoint-free: an operator with no pilot should learn so at
            # second zero, not after the serving box is up.
            if args.skip_pilot_check:
                # RECORDED, not silent: an unattended multi-day cycle's only
                # record is this output. Also re-surfaced in the final summary.
                print(
                    f"  pilot gate: SKIPPED via --skip-pilot-check for {mid} - "
                    "this arm carries NO pilot evidence; a skipped gate is not "
                    "a passed gate.",
                    file=sys.stderr,
                )
                pilot_skipped.append(mid)
            else:
                # An ABSENT pilot is a refusal, never a pass. "No pilot" and
                # "pilot passed" must never share an outcome: a fall-through
                # here would launch an arm with no pilot evidence at all.
                if not os.path.isdir(pilot_dir):
                    sys.exit(
                        f"REFUSING to run {mid}: no pilot directory at "
                        f"{pilot_dir}. An absent pilot is not a passing pilot - "
                        "nothing has been measured for this arm.\n"
                        "Run one first:\n  "
                        + " ".join(
                            pilot_gate_command(
                                args.config, mid, pilot_dir, skip_sweep=False
                            )
                        )
                        + "\nor name an existing one with --pilot-dir (or the "
                        "roster entry's 'pilot_dir' key), or launch without any "
                        "pilot evidence - loudly and on the record - with "
                        "--skip-pilot-check."
                    )
                rc = _run_step(p_cmd)
                if rc == _PILOT_GATE_FAILURE_EXIT_CODE:
                    # The gate's own VIOLATIONS block has just been printed to
                    # this console by the subprocess (streams are inherited,
                    # exactly as for sweep/diagnose) - that is the verdict, and
                    # it is not restated here in weaker words.
                    print(
                        f"\nREFUSING to run {mid}: the pilot gate judged "
                        f"{pilot_dir} UNUSABLE (exit {rc}); its VIOLATIONS are "
                        "printed immediately above. Fix the arm and re-pilot, "
                        "or re-run the gate by hand:\n  " + " ".join(p_cmd),
                        file=sys.stderr,
                    )
                    return PILOT_GATE_REFUSAL_EXIT_CODE
                if rc != 0:
                    # The gate itself broke (bad config, unsafe/unreadable
                    # --out-dir): pilot_gate.py's exit 1. That is not evidence
                    # about the arm, and it is not permission to launch either -
                    # a non-zero that is not the gate's own failure code must
                    # never fall through to the sweep.
                    print(
                        f"\nABORT: {' '.join(p_cmd)} exited {rc} - the pilot "
                        "gate itself failed (an operational failure of the "
                        "gate, NOT a verdict on this arm). Nothing has been "
                        "measured; refusing to launch.",
                        file=sys.stderr,
                    )
                    return 1
                print(f"  pilot gate: PASSED on {pilot_dir}")

            manifest_path: str | None = None
            try:
                served = preflight(base_url, mid)
            except PreflightError as exc:
                if not auto_swap:
                    print(f"\n{exc}", file=sys.stderr)
                    return 1
                # config over forks: swap-then-verify, decided by
                # serving.auto_swap, never a code branch guessed from context.
                # The preserved provider never reaches here
                # (the config-time check above aborts on preserved+auto_swap).
                print(f"  {exc}")
                print(
                    f"  serving.auto_swap is true - swapping {base_url} to "
                    f"{mid} (budget {swap_timeout_s:.0f}s)..."
                )
                controller = ServingController(
                    ssh_host=cast(
                        str, ssh_host
                    ),  # auto_swap requires ssh_host (config guard)
                    base_url=base_url,
                    unit_template=unit_template,
                )
                try:
                    resolved = controller.swap_to(mid, timeout_s=swap_timeout_s)
                except Exception as swap_exc:
                    # Every other failure in this script aborts the whole
                    # experiment loudly (no continue-on-error, by design) - a
                    # failed swap is no exception: never proceed against
                    # whatever happens to be listening. `except Exception`,
                    # not `except ServingError`: unlike the resolve() catch
                    # below, the outcome here is `return 1` either way - but a
                    # narrow catch means the ssh/docker/systemctl machinery
                    # under swap_to() (an unguarded `subprocess.run` in
                    # control.py, or an HTTP `IncompleteRead` mid-poll) can
                    # turn "abort with a clear message" into "abort with a raw
                    # traceback" - and swap_to is exactly where a box
                    # mid-restart is most likely to produce one. Same reason as
                    # preflight()'s broad catch: never let box flakiness
                    # surface as an opaque, uncontrolled exception in an
                    # otherwise-controlled no-continue-on-error script.
                    print(
                        f"\nABORT: swap to {mid} at {base_url} failed: " f"{swap_exc}",
                        file=sys.stderr,
                    )
                    return 1
                manifest_path = _write_serving_manifest(
                    out_dir, resolved, producer="swap"
                )
                # swap_to() already never returns while the wrong id is served
                # (control.py's own invariant) - this second preflight call is
                # the "verifies" half of "swaps and then verifies", re-run
                # through the same code path every non-swap model uses, so
                # nothing downstream needs to know a swap happened. Wrapped in
                # its own try/except: a failed
                # post-swap verify must exit through the same
                # print(..., file=sys.stderr); return 1 idiom every other
                # failure in this script uses, not an uncaught traceback.
                try:
                    served = preflight(base_url, mid)
                except PreflightError as verify_exc:
                    print(f"\n{verify_exc}", file=sys.stderr)
                    return 1
                # The post-swap re-verify succeeded - this
                # model is confirmed served, so teardown (if enabled) targets
                # it from here on, even if a later check below (chat-template
                # mismatch) still aborts this arm.
                last_served_model_id = mid
                last_served_base_url = base_url

                # A chat_template_hash that is only logged is compared by
                # Nothing -- a template that renders fine-but-wrong produces a
                # valid hash and raises nothing. The comparison here: swap_to()
                # above already resolved chat_template_hash once (swap-time,
                # `resolved.chat_template_hash`); re-resolve it here, moments
                # later, from the same endpoint this arm is about to be
                # dispatched against, and abort before the sweep subprocess on
                # any mismatch -- nothing legitimately explains the template
                # changing between a swap and the preflight check that
                # immediately follows it. Deliberately the lightweight
                # /tokenize+/detokenize round trip (matching run_sweep.py's own
                # `_resolve_serving_block` preflight idiom) rather than a
                # second full `controller.resolve()` -- only the hash is being
                # re-checked here, not the box-only fields the manifest above
                # already captured. A stored per-model expectation also exists:
                # configs/models.yaml's expected_chat_template_hash block,
                # looked up per arm below (_expected_chat_template_hash) and
                # passed to run_sweep as --expected-chat-template-hash, where
                # it is compared against the hash run_sweep resolves at its
                # own startup -- covering the never-swapped arm. The swap-vs-preflight check
                # Here is kept: it catches a template changing between two
                # observations moments apart, which the pin (measured from
                # landed arms) cannot distinguish from a stale pin.
                # No chat_template_kwargs: this detects serving drift, a
                # property of the served template alone.
                try:
                    verify_hash = resolve_chat_template_hash(base_url, mid)
                except ChatTemplateUnavailable as hash_exc:
                    print(
                        f"\nABORT: chat-template preflight FAILED for {mid} at "
                        f"{base_url} immediately after swapping: {hash_exc}. "
                        "Aborting before the sweep rather than logging a null "
                        "chat_template_hash.",
                        file=sys.stderr,
                    )
                    return 1
                if verify_hash != resolved.chat_template_hash:
                    print(
                        f"\nABORT: chat_template_hash mismatch for {mid} at "
                        f"{base_url} -- swap recorded {resolved.chat_template_hash!r}, "
                        f"the preflight check immediately after recorded "
                        f"{verify_hash!r}. No swap explains this change; "
                        "aborting before the sweep.",
                        file=sys.stderr,
                    )
                    return 1
            else:
                # Preflight succeeded without a swap - this
                # model is confirmed served, so teardown (if enabled) targets
                # it from here on.
                last_served_model_id = mid
                last_served_base_url = base_url
                # Preflight succeeded WITHOUT a swap - the common case for an
                # already-correctly-served model. Still attempt to capture the
                # box-only manifest here, so provenance is symmetric across
                # arms rather than recorded only for the ones that happened to
                # need a swap - otherwise --gpu-memory-utilization/
                # --enforce-eager would be recorded for swapped arms and
                # silently missing for the others, defeating the point of a
                # per-arm-comparable record. Requires ssh_host (no ssh info
                # configured -> nothing to inspect). A resolve() failure
                # degrades to "no manifest" rather than aborting an
                # otherwise-healthy run: recording nothing is bad, refusing to
                # run over it is worse.
                #
                # Provider dispatch. preserved uses
                # PreservedServing (config-supplied image_ref + HTTP provenance,
                # No ssh/systemctl), and aborts on missing image_ref (no silent
                # fallbacks). spark-systemd uses ServingController
                # when ssh_host is set.
                if provider == "pbs-injob":
                    try:
                        injob = InJobServing.from_config(mid, base_url, serving_cfg)
                        resolved = injob.resolve()
                        manifest_path = _write_serving_manifest(
                            out_dir,
                            resolved,
                            producer="pbs-injob",
                        )
                    except InJobServingError as injob_exc:
                        # Same ABORT semantics as preserved, for the same
                        # reason with a different token: a missing serving_env
                        # means no serving-stack provenance, and there is no
                        # container digest on this backend to stand in for it.
                        print(
                            f"\nABORT: pbs-injob provider resolve failed for "
                            f"{mid}: {injob_exc}",
                            file=sys.stderr,
                        )
                        return 1
                elif provider == "preserved":
                    try:
                        preserved = PreservedServing.from_config(
                            mid, base_url, serving_cfg
                        )
                        resolved = preserved.resolve()
                        manifest_path = _write_serving_manifest(
                            out_dir,
                            resolved,
                            producer="preserved",
                        )
                    except PreservedServingError as preserved_exc:
                        # A preserved arm's resolve() failure is an ABORT
                        # (unlike spark-systemd's degrade-to-no-manifest):
                        # missing image_ref means no serving-stack provenance,
                        # which is the minimum bar for a keepable arm.
                        print(
                            f"\nABORT: preserved provider resolve failed for "
                            f"{mid}: {preserved_exc}",
                            file=sys.stderr,
                        )
                        return 1
                elif ssh_host:
                    controller = ServingController(
                        ssh_host=ssh_host,
                        base_url=base_url,
                        unit_template=unit_template,
                    )
                    try:
                        resolved = controller.resolve(mid)
                        manifest_path = _write_serving_manifest(
                            out_dir,
                            resolved,
                            producer="resolve",
                        )
                    except Exception as resolve_exc:
                        # `except Exception`, not `except ServingError`: a
                        # narrow catch would let an `http.client.IncompleteRead`
                        # from `controller.current()` inside `resolve()` (a box
                        # mid-swap/restart truncating the response), or a local
                        # `subprocess.run` OSError from the unguarded ssh call
                        # in control.py, propagate raw and kill the whole
                        # experiment before the sweep starts, contradicting the
                        # degrade-to-no-manifest contract above. Same
                        # boundary-catch approach as preflight(), applied here
                        # because this is a second call site into the same
                        # box-flakiness-prone machinery.
                        print(
                            f"  serving.ssh_host is set but resolve() could "
                            f"not confirm a box-side manifest for {mid}: "
                            f"{resolve_exc!r} - proceeding without it."
                        )
            print(
                f"  preflight OK: {mid} served at {base_url} "
                f"({len(served)} model(s) listed)"
            )

            # Resolve the served-weights revision for this arm, now that
            # preflight has confirmed the model is what the endpoint serves,
            # and thread it into the sweep argv. Resolved here because this
            # script owns serving.ssh_host; run_sweep is HTTP-only and cannot
            # see the box's HF cache.
            # Aborts on failure, deliberately unlike the box-manifest
            # resolve() degrade above: manifest fields are optional
            # provenance, but a real sweep launched without a revision is
            # refused by run_sweep's own startup gate anyway -- aborting
            # here just says why with the resolver's specific reason
            # (no silent fallbacks).
            #
            # Check for a per-model pinned
            # served_weights_revision in the config before attempting ssh
            # resolution. When present and truthy, use it directly (no ssh
            # call); when present but empty, abort (matching run_sweep's
            # truthiness check); when absent, fall through to ssh resolution
            # (config over forks).
            pinned_revision = entry.get("served_weights_revision")

            # For pbs-injob arms, `resolved.vllm_args` (already
            # observed above by InJobServing.resolve() -- the real launch's
            # own argv) is a fact about what actually loaded; the config's
            # served_weights_revision is only a declaration (or, on the ssh
            # fallback path below, an inference from a filesystem cache).
            # Parse it now, before any comparison or fallback, so a wrong
            # pin (stale symlink, re-pull, operator typo) fails loudly here
            # -- never as an unexplained downstream result (no silent
            # fallbacks). Scoped to pbs-injob: spark-systemd's
            # launch is cross-repo and does not emit --revision; this same
            # mechanism lights up for it automatically once it does, with no
            # further change here.
            observed_revision: str | None = None
            observed_tokenizer_revision: str | None = None
            if provider == "pbs-injob" and resolved.vllm_args is not None:
                observed_revision, observed_tokenizer_revision = parse_revision_flags(
                    resolved.vllm_args
                )

            if pinned_revision is not None:
                # Key is present: truthiness check (empty string must abort)
                if not pinned_revision:
                    print(
                        f"\nABORT: served-weights revision for {mid} is "
                        "configured but EMPTY. An empty revision is refused "
                        "(matching run_sweep's own startup gate); unset the "
                        "key to fall back to ssh resolution, or set a truthy "
                        "value.",
                        file=sys.stderr,
                    )
                    return 1
                declared_revision = str(pinned_revision)
                if observed_revision and observed_revision != declared_revision:
                    print(
                        f"\nABORT: served-weights revision MISMATCH for {mid}: "
                        f"config declares served_weights_revision="
                        f"{declared_revision!r}, but the launched vLLM "
                        f"process's own argv pins --revision="
                        f"{observed_revision!r}. No swap or re-pull explains "
                        "this disagreement; aborting before the sweep.",
                        file=sys.stderr,
                    )
                    return 1
                if observed_revision:
                    # The observed value wins even on agreement: the source
                    # is a freshly-checked fact, not a trusted declaration,
                    # so served_weights_revision is repointed to carry it.
                    served_weights_revision = observed_revision
                    print(
                        f"  served_weights_revision: {served_weights_revision} "
                        "(observed in launch argv, agrees with config pin)"
                    )
                else:
                    served_weights_revision = declared_revision
                    # `observed_revision` being falsy has two distinct causes that
                    # must not read as one message. A launch with no --revision
                    # token in a real, identified argv "did not pin" it. A
                    # `resolved.vllm_args` of None means the argv scan could not
                    # name this job's server, so nothing about the launch's own
                    # --revision flag was checked, and saying "did not pin" would
                    # be a claim about the launch dressed up as a measurement.
                    if provider == "pbs-injob" and resolved.vllm_args is None:
                        print(
                            f"  served_weights_revision: {served_weights_revision} "
                            "(pinned in config; NOT VERIFIED: "
                            f"{resolved.vllm_args_unobserved_reason}; so --revision "
                            "could not be read back, not because the launch "
                            "omitted it)"
                        )
                    else:
                        print(
                            f"  served_weights_revision: {served_weights_revision} "
                            "(pinned in config; not verified: launch did not "
                            "pin --revision)"
                        )
            elif observed_revision:
                # No declared pin, but the pbs-injob launch itself pinned a
                # revision -- use it directly. There is nothing to compare
                # against and, for this topology, nothing to ssh into
                # either.
                served_weights_revision = observed_revision
                print(
                    f"  served_weights_revision: {served_weights_revision} "
                    "(observed in launch argv, no config pin declared)"
                )
            else:
                # Key is absent and nothing observed: ssh resolver path.
                try:
                    served_weights_revision = str(
                        _resolve_served_weights_revision(mid, ssh_host)
                    )
                except Exception as swr_exc:
                    print(
                        f"\nABORT: served-weights revision unresolvable for "
                        f"{mid}: {swr_exc}",
                        file=sys.stderr,
                    )
                    return 1
                print(f"  served_weights_revision: {served_weights_revision}")

            tokenizer_revision = observed_tokenizer_revision

            # Look up this model's expected_chat_template_hash pin in configs/models.yaml
            # and thread it into the sweep argv; run_sweep compares it
            # against the hash it resolves at startup and aborts on
            # mismatch. None (an unpinned model) passes no flag - run_sweep
            # then prints its loud NOT-pinned line and continues (pins roll
            # out gradually). An unreadable registry, by contrast, aborts
            # the arm: that is a broken checkout, and proceeding would dress
            # every arm up as merely not-yet-pinned (no silent
            # fallbacks).
            # Pass the experiment config's top-level
            # chat_template_kwargs so a kwargs arm looks up its variant pin,
            # not the bare one.
            chat_template_kwargs = cfg.get("chat_template_kwargs")
            try:
                expected_hash = _expected_chat_template_hash(mid, chat_template_kwargs)
            except Exception as pin_exc:
                print(
                    f"\nABORT: expected_chat_template_hash registry "
                    f"unreadable for {mid}: {pin_exc}",
                    file=sys.stderr,
                )
                return 1

            # Resolve per-model decode budgets from
            # configs/models.yaml's fullsize_models entries. None = absent.
            try:
                action_max_tokens, belief_max_tokens = _model_decode_budgets(mid)
            except Exception as budget_exc:
                print(
                    f"\nABORT: model decode-budget registry unreadable for "
                    f"{mid}: {budget_exc}",
                    file=sys.stderr,
                )
                return 1
            if action_max_tokens is not None:
                print(f"  action_max_tokens: {action_max_tokens}")
            if belief_max_tokens is not None:
                print(f"  belief_max_tokens: {belief_max_tokens}")

            s_cmd = sweep_command(
                args.config,
                entry,
                out_dir,
                backend,
                manifest_path,
                served_weights_revision=served_weights_revision,
                tokenizer_revision=tokenizer_revision,
                git_sha_override=_existing_shard_git_sha(out_dir),
                expected_chat_template_hash=expected_hash,
                action_max_tokens=action_max_tokens,
                belief_max_tokens=belief_max_tokens,
            )

            os.makedirs(out_dir, exist_ok=True)
            # The pass record is the last write of a finished arm, so one left by an earlier run
            # must not outlive the start of this one: an --allow-existing rerun killed mid-sweep
            # would otherwise keep reading as finished to --skip-collected.
            try:
                os.remove(os.path.join(out_dir, EXACT_POLICY_PASS_FILENAME))
            except FileNotFoundError:
                pass
            # The sweep is checked on its own, not in a shared loop with
            # diagnose, because only run_sweep.py has a
            # code meaning "I ran to completion and lost shards". The same 5 out
            # of diagnose.py is an unknown failure of a different tool and keeps
            # the documented `return rc` passthrough below. See
            # PARTIAL_ARM_EXIT_CODE.
            rc = _run_step(s_cmd)
            if rc == _SWEEP_PARTIAL_EXIT_CODE:
                # Loud now (an operator watching live) and again in the final
                # summary (a reader scrolling to the end, hours later) - never
                # silent (no silent fallbacks). The sweep has already
                # printed its own per-shard FAILED lines to this console; they
                # are not restated here in weaker words.
                print(
                    f"\nPARTIAL: {mid}'s sweep ran to completion but LOST "
                    f"SHARDS (exit {rc}). This arm is INCOMPLETE and must not "
                    f"be read as a finished one - {out_dir}/"
                    f"{_SHARD_OUTCOMES_FILENAME} names every requested shard "
                    "and its status.\n"
                    "  SKIPPING its diagnose and exact-policy pass: both "
                    "report over whatever Parquet is present, so running them "
                    "here would produce an artefact silently describing fewer "
                    "matchups than it claims.\n"
                    "  CONTINUING to the next arm - the remaining arms are "
                    "hours of GPU work and have nothing to do with these "
                    "missing shards.",
                    file=sys.stderr,
                )
                partial_arms.append((mid, out_dir))
                write_exact_policy_pass_record(out_dir, exact_policy_enabled, None)
                print()
                continue
            if rc != 0:
                print(
                    f"\nABORT: {' '.join(s_cmd)} exited {rc} - experiment "
                    "stopped (no continue-on-error).",
                    file=sys.stderr,
                )
                return rc
            rc = _run_step(d_cmd)
            if rc != 0:
                print(
                    f"\nABORT: {' '.join(d_cmd)} exited {rc} - experiment "
                    "stopped (no continue-on-error).",
                    file=sys.stderr,
                )
                return rc
            # `ran` means "this arm's data is COMPLETE", which is what every
            # consumer of it below assumes: the parquet tally, the printed
            # analyse_stats.py commands, and the exact-policy block's
            # own "this arm still counts as `ran`" reasoning. A partial arm
            # satisfies none of that, so it is deliberately not appended - it is
            # carried in `partial_arms` and reported separately. Appending it
            # would hand the operator a ready-to-paste analyse command for an
            # arm missing shards.
            ran.append((mid, out_dir))

            if exact_policy_enabled:
                # Deliberate exception to this script's own
                # no-continue-on-error rule above (used for the sweep/diagnose
                # pair). Reasoning: the sweep's Parquet files are hours of GPU
                # time and are already safely on disk; this pass is ~574 cheap
                # calls against the same endpoint (reference-arm figure). Losing
                # the diagnostic to a transient failure is recoverable - re-run
                # scripts/exact_policy_pass.py by hand for this one arm - but
                # aborting the whole experiment here would throw away every
                # remaining arm's sweep+diagnose too, for a failure in a strictly
                # cheaper, strictly later, non-blocking diagnostic pass. So: a
                # failure here is reported loudly (stderr now, tracked, and
                # surfaced again in the final summary below) and never silent
                # (no silent fallbacks) - but it does not abort, and
                # this arm still counts as `ran` (its sweep+diagnose data is
                # completely unaffected by this pass failing).
                rc = _run_step(e_cmd)
                if rc != 0:
                    print(
                        f"\nWARNING: exact-policy pass FAILED for {mid} "
                        f"(exit {rc}): {' '.join(e_cmd)}\n"
                        f"  This arm's sweep and diagnose data are UNAFFECTED "
                        f"and {mid} still counts as run. Re-run the command "
                        "above by hand to retry just this diagnostic.",
                        file=sys.stderr,
                    )
                    exact_policy_failures.append(mid)
                write_exact_policy_pass_record(out_dir, True, rc)
            else:
                write_exact_policy_pass_record(out_dir, False, None)
            print()

        if args.dry_run:
            print("Dry run: no preflight, no writes, no subprocesses executed.")
            return 0

        print("=" * 72)
        print("Experiment complete.")
        # Arms entered, i.e. gated and launched - `ran` (complete) plus
        # `partial_arms` (launched, incomplete). The two lists partition
        # the arms the loop got through, so this is the honest denominator for
        # anything that happened per-arm before the sweep's outcome was known.
        n_attempted = len(ran) + len(partial_arms)
        total = 0
        for mid, out_dir in ran:
            n = len([f for f in os.listdir(out_dir) if f.endswith(".parquet")])
            total += n
            print(f"  {mid}: {n} parquet files in {out_dir}")
        print(f"  Total: {total} parquet files across {len(ran)} model(s)")
        for mid, out_dir in skipped_collected:
            print(f"  {mid}: SKIPPED, already finished in {out_dir}")
        if skipped_collected and not ran and not partial_arms:
            print(
                f"  Every arm ({len(skipped_collected)}) was skipped as already finished: "
                "no sweep, diagnose, pilot gate or exact-policy pass ran in this invocation."
            )
            return 0
        # The summary half of "recorded, not silent": the per-arm
        # PARTIAL notice above scrolled past hours ago, and a reader who scrolls
        # to the END must still be able to name which arms are incomplete and
        # where the evidence is. Distinct wording from the per-arm notice on
        # purpose, so a log search can tell the two halves apart.
        if partial_arms:
            print(
                f"  PARTIAL ARMS: {len(partial_arms)} of {n_attempted} arm(s) "
                "LOST SHARDS - their sweeps ran to completion, their data is "
                "INCOMPLETE, and their diagnose/exact-policy/analyse steps "
                "were SKIPPED. Do not read these as finished arms:",
                file=sys.stderr,
            )
            for mid, out_dir in partial_arms:
                n = len([f for f in os.listdir(out_dir) if f.endswith(".parquet")])
                print(
                    f"    {mid}: {n} parquet file(s) in {out_dir} - see "
                    f"{out_dir}/{_SHARD_OUTCOMES_FILENAME} for which shards "
                    "failed and why.",
                    file=sys.stderr,
                )
        # The summary half of "recorded, not silent". A reader of an unattended
        # cycle's log must be able to tell a SKIPPED gate from a PASSED one
        # without reconstructing the argv, so the two branches never share a
        # line: the per-arm pass prints "pilot gate: PASSED on <dir>", and a
        # skipped run prints no PASSED line anywhere.
        #
        # The denominator is arms attempted, not `len(ran)`: the gate runs once
        # per arm entered, before the sweep, so an arm that later went partial
        # was still gated (or skipped). With `len(ran)` a single-arm run that
        # went partial would print "SKIPPED ... for 1 of 0 arm(s)", a
        # denominator below its own numerator, which reads as a broken tally
        # rather than as a missing arm.
        if pilot_skipped:
            print(
                f"  PILOT GATE: SKIPPED via --skip-pilot-check for "
                f"{len(pilot_skipped)} of {n_attempted} arm(s): {pilot_skipped} - "
                "no pilot evidence exists for them. A skipped gate is not a "
                "passed gate.",
                file=sys.stderr,
            )
        else:
            print(f"  pilot gate: PASSED for all {n_attempted} arm(s).")
        if exact_policy_enabled:
            if exact_policy_failures:
                print(
                    f"  WARNING: exact-policy pass FAILED for {len(exact_policy_failures)} "
                    f"of {len(ran)} model(s): {exact_policy_failures} - see the WARNING(s) "
                    "above for the exact command to re-run by hand.",
                    file=sys.stderr,
                )
            else:
                print(f"  exact-policy pass: OK for all {len(ran)} model(s).")
        print()
        print("Analysis is offline - run per model tree:")
        for mid, out_dir in ran:
            if exact_policy_enabled:
                # Name the --exact-policy flag explicitly; see analyse_stats_command.
                print(f"  {' '.join(analyse_stats_command(out_dir, model_short(mid)))}")
            else:
                print(f"  python scripts/analyse_stats.py --data {out_dir} --out <analysis dir>")
        # "Do not abort" and "exit 0" are
        # different decisions. Every arm above completed (nothing was lost, no
        # early return happened) -- but if the exact-policy pass failed on any
        # of them, that must still be visible to whatever wraps this script, not
        # just to a human reading scrollback that an unattended multi-day cycle
        # nobody is watching will never show. See "Exit codes" in this module's
        # own docstring for the full contract.
        # PARTIAL is checked first - one code out of a multi-arm loop
        # has to rank outcomes, and missing data outranks a missing diagnostic
        # (an exact-policy failure leaves the arm's Parquet untouched and is
        # re-runnable by hand; a partial arm's rows do not exist). Both have
        # already been printed above; the code ranks them, it does not hide one.
        # Same direction as run_sweep's own rule that its 5 outranks its
        # schema-violation exit 1. See PARTIAL_ARM_EXIT_CODE.
        if partial_arms:
            return PARTIAL_ARM_EXIT_CODE
        if exact_policy_enabled and exact_policy_failures:
            return EXACT_POLICY_FAILURE_EXIT_CODE
        return 0
    except _TeardownSignal as sig_exc:
        # A signal converted to an exception (only possible when
        # stop_on_teardown installed the handlers above) - report it
        # exactly as loudly as every other abort path in this script, then
        # fall through to `finally` for the one teardown call site.
        print(
            f"\nABORT: {sig_exc} - experiment interrupted, tearing down "
            "before exit.",
            file=sys.stderr,
        )
        return 1
    finally:
        # One call site: covers normal completion (every `return` above),
        # every abort `return` inside the loop, and the signal path via the
        # `except` just above - restore handlers before stopping the unit so
        # a second signal during a slow stop() falls through to Python's
        # default behaviour instead of re-entering this same handler.
        if stop_on_teardown:
            for _sig, _prev in _prev_signal_handlers.items():
                signal.signal(_sig, _prev)
        _stop_served_unit()


if __name__ == "__main__":
    sys.exit(main())
