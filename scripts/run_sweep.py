#!/usr/bin/env python
"""Sweep: run 1 LM × N games × M opponents, logging each matchup to Parquet.

Usage:
    # Start the vLLM server first, then run the sweep:
    .venv/bin/python scripts/run_sweep.py --config <rendered config>.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
import time
from typing import Any, Optional, Sequence, cast


def _tqdm(it=None, **kw):
    """
    the iterable wrapped in a progress bar when tqdm is installed, else unchanged
    :param it: the iterable (iterable or None)
    :param kw: tqdm's keyword arguments (dict)
    :returns: the iterable, wrapped when tqdm is available
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return it
    return tqdm(it, **kw)


import polars as pl

from lmopt.agents.rl_selfplay import RLSelfPlayAgent
from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

from lmopt.games.depth_reference import DEPTH_REFERENCES, load_depth_reference
from lmopt.games.families import GAME_FAMILIES
from lmopt.games.loaders import load_game
from lmopt.games.policy_cache import announce_cache_state, load_or_solve_cfr
from lmopt.harness.match_runner import (
    SEAT_MODES,
    _get_git_sha,
    log_driven_discard_warning,
    log_posterior_fallback_warning,
    merge_driven_discard_counts,
    merge_posterior_fallback_counts,
)
from lmopt.logging.job import (
    JobIdentityUnavailable,
    PredecessorUnreadable,
    job_block,
    previous_attempts,
)
from lmopt.logging.run_meta import (
    DEFAULT_SEED_POLICY,
    finalize_run_meta,
    update_run_meta,
    write_run_meta,
)
from lmopt.logging.schema import (
    BASE_PROBE_TYPE,
    OFFPATH_PROBE_TYPE,
    PROBE_TYPE_DOMAIN,
    REQUIRED_COLUMNS,
)

# The variant registry, imported for the startup whitelist -- names, pairings
# and recorded absences all have one source, so a typo here cannot validate against a stale copy.
from lmopt.prompting import get_game_fns_for_game
from lmopt.provenance.code_tree import (
    ENV_HEAD as CODE_TREE_ENV_HEAD,
    REPO_ROOT as CODE_TREE_ROOT,
    SOURCE_JOB_CHECK,
    CodeTreeError,
    recorded_code_tree,
)
from lmopt.prompting.probe_variants import (
    BASE_VARIANT_NAME,
    DIAGNOSTIC_PROBE_TYPES,
    OPTION_ORDER_PROBE_TYPE,
    PROMPT_VARIANT_NAMES,
    VARIANT_NAME_BY_PROBE,
    expected_variant_name,
    require_variant_spec,
    variant_absence,
    variant_spec,
)
from lmopt.logging.tracking import init_tracking
from lmopt.logging.writer import read_parquet
from lmopt.orchestration.local import run_shards_local
from lmopt.agents.lm_vllm import (
    BELIEF_PROTOCOL_FREE,
    BELIEF_PROTOCOLS,
)
from lmopt.agents.lm_vllm import validate_action_top_logprobs
from lmopt.agents.lm_vllm import (
    validate_elicitation_mode as _validate_elicitation_mode_value,
)
from lmopt.prompting.memory import (
    MEMORY_RENDER_HASH_KEY,
    MEMORY_RUNGS,
    memory_render_hash,
)
from lmopt.logging.run_meta import write_json_atomic
from lmopt.orchestration.shard import (
    DEFAULT_REQUEST_TIMEOUT_S,
    ADAPTIVE_AGENT0_SENTINEL,
    RANDOM_AGENT0_SENTINEL,
    SCRIPTED_AGENT0_SENTINELS,
    MatchupSpec,
    driven_discard_sidecar_path,
    posterior_fallback_sidecar_path,
    timing_sidecar_path,
)
from lmopt.serving.chat_template import (
    ChatTemplateUnavailable,
    resolve_chat_template_hash,
)
from lmopt.serving.control import (
    ResolvedServing,
    ServingControlError,
    ServingController,
)

# A sweep that lost shards is not a successful sweep, and it is
# not the same failure as a sweep whose rows are malformed, so it gets its own
# code: exit 1 already means "schema violations detected" here, and the two
# call for different operator actions (re-run the missing shards vs.
# investigate the writer).
#
# Why 5 and not the obvious 2 or 3. `scripts/run_experiment.py` runs this
# script as a subprocess and does `return rc` verbatim on a non-zero result,
# so these codes are not private to this file -- they surface as
# run_experiment's own exit status, sharing one namespace with its
# EXACT_POLICY_FAILURE_EXIT_CODE (3) and PILOT_GATE_REFUSAL_EXIT_CODE (4) and
# with pilot_gate.py's GATE_FAILURE_EXIT_CODE (2, which run_experiment mirrors
# as _PILOT_GATE_FAILURE_EXIT_CODE). 5 is the first value none of them claims.
#
# It takes precedence over the schema-violation exit 1 when both apply,
# because a violation count computed over an arm that is missing shards is
# itself not trustworthy: incompleteness is the more fundamental defect and
# must be the one the operator sees first. Both are always printed.
SHARD_FAILURE_EXIT_CODE = 5

# Written into every sweep's out_dir, always -- including on a clean run, where
# it records n_failed: 0. That is the point: `pilot_gate.py`'s non-vacuity
# criterion is the precedent that an explicit zero is evidence and an absent
# artefact is not. Every requested shard appears here with a status, so a
# downstream reader can tell a shard that FAILED (present, status "failed")
# from one that was never requested (absent from the list) from a sweep that
# never got this far (no file at all) -- the distinction that stops an
# incomplete arm being read as a complete one.
SHARD_OUTCOMES_FILENAME = "shard_outcomes.json"


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required: uv pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full matchup sweep.")
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--model", default=None)
    p.add_argument("--vllm-base-url", default=None)
    p.add_argument("--n-hands", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--rl-iterations", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument(
        "--request-timeout-s",
        type=float,
        default=None,
        help="Override the 'request_timeout_s' config key: the "
        "per-request HTTP ceiling on every LM call a shard "
        "makes. Default and full derivation: "
        "lmopt.orchestration.shard.DEFAULT_REQUEST_TIMEOUT_S. "
        "Raise it for a slow-generating model rather than "
        "editing code (config over forks).",
    )
    p.add_argument(
        "--belief-protocol",
        default=None,
        choices=sorted(BELIEF_PROTOCOLS),
        help="Override the 'belief_protocol' config key: how the "
        "sibling belief call is DECODED. "
        "'free' (default) is unconstrained generation, the "
        "protocol every landed arm used; a named schema "
        "protocol constrains the belief call only. This is a "
        "MANIPULATED VARIABLE -- the "
        "same goofspiel info-set answers differently under "
        "each -- so it is stamped on every row, not just in "
        "run_meta.json, and switching it makes an arm "
        "incomparable with arms collected under another.",
    )
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument(
        "--agent0",
        default=None,
        help="Override agent0 mode ('lm' or 'random'); see _resolve_agent0. "
        "Used by Gate P1 to run the "
        "deterministic-agent parity pair with no vLLM server needed.",
    )
    p.add_argument(
        "--runner",
        default=None,
        choices=["serial", "local-pool", "ray"],
        help="Override the 'runner' config key. Default (None) derives the "
        "runner from max_workers ('serial' if 1, else 'local-pool'); "
        "'ray' explicitly selects the Ray shard runner regardless of "
        "max_workers.",
    )
    p.add_argument(
        "--ray-address",
        default=None,
        help="Override the 'ray_address' config key (passed to ray.init when "
        "this process doesn't already own an active Ray runtime). None "
        "starts/attaches to a local single-box cluster.",
    )
    p.add_argument(
        "--games",
        nargs="+",
        default=None,
        help="override the config's games list (one variant per W&B sweep run; keys are GAMES "
        "registry keys, unknown keys fail loudly at spec build)",
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Override the 'backend' provenance tag recorded on every row "
        "(FP-determinism audit). One backend per run - e.g. "
        "'dgx-spark-vllm' for remotely served LM inference.",
    )
    p.add_argument(
        "--serving-manifest",
        default=None,
        help="Path to a JSON manifest of BOX-ONLY serving fields "
        "(image_ref, vllm_args, container, unit), written by "
        "run_experiment.py as a byproduct of a swap it just "
        "performed. None (default) for a "
        "standalone run_sweep invocation, or a model that needed "
        "no swap -- those fields are then explicitly absent in "
        "run_meta's serving block, never fabricated.",
    )
    p.add_argument(
        "--served-weights-revision",
        default=None,
        help="Override the 'served_weights_revision' config key: the "
        "HF weights revision (snapshot hash) of the model the "
        "endpoint serves, stamped on every LM row. Resolved by the "
        "LAUNCHER, never by this script: the value lives in the "
        "SERVING box's HF cache, which this script's HTTP-only "
        "serving resolution cannot see. run_experiment.py "
        "resolves it per arm via "
        "lmopt.provenance.weights_cache's "
        "resolve_prelaunch_revision; a standalone real-LM "
        "invocation must pass it here (a real sweep without it "
        "ABORTS at startup). Irrelevant for agent0=random.",
    )
    p.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Override the 'tokenizer_revision' config key: the HF "
        "tokenizer revision (snapshot hash), when it differs "
        "from served_weights_revision, stamped on every LM row. "
        "Resolved by "
        "the LAUNCHER from the vLLM launch's own observed "
        "--tokenizer-revision argv, same as "
        "--served-weights-revision above; None (default, most "
        "arms) omits the flag and the column stays null -- no "
        "startup gate requires it, unlike served_weights_"
        "revision, since there is no declared field it must "
        "agree with.",
    )
    p.add_argument(
        "--git-sha-override",
        default=None,
        help="Override the resolved git_sha this sweep stamps on every "
        "shard, instead of resolving a FRESH sha via _get_git_sha(). "
        "Without this, a kill-and-resume "
        "(run_experiment.py --allow-existing) launches a fresh "
        "run_sweep.py process that re-resolves a new sha at its own "
        "startup, so an arm resumed across a HEAD move can carry two "
        "model_revision values across its two invocations. "
        "run_experiment.py passes this "
        "when it detects an --allow-existing resume (reading an "
        "existing shard's own model_revision); a standalone "
        "run_sweep.py invocation omits it and resolves fresh.",
    )
    p.add_argument(
        "--expected-chat-template-hash",
        default=None,
        help="Override the 'expected_chat_template_hash' config "
        "key: the per-model pin from configs/models.yaml. "
        "Compared at startup against the hash this script "
        "resolves from the live endpoint: a MISMATCH aborts "
        "before any shard runs; no pin prints one loud "
        "NOT-PINNED line and the run continues (pins roll "
        "out gradually, absence is recorded, never silent). "
        "run_experiment.py passes it per arm from the "
        "models.yaml registry. Irrelevant for agent0=random.",
    )
    p.add_argument(
        "--action-max-tokens",
        type=int,
        default=None,
        help="Override action channel max_tokens: per-model decode "
        "budget from configs/models.yaml. "
        "CLI wins over YAML; None = agent constructor default "
        "(ACTION_MAX_TOKENS in lm_vllm.py). run_experiment.py "
        "resolves and passes per model. Irrelevant for agent0=random.",
    )
    p.add_argument(
        "--belief-max-tokens",
        type=int,
        default=None,
        help="Override belief channel max_tokens: per-model decode "
        "budget from configs/models.yaml. "
        "CLI wins over YAML; None = agent constructor default "
        "(BELIEF_MAX_TOKENS in lm_vllm.py). run_experiment.py "
        "resolves and passes per model. Irrelevant for agent0=random.",
    )
    return p.parse_args(argv)


def _merge(cfg: dict, args: argparse.Namespace) -> dict:
    result = dict(cfg)
    overrides = {
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
        "n_hands": args.n_hands,
        "seed": args.seed,
        "out_dir": args.out_dir,
        "rl_iterations": args.rl_iterations,
        "max_workers": args.max_workers,
        "request_timeout_s": args.request_timeout_s,
        "belief_protocol": args.belief_protocol,
        "n_seeds": args.n_seeds,
        "agent0": args.agent0,
        "runner": args.runner,
        "ray_address": args.ray_address,
        "games": args.games,
        "backend": args.backend,
        "serving_manifest": args.serving_manifest,
        "served_weights_revision": args.served_weights_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "git_sha_override": args.git_sha_override,
        "expected_chat_template_hash": args.expected_chat_template_hash,
        "action_max_tokens": args.action_max_tokens,
        "belief_max_tokens": args.belief_max_tokens,
    }
    for k, v in overrides.items():
        if v is not None:
            result[k] = v
    result.setdefault("games", ["kuhn", "leduc", "goofspiel"])
    result.setdefault("opponents", ["cfr_ref", "bayesian", "rl", "always_bet"])
    result.setdefault("agent0", "lm")
    result.setdefault("n_hands", 50)
    result.setdefault("seed", 42)
    result.setdefault("out_dir", "data/m3")
    result.setdefault("cfr_iters", 1000)
    result.setdefault("rl_iterations", 1_000)
    result.setdefault("cfr_cache_dir", "data/cfr_policies")
    result.setdefault("rl_cache_dir", "data/rl_policies")
    # True makes the parent-side CFR and RL warm raise on a cache miss instead
    # of solving it inline: a collection job that omits this
    # key fails loudly on a cold reference, matching what every real
    # collection config sets explicitly. A config that genuinely wants
    # solve-inline behaviour (local dev, an ad hoc smoke run) sets
    # require_cache_hit: false itself.
    result.setdefault("require_cache_hit", True)
    result.setdefault("n_seeds", 1)
    result.setdefault("elicitation_mode", "sibling")
    # Which seat agent0 occupies (SEAT_MODES, match_runner.py) -- run-level
    # config (config over forks). "fixed0" matches MatchupSpec's own default,
    # so an unconfigured sweep is unchanged.
    result.setdefault("seat_mode", "fixed0")
    # The probe-arm block -- run-level config (config over forks), the surface
    # that makes a P1/P4 probe arm collectable at all. An empty block is "not a
    # probe", so an unconfigured sweep dispatches byte-identical specs. Shape
    # and validation: _validate_probe below.
    result.setdefault("probe", {})
    # The P6 forced opening -- run-level config (config over forks), the
    # manipulation the `offpath_drive` probe type names. None is "not a driven
    # arm", so an unconfigured sweep dispatches byte-identical specs.
    # Validation: _validate_drive_openings below, and the pairing with
    # `probe.type` is checked in both directions.
    result.setdefault("drive_openings", None)
    # Per-run decode budgets for the LM's action/belief calls -- run-level
    # config (config over forks). None means "the agent's constructor
    # defaults" (ACTION_MAX_TOKENS / BELIEF_MAX_TOKENS, lm_vllm.py), so a
    # config without the key dispatches byte-identical requests. Set for a
    # thinking model, which can spend the default cap on its trace before
    # answering. Recorded via resolved_config in run_meta.json; per-row cost
    # lands in the completion_tokens columns.
    result.setdefault("action_max_tokens", None)
    result.setdefault("belief_max_tokens", None)
    # Memory knobs -- which memory rung (M0 = no history, M1 =
    # cross-hand summary line, M2 = transcript over window k, M3 = LM-written
    # scratchpad) and k for M2 (the transcript window size). Defaults "M0" / 5
    # match the agent constructor defaults (lm_vllm.py) and the MatchupSpec
    # defaults (shard.py), so a config without the key dispatches
    # byte-identical specs. Recorded via resolved_config in run_meta.json;
    # arm-level variation is the use case.
    result.setdefault("memory_rung", "M0")
    result.setdefault("memory_window_k", 5)
    # chat_template_kwargs as run-level config, one serving yielding two arms
    # (for example Qwen3-32B think vs non-think). None means the field is
    # ABSENT from every request (the agent's _complete method omits it), so a
    # config without the key dispatches byte-identical requests. Only None/absent means
    # byte-identical wire; an explicit {} is sent and treated as a distinct arm
    # (loud NOT-PINNED, never a silent bare-pin match). Recorded via
    # resolved_config in run_meta.json; the served template renders per run.
    result.setdefault("chat_template_kwargs", None)
    # Run-level top-k logprobs on the ACTION call only. None means the request
    # carries no logprob fields; an integer k (the serving
    # box's --max-logprobs) adds `logprobs` and `top_logprobs`. Recorded via
    # resolved_config in run_meta.json; the values land in the per-call log. An invalid
    # value fails here, before any shard runs.
    result["action_top_logprobs"] = validate_action_top_logprobs(
        result.setdefault("action_top_logprobs", None)
    )
    result.setdefault("max_retries", 3)
    result.setdefault("max_workers", 1)
    # The per-request HTTP ceiling, run-level config (config over forks).
    # The default is referenced, never re-typed, so this file cannot drift
    # from the derivation recorded on the constant itself.
    result.setdefault("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
    # The belief decoding protocol is run-level config, not a constructor
    # argument reachable only from tests. Defaulted to the free protocol by
    # reference, never a re-typed "free", so this file cannot name a protocol
    # lm_vllm.py has renamed or dropped.
    result.setdefault("belief_protocol", BELIEF_PROTOCOL_FREE)
    # Decoding temperature is run-level config (config over forks), not a
    # code branch. 0.0 is LMVllmAgent's own constructor default.
    result.setdefault("temperature", 0.0)
    # runner: None (default) derives serial/local-pool from max_workers below;
    # an explicit "ray" config key (or --runner ray) selects the Ray shard
    # runner regardless of max_workers - Ray submits every shard at once and lets the cluster
    # schedule it, so max_workers is not a Ray concurrency cap.
    result.setdefault("runner", None)
    result.setdefault("ray_address", None)
    result.setdefault("ray_resources_per_shard", None)
    # backend is a load-bearing logged provenance column (the FP-determinism
    # audit) and has no default here: a config or --backend must name it, and
    # `_validate_backend` refuses at startup when neither does.
    # serving_manifest: None (default) for every standalone run_sweep
    # invocation, or a model whose endpoint needed no swap this run - see
    # _resolve_serving_block's docstring for what an absent manifest means
    # downstream.
    result.setdefault("serving_manifest", None)
    return result


def build_config(argv: list[str]) -> dict[str, Any]:
    """Parse argv and merge CLI overrides with YAML config (test/deploy seam)."""
    args = parse_args(argv)
    cfg = _load_yaml(args.config)
    return _merge(cfg, args)


# The supported elicitation modes (`lmopt.agents.lm_vllm`, imported, not
# redefined here) are the two elicitation_mode values `LMVllmAgent` actually
# branches on. Anything else falls to the agent's own bare `else`, which
# silently disables belief elicitation for the whole arm
# (belief_raw_completion null on every row) rather than failing -- a
# one-character typo (e.g. "siblings") reads as a perfect belief record
# downstream (pilot gate prints belief_fail=0.0000 (calls=0) and passes).
def _validate_elicitation_mode(cfg: dict[str, Any]) -> None:
    """Fail loudly at sweep startup on an unrecognised elicitation_mode
    (no silent fallbacks), before any GPU call or shard spawn.

    cfg["elicitation_mode"] is always present by this point (`_merge`
    setdefaults it to "sibling") and there is no --elicitation-mode CLI
    flag on this script, so this one check covers both the config-file path
    (an explicit YAML value, valid or not) and the CLI/default path (the
    setdefault, always a legal value) for every sweep run.

    Thin wrapper: the actual whitelist and check live in
    `lmopt.agents.lm_vllm` (`SUPPORTED_ELICITATION_MODES`,
    `validate_elicitation_mode`) and are imported, not redefined, here. This
    wrapper's only job is translating the shared function's `ValueError`
    into this script's own `sys.exit` convention, at this script's own
    startup call site (before any GPU call or shard spawn).
    """
    try:
        _validate_elicitation_mode_value(cfg["elicitation_mode"])
    except ValueError as exc:
        sys.exit(str(exc))


def _validate_require_cache_hit(cfg: dict[str, Any]) -> None:
    """
    fail at sweep startup unless require_cache_hit is a real boolean: a YAML
    string such as "false" is truthy and would silently arm the guard
    :param cfg: the merged sweep config, whose require_cache_hit key is set
    :returns: None; exits non-zero with a message on a non-bool value
    """
    value = cfg["require_cache_hit"]
    if not isinstance(value, bool):
        sys.exit(
            f"require_cache_hit must be true or false, got {value!r} "
            f"({type(value).__name__})."
        )


def _validate_seat_mode(cfg: dict[str, Any]) -> None:
    """Fail loudly at sweep startup on an unrecognised seat_mode
    (no silent fallbacks), before any CFR solve or shard dispatch.

    Same shape as `_validate_elicitation_mode` above, for the same reason: the
    library-level guard (`MatchupSpec.__post_init__`) is real, but the first
    spec is built inside the per-game loop, after that game's
    `load_or_solve_cfr` and `RLSelfPlayAgent.load_or_train`. So
    without this check a one-character typo in YAML costs a CFR solve and, on
    a cache miss, an XFP train before anything complains. This is the
    operator-level guard; `MatchupSpec`'s stays as the library-level one, and
    neither makes the other redundant.

    SEAT_MODES is imported from match_runner, never re-typed here, so a mode
    added or renamed there cannot leave this script validating against a stale
    copy. cfg["seat_mode"] is always present by this point (`_merge`
    setdefaults it to "fixed0") and there is no --seat-mode CLI flag, so this
    one check covers both the YAML path and the default path.

    probe_type is checked in `_validate_probe` below, alongside the `probe:`
    key that makes it settable from YAML.
    """
    mode = cfg["seat_mode"]
    if mode not in SEAT_MODES:
        sys.exit(f"Unsupported seat_mode: {mode!r} (expected one of {SEAT_MODES!r})")


def _validate_backend(cfg: dict[str, Any]) -> None:
    """
    fail at sweep startup unless the run names its serving backend: every row
    records it, and a label the run did not choose describes hardware it never ran on
    :param cfg: the merged sweep config, whose backend key may be absent (dict)
    :returns: None; exits with a message naming the key and both ways to set it
    """
    value = cfg.get("backend")
    if not isinstance(value, str) or not value.strip():
        sys.exit(
            f"'backend' is required, got {value!r}: set the `backend:` key in the "
            "config or pass --backend (it is stamped on every row and compared "
            "across the pieces of an arm)."
        )


def _validate_memory_rung(cfg: dict[str, Any]) -> None:
    """Fail loudly at sweep startup on an unrecognised memory_rung
    (no silent fallbacks), before any CFR solve or shard dispatch.

    Same shape as `_validate_elicitation_mode` and `_validate_seat_mode` above,
    for the same reason: the library-level guard could live in
    `LMVllmAgent.__init__`, but this operator-level check catches the error
    before a CFR solve or a cache miss's XFP train. MEMORY_RUNGS is imported
    from lmopt.prompting.memory, never re-typed here, so a rung added or
    renamed there cannot leave this script validating against a stale copy.
    cfg["memory_rung"] is always present by this point (`_merge` setdefaults it
    to "M0") and there is no --memory-rung CLI flag, so this one check covers
    both the YAML path and the default path.
    """
    rung = cfg["memory_rung"]
    if rung not in MEMORY_RUNGS:
        sys.exit(
            f"Unsupported memory_rung: {rung!r} (expected one of {MEMORY_RUNGS!r})"
        )


# Probe types that run at memory level M0 only, with the reason each states when refused.
_M0_ONLY_PROBE_REASONS = {
    OPTION_ORDER_PROBE_TYPE: (
        "the option-order run measures M0 prompts only, and a memory block changes the prompt "
        "whose option order it reverses"
    ),
}

# The keys a `probe:` block may carry. A whitelist rather than a `.get` with a
# default: `probe: {typ: relabel}` must not read as "not a probe" (no silent
# fallbacks), which is the same failure shape as the elicitation_mode
# typo that read as a perfect belief record.
_PROBE_KEYS = ("type", "prompt_variant")


def _resolve_probe(cfg: dict[str, Any]) -> tuple[str, str]:
    """The (probe_type, prompt_variant) pair every MatchupSpec of this sweep
    carries, resolved from the `probe:` block.

    `prompt_variant` is optional and defaults to the registry's name for the
    probe type, because there is exactly one variant scheme per probe type
    (`probe_variants.VARIANT_NAME_BY_PROBE`); spelling it out in YAML is
    allowed so a config can be read without consulting the registry, and it is
    then checked to agree rather than trusted.

    Call `_validate_probe` first: this function assumes the block's shape.
    """
    block = cfg["probe"] or {}
    probe_type = block.get("type", BASE_PROBE_TYPE)
    return probe_type, block.get("prompt_variant", expected_variant_name(probe_type))


def _validate_probe(cfg: dict[str, Any]) -> None:
    """Fail loudly at sweep startup on an unusable `probe:` block
    (no silent fallbacks), before any CFR solve or shard dispatch.

    Same shape and the same reason as `_validate_seat_mode` above: the
    library-level guard (`MatchupSpec.__post_init__`, which owns the identical
    checks against the identical imported constants) fires inside the per-game
    loop, after that game's `load_or_solve_cfr` and
    `RLSelfPlayAgent.load_or_train`, so without this a one-character typo costs
    a CFR solve, and on a cache miss an XFP train, before anything complains.

    PROBE_TYPE_DOMAIN, PROMPT_VARIANT_NAMES and the per-game availability are
    all imported (logging.schema, prompting.probe_variants), never re-typed, so
    this script cannot validate against a stale copy of any of them.

    The per-game check is the one this guard adds beyond MatchupSpec's: a
    game with a recorded absence for the configured probe type (goofspiel for
    relabel; four of five games for payoff_scale) can never produce a probe
    arm, and `_build_agent0`'s `require_variant_spec` would refuse it one shard
    at a time, after the caches were warm. Here it refuses the sweep, with the
    registry's own reason text.
    """
    block = cfg["probe"]
    if not isinstance(block, dict):
        sys.exit(
            f"'probe' must be a mapping of {list(_PROBE_KEYS)}, got {type(block).__name__}"
        )
    unknown = sorted(set(block) - set(_PROBE_KEYS))
    if unknown:
        sys.exit(
            f"Unknown key(s) in the 'probe' block: {unknown} "
            f"(expected one of {list(_PROBE_KEYS)})"
        )
    probe_type = block.get("type", BASE_PROBE_TYPE)
    if probe_type not in PROBE_TYPE_DOMAIN:
        sys.exit(
            f"Unsupported probe type: {probe_type!r} "
            f"(expected one of {sorted(PROBE_TYPE_DOMAIN)})"
        )
    variant = block.get("prompt_variant", expected_variant_name(probe_type))
    expected = expected_variant_name(probe_type)
    # One check, for the reason MatchupSpec.__post_init__ records: a separate
    # PROMPT_VARIANT_NAMES membership test is dead, since `expected` is always
    # a member and this equality therefore rejects everything the membership
    # test would. The legal names ride in the message.
    if variant != expected:
        sys.exit(
            f"probe type {probe_type!r} renders prompt_variant {expected!r}, but "
            f"the config asks for {variant!r} (known variants: "
            f"{sorted(PROMPT_VARIANT_NAMES)}). The two labels are one claim about "
            f"every row of the arm and must agree."
        )
    if probe_type in VARIANT_NAME_BY_PROBE:
        for game_name in cfg["games"]:
            family = GAME_FAMILIES.get(game_name, game_name)
            absence = variant_absence(family, probe_type)
            if absence is not None:
                sys.exit(
                    f"No {probe_type!r} variant exists for {game_name!r}, so this "
                    f"sweep cannot collect a probe arm for it: {absence.reason} "
                    f"What it would need: {absence.would_need}"
                )
    if probe_type in DIAGNOSTIC_PROBE_TYPES:
        # a diagnostic variant is built from the family's spec, and a game that maps to that
        # family can still have a factory that refuses it (the deep poker games), so the
        # registry's per-family answer above is not enough: build the functions now
        for game_name in cfg["games"]:
            family = GAME_FAMILIES.get(game_name, game_name)
            try:
                get_game_fns_for_game(
                    load_game(game_name), spec=variant_spec(family, probe_type)
                )
            except ValueError as exc:
                sys.exit(
                    f"The {probe_type!r} variant cannot render {game_name!r}, so this sweep "
                    f"cannot collect a probe arm for it: {exc}"
                )
    if probe_type in _M0_ONLY_PROBE_REASONS and cfg.get("memory_rung", "M0") != "M0":
        sys.exit(
            f"probe type {probe_type!r} runs at memory_rung M0 only, got "
            f"{cfg.get('memory_rung')!r}: {_M0_ONLY_PROBE_REASONS[probe_type]}"
        )
    if probe_type == "payoff_scale" and cfg.get("memory_rung", "M0") != "M0":
        _refuse_unscaled_replayed_state(cfg)
    _validate_drive_openings(cfg, probe_type)


def _refuse_unscaled_replayed_state(cfg: dict[str, Any]) -> None:
    """
    exit when a payoff-scale arm above M0 names a game whose prompt states payoff magnitudes
    from its state but whose memory terms carry no state view
    The M2 transcript replays each past state, so such a game would show the model its prompt's
    scale in the rules and the engine's own unscaled figures in the replayed history, two
    scales at once. The game is found from the variant and the terms, never from a list of names.
    :param cfg: the merged sweep config, whose `games` and `probe` are already validated (dict)
    :returns: None; exits with the game and the reason when any game is unscaled in replay
    """
    for game_name in cfg["games"]:
        family = GAME_FAMILIES.get(game_name, game_name)
        spec = require_variant_spec(family, "payoff_scale")
        terms = get_game_fns_for_game(load_game(game_name), spec=spec).memory_terms
        if spec.display_scale != 1 and (terms is None or terms.state_view is None):
            sys.exit(
                f"probe type 'payoff_scale' at memory_rung {cfg['memory_rung']!r} cannot run "
                f"{game_name!r}: its prompt states its pot and stacks at the variant's scale, "
                f"but the memory block replays each past state unscaled, so the model would "
                f"see two scales at once. Give the game a memory state view first."
            )


def _validate_drive_openings(cfg: dict[str, Any], probe_type: str) -> None:
    """Fail loudly at sweep startup on an unusable `drive_openings` list
    (no silent fallbacks), before any CFR solve or shard dispatch.

    The probe type and the manipulation must be configured together, checked
    in both directions, so a config can neither label an unmanipulated arm as a probe nor collect a
    driven arm that claims to be canonical play.

    The per-game legality walk is the analogue of the recorded-absence loop
    above: `run_match` refuses an illegal forced action per hand, but only
    inside a worker, after that game's caches are warm. It is a necessary, not
    a sufficient, check -- it follows the first chance outcome at each deal
    node, so on a game whose action legality depended on the deal it could pass
    here and still refuse at run time. For the canonical five, legality is a
    function of the action prefix alone, so in practice this is the whole
    check; where it is not, the runner's refusal is what holds.

    `cfg["drive_openings"]` and `cfg["games"]` are always present by this point
    (`_merge` setdefaults them to `None` and to the three-game default), and
    every production caller reaches here through `_merge` -- `main` and
    `build_config` are the only two. Off that path both subscripts are bare
    `KeyError`s, a one-word traceback out of the function whose entire purpose
    is that failures here carry a paragraph.
    """
    drive = cfg["drive_openings"]
    is_offpath = probe_type == OFFPATH_PROBE_TYPE
    if drive is None:
        if is_offpath:
            sys.exit(
                f"probe type {OFFPATH_PROBE_TYPE!r} needs a `drive_openings` list: "
                f"it manipulates the MATCH rather than the prompt, so without one "
                f"this sweep would collect an entirely ordinary arm and stamp "
                f"every row with a probe label."
            )
        return
    if not is_offpath:
        sys.exit(
            f"`drive_openings` was configured with probe type {probe_type!r}: a "
            f"driven arm must be labelled {OFFPATH_PROBE_TYPE!r}, or its rows "
            f"claim canonical play and every scorer will pool them with it."
        )
    if not isinstance(drive, list) or not drive:
        sys.exit(
            f"`drive_openings` must be a non-empty list of action integers, got "
            f"{drive!r}. An empty list is not 'no driving': it would label a "
            f"canonical arm as a probe."
        )
    bad = [a for a in drive if not isinstance(a, int) or isinstance(a, bool) or a < 0]
    if bad:
        sys.exit(
            f"`drive_openings` must hold non-negative action integers; {bad!r} "
            f"is not. Actions are OpenSpiel action ids, not names."
        )
    for game_name in cfg["games"]:
        _refuse_illegal_drive(game_name, drive)


def _refuse_illegal_drive(game_name: str, drive: list[int]) -> None:
    """Walk one game's opening with the forced actions, first chance outcome
    at each deal node, and exit if the sequence cannot be played."""
    from lmopt.games.loaders import load_game

    state = load_game(game_name).new_initial_state()
    for i, action in enumerate(drive):
        while state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        if state.is_terminal():
            sys.exit(
                f"`drive_openings` {drive!r} ends the hand of {game_name!r} after "
                f"{i} action(s): no agent would ever choose, so the arm would be "
                f"an empty Parquet."
            )
        legal = state.legal_actions()
        if action not in legal:
            sys.exit(
                f"`drive_openings` {drive!r} is not playable in {game_name!r}: "
                f"forced action {action} at position {i} is not in the legal "
                f"actions {legal} of player {state.current_player()}."
            )
        state.apply_action(action)
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    if state.is_terminal():
        sys.exit(
            f"`drive_openings` {drive!r} ends the hand of {game_name!r}: the "
            f"state is terminal once the forced actions are applied, so no agent "
            f"ever chooses and the arm would be an empty Parquet."
        )


def _arm_path_component(probe_type: str, prompt_variant: str) -> Optional[str]:
    """The filename component that distinguishes this arm from the base arm.

    The variant names the arm whenever there is one, because that is what the
    paired scorer joins on and what a reader recognises. A probe type that
    leaves the prompt canonical (`offpath_drive`) carries variant "base", so
    the variant alone would give a driven arm the base arm's own filenames and
    overwrite it; there the probe type names it instead. A base arm gets no
    component at all, so base filenames keep the plain convention.
    """
    if prompt_variant != BASE_VARIANT_NAME:
        return prompt_variant
    if probe_type != BASE_PROBE_TYPE:
        return probe_type
    return None


def _shard_out_path(
    out_dir: str,
    game_name: str,
    opp_name: str,
    seed_idx: int,
    n_seeds: int,
    prompt_variant: str,
    probe_type: str,
) -> str:
    """The Parquet path one shard writes.

    Base arms keep the landed convention byte-identically
    (`{game}_lm_vs_{opp}[_seed{i}].parquet`); a probe arm appends its arm
    component (`_arm_path_component`) after the seed token.

    Why a variant component at all: without it a probe sweep pointed at a
    base arm's `out_dir` would overwrite it shard for shard, with
    `os.makedirs(..., exist_ok=True)` and no refusal anywhere, and the paired
    scorer would refuse the pair only after the base arm was destroyed.

    Why after the seed token. Readers parse this convention with
    `(.+?)_lm_vs_(.+)_seed(\\d+)\\.parquet$` (`analyse_stats.py`'s shard loader). A variant
    placed before the seed (`..._lm_vs_bayesian.relabel_v1_seed0.parquet`) matches that
    pattern, with the greedy opponent group swallowing the variant, so a reader would invent
    an opponent called "bayesian.relabel_v1". Placed after the seed the pattern does not match
    at all, and the reader prints a named skip.
    """
    stem = f"{game_name}_lm_vs_{opp_name}"
    if n_seeds > 1:
        stem = f"{stem}_seed{seed_idx}"
    component = _arm_path_component(probe_type, prompt_variant)
    if component is not None:
        stem = f"{stem}.{component}"
    return os.path.join(out_dir, f"{stem}.parquet")


def _discover_arm_shard_parquets(out_dir: str) -> list[str]:
    """Every shard Parquet currently sitting in this one arm's `out_dir`,
    sorted for a deterministic aggregation order.

    Scoped to exactly this named directory -- `os.listdir` returns only
    `out_dir`'s own entries -- never a cross-arm glob (a `data/run_experiment/
    */` glob spans preserved archive dirs). Filtered to files ending `.parquet`, the extension
    `_shard_out_path` always writes; non-shard files in the same directory
    (`run_meta.json`, the sidecars themselves, `shard_outcomes.json`) are
    excluded by construction, not by a second check.
    """
    if not os.path.isdir(out_dir):
        return []
    return sorted(
        os.path.join(out_dir, name)
        for name in os.listdir(out_dir)
        if name.endswith(".parquet")
    )


TIMING_KEYS = ("started_utc", "finished_utc", "duration_s")
#: Null on an entry whose sidecar was absent or read cleanly; the reason when the sidecar existed
#: and could not be read, so "no sidecar" and "a sidecar we could not read" differ in the record.
TIMING_ERROR_KEY = "timing_error"
ENTRY_TIMING_KEYS = TIMING_KEYS + (TIMING_ERROR_KEY,)


def _shard_timing(out_path: str) -> dict[str, Any]:
    """
    the clock of one shard, read from the sidecar its worker wrote
    A shard with no sidecar (collected before this existed, or killed before it started) and a
    shard whose sidecar holds no finish both read as None for the missing values: unknown, never
    zero and never an estimate. An unreadable sidecar also reads as unknown, is printed, and
    carries the reason under `timing_error`, so it is not recorded exactly like an absent one.
    Nothing here refuses: a timing problem must not fail a sweep at its last step.
    :param out_path: the shard's Parquet path (str)
    :returns: a dict with exactly ENTRY_TIMING_KEYS: each timing value the sidecar recorded or
        None, and `timing_error` None unless the sidecar existed and could not be read
    """
    path = timing_sidecar_path(out_path)
    unknown = {key: None for key in ENTRY_TIMING_KEYS}
    if not os.path.exists(path):
        return unknown
    try:
        with open(path) as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        reason = f"unreadable ({exc})"
    else:
        if isinstance(record, dict):
            return {
                **{key: record.get(key) for key in TIMING_KEYS},
                TIMING_ERROR_KEY: None,
            }
        reason = "not a JSON object"
    print(
        f"  NOTE: timing sidecar {path} is {reason}; its timing is recorded as unknown"
    )
    return {**unknown, TIMING_ERROR_KEY: reason}


def _aggregate_arm_sidecars(
    out_dir: str,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Pool posterior-fallback and driven-discard sidecars from every shard
    Parquet sitting in `out_dir` right now -- not just the shards this
    invocation of `main()` dispatched.

    A sweep killed mid-run and resumed via `run_experiment.py
    --allow-existing` launches a FRESH run_sweep.py process whose own `paths`
    (from `run_shards_local`/`run_shards_ray`) covers only the remaining
    shards. Aggregating by zipping `built` against that `paths` list would
    make a resumed invocation's run_meta.json silently reflect only the
    resume's shards, leaving the first invocation's sidecars unread in the
    same `out_dir`. This function
    reads the directory instead, which is what makes the tally an ARM-level
    fact (what run_meta.json claims to hold) rather than an
    invocation-level one.

    Re-run-shard de-duplication falls out of the filesystem, matching how
    shard outputs already de-duplicate on `--allow-existing`: a shard's
    identity is its Parquet path (`_shard_out_path` is a pure function of
    game/opponent/seed/variant, checked above), and `run_matchup_shard`
    (src/lmopt/orchestration/shard.py) opens both its Parquet and its two
    sidecars in truncating ("w") mode. Two invocations that both play the
    same shard leave exactly one Parquet and one pair of sidecars at that
    path -- the later invocation's write physically replaces the earlier
    one's. There is never a second, orphaned sidecar for one shard to
    disagree with, so "the latest write wins" is a fact about the directory
    this function reads, not a merge rule this function has to implement.
    """
    posterior_fallback_counts: dict[str, dict[str, int]] = {}
    driven_discard_counts: dict[str, dict[str, int]] = {}
    for shard_parquet in _discover_arm_shard_parquets(out_dir):
        sidecar_path = posterior_fallback_sidecar_path(shard_parquet)
        if os.path.exists(sidecar_path):
            with open(sidecar_path) as f:
                merge_posterior_fallback_counts(posterior_fallback_counts, json.load(f))
        discard_sidecar = driven_discard_sidecar_path(shard_parquet)
        if os.path.exists(discard_sidecar):
            with open(discard_sidecar) as f:
                merge_driven_discard_counts(driven_discard_counts, json.load(f))
    return posterior_fallback_counts, driven_discard_counts


def _refuse_certified_dir(out_dir: str) -> None:
    """
    refuse to run a sweep into a directory that holds a stamped `acceptance.json`, whatever
    the caller's allow-existing setting says. A certified directory is never rerun into: the
    rerun goes to a fresh directory, or the certified one is deliberately moved aside first
    :param out_dir: the sweep's output directory (str)
    :raises SystemExit: naming the acceptance file
    """
    stamp = os.path.join(out_dir, "acceptance.json")
    if os.path.lexists(stamp):
        sys.exit(
            f"{stamp} exists, so {out_dir} is certified and a sweep will not run into it. "
            "Choose a fresh out_dir, or move the certified directory aside deliberately."
        )


def _refuse_foreign_arm(
    out_dir: str,
    probe_type: str,
    prompt_variant: str,
    drive_openings: Optional[tuple[int, ...]],
    seat_mode: str,
) -> None:
    """Refuse to write this arm into a directory that already holds a different
    arm's `run_meta.json`.

    The variant component above stops the shards colliding, but
    `run_meta.json` is one file per directory, so a probe sweep aimed at a base
    arm's `out_dir` would still overwrite the one file whose entire job is
    provenance -- leaving the base arm's shards described by the probe arm's
    config.

    Scoped to a genuinely different arm, not to any second run: re-running a
    failed sweep into its own directory is the normal retry path and stays
    allowed. A `run_meta.json` with no `probe` key is read as a base arm.

    The arm's identity includes its forced opening. Two off-path arms driven into different subgames carry the same
    (probe_type, prompt_variant) pair and the same filenames -- the filename
    component is the probe type, deliberately, since encoding a list of action
    ids in a path is fragile -- so the pair alone reads them as one arm and the
    second silently overwrites the first. The drive is already in
    `run_meta.json` (it is run-level config), so the identity is read from
    there, not reconstructed. An absent key is None, which is what every base
    and prompt-probe arm carries.

    And it includes its seat mode. The same hole, one axis over: a
    `fixed0` arm and a `random` control arm carry the same
    (probe_type, prompt_variant, drive_openings) triple -- both are ordinarily
    base arms with no probe and no drive -- and `_shard_out_path` gives them
    byte-identical filenames, because seat mode is not a filename component.
    So the second run overwrites the first's `run_meta.json` and its shards,
    and the pair a seat contrast is built from is exactly a fixed arm beside a
    random one. `seat_mode` is already in `resolved_config` (written by the
    same `_merge` that setdefaults it to "fixed0"), so like the drive it is
    read rather than reconstructed; a meta with no seat_mode key is refused
    below rather than guessed.
    """
    meta_path = os.path.join(out_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        return
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
        if not isinstance(meta, dict):
            raise ValueError(f"top-level JSON is {type(meta).__name__}, not an object")
        resolved = meta.get("resolved_config", {}) or {}
        existing = resolved.get("probe", {}) or {}
        existing_drive = resolved.get("drive_openings")
        has_seat_mode = "seat_mode" in resolved
        existing_seat_mode = resolved.get("seat_mode")
    except (OSError, ValueError):
        # An unreadable meta is not evidence of a matching arm, and guessing
        # would be the silent fallback this guard exists to remove.
        sys.exit(
            f"{meta_path} exists but could not be read, so this sweep cannot "
            f"establish which arm already occupies {out_dir}. Move it aside or "
            f"choose a different out_dir."
        )
    if not has_seat_mode:
        # An absent seat mode is not read as "fixed0": the arm already in this directory
        # may have played another one, and the identity check below would then call it
        # the same arm and overwrite its shards.
        sys.exit(
            f"{meta_path} exists but its resolved_config records no seat_mode, so this sweep "
            f"cannot establish whether the arm already in {out_dir} used the seat mode this "
            f"run asks for. Move it aside or choose a different out_dir."
        )
    existing_type = existing.get("type", BASE_PROBE_TYPE)
    existing_variant = existing.get(
        "prompt_variant", expected_variant_name(existing_type)
    )
    # Compared as tuples so a JSON list and the config's tuple are one value;
    # None and () are both "not driven".
    existing_drive_key = tuple(existing_drive or ())
    drive_key = tuple(drive_openings or ())
    if (existing_type, existing_variant, existing_drive_key, existing_seat_mode) != (
        probe_type,
        prompt_variant,
        drive_key,
        seat_mode,
    ):
        sys.exit(
            f"{out_dir} already holds a run_meta.json for a DIFFERENT arm "
            f"(probe type {existing_type!r}, prompt_variant {existing_variant!r}, "
            f"drive_openings {list(existing_drive_key) or None}, "
            f"seat_mode {existing_seat_mode!r}); this sweep is "
            f"{probe_type!r}/{prompt_variant!r}/{list(drive_key) or None}/"
            f"{seat_mode!r}. Writing "
            f"here would overwrite that arm's provenance while its shards stay on "
            f"disk, leaving them described by the wrong config -- and two arms "
            f"driven into different subgames share their filenames as well, so the "
            f"shards themselves would be overwritten, as do two arms that differ "
            f"only in seat_mode. Use a separate out_dir per arm; the paired scorer "
            f"reads two directories."
        )


def _wait_for_server(base_url: str, model: str, timeout: int = 300) -> None:
    """Poll vLLM until a real inference succeeds (covers /health + GPU load + JIT warmup)."""
    from openai import OpenAI, APITimeoutError, APIConnectionError

    deadline = time.monotonic() + timeout
    warm_client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=300.0)

    print(
        f"  Waiting for vLLM server (polling every 5s, up to {timeout}s) ...",
        flush=True,
    )
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            warm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print("  Server ready.", flush=True)
            return
        except (APITimeoutError, APIConnectionError) as exc:
            last_exc = exc
            time.sleep(5)
        except Exception as exc:
            last_exc = exc
            time.sleep(5)
    sys.exit(
        f"\nServer not ready after {timeout}s: {last_exc}\nIs the vLLM server running?"
    )


def _resolve_agent0(cfg: dict) -> str:
    """Resolve the ``agent0`` config key to a MatchupSpec.model value.

    "lm" (default): validate model/vllm_base_url are set and wait for the vLLM
    server to be ready, then return the model id. "random": no server needed -
    return the shard-level RANDOM_AGENT0_SENTINEL (config over forks: this is
    plumbing for the existing agent0 config key, not a new mode - see
    lmopt.orchestration.shard).
    """
    mode = cfg.get("agent0", "lm")
    if mode == "random":
        return RANDOM_AGENT0_SENTINEL
    if mode == "adaptive":
        return ADAPTIVE_AGENT0_SENTINEL
    if mode != "lm":
        sys.exit(f"Unsupported agent0: {mode!r} (expected 'lm', 'random' or 'adaptive')")

    model = cfg.get("model")
    base_url = cfg.get("vllm_base_url")
    if not model:
        sys.exit("'model' is required in config or via --model")
    if not base_url:
        sys.exit("'vllm_base_url' is required in config or via --vllm-base-url")
    _wait_for_server(base_url, model)
    return model


def _manifest_vllm_args(
    manifest: dict[str, Any],
) -> tuple[tuple[str, ...] | None, str | None]:
    """
    the launch argv a serving manifest records, with the reason when it records none
    :param manifest: the parsed serving manifest, or {} when none was supplied (dict)
    :returns: (argv, None) when an argv was recorded, else (None, why it was not) (tuple)
    :raises ValueError: when `vllm_args` is neither a list nor null
    """
    if "vllm_args" not in manifest:
        return None, "no serving manifest recorded vllm_args"
    recorded = manifest["vllm_args"]
    if recorded is not None and not isinstance(recorded, list):
        raise ValueError(
            f"serving manifest vllm_args must be a list or null, got {type(recorded).__name__}"
        )
    if recorded:
        return tuple(recorded), None
    if recorded is None:
        return None, manifest.get("vllm_args_unobserved_reason") or (
            "the serving manifest records vllm_args as not observed and gives no reason"
        )
    return None, (
        "the serving manifest records an empty list, written before a launch argv that "
        "could not be observed had its own representation"
    )


def _resolve_serving_block(
    base_url: str,
    model_id: str,
    manifest_path: str | None,
    chat_template_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Resolve the run_meta.json serving block for a real-LM agent0 run.

    HTTP-ONLY, resolved directly from the endpoint this run is about to
    hammer, never from a config file's intent: served model id +
    max_model_len via ServingController.current() (ssh_host="" is not a real
    host - current() never shells out, it only does /v1/models over HTTP,
    same idiom as scripts/run_experiment.py's refactored preflight()), and
    chat_template_hash via resolve_chat_template_hash's /tokenize +
    /detokenize round trip.

    The BOX-ONLY fields (image_ref, vllm_args, container, unit, and, for the
    pbs-injob lane, serving_env, serving_env_hash) come from
    `manifest_path` if run_experiment.py supplied one as a byproduct of
    either a swap or a no-swap `resolve()` (a model already correctly served
    still gets a manifest); absent (never fabricated: no silent fallbacks)
    for a standalone run_sweep invocation or a config with no ssh_host
    configured. The returned dict also carries a "box_fields_source" sibling
    key disambiguating four states: `None` ("no manifest was ever supplied" - the box-only
    fields mean "never looked"), `"swap"` (the manifest is a byproduct of
    `swap_to()` - the fields mean "looked, after swapping"), `"resolve"` (the
    manifest is a byproduct of the no-swap `resolve()` call - the fields mean
    "looked, without swapping"), `"pbs-injob"` (the manifest is a byproduct of
    `InJobServing.resolve()` - image_ref is None as an observed fact rather
    than an unverified one; serving_env/serving_env_hash carry this lane's
    provenance instead). `"swap"` vs `"resolve"` matters: without the
    distinction a no-swap arm would falsely record that a swap produced its
    box fields.

    Fails loudly (sys.exit) here, at sweep startup, before any run_meta.json
    or shard exists - never logs a null chat_template_hash across an
    unattended multi-hour run and only discovers the endpoint problem later
    (no silent fallbacks).
    """
    controller = ServingController(ssh_host="", base_url=base_url)
    try:
        served = controller.current()
    except ServingControlError as exc:
        sys.exit(
            f"serving resolution FAILED for {model_id!r}: {base_url} "
            f"answered but could not be parsed ({exc}). Aborting before any "
            "shard runs."
        )
    if served is None:
        sys.exit(
            f"serving resolution FAILED for {model_id!r}: {base_url} is "
            "unreachable. Aborting before any shard runs."
        )
    if model_id not in served.served_ids:
        sys.exit(
            f"serving resolution FAILED for {model_id!r}: {base_url} serves "
            f"{list(served.served_ids)!r} instead. Aborting before any "
            "shard runs."
        )

    try:
        chat_template_hash = resolve_chat_template_hash(
            base_url, model_id, chat_template_kwargs=chat_template_kwargs
        )
    except ChatTemplateUnavailable as exc:
        sys.exit(
            f"serving resolution FAILED for {model_id!r}: chat-template "
            f"fingerprint unavailable from {base_url} ({exc}). Aborting "
            "before any shard runs, rather than logging a null "
            "chat_template_hash across the whole sweep."
        )

    manifest: dict[str, Any] = {}
    if manifest_path:
        # A missing/corrupt manifest must fail
        # loudly through this module's own sys.exit idiom, not a bare
        # FileNotFoundError/JSONDecodeError traceback.
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                raise ValueError(
                    f"top-level JSON is {type(manifest).__name__}, not an object"
                )
        except (OSError, ValueError) as exc:
            sys.exit(
                f"serving resolution FAILED for {model_id!r}: could not "
                f"read --serving-manifest {manifest_path!r}: {exc}. "
                "Aborting before any shard runs."
            )
        # Cross-model misattribution guard: the manifest carries an
        # Identity stamp (run_experiment.py's `_write_serving_manifest`,
        # never used as a resolved value here) precisely so a stale or
        # mismatched manifest - e.g. left over from a previous model's run,
        # or handed to the wrong --model by a scripting error - is refused
        # loudly rather than silently attributing its box-side provenance
        # to this model. A manifest with no identity stamp at all (an older
        # format, or a hand-authored one) skips the check rather than
        # failing on an absent field it never had a contract to carry.
        # Correspondence-CHECK: serving-manifest-identity
        manifest_model_id = manifest.get("model_id")
        if manifest_model_id is not None and manifest_model_id != model_id:
            sys.exit(
                f"serving resolution FAILED for {model_id!r}: "
                f"--serving-manifest {manifest_path!r} was captured for "
                f"{manifest_model_id!r} instead - refusing to attribute "
                "another model's box-side provenance to this run. "
                "Aborting before any shard runs."
            )

    vllm_args, vllm_args_unobserved_reason = _manifest_vllm_args(manifest)
    resolved = ResolvedServing(
        model_id=model_id,
        base_url=base_url,
        image_ref=manifest.get("image_ref"),
        vllm_args=vllm_args,
        vllm_args_unobserved_reason=vllm_args_unobserved_reason,
        container=manifest.get("container"),
        max_model_len=served.max_model_len,
        chat_template_hash=chat_template_hash,
        unit=manifest.get("unit"),
        # PBS lane: carried through from the manifest like the other
        # box-only fields. Omitting them here would drop the only serving-stack
        # provenance a pbs-injob arm has -- it records no image digest, by
        # design -- while every unit test on the dataclass and every check on
        # the manifest still passed. None on providers that record an
        # image_ref instead.
        serving_env=manifest.get("serving_env"),
        serving_env_hash=manifest.get("serving_env_hash"),
        gpu_name=manifest.get("gpu_name"),
        gpu_memory_total_mib=manifest.get("gpu_memory_total_mib"),
    )
    serving_block = resolved.to_run_meta()
    # Within the block, `None`/`[]` on the four box-only fields
    # (image_ref/vllm_args/container/unit) means several different things
    # that this key distinguishes - never looked (no manifest at all) vs.
    # looked-but-unverifiable, and looked via a swap vs. looked via a no-swap
    # resolve(). ResolvedServing itself (the lmopt.serving.control
    # dataclass) carries none of this, and it is not this module's place to
    # editorialise that dataclass - so it is added as a sibling key instead,
    # read from the manifest's own "producer" field (run_experiment.py's
    # `_write_serving_manifest`, required keyword, no default): "swap" or
    # "resolve" when a manifest was supplied and stamped its producer; the
    # neutral "box-manifest" fallback is for a manifest that was supplied but
    # omits that field (never written by current code, but never trusted to
    # silently guess "swap" either); None when no manifest was supplied at
    # all (matches the run_meta-level "not set" vs "not observed"
    # distinction one layer down).
    if manifest_path:
        box_fields_source = manifest.get("producer") or "box-manifest"
    else:
        box_fields_source = None
    serving_block["box_fields_source"] = box_fields_source
    return serving_block


def _extract_and_update_prompt_spec_hash_from_shards(
    meta_path: str, shard_summaries: list[dict[str, Any]]
) -> None:
    """
    after shards complete, extract prompt_spec_hash
    from the landed parquets and record it in run_meta if unique across shards.

    This reads the hash off the data (the rendered truth). A sweep constructs
    agents per-shard, so no agent object is available at sweep startup; but
    every row is stamped with prompt_spec_hash, so we read it from the
    completed parquets.

    Exactly one distinct hash across all successful shards -> record it at the
    top level of run_meta.json. Zero or several -> leave the key absent and say
    so out loud; several means the prompt changed mid-sweep, which is a data
    integrity concern, and zero means nothing on this arm was stamped.

    No silent skip of ANY SHARD (no silent fallbacks). The whole point of the
    function is to detect a multi-hash run, so swallowing one shard's read
    error could collapse a real 2-hash run into a 1-hash "record it" outcome
    -- the single figure an analyst would then trust would be fabricated. Any
    shard the
    sweep called `completed` whose parquet cannot be read is therefore reported
    by path and exception, and the key is REFUSED for the whole run: an
    unreadable shard means the hash set is unknown, not unique.
    :param meta_path: Path to the run_meta.json file written at sweep startup.
    :param shard_summaries: `main`'s `summary` list -- one dict per requested
        shard, with "status" in {"completed", "failed"} and "out" holding
        that shard's parquet path.
    """
    # Collect all completed shard parquets. The status literal is "completed",
    # which is what `main` appends and what SHARD_OUTCOMES_FILENAME records;
    # any other literal (e.g. "succeeded", a status the sweep never emits)
    # would return at the guard below on every real run and never open a
    # shard.
    successful_shards = [
        s["out"] for s in shard_summaries if s.get("status") == "completed"
    ]
    if not successful_shards:
        print(
            "WARNING: no shard reported `completed`, so no prompt_spec_hash could "
            "be read off the data; the run_meta key remains absent. "
            "shard_outcomes.json says what happened to each shard."
        )
        return

    # Extract prompt_spec_hash values from rows in successful shards.
    hashes: set[str] = set()
    unreadable: list[str] = []
    for shard_path in successful_shards:
        if not os.path.exists(shard_path):
            unreadable.append(
                f"{shard_path}: shard reported `completed` but its parquet is absent"
            )
            continue
        try:
            df = read_parquet(shard_path)
        except Exception as exc:  # - reported, never swallowed
            unreadable.append(f"{shard_path}: {type(exc).__name__}: {exc}")
            continue
        # prompt_spec_hash is present on LM rows, None on non-LM rows.
        # Collect only non-None values.
        if "prompt_spec_hash" in df.columns:
            shard_hashes = df["prompt_spec_hash"].drop_nulls().unique().to_list()
            hashes.update(h for h in shard_hashes if h is not None)

    if unreadable:
        print(
            f"WARNING: {len(unreadable)} of {len(successful_shards)} successful "
            "shard parquet(s) could not be read, so the sweep's prompt_spec_hash "
            "set is UNKNOWN rather than unique; the run_meta key is refused. "
            f"Hashes seen on the readable shards: {sorted(hashes)}."
        )
        for detail in unreadable:
            print(f"  unreadable shard: {detail}")
        return

    # If exactly one distinct hash, update run_meta.
    if len(hashes) == 1:
        hash_value = list(hashes)[0]
        with open(meta_path) as f:
            meta = json.load(f)
        meta["prompt_spec_hash"] = hash_value
        # Write it back atomically using the same pattern as write_run_meta.
        write_json_atomic(meta_path, meta)
    elif len(hashes) > 1:
        # Multiple distinct hashes: data integrity issue, warn and leave key absent.
        print(
            f"WARNING: Sweep produced {len(hashes)} distinct prompt_spec_hash values "
            f"across shards ({hashes}). This suggests the prompt spec changed "
            "mid-sweep. The run_meta key remains absent; investigate the run's config "
            "and shard logs."
        )
    else:
        # Zero hashes. The ordinary case for an all-non-LM sweep (nothing was
        # decoded, so nothing was stamped), but indistinguishable from an LM
        # sweep whose stamping regressed -- so it is said out loud rather than
        # returned silently ("0/many -> omit + warning").
        print(
            f"WARNING: no prompt_spec_hash value on any row across "
            f"{len(successful_shards)} successful shard(s); the run_meta key "
            "remains absent. Expected for a sweep with no LM agent; on an LM arm "
            "it means the per-row stamp is missing and the arm's prompt "
            "provenance is unrecoverable from run_meta."
        )


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(args.config)
    cfg = _merge(cfg, args)
    _validate_require_cache_hit(cfg)
    _validate_elicitation_mode(cfg)
    _validate_seat_mode(cfg)
    _validate_backend(cfg)
    _validate_memory_rung(cfg)
    # Same startup position as the validators above, and for the same reason -- before _resolve_agent0 (which waits on a vLLM server) and
    # before the first CFR solve.
    _validate_probe(cfg)
    probe_type, prompt_variant = _resolve_probe(cfg)
    # A tuple, because MatchupSpec is frozen and therefore
    # hashable, and YAML hands over a list. The list itself stays in the
    # resolved config that run_meta.json records, so the arm's provenance says
    # which subgame it was driven into.
    drive_openings = (
        tuple(cfg["drive_openings"]) if cfg["drive_openings"] is not None else None
    )
    # Before any CFR solve, any endpoint wait, and any write: refuse to land
    # this arm on top of a different one's provenance.
    _refuse_certified_dir(cfg["out_dir"])
    _refuse_foreign_arm(
        cfg["out_dir"], probe_type, prompt_variant, drive_openings, cfg["seat_mode"]
    )

    games_list: list[str] = cfg["games"]
    opponents_list: list[str] = cfg["opponents"]
    out_dir: str = cfg["out_dir"]
    n_seeds: int = cfg["n_seeds"]
    base_seed: int = cfg["seed"]
    max_workers: int = cfg["max_workers"]
    os.makedirs(out_dir, exist_ok=True)

    # The job this attempt belongs to, and the earlier attempts already in this directory,
    # both resolved before the endpoint wait and before `write_run_meta` replaces the file
    # they are read from. A cluster run that cannot name its job, or a `run_meta.json` that
    # cannot be read as a history, stops here rather than spend the GPU time.
    try:
        job = job_block(os.environ, sys.argv, socket.gethostname())
        previous_jobs = previous_attempts(os.path.join(out_dir, "run_meta.json"), job)
    except (JobIdentityUnavailable, PredecessorUnreadable, ValueError) as exc:
        sys.exit(f"FAIL: {exc}")
    # The code this run executed, named by tree ids, beside git_sha: a sha names the repository,
    # these name the content of the three code directories and the paths that depart from it.
    # Inside a cluster job the job's own start-of-job check is recorded and HEAD is only confirmed
    # unmoved by reading files; elsewhere git is asked, with a bounded retry. How it was obtained
    # is a per-piece fact and goes in the job block, not in the arm-level record.
    try:
        code_tree, code_tree_source = recorded_code_tree(
            CODE_TREE_ROOT, os.environ, bool(job["on_katana"])
        )
    except CodeTreeError as exc:
        sys.exit(f"FAIL: {exc}")
    job = {**job, "code_tree_source": code_tree_source}

    n_matchups = len(games_list) * len(opponents_list) * n_seeds
    print(
        f"Sweep: {len(games_list)} games × {len(opponents_list)} opponents × {n_seeds} seeds "
        f"= {n_matchups} Parquet files"
    )
    print(f"N={cfg['n_hands']} hands, base_seed={base_seed}")
    print(f"Output: {out_dir}/")
    print()

    # runner records what actually ran (FP-determinism-audit provenance). An
    # explicit "ray" config key (or --runner ray) always wins; otherwise it is
    # Derived from max_workers: "local-pool" whenever more than one worker is in play,
    # "serial" otherwise - max_workers=1 takes the byte-identical direct-loop
    # path inside run_shards_local, so "serial" is accurate for it too.
    runner_cfg = cfg.get("runner")
    if runner_cfg == "ray":
        runner = "ray"
    elif max_workers > 1:
        runner = "local-pool"
    else:
        runner = "serial"

    # DEFAULT_SEED_POLICY is a base/fallback dict (its own module docstring),
    # not "this run's policy" -- it hardcodes temperature=0.0
    # unconditionally, so using it unmodified would make a run configured at
    # any other temperature write a run_meta.json/W&B config claiming 0.0, a
    # provenance lie in the one file whose entire job is provenance. Resolve
    # Once here from cfg and reuse for both call sites below (write_run_meta
    # and init_tracking) so they cannot drift from each other.
    # per_request_seed is a description of the actual derivation formula
    # (LMVllmAgent.run_seed -> lmopt.agents.lm_vllm.derive_request_seed),
    # mirroring matchup_seed's
    # existing descriptive-string convention rather than a live numeric seed
    # (every row derives its own).
    seed_policy = {
        **DEFAULT_SEED_POLICY,
        "temperature": cfg["temperature"],
        "per_request_seed": (
            "lmopt.agents.lm_vllm.derive_request_seed(run_seed=matchup_seed, "
            "hand_idx, node_idx, call_kind, attempt)"
        ),
    }

    # A kill-and-resume launches a FRESH run_sweep.py process, which would
    # otherwise re-resolve a NEW sha here even though prior shards in the
    # same out_dir were stamped with an earlier one, so the per-row sha would
    # name a tree that never executed them. git_sha_override, when
    # run_experiment.py supplies it (a resumed arm's already-written
    # model_revision), makes this invocation reuse that value instead of
    # resolving fresh -- a standalone invocation with no override resolves
    # fresh.
    # Inside a job whose start-of-job check was recorded, the commit is the one that check exported,
    # which recorded_code_tree has just confirmed HEAD still names; asking git again there can stall
    # on the shared filesystem, and _get_git_sha answers a stall with the string "unknown".
    git_sha = cfg.get("git_sha_override") or (
        os.environ[CODE_TREE_ENV_HEAD]
        if code_tree_source == SOURCE_JOB_CHECK
        else _get_git_sha()
    )

    # agent0 is resolved before write_run_meta so a real-LM run's serving
    # block can be written in the same run_meta.json write, rather than a
    # second overwrite after the fact. This is also the "fail loudly at
    # sweep startup" point (no silent fallbacks): a chat-template
    # resolution failure below aborts before any run_meta.json or shard
    # exists.
    agent0_model = _resolve_agent0(cfg)

    # Resolve the serving configuration (and thread the chat-template
    # fingerprint into every MatchupSpec below) only when agent0 is a real
    # LM - a random-agent0 sweep needs no live endpoint at all.
    serving_block: dict[str, Any] | None = None
    chat_template_hash: str | None = None
    # The served-weights revision is LAUNCHER-SUPPLIED, never resolved here
    # -- it lives in the serving box's HF cache, which this function's
    # HTTP-only serving resolution cannot see, and a local-cache read on a
    # split-topology orchestrator box would silently stamp the wrong box's
    # snapshot. None on a random-agent0 sweep (nothing served) and on test/smoke paths, which pass it
    # explicitly; a real sweep without it aborts below, beside the
    # chat_template/served-ids resolution it rides with.
    served_weights_revision: str | None = cfg.get("served_weights_revision")
    # Same launcher-supplied channel as served_weights_revision above, but
    # no startup gate -- there is no declared config field it must agree
    # with, only the observed launch argv when one exists (parsed by
    # run_experiment.py's parse_revision_flags). None (the common case: no
    # distinct tokenizer revision, or a provider that does not observe it)
    # leaves the column null, never fabricated.
    tokenizer_revision: str | None = cfg.get("tokenizer_revision")
    if agent0_model not in SCRIPTED_AGENT0_SENTINELS:
        serving_block = _resolve_serving_block(
            cfg["vllm_base_url"],
            agent0_model,
            cfg.get("serving_manifest"),
            cfg["chat_template_kwargs"],
        )
        chat_template_hash = serving_block["chat_template_hash"]
        if not served_weights_revision:
            # Same fail-loud shape and moment as _resolve_serving_block's own
            # aborts: before any run_meta.json or shard exists, never a null
            # column across an unattended multi-hour run (no silent
            # fallbacks). `not`, not `is None`: `--served-weights-revision ""`
            # (or a YAML empty
            # string) is a supplied-but-empty value, and stamping "" on
            # every row is exactly the fabricated-provenance class this
            # gate exists to refuse.
            sys.exit(
                f"serving resolution FAILED for {agent0_model!r}: no "
                "served-weights revision was supplied. This script never "
                "resolves it itself (the value lives in the serving box's "
                "HF cache, invisible over HTTP): resolve it at launch -- "
                "run_experiment.py does this per arm via "
                "lmopt.provenance.weights_cache's "
                "resolve_prelaunch_revision -- or pass it explicitly with "
                "--served-weights-revision. Aborting before any shard runs, "
                "rather than logging a null served_weights_revision across "
                "the whole sweep."
            )
        # A resolved and logged hash that nothing compares catches nothing
        # -- a template that renders fine-but-wrong produces a valid hash
        # and raises nothing. The pin is the per-model
        # expected_chat_template_hash from configs/models.yaml, passed by
        # the launcher (run_experiment.py) as --expected-chat-template-hash;
        # this script never reads models.yaml itself (it receives all
        # per-model input via CLI/config, the served_weights_revision
        # pattern above). Same fail-loud moment as every gate in this
        # block: before any run_meta.json or shard exists. `not`, not
        # `is None`: an empty-string pin is no pin, and comparing against
        # "" would abort every healthy run instead of saying NOT PINNED.
        expected_hash = cfg.get("expected_chat_template_hash")
        if not expected_hash:
            # Pins roll out gradually: absence is recorded out loud, never
            # silent and never fatal (no silent fallbacks).
            print(
                f"chat_template_hash NOT PINNED for {agent0_model}: measured "
                f"{chat_template_hash} -- add expected_chat_template_hash to "
                "configs/models.yaml to pin it."
            )
        elif chat_template_hash != expected_hash:
            sys.exit(
                f"chat_template_hash MISMATCH for {agent0_model!r} at "
                f"{cfg['vllm_base_url']}: expected {expected_hash!r} (the "
                f"configs/models.yaml pin), the endpoint measured "
                f"{chat_template_hash!r}. The served template is not the one "
                "the pin was measured from -- fix the serving side, or "
                "update the pin if the change is deliberate (which makes new "
                "arms incomparable with landed ones). Aborting before any "
                "shard runs."
            )

    # The arm's identity at the TOP level of its own
    # provenance file, so a reader holding the file need not parse the
    # directory name or dig into resolved_config. `agent0_model` is the
    # resolved value, and for a real-LM run it is the id _resolve_serving_block
    # has already checked against what the endpoint reports serving (it exits
    # unless `model_id in served.served_ids`), so this is the served model, not
    # the config's intent. None for a random-agent0 sweep, which omits the key.
    run_model_id = None if agent0_model in SCRIPTED_AGENT0_SENTINELS else agent0_model

    # Write run-level provenance: concurrency, seed policy, resolved config, git sha.
    # The path is kept because the same file is stamped `finished_utc` at the
    # end of this function.
    # prompt_spec_hash is omitted here (None) because shards construct their
    # agents, and we have no agent object at this point.
    # Each row still carries its own prompt_spec_hash column, and
    # any run-level invariant is checked during analysis, not at sweep startup.
    meta_path = write_run_meta(
        out_dir,
        resolved_config=cfg,
        max_workers=max_workers,
        seed_policy=seed_policy,
        git_sha=git_sha,
        runner=runner,
        serving=serving_block,
        model_id=run_model_id,
        n_hands=cfg["n_hands"],
        prompt_spec_hash=None,
        job=job,
        previous_jobs=previous_jobs,
    )
    # The memory wording's digest, recorded at the top level beside
    # prompt_spec_hash for every run whatever its rung: the spec hash does not
    # cover the memory module and every M0 and hand-1 prompt is identical
    # across wordings, so without it two wordings are indistinguishable in the
    # record, and the pilot gate reads it to refuse a pilot collected under a
    # wording that is no longer live.
    update_run_meta(meta_path, **{MEMORY_RENDER_HASH_KEY: memory_render_hash()})
    update_run_meta(meta_path, code_tree=code_tree)

    # W&B tracking: default-off, offline-capable, mirror-only -
    # cfg["wandb"] absent/enabled:false (the default) returns a NullTracker,
    # so a run with tracking off never imports wandb. run_meta is passed through as the
    # wandb run's config when tracking is enabled.
    tracker = init_tracking(
        cfg,
        {
            "max_workers": max_workers,
            "runner": runner,
            "seed_policy": seed_policy,
            "resolved_config": cfg,
            "git_sha": git_sha,
        },
    )

    # Build every shard spec up front, one game at a time (still warming each
    # game's CFR/RL cache exactly once, before any shard for that game is
    # built) - actual dispatch happens in a single runner call
    # (run_shards_local or run_shards_ray) below so the pool/cluster can fan
    # out across the whole sweep, not one game at a time.
    built: list[tuple[MatchupSpec, dict]] = []

    for game_name in games_list:
        print(f"=== {game_name.upper()} ===")
        game = load_game(game_name)

        # CFR+ (plus=True) is the Stage-B working reference. This same policy is
        # the CFRRef agent, the Bayesian likelihood model, and the P3/P5/P7
        # scoring yardstick - one validated source. Cache-once-freeze:
        # solved once per (game, iters, solver, pyspiel version) key
        # and loaded thereafter. Warmed here (once per game, printed) so every
        # per-matchup run_matchup_shard call below is a guaranteed cache hit.
        depth_pkl = DEPTH_REFERENCES.get(game_name)
        # The announcement is inside the branch: above this `if`, a
        # depth-reference pickle load would be announced as "Loading/solving
        # CFR" - neither a solve nor CFR.
        if depth_pkl is not None:
            print(
                f"  Loading partial depth reference from {depth_pkl}...",
                end=" ",
                flush=True,
            )
            load_depth_reference(depth_pkl, game)
        else:
            announce_cache_state(
                game_name,
                iters=cfg["cfr_iters"],
                cache_dir=cfg["cfr_cache_dir"],
                require_hit=cfg["require_cache_hit"],
            )
            load_or_solve_cfr(
                game,
                cfg["cfr_iters"],
                plus=True,
                cache_dir=cfg["cfr_cache_dir"],
                require_hit=cfg["require_cache_hit"],
            )
        print("done")

        # Two guards on this warm, addressing different failures; they compose.
        # Order matters: the catastrophic case is checked first.
        #
        #  1. DEPTH_REFERENCES games: warming XFP here is not merely
        #     wasteful, it is the same catastrophic full-tree get_all_states
        #     enumeration guarded in precompute_references.py.
        #     Nothing can satisfy an "rl" opponent for such a game, so refuse
        #     loudly if one is requested; otherwise skip rather than crash a
        #     sweep that never needed it. Note the DEFAULT opponents list
        #     includes "rl", so this is reachable by a plain invocation.
        #  2. Non-depth games: only warm when an RL
        #     opponent will actually play. shard.py already gates its per-shard
        #     build the same way, so a non-RL sweep never needs the cache - an
        #     unused 1000-iter XFP train cost ~25 min per depth variant.
        #
        # Both guards read `opponents_list`, which is `cfg["opponents"]` (bound
        # above), so neither can drift from the loop it protects.
        if depth_pkl is not None:
            if "rl" in opponents_list:
                sys.exit(
                    f"{game_name} has no XFP RL reference and cannot have one "
                    "(full-tree get_all_states enumeration OOMs at this size). "
                    "Remove 'rl' from this sweep's opponents list."
                )
            print(
                "  XFP RL warm skipped (no full-tree RL reference possible "
                f"for {game_name})"
            )
        elif "rl" not in opponents_list:
            print("  XFP RL warm skipped (no 'rl' opponent in this sweep)")
        else:
            print(
                f"  Loading/training XFP RL agent ({cfg['rl_iterations']} iters)...",
                end=" ",
                flush=True,
            )
            # Cache-once-freeze: XFP is trained at most once per (game, iters,
            # seed, pyspiel version) key, making "RL agent is frozen after training"
            # literally true instead of retraining every run.
            # check_exploitability=True keeps the RL-vs-bar diagnostic; it
            # only runs on a cache miss (fresh train).
            # Warmed here (once per game) so every run_matchup_shard call below is a
            # guaranteed cache hit.
            RLSelfPlayAgent.load_or_train(
                game,
                num_iterations=cfg["rl_iterations"],
                seed=base_seed,
                cache_dir=cfg["rl_cache_dir"],
                check_exploitability=True,
                require_hit=cfg["require_cache_hit"],
            )
            print("done")

        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx

            for opp_name in opponents_list:
                # One helper, so the probe arm's variant component cannot be
                # added here and forgotten elsewhere. Base arms keep the plain
                # convention.
                out_path = _shard_out_path(
                    out_dir,
                    game_name,
                    opp_name,
                    seed_idx,
                    n_seeds,
                    prompt_variant,
                    probe_type,
                )

                spec = MatchupSpec(
                    game_name=game_name,
                    opponent=opp_name,
                    seed=seed,
                    n_hands=cfg["n_hands"],
                    out_path=out_path,
                    model=agent0_model,
                    vllm_base_url=cfg.get("vllm_base_url") or "",
                    elicitation_mode=cfg["elicitation_mode"],
                    max_retries=cfg["max_retries"],
                    cfr_iters=cfg["cfr_iters"],
                    rl_iterations=cfg["rl_iterations"],
                    cfr_cache_dir=cfg["cfr_cache_dir"],
                    rl_cache_dir=cfg["rl_cache_dir"],
                    backend=cfg["backend"],
                    temperature=cfg["temperature"],
                    # Per-request HTTP ceiling, from config.
                    request_timeout_s=cfg["request_timeout_s"],
                    # The belief decoding protocol, from config. Every spec
                    # in a sweep carries the same value; the Parquet column exists so that a future
                    # per-matchup override could not make the rows
                    # unreadable (schema.py's own column comment).
                    belief_protocol=cfg["belief_protocol"],
                    # Per-run decode budgets, from config. Without these two
                    # lines the knob is configurable, recorded in
                    # resolved_config, and never applied.
                    action_max_tokens=cfg["action_max_tokens"],
                    belief_max_tokens=cfg["belief_max_tokens"],
                    # chat_template_kwargs from config. Without this line the
                    # knob is configurable, recorded in resolved_config, and
                    # never applied.
                    chat_template_kwargs=cfg["chat_template_kwargs"],
                    # Without this line the knob is configurable, recorded in
                    # resolved_config, and never applied.
                    action_top_logprobs=cfg["action_top_logprobs"],
                    # Memory knobs from config. Without these two lines the
                    # knobs are configurable, recorded in resolved_config, and
                    # never applied.
                    memory_rung=cfg["memory_rung"],
                    memory_window_k=cfg["memory_window_k"],
                    # Which seat agent0 occupies, from config.
                    seat_mode=cfg["seat_mode"],
                    # The probe labels, resolved once at
                    # startup (above) from the `probe:` block, never re-derived
                    # per spec. Both are passed together because MatchupSpec
                    # refuses a pair that contradicts itself, and a probe arm
                    # resolves its variant spec from exactly these two values.
                    probe_type=probe_type,
                    prompt_variant=prompt_variant,
                    # The forced opening, straight from
                    # config. MatchupSpec refuses a spec whose drive and probe
                    # label contradict each other, so this argument cannot be
                    # dropped silently.
                    drive_openings=drive_openings,
                    # Resolved once above (not per
                    # shard) - None for a random-agent0 sweep (no live
                    # endpoint), the one resolved fingerprint for every
                    # shard of a real-LM sweep (chat_template_hash is a
                    # per-run property of the served template, not a
                    # per-request one - LMVllmAgent's own docstring).
                    chat_template_hash=chat_template_hash,
                    # Launcher-resolved once per arm, gated at startup above
                    # -- None only on a random-agent0 sweep, where no weights
                    # are served.
                    served_weights_revision=served_weights_revision,
                    # Same launcher-resolved-once-per-arm shape as
                    # served_weights_revision above -- None whenever the
                    # launch never observed a distinct --tokenizer-revision.
                    tokenizer_revision=tokenizer_revision,
                    # Unconditionally True, not conditioned on
                    # `runner`. Parallel workers must never race each other into
                    # solving/training the same cache miss concurrently; and a
                    # serial shard's own miss is not impossible either, even
                    # though caches are warmed once per game above -- a cache
                    # can go cold between that warm and this shard's own load, from a concurrent job on
                    # a shared checkout overwriting the blob under different
                    # code, or from a direct shard invocation that skips the
                    # warm entirely. A shard-level check is a dict comparison
                    # and a file stat, not a solve, so fail-loud costs nothing
                    # to keep on for every runner.
                    require_cache_hit=True,
                    # The one sha resolved above for the whole sweep (already
                    # used for run_meta.json's git_sha), threaded into every
                    # shard so `model_revision` cannot split across two values
                    # Within this one sweep invocation if a HEAD move happens
                    # between two shards of it -- see MatchupSpec.git_sha's
                    # docstring. Across a kill-and-resume (run_experiment.py
                    # --allow-existing), consistency depends on
                    # git_sha_override above.
                    git_sha=git_sha,
                )
                built.append(
                    (
                        spec,
                        {
                            "game_name": game_name,
                            "opp_name": opp_name,
                            "seed": seed,
                            "seed_idx": seed_idx,
                        },
                    )
                )

        print()

    specs = [spec for spec, _ in built]
    # Same throttle as match_runner.py's hand bar, and for
    # the same reason -- a redirected (`nohup ... > log 2>&1`) sweep has no
    # terminal to overwrite, so tqdm's default 0.1s mininterval turns every
    # matchup update into its own `\r`-separated redraw, burying the
    # per-shard timing line under noise. Non-TTY only; a live terminal keeps
    # the default. This bar is `leave=True` (the default), so close() does
    # unconditionally flush the final completed-bar line regardless of
    # mininterval (tqdm 4.67.3) -- unlike match_runner.py's
    # `leave=False` hand bar, this is the per-shard timing line the
    # scheduling analysis reads, and it survives. 10s: a shard runs minutes
    # to hours, so this caps a multi-hour sweep's log to a handful of
    # intermediate lines while a stall mid-sweep still surfaces within one
    # interval.
    _matchup_bar_kwargs = {} if sys.stderr.isatty() else {"mininterval": 10.0}
    matchup_bar = _tqdm(
        total=len(specs),
        desc="matchups",
        unit="matchup",
        **_matchup_bar_kwargs,
    )

    def _on_result(_spec: MatchupSpec, _path: str) -> None:
        matchup_bar.update(1)
        # Mirror-only: a completed shard is logged to W&B
        # After its Parquet file is already durably written by
        # run_matchup_shard - this never influences what gets logged. Runner-
        # agnostic: on_result is the same callback for serial, local-pool, and
        # ray dispatch (see run_shards_local / run_shards_ray calls below).
        tracker.log_shard(dataclasses.asdict(_spec), _path)

    # A single shard failure (e.g. one `Request timed out.`) must not discard
    # hours of remaining work. A shard failure is a recorded finding and the
    # sweep continues; it is never silent and never a pass (no silent
    # fallbacks) -- see the FAILED shards block, the
    # shard_outcomes.json artefact and SHARD_FAILURE_EXIT_CODE below.
    failed_by_index: dict[int, dict] = {}
    # Keyed on id(), not on the spec itself, for the reason local.py's own
    # future_to_index map records: MatchupSpecs are plain frozen dataclasses
    # and two shards can legitimately be field-equal, so they would collide as
    # dict keys. `specs` holds a reference to every one of them for the whole
    # dispatch, so the identities cannot be recycled underneath this map.
    index_of_spec = {id(spec): i for i, spec in enumerate(specs)}

    def _on_error(_spec: MatchupSpec, _exc: BaseException) -> None:
        matchup_bar.update(1)
        failed_by_index[index_of_spec[id(_spec)]] = {
            "error_type": type(_exc).__name__,
            "error": str(_exc) or repr(_exc),
        }
        # Printed the moment it happens, not only in the final summary: a
        # sweep that runs to completion could otherwise hide a failure
        # for hours behind a progress bar that looks healthy.
        print(
            f"\nSHARD FAILED (run continues): {_spec.game_name} vs "
            f"{_spec.opponent} seed={_spec.seed} -> "
            f"{type(_exc).__name__}: {_exc}",
            flush=True,
        )

    paths: Sequence[str | None]
    try:
        if runner == "ray":
            # Import ray via ray_runner (whose module top disables Ray's
            # uv-run worker-relaunch hook before ray is first imported -
            # see lmopt/orchestration/ray_runner.py docstring).
            from lmopt.orchestration.ray_runner import run_shards_ray
            import ray

            # Caller-managed Ray lifecycle: attach to an existing cluster
            # (e.g. `ray start --head` on this box, or a remote
            # ray_address) or start a throwaway local one. Shut down only
            # what this process itself started - never an operator's
            # long-lived cluster.
            owns_ray = not ray.is_initialized()
            if owns_ray:
                ray.init(address=cfg.get("ray_address"))
            try:
                paths = run_shards_ray(
                    specs,
                    address=cfg.get("ray_address"),
                    resources_per_shard=cfg.get("ray_resources_per_shard"),
                    on_result=_on_result,
                )
            finally:
                if owns_ray:
                    ray.shutdown()
        else:
            paths = run_shards_local(
                specs,
                max_workers=max_workers,
                on_result=_on_result,
                on_error=_on_error,
            )
    except RuntimeError as exc:
        matchup_bar.close()
        tracker.finish()
        print(f"\nFAIL: {exc}")
        print(
            "Hint: every shard requires its CFR/RL cache to already be warm "
            "(require_cache_hit=True) - warm it first with "
            "'uv run python scripts/precompute_references.py'."
        )
        # Reachable for the local runner only when the failure is the runner's
        # own, not a shard's (shard failures go to _on_error). It remains
        # the ordinary path for `runner: ray`: run_shards_ray still aborts the
        # whole run on one shard, so continue-on-error is a local-pool/serial
        # property. Said out loud rather than left for
        # a reader to infer from a silence.
        if runner == "ray":
            print(
                "NOTE: the 'ray' runner has no per-shard continue-on-error "
                "(it is wired into run_shards_local only); one "
                "shard failure still discards the rest of this sweep."
            )
        return 1

    matchup_bar.close()

    # Summary table accumulation.
    summary: list[dict] = []
    total_violations = 0

    for shard_index, ((spec, meta), shard_out_path) in enumerate(
        zip(built, paths, strict=True)
    ):
        # A shard reported through _on_error has `None` in its slot (never a
        # shifted neighbour - see run_shards_local's docstring). It is
        # summarised as FAILED and its shard_out_path is deliberately not read: a
        # Parquet may well be sitting there from an earlier run
        # (run_experiment.py's --allow-existing flow makes that routine), and
        # reporting its rows would describe a matchup this run never played,
        # which is exactly the incomplete-read-as-complete confusion this
        # bookkeeping exists to prevent.
        if shard_out_path is None:
            fail = failed_by_index[shard_index]
            seed_tag = f" seed={meta['seed_idx']}" if n_seeds > 1 else ""
            print(
                f"  {meta['game_name']} vs {meta['opp_name']:12s}{seed_tag}: "
                f"FAILED ({fail['error_type']}: {fail['error']})"
            )
            summary.append(
                {
                    "game": meta["game_name"],
                    "opponent": meta["opp_name"],
                    "seed": meta["seed"],
                    "status": "failed",
                    "error_type": fail["error_type"],
                    "error": fail["error"],
                    "out": spec.out_path,
                    **_shard_timing(spec.out_path),
                }
            )
            continue

        # Per-matchup stats, recomputed from the shard's own Parquet output -
        # run_matchup_shard returns only the path (the point of the
        # single-distribution-unit refactor: a parallel worker need not ship
        # the full in-memory record list back to the caller).
        # The shard's fallback sidecar, informational here - the run-level
        # tally itself is pooled from every Parquet in `out_dir` after this
        # loop (`_aggregate_arm_sidecars`), not by merging within it, so a
        # resumed sweep's earlier shards (already on disk, not in this
        # invocation's `paths`) are counted too. A missing sidecar here is not
        # a silent zero: it means this shard was written by an older build
        # (an --allow-existing rerun over an older arm's Parquet), so the
        # absence is printed for this shard even
        # though the arm-level aggregation below will likewise skip it.
        sidecar_path = posterior_fallback_sidecar_path(shard_out_path)
        if not os.path.exists(sidecar_path):
            print(
                f"  NOTE: no posterior-fallback sidecar beside {shard_out_path} - "
                f"this shard contributes no fallback counts to run_meta.json"
            )
        # The driven-discard sidecar exists exactly for driven shards, so its
        # absence is only notable when this shard was driven (an older
        # build's Parquet under --allow-existing).
        if spec.drive_openings:
            discard_sidecar = driven_discard_sidecar_path(shard_out_path)
            if not os.path.exists(discard_sidecar):
                print(
                    f"  NOTE: no driven-discard sidecar beside {shard_out_path} - "
                    f"this driven shard contributes no discard counts to "
                    f"run_meta.json"
                )

        df = read_parquet(shard_out_path)
        n_nodes = df.height
        # Hardcoding player_to_act==0 as "the tested seat" would print +nan
        # under seat_mode: fixed1 when no seat-0 node exists, and a
        # seat-0-anchored (not the tested seat's) number when one does.
        # `lm_seat` names AGENT0's seat, not generically "the agent
        # under test" (schema.py's own comment: every scorer keys on
        # player_to_act instead, deliberately, since agent0 is the
        # opponent whenever a principled agent is scored) -- but this
        # script's agent0 is always the LM or RANDOM_AGENT0_SENTINEL
        # (_resolve_agent0), never a principled agent, so "agent0's seat"
        # and "the LM's seat" coincide here specifically. Do not copy this
        # `player_to_act == lm_seat` pattern into a scorer or any script
        # where agent0 could be a non-LM agent under test.
        lm_df = df.filter(pl.col("player_to_act") == pl.col("lm_seat"))
        if lm_df.height > 0:
            p0_hands = lm_df.unique(subset=["hand_idx"], keep="first")
            mean_r = float(
                cast(float, p0_hands["reward"].mean())
            )  # polars types .mean() as a wide union
        else:
            mean_r = float("nan")
        violations = sum(int(df[col].null_count()) for col in REQUIRED_COLUMNS)
        total_violations += violations
        n_beliefs = int(df.filter(pl.col("elicited_belief").is_not_null()).height)

        seed_tag = f" seed={meta['seed_idx']}" if n_seeds > 1 else ""
        print(
            f"  {meta['game_name']} vs {meta['opp_name']:12s}{seed_tag}: "
            f"nodes={n_nodes:5d}  "
            f"mean_r={mean_r:+.4f}  "
            f"beliefs={n_beliefs:4d}  "
            f"violations={violations}"
        )

        summary.append(
            {
                "game": meta["game_name"],
                "opponent": meta["opp_name"],
                "seed": meta["seed"],
                "status": "completed",
                "nodes": n_nodes,
                "mean_reward": mean_r,
                "beliefs": n_beliefs,
                "violations": violations,
                "out": shard_out_path,
                **_shard_timing(shard_out_path),
            }
        )

    # Final summary.
    print("=" * 72)
    print(
        f"{'GAME':<12} {'OPPONENT':<12} {'SEED':>5} {'NODES':>7} {'MEAN_R':>8} "
        f"{'BELIEFS':>8} {'VIOL':>5}"
    )
    print("-" * 72)
    for row in summary:
        if row["status"] == "failed":
            print(
                f"{row['game']:<12} {row['opponent']:<12} {row['seed']:>5} "
                f"{'FAILED':>7} {row['error_type']}: {row['error']}"
            )
            continue
        print(
            f"{row['game']:<12} {row['opponent']:<12} {row['seed']:>5} {row['nodes']:>7} "
            f"{row['mean_reward']:>+8.4f} {row['beliefs']:>8} {row['violations']:>5}"
        )
    print("=" * 72)
    print(f"Total violations: {total_violations}")
    print()

    # The completeness record. Written on every run, clean or not (see
    # SHARD_OUTCOMES_FILENAME): one entry per requested shard, so absence from
    # this list means "never requested" and status "failed" means "requested,
    # attempted, lost". Written before the return so a non-zero exit still
    # leaves the evidence behind.
    outcomes_path = os.path.join(out_dir, SHARD_OUTCOMES_FILENAME)
    write_json_atomic(
        outcomes_path,
        {
            "n_requested": len(summary),
            "n_completed": sum(1 for r in summary if r["status"] == "completed"),
            "n_failed": sum(1 for r in summary if r["status"] == "failed"),
            "shards": [
                {
                    "game": r["game"],
                    "opponent": r["opponent"],
                    "seed": r["seed"],
                    "status": r["status"],
                    "out_path": r["out"],
                    "error_type": r.get("error_type"),
                    "error": r.get("error"),
                    **{key: r.get(key) for key in ENTRY_TIMING_KEYS},
                }
                for r in summary
            ],
        },
    )

    # The posterior-fallback tally lands on run_meta.json here, after the shard
    # loop -- it cannot land in the startup write,
    # which happens before a single hand is played. Written on every run that
    # completed at least one shard, clean or not, and for the same reason
    # shard_outcomes.json is: the evidence must survive a non-zero exit.
    #
    # Pooled from every shard sidecar in `out_dir` on disk, not from this invocation's own `built`/`paths` - see
    # `_aggregate_arm_sidecars`'s docstring. This is what makes a
    # kill-and-resume sweep's run_meta.json total the whole arm: a resumed
    # `main()` only dispatches the remaining shards, but the earlier
    # invocation's shards (and sidecars) are already durably in this same
    # directory.
    #
    # The key is OMITTED when the tally is empty, following write_run_meta's own
    # "not set" vs "not observed" rule: an empty dict means no shard computed a
    # posterior at all (or none completed), which is not the same claim as "the
    # oracle path ran and fell back zero times" -- that claim is a present key
    # with fell_back: 0.
    posterior_fallback_counts, driven_discard_counts = _aggregate_arm_sidecars(out_dir)
    if posterior_fallback_counts:
        log_posterior_fallback_warning(posterior_fallback_counts)
        update_run_meta(meta_path, posterior_fallback_counts=posterior_fallback_counts)

    # The driven-hand discard tally lands the same way,
    # key omitted when no driven shard contributed ("not a driven arm"),
    # present -- zeros included -- when one did.
    if driven_discard_counts:
        log_driven_discard_warning(driven_discard_counts)
        update_run_meta(meta_path, driven_discard_counts=driven_discard_counts)

    tracker.finish()

    n_failed = sum(1 for r in summary if r["status"] == "failed")
    if total_violations > 0:
        print("FAIL: schema violations detected.")
    if n_failed:
        # Precedence over the violation exit below: see
        # SHARD_FAILURE_EXIT_CODE. Both conditions are printed either way.
        print(
            f"FAIL: {n_failed} of {len(summary)} shards FAILED - this arm is "
            f"INCOMPLETE and must not be read as a finished one. Per-shard "
            f"detail: {outcomes_path}"
        )
        if any("cache miss" in (r.get("error") or "") for r in summary):
            print(
                "Hint: every shard requires its CFR/RL cache to already be "
                "warm (require_cache_hit=True) - warm it first with "
                "'uv run python scripts/precompute_references.py'."
            )
        return SHARD_FAILURE_EXIT_CODE
    if total_violations > 0:
        return 1
    # The sweep reached its clean conclusion, so stamp the
    # finish time onto the provenance file written at startup. Deliberately
    # Below both failure returns above: `finished_utc` present means every
    # requested shard completed with zero schema violations, and its absence
    # means the arm was killed, aborted, or concluded with failures --
    # shard_outcomes.json (written on every run, above) says which. Stamping
    # it on a failure path would tell a reader an INCOMPLETE arm is finished,
    # which is the same confusion SHARD_FAILURE_EXIT_CODE exists to prevent.

    # Extract prompt_spec_hash from shards and record
    # if unique. This happens before finalize_run_meta so the hash is persisted
    # in the final file.
    _extract_and_update_prompt_spec_hash_from_shards(meta_path, summary)

    finalize_run_meta(meta_path)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
