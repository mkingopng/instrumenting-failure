"""
The class of every value in a collection run's `run_meta.json`, as compose reads it.

A composite's `run_meta.json` is built from the pieces of one arm. Every value a piece records is
exactly one of:

* an ARM fact: a disagreement means the pieces are not the same arm and must not be composed;
* a piece fact: otherwise. It is dropped from the composite's top level and carried, in full, in
  that piece's `pieces[]` entry. A piece fact is identity when it differs between pieces by
  construction (a seed, an output directory, an endpoint), and environment when two pieces of one
  arm may legitimately agree or differ (a driver, a worker count). The spread of the environment
  facts is reported at the top of the composite;
* a COMPUTED fact: compose derives it, and no piece's copy is used.

There is no default class. A value in none of the sets is reported by name and compose refuses
the piece, so adding a key to `run_meta.json` forces someone to classify it.

`MUST_MATCH_FIELDS` in `scripts/compose_arm_pieces.py` is a different relation (the pairing of a
base arm with its probe arm, read by `probe_scores`), and a subset of the ARM facts here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

ARM = "arm"
PIECE_IDENTITY = "piece_identity"
PIECE_ENV = "piece_env"
COMPUTED = "computed"
#: An arm fact compose reads per game (a game-disjoint composite's pieces name one game each), by
#: its own rule in `scripts/compose_arm_pieces.py`.
PER_GAME = "per_game"
#: An arm fact whose agreement is required only above M0: the memory wording does not render at
#: M0, so a piece from before the digest existed composes with one from after. Compose checks it
#: by its own rule, and the composite carries it only when every piece records the same value.
ARM_GATED = "arm_gated"

CLASSES = (ARM, ARM_GATED, PIECE_IDENTITY, PIECE_ENV, COMPUTED, PER_GAME)

#: Every declared path and its class. A path names a leaf or a whole value (a dict compared or
#: carried as one value); a path that is only the parent of declared paths is a container that
#: compose walks into. A dotted path never names a key that itself contains a dot.
DECLARED: dict[str, str] = {
    # ---- top level
    "environment.pyspiel_version": ARM,
    "environment.numpy_version": ARM,
    "model_id": ARM,
    "n_hands": ARM,
    "prompt_spec_hash": ARM,
    "memory_render_hash": ARM_GATED,
    "seed_policy.temperature": ARM,
    "seed_policy.per_request_seed": ARM,
    "seed_policy.matchup_seed": ARM,
    "git_sha": ARM,
    "code_tree": ARM,
    "created_utc": PIECE_IDENTITY,
    "finished_utc": PIECE_IDENTITY,
    "job": PIECE_IDENTITY,
    "previous_jobs": PIECE_IDENTITY,
    "max_workers": PIECE_ENV,
    "runner": PIECE_ENV,
    "posterior_fallback_counts": COMPUTED,
    "driven_discard_counts": COMPUTED,
    "provenance": COMPUTED,
    "provenance_basis": COMPUTED,
    "pieces": COMPUTED,
    "composed_from": COMPUTED,
    "shape": COMPUTED,
    "n_pieces": COMPUTED,
    "n_expected": COMPUTED,
    "count_source": COMPUTED,
    "piece_fact_spread": COMPUTED,
    "git_shas": COMPUTED,
    "code_trees": COMPUTED,
    "mixed_git_sha": COMPUTED,
    # ---- serving
    "serving.model_id": ARM,
    "serving.image_ref": ARM,
    "serving.container": ARM,
    "serving.max_model_len": ARM,
    "serving.chat_template_hash": ARM,
    "serving.unit": ARM,
    "serving.box_fields_source": ARM,
    "serving.serving_env.lockfile_sha256": ARM,
    "serving.serving_env.modules": ARM,
    "serving.serving_env.python": ARM,
    "serving.serving_env.torch": ARM,
    "serving.serving_env.vllm": ARM,
    "serving.base_url": PIECE_IDENTITY,
    # Embeds the job's own `--port`, so it differs between jobs by construction. Null means "not
    # observed" (the argv scan could not name one server), with the reason beside it; an empty
    # list would say the server was launched with no arguments, which is never true.
    "serving.vllm_args": PIECE_IDENTITY,
    # Present only when `vllm_args` is null. Two pieces of one arm may differ in whether their
    # launch was observed, and the spread reports it.
    "serving.vllm_args_unobserved_reason": PIECE_ENV,
    "serving.gpu_name": PIECE_ENV,
    "serving.gpu_memory_total_mib": PIECE_ENV,
    # A hash takes the class of its most node-specific input: it covers the whole serving_env,
    # driver and CUDA runtime included.
    "serving.serving_env_hash": PIECE_ENV,
    "serving.serving_env.driver": PIECE_ENV,
    "serving.serving_env.cuda_runtime": PIECE_ENV,
    # ---- resolved_config
    "resolved_config.agent0": ARM,
    "resolved_config.action_max_tokens": ARM,
    "resolved_config.action_top_logprobs": ARM,
    "resolved_config.backend": ARM,
    "resolved_config.belief_max_tokens": ARM,
    "resolved_config.belief_protocol": ARM,
    "resolved_config.cfr_iters": ARM,
    "resolved_config.diagnostic": ARM,
    "resolved_config.chat_template_kwargs": ARM,
    "resolved_config.elicitation_mode": ARM,
    "resolved_config.exact_policy": ARM,
    "resolved_config.expected_chat_template_hash": ARM,
    "resolved_config.max_retries": ARM,
    "resolved_config.memory_rung": ARM,
    "resolved_config.memory_window_k": ARM,
    "resolved_config.model": ARM,
    "resolved_config.n_hands": ARM,
    "resolved_config.n_seeds": ARM,
    "resolved_config.opponents": ARM,
    "resolved_config.probe": ARM,
    "resolved_config.request_timeout_s": ARM,
    "resolved_config.require_cache_hit": ARM,
    "resolved_config.rl_iterations": ARM,
    "resolved_config.serving": ARM,
    "resolved_config.served_weights_revision": ARM,
    "resolved_config.temperature": ARM,
    "resolved_config.drive_openings": PER_GAME,
    "resolved_config.seat_mode": PER_GAME,
    "resolved_config.games": COMPUTED,
    "resolved_config.models": PIECE_IDENTITY,
    "resolved_config.out_dir": PIECE_IDENTITY,
    "resolved_config.out_root": PIECE_IDENTITY,
    "resolved_config.seed": PIECE_IDENTITY,
    "resolved_config.serving_manifest": PIECE_IDENTITY,
    "resolved_config.vllm_base_url": PIECE_IDENTITY,
    "resolved_config.max_workers": PIECE_ENV,
    "resolved_config.runner": PIECE_ENV,
    "resolved_config.ray_address": PIECE_ENV,
    "resolved_config.ray_resources_per_shard": PIECE_ENV,
    "resolved_config.cfr_cache_dir": PIECE_ENV,
    "resolved_config.rl_cache_dir": PIECE_ENV,
    "resolved_config.wandb": PIECE_ENV,
}

#: The paths that are containers (a declared path's proper ancestor that is not itself declared).
CONTAINERS: frozenset[str] = frozenset(
    ".".join(path.split(".")[:n])
    for path in DECLARED
    for n in range(1, len(path.split(".")))
) - frozenset(DECLARED)


#: The one field for which an absent key and a recorded None are the same value when a base arm is
#: paired with its probe arm: the key did not exist when every landed arm was collected, and its
#: absence has exactly the meaning of its None (the request carried no logprobs field). Read only by
#: `scripts/probe_scores.py` for that pairing. Within one arm it does not apply: pieces of one arm
#: are collected on one checkout, so a piece that omits the key beside one that records null is a
#: disagreement (`agreement_value`). Adding a name here is a decision about that field, never a
#: general fallback.
ABSENT_MEANS_NONE_FIELDS: frozenset[str] = frozenset(
    {"resolved_config.action_top_logprobs"}
)


class _Missing:
    """
    the value of a path a piece does not record, distinct from a recorded null
    """

    _instance: "_Missing | None" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<absent>"


#: Compared by identity, so an omitted key never equals a recorded `None`.
MISSING = _Missing()


def get_path(meta: Mapping[str, Any], path: str) -> Any:
    """
    :param meta: a parsed `run_meta.json` (mapping)
    :param path: a dotted path (str)
    :returns: the value at the path, or MISSING when any step is absent or not a mapping; a
        recorded null is returned as None, never as MISSING
    """
    node: Any = meta
    for step in path.split("."):
        if not isinstance(node, Mapping) or step not in node:
            return MISSING
        node = node[step]
    return node


def walk(meta: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    every declared value a run_meta records, and every path it records that no set names
    :param meta: a parsed `run_meta.json` (mapping)
    :returns: ({declared path: the value recorded there}, [paths in no set, sorted]). A container
        whose value is not an object, or a dict at a path that is neither declared nor a
        container, is reported as unclassified at that path.
    """
    found: dict[str, Any] = {}
    unknown: list[str] = []

    def visit(node: Mapping[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            if path in DECLARED:
                found[path] = value
            elif path in CONTAINERS:
                if isinstance(value, Mapping):
                    visit(value, path + ".")
                else:
                    unknown.append(
                        f"{path} (expected an object, found {type(value).__name__})"
                    )
            else:
                unknown.append(path)

    visit(meta, "")
    return found, sorted(unknown)


def paths_of(*classes: str) -> tuple[str, ...]:
    """
    :param classes: one or more of CLASSES (str)
    :returns: the declared paths in those classes, in declaration order (tuple of str)
    """
    return tuple(path for path, cls in DECLARED.items() if cls in classes)


def canonical(value: Any) -> str:
    """
    :param value: any JSON value
    :returns: a canonical text of it, so two values compare as equal exactly when their JSON is
        (str); MISSING is its own text
    """
    if value is MISSING:
        return "<absent>"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def distinct_values(metas: Sequence[Mapping[str, Any]], path: str) -> list[str]:
    """
    :param metas: parsed `run_meta.json` files, one per piece (sequence of mapping)
    :param path: a declared path (str)
    :returns: the distinct canonical values of the path across the pieces, an omitted key counted
        as its own value (list of str, sorted)
    """
    return sorted({canonical(get_path(meta, path)) for meta in metas})


def agreement_value(meta: Mapping[str, Any], path: str) -> str:
    """
    the value two pieces of one arm are compared by, at one path
    :param meta: a parsed `run_meta.json` (mapping)
    :param path: a declared path (str)
    :returns: the canonical text of the recorded value; an omitted key is its own value, distinct
        from a recorded null for every path (str)
    """
    return canonical(get_path(meta, path))
