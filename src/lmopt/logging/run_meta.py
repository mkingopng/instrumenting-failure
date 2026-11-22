"""Run-level provenance metadata - concurrency, seed policy, resolved config."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

# The sweep's DEFAULT/fallback seed policy (FP-determinism
# provenance): deterministic zero-temperature decoding, no per-request seed
# override, and matchup seeds derived as base_seed + seed_idx. This is a base
# dict for a run whose config resolves to exactly these defaults -- it is not
# itself "the policy of every run": a caller whose config
# resolves a different temperature must build a resolved seed_policy (e.g.
# {**DEFAULT_SEED_POLICY, "temperature": cfg["temperature"], ...}) rather than
# passing this constant verbatim, or run_meta.json states the wrong
# temperature for every run that isn't at the default.
# Single source of truth for the default values.
DEFAULT_SEED_POLICY: dict[str, Any] = {
    "temperature": 0.0,
    "per_request_seed": None,
    "matchup_seed": "base_seed + seed_idx",
}


def write_run_meta(
    out_dir: str,
    *,
    resolved_config: dict[str, Any],
    max_workers: int,
    seed_policy: dict[str, Any],
    git_sha: str,
    runner: str,
    serving: dict[str, Any] | None = None,
    model_id: str | None = None,
    n_hands: int | None = None,
    posterior_fallback_counts: dict[str, dict[str, int]] | None = None,
    prompt_spec_hash: str | None = None,
    job: dict[str, Any] | None = None,
    previous_jobs: list[dict[str, Any]] | None = None,
) -> str:
    """
    Write <out_dir>/run_meta.json; returns its path. Overwrites atomically.

    The `environment` block holds the collecting box's library versions, written
    unconditionally under the top-level `environment` key from
    `lmopt.provenance.versions.library_versions()`. Unlike the optional keys
    the parameters below describe, it is not a "not set" candidate: a collection run always has
    an OpenSpiel and a numpy, so an absent block means the run predates
    this record, never that the run had no versions. It is deliberately
    not part of `resolved_config`, which holds what the operator asked
    for; these are facts about the machine that answered.

    What reads it: a reference measurement taken under one OpenSpiel
    cannot be applied to a run collected under another, because the
    solved equilibrium the measurement is relative to may differ. The
    P3 sentinel anchors are the first such consumer
    (`lmopt.analysis.sentinel_anchors.VOIDING_PROVENANCE_FIELDS`), and
    they treat an ABSENT version as a disagreement rather than a match,
    so an arm collected before this key existed withholds its P3
    verdict instead of borrowing one. The spelling is shared with the
    reference cache's own key, so the two can never disagree about how
    a version is written.
    :param out_dir: Output directory (will be created if needed).
    :param resolved_config: Fully resolved configuration dict with all defaults applied.
    :param max_workers: Number of parallel workers (concurrency).
    :param seed_policy: Seed policy dict (e.g., {"temperature": 0.0, "per_request_seed": None,
        "matchup_seed": "base_seed + seed_idx"}).
    :param git_sha: Git commit SHA for provenance.
    :param runner: Runner backend name (e.g., "serial", "local-pool", "ray").
    :param serving: The resolved serving configuration as actually applied,
        typically ``ResolvedServing.to_run_meta()``
        (lmopt.serving.control) -- it must name
        settings as actually applied, not the config file's intent.
        Keyword-only with default None so no existing caller breaks
        (write_run_meta itself carries zero vLLM-specific knowledge; a
        caller with no serving info to report, e.g. a random-agent0
        sweep, simply omits the argument).

        "not set" vs "not observed": this top-level `serving` KEY is OMITTED
        from the written JSON entirely when `serving is None` -- "no
        resolved serving info applies to this run" (not set; e.g. a
        random-agent0 sweep needs no live endpoint at all, or the caller
        predates this parameter). When `serving` is provided, its
        individual fields (image_ref/vllm_args/container/unit) are
        written verbatim, including any `None` a field carries --
        those mean "attempted to observe on the box, could not verify"
        (ResolvedServing's own contract; `vllm_args` is null with a
        `vllm_args_unobserved_reason` beside it, never `[]`), a different
        state from the key being absent altogether. Never collapse the two into one `null`.
    :param model_id: The model this arm actually served, at the TOP level.
        Identity carried only in a directory name and inside
        `resolved_config` is lost silently when the directory is renamed
        or copied. Follows the same
        "not set" rule as `serving`: a run with no served model at all
        (a random-agent0 sweep) omits the key rather than writing
        `null`, so "no model was served" stays distinguishable from "a
        model was served and we failed to record which".
    :param n_hands: The arm's hand count, also at the top level and for the same
        reason. Omitted when the caller has none to report; a caller that
        knows its hand count always passes it.
    :param posterior_fallback_counts: Per-game posterior fallback tallies from the
        sweep, in the shape ``{game: {"fell_back": int, "rows": int}}``.
        Follows the same "not set" rule as
        `serving` and `model_id`: the key is OMITTED from run_meta.json when
        `None` (no posterior computation happened, e.g. a random-agent0 sweep),
        present when provided (even if every game's fell_back is 0). This
        distinguishes "no oracle path taken" from "oracle path taken, zero
        fallbacks measured".

        The sweep does not use this parameter, and cannot: the counts exist only once the matches have run,
        and this function is called before the first shard is dispatched.
        `scripts/run_sweep.py` lands the key through `update_run_meta` after
        the shard loop. The parameter is kept for a caller that already holds
        a finished tally (a re-write of a completed arm's provenance), so the key's shape and its
        "not set" rule have exactly one definition.
    :param prompt_spec_hash: SHA256 hash of the prompt content this arm renders,
        at the TOP level and following the same "not set" rule as
        `serving`/`model_id`. For an LM agent, the hash is computed at
        agent construction time and reflects the exact prompt spec (variant,
        game, context) the agent renders. Omitted when no LM agent is in
        play (e.g. a random-agent0 sweep), and distinguished from an LM
        run where the hash could not be recorded: the key is absent, not
        null, when the caller passes None. An LM caller reads the hash off
        the agent via getattr(agent, "prompt_spec_hash", None) and passes
        it here.
    :param job: the `job` block of this attempt (`lmopt.logging.job.job_block`), written at the
        top level when given. A collection run always passes one; a caller that predates it
        omits it and the key is omitted, never written as null.
    :param previous_jobs: the earlier attempts in this directory
        (`lmopt.logging.job.previous_attempts`), oldest first, written at the top level when
        given, including as an empty list, which says the directory had none.
    :returns: Path to the written run_meta.json file.
    """
    from lmopt.provenance.versions import library_versions

    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "max_workers": max_workers,
        "runner": runner,
        "seed_policy": seed_policy,
        "resolved_config": resolved_config,
        "git_sha": git_sha,
        "environment": library_versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Identity keys, each under the same "not set" rule as `serving` below:
    # present when the run resolved one, absent when it did not.
    if model_id is not None:
        meta["model_id"] = model_id
    if n_hands is not None:
        meta["n_hands"] = n_hands
    if prompt_spec_hash is not None:
        meta["prompt_spec_hash"] = prompt_spec_hash
    if serving is not None:
        meta["serving"] = serving
    if posterior_fallback_counts is not None:
        meta["posterior_fallback_counts"] = posterior_fallback_counts
    if job is not None:
        meta["job"] = job
    if previous_jobs is not None:
        meta["previous_jobs"] = previous_jobs

    path = os.path.join(out_dir, "run_meta.json")
    write_json_atomic(path, meta)
    return path


def update_run_meta(path: str, **updates: Any) -> str:
    """Merge `updates` into an already-written run_meta.json; returns its path.

    The post-pass half of `write_run_meta`. Some
    provenance is only knowable after the matches run -- the whole-sweep
    `posterior_fallback_counts` is the first such key -- and `write_run_meta` is
    called before the first shard is dispatched. Rather than delay the startup
    write (which is what makes a killed arm diagnosable at all), the same file is
    re-opened and the post-match keys merged in.

    Same read-modify-write shape as `finalize_run_meta`, and the same refusal: a
    missing or unparseable file raises rather than writing a fresh stub, because
    the file was written by this same process at startup and inventing a
    replacement would be a silent fallback wearing provenance.

    A caller with nothing to report must not call this with `None` values: the
    "not set" vs "not observed" rule of `write_run_meta` applies to every key
    here too, and a written `null` claims the second when it means the first.
    """
    with open(path) as f:
        meta = json.load(f)
    meta.update(updates)
    write_json_atomic(path, meta)
    return path


def finalize_run_meta(path: str) -> str:
    """Stamp `finished_utc` onto an already-written run_meta.json; returns its
    path.

    `created_utc` is written before the first shard is dispatched, so it
    cannot tell a finished arm from one killed hours into its sweep.
    This is the completion half: the caller invokes it
    where its run genuinely concludes, and the key's presence is the claim
    that the run got there.

    The whole existing file is read back and re-written, so every key the
    startup write resolved survives. A missing or unparseable file raises
    (FileNotFoundError / JSONDecodeError) rather than writing a fresh stub:
    the file was written by the same process at startup, so its absence is a
    real anomaly and inventing a replacement would be a silent fallback
    wearing a timestamp.
    """
    with open(path) as f:
        meta = json.load(f)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(path, meta)
    return path


def write_json_atomic(
    path: str | os.PathLike,
    record: Any,
    *,
    default: Any = None,
    trailing_newline: bool = False,
) -> None:
    """
    write one JSON document through a temporary file and a rename, so a kill mid-write leaves the
    previous file (or none), never half of one that a reader would parse as truncated or fail on
    :param path: the destination path (str | PathLike)
    :param record: any JSON-serialisable value (dict | list)
    :param default: passed to `json.dump` for values it cannot serialise (callable | None)
    :param trailing_newline: end the file with a newline (bool)
    :returns: None
    :raises TypeError: when `record` is not JSON-serialisable; the partial temporary file is removed
        first. Any error from the rename propagates and leaves the complete temporary file in place
    """
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Unique per call, so two writers of one path cannot share a temporary file: with one shared
    # name the second rename finds the file already moved and fails. Ends in ".tmp" so the arm
    # sync excludes it.
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    # A dump that fails leaves a partial file, so it is removed. A complete one stays if the rename
    # is interrupted: the record is then absent and its temporary file whole, never truncated.
    try:
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2, default=default)
            if trailing_newline:
                f.write("\n")
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    os.replace(tmp, path)
