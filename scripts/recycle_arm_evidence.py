"""Per-arm evidence for a collected run (collection half).

Two jobs, one code path:

  1. LIVE monitoring while an arm is running. The arm is then PARTIAL: some
     shards have landed as parquet, `shard_outcomes.json` does not exist yet
     (run_sweep writes it at sweep END, `scripts/run_sweep.py:1737`). The
     script reports what is on disk, says PARTIAL, and - given `--log <log>`
     - greps for the endpoint-refusal line that `_on_error` prints the moment a
     shard dies (`scripts/run_sweep.py:1548`).

     `--log` must be the PER-INVOCATION run_sweep / run_experiment log for this
     arm (the driver prints its path as `LOG : ...`), never the driver's own
     phase log. The grep is a plain substring match, so any file that merely
     mentions `BadRequestError` or `maximum context length` - an operator note,
     a pasted traceback, a driver that printed its own watch pattern - reads as
     a refusal and kills a healthy arm. An unreadable `--log` is exit 4, not a
     quiet pass: on a PARTIAL arm it is the entire kill signal.
  2. The landing record for a finished arm: the parser counts
     (shards, rows, LM rows, belief_failed, fallback_used, retries,
     memory_truncated, cap-hit per channel, seat split, provenance), plus a
     verdict computed over every quantity it measured.

Counts come from a parser (polars over the parquet, json over the sidecars),
never from `wc -l` or a filename pattern, and the parquet glob is scoped to the
one named arm directory (a broader glob spans preserved archive directories
and overcounts shards).

Exit codes (distinct per outcome; a success signal is not the work):

  0  CLEAN     no kill criterion fired and, if `shard_outcomes.json` exists,
               n_requested == n_completed. A PARTIAL arm can be CLEAN: it means
               "nothing measured so far is a reason to stop", not "complete".
  2  KILL      at least one kill criterion fired. Stop the arm.
  3  INCOMPLETE  shard_outcomes.json records n_completed < n_requested for a
               reason that is not a kill criterion (a timeout, say). The arm is
               not citable as collected, but nothing says stop the cycle.
  4  OPERATIONAL  nothing to judge, or judged without the evidence asked for:
               the arm directory is missing or empty; or its parquet files hold
               ZERO rows (independent of `shard_outcomes.json` -- every
               row-based criterion is SKIPPED on an empty corpus, so "clean"
               would be a pass no computation backs); or its rows hold ZERO LM rows
               (every failure rate is 0 of 0, the same empty evidence one level up); or
               an explicitly passed
               `--log` could not be read; or the LM/opponent row accounting does
               not partition the arm, which makes the printed `belief_failed` /
               `fallback_used` rates something other than their labels.
               Precedence: an accounting defect outranks everything (the numbers
               a verdict would read are not those numbers), then an unreadable
               `--log`, then a fired kill criterion (exit 2, the
               actionable answer, and the criteria that survive an empty corpus
               come from the sidecar and the log, which are never vacuous), then
               the empty corpus, then incompleteness.

KILL CRITERIA:

  * PRIMARY (M2/M3): `shard_outcomes.json` carries an entry with
    `status == "failed"` and `error_type == "BadRequestError"`. That is the
    vLLM context-length refusal: `lm_vllm._complete` does not catch it, it
    propagates out of the shard, and run_sweep loses the whole 1000-hand shard.
    It is deliberately distinct from the incidental exit 3.
  * The same refusal seen EARLIER, in the sweep log: `--log` greps for
    `BadRequestError` or `maximum context length`. This is the only signal
    that exists before `shard_outcomes.json` is written.
  * `belief_failed` or `fallback_used` above FAILURE_RATE_CAP of the arm's LM
    rows, on every model alike. A failed belief or a harness fallback is a model
    result: the row is dropped at scoring and its rate is reported beside every
    number, so a rate under the cap is admissible; a rate over it says the arm
    did not collect the model's play and is refused. A rate cap rather than a
    zero-tolerance rule keeps one looped belief from refusing a large arm.
  * `memory_truncated > 0` on any row. `MEMORY_CHAR_BUDGET` 6000 cannot bind
    at rung M1-M3 (the pessimistic leduc block is 2917 chars), so a single
    truncation means the memory block is not what the design says it is.
  * A prompt-epoch break: any row with `prompt_schema_version !=
    PROMPT_SCHEMA_VERSION` (the current version; older epochs such as v4 are
    killed as stale by design), or a
    `prompt_spec_hash` that is not the one its (game, prompt_variant) should
    render. The hash check is variant-aware: base rows must equal
    `PROMPT_SPEC_HASHES[PROMPT_SCHEMA_VERSION]`, while a probe arm's rows must equal
    `prompt_spec_hash(variant_spec(family, probe_type))`, computed per game from
    the same registry the run rendered with. A variant name that is unknown, or
    that names a (game, probe) pair the registry records as ABSENT, is also a
    kill: no expectation is computable, and "uncheckable" is not "clean".
    Rows from another epoch are not comparable to this cycle's. The arm's
    `run_meta.resolved_config.probe.type` must agree with its rows' `probe_type`
    for the same reason.
  * More than one distinct `model_revision` (the harness git sha stamped per
    shard, not the weights: see WEIGHTS_REVISION_COLUMN below) is not itself
    the criterion: the repo HEAD can move mid-arm on commits that never touch
    the harness (for example two shas across one arm's shards whose diff
    touches only configs, the driver and documentation). The criterion is a
    computation: for every pair
    of distinct shas, `git diff --stat <a> <b> -- src/lmopt
    scripts/run_sweep.py scripts/run_experiment.py configs/models.yaml
    <arm's own config, if known>`. Empty diff for every pair is not a kill
    (the harness the arm ran was identical); a non-empty in-scope diff is a
    kill, `--stat` output included; a sha this repo's git cannot resolve is
    also a kill ("uncheckable" is not "clean"). See `model_revision_check`.
  * The weights pin is not what the roster says: any row whose
    `served_weights_revision` differs from `EXPECTED_WEIGHTS_REVISIONS[model_id]`
    (or `--expected-revision`), a run_meta value that differs, or a pinned model
    with no revision stamped at all. A model in neither the table nor the
    override is reported "revision not pinned, not checked" and is not killed.

What this script cannot see. Resuming a watcher-killed arm with
`--allow-existing` mixes pre-kill and post-kill shards in one directory. If the
resumed sweep completes, `shard_outcomes.json` records
`n_requested == n_completed` for the resumed run only, while every count here is
computed over the union of both runs; nothing in this output distinguishes them.
A resumed arm is not self-certifying - the driver logs, not this script, are the
record of what was killed and re-run.

Acceptance stamping (`--stamp`). Given `--stamp`, an arm that is both complete (state
COMPLETE: `shard_outcomes.json` present with n_requested == n_completed) and
CLEAN (exit 0) gets a sidecar `acceptance.json` written beside its
`run_meta.json`, carrying exactly six top-level keys: `provenance`
("keepable"), `gate` ("recycle_arm_evidence"), `exit_code`, `git_sha`,
`timestamp`, and `criteria` (`{"fired": [...], "pilot_gate": ...,
"exact_policy_ok": ...}`). The same run adds top-level `provenance: "keepable"`
to `run_meta.json` as a convenience fallback; where the two disagree the
sidecar is authoritative, because it is the artefact the gate writes and
re-writes, while the run_meta key is a copy for readers that already open
run_meta. A PARTIAL arm is never stamped, however clean it looks: "nothing
measured so far is a reason to stop" is not "this arm is complete", and neither
is exit 2/3/4. Nothing is ever stamped "throwaway": an ABSENT acceptance.json
is the throwaway verdict (the renderer's rule), so a re-collection that
overwrites an arm directory drops the stamp with the data it certified, and the
arm reverts to throwaway until this gate certifies it again. Re-running with
`--stamp` overwrites an existing sidecar (a re-run re-certifies, and says so).
Without `--stamp` the script writes nothing at all.

ADMITTING A measured belief_failed count (`--admit-belief-failed N
--admit-reason <text>`). This is an explicit,
PER-ARM decision that a specific, already-measured `belief_failed` count on
one arm is keepable, recorded in the acceptance stamp -- it is not a
threshold and not a default: nothing is admitted unless an operator names
this exact count and a reason on the command line for this exact invocation.
If the arm's LM-row `belief_failed` count is `<= N`, the belief_failed kill
criterion does not fire for that arm; if it EXCEEDS `N`, the criterion still
fires and the message names both numbers. No other kill criterion is
affected. When the flag is used and the arm is stamped, `acceptance.json`'s
`criteria` gains `belief_failed_admitted: {"limit": N, "observed": M,
"reason": "<text>"}`, so an arm stamped this way carries its admitted count
into every downstream read of that stamp and can never be mistaken for a
zero-failure arm.

Pinning A prior epoch (`--epoch N --epoch-reason <text>`). By default this
gate certifies against the LIVE
`PROMPT_SCHEMA_VERSION`/`PROMPT_SPEC_HASHES` registry, computed at import
time -- so a schema bump on `main` between two runs of the same
already-collected arm can silently change what "expected" means and
false-kill an internally-consistent prior-epoch arm. `--epoch N` certifies against `PROMPT_SPEC_HASHES[N]` instead
(N must already be a key in that append-only dict; an unrecognised N is a
usage error, never a silent fallback to the live version), and
`--epoch-reason` records why, verbatim, in `acceptance.json`'s
`criteria.epoch: {"pinned": N, "spec_hash": "...", "reason": "<text>"}`
(recorded only when `--epoch` is given, exactly like
`belief_failed_admitted`). Scope, stated rather than left to be discovered:
`--epoch` only pins the BASE-variant schema-version/hash check. A probe or
variant arm's expected hash is always computed from the live
`variant_spec`/probe registry, because no historical versioning exists for
variant content at all -- so `--epoch` REFUSES outright (exit 4) on any
arm carrying a non-base `prompt_variant` row, rather than silently pinning
the base rows while leaving the variant rows checked against a different
(live) standard with no record of the mismatch. Self-deriving the
certification epoch from the arm's own observed `prompt_schema_version`
column was considered and rejected: it would make the schema-version kill
criterion tautological (comparing the arm's rows to a value derived from
those same rows), silently disabling the check's actual job -- catching
drift onto a stale epoch nobody reviewed -- for every prior epoch, not
just the ones an operator has explicitly decided to accept.

Usage:

    PYTHONPATH=$PWD/src:$PWD .venv/bin/python scripts/recycle_arm_evidence.py \
        data/run_experiment/recycle-v8-m2/<model> \
        [--log <this run's log file>] \
        [--json]

More than one arm directory may be named; the process exit code is the worst
verdict over all of them (2 beats 3 beats 4 beats 0).
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Sequence

import polars as pl

from lmopt.agents.lm_vllm import (
    ACTION_MAX_TOKENS,
    BELIEF_MAX_TOKENS,
    PROMPT_SCHEMA_VERSION,
    PROMPT_SPEC_HASHES,
)
from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

from lmopt.games.families import GAME_FAMILIES
from lmopt.logging import certified
from lmopt.logging.run_meta import write_json_atomic
from lmopt.prompting.probe_variants import (
    BASE_VARIANT_NAME,
    VARIANT_NAME_BY_PROBE,
    variant_spec,
)
from lmopt.prompting.spec import prompt_spec_hash

EXIT_CLEAN = 0
EXIT_KILL = 2
EXIT_INCOMPLETE = 3
EXIT_OPERATIONAL = 4

# One name per exit code, and the human line the report prints for it. The
# short name is what a withheld stamp reports; the line is what `print_report`
# has always printed, kept identical.
VERDICT_NAMES = {
    EXIT_CLEAN: "CLEAN",
    EXIT_KILL: "KILL",
    EXIT_INCOMPLETE: "INCOMPLETE",
    EXIT_OPERATIONAL: "OPERATIONAL",
}
VERDICT_LINES = {
    EXIT_CLEAN: "CLEAN (exit 0)",
    EXIT_KILL: "KILL (exit 2) - stop this arm",
    EXIT_INCOMPLETE: "INCOMPLETE (exit 3)",
    EXIT_OPERATIONAL: "OPERATIONAL (exit 4)",
}

# The default prompt epoch this cycle certifies against when --epoch is not
# given: whatever is LIVE in the running tree, so a future epoch bump cannot
# leave a stale literal behind here. `evaluate_arm` resolves the actual
# expected version/hash pair per invocation (default or --epoch-pinned) and
# threads it explicitly rather than reading this name directly, so a
# pinned certification never depends on which checkout the gate happens to
# run from.
DEFAULT_SCHEMA_VERSION = PROMPT_SCHEMA_VERSION

# The prompt_variant -> probe_type direction of the registry's own map. A probe
# arm's rows carry the variant name; the registry is keyed by probe type, so the
# expected hash for a variant row is only computable through this inverse.
PROBE_TYPE_BY_VARIANT_NAME: dict[str, str] = {
    name: probe for probe, name in VARIANT_NAME_BY_PROBE.items()
}

# The sweep-log signatures of a vLLM context-length refusal. The first is the
# exception class name `_on_error` prints; the second is the substring of
# vLLM's own 400 body, which survives into `str(exc)`.
REFUSAL_PATTERNS = ("BadRequestError", "maximum context length")

# ---------------------------------------------------------------- stamping
# The acceptance sidecar's fixed vocabulary. `ACCEPTANCE_PROVENANCE` has one
# value on purpose: this gate only ever certifies keepable arms, and an ABSENT
# sidecar is the throwaway verdict (see the module docstring).
ACCEPTANCE_FILENAME = "acceptance.json"
ACCEPTANCE_GATE = "recycle_arm_evidence"
ACCEPTANCE_PROVENANCE = "keepable"
RUN_META_PROVENANCE_KEY = "provenance"

# The repo this script belongs to, for the git sha the stamp records. Not
# `lmopt.harness.match_runner.get_git_sha`, which resolves against the CWD: the
# stamp must name the checkout the gate ran from, and a gate invoked from
# somewhere else would otherwise silently record another repo's HEAD (or, in a
# tmp directory, "unknown").
REPO_ROOT = Path(__file__).resolve().parent.parent

# run_experiment's own per-arm markers, quoted from the lines it prints:
# `scripts/run_experiment.py:1245` (pilot PASSED), `:1199` and `:1164` (pilot
# SKIPPED via --skip-pilot-check), `:1653` (exact-policy OK) and `:1586`/`:1649`
# (exact-policy FAILED). A marker that stops matching those lines is a stamp
# that records `null` where evidence exists, which is why the strings live here
# next to their source lines rather than inline.
PILOT_PASSED_MARKER = "pilot gate: PASSED on"
PILOT_SKIPPED_MARKER = "pilot gate: SKIPPED"
EXACT_POLICY_OK_MARKER = "exact-policy pass: OK"
EXACT_POLICY_FAILED_MARKER = "exact-policy pass FAILED"

#: The share of an arm's LM rows that may be belief failures, and separately the share that
#: may be harness fallbacks, before the arm is refused. Rows under the cap are dropped at
#: scoring with the rate reported beside every number.
FAILURE_RATE_CAP = 0.01

# Fallback only (the model roster's pins) when no --config is given: a single
# global revision per model cannot express a per-arm, per-serving-box fact.
# Two Qwen revisions are both real, certifiable data -- cf98f3b3... (arms
# collected on one serving box) and 989aa798... (v6 M1 arms collected on
# the cluster, whose Qwen snapshot is a different revision of the same model).
# Flipping this table's value to either one false-KILLs the other box's arms;
# it is deliberately left at the older value, and the OLMo and Mistral pins
# agree across both boxes. `revision_pin` below prefers the arm's own --config file's declared
# `served_weights_revision` over this table whenever --config is given, which
# is the version of this check that actually asks the right question ("did
# this arm serve what its config asked for", not "does every arm on every box
# serve one hardcoded value"); this table remains the fallback when --config
# is not passed, and a third, unlisted revision is still caught either way.
EXPECTED_WEIGHTS_REVISIONS: dict[str, str] = {
    "Qwen/Qwen2.5-14B-Instruct": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    "allenai/OLMo-2-1124-13B-Instruct": "3a5c85baefbb1896a54d56fe2e76c0395627ddf4",
    "mistralai/Mistral-Nemo-Instruct-2407": "04d8a90549d23fc6bd7f642064003592df51e9b3",
}

# Which column carries THE WEIGHTS pin, and why it is not the obvious one.
# `model_revision` holds the harness GIT SHA, not the served weights:
# match_runner stamps `model_revision=git_sha` unconditionally, and
# src/lmopt/logging/schema.py says so in as many words (a landed arm carries
# one 40-hex value on every row). Comparing it to an HF weights revision would
# be a check that can only fail. The served weights identity is the separate
# `served_weights_revision` column, which run_experiment resolves per arm and
# run_sweep stamps on every row and lands in run_meta.json under
# `resolved_config.served_weights_revision`.
WEIGHTS_REVISION_COLUMN = "served_weights_revision"
RUN_META_REVISION_KEY = "served_weights_revision"

# The harness surface a distinct-model_revision pair's diff is checked
# against. A commit that touches only paths
# Outside this scope -- documentation, the driver script, another arm's
# config -- cannot have changed what this arm collected. The arm's own config
# file, when known, is appended per invocation (see `--config` / CONFIG_PATH
# resolution below); it is not a literal here because it is per-arm, not
# per-repo.
IN_SCOPE_DIFF_PATHS: tuple[str, ...] = (
    "src/lmopt",
    "scripts/run_sweep.py",
    "scripts/run_experiment.py",
    "configs/models.yaml",
)


# Correspondence-CHECK: weights-revision-pin
def config_declared_revision(
    config_path: str | None, model_id: str | None
) -> str | None:
    """`served_weights_revision` this arm's own `--config` YAML declared.

    Reads the file fresh off disk rather than trusting `run_meta.resolved_
    config.served_weights_revision`, which is populated from the same config
    at launch time and would make the check tautological (comparing a value
    to itself can never fail) if used as the expectation too. A fresh read
    catches the two cases a self-comparison cannot: the config was corrected
    after this arm ran (a genuine, currently-relevant expectation), or the
    launch config drifted from what the operator intended for this model.

    `None` on a missing/unreadable path, a missing `models:` list, or no entry
    matching `model_id` -- the caller falls back to `EXPECTED_WEIGHTS_
    REVISIONS`, never treats an unreadable config as a pass.
    """
    if not config_path or not model_id:
        return None
    import yaml

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    for entry in cfg.get("models") or []:
        if isinstance(entry, dict) and entry.get("id") == model_id:
            rev = entry.get("served_weights_revision")
            return str(rev) if rev else None
    return None


def revision_pin(
    model_id: str | None, override: str | None, config_path: str | None = None
) -> tuple[str | None, str]:
    """(expected weights revision, human-readable rule) for this arm.

    Precedence: --expected-revision (explicit, always wins) > this arm's own
    --config file's declared value (a real per-arm check) > the
    EXPECTED_WEIGHTS_REVISIONS fallback table (a global, possibly-stale
    per-model guess, kept only for invocations with no --config).
    """
    if override:
        return override, f"--expected-revision {override} (explicit override)"
    from_config = config_declared_revision(config_path, model_id)
    if from_config:
        return from_config, (
            f"{config_path} models[].served_weights_revision for {model_id}: "
            f"{from_config}"
        )
    if model_id and model_id in EXPECTED_WEIGHTS_REVISIONS:
        pin = EXPECTED_WEIGHTS_REVISIONS[model_id]
        return pin, (
            f"model-roster fallback pin for {model_id}: {pin} "
            "(no --config given; see EXPECTED_WEIGHTS_REVISIONS docstring)"
        )
    return None, (
        f"revision not pinned, not checked: {model_id!r} is not in "
        "EXPECTED_WEIGHTS_REVISIONS and no --expected-revision or --config "
        "with a matching models[] entry was given"
    )


# ---------------------------------------------------------------- loading


def load_arm(arm_dir: Path) -> tuple[pl.DataFrame | None, dict[str, list[str]]]:
    """Read every parquet in this directory (no recursion, no sibling arms).

    Returns (combined frame or None, {game: [shard filenames]}). The per-game
    shard attribution is parsed from each shard's own `game` column, never from
    its filename.
    """
    if not arm_dir.is_dir():
        return None, {}
    files = sorted(arm_dir.glob("*.parquet"))
    if not files:
        return None, {}
    frames: list[pl.DataFrame] = []
    shards_by_game: dict[str, list[str]] = {}
    for path in files:
        frame = pl.read_parquet(path)
        frames.append(frame)
        if "game" in frame.columns and len(frame):
            games = sorted(str(g) for g in frame["game"].unique().to_list())
        else:
            games = ["<unknown>"]
        for game in games:
            shards_by_game.setdefault(game, []).append(path.name)
    combined = pl.concat(frames, how="diagonal_relaxed")
    # A parquet that exists but holds zero rows is not a clean arm: it is an
    # arm with nothing measured in it. Returning it as a frame would let every
    # kill criterion pass vacuously (no row can carry a bad schema version) and
    # print VERDICT: CLEAN over an empty corpus.
    if len(combined) == 0:
        return None, shards_by_game
    return combined, shards_by_game


def load_run_meta(arm_dir: Path) -> dict:
    path = arm_dir / "run_meta.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_outcomes(arm_dir: Path) -> dict | None:
    path = arm_dir / "shard_outcomes.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def resolve_budgets(meta: dict) -> tuple[int, int, str, str]:
    """Action and belief token budgets for this arm, with their provenance.

    run_experiment resolves the per-model budgets from configs/models.yaml and
    lands them in run_meta.json's resolved_config; the key is present-but-null
    on an arm that never set them, in which case lm_vllm's own defaults are
    what the run actually used.
    """
    resolved = meta.get("resolved_config", {}) or {}
    action = resolved.get("action_max_tokens")
    belief = resolved.get("belief_max_tokens")
    action_src = "run_meta.resolved_config"
    belief_src = "run_meta.resolved_config"
    if action is None:
        action, action_src = ACTION_MAX_TOKENS, "lm_vllm.ACTION_MAX_TOKENS default"
    if belief is None:
        belief, belief_src = BELIEF_MAX_TOKENS, "lm_vllm.BELIEF_MAX_TOKENS default"
    return int(action), int(belief), action_src, belief_src


def resolve_model_id(meta: dict, df: pl.DataFrame | None) -> str | None:
    model = meta.get("model_id") or (meta.get("resolved_config", {}) or {}).get("model")
    if model:
        return str(model)
    if df is not None and "model_id" in df.columns:
        ids = [m for m in df["model_id"].unique().to_list() if m is not None]
        if len(ids) == 1:
            return str(ids[0])
        if ids:
            return "|".join(sorted(str(i) for i in ids))
    return None


def belief_rule(model_id: str | None) -> tuple[bool, str]:
    """
    the belief_failed and fallback_used kill criterion, which applies to every model alike
    :param model_id: the arm's model id, kept so a caller's call shape does not change (str or None)
    :returns: (enforced, human-readable rule); enforced is always True (tuple of bool and str)
    """
    return True, (
        f"APPLIED to every model: belief_failed and fallback_used each capped at "
        f"{FAILURE_RATE_CAP:.0%} of LM rows; rows under the cap are dropped at scoring and "
        f"the rate is reported"
    )


def over_cap(count: int | None, n_lm: int | None) -> bool:
    """
    whether a failure count exceeds FAILURE_RATE_CAP of the arm's LM rows
    :param count: the failed rows (int or None)
    :param n_lm: the arm's LM rows (int or None)
    :returns: True only when both are known, n_lm is positive and count / n_lm > the cap (bool)
    """
    if not count or not n_lm:
        return False
    return count / n_lm > FAILURE_RATE_CAP


def belief_failed_admission(
    belief_enforced: bool,
    belief_count: int,
    admit_limit: int | None,
    admit_reason: str | None,
    n_lm: int,
) -> dict | None:
    """The `--admit-belief-failed` admission applied to this arm's belief_failed count.

    This is an explicit, PER-ARM admission of a specific, already-measured
    `belief_failed` count, recorded in the acceptance stamp when the arm is
    stamped -- it is not a threshold and not a default: it has no effect on
    any arm unless an operator names this exact count and a reason on the
    command line for this invocation, and it changes nothing about any other
    kill criterion. An arm stamped under this admission carries its admitted
    count (`belief_failed_admitted` in `acceptance.json`) into every
    downstream read of that stamp, so a reader can never mistake it for a
    zero-failure arm.

    Returns `None` when `admit_limit` is `None` (the flag was not given).
    `"admitted"` is True only when the belief_failed criterion would
    otherwise have fired (the count is over FAILURE_RATE_CAP of `n_lm`) and the
    observed count is within the admitted limit; a limit given for an arm
    that had nothing to admit (belief_failed within the cap) is
    recorded as not admitted, since nothing was actually forgiven.
    """
    if admit_limit is None:
        return None
    if not isinstance(n_lm, int):
        raise TypeError(f"n_lm must be the arm's LM row count, got {n_lm!r}")
    would_fire = belief_enforced and over_cap(belief_count, n_lm)
    return {
        "limit": admit_limit,
        "observed": belief_count,
        "reason": admit_reason,
        "admitted": would_fire and belief_count <= admit_limit,
    }


def fallback_used_admission(
    fallback_count: int,
    n_lm: int,
    admit_rate: float | None,
    admit_reason: str | None,
) -> dict | None:
    """
    the `--admit-fallback-rate` admission applied to this arm's fallback_used count
    An explicit, per-arm admission of an already-measured harness-fallback rate, recorded in
    the acceptance stamp as `fallback_used_admitted`. It is not a threshold and not a default:
    it has no effect unless an operator names a rate and a reason for this invocation, it never
    changes FAILURE_RATE_CAP, and it touches no other kill criterion.
    :param fallback_count: the arm's fallback_used rows (int)
    :param n_lm: the arm's LM row count (int)
    :param admit_rate: the highest fallback_used share of LM rows to admit, or None (float or None)
    :param admit_reason: the admission's reason, recorded verbatim (str or None)
    :returns: None when no rate was given, else a dict with limit, observed (share of LM rows),
        count, reason and admitted, which is True only when the global cap would have fired and
        the observed share is within the limit (dict or None)
    """
    if admit_rate is None:
        return None
    if not isinstance(n_lm, int):
        raise TypeError(f"n_lm must be the arm's LM row count, got {n_lm!r}")
    observed = fallback_count / n_lm if n_lm else 0.0
    return {
        "limit": admit_rate,
        "observed": observed,
        "count": fallback_count,
        "reason": admit_reason,
        "admitted": over_cap(fallback_count, n_lm) and observed <= admit_rate,
    }


def expected_variant_hash(
    game: str,
    variant: str,
    expected_schema_version: int,
    expected_spec_hash: str,
) -> tuple[str | None, str]:
    """(expected `prompt_spec_hash`, the rule) for one (game, prompt_variant).

    `expected_schema_version`/`expected_spec_hash` are the CALLER's resolved
    pair (the live default, or an `--epoch`-pinned override) -- this
    function never reads the module's own default constants directly, so a
    pinned certification is never silently mixed with the live one.

    A probe ARM DOES not carry THE BASE hash, and must not. `prompt_spec_hash`
    is content identity: `spec.py:_canonical_payload` substitutes the variant's
    content for its game and hashes all five games, so a relabel kuhn row's hash
    is `prompt_spec_hash(variant_spec("kuhn", "relabel"))` and differs from
    `PROMPT_SPEC_HASHES[4]` by construction (at schema version 4: relabel
    kuhn a1870a17..., leduc e8ab5bf8..., liarsdice ee010aed..., darkhex
    11af68ab..., payoff_scale kuhn 26e69ad3...). The
    expected value is COMPUTED from the registry the run rendered with, never
    tabulated here, so a variant whose content changes moves this expectation
    with it instead of leaving a stale literal behind.

    `None` as the expected hash means "no expectation is computable", which is a
    KILL, not a pass: an unrecognised variant name, or a variant name for a
    (game, probe) pair the registry records as ABSENT, describes rows whose
    content nothing in the harness can vouch for.

    Keyed on the game family (`games.families.GAME_FAMILIES`), matching
    `orchestration/shard.py:_resolve_prompt_spec`: the depth variants
    ("darkhex_3x3") share one family's prompt content.
    """
    if variant == BASE_VARIANT_NAME:
        return expected_spec_hash, (
            f"PROMPT_SPEC_HASHES[{expected_schema_version}] (base variant)"
        )
    probe_type = PROBE_TYPE_BY_VARIANT_NAME.get(variant)
    if probe_type is None:
        return None, (
            f"UNKNOWN prompt_variant {variant!r}: neither {BASE_VARIANT_NAME!r} "
            f"nor one of {sorted(PROBE_TYPE_BY_VARIANT_NAME)}, so no spec exists "
            "to compute an expected hash from"
        )
    family = GAME_FAMILIES.get(game, game)
    try:
        spec = variant_spec(family, probe_type)
    except ValueError as exc:
        return None, (
            f"no {probe_type!r} variant is computable for game {game!r} "
            f"(family {family!r}): {exc}"
        )
    if spec is None:
        return None, (
            f"the registry records NO {probe_type!r} variant for {family!r} "
            f"(recorded absence), so rows labelled {variant!r} claim a probe "
            "the harness cannot render"
        )
    return prompt_spec_hash(spec), (
        f"prompt_spec_hash(variant_spec({family!r}, {probe_type!r}))"
    )


def prompt_hash_groups(
    lm: pl.DataFrame,
    expected_schema_version: int,
    expected_spec_hash: str,
) -> list[dict]:
    """Per (game, prompt_variant): the observed hashes and the expected one.

    Grouped rather than compared arm-wide because a probe arm carries one hash
    PER game (each game's variant substitutes into a different slot of the same
    five-game payload), so a single arm-wide comparison can only be right for a
    single-game arm.

    An arm with no `prompt_variant` column (an older schema) is read as
    entirely base: that is what such rows were, and it keeps the check on rather
    than skipping it.

    `expected_schema_version`/`expected_spec_hash` are the caller's resolved
    pair (live default, or `--epoch`-pinned); see `expected_variant_hash`.
    """
    if not len(lm) or "prompt_spec_hash" not in lm.columns:
        return []
    frame = lm
    if "prompt_variant" not in frame.columns:
        frame = frame.with_columns(pl.lit(BASE_VARIANT_NAME).alias("prompt_variant"))
    if "game" not in frame.columns:
        frame = frame.with_columns(pl.lit("<unknown>").alias("game"))
    groups: list[dict] = []
    keys = (
        frame.select(["game", "prompt_variant"])
        .unique()
        .sort(["game", "prompt_variant"])
        .rows()
    )
    for game, variant in keys:
        block = frame.filter(
            (pl.col("game") == game) & (pl.col("prompt_variant") == variant)
        )
        observed = sorted(
            {h for h in block["prompt_spec_hash"].unique().to_list() if h is not None}
        )
        expected, rule = expected_variant_hash(
            str(game),
            str(variant),
            expected_schema_version,
            expected_spec_hash,
        )
        groups.append(
            {
                "game": str(game),
                "prompt_variant": str(variant),
                "rows": len(block),
                "observed": observed,
                "expected": expected,
                "rule": rule,
                "null_hashes": int(block["prompt_spec_hash"].is_null().sum()),
            }
        )
    return groups


def resolved_config_path(meta: dict) -> str | None:
    """This arm's own `--config` YAML path, if run_meta.json ever records it.

    `_load_yaml`/`load_config` (scripts/run_sweep.py,
    scripts/run_experiment.py) read `args.config` and never copy the path
    itself into `resolved_config`. This function
    exists so a future run_meta schema that does stamp it is picked up with no
    call-site change; today it returns None on every real arm and the caller
    falls back to `--config` / "config not checked".
    """
    resolved = meta.get("resolved_config", {}) or {}
    for key in ("config", "config_path", "__config_path__"):
        value = resolved.get(key)
        if value:
            return str(value)
    return None


def configured_probe_type(meta: dict) -> str:
    """The probe type this arm's run_meta says it ran.

    `"none"` is run_sweep's own default for an absent or empty `probe:` block
    (`scripts/run_sweep.py:475`), so a base arm and a config-less arm answer the
    same thing here as they do there, and an arm whose rows say `relabel` while
    its config says nothing is a disagreement, not a missing field.
    """
    return str(
        ((meta.get("resolved_config", {}) or {}).get("probe") or {}).get("type", "none")
    )


# ---------------------------------------------------------------- metrics


def _count_true(df: pl.DataFrame, column: str) -> int | None:
    """
    number of rows where a boolean column is True; None if absent
    A null in `fallback_used` counts as True, the pilot gate's reading of the never-null
    column; a null in any other column counts as False.
    :param df: the arm's frame (pl.DataFrame)
    :param column: the boolean column to count (str)
    :returns: the count, None when the column is absent (int | None)
    """
    if column not in df.columns:
        return None
    return int(df[column].fill_null(column == "fallback_used").cast(pl.Boolean).sum())


def _count_notnull(df: pl.DataFrame, column: str) -> int | None:
    """Number of rows where a column is non-null; None if the column is absent."""
    if column not in df.columns:
        return None
    return int(df[column].is_not_null().sum())


def _count_positive(df: pl.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    return int((df[column].fill_null(0) > 0).sum())


def _cap_hits(df: pl.DataFrame, column: str, budget: int) -> dict:
    """Cap hits on one decode channel: rows at or above the resolved budget.

    The budget is the one this arm resolved, not a module constant: a per-model
    budget change would otherwise be scored against the wrong ceiling.
    """
    if column not in df.columns:
        return {
            "present": False,
            "hits": None,
            "denom": 0,
            "null": len(df),
            "budget": budget,
        }
    series = df[column]
    denom = int(series.is_not_null().sum())
    nulls = int(series.is_null().sum())
    hits = int((series.fill_null(-1) >= budget).sum()) if denom else 0
    return {
        "present": True,
        "hits": hits,
        "denom": denom,
        "null": nulls,
        "budget": budget,
    }


def _distinct(df: pl.DataFrame, column: str) -> list | None:
    if column not in df.columns:
        return None
    return sorted(
        (v for v in df[column].unique().to_list() if v is not None),
        key=str,
    )


def block_metrics(df: pl.DataFrame, action_budget: int, belief_budget: int) -> dict:
    """Every per-scope quantity, over one frame (an arm, or one game of one)."""
    lm = (
        df.filter(pl.col("prompt").is_not_null())
        if "prompt" in df.columns
        else df.head(0)
    )
    n_lm = len(lm)
    belief_failed = _count_true(lm, "belief_failed")
    belief_salvaged = _count_notnull(lm, "belief_salvage_kind")
    fallback_used = _count_true(lm, "fallback_used")
    memory_truncated = _count_true(lm, "memory_truncated")
    seat = {}
    if "lm_seat" in lm.columns and n_lm:
        for value in sorted(
            v for v in lm["lm_seat"].unique().to_list() if v is not None
        ):
            seat[str(value)] = int((lm["lm_seat"] == value).sum())
    return {
        "rows": len(df),
        "lm_rows": n_lm,
        "belief_failed": belief_failed,
        "belief_failed_rate": (
            (belief_failed / n_lm) if (belief_failed is not None and n_lm) else None
        ),
        "belief_salvaged": belief_salvaged,
        "belief_salvaged_rate": (
            (belief_salvaged / n_lm) if (belief_salvaged is not None and n_lm) else None
        ),
        "fallback_used": fallback_used,
        "fallback_used_rate": (
            (fallback_used / n_lm) if (fallback_used is not None and n_lm) else None
        ),
        "n_retries_gt0": _count_positive(lm, "n_retries"),
        "belief_n_retries_gt0": _count_positive(lm, "belief_n_retries"),
        "memory_truncated": memory_truncated,
        "memory_truncated_rate": (
            (memory_truncated / n_lm)
            if (memory_truncated is not None and n_lm)
            else None
        ),
        "memory_truncated_column_present": "memory_truncated" in lm.columns,
        "action_cap": _cap_hits(lm, "completion_tokens", action_budget),
        "belief_cap": _cap_hits(lm, "belief_completion_tokens", belief_budget),
        "seat_split": seat,
    }


def outcomes_summary(outcomes: dict | None) -> dict:
    if outcomes is None:
        return {"present": False}
    shards = outcomes.get("shards", []) or []
    failed = [s for s in shards if s.get("status") == "failed"]
    error_types: dict[str, int] = {}
    for shard in failed:
        key = str(shard.get("error_type"))
        error_types[key] = error_types.get(key, 0) + 1
    per_game: dict[str, dict[str, int]] = {}
    for shard in shards:
        game = str(shard.get("game"))
        entry = per_game.setdefault(game, {"requested": 0, "completed": 0, "failed": 0})
        entry["requested"] += 1
        if shard.get("status") == "completed":
            entry["completed"] += 1
        elif shard.get("status") == "failed":
            entry["failed"] += 1
    return {
        "present": True,
        "n_requested": outcomes.get("n_requested"),
        "n_completed": outcomes.get("n_completed"),
        "n_failed": outcomes.get("n_failed"),
        "error_types": error_types,
        "badrequest_failures": [
            {
                "game": s.get("game"),
                "opponent": s.get("opponent"),
                "seed": s.get("seed"),
                "error": s.get("error"),
            }
            for s in failed
            if s.get("error_type") == "BadRequestError"
        ],
        "per_game": per_game,
    }


def grep_log(log_path: Path | None) -> dict:
    """Scan one run_sweep/run_experiment invocation log for the refusal lines.

    An explicitly passed `--log` that cannot be read is an OPERATIONAL failure,
    never a quiet pass: on a PARTIAL arm the log is the whole kill signal, so a
    typo in the path would otherwise report CLEAN while the endpoint refused
    every over-length request. `unreadable` is what `evaluate_arm` turns into
    exit 4.
    """
    if log_path is None:
        return {"present": False, "requested": False, "hits": []}
    try:
        text = log_path.read_text(errors="replace")
    except OSError as exc:
        return {
            "present": False,
            "requested": True,
            "path": str(log_path),
            "unreadable": str(exc),
            "hits": [],
        }
    hits = [
        line.strip()
        for line in text.splitlines()
        if any(pattern in line for pattern in REFUSAL_PATTERNS)
    ]
    return {"present": True, "requested": True, "path": str(log_path), "hits": hits}


def _git_sha_resolves(sha: str) -> bool:
    """Whether `sha` names a commit this repo's git can resolve.

    An unresolvable sha is a kill, not a pass:
    "uncheckable" is not "clean", and a diff against a sha this checkout does
    not have would otherwise silently report as empty.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
            text=True,
            timeout=10,
            # The exit code is read below, so a failure is handled, not raised.
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _git_diff_stat(
    sha_a: str, sha_b: str, scope_paths: Sequence[str]
) -> tuple[bool | None, str]:
    """
    the in-scope diff between two shas
    :param sha_a: one sha (str)
    :param sha_b: the other (str)
    :param scope_paths: the paths the diff is limited to (sequence of str)
    :returns: (is_empty, the `--stat` text); is_empty is None, with git's stderr in place of the
        text, when `git diff` exits non-zero, because its empty stdout then means "could not
        compare" and not "no difference" (tuple[bool | None, str])
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "diff",
            "--stat",
            sha_a,
            sha_b,
            "--",
            *scope_paths,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        # The exit code is read below, so a failure is handled, not raised.
        check=False,
    )
    if result.returncode != 0:
        return None, f"git diff exited {result.returncode}: {result.stderr.strip()}"
    output = result.stdout.strip()
    return (output == ""), output


def model_revision_check(revisions: list[str], config_path: str | None) -> dict:
    """The model_revision kill verdict: a computation, never a distinct-count.

    `revisions` is `>1` shas only when there is anything to check; with 0 or 1
    the arm's harness never moved and `verdict` stays `None`. Scope is
    IN_SCOPE_DIFF_PATHS plus the arm's own config file when known -- an
    unresolvable sha outranks a clean diff (an uncheckable pair anywhere in
    the set makes the whole verdict uncheckable), which in turn outranks
    reporting a diff for pairs that could not be checked.
    """
    scope = list(IN_SCOPE_DIFF_PATHS)
    if config_path:
        scope.append(str(config_path))
    result: dict = {
        "model_revisions": list(revisions),
        "scope_paths": scope,
        "config_checked": bool(config_path),
        "pairs": [],
        "unresolvable": [],
        "undiffable": [],
        "verdict": None,  # None | "clean" | "in_scope_diff" | "unresolvable"
    }
    if len(revisions) <= 1:
        return result
    for sha in revisions:
        if not _git_sha_resolves(sha):
            result["unresolvable"].append(sha)
    if result["unresolvable"]:
        result["verdict"] = "unresolvable"
        return result
    any_nonempty = False
    for sha_a, sha_b in itertools.combinations(sorted(revisions), 2):
        empty, stat = _git_diff_stat(sha_a, sha_b, scope)
        result["pairs"].append({"a": sha_a, "b": sha_b, "empty": empty, "stat": stat})
        if empty is None:
            result["undiffable"].append({"a": sha_a, "b": sha_b, "error": stat})
        else:
            any_nonempty = any_nonempty or not empty
    if result["undiffable"]:
        # A pair git could not diff is uncheckable, which outranks clean, the same as a sha that
        # does not resolve.
        result["verdict"] = "unresolvable"
        return result
    result["verdict"] = "in_scope_diff" if any_nonempty else "clean"
    return result


def opponent_fallback_used_count(df: pl.DataFrame) -> int | None:
    """`fallback_used` count on opponent rows (`prompt` is null).

    The Bayesian opponent stamps `fallback_used` on many rows of an arm.
    `block_metrics`'s LM figure already excludes these; this is the count it
    excludes, printed on its own line so it is never misread as an LM figure.
    """
    if "fallback_used" not in df.columns or "prompt" not in df.columns:
        return None
    return _count_true(df.filter(pl.col("prompt").is_null()), "fallback_used")


def _lm_only_partition_defect(
    df: pl.DataFrame, column: str, lm_count: int | None
) -> str | None:
    """The reason `column`'s LM/opponent split does not partition the arm, or None.

    LM-row count + opponent-row count for `column` must equal the arm total.
    This is the check that fails if `block_metrics`'s `lm` filter ever drifts
    from `prompt is not null`, or if an opponent row's flag is double-counted
    into the LM figure -- in which case the `belief_failed` / `fallback_used`
    rates this script prints are not the quantities they are labelled with, and
    no verdict over them is computable.

    It returns a reason instead of asserting, for two reasons. A bare `assert`
    exits 1 with a traceback, which collides with argparse's usage exit and with
    every other unhandled error, so the operator cannot tell an accounting
    defect from a bad command line; and `assert` is stripped under `python -O`,
    which would delete the guard entirely on the one interpreter flag that
    changes nothing else about this script. The caller maps a returned reason to
    EXIT_OPERATIONAL, so the behaviour is identical with and without `-O`.
    """
    if column not in df.columns or "prompt" not in df.columns:
        return None
    total = _count_true(df, column) or 0
    opponent = _count_true(df.filter(pl.col("prompt").is_null()), column) or 0
    lm = lm_count or 0
    if lm + opponent == total:
        return None
    return (
        f"{column} LM/opponent split does not partition all rows: "
        f"lm={lm} opponent={opponent} total={total}"
    )


# ---------------------------------------------------------------- verdict


def evaluate_arm(
    arm_dir: Path,
    log_path: Path | None,
    expected_revision: str | None = None,
    config_path: str | None = None,
    admit_belief_failed: int | None = None,
    admit_reason: str | None = None,
    epoch: int | None = None,
    epoch_reason: str | None = None,
    admit_fallback_rate: float | None = None,
    admit_fallback_reason: str | None = None,
) -> dict:
    """The whole report for one arm, verdict included.

    The verdict is a computation over every quantity below, not a lookup: each
    kill criterion reads a number this function already measured.

    `admit_belief_failed` / `admit_reason` apply only to the belief_failed
    criterion (see `belief_failed_admission`); every other kill criterion is
    computed exactly as without them. `admit_fallback_rate` / `admit_fallback_reason` likewise
    apply only to the fallback_used criterion (see `fallback_used_admission`).

    `epoch` / `epoch_reason` pin the BASE-variant
    schema-version/hash check to `PROMPT_SPEC_HASHES[epoch]` instead of the
    LIVE `PROMPT_SCHEMA_VERSION`, so re-certifying an arm collected under a
    prior epoch does not depend on which checkout the gate happens to run
    from. `None` (the default) uses the live epoch. A probe/variant arm's expected hash is never pinned
    (no historical variant-content versioning exists): `epoch` REFUSES
    outright (exit 4) on any arm carrying a non-base `prompt_variant` row,
    rather than silently leaving those rows on the live registry while
    pinning the rest.
    """
    df, shards_by_game = load_arm(arm_dir)
    meta = load_run_meta(arm_dir)
    outcomes = load_outcomes(arm_dir)
    log = grep_log(log_path)

    # The resolved (schema_version, spec_hash) pair this invocation certifies
    # against: the live default, or the --epoch override. Threaded explicitly
    # into every place that would otherwise read the module-level default.
    expected_schema_version = epoch if epoch is not None else DEFAULT_SCHEMA_VERSION
    expected_spec_hash = PROMPT_SPEC_HASHES[expected_schema_version]

    action_budget, belief_budget, action_src, belief_src = resolve_budgets(meta)
    model_id = resolve_model_id(meta, df)
    belief_enforced, belief_rule_text = belief_rule(model_id)
    # The arm's own config -- used for both the model_revision diff scope and
    # (below) the weights-revision pin: run_meta first (a future schema might
    # stamp it, see resolved_config_path), else the --config this invocation
    # was given.
    arm_config_path = resolved_config_path(meta) or config_path
    expected_pin, pin_rule_text = revision_pin(
        model_id, expected_revision, arm_config_path
    )

    resolved = meta.get("resolved_config", {}) or {}
    report: dict = {
        "arm_dir": str(arm_dir),
        "exists": arm_dir.is_dir(),
        "model_id": model_id,
        "memory_rung_config": resolved.get("memory_rung"),
        "seat_mode_config": resolved.get("seat_mode"),
        "budgets": {
            "action": action_budget,
            "action_source": action_src,
            "belief": belief_budget,
            "belief_source": belief_src,
        },
        "belief_rule": belief_rule_text,
        "configured_probe_type": configured_probe_type(meta),
        "revision_pin_rule": pin_rule_text,
        "expected_weights_revision": expected_pin,
        "run_meta_weights_revision": resolved.get(RUN_META_REVISION_KEY),
        "shards_on_disk": (
            len(list(arm_dir.glob("*.parquet"))) if arm_dir.is_dir() else 0
        ),
        "shards_on_disk_by_game": {
            g: len(v) for g, v in sorted(shards_by_game.items())
        },
        "outcomes": outcomes_summary(outcomes),
        "log": log,
        "kill_criteria_fired": [],
        "incomplete_reason": None,
        "model_revision_check": model_revision_check([], arm_config_path),
        "opponent_fallback_used": None,
        "belief_admission": None,
        "fallback_admission": None,
        "expected_schema_version": expected_schema_version,
        "epoch_pin": (
            {"pinned": epoch, "spec_hash": expected_spec_hash, "reason": epoch_reason}
            if epoch is not None
            else None
        ),
        "epoch_pin_refused": False,
    }
    # "shards on disk" is the parquet file count; the by-game split is parsed
    # from each shard's own `game` column, so a shard carrying two games is
    # counted under both there and once here.
    report["partial"] = not report["outcomes"]["present"]

    partition_defects: list[str] = []
    if df is not None:
        report["totals"] = block_metrics(df, action_budget, belief_budget)
        per_game = {}
        if "game" in df.columns:
            for game in sorted(
                str(g) for g in df["game"].unique().to_list() if g is not None
            ):
                per_game[game] = block_metrics(
                    df.filter(pl.col("game") == game), action_budget, belief_budget
                )
        report["per_game"] = per_game
        lm = (
            df.filter(pl.col("prompt").is_not_null())
            if "prompt" in df.columns
            else df.head(0)
        )
        report["prompt_hash_groups"] = prompt_hash_groups(
            lm,
            expected_schema_version,
            expected_spec_hash,
        )
        report["probe_type_rows"] = _distinct(df, "probe_type")
        report["distinct"] = {
            "prompt_schema_version": _distinct(lm, "prompt_schema_version"),
            "prompt_spec_hash": _distinct(lm, "prompt_spec_hash"),
            "prompt_variant": _distinct(lm, "prompt_variant"),
            "model_revision": _distinct(df, "model_revision"),
            WEIGHTS_REVISION_COLUMN: _distinct(df, WEIGHTS_REVISION_COLUMN),
            "backend": _distinct(df, "backend"),
            "memory_rung": _distinct(df, "memory_rung"),
        }
        # LM-only scoping, made explicit rather than assumed: `belief_failed`
        # and `fallback_used` in `report["totals"]` come from `block_metrics`'s
        # own `lm` filter (prompt is not null); this checks that filter still
        # partitions the arm with the opponent rows, and prints the opponent
        # count on its own line so it is never misread as an LM figure. A defect
        # here is OPERATIONAL (below), not an exception: it says the printed
        # rates are not the labelled quantities.
        report["opponent_fallback_used"] = opponent_fallback_used_count(df)
        partition_defects += [
            reason
            for reason in (
                _lm_only_partition_defect(
                    df, "fallback_used", report["totals"]["fallback_used"]
                ),
                _lm_only_partition_defect(
                    df, "belief_failed", report["totals"]["belief_failed"]
                ),
            )
            if reason
        ]
    else:
        report["totals"] = None
        report["per_game"] = {}
        report["distinct"] = {}
        report["prompt_hash_groups"] = []
        report["probe_type_rows"] = None

    fired: list[str] = report["kill_criteria_fired"]

    # PRIMARY: a shard lost to the endpoint's context-length refusal.
    for failure in report["outcomes"].get("badrequest_failures", []) or []:
        fired.append(
            "PRIMARY shard_outcomes BadRequestError: "
            f"{failure['game']} vs {failure['opponent']} seed={failure['seed']}"
        )
    # The same refusal, seen in the log before shard_outcomes.json exists.
    for line in log.get("hits", []):
        fired.append(f"sweep log refusal line: {line}")

    totals = report["totals"]
    belief_count = (totals["belief_failed"] or 0) if totals is not None else 0
    n_lm = totals["lm_rows"] if totals is not None else 0
    belief_admission = belief_failed_admission(
        belief_enforced,
        belief_count,
        admit_belief_failed,
        admit_reason,
        n_lm,
    )
    report["belief_admission"] = belief_admission
    if totals is not None:
        fallback_count = totals["fallback_used"] or 0
        fallback_admission = fallback_used_admission(
            fallback_count, n_lm, admit_fallback_rate, admit_fallback_reason
        )
        report["fallback_admission"] = fallback_admission
        if over_cap(fallback_count, n_lm) and not (
            fallback_admission is not None and fallback_admission["admitted"]
        ):
            fired.append(
                f"fallback_used = {fallback_count} of {n_lm} LM rows "
                f"({fallback_count / n_lm:.4%}) on {model_id} exceeds the "
                f"{FAILURE_RATE_CAP:.0%} cap ({belief_rule_text})"
            )
        if belief_enforced and over_cap(belief_count, n_lm):
            if belief_admission is not None and belief_admission["admitted"]:
                pass  # admitted explicitly; see belief_admission
            elif belief_admission is not None:
                fired.append(
                    f"belief_failed = {belief_count} on {model_id} EXCEEDS "
                    "the --admit-belief-failed admission (admitted up to "
                    f"{admit_belief_failed}, observed {belief_count}); "
                    f"{belief_rule_text}"
                )
            else:
                fired.append(
                    f"belief_failed = {belief_count} of {n_lm} LM rows "
                    f"({belief_count / n_lm:.4%}) on {model_id} exceeds the "
                    f"{FAILURE_RATE_CAP:.0%} cap ({belief_rule_text})"
                )
        if (totals["memory_truncated"] or 0) > 0:
            fired.append(
                f"memory_truncated = {totals['memory_truncated']} "
                "(the char budget cannot bind at M1-M3)"
            )
        versions = report["distinct"].get("prompt_schema_version") or []
        bad_versions = [v for v in versions if v != expected_schema_version]
        if bad_versions:
            fired.append(
                f"prompt_schema_version {bad_versions} != {expected_schema_version}"
            )
        # The hash check is variant-aware. A probe arm renders variant content,
        # so its rows carry that variant's hash per game; comparing them to the
        # base hash would kill every probe arm. Base rows are checked against
        # the base hash.
        for group in report["prompt_hash_groups"]:
            where = (
                f"game {group['game']} variant {group['prompt_variant']} "
                f"({group['rows']} LM rows)"
            )
            if group["expected"] is None:
                fired.append(
                    f"no expected prompt_spec_hash for {where}: {group['rule']}; "
                    f"observed {group['observed']}"
                )
                continue
            wrong = [h for h in group["observed"] if h != group["expected"]]
            if wrong:
                fired.append(
                    f"prompt_spec_hash {wrong} != {group['expected']} for "
                    f"{where}, expected from {group['rule']}"
                )

        # The arm's own config must agree with what its rows are labelled. A
        # probe arm whose run_meta says `none` (or a base arm whose rows say
        # `relabel`) is an arm nothing identifies: the variant check above would
        # then be verifying content against a probe the operator never
        # configured.
        configured = configured_probe_type(meta)
        row_probes = [str(p) for p in (report["probe_type_rows"] or [])]
        mismatched = [p for p in row_probes if p != configured]
        if mismatched:
            fired.append(
                f"probe_type {mismatched} on rows != run_meta.resolved_config."
                f"probe.type {configured!r}"
            )
        # model_revision: a computation over the in-scope source diff between
        # every pair of distinct shas, not a bare distinct-count (see
        # IN_SCOPE_DIFF_PATHS and the module docstring). Scoped to LM rows, matching "the LM rows carry >1
        # distinct model_revision" -- the column is stamped identically on
        # opponent rows too (match_runner, unconditionally), so this differs
        # from `report["distinct"]["model_revision"]` (all rows) only in
        # wording, never in the set of shas found.
        lm_revisions = _distinct(lm, "model_revision") or []
        mrc = model_revision_check(lm_revisions, arm_config_path)
        report["model_revision_check"] = mrc
        if mrc["verdict"] == "unresolvable":
            for sha in mrc["unresolvable"]:
                fired.append(
                    f"model_revision: sha {sha} is NOT RESOLVABLE in this "
                    "repo (git cat-file -e failed) -- uncheckable is not clean"
                )
            for pair in mrc["undiffable"]:
                fired.append(
                    f"model_revision: the in-scope diff between {pair['a']} and "
                    f"{pair['b']} could not be computed ({pair['error']}) -- "
                    "uncheckable is not clean"
                )
        elif mrc["verdict"] == "in_scope_diff":
            for pair in mrc["pairs"]:
                if pair["empty"]:
                    continue
                fired.append(
                    f"model_revision: {len(lm_revisions)} distinct shas "
                    f"{lm_revisions}, in-scope diff NON-EMPTY between "
                    f"{pair['a']} and {pair['b']} (scope {mrc['scope_paths']}):\n"
                    f"{pair['stat']}"
                )

        # The provenance clause, mechanised. Checked against served_weights_revision, not model_revision: see the
        # WEIGHTS_REVISION_COLUMN note at the top of this file.
        if expected_pin is not None:
            stamped = report["distinct"].get(WEIGHTS_REVISION_COLUMN)
            if not stamped:
                fired.append(
                    f"{WEIGHTS_REVISION_COLUMN} is NOT STAMPED on any row, so the "
                    f"pin {expected_pin} cannot be verified ({pin_rule_text})"
                )
            else:
                wrong = [r for r in stamped if r != expected_pin]
                if wrong:
                    fired.append(
                        f"{WEIGHTS_REVISION_COLUMN} {wrong} != expected pin "
                        f"{expected_pin} ({pin_rule_text})"
                    )
            # Symmetric with the row branch above: "not recorded" is not
            # "matches". A pinned model whose run_meta carries no revision has
            # nothing corroborating the stamp, and run_experiment aborts rather
            # than launch an arm with an unresolvable revision, so an absent key
            # on a v4 arm is anomalous by construction. Guarding this on
            # `is not None` would make the two halves of one criterion disagree
            # about what missing evidence means.
            meta_rev = report["run_meta_weights_revision"]
            if meta_rev is None:
                fired.append(
                    f"run_meta.resolved_config.{RUN_META_REVISION_KEY} is absent or "
                    f"null, so the pin {expected_pin} is uncorroborated by run_meta "
                    f"({pin_rule_text})"
                )
            elif meta_rev != expected_pin:
                fired.append(
                    f"run_meta.resolved_config.{RUN_META_REVISION_KEY} {meta_rev!r} "
                    f"!= expected pin {expected_pin}"
                )

    outcomes_sum = report["outcomes"]
    if outcomes_sum["present"]:
        requested = outcomes_sum.get("n_requested")
        completed = outcomes_sum.get("n_completed")
        if (
            isinstance(requested, int)
            and isinstance(completed, int)
            and completed != requested
        ):
            report["incomplete_reason"] = (
                f"shard_outcomes.json: {completed}/{requested} completed, "
                f"error_types {outcomes_sum['error_types']}"
            )
        elif not (isinstance(requested, int) and isinstance(completed, int)):
            report["incomplete_reason"] = (
                f"shard_outcomes.json malformed: n_requested={requested!r}, "
                f"n_completed={completed!r}"
            )

    # An LM/opponent partition defect outranks every other outcome, including
    # KILL: it says the `belief_failed` / `fallback_used` figures this run
    # computed are not the quantities they are labelled with, so neither CLEAN
    # nor a fired criterion over them is a claim the script is entitled to make.
    # It is not a bare `assert`, which would be an exit-1 traceback
    # indistinguishable from a usage error, and absent altogether under -O.
    if partition_defects:
        report["exit_code"] = EXIT_OPERATIONAL
        report["operational_reason"] = (
            "LM-row accounting is inconsistent, so no verdict over the LM rates "
            "is computable: " + "; ".join(partition_defects)
        )
    # An explicitly requested --log that could not be read is OPERATIONAL and
    # outranks everything else this run could conclude: on a PARTIAL arm the log
    # is the whole kill signal, so "CLEAN" computed without it is a claim the
    # script is not entitled to make.
    elif log.get("requested") and not log.get("present"):
        report["exit_code"] = EXIT_OPERATIONAL
        report["operational_reason"] = (
            f"--log {log.get('path')} could not be read ({log.get('unreadable')}); "
            "the refusal signal for a PARTIAL arm was NOT consulted"
        )
    # --epoch on a probe/variant arm is OPERATIONAL and outranks everything
    # else this run could conclude, same tier and same reasoning as the
    # unreadable --log case above: `--epoch` can only correctly pin the
    # BASE-variant schema-version/hash check (no historical versioning
    # exists for variant content, see expected_variant_hash), so a pin
    # requested on an arm carrying non-base rows is an input this gate
    # cannot honor, not a verdict about the arm. Checked against
    # `report["distinct"]["prompt_variant"]` (computed straight from the LM
    # rows' `prompt_variant` column), not `prompt_hash_groups`: the latter
    # goes `[]` whenever `prompt_spec_hash` is absent from the LM columns,
    # which would silently accept the pin on a non-base arm that happens to
    # be missing that column -- exactly the silent-partial-pin this branch
    # exists to prevent. `None` (no `prompt_variant` column at all, an
    # older schema) reads as "no non-base rows", same as
    # `prompt_hash_groups`' own base-only default for that case; the
    # empty-corpus/nothing-to-judge tiers below still catch a truly empty
    # arm, since they run after this branch, not before.
    elif epoch is not None and any(
        v != BASE_VARIANT_NAME for v in (report["distinct"].get("prompt_variant") or [])
    ):
        blocking = sorted(
            v for v in report["distinct"]["prompt_variant"] if v != BASE_VARIANT_NAME
        )
        report["exit_code"] = EXIT_OPERATIONAL
        report["epoch_pin_refused"] = True
        report["operational_reason"] = (
            f"--epoch {epoch} was requested but this arm carries non-base "
            f"prompt_variant row(s) {blocking}: --epoch only pins the "
            "BASE-variant check, and applying it partially here would "
            "silently leave those rows checked against the LIVE variant "
            "registry regardless of the pin"
        )
    elif fired:
        # A fired criterion still outranks the empty-corpus case below: the
        # criteria that can fire with no rows come from shard_outcomes.json and
        # the sweep log, which are independent evidence and are never vacuous.
        # Exit 2 is also the more actionable answer -- it says stop the arm.
        report["exit_code"] = EXIT_KILL
    elif df is None and report["shards_on_disk"] > 0:
        # Parquet files exist but hold zero rows. This is OPERATIONAL IN its own
        # Right, independent of shard_outcomes.json: every row-based criterion
        # lives inside `if totals is not None`, so on an empty corpus they are
        # all SKIPPED, not satisfied. A completed arm (n_requested ==
        # n_completed) whose shards are empty would otherwise print
        # VERDICT: CLEAN over nothing at all -- a pass that no computation
        # backs. Deliberately above the incompleteness branch so the verdict
        # cannot depend on whether the sidecar happens to exist.
        report["exit_code"] = EXIT_OPERATIONAL
        report["operational_reason"] = (
            f"{report['shards_on_disk']} parquet file(s) on disk under {arm_dir} "
            "but ZERO rows in them: every row-based kill criterion was SKIPPED, "
            "so no verdict on this arm is computable"
        )
    elif totals is not None and totals["lm_rows"] == 0:
        # Rows exist but none is an LM row, so every failure rate above is 0 of 0 and was
        # SKIPPED, not satisfied: `over_cap` is False for any count over zero LM rows. The
        # same empty-evidence case as the zero-row parquet, one level up.
        report["exit_code"] = EXIT_OPERATIONAL
        report["operational_reason"] = (
            f"{totals['rows']} row(s) under {arm_dir} but ZERO LM rows (no row carries a "
            "prompt): every failure-rate criterion was SKIPPED, so no verdict on this arm is "
            "computable"
        )
    elif report["incomplete_reason"]:
        report["exit_code"] = EXIT_INCOMPLETE
    elif df is None and not outcomes_sum["present"] and not log.get("hits"):
        report["exit_code"] = EXIT_OPERATIONAL
        report["operational_reason"] = (
            f"nothing to judge under {arm_dir}: no parquet files and no "
            "shard_outcomes.json"
        )
    else:
        report["exit_code"] = EXIT_CLEAN
    return report


# ---------------------------------------------------------------- stamping


def gate_git_sha() -> str:
    """HEAD of the repo this script lives in, or "unknown".

    "unknown" is a logged fallback, not a silent one (no silent
    fallbacks): it lands verbatim in acceptance.json's `git_sha`, where a
    reader can see that the gate could not resolve a sha.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            # The exit code is read below, so a failure is handled, not raised.
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def log_gate_markers(log_path: Path | None) -> tuple[str | None, bool | None]:
    """(pilot_gate, exact_policy_ok_marker) read from one run_experiment log.

    Deliberately re-reads the file rather than extending `grep_log`'s dict:
    without `--stamp` this script's output must not change, and `--json`
    prints the log dict verbatim.

    SKIPPED outranks PASSED when a log carries both (a multi-arm invocation).
    "A skipped gate is not a passed gate" is run_experiment's own wording at
    `scripts/run_experiment.py:1643`, so the ambiguous case reports the weaker
    evidence rather than certifying on another arm's PASSED line.

    Returns `exact_policy_ok` as the LOG's answer only; whether the arm ran an
    exact-policy pass at all is a run_meta question, resolved by the caller.
    """
    if log_path is None:
        return None, None
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None, None
    if PILOT_SKIPPED_MARKER in text:
        pilot = "SKIPPED"
    elif PILOT_PASSED_MARKER in text:
        pilot = "PASSED"
    else:
        pilot = None
    if EXACT_POLICY_FAILED_MARKER in text:
        exact = False
    elif EXACT_POLICY_OK_MARKER in text:
        exact = True
    else:
        exact = None
    return pilot, exact


def exact_policy_enabled(report_meta: dict) -> bool:
    """Did this arm's resolved config turn the exact-policy pass on?

    run_experiment gates the pass on `exact_policy.enabled` (`:1040`), and the
    resolved config lands in run_meta.json, so an arm that never ran the pass
    reports `exact_policy_ok: null` instead of `false` -- "not run" and "failed"
    are different findings.
    """
    resolved = report_meta.get("resolved_config", {}) or {}
    return bool((resolved.get("exact_policy") or {}).get("enabled", False))


def stampable(report: dict) -> str | None:
    """None if this arm may be stamped, else the reason it may not.

    The two conditions are independent: exit 0 says "nothing measured is a
    reason to stop", and COMPLETE says "everything that was requested landed".
    A PARTIAL arm can be exit 0 while its sweep is still running, so the
    completeness half is checked from `shard_outcomes.json` directly and not
    inferred from the verdict.
    """
    if report["exit_code"] != EXIT_CLEAN:
        return "verdict is not CLEAN"
    outcomes = report["outcomes"]
    if report["partial"] or not outcomes.get("present"):
        return "arm is PARTIAL (no shard_outcomes.json)"
    requested, completed = outcomes.get("n_requested"), outcomes.get("n_completed")
    if not (isinstance(requested, int) and isinstance(completed, int)):
        return (
            f"shard_outcomes.json malformed: n_requested={requested!r}, "
            f"n_completed={completed!r}"
        )
    if requested != completed:
        return f"shard_outcomes.json: {completed}/{requested} completed"
    return None


def acceptance_payload(report: dict, log_path: Path | None) -> dict:
    """The sidecar's six top-level keys, computed over this arm's report."""
    meta = load_run_meta(Path(report["arm_dir"]))
    pilot_gate, exact_from_log = log_gate_markers(log_path)
    exact_ok = exact_from_log if exact_policy_enabled(meta) else None
    # model_revision_diff_in_scope mirrors model_revision_check's verdict
    # vocabulary, with "clean" spelled "empty" for a reader of the sidecar
    # who never sees the Python: None means <=1 revision (nothing to check).
    mrc = report.get("model_revision_check") or {}
    diff_verdict = mrc.get("verdict")
    criteria: dict = {
        "fired": list(report["kill_criteria_fired"]),
        "pilot_gate": pilot_gate,
        "exact_policy_ok": exact_ok,
        "model_revisions": list(mrc.get("model_revisions") or []),
        "model_revision_diff_in_scope": (
            "empty" if diff_verdict == "clean" else diff_verdict
        ),
    }
    # Recorded only when --admit-belief-failed was actually given for this
    # invocation (belief_admission is not None): an explicit, per-arm decision,
    # never a default key on every stamp.
    admission = report.get("belief_admission")
    if admission is not None:
        criteria["belief_failed_admitted"] = {
            "limit": admission["limit"],
            "observed": admission["observed"],
            "reason": admission["reason"],
        }
    # Same convention for the fallback_used admission: present only when a rate was given.
    fallback_admission = report.get("fallback_admission")
    if fallback_admission is not None:
        criteria["fallback_used_admitted"] = {
            "limit": fallback_admission["limit"],
            "observed": fallback_admission["observed"],
            "count": fallback_admission["count"],
            "reason": fallback_admission["reason"],
        }
    # Recorded only when the arm holds at least one salvaged belief, on the
    # same "a key records a fact only when the fact exists" convention as
    # belief_failed_admitted. Information, never a kill criterion: a salvaged
    # belief counts as parsed (belief_failed False), so without this an arm
    # with salvaged beliefs would stamp as a zero-failure arm. Whether salvage
    # should ever fire a criterion is not decided here.
    totals = report.get("totals") or {}
    if totals.get("belief_salvaged"):
        criteria["belief_salvaged"] = {
            "count": totals["belief_salvaged"],
            "rate": totals["belief_salvaged_rate"],
        }
    # Recorded only when --epoch was actually given (epoch_pin is not None):
    # an explicit, per-arm certification decision, never a default key on
    # every stamp (same convention as belief_failed_admitted above).
    # spec_hash is included, not just the version integer, because
    # PROMPT_SPEC_HASHES is documented as editable in place -- this makes
    # the stamp self-contained even if a table entry is later edited.
    epoch_pin = report.get("epoch_pin")
    if epoch_pin is not None:
        criteria["epoch"] = {
            "pinned": epoch_pin["pinned"],
            "spec_hash": epoch_pin["spec_hash"],
            "reason": epoch_pin["reason"],
        }
    return {
        "provenance": ACCEPTANCE_PROVENANCE,
        "gate": ACCEPTANCE_GATE,
        "exit_code": report["exit_code"],
        "git_sha": gate_git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "criteria": criteria,
    }


def _refuse_other_names(*paths: Path) -> None:
    """
    refuse before any write when a file the stamp rewrites has other names
    :param paths: the files a stamp replaces (Path)
    :raises certified.ModifyRefused: when an existing one has any other name, naming them
    """
    for path in paths:
        if not path.is_file():
            continue
        names = certified.other_names(path)
        if names.count:
            listed = ", ".join(str(p) for p in names.paths) or "none under the root"
            raise certified.ModifyRefused(
                f"{path} has {names.count} other name(s) ({listed}; "
                f"{names.outside_root} outside {names.root}): re-certifying would leave them "
                "with the old bytes. Nothing was written. Recompose after: remove the "
                "composite that shares it, stamp, then compose again",
                other_paths=names.paths,
            )


def stamp_arm(
    report: dict, log_path: Path | None, stream: IO[str] = sys.stdout
) -> Path | None:
    """Write acceptance.json (+ run_meta provenance) for a CLEAN COMPLETE arm.

    Returns the sidecar path, or None when the stamp is withheld. A withheld
    stamp is announced, never silent: an operator who asked for a stamp and got
    no file must be told which verdict refused it. Both files are written
    through `certified.modify`, and the run is locked only once both are in
    place, so a withheld or refused stamp leaves the run untouched.
    """
    refusal = stampable(report)
    arm_dir = Path(report["arm_dir"])
    if refusal is not None:
        verdict = VERDICT_NAMES[report["exit_code"]]
        print(
            f"STAMP: withheld (verdict {verdict}, exit "
            f"{report['exit_code']}): {refusal}",
            file=stream,
        )
        return None
    path = arm_dir / ACCEPTANCE_FILENAME
    meta_path = arm_dir / "run_meta.json"
    _refuse_other_names(path, meta_path)
    if path.exists():
        print(
            f"STAMP: {path} already exists; a re-run re-certifies, " "overwriting it",
            file=stream,
        )
    payload = acceptance_payload(report, log_path)
    with certified.modify(
        arm_dir,
        ACCEPTANCE_FILENAME,
        holder="recycle_arm_evidence",
        purpose="stamp acceptance.json",
    ) as target:
        write_json_atomic(target, payload, default=str, trailing_newline=True)
    meta = load_run_meta(arm_dir)
    if meta:
        meta[RUN_META_PROVENANCE_KEY] = ACCEPTANCE_PROVENANCE
        with certified.modify(
            arm_dir,
            "run_meta.json",
            holder="recycle_arm_evidence",
            purpose="stamp run_meta.json",
        ) as target:
            write_json_atomic(target, meta, default=str, trailing_newline=True)
    else:
        # An arm with no readable run_meta.json still gets its sidecar (the
        # authoritative record); the convenience copy is what is missing, and
        # saying so beats writing a run_meta.json this gate did not author.
        print(
            f"STAMP: {meta_path} is missing or unreadable; the "
            f"{RUN_META_PROVENANCE_KEY} convenience key was NOT added "
            "(acceptance.json is the authority)",
            file=stream,
        )
    certified.lock_files(arm_dir)
    certified.lock_dir(arm_dir)
    print(f"STAMP: wrote {path} (provenance {ACCEPTANCE_PROVENANCE})", file=stream)
    return path


# ---------------------------------------------------------------- printing


def _fmt_rate(count, rate) -> str:
    if count is None:
        return "column ABSENT on this arm (older schema)"
    return f"{count}" + (f" (rate {rate:.6f})" if rate is not None else "")


def _print_block(name: str, block: dict, indent: str = "  ") -> None:
    print(f"{indent}{name}: rows {block['rows']}, LM rows {block['lm_rows']}")
    print(
        f"{indent}  belief_failed: {_fmt_rate(block['belief_failed'], block['belief_failed_rate'])}"
    )
    print(
        f"{indent}  belief_salvaged: {_fmt_rate(block['belief_salvaged'], block['belief_salvaged_rate'])}"
    )
    print(
        f"{indent}  fallback_used: {_fmt_rate(block['fallback_used'], block['fallback_used_rate'])}"
    )
    print(
        f"{indent}  n_retries>0: {block['n_retries_gt0']}   belief_n_retries>0: {block['belief_n_retries_gt0']}"
    )
    if block["memory_truncated_column_present"]:
        print(
            f"{indent}  memory_truncated: {_fmt_rate(block['memory_truncated'], block['memory_truncated_rate'])}"
        )
    else:
        print(f"{indent}  memory_truncated: column ABSENT on this arm (older schema)")
    action, belief = block["action_cap"], block["belief_cap"]
    if action["present"]:
        print(
            f"{indent}  action cap-hit: {action['hits']}/{action['denom']} "
            f"(completion_tokens >= {action['budget']}), null completion_tokens: {action['null']}"
        )
    else:
        print(f"{indent}  action cap-hit: completion_tokens column ABSENT")
    if belief["present"]:
        print(
            f"{indent}  belief cap-hit: {belief['hits']}/{belief['denom']} "
            f"(belief_completion_tokens >= {belief['budget']}), "
            f"null belief_completion_tokens: {belief['null']}"
        )
    else:
        print(f"{indent}  belief cap-hit: belief_completion_tokens column ABSENT")
    print(f"{indent}  seat split (lm_seat on LM rows): {block['seat_split'] or 'n/a'}")


def print_report(report: dict) -> None:
    print("=" * 78)
    print(f"ARM: {report['arm_dir']}")
    state = (
        "PARTIAL (no shard_outcomes.json yet: the sweep has not finished)"
        if report["partial"]
        else "COMPLETE (shard_outcomes.json present)"
    )
    print(f"state: {state}")
    print(
        f"model_id: {report['model_id']}   memory_rung(config): "
        f"{report['memory_rung_config']}   seat_mode(config): {report['seat_mode_config']}"
    )
    epoch_pin = report.get("epoch_pin")
    if epoch_pin is not None and report["epoch_pin_refused"]:
        print(
            f"certified epoch: REFUSED -- --epoch {epoch_pin['pinned']} was requested "
            "but this arm carries non-base prompt_variant rows (see OPERATIONAL "
            "reason below); nothing was certified against the pin"
        )
    elif epoch_pin is not None:
        print(
            f"certified epoch: {report['expected_schema_version']} "
            f"(PINNED via --epoch, reason: {epoch_pin['reason']})"
        )
    else:
        print(f"certified epoch: {report['expected_schema_version']} (live default)")
    budgets = report["budgets"]
    print(
        f"resolved budgets: action budget {budgets['action']} "
        f"[{budgets['action_source']}], belief budget {budgets['belief']} "
        f"[{budgets['belief_source']}]"
    )
    print(f"failure-rate kill criterion: {report['belief_rule']}")
    admission = report.get("belief_admission")
    if admission and admission.get("admitted"):
        print(
            f"belief_failed: {admission['observed']} row(s) ADMITTED by "
            f"explicit admission (limit {admission['limit']}): "
            f"{admission['reason']}"
        )
    fb_admission = report.get("fallback_admission")
    if fb_admission and fb_admission.get("admitted"):
        print(
            f"fallback_used: {fb_admission['count']} row(s) ({fb_admission['observed']:.4%} "
            f"of LM rows) ADMITTED by explicit admission (limit {fb_admission['limit']:.4%}): "
            f"{fb_admission['reason']}"
        )
    print(f"weights-revision pin: {report['revision_pin_rule']}")
    print(f"run_meta {RUN_META_REVISION_KEY}: {report['run_meta_weights_revision']}")
    print(
        f"shards on disk: {report['shards_on_disk']} parquet file(s); "
        f"by game {report['shards_on_disk_by_game'] or '{}'}"
    )

    outcomes = report["outcomes"]
    if outcomes["present"]:
        print(
            f"shard_outcomes.json: requested {outcomes['n_requested']}, "
            f"completed {outcomes['n_completed']}, failed {outcomes['n_failed']}; "
            f"error_types {outcomes['error_types'] or '{}'}"
        )
        for game, entry in sorted(outcomes["per_game"].items()):
            print(
                f"  {game:<12} requested {entry['requested']:>4} "
                f"completed {entry['completed']:>4} failed {entry['failed']:>4}"
            )
    else:
        print(
            "shard_outcomes.json: ABSENT (arm PARTIAL; the sweep log is the "
            "only refusal signal until it is written)"
        )

    log = report["log"]
    if log.get("present"):
        print(
            f"log scan ({log['path']}): {len(log['hits'])} refusal line(s) "
            f"matching {REFUSAL_PATTERNS}"
        )
        for line in log["hits"][:5]:
            print(f"  ! {line}")
    elif log.get("unreadable"):
        print(f"log scan: UNREADABLE {log.get('path')} ({log['unreadable']})")

    if report["totals"] is not None:
        print("-" * 78)
        _print_block("TOTAL", report["totals"])
        print(
            f"opponent fallback_used (not an LM figure): "
            f"{report.get('opponent_fallback_used')}"
        )
        for game, block in sorted(report["per_game"].items()):
            _print_block(game, block)
        print("-" * 78)
        print(
            f"probe_type: rows {report['probe_type_rows']}, "
            f"run_meta.resolved_config.probe.type "
            f"{report['configured_probe_type']!r}"
        )
        print(
            "prompt_spec_hash per (game, prompt_variant) - expected is COMPUTED "
            "from the variant registry, not tabulated:"
        )
        for group in report["prompt_hash_groups"]:
            print(
                f"  {group['game']:<12} {group['prompt_variant']:<18} "
                f"rows {group['rows']:>6}  expected {group['expected']}  "
                f"observed {group['observed']}"
            )
            print(f"      rule: {group['rule']}")
        print("distinct provenance values (LM rows for prompt_*, all rows otherwise):")
        for key, values in report["distinct"].items():
            print(f"  {key}: {values}")
        mrc = report.get("model_revision_check") or {}
        revisions = mrc.get("model_revisions") or []
        if len(revisions) > 1:
            if not mrc.get("config_checked"):
                print(
                    "model_revision: config not checked (no --config given; "
                    f"scope is {list(IN_SCOPE_DIFF_PATHS)} only, the arm's own "
                    "config file is NOT in scope)"
                )
            if mrc.get("verdict") == "clean":
                print(
                    f"model_revision: {len(revisions)} distinct shas "
                    f"{revisions}, in-scope diff EMPTY (harness identical)"
                )
    elif report["shards_on_disk"] > 0:
        # Not "no parquet": there are files, and they are EMPTY. Saying "none
        # yet" here would describe a sweep that has not started, which is a
        # different arm from one whose shards landed with nothing in them.
        print(
            f"{report['shards_on_disk']} parquet file(s) present but ZERO rows "
            "in them: no per-game or per-channel counts are computable"
        )
    else:
        print("no parquet in this directory yet")

    print("-" * 78)
    if report["kill_criteria_fired"]:
        print(f"KILL CRITERIA FIRED ({len(report['kill_criteria_fired'])}):")
        for item in report["kill_criteria_fired"]:
            print(f"  * {item}")
    if report.get("incomplete_reason"):
        print(f"INCOMPLETE: {report['incomplete_reason']}")
    if report.get("operational_reason"):
        print(f"OPERATIONAL: {report['operational_reason']}")
    print(f"VERDICT: {VERDICT_LINES[report['exit_code']]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-arm evidence and kill-criterion verdict for "
        "a collected run.",
        epilog="Exit codes: 0 clean, 2 kill criterion fired, 3 shards "
        "incomplete for another reason, 4 nothing to judge.",
    )
    parser.add_argument(
        "arm_dir",
        nargs="+",
        type=Path,
        help="Arm directory, e.g. "
        "data/run_experiment/recycle-v8-m2/<model>",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="The PER-INVOCATION run_sweep/run_experiment log "
        "for this arm, to grep for the endpoint "
        "context-length refusal (the only signal a PARTIAL "
        "arm has). NOT the driver's phase log: see the "
        "module docstring. Unreadable => exit 4.",
    )
    parser.add_argument(
        "--expected-revision",
        default=None,
        help="Override the model-roster weights-revision "
        "pin for this arm's model.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="The --config YAML this arm was launched with. "
        "Extends the model_revision in-scope diff to this "
        "arm's own config (run_meta.json never records the "
        "path itself, see resolved_config_path). Without "
        "this, and with no run_meta fallback, the diff is "
        "scoped to src/lmopt + the two orchestrator "
        "entrypoints + configs/models.yaml only, and the "
        "report says so loudly.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable report instead of the " "human one.",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="Write acceptance.json (provenance keepable) and "
        "run_meta.json's provenance key for each arm that "
        "is CLEAN (exit 0) AND COMPLETE. Withheld, loudly, "
        "on a PARTIAL arm or any other verdict. Without "
        "this flag the script writes nothing.",
    )
    parser.add_argument(
        "--admit-belief-failed",
        type=int,
        default=None,
        metavar="N",
        help="Explicit, per-arm admission "
        "that a belief_failed count of "
        "up to N on THIS arm is keepable. Requires "
        "--admit-reason. Does not touch any other kill "
        "criterion; if the observed count exceeds N the "
        "criterion still fires. Not a threshold, not a "
        "default: has no effect unless given.",
    )
    parser.add_argument(
        "--admit-reason",
        type=str,
        default=None,
        metavar="TEXT",
        help="Required companion to --admit-belief-failed: "
        "the reason for the admission, recorded verbatim in "
        "acceptance.json's belief_failed_admitted.",
    )
    parser.add_argument(
        "--admit-fallback-rate",
        type=float,
        default=None,
        metavar="RATE",
        help="Explicit, per-arm admission that a fallback_used share of "
        "up to RATE of THIS arm's LM rows (0.015 is 1.5%%) is "
        "keepable. Requires --admit-fallback-reason. Never "
        "changes the global cap or any other kill criterion; if "
        "the observed share exceeds RATE the criterion still "
        "fires. Not a default: has no effect unless given.",
    )
    parser.add_argument(
        "--admit-fallback-reason",
        type=str,
        default=None,
        metavar="TEXT",
        help="Required companion to --admit-fallback-rate, recorded "
        "verbatim in acceptance.json's fallback_used_admitted.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        metavar="N",
        help="Certify against PROMPT_SPEC_HASHES[N] instead of "
        "the LIVE PROMPT_SCHEMA_VERSION ("
        "a schema bump false-kills an internally-consistent "
        "prior-epoch arm when the gate is re-run from a "
        "post-bump checkout). Requires --epoch-reason. Only "
        "pins the BASE-variant schema-version/hash check: a "
        "probe/variant arm's expected hash is always "
        "computed from the live variant registry (no "
        "historical variant-content versioning exists), so "
        "--epoch REFUSES outright (exit 4) on any arm "
        "carrying a non-base prompt_variant row, rather "
        "than silently applying a partial pin. An N not in "
        "PROMPT_SPEC_HASHES is a usage error, never a "
        "silent fallback to the live version. Not a "
        "default: has no effect unless given.",
    )
    parser.add_argument(
        "--epoch-reason",
        type=str,
        default=None,
        metavar="TEXT",
        help="Required companion to --epoch: the reason this "
        "arm is certified against a specific epoch, "
        "recorded verbatim in acceptance.json's "
        "criteria.epoch.",
    )
    args = parser.parse_args(argv)

    if (args.admit_belief_failed is None) != (args.admit_reason is None):
        parser.error(
            "--admit-belief-failed and --admit-reason must be given "
            "together (an admission is never recorded without its reason)"
        )
    if args.admit_belief_failed is not None and args.admit_belief_failed < 1:
        parser.error(
            f"--admit-belief-failed must be >= 1, got {args.admit_belief_failed}"
        )
    if (args.admit_fallback_rate is None) != (args.admit_fallback_reason is None):
        parser.error(
            "--admit-fallback-rate and --admit-fallback-reason must be given "
            "together (an admission is never recorded without its reason)"
        )
    if args.admit_fallback_rate is not None and not (
        FAILURE_RATE_CAP < args.admit_fallback_rate < 1
    ):
        parser.error(
            f"--admit-fallback-rate must be above the {FAILURE_RATE_CAP} cap and below 1, "
            f"got {args.admit_fallback_rate}"
        )
    if (args.epoch is None) != (args.epoch_reason is None):
        parser.error(
            "--epoch and --epoch-reason must be given together (a pinned "
            "epoch is never recorded without its reason)"
        )
    if args.epoch is not None and args.epoch not in PROMPT_SPEC_HASHES:
        parser.error(
            f"--epoch {args.epoch} is not a recognized epoch; "
            f"PROMPT_SPEC_HASHES has {sorted(PROMPT_SPEC_HASHES)}"
        )

    reports = [
        evaluate_arm(
            arm,
            args.log,
            args.expected_revision,
            args.config,
            args.admit_belief_failed,
            args.admit_reason,
            args.epoch,
            args.epoch_reason,
            args.admit_fallback_rate,
            args.admit_fallback_reason,
        )
        for arm in args.arm_dir
    ]
    # Worst verdict wins; the payload's exit_code is the returned code, taken
    # from the same computation the human report prints.
    order = {EXIT_KILL: 3, EXIT_INCOMPLETE: 2, EXIT_OPERATIONAL: 1, EXIT_CLEAN: 0}
    exit_code = max((r["exit_code"] for r in reports), key=lambda c: order[c])
    fired = [item for r in reports for item in r["kill_criteria_fired"]]

    if args.json:
        print(
            json.dumps(
                {"exit_code": exit_code, "kill_criteria_fired": fired, "arms": reports},
                indent=2,
                default=str,
            )
        )
    else:
        for report in reports:
            print_report(report)
        if len(reports) > 1:
            print("=" * 78)
            print(f"OVERALL VERDICT: exit {exit_code}")

    if args.stamp:
        # Per ARM, not per invocation: the worst verdict decides the process
        # exit code, but a clean complete arm named alongside a killed one is
        # still clean, and each arm's sidecar records its own verdict.
        # Under --json the STAMP lines go to stderr so stdout stays parseable.
        stream = sys.stderr if args.json else sys.stdout
        for report in reports:
            try:
                stamp_arm(report, args.log, stream)
            except certified.CertifiedError as exc:
                print(f"STAMP: refused for {report['arm_dir']}: {exc}", file=stream)
                exit_code = max(exit_code, EXIT_OPERATIONAL, key=lambda c: order[c])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
