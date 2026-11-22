#!/usr/bin/env python
"""Exact per-info-set action distribution, computed as an offline pass.

Computes ``P(action | info-set)`` exactly (not sampled, not single-token
top-k) by asking the served model's own completions endpoint for the
token-level logprobs of each legal action's literal text, conditioned on the
model's own chat-templated rendering of the info-set's logged prompt, via
``POST /v1/completions`` with ``echo=True, max_tokens=0, logprobs=1``. That
combination returns per-token logprobs for caller-SUPPLIED tokens (the
"echo" tokens), so ``P(action) = prod_i P(token_i | prefix)``, renormalised
over the info-set's legal actions.

Three hard requirements (the cost of getting any one of them wrong is noted
inline at the enforcing code):

1. Read the logged ``prompt`` column verbatim; never call ``build_prompt``.
   Otherwise a later prompt-format change invalidates, after the fact,
   every number this pass produces, silently.
2. Render the served model's own chat template, never a hardcoded string.
   Raw vs templated gave 0.9969 vs 0.6792 on the same Kuhn node, a 3x
   error that would propagate silently into every downstream number.
   Resolved from the live endpoint's own ``/tokenize`` (a vLLM extension
   outside the OpenAI schema), as token IDS carried end to end: never from
   local weights (not present for the 14B arm), never from a bundled
   template file, and never via a ``/detokenize`` text round trip. That
   round trip is lossy for control tokens on at least one served model
   (``render_chat_prefix_ids``'s docstring has the measurement) and
   reintroduces server-side re-tokenisation, which is the defect this pass
   exists to avoid. The one ``/detokenize`` call in this pass's dependency
   chain (``resolve_chat_template_hash``, imported from
   ``lmopt.serving.chat_template``) is provenance-only: it fingerprints
   endpoint configuration for drift detection and never supplies a prompt
   this pass scores against.
3. Cost is per (info-set, action), not per node: this pass deduplicates
   logged decision nodes down to distinct (game, game_params, info_set)
   keys before making any network call.

No silent fallbacks. Every failure mode this pass can hit (an unreachable
endpoint, an unresolvable chat template, an action_labels literal absent
from the logged prompt, an action missing from action_labels' key set, an
echoed token-id sequence whose length does not match what was submitted, an
action whose tokenisation is not the canonical continuation of the prefix, a
prompt that would exceed ``max_model_len``, a renormalised distribution
whose coverage mass falls below the trust floor) is caught, counted, and
emitted as an explicit failure row. A distinct info-set never silently
vanishes from the output as though it had been measured;
``len(scores) == n_legal_actions`` holds for every info-set this pass
touches, success or failure.

Submitting token ids makes the prefix/action boundary exact by construction,
so no check for a token straddling that boundary is needed; the
canonical-continuation condition (``assert_canonical_continuation``) takes
its place.

Where the action-text vocabulary comes from: this pass calls
``GamePromptFns.action_labels(info_set)`` (``src/lmopt/prompting``) and
never parses prompt text to derive candidate literals. ``action_labels`` is
data the prompting layer already computes from the same source
``action_suffix`` renders from (e.g. darkhex's own ``action_label`` dict);
a standing full-tree invariant in ``tests/test_prompt_fidelity.py``
(``test_action_labels_agree_with_action_suffix_and_legal_actions``) checks,
over the whole tree of every tractable registered variant, that every label
appears verbatim in ``action_suffix``'s rendered text and that the label key
set equals ``legal_actions()``. This pass then uses the logged ``prompt`` as
an oracle, not a source: each label must appear verbatim (plain substring
containment, no regex) in the row's own logged prompt, or
``PromptVocabularyUnparseable`` is raised. This is what makes a label/prompt
drift fail loudly against earlier data, the after-the-fact invalidation
guarantee hard requirement 1 names. A regex over prompt text reproduces the
same defect class one game later, which is why none is used.

Known data caveat for arms collected before the Leduc card-name fix:
``src/lmopt/prompting/leduc.py``'s ``CARD_NAMES`` mapped only 3 of the 6
chance-outcome indices OpenSpiel's ``leduc_poker`` actually deals, so 52.1%
of logged Leduc rows (29,908/57,408) in such an arm show "Unknown" for the
private card. On that arm, of Leduc's 135 distinct info-sets, 31 groups of
info-sets share byte-identical ``prompt`` text, covering 105 of the 135
info-sets in total (135 info-sets collapse to 61 distinct prompts). This
pass reads ``prompt`` verbatim (hard requirement 1) and so faithfully
reproduces "Unknown" and the resulting info-set collisions rather than
silently working around them. Read any Leduc exact-policy number from such
an arm with that in mind.

Exit codes (do not add a code without adding it here too):
  0 - every distinct info-set this run processed was scored with no
      failures (``summary["n_failures"] == 0``).
  1 - at least one (info-set, action) failure occurred (any ``ErrorKind``),
      but the run otherwise completed and, if ``--out`` was given, wrote a
      complete CSV covering every processed info-set (failures included as
      rows, never dropped); see ``main``'s final ``return``.

      The three canonical-continuation kinds all land here, never as a
      distinct exit code: ``ErrorKind.NONCANONICAL_PREFIX``,
      ``ErrorKind.NONCANONICAL_ACTION`` and ``ErrorKind.NONCANONICAL_SUFFIX``.
      The last can only fire on a template whose measured generation-prompt
      suffix (``gen_suffix``, spec's ``G``) is non-empty; with ``gen_suffix``
      empty, ``assert_canonical_continuation`` never raises it.

      The bare ``ValueError``s ``distinct_p0_info_sets`` raises are caught in
      ``main`` and mapped to 8 (below), so 1 means only the "completed with
      per-row failures" case.

      Known ambiguity: 1 also fires on a healthy arm.
      ``coverage_below_floor`` rows count toward ``summary["n_failures"]``,
      so an arm whose only failures are below-floor readings returns 1, and
      ``scripts/run_experiment.py`` maps any non-zero rc to "exact-policy
      pass FAILED" (its exit 3), even though ``COVERAGE_BELOW_FLOOR`` is an
      expected outcome on a healthy arm (OLMo carries 3 of them). A possible
      fix, not applied: return 0, or a distinct code, when every failure is
      ``COVERAGE_BELOW_FLOOR``.
  2 - not returned by this script. argparse's own usage-error exit (a
      missing ``--model``, neither ``--data-dir`` nor ``--parquet``) is 2,
      so a 2 a caller sees came from ``argparse``. The no-parquet case
      returns 9 (below) to keep the two apart.
  3 - deliberately not used by this script. Reserved by the orchestrator:
      ``scripts/run_experiment.py``'s ``EXACT_POLICY_FAILURE_EXIT_CODE = 3``
      is a different thing at a different layer: it is what the
      orchestrator itself returns, after running this script as a
      subprocess, to mean "every arm's sweep+diagnose completed but at
      least one arm's exact-policy pass failed" (see that script's own
      Exit codes section). If this script also returned 3 for one of its
      own checks, a reader of the outer orchestrator's exit code could not
      tell whether the inner pass returned 3 directly or the orchestrator
      synthesised it, so 3 is reserved here permanently.
  4 - ``ArmProvenanceMismatch``: the live endpoint about to be queried is
      not the one that collected this arm (``verify_arm_provenance``).
  5 - ``ShardCountMismatch``: this run did not process the whole arm
      (``verify_cardinality``, ``verify_full_population_processed``, or
      ``verify_row_count_completeness``; see ``ShardCountMismatch``'s own
      docstring for which of the three raised).
  6 - ``DetectorCountMismatch``: ``run_pass``'s own summary counters for
      the canonical-continuation detector disagree with the ``error_kind``
      column of the rows about to be written (``verify_detector_counts_
      consistent``); this pass miscounted its own findings.
  7 - the generation-prompt suffix could not be established (probes
      disagreed, or a probe's reference was not a canonical continuation),
      or it was established but disagreed with
      --expect-generation-prompt-suffix. Nothing was scored and no CSV was
      written. This is a template-level failure, distinct from the
      per-(info-set, action) NONCANONICAL_* kinds, which are rows in the CSV.
  8 - ``distinct_p0_info_sets`` raised a ``ValueError``: a logged LM row had
      a null ``lm_seat``, or its ``player_to_act`` disagreed with its
      ``lm_seat``, or an info-set carried more than one distinct
      ``legal_mask`` across logged rows (see that function's own docstring
      for all three conditions). Any of them means the arm's own logged data
      violates an invariant this pass's P0 framing, or its per-info-set
      deduplication, rests on: not a scoring failure, a data problem.
      Reachable from two call sites that both read the same paths (``main``
      catches it at both): ``_generation_prompt_suffix_probes`` (before
      ``run_pass``, non-``--dry-run`` only) and ``run_pass`` itself
      (unconditional, including under ``--dry-run``). Nothing was scored
      and no CSV was written; the stderr line names which condition and the
      offending rows, with no traceback.
  9 - no parquet files found (``--data-dir``/``--parquet`` resolved to
      nothing); nothing ran.

  10 - the arm was collected above memory level M0 (``verify_memory_level`` on the run meta,
      or an LM row whose ``memory_rung`` is not M0). Nothing was scored and no CSV was
      written; the stderr line names the level. Above M0 one info-set has many prompts and
      keep-first would present an arbitrary one as its policy. ``run_experiment.py`` reads any
      non-zero pass status as "the pass failed" and does not tell this one apart.
  11 - the arm was collected under a prompt version other than the live one (``verify_prompt_epoch``
      on the run meta's ``prompt_spec_hash``). The pass re-renders prompts with the live code, so on
      such an arm it would score prompts the model never saw. Nothing was scored and no CSV was
      written; the stderr line names both hashes.

Usage:
    uv run python scripts/exact_policy_pass.py \\
        --data-dir data/run_experiment/qwen2.5-14b-instruct \\
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-14B-Instruct \\
        --dry-run

    uv run python scripts/exact_policy_pass.py \\
        --data-dir data/run_experiment/qwen2.5-14b-instruct \\
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-14B-Instruct \\
        --limit 5 --out /tmp/exact_policy_smoke.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import enum
import glob
import hashlib
import json
import math
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

import polars as pl

# Reads only get_game_fns_for_game (dispatch) and load_game (game
# reconstruction from the logged game/game_params columns, for dispatch
# only). This module never calls `.build_prompt` on the returned
# GamePromptFns, and never parses prompt text with a regex:
# `action_texts_for_info_set` below calls `.action_labels(info_set)`, which
# is data the prompting layer computes from the same source `action_suffix`
# renders from, and checks each label's verbatim presence in the logged
# `prompt` (plain substring containment) as an oracle, never a source.
# (Hard requirement 1.)
from lmopt.agents.lm_vllm import PROMPT_SCHEMA_VERSION, PROMPT_SPEC_HASHES
from lmopt.games.loaders import load_game
from lmopt.games.families import game_family
from lmopt.prompting import get_game_fns_for_game
from lmopt.prompting.probe_variants import (
    NON_PROMPT_PROBE_TYPES,
    expected_variant_name,
    require_variant_spec,
)

# Name trap: src/lmopt/serving/chat_template.py
# defines its own `ChatTemplateUnavailable`, a different class from the one
# this module defines below. Importing it under its own name would silently
# shadow the local class and change which exceptions this module's
# `except ChatTemplateUnavailable` clauses catch. Aliased on import; every
# existing `except ChatTemplateUnavailable` in this file stays bound to the
# Local class, never this one.
from lmopt.serving.chat_template import (
    ChatTemplateUnavailable as ServingChatTemplateUnavailable,
    resolve_chat_template_hash,
)

# Cross-checkout import tripwire. Hard failure, at import: this pass writes `exact_policy.csv`, a scientific artefact that
# `scripts/analyse_stats.py` reads straight into the P3 property table. Its
# numbers depend on `lmopt.prompting` and `lmopt.serving.chat_template`, so a
# foreign `lmopt` silently changes the prompts the model is scored against. The
# same class of prefix error was worth a 0.996-vs-0.095 swing on kuhn, with no
# visible symptom. No silent fallbacks.
from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)


class ChatTemplateUnavailable(RuntimeError):
    """The served model's chat template could not be resolved from the endpoint.

    Raised rather than silently falling back to a hardcoded/naive template -
    that is the exact defect this pass exists to avoid: raw vs templated gave
    0.9969 vs 0.6792 on the same node. (Hard requirement 2.)
    """


class PromptVocabularyUnparseable(RuntimeError):
    """One or more of `action_labels(info_set)`'s literals is not present
    verbatim (substring) in this row's own logged `prompt`.

    It means the current prompting layer's `action_labels` disagrees with what this ROW's logged
    `prompt` actually contains - code/data drift between the prompting
    module version that produced this arm and the one running this pass.
    Raised rather than silently scoring against a label the model was never
    shown; this is the after-the-fact invalidation guarantee hard requirement 1
    names (a later prompt-format change must fail loudly against earlier
    logged data, not silently propagate wrong literals into it).
    """


class ArmProvenanceMismatch(RuntimeError):
    """The endpoint about to be queried is not configured the same way as the
    endpoint that COLLECTED this arm.

    Deliberately a NEW class, not a reuse of `ChatTemplateUnavailable` (either
    the local one above or `ServingChatTemplateUnavailable`): those two mean
    "the live endpoint could not be resolved at all"; this one means "the
    live endpoint answered, and what it answered disagrees with what this
    arm's own logged data says was used to collect it" - a materially
    different failure a caller needs to tell apart. See
    `verify_arm_provenance` for the checks that raise this.
    """


class ContextLengthExceeded(RuntimeError):
    """prefix + action text exceeds the server's max_model_len."""


class EchoedSequenceMismatch(RuntimeError):
    """The server echoed back a different number of token_logprobs than the
    number of token ids submitted in the prompt.

    Submitting token ids makes the prefix/action boundary exact by
    construction, so no character-offset boundary recovery exists, but a
    server that silently truncated or re-tokenised the submitted ids would
    then be scored as though it had not, with no boundary check left to catch
    it. This is the response-side anchor: it validates what was actually
    scored, not just what was requested.
    """


class NonCanonicalPrefix(RuntimeError):
    """The submitted ``prefix_ids`` are not equal to the SERVER's own
    canonical ids for the same (user turn) prefix, per
    ``canonical_assistant_ids``.

    An exact prefix/action boundary (ids, not text)
    does not imply the submitted sequence is the one the server would
    itself produce; this class isolates a prefix-side defect from an
    action-side one (``NonCanonicalActionTokenisation``) - see
    ``assert_canonical_continuation``.
    """


class NonCanonicalActionTokenisation(RuntimeError):
    """The submitted ``action_ids`` do not match the SERVER's own canonical
    continuation for this (prefix, action_text) pair, per
    ``canonical_assistant_ids`` - either a length mismatch against the
    canonical sequence (checked before slicing, so a template that appends a
    terminator despite ``continue_final_message`` cannot masquerade as a
    within-length action-tail mismatch) or a tail mismatch once lengths
    agree. Sibling to ``NonCanonicalPrefix``, see
    ``assert_canonical_continuation``.
    """


class NonCanonicalSuffix(RuntimeError):
    """The served prefix diverges from the canonical reference by something
    other than the measured generation-prompt suffix.

    Third sibling to ``NonCanonicalPrefix`` and
    ``NonCanonicalActionTokenisation``. It exists because some chat templates
    legitimately emit tokens in the generation prompt that the canonical
    assistant encoding does not: Gemma 4 opens and closes an empty thinking
    channel when thinking is off, which is its default. That divergence is
    real and is what the model is actually shown, so it must be permitted --
    but only at exactly the value measured from the endpoint at bring-up.
    Any other divergence is template drift and is a failure, never a warning.
    No silent fallbacks.
    """


class PromptEpochNotLive(ValueError):
    """The arm was collected under a prompt version other than the live one.

    The pass rebuilds the prompts it queries from the live prompt code, so an arm logged under
    another version would be scored on prompts its model never saw. ``main`` exits 11 for it.
    A ``ValueError`` so the library callers that already handle one see a refusal.
    """


class MemoryLevelNotM0(ValueError):
    """The arm was collected above memory level M0, which this pass refuses.

    Above M0 the logged prompt carries a per-hand memory prefix, so the policy at an
    info-set is a function of (info-set, memory state), not one distribution. The pass keeps
    one prompt per info-set and would present an arbitrary memory-conditioned prompt as the
    info-set's policy. ``main`` exits 10 for it, so it never shares a code with the
    shard-count and data-invariant refusals. A ``ValueError`` so the library callers that
    already refuse on bad arm data refuse on this too.
    """


class ShardCountMismatch(RuntimeError):
    """This run did not process the whole arm.

    Two independent things raise this, both surfaced identically as exit
    code 5 in ``main``:

    1. ``verify_cardinality``: ``len(paths)`` disagrees with the shard count
       ``run_meta.json``'s own ``resolved_config`` declares (``len(games) *
       len(opponents) * n_seeds``), or ``run_meta.json`` could not be found
       or parsed at all. Comparing the processed info-set count against
       ``distinct_p0_info_sets(paths)`` computed over the same ``paths`` is
       circular - a ``--data-dir`` resolving to a partial shard set produces
       a reference computed from that same partial read, so it always
       agrees with itself. ``run_meta.json``,
       written once at collection time from the resolved sweep config, is
       the only reference independent of what ``--data-dir``/``--parquet``
       happens to resolve today. No silent fallbacks: a missing or
       unparseable ``run_meta.json`` raises this too, rather than being read
       as "check not applicable". The most truncated real directory
       (``qwen2.5-14b-instruct.mw32-partial-8of100/``, 8 of 100 shards) is
       the only one of six with no ``run_meta.json`` at all, so skipping the
       check on absence would exempt exactly the worst case.
    2. ``verify_row_count_completeness``: the number of (info-set, action)
       rows this run actually produced disagrees with the sum of
       ``n_legal`` across the info-sets it processed - every branch of
       ``score_info_set``/``_all_failed_scores`` emits exactly one row per
       legal action, success or failure (this module's own docstring
       invariant), so this is a regression guard against a future change
       that silently breaks it for some subset of info-sets while leaving
       the total info-set count unchanged.
    """


class DetectorCountMismatch(RuntimeError):
    """``run_pass``'s own summary counters (``n_detector_prefix_failures``,
    ``n_detector_action_failures``) disagree with the number of rows about
    to be written that carry the corresponding ``error_kind``.

    Two independent representations of the same quantity exist:
    ``run_pass`` increments its counters inline, per action, while it is
    building ``results``; the acceptance gate instead recomputes the same quantity by
    reading ``error_kind`` off the output rows, because a gate that reads an
    artefact must read the artefact, not a number the producing process
    merely asserts about itself. Nothing forces those two computations to
    agree after a future edit touches one and not the other (e.g. an
    increment moved outside the branch that also sets ``error_kind``, or a
    future refactor keys the increment by info-set rather than by action).
    ``verify_detector_counts_consistent`` closes that gap: raised loudly,
    before the CSV is written, rather than shipping an artefact whose
    stdout summary and whose own rows already disagree about how many times
    each detector fired.
    """


# ---------------------------------------------------------------------------
# HTTP - raw JSON POST via the stdlib (matches scripts/run_experiment.py's
# urllib.request.urlopen preflight pattern; no new dependency). /tokenize is
# a vLLM server extension outside the openai-python schema, so a raw client
# is used uniformly for the endpoints this pass itself calls (/tokenize and
# /v1/completions). This pass does not call /detokenize; the one /detokenize
# call in the dependency chain lives in
# lmopt.serving.chat_template.resolve_chat_template_hash, provenance-only.
# ---------------------------------------------------------------------------


def _root_url(base_url: str) -> str:
    """Strip a trailing '/v1' so /tokenize (not under /v1, unlike
    /v1/completions) resolves correctly from the same config value
    (configs/run_experiment.yaml's vllm_base_url carries the '/v1' suffix).
    """
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[: -len("/v1")]
    return b


def _post_json(root_url: str, path: str, payload: dict, timeout: float) -> dict:
    url = root_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} -> HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc


def render_chat_prefix_ids(
    root_url: str,
    model: str,
    user_content: str,
    timeout: float = 60.0,
    chat_template_kwargs: dict | None = None,
) -> list[int]:
    """The served model's own chat template applied to `user_content`, as token IDS.

    `chat_template_kwargs`: when the collecting run served
    with non-default chat-template kwargs (e.g. Qwen3's
    `enable_thinking=False`), the /tokenize request must carry the same
    kwargs, or the rendered prefix this pass measures diverges from the one
    the model was actually shown -- exactly `assert_canonical_continuation`'s
    `NonCanonicalPrefix` failure mode, for a reason external to the model.
    `None` (the default) sends no `chat_template_kwargs` key at all;
    mirrors `src/lmopt/serving/chat_template.py`'s
    `resolve_chat_template_hash`, the established pattern for this exact
    kwarg on a /tokenize call.

    Returning ids rather than text avoids a /tokenize(messages) ->
    /detokenize(ids) round trip followed by server-side re-tokenisation.
    That round trip is LOSSY whenever the template's control tokens do not
    re-parse from text. Against Mistral-NeMo under `--tokenizer-mode
    mistral`: 28 ids in, `'<s>[INST]...[/INST]'` out, 36 ids back --
    `[INST]` (id 3, a control token) returns as the five literal characters,
    so every Mistral info-set would be scored against a prompt the model was
    never shown (coverage_mass median 0.067, all 481 rows refused).
    mistral-common deliberately does not re-parse control tokens out of user
    text. With ids, template round-trip fidelity is not load-bearing for any
    arm. (Hard requirement 2: the live endpoint's own template, never a
    hardcoded string and never local weights.)
    """
    try:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
        }
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = chat_template_kwargs
        tok = _post_json(root_url, "/tokenize", payload, timeout)
        token_ids = tok["tokens"]
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"/tokenize returned no tokens: {tok!r}")
        return token_ids
    except Exception as exc:
        raise ChatTemplateUnavailable(
            f"could not resolve the served model's chat template from {root_url} "
            f"for model {model!r}: {exc}"
        ) from exc


#: `tokenize_action`'s per-run cache, keyed by the full `(root_url, model, action_text)` triple.
_TOKENIZE_CACHE: dict[tuple[str, str, str], list[int]] = {}


def tokenize_action(
    root_url: str, model: str, action_text: str, timeout: float = 60.0
) -> list[int]:
    """Token ids for a bare action literal, with no special tokens.

    `add_special_tokens=False` is load-bearing, not a default worth trusting:
    `/tokenize {"prompt": "PASS"}` returns
    [1, 12118, 4290] on Mistral and only the explicit false gives
    [12118, 4290]. A BOS in the middle of the scored sequence would depress
    coverage_mass with no visible failure.

    Cached within a run (a module-level dict): the same action literal
    recurs across many info-sets, and each call is otherwise a network round
    trip. The key is the full argument triple `(root_url, model,
    action_text)`, not the literal alone. Keying on the literal alone is
    correct only while "one subprocess per arm" holds, an invariant enforced
    nowhere in this file: two models tokenise the same literal differently by
    construction, so an in-process second arm would silently read the first
    arm's ids.
    """
    key = (root_url, model, action_text)
    if key in _TOKENIZE_CACHE:
        return _TOKENIZE_CACHE[key]
    tok = _post_json(
        root_url,
        "/tokenize",
        {"model": model, "prompt": action_text, "add_special_tokens": False},
        timeout,
    )
    ids = tok["tokens"]
    if not isinstance(ids, list) or not ids:
        raise RuntimeError(
            f"empty token id list for action text {action_text!r}: {tok!r}"
        )
    _TOKENIZE_CACHE[key] = ids
    return ids


# ---------------------------------------------------------------------------
# Action-text resolution.
#
# Candidate literals are never derived from a hand-written per-game scheme
# (e.g. darkhex "{r},{c}" coordinate pairs) validated by round-tripping
# through parse_action: parse_action accepts multiple surface forms per
# action, so a hand-written candidate that round-trips is necessary but not
# Sufficient. Nor are they regex-parsed out of the logged prompt text: a
# liarsdice regex of that kind greedily bleeds the next prompt line into a
# bid literal ("Or respond: CHALLENGE" as a "bid") when the legal-bid list is
# empty. Regex text-matching a *rendering* of data is inherently fragile
# against the renderer's own formatting choices.
#
# This file does no text-parsing.
# `action_labels(info_set)` (src/lmopt/prompting) is data the prompting
# layer already computes from the exact same source `action_suffix` renders
# from (e.g. darkhex's own `action_label` dict) - never re-derived here, and
# never regex-matched out of rendered text. A standing full-tree invariant
# (tests/test_prompt_fidelity.py) checks, over the whole tree of every
# tractable registered variant, that every label appears verbatim in
# action_suffix's own rendered text and the label key set equals
# legal_actions() - so correctness is enforced once, in the prompting layer,
# not re-derived and re-risked here. This module's only
# responsibility is treating the logged prompt as an oracle: confirming
# (the rendered action_suffix must occur in it, and each label must be a whole token of that suffix) that each label the current code
# claims actually appears in what was actually logged, so a future
# prompting-layer change fails loudly against earlier data instead of
# silently mismatching it. `.build_prompt` is never referenced anywhere in
# this module.
# ---------------------------------------------------------------------------


def prompt_spec_for_arm(game: str, probe_type: str):
    """
    the prompt content the arm rendered for one game, so the candidate action literals come from the
    same vocabulary the logged prompt was written in
    :param game: the game registry key as logged (str)
    :param probe_type: the arm's logged probe type (str)
    :returns: None for an arm whose prompts are canonical, else the variant spec for this game's
        family; raises when the family has no variant for that probe type (GamePromptSpec or None)
    """
    if probe_type in NON_PROMPT_PROBE_TYPES:
        return None
    return require_variant_spec(game_family(game), probe_type)


def action_texts_for_info_set(fns, info_set: str, prompt: str) -> dict[int, str]:
    """
    legal action -> exact literal to append, for every legal action at this info-set, sourced
    from `fns.action_labels` and never parsed from `prompt`. The logged `prompt` is only an
    oracle, and the oracle is read where the options are rendered: the arm's own
    `action_suffix` for this info-set must occur verbatim in the prompt, and every label must
    occur in that suffix as a whole token. A label found elsewhere in the prompt (the rules
    text, a remaining-cards list, the word opponent around one) proves nothing about the
    options section, so it is never searched.
    :param fns: the arm's prompt functions, built from the arm's own variant spec (GamePromptFns)
    :param info_set: the decision node's information-state string (str)
    :param prompt: the prompt as logged for this node (str)
    :returns: action -> label literal (dict[int, str])
    :raises PromptVocabularyUnparseable: when the rendered suffix is not in the logged prompt,
        or a label is not a whole token of that suffix
    """
    labels = fns.action_labels(info_set)
    suffix = fns.action_suffix(info_set)
    if suffix not in prompt:
        raise PromptVocabularyUnparseable(
            f"the options section action_suffix(info_set={info_set!r}) renders is not in the "
            f"logged prompt, so its labels {list(labels.values())!r} cannot be confirmed"
        )
    missing = [
        lab
        for lab in labels.values()
        if not re.search(rf"(?<!\w){re.escape(lab)}(?!\w)", suffix)
    ]
    if missing:
        raise PromptVocabularyUnparseable(
            f"label(s) {missing!r} from action_labels(info_set={info_set!r}) = "
            f"{labels!r} are not whole tokens of the rendered action_suffix"
        )
    return labels


# ---------------------------------------------------------------------------
# Per-action scoring. No character-offset boundary detection is needed:
# submitting token ids makes the prefix/action boundary exact by
# construction. See score_action_logprob's docstring for the checks that
# guard the scored sequence instead.
# ---------------------------------------------------------------------------


def canonical_assistant_ids(
    root_url: str,
    model: str,
    user_content: str,
    action_text: str,
    timeout: float = 60.0,
    chat_template_kwargs: dict | None = None,
) -> list[int]:
    """The server's own canonical encoding of (user turn, assistant = action_text).

    ``continue_final_message=True`` with ``add_generation_prompt=False`` asks the
    server to encode the assistant turn natively, so this is the reference the
    submitted ``prefix_ids ++ action_ids`` must equal. Verified live on
    Mistral: PASS -> [12118, 4290], BET -> [1066, 4321],
    CHALLENGE -> [7247, 4286, 3561, 16705, 1069], 1-4 -> [1049, 1045, 1052],
    all matching the concatenation exactly.

    `chat_template_kwargs`: threaded identically to
    `render_chat_prefix_ids` -- the canonical reference and the served
    prefix must be rendered with the same kwargs, or they compare a
    non-think canonical encoding against a think-mode prefix (or vice
    versa) instead of measuring the real prefix/action boundary.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": action_text},
        ],
        "continue_final_message": True,
        "add_generation_prompt": False,
    }
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    tok = _post_json(root_url, "/tokenize", payload, timeout)
    ids = tok["tokens"]
    if not isinstance(ids, list) or not ids:
        raise RuntimeError(
            f"/tokenize(continue_final_message) returned no tokens: {tok!r}"
        )
    return ids


class GenerationPromptSuffixUnstable(RuntimeError):
    """Two probes disagreed on the generation-prompt suffix.

    The suffix must be a property of the template, not of the prompt. If it
    varies with the user turn or the action literal, it cannot be factored
    out of the continuation check (``assert_canonical_continuation``'s
    ``gen_suffix``) and the arm must not be scored.
    """


class GenerationPromptSuffixProbesUnavailable(RuntimeError):
    """No probe could be built for ``measure_generation_prompt_suffix`` from
    any distinct game in this run's resolved info-sets.

    Distinct from ``GenerationPromptSuffixUnstable`` (probes disagreed with
    each other) and from ``NonCanonicalPrefix`` / ``NonCanonicalAction
    Tokenisation`` (a specific submitted probe's own reference was not
    canonical): this fires before any probe is even submitted to
    ``measure_generation_prompt_suffix``, when every candidate row
    ``_generation_prompt_suffix_probes`` tried failed for its own reason --
    a game-load failure, the prompt-vocabulary oracle rejecting the row, or
    the logged ``parsed_action`` missing from that info-set's own
    ``action_labels``. These are exactly the three failure classes
    ``score_info_set`` already turns into a diagnosable CSV row
    (``GAME_LOAD_FAILED`` / ``PROMPT_VOCABULARY_UNPARSEABLE`` /
    ``ACTION_NOT_IN_ACTION_LABELS`` respectively); a probe candidate hitting
    one of them is not a reason to crash the whole run before scoring even
    starts, only a reason to try a different game. Caught by ``main``
    alongside the three exceptions above and mapped to exit 7 -- never an
    uncaught traceback (no silent fallbacks means loud and typed, not a
    crash).
    """


def measure_generation_prompt_suffix(
    root_url: str,
    model: str,
    probes: Sequence[tuple[str, str]],
    timeout: float = 60.0,
    chat_template_kwargs: dict | None = None,
) -> list[int]:
    """The ids the served generation prompt carries beyond the canonical head
    (spec's ``G``): what ``assert_canonical_continuation``'s ``gen_suffix``
    argument is measured from, once per arm at bring-up, never hardcoded.

    Defined against the reference (``canonical_assistant_ids``), not as the
    difference between the two ``add_generation_prompt`` renders. Those
    differ additionally by the assistant turn opener (``<|turn>model\\n`` on
    Gemma 4), which the ``continue_final_message`` reference does emit, so
    the naive difference over-measures and would mask a real prefix defect.
    Per probe: the canonical reference must end with the action's own ids
    (``NonCanonicalActionTokenisation`` if not -- there is then no principled
    boundary to slice ``head`` at), and the served prefix must open with that
    canonical ``head`` (``NonCanonicalPrefix`` if not -- the served template
    has already diverged before the point this function is trying to
    measure past). Only once both hold is ``prefix_ids[len(head):]`` a
    reading of the suffix at all.

    Measured across several probes because deriving the suffix from one
    probe and then checking it on that same probe is an algebraic identity,
    true by construction, that cannot fail. Only agreement across probes
    from different games and different action literals is evidence that the
    suffix is a property of the template rather than of one particular
    prompt or literal; that cross-probe argument holds for any N >= 2 such
    probes and does not itself require a fixed count.

    Two different numbers must not be read as one: the bring-up measurement
    against a live endpoint used four hand-picked probes across
    four games (kuhn, liarsdice, goofspiel, leduc), with action literals of
    1, 3, 1 and 2 token ids; the selector this module uses,
    ``_generation_prompt_suffix_probes``, caps at three (one per distinct
    game, fewer is explicitly not an error -- see its own docstring). A run
    through that selector therefore probes at most three games; the
    invariance argument above still holds at three, it is simply narrower
    evidence than the four-probe bring-up measurement.
    No silent fallbacks: any disagreement across probes raises
    ``GenerationPromptSuffixUnstable`` rather than picking a probe, averaging,
    or falling back to the first measurement.
    """
    if not probes:
        raise ValueError("measure_generation_prompt_suffix needs at least one probe")
    # The label carries the probe's index and its whole (user_content,
    # action_text) pair, not just action_text. Action literals recur across
    # games by construction ("PASS" in both Kuhn and Leduc); a bare-literal
    # label collapses two different probes to the same displayed name in the
    # disagreement message below, e.g. "probe 'PASS' measured [...] but probe
    # 'PASS' measured [...]", which names nothing a caller could act on.
    # The index alone would disambiguate but
    # not explain; carrying the pair too means the message states exactly
    # which game/prompt and which literal diverged.
    measured: list[tuple[tuple[int, str, str], list[int]]] = []
    for i, (user_content, action_text) in enumerate(probes):
        label = (i, user_content, action_text)
        prefix_ids = render_chat_prefix_ids(
            root_url, model, user_content, timeout, chat_template_kwargs
        )
        action_ids = tokenize_action(root_url, model, action_text, timeout)
        canonical = canonical_assistant_ids(
            root_url, model, user_content, action_text, timeout, chat_template_kwargs
        )
        if len(canonical) <= len(action_ids) or canonical[
            len(canonical) - len(action_ids) :
        ] != list(action_ids):
            raise NonCanonicalActionTokenisation(
                f"probe {label!r}: canonical reference does not end with the "
                f"action ids {list(action_ids)!r}; cannot derive a suffix from it"
            )
        head = canonical[: len(canonical) - len(action_ids)]
        if len(prefix_ids) < len(head) or prefix_ids[: len(head)] != head:
            raise NonCanonicalPrefix(
                f"probe {label!r}: served prefix does not open with the "
                f"canonical head; cannot derive a suffix from it"
            )
        measured.append((label, prefix_ids[len(head) :]))
    first_label, first = measured[0]
    for label, value in measured[1:]:
        if value != first:
            raise GenerationPromptSuffixUnstable(
                f"probe {label!r} measured suffix {value!r} but probe "
                f"{first_label!r} measured {first!r}; the suffix is not a "
                "template property and this arm cannot be scored"
            )
    return first


def assert_canonical_continuation(
    canonical: list[int],
    prefix_ids: list[int],
    action_ids: list[int],
    gen_suffix: Sequence[int] = (),
) -> None:
    """The scored sequence must be the canonical one, decomposed into three
    separately diagnosable classes, checked prefix first.

    ``gen_suffix``: the ids, if any, the served generation prompt carries
    beyond the canonical head -- ``measure_generation_prompt_suffix``'s
    return value (spec's ``G``), threaded through by the caller. Defaults to
    ``()``, the empty sequence, which means "this template's served
    generation prompt carries nothing beyond the canonical head" (every
    non-Gemma-4 arm). With ``gen_suffix`` empty the suffix check is skipped
    entirely -- see the "SUBSTITUTION, not a reordering" note at the top of
    the implementation below. A non-empty value is permitted only at exactly the value measured live
    from the endpoint at bring-up; any other served suffix is
    ``NonCanonicalSuffix``, never silently accepted or widened.

    Submitting ids makes the prefix/action boundary exact by construction,
    which does not make the sequence canonical: without this check a
    non-canonical sequence shows up only as a silently low coverage_mass,
    i.e. it would be published as model hedging. The failure
    needs the prefix's final token to be able to absorb the literal's first
    character: no current template can (all end in a control token or "\\n"),
    but a template ending "Answer: " would give joint `":" " PASS"` against
    concat `":" "PASS"`, measured at 0.858591 vs 0.000000.

    Why prefix first: an ordering that checks total length first sends any
    corruption that changes the prefix's own token count (such as a lossy
    /tokenize->/detokenize->/tokenize round trip) down the length branch,
    misclassifying it ``NonCanonicalActionTokenisation`` even though the
    defect is entirely prefix-side. On the full 481-row Mistral arm,
    reproducing that round-trip defect under length-first ordering gave
    ``n_detector_prefix_failures=0``, ``n_detector_action_failures=481``:
    the two counters failed to discriminate for the precise case the
    detector exists to catch. The prefix region is therefore judged first,
    on its own terms (length shortfall or content mismatch), before any
    total-length or tail comparison runs, so a prefix-side defect is
    attributed to the prefix regardless of whether it also changes the total
    length.

    What the three classes split on: the split is by region of the id sequence, not by cause.
    A boundary-absorption defect is action-side in cause but shows up as a
    differing prefix region, so it lands in ``NonCanonicalPrefix``; a stray
    generation-prompt token is prefix-side in cause but can leave the prefix
    region intact and shift the tail, landing in
    ``NonCanonicalActionTokenisation``; a served generation-prompt suffix
    that diverges from the value measured at bring-up lands in
    ``NonCanonicalSuffix`` (the suffix region, checked first, below). Neither
    of the first two causes is decidable from ids alone, so the three
    classes must not be read as a diagnosis.

    A consumer that counts only prefix and action rows passes a
    ``NonCanonicalSuffix`` row uncounted; suffix rows are visible through the
    ``error_kind`` column.
    """
    gen_suffix = list(gen_suffix)
    # Suffix region, checked first because it defines the conversation head
    # the two regions below are judged against. Some chat templates emit
    # tokens in the generation prompt that the canonical assistant encoding
    # does not (Gemma 4 opens and closes an empty thinking channel when
    # thinking is off, its default). That divergence is real, is what the
    # model is actually shown, and is permitted at exactly the value measured
    # from the endpoint at bring-up -- and at no other value.
    #
    # This is a substitution into the checks below, not a reordering of them:
    # with gen_suffix empty, `head` is `prefix_ids` and everything downstream
    # runs unchanged, including the prefix-first ordering documented above.
    # Deriving the head from the canonical length instead, and judging the
    # action region first, reintroduces the exact
    # misclassification this function's docstring describes.
    if gen_suffix:
        # No separate `len(prefix_ids) < len(gen_suffix)` disjunct:
        # `prefix_ids[-len(gen_suffix):]` clamps to at
        # most `len(prefix_ids)` elements, so on a prefix shorter than the
        # declared suffix the slice is strictly shorter than `gen_suffix`
        # and can never equal it on shape alone -- the content check below
        # already covers this case, and unlike the neighbouring redundant
        # guard at `canonical[: len(head)]` below (which stays: Python
        # slicing past a list's end truncates silently rather than raising,
        # so that comparison could in principle coincide by shape), there is
        # no analogous silent-coincidence risk here to guard against.
        if list(prefix_ids[-len(gen_suffix) :]) != gen_suffix:
            served_tail = list(prefix_ids[-len(gen_suffix) :])
            raise NonCanonicalSuffix(
                f"served prefix does not end with the measured generation-prompt "
                f"suffix {gen_suffix!r}; its last {len(served_tail)} of "
                f"{len(prefix_ids)} served ids are {served_tail!r}. The served "
                f"chat template has drifted from the one measured at bring-up."
            )
        head = list(prefix_ids[: len(prefix_ids) - len(gen_suffix)])
    else:
        head = list(prefix_ids)

    if len(canonical) < len(head) or canonical[: len(head)] != list(head):
        # The length guard is load-bearing on its own, not just belt-and-
        # braces: Python slicing past the end of a list truncates silently
        # rather than raising, so `canonical[:len(head)]` when
        # `canonical` is shorter than `head` yields a short list that
        # can never equal `head` on content grounds alone by
        # coincidence of shape -- but relying on that coincidence would
        # leave a canonical-shorter-than-prefix case undiagnosed as exactly
        # what it is (a prefix mismatch) if a future edit ever compared
        # lengths differently. Checked explicitly instead.
        raise NonCanonicalPrefix(
            f"canonical prefix {canonical[: len(head)]!r} != submitted {head!r}"
            + (
                ""
                if len(canonical) >= len(head)
                else f" (canonical has only {len(canonical)} ids, shorter than "
                f"the {len(head)}-id submitted prefix)"
            )
        )
    # The prefix is now known good (right length, right content). A total
    # length mismatch from here on is genuinely action-side.
    if len(canonical) != len(head) + len(action_ids):
        raise NonCanonicalActionTokenisation(
            f"canonical length {len(canonical)} != prefix {len(head)} + "
            f"action {len(action_ids)} (length mismatch, prefix already verified canonical)"
        )
    if canonical[len(head) :] != list(action_ids):
        raise NonCanonicalActionTokenisation(
            f"canonical action tail {canonical[len(head):]!r} != submitted {action_ids!r}"
        )


def score_action_logprob(
    root_url: str,
    model: str,
    prefix_ids: list[int],
    action_ids: list[int],
    timeout: float = 60.0,
) -> tuple[float, int]:
    """log P(action_ids | prefix_ids) via a token-ID echo request.

    The action span is the trailing len(action_ids) logprobs: submitting ids
    makes the prefix/action boundary exact by construction, so there is no
    character offset to recover. A non-canonical tokenisation is not thereby
    impossible, only invisible, so the canonical-continuation detector guards
    against it (see `canonical_assistant_ids`).
    """
    ids = list(prefix_ids) + list(action_ids)
    try:
        resp = _post_json(
            root_url,
            "/v1/completions",
            {
                "model": model,
                "prompt": ids,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
            },
            timeout,
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if (
            "context length" in msg
            or "max_model_len" in msg
            or "maximum context" in msg
        ):
            raise ContextLengthExceeded(str(exc)) from exc
        raise

    lp = resp["choices"][0]["logprobs"]
    token_logprobs: list[Optional[float]] = lp["token_logprobs"]
    if len(token_logprobs) != len(ids):
        raise EchoedSequenceMismatch(
            f"echoed {len(token_logprobs)} logprobs for {len(ids)} submitted token ids; "
            "the server did not score the sequence that was sent"
        )
    span: list[float] = []
    for i, v in enumerate(token_logprobs[len(prefix_ids) :]):
        if v is None:
            raise RuntimeError(
                f"None token_logprob inside the action span (index {len(prefix_ids) + i}); "
                "only index 0 of the whole prompt should ever be None"
            )
        span.append(v)
    return sum(span), len(action_ids)


# ---------------------------------------------------------------------------
# Distinct (game, game_params, info_set) extraction - hard requirement 3.
# ---------------------------------------------------------------------------


def distinct_p0_info_sets(parquet_paths: list[str]) -> pl.DataFrame:
    """One row per distinct (game, game_params, lm_seat, info_set), prompt read verbatim.

    Seat-generalised: reads `lm_seat` and filters to the LM's own
    decision nodes (`player_to_act == lm_seat`), so this pass can score arms
    collected under `seat_mode: random`. The scorers' seat generalisation
    (actor_rows) is the pattern.

    Reads only the columns needed (never reconstructs `prompt`). Filters to
    rows where `prompt` is not null (LM rows), then to rows where the LM is
    the acting player. Raises loudly, rather than silently filtering or
    trusting the first occurrence, if:
      - any LM row has a NULL `lm_seat` (checked first, and separately, because
        `null != x` is null in polars, so such a row would slip through the
        mismatch check below rather than fail it), or
      - any LM row has `player_to_act` disagree with `lm_seat` (the LM's own
        logged decision nodes must be the ones this pass scores; a mismatch
        means the data does not match the arm's own seat assignment), or
      - two rows sharing a (game, game_params, lm_seat, info_set) key carry
        different legal_mask values (the info-set string would then not
        determine the legal action set, breaking per-info-set deduplication's
        correctness), or
      - the rows do not carry exactly one (probe_type, prompt_variant) pair that
        agrees with the variant registry. The returned frame carries both columns
        so the scorer builds its action literals from the variant the arm rendered.
        A parquet with neither column is a canonical arm.

    Also computes, from the full (non-deduplicated) set of logged rows for
    each (game, game_params, lm_seat, info_set) key before collapsing to one
    representative row - this is the cross-check against logged behaviour:
      - ``logged_n_rows``: how many of the model's own logged decision nodes
        share this key.
      - ``logged_n_fallback_rows``: how many more nodes share it whose action
        the harness substituted (``fallback_used``). They are not the model's
        action, so they are in neither the modal action nor ``logged_n_rows``.
      - ``logged_modal_action``: the most frequent ``parsed_action`` among
        the model's own rows (ties broken toward the lower action index,
        deterministically); None when every row for the key is a fallback.
      - ``logged_modal_action_count``: how many of ``logged_n_rows`` took
        that action.
    """
    frames = []
    for p in parquet_paths:
        columns = [
            "game",
            "game_params",
            "info_set",
            "player_to_act",
            "lm_seat",
            "prompt",
            "legal_mask",
            "parsed_action",
        ]
        schema = pl.read_parquet_schema(p)
        has_fallback = "fallback_used" in schema
        has_probe = "probe_type" in schema
        has_variant = has_probe and "prompt_variant" in schema
        has_rung = "memory_rung" in schema
        df = pl.read_parquet(
            p,
            columns=columns
            + (["fallback_used"] if has_fallback else [])
            + (["probe_type"] if has_probe else [])
            + (["prompt_variant"] if has_variant else [])
            + (["memory_rung"] if has_rung else []),
        )
        if has_rung:
            # A column absent from the shard reads as M0 (the reader-side default for rows
            # written before the column existed); a recorded value other than exactly "M0"
            # is refused, on the rows keep-first would otherwise choose from.
            rungs = df.filter(pl.col("prompt").is_not_null())["memory_rung"].unique()
            above = sorted(
                str(r) for r in rungs.to_list() if r is not None and r != "M0"
            )
            if above:
                raise MemoryLevelNotM0(
                    f"{p} holds LM rows logged at memory level {above}; the pass refuses "
                    f"every arm above M0, where one info-set has many prompts"
                )
            df = df.drop("memory_rung")
        if not has_fallback:
            df = df.with_columns(pl.lit(False).alias("fallback_used"))
        if not has_probe:
            df = df.with_columns(pl.lit("none").alias("probe_type"))
        if not has_variant:
            # a schema that predates prompt_variant can only vouch for canonical prompts
            prompt_probes = (
                set(df["probe_type"].unique().to_list()) - NON_PROMPT_PROBE_TYPES
            )
            if prompt_probes:
                raise ValueError(
                    f"{p} logs probe_type {sorted(map(str, prompt_probes))} but no "
                    f"prompt_variant column, so the variant that rendered its prompts "
                    f"cannot be confirmed"
                )
            df = df.with_columns(pl.lit("base").alias("prompt_variant"))
        frames.append(df.filter(pl.col("prompt").is_not_null()))
    if not frames:
        raise ValueError("no parquet files given")
    df = pl.concat(frames)
    if df.height == 0:
        return df.unique(
            subset=["game", "game_params", "lm_seat", "info_set"]
        ).with_columns(
            pl.lit(None, dtype=pl.Int64).alias("logged_n_rows"),
            pl.lit(None, dtype=pl.Int64).alias("logged_n_fallback_rows"),
            pl.lit(None, dtype=pl.Int64).alias("logged_modal_action"),
            pl.lit(None, dtype=pl.Int64).alias("logged_modal_action_count"),
        )

    # The arm's prompt content decides which action literals the logged prompt
    # can contain, so it must be one known value for every logged row: a null,
    # a pair that disagrees with the registry, or two arms in one call would
    # each score some keys against the wrong vocabulary.
    pairs = df.select("probe_type", "prompt_variant").unique()
    if pairs.height != 1:
        raise ValueError(
            f"the logged rows carry {pairs.height} distinct (probe_type, prompt_variant) "
            f"pairs, expected exactly one per arm: {pairs.to_dicts()}"
        )
    arm_probe, arm_variant = pairs.row(0)
    if arm_probe is None or arm_variant is None:
        raise ValueError(
            f"the logged rows have a null probe_type or prompt_variant ({pairs.to_dicts()}); "
            f"the action literals cannot be chosen without them"
        )
    if arm_variant != expected_variant_name(arm_probe):
        raise ValueError(
            f"probe_type {arm_probe!r} must log prompt_variant "
            f"{expected_variant_name(arm_probe)!r}, found {arm_variant!r}"
        )

    # A null `lm_seat` must be refused before the mismatch check, not left to
    # it: in polars `null != anything` evaluates to null, not
    # true, so `filter` drops those rows and the mismatch guard below is
    # Vacuous for exactly the rows whose seat is unknown. Those rows then flow
    # into the (game, game_params, lm_seat, info_set) key with a null seat
    # component, which is the one value that cannot be checked against
    # `player_to_act` at all. An arm that logged no seat is a data problem of
    # the same class as a seat that disagrees, so it gets the same loud refusal
    # rather than a silent pass (no silent fallbacks).
    null_seat = df.filter(pl.col("lm_seat").is_null())
    if null_seat.height > 0:
        raise ValueError(
            f"{null_seat.height} logged LM row(s) have a null lm_seat - this pass "
            f"scores the LM's own decision nodes and cannot tell which those are "
            f"without the seat; note that a null seat also silently passes the "
            f"player_to_act != lm_seat check below, since null != x is null in "
            f"polars (examples: {null_seat.head(3).to_dicts()})"
        )

    # Seat-generalised: filter to the LM's own decision nodes (player_to_act == lm_seat),
    # and refuse any mismatch rather than silently dropping rows.
    mismatched_seat = df.filter(pl.col("player_to_act") != pl.col("lm_seat"))
    if mismatched_seat.height > 0:
        raise ValueError(
            f"{mismatched_seat.height} logged LM row(s) have player_to_act != lm_seat - "
            f"this pass scores the LM's own decision nodes, and a mismatch means the data "
            f"does not match the arm's seat assignment (examples: {mismatched_seat.head(3).to_dicts()})"
        )

    keyed = df.with_columns(
        pl.col("legal_mask")
        .cast(pl.List(pl.Utf8))
        .list.join(",")
        .alias("_legal_mask_key")
    )
    # Per-seat framing: the key is (game, game_params, lm_seat, info_set).
    key = ["game", "game_params", "lm_seat", "info_set"]
    variants = keyed.group_by(key).agg(
        pl.col("_legal_mask_key").n_unique().alias("_n_variants")
    )
    inconsistent = variants.filter(pl.col("_n_variants") > 1)
    if inconsistent.height > 0:
        raise ValueError(
            f"{inconsistent.height} (game, game_params, lm_seat, info_set) key(s) carry "
            f"more than one distinct legal_mask across logged rows: {inconsistent.to_dicts()}"
        )

    is_fallback = pl.col("fallback_used").fill_null(True)
    own = df.filter(~is_fallback)
    totals = df.group_by(key).agg(
        (~is_fallback).sum().alias("logged_n_rows"),
        is_fallback.sum().alias("logged_n_fallback_rows"),
    )
    # Deterministic mode over the model's own rows: sort each group by
    # (count desc, action asc) and take the first row -- ties broken toward
    # the lower action index rather than whatever order polars happens to
    # return. A key held only by fallback rows has no modal action.
    per_action = own.group_by(key + ["parsed_action"]).agg(pl.len().alias("_n"))
    modal = (
        per_action.sort(["_n", "parsed_action"], descending=[True, False])
        .unique(subset=key, keep="first")
        .rename(
            {"parsed_action": "logged_modal_action", "_n": "logged_modal_action_count"}
        )
    )
    stats = totals.join(modal, on=key, how="left")

    reps = df.unique(subset=key, keep="first", maintain_order=True)
    reps = reps.join(stats, on=key)
    return reps.sort(["game", "lm_seat", "info_set"])


# ---------------------------------------------------------------------------
# Arm provenance.
#
# The canonical-continuation detector (assert_canonical_continuation) proves
# the endpoint is self-consistent: its /tokenize agrees with its own
# /tokenize. It proves nothing about fidelity to the endpoint that
# Originally collected this arm. An endpoint started with a --chat-template
# override, a different revision, or a different tokenizer mode satisfies
# the self-consistency detector while every scored prefix is one the arm
# never saw. verify_arm_provenance closes that gap, at the granularity the
# logged Parquet actually carries.
# ---------------------------------------------------------------------------


def verify_arm_provenance(
    paths: list[str],
    root_url: str,
    model: str,
    base_url: str,
    timeout: float = 30.0,
) -> None:
    """The endpoint about to be queried must match the one that
    COLLECTED this arm, and the arm being read must be the one `--model`
    names.

    The canonical-continuation detector compares /tokenize against /tokenize
    on the same endpoint, so it proves self-consistency and nothing about
    fidelity to the arm. An endpoint configured differently from the
    collecting one (a --chat-template override, a swapped model, a different
    tokenizer mode) satisfies both detector classes while every scored
    prefix is one the arm never saw. Separately, this catches the much more
    mundane operator error of pointing `--data-dir` at one arm while
    `--model` names another, which would silently produce a full CSV of
    confidently wrong numbers.

    Reads `chat_template_hash` and `model_id` from the arm's own Parquet,
    filtered to LM rows (`prompt.is_not_null()`) -- a healthy arm also
    carries non-LM rows (this harness's CFR/random/etc. reference agents)
    with `chat_template_hash`/`model_id` null and `prompt` null; reading
    unfiltered would see the null alongside the real value as a second
    "distinct value" and refuse a healthy arm. Requires exactly one distinct
    value of each across the LM rows, then compares:

      - `chat_template_hash` against `resolve_chat_template_hash(base_url,
        model)`, live, resolved fresh from the endpoint about to be queried.
      - `model_id` against the `model` argument (`--model` at the CLI) --
        the arm's own record of which served model produced it, compared
        against what this invocation claims it is scoring.

    `model_revision` is deliberately not compared against this checkout's
    own git SHA. `model_revision` is the HARNESS's own git SHA at collection
    time (match_runner.py stamps `model_revision=git_sha` unconditionally;
    see src/lmopt/logging/schema.py's `model_revision` field), so that
    comparison would be "the commit that collected the arm" against "the
    commit re-scoring it", which differ by construction on every real re-run
    and would abort every real invocation, not just a drifted one. There is
    no served weights revision resolvable from a running vLLM endpoint
    anywhere in this codebase -- `/v1/models` reports only the model id,
    nothing identifies which weights snapshot is loaded, and no logged column
    identifies which weights produced a row -- so nothing here can gate on
    weights identity. `model_revision` is a NON-gating provenance record
    instead -- see `arm_revision_provenance`, below, which writes it (and, if
    a `weights_provenance.json` sidecar exists, the separately-INFERRED
    served-weights revision)
    into the output artefact for a reader to see, never into a check that
    aborts a run.

    Drift-detection, NOT identity, and the gap is concrete rather than
    theoretical: CHAT_TEMPLATE_PROBE (chat_template.py) is four turns
    with a system message, while every prompt this pass scores is a single
    bare user turn with none (`LMVllmAgent._complete`'s
    `messages=[{"role": "user", ...}]`, lm_vllm.py). A template change
    confined to
    the no-system-message path -- a conditional default-system injection
    being the ordinary case -- renders the probe byte-identically and still
    moves every scored prefix. This residual is narrowed by
    `canonical_prefix_ids_sha256` (below), which hashes the live-resolved
    prefix ids for one named canonical info-set PER game -- the shape
    actually scored -- into the output artefact, so a later re-run over this
    arm can diff against that shape rather than only the probe's. Without a
    prior recorded value to diff against, this check is drift-only.

    What this check covers, no more: it verifies the served chat template
    has not drifted from the one that collected the arm, and that the arm
    being read and the `--model` argument agree. It does not verify
    served-weights identity, because nothing in this system can -- a caller
    must not read a clean pass through this check as proof the weights
    match.

    Raises `ArmProvenanceMismatch` -- never `ChatTemplateUnavailable` or
    `ServingChatTemplateUnavailable` -- on any disagreement, including when
    the live endpoint cannot be resolved at all (that failure is wrapped,
    not left as the serving-layer's own exception type, so a caller needs
    only one `except` clause for every way this check can fail).
    """
    frames = []
    for p in paths:
        df = pl.read_parquet(p, columns=["prompt", "chat_template_hash", "model_id"])
        frames.append(df.filter(pl.col("prompt").is_not_null()))
    lm_rows = (
        pl.concat(frames)
        if frames
        else pl.DataFrame(
            schema={
                "prompt": pl.Utf8,
                "chat_template_hash": pl.Utf8,
                "model_id": pl.Utf8,
            }
        )
    )
    if lm_rows.height == 0:
        raise ArmProvenanceMismatch(
            f"no LM rows (prompt.is_not_null()) found across {paths!r} -- cannot verify "
            "this arm's chat_template_hash/model_id against the live endpoint / --model"
        )

    hashes = lm_rows["chat_template_hash"].unique().to_list()
    if len(hashes) != 1:
        raise ArmProvenanceMismatch(
            f"chat_template_hash is not a single distinct value across this arm's LM rows: "
            f"{hashes!r} -- the arm's own data is internally inconsistent, nothing to compare "
            "the live endpoint against"
        )
    arm_hash = hashes[0]

    try:
        live_hash = resolve_chat_template_hash(base_url, model, timeout=timeout)
    except ServingChatTemplateUnavailable as exc:
        raise ArmProvenanceMismatch(
            f"could not resolve chat_template_hash from the live endpoint at {base_url} for "
            f"model {model!r}: {exc}"
        ) from exc
    if live_hash != arm_hash:
        raise ArmProvenanceMismatch(
            f"chat_template_hash mismatch: this arm was collected with chat_template_hash="
            f"{arm_hash!r}, but the endpoint at {base_url} for model {model!r} currently "
            f"resolves to {live_hash!r} -- the endpoint about to be queried is not configured "
            "the same way as the one that collected this arm (a --chat-template override, a "
            "swapped model, or a different tokenizer mode are the known causes); refusing to "
            "score against it"
        )

    model_ids = lm_rows["model_id"].unique().to_list()
    if len(model_ids) != 1:
        raise ArmProvenanceMismatch(
            f"model_id is not a single distinct value across this arm's LM rows: "
            f"{model_ids!r} -- the arm's own data is internally inconsistent, nothing to "
            "compare --model against"
        )
    arm_model_id = model_ids[0]
    if arm_model_id != model:
        raise ArmProvenanceMismatch(
            f"model_id mismatch: this arm's own logged rows say it was collected as "
            f"model_id={arm_model_id!r}, but this invocation's --model is {model!r} -- "
            "--data-dir/--parquet is very likely pointed at a DIFFERENT arm than --model "
            "names; refusing to score this arm's prompts as though they were that model's"
        )


def canonical_prefix_ids_sha256(
    paths: list[str],
    root_url: str,
    model: str,
    timeout: float = 30.0,
    chat_template_kwargs: dict | None = None,
) -> dict[str, str]:
    """Live-resolved `prefix_ids` sha256, for one named canonical info-set
    PER game -- the alphabetically-first distinct `info_set` among this
    arm's LM rows, per game.

    Narrows the residual `verify_arm_provenance` documents: CHAT_TEMPLATE_
    PROBE's four-turn-with-system shape can render byte-identically across a
    template change confined to the single-bare-user-turn shape this pass
    actually scores, so a probe-hash match is not proof the scored shape is
    unchanged. With no prior recorded value to diff against, it is
    drift-only -- but a later re-run over this same arm can diff its own
    per-game hash against what this run recorded, which is an identity check
    on the shape actually scored, not on the probe's.

    Never raises: a per-game `ChatTemplateUnavailable` (the local class,
    from `render_chat_prefix_ids`) is recorded as an `"error: ..."` string
    value for that game rather than aborting the whole run -- this is a
    forward-looking provenance nicety, not a gate, so one game's endpoint
    hiccup must not discard results already collected for every other game.
    The failure stays visible in the artefact rather than being silently
    dropped.
    """
    frames = []
    for p in paths:
        df = pl.read_parquet(p, columns=["game", "info_set", "prompt"])
        frames.append(df.filter(pl.col("prompt").is_not_null()))
    lm_rows = (
        pl.concat(frames)
        if frames
        else pl.DataFrame(
            schema={"game": pl.Utf8, "info_set": pl.Utf8, "prompt": pl.Utf8}
        )
    )
    if lm_rows.height == 0:
        return {}

    reps = lm_rows.sort(["game", "info_set"]).unique(
        subset=["game"], keep="first", maintain_order=True
    )
    result: dict[str, str] = {}
    for row in reps.iter_rows(named=True):
        try:
            ids = render_chat_prefix_ids(
                root_url, model, row["prompt"], timeout, chat_template_kwargs
            )
        except ChatTemplateUnavailable as exc:
            result[row["game"]] = f"error: chat_template_unavailable: {exc}"
            continue
        result[row["game"]] = hashlib.sha256(
            json.dumps(ids).encode("utf-8")
        ).hexdigest()
    return result


def arm_revision_provenance(paths: list[str]) -> dict[str, Optional[str]]:
    """Non-gating provenance record of what this arm's own data says about
    which harness commit, and (if known) which served weights, collected
    it -- recorded for a reader, never compared against anything live.

    `model_revision` (read here from the arm's own LM rows, filtered the
    same way `verify_arm_provenance` filters) is the HARNESS's own git SHA
    at collection time, not a served-weights identity -- see
    `verify_arm_provenance`'s docstring for why comparing it against
    anything live cannot gate. No logged column identifies which weights
    produced a row; this function does not attempt to close that gap
    either, it only surfaces what is known.

    A served-weights revision, when one has been separately established,
    comes from an OFFLINE inference pass this module does not own, which writes a
    `weights_provenance.json` sidecar beside the arm's `run_meta.json` (same
    directory as the arm's Parquet files) -- read here, if present, for its
    `weights_revision` and `confidence` fields. A missing sidecar is
    recorded as "not recorded" (all three keys `None`), never an error:
    most arms will not have one, and this pass does not produce one itself.
    A malformed sidecar degrades the same way -- this is a best-effort
    provenance nicety, not an invariant the per-info-set scoring loop is
    held to, so a corrupt sidecar must not crash an otherwise-healthy run.

    Returns `{"model_revision": ..., "weights_revision": ...,
    "weights_revision_confidence": ...}`; `model_revision` is `None` only
    when there are no LM rows to read it from, or the literal joined string
    of every distinct value seen (comma-separated) if the arm's own data is
    internally inconsistent -- unlike `verify_arm_provenance`'s checks, this
    function never raises, so an inconsistency is surfaced in the recorded
    value itself rather than by aborting.
    """
    frames = []
    for p in paths:
        df = pl.read_parquet(p, columns=["prompt", "model_revision"])
        frames.append(df.filter(pl.col("prompt").is_not_null()))
    lm_rows = (
        pl.concat(frames)
        if frames
        else pl.DataFrame(schema={"prompt": pl.Utf8, "model_revision": pl.Utf8})
    )
    revisions = lm_rows["model_revision"].unique().to_list() if lm_rows.height else []
    if len(revisions) == 0:
        model_revision = None
    elif len(revisions) == 1:
        model_revision = revisions[0]
    else:
        model_revision = ",".join(sorted(str(r) for r in revisions))

    weights_revision = None
    weights_revision_confidence = None
    if paths:
        sidecar = Path(paths[0]).parent / "weights_provenance.json"
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text())
                weights_revision = data.get("weights_revision")
                weights_revision_confidence = data.get("confidence")
            except Exception:
                # Best-effort provenance nicety (see docstring) -- a
                # malformed sidecar is recorded as "not recorded", not a
                # crash, matching _git_sha()'s own "never raises" posture
                # for provenance-only helpers.
                pass

    return {
        "model_revision": model_revision,
        "weights_revision": weights_revision,
        "weights_revision_confidence": weights_revision_confidence,
    }


# ---------------------------------------------------------------------------
# Cardinality.
#
# verify_arm_provenance proves the endpoint about to be queried matches the
# one that COLLECTED this arm. It proves nothing about whether `paths` (from
# --data-dir/--parquet) is the arm's whole shard set: a truncated
# `--data-dir` would pass every other check and ship as if complete.
# `verify_cardinality` closes that gap using run_meta.json as an independent
# reference -- see ShardCountMismatch's docstring for why a same-paths
# self-comparison (distinct_p0_info_sets(paths) vs paths) cannot do this.
# ---------------------------------------------------------------------------


def _read_run_meta_json(paths: list[str]) -> Optional[dict]:
    """Loads run_meta.json from beside paths[0] (the same directory the
    arm's Parquet shards live in -- the pattern arm_revision_provenance's
    weights_provenance.json sidecar lookup already uses).

    Returns None on any failure to obtain a dict: no paths given, the file
    does not exist, it cannot be read, or it is not valid JSON. `None` is
    not "check not applicable" -- verify_cardinality treats it as an abort
    condition, identically to a declared-vs-actual shard count mismatch
    (the most truncated real directory on disk has no run_meta.json at all,
    so treating absence as a pass would exempt exactly that case).
    Kept separate from verify_cardinality so that function stays a pure
    comparison, trivial to unit-test without touching disk.
    """
    if not paths:
        return None
    meta_path = Path(paths[0]).parent / "run_meta.json"
    try:
        data = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


#: The checkout this script lives in. A composite's `composed_from` lists its pieces relative to it.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_piece_metas(
    run_meta: Optional[dict], repo_root: Path = REPO_ROOT
) -> Optional[list[dict]]:
    """
    the run_meta.json of every piece a composite was composed from, read from each piece's own
    directory, so the composite's shard count can be derived from what each piece declares
    :param run_meta: the arm's own run_meta.json, or None (dict | None)
    :param repo_root: what a relative `composed_from` path is relative to (Path)
    :returns: one dict per piece in `composed_from` order, or None when the arm is not a composite
        (it has no `composed_from`) (list[dict] | None)
    :raises ShardCountMismatch: naming the piece, when a piece's run_meta.json cannot be read; a
        composite whose pieces cannot be counted aborts, it is never counted from what remains
    """
    if run_meta is None or "composed_from" not in run_meta:
        return None
    metas = []
    for raw in run_meta["composed_from"]:
        piece = Path(raw)
        if not piece.is_absolute():
            piece = repo_root / piece
        try:
            metas.append(json.loads((piece / "run_meta.json").read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            raise ShardCountMismatch(
                f"composite piece {raw}: its run_meta.json cannot be read ({exc}), so the "
                "composite's shard count cannot be derived; nothing is counted from the pieces "
                "that remain"
            ) from exc
    return metas


def _expected_shard_count(
    run_meta: dict, piece_metas: Optional[list[dict]] = None
) -> int:
    """``len(games) * len(opponents) * n_seeds`` from
    ``run_meta['resolved_config']`` -- the same shape
    scripts/run_experiment.py's own `n_files` print and
    scripts/run_sweep.py's shard count (one Parquet file per
    (game, opponent, seed) triple, built in its per-seed/per-opponent loop
    via `_shard_out_path`) derive from at collection time, applied here in
    reverse as the independent reference an already-landed arm ought to
    satisfy. For example, an arm declaring 5 x 4 x 5 = 100 holds exactly 100
    parquet shards.

    Raises ShardCountMismatch (never KeyError/TypeError/ValueError) if
    ``resolved_config`` or any of the three keys is missing or the wrong
    shape -- an unparseable ``resolved_config`` must ABORT, not be treated
    as "nothing to check".


    A composite is the sum over its pieces, each counted from its own
    ``resolved_config``. Its own ``resolved_config`` cannot stand in: it is the
    union of the pieces' games with one piece's ``n_seeds``, so it is wrong for
    seed-disjoint pieces (one seed each) and for game-disjoint ones alike, and
    no product of ``n_pieces`` with it is right for both. A composite handed
    in without its pieces' run_meta is refused rather than counted.
    """
    if piece_metas is not None:
        return sum(_expected_shard_count(meta) for meta in piece_metas)
    if "composed_from" in run_meta:
        raise ShardCountMismatch(
            "this run_meta.json is a composite's (it has composed_from); its shard count is the "
            "sum over its pieces, so pass the pieces' run_meta.json (_read_piece_metas)"
        )
    try:
        cfg = run_meta["resolved_config"]
        return len(cfg["games"]) * len(cfg["opponents"]) * int(cfg["n_seeds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ShardCountMismatch(
            f"run_meta.json's resolved_config is missing or malformed (need "
            f"list-like 'games'/'opponents' and an int-like 'n_seeds'): {exc} -- "
            f"run_meta was {run_meta!r}"
        ) from exc


def _refuse_memory_level(exc: MemoryLevelNotM0) -> int:
    """
    report a memory-level refusal on stderr, naming the level, the same way at every site in main
    :param exc: the refusal (MemoryLevelNotM0)
    :returns: the exit status, 10 (int)
    """
    print(f"\nABORT: memory level not M0 -- {exc}", file=sys.stderr)
    return 10


def verify_memory_level(run_meta: Optional[dict]) -> None:
    """
    refuse an arm whose own run meta does not say it was collected at memory level M0
    :param run_meta: the arm's run_meta.json as a dict, None when it could not be read, which
        is left to the cardinality check (exit 5) and not judged here (dict or None)
    :raises MemoryLevelNotM0: when resolved_config.memory_rung is absent, or is anything but the
        string M0; an absent key is not read as M0, because every recorded arm carries it
    """
    if run_meta is None:
        return
    config = run_meta.get("resolved_config")
    rung = config.get("memory_rung") if isinstance(config, dict) else None
    if rung != "M0":
        raise MemoryLevelNotM0(
            f"run_meta.json records memory_rung {rung!r}, not 'M0'; the pass refuses every "
            f"arm above M0 (above it one info-set has many prompts and keep-first would "
            f"score an arbitrary one), and an arm whose level is not recorded cannot be "
            f"told from one that is above it"
        )


def _refuse_prompt_epoch(exc: PromptEpochNotLive) -> int:
    """
    report a prompt-version refusal on stderr, the same way at every site in main
    :param exc: the refusal (PromptEpochNotLive)
    :returns: the exit status, 11 (int)
    """
    print(f"\nABORT: prompt version not live -- {exc}", file=sys.stderr)
    return 11


def verify_prompt_epoch(run_meta: Optional[dict]) -> None:
    """
    refuse an arm whose own run meta does not record the live prompt spec hash
    :param run_meta: the arm's run_meta.json as a dict, None when it could not be read, which
        is left to the cardinality check (exit 5) and not judged here (dict or None)
    :raises PromptEpochNotLive: when run_meta's prompt_spec_hash is absent or differs from the
        registry entry of the live schema version; an absent key is not read as live
    """
    if run_meta is None:
        return
    recorded = run_meta.get("prompt_spec_hash")
    live = PROMPT_SPEC_HASHES[PROMPT_SCHEMA_VERSION]
    if recorded != live:
        raise PromptEpochNotLive(
            f"run_meta.json records prompt_spec_hash {recorded!r}, not the live schema "
            f"{PROMPT_SCHEMA_VERSION} hash {live!r}; the pass re-renders prompts with the live "
            f"code, so it cannot score an arm collected under another prompt version"
        )


def verify_cardinality(
    paths: list[str],
    run_meta: Optional[dict],
    piece_metas: Optional[list[dict]] = None,
) -> None:
    """``len(paths)`` must equal the shard count ``run_meta.json``
    itself declares this arm was collected with.

    ``run_meta=None`` means the caller (``_read_run_meta_json``) could not
    find or parse ``run_meta.json`` -- treated identically to a declared-
    vs-actual mismatch: this raises, it does not skip (no silent fallbacks).

    Raises ``ShardCountMismatch``, naming both ``len(paths)`` and the
    declared shard count, on any disagreement.
    """
    if run_meta is None:
        raise ShardCountMismatch(
            f"run_meta.json not found or unparseable for the {len(paths)} parquet "
            "file(s) this run resolved -- cannot verify independently that this is "
            "the arm's FULL shard set; a missing/unparseable run_meta.json aborts "
            "rather than skipping this check (no silent fallbacks)"
        )
    expected = _expected_shard_count(run_meta, piece_metas)
    if len(paths) != expected:
        raise ShardCountMismatch(
            f"{len(paths)} parquet file(s) resolved, but run_meta.json's "
            f"resolved_config declares {expected} shard(s) (len(games) x "
            "len(opponents) x n_seeds) -- this --data-dir/--parquet very likely "
            "resolves to a PARTIAL shard set"
        )


def verify_row_count_completeness(results: list["InfoSetResult"]) -> None:
    """Sibling check to ``verify_cardinality``: the number of (info-set,
    action) rows this run actually produced must equal the sum of
    ``n_legal`` across the info-sets it processed.

    Every branch of ``score_info_set``/``_all_failed_scores`` emits exactly
    one row per legal action, success or failure, by construction (see this
    module's own docstring: "len(scores) == n_legal_actions holds for every
    info-set this pass touches"). Comparing the sum across `results` (not
    trusting that per-result invariant blindly) is a regression guard: a
    future change that broke it for exactly one info-set, while leaving the
    total NUMBER of info-sets unchanged, would otherwise ship silently past
    the "n_info_sets" summary line alone.

    Callers must not call this on a ``--dry-run`` result list: dry-run's
    ``InfoSetResult.scores`` is deliberately ``[]`` (see ``run_pass``), a
    documented shape unrelated to truncation, not something this check
    should evaluate.
    """
    n_output_rows = sum(len(r.scores) for r in results)
    n_expected_rows = sum(r.n_legal for r in results)
    if n_output_rows != n_expected_rows:
        raise ShardCountMismatch(
            f"{n_output_rows} output row(s) across {len(results)} processed "
            f"info-set(s), but their legal_mask columns declare {n_expected_rows} "
            "legal action(s) in total -- rows were dropped or duplicated between "
            "scoring and writing"
        )


def verify_full_population_processed(
    n_processed: int,
    n_available: int,
    limit: Optional[int],
) -> None:
    """When no ``--limit`` was given, the number of info-sets this run
    actually processed (``n_processed``, ``summary["n_info_sets"]``) must
    equal the arm's full distinct-P0 population (``n_available``,
    ``summary["n_info_sets_available"]`` -- ``run_pass``'s own pre-limit
    ``reps.height``, never recomputed here from ``paths`` a second time,
    which would reintroduce the circularity ``ShardCountMismatch``'s
    docstring describes).

    This is the check ``verify_cardinality`` and ``verify_row_count_
    completeness`` together still miss: trace ``--data-dir <full 100-shard
    arm> --limit 5`` through both of them. ``verify_cardinality`` compares
    ``len(paths)`` (100) against ``run_meta.json`` (100) -- passes.
    ``verify_row_count_completeness`` compares output rows against
    ``sum(r.n_legal for r in results)`` over the 5 processed info-sets --
    both sides shrink together under ``--limit``, so it passes too. Without
    this check, a stale ``--limit``, a copy-pasted smoke command, or a
    leftover config value produces a small, clean-looking, artefact
    indistinguishable from a complete arm.

    When ``limit is not None``, this is a deliberate no-op: a genuinely
    partial smoke run is legitimate and must not abort. Instead its
    partiality is made to travel with the output artefact (``run_limit`` /
    ``n_info_sets_available`` columns, wired in ``main``) so it cannot later
    be mistaken for a complete arm -- see those columns' docstring note in
    ``main`` for the consumer-facing contract.

    Raises ``ShardCountMismatch``, naming both numbers, when ``limit is
    None`` and they disagree.
    """
    if limit is not None:
        return
    if n_processed != n_available:
        raise ShardCountMismatch(
            f"processed {n_processed} info-set(s), but the arm's Parquet holds "
            # "per-seat", not "P0": the population is keyed on
            # (game, game_params, lm_seat, info_set), so an arm collected under
            # seat_mode: random contributes info-sets at both seats and "P0"
            # would understate what the number counts.
            f"{n_available} distinct per-seat LM info-set(s) -- no --limit was given, so "
            "these must match; a stale --limit, a copy-pasted smoke command, or a "
            "leftover config value would silently produce exactly this shortfall"
        )


def population_summary(results: list["InfoSetResult"]) -> dict:
    """The arm's distinct info-set population, from `results` (one entry per
    distinct (game, game_params, info_set) this run processed -- no
    re-derivation from Parquet needed: `results` already carries exactly one
    entry per distinct info-set by construction, `run_pass`'s own
    (possibly ``--limit``-ed) ``reps``).

    Returns ``n_info_sets`` (total), ``per_game_counts`` (dict game ->
    count), and ``key_set_sha256`` -- sha256 over the sorted, JSON-encoded
    list of ``(game, info_set)`` pairs, so two arms' artefacts can be
    intersected downstream by comparing keys directly, without re-reading
    either arm's Parquet.

    Cross-arm populations are endogenous and differ substantially (on three
    seat-0-only arms: Qwen 185, OLMo 133, Mistral 127 distinct P0 info-sets;
    three-way intersection only 55; leduc is 73.0% of Qwen's info-sets
    against 56.7% of Mistral's; the population this pass builds is per-seat,
    so a seat-random arm's count is not comparable with those P0 counts) --
    an arm-level median is
    largely a leduc median, and two arms' medians are not directly
    comparable. This is why the population is written as data a downstream
    consumer computes an intersection over, never as a literal figure in
    prose: a hard-coded intersection size is stale the moment either arm's
    collection-path or game roster changes.
    """
    per_game: dict[str, int] = {}
    keys: list[list[str]] = []
    for r in results:
        per_game[r.game] = per_game.get(r.game, 0) + 1
        keys.append([r.game, r.info_set])
    keys.sort()
    key_set_sha256 = hashlib.sha256(
        json.dumps(keys, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "n_info_sets": len(results),
        "per_game_counts": per_game,
        "key_set_sha256": key_set_sha256,
    }


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


class ErrorKind(str, enum.Enum):
    """Structured failure classification.

    Classifying by string-matching the error prose would itself be a
    one-name lookup table, and the acceptance gate has to compute over the
    column rather than over a list of names. Every populated
    `ActionScore.error` carries exactly one of these.

    There is deliberately no member for a partial distribution. The partial
    case is `score_info_set`'s `elif legal:` branch, which sets
    `InfoSetResult.error` only -- and `InfoSetResult.error` is never written
    to the CSV. Stamping a kind onto the per-action rows instead would
    overwrite each failed action's own, strictly more specific kind
    (scoring_failed, echo_mismatch, ...), which is what consumers read. An
    unassignable member would read as a classification the gate might see
    and cannot.
    """

    GAME_LOAD_FAILED = "game_load_failed"
    VARIANT_SPEC_MISSING = "variant_spec_missing"
    PROMPT_VOCABULARY_UNPARSEABLE = "prompt_vocabulary_unparseable"
    CHAT_TEMPLATE_UNAVAILABLE = "chat_template_unavailable"
    ACTION_NOT_IN_ACTION_LABELS = "action_not_in_action_labels"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    SCORING_FAILED = "scoring_failed"
    INFO_SET_SCORING_CRASHED = "info_set_scoring_crashed"
    NONCANONICAL_PREFIX = "noncanonical_prefix"
    NONCANONICAL_ACTION = "noncanonical_action"
    NONCANONICAL_SUFFIX = "noncanonical_suffix"
    ECHO_MISMATCH = "echo_mismatch"
    # The only kind the acceptance criterion permits on a healthy arm: a
    # measurement of model behaviour, not an instrument fault.
    COVERAGE_BELOW_FLOOR = "coverage_below_floor"


@dataclasses.dataclass
class ActionScore:
    game: str
    game_params: str
    info_set: str
    action: int
    action_text: str
    raw_logprob: Optional[float]  # unnormalised log P(action_text | templated prefix)
    prob: Optional[float]  # renormalised over the info-set's legal actions
    n_tokens: Optional[int]
    error: Optional[str]  # None on success
    # Structured classification of `error`. Stored as
    # the enum member's plain `.value` str, never the enum member itself, so
    # dataclasses.asdict and polars both see a str. None iff `error` is None;
    # every populated `error` carries exactly one of these - see ErrorKind.
    error_kind: Optional[str] = None
    # Σ exp(raw_logprob) over all legal actions at this info-set (denormalised
    # onto every row of the info-set) - the coverage diagnostic. See
    # COVERAGE_MASS_FLOOR for what it means.
    coverage_mass: Optional[float] = None
    # Cross-check against logged behaviour -
    # denormalised onto every row of the info-set. exact_modal_action is
    # this pass's own argmax(prob) (None if no complete/trusted distribution
    # exists); logged_* come from the full set of logged rows sharing this
    # info-set (not just the deduplicated representative), independent of
    # whether this pass could score anything at all.
    exact_modal_action: Optional[int] = None
    logged_modal_action: Optional[int] = None
    logged_modal_action_count: Optional[int] = None
    logged_n_rows: Optional[int] = None
    logged_n_fallback_rows: Optional[int] = None
    modal_agreement: Optional[bool] = None  # exact_modal_action == logged_modal_action


@dataclasses.dataclass
class InfoSetResult:
    game: str
    game_params: str
    info_set: str
    n_legal: int
    scores: list[ActionScore]
    prefix_ids: Optional[list[int]]
    # Set whenever this info-set does not have a trustworthy, complete
    # renormalised `prob` distribution: a whole-info-set failure before
    # per-action scoring (game load/dispatch, unparseable prompt vocabulary,
    # chat-template unavailable), a PARTIAL per-action failure (fewer than
    # n_legal actions scored), or a complete-but-untrusted distribution
    # (coverage_mass below COVERAGE_MASS_FLOOR). None only when every legal
    # action was scored and coverage_mass cleared the floor.
    error: Optional[str]
    coverage_mass: Optional[float] = None
    exact_modal_action: Optional[int] = None
    logged_modal_action: Optional[int] = None
    logged_modal_action_count: Optional[int] = None
    logged_n_rows: Optional[int] = None
    logged_n_fallback_rows: Optional[int] = None
    modal_agreement: Optional[bool] = None


def _legal_actions(legal_mask: list[int]) -> list[int]:
    return [i for i, m in enumerate(legal_mask) if m]


def _all_failed_scores(
    game: str,
    game_params: str,
    info_set: str,
    legal_actions: list[int],
    error: str,
    error_kind: str,
) -> list[ActionScore]:
    """One failure row per legal action - never fewer, so a whole-info-set
    failure (e.g. chat template unavailable) still surfaces every action it
    would have cost, rather than vanishing from the output. `error_kind` is
    the caller's own ErrorKind (its `.value` str) stamped onto every row."""
    return [
        ActionScore(
            game,
            game_params,
            info_set,
            a,
            "",
            None,
            None,
            None,
            error,
            error_kind=error_kind,
        )
        for a in legal_actions
    ]


# Minimum Σ exp(raw_logprob) over an info-set's full legal-action set for its
# renormalised `prob` distribution to be trusted. Against the live endpoint,
# a wrong vocabulary reads 1e-17 to 1e-36, so 0.5 is unreachable by a
# realistic wrong-vocabulary failure. Legitimate readings sit closer than
# that: Mistral's legitimate coverage_mass readings reach 0.619 at minimum,
# and liarsdice's median is 0.769, so the separation between 0.5 and the
# lowest legitimate reading is 0.12, not orders of magnitude. The constant is
# pre-registered and must not be tuned on the strength of observed readings.
#
# What a low reading means: `action_labels` is sourced directly from the
# prompting layer's own rendering data and is standing-invariant-checked
# against action_suffix over the full tree of every registered variant
# (tests/test_prompt_fidelity.py), so a vocabulary mismatch is near-impossible
# by construction. A low reading should therefore be read as evidence about
# MODEL BEHAVIOUR (the model spreading mass onto a preamble, hedge, or
# reasoning tokens rather than the bare requested literal) rather than an
# instrument fault. Any behavioural threshold must be pre-registered before
# the arms it applies to run, not tuned after seeing their numbers.
COVERAGE_MASS_FLOOR = 0.5


def score_info_set(
    root_url: str,
    model: str,
    game: str,
    game_params: str,
    info_set: str,
    prompt: str,
    legal_mask: list[int],
    timeout: float = 60.0,
    gen_suffix: Sequence[int] = (),
    chat_template_kwargs: dict | None = None,
    *,
    probe_type: str,
) -> InfoSetResult:
    legal = _legal_actions(legal_mask)

    try:
        prompt_spec = prompt_spec_for_arm(game, probe_type)
    except Exception as exc:
        err = f"variant_spec_missing: {exc}"
        return InfoSetResult(
            game,
            game_params,
            info_set,
            len(legal),
            _all_failed_scores(
                game,
                game_params,
                info_set,
                legal,
                err,
                ErrorKind.VARIANT_SPEC_MISSING.value,
            ),
            None,
            err,
        )

    try:
        game_obj = load_game(game, **json.loads(game_params))
        fns = get_game_fns_for_game(
            game_obj, prompt_spec
        )  # dispatch only - .build_prompt is never called
    except Exception as exc:
        err = f"game_load_failed: {exc}"
        return InfoSetResult(
            game,
            game_params,
            info_set,
            len(legal),
            _all_failed_scores(
                game,
                game_params,
                info_set,
                legal,
                err,
                ErrorKind.GAME_LOAD_FAILED.value,
            ),
            None,
            err,
        )

    try:
        labels = action_texts_for_info_set(fns, info_set, prompt)
    except PromptVocabularyUnparseable as exc:
        err = f"prompt_vocabulary_unparseable: {exc}"
        return InfoSetResult(
            game,
            game_params,
            info_set,
            len(legal),
            _all_failed_scores(
                game,
                game_params,
                info_set,
                legal,
                err,
                ErrorKind.PROMPT_VOCABULARY_UNPARSEABLE.value,
            ),
            None,
            err,
        )

    try:
        prefix_ids = render_chat_prefix_ids(
            root_url, model, prompt, timeout, chat_template_kwargs
        )
    except ChatTemplateUnavailable as exc:
        err = f"chat_template_unavailable: {exc}"
        return InfoSetResult(
            game,
            game_params,
            info_set,
            len(legal),
            _all_failed_scores(
                game,
                game_params,
                info_set,
                legal,
                err,
                ErrorKind.CHAT_TEMPLATE_UNAVAILABLE.value,
            ),
            None,
            err,
        )

    scores: list[ActionScore] = []
    raw_logprobs: dict[int, float] = {}

    for action in legal:
        action_text = labels.get(action)
        if action_text is None:
            # Should not happen: the standing invariant test in
            # tests/test_prompt_fidelity.py checks action_labels' key set
            # equals legal_actions() over the full tree of the current code.
            # A mismatch here means this row's logged legal_mask disagrees
            # with what the current prompting layer computes for this
            # info-set - code/data drift, not silently guessed or skipped.
            scores.append(
                ActionScore(
                    game,
                    game_params,
                    info_set,
                    action,
                    "",
                    None,
                    None,
                    None,
                    f"action_not_in_action_labels: action {action} missing from "
                    f"action_labels({info_set!r}) = {labels!r}",
                    error_kind=ErrorKind.ACTION_NOT_IN_ACTION_LABELS.value,
                )
            )
            continue
        try:
            action_ids = tokenize_action(root_url, model, action_text, timeout)
            # The canonical-continuation detector, called per (info-set,
            # action) before scoring. An exact prefix/action boundary (ids,
            # not text) does not imply the submitted sequence is the one the
            # server would itself produce; see assert_canonical_continuation's
            # docstring for the failure mode it guards against.
            canonical = canonical_assistant_ids(
                root_url, model, prompt, action_text, timeout, chat_template_kwargs
            )
            assert_canonical_continuation(canonical, prefix_ids, action_ids, gen_suffix)
            raw_lp, n_tok = score_action_logprob(
                root_url, model, prefix_ids, action_ids, timeout
            )
            raw_logprobs[action] = raw_lp
            scores.append(
                ActionScore(
                    game,
                    game_params,
                    info_set,
                    action,
                    action_text,
                    raw_lp,
                    None,
                    n_tok,
                    None,
                )
            )
        except NonCanonicalPrefix as exc:
            scores.append(
                ActionScore(
                    game,
                    game_params,
                    info_set,
                    action,
                    action_text,
                    None,
                    None,
                    None,
                    f"noncanonical_prefix: {exc}",
                    error_kind=ErrorKind.NONCANONICAL_PREFIX.value,
                )
            )
        except NonCanonicalSuffix as exc:
            scores.append(
                ActionScore(
                    game,
                    game_params,
                    info_set,
                    action,
                    action_text,
                    None,
                    None,
                    None,
                    f"noncanonical_suffix: {exc}",
                    error_kind=ErrorKind.NONCANONICAL_SUFFIX.value,
                )
            )
        except NonCanonicalActionTokenisation as exc:
            scores.append(
                ActionScore(
                    game,
                    game_params,
                    info_set,
                    action,
                    action_text,
                    None,
                    None,
                    None,
                    f"noncanonical_action: {exc}",
                    error_kind=ErrorKind.NONCANONICAL_ACTION.value,
                )
            )
        except ContextLengthExceeded as exc:
            scores.append(
                ActionScore(
                    game,
                    game_params,
                    info_set,
                    action,
                    action_text,
                    None,
                    None,
                    None,
                    f"context_length_exceeded: {exc}",
                    error_kind=ErrorKind.CONTEXT_LENGTH_EXCEEDED.value,
                )
            )
        except EchoedSequenceMismatch as exc:
            scores.append(
                ActionScore(
                    game,
                    game_params,
                    info_set,
                    action,
                    action_text,
                    None,
                    None,
                    None,
                    f"echo_mismatch: {exc}",
                    error_kind=ErrorKind.ECHO_MISMATCH.value,
                )
            )
        except Exception as exc:
            scores.append(
                ActionScore(
                    game,
                    game_params,
                    info_set,
                    action,
                    action_text,
                    None,
                    None,
                    None,
                    f"scoring_failed: {exc}",
                    error_kind=ErrorKind.SCORING_FAILED.value,
                )
            )

    # Renormalise only if every legal action was scored - a partial
    # renormalisation would silently present an incomplete distribution as a
    # complete one (the raw_logprob for whichever actions did succeed stays
    # visible either way; only `prob` requires completeness). And only if
    # coverage_mass clears COVERAGE_MASS_FLOOR - a low-coverage renormalised
    # distribution divides two vanishingly small (and noisy) residues into
    # something that looks like a confident probability, which is worse than
    # refusing to report one.
    info_set_error = None
    coverage_mass: Optional[float] = None
    if legal and len(raw_logprobs) == len(legal):
        coverage_mass = sum(math.exp(v) for v in raw_logprobs.values())
        if coverage_mass < COVERAGE_MASS_FLOOR:
            # The vocabulary is standing-invariant-checked against
            # action_suffix over the full tree (tests/test_prompt_fidelity.py),
            # and the prefix is verified canonical per call
            # (assert_canonical_continuation) before scoring ever runs. Both
            # are therefore verified by the time this branch is reached, so a
            # low reading here is evidence about MODEL BEHAVIOUR - the model
            # spreading probability mass onto a preamble, hedge, or reasoning
            # tokens rather than the bare requested literal - not a bug in
            # this script. It is still a counted, loud failure.
            info_set_error = (
                f"coverage_mass {coverage_mass:.3e} < floor {COVERAGE_MASS_FLOOR} - "
                "the action vocabulary (checked against action_suffix over the full tree) "
                "and the prefix (verified canonical per call) are both confirmed correct; "
                "this reading is evidence about MODEL BEHAVIOUR, not an instrument defect; "
                "refusing to renormalise a low-coverage distribution"
            )
            # Each individual HTTP call succeeded (raw_logprob is a real
            # measurement), but the aggregate is not trustworthy - mark every
            # action failed too, not just the info-set, so this is exactly as
            # loud (counted in run_pass's n_failures, printed by main's
            # failure loop) as every other failure class. A per-action
            # error=None here would make a rejected info-set invisible to
            # both.
            for s in scores:
                s.error = info_set_error
                s.error_kind = ErrorKind.COVERAGE_BELOW_FLOOR.value
        else:
            m = max(raw_logprobs.values())
            denom = sum(math.exp(v - m) for v in raw_logprobs.values())
            for s in scores:
                if s.raw_logprob is not None:
                    s.prob = math.exp(s.raw_logprob - m) / denom
    elif legal:
        info_set_error = f"{len(raw_logprobs)}/{len(legal)} legal actions scored - no complete distribution"

    exact_modal_action = None
    scored = [(s.prob, s) for s in scores if s.prob is not None]
    if scored:
        exact_modal_action = max(scored, key=lambda pair: pair[0])[1].action

    for s in scores:
        s.coverage_mass = coverage_mass
        s.exact_modal_action = exact_modal_action

    return InfoSetResult(
        game,
        game_params,
        info_set,
        len(legal),
        scores,
        prefix_ids,
        info_set_error,
        coverage_mass,
        exact_modal_action,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pass(
    parquet_paths: list[str],
    root_url: str,
    model: str,
    limit: Optional[int] = None,
    dry_run: bool = False,
    timeout: float = 60.0,
    gen_suffix: Sequence[int] = (),
    chat_template_kwargs: dict | None = None,
) -> tuple[list[InfoSetResult], dict[str, int]]:
    reps = distinct_p0_info_sets(parquet_paths)
    # Captured before --limit truncates
    # `reps` -- the arm's full distinct-P0 population, independent of
    # whatever --limit this invocation happens to carry. main() compares
    # this against the processed count when --limit is None (a real
    # shortfall then means something silently dropped info-sets, not a
    # deliberate smoke test) and writes it into the output artefact
    # alongside `run_limit` when --limit is set, so a partial run cannot
    # later be mistaken for a complete arm. This is not a second read of
    # the Parquet and does not reintroduce the circularity ShardCountMismatch
    # describes: it is the same `distinct_p0_info_sets(parquet_paths)` call
    # already made above, read before truncation rather than recomputed
    # from `paths` a second time.
    n_info_sets_available = reps.height
    if limit is not None:
        reps = reps.head(limit)

    results: list[InfoSetResult] = []
    n_calls = 0
    n_failures = 0
    n_modal_disagreements = 0
    # Counted independently of n_failures (every detector failure is also a
    # failure and is included in n_failures, as every other failure class
    # is) so an acceptance gate can read each detector class on its own,
    # rather than only ever seeing them folded anonymously into one total.
    #
    # Counting unit: both counters tally per (info-set, action) ROW, not per
    # info-set. Do not expect either to land on the arm's distinct-info-set
    # count (127 for the Mistral arm) -- that arm has 481 (info-set, action)
    # rows across those 127 info-sets, and a uniform defect that fires on
    # every row of the arm produces 481, not 127 (the lossy-round-trip and
    # BOS-action proof runs each produced 481).
    n_detector_prefix_failures = 0
    n_detector_action_failures = 0

    for row in reps.iter_rows(named=True):
        legal = _legal_actions(row["legal_mask"])
        logged_modal_action = row.get("logged_modal_action")
        logged_modal_action_count = row.get("logged_modal_action_count")
        logged_n_rows = row.get("logged_n_rows")
        logged_n_fallback_rows = row.get("logged_n_fallback_rows")

        if dry_run:
            results.append(
                InfoSetResult(
                    row["game"],
                    row["game_params"],
                    row["info_set"],
                    len(legal),
                    [],
                    None,
                    "dry_run",
                    logged_modal_action=logged_modal_action,
                    logged_modal_action_count=logged_modal_action_count,
                    logged_n_rows=logged_n_rows,
                    logged_n_fallback_rows=logged_n_fallback_rows,
                )
            )
            continue

        # An unanticipated exception anywhere inside
        # score_info_set (a transient network blip not already caught by its
        # own per-action try/except, an unknown game family, ...) must not
        # discard every result already collected for prior info-sets in this
        # run - isolate it to a failure row for this info-set and continue.
        # (Retries and incremental disk writes are deliberately not added
        # here; that gap is known.)
        try:
            result = score_info_set(
                root_url,
                model,
                row["game"],
                row["game_params"],
                row["info_set"],
                row["prompt"],
                row["legal_mask"],
                timeout,
                gen_suffix,
                chat_template_kwargs,
                probe_type=row["probe_type"],
            )
        except Exception as exc:
            err = f"info_set_scoring_crashed: {exc}"
            result = InfoSetResult(
                row["game"],
                row["game_params"],
                row["info_set"],
                len(legal),
                _all_failed_scores(
                    row["game"],
                    row["game_params"],
                    row["info_set"],
                    legal,
                    err,
                    ErrorKind.INFO_SET_SCORING_CRASHED.value,
                ),
                None,
                err,
            )

        # Attach the cross-check against logged
        # behaviour - computed once in distinct_p0_info_sets from the full
        # (non-deduplicated) set of logged rows for this info-set, joined
        # here onto every per-action row of this pass's own result.
        agreement = None
        if result.exact_modal_action is not None and logged_modal_action is not None:
            agreement = result.exact_modal_action == logged_modal_action
        result.logged_modal_action = logged_modal_action
        result.logged_modal_action_count = logged_modal_action_count
        result.logged_n_rows = logged_n_rows
        result.logged_n_fallback_rows = logged_n_fallback_rows
        result.modal_agreement = agreement
        for s in result.scores:
            s.logged_modal_action = logged_modal_action
            s.logged_modal_action_count = logged_modal_action_count
            s.logged_n_rows = logged_n_rows
            s.logged_n_fallback_rows = logged_n_fallback_rows
            s.modal_agreement = agreement
        if agreement is False:
            n_modal_disagreements += 1

        results.append(result)
        for s in result.scores:
            if s.error is None:
                n_calls += 1
            else:
                n_failures += 1
                if s.error_kind == ErrorKind.NONCANONICAL_PREFIX.value:
                    n_detector_prefix_failures += 1
                elif s.error_kind == ErrorKind.NONCANONICAL_ACTION.value:
                    n_detector_action_failures += 1

    # Modal disagreements warn, they do not gate. A disagreement between the
    # exact and logged modal actions is invisible at the console unless
    # someone opens the CSV, so it is surfaced in the summary and (below, in main()) listed on stderr, but not folded into
    # n_failures / the exit code: a disagreement can be legitimate (the
    # model's true distribution can differ from a small empirical sample's
    # mode, especially at low logged_n_rows), and main()'s exit code already
    # means "the instrument failed" - conflating a scientific finding with
    # an instrument failure would make both harder to read.
    return results, {
        "n_info_sets": len(results),
        "n_info_sets_available": n_info_sets_available,
        "n_calls": n_calls,
        "n_failures": n_failures,
        "n_modal_disagreements": n_modal_disagreements,
        "n_detector_prefix_failures": n_detector_prefix_failures,
        "n_detector_action_failures": n_detector_action_failures,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Best-effort git SHA of the checkout that produced a CSV - every run
    records backend/model revision/seed/git sha, and an output destined for
    publication must be able to say which commit produced it. Never raises: "unknown" on any failure
    (not a repo, git absent, ...) rather than crashing the whole pass over a
    provenance nicety.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            # This checkout's commit, whatever directory the pass was started from.
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


# Explicit output schema. Every ActionScore field, in
# Declaration order, followed by the twelve provenance keys main() merges in
# (also in the order they are inserted there) - this is exactly the column
# order `{dataclasses.asdict(s), provenance}` produces. See write_rows
# for why an explicit schema (rather than polars' inferred one) is required.
# `canonical_prefix_ids_sha256` is a JSON object string
# (`{game: sha256_hex_or_error}`), the same value repeated on every row of a
# run, not a per-row measurement - see canonical_prefix_ids_sha256's own
# docstring for what it narrows. `arm_revision_provenance` is likewise a
# JSON object string
# (`{model_revision, weights_revision, weights_revision_confidence}`) - see
# arm_revision_provenance's own docstring; neither of these two gates the
# run, both are recorded for a reader. `generation_prompt_suffix` is
# `repr(gen_suffix)` -- the same `list[int]` value
# (spec's `G`) `measure_generation_prompt_suffix` measured once for this run
# and every call in this run threaded into `assert_canonical_continuation`,
# repeated on every row like its two neighbours above. Unlike those two, it
# Is enforceable: `--expect-generation-prompt-suffix`, if given, aborts the
# run (exit 7, before any row is written) on disagreement. Recorded always;
# enforced only when the operator passes that flag by hand -- the
# orchestrator does not pass it (`scripts/run_experiment.py` builds its
# exact-policy command without it and `configs/run_experiment.yaml` has no
# knob for it), so a republished template is recorded per-row but does not
# by itself abort a run. `repr([])` (`"[]"`),
# never a blank, distinguishes "measured, and empty" from "not measured" -
# the latter is written as an actual null, which only happens under
# `--dry-run` (see `main`'s dry-run NOTE). `population_n_info_sets` /
# `population_per_game_counts` / `population_key_set_sha256` are
# `population_summary`'s output, likewise the same
# value repeated on every row - see population_summary's own docstring for
# why the population is written as data, never a literal figure in prose.
# `run_limit` / `n_info_sets_available` mark a `--limit`-truncated run's partiality so it travels with
# the artefact rather than only being caught at abort time - a caller
# treating this CSV as a complete arm must check `run_limit is not NULL`
# first (see the comment where `main` builds `provenance`).
_OUTPUT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "game": pl.Utf8,
    "game_params": pl.Utf8,
    "info_set": pl.Utf8,
    "action": pl.Int64,
    "action_text": pl.Utf8,
    "raw_logprob": pl.Float64,
    "prob": pl.Float64,
    "n_tokens": pl.Int64,
    "error": pl.Utf8,
    "error_kind": pl.Utf8,
    "coverage_mass": pl.Float64,
    "exact_modal_action": pl.Int64,
    "logged_modal_action": pl.Int64,
    "logged_modal_action_count": pl.Int64,
    "logged_n_rows": pl.Int64,
    "logged_n_fallback_rows": pl.Int64,
    "modal_agreement": pl.Boolean,
    "model": pl.Utf8,
    "base_url": pl.Utf8,
    "run_timestamp": pl.Utf8,
    "git_sha": pl.Utf8,
    "canonical_prefix_ids_sha256": pl.Utf8,
    "arm_revision_provenance": pl.Utf8,
    "generation_prompt_suffix": pl.Utf8,
    "population_n_info_sets": pl.Int64,
    "population_per_game_counts": pl.Utf8,
    "population_key_set_sha256": pl.Utf8,
    "run_limit": pl.Int64,
    "n_info_sets_available": pl.Int64,
}


def verify_detector_counts_consistent(summary: dict, rows: list[dict]) -> None:
    """The two representations of the canonical-continuation detector counts
    must agree before the CSV is written.

    ``summary`` (``run_pass``'s own inline counters,
    ``n_detector_prefix_failures`` / ``n_detector_action_failures``, printed
    to stdout by ``main``) must equal, respectively, the number of ``rows``
    whose ``error_kind`` is ``ErrorKind.NONCANONICAL_PREFIX`` /
    ``ErrorKind.NONCANONICAL_ACTION``. ``rows`` here is the same list of
    dicts ``main`` is about to hand to ``write_rows`` - the artefact an
    acceptance gate reads, while the stdout summary is the other
    representation of this same quantity. A
    mismatch means this pass miscounted its own findings, not merely that
    two cosmetic renderings differ, so it raises rather than warns.
    """
    n_prefix_rows = sum(
        1 for r in rows if r.get("error_kind") == ErrorKind.NONCANONICAL_PREFIX.value
    )
    n_action_rows = sum(
        1 for r in rows if r.get("error_kind") == ErrorKind.NONCANONICAL_ACTION.value
    )
    if summary["n_detector_prefix_failures"] != n_prefix_rows:
        raise DetectorCountMismatch(
            f"summary n_detector_prefix_failures={summary['n_detector_prefix_failures']} "
            f"!= {n_prefix_rows} output row(s) with "
            f"error_kind={ErrorKind.NONCANONICAL_PREFIX.value!r}"
        )
    if summary["n_detector_action_failures"] != n_action_rows:
        raise DetectorCountMismatch(
            f"summary n_detector_action_failures={summary['n_detector_action_failures']} "
            f"!= {n_action_rows} output row(s) with "
            f"error_kind={ErrorKind.NONCANONICAL_ACTION.value!r}"
        )


def write_rows(rows: list[dict], out_path: str) -> None:
    """Write the per-(info-set, action) CSV under an explicit schema.

    polars' default infer_schema_length=100 reads the dtype of
    `error`/`error_kind` from the first 100 rows only; on an arm whose first
    failure lands past row 100 the column is inferred Null and appending a
    str raises ComputeError (100 nulls then a str crashes, 99 then a str
    does not). It fires only on partial success -- i.e. only on the arms
    whose output is most informative.
    """
    pl.DataFrame(rows, schema=_OUTPUT_SCHEMA).write_csv(out_path)


def _parse_token_id_list(s: str) -> list[int]:
    """``argparse`` ``type=`` for ``--expect-generation-prompt-suffix``:
    a malformed value (a non-integer token) is an operator typo, caught at
    argument-parsing time as argparse's own usage error (exit 2, see this
    file's Exit codes table), not a bare ``ValueError`` left to propagate
    uncaught out of `main` and collapse into Python's exit 1.
    """
    try:
        return [int(t) for t in s.split(",") if t.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be a comma-separated list of integers, got {s!r}: {exc}"
        )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exact per-info-set action distribution - offline echo-logprob pass."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data-dir", help="Directory of *.parquet match logs (non-recursive glob)."
    )
    src.add_argument("--parquet", nargs="+", help="Explicit parquet file path(s).")
    p.add_argument(
        "--base-url",
        required=True,
        help="vLLM OpenAI-compatible base URL (e.g. http://localhost:8000/v1 - the "
        "'/v1' suffix, if present, is stripped for /tokenize, which is NOT "
        "under /v1; this script does not call /detokenize).",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model id as served (e.g. Qwen/Qwen2.5-14B-Instruct).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N distinct info-sets (smoke testing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate distinct info-sets and legal actions; make NO network "
        "calls. The arm provenance check (exit 4), the "
        "canonical_prefix_ids_sha256 hash, and the generation-prompt-suffix "
        "measurement (exit 7; also means --expect-generation-prompt-suffix, "
        "if given, is NOT checked) are therefore SKIPPED, and the skip is "
        "announced on stderr - a dry run proves nothing about endpoint/arm "
        "agreement.",
    )
    p.add_argument(
        "--out", default=None, help="Optional CSV path for per-(info-set, action) rows."
    )
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument(
        "--expect-generation-prompt-suffix",
        default=None,
        type=_parse_token_id_list,
        help="comma-separated token ids previously measured for this model. "
        "The pass re-measures and exits 7 on any disagreement, so a "
        "template republished under the same model id cannot silently "
        "change what was scored. Ignored (with a NOTE on stderr) under "
        "--dry-run, which measures nothing to compare it against.",
    )
    return p.parse_args(argv)


def _generation_prompt_suffix_probes(paths: list[str]) -> list[tuple[str, str]]:
    """Selects up to three ``(user_content, action_text)`` probes for
    ``measure_generation_prompt_suffix``, drawn from this run's own resolved
    info-sets rather than synthesised -- so the suffix is measured against
    real prompts and real action literals, the same shape ``score_info_set``
    submits moments later, not the shape a hand-written probe would guess.

    One probe per distinct game, in the order games first appear in
    ``distinct_p0_info_sets``'s output, capped at three. Fewer than three
    successful probes is not an error: what matters for cross-probe
    agreement is different games/literals, not a fixed count -- an arm with
    one or two usable games only offers that many, and the caller's own
    printed line records how many were actually used. This cap of three is
    the number in use; ``measure_generation_prompt_suffix``'s own
    docstring separately records a four-probe bring-up measurement taken
    directly against the live endpoint, not through this selector -- do not
    conflate the two.

    Each probe's action literal is the info-set's own logged
    ``parsed_action`` -- the action this arm's LM (or reference agent)
    actually took there -- resolved through the same ``action_texts_for_
    info_set`` oracle ``score_info_set`` uses (containment check against
    ``prompt`` included), so a probe never submits a literal
    ``score_info_set`` itself would refuse as unparseable.

    A candidate row's own probe-building failing -- ``load_game`` /
    ``get_game_fns_for_game`` raising, ``action_texts_for_info_set`` raising
    ``PromptVocabularyUnparseable``, or the logged ``parsed_action`` missing
    from that info-set's own ``action_labels`` (``KeyError``) -- is exactly
    the same class of per-row failure ``score_info_set`` already handles
    gracefully as a diagnosable CSV row (``GAME_LOAD_FAILED`` /
    ``PROMPT_VOCABULARY_UNPARSEABLE`` / ``ACTION_NOT_IN_ACTION_LABELS``
    respectively). It must not be allowed to
    crash this function and abort the whole run before scoring even starts.
    A failing candidate is skipped as a probe source -- loudly, a "skipping
    ... as a probe source" NOTE on stderr naming the game, info-set and
    reason -- and the next distinct game is tried instead (one candidate row
    per game; a game whose one candidate row fails is not retried against a
    different info-set of the same game). Only if every distinct game's
    candidate fails does this function give up, raising
    ``GenerationPromptSuffixProbesUnavailable`` with every attempt's own
    reason attached -- caught by ``main`` alongside ``measure_generation_
    prompt_suffix``'s own three exceptions and mapped to exit 7, never an
    uncaught traceback.
    """
    reps = distinct_p0_info_sets(paths)
    attempted_games: set[str] = set()
    probes: list[tuple[str, str]] = []
    failures: list[str] = []
    for row in reps.iter_rows(named=True):
        game = row["game"]
        if game in attempted_games:
            continue
        attempted_games.add(game)
        try:
            game_obj = load_game(game, **json.loads(row["game_params"]))
            fns = get_game_fns_for_game(
                game_obj, prompt_spec_for_arm(game, row["probe_type"])
            )
            labels = action_texts_for_info_set(fns, row["info_set"], row["prompt"])
            action_text = labels[row["parsed_action"]]
        except Exception as exc:
            reason = f"{game}/{row['info_set']}: {exc}"
            failures.append(reason)
            print(
                f"NOTE: skipping ({game}/{row['info_set']}) as a generation-prompt-suffix "
                f"probe source: {exc}",
                file=sys.stderr,
            )
            continue
        probes.append((row["prompt"], action_text))
        if len(probes) == 3:
            break
    if not probes:
        tried = (
            "; ".join(failures)
            if failures
            else ("the run's resolved info-set population was empty")
        )
        raise GenerationPromptSuffixProbesUnavailable(
            f"no usable generation-prompt-suffix probe could be built from any distinct "
            f"game in this run ({tried})"
        )
    return probes


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.data_dir:
        paths = sorted(glob.glob(str(Path(args.data_dir) / "*.parquet")))
    else:
        paths = list(args.parquet)
    if not paths:
        print("error: no parquet files found", file=sys.stderr)
        # Not 2: 2 also fires from argparse's own usage error, before
        # main's body runs at all, and the two would be indistinguishable by
        # exit code alone. 9 keeps this script's own "nothing ran" case
        # apart from argparse's.
        return 9

    root = _root_url(args.base_url)

    # Memory level, read from the arm's own run meta before anything touches the network or
    # the rows; a missing run_meta is left to the cardinality check below (exit 5), not judged here.
    try:
        verify_memory_level(_read_run_meta_json(paths))
    except MemoryLevelNotM0 as exc:
        return _refuse_memory_level(exc)

    # Arm provenance: the endpoint about to be queried must
    # match the one that collected this arm, at the granularity the logged
    # Parquet carries (chat_template_hash, model_revision). Runs before
    # run_pass -- there is no point spending the per-info-set call budget
    # against an endpoint already proven not to be the collecting one.
    #
    # SKIPPED under --dry-run. --dry-run's own argparse help promises "make
    # No network calls", and this check is a network call:
    # verify_arm_provenance resolves chat_template_hash live from the
    # endpoint. The operator use case for --dry-run is enumerating an arm's
    # info-sets while the endpoint is down, which is precisely when an
    # unconditional call would abort with exit 4. The skip is announced on
    # stderr rather than made silently (no silent fallbacks): a dry run must
    # not be readable as "the provenance check passed", only as "it was not
    # asked".
    if args.dry_run:
        print(
            "NOTE: --dry-run -- the arm provenance check, the "
            "canonical_prefix_ids_sha256 provenance hash, and the generation-prompt "
            "suffix measurement are all SKIPPED, because all three query the live "
            "endpoint and --dry-run makes no network calls. This run therefore proves "
            "nothing about endpoint/arm agreement, and generation_prompt_suffix is "
            "recorded as a null, not a measured value.",
            file=sys.stderr,
        )
        if args.expect_generation_prompt_suffix is not None:
            # Without this, `--dry-run
            # --expect-generation-prompt-suffix ...` exits 0 and a CI caller
            # scripted against the exit code alone could read that as "no
            # drift", when nothing was measured to compare against at all.
            print(
                "NOTE: --expect-generation-prompt-suffix was given but is NOT checked "
                "under --dry-run -- nothing was measured to compare it against. Exit 0 "
                "here must not be read as 'no drift'.",
                file=sys.stderr,
            )
    else:
        try:
            verify_arm_provenance(
                paths, root, args.model, args.base_url, timeout=args.timeout
            )
        except ArmProvenanceMismatch as exc:
            print(f"\nABORT: arm provenance check failed -- {exc}", file=sys.stderr)
            return 4

    # Prompt version: a scoring run re-renders prompts with the live code, so an arm recorded under
    # another version is refused here, after the endpoint check and before any scoring query. A dry
    # run only counts info-sets and scores nothing, so it is not refused.
    if not args.dry_run:
        try:
            verify_prompt_epoch(_read_run_meta_json(paths))
        except PromptEpochNotLive as exc:
            return _refuse_prompt_epoch(exc)

    # Cardinality: len(paths) must equal the shard count
    # run_meta.json's own resolved_config declares this arm was collected
    # with -- independent of --limit/--dry-run, both of which bound how much
    # of `paths` gets processed, never which files `paths` itself resolves
    # to. Runs before run_pass, same rationale as the provenance check: no
    # point spending the call budget against an arm already proven partial.
    run_meta = _read_run_meta_json(paths)
    try:
        verify_cardinality(paths, run_meta, _read_piece_metas(run_meta))
    except ShardCountMismatch as exc:
        print(f"\nABORT: cardinality check failed -- {exc}", file=sys.stderr)
        return 5

    # The run's own recorded chat_template_kwargs (never a flag someone must
    # remember to pass) -- absent on arms collected without the knob, which
    # is exactly the None default every /tokenize-calling function above
    # treats as "send no chat_template_kwargs".
    chat_template_kwargs = (
        (run_meta or {}).get("resolved_config", {}).get("chat_template_kwargs")
    )

    # Generation-prompt suffix, spec's `G`: the ids the served generation
    # prompt carries beyond the canonical assistant encoding
    # (`measure_generation_prompt_suffix`'s own docstring says what this is
    # and why it is measured live, once per run, never hardcoded. A template
    # that opens an empty thinking channel in its generation prompt
    # measures a non-empty suffix here; every roster model measures
    # `[]`). Measured once per run, here, after the cardinality check and
    # Before run_pass -- same rationale the provenance and cardinality checks
    # give for their own placement above: do not spend the per-info-set call
    # budget against a template whose generation-prompt suffix could not
    # even be established.
    #
    # SKIPPED under --dry-run, for the identical reason the provenance check
    # is (see the NOTE above): measure_generation_prompt_suffix makes
    # Three network calls per probe, and --dry-run's own argparse help
    # promises none. `gen_suffix` is left `None` in that case -- recorded as
    # an actual null in the CSV (see _OUTPUT_SCHEMA's comment), distinguishable
    # from a measured-and-empty `[]` -- and `()` is threaded into `run_pass`
    # instead, which is moot: every InfoSetResult.scores is `[]` under
    # --dry-run (run_pass never reaches score_info_set there), so `()` never
    # reaches assert_canonical_continuation either.
    if args.dry_run:
        gen_suffix: Optional[list[int]] = None
    else:
        # `_generation_prompt_suffix_probes` is inside this try: it can raise
        # GenerationPromptSuffixProbesUnavailable when every candidate row
        # it tried failed for its own reason -- a template-establishment
        # failure exactly like the three measure_generation_prompt_suffix
        # itself raises, so it is caught the same way and mapped to the
        # same exit code, never left to propagate as an uncaught traceback.
        #
        # ChatTemplateUnavailable and bare RuntimeError are also caught here:
        # the primitives this call composes raise them too --
        # render_chat_prefix_ids wraps any internal failure as
        # ChatTemplateUnavailable, and tokenize_action / canonical_
        # assistant_ids / _post_json raise bare RuntimeError on HTTP/URL
        # errors -- so a half-alive endpoint (a 5xx from /tokenize, a
        # mid-run timeout, a context-length rejection on a long probe
        # prompt) must map to exit 7 like every other way the suffix could
        # not be established, not collapse into an uncaught traceback and
        # Python's own exit 1, which would read as "completed with per-row
        # failures". This call sits on the path every arm takes before
        # scoring.
        try:
            probes = _generation_prompt_suffix_probes(paths)
            gen_suffix = measure_generation_prompt_suffix(
                root, args.model, probes, args.timeout, chat_template_kwargs
            )
        except (
            GenerationPromptSuffixUnstable,
            NonCanonicalPrefix,
            NonCanonicalActionTokenisation,
            GenerationPromptSuffixProbesUnavailable,
            ChatTemplateUnavailable,
            RuntimeError,
        ) as exc:
            print(
                f"FATAL: could not establish the generation-prompt suffix: {exc}",
                file=sys.stderr,
            )
            return 7
        except MemoryLevelNotM0 as exc:
            return _refuse_memory_level(exc)
        except ValueError as exc:
            # distinct_p0_info_sets is called here via
            # _generation_prompt_suffix_probes, before any scoring, and raises
            # a bare ValueError when the arm's own logged data violates an
            # invariant this pass rests on -- a null lm_seat, player_to_act
            # disagreeing with lm_seat, or an info-set carrying more than one
            # legal_mask (see distinct_p0_info_sets's own docstring).
            # Uncaught, it would reach the shell as Python's own exit 1 with a
            # traceback -- indistinguishable, by exit code alone, from
            # "completed with per-row failures". Caught and mapped to 8: no
            # traceback, nothing scored, no CSV written.
            print(f"\nABORT: arm data invariant violated -- {exc}", file=sys.stderr)
            return 8
        # Already a list[int] (or None) -- _parse_token_id_list validated it
        # at argument-parsing time, so a
        # malformed value never reaches here as an uncaught ValueError.
        expected = args.expect_generation_prompt_suffix
        if expected is not None and gen_suffix != expected:
            print(
                f"FATAL: generation-prompt suffix drift: measured {gen_suffix!r}, "
                f"expected {expected!r}. The served template changed under a "
                f"model id that did not.",
                file=sys.stderr,
            )
            return 7
        print(
            f"generation-prompt suffix: {gen_suffix!r} (measured from {len(probes)} probes)"
        )

    try:
        # `run_pass` calls `distinct_p0_info_sets(paths)` unconditionally,
        # including under `--dry-run` (which skips the try/except ValueError
        # above entirely, since the generation-prompt-suffix branch is never
        # reached there). Same handling, same exit code.
        results, summary = run_pass(
            paths,
            root,
            args.model,
            limit=args.limit,
            dry_run=args.dry_run,
            timeout=args.timeout,
            gen_suffix=gen_suffix or (),
            chat_template_kwargs=chat_template_kwargs,
        )
    except MemoryLevelNotM0 as exc:
        return _refuse_memory_level(exc)
    except ValueError as exc:
        print(f"\nABORT: arm data invariant violated -- {exc}", file=sys.stderr)
        return 8

    # Full-population check: when no --limit was given, the
    # processed count must equal the arm's full distinct-P0 population.
    # Unaffected by --dry-run -- n_info_sets/n_info_sets_available are both
    # derived from `reps`, never from whether network calls were made.
    # Runs regardless of --limit's value: verify_full_population_processed
    # itself is a no-op whenever limit is not None (see its own docstring
    # for why aborting a deliberate --limit smoke run would be wrong).
    try:
        verify_full_population_processed(
            summary["n_info_sets"],
            summary["n_info_sets_available"],
            args.limit,
        )
    except ShardCountMismatch as exc:
        print(f"\nABORT: cardinality check failed -- {exc}", file=sys.stderr)
        return 5

    # Row-count check: every legal action this run actually touched
    # must appear as exactly one output row. Skipped under --dry-run, whose
    # InfoSetResult.scores is deliberately [] (see verify_row_count_
    # completeness's own docstring) -- not something this check should ever
    # evaluate.
    if not args.dry_run:
        try:
            verify_row_count_completeness(results)
        except ShardCountMismatch as exc:
            print(f"\nABORT: cardinality check failed -- {exc}", file=sys.stderr)
            return 5

    print(
        f"distinct info-sets: {summary['n_info_sets']}  "
        f"calls scored: {summary['n_calls']}  failures: {summary['n_failures']}  "
        f"modal disagreements: {summary['n_modal_disagreements']}  "
        f"detector prefix failures: {summary['n_detector_prefix_failures']}  "
        f"detector action failures: {summary['n_detector_action_failures']}"
    )
    if summary["n_failures"]:
        print("FAILURES (loud, counted - never silently dropped):", file=sys.stderr)
        for r in results:
            for s in r.scores:
                if s.error is not None:
                    print(
                        f"  [{s.game}/{s.info_set}/action={s.action}] {s.error}",
                        file=sys.stderr,
                    )
    if summary["n_modal_disagreements"]:
        # Warn, not gate (see run_pass's comment) - listed, but does not
        # affect n_failures or the exit code. It must never be visible only
        # in the CSV.
        print(
            f"MODAL DISAGREEMENTS ({summary['n_modal_disagreements']}, this pass's own exact "
            "modal action vs the arm's logged empirical modal action - a scientific finding, "
            "not an instrument failure; not counted in FAILURES or the exit code):",
            file=sys.stderr,
        )
        for r in results:
            if r.modal_agreement is False:
                print(
                    f"  [{r.game}/{r.info_set}] exact_modal_action={r.exact_modal_action} "
                    f"vs logged_modal_action={r.logged_modal_action} "
                    f"({r.logged_modal_action_count}/{r.logged_n_rows} logged rows)",
                    file=sys.stderr,
                )

    # Deliberately outside `if args.out:`. The two representations of the
    # detector counts (this run's own inline summary counters vs the
    # error_kind column of the rows) must agree whether or not a CSV is
    # written: without --out, the stdout summary is the artefact a reader
    # takes the numbers from.
    # `base_rows` below is exactly what `if args.out:` writes, minus the
    # provenance keys merged on top -- those keys are disjoint from
    # `error_kind` (see _OUTPUT_SCHEMA), so checking here checks the written
    # rows too.
    base_rows: list[dict[str, Any]] = [
        dataclasses.asdict(s) for r in results for s in r.scores
    ]
    try:
        verify_detector_counts_consistent(summary, base_rows)
    except DetectorCountMismatch as exc:
        print(
            f"\nABORT: detector count consistency check failed -- {exc}",
            file=sys.stderr,
        )
        return 6

    if args.out:
        # Provenance: a CSV destined for publication must be able to say
        # which endpoint and which commit produced it - the per-row
        # backend/model-revision/seed/git-sha requirement, applied to this
        # offline pass's own output.
        #
        # canonical_prefix_ids_sha256: the live-resolved
        # prefix_ids hash for one named canonical info-set per game -- the
        # shape this pass actually scores, narrower than CHAT_TEMPLATE_
        # PROBE's four-turn-with-system shape (see verify_arm_provenance's
        # docstring). It is recorded so a later re-run over this same arm
        # has something to compare against.
        #
        # arm_revision_provenance: the
        # arm's own logged model_revision (harness git SHA at collection
        # time) plus, if a weights_provenance.json sidecar exists, the
        # separately-inferred served-weights revision. Neither gates -
        # see arm_revision_provenance's own docstring for why model_revision
        # cannot be compared against anything live.
        #
        # generation_prompt_suffix: `repr(gen_suffix)` -- the same value measured once, above, and threaded into
        # every `assert_canonical_continuation` call this run made. Unlike
        # its two neighbours above, this one does gate: a disagreement with
        # --expect-generation-prompt-suffix already returned 7 before this
        # block was reached, so every row this block writes carries the
        # measured value the run was actually scored against. `None` here
        # (an actual null, not the string "None") only under --dry-run,
        # where `rows` below is always empty anyway (see run_pass) - there
        # is no row for it to ever appear on regardless of this value.
        #
        # population_summary: the arm's
        # distinct info-set population THIS RUN processed (per-game counts
        # plus a stable key-set hash), written as data so a downstream
        # consumer can compute a cross-arm intersection over whatever arms
        # are in its figure, without re-reading either arm's Parquet - see
        # population_summary's own docstring for why (cross-arm populations
        # are endogenous and differ substantially).
        #
        # run_limit / n_info_sets_available: make a --limit smoke run's partiality travel with
        # the artefact rather than being caught only at abort time (which
        # verify_full_population_processed deliberately does not do when
        # --limit is set - a smoke run is legitimate). `run_limit` is the
        # raw --limit value (None when not given); `n_info_sets_available`
        # is the arm's full distinct-P0 population (run_pass's pre-limit
        # reps.height). Any consumer of THIS CSV THAT treats it as A
        # COMPLETE ARM MUST CHECK `run_limit is not NULL` (equivalently,
        # `population_n_info_sets < n_info_sets_available`) first - a small,
        # clean-looking CSV from a `--limit 5` smoke run carries no other
        # visible marker that it is partial.
        #
        # --dry-run reaches this `if args.out:` block too (nothing forbids
        # combining --dry-run with --out), but every InfoSetResult.scores is
        # deliberately [] under --dry-run (see run_pass), so `rows` below is
        # always empty regardless of these two fields' values - there is no
        # separate row for them to ever appear on. No dry-run-specific
        # branch is added here for exactly that reason: it would be dead
        # code, not a real gap.
        population = population_summary(results)
        # canonical_prefix_ids_sha256
        # calls render_chat_prefix_ids, i.e. /tokenize, once per game. Under
        # --dry-run that violates the flag's own documented "make no network
        # calls" contract, so it is not called at all; the empty dict is
        # what gets recorded. `rows` is always empty under --dry-run anyway
        # (every InfoSetResult.scores is [], see run_pass), so no row ever
        # carries either value -- the difference is purely whether the call
        # is made.
        prefix_ids_hashes = (
            {}
            if args.dry_run
            else canonical_prefix_ids_sha256(
                paths,
                root,
                args.model,
                timeout=args.timeout,
                chat_template_kwargs=chat_template_kwargs,
            )
        )
        provenance = {
            "model": args.model,
            "base_url": args.base_url,
            "run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "canonical_prefix_ids_sha256": json.dumps(
                prefix_ids_hashes, sort_keys=True
            ),
            "arm_revision_provenance": json.dumps(
                arm_revision_provenance(paths),
                sort_keys=True,
            ),
            "generation_prompt_suffix": (
                None if gen_suffix is None else repr(gen_suffix)
            ),
            "population_n_info_sets": population["n_info_sets"],
            "population_per_game_counts": json.dumps(
                population["per_game_counts"],
                sort_keys=True,
            ),
            "population_key_set_sha256": population["key_set_sha256"],
            "run_limit": args.limit,
            "n_info_sets_available": summary["n_info_sets_available"],
        }
        # `base_rows` was already checked against the summary counters above
        # (verify_detector_counts_consistent); the provenance keys merged on here
        # cannot change `error_kind`, so no second check is needed.
        rows: list[dict[str, Any]] = [{**br, **provenance} for br in base_rows]

        write_rows(rows, args.out)
        print(f"wrote {len(rows)} (info-set, action) rows to {args.out}")

    return 1 if summary["n_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
