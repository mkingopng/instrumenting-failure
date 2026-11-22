"""LM agent backed by a vLLM OpenAI-compatible server."""

from __future__ import annotations
import hashlib
import math
import random
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Optional, cast
import numpy as np
import pyspiel
from openai import APIConnectionError, APIStatusError, OpenAI
from lmopt.agents.base import SeatAware
from lmopt.logging.call_log import CallLog
from lmopt.prompting import GamePromptFns, get_game_fns_for_game
from lmopt.prompting.memory import (
    HandMemory,
    MEMORY_CHAR_BUDGET,
    MEMORY_RUNGS,
    MemoryTerms,
    render_memory,
    truncate_memory_block,
)
from lmopt.prompting.spec import (
    GamePromptSpec,
    parse_belief_salvaging,
    prompt_spec_hash,
    split_reasoning,
)

# Version of the prompting code (src/lmopt/prompting/*.py) that this agent's
# rendered prompts reflect; it feeds the `prompt_schema_version` column
# (schema.py). Value 1 denotes the prompt schema with leduc rank-space
# private/public card naming, full cross-round betting history, and dark hex
# move-count and own-move-history rendering.
#
# Value 2 covers: (1) prompt content extracted out of the per-game render
# functions into prompting/spec.py's per-game GamePromptSpec data, which
# exposes `prompt_spec_hash()` as the content-drift detector; (2) belief
# exemplars re-keyed from real card/cell names to placeholder tokens with
# per-game/per-variant slots (e.g. `<rank_A>`, `<card_{i}>`, `<cell_{i}>`),
# so a verbatim echo of the exemplar cannot parse as a valid belief; (3) a
# separate belief-call token budget (`BELIEF_MAX_TOKENS`); (4) action/belief
# parsers across all five games reading the same extracted vocabulary spec.py
# owns, with stricter multi-label/ambiguous-match handling; and (5) every
# game's `belief_question` carrying a precision/format instruction sentence.
# This is a prompt-content schema version only, not a protocol change: free
# generation (not constrained/guided decoding) is unchanged by it. The
# fine-grained content identity is `spec.prompt_spec_hash()`; this integer is
# the coarse collection-era marker.
#
# Change-detector consequence: the earliest landed Qwen-14B arm has no
# `prompt_schema_version` column at all, so it is a cross-schema reference.
# `pilot_gate.py`'s `compare_to_reference` 0.05 rise-block
# (`max_reference_rise`) is a valid change-detector only within a single
# schema version; per-game deltas it reports against that arm must be
# attributed to the accumulated prompt/schema changes since that run, not
# read as a same-schema regression signal.
#
# Bump this when a prompting/*.py change alters what a rendered prompt
# Contains at a fixed info-set (not mere wording) -- i.e. a change that
# could shift the measured action/belief distribution -- so a Parquet
# reader can tell which prompting code produced any given row without
# having to reconstruct it from a git-log/timestamp join. This is the
# single source of truth; schema.py's docstring for the column points back
# here rather than restating the value.
# v3: the belief questions ask for the posterior, which is what they are
# scored against. A question that describes only the dealing process while
# `true_posterior` conditions on observed play makes a model that answers the
# question correctly score as badly calibrated. Every game names the evidence
# the prompt already renders (the bid history, the cross-round betting, the
# action history, the round outcomes, the discovered stones). Which posterior
# P7a should target is deliberately not decided here.
# v5: memory-rung verbosity gate. liarsdice + goofspiel action_suffix gain an
# "action_only_clause"; kuhn/leduc/darkhex unchanged.
#
# v6: `prompt_spec_hash()` hashes all five canonical GamePromptSpec objects
# Together, so combining two independent content changes -- v5's
# GOOFSPIEL_SPEC/LIARSDICE_SPEC `action_only_clause`, and KUHN_SPEC/LEDUC_SPEC
# templating (`instruction["rules"]`/`belief_question` take
# `{num_ranks}`/`{rank_range_desc}`/`{num_cards}`/etc. placeholders, filled by
# kuhn.py/leduc.py at canonical values and by kuhn_deep.py/leduc_deep.py at
# their chosen numRanks) -- produces one new hash, registered as one entry.
# Canonical kuhn/leduc rendered BYTES are unchanged by the templating;
# the hash moves because it is sensitive
# to the source literal, not the render (point 2 below).
#
# v7: the prompt rewrite. All five games change: the rules prose states the
# objective and every magnitude the engine actually pays out, the belief
# question asks for the posterior in three numbered steps and names its keys
# in prose instead of in a parseable object, and the action call ends in one
# shared legal-response sentence built by `spec.legal_response_suffix`. Every
# arm collected under an earlier version is incomparable with this epoch.
# Goofspiel's round-history line reports each bid as the card bid (not the
# engine's action index), names the round's winner from the acting seat's
# view, and renders a tie explicitly. That rendering lives in goofspiel.py,
# which the spec hash does not cover, so the v7 hash does not reflect it while
# the goofspiel render pins do.
PROMPT_SCHEMA_VERSION = 7


# The version -> prompt_spec_hash() registry. Every entry records the exact
# `spec.prompt_spec_hash()` that the corresponding `PROMPT_SCHEMA_VERSION`
# denotes, so the coarse integer above is a checkable assertion about the
# prompt rather than an unverifiable one: a content change that does not bump
# the version fails the pin test.
#
# Limitation: this is a TRIPWIRE, not a gate. Two ways it fails to bind:
#   1. The pin is editable in place. A change to prompt content can satisfy
#      the test by pasting the new digest over an existing entry instead of
#      adding a new one, and the failing assertion prints the new digest.
#      What this buys is that the edit becomes deliberate and reviewable in a
#      diff. The un-editable version pins against the hash recorded in landed
#      shards.
#   2. It is under-sensitive by construction. `prompt_spec_hash()` covers only
#      the four `GamePromptSpec` fields. Mutating `kuhn._describe_history`
#      changes the rendered prompt ('- History: Player 0 bet.' -> '- History:
#      Mutated history.') and leaves the hash byte-identical. `build_prompt`'s
#      scaffold, CARD_NAMES, the situation sentences and `action_suffix`'s
#      wrapper text are all outside it. This catches spec-content drift; it
#      does not catch rendering drift.
#
# Append, never edit: a new content version adds a new key. The `2` entry is the
# hash the three landed v2 arms (mistral-nemo, olmo-2-13b, qwen2.5-14b, 100
# shards each) were collected under, confirmed by recomputing at both git
# shas their run_meta.json records.
PROMPT_SPEC_HASHES: dict[int, str] = {
    2: "e3b19c4625607e5a97c512730ba597dce4ca0fe91472c4785b6edd5b11f87884",
    # v3: the belief-question rewrite. Appended, not substituted for the v2
    # entry -- v2 still denotes exactly what the three landed arms were
    # collected under, and overwriting it would relabel collected data.
    3: "9efb361ea22f66321a6fda82fec56ac2c165f64b358000aa8182a4bd7934946b",
    # v4: opponent-model sentence + non-uniform exemplars. Appended.
    4: "675c0cc20de9868d56edfc0d21711fe81c754cd3140fbaeec2888d37c7693578",
    # v5: memory-rung verbosity gate. liarsdice + goofspiel action_suffix gain
    # an "action_only_clause" instructing the model to suppress explanatory
    # prose; kuhn/leduc/darkhex unchanged. Appended per the same convention as
    # v3/v4.
    5: "bc4219676b08b9519c926918be64e82817e025232aec8000f77cd647f73a54cd",
    # v6: the combined kuhn/leduc templating and v5 content (see
    # PROMPT_SCHEMA_VERSION's comment above for why this is one entry) --
    # computed via `.venv/bin/python -c "from lmopt.prompting.spec import
    # prompt_spec_hash; print(prompt_spec_hash())"` against the combined tree.
    6: "543591646b6a83e08c4c196e0ec93af2e89ec60ff094ea6baf8cc98f6728b3b3",
    # v7: the prompt rewrite. Every game's rules prose states the
    # objective and every magnitude it can be scored against (Kuhn's ante and both
    # payoffs, Leduc's ante and per-round bet sizes, Goofspiel's point-difference
    # return, Liar's Dice's wild six and its maximum bid, Dark Hex's connection
    # payoff); every belief question asks for the posterior in the same three
    # numbered steps and names its keys in prose rather than in a parseable object;
    # and every action call closes with one shared legal-response sentence.
    # Appended per the same convention as v3/v4/v5/v6 -- computed via
    # `.venv/bin/python -c "from lmopt.prompting.spec import prompt_spec_hash;
    # print(prompt_spec_hash())"` against this tree.
    7: "3dcdecc6b754d0e736d934ae5e9944d04477967e75289a188b35f637d5151f9a",
}


# The belief call's own token budget, separate from the action call's
# `max_tokens` (`ACTION_MAX_TOKENS`). A budget sized too tightly truncates a
# correct multi-key JSON belief mid-object before the closing brace, and
# `parse_belief`'s first-`{...}` extraction then finds no complete object and
# returns None for a model that answered correctly. Some models emit values
# at full float precision (17 significant digits, e.g.
#   {"1": 0.16666666666666666, "2": 0.16666666666666666, "3": 0.16666666666666666, "4": 0.16666666666666666, "5": 0.16666666666666666, "6": 0.16666666666666667}
# -- 155 characters, ~130 tokens at ~1.19 chars/token), so a 128-token budget
# truncates even a terse six-key liarsdice answer.
#
# The budget is not sized for terseness: models are locally hosted, there is
# no cost reason to cap output, and a budget sized for terseness silently
# truncates a model that reasons before answering. The study's interest is in
# how beliefs guide decisions under uncertainty, so a reasoning trace at the
# belief step is data, not noise to be suppressed.
#
# A larger cap does not break comparability with the three landed v2 arms
# (collected at 256): over all 661,644 non-null belief rows belief_failed is 0
# on every arm and p99 completion length is 66-78 characters against a
# 256-token cap. The single longest belief in the corpus (mistral, 822 chars)
# ends with a complete ```json fence and parsed successfully, so it is a verbose
# answer and not a truncation. No landed belief approached the cap, therefore a
# larger cap admits only outputs that were never being cut and leaves every
# existing row's generation unchanged.
#
# 2048 sits at the bottom of the 2-8k that configs/models.yaml sizes for
# `Olmo-3.1-32B-Think`, and deliberately so: the ladder's smallest window is
# OLMo-2-13B's 4096 (its trained maximum), which the prompt already consumes
# several hundred tokens of, so the top of that range does not fit. Per-model
# budgets live in configs/models.yaml; this module-level constant is the
# roster-wide floor, not the thinking-model answer.
#
# Kept strictly greater than ACTION_MAX_TOKENS below, and that is load-bearing
# for testing rather than merely tidy: a test can discriminate
# the two call sites only by their budget, so collapsing them to one value would
# leave it passing even if the action
# call wrongly reused the belief budget.
BELIEF_MAX_TOKENS = 2048


# The action call's budget, and the `max_tokens` constructor default, sized
# for the reasons given on BELIEF_MAX_TOKENS above. A named constant rather
# than a bare literal in the constructor signature, so any diagnostic can
# Import it instead of
# repeating the value and asserting by hand that the two match, which is
# exactly the arrangement that lets a diagnostic silently stop matching the
# production path it exists to diagnose.
ACTION_MAX_TOKENS = 1024


# The optional `{reasoning, belief}` decoding protocol for the belief call.
# Off by default; see the `belief_protocol`
# constructor kwarg for why the default is load-bearing.
#
# Named, not `BELIEF_SCHEMA`. Two reasons.
# (1) `scripts/pilot_gate.py` already owns `belief_schema_present` for an
# unrelated concept -- whether a Parquet carries the `belief_failed` column --
# so `belief_schema` here made one grep return a gate's schema-presence check
# and an agent's decoding protocol. (2) This object is one protocol among
# several: a belief-only schema is a third protocol with its own answer at the
# same info-set, so the selector is a name (`BELIEF_PROTOCOLS` below), never a
# boolean.
#
# `reasoning` is listed first by intent -- the model writes its inference
# before committing to numbers, rather than emitting a posterior and then
# rationalising it. That is an intent about generation order and is not a
# guarantee about the decoded response: key order in a response is a property
# of the decoder, and `prompting/spec.py:extract_belief_object` is written to
# read the "belief" key in either order for exactly that reason. Nothing
# downstream may assume this ordering.
#
# Field order. The generation-order argument alone decides it: a posterior
# emitted before any inference is a different construct from the one this
# protocol exists to measure.
#
# Truncation cost does not discriminate between the orders: the fail-closed
# unterminated-object guard (rule (2) of `prompting/spec.py`'s
# `extract_belief_object`, which numbers the four rules) fires on a truncated
# Outer object in either field order, so both orders return None ->
# `belief_failed=True`, a recorded finding.
#
# `additionalProperties: False` so nothing else appears, which is what keeps
# the reasoning a sibling string rather than prose wrapped around the object.
# That distinction is the whole point: the reasoning trace is data, and
# `last_belief_raw_completion` logs the full first-attempt text verbatim, so
# the trace survives into the Parquet rather than being parsed away.
#
# The constraint provably binds on this image, and that had to be measured,
# not assumed. `guided_choice` is accepted (HTTP 200) and silently ignored
# here, so a structured-output path with no sentinel is a check that cannot
# fail. Against vLLM serving a small model, a schema
# requiring a nonsense `zzq_sentinel` integer returned `{"zzq_sentinel": 1}`,
# and the negative control -- the identical prompt with no `structured_outputs`
# -- returned prose containing no such key, which is what makes the positive
# result evidence of binding rather than of the model being agreeable. This
# schema itself then returned `finish_reason=stop` at 534 completion tokens
# with keys `['reasoning', 'belief']`.
#
# Falsification condition for that measurement (it licenses trusting a
# structured-output path, so it must state what would revoke it): any change
# of vLLM image/version, of the served model, or of the `structured_outputs`
# request key. Re-run the sentinel with its negative control before trusting
# any arm collected under this protocol. A bind check without the negative
# control does not count.
#
# Sizing note: an unconstrained-length `reasoning` string can be truncated by
# `BELIEF_MAX_TOKENS`, and a truncated response under this schema is invalid
# JSON (an unterminated string), not a short-but-parseable object. The same
# call at max_tokens=200 returned unterminated JSON; at 2048 it returned 534
# tokens and stopped cleanly. The belief call sends BELIEF_MAX_TOKENS = 2048;
# a thinking model may still need more (per-model budgets). Truncation is
# diagnosable offline from `finish_reason`, and because
# `belief_raw_completion` keeps the first attempt's text verbatim: a
# truncated schema row is `belief_failed=True` with a completion that does
# not end in '}'.
REASONING_BELIEF_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "belief": {"type": "object", "additionalProperties": {"type": "number"}},
    },
    "required": ["reasoning", "belief"],
    "additionalProperties": False,
}


# The belief decoding protocol, a manipulated variable: run-level config
# (`configs/run_experiment.yaml: belief_protocol`), and stamped on every LM
# row as the `belief_protocol` Parquet column (logging/schema.py).
#
# The column is the point, not the config key. `write_run_meta`
# (logging/run_meta.py) already embeds the whole resolved config, which makes
# the protocol recoverable PER RUN -- but a per-matchup override or a mid-arm
# change would then leave nothing behind, and a row whose decoding protocol is
# unrecoverable cannot be compared with any other row. One goofspiel
# info-set under three protocols: free
# `{.1,.4,.4,.1}`, belief-only schema `{.33,.33,0,.33}`, reasoning+belief
# schema `{.2,.3,.1,.4}`. Three protocols, three posteriors, same info-set.
#
# Adding a protocol means adding a key here and nothing else in the agent: the
# name is the logged value, and the mapped value is what reaches the wire, so
# the two cannot drift. A name absent from this mapping is REFUSED at
# construction rather than defaulted (see __init__) -- a typo that silently
# collected an arm under free generation while the column claimed otherwise is
# strictly worse than no column at all.
BELIEF_PROTOCOL_FREE = "free"
BELIEF_PROTOCOL_REASONING_BELIEF = "schema_reasoning_belief"

BELIEF_PROTOCOLS: dict[str, Optional[dict]] = {
    # free generation: no `structured_outputs` on the wire at all, byte-
    # identical to every landed arm. The default, and the null arm the
    # structured protocols are compared against.
    BELIEF_PROTOCOL_FREE: None,
    BELIEF_PROTOCOL_REASONING_BELIEF: REASONING_BELIEF_JSON_SCHEMA,
}


# The two elicitation_mode values this
# agent actually implements -- `act` branches on these two below (the
# `elif self._elicitation_mode == "sibling"` and
# `elif self._elicitation_mode == "combined"` branches); anything else falls
# to the bare `else`, which silently disables belief elicitation for the
# whole arm rather than failing. Lives
# here, beside BELIEF_PROTOCOLS, rather than in either script that validates
# a run's config: this is the single source of truth `run_sweep.py`
# imports, so a config-validation whitelist can never drift
# from what this agent actually branches on.
SUPPORTED_ELICITATION_MODES = ("sibling", "combined")


def validate_elicitation_mode(mode: str) -> None:
    """Raise ValueError naming the bad value and the supported set if `mode`
    is not one this agent implements (no silent fallbacks).

    Library-level: raises rather than exiting the process, so a script-level
    caller (`run_sweep.py`) translates the exception into its
    own `sys.exit` with its own call-site framing, instead of this module
    reaching into process control on their behalf.
    """
    if mode not in SUPPORTED_ELICITATION_MODES:
        raise ValueError(
            f"Unsupported elicitation_mode: {mode!r} "
            f"(expected one of {SUPPORTED_ELICITATION_MODES!r})"
        )


def validate_memory_rung(rung: str) -> None:
    """Raise ValueError if `rung` is not a valid memory rung (no silent fallbacks).

    Library-level: raises rather than exiting the process, matching validate_elicitation_mode.
    """
    if rung not in MEMORY_RUNGS:
        raise ValueError(
            f"Unsupported memory_rung: {rung!r} " f"(expected one of {MEMORY_RUNGS!r})"
        )


def derive_request_seed(
    run_seed: int, hand_idx: int, node_idx: int, call_kind: str, attempt: int
) -> int:
    """Deterministically derive the per-request `seed` sent to the vLLM API.

    Keyed on ``(run_seed, hand_idx, node_idx, call_kind, attempt)``
    -- deliberately not on a call counter. "Derive the seed from (run_id,
    hand_idx, node_idx, call_kind), never a call counter" is right about the
    counter but incomplete as stated. A global
    monotonic counter (e.g. "the Nth request this run has made") is forbidden
    because it makes seed N depend on how many calls happened before it
    anywhere in the run: one extra retry at an earlier node would silently
    shift every subsequent seed and destroy replication for rows that never
    themselves changed. But naively keying only on (run_seed, hand_idx,
    node_idx, call_kind) -- with no `attempt` -- creates a second, opposite
    defect: every retry at the same node with the same call_kind would derive
    the identical seed, redraw the identical sample, and make the retry loop
    a no-op at any temperature. The action path retries specifically because
    attempt 0 produced an illegal action; re-sampling with an identical seed
    re-draws that same illegal action forever.

    The resolution: `attempt` is a local STRUCTURAL index (position within
    one node's retry loop) and is fine to include; a global call counter is
    not, because only the local index is invariant to anything outside its
    own node. Including `attempt` changes nothing about any other node's
    seeds -- it only lets attempt 1 differ from attempt 0 at the same node.
    `hand_idx`/`node_idx` are themselves local structural indices too (see
    LMVllmAgent._hand_idx/_node_idx): they count this agent's own decisions,
    reset per hand, so they do not shift with retries anywhere, including at
    earlier nodes in the same hand (see LMVllmAgent for the invariant that
    makes this true in production).

    `run_seed` is bound to the MatchupSpec/NodeRecord matchup seed (the
    `seed` column, "one shard = one (game, opponent, seed) triple" --
    orchestration/shard.py's own docstring), not match_runner's `run_id`
    (a fresh uuid4 per `run_match()` call, unrelated to reproducibility --
    see LMVllmAgent's `run_seed` constructor kwarg docstring for why).

    Hashed (sha256) rather than passed through Python's built-in `hash()`,
    which is salted per-process for strings unless PYTHONHASHSEED is pinned
    -- this must be reproducible across processes/machines, not just within
    one interpreter run. Reduced to a non-negative value < 2**31 - 1 (a safe
    range for any JSON/int64 client, not merely this one).
    """
    key = f"{run_seed}:{hand_idx}:{node_idx}:{call_kind}:{attempt}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


@dataclass(frozen=True)
class CompletionResult:
    """One `_complete()` call's response, read at the boundary.

    `content` is the parsed-action/belief text
    (`resp.choices[0].message.content or ""`). The other
    three fields are additional reads of the same response object, added
    without changing what is sent on the wire: `reasoning_content` is a
    vLLM/openai extra field (`getattr(..., "reasoning_content", None)`,
    absent -> None, not an error); `finish_reason` and `completion_tokens`
    are the standard OpenAI-schema fields (`choices[0].finish_reason`,
    `usage.completion_tokens`), None when the server omits `usage` or sends
    a null `finish_reason`. Frozen: a call's response is a fact, not
    something a caller should be able to mutate after the fact.

    Three more reads of the same response and one clock read, all
    defaulted so a caller that builds a result by hand keeps working: `latency_s`
    is wall seconds around the client call only (`time.monotonic`), `prompt_tokens`
    is `usage.prompt_tokens`, `response_id` is the response's own id. The last two
    are None unless the server sent a value of the right type (a string id, an
    integer count that is not a bool): a wrong-typed value becomes None, never a
    coerced one.
    """

    content: str
    reasoning_content: Optional[str]
    finish_reason: Optional[str]
    completion_tokens: Optional[int]
    latency_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    response_id: Optional[str] = None
    top_logprobs: Optional[list] = None


def validate_action_top_logprobs(value) -> Optional[int]:
    """
    check the run-level top-k logprobs setting. None means the request is the one every arm
    collected before this setting existed sent, byte for byte
    :param value: the configured k, or None for off (Any)
    :returns: the validated k, or None (Optional[int])
    :raises ValueError: when the value is neither None nor an integer of at least 1 (a bool
        is refused: True would otherwise read as 1)
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"action_top_logprobs must be None (off) or an integer >= 1, got {value!r}"
        )
    return value


def _typed_float(value) -> Optional[float]:
    """
    keep a value only when it is a finite number; a logprob of minus infinity, a bool or a
    test double's child mock must never reach the call log
    :param value: whatever the response carried for a logprob (Any)
    :returns: the value as a float when it is a finite int or float and not a bool, else None
        (Optional[float])
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return (
        number
        if number == number and number not in (float("inf"), float("-inf"))
        else None
    )


def _field(obj, name: str):
    """
    read one field off a response entry, whether the SDK handed back an object or a mapping
    :param obj: a logprob entry (Any)
    :param name: the field name (str)
    :returns: the field's value, or None when it is absent (Any)
    """
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _read_top_logprobs(choice) -> Optional[list]:
    """
    read the per-token top-k logprobs off a choice. Returns None, never an empty list, when
    the response carries none. Every value is type-checked; a token the server sent with a
    non-finite logprob keeps its token and carries a null logprob
    :param choice: the response's first choice (Any)
    :returns: one {"token", "logprob", "top": [{"token", "logprob"}, ...] or None} per
        generated token (`top` is None when the entry carried no alternatives list), or None
        when the response carries no logprobs (Optional[list])
    """
    logprobs = _field(choice, "logprobs")
    content = _field(logprobs, "content") if logprobs is not None else None
    if not isinstance(content, list) or not content:
        return None
    tokens = []
    for entry in content:
        alternatives = _field(entry, "top_logprobs")
        tokens.append(
            {
                "token": _typed_text(_field(entry, "token")),
                "logprob": _typed_float(_field(entry, "logprob")),
                # None when the entry carried no alternatives list: an empty list would say
                # the server sent none, a null says we could not read one.
                "top": (
                    [
                        {
                            "token": _typed_text(_field(alt, "token")),
                            "logprob": _typed_float(_field(alt, "logprob")),
                        }
                        for alt in alternatives
                    ]
                    if isinstance(alternatives, list)
                    else None
                ),
            }
        )
    return tokens


def _typed_count(value) -> Optional[int]:
    """
    keep a value only when it is an integer count; a bool is not one. A test double
    answers an unset attribute with a child mock and a server can send a null, and
    neither may reach an integer column
    :param value: whatever the response carried for a count (Any)
    :returns: the value when it is an int and not a bool, else None (Optional[int])
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _typed_text(value) -> Optional[str]:
    """
    keep a value only when it is a string
    :param value: whatever the response carried for a text field (Any)
    :returns: the value when it is a str, else None (Optional[str])
    """
    return value if isinstance(value, str) else None


def _compose_trace_and_answer(result: "CompletionResult") -> tuple[Optional[str], str]:
    """(trace, answer) for a `CompletionResult`, per call-site composition.
    The server's own `reasoning_content` takes
    priority when present -- on a `--reasoning-parser` arm the server has
    already split the trace out of `content`, so `content` there is the
    answer, and re-running `split_reasoning` on it would be both redundant
    and (property 6) actively wrong: a stray `</think>` inside an
    already-clean answer must not be treated as a second delimiter.
    `split_reasoning` runs only when `reasoning_content` is None, i.e. on a
    no-parser arm where a `<think>...</think>` block, if any, is still
    embedded in `content` verbatim.

    `parse_action`/`parse_belief`/`extract_belief_object` must see only the
    returned `answer`; the returned `trace` is logged, never parsed.
    """
    trace = result.reasoning_content
    answer = result.content
    if trace is None:
        trace, answer = split_reasoning(result.content)
    return trace, answer


# HandMemory moved to lmopt.prompting.memory to avoid import cycle
# (lm_vllm imports from lmopt.prompting; memory rendering needs HandMemory)
# Re-exported here so existing imports keep working.


#: How many times a request is retried after a transport failure, the openai SDK's own default
#: (`openai._constants.DEFAULT_MAX_RETRIES`, 3.8.0). The agent runs these retries itself so that
#: each one is a logged line, not a silent repeat inside the client.
_TRANSPORT_RETRIES = 2
#: The SDK's backoff (`openai._constants`, 3.8.0): 0.5 s doubling to 8 s, and a server's
#: Retry-After honoured up to 120 s.
_INITIAL_RETRY_DELAY_S = 0.5
_MAX_RETRY_DELAY_S = 8.0
_MAX_RETRY_AFTER_S = 120.0
#: Patched by tests so a retry does not wait.
_sleep = time.sleep


def _retry_after_s(exc: BaseException) -> Optional[float]:
    """
    the server-directed delay on a failed response, as the SDK reads it
    :param exc: the exception the client raised (BaseException)
    :returns: seconds from `retry-after-ms` or `retry-after`, or None when absent or not a number
        (an HTTP-date Retry-After is not read; the SDK parses one, vLLM sends none) (float | None)
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            return float(headers.get(name)) / scale
        except (TypeError, ValueError):
            continue
    return None


def _sdk_would_retry(exc: BaseException) -> bool:
    """
    whether the openai SDK 3.8.0 would retry this failure: a timeout or any other transport error
    (`_base_client.py` `_request`, the `httpx.TimeoutException` and `Exception` branches), or a
    response it accepts by `_should_retry` (`_base_client.py:815-857`): an `x-should-retry` header
    decides first, a Retry-After above two minutes refuses, then 408, 409, 429 and 5xx retry
    :param exc: the exception the client raised (BaseException)
    :returns: True when the SDK would have retried it (bool)
    """
    if isinstance(exc, APIConnectionError):
        return True
    if not isinstance(exc, APIStatusError):
        return False
    after = _retry_after_s(exc)
    if after is not None and math.isfinite(after) and after > _MAX_RETRY_AFTER_S:
        return False
    flag = exc.response.headers.get("x-should-retry")
    if flag in ("true", "false"):
        return flag == "true"
    status = exc.status_code
    return status in (408, 409, 429) or status >= 500


def _sdk_retry_delay(exc: BaseException, retries_taken: int) -> float:
    """
    the wait before the next attempt, by the SDK's rule (`_calculate_retry_timeout`)
    :param exc: the exception the client raised (BaseException)
    :param retries_taken: retries already made for this request (int)
    :returns: seconds (float)
    """
    after = _retry_after_s(exc)
    if after is not None and math.isfinite(after) and 0 < after <= _MAX_RETRY_AFTER_S:
        return after
    base = min(_INITIAL_RETRY_DELAY_S * 2.0**retries_taken, _MAX_RETRY_DELAY_S)
    return max(base * (1 - 0.25 * random.random()), 0.0)


class LMVllmAgent(SeatAware):
    """Calls a vLLM-served open-weight model for action selection and belief elicitation.

    Implements the Agent protocol.  After each act() call, the following
    instance attributes are updated for the match runner to log:

        last_prompt:          str  - full action prompt (base + action suffix)
        last_raw_completion:  str  - first action response from the model
        last_n_retries:       int  - extra calls made after the first attempt
        last_fallback_used:   bool - True if all retries exhausted; action = legal[0]
        last_elicited_belief: Optional[str] - JSON belief string from sibling call
        last_belief_raw_completion: Optional[str] - first belief-call attempt's
            raw text. Non-null iff a belief call was actually
            made; None when belief_suffix skipped the question entirely.
        last_belief_n_retries: int - extra belief-call attempts after the
            first. 0 for a skip, a first-try success, or any
            non-LM agent.
        last_belief_failed: bool - True only when a belief call was made,
            every attempt failed to parse, and no recovery supplied a belief.
            False on a skip, so "never asked" and "asked and
            unparseable" stay distinguishable downstream and a skip cannot
            manufacture a failure statistic.
        last_belief_salvage_kind: Optional[str] - "key_quoted" when every
            strict attempt failed and the bare-integer-key recovery supplied
            `last_elicited_belief` (with `last_belief_failed` False); None
            otherwise.
        last_belief_salvage_attempt: Optional[int] - the belief attempt
            (0-based) whose text the recovery used; None unless
            `last_belief_salvage_kind` is set.

    Also exposed for per-run provenance logging (additive to
    match_runner's existing model_revision=git_sha, not a redefinition of it):

        model_id:             Optional[str] - the served model identifier
            (this agent's own `model` constructor arg).
        chat_template_hash:   Optional[str] - accepts a caller-supplied hash
            of the chat template the served endpoint renders prompts under.
            The sweep path passes it: `scripts/run_sweep.py`'s
            `_resolve_serving_block` resolves it once at sweep startup (the
            same /tokenize+/detokenize round trip
            `scripts/exact_policy_pass.py`'s `render_chat_prefix_ids` uses,
            never a second, independently-invented mechanism) and threads it
            through `MatchupSpec.chat_template_hash` ->
            `orchestration/shard.py`'s `_build_agent0` -> here, at
            construction (not per request: `_complete` sends untemplated
            messages and lets the vLLM server apply its own template
            server-side, so there is no in-agent template text to hash on
            every call -- construction-time is the granularity this
            parameter offers).
            Left None (logged as null, matching the LM-only-columns
            null-for-non-LM pattern) at any construction site that omits
            it, rather than defaulting to any fabricated string (no silent
            fallbacks).
        served_weights_revision: Optional[str] - the served model's HF
            weights revision,
            caller-supplied at construction on the same rule as
            chat_template_hash above: the sweep path passes the
            launcher-resolved value (run_sweep's --served-weights-revision
            -> MatchupSpec -> shard._build_agent0), and None is logged as null
            rather than fabricated.
        tokenizer_revision: Optional[str] - the served endpoint's HF
            tokenizer revision, caller-supplied at construction on the identical rule as
            served_weights_revision above.
        prompt_schema_version: Optional[int] - module-level
            PROMPT_SCHEMA_VERSION above, stamped once at construction.
        belief_protocol: str - the belief decoding protocol this agent was
            constructed with, a key of BELIEF_PROTOCOLS. Stamped
            on every LM row's `belief_protocol` column. Non-null on every LM
            row including belief skips (belief_suffix is None) and combined
            mode: it describes how this agent decodes a belief call, which is
            a property of the run, not of whether one happened at this
            info-set -- "was a belief asked here" is already answered by
            belief_raw_completion/belief_failed.

    elicitation_mode:
        "sibling"  - two calls from the same base prompt (default, recommended).
        "combined" - one call that asks for both action and belief.
        Any other value is REFUSED at construction.

    belief_protocol: str: a protocol name. `"free"` (default) leaves the
        belief call as free generation, byte-identical on the wire to every
        landed arm. `"schema_reasoning_belief"` constrains the sibling belief
        CALL only with `REASONING_BELIEF_JSON_SCHEMA`, yielding
        `{reasoning, belief}`; the reasoning is then logged verbatim in
        `last_belief_raw_completion` (the trace is the measurement) and
        `prompting/spec.py:extract_belief_object` reads the `belief` key out
        of it. An unknown name, or any non-free protocol in "combined" mode
        (which has no separate belief call to constrain), is REFUSED at
        construction rather than ignored. Decoding protocol is a manipulated
        variable: see BELIEF_PROTOCOLS and the constructor's own comment.

    run_seed: Optional[int]: when set, every `_complete()` call
        (action attempts and belief attempts alike) passes a per-request
        `seed` to the vLLM API, deterministically derived by
        `derive_request_seed` from (run_seed, this agent's own hand/node
        position, call_kind, attempt) -- see that function's docstring for
        the full reasoning. None (the default) omits `seed` from the API
        call entirely. Bound to the MatchupSpec/NodeRecord matchup
        seed, not match_runner's `run_id` -- see the constructor's own inline
        comment for why.
    """

    requires_model = True

    def __init__(
        self,
        client,  # openai.OpenAI instance
        model: str,
        game_name: str = "kuhn",  # informational only - reset() resolves _fns
        elicitation_mode: str = "sibling",
        max_retries: int = 3,
        backend: str = "cuda",
        temperature: float = 0.0,
        # ACTION_MAX_TOKENS (see that constant for its rationale).
        # Comparability with the three landed v2 arms (collected at a 64-token
        # cap) holds by measurement: action completions across all 661,644
        # non-null rows are max 9 characters (p99.9 = 9, zero rows >= 190
        # chars), so nothing was ever truncated and a larger cap cannot change
        # a single landed generation.
        max_tokens: int = ACTION_MAX_TOKENS,
        # The sibling belief call's decode budget. A run-level knob for the
        # same reason `max_tokens` above is one: a thinking model spends its
        # cap on the trace before the answer (OLMo Think can consume all 1024
        # action tokens without closing its think block), and a larger budget
        # must be an explicit, recorded run-config choice, never a constant
        # edit that would silently retime every arm. Default references the
        # constant, so an unconfigured run is byte-identical on the wire.
        belief_max_tokens: int = BELIEF_MAX_TOKENS,
        # Reasoning knob (e.g. Qwen3): vLLM extra_body kwargs
        # threaded to both action and belief calls. A run-level knob for
        # toggling reasoning-enabled chat templates on a model serving
        # a reasoning-capable base (one Qwen3-32B serving, two recorded
        # arms: think vs non-think, distinguished only by this dict). Default
        # None is byte-identical on the wire (no key sent), so every existing
        # arm and every unconfigured run carries no extra_body entry for this.
        chat_template_kwargs: Optional[dict] = None,
        # Run-level top-k logprobs on the action call only. None (the default, and every
        # arm collected before this existed) leaves the request byte-identical; an integer k
        # adds `logprobs=True` and `top_logprobs=k` to action requests and nothing else.
        # The values are stored in the per-call log, never on the row.
        action_top_logprobs: Optional[int] = None,
        chat_template_hash: Optional[str] = None,
        # The HF weights
        # revision of the model the endpoint serves, resolved by the
        # Launcher (never in-agent -- the value lives in the serving box's
        # HF cache, which this client cannot see) and threaded here exactly
        # like chat_template_hash above: run_experiment resolves ->
        # run_sweep --served-weights-revision -> MatchupSpec ->
        # shard._build_agent0 -> this kwarg -> match_runner's getattr ->
        # the row. None (default; every path with no resolved value)
        # logs as null, never a fabricated placeholder
        # (no silent fallbacks).
        served_weights_revision: Optional[str] = None,
        # Tokenizer revision: same thread, same reasoning as
        # served_weights_revision above.
        tokenizer_revision: Optional[str] = None,
        run_seed: Optional[int] = None,
        # A protocol name, not a boolean (see BELIEF_PROTOCOLS for both
        # reasons). Selects how the sibling belief
        # CALL only is decoded; a non-free protocol is sent as a vLLM
        # `structured_outputs` constraint. The action call is never
        # constrained: an action is a bare label and stays free-generation.
        #
        # Default "free" is load-bearing, not a courtesy. Decoding
        # protocol is a manipulated variable, not a neutral wrapper: three
        # protocols gave three different beliefs at one goofspiel info-set --
        # free `{.1,.4,.4,.1}`, belief-only schema `{.33,.33,0,.33}`,
        # reasoning+belief schema `{.2,.3,.1,.4}`. Turning it on therefore
        # breaks comparability with previously collected arms in the same
        # class as a prompt-content change, so it must be an
        # explicit, recorded run-level choice and can never be acquired by
        # default -- and it is recorded on THE ROW, not merely
        # in the run's config.
        belief_protocol: str = BELIEF_PROTOCOL_FREE,
        # The prompt content this agent renders with. None
        # (default, every base arm) resolves the
        # canonical shared prompting fns. A probe
        # arm passes the registry's variant spec
        # (lmopt.prompting.probe_variants.require_variant_spec), which is both
        # what renders and what `prompt_spec_hash` below is computed from --
        # the two must be the same object, or the logged provenance describes
        # bytes the endpoint never saw.
        prompt_spec: Optional[GamePromptSpec] = None,
        # Memory rungs (M0-M3), window size for M2,
        # and scratchpad budget for M3.
        memory_rung: str = "M0",
        memory_window_k: int = 5,
        scratchpad_max_tokens: int = 256,
    ) -> None:
        validate_memory_rung(memory_rung)
        # Every construction path is guarded here, not only the scripts that validate a config:
        # an unknown mode would reach `act`'s bare `else` and skip every belief with
        # `belief_failed=False`.
        validate_elicitation_mode(elicitation_mode)
        # The openai SDK retries a failed request twice on its own and says nothing. The agent
        # turns that off on a real client and runs the same retries itself, logging each failed
        # attempt (`_TRANSPORT_RETRIES`, `_sdk_would_retry`, `_sdk_retry_delay`).
        self._client = (
            client.with_options(max_retries=0) if isinstance(client, OpenAI) else client
        )
        self._model = model
        self._elicitation_mode = elicitation_mode
        self._max_retries = max_retries
        self.backend = backend
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._belief_max_tokens = belief_max_tokens
        self._chat_template_kwargs = chat_template_kwargs
        self._action_top_logprobs = validate_action_top_logprobs(action_top_logprobs)

        # Refuse, never default: an unknown name here would otherwise collect
        # a whole arm under free generation while every row claimed a protocol
        # that never ran (no silent fallbacks). Combined mode is
        # refused for the same reason at one remove -- it makes no separate
        # belief call, so the constraint would be a no-op that the column
        # would nonetheless assert.
        if belief_protocol not in BELIEF_PROTOCOLS:
            raise ValueError(
                f"Unknown belief_protocol {belief_protocol!r}; "
                f"known protocols: {sorted(BELIEF_PROTOCOLS)}"
            )
        if elicitation_mode == "combined" and belief_protocol != BELIEF_PROTOCOL_FREE:
            raise ValueError(
                f"belief_protocol={belief_protocol!r} has no effect in "
                f"elicitation_mode='combined' (one shared call, no separate "
                f"belief call to constrain), and logging it would mislabel "
                f"every row of the run"
            )
        # Public (logged via match_runner's _get_lm_metadata, exactly as
        # prompt_schema_version is); the mapped schema is private, because
        # only the name is provenance.
        self.belief_protocol: str = belief_protocol
        self._belief_json_schema: Optional[dict] = BELIEF_PROTOCOLS[belief_protocol]

        # Per-request-seed provenance and reproducibility.
        # `run_seed` is deliberately bound to the matchup seed
        # (orchestration/shard.py's MatchupSpec.seed == the harness `seed`
        # arg to run_match, also logged verbatim as NodeRecord.seed) rather
        # than match_runner's `run_id` (a fresh uuid4 generated inside every
        # run_match() call when the caller does not supply one -- which
        # shard.py's run_matchup_shard does not). MatchupSpec.seed is this
        # codebase's actual stable-and-reproducible run identity axis --
        # shard.py's own module docstring: "One shard = one (game, opponent,
        # seed) triple" -- so re-running the same shard (same seed) must
        # re-derive the same per-request seeds, which binding to `run_id`
        # could not guarantee. None (default) means "no run_seed configured"
        # -- e.g. every caller that passes none -- and `_complete` is called
        # with seed=None (omitted from the API call).
        # Public: match_runner stamps it onto every row's `run_seed` column
        # via the same getattr pattern as model_id above, so a Parquet reader
        # can re-derive the per-request seeds from the row alone (schema.py's
        # run_seed comment for why the existing `seed` column is not enough).
        self.run_seed: Optional[int] = run_seed
        # Local structural position within the current hand: reset to 0 by
        # reset()/observe_hand_end() (a hand boundary), incremented once per
        # act() call (this agent's own decision count), never per retry
        # attempt or per underlying _complete() call -- see
        # derive_request_seed's docstring for why a per-node index is safe
        # and a global call counter is not. hand_idx mirrors it one level up:
        # incremented once per observe_hand_end() (called exactly once per
        # hand by match_runner for every agent, LM or not).
        self._hand_idx: int = 0
        self._node_idx: int = 0

        # Provenance: model_id reuses the existing `model` arg
        # (already the served model identifier -- no redefinition of
        # match_runner's model_revision=git_sha needed, just a second,
        # additive column). chat_template_hash accepts a caller-supplied
        # value. The sweep path passes a resolved hash; other construction
        # sites leave it None, logged as null (see class docstring for the
        # full status and why-not-construction-time note).
        self.model_id: Optional[str] = model
        self.chat_template_hash: Optional[str] = chat_template_hash
        # Public for the same reason as the two neighbours -- match_runner's _get_lm_metadata getattr stamps it
        # onto every row this agent produces.
        self.served_weights_revision: Optional[str] = served_weights_revision
        # Public for the same reason as served_weights_revision -- match_runner's getattr stamps it onto
        # every row this agent produces.
        self.tokenizer_revision: Optional[str] = tokenizer_revision
        self.prompt_schema_version: Optional[int] = PROMPT_SCHEMA_VERSION
        # prompt_schema_version
        # above is a coarse integer that denotes a content hash through
        # PROMPT_SPEC_HASHES, so on a variant arm it asserts canonical content
        # against variant bytes. This column carries the digest of the content
        # actually rendered: equal to PROMPT_SPEC_HASHES[prompt_schema_version]
        # on a base arm (what a reader checks), and provably different on a
        # variant arm. Computed once at construction from the same spec object
        # reset() renders with, never re-derived from a label.
        self._prompt_spec: Optional[GamePromptSpec] = prompt_spec
        self.prompt_spec_hash: Optional[str] = prompt_spec_hash(prompt_spec)

        self._seat: Optional[int] = None
        # --- cross-hand memory ---------------------------------------------
        # Memory accumulates only from this agent's own
        # observation stream: the `info_str` it was handed at each of its own
        # decisions, and the action it chose there, plus the hand payoff the
        # harness supplies via `observe_hand_end`. It is never sourced from
        # `observe_hand_end`'s `history` argument or from
        # `terminal_state.history()`, both of which carry the OPPONENT's moves
        # too. In Kuhn and Leduc those are public; in Dark Hex they are not
        # (a stone is revealed only on collision), so sourcing from them would
        # silently turn an imperfect-information game into near-perfect
        # information.
        #
        # Why an info_str is a legitimate source and a terminal history is not:
        # an info-set string is, by OpenSpiel's own definition, what the player
        # is entitled to know at A decision node. That warrant is about the
        # information partition over decision nodes, and does not extend to
        # terminal states -- on darkhex 2x2 seat 0, the terminal info string distinguishes 36 of 72 own-observation classes
        # via its global move counter alone, which counts opponent collisions
        # the player never saw.
        self._memory: list[HandMemory] = []
        self._current_hand: list[tuple[str, int]] = []
        # Resolved from the live game object at reset() - the harness always
        # resets before act (match_runner). Left None until then so any
        # pre-reset use fails loudly instead of silently building a
        # wrong-size prompt for a depth variant.
        self._fns: Optional[GamePromptFns] = None

        # Memory rung configuration and state.
        self.memory_rung: str = memory_rung
        self._memory_window_k: int = memory_window_k
        self._scratchpad_max_tokens: int = scratchpad_max_tokens
        self._scratchpad: Optional[str] = None
        # Truncation logging: None when rung is M0 or no block rendered this node,
        # mirroring last_completion_tokens nullable pattern.
        self.last_memory_truncated: Optional[bool] = None
        # The shard's per-call log, attached by run_match; None when no log is wanted.
        self._call_log: Optional[CallLog] = None
        # The prompts each call kind's row column already holds, so the call log writes a
        # prompt's text only where it differs; and the memory block as rendered before
        # truncation, kept only when truncation changed it.
        self._row_prompts: dict[str, str] = {}
        self._call_memory_raw: Optional[str] = None
        self.last_memory_chars_raw: Optional[int] = None
        self.last_memory_chars_sent: Optional[int] = None
        # Per-call reads of the first action and belief calls, and the winning attempt.
        self.last_call_latency_s: Optional[float] = None
        self.last_prompt_tokens: Optional[int] = None
        self.last_response_id: Optional[str] = None
        self.last_request_seed: Optional[int] = None
        self.last_belief_call_latency_s: Optional[float] = None
        self.last_belief_prompt_tokens: Optional[int] = None
        self.last_belief_response_id: Optional[str] = None
        self.last_belief_request_seed: Optional[int] = None
        self.last_winning_attempt: Optional[int] = None
        self.last_winning_raw_completion: Optional[str] = None

        # Populated after every act() call.
        self.last_prompt: Optional[str] = None
        self.last_raw_completion: Optional[str] = None
        self.last_n_retries: int = 0
        self.last_fallback_used: bool = False
        self.last_bet_aliased: bool = False
        self.last_action_parse_vocab: Optional[str] = None
        self.last_elicited_belief: Optional[str] = None
        self.last_belief_raw_completion: Optional[str] = None
        self.last_belief_n_retries: int = 0
        self.last_belief_failed: bool = False
        # Set only when every belief attempt failed strict parsing and a
        # bare-integer-key recovery supplied the belief; None otherwise.
        self.last_belief_salvage_kind: Optional[str] = None
        self.last_belief_salvage_attempt: Optional[int] = None
        # CompletionResult fields read at the _complete()
        # boundary, first-attempt semantics matching last_raw_completion /
        # last_belief_raw_completion above.
        self.last_finish_reason: Optional[str] = None
        self.last_completion_tokens: Optional[int] = None
        self.last_belief_finish_reason: Optional[str] = None
        self.last_belief_completion_tokens: Optional[int] = None
        # The reasoning trace `split_reasoning` (or the server's
        # own `reasoning_content`) surfaced at each call site, first-attempt
        # semantics matching every other `last_*` pair above.
        self.last_reasoning_trace: Optional[str] = None
        self.last_belief_reasoning_trace: Optional[str] = None
        # Declared here like the other `last_*` attrs, so `hasattr` holds
        # before the first decision.
        self.last_belief_prompt: Optional[str] = None
        # The per-decision ordinal `_derive_seed` used
        # for the current decision -- captured at act() entry, i.e. the value
        # `_node_idx` held before this act() advanced it, which is exactly the
        # `node_idx` component of every seed this decision's requests carried.
        # Exposed so match_runner can stamp it onto the row's agent_node_idx
        # column (the logged all-players `node_idx` does not equal it whenever
        # another player acts between two of this agent's decisions, which is
        # a majority of LM rows in a typical arm). Read-only exposure of the
        # existing counter.
        self.last_agent_node_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    @property
    def memory(self) -> list[HandMemory]:
        """Completed hands, oldest first, as this agent observed them.

        Read-only view of the cross-hand memory accumulator. Exposed as a
        property rather than a bare attribute so a caller cannot rebind it to
        a wider source (e.g. a terminal history) and quietly defeat the
        own-observations-only construction rule.
        """
        return list(self._memory)

    def attach_call_log(
        self, call_log: Optional[CallLog], run_id: Optional[str] = None
    ) -> None:
        """
        direct every following API call into a per-call log
        :param call_log: the shard writer's call log, or None to stop logging (Optional[CallLog])
        :param run_id: the run id its lines carry (Optional[str])
        """
        self._call_log = call_log
        if call_log is not None:
            call_log.attach(run_id)

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        """
        start a new match: bind the game's prompt functions and clear the per-match counters and memory
        :param game: the game this match plays (pyspiel.Game)
        :param seat: this agent's seat (int)
        :param opponent_id: the opponent's identifier (str)
        :raises ValueError: when the memory rung is not M0 and the game's prompt functions carry no
            memory terms
        """
        self._seat = seat
        # Authoritative re-dispatch from the live game object: name-keyed
        # lookups cannot distinguish depth variants (all Dark Hex rectangles
        # share one short_name), so dims come from game.get_parameters().
        # `spec=self._prompt_spec` is the probe injection point. It
        # is passed even when None (the canonical shared instance), so the
        # variant and base paths differ only in the object, not in the call.
        self._fns = get_game_fns_for_game(game, spec=self._prompt_spec)
        if self.memory_rung != "M0" and self._fns.memory_terms is None:
            raise ValueError(
                f"memory_rung {self.memory_rung!r} needs the game's memory terms, and the prompt "
                f"functions for {game.get_type().short_name!r} carry none"
            )
        # Defensive reset: reset() is called once per match
        # (match_runner.py, before the hand loop), not once per hand, so this
        # only matters if the same agent instance is reused across more than
        # one run_match() call -- keeps seed derivation from silently
        # continuing a prior match's node/hand count into a new one.
        self._hand_idx = 0
        self._node_idx = 0
        # A new match must not inherit the previous match's memory.
        # reset() is per-match, and some entrypoints reuse an agent object
        # across matchups, so carrying over would cross-contaminate arms with
        # no logged trace.
        self._memory = []
        self._current_hand = []
        self._scratchpad = None

    @property
    def _game_fns(self) -> GamePromptFns:
        """
        the prompt functions `reset()` resolved
        Before `reset()` `_fns` is None and this still returns it unchecked, so a pre-reset `act()`
        fails at the first attribute access.
        :returns: the resolved prompt functions (GamePromptFns)
        """
        return cast(GamePromptFns, self._fns)

    def act(self, state: pyspiel.State) -> int:
        """
        choose this agent's action at a decision node by asking the served model
        :param state: the current game state, with this agent to move (pyspiel.State)
        :returns: a legal action id (int)
        """
        player = state.current_player()
        info_str = state.information_state_string(player)
        legal = state.legal_actions()

        base = self._game_fns.build_prompt(info_str, player)

        # Render memory block once per node, inject into both channels.
        memory_block_raw = render_memory(
            self._memory,
            self.memory_rung,
            self._memory_window_k,
            self._scratchpad,
            # None only at M0, where render_memory returns before reading it (reset() refuses
            # any other rung without memory terms).
            cast(MemoryTerms, self._game_fns.memory_terms),
        )
        memory_block, was_truncated = truncate_memory_block(
            memory_block_raw, MEMORY_CHAR_BUDGET
        )
        # Truncation flag: None when rung is M0 or empty block (no memory to truncate).
        self.last_memory_truncated = was_truncated if memory_block else None
        self.last_memory_chars_raw = len(memory_block_raw) if memory_block else None
        self.last_memory_chars_sent = len(memory_block) if memory_block else None
        self._call_memory_raw = (
            memory_block_raw if (memory_block and was_truncated) else None
        )

        act_sfx = self._game_fns.action_suffix(info_str)
        full_action_prompt = base + memory_block + act_sfx

        self.last_prompt = full_action_prompt
        self._row_prompts = {"action": full_action_prompt}
        # Freeze this decision's ordinal before any
        # call is made. _node_idx advances at the END of act(), so by the time
        # match_runner reads the agent (after act() returns) this attribute is
        # where the value the derivation actually used survives.
        self.last_agent_node_idx = self._node_idx
        self.last_n_retries = 0
        self.last_fallback_used = False
        self.last_bet_aliased = False
        self.last_action_parse_vocab = None
        self.last_belief_raw_completion = None
        self.last_belief_n_retries = 0
        self.last_belief_failed = False
        self.last_belief_salvage_kind = None
        self.last_belief_salvage_attempt = None
        self.last_belief_finish_reason = None
        self.last_belief_completion_tokens = None
        self.last_reasoning_trace = None
        self.last_belief_reasoning_trace = None
        self.last_belief_prompt = None
        self.last_call_latency_s = None
        self.last_prompt_tokens = None
        self.last_response_id = None
        self.last_request_seed = None
        self.last_belief_call_latency_s = None
        self.last_belief_prompt_tokens = None
        self.last_belief_response_id = None
        self.last_belief_request_seed = None
        self.last_winning_attempt = None
        self.last_winning_raw_completion = None

        # Action call with bounded retry.
        action: Optional[int] = None
        first_raw = ""
        first_finish_reason: Optional[str] = None
        first_completion_tokens: Optional[int] = None
        first_trace: Optional[str] = None
        first_answer = ""
        winning_answer = ""
        for attempt in range(self._max_retries + 1):
            req_seed = self._derive_seed("action", attempt)
            result = self._complete(
                full_action_prompt, seed=req_seed, call_kind="action", attempt=attempt
            )
            # `answer` (not `result.content`) is what gets
            # parsed -- see `_compose_trace_and_answer`'s docstring. `raw`
            # (`result.content`, unmodified) is what `last_raw_completion`
            # logs, per attempt.
            trace, answer = _compose_trace_and_answer(result)
            if attempt == 0:
                first_raw = result.content
                first_finish_reason = result.finish_reason
                first_completion_tokens = result.completion_tokens
                first_trace = trace
                # Combined mode reuses this attempt's answer
                # segment for belief parsing below (mirrors first_raw's
                # existing reuse) -- captured regardless of elicitation
                # mode, since it is cheap and sibling mode never reads it.
                first_answer = answer
                # The first attempt's per-call reads, beside the others.
                # `req_seed` is the value actually handed to the client.
                self.last_call_latency_s = result.latency_s
                self.last_prompt_tokens = result.prompt_tokens
                self.last_response_id = result.response_id
                self.last_request_seed = req_seed
            parsed = self._game_fns.parse_action(answer, info_str)
            if parsed is not None and parsed in legal:
                action = parsed
                winning_answer = answer
                self.last_n_retries = attempt
                self.last_winning_attempt = attempt
                self.last_winning_raw_completion = result.content
                break

        self.last_raw_completion = first_raw
        self.last_finish_reason = first_finish_reason
        self.last_completion_tokens = first_completion_tokens
        self.last_reasoning_trace = first_trace

        if action is None:
            action = legal[0]
            self.last_n_retries = self._max_retries
            self.last_fallback_used = True
        else:
            if self._game_fns.parse_action_aliased is not None:
                # Log whether the (successful) action resolved via a vocabulary alias
                # (Leduc BET->RAISE) so the silent normalisation is auditable.
                self.last_bet_aliased = self._game_fns.parse_action_aliased(
                    winning_answer, info_str
                )
            if self._game_fns.parse_action_vocab is not None:
                # Same string parse_action consumed after split_reasoning: the
                # answer half, never the raw trace.
                self.last_action_parse_vocab = self._game_fns.parse_action_vocab(
                    winning_answer, info_str
                )

        # Belief elicitation, with a bounded retry mirroring the action path
        # above. A single unlogged attempt that silently yielded None on parse
        # failure would violate the no-silent-fallbacks rule.
        # last_belief_raw_completion/
        # last_belief_n_retries/last_belief_failed were reset to their
        # "never asked" defaults (None/0/False) above, so every branch below
        # that skips the call needs no further action.
        bel_sfx = self._game_fns.belief_suffix(info_str)
        if bel_sfx is None:
            # belief_suffix's own contract is `str | None`
            # (prompting/__init__.py): None means no belief question is
            # askable at this info-set, not that one was asked and failed.
            # Goofspiel's belief_suffix never returns None. The only source
            # path that can return None is Dark Hex's (`darkhex.py`'s
            # `belief_suffix`), when every unknown cell has already been
            # discovered -- but that is dead by construction for Dark Hex at
            # Any board size: a full-tree walk of all eight registered
            # variants (kuhn, leduc, goofspiel x2, liarsdice, darkhex x3) hit
            # belief_suffix() is None at 0 of 315,423 decision nodes, and
            # Dark Hex's closure's legality view equals state.legal_actions()
            # at all 151,131 darkhex nodes across those boards -- so the skip
            # branch fires only on an empty legal set, which cannot occur at
            # a decision node (a full Hex board always has a winner and the
            # state is terminal by then). This branch is retained as
            # Contract-only defensive code: only a future game with a
            # different belief_suffix contract could revive this path.
            self.last_elicited_belief = None
        elif self._elicitation_mode == "sibling":
            full_belief_prompt = base + memory_block + bel_sfx
            self.last_belief_prompt = full_belief_prompt
            self._row_prompts["belief"] = full_belief_prompt
            parsed_belief: Optional[str] = None
            first_bel_raw = ""
            first_bel_finish_reason: Optional[str] = None
            first_bel_completion_tokens: Optional[int] = None
            first_bel_trace: Optional[str] = None
            failed_bel_attempts: list[tuple[int, str, Optional[str]]] = []
            for bel_attempt in range(self._max_retries + 1):
                req_seed = self._derive_seed("belief", bel_attempt)
                # Belief call gets its own (larger) token budget --
                # see BELIEF_MAX_TOKENS's derivation comment above. The
                # action call above and combined mode's shared call below
                # are unchanged (no max_tokens override -> self._max_tokens).
                # The schema goes here and nowhere else. The
                # action retry loop above and combined mode's single shared
                # call below both omit `json_schema`, so an action stays free
                # generation of a bare label -- constraining it would be a
                # different manipulated variable, not this one.
                bel_result = self._complete(
                    full_belief_prompt,
                    seed=req_seed,
                    call_kind="belief",
                    attempt=bel_attempt,
                    max_tokens=self._belief_max_tokens,
                    json_schema=self._belief_json_schema,
                )
                # `bel_answer` (not `bel_result.content`) is what
                # gets parsed -- same composition as the action loop above.
                bel_trace, bel_answer = _compose_trace_and_answer(bel_result)
                if bel_attempt == 0:
                    first_bel_raw = bel_result.content
                    first_bel_finish_reason = bel_result.finish_reason
                    first_bel_completion_tokens = bel_result.completion_tokens
                    first_bel_trace = bel_trace
                    self.last_belief_call_latency_s = bel_result.latency_s
                    self.last_belief_prompt_tokens = bel_result.prompt_tokens
                    self.last_belief_response_id = bel_result.response_id
                    self.last_belief_request_seed = req_seed
                parsed_belief = self._game_fns.parse_belief(bel_answer, info_str)
                if parsed_belief is not None:
                    self.last_belief_n_retries = bel_attempt
                    break
                failed_bel_attempts.append(
                    (bel_attempt, bel_answer, bel_result.finish_reason)
                )

            # Design decision: log the first attempt's raw text,
            # not the winning one -- deliberately consistent with the action
            # path's last_raw_completion = first_raw above, for two
            # reasons. (1) Uniform semantics: an analyst who already knows
            # last_raw_completion is "the first sample, not necessarily the
            # text that determined the logged value" needs the same rule for
            # belief_raw_completion, or the two columns silently disagree on
            # what "raw completion" means. (2) The first-attempt text is the
            # reproducible sample independent of how many retries the
            # parse-fragility workaround needed -- useful for diagnosing why
            # a parse failed (formatting quirk, refusal, ...) and for
            # measuring how often a fixed prompt's completion varies at all,
            # which matters here because T=0 is not guaranteed deterministic
            # across calls (a 32-request probe showed variation). Logging the winning attempt instead would
            # survivorship-bias that measurement away.
            self.last_belief_raw_completion = first_bel_raw
            self.last_belief_finish_reason = first_bel_finish_reason
            self.last_belief_completion_tokens = first_bel_completion_tokens
            self.last_belief_reasoning_trace = first_bel_trace

            if parsed_belief is None:
                self.last_belief_n_retries = self._max_retries
                # Every attempt failed strict parsing; only now is a
                # recovery tried, so a clean retry always beats it.
                for bel_attempt, bel_answer, bel_finish in failed_bel_attempts:
                    salvaged = parse_belief_salvaging(
                        self._game_fns.parse_belief, bel_answer, info_str, bel_finish
                    )
                    if salvaged is not None:
                        parsed_belief, self.last_belief_salvage_kind = salvaged
                        self.last_belief_salvage_attempt = bel_attempt
                        break
                self.last_belief_failed = parsed_belief is None
            self.last_elicited_belief = parsed_belief
        elif self._elicitation_mode == "combined":
            # Single call shared with the action -- there is no separate
            # belief call to retry (retrying here would also re-ask for the
            # action, contradicting combined mode's one-call design), so
            # belief_raw_completion reuses first_raw (the one and only
            # completion this mode makes) and belief_n_retries stays 0 by
            # construction. belief_failed reflects whether that single
            # shared response parsed as a belief. The
            # finish_reason/completion_tokens attrs follow the same reuse --
            # there is no second response to read them from.
            # `split_reasoning` was already applied
            # Once to this shared completion in the action loop above, so
            # both the action parse and this belief extraction see the same
            # answer segment (`first_answer`); the belief trace attr mirrors
            # the action trace attr, following this same reuse pattern.
            self.last_belief_raw_completion = first_raw
            self.last_belief_finish_reason = first_finish_reason
            self.last_belief_completion_tokens = first_completion_tokens
            self.last_belief_reasoning_trace = first_trace
            # Record the prompt that was actually sent, not `base + bel_sfx`,
            # which combined mode never puts on the wire: the mode issues
            # exactly one request, with `full_action_prompt = base + act_sfx`
            # (see the action loop above), and `bel_sfx` is used here only to
            # decide that a belief is askable at all. `belief_prompt` exists so
            # the prompt is recoverable, so filling it with unsent bytes would
            # be a provenance fabrication -- worse than a null, because it
            # reads as ground truth. In combined mode the belief prompt is the action
            # prompt; `elicitation_mode` in run_meta says which mode a row came
            # from, so `belief_prompt == prompt` here is the truthful record.
            self.last_belief_prompt = full_action_prompt
            self.last_elicited_belief = self._game_fns.parse_belief(
                first_answer, info_str
            )
            if self.last_elicited_belief is None:
                salvaged = parse_belief_salvaging(
                    self._game_fns.parse_belief,
                    first_answer,
                    info_str,
                    first_finish_reason,
                )
                if salvaged is not None:
                    self.last_elicited_belief, self.last_belief_salvage_kind = salvaged
                    self.last_belief_salvage_attempt = 0
            self.last_belief_failed = self.last_elicited_belief is None
        else:
            self.last_elicited_belief = None

        # Advance the local per-hand node counter exactly once
        # per act() call -- i.e. once per decision this agent itself makes,
        # regardless of how many _complete() attempts (action retries +
        # belief retries) that decision needed. This is what keeps node_idx
        # a structural index invariant to retry counts: derive_request_seed's
        # docstring states why that invariance is what makes it safe to
        # include in the seed key at all.
        self._node_idx += 1

        # Record this decision as observed -- the info_str the harness
        # handed us and the action we chose from it. Appended after the action
        # is final (post-retry) so memory reflects what was played, not what
        # was attempted. Never sourced from the terminal history; see the
        # constructor's cross-hand memory note.
        self._current_hand.append((info_str, action))

        return action

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        """
        no belief through this hook: the LM's belief is elicited inside `act` and the true
        posterior is computed by the harness
        :param state: the current game state (pyspiel.State)
        :returns: None (None)
        """
        # LM belief is elicited and stored in last_elicited_belief.
        # True posterior is computed by the harness via oracle_policy.
        return None

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        """
        close the hand: store it in this agent's own memory, advance the hand counter and reset the
        node counter that the per-request seed is derived from, and at M3 update the scratchpad
        :param history: both players' actions for the hand, accepted for the agent protocol and never
            read (list of tuple[int, int])
        :param own_payoff: this agent's payoff for the hand, stored with its memory (float or None)
        """
        # match_runner.py calls observe_hand_end() exactly once
        # per hand for both agents regardless of type (run_match's hand
        # loop), so this is the one pre-existing hook that marks a hand
        # boundary without requiring any Agent-protocol signature change
        # (act() takes only `state`; there is no hand_idx/node_idx parameter
        # to thread through). Advancing hand_idx and resetting node_idx here
        # keeps both counters local structural indices scoped to this
        # agent's own decisions -- see derive_request_seed's docstring.
        self._hand_idx += 1
        self._node_idx = 0

        # Close the hand. `history` is accepted for protocol
        # compatibility and is deliberately unread -- it carries both players'
        # actions and is the leak source the memory construction excludes.
        # Only `own_payoff` is taken from the caller.
        just_closed = HandMemory(decisions=tuple(self._current_hand), payoff=own_payoff)
        self._memory.append(just_closed)
        self._current_hand = []

        # M3 scratchpad update.
        if self.memory_rung == "M3":
            # Built from own stream only (the just-closed HandMemory and prior scratchpad),
            # never from the history argument.
            decisions_str = "; ".join(
                f"{info_str} -> action {action}"
                for info_str, action in just_closed.decisions
            )
            payoff_val = just_closed.payoff if just_closed.payoff is not None else 0.0
            prev_notes = self._scratchpad if self._scratchpad is not None else "(none)"
            scratchpad_prompt = (
                f"You are playing a repeated game. Update your private notes about your "
                f"opponent's strategy based on this hand. Your decisions this hand: "
                f"{decisions_str}; your payoff: {payoff_val:+g}. Previous notes: {prev_notes}. "
                f"Reply with the updated notes only, under 150 words."
            )
            req_seed = self._derive_seed("scratchpad", 0)
            result = self._complete(
                scratchpad_prompt,
                seed=req_seed,
                call_kind="scratchpad",
                attempt=0,
                max_tokens=self._scratchpad_max_tokens,
            )
            self._scratchpad = result.content

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _derive_seed(self, call_kind: str, attempt: int) -> Optional[int]:
        """`derive_request_seed` bound to this agent's current position, or
        None when no `run_seed` was configured (the default behaviour:
        `_complete` omits `seed` entirely)."""
        if self.run_seed is None:
            return None
        return derive_request_seed(
            self.run_seed, self._hand_idx, self._node_idx, call_kind, attempt
        )

    def _complete(
        self,
        prompt: str,
        *,
        call_kind: str,
        attempt: int,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
    ) -> CompletionResult:
        """`max_tokens`: per-call override. Used only
        by the sibling-mode belief call site (`BELIEF_MAX_TOKENS`) -- every
        other caller (the action retry loop, combined mode's single shared
        call) omits it and keeps getting `self._max_tokens` (constructor
        default `ACTION_MAX_TOKENS`).

        `json_schema`: per-call structured-output constraint,
        sent as vLLM's `extra_body={"structured_outputs": {"json": ...}}`.
        Like `max_tokens`, it is used only by the sibling-mode belief call
        site, and only when the agent was constructed with a non-free
        `belief_protocol`; None (every other caller, and the default)
        omits `extra_body` from the request entirely rather than sending an
        empty or null one, so the off path is byte-identical on the wire to
        every call made without it. `extra_body` is the openai client's escape
        hatch for server-specific fields, which `structured_outputs` is.

        Returns a `CompletionResult`: the response's `.content` plus the
        further fields read off `resp`.
        """
        # Keyword arguments for the client call, of mixed types by design (int, dict, bool).
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if json_schema is not None:
            kwargs["extra_body"] = {"structured_outputs": {"json": json_schema}}
        if self._chat_template_kwargs is not None:
            kwargs.setdefault("extra_body", {})[
                "chat_template_kwargs"
            ] = self._chat_template_kwargs
        top_logprobs_requested = (
            self._action_top_logprobs is not None and call_kind == "action"
        )
        if top_logprobs_requested:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self._action_top_logprobs
        for transport_attempt in range(_TRANSPORT_RETRIES + 1):
            started = time.monotonic()
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    max_tokens=(
                        max_tokens if max_tokens is not None else self._max_tokens
                    ),
                    **kwargs,
                )
            except Exception as exc:
                # The API error is what the caller must see. A failure to log it is attached
                # as a note, never allowed to replace it. Every failed attempt is a line, so a
                # retry is a logged finding and not a silent repeat inside the client.
                try:
                    self._log_call(
                        call_kind,
                        attempt,
                        prompt,
                        seed,
                        None,
                        time.monotonic() - started,
                        error=exc,
                        transport_attempt=transport_attempt,
                    )
                except Exception as log_exc:
                    exc.add_note(f"call log write failed: {log_exc!r}")
                if transport_attempt < _TRANSPORT_RETRIES and _sdk_would_retry(exc):
                    _sleep(_sdk_retry_delay(exc, transport_attempt))
                    continue
                raise
            break
        latency_s = time.monotonic() - started
        choice = resp.choices[0]
        message = choice.message
        usage = getattr(resp, "usage", None)
        # The pinned vLLM serving image names the trace field
        # `reasoning`; `reasoning_content` is the earlier vLLM API. On the
        # pinned image the response message carries `reasoning` and no
        # `reasoning_content`, so reading only the old name would log a null
        # trace on every row of a parser arm while looking exactly like "the
        # model did not reason". Read the older name first (absent on this
        # image, harmless), then the live one.
        reasoning_trace = getattr(message, "reasoning_content", None)
        if reasoning_trace is None:
            reasoning_trace = getattr(message, "reasoning", None)
        completion = CompletionResult(
            content=message.content or "",
            reasoning_content=reasoning_trace,
            finish_reason=getattr(choice, "finish_reason", None),
            completion_tokens=(
                getattr(usage, "completion_tokens", None) if usage is not None else None
            ),
            latency_s=latency_s,
            prompt_tokens=(
                _typed_count(getattr(usage, "prompt_tokens", None))
                if usage is not None
                else None
            ),
            response_id=_typed_text(getattr(resp, "id", None)),
            top_logprobs=_read_top_logprobs(choice) if top_logprobs_requested else None,
        )
        self._log_call(
            call_kind,
            attempt,
            prompt,
            seed,
            completion,
            latency_s,
            transport_attempt=transport_attempt,
        )
        return completion

    def _log_call(
        self,
        call_kind: str,
        attempt: int,
        prompt: str,
        seed: Optional[int],
        result: Optional[CompletionResult],
        latency_s: float,
        error: Optional[BaseException] = None,
        transport_attempt: Optional[int] = None,
    ) -> None:
        """
        write one line to the shard's call log, if one is attached. The line carries the
        join keys, what was sent (a hash always, the text only when it is not already the
        prompt the row logs for this call kind) and everything read off the response. A call
        that raised carries the error and null response fields
        :param call_kind: "action", "belief" or "scratchpad" (str)
        :param attempt: 0-based attempt within the call kind (int)
        :param prompt: the exact prompt sent (str)
        :param seed: the request seed passed to the client, None when none was sent (Optional[int])
        :param result: the parsed response, None when the call raised (Optional[CompletionResult])
        :param latency_s: seconds around the client call (float)
        :param error: the exception the call raised, if any (Optional[BaseException])
        :param transport_attempt: 0-based try of this request after transport failures; a line
            written before this field existed has no key (Optional[int])
        """
        if self._call_log is None:
            return
        scratchpad = call_kind == "scratchpad"
        row_prompt = self._row_prompts.get(call_kind)
        self._call_log.append(
            {
                # The scratchpad call runs after observe_hand_end advanced the counters; it
                # summarises the hand just closed, which has no single decision row.
                "hand_idx": self._hand_idx - 1 if scratchpad else self._hand_idx,
                "agent_node_idx": None if scratchpad else self._node_idx,
                "call_kind": call_kind,
                "attempt": attempt,
                "request_seed": seed,
                # Every response-derived value is type-checked: a test double answers an
                # unset attribute with a child mock, which must never reach the file.
                "response_id": _typed_text(result.response_id) if result else None,
                "finish_reason": _typed_text(result.finish_reason) if result else None,
                "prompt_tokens": _typed_count(result.prompt_tokens) if result else None,
                "completion_tokens": (
                    _typed_count(result.completion_tokens) if result else None
                ),
                "latency_s": latency_s,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "raw_completion": _typed_text(result.content) if result else None,
                "reasoning_content": (
                    _typed_text(result.reasoning_content) if result else None
                ),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": None if prompt == row_prompt else prompt,
                "memory_block_raw": (
                    self._call_memory_raw
                    if call_kind == "action" and attempt == 0
                    else None
                ),
                "top_logprobs": result.top_logprobs if result else None,
                # The k this call asked for, null when it asked for none: without it a null
                # `top_logprobs` cannot tell "not requested" from "requested, none returned".
                "top_logprobs_k": (
                    self._action_top_logprobs if call_kind == "action" else None
                ),
                "error_type": type(error).__name__ if error is not None else None,
                "error": str(error) if error is not None else None,
                "transport_attempt": transport_attempt,
            }
        )


# todo: I feel like this needs work
