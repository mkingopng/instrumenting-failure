"""Matchup shard: the single unit of distribution.

One shard = one (game, opponent, seed) triple for n_hands, writing exactly one
Parquet file. Local pools (ProcessPoolExecutor) and Ray tasks both call
``run_matchup_shard``; the inner match loop is byte-identical to the serial
path in ``scripts/run_sweep.py`` - a
``MatchupSpec`` is a plain, picklable, frozen dataclass so it crosses a
process/Ray boundary without any shared state, and ``run_matchup_shard``
builds the game, agents, and writer fresh inside the calling process.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from lmopt.agents.bayesian import BayesianAgent
from lmopt.agents.cfr_ref import CFRRefAgent
from lmopt.agents.classical import AlwaysBetAgent

# Imported at module level for the names only (no `openai` dependency: the
# LMVllmAgent import itself stays lazy inside _build_agent0, and lm_vllm's own
# module-level imports are pyspiel/prompting, which this module already pulls
# via match_runner). Importing rather than re-typing "free" is deliberate:
# the mapping is the single source of truth for which protocols exist, so a
# new protocol cannot become dispatchable here without being decodable there.
from lmopt.agents.lm_vllm import (
    BELIEF_PROTOCOL_FREE,
    BELIEF_PROTOCOLS,
    validate_elicitation_mode,
    validate_memory_rung,
)
from lmopt.agents.rl_selfplay import RLSelfPlayAgent
from lmopt.games.depth_reference import DEPTH_REFERENCES, load_depth_reference
from lmopt.games.loaders import game_params_for, load_game
from lmopt.games.policy_cache import load_or_solve_cfr
from lmopt.harness.match_runner import SEAT_MODES, run_match
from lmopt.logging.run_meta import write_json_atomic
from lmopt.logging.schema import BASE_PROBE_TYPE, OFFPATH_PROBE_TYPE, PROBE_TYPE_DOMAIN
from lmopt.logging.writer import MatchWriter

# The variant registry is imported, never re-typed here --
# the names, their pairing with probe types, and the recorded absences all have
# exactly one source.
from lmopt.prompting.probe_variants import (
    BASE_VARIANT_NAME,
    PROMPT_VARIANT_NAMES,
    expected_variant_name,
    require_variant_spec,
)

# agent0 sentinel: "no vLLM client, agent0 is the seeded RandomAgent" - plumbing
# for the existing run_sweep.py `agent0: lm|random` config key (config over
# forks), not a new mode. Lets shard tests and parity runs
# run with no vLLM server.
RANDOM_AGENT0_SENTINEL = "__random__"
# agent0 sentinel: "no vLLM client, agent0 is the adaptive reference" (the P2 ceiling; lmopt.agents.adaptive)
ADAPTIVE_AGENT0_SENTINEL = "__adaptive__"
# the agent0 sentinels that need no served model: the sweep skips serving provenance for them
SCRIPTED_AGENT0_SENTINELS = frozenset({RANDOM_AGENT0_SENTINEL, ADAPTIVE_AGENT0_SENTINEL})


# The per-request HTTP ceiling on every LM call a shard makes, and the default
# for `MatchupSpec.request_timeout_s`. The number below is derived, in the
# manner of `BELIEF_MAX_TOKENS` (lmopt/agents/lm_vllm.py), not chosen.
#
# A 120 s ceiling is too small: it is ample for terse beliefs (22-70
# completion tokens) but not once the belief questions ask the model to infer
# a posterior from evidence under a 2048-token budget. Goofspiel belief
# completions then run p50 1,295 / p99 2,023 / max 2,137 characters (against
# p50 39 for kuhn and 54 for leduc), and a 120 s timeout kills shards that are
# making normal progress.
#
# The derivation. The ceiling must cover the largest response the harness
# itself permits, not the largest yet observed, because `max_tokens` is the
# only hard bound on generation length:
#   (a) numerator: BELIEF_MAX_TOKENS = 2048 tokens (lm_vllm.py). The belief
#       call is the long one; the action call shares the same client.
#   (b) denominator: the slowest per-request decode rate measured on this
#       instrument at the configured width. Against vLLM serving
#       a small model, on a real goofspiel belief prompt rendered by
#       this repo's own prompting code:
#       single-stream 6.94 tok/s (3 calls, 364-490 tokens, 52.3-70.7 s);
#       80 concurrent 4.96 tok/s per request (456 tokens each, 91.9 s wall),
#       i.e. running at the configured max_workers costs ~29 % of per-request
#       rate. A separate low-contention draw over real deep goofspiel
#       info-sets produced a 1,334-token belief in 193.5 s -- 1.6x a 120 s
#       ceiling with the endpoint essentially idle.
#   (c) 2048 / 4.96 = 413 s is therefore the worst case for a request that is
#       making normal progress.
#   (d) Set the ceiling at 2x that, rounded to a whole quarter-hour: 900 s.
#       The rule this encodes is "fire only once throughput has fallen below
#       Half the slowest rate ever measured here", i.e. the request is stalled
#       rather than merely slow. 900 s implies a floor of 2048 / 900 =
#       2.28 tok/s sustained.
#
# What would falsify it (a timeout justifies inaction, so it needs a stated
# falsifier):
#   1. Any served model whose sustained per-request rate at the configured
#      max_workers is below 2.28 tok/s. `Olmo-3.1-32B-Think` is budgeted at
#      ~4 tok/s single-stream; apply the 29 % contention penalty measured above and it
#      lands near 2.8 tok/s, inside the floor by under 25 % -- serving it
#      falsifies this default and forces a per-model `request_timeout_s`.
#   2. Raising BELIEF_MAX_TOKENS above 2048, or max_workers above 80: both
#      move a term of (c) and the arithmetic must be redone.
#   3. Direct evidence: a shard failing with `Request timed out` while the
#      endpoint's /metrics still counts that request in `num_requests_running`
#      with its generation tokens advancing -- that is a request killed while
#      working, which is exactly what this value exists to prevent.
#   4. The opposite evidence retires the derivation rather than the number: if
#      timeouts fire on requests that produced zero tokens, the request is not
#      slow but hung, and the right instrument is a stall detector, not a
#      larger ceiling.
#
# Not a comparability-breaking setting: a timeout changes which requests survive, never what a surviving request was
# asked or how it was decoded. Every completion admitted at 120 s is admitted
# byte-identically at 900 s.
DEFAULT_REQUEST_TIMEOUT_S = 900.0

# One matchup seed, five consumers that must be statistically
# independent: the deal (chance nodes, match_runner.py), the OPPONENT's action
# sampling, the scoring reference, a random agent0, and
# the per-hand seat draw match_runner.py's `_seat_stream` makes under
# seat_mode="random". Only "random" draws: seat_mode="alternate" derives its
# seat from hand_idx and consumes nothing from this stream.
#
# The order is load-bearing -- it is the reproducibility contract. `spawn`
# derives streams positionally, so reordering or inserting a name changes every
# stream after it and silently changes what a recorded seed reproduces. Append
# only, and never renumber. STREAM_SEAT is at position 4 (this module's
# own spawn space, rooted at spec.seed).
#
# N_STREAMS_PER_MATCHUP is free to grow: match_runner.py's seat stream is
# rooted in a salted entropy space of its own (SEAT_SALT), not at a spawn index
# that has to stay ahead of this count, so appending a stream here has no
# consequence there. That decoupling is deliberate: an index-and-pin
# arrangement would make every append change what a recorded seed replays for
# seat draws (see SEAT_SALT's own comment for what a change there would cost).
STREAM_DEAL, STREAM_OPPONENT, STREAM_REFERENCE, STREAM_AGENT0, STREAM_SEAT = range(5)
N_STREAMS_PER_MATCHUP = 5


def derive_stream_seeds(matchup_seed: int) -> list:
    """Spawn ``N_STREAMS_PER_MATCHUP`` independent streams from one matchup seed.

    Why this exists. Seeding streams by offset (``default_rng(seed)`` for
    the deal, ``default_rng(seed + 1)`` for the opponent) is independent
    within one shard, because ``default_rng`` runs its argument through
    ``SeedSequence`` entropy mixing, but not across shards. Sweep seeds are
    ``base_seed + seed_idx`` with spacing 1, so a ``+1`` opponent offset makes
    shard *S*'s opponent stream byte-identical to shard *S+1*'s deal stream,
    and a ``+7`` agent0 offset collides with the deals once a sweep reaches 8
    or more seeds.

    Why it matters here rather than as hygiene. P3 scores the divergence of
    the LM's action distribution from the equilibrium mixture, and the
    opponent's mixing is what realises that mixture in play. Shared entropy
    between what is dealt and how the opponent mixes sits underneath a
    randomisation measurement, and sharing it across shards couples the sample
    to itself rather than confining the confound to one matchup.

    Reproducibility. The derived values are deliberately not logged. This is
    a pinned pure function of ``matchup_seed``, which every spec and every
    ``run_meta.json`` already records, so the seed alone reproduces the streams.
    A logged value could not detect a change to the derivation.

    Returns ``SeedSequence`` objects, which ``np.random.default_rng`` accepts
    directly, so no consumer needs to change how it seeds itself.
    """
    return np.random.SeedSequence(matchup_seed).spawn(N_STREAMS_PER_MATCHUP)


@dataclass(frozen=True)
class MatchupSpec:
    """Everything one shard needs to play a matchup and write its Parquet file."""

    game_name: str  # key into GAMES registry
    opponent: str  # "cfr_ref" | "bayesian" | "rl" | "always_bet"
    seed: int
    n_hands: int
    out_path: str  # exact parquet path this shard writes
    model: str  # HF model id, or RANDOM_AGENT0_SENTINEL
    vllm_base_url: str
    elicitation_mode: str
    max_retries: int
    cfr_iters: int
    rl_iterations: int
    cfr_cache_dir: str
    rl_cache_dir: str
    # The serving backend label stamped on every LM row, from cfg["backend"]. Required: a
    # default would stamp a card label onto rows that ran somewhere else.
    backend: str
    require_cache_hit: bool = False  # True in parallel runs - fail loudly on miss
    # Sourced from cfg["temperature"] (run_sweep.py), defaulting to
    # 0.0, matching LMVllmAgent's own constructor default. Flows into the LM agent's decoding
    # temperature and (via `seed=spec.seed` below) is the identity `run_seed`
    # the per-request seed is derived from -- see LMVllmAgent's `run_seed`
    # docstring for why `seed`, not a new field, is that identity.
    temperature: float = 0.0
    # Sourced from run_sweep.py's one-time startup
    # resolution (scripts/run_sweep.py's `_resolve_serving_block`), never
    # re-resolved per shard -- LMVllmAgent's own docstring is explicit that
    # this is a per-run fingerprint of the served template, not a per-request
    # value. None (default; every non-LM shard via RANDOM_AGENT0_SENTINEL,
    # and every older caller/test) means "not resolved for this
    # shard" -- never a fabricated placeholder (no silent fallbacks).
    chat_template_hash: Optional[str] = None
    # The served model's HF
    # weights revision, sourced from run_sweep.py's startup gate -- which
    # itself never resolves it (the value lives in the serving box's HF
    # cache, invisible to the sweep's HTTP-only resolution): the launcher
    # (run_experiment.py, via lmopt.provenance.weights_cache's
    # resolve_prelaunch_revision) resolves it once per arm and hands it to
    # run_sweep as --served-weights-revision. Threaded from here into
    # LMVllmAgent exactly like chat_template_hash above, and stamped on
    # every LM row (logging/schema.py's served_weights_revision column).
    # None (default; every non-LM shard and every caller/test that omits it)
    # means "not resolved for this shard" -- never a fabricated placeholder
    # (no silent fallbacks); a real sweep cannot reach shard dispatch with
    # None because run_sweep aborts at startup first.
    served_weights_revision: Optional[str] = None
    # The served endpoint's HF tokenizer revision, threaded through the
    # identical route as served_weights_revision above -- same
    # None-means-not-resolved rule.
    tokenizer_revision: Optional[str] = None
    # The per-request HTTP ceiling on this shard's LM calls,
    # sourced from cfg["request_timeout_s"] (run_sweep.py) -- config over
    # forks, so a slow model gets a larger budget without a code
    # edit. Defaults to the module constant rather than a re-typed literal, so
    # the value and its derivation cannot drift apart; see
    # DEFAULT_REQUEST_TIMEOUT_S for the derivation and its falsifiers.
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    # The belief decoding protocol, sourced from
    # cfg["belief_protocol"] (run_sweep.py) -- config over forks. Without a
    # production caller the protocol is reachable
    # only from tests; this field is what makes an
    # arm collectable under it. Defaults to free generation, the protocol
    # every landed arm used, so an unconfigured sweep is unchanged on the
    # wire. Whatever value lands here is stamped on every LM row
    # (logging/schema.py's `belief_protocol` column), not merely on
    # run_meta.json.
    belief_protocol: str = BELIEF_PROTOCOL_FREE
    # Per-run decode budgets for the LM's action and belief calls -- run-level
    # config (config over forks), sourced from cfg["action_max_tokens"] /
    # cfg["belief_max_tokens"] (run_sweep.py). None (default) means "the
    # agent's own constructor defaults", i.e. ACTION_MAX_TOKENS /
    # BELIEF_MAX_TOKENS in lm_vllm.py -- so an unconfigured sweep is
    # byte-identical on the wire. Exists because a thinking model spends the
    # cap on its trace before the answer; the value used is recorded in
    # run_meta.json's resolved_config, and per-row cost is measurable from the
    # completion_tokens columns.
    action_max_tokens: Optional[int] = None
    belief_max_tokens: Optional[int] = None
    # chat_template_kwargs knob: per-run kwargs passed to the served
    # endpoint's chat template renderer, sourced from cfg["chat_template_kwargs"].
    # None (default, every config to date) means agent default, field absent
    # on the wire. Recorded via resolved_config. The two Qwen3 arms
    # (enable_thinking True/False) are the motivating case. A dict value makes
    # the frozen dataclass's immutability shallow and a kwargs spec unhashable;
    # spec-keyed maps must keep id() keying (run_sweep.py's index_of_spec does).
    chat_template_kwargs: Optional[dict] = None
    # Run-level top-k logprobs on the action call only, from cfg["action_top_logprobs"].
    # None (every config to date) leaves the request byte-identical; an integer k adds
    # `logprobs` and `top_logprobs` to action requests. The values go to the per-call log.
    action_top_logprobs: Optional[int] = None
    # Memory knobs: the memory rung (M0 = no history, M1 =
    # cross-hand summary line, M2 = transcript over window k, M3 = LM-written
    # scratchpad) and k for M2 (the transcript window size), sourced from
    # cfg["memory_rung"] / cfg["memory_window_k"]. Defaults are "M0" / 5,
    # matching the agent constructor defaults (lm_vllm.py).
    memory_rung: str = "M0"
    memory_window_k: int = 5
    # Which seat agent0 occupies (SEAT_MODES,
    # match_runner.py) -- run-level config (config over forks), sourced from
    # cfg["seat_mode"] (run_sweep.py). Defaults to "fixed0", run_match's own
    # default.
    seat_mode: str = "fixed0"
    # The probe-arm label (PROBE_TYPE_DOMAIN,
    # logging/schema.py) stamped on every row via run_match. "none" -- not a
    # probe -- is the default; a `probe:`
    # per-arm config key is what sets this to something else.
    probe_type: str = BASE_PROBE_TYPE
    # The row-level identity of the prompt content this arm
    # renders (PROMPT_VARIANT_NAMES, prompting/probe_variants.py), and the
    # paired scorer's join key. "base" -- canonical content -- is the default.
    # It is a separate field from probe_type rather than
    # derived from it because the two answer different questions on the row
    # ("what was manipulated" vs "what was rendered"), and because
    # offpath_drive manipulates the match without touching the prompt; the
    # __post_init__ check below is what keeps the pair from contradicting
    # itself.
    prompt_variant: str = BASE_VARIANT_NAME
    # The P6 forced opening, sourced from
    # cfg["drive_openings"] (run_sweep.py) -- run-level config (config over
    # forks). None means not a driven arm. A TUPLE, not the list YAML supplies: this dataclass is frozen and
    # therefore hashable, and one unhashable field would silently withdraw
    # that guarantee for exactly the arm shape this adds. __post_init__
    # normalises, so a caller may pass either.
    drive_openings: Optional[tuple[int, ...]] = None
    # Sourced from scripts/run_sweep.py's one-time startup resolution
    # (`git_sha = _get_git_sha()`, also used for the sweep-level
    # run_meta.json), threaded through here exactly like `chat_template_hash`
    # and `served_weights_revision` above -- never re-resolved per shard. If
    # `run_match` resolved its own sha on every call, an arm's shards,
    # dispatched over hours, would split between two different shas when the
    # checkout's HEAD moved mid-sweep. None (default; callers and tests that
    # omit it) means "let run_match resolve its own" -- never a fabricated
    # placeholder (no silent fallbacks).
    git_sha: Optional[str] = None

    def __post_init__(self) -> None:
        # Fail in the dispatching process, before any shard is submitted:
        # LMVllmAgent refuses an unknown protocol too, but that refusal
        # happens inside a worker, after the expensive setup, and (with
        # on_error) would be recorded as a per-shard failure rather than
        # as the config error it is. A typo must not cost a partial arm.
        #
        # This check runs only after the first game's CFR solve, not before any solve:
        # run_sweep.py builds its first MatchupSpec inside the per-game loop,
        # after that game's load_or_solve_cfr and
        # RLSelfPlayAgent.load_or_train. A bad
        # belief_protocol in YAML therefore still costs one game's cache warm;
        # only the argparse `choices` on the CLI flag catches it earlier, and
        # only for the flag.
        if self.belief_protocol not in BELIEF_PROTOCOLS:
            raise ValueError(
                f"Unknown belief_protocol {self.belief_protocol!r}; "
                f"known protocols: {sorted(BELIEF_PROTOCOLS)}"
            )
        # The same reasoning for the elicitation mode: refused here, in the dispatching process,
        # and again by the agent for any caller that builds one directly.
        validate_elicitation_mode(self.elicitation_mode)
        # The memory knobs likewise: an unknown rung or a window below one would otherwise
        # reach the agent constructor inside a worker, after the CFR solve.
        validate_memory_rung(self.memory_rung)
        if self.memory_window_k < 1:
            raise ValueError(
                f"memory_window_k must be at least 1, got {self.memory_window_k!r}; set "
                f"memory_window_k in the run config to a positive transcript window"
            )
        # The library-level guard for seat_mode, and the only guard for
        # probe_type. run_match's own SEAT_MODES/PROBE_TYPE_DOMAIN checks
        # would still fire, but only inside a worker, after CFR/RL cache
        # setup, and would be recorded and survived (on_error)
        # rather than surfacing as the config error it is. Checking here
        # means no caller can build a spec that is certain to fail later.
        #
        # For a sweep, this is the second line of defence, not the first:
        # run_sweep.py's `_validate_seat_mode(cfg)` rejects a typo'd
        # seat_mode at startup, before the first CFR solve.
        # A typo'd probe block is rejected at startup the same way, by
        # run_sweep.py's `_validate_probe`.
        if self.seat_mode not in SEAT_MODES:
            raise ValueError(
                f"unknown seat_mode {self.seat_mode!r}; expected one of {SEAT_MODES}"
            )
        if self.probe_type not in PROBE_TYPE_DOMAIN:
            raise ValueError(
                f"unknown probe_type {self.probe_type!r}; expected one of "
                f"{sorted(PROBE_TYPE_DOMAIN)}"
            )
        # The library-level guard on
        # prompt_variant, against the imported registry (never a re-typed list,
        # so a variant added there cannot leave this file checking a stale
        # copy). run_match takes prompt_variant as a free string by design, so
        # without this check a typo would reach the Parquet as a variant label
        # with no spec behind it, and the paired scorer would join two arms on
        # a name nothing can resolve.
        #
        # One check, not two. A separate `prompt_variant in
        # PROMPT_VARIANT_NAMES` membership test would be dead:
        # `expected_variant_name(probe_type)` always returns a member of that
        # set, so every value a membership test rejects this equality already
        # rejects. The known-names list rides in the message instead, where it
        # helps a reader.
        #
        # Ordering matters: probe_type is validated first, above, so
        # expected_variant_name can only raise for a probe type that is in the
        # domain but unclassified by the registry -- a real defect, not a typo.
        expected = expected_variant_name(self.probe_type)
        if self.prompt_variant != expected:
            raise ValueError(
                f"probe_type {self.probe_type!r} renders prompt_variant "
                f"{expected!r}, got {self.prompt_variant!r} (known variants: "
                f"{sorted(PROMPT_VARIANT_NAMES)}). The two labels are one claim "
                f"about the row: an arm labelled with a probe type it did not "
                f"render (or rendering a variant it is not labelled with) is a "
                f"fabricated measurement, not a mislabelled one."
            )
        # The same one-claim rule for the other probe axis.
        # offpath_drive manipulates the match rather than the prompt, so its
        # evidence is `drive_openings`, and the two must agree in both
        # directions: a driven arm labelled "none" hides the manipulation from
        # every scorer, and the label with no driver is a fabricated
        # probe. run_match refuses both too, but only inside a worker.
        if self.drive_openings is not None:
            object.__setattr__(self, "drive_openings", tuple(self.drive_openings))
        if bool(self.drive_openings) != (self.probe_type == OFFPATH_PROBE_TYPE):
            raise ValueError(
                f"drive_openings={self.drive_openings!r} and probe_type "
                f"{self.probe_type!r} contradict each other: a driven arm is "
                f"labelled {OFFPATH_PROBE_TYPE!r} and an arm labelled "
                f"{OFFPATH_PROBE_TYPE!r} is driven. One without the other is a "
                f"manipulation with no label, or a label with no manipulation."
            )


def _resolve_prompt_spec(spec: MatchupSpec):
    """The prompt content this arm renders: None for a base arm, else the
    registry's variant spec for (this game's family, this probe type).

    ``require_variant_spec`` rather than ``variant_spec`` (no silent
    fallbacks): ``variant_spec`` returns None for a recorded absence, and None
    means "canonical" one call later in ``get_game_fns_for_game``, so a
    goofspiel P1 arm would render canonical bytes under a probe label. The
    raise carries the recorded reason and what the variant would need.

    Resolved for both agent0 kinds, before the LM/sentinel branch: a
    random-agent0 probe arm (the zero-GPU pilot's shape) renders no prompt at
    all, but it still labels every row with a probe type, so an arm whose
    variant does not exist must refuse to launch either way.

    Keyed on the family, never the raw registry key: the depth variants
    ("darkhex_3x3") share one family's prompt content, and the variant registry
    is keyed by family (``probe_variants.GAME_KEYS``).
    """
    if spec.prompt_variant == BASE_VARIANT_NAME:
        return None
    from lmopt.games.families import GAME_FAMILIES

    family = GAME_FAMILIES.get(spec.game_name, spec.game_name)
    return require_variant_spec(family, spec.probe_type)


def _build_agent0(spec: MatchupSpec, game_name: str, cfr_policy=None):
    """agent0 is the LM under test, or (via a sentinel) a seeded RandomAgent or the adaptive reference.

    :param cfr_policy: the game's CFR+ reference, the adaptive reference's prior; required by that sentinel only
    """
    prompt_spec = _resolve_prompt_spec(spec)
    if spec.model == ADAPTIVE_AGENT0_SENTINEL:
        from lmopt.agents.adaptive import AdaptiveAgent

        if cfr_policy is None:
            raise ValueError("the adaptive agent0 needs the game's CFR+ policy as its prior; none was passed")
        return AdaptiveAgent(
            cfr_policy,
            seed=derive_stream_seeds(spec.seed)[STREAM_AGENT0],
            memory_rung=spec.memory_rung,
        )
    if spec.model == RANDOM_AGENT0_SENTINEL:
        from lmopt.agents.random_agent import RandomAgent

        # A spawned stream, not a fixed offset such as `spec.seed + 7`: a fixed
        # offset only decorrelates within one shard, and with sweep seeds
        # spaced by 1 it collides with other shards' deal streams once the
        # sweep has enough seeds.
        return RandomAgent(seed=derive_stream_seeds(spec.seed)[STREAM_AGENT0])

    from openai import OpenAI
    from lmopt.agents.lm_vllm import LMVllmAgent

    # timeout comes from the spec, never a literal. See
    # DEFAULT_REQUEST_TIMEOUT_S.
    client = OpenAI(
        api_key="EMPTY",
        base_url=spec.vllm_base_url,
        timeout=spec.request_timeout_s,
    )
    # Config-set overrides are passed only when the config
    # set them -- None must reach the constructor as an omitted argument, never
    # as an explicit None, so the agent's own defaults (and their derivation
    # comments) remain the single source of truth.
    # Heterogeneous passthrough kwargs (ints, a dict, a str), so the value type is Any.
    agent_kwargs: dict[str, Any] = {}
    if spec.action_max_tokens is not None:
        agent_kwargs["max_tokens"] = spec.action_max_tokens
    if spec.belief_max_tokens is not None:
        agent_kwargs["belief_max_tokens"] = spec.belief_max_tokens
    if spec.chat_template_kwargs is not None:
        agent_kwargs["chat_template_kwargs"] = spec.chat_template_kwargs
    if spec.action_top_logprobs is not None:
        agent_kwargs["action_top_logprobs"] = spec.action_top_logprobs
    # Memory knobs from config. MatchupSpec fields have the same
    # defaults as the agent constructor, so unconditional passthrough would work
    # for byte-identical dispatch; but the same reasoning applies: the agent's
    # own defaults and their derivation comments remain the single source of
    # truth, and the spec's defaults exist only for dataclass equality of specs
    # built without the keys. Omit the kwargs when the spec has the defaults.
    if spec.memory_rung != "M0":
        agent_kwargs["memory_rung"] = spec.memory_rung
    if spec.memory_window_k != 5:
        agent_kwargs["memory_window_k"] = spec.memory_window_k
    return LMVllmAgent(
        client=client,
        model=spec.model,
        game_name=game_name,
        elicitation_mode=spec.elicitation_mode,
        max_retries=spec.max_retries,
        backend=spec.backend,
        temperature=spec.temperature,
        **agent_kwargs,
        # run_seed=spec.seed: the shard's own matchup seed is
        # this codebase's stable per-shard identity ("one shard = one
        # (game, opponent, seed) triple" -- this module's own docstring),
        # not match_runner's ephemeral per-call `run_id` uuid -- see
        # LMVllmAgent's run_seed docstring for the full reasoning.
        run_seed=spec.seed,
        # Threaded from the spec, not re-resolved here --
        # see MatchupSpec.chat_template_hash's own docstring for the
        # once-per-run-not-per-shard reasoning.
        chat_template_hash=spec.chat_template_hash,
        # Same thread, same reasoning -- see
        # MatchupSpec.served_weights_revision's own docstring. Without this
        # line the column is configurable, gated at startup, and never
        # actually stamped.
        served_weights_revision=spec.served_weights_revision,
        # Same thread, same reasoning -- see MatchupSpec.tokenizer_revision's
        # own docstring.
        tokenizer_revision=spec.tokenizer_revision,
        # Without this line the protocol is configurable, logged,
        # and never actually applied.
        belief_protocol=spec.belief_protocol,
        # The resolved variant spec, which is both what the
        # agent renders with and what its logged prompt_spec_hash is computed
        # from. Passing the label instead would let the rendering and the
        # provenance disagree with no trace.
        prompt_spec=prompt_spec,
    )


def _build_opponent(
    name: str, game, cfr_policy, seed: int, rl_agent: RLSelfPlayAgent | None
):
    if name == "cfr_ref":
        return CFRRefAgent(cfr_policy, seed=seed)
    if name == "bayesian":
        return BayesianAgent(opponent_policy=cfr_policy, seed=seed)
    if name == "rl":
        if rl_agent is None:
            raise ValueError(
                "opponent 'rl' needs a trained RL agent, got None; load or train one with "
                "RLSelfPlayAgent.load_or_train before building the opponent"
            )
        return rl_agent
    if name == "always_bet":
        return AlwaysBetAgent()
    raise ValueError(f"Unknown opponent: {name!r}")


def posterior_fallback_sidecar_path(out_path: str) -> str:
    """The per-shard posterior-fallback sidecar beside `out_path`'s Parquet.

    One function, two callers -- `run_matchup_shard` (writer, in the worker
    process) and `scripts/run_sweep.py` (reader, in the parent) -- so the two
    sides cannot disagree about the filename. Naming the file from the shard's
    own `out_path` also keeps it index-free: a shard reported through
    `on_error` simply has no sidecar, and the parent's aggregation skips it
    without needing to know which shards failed.
    """
    return os.path.splitext(out_path)[0] + ".posterior_fallback.json"


def driven_discard_sidecar_path(out_path: str) -> str:
    """The per-shard driven-hand discard sidecar beside `out_path`'s Parquet.

    Same one-function-two-callers contract as
    `posterior_fallback_sidecar_path` above. Written only for a driven shard
    (zero counts included -- a driven arm's measured zero is a fact); an
    undriven shard leaves no file, so absence keeps meaning "not a driven
    arm" rather than "discards unmeasured".
    """
    return os.path.splitext(out_path)[0] + ".driven_discards.json"


def timing_sidecar_path(out_path: str) -> str:
    """
    the per-shard timing sidecar beside `out_path`'s Parquet
    one function, two callers: `run_matchup_shard` writes it in the worker and
    `scripts/run_sweep.py` reads it in the parent. The name matches none of `*.parquet`,
    `*.tmp` or a dot-name, so no Parquet glob reads it, no rsync exclude drops it, and compose
    carries it with its shard under the shard's composite name.
    :param out_path: the shard's Parquet path (str)
    :returns: the sidecar path (str)
    """
    return os.path.splitext(out_path)[0] + ".timing.json"


def _write_timing(path: str, record: dict) -> None:
    """
    write one timing record atomically, so a kill never leaves half a file
    :param path: the sidecar path (str)
    :param record: started_utc, finished_utc, duration_s and status (dict)
    :returns: None
    """
    write_json_atomic(path, record)


def timed_shard(play):
    """
    wrap a shard function so it leaves its clock beside the Parquet
    The timing sidecar is written when the shard starts, with `started_utc` and no finish, and
    completed in a `finally` with `finished_utc`, `duration_s` and the status. A `finally` does
    not run on SIGKILL, so a wall-killed shard leaves a start with no finish, never no file.
    Time is taken here, in the worker, because the parent would count the pool's queue wait as
    shard time. `duration_s` is a monotonic-clock difference; the UTC stamps are wall clock.
    A decorator, not a rewrite of the shard body, so the body and its source-level pins are
    untouched.
    :param play: a function taking a MatchupSpec and returning the Parquet path (callable)
    :returns: the wrapped function, which returns exactly what `play` returns (callable)
    """

    @functools.wraps(play)
    def run(spec):
        path = timing_sidecar_path(spec.out_path)
        started = datetime.now(timezone.utc)
        t0 = time.monotonic()
        _write_timing(
            path,
            {
                "started_utc": started.isoformat(),
                "finished_utc": None,
                "duration_s": None,
                "status": "started",
            },
        )
        status = "failed"
        try:
            result = play(spec)
            status = "completed"
            return result
        finally:
            _write_timing(
                path,
                {
                    "started_utc": started.isoformat(),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_s": time.monotonic() - t0,
                    "status": status,
                },
            )

    return run


@timed_shard
def run_matchup_shard(spec: MatchupSpec) -> str:
    """Play one matchup shard, write one Parquet file, return its path.

    Also writes `posterior_fallback_sidecar_path(spec.out_path)`: this shard's
    per-game posterior-fallback tally, for the sweep parent to aggregate into
    `run_meta.json`.

    Why a sidecar and not a richer return value. This function's return type
    is load-bearing: `run_shards_local` / `run_shards_ray` return
    `list[Optional[str]]` index-aligned with their specs, with `None` marking a
    failed shard, and `run_sweep.py` zips that list against its own `built` list
    to build the per-matchup summary (see `run_shards_local`'s docstring on why
    the list is never shortened). Widening the element type would touch both
    runners, both of their contracts, and every caller that reads a path out of
    them, to carry a diagnostic. A file beside the Parquet crosses the process
    boundary for free, is identical under serial / local-pool / ray dispatch,
    and survives the run, so the per-shard evidence outlives the aggregate.
    """
    game = load_game(spec.game_name)

    # CFR+ (plus=True) is the Stage-B working reference: the CFRRef agent, the
    # Bayesian likelihood model, and the P3/P5/P7 scoring yardstick - one
    # validated source. Cache-once-freeze: solved once per
    # (game, iters, solver, pyspiel version) key and loaded thereafter.
    # require_cache_hit=True (parallel runs) fails loudly on a miss rather than
    # letting N workers re-solve concurrently (no silent fallbacks).
    depth_pkl = DEPTH_REFERENCES.get(spec.game_name)
    if depth_pkl is not None:
        cfr_policy = load_depth_reference(depth_pkl, game)
    else:
        cfr_policy = load_or_solve_cfr(
            game,
            spec.cfr_iters,
            plus=True,
            cache_dir=spec.cfr_cache_dir,
            require_hit=spec.require_cache_hit,
        )

    # Cache-once-freeze: XFP is trained at most once per (game, iters,
    # seed, pyspiel version) key, making "RL agent is frozen after training"
    # literally true. check_exploitability=False here:
    # the caller (scripts/run_sweep.py) already warms this cache once per game
    # with check_exploitability=True before dispatching shards -- gated on the
    # same "rl" in opponents condition as this block, so whenever this shard
    # actually reaches load_or_train the caller's warm has already run, and a
    # shard-level call is a guaranteed cache hit in normal operation. Only
    # built when this shard actually plays the RL opponent: with
    # require_cache_hit=True a non-RL shard must not fail on a cold RL cache,
    # and a per-shard pickle load for an unused agent is wasted I/O.
    rl_agent = None
    if spec.opponent == "rl":
        rl_agent = RLSelfPlayAgent.load_or_train(
            game,
            num_iterations=spec.rl_iterations,
            seed=spec.seed,
            cache_dir=spec.rl_cache_dir,
            require_hit=spec.require_cache_hit,
        )

    # Five independent streams spawned from the one matchup seed, not
    # `spec.seed` / `spec.seed + 1` integer offsets: with sweep seeds spaced
    # by 1, `+1` would make this shard's opponent stream byte-identical to the
    # Next shard's deal stream. See derive_stream_seeds' docstring.
    streams = derive_stream_seeds(spec.seed)

    cfr_ref = CFRRefAgent(cfr_policy, seed=streams[STREAM_REFERENCE])
    opponent = _build_opponent(
        spec.opponent, game, cfr_policy, streams[STREAM_OPPONENT], rl_agent
    )
    agent0 = _build_agent0(spec, spec.game_name, cfr_policy)
    writer = MatchWriter(spec.out_path)

    # The integer seat_seed match_runner.py's _seat_stream
    # requires (int-only; see _seat_stream's docstring). Derived from its
    # own spawned stream (STREAM_SEAT), never from spec.seed directly, so it
    # cannot collide with the deal/opponent/reference/agent0 streams above.
    # Computed unconditionally (cheap -- one Generator draw) rather than
    # gated on seat_mode: run_match itself is what skips touching seat
    # entropy in fixed modes,
    # so a fixed0 shard passing this value costs nothing and needs no branch
    # here. "its first draw of a bounded integer" -- equal matchup seeds
    # always give equal seat sequences, since this is a pinned pure function
    # of spec.seed via derive_stream_seeds.
    #
    # Under fixed0/fixed1/alternate this value is computed and discarded,
    # which is harmless: it never reaches resolved_config/run_meta.json (no
    # configs/*.yaml key, no CLI flag and no MatchupSpec field exists for it).
    seat_seed = int(np.random.default_rng(streams[STREAM_SEAT]).integers(2**32))

    # All five games have a true_posterior path (Kuhn/Leduc/Liar's Dice
    # dealt-card, Goofspiel reach-weighted bid posterior, Dark Hex cell-occupancy
    # marginal - the last two computed in the match runner), so the oracle
    # (CFR+ policy) is passed for every game.
    posterior_fallback: dict[str, dict[str, int]] = {}
    # The driven-hand discard sink, same contract.
    driven_discards: dict[str, dict[str, int | list[int]]] = {}

    run_match(
        game,
        agent0,
        opponent,
        cfr_ref,
        n_hands=spec.n_hands,
        seed=spec.seed,
        deal_seed=streams[STREAM_DEAL],
        writer=writer,
        game_params=game_params_for(spec.game_name, game),
        oracle_policy=cfr_policy,
        game_key=spec.game_name,
        seat_mode=spec.seat_mode,
        seat_seed=seat_seed,
        probe_type=spec.probe_type,
        # The row-level content identity. Without this line
        # a probe arm's rows would carry run_match's own "base" default while
        # the agent rendered a variant, and the paired scorer would read the
        # pair as two base arms.
        prompt_variant=spec.prompt_variant,
        # The forced opening itself. run_match refuses the
        # offpath_drive label without it, so a dropped argument fails the shard
        # loudly instead of collecting canonical play under a probe label.
        drive_openings=spec.drive_openings,
        # The tally sink. Read here, in the same process that ran the match,
        # and handed on as a file -- a module global would be written in the
        # worker and read in the parent, so on any pooled sweep the parent
        # would read an empty dict.
        posterior_fallback_out=posterior_fallback,
        # The driven-hand discard tally, handed on as a
        # sidecar below for the same process-boundary reason.
        driven_discards_out=driven_discards,
        # The one sha run_sweep.py
        # resolved for the whole sweep, not a fresh per-shard resolution --
        # see MatchupSpec.git_sha's own docstring above.
        git_sha=spec.git_sha,
    )

    # Written unconditionally, including the empty-dict case: "this shard
    # computed no posterior at all" and "this shard's sidecar was never written"
    # are different facts, and the parent must not have to guess which it is
    # holding. Written after run_match returns, so a sidecar's presence means the
    # match completed.
    write_json_atomic(
        posterior_fallback_sidecar_path(spec.out_path), posterior_fallback
    )

    # Written for every driven shard, zero counts
    # included (see driven_discard_sidecar_path); an undriven shard writes
    # nothing so absence stays meaningful.
    if spec.drive_openings:
        write_json_atomic(driven_discard_sidecar_path(spec.out_path), driven_discards)

    return spec.out_path
