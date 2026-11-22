"""Per-node Parquet schema for the LM Optimality Tournament harness.

One NodeRecord per decision node. LM-only columns are present in the schema
but null for non-LM agents, so every agent type writes the same columns.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import polars as pl

from lmopt.prompting.memory import MEMORY_RUNGS

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@dataclasses.dataclass
class NodeRecord:
    # --- Identity ---
    run_id: str
    matchup_id: str  # "{agent0_type}_vs_{agent1_type}"
    game: str  # "kuhn_iter" | "leduc" | "goofspiel"
    game_params: str  # JSON string of game parameters
    seed: int
    hand_idx: int
    node_idx: int  # 0-indexed within the hand
    player_to_act: int

    # --- Attribution ---
    # matchup_id and player_to_act identify the MATCHUP and the per-node
    # Seat, but neither says which seat holds agent0 once seats stop being
    # fixed (per-hand seat randomisation), nor which concrete
    # agent produced a given row's action when a matchup mixes agent types.
    # These two columns make both explicit so a later analysis never has to
    # re-derive them from run_meta.json config.
    lm_seat: int  # the seat agent0 occupies this hand: under seat_mode
    # "random" it varies hand to hand, under
    # "alternate" it is hand_idx % 2,
    # under "fixed0"/"fixed1" it is constant at that
    # seat.
    # Not "the seat of the agent being scored": it
    # names AGENT0's seat, so it points at the
    # Opponent whenever a principled agent is the one
    # under test. Every scorer keys on
    # player_to_act instead, deliberately.
    # run_match fills it unconditionally, never null,
    # so downstream joins never special-case
    # a missing seat.
    actor: str  # identifies which agent acted at this node:
    # agent's model_id when set (LM agents only),
    # else _agent_id(agent) (the type-derived id
    # every agent already reports via matchup_id).
    # Never null -- every agent has at least a type
    # name.

    # --- State ---
    info_set: str  # state.information_state_string(player)
    info_state_tensor: Optional[list[float]]  # nullable; populated where the agent supplies it
    legal_mask: list[int]  # state.legal_actions_mask()

    # --- LM columns (null for non-LM agents) ---
    prompt: Optional[str]
    raw_completion: Optional[str]

    # --- Action ---
    parsed_action: int
    action_str: str  # state.action_to_string(player, action): the
    # game's own human-readable label for
    # parsed_action (e.g. "Bet"/"Pass" in Kuhn), so
    # analysis never has to replay the game from
    # action ints alone to know what happened.
    n_retries: int  # 0 for non-LM agents
    # True when the acting agent could not produce its intended action and fell
    # back: an LM that exhausted illegal-action retries, or the Bayesian agent on
    # a game with no fixed private deal (Goofspiel) where it cannot compute a
    # posterior and uses CFR argmax. Not LM-exclusive - analyses of LM fallback
    # rate must filter on prompt.is_not_null(). (No silent fallbacks.)
    fallback_used: bool
    bet_aliased: Optional[
        bool
    ]  # True when Leduc BET->RAISE alias fired; null/False otherwise
    action_parse_vocab: Optional[
        str
    ]  # Which vocabulary resolved the action: "label"/"verbose"/"coords"/"digit"; null for games without multiple vocabularies or fallback rows

    # --- CFR reference (logged at every node regardless of which agent acts) ---
    cfr_action: int  # argmax of cfr_policy, or pyspiel.INVALID_ACTION (-1)
    # when the reference had no entry for this info-set.
    # Never the executed action, including on a row where
    # the CFR-ref agent itself is player_to_act: what it
    # actually sampled and played is parsed_action/
    # action_str below, which can (and does, on a genuinely
    # mixed node) disagree with this deterministic argmax
    # reference. This column's constancy at a real
    # 50/50-mixed node is expected, not a broken RNG.
    cfr_policy: str  # JSON: '{"0": 0.78, "1": 0.22}'
    # True when the reference had no entry for this info-set and returned its
    # library uniform default. Mandatory and never config-gated: it is what makes
    # the off-support choice reversible offline (scoring is offline),
    # and what stops a uniform substitute from being silent. Always False for the
    # fully-enumerated CFR+ references.
    cfr_ref_off_support: bool

    # --- Belief columns ---
    elicited_belief: Optional[str]  # LM-stated distribution; null for non-LM agents
    true_posterior: Optional[str]  # JSON exact posterior, where one is computed

    # --- Belief provenance ---
    # Without these columns a belief call would be a single unlogged attempt
    # that silently yields None on parse failure (a silent fallback: liarsdice
    # can be null on 76.6% of belief rows with no retry logged). These three
    # columns are the belief-side counterpart of raw_completion/n_retries/fallback_used
    # above, with one extra distinction those don't need: a belief can be
    # legitimately SKIPPED (belief_suffix's own str | None contract,
    # prompting/__init__.py -- returns None when no belief question is
    # askable at this info-set) rather than asked-and-failed, so
    # belief_failed must stay False on a skip or it manufactures a failure
    # statistic out of nothing (see lm_vllm.py's act() for how the three columns jointly distinguish "never asked" from
    # "asked and unparseable" from "asked and parsed").
    belief_raw_completion: Optional[str]  # first belief-call attempt's raw
    # text; non-null iff a belief call
    # was actually made (null on skip)
    belief_n_retries: int  # extra belief-call attempts after the first;
    # 0 for non-LM agents, for a skipped belief, and
    # for a belief that parsed on the first attempt
    # (mirrors n_retries' own "0 means no extra
    # attempts were needed", not "no attempt made")
    belief_failed: bool  # True only when a belief call was made, every
    # attempt failed to parse and no recovery
    # (belief_salvage_kind) supplied a belief; False
    # for non-LM agents, for a skipped belief, and
    # for a parsed or salvaged belief

    # --- Memory configuration ---
    # The agent's MEMORY_RUNGS rung on LM rows, null on non-LM rows. Where
    # non-null, values are constrained to MEMORY_RUNGS (imported from
    # lmopt.prompting.memory). Reader-side M0 defaulting for arms collected
    # before this column existed is handled by the scoring condition filter,
    # not here; this column records what the agent actually was.
    memory_rung: Optional[str]
    memory_truncated: Optional[bool]  # agent's last_memory_truncated; null for non-LM

    # --- Outcome ---
    reward: float  # hand-level terminal return for player_to_act
    probe_type: str  # which manipulation produced this row. All four
    # members of PROBE_TYPE_DOMAIN are emitted by a
    # sweep: "none", "relabel",
    # "payoff_scale" and "offpath_drive". Full
    # declared domain:
    # PROBE_TYPE_DOMAIN below; a sweep validates
    # against it before any CFR solve. A fifth member,
    # "option_order_reversed" (prompt_variant
    # "option_order_reversed_v1"), is a diagnostic
    # probe: its rows feed no scoreboard cell.
    prompt_variant: str  # Which prompt content produced this row:
    # "base" on a
    # canonical arm, "relabel_v1" / "payoff_scale_x10"
    # on a prompt-varying probe arm, and "base" on a
    # driven arm, whose manipulation is the match and
    # not the prompt. On the same footing as prompt_schema_version
    # above: a prompt-content change makes prior rows
    # incomparable, so which variant produced a row
    # must ride with the row. prompt_spec_hash beside
    # it pins the BYTES, so a variant arm cannot log
    # a canonical digest.

    # --- Provenance ---
    backend: str  # e.g. "cpu" for non-LM agents
    model_revision: Optional[str]  # harness git SHA at collection time
    timestamp: str  # ISO 8601

    # --- Model/prompt provenance ---
    # model_revision (above) holds the harness git SHA, not the served model
    # id (match_runner.py stamps model_revision=git_sha unconditionally), so
    # without the columns below the model id would survive only in an output
    # directory name. These three are additive, not a redefinition of
    # model_revision, whose meaning is already established in collected data.
    # The served weights identity is the separate `served_weights_revision`
    # column at the end of this class -- additive on the same rule.
    model_id: Optional[str]  # served model identifier (LMVllmAgent's `model`
    # constructor arg, e.g. "qwen2.5-14b-instruct");
    # null for non-LM agents, same null-for-non-LM
    # pattern as prompt/raw_completion above
    chat_template_hash: Optional[str]  # hash of the chat template the served
    # endpoint rendered this row's prompt under.
    # The sweep path resolves it once per run
    # (run_sweep.py's resolve_chat_template_hash
    # call), threads it through
    # MatchupSpec.chat_template_hash, and
    # orchestration/shard.py passes it into the
    # agent, from which match_runner's
    # _lm_provenance getattr stamps it onto the
    # row, so it is populated on sweep-produced
    # LM arms. It is null on paths that build an
    # agent without a hash, and null for
    # non-LM agents regardless, so a null here
    # means "this producer did not resolve one",
    # never "the mechanism does not exist".
    # The /tokenize probe forwards the run's
    # chat_template_kwargs when set, so on a kwargs
    # arm this fingerprints the template as applied
    # with those kwargs; with no kwargs the probe
    # payload carries none.
    # Resolution uses the same /tokenize round
    # trip exact_policy_pass's
    # render_chat_prefix_ids uses --
    # never a second, independently-invented
    # mechanism -- hashed once per run, not per
    # request (see LMVllmAgent's docstring for
    # the granularity discussion).
    prompt_schema_version: Optional[int]  # version of the prompting code
    # (src/lmopt/prompting/*.py) that produced this
    # row's `prompt`. Value 1 denotes the
    # prompt schema with leduc rank-space
    # private/public card naming + full cross-round
    # betting history and dark hex move-count +
    # own-move-history rendering. A bump means a change to
    # prompting/*.py that alters what a rendered
    # prompt contains at a fixed info-set (not mere
    # wording), i.e. a change that could shift the
    # measured action/belief distribution -- see
    # lm_vllm.py's PROMPT_SCHEMA_VERSION constant,
    # the single source of truth for the current
    # value. Null for non-LM agents (no prompt was
    # rendered for that row).
    prompt_spec_hash: Optional[str]  # SHA256 of the prompt content this row
    # was rendered under: spec.prompt_spec_hash()
    # over the five GamePromptSpecs, with this arm's
    # variant substituted for its game. Null for
    # non-LM agents, on the same
    # rule as prompt_schema_version above.
    #
    # Why it is A column and not prompt_variant plus
    # A convention. prompt_schema_version is a coarse
    # integer that denotes a content digest through
    # lm_vllm.PROMPT_SPEC_HASHES, so on a probe arm
    # it positively asserts canonical content against
    # variant bytes -- a fabricated provenance rather
    # than a missing one. A reader checks a base row with
    # `hash == PROMPT_SPEC_HASHES[version]` and a
    # probe row by recomputing the registry variant's
    # digest; prompt_variant names which variant,
    # this column proves what it contained. Same
    # under-sensitivity as the registry entry it is
    # compared against: it covers GamePromptSpec
    # content, not the rendering code (the byte pin is what covers
    # that).
    belief_protocol: Optional[str]  # how this row's belief call was decoded:
    # "free" (unconstrained generation) or a named
    # structured-output
    # protocol such as "schema_reasoning_belief"
    # (lm_vllm.py's BELIEF_PROTOCOLS is the single
    # source of truth for the names, and refuses any
    # other).
    #
    # Why it is A column and not just A CONFIG KEY.
    # Decoding protocol is a manipulated variable,
    # not a wrapper: one goofspiel info-set
    # answered {.1,.4,.4,.1} free,
    # {.33,.33,0,.33} under a belief-only schema and
    # {.2,.3,.1,.4} under {reasoning,belief}. Two
    # rows collected under different protocols are
    # therefore no more comparable than two rows
    # under different prompts.
    # run_meta.json's embedded resolved_config makes
    # the choice recoverable PER RUN, which is not
    # enough: a per-matchup override or a mid-arm
    # change leaves no trace at all, which is the
    # defect prompt_spec_hash had.
    # Null for non-LM agents (nothing was decoded);
    # NON-null on every LM row, including rows where
    # no belief was asked -- it describes the run's
    # decoding protocol, and "was a belief asked
    # here" is what belief_raw_completion answers.
    # Arms collected before the column existed lack
    # it, exactly as they lack prompt_schema_version;
    # readers must treat its absence as "free", the
    # only protocol those runs could use.

    # --- Reasoning trace columns ---
    # On a --reasoning-parser arm the server splits the trace out of content,
    # so raw_completion holds the post-split answer and the full completion is
    # reasoning_trace + raw_completion; on a no-parser arm raw_completion holds
    # everything as today; the per-arm difference is keyed by
    # run_meta.serving.vllm_args. Null for non-LM agents.
    # All seven carry first-attempt semantics, like raw_completion and
    # belief_raw_completion: they describe attempt 0 of their retry loop, not
    # the attempt whose parse succeeded (lm_vllm.py captures them under
    # `if attempt == 0`). On a row with n_retries > 0 they therefore describe a
    # Different response than parsed_action came from. Stated here because the
    # schema comment is what an analyst reads.
    reasoning_trace: Optional[str]  # the action call's first
    # attempt's reasoning trace. Two sources, in this
    # order (lm_vllm._compose_trace_and_answer):
    # (1) the server's own `reasoning_content` field
    # when present, i.e. a --reasoning-parser arm;
    # (2) otherwise `split_reasoning(content)`'s
    # trace half -- the `<think>...</think>` block
    # split out of `content` locally, which is the
    # Entire mechanism on a no-parser arm. Null when neither yields one
    # (no `reasoning_content`, no `</think>`
    # delimiter) or on a non-LM row. Logged in
    # match_runner alongside raw_completion, which is
    # `content` verbatim: on a --reasoning-parser arm
    # the server already removed the trace, so
    # `content` is the answer; on a no-parser arm
    # `content` still holds trace + answer together,
    # so raw_completion and reasoning_trace overlap
    # there rather than partitioning the completion.
    belief_reasoning_trace: Optional[str]  # the belief call's first attempt's
    # reasoning trace, same two sources and same
    # split_reasoning fallback as reasoning_trace
    # (null if neither yields one, if no belief was
    # asked, or on a non-LM row). In elicitation_mode
    # "combined" there is only one response, so this
    # mirrors reasoning_trace by construction.
    finish_reason: Optional[str]  # the action call's first attempt's
    # finish_reason (e.g. "stop", "length"), null if
    # the server omits it or on a non-LM row.
    belief_finish_reason: Optional[str]  # the belief call's first attempt's
    # finish_reason, null if the server omits it, if
    # no belief was asked, or on a non-LM row.
    belief_prompt: Optional[str]  # (Utf8) the exact rendered prompt bytes this
    # row's belief call was made with, null if no
    # belief was asked or on a non-LM row. In
    # elicitation_mode "combined" the belief prompt
    # Equals the action prompt: that mode issues one
    # request, carrying the action suffix, and reads
    # the belief out of the same response -- so this
    # column equals `prompt` there (not base +
    # belief_suffix, bytes combined mode never
    # sends). `elicitation_mode` in run_meta says
    # which mode a row came from.
    completion_tokens: Optional[int]  # (Int64) the action call's first
    # attempt's token count from
    # usage.completion_tokens, null if the server
    # omits usage or on a non-LM row.
    belief_completion_tokens: Optional[int]  # (Int64) the belief call's first
    # attempt's usage.completion_tokens, null if the
    # server omits it, if no belief was asked, or on
    # a non-LM row.

    # --- Served weights provenance ---
    served_weights_revision: Optional[str] = None  # the HF weights revision
    # (snapshot hash) of the model the endpoint was
    # serving when this row was collected. Belongs
    # with the Model/prompt provenance block above
    # (see its comment); it sits at the END of the
    # class only because it is the one defaulted
    # field (nullable, so every existing construction
    # site keeps working) and dataclasses require
    # defaults last. Never resolved by run_sweep
    # itself: the value lives in the serving box's
    # HF cache, which the sweep's HTTP-only serving
    # resolution cannot see -- the launcher
    # (run_experiment.py, via
    # lmopt.provenance.weights_cache's
    # resolve_prelaunch_revision) resolves it once
    # per arm and passes it as
    # `--served-weights-revision`; run_sweep threads
    # it exactly like chat_template_hash
    # (MatchupSpec -> shard -> LMVllmAgent ->
    # match_runner getattr -> here) and a real sweep
    # launched without it aborts at startup. Null on
    # non-LM rows, on rows from paths that build an
    # agent without one, and
    # on every arm collected before this column
    # existed: a null means "this producer did not
    # resolve one", never a fabricated placeholder
    # (no silent fallbacks). Additive:
    # model_revision above keeps its established
    # harness-git-sha meaning untouched.

    # --- Tokenizer provenance ---
    tokenizer_revision: Optional[str] = None  # the HF tokenizer revision
    # (snapshot hash) the served endpoint loaded,
    # when it differs from served_weights_revision.
    # Threaded through the identical hop points as
    # served_weights_revision (MatchupSpec -> shard
    # -> LMVllmAgent -> match_runner getattr ->
    # here). Null on non-LM rows and whenever the
    # arm never declared or observed a distinct
    # tokenizer revision (no silent fallbacks).

    # --- Request-seed re-derivability ---
    # Why two columns when `node_idx` and `seed` already exist: the per-request
    # seed derivation (lm_vllm.derive_request_seed) keys on the AGENT's own
    # per-decision ordinal within the hand, which the logged `node_idx`
    # (all players' decisions) does not equal whenever another player acts
    # between two of this agent's decisions -- 57.2% of the Qwen-14B arm's LM
    # rows. These columns log the missing quantities instead of re-keying the
    # derivation, so the seed values sent and the Agent protocol are
    # untouched. Both nullable Int64, additive on the same rule as
    # served_weights_revision (defaulted, every existing construction site
    # keeps working; old arms null-fill through the diagonal_relaxed readers).
    agent_node_idx: Optional[int] = None  # the acting agent's per-decision
    # ordinal within the hand -- the value its
    # private counter held when it made this
    # decision (LMVllmAgent exposes it as
    # last_agent_node_idx at act() time; match_runner
    # reads it off the agent via the same getattr
    # pattern as n_retries). Null for agents that do
    # not expose one (every non-LM agent). Equals
    # this row's rank among the hand's rows by the
    # Same actor, never the all-players node_idx
    # above.
    run_seed: Optional[int] = None  # the run-level seed the acting agent's
    # request seeds were derived from -- read off
    # the agent (its run_seed constructor kwarg),
    # never copied from the runner's match `seed`.
    # On the sweep path shard.py binds
    # run_seed=spec.seed, so there it equals the
    # `seed` column by construction; it is a column
    # anyway because null here means "this row's
    # requests carried no seed" (unseeded agent,
    # and
    # every non-LM row), which the required-non-null
    # `seed` column can never express. For every
    # logged LM row with run_seed non-null,
    # derive_request_seed(run_seed, hand_idx,
    # agent_node_idx, call_kind, attempt) is the
    # seed that row's request actually carried.
    # Mode scope: the action leg
    # (call_kind "action", attempts
    # 0..n_retries) applies to every seeded LM row;
    # the belief leg (call_kind "belief", attempts
    # 0..belief_n_retries, gated on
    # belief_raw_completion non-null) applies to
    # Sibling-mode rows only. Combined mode sends
    # No belief request -- lm_vllm.py's combined
    # branch fills belief_raw_completion by reusing
    # the single shared action completion -- so a
    # combined row has no belief seeds to
    # reconstruct, and applying the belief leg to
    # it would fabricate seeds for requests that
    # never hit the wire. run_meta's
    # elicitation_mode names a row's mode.
    # --- Belief salvage provenance ---
    # A belief the strict parser rejected on every attempt can be recovered
    # when its only defect is a bare integer key (`spec.salvage_belief_object`).
    # A recovered belief is stored in `elicited_belief` with `belief_failed`
    # False, so it is scored as parsed; these two columns keep the recovery
    # visible. Both are null for a clean parse, a failed belief, a skipped
    # belief, every non-LM row, and every arm collected before they existed.
    belief_salvage_kind: Optional[str] = None  # what the recovery did;
    # "key_quoted" is the only value: it inserted
    # quote characters around a bare integer key
    # and changed no value.
    belief_salvage_attempt: Optional[int] = None  # the belief attempt
    # (0-based) whose text supplied the belief.
    # `belief_raw_completion` logs attempt 0 only, so
    # for a value above 0 the source text is not in
    # the log. `belief_n_retries` equal to the retry
    # cap with `belief_failed` False is ambiguous
    # between "parsed on the last attempt" and a
    # salvage; a non-null `belief_salvage_kind`
    # settles it.

    # --- Per-call fields ---
    # Every column in this block is nullable, defaulted (so every existing
    # construction site keeps working) and additive: no column above changes name,
    # type or meaning. Arms collected before the block existed lack the columns
    # entirely; `load_frames` reads them as null, while `concat_strict` refuses a
    # cell that mixes the two, by design.
    #
    # The four `call_latency_s` / `prompt_tokens` / `response_id` / `request_seed`
    # columns describe the action call's first attempt, exactly like
    # raw_completion, finish_reason and completion_tokens. They are not the
    # decision's total latency or token cost: a row with n_retries > 0 made more
    # calls than these describe, and every attempt is recorded in the per-call log.
    # The `belief_` twins describe the first sibling-mode belief call. Combined
    # elicitation mode sends no belief request, so the twins are NULL there even
    # though belief_finish_reason and belief_completion_tokens mirror the action
    # call in that mode by their own older convention. Never mirror these.
    call_latency_s: Optional[float] = None  # wall seconds around the client call
    # only (time.monotonic, before the request is sent to
    # after the SDK returns), attempt 0. A call that
    # raises has no latency here.
    belief_call_latency_s: Optional[float] = None
    prompt_tokens: Optional[int] = None  # usage.prompt_tokens as the server
    # reported it; null when the server sends no usage
    # block or a non-integer count (a bool is not a count).
    belief_prompt_tokens: Optional[int] = None
    response_id: Optional[str] = None  # the response's own id; null when absent or
    # not a string.
    belief_response_id: Optional[str] = None
    request_seed: Optional[int] = None  # the seed value actually passed to the
    # client for this call (null when none was sent), not
    # a recomputation from run_seed. For attempt 0 it is
    # also derivable from run_seed and agent_node_idx.
    belief_request_seed: Optional[int] = None
    winning_attempt: Optional[int] = None  # 0-based index of the action attempt
    # whose answer parsed to a legal action. Null on a
    # fallback row (no attempt won); n_retries is the cap
    # there, so it must never stand in for this.
    winning_raw_completion: Optional[str] = None  # that attempt's response
    # content verbatim (not the post-split answer), null
    # on a fallback row. raw_completion stays the first
    # attempt's text.
    memory_chars_raw: Optional[int] = None  # len() in characters of the rendered
    # memory block before truncation. Null when no block
    # was rendered (M0, or an empty block), the same
    # condition under which memory_truncated is null.
    memory_chars_sent: Optional[int] = None  # len() in characters of the block that
    # went into the prompt, after truncation. Same nulls.
    posterior_fell_back: Optional[bool] = None  # True when the oracle posterior used
    # its uniform fallback for this row (every hypothesis
    # had zero likelihood). Set only on the oracle-policy
    # branch; null wherever that branch is not taken
    # (Goofspiel and Dark Hex use precomputed maps, and runs
    # with no oracle policy compute none). On a driven row
    # it records the real fallback even though
    # true_posterior is nulled there.


# The full domain probe_type may take (the probe modes). The four scored
# probe types are emitted by a sweep. Declared here, rather
# than only inline where a probe injector sets the field, so every consumer
# (and test_parquet_schema.py's domain pin test) has one place to check a
# probe_type string against instead of re-deriving the allowed set.
# "option_order_reversed" is a diagnostic probe: it lists the legal options in the opposite order
# and feeds no scoreboard cell, so it is a member of this domain and of no acceptance table.
PROBE_TYPE_DOMAIN = frozenset(
    {"none", "relabel", "payoff_scale", "offpath_drive", "option_order_reversed"}
)

# The full domain memory_rung may take (lmopt.prompting.memory.MEMORY_RUNGS),
# imported rather than re-typed so the domain cannot drift from the source of
# truth. Declared here beside PROBE_TYPE_DOMAIN for the same reason: every
# consumer and test_parquet_schema.py's domain pin test has one place to check
# a memory_rung string against.
MEMORY_RUNG_DOMAIN = frozenset(MEMORY_RUNGS)

# The one member of the domain that manipulates the match rather than the
# prompt: rows carry it exactly when the hand was played
# from a forced opening (`run_match(drive_openings=...)`). Named here, beside
# the domain it belongs to, because three layers check the label against the
# manipulation (the runner, MatchupSpec, run_sweep) and the P6 scorer refuses
# any other value: a string literal repeated in five files is a typo away from
# a check that never fires. Kept out of the runner so that scoring, which is
# offline, does not import the harness to learn a label.
OFFPATH_PROBE_TYPE = "offpath_drive"

# The domain member every UN-probed row carries. Named here for
# the same reason as OFFPATH_PROBE_TYPE: the plain (non-paired) scorers
# (P2/P3/P5/P7/P8) score exactly one arm's own base population and refuse a
# frame that also carries a probe arm's rows, so this string is the value
# their guard checks every row against rather than re-typing it per scorer.
BASE_PROBE_TYPE = "none"


# Polars schema used by MatchWriter and round-trip tests.
PARQUET_SCHEMA: dict[str, PolarsDataType] = {
    "run_id": pl.Utf8,
    "matchup_id": pl.Utf8,
    "game": pl.Utf8,
    "game_params": pl.Utf8,
    "seed": pl.Int64,
    "hand_idx": pl.Int64,
    "node_idx": pl.Int64,
    "player_to_act": pl.Int64,
    "lm_seat": pl.Int64,
    "actor": pl.Utf8,
    "info_set": pl.Utf8,
    "info_state_tensor": pl.List(pl.Float32),
    "legal_mask": pl.List(pl.Int64),
    "prompt": pl.Utf8,
    "raw_completion": pl.Utf8,
    "parsed_action": pl.Int64,
    "action_str": pl.Utf8,
    "n_retries": pl.Int64,
    "fallback_used": pl.Boolean,
    "bet_aliased": pl.Boolean,
    "action_parse_vocab": pl.Utf8,
    "memory_rung": pl.Utf8,
    "memory_truncated": pl.Boolean,
    "cfr_action": pl.Int64,
    "cfr_policy": pl.Utf8,
    "cfr_ref_off_support": pl.Boolean,
    "elicited_belief": pl.Utf8,
    "true_posterior": pl.Utf8,
    "belief_raw_completion": pl.Utf8,
    "belief_n_retries": pl.Int64,
    "belief_failed": pl.Boolean,
    "reward": pl.Float64,
    "probe_type": pl.Utf8,
    "prompt_variant": pl.Utf8,
    "backend": pl.Utf8,
    "model_revision": pl.Utf8,
    "timestamp": pl.Utf8,
    "model_id": pl.Utf8,
    "chat_template_hash": pl.Utf8,
    "prompt_schema_version": pl.Int64,
    "prompt_spec_hash": pl.Utf8,
    "belief_protocol": pl.Utf8,
    "reasoning_trace": pl.Utf8,
    "belief_reasoning_trace": pl.Utf8,
    "finish_reason": pl.Utf8,
    "belief_finish_reason": pl.Utf8,
    "belief_prompt": pl.Utf8,
    "completion_tokens": pl.Int64,
    "belief_completion_tokens": pl.Int64,
    # Appended, so old arms' parquet (no such column) keeps loading via the
    # diagonal_relaxed null-fill readers.
    "served_weights_revision": pl.Utf8,
    # Appended on the same rule.
    "tokenizer_revision": pl.Utf8,
    # Appended on the same rule; see the
    # NodeRecord field comments for the re-derivability contract.
    "agent_node_idx": pl.Int64,
    "run_seed": pl.Int64,
    # Belief salvage provenance: appended on the same rule.
    "belief_salvage_kind": pl.Utf8,
    "belief_salvage_attempt": pl.Int64,
    # Per-call fields: appended on the same rule; see NodeRecord.
    "call_latency_s": pl.Float64,
    "belief_call_latency_s": pl.Float64,
    "prompt_tokens": pl.Int64,
    "belief_prompt_tokens": pl.Int64,
    "response_id": pl.Utf8,
    "belief_response_id": pl.Utf8,
    "request_seed": pl.Int64,
    "belief_request_seed": pl.Int64,
    "winning_attempt": pl.Int64,
    "winning_raw_completion": pl.Utf8,
    "memory_chars_raw": pl.Int64,
    "memory_chars_sent": pl.Int64,
    "posterior_fell_back": pl.Boolean,
}

# The per-call columns, in one place so a test can name them. Every
# one is null on non-LM rows and on every arm collected before the block existed.
NEW_CALL_COLUMNS = (
    "call_latency_s",
    "belief_call_latency_s",
    "prompt_tokens",
    "belief_prompt_tokens",
    "response_id",
    "belief_response_id",
    "request_seed",
    "belief_request_seed",
    "winning_attempt",
    "winning_raw_completion",
    "memory_chars_raw",
    "memory_chars_sent",
    "posterior_fell_back",
)

# Columns that must never be null in a valid record.
REQUIRED_COLUMNS = frozenset(
    {
        "run_id",
        "matchup_id",
        "game",
        "game_params",
        "seed",
        "hand_idx",
        "node_idx",
        "player_to_act",
        "info_set",
        "legal_mask",
        "parsed_action",
        "n_retries",
        "fallback_used",
        "cfr_action",
        "cfr_policy",
        "cfr_ref_off_support",
        "reward",
        "probe_type",
        "backend",
        "timestamp",
        # belief_n_retries/belief_failed mirror n_retries/fallback_used above:
        # never null, 0/False default for non-LM agents and for a skipped belief.
        # belief_raw_completion is deliberately not required --
        # null is the correct, meaningful value on a skip or a non-LM row.
        "belief_n_retries",
        "belief_failed",
        # lm_seat/actor/action_str/prompt_variant are filled
        # unconditionally by run_match for every agent type (unlike prompt/
        # model_id, which are legitimately null for non-LM agents), so all four
        # belong in REQUIRED_COLUMNS rather than the nullable set.
        "lm_seat",
        "actor",
        "action_str",
        "prompt_variant",
    }
)
