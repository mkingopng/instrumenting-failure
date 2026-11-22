"""The pilot pre-flight gate.

Before a multi-day GPU experiment commits to a full-size sweep of a model
("arm"), run a short pilot (`--n-hands`, default 20) and evaluate it against
the Tier-1 criteria below. Criteria 1-4 are pre-registered, and every
threshold below is a measured value, not an arbitrary one. Criteria 5 and 6
carry no calibrated threshold because they are not rate comparisons: they ask
whether this pilot is evidence about the code about to run, and whether it
finished at all. Criterion 7 reads a threshold another module pre-registered:

  1. verbose completions: action `raw_completion` longer than
     `max_words` (5) normalised words, rate > `max_verbose_rate` (0.05).
     Primary criterion. Good arms measure 0.00000 (hard max 2 words
     across 871k+ rows); a non-compliant small model 1.00000.
  2. belief failure: `belief_failed` rate >= `max_belief_failure_rate`
     (0.90), restricted to rows where a belief call was actually made
     (`belief_raw_completion` non-null). Qwen's worst (calibration arm)
     measures 0.7659; a non-compliant small model 1.0000.
  3. action fallback: `fallback_used` rate > `max_fallback_rate` (0.50).
     Prospective -- zero measured separation between good (0.0410 worst)
     and bad (0.5156/0.6451) arms, so this threshold is conservative, not
     discriminating.
  4. non-vacuity: every requested game must have >= `min_rows_per_game`
     (100) LM rows in the data. Catches "the sweep silently dropped a
     game" rather than reporting it as clean.
  5. prompt-schema staleness: every scored row's
     `prompt_schema_version` must equal the live
     `lmopt.agents.lm_vllm.PROMPT_SCHEMA_VERSION`. A pilot is evidence
     about the code that produced it, and a passing pilot is a
     precondition for launching an arm -- so a pilot describing a prompt
     that no longer exists certifies nothing. More than one distinct
     version, a null version, or no column at all is also a refusal:
     "cannot tell" is not "passes".
  6. completeness: every shard `scripts/run_sweep.py` was asked
     to run must appear in its `shard_outcomes.json` record with status
     "completed". Criterion 4 catches "the sweep dropped a GAME"; it
     cannot catch "the sweep dropped half a game" (for example 11 of 20
     goofspiel shards, whose 660 surviving LM rows clear the 100-row
     vacuity floor 6.6x). An ABSENT or unreadable record is a refusal for
     the same reason as above.
  7. exact-policy coverage: when the pilot directory
     holds an `exact_policy.csv`, the median `coverage_mass` over its
     non-null rows must clear `COVERAGE_MASS_FLOOR` (0.5, pre-registered
     in and read from scripts/exact_policy_pass.py). That pass re-queries
     the live endpoint for the probability the served model puts on each
     legal action's own literal; a model whose mass lands elsewhere yields
     no usable probability data at all. Checking it here moves detection
     from after hours of GPU per arm to the 2-minute pilot. An
     ABSENT artefact while the config enables the pass
     (`exact_policy.enabled`, resolved in run_experiment.py's `main`) is a
     refusal for
     criterion 6's reason: "cannot tell" is not "passes". With the pass
     not enabled and no artefact present, the criterion reports "not
     evaluated" and does not fire. This gate remains a pure consumer: it
     reads the artefact the pass already wrote and scores no info-set
     itself.
     A refusal here is evidence about the model and its artefact, not a
     suspicion about this pass. Size the expectation accordingly:
     legitimate readings reach 0.619 at minimum, and liarsdice's median is
     0.769, so the headroom over the floor is 0.12-0.27.
  8. belief-channel verbosity: `belief_completion_tokens`
     rate > `max_belief_tokens`, exceeding `max_belief_verbose_rate` of
     belief calls made. Not active by default -- both threshold fields are
     None until a caller sets them explicitly (property thresholds are not
     this project's to invent). Its state is always
     printed (not-configured / not-evaluated / measured), never silent.
     Motivating case: an arm can pass the other seven criteria cleanly
     at pilot scale while its belief tokens run far above its own action
     channel, and BELIEF_MAX_TOKENS=2048 (a stall-detection ceiling, not
     a quality gate) is never approached.

  9, 10. the stamp's cap: `belief_failed` and `fallback_used` each
     over `FAILURE_RATE_CAP` (recycle_arm_evidence.py, read from its source, never
     re-typed here) of the pilot's LM rows, strictly, over the whole pilot and not per
     game. Criteria 2 and 3 above are the BROKEN-pipeline guards (their thresholds
     were measured against a bad model) and cannot predict the stamp's refusal, which
     fires at 1%; these two can. Without them a pilot at, say, 50% belief failure would
     PASS and the arm collected from it would then be refused. They are a tripwire and
     not a predictor: a pilot is a few thousand LM rows (about 3,700 to 4,300 for the
     v7 and v8 pilots), so it shows zero for a rate like 1 in 79,923, and the rate moves
     with the memory level (a v4 pilot read 5.1% fallback at M2, 0.29% at M0). The stamp
     can admit a measured belief_failed
     count (`--admit-belief-failed`); this gate cannot, so a pilot refusal here is not
     waivable. An arm at the broken-pipeline level usually fires both pairs, but not always: 2 and 3 are
     per game and 9 and 10 are over the whole pilot.

Any criterion firing on any game aborts the arm (exit code
GATE_FAILURE_EXIT_CODE=2, distinct in kind from exit code 1 = an
operational failure of the gate itself -- bad config, unsafe --out-dir, a
failed run_sweep.py subprocess -- following the EXACT_POLICY_FAILURE_EXIT_CODE
precedent in scripts/run_experiment.py: "the arm is bad" and "the tool
broke" must never share a code).

Tier 2 (per-game thresholds calibrated from the first, labelled arm) is not
implemented here and must not be invented in this file -- those numbers are
pre-registered from the Qwen calibration run before the
next arm runs. `compare_to_reference` below is the mechanism Tier 2 will use
(a change-detector against a labelled reference directory); it is wired into
main() but no Tier-2 threshold is loaded by this file.

This script is a pure consumer of the logged Parquet (scoring is offline):
it reads `src/lmopt/logging/schema.py`'s columns
directly with polars and never imports OpenSpiel.
"""

import argparse
import dataclasses
import functools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

_REPO = Path(__file__).resolve().parent.parent

GAMES: tuple[str, ...] = ("kuhn", "leduc", "goofspiel", "liarsdice", "darkhex")

# Distinct in kind from exit code 1 (operational failure of the gate itself):
# 2 means "the arm is unusable", following the EXACT_POLICY_FAILURE_EXIT_CODE
# (=3) precedent in scripts/run_experiment.py -- a fired criterion and a
# broken tool must never share a code.
GATE_FAILURE_EXIT_CODE = 2

# The two criteria below are properties of the directory, not of any
# one game, so their violations carry this in the `game` slot rather than a
# game name (`_print_report` prints it verbatim).
DIRECTORY_SCOPE = "(directory)"

# The logged column carrying the prompt-code version each row was produced
# under (src/lmopt/logging/schema.py; fed by lm_vllm.py's
# PROMPT_SCHEMA_VERSION, whose comment is the single source of truth for what
# each value means). Absent from arms collected before the column existed.
PROMPT_SCHEMA_COLUMN = "prompt_schema_version"

# Mirror of scripts/run_sweep.py's own SHARD_OUTCOMES_FILENAME -- the
# completeness record it writes on every run, clean or not. Must agree
# with run_sweep.py's own source, for the reason
# _RUN_SWEEP_DEFAULTS below records at length: this is a second copy of a
# value, and the invariant is that it equals the original in run_sweep.py.
# A silently misspelled filename here would read as "no record
# written", which is a refusal rather than a false pass, but it would refuse
# every healthy arm -- the other catastrophic direction.
SHARD_OUTCOMES_FILENAME = "shard_outcomes.json"

# Criterion 7. The artefact scripts/exact_policy_pass.py writes, named by
# scripts/run_experiment.py's `exact_policy_pass_command` (which writes it)
# and `analyse_stats_command` (which reads it back via `--exact-policy`)
# for every arm. Second copy of another module's filename, so it carries the
# same agreement requirement SHARD_OUTCOMES_FILENAME does: a
# misspelling here reads as "no artefact written", which refuses every
# healthy arm rather than passing a bad one -- still one of the two
# catastrophic directions.
EXACT_POLICY_FILENAME = "exact_policy.csv"

# The column of that artefact criterion 7 reads: the probability mass the
# served model put on the legal actions' own literals at one info-set
# (exact_policy_pass.py's _OUTPUT_SCHEMA).
COVERAGE_MASS_COLUMN = "coverage_mass"

# The columns the exact-policy parse RATE reads from that artefact. An info-set is
# identified by `EXACT_POLICY_PARSE_KEY`, and its distribution is trusted exactly
# when no row of it carries a value in `EXACT_POLICY_ERROR_COLUMN`. The pass writes
# an info-set-level failure (coverage below the floor, an unusable prompt) onto
# every row of that info-set, and a partially scored action set leaves the error on
# the failed action rows, so "no row has an error" is the one test that covers both.
EXACT_POLICY_PARSE_KEY = ("game", "game_params", "info_set")
EXACT_POLICY_ERROR_COLUMN = "error"
# The classification of an error row (`ErrorKind`'s value), and the label for a failed
# info-set whose rows carry an error but no kind, so no failure is dropped from the count.
EXACT_POLICY_ERROR_KIND_COLUMN = "error_kind"
UNCLASSIFIED_ERROR_KIND = "unclassified"
# The memory rung above which the exact pass does not run, so no parse rate exists.
NO_MEMORY_RUNG_FOR_EXACT_POLICY = "M0"
# The rungs above it that the structural wording may be printed for.
MEMORY_RUNGS_ABOVE_M0 = ("M1", "M2")

_LM_VLLM_SOURCE = _REPO / "src" / "lmopt" / "agents" / "lm_vllm.py"
_EXACT_POLICY_SOURCE = _REPO / "scripts" / "exact_policy_pass.py"
_RECYCLE_EVIDENCE_SOURCE = _REPO / "scripts" / "recycle_arm_evidence.py"

# The one memory level that renders no memory block, so a pilot at it certifies no wording
# and is not asked for one. The run_meta key the wording digest is recorded under is
# `lmopt.prompting.memory.MEMORY_RENDER_HASH_KEY`, imported where it is read, so the writer
# (scripts/run_sweep.py) and this reader cannot name two different keys.
NO_MEMORY_RUNG = "M0"

# ---------------------------------------------------------------------------
# The resolved-config defaults. This gate drives scripts/run_sweep.py, so
# every shape it predicts must be resolved with run_sweep.py's own defaults.
# Re-deriving the resolved config in several places with different defaults
# produces false exits (e.g. assuming all five games where run_sweep.py
# defaults to three gives a false exit 2; assuming no opponents gives a false
# exit 1).
#
# Invariant: these defaults equal run_sweep.py's own. So: one mirror of
# scripts/run_sweep.py's `_merge` setdefault block (the source of truth --
# edit there first),
# applied through one function.
#
# NOTE the semantics are `setdefault`, verbatim: a key that is present but
# null/empty is left exactly as the config wrote it, never replaced. That
# matters -- `games:` (YAML null) must reach _validate_games and be
# rejected as the config error it is, not silently healed into a default.
_RUN_SWEEP_DEFAULTS: dict[str, object] = {
    "games": ["kuhn", "leduc", "goofspiel"],
    "opponents": ["cfr_ref", "bayesian", "rl", "always_bet"],
    "n_seeds": 1,
    # So _resolve_sweep_config can tell _verify_pilot_rung what this launch's
    # config resolves to when either key is omitted. Verbatim mirror of
    # run_sweep.py `_merge`'s own setdefaults ("fixed0", "M0").
    "seat_mode": "fixed0",
    "memory_rung": "M0",
    # Verbatim mirror of run_sweep.py `_merge`'s `setdefault("temperature", 0.0)`, so the
    # temperature this launch resolves to is known when the pilot's own is checked against it.
    "temperature": 0.0,
}


def _resolve_sweep_config(cfg: dict) -> dict:
    """Resolve `cfg` the way scripts/run_sweep.py's `_merge` would.

    The gate must judge the sweep that will actually run, not the literal
    YAML. Every consumer of a defaulted key in this file goes through here
    -- `main()`'s games list and `_validate_pilot_shape`'s row-count
    prediction -- so there is exactly one place a default can be wrong.
    """
    resolved = dict(cfg)
    for key, default in _RUN_SWEEP_DEFAULTS.items():
        resolved.setdefault(
            key, list(default) if isinstance(default, list) else default
        )
    return resolved


_WORD_RE = re.compile(r"\s+")


def count_words(text: str | None) -> int:
    """Whitespace-NORMALISED word count.

    Not `str.split(" ")` -- that undercounts newline-separated text
    (one verbose arm measured 0.4912 where the correct value is 1.0000).
    """
    return len(_WORD_RE.sub(" ", (text or "").strip()).split())


@dataclasses.dataclass(frozen=True)
class PilotThresholds:
    max_words: int = 5  # action completions longer than this are non-compliant
    max_verbose_rate: float = 0.05  # Primary abort criterion
    # Criteria 2 and 3 detect a BROKEN pipeline, not an unacceptable arm: their
    # thresholds were measured against a bad model (1.0 and 0.52 to 0.65).
    # The stamp refuses an arm above 1% of either rate, which these two
    # cannot predict; criteria 9 and 10 below carry that cap.
    max_belief_failure_rate: float = 0.90
    max_fallback_rate: float = 0.50  # prospective; zero measured separation
    # Criteria 9 and 10: the stamp's cap on belief_failed and fallback_used, as a share of
    # the arm's LM rows. Read from the stamp's own source so there is one number; None
    # switches both off and only the debug-shape acceptance check passes it.
    max_arm_failure_rate: float | None = dataclasses.field(
        default_factory=lambda: _failure_rate_cap_from_source()
    )
    min_rows_per_game: int = 100  # non-vacuity floor
    max_reference_rise: float = 0.05  # change-detector, calibration arm only
    # Belief-channel verbosity. NO default -- unlike every other field
    # above, this number is not this project's to invent (property
    # thresholds have no scientific basis yet).
    # None on either field means the criterion is not merely "always
    # passing" but genuinely UNCONFIGURED -- see GameStats.belief_verbose_configured
    # and _print_report, which must show this as its own visible state.
    max_belief_verbose_rate: float | None = None
    max_belief_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class GameStats:
    game: str
    n_rows: int
    # The count of LM rows in this cell whose
    # raw_completion produced at least one word -- verbose_rate's own
    # denominator. `verbose_rate` is a meaningless 0.0 placeholder when
    # this is 0 -- do not read verbose_rate without checking this field
    # first; the same discipline `belief_schema_present` already
    # documents for `belief_failure_rate` below, applied to the primary
    # criterion. `n_rows - n_rows_with_completion` is the empty count.
    n_rows_with_completion: int
    verbose_rate: float
    belief_failure_rate: float
    fallback_rate: float
    belief_calls_made: int
    calls_per_node: float
    median_words: int
    max_words: int
    # True iff the columns criterion 2 needs
    # (belief_raw_completion, belief_failed) are present in the data.
    # `belief_failure_rate` is a meaningless 0.0 placeholder when this is
    # False -- do not read belief_failure_rate without checking this flag
    # first; a real 0.0 (measured, no failures) and an unmeasurable 0.0
    # (criterion 2 never ran) must never be the same value to a caller.
    belief_schema_present: bool
    # Three-state discipline (a fourth beyond criterion 2's two,
    # because this criterion additionally has an optional threshold).
    # belief_verbose_configured=False means the other two fields below are
    # meaningless placeholders, exactly like belief_schema_present=False
    # already means for belief_failure_rate above.
    belief_verbose_rate: float
    belief_tokens_schema_present: bool
    belief_verbose_configured: bool
    # Share of belief calls whose belief was recovered by the bare-integer-key
    # salvage. Those rows count as parsed, so `belief_failure_rate` excludes
    # them; this reports them beside it. None (not 0.0) when the data predates
    # `belief_salvage_kind`, or when no belief call was made.
    belief_salvaged_rate: Optional[float] = None
    # LM rows whose action needed at least one retry. Report only, no cap: it moves
    # before the capped fallback rate does.
    n_retried: int = 0

    @property
    def retry_rate(self) -> Optional[float]:
        """
        the share of this cell's LM rows whose action needed a retry
        :returns: the share, or None for a cell with no LM rows (Optional[float])
        """
        return self.n_retried / self.n_rows if self.n_rows else None


@dataclasses.dataclass(frozen=True)
class Violation:
    criterion: str  # "verbose"|"verbose_unavailable"|
    # "belief"|"belief_unavailable"|
    # "fallback"|"vacuity"|
    # "prompt_schema"|"prompt_schema_mixed"|
    # "prompt_schema_unavailable"|
    # "completeness"|"completeness_unaccounted"|
    # "completeness_unavailable"|
    # "belief_cap"|"fallback_cap"|
    # "exact_policy_coverage"|
    # "exact_policy_unavailable"|
    # "reference_rise_verbose_rate"|
    # "reference_rise_belief_failure_rate"|
    # "reference_rise_fallback_rate"
    # (the last three: compare_to_reference, Tier 2;
    # "belief_unavailable": criterion 2
    # could not be evaluated at all -- distinct from a
    # genuine "belief" failure. The `*_unavailable` and
    # `*_unaccounted` names of criteria 5/6
    # follow that precedent: "could not be established"
    # is reported as its own thing, never folded into
    # the measured failure it is not.)
    game: str  # a game name, or DIRECTORY_SCOPE for a
    # criterion that is a property of the directory
    value: float
    threshold: float


#: What a fired criterion means, printed beneath it. Criteria 2 and 3 and criteria 9 and 10 read
#: the same two columns for different questions, so each says which question it answers.
VIOLATION_NOTES = {
    "belief": (
        "the belief channel looks BROKEN (a failure rate this high means the pipeline is "
        "not working); this says nothing about the 1% cap the stamp applies (see belief_cap)"
    ),
    "fallback": (
        "the action channel looks BROKEN (most rows fell back to the harness); this says "
        "nothing about the 1% cap the stamp applies (see fallback_cap)"
    ),
    "belief_cap": (
        "belief_failed is over the cap of the arm's LM rows that the stamp refuses "
        "an arm above, so an arm collected like this pilot would be refused at the stamp. "
        "A tripwire, not a predictor: a pilot under the cap does not imply the arm is, and "
        "the stamp may admit a measured count that this gate does not"
    ),
    "fallback_cap": (
        "fallback_used is over the cap of the arm's LM rows that the stamp refuses "
        "an arm above, so an arm collected like this pilot would be refused at the stamp. "
        "A tripwire, not a predictor: a pilot under the cap does not imply the arm is"
    ),
}


class PilotGateConfigError(Exception):
    """An operational failure resolving --config/--model, distinct from a
    fired Tier-1/Tier-2 criterion. main() maps this to exit 1, never
    GATE_FAILURE_EXIT_CODE."""


class ReferenceComparisonError(Exception):
    """--reference-dir cannot support a like-for-like comparison (no LM rows
    at all, or none within `hand_limit`). main() maps this to exit 1: an
    uninformative reference is a tool/setup failure, not evidence the arm
    under test is bad."""


def _failure_rate_cap_from_source() -> float:
    """Read `FAILURE_RATE_CAP` out of recycle_arm_evidence.py's source with `ast`.

    The stamp's cap is written down there and must not be re-typed here, so the
    two numbers cannot drift apart. Read rather than imported for the
    `_coverage_mass_floor_from_source` reason: that module imports `lmopt.agents.lm_vllm`,
    which this gate does not depend on.

    :returns: the cap, a share of the arm's LM rows (float)
    :raises PilotGateConfigError: when the file cannot be read or holds no literal assignment
    """
    import ast

    try:
        tree = ast.parse(_RECYCLE_EVIDENCE_SOURCE.read_text())
    except OSError as exc:
        raise PilotGateConfigError(
            f"cannot read {_RECYCLE_EVIDENCE_SOURCE} to establish FAILURE_RATE_CAP: {exc}"
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FAILURE_RATE_CAP" for t in node.targets
        ):
            try:
                return float(ast.literal_eval(node.value))
            except (ValueError, TypeError) as exc:
                raise PilotGateConfigError(
                    f"FAILURE_RATE_CAP in {_RECYCLE_EVIDENCE_SOURCE} is not a literal: {exc}"
                ) from exc
    raise PilotGateConfigError(
        f"no FAILURE_RATE_CAP assignment found in {_RECYCLE_EVIDENCE_SOURCE}"
    )


def _check_arm_failure_caps(lm: pl.DataFrame, cap: float) -> list[Violation]:
    """
    criteria 9 and 10: the stamp's cap on belief_failed and fallback_used, over the whole
    pilot rather than per game
    :param lm: the pilot's LM rows inside the scored window (pl.DataFrame)
    :param cap: the share of LM rows above which the stamp refuses an arm, strictly (float)
    :returns: one violation per rate over the cap, directory scope (list of Violation)
    """
    out: list[Violation] = []
    n_lm = lm.height
    if not n_lm:
        return out
    for criterion, column in (
        ("belief_cap", "belief_failed"),
        ("fallback_cap", "fallback_used"),
    ):
        if column not in lm.columns:
            continue
        count = int(lm[column].fill_null(False).cast(pl.Boolean).sum())
        if count / n_lm > cap:
            out.append(Violation(criterion, DIRECTORY_SCOPE, count / n_lm, cap))
    return out


def _prompt_schema_version_from_source() -> int:
    """Read `PROMPT_SCHEMA_VERSION` out of lm_vllm.py's source with `ast`.

    The fallback for `_live_prompt_schema_version` below: lm_vllm.py
    imports pyspiel at module level, and
    this gate's docstring says it does not
    depend on OpenSpiel. Reading the assignment is not re-typing the
    integer: there is still exactly one place the value is written down.
    """
    import ast

    try:
        tree = ast.parse(_LM_VLLM_SOURCE.read_text())
    except OSError as exc:
        raise PilotGateConfigError(
            f"cannot read {_LM_VLLM_SOURCE} to establish the live "
            f"prompt_schema_version: {exc}"
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PROMPT_SCHEMA_VERSION"
            for t in node.targets
        ):
            try:
                return int(ast.literal_eval(node.value))
            except (ValueError, TypeError) as exc:
                raise PilotGateConfigError(
                    f"PROMPT_SCHEMA_VERSION in {_LM_VLLM_SOURCE} is not a "
                    f"literal integer: {exc}"
                ) from exc
    raise PilotGateConfigError(
        f"no PROMPT_SCHEMA_VERSION assignment found in {_LM_VLLM_SOURCE} -- "
        f"cannot establish which prompt schema is live, so no pilot can be "
        f"certified as describing it."
    )


def _coverage_mass_floor_from_source() -> float:
    """Read `COVERAGE_MASS_FLOOR` out of exact_policy_pass.py's source with `ast`.

    Criterion 7's threshold is pre-registered there (the constant's own
    comment says it must not move), so re-typing it here would be the
    "two representations, one of them never checked" defect this file
    already carries two guards against. Read rather than imported for the
    `_prompt_schema_version_from_source` reason: exact_policy_pass.py
    imports `lmopt.games.loaders` at module level, which imports pyspiel,
    and this gate's docstring says it does not
    depend on OpenSpiel.
    """
    import ast

    try:
        tree = ast.parse(_EXACT_POLICY_SOURCE.read_text())
    except OSError as exc:
        raise PilotGateConfigError(
            f"cannot read {_EXACT_POLICY_SOURCE} to establish "
            f"COVERAGE_MASS_FLOOR: {exc}"
        ) from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "COVERAGE_MASS_FLOOR"
            for t in node.targets
        ):
            try:
                return float(ast.literal_eval(node.value))
            except (ValueError, TypeError) as exc:
                raise PilotGateConfigError(
                    f"COVERAGE_MASS_FLOOR in {_EXACT_POLICY_SOURCE} is not a "
                    f"literal number: {exc}"
                ) from exc
    raise PilotGateConfigError(
        f"no COVERAGE_MASS_FLOOR assignment found in {_EXACT_POLICY_SOURCE} -- "
        f"cannot establish the coverage floor a pilot's exact-policy artefact "
        f"has to clear, so no pilot can be certified against it."
    )


def _exact_policy_enabled(cfg: dict) -> bool:
    """Does this config ask each arm to run scripts/exact_policy_pass.py?

    Mirrors run_experiment.py's own `exact_policy_cfg`/`exact_policy_enabled`
    resolution in `main` verbatim, including its deliberate
    default asymmetry: an absent `exact_policy` block, or an absent
    `enabled` sub-key, is FALSE. Only the collection cycle config
    (configs/run_experiment.yaml) opts in, and criterion 7 must ask the
    same question of the same config the orchestrator will, or it would
    refuse pilots for a pass that is never going to run (or, worse, wave
    through the one cycle where the pass is the point).
    """
    return bool((cfg.get("exact_policy") or {}).get("enabled", False))


def _live_prompt_schema_version() -> int:
    """The version of the prompting code that is about to RUN.

    Imported from `lmopt.agents.lm_vllm`, never re-typed here: a hand-copied
    integer is exactly the "two representations, one of them never checked"
    defect class this repo keeps recording, and it would go stale at the
    next bump with nothing to notice. Equally, this must never be compared
    against the value the pilot recorded -- that is an algebraic identity
    and can only ever pass.

    Falls back to reading the same assignment out of lm_vllm.py's source
    when the import fails (macOS has no pyspiel wheel, and lm_vllm.py
    imports pyspiel at module level), and says so on stderr rather than
    falling back silently (no silent fallbacks). If neither
    works, `_prompt_schema_version_from_source` raises
    PilotGateConfigError: a gate that cannot tell which prompt is live must
    fail, not wave the arm through.
    """
    try:
        from lmopt.agents.lm_vllm import PROMPT_SCHEMA_VERSION

        return int(PROMPT_SCHEMA_VERSION)
    except Exception as exc:  # ImportError here in practice; never swallowed
        print(
            f"WARNING: could not import lmopt.agents.lm_vllm to read the "
            f"live PROMPT_SCHEMA_VERSION ({type(exc).__name__}: {exc}); "
            f"reading it from {_LM_VLLM_SOURCE} instead.",
            file=sys.stderr,
        )
        return _prompt_schema_version_from_source()


@dataclasses.dataclass(frozen=True)
class PilotReport:
    model: str
    per_game: dict[str, GameStats]
    violations: tuple[Violation, ...]
    total_calls: int
    wall_clock_s: float | None
    # The window the verdict was actually computed over, and
    # how much of the directory fell outside it. `--n-hands` bounds
    # what --skip-sweep scores, so a report that does not
    # state its own window lets an operator read a `vacuity` abort on a
    # healthy 1000-hand arm as "the arm is unusable" while 98% of the data
    # sits unscored and unmentioned. None = no limit applied.
    hand_limit: int | None = None
    n_rows_outside_hand_limit: int = 0
    # What criteria 5 and 6 actually saw, so a PASS states the
    # evidence it passed on rather than leaving it to be inferred from the
    # absence of a violation: the distinct non-null prompt_schema_version
    # values in the scored rows, the live version they were judged against,
    # and (requested, completed, failed) from shard_outcomes.json (None =
    # no readable record).
    prompt_schema_observed: tuple[int, ...] = ()
    n_rows_missing_prompt_schema: int = 0
    live_prompt_schema_version: int | None = None
    shard_outcomes: tuple[int, int, int] | None = None
    # Criterion 7, same contract: what it saw, and both halves of the
    # comparison. None = no readable exact_policy.csv, which is a refusal
    # only when `exact_policy_enabled` (carried here so the report can tell
    # "not evaluated" from "could not be evaluated").
    exact_policy_coverage_median: float | None = None
    coverage_mass_floor: float | None = None
    exact_policy_enabled: bool = False
    # Per game, (info-sets with a trusted distribution, info-sets), reported beside
    # the belief-failure rate. Report only: neither rate is a criterion. None = no
    # readable evidence, which is distinct from a measured 0.
    exact_policy_parse_counts: dict[str, tuple[int, int]] | None = None
    # Failed exact-policy info-sets per error kind (None = no readable evidence), and the
    # memory rung the pilot directory records, which decides whether the parse rate is
    # absent by construction. Both report only.
    exact_policy_error_kinds: dict[str, int] | None = None
    memory_rung: str | None = None
    recorded_exact_policy_enabled: bool | None = None

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0


def _read_lm_rows(data_dir: Path) -> pl.DataFrame:
    """Concat every Parquet shard in `data_dir` and keep only LM rows.

    `pl.col("prompt").is_not_null()` matches scripts/diagnose.py:86 exactly
    -- non-LM agents (CFR ref, Bayesian, RL, always_bet) log a row too, but
    never render a prompt. `diagonal_relaxed` mirrors diagnose.py's own
    schema-migration policy (silently null-fills a column an older shard
    lacks) -- this gate reads a single pilot run's own fresh shards, so that
    policy is inherited for consistency, not because it is exercised here.
    """
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema={"game": pl.Utf8})
    from lmopt.analysis.shards import load_shards

    df = load_shards(files)
    return df.filter(pl.col("prompt").is_not_null())


def _configured(value: float | None, name: str) -> float:
    """
    a threshold the stats say was configured
    :param value: the threshold field, `None` when the caller left it unset (float or None)
    :param name: the field's name, for the error message (str)
    :returns: the threshold (float)
    :raises ValueError: when stats computed with the criterion configured are judged against
        thresholds that leave the field unset
    """
    if value is None:
        raise ValueError(
            f"the stats were computed with {name} configured, but it is unset"
        )
    return value


def _rate(col: pl.Series, n: int) -> float:
    """Aggregate a required (schema.py) boolean column as a rate over `n` rows.

    Pairs the aggregate with `null_count()` rather than calling `.sum()` (or
    `.all()`) blind: polars silently treats a null as absent from a boolean
    reduction, which would misreport an anomalous null in a column the
    schema declares never-null. An unexpected null is treated conservatively
    as a positive (failure/fallback) rather than dropped from the rate.
    """
    if n == 0:
        return 0.0
    if col.null_count() == 0:
        return float(col.sum()) / n
    return float(col.fill_null(True).sum()) / n


# Criterion 2 (belief_failure_rate) needs
# Exactly these two columns -- belief_raw_completion to know which rows had
# a call made (the load-bearing restriction), and
# belief_failed as the failure indicator itself. belief_n_retries is not
# one of them: it feeds calls_per_node only (a separate, non-gating
# statistic, handled by its own presence check in _game_stats below), so
# requiring it here would wrongly disarm criterion 2 on a frame that has
# both columns criterion 2 actually needs but happens to lack
# belief_n_retries.
_BELIEF_FAILURE_COLUMNS = ("belief_raw_completion", "belief_failed")


def _has_belief_failure_schema(df: pl.DataFrame) -> bool:
    """True iff `df` carries the columns criterion 2 needs to be evaluated.

    A real, supported older-schema absence, not corrupt data:
    `belief_raw_completion` is deliberately excluded from schema.py's
    REQUIRED_COLUMNS, and the landed `data/run_experiment/qwen2.5-14b-instruct/`
    shards predate both these columns (they carry `elicited_belief` but not
    these).
    """
    return all(c in df.columns for c in _BELIEF_FAILURE_COLUMNS)


def _game_stats(lm: pl.DataFrame, game: str, thresholds: PilotThresholds) -> GameStats:
    subset = lm.filter(pl.col("game") == game)
    n_rows = subset.height
    if n_rows == 0:
        return GameStats(
            game=game,
            n_rows=0,
            n_rows_with_completion=0,
            verbose_rate=0.0,
            belief_failure_rate=0.0,
            fallback_rate=0.0,
            belief_calls_made=0,
            calls_per_node=0.0,
            median_words=0,
            max_words=0,
            belief_schema_present=_has_belief_failure_schema(subset),
            belief_verbose_rate=0.0,
            belief_tokens_schema_present="belief_completion_tokens" in subset.columns,
            belief_verbose_configured=(
                thresholds.max_belief_verbose_rate is not None
                and thresholds.max_belief_tokens is not None
            ),
        )

    # Primary criterion's own column discipline: exclude empty rows from the
    # denominator rather than scoring them as non-compliant; an all-empty
    # cell gets a distinct not-evaluable state (see evaluate_pilot's firing
    # logic below). Mirrors the belief
    # criterion's already-load-bearing discipline twenty lines below: a row
    # with no completion at all (raw_completion null or the empty string)
    # is neither compliant nor non-compliant on word count -- no completion
    # was produced to measure -- so counting it in the denominator
    # manufactures a rate the same way the belief criterion's own comment
    # warns against. `count_words` resolves both `None` and `""` to 0 via
    # `(text or "").strip()`, so `wc == 0` is the exact predicate for "no
    # completion produced" here -- no separate raw_completion re-check
    # needed. On the qwen2.5-14b-instruct-era
    # data/run_experiment/step5-debug/ arms, 16 of 713 LM rows (2.2440%)
    # carry an empty/null raw_completion, and one real cell
    # (gptoss-20b-r16384/leduc) flips from PASS (0.0500) to ABORT (0.0526)
    # under this rule.
    #
    # word_counts itself is left computed over every row, unchanged --
    # median_words/max_words below are still whole-cell statistics
    # (an empty completion legitimately contributes its true wc=0 to
    # those), and empty rows already cannot inflate verbose_rate's
    # numerator (wc=0 never exceeds max_words), so only the denominator
    # needed narrowing. fallback_rate and calls_per_node (below) are
    # Deliberately left on the full n_rows: an empty completion is a real,
    # loudly-logged event (every one of the 16 measured rows above carries
    # n_retries > 0, 7 also carry fallback_used=True) that must stay
    # visible to those other statistics, not be silently dropped from the
    # cell's own row count everywhere.
    #
    # `verbose_rate` stays a literal `0.0`
    # placeholder when `n_rows_with_completion == 0` -- matching
    # `belief_failure_rate`'s own established "meaningless 0.0 placeholder,
    # check the flag first" idiom exactly (see `belief_schema_present`
    # below), not a sentinel value. `n_rows_with_completion` is a real
    # `GameStats` field (the flag callers must check), not just a local
    # used to compute the rate -- `evaluate_pilot` reads it to fire a
    # distinct `"verbose_unavailable"` criterion before ever consulting
    # this placeholder `verbose_rate`, mirroring `belief_unavailable`'s own
    # placement exactly.
    word_counts = [count_words(c) for c in subset["raw_completion"].to_list()]
    n_rows_with_completion = sum(1 for wc in word_counts if wc > 0)
    verbose_rate = (
        sum(1 for wc in word_counts if wc > thresholds.max_words)
        / n_rows_with_completion
        if n_rows_with_completion > 0
        else 0.0
    )

    fallback_rate = _rate(subset["fallback_used"], n_rows)

    # Belief criterion's column discipline (load-bearing): restrict to rows where belief_raw_completion is
    # non-null -- proof a belief call was actually made -- and only then
    # read belief_failed. elicited_belief.is_null() is true both when a
    # call failed and when none was made, so using it (or computing the
    # rate over all rows rather than this restricted subset) manufactures
    # a false abort at zero calls issued.
    #
    # An older-schema directory (the
    # landed data/run_experiment/qwen2.5-14b-instruct/ arm) lacks the
    # belief_raw_completion/belief_failed columns outright -- a supported
    # absence (schema.py REQUIRED_COLUMNS deliberately excludes
    # belief_raw_completion), not corrupt data -- so this must degrade to
    # "belief unavailable" rather than ColumnNotFoundError. Critically,
    # "unavailable" is carried as its own `belief_schema_present`
    # flag, not as a fabricated passing belief_failure_rate=0.0 --
    # conflating "measured zero failures" with "never measured" would let
    # a pilot with belief elicitation totally broken print a clean PASS.
    belief_schema_present = _has_belief_failure_schema(subset)
    belief_salvaged_rate: float | None = None
    if belief_schema_present:
        belief_subset = subset.filter(pl.col("belief_raw_completion").is_not_null())
        belief_calls_made = belief_subset.height
        belief_failure_rate = _rate(belief_subset["belief_failed"], belief_calls_made)
        if "belief_salvage_kind" in belief_subset.columns and belief_calls_made > 0:
            belief_salvaged_rate = (
                float(belief_subset["belief_salvage_kind"].is_not_null().sum())
                / belief_calls_made
            )
    else:
        belief_calls_made = 0
        belief_failure_rate = (
            0.0  # placeholder; belief_schema_present=False marks it unmeasured
        )

    # Belief-channel verbosity. belief_verbose_configured is a
    # property of the CALLER's thresholds, not of this directory's data --
    # computed first so the two data-dependent flags below can short-circuit
    # cleanly when the criterion was never asked to run.
    belief_verbose_configured = (
        thresholds.max_belief_verbose_rate is not None
        and thresholds.max_belief_tokens is not None
    )
    belief_tokens_schema_present = "belief_completion_tokens" in subset.columns
    if (
        belief_verbose_configured
        and belief_tokens_schema_present
        and belief_schema_present
    ):
        bt_subset = subset.filter(pl.col("belief_raw_completion").is_not_null())
        bt = bt_subset["belief_completion_tokens"]
        n_bt = bt_subset.height
        if n_bt:
            belief_verbose_rate = (
                sum(
                    1
                    for v in bt.to_list()
                    if v is not None and v > thresholds.max_belief_tokens
                )
                / n_bt
            )
        else:
            belief_verbose_rate = 0.0
    else:
        belief_verbose_rate = 0.0

    # belief_n_retries feeds only calls_per_node, a separate non-gating
    # statistic -- its presence is checked independently of
    # belief_schema_present above, so a
    # frame missing just this column does not disarm criterion 2.
    if "belief_n_retries" in subset.columns:
        belief_n_retries_total = float(subset["belief_n_retries"].fill_null(0).sum())
    else:
        belief_n_retries_total = 0.0

    # calls_per_node = (n_retries + 1) + (belief_n_retries + 1), summed
    # over rows / n_rows. The "+1" on
    # each term counts the first attempt; n_retries/belief_n_retries count
    # only the extra attempts (schema.py), and belief_n_retries is 0 (not
    # null) on a skipped/never-asked belief, so it always contributes.
    n_retries_total = float(subset["n_retries"].fill_null(0).sum())
    calls_per_node = (
        (n_retries_total + n_rows) + (belief_n_retries_total + n_rows)
    ) / n_rows

    return GameStats(
        game=game,
        n_rows=n_rows,
        n_rows_with_completion=n_rows_with_completion,
        verbose_rate=verbose_rate,
        belief_failure_rate=belief_failure_rate,
        fallback_rate=fallback_rate,
        belief_calls_made=belief_calls_made,
        calls_per_node=calls_per_node,
        median_words=int(statistics.median(word_counts)),
        max_words=max(word_counts),
        belief_schema_present=belief_schema_present,
        belief_verbose_rate=belief_verbose_rate,
        belief_tokens_schema_present=belief_tokens_schema_present,
        belief_verbose_configured=belief_verbose_configured,
        belief_salvaged_rate=belief_salvaged_rate,
        n_retried=int((subset["n_retries"].fill_null(0) > 0).sum()),
    )


def _prompt_schema_values(lm: pl.DataFrame) -> tuple[tuple[int, ...], int]:
    """The distinct non-null `prompt_schema_version` values in `lm`, and how
    many rows carry no version at all (column absent, or null on the row).

    A missing column is reported as "every row missing", not as an empty
    frame of versions: the two are the same evidence (none) and criterion 5
    must not have a branch where one of them can pass.
    """
    if PROMPT_SCHEMA_COLUMN not in lm.columns:
        return (), lm.height
    values = lm[PROMPT_SCHEMA_COLUMN].to_list()
    distinct = tuple(sorted({int(v) for v in values if v is not None}))
    return distinct, sum(1 for v in values if v is None)


def _check_prompt_schema(
    observed: Sequence[int], n_missing: int, live_version: int
) -> list[Violation]:
    """Criterion 5: does this pilot describe the prompt code about to run?

    Three refusals, and only one of them is "stale". A pilot with no
    version (data predating the column, or a null column) and a pilot with several
    versions are both refusals for the same reason the belief criterion
    grew `belief_unavailable`: a criterion that cannot be evaluated has not
    been passed, and the alternative -- exiting 0 having established
    nothing -- is the exact shape this gate exists to prevent.

    `live_version` is the caller's, and main() never supplies one: see
    `_live_prompt_schema_version`. Comparing `observed` against a version
    read out of the same rows would be an algebraic identity, and could
    only ever pass.
    """
    if not observed:
        return [Violation("prompt_schema_unavailable", DIRECTORY_SCOPE, 1.0, 0.0)]
    if len(observed) > 1 or n_missing:
        # "How many distinct provenances does this directory contain",
        # counting "no version at all" as one of them. 1 is the only
        # answer from which a verdict about the prompt can be reached.
        return [
            Violation(
                "prompt_schema_mixed",
                DIRECTORY_SCOPE,
                float(len(observed) + (1 if n_missing else 0)),
                1.0,
            )
        ]
    if observed[0] != live_version:
        return [
            Violation(
                "prompt_schema",
                DIRECTORY_SCOPE,
                float(observed[0]),
                float(live_version),
            )
        ]
    return []


def _read_shard_outcomes(data_dir: Path) -> dict | None:
    """run_sweep.py's completeness record, or None if there is nothing
    usable there: absent, unreadable, not an object, or without the two
    fields criterion 6 reads (`shards` list, integer `n_requested`).

    Every one of those is the same finding -- "this directory carries no
    evidence that its sweep finished" -- so they collapse to one None
    rather than to a taxonomy no operator would act on differently.
    """
    path = Path(data_dir) / SHARD_OUTCOMES_FILENAME
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if not isinstance(record.get("shards"), list):
        return None
    if not isinstance(record.get("n_requested"), int) or isinstance(
        record.get("n_requested"), bool
    ):
        return None
    return record


def _check_shard_completeness(record: dict | None) -> list[Violation]:
    """Criterion 6: did every shard this sweep requested actually finish?

    A pilot that ran 11 of 20 goofspiel shards can still log 660 LM rows
    for the game, 6.6x the non-vacuity floor -- criterion 4 sees a healthy
    game. The two criteria
    ask different questions: 4 asks "is there data", 6 asks "is there all
    the data", and only the sweep's own record can answer the second.

    Refusals: any shard whose status is not "completed" (grouped per game,
    so the report names which game to re-run), and any shortfall between
    `n_requested` and the number of entries -- a requested shard with no
    entry is not evidence of success. A `None` record (see
    `_read_shard_outcomes`) is `completeness_unavailable`: "cannot tell" is
    not "passes", and it is the state every pilot directory collected
    before the record existed is in.
    """
    if record is None:
        return [Violation("completeness_unavailable", DIRECTORY_SCOPE, 1.0, 0.0)]
    shards = record["shards"]
    unfinished: dict[str, int] = {}
    for entry in shards:
        # Not `status == "failed"`: run_sweep.py writes only "completed" and
        # "failed" today, but a killed sweep's half-written record, or any
        # future third status, must not be read as success by default.
        if not isinstance(entry, dict) or entry.get("status") != "completed":
            game = (
                str(entry.get("game", DIRECTORY_SCOPE))
                if isinstance(entry, dict)
                else DIRECTORY_SCOPE
            )
            unfinished[game] = unfinished.get(game, 0) + 1
    violations = [
        Violation("completeness", game, float(unfinished[game]), 0.0)
        for game in sorted(unfinished)
    ]
    unaccounted = record["n_requested"] - len(shards)
    if unaccounted != 0:
        violations.append(
            Violation(
                "completeness_unaccounted", DIRECTORY_SCOPE, float(unaccounted), 0.0
            )
        )
    return violations


def _read_exact_policy_coverage(data_dir: Path) -> float | None:
    """The median `coverage_mass` over the readable rows of
    `<data_dir>/exact_policy.csv`, or None if there is nothing usable
    there: absent, unreadable, no `coverage_mass` column, or that column
    null, non-finite or non-numeric on every row.

    Every one of those is the same finding -- "this directory carries no
    evidence about exact-policy coverage" -- so they collapse to one None,
    exactly as `_read_shard_outcomes` collapses its four.

    The median, not the min or the mean, because that is the statistic
    that separates the arms: Qwen 1.000 (min 0.998), Mistral-NeMo 0.0671
    (max 0.4026) under a lossy prefix round trip. A min would refuse an
    otherwise-clean arm on one hard info-set; a mean is dragged by the
    same tail.

    Read with polars, never by splitting lines: a `coverage_mass` figure
    that reaches an operator comes from a parser (on exact_policy.csv,
    `wc -l` counts phantom rows from embedded newlines).
    """
    path = Path(data_dir) / EXACT_POLICY_FILENAME
    try:
        df = pl.read_csv(path, infer_schema_length=None)
    except Exception:
        # Absent, truncated, or not a CSV at all -- one finding (see above).
        return None
    if COVERAGE_MASS_COLUMN not in df.columns:
        return None
    values: list[float] = []
    for v in df[COVERAGE_MASS_COLUMN].to_list():
        if v is None:
            continue
        try:
            value = float(v)
        except (TypeError, ValueError):
            # A non-numeric coverage column is not a low reading, it is no
            # reading: refuse to median a mixture of numbers and prose. The
            # whole column is discarded here, unlike the per-row exclusion
            # below, because prose in a numeric column means the FILE's shape
            # is wrong, not that one info-set went unscored.
            return None
        if not math.isfinite(value):
            # NaN and +/-inf are excluded ROW BY ROW, exactly as a null is,
            # and never compared. A NaN survives `float()`, and both
            # `statistics.median` and every comparison against it then
            # propagate it silently: `median([nan, ...])` is nan and
            # `nan < floor` is False, so a coverage column that is NaN on
            # its middle row would PASS this criterion -- the silent-pass shape
            # this gate exists to prevent. An inf is excluded on the same
            # rule rather than compared: coverage_mass is a sum of
            # exponentiated log-probabilities, so a non-finite entry is a
            # row that was not scored, whichever direction it points.
            continue
        values.append(value)
    if not values:
        # Every row null or non-finite: an artefact with no reading at all,
        # which is the same finding as an absent one (and, with the pass
        # enabled, the same refusal).
        return None
    return float(statistics.median(values))


def _read_exact_policy_info_sets(data_dir: Path) -> pl.DataFrame | None:
    """
    one row per exact-policy info-set: its game, whether it parsed, and the kinds of its errors
    An info-set is parsed only when none of its rows carries an error, so a partially scored
    action set is a failure and one failed info-set is one failure however many actions it
    has. The key includes `game_params`, because the same info-set string under two
    parameterisations is two info-sets. Every column is read as text so an info-set string
    such as `01` is never merged with `1`. Nothing here can raise out of the gate: a directory
    with no evidence is None, never a clean result, because a real 1.0 and an unmeasurable
    1.0 must never be the same value.
    :param data_dir: the pilot directory holding `exact_policy.csv` (Path)
    :returns: a frame with columns game, parsed (bool) and kinds (list of str, the distinct
        kinds of the info-set's error rows, `unclassified` where a row has an error and no
        kind), or None when the file is absent, unreadable, empty, or lacks a key column or
        the error column (pl.DataFrame | None)
    """
    try:
        df = pl.read_csv(Path(data_dir) / EXACT_POLICY_FILENAME, infer_schema_length=0)
    except Exception:
        return None
    needed = (*EXACT_POLICY_PARSE_KEY, EXACT_POLICY_ERROR_COLUMN)
    if df.height == 0 or any(c not in df.columns for c in needed):
        return None
    if EXACT_POLICY_ERROR_KIND_COLUMN in df.columns:
        kind = pl.col(EXACT_POLICY_ERROR_KIND_COLUMN).fill_null(UNCLASSIFIED_ERROR_KIND)
    else:
        kind = pl.lit(UNCLASSIFIED_ERROR_KIND)
    return df.group_by(list(EXACT_POLICY_PARSE_KEY)).agg(
        pl.col(EXACT_POLICY_ERROR_COLUMN).is_null().all().alias("parsed"),
        kind.filter(pl.col(EXACT_POLICY_ERROR_COLUMN).is_not_null())
        .unique()
        .sort()
        .alias("kinds"),
    )


def _read_exact_policy_parse_counts(
    data_dir: Path,
) -> dict[str, tuple[int, int]] | None:
    """
    per game, the info-sets of the exact-policy artefact whose distribution is trusted
    :param data_dir: the pilot directory holding `exact_policy.csv` (Path)
    :returns: {game: (parsed info-sets, info-sets)}, or None when there is no readable
        evidence (dict | None)
    """
    info_sets = _read_exact_policy_info_sets(data_dir)
    if info_sets is None:
        return None
    counts: dict[str, tuple[int, int]] = {}
    for game in sorted(info_sets["game"].unique().to_list()):
        sub = info_sets.filter(pl.col("game") == game)
        counts[game] = (int(sub["parsed"].sum()), sub.height)
    return counts


def _read_exact_policy_error_kinds(data_dir: Path) -> dict[str, int] | None:
    """
    the number of failed exact-policy info-sets per error kind
    An info-set whose rows carry several kinds counts once under each, so the counts can sum
    to more than the number of failed info-sets. A failed info-set with no recorded kind
    counts under `unclassified` rather than disappearing, because a breakdown that omits the
    kinds it does not recognise cannot explain a rate below 1.0.
    :param data_dir: the pilot directory holding `exact_policy.csv` (Path)
    :returns: {kind: failed info-sets}, empty when nothing failed, or None when there is no
        readable evidence (dict | None)
    """
    info_sets = _read_exact_policy_info_sets(data_dir)
    if info_sets is None:
        return None
    kinds: dict[str, int] = {}
    for row_kinds in info_sets.filter(~pl.col("parsed"))["kinds"].to_list():
        for kind in row_kinds:
            kinds[kind] = kinds.get(kind, 0) + 1
    return dict(sorted(kinds.items()))


def _recorded_resolved_config(data_dir: Path) -> dict | None:
    """
    the `resolved_config` of the pilot directory's own `run_meta.json`
    An absent file is the expected state of an old or fresh directory and reads None quietly.
    A file that is present but unreadable or not shaped as expected also reads None, but says
    so on stderr, because a silent None would hide a corrupt record behind an ordinary
    "not recorded".
    :param data_dir: the pilot directory (Path)
    :returns: the recorded resolved config, or None (dict | None)
    """
    path = Path(data_dir) / "run_meta.json"
    try:
        resolved = json.loads(path.read_text())["resolved_config"]
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            f"WARNING: {path} is present but its resolved_config could not be read "
            f"({type(exc).__name__}: {exc}); treating its memory rung and exact-policy "
            f"setting as not recorded.",
            file=sys.stderr,
        )
        return None
    return resolved if isinstance(resolved, dict) else None


def _recorded_memory_rung(data_dir: Path) -> str | None:
    """
    the memory rung the pilot directory's own `run_meta.json` records
    Read from the recorded artefact, never derived from the directory's name: a rung inferred
    from a path is a conversion that cannot fail, and a wrongly named directory would then
    print a structural reason for an arm where the exact pass should have run.
    :param data_dir: the pilot directory (Path)
    :returns: the recorded `memory_rung`, or None when it is absent or not a string
        (str | None)
    """
    resolved = _recorded_resolved_config(data_dir)
    rung = resolved.get("memory_rung") if resolved is not None else None
    return rung if isinstance(rung, str) else None


def _recorded_exact_policy_enabled(data_dir: Path) -> bool | None:
    """
    whether the pilot directory's recorded config switched the exact-policy pass on
    :param data_dir: the pilot directory (Path)
    :returns: the recorded `exact_policy.enabled`, or None when it is not recorded as a
        boolean (bool | None)
    """
    resolved = _recorded_resolved_config(data_dir)
    block = resolved.get("exact_policy") if resolved is not None else None
    enabled = block.get("enabled") if isinstance(block, dict) else None
    return enabled if isinstance(enabled, bool) else None


def _check_exact_policy_coverage(
    median_coverage: float | None, enabled: bool, floor: float
) -> list[Violation]:
    """Criterion 7: can the exact-policy pass say anything about this model?

    Checking at the pilot moves detection of an unusable exact-policy
    artefact from after hours of GPU per arm to before the arm launches.

    Two refusals. A median below `floor` is the measured one. An artefact
    that is absent or unreadable while the config enables the pass is
    `exact_policy_unavailable`, on criterion 6's reasoning: a pilot that
    was never scored for coverage certifies nothing about coverage, and a
    cycle whose config turns the pass on will spend the GPU time either
    way.

    With the pass not enabled and no artefact present, the criterion is
    dormant -- there is no question to answer, so there is no verdict to
    fail. A present artefact is judged either way: its coverage is
    evidence about this model regardless of which config invoked the
    gate.
    """
    if median_coverage is None:
        if enabled:
            return [Violation("exact_policy_unavailable", DIRECTORY_SCOPE, 1.0, 0.0)]
        return []
    if median_coverage < floor:
        return [
            Violation("exact_policy_coverage", DIRECTORY_SCOPE, median_coverage, floor)
        ]
    return []


def evaluate_pilot(
    data_dir: Path,
    games: Sequence[str],
    thresholds: PilotThresholds,
    hand_limit: int | None = None,
    live_prompt_schema_version: int | None = None,
    exact_policy_enabled: bool = False,
    coverage_mass_floor: float | None = None,
) -> PilotReport:
    """Score `data_dir` against the eight Tier-1 criteria.

    `exact_policy_enabled` is criterion 7's premise, not its threshold:
    whether the config that will run this arm asks for an exact-policy
    pass, so an absent artefact can be told apart from one that was never
    going to exist (main() resolves it with `_exact_policy_enabled`; the
    default is run_experiment.py's own conservative False).
    `coverage_mass_floor` defaults to the pre-registered figure read out of
    exact_policy_pass.py, and is a parameter only so a unit test can fix
    both sides of that comparison.

    `live_prompt_schema_version` is the version criterion 5 judges
    the pilot's rows against; `None` (main()'s case, and the default)
    resolves it from `lmopt.agents.lm_vllm` through
    `_live_prompt_schema_version`. It is a parameter only so a unit test can
    fix both sides of the comparison; production never passes one, and a
    caller passing the PILOT's own version would be asserting an identity.

    `hand_limit` restricts the pilot to `hand_idx <
    hand_limit`, the same like-for-like window `compare_to_reference`
    applies to the reference. Applied to one side only, under --skip-sweep
    against a directory holding more hands than --n-hands the Tier-2
    comparison would be exactly the asymmetric one the like-for-like filter
    exists to remove (it reintroduces a measured +0.018 sampling bias).
    On a fresh sweep this is a
    no-op -- run_sweep.py runs exactly --n-hands hands -- so it changes
    only the --skip-sweep re-evaluation path, where it makes the Tier-1
    verdict reproducible: the gate judges the first `hand_limit` hands,
    the same hands it compares.

    `None` (default) evaluates every row, for direct unit-level callers.
    """
    lm = _read_lm_rows(Path(data_dir))
    n_rows_before_limit = lm.height
    if hand_limit is not None and "hand_idx" in lm.columns:
        lm = lm.filter(pl.col("hand_idx") < hand_limit)
    # Counted over EVERY LM row in the directory, not just the requested
    # `games` -- this figure answers "how much of this directory am I not
    # looking at", which is a property of the directory, not of the games
    # the caller happened to ask about.
    n_rows_outside_hand_limit = n_rows_before_limit - lm.height
    per_game: dict[str, GameStats] = {}
    violations: list[Violation] = []
    total_calls = 0

    # Criteria 5 and 6, and criterion 7 below them. All three are
    # properties of the directory rather than of any one game; 5 and 6 are
    # computed over the scored window (the rows the verdict below is
    # actually about), while 7 reads a separate artefact that carries no
    # hand index and so has no window to be restricted to.
    if live_prompt_schema_version is None:
        live_prompt_schema_version = _live_prompt_schema_version()
    observed_versions, n_rows_missing_prompt_schema = _prompt_schema_values(lm)
    violations += _check_prompt_schema(
        observed_versions, n_rows_missing_prompt_schema, live_prompt_schema_version
    )
    outcomes_record = _read_shard_outcomes(Path(data_dir))
    violations += _check_shard_completeness(outcomes_record)
    shard_outcomes = (
        None
        if outcomes_record is None
        else (
            outcomes_record["n_requested"],
            sum(
                1
                for s in outcomes_record["shards"]
                if isinstance(s, dict) and s.get("status") == "completed"
            ),
            sum(
                1
                for s in outcomes_record["shards"]
                if isinstance(s, dict) and s.get("status") == "failed"
            ),
        )
    )
    if coverage_mass_floor is None:
        coverage_mass_floor = _coverage_mass_floor_from_source()
    exact_policy_coverage_median = _read_exact_policy_coverage(Path(data_dir))
    exact_policy_parse_counts = _read_exact_policy_parse_counts(Path(data_dir))
    exact_policy_error_kinds = _read_exact_policy_error_kinds(Path(data_dir))
    memory_rung = _recorded_memory_rung(Path(data_dir))
    recorded_exact_policy_enabled = _recorded_exact_policy_enabled(Path(data_dir))
    violations += _check_exact_policy_coverage(
        exact_policy_coverage_median, exact_policy_enabled, coverage_mass_floor
    )

    if thresholds.max_arm_failure_rate is not None:
        violations += _check_arm_failure_caps(lm, thresholds.max_arm_failure_rate)

    for game in games:
        stats = _game_stats(lm, game, thresholds)
        per_game[game] = stats
        total_calls += round(stats.calls_per_node * stats.n_rows)

        # A cell whose LM rows all produced no
        # completion at all is unmeasurable on word count, not cleanly
        # passing -- the same "exits 0 having established nothing" shape
        # belief_unavailable (below) exists to prevent.
        # Checked before the ordinary verbose_rate comparison (elif, not a
        # separate if) so the two never both fire for the same cell,
        # mirroring belief_unavailable/belief's own mutual-exclusion
        # exactly. `stats.n_rows > 0` excludes the already-handled
        # zero-rows-in-cell case (n_rows==0 returns early in _game_stats
        # with n_rows_with_completion=0 too, but that shape is a game with
        # no data at all, not a game whose rows all failed to complete).
        if stats.n_rows > 0 and stats.n_rows_with_completion == 0:
            violations.append(Violation("verbose_unavailable", game, 1.0, 0.0))
        elif stats.verbose_rate > thresholds.max_verbose_rate:
            violations.append(
                Violation(
                    "verbose", game, stats.verbose_rate, thresholds.max_verbose_rate
                )
            )
        # If criterion 2 could not be evaluated at all
        # (belief-provenance columns absent from a pilot directory), that
        # is itself a fired criterion -- "exits 0 having established
        # nothing" is exactly the shape this gate exists to prevent. A
        # distinct criterion name ("belief_unavailable") lets an operator
        # tell it apart from a genuine measured belief failure ("belief").
        if not stats.belief_schema_present:
            violations.append(Violation("belief_unavailable", game, 1.0, 0.0))
        elif stats.belief_failure_rate >= thresholds.max_belief_failure_rate:
            violations.append(
                Violation(
                    "belief",
                    game,
                    stats.belief_failure_rate,
                    thresholds.max_belief_failure_rate,
                )
            )
        # Only evaluated when the caller has explicitly configured
        # both threshold fields (belief_verbose_configured) -- otherwise
        # this criterion is genuinely skipped, never "always passes".
        if stats.belief_verbose_configured:
            if not stats.belief_tokens_schema_present:
                violations.append(
                    Violation("belief_verbose_unavailable", game, 1.0, 0.0)
                )
            elif stats.belief_verbose_rate > (
                max_verbose := _configured(
                    thresholds.max_belief_verbose_rate, "max_belief_verbose_rate"
                )
            ):
                violations.append(
                    Violation(
                        "belief_verbose", game, stats.belief_verbose_rate, max_verbose
                    )
                )
        if stats.fallback_rate > thresholds.max_fallback_rate:
            violations.append(
                Violation(
                    "fallback", game, stats.fallback_rate, thresholds.max_fallback_rate
                )
            )
        if stats.n_rows < thresholds.min_rows_per_game:
            violations.append(
                Violation(
                    "vacuity",
                    game,
                    float(stats.n_rows),
                    float(thresholds.min_rows_per_game),
                )
            )

    return PilotReport(
        model="",
        per_game=per_game,
        violations=tuple(violations),
        total_calls=total_calls,
        wall_clock_s=None,
        hand_limit=hand_limit,
        n_rows_outside_hand_limit=n_rows_outside_hand_limit,
        prompt_schema_observed=observed_versions,
        n_rows_missing_prompt_schema=n_rows_missing_prompt_schema,
        live_prompt_schema_version=live_prompt_schema_version,
        shard_outcomes=shard_outcomes,
        exact_policy_coverage_median=exact_policy_coverage_median,
        coverage_mass_floor=coverage_mass_floor,
        exact_policy_enabled=exact_policy_enabled,
        exact_policy_parse_counts=exact_policy_parse_counts,
        exact_policy_error_kinds=exact_policy_error_kinds,
        memory_rung=memory_rung,
        recorded_exact_policy_enabled=recorded_exact_policy_enabled,
    )


def compare_to_reference(
    report: PilotReport, reference_dir: Path, max_rise: float, hand_limit: int = 20
) -> tuple[Violation, ...]:
    """
    the violations of the Tier-2 change-detector, without the record of what it skipped
    :param report: the pilot's report (PilotReport)
    :param reference_dir: the labelled calibration arm's directory (Path)
    :param max_rise: the largest rise in a per-game rate that is not a violation (float)
    :param hand_limit: only reference rows with `hand_idx` below this are compared (int)
    :returns: the violations, empty when no rate rose (tuple[Violation, ...])
    """
    return compare_to_reference_detailed(report, reference_dir, max_rise, hand_limit)[0]


def compare_to_reference_detailed(
    report: PilotReport, reference_dir: Path, max_rise: float, hand_limit: int = 20
) -> tuple[tuple[Violation, ...], dict[str, tuple[str, ...]]]:
    """Tier-2 change-detector: flag a per-game rate that rose relative to a
    labelled calibration (reference) arm's early hands.

    Restricted to the reference's `hand_idx < hand_limit` rows only.
    match_runner.py seeds the deal `rng` once before the hand loop, and
    `n_hands` only bounds that loop, so hands 0..hand_limit-1 draw the same
    deals across arms for games with start-only or no chance nodes (kuhn and
    liarsdice at depths [0, 1]; goofspiel and darkhex have none). Leduc has a
    mid-hand chance node at depths [4, 5], so divergent play desynchronises
    it there; its residual bias is +0.0070 against sd 0.0077 (z=5.58,
    false-block ~0), and the binding game liarsdice is exact. Comparing
    against the reference's full hand range instead of this early window
    measured a +0.018 sampling bias, taking the family-wise false-block
    probability from 0.021 to 0.00078.

    Raises `ReferenceComparisonError` when `reference_dir`
    cannot support a like-for-like comparison at all: no LM rows (hence no
    `hand_idx` column -- an empty/unreadable/mistyped directory), zero
    rows survive the `hand_idx < hand_limit` filter, or
    the surviving rows cover NONE of the games being judged. Silently
    returning `()` in any of these cases reads as "no rise detected" when
    the true state is "nothing was compared" -- indistinguishable from a
    clean pass to a caller that only checks `violations == ()`. A
    reference covering only some of the games is not an error (that is a
    legitimate partial comparison, per-game `continue` below), but its
    caller should report the coverage -- see main()'s tier2_status, which
    uses `_count_reference_games_covered` for exactly that.

    If `reference_dir`'s data lacks the belief-provenance columns entirely
    (a supported older schema, e.g. the landed
    `data/run_experiment/qwen2.5-14b-instruct/` arm predates them), the
    belief_failure_rate metric is skipped for that comparison rather than
    comparing the pilot's real rate against a fabricated reference rate of
    0.0 (which would falsely read as a rise on every arm with any belief
    failures at all).

    A metric that was left out of a game's comparison for either reason is named in the second
    return value, so a skipped metric does not read as a metric with no rise.
    :param report: the pilot's report (PilotReport)
    :param reference_dir: the labelled calibration arm's directory (Path)
    :param max_rise: the largest rise in a per-game rate that is not a violation (float)
    :param hand_limit: only reference rows with `hand_idx` below this are compared (int)
    :returns: the violations, and for each compared game the metrics that were not compared
        (tuple[tuple[Violation, ...], dict[str, tuple[str, ...]]])
    """
    ref_lm = _read_lm_rows(Path(reference_dir))
    if "hand_idx" not in ref_lm.columns:
        raise ReferenceComparisonError(
            f"--reference-dir {reference_dir} has no hand_idx column -- no "
            f"LM rows found (empty, unreadable, or mistyped directory). "
            f"Cannot run the Tier-2 change-detector against it."
        )
    ref_lm = ref_lm.filter(pl.col("hand_idx") < hand_limit)
    if ref_lm.height == 0:
        raise ReferenceComparisonError(
            f"--reference-dir {reference_dir} has zero LM rows with "
            f"hand_idx < {hand_limit} -- nothing comparable to the pilot's "
            f"early hands. Cannot run the Tier-2 change-detector against it."
        )
    # A reference that has some hand_idx < hand_limit rows
    # overall (passing the height==0 guard above) but none for any of the
    # games being judged is exactly as uninformative as an empty
    # directory -- "a reference covering no games" must be the error case
    # it is at the whole-directory level, not a silent
    # zero-comparison pass that reads identically to a clean PASS.
    covered_games = set(ref_lm["game"].unique().to_list()) & set(report.per_game.keys())
    if not covered_games:
        raise ReferenceComparisonError(
            f"--reference-dir {reference_dir} has hand_idx < {hand_limit} rows, "
            f"but none for any of the {len(report.per_game)} game(s) being "
            f"compared ({sorted(report.per_game)}). Cannot run the Tier-2 "
            f"change-detector against it."
        )

    belief_available = _has_belief_failure_schema(ref_lm)
    thresholds = PilotThresholds()  # only .max_words is used by _game_stats

    violations: list[Violation] = []
    skipped: dict[str, tuple[str, ...]] = {}
    for game, stats in report.per_game.items():
        ref_stats = _game_stats(ref_lm, game, thresholds)
        if ref_stats.n_rows == 0:
            continue  # this game has no reference rows; nothing to compare
        metrics = [
            ("fallback_rate", stats.fallback_rate, ref_stats.fallback_rate),
        ]
        # Same discipline as the belief guard just
        # below, applied to the primary criterion -- comparing
        # stats.verbose_rate when it is itself an unmeasured 0.0 placeholder
        # (n_rows_with_completion=0) against a real reference rate would
        # silently read as "no rise" instead of surfacing that this cell
        # could not be evaluated at all (which evaluate_pilot already flags
        # separately via "verbose_unavailable"). Guards both sides, unlike
        # belief's whole-dataframe `belief_available` column-presence check
        # below: raw_completion is a required column (always present per
        # schema.py, unlike the optional belief_raw_completion), so the
        # failure mode here is per-cell row-level emptiness, not column
        # absence -- either side's cell can independently be all-empty.
        if stats.n_rows_with_completion > 0 and ref_stats.n_rows_with_completion > 0:
            metrics.append(("verbose_rate", stats.verbose_rate, ref_stats.verbose_rate))
        # Also require the PILOT's own belief_schema_present
        # -- comparing stats.belief_failure_rate when it is itself an
        # unmeasured placeholder (belief_schema_present=False) against a
        # real reference rate would silently read as "no rise" instead of
        # surfacing that criterion 2 could not be evaluated at all (which
        # evaluate_pilot already flags separately via "belief_unavailable").
        if belief_available and stats.belief_schema_present:
            metrics.append(
                (
                    "belief_failure_rate",
                    stats.belief_failure_rate,
                    ref_stats.belief_failure_rate,
                )
            )
        not_compared = tuple(
            name
            for name in ("verbose_rate", "belief_failure_rate")
            if name not in {m[0] for m in metrics}
        )
        if not_compared:
            skipped[game] = not_compared
        for metric_name, pilot_val, ref_val in metrics:
            rise = pilot_val - ref_val
            if rise > max_rise:
                violations.append(
                    Violation(f"reference_rise_{metric_name}", game, rise, max_rise)
                )
    return tuple(violations), skipped


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _count_reference_games_covered(
    games: Sequence[str], reference_dir: Path, hand_limit: int
) -> int:
    """How many of `games` have >= 1 hand_idx < hand_limit row in `reference_dir`.

    Used only for main()'s Tier-2 status reporting --
    `compare_to_reference` `continue`s past a game the reference
    lacks, so without this `tier2_status` would print "N
    violation(s)" for a comparison that may have covered a fraction of
    the games being judged. Re-reads `reference_dir` independently of
    `compare_to_reference`'s own internal read: a small duplication
    accepted for simplicity, since pilot/reference directories are small
    and this keeps `compare_to_reference`'s return type
    (`tuple[Violation, ...]`) unchanged for every existing caller.
    """
    ref_lm = _read_lm_rows(Path(reference_dir))
    if "hand_idx" not in ref_lm.columns:
        return 0
    ref_lm = ref_lm.filter(pl.col("hand_idx") < hand_limit)
    if ref_lm.height == 0:
        return 0
    present = set(ref_lm["game"].unique().to_list())
    return sum(1 for g in games if g in present)


def _verify_pilot_provenance(out_path: Path, model: str) -> None:
    """Under --skip-sweep, confirm `out_path`'s data actually belongs to
    `model` before evaluating it as though it does.

    Without this check the report is labelled with whatever --model says
    regardless of what is actually in out_path. Scenario: a pilot dies,
    the operator re-runs --skip-sweep against a tab-completed NEIGHBOURING
    directory, and the output reads as though it measured `model` when it
    measured something else entirely -- then 1.5-3 GPU-days go to an arm
    that was never really gated.

    Prefers the `model_id` column on the rows themselves (present on
    every LM row of current data; ABSENT on the older landed
    `data/run_experiment/qwen2.5-14b-instruct/` arm).
    Falls back to `run_meta.json`'s `resolved_config.model` (written by
    `lmopt.logging.run_meta.write_run_meta`, called from
    `scripts/run_sweep.py`'s `run_sweep` with `resolved_config=cfg`, whose
    `model` key is set from `--model`/config at build time;
    present on that same landed arm at
    `data/run_experiment/qwen2.5-14b-instruct/run_meta.json` with
    `resolved_config.model == "Qwen/Qwen2.5-14B-Instruct"`).

    Raises PilotGateConfigError (mapped to exit 1) on a confirmed mismatch,
    and also when a source exists but cannot be read: shards that fail to
    load, or a run_meta.json that is unreadable or names no resolved model.
    Those are knowably wrong now, so they are refused rather than gated under
    a label the directory may not carry. Only when neither record exists at
    all (readable shards with no model_id column and no run_meta.json file)
    does it print a warning and return: such a directory is older than both
    records, cannot be verified either way, and refusing it would make
    --skip-sweep unusable against exactly the oldest arms.
    """
    observed_model = None
    source = None

    files = sorted(Path(out_path).glob("*.parquet"))
    shards_unreadable = None
    if files:
        try:
            from lmopt.analysis.shards import load_shards

            df = load_shards(files)
        except Exception as exc:
            df = None
            shards_unreadable = exc
        if df is not None and "model_id" in df.columns:
            ids = df["model_id"].drop_nulls().unique().to_list()
            if len(ids) == 1:
                observed_model = ids[0]
                source = "model_id column"
            elif len(ids) > 1:
                raise PilotGateConfigError(
                    f"--out-dir {out_path} mixes multiple model_id values "
                    f"{sorted(ids)!r} -- refusing to evaluate it as one arm."
                )

    if observed_model is None:
        meta_path = Path(out_path) / "run_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                found = meta.get("resolved_config", {}).get("model")
            except Exception as exc:
                raise PilotGateConfigError(
                    f"--out-dir {out_path}: run_meta.json exists but cannot be read ({exc}); "
                    "its provenance cannot be checked against --model, so it is refused"
                ) from exc
            if not found:
                raise PilotGateConfigError(
                    f"--out-dir {out_path}: run_meta.json names no resolved_config.model, "
                    "so its provenance cannot be checked against --model; refused"
                )
            observed_model = found
            source = "run_meta.json"

    if observed_model is None and shards_unreadable is not None:
        raise PilotGateConfigError(
            f"--out-dir {out_path}: its shards cannot be read ({shards_unreadable}) and it has "
            "no run_meta.json, so its provenance cannot be checked; refused"
        )

    if observed_model is None:
        print(
            f"WARNING: could not verify --out-dir {out_path}'s provenance "
            f"against --model {model!r} -- no model_id column and no "
            f"readable run_meta.json with a resolved model found. "
            f"Proceeding WITHOUT a provenance check.",
            file=sys.stderr,
        )
        return

    if observed_model != model:
        raise PilotGateConfigError(
            f"--out-dir {out_path} was produced by model {observed_model!r} "
            f"({source}), not --model {model!r}. Refusing to evaluate a "
            f"mismatched directory under --skip-sweep."
        )


def _verify_pilot_rung(
    out_path: Path, expected_rung: str, expected_seat_mode: str
) -> None:
    """Under --skip-sweep, confirm out_path's pilot certifies the rung this
    launch is about to run, and warn (non-blocking) on a differing seat_mode.

    Without this check an M0 pilot directory would pass the gate for an
    M3 arm. Recycle configs give every rung its own per-model
    pilot_dir, but a stale --pilot-dir override or a copy-pasted pilot_dir key
    that forgot to bump its rung suffix would certify the wrong thing with
    nothing to catch it, following exactly _verify_pilot_provenance's own
    "tab-completed NEIGHBOURING directory" scenario one field over.

    memory_rung is a hard gate (raises PilotGateConfigError, exit 1): a
    census of every configs/*.yaml models[].pilot_dir entry against the
    run_meta.json actually at that path found 0 memory_rung divergences
    (91 of 120 pairs had a readable run_meta.json to
    check; the other 29 name a pilot not yet collected) -- every landed
    recycle config's own memory_rung matches its pilot_dir 1:1, so a
    divergence here is a wiring error, never a deliberate design choice.

    seat_mode is not hard-gated, unlike memory_rung: the same census found
    21 seat_mode divergences, all in 5 P6 driven-subgame configs plus one
    smoke config, and all by documented design --
    one such config's own header: "seat_mode
    fixed1: the drive's parity fixes which seat acts first in the subgame ...
    pilot_dir stays the M0 pilot (this cell is an M0 arm)". A hard-equality
    check would refuse those configs. But seat_mode
    is not irrelevant either -- run_sweep.py's `_refuse_foreign_arm`
    treats it as identity-load-bearing for a real arm's own out_dir -- so a
    differing seat_mode is reported as a WARNING, never a refusal: visible to
    whoever reads the gate's output, without breaking a documented,
    intentional pattern this gate has no basis to second-guess.

    Absent run_meta.json, or a run_meta.json with no memory_rung key at all
    (a pilot collected before this field existed), cannot be verified either
    way -- warn and proceed rather than guess, mirroring
    _verify_pilot_provenance's identical reasoning for an unverifiable
    model. Never default an unreadable historical pilot's rung to "M0": that
    would silently pass exactly the mismatch this function exists to catch
    when a genuinely-untracked directory (some real
    ones on disk, e.g. data/pilot/olmo2-13b, predate this field) is reused
    for an M1+ arm.
    """
    meta_path = Path(out_path) / "run_meta.json"
    try:
        meta = json.loads(meta_path.read_text())
        resolved = meta.get("resolved_config", {}) or {}
    except (OSError, ValueError):
        print(
            f"WARNING: could not read {meta_path} to verify --out-dir "
            f"{out_path}'s pilot rung/seat_mode -- proceeding WITHOUT a "
            f"rung check.",
            file=sys.stderr,
        )
        return

    observed_rung = resolved.get("memory_rung")
    if observed_rung is None:
        print(
            f"WARNING: {meta_path} has no memory_rung recorded (a pilot "
            f"collected before this field existed) -- cannot verify it "
            f"certifies {expected_rung!r}. Proceeding WITHOUT a rung "
            f"check.",
            file=sys.stderr,
        )
    elif observed_rung != expected_rung:
        raise PilotGateConfigError(
            f"--out-dir {out_path} was piloted at memory_rung "
            f"{observed_rung!r} ({meta_path}), not {expected_rung!r} as "
            f"this launch's --config declares. Refusing to certify a "
            f"{expected_rung!r} arm against a pilot for a different rung."
        )

    if "seat_mode" not in resolved:
        # Like an absent memory_rung: the pilot's seat mode cannot be verified either way, so
        # say so and compare nothing, never read the absent key as "fixed0".
        print(
            f"WARNING: {meta_path} records no seat_mode (a pilot collected before this field "
            f"existed) -- cannot verify it matches this launch's seat_mode "
            f"{expected_seat_mode!r}. Proceeding WITHOUT a seat_mode check.",
            file=sys.stderr,
        )
        return
    observed_seat_mode = resolved["seat_mode"]
    if observed_seat_mode != expected_seat_mode:
        print(
            f"WARNING: --out-dir {out_path} was piloted under seat_mode "
            f"{observed_seat_mode!r}, but this launch's --config declares "
            f"seat_mode {expected_seat_mode!r}. Not refused (some "
            f"configs deliberately reuse a pilot across seat_mode, e.g. "
            f"P6's driven-subgame cells), but Tier-1 rates measured under "
            f"one seat_mode may not transfer to another -- surfaced "
            f"rather than assumed.",
            file=sys.stderr,
        )


def _verify_pilot_temperature(out_path: Path, expected_temperature: float) -> None:
    """
    under --skip-sweep, refuse a pilot whose recorded temperature is not this launch's
    The pilot gate passes up to 90% belief failure and never read a temperature, so a pilot
    sampled at 0.0 could certify an arm sampled at 0.2, and the pilot's rates describe a
    different decoding from the arm's. An absent run_meta.json cannot be verified either way
    and warns and proceeds, as the memory-level check does; a run_meta.json that records no
    temperature, or a non-numeric one, refuses, because every pilot collected since the
    runner began writing it does record one and a missing key must not read as a match
    :param out_path: the pilot directory, whose run_meta.json is read (Path)
    :param expected_temperature: the temperature this launch's config resolves to (float)
    :returns: None
    :raises PilotGateConfigError: when the pilot's run_meta.json records no usable temperature
        or one that differs from the launch's
    """
    meta_path = Path(out_path) / "run_meta.json"
    try:
        meta = json.loads(meta_path.read_text())
        resolved = meta.get("resolved_config", {}) or {}
    except (OSError, ValueError):
        print(
            f"WARNING: could not read {meta_path} to verify --out-dir {out_path}'s pilot "
            f"temperature -- proceeding WITHOUT a temperature check.",
            file=sys.stderr,
        )
        return
    if "temperature" not in resolved:
        raise PilotGateConfigError(
            f"--out-dir {out_path} ({meta_path}) records no resolved_config.temperature, so it "
            f"cannot certify an arm sampled at {expected_temperature!r}. Re-run the pilot with "
            f"the current runner."
        )
    observed = resolved["temperature"]
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise PilotGateConfigError(
            f"--out-dir {out_path} ({meta_path}) records resolved_config.temperature "
            f"{observed!r}, which is not a number."
        )
    if observed != expected_temperature:
        raise PilotGateConfigError(
            f"--out-dir {out_path} was piloted at temperature {observed!r} ({meta_path}), not "
            f"{expected_temperature!r} as this launch's --config declares. Refusing to certify an "
            f"arm against a pilot sampled differently."
        )


def _live_memory_render_hash() -> tuple[str, str]:
    """
    the run_meta key and the digest of the memory wording that is about to RUN, imported from
    the module that renders it and never compared against anything the pilot recorded about
    itself. The digest renders every canonical game in its own terms, so it needs the game
    package; a gate that cannot import it cannot tell which wording is live, and refuses
    :returns: (the run_meta key, the sha256 hex digest of the live wording) (tuple of str)
    :raises PilotGateConfigError: when the import or the digest fails
    """
    try:
        from lmopt.prompting.memory import MEMORY_RENDER_HASH_KEY, memory_render_hash

        return MEMORY_RENDER_HASH_KEY, str(memory_render_hash())
    except Exception as exc:
        raise PilotGateConfigError(
            f"cannot establish the live memory wording ({type(exc).__name__}: {exc}); a "
            "pilot above M0 cannot be certified without it"
        ) from exc


def _verify_pilot_memory_wording(
    out_path: Path, expected_rung: str, live_hash: str | None = None
) -> None:
    """
    refuse a pilot that cannot be shown to have been collected under the memory wording that
    is live now. The prompt spec hash does not cover the memory module and a pilot's M0 and
    hand-1 prompts are identical under every wording, so without this a pilot collected on an
    old wording would certify a new one. A launch at the memory level that renders no memory
    needs no wording and is not checked
    :param out_path: the pilot directory, whose run_meta.json is read (Path)
    :param expected_rung: the memory level this launch's config resolves to (str)
    :param live_hash: the live wording's digest, or None to compute it; a parameter only so a
        test can pin it (str or None)
    :raises PilotGateConfigError: when the launch is above M0 and the pilot's run_meta.json is
        unreadable, records no wording digest, or records a different one
    """
    if expected_rung == NO_MEMORY_RUNG:
        return
    meta_path = Path(out_path) / "run_meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        raise PilotGateConfigError(
            f"cannot read {meta_path} to check which memory wording the pilot was "
            f"collected under ({exc}); a {expected_rung!r} pilot must record it. "
            f"Re-run the pilot on the current code."
        ) from exc
    key, computed = _live_memory_render_hash()
    live = computed if live_hash is None else live_hash
    recorded = meta.get(key) if isinstance(meta, dict) else None
    if recorded is None:
        raise PilotGateConfigError(
            f"{meta_path} records no {key}, so this "
            f"{expected_rung!r} pilot cannot be shown to use the live memory "
            f"wording ({live}). Re-run the pilot on the current code."
        )
    if recorded != live:
        raise PilotGateConfigError(
            f"{meta_path} was collected under memory wording {recorded!r}, not the "
            f"live {live!r}; a pilot on another wording certifies nothing about "
            f"this one. Re-run the pilot on the current code."
        )


# Measured directly against the landed
# data/run_experiment/qwen2.5-14b-instruct/ arm's own hand_idx < 20 rows:
# kuhn (the binding game -- shortest tree, fewest decision nodes/hand of
# the five; the other four measured 143.4-240.0 rows/seed at the same
# shape, all comfortably above the floor) produced 473 LM rows total
# across 5 seeds x 4 opponents x 20 hands = 400 (seed, opponent, hand)
# triples: 473 / 400 = 1.1825 rows/hand/opponent. This reproduces both
# cited figures exactly: 20 hands x 1 seed x 4 opponents x 1.1825 = 94.6
# (just under the 100-row floor -- the single-seed false-abort risk), and
# 20 x 5 x 4 x 1.1825 = 473 (configs/run_experiment.yaml's n_seeds=5,
# clearing the floor 4.7x). See test_validate_pilot_shape_reproduces_the_
# measured_single_and_five_seed_figures for the reproduction, locked in.
_KUHN_ROWS_PER_HAND_PER_OPPONENT = 473 / (20 * 5 * 4)


def _validate_pilot_shape(cfg: dict, n_hands: int, min_rows_per_game: int) -> None:
    """Pre-flight: would this config's resolved shape even clear the
    non-vacuity floor for kuhn (the binding game), before any GPU time is
    spent on a real sweep?

    run_sweep.py's `_merge` setdefaults `n_seeds` to 1, and several
    shipped configs never set it. At the gate's default --n-hands 20
    with a single-seed, standard-4-opponent config, kuhn is expected to
    produce only ~94.6 rows -- just under the 100-row min_rows_per_game
    floor -- so the pilot would abort post-HOC with a spurious `vacuity`
    violation on an otherwise healthy arm, after the GPU time is already
    spent. Converting that into a pre-flight config error is the point of
    a pre-flight gate. Raises PilotGateConfigError (mapped to exit 1) if
    the resolved shape cannot clear the floor.

    The shape is predicted from the resolved config, not the literal one:

      * an absent `opponents:` key read literally would mean zero opponents,
        so every config that relies on run_sweep.py `_merge`'s own
        4-opponent `opponents` default would be refused with "~0.0 kuhn
        rows" for a sweep that really produces ~473.
      * kuhn's density must not be judged when kuhn is not among the games:
        a depth-sweep config shape
        (`games: [darkhex_2x3, darkhex_3x2, goofspiel_5]`, 2 opponents,
        n_seeds 1) would be refused for producing "only ~47.3 kuhn rows" in
        a pilot that never runs kuhn, while its actually-configured games
        sit at 1,123-1,200 rows, an order of magnitude above the floor.

    Hence resolving through `_resolve_sweep_config` (the single
    mirror of run_sweep.py `_merge`'s setdefault block) and returning
    early when kuhn is
    absent. kuhn is the binding game -- shortest tree, fewest decision
    nodes per hand of the five -- so it is the only one whose density this
    prediction is calibrated for; a games list without it gets no
    prediction rather than kuhn's wrong one.
    """
    resolved = _resolve_sweep_config(cfg)
    games = resolved["games"]
    if not isinstance(games, (list, tuple)) or "kuhn" not in games:
        return
    n_seeds = resolved["n_seeds"]
    n_opponents = len(resolved["opponents"] or [])
    expected_kuhn_rows = (
        n_hands * n_seeds * n_opponents * _KUHN_ROWS_PER_HAND_PER_OPPONENT
    )
    if expected_kuhn_rows < min_rows_per_game:
        raise PilotGateConfigError(
            f"this config's resolved shape (--n-hands {n_hands} x n_seeds "
            f"{n_seeds} x {n_opponents} opponent(s)) is expected to produce "
            f"only ~{expected_kuhn_rows:.1f} kuhn rows -- kuhn is the "
            f"binding game at a measured {_KUHN_ROWS_PER_HAND_PER_OPPONENT:.4f} "
            f"rows/hand/opponent -- below the {min_rows_per_game}-row "
            f"non-vacuity floor. Raise --n-hands or the config's n_seeds, "
            f"or the pilot will abort post-hoc on a spurious vacuity "
            f"violation after GPU time is already spent."
        )


def _resolve_vllm_base_url(cfg: dict, model: str) -> str:
    """Look up `model`'s `vllm_base_url` from the config's `models:` roster.

    run_experiment.py's `sweep_command` forwards
    `entry["vllm_base_url"]` to run_sweep.py for exactly this reason --
    run_sweep.py's `run_sweep` hard-exits `"'vllm_base_url' is required in
    config or via --vllm-base-url"` without it, so every pilot would abort
    at second zero. Raises PilotGateConfigError (mapped to exit 1, an operational
    failure, not GATE_FAILURE_EXIT_CODE) when --model has no matching
    `models:` entry, or that entry has no vllm_base_url.
    """
    for entry in cfg.get("models") or []:
        if entry.get("id") == model:
            base_url = entry.get("vllm_base_url")
            if not base_url:
                raise PilotGateConfigError(
                    f"--config's models: entry for {model!r} has no vllm_base_url"
                )
            return base_url
    raise PilotGateConfigError(
        f"--model {model!r} has no matching entry in --config's models: "
        f"roster (looked for models[*].id == {model!r})"
    )


def _validate_games(games) -> Sequence[str]:
    """Validate a config's resolved `games:` value before it drives the
    non-vacuity criterion.

    One-typo-away failure modes without this check: `games: []` silently reports a clean PASS with
    zero criteria evaluated (exit 0); `games:` (YAML null) raises an
    uncaught `TypeError: 'NoneType' object is not iterable` instead of a
    clean operational failure; `games: {kuhn: 1}` (a mapping) iterates
    dict keys and "works" by accident. Raises PilotGateConfigError
    (mapped to exit 1) on any of these -- the non-vacuity criterion is the
    one criterion standing between a broken sweep and a silent exit 0, so
    it must not itself be silently disarmable by a config typo.
    """
    if (
        not isinstance(games, (list, tuple))
        or len(games) == 0
        or not all(isinstance(g, str) for g in games)
    ):
        raise PilotGateConfigError(
            f"--config's games: must be a non-empty list of strings, got {games!r}"
        )
    return games


def _print_report(report: PilotReport, tier2_status: str) -> None:
    print(f"\n=== Pilot gate report: {report.model or '(unlabelled)'} ===")
    if report.wall_clock_s is not None:
        print(f"  wall_clock_s={report.wall_clock_s:.1f}")
    print(f"  total_calls={report.total_calls}")
    # State the window every number below was computed over. On the real
    # landed Qwen arm's seed-0 shards, kuhn holds 4,863 rows on disk but 95
    # at hand_idx < 20 -- so an operator re-checking that healthy arm at the
    # default --n-hands 20 would read "[vacuity] kuhn: 95.0000 vs threshold
    # 100.0000" and exit 2, "the arm is unusable", with 98% of the data
    # discarded and nothing saying so.
    if report.hand_limit is None:
        print("  scored: every hand in the directory (no hand limit applied)")
    else:
        print(
            f"  scored hand_idx < {report.hand_limit} "
            f"({report.n_rows_outside_hand_limit} LM row(s) outside the "
            f"window, NOT scored)"
        )
    # State what criteria 5 and 6 saw, on a PASS as well as on a
    # refusal. A gate that reports only its violations leaves an operator
    # unable to tell "checked and current" from "not checked".
    observed = (
        ", ".join(str(v) for v in report.prompt_schema_observed) or "NONE RECORDED"
    )
    missing = (
        f", {report.n_rows_missing_prompt_schema} row(s) with no version"
        if report.n_rows_missing_prompt_schema
        else ""
    )
    print(
        f"  prompt_schema: observed {observed}{missing} "
        f"vs live {report.live_prompt_schema_version}"
    )
    if report.shard_outcomes is None:
        print(
            f"  completeness: NO readable {SHARD_OUTCOMES_FILENAME} -- this "
            f"directory carries no evidence that its sweep finished"
        )
    else:
        requested, completed, failed = report.shard_outcomes
        print(
            f"  completeness: {completed}/{requested} shard(s) completed, "
            f"{failed} failed"
        )
    # Criterion 7, same contract as 5 and 6: state what was seen, on a PASS
    # too. The refusal branch names the command that produces the missing
    # artefact -- a fresh pilot sweep does not write one, so under a config
    # that enables the pass this is the first thing an operator meets, and a
    # gate that refuses without saying what to run is a gate people learn to
    # bypass.
    if report.exact_policy_coverage_median is None:
        if report.exact_policy_enabled:
            print(
                f"  exact_policy: NO readable {EXACT_POLICY_FILENAME} -- this "
                f"config enables the exact-policy pass, so this pilot carries "
                f"no evidence that this model's action literals are scorable "
                f"at all. Run scripts/exact_policy_pass.py against this "
                f"directory first."
            )
        else:
            # "no READABLE": this branch is also where a present but corrupt
            # artefact lands, and the line must not claim the directory is
            # empty when it is not.
            print(
                f"  exact_policy: not evaluated (this config does not enable "
                f"the exact-policy pass, and there is no readable "
                f"{EXACT_POLICY_FILENAME} here to judge)"
            )
    else:
        print(
            f"  exact_policy: median coverage_mass "
            f"{report.exact_policy_coverage_median:.4f} vs floor "
            f"{report.coverage_mass_floor:.4f}"
        )
    above_m0 = report.memory_rung in MEMORY_RUNGS_ABOVE_M0
    if report.exact_policy_parse_counts is None:
        if above_m0 and report.recorded_exact_policy_enabled is False:
            print(
                f"  exact_policy parse rate: NOT MEASURED BY CONSTRUCTION (recorded "
                f"memory_rung {report.memory_rung} with exact_policy.enabled false: the "
                f"pass is configured off above {NO_MEMORY_RUNG_FOR_EXACT_POLICY}, so "
                f"this is not a failure)"
            )
        elif above_m0 and report.recorded_exact_policy_enabled is True:
            print(
                f"  exact_policy parse rate: NOT MEASURED (recorded memory_rung "
                f"{report.memory_rung} with exact_policy.enabled true, yet no readable "
                f"{EXACT_POLICY_FILENAME}: the pass was asked for and left nothing)"
            )
        else:
            print(
                "  exact_policy parse rate: NOT MEASURED (no readable "
                f"{EXACT_POLICY_FILENAME} with the key and error columns)"
            )
    else:
        if above_m0:
            print(
                f"  exact_policy WARNING: recorded memory_rung is {report.memory_rung} but "
                f"an {EXACT_POLICY_FILENAME} is present; above "
                f"{NO_MEMORY_RUNG_FOR_EXACT_POLICY} the one prompt kept per info-set "
                f"carries a memory prefix, so this rate is not comparable with "
                f"{NO_MEMORY_RUNG_FOR_EXACT_POLICY} rates"
            )
        parts = ", ".join(
            f"{game} {parsed / total:.4f} ({parsed}/{total})"
            for game, (parsed, total) in report.exact_policy_parse_counts.items()
        )
        print(
            f"  exact_policy parse rate (info-sets with a trusted distribution): {parts}"
        )
        kinds = report.exact_policy_error_kinds or {}
        if kinds:
            print(
                "  exact_policy failed info-sets by error_kind: "
                + ", ".join(f"{k}={n}" for k, n in kinds.items())
            )
        else:
            print("  exact_policy failed info-sets by error_kind: none failed")
    for game, stats in report.per_game.items():
        # Never print a number for an unmeasured
        # Primary criterion either -- same discipline as belief_str below,
        # applied to verbose_rate, which is a meaningless 0.0 placeholder
        # when n_rows_with_completion is 0.
        if stats.n_rows_with_completion > 0:
            verbose_str = f"verbose={stats.verbose_rate:.4f}"
        else:
            verbose_str = (
                "verbose: NOT EVALUATED (no LM row in this cell produced a completion)"
            )
        # Never print a number for an unmeasured criterion 2
        # -- belief_failure_rate is a meaningless 0.0 placeholder when
        # belief_schema_present is False, and printing it as though it were
        # measured is exactly the "exits 0 having established nothing" shape
        # this gate exists to prevent.
        if stats.belief_schema_present:
            salvaged_str = (
                ""
                if stats.belief_salvaged_rate is None
                else f" salvaged={stats.belief_salvaged_rate:.4f}"
            )
            belief_str = (
                f"belief_fail={stats.belief_failure_rate:.4f}{salvaged_str} "
                f"({round(stats.belief_failure_rate * stats.belief_calls_made)}"
                f"/{stats.belief_calls_made} calls)"
            )
        else:
            belief_str = "criterion 2: NOT EVALUATED (belief-provenance columns absent)"
        # Three distinguishable states, printed on every run -- an
        # operator must never be able to mistake "nobody has configured
        # this criterion yet" for "measured and clean".
        if not stats.belief_verbose_configured:
            belief_verbose_str = "belief_verbose: NOT CONFIGURED (no threshold set)"
        elif not stats.belief_tokens_schema_present:
            belief_verbose_str = (
                "belief_verbose: NOT EVALUATED (belief_completion_tokens absent)"
            )
        else:
            belief_verbose_str = f"belief_verbose={stats.belief_verbose_rate:.4f}"
        if stats.retry_rate is None:
            retry_str = "retry: NOT MEASURED (no LM rows)"
        else:
            retry_str = (
                f"retry={stats.retry_rate:.4f} ({stats.n_retried}/{stats.n_rows})"
            )
        # median_words/max_words are printed because the "hard max 2 words"
        # evidence is the single most legible Tier-1 signal an operator has.
        print(
            f"  {game:<10} n={stats.n_rows:<5} {verbose_str} "
            f"median_words={stats.median_words} max_words={stats.max_words} "
            f"{belief_str} fallback={stats.fallback_rate:.4f} "
            f"({round(stats.fallback_rate * stats.n_rows)}/{stats.n_rows}) {retry_str} "
            f"calls/node={stats.calls_per_node:.2f} {belief_verbose_str}"
        )
    print(f"  Tier-2 reference comparison: {tier2_status}")
    if report.violations:
        print("VIOLATIONS (arm is unusable):")
        for v in report.violations:
            print(
                f"  [{v.criterion}] {v.game}: {v.value:.4f} vs threshold {v.threshold:.4f}"
            )
            note = VIOLATION_NOTES.get(v.criterion)
            if note:
                print(f"      {note}")
    else:
        print("PASS -- no Tier-1 criterion fired.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-flight pilot gate: abort a multi-day GPU experiment "
        "when an arm is too broken to be worth measuring"
        "."
    )
    p.add_argument("--config", required=True, help="run_experiment-style YAML config")
    p.add_argument("--model", required=True, help="model id / label for this arm")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Pilot data directory. Must be empty or absent for a "
        "fresh sweep, and must never be under "
        "data/run_experiment/.",
    )
    p.add_argument("--n-hands", type=int, default=20)
    p.add_argument(
        "--calibration",
        action="store_true",
        help="Mark this run as the labelled Tier-2 calibration arm "
        "(Tier 1 still applies; no reference comparison is made).",
    )
    p.add_argument(
        "--reference-dir",
        default=None,
        help="Prior calibration arm's data directory, for the "
        "Tier-2 change-detector (compare_to_reference).",
    )
    p.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Evaluate an existing --out-dir instead of running "
        "scripts/run_sweep.py first.",
    )
    return p.parse_args(argv)


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# The GIT_* variables git itself honours for repo discovery, in precedence
# order over --git-common-dir's normal cwd-based resolution. Many tools
# can leave one of these set in an inherited shell -- stripped rather than merely
# out-cwd-pinned, because pinning `cwd=` alone does not override them (a polluted
# `GIT_DIR` pointing at an unrelated real repo returns rc=0 with that
# repo's own .git path even when the subprocess's cwd is pinned correctly;
# only a *nonexistent* polluted path returns a clean non-zero exit).
_GIT_DISCOVERY_ENV_VARS = (
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
)


@functools.lru_cache(maxsize=1)
def _primary_data_run_experiment_root() -> Path | None:
    """Absolute `data/run_experiment` root of the primary checkout, from any
    worktree. `None` (never raises) if it cannot be determined -- the caller
    must fail closed rather than silently skip the cross-checkout check
    (no silent fallbacks).

    Cached (`lru_cache`, no arguments): a checkout's primary-ness is a git
    repository-structure fact that cannot change within one process's
    lifetime, so re-invoking `git` on every call buys nothing. Tests that
    exercise this function's own git-subprocess behaviour directly must
    call `.cache_clear()` first (and again after, so a mocked result never
    leaks into an unrelated later test in the same process).
    """
    env = {k: v for k, v in os.environ.items() if k not in _GIT_DISCOVERY_ENV_VARS}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            # The exit code is read below, so a failure is handled, not raised.
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    common_dir_str = result.stdout.strip()
    if not common_dir_str:
        return None
    # (_REPO / common_dir_str) handles both an absolute and a cwd-relative
    # result uniformly: pathlib's `/` discards the left operand when the
    # right one is already absolute, and a relative result is correctly
    # anchored to _REPO because that is the cwd the subprocess was pinned
    # to (git's own contract: a relative --git-common-dir result is
    # relative to the invocation's cwd).
    common_dir = (Path(_REPO) / common_dir_str).resolve()
    return common_dir.parent


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path_arg = Path(args.out_dir)

    # A relative --out-dir must be canonicalised to a single
    # absolute path used for every downstream operation (guard, mkdir, the
    # subprocess's --out-dir argument, evaluate_pilot). Path.mkdir() and
    # polars' glob() resolve a relative path against this PROCESS's actual
    # cwd, while the run_sweep.py subprocess is invoked with cwd=_REPO
    # (below), so a relative string handed to it resolves there instead.
    # From a non-repo cwd those diverge: the sweep writes to one directory,
    # the gate reads an empty one, and spurious vacuity violations "prove" a
    # working arm broken (exit 2, "the arm is unusable", for what is
    # actually a tool/cwd bug). Guard against both the true cwd-resolved
    # candidate and the repo-relative one an operator running from the repo
    # root intends, then use the cwd-resolved one (what mkdir/reads/writes
    # actually do) everywhere below.
    cwd_resolved = out_path_arg.resolve()
    repo_resolved = (
        out_path_arg if out_path_arg.is_absolute() else (_REPO / out_path_arg)
    ).resolve()
    forbidden = (_REPO / "data" / "run_experiment").resolve()
    # `forbidden` above is scoped to this process's own checkout only.
    # A worktree's data/run_experiment entries may be
    # symlinks into the primary checkout's real data/run_experiment
    # (top-level whole-directory symlinks for most landed arms, per-file
    # symlinks inside a real top-level directory for a few -- both shapes
    # observed live on disk). `.resolve()` follows either shape, so an
    # --out-dir naming an existing symlinked entry (or a path reaching
    # through one) resolves outside `forbidden`'s worktree-scoped tree even
    # though the write target is a real landed arm's directory on the
    # Primary checkout. Guard against the primary's own data/run_experiment
    # too, derived fresh every run (never cached: a checkout's primary-ness
    # is a git-state fact, not a constant). This is a real, reproduced
    # bypass, not a hypothetical.
    primary_root = _primary_data_run_experiment_root()
    if primary_root is None:
        print(
            "ERROR: could not determine the primary checkout's root via "
            "`git rev-parse --git-common-dir` -- refusing rather than "
            "silently skipping the cross-checkout forbidden-path check.",
            file=sys.stderr,
        )
        return 1
    forbidden_primary = (primary_root / "data" / "run_experiment").resolve()
    # Guard 1 (unconditional -- checked even under --skip-sweep):
    # run_experiment.py's `main` derives out_root/<short> and
    # run_sweep.py's `_shard_out_path` writes
    # {game}_lm_vs_{opp}_seed{i}.parquet, so a --out-dir under
    # data/run_experiment/ inherits that naming convention and would
    # silently overwrite a landed arm (e.g. the landed 100-shard Qwen arm).
    if (
        _under(cwd_resolved, forbidden)
        or _under(repo_resolved, forbidden)
        or _under(cwd_resolved, forbidden_primary)
        or _under(repo_resolved, forbidden_primary)
    ):
        print(
            f"ERROR: --out-dir {args.out_dir!r} is under data/run_experiment/, "
            f"which run_sweep.py's shard-naming convention would silently "
            f"overwrite a landed arm. Refusing.",
            file=sys.stderr,
        )
        return 1
    out_path = cwd_resolved

    # Guard 2: a fresh sweep must never write into a dirty directory.
    # --skip-sweep exists precisely to evaluate an already-populated
    # directory, so this guard does not apply there.
    if not args.skip_sweep and out_path.exists() and any(out_path.iterdir()):
        print(
            f"ERROR: --out-dir {out_path} already exists and is not empty. "
            f"Refusing to mix a fresh pilot sweep into it.",
            file=sys.stderr,
        )
        return 1

    # Resolve --config to an absolute path once, for exactly
    # the same divergence as --out-dir above -- passed raw into the
    # subprocess cmd below while that subprocess runs with
    # cwd=_REPO, a relative --config (e.g. `../configs/run_experiment.yaml`
    # from scripts/) loads fine in this process (which resolves it against
    # its own actual cwd) then dies with FileNotFoundError in the child
    # (which resolves the same string against _REPO instead). --reference-dir
    # needs no equivalent fix: compare_to_reference reads it in-process,
    # never via a subprocess with a different cwd, so it always resolves
    # against this process's one, unchanging, actual cwd.
    config_path = Path(args.config).resolve()
    try:
        cfg = _load_yaml(str(config_path))
    except Exception as exc:  # bad/missing config is an operational failure
        print(f"ERROR: could not load --config {args.config!r}: {exc}", file=sys.stderr)
        return 1

    # The cycle CONFIG's own games list, not the module's
    # hard-coded GAMES tuple -- run_sweep.py `_merge` setdefaults to a 3-game
    # subset and --games overrides it, and a config may use keys absent
    # from GAMES entirely.
    #
    # When the key is ABSENT the fallback must be
    # run_sweep.py's own 3-game default, not this module's 5-game GAMES
    # tuple. With the 5-game fallback, a config with no
    # `games:` key evaluated against a directory holding exactly what
    # run_sweep would have written returns rc=2 with two spurious
    # `vacuity` violations -- "the arm is unusable" for a healthy arm,
    # one of the two catastrophic directions this gate exists to avoid.
    # `_resolve_sweep_config` is the single mirror; see its comment.
    resolved_cfg = _resolve_sweep_config(cfg)
    try:
        games = _validate_games(resolved_cfg["games"])
    except PilotGateConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    thresholds = PilotThresholds()

    # Under --skip-sweep, confirm out_path actually
    # belongs to --model before evaluating it as though it does. A fresh
    # sweep (the `not args.skip_sweep` branch below) needs no equivalent
    # check -- it writes --model's own data by construction.
    #
    # Same reasoning, one axis over -- confirm out_path's pilot
    # certifies the memory_rung this launch resolves to (hard gate) and
    # surface a differing seat_mode (warning only; see _verify_pilot_rung's
    # docstring for why seat_mode is not hard-gated). A fresh sweep needs no
    # equivalent check either: it writes run_meta.json from this same
    # resolved_cfg, so rung/seat_mode cannot diverge from what it just wrote.
    if args.skip_sweep:
        try:
            _verify_pilot_provenance(out_path, args.model)
            _verify_pilot_rung(
                out_path, resolved_cfg["memory_rung"], resolved_cfg["seat_mode"]
            )
            _verify_pilot_temperature(out_path, resolved_cfg["temperature"])
        except PilotGateConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    wall_clock_s = None
    if not args.skip_sweep:
        # run_sweep.py's `run_sweep` hard-exits without a
        # vllm_base_url; look it up from the matching `models:` roster
        # entry before spending any wall clock, following
        # run_experiment.py's `sweep_command`.
        try:
            vllm_base_url = _resolve_vllm_base_url(cfg, args.model)
        except PilotGateConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        # PRE-flight -- would this config's resolved shape
        # even clear the non-vacuity floor for kuhn (the binding game)?
        # Checked before any GPU time is spent, not left to abort the
        # pilot post-hoc on a spurious vacuity violation.
        try:
            _validate_pilot_shape(
                resolved_cfg, args.n_hands, thresholds.min_rows_per_game
            )
        except PilotGateConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        out_path.mkdir(parents=True, exist_ok=True)
        # Subprocess, not import: exercises the identical code path an
        # operator runs by hand. --out-dir
        # is the already-canonicalised absolute `out_path`, not the raw CLI
        # string, so this subprocess (run with cwd=_REPO below) writes to
        # the exact directory evaluate_pilot reads from afterwards,
        # regardless of the gate's own process cwd.
        cmd = [
            sys.executable,
            str(_REPO / "scripts" / "run_sweep.py"),
            "--config",
            str(config_path),
            "--model",
            args.model,
            "--vllm-base-url",
            vllm_base_url,
            "--out-dir",
            str(out_path),
            "--n-hands",
            str(args.n_hands),
        ]
        # `backend` is a load-bearing logged provenance column (every run
        # records its backend) and
        # run_sweep.py refuses a config that does not name one, so the pilot is
        # forwarded exactly the backend of the arm it gates. Left out of
        # _RUN_SWEEP_DEFAULTS deliberately: there is no default to mirror.
        backend = resolved_cfg.get("backend")
        if backend:
            cmd += ["--backend", backend]
        t0 = time.monotonic()
        # check=False: the exit code is read below, so a failure is handled, not raised.
        result = subprocess.run(cmd, cwd=_REPO, check=False)
        wall_clock_s = time.monotonic() - t0
        if result.returncode != 0:
            print(
                f"ERROR: run_sweep.py failed (exit {result.returncode}) -- "
                f"not a gate verdict, a tool failure. See output above.",
                file=sys.stderr,
            )
            return 1

    # Both paths: under --skip-sweep the directory may predate the live
    # memory wording, and after a fresh sweep this confirms run_sweep.py
    # recorded the wording it rendered.
    try:
        _verify_pilot_memory_wording(out_path, resolved_cfg["memory_rung"])
    except PilotGateConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # The pilot is restricted to the same hand window the
    # reference is (compare_to_reference's hand_limit=args.n_hands below).
    # Filtering one side only would make the Tier-2 comparison asymmetric under
    # --skip-sweep against a directory with more hands than --n-hands.
    # Criterion 7's premise comes from the config the orchestrator will read
    # (run_experiment.py's `exact_policy_enabled` resolution in `main`), not
    # from what happens to be on disk: an
    # arm whose cycle runs the exact-policy pass must not be launched on a
    # pilot that never established the pass can say anything about it.
    report = evaluate_pilot(
        out_path,
        games,
        thresholds,
        hand_limit=args.n_hands,
        exact_policy_enabled=_exact_policy_enabled(cfg),
    )
    report = dataclasses.replace(report, model=args.model, wall_clock_s=wall_clock_s)

    # Warn loudly when --skip-sweep is scoring a fraction of
    # what the directory holds. `_validate_pilot_shape` does this
    # arithmetic pre-flight, but it sits inside `if not args.skip_sweep:`
    # above -- it never runs on the path where this failure lives.
    # Only --skip-sweep can hit it: a fresh sweep writes exactly
    # --n-hands hands, so nothing can fall outside the window.
    if args.skip_sweep and report.n_rows_outside_hand_limit > 0:
        print(
            f"WARNING: --out-dir {out_path} holds "
            f"{report.n_rows_outside_hand_limit} LM row(s) at hand_idx >= "
            f"{args.n_hands}, which are NOT scored. The verdict below "
            f"covers only hand_idx < {args.n_hands}. If this directory is "
            f"a full arm rather than a pilot, re-run with a --n-hands "
            f"large enough to cover it, or a `vacuity` violation here "
            f"means the WINDOW is too small, not that the arm is bad.",
            file=sys.stderr,
        )

    # State explicitly whether a Tier-2 comparison ran, and
    # treat an uninformative --reference-dir as an operational failure
    # (exit 1) rather than silently reporting "PASS" with no indication
    # Tier 2 never ran.
    if args.reference_dir and args.calibration:
        tier2_status = "skipped -- this run is the --calibration arm itself"
    elif args.reference_dir:
        # hand_limit must be derived from this pilot's own
        # --n-hands, not compare_to_reference's bare default (=20) --
        # with any --n-hands != 20 the comparison becomes asymmetric
        # (e.g. pilot 5 hands vs reference's first 20), reintroducing the
        # exact sampling bias the like-for-like filter exists to remove.
        try:
            ref_violations, ref_skipped = compare_to_reference_detailed(
                report,
                Path(args.reference_dir),
                thresholds.max_reference_rise,
                hand_limit=args.n_hands,
            )
        except ReferenceComparisonError as exc:
            print(f"ERROR: Tier-2 reference comparison failed: {exc}", file=sys.stderr)
            return 1
        if ref_violations:
            report = dataclasses.replace(
                report, violations=report.violations + ref_violations
            )
        # Report how many of the games were actually
        # compared, so a reference covering only some of them does not
        # print the same "ran against <dir> -- N violation(s)" as one
        # covering all of them.
        n_covered = _count_reference_games_covered(
            list(games), Path(args.reference_dir), args.n_hands
        )
        tier2_status = (
            f"ran against {args.reference_dir} -- {n_covered}/{len(games)} "
            f"game(s) compared, {len(ref_violations)} violation(s)"
        )
        if ref_skipped:
            tier2_status += "; metrics not compared: " + "; ".join(
                f"{game}: {', '.join(names)}"
                for game, names in sorted(ref_skipped.items())
            )
    else:
        tier2_status = "not requested (no --reference-dir)"

    _print_report(report, tier2_status=tier2_status)

    if report.violations:
        return GATE_FAILURE_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
