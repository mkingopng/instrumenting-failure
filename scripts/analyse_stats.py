#!/usr/bin/env python
"""Repeatable per-matchup statistical analysis of a sweep.

This script writes every number it computes to
CSV so a run can be re-analysed, diffed and cited without re-deriving anything.

Three tables, matching the three requested columns:

  1. outcomes.csv   - win / loss / draw counts and rates per (game, opponent),
                      alongside mean reward with a bootstrap interval, all over
                      clean hands (a hand holding a harness-fallback row of the
                      scored LM is dropped whole; `dropped_rate` and
                      `mean_reward_all_hands` sit beside the headline).
  2. properties.csv - each optimality property scored per (game, opponent)
                      cell, using the existing pure scorers on a slice.
  3. decision.csv   - the pre-registered decision rule: estimand, threshold and
                      current status per property.

Design notes that are load-bearing, not decoration:

* Per-cell, not pooled. Keying P3/P5/P7/P8 by game only leaves
  per-opponent structure invisible. Slicing the same pure scorers per
  (game, opponent) recovers it at zero scorer cost (scoring is offline: these
  remain pure functions over the logged Parquet).

* Draws are counted per game, never pooled. Draws are structurally
  impossible in kuhn/liarsdice/darkhex and common in leduc. A pooled draw rate
  describes no game in the run.

* Effective sample size travels with every estimate. A deterministic policy
  pair in a game with no chance nodes replays one identical hand N times, so
  `n_distinct_outcomes` is reported next to every rate and a cell whose outcome
  never varies is flagged DEGENERATE rather than given a spurious interval.

* Known-bad numbers are labelled, not silently dropped (no silent
  fallbacks). P2's br_gap is a best response to a hybrid opponent
  policy (empirical where the opponent's info-set was logged, uniform
  elsewhere), computed with OpenSpiel's imported `BestResponsePolicy`, not a
  clairvoyant walk: see `PROPERTY_STATUS["P2_adaptivity"]` below and
  `P2_adaptivity_coverage`'s per-cell `on_support_fraction`. The hybrid's
  known limitation is `_EmpiricalPolicy`'s silent-uniform denominator.

Usage:
    uv run python scripts/analyse_stats.py --data data/run_experiment/qwen2.5-14b-instruct \
        --out analysis/run_experiment/qwen2.5-14b-instruct

--out is not re-keyed by anything else in this script: two different runs
pointed at the same --out silently share one directory. `run_experiment.py`'s
own `analyse_stats_command` re-keys its default --out on the run's epoch;
a manual/direct invocation like the one above does not get that for
free. Refuses if --out already exists and is non-empty, unless
--allow-overwrite is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

# Cross-worktree import tripwire. `analyse_stats.py` writes the published numbers
# and is invoked directly, so `scripts/__init__.py`'s guard does not cover it.
from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

# Every name below moved to the library; the script keeps them importable under their old
# names, and some are only re-exported here, not used.
from lmopt.analysis.scoring_vocabulary import (
    DECISION_RULE,
    PROPERTY_STATUS,
    FAILED_REASON,
    _STATE_BLIND_SHARE_KEY,
    cell_of,
    decision_table,
    load_exact_policy,
)

_P8_OPPONENT_ROW_REASON = (
    "opponent-side row that P8 does not score; P8 is scored on the LM seat"
)

# The six rows `property_table`'s
# within-seat block publishes, in emit order. Named once so the block's body
# and its exception handler cannot drift apart about which keys the cell owes
# a row.
_WITHIN_SEAT_KEYS = (
    "state_sensitivity_multistate_share_within_seat",
    "state_sensitivity_bins_total",
    "state_sensitivity_bins_multistate_pooled",
    "state_sensitivity_rows_multistate_pooled",
    "state_sensitivity_bins_multistate_within_seat",
    "state_sensitivity_rows_multistate_within_seat",
)


def _p3_temperature_caveat(temperature: float) -> str:
    """
    the caveat the P3 regret row carries: what the number is, and the decoding
    temperature the run was configured with, as provenance. Regret against the
    CFR equilibrium value is defined on whatever actions the run logged, so no
    temperature makes it unmeasurable and this text claims no confound; a
    temperature-1 run's row states its own temperature rather than a default
    :param temperature: the run's configured decoding temperature (float)
    :returns: the caveat text for the row (str)
    """
    return (
        f"regret against the CFR equilibrium value (at most 0, 0 is "
        f"equilibrium-optimal), decoding temperature {temperature}"
    )


# Belief-pipeline defects are game-specific, so the caveat
# has to be too. Attaching leduc's support mismatch to a darkhex row, or the
# darkhex zero-posterior defect to a game that has no P8, would be a misleading
# caveat, which is no better than a missing one.
GAME_CAVEATS = {
    ("P5_calibration", "leduc"): "elicited belief covers 3 keys vs 6 in the posterior; "
    "ECE scores only the stated subset",
    ("P5_calibration", "kuhn"): "elicited belief == true posterior on 83.2% of rows",
    ("P5_calibration", "liarsdice"): "elicited belief null on 76.6% of rows",
    ("P7a_belief_error", "leduc"): "belief/posterior support mismatch on 100% of rows",
    ("P7a_belief_error", "kuhn"): "elicited belief == true posterior on 83.2% of rows",
    ("P7a_belief_error", "liarsdice"): "elicited belief null on 76.6% of rows",
    (
        "P7a_belief_error",
        "darkhex",
    ): "true_posterior is empty, or carries a probability outside [0, 1]",
    ("P7b_action_belief", "liarsdice"): "elicited belief null on 76.6% of rows",
    # The defect that blocks P5_calibration and P7a_belief_error on the kuhn
    # rows applies to P7b on the identical rows (`elicited_equals_posterior`
    # 0.7503 / 0.8630 / 0.8630 / 0.8631 over the four kuhn cells). P7b's
    # estimand is "fraction of actions best-responding to the LM's own belief"
    # (DECISION_RULE below); when that belief is the true posterior on ~86% of
    # rows, the row measures best-response-to-the-TRUE-posterior instead, which
    # is a different quantity and not the one the threshold was pre-registered
    # against. Without this pairing `derive_game_caveat` returns "" and all
    # four cells promote.
    ("P7b_action_belief", "kuhn"): "elicited belief == true posterior on 83.2% of rows",
    (
        "P7b_action_belief",
        "darkhex",
    ): "true_posterior is empty, or carries a probability outside [0, 1]",
    ("P7b_action_belief", "leduc"): "belief/posterior support mismatch on 100% of rows",
    # Omitting goofspiel's mismatch would make goofspiel the only "admissible"
    # P7a game and produce a LM_FAILS_CRITERION verdict resting entirely on
    # defective cells. A defect map that is silently incomplete is worse than
    # no map: it manufactures confidence in exactly the direction the
    # hypothesis wants.
    (
        "P5_calibration",
        "goofspiel",
    ): "belief/posterior key-set mismatch on 56.6% of rows",
    # Dark Hex's elicited belief is deliberately not a distribution: it is one
    # independent Bernoulli per unseen cell, so it neither sums to 1 nor needs
    # to, and the scorers compare it cell by cell. The quantity worth naming
    # beside a Dark Hex calibration number is therefore the same one its P7
    # siblings name -- a posterior summing to zero, i.e. an opponent with no
    # stone in any unknown cell -- not a normalisation the estimand never
    # assumed.
    (
        "P5_calibration",
        "darkhex",
    ): "true_posterior is empty, or carries a probability outside [0, 1]",
    (
        "P7a_belief_error",
        "goofspiel",
    ): "belief/posterior key-set mismatch on 56.6% of rows",
    (
        "P7b_action_belief",
        "goofspiel",
    ): "belief/posterior key-set mismatch on 56.6% of rows",
}

# P5_calibration_per_pair
# And P5_calibration_per_pair_margin score the same rows as P5_calibration
# (property5.py's per-game loop computes all three from one pass over
# `pairs`/`baseline_pairs`), so both inherit identical per-game defect
# caveats rather than risking a hand-copied set silently drifting from the
# entries above. Extending the comprehension (not hand-typing a third
# literal set) is the point: a future fourth P5 sibling property joins the
# same way, by adding its name to the tuple below.
#
# A single dict comprehension with two nested `for` clauses, not an explicit
# module-level `for` statement -- a comprehension has its own scope in
# Python 3 (its loop variables never leak), unlike a bare `for` at module
# level, which would leave a stray name bound in this module's namespace
# after the loop finished. The whole right-hand
# side is fully evaluated into a new dict before `.update()` mutates
# `GAME_CAVEATS`, so there is no risk of iterating a dict while updating it.
GAME_CAVEATS.update(
    {
        (sibling_prop, game): caveat
        for sibling_prop in (
            "P5_calibration_per_pair",
            "P5_calibration_per_pair_margin",
        )
        for (prop, game), caveat in list(GAME_CAVEATS.items())
        if prop == "P5_calibration"
    }
)

# Which defect each (property, game) caveat asserts, and the
# sentence to render once its magnitude has been measured on the arm in hand.
# The percentages baked into `GAME_CAVEATS` above came from one pre-repair
# arm; used as literals, they would make every later arm inherit a
# pre-repair defect profile as its admissibility gate, emitting the same
# blocker text even where the defect the text asserts is gone.
#
# The two figures involved carry different denominators:
# the 76.6% belief-null rate is LIARSDICE-only in the superseded v1 Qwen arm
# (27,559 null of 35,981 liarsdice LM rows); the 661,644 figure is the
# Cycle-wide LM row count across all five games and all three v2 arms, over
# which elicited-belief nulls are exactly 0.
#
# `GAME_CAVEATS` deliberately stays: it remains the authority on which defect
# applies to which (property, game) pairing, and the two key sets must stay equal
# so a new entry in either table cannot silently lack its partner. Only the
# percentage and the presence of the caveat become data-driven.
CAVEAT_TEMPLATES = {
    ("P5_calibration", "leduc"): (
        "key_set_mismatch",
        "elicited belief covers a different key set on {pct} of rows; "
        "ECE scores only the stated subset",
    ),
    ("P5_calibration", "kuhn"): (
        "elicited_equals_posterior",
        "elicited belief == true posterior on {pct} of rows",
    ),
    ("P5_calibration", "liarsdice"): (
        "elicited_null",
        "elicited belief null on {pct} of rows",
    ),
    ("P5_calibration", "goofspiel"): (
        "key_set_mismatch",
        "belief/posterior key-set mismatch on {pct} of rows",
    ),
    # The Dark Hex family is scored, so this pairing is reached, and all three
    # of its entries name the quantity the gate actually asks of a per-cell
    # posterior. A zero-summing posterior
    # is not that quantity here: it is a legitimate answer, and it still
    # reaches the caveat as an ordinary following census clause at its own
    # measured rate.
    ("P5_calibration", "darkhex"): (
        "posterior_cells_malformed",
        "true_posterior is empty, or carries a probability outside [0, 1], on "
        "{pct} of rows",
    ),
    ("P7a_belief_error", "leduc"): (
        "support_mismatch",
        "belief/posterior support mismatch on {pct} of rows",
    ),
    ("P7a_belief_error", "kuhn"): (
        "elicited_equals_posterior",
        "elicited belief == true posterior on {pct} of rows",
    ),
    ("P7a_belief_error", "liarsdice"): (
        "elicited_null",
        "elicited belief null on {pct} of rows",
    ),
    ("P7a_belief_error", "darkhex"): (
        "posterior_cells_malformed",
        "true_posterior is empty, or carries a probability outside [0, 1], on "
        "{pct} of rows",
    ),
    ("P7a_belief_error", "goofspiel"): (
        "key_set_mismatch",
        "belief/posterior key-set mismatch on {pct} of rows",
    ),
    ("P7b_action_belief", "liarsdice"): (
        "elicited_null",
        "elicited belief null on {pct} of rows",
    ),
    # Same defect key and same sentence as
    # ("P7a_belief_error", "kuhn") and ("P5_calibration", "kuhn") above: the
    # three properties are scored on the identical kuhn rows, so a defect that
    # blocks two of them cannot leave the third clean. See the rationale on the
    # matching `GAME_CAVEATS` entry.
    ("P7b_action_belief", "kuhn"): (
        "elicited_equals_posterior",
        "elicited belief == true posterior on {pct} of rows",
    ),
    ("P7b_action_belief", "darkhex"): (
        "posterior_cells_malformed",
        "true_posterior is empty, or carries a probability outside [0, 1], on "
        "{pct} of rows",
    ),
    ("P7b_action_belief", "leduc"): (
        "support_mismatch",
        "belief/posterior support mismatch on {pct} of rows",
    ),
    ("P7b_action_belief", "goofspiel"): (
        "key_set_mismatch",
        "belief/posterior key-set mismatch on {pct} of rows",
    ),
}
# The P5 siblings inherit their templates by the same comprehension (and for
# the same reason) as they inherit the `GAME_CAVEATS` entries above: all three
# score the same rows, so a hand-copied third literal set could drift.
CAVEAT_TEMPLATES.update(
    {
        (sibling_prop, game): entry
        for sibling_prop in (
            "P5_calibration_per_pair",
            "P5_calibration_per_pair_margin",
        )
        for (prop, game), entry in list(CAVEAT_TEMPLATES.items())
        if prop == "P5_calibration"
    }
)

# GAME_CAVEATS' percentages above were measured once, on one pre-repair arm;
# re-emitted verbatim, they would hide any belief-pipeline repair from the
# caveat text. `measure_belief_defects` recomputes the same quantities
# from the arm actually being analysed; `derive_game_caveat` wires its output
# into the caveat string in place of the hardcoded literal. This function does not
# touch GAME_CAVEATS or emit() itself.
# `node_idx` is a required column: the identity census is
# decomposed at the node (`identity_trivial_prior` is an identity at
# `node_idx == 0`, where restating the prior is the correct answer and evidence
# of nothing), so a frame without the ordinal cannot be decomposed at all. It
# is a logged column of the schema (`src/lmopt/logging/schema.py`), never
# derived here, and its absence raises rather than defaulting to 0 -- a default
# would classify every tracking identity as a trivial prior.
_BELIEF_DEFECT_COLUMNS = ("prompt", "elicited_belief", "true_posterior", "node_idx")
# `state_sensitivity_cell_b_clean` needs
# `legal_mask`/`info_set`/`parsed_action` (via `state_sensitivity.
# seat_bin_stats()`), which are not in `_BELIEF_DEFECT_COLUMNS` above:
# unlike that tuple's four columns, these three are irrelevant to every
# Other key this function computes, and many existing test fixtures
# never carry them. Requiring them unconditionally
# would raise `KeyError` for the whole cell -- blanking eight unrelated,
# perfectly computable keys -- over a column only the ninth needs.
# `_state_sensitivity_cell_b_clean` below therefore catches their absence
# itself and reads `None` (unmeasurable, same as the no-`lm_seat`-column
# case) rather than raising. `polars.exceptions.ColumnNotFoundError` does
# Not inherit from `KeyError`/`AttributeError`/`TypeError` (in polars 1.44
# its MRO includes none of the three), so a missing column here is checked proactively
# rather than relied on to raise a catchable type.
_STATE_SENSITIVITY_CELL_B_COLUMNS = ("legal_mask", "info_set", "parsed_action")
# The quantities that compare `elicited_belief`
# against `true_posterior`, and therefore share the assessable-rows
# denominator. Defined at module level so the full key set has one authority: `property_table`'s
# unmeasurable-cell path builds an all-`None` dict from it, and a key that
# exists in one place but not the other would be a `KeyError` in
# `derive_game_caveat`, never a silently absent caveat.
#
# All but `key_set_mismatch_raw` are computed on the
# scorer's aligned pair (leduc rank marginalisation, goofspiel zero-pad after
# the support precondition), so the census speaks about the representation the
# scorers actually compare. `key_set_mismatch_raw` keeps the raw unaligned
# comparison for continuity with earlier records that quote it (leduc
# 100 %). The three
# `identity_*` keys decompose `elicited_equals_posterior` on the same
# denominator and must sum to it; the falsification condition for reading the
# identity census as benign is `identity_tracking > 0` on a future arm, which
# re-opens the question.
_BELIEF_DEFECT_TWO_SIDED_KEYS = (
    "key_set_mismatch",
    "key_set_mismatch_raw",
    "elicited_equals_posterior",
    "support_mismatch",
    "identity_trivial_prior",
    "identity_tracking",
    "identity_zero_sum",
)
# Census: reported, never blocking. These are descriptions of
# what the cell's beliefs look like, not of whether the scorer can score them.
# `aligned_frac` is `assessable_frac`'s aligned twin and exists
# because the two-sided rates do not share the assessable denominator: they
# speak for the rows that have a scorer representation, which is fewer whenever
# a goofspiel row is refused. It is a coverage report (1.0 is clean), never a
# defect, so it carries no `_DEFECT_SENTENCES` entry and cannot render a clause
# -- exactly as `assessable_frac` does not.
# `posterior_unparseable` exists because `posterior_well_formed` is
# denominated on assessable rows: a row whose `true_posterior` does not parse
# is not assessable, so it cannot trip the gate, and a defect that can neither
# block nor be read is a defect that vanished. It is denominated on LM rows carrying a NON-NULL
# `true_posterior` (the rows where parsing was attempted), which is the same
# population `posterior_zero_sum` speaks for one step later.
# `posterior_cells_malformed` is the per-cell posterior's own well-formedness,
# measured for every family because the question is meaningful for any
# posterior: an empty hypothesis map, or a probability outside [0, 1], is not a
# posterior whatever shape the family's belief takes. It is what
# `posterior_well_formed` gates on for a family whose belief is not a simplex,
# where summing to zero is a legitimate answer rather than a defect.
_BELIEF_CENSUS_KEYS = (
    "elicited_null",
    "assessable_frac",
    "aligned_frac",
    "posterior_zero_sum",
    "posterior_unparseable",
    "posterior_cells_malformed",
) + _BELIEF_DEFECT_TWO_SIDED_KEYS
# SCORABILITY: the gate. Each answers "can the scorer produce a
# number on this cell at all", which is the only question an admissibility gate
# is entitled to ask of a belief column. `alignment_ok` and
# `true_support_within_stated` are the two failure modes of the scorers' own
# alignment step (`src/lmopt/scoring/_belief_alignment.py`), consumed here and
# never re-implemented; `posterior_well_formed` is the posterior-side
# precondition; `assessable_rows` is the count the other three speak for.
# `state_sensitivity_cell_b_clean` implements the scoring protocol's
# state-sensitivity rule: gate exactly the bins landing in the pre-registered
# S7 cell (b) -- action-constant
# despite genuine within-seat state variation, and the belief label "flat"
# (not tracking) -- computed fresh per cell from that cell's own bins, never
# a fixed list. Independent of the other three: it asks about the ACTION/SEAT
# axis, not about belief/posterior parsing.
_BELIEF_SCORABILITY_KEYS = (
    "alignment_ok",
    "true_support_within_stated",
    "posterior_well_formed",
    "assessable_rows",
    "state_sensitivity_cell_b_clean",
)
_BELIEF_DEFECT_KEYS = _BELIEF_CENSUS_KEYS + _BELIEF_SCORABILITY_KEYS
#: What a belief-defect dict holds: the measured quantities, or None where unreadable. The fallback
#: dict for a cell that raised also carries the census-error text under `_CENSUS_ERROR_KEY`.
BeliefDefects = dict[str, float | int | bool | str | None]
# The sentinel `property_table` puts in the
# all-`None` fallback dict when `measure_belief_defects` raised on a cell. It
# is deliberately not a member of `_BELIEF_DEFECT_KEYS`: it is not a measured
# quantity, it is the reason there are none, and it must not acquire a
# sentence, a denominator or a place in any totality test. `measure_belief_
# defects` itself never sets it.
_CENSUS_ERROR_KEY = "_census_error"
# The order the scorability keys are asked in, root cause first,
# which is not the declaration order above and is deliberately its own tuple
# (a per
_BELIEF_GATE_ORDER = (
    "assessable_rows",
    "true_support_within_stated",
    "alignment_ok",
    "posterior_well_formed",
    "state_sensitivity_cell_b_clean",
)
# The keys whose meaning is family-scoped: they exist only where
# the scorers define an aligned belief/posterior pair. Darkhex belief is
# per-cell occupancy (independent Bernoullis, not a simplex), and
# `_belief_alignment.game_row_tvds` raises on the family; there is no aligned
# pair to compare, so these read `None` with
# the family reason rather than a raw comparison that would look like a
# measurement of something. `posterior_zero_sum` is deliberately not in this
# set: it asks about the posterior alone and is a real, measured darkhex defect
# (33-42 % of v4 pilot rows).
_BELIEF_FAMILY_SCOPED_KEYS = _BELIEF_DEFECT_TWO_SIDED_KEYS + (
    "aligned_frac",
    "alignment_ok",
    "true_support_within_stated",
)
_FAMILY_NOT_APPLICABLE_REASON = (
    "family has no simplex belief/posterior pair (per-cell occupancy)"
)
# The families whose belief is not a probability simplex, so the simplex-shaped
# Census keys above have nothing to describe on them. Distinct from
# `_BELIEF_NOT_APPLICABLE_FAMILIES` below, and the distinction is the point:
# Dark Hex belief is scored by P5/P7, with its own per-cell estimand
# (`src/lmopt/scoring/darkhex_belief.py`), so the family is not
# not-applicable; what it lacks is the aligned simplex pair
# `key_set_mismatch` and its siblings are defined over.
_NO_SIMPLEX_PAIR_FAMILIES = ("darkhex",)
# The families for which P5/P7a/P7b are NOT_APPLICABLE
# by scorer design, and the reason a row carries when they are. This is a
# different statement from every other gate reason below: not "the scorer could
# not score this cell" but "this property is not the one that measures this
# family's beliefs".
#
# Empty, and the mechanism is kept rather than deleted: `score_property5`/
# `score_property7` score Dark Hex with a per-cell estimand
# (`src/lmopt/scoring/darkhex_belief.py`), so no family is
# belief-not-applicable. The tuple is
# what `scripts/scoreboard.py:structural_by_design` consults to decide whether
# a blank P5/P7 cell is structural, so a future family that genuinely has no
# belief property joins it here and needs no second mechanism.
#
# Kept as a tuple of families rather than a per-game list: `GAME_FAMILIES`
# resolves every depth variant to its family, exactly as the scorers' own loops
# do.
_BELIEF_NOT_APPLICABLE_FAMILIES = ()
_BELIEF_NOT_APPLICABLE_REASON = "belief property is P8 (occupancy)"
# The one family whose scorer asserts the align-to-stated precondition rather
# than silently applying it. A goofspiel row whose `true_posterior` carries a
# key the elicited belief does not state is refused by both `score_property5`
# and `score_property7` (`_belief_alignment.assert_true_support_within_stated`,
# called only from each scorer's `elif family == "goofspiel":` branch), while
# the LEDUC equivalent -- a rank-marginalised posterior with mass outside the
# stated card names -- returns ECE 0.5 and TVD 1.0 without complaint. So the
# same measured quantity is a scorer refusal on goofspiel and a census fact
# everywhere else. Each scorer confines the refusal to the offending game's
# `not_applicable` entry, so one corrupt goofspiel row does not abort every
# Other game's results in the same call; this file does not call the scorers
# for this check, only the shared assertion directly.
#
# The reason for the scoping is that asymmetry, and nothing else. On real
# leduc the posterior is marginalised to rank before the check, and the check
# then passes: `true_support_within_stated` is True on all 64 non-darkhex
# (game, opponent) cells of the four pilot arms (sdv3 x3 plus recycle-v4 M0,
# via `measure_belief_defects` over each cell's shards). Gating it universally
# would block nothing at all; it is scoped because the non-goofspiel scorers
# do not refuse these rows -- they score the stated subset and silently drop
# posterior mass outside it -- so on those families the condition is a census
# fact this file reports rather than a refusal it can inherit.
#
# Falsification condition: `true_support_within_stated` reading False on any
# Scored non-goofspiel cell re-opens whether the scorers' silent mass drop is
# acceptable. That is a scorer-semantics question, not a gate change made here.
_SCORER_ASSERTS_SUPPORT_FAMILIES = ("goofspiel",)

# Why a cell could not be measured, PER defect key. A single reason for all of
# them ("no rows carried a parseable belief/posterior pair") is true only of
# the two-sided keys. The denominators differ (see `measure_belief_defects`'
# docstring), so `None` means something different in each case, and a wrong
# reason would reach properties.csv and -- via `decision_table`'s
# `caveat.split(";")[0]` -- decision.csv's blocker column, telling a reader to
# go looking for unparseable belief pairs in a cell that simply had no LM rows.
# Only the `None`-producing branch of each denominator is described:
#   * `elicited_null` is denominated on ALL LM rows, so it is `None` only from
#     `measure_belief_defects`' `n == 0` early return (or the whole-cell
#     unmeasurable fallback in `property_table`), i.e. no LM rows at all.
#   * `assessable_frac` shares that denominator (never `None` otherwise) and is
#     named by no template; carried so this table stays total over
#     `_BELIEF_DEFECT_KEYS`.
#   * `posterior_zero_sum` is denominated on rows whose `true_posterior` parses,
#     independent of the elicited belief.
#   * the two-sided keys are denominated on assessable or aligned rows (below).
_NOT_ASSESSABLE_REASONS = {
    "elicited_null": "no LM rows in this cell",
    "assessable_frac": "no LM rows in this cell",
    "posterior_zero_sum": "no rows carried a parseable true_posterior",
    "posterior_unparseable": "no rows carried a true_posterior at all",
    "posterior_cells_malformed": "no rows carried a parseable true_posterior",
    # Split, because the two-sided keys do not share a denominator.
    # `key_set_mismatch_raw` is denominated on assessable rows, so its
    # `None` really does mean no pair parsed. Every other two-sided key (and
    # `aligned_frac`) is denominated on aligned rows, which a row can fail to
    # reach with both columns parsing perfectly: an all-refused goofspiel cell
    # parses every row and aligns none, and telling a reader to go hunting for
    # unparseable JSON there would be false.
    "key_set_mismatch_raw": "no rows carried a parseable belief/posterior pair",
    **dict.fromkeys(
        [k for k in _BELIEF_DEFECT_TWO_SIDED_KEYS if k != "key_set_mismatch_raw"],
        "no rows carried a belief/posterior pair the scorer could parse " "and align",
    ),
    # The scorability keys' own None branches. `alignment_ok` and
    # `true_support_within_stated` are judgements about assessable rows, so
    # they are None exactly when there are none; `posterior_well_formed` shares
    # that denominator; and
    # `assessable_rows` is an honest 0 whenever there are LM rows, so it is
    # None only from the no-LM-rows early return.
    "aligned_frac": (
        "no rows carried a belief/posterior pair the scorer " "could parse and align"
    ),
    "alignment_ok": "no assessable belief/posterior pair to align",
    "true_support_within_stated": "no assessable belief/posterior pair to check",
    "posterior_well_formed": "no assessable belief/posterior pair to check",
    "assessable_rows": "no LM rows in this cell",
    # One static reason covering all three
    # underlying causes (missing `legal_mask`/`info_set`/`parsed_action`, no
    # `lm_seat` column, or a seated frame where no legal-mask bin held two
    # or more distinct prompts within a single seat) -- matching every
    # sibling scorability key's single-sentence convention above, rather
    # than inventing a new per-key dynamic-reason mechanism this file does
    # not otherwise have.
    "state_sensitivity_cell_b_clean": (
        "no within-seat state-sensitivity evidence exists in this cell "
        "(missing legal_mask/info_set/parsed_action, no `lm_seat` column, "
        "or no legal-mask bin held two or more distinct prompts within a "
        "single seat), so whether an S7 cell-(b) bin exists "
        "could not be checked"
    ),
}


def _not_assessable_reason(defect_key, game) -> str:
    """Why `defect_key` reads `None` on a cell of `game`.

    The reason is per KEY (the denominators differ, see
    `_NOT_ASSESSABLE_REASONS`) and per family: on darkhex the two-sided keys
    and the two alignment keys are `None` because the family has no aligned
    belief/posterior pair at all, not because this cell happened to carry no
    parseable one. Telling a reader to go looking for unparseable pairs in a
    darkhex cell where every row parses would be a false statement.

    Deliberately `[]` rather than `.get(..., <default>)`: a defect key with no
    recorded reason is a bug in this file and must stop the run, not render a
    plausible-sounding sentence that is false.
    """
    from lmopt.games.families import GAME_FAMILIES

    if (
        GAME_FAMILIES.get(game, game) in _NO_SIMPLEX_PAIR_FAMILIES
        and defect_key in _BELIEF_FAMILY_SCOPED_KEYS
    ):
        return _FAMILY_NOT_APPLICABLE_REASON
    return _NOT_ASSESSABLE_REASONS[defect_key]


# The properties whose value is computed from
# `elicited_belief` and/or `true_posterior`, and which are therefore gated on
# Every belief defect rather than on the single one their `CAVEAT_TEMPLATES`
# entry happens to name.
#
# Why this list exists at all. Consulting exactly one key per (property, game)
# certifies a cell against one question and leaves it blind to the others. On
# one arm, `elicited_equals_posterior` read 0.3391-0.5145 on mistral-nemo's
# kuhn cells (blocked, because kuhn's template names that key) and
# 0.5000-0.5866 on its liarsdice cells (which would be admissible, because
# liarsdice's template names `elicited_null` only, which reads 0.0). A gate
# that asks one question is not a gate; it is a coincidence.
#
# `P8_value_of_information` is deliberately not a member.
# `score_property8` reads `game`, `prompt`/`actor`, `player_to_act`, `info_set`
# and `parsed_action`, and touches neither `elicited_belief` nor
# `true_posterior`. Gating it on belief defects blocks every P8 cell on
# `posterior_zero_sum` while the scorer emits a shortfall, a capture rate and
# thousands of informative nodes per cell, and on several arms the blocking
# defect's rows and the scored rows are disjoint (`informative and
# zero-posterior` = 0 rows). This is the same defect class the paragraph below
# excludes P1/P2/P3/P4/P6 for.
#
# P8 is not thereby promoted. It gets its own caveat, computed per cell from its
# own measured output by `p8_caveat()` and passed through `emit`'s
# `caveat_override`, exactly as P3_randomisation's temperature caveat is: a
# computation over what was measured, never a literal.
#
# Non-belief properties (P1/P2/P3/P4/P6) and `state_sensitivity_mask_constant`
# are deliberately ABSENT and must stay so: none of them reads a belief column,
# so a belief defect is not evidence about their value, and gating them on one
# would be a caveat that does not name a defect of the thing it blocks. This
# tuple must equal `CAVEAT_TEMPLATES`' property set, so a seventh
# belief property is not added to one and forgotten in the other.
_BELIEF_SCORED_PROPERTIES = (
    "P5_calibration",
    "P5_calibration_per_pair",
    "P5_calibration_per_pair_margin",
    "P7a_belief_error",
    "P7b_action_belief",
)

# Which quantities gate admissibility is `belief_scorability_gates(game)`
# below, computed per family from the scorers' own refusal conditions. There is
# deliberately no module-level `_BELIEF_GATING_KEYS` tuple: a constant that
# nothing consults, pinned by a test that therefore always passes, is a success
# signal rather than the work, and it would go stale because the gate is
# per-family. Invariant: the gate set comes only from that function, for
# every registered game.
#
# The gate is not "every measured quantity except `assessable_frac`", i.e. the
# whole census. That gate asks the wrong question: it blocks leduc at
# `key_set_mismatch` 100 % over a rank-vs-index granularity difference the
# scorers reconcile (`leduc_rank.marginalise_to_rank`), and blocks kuhn at
# `elicited_equals_posterior` on rows that are all `node_idx == 0` identities
# with a support-uniform posterior, i.e. the model correctly restating the
# trivial prior. Neither fact stops `score_property5`/`score_property7`
# producing a number, and an admissibility gate answers exactly one question:
# can the scorer score this cell. The census keys stay measured, stay
# published, and lead the caveat text (`CAVEAT_TEMPLATES`), but do not gate.
#
# `assessable_frac` stays out of the gate by construction: it is a
# Coverage report, not a defect, and it is a census key. `assessable_rows` is
# the gating face of the same rows -- zero assessable rows means nothing to
# score, which is a scorability failure rather than a coverage figure.

# The sentence for a defect named as an additional blocker (i.e. any gating key
# that is not the one `CAVEAT_TEMPLATES` names first for this (property, game)).
# `CAVEAT_TEMPLATES` remains the authority on which defect is named first and on
# the exact wording of that primary sentence -- several of its entries carry a
# game-specific second clause (leduc's "ECE scores only the stated subset") that
# would be wrong to repeat for another game. These are the neutral, game-
# independent renderings for the rest, worded identically to the primary
# sentences already in `CAVEAT_TEMPLATES` so a reader meets one phrasing per
# defect, wherever it appears in the caveat.
_DEFECT_SENTENCES = {
    "elicited_null": "elicited belief null on {pct} of rows",
    "posterior_zero_sum": "the opponent holds no unseen cell on {pct} of rows",
    # The rows the scorers skip in silence
    # (`score_property5`'s `except (json.JSONDecodeError, TypeError):
    # continue`), reported so a reader can size what the gate does not see.
    "posterior_unparseable": (
        "true_posterior present but unparseable on {pct} of rows (these rows "
        "are outside the scored population, so they are reported here rather "
        "than withholding the cell)"
    ),
    "posterior_cells_malformed": (
        "true_posterior is empty, or carries a probability outside [0, 1], on "
        "{pct} of rows"
    ),
    "key_set_mismatch": "belief/posterior key-set mismatch on {pct} of rows",
    "elicited_equals_posterior": "elicited belief == true posterior on {pct} of rows",
    "support_mismatch": "belief/posterior support mismatch on {pct} of rows",
    # The four decomposition census keys; a census quantity nobody can read is
    # not a census.
    "key_set_mismatch_raw": (
        "belief/posterior key-set mismatch BEFORE the scorer's own alignment "
        "on {pct} of rows"
    ),
    "identity_trivial_prior": (
        "elicited belief == true posterior at the first node of the hand, on "
        "a posterior uniform over its support, on {pct} of rows (restating "
        "the trivial prior, which is the correct answer there)"
    ),
    # The falsification condition travels with the number, in the artefact a
    # reader meets: a measured figure licensing inaction is the one nobody
    # re-checks.
    "identity_tracking": (
        "elicited belief == true posterior away from the trivial prior on "
        "{pct} of rows (identity_tracking > 0 on any arm re-opens the "
        "identity question)"
    ),
    "identity_zero_sum": (
        "elicited belief == true posterior on a zero-sum posterior on {pct} " "of rows"
    ),
}

# The census keys that get a clause in the caveat, in the order
# they are rendered. Derived from `_DEFECT_SENTENCES` over `_BELIEF_CENSUS_KEYS`
# rather than hand-typed, so a census key cannot be measured, published and
# then silently unreadable: adding a key to `_BELIEF_CENSUS_KEYS` and a
# sentence to `_DEFECT_SENTENCES` is all it takes to make it render.
# The four decomposition census keys (`key_set_mismatch_raw`, the three
# `identity_*`) have sentences and therefore clauses, which is the whole
# point of decomposing `elicited_equals_posterior`: a reader of the caveat
# sees which kind of identity the cell carries, not only that it carries one.
# The two coverage keys (`assessable_frac`, `aligned_frac`) deliberately have
# none and can never render a clause -- they are clean at 1.0, not at 0.0.
_BELIEF_CENSUS_CLAUSE_KEYS = tuple(
    k for k in _BELIEF_CENSUS_KEYS if k in _DEFECT_SENTENCES
)

# What a FAILED scorability check says. These are not rates:
# each is a yes/no precondition of the scorer, so the sentence names the
# refusal, never a percentage.
_SCORABILITY_SENTENCES = {
    "alignment_ok": (
        "belief/posterior alignment failed on some assessable "
        "rows, so the scorer's representation does not exist "
        "for them"
    ),
    "true_support_within_stated": (
        "true_posterior carries mass on keys the "
        "elicited belief does not state, which the "
        "scorers' align-to-stated convention would "
        "discard rather than compare"
    ),
    # The blocker text carries its own falsification condition, because
    # withholding the cell is a decision not to reconsider a divergence from
    # scorer authority, and a figure that licenses inaction is never
    # questioned again unless the condition travels with it.
    # The clause is worded on the zero-sum case alone: the key is
    # denominated on assessable rows, where "parseable" holds by construction.
    "posterior_well_formed": (
        "true_posterior sums to zero on some of the rows "
        "the scorers would score, so a calibration "
        "number here would be scored against a "
        "non-distribution -- the scorers do NOT refuse "
        "these rows (ECE/TVD 0.5), so the cell is "
        "withheld deliberately and this clause "
        "appearing at all means the rows need investigating"
    ),
    "assessable_rows": "no assessable belief/posterior rows in this cell",
    # At least one
    # (opponent, legal-mask) bin in this cell is an S7 pre-registered cell-(b)
    # bin: action-constant within a seat that genuinely held two or more
    # distinct prompts (so the state varied and the action did not respond),
    # And the belief label is "flat" (elicited_belief does not track
    # true_posterior on that bin either) -- computed fresh from this cell's
    # own rows, per state_sensitivity.qualifying_masks_action_constant, never
    # a fixed list. Per-bin naming (which (opponent, mask) pairs) is
    # deliberately absent, as decision_table carries no per-bin row
    # granularity: this
    # sentence names the condition, decision.csv's caller can re-derive the
    # specific bins from the same cell's Parquet if needed.
    "state_sensitivity_cell_b_clean": (
        "at least one legal-mask bin in this cell is action-constant despite "
        "genuine within-seat state variation, with a belief that does not "
        "track the posterior either (S7 pre-registered cell (b))"
    ),
}

# What a FAILED scorability check says on a family whose belief is not a
# simplex. `posterior_well_formed` asks a different question there, so it needs
# a different sentence: a posterior summing to zero is a legitimate per-cell
# answer (the opponent holds none of the unseen cells), and what would make the
# posterior unusable instead is naming no cell at all, or a probability outside
# [0, 1].
_NO_SIMPLEX_SCORABILITY_SENTENCES = {
    "posterior_well_formed": (
        "true_posterior is empty, or carries a probability outside [0, 1], on "
        "some of the rows the scorers would score, so there is no posterior to "
        "score the belief against on those rows"
    ),
}


#: The minimum number of assessable rows a
#: belief-scored cell needs before its ECE/TVD is admissible. Thirty is not a
#: new number: it is the 30+ non-null-row survival threshold the belief
#: properties use elsewhere,
#: reused here so one instrument does not carry two different ideas of "enough
#: rows to say something about beliefs". Below it the scorers still return a
#: number -- that is exactly the problem, and the reason this is a gate rather
#: than a caveat: a calibration verdict resting on a handful of rows reads
#: identically to one resting on thousands. `elicited_null` and
#: `assessable_frac` stay census: they describe coverage, this decides
#: admissibility.
BELIEF_MIN_ASSESSABLE_ROWS = 30

#: The floor's blocking clause. Formatted rather than static because the count
#: is the whole content: "too few rows" without n sends a reader back to the
#: parquet to find out how few.
_ASSESSABLE_FLOOR_SENTENCE = (
    "fewer than {floor} assessable rows (n={n}), which is the belief "
    "properties' own 30-row survival threshold: the scorers return a number "
    "here, and it would rest on too few rows to carry a verdict"
)


# What a FAILED scorability check says on a family
# whose scorer does not refuse on it. `true_support_within_stated` is measured
# for every non-darkhex family but gates only goofspiel: elsewhere
# `score_property5`'s `for card_key in stated` loop zero-pads to stated and
# returns a number. Un-gated is not un-reported: a
# quantity that is measured, stored in `properties.csv`'s underlying dict and
# rendered nowhere is measured-then-invisible, the same defect class
# as stale caveat literals. So the failed check is
# published as a census clause instead, worded so it cannot be mistaken for a
# refusal.
#
# Deliberately a separate table from `_DEFECT_SENTENCES`, not an entry in it:
# `_DEFECT_SENTENCES` is the census's rate vocabulary and every sentence there
# carries `{pct}`, while these are yes/no preconditions. Putting a boolean into
# that table would send it through `_render_defect`, where `False == 0.0` reads
# as "defect absent" and the clause would silently vanish -- the exact polarity
# trap `_render_scorability` exists to avoid. The two key sets are disjoint.
_SCORABILITY_CENSUS_SENTENCES = {
    "true_support_within_stated": (
        "true_posterior carries mass on keys the elicited belief does not "
        "state; the scorer zero-pads to stated on this family rather than "
        "refusing, so this describes the rows and does not withhold the cell"
    ),
}


def _render_scorability(key, value, game) -> str:
    """One scorability key's clause: `""` if the precondition holds.

    The polarity is the opposite of `_render_defect`'s, which is
    exactly why this is a separate function rather than a branch inside it: a
    census key blocks when it is non-zero, a scorability key blocks when it is
    False, and `False == 0.0` in Python, so one shared renderer would read
    every failed check as a clean one.

    Three outcomes, mirroring `_render_defect`'s:

    * `True` (or a positive `assessable_rows`) -> `""`. The scorer can score
      this cell on this question.
    * `False` (or `assessable_rows == 0`) -> the refusal sentence.
    * `None` -> a NON-empty clause naming why the question could not be asked,
      with the reason resolved per key and per family
      (`_not_assessable_reason`). No silent fallbacks: "not
      asked" is never "passed".
    """
    if value is None:
        return (
            f"{key} not assessable on this cell "
            f"({_not_assessable_reason(key, game)})"
        )
    if key == "assessable_rows":
        # Three outcomes, not two: none at all (nothing to score),
        # too few (a number the scorers will return, resting on too little),
        # and enough.
        if value == 0:
            return _SCORABILITY_SENTENCES[key]
        if value < BELIEF_MIN_ASSESSABLE_ROWS:
            return _ASSESSABLE_FLOOR_SENTENCE.format(
                floor=BELIEF_MIN_ASSESSABLE_ROWS, n=value
            )
        return ""
    if value:
        return ""
    from lmopt.games.families import GAME_FAMILIES

    if GAME_FAMILIES.get(game, game) in _NO_SIMPLEX_PAIR_FAMILIES:
        override = _NO_SIMPLEX_SCORABILITY_SENTENCES.get(key)
        if override:
            return override
    return _SCORABILITY_SENTENCES[key]


def _render_defect(defect_key, frac, template, reason) -> str:
    """One defect's clause: `""` if absent, a sentence if present or unmeasured.

    Shared by every caller so the
    primary defect and the additional ones cannot drift into two different
    renderings of "measured 0.0", "measured positive" and "not measurable".
    Every judgement about what a `frac` means lives here; the caller decides
    only which keys to ask about and in what order.

    `reason` is passed in rather than looked up here, because it
    depends on the cell's game as well as on the key (`_not_assessable_reason`)
    and this function is deliberately game-blind.
    """
    if frac is None:
        # The reason is looked up PER key (and per family) by
        # `_not_assessable_reason`, deliberately
        # with `[]` rather than `.get(..., <default>)`: a defect key with no
        # recorded reason is a bug in this file and must stop the run, not
        # render a plausible-sounding sentence that is false.
        return f"{defect_key} not assessable on this cell ({reason})"
    if frac == 0.0:
        return ""
    # A present defect must never print as "0.0% of rows".
    # `.1f` alone rounds every rate below 0.05% to "0.0%", and one null belief
    # among the 35,981 liarsdice LM rows of a real arm is 0.00278% -- so the
    # caveat would assert a zero rate for a defect that is there, and
    # would read byte-identically to what the deleted-`if frac == 0.0` mutant
    # emits.
    # Admissibility is unaffected either way (the string is non-empty, so the
    # cell stays blocked); the wrong figure in properties.csv/decision.csv is
    # the defect. Note this branch sits after the `frac == 0.0` return above,
    # so a genuinely absent defect still renders the empty caveat that lets
    # `emit()` promote the cell -- "<0.1%" never stands in for "none".
    pct = frac * 100
    return template.format(pct="<0.1%" if pct < 0.05 else f"{pct:.1f}%")


def measure_belief_defects(df, game) -> BeliefDefects:
    """Recompute, from this arm's rows, each quantity `GAME_CAVEATS` asserts.

    Returns two groups of keys, and the split is the point:

    * census (`_BELIEF_CENSUS_KEYS`): what this cell's beliefs look like.
      Reported, published, never blocking. The two-sided members are computed
      on the scorer's aligned pair, so the census describes the representation
      `score_property5`/`score_property7` actually compare;
      `key_set_mismatch_raw` keeps the raw unaligned comparison for continuity
      with earlier records. `elicited_equals_posterior` is decomposed into
      `identity_trivial_prior` (identity at `node_idx == 0` on a posterior
      uniform over its positive-mass support: the model restating the trivial
      prior, which is the correct answer there), `identity_zero_sum` (identity
      on a zero-sum posterior) and `identity_tracking` (every other identity,
      including single-support). The three are disjoint by construction and sum
      to `elicited_equals_posterior` on every cell.
    * SCORABILITY (`_BELIEF_SCORABILITY_KEYS`): whether the scorer can produce
      a number at all. `belief_scorability_gates(game)` says which of
      them the scorer actually refuses on, per family, and
      `derive_game_block` is the gate.

    `game` resolves the family, because
    alignment is per family and this function never re-implements it: it calls
    `leduc_rank.marginalise_to_rank` and
    `_belief_alignment.assert_true_support_within_stated`, the same helpers
    `property5.py` and `property7.py` call. On the darkhex family there is no
    aligned pair to compare at all (belief is per-cell occupancy, not a
    simplex, and `_belief_alignment.game_row_tvds` raises), so every
    family-scoped key reads `None` with the family reason
    while `elicited_null`, `assessable_frac` and `posterior_zero_sum` are still
    measured.

    `GAME_CAVEATS`' percentages were measured on one pre-repair arm; re-emitted
    verbatim they would hide any belief repair from the verdict layer. This
    makes the static table a template for which defect to check; the presence
    and the magnitude come from data.

    `df` is one cell's rows (the same frame `property_table` already
    slices). Rates are computed over LM rows only (`prompt` non-null) --
    non-LM (e.g. CFR reference) rows never carry elicited beliefs and must
    not dilute any denominator (scoring is offline: a pure
    function over the logged Parquet, nothing here contacts an endpoint).

    Three denominators, and the distinction is load-bearing:

    * `elicited_null` is denominated on ALL LM rows. It is a statement
      about rows -- "the model returned no belief here" -- so a row with a
      null belief is precisely the thing being counted, not an exclusion.
    * `posterior_zero_sum` depends on `true_posterior` alone -- it asks
      whether the posterior itself sums to zero, nothing about the
      elicited belief. Its denominator is rows where `true_posterior` is
      non-null and parses, full stop. Gating it behind `elicited_belief`
      being non-null is a category error: a row with a null elicited belief but a present,
      parseable, zero-summing posterior is a real positive for this
      quantity, not an exclusion.
    * The two-sided keys each compare `elicited_belief` against
      `true_posterior`, so both must be present and parse before either row
      can be judged at all. Assessable rows -- `elicited_belief` and
      `true_posterior` both non-null and both parsing as JSON -- is the
      denominator of `key_set_mismatch_raw` alone, and `assessable_frac`
      reports that count / LM rows. It is not `posterior_zero_sum`'s own
      denominator, which can be larger (see the pinned test below).
    * Every other two-sided key (`key_set_mismatch`,
      `support_mismatch`, `elicited_equals_posterior` and its three
      `identity_*` parts) is denominated on aligned rows: assessable rows
      that also have a scorer representation, i.e. the family transform ran
      and -- on goofspiel, the one family whose scorer asserts it -- the
      support precondition holds. That count is <= assessable and strictly
      smaller whenever a goofspiel row is refused, so `assessable_frac` does
      not report these rates' coverage: `aligned_frac` (aligned / LM rows)
      does, and is the census key to read beside them. Both coverage figures
      are reported for the same reason: a rate whose denominator a reader
      cannot see is a rate they cannot size.
    * The SCORABILITY judgements are all denominated
      on assessable rows, `posterior_well_formed` included: denominated on
      every LM row carrying a `true_posterior`, a row the scorers skip (null
      or unparseable elicited belief) could block a cell whose scored rows
      were clean. The
      census key `posterior_zero_sum` keeps the broader posterior-only
      denominator, which is why the two can disagree on the same cell: a
      positive census rate with a clean gate means every zero-sum posterior
      sits on a row no scorer reads.

    The hardcoded `GAME_CAVEATS` literals were authored on these
    same denominators: leduc asserts "support mismatch on 100% of rows",
    which is exactly 1.0000 over assessable rows and 0.9257 over all LM
    rows on the arm it was measured on. An all-rows denominator both
    misreproduces the literals and produces the tell-tale identity
    `elicited_null + key_set_mismatch == 1.0` -- the signature of gating a
    posterior-only quantity behind a non-null elicited belief.

    Returns None, never 0.0, for a quantity that cannot be computed
    (no silent fallbacks; "measured zero vs unmeasurable" is a
    recurring defect class). No LM rows
    at all => every key None, `assessable_rows` included (there is no cell to
    count). LM rows but zero rows with a parseable `true_posterior` =>
    `posterior_zero_sum` is also None (it has nothing to measure). LM rows and
    a parseable posterior somewhere, but zero assessable rows (e.g. every
    elicited belief is null) => `elicited_null`, `assessable_frac`,
    `posterior_zero_sum` and `assessable_rows` (an honest 0) are still real
    values; the two-sided quantities and the two alignment judgements read
    None. Neither case raises: a legitimate non-LM cell must not crash the run.

    Raises KeyError if a required column is missing (no silent
    fallbacks): a caveat that cannot be recomputed must stop the
    run, not silently degrade to the stale literal or to "no defect".
    """
    import polars as pl
    import json

    for col in _BELIEF_DEFECT_COLUMNS:
        if col not in df.columns:
            raise KeyError(
                f"measure_belief_defects requires column {col!r}; "
                f"present: {sorted(df.columns)}"
            )
    # The two scorer helpers, imported, never re-implemented
    # (scorers are consumed, never rebuilt). `marginalise_to_rank` is the
    # same call `property5.py`'s `family == "leduc"` branch makes;
    # `assert_true_support_within_stated` is the same precondition both
    # scorers assert for goofspiel, wrapped here to a per-row boolean because
    # a census must report a refusal, not propagate it.
    from lmopt.games.families import GAME_FAMILIES
    from lmopt.scoring._belief_alignment import assert_true_support_within_stated
    from lmopt.scoring.leduc_rank import marginalise_to_rank

    family = GAME_FAMILIES.get(game, game)
    aligned_pair_exists = family not in _NO_SIMPLEX_PAIR_FAMILIES
    lm = df.filter(pl.col("prompt").is_not_null())
    n = lm.height
    if n == 0:
        return dict.fromkeys(_BELIEF_DEFECT_KEYS)

    eb = lm["elicited_belief"].to_list()
    tp = lm["true_posterior"].to_list()
    ni = lm["node_idx"].to_list()
    null = assessable = 0
    zero = posterior_ok = zero_assessable = 0
    posterior_present = posterior_bad = 0
    malformed = malformed_assessable = 0
    n_aligned = align_failed = support_failed = 0
    # Keyed accumulation. Binding counters to names by position (a
    # `dict(zip(keys, counters))`) would silently re-label every one of them
    # when a quantity is added.
    counts = dict.fromkeys(_BELIEF_DEFECT_TWO_SIDED_KEYS, 0)
    for a, b, node_idx in zip(eb, tp, ni, strict=True):
        if a is None:
            null += 1

        # posterior_zero_sum depends on `true_posterior` alone -- parse and
        # score it before any check on `a`, so a null elicited belief never
        # excludes a row this quantity could otherwise measure.
        B = None
        b_zero_sum = False
        b_malformed = False
        if b is not None:
            posterior_present += 1
            try:
                B = json.loads(b)
            except (ValueError, TypeError):
                B = None
            if B is None:
                posterior_bad += 1
        if B is not None:
            posterior_ok += 1
            if abs(sum(B.values())) < 1e-12:
                zero += 1
                b_zero_sum = True
            # Well-formedness of the hypothesis map itself, independent of what
            # it sums to: an empty map names no hypothesis, and a probability
            # outside [0, 1] is not one.
            if not B or any(not (0.0 <= float(v) <= 1.0) for v in B.values()):
                malformed += 1
                b_malformed = True

        if a is None or B is None:
            continue
        try:
            A = json.loads(a)
        except (ValueError, TypeError):
            continue
        assessable += 1
        if b_zero_sum:
            # The assessable-row zero-sum count, which
            # is what `posterior_well_formed` is denominated on: the scorers
            # skip a row whose elicited belief is null or unparseable
            # (`score_property5`'s `calibration_rows` filter and its
            # `except (json.JSONDecodeError, TypeError): continue`), so a
            # zero-sum posterior on such a row is outside the population the
            # gate speaks for and must not withhold the cell.
            zero_assessable += 1
        if b_malformed:
            malformed_assessable += 1
        if not aligned_pair_exists:
            # darkhex: the row is assessable (both columns parse) but there is
            # no aligned representation to compare, so it contributes to
            # `assessable_frac`/`assessable_rows` and to nothing two-sided.
            continue

        # The raw comparison, kept for continuity with earlier records:
        # leduc reads 1.0 here on every arm and 0.0 aligned.
        if set(A) != set(B):
            counts["key_set_mismatch_raw"] += 1

        # The family transform, in the scorers' own order: leduc marginalises
        # the posterior to rank space first, so the support precondition is
        # asked of the representation that is actually compared.
        T = B
        if family == "leduc":
            try:
                T = marginalise_to_rank(T)
            except (ValueError, TypeError, KeyError):
                # A posterior whose keys are not card indices cannot be put
                # into the scorer's representation at all.
                align_failed += 1
                continue
        try:
            assert_true_support_within_stated(
                A, T, scorer_name="measure_belief_defects"
            )
        except ValueError:
            # Measured for every family, excluding only goofspiel's rows, and
            # the asymmetry is the scorers': `score_property5`'s `for card_key
            # in stated` loop applies the zero-pad-to-stated convention to
            # every family, so a violation always means posterior mass the
            # scorer silently discards -- but only the goofspiel branch
            # Asserts it, so only there does the scorer refuse the row and
            # leave no aligned pair for the census to describe. Elsewhere the
            # row still has an aligned pair (the raw one for kuhn/liarsdice,
            # the rank-marginalised one for leduc) and stays in the census,
            # while the scorability key records the discard.
            support_failed += 1
            if family == "goofspiel":
                align_failed += 1
                continue
        if family == "goofspiel":
            # The zero-pad-to-stated convention, legitimate only because the
            # precondition above holds (`_belief_alignment`'s docstring).
            T = {k: T.get(k, 0.0) for k in A}

        n_aligned += 1
        if set(A) != set(T):
            counts["key_set_mismatch"] += 1
        if {k for k, v in A.items() if v > 0} != {k for k, v in T.items() if v > 0}:
            counts["support_mismatch"] += 1
        if A == T:
            counts["elicited_equals_posterior"] += 1
            positive = [v for v in T.values() if v > 0]
            if abs(sum(T.values())) < 1e-12:
                counts["identity_zero_sum"] += 1
            elif (
                node_idx == 0
                and len(positive) >= 2
                and max(positive) - min(positive) < 1e-9
            ):
                # The trivial prior at the first node of the hand: uniform
                # over the cards still in play, which is the correct answer
                # there and evidence of nothing about belief tracking.
                # `node_idx` is the ALL-players ordinal within the hand
                # (`logging/schema.py`: "0-indexed within the hand"), not the
                # acting agent's own: a seat-1 LM's first decision sits at
                # `node_idx >= 1`, so a trivial-looking identity there
                # classifies as tracking. That is the rule as specified and is
                # deliberately not softened here; it reads 0 on the
                # random-seat sdv3 arms, and `identity_tracking > 0` is exactly
                # the recorded falsification condition that re-opens the question.
                counts["identity_trivial_prior"] += 1
            else:
                counts["identity_tracking"] += 1

    out: BeliefDefects = {
        "elicited_null": null / n,
        "assessable_frac": assessable / n,
        # The two-sided rates' own coverage, on the same LM-row denominator as
        # `assessable_frac` so the two are directly comparable; `None` on a
        # family with no aligned representation at all.
        "aligned_frac": (n_aligned / n) if aligned_pair_exists else None,
        "posterior_zero_sum": (zero / posterior_ok) if posterior_ok else None,
        "posterior_unparseable": (
            (posterior_bad / posterior_present) if posterior_present else None
        ),
        "posterior_cells_malformed": (
            (malformed / posterior_ok) if posterior_ok else None
        ),
    }
    for key in _BELIEF_DEFECT_TWO_SIDED_KEYS:
        if not aligned_pair_exists:
            out[key] = None
        elif key == "key_set_mismatch_raw":
            out[key] = counts[key] / assessable if assessable else None
        else:
            out[key] = counts[key] / n_aligned if n_aligned else None
    unasked = not aligned_pair_exists or assessable == 0
    out["alignment_ok"] = None if unasked else align_failed == 0
    out["true_support_within_stated"] = None if unasked else support_failed == 0
    # Denominated on assessable rows -- the population
    # the scorers score -- not on every LM row that carries a posterior. On
    # the broader denominator one row with a null `elicited_belief` and a
    # zero-sum `true_posterior` would withhold a cell whose scored rows are all
    # clean: the gate blocking on a row the scorer never reads. The
    # "parseable" half of the check is vacuous on this denominator by
    # construction (a row is assessable only once both columns parsed); that
    # is not a loss, because an unparseable posterior is skipped by the
    # scorers too (`score_property5`'s `except (json.JSONDecodeError,
    # TypeError): continue`) and is reported by `assessable_frac`'s coverage.
    # `None` iff no assessable rows, where `assessable_rows` is the one honest
    # reason and this question was never asked.
    out["posterior_well_formed"] = (
        None
        if assessable == 0
        else (
            malformed_assessable == 0
            if not aligned_pair_exists
            else zero_assessable == 0
        )
    )
    out["assessable_rows"] = assessable
    out["state_sensitivity_cell_b_clean"] = _state_sensitivity_cell_b_clean(df, lm)
    return out


def state_sensitivity_within_seat_reason(df) -> str | None:
    """Why `state_sensitivity_cell_b_clean` returned `None` for this cell's
    `df`, or `None` when it did not (nothing to explain).

    `state_sensitivity_cell_b_clean`
    returns `None` for six distinct causes, not one -- five inside
    `state_sensitivity()` plus this function's own missing-base-columns guard
    below, mirrored exactly so the two cannot silently diverge (a pinned test
    asserts `(cell_b_clean is None) == (this return is not None)`). Only
    `_NO_WITHIN_SEAT_BIN` ("bins and seat_bins exist but none is multistate")
    is genuinely "too few distinct trajectories" -- the other five are schema
    or collection defects (a missing column, an unseated frame, zero LM rows,
    every LM row dropped as a harness fallback) that must never be folded
    into an approved partial-cell reason.

    Deliberately not a key inside `measure_belief_defects`'s return dict:
    that dict's closed-set contract requires
    every key to hold a bool/float, never a string, so a string-valued
    diagnostic is threaded as its own separate fact instead, computed at the
    one call site that already has `df` in scope and passed down explicitly.

    :param df: this cell's own belief-scored dataframe, the same `df`
        `measure_belief_defects(df, game)` was called on
    :returns: the six-cause reason string, or `None` when
        `state_sensitivity_cell_b_clean` is not `None`
    """
    if any(col not in df.columns for col in _STATE_SENSITIVITY_CELL_B_COLUMNS):
        return "missing_base_columns"
    from lmopt.analysis import state_sensitivity

    return state_sensitivity.within_seat_reason(df)


def _state_sensitivity_cell_b_clean(df, lm) -> bool | None:
    """The state-sensitivity admissibility gate: does this cell contain an S7 cell-(b) bin?

    `False` iff at
    least one (opponent-scoped, this function runs per opponent already)
    legal-mask bin is both action-constant within a seat that genuinely held
    >= 2 distinct prompts (`state_sensitivity.qualifying_masks_action_constant`) and
    labelled "flat" by the belief-tracking axis (computed here, pooled across
    seats -- a deliberate, separate choice from the seat-aware axis above).

    `None`: when `df` carries no
    `lm_seat` column, or when no legal-mask bin held >= 2 distinct prompts
    within any single seat at all -- mirrors
    `state_sensitivity.within_seat_reason`'s own convention (its sibling
    `share_within_seat` statistic is `None` for exactly these same two
    causes, `src/lmopt/analysis/state_sensitivity.py:314-334`): the
    state-tracking question was never put to the model in this cell, so the
    gate cannot honestly say cell-(b) was ruled out. On collected M0 data this
    flips 8 of 12 goofspiel (game, opponent) cells
    from admissible to inadmissible for P5/P7 relative to a vacuous-`True`
    reading; kuhn/leduc/liarsdice have zero such cells so are
    unaffected either way.

    `True` when at least one within-seat qualifying bin exists (whether or
    not it happens to be action-constant) and none of the action-constant
    ones among them is "flat" -- a real, non-vacuous null finding over the
    bins that were examined, not an assertion about bins that were not.

    `df` is the same per-cell frame `measure_belief_defects` was called
    with (unfiltered); `lm` is its already-computed LM-rows-only frame, so
    this never re-filters what the caller already has. The two are not
    independently substitutable: `lm` must be `df` filtered to
    `prompt.is_not_null()` (the only call site, `measure_belief_defects`
    above, already guarantees this by construction).

    `None`, not a raise, when `df` lacks `legal_mask`/`info_set`/
    `parsed_action` (`_STATE_SENSITIVITY_CELL_B_COLUMNS`): those columns
    are irrelevant to every other key `measure_belief_defects` computes,
    so their absence is this key's own "unmeasurable", not a reason to
    fail the whole cell (no silent fallbacks: this is the
    honest `None`, not a swallowed error; a genuinely malformed real arm
    missing a base schema column is a defect worth finding through
    `elicited_null`/`assessable_frac` etc. still being computed and
    published, not through an opaque whole-cell crash).
    """
    if any(col not in df.columns for col in _STATE_SENSITIVITY_CELL_B_COLUMNS):
        return None
    from lmopt.analysis import state_sensitivity

    if state_sensitivity.within_seat_reason(df) is not None:
        return None
    qualifying = state_sensitivity.qualifying_masks_action_constant(df)
    labels = _mask_belief_labels(lm)
    return not any(
        const and labels.get(mask) == "flat" for mask, const in qualifying.items()
    )


#: `_reachable_chance_node`'s own memo, keyed by (game, node_cap): a plain
#: dict rather than `functools.lru_cache`, matching this file's own
#: established local-import-per-function convention -- a decorator needs a
#: module-level import, which nothing else here does.
_REACHABLE_CHANCE_NODE_CACHE: dict[tuple[str, int], bool] = {}


def _reachable_chance_node(game: str, node_cap: int = 200_000) -> bool:
    """Does this game's tree hold a chance node an LM could ever reach?

    The half of the STRUCTURAL reason
    that is about the game, not the opponent. `pyspiel.GameType.chance_mode`
    is not this question -- it answers "can this game class ever have chance
    nodes", not "does this configured game have a reachable one". This
    project's Goofspiel (`points_order=descending`,
    a fixed initial deal shape) reports `chance_mode = EXPLICIT_STOCHASTIC` at
    the type level while a full tree walk finds zero reachable chance nodes --
    the type attribute would give the wrong answer for exactly the two
    games (goofspiel, darkhex) this needs to classify correctly.

    A real, bounded depth-first walk from `new_initial_state()`: returns
    `True` the moment any reachable state is a chance node, `False` if the
    whole tree (every real game this project's games are small enough for)
    is exhausted first, and `True` -- the conservative direction, never
    silently STRUCTURAL -- if `node_cap` is exceeded before either.
    Memoized per game key: called once per distinct game across a whole
    scoring run, not once per row.

    :param game: an exact `lmopt.games.loaders.load_game` key (not
        necessarily a family -- a depth variant may have a different chance
        structure than its base family, so the caller's own exact key is
        used, matching `state_sensitivity_cell_b_clean`'s own convention of
        reading `game` rather than `GAME_FAMILIES.get(game, game)`).
    :param node_cap: the walk budget; every game this project scores
        is small enough to exhaust well under this (kuhn,
        leduc, goofspiel, liarsdice and darkhex all complete in well under
        a second), so hitting it is itself informative -- it means a variant
        grew larger than anything measured so far, and treating that as
        "assume a chance node exists" is the same "no silent fallbacks"
        default this whole module follows elsewhere.
    :returns: whether a reachable chance node exists (or the cap was hit)
    """
    key = (game, node_cap)
    if key in _REACHABLE_CHANCE_NODE_CACHE:
        return _REACHABLE_CHANCE_NODE_CACHE[key]

    from lmopt.games.loaders import load_game

    g = load_game(game)
    stack = [g.new_initial_state()]
    seen = 0
    result = False
    while stack:
        state = stack.pop()
        seen += 1
        if seen > node_cap:
            result = True
            break
        if state.is_terminal():
            continue
        if state.is_chance_node():
            result = True
            break
        for action in state.legal_actions():
            stack.append(state.child(action))
    _REACHABLE_CHANCE_NODE_CACHE[key] = result
    return result


#: `RLSelfPlayAgent.deterministic_action` is deliberately not set on the class
#: (`src/lmopt/agents/rl_selfplay.py` forms the reference cache's `code_hash`, so editing it
#: invalidates every warmed CFR/RL reference and needs a scheduled rewarm). Without an
#: explicit exclusion here, `rl` would silently
#: inherit `SeatAware`'s `True` default -- a wrong, plausible-looking answer that could
#: falsely approve an `rl` cell as STRUCTURAL. Named so it fails loudly (`None`, "not yet
#: classified") rather than fabricating one [no silent fallbacks].
_DETERMINISM_NOT_YET_AVAILABLE = frozenset({"rl"})


def _opponent_deterministic(opponent: str) -> bool | None:
    """Can this opponent's `act()` ever return a different action on two
    calls with the identical state, over every reachable code path?

    A class-level, source-verified
    fact (`Agent.deterministic_action`, `src/lmopt/agents/base.py`), read
    through the same name -> class association the harness's own opponent
    factory uses (`_build_opponent`,
    `src/lmopt/orchestration/shard.py`), so this small mapping follows what
    the harness actually constructs. This is not a (game, opponent) -> reason table, which
    is deliberately avoided because it would be hand-tuned against
    the current board; this maps a name to a class (unavoidable -- something has
    to), and the class decides its own determinism fact by being read in
    full, once, not by the opponent's name.

    Returns `None`, never a guessed bool, for `_DETERMINISM_NOT_YET_AVAILABLE`
    (currently `rl` alone -- see that constant's own docstring) and for an
    opponent this project's harness does not recognise: the caller already
    has a genuinely-blocked cell in hand, and refusing to approve an
    unclassifiable opponent is the conservative direction [no silent
    fallbacks]. Only a real, known `True`/`False` fact ever reaches the
    STRUCTURAL branch.
    """
    if opponent in _DETERMINISM_NOT_YET_AVAILABLE:
        return None
    from lmopt.agents.base import SeatAware
    from lmopt.agents.bayesian import BayesianAgent
    from lmopt.agents.cfr_ref import CFRRefAgent
    from lmopt.agents.classical import AlwaysBetAgent

    opponent_class: type[SeatAware] | None = {
        "always_bet": AlwaysBetAgent,
        "bayesian": BayesianAgent,
        "cfr_ref": CFRRefAgent,
    }.get(opponent)
    if opponent_class is None:
        return None
    return opponent_class.deterministic_action


#: What `_state_sensitivity_cell_b_clean` returning `None` renders as, when
#: it is one of the three approved partial-cell reasons -- read by `_property_row` to both
#: rewrite the rendered caveat clause and set `partial_reason_class`.
STRUCTURAL_REASON = "no within-seat variation against this opponent"
THIN_REASON = "too few distinct trajectories to assess"
#: The fourth approved class: a P8 row whose arm played real rows and none reached an
#: information-buying node, so there is no value to score. Its recorded reason is "no
#: information-buying node reached on this board"; the row's own caveat (the zero-informative
#: clause, which says whether the zero is this arm's or the board's) is what the board renders.
#: Only the scorer's own zero-informative branch emits it (`_not_reached_row`).
NOT_REACHED_CLASS = "not_reached"


def _partial_reason_class(
    prop: str,
    game: str,
    opponent: str,
    measured: dict,
    game_block: list[str],
    within_seat_reason: str | None,
) -> tuple[str, str] | None:
    """Why this missing opponent is missing, as one of the three approved
    reasons, or `None` when it matches none of them and still blocks.

    A computation over what `measured` already
    holds for this exact (game, opponent, property) cell, never a lookup
    keyed by (game, opponent) or by opponent name alone -- the two facts
    that decide "structural" versus "thin" (`_opponent_deterministic`,
    `_reachable_chance_node`) are each independently verifiable against the
    real agent classes and the real game object, not asserted here.

    :param prop: this row's property name; only P5/P7-family properties
        (`_BELIEF_SCORED_PROPERTIES`) carry the `state_sensitivity_cell_b_clean`
        gate at all, so every other property returns `None` unconditionally.
    :param game: the exact game key (not necessarily the family -- a depth
        variant may have a different chance-node structure than its base).
    :param opponent: this row's opponent name.
    :param measured: this cell's `measure_belief_defects` output.
    :param game_block: this (prop, game) cell's `derive_game_block(...)`
        result -- when it holds more than the one `state_sensitivity_cell_b_clean`
        reason, some other defect also blocks this row, so the missing
        opponent is not solely explained by one of the three reasons and
        stays unclassified, even where `state_sensitivity_cell_b_clean`
        alone would have qualified.
    :param within_seat_reason: `state_sensitivity_within_seat_reason(df)`'s
        result for this same cell, computed by the caller (not read off
        `measured`, whose closed-set contract admits only bool/float values).
    :returns: `(class_tag, canonical_text)` where `class_tag` is one of
        `"structural"`/`"thin"`/`"failed"`, or `None` when unclassified
        (`cell_b_clean` is `True` -- nothing to explain --, `game_block` has
        other content, or the `None` cause is one of the five that are not
        "too few distinct trajectories")
    """
    if prop not in _BELIEF_SCORED_PROPERTIES:
        return None
    cell_b = measured.get("state_sensitivity_cell_b_clean")
    if cell_b is True or len(game_block) != 1:
        return None
    if cell_b is False:
        return ("failed", FAILED_REASON)
    from lmopt.analysis import state_sensitivity

    if within_seat_reason != state_sensitivity._NO_WITHIN_SEAT_BIN:
        return None
    deterministic = _opponent_deterministic(opponent)
    # `None` (not yet classifiable -- `rl`, or an unrecognised name) stays unclassified
    # rather than falling through to thin, which would assert a specific, unverified claim
    # ("too few distinct trajectories") about an opponent this function cannot yet judge.
    if deterministic is None:
        return None
    if deterministic and not _reachable_chance_node(game):
        return ("structural", STRUCTURAL_REASON)
    return ("thin", THIN_REASON)


def _mask_belief_labels(lm) -> dict[str, str]:
    """`legal_mask` -> "tracks"|"flat"|"excluded" (§8.1), pooled across seats.

    The belief-tracking axis stays pooled by deliberate
    choice, so this is
    kept separate from the seat-aware qualifying/
    action-constancy axis, which is shared via
    `state_sensitivity.qualifying_masks_action_constant`.
    """
    import polars as pl

    from lmopt.analysis.state_sensitivity import add_mask_key

    keyed = add_mask_key(lm)
    agg = keyed.group_by("_mask").agg(
        pl.col("true_posterior")
        .filter(pl.col("true_posterior").is_not_null())
        .n_unique()
        .alias("n_post"),
        pl.col("elicited_belief")
        .filter(pl.col("true_posterior").is_not_null())
        .drop_nulls()
        .n_unique()
        .alias("n_belief_on_post_rows"),
    )
    labels = {}
    for r in agg.iter_rows(named=True):
        if r["n_post"] >= 2:
            labels[r["_mask"]] = "tracks" if r["n_belief_on_post_rows"] >= 2 else "flat"
        else:
            labels[r["_mask"]] = "excluded"
    return labels


def belief_scorability_gates(game) -> tuple[str, ...]:
    """Which scorability keys gate a belief property on `game`'s family.

    The gate is the SCORER's own refusal
    set, per family, not a hand list of exceptions and not the whole
    scorability tuple. Established by running `score_property5`
    and `score_property7` on synthetic one-row cells, and exercised against
    the real scorers rather than a restatement of them:

    * darkhex family -> all five but `alignment_ok` and
      `true_support_within_stated`. The scorers score this family, with a
      per-cell estimand, so it reaches the per-key loop like any other.
      `alignment_ok` is NOT APPLICABLE, not withheld: it reports whether the
      scorers' simplex alignment step ran and this family has none, while a
      stated cell set that differs from the true one is the model's failure,
      reported as a rate beside the number rather than gated.
      `true_support_within_stated` is excluded by the same rule that excludes
      it for kuhn/leduc/liarsdice: the scorer does not assert it on this
      family. `posterior_well_formed` is asked, on this family's own question
      (see `_NO_SIMPLEX_SCORABILITY_SENTENCES`).
    * goofspiel -> all five. `assert_true_support_within_stated` raises out of
      both scorers on a support violation, so `true_support_within_stated` is
      a refusal here.
    * kuhn / leduc / liarsdice -> all five but `true_support_within_stated`. The
      same violation on leduc returns ECE 0.5 (the align-to-stated convention
      is applied silently), so the quantity is a census fact, not a refusal.
      The scoping rests on that asymmetry alone: after rank marginalisation
      the check passes on real leduc, and reads True on all 64 non-darkhex
      cells of the four pilot arms. Falsification condition: a False
      reading on any scored non-goofspiel cell re-opens whether the scorers'
      silent drop of out-of-support posterior mass is acceptable (a
      scorer-semantics question, not a change to this gate).

    Documented divergence from the scorers, kept deliberately:
    `posterior_well_formed` gates every family, but a zero-sum
    posterior does not make the scorers refuse -- kuhn and liarsdice cells
    whose `true_posterior` sums to zero return ECE 0.5 and TVD 0.5. The gate is
    stricter than the scorer here on purpose: a calibration
    error against a non-distribution is a defined number about the wrong
    estimand. The divergence is conservative-only (it can block, never promote)
    and reads False on no non-darkhex cell of any pilot arm,
    darkhex being the only family that carries zero-sum posteriors
    at all (0.3696 of posterior-carrying rows, arm-pooled over
    the four sdv3 qwen darkhex cells). A family whose belief is not a simplex
    is asked a different question by this key, computed rather than looked up:
    every scored row's posterior must name at least one cell and keep every
    probability in [0, 1]. Summing to zero is a legitimate answer there -- the
    opponent holds none of the unseen cells -- so it does not fail the key.

    Falsification condition for the divergence: the "measured 0
    on every non-darkhex pilot cell" figure above is a licence not to revisit
    the divergence, so it carries the condition that voids it.
    `posterior_well_formed` reading False on any cell the scorers score
    re-opens the decision -- the cell is one the scorer would have handed a
    number and the gate withheld, which is exactly the trade the divergence
    asserts is empty. The condition is also carried in the blocker text a reader of
    `decision.csv` meets (`_SCORABILITY_SENTENCES`), because the person who
    would notice the breach reads that column, not this docstring.

    The key is denominated on assessable rows only, so an
    out-of-population row (null or unparseable elicited belief) cannot
    trip it; see `measure_belief_defects`.
    """
    from lmopt.games.families import GAME_FAMILIES

    family = GAME_FAMILIES.get(game, game)
    if family in _BELIEF_NOT_APPLICABLE_FAMILIES:
        return ()
    no_simplex = family in _NO_SIMPLEX_PAIR_FAMILIES
    return tuple(
        key
        for key in _BELIEF_GATE_ORDER
        if (
            key != "true_support_within_stated"
            or family in _SCORER_ASSERTS_SUPPORT_FAMILIES
        )
        # `alignment_ok` reports whether the scorers' simplex alignment step
        # ran, and a family without one has no such step to report on; a
        # stated cell set that differs from the true one is the model's
        # failure, reported as a rate beside the number, never a gate.
        and (key != "alignment_ok" or not no_simplex)
    )


def derive_game_block(prop, game, measured) -> list[str]:
    """The reasons this (prop, game) cell cannot be scored. Empty = scorable.

    This is the admissibility gate, and it
    is the whole gate: `emit()` demotes a row iff this list is non-empty. It
    asks exactly one question -- can the scorer produce a number on this cell
    -- which is the only question an admissibility gate is entitled to ask of a
    belief column. The census quantities (`derive_game_census` below) are
    reported in the same caveat and never appear here.

    Returned as a list rather than sniffed out of the caveat string, so
    `decision_table`'s `caveat.split(";")[0]` contract stays honest by
    construction: when this list is non-empty its first element is the first
    clause of the caveat, and when it is empty the row is MEASURABLE and never
    reaches `decision_table`'s `blockers` at all.

    Three shapes:

    * a NON-belief property -> `[]`. A belief defect is not evidence about
      P3's KL or P2's best-response gap.
    * a belief property on a family the scorers declare NOT_APPLICABLE ->
      the family reason, alone, with `property_table` diverting those rows to
      an explicit NOT_APPLICABLE status before `emit()` sees them. No family
      is in that state (`_BELIEF_NOT_APPLICABLE_FAMILIES` is empty because
      darkhex has its per-cell estimand), so this branch is dormant; it is
      kept because a family that genuinely has no belief property needs it and
      should need no second mechanism.
    * otherwise -> one clause per FAILED scorability check, in
      `belief_scorability_gates(game)` order (root cause first).

    The `assessable_rows` short-circuit is not an optimisation: with no rows,
    the other three belief-parsing questions were not answered because there
    was nothing to answer them about, and rendering their three `None`
    clauses beside it would publish one defect as four. It fires on the empty
    cell only; a cell blocked by the assessable-row floor still publishes whatever
    else failed, because those questions were answered.

    `state_sensitivity_cell_b_clean`
    is exempt from the short-circuit, and is checked separately below rather
    than through the loop above: unlike the other three, it is denominated on
    the ACTION/SEAT axis, not on assessable (belief-parsed) rows, so
    `assessable_rows == 0` does not make it unanswerable -- a cell can have
    zero belief-assessable rows and still carry a real, independently
    computed cell-(b) finding. Silently dropping that finding behind the
    belief-axis short-circuit would not change whether the cell is
    INADMISSIBLE (assessable_rows == 0 already blocks it), but it would
    understate why, which the "no silent fallbacks" invariant does not permit
    for a quantity that genuinely was computed.

    `assessable_rows` blocks at two levels: zero rows
    (nothing to score) and fewer than `BELIEF_MIN_ASSESSABLE_ROWS` (a number
    the scorers return, resting on too few rows to carry a verdict).
    """
    from lmopt.games.families import GAME_FAMILIES

    if prop not in _BELIEF_SCORED_PROPERTIES:
        return []
    if measured.get(_CENSUS_ERROR_KEY):
        # The census could not be computed
        # at all, so every key is `None` for one reason and the per-key
        # "not assessable" sentences would each assert a different, invented
        # one -- for example "assessable_rows not assessable on
        # this cell (no LM rows in this cell)" on a cell that has LM rows and
        # is missing a column. One reason, the real one, verbatim.
        return [f"belief census unavailable: {measured[_CENSUS_ERROR_KEY]}"]
    if GAME_FAMILIES.get(game, game) in _BELIEF_NOT_APPLICABLE_FAMILIES:
        return [_BELIEF_NOT_APPLICABLE_REASON]
    gates = belief_scorability_gates(game)
    reasons = []
    for key in gates:
        if key == "state_sensitivity_cell_b_clean":
            # Checked once, after this loop, unconditionally -- see the
            # docstring's "EXEMPT from the short-circuit" note.
            continue
        # `measured[key]`, strict, never `.get`: the gate set is derived from
        # `measure_belief_defects`' own key set, so a missing key is a bug in
        # this file and must stop the run rather than read as "passed"
        # (no silent fallbacks).
        clause = _render_scorability(key, measured[key], game)
        if not clause:
            continue
        reasons.append(clause)
        if key == "assessable_rows" and measured[key] == 0:
            # The short-circuit is for the empty cell only: with no rows the
            # other three questions were not answered because there was
            # nothing to answer them about, and rendering their three `None`
            # clauses beside this one would publish one defect as four. A cell
            # that merely falls under the row floor has answers to them, and
            # withholding those answers would hide a real defect behind a
            # counting one.
            break
    if "state_sensitivity_cell_b_clean" in gates:
        clause = _render_scorability(
            "state_sensitivity_cell_b_clean",
            measured["state_sensitivity_cell_b_clean"],
            game,
        )
        if clause:
            reasons.append(clause)
    return reasons


def derive_game_census(prop, game, measured) -> list[str]:
    """What this cell's beliefs look like. Reported, never blocking.

    Each clause is prefixed `"census: "` so a reader
    of `properties.csv` can tell a description of the data from a reason the
    cell was withheld, and so no census clause can ever be mistaken for
    decision.csv's blocker (which is the caveat's first clause, and a census
    clause is first only when nothing blocks -- in which case the row is
    MEASURABLE and `decision_table` never reads it).

    `CAVEAT_TEMPLATES` has one job here: it decides which census
    clause leads for this (prop, game) and in whose words -- several entries
    carry a game-specific second half (leduc's "ECE scores only the stated
    subset") that would be wrong to repeat elsewhere. The rest follow in
    `_BELIEF_CENSUS_CLAUSE_KEYS` order, deduplicated BY KEY so the primary is
    never restated in the neutral `_DEFECT_SENTENCES` wording.

    Three outcomes per key, decided by `_render_defect`: measured exactly 0.0
    renders nothing; measured positive renders the sentence at this arm's own
    percentage; `None` renders an explicit "not assessable" clause naming the
    key's own denominator, never an empty string (no silent
    fallbacks).

    A key whose `None` is family-scoped is skipped entirely rather than
    rendered as unassessable: on darkhex there is no aligned belief/posterior
    pair to describe at all, so "key_set_mismatch not assessable on this cell"
    is not a census fact about the cell, it is a fact about the family, and
    `derive_game_block` already states it once.
    """
    from lmopt.games.families import GAME_FAMILIES

    if measured.get(_CENSUS_ERROR_KEY):
        # Nothing was measured, so there is nothing to describe; the single
        # blocking reason `derive_game_block` returns says why, once.
        return []

    scoped_out = GAME_FAMILIES.get(game, game) in _NO_SIMPLEX_PAIR_FAMILIES

    def _skip(key):
        return scoped_out and key in _BELIEF_FAMILY_SCOPED_KEYS

    clauses = []
    named = set()
    entry = CAVEAT_TEMPLATES.get((prop, game))
    if entry is not None:
        defect_key, template = entry
        named.add(defect_key)
        if not _skip(defect_key):
            # Deliberately `measured[defect_key]`, not `.get()`: a template
            # naming a quantity `measure_belief_defects` does not return is a
            # bug in this file, and must stop the run rather than read as "no
            # defect".
            clause = _render_defect(
                defect_key,
                measured[defect_key],
                template,
                _not_assessable_reason(defect_key, game),
            )
            if clause:
                clauses.append(f"census: {clause}")
    if prop in _BELIEF_SCORED_PROPERTIES:
        for key in _BELIEF_CENSUS_CLAUSE_KEYS:
            if key in named or _skip(key):
                continue
            clause = _render_defect(
                key,
                measured[key],
                _DEFECT_SENTENCES[key],
                _not_assessable_reason(key, game),
            )
            if clause:
                clauses.append(f"census: {clause}")
        # The scorability keys this family's scorer
        # does not refuse on. They are still measured, so a failure is a census
        # fact and must be published; on the families where the scorer does
        # refuse, the same failure is already stated once by
        # `derive_game_block` and is deliberately not restated here.
        #
        # `is False`, strictly, never a truthiness test: `None` means the
        # question was not asked (no assessable rows), which is not a census
        # fact about the rows, and on every such cell `assessable_rows` has
        # already blocked and named the real reason.
        if not scoped_out:
            gated = set(belief_scorability_gates(game))
            for key in _BELIEF_GATE_ORDER:
                if key in gated or key not in _SCORABILITY_CENSUS_SENTENCES:
                    continue
                if measured[key] is False:
                    clauses.append(f"census: {_SCORABILITY_CENSUS_SENTENCES[key]}")
    return clauses


def derive_game_caveat(prop, game, measured) -> str:
    """Caveat text for (prop, game) at this arm's measured magnitude.

    Blocking reasons first, then the
    census. That order is the contract `decision_table` depends on: it
    publishes `caveat.split(";")[0]` as decision.csv's blocker, so the first
    clause must be a reason the cell was withheld whenever it was withheld.
    Whether it was withheld is decided by `derive_game_block`, not by whether
    this string is empty -- a cell can carry a page of census text and still be
    perfectly MEASURABLE.

    Why the census does not gate. A gate of "every measured quantity except
    `assessable_frac`", i.e. the whole census, blocks leduc at
    `key_set_mismatch` 100 % over a rank-vs-index granularity
    difference the scorers reconcile themselves
    (`leduc_rank.marginalise_to_rank`; the aligned mismatch is 0.0), and
    blocks kuhn at `elicited_equals_posterior` on rows that are all
    `node_idx == 0` identities over a support-uniform posterior, i.e. the model
    correctly restating the trivial prior. Neither fact stops
    `score_property5`/`score_property7` producing a number, so neither is an
    admissibility question. Both remain measured, published and readable, as
    `census:` clauses.

    A NON-belief `(prop, game)` with no template returns `""` unchanged, and
    `measured` is not read at all.
    """
    return "; ".join(
        derive_game_block(prop, game, measured)
        + derive_game_census(prop, game, measured)
    )


def mask_constant_share(df) -> float | None:
    """Share of LM rows whose action is a constant function of `legal_mask`.

    1.0 means the logged action carries no information beyond which actions
    were legal: the private card, the board and the history are all
    irrelevant to it. Such a cell records a prompt-format response, not a
    policy over game states, and cannot support an optimality claim. The
    harness logs one row per decision node either way, so nothing else in
    properties.csv distinguishes the two -- this row is what makes the
    difference visible per cell instead of leaving it for a human to notice.

    Over one three-arm cycle, pooled by (arm, game) -- not this per-cell
    function's own unit: 496,659 / 661,644 = 75.1 % of
    all LM rows (Qwen 61.8 %, OLMo 71.8 %, Mistral 91.1 %). At the per-cell
    (game, opponent, mask) unit this function actually computes, the same
    cycle reads 512,945/661,644 = 0.7753; restricted to bins with >= 2
    distinct prompts, 269,902/398,406 = 0.6775. The three units
    are not interchangeable -- quoting one as another is a defect.

    Not a positional test, and the distinction is the whole point. A test of
    "share of rows whose action is the LOWEST-index legal
    action" reads 0.0000 for goofspiel, suggesting it is clean. Goofspiel
    is in fact 100 % constant-per-mask for two of the three arms; each arm
    simply maps the legal-set size to its own constant, which is not the
    first index (Qwen maps
    |A|=4 -> action 3, |A|=3 -> 1, |A|=2 -> 2; Mistral identically; OLMo
    4 -> 2, 3 -> 1, 2 -> {3, 0}). Grouping by the mask and asking whether the
    action ever varies inside a bin is mechanism-free: it catches any
    constant, whatever it happens to be.

    Denominated on rows, not on mask bins. The two agree only when every bin
    is the same size, and the headline figure above is a row count.

    Computed over LM rows only (`prompt` non-null), for the same reason
    `measure_belief_defects` is: an opponent's action under the same mask is
    not evidence about the LM's policy, and letting it into a bin can only
    add spurious variation (scoring is offline: a pure function
    over the logged Parquet, nothing here contacts an endpoint).

    A row whose action the harness substituted (`fallback_used`) is not the model's
    choice and is left out of both the numerator and the denominator; the count is
    reported beside the share (`state_sensitivity_` rows of properties.csv).

    Returns None, never 0.0, when there are no LM rows to measure
    (no silent fallbacks; "measured zero vs unmeasurable" is a
    recurring defect class): 0.0 is this statistic's CLEANEST possible
    reading ("no row's
    action was a constant function of its mask"), so returning it for a cell
    with nothing to measure would assert perfect state-sensitivity from an
    empty frame.

    Raises (polars ColumnNotFoundError) if `prompt`, `legal_mask` or
    `parsed_action` is absent, rather than degrading to a number: the caller
    in `property_table` catches it and emits the row with `value=None` and a
    caveat naming the raise.
    """
    import polars as pl

    from lmopt.analysis.state_sensitivity import add_mask_key, fallback_drop_report

    lm = fallback_drop_report(df).kept
    if lm.is_empty():
        return None
    keyed = add_mask_key(lm)
    bins = keyed.group_by("_mask").agg(
        pl.col("parsed_action").n_unique().alias("nu"), pl.len().alias("n")
    )
    constant = cast(
        int, bins.filter(pl.col("nu") == 1)["n"].sum() or 0
    )  # polars types .sum() as a wide union
    return constant / keyed.height


# The comparators `DECISION_RULE`'s `direction` field may name, and the sentence
# each one publishes in decision.csv's `direction` column. Kept as two tables
# with the same keys so a
# comparator can never ship without the sentence that tells a reader what it
# did, which is how a `gte` rule would otherwise be published as a `gt` one.
#
# `gte` belongs to `P5_calibration_per_pair_margin`, where the exact tie is
# reachable. `P2_adaptivity_delta` is the
# second sanctioned `gte` -- the pre-registered table's ">= 0.0 fails" decides
# the exact tie (delta == 0.0 is "no shrinkage"), so `gt` would mispublish it.
# `gte` is exclusive to exactly these two entries.
#
# `lte` belongs to `P3_randomisation_margin` alone, and is a separate
# comparator rather than a reuse of `lt` for the same reason `gte` is not
# `gt`: the P3 rule fails at the exact tie (a margin of 0.0 is "no better
# than random"), and `lt` would publish that tie as a pass. The two differ
# only at the tie, which is exactly the case the rule speaks to, so reaching
# for `lt` would be publishing one rule under another's name.
#
# "rule" belongs to a sentinel-rule row (`threshold=None`), of which
# `DECISION_RULE` carries exactly one: P6, whose threshold is per game and
# lives in the off-path sentinel anchor artefact, so no single number can sit
# in its column. It must never be called: `decision_table` branches on
# `threshold is None` before any comparison, and a None threshold reaching a
# comparator would publish a verdict under a rule with no number (no
# silent fallbacks) -- so the entry raises instead of comparing.
#
# "none" belongs to the rows that carry no threshold by design (P1, P4, P7a, P7b). It is never
# Called either: `decision_table` issues NO_THRESHOLD for such a row before any comparison.


# Minimum independent outcomes a cell needs before any interval on it is
# reported. Below this the cell is DEGENERATE: the policy pair is deterministic
# and the game supplied too little chance variation to make N hands informative.
MIN_DISTINCT_OUTCOMES = 2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data", required=True, help="Directory containing the run's Parquet files."
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output directory for the CSV tables and the report.",
    )
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Proceed even if --out already exists and is non-empty. Default "
        "refuses: the default --out convention "
        "derives only from the model's short name, with no epoch/rung/arm "
        "token, so a second run for a different epoch would otherwise "
        "silently overwrite the first's committed tables.",
    )
    ap.add_argument(
        "--cfr-iters",
        type=int,
        default=1000,
        help="CFR+ iterations for the reference (must match the sweep's).",
    )
    ap.add_argument(
        "--cfr-cache-dir",
        default="data/cfr_policies",
        help="Cache directory for solved CFR policies.",
    )
    ap.add_argument(
        "--require-cache-hit",
        action="store_true",
        help="Refuse a CFR cache miss instead of solving, so a cold cache fails up front "
        "rather than letting parallel runs each solve the same game.",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples for the reward interval (0 disables).",
    )
    ap.add_argument(
        "--skip-properties",
        action="store_true",
        help="Emit the outcome table only; skip the (slow) property scorers.",
    )
    ap.add_argument(
        "--exact-policy",
        default=None,
        help="Path to the exact per-info-set action-distribution CSV "
        "(produced by scripts/exact_policy_pass.py). Default: "
        "<data>/exact_policy.csv when the arm directory holds "
        "one, else <out>/exact_policy.csv. Feeds "
        "P3_randomisation_exact, a "
        "diagnostic reported beside P3_randomisation.",
    )
    ap.add_argument(
        "--p3-sentinel-anchors",
        default=DEFAULT_P3_SENTINEL_ANCHORS,
        help="Path to the P3 sentinel anchor artefact (one "
        "uniform-random regret anchor per (game, opponent)). "
        "Feeds P3_randomisation_margin, the row that carries "
        "P3's verdict. A cell with no anchor there, or whose "
        "anchor the run's own provenance voids, reads "
        "AWAITING_DATA with that reason.",
    )
    ap.add_argument(
        "--p2-hand-bins",
        type=int,
        default=None,
        help="BIN COUNT (not hands-per-bin!): number of equal-width "
        "hand-index bins for the windowed P2 trajectory, "
        "threaded to score_property2(n_hand_bins=...). "
        "Default (omitted): computed per cell as "
        "floor(fewest LM hands in any seed / licensed hands "
        "per bin), where the licensed hands per bin is "
        "smoke_test_licensed_w in configs/"
        "p2_variance_budget.json. A cell below two licensed "
        "bins gets an AWAITING_DATA delta row that says how "
        "many hands per seed it needs. 0 disables the "
        "trajectory: no trajectory or delta rows, the "
        "unwindowed artifact of record. MORE bins means FEWER "
        "hands per bin: the s2 budget's w numbers (e.g. "
        "w=1000) are in the OPPOSITE unit, hands per bin, so "
        "passing w straight through (--p2-hand-bins=1000) "
        "produces ~1 hand per bin. Convert with "
        "n_hand_bins = round(n_hands_per_seed / w). An "
        "explicit count whose implied hands-per-bin falls "
        "below the licensed floor is REFUSED when the "
        "artefact is present, and only a WARNING when it is "
        "absent or unreadable.",
    )
    return ap.parse_args(argv)


#: The repo this script lives in; `__file__`, never the cwd.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]


def _display_data_dir(data_dir: Path) -> str:
    """`data_dir` as REPO-relative when it is inside the repo, else absolute.

    `run_meta.json` is read back by the cross-arm loader, which resolves a
    relative value against the repo root. A cwd-relative string (`str(args.data)`)
    would resolve correctly only when the run started
    at the repo root, and silently point at a non-existent directory
    otherwise -- which `arm_provenance` reads as "source dir missing", i.e. a
    keepable arm reported as throwaway.
    """
    # `absolute()` before `resolve()`, and the unresolved form wins: `data/` in
    # a worktree can hold symlinks into another checkout's data directories,
    # so resolving first turns `data/pilot/...` into an
    # absolute path in another checkout and the committed run_meta.json ends up
    # naming one machine's home directory. The unresolved absolute path is
    # inside this checkout, which is what a reader can act on.
    absolute = Path(data_dir).absolute()
    for candidate in (absolute, absolute.resolve()):
        try:
            return candidate.relative_to(_SCRIPT_REPO_ROOT).as_posix()
        except ValueError:
            continue
    return str(absolute.resolve())


def load_cells(data_dir: Path):
    """Load every shard, grouped into {(game, opponent): [(seed, df), ...]}."""
    import polars as pl

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files found in {data_dir}")
    cells: dict[tuple[str, str], list] = {}
    skipped = []
    for f in files:
        cell = cell_of(f)
        if cell is None:
            skipped.append(f.name)
            continue
        game, opponent, seed = cell
        cells.setdefault((game, opponent), []).append((seed, pl.read_parquet(f)))
    if skipped:
        # Never silently ignore a shard: an unparseable name means a cell is
        # missing from the table, which is a finding, not a detail.
        print(
            f"WARNING: {len(skipped)} file(s) skipped, name not "
            f"'<game>_lm_vs_<opponent>_seed<N>.parquet': {skipped}",
            file=sys.stderr,
        )
    if not cells:
        raise SystemExit(
            f"No shard in {data_dir} parsed into a (game, opponent) cell, so there is "
            "nothing to analyse"
        )
    print(
        f"Loaded {len(files) - len(skipped)} shards into {len(cells)} "
        f"(game, opponent) cells from {data_dir}"
    )
    return cells


def row_prompt_schema_versions(cells) -> list[int]:
    """
    the distinct prompt schema versions the loaded LM rows carry
    :param cells: {(game, opponent): [(seed, DataFrame), ...]} as `load_cells` returns it
    :returns: the sorted distinct non-null `prompt_schema_version` values over LM rows (rows
        with a non-null `prompt`, the same scoping the arm-evidence gate uses); empty when no
        shard carries the column
    """
    import polars as pl

    versions: set[int] = set()
    for shards in cells.values():
        for _seed, frame in shards:
            if "prompt_schema_version" not in frame.columns:
                continue
            lm = (
                frame.filter(pl.col("prompt").is_not_null())
                if "prompt" in frame.columns
                else frame
            )
            versions.update(lm["prompt_schema_version"].drop_nulls().unique().to_list())
    return sorted(versions)


def configured_cells(data_dir: Path) -> set[tuple[str, str]] | None:
    """
    every (game, opponent) pair this arm's own run_meta.json says it was configured for

    `load_cells` above reads only what a shard glob found, so a configured (game,
    opponent) that lost its shard to a walltime kill mid-sweep is indistinguishable
    from one that was never configured at all -- both simply never appear in `cells`.
    `property_table` uses this set to tell the two apart, mirroring `probe_scores.py`'s
    `_configured`/`_missing_status` (the pattern P1/P4/P6 already use): a game the
    union of `composed_from` pieces' own `resolved_config.games` covers, at this arm's
    own `resolved_config.opponents`.
    :param data_dir: the arm directory `run_meta.json` names its provenance for
    :returns: {(game, opponent)}, or None when the games list cannot be read at all
        (no run_meta.json, or a composed piece whose own run_meta.json is unreadable) --
        None means "cannot tell a collection gap from a lost piece", never an empty set
    """
    meta_path = data_dir / "run_meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    resolved = meta.get("resolved_config") or {}
    opponents = resolved.get("opponents")
    if not opponents:
        return None
    pieces = meta.get("composed_from")
    if pieces:
        games: set[str] = set()
        for piece in pieces:
            piece_path = Path(piece)
            if not piece_path.is_absolute():
                piece_path = _SCRIPT_REPO_ROOT / piece_path
            try:
                piece_meta = json.loads((piece_path / "run_meta.json").read_text())
            except (OSError, ValueError):
                return None
            piece_games = (piece_meta.get("resolved_config") or {}).get("games")
            if not piece_games:
                return None
            games.update(piece_games)
    else:
        games = set(resolved.get("games") or ())
        if not games:
            return None
    return {(g, o) for g in games for o in opponents}


def hand_rewards(df):
    """One reward per hand, from the LM's own rows.

    Each logged row carries the reward of the player who acted, so a hand holds
    two values (the LM's and the opponent's). Taking the first LM row per hand
    is what makes this a per-hand outcome rather than a per-node sum.
    """
    import polars as pl

    lm = df.filter(pl.col("prompt").is_not_null())
    if lm.is_empty():
        return []
    per_hand = lm.group_by("hand_idx", maintain_order=True).agg(
        pl.col("reward").first().alias("r")
    )
    return per_hand["r"].to_list()


# `--p2-hand-bins` is a BIN COUNT
# (score_property2's own `n_hand_bins`), but the variance budget's
# W-justification reasons in the OPPOSITE unit,
# hands-per-bin -- passing the licensed w value straight through as a bin
# count produces ~1 hand per bin instead of the
# intended whole-cell window. This guard converts and checks before scoring
# runs, rather than trusting the CLI help text alone to prevent the mistake.
S2_VARIANCE_BUDGET_ARTIFACT = (
    Path(__file__).resolve().parent.parent / "configs" / "p2_variance_budget.json"
)


def p2_hand_bins_min_hands_per_seed(cells) -> int | None:
    """The FEWEST hands any one (game, opponent, seed) shard among `cells`
    contributes, via the same `hand_rewards` every other per-hand count in
    this file uses. The conservative denominator for the `--p2-hand-bins`
    unit guard below: checking against the shard with the fewest hands means
    no shard's implied hands-per-bin can silently exceed the licensed floor
    even if shards are unequal. A shard with no LM hand counts as the zero it
    is, matching `_p2_thinnest_endpoint_bin_hands`, which already reports 0
    for such a seed; a seed that contributed nothing must not be able to
    license a finer binning than the cell can support. `None` when no cell has
    any LM hand at all.
    """
    counts = [
        len(hand_rewards(df)) for shards in cells.values() for _seed, df in shards
    ]
    if not any(counts):
        return None
    return min(counts)


def p2_hand_bins_licensed_floor(
    artifact_path: Path = S2_VARIANCE_BUDGET_ARTIFACT,
) -> int | None:
    """Hands-per-bin floor `configs/p2_variance_budget.json` licenses
    for the P2 windowed trajectory, read from its own
    `smoke_test_licensed_w` (the P2 trajectory's own W-justification, the
    most conservative -- largest-w-required -- of its three cited cells).
    `None` when the artefact is absent, unreadable, or carries no such key:
    the caller warns rather than refuses in that case (see
    `validate_p2_hand_bins`) -- a missing artefact is not itself proof a
    requested binning is unsafe.
    """
    try:
        data = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    floor = data.get("smoke_test_licensed_w")
    return int(floor) if isinstance(floor, (int, float)) else None


def validate_p2_hand_bins(
    p2_hand_bins: int,
    n_hands_per_seed: int | None,
    artifact_path: Path = S2_VARIANCE_BUDGET_ARTIFACT,
) -> None:
    """Refuse (or warn about) a `--p2-hand-bins` BIN COUNT whose implied
    hands-per-bin (`n_hands_per_seed / p2_hand_bins`) falls below the floor
    `s2_variance_budget.json` licenses -- the units-confusion defect this
    module's `--p2-hand-bins` help text also documents.

    No-op (byte-identical default path; no silent fallback, since nothing
    is silently corrected) when
    `p2_hand_bins` is None (the computed default, which is licensed by
    construction) or `<= 0` (disabled), or `n_hands_per_seed` is None
    (unknown: no LM hands loaded, e.g. `--skip-properties`). A figure of 0
    is known, not unknown: a run holding an empty seed implies 0 hands per
    bin and is refused like any other count below the floor.

    Raises `SystemExit` -- refuses the run -- when the artefact is present
    and readable and the implied hands-per-bin is below its licensed floor.
    Only warns (stderr, does not refuse) when the artefact is missing or
    unreadable: its absence is not itself evidence the requested binning is
    unsafe, so the run proceeds, matching this file's own default-off
    posture for a diagnostic it cannot check.
    """
    if p2_hand_bins is None or p2_hand_bins <= 0 or n_hands_per_seed is None:
        return
    implied_hands_per_bin = n_hands_per_seed / p2_hand_bins
    floor = p2_hand_bins_licensed_floor(artifact_path)
    if floor is None:
        print(
            f"WARNING: --p2-hand-bins={p2_hand_bins} implies "
            f"{implied_hands_per_bin:.1f} hands/bin (n_hands_per_seed="
            f"{n_hands_per_seed}); {artifact_path} is missing or unreadable, "
            "so this could not be checked against its licensed floor. See "
            "configs/p2_variance_budget.json before trusting a P2 "
            "trajectory built from this run.",
            file=sys.stderr,
        )
        return
    if implied_hands_per_bin < floor:
        raise SystemExit(
            f"--p2-hand-bins={p2_hand_bins} is a BIN COUNT, not a "
            f"hands-per-bin figure: on this data (n_hands_per_seed="
            f"{n_hands_per_seed}) it implies {implied_hands_per_bin:.1f} "
            f"hands/bin, below the {floor}-hands/bin floor {artifact_path} "
            "licenses (smoke_test_licensed_w). Pass a SMALLER bin count "
            "(fewer, wider bins) or see configs/p2_variance_budget.json "
            "for the W justification."
        )


def playout_keys(df):
    """Distinct LM info-set sequences, one key per hand, as a set of tuples.

    This key is blind: it sees the LM's own info-set sequence
    only, filtered to `prompt`-bearing rows, so it does not see the
    opponent's hidden card or any other part of the chance stream the
    opponent's hand alone randomises. It therefore under-counts true
    distinct trajectories (leduc/always_bet: 30 LM-only keys
    against 120 joint-sequence trajectories). `outcome_table` compares this
    key's count against `n_distinct_outcomes` to flag when the blindness
    has provably lost resolution.
    """
    import polars as pl

    lm = df.filter(pl.col("prompt").is_not_null())
    if lm.is_empty():
        return set()
    seqs = lm.group_by("hand_idx", maintain_order=True).agg(
        pl.col("info_set").alias("seq")
    )
    return {tuple(s) for s in seqs["seq"].to_list()}


def distinct_playouts(df):
    """Number of distinct LM info-set sequences, i.e. independent games played.

    Kept for any existing caller; see `playout_keys` for the actual key and
    its documented blindness.
    """
    return len(playout_keys(df))


def bootstrap_mean_ci(values, n_boot, seed=0):
    """Percentile bootstrap interval for the mean.

    Valid as a Monte Carlo interval for E[reward | this policy pair] when hands
    are independent draws from the chance distribution. Returns (nan, nan) for a
    constant series, where the policy pair admits no variation to resample.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or n_boot <= 0 or len(set(arr.tolist())) < MIN_DISTINCT_OUTCOMES:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


#: Every column of an `outcomes.csv` row, in order. The row of a cell with no clean hand is
#: built from this tuple, and a test asserts that a normal row carries exactly these keys,
#: so the two cannot drift apart.
OUTCOME_COLUMNS = (
    "game",
    "opponent",
    "n_hands",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "win_rate_pooled",
    "loss_rate",
    "draw_rate",
    "win_rate_se",
    "win_rate_se_binomial",
    "mean_reward",
    "mean_reward_pooled",
    "reward_ci_lo",
    "reward_ci_hi",
    "reward_ci_method",
    "reward_ci_note",
    "reward_ci_lo_hand",
    "reward_ci_hi_hand",
    "n_distinct_outcomes",
    "n_distinct_playouts",
    "playout_key_blind",
    "n_seeds",
    "n_effective_shards",
    "flag",
    "mean_reward_all_hands",
    "dropped_rate",
    "n_dropped",
    "n_total",
    "dropped_unit",
    "n_shards_all_dropped",
)


def _replicate_point(shard_reward_lists):
    """
    the mean of the per-shard mean rewards, the estimator `mean_reward` reports
    :param shard_reward_lists: one list of per-hand rewards per shard (list[list[float]])
    :returns: the rounded replicate point, or None when no shard has a hand (float | None)
    """
    from lmopt.scoring.replicate import replicate_ci

    means = [sum(r) / len(r) for r in shard_reward_lists if r]
    if not means:
        return None
    point = replicate_ci(means)["point"]
    return None if point is None or math.isnan(point) else round(point, 6)


def _clean_hand_shard(df):
    """
    remove every hand that holds one of the scored LM's harness-fallback rows from one
    shard's frame, keeping the opponent's rows of the hands that stay
    :param df: one shard's rows, LM and opponent (pl.DataFrame)
    :returns: (the frame without the dropped hands, hands dropped, hands with an LM row) (tuple)
    """
    import polars as pl

    from lmopt.scoring._actor import drop_fallback_hands

    hand_keys = [c for c in ("seed", "hand_idx") if c in df.columns] or ["hand_idx"]
    report = drop_fallback_hands(df.filter(pl.col("prompt").is_not_null()), hand_keys)
    if not report.n_dropped:
        return df, 0, report.n_total
    clean_keys = report.kept.select(hand_keys).unique()
    clean = df.join(clean_keys, on=hand_keys, how="semi", maintain_order="left")
    return clean, report.n_dropped, report.n_total


def outcome_table(cells, n_boot):
    """Table 1: win/loss/draw and mean reward per (game, opponent).

    The published interval is a Student-t interval over the 5
    per-shard means, not a bootstrap over pooled hands. The shard is the unit
    of independent replication (run_sweep.py's per-seed shard loop into
    `run_match`'s deal `rng`);
    hands within a shard are not guaranteed independent, because a stateful
    vLLM server can correlate them. The pooled hand-level interval is retained
    in `reward_ci_*_hand` for audit.

    `mean_reward` and `win_rate` are the
    replicate point (`rci["point"]`/`wci["point"]`, the mean of the per-shard
    means), matching the rule applied in
    `replicate_ci_by_key`: the reported point and its CI must come from the
    same estimator, or the point can fall outside its own interval. The
    pooled hand-level values are retained as `mean_reward_pooled` and
    `win_rate_pooled` for audit. With equal shard sizes the two coincide; they
    diverge with unequal shard sizes (a partial or aborted shard), which is
    exactly the case a fixed-width table cannot show as a flag.

    Every statistic above is over clean hands: a hand that holds any harness-fallback
    row of the scored LM is dropped whole, because the substituted move helped
    produce its reward. `n_dropped`, `n_total` (hands with an LM row) and
    `dropped_unit` ("hands") say how many, `dropped_rate` is their ratio, and
    `mean_reward_all_hands` is the same replicate-mean estimator over all hands, so the
    gap between it and `mean_reward` shows how selective the drop was. A cell whose only
    outcome variation came from fallback hands can flag DEGENERATE_CONSTANT_OUTCOME.
    :param cells: (game, opponent) -> list of (seed, shard frame) (dict)
    :param n_boot: resamples for the retained pooled hand-level interval (int)
    :returns: one row dict per (game, opponent) cell, each carrying `OUTCOME_COLUMNS` (list[dict])
    """
    from lmopt.scoring.replicate import effective_replicates, replicate_ci

    rows = []
    for (game, opponent), all_shards in sorted(cells.items()):
        cleaned = [(seed, *_clean_hand_shard(df)) for seed, df in all_shards]
        shards = [(seed, clean) for seed, clean, _d, _t in cleaned]
        n_dropped = sum(d for _s, _c, d, _t in cleaned)
        n_total = sum(t for _s, _c, _d, t in cleaned)
        per_shard_rewards = [hand_rewards(df) for _seed, df in shards]
        # A shard whose every hand was dropped leaves the replicate mean and its
        # interval, so the row counts them instead of letting `n_seeds` hide it.
        n_shards_all_dropped = sum(
            1
            for (_s, _c, _d, t), r in zip(cleaned, per_shard_rewards, strict=True)
            if t and not r
        )
        per_shard_rewards = [r for r in per_shard_rewards if r]
        rewards = [r for shard in per_shard_rewards for r in shard]
        n = len(rewards)
        dropped_columns = {
            "mean_reward_all_hands": _replicate_point(
                [hand_rewards(df) for _seed, df in all_shards]
            ),
            "dropped_rate": round(n_dropped / n_total, 6) if n_total else None,
            "n_dropped": n_dropped,
            "n_total": n_total,
            "dropped_unit": "hands",
            "n_shards_all_dropped": n_shards_all_dropped,
        }
        if n == 0:
            if n_total:
                # Every hand held a fallback row. The cell keeps a row that says
                # so; leaving it out would read as a cell that was never played.
                rows.append(
                    {
                        **dict.fromkeys(OUTCOME_COLUMNS),
                        "game": game,
                        "opponent": opponent,
                        "n_hands": 0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "reward_ci_note": "every hand held a harness-fallback row",
                        "n_distinct_outcomes": 0,
                        "n_distinct_playouts": 0,
                        "playout_key_blind": False,
                        "n_seeds": len(all_shards),
                        "n_effective_shards": 0,
                        "flag": "ALL_HANDS_DROPPED",
                        **dropped_columns,
                    }
                )
            continue
        # Deduplicate across shards. Summing the per-shard counts would
        # report up to 5x the true figure (ratio range 3.29-5.00 across
        # the 20 OLMo cells). This is a
        # Diagnostic only: it is never the resampling unit. The resampling
        # unit is the shard, implemented in lmopt/scoring/replicate.py.
        traj = set()
        for _seed, df in shards:
            traj |= playout_keys(df)
        playouts = len(traj)

        shard_means = [sum(r) / len(r) for r in per_shard_rewards]
        shard_win_rates = [
            sum(1 for x in r if x > 0) / len(r) for r in per_shard_rewards
        ]
        eff = effective_replicates(per_shard_rewards)
        rci = replicate_ci(shard_means)
        wci = replicate_ci(shard_win_rates)

        wins = sum(1 for r in rewards if r > 0)
        losses = sum(1 for r in rewards if r < 0)
        draws = sum(1 for r in rewards if r == 0)
        mean = sum(rewards) / n
        n_out = len(set(rewards))
        # The key is blind: it sees LM info-set sequences only, so it misses
        # part of what the chance stream randomises. When the reward takes
        # more distinct values than the key admits trajectories, the key has
        # provably under-resolved the cell (qwen and mistral
        # goofspiel/cfr_ref each collapse to 1 distinct trajectory while the
        # reward takes 2 distinct values over 5,000 hands).
        playout_key_blind = n_out > playouts
        # Retained for audit: the superseded pooled-hand interval.
        lo_h, hi_h = bootstrap_mean_ci(rewards, n_boot)
        degenerate = n_out < MIN_DISTINCT_OUTCOMES
        p = wins / n
        se_binom = math.sqrt(p * (1 - p) / n) if n else float("nan")

        def _r(x):
            return None if x is None or math.isnan(x) else round(x, 6)

        rows.append(
            {
                "game": game,
                "opponent": opponent,
                "n_hands": n,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                # The replicate point, matching
                # the CI it is reported beside; see the pooled sibling below.
                "win_rate": _r(wci["point"]),
                "win_rate_pooled": round(p, 6),
                "loss_rate": round(losses / n, 6),
                "draw_rate": round(draws / n, 6),
                "win_rate_se": _r(wci["se"]),
                "win_rate_se_binomial": round(se_binom, 6),
                "mean_reward": _r(rci["point"]),
                "mean_reward_pooled": round(mean, 6),
                "reward_ci_lo": _r(rci["lo"]),
                "reward_ci_hi": _r(rci["hi"]),
                "reward_ci_method": rci["method"],
                "reward_ci_note": rci["suppressed_reason"],
                "reward_ci_lo_hand": _r(lo_h),
                "reward_ci_hi_hand": _r(hi_h),
                "n_distinct_outcomes": n_out,
                "n_distinct_playouts": playouts,
                "playout_key_blind": playout_key_blind,
                "n_seeds": len(shards),
                "n_effective_shards": eff["n_effective"],
                "flag": "DEGENERATE_CONSTANT_OUTCOME" if degenerate else "",
                **dropped_columns,
            }
        )
    return rows


# Statuses `emit()`'s `force_status` may set.
# `force_status` is a demote-only escape hatch: every
# other emit() call site relies on PROPERTY_STATUS + the caveat-driven
# downgrade to decide status, and `decision_table` treats "MEASURABLE" as
# the only passing status (everything else buckets into `blocked`). If
# `force_status` could be set to "MEASURABLE" (or any unrecognised string),
# a caller could silently promote a row past the caveat gate -- exactly
# backwards from why the parameter exists. "MEASURABLE" is deliberately
# excluded from this set.
_FORCEABLE_STATUSES = frozenset(
    {
        "MEASURABLE_FLAWED",
        "MEASURABLE_CONFOUNDED",
        "UNAVAILABLE",
    }
)


def _validate_force_status(force_status) -> None:
    """Raise if `force_status` would promote a row rather than demote it.

    `None` (the default -- "do not override") is always fine. Anything else
    must be a recognised non-passing status from `_FORCEABLE_STATUSES`;
    in particular it must never be "MEASURABLE" itself.
    """
    if force_status is not None and force_status not in _FORCEABLE_STATUSES:
        raise ValueError(
            f"force_status={force_status!r} is not an allowed demote-only "
            f"status ({sorted(_FORCEABLE_STATUSES)}); force_status must "
            "never be able to promote a row past the caveat gate"
        )


def _p8_zero_informative_clause(p8: dict, game: str) -> str | None:
    """Fires only when this cell's `n_informative` is 0,
    naming which of two very different facts that zero is, rather than
    letting it read as the "all-degenerate board" case
    `score_property8`'s own module docstring describes.

    A bare `n_informative=0` is indistinguishable, to a reader, from "this
    board has no signal", even though `tests/test_property8.py` measures 2x2
    Dark Hex as not degenerate (25/33 P0 info-states have VoI* > 0), and
    other arms score real, nonzero `n_informative` on the same board via
    the same code path. Such a zero can be
    a fact about one arm's own logged rows, not about the board -- this
    clause makes that distinction explicit using `n_board_informative`
    (`score_property8`'s own count of the reference's informative nodes,
    independent of which ones this arm actually visited).

    `n_nodes` is included in every branch, not just when it distinguishes
    the arm-specific case: without it a reader cannot tell "no data
    reached this cell" from "6,250 real rows scored, none informative" --
    the same absence-vs-measured-zero distinction as `verbose_unavailable`
    and the `REFUSED` state, applied here to a caveat string rather than
    a criterion.
    """
    n_inf = (p8.get("n_informative") or {}).get(game)
    if n_inf != 0:
        return None
    n_nodes = (p8.get("n_nodes") or {}).get(game) or 0
    n_board_inf = (p8.get("n_board_informative") or {}).get(game) or 0
    # `n_nodes` must be checked before
    # `n_board_inf`, not after: `n_board_informative` is computed from
    # `worlds_by_seat` (score_property8), which is built for a seat as
    # soon as any row claims that seat, before the row's own info_set is
    # checked against the map -- so a cell where every row is a lookup
    # miss (info_set absent from the reference, `n_nodes=0`,
    # `discarded_rows` populated) can still carry a real, nonzero
    # `n_board_informative` from the seat's own reference map. Checking
    # `n_board_inf` first there would confidently claim "this arm made
    # real decisions and avoided informative nodes" when in fact zero
    # real decisions were scored -- exactly
    # the misleading-caveat class this clause exists to prevent. For
    # example, a single row with a real `info_set` string absent
    # from the board's worlds map gives `n_nodes=0, n_board_informative=12`
    # (the seat's real reference count) on the live 2x2 fixture.
    if not n_nodes:
        return (
            "n_informative=0; n_nodes=0, no rows from this arm reached the "
            "reference's worlds map at all"
        )
    if n_board_inf:
        # How many distinct info-states the rows reached is the why: an arm that plays one
        # path through the board visits few, and none of them need be informative.
        n_visited = (p8.get("n_visited_infosets") or {}).get(game)
        visited = (
            f"; its rows reached only {n_visited} distinct info-state(s), none of them "
            "informative"
            if n_visited
            else ""
        )
        return (
            f"n_informative=0 of n_nodes={n_nodes} real scored rows; the "
            f"reference board has {n_board_inf} informative info-state(s) "
            "at this threshold that this arm's own rows never reached -- "
            "an ARM-SPECIFIC property of this arm's play, not board "
            f"degeneracy{visited}"
        )
    # n_nodes is truthy here (the `not n_nodes` branch above already
    # returned), so this is the remaining case: real data was scored and
    # the reference itself has no informative info-state at this
    # threshold over the seat(s) this arm occupied.
    return (
        f"n_informative=0 of n_nodes={n_nodes} real scored rows; the "
        "reference board itself has no informative info-state at this "
        "threshold over the seat(s) this arm occupied"
    )


def _p8_range_clause(p8: dict, game: str, n_inf) -> str | None:
    """Clause 1 of `p8_caveat()`: the shortfall-range defect, alone.

    The P8 specification reads the shortfall as a fraction
    bounded at 1, but nothing bounds an action-value gap on a +/-1 payoff
    scale by EVPI: 8 of 16 pilot cells exceed
    1, over 46,265 scored nodes, node maximum 2.0. Fires iff this cell's
    `n_shortfall_gt1` is non-zero, so a cell that stays inside the spec's
    range is not blocked by a defect it does not exhibit.

    Named and callable on its own, not just as `p8_caveat()`'s first part:
    it is the one clause the admissibility rule treats as
    blocking, and the call site needs to ask for exactly that, rather than
    inferring "only the other clause fired" from a side count.
    """
    n_gt1 = (p8.get("n_shortfall_gt1") or {}).get(game) or 0
    if not n_gt1:
        return None
    return (
        f"shortfall exceeds 1 on {n_gt1}/{n_inf} scored nodes, so the "
        "reported mean is a RATIO in units of available VoI, not the "
        "fraction the P8 definition calls it"
    )


def _p8_occupancy_clause(p8: dict, game: str, n_inf) -> str | None:
    """Clause 2 of `p8_caveat()`: the reference-occupancy defect, alone.

    Fires iff the scored info-sets carry a SMALLER share of the CFR+
    reference's own visit mass than of the agent's own decision nodes. Two
    measured quantities compared, no invented constant: when the agent plays
    like the reference the two shares are comparable and this stays silent.
    On the 2x2 pilot arms it is ~1e-6 against
    42.9 %, because an information-optimal player avoids ever needing to
    probe, so P8's yardstick is evaluated almost entirely at positions CFR+
    would never reach.

    Non-blocking.
    Restricting the scored population to where the reference plays was
    measured and rejected (the entire informative population carries 0.0-1e-6 of the
    reference's own visit mass on every board/seat), so this clause is
    structural, not a data-quality defect, and the call site reports it as a
    non-blocking note (`extra`) rather than a `caveat_override`.
    """
    share = (p8.get("reference_visit_share") or {}).get(game)
    n_nodes = (p8.get("n_nodes") or {}).get(game) or 0
    agent_share = (n_inf / n_nodes) if n_nodes else None
    if not (
        share is not None
        and agent_share is not None
        and not math.isnan(share)
        and share < agent_share
    ):
        return None
    # `{:.2%}` renders 2.1e-7 as "0.00%", a present quantity printed as zero. A share
    # this small is the whole point of the clause, so it must never read as none at all.
    pct = share * 100
    shown = "<0.01%" if 0 < pct < 0.005 else f"{pct:.2f}%"
    return (
        f"scored at info-sets carrying {shown} of the CFR+ reference's "
        f"own visit mass but {agent_share:.1%} of the agent's decision "
        "nodes, so the reference continuation values are read where the "
        "reference does not play"
    )


def p8_caveat(p8: dict, game: str) -> str | None:
    """P8's own caveat for one cell, COMPUTED from that cell's measured output.

    P8 is not in `_BELIEF_SCORED_PROPERTIES` because a belief-defect caveat
    would name a defect it does not have; this names the two limitations P8
    actually has, each rendered only when this cell's numbers show it, and
    each printing the numbers it rests on. It is deliberately not a
    `GAME_CAVEATS`-shaped literal: a table encodes which question to ask, and
    the question set goes stale exactly as readily as the answers.

    Returns `None` when neither limitation is present, which lets `emit()`'s
    MEASURABLE_FLAWED -> MEASURABLE downgrade fire normally. `None` rather than
    `""`: `_validate_caveat_override` refuses a blank override precisely so a
    caveat can never be silently cleared, and "no override" is the honest way to
    say "this cell has nothing for me to name".

    The two clauses are `_p8_range_clause` and `_p8_occupancy_clause`,
    separate so a caller can ask which one fired,
    rather than reverse-engineering it from this function's combined text or
    from a side count like `n_shortfall_gt1` -- see the P8 call site's use of
    `_p8_range_clause` directly. This function returns the full combined
    text, in that order, or `None` when neither clause fires.
    """
    n_inf = (p8.get("n_informative") or {}).get(game)
    if not n_inf:
        return None
    parts = [
        _p8_range_clause(p8, game, n_inf),
        _p8_occupancy_clause(p8, game, n_inf),
    ]
    return "; ".join(p for p in parts if p) or None


def _validate_caveat_override(caveat_override) -> None:
    """Raise if `caveat_override` would silently clear a property's caveat.

    `force_status` above has this same demotion-only guarantee. `emit()`'s
    own downgrade rule --

        if status == "MEASURABLE_FLAWED" and not caveat and not game_caveat:
            status = "MEASURABLE"

    -- fires on an empty caveat, not merely a missing one. Without this check,
    `emit("P2_adaptivity", gap, ..., caveat_override="")` on a cell with no
    `GAME_CAVEATS` entry would promote a real, non-clean measurement straight past
    the downgrade gate to `status="MEASURABLE"` with `caveat=""` -- exactly
    the forbidden signature. A check on what production code does produce
    cannot catch what a hostile argument could produce.
    `None` (the default -- "do not override") is always fine, and any
    non-empty, non-whitespace string is fine (that is the feature: replacing
    a caveat's text, never its presence). Only a blank/whitespace-only
    override is refused, mirroring `_validate_force_status`'s shape.
    """
    if caveat_override is not None and not caveat_override.strip():
        raise ValueError(
            f"caveat_override={caveat_override!r} is blank/whitespace-only; "
            "caveat_override must never be able to silently clear a "
            "property's caveat and promote it past the MEASURABLE_FLAWED "
            "-> MEASURABLE downgrade gate with no reason recorded"
        )


# The properties whose per-cell rows carry
# a shard-replicate interval, and therefore every key `_cell_replicate_cis`
# returns. These are the thresholded statistics of the pre-registered
# threshold table plus the two
# rule properties (P3 and P6; their rows still get an interval when one is
# computable):
#   P2 br_gap (the level -- not used by the decision layer, but its
#   properties.csv row keeps the
#   interval: reported statistics keep their uncertainty, exactly as the
#   unthresholded shortfall would if it were listed here) and the P2
#   Shrinkage delta (`P2_adaptivity_delta`, same
#   per-shard `score_property2` loop, each shard's delta computed from that
#   shard's own trajectory; (None, None) at the default p2_hand_bins=0,
#   where no trajectory exists),
#   P5 per-pair margin, P7a TVD, P7b BR-to-own-belief fraction,
#   P8 capture_rate (the pre-registered table's P8 statistic -- not the shortfall,
#   which stays reported unthresholded), P3 mean KL. P1/P4 (paired TVD under
#   a probe arm) and P6 are scored by a different entrypoint
#   (scripts/probe_scores.py) and have no population reaching this function's per-cell CI
#   machinery -- not a claim that no P1/P4/P6 data exists anywhere: P6 has
#   real driven arms under data/run_experiment/recycle-v4/p6-*/. Listed
#   here only to document that their rows carry (None, None) in this table
#   deliberately -- absent population for property_table, never a crash,
#   never a number.
_CI_PROPERTIES = (
    "P1_invariance",
    "P2_adaptivity",
    "P2_adaptivity_delta",
    "P3_randomisation",
    "P4_payoff_sensitivity",
    "P5_calibration_per_pair_margin",
    "P6_subgame_perfection",
    "P7a_belief_error",
    "P7b_action_belief",
    "P8_capture_rate",
)


def _trajectory_delta(means) -> float:
    """
    The P2 shrinkage delta of one windowed trajectory, read from the LM's mean
    reward per bin. With the best-response target pinned across bins,
    br_gap(late) - br_gap(early) equals the first bin's mean minus the last
    bin's, so the delta needs no best-response value and stays defined when
    that walk fails. Negative = the gap shrank; >= 0.0 fails the signed rule.
    The two ends are the least and the most accumulated evidence the cell
    offers; intermediate bins are reported in the `P2_adaptivity_trajectory`
    row. One definition shared by the pooled row and the per-shard
    replicates, so the two cannot diverge on which bins "early" and "late"
    name.
    :param means: the LM mean reward of each hand-index bin, first bin first (list[float] | None)
    :returns: mean(first bin) - mean(last bin), or nan when the trajectory is absent, has fewer than 2 bins, or an endpoint bin is empty (float)
    """
    if not means or len(means) < 2:
        return float("nan")
    early, late = means[0], means[-1]
    if early is None or late is None or math.isnan(early) or math.isnan(late):
        return float("nan")
    return early - late


P2_NO_MEMORY_REASON = (
    "no cross-hand memory: the policy is a fixed function of the information "
    "set, so P2 is identically zero by construction, and P2 is scored from M1 "
    "and M2 only"
)

P2_DELTA_NOTE = (
    "br_gap(late) - br_gap(early) over {bins} hand-index bins, which equals the "
    "first bin's mean reward minus the last bin's because the best-response "
    "target cancels; mean reward is a proxy that still carries the luck of the "
    "deal, and the October P2 built on per-node regret measures the same thing "
    "with far less variance; negative means the gap shrank (the model improved) "
    "and passes, zero or positive fails; the verdict uses the between-seed "
    "interval"
)


def _resolve_p2_hand_bins(p2_hand_bins, fewest_hands, floor):
    """
    The bin count one cell is scored with. An explicit count (0 disables the
    trajectory) is used as given. `None` derives the count from the cell's own
    data: floor(fewest LM hands in any seed / licensed hands per bin), and when
    that is below two the cell cannot be windowed, so the reason is returned
    for the row that says so.
    :param p2_hand_bins: an explicit bin count, or None for the computed default (int | None)
    :param fewest_hands: the fewest LM hands any one seed of the cell holds, before fallback drops (int)
    :param floor: the licensed hands per bin, per seed; required when `p2_hand_bins` is None (int | None)
    :returns: (bin count to score with, reason the default cannot reach two licensed bins or None) (tuple[int, str | None])
    """
    if p2_hand_bins is not None:
        return p2_hand_bins, None
    bins = fewest_hands // floor
    if bins >= 2:
        return bins, None
    return 0, (
        f"P2 needs at least {2 * floor} hands per seed (two bins of {floor} "
        f"hands), and this run has {fewest_hands}"
    )


def _p2_memory_state(df):
    """
    Whether the agent under test could condition on earlier hands, read from the
    `memory_rung` of the cell's LM rows.
    :param df: the cell's frame (polars DataFrame)
    :returns: ("ok", None), ("none", reason) when every LM row is at the baseline rung, or ("blocked", reason) when the rung is unrecorded, mixed or unknown (tuple[str, str | None])
    """
    import polars as pl
    from lmopt.analysis.contrast.contrast_memory import BASELINE_RUNG
    from lmopt.prompting.memory import MEMORY_RUNGS

    lm = df.filter(pl.col("prompt").is_not_null()) if "prompt" in df.columns else df
    rungs = (
        set(lm["memory_rung"].drop_nulls().unique().to_list())
        if "memory_rung" in lm.columns
        else set()
    )
    if not rungs:
        return "blocked", (
            "the memory rung is not recorded for this cell, so it cannot be "
            "told whether the agent can condition on earlier hands"
        )
    if rungs == {BASELINE_RUNG}:
        return "none", P2_NO_MEMORY_REASON
    if BASELINE_RUNG in rungs:
        return "blocked", f"the cell mixes memory rungs ({', '.join(sorted(rungs))})"
    unknown = sorted(r for r in rungs if r not in MEMORY_RUNGS)
    if unknown:
        return (
            "blocked",
            f"the cell carries unrecognised memory rungs ({', '.join(unknown)})",
        )
    return "ok", None


def _p2_seed_hand_indices(shards):
    """
    The LM hand indices each seed of one cell holds, before fallback drops, in
    shard order. The population is the scorer's own: the rows that carry a
    prompt.
    :param shards: the cell's [(seed, frame)] shards (list)
    :returns: one list of hand indices per shard, empty for a shard with no LM hands (list[list[int]])
    """
    import polars as pl

    seeds = []
    for _seed, frame in shards:
        lm = (
            frame.filter(pl.col("prompt").is_not_null())
            if "prompt" in frame.columns
            else frame
        )
        seeds.append(
            lm["hand_idx"].unique().to_list() if "hand_idx" in lm.columns else []
        )
    return seeds


def _p2_pooled_hand_span(shards):
    """
    The span of hand indices one cell nominally holds, pooled over its seeds and
    taken before fallback drops. Every shard of the cell, the pooled cell itself
    and the licence are cut on these edges, so a seed shorter than the longest
    one is not given a window of its own.
    :param shards: the cell's [(seed, frame)] shards (list)
    :returns: (first hand index, last hand index), or None when no seed holds an LM hand (tuple[int, int] | None)
    """
    every = [h for hands in _p2_seed_hand_indices(shards) for h in hands]
    if not every:
        return None
    return int(min(every)), int(max(every))


def _p2_thinnest_endpoint_bin_hands(shards, bins):
    """
    The fewest LM hands any seed holds in an endpoint bin, on the bins
    `score_property2` will actually cut: equal-width over the pooled span of the
    cell's hand indices, so a seed shorter than the longest one holds fewer hands
    in the late bin than its own length suggests. Counted before fallback drops,
    because the licence is about the window the variance budget was computed
    for; the hands dropped from it are reported beside the number as its dropped
    rate.
    :param shards: the cell's [(seed, frame)] shards (list)
    :param bins: the bin count the cell is scored with, at least 2 (int)
    :returns: the fewest hands in the first or last bin over the seeds, 0 for a seed with no LM hands (int)
    """
    from lmopt.scoring.property2 import hand_bin_index

    seeds = _p2_seed_hand_indices(shards)
    span = _p2_pooled_hand_span(shards)
    if span is None:
        return 0
    thinnest = None
    for hands in seeds:
        in_bin = [hand_bin_index(h, span, bins) for h in hands]
        ends = min(in_bin.count(0), in_bin.count(bins - 1))
        thinnest = ends if thinnest is None else min(thinnest, ends)
    return thinnest


def _p2_hands_per_bin_reason(thinnest_bin_hands, floor):
    """
    Why a windowed delta is not licensed by the size of its bins, or None when
    it is.
    :param thinnest_bin_hands: the fewest hands any seed holds in an endpoint bin, before fallback drops (int)
    :param floor: the licensed hands per bin, per seed, or None when the variance budget is unavailable (int | None)
    :returns: the reason, or None when every seed's endpoint bins hold at least `floor` hands (str | None)
    """
    if floor is None:
        return (
            "the licensed hands per bin is not available, because "
            "configs/p2_variance_budget.json is missing or unreadable"
        )
    if thinnest_bin_hands < floor:
        return (
            f"an endpoint bin holds {thinnest_bin_hands} hands in its thinnest "
            f"seed before fallback drops, below the licensed {floor}"
        )
    return None


#: The distribution-basis columns the end-of-cell back-fill adds to every `properties.csv` row, filled
#: only on an LM P7b row; named once so a reader of the writer's columns need not find the back-fill.
EXACT_BASIS_COLUMNS = ("value_exact", "ci_low_exact", "ci_high_exact", "exact_reason")

#: The exit status `main` returns when the exact-policy P7b is handed an arm above memory level M0,
#: the same status `exact_policy_pass.py` returns for its own memory-level refusal.
EXIT_EXACT_NOT_M0 = 10

#: Why `exact_policy` is None, by `load_exact_policy`'s refusal tag, in the words a row's caveat uses.
_EXACT_ARTEFACT_REFUSAL_TEXT = {
    "missing_file": "the artefact file does not exist",
    "unreadable": "the artefact file could not be read/parsed",
    "missing_columns": "the artefact is missing required column(s)",
    "multi_model_mix": "REFUSED: the artefact carries more than one distinct model",
    "model_mismatch": "REFUSED: the artefact was produced by a DIFFERENT model than this run's own",
}


def _p7b_exact_cell(shards, df, game, g, c, exact_policy, refusal_reason=None) -> dict:
    """
    one cell's P7b on the exact policy: the pooled value, its seed-mean centre and its replicate interval
    :param shards: the cell's `[(seed, frame)]` shard list (list)
    :param df: the cell's whole frame (pl.DataFrame)
    :param game: the game key (str)
    :param g: `{game: pyspiel.Game}` for this game, or None (dict | None)
    :param c: `{game: policy}` for this game, or None (dict | None)
    :param exact_policy: `load_exact_policy(...).policy`, or None when no pass was loaded (dict | None)
    :param refusal_reason: why `exact_policy` is None, as `load_exact_policy` tags it (str | None)
    :returns: dict with `value` (the seed mean where an interval forms, else the pooled value, None when
        there is none), `value_pooled`, `ci` ((lo, hi) or (None, None)), `n_replicates`,
        `n_distinct` and `reason` (why there is no value, "" when there is one)
    :raises ExactScorerNotM0: when the cell's rows are above memory level M0
    """
    from lmopt.scoring.property7 import ExactScorerNotM0, score_property7_exact
    from lmopt.scoring.replicate import replicate_ci

    empty = {
        "value": None,
        "value_pooled": None,
        "ci": (None, None),
        "n_replicates": None,
        "n_distinct": None,
        "reason": "",
    }
    if exact_policy is None:
        levels = (
            set(df["memory_rung"].drop_nulls().unique().to_list())
            if "memory_rung" in df.columns
            else set()
        )
        if levels - {"M0"}:
            return {
                **empty,
                "reason": f"memory level {sorted(levels)}: above M0 the policy at an info-set is "
                f"not one distribution, so there is no exact value to take",
            }
        detail = _EXACT_ARTEFACT_REFUSAL_TEXT.get(refusal_reason)
        return {
            **empty,
            "reason": "no exact-policy artefact was loaded for this run"
            + (f" ({detail})" if detail else ""),
        }
    if g is None or c is None:
        return {**empty, "reason": "no game or CFR reference for this cell"}

    # One dict for the pooled call and every shard call: a best-response set is a pure function of
    # (seat, info-set, stated belief) for one game and one reference policy, and the shards' rows are
    # the pooled call's rows. The reference policy `c` is game-level, so this per-cell boundary is
    # deliberately narrower than correctness requires. Do not widen it for speed; a dict that outlives
    # one reference policy returns another policy's best response, wrong in value and not in cost.
    br_cache: dict = {}

    def _score(frame):
        out = score_property7_exact(
            frame, exact_policy, c, g, include_sampled=False, br_cache=br_cache
        )
        if game in out["not_applicable_exact"]:
            return None, out["not_applicable_exact"][game]
        value = out["action_belief_agreement_exact"].get(game)
        if value is None or math.isnan(value):
            return None, "the exact pass scored no P7b row for this cell"
        return float(value), ""

    try:
        pooled, why = _score(df)
    except ExactScorerNotM0:
        raise
    except Exception as e:
        return {
            **empty,
            "reason": f"the exact P7b scorer raised: {type(e).__name__}: {e}",
        }
    if pooled is None:
        return {**empty, "reason": why}
    out = {**empty, "value": round(pooled, 6), "value_pooled": round(pooled, 6)}
    if len(shards) >= 2:
        values = []
        for _seed, frame in shards:
            value, _ = _score(frame)
            values.append(float("nan") if value is None else value)
        rci = replicate_ci(values)
        if not (math.isnan(rci["lo"]) or math.isnan(rci["hi"])):
            finite = [v for v in values if not math.isnan(v)]
            out.update(
                value=round(float(rci["point"]), 6),
                ci=(round(rci["lo"], 6), round(rci["hi"], 6)),
                n_replicates=rci["n_replicates"],
                n_distinct=len(set(finite)),
            )
    return out


def _cell_replicate_cis(
    shards,
    game,
    opponent,
    g,
    c,
    p2_hand_bins: int = 0,
    p2_hand_span=None,
    counts: dict | None = None,
    means: dict | None = None,
    br_cache: dict | None = None,
):
    """Shard-replicate 95% t-intervals for one cell's thresholded statistics.

    :param br_cache: an optional dict `score_property7` fills with its 7b best-response sets, so the
        per-shard calls reuse the sets the cell's other calls computed. It is valid only within one
        cell, whose reference policy `c` is fixed; the caller creates it per cell and shares it with
        nothing wider (dict | None)

    :param counts: an optional dict this function fills in place,
        `{property: (n_replicates, n_distinct)}` for every property it
        forms an interval for (nothing is recorded for a property whose
        interval is `(None, None)`). `n_replicates` is `replicate_ci`'s
        own surviving count (after its NaN drop); `n_distinct` is the
        number of distinct raw, un-rounded finite per-shard values the
        interval was formed over -- raw because `replicate_ci` itself sees
        the raw floats, so a distinct-count computed on the rounded 6-dp
        `ci_lo`/`ci_hi` output would describe a different (coarser)
        quantity than the one the interval was actually built from. `None`
        (the default) skips the bookkeeping entirely. A recorded count always has `n_replicates >=
        2`: it is only ever recorded once `replicate_ci` has formed a real
        (non-suppressed) interval, and `replicate_ci` itself never forms
        one below 2 surviving replicates.

    :param means: an optional dict this function fills in place,
        `{property: point}` for every property it forms an interval for,
        `point` being `replicate_ci`'s own NaN-dropped mean of the
        per-shard values (the same mean the interval is centred on).
        Nothing is recorded for a property whose interval is `(None,
        None)`. `None` (the default) skips the bookkeeping entirely.
        This is the per-cell
        seed mean `property_table` publishes as a row's headline `value`;
        the pooled per-cell estimand is kept beside it
        as `value_pooled`.

    p2_hand_span: the cell's pooled pre-drop span of hand indices
    (`_p2_pooled_hand_span`), passed to every per-shard `score_property2` call
    so a short shard is cut on the cell's edges rather than on its own. Without
    it the between-seed interval and the licence read different windows
    whenever the seeds differ in length. `None` (the default) leaves each shard
    on its own span, which is what a caller with no cell to pool over wants.

    p2_hand_bins: threaded from `property_table` so
    the P2 shrinkage delta gets the same per-shard treatment as the level --
    each shard's replicate is `_trajectory_delta` of that shard's own
    trajectory (its own first matchup, mirroring the level's per-shard
    matchup resolution below -- the point-vs-interval note below
    applies to the delta identically: the pooled delta row resolves the
    pooled frame's first matchup). Defaults to 0,
    where no trajectory exists and `P2_adaptivity_delta` stays (None, None).

    The shard-replicate estimator (`lmopt.scoring.replicate
    .replicate_ci`: shard = the unit of independent replication, Student-t
    over the per-shard values, t(4) at 5 shards), applied to the thresholded
    statistic of each `_CI_PROPERTIES` entry by re-invoking the same pure
    scorer the pooled row uses on each shard's own frame. No scorer is
    modified; a shard here is one `(seed, df)` entry of `load_cells`' output
    -- the same per-seed frames `outcome_table`'s reward CIs already treat as
    replicates (there is no shard column in the logged frame; the shard is
    the per-seed Parquet file itself).

    The seed is the unit of independent replication, not the value the seed
    happened to produce: the interval is formed over every shard of the
    cell, one value per shard, with no filtering or deduplication by
    whether two shards' values (or their underlying per-hand series) agree.
    Two shards that land on the same reward or belief series are still two
    genuine draws of the chance stream; the value they share is itself a
    finding, not evidence that one of them is redundant. `replicate_ci`
    drops a NaN per-shard value (a shard the per-shard scorer could not
    score, e.g. no LM rows survived, or all of them were harness-fallback
    rows the scorer drops) and reports the surviving count; a shard set
    that is a genuine constant yields a real, zero-width interval, not a
    suppressed one.

    Returns `{property: (ci_lo, ci_hi)}` for every `_CI_PROPERTIES` key,
    `(None, None)` wherever no interval is formable: fewer than 2 shards,
    fewer than 2 shards with a finite per-shard value once NaNs are dropped
    (`replicate_ci`'s own suppression contract), a property this function
    has no scorer wired for (P1/P4/P6 -- scored by a different entrypoint,
    scripts/probe_scores.py), or a per-shard scorer raise (warned on
    stderr, never silently narrowed to the surviving shards -- a raise is a
    defect to surface, not a NaN to drop). `decision_table` maps (None,
    None) on a thresholded row to CI_UNAVAILABLE.

    Point-vs-interval note: a row's headline `value` is the interval's own
    centre -- the mean of the per-shard values this function forms the
    interval over (`means`, above) -- so a row can never disagree with its
    own `[ci_lo, ci_hi]`. The per-cell pooled estimand this function never
    computes (the pooled scorer call the caller makes separately) is
    published beside it as `value_pooled`. The two coincide at equal shard
    sizes and can diverge slightly with unequal ones, exactly as
    `mean_reward` vs `mean_reward_pooled` do in outcomes.csv -- the audit
    trail there names the same divergence.

    Cost: O(shards) extra invocations per scorer per cell (5 scorers), only
    on cells with >= 2 shards.
    """
    from lmopt.scoring.property2 import score_property2
    from lmopt.scoring.property3 import score_property3
    from lmopt.scoring.property5 import score_property5
    from lmopt.scoring.property7 import score_property7
    from lmopt.scoring.property8 import score_property8
    from lmopt.scoring.replicate import replicate_ci

    none_pair = (None, None)
    out = {prop: none_pair for prop in _CI_PROPERTIES}
    if len(shards) < 2:
        return out
    shard_dfs = [d for _s, d in shards]
    nan = float("nan")

    def _num(v):
        return nan if v is None else float(v)

    def _interval(prop, values):
        rci = replicate_ci(values)
        if math.isnan(rci["lo"]) or math.isnan(rci["hi"]):
            return none_pair
        if counts is not None:
            finite = [
                float(v) for v in values if v is not None and not math.isnan(float(v))
            ]
            counts[prop] = (rci["n_replicates"], len(set(finite)))
        if means is not None:
            means[prop] = rci["point"]
        return (round(rci["lo"], 6), round(rci["hi"], 6))

    def _warn(props, e):
        print(
            f"WARNING: no replicate interval for {'/'.join(props)} on cell "
            f"({game}, {opponent}): per-shard scorer raised "
            f"({type(e).__name__}: {e}); ci_lo/ci_hi stay empty",
            file=sys.stderr,
        )

    # One loop per scorer (not per statistic), so a scorer that yields two
    # thresholded statistics (P7a/P7b) is invoked once per shard, keeping
    # the cost O(shards) per scorer.
    try:
        vals = []
        for sdf in shard_dfs:
            kl = score_property3(sdf, games=g, cfr_policies=c)["mean_kl"].get(game)
            vals.append(nan if kl is None else float(kl.value))
        out["P3_randomisation"] = _interval("P3_randomisation", vals)
    except Exception as e:
        _warn(("P3_randomisation",), e)

    try:
        vals = []
        for sdf in shard_dfs:
            vals.append(_num(score_property5(sdf)["per_pair_margin"].get(game)))
        out["P5_calibration_per_pair_margin"] = _interval(
            "P5_calibration_per_pair_margin", vals
        )
    except Exception as e:
        _warn(("P5_calibration_per_pair_margin",), e)

    try:
        tvds, agrs = [], []
        for sdf in shard_dfs:
            p7 = score_property7(sdf, cfr_policies=c, games=g, br_cache=br_cache)
            tvds.append(_num(p7["tvd_mean"].get(game)))
            agrs.append(_num(p7["action_belief_agreement"].get(game)))
        out["P7a_belief_error"] = _interval("P7a_belief_error", tvds)
        out["P7b_action_belief"] = _interval("P7b_action_belief", agrs)
    except Exception as e:
        _warn(("P7a_belief_error", "P7b_action_belief"), e)

    try:
        vals, deltas = [], []
        for sdf in shard_dfs:
            # One call yields both P2 statistics (level + delta),
            # keeping the cost O(shards) per scorer, the P7a/P7b pattern
            # above. `n_hand_bins > 0` leaves every pre-existing key
            # byte-identical (score_property2's own contract), so threading
            # it here cannot move the level replicate.
            p2 = score_property2(
                sdf,
                games=g,
                cfr_policies=c,
                n_hand_bins=p2_hand_bins,
                hand_span=p2_hand_span,
            )
            # The same one-matchup resolution the pooled P2 block uses:
            # index every dict off a single
            # matchup_id so the per-shard gap describes one matchup.
            mid = next(iter(p2.get("br_gap", {})), None)
            gap = (
                p2.get("br_gap", {}).get(mid, {}).get(game) if mid is not None else None
            )
            vals.append(_num(gap))
            if p2_hand_bins > 0:
                # This shard's delta from its own per-bin means (its own
                # first matchup): nan wherever the shard could not be
                # binned, which `replicate_ci` drops by its own contract.
                tmid = (
                    mid
                    if mid is not None
                    else next(iter(p2.get("lm_mean_trajectory", {})), None)
                )
                # Named `bin_means`, not `means`: this function's own
                # `means` parameter (the per-cell seed-mean bookkeeping
                # dict `_interval` writes into) is a plain local variable
                # here too, since Python has no block scoping -- reusing
                # the name would rebind that parameter's local slot to
                # this shard's per-bin trajectory value on every
                # iteration, corrupting every `_interval` call after the
                # first one this try block makes.
                bin_means = (
                    p2.get("lm_mean_trajectory", {}).get(tmid, {}).get(game)
                    if tmid is not None
                    else None
                )
                deltas.append(_trajectory_delta(bin_means))
        out["P2_adaptivity"] = _interval("P2_adaptivity", vals)
        if p2_hand_bins > 0:
            out["P2_adaptivity_delta"] = _interval("P2_adaptivity_delta", deltas)
    except Exception as e:
        _warn(("P2_adaptivity", "P2_adaptivity_delta"), e)

    try:
        vals = []
        for sdf in shard_dfs:
            p8 = score_property8(sdf, cfr_policies=c, games=g)
            vals.append(
                _num(p8["capture_rate"].get(game)) if p8.get("available") else nan
            )
        out["P8_capture_rate"] = _interval("P8_capture_rate", vals)
    except Exception as e:
        _warn(("P8_capture_rate",), e)

    return out


def _is_undefined_value(x) -> bool:
    """True if `x` is an absent/undefined measurement: `None`, or a float NaN.

    Shared by `_property_row` and `property_table`. On the call site fed by
    `_trajectory_delta` (whose return is always a plain Python float, never
    `None`) the `is None`/`isinstance` checks are no-ops, kept deliberately
    for uniformity.

    The `isinstance(x, float)` guard is inert for every current call site
    -- every value that reaches
    them is a plain Python float (each producing
    scorer: `score_property3_exact`, `score_property5`, `score_property7`,
    `_p8_occupancy_clause`, `_trajectory_delta`; none ever returns
    `numpy.float32` or `Decimal`) -- but it is a genuine, live correctness
    difference in general: `isinstance(numpy.float32('nan'), float)` is
    `False`, so an `np.float32` NaN would short-circuit past `math.isnan`
    and read as "defined" under this guard, while a plain Python float or
    `numpy.float64` NaN reads correctly as undefined. It is not dead
    code -- if a future caller's dtype changes, this
    guard's behaviour changes with it, silently.
    """
    return x is None or (isinstance(x, float) and math.isnan(x))


# One text shared by every site that uses it. Each site's distinguishing information
# (the property name) lives in the surrounding `emit()` call, not in this
# text.
_NO_CALIBRATION_ROWS_CAVEAT = (
    "no calibration rows for this cell (no elicited_belief/"
    "true_posterior pairs, or the game excluded from this "
    "scorer)"
)


#: The committed P3 sentinel anchor artefact, one anchor per (game, opponent). The default `main()`
#: reads; `--p3-sentinel-anchors` overrides it.
DEFAULT_P3_SENTINEL_ANCHORS = "configs/anchors/p3_sentinel_anchors.json"


def _interval_text(lo, hi) -> str:
    """
    an interval rendered for a caveat a human reads, saying so when it is
    absent rather than printing a bound that was never computed
    :param lo: the interval's lower bound, or None (float | None)
    :param hi: the interval's upper bound, or None (float | None)
    :returns: "[lo, hi]" to six decimals, or "unavailable" (str)
    """
    if lo is None or hi is None:
        return "unavailable"
    return f"[{lo:.6f}, {hi:.6f}]"


#: Columns every `properties.csv` row carries for the rate at which the scored
#: LM's unusable rows were dropped before the number was computed. `write_csv`
#: raises on rows with differing key sets, so every row carries all of them.
DROPPED_COLUMNS = (
    "dropped_rate",
    "n_dropped",
    "n_total",
    "dropped_unit",
    "n_infosets",
    "n_infosets_fully_dropped",
    "fallback_row_rate",
)

#: Which scorer's drop report backs which property, first prefix match wins.
#: The P2 trajectory delta is computed from the same kept hands as the cell's
#: P2 estimate, so it reads P2's report; `P3_randomisation_margin` reads P3's
#: for the same reason, being that cell's own regret shifted by a constant.
#: A property matching no prefix is not computed from one scorer's LM rows and
#: carries no rate.
_DROPPED_SOURCE_BY_PREFIX = (
    ("P3_randomisation_exact", "P3x"),
    ("P3_randomisation", "P3"),
    ("P2_", "P2"),
    ("P5_", "P5"),
    ("P7a_", "P7a"),
    ("P7b_", "P7b"),
    ("P8_", "P8"),
    ("state_sensitivity_", "SS"),
)


def _dropped_source(prop):
    """
    which scorer's drop report a property row reads
    :param prop: the property name of a row (str)
    :returns: "P2", "P3", "P3x", "P5", "P7a", "P7b", "P8", "SS" (the state-sensitivity rows), or None when no scorer backs it (str | None)
    """
    for prefix, source in _DROPPED_SOURCE_BY_PREFIX:
        if prop.startswith(prefix):
            return source
    return None


def _dropped_fields(detail):
    """
    the seven `DROPPED_COLUMNS` for one row
    :param detail: a scorer's `dropped_detail` leaf, or None (dict | None)
    :returns: dict keyed by DROPPED_COLUMNS, all None when the scorer reported no counts
    """
    if not detail or "n_dropped" not in detail:
        return dict.fromkeys(DROPPED_COLUMNS)
    n_total = detail["n_total"]
    rate = detail["n_dropped"] / n_total if n_total else None
    row_rate = detail.get("fallback_row_rate")
    return {
        "dropped_rate": None if rate is None else round(rate, 6),
        "n_dropped": detail["n_dropped"],
        "n_total": n_total,
        "dropped_unit": detail.get("unit"),
        "n_infosets": detail.get("n_infosets"),
        "n_infosets_fully_dropped": detail.get("n_infosets_fully_dropped"),
        "fallback_row_rate": None if row_rate is None else round(row_rate, 6),
    }


def _game_detail(result, key, game):
    """
    one game's leaf of a scorer's per-game drop detail
    :param result: a scorer's return dict (dict)
    :param key: "dropped_detail" or "belief_dropped_detail" (str)
    :param game: the cell's game key (str)
    :returns: the leaf dict, or None when the scorer reported none for the game (dict | None)
    """
    return (result.get(key) or {}).get(game)


def _p2_game_detail(p2, game):
    """
    the P2 drop leaf for the matchup the cell's P2 rows are read from
    A cell frame can hold several matchups, and the P2 rows take their value
    from the first matchup in `br_gap`; the drop counts come from that same
    matchup so a row never pairs one matchup's value with another's counts. A
    cell whose every hand was dropped has no `br_gap` entry, so the first
    matchup in `dropped_detail` is read instead.
    :param p2: `score_property2`'s return dict (dict)
    :param game: the cell's game key (str)
    :returns: that matchup's leaf for the game, or None (dict | None)
    """
    by_matchup = p2.get("dropped_detail") or {}
    matchup = next(iter(p2.get("br_gap") or {}), None)
    if matchup is None:
        matchup = next(iter(by_matchup), None)
    return None if matchup is None else (by_matchup.get(matchup) or {}).get(game)


def _bare_row(game, opponent, prop, status, caveat, *, player="LM") -> dict:
    """One `property_table` row bypassing `_property_row` entirely: a
    placeholder for a state `_property_row`'s own downgrade-gate/game-
    caveat/game-block logic is not designed to produce.

    Every call site in `property_table` emits `status` `NOT_APPLICABLE`,
    `NOT_COMPUTED`
    or `MEASURABLE_FLAWED` with no measured value (`AWAITING_DATA`, a P2
    delta row for a collection too small to window, is the one further
    status: it also carries no value) -- `value` is
    deliberately not a parameter here, since a parameter that is always the
    same constant at every real call site invites a future caller to pass
    something else, silently defeating the point of a placeholder-row
    constructor. A
    site needing a real measured value belongs through `_property_row`,
    not this function.

    The risk this closes is not key order -- `write_csv`'s
    `csv.DictWriter` is fieldname-keyed via `fieldnames =
    list(filtered_rows[0].keys())`, order-independent -- it is key-SET
    consistency: every row `property_table` produces, bare or emitted,
    must carry the same 8 keys (`game`, `opponent`, `player`,
    `property`, `value`, `value_pooled`, `status`, `caveat`) as
    `_property_row`'s own output, or `DictWriter` either raises
    `ValueError` (a later row carrying an extra key row 0 lacks -- the
    loud, likely-in-practice failure) or, the genuinely silent case,
    fills a blank cell via `restval` for a row that is missing a key row
    0 happens to carry (that row's own value for the column is silently
    absent; the column itself and every other row's value for it are
    unaffected). Every site agrees with
    `_property_row`'s key set.

    `value_pooled` is always `None` here, matching `value`: a bare row
    never carries a measured value of any kind, pooled or seed-mean, so
    the two coincide the same way they do for an interval-free
    `_property_row` row. A row this function produces inside
    `property_table`'s per-cell loop still passes through the end-of-cell
    back-fill afterwards, which re-derives `value_pooled` from `value`
    for every row of the cell uniformly; a row produced for a
    `configured` cell with no shards at all never reaches that back-fill,
    so it needs its own `value_pooled` set here instead.
    """
    return {
        "game": game,
        "opponent": opponent,
        "player": player,
        "property": prop,
        "value": None,
        "value_pooled": None,
        "status": status,
        "caveat": caveat,
        # Always empty here -- a bare
        # row bypasses `measure_belief_defects`/`_partial_reason_class`
        # entirely, so it never carries one of the three approved reasons.
        # Present so this row's key set matches `_property_row`'s own.
        "partial_reason_class": "",
        "n_nodes": None,
        "n_informative": None,
        "n_hands": None,
        "n_distinct_prompts": None,
    }


#: The opponent whose own P2 and P3 cells are scored. It is the control arm: an
#: equilibrium player, so its numbers say what the scorers read for a player with no
#: error to find. The other opponents' cells stay NOT_COMPUTED.
_SCORED_CONTROL_OPPONENT = "cfr_ref"

#: The rows the control arm gets from the LM arms' own scorers, in table order.
_CONTROL_ARM_SCORED_PROPERTIES = (
    "P3_randomisation",
    "P2_adaptivity",
    "P2_adaptivity_coverage",
)


def _control_actor_label(opponent) -> str:
    """
    the `actor` value the match runner logs for an opponent, derived from the opponent's key
    :param opponent: the opponent key, such as "cfr_ref" (str)
    :returns: the logged actor label, such as "cfrref" (str)
    """
    return opponent.replace("_", "")


def _control_arm_row(game, opponent, prop, value, status, caveat) -> dict:
    """
    one `property_table` row for the opponent's own play, carrying a measured value. It is built
    on `_bare_row` so its key set is the table's, then given the value `_bare_row` withholds on
    purpose; the LM arms' rows keep going through `_property_row`
    :param game: the game key (str)
    :param opponent: the opponent name, also written as the row's player (str)
    :param prop: the property row name (str)
    :param value: the scored value, or None / NaN when the scorer returned none (float | None)
    :param status: the row status (str)
    :param caveat: what the reader needs beside the number (str)
    :returns: the row dict (dict)
    """
    row = _bare_row(game, opponent, prop, status, caveat, player=opponent)
    if not _is_undefined_value(value):
        row["value"] = row["value_pooled"] = round(float(value), 6)
    return row


def _control_arm_rows(df, game, opponent, games, cfr_policies) -> dict:
    """
    the opponent's own P3 regret and P2 best-response gap and coverage, from the same scorers
    the LM arms use, asked for the opponent's rows with `actor=`. A scorer that raises, skips the
    game, or returns an undefined value gives a MEASURABLE_FLAWED row naming why; it never gives
    a clean row with no number, and never a NOT_COMPUTED row that hides the reason
    :param df: the cell's rows, LM and opponent together (pl.DataFrame)
    :param game: the game key (str)
    :param opponent: the opponent name, such as "cfr_ref" (str)
    :param games: {game key: pyspiel game} (dict)
    :param cfr_policies: {game key: the CFR reference policy} (dict)
    :returns: {property row name: row dict} for each of `_CONTROL_ARM_SCORED_PROPERTIES`, or {}
        when `opponent` is not the scored control opponent (dict)
    """
    if opponent != _SCORED_CONTROL_OPPONENT:
        return {}
    from lmopt.scoring.property2 import score_property2
    from lmopt.scoring.property3 import score_property3

    actor = _control_actor_label(opponent)
    note = (
        f"control arm: the opponent's own rows, scored with actor={actor} by the scorer "
        "the LM arms use"
    )

    def flawed(prop, why):
        return _control_arm_row(
            game, opponent, prop, None, "MEASURABLE_FLAWED", f"{note}; {why}"
        )

    rows = {}
    if "actor" in df.columns:
        logged = sorted(df["actor"].drop_nulls().unique().to_list())
        if actor not in logged:
            for prop in _CONTROL_ARM_SCORED_PROPERTIES:
                rows[prop] = flawed(
                    prop,
                    f"actor label {actor!r} is not in the cell's actor column; found {logged}",
                )
            return rows
    try:
        p3 = score_property3(df, actor=actor, games=games, cfr_policies=cfr_policies)
        kl = p3["mean_kl"].get(game)
        if kl is None:
            rows["P3_randomisation"] = flawed(
                "P3_randomisation", "score_property3 returned no entry for this game"
            )
        elif _is_undefined_value(kl.value):
            rows["P3_randomisation"] = flawed(
                "P3_randomisation",
                "score_property3 returned an undefined value; "
                f"on_support={kl.on_support_fraction:.3f}",
            )
        else:
            rows["P3_randomisation"] = _control_arm_row(
                game,
                opponent,
                "P3_randomisation",
                kl.value,
                "MEASURABLE",
                f"{note}; on_support={kl.on_support_fraction:.3f}",
            )
            rows["P3_randomisation"]["on_support_fraction"] = kl.on_support_fraction
            rows["P3_randomisation"]["n_unmatched_rows"] = p3.get(
                "n_unmatched_rows", {}
            ).get(game)
    except Exception as e:
        rows["P3_randomisation"] = flawed("P3_randomisation", f"scorer raised: {e}")

    try:
        p2 = score_property2(df, actor=actor, games=games, cfr_policies=cfr_policies)
        mids = list(p2.get("br_gap", {}))
        if len(mids) > 1:
            for prop in ("P2_adaptivity", "P2_adaptivity_coverage"):
                rows[prop] = flawed(
                    prop,
                    f"{len(mids)} matchup ids in one control cell {mids}; refusing to pick one",
                )
            return rows
        mid = mids[0] if mids else None
        gap = p2["br_gap"].get(mid, {}).get(game) if mid is not None else None
        cov = (
            p2.get("on_support_fraction", {}).get(mid, {}).get(game)
            if mid is not None
            else None
        )
        err = p2.get("br_error", {}).get(mid, {}).get(game) if mid is not None else None
        skip = p2.get("not_applicable", {}).get(game)
        why = (
            f"BR walk raised: {err}"
            if err
            else (
                f"BR skipped: {skip}"
                if skip
                else (
                    "coverage undefined for this cell"
                    if _is_undefined_value(cov)
                    else None
                )
            )
        )
        if why is None and _is_undefined_value(gap):
            why = "no best-response gap for this cell"
        for prop, value in (("P2_adaptivity", gap), ("P2_adaptivity_coverage", cov)):
            if why is not None:
                rows[prop] = flawed(prop, why)
            else:
                rows[prop] = _control_arm_row(
                    game,
                    opponent,
                    prop,
                    value,
                    "MEASURABLE",
                    f"{note}; on_support={cov:.3f}",
                )
    except Exception as e:
        for prop in ("P2_adaptivity", "P2_adaptivity_coverage"):
            rows[prop] = flawed(prop, f"scorer raised: {e}")
    return rows


def apply_control_arm_rows(props, cells, games, cfr_policies) -> list:
    """
    replace the control opponent's NOT_COMPUTED P2 and P3 rows in a finished property table with
    rows scored from the opponent's own decisions. This runs after `property_table`, from `main`,
    and is deliberately not reachable from `property_table` or `decision_table`: the
    pre-registered decision rule stays actor-blind, and these rows are never read by it
    :param props: the rows `property_table` returned (list[dict])
    :param cells: {(game, opponent): [(shard, frame), ...]} as `property_table` takes it (dict)
    :param games: {game key: pyspiel game} (dict)
    :param cfr_policies: {game key: the CFR reference policy} (dict)
    :returns: a new list of rows, in the same order, with each scored control row merged over the
        placeholder it replaces so the cell's back-filled columns are kept (list[dict])
    """
    from lmopt.analysis.shards import concat_strict

    replaced: dict = {}
    for (game, opponent), shards in sorted(cells.items()):
        if opponent != _SCORED_CONTROL_OPPONENT or game not in games:
            continue
        df = concat_strict(
            [d for _s, d in shards],
            how="vertical_relaxed",
            what=f"{game} vs {opponent}",
        )
        for prop, row in _control_arm_rows(
            df, game, opponent, games, cfr_policies
        ).items():
            replaced[(game, opponent, prop)] = row
    out = []
    for r in props:
        key = (r["game"], r["opponent"], r["property"])
        if r["player"] == r["opponent"] and key in replaced:
            r = {**r, **replaced.pop(key)}
        out.append(r)
    return out


def _not_reached_row(
    game, opponent, prop, caveat, n_nodes=None, n_informative=None
) -> dict:
    """
    the row for a P8 cell whose arm played real rows and reached no informative node: a bare
    `NOT_REACHED` row carrying the `not_reached` approved class, so a cell with some opponents
    measured and some unreached reads `partial (approved)` and not plain `partial`.
    Only `property_table`'s zero-informative branch calls this, and that branch's own condition
    (`n_informative == 0` with `n_nodes > 0`) is what keeps the class honest: a row with no
    scored nodes at all, or one empty because rows were dropped or unparseable, never reaches
    it. The class is deliberately not set in `_bare_row`, so `AWAITING_DATA`, `NOT_COMPUTED` and
    `NOT_APPLICABLE` bare rows stay unclassified and block a cell as before
    :param game: the game key (str)
    :param opponent: the opponent name (str)
    :param prop: the P8 property row name (str)
    :param caveat: the scorer's stated reason for the zero (str)
    :param n_nodes: the real scored rows of this cell, so a reader never parses the caveat (int)
    :param n_informative: the informative nodes among them, 0 on this branch (int)
    :returns: the row dict (dict)
    """
    row = _bare_row(game, opponent, prop, "NOT_REACHED", caveat)
    row["partial_reason_class"] = NOT_REACHED_CLASS
    row["n_nodes"] = n_nodes
    row["n_informative"] = n_informative
    return row


def _property_row(
    prop,
    value,
    *,
    game,
    opponent,
    measured,
    extra="",
    force_status=None,
    caveat_override=None,
    within_seat_reason: str | None = None,
) -> dict:
    """One `property_table` row: the property's static status/caveat, resolved
    against this cell's own measured belief defects and game-family caveats.

    This is the file's most load-bearing logic (the
    `MEASURABLE_FLAWED -> MEASURABLE` downgrade gate below), kept separate
    from `property_table`'s `emit()` closure so it is a genuinely pure,
    standalone, unit-testable computation: `game`, `opponent` and `measured`
    are explicit keyword-only parameters rather than closure variables.
    `GAME_FAMILIES` is imported locally, matching this file's
    per-function convention (it is not a module-level import).

    `_property_row`'s callers (all inside `property_table`'s own `emit`
    closure) must pass the same `game`/`opponent`/`measured` that closure
    itself was defined under.

    `within_seat_reason`: this cell's
    `state_sensitivity_within_seat_reason(df)`, computed once by the same
    caller that computed `measured` and passed through unchanged. Default
    `None` is correct and inert for every property `_partial_reason_class`
    does not gate (it checks `prop in _BELIEF_SCORED_PROPERTIES` first), so
    no other call site needs updating.
    """
    from lmopt.games.families import GAME_FAMILIES

    _validate_force_status(force_status)
    _validate_caveat_override(caveat_override)
    status, caveat = PROPERTY_STATUS[prop]
    # Opt-in override (default None, no behaviour change
    # for any other property/call site), mirroring force_status's own
    # shape -- only P3_randomisation's call site below passes one,
    # to replace the static "temperature 0" caveat with one correct
    # for the run's actual configured temperature. Overrides the
    # Caveat text only, never the status.
    if caveat_override is not None:
        caveat = caveat_override
    # `GAME_CAVEATS` is
    # authored per family (leduc/kuhn/liarsdice/darkhex/goofspiel),
    # but a depth variant's exact registry string (e.g.
    # "goofspiel_5", "darkhex_3x3") is a different dict key. Resolve
    # the game to its family first and look up the caveat there;
    # fall back to the exact game string only if no family-keyed
    # entry exists (for plain game names, game == family already).
    # Without this, a variant silently loses its family's caveat and can be
    # promoted past the `MEASURABLE_FLAWED` -> `MEASURABLE` gate
    # below with a real (flawed) value and an empty caveat.
    #
    # The lookup is `derive_game_caveat`
    # against this cell's `measured`, not a `GAME_CAVEATS` literal.
    #
    # What the `or` actually does, precisely: the second
    # call is evaluated whenever the first returns `""`, which is not
    # only the depth-variant case. For a plain game `family == game`,
    # so the second call repeats the first and necessarily returns the
    # same `""` -- redundant, never divergent. For a depth variant
    # (e.g. "goofspiel_5") the second call always returns `""`,
    # because no variant-keyed entry exists in `CAVEAT_TEMPLATES`;
    # the variant gets its caveat from the first call, on the family
    # key. So the fallback never actually supplies text. It is
    # kept as the structural mirror of a `GAME_CAVEATS.get(family,
    # .get(game, ""))` lookup: an exact-game entry added later
    # would be honoured without another edit here. Note the two forms
    # are not semantically identical -- that one branches on an
    # entry's presence, this one on the rendered text's truthiness --
    # but the divergence needs both keys present with different defect
    # keys, which the family-only tables make unreachable
    # while the two tables' key sets stay equal.
    family = GAME_FAMILIES.get(game, game)
    game_caveat = derive_game_caveat(prop, family, measured) or derive_game_caveat(
        prop, game, measured
    )
    # The demotion decision reads
    # the structured blocking list, not the caveat text. The two differ
    # because the census is reporting only: a cell can
    # carry several `census:` clauses -- kuhn's trivial-prior identity,
    # leduc's raw key-set mismatch -- and still be perfectly scorable,
    # and a text-driven rule would withhold every one of those cells
    # from the verdict it should carry (all four kuhn and all four leduc
    # cells of every pilot arm). Sniffing the string for a prefix would work and is
    # deliberately not done: the gate is a computation over what was
    # measured, so it returns reasons, and the text is rendered from
    # them rather than parsed back OUT of them.
    # No family-then-exact `or` here, unlike `game_caveat` above:
    # `derive_game_block` resolves the family itself and consults no
    # game-keyed table, so the two calls are the same call. Writing the
    # fallback anyway would assert a distinction that does not exist.
    game_block = derive_game_block(prop, game, measured)
    # When this missing opponent's
    # sole blocking reason is one of the three approved ones, rewrite the
    # rendered clause to the canonical sentence and record which class
    # on the row, so `scoreboard.py` can tally approved-versus-blocking
    # without re-parsing rendered text. Admissibility (`status`) is
    # Unchanged either way -- this only changes what the reason says, never
    # whether the cell is blocked. `game_block[0]` is the exact substring
    # to replace: the classifier only proceeds past its `len(game_block) !=
    # 1` guard when `game_block` holds exactly the one `state_sensitivity_
    # cell_b_clean` clause this file's own `_render_scorability` produced.
    partial_reason_class = None
    reason_class = _partial_reason_class(
        prop, game, opponent, measured, game_block, within_seat_reason
    )
    if reason_class is not None:
        partial_reason_class, canonical_text = reason_class
        game_caveat = game_caveat.replace(game_block[0], canonical_text, 1)
    # A defect flag must name its defect. The belief defects are
    # game-specific, so a cell the defects do not touch is downgraded
    # from FLAWED to MEASURABLE rather than carrying a blank caveat --
    # otherwise the decision rule marks admissible cells INADMISSIBLE on
    # the strength of a flag with no reason behind it.
    if status == "MEASURABLE_FLAWED" and not caveat and not game_block:
        status = "MEASURABLE"
    # `force_status` is an escape hatch
    # for a single call site (P2's `br_error` path below): a raised
    # exception must not downgrade to a bare "MEASURABLE" just
    # because the property's static caveat happens to be empty --
    # that would reproduce the "crashed scorer reads as a clean
    # measurement" defect score_property2 itself guards against,
    # one layer up, in the table this function writes. Left as an
    # opt-in parameter (default None, no behaviour change for any
    # other property/call site) rather than changing the downgrade
    # rule globally, since e.g. P8_value_of_information always passes
    # a non-empty `extra` (`n_informative=...`) on a clean measurement
    # too, and folding `extra` into the downgrade condition would
    # silently stop P8 from ever downgrading.
    if force_status is not None:
        status = force_status
    parts = [caveat, game_caveat, extra]
    # `value_pooled` starts equal to `value` -- the pooled per-cell
    # estimand this function is handed -- and stays that way for any row
    # that never reaches the end-of-cell back-fill (a `configured` cell
    # with no shards emits no rows through `_property_row` at all, so
    # that case does not apply here; every real call site does reach the
    # back-fill). The back-fill is where the two diverge: it overwrites
    # `value` with the per-cell seed mean on a row whose property forms a
    # replicate interval, leaving `value_pooled` as the figure computed
    # here.
    value_field = None if _is_undefined_value(value) else round(float(value), 6)
    return {
        "game": game,
        "opponent": opponent,
        "player": "LM",
        "property": prop,
        "value": value_field,
        "value_pooled": value_field,
        "status": status,
        "caveat": "; ".join(x for x in parts if x),
        "partial_reason_class": partial_reason_class or "",
        "n_nodes": None,
        "n_informative": None,
        "n_hands": None,
        "n_distinct_prompts": None,
    }


def property_table(
    cells,
    games,
    cfr_policies,
    temperature: float = 0.0,
    exact_policy=None,
    exact_policy_provenance=None,
    exact_policy_refusal_reason=None,
    p2_hand_bins: int | None = 0,
    sentinel_anchors=None,
    run_provenance=None,
    configured=None,
):
    """Table 2: every implemented property, scored per (game, opponent) cell.

    Each scorer is a pure function over the logged rows; only the slice
    differs. Properties with no scorer are emitted as explicit UNAVAILABLE rows
    so the table always has one line per property per cell, and a missing
    measurement can never be mistaken for a passing one.

    temperature: the run's configured decoding temperature,
    read by `main()` from the sweep's own run_meta.json and threaded through
    so P3_randomisation's caveat can be correct for it (see
    `_p3_temperature_caveat`). Defaults to 0.0 -- a caller that does not
    pass it gets the "temperature 0" caveat text.

    exact_policy: the exact-policy pass's `{(game, info_set): {action: prob}}` dict --
    the plain policy mapping `score_property3_exact` itself takes (its
    signature is unchanged; no provenance string ever reaches
    `src/lmopt/scoring/`). `load_exact_policy(path).policy`, or `None`.
    Defaults to `None` -- mirroring how `temperature` is threaded. Feeds the
    `P3_randomisation_exact` diagnostic
    only; every other row is unaffected by this argument.

    exact_policy_provenance: optional
    `{"model": str | None, "git_sha": str | None, "run_model_check":
    str | None}` describing which model's pass produced `exact_policy`
    (`load_exact_policy(path).model`/`.git_sha`/`.run_model_check`, or
    `None` if unknown/not loaded). `run_model_check`
    is `"verified"` when the artefact's model was confirmed to match this
    run's own model, or `None` when that check could not be performed
    (never a value when they disagreed -- that case never reaches this
    function at all, since `load_exact_policy` refuses the whole artefact
    outright and `exact_policy` is `None` by the time it would get here).
    Purely a consumer-side annotation on the `P3_randomisation_exact` row's
    caveat -- never read by `score_property3_exact`, which stays a pure
    function of `df` and `exact_policy` alone. Defaults to `None` (no
    provenance line): a caller that only passes `exact_policy` gets the
    row without the "exact-policy from" prefix.

    exact_policy_refusal_reason:
    optional short tag (`load_exact_policy(path).refusal_reason`, e.g.
    `"missing_file"` or `"model_mismatch"`) naming why `exact_policy is
    None`, only read on that branch. Without it, "the
    artefact file genuinely does not exist" and "the artefact existed but
    was REFUSED (wrong model, or internally inconsistent)" emit the
    byte-identical caveat -- indistinguishable from `properties.csv`
    alone, with only a stderr WARNING (never part of the artefact) naming
    the real reason. Defaults to `None`, which falls back to the
    undifferentiated wording; `main()`'s own call site always
    supplies it from `load_exact_policy`'s own return value.

    p2_hand_bins: threaded straight to
    `score_property2(n_hand_bins=...)` (`property2.py:52`), which defines the
    windowed P2 trajectory (`br_gap_trajectory`/`lm_mean_trajectory`/
    `trajectory_bin_n_hands`/`trajectory_not_applicable`). Defaults to 0,
    `score_property2`'s own default, which leaves the untrajectoried keys
    byte-identical -- so a caller that never passes this argument gets a
    properties.csv identical to an explicit `p2_hand_bins=0` call. When positive, this cell's P2 block additionally emits a
    `P2_adaptivity_trajectory` row (bypassing `emit()`, the same pattern
    `P5_calibration`'s darkhex NOT_APPLICABLE row and P8's opponent-row loop
    already use for a row whose status is not decided by the static
    `PROPERTY_STATUS` table): `NOT_APPLICABLE` with the reason string when
    `score_property2` could not bin this cell (too few distinct hand indices,
    no `hand_idx` column, or an end bin holding no kept hand), `MEASURABLE_FLAWED`
    with the trajectory data JSON-encoded into `caveat` otherwise
    (`br_gap_trajectory`, `lm_mean_trajectory`, `n_hand_bins`, and the three
    per-bin counts `trajectory_bin_n_nominal`, `trajectory_bin_n_kept` and
    `trajectory_bin_dropped_rate`, each named for what it counts). Not part
    of `DECISION_RULE` -- a diagnostic, exactly like `P2_adaptivity_coverage`
    -- so `decision_table` never reads it. See `configs/p2_variance_budget.json`
    for the W (window-size) justification behind any non-zero value passed at
    a real call site.

    When binning is in play (a positive count, or `None`, the computed
    default: floor(fewest LM hands in any seed / licensed hands per bin), which
    a cell below two licensed bins cannot reach), this cell's P2 block also
    emits the `P2_adaptivity_delta` row, always one per cell: the
    pre-registered table's thresholded P2 statistic (`_trajectory_delta`: the first bin's
    mean LM reward minus the last bin's, which equals
    br_gap(last bin) - br_gap(first bin) because the target cancels; >= 0.0
    fails, DECISION_RULE entry), with per-shard replicate CIs threaded through
    `_cell_replicate_cis(p2_hand_bins=...)`. Its status comes from the cell:
    MEASURABLE when the memory rung is M1 or above and both endpoint bins hold
    the licensed hands per seed, NOT_APPLICABLE at M0, AWAITING_DATA when the
    collection is below two licensed bins, MEASURABLE_FLAWED with the reason
    otherwise. At an explicit 0 no delta row exists and decision.csv's delta
    verdict reads NOT_MEASURED.

    sentinel_anchors: the P3 sentinel anchor artefact, read once by the caller
    from `configs/anchors/p3_sentinel_anchors.json` (or a path given on the
    command line) and passed in whole. Each cell's `P3_randomisation_margin`
    row subtracts that cell's own (game, opponent) anchor mean from its
    regret, through `lmopt.analysis.sentinel_anchors.anchor_for`, and the
    margin's interval is the conservative one: the LM interval's low minus
    the anchor's high, and the LM interval's high minus the anchor's low.
    `None` (no artefact loaded), a cell the artefact has no anchor for, and a
    cell whose anchor the run's provenance voids all produce the same shape:
    an AWAITING_DATA row with no value, carrying the reason. No cell ever
    borrows another cell's anchor.

    run_provenance: the provenance of the run being scored, carrying the
    fields `sentinel_anchors.VOIDING_PROVENANCE_FIELDS` names. A field the
    run does not record does not match the artefact's, so the anchor is
    voided with a reason naming it, rather than treated as agreeing.

    configured: `configured_cells(data_dir)`'s {(game, opponent)} this arm was
    configured for, or None when that cannot be read. `cells` above holds only
    what a shard glob found, so a configured cell that lost its shard to a
    walltime kill mid-sweep never reaches the main loop below at all -- no
    row of any kind is emitted for it, identical to a game genuinely never
    configured. Every property number this table scores (2, 3, 5, 7, 8;
    1/4/6 are always UNAVAILABLE here regardless, scored instead by
    `probe_scores.py`) gets an explicit `NOT_COMPUTED` row per `(game,
    opponent)` in `configured - cells.keys()`, mirroring `probe_scores.py`'s
    `_missing_status`: the row exists so the reason reaches `properties.csv`
    and the scoreboard cell, rather than the cell reading a bare, undifferentiated
    "no {row} row" that cannot be told apart from "never configured".
    """
    from lmopt.analysis.sentinel_anchors import anchor_for
    from lmopt.analysis.shards import concat_strict
    from lmopt.games.families import GAME_FAMILIES
    from lmopt.scoring.property2 import score_property2
    from lmopt.scoring.property3 import score_property3, score_property3_exact
    from lmopt.scoring.property5 import score_property5
    from lmopt.scoring.property7 import score_property7
    from lmopt.scoring.property8 import score_property8
    from lmopt.analysis.state_sensitivity import (
        ALL_LM_ROWS_FALLBACK,
        fallback_drop_report,
        multistate_share,
        state_sensitivity,
    )
    from lmopt.analysis.state_blind_label import STATE_BLIND_COLUMN, label_for_share

    # The licensed hands per bin is read once, before any cell is scored: the
    # computed default cannot be resolved without it, and a missing artefact
    # must fail here, cleanly, rather than half way through a cell's rows.
    p2_binning = p2_hand_bins is None or p2_hand_bins > 0
    p2_floor = p2_hand_bins_licensed_floor() if p2_binning else None
    if p2_hand_bins is None and p2_floor is None:
        raise ValueError(
            f"{S2_VARIANCE_BUDGET_ARTIFACT} is missing or carries no "
            "smoke_test_licensed_w, so the computed P2 bin count has no "
            "licensed hands per bin to be derived from; pass an explicit "
            "--p2-hand-bins"
        )

    rows: list[dict[str, Any]] = []
    for (game, opponent), shards in sorted(cells.items()):
        # Where this cell's rows begin, so the back-fill below
        # (after every row this cell produces, including the two sites that
        # bypass `emit`) can slice exactly this cell's rows regardless of how
        # many properties it emitted.
        cell_start = len(rows)
        cell_dropped: dict = {}
        # Schema-migration policy (for columns such as belief_raw_completion/
        # belief_n_retries/belief_failed/model_id/chat_template_hash/
        # prompt_schema_version): RE-RUN WHOLE cells only when the schema
        # changes. `concat_strict` here is the loud site -- it raises a
        # ValueError naming the columns and the fix if one shard in this cell
        # has a different column set than another, so an old-schema shard left
        # in a cell after a partial re-run is caught, not silently mixed.
        df = concat_strict(
            [d for _s, d in shards],
            how="vertical_relaxed",
            what=f"{game} vs {opponent}",
        )
        # Measure this cell's belief defects once, here,
        # so every `emit()` below closes over the same numbers (one pass over
        # the frame, and no chance of two properties disagreeing about the
        # same cell). `emit()` renders them through `derive_game_caveat`
        # instead of reading the fixed literals out of `GAME_CAVEATS`.
        #
        # A cell whose belief data cannot be read at all is treated exactly as
        # `measure_belief_defects` treats a cell it cannot judge: every key
        # `None`, which `derive_game_caveat` renders as an explicit, NON-empty
        # "not assessable on this cell" caveat that keeps the cell
        # MEASURABLE_FLAWED and therefore INADMISSIBLE. That is the loud
        # direction (no silent fallbacks: the fallback is reported
        # in the artefact itself and as a stderr WARNING naming the exception,
        # the same shape `_read_sweep_temperature` and `load_exact_policy`
        # use). The alternative, letting it escape, would abort the
        # Whole analysis run over one cell's data.
        #
        # Three exception types, all reachable by direct
        # execution -- the guard must cover the whole class it
        # exists for:
        #   * KeyError -- a required column is absent (`measure_belief_defects`
        #     raises this itself, by its own contract).
        #   * AttributeError -- the column holds valid JSON that is not an
        #     object, so `.values()`/`.items()` has nothing to call:
        #     `"[0.5,0.5]"`, `"1.0"`, `"\"x\""`.
        #   * TypeError -- a JSON object whose values are not numbers, so
        #     `sum(...)` or `v > 0` fails: `'{"a":"notanumber"}'`.
        # Both belief columns are written by `json.dumps(dict)`, so none of
        # these is expected in practice; this is robustness against a
        # hand-edited or foreign Parquet, not a live defect.
        measured: BeliefDefects
        try:
            measured = measure_belief_defects(df, game)
            # Computed once per cell,
            # alongside `measured`, and threaded explicitly to `_property_row`
            # -- not folded into `measured`'s closed-set dict, per that
            # function's own docstring.
            within_seat_reason = state_sensitivity_within_seat_reason(df)
        except (KeyError, AttributeError, TypeError) as e:
            print(
                f"WARNING: cell ({game}, {opponent}) cannot be assessed for "
                f"belief defects ({type(e).__name__}: {e}); every belief "
                "caveat on it will say so explicitly and the cell stays "
                "INADMISSIBLE",
                file=sys.stderr,
            )
            measured = dict.fromkeys(_BELIEF_DEFECT_KEYS)
            # The cell could not be read at all, so there is no reason to
            # thread through either: `_partial_reason_class` treats `None`
            # here as "not the one cause that can be classified", which is
            # the correct, conservative outcome for an unreadable cell.
            within_seat_reason = None
            # The reason travels with the fallback dict, so the caveat states
            # the real failure instead of the per-key "not assessable"
            # sentences, whose denominators are invented in this case.
            # The `;` -> `,` substitution is load-bearing, not cosmetic:
            # `decision_table` publishes `caveat.split(";")[0]` as the blocker,
            # and `measure_belief_defects`' own KeyError text contains a
            # semicolon ("requires column 'node_idx'; present: [...]"), which
            # would truncate the blocker mid-sentence in decision.csv.
            measured[_CENSUS_ERROR_KEY] = f"{type(e).__name__}: {e}".replace(";", ",")
        g = {game: games[game]} if game in games else None
        c = {game: cfr_policies[game]} if game in cfr_policies else None

        # Per-cell shard-replicate intervals for the
        # thresholded statistics, computed once per cell from the same
        # per-seed shard frames and attached in the end-of-cell back-fill
        # below (additive `ci_lo`/`ci_hi` on every row, None wherever no
        # interval is formable -- see `_cell_replicate_cis`).
        # `None` derives this cell's bin count from its own hands per seed; an
        # explicit count (0 disables the trajectory) is used as given.
        fewest_hands = p2_hand_bins_min_hands_per_seed({None: shards}) or 0
        cell_bins, hands_gap = _resolve_p2_hand_bins(
            p2_hand_bins, fewest_hands, p2_floor
        )
        # One span for the whole cell, so the pooled row, every shard's
        # replicate and the licence's endpoint counts cut on the same edges.
        cell_span = _p2_pooled_hand_span(shards)
        cell_counts: dict = {}
        cell_means: dict = {}
        # The 7b best-response sets, shared by this cell's pooled call and its per-shard calls. One
        # dict per (game, opponent) cell. A set depends on the game and on the reference policy `c`,
        # which is game-level, so sharing across opponents of one game would also be correct: this
        # boundary is deliberately narrower than correctness requires. Do not widen it for speed:
        # a dict that outlives one
        # reference policy returns another policy's best response, wrong in value and not in cost.
        # Widen it only with a test that fails on a value when two policies share the dict.
        cell_br_cache: dict = {}
        cell_cis = _cell_replicate_cis(
            shards,
            game,
            opponent,
            g,
            c,
            p2_hand_bins=cell_bins,
            p2_hand_span=cell_span,
            counts=cell_counts,
            means=cell_means,
            br_cache=cell_br_cache,
        )
        cell_exact = _p7b_exact_cell(
            shards, df, game, g, c, exact_policy, exact_policy_refusal_reason
        )

        def emit(prop, value, extra="", force_status=None, caveat_override=None):
            # Delegates to the module-level `_property_row`
            # (defined just above `property_table`), passing this iteration's
            # closed-over `game`/`opponent`/`measured` explicitly -- see
            # `_property_row`'s own docstring for the load-bearing logic it
            # carries.
            rows.append(
                _property_row(
                    prop,
                    value,
                    game=game,
                    opponent=opponent,
                    measured=measured,
                    extra=extra,
                    force_status=force_status,
                    caveat_override=caveat_override,
                    within_seat_reason=within_seat_reason,
                )
            )

        # `p3_regret` is this cell's regret against the CFR equilibrium value
        # and `p3_absent` is the reason there is none, exactly one of the two
        # being set. The margin row below reads both: a cell with no regret
        # has no margin either, and says which of the two is missing rather
        # than emitting a bare empty row.
        p3_regret, p3_absent = None, None
        try:
            p3 = score_property3(df, games=g, cfr_policies=c)
            cell_dropped["P3"] = _game_detail(p3, "dropped_detail", game)
            kl = p3["mean_kl"].get(game)
            if kl is None:
                p3_absent = (
                    "no regret for this cell: score_property3 "
                    "returned no entry for this game"
                )
                extra = p3_absent
            elif _is_undefined_value(kl.value):
                # The coverage figure still travels: the scorer reached this
                # game and reported how much of it was on support, so the row
                # says both what it could not compute and what it saw.
                p3_absent = (
                    "no regret for this cell: score_property3 "
                    "returned an undefined value for this game"
                )
                extra = f"{p3_absent}; on_support={kl.on_support_fraction:.3f}"
            else:
                p3_regret = float(kl.value)
                extra = f"on_support={kl.on_support_fraction:.3f}"
            # A demoted status on the absent path, not the bare MEASURABLE the
            # empty static caveat would otherwise leave: a cell the scorer
            # could not score must not read like one it scored cleanly
            # (no silent fallbacks).
            emit(
                "P3_randomisation",
                p3_regret,
                extra,
                force_status=None if p3_absent is None else "MEASURABLE_FLAWED",
                caveat_override=_p3_temperature_caveat(temperature),
            )
            if kl is not None:
                # The coverage the caveat text carries, as fields: a reader of properties.csv
                # should not have to parse `on_support=` out of prose.
                rows[-1]["on_support_fraction"] = kl.on_support_fraction
                rows[-1]["n_unmatched_rows"] = p3.get("n_unmatched_rows", {}).get(game)
        except Exception as e:
            p3_absent = f"scorer raised: {e}"
            emit(
                "P3_randomisation",
                None,
                p3_absent,
                force_status="MEASURABLE_FLAWED",
                caveat_override=_p3_temperature_caveat(temperature),
            )

        # P3's verdict row: this cell's regret minus the uniform-random
        # sentinel's regret on the same (game, opponent). The lookup is by the
        # exact game key, never the family, so a depth variant asks for its own
        # anchor and reads awaiting-data rather than borrowing the base game's.
        anchor: dict[str, Any] | str
        if sentinel_anchors is None:
            anchor = (
                "no sentinel anchor artefact was loaded for this run (expected "
                f"{DEFAULT_P3_SENTINEL_ANCHORS}, or pass --p3-sentinel-anchors)"
            )
        else:
            anchor = anchor_for(sentinel_anchors, game, opponent, run_provenance or {})
        if isinstance(anchor, str):
            # The reason travels as the row's own caveat, which is what
            # `decision_table` publishes as the blocker and what the
            # scoreboard renders: a withheld cell states what it is waiting
            # for, and no other cell's anchor is ever substituted.
            rows.append(
                _bare_row(
                    game, opponent, "P3_randomisation_margin", "AWAITING_DATA", anchor
                )
            )
        elif p3_absent is not None:
            emit(
                "P3_randomisation_margin",
                None,
                f"no margin without a regret for this cell: {p3_absent}",
                force_status="MEASURABLE_FLAWED",
            )
        elif anchor.get("mean") is None:
            emit(
                "P3_randomisation_margin",
                None,
                "the sentinel anchor for this cell carries no replicate mean",
                force_status="MEASURABLE_FLAWED",
            )
        else:
            emit(
                "P3_randomisation_margin",
                p3_regret - anchor["mean"],
                f"sentinel anchor mean={anchor['mean']:.6f}, n={anchor['n']}, "
                f"interval {_interval_text(anchor.get('ci_low'), anchor.get('ci_high'))}",
            )
            # The conservative interval, built here rather than in
            # `_cell_replicate_cis` because it is not a per-shard re-score of
            # a scorer: it is the LM cell's own replicate interval shifted by
            # the anchor's, taking the widest difference either interval
            # allows. The LM must therefore clear the sentinel's whole
            # interval to read a verdict, not merely its mean. No LM interval
            # and no anchor interval both leave it unformable, which
            # `decision_table` reports as CI_UNAVAILABLE.
            lm_lo, lm_hi = cell_cis.get("P3_randomisation", (None, None))
            if (
                lm_lo is not None
                and anchor.get("ci_low") is not None
                and anchor.get("ci_high") is not None
            ):
                cell_cis["P3_randomisation_margin"] = (
                    round(lm_lo - anchor["ci_high"], 6),
                    round(lm_hi - anchor["ci_low"], 6),
                )
                # The margin row is the LM's own regret interval shifted by
                # the anchor's, so its replicate counts are the regret row's
                # own -- the anchor contributes no additional shards.
                if "P3_randomisation" in cell_counts:
                    cell_counts["P3_randomisation_margin"] = cell_counts[
                        "P3_randomisation"
                    ]
                if "P3_randomisation" in cell_means:
                    # The margin's own seed mean: the LM regret's per-cell
                    # seed mean shifted by the anchor's mean, mirroring the
                    # pooled margin's `p3_regret - anchor["mean"]` above.
                    # Recorded only when the margin's own interval could
                    # also form (the anchor's ci_low/ci_high present, gated
                    # by this same `if`): a margin row with no interval
                    # keeps `value == value_pooled`, per the value-centre
                    # rule, rather than carrying a seed mean with no
                    # interval to centre.
                    cell_means["P3_randomisation_margin"] = (
                        cell_means["P3_randomisation"] - anchor["mean"]
                    )

        try:
            # The exact-distribution KL diagnostic,
            # emitted immediately after P3_randomisation, in its own
            # try/except (a raise here must not touch the primary P3 row
            # above, which has already been emitted by this point). Its own
            # `try` covers only score_property3_exact -- `exact_policy is
            # None` (no artefact loaded) is handled first, explicitly, so
            # that case never reaches (or needs) the `except` below.
            if exact_policy is None:
                # value=None with a non-empty extra naming the absent
                # artefact -- never omitted, never a bare MEASURABLE with an
                # empty caveat. PROPERTY_STATUS["P3_randomisation_exact"]'s
                # own static caveat is already non-empty (structurally
                # immune to the downgrade rule regardless), so this extra
                # is purely for a reader of properties.csv, not a safety
                # requirement on its own.
                # "The file does not exist" and "the file exists but was
                # REFUSED (wrong model, or internally inconsistent)" must not
                # emit a byte-identical caveat here -- indistinguishable from
                # properties.csv alone, with only a stderr WARNING (never
                # part of the artefact) naming which one actually happened.
                # `_REFUSAL_TEXT` makes that distinction reach the row
                # itself; an unrecognised/absent reason falls back to the
                # undifferentiated wording (`detail = ""`), as for a
                # caller of property_table that passes no `exact_policy_
                # refusal_reason` argument at all.
                _REFUSAL_TEXT = _EXACT_ARTEFACT_REFUSAL_TEXT
                reason_text = _REFUSAL_TEXT.get(exact_policy_refusal_reason)
                detail = f" ({reason_text})" if reason_text else ""
                emit(
                    "P3_randomisation_exact",
                    None,
                    f"no exact-policy artefact was loaded for this run{detail} "
                    "(see the WARNING printed on stderr naming "
                    "the missing/unreadable path; produced by "
                    "scripts/exact_policy_pass.py)",
                )
            else:
                # Names which model's pass produced
                # `exact_policy`, so an arm that borrows another
                # model's artefact (silently, via the `--exact-policy`
                # default resolving to the wrong `out_dir`) is at least
                # visible in the row itself, not only in a WARNING printed
                # at load time. A pure annotation on the CONSUMER's emitted
                # row, never read by `score_property3_exact` itself.
                prov = ""
                if exact_policy_provenance:
                    # `run_model_check` names whether
                    # this artefact's model was actually confirmed to match
                    # This run's own model, or whether that check could not
                    # be performed at all -- the same not-observed-vs-not-
                    # set distinction drawn for `box_fields_source`.
                    # A disagreement never reaches here: `load_exact_policy`
                    # refuses the whole artefact outright on mismatch, so
                    # `exact_policy` is `None` and this whole branch is
                    # skipped in that case.
                    rmc = exact_policy_provenance.get("run_model_check")
                    prov = (
                        f"exact-policy from model="
                        f"{exact_policy_provenance.get('model')!r}, git_sha="
                        f"{exact_policy_provenance.get('git_sha')!r}, "
                        f"run_model_check={(rmc if rmc else 'not verified')!r}; "
                    )
                p3x = score_property3_exact(df, exact_policy)
                cell_dropped["P3x"] = _game_detail(p3x, "dropped_detail", game)
                kl_x = p3x["mean_kl"].get(game)
                mf = p3x["matched_fraction"].get(game)
                mf_str = "n/a" if _is_undefined_value(mf) else f"{mf:.3f}"
                n_m, n_u = p3x["n_matched"].get(game), p3x["n_unmatched"].get(game)
                if kl_x is None:
                    emit(
                        "P3_randomisation_exact",
                        None,
                        f"{prov}no matched info-sets for this cell "
                        f"(matched_fraction={mf_str}, n_matched={n_m}, "
                        f"n_unmatched={n_u})",
                    )
                else:
                    emit(
                        "P3_randomisation_exact",
                        kl_x.value,
                        f"{prov}on_support={kl_x.on_support_fraction:.3f}; "
                        f"matched_fraction={mf_str} (n_matched={n_m}, "
                        f"n_unmatched={n_u})",
                    )
        except Exception as e:
            emit("P3_randomisation_exact", None, f"scorer raised: {e}")

        try:
            # One call, two rows: `per_pair_error`
            # is emitted alongside `ece`, never instead of it, because
            # property5.py's bin-based ECE cancels signed errors within a
            # bin and the two can diverge sharply
            # on the same rows (kuhn: 0.0 vs 0.056074). Not
            # in DECISION_RULE -- that is the pre-registered
            # verdict threshold list and adding a new thresholded property
            # to it is a research-protocol decision;
            # `P5_calibration_per_pair` therefore reaches properties.csv
            # (Table 2) but not decision.csv (Table 3).
            p5 = score_property5(df)
            cell_dropped["P5"] = _game_detail(p5, "dropped_detail", game)
            # score_property5 excludes the dark-hex family from
            # Both "ece" and "per_pair_error" with an explicit
            # `not_applicable[game]` reason (property5.py, mirroring
            # property7.py's family-level skip). `force_status` cannot
            # express this: "NOT_APPLICABLE" is deliberately excluded from
            # `_FORCEABLE_STATUSES` above (emit() would raise -- it is a
            # Demote-only escape hatch, and "not applicable" is a different
            # state from "measured but flawed", the same distinction P8's
            # Dark-Hex-only branch below draws in the opposite direction).
            # So this bypasses `emit()` and appends the row directly, same
            # as that P8 branch does.
            p5_na = p5.get("not_applicable", {}).get(game)
            if p5_na:
                # The margin joins this same diversion:
                # score_property5's darkhex exclusion covers all three
                # per-game dicts (ece/per_pair_error/per_pair_margin)
                # together (property5.py's `continue` sits before any of
                # them is populated), so darkhex must get a
                # NOT_APPLICABLE margin row too, not silently no row at all.
                for prop in (
                    "P5_calibration",
                    "P5_calibration_per_pair",
                    "P5_calibration_per_pair_margin",
                ):
                    # The census clauses ride along, exactly as
                    # they do on the P7 diversion below: `posterior_zero_sum`
                    # is a real, measured darkhex quantity and a
                    # NOT_APPLICABLE row is still a row a reader reads. The
                    # family-scoped keys (every two-sided one) are suppressed
                    # by `derive_game_census` itself, so this adds the
                    # measured clause and no "not assessable" noise.
                    census = derive_game_census(
                        prop, GAME_FAMILIES.get(game, game), measured
                    )
                    rows.append(
                        _bare_row(
                            game,
                            opponent,
                            prop,
                            "NOT_APPLICABLE",
                            "; ".join([p5_na, *census]),
                        )
                    )
            else:
                # Mirrors the P7b shape immediately
                # below: `ece`/`per_pair_error` can genuinely be undefined for
                # a game this `else` branch reaches (e.g. `calibration_rows`
                # has no rows for that game -- property5.py's `if not pairs:`
                # -> nan), and P5_calibration(_per_pair)'s static caveat is
                # empty, so a game lacking a `GAME_CAVEATS` entry (any depth
                # variant -- `GAME_CAVEATS` is keyed on the exact game string,
                # not the family) would downgrade
                # straight to `status="MEASURABLE", caveat=""`. For example
                # `goofspiel_5` (a real `GAME_FAMILIES` key,
                # family "goofspiel") has no `("P5_calibration",
                # "goofspiel_5")` entry.
                # A game that does have a `GAME_CAVEATS`
                # entry (e.g. plain "goofspiel") is unaffected: `extra` is
                # only non-empty when the value itself is undefined, and the
                # existing entry is preserved as `parts[1]`, ahead of `extra`
                # in `emit()`'s join order, so its leading clause never moves.
                ece = p5["ece"].get(game)
                ece_undefined = _is_undefined_value(ece)
                emit(
                    "P5_calibration",
                    ece,
                    "" if not ece_undefined else _NO_CALIBRATION_ROWS_CAVEAT,
                )
                pp = p5["per_pair_error"].get(game)
                pp_undefined = _is_undefined_value(pp)
                emit(
                    "P5_calibration_per_pair",
                    pp,
                    "" if not pp_undefined else _NO_CALIBRATION_ROWS_CAVEAT,
                )
                # The margin, emitted alongside its two
                # siblings above, using the identical extra-when-undefined
                # shape. This is a DECISION_RULE property (unlike its
                # siblings) -- see that list's comment for the pre-registered
                # "beat-uninformed" rule this feeds.
                margin = p5["per_pair_margin"].get(game)
                margin_undefined = _is_undefined_value(margin)
                emit(
                    "P5_calibration_per_pair_margin",
                    margin,
                    "" if not margin_undefined else _NO_CALIBRATION_ROWS_CAVEAT,
                )
        except Exception as e:
            emit("P5_calibration", None, f"scorer raised: {e}")
            emit("P5_calibration_per_pair", None, f"scorer raised: {e}")
            emit("P5_calibration_per_pair_margin", None, f"scorer raised: {e}")

        try:
            p7 = score_property7(df, cfr_policies=c, games=g, br_cache=cell_br_cache)
            cell_dropped["P7a"] = _game_detail(p7, "belief_dropped_detail", game)
            cell_dropped["P7b"] = _game_detail(p7, "dropped_detail", game)
            # Same shape as the P7b block just below,
            # applied to P7a for the same reason: `tvd_mean` can genuinely
            # have no entry for a game (when `score_property7` excludes the
            # game before 7a runs)
            # or for any depth variant `GAME_CAVEATS` has no literal-string
            # entry for (e.g. "goofspiel_5"). Deliberately unconditional
            # (matches P7b, not a
            # "only if no existing caveat" guard) -- `emit()`'s `parts =
            # [caveat, game_caveat, extra]` join order means an existing
            # `game_caveat` always stays the leading (`;`-split-first)
            # clause regardless, so `decision_table`'s blocker text and
            # admissibility bucketing are unaffected either way.
            # The diversion below mirrors the P5 block above: where
            # `score_property7` declares a game not applicable and
            # populates no metric, the census un-gated cannot block, so
            # without this branch those two rows would promote to
            # MEASURABLE on a game the scorer refuses -- the
            # "skipped reads as clean" signature.
            #
            # Not keyed on `not_applicable` alone: that dict also carries
            # `score_property7`'s partial-reference skip for ordinary games
            # (property7.py), which is a P7b-only condition (7a already
            # succeeded and populated `tvd_mean[game]`) and must not divert
            # P7a. Gated instead on `game not in tvd_mean` -- true for a
            # game skipped before 7a runs and for the goofspiel
            # support-violation degrade
            # (`GoofspielSupportViolation` caught in property7.py's per-game
            # loop before `tvd_mean[game]` is set, discarding 7a for that
            # game too) -- but false for partial-reference, whose game does
            # have a `tvd_mean` entry. A family-membership gate
            # (`GAME_FAMILIES.get(game, game) in _BELIEF_NOT_APPLICABLE_FAMILIES`)
            # would miss the goofspiel reason, letting it
            # fall through to the generic "no belief-error rows" caveat
            # below and misreport a data-integrity finding as an ordinary
            # empty-data gap. Bypasses
            # `emit()` exactly as the P5 branch does, because
            # "NOT_APPLICABLE" is deliberately absent from
            # `_FORCEABLE_STATUSES` (a demote-only escape hatch, and "not
            # applicable" is a different state from "measured but flawed").
            p7_na = (
                p7.get("not_applicable", {}).get(game)
                if game not in p7.get("tvd_mean", {})
                else None
            )
            if p7_na:
                for prop in ("P7a_belief_error", "P7b_action_belief"):
                    # The census clauses ride along: `posterior_zero_sum` is a
                    # real, measured darkhex quantity (0.3696 on
                    # sdv3 qwen) and these are the only rows whose template
                    # names it, so dropping them here would delete it from the
                    # artefact entirely.
                    census = derive_game_census(
                        prop, GAME_FAMILIES.get(game, game), measured
                    )
                    rows.append(
                        _bare_row(
                            game,
                            opponent,
                            prop,
                            "NOT_APPLICABLE",
                            "; ".join([p7_na, *census]),
                        )
                    )
            else:
                tvd = p7["tvd_mean"].get(game)
                tvd_undefined = _is_undefined_value(tvd)
                emit(
                    "P7a_belief_error",
                    tvd,
                    (
                        ""
                        if not tvd_undefined
                        else "no belief-error rows for this cell (no elicited_belief/"
                        "true_posterior pairs, or the game excluded from this "
                        "scorer)"
                    ),
                )
                # `action_belief_agreement` can genuinely be nan for any
                # game (the three `agreement_per_game[game_name] = ...nan` writes
                # in `score_property7` -- empty LM belief rows, a missing
                # game/policy reference, or a BR walk that resolved zero
                # actions), but GAME_CAVEATS has no entry for every game.
                # P7b_action_belief's static caveat is also empty
                # (PROPERTY_STATUS above), so a nan reaching `emit()` with no
                # `extra` and no matching GAME_CAVEATS entry would downgrade straight
                # to `status="MEASURABLE", caveat=""` -- the "skipped/
                # crashed reads as clean" signature. Passing a non-empty `extra` whenever the value is
                # undefined closes this generally, for every game, rather than
                # adding one more hand-typed GAME_CAVEATS entry that the next new
                # game could just as easily be missing.
                agr = p7["action_belief_agreement"].get(game)
                agr_undefined = _is_undefined_value(agr)
                # An agreement figure is only as strong as the choice behind it. Where
                # every legal action attains the best value under the stated belief, the
                # row agrees whatever the model played, so the share of such rows travels
                # with the number for every scored cell, and the rows whose stated belief
                # named no best response at all are counted beside it when there are any.
                tie_share = p7.get("action_belief_all_tie_share", {}).get(game)
                no_mass = p7.get("action_belief_no_mass_rows", {}).get(game)
                p7b_clauses = []
                if agr_undefined:
                    p7b_clauses.append(
                        "no action-belief agreement rows for this cell (no LM belief "
                        "rows, missing game/policy reference, or zero resolvable "
                        "best-response actions)"
                    )
                if not _is_undefined_value(tie_share):
                    p7b_clauses.append(
                        f"every legal action ties under the stated belief on "
                        f"{tie_share * 100:.2f}% of scored rows"
                    )
                if no_mass:
                    p7b_clauses.append(
                        f"{no_mass} row(s) stated a belief with no mass on any card and "
                        f"were counted rather than scored"
                    )
                emit("P7b_action_belief", agr, "; ".join(p7b_clauses))
        except Exception as e:
            emit("P7a_belief_error", None, f"scorer raised: {e}")
            emit("P7b_action_belief", None, f"scorer raised: {e}")

        try:
            # Both games= and cfr_policies= are passed deliberately: passing
            # games without cfr_policies disables the partial-reference guard,
            # which is an unbounded tree walk on a 3x3 frame (property2.py).
            p2 = score_property2(
                df, games=g, cfr_policies=c, n_hand_bins=cell_bins, hand_span=cell_span
            )
            cell_dropped["P2"] = _p2_game_detail(p2, game)
            # `br_gap`/`on_support_fraction`/
            # `br_error` are all keyed {matchup_id: {game: ...}}, but
            # `br_error` is sparse (only matchups that raised have an entry)
            # while the other two are dense (one entry per matchup that
            # reached the loop). `next(iter(p2["br_gap"].values()))` and a
            # Separate `next(iter(p2.get("br_error", {}).values()))` can
            # therefore each pick a different matchup_id when a cell spans
            # more than one (a real arm has 4 distinct matchup_id per
            # game), pairing one matchup's
            # gap/coverage with a different matchup's error text. So one
            # matchup_id (`mid`, the first in `br_gap`) is resolved and every
            # dict indexed off
            # it, so gap, cov and err always describe the same matchup.
            mid = next(iter(p2.get("br_gap", {})), None)
            gap = (
                p2["br_gap"].get(mid, {}).get(game)
                if mid is not None and p2.get("br_gap")
                else None
            )
            # on_support_fraction reaches this row as the same
            # `on_support=` figure P3 attaches to its KL score two `emit`
            # calls above -- one call to score_property2, two things read off
            # it, mirroring the P5 per_pair_error wiring.
            cov = (
                p2["on_support_fraction"].get(mid, {}).get(game)
                if mid is not None and p2.get("on_support_fraction")
                else None
            )
            # score_property2 does not swallow
            # an in-walk exception into a bare nan -- it surfaces the reason
            # via `br_error`. Without reading it here, a raised
            # exception would reach this point as `gap=nan, cov=nan` with
            # no signal, `emit`'s own caveat-driven downgrade rule would
            # label the row a bare "MEASURABLE" with an empty caveat, and
            # the no-silent-fallbacks rule would be violated one layer up
            # from the scorer.
            err = (
                p2.get("br_error", {}).get(mid, {}).get(game)
                if mid is not None
                else None
            )
            # `score_property2` also skips a
            # cell outright (partial-reference guard, property2.py) and
            # records why in `not_applicable[game]` -- keyed by game, not
            # matchup_id, since the guard's reason is the same for every
            # matchup of that game. Unread, a skipped cell would produce the
            # identical `value=None, status='MEASURABLE', caveat=''` shape
            # as a crashed cell -- the "skipped/crashed reads as clean"
            # defect.
            skip_reason = p2.get("not_applicable", {}).get(game)
            if err:
                emit(
                    "P2_adaptivity",
                    gap,
                    f"BR walk raised: {err}",
                    force_status="MEASURABLE_FLAWED",
                )
                emit(
                    "P2_adaptivity_coverage",
                    cov,
                    f"BR walk raised: {err}",
                    force_status="MEASURABLE_FLAWED",
                )
            elif skip_reason:
                emit(
                    "P2_adaptivity",
                    gap,
                    f"BR skipped: {skip_reason}",
                    force_status="MEASURABLE_FLAWED",
                )
                emit(
                    "P2_adaptivity_coverage",
                    cov,
                    f"BR skipped: {skip_reason}",
                    force_status="MEASURABLE_FLAWED",
                )
            elif _is_undefined_value(cov):
                # The same "skipped/crashed reads as clean" signature: a
                # bare nan with neither `br_error` nor `not_applicable` set
                # would fall through to a clean `status='MEASURABLE',
                # caveat=''` row -- indistinguishable from "measured,
                # genuinely nothing to report". Practically unreachable on
                # real arms (kuhn 6/6, goofspiel_4 81/81, liarsdice
                # 12288/12288 opponent info-sets always queried), but the
                # shape is identical, so it gets the same treatment.
                #
                # Three causes can route here, not two:
                # (1) zero opponent rows for this matchup/game -> an empty
                # `emp_policy` (property2.py's `if emp_policy:` guard),
                # (2) `game_obj is None` because the game isn't in `games`,
                # or (3) a BR walk that ran and succeeded but never queried
                # the opponent's policy at all -- `_EmpiricalPolicy._seen`
                # stays empty, so `on_support_fraction` returns nan
                # (property2.py's `if not self._seen: return float("nan")`)
                # while `gap`/`br_val` can be a real number. Only causes
                # (1)/(2) produce a nan `gap`; wording that says "no BR
                # value" unconditionally would misdescribe a cell hitting
                # cause (3), where a real BR value exists and only the
                # coverage fraction is undefined.
                emit(
                    "P2_adaptivity",
                    gap,
                    "coverage undefined for this cell: no opponent rows, "
                    "no game reference, or a BR walk that never queried "
                    "the opponent's policy",
                    force_status="MEASURABLE_FLAWED",
                )
                emit(
                    "P2_adaptivity_coverage",
                    cov,
                    "coverage undefined for this cell: no opponent rows, "
                    "no game reference, or a BR walk that never queried "
                    "the opponent's policy",
                    force_status="MEASURABLE_FLAWED",
                )
            else:
                emit("P2_adaptivity", gap, f"on_support={cov:.3f}")
                emit("P2_adaptivity_coverage", cov)

            # The windowed P2 trajectory, surfaced only
            # when the caller opted in (`p2_hand_bins > 0`) -- `score_property2`
            # itself only populates these four keys under that same condition
            # (property2.py's own "byte-identical" contract), so this whole
            # block is a no-op at the default: bins-absent equals an explicit
            # bins=0 exactly.
            # Independent of whether `err`/`skip_reason` fired above:
            # `score_property2` computes the trajectory before its BR-walk
            # try/except, so a crashed or skipped BR
            # walk still leaves real trajectory data (or a real
            # not-applicable reason) behind for this cell.
            #
            # Bypasses `emit()` deliberately, the same pattern the P5
            # darkhex/P8 opponent-only NOT_APPLICABLE rows above already use:
            # this row's status is not one of PROPERTY_STATUS's two fixed
            # outcomes for "P2_adaptivity" (MEASURABLE_FLAWED, always) -- it
            # is NOT_APPLICABLE on a genuinely un-binnable cell and
            # MEASURABLE_FLAWED (diagnostic, no DECISION_RULE entry, mirroring
            # P2_adaptivity_coverage's own non-thresholded treatment)
            # otherwise. `value=None` in both cases: the payload is a JSON
            # object, not a scalar, so it travels in `caveat` rather than
            # forcing `emit()`'s float-or-None contract to bend around a list.
            # The `P2_adaptivity_delta` row is settled here too, and there is
            # always one when binning is in play (an explicit count above 0,
            # or the computed default): a number from the per-bin means when
            # every condition holds, NOT_APPLICABLE when the cell has no
            # cross-hand memory or no trajectory, AWAITING_DATA when the
            # collection is too small to window, and MEASURABLE_FLAWED with
            # the reason when a computed condition fails or the delta is
            # undefined. A missing measurement is never silently absent. With
            # an explicit 0 no delta row exists and `decision_table` reads the
            # absence as NOT_MEASURED, never a fake zero, which the >= 0.0
            # rule would read as a fail nobody measured. The pooled row
            # resolves the pooled frame's first matchup `mid`, exactly as the
            # level row above does; the per-shard replicates each resolve
            # their own (see `_cell_replicate_cis`).
            mem_state, mem_reason = _p2_memory_state(df) if p2_binning else ("ok", None)
            # A row the cell decides on its own, before any trajectory: no
            # cross-hand memory (structural), or a collection too small to
            # window. The trajectory diagnostic row is still emitted below
            # whenever bins were computed.
            delta_settled = None
            if p2_binning and mem_state == "none":
                delta_settled = ("NOT_APPLICABLE", mem_reason)
            elif p2_binning and hands_gap:
                delta_settled = ("AWAITING_DATA", hands_gap)
            if delta_settled is not None:
                rows.append(
                    _bare_row(game, opponent, "P2_adaptivity_delta", *delta_settled)
                )
            if cell_bins > 0:
                # The trajectory keys do not depend on the best-response walk,
                # so a cell whose walk was skipped (no `br_gap` entry, `mid`
                # None) still resolves its matchup from the trajectory itself.
                # A cell whose trajectory was withheld has no entry under
                # `lm_mean_trajectory` either, and its reason is the one the
                # row must carry, so the withheld-reason dict is the last
                # place looked: without it a skipped walk hides the real
                # reason behind the skip's own text.
                tmid = mid
                for key in ("lm_mean_trajectory", "trajectory_not_applicable"):
                    if tmid is not None:
                        break
                    tmid = next(iter(p2.get(key, {})), None)
                traj_na = (
                    p2.get("trajectory_not_applicable", {}).get(tmid, {}).get(game)
                    if tmid is not None
                    else None
                )
                if traj_na:
                    rows.append(
                        _bare_row(
                            game,
                            opponent,
                            "P2_adaptivity_trajectory",
                            "NOT_APPLICABLE",
                            traj_na,
                        )
                    )
                    if delta_settled is None:
                        rows.append(
                            _bare_row(
                                game,
                                opponent,
                                "P2_adaptivity_delta",
                                "NOT_APPLICABLE",
                                traj_na,
                            )
                        )
                else:
                    means = (
                        p2.get("lm_mean_trajectory", {}).get(tmid, {}).get(game)
                        if tmid is not None
                        else None
                    )
                    traj_gap = (
                        p2.get("br_gap_trajectory", {}).get(tmid, {}).get(game)
                        if tmid is not None
                        else None
                    )
                    # A cell the BR-gap partial-reference guard skips has no
                    # `br_gap_trajectory` and no `trajectory_not_applicable`;
                    # the guard's own reason lives in `not_applicable`, keyed
                    # by game only. Read it rather than dropping the row: a
                    # missing measurement must never be mistaken for a passing
                    # one, or simply absent (no silent fallbacks).
                    na_reason = (
                        p2.get("not_applicable", {}).get(game)
                        or "trajectory not computed for this cell and "
                        "score_property2 recorded no reason"
                    )
                    if traj_gap is not None:
                        rows.append(
                            _bare_row(
                                game,
                                opponent,
                                "P2_adaptivity_trajectory",
                                "MEASURABLE_FLAWED",
                                json.dumps(
                                    {
                                        "n_hand_bins": cell_bins,
                                        "br_gap_trajectory": p2["br_gap_trajectory"][
                                            tmid
                                        ][game],
                                        "lm_mean_trajectory": means,
                                        # Three counts, each named for what it counts:
                                        # the hands the bin nominally holds, the hands
                                        # its mean was taken over, and the share
                                        # between them the fallback drop removed.
                                        "trajectory_bin_n_nominal": p2.get(
                                            "trajectory_bin_n_hands", {}
                                        )
                                        .get(tmid, {})
                                        .get(game),
                                        "trajectory_bin_n_kept": p2.get(
                                            "trajectory_bin_n_kept", {}
                                        )
                                        .get(tmid, {})
                                        .get(game),
                                        "trajectory_bin_dropped_rate": p2.get(
                                            "trajectory_bin_dropped_rate", {}
                                        )
                                        .get(tmid, {})
                                        .get(game),
                                    }
                                ),
                            )
                        )
                    else:
                        rows.append(
                            _bare_row(
                                game,
                                opponent,
                                "P2_adaptivity_trajectory",
                                "NOT_APPLICABLE",
                                na_reason,
                            )
                        )
                    if delta_settled is None and means is None:
                        rows.append(
                            _bare_row(
                                game,
                                opponent,
                                "P2_adaptivity_delta",
                                "NOT_APPLICABLE",
                                na_reason,
                            )
                        )
                    elif delta_settled is None:
                        # The thresholded delta, through emit() so it gets the
                        # end-of-cell CI back-fill like every thresholded
                        # property. It is read from the per-bin means, so a
                        # failed or skipped best-response walk leaves it
                        # defined. A cell that fails a computed condition
                        # keeps its number and is demoted with the reason as
                        # the caveat; nan -> value=None with the reason named,
                        # never a bare nan read as "measured, no signal".
                        delta = _trajectory_delta(means)
                        block = mem_reason or _p2_hands_per_bin_reason(
                            _p2_thinnest_endpoint_bin_hands(shards, cell_bins), p2_floor
                        )
                        if _is_undefined_value(delta):
                            emit(
                                "P2_adaptivity_delta",
                                None,
                                "delta undefined: fewer than 2 bins or an "
                                "empty endpoint bin at either end of the "
                                "trajectory",
                                force_status="MEASURABLE_FLAWED",
                            )
                        elif block:
                            emit(
                                "P2_adaptivity_delta",
                                delta,
                                block,
                                force_status="MEASURABLE_FLAWED",
                            )
                        else:
                            emit(
                                "P2_adaptivity_delta",
                                delta,
                                P2_DELTA_NOTE.format(bins=cell_bins),
                            )
        except Exception as e:
            emit("P2_adaptivity", None, f"scorer raised: {e}")
            emit("P2_adaptivity_coverage", None, f"scorer raised: {e}")
            if p2_binning:
                # A `score_property2` crash must not drop the trajectory and
                # delta rows when the caller asked for binning: both are
                # emitted with the failure as their reason. The delta's status
                # is forced, because its static caveat is empty and a
                # value-less row would otherwise read as a clean measurement.
                rows.append(
                    _bare_row(
                        game,
                        opponent,
                        "P2_adaptivity_trajectory",
                        "NOT_APPLICABLE",
                        f"scorer raised: {e}",
                    )
                )
                emit(
                    "P2_adaptivity_delta",
                    None,
                    f"scorer raised: {e}",
                    force_status="MEASURABLE_FLAWED",
                )

        try:
            p8 = score_property8(df, cfr_policies=c, games=g)
            cell_dropped["P8"] = _game_detail(p8, "dropped_detail", game)
            if p8.get("available"):
                # Capture rate is the
                # Headline and the shortfall sits beside it, so it is emitted
                # First and this ordering is the readout ordering a reader of
                # properties.csv sees. No number changes: both were already
                # computed, and `DECISION_RULE` deliberately still carries its
                # pre-registered threshold on the shortfall alone -- a threshold
                # for the capture rate is not pre-registered, and inventing one
                # here would be an unregistered verdict.
                # Asked directly of each named clause, not inferred from a side
                # count. `_p8_occupancy_clause` is structurally unavoidable
                # for Dark Hex -- an information-optimal CFR+ reference
                # avoids exactly the nodes P8 scores -- and restricting the
                # scored population to where the reference plays was
                # measured and rejected (the whole informative population carries
                # 0.0-1e-6 of the reference's own visit mass on every
                # board/seat), so it reports as a non-blocking `extra`
                # instead of a `caveat_override` -- the same mechanism the
                # existing `n_informative=...` extra already uses to reach
                # the rendered caveat text without tripping the
                # MEASURABLE_FLAWED -> MEASURABLE downgrade.
                #
                # `_p8_range_clause` does not block `P8_capture_rate` either:
                # capture_rate cannot be the thing the range defect is
                # a defect in. Proof, read at property8.py's own scoring
                # loop (`captures.append(int(best - v_lm <= epsilon))`,
                # guarded above by `if voi <= voi_threshold: continue`): a
                # node reaches `shortfalls`/`captures` only once
                # voi > voi_threshold = 1e-2, so shortfall > 1 forces
                # `best - v_lm > voi_threshold = 1e-2 >> epsilon = 1e-6`,
                # which makes `captures.append(0)` mandatory at every node
                # `n_shortfall_gt1` counts. A range-violating node is
                # therefore always already a capture-rate failure, never a
                # capture-rate value the defect could have corrupted --
                # the clause names a defect in the shortfall ratio's own
                # bounds, one `P8_capture_rate`'s definition does not share.
                # `P8_value_of_information` (the shortfall) keeps the
                # caveat_override: the clause is squarely about
                # that row, even though `DECISION_RULE` carries
                # no shortfall entry, so this
                # row's status is properties.csv-display-only and is not
                # what decision.csv's admissibility depends on.
                n_inf = (p8.get("n_informative") or {}).get(game)
                p8_range = _p8_range_clause(p8, game, n_inf) if n_inf else None
                p8_occupancy = _p8_occupancy_clause(p8, game, n_inf) if n_inf else None
                # n_inf == 0 gets its own clause, naming
                # Which zero this is (arm-specific vs board-level vs no
                # data) rather than a bare "n_informative=0" that would
                # read identically to score_property8's own "all-degenerate
                # board" case -- see _p8_zero_informative_clause's docstring.
                p8_zero = _p8_zero_informative_clause(p8, game) if n_inf == 0 else None
                p8_extra = p8_zero if p8_zero is not None else f"n_informative={n_inf}"
                if p8_occupancy:
                    p8_extra = f"{p8_extra}; {p8_occupancy}"
                p8_capture_extra = p8_extra
                if p8_range:
                    p8_capture_extra = f"{p8_capture_extra}; {p8_range}"
                if n_inf == 0 and (p8.get("n_nodes") or {}).get(game):
                    # Real rows were scored and none reached an informative node: the
                    # property has no value to report, and the scorer says so in its own
                    # status word. A MEASURABLE row with an empty value would contradict
                    # the finite-value gate and read BROKEN.
                    for _prop in ("P8_capture_rate", "P8_value_of_information"):
                        rows.append(
                            _not_reached_row(
                                game,
                                opponent,
                                _prop,
                                p8_extra,
                                (p8.get("n_nodes") or {}).get(game),
                                n_inf,
                            )
                        )
                else:
                    emit(
                        "P8_capture_rate",
                        p8["capture_rate"].get(game),
                        p8_capture_extra,
                    )
                    emit(
                        "P8_value_of_information",
                        p8["p8_shortfall"].get(game),
                        p8_extra,
                        caveat_override=p8_range,
                    )
            else:
                # "Not applicable" is a different state from "measured but
                # flawed", and collapsing them would let an absent measurement
                # read as a defective one. But `available=False` collapses at
                # least three distinct causes -- this game's family isn't Dark
                # Hex at all; it is Dark Hex but no reference was supplied for
                # it (e.g. darkhex_3x3); or a reference
                # exists but is partial -- and a hardcoded "P8 is defined for
                # Dark Hex only" is wrong for the latter two, mislabelling a
                # missing-artefact problem as a game-applicability one. Same
                # pattern P2's own `not_applicable` consultation uses above:
                # `score_property8` records the specific reason for both
                # missing-reference states and for
                # the partial-reference state; consult it first,
                # and only fall back to the game-applicability message when
                # score_property8 genuinely has nothing to say about this game
                # (because its family isn't Dark Hex, which score_property8
                # never even considers).
                p8_na = p8.get("not_applicable", {}).get(game)
                p8_na_reason = p8_na or "P8 is defined for Dark Hex only"
                for _prop in ("P8_capture_rate", "P8_value_of_information"):
                    rows.append(
                        _bare_row(game, opponent, _prop, "NOT_APPLICABLE", p8_na_reason)
                    )
        except Exception as e:
            emit("P8_capture_rate", None, f"scorer raised: {e}")
            emit("P8_value_of_information", None, f"scorer raised: {e}")

        # Bound before the try, not inside it: `for` does not
        # scope its body, so without this a raise from `mask_constant_share`
        # would leave `cell_mask_constant_share` holding the previous cell's
        # value, and the back-fill below would silently mislabel this cell
        # with someone else's number instead of naming the absence via
        # `label_for_share(None)`.
        #
        # Deliberately its own name, not `share` --
        # `share` is reassigned three try-blocks down (`share, denom =
        # multistate_share(df)`, the different multistate-restricted
        # statistic:
        # 512,945/661,644 = 0.7753 mask-constant vs 269,902/398,406 = 0.6775
        # multistate-restricted). The back-fill below reads this name at the
        # END of the per-cell body, by which point a same-named `share`
        # would already hold the later statistic -- the exact "right
        # number, wrong object" defect this rename exists to make
        # structurally unreachable, not just fixed once.
        cell_mask_constant_share = None
        # `cell_mask_constant_share`
        # stays `None` on both the legitimate "no LM rows in this cell" path
        # and a raise from `mask_constant_share` below -- `label_for_share`
        # cannot tell them apart from the value alone. This second variable
        # carries the raise's own reason so the back-fill can name it
        # correctly instead of defaulting to "no LM rows", which would name a
        # cause that did not occur. `None` here means "no raise happened",
        # not "no reason exists" -- the no-LM-rows case has its own reason
        # inline below and needs no separate variable.
        cell_mask_constant_none_reason = None
        cell_all_fallback = False
        try:
            # The state-sensitivity diagnostic. It reads
            # the raw log rather than calling a scorer, so it sits after every
            # scored property and depends on none of them; its own try/except
            # keeps a raise here from touching any row already emitted above.
            #
            # Over one three-arm cycle, pooled by (arm,
            # game) -- not this per-cell block's own unit, see
            # mask_constant_share's docstring: 496,659 of 661,644 LM rows
            # (75.1 %) have an action that is a constant function of
            # `legal_mask` alone. Nothing else in properties.csv says so --
            # every one of those rows is logged as an ordinary decision node --
            # so without this row the fact is left for a human to notice.
            #
            # A raise reaches the row as `value=None` plus a caveat naming it,
            # never as a fallback number (no silent fallbacks): 0.0
            # is this statistic's cleanest reading, so a silent fallback here
            # would read as a perfectly state-sensitive cell.
            ss_drop = fallback_drop_report(df)
            cell_dropped["SS"] = ss_drop.detail()
            cell_mask_constant_share = mask_constant_share(df)
            cell_all_fallback = bool(ss_drop.n_lm_rows) and ss_drop.kept.is_empty()
            if cell_mask_constant_share is not None:
                mask_constant_caveat = ""
            elif cell_all_fallback:
                mask_constant_caveat = ALL_LM_ROWS_FALLBACK
                cell_mask_constant_none_reason = ALL_LM_ROWS_FALLBACK
            else:
                mask_constant_caveat = (
                    "no LM rows in this cell, so the action's dependence on the "
                    "legal mask could not be measured"
                )
            emit(
                "state_sensitivity_mask_constant",
                cell_mask_constant_share,
                mask_constant_caveat,
            )
        except Exception as e:
            cell_mask_constant_none_reason = f"state-sensitivity diagnostic raised: {e}"
            emit(
                "state_sensitivity_mask_constant", None, cell_mask_constant_none_reason
            )

        try:
            # Effective-n companion. Its own try/except: a raise
            # here must not disturb `state_sensitivity_mask_constant`, which
            # has already been emitted, and must reach the row as value=None
            # plus a caveat naming the raise (no silent fallbacks).
            share, denom = multistate_share(df)
            reason = (
                ""
                if share is not None
                else (
                    ALL_LM_ROWS_FALLBACK
                    if cell_all_fallback
                    else "no legal-mask bin in this cell held two or more distinct "
                    "prompts, so the action's dependence on the state could not "
                    "be separated from its dependence on the legal mask"
                )
            )
            # Deliberate asymmetry
            # on the no-multi-state-bin branch, not an oversight: `share` is
            # `None` because a RATIO over an empty set is undefined -- 0.0
            # would be an invented perfect result. `denom` stays a real `0.0`
            # (never `None`) because a COUNT over an empty set is a true
            # measured fact: zero rows fell in a multi-state bin. This is the
            # project's own "measured zero vs unmeasurable" distinction
            # (`mask_constant_share`'s own docstring, above); collapsing both
            # to `None` would destroy it.
            # Both rows carry the same non-empty `reason` on this branch, so
            # the `0.0` can never be misread as a clean measurement.
            emit("state_sensitivity_multistate_share", share, reason)
            emit("state_sensitivity_multistate_rows", denom, reason)
        except Exception as e:
            emit(
                "state_sensitivity_multistate_share",
                None,
                f"state-sensitivity effective-n diagnostic raised: {e}",
            )
            emit(
                "state_sensitivity_multistate_rows",
                None,
                f"state-sensitivity effective-n diagnostic raised: {e}",
            )

        # Bound before the `try`, not inside it: `except` runs with whatever
        # the body left behind, so the handler's guard must exist even if the
        # very first statement of the body raised.
        _emitted_within_seat: set[str] = set()
        try:
            # The within-seat share and every denominator,
            # emitted beside the pooled rows above, never in place of them:
            # the two shares answer different questions (census vs
            # state-blindness claim) and the pooled row is published science
            # whose value must not move.
            #
            # One `state_sensitivity(df)` call for all six rows: the share and
            # its denominator must come from the same pass, or a cell could
            # publish a ratio against a count that was never its own, and the
            # cell is binned once rather than once for the share and again
            # for its reason.
            #
            # Its own try/except for the same reason every block here has one:
            # a raise must not touch the rows already emitted above, and must
            # reach these rows as value=None plus a caveat naming the raise
            # rather than as a number (no silent fallbacks).
            #
            # `within_seat_reason` is None exactly when the share is not, so
            # the pair can never be a real number beside an excuse, nor a None
            # beside an empty caveat. On a schema-v2 arm (no `lm_seat` column
            # at all) the reason names the missing column: the quantity was
            # not measured there, and repeating the pooled number would import
            # the seat confound this row exists to remove.
            #
            # Deliberately not named `share`: the back-fill at the end of this
            # per-cell body reads `cell_mask_constant_share`, and the block
            # above documents what rebinding `share` here would cost.
            st = state_sensitivity(df)
            # The counts are measured facts even where the share is undefined
            # (a COUNT over an empty set is 0; a RATIO over one is not 0.0 --
            # the same rule that governs the pooled pair). Only
            # the absence of `lm_seat` leaves the two within-seat counts with
            # nothing to count over, and that reaches them as None plus the
            # reason, never as a 0 that would read as a measurement.
            _within_seat_absent = (
                ""
                if st.rows_multistate_within_seat is not None
                else st.within_seat_reason or ""
            )
            # The six rows are built before any is
            # emitted, and each is recorded in `_emitted` as it goes out, so
            # the handler below can re-emit only what never made it. Without
            # that bookkeeping a raise landing part-way through would emit the
            # already-published rows a second time, and properties.csv would
            # carry one cell twice for the same property, once with a value
            # and once with a refusal. Unreachable on the current code (the only
            # raise site, `state_sensitivity(df)`, precedes every emit), which
            # is exactly why it is worth making structural: the next line
            # added between two emits would make it reachable silently.
            for _prop, _value, _extra in (
                (
                    "state_sensitivity_multistate_share_within_seat",
                    st.share_within_seat,
                    st.within_seat_reason or "",
                ),
                ("state_sensitivity_bins_total", st.bins_total, ""),
                (
                    "state_sensitivity_bins_multistate_pooled",
                    st.bins_multistate_pooled,
                    "",
                ),
                (
                    "state_sensitivity_rows_multistate_pooled",
                    st.rows_multistate_pooled,
                    "",
                ),
                (
                    "state_sensitivity_bins_multistate_within_seat",
                    st.bins_multistate_within_seat,
                    _within_seat_absent,
                ),
                (
                    "state_sensitivity_rows_multistate_within_seat",
                    st.rows_multistate_within_seat,
                    _within_seat_absent,
                ),
            ):
                emit(_prop, _value, _extra)
                _emitted_within_seat.add(_prop)
        except Exception as e:
            _raised = f"within-seat state-sensitivity diagnostic raised: {e}"
            for _prop in _WITHIN_SEAT_KEYS:
                if _prop not in _emitted_within_seat:
                    emit(_prop, None, _raised)

        for prop in ("P1_invariance", "P4_payoff_sensitivity", "P6_subgame_perfection"):
            emit(prop, None)

        # The opponent arm the control would need. Recorded as explicit rows so
        # its absence is visible in the table rather than inferred from a gap.
        for prop, reason in (
            (
                "P3_randomisation",
                "computable from parsed_action + cfr_policy; scorer filters "
                "opponent rows out (property3.py)",
            ),
            (
                "P3_randomisation_margin",
                "computable from parsed_action + cfr_policy + the sentinel anchor; scorer "
                "filters opponent rows out (property3.py)",
            ),
            (
                "P3_randomisation_exact",
                "computable from the exact-policy artefact + cfr_policy; "
                "scorer filters opponent rows out (property3.py)",
            ),
            ("P5_calibration", "impossible: elicited_belief null on all opponent rows"),
            (
                "P5_calibration_per_pair",
                "impossible: elicited_belief null on all opponent rows",
            ),
            (
                "P5_calibration_per_pair_margin",
                "impossible: elicited_belief null on all opponent rows",
            ),
            (
                "P7a_belief_error",
                "impossible: elicited_belief null on all opponent rows",
            ),
            (
                "P7b_action_belief",
                "impossible: elicited_belief null on all opponent rows",
            ),
            (
                "P2_adaptivity",
                "needs a seat-aware BR and opponent-side empirical policy",
            ),
            (
                "P2_adaptivity_coverage",
                "needs a seat-aware BR and opponent-side empirical policy",
            ),
            ("P8_capture_rate", _P8_OPPONENT_ROW_REASON),
            ("P8_value_of_information", _P8_OPPONENT_ROW_REASON),
        ):
            # The only `_bare_row` site that
            # needs `player=opponent`; every other site keeps `_bare_row`'s
            # default `player="LM"`.
            rows.append(
                _bare_row(game, opponent, prop, "NOT_COMPUTED", reason, player=opponent)
            )

        # Label propagation: the label must travel with the
        # number, not in a separate row a reader has to join by hand.
        # `cell_mask_constant_share` (not the multistate-restricted `share`
        # rebound above, a different statistic over a different denominator
        # -- see the note at its declaration) is a property of the cell, so
        # it back-fills onto every row this cell produced -- including the
        # sites that bypass `emit`
        # (P8 NOT_APPLICABLE, and the opponent-row loop's
        # NOT_COMPUTED rows). Placed at the END of the per-cell body, after
        # every `rows.append`/`emit` call this cell makes (not "immediately
        # after state_sensitivity_mask_constant", which would run before the
        # multistate-share/rows, P1/P4/P6 and opponent rows exist and leave
        # them without the column) -- `rows[cell_start:]` at this point is
        # exactly this cell's rows, no more and no fewer.
        #
        # A NEW column, never the `caveat` string:
        # a non-empty caveat suppresses the MEASURABLE_FLAWED -> MEASURABLE
        # downgrade (the `if status == "MEASURABLE_FLAWED"` line in
        # `_property_row`) and would flip P5-margin/P7a/P7b
        # from LM_FAILS_CRITERION to INADMISSIBLE, i.e. turn this label
        # into an admissibility gate, which it deliberately is not.
        # The two counts a thin cell is explained by, from the rows the state-sensitivity
        # check itself reads: hands played, and distinct prompts among them.
        # None, never a guess, when the frame lacks the columns.
        ss_kept = fallback_drop_report(df).kept if "prompt" in df.columns else None
        cell_n_hands = (
            ss_kept.select(["seed", "hand_idx"]).unique().height
            if ss_kept is not None and {"seed", "hand_idx"} <= set(ss_kept.columns)
            else None
        )
        cell_n_prompts = ss_kept["prompt"].n_unique() if ss_kept is not None else None
        for r in rows[cell_start:]:
            if r["player"] == "LM" and r["property"] in _BELIEF_SCORED_PROPERTIES:
                r["n_hands"] = cell_n_hands
                r["n_distinct_prompts"] = cell_n_prompts
            # The row's value AS COMPUTED above (by `emit`/`_property_row`
            # or a direct `_bare_row`/margin append) is the per-cell pooled
            # estimand; it is published unchanged as `value_pooled` before
            # `value` itself is possibly overwritten below with the seed
            # mean recorded in `cell_means` for this property. `cell_means`
            # only ever carries an entry for a property whose own replicate
            # interval formed -- `_cell_replicate_cis` records one exactly
            # when it records the matching interval, and the margin's
            # hand-rolled entry above is gated the same way. So a
            # property whose interval could not form, and every opponent
            # row (`cell_means` is an LM-only population), leaves `value`
            # untouched, and the two stay equal: `value`/`value_pooled`
            # coincide wherever no interval forms.
            r["value_pooled"] = r["value"]
            # Sparse columns: filled only on the P3 regret row above, None on every other row
            # (write_csv takes its field names from the first row and refuses a later row with
            # an extra key).
            r.setdefault("on_support_fraction", None)
            r.setdefault("n_unmatched_rows", None)
            seed_mean = cell_means.get(r["property"]) if r["player"] == "LM" else None
            if seed_mean is not None and r["value"] is not None:
                r["value"] = round(float(seed_mean), 6)
            # Additive nullable interval columns on
            # Every row (write_csv's uniform-key contract), real bounds only
            # on an LM row whose property carries a computed interval. An
            # opponent row never gets one: the per-shard recomputation above
            # scores the LM population, and pinning another population's
            # interval to these rows would be a right-number-wrong-object
            # defect.
            r["ci_lo"], r["ci_hi"] = (
                cell_cis.get(r["property"], (None, None))
                if r["player"] == "LM"
                else (None, None)
            )
            # Additive nullable replicate-count columns, the same
            # uniform-key contract as ci_lo/ci_hi above: real counts only on
            # an LM row whose property carries a computed interval, None
            # everywhere else. A zero-width interval formed over more than
            # one seed states its own reason in the caveat, rather than
            # reading like an ordinary tight interval with nothing behind it.
            n_replicates, n_distinct = (
                cell_counts.get(r["property"], (None, None))
                if r["player"] == "LM"
                else (None, None)
            )
            r["n_replicates"] = n_replicates
            r["n_replicates_distinct"] = n_distinct
            # Additive nullable distribution-basis columns on every row (the same uniform-key
            # contract): P7b's headline taken on the exact policy, the sampled `value` left as it
            # was. Only an LM P7b row that has a sampled value carries them; `exact_reason` says why
            # there is no exact value there, and is blank where there is one.
            on_p7b = (
                r["player"] == "LM"
                and r["property"] == "P7b_action_belief"
                and r["value"] is not None
            )
            r["value_exact"] = cell_exact["value"] if on_p7b else None
            r["ci_low_exact"] = cell_exact["ci"][0] if on_p7b else None
            r["ci_high_exact"] = cell_exact["ci"][1] if on_p7b else None
            r["exact_reason"] = (
                (cell_exact["reason"] or None)
                if on_p7b and cell_exact["value"] is None
                else None
            )
            if n_distinct == 1:
                zero_variance_clause = (
                    f"zero between-seed variance: 1 distinct value across "
                    f"{n_replicates} seeds"
                )
                r["caveat"] = "; ".join(
                    x for x in (r["caveat"], zero_variance_clause) if x
                )
            # The drop counts are a property of the LM's rows only, so an
            # opponent row and a property with no single backing scorer carry
            # None in every column rather than a number about another object.
            r.update(
                _dropped_fields(
                    cell_dropped.get(_dropped_source(r["property"]))
                    if r["player"] == "LM"
                    else None
                )
            )
            r[STATE_BLIND_COLUMN] = label_for_share(
                cell_mask_constant_share, none_reason=cell_mask_constant_none_reason
            )
            # The float travels
            # alongside the rendered label, under a private (leading "_")
            # key `write_csv` excludes from properties.csv. Consumed by
            # `decision_table` below so the admissible-range computation
            # never has to regex-reparse `label_for_share`'s "NN.NN%" text
            # -- a label built for a human reader is not this statistic's
            # interface, and (for a `none_reason` label) is free text that
            # could coincidentally contain a percent-shaped substring a
            # regex would silently misread as a share.
            r[_STATE_BLIND_SHARE_KEY] = cell_mask_constant_share
    # A configured (game, opponent) with no cell in `cells` never reaches the loop
    # above at all, so it gets zero rows of any kind -- identical to a game genuinely
    # never configured. Mirrors `probe_scores.py`'s `_missing_status`:
    # every property this table scores (never 1/4/6, always UNAVAILABLE here regardless)
    # gets an explicit NOT_COMPUTED row naming the gap, so `properties.csv` and the
    # scoreboard cell state it rather than falling through to an undifferentiated
    # "no {row} row".
    if configured is not None:
        from lmopt.analysis.scoreboard.scoreboard import headline_rows

        missing_configured = sorted(set(configured) - set(cells.keys()))
        if missing_configured:
            heads = headline_rows(DECISION_RULE)
            names = [name for number in (2, 3, 5, 7, 8) for name in heads[number]]
            for game, opponent in missing_configured:
                reason = (
                    f"the run was configured for {game} against {opponent} but wrote "
                    "no rows; check shard_outcomes.json for a lost shard before "
                    "assuming this game was never collected"
                )
                for name in names:
                    rows.append(_bare_row(game, opponent, name, "NOT_COMPUTED", reason))
            # The writer fills a column a row lacks with a blank, whatever the reason it is
            # missing. A placeholder row states every column the table has, as None, so the
            # blank is written here on purpose and not by the writer's default.
            columns = list(rows[0])
            for row in rows:
                for column in columns:
                    row.setdefault(column, None)
    return rows


def write_csv(path: Path, rows):
    if not rows:
        print(f"  skipped {path}: the table has no rows, so no file was written")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Leading-underscore keys (e.g. _STATE_BLIND_SHARE_KEY) are internal hand-offs between table-building
    # functions, not published columns -- excluded here rather than never
    # attached to the row, so the row dict stays the single source `main()`
    # both writes and prints from.
    #
    # Filtered PER ROW before the DictWriter ever sees them, with
    # `extrasaction` left at its default (raise): `extrasaction="ignore"`
    # would silently drop any unexpected key, not just the underscore-prefixed
    # ones, in a module that allows no silent fallbacks. A row with
    # a key absent from `rows[0]` therefore raises `ValueError`. A row that
    # lacks a key present in `rows[0]` is not caught: its cell is written
    # blank. Empty `rows` writes no file.
    filtered_rows = [
        {k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows
    ]
    fieldnames = list(filtered_rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(filtered_rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def print_table(title, rows, cols):
    print(f"\n=== {title} ===")
    widths = {c: max([len(c), *(len(str(r.get(c, ""))) for r in rows)]) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print(
            "  ".join(
                str(r.get(c, "") if r.get(c) is not None else "").ljust(widths[c])
                for c in cols
            )
        )


def _read_sweep_temperature(data_dir: Path) -> float:
    """Read the SWEEP's own run_meta.json (written by run_sweep.py's
    write_run_meta, in `data_dir`) for `seed_policy.temperature`.

    This is a different run_meta.json from the one `main()` itself writes
    into `out_dir` below (analysis-level metadata, not sweep provenance).
    Defaults to 0.0 (the de facto value before temperature was recorded and
    the value every such sweep's run_meta.json would have carried had it recorded the
    truth) when the file is absent or does not carry the key, e.g. a sweep
    run before this file existed, or `--data` pointed at a hand-assembled
    directory with no run_meta.json at all -- printed as a WARNING rather
    than silently assumed, since a wrong default here would misreport the
    real caveat as if it were measured (no silent fallbacks).
    """
    meta_path = data_dir / "run_meta.json"
    if not meta_path.exists():
        print(
            f"WARNING: no run_meta.json in {data_dir}; assuming temperature=0.0 "
            "for the P3_randomisation caveat (may be wrong for this run)",
            file=sys.stderr,
        )
        return 0.0
    try:
        meta = json.loads(meta_path.read_text())
        return float(meta["seed_policy"]["temperature"])
    # OSError: meta_path.exists() being True is not a
    # guarantee that .read_text() succeeds -- a permissions error, a broken
    # symlink, or a race where the file is removed/replaced between the
    # exists() check and the read all raise OSError, which would otherwise
    # crash analyse_stats.py entirely on an existing-but-
    # unreadable file instead of warning and defaulting like every other
    # failure mode this function handles.
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        print(
            f"WARNING: {meta_path} unreadable/missing seed_policy.temperature "
            f"({e}); assuming temperature=0.0 for the P3_randomisation caveat",
            file=sys.stderr,
        )
        return 0.0


def _load_sentinel_anchors(path: Path):
    """
    read the committed P3 sentinel anchor artefact, or say why there is none
    An unreadable or absent artefact is a WARNING and `None`, never a silent
    empty artefact: `property_table` turns `None` into an AWAITING_DATA margin
    row per cell that names the missing file, so the gap reaches the table a
    reader reads rather than only the terminal (no silent fallbacks).
    :param path: the artefact's path (Path)
    :returns: the parsed artefact, or None (dict | None)
    """
    if not path.exists():
        print(
            f"WARNING: no P3 sentinel anchor artefact at {path}; every "
            "P3_randomisation_margin row will read AWAITING_DATA",
            file=sys.stderr,
        )
        return None
    try:
        artefact = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"WARNING: {path} could not be read ({e}); every "
            "P3_randomisation_margin row will read AWAITING_DATA",
            file=sys.stderr,
        )
        return None
    if not isinstance(artefact, dict) or "anchors" not in artefact:
        print(
            f"WARNING: {path} carries no 'anchors' block; every "
            "P3_randomisation_margin row will read AWAITING_DATA",
            file=sys.stderr,
        )
        return None
    return artefact


def _read_run_provenance(data_dir: Path) -> dict:
    """
    the fields of the SWEEP's own `run_meta.json` that a sentinel anchor is
    checked against, read from `data_dir` (the same file
    `_read_sweep_temperature` reads, not the analysis `run_meta.json` this
    script writes into `--out`)
    Two blocks are read, because the run records two kinds of fact: what the
    operator asked for lives in `resolved_config` (the reference iteration
    counts, the seat mode) and what the collecting box actually was lives in
    `environment` (the OpenSpiel and numpy versions, written by
    `write_run_meta`). The field names are disjoint, so the two are merged.
    A field the sweep does not record is ABSENT from the returned mapping, not
    filled in from this machine or from a default: the analysing box is not
    the collecting box, and its installed OpenSpiel says nothing about which
    one solved the reference the run played against. `anchor_for` then reads
    the absence as a disagreement and voids the anchor with a reason naming
    the field, which is the loud direction.
    `seat_mode` is shape-checked once it is read. A P3 run is never a
    game-disjoint composite, so a per-game dict here is itself the wrong
    shape for this consumer (mirrors `contrast_arms._read_seat_mode`'s
    policy); either that or a genuinely unexpected shape (a list, say) is
    treated the same as an absent field -- printed as a WARNING and left out
    of the returned mapping, so `anchor_for` voids the anchor loudly rather
    than this function crashing the whole analysis run over one bad
    `run_meta.json` among many being processed.
    :param data_dir: the sweep's data directory (Path)
    :returns: whichever of the voiding provenance fields the run records (dict)
    """
    from lmopt.analysis.condition import (
        ResolvedConfigShapeError,
        parse_seat_mode_field,
    )
    from lmopt.analysis.sentinel_anchors import VOIDING_PROVENANCE_FIELDS

    meta_path = data_dir / "run_meta.json"
    if not meta_path.exists():
        print(
            f"WARNING: no run_meta.json in {data_dir}; no sentinel anchor "
            "can be matched to this run and every P3_randomisation_margin "
            "row will read AWAITING_DATA",
            file=sys.stderr,
        )
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"WARNING: {meta_path} unreadable ({e}); no sentinel anchor can "
            "be matched to this run",
            file=sys.stderr,
        )
        return {}
    recorded = {
        **(meta.get("resolved_config") or {}),
        **(meta.get("environment") or {}),
    }
    found = {f: recorded[f] for f in VOIDING_PROVENANCE_FIELDS if f in recorded}
    if "seat_mode" in found:
        try:
            seat_mode = parse_seat_mode_field(
                found["seat_mode"], context=str(meta_path)
            )
            if isinstance(seat_mode, dict):
                raise ResolvedConfigShapeError(
                    f"{meta_path}: resolved_config['seat_mode'] is a per-game record "
                    f"{seat_mode!r}, but a P3 sentinel run is never a game-disjoint "
                    "composite and this consumer reads one seat mode per run"
                )
        except ResolvedConfigShapeError as exc:
            print(
                f"WARNING: {exc}; treated as an unrecorded seat_mode so the "
                "anchor is voided rather than checked against a value that "
                "cannot be trusted",
                file=sys.stderr,
            )
            del found["seat_mode"]
    missing = [f for f in VOIDING_PROVENANCE_FIELDS if f not in found]
    if missing:
        print(
            f"WARNING: {meta_path} records no {', '.join(missing)}; the "
            "sentinel anchors state a value for each, so their cells will "
            "read AWAITING_DATA rather than assume agreement",
            file=sys.stderr,
        )
    return found


def _read_run_model_id(data_dir: Path) -> str | None:
    """Read the SWEEP's own run_meta.json (`data_dir`, the same file
    `_read_sweep_temperature` reads) for `serving.model_id` -- the
    by-construction half of `load_exact_policy`'s cross-model
    guard.

    Returns `None` (never raises, see below) whenever the
    run's own model cannot be determined: `run_meta.json` missing or
    unreadable, no top-level `serving` key at all (or a `serving` key that
    is present but itself FALSY -- an empty dict `{}`, `null`, `0`, `""` --
    treated identically to "absent", not the "present but incomplete" case
    below; `run_meta.write_run_meta` never actually writes an empty dict,
    it omits the key entirely, so this exact shape is not
    producible by this repo's own writer, but is handled this way
    deliberately rather than left to diverge from this docstring for a
    hand-authored or future `run_meta.json`), or a NON-empty `serving`
    block with no `model_id`. A missing (or falsy) `serving` key is not
    itself a defect -- it is the "not set" contract
    (`run_meta.write_run_meta`'s docstring: the key is OMITTED entirely
    for a run with no resolved serving info, e.g. a non-LM sweep) -- so
    that case alone is not printed as a WARNING; the other two (file
    missing/unreadable, or a non-empty `serving` block with no `model_id`)
    are printed, since those describe a run that does carry serving info
    but not enough of it, or a run whose provenance file itself could not
    be read at all.

    This function performs no refusal on its own -- it only answers "what
    is this run's own model, if knowable at all". The refuse-or-not
    decision (comparing this against the exact-policy artefact's own
    `model`) lives in `load_exact_policy`, which is the function that
    already owns every other refusal this loader can make.

    A `run_meta.json` whose top level parses to
    something other than a JSON object (a bare array/string/number -- all
    valid JSON) would raise `AttributeError` from a bare `meta.get("serving")`
    call, taking `main()` down with it. Same failure mode if `serving`
    itself is present, truthy, but not an object (e.g. a string or a
    non-empty list). `_read_sweep_temperature`, the function this one
    mirrors, degrades to a WARNING on exactly
    this shape of input (its own `except` tuple includes `TypeError`) --
    this function does too, rather than raising.
    """
    meta_path = data_dir / "run_meta.json"
    if not meta_path.exists():
        print(
            f"WARNING: no run_meta.json in {data_dir}; cannot verify the "
            "exact-policy artefact's model against this run's own model "
            "(run_model_check will read 'not verified')",
            file=sys.stderr,
        )
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"WARNING: {meta_path} unreadable ({e}); cannot verify the "
            "exact-policy artefact's model against this run's own model "
            "(run_model_check will read 'not verified')",
            file=sys.stderr,
        )
        return None
    # `.get()` on either `meta` or
    # `serving` assumes a dict -- a top-level (or `serving`-value) shape
    # that is not one raises AttributeError, which a bare `.get()` chain
    # does not catch. Both calls live inside one try/except, mirroring
    # `_read_sweep_temperature`'s own degrade-to-WARNING contract exactly.
    try:
        serving = meta.get("serving")
        if not serving:
            # Not set, not not-observed (the "not set" contract) -- expected
            # for e.g. a non-LM sweep or a run_meta.json predating the
            # serving block; not a defect on its own, so no WARNING here.
            return None
        model_id = serving.get("model_id")
    except (AttributeError, TypeError) as e:
        print(
            f"WARNING: {meta_path} has an unexpected shape ({e}); cannot "
            "verify the exact-policy artefact's model against this run's "
            "own model (run_model_check will read 'not verified')",
            file=sys.stderr,
        )
        return None
    if not model_id:
        print(
            f"WARNING: {meta_path}'s serving block carries no model_id; "
            "cannot verify the exact-policy artefact's model against "
            "this run's own model (run_model_check will read "
            "'not verified')",
            file=sys.stderr,
        )
        return None
    return model_id


def default_exact_policy_path(data_dir: Path, out_dir: Path) -> Path:
    """
    where the exact-policy artefact is looked for when `--exact-policy` is not given: the
    arm's own `exact_policy.csv` when its data directory holds one, else the analysis output
    directory's. A composite arm carries the file in its data directory, and the output
    directory is a different one for every scored arm
    :param data_dir: the arm directory `--data` names (Path)
    :param out_dir: the analysis output directory `--out` names (Path)
    :returns: `data_dir / "exact_policy.csv"` when that file exists, else
        `out_dir / "exact_policy.csv"`, which is then reported missing by
        `load_exact_policy` if it does not exist either (Path)
    """
    arm_own = data_dir / "exact_policy.csv"
    return arm_own if arm_own.is_file() else out_dir / "exact_policy.csv"


def _line_buffer_stdout() -> None:
    """
    make progress lines appear as they are printed, so a long scoring run under `nohup` shows
    where it is: stdout is block-buffered when it is not a terminal
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(line_buffering=True)


def main(argv=None):
    _line_buffer_stdout()
    args = parse_args(argv)
    data_dir, out_dir = Path(args.data), Path(args.out)

    # Refuse a colliding overwrite before any work happens. See
    # lmopt.analysis.output_guard's own docstring for why.
    from lmopt.analysis.output_guard import refuse_non_empty_out_dir
    from lmopt.analysis.property_rows import lm_rows

    if refuse_non_empty_out_dir(out_dir, args.allow_overwrite):
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(data_dir)
    configured = configured_cells(data_dir)
    if configured is None:
        # Without the configured set, a cell lost to a killed shard falls back to the
        # undifferentiated "no {row} row" this run cannot tell from never-configured.
        # Said out loud rather than silently assumed (no silent fallbacks).
        print(
            f"WARNING: {data_dir / 'run_meta.json'} does not name this run's "
            "configured games and opponents (missing, unreadable, or a composed "
            "piece is); a configured cell that wrote no rows will read as a bare "
            "BROKEN, not as a lost shard, in this run's properties.csv",
            file=sys.stderr,
        )
    temperature = _read_sweep_temperature(data_dir)

    print("\nBuilding outcome table...")
    outcomes = outcome_table(cells, args.bootstrap)
    write_csv(out_dir / "outcomes.csv", outcomes)
    print_table(
        "1. OUTCOMES per (game, opponent)",
        outcomes,
        [
            "game",
            "opponent",
            "n_hands",
            "wins",
            "losses",
            "draws",
            "win_rate",
            "win_rate_pooled",
            "win_rate_se_binomial",
            "mean_reward",
            "mean_reward_pooled",
            "reward_ci_lo",
            "reward_ci_hi",
            "reward_ci_method",
            "reward_ci_note",
            "reward_ci_lo_hand",
            "reward_ci_hi_hand",
            "n_distinct_outcomes",
            "n_distinct_playouts",
            "playout_key_blind",
            "n_effective_shards",
            "flag",
            "dropped_rate",
            "n_dropped",
            "n_total",
            "dropped_unit",
            "n_shards_all_dropped",
            "mean_reward_all_hands",
        ],
    )

    props = []
    decisions: list = []
    # The reference pickle each loaded game's yardstick came from, by content hash; empty when no
    # reference was loaded (`--skip-properties`).
    reference_records: dict[str, dict] = {}
    if not args.skip_properties:
        # Checked before any
        # scoring runs: --p2-hand-bins is a BIN COUNT, but a value that
        # implies too few hands per bin against s2_variance_budget.json's
        # licensed floor is refused loudly here rather than silently
        # producing an under-powered trajectory downstream.
        validate_p2_hand_bins(args.p2_hand_bins, p2_hand_bins_min_hands_per_seed(cells))
        # Inside this branch because `exact_policy` is only ever read by
        # `property_table` below, which `--skip-properties` skips entirely.
        # Loading it unconditionally would print a spurious "no exact-policy
        # artefact" WARNING even for a `--skip-properties` run that never
        # scores a single property.
        exact_policy_path = (
            Path(args.exact_policy)
            if args.exact_policy
            else default_exact_policy_path(data_dir, out_dir)
        )
        # The run's own model, read from data_dir's
        # run_meta.json (serving.model_id) -- the same data_dir
        # `_read_sweep_temperature` above already reads, not `out_dir`
        # (this script's own --out, a separate analysis directory).
        # `load_exact_policy` uses this to refuse an artefact attributed
        # to a different model than this run's own; `None` (run_meta.json
        # missing/unreadable, no serving block, or no model_id) means the
        # check cannot be performed at all, printed as its own WARNING by
        # `_read_run_model_id` and surfaced downstream as "not verified"
        # rather than silently treated as passed.
        run_model_id = _read_run_model_id(data_dir)
        exact_policy_load = load_exact_policy(
            exact_policy_path, run_model_id=run_model_id
        )
        exact_policy = exact_policy_load.policy

        # Function-local because these pull
        # in pyspiel, which is absent on the Mac control plane.
        from lmopt.games.loaders import GAMES
        from lmopt.games.policy_cache import announce_cache_state, load_or_solve_cfr
        from lmopt.analysis.environment import reference_record

        # Only for the `print_table` column list below --
        # `property_table` imports its own copy locally, unaffected by this.
        from lmopt.analysis.state_blind_label import STATE_BLIND_COLUMN

        games, cfr_policies = {}, {}
        for game in sorted({g for g, _o in cells}):
            if game not in GAMES:
                print(
                    f"  WARNING: unknown game '{game}', no CFR reference",
                    file=sys.stderr,
                )
                continue
            # Announce cache HIT vs miss rather than a generic
            # "Loading/solving" line.
            announce_cache_state(
                game, iters=args.cfr_iters, cache_dir=args.cfr_cache_dir
            )
            games[game] = GAMES[game]()
            cfr_policies[game] = load_or_solve_cfr(
                games[game],
                args.cfr_iters,
                plus=True,
                cache_dir=args.cfr_cache_dir,
                require_hit=args.require_cache_hit,
            )
            reference_records[game] = reference_record(
                games[game], game, args.cfr_iters, args.cfr_cache_dir
            )
            print("  done")

        print("\nScoring properties per cell...")
        from lmopt.scoring.property7 import ExactScorerNotM0

        try:
            props = property_table(
                cells,
                games,
                cfr_policies,
                temperature=temperature,
                exact_policy=exact_policy,
                exact_policy_provenance=(
                    {
                        "model": exact_policy_load.model,
                        "git_sha": exact_policy_load.git_sha,
                        "run_model_check": exact_policy_load.run_model_check,
                    }
                    if exact_policy is not None
                    else None
                ),
                # Only meaningful (and only read)
                # on the `exact_policy is None` branch -- see property_table's
                # own docstring for why this reaches the emitted row.
                exact_policy_refusal_reason=exact_policy_load.refusal_reason,
                p2_hand_bins=args.p2_hand_bins,
                # Read once for the whole table, from the same `data_dir` the
                # temperature comes from: every cell's margin is then checked
                # against one artefact and one run provenance, so two cells can
                # never disagree about which sentinel run they were compared to.
                sentinel_anchors=_load_sentinel_anchors(Path(args.p3_sentinel_anchors)),
                run_provenance=_read_run_provenance(data_dir),
                configured=configured,
            )
        except ExactScorerNotM0 as exc:
            print(f"\nABORT: memory level not M0 -- {exc}", file=sys.stderr)
            return EXIT_EXACT_NOT_M0
        props = apply_control_arm_rows(props, cells, games, cfr_policies)
        lm_props = lm_rows(props)
        write_csv(out_dir / "properties.csv", props)
        print_table(
            "2. PROPERTIES per (game, opponent), LM rows",
            lm_props,
            [
                "game",
                "opponent",
                "property",
                "value",
                "dropped_rate",
                "dropped_unit",
                "status",
                "caveat",
                STATE_BLIND_COLUMN,
            ],
        )

        decisions = decision_table(props)
        write_csv(out_dir / "decision.csv", decisions)
        print_table(
            "3. PRE-REGISTERED DECISION RULE",
            decisions,
            [
                "property",
                "fail_threshold",
                "observed_mean_all_cells",
                "observed_mean_admissible",
                "n_cells_admissible",
                "n_cells_blocked",
                "dropped_rate",
                "dropped_unit",
                "verdict",
                "blocker",
                STATE_BLIND_COLUMN,
            ],
        )

    # The shared provenance block, identical across
    # every analysis entrypoint, in the same run_meta.json this export's
    # properties.csv/decision.csv are keyed to. `bootstrap_seed` is read from
    # `bootstrap_mean_ci`'s own default rather than typed as 0: this entrypoint
    # never passes a seed, and a literal here would go stale silently the day
    # one is threaded through.
    import inspect

    from lmopt.analysis.condition import (
        CONDITION_KEY,
        DIAGNOSTIC_KEY,
        SOURCE_ANALYSE_STATS,
        ConditionUnavailable,
        condition_block,
        load_collection_meta,
    )
    from lmopt.analysis.environment import (
        PROVENANCE_RECORDS_KEY,
        environment_record,
        merge_run_meta,
    )

    meta = {
        # Repo-relative when the data dir is inside the
        # repo, absolute otherwise: this file is read back by
        # the cross-arm loader, which resolves a relative value
        # against the repo root. Writing `str(args.data)` verbatim would record a
        # cwd-relative path that only resolves from the directory the run
        # happened to start in.
        "data_dir": _display_data_dir(data_dir),
        "n_cells": len(cells),
        "cfr_iters": args.cfr_iters,
        "reference_pickles": reference_records,
        "bootstrap": args.bootstrap,
        "min_distinct_outcomes": MIN_DISTINCT_OUTCOMES,
        "property_status": {k: v[0] for k, v in PROPERTY_STATUS.items()},
    }
    # The condition these tables analysed, copied from the collection run_meta so a reader
    # of the analysis output alone can tell which memory level, hand count and probe role
    # the numbers belong to. A collection that recorded none leaves the key absent and says
    # so: every consumer then reads the output as not stating its condition, which is what
    # is true, rather than assuming one.
    collection_meta = None
    try:
        collection_meta = load_collection_meta(data_dir)
        meta[CONDITION_KEY] = condition_block(
            collection_meta,
            SOURCE_ANALYSE_STATS,
            row_prompt_schema_versions(cells),
        )
    except ConditionUnavailable as exc:
        # A collection that carries the diagnostic key must never be exported without the
        # marker: an export with no block reads as an arm that states no condition, and the
        # key's presence is what the scoreboard refuses on. So it stops here.
        if collection_meta is not None and DIAGNOSTIC_KEY in (
            collection_meta.get("resolved_config") or {}
        ):
            raise SystemExit(
                f"REFUSED: {exc}; this collection is marked diagnostic, and an export that "
                "cannot record the marker would read as an unmarked arm"
            )
        print(f"WARNING: {exc}; this export will not state which condition it analysed")
    # This export's descriptive keys are replaced wholesale (they describe this
    # run), but the `provenance_records` map is carried over first: it holds
    # the blocks other passes write for
    # Their artifacts, which sit in this same directory and are still there.
    meta_path = out_dir / "run_meta.json"
    if meta_path.exists():
        try:
            previous = json.loads(meta_path.read_text())
        except Exception:  # a corrupt previous file is replaced
            previous = {}
        if isinstance(previous, dict) and isinstance(
            previous.get(PROVENANCE_RECORDS_KEY), dict
        ):
            meta[PROVENANCE_RECORDS_KEY] = previous[PROVENANCE_RECORDS_KEY]
    meta_path.write_text(json.dumps(meta, indent=2))
    merge_run_meta(
        out_dir / "run_meta.json",
        "scripts/analyse_stats.py",
        environment_record(
            out_dir,
            bootstrap_seed=inspect.signature(bootstrap_mean_ci)
            .parameters["seed"]
            .default,
            n_resamples=args.bootstrap,
        ),
        extra={
            "artifacts": [
                name
                for name, rows in (
                    ("properties.csv", props),
                    ("decision.csv", decisions),
                    ("outcomes.csv", outcomes),
                )
                if rows
            ]
        },
    )
    print(f"\nWrote tables to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
