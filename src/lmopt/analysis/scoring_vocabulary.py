"""The scoring vocabulary: the property status table, the pre-registered decision rule, the
verdict words, the decision table builder, the exact-policy loader and the shard-name parser.

A library module cannot import `scripts/analyse_stats.py`, so these live here and every reader
(the scoreboard and the off-path scorer among them) imports the one definition.
`scripts/analyse_stats.py` re-exports every name below.
"""

from __future__ import annotations

import dataclasses
import math
import re
import sys
from pathlib import Path
from typing import NamedTuple

#: The columns of the three scored CSVs `scripts/analyse_stats.py` writes, in the order they appear.
#: `write_csv` takes its header from the first row's keys, so a test pins these to what it writes.
OUTCOMES_COLUMNS = (
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
PROPERTIES_COLUMNS = (
    "game",
    "opponent",
    "player",
    "property",
    "value",
    "value_pooled",
    "status",
    "caveat",
    "partial_reason_class",
    "n_nodes",
    "n_informative",
    "n_hands",
    "n_distinct_prompts",
    "on_support_fraction",
    "n_unmatched_rows",
    "ci_lo",
    "ci_hi",
    "n_replicates",
    "n_replicates_distinct",
    "value_exact",
    "ci_low_exact",
    "ci_high_exact",
    "exact_reason",
    "dropped_rate",
    "n_dropped",
    "n_total",
    "dropped_unit",
    "n_infosets",
    "n_infosets_fully_dropped",
    "fallback_row_rate",
    "cell_lm_mask_constant_share",
)
DECISION_COLUMNS = (
    "property",
    "estimand",
    "fail_threshold",
    "direction",
    "observed_mean_all_cells",
    "observed_mean_admissible",
    "ci_lo",
    "ci_hi",
    "observed_mean_admissible_executed",
    "n_cells_admissible",
    "n_cells_blocked",
    "dropped_rate",
    "dropped_unit",
    "status",
    "verdict",
    "verdict_reason",
    "basis",
    "basis_reason",
    "blocker",
    "blocker_class",
    "cell_lm_mask_constant_share",
)

# Scoring status per property. UNAVAILABLE entries cannot be produced by any log
# the current harness emits: `probe_type` is a hard-coded literal
# (harness/match_runner.py) and tests pin the column to {"none"}, so the probe
# conditions P1/P4/P6 need are not produced by this path.
PROPERTY_STATUS = {
    # The caveat says what is true of this entrypoint,
    # not what is true of the data: the probe
    # conditions are collected, they are simply not read here. P1/P4/P6 are
    # scored for the board by `scripts/probe_scores.py` (probe_scores.csv), which
    # verifies each arm with `arm_reading.read_arm` and calls the paired scorers.
    "P1_invariance": (
        "UNAVAILABLE",
        "not scored by this entrypoint: the relabelled-action probe "
        "condition is scored by scripts/probe_scores.py",
    ),
    # br_value comes from OpenSpiel's imported `BestResponsePolicy`, which pools
    # over information sets rather than taking `max` per state. The residual
    # caveat: br_value is a best response to a hybrid
    # policy (empirical where the opponent's info-set was logged, uniform
    # elsewhere), so its trustworthiness depends on coverage -- reported as
    # the P2_adaptivity_coverage row and (in properties.csv, not here) the
    # `on_support=` figure attached to each P2_adaptivity row.
    #
    # This text is `decision_table`'s blocker string for
    # every P2_adaptivity cell (`r["caveat"].split(";")[0]` at
    # `decision_table`, i.e. everything before the first ";" -- the
    # per-cell `on_support=` figure `emit` appends after a ";" never
    # survives into decision.csv). Kept self-contained and semicolon-free here on purpose so
    # a reader of decision.csv alone, not properties.csv, still gets an
    # accurate sentence.
    # The level has no decision.csv row (see DECISION_RULE's omissions list),
    # so this text does not reach decision.csv. `P2_adaptivity_delta` does not
    # share it: the best-response target cancels out of the delta, which
    # blocks only on its own computed reasons (`derive_p2_delta_status`).
    "P2_adaptivity": (
        "MEASURABLE_FLAWED",
        "br_value is a best response to a "
        "hybrid (empirical + uniform-fallback) opponent policy "
        "-- per-cell coverage is reported separately as the "
        "P2_adaptivity_coverage row",
    ),
    # Diagnostic companion to P2_adaptivity, same rows/scorer call (one
    # `score_property2` call emits both) -- not in DECISION_RULE below: it is
    # a coverage denominator, not itself a thresholded optimality claim.
    "P2_adaptivity_coverage": ("MEASURABLE_FLAWED", ""),
    # The thresholded P2 statistic: the shrinkage delta, br_gap(late bin) -
    # br_gap(early bin), early = the first bin and late = the last. The best-
    # response value is pinned across bins, so the target cancels and the delta
    # equals the first bin's mean LM reward minus the last bin's: the hybrid-
    # target caveat that governs the level row does not describe it. The
    # static caveat is empty on purpose. Whether a cell is admissible is
    # computed per cell (`derive_p2_delta_status`: memory rung, hands per
    # endpoint bin), and a blocked cell carries its own reason as the caveat;
    # a non-empty static caveat would hold every cell at MEASURABLE_FLAWED
    # whatever was measured. Emitted only when binning is on (an explicit
    # count or the computed default); the unwindowed artifact of record has no
    # delta row, which `decision_table` reads as NOT_MEASURED, never a zero.
    "P2_adaptivity_delta": ("MEASURABLE_FLAWED", ""),
    # The row reports regret against the CFR equilibrium value
    # (`score_property3`): at most 0, and 0 is equilibrium-optimal. That
    # statistic is defined on whatever actions the run logged, so the decoding
    # temperature is provenance on it, not a confound of it -- the caveat
    # `property_table` supplies through `caveat_override` names the configured
    # temperature and claims nothing further (`_p3_temperature_caveat`). The
    # row is reported-only: P3's verdict is carried by
    # `P3_randomisation_margin`, which is the row `DECISION_RULE` lists, the
    # same split P5 draws between its raw per-pair error and its margin.
    "P3_randomisation": ("MEASURABLE", ""),
    # This cell's regret minus the uniform-random sentinel's own regret on the
    # Same (game, opponent), read from the committed anchor artefact. The
    # status is MEASURABLE_FLAWED with an empty caveat, exactly as
    # `P5_calibration_per_pair_margin`'s is: no belief defect blocks a regret
    # row, so `derive_game_block` returns nothing and `_property_row`'s
    # downgrade gate promotes a clean cell to MEASURABLE. A cell with no
    # usable anchor never reaches that gate: `property_table` emits it as an
    # AWAITING_DATA row carrying the lookup's own reason instead.
    "P3_randomisation_margin": ("MEASURABLE_FLAWED", ""),
    # The exact-distribution KL diagnostic
    # (score_property3_exact) - a diagnostic beside P3_randomisation, not a
    # replacement for it: parsed_action stays the primary scored quantity. Status is
    # deliberately not "MEASURABLE": it has no pre-registered threshold (no
    # DECISION_RULE entry -- see the comment there), and the exact
    # distribution is renormalised over the legal action labels only (a
    # confound of its own, distinct from P3_randomisation's).
    #
    # The exact-policy pass is per-arm: each arm's
    # `properties.csv` row names its own verified provenance via the
    # dynamic `run_model_check='verified'` clause `property_table` appends
    # per row (see `load_exact_policy`). What remains is a
    # Population caveat: arms resolve different, endogenous
    # info-set populations (e.g. 185/133/127 distinct
    # info-sets, three-way intersection 55), so a cross-arm comparison of
    # this diagnostic confounds randomisation quality with reachability:
    # cross-arm `coverage_mass` and `P3_randomisation_exact` are not
    # comparable across arms.
    #
    # A non-empty static caveat here (unlike P3_randomisation's own
    # MEASURABLE_CONFOUNDED sibling, which is also non-empty) makes this
    # row structurally immune to emit()'s MEASURABLE_FLAWED -> MEASURABLE
    # downgrade, mirroring P3_randomisation's own protection. The caveat
    # below must stay non-empty for that reason -- do not shorten it to "".
    "P3_randomisation_exact": (
        "MEASURABLE_FLAWED",
        "diagnostic with no pre-registered threshold; the exact "
        "distribution is renormalised over the legal action labels only; "
        "cross-arm comparison is confounded by endogenous, per-arm info-set "
        "populations",
    ),
    "P4_payoff_sensitivity": (
        "UNAVAILABLE",
        "not scored by this entrypoint: the payoff-rescaling probe condition "
        "is scored by scripts/probe_scores.py",
    ),
    # A NON-empty static caveat, which is what makes
    # this row structurally immune to `emit()`'s MEASURABLE_FLAWED ->
    # MEASURABLE promotion below -- the same protection `P2_adaptivity` and
    # `P3_randomisation_exact` already carry, and for the same reason: the row
    # reports a quantity whose defect is a property of the metric, not of the
    # arm, so no per-arm measurement can clear it.
    #
    # The defect: `score_property5`'s bin-based ECE bins by stated probability,
    # then compares bin-mean stated to bin-mean true and takes the absolute
    # value only after that averaging (property5.py's bin loop, `ece +=
    # (count/total)*abs(mean_s - mean_t)`). Signed errors therefore cancel
    # Inside a bin, so a small ECE is evidence of cancellation exactly as
    # readily as of calibration. On liarsdice/always_bet,
    # `P5_calibration` reads 0.0 against `P5_calibration_per_pair` 0.076932 on the
    # Same rows. The mechanism is pinned by a standing test,
    # tests/test_scorer_property5.py::TestPerPairCalibrationErrorProductionPath
    # ::test_kuhn_arm_level_ece_stays_zero_but_per_pair_error_reveals_binning_
    # cancellation; this caveat stops the artefact of that mechanism from
    # reaching a verdict.
    #
    # Without it, in analysis/a3-cycle-20260806-postfix/*/decision.csv
    # `P5_calibration` would publish LM_MEETS_CRITERION on qwen (observed 0.0)
    # and olmo (0.001804) while `P5_calibration_per_pair_margin` -- the
    # PRE-registered rule, over the identical 4 admissible cells and the
    # identical rows -- returns LM_FAILS_CRITERION (0.003644 and 0.003287):
    # two rows of one decision.csv contradicting each other, with the
    # contradiction resolved in the hypothesis' favour by the row whose 0.0 is
    # a binning artefact.
    #
    # Deliberately not a removal from `DECISION_RULE` below: which properties
    # carry a pre-registered threshold is a research-protocol decision (see
    # that list's own comment). The row keeps its
    # threshold, its number and its place in decision.csv; what it loses is the
    # ability to be counted admissible and therefore to carry a verdict.
    #
    # Semicolon-free on purpose, exactly as `P2_adaptivity`'s text above is:
    # `decision_table` uses `r["caveat"].split(";")[0]` as decision.csv's
    # blocker, so anything after a ";" never reaches a reader of that file.
    "P5_calibration": (
        "MEASURABLE_FLAWED",
        "bin-based ECE averages stated and true probability WITHIN a bin "
        "before taking the absolute difference, so signed errors cancel "
        "inside a bin and a near-zero value can mean cancellation rather "
        "than calibration (measured on liarsdice/always_bet: ECE 0.0 against "
        "per-pair error 0.076932 on the same rows) -- the pre-registered "
        "calibration verdict is P5_calibration_per_pair_margin, not this row",
    ),
    # Reports property5.py's per-pair
    # calibration metric, computed on the same rows as P5_calibration --
    # not part of the pre-registered DECISION_RULE below (see the emit
    # call site's comment), so it carries no decision.csv verdict.
    "P5_calibration_per_pair": ("MEASURABLE_FLAWED", ""),
    # Reports property5.py's margin
    # (per_pair_error - uniform_baseline_per_pair on the same rows). Unlike
    # its two siblings above, this one is in DECISION_RULE below: a relative
    # "beat-uninformed" threshold is pre-registered for it specifically,
    # while the raw ECE/per-pair-error properties remain un-thresholded
    # diagnostics.
    "P5_calibration_per_pair_margin": ("MEASURABLE_FLAWED", ""),
    "P6_subgame_perfection": (
        "UNAVAILABLE",
        "not scored by this entrypoint: the off-path driven condition is "
        "scored by scripts/probe_scores.py",
    ),
    "P7a_belief_error": ("MEASURABLE_FLAWED", ""),
    "P7b_action_belief": ("MEASURABLE_FLAWED", ""),
    # Capture rate is the headline P8
    # readout and the shortfall sits beside it, so it is emitted first at the
    # call site. The shortfall is a reported, unthresholded row.
    # Both P8 rows share one caveat, computed per cell by `p8_caveat()`.
    "P8_capture_rate": ("MEASURABLE_FLAWED", ""),
    "P8_value_of_information": ("MEASURABLE_FLAWED", ""),
    # A diagnostic about the log itself, not one of the
    # eight optimality properties -- hence the lower-case name, which no P-
    # numbered row shares, and hence no DECISION_RULE entry below (see that
    # list's own deliberate-omissions comment: an emitted metric with no
    # pre-registered threshold gets a row in properties.csv and no verdict in
    # decision.csv). Status is MEASURABLE with an empty caveat on purpose:
    # unlike the belief properties above it asserts nothing that could be
    # defective, so it must neither block nor promote any other property's
    # cell. Because its static status is not MEASURABLE_FLAWED, `emit()`'s
    # MEASURABLE_FLAWED -> MEASURABLE downgrade rule never touches it, and
    # its empty caveat is not a promotion.
    "state_sensitivity_mask_constant": ("MEASURABLE", ""),
    # The effective-n companions to the row above, because
    # that row alone cannot distinguish a cell whose action ignored the state
    # from a cell whose reached distribution supplied no state to respond to.
    # At the POOLED (arm, game, mask) unit, 8 bins
    # totalling 122,213 rows hold exactly one distinct prompt, so "the action
    # is a constant function of legal_mask" there is a determinism
    # observation, not a state-blindness one.
    #
    # At the PER-cell (game, opponent, mask) unit this function is actually
    # called on, restricting to bins with >= 2 distinct prompts moves the
    # figure 0.7753 -> 0.6775 (269,902 / 398,406), setting aside 90 one-prompt
    # bins / 263,238 rows. Both units are named explicitly because quoting a
    # pooled figure inside a per-cell function is a known error: the pooled
    # mask-constant share is 0.7506 (496,659 / 661,644), not 0.7753.
    #
    # Report-only, and deliberately absent from
    # DECISION_RULE: whether a state-blind cell should be INADMISSIBLE is a
    # research-protocol decision that cannot be taken until the determinism-
    # vs-blindness reading is resolved. Status MEASURABLE with an empty
    # caveat for the same reason the row above carries one: it asserts
    # nothing that could be defective, so it must neither block nor promote
    # any other property's cell.
    "state_sensitivity_multistate_share": ("MEASURABLE", ""),
    "state_sensitivity_multistate_rows": ("MEASURABLE", ""),
    # The within-seat companion of
    # the row above, and the measure that carries
    # state-blindness claims; the pooled row stays exactly as it is, as the
    # census. Same status and same empty caveat for the same reason: it
    # asserts nothing that could be defective, so it must neither block nor
    # promote any other property's cell, and it is deliberately absent from
    # DECISION_RULE (removing the seat confound does not resolve the
    # determinism-vs-blindness reading that left its sibling unthresholded).
    "state_sensitivity_multistate_share_within_seat": ("MEASURABLE", ""),
    # The denominators,
    # published beside the shares they weight: a share of 1.0 over 2 rows and
    # one over 400 are the same number and not the same evidence, and neither
    # the pooled `_multistate_rows` row nor the shares themselves said which
    # was which for the within-seat measure. Each static caveat names the unit
    # because the two families are counted at different ones and quoting one
    # as the other is an easy error
    # (state_sensitivity.py's "three numbers, three units"). Report-only, so
    # none of the five is in DECISION_RULE.
    "state_sensitivity_bins_total": (
        "MEASURABLE",
        "count of legal_mask bins in this cell's LM rows (unit: bin)",
    ),
    "state_sensitivity_bins_multistate_pooled": (
        "MEASURABLE",
        "count of legal_mask bins holding >= 2 distinct prompts with the "
        "seats POOLED (unit: bin)",
    ),
    "state_sensitivity_bins_multistate_within_seat": (
        "MEASURABLE",
        "count of (legal_mask bin, seat) PAIRS holding >= 2 distinct prompts "
        "within that seat (unit: bin-seat pair, NOT bin)",
    ),
    "state_sensitivity_rows_multistate_pooled": (
        "MEASURABLE",
        "LM rows in the pooled multi-state bins; the denominator of "
        "state_sensitivity_multistate_share, and the same quantity the older "
        "state_sensitivity_multistate_rows row carries (unit: row)",
    ),
    "state_sensitivity_rows_multistate_within_seat": (
        "MEASURABLE",
        "LM rows in the within-seat multi-state (bin, seat) pairs; the "
        "denominator of state_sensitivity_multistate_share_within_seat "
        "(unit: row)",
    ),
}


# The pre-registered decision rule. `threshold` is the value at or beyond which
# the property counts as FAILED by the LM. These are recorded before looking at
# the numbers so the rule cannot be tuned to the result.
#
# Thresholds that sit far below the observed statistic scale fail every cell
# vacuously, so each entry here is chosen against that scale:
#   * P8's statistic is capture_rate, not the VoI shortfall; it carries no
#     threshold (see the unthresholded rows below). The shortfall
#     row stays reported in properties.csv, unthresholded -- see the
#     deliberate-omissions list below.
#   * P6 carries a pre-registered RULE and no numeric threshold
#     (`threshold=None`, direction "rule"). `decision_table` branches on
#     `threshold is None` before any comparator; a numeric here would be a
#     fake. P3's number is the margin row's 0.0 against the
#     committed sentinel anchor artefact. P6 is the sole rule row, and
#     permanently: its criterion is per game, so the single-valued column
#     cannot hold it. See the note on the P6 entry.
#   * P1, P4, P7a, P7b and P8 carry no threshold (`threshold=None`, direction
#     "none"): there is no basis for a numeric threshold on them. P5 margin
#     keeps 0.0/gte.
#   * P2's thresholded statistic is the shrinkage delta: the
#     `P2_adaptivity_delta` entry below carries the row (>= 0.0
#     fails, `gte`). The level statistic stays reported (unthresholded)
#     beside it; keeping a threshold on both would be a two-sources-of-truth
#     defect. The level row stays reported in properties.csv -- see the
#     deliberate-omissions list below.
#   * P5_calibration's ECE carries no entry: exactly one P5 row (the margin)
#     is thresholded. The ECE keeps its properties.csv row.
#
# Deliberate omissions from this list (properties that do reach
# properties.csv but carry no decision.csv verdict) -- documented here, not
# left to be inferred from absence:
#   * P2_adaptivity_coverage -- a coverage denominator for P2_adaptivity's
#     br_gap, not itself a thresholded optimality claim (see its
#     PROPERTY_STATUS comment above).
#   * P2_adaptivity (the level br_gap) -- the level statistic stays reported
#     (unthresholded) and the only P2 row here is the shrinkage delta, which
#     `P2_adaptivity_delta` below carries. Same shape as the P8 shortfall
#     and P5 ECE: the row keeps its properties.csv place,
#     its value and its per-cell replicate CI, and loses only the verdict.
#   * P5_calibration / P5_calibration_per_pair -- the raw ECE / per-pair-
#     error numbers. Adding a NEW thresholded property to this list is a
#     research-protocol decision; only the relative margin below
#     (P5_calibration_per_pair_margin) is pre-registered. The ECE row stays
#     Reported in properties.csv, unthresholded.
#   * P3_randomisation (the raw regret against the CFR equilibrium value) --
#     reported-only, for the same reason the raw P5 error is: the pre-registered
#     comparison is against the uniform-random sentinel, and the row that
#     carries it is `P3_randomisation_margin` below. The regret row keeps its
#     properties.csv place, its value and its per-cell replicate CI, and
#     loses only the verdict. Exactly one P3 row reaches decision.csv, which
#     is also what keeps the scoreboard's P3 cell a single headline row
#     rather than a two-part cell whose first part could never have one.
#   * P3_randomisation_exact -- the exact-
#     distribution KL diagnostic. `parsed_action` (P3_randomisation) is the
#     Primary scored quantity; the exact-distribution KL is reported beside it
#     as a named confound diagnostic, with no pre-registered threshold of
#     its own.
#   * state_sensitivity_mask_constant -- the share of LM
#     rows whose action is a constant function of `legal_mask`. It describes
#     what the log contains, not how well the LM played, so there is nothing
#     for a threshold to adjudicate; a reader uses it to decide whether an
#     optimality number on that cell is about a policy at all. Adding a
#     threshold for it would be a research-protocol decision.
#   * P8_value_of_information (the VoI shortfall) -- not the P8 headline;
#     `P8_capture_rate` is. The shortfall row stays in
#     properties.csv, unthresholded, exactly like the two raw P5 rows above.
DECISION_RULE = [
    (
        "P1_invariance",
        "paired policy TVD under payoff-preserving relabelling, per probe pair",
        None,
        "none",
    ),
    # The only P2 row: boundary-at-zero on
    # the shrinkage the property claims (`delta_t -> 0`), scale-free where a
    # level threshold cannot be game-uniform
    # (per-cell br_gap spans -0.031 to 4.376 across pot scales).
    # `gte`, not `gt`: a delta of exactly 0.0 is "no shrinkage" and FAILS,
    # the same tie logic as the P5 margin's `gte` below. Measured only when
    # --p2-hand-bins > 0; the artifact of record (bins=0) has zero
    # trajectory rows, so this row reads NOT_MEASURED there (see
    # PROPERTY_STATUS's entry).
    #
    # The level statistic ("P2_adaptivity") has no entry: a threshold on it
    # beside this delta would be a two-sources-of-truth defect. The
    # level row stays reported in properties.csv (only decision.csv
    # membership is affected; the row keeps its value, caveat and
    # per-cell replicate CI) -- see the deliberate-omissions list above.
    (
        "P2_adaptivity_delta",
        "shrinkage of the best-response gap across the windowed hand-index "
        "trajectory: br_gap(late bin) - br_gap(early bin), early = first bin, "
        "late = last bin; requires --p2-hand-bins > 0, NOT_MEASURED on the "
        "unwindowed artifact of record",
        0.0,
        "gte",
    ),
    # P3's verdict is carried by the margin, not by the raw regret. Regret
    # against the CFR equilibrium value is at most 0 and 0 is
    # equilibrium-optimal, so no absolute threshold on it separates play that
    # beats chance from play that does not: every real cell sits below 0 and
    # the comparison the property actually names is against the uniform-random
    # sentinel. The margin (this cell's regret minus that cell's sentinel
    # anchor) makes the one threshold 0 game-uniform, which is what a single
    # `DECISION_RULE` row needs, and `lte` is the rule verbatim: a margin at
    # or below 0 is "no better than random" and FAILS, the tie included.
    # Per (game, opponent), never one number for fifteen cells: random play's
    # regret differs by game and by which opponent it faces, and the P6 anchor
    # above is the counter-example worth not repeating (one kuhn-measured
    # number applied to every game). The anchor comes from
    # `configs/anchors/p3_sentinel_anchors.json` through
    # `lmopt.analysis.sentinel_anchors.anchor_for`, which refuses a cell whose
    # anchor is missing or whose recorded provenance disagrees with the run
    # being scored, and never substitutes another cell's anchor. The
    # interval leg is conservative and is built in `property_table`: the LM
    # interval's low minus the anchor's high and the LM interval's high minus
    # the anchor's low, so the verdict machinery's own straddle check makes
    # the LM clear the sentinel's whole interval, not only its mean.
    # Falsification condition for the anchor artefact: it is wrong the day an
    # opponent policy, an iteration count, the seat mode or the pyspiel
    # version changes, which is what the artefact's provenance block exists
    # for.
    (
        "P3_randomisation_margin",
        "regret against the CFR equilibrium value minus the uniform-random "
        "sentinel's regret on the same (game, opponent) cell, from the "
        "committed sentinel anchor artefact; a margin at or below 0 is no "
        "better than random and fails, and the LM interval must clear the "
        "sentinel's interval, not only its mean",
        0.0,
        "lte",
    ),
    (
        "P4_payoff_sensitivity",
        "paired policy TVD under positive affine payoff rescaling "
        "(payoff_scale_x10 probe)",
        None,
        "none",
    ),
    # The "beat-uninformed" rule. FAILS unless the LM's mean per-pair
    # calibration error is strictly below the uniform-stater baseline
    # computed on the same rows -- i.e. `margin < 0.0` is the only PASS.
    # Framed as a margin (`per_pair_error - uniform_baseline_per_pair`,
    # property5.py), so the rule is a comparison of the margin against 0.0.
    #
    # The comparator is `gte`, not `gt`: the exact tie (margin == 0.0) must
    # FAIL, and under `gt` `observed > 0.0` is False there, so `decision_table`
    # would report LM_MEETS_CRITERION. The tie is reachable, not measure-zero:
    # an LM that always states the uniform distribution ties the uniform-stater
    # baseline by construction. For example, on mistral-nemo-instruct-2407,
    # liarsdice/always_bet,
    # `per_pair_error == uniform_baseline_per_pair == 0.07195234905502011`,
    # margin exactly 0.0, because that arm's `elicited_belief` is the constant
    # uniform {"0":1/6,...,"5":1/6} on 10,000 of 10,000 LM rows. `gte` means
    # "margin >= 0.0 => FAIL", tie included.
    (
        "P5_calibration_per_pair_margin",
        "mean per-pair calibration error minus the uniform-stater baseline on the same rows",
        0.0,
        "gte",
    ),
    # P6's threshold is PER game, and no single number can stand in this
    # column. The off-path distance of uniform-random play differs by game and
    # by the subgame the opening drives into, so one anchor measured on one
    # game applied to five is the defect the P3 margin row above exists not to
    # repeat. The five anchors live in
    # `configs/anchors/p6_sentinel_anchors.json`, and the cell's verdict
    # against its own game's anchor is computed where the P6 measurement lives,
    # in `scripts/probe_scores.py` through
    # `lmopt.analysis.probe_scores.offpath_verdict`, which refuses a game whose
    # anchor is missing or whose recorded provenance disagrees with the arm
    # being scored and never substitutes another game's anchor. This row is
    # therefore a rule row: the number is not absent, it is plural, and a
    # single-valued column that named one of the five would be read as binding
    # on all of them. Falsification condition for the anchors: they are wrong
    # the day the opening, the seat mode, the tremble mass, the reference
    # iteration count or the OpenSpiel version changes, which is what the
    # artefact's provenance block exists for.
    (
        "P6_subgame_perfection",
        "off-path TVD vs the tremble-refined reference: fails unless below the "
        "uniform-random sentinel anchor measured on the SAME game, from the "
        "committed off-path sentinel anchor artefact, interval-excluding it -- "
        "the cell's replicate interval must lie entirely below the anchor's "
        "interval, so an overlap is indeterminate and never a pass; the threshold "
        "is per game and is not a number this column can carry; the signed row's "
        "aspirational secondary bound of 0.1 stays stated as text only, not "
        "binding",
        None,
        "rule",
    ),
    ("P7a_belief_error", "TVD(elicited belief, true posterior)", None, "none"),
    (
        "P7b_action_belief",
        "fraction of actions best-responding to the LM's OWN belief",
        None,
        "none",
    ),
    (
        "P8_capture_rate",
        "VoI capture rate over informative Dark Hex nodes (the shortfall stays "
        "reported beside it, unthresholded)",
        None,
        "none",
    ),
]


def _rule_has_no_comparator(observed, threshold):
    raise RuntimeError(
        "a sentinel-rule DECISION_RULE row (threshold None) reached the "
        "numeric comparator; decision_table must branch on `threshold is "
        "None` before any comparison (no silent fallbacks)"
    )


_DIRECTION_COMPARATORS = {
    "gt": lambda observed, threshold: observed > threshold,
    "gte": lambda observed, threshold: observed >= threshold,
    "lt": lambda observed, threshold: observed < threshold,
    "lte": lambda observed, threshold: observed <= threshold,
    "rule": _rule_has_no_comparator,
    "none": _rule_has_no_comparator,
}


_DIRECTION_TEXT = {
    "gt": "value > threshold means FAIL",
    "gte": "value >= threshold means FAIL",
    "lt": "value < threshold means FAIL",
    "lte": "value <= threshold means FAIL",
    "none": (
        "no threshold: the value is reported with its interval and no pass or fail is issued"
    ),
    "rule": (
        "no single numeric threshold: the estimand names the "
        "pre-registered RULE and where its per-game criterion is held; "
        "the verdict is issued where the property is measured"
    ),
}


# Canonical enumeration of every verdict string `decision_table` can emit.
# Single source of truth: `decision_table` asserts every verdict it assigns
# is a member of this tuple before the row is built, and figure code
# (`figures.cross_arm_shared._VERDICT_COLORS`, `figures.threshold_verdicts_robustness.render_threshold_verdicts`)
# imports this tuple to build/validate its colour maps -- so the verdict
# space and the colour space are enumerated once and cannot silently drift
# apart. Without this, a verdict added to `decision_table` without a matching
# update to a figure's colour map renders as visually
# indistinguishable-from-missing "unknown verdict" grey.
VERDICT_MEETS = "LM_MEETS_CRITERION"
VERDICT_FAILS = "LM_FAILS_CRITERION"
VERDICT_INDETERMINATE = "CI_INDETERMINATE"
VERDICT_CI_UNAVAILABLE = "CI_UNAVAILABLE"
#: The model reached none of the informative states the property is defined on, on any cell
#: that could have had them: a fact about this arm's play, not a property that cannot exist.
VERDICT_NOT_REACHED = "NOT_REACHED"
#: A property that carries no pass threshold, because there is no basis for one: its admissible
#: cells give a number and an interval, and no judgement.
VERDICT_NO_THRESHOLD = "NO_THRESHOLD"
VERDICTS = (
    VERDICT_MEETS,
    VERDICT_FAILS,
    VERDICT_INDETERMINATE,
    VERDICT_CI_UNAVAILABLE,
    "INADMISSIBLE",
    "NOT_MEASURED",
    VERDICT_NOT_REACHED,
    VERDICT_NO_THRESHOLD,
)

#: The state-sensitivity failure clause the scoreboard reads to tell an inadmissible cell from a
#: BROKEN one.
FAILED_REASON = "failed the pre-registered state-sensitivity check"


#: The one shard filename convention `run_sweep.py::_shard_out_path` writes:
#: `{game}_lm_vs_{opponent}[_seed{i}][.{arm}].parquet`. The seed token is present whenever the run
#: has more than one seed; the arm component only on a probe arm, always after the seed token.
_SHARD_NAME_RE = re.compile(
    r"^(?P<game>.+?)_lm_vs_(?P<opponent>.+?)(?:_seed(?P<seed>\d+))?(?:\.(?P<arm>[^.]+))?\.parquet$"
)


class ShardName(NamedTuple):
    """The fields one shard filename carries."""

    game: str
    opponent: str
    seed: int | None
    arm: str | None


def parse_shard_name(name: str) -> ShardName | None:
    """
    the single parser of the shard filename convention, shared by every reader
    :param name: a file name, not a path (str)
    :returns: the parsed fields, with seed and arm None when the name carries no such component;
        None when the name is not a shard name at all (ShardName | None)
    """
    m = _SHARD_NAME_RE.match(name)
    if m is None:
        return None
    seed = m.group("seed")
    return ShardName(
        m.group("game"),
        m.group("opponent"),
        None if seed is None else int(seed),
        m.group("arm"),
    )


def cell_of(path: Path):
    """Recover (game, opponent, seed) from a base-arm shard filename.

    Needed because `matchup_id` in the Parquet collapses game and seed: it has
    only one value per opponent, so it cannot address a cell on its own. A name
    without a seed token, or with an arm component, is not a cell and returns None.
    """
    parsed = parse_shard_name(path.name)
    if parsed is None or parsed.seed is None or parsed.arm is not None:
        return None
    return parsed.game, parsed.opponent, parsed.seed


#: Private row key carrying the exact `cell_mask_constant_share` float
#: alongside STATE_BLIND_COLUMN's rendered label. Leading
#: underscore so `write_csv` strips it from properties.csv -- it is an
#: internal hand-off to `decision_table`, not part of the published schema.
_STATE_BLIND_SHARE_KEY = "_state_blind_share_value"


def _state_blind_range_for_admissible(shares: list) -> str:
    """Render the state-blindness of the cells `decision_table` calls
    admissible -- the exact population `observed_mean_admissible` and
    `n_cells_admissible` are computed over -- as a range for the new column.

    Deliberately reads the admissible cells' shares only,
    never the blocked/all-cells ones: the verdict is computed from the
    admissible population, and a range describing a different population
    would be a real number attached to the wrong object. For example,
    from `analysis/a3-cycle-20260806-postfix2/<arm>/properties.csv`
    (P7a_belief_error, 4 admissible liarsdice cells per arm):
    qwen2.5-14b-instruct's admissible cells sit at 0.399688-0.500000;
    olmo-2-1124-13b-instruct's (a different arm) sit at
    0.284561-0.500000. `0.7506` is neither arm's pooled figure: it is the
    three-ARM combined cycle figure, 496,659/661,644
    (`src/lmopt/analysis/state_blind_label.py` module docstring), whose
    per-arm components are Qwen 61.8%, OLMo 71.8%, Mistral 91.1% -- quoting it
    unqualified next to a per-arm range is the same right-number-wrong-object
    defect this function exists to prevent.

    `shares`: the exact per-cell floats
    (`_STATE_BLIND_SHARE_KEY`, `None` where unmeasurable), not the rendered
    "NN.NN%" labels -- a regex re-parse of the label loses whatever
    precision `label_for_share`'s 2dp formatting discarded, and (for a
    `None`-share row's free-text `none_reason`) risks matching an unrelated
    percent-shaped substring in the reason text instead of correctly
    reporting "unavailable".

    Never returns "" for the empty case: a blank reads as "not
    state-blind" (no silent fallbacks), the same hazard
    `label_for_share(None)` exists to avoid one level down.
    """
    if not shares:
        return (
            "no admissible cells stand behind this verdict, so the "
            "state-blindness of the reported population is not applicable"
        )
    measured = [s for s in shares if s is not None]
    if not measured:
        return (
            f"state-blindness share unavailable for all {len(shares)} "
            "admissible cell(s) behind this verdict"
        )
    lo, hi = min(measured), max(measured)
    span = f"{lo:.2%}" if lo == hi else f"{lo:.2%}-{hi:.2%}"
    return (
        f"action is constant in `legal_mask` on {span} of LM rows across "
        f"the {len(shares)} admissible cell(s) behind this verdict"
    )


#: The property whose headline is taken on the exact-policy distribution where an arm has one.
_EXACT_BASIS_PROPERTY = "P7b_action_belief"


BASIS_DISTRIBUTION = "distribution"


BASIS_SAMPLED = "sampled"


def _headline_basis(prop_rows):
    """
    choose the basis P7b's verdict is judged on, and re-point the rows at it
    All or nothing per arm: the exact value is the headline only when EVERY LM P7b row that has a
    sampled value also carries an exact one, because one pooled mean over cells on two bases is
    neither the distribution value nor the sampled one. An arm where no P7b row has an exact value
    stays on the sampled basis without comment; an arm where only some do stays on it too, and
    `reason` names the cells that lack one. The rows handed back are copies, so `properties.csv`
    keeps its sampled `value`; only the copies' `value`, `ci_lo` and `ci_hi` are replaced.
    :param prop_rows: `property_table`'s rows (list[dict])
    :returns: (rows to judge, BASIS_DISTRIBUTION or BASIS_SAMPLED, why the sampled basis was kept
        when some cells had an exact value, else "") (tuple)
    """
    from lmopt.analysis.property_rows import lm_rows

    scored = [
        r
        for r in lm_rows(prop_rows)
        if r["property"] == _EXACT_BASIS_PROPERTY and r["value"] is not None
    ]
    have = [
        r
        for r in scored
        if r.get("value_exact") is not None and not math.isnan(r["value_exact"])
    ]
    if not have:
        return prop_rows, BASIS_SAMPLED, ""
    swapped = {id(r) for r in have}
    lacking = sorted(
        f"{r['game']}/{r['opponent']}" for r in scored if id(r) not in swapped
    )
    if lacking:
        return (
            prop_rows,
            BASIS_SAMPLED,
            "the exact value is missing for " + ", ".join(lacking),
        )
    return (
        [
            (
                {
                    **r,
                    "value": r["value_exact"],
                    "ci_lo": r.get("ci_low_exact"),
                    "ci_hi": r.get("ci_high_exact"),
                }
                if id(r) in swapped
                else r
            )
            for r in prop_rows
        ],
        BASIS_DISTRIBUTION,
        "",
    )


def _round_executed(x):
    """
    a mean rounded as the decision table rounds its own
    :param x: a mean or None
    :returns: the mean to six places, as `observed_mean_admissible` is, or None
    """
    return None if x is None else round(x, 6)


def executed_mean_admissible(prop_rows, prop=_EXACT_BASIS_PROPERTY):
    """
    the mean of the executed-move (sampled) values of the cells `decision_table` pools into
    `observed_mean_admissible`: the LM rows of `prop` that are MEASURABLE with a value
    :param prop_rows: properties rows (list[dict]), as `property_table` returns them or
        `properties.csv` read with numeric columns typed; `value` is the sampled rate on every row
    :param prop: the property name, P7b by default, the one whose headline is taken on the exact
        distribution where an arm has one
    :returns: the mean over the admissible cells, or None when there are none (float | None)
    """
    from lmopt.analysis.property_rows import lm_rows

    values = [
        r["value"]
        for r in lm_rows(prop_rows)
        if r["property"] == prop
        and r["status"] == "MEASURABLE"
        and r["value"] is not None
    ]
    return sum(values) / len(values) if values else None


#: The decision fields a reader may derive in process from an existing properties table, without
#: the arm's `decision.csv`. Nothing else of the decision row is faithful from `properties.csv`
#: alone (the state-blindness label needs a private per-cell share the table does not hold).
DERIVED_VERDICT_FIELDS = (
    "verdict",
    "verdict_reason",
    "observed_mean_admissible_executed",
)


def derived_verdict_fields(prop_rows):
    """
    the verdict, its reason and the executed-move mean of every property, derived in process from
    a properties table, for a figure or board over an arm scored before those fields existed
    This is not a decision table: the returned dicts carry exactly `DERIVED_VERDICT_FIELDS`, and a
    caller must never write them as, or present them as, an arm's `decision.csv`.
    :param prop_rows: properties rows (list[dict]), numeric columns typed and missing values None
    :returns: {property name: {field: value}} with exactly the three fields per property (dict)
    """
    return {
        row["property"]: {field: row[field] for field in DERIVED_VERDICT_FIELDS}
        for row in decision_table(prop_rows)
    }


def decision_table(prop_rows):
    """Table 3: the pre-registered rule, evaluated where it legitimately can be."""
    # Function-local, matching this file's own convention
    # for lmopt.analysis imports (property_table does the same).
    from lmopt.analysis.property_rows import lm_rows
    from lmopt.analysis.state_blind_label import STATE_BLIND_COLUMN

    executed_rows = prop_rows
    prop_rows, p7b_basis, p7b_basis_reason = _headline_basis(prop_rows)

    # Admissibility is decided PER cell, not per property: the belief defects are
    # game-specific, so a property can be admissible on some games and not others.
    # Aggregating over both kinds would let a defective cell drag an otherwise
    # sound verdict, or vice versa.
    admissible: dict[str, list[float]] = {}
    blocked: dict[str, list[float]] = {}
    blockers: dict[str, set] = {}
    # Collected alongside `admissible`, from the same
    # rows that populate it (same `continue` filter, same MEASURABLE check),
    # so its length always matches `len(ok)` below -- never recomputed by a
    # second, possibly-divergent pass over `prop_rows`.
    admissible_labels: dict[str, list] = {}
    # The admissible cells' per-cell replicate
    # intervals (property_table's `ci_lo`/`ci_hi`), collected from the same
    # rows and under the same filter as `admissible` itself -- the population
    # the CI-aware verdicts are computed over must be exactly the one
    # the point verdict already uses.
    admissible_cis: dict[str, list[tuple]] = {}
    # The admissible cells that carry no interval, collected in the same
    # loop and under the SAME MEASURABLE filter as `admissible_cis` itself,
    # so a withheld CI_UNAVAILABLE verdict can name exactly which cell(s)
    # behind it lack one rather than leaving a reader to re-derive it.
    cells_missing_interval: dict[str, set] = {}
    # The drop counts of every LM row that carries them, including a row whose
    # value is None because the whole cell was dropped, pooled by counts:
    # rates over cells of different size cannot be averaged.
    dropped_counts: dict[str, list[int]] = {}
    dropped_units: dict[str, set] = {}
    awaiting: set = set()
    # Per property, the LM cells that are not NOT_APPLICABLE, those among them that are NOT_REACHED,
    # and the nodes those cells scored, so a property whose every applicable cell was never reached
    # reads NOT_REACHED and not NOT_MEASURED.
    applicable_cells: dict[str, int] = {}
    unreached_cells: dict[str, int] = {}
    unreached_nodes: dict[str, int] = {}
    unreached_counted: dict[str, int] = {}
    board_cells: dict[str, int] = {}
    # The machine-readable approved-reason classes of the rows whose caveat feeds `blocker`,
    # so a reader decides what a blocker is from this field and never from its wording.
    blocker_classes: dict[str, set] = {}
    for r in lm_rows(prop_rows):
        board_cells[r["property"]] = board_cells.get(r["property"], 0) + 1
        if r["status"] != "NOT_APPLICABLE":
            applicable_cells[r["property"]] = applicable_cells.get(r["property"], 0) + 1
        if r["status"] == "NOT_REACHED":
            unreached_cells[r["property"]] = unreached_cells.get(r["property"], 0) + 1
            if r.get("n_nodes") is not None:
                unreached_nodes[r["property"]] = (
                    unreached_nodes.get(r["property"], 0) + r["n_nodes"]
                )
                unreached_counted[r["property"]] = (
                    unreached_counted.get(r["property"], 0) + 1
                )
        if r.get("n_total") is not None:
            pair = dropped_counts.setdefault(r["property"], [0, 0])
            pair[0] += r.get("n_dropped") or 0
            pair[1] += r["n_total"]
            dropped_units.setdefault(r["property"], set()).add(r.get("dropped_unit"))
        if r["status"] == "AWAITING_DATA" and r["caveat"]:
            # A cell too small to score has no value, but what it lacks belongs
            # in the blocker column, not nowhere.
            blockers.setdefault(r["property"], set()).add(
                r["caveat"].split(";")[0].strip()
            )
            if r.get("partial_reason_class"):
                blocker_classes.setdefault(r["property"], set()).add(
                    r["partial_reason_class"]
                )
            awaiting.add(r["property"])
        if r["value"] is None:
            continue
        bucket = admissible if r["status"] == "MEASURABLE" else blocked
        bucket.setdefault(r["property"], []).append(r["value"])
        if r["status"] == "MEASURABLE":
            # The exact float, not a re-parse of
            # STATE_BLIND_COLUMN's rendered label -- see _STATE_BLIND_SHARE_KEY.
            admissible_labels.setdefault(r["property"], []).append(
                r.get(_STATE_BLIND_SHARE_KEY)
            )
            admissible_cis.setdefault(r["property"], []).append(
                (r.get("ci_lo"), r.get("ci_hi"))
            )
            if r.get("ci_lo") is None or r.get("ci_hi") is None:
                cells_missing_interval.setdefault(r["property"], set()).add(
                    f"{r['game']}/{r['opponent']}"
                )
        if r["status"] != "MEASURABLE" and r["caveat"]:
            blockers.setdefault(r["property"], set()).add(
                r["caveat"].split(";")[0].strip()
            )
            if r.get("partial_reason_class"):
                blocker_classes.setdefault(r["property"], set()).add(
                    r["partial_reason_class"]
                )

    rows = []
    for prop, estimand, threshold, direction in DECISION_RULE:
        ok, bad = admissible.get(prop, []), blocked.get(prop, [])
        observed = sum(ok) / len(ok) if ok else None
        # Shown even when inadmissible, so the direction of the evidence stays
        # visible; it is the verdict that withholds, not the number.
        all_vals = ok + bad
        shown = sum(all_vals) / len(all_vals) if all_vals else None
        # Additive nullable interval columns for the
        # decision row: the envelope (min lo, max hi) over the admissible
        # cells' per-cell replicate intervals. Deliberately not a 95% CI of
        # `observed_mean_admissible` (per-cell intervals cannot be combined
        # into one without treating cells as replicates, which is not
        # justified): it is the conservative straddle
        # input for the CI-aware verdicts below -- it straddles the threshold
        # iff some admissible cell's interval does, or admissible cells sit
        # on both sides of it, and it reduces to the cell's own interval
        # when one admissible cell backs the verdict. None unless every
        # admissible cell carries an interval: an envelope over a subset
        # would silently understate the uncertainty of the exact population
        # the verdict is computed over (no silent fallbacks).
        cis = admissible_cis.get(prop, [])
        if cis and all(lo is not None and hi is not None for lo, hi in cis):
            ci_lo, ci_hi = min(lo for lo, _ in cis), max(hi for _, hi in cis)
        else:
            ci_lo = ci_hi = None
        # CI-aware verdicts: on admissible data the interval envelope above gets a vote,
        # against `observed` (the mean of the admissible cells' own seed
        # means, so it is always the envelope's own point under the
        # value-centre rule -- see the "Unreachable by construction" comment
        # below). Three outcomes follow, not four: straddles the threshold ->
        # CI_INDETERMINATE (a straddling cell never rounds to a clean
        # verdict; load-bearing at P8's exact-0.500 knife-edge cells and
        # P7b's 0.9005 cell) or not formable -> CI_UNAVAILABLE (a point
        # verdict no interval can back is withheld, not published); agrees
        # with the point -> the point verdict. A fourth state -- the envelope
        # lying entirely on the opposite side from the point -- cannot occur,
        # because every admissible row's `value` is its own cell's seed mean,
        # so the branch below raises instead of publishing a verdict for it. INADMISSIBLE /
        # NOT_MEASURED keep their meanings.
        reason = None
        if direction == "none" and observed is not None:
            # An unthresholded row: nothing to compare against, so the admissible cells give a
            # number and their interval envelope and no verdict class. Blocked, not reached and
            # absent cells fall through to the branches below and keep their meanings. No reason
            # is written, as for a clean point verdict: the direction column already says that
            # no threshold is in force, and a repeat of it would crowd the scoreboard cell.
            verdict = VERDICT_NO_THRESHOLD
        elif threshold is None and direction == "rule":
            # A rule row (direction "rule"): this row's criterion is not a
            # number this table can carry, so no verdict is issued here. P6 is
            # the one such row, and its criterion is per game, held in the
            # off-path sentinel anchor artefact and applied where the P6
            # measurement itself lives; its own DECISION_RULE entry says so.
            # The observed columns stay visible (it is the verdict that
            # withholds, not the number); the comparator is never consulted
            # (it raises if it is).
            #
            # A rule row with only blocked cells
            # also reads NOT_MEASURED here, not INADMISSIBLE, on purpose:
            # INADMISSIBLE means "defects blocked cells from a verdict the
            # rule could otherwise issue", and with no criterion in force
            # there is no such verdict to block -- the outcome is identical
            # whatever the cells' status. The defects are not hidden: the
            # blockers dict populates regardless of verdict, so the blocker
            # column still names them on the emitted row.
            verdict = "NOT_MEASURED"
            reason = (
                "no single numeric threshold is in force: this row's "
                "criterion is per game (see estimand) and is applied "
                "where the property is measured, not in this table"
            )
        elif observed is not None:
            # Indexed, not an `if/else` chain with a fall-through default: an
            # unrecognised direction is a bug in `DECISION_RULE`, and under a
            # chain it would silently adopt whichever comparator sat in the
            # `else` -- publishing a verdict under a rule nobody wrote
            # (no silent fallbacks).
            failed = _DIRECTION_COMPARATORS[direction](observed, threshold)
            if ci_lo is not None:
                lo_failed = _DIRECTION_COMPARATORS[direction](ci_lo, threshold)
                hi_failed = _DIRECTION_COMPARATORS[direction](ci_hi, threshold)
                # The published reasons name the quantity for what it is: the
                # interval envelope over the admissible cells' replicate
                # intervals (min lo, max hi -- see the interval-columns comment above),
                # which reduces to a single cell's own replicate interval only
                # when one admissible cell backs the verdict. Calling it "the
                # replicate interval" in decision.csv would overstate it as one
                # cell's t-interval.
                if lo_failed != hi_failed:
                    verdict = "CI_INDETERMINATE"
                    reason = (
                        f"interval envelope over the admissible cells' "
                        f"replicate intervals [{ci_lo:.6g}, {ci_hi:.6g}] "
                        f"straddles the fail threshold {threshold:g} (point "
                        f"{observed:.6g} {'fails' if failed else 'passes'}): "
                        "the design cannot resolve this verdict at the "
                        "current replicate count"
                    )
                elif lo_failed != failed:
                    # Unreachable by construction: every admissible row's
                    # `value` is its own cell's seed mean, the centre of
                    # that cell's own `[ci_lo, ci_hi]` replicate interval, so
                    # `observed` (the mean of those centres) always lies
                    # inside the envelope (min lo, max hi) over the same
                    # cells' intervals -- a point cannot sit strictly
                    # between an interval's own bounds while its
                    # pass/fail comparator disagrees with the lower bound
                    # but agrees with the upper one, which is exactly what
                    # `lo_failed == hi_failed` (excluded above) together
                    # with `lo_failed != failed` would require. Reaching
                    # this branch means a row's `value` and its own
                    # `[ci_lo, ci_hi]` have come apart upstream, which is a
                    # bug to surface, not a verdict to publish -- publishing
                    # one here would be a silent fallback into a state the
                    # design does not allow to exist. The only known
                    # residue this could still catch is a <= 1e-6
                    # double-rounding sliver on a degenerate P3 margin cell
                    # (the margin's `value` and its `ci_lo`/`ci_hi` are each
                    # rounded to 6dp from a different subtraction, so a
                    # cell sitting exactly at a threshold boundary could in
                    # principle round to opposite sides by that much); it
                    # has never been observed to fire.
                    raise RuntimeError(
                        f"{prop}: observed value {observed:.6g} lies outside "
                        f"its own admissible-cell interval envelope "
                        f"[{ci_lo:.6g}, {ci_hi:.6g}]; a row's value is "
                        "defined as its own interval's centre, so this "
                        "state must never be reached"
                    )
                else:
                    verdict = "LM_FAILS_CRITERION" if failed else "LM_MEETS_CRITERION"
            else:
                verdict = "CI_UNAVAILABLE"
                missing = ", ".join(sorted(cells_missing_interval.get(prop, set())))
                reason = (
                    "no interval envelope is formable over the "
                    "admissible cells' replicate intervals (fewer than "
                    "2 replicates, or an admissible cell without an "
                    "interval), so the point verdict is withheld "
                    "rather than published without the CI check the "
                    "framework requires; cells without an interval: "
                    f"{missing}"
                )
        elif bad:
            verdict = "INADMISSIBLE"
        else:
            # A NOT_MEASURED verdict means no admissible cells
            # exist (no rows, or all rows are blocked). The blocker column
            # already names the defects when cells were blocked; when there are
            # No cells at all (neither admissible nor blocked), the blocker
            # fallback to PROPERTY_STATUS[prop][1] can be empty (e.g. P8), so
            # the verdict_reason must name the absence explicitly.
            verdict = "NOT_MEASURED"
            n_unreached = unreached_cells.get(prop, 0)
            if n_unreached and n_unreached == applicable_cells.get(prop):
                verdict = VERDICT_NOT_REACHED
                # An arm scored before the row carried its counts has none to report, and
                # a sum over absent counts would read as zero scored rows.
                scored_rows = (
                    f"0 of {unreached_nodes[prop]} scored rows across "
                    if unreached_counted.get(prop) == n_unreached
                    else "scored-row count not recorded, across "
                )
                reason = (
                    "the model reached none of the informative states: "
                    f"{scored_rows}{n_unreached} cells; the board has "
                    f"{board_cells[prop]}"
                )
            elif not ok and not bad and prop in awaiting:
                # Not "every cell is too small to score": two properties
                # emit AWAITING_DATA rows for different reasons (a P2 bin
                # below the licensed hands, a P3 cell with no usable sentinel
                # anchor), and the blocker column carries the one that
                # applies.
                reason = (
                    "awaiting data: no cell of this arm carries a "
                    "scored value, see the blocker"
                )
            elif not ok and not bad:
                reason = (
                    "no data: the property was not measured on any cell "
                    "of this arm (no rows collected, or scorer not "
                    "applicable to any collected cell)"
                )
        # Consume VERDICTS as the single source of truth for the emitted
        # verdict space: a branch above assigning a string not in VERDICTS
        # is a bug in decision_table itself (a new verdict value introduced
        # without updating the canonical tuple), and downstream colour maps
        # would silently render it as unknown-verdict grey rather than a
        # loud failure here (no silent fallbacks).
        assert verdict in VERDICTS, (
            f"decision_table emitted verdict {verdict!r} for property "
            f"{prop!r}, not a member of VERDICTS -- update the VERDICTS "
            "tuple (and every colour map that keys off it) to cover it"
        )
        n_dropped_pooled, n_total_pooled = dropped_counts.get(prop, (0, 0))
        units = dropped_units.get(prop, set())
        rows.append(
            {
                "property": prop,
                "estimand": estimand,
                "fail_threshold": threshold,
                "direction": _DIRECTION_TEXT[direction],
                "observed_mean_all_cells": None if shown is None else round(shown, 6),
                "observed_mean_admissible": (
                    None if observed is None else round(observed, 6)
                ),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                # Additive: the same admissible cells' mean on the sampled executed-move value,
                # beside the distribution-basis headline above. P7b only; None elsewhere.
                "observed_mean_admissible_executed": (
                    _round_executed(executed_mean_admissible(executed_rows, prop))
                    if prop == _EXACT_BASIS_PROPERTY
                    else None
                ),
                "n_cells_admissible": len(ok),
                "n_cells_blocked": len(bad),
                "dropped_rate": (
                    round(n_dropped_pooled / n_total_pooled, 6)
                    if n_total_pooled
                    else None
                ),
                "dropped_unit": (
                    next(iter(units))
                    if len(units) == 1
                    else ("mixed" if units else None)
                ),
                # The property's static, pre-registered status class, the same for every arm
                # and every run; it is not this row's state. The state is `verdict`, and each
                # cell's own state is `status` in properties.csv. Documented rather than renamed,
                # so no reader of this column changes.
                "status": PROPERTY_STATUS[prop][0],
                "verdict": verdict,
                # Nullable; non-None exactly on the two
                # CI states and the rule-deferred rows, naming what the verdict
                # withheld and why -- a bare CI_INDETERMINATE with no numbers
                # would make a reader re-derive the conflict it flags.
                "verdict_reason": reason,
                "basis": (
                    p7b_basis if prop == _EXACT_BASIS_PROPERTY else BASIS_SAMPLED
                ),
                "basis_reason": (
                    (p7b_basis_reason or None)
                    if prop == _EXACT_BASIS_PROPERTY
                    else None
                ),
                "blocker": "; ".join(sorted(blockers.get(prop, set())))
                or PROPERTY_STATUS[prop][1],
                "blocker_class": "; ".join(sorted(blocker_classes.get(prop, set()))),
                STATE_BLIND_COLUMN: _state_blind_range_for_admissible(
                    admissible_labels.get(prop, [])
                ),
            }
        )
    return rows


@dataclasses.dataclass(frozen=True)
class ExactPolicyLoad:
    """Result of `load_exact_policy`: the policy dict plus which model's
    pass produced it.

    `policy` is `None` iff the artefact could not be used at all (missing,
    unreadable, missing required columns, more than one distinct `model`
    value present, or the
    artefact's own `model` disagrees with the RUN's own model, see
    `load_exact_policy`'s docstring). `model`/`git_sha` are `None`
    whenever `policy` is `None`, and also `None` (not fabricated) if the
    artefact loaded fine but the provenance columns are absent or entirely
    null -- a run of `scripts/exact_policy_pass.py` predating the
    provenance columns is handled the same way `PROMPT_SCHEMA_VERSION`-
    predating data is elsewhere in this harness: named as unknown, not
    silently assumed.

    `run_model_check` is the same not-observed-vs-
    not-set distinction drawn for `run_meta.json`'s own
    `serving.box_fields_source` field, applied here to whether this
    function could verify the artefact against the run's own model at
    all: `None` means "not verified" (the run's own model could not be
    determined -- `run_meta.json` missing/unreadable, no `serving` block,
    no `serving.model_id` -- or the artefact itself carries no `model` to
    compare, e.g. a pre-provenance artefact); the literal string
    `"verified"` means both sides were available and agreed. A
    disagreement never reaches this field as a value -- it is a full
    refusal (`policy=None`, `model=None`, `git_sha=None`, `run_model_check
    =None`, WARNING on stderr), the same shape `load_exact_policy` already
    uses for an internally-mixed-model artefact, since a cross-model
    misattribution is exactly as wrong as an internally-mixed one.

    `refusal_reason`: `None`
    whenever `policy` is not `None` (loaded fine, nothing to explain).
    When `policy` is `None`, a short machine-stable tag naming why:
    `"missing_file"`, `"unreadable"`, `"missing_columns"`,
    `"multi_model_mix"`, or `"model_mismatch"`. Exists so
    `property_table`'s `P3_randomisation_exact` row can tell "the file
    genuinely does not exist" from "the file exists but was
    REFUSED (wrong model, or internally inconsistent)" in
    `properties.csv` itself; stderr is not part of the artefact a later
    reader cites.
    """

    policy: dict[tuple[str, str], dict[int, float]] | None
    model: str | None
    git_sha: str | None
    run_model_check: str | None = None
    refusal_reason: str | None = None


@dataclasses.dataclass(frozen=True)
class _PolicyArtefactPreamble:
    """Internal-only result of `_load_policy_artefact`'s shared
    exists/read/required-columns/multi-model/git_sha/run_model_id
    preamble. Not part of this module's public surface --
    `ExactPolicyLoad` is the real return type callers see.

    On any refusal (`refusal_reason` set), `df`/`model`/`git_sha` are
    `None` and `run_model_check` is `None` -- matching
    `ExactPolicyLoad`'s own "model/git_sha are
    None whenever policy is None" contract exactly.
    """

    df: object | None
    model: str | None
    git_sha: str | None
    run_model_check: str | None
    refusal_reason: str | None


def _load_policy_artefact(
    path: Path,
    *,
    required: set[str],
    artefact_label: str,
    producer_script: str,
    consequence_noun: str,
    run_model_id: str | None,
):
    """The shared loading preamble for a policy artefact: exists -> read -> required-
    columns -> multi-model refusal -> git_sha -> run_model_id mismatch.
    The caller keeps only what is genuinely different after this point
    (`load_exact_policy`'s `group_by`-based policy dict).

    Deliberately 3 caller-supplied strings, not a single `noun`: the
    five WARNING messages vary along three independent axes. A
    short artefact label (e.g. "exact-policy") appears in
    the missing-file and model-mismatch messages; a producer-script name
    appears only in the missing-file message; a consequence phrase
    (e.g. "P3_randomisation_exact") appears in all
    five messages as the "... will be emitted with value=None" suffix. A
    single `noun` can carry the third axis but would corrupt the
    model-mismatch message's artefact-label mention if forced to also
    supply the first.

    `git_sha` is read as: the first
    non-null value, never independently guarded for multiplicity the way
    `model` is (a differing `git_sha` under one `model` is a provenance
    nuance, not a cross-model misattribution risk -- see
    `load_exact_policy`'s own docstring for the fuller rationale, kept
    there rather than duplicated here since it explains a design choice,
    not a mechanical step).

    `run_model_check` is computed uniformly (`None` = not checked,
    `"verified"` = checked and matched, refusal = mismatch) regardless of
    whether the caller's own dataclass has a field for it.
    """
    import polars as pl

    if not path.exists():
        print(
            f"WARNING: no {artefact_label} artefact at {path} "
            f"(produced by {producer_script}); "
            f"{consequence_noun} will be emitted with value=None",
            file=sys.stderr,
        )
        return _PolicyArtefactPreamble(None, None, None, None, "missing_file")
    try:
        df = pl.read_csv(
            path,
            schema_overrides={"game": pl.Utf8, "info_set": pl.Utf8},
            infer_schema_length=None,
        )
    except Exception as e:
        print(
            f"WARNING: {path} unreadable ({e}); {consequence_noun} "
            "will be emitted with value=None",
            file=sys.stderr,
        )
        return _PolicyArtefactPreamble(None, None, None, None, "unreadable")

    missing_cols = required - set(df.columns)
    if missing_cols:
        print(
            f"WARNING: {path} missing column(s) {sorted(missing_cols)}; "
            f"{consequence_noun} will be emitted with value=None",
            file=sys.stderr,
        )
        return _PolicyArtefactPreamble(None, None, None, None, "missing_columns")

    model = None
    if "model" in df.columns:
        models = df["model"].drop_nulls().unique().to_list()
        if len(models) > 1:
            print(
                f"WARNING: {path} carries {len(models)} distinct 'model' "
                f"values {sorted(models)!r} -- refusing to use this "
                "artefact rather than silently attributing it to one of "
                f"them; {consequence_noun} will be emitted with "
                "value=None",
                file=sys.stderr,
            )
            return _PolicyArtefactPreamble(None, None, None, None, "multi_model_mix")
        model = models[0] if models else None
    git_sha = None
    if "git_sha" in df.columns:
        shas = df["git_sha"].drop_nulls().unique().to_list()
        git_sha = shas[0] if shas else None

    run_model_check: str | None = None
    if run_model_id is not None and model is not None:
        if model != run_model_id:
            print(
                f"WARNING: {path} was produced by model={model!r}, but "
                f"this run's own model (run_meta.json serving.model_id) "
                f"is {run_model_id!r} -- refusing to attribute a "
                f"DIFFERENT model's {artefact_label} artefact to this "
                f"run; {consequence_noun} will be emitted with "
                "value=None",
                file=sys.stderr,
            )
            return _PolicyArtefactPreamble(None, None, None, None, "model_mismatch")
        run_model_check = "verified"

    return _PolicyArtefactPreamble(df, model, git_sha, run_model_check, None)


def load_exact_policy(path: Path, run_model_id: str | None = None) -> ExactPolicyLoad:
    """Load `scripts/exact_policy_pass.py`'s own CSV output into the
    `{(game, info_set): {action: prob}}` shape `score_property3_exact`
    needs. Modelled on `_read_sweep_temperature`'s
    shape: returns `ExactPolicyLoad(policy=..., model=..., git_sha=...)`,
    with `policy=None` on any failure and a printed WARNING (to stderr,
    naming the missing/unreadable/ambiguous path) (no silent
    fallbacks).

    Deliberately never returns an empty `policy` dict for a missing/
    unreadable file: `score_property3_exact` treats "no entry for this
    info-set" as "unmatched, counted", so an empty dict for a missing
    artefact would read as "loaded, and every info-set genuinely
    disagreed" -- a materially different (and false) claim from "the
    artefact could not be read at all". An empty dict is returned,
    correctly, when the file loads but happens to contain zero COMPLETE
    (game, info_set) groups -- that is a real, if degenerate, answer, not
    a missing-file case.

    Only info-sets whose full legal-action set scored with no `error` and
    a non-null `prob` are included -- a partial/untrusted distribution
    (`scripts/exact_policy_pass.py`'s own coverage-mass-floor or
    partial-scoring failure modes) is treated as ABSENT, never used
    half-renormalised. On the real
    qwen2.5-14b-instruct arm's `exact_policy.csv`, 0 of 574 rows carry a
    non-null `error` and every one of the 185 distinct (game, info_set)
    groups is complete, so this filter is a no-op on that arm -- it is
    a defensive contract for a less-clean pass.

    `--exact-policy` defaults to the arm's own
    `exact_policy.csv` when its data directory holds one, else
    `out_dir / "exact_policy.csv"` (`default_exact_policy_path`).
    Without a check of which model's pass a resolved path actually holds,
    an arm whose `out_dir` happens to hold a different model's
    artefact would join silently and misattribute the diagnostic, with
    only the generic static caveat as warning. So this function reads the artefact's own
    `model`/`git_sha` columns and surfaces them (never inside
    `src/lmopt/scoring/` -- that must stay a pure function over any frame;
    this is the CONSUMER's job, in `property_table` below). If the
    artefact carries more than one distinct non-null `model` value, that
    is a loud failure -- refused (`policy=None`, WARNING), not something
    to average over or silently pick one of.

    `run_model_id`: the RUN's own model, from
    `_read_run_model_id(data_dir)`/`run_meta.json`'s `serving.model_id`,
    i.e. the full by-construction check.
    Three outcomes, matching the not-observed-vs-not-set distinction drawn
    for `box_fields_source`:

      - `run_model_id is None` or the artefact's own `model` is `None`
        (either side unknowable) -- the check cannot be performed. This is
        not a refusal: `.policy` loads normally if it otherwise would, and
        `.run_model_check` stays `None` ("not verified"), surfaced
        downstream rather than silently treated as passed.
      - Both present and equal -- `.run_model_check = "verified"`,
        `.policy` loads normally.
      - Both present and unequal -- refused exactly like the internally-
        mixed-model case above: `.policy`/`.model`/`.git_sha` all `None`,
        loud WARNING naming both models and `path`. A cross-model
        misattribution (this arm silently joining a different model's
        artefact) is exactly as wrong as an internally-mixed one, so it
        gets the same refuse-outright treatment, not a load-and-warn.

    `model != run_model_id` is a
    plain, exact string comparison -- no `.strip()`, no `.casefold()`, and
    no override for a hand-produced artefact whose `model` column differs
    from `run_meta.json`'s `serving.model_id` only in case or incidental
    whitespace. This is deliberate, not an oversight: it is the same
    refuse-loudly rule this whole guard exists to enforce -- a fuzzy
    match would mean deciding, silently, which near-miss strings count as
    "the same model", exactly the kind of guess "no silent fallbacks"
    rules out. The only escape hatch is fixing the artefact (or
    `run_meta.json`) itself so the two strings agree byte-for-byte, same
    as the roster's own `model_id` values are already expected to.
    """
    # Pin `game`/`info_set` to Utf8 rather than letting
    # polars infer the dtype from the CSV's own contents (inside
    # `_load_policy_artefact`, explained here since it is a design
    # choice this function's caller relies on, not a mechanical step). Both
    # are join keys (this function's returned dict is keyed by `(game,
    # info_set)`, matched against the parquet's own string columns in
    # `score_property3_exact`) -- polars infers a purely numeric-looking
    # column (e.g. every `info_set` value being "0", "1", "2", ...) as
    # Int64, which then never equality-matches the parquet's Utf8
    # `info_set` column. A synthetic single-row
    # artefact with `info_set="0"` loads with `dtypes == [String, Int64,
    # ...]` and `df["info_set"].to_list() == [0]` (an int) without this
    # pin. The failure mode is silent, not a crash: every info-set in that
    # game reads `matched_fraction=0.000`, which looks exactly like "the
    # model disagreed with the exact pass everywhere" -- a materially false
    # claim, not an obviously-broken one. The real landed arm is unaffected
    # only because its `info_set` column mixes in Dark Hex's non-numeric
    # board strings, which forces polars' whole-column inference to Utf8
    # anyway -- a single-game or `--limit` artefact (exactly the
    # debug/smoke-test shape an arm's collection is likely to produce
    # first) has no such mixing and would trip this silently. `game` pinned
    # too, for the same join-key reason, even though no registered game
    # name is purely numeric (kuhn, leduc, goofspiel, liarsdice,
    # darkhex, ...) -- defensive, and costs nothing. `action`/`prob`/`error` are not join keys (action is
    # cast via `int(a)` below regardless of its inferred dtype, prob via
    # `float(p)`, error is only ever null-checked) and are left to
    # inference.
    pre = _load_policy_artefact(
        path,
        required={"game", "info_set", "action", "prob", "error"},
        artefact_label="exact-policy",
        producer_script="scripts/exact_policy_pass.py",
        consequence_noun="P3_randomisation_exact",
        run_model_id=run_model_id,
    )
    if pre.refusal_reason is not None:
        return ExactPolicyLoad(
            None,
            None,
            None,
            run_model_check=None,
            refusal_reason=pre.refusal_reason,
        )

    ep = pre.df
    exact: dict[tuple[str, str], dict[int, float]] = {}
    for (game, info_set), grp in ep.group_by(["game", "info_set"]):
        if grp["error"].null_count() != len(grp) or grp["prob"].null_count() > 0:
            continue
        exact[(game, info_set)] = {
            int(a): float(p) for a, p in zip(grp["action"], grp["prob"], strict=True)
        }
    return ExactPolicyLoad(
        exact, pre.model, pre.git_sha, run_model_check=pre.run_model_check
    )
