"""Property 7: Bayesian Consistency.

7a: belief error - TVD between LM's stated belief and the true Bayesian posterior.
7b: action-vs-belief - does the LM's action best-respond to its own stated belief?

The two sub-scores can dissociate: high 7a (belief wrong) + low 7b (acts on beliefs)
means the belief is the failure point; low 7a + high 7b (acts against own belief)
means the action-selection mechanism is broken even when belief is correct.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import polars as pl

from lmopt.games.belief_br import pooled_action_values

# Shared duck-typing guard, used here and in property8.py, lives in
# lmopt.scoring._reference so a loosening can't fix one copy and miss the
# other. Aliased to a private name for this module's internal call sites.
from lmopt.scoring._actor import (
    LEGACY_BELIEF_AND_PROMPT,
    actor_label,
    actor_rows,
    belief_drop_report,
    drop_fallback_rows,
    refuse_foreign_probe_type,
    require_actor_column,
)
from lmopt.scoring._belief_alignment import (
    GoofspielSupportViolation,
    # Re-exported, not used here: a test asserts property5 and property7 expose the same
    # helper object, so the goofspiel support guard cannot be duplicated inline again.
    game_row_tvds,
)
from lmopt.scoring._reference import is_partial_reference as _is_partial_reference
from lmopt.scoring._tree import build_infoset_map as _build_infoset_map
from lmopt.scoring.darkhex_belief import (
    aligned_cells,
    belief_action_values,
    best_actions,
    per_cell_abs_error,
    world_weights,
)
from lmopt.scoring.leduc_rank import (
    CARDS_PER_RANK,
    expand_rank_to_indices,
    leduc_num_ranks,
)

#: The slack "attains the best action value" is judged with, the same default
#: `score_property8`'s `epsilon` uses for the identical question about a Dark
#: Hex action's continuation value. One idea of "ties with the best" across
#: every family 7b scores, not one per family: crediting only the lowest-index
#: action of a tie would score a correct choice as wrong, and would reward a
#: model that always answers with the first option listed.
_BEST_ACTION_TOLERANCE = 1e-6

#: What a best-response computation against a stated belief can report.
BR_OK = "ok"
BR_NO_LEGAL = "no_legal"
BR_NO_MASS = "no_mass"


def _combine_keys(stated: dict, true: dict) -> list:
    """Key set for the per-row 7a TVD arrays: union of the stated and true
    dicts' keys. Module-level (not inlined) so a test can monkeypatch
    it to intersection and prove the goofspiel alignment does not
    depend on this choice - see tests/test_scorer_property7.py::
    TestGoofspielZeroPadding::test_zero_padding_survives_intersection_instead_of_union.
    Leduc's rank-marginalisation and goofspiel's explicit zero-padding
    (in `score_property7`) are the load-bearing alignment
    steps; this union is only a fallback for families that need no
    alignment (their key sets already match).
    """
    return sorted(set(stated.keys()) | set(true.keys()))


def score_property7(
    df: pl.DataFrame,
    cfr_policies: Optional[dict] = None,
    games: Optional[dict] = None,
    actor: Optional[str] = None,
    br_cache: Optional[dict] = None,
) -> dict:
    """Compute both sub-scores of Property 7.

	:param df: full match DataFrame.
	:param cfr_policies: {game_name: policy} - needed for 7b EV computation.
		If None, only 7a is computed.
	:param games: {game_name: pyspiel.Game} - needed for 7b. If None, 7b is skipped.
	:param actor: the `actor` column value of the agent under test, or `None` for
		populations without an `actor` column. Those exclude non-LM rows only
		incidentally: 7a selects on the two belief columns, and no
		principled opponent elicits a belief, so a belief-carrying
		opponent would be pooled into the LM's TVD. Naming an actor makes the
		exclusion stated. `None` keeps the legacy selection, which is what
		arms without an `actor` column need.
	:param br_cache:
	:returns: dict with keys:
		"tvd_mean": {game: float} - mean TVD (7a) per game.
		"tvd_per_infoset": {game: {info_set: float}} - per-info-set mean of 7a
			TVDs, pooling exactly the same per-row TVD values the existing 7a
			mean uses (same alignment: leduc rank-marginalisation, goofspiel
			zero-padding, dark hex per-cell alignment). ADD-only: renaming this key
			reddens tests. Consumer rule: same as other per-game dicts (a game
			absent from this dict was skipped for a reason in `not_applicable`).
		"tvd_global_mean": float.
		"action_belief_agreement": {game: float} - fraction matching BR (7b) per game.
			A row scores 1 when its action is among the best actions, never only when it
			is the lowest-index one: several actions can attain the same value under a
			stated belief, and crediting one of them scores a correct choice as wrong.
		"action_belief_per_infoset": {game: {info_set: float}} - per-info-set 7b fraction.
		"action_belief_all_tie_share": {game: float} - the share of 7b-scored rows on
			which every legal action attained the best value, so the row agrees trivially
			whatever the model played. NaN when the game scored no row. A state with one
			legal action is a forced move, not a tie, and is not counted.
		"action_belief_no_mass_rows": {game: int} - rows excluded from 7b because the
			stated belief put positive mass nowhere the best response could use. Such a
			row names no best response, so it is counted here rather than credited.
		"action_belief_global": float.
		"n_belief_rows": {game: int} - rows with both columns non-null.
		"7b_available": bool - whether 7b was computed.
		"not_applicable": {game: str} - a metric was skipped with an explicit
			reason. Two kinds live in this one dict (it is add-only; renaming the
			key reddens the `not_applicable` assertions in
			tests/test_scorer_property7.py):
			  * 7a-scoped (a goofspiel support violation): explains 7a only.
			  * 7b-scoped (PARTIAL depth reference): explains 7b only. The game
				is present in `tvd_mean` (written where `tvd_per_game[
				game_name]` is assigned, above the `_is_partial_reference`
				guard that skips 7b) and absent from
				`action_belief_agreement`.
			The Dark Hex family is scored: its
			belief is per-cell occupancy, so 7a is the mean per-cell absolute
			error (one Bernoulli's TVD, averaged over the cells) and 7b
			best-responds to the world distribution the stated per-cell
			probabilities induce under independence.
			Consumer rule - a reason explains a given metric IFF the game is
			absent from that metric's dict. Do not test membership of this dict
			alone.
		"key_mismatch_rows": {game: {"n_rows": int, "reason": str}} - Dark Hex
			7a rows whose stated cell set and true-posterior cell set differ.
			They are scored, over the cells both name. A game with no mismatch
			is ABSENT, matching `discarded_rows`' absence convention. 7b's own
			equivalent count lives inside its `discarded_rows` entry, because
			7b compares the stated cells against the info-state's unknown
			cells rather than against the posterior's.
		"unscorable_rows": {game: {"n_rows": int, "reason": str}} - Dark Hex
			7a rows whose stated and true cell sets do not intersect at all,
			so no comparison exists. Counted, never defaulted or imputed. A
			game with none is ABSENT.
		"discarded_rows": {game: {"n_rows": int, "reason": str}} - a row
			population that entered the scorer and did not reach the estimand,
			counted with a reason (no silent fallbacks). One
			entry is goofspiel 7b's degenerate/off-support belief rows: a
			stated belief with no positive mass on any currently-legal
			opponent action is discarded rather than silently credited as a
			match (see `_score_7b_goofspiel`). Both seats contribute to
			this game's agreement figure. A
			game with nothing dropped is ABSENT from this dict, so a
			fixed-seat arm grows no spurious caveat.
		"skipped_unparseable_json": {game: int} - count of rows where
			elicited_belief or true_posterior failed JSON parse, made visible
			rather than silently skipped. Games with zero skips
			are ABSENT, matching the discarded_rows absence convention.
		"actor_filter": str - which population produced every number above,
			`"actor=<name>"` or `"legacy-belief-columns-and-prompt-null"`.
		"action_belief_reference_missing": {game: str}: a game whose 7b was
			asked for but whose game object or reference policy was not
			supplied. Its agreement stays NaN; it is not a `not_applicable`
			entry. A game with a reference is ABSENT.
		"action_belief_unscored_rows": {game: {reason: int}}: 7b rows that
			reached no comparison for a reason `discarded_rows` does not name:
			a null parsed_action, a belief that is not a JSON object, no legal
			action, an empty best-response set, and Goofspiel's unparseable
			belief or unreachable info-state. A game with none is ABSENT.
		"tvd_unscored_rows": {game: {reason: int}}: Dark Hex 7a rows whose
			belief or posterior parsed to something other than a JSON object.
			A game with none is ABSENT.

	`br_cache` is an optional dict, `{game: {key: best-response set}}`, that remembers each 7b best
	response by (seat, info-set, stated belief). That set is a pure function of those three for one
	game and one reference policy, so a caller that scores overlapping row sets against the same policy,
	a cell's pooled frame and its shards, passes one dict and computes each distinct set once. The dict
	must never outlive one reference policy: a hit returns the set computed for whichever policy filled
	it. None computes every row afresh and returns the same numbers.

	7b reads the logged `parsed_action`, so the scored LM's harness-fallback rows
	(`fallback_used` set, `prompt` non-null) are removed from its estimate and from
	`action_belief_coverage`'s denominator. 7a scores the model's stated belief,
	which is a separate call from the action, so fallback rows keep their valid
	beliefs and stay in 7a.
	:returns: the keys above plus four additive keys, each nested `{game: ...}` and
		present for every game with a belief row, the dark hex family included:
		`"dropped_rate"` is the action-side fraction of the game's LM belief rows
		dropped from 7b (None when the game has no LM belief row);
		`"dropped_detail"` holds the counts behind it (`DropReport.detail()`);
		`"belief_dropped_rate"` is the belief-side fraction of attempted LM beliefs
		that yielded no usable belief, including rows counted in
		`skipped_unparseable_json` (None when none was attempted);
		`"belief_dropped_detail"` holds the counts behind it.

	Args:
		br_cache:
	"""
    require_actor_column(df, actor, "score_property7")
    refuse_foreign_probe_type(df, "score_property7")
    actor_filter = actor_label(actor, LEGACY_BELIEF_AND_PROMPT)
    belief_rows = actor_rows(
        df,
        actor,
        legacy=pl.col("elicited_belief").is_not_null()
        & pl.col("true_posterior").is_not_null(),
    )
    lm_belief_rows = actor_rows(
        df,
        actor,
        legacy=pl.col("elicited_belief").is_not_null() & pl.col("prompt").is_not_null(),
    )
    if actor is not None:
        # A named actor selects whose rows. Each sub-score still needs the
        # columns it consumes, so the null guards stay as conjuncts: 7a needs
        # both halves of the (stated, true) pair, 7b needs the stated belief it
        # best-responds to.
        belief_rows = belief_rows.filter(
            pl.col("elicited_belief").is_not_null()
            & pl.col("true_posterior").is_not_null()
        )
        lm_belief_rows = lm_belief_rows.filter(pl.col("elicited_belief").is_not_null())
    lm_actor_rows = actor_rows(df, actor, legacy=pl.col("prompt").is_not_null())
    discarded_rows: dict[str, dict] = {}

    dropped_rate: dict[str, float | None] = {}
    dropped_detail: dict[str, dict] = {}
    belief_dropped_rate: dict[str, float | None] = {}
    belief_dropped_detail: dict[str, dict] = {}
    tvd_per_game: dict[str, float] = {}
    tvd_per_infoset: dict[str, dict[str, float]] = {}
    agreement_per_game: dict[str, float] = {}
    agreement_per_infoset: dict[str, dict[str, float]] = {}
    coverage_per_game: dict[str, float] = {}
    all_tie_share: dict[str, float] = {}
    no_mass_rows: dict[str, int] = {}
    n_belief: dict[str, int] = {}
    not_applicable: dict[str, str] = {}
    skipped_unparseable_json: dict[str, int] = {}
    reference_missing: dict[str, str] = {}
    unscored_7b: dict[str, dict[str, int]] = {}
    unscored_7a: dict[str, dict[str, int]] = {}

    can_do_7b = cfr_policies is not None and games is not None

    all_game_names = set(belief_rows["game"].unique().to_list()) | set(
        lm_belief_rows["game"].unique().to_list()
    )

    from lmopt.games.families import GAME_FAMILIES

    key_mismatch_rows: dict[str, dict] = {}
    unscorable_rows: dict[str, dict] = {}

    for game_name in all_game_names:
        # .get with fallback-to-self (not the raising game_family()): offline
        # scorers must tolerate foreign game values in the Parquet (e.g.
        # synthetic test fixtures); registry keys are covered - the map
        # covers every GAMES key and
        # run_match validates game_key at write time.
        family = GAME_FAMILIES.get(game_name, game_name)

        # 7b drops the LM's fallback rows; 7a does not (see the docstring).
        game_drop = drop_fallback_rows(
            lm_belief_rows.filter(pl.col("game") == game_name)
        )
        dropped_rate[game_name] = game_drop.rate
        dropped_detail[game_name] = game_drop.detail()

        # --- 7a: TVD ---
        game_belief = belief_rows.filter(pl.col("game") == game_name)
        # Real per-game rank count for the
        # "leduc" family (leduc_deep's own numRanks, not the canonical 3);
        # leduc_num_ranks falls back to 3 for native "leduc" itself.
        # Read only on the leduc branch of `game_row_tvds`; 3 is its default.
        leduc_n_ranks = leduc_num_ranks(game_name) if family == "leduc" else 3
        n_unparseable = 0
        if family == "darkhex":
            # Dark Hex belief is per-cell occupancy (one independent Bernoulli
            # per unseen cell), so 7a's per-row error is the mean absolute
            # per-cell error rather than a TVD over a simplex -- for one
            # Bernoulli the two coincide, so the per-cell mean is the mean TVD
            # over the cells and belongs in the same result dicts. It does not
            # route through `game_row_tvds`, whose alignment steps are
            # simplex-shaped and which refuses this family by design.
            not_object_7a: dict[str, int] = {}
            records, n_unparseable, n_mismatch, n_unscorable = _darkhex_belief_errors(
                game_belief, not_object=not_object_7a
            )
            if not_object_7a:
                unscored_7a[game_name] = not_object_7a
            if n_mismatch:
                key_mismatch_rows[game_name] = {
                    "n_rows": n_mismatch,
                    "reason": (
                        "the elicited belief and the true posterior name "
                        "different cells; these rows are scored over the cells "
                        "both name, never over a zero-padded union, except a row "
                        "whose two cell sets share no cell, which is also counted "
                        "in unscorable_rows and left unscored"
                    ),
                }
            if n_unscorable:
                unscorable_rows[game_name] = {
                    "n_rows": n_unscorable,
                    "reason": (
                        "the elicited belief and the true posterior share no "
                        "cell, so there is nothing to compare; counted and left "
                        "unscored rather than defaulted or imputed"
                    ),
                }
            _record_7a(
                game_name,
                records,
                n_unparseable,
                tvd_per_game,
                n_belief,
                tvd_per_infoset,
                skipped_unparseable_json,
            )
            _finish_belief_drop(
                game_name,
                lm_actor_rows,
                belief_dropped_rate,
                belief_dropped_detail,
            )
            if cfr_policies is None or games is None:
                continue
            game_lm_belief = game_drop.kept
            if game_lm_belief.is_empty():
                agreement_per_game[game_name] = float("nan")
                continue
            game_obj = games.get(game_name)
            cfr_pol = cfr_policies.get(game_name)
            if game_obj is None or cfr_pol is None:
                agreement_per_game[game_name] = float("nan")
                reference_missing[game_name] = _reference_missing_reason(
                    game_obj, cfr_pol
                )
                continue
            if _is_partial_reference(cfr_pol):
                not_applicable[game_name] = (
                    f"partial reference ({cfr_pol.n_infosets} info-sets): a "
                    "full-tree best-response is not defined against a policy "
                    "with off-support gaps, and the tree is intractable at "
                    "this depth"
                )
                continue
            agr, per_is, coverage, discarded, ties = _score_7b_darkhex(
                game_lm_belief, game_obj, cfr_pol
            )
            agreement_per_game[game_name] = agr
            agreement_per_infoset[game_name] = per_is
            coverage_per_game[game_name] = coverage
            all_tie_share[game_name] = ties["all_tie_share"]
            no_mass_rows[game_name] = ties["n_no_mass"]
            if ties["unscored_exits"]:
                unscored_7b[game_name] = ties["unscored_exits"]
            if discarded:
                discarded_rows[game_name] = discarded
            continue
        try:
            records, n_unparseable = game_row_tvds(
                game_belief,
                family,
                combine_keys=_combine_keys,
                scorer_name="score_property7",
                game_name=game_name,
                leduc_n_ranks=leduc_n_ranks,
            )
        except GoofspielSupportViolation as exc:
            # Confine the blast radius to this game's 7a only. 7b's inputs
            # (game_lm_belief/game_obj/cfr_pol) are built independently and
            # do not depend on game_row_tvds or `records` at all, so a
            # 7a-only violation must not zero a game's 7b too. Record the 7a
            # reason and fall through to 7b (the `else:` below holds the 7a
            # recording). 0 of 899,894 real goofspiel rows (180 files) trip
            # this guard.
            not_applicable[game_name] = (
                f"goofspiel: true_posterior key(s) {sorted(exc.offending_keys)} "
                f"absent from elicited belief on {exc.n_violations} row(s) - "
                "investigate before trusting this game's P7a"
            )
        else:
            _record_7a(
                game_name,
                records,
                n_unparseable,
                tvd_per_game,
                n_belief,
                tvd_per_infoset,
                skipped_unparseable_json,
            )

        _finish_belief_drop(
            game_name,
            lm_actor_rows,
            belief_dropped_rate,
            belief_dropped_detail,
        )

        # --- 7b: action-vs-belief --- (runs regardless of a 7a-only
        # violation above -- independent inputs, no dependency on `records`)
        if cfr_policies is None or games is None:
            continue

        game_lm_belief = game_drop.kept
        if game_lm_belief.is_empty():
            agreement_per_game[game_name] = float("nan")
            continue


        game_obj = games.get(game_name)
        cfr_pol = cfr_policies.get(game_name)
        if game_obj is None or cfr_pol is None:
            agreement_per_game[game_name] = float("nan")
            reference_missing[game_name] = _reference_missing_reason(game_obj, cfr_pol)
            continue

        if _is_partial_reference(cfr_pol):
            # No silent fallbacks: record a reason, never NaN (a
            # skip signal elsewhere in this codebase) and never a bare
            # `continue` - a full-tree best-response is not defined against a
            # policy with off-support gaps, and the tree is intractable at
            # this depth.
            not_applicable[game_name] = (
                f"partial reference ({cfr_pol.n_infosets} info-sets): a full-tree "
                "best-response is not defined against a policy with off-support "
                "gaps, and the tree is intractable at this depth"
            )
            continue

        # Goofspiel: simultaneous-move BR to the stated bid-belief, over the
        # reach-weighted CFR+ continuation (the card-game private-deal path below
        # does not apply - Goofspiel has no dealt private card).
        if family == "goofspiel":
            # Coverage is row-granularity on this path as on the generic one below: the
            # rows that reached the estimand over the rows that entered it.
            agr, per_is, n_discarded, ties, n_scored = _score_7b_goofspiel(
                game_lm_belief,
                game_obj,
                cfr_pol,
                br_cache=(
                    None if br_cache is None else br_cache.setdefault(game_name, {})
                ),
            )
            agreement_per_game[game_name] = agr
            agreement_per_infoset[game_name] = per_is
            all_tie_share[game_name] = ties["all_tie_share"]
            n_candidates = game_lm_belief.height
            coverage_per_game[game_name] = (
                (n_scored / n_candidates) if n_candidates else 0.0
            )
            no_mass_rows[game_name] = n_discarded
            if ties["unscored_exits"]:
                unscored_7b[game_name] = ties["unscored_exits"]
            if n_discarded:
                discarded_rows[game_name] = {
                    "n_rows": n_discarded,
                    "reason": (
                        "goofspiel 7b: elicited_belief carried no positive "
                        "mass on any currently-legal opponent action (empty, "
                        "all-zero, or entirely off-support). "
                        "goofspiel_br_set/goofspiel_br_set_seat1 return the "
                        "FULL legal action set (not empty) in this case, so a "
                        "bare `if not br: continue` guard cannot detect it - "
                        "these rows are counted and excluded from "
                        "`matches`/`total` rather than silently credited as a "
                        "hit."
                    ),
                }
            continue

        matches = 0
        total = 0
        n_all_tie = 0
        n_no_mass = 0
        # Rows that leave the loop without being scored, by reason, so none leaves silently.
        exits: dict[str, int] = {}
        # Exits the reported `discarded_rows` reason does not name, counted apart so that
        # reason, and every number beside it, reads as before.
        new_exits: dict[str, int] = {}
        # Per-info-set tracking: {info_set: [matches, total]}
        is_counts: dict[str, list[int]] = {}
        for item in _card_game_scored_rows(
            game_lm_belief,
            game_obj,
            family,
            cfr_pol,
            br_cache=(None if br_cache is None else br_cache.setdefault(game_name, {})),
        ):
            if item is _CARD_NO_MASS:
                n_no_mass += 1
            elif isinstance(item, _CardGameExit):
                _count(
                    exits if item in _DISCARDED_NAMED_EXITS else new_exits, str(item)
                )
            else:
                info_str, parsed_action, br, state = item
                n_all_tie += int(_is_all_tie(br, state.legal_actions()))
                hit = int(parsed_action in br)
                matches += hit
                total += 1
                counts = is_counts.setdefault(info_str, [0, 0])
                counts[0] += hit
                counts[1] += 1

        all_tie_share[game_name] = n_all_tie / total if total > 0 else float("nan")
        no_mass_rows[game_name] = n_no_mass
        if new_exits:
            unscored_7b[game_name] = new_exits
        if exits:
            discarded_rows[game_name] = {
                "n_rows": sum(exits.values()),
                "reason": "card-game 7b rows not scored: "
                + "; ".join(f"{n} {why}" for why, n in sorted(exits.items())),
            }
        agreement_per_game[game_name] = matches / total if total > 0 else float("nan")
        agreement_per_infoset[game_name] = {
            is_: m / t for is_, (m, t) in is_counts.items() if t > 0
        }
        # Coverage signal, row-granularity here (raw 7b iterates rows, not
        # unique info-sets), in "resolved / candidate" shape.
        n_candidates = game_lm_belief.height
        coverage_per_game[game_name] = (total / n_candidates) if n_candidates else 0.0

    valid_tvd = [v for v in tvd_per_game.values() if not math.isnan(v)]
    global_tvd = float(np.mean(valid_tvd)) if valid_tvd else float("nan")

    valid_agree = [v for v in agreement_per_game.values() if not math.isnan(v)]
    global_agree = float(np.mean(valid_agree)) if valid_agree else float("nan")

    return {
        "tvd_mean": tvd_per_game,
        "tvd_per_infoset": tvd_per_infoset,
        "tvd_global_mean": global_tvd,
        "action_belief_agreement": agreement_per_game,
        "action_belief_per_infoset": agreement_per_infoset,
        "action_belief_coverage": coverage_per_game,
        "action_belief_all_tie_share": all_tie_share,
        "action_belief_no_mass_rows": no_mass_rows,
        "action_belief_global": global_agree,
        "n_belief_rows": n_belief,
        "7b_available": can_do_7b,
        "not_applicable": not_applicable,
        "discarded_rows": discarded_rows,
        "key_mismatch_rows": key_mismatch_rows,
        "unscorable_rows": unscorable_rows,
        "skipped_unparseable_json": skipped_unparseable_json,
        "actor_filter": actor_filter,
        "dropped_rate": dropped_rate,
        "dropped_detail": dropped_detail,
        "belief_dropped_rate": belief_dropped_rate,
        "belief_dropped_detail": belief_dropped_detail,
        "action_belief_reference_missing": reference_missing,
        "action_belief_unscored_rows": unscored_7b,
        "tvd_unscored_rows": unscored_7a,
    }


def _reference_missing_reason(game_obj, cfr_pol) -> str:
    """
    why a game's 7b has no value when its game object or reference policy was not supplied
    :param game_obj: the game object the caller passed for this game, or None (pyspiel.Game | None)
    :param cfr_pol: the reference policy the caller passed for this game, or None (policy | None)
    :returns: the reason, naming which of the two is missing (str)
    """
    missing = [
        name
        for name, value in (("game object", game_obj), ("reference policy", cfr_pol))
        if value is None
    ]
    return (
        f"no {' and no '.join(missing)} was supplied for this game, so 7b has no "
        f"reference to best-respond against; its agreement reads NaN"
    )


def _count(counter: dict[str, int], reason: str) -> None:
    """
    add one row to a named exit count
    :param counter: the exit counts by reason, mutated in place (dict[str, int])
    :param reason: why the row left unscored (str)
    :returns: None
    """
    counter[reason] = counter.get(reason, 0) + 1


# ---------------------------------------------------------------------------
# Helpers shared by 7a
# ---------------------------------------------------------------------------


def _record_7a(
    game_name,
    records,
    n_unparseable,
    tvd_per_game,
    n_belief,
    tvd_per_infoset,
    skipped_unparseable_json,
) -> None:
    """
    write one game's per-row belief errors into 7a's result dicts
    One definition for every family: the per-row quantity differs (a simplex
    TVD for the card games, a mean per-cell absolute error for Dark Hex) but
    what is done with it -- the game mean, the per-info-set means, the row
    count and the unparseable count -- must not.
    :param game_name: the game these records belong to (str)
    :param records: per-row `{"info_set": str, "tvd": float}` dicts (list)
    :param n_unparseable: rows whose belief or posterior JSON failed to parse (int)
    :param tvd_per_game: the `{game: mean}` dict to write into (dict)
    :param n_belief: the `{game: n_rows}` dict to write into (dict)
    :param tvd_per_infoset: the `{game: {info_set: mean}}` dict to write into (dict)
    :param skipped_unparseable_json: the `{game: count}` dict, written only
        when the count is positive (dict)
    :returns: None; the four dicts are mutated in place
    """
    tvds = [rec["tvd"] for rec in records]
    per_is: dict[str, list[float]] = {}
    for rec in records:
        per_is.setdefault(rec["info_set"], []).append(rec["tvd"])
    tvd_per_game[game_name] = float(np.mean(tvds)) if tvds else float("nan")
    n_belief[game_name] = len(tvds)
    tvd_per_infoset[game_name] = {
        is_: float(np.mean(vals)) for is_, vals in per_is.items() if vals
    }
    if n_unparseable > 0:
        skipped_unparseable_json[game_name] = n_unparseable


def _finish_belief_drop(
    game_name,
    lm_actor_rows,
    belief_dropped_rate,
    belief_dropped_detail,
) -> None:
    """
    record one game's belief-side dropped rate and the counts behind it
    :param game_name: the game to report on (str)
    :param lm_actor_rows: the scored actor's LM rows across every game (pl.DataFrame)
    :param belief_dropped_rate: the `{game: rate}` dict to write into (dict)
    :param belief_dropped_detail: the `{game: counts}` dict to write into (dict)
    :returns: None; both dicts are mutated in place
    """
    report = belief_drop_report(lm_actor_rows.filter(pl.col("game") == game_name))
    belief_dropped_rate[game_name] = report.rate
    belief_dropped_detail[game_name] = report.detail()


def _darkhex_belief_errors(
    game_belief, not_object: Optional[dict] = None
) -> tuple[list, int, int, int]:
    """
    Dark Hex 7a: one mean per-cell absolute error per row

    The cells scored are the ones the stated belief and the true posterior both
    name. A row whose two cell sets merely differ is still scored, over the
    shared cells, and counted; a row whose cell sets do not intersect reaches no
    comparison and is counted unscored. Neither is defaulted, imputed or
    silently dropped (no silent fallbacks).

    :param game_belief: one game's rows, both belief columns non-null
        (pl.DataFrame)
    :param not_object: a reason-to-count dict that a row whose belief or
        posterior parses to something other than a JSON object is counted in;
        None raises on such a row instead (dict | None)
    :returns: (records, n_unparseable, n_key_mismatch, n_unscorable), where
        records are `{"info_set": str, "tvd": float}` dicts shaped exactly like
        `_belief_alignment.game_row_tvds`' so 7a's recording path is shared
    """
    records: list[dict] = []
    n_unparseable = n_mismatch = n_unscorable = 0
    for row in game_belief.iter_rows(named=True):
        try:
            stated = json.loads(row["elicited_belief"])
            true = json.loads(row["true_posterior"])
        except (json.JSONDecodeError, TypeError):
            n_unparseable += 1
            continue
        if not isinstance(stated, dict) or not isinstance(true, dict):
            if not_object is None:
                raise ValueError(
                    f"a Dark Hex belief row at info_set {row.get('info_set')!r} parsed to "
                    f"something other than a JSON object; pass a `not_object` counter to "
                    f"count such rows"
                )
            _count(not_object, "elicited_belief or true_posterior is not a JSON object")
            continue
        if set(stated) != set(true):
            n_mismatch += 1
        cells = aligned_cells(stated, true)
        if not cells:
            n_unscorable += 1
            continue
        records.append(
            {
                "info_set": row.get("info_set", ""),
                "tvd": per_cell_abs_error(stated, true, cells),
            }
        )
    return records, n_unparseable, n_mismatch, n_unscorable


# ---------------------------------------------------------------------------
# Helpers for 7b
# ---------------------------------------------------------------------------


def _darkhex_world_action_values(worlds_for_info, game_obj, cfr_pol, seat):
    """
    one info-state's worlds as (opponent-occupied cells, per-action value)

    The value is `policy_value(state.child(a), [cfr_pol, cfr_pol])[seat]` --
    exactly the integrand `property8._voi_and_action_values` sums to get
    V(a|I), read at the acting seat's own index because Dark Hex is zero-sum.
    Only the weights differ between the two scorers: P8 weights by the true
    reach-weighted posterior mu(w|I), 7b weights by the LM's own stated belief.

    :param worlds_for_info: `[(state, opponent_reach), ...]` from
        `darkhex_posterior.gather_worlds` (list)
    :param game_obj: the OpenSpiel Dark Hex game (pyspiel.Game)
    :param cfr_pol: the full CFR+ reference policy for `game_obj`
    :param seat: the acting seat the values are read at, 0 or 1 (int)
    :returns: (legal_actions, [(occupied_cells, opponent_reach, {action: value}), ...])
    """
    from open_spiel.python.algorithms.expected_game_score import policy_value

    from lmopt.scoring.darkhex_posterior import opponent_stone_cells

    legal = worlds_for_info[0][0].legal_actions()
    per_world = []
    for state, reach in worlds_for_info:
        values = {
            a: policy_value(state.child(a), [cfr_pol, cfr_pol])[seat] for a in legal
        }
        per_world.append(
            (opponent_stone_cells(state, game_obj, seat=seat), reach, values)
        )
    return legal, per_world


def _score_7b_darkhex(game_lm_belief, game_obj, cfr_pol, row_score=None) -> tuple:
    """
    Dark Hex 7b: does the executed move best-respond to the LM's own belief?

    The stated per-cell probabilities are turned into a distribution over the
    opponent boards consistent with the row's info-state under an independence
    assumption, and then across the worlds of each board by opponent reach
    (`darkhex_belief.world_weights` -- its docstring names what both steps
    cost). Each action's expected continuation value is taken over those
    worlds with the reference's own value routine, and the row scores 1 when
    the logged `parsed_action` is among the best actions within the same
    tolerance property 8 uses for "attains the best action value".

    The cells the weighting runs over are the ones the stated belief and the
    info-state's own unknown cells both name, so a stated probability for a
    cell the observer can already see never enters, and a cell the model did
    not speak about is never given a fabricated probability.

    :param game_lm_belief: the game's LM belief rows, fallback rows already
        dropped (pl.DataFrame)
    :param game_obj: the OpenSpiel Dark Hex game (pyspiel.Game)
    :param cfr_pol: the full CFR+ reference policy for `game_obj`
    :param row_score: an optional `(info_set, best-response set, logged action) -> float | None` that
        replaces the sampled hit (1 when the logged action is a best response, else 0) with another
        per-row credit, as the exact-policy 7b does; None from it leaves the row out of the comparison
        and out of the tie count. None (the default) keeps the sampled hit exactly (callable | None)
    :returns: (agreement, {info_set: fraction}, coverage, discarded, ties), where
        coverage is scored rows over candidate rows, `discarded` is the
        reason dict for the rows that reached no comparison, or `{}` when none
        did, and `ties` holds the share of scored rows on which every legal
        action attained the best value, the count of rows whose stated
        belief gave no world any weight, and under "unscored_exits" the
        {reason: count} of rows `discarded` does not name (a belief that is
        not a JSON object, a null parsed_action)
    """
    from lmopt.scoring.darkhex_posterior import gather_worlds

    worlds_by_seat: dict[int, dict] = {}
    cache: dict[tuple[int, str], tuple] = {}
    matches = total = 0
    n_all_tie = n_no_mass = 0
    n_missing_info = n_unscorable = n_mismatch = 0
    unscored_exits: dict[str, int] = {}
    is_counts: dict[str, list[int]] = {}
    for row in game_lm_belief.iter_rows(named=True):
        seat = int(row["player_to_act"])
        if seat not in (0, 1):
            continue  # unreachable for a 2-player game; nothing to score
        try:
            raw = json.loads(row["elicited_belief"])
        except (json.JSONDecodeError, TypeError):
            n_unscorable += 1
            continue
        if not isinstance(raw, dict):
            _count(unscored_exits, _EXIT_NOT_OBJECT)
            continue
        stated: dict[int, float] = {}
        for k, v in raw.items():
            try:
                stated[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
        info_str = row["info_set"]
        if seat not in worlds_by_seat:
            worlds_by_seat[seat] = gather_worlds(game_obj, cfr_pol, seat=seat)
        entries = worlds_by_seat[seat].get(info_str)
        if not entries:
            n_missing_info += 1
            continue
        if (seat, info_str) not in cache:
            cache[(seat, info_str)] = _darkhex_world_action_values(
                entries, game_obj, cfr_pol, seat
            )
        legal, per_world = cache[(seat, info_str)]
        cells = aligned_cells(stated, {a: None for a in legal})
        if set(stated) != set(legal):
            n_mismatch += 1
        if not cells:
            n_unscorable += 1
            continue
        weights = world_weights(
            stated,
            [occ for occ, _, _ in per_world],
            [reach for _, reach, _ in per_world],
            cells,
        )
        if not weights:
            n_unscorable += 1
            n_no_mass += 1
            continue
        values = belief_action_values(
            [vals for _, _, vals in per_world], weights, legal
        )
        br = best_actions(values, _BEST_ACTION_TOLERANCE)
        if not br:
            n_unscorable += 1
            continue
        if row["parsed_action"] is None:
            _count(unscored_exits, _EXIT_NULL_ACTION)
            continue
        logged = int(row["parsed_action"])
        hit = (
            int(logged in br) if row_score is None else row_score(info_str, br, logged)
        )
        if hit is None:
            continue
        n_all_tie += int(_is_all_tie(br, legal))
        matches += hit
        total += 1
        counts = is_counts.setdefault(info_str, [0, 0])
        counts[0] += hit
        counts[1] += 1

    agreement = matches / total if total > 0 else float("nan")
    per_infoset = {is_: m / t for is_, (m, t) in is_counts.items() if t > 0}
    n_candidates = game_lm_belief.height
    coverage = (total / n_candidates) if n_candidates else 0.0
    discarded: dict = {}
    if n_missing_info or n_unscorable or n_mismatch:
        discarded = {
            "n_rows": n_missing_info + n_unscorable,
            "n_info_set_absent_from_worlds": n_missing_info,
            "n_no_scorable_belief": n_unscorable,
            "n_cell_set_mismatch": n_mismatch,
            "reason": (
                "a row reaches no comparison when its info_set is absent from "
                "the worlds map built for its acting seat, when the stated "
                "belief does not parse, when it names none of the info-state's unknown cells, or when "
                "the stated belief gives zero weight to every consistent "
                "world; each is counted rather than credited as a hit. A row "
                "whose stated cells differ from the unknown cells is counted in "
                "n_cell_set_mismatch; it IS scored over the cells both name when "
                "they share one, and when they share none it is also counted in "
                "n_no_scorable_belief and left unscored"
            ),
        }
    ties = {
        "all_tie_share": n_all_tie / total if total > 0 else float("nan"),
        "n_no_mass": n_no_mass,
        "unscored_exits": unscored_exits,
    }
    return agreement, per_infoset, coverage, discarded, ties


#: What `_goofspiel_scored_rows` yields for a row it discards for a degenerate belief.
_DEGENERATE_BELIEF = object()


class _GoofspielExit(str):
    """
    what `_goofspiel_scored_rows` yields for a row that reaches no best response for a reason
    other than a degenerate belief; the string is the reason a caller counts it under
    """


_GOOF_NOT_JSON = _GoofspielExit("elicited_belief is not JSON")
_GOOF_NOT_OBJECT = _GoofspielExit("elicited_belief is not a JSON object")
_GOOF_NULL_ACTION = _GoofspielExit("parsed_action is null")
_GOOF_NO_STATE = _GoofspielExit("info_set is unreachable in the reference's worlds")
_GOOF_EMPTY_BR = _GoofspielExit("the best-response set is empty")


def _goofspiel_scored_rows(game_lm_belief, game_obj, cfr_pol, br_cache=None):
    """
    the goofspiel 7b rows that reach a best response, one at a time
    Both the sampled and the exact 7b read their rows from here, so the seat handling, the
    unreachable-info-state skip, the degenerate-belief guard and the best-response set are one
    code path.
    :param game_lm_belief: the game's LM belief rows (pl.DataFrame)
    :param game_obj: the pyspiel goofspiel game
    :param cfr_pol: the CFR+ continuation policy
    :param br_cache: an optional dict that remembers each best-response set by (seat, info-set, stated
        belief). The set is a pure function of those three for one game and one policy, so a caller that
        scores overlapping row sets against the same policy (a pooled cell and its shards) passes one
        dict and computes each distinct set once; None computes every row afresh (dict | None)
    :returns: a generator of (info_set, parsed_action, best-response set, the seat's own legal bids), or
        `_DEGENERATE_BELIEF` for a row whose belief carries no mass on a legal opponent action, or a
        `_GoofspielExit` naming why any other row reached no best response
    """
    from lmopt.scoring.goofspiel_p7b import (
        gather_p0_worlds,
        gather_p1_prefix_worlds,
        goofspiel_br_set,
        goofspiel_br_set_seat1,
    )

    worlds0 = gather_p0_worlds(game_obj, cfr_pol)  # {p0_info: [(state, reach)]}
    worlds1 = None  # built lazily: most cells (fixed-seat arms) carry no
    # seat-1 rows at all, and the tree walk is not free.
    for row in game_lm_belief.iter_rows(named=True):
        seat = int(row["player_to_act"])
        if seat not in (0, 1):
            continue  # unreachable for a 2-player game; nothing to score
        try:
            stated = json.loads(row["elicited_belief"])
        except (json.JSONDecodeError, TypeError):
            yield _GOOF_NOT_JSON
            continue
        if not isinstance(stated, dict):
            yield _GOOF_NOT_OBJECT
            continue
        info_str = row["info_set"]
        if row["parsed_action"] is None:
            yield _GOOF_NULL_ACTION
            continue
        parsed_action = int(row["parsed_action"])
        belief: dict[int, float] = {}
        for k, v in stated.items():
            try:
                belief[int(k)] = float(v)
            except (TypeError, ValueError):
                pass

        if seat == 0:
            entries = worlds0.get(info_str, [])
        else:
            if worlds1 is None:
                worlds1 = gather_p1_prefix_worlds(game_obj, cfr_pol)
            entries = worlds1.get(info_str, [])
        if not entries:
            yield _GOOF_NO_STATE  # unreachable info-state; not a degenerate-belief case
            continue

        legal_opp = _goofspiel_legal_opponent_actions(entries, seat)
        if not any(belief.get(a, 0.0) > 0.0 for a in legal_opp):
            yield _DEGENERATE_BELIEF
            continue

        cache_key = (seat, info_str, tuple(sorted(belief.items())))
        if br_cache is not None and cache_key in br_cache:
            br = br_cache[cache_key]
        else:
            br = (
                goofspiel_br_set(entries, cfr_pol, belief)
                if seat == 0
                else goofspiel_br_set_seat1(entries, cfr_pol, belief)
            )
            if br_cache is not None:
                br_cache[cache_key] = br
        if not br:
            yield _GOOF_EMPTY_BR
            continue
        yield info_str, parsed_action, br, _goofspiel_own_bids(entries, seat)


def _score_7b_goofspiel(
    game_lm_belief, game_obj, cfr_pol, br_cache=None
) -> tuple[float, dict, int, dict, int]:
    """Goofspiel 7b: fraction of nodes, both seats, where the executed bid
    best-responds to the stated bid-belief under the CFR+ continuation
    (reach-weighted worlds).

    Seat 0 uses `gather_p0_worlds`/`goofspiel_br_set`. Seat 1 uses
    `gather_p1_prefix_worlds`/`goofspiel_br_set_seat1` - a genuinely
    different summation, not a seat swap of the seat-0 machinery: in
    OpenSpiel's turn-based conversion of simultaneous Goofspiel, P0 bids
    first, so at a P0 info-state the opponent's bid is a child action and
    `goofspiel_br_set` descends `state.child(b0).child(b1)`. At a P1
    info-state that bid is already applied and hidden, so it indexes the
    world instead of being a child action, and the continuation value is
    read at the other index (see `goofspiel_br_set_seat1`'s docstring). Both
    branches accumulate into the same `matches`/`total`/`is_counts`, so the
    returned `agreement` is one unified figure over both seats
    (property8's `discarded_rows` is an unrelated, differently-shaped dict).

    A second, symmetric-across-both-seats guard catches a degenerate or
    off-support `elicited_belief`: when a belief carries no positive mass on
    any currently-legal opponent action, `goofspiel_br_set`/
    `goofspiel_br_set_seat1` return the full legal action set rather than an
    empty one (every candidate action's EV ties at 0.0 - confirmed by direct
    execution against a real fixture),
    so a bare `if not br: continue` guard never fires for this case and
    would silently credit the row as a hit no matter what was played. This
    function checks the belief's support against the legal opponent actions
    before calling either BR-set function, so such rows are counted and
    excluded from the estimand rather than scored (no silent
    fallbacks) - including a belief whose only keys are off-support, and one
    whose keys are on-support but all zero-mass, not merely an empty dict.

    Returns (agreement_fraction, {info_set: fraction},
    n_discarded_degenerate_belief, {"all_tie_share": float, "unscored_exits": {reason: int}},
    n_scored_rows). `n_scored_rows` is
    the row count behind the agreement, the numerator of the game's `action_belief_coverage`,
    which the caller divides by the rows that entered, exactly as the generic path does.
    """
    matches = 0
    total = 0
    n_degenerate_belief = 0
    n_all_tie = 0
    unscored_exits: dict[str, int] = {}
    is_counts: dict[str, list[int]] = {}
    for item in _goofspiel_scored_rows(
        game_lm_belief, game_obj, cfr_pol, br_cache=br_cache
    ):
        if item is _DEGENERATE_BELIEF:
            n_degenerate_belief += 1
            continue
        if isinstance(item, _GoofspielExit):
            _count(unscored_exits, str(item))
            continue
        info_str, parsed_action, br, own_bids = item
        n_all_tie += int(_is_all_tie(set(br), own_bids))
        hit = int(parsed_action in br)
        matches += hit
        total += 1
        counts = is_counts.setdefault(info_str, [0, 0])
        counts[0] += hit
        counts[1] += 1

    agreement = matches / total if total > 0 else float("nan")
    per_infoset = {is_: m / t for is_, (m, t) in is_counts.items() if t > 0}
    ties = {
        "all_tie_share": n_all_tie / total if total > 0 else float("nan"),
        "unscored_exits": unscored_exits,
    }
    return agreement, per_infoset, n_degenerate_belief, ties, total


def _goofspiel_own_bids(entries: list, seat: int) -> list:
    """
    the bids the scored seat chooses between at one information state
    Derived exactly as the two BR-set functions derive their own candidate set, so
    "every candidate tied" is asked of the same set the best response was chosen from.
    :param entries: the (state, reach) worlds of the information state
    :param seat: the scored seat, 0 or 1 (int)
    :returns: the seat's legal current-round bids
    """
    probe = entries[0][0]
    if seat == 0:
        return probe.legal_actions()
    return probe.child(probe.legal_actions()[0]).legal_actions()


def _goofspiel_legal_opponent_actions(entries: list, seat: int) -> set[int]:
    """The union, over every reach>0 world sharing this info-state, of the
    opponent's legal current-round bids - the action set an `elicited_belief`
    must place positive mass on to count as having real support.

    Not a single arbitrary world (``entries[0][0]``):
    `goofspiel_br_set`/`goofspiel_br_set_seat1` recompute the
    opponent's legal actions PER world they sum over
    (`goofspiel_p7b.py`'s `after_b0.legal_actions()` inside the
    `for state, reach in worlds_for_info` loop, and
    `prefix_state.legal_actions()` likewise). Different histories reaching
    the same info-state can have consumed different opponent cards, so the
    opponent's remaining hand - and hence legal actions - genuinely differs
    across worlds sharing one info-state. A single-world probe is therefore a
    strict subset of what the BR functions actually consult: on a real
    CFR+-solved 4-card fixture, the single-world probe
    disagrees with this union on 40/81 P0 info-states and 40/81 P1 info-states,
    and on the qwen2.5-14b-instruct/recycle-v4/M0 real data it false-narrows
    every one of 400 reported "degenerate belief" discards. This function mirrors the
    BR functions' own accumulation exactly (reach<=0 worlds skipped the same
    way), so probe ⊆ actually-consulted is structural, not incidental.

    Seat 0: each `state` is a P0-to-act node from `gather_p0_worlds`; the
    opponent is P1, whose bid is a child action, so `state.child(b0)` reaches
    a P1 node for each of P0's own legal `b0` (`legal_b0`, taken from the
    first world only - mirroring `goofspiel_br_set`'s own `legal_b0 =
    worlds_for_info[0][0].legal_actions()  # P0 hand: same across worlds`
    assumption).

    Seat 1: each `state` is a P0-to-act prefix node from
    `gather_p1_prefix_worlds`, immediately before the current round's P0 bid;
    the opponent is P0, and the bid under debate is that prefix state's own
    `legal_actions()` directly - no descent needed.
    """
    legal_opp: set[int] = set()
    if seat == 0:
        legal_b0 = entries[0][0].legal_actions()  # P0 hand: same across worlds
        for state, reach in entries:
            if reach <= 0.0:
                continue
            for b0 in legal_b0:
                legal_opp.update(state.child(b0).legal_actions())
    else:
        for state, reach in entries:
            if reach <= 0.0:
                continue
            legal_opp.update(state.legal_actions())
    return legal_opp


def _leduc_dealt_cards(state) -> list[int]:
    """Replay `state`'s full history from a fresh state, collecting every
    action applied while the state was at a chance node - i.e. every dealt
    card, in deal order: `[p0_private, p1_private, board?]` (the board
    entry is present only once round 2 has begun). Deal order is fixed by
    the game (P0's private card first, then P1's), which is also what
    `_br_to_belief`'s `history[seat]` already assumes.
    """
    game = state.get_game()
    s = game.new_initial_state()
    dealt = []
    for a in state.history():
        if s.is_chance_node():
            dealt.append(a)
        s.apply_action(a)
    return dealt


def _leduc_excluded_indices(state, seat: int) -> frozenset[int]:
    """Card indices the opponent of `seat` provably cannot hold at `state`:
    `seat`'s own dealt card, plus the public board card once revealed."""
    dealt = _leduc_dealt_cards(state)
    excluded = {dealt[seat]}
    if len(dealt) > 2:
        excluded.add(dealt[2])
    return frozenset(excluded)


#: Why a card-game 7b row can leave the scoring loop unscored. Each is counted into
#: `discarded_rows`, so an unscored row is a reported number and never a silent skip.
_EXIT_BELIEF_NOT_JSON = "elicited_belief is not JSON"
_EXIT_NO_STATE = "info_set has no representative state in the game tree"
_EXIT_NOT_MAPPED = "the belief could not be mapped onto card indices"
#: Exits counted in `action_belief_unscored_rows`, which `discarded_rows` does not name.
_EXIT_NULL_ACTION = "parsed_action is null"
_EXIT_NOT_OBJECT = "elicited_belief is not a JSON object"
_EXIT_NO_LEGAL = "the info-state has no legal action"
_EXIT_EMPTY_BR = "the best-response set is empty"
_EXIT_NO_MASS = "the stated belief puts no mass on any card"


class _CardGameExit(str):
    """
    what `_card_game_scored_rows` yields for a row that reaches no best response; the string is
    the reason a caller counts it under
    """


_CARD_NOT_JSON = _CardGameExit(_EXIT_BELIEF_NOT_JSON)
_CARD_NULL_ACTION = _CardGameExit(_EXIT_NULL_ACTION)
_CARD_NO_STATE = _CardGameExit(_EXIT_NO_STATE)
_CARD_NOT_MAPPED = _CardGameExit(_EXIT_NOT_MAPPED)
_CARD_NO_MASS = _CardGameExit(_EXIT_NO_MASS)
_CARD_NO_LEGAL = _CardGameExit(_EXIT_NO_LEGAL)
_CARD_EMPTY_BR = _CardGameExit(_EXIT_EMPTY_BR)
#: The exits the sampled 7b `discarded_rows` reason names; every other exit is counted apart.
_DISCARDED_NAMED_EXITS = frozenset(
    (_EXIT_BELIEF_NOT_JSON, _EXIT_NO_STATE, _EXIT_NOT_MAPPED)
)


def _card_game_scored_rows(
    game_lm_belief, game_obj, family, cfr_pol, br_cache=None, needs_action=True
):
    """
    the card-game 7b rows that reach a best response, one at a time
    Both the sampled and the exact 7b read their Kuhn and Leduc rows from here, so the
    info-set lookup, the belief mapping, the best-response set and the reason every other row
    leaves unscored are one code path.
    :param game_lm_belief: the game's LM belief rows (pl.DataFrame)
    :param game_obj: the pyspiel card game
    :param family: the game's family, which selects the belief mapping (str)
    :param cfr_pol: the CFR+ continuation policy
    :param br_cache: an optional dict that remembers each best-response set and status by (seat,
        info-set, stated belief). The pair is a pure function of those three for one game and one
        policy, so a caller that scores overlapping row sets passes one dict; None computes every
        row afresh (dict | None)
    :param needs_action: whether a row must carry a parsed action; the exact pass scores the
        action's distribution, not the action, so it passes False (bool)
    :returns: a generator of (info_set, parsed action or None, best-response set, the info-set's
        representative state), or a `_CardGameExit` naming why any other row reached no best
        response; `_CARD_NO_MASS` is the exit for a belief with no mass on any card
    """
    infoset_map = _build_infoset_map(game_obj)
    for row in game_lm_belief.iter_rows(named=True):
        try:
            stated = json.loads(row["elicited_belief"])
        except (json.JSONDecodeError, TypeError):
            yield _CARD_NOT_JSON
            continue
        player = int(row["player_to_act"])
        info_str = row["info_set"]
        parsed_action = None
        if needs_action:
            if row["parsed_action"] is None:
                yield _CARD_NULL_ACTION
                continue
            parsed_action = int(row["parsed_action"])
        state = infoset_map.get((player, info_str))
        if state is None:
            yield _CARD_NO_STATE
            continue
        belief_arr = _parse_belief_array(
            stated, game_obj, family, state=state, seat=player
        )
        if belief_arr is None:
            yield _CARD_NOT_MAPPED
            continue
        cache_key = (player, info_str, belief_arr.tobytes())
        if br_cache is not None and cache_key in br_cache:
            br, status = br_cache[cache_key]
        else:
            br, status = _br_set_to_belief(state, player, belief_arr, cfr_pol)
            if br_cache is not None:
                br_cache[cache_key] = (br, status)
        if status == BR_NO_MASS:
            yield _CARD_NO_MASS
        elif status == BR_NO_LEGAL:
            yield _CARD_NO_LEGAL
        elif not br:
            yield _CARD_EMPTY_BR
        else:
            yield info_str, parsed_action, br, state


def _parse_belief_array(
    stated: dict,
    game,
    family: Optional[str] = None,
    state=None,
    seat: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Convert a stated belief dict to a numpy array indexed by card.

    Leduc's stated belief is a 3-key rank simplex, but `_br_to_belief`'s
    counterfactuals are built against specific card indices (6, two per
    rank) - the CFR+ policy and OpenSpiel state history are indexed by
    dealt card, not rank. Writing `arr[int(k)] = v` for k in {"0","1","2"}
    would put all the rank mass on indices 0-2 and leave 3-5 (half the deck)
    at zero, so P7b would best-respond to a belief with no mass on half the
    opponent's possible hands.

    So this expands each rank's mass across its two card indices via
    `leduc_rank.expand_rank_to_indices`, but availability-aware, not an
    even split: the opponent cannot hold `seat`'s own dealt card, nor (once
    round 2 has begun) the revealed public board card, so those indices are
    excluded before splitting a rank's mass across whichever copies remain
    live. An even split ignoring availability is not just a numerical
    approximation - it can assign mass to a card index that is provably
    already held elsewhere, which `_br_to_belief`'s counterfactual then
    tries to deal a second time, raising an uncaught OpenSpiel `SpielError`
    ("deck_[move] != kInvalidCard") rather than merely mis-scoring
    (measured: 40,317 of 53,145 landed-arm rows would have hit this).
    `state` and `seat` are therefore required for the `family == "leduc"`
    path (`_leduc_excluded_indices` reads the exclusion set off `state`).
    """
    if not isinstance(stated, dict):
        return None
    try:
        num_cards = len(game.new_initial_state().chance_outcomes())
        arr = np.zeros(num_cards, dtype=float)
        if family == "leduc":
            if state is None or seat is None:
                return None  # cannot determine availability without the state
            excluded = _leduc_excluded_indices(state, seat)
            # n_ranks derived from num_cards
            # (already computed above, two indices per rank) rather than a
            # second game-parameter lookup - leduc_deep's real rank count
            # for this row, and 3 for canonical
            # "leduc" (num_cards=6 there, 6 // CARDS_PER_RANK == 3).
            source = expand_rank_to_indices(
                stated, excluded, n_ranks=num_cards // CARDS_PER_RANK
            )
            if sum(source.values()) <= 0.0:
                # All stated mass was on a rank now provably impossible
                # (both copies excluded) - no belief survives to
                # best-respond to; no silent fallback to a
                # fabricated (e.g. uniform) prior.
                return None
        else:
            source = stated
        for k, v in source.items():
            arr[int(k)] = float(v)
        return arr
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return None


def _br_set_to_belief(state, seat: int, belief: np.ndarray, cfr_pol) -> tuple:
    """
    every action for `seat` that attains the best expected value under a stated belief
    A set, not one action: several actions can attain the same value, and any of them
    best-responds. Values come from `lmopt.games.belief_br.pooled_action_values`, the same
    operator BayesianAgent uses, with the tie tolerance the Dark Hex path and property 8 use.
    :param state: a representative state for the row's information set
    :param seat: the seat whose best response is wanted (int)
    :param belief: the stated belief over the opponent's card (np.ndarray)
    :param cfr_pol: the solved reference policy the continuation is valued under
    :returns: (the set of best actions, BR_OK), or (None, BR_NO_LEGAL) when the state has
        no legal action, or (None, BR_NO_MASS) when the stated belief puts positive mass on
        no card at all; a belief with no mass names no best response, so the row is
        unscorable rather than credited to the lowest-index action
    """
    legal = state.legal_actions()
    if not legal:
        return None, BR_NO_LEGAL
    if len(legal) == 1:
        return {legal[0]}, BR_OK

    if not any(belief[c] > 0 for c in range(len(belief))):
        return None, BR_NO_MASS

    values = pooled_action_values(state, seat, belief, cfr_pol)
    return best_actions(values, _BEST_ACTION_TOLERANCE), BR_OK


def _is_all_tie(br, legal) -> bool:
    """
    whether a best-response set leaves the choice open
    :param br: the best-action set, or None
    :param legal: the legal actions the set was chosen from
    :returns: True when every legal action attains the best value and there was more than
        one to choose between; a state with a single legal action is a forced move, not a
        tie, and is not counted as one
    """
    return bool(br) and len(legal) > 1 and len(br) == len(legal)


class ExactScorerNotM0(ValueError):
    """
    raised when the exact-policy P7b is asked to score rows from a memory level above M0
    """


#: The one memory level at which a policy at an info-set is a single distribution.
EXACT_MEMORY_LEVEL = "M0"


def _refuse_above_m0(rows: pl.DataFrame) -> None:
    """
    :param rows: the LM rows about to be scored (pl.DataFrame)
    :raises ExactScorerNotM0: when the memory_rung column is absent, or any LM row is not M0
    """
    if "memory_rung" not in rows.columns:
        raise ExactScorerNotM0(
            "score_property7_exact: the frame has no memory_rung column, so its memory level "
            "cannot be confirmed as M0"
        )
    levels = set(rows["memory_rung"].unique().to_list())
    if levels - {EXACT_MEMORY_LEVEL}:
        raise ExactScorerNotM0(
            f"score_property7_exact: memory_rung {sorted(map(str, levels))}; above M0 the policy "
            f"at an info-set is not one distribution, so there is no exact value to take"
        )


def score_property7_exact(
    df: pl.DataFrame,
    exact_policy: dict[tuple[str, str], dict[int, float]],
    cfr_policies: dict,
    games: dict,
    actor: Optional[str] = None,
    include_sampled: bool = True,
    br_cache: Optional[dict] = None,
) -> dict:
    """
    P7b taken on the exact per-info-set policy: the mass the policy places on the best response to
    the row's own stated belief, in place of the sampled action's hit or miss
    Each row is handled exactly as `score_property7`'s 7b loop handles it (same stated belief, same
    best-response set, same fallback drop, same no-mass exclusion); the only change is the hit
    `parsed_action in br`, replaced by `sum(exact_policy[(game, info_set)][a] for a in br)`, which is
    that indicator's expectation. A row whose info-set the exact pass does not cover is counted
    apart and never scored as zero mass. The sampled 7b is returned beside it, unchanged.
    Cost, measured: this scorer recomputes the best-response set for every row, and a cell's rows share
    few distinct (info-set, belief) pairs, so one liarsdice cell (10 shards, 31,630 rows) takes about 17
    minutes (1,026 s) and looks like a hang, against 6.7 s when the sets are remembered across the cell's
    calls. Pass `br_cache` (one dict per cell, shared by the pooled call and every shard call, never wider:
    the sets depend on the game and the reference policy) and each distinct set is computed once;
    `analyse_stats._p7b_exact_cell` does this. Without it, expect that cost and do not read it as a fault.
    Goofspiel reads its rows from `_goofspiel_scored_rows`, the generator the sampled goofspiel 7b reads
    too, and Dark Hex runs `_score_7b_darkhex` itself with the exact mass in place of the sampled hit.
    :param df: the M0 arm's match frame, as `score_property7` takes it (pl.DataFrame)
    :param exact_policy: `{(game, info_set): {action: probability}}`, as
        `analyse_stats.load_exact_policy` returns it; None is refused (dict)
    :param cfr_policies: `{game_name: policy}` for the best response (dict)
    :param games: `{game_name: pyspiel.Game}` (dict)
    :param actor: which agent's rows to score, as `score_property7` takes it (str | None)
    :param include_sampled: whether to also run the sampled `score_property7`; a caller that already
        holds the sampled result, as the per-shard replicate loop does, passes False (bool)
    :param br_cache: an optional dict remembering each best-response set by (game, seat, info-set,
        stated belief); a pure function of those for one policy, so the pooled call and the per-shard
        calls of one cell can share it and compute each distinct set once (dict | None)
    :returns: dict with keys
        "sampled": `score_property7`'s whole output, unchanged, or None when `include_sampled` is False.
        "action_belief_agreement_exact": {game: mean exact mass on the best response, NaN when no row scored}.
        "action_belief_per_infoset_exact": {game: {info_set: mean exact mass}}.
        "n_scored_exact": {game: rows scored}.
        "n_no_exact_entry": {game: rows whose info-set has no exact distribution}.
        "not_applicable_exact": {game: why no exact value exists}.
        "reference_missing_exact": {game: why no exact value exists}, for a game whose game object or
            reference policy was not supplied; its agreement reads NaN and it is absent from the
            per-info-set and count dicts. A game with a reference is ABSENT.
        "discarded_exact": {game: {reason: rows}}, every row of a scored game that reached no best
            response, by reason. A game with none is ABSENT.
    :raises ValueError: when `exact_policy` is None
    :raises ExactScorerNotM0: when any LM row is not at memory level M0
    """
    if exact_policy is None:
        raise ValueError(
            "score_property7_exact: an arm with no exact-policy pass has no distribution basis "
            "and keeps its sampled headline, which the caller decides"
        )
    require_actor_column(df, actor, "score_property7_exact")
    sampled = (
        score_property7(df, cfr_policies=cfr_policies, games=games, actor=actor)
        if include_sampled
        else None
    )
    lm_rows = actor_rows(
        df,
        actor,
        legacy=pl.col("elicited_belief").is_not_null() & pl.col("prompt").is_not_null(),
    )
    if actor is not None:
        lm_rows = lm_rows.filter(pl.col("elicited_belief").is_not_null())
    _refuse_above_m0(lm_rows)

    from lmopt.games.families import GAME_FAMILIES

    def _game_cache(game_name: str):
        """
        :returns: the shared best-response dict for this game, or None when no cache was passed
        """
        if br_cache is None:
            return None
        return br_cache.setdefault(game_name, {})

    agreement: dict[str, float] = {}
    per_infoset: dict[str, dict[str, float]] = {}
    n_scored: dict[str, int] = {}
    n_missing: dict[str, int] = {}
    not_applicable: dict[str, str] = {}
    reference_missing: dict[str, str] = {}
    discarded: dict[str, dict[str, int]] = {}
    for game_name in lm_rows["game"].unique().to_list():
        family = GAME_FAMILIES.get(game_name, game_name)
        game_obj, cfr_pol = games.get(game_name), cfr_policies.get(game_name)
        if game_obj is None or cfr_pol is None:
            agreement[game_name] = float("nan")
            reference_missing[game_name] = _reference_missing_reason(game_obj, cfr_pol)
            continue
        if _is_partial_reference(cfr_pol):
            not_applicable[game_name] = (
                f"partial reference ({cfr_pol.n_infosets} info-sets): a full-tree best response "
                f"is not defined against a policy with off-support gaps"
            )
            continue
        kept = drop_fallback_rows(lm_rows.filter(pl.col("game") == game_name)).kept
        total, mass_sum, missing = 0, 0.0, 0
        by_infoset: dict[str, list[float]] = {}
        exits: dict[str, int] = {}

        def _credit(info_str: str, br) -> None:
            """
            add one row's exact mass on its best-response set, or count the row apart when the
            exact pass holds no distribution for its info-set
            """
            nonlocal total, mass_sum, missing
            distribution = exact_policy.get((game_name, info_str))
            if distribution is None:
                missing += 1
                return
            mass = sum(distribution.get(a, 0.0) for a in br)
            total += 1
            mass_sum += mass
            by_infoset.setdefault(info_str, []).append(mass)

        if family == "darkhex":

            def _darkhex_mass(info_str: str, br, logged: int, game_name=game_name):
                nonlocal total, missing
                distribution = exact_policy.get((game_name, info_str))
                if distribution is None:
                    missing += 1
                    return None
                total += 1
                return sum(distribution.get(a, 0.0) for a in br)

            agr, per_is, _coverage, dh_discarded, dh_ties = _score_7b_darkhex(
                kept, game_obj, cfr_pol, row_score=_darkhex_mass
            )
            for key in ("n_info_set_absent_from_worlds", "n_no_scorable_belief"):
                if dh_discarded.get(key):
                    exits[key] = dh_discarded[key]
            exits.update(dh_ties["unscored_exits"])
            if exits:
                discarded[game_name] = exits
            n_scored[game_name], n_missing[game_name] = total, missing
            agreement[game_name] = agr
            per_infoset[game_name] = per_is
            continue
        if family == "goofspiel":
            for item in _goofspiel_scored_rows(
                kept, game_obj, cfr_pol, br_cache=_game_cache(game_name)
            ):
                if item is _DEGENERATE_BELIEF:
                    _count(exits, "the stated belief puts no mass on a legal opponent bid")
                elif isinstance(item, _GoofspielExit):
                    _count(exits, str(item))
                else:
                    _credit(item[0], item[2])
            if exits:
                discarded[game_name] = exits
            n_scored[game_name], n_missing[game_name] = total, missing
            agreement[game_name] = mass_sum / total if total else float("nan")
            per_infoset[game_name] = {k: sum(v) / len(v) for k, v in by_infoset.items()}
            continue
        for item in _card_game_scored_rows(
            kept,
            game_obj,
            family,
            cfr_pol,
            br_cache=_game_cache(game_name),
            needs_action=False,
        ):
            if isinstance(item, _CardGameExit):
                _count(exits, str(item))
            else:
                _credit(item[0], item[2])
        if exits:
            discarded[game_name] = exits
        n_scored[game_name], n_missing[game_name] = total, missing
        agreement[game_name] = mass_sum / total if total else float("nan")
        per_infoset[game_name] = {k: sum(v) / len(v) for k, v in by_infoset.items()}
    return {
        "sampled": sampled,
        "action_belief_agreement_exact": agreement,
        "action_belief_per_infoset_exact": per_infoset,
        "n_scored_exact": n_scored,
        "n_no_exact_entry": n_missing,
        "not_applicable_exact": not_applicable,
        "reference_missing_exact": reference_missing,
        "discarded_exact": discarded,
    }
