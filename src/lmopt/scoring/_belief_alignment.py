"""Shared belief/posterior alignment helpers: the goofspiel support guard for
P5 and P7a, and the one canonical total-variation distance `tvd`.

Both `property5.py` and `property7.py` zero-pad `true_posterior` to the
elicited belief's key set for goofspiel (`stated` is dense over the full
deck by construction: `prompting/goofspiel.py`'s `parse_belief` returns a
dict comprehension over `range(1, num_cards + 1)`, so every card gets a
key), which assumes `true`'s keys are always a subset of `stated`'s. This
module is the single place that assumption is checked, shared by both
scorers rather than duplicated in each.

There is deliberately no kuhn/liarsdice call site. It would be structurally
unfireable (0/378,498 kuhn rows, 0/458,240 liarsdice rows - both parsers
unconditionally pad to the same fixed range `true_posterior` is built over,
so `true` having a key `stated` lacks is impossible, not just unobserved),
and an ungated scorability failure on a non-goofspiel family is
deliberately published as census (`scripts/analyse_stats.py`), not dropped
to `not_applicable`. See `property5.py`'s goofspiel branch comment.
"""

from __future__ import annotations

import json

import numpy as np

from lmopt.scoring.leduc_rank import marginalise_to_rank


class GoofspielSupportViolation(ValueError):
    """One game's true_posterior support assumption failed on 1+ rows.

    A plain `ValueError` raised on the first offending row from inside a
    per-game loop, with no try/except anywhere above it, would abort every
    other game already computed in the same
    `score_property7`/`score_property5` call - a single corrupt goofspiel
    row could silently cost kuhn/leduc/liarsdice results too.
    `game_row_tvds` scans a game's rows to completion, counting every
    violation and unioning their offending keys, and raises this once per
    game so a caller can confine the failure to that one game
    (`not_applicable`-style) while other games keep scoring - loud and
    specific, not silent, but not scoped wider than the game that is
    actually corrupt.
    """

    def __init__(
        self,
        *,
        n_violations: int,
        offending_keys: frozenset,
        game_name: str,
        scorer_name: str,
    ):
        self.n_violations = n_violations
        self.offending_keys = offending_keys
        self.game_name = game_name
        self.scorer_name = scorer_name
        super().__init__(
            f"{scorer_name}: game {game_name!r} has {n_violations} goofspiel "
            f"row(s) whose true_posterior carries key(s) {sorted(offending_keys)} "
            "absent from the elicited belief - the zero-pad-to-stated convention "
            "assumes stated is dense over the full deck (prompting/goofspiel.py's "
            "parse_belief) and true's keys are always a subset of stated's; these "
            "rows violate that assumption, and computing around it would silently "
            "discard true_posterior mass rather than compare it. Investigate "
            "before scoring: either goofspiel's elicitation stopped being dense, "
            "or these rows are corrupt."
        )


def tvd(p: dict, q: dict) -> float:
    """
    Total variation distance between two distributions given as dicts.

    Canonical form: `0.5 * float(np.sum(np.abs(s_arr - t_arr)))` over
    `keys = sorted(set(p) | set(q))`, with missing keys reading 0.0. Keys may
    be str or int, never mixed within one call (no caller mixes them).

    This is the one shared definition behind the four total-variation
    implementations that grew independently -
    a tree viewer's `tvd`, a figure module's
    `_tvd_by_infoset` inline array steps, `scoring/property7.py`'s
    `score_property7` inline array steps (over `_combine_keys`, itself
    `sorted(set | set)`), and `scoring/property_paired.py`'s
    `_total_variation`. Same reasoning as this module's
    `assert_true_support_within_stated` below: one definition, not copies
    that can drift (one copy fed a published figure; see
    `_tvd_by_infoset`'s docstring).

    Why the numpy form and not a plain-python sum: the two legacy families
    disagree with each other at up to 2 ulp (numpy's
    blocked summation vs python's sequential sum), so no single function can
    bit-match all four legacy sites. The numpy form is bit-identical to the
    two scoring/figure sites (`property7.py` and the figure module)
    and within 2 ulp of the two python sites; the reverse choice flips 2 of
    the 8 leduc golden rows.

    The key set is the union of the two supports, not the intersection: an
    outcome one distribution puts mass on and the other never does is exactly
    the divergence P1/P4 (paired probes) and P7a (belief error) are looking
    for, and intersecting would score it as no divergence at all
    (`property_paired.py`'s `_total_variation` docstring, preserved here).
    :param p: first distribution, {key: probability}; missing keys read 0.0.
    :param q: second distribution, same key type as `p`.
    :returns: Half the L1 distance over the key union, as a python float.
    """
    keys = sorted(set(p) | set(q))
    s_arr = np.array([float(p.get(k, 0.0)) for k in keys])
    t_arr = np.array([float(q.get(k, 0.0)) for k in keys])
    return 0.5 * float(np.sum(np.abs(s_arr - t_arr)))


def assert_true_support_within_stated(
    stated: dict, true: dict, *, scorer_name: str
) -> None:
    """
    Raise ValueError if `true` has a key absent from `stated`.

    No silent fallbacks: the zero-pad-to-`stated` convention
    both P5 and P7a use for goofspiel assumes `stated` is dense over the
    full deck (`prompting/goofspiel.py`'s `parse_belief`, whose returned
    comprehension ranges over every card), so `true`'s keys are
    always a subset of `stated`'s. Computing around a violation of that
    assumption (e.g. by only ever looking up `true.get(k, 0.0)` for `k` in
    `stated`) would silently discard `true_posterior` mass rather than
    compare it - raise loudly instead, before either scorer's alignment
    step runs.
    :param stated: the elicited belief dict for one row.
    :param true: the true_posterior dict for the same row (after any
        family-specific transform has already been applied, e.g.
        leduc's rank-marginalisation; this function is goofspiel-only
        in practice, since it is called from each scorer's
        `elif family == "goofspiel":` branch).
    :param scorer_name: which scorer is asking, e.g. "score_property5",
        folded into the raised message so a traceback names its caller.
    :raises ValueError: naming the offending keys, if `true` has any key
        `stated` lacks. No-op (returns None) otherwise.
    """
    extra = set(true.keys()) - set(stated.keys())
    if extra:
        raise ValueError(
            f"{scorer_name}: goofspiel true_posterior has key(s) "
            f"{sorted(extra)} absent from the elicited belief - the "
            "zero-pad-to-stated convention assumes stated is dense over "
            "the full deck (prompting/goofspiel.py's parse_belief) and true's "
            "keys are always a subset of stated's; this row violates that "
            "assumption, and computing around it would silently discard "
            "true_posterior mass rather than compare it. Investigate "
            "before scoring: either goofspiel's elicitation stopped being "
            "dense, or this row is corrupt."
        )


def game_row_tvds(
    game_rows,
    family: str,
    *,
    combine_keys,
    scorer_name: str,
    game_name: str,
    leduc_n_ranks: int = 3,
) -> tuple[list[dict], int]:
    """
    Compute per-row TVDs with family-specific alignment for one game.

    `leduc_n_ranks`: defaults to 3 (canonical Leduc), so a caller
    that does not pass it uses the canonical rank count.
    `property7.py`'s own call site passes the real per-row rank count
    (`leduc_rank.leduc_num_ranks`/`num_cards // CARDS_PER_RANK`, matching
    `_parse_belief_array`'s P7b side), since that is the one on the live
    P5/P7 admissibility-gate path leduc_deep's belief scoring is scoped to.
    The other two callers are analysis/reporting tools outside that path;
    left un-threaded here as a named, deliberate scope boundary rather than
    a silent gap, which matters only if leduc_deep rows ever reach them.

    Canonical per-row TVD computation, shared with score_property7's inner
    loop. Handles the family-specific
    alignment: Leduc rank-marginalisation, Goofspiel support assertion plus
    zero-pad, darkhex family N/A (raises).
    :param game_rows: pl.DataFrame rows of one game, both belief columns non-null
        (caller pre-filters). Must have columns: elicited_belief,
        true_posterior. Optional: info_set (defaults to empty string if
        absent).
    :param family: GAME_FAMILIES value (e.g. "leduc", "goofspiel", "darkhex");
        caller resolves it.
    :param combine_keys: Callable[[dict, dict], list]; property7 passes its
        module-level _combine_keys so the monkeypatch seam survives.
        Returns the key list for alignment (union or intersection).
    :param scorer_name: which scorer is calling, for error messages.
    :param game_name: which game `game_rows` holds (caller already filtered by
        it); named only for GoofspielSupportViolation's message.
    :param leduc_n_ranks: the Leduc rank count the rank marginalisation uses, 3 by default (int)
    :returns: Tuple of (records, n_unparseable):
        - records: list of {"info_set": str, "tvd": float}, one per parseable
        row.
        - n_unparseable: count of rows with JSON parse failures (a counted,
        visible category rather than a silent `continue`).
    :raises ValueError: if family == "darkhex" (darkhex belief is per-cell
        occupancy, not a simplex, so TVD-over-a-distribution does not
        apply).
    :raises GoofspielSupportViolation: if family == "goofspiel" and 1+ rows have
        a true_posterior key absent from stated. Raised once after the
        whole game's rows are scanned, never mid-loop,
        so the count and the union of offending keys are exact, not
        "at least the first one found." No records are returned for a
        game that raises; the whole game is discarded, matching the
        "set tvd_mean to nan" degrade the caller applies, not a partial
        score built from only the clean rows.
    """
    if family == "darkhex":
        raise ValueError(
            f"{scorer_name}: family 'darkhex' reached game_row_tvds - darkhex "
            "belief is per-cell occupancy (independent Bernoullis, not a "
            "simplex), so TVD-over-a-distribution does not apply. "
            "Caller must gate darkhex out."
        )

    records = []
    n_unparseable = 0
    n_violations = 0
    offending_keys: set = set()

    for row in game_rows.iter_rows(named=True):
        try:
            stated = json.loads(row["elicited_belief"])
            true = json.loads(row["true_posterior"])
        except (json.JSONDecodeError, TypeError):
            n_unparseable += 1
            continue

        if family == "leduc":
            # Leduc's true_posterior is over 6 card indices; the stated
            # belief is a 3-key rank simplex. Unioning the raw key sets
            # (below) would treat "no stated mass at index 3" as a genuine
            # disagreement instead of a granularity mismatch, inflating
            # TVD 0.5858 (raw keys) vs. 0.3542 (rank-marginalised) - a
            # 65% overstatement on a collected arm.
            # Marginalise the posterior to rank space so both sides
            # share the same 3 keys before comparing. See leduc_rank.py.
            true = marginalise_to_rank(true, n_ranks=leduc_n_ranks)
        elif family == "goofspiel":
            # goofspiel_all_posteriors' support omits deck cards P1
            # cannot hold in any world consistent with P0's info-state -
            # Legitimate deduction, not a clairvoyant reference (proven on
            # a real CFR+-solved game by
            # tests/test_goofspiel_posterior.py::
            # test_missing_posterior_keys_are_deducibly_unplayable_by_p1).
            # The correct alignment is therefore to zero-pad `true` to
            # the LM's stated support, which is always dense over the
            # full deck (prompting/goofspiel.py's `parse_belief`, whose
            # returned comprehension ranges over every card). A raw
            # key-union would achieve this padding only incidentally -
            # right for goofspiel by coincidence, wrong for leduc
            # (rank-vs-index category error, handled above). Padding here
            # first makes `true`'s keys a subset of `stated`'s, so the
            # alignment does not depend on union vs. intersection below; see
            # tests/test_scorer_property7.py::TestGoofspielZeroPadding::
            # test_zero_padding_survives_intersection_instead_of_union.
            #
            # Unlike a raw key-union (which could only ever gain keys,
            # never lose posterior mass), this explicit padding conforms
            # `true` to `stated`'s key set and so can silently drop `true`
            # mass that `stated` doesn't cover - always moving TVD downward
            # (toward "the model was more right than it was"), never up. It
            # is safe only because `stated` is guaranteed dense over the
            # full deck by the sole writer (`prompting/goofspiel.py`'s
            # `parse_belief`, reached from `agents/lm_vllm.py`'s belief path
            # -- the sibling-mode retry loop's `parse_belief` call and the
            # assignment to `last_elicited_belief` after it, plus combined
            # mode's own `last_elicited_belief = ...parse_belief(...)` --
            # and read off the agent by `harness/match_runner.py`'s
            # LM-provenance `getattr(agent, "last_elicited_belief", None)`).
            # No silent fallbacks forbids computing around a
            # violation of that assumption without surfacing it - so raise
            # loudly the moment it's false, rather than let this padding
            # quietly under-report a future divergence (e.g. if goofspiel
            # elicitation is ever narrowed). See tests/test_scorer_property7.py::
            # TestGoofspielPaddingInvariant.
            #
            # The check is `_belief_alignment.assert_true_support_
            # within_stated`, shared with `property5.py`'s equivalent
            # guard.
            #
            # The assertion itself raises on the first violation it sees;
            # This caller does not let that propagate mid-loop. A violating
            # row is counted and its offending keys unioned, then skipped
            # (never added to `records`) so the rest of the game's rows
            # are still scanned - the aggregated count and key set are
            # exact, not "first one found," when raised once below.
            try:
                assert_true_support_within_stated(stated, true, scorer_name=scorer_name)
            except ValueError:
                n_violations += 1
                offending_keys |= set(true.keys()) - set(stated.keys())
                continue
            true = {k: true.get(k, 0.0) for k in stated}

        # Key selection stays with the caller's `combine_keys` (the
        # monkeypatchable seam the two intersection tests in
        # tests/test_scorer_property7.py depend on); the arithmetic is the
        # shared canonical `tvd`. Both dicts are aligned to exactly `keys`
        # first, so `tvd`'s own `sorted(set | set)` re-derives `keys`
        # identically and the alignment is unchanged.
        keys = combine_keys(stated, true)
        row_tvd = tvd(
            {k: stated.get(k, 0.0) for k in keys},
            {k: true.get(k, 0.0) for k in keys},
        )

        records.append({"info_set": row.get("info_set", ""), "tvd": row_tvd})

    if n_violations:
        # Whole game discarded: raising here, after the full scan, rather
        # than returning `records` built only from the clean rows, sets
        # goofspiel's tvd_mean to nan - a game with any corrupt row is
        # not_applicable, not partially scored from whatever survived.
        raise GoofspielSupportViolation(
            n_violations=n_violations,
            offending_keys=frozenset(offending_keys),
            game_name=game_name,
            scorer_name=scorer_name,
        )

    return records, n_unparseable
