"""State-sensitivity decomposition.

`mask_constant_share` (scripts/analyse_stats.py) answers "does the action ever
vary inside a legal_mask bin". It is correct for that question, but it is
denominated on rows with no companion measure of how many distinct states the
bin contained, so a bin that supplied no state variation reads identically to
one the model was blind to.

Over one collection cycle, at the pooled (arm, game, mask) unit: 8 bins totalling 122,213 rows hold exactly one
distinct prompt. Those are not vacuous -- OLMo darkhex's
`1,1,1,1` bin is one prompt, 20,000 rows, and two distinct actions split
14,428/5,572 in all five seeds -- but a constant action in such a bin is a
determinism observation, not a state-blindness one, and nothing in
properties.csv distinguishes the two readings.

This module supplies the missing decomposition. `multistate_share` restricts to
bins the model could actually have responded to and reports the same *kind* of
quantity as `mask_constant_share`: the share of rows whose action is constant.
At the PER-cell (game, opponent, mask) unit, which is
the unit `property_table` calls this function on, summed across the 60
(game, opponent) cells: `mask_constant_share` 512,945/661,644 = 0.7753 against
`multistate_share` 269,902/398,406 = 0.6775, with 90 one-prompt bins /
263,238 rows set aside as single-state.

The two units above are not interchangeable and the figures differ by 2.15x on
the set-aside rows, so each is labelled. "Pooled" here means opponents merged
before binning; it never means summed across cells.

MULTI-state is keyed on distinct prompts, not distinct INFO-sets, and the
distinction is load-bearing. Leduc deliberately renders 468 info-sets to 144
prompts (suit marginalisation): `[Private: 0]` and
`[Private: 1]` are both a Jack and are strategically identical. Info-sets that
render to the same prompt are indistinguishable to any model, so counting them
as state variation would credit the model with information it was never shown.
A leduc bin holds 36 info-sets across 12 prompts.

The same argument applied
once more, to the seat. On a random-seat arm the prompt carries which seat the
model occupies, so two prompts differing only in that respect make a bin
"multi-state" without the model ever having been shown a state change at a
fixed seat. Worse in the other direction: the model can play the two seats
differently, which makes the bin non-constant without any state-sensitivity
either. Both readings are seat observations, not state-blindness ones.

`multistate_share_within_seat` therefore changes the unit rather than
filtering the bins. It is computed over (`legal_mask` bin, seat) pairs: a pair
enters the denominator iff that seat held >= 2 distinct prompts, and its rows
count as constant iff that seat played exactly one action. A bin where only
seat 0 is multi-state contributes seat 0's rows and nothing else; seat 1's
rows are not in the denominator at all. Selecting bins on the pair criterion
but scoring constancy on the pooled bin would still let a seat-driven action
switch read as state-sensitivity.

The within-seat share is the measure that carries state-blindness claims;
`multistate_share` is retained as the census, and each share is published
beside its own denominators.
Frames with no `lm_seat` column (every schema-v2 arm) refuse rather than
assume one seat.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from lmopt.scoring._actor import DropReport, drop_fallback_rows


@dataclass(frozen=True)
class BinStat:
    """One `legal_mask` bin of one (game, opponent) cell's LM rows.

    The last two fields are the
    within-seat companions of `n_prompt`/`multistate`. Under random seats the
    prompt carries the seat, so two prompts that differ only in which seat the
    model occupies make a bin pooled-multi-state without the model ever having
    been shown a state change at a fixed seat. `n_prompt_within_seat_max` is
    the maximum over seats of the distinct prompts that seat held; a bin is
    within-seat multi-state iff some seat held two or more.

    Both are `None` -- never `0`/`False` -- when the frame carries no
    `lm_seat` column at all, which is every schema-v2 arm: the quantity was
    not measured there, and `False` would assert the seat was held fixed
    (no silent fallbacks). A row whose `lm_seat` is NULL on an
    otherwise-seated frame counts toward no seat (it is evidence about no
    seat), so a bin of only null-seat rows has a measured max of `0`.
    """

    mask: str
    rows: int
    n_prompt: int
    n_infoset: int
    n_action: int
    constant: bool
    multistate: bool
    n_prompt_within_seat_max: int | None = None
    multistate_within_seat: bool | None = None


def fallback_drop_report(df: pl.DataFrame) -> DropReport:
    """
    the scored LM's rows that every quantity in this module leaves out because the harness
    substituted their action, with the counts behind the dropped rate
    :param df: one cell's rows, LM and opponent (pl.DataFrame)
    :returns: DropReport with unit "rows" over the LM rows; `kept` is the LM rows that stay (DropReport)
    """
    return drop_fallback_rows(df.filter(pl.col("prompt").is_not_null()))


def add_mask_key(df: pl.DataFrame) -> pl.DataFrame:
    """
    `df` with a `_mask` column: the list-typed `legal_mask` rendered to a scalar string key.
    `pl.List` cannot be a `group_by` key directly, and the cast keeps the key canonical if a
    Parquet ever arrives with a narrower integer width. Every place that bins rows by their
    legal-action mask builds the key here, so no two can disagree about what a bin is
    :param df: rows carrying a `legal_mask` list column (pl.DataFrame)
    :returns: `df` plus a `_mask` column (pl.DataFrame)
    """
    return df.with_columns(
        pl.col("legal_mask")
        .cast(pl.List(pl.Int64))
        .list.eval(pl.element().cast(pl.String))
        .list.join(",")
        .alias("_mask")
    )


def _keyed(df: pl.DataFrame) -> pl.DataFrame:
    """
    the LM rows every statistic here is computed from, with the list-typed `legal_mask`
    rendered to a scalar key. A row whose action the harness substituted (`fallback_used`)
    is left out: it says nothing about what the model chose, and counting its action would
    turn a constant bin into a non-constant one
    :param df: one cell's rows, LM and opponent (pl.DataFrame)
    :returns: the kept LM rows with a `_mask` column (pl.DataFrame)
    """
    return add_mask_key(fallback_drop_report(df).kept)


@dataclass(frozen=True)
class SeatBinStat:
    """One (`legal_mask` bin, seat) pair of one cell's seated LM rows.

    The unit the
    within-seat statistic is computed at, and the reason this type exists
    rather than two more `BinStat` fields. Selecting bins by within-seat
    multi-stateness but then reading the pooled `rows`/`constant` off the bin
    counts a seat-driven action switch as state-sensitivity, which is the
    confound the measure exists to remove. On shakedown-v3 olmo
    goofspiel/rl, bin `1,0,0,1`: pooled `n_action` is 2
    over 200 rows, but seat 1 alone plays action 3 in all 94 of its rows
    across 4 distinct prompts. Pooled constancy scores that seat as
    state-sensitive; per-seat constancy scores it blind, which is what the
    model actually did.

    `constant` and `multistate` are therefore both judged within this seat:
    `n_action == 1` and `n_prompt >= 2` over this pair's rows alone. Rows
    whose `lm_seat` is null are in no pair at all.
    """

    mask: str
    seat: int
    rows: int
    n_prompt: int
    n_infoset: int
    n_action: int
    constant: bool
    multistate: bool


def _seat_bins(keyed: pl.DataFrame) -> list[SeatBinStat]:
    """One `SeatBinStat` per (mask, seat) of an already-`_keyed` frame.

    Empty when the frame has no `lm_seat` column and when it has one that no
    LM row fills. Rows whose `lm_seat` is null are filtered, not grouped:
    polars groups nulls as a group of their own by default, which would let
    rows that never said which seat they were played from form a "seat" and
    supply within-seat state variation.
    """
    if "lm_seat" not in keyed.columns:
        return []
    seated = keyed.filter(pl.col("lm_seat").is_not_null())
    if seated.is_empty():
        return []
    agg = (
        seated.group_by(["_mask", "lm_seat"])
        .agg(
            pl.len().alias("rows"),
            pl.col("prompt").n_unique().alias("n_prompt"),
            pl.col("info_set").n_unique().alias("n_infoset"),
            pl.col("parsed_action").n_unique().alias("n_action"),
        )
        .sort(["_mask", "lm_seat"])
    )
    return [
        SeatBinStat(
            mask=r["_mask"],
            seat=int(r["lm_seat"]),
            rows=r["rows"],
            n_prompt=r["n_prompt"],
            n_infoset=r["n_infoset"],
            n_action=r["n_action"],
            constant=r["n_action"] == 1,
            multistate=r["n_prompt"] >= 2,
        )
        for r in agg.iter_rows(named=True)
    ]


def seat_bin_stats(df: pl.DataFrame) -> list[SeatBinStat]:
    """One `SeatBinStat` per (`legal_mask` bin, seat) of `df`'s LM rows.

    Empty when `df` carries no `lm_seat` column (every schema-v2 arm) and
    when no LM row fills it; those two are different facts, and the caller
    that needs to tell them apart tests `"lm_seat" in df.columns` itself, as
    `state_sensitivity` does.
    """
    return _seat_bins(_keyed(df))


def bin_stats(df: pl.DataFrame) -> list[BinStat]:
    """One `BinStat` per `legal_mask` bin of `df`'s LM rows, mask-sorted.

    The within-seat fields are populated only when `df` carries an `lm_seat`
    column (schema-v3+); otherwise they stay `None`, see `BinStat`. They are
    bin-level summaries for readers of a single bin; the within-seat share is
    not computed from them, see `SeatBinStat`.
    """
    keyed = _keyed(df)
    if keyed.is_empty():
        return []
    agg = (
        keyed.group_by("_mask")
        .agg(
            pl.len().alias("rows"),
            pl.col("prompt").n_unique().alias("n_prompt"),
            pl.col("info_set").n_unique().alias("n_infoset"),
            pl.col("parsed_action").n_unique().alias("n_action"),
        )
        .sort("_mask")
    )
    seated = "lm_seat" in keyed.columns
    within: dict[str, int] = {}
    for sb in _seat_bins(keyed):
        within[sb.mask] = max(within.get(sb.mask, 0), sb.n_prompt)
    out = []
    for r in agg.iter_rows(named=True):
        wmax = within.get(r["_mask"], 0) if seated else None
        out.append(
            BinStat(
                mask=r["_mask"],
                rows=r["rows"],
                n_prompt=r["n_prompt"],
                n_infoset=r["n_infoset"],
                n_action=r["n_action"],
                constant=r["n_action"] == 1,
                multistate=r["n_prompt"] >= 2,
                n_prompt_within_seat_max=wmax,
                multistate_within_seat=None if wmax is None else wmax >= 2,
            )
        )
    return out


def multistate_share(df: pl.DataFrame) -> tuple[float | None, int]:
    """(share, denominator) over bins that held >= 2 distinct prompts.

    `share` is the fraction of those bins' rows whose action is a constant
    function of `legal_mask` -- the same quantity `mask_constant_share`
    reports, restricted to bins where the model actually saw the state vary.

    Returns `(None, 0)`, never `(0.0, 0)`, when the cell has no multi-state
    bin: 0.0 is this statistic's cleanest possible reading, so returning it
    for a cell with nothing to measure would assert perfect state-sensitivity
    from no evidence (no silent fallbacks; "measured zero vs unmeasurable"
    is a recurring defect class here). 12 of 60 collected cells take this
    branch.
    """
    multi = [b for b in bin_stats(df) if b.multistate]
    denom = sum(b.rows for b in multi)
    if denom == 0:
        return None, 0
    return sum(b.rows for b in multi if b.constant) / denom, denom


_NO_SEAT_COLUMN = (
    "this arm's log carries no `lm_seat` column (schema-v2), so the seat "
    "could not be held fixed and the within-seat state-sensitivity share "
    "could not be measured"
)
_ALL_SEATS_NULL = (
    "every LM row in this cell has a null `lm_seat`, so no row is evidence "
    "about any seat and the within-seat state-sensitivity share could not be "
    "measured"
)
_NO_LM_ROWS = (
    "this cell has no LM rows, so the within-seat state-sensitivity share "
    "could not be measured"
)
ALL_LM_ROWS_FALLBACK = (
    "every LM row in this cell is a harness fallback row, so no action the "
    "model chose is left to measure"
)
_NO_WITHIN_SEAT_BIN = (
    "no (legal-mask bin, seat) pair in this cell held two or more distinct "
    "prompts within that seat, so the action's dependence on the state could "
    "not be separated from its dependence on the legal mask and the seat"
)


@dataclass(frozen=True)
class StateSensitivity:
    """Every state-sensitivity quantity of one cell, from one pass.

    One type so the
    property table can publish each share beside its own denominators without
    re-binning the cell per row, and so a share can never be reported against
    a denominator computed from a different pass.

    The two units are not interchangeable and are named in the field names:
    the pooled counts are over `legal_mask` bins, the within-seat counts are
    over (bin, seat) pairs. `bins_multistate_within_seat` and
    `rows_multistate_within_seat` are `None` only when the frame carries no
    `lm_seat` column, since there is then nothing to count over; with a real
    seat column an empty count is a measured `0`, exactly as
    `rows_multistate_pooled` is. Each share is `None`, never `0.0`, when its
    denominator is empty; `within_seat_reason` is `None` iff
    `share_within_seat` is not, and names which of five causes applied.
    """

    bins_total: int
    bins_multistate_pooled: int
    rows_multistate_pooled: int
    share_pooled: float | None
    bins_multistate_within_seat: int | None
    rows_multistate_within_seat: int | None
    share_within_seat: float | None
    within_seat_reason: str | None


def state_sensitivity(df: pl.DataFrame) -> StateSensitivity:
    """Both shares and all five denominators for one cell, in one pass.

    The within-seat statistic is computed at the (bin, seat) unit throughout:
    a pair counts iff that seat held >= 2 distinct prompts, and its rows
    count as constant iff that seat played one action. Reading the pooled
    `constant` off a within-seat-selected bin would let a seat-driven action
    switch read as state-sensitivity, which is the whole confound the measure
    removes. On shakedown-v3, the per-seat reading differs from the pooled
    one in 12 of the 44 cells with a measurable within-seat share: 9 up,
    toward more blindness, and 3 down (all mistral liarsdice). The direction
    is not monotone (the pooled reading does not always understate
    blindness), because the per-seat unit changes the numerator and the
    denominator at once: a seat that is dropped from the denominator takes
    its rows with it.
    """
    keyed = _keyed(df)
    bins = bin_stats(df)
    pooled_multi = [b for b in bins if b.multistate]
    pooled_rows = sum(b.rows for b in pooled_multi)
    pooled_share = (
        sum(b.rows for b in pooled_multi if b.constant) / pooled_rows
        if pooled_rows
        else None
    )

    if "lm_seat" not in df.columns:
        return StateSensitivity(
            bins_total=len(bins),
            bins_multistate_pooled=len(pooled_multi),
            rows_multistate_pooled=pooled_rows,
            share_pooled=pooled_share,
            bins_multistate_within_seat=None,
            rows_multistate_within_seat=None,
            share_within_seat=None,
            within_seat_reason=_NO_SEAT_COLUMN,
        )

    seat_bins = _seat_bins(keyed)
    within_multi = [sb for sb in seat_bins if sb.multistate]
    within_rows = sum(sb.rows for sb in within_multi)
    if within_rows:
        share = sum(sb.rows for sb in within_multi if sb.constant) / within_rows
        reason = None
    else:
        share = None
        if not bins:
            reason = (
                ALL_LM_ROWS_FALLBACK
                if fallback_drop_report(df).n_lm_rows
                else _NO_LM_ROWS
            )
        elif not seat_bins:
            reason = _ALL_SEATS_NULL
        else:
            reason = _NO_WITHIN_SEAT_BIN
    return StateSensitivity(
        bins_total=len(bins),
        bins_multistate_pooled=len(pooled_multi),
        rows_multistate_pooled=pooled_rows,
        share_pooled=pooled_share,
        bins_multistate_within_seat=len(within_multi),
        rows_multistate_within_seat=within_rows,
        share_within_seat=share,
        within_seat_reason=reason,
    )


def multistate_share_within_seat(df: pl.DataFrame) -> tuple[float | None, int]:
    """`multistate_share` with the seat confound removed.

    Same `(None, 0)`-never-`(0.0, 0)` contract, computed at the (bin, seat)
    unit: the denominator is the rows of every (bin, seat) pair whose seat
    held >= 2 distinct prompts, and the numerator is those of them whose seat
    played one action. Under a random seat assignment the prompt carries the
    seat, so a bin can be pooled-multi-state purely because the model played
    it from both sides, and it can be pooled-NON-constant purely because it
    played the two sides differently; both readings are seat observations,
    not state-blindness ones, in exactly the way a constant action in a
    one-prompt bin is a determinism observation.

    This is the measure that carries state-blindness claims;
    `multistate_share` is retained as the census. `(None, 0)` covers five distinct causes; call `within_seat_reason`
    for the one that applies, or `state_sensitivity` for both plus every
    denominator in one pass, which is what the property table does.
    """
    st = state_sensitivity(df)
    return st.share_within_seat, st.rows_multistate_within_seat or 0


def within_seat_reason(df: pl.DataFrame) -> str | None:
    """Why `multistate_share_within_seat(df)` returned `None`, or `None`.

    `within_seat_reason(df) is None` exactly when
    `multistate_share_within_seat(df)[0] is not None`, so a caller can emit
    the pair without ever pairing a real number with an excuse, or a `None`
    with an empty caveat.
    """
    return state_sensitivity(df).within_seat_reason


def qualifying_masks_action_constant(df: pl.DataFrame) -> dict[str, bool]:
    """`legal_mask` -> action-constant, for masks with >= 1 within-seat
    qualifying (multistate) seat. Masks with no qualifying seat are absent.

    The shared definition behind the cross-arm S7 reading
    and `analyse_stats.py`'s per-arm `state_sensitivity_cell_b_clean`
    admissibility gate, so both apply the same within-seat
    qualifying/constancy criterion rather than two hand-rolled copies. A mask with multiple qualifying seats is scored
    conservatively: `True` (action-constant) iff any qualifying seat is
    itself constant, matching both callers' whole-cell-inadmissible-on-any
    -dirty-bin design -- a mask is never partially admitted.

    Empty when `qualifying_masks_action_constant`'s own precondition
    (`seat_bin_stats(df)` non-empty and at least one bin multistate) does
    not hold; callers distinguish "nothing qualifies" from "unmeasurable"
    via `within_seat_reason(df)`, not by inspecting this dict's emptiness
    alone (an unseated frame and a seated-but-all-single-prompt frame both
    return `{}` here, for different reasons).
    """
    out: dict[str, bool] = {}
    for sb in seat_bin_stats(df):
        if not sb.multistate:
            continue
        out[sb.mask] = out.get(sb.mask, False) or sb.constant
    return out
