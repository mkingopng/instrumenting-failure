"""Which rows belong to the agent under test.

Keyed on `prompt.is_not_null()`, a scorer selects LM rows only by accident of
who gets a prompt; keyed on `elicited_belief.is_not_null()`, it excludes non-LM
rows only because the opponent registry has no belief-eliciting branch.
Neither can score a principled opponent, and neither says which filter it used.

An `actor` column (the model id on LM rows, else the agent type id) makes
the population a stated predicate. `actor=None` keeps the legacy
filter and labels it, which is what arms without an `actor` column need.

Scoring stays offline: these are pure functions over a frame.
"""

from __future__ import annotations

import dataclasses
import json

import polars as pl

from lmopt.logging.schema import BASE_PROBE_TYPE

#: The value of `"actor_filter"` when no actor was named and the legacy
#: prompt-null filter selected the population.
LEGACY_PROMPT = "legacy-prompt-null"
#: ... when the population was the two belief columns (P5).
LEGACY_BELIEF = "legacy-belief-columns"
#: ... when it was the belief columns and the prompt (P7).
LEGACY_BELIEF_AND_PROMPT = "legacy-belief-columns-and-prompt-null"


def require_actor_column(df: pl.DataFrame, actor: str | None, where: str) -> None:
    """Refuse to score a named actor on a frame that cannot identify actors.

    Falling back to the prompt-null population here would silently substitute
    the estimand: the caller asked for one agent's rows and would receive the
    LM's, under the opponent's name. That is the failure mode the
    no-silent-fallbacks rule exists for, and unlike the `actor=None`
    path there is no legacy arm that needs it.
    """
    if actor is not None and "actor" not in df.columns:
        raise ValueError(
            f"{where}: actor={actor!r} was requested but the frame has no "
            "'actor' column (an arm collected before it existed). Re-run the arm, or score it with "
            f"actor=None, which uses the {LEGACY_PROMPT!r} filter."
        )


def actor_label(actor: str | None, legacy: str = LEGACY_PROMPT) -> str:
    """The `"actor_filter"` value to report for this call."""
    return f"actor={actor}" if actor is not None else legacy


def actor_rows(df: pl.DataFrame, actor: str | None, *, legacy: pl.Expr) -> pl.DataFrame:
    """The scored agent's own rows: `actor == actor`, else the legacy filter."""
    if actor is None:
        return df.filter(legacy)
    return df.filter(pl.col("actor") == actor)


def environment_rows(
    df: pl.DataFrame, actor: str | None, *, legacy: pl.Expr
) -> pl.DataFrame:
    """The rows of everyone the scored agent faced.

    `ne_missing` rather than `!=` so that a null `actor` counts as "not the
    scored agent" instead of propagating null and dropping the row: a row whose
    actor is unknown is still part of the environment the agent played against,
    and dropping it silently would shrink the empirical policy's support.
    """
    if actor is None:
        return df.filter(legacy)
    return df.filter(pl.col("actor").ne_missing(actor))


@dataclasses.dataclass(frozen=True)
class DropReport:
    """
    the rows a scorer keeps after dropping the scored LM's unusable rows, with the
    counts behind the dropped rate
    :param kept: the frame the estimate is computed from (pl.DataFrame)
    :param unit: what `n_dropped` and `n_total` count, "rows" or "hands" (str)
    :param n_dropped: units removed from the estimate (int)
    :param n_total: units in the scored population before the drop (int)
    :param n_fallback_rows: LM rows with `fallback_used` set, None on the belief side (int | None)
    :param n_lm_rows: LM rows before the drop, None on the belief side (int | None)
    :param n_infosets: distinct info-sets among the LM rows before the drop (int | None)
    :param n_infosets_fully_dropped: info-sets left with no kept LM row (int | None)
    """

    kept: pl.DataFrame
    unit: str
    n_dropped: int
    n_total: int
    n_fallback_rows: int | None = None
    n_lm_rows: int | None = None
    n_infosets: int | None = None
    n_infosets_fully_dropped: int | None = None

    @property
    def rate(self) -> float | None:
        """
        the fraction of the unit that was dropped
        :returns: n_dropped / n_total, or None when nothing was in the population
        """
        return self.n_dropped / self.n_total if self.n_total else None

    def detail(self) -> dict:
        """
        the counts behind the rate as one flat dict, without the frame
        :returns: dict with unit, n_dropped, n_total, n_fallback_rows, n_lm_rows,
            fallback_row_rate, n_infosets and n_infosets_fully_dropped
        """
        row_rate = (
            self.n_fallback_rows / self.n_lm_rows
            if self.n_lm_rows and self.n_fallback_rows is not None
            else None
        )
        return {
            "unit": self.unit,
            "n_dropped": self.n_dropped,
            "n_total": self.n_total,
            "n_fallback_rows": self.n_fallback_rows,
            "n_lm_rows": self.n_lm_rows,
            "fallback_row_rate": row_rate,
            "n_infosets": self.n_infosets,
            "n_infosets_fully_dropped": self.n_infosets_fully_dropped,
        }


def _infoset_counts(
    lm_before: pl.DataFrame, lm_kept: pl.DataFrame
) -> tuple[int | None, int | None]:
    """
    how many info-sets the LM visited and how many were left with no kept row
    :param lm_before: the LM rows before the drop (pl.DataFrame)
    :param lm_kept: the LM rows after the drop (pl.DataFrame)
    :returns: (n_infosets, n_infosets_fully_dropped), both None without an info_set column
    """
    if "info_set" not in lm_before.columns:
        return None, None
    before = set(lm_before["info_set"].to_list())
    kept = set(lm_kept["info_set"].to_list())
    return len(before), len(before - kept)


def _lm_mask(rows: pl.DataFrame) -> pl.Expr:
    """
    which rows are the scored LM's own
    A row is the LM's when it carries a prompt. A frame with neither `prompt` nor
    `fallback_used` has nothing to drop, so every row counts as the LM's; a frame
    with `fallback_used` but no `prompt` cannot tell an LM fallback from an
    opponent's legitimate one, so it is refused rather than guessed at.
    :param rows: the scorer's already-selected population (pl.DataFrame)
    :returns: a boolean expression selecting the LM rows (pl.Expr)
    """
    if "prompt" in rows.columns:
        return pl.col("prompt").is_not_null()
    if "fallback_used" in rows.columns:
        raise ValueError(
            "the frame has a `fallback_used` column but no `prompt` column, so the "
            "scored LM's rows cannot be told from an opponent's legitimate fallback "
            "rows"
        )
    return pl.lit(True)


def drop_fallback_rows(rows: pl.DataFrame) -> DropReport:
    """
    remove the scored LM's own harness-fallback rows from a row-level estimate
    Only rows with a non-null `prompt` are ever removed: the Bayesian and
    CFR-reference agents set `fallback_used` legitimately and carry no prompt.
    A null `fallback_used` counts as a fallback: the column is never null when the harness
    wrote the row, so a null is a row nobody can vouch for, and it is dropped and counted in
    the dropped rate, the same reading the pilot gate takes. A frame without the column
    drops nothing.
    :param rows: the scorer's already-selected population (pl.DataFrame)
    :returns: DropReport with unit "rows", `kept` being `rows` minus the LM fallback rows
    """
    is_lm = _lm_mask(rows)
    lm_before = rows.filter(is_lm)
    if "fallback_used" not in rows.columns:
        n_infosets, n_full = _infoset_counts(lm_before, lm_before)
        return DropReport(
            kept=rows,
            unit="rows",
            n_dropped=0,
            n_total=lm_before.height,
            n_fallback_rows=0,
            n_lm_rows=lm_before.height,
            n_infosets=n_infosets,
            n_infosets_fully_dropped=n_full,
        )
    is_fallback = is_lm & pl.col("fallback_used").fill_null(True)
    kept = rows.filter(~is_fallback)
    n_fallback = rows.filter(is_fallback).height
    n_infosets, n_full = _infoset_counts(lm_before, kept.filter(is_lm))
    return DropReport(
        kept=kept,
        unit="rows",
        n_dropped=n_fallback,
        n_total=lm_before.height,
        n_fallback_rows=n_fallback,
        n_lm_rows=lm_before.height,
        n_infosets=n_infosets,
        n_infosets_fully_dropped=n_full,
    )


def drop_fallback_hands(rows: pl.DataFrame, hand_keys: list[str]) -> DropReport:
    """
    remove every hand that holds one of the scored LM's harness-fallback rows,
    for an estimate whose unit is the hand (one reward per hand)
    Dropping only the fallback row would leave the hand's reward, which the
    harness's substituted move helped produce, in the estimate.
    :param rows: the scorer's already-selected population (pl.DataFrame)
    :param hand_keys: the columns that identify one hand, e.g. seed and hand_idx (list[str])
    :returns: DropReport with unit "hands"; `n_total` counts hands with an LM row and
        the row counts stay row counts
    """
    is_lm = _lm_mask(rows)
    lm_before = rows.filter(is_lm)
    n_hands = lm_before.select(hand_keys).unique().height
    if "fallback_used" not in rows.columns:
        n_infosets, n_full = _infoset_counts(lm_before, lm_before)
        return DropReport(
            kept=rows,
            unit="hands",
            n_dropped=0,
            n_total=n_hands,
            n_fallback_rows=0,
            n_lm_rows=lm_before.height,
            n_infosets=n_infosets,
            n_infosets_fully_dropped=n_full,
        )
    fallback_rows = rows.filter(is_lm & pl.col("fallback_used").fill_null(True))
    bad_hands = fallback_rows.select(hand_keys).unique()
    kept = rows.join(bad_hands, on=hand_keys, how="anti", maintain_order="left")
    n_infosets, n_full = _infoset_counts(lm_before, kept.filter(is_lm))
    return DropReport(
        kept=kept,
        unit="hands",
        n_dropped=bad_hands.height,
        n_total=n_hands,
        n_fallback_rows=fallback_rows.height,
        n_lm_rows=lm_before.height,
        n_infosets=n_infosets,
        n_infosets_fully_dropped=n_full,
    )


def belief_drop_report(lm_rows: pl.DataFrame) -> DropReport:
    """
    count the belief-side rows a belief scorer already excludes, without adding a gate
    A row counts as attempted when it carries a belief or is flagged `belief_failed`.
    It counts as dropped when the attempt produced no usable belief, or when the
    stated belief is not parseable JSON. Both counts are taken over the same rows,
    so the rate cannot exceed 1. Rows excluded for lacking a `true_posterior`, or
    by a scorer's own key alignment, are not the model's failure and are not
    counted. `fallback_used` is never read: it describes the action call, and a
    belief is the model's own separate output.
    :param lm_rows: the scored LM's rows (pl.DataFrame)
    :returns: DropReport with unit "rows"; `kept` holds the rows with a belief present
    """
    has_belief = pl.col("elicited_belief").is_not_null()
    attempted = has_belief
    if "belief_failed" in lm_rows.columns:
        attempted = attempted | pl.col("belief_failed").fill_null(False)
    n_attempted = lm_rows.filter(attempted).height
    n_missing = lm_rows.filter(attempted & ~has_belief).height
    n_unparseable = sum(
        not _parses(belief)
        for belief in lm_rows.filter(has_belief)["elicited_belief"].to_list()
    )
    return DropReport(
        kept=lm_rows.filter(has_belief),
        unit="rows",
        n_dropped=n_missing + n_unparseable,
        n_total=n_attempted,
    )


def _parses(belief: str) -> bool:
    """
    whether a stated belief is parseable JSON
    :param belief: the `elicited_belief` cell (str)
    :returns: False when `json.loads` raises the errors the scorers skip on (bool)
    """
    try:
        json.loads(belief)
    except (json.JSONDecodeError, TypeError):
        return False
    return True


def refuse_foreign_probe_type(df: pl.DataFrame, where: str) -> None:
    """Refuse a frame that carries a probe arm's rows beside a base arm's.

    P2/P3/P5/P7/P8 each score exactly one arm's own base population; a probe
    arm (relabel/payoff_scale/offpath_drive) lives in its own directory and is
    read only by the paired scorers (P1/P4) or by P6, which already refuse a
    foreign `probe_type` the same way. Without this guard, a frame assembled
    from a directory that also holds a probe arm's shards (reachable
    through an unfiltered `--data` glob) would
    be silently averaged in as though it were more base-arm evidence.

    No-op when the column is absent: every collected arm writes
    `probe_type` on every row, so an absent column means a synthetic/legacy
    frame this guard was never meant to police, not a base arm to trust
    blindly. Unlike P6's mirror-image guard, a null value here counts as safe
    rather than foreign: these five scorers want rows that make no special
    claim, which is exactly what a base row is, whereas P6 needs rows to
    affirmatively be the offpath population it is measuring.
    """
    if "probe_type" not in df.columns:
        return
    foreign = sorted(
        {
            p
            for p in df["probe_type"].to_list()
            if p is not None and p != BASE_PROBE_TYPE
        },
        key=str,
    )
    if foreign:
        raise ValueError(
            f"{where}: the frame carries probe_type(s) {foreign} beside "
            f"{BASE_PROBE_TYPE!r}. This scorer measures the BASE arm's own "
            "population; a probe arm's rows mixed in would be averaged into "
            "the base estimate with nothing saying so. Score the base arm's "
            "own directory."
        )


def condition_rows(df: pl.DataFrame, *, memory_rung: str = "M0") -> pl.DataFrame:
    """
    Select rows matching the requested memory rung.

    Composes with `actor_rows` and existing probe_type handling (one selection
    seam, not per-column filters). A column nobody filters on is not a
    separation - this default-M0 filter prevents M0 and
    M1-M3 rows from silently pooling in scorer aggregates.
    :param df: Frame with optional `memory_rung` column (Utf8, nullable).
    :param memory_rung: The rung to select ("M0", "M1", "M2", "M3").
    :returns: Filtered frame:
        - Frame has column: rows where `memory_rung` equals the requested value,
        plus null rows at every rung (null means non-LM row, the environment
        population that `environment_rows` needs for P2's empirical opponent
        policy; dropping nulls at non-M0 rungs empties the policy and leaves
        br_val NaN).
        - Frame lacks column: M0 request passes through unchanged; non-M0
        request raises ValueError (no silent fallbacks).
    :raises ValueError: If memory_rung is not in MEMORY_RUNGS, or if a non-M0 rung
        is requested but the frame lacks the column.
    """
    from lmopt.prompting.memory import MEMORY_RUNGS

    if memory_rung not in MEMORY_RUNGS:
        raise ValueError(
            f"Unknown memory_rung: {memory_rung!r}. Valid rungs: {MEMORY_RUNGS}"
        )

    if "memory_rung" not in df.columns:
        # Column absent: M0 passes through (landed arms); non-M0 refuses.
        if memory_rung == "M0":
            return df
        raise ValueError(
            f"condition_rows: memory_rung={memory_rung!r} was requested but the "
            "frame has no 'memory_rung' column (an arm collected before it existed). Re-run the arm, "
            f"or score it with memory_rung='M0' (the default)."
        )

    # Column present: match the requested value, plus nulls at every rung.
    # Null memory_rung means non-LM row (environment population for P2).
    return df.filter(
        (pl.col("memory_rung") == memory_rung) | pl.col("memory_rung").is_null()
    )
