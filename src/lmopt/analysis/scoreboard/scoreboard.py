"""
The scoreboard: one board per memory level, each one cell per (model, game, property),
classified from what the scorers returned.

A cell reads a number, ``partial: k/n opponents``, ``structural: <reason>``,
``awaiting data: <reason>``, ``inadmissible: <value>, verdict withheld: <reason>`` or
``BROKEN: <reason>``, and BROKEN is reserved for an instrument fault. The classification is a computation
over the scored rows and the project's own admissibility gate, never a table of properties
or reasons. This module is pure: it takes rows and callables from the caller and touches no
files beyond hashing the one it is handed.

Every cell carries two texts: the one the table shows, which may show only the leading
clause of a reason, and the whole text, which the detail section under the table prints
unabridged. A reason is never cut without a mark.
"""

import collections
import functools
import dataclasses
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from lmopt.analysis.analysis_helpers import is_finite_number
from lmopt.analysis.scoreboard.scoreboard_sources import file_digest
from lmopt.analysis.property_rows import lm_rows
from lmopt.analysis.scoreboard.scoreboard_sources import (
    MEMORY_NOTE_PREFIX,
    SCOREBOARD_GAMES,
)
from lmopt.analysis.scoring_vocabulary import FAILED_REASON

BEGIN_MARKER = "<!-- SCOREBOARD:BEGIN -->"
END_MARKER = "<!-- SCOREBOARD:END -->"
NUMBER = "number"
PARTIAL = "partial"
#: A partial cell whose every not-scored opponent carries an approved reason: one of the three
#: kept-out classes or `not_reached` (`partial_reason_class` on the properties row).
#: A missing opponent (no row at all) never qualifies -- there is nothing measured to
#: approve. Rendered text is unchanged (still `"partial: k/n opponents"`); only the kind
#: -- and so the tally -- differs, which is what lets the bar ("no `partial` cell unless
#: every missing opponent carries an approved reason") read directly off the board.
PARTIAL_APPROVED = "partial_approved"
STRUCTURAL = "structural"
AWAITING = "awaiting"
#: A number shown with its verdict withheld, for a cell the project's own admissibility gate
#: blocked on a defect the pre-registration names. The gate working as designed, not a fault.
INADMISSIBLE = "inadmissible"
BROKEN = "BROKEN"
#: Every kind a cell can have, in the order the header counts them. BROKEN is last and is
#: reserved for an instrument fault: a row the scorer should have written and did not, a
#: duplicate or unexpected opponent row, a status the board does not know, a gate count that
#: disagrees with the rows, or a measurable row with neither a value nor a reason.
KINDS = (NUMBER, PARTIAL, PARTIAL_APPROVED, STRUCTURAL, AWAITING, INADMISSIBLE, BROKEN)
PROPERTY_NUMBERS = tuple(range(1, 9))
REQUIRED_COLUMNS = (
    "game",
    "opponent",
    "player",
    "property",
    "value",
    "status",
    "caveat",
)
MAX_REASON_CHARS = 110
#: The words a scorer's zero-variance clause opens with. That clause says why a zero-width
#: interval is not a tight one, so a table cell may not cut it into a `(+N more)` count.
ZERO_VARIANCE_PREFIX = "zero between-seed variance"
#: Every column a scorer writes to say why, by artefact, that a board cell must be able to show.
#: `blocker` renders on a blocked or inadmissible cell; on an admissible one it carries only the
#: property's standing text, which no cell repeats.
#: A standing test discovers the reason-named columns each writer emits and refuses one this
#: table neither lists nor names in NOT_RENDERED_REASON_COLUMNS, and pushes a sentinel through
#: each listed column to a cell's text, so a scorer's reason cannot be written and silently not
#: read.
REASON_COLUMNS = {
    "properties.csv": ("caveat", "exact_reason"),
    "decision.csv": ("blocker", "verdict_reason"),
    "probe_scores.csv": ("caveat", "verdict_reason", "exact_reason"),
}
#: Reason columns deliberately not rendered, each with the reason it is not.
NOT_RENDERED_REASON_COLUMNS: dict[str, dict[str, str]] = {
    "properties.csv": {
        # The text already reaches a
        # cell via `caveat`, which `_property_row` rewrites to the matching
        # canonical sentence when this column is non-empty; this column
        # is the machine-readable form `classify()` reads to tally
        # approved-versus-blocking without re-parsing rendered text.
        "partial_reason_class": (
            "text already reaches a cell via caveat; this is the "
            "machine-readable form scoreboard.py reads for the tally"
        ),
    },
}
#: Reason columns deliberately not rendered, continued: the decision row's own account of why P7b stayed on the
#: sampled basis is the cells' `exact_reason`s said once for the property, and a cell renders those.
NOT_RENDERED_REASON_COLUMNS["decision.csv"] = {
    "basis_reason": (
        "the same cells' exact_reason reaches the cell through properties.csv; this is the "
        "property-level statement of it for a reader of decision.csv"
    ),
    "blocker_class": (
        "the machine-readable approved-reason classes of the rows behind `blocker`, which "
        "`classify()` reads to tell an inadmissible cell from a BROKEN one; the words reach a "
        "cell through `blocker`"
    ),
}
#: Clauses a short cell always shows in full, ahead of the leading clause.
PROTECTED_CLAUSE_PREFIXES = (ZERO_VARIANCE_PREFIX, MEMORY_NOTE_PREFIX)
#: A property with two headline rows (P7) caps each part so neither can push the other out.
MAX_PART_CHARS = 260
#: The longest text of a cell with a single headline row.
MAX_CELL_CHARS = 520
MEASURABLE = "MEASURABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
AWAITING_DATA = "AWAITING_DATA"
#: A scored cell whose measurement is genuinely absent because the arm never reached the
#: nodes the property is defined on; the scorer states the reason and the board renders it.
NOT_REACHED = "NOT_REACHED"
#: legacy only. Text a gate blocker must contain for the withheld number to read as
#: inadmissible, not BROKEN, for an arm whose rows predate `partial_reason_class`. A current
#: arm is decided by the `blocker_class` field `decision_table` writes, never by this wording.
#: This constant and its text branch are deleted once the landed arms are re-scored.
#: `_partial_reason_class` rewrites the state-sensitivity-failure caveat's own sentence
#: (the legacy text starting "S7 pre-registered cell") to `FAILED_REASON`'s
#: canonical text wherever the check genuinely fails, so matching the legacy string alone
#: would let such cells fall silently from `inadmissible` to `BROKEN`. Both strings are kept:
#: the canonical one because it is what the scorer writes, the legacy one because a stray
#: legacy artefact may still carry it.
PREREGISTERED_BLOCKERS = (
    "S7 pre-registered cell",
    FAILED_REASON,
)
#: The leading words of a structural cell that is a fact about one arm's play and not about
#: the game, so no other board may borrow it as if the property could not exist there.
NOT_REACHED_PREFIX = "structural (not reached, not by design): "
#: Why both verdicts of an off-path cell are withheld when its analytic uniform baseline is
#: exactly zero: the reference is then uniform at every scored node, so the criterion "below the
#: uniform baseline" cannot be met by any policy and a verdict would only restate that.
#: `probe_scores` writes this exact text and the board reads it as an approved structural
#: reason, so both share this one constant (it lives here because `probe_scores` already
#: imports from this module).
#: The `exact_reason` of a paired (P1 or P4) row whose distribution-basis value is present and
#: which carries no verdict on that basis, by design: the verdict of those two properties is
#: issued by the decision table, on the sampled scale, at the threshold that table holds. The
#: board reads this exact text as its own state, distinct from a distribution value that has no
#: verdict for any other reason, which reads as awaiting data.
PAIRED_EXACT_NOTE = (
    "the distribution value is reported beside the sampled one and the verdict of this "
    "property is issued by the decision table on the sampled scale, and its interval spans "
    "the info-sets each seed visited, since each arm has one pass, so it is not a sampling interval"
)
ZERO_BASELINE_REASON = (
    "off-path reference carries no information here: it is uniform at every scored node, "
    "so the criterion cannot be met"
)
#: The leading words of a P6 cell whose off-path reference is uniform at every scored node, so
#: no verdict can be given to any policy. Like the not-reached prefix it is a stated reason and
#: not a property that cannot exist, but it is a fact about the game's reference and not about
#: one model's play, so it reads the same for every model on that game.
ZERO_BASELINE_PREFIX = "structural (the reference carries no information here): "
PROBE_NOTE = (
    "P1, P4 and P6 cells are read from `probe_scores.csv` (paired total variation over the base "
    "and probe arms, or the off-path distance to the tremble reference): they are per game, not "
    "pooled over opponents, and no admissibility gate is applied to them. A P6 cell carries its "
    "verdict against the sentinel anchor measured on its own game; a game whose anchor is missing, "
    "or whose recorded conditions the arm disagrees with, reads awaiting data with that reason, "
    "unless the cell is BROKEN. A P6 cell reads its DISTRIBUTION basis (the arm's exact "
    "per-info-set policy against the analytic uniform baseline) wherever the arm has an "
    "exact-policy pass, with the sampled distance beside it, and the sampled basis alone where it "
    "has none; the cell names which basis its number is on."
)
#: The per-arm file `scripts/arm_collapse_pass.py` writes beside `properties.csv`, and its
#: columns. An arm without one has not been checked for collapse, which is never the same
#: statement as having been checked and found distinct.
COLLAPSE_FILE = "collapse.csv"
COLLAPSE_COLUMNS = (
    "game",
    "fingerprint",
    "collapsed_with",
    "first_listed_share",
    "n_rows",
)
COLLAPSE_NOT_CHECKED = "collapse not checked"
#: The per-arm file `scripts/arm_collapse_pass.py --offpath-root` writes beside `collapse.csv`: the
#: collapse record of the arm's driven off-path play, which is what a P6 cell reads. `offpath_root`
#: names the directory the pass read, so a record can be traced to its data.
COLLAPSE_OFFPATH_FILE = "collapse_offpath.csv"
COLLAPSE_OFFPATH_COLUMNS = (
    "game",
    "fingerprint",
    "collapsed_with",
    "constant_action",
    "n_rows",
    "n_infosets",
    "offpath_root",
)
OFFPATH_COLLAPSE_NOT_CHECKED = "; off-path collapse not checked"
DETAIL_NOTE = (
    "{n} cell(s) above show only their leading clause, marked `(+N more)` or cut with an "
    "ellipsis. Each one's whole text is below; nothing a scorer said is lost."
)
PROBE_FILE_MISSING = (
    "awaiting data: no probe_scores.csv for this arm (scripts/probe_scores.py has not been "
    "given this arm's relabel, payoff-scale and off-path runs)"
)


@dataclasses.dataclass(frozen=True)
class Cell:
    """
    one classified cell
    :param kind: NUMBER, PARTIAL, STRUCTURAL, AWAITING, INADMISSIBLE or BROKEN
    :param text: the text the table cell renders as, which may show only a leading clause
    :param detail: the whole text with every clause, empty when it equals `text`
    :param legacy_blocker_rule: True when the kind was decided by matching blocker wording
        because the arm's rows predate `partial_reason_class`; False otherwise
    """

    kind: str
    text: str
    detail: str = ""
    legacy_blocker_rule: bool = False

    @property
    def full_text(self) -> str:
        """
        :returns: the whole text of the cell, whether or not the table cell was shortened
        """
        return self.detail or self.text

    @property
    def shortened(self) -> bool:
        """
        :returns: True when the table cell shows less than the whole text
        """
        return bool(self.detail) and self.detail != self.text


@dataclasses.dataclass(frozen=True)
class GridCell:
    """
    a cell with its coordinates
    :param arm: the arm (model) name
    :param game: the game key
    :param number: the property number 1 to 8
    :param cell: the classified Cell
    """

    arm: str
    game: str
    number: int
    cell: Cell


@dataclasses.dataclass(frozen=True)
class ArmSummary:
    """
    the provenance shown for one arm in the block header
    :param name: the arm (model) name
    :param data: the shortened data directory its run_meta names
    :param sha256: short sha256 of its properties.csv
    :param probe_sha256: short sha256 of its probe_scores.csv, or None when it has none
    :param collapsed_games: the games on which this arm shares a policy with another arm, or
        None when the arm has no collapse file and was therefore never checked
    :param offpath_collapsed_games: the games on which this arm's driven off-path play is one
        another arm also plays, or None when the arm has no off-path collapse file
    :param memory_rung: the memory level the arm's own condition block records, or None when
        it records no condition
    """

    name: str
    data: str
    sha256: str
    probe_sha256: str | None = None
    collapsed_games: tuple[str, ...] | None = None
    memory_rung: str | None = None
    offpath_collapsed_games: tuple[str, ...] | None = None


def _four_places(x: float) -> str:
    """
    :param x: a finite number
    :returns: the number to four decimals, with a negative zero shown as zero
    """
    text = f"{x:.4f}"
    return "0.0000" if text == "-0.0000" else text


def split_clauses(text) -> list[str]:
    """
    the clauses of a caveat or a blocker, split in the one place the project splits them
    :param text: caveat or blocker text, or None
    :returns: the semicolon-separated clauses, whitespace collapsed, pipes made safe for a
        table cell, empty clauses dropped
    """
    clauses = []
    for raw in str(text or "").split(";"):
        clause = re.sub(r"\s+", " ", raw).strip().replace("|", "/")
        if clause:
            clauses.append(clause)
    return clauses


def replicate_clause(n, distinct, spread) -> str:
    """
    the words for what a cell's per-seed values did, one wording per case from one place
    A zero-width interval reads as agreement among draws, so that case alone is worded as a
    sentence a caveat carries; a cell whose seeds scored different values states its own
    width, and its wording is for the reader of the columns.
    :param n: the count of per-seed values the interval was formed over, or None (int | float)
    :param distinct: how many distinct raw values those were, or None (int | float)
    :param spread: the largest per-seed value minus the smallest, or None (float)
    :returns: the sentence for one value repeated across two or more seeds, the
        count-and-spread wording for differing values (the spread clause omitted when the
        spread is not a finite number), and an empty string for fewer than two seeds or no data
    """
    if not (is_finite_number(n) and is_finite_number(distinct)) or n < 2:
        return ""
    if distinct == 1:
        return f"{ZERO_VARIANCE_PREFIX}: 1 distinct value across {int(n)} seeds"
    counts = f"{int(n)} seeds, {int(distinct)} distinct values"
    return f"{counts}, spread {spread:.3g}" if is_finite_number(spread) else counts


def _degenerate_clause(row: Mapping, suffix: str = "") -> str:
    """
    :param row: an off-path `probe_scores.csv` row (Mapping)
    :param suffix: "" for the sampled basis, "_exact" for the distribution basis
    :returns: the zero-variance sentence when that basis's seeds all scored one value, else an
        empty string; a row from a file that predates the columns has none
    """
    n, distinct = row.get(f"n_replicates{suffix}"), row.get(
        f"n_replicates_distinct{suffix}"
    )
    if not (is_finite_number(n) and is_finite_number(distinct) and distinct == 1):
        return ""
    return replicate_clause(n, distinct, 0.0)


def _clause(caveat) -> str:
    """
    the leading clause of a caveat, marked whenever later clauses were cut
    :param caveat: caveat text or None
    :returns: the first clause, capped, followed by `(+N more)` when N clauses were cut; a
        clause opening with a PROTECTED_CLAUSE_PREFIXES entry is never cut, and is shown
        first, so it is never the clause a `(+N more)` hides
    """
    clauses = split_clauses(caveat)
    if not clauses:
        return ""
    protected = [c for c in clauses if c.startswith(PROTECTED_CLAUSE_PREFIXES)]
    rest = [c for c in clauses if not c.startswith(PROTECTED_CLAUSE_PREFIXES)]
    shown = list(protected)
    if rest:
        text = rest[0]
        if len(text) > MAX_REASON_CHARS:
            text = text[: MAX_REASON_CHARS - 3].rstrip() + "..."
        shown.append(text)
    joined = "; ".join(shown)
    hidden = len(rest) - 1
    return f"{joined} (+{hidden} more)" if hidden > 0 else joined


def _whole(caveat) -> str:
    """
    :param caveat: caveat text or None
    :returns: every clause, joined back with semicolons and nothing cut
    """
    return "; ".join(split_clauses(caveat))


def _reason_text(caveat, short: bool) -> str:
    """
    :param caveat: caveat text or None
    :param short: whether the table form (leading clause, marked) is wanted
    :returns: the leading clause when short, the whole text otherwise
    """
    return _clause(caveat) if short else _whole(caveat)


def _is_scored(row: Mapping) -> bool:
    """
    :param row: a properties row
    :returns: True when the row is admissible under the project's gate (MEASURABLE with a
        finite value); the gate itself lives in `decision_table`, and `classify` checks the
        two agree
    """
    return row["status"] == MEASURABLE and is_finite_number(row["value"])


#: What stopped a row the scorer called measurable: it is the finite-value gate, never the
#: status, so the cell names the gate rather than echoing MEASURABLE back at the reader.
NO_FINITE_VALUE = (
    "stopped by the finite-value gate: the scorer returned no finite value"
)


def _row_reason(row: Mapping, short: bool = True) -> str:
    """
    :param row: a properties row
    :param short: whether the table form of the caveat is wanted
    :returns: what stopped the row, named by the gate it failed for a measurable row and by
        its own status otherwise, with its caveat
    """
    clause = _reason_text(row["caveat"], short)
    head = NO_FINITE_VALUE if row["status"] == MEASURABLE else str(row["status"])
    return f"{head}: {clause}" if clause else head


#: The wording of each approved class of an opponent whose measured value is kept out of the
#: pooled mean and the verdict. Keyed on `partial_reason_class`, never on the caveat's text.
THIN_LABEL = (
    "One prompt per (legal-mask bin, seat), so state-sensitivity cannot be tested"
)
SINGLE_LINE_LABEL = "single line of play: one distinct prompt over {hands} hands"
STRUCTURAL_LABEL = (
    "deterministic opponent and no chance node, so no within-seat variation"
)
FAILED_LABEL = "constant action or flat belief, a model result"
KEPT_OUT_CLASSES = ("thin", "structural", "failed")


def _kept_out_text(row: Mapping) -> str | None:
    """
    the text of an opponent the gate kept out of the mean, with its value and class label
    :param row: a properties row of an opponent the gate did not admit
    :returns: the opponent's measured value with its class label, or None when the row carries
        no finite value or its `partial_reason_class` is none of thin, structural or failed
    """
    kind = row.get("partial_reason_class")
    if kind not in KEPT_OUT_CLASSES or not is_finite_number(row["value"]):
        return None
    label = {"structural": STRUCTURAL_LABEL, "failed": FAILED_LABEL}.get(
        kind, THIN_LABEL
    )
    if kind == "thin" and row.get("n_distinct_prompts") == 1:
        label = SINGLE_LINE_LABEL.format(hands=row.get("n_hands"))
    return f"{_four_places(row['value'])} ({kind}: {label}; kept out of the mean)"


def dropped_text(rate) -> str:
    """
    the dropped rate as a cell shows it, never defaulted to zero
    :param rate: a fraction of LM rows dropped, or None when nothing recorded one
    :returns: the rate as a percentage, or a statement that no rate was recorded
    """
    if not is_finite_number(rate):
        return "dropped rate not recorded"
    return f"dropped {rate * 100:.2f}%"


def _verdict_text(decision_row: Mapping | None) -> str:
    """
    the interval and the verdict the project's own gate reached, beside the mean
    :param decision_row: a `decision_table` row, or None
    :returns: " [lo, hi] VERDICT" when the row carries both, " VERDICT" when only the
        verdict, and an empty string when the row carries neither
    """
    if not decision_row:
        return ""
    verdict = decision_row.get("verdict")
    lo, hi = decision_row.get("ci_lo"), decision_row.get("ci_hi")
    interval = (
        f" [{_four_places(lo)}, {_four_places(hi)}]"
        if is_finite_number(lo) and is_finite_number(hi)
        else ""
    )
    return f"{interval} {verdict}" if verdict else interval


def _row_set_problem(lm: Sequence[Mapping], expected: Sequence[str]) -> Cell | None:
    """
    the first gate of `classify`: the LM rows must cover each expected opponent at most once
    :param lm: the LM rows of one game and row
    :param expected: the opponents the rows may name
    :returns: a BROKEN cell naming every duplicated or unexpected opponent, or None when the
        row set is well formed
    """
    seen = collections.Counter(str(r["opponent"]) for r in lm)
    problems = []
    duplicated = sorted(o for o, n in seen.items() if n > 1)
    unexpected = sorted(o for o in seen if o not in expected)
    if duplicated:
        problems.append("duplicate LM rows for " + ", ".join(duplicated))
    if unexpected:
        problems.append("unexpected opponent(s) " + ", ".join(unexpected))
    if problems:
        return Cell(BROKEN, "BROKEN: " + "; ".join(problems))
    return None


def _valued_cell(
    decision_row: Mapping | None,
    lm: Sequence[Mapping],
    scored: Sequence[Mapping],
    expected: Sequence[str],
    missing: Sequence[str],
    mean: float,
    distribution: bool,
) -> Cell:
    """
    the second gate of `classify`: a cell carrying the gate's mean, a number or a partial
    :param decision_row: the `decision_table` row for that game and row, or None
    :param lm: every LM row
    :param scored: the LM rows the scorer counted as admissible, at least one
    :param expected: the opponents every LM row set must cover
    :param missing: the expected opponents with no row
    :param mean: the gate's finite admissible mean
    :param distribution: whether the gate's basis is the distribution basis
    :returns: NUMBER when every expected opponent is scored, PARTIAL_APPROVED or PARTIAL
        otherwise, and BROKEN when a distribution-basis row lacks a finite value_exact
    """
    # The valued text prints each scored row's `value_exact` on this basis, so a row
    # without a finite one is a defect in the rows, shown as one rather than raised.
    unvalued = sorted(
        str(r["opponent"])
        for r in scored
        if distribution and not is_finite_number(r.get("value_exact"))
    )
    if unvalued:
        return Cell(
            BROKEN,
            f"BROKEN: distribution basis but no finite value_exact for "
            f"{', '.join(unvalued)}",
        )
    complete = len(scored) == len(expected)
    # The gate's own recorded executed-move mean, never a second computation here, so the
    # cell and decision.csv cannot disagree; a missing or non-finite one is said so.
    executed = (decision_row or {}).get("observed_mean_admissible_executed")
    executed_text = (
        f"executed mean {_four_places(executed)}"
        if is_finite_number(executed)
        else "executed mean not recorded"
    )

    def valued(short: bool) -> str:
        head = (
            f"{_four_places(mean)}{_verdict_text(decision_row)} "
            f"({len(scored)}/{len(expected)} opponents)"
            + (f"; distribution basis; {executed_text}" if distribution else "")
        )
        if not complete:
            head = f"partial: {len(scored)}/{len(expected)} opponents; mean {head}"
        values = ", ".join(
            (
                f"{str(r['opponent'])} exact {_four_places(r['value_exact'])} "
                f"(executed {_four_places(r['value'])})"
                if distribution
                else f"{str(r['opponent'])} {_four_places(r['value'])}"
            )
            for r in sorted(scored, key=lambda r: str(r["opponent"]))
        )
        others = sorted(
            [
                (
                    str(r["opponent"]),
                    _kept_out_text(r) or _row_reason(r, short),
                )
                for r in lm
                if not _is_scored(r)
            ]
            + [(o, "no row") for o in missing]
        )
        parts = [
            head,
            values,
            dropped_text(decision_row.get("dropped_rate") if decision_row else None),
        ]
        # Why the gate withheld or qualified the verdict, when it says. The reason is
        # written to decision.csv; a cell that omitted it read as though none was given.
        verdict_why = _reason_text((decision_row or {}).get("verdict_reason"), short)
        if verdict_why:
            parts.append("verdict reason: " + verdict_why)
        # What the scorer said about the rows that did contribute. Without this, a
        # caveat on an admissible row reaches no reader: the other parts list reasons
        # only for the opponents that did not contribute, so a note qualifying the
        # number itself, such as the share of rows where every action tied, is invisible.
        notes = sorted(
            {
                _reason_text(r["caveat"], short)
                for r in scored
                if split_clauses(r["caveat"])
            }
        )
        if notes:
            parts.append("notes: " + "; ".join(notes))
        if not distribution:
            absent = sorted(
                {
                    text
                    for text in [(decision_row or {}).get("basis_reason")]
                    + [r.get("exact_reason") for r in scored]
                    if text
                }
            )
            if absent:
                parts.append(
                    "sampled basis, no distribution basis: "
                    + "; ".join(_reason_text(t, short) for t in absent)
                )
        if others:
            parts.append("not scored: " + "; ".join(f"{o} {why}" for o, why in others))
        return "; ".join(parts)

    # A `partial` cell reads
    # `PARTIAL_APPROVED` instead of `PARTIAL` when every not-scored
    # opponent's row carries an approved reason (the three kept-out classes, or `not_reached`)
    # (`partial_reason_class`, non-empty). A genuinely missing opponent
    # (`missing`, no row at all) never qualifies -- there is nothing
    # measured to approve -- so `PARTIAL_APPROVED` requires `not missing`
    # as well as the all(...) check, never one alone.
    not_scored = [r for r in lm if not _is_scored(r)]
    approved = (
        not missing
        and bool(not_scored)
        and all(r.get("partial_reason_class") for r in not_scored)
    )
    kind = NUMBER if complete else (PARTIAL_APPROVED if approved else PARTIAL)
    return Cell(kind, valued(True), valued(False))


def _reasoned_cell(
    lm: Sequence[Mapping], missing: Sequence[str], structural_by_design: bool
) -> Cell | None:
    """
    the third gate of `classify`: awaiting data, structural by design, or never reached
    :param lm: every LM row
    :param missing: the expected opponents with no row
    :param structural_by_design: whether the property cannot exist for this game's family
    :returns: the awaiting or structural cell when every LM row states a reason of one of
        those kinds and no opponent is missing, otherwise None
    """
    if (
        not missing
        and all(split_clauses(r["caveat"]) for r in lm)
        and all(r["status"] == AWAITING_DATA and r["value"] is None for r in lm)
    ):
        return _reasoned(AWAITING, "awaiting data: ", lm)
    if (
        structural_by_design
        and not missing
        and all(split_clauses(r["caveat"]) for r in lm)
        and all(r["status"] == NOT_APPLICABLE for r in lm)
    ):
        return _reasoned(STRUCTURAL, "structural: ", lm)
    if (
        not missing
        and all(split_clauses(r["caveat"]) for r in lm)
        and all(r["status"] == NOT_REACHED and r["value"] is None for r in lm)
    ):
        # Not by design: the property exists for this game, and this arm's rows never
        # reached the nodes it is defined on. The prefix says so, so a reader can tell it
        # from a property that cannot exist for the game at all.
        return _reasoned(STRUCTURAL, NOT_REACHED_PREFIX, lm)
    return None


def _blocked_cell(
    decision_row: Mapping,
    lm: Sequence[Mapping],
    missing: Sequence[str],
    blocked: float,
) -> Cell:
    """
    the fourth gate of `classify`: the gate withheld a verdict but recorded a value
    :param decision_row: the `decision_table` row for that game and row
    :param lm: every LM row
    :param missing: the expected opponents with no row
    :param blocked: the gate's finite mean over all cells
    :returns: INADMISSIBLE when the blocker is pre-registered and no opponent is missing,
        BROKEN otherwise
    """
    missing_note = ("; no row for " + ", ".join(missing)) if missing else ""
    structured = "failed" in split_clauses(decision_row["blocker_class"])
    # An arm scored before `partial_reason_class` existed has no such key on its rows, so
    # the only evidence left is the blocker's wording; the cell says which rule decided it.
    predates_class = any("partial_reason_class" not in r for r in lm)
    by_wording = (
        not structured
        and predates_class
        and any(
            marker in _whole(decision_row["blocker"])
            for marker in PREREGISTERED_BLOCKERS
        )
    )
    preregistered = structured or by_wording

    def blocker(short: bool) -> str:
        stated = _reason_text(decision_row["blocker"], short) or "; ".join(
            sorted({_row_reason(r, short) for r in lm})
        )
        if preregistered and not missing:
            return (
                f"inadmissible: value {_four_places(blocked)} shown, verdict withheld: "
                f"{stated}"
            )
        return f"BROKEN: {stated} (value {_four_places(blocked)}){missing_note}"

    if preregistered and not missing:
        return Cell(INADMISSIBLE, blocker(True), blocker(False), by_wording)
    return Cell(BROKEN, blocker(True), blocker(False))


def classify(
    row_name: str,
    decision_row: Mapping | None,
    property_rows: Sequence[Mapping],
    structural_by_design: bool,
    expected_opponents: Sequence[str],
) -> Cell:
    """
    classify one headline row of one game from what the scorer and the gate returned
    :param row_name: the headline row name, used in the absent-row message
    :param decision_row: the `decision_table` row for that game and row, or None
    :param property_rows: every properties row for that game and row (all players)
    :param structural_by_design: whether the property cannot exist for this game's family
    :param expected_opponents: the opponents every LM row set must cover exactly once; the
        denominator of "k/n opponents" is its length, never the number of rows present
    :returns: a Cell; a number only when every expected opponent is admissible, partial when
        some but not all are, structural only when every expected opponent has a
        NOT_APPLICABLE LM row with a stated reason on a by-design cell, BROKEN otherwise
    :raises ValueError: when the gate's admissible count disagrees with the rows
    """
    if not property_rows:
        return Cell(BROKEN, f"BROKEN: no {row_name} row")
    lm = lm_rows(property_rows)
    if not lm:
        return Cell(BROKEN, f"BROKEN: no LM row for {row_name}")
    expected = [str(o) for o in expected_opponents]
    problem = _row_set_problem(lm, expected)
    if problem is not None:
        return problem
    seen = {str(r["opponent"]) for r in lm}
    missing = sorted(o for o in expected if o not in seen)
    missing_note = ("; no row for " + ", ".join(missing)) if missing else ""
    scored = [r for r in lm if _is_scored(r)]
    mean = decision_row["observed_mean_admissible"] if decision_row else None
    distribution = (decision_row or {}).get("basis") == "distribution"
    if decision_row is not None and len(scored) != decision_row["n_cells_admissible"]:
        raise ValueError(
            f"{row_name}: the gate counts {decision_row['n_cells_admissible']} admissible "
            f"cells but the rows show {len(scored)}"
        )
    if is_finite_number(mean) and scored:
        return _valued_cell(
            decision_row, lm, scored, expected, missing, mean, distribution
        )
    reasoned = _reasoned_cell(lm, missing, structural_by_design)
    if reasoned is not None:
        return reasoned
    blocked = decision_row["observed_mean_all_cells"] if decision_row else None
    if decision_row is not None and is_finite_number(blocked):
        return _blocked_cell(decision_row, lm, missing, blocked)

    def unscored(short: bool) -> str:
        return (
            "BROKEN: "
            + "; ".join(sorted({_row_reason(r, short) for r in lm}))
            + missing_note
        )

    return Cell(BROKEN, unscored(True), unscored(False))


def _reasoned(kind: str, prefix: str, lm: Sequence[Mapping]) -> Cell:
    """
    a cell whose whole content is the distinct reasons its LM rows state
    :param kind: the cell kind
    :param prefix: the leading words of the cell text
    :param lm: the LM rows
    :returns: the Cell, with the leading clause of each reason in the table text and every
        clause in the detail text
    """
    short = sorted({_clause(r["caveat"]) for r in lm})
    whole = sorted({_whole(r["caveat"]) for r in lm})
    return Cell(kind, prefix + "; ".join(short), prefix + "; ".join(whole))


def _distribution_basis(row: Mapping):
    """
    the off-path cell as its DISTRIBUTION basis reads it, when the row carries one
    The distribution basis is the headline wherever it exists: it measures the policy
    itself, so a temperature-0 arm and a temperature-1 arm are comparable on it, which the
    sampled distance between point masses is not. The sampled basis is shown beside it, and
    a row with no exact-policy pass has no distribution basis at all.
    A judged cell labels both bases, `exact: <verdict>; sampled: <verdict> (<reason>)`, with
    the exact verdict as the headline, and adds `bases disagree` when both bases carry a
    verdict and the two differ. The labels and the sampled reason come before the sampled
    detail, so the cell cap, which cuts from the end, cannot remove them.
    :param row: an off-path `probe_scores.csv` row (Mapping)
    :returns: a callable taking the sampled cell's (short, full) text pair, without its
        verdict, and returning the Cell, or None when the row carries no distribution-basis
        value
    """
    value = row.get("value_exact")
    if not is_finite_number(value):
        return None
    verdict = row.get("verdict_exact")
    reason = row.get("exact_reason") or ""
    baseline = row.get("baseline_exact")
    coverage = row.get("exact_coverage")
    covered = (
        f", the pass covers {coverage * 100:.1f}% of them"
        if is_finite_number(coverage)
        else ""
    )
    against = (
        f"baseline {_four_places(baseline)}"
        if is_finite_number(baseline)
        else "no baseline"
    )
    # `_degenerate_clause`'s own sentence, not `row['caveat']`'s: the distribution basis has
    # its own `n_replicates_exact`/`n_replicates_distinct_exact` pair, distinct from the
    # sampled basis's, and no writer embeds this one in any caveat column, so appending it
    # here is not the duplicated-caveat shape -- there is nothing else to duplicate it with.
    note = _degenerate_clause(row, "_exact")
    note = f"distribution basis, {note}" if note else ""

    def cell(sampled_short: str, sampled_full: str) -> Cell:
        if not verdict and reason == PAIRED_EXACT_NOTE:
            # A paired row never carries a verdict on either basis: that is the decision table's.
            # The value is present and computed, so the cell shows it, and it must never read as
            # awaiting data, which would be a false structural reason for a measured number.
            head = (
                f"{_four_places(value)} (distribution basis, no verdict here{covered})"
            )
            tail = f"; {note}" if note else ""
            return Cell(
                NUMBER,
                f"{head}; sampled {sampled_short}{tail}",
                f"{head}; sampled {sampled_full}{tail}",
            )
        if not verdict:
            # The same rule the sampled basis follows: a distance with nothing to judge it
            # against is not published as a board number, and the reason is what the cell
            # carries instead. The measurement stays in probe_scores.csv.
            return Cell(
                AWAITING,
                f"awaiting data: {_clause(reason)}",
                f"awaiting data: {_whole(reason)}",
            )
        head = f"{_four_places(value)} exact: {verdict} (distribution basis, {against}{covered})"
        tail = f"; {note}" if note else ""
        return Cell(
            NUMBER,
            f"{head}; {sampled_verdict(True)}; sampled {sampled_short}{tail}",
            f"{head}; {sampled_verdict(False)}; sampled {sampled_full}{tail}",
        )

    def sampled_verdict(short: bool) -> str:
        sampled, why = row.get("verdict"), row.get("verdict_reason")
        text = f"sampled: {sampled or 'no verdict'}"
        if split_clauses(why):
            text += f" ({_reason_text(why, short)})"
        if sampled and sampled != verdict:
            text += "; bases disagree"
        return text

    return cell


def _reference_uniform_cell(row: Mapping) -> Cell | None:
    """
    the approved structural cell for an off-path row whose verdicts were withheld because its
    analytic baseline is zero
    :param row: an off-path `probe_scores.csv` row (Mapping)
    :returns: the structural cell when both bases are withheld with the canonical zero-baseline
        reason and the row records a baseline of exactly 0.0; None when neither carries the
        reason; a BROKEN cell naming the inconsistency when only one basis carries it, a verdict
        stands beside it, or the recorded baseline is missing or not zero, so a half-withheld or
        falsely withheld cell never reads as a number or as the approved reason (Cell | None)
    """
    sampled = row.get("verdict_reason") == ZERO_BASELINE_REASON
    exact = row.get("exact_reason") == ZERO_BASELINE_REASON
    if not (sampled or exact):
        return None
    baseline = row.get("baseline_exact")
    if not (is_finite_number(baseline) and baseline == 0.0):
        return Cell(
            BROKEN,
            "BROKEN: the zero-baseline reason is only valid for a recorded baseline of "
            "exactly 0.0, and this row records none or a different one",
        )
    if sampled and exact and not row.get("verdict") and not row.get("verdict_exact"):
        return Cell(STRUCTURAL, ZERO_BASELINE_PREFIX + ZERO_BASELINE_REASON)
    return Cell(
        BROKEN,
        "BROKEN: the zero-baseline reason must withhold both verdicts of an off-path cell, "
        "and this row withholds only part of it",
    )


def _probe_value_problem(row: Mapping, value: float) -> str | None:
    """
    the first gate of `classify_probe`: the value, count and coverage must be well formed
    :param row: a measurable `probe_scores.csv` row
    :param value: the row's finite value
    :returns: the sentence naming the first gate that stopped the row, or None when none did
    """
    count, coverage = row["n_infosets"], row["paired_coverage"]
    problem = None
    exact_value = row.get("value_exact")
    if not 0.0 <= value <= 1.0:
        problem = "the value gate stopped it: the value is outside [0, 1]"
    elif exact_value is not None and not (
        is_finite_number(exact_value) and 0.0 <= exact_value <= 1.0
    ):
        problem = (
            "the value gate stopped it: the distribution-basis value is not a "
            "distance in [0, 1]"
        )
    elif not (is_finite_number(count) and count > 0 and int(count) == count):
        problem = "the info-set gate stopped it: the info-set count is missing"
    elif coverage is not None and not (
        is_finite_number(coverage) and 0.0 <= coverage <= 1.0
    ):
        problem = "the coverage gate stopped it: the coverage is not a fraction"
    return problem


def _probe_shape(
    row: Mapping, value: float, floor: float | None
) -> tuple[str, Cell | None]:
    """
    the second gate of `classify_probe`: the info-set count and coverage beside the number, and
    the pre-registered coverage floor
    :param row: a measurable `probe_scores.csv` row that passed the value gates
    :param value: the row's finite value
    :param floor: the pre-registered minimum within-seed coverage, or None
    :returns: the phrase describing the info-sets and coverage, and an INADMISSIBLE cell when
        the within-seed coverage is below the floor, else None
    """
    count = int(row["n_infosets"])
    coverage = row["paired_coverage"]
    if coverage is not None:
        # P1/P4's headline `value` is the within-seed estimate, so the info-set
        # count and coverage beside it should be the per-seed pair it was formed
        # over, not the cross-seed union pair `n_infosets`/`paired_coverage` alone
        # describe -- the union pair stays visible too, since `value_pooled` is
        # still on the row, but neither number silently stands for the other's
        # estimand. An artefact written before the seed-mean columns existed
        # carries neither, and the cell falls back to the union pair alone.
        seed_count = row.get("n_infosets_seed_mean")
        seed_coverage = row.get("paired_coverage_seed_mean")
        if (
            seed_count is not None
            and seed_coverage is not None
            and is_finite_number(seed_count)
            and is_finite_number(seed_coverage)
        ):
            shape = (
                f"{count} union / {seed_count:.1f} per seed paired info-sets, "
                f"coverage {coverage * 100:.1f}% union / "
                f"{seed_coverage * 100:.1f}% per seed"
            )
            # The floor is checked against the within-seed coverage that backs the
            # headline value, never the union pair above -- a row with no seed-mean
            # columns at all falls into the `else` branch and is never checked, since
            # there is nothing of the right estimand to compare.
            if floor is not None and seed_coverage < floor:
                text = (
                    f"inadmissible: {_four_places(value)} ({shape}); paired coverage "
                    f"{seed_coverage * 100:.1f}% below the pre-registered floor "
                    f"{floor * 100:.1f}% for this (game, property) cell"
                )
                return shape, Cell(INADMISSIBLE, text, text)
        else:
            shape = f"{count} paired info-sets, coverage {coverage * 100:.1f}%"
    else:
        shape = f"{count} off-path info-sets"
    return shape, None


def _probe_sampled_text(
    row: Mapping, value: float, shape: str, short: bool, with_verdict: bool = True
) -> str:
    """
    the sampled-basis text of a measured probe cell
    :param row: a measurable `probe_scores.csv` row
    :param value: the row's finite value
    :param shape: the info-set and coverage phrase from `_probe_shape`
    :param short: whether clauses are cut to the table length
    :param with_verdict: whether the verdict and its reason are part of the text
    :returns: the cell text
    """
    rate = dropped_text(row.get("dropped_rate"))
    verdict, why = row.get("verdict"), row.get("verdict_reason")
    # Built the same way `classify`'s own `valued(short)` is: each reason clause
    # goes through `_reason_text`, which caps an ordinary clause and shows a
    # `ZERO_VARIANCE_PREFIX` one in full regardless of length (`_clause`'s own
    # protected-clause rule) -- this alone keeps the row capped, since uncapped
    # `_whole` calls could push the row past `_capped`'s blind cut before
    # `_degenerate_clause` below is appended. Under a distribution basis the
    # verdict and its reason are that cell's own labelled clause, so they are left out.
    judged = f" {verdict}" if verdict and with_verdict else ""
    text = f"{_four_places(value)}{judged} ({shape}); {rate}"
    if with_verdict and verdict and why:
        text += f"; verdict reason: {_reason_text(why, short)}"
    notes = (
        _reason_text(row["caveat"], short) if split_clauses(row.get("caveat")) else ""
    )
    if notes:
        text += f"; notes: {notes}"
    # `classify_probe` computes this from `n_replicates`/`n_replicates_distinct`
    # directly, independent of `caveat`: a caller may construct a row with no
    # caveat at all and still expect the sentence. A writer that also couples
    # `n_replicates_distinct == 1` to this exact sentence in `caveat` (every
    # current one does: `_zero_variance` in `probe_scores.py`) makes `notes`
    # already carry it, so appending it again would duplicate it
    # -- skip only that one case, never the general one.
    flat = _degenerate_clause(row)
    if flat and flat not in notes:
        text += f"; {flat}"
    return text


def classify_probe(
    property_name: str,
    row: Mapping | None,
    csv_present: bool,
    floor: float | None = None,
) -> Cell:
    """
    classify one game of a probe-sourced property from its `probe_scores.csv` row
    :param property_name: the headline row name
    :param row: the row for that game (value a finite float or None, status, caveat,
        n_infosets, paired_coverage, the optional per-seed
        n_infosets_seed_mean/paired_coverage_seed_mean, and for an off-path row its verdict
        and the reason on each basis), or None when the file has none
    :param csv_present: whether the arm has a probe_scores.csv at all
    :param floor: the pre-registered minimum `paired_coverage_seed_mean` for this (game,
        property) cell, or None when this property/game has no pre-registered floor (off-path
        rows, or a cell the floor artefact does not cover). A row with no seed-mean coverage
        at all (written before those columns existed) is never checked against a floor, since
        there is nothing of the right estimand to compare.
    :returns: a number only for a MEASURABLE row with a finite value, no missing threshold,
        and (when a floor applies) within-seed coverage at or above its pre-registered floor;
        INADMISSIBLE -- the number and its coverage shown, the verdict withheld -- for a
        row whose within-seed coverage falls below its floor (the project's own
        admissibility gate blocking on a defect the pre-registration names, not a fault);
        awaiting for an AWAITING_DATA row with no value and a stated reason, and for a
        measured row whose own threshold is missing or voided, which is a fact about the
        anchor rather than about the measurement, so the number stays in the file and the
        cell says what it waits for, and awaiting when the arm has no probe_scores.csv at
        all (an input never supplied); BROKEN for everything else, including a present file
        with no row for the game, with the row's own status echoed. An off-path row carrying a distribution-basis value is headed by
        that basis, with the sampled one beside it. A probe cell is never structural: a
        property that cannot exist at a memory level is settled by the board, before any
        row is read.
    """
    if not csv_present:
        return Cell(AWAITING, PROBE_FILE_MISSING)
    if row is None:
        return Cell(BROKEN, f"BROKEN: no {property_name} probe score for this game")
    status, value = row["status"], row["value"]
    if status == MEASURABLE and is_finite_number(value):
        problem = _probe_value_problem(row, value)
        if problem:
            return Cell(BROKEN, f"BROKEN: {problem}")
        withheld = _reference_uniform_cell(row)
        if withheld is not None:
            return withheld
        shape, inadmissible = _probe_shape(row, value, floor)
        if inadmissible is not None:
            return inadmissible
        verdict, why = row.get("verdict"), row.get("verdict_reason")
        sampled_text = functools.partial(_probe_sampled_text, row, value, shape)

        exact = _distribution_basis(row)
        if exact is not None:
            return exact(sampled_text(True, False), sampled_text(False, False))
        # A measured distance with no threshold to judge it against is not a number the
        # board can publish: it would sit beside a threshold nobody measured on this game.
        # The measurement is not lost, it stays in probe_scores.csv.
        if why and not verdict:
            return Cell(
                AWAITING,
                f"awaiting data: {_clause(why)}",
                f"awaiting data: {_whole(why)}",
            )
        absent = row.get("exact_reason") or ""
        # Which basis the number is on, and why the other one is not there. Off-path rows
        # carry this, and so does a paired row whose arms have no usable exact-policy pass.
        note = f"; sampled basis, no distribution basis: {absent}" if absent else ""
        return Cell(NUMBER, sampled_text(True) + note, sampled_text(False) + note)
    if status == AWAITING_DATA and value is None and split_clauses(row["caveat"]):
        return Cell(
            AWAITING,
            f"awaiting data: {_clause(row['caveat'])}",
            f"awaiting data: {_whole(row['caveat'])}",
        )
    if status == MEASURABLE:
        return Cell(BROKEN, f"BROKEN: {NO_FINITE_VALUE}")

    def stopped(short: bool) -> str:
        reason = _reason_text(row["caveat"], short)
        shown = _clause(status)
        return f"BROKEN: {shown}: {reason}" if reason else f"BROKEN: {shown}"

    return Cell(BROKEN, stopped(True), stopped(False))


def _capped(cell: Cell, limit: int) -> Cell:
    """
    :param cell: a classified Cell
    :param limit: the longest text allowed
    :returns: the Cell with its table text cut to `limit`, marked with an ellipsis and with
        the whole text kept in `detail`; the kind and the leading form are never cut
    """
    if len(cell.text) <= limit:
        return cell
    return Cell(
        cell.kind,
        cell.text[: limit - 3].rstrip() + "...",
        cell.full_text,
        cell.legacy_blocker_rule,
    )


def _scored_part(cell: Cell, text: str) -> str:
    """
    the words for a scored part of a multi-part cell when another part is not scored
    A number part leads with its value. A partial part's own text reads
    "partial: <k/n> opponents; mean <value> ...", so its value is the first word of the
    second clause; it is shown beside how many opponents back it, as a number part's is.
    :param cell: a NUMBER, PARTIAL or PARTIAL_APPROVED part (Cell)
    :param text: that part's short or full text (str)
    :returns: "ok <value>", "partial <value> (<k/n> opponents)" or
        "partial, approved <value> (<k/n> opponents)" (str)
    """
    if cell.kind == NUMBER:
        return f"ok {text.split(' ')[0]}"
    clauses = split_clauses(text.removeprefix("partial: ")) or [""]
    backing = clauses[0]
    mean = ""
    if len(clauses) > 1 and clauses[1].startswith("mean "):
        mean = " " + clauses[1].removeprefix("mean ").split(" ")[0]
    kind = "partial, approved" if cell.kind == PARTIAL_APPROVED else "partial"
    return f"{kind}{mean} ({backing})"


def combine(labels: Sequence[str], cells: Sequence[Cell]) -> Cell:
    """
    combine the parts of one property (P7 has two headline rows)
    :param labels: a short label per part, such as "7a"
    :param cells: one Cell per part, same order
    :returns: the single part unchanged; BROKEN if any part is; awaiting if any part is and
        none is BROKEN; structural if all are; inadmissible if any part is and none is worse;
        partial if any part is and none is worse and at least one is unapproved-partial;
        partial_approved if any part is and none is worse and every partial part is approved;
        otherwise a number cell naming each part
    """
    if len(cells) == 1:
        return cells[0]
    if any(c.kind in (BROKEN, AWAITING) for c in cells):
        broken = any(c.kind == BROKEN for c in cells)

        def joined(short: bool) -> str:
            parts = []
            for label, cell in zip(labels, cells, strict=True):
                text = cell.text if short else cell.full_text
                if cell.kind == BROKEN:
                    parts.append(f"{label} {text.removeprefix('BROKEN: ')}")
                elif cell.kind == AWAITING:
                    reason = text.removeprefix("awaiting data: ")
                    parts.append(
                        f"{label} awaiting: {reason}" if broken else f"{label} {reason}"
                    )
                elif cell.kind in (NUMBER, PARTIAL, PARTIAL_APPROVED):
                    parts.append(f"{label} {_scored_part(cell, text)}")
                elif cell.kind == INADMISSIBLE:
                    parts.append(f"{label} {text}")
                else:
                    parts.append(f"{label} structural")
            prefix = "BROKEN: " if broken else "awaiting data: "
            return prefix + "; ".join(parts)

        return Cell(BROKEN if broken else AWAITING, joined(True), joined(False))
    if all(c.kind == STRUCTURAL for c in cells):
        return cells[0]
    if any(c.kind == INADMISSIBLE for c in cells):

        def withheld(short: bool) -> str:
            parts = []
            for label, cell in zip(labels, cells, strict=True):
                text = cell.text if short else cell.full_text
                if cell.kind == INADMISSIBLE:
                    parts.append(f"{label} {text.removeprefix('inadmissible: ')}")
                elif cell.kind in (NUMBER, PARTIAL, PARTIAL_APPROVED):
                    parts.append(f"{label} {_scored_part(cell, text)}")
                else:
                    parts.append(f"{label} structural")
            return "inadmissible: " + "; ".join(parts)

        return Cell(INADMISSIBLE, withheld(True), withheld(False))
    # PARTIAL_APPROVED is a third kind alongside NUMBER/PARTIAL; a check naming only
    # PARTIAL would let a PARTIAL_APPROVED part fall through to NUMBER, silently
    # dropping the "partial: " marker even though the part's own text reads
    # "k/n opponents". An unapproved PARTIAL anywhere makes the whole combined cell
    # PARTIAL (it is not fully approved); only when every partial part is approved
    # does the combination read PARTIAL_APPROVED.
    if any(c.kind == PARTIAL for c in cells):
        kind = PARTIAL
    elif any(c.kind == PARTIAL_APPROVED for c in cells):
        kind = PARTIAL_APPROVED
    else:
        kind = NUMBER
    prefix = "partial: " if kind in (PARTIAL, PARTIAL_APPROVED) else ""

    def side_by_side(texts: Sequence[str]) -> str:
        return prefix + " / ".join(
            f"{l} {t.removeprefix('partial: ')}"
            for l, t in zip(labels, texts, strict=True)
        )

    return Cell(
        kind,
        side_by_side([c.text for c in cells]),
        side_by_side([c.full_text for c in cells]),
    )


def headline_rows(decision_rule: Sequence[Sequence]) -> dict[int, list[str]]:
    """
    the project's own headline row per property, numbered from the row name prefix
    :param decision_rule: the DECISION_RULE list of (row name, estimand, threshold, direction)
    :returns: {property number: [row names]} in rule order
    :raises ValueError: when a property from 1 to 8 has no headline row
    """
    heads: dict[int, list[str]] = {}
    for entry in decision_rule:
        match = re.match(r"P(\d)", entry[0])
        if match:
            heads.setdefault(int(match.group(1)), []).append(entry[0])
    missing = [n for n in PROPERTY_NUMBERS if n not in heads]
    if missing:
        raise ValueError(f"no headline row for properties {missing}")
    return heads


def _label(row_name: str) -> str:
    """
    :param row_name: a headline row name such as P7a_belief_error
    :returns: its short label such as 7a
    """
    match = re.match(r"P(\d[a-z]?)", row_name)
    if match is None:
        raise ValueError(f"not a headline row name: {row_name!r}")
    return match.group(1)


def check_required_columns(columns: Sequence[str]) -> None:
    """
    :param columns: the column names of a properties table
    :raises ValueError: naming every required column that is absent; extra columns are fine
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f"properties table lacks required column(s): {', '.join(missing)}"
        )


def _sanitised(rows: Sequence[Mapping]) -> list[dict]:
    """
    copy the rows in a canonical order with non-finite values turned into missing values
    :param rows: properties rows
    :returns: new rows sorted by (game, property, opponent, player); a non-finite value
        becomes None and its caveat's first clause records it, so it can never enter a mean
    """
    clean = []
    for row in rows:
        row = dict(row)
        value = row["value"]
        if value is not None and not is_finite_number(value):
            row["value"] = None
            head, sep, tail = (row["caveat"] or "").partition(";")
            head = head.strip()
            marked = f"{head} (non-finite value)" if head else "non-finite value"
            row["caveat"] = marked + (sep + tail if sep else "")
        clean.append(row)
    clean.sort(
        key=lambda r: (
            str(r["game"]),
            str(r["property"]),
            str(r["opponent"]),
            str(r["player"]),
        )
    )
    return clean


#: The words a cell carries when its arm chose the first-listed legal action on every row of the
#: game and no other arm in the root shares its policy, so a fixed rule is not mistaken for a
#: choice.
LONE_FIRST_LISTED_WORDING = "; constant policy: one action at every info-set"


def collapse_marker(entry: Mapping | None) -> str:
    """
    the words a cell of a collapsed (arm, game) must carry
    :param entry: that arm's collapse row for the game, or None when the arm was not checked
        or the game was not measured
    :returns: the marker naming the other arms playing the same action policy (the
        fingerprint covers actions and not elicited beliefs, so two arms can share it and
        still differ on the belief properties) and how often this arm chose the first-listed
        legal action; `LONE_FIRST_LISTED_WORDING` when no other arm shares the policy but this
        arm chose the first-listed legal action on every row; an empty string otherwise
    """
    if not entry:
        return ""
    others = str(entry.get("collapsed_with") or "").strip()
    share = entry.get("first_listed_share")
    if not others:
        return (
            LONE_FIRST_LISTED_WORDING
            if is_finite_number(share) and share >= 1.0
            else ""
        )
    reading = (
        f"first-listed legal action {share * 100:.2f}% of rows"
        if is_finite_number(share)
        else "first-listed share not recorded"
    )
    named = ", ".join(sorted(o for o in others.split(",") if o))
    return f"; same action policy as {named} ({reading})"


#: The wording of a P6 cell whose driven policy is one action and no other arm plays it.
LONE_CONSTANT_WORDING = (
    "; constant driven policy: one action at every off-path info-set"
)
#: The property whose cell reads the arm's driven off-path play, so the collapse that matters for
#: it is the off-path collapse and not the on-path one.
P6_NUMBER = 6
#: The kinds of cell that carry a measurement, and so a collapse marker.
MEASURED_KINDS = (NUMBER, PARTIAL, PARTIAL_APPROVED, INADMISSIBLE)


def offpath_collapse_marker(table: Mapping | None, game: str) -> str:
    """
    the words a P6 cell must carry about its driven off-path play
    :param table: {game: that arm's collapse_offpath row}, or None when the arm has no such file
    :param game: the cell's game
    :returns: `OFFPATH_COLLAPSE_NOT_CHECKED` when there is no file or the file has no row for the
        game, so an unchecked cell never reads as clean; the arms sharing the driven policy, with
        the constant action when it is one; `LONE_CONSTANT_WORDING` when the policy is one action
        no other arm plays; an empty string when the arm was checked and plays more than one action
        nobody else plays
    """
    entry = (table or {}).get(game)
    if not entry:
        return OFFPATH_COLLAPSE_NOT_CHECKED
    others = str(entry.get("collapsed_with") or "").strip()
    action = str(entry.get("constant_action") or "").strip()
    if not others:
        return LONE_CONSTANT_WORDING if action else ""
    named = ", ".join(sorted(o for o in others.split(",") if o))
    reading = f" (constant action {action})" if action else ""
    return f"; same driven off-path policy as {named}{reading}"


#: The property whose second leg is not built: P4 is payoff-sensitivity, an invariance leg and a
#: genuine-structural leg, and only the invariance leg is scored.
P4_NUMBER = 4
#: The words a P4 cell that reports a measurement carries, so a reader never takes it for the
#: whole property.
P4_LEG_LABEL = "; invariance leg only"


def _legged(cell: Cell, number: int) -> Cell:
    """
    :param cell: a classified Cell, already capped
    :param number: the property number of the cell
    :returns: for a P4 cell that reports a measurement (a number or an inadmissible value), the
        Cell with the leg label appended to both texts, keeping its kind; the label is added
        after capping so it can never be the part that is cut; any other cell unchanged
    """
    if number != P4_NUMBER or cell.kind not in (NUMBER, INADMISSIBLE):
        return cell
    return Cell(
        cell.kind,
        cell.text + P4_LEG_LABEL,
        cell.full_text + P4_LEG_LABEL,
        cell.legacy_blocker_rule,
    )


def _marked(cell: Cell, entry: Mapping | None) -> Cell:
    """
    :param cell: a classified Cell, already capped
    :param entry: that (arm, game)'s collapse row, or None
    :returns: the Cell with the collapse marker appended to both texts, keeping its kind;
        the marker is added after capping so it can never be the part that is cut
    """
    marker = collapse_marker(entry)
    if not marker:
        return cell
    return Cell(cell.kind, cell.text + marker, cell.full_text + marker)


def _marked_offpath(cell: Cell, table: Mapping | None, game: str) -> Cell:
    """
    :param cell: a classified P6 Cell, already capped
    :param table: {game: the arm's off-path collapse row}, or None when it has no such file
    :param game: the cell's game
    :returns: the Cell with the off-path collapse marker appended to both texts, in place of the
        on-path marker, which describes a different run; added after capping so it is never cut
    """
    marker = offpath_collapse_marker(table, game)
    if not marker:
        return cell
    return Cell(cell.kind, cell.text + marker, cell.full_text + marker)


def _probe_cell(
    name: str,
    found: Sequence[Mapping],
    csv_present: bool,
    floor: float | None = None,
) -> Cell:
    """
    :param name: the headline row name
    :param found: every probe_scores.csv row for that (property, game)
    :param csv_present: whether the arm has a probe_scores.csv
    :param floor: the pre-registered coverage floor for this (game, property) cell, or None
    :returns: the Cell; two rows for one (property, game) read BROKEN rather than picking one
    """
    if len(found) > 1:
        return Cell(BROKEN, f"BROKEN: duplicate {name} probe rows for this game")
    return classify_probe(name, found[0] if found else None, csv_present, floor)


def build_cells(
    arm: str,
    properties_rows: Sequence[Mapping],
    decision_rule: Sequence[Sequence],
    decision_table_fn: Callable,
    structural_by_design_fn: Callable[[str, int, str | None], bool],
    expected_opponents: Sequence[str],
    probe_names: Sequence[str] = (),
    probe_rows: Sequence[Mapping] | None = None,
    memory_rung: str | None = None,
    coverage_floors: Mapping[str, Mapping[str, Mapping]] | None = None,
) -> list[GridCell]:
    """
    the 40 cells of one arm
    :param arm: the arm (model) name
    :param properties_rows: every properties.csv row of the arm, as dicts
    :param decision_rule: DECISION_RULE, giving the headline row per property
    :param decision_table_fn: the project's `decision_table`, applied per game
    :param structural_by_design_fn: (game, property number, memory level) -> whether the
        property cannot exist for that game's family at that memory level
    :param expected_opponents: the opponents each cell must cover (the project's ladder)
    :param probe_names: the headline rows read from `probe_scores.csv` and not from the
        properties rows
    :param probe_rows: the arm's probe_scores.csv rows, or None when the arm has no such file
    :param memory_rung: the memory level this arm was collected at, passed to the by-design
        predicate; None counts as M0, the level a run with no recorded rung was collected at
    :param coverage_floors: the pre-registered `paired_coverage` floors, as
        `{game: {property: {"floor": float, ...}}}` (the floor
        artefact's shape), or None when no floor artefact was supplied -- every probe cell is
        then classified with no floor check at all.
    :returns: one GridCell per game and property, games in board order, properties 1 to 8
    :raises ValueError: when a headline row names a game the board does not list
    """
    heads = headline_rows(decision_rule)
    head_names = {name for names in heads.values() for name in names}
    rows = [r for r in _sanitised(properties_rows) if r["property"] in head_names]
    unknown = sorted({str(r["game"]) for r in rows} - set(SCOREBOARD_GAMES))
    if unknown:
        raise ValueError(f"headline rows for games the board does not list: {unknown}")
    probe_lookup: dict[tuple[str, str], list[Mapping]] = {}
    for probe_row in probe_rows or ():
        probe_lookup.setdefault((probe_row["property"], probe_row["game"]), []).append(
            probe_row
        )
    cells = []
    by_wording = 0
    for game in SCOREBOARD_GAMES:
        game_rows = [r for r in rows if r["game"] == game]
        decisions = {d["property"]: d for d in decision_table_fn(game_rows)}
        for number in PROPERTY_NUMBERS:
            limit = MAX_PART_CHARS if len(heads[number]) > 1 else MAX_CELL_CHARS
            parts = [
                _legged(
                    _capped(
                        (
                            _probe_cell(
                                name,
                                probe_lookup.get((name, game), []),
                                probe_rows is not None,
                                (coverage_floors or {})
                                .get(game, {})
                                .get(name, {})
                                .get("floor"),
                            )
                            if name in probe_names
                            else classify(
                                name,
                                decisions.get(name),
                                [r for r in game_rows if r["property"] == name],
                                structural_by_design_fn(game, number, memory_rung),
                                expected_opponents,
                            )
                        ),
                        limit,
                    ),
                    number,
                )
                for name in heads[number]
            ]
            by_wording += sum(part.legacy_blocker_rule for part in parts)
            cells.append(
                GridCell(
                    arm,
                    game,
                    number,
                    combine([_label(n) for n in heads[number]], parts),
                )
            )
    if by_wording:
        print(
            f"scoreboard: {arm}: {by_wording} cell(s) classified inadmissible by the legacy "
            f"blocker-wording rule, because this arm's properties.csv predates "
            f"`partial_reason_class`; re-score the arm to retire this path",
            file=sys.stderr,
        )
    return cells


@dataclasses.dataclass(frozen=True)
class ArmInput:
    """
    one scored condition of one model, as the scoreboard reads it
    :param name: the arm directory name
    :param condition: the condition block the arm's run_meta records, or None when it
        records none
    :param cells: that arm's own 40 cells, keyed by (game, property number)
    :param collapse: {game: that arm's collapse row}, or None when it was never checked
    :param collapse_offpath: {game: that arm's off-path collapse row}, or None when it has none
    """

    name: str
    condition: Mapping | None
    cells: Mapping
    collapse: Mapping | None = None
    collapse_offpath: Mapping | None = None


NO_CONDITION = (
    "BROKEN: analysis output does not record its condition, so the board cannot tell which "
    "run this is"
)


@dataclasses.dataclass(frozen=True)
class Board:
    """
    one memory level's whole grid of cells
    :param level: the memory level every cell on it reads
    :param cells: its GridCells, models in board order, then games, then properties
    """

    level: str
    cells: tuple[GridCell, ...]


def build_level_cells(
    model: str,
    level: str,
    arms: Sequence[ArmInput],
    headline_of: Callable[[int, Mapping, str], Sequence[str]],
    structural_of: Callable[[int, str | None], str],
    condition_of: Callable[[int, str], str],
) -> list[GridCell]:
    """
    the 40 cells of one model on one memory level's board
    :param model: the model name every arm here belongs to
    :param arms: that model's scored conditions, at every level it was collected at
    :param headline_of: (property number, {arm name: condition}, level) -> the arm names
        whose condition the property reads on this board
    :param structural_of: (property number, memory level) -> the reason the property cannot
        exist at that level, or an empty string
    :param condition_of: (property number, memory level) -> the run the cell reads, in words
    :returns: one GridCell per game and property, games in board order, properties 1 to 8;
        an arm with no recorded condition makes every cell BROKEN, since the board cannot
        tell which run it is and guessing a memory level would be the silent fallback this
        block exists to remove
    """
    unrecorded = [a.name for a in arms if a.condition is None]
    if unrecorded:
        text = f"{NO_CONDITION} ({', '.join(sorted(unrecorded))})"
        return [
            GridCell(model, game, number, Cell(BROKEN, text))
            for game in SCOREBOARD_GAMES
            for number in PROPERTY_NUMBERS
        ]
    conditions = {a.name: a.condition for a in arms}
    by_name = {a.name: a for a in arms}
    cells = []
    for game in SCOREBOARD_GAMES:
        for number in PROPERTY_NUMBERS:
            reason = structural_of(number, level)
            if reason:
                cells.append(
                    GridCell(
                        model, game, number, Cell(STRUCTURAL, f"structural: {reason}")
                    )
                )
                continue
            chosen = list(headline_of(number, conditions, level))
            collapse, offpath = None, None
            if len(chosen) > 1:
                head = Cell(
                    BROKEN,
                    "BROKEN: "
                    + (
                        f"{len(chosen)} arms match the condition this property reads "
                        f"({condition_of(number, level)}): {', '.join(chosen)}"
                    ),
                )
            elif chosen:
                head = by_name[chosen[0]].cells[(game, number)]
                collapse = by_name[chosen[0]].collapse
                offpath = by_name[chosen[0]].collapse_offpath
            else:
                head = _absent_headline(
                    number, game, level, arms, structural_of, condition_of
                )
            if number == P6_NUMBER and head.kind in MEASURED_KINDS:
                marked = _marked_offpath(head, offpath, game)
            else:
                marked = _marked(head, (collapse or {}).get(game))
            cells.append(GridCell(model, game, number, marked))
    return cells


def _absent_headline(number, game, level, arms, structural_of, condition_of) -> Cell:
    """
    the cell of a property this board's level has no arm to read
    :param number: the property number 1 to 8
    :param game: the game key
    :param level: the memory level of the board
    :param arms: every arm of the model, at whatever level, read only to see whether the
        property is structurally absent for this game at all
    :param structural_of: (property number, memory level) -> the reason the property cannot
        exist at that level, or an empty string
    :param condition_of: (property number, memory level) -> the run the cell reads, in words
    :returns: the structural cell another level of the same model already found for this
        game, when that absence is a fact about the game and not about the memory level;
        awaiting data naming the run this board waits for otherwise
    """
    for arm in arms:
        cell = arm.cells[(game, number)]
        if (
            cell.kind == STRUCTURAL
            and not cell.text.startswith(NOT_REACHED_PREFIX)
            and not structural_of(number, arm.condition.get("memory_rung"))
        ):
            return cell
    return Cell(
        AWAITING, f"awaiting data: this model has no {condition_of(number, level)}"
    )


def short_data_dir(data_dir: str) -> str:
    """
    :param data_dir: a data directory from a run_meta, relative or absolute
    :returns: its last two path components, so a host path never reaches the block
    """
    return "/".join(Path(data_dir).parts[-2:])


def arm_identity(model_id: str | None, memory_rung: str | None, role: str) -> str:
    """
    the identity a `probe_scores.csv` row binds on, in place of a directory path

    A path-derived key collides whenever two arms' directories happen to share their last
    components: `short_data_dir` distinguishes recycle-v7-probes/relabel/i0/qwen from
    recycle-v6-m1/composite-qwen, but not from recycle-v7-probes/payoff_scale/i0/qwen, since
    both truncate to `i0/qwen...`. This identity is the arm's own recorded meaning instead,
    so it needs no path depth to be right, and no kind is named here by hand: a role this
    function has never seen still gets its own key, distinct from every other role's.
    :param model_id: the recorded model identity, or None
    :param memory_rung: the recorded memory rung; None or empty means M0
    :param role: the arm's recorded role (base, relabel, payoff_scale, offpath_drive, or any
        role added later)
    :returns: a stable identity string, independent of which host or directory layout
        produced the arm
    """
    return f"{role}:{model_id}:{memory_rung or 'M0'}"


def file_sha256_short(path: str | Path) -> str:
    """
    a short content hash that survives a checkout: git stores these CSVs with LF endings
    while the scorer wrote CRLF, so the hash is taken over the LF-normalised bytes
    :param path: a file
    :returns: the first 12 hex characters of the sha256 of its bytes with CRLF read as LF
    """
    return file_digest(path)[:12]


def _collapse_note(arms: Sequence[ArmSummary]) -> str:
    """
    what the header says about arms that play one policy
    :param arms: provenance per arm
    :returns: the count of collapsed (arm, game) pairs and the arms never checked, which is
        never reported as an absence of collapse
    """
    checked = [a.collapsed_games for a in arms if a.collapsed_games is not None]
    pairs = sum(len(games) for games in checked)
    unchecked = [a.name for a in arms if a.collapsed_games is None]
    note = (
        f"Collapse: {pairs} (arm, game) pair(s) play a policy another arm also plays, over "
        f"{len(checked)} arm(s) checked."
    )
    if unchecked:
        note += (
            f" {COLLAPSE_NOT_CHECKED} for {', '.join(unchecked)}: no {COLLAPSE_FILE}, which "
            "is not the same as checked and found distinct."
        )
    probed = [a for a in arms if a.probe_sha256 is not None]
    off_checked = [
        a.offpath_collapsed_games
        for a in probed
        if a.offpath_collapsed_games is not None
    ]
    off_unchecked = [a.name for a in probed if a.offpath_collapsed_games is None]
    if probed:
        pairs = sum(len(games) for games in off_checked)
        note += (
            f" Off-path collapse (P6): {pairs} (arm, game) pair(s) play a driven policy another "
            f"arm also plays, over {len(off_checked)} arm(s) checked."
        )
    if off_unchecked:
        note += (
            f" off-path collapse not checked for {', '.join(off_unchecked)}: no "
            f"{COLLAPSE_OFFPATH_FILE}, which is not the same as checked and found distinct."
        )
    return note


def _kind_counts(cells: Sequence[GridCell]) -> dict[str, int]:
    """
    :param cells: the cells to count
    :returns: {kind: count} over every kind the header counts
    :raises ValueError: when a cell carries a kind KINDS does not count, which would let it
        vanish from every total
    """
    uncounted = sorted({c.cell.kind for c in cells} - set(KINDS))
    if uncounted:
        raise ValueError(
            f"cells carry kind(s) the header does not count: {uncounted}; a kind left out "
            "of KINDS would vanish from every total"
        )
    return {k: sum(1 for c in cells if c.cell.kind == k) for k in KINDS}


def _tally(counts: Mapping[str, int]) -> str:
    """
    :param counts: {kind: count}
    :returns: the kinds in the order the header counts them, in words
    """
    return (
        f"{counts[NUMBER]} numbers, {counts[PARTIAL]} partial, "
        f"{counts[PARTIAL_APPROVED]} partial (approved), {counts[STRUCTURAL]} "
        f"structural, {counts[AWAITING]} awaiting data, {counts[INADMISSIBLE]} inadmissible, "
        f"{counts[BROKEN]} BROKEN"
    )


def render_block(
    arms: Sequence[ArmSummary],
    boards: Sequence[Board],
    decision_rule: Sequence[Sequence],
    direction_text: Mapping[str, str],
    footnotes: Sequence[str] = (),
    *,
    expected_cells: int | None = None,
    missing_models: Sequence[str] = (),
    partial_arms: tuple[int, Sequence[str]] | None = None,
    epoch_lines: Sequence[str] = (),
) -> str:
    """
    the text that goes between the markers
    :param arms: provenance per arm, in board order
    :param boards: one Board per memory level, in the order they are printed
    :param decision_rule: DECISION_RULE, for the legend
    :param direction_text: direction code -> words, for the legend
    :param footnotes: lines appended verbatim under the tables
    :param expected_cells: the cell count a full roster would produce
        (`scripts/scoreboard.expected_models`), stated beside the actual count so a board
        missing a whole model cannot read as complete; omitted (None) states the actual
        count alone
    :param missing_models: models the roster expects that hold zero arms anywhere in the
        root, named in the header rather than only being absent from the row list below
    :param partial_arms: (the roster's arm count, the arms the root lacks) when the board is a
        partial one the caller was told to render, else None; a line `PARTIAL: N of M arms`
        names the missing ones
    :param epoch_lines: header lines placed after the first, one per collection epoch when
        the caller asks for a per-epoch count; empty adds none
    :returns: a deterministic markdown block with no timestamps or host paths; the arm
        provenance, the legend, the collapse note, the detail section and the footnotes are
        written once, since they say the same thing about every board
    :raises ValueError: when any cell carries a kind the header does not count
    """
    every = [c for board in boards for c in board.cells]
    overall = _kind_counts(every)
    levels = ", ".join(board.level for board in boards)
    cell_phrase = (
        f"{len(every)} cells in all"
        if expected_cells is None
        else f"{len(every)} of {expected_cells} expected cells in all"
    )
    lines = [
        f"Generated by `scripts/scoreboard.py`; do not edit by hand. One board per memory "
        f"level ({levels}); {cell_phrase} over {len(arms)} arms: "
        f"{_tally(overall)}.",
    ]
    if partial_arms is not None:
        expected, absent = partial_arms
        lines.append(
            f"PARTIAL: {expected - len(absent)} of {expected} arms; missing: "
            + ", ".join(absent)
            + "."
        )
    lines += list(epoch_lines)
    if missing_models:
        lines.append(
            "Missing entirely from this root (zero arms at any level): "
            + ", ".join(missing_models)
            + "."
        )
    lines += [
        "",
        "One arm directory is one condition of one model. A board reads its own memory "
        "level and nothing else: a model with no arm at that level reads awaiting data "
        "there, naming the run it waits for.",
        "",
        "| arm | memory level | data | properties.csv sha256 (LF-normalised, first 12) | "
        "probe_scores.csv sha256 (LF-normalised, first 12) |",
        "|---|---|---|---|---|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm.name} | {arm.memory_rung or 'not recorded'} | {arm.data} | {arm.sha256} | "
            f"{arm.probe_sha256 or 'none'} |"
        )
    lines += ["", "| property | headline row | fails when |", "|---|---|---|"]
    heads = headline_rows(decision_rule)
    rule = {entry[0]: entry for entry in decision_rule}
    for number in PROPERTY_NUMBERS:
        for name in heads[number]:
            _n, _e, threshold, direction = rule[name]
            words = direction_text.get(direction, direction)
            shown = words if threshold is None else f"{words} ({threshold})"
            lines.append(f"| P{number} | {name} | {shown} |")
    lines += ["", PROBE_NOTE, "", _collapse_note(arms)]
    for board in boards:
        counts = _kind_counts(board.cells)
        lines += [
            "",
            f"#### Memory level {board.level}",
            "",
            f"{len(board.cells)} cells: {_tally(counts)}.",
            "",
            "| model | numbers | partial | partial (approved) | structural | awaiting | "
            "inadmissible | BROKEN |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model in sorted({c.arm for c in board.cells}):
            lines.append(
                f"| {model} | "
                + " | ".join(
                    str(
                        sum(
                            1
                            for c in board.cells
                            if c.arm == model and c.cell.kind == k
                        )
                    )
                    for k in KINDS
                )
                + " |"
            )
        lines += [
            "",
            "| level | model | game | property | cell |",
            "|---|---|---|---|---|",
        ]
        for c in board.cells:
            lines.append(
                f"| {board.level} | {c.arm} | {c.game} | P{c.number} | {c.cell.text} |"
            )
    shortened = [
        (board.level, c) for board in boards for c in board.cells if c.cell.shortened
    ]
    lines += ["", DETAIL_NOTE.format(n=len(shortened))]
    if shortened:
        lines += ["", "| cell | whole text |", "|---|---|"]
        for level, c in shortened:
            lines.append(
                f"| {level} {c.arm} {c.game} P{c.number} | {c.cell.full_text} |"
            )
    if footnotes:
        lines += [""] + list(footnotes)
    return "\n".join(lines)


def replace_scoreboard_block(text: str, block: str) -> str:
    """
    replace only the text between the scoreboard markers
    :param text: the whole file text
    :param block: the new block, without markers
    :returns: the file text with the block replaced; every other byte unchanged
    :raises ValueError: when the markers are missing, duplicated or out of order, or when the
        block itself contains a marker (which would corrupt every later run)
    """
    if BEGIN_MARKER in block or END_MARKER in block:
        raise ValueError("the block contains a scoreboard marker")
    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError(
            f"expected exactly one {BEGIN_MARKER} and one {END_MARKER}; found "
            f"{text.count(BEGIN_MARKER)} and {text.count(END_MARKER)}"
        )
    begin = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER)
    if end < begin:
        raise ValueError("the scoreboard markers are out of order")
    return text[:begin] + "\n" + block.rstrip("\n") + "\n" + text[end:]
