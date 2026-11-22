"""
Tests for the scoreboard: the pure classifier and renderer in
``lmopt.analysis.scoreboard`` and the CLI in ``scripts/scoreboard.py``.

The fixtures under ``tests/fixtures/scoreboard/<model>/`` are trimmed slices of the real
scoring output for the three v6 memory-level-1 arms: ``properties.csv`` keeps only the
``DECISION_RULE`` headline rows (every player), while ``decision.csv`` and ``outcomes.csv``
are copied verbatim and ``run_meta.json`` keeps only ``data_dir``. They were built from
``analysis/recycle-v6-m1/<model>/`` by filtering ``properties.csv`` to
``scripts.analyse_stats.DECISION_RULE`` names.
"""

import copy
import inspect
import json
import hashlib
import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import scripts.analyse_stats as analyse_stats
import scripts.arm_collapse_pass as collapse_cli
import scripts.backfill_condition as backfill_cli
from lmopt.analysis import arm_reading
import scripts.scoreboard as cli
from lmopt.analysis import arm_collapse
from lmopt.analysis import probe_scores as ps
from lmopt.analysis.scoreboard import scoreboard as sb
from lmopt.analysis import scoring_vocabulary as vocabulary
from lmopt.analysis.cross_arm import OPPONENT_LADDER
from lmopt.games.families import GAME_FAMILIES

FIXTURES = Path(__file__).parent / "fixtures" / "scoreboard"
MODELS = (
    "mistral-nemo-instruct-2407",
    "olmo-2-1124-13b-instruct",
    "qwen2.5-14b-instruct",
)
#: The model each fixture arm directory records in its condition block. An arm directory is
#: one condition of one model, so the board is keyed by model and never by directory.
MODEL_OF = {
    "mistral-nemo-instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407",
    "olmo-2-1124-13b-instruct": "allenai/OLMo-2-1124-13B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
}
QWEN, OLMO, MISTRAL = (
    MODEL_OF["qwen2.5-14b-instruct"],
    MODEL_OF["olmo-2-1124-13b-instruct"],
    MODEL_OF["mistral-nemo-instruct-2407"],
)
#: The memory level the fixture arms record, and so the only board of the three that has a
#: run to read. The others say what they are waiting for.
FIXTURE_LEVEL = "M1"
#: (numbers, partial, structural, awaiting, inadmissible, BROKEN) per model per board over the real v6
#: slices. Each model has one arm and it is at memory level M1.
#: On the M0 board nothing was collected, so 5 cells are P2 (structural, nothing carries
#: between hands at M0), 4 are P8 off Dark Hex (a structural absence is a fact about the
#: game, so it holds on a board with no arm) and the other 31 wait for an M0 run. P1 and P4
#: are M0 properties, so on this board they wait rather than reading structural.
#: On the M1 board the arm is read: the 15 P1, P4 and P6 cells wait for a probe run, because
#: the fixture has no probe_scores.csv and those properties are defined at every level. P2 is
#: BROKEN for want of P2 rows the scoring run never produced, and P3, P5, P7 and P8 give the
#: numbers. Mistral's two inadmissible cells are Dark Hex cells blocked by the S7
#: pre-registered cell, a number shown with its verdict withheld.
#: On the M2 board nothing was collected and P2 is not structural there, so the 4 P8 cells
#: are structural and the other 36 wait, the 15 P1, P4 and P6 cells among them.
#: Each tuple has 7 slots: the fixture root's properties.csv rows have no
#: `partial_reason_class` column, so the approved-partial slot (right after `partial`,
#: matching `KINDS`' order) is 0 -- `classify()` reads the missing column as `None` via
#: `.get`, never approving a partial cell on this fixture.
TALLIES = {
    "M0": {
        QWEN: (0, 0, 0, 9, 31, 0, 0),
        MISTRAL: (0, 0, 0, 9, 31, 0, 0),
        OLMO: (0, 0, 0, 9, 31, 0, 0),
    },
    "M1": {
        QWEN: (7, 2, 0, 4, 15, 0, 12),
        MISTRAL: (5, 2, 0, 4, 15, 2, 12),
        OLMO: (8, 0, 0, 4, 15, 0, 13),
    },
    "M2": {
        QWEN: (0, 0, 0, 4, 36, 0, 0),
        MISTRAL: (0, 0, 0, 4, 36, 0, 0),
        OLMO: (0, 0, 0, 4, 36, 0, 0),
    },
}
BOARD_FILE = (
    "# tracker\n\nbefore the block\n"
    "<!-- SCOREBOARD:BEGIN -->\n_not yet generated_\n<!-- SCOREBOARD:END -->\n"
    "after the block\n"
)
ROW = re.compile(r"^\| (M\d) \| ([^|]+) \| ([^|]+) \| (P[1-8]) \| (.+) \|$", re.M)
NUMBER_FORM = re.compile(r"^(7a )?-?\d+\.\d{4}[ \[]")


def _row(
    game, opponent, prop, value, status, caveat="", player="LM", partial_reason_class=""
):
    """
    one properties.csv row as a dict, in the real schema
    :param game: game key
    :param opponent: opponent name
    :param prop: property row name
    :param value: float or None
    :param status: status string
    :param caveat: caveat text
    :param player: the player the row scores
    :param partial_reason_class: the machine-readable
        approved-reason class ("structural"/"thin"/"failed"/"not_reached"), or "" when this row carries
        none
    :returns: a dict with the required columns
    """
    return {
        "game": game,
        "opponent": opponent,
        "player": player,
        "property": prop,
        "value": value,
        "status": status,
        "caveat": caveat,
        "partial_reason_class": partial_reason_class,
    }


def _classify(rows, name="P3_randomisation_margin", structural=False):
    """
    classify one headline row through the real decision_table partition
    :param rows: properties rows of a single game
    :param name: the headline row name
    :param structural: whether the cell is structurally inapplicable by design
    :returns: the Cell
    """
    decision = {d["property"]: d for d in analyse_stats.decision_table(rows)}
    return sb.classify(
        name,
        decision.get(name),
        [r for r in rows if r["property"] == name],
        structural,
        OPPONENT_LADDER,
    )


def _opponents(values, status="MEASURABLE", caveat=""):
    """
    LM rows for one game across up to four opponents
    :param values: list of values, one per opponent in a fixed opponent order
    :param status: the status every row carries
    :param caveat: the caveat every row carries
    :returns: list of row dicts for property P3_randomisation_margin in game kuhn
    """
    names = OPPONENT_LADDER
    return [
        _row("kuhn", names[i], "P3_randomisation_margin", v, status, caveat)
        for i, v in enumerate(values)
    ]


def _copy_root(tmp_path, rung=None):
    """
    a writable copy of the fixture root
    :param tmp_path: pytest tmp_path
    :param rung: a memory level to record on every arm, or None to keep the recorded one;
        the fixtures are M1 runs, and a test about how a cell is classified needs them at
        the level whose condition the properties actually read
    :returns: the copied root
    """
    root = tmp_path / "root"
    shutil.copytree(FIXTURES, root)
    if rung is not None:
        for arm in MODEL_OF:
            _set_rung(root, arm, rung)
    return root


def _synthetic_collection_run_metas(root, tmp_path, monkeypatch):
    """
    give every fixture arm a collection run_meta to be backfilled from, built from the arm's own
    recorded condition, under a throwaway repo root the backfill script is pointed at. The
    fixtures name collection directories of arms that are no longer on disk; the backfill reads
    only the fields `condition_block` copies, so a meta carrying those reproduces what it reads
    :param root: the copied analysis root
    :param tmp_path: pytest tmp_path
    :param monkeypatch: pytest monkeypatch
    :returns: None
    """
    repo = tmp_path / "repo"
    for arm in MODEL_OF:
        meta = json.loads((root / arm / "run_meta.json").read_text())
        block = meta["condition"]
        collection = repo / meta["data_dir"]
        collection.mkdir(parents=True)
        (collection / "run_meta.json").write_text(
            json.dumps(
                {
                    "model_id": block["model_id"],
                    "n_hands": block["n_hands"],
                    "composed_from": ["piece"] * block["n_pieces"],
                    "resolved_config": {
                        "memory_rung": block["memory_rung"],
                        "n_seeds": block["n_seeds"],
                    },
                }
            )
        )
    monkeypatch.setattr(backfill_cli, "REPO_ROOT", repo)


def _set_rung(root, arm, rung):
    """
    rewrite one arm's recorded memory level, so a fixture can stand in for another condition
    :param root: the analysis root
    :param arm: the arm directory name
    :param rung: the memory level to record
    :returns: None
    """
    path = Path(root) / arm / "run_meta.json"
    meta = json.loads(path.read_text())
    meta["condition"] = dict(meta["condition"], memory_rung=rung)
    path.write_text(json.dumps(meta))


def _edit(root, arm, fn):
    """
    rewrite one arm's properties.csv through a polars transform
    :param root: the analysis root
    :param arm: the arm directory name
    :param fn: function DataFrame -> DataFrame
    :returns: None
    """
    path = Path(root) / arm / "properties.csv"
    df = pl.read_csv(path, infer_schema_length=10000)
    fn(df).write_csv(path)


#: A fixed test roster with exactly the three fixture models, so a test's expected
#: cell counts do not depend on configs/run_experiment.yaml's current (production, and
#: independently changing) content. `_run` passes this by default; a test that wants to
#: exercise --roster-config itself overrides it via `extra`.
ROSTER_CONFIG = FIXTURES / "roster.yaml"


def _run(tmp_path, root=FIXTURES, extra=(), board_file_text=BOARD_FILE, partial=True):
    """
    run the real CLI entrypoint against a temp board file
    :param tmp_path: pytest tmp_path
    :param root: analysis root
    :param extra: extra argv
    :param board_file_text: contents of the board file
    :param partial: whether to pass `--partial`; the fixture roots hold arms of one memory
        level only, so they lack arms of the roster and need it (bool)
    :returns: (exit code, board file text)
    """
    board_file = tmp_path / "board_file.md"
    board_file.write_text(board_file_text)
    code = cli.main(
        [
            "--analysis-root",
            str(root),
            "--board-file",
            str(board_file),
            "--expect-arms",
            "3",
            "--roster-config",
            str(ROSTER_CONFIG),
            *(["--partial"] if partial else []),
            *extra,
        ]
    )
    return code, board_file.read_text()


def _all_cells(text):
    """
    parse every board's table out of a rendered file
    :param text: file text
    :returns: dict {(level, model, game, prop): cell text}
    """
    return {
        (m.group(1), m.group(2), m.group(3), m.group(4)): m.group(5)
        for m in ROW.finditer(text)
    }


def _cells(text, level=FIXTURE_LEVEL):
    """
    parse one board's table out of a rendered file
    :param text: file text
    :param level: the memory level of the board wanted
    :returns: dict {(model, game, prop): cell text} for that board alone
    """
    return {key[1:]: cell for key, cell in _all_cells(text).items() if key[0] == level}


def _kind(cell):
    """
    the form a cell text reads as
    :param cell: cell text
    :returns: "number", "partial", "structural", "awaiting", "inadmissible" or "BROKEN"
    """
    if cell.startswith("partial: "):
        return "partial"
    if NUMBER_FORM.match(cell):
        return "number"
    if cell.startswith("structural"):
        return "structural"
    if cell.startswith("inadmissible: "):
        return "inadmissible"
    if cell.startswith("awaiting data: "):
        return "awaiting"
    assert cell.startswith("BROKEN: "), cell
    return "BROKEN"


class TestClassify:
    def test_admissible_mean_over_part_of_the_ladder_is_partial_not_a_number(self):
        """
        Two admissible cells (0.2, 0.4) and two blocked cells (0.9): the mean 0.3 is shown
        over 2 of 4 opponents, both blocked opponents are named, and the cell is partial.
        """
        rows = _opponents([0.2, 0.9, 0.9, 0.4])
        rows[1]["status"] = rows[2]["status"] = "MEASURABLE_FLAWED"
        rows[1]["caveat"] = rows[2]["caveat"] = "defect x"
        cell = _classify(rows)
        assert cell.kind == sb.PARTIAL
        assert cell.text.startswith("partial: 2/4 opponents; mean 0.3000")
        assert (
            "MEASURABLE_FLAWED" in cell.text
            and "cfr_ref" in cell.text
            and "rl" not in (cell.text.split("not scored:")[0])
        )

    def test_partial_approved_when_every_not_scored_opponent_carries_an_approved_reason(
        self,
    ):
        """Every opponent has a row, two are
        admissible and two carry an approved `partial_reason_class`, so the cell reads
        `PARTIAL_APPROVED` rather than plain `PARTIAL` -- the rendered text is unchanged.
        """
        rows = _opponents([0.2, 0.9, 0.9, 0.4])
        rows[1]["status"] = rows[2]["status"] = "MEASURABLE_FLAWED"
        rows[1]["caveat"] = rows[2]["caveat"] = (
            "no within-seat variation against this opponent"
        )
        rows[1]["partial_reason_class"] = "structural"
        rows[2]["partial_reason_class"] = "thin"
        cell = _classify(rows)
        assert cell.kind == sb.PARTIAL_APPROVED
        assert cell.text.startswith("partial: 2/4 opponents; mean 0.3000")

    def test_partial_stays_partial_when_one_not_scored_opponent_is_unclassified(self):
        """The mixed case: one not-scored opponent is approved, the other is not -- the
        cell must not read `PARTIAL_APPROVED` on the strength of the one that is.
        """
        rows = _opponents([0.2, 0.9, 0.9, 0.4])
        rows[1]["status"] = rows[2]["status"] = "MEASURABLE_FLAWED"
        rows[1]["caveat"] = "no within-seat variation against this opponent"
        rows[1]["partial_reason_class"] = "structural"
        rows[2]["caveat"] = "an unrelated, unclassified defect"
        # rows[2]["partial_reason_class"] stays "" -- unclassified.
        cell = _classify(rows)
        assert cell.kind == sb.PARTIAL

    def _p8_mixed_rows(self, unreached_row):
        """
        the four Dark Hex P8 capture-rate rows of one arm: cfr_ref and rl measured, always_bet
        and bayesian unreached
        :param unreached_row: builds the row for an unreached opponent, (opponent) -> dict
        :returns: the four row dicts
        """
        clause = "n_informative=0 of n_nodes=50010 real scored rows"
        measured = {"cfr_ref": 0.9, "rl": 0.8}
        rows = []
        for opponent in OPPONENT_LADDER:
            if opponent in measured:
                rows.append(
                    _row(
                        "darkhex",
                        opponent,
                        "P8_capture_rate",
                        measured[opponent],
                        "MEASURABLE",
                    )
                )
            else:
                rows.append(unreached_row(opponent, clause))
        return rows

    def test_a_not_reached_opponent_alongside_measured_ones_reads_partial_approved(
        self,
    ):
        """
        The mixed case the uniform NOT_REACHED branch never covered: two opponents measured, two
        unreached, every unreached row built by the scorer's own `_not_reached_row`. The cell
        reads `partial (approved)`, not plain `partial`.
        """
        rows = self._p8_mixed_rows(
            lambda opponent, clause: analyse_stats._not_reached_row(
                "darkhex", opponent, "P8_capture_rate", clause
            )
        )
        cell = _classify(rows, "P8_capture_rate")
        assert cell.kind == sb.PARTIAL_APPROVED
        assert "always_bet" in cell.text and "bayesian" in cell.text
        assert "n_informative=0 of n_nodes=50010" in cell.text

    def test_a_not_computed_bare_row_never_approves_a_partial_cell(self):
        """
        The class belongs to the scorer's zero-informative branch alone: a `NOT_COMPUTED` bare row
        beside measured rows still blocks the cell, so the cell reads plain `partial`.
        """
        rows = self._p8_mixed_rows(
            lambda opponent, clause: analyse_stats._bare_row(
                "darkhex", opponent, "P8_capture_rate", "NOT_COMPUTED", clause
            )
        )
        cell = _classify(rows, "P8_capture_rate")
        assert cell.kind == sb.PARTIAL

    def test_every_opponent_unreached_still_reads_structural_as_before(self):
        """
        The uniform case is unchanged by the class: all four opponents unreached, each built by
        `_not_reached_row`, reads structural and names that it is not by design.
        """
        clause = "n_informative=0 of n_nodes=50010 real scored rows"
        rows = [
            analyse_stats._not_reached_row("darkhex", o, "P8_capture_rate", clause)
            for o in OPPONENT_LADDER
        ]
        cell = _classify(rows, "P8_capture_rate")
        assert cell.kind == sb.STRUCTURAL
        assert cell.text.startswith(sb.NOT_REACHED_PREFIX)

    def test_partial_stays_partial_when_an_opponent_has_no_row_at_all(self):
        """A genuinely missing opponent (no row at all, as opposed to a present but
        not-scored one) never qualifies for `PARTIAL_APPROVED`, even when every opponent
        that does have a row is approved -- there is nothing measured for the missing one
        to approve.
        """
        rows = _opponents([0.2, 0.9, 0.4])  # only 3 of 4 opponents have a row at all
        rows[1]["status"] = "MEASURABLE_FLAWED"
        rows[1]["caveat"] = "no within-seat variation against this opponent"
        rows[1]["partial_reason_class"] = "structural"
        cell = _classify(rows)
        assert cell.kind == sb.PARTIAL
        assert "no row" in cell.full_text

    def test_a_full_ladder_is_a_number_showing_every_opponents_value(self):
        """
        Four admissible cells: the cell is a number, shows the admissible mean and each
        opponent's own value.
        """
        cell = _classify(_opponents([0.1, 0.2, 0.3, 0.4]))
        assert cell.kind == sb.NUMBER
        assert cell.text.startswith("0.2500")
        assert "(4/4 opponents)" in cell.text
        for opponent, value in zip(
            OPPONENT_LADDER, ("0.1000", "0.2000", "0.3000", "0.4000"), strict=False
        ):
            assert f"{opponent} {value}" in cell.text

    def test_a_zero_between_seed_variance_caveat_reaches_notes(self):
        """
        The real shape: the clause is appended after an existing caveat clause
        (`property_table`'s own `"; ".join(...)`, never the row's only clause), so the
        row's caveat here is two clauses, an ordinary one first and the zero-variance one
        second -- no scoreboard code change is needed for this, since both clauses reach the
        cell through the existing caveat mechanism.

        The short table-cell form (`.text`) shows the zero-variance clause in full, ahead of
        the leading clause, and marks the ordinary clause it does not show `(+1 more)`: the
        clause exists to say why a zero-width interval is not a tight one, so it is the one
        thing a short cell may not cut. The long `.detail` form carries both in full.
        """
        rows = _opponents(
            [0.1, 0.2, 0.3, 0.4],
            caveat=(
                "regret against the CFR equilibrium value (at most 0, 0 is "
                "equilibrium-optimal), decoding temperature 0.0; "
                "zero between-seed variance: 1 distinct value across 10 seeds"
            ),
        )
        cell = _classify(rows)
        assert cell.kind == sb.NUMBER
        assert (
            "zero between-seed variance: 1 distinct value across 10 seeds"
            in cell.detail
        )
        assert (
            "zero between-seed variance: 1 distinct value across 10 seeds" in cell.text
        )
        assert "(+1 more)" in cell.text

    def test_a_null_valued_measurable_row_is_named_not_dropped(self):
        """
        Three valued cells plus one MEASURABLE row whose scorer raised: the cell is partial
        and the failure text appears, so a 3-of-4 cell never hides the fourth.
        """
        rows = _opponents([0.1, 0.2, 0.3, None])
        rows[3]["caveat"] = "scorer raised: boom"
        cell = _classify(rows)
        assert cell.kind == sb.PARTIAL
        assert "3/4 opponents" in cell.text
        assert "scorer raised: boom" in cell.text

    def test_not_scored_opponents_are_listed_in_sorted_order(self):
        """
        Three non-contributing opponents supplied in reverse order are listed sorted, so
        the text does not depend on row order.
        """
        rows = [
            _row("kuhn", o, "P3_randomisation_margin", 0.5, "MEASURABLE_FLAWED", "d")
            for o in ("rl", "cfr_ref", "bayesian")
        ] + [_row("kuhn", "always_bet", "P3_randomisation_margin", 0.1, "MEASURABLE")]
        text = _classify(rows).text
        assert text.index("bayesian") < text.index("cfr_ref") < text.index("rl")

    def test_a_value_the_gate_withholds_reads_broken_with_the_value(self):
        """
        Every cell FLAWED: no admissible cell, so BROKEN with the blocker and the value.
        """
        cell = _classify(_opponents([0.5] * 4, "MEASURABLE_FLAWED", "defect x; more"))
        assert cell.kind == sb.BROKEN
        assert cell.text == "BROKEN: defect x (value 0.5000)"

    def test_a_refused_status_with_a_value_is_never_a_number(self):
        """
        A contaminated-game refusal that still carries a number reads BROKEN, the number
        shown only as the blocked value.
        """
        cell = _classify(
            _opponents([0.3] * 4, "REFUSED_FALLBACK_CONTAMINATED", "fallback rows")
        )
        assert cell.kind == sb.BROKEN
        assert "(value 0.3000)" in cell.text

    def test_an_empty_frame_reads_broken_with_the_scorer_text(self):
        """
        NOT_COMPUTED rows with no value read BROKEN and echo status and reason.
        """
        cell = _classify(
            _opponents([None] * 4, "NOT_COMPUTED", "no rows for this cell")
        )
        assert cell.kind == sb.BROKEN
        assert cell.text == "BROKEN: NOT_COMPUTED: no rows for this cell"

    def test_a_non_finite_mean_never_reads_as_a_number(self):
        """
        A decision row whose admissible mean is NaN, and one whose is infinite, read BROKEN
        and the text never contains nan or inf.
        """
        rows = _opponents([0.2, 0.4, 0.2, 0.4])
        for bad in (float("nan"), float("inf")):
            decision = {
                "observed_mean_admissible": bad,
                "observed_mean_all_cells": bad,
                "n_cells_admissible": 4,
                "blocker": "",
            }
            cell = sb.classify(
                "P3_randomisation_margin", decision, rows, False, OPPONENT_LADDER
            )
            assert cell.kind == sb.BROKEN
            assert "nan" not in cell.text.lower() and "inf" not in cell.text.lower()

    def test_a_decision_row_that_disagrees_with_the_rows_raises(self):
        """
        If the gate reports a different admissible count than the rows imply, the
        classifier raises rather than print a number from a rule it no longer shares.
        """
        rows = _opponents([0.2, 0.4, 0.2, 0.4])
        decision = {
            "observed_mean_admissible": 0.3,
            "observed_mean_all_cells": 0.3,
            "n_cells_admissible": 1,
            "blocker": "",
        }
        with pytest.raises(ValueError):
            sb.classify(
                "P3_randomisation_margin", decision, rows, False, OPPONENT_LADDER
            )

    def test_structural_needs_every_row_not_applicable_and_the_predicate(self):
        """
        Four NOT_APPLICABLE rows read structural when the predicate holds and BROKEN when
        it does not (a missing-reference NOT_APPLICABLE is an instrument gap).
        """
        rows = _opponents(
            [None] * 4, "NOT_APPLICABLE", "no CFR reference for darkhex yet"
        )
        assert _classify(rows, structural=True).text == (
            "structural: no CFR reference for darkhex yet"
        )
        gap = _classify(rows, structural=False)
        assert gap.kind == sb.BROKEN and gap.text.startswith("BROKEN: NOT_APPLICABLE")

    def test_one_not_computed_row_among_not_applicable_rows_is_broken(self):
        """
        Three NOT_APPLICABLE rows and one NOT_COMPUTED row with the predicate true: BROKEN.
        """
        rows = _opponents([None] * 4, "NOT_APPLICABLE", "by design")
        rows[3]["status"] = "NOT_COMPUTED"
        assert _classify(rows, structural=True).kind == sb.BROKEN

    def test_rows_present_but_no_lm_row_reads_broken_never_structural(self):
        """
        Only non-LM rows exist: BROKEN even with the predicate true, since all([]) is True.
        """
        rows = [
            _row(
                "kuhn",
                "rl",
                "P3_randomisation_margin",
                None,
                "NOT_APPLICABLE",
                "x",
                player="rl",
            )
        ]
        cell = _classify(rows, structural=True)
        assert cell.kind == sb.BROKEN and "no LM row" in cell.text

    def test_an_unseen_status_with_no_value_is_broken_and_echoed(self):
        """
        A status the code has never seen, with a null value, reads BROKEN naming it.
        """
        cell = _classify(
            _opponents([None] * 4, "SOMETHING_NEW", "odd"), structural=True
        )
        assert cell.kind == sb.BROKEN and "SOMETHING_NEW" in cell.text

    def test_non_lm_rows_never_contribute(self):
        """
        A MEASURABLE value on a non-LM row alone leaves the cell BROKEN.
        """
        rows = [
            _row(
                "kuhn",
                "always_bet",
                "P3_randomisation_margin",
                0.9,
                "MEASURABLE",
                player="rl",
            ),
            _row(
                "kuhn",
                "rl",
                "P3_randomisation_margin",
                None,
                "NOT_COMPUTED",
                "n",
                player="LM",
            ),
        ]
        assert _classify(rows).kind == sb.BROKEN

    def test_an_absent_row_reads_broken_no_row(self):
        """
        No rows for the headline property: `BROKEN: no <row> row`.
        """
        cell = sb.classify("P2_adaptivity_delta", None, [], False, OPPONENT_LADDER)
        assert cell.text == "BROKEN: no P2_adaptivity_delta row"

    def test_a_missing_opponent_never_shrinks_the_denominator(self):
        """
        Three of four opponents present: the cell reads 3 of 4 and names the absent one as
        "no row", instead of a complete-looking 3 of 3.
        """
        rows = _opponents([0.1, 0.2, 0.3, 0.4])[:3]
        cell = _classify(rows)
        assert cell.kind == sb.PARTIAL
        assert cell.text.startswith("partial: 3/4 opponents; mean 0.2000")
        assert cell.text.endswith("; not scored: bayesian no row")

    def test_a_duplicated_or_unexpected_opponent_row_reads_broken(self):
        """
        Two rows for one opponent, or a row for an opponent outside the ladder, is a
        malformed input and reads BROKEN rather than moving the mean.
        """
        rows = _opponents([0.1, 0.2, 0.3, 0.4])
        dup = _classify(rows + [copy.deepcopy(rows[1])])
        assert dup.kind == sb.BROKEN and "duplicate LM rows for" in dup.text
        stray = _classify(
            rows
            + [_row("kuhn", "mystery", "P3_randomisation_margin", 0.9, "MEASURABLE")]
        )
        assert stray.kind == sb.BROKEN and "unexpected opponent" in stray.text

    def test_structural_needs_every_expected_opponent_to_have_a_row(self):
        """
        Three NOT_APPLICABLE rows on a by-design cell with the fourth opponent absent: the
        missing data reads BROKEN, never structural.
        """
        rows = _opponents([None] * 3, "NOT_APPLICABLE", "by design")
        cell = _classify(rows, structural=True)
        assert cell.kind == sb.BROKEN and "no row for bayesian" in cell.text

    def test_a_structural_cell_must_state_a_reason(self):
        """
        NOT_APPLICABLE rows with no caveat on a by-design cell read BROKEN, not an empty
        `structural: `.
        """
        cell = _classify(_opponents([None] * 4, "NOT_APPLICABLE", ""), structural=True)
        assert cell.kind == sb.BROKEN

    def test_one_refused_opponent_among_clean_ones_makes_the_cell_partial(self):
        """
        A single contaminated-game refusal among three clean opponents is partial, never a
        number, and names the refusal.
        """
        rows = _opponents([0.1, 0.2, 0.3, 0.4])
        rows[3]["status"] = "REFUSED_FALLBACK_CONTAMINATED"
        cell = _classify(rows)
        assert cell.kind == sb.PARTIAL
        assert (
            "3/4 opponents" in cell.text
            and "REFUSED_FALLBACK_CONTAMINATED" in cell.text
        )

    def test_a_number_cell_without_a_recorded_dropped_rate_says_so(self):
        """
        Rows carrying no `n_total` give the gate no dropped rate, and the cell says the rate
        was not recorded rather than showing a clean zero.
        """
        cell = _classify(_opponents([0.1, 0.2, 0.3, 0.4]))
        assert cell.kind == sb.NUMBER
        assert "dropped rate not recorded" in cell.text
        assert "dropped 0.00%" not in cell.text

    def test_a_number_cell_shows_the_dropped_rate_the_rows_record(self):
        """
        Rows carrying drop counts reach the cell as a percentage pooled by the project's own
        gate, beside the value.
        """
        rows = _opponents([0.1, 0.2, 0.3, 0.4])
        for row in rows:
            row["n_total"], row["n_dropped"], row["dropped_unit"] = 100, 3, "rows"
        cell = _classify(rows)
        assert cell.kind == sb.NUMBER and "dropped 3.00%" in cell.text

    def test_a_number_cell_shows_the_gates_interval_and_verdict(self):
        """
        A cell whose gate row carries a replicate-interval envelope and a verdict shows both
        beside the mean, in the form `<mean> [lo, hi] <verdict> (4/4 opponents)`.
        """
        rows = _opponents([-0.02, 0.0, 0.01, 0.02])
        for row, (lo, hi) in zip(
            rows, ((-0.03, -0.01), (-0.01, 0.01), (0.0, 0.02), (0.01, 0.03)), strict=False
        ):
            row["ci_lo"], row["ci_hi"] = lo, hi
        cell = _classify(rows)
        assert cell.kind == sb.NUMBER
        assert cell.text.startswith(
            "0.0025 [-0.0300, 0.0300] CI_INDETERMINATE (4/4 opponents)"
        )

    def test_a_p2_number_cell_carries_its_verdict_through_the_p2_headline_row(self):
        """
        The P2 headline row is the one the project's decision rule thresholds at zero, and
        its cell reads the mean, the interval and the verdict together.
        """
        rows = [
            _row("kuhn", o, "P2_adaptivity_delta", v, "MEASURABLE")
            for o, v in zip(OPPONENT_LADDER, (-0.01, -0.002, 0.001, 0.0066), strict=False)
        ]
        for row, (lo, hi) in zip(
            rows, ((-0.02, 0.0), (-0.01, 0.005), (-0.004, 0.006), (0.001, 0.012)), strict=False
        ):
            row["ci_lo"], row["ci_hi"] = lo, hi
        cell = _classify(rows, name="P2_adaptivity_delta")
        assert cell.kind == sb.NUMBER
        assert re.match(
            r"^-?\d+\.\d{4} \[-?\d+\.\d{4}, -?\d+\.\d{4}\] [A-Z_]+ \(4/4 opponents\)",
            cell.text,
        ), cell.text

    def test_a_pipe_in_a_reason_cannot_break_the_table_row(self):
        """
        A caveat containing a pipe reads without one, so the markdown row keeps its columns.
        """
        cell = _classify(_opponents([None] * 4, "NOT_COMPUTED", "a | b"))
        assert "|" not in cell.text and "a / b" in cell.text

    def test_a_tiny_negative_mean_never_renders_as_negative_zero(self):
        """
        A mean of -0.00001 renders as 0.0000, not -0.0000.
        """
        cell = _classify(_opponents([-0.00001] * 4))
        assert cell.text.startswith("0.0000")
        assert "-0.0000" not in cell.text

    def test_combine_never_reads_an_awaiting_part_as_a_number(self):
        """
        A property with two parts (P7) where a part is awaiting: two awaiting parts read
        awaiting, an awaiting part beside a number reads awaiting, and beside a BROKEN part the
        cell is BROKEN naming the awaiting one. None is a number or structural.
        """
        wait = sb.Cell(sb.AWAITING, "awaiting data: not collected yet")
        num = sb.Cell(sb.NUMBER, "0.1000 (4/4 opponents)")
        bad = sb.Cell(sb.BROKEN, "BROKEN: x")
        both = sb.combine(["7a", "7b"], [wait, wait])
        assert both.kind == sb.AWAITING and both.text.startswith("awaiting data: ")
        assert sb.combine(["7a", "7b"], [wait, num]).kind == sb.AWAITING
        mixed = sb.combine(["7a", "7b"], [wait, bad])
        assert mixed.kind == sb.BROKEN and "7a awaiting" in mixed.text

    def test_combine_is_broken_when_either_part_is(self):
        """
        Two parts, one number and one BROKEN: the cell is BROKEN and names both; both
        structural: structural; both numbers: a number form.
        """
        num = sb.Cell(sb.NUMBER, "0.1000 (4/4 opponents)")
        bad = sb.Cell(sb.BROKEN, "BROKEN: x")
        na = sb.Cell(sb.STRUCTURAL, "structural: by design")
        assert sb.combine(["7a", "7b"], [num, bad]).kind == sb.BROKEN
        assert sb.combine(["7a", "7b"], [bad, num]).kind == sb.BROKEN
        assert "7a ok 0.1000" in sb.combine(["7a", "7b"], [num, bad]).text
        assert sb.combine(["7a", "7b"], [na, na]).kind == sb.STRUCTURAL
        both = sb.combine(["7a", "7b"], [num, num])
        assert both.kind == sb.NUMBER and NUMBER_FORM.match(both.text)

    def test_combine_carries_partial_approved_through_a_two_part_property(self):
        """
        Regression: if `combine`'s kind check names only `PARTIAL`, a part scored
        `PARTIAL_APPROVED` falls through to `NUMBER` and the "partial: " marker is silently
        dropped even though the part's own text still reads "k/n opponents". A
        `PARTIAL_APPROVED` part beside a `NUMBER` part
        combines to `PARTIAL_APPROVED`; a `PARTIAL_APPROVED` part beside an unapproved
        `PARTIAL` part combines to plain `PARTIAL`, since the combined cell is not fully
        approved while any part is not.
        """
        approved = sb.Cell(sb.PARTIAL_APPROVED, "partial: 1/4 opponents; mean 0.1000")
        num = sb.Cell(sb.NUMBER, "0.2000 (4/4 opponents)")
        unapproved = sb.Cell(sb.PARTIAL, "partial: 2/4 opponents; mean 0.1500")

        both_approved = sb.combine(["7a", "7b"], [approved, num])
        assert both_approved.kind == sb.PARTIAL_APPROVED
        assert both_approved.text.startswith("partial: ")
        assert "7a 1/4 opponents" in both_approved.text

        mixed = sb.combine(["7a", "7b"], [approved, unapproved])
        assert mixed.kind == sb.PARTIAL
        assert mixed.text.startswith("partial: ")


class TestHeadlineAndPredicate:
    def test_every_property_has_a_headline_row_and_p7_has_two(self):
        """
        DECISION_RULE covers P1 to P8, P7 has two rows, every row name has a static status.
        """
        heads = sb.headline_rows(analyse_stats.DECISION_RULE)
        assert sorted(heads) == list(range(1, 9))
        assert len(heads[7]) == 2
        for names in heads.values():
            for name in names:
                assert name in analyse_stats.PROPERTY_STATUS
        without_p8 = [
            d for d in analyse_stats.DECISION_RULE if not d[0].startswith("P8")
        ]
        with pytest.raises(ValueError):
            sb.headline_rows(without_p8)

    def test_structural_predicate_follows_the_project_family_tests(self):
        """
        P5 and P7 are structural for no family -- darkhex is not in
        `_BELIEF_NOT_APPLICABLE_FAMILIES` because it has a per-cell belief estimand, so
        a blank Dark Hex belief cell is a scoring problem, not a structural absence.
        P8 stays structural for every family but darkhex, P3 is never structural for any
        game at any level, and P6 is never structural at M0 (its M1 and M2 absence is the
        memory rule's, tested with P1 and P4).
        """
        p8_family = {"darkhex", "darkhex_3x3"}
        for game in (
            "kuhn",
            "leduc",
            "goofspiel",
            "liarsdice",
            "darkhex",
            "darkhex_3x3",
        ):
            assert cli.structural_by_design(game, 5, "M1") is False
            assert cli.structural_by_design(game, 7, "M1") is False
            assert cli.structural_by_design(game, 8, "M1") == (game not in p8_family)
        for game in GAME_FAMILIES:
            for level in ("M0", "M1", "M2", None):
                assert cli.structural_by_design(game, 3, level) is False
            for level in ("M0", None):
                assert cli.structural_by_design(game, 6, level) is False

    def test_p1_p4_and_p6_are_never_structural(self):
        """
        P1, P4 and P6 are defined at every memory level, so the predicate says structural at
        none of them, for every game.
        """
        for game in GAME_FAMILIES:
            for number in (1, 4, 6):
                for level in ("M0", "M1", "M2", None):
                    assert cli.structural_by_design(game, number, level) is False

    def test_p2_is_structural_at_m0_and_not_at_m1_or_m2(self):
        """
        Nothing carries between hands at M0, so P2 cannot exist there and the predicate
        says so for every game; at M1 and M2 it is measurable and the predicate does not.
        """
        for game in GAME_FAMILIES:
            assert cli.structural_by_design(game, 2, "M0") is True
            assert cli.structural_by_design(game, 2, None) is True
            assert cli.structural_by_design(game, 2, "M1") is False
            assert cli.structural_by_design(game, 2, "M2") is False

    def test_no_by_design_cell_in_the_fixtures_carries_a_valued_lm_row(self):
        """
        Over the real v6 slices every by-design (game, property) carries an unvalued
        NOT_APPLICABLE LM headline row, and the converse does not hold: these
        fixtures were scored without a Dark Hex belief estimand, so their Dark
        Hex P5/P7 rows are NOT_APPLICABLE on a (game, property) that is not
        by-design. Such a cell reads BROKEN, which is the intended behaviour -- the
        artefact records a state the current scorers do not produce.
        """
        heads = sb.headline_rows(analyse_stats.DECISION_RULE)
        number_of = {name: n for n, names in heads.items() for name in names}
        stale_belief_rows = 0
        for model in MODELS:
            df = pl.read_csv(
                FIXTURES / model / "properties.csv", infer_schema_length=10000
            )
            for row in df.filter(pl.col("player") == "LM").iter_rows(named=True):
                number = number_of[row["property"]]
                by_design = cli.structural_by_design(row["game"], number)
                if by_design:
                    assert (
                        row["status"] == "NOT_APPLICABLE" and row["value"] is None
                    ), row
                if row["status"] == "NOT_APPLICABLE" and not by_design:
                    assert number in (5, 7) and row["game"] == "darkhex", row
                    stale_belief_rows += 1
        assert stale_belief_rows > 0


class TestBuildCells:
    def _rows(self, model="qwen2.5-14b-instruct"):
        return pl.read_csv(
            FIXTURES / model / "properties.csv", infer_schema_length=10000
        ).to_dicts()

    def _build(self, rows, rule=analyse_stats.DECISION_RULE):
        return sb.build_cells(
            "qwen2.5-14b-instruct",
            rows,
            rule,
            analyse_stats.decision_table,
            cli.structural_by_design,
            OPPONENT_LADDER,
        )

    def test_forty_cells_per_arm_in_a_fixed_order(self):
        """
        One arm gives 40 cells, games in the fixed constant order, properties 1 to 8.
        """
        cells = self._build(self._rows())
        assert len(cells) == 40
        assert [c.game for c in cells[::8]] == list(sb.SCOREBOARD_GAMES)
        assert [c.number for c in cells[:8]] == list(range(1, 9))

    def test_a_game_with_no_rows_still_gets_eight_broken_cells(self):
        """
        Dropping every kuhn row leaves 40 cells with kuhn reading BROKEN: no row.
        """
        rows = [r for r in self._rows() if r["game"] != "kuhn"]
        cells = self._build(rows)
        assert len(cells) == 40
        assert all(c.cell.kind == sb.BROKEN for c in cells if c.game == "kuhn")

    def test_a_game_outside_the_constant_raises(self):
        """
        A headline row for a game the board does not list is an error, not a silent drop.
        """
        rows = self._rows() + [
            _row("chess", "rl", "P3_randomisation_margin", 0.1, "MEASURABLE")
        ]
        with pytest.raises(ValueError):
            self._build(rows)

    def test_row_order_does_not_change_the_cells(self):
        """
        Reversing the input row order gives the same cells, including a planted cell with
        three non-contributing opponents whose listing order would follow the input order.
        """
        prop = "P5_calibration_per_pair_margin"
        rows = [
            r
            for r in self._rows()
            if not (r["game"] == "kuhn" and r["property"] == prop)
        ]
        rows.append(_row("kuhn", "always_bet", prop, -0.1, "MEASURABLE"))
        rows += [
            _row("kuhn", o, prop, 0.5, "MEASURABLE_FLAWED", "defect")
            for o in ("rl", "cfr_ref", "bayesian")
        ]
        forward = self._build(rows)
        assert forward == self._build(list(reversed(rows)))
        kuhn5 = next(c for c in forward if (c.game, c.number) == ("kuhn", 5))
        text = kuhn5.cell.text
        assert text.index("bayesian") < text.index("cfr_ref") < text.index("rl")

    def test_a_nan_value_cannot_poison_the_mean_of_its_neighbours(self):
        """
        One MEASURABLE row with a NaN value beside three valid rows: the mean stays
        finite over the three and the NaN opponent is named as not scored.
        """
        rows = [
            r
            for r in self._rows()
            if not (
                r["game"] == "kuhn"
                and r["property"] == "P5_calibration_per_pair_margin"
            )
        ]
        for i, opp in enumerate(("always_bet", "bayesian", "cfr_ref", "rl")):
            rows.append(
                _row(
                    "kuhn",
                    opp,
                    "P5_calibration_per_pair_margin",
                    float("nan") if i == 3 else -0.1 * (i + 1),
                    "MEASURABLE",
                )
            )
        cell = next(c for c in self._build(rows) if (c.game, c.number) == ("kuhn", 5))
        assert cell.cell.kind == sb.PARTIAL
        assert cell.cell.text.startswith("partial: 3/4 opponents; mean -0.2000")
        assert "rl" in cell.cell.text and "nan" not in cell.cell.text.lower().replace(
            "non-finite", ""
        )

    def test_a_long_run_of_reasons_is_capped_and_keeps_its_form(self):
        """
        Four opponents each carrying a long distinct reason yield a cell no longer than the
        cap, still reading BROKEN and ending in an ellipsis.
        """
        rows = [
            _row(
                "kuhn",
                o,
                "P3_randomisation_margin",
                None,
                "MEASURABLE",
                f"reason {o} " + "x" * 200,
            )
            for o in ("always_bet", "bayesian", "cfr_ref", "rl")
        ]
        cell = next(
            c
            for c in self._build(
                rows + self._rows_without("kuhn", "P3_randomisation_margin")
            )
            if (c.game, c.number) == ("kuhn", 3)
        )
        assert cell.cell.kind == sb.BROKEN and cell.cell.text.startswith("BROKEN: ")
        assert len(cell.cell.text) <= sb.MAX_CELL_CHARS and cell.cell.text.endswith(
            "..."
        )

    def test_a_long_seven_a_tail_cannot_push_seven_b_out_of_the_cell(self):
        """
        Property 7 has two parts; a long "not scored" list on 7a leaves 7b's number in the
        cell, because each part is capped on its own and the whole stays short.
        """
        rows = self._rows_without("kuhn", "P7a_belief_error")
        for i, o in enumerate(OPPONENT_LADDER):
            rows.append(
                _row(
                    "kuhn",
                    o,
                    "P7a_belief_error",
                    0.2 if i < 1 else 0.5,
                    "MEASURABLE" if i < 1 else "MEASURABLE_FLAWED",
                    f"flaw {o} " + "y" * 200,
                )
            )
        cell = next(c for c in self._build(rows) if (c.game, c.number) == ("kuhn", 7))
        assert cell.cell.kind == sb.PARTIAL
        assert cell.cell.text.startswith("partial: 7a 1/4 opponents; mean 0.2000")
        assert re.search(r"/ 7b -?\d+\.\d{4}", cell.cell.text)
        assert "(4/4 opponents)" in cell.cell.text.split("/ 7b ")[1]
        assert len(cell.cell.text) <= 2 * sb.MAX_PART_CHARS + 30

    def test_a_non_finite_value_keeps_the_scorers_own_reason_visible(self):
        """
        Every P3 row NaN with the scorer's caveat: the cell reads BROKEN naming both the
        scorer's reason and the non-finite value.
        """
        rows = self._rows_without("kuhn", "P3_randomisation_margin")
        rows += [
            _row(
                "kuhn",
                o,
                "P3_randomisation_margin",
                float("nan"),
                "MEASURABLE_CONFOUNDED",
                "the scorer's own stated reason; extra",
            )
            for o in OPPONENT_LADDER
        ]
        cell = next(c for c in self._build(rows) if (c.game, c.number) == ("kuhn", 3))
        assert cell.cell.kind == sb.BROKEN
        assert "the scorer's own stated reason" in cell.cell.text
        assert "non-finite value" in cell.cell.text

    def _rows_without(self, game, prop):
        return [
            r for r in self._rows() if not (r["game"] == game and r["property"] == prop)
        ]

    def test_missing_required_columns_are_named(self):
        """
        A column set without `status` raises naming it; extra columns are accepted.
        """
        with pytest.raises(ValueError, match="status"):
            sb.check_required_columns(
                ["game", "opponent", "player", "property", "value", "caveat"]
            )
        sb.check_required_columns(
            [
                "game",
                "opponent",
                "player",
                "property",
                "value",
                "status",
                "caveat",
                "x_new",
            ]
        )

    def test_coverage_floors_are_looked_up_by_game_not_applied_globally(self):
        """
        `build_cells` threads `coverage_floors` through per-(game, property), not as
        one global floor. Leduc's row is below its own floor (0.5) and reads INADMISSIBLE;
        kuhn's row, with the same coverage value but no floor entry for kuhn in this
        artefact, is unaffected.
        """
        probe_rows = [
            _probe_row(
                game="leduc",
                prop="P1_invariance",
                value=0.3,
                n=10,
                coverage=0.9,
                n_seed_mean=8.0,
                coverage_seed_mean=0.2,
            ),
            _probe_row(
                game="kuhn",
                prop="P1_invariance",
                value=0.3,
                n=10,
                coverage=0.9,
                n_seed_mean=8.0,
                coverage_seed_mean=0.2,
            ),
        ]
        coverage_floors = {"leduc": {"P1_invariance": {"floor": 0.5}}}
        cells = sb.build_cells(
            "qwen2.5-14b-instruct",
            self._rows(),
            analyse_stats.DECISION_RULE,
            analyse_stats.decision_table,
            cli.structural_by_design,
            OPPONENT_LADDER,
            probe_names=("P1_invariance",),
            probe_rows=probe_rows,
            coverage_floors=coverage_floors,
        )
        p1_number = next(
            n
            for n, names in sb.headline_rows(analyse_stats.DECISION_RULE).items()
            if names == ["P1_invariance"]
        )
        leduc_cell = next(
            c for c in cells if c.game == "leduc" and c.number == p1_number
        )
        kuhn_cell = next(c for c in cells if c.game == "kuhn" and c.number == p1_number)
        assert leduc_cell.cell.kind == sb.INADMISSIBLE
        assert kuhn_cell.cell.kind == sb.NUMBER


class TestRenderAndSplice:
    def _block(self, footnotes=()):
        cells = []
        arms = []
        for model in MODELS:
            rows = pl.read_csv(
                FIXTURES / model / "properties.csv", infer_schema_length=10000
            ).to_dicts()
            cells += sb.build_cells(
                model,
                rows,
                vocabulary.DECISION_RULE,
                vocabulary.decision_table,
                cli.structural_by_design,
                OPPONENT_LADDER,
            )
            arms.append(
                sb.ArmSummary(
                    model,
                    sb.short_data_dir(
                        "/home/x/data/run_experiment/recycle-v6-m1/c-" + model
                    ),
                    sb.file_sha256_short(FIXTURES / model / "properties.csv"),
                )
            )
        boards = [sb.Board(level, tuple(cells)) for level in ("M0", "M1", "M2")]
        return sb.render_block(
            arms,
            boards,
            analyse_stats.DECISION_RULE,
            vocabulary._DIRECTION_TEXT,
            footnotes,
        )

    def test_rendering_is_deterministic_and_carries_no_timestamp_or_host_path(self):
        """
        Two renders are byte-identical; the header and the table hold no year-month-day
        stamp, and no absolute path appears anywhere. A scorer's own caveat may quote a
        date, so the stamp check is over everything the generator itself writes.
        """
        block = self._block()
        assert block == self._block()
        header = block.split("| model | game | property | cell |")[0]
        assert not re.search(r"\d{4}-\d{2}-\d{2}", header)
        assert "/home/" not in block and "/tmp/" not in block

    def test_each_board_has_120_cells_and_one_legend_names_every_headline_row(self):
        """
        Three boards of 120 rows give 360 rows in all, and the legend, written once, names
        every DECISION_RULE row (P8's included).
        """
        block = self._block()
        assert len(ROW.findall(block)) == 360
        for level in ("M0", "M1", "M2"):
            assert sum(1 for m in ROW.finditer(block) if m.group(1) == level) == 120
        for name, *_ in analyse_stats.DECISION_RULE:
            assert name in block

    def test_footnotes_are_appended_verbatim_below_the_table(self):
        """
        A footnote line supplied by the caller appears after the table, unchanged, separated
        from the last row by a blank line so a markdown renderer does not read it as a row.
        """
        block = self._block(["a caller footnote"])
        assert block.endswith("|\n\na caller footnote")

    def test_the_header_hash_follows_the_file_contents(self, tmp_path):
        """
        Changing one byte of a properties.csv changes its short sha256.
        """
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        a.write_text("x\n")
        b.write_text("y\n")
        assert sb.file_sha256_short(a) != sb.file_sha256_short(b)
        assert sb.file_sha256_short(a) == hashlib.sha256(b"x\n").hexdigest()[:12]

    def test_the_header_hash_ignores_line_endings(self, tmp_path):
        """
        The same file written with CRLF and with LF hashes the same, so a block generated
        from a working copy still verifies against a fresh checkout of the committed LF file.
        """
        crlf = tmp_path / "crlf.csv"
        lf = tmp_path / "lf.csv"
        crlf.write_bytes(b"a,b\r\n1,2\r\n")
        lf.write_bytes(b"a,b\n1,2\n")
        assert sb.file_sha256_short(crlf) == sb.file_sha256_short(lf)

    def test_the_header_says_the_hash_is_over_line_ending_normalised_bytes(self):
        """
        The provenance column names its own normalisation, so a reader running sha256sum on a
        CRLF working copy is not misled by a mismatch.
        """
        header = next(
            line for line in self._block().splitlines() if line.startswith("| arm |")
        )
        assert "LF-normalised" in header

    def test_a_block_carrying_a_marker_is_refused(self):
        """
        A block containing either marker would corrupt every later run, so the splice
        refuses it.
        """
        for marker in (sb.BEGIN_MARKER, sb.END_MARKER):
            with pytest.raises(ValueError):
                sb.replace_scoreboard_block(BOARD_FILE, f"text {marker} text")

    def test_data_dir_shows_only_its_last_two_components(self):
        """
        An absolute host path is reduced to `<parent>/<name>`.
        """
        assert sb.short_data_dir("/home/a/b/recycle-v6-m1/composite-x") == (
            "recycle-v6-m1/composite-x"
        )
        assert sb.short_data_dir("data/run_experiment/recycle-v6-m1/composite-x") == (
            "recycle-v6-m1/composite-x"
        )

    def test_splice_changes_only_the_bytes_between_the_markers(self):
        """
        Text before and after, and the markers, are byte-identical; applying twice equals
        applying once.
        """
        once = sb.replace_scoreboard_block(BOARD_FILE, "NEW BLOCK")
        assert once.startswith(
            "# tracker\n\nbefore the block\n<!-- SCOREBOARD:BEGIN -->\n"
        )
        assert once.endswith("\n<!-- SCOREBOARD:END -->\nafter the block\n")
        assert "NEW BLOCK" in once and "_not yet generated_" not in once
        assert sb.replace_scoreboard_block(once, "NEW BLOCK") == once

    @pytest.mark.parametrize(
        "text",
        [
            "no markers at all\n",
            "<!-- SCOREBOARD:BEGIN -->\nonly begin\n",
            "<!-- SCOREBOARD:END -->\n<!-- SCOREBOARD:BEGIN -->\nreversed\n",
            BOARD_FILE + BOARD_FILE,
        ],
    )
    def test_missing_duplicate_or_reversed_markers_raise(self, text):
        """
        Zero, one, reversed or duplicated markers raise and never append a block.
        """
        with pytest.raises(ValueError):
            sb.replace_scoreboard_block(text, "BLOCK")


class TestCli:
    def test_the_real_entrypoint_gives_three_boards_of_120_with_measured_tallies(
        self,
        tmp_path,
    ):
        """
        Production anchor: `main` over the fixture root writes one 120-cell board per memory
        level, each cell in one of the five forms, with the per-model per-board tallies an
        independent run measured, and leaves every byte outside the markers alone.
        """
        code, text = _run(tmp_path)
        assert code == cli.EXIT_OK
        cells = _all_cells(text)
        assert len(cells) == 360
        for level, per_model in TALLIES.items():
            for model in MODEL_OF.values():
                kinds = [
                    _kind(c)
                    for (lv, m, _g, _p), c in cells.items()
                    if lv == level and m == model
                ]
                tally = tuple(
                    kinds.count(k)
                    for k in (
                        "number",
                        "partial",
                        "partial_approved",
                        "structural",
                        "awaiting",
                        "inadmissible",
                        "BROKEN",
                    )
                )
                assert tally == per_model[model], (level, model, tally)
                assert sum(tally) == 40
        assert text.startswith(
            "# tracker\n\nbefore the block\n<!-- SCOREBOARD:BEGIN -->\n"
        )
        assert text.endswith("<!-- SCOREBOARD:END -->\nafter the block\n")
        for (level, model, game, _prop), cell in cells.items():
            if _kind(cell) == "number":
                assert "(4/4 opponents)" in cell, cell
            if _kind(cell) == "awaiting":
                assert (
                    f"this model has no {level} " in cell
                    or "no probe_scores.csv for this arm" in cell
                ), (level, model, game, cell)

    def test_a_model_read_at_one_level_carries_no_tail_from_another(self, tmp_path):
        """
        A model with an M0 arm and an M1 arm gets full cells on both boards, and the M1 cell
        states only its own condition: the M0 value is on the M0 board, never appended to it.
        """
        root = _copy_root(tmp_path, rung="M0")
        twin = root / "qwen-second"
        shutil.copytree(root / "qwen2.5-14b-instruct", twin)
        _set_rung(root, "qwen-second", "M1")
        cells = _all_cells(_run(tmp_path, root, extra=["--expect-arms", "4"])[1])
        m0, m1 = cells[("M0", QWEN, "kuhn", "P5")], cells[("M1", QWEN, "kuhn", "P5")]
        assert _kind(m0) == "number" and _kind(m1) == "number"
        assert "M1 " not in m0 and "M0 " not in m1
        assert not re.search(r"; M\d ", m0) and not re.search(r"; M\d ", m1)

    def test_a_level_a_model_has_no_arm_for_waits_and_never_reads_another_level(
        self,
        tmp_path,
    ):
        """
        With only an M1 arm, every M0 cell that is not structurally absent reads awaiting
        data naming the M0 run, while the M1 board reads the arm.
        """
        cells = _all_cells(_run(tmp_path)[1])
        m0 = {k: v for k, v in cells.items() if k[0] == "M0" and k[1] == QWEN}
        assert len(m0) == 40
        assert not any(_kind(v) in ("number", "partial") for v in m0.values())
        assert _kind(cells[("M0", QWEN, "kuhn", "P5")]) == "awaiting"
        assert "this model has no M0 run" in cells[("M0", QWEN, "kuhn", "P5")]
        assert _kind(cells[("M1", QWEN, "kuhn", "P5")]) == "number"

    def test_the_six_kinds_are_counted_per_board_and_the_boards_sum_to_the_overall_line(
        self,
        tmp_path,
    ):
        """
        Each board's own count line names all six kinds and sums to 120, and the overall
        line is the sum of the three boards, so neither `partial` nor a whole board can be
        folded away unnoticed.
        """
        _code, text = _run(tmp_path)
        overall = [0] * len(sb.KINDS)
        for level, per_model in TALLIES.items():
            board = tuple(
                sum(t[i] for t in per_model.values()) for i in range(len(sb.KINDS))
            )
            assert sum(board) == 120, level
            assert (
                f"120 cells: {board[0]} numbers, {board[1]} partial, "
                f"{board[2]} partial (approved), {board[3]} "
                f"structural, {board[4]} awaiting data, {board[5]} inadmissible, "
                f"{board[6]} BROKEN."
            ) in text
            assert f"#### Memory level {level}" in text
            overall = [a + b for a, b in zip(overall, board, strict=False)]
        assert sum(overall) == 360
        assert (
            "One board per memory level (M0, M1, M2); 360 cells in all "
            f"over 3 arms: {overall[0]} numbers, {overall[1]} partial, "
            f"{overall[2]} partial (approved), {overall[3]} structural, "
            f"{overall[4]} awaiting data, {overall[5]} inadmissible, "
            f"{overall[6]} BROKEN."
        ) in text

    def test_p2_is_read_and_p1_p4_p6_wait_for_a_probe_run_on_the_m1_baseline(
        self, tmp_path
    ):
        """
        On the real v6 slices, which are M1 runs with no probe file, no P2 cell of the M1
        board is structural or a number: its rows were never produced. P1, P4 and P6 at M1
        wait for the probe run they are defined on, and none reads structural.
        """
        _code, text = _run(tmp_path)
        for (_m, game, prop), cell in _cells(text).items():
            if prop == "P2":
                assert _kind(cell) == "BROKEN", (prop, game, cell)
            if prop in ("P1", "P4", "P6"):
                assert _kind(cell) == "awaiting", (prop, game, cell)
                assert "no probe_scores.csv for this arm" in cell, (prop, game, cell)

    def test_p2_is_structural_on_the_m0_board_and_read_on_the_m1_board(self, tmp_path):
        """
        P2 on the M0 board states that nothing carries between hands there, whatever the
        model; on the M1 board the same model and game reads the M1 run, so it is a number,
        partial, awaiting or BROKEN, never structural.
        """
        cells = _all_cells(_run(tmp_path)[1])
        for model in MODEL_OF.values():
            for game in sb.SCOREBOARD_GAMES:
                m0 = cells[("M0", model, game, "P2")]
                assert _kind(m0) == "structural", (model, game, m0)
                assert "nothing carries between hands at M0" in m0
                assert _kind(cells[("M1", model, game, "P2")]) != "structural"
                assert _kind(cells[("M2", model, game, "P2")]) != "structural"

    def test_data_not_collected_on_a_by_design_cell_reads_broken(self, tmp_path):
        """
        Kuhn P8 is structural by design, but when its LM rows say NOT_COMPUTED (nothing
        collected) the cell reads BROKEN, not structural.
        """
        root = _copy_root(tmp_path, rung="M0")

        def plant(df):
            mask = (pl.col("game") == "kuhn") & (
                pl.col("property") == "P8_capture_rate"
            )
            return df.with_columns(
                pl.when(mask)
                .then(pl.lit("NOT_COMPUTED"))
                .otherwise(pl.col("status"))
                .alias("status")
            )

        _edit(root, "qwen2.5-14b-instruct", plant)
        _code, text = _run(tmp_path, root)
        assert _kind(_cells(text, "M0")[(QWEN, "kuhn", "P8")]) == "BROKEN"

    def test_refusals_nan_and_empty_frames_never_read_as_numbers(self, tmp_path):
        """
        Planted through the real CLI: a refused status with a value, an all-NaN cell and an
        emptied cell all read BROKEN, and the block never contains "nan".
        """
        root = _copy_root(tmp_path, rung="M0")
        prop = "P5_calibration_per_pair_margin"

        def plant(df):
            kuhn5 = (pl.col("game") == "kuhn") & (pl.col("property") == prop)
            leduc5 = (pl.col("game") == "leduc") & (pl.col("property") == prop)
            gs5 = (pl.col("game") == "goofspiel") & (pl.col("property") == prop)
            return df.with_columns(
                pl.when(kuhn5)
                .then(pl.lit("REFUSED_FALLBACK_CONTAMINATED"))
                .when(gs5)
                .then(pl.lit("NOT_COMPUTED"))
                .otherwise(pl.col("status"))
                .alias("status"),
                pl.when(leduc5)
                .then(pl.lit(float("nan")))
                .when(gs5)
                .then(pl.lit(None))
                .otherwise(pl.col("value"))
                .alias("value"),
            )

        _edit(root, "qwen2.5-14b-instruct", plant)
        _code, text = _run(tmp_path, root)
        cells = _cells(text, "M0")
        for game in ("kuhn", "leduc", "goofspiel"):
            assert _kind(cells[(QWEN, game, "P5")]) == "BROKEN", game
        assert "nan" not in text.lower().replace("non-finite", "")

    def test_all_120_cells_of_every_board_are_present_whatever_the_input_lacks(
        self,
        tmp_path,
    ):
        """
        An arm whose properties.csv holds only the header still yields 120 rows on each of
        the three boards, with that arm's 40 M0 cells carrying no value.
        """
        root = _copy_root(tmp_path, rung="M0")
        _edit(root, "olmo-2-1124-13b-instruct", lambda df: df.head(0))
        _code, text = _run(tmp_path, root)
        assert len(_all_cells(text)) == 360
        for level in ("M0", "M1", "M2"):
            assert len(_cells(text, level)) == 120, level
        olmo = [c for (m, _g, _p), c in _cells(text, "M0").items() if m == OLMO]
        assert len(olmo) == 40 and not any(
            _kind(c) in ("number", "partial") for c in olmo
        )

    def test_a_rerun_and_a_shuffled_input_give_the_same_bytes(self, tmp_path):
        """
        Running twice, and running over shuffled row order, produce identical files.
        """
        first = _run(tmp_path)[1]
        assert _run(tmp_path)[1] == first
        root = _copy_root(tmp_path)
        for model in MODELS:
            _edit(root, model, lambda df: df.sample(fraction=1.0, shuffle=True, seed=7))
        shuffled = _run(tmp_path, root)[1]
        # The header carries each file's hash, which the shuffle changes; the table is what
        # must not move.
        assert _cells(shuffled) == _cells(first)

    def test_an_extra_column_is_ignored(self, tmp_path):
        """
        A later scorer adding a column (for example a dropped rate) changes no cell.
        """
        root = _copy_root(tmp_path)
        _edit(
            root,
            "qwen2.5-14b-instruct",
            lambda df: df.with_columns(pl.lit(0.0).alias("dropped_rate")),
        )
        assert _cells(_run(tmp_path, root)[1]) == _cells(_run(tmp_path)[1])

    def test_a_missing_required_column_exits_1_naming_it(self, tmp_path, capsys):
        """
        properties.csv without `status` exits 1 and names the column; the board file is
        untouched.
        """
        root = _copy_root(tmp_path)
        _edit(root, "qwen2.5-14b-instruct", lambda df: df.drop("status"))
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_ARM_PROBLEM and text == BOARD_FILE
        assert "status" in capsys.readouterr().err

    def test_a_partial_arm_exits_1_and_leaves_the_file_alone(self, tmp_path):
        """
        An arm missing outcomes.csv is refused: exit 1, file untouched.
        """
        root = _copy_root(tmp_path)
        (root / "qwen2.5-14b-instruct" / "outcomes.csv").unlink()
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_ARM_PROBLEM and text == BOARD_FILE

    def test_a_missing_data_dir_exits_1(self, tmp_path):
        """
        A run_meta.json with no data_dir says nothing about which data the numbers
        describe: exit 1, file untouched.
        """
        root = _copy_root(tmp_path)
        (root / "qwen2.5-14b-instruct" / "run_meta.json").write_text("{}\n")
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_ARM_PROBLEM and text == BOARD_FILE

    def test_the_wrong_number_of_arms_exits_4_and_writes_nothing(self, tmp_path):
        """
        Two arms against an expectation of three: exit 4, file untouched; `--expect-arms 2`
        accepts it and writes 80 cells.
        """
        root = _copy_root(tmp_path)
        shutil.rmtree(root / "qwen2.5-14b-instruct")
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_ARM_COUNT and text == BOARD_FILE
        code, text = _run(tmp_path, root, ["--expect-arms", "2"])
        assert code == cli.EXIT_OK and len(_cells(text)) == 80

    def test_check_with_the_wrong_expected_count_fails_and_the_right_one_passes(
        self, tmp_path
    ):
        """
        `--check` on a committed one-model board fails with exit 4 when told to expect three
        arms and passes with exit 0 when told to expect one, so the check can fail and is
        not satisfied by whatever the root holds.
        """
        root = _copy_root(tmp_path)
        shutil.rmtree(root / "qwen2.5-14b-instruct")
        shutil.rmtree(root / "mistral-nemo-instruct-2407")
        code, text = _run(tmp_path, root, ["--expect-arms", "1"])
        assert code == cli.EXIT_OK
        board_file = tmp_path / "board_file.md"
        board_file.write_text(text)
        base = [
            "--analysis-root",
            str(root),
            "--board-file",
            str(board_file),
            "--check",
            "--partial",
            "--roster-config",
            str(ROSTER_CONFIG),
        ]
        assert cli.main([*base, "--expect-arms", "3"]) == cli.EXIT_ARM_COUNT
        assert cli.main([*base, "--expect-arms", "1"]) == cli.EXIT_OK

    def test_omitting_expect_arms_exits_7_naming_the_flag_and_writes_nothing(
        self, tmp_path, capsys
    ):
        """
        With no `--expect-arms`, the write path and the `--check` path both exit 7 with a
        message naming the flag, and the board file is untouched.
        """
        board_file = tmp_path / "board_file.md"
        board_file.write_text(BOARD_FILE)
        for extra in ([], ["--check"]):
            code = cli.main(
                ["--analysis-root", str(FIXTURES), "--board-file", str(board_file), *extra]
            )
            assert code == cli.EXIT_NO_EXPECTATION
            assert "--expect-arms" in capsys.readouterr().err
        assert board_file.read_text() == BOARD_FILE

    def test_an_empty_root_exits_3(self, tmp_path):
        """
        A root with no arms exits 3.
        """
        (tmp_path / "empty").mkdir()
        code, text = _run(tmp_path, tmp_path / "empty")
        assert code == cli.EXIT_NO_ARMS and text == BOARD_FILE

    def test_a_board_file_without_markers_exits_2_untouched(self, tmp_path):
        """
        No markers: exit 2 and the file is exactly as it was.
        """
        code, text = _run(tmp_path, board_file_text="no markers here\n")
        assert code == cli.EXIT_MARKERS and text == "no markers here\n"

    def test_check_mode_flags_a_stale_block_and_never_writes(self, tmp_path):
        """
        `--check` exits 5 on a stale block without writing, and 0 once the block is fresh.
        """
        code, text = _run(tmp_path, extra=["--check"])
        assert code == cli.EXIT_STALE and text == BOARD_FILE
        _code, fresh = _run(tmp_path)
        code, text = _run(tmp_path, extra=["--check"], board_file_text=fresh)
        assert code == cli.EXIT_OK and text == fresh

    def test_a_footnote_file_beside_the_data_is_appended_below_the_table(
        self, tmp_path
    ):
        """
        The footnote lines come from footnotes.md in the analysis root, so the block is a
        function of its inputs and nothing is passed on the command line.
        """
        root = _copy_root(tmp_path)
        (root / cli.FOOTNOTES_FILE).write_text("note one\n\nnote two\n")
        _code, text = _run(tmp_path, root)
        inside = text.split("<!-- SCOREBOARD:BEGIN -->")[1].split(
            "<!-- SCOREBOARD:END -->"
        )[0]
        assert inside.rstrip().endswith("note one\nnote two")
        with pytest.raises(SystemExit):
            cli.parse_args(
                ["--analysis-root", str(root), "--footnote", "from the command line"]
            )

    def test_check_passes_on_a_written_block_and_fails_when_a_footnote_changes(
        self, tmp_path
    ):
        """
        `--check` is clean against the block the same inputs produced, and exits 5 once
        footnotes.md changes, which is what makes a committed block verifiable at all.
        """
        root = _copy_root(tmp_path)
        (root / cli.FOOTNOTES_FILE).write_text("the first note\n")
        _code, written = _run(tmp_path, root)
        code, _text = _run(tmp_path, root, extra=["--check"], board_file_text=written)
        assert code == cli.EXIT_OK
        (root / cli.FOOTNOTES_FILE).write_text("a different note\n")
        code, text = _run(tmp_path, root, extra=["--check"], board_file_text=written)
        assert code == cli.EXIT_STALE and text == written

    def test_check_sees_a_stale_cell_on_any_of_the_three_boards(self, tmp_path):
        """
        A block is stale when any board's table differs, not only the first: tampering with
        one cell row of each board in turn exits 5, and the untouched file exits 0 again.
        """
        _code, written = _run(tmp_path)
        assert _run(tmp_path, extra=["--check"], board_file_text=written)[0] == cli.EXIT_OK
        for level in ("M0", "M1", "M2"):
            row = next(
                line
                for line in written.splitlines()
                if line.startswith(f"| {level} | ")
            )
            tampered = written.replace(row, row[:-1] + "tampered |")
            assert tampered != written, level
            code, text = _run(tmp_path, extra=["--check"], board_file_text=tampered)
            assert code == cli.EXIT_STALE and text == tampered, level

    def test_a_root_without_a_footnote_file_writes_a_block_all_the_same(self, tmp_path):
        """
        A root with no footnotes.md produces a block with no footnotes, not a refusal.
        """
        root = _copy_root(tmp_path)
        assert not (root / cli.FOOTNOTES_FILE).exists()
        code, _text = _run(tmp_path, root)
        assert code == cli.EXIT_OK

    def test_a_dropped_opponent_reads_as_missing_through_the_real_entrypoint(
        self, tmp_path
    ):
        """
        Deleting every `rl` row from one arm leaves that arm's numbers reading over 3 of 4
        opponents with `rl no row` named, never a complete-looking 3 of 3.
        """
        root = _copy_root(tmp_path, rung="M0")
        _edit(
            root,
            "qwen2.5-14b-instruct",
            lambda df: df.filter(pl.col("opponent") != "rl"),
        )
        cell = _cells(_run(tmp_path, root)[1], "M0")[(QWEN, "kuhn", "P5")]
        assert "(3/4 opponents)" in cell and "rl no row" in cell

    def test_leading_null_values_do_not_cost_an_arm_its_numbers(self, tmp_path):
        """
        A properties.csv whose null-valued rows come first (which the default 100-row schema
        inference reads as text) still yields the same table as the unsorted file.
        """
        root = _copy_root(tmp_path)
        _edit(
            root, "qwen2.5-14b-instruct", lambda df: df.sort("value", nulls_last=False)
        )
        assert _cells(_run(tmp_path, root)[1]) == _cells(_run(tmp_path)[1])

    def test_a_non_numeric_value_column_exits_1_naming_it(self, tmp_path, capsys):
        """
        A `value` column that holds text on a non-empty table is refused, not read as a
        table of non-finite numbers.
        """
        root = _copy_root(tmp_path)
        _edit(
            root,
            "qwen2.5-14b-instruct",
            lambda df: df.with_columns(pl.lit("oops").alias("value")),
        )
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_ARM_PROBLEM and text == BOARD_FILE
        assert "value" in capsys.readouterr().err

    def test_a_missing_root_or_board_file_exits_6_not_1(self, tmp_path):
        """
        A nonexistent analysis root and a nonexistent board file both exit 6, apart from
        "an arm was refused" (1).
        """
        board_file = tmp_path / "board_file.md"
        board_file.write_text(BOARD_FILE)
        assert (
            cli.main(
                [
                    "--analysis-root",
                    str(tmp_path / "nope"),
                    "--expect-arms",
                    "3",
                    "--partial",
                    "--board-file",
                    str(board_file),
                ]
            )
            == cli.EXIT_INPUT
        )
        assert (
            cli.main(
                [
                    "--analysis-root",
                    str(FIXTURES),
                    "--expect-arms",
                    "3",
                    "--partial",
                    "--board-file",
                    str(tmp_path / "absent.md"),
                ]
            )
            == cli.EXIT_INPUT
        )

    def test_the_board_file_keeps_its_line_endings_mode_and_symlink(self, tmp_path):
        """
        A CRLF board file keeps every byte outside the markers, its permission bits, and a
        symlinked path stays a symlink with the target updated.
        """
        crlf = BOARD_FILE.replace("\n", "\r\n")
        real = tmp_path / "real.md"
        real.write_bytes(crlf.encode("utf-8"))
        os.chmod(real, 0o640)
        link = tmp_path / "link.md"
        link.symlink_to(real)
        code = cli.main(
            [
                "--analysis-root",
                str(FIXTURES),
                "--expect-arms",
                "3",
                "--partial",
                "--board-file",
                str(link),
            ]
        )
        assert code == cli.EXIT_OK
        assert link.is_symlink() and stat.S_IMODE(os.stat(real).st_mode) == 0o640
        data = real.read_bytes().decode("utf-8")
        assert data.startswith(
            "# tracker\r\n\r\nbefore the block\r\n<!-- SCOREBOARD:BEGIN -->"
        )
        assert data.endswith("<!-- SCOREBOARD:END -->\r\nafter the block\r\n")

    def test_output_is_identical_across_processes_with_different_hash_seeds(
        self, tmp_path
    ):
        """
        Three separate interpreter processes with different string-hash seeds write identical
        bytes, so no set or dict ordering leaks into a cell.
        """
        repo = Path(cli.__file__).resolve().parents[1]
        outputs = []
        for seed in (1, 2, 3):
            board_file = tmp_path / f"board_file{seed}.md"
            board_file.write_text(BOARD_FILE)
            env = dict(
                os.environ,
                PYTHONHASHSEED=str(seed),
                PYTHONPATH=f"{repo}:{repo / 'src'}",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "scoreboard.py"),
                    "--analysis-root",
                    str(FIXTURES),
                    "--expect-arms",
                    "3",
                    "--partial",
                    "--roster-config",
                    str(ROSTER_CONFIG),
                    "--board-file",
                    str(board_file),
                ],
                check=True,
                env=env,
                cwd=repo,
            )
            outputs.append(board_file.read_text())
        assert outputs[0] == outputs[1] == outputs[2]

    def test_the_script_installs_the_import_guard(self, monkeypatch):
        """
        Loading the script under a foreign checkout raises, so an ad-hoc run cannot score
        with another branch's code.
        """
        import lmopt._import_guard as guard

        monkeypatch.setattr(guard, "check_lmopt_is_local", lambda *a, **k: "foreign")
        spec = importlib.util.spec_from_file_location(
            "scoreboard_guard_probe", Path(cli.__file__)
        )
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(guard.ForeignCheckoutError):
            spec.loader.exec_module(module)


def test_headline_rows_do_not_mutate_the_shared_rule():
    """
    Grouping the headline rows leaves the project's DECISION_RULE untouched.
    """
    before = copy.deepcopy(analyse_stats.DECISION_RULE)
    sb.headline_rows(analyse_stats.DECISION_RULE)
    assert analyse_stats.DECISION_RULE == before


# ---------------------------------------------------------------------------
# The awaiting kind and the probe-sourced cells
# ---------------------------------------------------------------------------
import csv as _csvmod

import scripts.probe_scores as probe_cli
from lmopt.analysis.scoreboard import scoreboard_sources as sources
from tests._probe_arms import rows as _arm_rows
from tests._probe_arms import warm_cfr_cache, write_arm

PROBE_NAMES = ("P1_invariance", "P4_payoff_sensitivity", "P6_subgame_perfection")
_PBASE = {"0": [0, 1, 0, 1], "1": [0, 0, 1, 1], "0b": [0, 1, 0, 1]}
_PPROBE = {"0": [0, 1, 1, 1], "1": [0, 0, 1, 1], "0b": [0, 1, 0, 1]}
_PDRIVEN = {"0b": [0, 1, 0, 1], "1b": [0, 0, 1, 1]}


def _probe_row(
    game="kuhn",
    prop="P1_invariance",
    value=0.1,
    status="MEASURABLE",
    caveat="",
    n=3,
    coverage=1.0,
    n_seed_mean=None,
    coverage_seed_mean=None,
):
    """
    one probe_scores.csv row as the scoreboard reads it
    :param game: game key
    :param prop: property row name
    :param value: float or None
    :param status: status word
    :param caveat: caveat text
    :param n: info-set count (the cross-seed union pairing)
    :param coverage: paired coverage, or None (the cross-seed union pairing)
    :param n_seed_mean: the per-seed mean info-set count, or None for a row/artefact that
        carries no seed-mean columns at all
    :param coverage_seed_mean: the per-seed mean paired coverage, or None likewise
    :returns: a dict
    """
    return {
        "game": game,
        "property": prop,
        "value": value,
        "status": status,
        "caveat": caveat,
        "n_infosets": n,
        "paired_coverage": coverage,
        "n_infosets_seed_mean": n_seed_mean,
        "paired_coverage_seed_mean": coverage_seed_mean,
    }


class TestAwaitingKind:
    def test_awaiting_rows_read_awaiting_and_are_not_a_number_or_broken(self):
        """
        Four LM rows with status AWAITING_DATA, no value and a stated reason read `awaiting data:`
        with the reason, and the cell's kind is neither number nor BROKEN.
        """
        cell = _classify(
            _opponents([None] * 4, "AWAITING_DATA", "P2 needs at least 2000 hands")
        )
        assert cell.kind == sb.AWAITING
        assert cell.text == "awaiting data: P2 needs at least 2000 hands"

    def test_an_awaiting_row_with_no_reason_or_a_value_stays_broken(self):
        """
        AWAITING_DATA without a stated reason, or carrying a value, is not a collection-gap
        claim: it reads BROKEN, so awaiting cannot be used to hide a defect.
        """
        assert _classify(_opponents([None] * 4, "AWAITING_DATA", "")).kind == sb.BROKEN
        assert (
            _classify(_opponents([0.2] * 4, "AWAITING_DATA", "waiting")).kind
            == sb.BROKEN
        )

    def test_a_mixture_of_awaiting_and_blocked_rows_reads_broken_naming_both(self):
        """
        Two opponents awaiting and two refused: BROKEN, with both statuses in the reason.
        """
        rows = _opponents([None] * 4, "AWAITING_DATA", "waiting")
        rows[2]["status"] = rows[3]["status"] = "NOT_COMPUTED"
        cell = _classify(rows)
        assert cell.kind == sb.BROKEN
        assert "AWAITING_DATA" in cell.text and "NOT_COMPUTED" in cell.text

    def test_a_missing_opponent_row_keeps_an_awaiting_cell_from_reading_awaiting(self):
        """
        Three awaiting rows with the fourth opponent absent is not a complete collection-gap
        claim: BROKEN naming the missing row.
        """
        cell = _classify(_opponents([None] * 3, "AWAITING_DATA", "waiting"))
        assert cell.kind == sb.BROKEN and "no row for" in cell.text

    def test_a_cut_reason_is_marked_and_its_whole_text_is_kept(self):
        """
        A two-clause caveat shows its leading clause marked `(+1 more)` in the table text and
        both clauses in the cell's whole text, so nothing a scorer said is lost silently.
        """
        cell = _classify(
            _opponents([None] * 4, "AWAITING_DATA", "first clause; second clause")
        )
        assert cell.text == "awaiting data: first clause (+1 more)"
        assert cell.full_text == "awaiting data: first clause; second clause"
        assert cell.shortened

    def test_a_single_clause_reason_is_not_marked_and_is_not_shortened(self):
        """
        A caveat with one clause carries no marker and the cell reports itself unshortened,
        so the detail section holds only the cells that were actually cut.
        """
        cell = _classify(_opponents([None] * 4, "AWAITING_DATA", "only clause"))
        assert cell.text == "awaiting data: only clause"
        assert not cell.shortened

    def test_the_clause_split_is_one_function(self):
        """
        The caveat, the blocker and the probe reason all read their clauses from
        `split_clauses`, so a second splitter cannot drift from the first.
        """
        assert sb.split_clauses("a; b ;; c ") == ["a", "b", "c"]
        assert sb.split_clauses(None) == []
        assert sb.split_clauses("a | b") == ["a / b"]
        source = inspect.getsource(sb)
        assert source.count('.split(";")') == 1


class TestClassifyProbe:
    def test_a_measurable_p1_row_is_a_number_with_its_count_and_coverage(self):
        """
        MEASURABLE with a finite value: the number, the paired info-set count and the coverage.
        """
        cell = sb.classify_probe(
            "P1_invariance", _probe_row(value=0.0833, n=3, coverage=1.0), True
        )
        assert cell.kind == sb.NUMBER
        assert cell.text == (
            "0.0833 (3 paired info-sets, coverage 100.0%); dropped rate not recorded"
        )

    def test_a_p1_row_with_seed_mean_columns_shows_the_seed_mean_pair_beside_the_union_pair(
        self,
    ):
        """
        The headline `value` on a P1/P4 row is the within-seed estimate, so the cell must
        show the per-seed info-set count and coverage the value was actually formed over,
        not only the cross-seed union `n_infosets`/`paired_coverage` pair -- the union pair
        stays too, named as such, so a reader is never left guessing which pairing a number
        describes.
        """
        cell = sb.classify_probe(
            "P1_invariance",
            _probe_row(
                value=0.262375,
                n=116,
                coverage=0.522,
                n_seed_mean=82.0,
                coverage_seed_mean=0.396,
            ),
            True,
        )
        assert cell.kind == sb.NUMBER
        assert "82.0 per seed" in cell.text
        assert "116 union" in cell.text
        assert "39.6% per seed" in cell.text
        assert "52.2% union" in cell.text

    def test_a_p1_row_with_no_seed_mean_columns_falls_back_to_the_union_pair_alone(
        self,
    ):
        """
        An older artefact written before the seed-mean columns existed carries neither: the
        cell falls back to the plain union-pair rendering it always had, unchanged.
        """
        cell = sb.classify_probe(
            "P1_invariance", _probe_row(value=0.0833, n=3, coverage=1.0), True
        )
        assert cell.kind == sb.NUMBER
        assert "per seed" not in cell.text
        assert cell.text == (
            "0.0833 (3 paired info-sets, coverage 100.0%); dropped rate not recorded"
        )

    def test_a_row_below_its_t15_floor_reads_inadmissible_with_coverage_shown(self):
        """
        A row whose within-seed coverage falls below its pre-registered floor reads
        INADMISSIBLE -- the number and its coverage shown, the verdict withheld -- never a
        silent NUMBER.
        """
        cell = sb.classify_probe(
            "P1_invariance",
            _probe_row(
                value=0.262375,
                n=116,
                coverage=0.9,
                n_seed_mean=82.0,
                coverage_seed_mean=0.396,
            ),
            True,
            floor=0.5,
        )
        assert cell.kind == sb.INADMISSIBLE
        assert "0.2624" in cell.text
        assert "39.6%" in cell.text
        assert "paired coverage" in cell.text
        assert "below the pre-registered floor" in cell.text
        assert "50.0%" in cell.text

    def test_a_row_at_or_above_its_t15_floor_is_unaffected(self):
        cell = sb.classify_probe(
            "P1_invariance",
            _probe_row(
                value=0.262375,
                n=116,
                coverage=0.9,
                n_seed_mean=82.0,
                coverage_seed_mean=0.5,
            ),
            True,
            floor=0.5,
        )
        assert cell.kind == sb.NUMBER

    def test_a_row_with_no_seed_mean_coverage_is_never_checked_against_a_floor(self):
        """
        An artefact with no seed-mean columns at all has nothing of the right estimand to
        compare, so a floor must not be applied to the union pair instead.
        """
        cell = sb.classify_probe(
            "P1_invariance",
            _probe_row(value=0.0833, n=3, coverage=0.1),
            True,
            floor=0.5,
        )
        assert cell.kind == sb.NUMBER

    def test_no_floor_supplied_is_the_pre_t15_behaviour(self):
        cell = sb.classify_probe(
            "P1_invariance",
            _probe_row(
                value=0.0833,
                n=3,
                coverage=0.9,
                n_seed_mean=3.0,
                coverage_seed_mean=0.01,
            ),
            True,
        )
        assert cell.kind == sb.NUMBER

    def test_a_probe_number_shows_its_dropped_rate_and_never_defaults_it(self):
        """
        A probe row carrying a dropped rate shows it beside the number; a row carrying none
        says so, rather than reading as a clean zero.
        """
        with_rate = sb.classify_probe(
            "P1_invariance",
            dict(_probe_row(value=0.0833, n=3, coverage=1.0), dropped_rate=0.0125),
            True,
        )
        assert with_rate.text.endswith("; dropped 1.25%")
        without = sb.classify_probe(
            "P1_invariance", _probe_row(value=0.0833, n=3), True
        )
        assert without.text.endswith("; dropped rate not recorded")

    def test_a_p6_row_with_no_coverage_reads_its_off_path_count(self):
        """
        A P6 row (blank coverage) reads `(n off-path info-sets)`.
        """
        cell = sb.classify_probe(
            "P6_subgame_perfection",
            _probe_row(prop="P6_subgame_perfection", n=2, coverage=None),
            True,
        )
        assert cell.text.startswith("0.1000 (2 off-path info-sets)")

    def test_awaiting_data_reads_awaiting(self):
        """
        AWAITING_DATA with no value and a reason reads `awaiting data: <reason>`.
        """
        cell = sb.classify_probe(
            "P1_invariance",
            _probe_row(
                value=None,
                status="AWAITING_DATA",
                caveat="no leduc rows in the base arm",
            ),
            True,
        )
        assert (
            cell.kind == sb.AWAITING
            and cell.text == "awaiting data: no leduc rows in the base arm"
        )

    @pytest.mark.parametrize(
        "status,value",
        [
            ("NOT_COMPUTED", None),
            ("RECORDED_ABSENCE", None),
            ("SOMETHING_NEW", None),
            ("SOMETHING_NEW", 0.3),
            ("NOT_APPLICABLE", None),
        ],
    )
    def test_every_other_status_reads_broken_with_the_status_echoed(
        self, status, value
    ):
        """
        NOT_COMPUTED, RECORDED_ABSENCE, an unseen status (with or without a value) and
        NOT_APPLICABLE all read BROKEN: a probe cell is never structural and never a number
        without a finite value.
        """
        cell = sb.classify_probe(
            "P1_invariance", _probe_row(value=value, status=status, caveat="why"), True
        )
        assert cell.kind == sb.BROKEN and status in cell.text

    @pytest.mark.parametrize("value", [None, float("nan")])
    def test_a_measurable_probe_row_without_a_value_names_the_gate_not_the_status(
        self, value
    ):
        """
        A row the scorer called measurable that carries no finite value reads BROKEN naming
        the gate that stopped it, never the bare wording `MEASURABLE: <caveat>`.
        """
        cell = sb.classify_probe(
            "P1_invariance",
            _probe_row(value=value, status="MEASURABLE", caveat="why"),
            True,
        )
        assert cell.kind == sb.BROKEN
        assert cell.text == f"BROKEN: {sb.NO_FINITE_VALUE}"
        assert "MEASURABLE" not in cell.text

    @pytest.mark.parametrize(
        "row_kw,gate",
        [
            ({"value": 7.0}, "the value gate"),
            ({"n": 0}, "the info-set gate"),
            ({"coverage": 5.0}, "the coverage gate"),
        ],
    )
    def test_a_measurable_probe_row_stopped_by_a_range_check_names_that_gate(
        self, row_kw, gate
    ):
        """
        A measurable probe row rejected by a range check says which gate rejected it, so a
        reader is not left with `MEASURABLE: <problem>`.
        """
        cell = sb.classify_probe("P1_invariance", _probe_row(**row_kw), True)
        assert cell.kind == sb.BROKEN and cell.text.startswith(
            f"BROKEN: {gate} stopped it:"
        )

    @pytest.mark.parametrize("count", [None, float("nan"), -1.0, 2.5])
    def test_a_measurable_row_without_a_valid_count_reads_broken(self, count):
        """
        A MEASURABLE probe row whose info-set count is missing, NaN, negative or fractional is not
        a measurement the cell can describe: BROKEN, never `None paired info-sets`.
        """
        cell = sb.classify_probe("P1_invariance", _probe_row(n=count), True)
        assert cell.kind == sb.BROKEN and "count" in cell.text

    @pytest.mark.parametrize(
        "row_kw",
        [
            {"value": 0.3, "status": "AWAITING_DATA", "caveat": "waiting"},
            {"value": None, "status": "AWAITING_DATA", "caveat": ""},
            {"n": 0},
            {"coverage": 5.0},
            {"coverage": float("nan")},
            {"value": 7.0},
            {"value": -0.5},
        ],
    )
    def test_a_probe_row_that_cannot_be_a_measurement_reads_broken(self, row_kw):
        """
        AWAITING_DATA carrying a value, or with no stated reason, is not a collection-gap claim;
        a MEASURABLE row with a zero count, a coverage outside [0, 1] (or NaN), or a value outside
        [0, 1] (a total variation cannot be) reads BROKEN, never a number or awaiting.
        """
        cell = sb.classify_probe("P1_invariance", _probe_row(**row_kw), True)
        assert cell.kind == sb.BROKEN

    def test_an_unknown_status_is_made_safe_for_a_table_cell(self):
        """
        A hand-edited status containing a pipe and a newline cannot break the markdown table
        row: it is flattened like any other clause.
        """
        cell = sb.classify_probe(
            "P1_invariance", _probe_row(value=None, status="A|B\nC"), True
        )
        assert "|" not in cell.text and "\n" not in cell.text

    def test_a_missing_row_and_a_missing_file_read_broken(self):
        """
        A present file with no row for the game reads BROKEN, and an absent file reads awaiting:
        the first is a row the scorer should have written, the second an input never supplied.
        """
        assert sb.classify_probe("P1_invariance", None, False).kind == sb.AWAITING
        assert sb.classify_probe("P1_invariance", None, True).kind == sb.BROKEN
        assert sb.classify_probe("P1_invariance", None, True).text == (
            "BROKEN: no P1_invariance probe score for this game"
        )
        assert (
            sb.classify_probe("P1_invariance", None, False).text
            == sb.PROBE_FILE_MISSING
        )


def _play_frame(actions, *, game="leduc", seed=42):
    """
    shard rows carrying everything a fingerprint and a first-listed share need
    :param actions: the parsed action of each row, in node order
    :param game: the game key
    :param seed: the shard seed
    :returns: a pl.DataFrame in the shard schema, LM rows only
    """
    return pl.DataFrame(
        [
            {
                "game": game,
                "seed": seed,
                "hand_idx": i // 2,
                "node_idx": i % 2,
                "player_to_act": 0,
                "info_set": f"s{i % 3}",
                "parsed_action": action,
                "legal_mask": [1, 1, 1],
                "prompt": "p",
                "fallback_used": False,
            }
            for i, action in enumerate(actions)
        ]
    )


def _collapse_file(arm_dir, rows):
    """
    write one arm's collapse record the way the pass writes it
    :param arm_dir: the analysis arm directory
    :param rows: dicts with the collapse columns
    :returns: None
    """
    with open(Path(arm_dir) / sb.COLLAPSE_FILE, "w", newline="") as handle:
        writer = _csvmod.DictWriter(handle, fieldnames=list(sb.COLLAPSE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _collapse_row(
    game="leduc", fingerprint="abc", collapsed_with="", share="1.0", n_rows=10
):
    """
    one collapse.csv row
    :param game: game key
    :param fingerprint: the play digest
    :param collapsed_with: comma-separated other arms sharing it
    :param share: the first-listed legal action share as written
    :param n_rows: the rows behind the share
    :returns: a dict
    """
    return {
        "game": game,
        "fingerprint": fingerprint,
        "collapsed_with": collapsed_with,
        "first_listed_share": share,
        "n_rows": n_rows,
    }


class TestConditionSelection:
    def _two_conditions(self, tmp_path, second_rung):
        """
        a root where one model has two arms: the fixture qwen arm and a copy at another
        memory level, both recording the same model
        :param tmp_path: pytest tmp_path
        :param second_rung: the memory level the copy records
        :returns: the root
        """
        root = _copy_root(tmp_path, rung="M0")
        twin = root / "qwen-second"
        shutil.copytree(root / "qwen2.5-14b-instruct", twin)
        _set_rung(root, "qwen-second", second_rung)
        return root

    def test_each_board_reads_its_own_level_and_carries_no_tail(self, tmp_path):
        """
        A model with an M0 and an M1 arm gets its own number on each board, and neither
        cell mentions the other level: the value of a level lives on that level's board.
        """
        root = self._two_conditions(tmp_path, "M1")
        cells = _all_cells(_run(tmp_path, root, extra=["--expect-arms", "4"])[1])
        for level in ("M0", "M1"):
            cell = cells[(level, QWEN, "kuhn", "P5")]
            assert _kind(cell) == "number", (level, cell)
            assert not re.search(r"; M\d ", cell), (level, cell)

    def test_two_arms_of_one_model_at_two_levels_each_feed_only_their_own_board(
        self, tmp_path
    ):
        """
        One M0 arm and one M1 arm of the same model, same temperature and prompt schema
        version, sit in one root and carry different values. The scoreboard groups them into
        one model entry (`group_by_epoch` keys on temperature and prompt version, not memory
        level), so the only thing keeping the boards apart is `sources.headline_arms`
        keeping an arm only when its recorded `memory_rung` equals the board's level. The M0
        board must read the original arm's cell unchanged and the M1 board a different one.
        """
        (tmp_path / "alone").mkdir()
        (tmp_path / "pair").mkdir()
        alone = _all_cells(
            _run(tmp_path / "alone", _copy_root(tmp_path / "alone", rung="M0"))[1]
        )
        root = self._two_conditions(tmp_path / "pair", "M1")
        shifted = (
            (pl.col("game") == "kuhn")
            & pl.col("property").str.starts_with("P5")
            & (pl.col("status") == "MEASURABLE")
        )
        _edit(
            root,
            "qwen-second",
            lambda df: df.with_columns(
                [
                    pl.when(shifted).then(pl.col(c) + 0.3).otherwise(pl.col(c)).alias(c)
                    for c in ("value", "ci_lo", "ci_hi")
                ]
            ),
        )
        cells = _all_cells(
            _run(tmp_path / "pair", root, extra=["--expect-arms", "4"])[1]
        )
        key = (QWEN, "kuhn", "P5")
        assert cells[("M0",) + key] == alone[("M0",) + key]
        assert _kind(cells[("M1",) + key]) == "number"
        assert cells[("M1",) + key] != cells[("M0",) + key]

    def test_a_board_with_no_arm_waits_and_never_borrows_another_level(self, tmp_path):
        """
        With only an M1 arm the M0 board has nothing to read, so its P5 cell reads awaiting
        data naming the M0 run and shows no value from anywhere else.
        """
        cells = _all_cells(_run(tmp_path)[1])
        cell = cells[("M0", QWEN, "kuhn", "P5")]
        assert _kind(cell) == "awaiting"
        assert cell == "awaiting data: this model has no M0 run"
        assert _kind(cells[("M1", QWEN, "kuhn", "P5")]) == "number"

    def test_a_level_where_the_property_cannot_exist_is_structural_with_no_arm(
        self, tmp_path
    ):
        """
        P2 on the M0 board states why it cannot exist there whether or not the model has an
        M0 arm, and never reports that arm's P2 rows as a defect.
        """
        root = _copy_root(tmp_path, rung="M0")
        with_arm = _all_cells(_run(tmp_path, root)[1])[("M0", QWEN, "kuhn", "P2")]
        without_arm = _all_cells(_run(tmp_path)[1])[("M0", QWEN, "kuhn", "P2")]
        assert _kind(with_arm) == _kind(without_arm) == "structural"
        assert with_arm == without_arm
        assert "BROKEN" not in with_arm

    def test_p2_reads_the_board_level_and_needs_its_hands_at_each(self, tmp_path):
        """
        P2 is declared once for M1 and M2, and the selection is asked per board: an M1 arm
        answers the M1 board and never the M2 board, and neither answers M0.
        """
        conditions = {
            "a": {"memory_rung": "M1", "n_hands": 2000, "probe_type": "none"},
            "b": {"memory_rung": "M2", "n_hands": 2000, "probe_type": "none"},
        }
        assert sources.headline_arms(2, conditions, "M1", 1000) == ["a"]
        assert sources.headline_arms(2, conditions, "M2", 1000) == ["b"]
        assert sources.headline_arms(2, conditions, "M0", 1000) == []
        root = _copy_root(tmp_path, rung="M1")
        cells = _all_cells(_run(tmp_path, root)[1])
        assert _kind(cells[("M2", QWEN, "kuhn", "P2")]) == "awaiting"
        assert "no M2 run with enough hands" in cells[("M2", QWEN, "kuhn", "P2")]

    def test_two_arms_matching_one_condition_read_broken_naming_both(self, tmp_path):
        """
        Two arms of one model at the same memory level is an ambiguity the board refuses to
        resolve: the cell reads BROKEN and names both arm directories.
        """
        root = self._two_conditions(tmp_path, "M0")
        cell = _cells(_run(tmp_path, root, extra=["--expect-arms", "4"])[1], "M0")[
            (QWEN, "kuhn", "P5")
        ]
        assert _kind(cell) == "BROKEN"
        assert "qwen-second" in cell and "qwen2.5-14b-instruct" in cell

    def _set_epoch(self, root, arm, temperature, version):
        """
        record a temperature and prompt schema version on one arm's condition block
        :param root: the analysis root
        :param arm: the arm directory name
        :param temperature: the temperature to record
        :param version: the prompt schema version to record
        :returns: None
        """
        path = Path(root) / arm / "run_meta.json"
        meta = json.loads(path.read_text())
        meta["condition"] = dict(
            meta["condition"], temperature=temperature, prompt_schema_version=version
        )
        path.write_text(json.dumps(meta))

    def test_a_v7_and_a_v8_run_of_one_model_at_one_level_score_side_by_side(
        self, tmp_path
    ):
        """
        Two arms of one model at one memory level that differ in temperature and prompt
        schema version are two rows of the board, each reading its own arm, and neither reads
        the ambiguity BROKEN. (The fixture slices carry no probe_scores.csv or P3 rows, so
        some cells read BROKEN for that reason on a single arm too.) Through the real
        scoreboard CLI.
        """
        root = self._two_conditions(tmp_path, "M0")
        self._set_epoch(root, "qwen2.5-14b-instruct", 0.0, 7)
        self._set_epoch(root, "qwen-second", 1.0, 8)
        cells = _cells(_run(tmp_path, root, extra=["--expect-arms", "4"])[1], "M0")
        v7 = f"{QWEN} [temperature 0.0, prompt v7]"
        v8 = f"{QWEN} [temperature 1.0, prompt v8]"
        assert _kind(cells[(v7, "kuhn", "P5")]) == "number", cells[(v7, "kuhn", "P5")]
        assert _kind(cells[(v8, "kuhn", "P5")]) == "number", cells[(v8, "kuhn", "P5")]
        assert not any("arms match the condition" in c for c in cells.values())
        assert (QWEN, "kuhn", "P5") not in cells

    def _two_epoch_header(self, tmp_path, mode):
        """
        the header line(s) of a root holding a v7 and a v8 run of Qwen at M0
        :param tmp_path: pytest tmp_path
        :param mode: the --epoch-denominator value, or None for the default (str | None)
        :returns: (first header line, every header line before the arm table) (tuple)
        """
        root = self._two_conditions(tmp_path, "M0")
        self._set_epoch(root, "qwen2.5-14b-instruct", 0.0, 7)
        self._set_epoch(root, "qwen-second", 1.0, 8)
        extra = ["--expect-arms", "4"] + (
            [] if mode is None else ["--epoch-denominator", mode]
        )
        text = _run(tmp_path, root, extra=extra)[1]
        block = text.split("<!-- SCOREBOARD:BEGIN -->")[1].split("| arm |")[0]
        lines = [line for line in block.splitlines() if line.strip()]
        return lines[0], lines

    def _fraction(self, line):
        """
        :param line: a header line carrying "N of M expected cells" (str)
        :returns: (N, M) (tuple[int, int])
        """
        n, m = re.search(r"(\d+) of (\d+) expected cells", line).groups()
        return int(n), int(m)

    def test_by_default_a_two_epoch_header_prints_one_line_per_epoch(self, tmp_path):
        """
        The default is 'per-epoch': the header given no flag is byte-identical to the one
        given `--epoch-denominator per-epoch`, and no line carries a numerator above its
        denominator.
        """
        default = self._two_epoch_header(tmp_path, None)[1]
        assert default == self._two_epoch_header(tmp_path / "p", "per-epoch")[1]
        assert "expected cells" not in default[0]
        fractions = [
            self._fraction(line) for line in default if line.startswith("Epoch [")
        ]
        assert fractions and all(n <= m for n, m in fractions)

    def test_single_keeps_one_epoch_per_model_so_a_two_epoch_root_reads_over_one(
        self, tmp_path
    ):
        """
        'single' is still selectable and counts one epoch per model, so a two-epoch root reads
        a numerator above its denominator.
        """
        n, m = self._fraction(self._two_epoch_header(tmp_path, "single")[0])
        assert n > m

    def test_scale_counts_one_full_set_of_cells_per_model_and_epoch_row(self, tmp_path):
        """
        'scale': the denominator is a full set of cells per (model, epoch) row found plus one per
        roster model with no arm, so the fraction never exceeds one.
        """
        n_default, m_default = self._fraction(
            self._two_epoch_header(tmp_path, "single")[0]
        )
        first, _lines = self._two_epoch_header(tmp_path / "s", "scale")
        n, m = self._fraction(first)
        assert n == n_default and n <= m and m > m_default

    def test_per_epoch_prints_one_line_per_epoch_each_within_its_denominator(
        self, tmp_path
    ):
        """
        'per-epoch': the first line drops the denominator and one line per epoch follows, each
        against a full roster, and the per-epoch counts sum to the board's total.
        """
        n_default, _m = self._fraction(self._two_epoch_header(tmp_path, "single")[0])
        first, lines = self._two_epoch_header(tmp_path / "p", "per-epoch")
        assert "expected cells" not in first
        epoch_lines = [line for line in lines if line.startswith("Epoch [")]
        # Qwen's two runs, plus the fixture's other two models, whose condition blocks record
        # no epoch and so share the "not recorded" one.
        assert len(epoch_lines) == 3
        assert any("[temperature 0.0, prompt v7]" in line for line in epoch_lines)
        assert any("[temperature 1.0, prompt v8]" in line for line in epoch_lines)
        fractions = [self._fraction(line) for line in epoch_lines]
        assert all(n <= m for n, m in fractions)
        assert sum(n for n, _m in fractions) == n_default

    def test_a_row_says_not_recorded_for_an_epoch_field_the_block_lacks(self):
        """
        An arm whose block predates the two fields sits beside a recorded one as its own row,
        and its label says the fields were not recorded instead of printing None.
        """
        staged = [("m", (None, None), "old"), ("m", (0.0, 7), "new")]
        assert cli.group_by_epoch(staged) == {
            "m [temperature not recorded, prompt not recorded]": ["old"],
            "m [temperature 0.0, prompt v7]": ["new"],
        }

    def test_two_runs_of_one_epoch_still_read_broken(self, tmp_path):
        """
        Recording the same temperature and version on both arms leaves them one epoch, so
        the ambiguity is still refused and the row keeps its plain model name.
        """
        root = self._two_conditions(tmp_path, "M0")
        self._set_epoch(root, "qwen2.5-14b-instruct", 0.0, 7)
        self._set_epoch(root, "qwen-second", 0.0, 7)
        cell = _cells(_run(tmp_path, root, extra=["--expect-arms", "4"])[1], "M0")[
            (QWEN, "kuhn", "P5")
        ]
        assert _kind(cell) == "BROKEN"

    def test_an_arm_with_no_condition_block_reads_broken_on_every_board(self, tmp_path):
        """
        An analysis output that does not record its condition cannot be placed on any board:
        all 40 of its cells on each of the three read BROKEN naming that, and none reads M0.
        """
        root = _copy_root(tmp_path)
        path = root / "qwen2.5-14b-instruct" / "run_meta.json"
        meta = json.loads(path.read_text())
        del meta["condition"]
        path.write_text(json.dumps(meta))
        cells = _all_cells(_run(tmp_path, root)[1])
        orphan = {k: v for k, v in cells.items() if k[1] == "qwen2.5-14b-instruct"}
        assert len(orphan) == 120
        for level in ("M0", "M1", "M2"):
            assert sum(1 for k in orphan if k[0] == level) == 40
        assert all(_kind(v) == "BROKEN" for v in orphan.values())
        assert all("does not record its condition" in v for v in orphan.values())
        assert QWEN not in {k[1] for k in cells}

    def test_the_backfill_updates_only_the_blocks_it_wrote_itself(
        self, tmp_path, monkeypatch
    ):
        """
        `--update-backfilled` rewrites a block an earlier backfill wrote, so a field added
        later reaches it, and leaves a block the scoring run wrote exactly as it is.
        """
        root = _copy_root(tmp_path)
        _synthetic_collection_run_metas(root, tmp_path, monkeypatch)
        scored = root / "olmo-2-1124-13b-instruct" / "run_meta.json"
        meta = json.loads(scored.read_text())
        meta["condition"] = dict(meta["condition"], source="analyse_stats", n_hands=99)
        scored.write_text(json.dumps(meta))
        stale = root / "qwen2.5-14b-instruct" / "run_meta.json"
        was = json.loads(stale.read_text())
        was["condition"] = dict(was["condition"], n_hands=99)
        stale.write_text(json.dumps(was))
        assert (
            backfill_cli.main(["--analysis-root", str(root), "--update-backfilled"])
            == backfill_cli.EXIT_OK
        )
        assert json.loads(stale.read_text())["condition"]["n_hands"] == 2000
        assert json.loads(scored.read_text())["condition"]["n_hands"] == 99

    def test_the_condition_counts_the_pieces_a_composed_arm_was_built_from(
        self, tmp_path
    ):
        """
        `n_seeds` is one piece's seed count, so the block carries `n_pieces` beside it and
        a composed arm cannot be read as a single replicate.
        """
        from lmopt.analysis.condition import condition_block

        composed = {
            "composed_from": ["a", "b", "c", "d"],
            "resolved_config": {"n_seeds": 1},
        }
        assert condition_block(composed, "x")["n_pieces"] == 4
        assert condition_block(composed, "x")["n_seeds"] == 1
        assert (
            condition_block({"resolved_config": {"n_seeds": 5}}, "x")["n_pieces"] == 1
        )
        block = json.loads(
            (FIXTURES / "qwen2.5-14b-instruct" / "run_meta.json").read_text()
        )["condition"]
        assert (block["n_seeds"], block["n_pieces"]) == (1, 10)

    def test_the_backfill_writes_a_condition_and_refuses_to_overwrite_one(
        self, tmp_path
    ):
        """
        Production anchor for the condition block: the backfill script writes one from the
        collection run_meta the analysis output names, leaves an existing one alone, and
        refuses an arm whose collection run_meta cannot be read.
        """
        root = _copy_root(tmp_path)
        original = json.loads(
            (root / "qwen2.5-14b-instruct" / "run_meta.json").read_text()
        )["condition"]
        assert backfill_cli.main(["--analysis-root", str(root)]) == backfill_cli.EXIT_OK
        assert (
            json.loads((root / "qwen2.5-14b-instruct" / "run_meta.json").read_text())[
                "condition"
            ]
            == original
        )
        path = root / "qwen2.5-14b-instruct" / "run_meta.json"
        path.write_text(json.dumps({"data_dir": "no/such/directory"}))
        assert backfill_cli.main(["--analysis-root", str(root)]) == (
            backfill_cli.EXIT_ARM_PROBLEM
        )
        assert "condition" not in json.loads(path.read_text())


class TestCellKinds:
    """
    BROKEN is reserved for an instrument fault. Each other outcome the scorer or the gate
    produces deliberately has a kind that names it: a not-reached structural cell, an
    inadmissible number with its verdict withheld, and awaiting for an input never supplied.
    """

    S7 = (
        "at least one legal-mask bin in this cell is action-constant despite genuine "
        "within-seat state variation (S7 pre-registered cell (b))"
    )

    def _flawed(self, blocker):
        """
        LM rows every opponent of which the gate blocked on one stated defect
        :param blocker: the caveat every row carries
        :returns: the four rows
        """
        return _opponents([0.25, 0.26, 0.24, 0.27], "MEASURABLE_FLAWED", blocker)

    def _legacy(self, blocker):
        """
        the same rows as an arm scored before `partial_reason_class` existed carries them
        :param blocker: the caveat every row carries
        :returns: the four rows, with the `partial_reason_class` key absent
        """
        return [
            {k: v for k, v in r.items() if k != "partial_reason_class"}
            for r in self._flawed(blocker)
        ]

    def test_a_not_reached_row_set_is_structural_and_says_it_is_not_by_design(self):
        """
        Every opponent's row is NOT_REACHED with a stated reason: the cell is structural, its
        text names the kind, and the reason is read verbatim.
        """
        reason = "n_informative=0 of n_nodes=25193 real scored rows"
        cell = _classify(_opponents([None] * 4, "NOT_REACHED", reason))
        assert cell.kind == sb.STRUCTURAL
        assert cell.text == sb.NOT_REACHED_PREFIX + reason
        assert not cell.text.startswith("structural: ")

    def test_the_real_entrypoint_renders_a_not_reached_dark_hex_cell_as_structural(
        self, tmp_path
    ):
        """
        Production anchor: rewrite every Dark Hex P8 row of one arm the way the scorer
        emits a zero-informative cell (NOT_REACHED, no value, the reason as caveat), run the
        real CLI, and read that cell off the rendered board: structural, naming its kind,
        counted as structural and never as BROKEN.
        """
        root = _copy_root(tmp_path, rung="M1")
        reason = "n_informative=0 of n_nodes=25193 real scored rows"

        def reached_none(df):
            hit = (pl.col("game") == "darkhex") & pl.col("property").str.starts_with(
                "P8"
            )
            return df.with_columns(
                pl.when(hit)
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(pl.col("value"))
                .alias("value"),
                pl.when(hit)
                .then(pl.lit("NOT_REACHED"))
                .otherwise(pl.col("status"))
                .alias("status"),
                pl.when(hit)
                .then(pl.lit(reason))
                .otherwise(pl.col("caveat"))
                .alias("caveat"),
            )

        _edit(root, "qwen2.5-14b-instruct", reached_none)
        cell = _cells(_run(tmp_path, root)[1], "M1")[(QWEN, "darkhex", "P8")]
        assert cell == sb.NOT_REACHED_PREFIX + reason
        assert _kind(cell) == "structural"

    def test_a_not_reached_cell_is_never_borrowed_by_another_board(self):
        """
        A cell that is a fact about one arm's play is not offered to a level with no arm as if
        the property could not exist there.
        """
        reached = sb.Cell(sb.STRUCTURAL, sb.NOT_REACHED_PREFIX + "x")
        by_design = sb.Cell(sb.STRUCTURAL, "structural: y")
        arm = lambda cell: sb.ArmInput(
            "a", {"memory_rung": "M0"}, {("darkhex", 8): cell}, None
        )
        never = lambda number, level: ""
        wait = lambda number, level: "M1 run"
        assert sb._absent_headline(
            8, "darkhex", "M1", [arm(reached)], never, wait
        ).kind == (sb.AWAITING)
        assert sb._absent_headline(
            8, "darkhex", "M1", [arm(by_design)], never, wait
        ) == (by_design)

    def test_not_reached_needs_every_row_and_a_reason(self):
        """
        A missing opponent row, a row with no reason, or a row of another status keeps the
        cell BROKEN: only a complete set of stated NOT_REACHED rows is structural.
        """
        reason = "n_informative=0 of n_nodes=10 real scored rows"
        rows = _opponents([None] * 4, "NOT_REACHED", reason)
        assert _classify(rows[:3]).kind == sb.BROKEN
        assert _classify(_opponents([None] * 4, "NOT_REACHED", "")).kind == sb.BROKEN
        mixed = rows[:3] + [
            _row(
                "kuhn",
                OPPONENT_LADDER[3],
                "P3_randomisation_margin",
                None,
                "MEASURABLE",
                reason,
            )
        ]
        assert _classify(mixed).kind == sb.BROKEN

    def test_a_preregistered_blocker_reads_inadmissible_with_the_number_and_the_reason(
        self,
    ):
        """
        Every opponent scored and the gate blocked all four on the pre-registered defect: the
        cell shows the value, says the verdict is withheld, and states the reason.
        """
        cell = _classify(self._legacy(self.S7))
        assert cell.kind == sb.INADMISSIBLE
        assert cell.text.startswith(
            "inadmissible: value 0.2550 shown, verdict withheld: "
        )
        assert "S7 pre-registered cell" in cell.full_text

    def test_a_failed_state_sensitivity_check_reads_inadmissible_not_broken(self):
        """
        Regression: `_partial_reason_class` rewrites a state-sensitivity-failure caveat to
        `vocabulary.FAILED_REASON`'s canonical text, which does not match
        `PREREGISTERED_BLOCKERS`'s old "S7 pre-registered cell" string -- without the
        `FAILED_REASON` entry, real cells with the exact text this test uses fall from
        `inadmissible` to `BROKEN`.
        Uses `vocabulary.FAILED_REASON` itself, not a copy of its text, so a future
        rename of the constant cannot silently re-break this test into false confidence.
        """
        cell = _classify(self._legacy(vocabulary.FAILED_REASON))
        assert cell.kind == sb.INADMISSIBLE
        assert cell.text.startswith(
            "inadmissible: value 0.2550 shown, verdict withheld: "
        )

    def test_preregistered_blockers_recognises_analyse_stats_own_failed_reason(self):
        """
        Standing invariant, cheap to keep true: `vocabulary.FAILED_REASON`'s literal
        text is a substring of at least one entry in `sb.PREREGISTERED_BLOCKERS`. Enumerates
        the cross-file agreement the two constants must hold, per the project's own
        preference for a standing test over relying on a future reviewer noticing a rename
        on one side and not the other.
        """
        assert any(
            vocabulary.FAILED_REASON in marker for marker in sb.PREREGISTERED_BLOCKERS
        )

    def test_a_failed_class_decides_inadmissible_even_when_the_text_is_reworded(self):
        """
        The cell is INADMISSIBLE because the rows carry `partial_reason_class == "failed"`, not
        because the blocker sentence contains a known substring: rewording the sentence cannot
        flip it to BROKEN again.
        """
        rows = _opponents(
            [0.25, 0.26, 0.24, 0.27],
            "MEASURABLE_FLAWED",
            "a sentence nobody pre-registered",
        )
        for r in rows:
            r["partial_reason_class"] = "failed"
        assert _classify(rows).kind == sb.INADMISSIBLE

    def test_a_row_set_with_no_class_key_still_reads_inadmissible_through_the_legacy_path(
        self,
    ):
        """
        An arm scored before `partial_reason_class` existed has no such key on its rows; its
        pre-registered blocker is recognised by the old text rule, and the cell says so.
        """
        cell = _classify(self._legacy(vocabulary.FAILED_REASON))
        assert cell.kind == sb.INADMISSIBLE
        assert cell.legacy_blocker_rule is True

    def test_the_structured_path_does_not_flag_the_legacy_rule(self):
        rows = self._flawed(vocabulary.FAILED_REASON)
        for r in rows:
            r["partial_reason_class"] = "failed"
        cell = _classify(rows)
        assert cell.kind == sb.INADMISSIBLE
        assert cell.legacy_blocker_rule is False

    def test_the_board_names_the_legacy_rule_on_stderr_exactly_when_it_is_used(
        self, capsys
    ):
        """
        build_cells prints one stderr line naming the count of cells classified by the text rule
        and that the arm predates `partial_reason_class`; an arm whose rows carry the key prints
        nothing.
        """

        def build(rows):
            return sb.build_cells(
                "qwen2.5-14b-instruct",
                rows,
                analyse_stats.DECISION_RULE,
                analyse_stats.decision_table,
                cli.structural_by_design,
                OPPONENT_LADDER,
            )

        build(self._legacy(vocabulary.FAILED_REASON))
        err = capsys.readouterr().err
        assert "1 cell(s)" in err and "partial_reason_class" in err

        modern = self._flawed(vocabulary.FAILED_REASON)
        for r in modern:
            r["partial_reason_class"] = "failed"
        build(modern)
        assert capsys.readouterr().err == ""

    def test_a_blocker_no_one_pre_registered_stays_broken(self):
        """
        The same blocked value with a defect the pre-registration does not name is still
        BROKEN: inadmissible is a decision about a named defect, not a bucket for every block.
        """
        cell = _classify(self._flawed("the scorer returned a value outside its range"))
        assert cell.kind == sb.BROKEN
        assert "value 0.2550" in cell.text

    def test_a_missing_opponent_row_makes_an_inadmissible_cell_broken(self):
        """
        A row the scorer should have written and did not is an instrument fault even when the
        rows present were blocked on the pre-registered defect.
        """
        assert _classify(self._flawed(self.S7)[:3]).kind == sb.BROKEN

    def test_two_inadmissible_parts_of_one_property_combine_to_inadmissible(self):
        """
        P7 has two headline rows: two inadmissible parts read inadmissible with both values,
        and an inadmissible part beside a BROKEN one reads BROKEN.
        """
        a = sb.Cell(
            sb.INADMISSIBLE, "inadmissible: value 0.2 shown, verdict withheld: r"
        )
        b = sb.Cell(
            sb.INADMISSIBLE, "inadmissible: value 0.4 shown, verdict withheld: r"
        )
        both = sb.combine(("7a", "7b"), (a, b))
        assert both.kind == sb.INADMISSIBLE
        assert "7a value 0.2" in both.text and "7b value 0.4" in both.text
        assert (
            sb.combine(("7a", "7b"), (a, sb.Cell(sb.BROKEN, "BROKEN: x"))).kind
            == sb.BROKEN
        )

    def test_the_tally_counts_inadmissible_and_refuses_an_unknown_kind(self):
        grid = [sb.GridCell("m", "kuhn", 5, sb.Cell(sb.INADMISSIBLE, "x"))]
        assert sb._kind_counts(grid)[sb.INADMISSIBLE] == 1
        with pytest.raises(ValueError, match="does not count"):
            sb._kind_counts([sb.GridCell("m", "kuhn", 5, sb.Cell("mystery", "x"))])


class TestReasonsReachTheCell:
    """
    A scorer writes a reason to say why a number is qualified or withheld; the board must show
    it. Three shapes of the same failure: a real clause cut into `(+N more)`, a verdict reason
    written to decision.csv and never read, and a probe row's reason columns read on one branch
    and ignored on another.
    """

    ZERO = "zero between-seed variance: 1 distinct value across 10 seeds"

    def test_a_protected_clause_is_shown_in_full_and_never_counted_as_cut(self):
        """
        With three clauses and the zero-variance one last, the short form leads with it and
        counts only the ordinary clause it does not show; a caveat without it reads as before.
        """
        assert (
            sb._clause(f"first; second; {self.ZERO}") == f"{self.ZERO}; first (+1 more)"
        )
        assert sb._clause(self.ZERO) == self.ZERO
        assert sb._clause("first; second") == "first (+1 more)"
        assert sb._clause(f"first; {self.ZERO}") == f"{self.ZERO}; first"

    def test_the_scorers_own_clause_opens_with_the_protected_prefix(self):
        """
        `analyse_stats` writes the clause as its own literal, so the prefix the board protects
        must stay the prefix that literal opens with.
        """
        import inspect

        source = inspect.getsource(analyse_stats.property_table)
        assert f'f"{sb.ZERO_VARIANCE_PREFIX}: 1 distinct value across "' in source
        assert sb.replicate_clause(10, 1, 0.0).startswith(sb.ZERO_VARIANCE_PREFIX)

    def test_a_verdict_reason_reaches_the_headline_cell_in_both_forms(self):
        """
        The decision row's `verdict_reason` is rendered beside the verdict, capped in the table
        form and whole in the detail form.
        """
        rows = _opponents([0.1, 0.2, 0.3, 0.4])
        decision = {d["property"]: d for d in analyse_stats.decision_table(rows)}[
            "P3_randomisation_margin"
        ]
        reason = "interval envelope straddles the fail threshold; " + "x" * 200
        cell = sb.classify(
            "P3_randomisation_margin",
            dict(decision, verdict_reason=reason),
            rows,
            False,
            OPPONENT_LADDER,
        )
        assert cell.kind == sb.NUMBER
        assert (
            "verdict reason: interval envelope straddles the fail threshold"
            in cell.text
        )
        assert "x" * 200 in cell.full_text and "x" * 200 not in cell.text
        bare = sb.classify(
            "P3_randomisation_margin",
            dict(decision, verdict_reason=None),
            rows,
            False,
            OPPONENT_LADDER,
        )
        assert "verdict reason" not in bare.full_text

    def test_the_real_entrypoint_shows_verdict_reasons_on_the_board(self, tmp_path):
        """
        Production anchor: through the CLI over the real v6 slices, cells whose decision row
        carries a verdict reason show it.
        """
        _code, text = _run(tmp_path)
        assert "verdict reason:" in text

    def test_a_probe_verdict_reason_and_caveat_reach_a_number_cell(self):
        """
        An off-path row with a verdict, a verdict reason and a caveat shows all three, the
        reason beside the verdict and the caveat on a number cell too.
        """
        row = dict(
            _probe_row(prop="P6_subgame_perfection", caveat="a stated caveat"),
            verdict="LM_MEETS_CRITERION",
            verdict_reason="interval lies entirely below the anchor",
        )
        cell = sb.classify_probe("P6_subgame_perfection", row, True)
        assert cell.kind == sb.NUMBER
        assert "verdict reason: interval lies entirely below the anchor" in cell.text
        assert "notes: a stated caveat" in cell.text

    def test_a_probe_cells_zero_variance_clause_is_not_duplicated(self):
        """
        `probe_scores.py`'s writers (`_zero_variance`/`_paired_caveat`) put the zero-variance
        sentence in the row's own `caveat` whenever `n_replicates_distinct == 1`. The cell must
        show it once, from `caveat` under `notes:`, not once more from a second computation of
        the same sentence off `n_replicates`/`n_replicates_distinct`.
        """
        row = dict(
            _probe_row(prop="P6_subgame_perfection", caveat=self.ZERO),
            n_replicates=10,
            n_replicates_distinct=1,
            dropped_rate=0.0,
        )
        cell = sb.classify_probe("P6_subgame_perfection", row, True)
        assert cell.text.count(self.ZERO) == 1
        assert f"notes: {self.ZERO}" in cell.text

    def test_a_long_probe_caveat_keeps_its_zero_variance_clause_after_capping(self):
        """
        A probe cell whose ordinary caveat clause is long enough that an uncapped
        concatenation would exceed `_capped`'s limit still shows the zero-variance clause in
        full, in the short `.text` itself -- the table form never needs `_capped`'s blind
        character cut to do it, only `_reason_text`'s own per-clause capping.
        """
        long_clause = "x" * 500
        caveat = f"{long_clause}; {self.ZERO}"
        row = dict(
            _probe_row(prop="P6_subgame_perfection", caveat=caveat), dropped_rate=0.0
        )
        cell = sb.classify_probe("P6_subgame_perfection", row, True)
        assert self.ZERO in cell.text
        assert long_clause not in cell.text
        assert len(cell.text) <= sb.MAX_CELL_CHARS
        assert long_clause in cell.full_text
        capped = sb._capped(cell, sb.MAX_CELL_CHARS)
        assert capped.text == cell.text

    def test_a_long_verdict_reason_keeps_a_distribution_basis_note_after_capping(self):
        """
        An off-path row with a distribution basis and a long sampled-side verdict reason: the
        distribution-basis cell's own zero-variance note (a different statistic, on the exact
        basis) still reaches `.text` in full, and the short form stays under the cap without
        `_capped`'s blind cut.
        """
        long_reason = "y" * 500
        row = dict(
            _probe_row(prop="P6_subgame_perfection", caveat=""),
            dropped_rate=0.0,
            verdict="LM_MEETS_CRITERION",
            verdict_reason=long_reason,
            value_exact=0.05,
            verdict_exact="LM_MEETS_CRITERION",
            n_replicates_exact=5,
            n_replicates_distinct_exact=1,
        )
        cell = sb.classify_probe("P6_subgame_perfection", row, True)
        assert cell.kind == sb.NUMBER
        assert sb.ZERO_VARIANCE_PREFIX in cell.text
        assert long_reason not in cell.text
        assert len(cell.text) <= sb.MAX_CELL_CHARS
        assert long_reason in cell.full_text

    def test_every_reason_column_a_writer_emits_is_registered_and_reaches_a_cell(self):
        """
        The standing invariant, the `fidelity.py` pattern: the reason-named columns the three
        writers emit are exactly those REASON_COLUMNS lists (plus any named as deliberately not
        rendered), and each listed column, given a unique sentinel, shows it in a cell's whole
        text. A new reason column fails the first half until someone decides where it renders.
        """
        from lmopt.analysis import probe_scores

        def reason_named(names):
            return {
                n for n in names if any(k in n for k in ("reason", "caveat", "blocker"))
            }

        rows = _opponents([0.1, 0.2, 0.3, 0.4])
        decision = {d["property"]: d for d in analyse_stats.decision_table(rows)}[
            "P3_randomisation_margin"
        ]
        written = {
            "properties.csv": reason_named(
                set(
                    analyse_stats._bare_row(
                        "kuhn", "cfr_ref", "P3_randomisation", "AWAITING_DATA", "x"
                    )
                )
                | set(analyse_stats.EXACT_BASIS_COLUMNS)
            ),
            "decision.csv": reason_named(decision),
            "probe_scores.csv": reason_named(probe_scores.COLUMNS),
        }
        declared = {
            artefact: set(sb.REASON_COLUMNS.get(artefact, ()))
            | set(sb.NOT_RENDERED_REASON_COLUMNS.get(artefact, {}))
            for artefact in written
        }
        assert written == declared

        for column in sb.REASON_COLUMNS["properties.csv"]:
            sentinel = f"SENTINEL-properties-{column}"
            cell = _classify(_opponents([0.1, 0.2, 0.3, 0.4], caveat=sentinel))
            assert sentinel in cell.full_text, column
        blocked_rows = _opponents(
            [0.1, 0.2, 0.3, 0.4], "MEASURABLE_FLAWED", "a blocking defect"
        )
        blocked = {
            d["property"]: d for d in analyse_stats.decision_table(blocked_rows)
        }["P3_randomisation_margin"]
        for column in sb.REASON_COLUMNS["decision.csv"]:
            # `blocker` is a reason for a withheld cell, so it is pushed through a blocked one;
            # `verdict_reason` qualifies a verdict, so it is pushed through an admissible one.
            sentinel = f"SENTINEL-decision-{column}"
            cell = (
                sb.classify(
                    "P3_randomisation_margin",
                    dict(blocked, **{column: sentinel}),
                    blocked_rows,
                    False,
                    OPPONENT_LADDER,
                )
                if column == "blocker"
                else sb.classify(
                    "P3_randomisation_margin",
                    dict(decision, **{column: sentinel}),
                    rows,
                    False,
                    OPPONENT_LADDER,
                )
            )
            assert sentinel in cell.full_text, column
        for column in sb.REASON_COLUMNS["probe_scores.csv"]:
            sentinel = f"SENTINEL-probe-{column}"
            row = dict(
                _probe_row(prop="P6_subgame_perfection"),
                verdict="LM_MEETS_CRITERION",
                **{column: sentinel},
            )
            cell = sb.classify_probe("P6_subgame_perfection", row, True)
            assert sentinel in cell.full_text, column


class TestSevenBTieClause:
    def _scored_cell(self):
        """
        one Kuhn cell scored through the real property_table, with enough rows to clear
        the belief properties' own survival threshold
        :returns: the P7b row
        """
        from lmopt.games.cfr import solve_cfr
        from lmopt.games.loaders import load_kuhn

        game = load_kuhn()
        rows = [
            {
                "matchup_id": "m1",
                "game": "kuhn",
                "seed": h % 4,
                "hand_idx": h,
                "node_idx": 0,
                "reward": 0.0,
                "prompt": "x",
                "player_to_act": 0,
                "info_set": "0",
                "parsed_action": h % 2,
                "lm_seat": 0,
                "legal_mask": [1, 1],
                "elicited_belief": json.dumps({"1": 0.5, "2": 0.5}),
                "true_posterior": json.dumps({"1": 0.5, "2": 0.5}),
            }
            for h in range(40)
        ]
        table = analyse_stats.property_table(
            {("kuhn", "cfr_ref"): [("s0", pl.DataFrame(rows))]},
            {"kuhn": game},
            {"kuhn": solve_cfr(game, 200)},
        )
        return next(
            r
            for r in table
            if r["property"] == "P7b_action_belief" and r["value"] is not None
        )

    def test_a_scored_p7b_row_states_its_all_tie_share(self):
        """
        Production anchor at the emit site: every scored P7b row carries the share of rows
        on which every legal action ties, in one fixed form, so a reader never sees an
        agreement figure without knowing how much of it was a free choice.
        """
        row = self._scored_cell()
        assert re.search(
            r"every legal action ties under the stated belief on \d+\.\d{2}% of scored rows",
            row["caveat"],
        ), row["caveat"]

    def test_the_clause_reaches_the_p7_cell_whole_even_when_the_table_shortens_it(
        self,
        tmp_path,
    ):
        """
        The clause is the last of the P7b row's caveat clauses, so the table cell shows the
        leading clause marked `(+N more)` and the detail section under the table carries the
        clause itself. Nothing a reader needs to judge the number is lost.
        """
        clause = (
            "every legal action ties under the stated belief on 32.23% of scored rows"
        )
        root = _copy_root(tmp_path, rung="M0")
        _edit(
            root,
            "qwen2.5-14b-instruct",
            lambda df: df.with_columns(
                pl.when(pl.col("property") == "P7b_action_belief")
                .then(pl.lit(f"a leading clause; {clause}"))
                .otherwise(pl.col("caveat"))
                .alias("caveat")
            ),
        )
        _code, text = _run(tmp_path, root)
        block = text.split("<!-- SCOREBOARD:BEGIN -->")[1]
        table_cell = _cells(text, "M0")[(QWEN, "kuhn", "P7")]
        assert "(+1 more)" in table_cell and clause not in table_cell
        detail = block.split("cell(s) above show only their leading clause")[1]
        assert clause in detail


class TestCollapseMarker:
    def test_a_collapsed_cell_says_so_with_the_first_listed_share_and_keeps_its_kind(
        self,
        tmp_path,
    ):
        """
        Two arms sharing a policy on Leduc: every Leduc cell of both arms names the other
        arm and its first-listed share, no other cell is marked, and no cell's kind moves.
        """
        root = _copy_root(tmp_path, rung="M0")
        before = _cells(_run(tmp_path, root)[1], "M0")
        pair = ("mistral-nemo-instruct-2407", "olmo-2-1124-13b-instruct")
        models = tuple(MODEL_OF[a] for a in pair)
        for arm, other in (pair, pair[::-1]):
            _collapse_file(
                root / arm,
                [
                    _collapse_row("kuhn", "k-" + arm),
                    _collapse_row("leduc", "shared", other, "1.0"),
                ],
            )
        after = _cells(_run(tmp_path, root)[1], "M0")
        for key, cell in after.items():
            model, game, prop = key
            # P2 is structural at M0 and reads no arm's rows, so there is no arm whose
            # collapse could be reported beside it.
            marked = model in models and game == "leduc" and prop != "P2"
            assert ("same action policy as" in cell) == marked, key
            if marked:
                assert "first-listed legal action 100.00% of rows" in cell, key
            assert _kind(cell) == _kind(before[key]), key

    def test_a_lone_arm_that_always_plays_the_first_listed_action_says_so(
        self, tmp_path
    ):
        """
        An arm with no twin whose Leduc share is 1.0 carries the fixed-rule wording on its Leduc
        cells, and an arm with no twin and a share below 1.0 carries nothing.
        """
        root = _copy_root(tmp_path, rung="M0")
        _collapse_file(
            root / "qwen2.5-14b-instruct",
            [
                _collapse_row("leduc", "q", "", "1.0"),
                _collapse_row("kuhn", "k", "", "0.5"),
            ],
        )
        cells = _cells(_run(tmp_path, root)[1], "M0")
        assert (
            sb.LONE_FIRST_LISTED_WORDING
            == "; constant policy: one action at every info-set"
        )
        assert sb.LONE_FIRST_LISTED_WORDING.strip("; ") in cells[(QWEN, "leduc", "P5")]
        assert "fixed rule" not in cells[(QWEN, "kuhn", "P5")]
        assert "same action policy as" not in cells[(QWEN, "leduc", "P5")]

    def test_a_missing_collapse_file_reads_as_not_checked_never_as_no_collapse(
        self, tmp_path
    ):
        """
        An arm with no collapse file is named in the header as unchecked, and the header
        never reports it as an arm checked and found distinct.
        """
        _code, text = _run(tmp_path)
        assert sb.COLLAPSE_NOT_CHECKED in text
        for model in MODELS:
            assert model in text.split(sb.COLLAPSE_NOT_CHECKED)[1].split("\n")[0]
        assert "over 0 arm(s) checked" in text
        assert "same action policy as" not in text

    def test_the_header_counts_collapsed_arm_game_pairs(self, tmp_path):
        """
        The header counts (arm, game) pairs, not arms and not groups, and counts only the
        arms that carry a collapse file.
        """
        root = _copy_root(tmp_path, rung="M0")
        pair = ("mistral-nemo-instruct-2407", "olmo-2-1124-13b-instruct")
        for arm, other in (pair, pair[::-1]):
            _collapse_file(
                root / arm,
                [
                    _collapse_row("leduc", "shared", other),
                    _collapse_row("kuhn", "shared-k", other),
                ],
            )
        _code, text = _run(tmp_path, root)
        assert "Collapse: 4 (arm, game) pair(s)" in text
        assert "over 2 arm(s) checked" in text
        assert sb.COLLAPSE_NOT_CHECKED in text

    def test_a_collapse_row_without_a_share_says_the_share_was_not_recorded(
        self, tmp_path
    ):
        """
        A collapse row whose share is blank marks the cell without inventing a share.
        """
        root = _copy_root(tmp_path, rung="M0")
        _collapse_file(
            root / "qwen2.5-14b-instruct",
            [_collapse_row("kuhn", "x", "olmo-2-1124-13b-instruct", "")],
        )
        cell = _cells(_run(tmp_path, root)[1], "M0")[(QWEN, "kuhn", "P5")]
        assert "first-listed share not recorded" in cell

    def test_a_collapse_file_with_a_duplicate_game_or_a_missing_column_is_refused(
        self,
        tmp_path,
        capsys,
    ):
        """
        Two rows for one game, or a file missing a column, is a malformed record and refuses
        the run rather than picking one row.
        """
        root = _copy_root(tmp_path)
        _collapse_file(
            root / "qwen2.5-14b-instruct",
            [_collapse_row("kuhn"), _collapse_row("kuhn")],
        )
        assert _run(tmp_path, root)[0] == cli.EXIT_ARM_PROBLEM
        assert "two rows for kuhn" in capsys.readouterr().err
        (root / "qwen2.5-14b-instruct" / sb.COLLAPSE_FILE).write_text("game\nkuhn\n")
        assert _run(tmp_path, root)[0] == cli.EXIT_ARM_PROBLEM

    def test_the_first_listed_share_counts_the_lowest_index_legal_action(self):
        """
        The share is over the arm's own kept LM rows and counts the lowest-index legal
        action, not action zero and not the opponent's rows.
        """
        frame = pl.DataFrame(
            [
                {
                    "legal_mask": [0, 1, 1],
                    "parsed_action": 1,
                    "prompt": "p",
                    "fallback_used": False,
                },
                {
                    "legal_mask": [0, 1, 1],
                    "parsed_action": 2,
                    "prompt": "p",
                    "fallback_used": False,
                },
                {
                    "legal_mask": [1, 1, 1],
                    "parsed_action": 0,
                    "prompt": "p",
                    "fallback_used": False,
                },
                {
                    "legal_mask": [1, 1, 1],
                    "parsed_action": 2,
                    "prompt": None,
                    "fallback_used": False,
                },
                {
                    "legal_mask": [1, 1, 1],
                    "parsed_action": 2,
                    "prompt": "p",
                    "fallback_used": True,
                },
            ]
        )
        share, n_rows = arm_collapse.first_listed_action_share(frame)
        assert (share, n_rows) == (2 / 3, 3)

    def test_the_first_listed_share_refuses_rather_than_defaulting(self):
        """
        A frame with no legal mask, and one whose only LM rows are fallbacks, refuse: a
        defaulted share would read as a measured one.
        """
        with pytest.raises(arm_collapse.FingerprintUnavailable):
            arm_collapse.first_listed_action_share(pl.DataFrame({"parsed_action": [0]}))
        only_fallback = pl.DataFrame(
            [
                {
                    "legal_mask": [1, 1],
                    "parsed_action": 0,
                    "prompt": "p",
                    "fallback_used": True,
                },
            ]
        )
        with pytest.raises(arm_collapse.FingerprintUnavailable):
            arm_collapse.first_listed_action_share(only_fallback)

    def test_the_pass_writes_a_file_the_scoreboard_marks_cells_from(self, tmp_path):
        """
        Production anchor over both scripts: `arm_collapse_pass.py` reads two arms' shards,
        finds the identical play, writes each arm's collapse.csv, and `scoreboard.py` then
        marks exactly that game's cells.
        """
        root = _copy_root(tmp_path, rung="M0")
        same, different = [0, 1, 2, 0, 1, 2], [0, 0, 0, 0, 0, 0]
        plays = dict(zip(MODELS, (same, same, different), strict=False))
        for model, actions in plays.items():
            shards = tmp_path / "shards" / model
            shards.mkdir(parents=True)
            _play_frame(actions).write_parquet(shards / "leduc_lm_vs_rl_seed42.parquet")
            meta = json.loads((root / model / "run_meta.json").read_text())
            meta["data_dir"] = str(shards)
            (root / model / "run_meta.json").write_text(json.dumps(meta))
        assert collapse_cli.main(["--analysis-root", str(root)]) == collapse_cli.EXIT_OK
        written = {
            model: list(
                _csvmod.DictReader((root / model / sb.COLLAPSE_FILE).open(newline=""))
            )
            for model in MODELS
        }
        assert [r["collapsed_with"] for r in written[MODELS[0]]] == [MODELS[1]]
        assert [r["collapsed_with"] for r in written[MODELS[2]]] == [""]
        cells = _cells(_run(tmp_path, root)[1], "M0")
        assert (
            "same action policy as " + MODELS[1]
            in cells[(MODEL_OF[MODELS[0]], "leduc", "P5")]
        )
        assert (
            "same action policy as" not in cells[(MODEL_OF[MODELS[2]], "leduc", "P5")]
        )
        assert "same action policy as" not in cells[(MODEL_OF[MODELS[0]], "kuhn", "P5")]


def _offpath_row(game="leduc", collapsed_with="", action="", n_rows=300, n_infosets=""):
    """
    one collapse_offpath.csv row
    :param game: game key
    :param collapsed_with: comma-separated other arms sharing the driven policy
    :param action: the constant action as written, or empty
    :param n_rows: the driven rows behind it
    :param n_infosets: the info-sets the constant action covers, or empty
    :returns: a dict
    """
    return {
        "game": game,
        "fingerprint": "f-" + game,
        "collapsed_with": collapsed_with,
        "constant_action": action,
        "n_rows": n_rows,
        "n_infosets": n_infosets,
        "offpath_root": "data/driven",
    }


def _level_cells(arm_collapse, arm_offpath, kind=sb.NUMBER):
    """
    the cells of one model on the M0 board for a single arm
    :param arm_collapse: that arm's on-path collapse table, or None
    :param arm_offpath: that arm's off-path collapse table, or None
    :param kind: the kind every cell of the arm carries
    :returns: {(game, property number): Cell}
    """
    cells = {
        (game, number): sb.Cell(kind, "x")
        for game in sb.SCOREBOARD_GAMES
        for number in sb.PROPERTY_NUMBERS
    }
    arm = sb.ArmInput("a", {"memory_rung": "M0"}, cells, arm_collapse, arm_offpath)
    built = sb.build_level_cells(
        "m",
        "M0",
        [arm],
        lambda number, conditions, level: list(conditions),
        lambda number, level: "",
        lambda number, level: "M0 run",
    )
    return {(c.game, c.number): c.cell for c in built}


class TestOffpathCollapse:
    def test_a_p6_cell_shows_the_offpath_marker_in_place_of_the_onpath_one(self):
        """
        The arm shares its on-path play with another arm on Leduc, and its driven play with a
        third: its P6 cell names the driven partner and not the on-path one, while its other
        Leduc cells keep the on-path marker.
        """
        on = {"leduc": {"collapsed_with": "onpath-twin", "first_listed_share": 1.0}}
        off = {"leduc": _offpath_row(collapsed_with="drive-twin", action="1 (Call)")}
        cells = _level_cells(on, off)
        p6 = cells[("leduc", 6)].text
        assert (
            "same driven off-path policy as drive-twin (constant action 1 (Call))" in p6
        )
        assert "onpath-twin" not in p6
        assert "same action policy as onpath-twin" in cells[("leduc", 5)].text
        assert "driven" not in cells[("leduc", 5)].text
        assert cells[("leduc", 6)].kind == sb.NUMBER

    def test_a_missing_offpath_file_reads_as_not_checked_never_as_clean(self):
        """
        An arm with a measured P6 cell and no collapse_offpath.csv says so in the cell, and a
        game absent from a present file says the same.
        """
        for table in (None, {"kuhn": _offpath_row("kuhn")}):
            cells = _level_cells(None, table)
            assert cells[("leduc", 6)].text.endswith(sb.OFFPATH_COLLAPSE_NOT_CHECKED)
            assert sb.OFFPATH_COLLAPSE_NOT_CHECKED not in cells[("leduc", 5)].text

    def test_only_measured_p6_cells_carry_the_marker(self):
        """
        An awaiting or structural P6 cell has no driven play to mark, so it reads unchanged.
        """
        for kind in (sb.AWAITING, sb.STRUCTURAL, sb.BROKEN):
            cells = _level_cells(None, None, kind)
            assert cells[("leduc", 6)].text == "x", kind

    def test_a_lone_constant_policy_and_a_varied_one_are_worded_apart(self):
        """
        A constant driven policy nobody else plays is named as such; an arm that played more
        than one action and shares nothing carries no marker at all.
        """
        off = {
            "leduc": _offpath_row("leduc", "", "0 (Fold)"),
            "kuhn": _offpath_row("kuhn", "", ""),
        }
        cells = _level_cells(None, off)
        assert cells[("leduc", 6)].text.endswith(sb.LONE_CONSTANT_WORDING)
        assert cells[("kuhn", 6)].text == "x"

    def test_the_header_counts_offpath_pairs_and_names_arms_never_checked(
        self, tmp_path
    ):
        """
        The header counts the off-path pairs over the probed arms checked and names the probed
        arms without a file, which is not the same as checked and found distinct.
        """
        arms = [
            sb.ArmSummary("a", "d", "s", "p", None, None, ("leduc", "liarsdice")),
            sb.ArmSummary("b", "d", "s", "p", None, None, None),
            sb.ArmSummary("c", "d", "s", None, None, None, None),
        ]
        note = sb._collapse_note(arms)
        assert "Off-path collapse (P6): 2 (arm, game) pair(s)" in note
        assert "over 1 arm(s) checked" in note.split("Off-path collapse")[1]
        assert "off-path collapse not checked for b:" in note
        assert "for c" not in note.split("off-path collapse not checked")[1]

    def test_the_pass_reads_the_driven_runs_and_the_scoreboard_loads_what_it_wrote(
        self, tmp_path
    ):
        """
        Production anchor over both scripts: `arm_collapse_pass.py --offpath-root` reads two
        arms' driven shards, finds the same constant action on Leduc, writes each probed arm's
        collapse_offpath.csv with the root it read, and `scoreboard.py`'s loader returns it.
        An arm with no probe_scores.csv gets no file.
        """
        root = _copy_root(tmp_path, rung="M0")
        driven = tmp_path / "driven"
        probed = MODELS[:2]
        for model in MODELS:
            shards = tmp_path / "shards" / model
            shards.mkdir(parents=True)
            _play_frame([0, 1, 2, 0, 1, 2]).write_parquet(
                shards / "leduc_lm_vs_rl_seed42.parquet"
            )
            meta = json.loads((root / model / "run_meta.json").read_text())
            meta["data_dir"] = str(shards)
            (root / model / "run_meta.json").write_text(json.dumps(meta))
        for model, action in zip(probed, ([1] * 6, [1] * 6), strict=False):
            (root / model / "probe_scores.csv").write_text("game\n")
            (driven / model).mkdir(parents=True)
            _play_frame(action).write_parquet(
                driven / model / "leduc_lm_vs_cfr_ref.offpath_drive_seed42.parquet"
            )
        code = collapse_cli.main(
            ["--analysis-root", str(root), "--offpath-root", str(driven)]
        )
        assert code == collapse_cli.EXIT_OK
        written = {model: cli._load_collapse_offpath(root / model) for model in MODELS}
        assert written[MODELS[2]] is None
        assert written[MODELS[0]]["leduc"]["collapsed_with"] == MODELS[1]
        assert written[MODELS[1]]["leduc"]["collapsed_with"] == MODELS[0]
        assert written[MODELS[0]]["leduc"]["constant_action"] == "1"
        assert written[MODELS[0]]["leduc"]["offpath_root"] == str(driven)

    def test_the_pass_refuses_an_unmatched_or_ambiguous_driven_directory(
        self, tmp_path, capsys
    ):
        """
        A probed arm with no matching subdirectory under the root, or more than one, exits 1
        with the matching refusal and writes no off-path file, since a guess would attach one
        model's play to another.
        """
        root = _copy_root(tmp_path, rung="M0")
        model = MODELS[0]
        shards = tmp_path / "shards"
        shards.mkdir()
        _play_frame([0, 1, 2, 0, 1, 2]).write_parquet(
            shards / "leduc_lm_vs_rl_seed42.parquet"
        )
        for name in MODELS:
            meta = json.loads((root / name / "run_meta.json").read_text())
            meta["data_dir"] = str(shards)
            (root / name / "run_meta.json").write_text(json.dumps(meta))
        (root / model / "probe_scores.csv").write_text("game\n")
        driven = tmp_path / "driven"
        (driven / "other-model").mkdir(parents=True)
        args = ["--analysis-root", str(root), "--offpath-root", str(driven)]
        assert collapse_cli.main(args) == collapse_cli.EXIT_ARM_PROBLEM
        assert "0 matching subdirectories" in capsys.readouterr().err
        arm = model + "-m0"
        (root / model).rename(root / arm)
        (driven / model).mkdir()
        (driven / arm).mkdir()
        assert collapse_cli.main(args) == collapse_cli.EXIT_ARM_PROBLEM
        assert "2 matching subdirectories" in capsys.readouterr().err
        assert not (root / arm / sb.COLLAPSE_OFFPATH_FILE).exists()

    def test_two_levels_of_one_model_are_refused_not_reported_as_each_others_twin(
        self, tmp_path, capsys
    ):
        """
        The m0 and m2 arms of one model both match the model's one driven directory; the pass
        refuses rather than fingerprint the same play twice and report each arm as collapsed with
        its own sibling.
        """
        root = _copy_root(tmp_path, rung="M0")
        model = MODELS[0]
        shards = tmp_path / "shards"
        shards.mkdir()
        _play_frame([0, 1, 2, 0, 1, 2]).write_parquet(
            shards / "leduc_lm_vs_rl_seed42.parquet"
        )
        for name in MODELS:
            meta = json.loads((root / name / "run_meta.json").read_text())
            meta["data_dir"] = str(shards)
            (root / name / "run_meta.json").write_text(json.dumps(meta))
        first = root / (model + "-m0")
        second = root / (model + "-m2")
        (root / model).rename(first)
        shutil.copytree(first, second)
        for arm in (first, second):
            (arm / "probe_scores.csv").write_text("game\n")
        driven = tmp_path / "driven"
        (driven / model).mkdir(parents=True)
        _play_frame([1] * 6).write_parquet(
            driven / model / "leduc_lm_vs_cfr_ref.offpath_drive_seed42.parquet"
        )
        args = ["--analysis-root", str(root), "--offpath-root", str(driven)]
        assert collapse_cli.main(args) == collapse_cli.EXIT_ARM_PROBLEM
        assert "resolve to one driven directory" in capsys.readouterr().err
        assert not (first / sb.COLLAPSE_OFFPATH_FILE).exists()


class TestHeaderKinds:
    def _block(self, cells, level="M1"):
        arms = [sb.ArmSummary("m", "root/x", "abc", None)]
        return sb.render_block(
            arms,
            [sb.Board(level, tuple(cells))],
            analyse_stats.DECISION_RULE,
            vocabulary._DIRECTION_TEXT,
        )

    def test_awaiting_is_counted_in_its_own_column_never_as_broken(self):
        """
        The header line and the arm table count awaiting cells separately; the BROKEN total does
        not include them.
        """
        # PARTIAL_APPROVED's count must appear in its own column, between `partial` and
        # `structural` -- neither merged into `partial` (which would hide an approved cell)
        # nor dropped from the tally.
        kinds = [
            (sb.NUMBER, 1),
            (sb.PARTIAL, 5),
            (sb.PARTIAL_APPROVED, 7),
            (sb.STRUCTURAL, 2),
            (sb.AWAITING, 3),
            (sb.INADMISSIBLE, 6),
            (sb.BROKEN, 4),
        ]
        cells, number = [], 0
        for kind, count in kinds:
            for _ in range(count):
                number += 1
                cells.append(
                    sb.GridCell("m", "kuhn", number % 8 + 1, sb.Cell(kind, "x"))
                )
        block = self._block(cells)
        assert (
            "1 numbers, 5 partial, 7 partial (approved), 2 structural, 3 awaiting data, "
            "6 inadmissible, 4 BROKEN"
        ) in block
        header = next(
            line for line in block.splitlines() if line.startswith("| model |")
        )
        assert (
            header.index("numbers")
            < header.index("partial")
            < header.index("partial (approved)")
            < header.index("structural")
            < header.index("awaiting")
            < header.index("inadmissible")
            < header.index("BROKEN")
        )
        assert "| m | 1 | 5 | 7 | 2 | 3 | 6 | 4 |" in block

    def test_the_legend_says_probe_cells_apply_no_gate_and_are_not_pooled(self):
        """
        The block states that P1, P4 and P6 come from probe_scores.csv, per game, with no
        admissibility gate, so a reader does not assume the other cells' rules.
        """
        block = self._block(())
        assert sb.PROBE_NOTE in block and "no admissibility gate" in block

    def test_the_probe_file_hash_is_shown_when_there_is_one(self):
        """
        An arm with a probe_scores.csv shows its short hash; one without shows `none`.
        """
        arm = sb.ArmSummary("m", "root/x", "abc", "def123", memory_rung="M1")
        block = sb.render_block(
            [arm],
            [sb.Board("M1", ())],
            analyse_stats.DECISION_RULE,
            vocabulary._DIRECTION_TEXT,
        )
        assert "| m | M1 | root/x | abc | def123 |" in block
        bare = sb.ArmSummary("m", "root/x", "abc")
        assert "| m | not recorded | root/x | abc | none |" in sb.render_block(
            [bare],
            [sb.Board("M1", ())],
            analyse_stats.DECISION_RULE,
            vocabulary._DIRECTION_TEXT,
        )


#: The seeds the driven fixture arm carries. More than one, because the P6 rule compares
#: the cell's own replicate interval with the anchor's, and one seed forms no interval.
_PDRIVEN_SEEDS = (42, 43, 44, 45, 46)


def _anchors_for(path, mean=0.9, half=0.01):
    """
    an off-path anchor artefact matching the driven arm `_score_root` writes
    :param path: where to write it (Path)
    :param mean: the Kuhn anchor's replicate mean (float)
    :param half: half the spread between its two seed values (float)
    :returns: the path as a str
    """
    from lmopt.analysis.sentinel_anchors import (
        P6_ESTIMAND,
        SAMPLED_BASIS,
        build_artefact,
    )
    from lmopt.provenance.versions import library_versions
    from lmopt.scoring.property6 import DEFAULT_TREMBLE_EPS
    from tests._probe_arms import CFR_ITERS

    artefact = build_artefact(
        {
            ("kuhn", "cfr_ref"): {
                "seeds": [42, 43],
                "values": [mean - half, mean + half],
                "drive_openings": [1],
                "seat_mode": "random",
            }
        },
        {
            "cfr_iters": CFR_ITERS,
            "tremble_eps": DEFAULT_TREMBLE_EPS,
            "estimand": P6_ESTIMAND,
            "basis": SAMPLED_BASIS,
            **library_versions(),
        },
    )
    Path(path).write_text(json.dumps(artefact))
    return str(path)


def _score_root(tmp_path, anchors=None, rung="M0"):
    """
    a scoreboard root of one arm whose probe_scores.csv was written by the real scoring script
    :param tmp_path: pytest tmp_path
    :param anchors: the off-path anchor artefact path to score against, or None for one
        matching the driven arm on Kuhn alone
    :param rung: the memory level every arm records, M0 unless a test reads another board
    :returns: (analysis root, the probe_scores.csv path, the arm name)
    """
    cache = tmp_path / "cache"
    warm_cfr_cache(cache)
    arms = tmp_path / "arms"
    # `base_dir`/`probe_dir` bind on (role, model_id, memory_rung), not on the arm's
    # directory, so every arm here must carry the same model_id as the analysed arm below
    # (both to match `base_dir` and because `model_id` is itself a MUST_MATCH_FIELDS field).
    base = write_arm(
        arms / "base",
        {"kuhn": _arm_rows(_PBASE)},
        cfr_cache_dir=str(cache),
        model_id=QWEN,
        memory_rung=rung,
    )
    relabel = write_arm(
        arms / "relabel",
        {"kuhn": _arm_rows(_PPROBE, probe_type="relabel", prompt_variant="relabel_v1")},
        cfr_cache_dir=str(cache),
        probe_type="relabel",
        prompt_variant="relabel_v1",
        model_id=QWEN,
        memory_rung=rung,
    )
    driven_frame = pl.concat(
        [
            _arm_rows(
                {k: ([1] * n + [0] * (4 - n)) for k in _PDRIVEN},
                probe_type="offpath_drive",
                player=1,
                seed=seed,
            )
            for n, seed in enumerate(_PDRIVEN_SEEDS)
        ]
    )
    driven = write_arm(
        arms / "driven",
        {"kuhn": driven_frame},
        cfr_cache_dir=str(cache),
        probe_type="offpath_drive",
        drive_openings=[1],
        n_seeds=len(_PDRIVEN_SEEDS),
        model_id=QWEN,
        memory_rung=rung,
    )
    root = tmp_path / "root"
    shutil.copytree(FIXTURES, root)
    arm_dir = root / "qwen2.5-14b-instruct"
    meta = json.loads((arm_dir / "run_meta.json").read_text())
    meta["data_dir"] = f"anything/{Path(base).parent.name}/{Path(base).name}"
    # A probe-scored arm is the base run its probe partners are paired against, which
    # is the condition P1, P4 and P6 read.
    meta["condition"] = dict(meta["condition"], memory_rung=rung)
    (arm_dir / "run_meta.json").write_text(json.dumps(meta))
    code = probe_cli.main(
        [
            "--base",
            str(base),
            "--relabel",
            str(relabel),
            "--driven",
            str(driven),
            "--out",
            str(arm_dir),
            "--p6-sentinel-anchors",
            anchors or _anchors_for(tmp_path / "p6_anchors.json"),
        ]
    )
    assert code == 0
    return root, arm_dir / "probe_scores.csv", QWEN


class TestProbeCellsThroughTheRealScripts:
    def test_scored_probe_arms_reach_the_scoreboard_as_numbers_and_gaps(self, tmp_path):
        """
        Production anchor for the whole path: `probe_scores.py` writes the file, `scoreboard.py`
        reads it. Kuhn P1 and P6 read numbers from the scorers, P4 (no payoff arm supplied) reads
        BROKEN naming the missing role, games the arms were not configured for read `awaiting
        data`, and the other two models (no file) read BROKEN naming the missing file.
        """
        root, _csv_path, arm = _score_root(tmp_path)
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_OK
        cells = _cells(text, "M0")
        assert re.fullmatch(
            r"-?\d+\.\d{4} \(\d+ union / \d+\.\d per seed paired info-sets, "
            r"coverage \d+\.\d% union / \d+\.\d% per seed\); dropped [\d.]+%"
            r"(?:; notes: .+)?",
            cells[(arm, "kuhn", "P1")],
        ), cells[(arm, "kuhn", "P1")]
        assert re.fullmatch(
            r"-?\d+\.\d{4} LM_\w+ \(\d+ off-path info-sets\); dropped [\d.]+%; "
            r"(?:verdict reason: [^;]+; )?"
            r"notes: " + re.escape(ps.DISTANCE_NOTE) + r"; "
            r"sampled basis, no distribution basis: no exact-policy pass on this arm"
            + re.escape(sb.OFFPATH_COLLAPSE_NOT_CHECKED),
            cells[(arm, "kuhn", "P6")],
        ), cells[(arm, "kuhn", "P6")]
        assert "no payoff_scale arm was supplied" in cells[(arm, "kuhn", "P4")]
        assert _kind(cells[(arm, "leduc", "P6")]) == "awaiting"
        for other in MODEL_OF.values():
            if other != arm:
                assert (
                    "M0 run paired with its relabel run" in cells[(other, "kuhn", "P1")]
                )
        overall = text.split("<!-- SCOREBOARD:BEGIN -->\n")[1].split("\n")[0]
        assert "awaiting data" in overall and overall.startswith("Generated by")

    def test_a_failing_constant_action_check_leaves_the_measured_p6_value(
        self, tmp_path, monkeypatch
    ):
        """
        Through the real scoring script: when the constant-action diagnostic raises, the Kuhn P6
        row is still MEASURABLE with its value and verdict, and its caveat leads with the failure.
        """

        def raising(frame, actor=None):
            raise arm_collapse.FingerprintUnavailable("forced failure")

        monkeypatch.setattr(probe_cli, "constant_action", raising)
        _root, csv_path, _arm = _score_root(tmp_path)
        with open(csv_path, newline="") as fh:
            row = next(
                r
                for r in _csvmod.DictReader(fh)
                if r["game"] == "kuhn" and r["property"].startswith("P6")
            )
        assert row["status"] == "MEASURABLE" and row["value"] and row["verdict"]
        assert row["caveat"].startswith("constant-action check failed: forced failure")

    def test_probe_cells_above_m0_read_numbers_with_the_memory_sentence(self, tmp_path):
        """
        Production anchor for the widened levels: an M1 base arm paired with M1 relabel and
        off-path arms through `probe_scores.py` reaches the M1 board as numbers, and the P1
        cell carries the sentence saying the distance
        includes the memory divergence. The M0 board of the same root still waits.
        """
        root, _csv_path, arm = _score_root(tmp_path, rung="M1")
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_OK
        cells = _cells(text, "M1")
        assert _kind(cells[(arm, "kuhn", "P1")]) == "number", cells[(arm, "kuhn", "P1")]
        assert (
            "Memory: above M0 the distance includes the divergence"
            in cells[(arm, "kuhn", "P1")]
        )
        assert _kind(cells[(arm, "kuhn", "P6")]) == "number", cells[(arm, "kuhn", "P6")]
        assert _kind(_cells(text, "M0")[(arm, "kuhn", "P1")]) == "awaiting"

    def test_a_measured_p6_number_whose_own_anchor_is_missing_reads_awaiting(
        self,
        tmp_path,
    ):
        """
        A leduc P6 row the scorer measured, sitting in probe_scores.csv beside Kuhn's, is
        withheld by the scoreboard: with no leduc anchor the row carries no verdict, so the
        cell reads awaiting data naming the missing anchor and never a number, while the
        Kuhn cell keeps its number and the CSV keeps both measurements.
        """
        root, csv_path, arm = _score_root(tmp_path)
        with open(csv_path, newline="") as fh:
            table = list(_csvmod.DictReader(fh))
        kuhn_p6 = next(
            r for r in table if r["game"] == "kuhn" and r["property"].startswith("P6")
        )
        assert kuhn_p6["status"] == "MEASURABLE" and kuhn_p6["verdict"]
        table = [
            r
            for r in table
            if not (r["game"] == "leduc" and r["property"].startswith("P6"))
        ]
        table.append(
            dict(
                kuhn_p6,
                game="leduc",
                verdict="",
                anchor_mean="",
                anchor_ci_lo="",
                anchor_ci_hi="",
                verdict_reason="no sentinel anchor for (game='leduc', opponent='cfr_ref')",
            )
        )
        with open(csv_path, "w", newline="") as fh:
            w = _csvmod.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        cells = _cells(_run(tmp_path, root)[1], "M0")
        leduc = cells[(arm, "leduc", "P6")]
        assert _kind(leduc) == "awaiting" and "no sentinel anchor" in leduc, leduc
        assert _kind(cells[(arm, "kuhn", "P6")]) == "number"

    def test_a_p6_cell_with_both_bases_labels_each_through_the_real_scripts(
        self, tmp_path
    ):
        """
        Production anchor for the two-basis label: the scorer's own Kuhn P6 row, given a
        distribution-basis value and a verdict that differs from the sampled one, renders
        through `scoreboard.py` headed `exact: <verdict>`, then `sampled: <verdict>`, then
        `bases disagree`, and the sampled verdict is not repeated in the sampled detail.
        """
        root, csv_path, arm = _score_root(tmp_path)
        with open(csv_path, newline="") as fh:
            table = list(_csvmod.DictReader(fh))
        row = next(
            r for r in table if r["game"] == "kuhn" and r["property"].startswith("P6")
        )
        sampled = row["verdict"]
        assert row["status"] == "MEASURABLE" and sampled
        exact = next(v for v in sorted(ps.OFFPATH_VERDICTS) if v != sampled)
        row.update(
            value_exact="0.05",
            baseline_exact="0.3",
            exact_coverage="1.0",
            verdict_exact=exact,
            exact_reason="",
        )
        with open(csv_path, "w", newline="") as fh:
            w = _csvmod.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_OK
        cell = _cells(text, "M0")[(arm, "kuhn", "P6")]
        assert cell.startswith(f"0.0500 exact: {exact} (distribution basis, "), cell
        assert (
            f"; sampled: {sampled}" in cell and "; bases disagree; sampled " in cell
        ), cell
        assert cell.count(sampled) == 1, cell

    def test_the_verdict_a_p6_cell_shows_is_the_one_its_own_anchor_produced(
        self, tmp_path
    ):
        """
        The same arm against a high Kuhn anchor and against a low one gives two different
        verdicts in the cell, so the board really is showing the anchor's answer.
        """
        high = _score_root(
            tmp_path / "high",
            anchors=_anchors_for(tmp_path / "high-anchors.json", mean=0.95),
        )
        low = _score_root(
            tmp_path / "low",
            anchors=_anchors_for(tmp_path / "low-anchors.json", mean=0.01, half=0.001),
        )
        passing = _cells(_run(tmp_path / "high", high[0])[1], "M0")[
            (high[2], "kuhn", "P6")
        ]
        failing = _cells(_run(tmp_path / "low", low[0])[1], "M0")[
            (low[2], "kuhn", "P6")
        ]
        assert "LM_MEETS_CRITERION" in passing, passing
        assert "LM_FAILS_CRITERION" in failing, failing

    @pytest.mark.parametrize(
        "status,value,caveat,kind",
        [
            ("NOT_COMPUTED", "", "reference cache is cold for leduc", "BROKEN"),
            ("RECORDED_ABSENCE", "", "the scorer recorded an absence", "BROKEN"),
            ("MEASURABLE", "", "", "BROKEN"),
            ("AWAITING_DATA", "", "no leduc rows in the driven arm", "awaiting"),
        ],
    )
    def test_a_broken_p6_row_on_a_game_without_an_anchor_stays_broken(
        self,
        tmp_path,
        status,
        value,
        caveat,
        kind,
    ):
        """
        Awaiting means collection alone would fix the cell. A leduc P6 row that is a defect
        (not computed, a recorded absence, a measurable row with no value) keeps its own BROKEN
        reason on the scoreboard; only a row that would otherwise read awaiting shows a reason
        collection can answer.
        """
        root, csv_path, arm = _score_root(tmp_path)
        with open(csv_path, newline="") as fh:
            table = list(_csvmod.DictReader(fh))
        kuhn_p6 = next(
            r for r in table if r["game"] == "kuhn" and r["property"].startswith("P6")
        )
        table = [
            r
            for r in table
            if not (r["game"] == "leduc" and r["property"].startswith("P6"))
        ]
        table.append(
            dict(
                kuhn_p6,
                game="leduc",
                status=status,
                value=value,
                caveat=caveat,
                verdict="",
                verdict_reason="no sentinel anchor for leduc",
            )
        )
        with open(csv_path, "w", newline="") as fh:
            w = _csvmod.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        cell = _cells(_run(tmp_path, root)[1], "M0")[(arm, "leduc", "P6")]
        assert _kind(cell) == kind, cell
        if kind == "BROKEN":
            assert cell.startswith("BROKEN") and "no sentinel anchor" not in cell
        else:
            assert caveat in cell

    def test_p1_p4_p6_never_read_structural_even_with_planted_not_applicable_probe_rows(
        self,
        tmp_path,
    ):
        """
        Rewriting every probe row's status to NOT_APPLICABLE leaves P1, P4 and P6 BROKEN, never
        structural, on every game.
        """
        root, csv_path, arm = _score_root(tmp_path)
        with open(csv_path, newline="") as fh:
            table = list(_csvmod.DictReader(fh))
        for r in table:
            r["status"], r["value"] = "NOT_APPLICABLE", ""
            r["caveat"] = "no variant exists"
        with open(csv_path, "w", newline="") as fh:
            w = _csvmod.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        cells = _cells(_run(tmp_path, root)[1], "M0")
        for (m, game, prop), cell in cells.items():
            if m == arm and prop in ("P1", "P4", "P6"):
                assert _kind(cell) == "BROKEN", (prop, game, cell)

    def test_a_probe_file_bound_to_another_arm_refuses_the_whole_run(
        self, tmp_path, capsys
    ):
        """
        A probe_scores.csv whose `base_dir` names a different model than the arm it sits
        beside exits with the arm-refusal code and writes nothing to the board file.
        """
        root, csv_path, _arm = _score_root(tmp_path)
        csv_path.write_text(
            csv_path.read_text().replace(f"base:{QWEN}:", "base:elsewhere:")
        )
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_ARM_PROBLEM and text == BOARD_FILE
        assert "bound to" in capsys.readouterr().err

    def test_the_header_carries_the_probe_file_hash_and_changes_with_it(self, tmp_path):
        """
        Adding a column the scoreboard does not read to probe_scores.csv leaves every cell alone
        but changes the block's provenance line, so a changed file cannot go unrecorded.
        """
        root, csv_path, _arm = _score_root(tmp_path)
        first = _run(tmp_path, root)[1]
        lines = csv_path.read_text().rstrip("\n").split("\n")
        csv_path.write_text(
            "\n".join([lines[0] + ",note"] + [ln + ",x" for ln in lines[1:]]) + "\n"
        )
        second = _run(tmp_path, root)[1]
        assert _all_cells(first) == _all_cells(second)
        assert first != second

    def test_a_probe_file_missing_a_column_or_with_a_duplicate_row_is_handled_loudly(
        self,
        tmp_path,
    ):
        """
        A probe_scores.csv without `status` refuses the run naming it; a duplicated (game,
        property) row makes that cell BROKEN rather than picking one.
        """
        root, csv_path, arm = _score_root(tmp_path)
        original = csv_path.read_text()
        lines = original.rstrip("\n").split("\n")
        kuhn_p1 = next(
            line for line in lines[1:] if line.startswith("kuhn,P1_invariance")
        )
        csv_path.write_text(original + kuhn_p1 + "\n")
        cells = _cells(_run(tmp_path, root)[1], "M0")
        assert "duplicate" in cells[(arm, "kuhn", "P1")]
        header = lines[0].replace(",status,", ",statusx,")
        csv_path.write_text("\n".join([header] + lines[1:]) + "\n")
        code, text = _run(tmp_path, root)
        assert code == cli.EXIT_ARM_PROBLEM and text == BOARD_FILE

    def test_the_role_table_must_match_the_scorers_unavailable_rows_at_run_time(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        """
        If the scorer's status table stops marking a probe-sourced row UNAVAILABLE, the scoreboard
        refuses rather than silently reading it from the wrong file.
        """
        flipped = dict(vocabulary.PROPERTY_STATUS)
        flipped["P6_subgame_perfection"] = ("MEASURABLE", "")
        monkeypatch.setattr(vocabulary, "PROPERTY_STATUS", flipped)
        code, text = _run(tmp_path)
        assert code == cli.EXIT_ARM_PROBLEM and text == BOARD_FILE
        assert "P6_subgame_perfection" in capsys.readouterr().err

    def test_a_rerun_over_probe_scored_arms_is_byte_identical_and_checks_clean(
        self, tmp_path
    ):
        """
        Two runs over the probe-scored root write identical bytes and `--check` then exits 0.
        """
        root, _csv_path, _arm = _score_root(tmp_path)
        first = _run(tmp_path, root)[1]
        assert _run(tmp_path, root)[1] == first
        code, _text = _run(tmp_path, root, extra=["--check"], board_file_text=first)
        assert code == cli.EXIT_OK
        repo = Path(cli.__file__).resolve().parents[1]
        for seed in (1, 2, 3):
            board_file = tmp_path / f"seed{seed}.md"
            board_file.write_text(BOARD_FILE)
            env = dict(
                os.environ,
                PYTHONHASHSEED=str(seed),
                PYTHONPATH=f"{repo}:{repo / 'src'}",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(repo / "scripts" / "scoreboard.py"),
                    "--analysis-root",
                    str(root),
                    "--expect-arms",
                    "3",
                    "--partial",
                    "--roster-config",
                    str(ROSTER_CONFIG),
                    "--board-file",
                    str(board_file),
                ],
                check=True,
                env=env,
                cwd=repo,
            )
            assert board_file.read_text() == first


class TestEntrypointDrift:
    def test_a_decision_rule_without_a_headline_row_refuses_instead_of_crashing(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        """
        If the project's decision rule stops naming a headline row for a property, the scoreboard
        exits with the arm-refusal code and a message, not a traceback, and writes nothing.
        """
        trimmed = [d for d in vocabulary.DECISION_RULE if not d[0].startswith("P8")]
        monkeypatch.setattr(vocabulary, "DECISION_RULE", trimmed)
        code, text = _run(tmp_path)
        assert code == cli.EXIT_ARM_PROBLEM and text == BOARD_FILE
        assert "no headline row" in capsys.readouterr().err


class TestExpectedCellCount:
    """The board's cell count is derived from a model roster, not from whatever the
    analysis root happens to hold, so a model absent from the root is visible in the
    header rather than only being absent from the row list.
    """

    def _roster(self, tmp_path, models):
        """
        write a synthetic roster config and return its path
        :param tmp_path: pytest tmp_path
        :param models: [(id, enabled)]
        :returns: the written file's path
        """
        path = tmp_path / "roster.yaml"
        lines = ["models:"]
        for model_id, enabled in models:
            lines.append(f"  - id: {model_id}")
            lines.append(f"    enabled: {str(enabled).lower()}")
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_every_enabled_entry_is_returned_sorted_disabled_and_id_less_entries_dropped(
        self,
        tmp_path,
    ):
        """
        Two enabled entries come back sorted; a disabled entry and an entry missing `id` are
        both dropped, not counted or crashed on.
        """
        path = self._roster(
            tmp_path, [("z/one", True), ("a/two", True), ("x/three", False)]
        )
        path.write_text(path.read_text() + "  - enabled: true\n")  # no id
        assert cli.expected_models(path) == ("a/two", "z/one")

    def test_a_missing_file_an_empty_roster_and_a_roster_with_nothing_enabled_all_raise(
        self,
        tmp_path,
    ):
        """
        Three distinct ways to have no expected models, each refused with a message naming
        the file, not a silently-empty roster that would make every found model read as
        unexpected.
        """
        with pytest.raises(ValueError, match="unreadable"):
            cli.expected_models(tmp_path / "nope.yaml")
        no_models = tmp_path / "no_models.yaml"
        no_models.write_text("other: 1\n")
        with pytest.raises(ValueError, match="no 'models' list"):
            cli.expected_models(no_models)
        none_enabled = self._roster(tmp_path, [("a/one", False)])
        with pytest.raises(ValueError, match="no entry with enabled"):
            cli.expected_models(none_enabled)

    def test_malformed_yaml_raises_rather_than_propagating_the_parser_exception(
        self, tmp_path
    ):
        """
        A syntax error in the roster file is reported as this function's own ValueError, not
        a bare yaml.YAMLError a caller would need to know to catch.
        """
        bad = tmp_path / "bad.yaml"
        bad.write_text("models: [this is: not: valid\n")
        with pytest.raises(ValueError, match="not valid YAML"):
            cli.expected_models(bad)

    def test_a_complete_root_reads_the_derived_count_with_nothing_missing(
        self, tmp_path
    ):
        """
        With every roster model present, the header's epoch line states N of N (the two numbers
        equal) and no "Missing entirely" line appears.
        """
        _code, text = _run(tmp_path)
        assert "360 cells in all over 3 arms" in text
        assert (
            "Epoch [temperature not recorded, prompt not recorded]: 360 of 360 expected cells over 3 model(s)."
            in text
        )
        assert "Missing entirely" not in text

    def test_a_model_with_zero_arms_anywhere_is_named_in_the_header_not_only_absent_from_rows(
        self,
        tmp_path,
    ):
        """
        Deleting a whole model's arm directory (not just one level's
        worth) makes the header state the shrunken total against the full expectation and name
        the missing model, rather than reading as a smaller but complete board.
        """
        root = _copy_root(tmp_path)
        shutil.rmtree(root / "mistral-nemo-instruct-2407")
        code, text = _run(tmp_path, root, ["--expect-arms", "2"])
        assert code == cli.EXIT_OK
        assert "240 cells in all over 2 arms" in text
        assert (
            "Epoch [temperature not recorded, prompt not recorded]: 240 of 360 expected cells over 2 model(s)."
            in text
        )
        assert (
            "Missing entirely from this root (zero arms at any level): "
            "mistralai/Mistral-Nemo-Instruct-2407." in text
        )

    def test_two_models_missing_are_both_named_comma_separated_and_sorted(
        self, tmp_path
    ):
        """
        More than one absent model is named in one line, sorted, not one line per model and
        not silently only the first.
        """
        root = _copy_root(tmp_path)
        shutil.rmtree(root / "mistral-nemo-instruct-2407")
        shutil.rmtree(root / "olmo-2-1124-13b-instruct")
        code, text = _run(tmp_path, root, ["--expect-arms", "1"])
        assert code == cli.EXIT_OK
        assert "120 cells in all over 1 arms" in text
        assert (
            "Epoch [temperature not recorded, prompt not recorded]: 120 of 360 expected cells over 1 model(s)."
            in text
        )
        assert (
            "Missing entirely from this root (zero arms at any level): "
            "allenai/OLMo-2-1124-13B-Instruct, mistralai/Mistral-Nemo-Instruct-2407."
            in text
        )

    def test_a_roster_that_cannot_be_read_exits_8_and_writes_nothing(self, tmp_path):
        """
        A missing or malformed --roster-config refuses the whole run (exit 8) rather than
        silently falling back to counting whatever models were found, which would hide a
        missing model the same way.
        """
        board_file = tmp_path / "board_file.md"
        board_file.write_text(BOARD_FILE)
        code = cli.main(
            [
                "--analysis-root",
                str(FIXTURES),
                "--board-file",
                str(board_file),
                "--expect-arms",
                "3",
                "--roster-config",
                str(tmp_path / "absent.yaml"),
            ]
        )
        assert code == cli.EXIT_ROSTER
        assert board_file.read_text() == BOARD_FILE

    def test_MUTATION_deriving_expected_cells_from_found_models_hides_a_missing_model(
        self,
        tmp_path,
    ):
        """
        The named mutation from the two tests above, applied for real: computing
        expected_cells from len(inputs) instead of the roster makes the
        "missing entirely" case read as a complete board -- the exact regression these
        tests exist to prevent. Proves the tests above are not vacuous.
        """
        root = _copy_root(tmp_path)
        shutil.rmtree(root / "mistral-nemo-instruct-2407")
        cells = []
        for model in ("qwen2.5-14b-instruct", "olmo-2-1124-13b-instruct"):
            rows = pl.read_csv(
                FIXTURES / model / "properties.csv", infer_schema_length=10000
            ).to_dicts()
            cells += sb.build_cells(
                model,
                rows,
                analyse_stats.DECISION_RULE,
                analyse_stats.decision_table,
                cli.structural_by_design,
                OPPONENT_LADDER,
            )
        boards = [sb.Board(level, tuple(cells)) for level in ("M0", "M1", "M2")]
        arms = [
            sb.ArmSummary(m, sb.short_data_dir("x/" + m), "sha")
            for m in ("qwen2.5-14b-instruct", "olmo-2-1124-13b-instruct")
        ]
        # The
        mutated_expected_cells = (
            len(arms) * 3 * len(sb.SCOREBOARD_GAMES) * len(sb.PROPERTY_NUMBERS)
        )
        block = sb.render_block(
            arms,
            boards,
            analyse_stats.DECISION_RULE,
            vocabulary._DIRECTION_TEXT,
            expected_cells=mutated_expected_cells,
            missing_models=("mistralai/Mistral-Nemo-Instruct-2407",),
        )
        assert "240 of 240 expected cells" in block, (
            "the mutation should make the two numbers agree (the regression), confirming "
            "the real code path (which derives from the roster, not len(inputs)) is what "
            "the tests above are actually pinning"
        )


class TestLoadCoverageFloors:
    def test_an_unnamed_path_returns_none_not_an_error(self):
        """
        The floor is optional: no `--coverage-floor` at all renders with no floor applied.
        """
        assert cli._load_coverage_floors(None) is None

    def test_a_malformed_file_raises_rather_than_being_read_as_absent(self, tmp_path):
        """
        A present-but-broken artefact must be refused loudly, not treated the same as no
        artefact at all -- that would make every cell it was meant to gate silently read as
        if the floor had passed.
        """
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(ValueError, match="unreadable"):
            cli._load_coverage_floors(bad)

    def test_a_file_with_no_floors_key_raises(self, tmp_path):
        wrong = tmp_path / "wrong.json"
        wrong.write_text(json.dumps({"other": 1}))
        with pytest.raises(ValueError, match="no 'floors' key"):
            cli._load_coverage_floors(wrong)

    def test_a_well_formed_file_returns_its_floors_mapping(self, tmp_path):
        artefact = tmp_path / "floor.json"
        artefact.write_text(
            json.dumps(
                {
                    "floors": {"leduc": {"P1_invariance": {"floor": 0.5}}},
                    "derivation": "median",
                }
            )
        )
        assert cli._load_coverage_floors(artefact) == {
            "leduc": {"P1_invariance": {"floor": 0.5}}
        }

    def test_real_cli_applies_the_floor_and_renders_inadmissible(self, tmp_path):
        """
        Production anchor: drives `cli.main` end to end with a real `--coverage-floor` file
        over a real analysis root with a genuinely-bound synthetic probe_scores.csv (the
        same `base_dir` identity check `_load_probe_rows` applies in production), not
        `build_cells` directly.
        """
        # P1/P4 are M0-only (structural at every other level), so the fixture's
        # recorded M1 rung is rewritten to M0 -- otherwise the floor check is never reached
        # at all, regardless of what the probe row or artefact says.
        root = _copy_root(tmp_path, rung="M0")
        arm_dir = root / "qwen2.5-14b-instruct"
        condition = json.loads((arm_dir / "run_meta.json").read_text())["condition"]
        bound = sb.arm_identity(
            condition["model_id"],
            condition["memory_rung"],
            arm_reading.ROLE_BASE,
        )
        row = dict(
            _probe_row(
                game="leduc",
                prop="P1_invariance",
                value=0.3,
                n=10,
                coverage=0.9,
                n_seed_mean=8.0,
                coverage_seed_mean=0.2,
            ),
            base_dir=bound,
        )
        pl.DataFrame([row]).write_csv(arm_dir / "probe_scores.csv")
        floor_path = tmp_path / "floor.json"
        floor_path.write_text(
            json.dumps(
                {
                    "floors": {"leduc": {"P1_invariance": {"floor": 0.5}}},
                }
            )
        )
        code, text = _run(tmp_path, root, ["--coverage-floor", str(floor_path)])
        assert code == cli.EXIT_OK
        cells = _all_cells(text)
        # The row renders under the arm's own model_id, not its directory name.
        leduc_cell = cells[("M0", condition["model_id"], "leduc", "P1")]
        assert "inadmissible" in leduc_cell.lower()

    def test_the_flag_is_opt_in_a_render_with_no_flag_applies_no_floor_at_all(
        self,
        tmp_path,
    ):
        """
        Experiment 7's three boards never pass `--coverage-floor`,
        so every one of their already-accepted arms must render with no floor applied,
        even when a real, committed floor artefact that would flag
        this row sits at the conventional path on disk. The flag has no default; only an
        explicit `--coverage-floor` on the command line applies a floor.
        """
        root = _copy_root(tmp_path, rung="M0")
        arm_dir = root / "qwen2.5-14b-instruct"
        condition = json.loads((arm_dir / "run_meta.json").read_text())["condition"]
        bound = sb.arm_identity(
            condition["model_id"],
            condition["memory_rung"],
            arm_reading.ROLE_BASE,
        )
        row = dict(
            _probe_row(
                game="leduc",
                prop="P1_invariance",
                value=0.3,
                n=10,
                coverage=0.9,
                n_seed_mean=8.0,
                coverage_seed_mean=0.2,
            ),
            base_dir=bound,
        )
        pl.DataFrame([row]).write_csv(arm_dir / "probe_scores.csv")
        # A real floor artefact that would flag this exact row, sitting on disk -- proving
        # its mere presence is not enough; only an explicit flag applies it.
        artefact_path = tmp_path / "floor.json"
        artefact_path.write_text(
            json.dumps(
                {
                    "floors": {"leduc": {"P1_invariance": {"floor": 0.5}}},
                }
            )
        )
        code, text = _run(tmp_path, root, [])  # no --coverage-floor at all
        assert code == cli.EXIT_OK
        cells = _all_cells(text)
        leduc_cell = cells[("M0", condition["model_id"], "leduc", "P1")]
        assert "inadmissible" not in leduc_cell.lower()

    def test_a_broken_coverage_floor_file_exits_6_and_writes_nothing(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        board_file = tmp_path / "board_file.md"
        board_file.write_text(BOARD_FILE)
        code = cli.main(
            [
                "--analysis-root",
                str(FIXTURES),
                "--board-file",
                str(board_file),
                "--expect-arms",
                "3",
                "--roster-config",
                str(ROSTER_CONFIG),
                "--coverage-floor",
                str(bad),
            ]
        )
        assert code == cli.EXIT_INPUT
        assert board_file.read_text() == BOARD_FILE


class TestCombineEveryPairOfParts:
    """
    a two-part cell (P7's 7a and 7b) over every (kind, kind) pair of its parts
    """

    SAMPLES = {
        sb.NUMBER: sb.Cell(
            sb.NUMBER, "0.1763 [0.1345, 0.2540] (4/4 opponents)", "0.1763 [full]"
        ),
        sb.PARTIAL: sb.Cell(
            sb.PARTIAL,
            "partial: 1/4 opponents; mean 0.6466",
            "partial: 1/4 opponents; full",
        ),
        sb.PARTIAL_APPROVED: sb.Cell(
            sb.PARTIAL_APPROVED,
            "partial: 3/4 opponents; mean 0.5000",
            "partial: 3/4 opponents; full",
        ),
        sb.AWAITING: sb.Cell(
            sb.AWAITING, "awaiting data: no run", "awaiting data: no run"
        ),
        sb.BROKEN: sb.Cell(sb.BROKEN, "BROKEN: no row", "BROKEN: no row"),
        sb.INADMISSIBLE: sb.Cell(
            sb.INADMISSIBLE, "inadmissible: 0.4 shown", "inadmissible: 0.4 shown"
        ),
        sb.STRUCTURAL: sb.Cell(
            sb.STRUCTURAL, "structural: by design", "structural: by design"
        ),
    }

    @pytest.mark.parametrize("first", list(SAMPLES))
    @pytest.mark.parametrize("second", list(SAMPLES))
    def test_no_part_is_called_ok_unless_it_carries_a_number(self, first, second):
        """
        Every pair of part kinds: a partial part beside an unscored one reads "partial <mean>
        (k/n opponents)", carrying its mean as a number part carries its value, and never "ok
        partial:"; a number part reads "ok <value>", and the combined kind is unchanged.
        """
        cell = sb.combine(["7a", "7b"], [self.SAMPLES[first], self.SAMPLES[second]])
        for text in (cell.text, cell.full_text):
            assert "ok partial" not in text
            for label, kind in (("7a", first), ("7b", second)):
                if cell.kind in (sb.AWAITING, sb.BROKEN, sb.INADMISSIBLE):
                    if kind == sb.PARTIAL:
                        assert f"{label} partial " in text and "opponents)" in text
                    if kind == sb.PARTIAL_APPROVED:
                        assert f"{label} partial, approved " in text
                    if kind == sb.NUMBER:
                        assert f"{label} ok 0.1763" in text
        for label, kind in (("7a", first), ("7b", second)):
            if cell.kind in (sb.AWAITING, sb.BROKEN, sb.INADMISSIBLE):
                if kind == sb.PARTIAL:
                    assert f"{label} partial 0.6466 (1/4 opponents)" in cell.text
                if kind == sb.PARTIAL_APPROVED:
                    assert (
                        f"{label} partial, approved 0.5000 (3/4 opponents)" in cell.text
                    )

    def test_the_combined_kind_follows_the_worst_part(self):
        """
        The wording change moves no kind: a partial part beside a broken one is BROKEN, beside an
        awaiting one is awaiting, beside an inadmissible one is inadmissible.
        """
        partial = self.SAMPLES[sb.PARTIAL]
        for other in (sb.BROKEN, sb.AWAITING, sb.INADMISSIBLE):
            assert (
                sb.combine(["7a", "7b"], [partial, self.SAMPLES[other]]).kind == other
            )


def _full_root(tmp_path):
    """
    a root holding every arm of the fixture roster: each model at M0, M1 and M2
    :param tmp_path: pytest tmp_path
    :returns: the root (Path)
    """
    root = _copy_root(tmp_path, rung="M0")
    for arm in MODEL_OF:
        for rung in ("M1", "M2"):
            shutil.copytree(root / arm, root / f"{arm}-{rung.lower()}")
            _set_rung(root, f"{arm}-{rung.lower()}", rung)
    return root


class TestPartialRoots:
    """A root that lacks arms of the roster is refused unless `--partial` says it is meant."""

    def test_a_root_missing_roster_arms_exits_9_and_writes_nothing(
        self, tmp_path, capsys
    ):
        """
        The fixture root holds three arms of the nine the roster expects. Without `--partial`
        the run exits 9, names the pairs it lacks and the flag, and leaves the board file as it
        was.
        """
        code, text = _run(tmp_path, partial=False)
        assert code == cli.EXIT_PARTIAL == 9
        assert text == BOARD_FILE
        err = capsys.readouterr().err
        assert "--partial" in err and "lacks 6 of 9 roster arms" in err
        assert "mistralai/Mistral-Nemo-Instruct-2407 M0" in err

    def test_partial_renders_the_header_naming_the_missing_arms(self, tmp_path):
        """
        With `--partial` the same root renders, and the first header lines say how many arms it
        holds of how many and name every missing one, sorted.
        """
        code, text = _run(tmp_path, partial=True)
        assert code == cli.EXIT_OK
        line = next(line for line in text.splitlines() if line.startswith("PARTIAL: "))
        found, expected, names = re.fullmatch(
            r"PARTIAL: (\d+) of (\d+) arms; missing: (.+)\.", line
        ).groups()
        assert (found, expected) == ("3", "9")
        missing = names.split(", ")
        assert len(missing) == 6 and missing == sorted(missing)
        assert "mistralai/Mistral-Nemo-Instruct-2407 M0" in missing

    def test_a_complete_root_renders_the_same_with_or_without_the_flag(self, tmp_path):
        """
        A root with every roster arm exits 0 without `--partial`, prints no PARTIAL line, and is
        byte-identical to the render that passed the flag.
        """
        root = _full_root(tmp_path)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        code, plain = _run(tmp_path / "a", root, ["--expect-arms", "9"], partial=False)
        assert code == cli.EXIT_OK and "PARTIAL:" not in plain
        flagged = _run(tmp_path / "b", root, ["--expect-arms", "9"], partial=True)[1]
        assert flagged == plain

    def test_two_epochs_of_one_model_at_one_level_are_one_arm(self, tmp_path):
        """
        A tenth arm that is a second epoch of Qwen at M0 adds no roster arm: the root is still
        complete and still exits 0 without `--partial`.
        """
        root = _full_root(tmp_path)
        shutil.copytree(root / "qwen2.5-14b-instruct", root / "qwen-second")
        meta_path = root / "qwen-second" / "run_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["condition"] = dict(meta["condition"], temperature=1.0)
        meta_path.write_text(json.dumps(meta))
        (tmp_path / "e").mkdir()
        code, text = _run(tmp_path / "e", root, ["--expect-arms", "10"], partial=False)
        assert code == cli.EXIT_OK and "PARTIAL:" not in text

    def test_check_applies_the_same_rule(self, tmp_path):
        """
        `--check` on a partial root exits 9 without `--partial` and 0 with it once the block is
        written.
        """
        board_file = tmp_path / "board_file.md"
        board_file.write_text(BOARD_FILE)
        base = [
            "--analysis-root",
            str(FIXTURES),
            "--board-file",
            str(board_file),
            "--expect-arms",
            "3",
            "--roster-config",
            str(ROSTER_CONFIG),
        ]
        assert cli.main([*base, "--partial"]) == cli.EXIT_OK
        assert cli.main([*base, "--check", "--partial"]) == cli.EXIT_OK
        assert cli.main([*base, "--check"]) == cli.EXIT_PARTIAL

    def test_the_arms_come_from_the_roster_config(self, tmp_path):
        """
        A fourth enabled model in the roster file adds three missing arms (one per level) to a
        root that otherwise holds every arm of the three, so the expectation is read from the
        roster the cell count uses and not a copy.
        """
        roster = tmp_path / "roster.yaml"
        roster.write_text(
            ROSTER_CONFIG.read_text() + "  - id: extra/Extra-Model\n    enabled: true\n"
        )
        root = _full_root(tmp_path)
        board_file = tmp_path / "board_file.md"
        board_file.write_text(BOARD_FILE)
        base = [
            "--analysis-root",
            str(root),
            "--board-file",
            str(board_file),
            "--expect-arms",
            "9",
            "--roster-config",
            str(roster),
        ]
        assert cli.main(base) == cli.EXIT_PARTIAL
        assert cli.main([*base, "--partial"]) == cli.EXIT_OK
        line = next(
            line for line in board_file.read_text().splitlines() if "PARTIAL:" in line
        )
        assert "PARTIAL: 9 of 12 arms" in line
        assert (
            "extra/Extra-Model M0, extra/Extra-Model M1, extra/Extra-Model M2" in line
        )


class TestPayoffLegLabel:
    """A P4 cell that reports a measurement says it is the invariance leg only."""

    def test_a_p4_number_carries_the_label_through_the_real_scripts(self, tmp_path):
        """
        The scorer's own Kuhn P1 row, relabelled as the P4 row, renders through
        `scoreboard.py` as a number ending in the label; the P1 number beside it and the P4
        cells that make no measurement (here Leduc, BROKEN for want of a payoff arm) do not carry it.
        """
        root, csv_path, arm = _score_root(tmp_path)
        with open(csv_path, newline="") as fh:
            table = list(_csvmod.DictReader(fh))
        p1 = next(
            r for r in table if r["game"] == "kuhn" and r["property"].startswith("P1")
        )
        index = next(
            i
            for i, r in enumerate(table)
            if r["game"] == "kuhn" and r["property"].startswith("P4")
        )
        table[index] = dict(p1, property=table[index]["property"])
        with open(csv_path, "w", newline="") as fh:
            w = _csvmod.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        cells = _cells(_run(tmp_path, root)[1], "M0")
        label = sb.P4_LEG_LABEL
        assert _kind(cells[(arm, "kuhn", "P4")]) == "number"
        assert cells[(arm, "kuhn", "P4")].endswith(label)
        assert label not in cells[(arm, "kuhn", "P1")]
        assert _kind(cells[(arm, "leduc", "P4")]) != "number"
        assert label not in cells[(arm, "leduc", "P4")]

    @pytest.mark.parametrize(
        "kind, labelled",
        [
            (sb.NUMBER, True),
            (sb.INADMISSIBLE, True),
            (sb.AWAITING, False),
            (sb.BROKEN, False),
            (sb.STRUCTURAL, False),
        ],
    )
    def test_only_a_cell_that_reports_a_measurement_is_labelled(self, kind, labelled):
        """
        A P4 number or inadmissible value is labelled in both texts; a cell that makes no P4
        claim is not, and any other property is never labelled.
        """
        cell = sb.Cell(kind, "text", "full text")
        out = sb._legged(cell, sb.P4_NUMBER)
        assert out.text.endswith(sb.P4_LEG_LABEL) == labelled
        assert out.full_text.endswith(sb.P4_LEG_LABEL) == labelled
        assert out.kind == kind
        assert sb._legged(cell, 1) == cell


class TestSilentFallbackRefusals:
    @pytest.mark.parametrize("spread", [None, float("nan"), float("inf")])
    def test_replicate_clause_omits_a_non_finite_spread(self, spread):
        """
        differing per-seed values with no finite spread state their counts and drop the spread
        clause, rather than raising on the format or printing "spread nan"
        """
        assert sb.replicate_clause(3, 2, spread) == "3 seeds, 2 distinct values"

    def test_replicate_clause_keeps_a_finite_spread(self):
        """
        a finite spread is still stated
        """
        assert (
            sb.replicate_clause(3, 2, 0.125)
            == "3 seeds, 2 distinct values, spread 0.125"
        )

    def test_distribution_basis_row_without_a_finite_value_exact_is_broken(self):
        """
        on the distribution basis every scored row must carry a finite `value_exact`; one that
        does not renders a BROKEN cell naming the opponent, not a TypeError at render time
        """
        rows = _opponents([0.1, 0.2, 0.3, 0.4])
        for r, exact in zip(rows, [0.11, None, 0.31, float("nan")], strict=True):
            r["value_exact"] = exact
        decision = {
            "observed_mean_admissible": 0.25,
            "basis": "distribution",
            "n_cells_admissible": 4,
            "verdict": "",
        }
        cell = sb.classify(
            "P3_randomisation_margin", decision, rows, False, OPPONENT_LADDER
        )
        assert cell.kind == sb.BROKEN, cell
        assert OPPONENT_LADDER[1] in cell.text and OPPONENT_LADDER[3] in cell.text
        assert "value_exact" in cell.text

    def test_combine_keeps_an_inadmissible_part_beside_an_awaiting_part(self):
        """
        an inadmissible part combined with an awaiting part keeps its own text, never the
        word "structural", so the withheld verdict stays visible
        """
        cell = sb.combine(
            ["7a", "7b"],
            [
                sb.Cell(sb.INADMISSIBLE, "inadmissible: gate withheld x"),
                sb.Cell(sb.AWAITING, "awaiting data: no rows"),
            ],
        )
        assert cell.kind == sb.AWAITING
        assert "7a structural" not in cell.text
        assert "7a inadmissible: gate withheld x" in cell.text, cell.text

    def test_legged_keeps_the_legacy_blocker_rule_flag(self):
        """
        appending the P4 leg label keeps the cell's legacy-wording flag, so the stderr count of
        cells classified by the legacy rule still sees a P4 cell
        """
        cell = sb.Cell(sb.INADMISSIBLE, "inadmissible: x", legacy_blocker_rule=True)
        legged = sb._legged(cell, sb.P4_NUMBER)
        assert legged.text.endswith(sb.P4_LEG_LABEL)
        assert legged.legacy_blocker_rule is True
