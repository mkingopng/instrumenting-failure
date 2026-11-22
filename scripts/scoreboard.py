#!/usr/bin/env python
"""
Scoreboard entrypoint: regenerate one 120-cell board (models x games x properties) per
memory level from a root of scored arm directories and rewrite the text between the
SCOREBOARD markers of a markdown file.

Usage:
    .venv/bin/python scripts/scoreboard.py --analysis-root analysis/recycle-v6-m1

The root holds one directory per condition of a model, each written by
`scripts/analyse_stats.py` (`properties.csv`, `decision.csv`, `outcomes.csv`) plus its
`run_meta.json`, and an optional `FOOTNOTES.md` whose non-empty lines are appended under the
tables. Arms group into models by the `condition` block their run_meta records, and every
cell of a board reads that board's memory level. Scoring is a separate step and is never run
here: it takes hours per condition.

Every input the block is built from lives under the root, so a committed block can be
verified against it with `--check`.

Exit codes:
    0  block written (or, with --check, already current)
    1  an arm was refused, a required column is missing, or run_meta names no data_dir
    2  the markdown file's markers are missing, duplicated or out of order
    3  no arms found in the root
    4  the number of arms differs from --expect-arms
    5  --check found a stale block
    6  the analysis root or the markdown file is missing or unreadable
    7  --expect-arms was not given
    8  --roster-config could not be read as a model roster
    9  the root lacks arms of the roster (a model at a memory level) and --partial was not given
    10 --manifest was given and the root differs from it (each difference is printed)

An arm's P1, P4 and P6 cells are read from its `probe_scores.csv` (written by
`scripts/probe_scores.py`); a file bound to a different arm, or missing a column, refuses the run
with exit 1.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import polars as pl
import yaml

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmopt.analysis.scoreboard import scoreboard_sources as board_manifest
from lmopt.analysis import probe_scores
from lmopt.analysis.scoreboard import scoreboard as sb
from lmopt.analysis.scoreboard import scoreboard_sources as sources
from lmopt.analysis.condition import condition_epoch, condition_of
from lmopt.analysis.cross_arm import (
    OPPONENT_LADDER,
    SPARSE_NUMERIC_COLUMN_SCHEMA,
    discover_arm_artifacts,
)
from lmopt.games.families import GAME_FAMILIES
from lmopt.analysis import scoring_vocabulary as vocabulary
from scripts import analyse_stats
from lmopt.analysis import arm_reading

EXIT_OK = 0
EXIT_ARM_PROBLEM = 1
EXIT_MARKERS = 2
EXIT_NO_ARMS = 3
EXIT_ARM_COUNT = 4
EXIT_STALE = 5
EXIT_INPUT = 6
EXIT_NO_EXPECTATION = 7
EXIT_ROSTER = 8
EXIT_PARTIAL = 9
EXIT_MANIFEST = 10

#: "SINGLE source for the full-size roster" (configs/run_experiment.yaml's own docstring).
#: A model absent from the analysis root must be visible in the header, not only in
#: the row list, and deriving the expected count from here means it cannot go stale the
#: way a hand-typed --expect-cells integer would the first time a game or property is added.
DEFAULT_ROSTER_CONFIG = REPO_ROOT / "configs" / "run_experiment.yaml"

#: The family P8 is defined for; `score_property8` applies the same test.
P8_FAMILY = "darkhex"
BELIEF_PROPERTY_NUMBERS = (5, 7)
P8_NUMBER = 8


def structural_by_design(
    game: str, number: int, memory_rung: str | None = None
) -> bool:
    """
    whether a property cannot exist for a game's family at a memory level, by the scorers'
    own family tests and the source declaration's own level list
    :param game: a game key
    :param number: the property number 1 to 8
    :param memory_rung: the memory level the run was collected at, None counting as M0
    :returns: True for P5 and P7 on a family whose belief is scored by P8, for P8 on
        every family but Dark Hex, and for a property the source declaration marks
        inapplicable at this memory level; False otherwise, so a missing probe run or a
        missing reference can never read as a structural absence
    """
    family = GAME_FAMILIES.get(game, game)
    if sources.structural_at(number, memory_rung):
        return True
    if number in BELIEF_PROPERTY_NUMBERS:
        return family in analyse_stats._BELIEF_NOT_APPLICABLE_FAMILIES
    if number == P8_NUMBER:
        return family != P8_FAMILY
    return False


def _read_meta(arm_dir: Path) -> dict:
    """
    :param arm_dir: a scored arm directory
    :returns: its parsed run_meta.json
    :raises ValueError: when run_meta.json is missing or unreadable
    """
    path = arm_dir / "run_meta.json"
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{arm_dir.name}: run_meta.json unreadable ({exc})") from exc
    return meta if isinstance(meta, dict) else {}


def _read_data_dir(arm_dir: Path) -> str:
    """
    :param arm_dir: a scored arm directory
    :returns: the `data_dir` its run_meta.json names
    :raises ValueError: when run_meta.json is missing, unreadable or names no data_dir
    """
    meta = _read_meta(arm_dir)
    data_dir = meta.get("data_dir") if isinstance(meta, dict) else None
    if not data_dir:
        raise ValueError(f"{arm_dir.name}: run_meta.json names no data_dir")
    return str(data_dir)


def expected_models(roster_config: Path) -> tuple[str, ...]:
    """
    the model ids a board's cell count should be derived from, so a model with zero
    scored arms still shrinks the board visibly rather than silently
    :param roster_config: `configs/run_experiment.yaml`, documented there as "SINGLE
        source for the full-size roster"; its `models:` list
        entries carry the same full HF-style `id` string `condition_of()` reads into
        `model_id`, confirmed against the current board's own per-model rows
    :returns: the `id` of every entry whose `enabled` is true, sorted; never empty
    :raises ValueError: when the file is missing or unreadable, is not a mapping with a
        `models` list, or that list yields zero enabled entries -- a silently empty
        roster would make every model in the root read as "unexpected" rather than
        surfacing that the roster itself could not be read
    """
    try:
        text = roster_config.read_text()
    except OSError as exc:
        raise ValueError(f"{roster_config}: unreadable ({exc})") from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{roster_config}: not valid YAML ({exc})") from exc
    models = (doc or {}).get("models") if isinstance(doc, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"{roster_config}: no 'models' list")
    ids = sorted(
        {
            entry["id"]
            for entry in models
            if isinstance(entry, dict) and entry.get("enabled") and entry.get("id")
        }
    )
    if not ids:
        raise ValueError(f"{roster_config}: 'models' has no entry with enabled: true")
    return tuple(ids)


def _load_properties(arm_dir: Path) -> list[dict]:
    """
    read one arm's properties.csv with a full-file schema scan
    :param arm_dir: a scored arm directory
    :returns: the rows as dicts; the loader `discover_arm_artifacts` uses infers a type from
        the first 100 rows only, which mistypes `value` as text when those rows are all null
    :raises ValueError: when a required column is missing, or `value` is not numeric on a
        table that has rows (an empty table is valid and yields cells reading BROKEN)
    """
    df = pl.read_csv(
        arm_dir / "properties.csv",
        infer_schema_length=None,
        schema_overrides=SPARSE_NUMERIC_COLUMN_SCHEMA,
    )
    sb.check_required_columns(df.columns)
    dtype = df.schema["value"]
    if df.height and not (dtype.is_numeric() or dtype == pl.Null):
        raise ValueError(f"column 'value' reads as {dtype}, expected numeric")
    return df.to_dicts()


PROBE_FILE = "probe_scores.csv"
#: Footnotes live beside the data, never on the command line, so the block is a function of
#: its inputs and `--check` can verify a committed block against them.
FOOTNOTES_FILE = "FOOTNOTES.md"
PROBE_REQUIRED_COLUMNS = (
    "game",
    "property",
    "value",
    "status",
    "caveat",
    "n_infosets",
    "paired_coverage",
    "base_dir",
)
#: The distribution-basis numbers of an off-path or paired row. Not required: a file written before
#: the basis existed, and any row of an arm with no pass, simply carries none, which the cell reads
#: as no distribution basis rather than as a malformed file.
PROBE_EXACT_COLUMNS = (
    "value_exact",
    "ci_low_exact",
    "ci_high_exact",
    "baseline_exact",
    "exact_coverage",
    "exact_coverage_mass",
)
#: The replicate diagnostics of an off-path row, per basis. Like the distribution-basis numbers
#: they are optional: a file written before they existed carries none, and the cell then says
#: nothing about them.
PROBE_REPLICATE_COLUMNS = (
    "n_replicates",
    "n_replicates_distinct",
    "n_replicates_exact",
    "n_replicates_distinct_exact",
)


def _number_or_none(text: str):
    """
    :param text: a CSV field
    :returns: the float it holds, or None for an empty field
    :raises ValueError: when a non-empty field is not a number, or the field is absent (a short row)
    """
    if text is None:
        raise ValueError("field absent: the row is shorter than the header")
    return None if text == "" else float(text)


def _load_probe_rows(arm_dir: Path, condition: dict | None):
    """
    read an arm's probe_scores.csv and check it belongs to this arm

    The check binds on the arm's own recorded identity (model, memory level, and the base
    role every row in this file is scored relative to), never on a directory path: two arms
    whose directories happen to share their last path components, such as a relabel and a
    payoff-scale probe run under the same index, must not be mistaken for the same arm here.
    :param arm_dir: a scored arm directory
    :param condition: the arm's own `condition` block (model_id, memory_rung, ...), or None
        when the arm never recorded one
    :returns: the rows as dicts (every numeric column as a number or None), or
        None when the arm has no probe_scores.csv
    :raises ValueError: when a required column is missing, a number does not parse, or a row's
        `base_dir` is not this arm's own identity
    """
    path = arm_dir / PROBE_FILE
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [
            c for c in PROBE_REQUIRED_COLUMNS if c not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(
                f"{PROBE_FILE} lacks required column(s): {', '.join(missing)}"
            )
        table = list(reader)
    bound = sb.arm_identity(
        (condition or {}).get("model_id"),
        (condition or {}).get("memory_rung"),
        arm_reading.ROLE_BASE,
    )
    rows = []
    for record in table:
        if record["base_dir"] != bound:
            raise ValueError(
                f"{PROBE_FILE} is bound to {record['base_dir']!r} but this arm's identity is "
                f"{bound!r}"
            )
        rows.append(
            dict(
                record,
                value=_number_or_none(record["value"]),
                n_infosets=_number_or_none(record["n_infosets"]),
                paired_coverage=_number_or_none(record["paired_coverage"]),
                dropped_rate=_number_or_none(record.get("dropped_rate", "")),
                n_infosets_seed_mean=_number_or_none(
                    record.get("n_infosets_seed_mean", "")
                ),
                paired_coverage_seed_mean=_number_or_none(
                    record.get("paired_coverage_seed_mean", "")
                ),
                **{
                    c: _number_or_none(record.get(c, ""))
                    for c in PROBE_EXACT_COLUMNS + PROBE_REPLICATE_COLUMNS
                },
            )
        )
    return rows


def _load_collapse(arm_dir: Path):
    """
    read an arm's collapse record, written by `scripts/arm_collapse_pass.py`
    :param arm_dir: a scored arm directory
    :returns: {game: row} with `first_listed_share` as a number or None, or None when the arm
        has no collapse file and was therefore never checked
    :raises ValueError: when a required column is missing, a number does not parse, or a game
        appears twice
    """
    path = arm_dir / sb.COLLAPSE_FILE
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in sb.COLLAPSE_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(
                f"{sb.COLLAPSE_FILE} lacks required column(s): {', '.join(missing)}"
            )
        table = list(reader)
    by_game = {}
    for record in table:
        game = record["game"]
        if game in by_game:
            raise ValueError(f"{sb.COLLAPSE_FILE} has two rows for {game}")
        by_game[game] = dict(
            record, first_listed_share=_number_or_none(record["first_listed_share"])
        )
    return by_game


def _load_collapse_offpath(arm_dir: Path):
    """
    read an arm's off-path collapse record, written by `scripts/arm_collapse_pass.py --offpath-root`
    :param arm_dir: a scored arm directory
    :returns: {game: row}, or None when the arm has no such file and its driven play was
        therefore never checked
    :raises ValueError: when a required column is missing or a game appears twice
    """
    path = arm_dir / sb.COLLAPSE_OFFPATH_FILE
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            c for c in sb.COLLAPSE_OFFPATH_COLUMNS if c not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(
                f"{sb.COLLAPSE_OFFPATH_FILE} lacks required column(s): {', '.join(missing)}"
            )
        table = list(reader)
    by_game = {}
    for record in table:
        game = record["game"]
        if game in by_game:
            raise ValueError(f"{sb.COLLAPSE_OFFPATH_FILE} has two rows for {game}")
        by_game[game] = dict(record)
    return by_game


def _write_preserving(path: Path, text: str) -> None:
    """
    replace a file's contents, keeping its mode and following a symlink
    :param path: the file to rewrite
    :param text: the new contents, written as UTF-8 with no newline translation
    :returns: None
    """
    target = path.resolve()
    tmp = target.with_name(target.name + ".scoreboard.tmp")
    tmp.write_bytes(text.encode("utf-8"))
    shutil.copymode(target, tmp)
    os.replace(tmp, target)


def parse_args(argv=None):
    """
    :param argv: argument list, or None for sys.argv
    :returns: the parsed namespace
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--analysis-root",
        required=True,
        help="Directory of scored arm directories, one per model.",
    )
    parser.add_argument(
        "--board-file",
        default=str(REPO_ROOT / "scoreboard.md"),
        help="File holding the SCOREBOARD markers.",
    )
    parser.add_argument(
        "--expect-arms",
        type=int,
        default=None,
        help="Required. Exit 4 unless the root yields exactly this many arms; "
        "exit 7 when omitted, because a count read back from the root "
        "would pass for any root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 5 if the file's block differs from a fresh one; write nothing.",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Render a root that lacks arms of the roster (a model at a memory level). "
        "Without it such a root exits 9 and writes nothing; with it the header says "
        "'PARTIAL: N of M arms' and names the missing ones. A complete root renders the "
        "same either way, and --check applies the same rule.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="A committed manifest of the root (scripts/board_manifest.py). The root is checked "
        "against it before anything is rendered or compared, and a missing, extra or changed "
        "file exits 10, because a block rendered from bytes the repository does not record is "
        "not reproducible from it. Applies to --check as well.",
    )
    parser.add_argument(
        "--roster-config",
        default=str(DEFAULT_ROSTER_CONFIG),
        help="Model roster the expected cell count is derived from; "
        "default configs/run_experiment.yaml, its own documented "
        "single source for the full-size roster.",
    )
    parser.add_argument(
        "--coverage-floor",
        default=None,
        help="The pre-registered paired_coverage floor artefact. OPT-IN, no "
        "default: a render that does not pass this flag applies no "
        "floor at all -- the three "
        "experiment-7 boards never pass it, since every one of their "
        "accepted arms would otherwise be re-judged against a cell "
        "kind their own acceptance never considered. The command line "
        "that DOES pass it is itself the record of that decision.",
    )
    parser.add_argument(
        "--epoch-denominator",
        choices=EPOCH_DENOMINATOR_MODES,
        default=EPOCH_PER_EPOCH,
        help="How the header's expected-cell count treats a root holding more than one "
        "collection epoch (temperature and prompt version) of a model. 'per-epoch' (the "
        "default) prints one line per epoch, each against a full roster, and drops the "
        "denominator from the first line; 'single' counts one epoch per model, so a "
        "two-epoch root reads more cells than it expects; 'scale' counts one full set of "
        "cells per (model, epoch) row found, plus one per roster model with no arm.",
    )
    return parser.parse_args(argv)


def _load_coverage_floors(path: Path | None) -> dict | None:
    """
    the pre-registered floor artefact, read once per run
    :param path: the artefact path, or None when `--coverage-floor` was not given at all --
        the flag is opt-in with no default, so this is the ordinary case for every render
        that does not explicitly ask for a floor
    :returns: its "floors" mapping ({game: {property: {"floor": float, ...}}}), or None when
        no path was given
    :raises ValueError: a path was given and it does not exist, or exists but is not valid JSON, or carries no
        "floors" key -- a present-but-broken artefact is refused rather than silently
        treated as absent, which would make every cell it was meant to gate read as if the
        floor had passed
    """
    if path is None:
        return None
    if not path.exists():
        raise ValueError(
            f"{path}: the coverage-floor artefact named on the command line does not exist"
        )
    try:
        parsed = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: unreadable coverage-floor artefact ({exc})") from exc
    if "floors" not in parsed:
        raise ValueError(
            f"{path}: carries no 'floors' key, so it is not a floor artefact"
        )
    return parsed["floors"]


def read_footnotes(analysis_root: Path) -> list[str]:
    """
    the lines appended under the table, which live beside the data and not on the command line
    :param analysis_root: the root of scored arm directories
    :returns: the non-empty lines of FOOTNOTES.md in that root, in file order, or an empty
        list when the root has no such file
    :raises ValueError: when the file exists and cannot be read
    """
    path = Path(analysis_root) / FOOTNOTES_FILE
    if not path.exists():
        return []
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, ValueError) as exc:
        raise ValueError(f"{FOOTNOTES_FILE} unreadable ({exc})") from exc
    return [line.rstrip() for line in text.splitlines() if line.strip()]


#: How the header's expected-cell count treats more than one collection epoch of a model.
EPOCH_SINGLE = "single"
EPOCH_SCALE = "scale"
EPOCH_PER_EPOCH = "per-epoch"
EPOCH_DENOMINATOR_MODES = (EPOCH_SINGLE, EPOCH_SCALE, EPOCH_PER_EPOCH)


def epoch_label(epoch: tuple) -> str:
    """
    :param epoch: (temperature, prompt_schema_version) as `condition_epoch` returns it
    :returns: the bracketed words a board row and a header line use for it (str)
    """
    temperature, version = epoch
    said_temperature = "not recorded" if temperature is None else temperature
    said_version = "not recorded" if version is None else f"v{version}"
    return f"[temperature {said_temperature}, prompt {said_version}]"


def missing_arms(roster, staged) -> tuple[str, ...]:
    """
    the roster's arms the root does not hold: one per model and memory level
    :param roster: the roster's model ids, from `expected_models`, the same single source the
        expected cell count is derived from (Sequence[str])
    :param staged: [(model, epoch, ArmInput)] as the board rows are built from (list)
    :returns: "<model> <level>" per missing pair, sorted; two epochs of one model at one level
        are one arm (tuple[str, ...])
    """
    held = {
        (model, (arm.condition or {}).get("memory_rung") or sources.M0)
        for model, _epoch, arm in staged
    }
    return tuple(
        f"{model} {level}"
        for model in sorted(roster)
        for level in sources.BOARD_LEVELS
        if (model, level) not in held
    )


def epoch_header_lines(staged, boards, roster, cells_per_model: int) -> tuple[str, ...]:
    """
    one header line per collection epoch: its cells against a full roster's worth
    :param staged: [(model, epoch, ArmInput)] as the board rows were built from (list)
    :param boards: the Boards rendered (Sequence[sb.Board])
    :param roster: the roster's model ids (Sequence[str])
    :param cells_per_model: the cells one model contributes at one epoch (int)
    :returns: the lines, epochs in sorted order (tuple[str, ...])
    """
    labels_of: dict[tuple, set] = {}
    epochs_of: dict[str, set] = {}
    for model, epoch, _arm in staged:
        epochs_of.setdefault(model, set()).add(epoch)
    for model, epoch, _arm in staged:
        label = model if len(epochs_of[model]) == 1 else f"{model} {epoch_label(epoch)}"
        labels_of.setdefault(epoch, set()).add(label)
    lines = []
    for epoch in sorted(labels_of, key=repr):
        n = sum(
            1
            for board in boards
            for cell in board.cells
            if cell.arm in labels_of[epoch]
        )
        lines.append(
            f"Epoch {epoch_label(epoch)}: {n} of {len(roster) * cells_per_model} expected "
            f"cells over {len({m for m, e, _a in staged if e == epoch})} model(s)."
        )
    return tuple(lines)


def group_by_epoch(staged) -> dict:
    """
    group scored arms into the rows a board draws, one per model and collection epoch
    :param staged: [(model, epoch, ArmInput)] with epoch as `condition_epoch` returns it
    :returns: {row label: [ArmInput]}; a model whose arms all share one epoch keeps its plain
        name, and a model with arms from more than one epoch gets one row per epoch labelled
        with the temperature and prompt schema version, so a run at each is two rows and
        never two candidates for one cell
    """
    epochs_of: dict[str, set] = {}
    for model, epoch, _arm in staged:
        epochs_of.setdefault(model, set()).add(epoch)
    inputs: dict[str, list] = {}
    for model, epoch, arm in staged:
        label = model
        if len(epochs_of[model]) > 1:
            label = f"{model} {epoch_label(epoch)}"
        inputs.setdefault(label, []).append(arm)
    return inputs


def main(argv=None) -> int:
    """
    build the scoreboard and splice it into the markdown file
    :param argv: argument list, or None for sys.argv
    :returns: the exit code documented in the module docstring
    """
    args = parse_args(argv)
    if args.expect_arms is None:
        print(
            "--expect-arms is required: state how many arms the root should hold",
            file=sys.stderr,
        )
        return EXIT_NO_EXPECTATION
    try:
        coverage_floors = _load_coverage_floors(
            Path(args.coverage_floor) if args.coverage_floor is not None else None
        )
    except ValueError as exc:
        print(f"coverage floor: {exc}", file=sys.stderr)
        return EXIT_INPUT
    if args.manifest is not None:
        try:
            differences = board_manifest.verify(
                args.analysis_root, board_manifest.load_manifest(args.manifest)
            )
        except ValueError as exc:
            print(f"manifest: {exc}", file=sys.stderr)
            return EXIT_INPUT
        for difference in differences:
            print(f"MANIFEST: {difference}", file=sys.stderr)
        if differences:
            return EXIT_MANIFEST
    try:
        arms, problems = discover_arm_artifacts(args.analysis_root)
    except ValueError as exc:
        print(f"analysis root: {exc}", file=sys.stderr)
        return EXIT_INPUT
    for problem in problems:
        print(f"REFUSED: {problem}", file=sys.stderr)
    if problems:
        return EXIT_ARM_PROBLEM
    if not arms:
        print(f"no arms found in {args.analysis_root}", file=sys.stderr)
        return EXIT_NO_ARMS
    if len(arms) != args.expect_arms:
        print(
            f"found {len(arms)} arms {sorted(arms)}, expected {args.expect_arms}",
            file=sys.stderr,
        )
        return EXIT_ARM_COUNT

    try:
        roster = expected_models(Path(args.roster_config))
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_ROSTER

    try:
        heads = sb.headline_rows(vocabulary.DECISION_RULE)
        sources.check_probe_sources(vocabulary.PROPERTY_STATUS, heads)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_ARM_PROBLEM
    probe_names = tuple(sorted(sources.probe_roles(heads)))

    licensed_hands = analyse_stats.p2_hand_bins_licensed_floor()
    summaries: list = []
    staged: list = []
    for name in sorted(arms):
        arm = arms[name]
        try:
            rows = _load_properties(arm.path)
            data_dir = _read_data_dir(arm.path)
            condition = condition_of(_read_meta(arm.path))
            probe_rows = _load_probe_rows(arm.path, condition)
            violations = probe_scores.withholding_violations(probe_rows or [])
            if violations:
                raise ValueError(
                    f"{PROBE_FILE} breaks the zero-baseline withholding rule: "
                    + "; ".join(violations)
                )
            collapse = _load_collapse(arm.path)
            collapse_offpath = _load_collapse_offpath(arm.path)
            arm_cells = sb.build_cells(
                name,
                rows,
                vocabulary.DECISION_RULE,
                vocabulary.decision_table,
                structural_by_design,
                OPPONENT_LADDER,
                probe_names,
                probe_rows,
                (condition or {}).get("memory_rung"),
                coverage_floors,
            )
        except ValueError as exc:
            print(f"REFUSED: {name}: {exc}", file=sys.stderr)
            return EXIT_ARM_PROBLEM
        model = (condition or {}).get("model_id") or name
        staged.append(
            (
                model,
                condition_epoch(condition),
                sb.ArmInput(
                    name,
                    condition,
                    {(c.game, c.number): c.cell for c in arm_cells},
                    collapse,
                    collapse_offpath,
                ),
            )
        )
        probe_path = arm.path / PROBE_FILE
        summaries.append(
            sb.ArmSummary(
                name,
                sb.short_data_dir(data_dir),
                sb.file_sha256_short(arm.path / "properties.csv"),
                sb.file_sha256_short(probe_path) if probe_path.exists() else None,
                (
                    None
                    if collapse is None
                    else tuple(
                        game
                        for game, row in sorted(collapse.items())
                        if row["collapsed_with"]
                    )
                ),
                (condition or {}).get("memory_rung") if condition else None,
                (
                    None
                    if collapse_offpath is None
                    else tuple(
                        game
                        for game, row in sorted(collapse_offpath.items())
                        if row["collapsed_with"]
                    )
                ),
            )
        )

    found_models = {model for model, _epoch, _arm in staged}
    missing_models = tuple(sorted(set(roster) - found_models))
    absent_arms = missing_arms(roster, staged)
    if absent_arms and not args.partial:
        print(
            f"the root lacks {len(absent_arms)} of {len(roster) * len(sources.BOARD_LEVELS)} "
            f"roster arms ({'; '.join(absent_arms)}); pass --partial to render a partial board",
            file=sys.stderr,
        )
        return EXIT_PARTIAL
    cells_per_model = (
        len(sources.BOARD_LEVELS)
        * len(sources.SCOREBOARD_GAMES)
        * len(sb.PROPERTY_NUMBERS)
    )
    expected_cells = len(roster) * cells_per_model
    if args.epoch_denominator == EPOCH_SCALE:
        rows_found = {(model, epoch) for model, epoch, _arm in staged}
        expected_cells = (len(rows_found) + len(missing_models)) * cells_per_model

    inputs = group_by_epoch(staged)
    boards = []
    for level in sources.BOARD_LEVELS:
        level_cells = []
        for model in sorted(inputs):
            level_cells += sb.build_level_cells(
                model,
                level,
                inputs[model],
                lambda number, conditions, board_level: sources.headline_arms(
                    number, conditions, board_level, licensed_hands
                ),
                sources.structural_at,
                sources.condition_at,
            )
        boards.append(sb.Board(level, tuple(level_cells)))

    try:
        footnotes = read_footnotes(Path(args.analysis_root))
    except ValueError as exc:
        print(f"analysis root: {exc}", file=sys.stderr)
        return EXIT_INPUT
    block = sb.render_block(
        summaries,
        boards,
        vocabulary.DECISION_RULE,
        vocabulary._DIRECTION_TEXT,
        footnotes,
        expected_cells=(
            None if args.epoch_denominator == EPOCH_PER_EPOCH else expected_cells
        ),
        missing_models=missing_models,
        partial_arms=(
            (len(roster) * len(sources.BOARD_LEVELS), absent_arms)
            if absent_arms
            else None
        ),
        epoch_lines=(
            epoch_header_lines(staged, boards, roster, cells_per_model)
            if args.epoch_denominator == EPOCH_PER_EPOCH
            else ()
        ),
    )
    board_file = Path(args.board_file)
    try:
        old = board_file.read_bytes().decode("utf-8")
    except (OSError, ValueError) as exc:
        print(f"board file: {exc}", file=sys.stderr)
        return EXIT_INPUT
    try:
        new = sb.replace_scoreboard_block(old, block)
    except ValueError as exc:
        print(f"{board_file}: {exc}", file=sys.stderr)
        return EXIT_MARKERS
    if args.check:
        return EXIT_OK if new == old else EXIT_STALE
    if new != old:
        _write_preserving(board_file, new)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
