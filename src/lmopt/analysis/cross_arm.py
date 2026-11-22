"""Cross-arm artifact discovery and loading for multi-arm analysis.

Discovers per-arm analysis artifact directories (properties.csv, decision.csv,
outcomes.csv), never raw Parquet shards.
"""

import dataclasses
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from lmopt.analysis.condition import is_diagnostic

ARTIFACT_NAMES = ("properties.csv", "decision.csv", "outcomes.csv")
OPPONENT_LADDER = ("always_bet", "rl", "cfr_ref", "bayesian")


def _coerce_numeric(value, arm: str, column: str):
    """
    Coerce a value to float | None, raising on non-numeric non-null.
    :param value: Value to coerce (from polars cell)
    :param arm: Arm name for error message
    :param column: Column name for error message
    :returns: float if numeric, None if null
    :raises ValueError: If value is non-null and non-numeric, a bool, or non-finite
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"Arm {arm!r}, column {column!r}: expected numeric or null, got bool {value!r}; "
            "a boolean cell is not a measurement, so re-export the arm's CSV"
        )
    if isinstance(value, (float, int)):
        if not math.isfinite(value):
            raise ValueError(
                f"Arm {arm!r}, column {column!r}: expected a finite number or null, got "
                f"{value!r}; the exporter writes an unmeasured value as null, so re-export "
                "the arm's CSV"
            )
        return float(value)
    # Non-numeric non-null: raise
    raise ValueError(
        f"Arm {arm!r}, column {column!r}: expected numeric or null, got {type(value).__name__} {value!r}"
    )


@dataclasses.dataclass
class ArmArtifacts:
    """Per-arm analysis artifacts loaded from CSV files.

    Attributes:
        name: Directory name of the arm
        path: Full path to the arm directory
        properties: DataFrame from properties.csv
        decision: DataFrame from decision.csv
        outcomes: DataFrame from outcomes.csv
        per_infoset: Optional DataFrame from per_infoset.csv (None if absent)
    """

    name: str
    path: Path
    properties: pl.DataFrame
    decision: pl.DataFrame
    outcomes: pl.DataFrame
    per_infoset: pl.DataFrame | None = None


#: The numeric columns of `properties.csv` and `per_infoset.csv` that are blank on most rows, typed explicitly on every read so
#: they read as numbers whatever an arm's file holds. A column that is blank on every row of an arm (an
#: arm with no exact-policy pass has no `value_exact`) reads as text while the same column on an arm that
#: has values reads as Float64, so two arms' frames could not be stacked strictly, and a relaxed stack would
#: turn the numbers into text. An override is a strict type, never a coercion: a cell that is not a number
#: raises at read time. A column an older arm lacks is simply absent. Append a column here when a writer
#: adds a sparse numeric one to `properties.csv` or `per_infoset.csv`. `n_fallback_rows` is blank on every
#: row of a `per_infoset.csv` written from a frame with no `fallback_used` column, and a count otherwise.
SPARSE_NUMERIC_COLUMN_SCHEMA = {
    "value_exact": pl.Float64,
    "ci_low_exact": pl.Float64,
    "ci_high_exact": pl.Float64,
    "on_support_fraction": pl.Float64,
    "n_unmatched_rows": pl.Int64,
    "n_fallback_rows": pl.Int64,
}


def _read_artifact(path: Path) -> pl.DataFrame:
    """
    read one arm artifact CSV, typing every column from the whole file
    Polars infers a schema from the first 100 rows by default, so a column whose first 100
    cells are empty would be typed as text and every later number would arrive as a string.
    :param path: the CSV file (Path)
    :returns: the loaded frame (pl.DataFrame)
    """
    return pl.read_csv(
        path, infer_schema_length=None, schema_overrides=SPARSE_NUMERIC_COLUMN_SCHEMA
    )


def discover_arm_artifacts(
    root: str | Path,
) -> tuple[dict[str, ArmArtifacts], list[str]]:
    """
    Discover and load analysis artifacts from arm directories.

    Scans immediate subdirectories of `root` for the three required CSV artifacts.
    Returns successfully loaded arms keyed by directory name, plus a list of
    problems for arms that were refused.

    Refusal semantics:
    - Subdir with all three artifacts parsing cleanly -> an arm, keyed by dir name
    - Subdir with NONE of the three -> skipped silently (root may hold logs)
    - Subdir with some but not all, or any parse failure -> refused loudly (problem
      string names the arm dir, file, and reason; arm not returned)
    - Subdir whose run_meta.json condition block carries the diagnostic marker -> refused
      loudly, whatever its artifacts (a diagnostic arm is never a scoreboard arm)
    - Non-dirs ignored
    - Scan is sorted(os.listdir(root)) on the one named root, no recursive glob
    :param root: Directory to scan for arm subdirectories
    :returns: (arms_dict, problems_list) where arms_dict maps directory name to ArmArtifacts
        and problems_list contains refusal messages for rejected arms
    :raises ValueError: If root is not a directory
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    arms: dict[str, ArmArtifacts] = {}
    problems: list[str] = []

    for entry in sorted(os.listdir(root)):
        entry_path = root / entry
        if not entry_path.is_dir():
            continue

        # Check which artifacts are present (build from ARTIFACT_NAMES constant)
        artifact_paths = {name: entry_path / name for name in ARTIFACT_NAMES}
        present = {name: path.exists() for name, path in artifact_paths.items()}
        n_present = sum(present.values())

        # Silent skip: zero artifacts (legitimate log/metadata directory)
        if n_present == 0:
            continue

        # Loud refusal: a diagnostic arm is never a scoreboard arm. Keyed on the marker's
        # presence in the arm's own condition block; an arm with no run_meta.json, or one
        # that does not parse, is not refused here and reads as before.
        try:
            meta = json.loads((entry_path / "run_meta.json").read_text())
        except (OSError, ValueError):
            meta = None
        if is_diagnostic(meta):
            problems.append(
                f"{entry}: marked diagnostic in its run_meta condition block, so it is "
                "never a scoreboard arm"
            )
            continue

        # Loud refusal: partial artifacts
        if n_present < 3:
            missing = [name for name, exists in present.items() if not exists]
            problems.append(
                f"{entry}: partial artifacts (missing {', '.join(missing)})"
            )
            continue

        # All three present, attempt to load each file
        try:
            properties = _read_artifact(artifact_paths["properties.csv"])
        except Exception as e:
            problems.append(
                f"{entry}: failed to parse properties.csv ({type(e).__name__}: {e})"
            )
            continue

        try:
            decision = _read_artifact(artifact_paths["decision.csv"])
        except Exception as e:
            problems.append(
                f"{entry}: failed to parse decision.csv ({type(e).__name__}: {e})"
            )
            continue

        try:
            outcomes = _read_artifact(artifact_paths["outcomes.csv"])
        except Exception as e:
            problems.append(
                f"{entry}: failed to parse outcomes.csv ({type(e).__name__}: {e})"
            )
            continue

        # Check for optional per_infoset.csv
        per_infoset_path = entry_path / "per_infoset.csv"
        per_infoset = None
        if per_infoset_path.exists():
            try:
                per_infoset = _read_artifact(per_infoset_path)
            except Exception as e:
                problems.append(
                    f"{entry}: failed to parse per_infoset.csv ({type(e).__name__}: {e})"
                )
                continue

        # Successfully loaded
        arms[entry] = ArmArtifacts(
            name=entry,
            path=entry_path,
            properties=properties,
            decision=decision,
            outcomes=outcomes,
            per_infoset=per_infoset,
        )

    return arms, problems


def matrix_rows(
    arms: dict[str, ArmArtifacts],
    provenance: str | Mapping[str, str],
    collapsed_with: Mapping[str, str] | None = None,
) -> list[dict]:
    """
    Build cross-arm matrix rows (arm × property).

    One row per (arm, decision-rule property), sourced from each arm's decision.csv.
    Sorted by (arm, property) for stable rendering.

    The `collapsed_with` column exists because two different models can
    produce byte-identical play, and a cross-model table then reads as two
    models independently agreeing when the truth is one policy observed twice.
    On the v4 keepable arms, OLMo and Mistral are identical on Leduc, and
    Mistral and Qwen are identical on Goofspiel. Any statistic pooling such a
    pair has an effective n of one arm.

    It follows the same reasoning as the per-arm provenance column below,
    which is why it is a column and not a caption: a per-row false statement
    in a machine-readable artifact is
    the silent-fallback class. A row that silently omits a collapse makes that
    false statement to every consumer that does not read the caption.

    The value is game-scoped while the row is property-scoped, because a
    decision row aggregates over games. So it names which games contribute
    non-independent cells and with whom, e.g.
    ``"leduc:mistral-nemo-instruct-2407"``, semicolon-separated for several.

    The `provenance` column is PER ARM. A mixed roster
    holds one arm the acceptance gate passed and one it did not, and labelling
    the keepable arm's rows "throwaway" because a NEIGHBOUR was throwaway
    destroys the only per-arm record of the gate's verdict. A figure mixes its
    arms into one picture and is therefore citable only if every feeding arm
    is: a reader recovers that figure-level value by ANDing this column across
    the arms a figure drew from, which is exactly what
    `lmopt.analysis.provenance.combine_provenance` computes.
    :param arms: Loaded arm artifacts, keyed by arm name
    :param provenance: either one label ("throwaway" or "keepable") applied to
        every row, or a mapping from arm name to that arm's own label. A
        mapping missing an arm raises: a row whose provenance was guessed
        is worse than a run that stops.
    :param collapsed_with: arm name -> its game-scoped collapse label, or "" for
        an arm checked and found distinct. Computed by the caller, which is
        where Parquet is reachable; this module reads analysis artifacts
        only, by deliberate design, and re-reading shards here would invert
        that boundary. Pass None when no check was run: every row then reads
        ``"not checked"`` rather than ``""``, because an empty string reads
        as "verified distinct" and asserting that without checking is the
        same false statement in the other direction.
    :returns: List of dicts with keys: arm, property, verdict, observed_mean_admissible,
        blocker, provenance, collapsed_with, dropped_rate. observed_mean_admissible
        and dropped_rate are float | None; dropped_rate is None for an arm whose
        decision.csv predates the column.
    :raises KeyError: `provenance` is a mapping and an arm is absent from it, or
        `collapsed_with` is supplied and an arm is absent from it.
    """
    rows = []

    for arm_name, artifacts in arms.items():
        if isinstance(provenance, str):
            arm_provenance_label = provenance
        elif arm_name in provenance:
            arm_provenance_label = provenance[arm_name]
        else:
            raise KeyError(
                f"provenance mapping has no entry for arm {arm_name!r} "
                f"(it names {sorted(provenance)}); every rendered arm must "
                "carry its own verdict, and guessing one would be a silent "
                "fallback"
            )
        if collapsed_with is None:
            arm_collapse_label = "not checked"
        elif arm_name in collapsed_with:
            arm_collapse_label = collapsed_with[arm_name]
        else:
            raise KeyError(
                f"collapsed_with mapping has no entry for arm {arm_name!r} "
                f"(it names {sorted(collapsed_with)}); an absent entry is "
                "ambiguous between 'checked and distinct' and 'never checked', "
                "and defaulting it either way states something the check did "
                "not establish"
            )
        # Read decision.csv columns: property, verdict, observed_mean_admissible, blocker
        for row in artifacts.decision.iter_rows(named=True):
            # Extract observed_mean_admissible as float | None (raise on non-numeric non-null)
            value = _coerce_numeric(
                row["observed_mean_admissible"], arm_name, "observed_mean_admissible"
            )

            rows.append(
                {
                    "arm": arm_name,
                    "property": row["property"],
                    "verdict": row["verdict"],
                    "observed_mean_admissible": value,
                    "blocker": row["blocker"],
                    "provenance": arm_provenance_label,
                    "collapsed_with": arm_collapse_label,
                    "dropped_rate": _coerce_numeric(
                        row.get("dropped_rate"), arm_name, "dropped_rate"
                    ),
                }
            )

    # Sort by (arm, property)
    rows.sort(key=lambda r: (r["arm"], r["property"]))

    return rows


def ladder_series(arms: dict[str, ArmArtifacts]) -> dict[str, dict[str, list[dict]]]:
    """
    Build opponent-ladder discrimination curves data.

    Returns nested dict {game: {arm: [point, ...]}} where points are ordered by
    OPPONENT_LADDER and each point contains keys: opponent, mean_reward (float | None),
    reward_ci_lo (float | None), reward_ci_hi (float | None), flag (str),
    dropped_rate (float | None; None when the arm's outcomes.csv has no such column, which is
    unknown and not zero).

    An opponent absent from an arm's outcomes yields no point (absence visible as
    a gap, never interpolated).
    :param arms: Loaded arm artifacts, keyed by arm name
    :returns: Nested dict {game: {arm: [point, ...]}} with points ordered by OPPONENT_LADDER
    """
    series: dict[str, dict[str, list[dict]]] = {}

    for arm_name, artifacts in arms.items():
        # Read outcomes.csv rows
        for row in artifacts.outcomes.iter_rows(named=True):
            game = row["game"]
            opponent = row["opponent"]

            # Initialize game and arm if needed
            if game not in series:
                series[game] = {}
            if arm_name not in series[game]:
                series[game][arm_name] = []

            # Extract fields, preserving None for missing CIs (raise on non-numeric non-null)
            mean_reward = _coerce_numeric(row["mean_reward"], arm_name, "mean_reward")
            reward_ci_lo = _coerce_numeric(
                row["reward_ci_lo"], arm_name, "reward_ci_lo"
            )
            reward_ci_hi = _coerce_numeric(
                row["reward_ci_hi"], arm_name, "reward_ci_hi"
            )

            flag = row["flag"] if row["flag"] else ""
            dropped_rate = _coerce_numeric(
                row.get("dropped_rate"), arm_name, "dropped_rate"
            )

            # Append point
            series[game][arm_name].append(
                {
                    "opponent": opponent,
                    "mean_reward": mean_reward,
                    "reward_ci_lo": reward_ci_lo,
                    "reward_ci_hi": reward_ci_hi,
                    "flag": flag,
                    "dropped_rate": dropped_rate,
                }
            )

    # Sort points by OPPONENT_LADDER order
    ladder_index = {opp: i for i, opp in enumerate(OPPONENT_LADDER)}

    for game in series:
        for arm_name in series[game]:
            series[game][arm_name].sort(
                key=lambda p: ladder_index.get(p["opponent"], 999)
            )

    return series
