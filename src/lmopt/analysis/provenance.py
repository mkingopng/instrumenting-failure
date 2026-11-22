"""Render-time provenance: keepable or throwaway, computed from the arms rendered.

A figure is citable only if every arm feeding it is keepable, and that is a
computation over the artifacts on disk, never a constant carried with the figure.
A default provenance is used only when a caller passes none at all (tests, demos): every
entrypoint passes the roster's computed verdict to every entry it renders, `threshold_table` included, so a table of thresholds
rendered beside non-keepable arms carries the same not-citable chrome the
figures beside it do.

The acceptance sidecar contract (agreed with the tier1-recycle stamping side)
-------------------------------------------------------------------------
`<arm_dir>/acceptance.json` is a JSON object written by the arm-evidence gate
beside the arm's `run_meta.json`. Its keys:

- `provenance`: "keepable" or "throwaway", the gate's verdict.
- `gate`: the gate that wrote it, `"recycle_arm_evidence"`.
- `exit_code`: the gate run's exit code; the stamp is written only for an
  exit-0 complete arm with a keepable verdict.
- `git_sha`, `timestamp`: which tree ran the gate, and when.
- `criteria`: an arbitrary JSON object recording what the gate checked. The
  criteria precedence (`pilot_gate`: SKIPPED outranks PASSED; `exact_policy`:
  FAILED outranks OK) is the stamper's rule; this reader never re-derives a
  verdict from `criteria`. `criteria.epoch` (present only when
  the stamping invocation used `--epoch`) records which
  `PROMPT_SPEC_HASHES` version the arm was certified against and why; like
  every other `criteria` key it is informational here, never read for the
  keepable/throwaway verdict.

This reader is deliberately tolerant. The verdict reads `provenance` and
`exit_code` only; `gate` is recorded and passed through, never checked
(checking it would fight the tolerant contract, and a gate rename would
silently downgrade every keepable arm). Unknown extra
top-level keys and anything at all under `criteria` are carried through into
`ArmProvenance.detail` untouched, so the stamping side can extend the record
without a lockstep release here. Extra top-level keys are agreed between the
two sides before use.

No silent fallbacks: an absent or unparseable record is
throwaway with a named source ("absent", "unreadable") and a reason in
`detail`. This module never raises on a bad file; a corrupt sidecar on one arm
must not take down a render pass, and it must not silently promote that arm to
citable either.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

#: The closed provenance vocabulary. Single home: every reader validates against this same tuple.
KEEPABLE = "keepable"
THROWAWAY = "throwaway"
PROVENANCE_VALUES = (THROWAWAY, KEEPABLE)

ACCEPTANCE_FILENAME = "acceptance.json"
RUN_META_FILENAME = "run_meta.json"

#: `ArmProvenance.source` vocabulary: which evidence decided the value.
SOURCE_ACCEPTANCE = "acceptance.json"
SOURCE_RUN_META = "run_meta.provenance"
SOURCE_ABSENT = "absent"
SOURCE_UNREADABLE = "unreadable"
SOURCE_VALUES = (
    SOURCE_ACCEPTANCE,
    SOURCE_RUN_META,
    SOURCE_ABSENT,
    SOURCE_UNREADABLE,
)

#: The gate that writes the sidecar (recorded under `gate`).
ACCEPTANCE_GATE = "recycle_arm_evidence"

#: The two keys the verdict reads. Everything else in the file, `gate`
#: included, is record: it is reported in `detail` and never gates.
ACCEPTANCE_VERDICT_KEYS = ("provenance", "exit_code")

#: The full contract, for documentation and for the freeze record. Absence of
#: any key outside `ACCEPTANCE_VERDICT_KEYS` is not a refusal.
ACCEPTANCE_CONTRACT_KEYS = (
    "provenance",
    "gate",
    "exit_code",
    "git_sha",
    "timestamp",
    "criteria",
)

#: The note `figure_provenance` returns when it was handed no arms at all.
NO_FEEDING_ARMS_NOTE = "no feeding arms"

#: Leads the `detail["reason"]` when the arm directory itself is not there.
SOURCE_DIR_MISSING_REASON = "source dir missing"


@dataclasses.dataclass(frozen=True)
class ArmProvenance:
    """One arm's provenance verdict, the evidence that decided it, and why.

    Attributes:
        value: "keepable" or "throwaway".
        source: which evidence decided it (`SOURCE_VALUES`).
        detail: the record. Always carries `path` and `reason`; an
            `acceptance.json` read also carries the whole parsed sidecar under
            `acceptance`, so unknown keys survive for the caller to log.
    """

    value: str
    source: str
    detail: dict


def _read_json(path: Path) -> tuple[object | None, str]:
    """`(parsed, error)`; `error` is "" on success and names the failure otherwise.

    Never raises: a corrupt file is a finding, not a crash.
    """
    try:
        return json.loads(path.read_text()), ""
    except Exception as exc:  # any read/parse failure is a finding
        return None, f"{type(exc).__name__}: {exc}"


def _is_exit_zero(value: object) -> bool:
    """True only for the integer 0.

    `False == 0` in Python, so a boolean under `exit_code` would otherwise be
    read as a clean gate run and promote a broken stamp to keepable.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def arm_provenance(arm_dir: str | Path) -> ArmProvenance:
    """
    Read one arm's provenance from its directory.

    Order of authority: `acceptance.json` (the gate's own stamp), then
    `run_meta.json`'s top-level `provenance` key (the run's own claim), then
    nothing. A file that exists but cannot be parsed as a JSON object is
    "unreadable", never a pass-through to the next source: an unreadable stamp
    is exactly the case where guessing is worst.
    :param arm_dir: the directory holding the arm's `acceptance.json` /
        `run_meta.json`.
    :returns: ArmProvenance; never raises.
    """
    arm_dir = Path(arm_dir)

    # A dir that is not there at all is not the same finding as a dir with no
    # stamp in it: a moved or deleted source shard dir would otherwise read
    # exactly like an arm the gate never judged.
    if not arm_dir.is_dir():
        return ArmProvenance(
            THROWAWAY,
            SOURCE_ABSENT,
            {
                "path": str(arm_dir),
                "reason": f"{SOURCE_DIR_MISSING_REASON}: {arm_dir}",
            },
        )

    acceptance_path = arm_dir / ACCEPTANCE_FILENAME
    if acceptance_path.exists():
        payload, error = _read_json(acceptance_path)
        if error:
            return ArmProvenance(
                THROWAWAY,
                SOURCE_UNREADABLE,
                {"path": str(acceptance_path), "reason": error},
            )
        if not isinstance(payload, dict):
            return ArmProvenance(
                THROWAWAY,
                SOURCE_UNREADABLE,
                {
                    "path": str(acceptance_path),
                    "reason": (
                        f"{ACCEPTANCE_FILENAME} parsed as "
                        f"{type(payload).__name__}, expected a JSON object "
                        f"carrying {', '.join(ACCEPTANCE_VERDICT_KEYS)}"
                    ),
                },
            )
        stated = payload.get("provenance")
        exit_code = payload.get("exit_code")
        keepable = stated == KEEPABLE and _is_exit_zero(exit_code)
        reason = (
            f"{ACCEPTANCE_FILENAME} states provenance={stated!r}, "
            f"gate={payload.get('gate')!r}, exit_code={exit_code!r}"
        )
        return ArmProvenance(
            KEEPABLE if keepable else THROWAWAY,
            SOURCE_ACCEPTANCE,
            {
                "path": str(acceptance_path),
                "reason": reason,
                # The whole sidecar, verbatim: unknown top-level keys and the
                # `criteria` object pass through for the caller to record.
                "acceptance": payload,
            },
        )

    run_meta_path = arm_dir / RUN_META_FILENAME
    if run_meta_path.exists():
        payload, error = _read_json(run_meta_path)
        if error:
            return ArmProvenance(
                THROWAWAY,
                SOURCE_UNREADABLE,
                {"path": str(run_meta_path), "reason": error},
            )
        if not isinstance(payload, dict):
            # Symmetric with the sidecar branch above: a
            # `run_meta.json` holding valid JSON that is not an object is
            # UNREADABLE, not "absent". Both values are throwaway, so nothing
            # about citability moves; what moves is what the note tells the
            # person fixing it -- "absent" sends them to look for a file that
            # is right there.
            return ArmProvenance(
                THROWAWAY,
                SOURCE_UNREADABLE,
                {
                    "path": str(run_meta_path),
                    "reason": (
                        f"{RUN_META_FILENAME} parsed as "
                        f"{type(payload).__name__}, expected a JSON object"
                    ),
                },
            )
        if "provenance" in payload:
            stated = payload["provenance"]
            return ArmProvenance(
                KEEPABLE if stated == KEEPABLE else THROWAWAY,
                SOURCE_RUN_META,
                {
                    "path": str(run_meta_path),
                    "reason": (
                        f"{RUN_META_FILENAME} states provenance={stated!r} and "
                        f"no {ACCEPTANCE_FILENAME} is present"
                    ),
                    "provenance": stated,
                },
            )

    return ArmProvenance(
        THROWAWAY,
        SOURCE_ABSENT,
        {
            "path": str(arm_dir),
            "reason": (
                f"no {ACCEPTANCE_FILENAME} and no top-level 'provenance' key in "
                f"{RUN_META_FILENAME}; an arm with no keepable evidence is "
                "throwaway"
            ),
        },
    )


def combine_provenance(records: Mapping[str, ArmProvenance]) -> tuple[str, str]:
    """
    The figure rule over already-read per-arm records: all keepable, or none.

    A figure mixes its arms into one picture, so it is citable only if every
    arm feeding it is citable. This is deliberately stricter than the per-arm
    label `cross_arm_matrix.csv` carries: the CSV keeps
    each arm's own verdict, and a reader ANDs that column to recover exactly
    the value this function returns.
    :param records: arm label -> its `ArmProvenance`. The label is what the note
        names, so callers with real arm names should pass those.
    :returns: `(value, note)`. The note is "" when keepable, and otherwise names the
        counts and every non-keepable arm with the source that decided it, so
        the not-citable chrome can say which arm spoiled the figure.
    """
    if not records:
        return THROWAWAY, NO_FEEDING_ARMS_NOTE

    offenders = [
        (label, rec) for label, rec in records.items() if rec.value != KEEPABLE
    ]
    if not offenders:
        return KEEPABLE, ""

    named = ", ".join(f"{label} ({rec.source})" for label, rec in offenders)
    note = (
        f"{len(offenders)} of {len(records)} feeding arm(s) are not keepable: "
        f"{named}."
    )
    return THROWAWAY, note


def _labels(paths: list[Path]) -> list[str]:
    """Distinct labels for `paths`: basenames when they are unique, else paths.

    Two arms can share a basename (`.../qwen/M0`, `.../mistral/M0`), and a note
    naming "M0 (absent)" twice tells a reader nothing about which arm to look
    at. When any basename repeats, every label becomes the full path, so one
    note never mixes the two kinds of name.
    """
    names = [path.name for path in paths]
    if len(set(names)) != len(names):
        names = [str(path) for path in paths]
    if len(set(names)) != len(names):
        # The same directory was fed in twice. Keep the labels distinct so the
        # note's counts still match the number of arms the caller passed.
        names = [f"{name} [#{i}]" for i, name in enumerate(names)]
    return names


def figure_provenance(arm_dirs: Iterable[str | Path]) -> tuple[str, str]:
    """The provenance of a figure fed by `arm_dirs`, and the note explaining it.

    Keepable iff there is at least one feeding arm and every one of them is
    keepable. Zero arms is throwaway: a figure drawn from nothing measured
    nothing. Callers that already know each arm's name should read
    `arm_provenance` per arm and combine with `combine_provenance`, which
    names the arms rather than their directories.
    """
    paths = [Path(arm_dir) for arm_dir in arm_dirs]
    return combine_provenance(
        dict(zip(_labels(paths), (arm_provenance(path) for path in paths), strict=True))
    )
