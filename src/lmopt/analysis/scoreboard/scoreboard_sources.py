"""
The one declaration of the condition that feeds each property's scoreboard cell.

An arm directory is one condition of one model. There is one board per memory level, and
every cell on a board reads that level's run of its model: which file it comes from, with
how many hands, and (for the probe properties) which probe arm is paired with the base arm.
Everything that needs to know reads `PROPERTY_SOURCES`; nothing restates it, and the level
is a parameter of these functions rather than a second mapping.

A property reads the same run on every board it is defined at, so a level a model has no arm
for reads awaiting data naming the run it waits for, and never borrows another level's value.
"""

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

SOURCE_PROPERTIES = "properties.csv"
SOURCE_PROBE = "probe_scores.csv"
ROLE_RELABEL = "relabel"
ROLE_PAYOFF = "payoff_scale"
ROLE_DRIVEN = "offpath_drive"
M0, M1, M2 = "M0", "M1", "M2"
#: One board per memory level, in the order the block prints them. Every board is the full
#: model x game x property grid, so a level with no arm collected still states what it waits
#: for rather than disappearing from the tally.
BOARD_LEVELS = (M0, M1, M2)
#: The opening of the clause a paired probe row carries above M0, which a short cell shows in
#: full: it says the distance includes the divergence of the two arms' memory, so a number is
#: never read as the variant's effect alone.
MEMORY_NOTE_PREFIX = "Memory:"
UNAVAILABLE = "UNAVAILABLE"
#: The five games, in board order. Fixed, never derived from the data, so a game with no
#: rows still gets its cells.
SCOREBOARD_GAMES = ("kuhn", "leduc", "goofspiel", "liarsdice", "darkhex")
#: The words a scored row's `status` can carry in `probe_scores.csv`.
STATUS_MEASURABLE = "MEASURABLE"
STATUS_AWAITING_DATA = "AWAITING_DATA"
STATUS_RECORDED_ABSENCE = "RECORDED_ABSENCE"
STATUS_NOT_COMPUTED = "NOT_COMPUTED"
#: The `probe_type` of a run collected without a probe, as `lmopt.logging.schema` defines
#: it. Only such a run is a headline: a relabel or off-path run is the partner of one, and
#: which partner was used is recorded inside `probe_scores.csv`, not here.
BASE_PROBE_TYPE = "none"


@dataclasses.dataclass(frozen=True)
class PropertySource:
    """
    where one property's scoreboard cell comes from
    :param number: the property number 1 to 8
    :param source: SOURCE_PROPERTIES or SOURCE_PROBE, the file the cell is read from
    :param condition: the run feeding the cell, in words, with `{level}` standing for the
        memory level of the board the cell is on
    :param levels: the memory levels the property is defined at; empty constrains nothing
    :param probe_role: the probe arm paired with the base arm, or None
    :param pairable_levels: the memory levels at which the project can produce this
        property's pairing today; a board outside them has no paired run to read, whatever
        base arms exist. Empty constrains nothing, which is what a property that needs no
        pairing declares
    :param structural_levels: the levels at which the property is inapplicable by design
    :param structural_reason: why the property cannot exist at `structural_levels`
    :param min_licensed_bins: how many licensed hand bins the property's estimand needs, so
        the hand floor follows the P2 licence rather than a number typed here; None when the
        property has no hand requirement
    """

    number: int
    source: str
    condition: str
    levels: tuple[str, ...]
    probe_role: str | None = None
    pairable_levels: tuple[str, ...] = ()
    structural_levels: tuple[str, ...] = ()
    structural_reason: str = ""
    min_licensed_bins: int | None = None


#: Why property 2 cannot exist at memory level M0, in the words the cell shows.
P2_M0_REASON = (
    "nothing carries between hands at M0, so there is no adaptivity to measure"
)
#: How many licensed hand bins property 2's delta needs: a first bin and a last bin. The
#: hands that implies come from the licence artefact, never from a number typed here.
P2_LICENSED_BINS = 2

PROPERTY_SOURCES: dict[int, PropertySource] = {
    1: PropertySource(
        1,
        SOURCE_PROBE,
        "{level} run paired with its relabel run",
        BOARD_LEVELS,
        ROLE_RELABEL,
        pairable_levels=BOARD_LEVELS,
    ),
    2: PropertySource(
        2,
        SOURCE_PROPERTIES,
        "{level} run with enough hands for two licensed bins",
        (M1, M2),
        structural_levels=(M0,),
        structural_reason=P2_M0_REASON,
        min_licensed_bins=P2_LICENSED_BINS,
    ),
    3: PropertySource(3, SOURCE_PROPERTIES, "{level} run", BOARD_LEVELS),
    4: PropertySource(
        4,
        SOURCE_PROBE,
        "{level} run paired with its payoff-scale run",
        BOARD_LEVELS,
        ROLE_PAYOFF,
        pairable_levels=BOARD_LEVELS,
    ),
    5: PropertySource(5, SOURCE_PROPERTIES, "{level} run", BOARD_LEVELS),
    6: PropertySource(
        6,
        SOURCE_PROBE,
        "{level} run paired with its off-path run",
        BOARD_LEVELS,
        ROLE_DRIVEN,
        pairable_levels=BOARD_LEVELS,
    ),
    7: PropertySource(7, SOURCE_PROPERTIES, "{level} run", BOARD_LEVELS),
    8: PropertySource(8, SOURCE_PROPERTIES, "{level} run", BOARD_LEVELS),
}


def condition_at(number: int, level: str) -> str:
    """
    the run one property's cell reads on one board, in words
    :param number: the property number 1 to 8
    :param level: the memory level of the board
    :returns: the declared condition with the board's level filled in, which is what an
        awaiting cell names and what an ambiguous cell reports
    """
    return PROPERTY_SOURCES[number].condition.format(level=level)


def headline_arms(
    number: int,
    conditions: Mapping[str, Mapping],
    level: str,
    licensed_hands=None,
) -> list:
    """
    the arms of one model whose condition is the one a property reads on one board
    :param number: the property number 1 to 8
    :param conditions: {arm name: that arm's recorded condition block}
    :param level: the memory level of the board, which every candidate arm must record
    :param licensed_hands: the hands one licensed bin covers, or None when the licence
        artefact could not be read; a hand requirement no licence backs is not enforced
    :returns: the matching arm names, sorted; more than one is an ambiguity the caller must
        refuse rather than resolve, and none means the board's level has nothing to read
    """
    entry = PROPERTY_SOURCES[number]
    if not memory_level_ok(entry.levels, level):
        return []
    if not memory_level_ok(entry.pairable_levels, level):
        return []
    floor = None
    if entry.min_licensed_bins is not None and licensed_hands:
        floor = entry.min_licensed_bins * licensed_hands
    matching = []
    for name, condition in sorted(conditions.items()):
        if condition.get("probe_type", BASE_PROBE_TYPE) != BASE_PROBE_TYPE:
            continue
        if not memory_level_ok((level,), condition.get("memory_rung")):
            continue
        hands = condition.get("n_hands")
        if floor is not None and not (
            isinstance(hands, (int, float)) and hands >= floor
        ):
            continue
        matching.append(name)
    return matching


def structural_at(number: int, rung: str | None) -> str:
    """
    whether a property cannot exist at one memory level
    :param number: the property number 1 to 8
    :param rung: the memory level, None counting as M0
    :returns: the stated reason when the property is inapplicable by design at that level,
        and an empty string otherwise
    """
    entry = PROPERTY_SOURCES[number]
    if not entry.structural_levels:
        return ""
    present = M0 if rung is None else rung
    return entry.structural_reason if present in entry.structural_levels else ""


def probe_roles(headline_rows: Mapping[int, Sequence[str]]) -> dict[str, str]:
    """
    the probe role of each probe-sourced headline row
    :param headline_rows: {property number: [headline row names]} from the project's decision rule
    :returns: {headline row name: probe role} for every probe-sourced property
    """
    return {
        name: entry.probe_role
        for number, entry in PROPERTY_SOURCES.items()
        if entry.probe_role
        for name in headline_rows[number]
    }


def check_probe_sources(
    property_status: Mapping[str, Sequence],
    headline_rows: Mapping[int, Sequence[str]],
) -> None:
    """
    assert the probe-sourced rows are exactly the headline rows the scorer marks UNAVAILABLE
    :param property_status: the scorer's {row name: (status, caveat)} table
    :param headline_rows: {property number: [headline row names]} from the decision rule
    :raises ValueError: naming every row in one set and not the other, so a change to the
        scorer's statuses cannot silently send a property back to an unscored state
    """
    probe = set(probe_roles(headline_rows))
    unavailable = {
        name
        for names in headline_rows.values()
        for name in names
        if property_status.get(name, ("",))[0] == UNAVAILABLE
    }
    if probe != unavailable:
        raise ValueError(
            "probe-sourced rows and UNAVAILABLE headline rows disagree: only probe-sourced "
            f"{sorted(probe - unavailable)}, only UNAVAILABLE {sorted(unavailable - probe)}"
        )


def memory_level_ok(levels: Sequence[str], rung: str | None) -> bool:
    """
    :param levels: the memory levels a condition allows; empty constrains nothing
    :param rung: an arm's recorded `memory_rung`, where None counts as M0
    :returns: True when the rung satisfies the condition
    """
    if not levels:
        return True
    return (M0 if rung is None else rung) in levels


# --- The committed manifest of a board's input files ---------------------------------------
# A committed record of the files a scoreboard block was rendered from.
#
# The scoreboard reads a root of scored arm directories that is not in git, so a committed block
# cannot say which bytes produced it. The block itself embeds a short hash of two files per arm; this
# module pins every file the board reads, so a copy of the root can be checked against the repository.
#
# A manifest names each file by its path under the root and a full sha256 taken over the bytes with
# CRLF read as LF, the same normalisation as the block's short hash, so the short hash is a prefix of
# the manifest's. It detects change; it does not prevent it.

# The files of an arm directory the scoreboard reads. Any other file in a root (a rendered record,
# a note) is not an input and is not listed.
INPUT_NAMES = (
    "collapse.csv",
    "decision.csv",
    "outcomes.csv",
    "probe_scores.csv",
    "properties.csv",
    "run_meta.json",
)

# The root-level file whose non-empty lines the board appends under its tables.
FOOTNOTES_FILE = "FOOTNOTES.md"

# How a digest treats line endings, recorded on the manifest so it reads without this module.
NORMALISATION = "CRLF read as LF"


def file_digest(path: str | Path) -> str:
    """
    the content hash a manifest and a block's short hash both use
    :param path: a file
    :returns: the full sha256 hex digest of its bytes with CRLF read as LF (str)
    """
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def scan(root: str | Path) -> dict[str, str]:
    """
    every input file of a root, by path under it
    :param root: a root of scored arm directories
    :returns: {"<arm>/<name>": digest} for each input file of each immediate subdirectory, plus
        {FOOTNOTES_FILE: digest} when the root holds one, sorted by key (dict[str, str])
    :raises ValueError: when the root is not a directory
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")
    found: dict[str, str] = {}
    for arm in sorted(p for p in root.iterdir() if p.is_dir()):
        for name in INPUT_NAMES:
            path = arm / name
            if path.is_file():
                found[f"{arm.name}/{name}"] = file_digest(path)
    footnotes = root / FOOTNOTES_FILE
    if footnotes.is_file():
        found[FOOTNOTES_FILE] = file_digest(footnotes)
    return found


def build(root: str | Path) -> dict:
    """
    the manifest of a root
    :param root: a root of scored arm directories
    :returns: {"root": its directory name, "normalisation": ..., "files": scan(root), "absent":
        [FOOTNOTES_FILE] when the root holds none} (dict)
    """
    root = Path(root)
    files = scan(root)
    return {
        "root": root.name,
        "normalisation": NORMALISATION,
        "files": files,
        "absent": [] if FOOTNOTES_FILE in files else [FOOTNOTES_FILE],
    }


def dumps(manifest: Mapping) -> str:
    """
    :param manifest: a manifest from `build`
    :returns: its committed text: sorted keys, two-space indent, one trailing newline (str)
    """
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def load_manifest(path: str | Path) -> dict:
    """
    :param path: a committed manifest
    :returns: the parsed manifest (dict)
    :raises ValueError: when the file is unreadable, is not JSON, or lacks the `files` mapping
    """
    try:
        manifest = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"manifest {path} unreadable ({exc})") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"manifest {path} has no `files` mapping")
    return manifest


def verify(root: str | Path, manifest: Mapping) -> list[str]:
    """
    what differs between a root and a manifest
    :param root: a root of scored arm directories
    :param manifest: a manifest from `build` or `load`
    :returns: one sentence per file that is missing from the root, present in it but not in the
        manifest (a footnotes file the manifest records as absent is one of these), or changed;
        empty when the root is exactly what the manifest records (list[str])
    :raises ValueError: when the root is not a directory
    """
    expected = manifest["files"]
    actual = scan(root)
    problems = []
    for key in sorted(expected.keys() - actual.keys()):
        problems.append(f"missing: {key}")
    for key in sorted(actual.keys() - expected.keys()):
        problems.append(f"extra: {key} is in the root and not in the manifest")
    for key in sorted(expected.keys() & actual.keys()):
        if expected[key] != actual[key]:
            problems.append(
                f"changed: {key} (manifest {expected[key][:12]}, root {actual[key][:12]})"
            )
    return problems
