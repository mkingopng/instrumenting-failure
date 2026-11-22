#!/usr/bin/env python3
"""Print the collection state of the model ladder, read from disk in this checkout.

Use it to check how much of a collection exists before composing or scoring. It is read-only.

What it reads. The ladder's shape comes from `lmopt.orchestration.roster` alone: every `JOBS` kind
with run indices (`n_indices`), every `MODELS` entry, and each kind's `out_root`, which names
one directory per run. Nothing is globbed by prefix, so a sibling such as
`recycle-v7-m0-partials/` is never read. Per run it reads the runner's own records:
`run_meta.json` for the configured (game, opponent) cells, `shard_outcomes.json` for the cells
that completed, and `lmopt.analysis.provenance.arm_provenance` for acceptance.

Every verdict line names the stage it certifies and states the scoring gap in counts.

Exit codes:

  0  COLLECTION COMPLETE: every expected run is collected; or SCORABLE: that, and every run
     accepted and every composite present.
     With --require-scorable, only SCORABLE exits 0.
  2  usage error.
  3  INCOMPLETE: some expected run is absent or incomplete (or, with --require-scorable, not
     yet scorable), and nothing below applies.
  4  CANNOT TELL: a run's run_meta.json or shard_outcomes.json is unreadable, or a directory
     exists that JOBS does not declare.
  5  only with --history: the ops event log has a malformed line (the file and line are named).

A WARNING line, printed just above the verdict only when it applies, counts writable files and
directories inside accepted runs and composites in this checkout. It never changes the verdict
or the exit code.

Usage:
  .venv/bin/python scripts/ladder_state.py                  # experiment 7's kinds
  .venv/bin/python scripts/ladder_state.py --experiment 8   # experiment 8's kinds
  .venv/bin/python scripts/ladder_state.py --kind m0 --kind relabel
  .venv/bin/python scripts/ladder_state.py --history --kind m1 --model qwen --index 3
                                                            # the ops events of one run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lmopt.analysis.provenance import KEEPABLE, arm_provenance
from lmopt.logging import certified
from lmopt.logging.call_log import orphan_call_logs
from lmopt.logging.ops_events import (
    events_for_runs,
    experiment_root,
    format_event,
    ops_events_path,
    read_events,
)
from lmopt.orchestration.roster import JOBS, MODELS

EXIT_COMPLETE = 0
EXIT_USAGE = 2
EXIT_INCOMPLETE = 3
EXIT_CANNOT_TELL = 4
EXIT_HISTORY_UNREADABLE = 5

ABSENT = "ABSENT"
UNREADABLE = "UNREADABLE"
INCOMPLETE = "INCOMPLETE"
COLLECTED = "COLLECTED"

DEFAULT_EXPERIMENT = 7

@dataclass(frozen=True)
class RunKey:
    """
    one expected run: a kind, a model, the game for per-game kinds, and a run index
    :param kind: JOBS key (str)
    :param model: MODELS key (str)
    :param game: game named by the kind's out_root, else None (str or None)
    :param index: run index (int)
    :param rel_dir: the run directory relative to the repo root (str)
    """

    kind: str
    model: str
    game: str | None
    index: int
    rel_dir: str


@dataclass(frozen=True)
class RunStatus:
    """
    what one run directory holds
    :param status: ABSENT, UNREADABLE, INCOMPLETE or COLLECTED (str)
    :param missing: configured (game, opponent) cells not collected (frozenset)
    :param accepted: arm_provenance's value, "keepable" or "throwaway" (str)
    :param reason: why the status is not COLLECTED, empty when it is (str)
    """

    status: str
    missing: frozenset = frozenset()
    accepted: str = "throwaway"
    reason: str = ""


@dataclass
class HostReading:
    """
    one host's reading of the ladder
    :param name: host label (str)
    :param root: where it was read (str)
    :param read_at: UTC time of the read (str)
    :param statuses: RunKey -> RunStatus (dict)
    :param unexpected: directories JOBS does not declare, relative to the repo root (list)
    :param composites: (kind root, model) -> composed piece count, or None when absent (dict)
    :param undeclared_composites: composites on disk at no path JOBS declares (list of Path)
    """

    name: str
    root: str
    read_at: str
    statuses: dict = field(default_factory=dict)
    unexpected: list = field(default_factory=list)
    composites: dict = field(default_factory=dict)
    undeclared_composites: list = field(default_factory=list)


def indexed_kinds() -> list[str]:
    """
    every JOBS kind that declares run indices, in JOBS order
    :returns: kind keys (list of str)
    """
    return [k for k, spec in JOBS.items() if spec.n_indices is not None]


def default_kinds(experiment: int = DEFAULT_EXPERIMENT) -> list[str]:
    """
    the kinds read when no --kind is given: one experiment's indexed kinds, in JOBS order,
    without the diagnostic kinds, which are never scoreboard arms and would otherwise count as
    missing from the ladder. `indexed_kinds()` and `expected_runs()` are not filtered, so a kind
    of either experiment, a diagnostic one included, named on the command line works, and
    compose matches a piece against every kind's roots
    :param experiment: 7 or 8 (int)
    :returns: kind keys (list of str)
    """
    return [
        k
        for k in indexed_kinds()
        if JOBS[k].experiment == experiment and not JOBS[k].diagnostic
    ]


def kind_roots(kind: str) -> list[tuple[str | None, str]]:
    """
    the directory that holds one kind's run-index directories, once per game when the kind's
    out_root names a game
    :param kind: JOBS key (str)
    :returns: (game or None, root relative to the repo root) pairs (list of tuple)
    """
    spec = JOBS[kind]
    parent = spec.out_root.rsplit("/", 1)[0]
    if "{game}" in spec.out_root:
        # One tree per game assumes every model plays every game of the kind; a kind that
        # also narrowed games per model would expect runs no job ever writes.
        if any(set(g) != set(spec.games) for g in spec.model_games.values() if g):
            raise ValueError(
                f"kind {kind!r} names a game in out_root and narrows games per "
                "model; ladder_state cannot tell which trees each model expects"
            )
        return [(g, parent.format(game=g)) for g in spec.games]
    return [(None, parent)]


def expected_runs(kinds: list[str]) -> list[RunKey]:
    """
    every run the ladder declares for the given kinds, whether or not anything exists for it
    :param kinds: JOBS keys (list of str)
    :returns: one key per (kind, model, game, index) (list of RunKey)
    """
    keys = []
    for kind in kinds:
        spec = JOBS[kind]
        for model in MODELS:
            if not spec.model_games.get(model):
                continue
            assert spec.n_indices is not None  # indexed_kinds() only
            for game, root in kind_roots(kind):
                for index in range(spec.n_indices):
                    keys.append(
                        RunKey(
                            kind,
                            model,
                            game,
                            index,
                            f"{root}/i{index}/{MODELS[model].short}",
                        )
                    )
    return keys


def _read_json(path: Path):
    """
    parse one JSON record
    :param path: the file (Path)
    :returns: the parsed object, or None when the file is missing or unparseable
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_status(run_dir: Path) -> RunStatus:
    """
    read one run directory: configured cells from run_meta.json, collected cells from the
    runner's own shard_outcomes.json (status completed and the recorded file present), never
    from filenames, since probe shards carry a suffix
    :param run_dir: the run directory (Path)
    :returns: the run's status (RunStatus)
    """
    if not run_dir.is_dir():
        return RunStatus(ABSENT, reason="no run directory")
    accepted = arm_provenance(run_dir).value
    meta = _read_json(run_dir / "run_meta.json")
    resolved = meta.get("resolved_config") if isinstance(meta, dict) else None
    games = resolved.get("games") if isinstance(resolved, dict) else None
    opponents = resolved.get("opponents") if isinstance(resolved, dict) else None
    if not games or not opponents:
        return RunStatus(
            UNREADABLE,
            accepted=accepted,
            reason="run_meta.json missing, unparseable, or without games/opponents",
        )
    configured = frozenset((g, o) for g in games for o in opponents)
    outcomes_path = run_dir / "shard_outcomes.json"
    if not outcomes_path.exists():
        return RunStatus(
            INCOMPLETE,
            configured,
            accepted,
            "no shard_outcomes.json: the job did not finish",
        )
    outcomes = _read_json(outcomes_path)
    shards = outcomes.get("shards") if isinstance(outcomes, dict) else None
    if not isinstance(shards, list):
        return RunStatus(
            UNREADABLE,
            accepted=accepted,
            reason="shard_outcomes.json unparseable or without shards",
        )
    collected = {
        (s.get("game"), s.get("opponent"))
        for s in shards
        if isinstance(s, dict)
        and s.get("status") == "completed"
        and (run_dir / Path(str(s.get("out_path", ""))).name).is_file()
    }
    missing = configured - collected
    if missing:
        reason = f"{len(missing)} of {len(configured)} configured cells not collected"
        orphans = orphan_call_logs(str(run_dir))
        if orphans:
            reason += f"; call log without a shard (a killed shard's calls): {', '.join(orphans)}"
        return RunStatus(INCOMPLETE, frozenset(missing), accepted, reason)
    return RunStatus(COLLECTED, accepted=accepted)


def unexpected_dirs(repo: Path, kinds: list[str]) -> list[str]:
    """
    directories under a kind's root that JOBS does not declare for that kind: an index outside
    range(n_indices), a name that is not an index or an expected model's composite, or a model
    directory for a model JOBS gives no game at this kind (such a directory holds real data
    that no expected run would ever read)
    :param repo: repo root (Path)
    :param kinds: JOBS keys (list of str)
    :returns: offending directories relative to the repo root (list of str)
    """
    found = []
    for kind in kinds:
        spec = JOBS[kind]
        shorts = {MODELS[m].short for m in MODELS if spec.model_games.get(m)}
        composites = {f"composite-{s}" for s in shorts}
        assert spec.n_indices is not None  # indexed_kinds() only
        declared = {f"i{i}" for i in range(spec.n_indices)}
        for _, root in kind_roots(kind):
            base = repo / root
            if not base.is_dir():
                continue
            for child in sorted(p for p in base.iterdir() if p.is_dir()):
                if child.name in composites:
                    continue
                if child.name not in declared:
                    found.append(f"{root}/{child.name}")
                    continue
                found.extend(
                    f"{root}/{child.name}/{m.name}"
                    for m in sorted(child.iterdir())
                    if m.is_dir() and m.name not in shorts
                )
    return found


#: How far below a kind's top-level tree a composite is looked for. A declared run is three levels
#: down (`<top>/<kind dir>/i0/<model>`), and the cross-game off-path composites are three
#: (`<top>/composite-offpath/seeds42-46/<model>`).
UNDECLARED_COMPOSITE_DEPTH = 4


def undeclared_composites(repo: Path, kinds: list[str]) -> list[Path]:
    """
    composites on disk at no path JOBS declares, found by what they record (a `composed_from`
    list in run_meta.json) and not by the directory name the declared lookup expects. The declared
    lookup (`composite-<model short>` under each kind root) cannot see a composite kept elsewhere,
    such as the cross-game off-path composites, so a lock scan or a count built on it silently
    leaves them out. The lock scan and the listing read them, and so does the composite count of
    a kind that keeps one run tree per game (cross_game_composite_pieces)
    :param repo: repo root (Path)
    :param kinds: JOBS keys (list of str)
    :returns: the composite directories, sorted (list of Path)
    """
    declared = {
        repo / root / f"composite-{spec.short}"
        for kind in kinds
        for _, root in kind_roots(kind)
        for spec in MODELS.values()
    }
    tops = sorted(
        {
            repo / Path(*Path(root).parts[:3])
            for kind in kinds
            for _, root in kind_roots(kind)
        }
    )
    found = []
    for top in tops:
        if not top.is_dir():
            continue
        for dirpath, dirs, files in os.walk(top, followlinks=False):
            here = Path(dirpath)
            if (
                here in declared
                or len(here.relative_to(top).parts) > UNDECLARED_COMPOSITE_DEPTH
            ):
                dirs[:] = []
                continue
            if "run_meta.json" in files:
                meta = _read_json(here / "run_meta.json")
                if isinstance(meta, dict) and isinstance(
                    meta.get("composed_from"), list
                ):
                    found.append(here)
                dirs[:] = []
    return sorted(found)


def read_host(repo: Path, kinds: list[str], name: str = "local") -> HostReading:
    """
    read every expected run, unexpected directory and composite under one repo root
    :param repo: repo root (Path)
    :param kinds: JOBS keys (list of str)
    :param name: host label (str)
    :returns: the reading (HostReading)
    """
    reading = HostReading(name, str(repo), _now())
    for key in expected_runs(kinds):
        reading.statuses[key] = run_status(repo / key.rel_dir)
    reading.unexpected = unexpected_dirs(repo, kinds)
    reading.undeclared_composites = undeclared_composites(repo, kinds)
    cross_game = cross_game_composite_pieces(repo, reading.undeclared_composites)
    for kind in kinds:
        per_game = "{game}" in JOBS[kind].out_root
        for _, root in kind_roots(kind):
            for model, spec in MODELS.items():
                if per_game and (root, spec.short) in cross_game:
                    reading.composites[(root, model)] = cross_game[(root, spec.short)]
                    continue
                meta = _read_json(
                    repo / root / f"composite-{spec.short}" / "run_meta.json"
                )
                pieces = meta.get("composed_from") if isinstance(meta, dict) else None
                reading.composites[(root, model)] = (
                    len(pieces) if isinstance(pieces, list) else None
                )
    return reading


def cross_game_composite_pieces(repo: Path, composites: list[Path]) -> dict:
    """
    for a kind that keeps one run tree per game, the composite is one arm across every game
    (`<top>/composite-offpath/seeds42-46/<model>`), at no path a per-game root declares. It is
    found by what it records: each piece in its `composed_from` names `<root>/i<index>/<model>`,
    so the composite covers that root for that model. Only a composite that records the shape
    compose gave it counts (`arm`, then `index`, then `partial`, best first); one without a
    `shape`, composed before compose recorded completeness, is listed but never counted. Two
    composites of the same best shape covering one root and model are refused, since either
    could be the one scored
    :param repo: repo root (Path)
    :param composites: composites at no declared path, from undeclared_composites (list of Path)
    :returns: (root, model short) -> pieces of that root in the chosen composite (dict)
    :raises ValueError: when two composites of the same best shape cover one root and model
    """
    rank = {"arm": 0, "index": 1, "partial": 2}
    best: dict = {}
    for comp in composites:
        meta = _read_json(comp / "run_meta.json")
        if not isinstance(meta, dict) or meta.get("shape") not in rank:
            continue
        pieces = meta.get("composed_from")
        counts: dict = {}
        for piece in pieces if isinstance(pieces, list) else []:
            path = Path(str(piece))
            if path.is_absolute():
                try:
                    path = path.relative_to(repo)
                except ValueError:
                    continue
            parts = path.parts
            if len(parts) < 3 or not re.fullmatch(r"i\d+", parts[-2]):
                continue
            key = (str(Path(*parts[:-2])), parts[-1])
            counts[key] = counts.get(key, 0) + 1
        here = rank[meta["shape"]]
        for key, n in counts.items():
            held = best.get(key)
            if held is not None and held[0] == here:
                raise ValueError(
                    f"two {meta['shape']} composites cover {key[0]} for {key[1]}: "
                    f"{held[1]} and {comp}"
                )
            if held is None or here < held[0]:
                best[key] = (here, comp, n)
    return {key: n for key, (_, _, n) in best.items()}


def _now() -> str:
    """
    the current UTC time, to the second
    :returns: ISO-8601 timestamp (str)
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _level_label(kind: str, game: str | None) -> str:
    """
    the name a status table uses for one kind (and game, for per-game kinds)
    :param kind: JOBS key (str)
    :param game: game or None (str or None)
    :returns: the kind's own label with the game appended for a per-game kind, e.g. "M0",
        "relabel", "offpath-kuhn", and for experiment 8 "v8 M0", "v8 offpath-kuhn" (str)
    """
    label = JOBS[kind].label
    if game is not None:
        return f"{label}-{game}"
    return label


def summary_lines(reading: HostReading, kinds: list[str], host: str) -> list[str]:
    """
    one line per model x kind (x game), every one printed whether or not anything exists, all
    counts in runs, in the exact form a status-table row quotes. Each line names its own host,
    so a line copied out of truncated output still says what it is about.
    :param reading: one host's reading (HostReading)
    :param kinds: JOBS keys (list of str)
    :param host: this reading's host label, e.g. "local" (str)
    :returns: lines such as "[local] mistral M0: 1 of 10 runs present, 1 collected, 0
        accepted, composite absent" (list of str)
    """
    lines = []
    for kind in kinds:
        spec = JOBS[kind]
        for game, root in kind_roots(kind):
            label = _level_label(kind, game)
            for model in MODELS:
                if not spec.model_games.get(model):
                    lines.append(
                        f"[{host}] {model} {label}: not expected, JOBS gives this "
                        "model no game at this level"
                    )
                    continue
                runs = [
                    s
                    for k, s in reading.statuses.items()
                    if k.kind == kind and k.model == model and k.game == game
                ]
                present = sum(s.status != ABSENT for s in runs)
                collected = sum(s.status == COLLECTED for s in runs)
                accepted = sum(s.accepted == KEEPABLE for s in runs)
                pieces = reading.composites.get((root, model))
                composite = (
                    "composite absent"
                    if pieces is None
                    else f"composite of {pieces} pieces"
                )
                lines.append(
                    f"[{host}] {model} {label}: {present} of {spec.n_indices} runs "
                    f"present, {collected} collected, {accepted} accepted, {composite}"
                )
    return lines


def detail_lines(reading: HostReading) -> list[str]:
    """
    every run that is not COLLECTED, with its missing cells by name
    :param reading: one host's reading (HostReading)
    :returns: lines (list of str)
    """
    lines = []
    absent: dict[str, list[str]] = {}
    for key, st in reading.statuses.items():
        name = f"{key.kind}{'-' + key.game if key.game else ''} {key.model}"
        if st.status == ABSENT:
            absent.setdefault(name, []).append(f"i{key.index}")
        elif st.status != COLLECTED:
            cells = ", ".join(f"{g}/{o}" for g, o in sorted(st.missing))
            lines.append(
                f"  {name} i{key.index}: {st.status}, {st.reason}"
                + (f" [{cells}]" if cells else "")
            )
    lines.extend(f"  {name}: absent {' '.join(idx)}" for name, idx in absent.items())
    return lines


VERDICT_WORDS = ("COLLECTION COMPLETE", "SCORABLE", "INCOMPLETE", "CANNOT TELL")


def _verdict_line(word: str, code: int, *clauses: str) -> tuple[int, str]:
    """
    build a verdict line; the only way verdict() forms one, so a word outside VERDICT_WORDS
    cannot reach the report
    :param word: one of VERDICT_WORDS (str)
    :param code: the exit code the word stands for (int)
    :param clauses: the clauses that follow the word, in order (str)
    :returns: (exit code, verdict line without the "VERDICT: " prefix) (tuple)
    """
    if word not in VERDICT_WORDS:
        raise ValueError(f"verdict word {word!r} is not one of {VERDICT_WORDS}")
    return code, "; ".join((f"{word} (exit {code})", *clauses))


def scoring_gap(reading: HostReading, kinds: list[str]) -> tuple[bool, str]:
    """
    what stands between collection and scoring, in counts, always stated: accepted runs out of
    every expected run, and composites present out of every model x kind (x game) that expects
    runs
    :param reading: the judged host's reading (HostReading)
    :param kinds: JOBS keys (list of str)
    :returns: (whether nothing stands in the way, the clause) (tuple)
    """
    runs = len(reading.statuses)
    accepted = sum(s.accepted == KEEPABLE for s in reading.statuses.values())
    expected = [
        (root, model)
        for kind in kinds
        for _, root in kind_roots(kind)
        for model in MODELS
        if JOBS[kind].model_games.get(model)
    ]
    present = sum(reading.composites.get(k) is not None for k in expected)
    scorable = accepted == runs and present == len(expected)
    prefix = "scorable" if scorable else "not yet scorable"
    return scorable, (
        f"{prefix}: {accepted} of {runs} runs accepted, "
        f"{present} of {len(expected)} composites present"
    )


def verdict(
    dev: HostReading,
    kinds: list[str],
    require_scorable: bool,
) -> tuple[int, str]:
    """
    judge the checkout that gets scored. The verdict word names the stage it certifies, and
    the scoring gap is stated on every verdict, so no reader has to guess what "done" covered
    :param dev: the checkout's reading (HostReading)
    :param kinds: JOBS keys (list of str)
    :param require_scorable: pass only when every run is accepted and every composite exists
        (bool)
    :returns: (exit code, verdict line without the "VERDICT: " prefix) (tuple)
    """
    scorable, gap = scoring_gap(dev, kinds)
    reasons = []
    if any(s.status == UNREADABLE for s in dev.statuses.values()):
        reasons.append("a local run is unreadable")
    if dev.unexpected:
        reasons.append("this checkout holds directories JOBS does not declare")
    if not dev.statuses:
        reasons.append("the selected kinds expand to no expected runs")
    if reasons:
        return _verdict_line("CANNOT TELL", EXIT_CANNOT_TELL, gap, *reasons)
    uncollected = sum(s.status != COLLECTED for s in dev.statuses.values())
    if uncollected:
        return _verdict_line(
            "INCOMPLETE",
            EXIT_INCOMPLETE,
            f"{uncollected} of {len(dev.statuses)} expected runs not collected",
            gap,
        )
    if scorable:
        return _verdict_line("SCORABLE", EXIT_COMPLETE, gap)
    if require_scorable:
        return _verdict_line(
            "INCOMPLETE",
            EXIT_INCOMPLETE,
            "every run collected, --require-scorable not met",
            gap,
        )
    return _verdict_line("COLLECTION COMPLETE", EXIT_COMPLETE, gap)


def writable_certified(reading: HostReading, repo: Path) -> list[Path]:
    """
    every writable file or directory inside an accepted run or a present composite on one host.
    Lock state is integrity, not collection state, so this never feeds the verdict
    :param reading: this checkout's reading (HostReading)
    :param repo: the repo root that reading was taken from (Path)
    :returns: the writable paths, sorted (list of Path)
    """
    certified_dirs = [
        repo / key.rel_dir
        for key, st in reading.statuses.items()
        if st.accepted == KEEPABLE
    ]
    certified_dirs.extend(
        repo / root / f"composite-{MODELS[model].short}"
        for (root, model), pieces in reading.composites.items()
        if pieces is not None
    )
    certified_dirs.extend(reading.undeclared_composites)
    return sorted(
        path
        for directory in certified_dirs
        for path in certified.writable_entries(directory)
    )


def lock_warning(writable: list[Path]) -> str | None:
    """
    the standing integrity line, kept apart from every collection-state line and the verdict
    :param writable: writable_certified's paths (list of Path)
    :returns: the warning line, or None when nothing is writable (str or None)
    """
    if not writable:
        return None
    return (
        f"WARNING: {len(writable)} writable files or directories inside accepted runs or "
        "composites in this checkout (certified data should be read-only; first: "
        f"{writable[0]}); this does not change the verdict"
    )


def print_history(args: argparse.Namespace) -> int:
    """
    print the ops events of the selected runs
    :param args: the parsed command line (argparse.Namespace)
    :returns: 0 when the log was read (even if no event matched), EXIT_HISTORY_UNREADABLE when
        a line is malformed, with the file and line named (int)
    """
    path = ops_events_path(experiment_root(args.repo_root))
    try:
        events = read_events(path)
    except ValueError as exc:
        print(f"ops events unreadable: {exc}")
        return EXIT_HISTORY_UNREADABLE
    if args.kind:
        chosen = events_for_runs(events, args.kind, args.model, args.index)
    else:
        # No --kind: the other experiment's kinds drop out, and an event that names no kind
        # (a data pull can) stays, since filtering by kind would lose it.
        other = {k for k, v in JOBS.items() if v.experiment != args.experiment}
        kept = [e for e in events if e["kind"] not in other]
        chosen = events_for_runs(kept, None, args.model, args.index)
    print(f"{path}: {len(events)} events recorded, {len(chosen)} for this selection")
    for record in chosen:
        print(format_event(record))
    return EXIT_COMPLETE


def main(argv: list[str] | None = None) -> int:
    """
    read the checkout, print the report, exit with the verdict's code
    :param argv: command-line arguments, sys.argv[1:] when None (list of str or None)
    :returns: exit code (int)
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--kind",
        action="append",
        choices=indexed_kinds(),
        help="restrict to these JOBS kinds (repeatable); default every kind of the "
        "selected experiment",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=(7, 8),
        default=None,
        help="which experiment's kinds the default set holds (default 7); not with --kind",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repo root to read (default: this checkout)",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="print the operational events "
        "recorded for the runs --kind/--model/--index select, from "
        "data/run_experiment/ops_events.jsonl, and exit; reads no host and "
        "makes no claim about collection state",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(MODELS),
        help="with --history: restrict to these models (repeatable)",
    )
    parser.add_argument(
        "--index",
        action="append",
        type=int,
        help="with --history: restrict to these run indices (repeatable)",
    )
    parser.add_argument(
        "--require-scorable",
        action="store_true",
        help="pass only when every run is accepted and every composite exists",
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_COMPLETE
    if args.kind and args.experiment is not None:
        print(
            "ladder_state: --experiment selects the default kind set and cannot be given with --kind"
        )
        return EXIT_USAGE
    if (args.model or args.index) and not args.history:
        print("ladder_state: --model and --index apply only with --history")
        return EXIT_USAGE
    args.experiment = DEFAULT_EXPERIMENT if args.experiment is None else args.experiment
    if args.history:
        return print_history(args)
    kinds = args.kind or default_kinds(args.experiment)

    try:
        dev = read_host(args.repo_root, kinds)
    except ValueError as exc:
        print(
            f"\nVERDICT: {_verdict_line('CANNOT TELL', EXIT_CANNOT_TELL, f'this checkout could not be read: {exc}')[1]}"
        )
        return EXIT_CANNOT_TELL

    print(f"local: {dev.root} read {dev.read_at}")
    print("\nLocal:")
    print("\n".join(summary_lines(dev, kinds, "local")))
    details = detail_lines(dev)
    if details:
        print("\nLocal runs not collected:")
        print("\n".join(details))
    if dev.unexpected:
        print("\nLocal directories JOBS does not declare:")
        print("\n".join(f"  {d}" for d in dev.unexpected))
    if dev.undeclared_composites:
        print(
            "\nLocal composites at no path JOBS declares (not counted above; "
            "the lock scan reads them):"
        )
        print(
            "\n".join(
                f"  {c.relative_to(args.repo_root)}" for c in dev.undeclared_composites
            )
        )
    try:
        warning = lock_warning(writable_certified(dev, args.repo_root))
    except OSError as exc:
        warning = f"WARNING: lock state could not be read ({exc}); this does not change the verdict"
    if warning:
        print(f"\n{warning}")
    code, text = verdict(dev, kinds, args.require_scorable)
    print(f"\nVERDICT: {text}")
    return code


if __name__ == "__main__":
    sys.exit(main())
