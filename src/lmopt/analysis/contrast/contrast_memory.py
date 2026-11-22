"""
Suite 2 of the contrast suites: the memory ladder, E3 against E4.

E3 is the change in constant-action share from the M0 baseline to each memory rung, per (arm,
game). E4 is the falsification clause written before collection: "memory reduces state-blindness"
is REFUTED if the delta sits inside the between-seed spread at every rung, or is non-monotone with
no rung clearing it (E2 defines the population, E4 the clause).

- Bins come from `state_sensitivity.bin_stats` and rung selection from
  `scoring._actor.condition_rows`; the statistic is not rebuilt here.
- Deltas are taken over the intersection of bins populated in both arms, fixed once per (model,
  game) from the pooled frames. A degenerate arm reaches an easier population, so
  comparing each arm over its own would flatter it.
- The hypothesis is directional: a rung that clears the spread by raising the share is
  CONTRARY, not support. SUPPORTED needs a clearing negative rung on a monotone non-increasing
  ladder.
- A ladder that cannot be read (one seed, empty intersection, incomparable epoch, undeclared
  rung, no M0) is INDETERMINATE with the reason attached, never a zero or a silent skip.

A harness-fallback row is not the model's choice and is left out of every bin. Each E3 row reports
each arm's fallback count at its rung and game, and a game whose rows are all fallbacks reads
`NO_INTERSECTION` with its counts. A ladder is per model, since one model's M0 against another's M1
would measure the model gap, and per prompt epoch. Scoring stays offline: these are
pure functions over the logged frames.
"""

from __future__ import annotations

from typing import Mapping, cast

import polars as pl

from lmopt.analysis.contrast.contrast_arms import (
    ContrastArm,
    comparable,
    filter_excluded,
    lm_decision_rows,
)
from lmopt.analysis.contrast.contrast_think import (
    STATUS_NOT_COMPARABLE,
    _rate,
    parse_opponent,
)
from lmopt.analysis.state_sensitivity import bin_stats, fallback_drop_report
from lmopt.prompting.memory import MEMORY_RUNGS
from lmopt.scoring._actor import condition_rows

# The reading of suite 2 until a memory ladder is collected.
NOT_MEASURED_LABEL = (
    "NOT_MEASURED (no memory arms; populate configs/contrast_arms.yaml "
    "memory_arms once a memory ladder is collected)"
)

# Named on every row: which population the shares are denominated on.
POPULATION_RULE = (
    "intersection of populated bins, fixed once from the pooled frames "
    "(E2)"
)
# Named on every row: the clause a verdict is read against.
FALSIFICATION_CLAUSE = "E4"

# Named on every row: which direction of the delta favours the hypothesis.
SIGN_NOTE = (
    "delta = share_rung - share_m0, so NEGATIVE means the rung lowered the "
    "constant-action share, which is the direction memory is hypothesised to "
    "move it."
)
STATUS_OK = "OK"
STATUS_NO_INTERSECTION = "NO_INTERSECTION (E2 population empty)"
STATUS_SPREAD_UNMEASURABLE = "SPREAD_UNMEASURABLE (1 seed)"
# `STATUS_NOT_COMPARABLE` is imported from suite 1 rather than restated, so
# the two suites label one defect with one string.

NOT_MEASURED = "NOT_MEASURED"
REFUTED = "REFUTED"
SUPPORTED = "SUPPORTED"
# A clearing rung that moved the share the wrong way: a finding
# against the hypothesis, which is neither a null nor support.
CONTRARY = "CONTRARY (memory increased constant-action share)"
INDETERMINATE = "INDETERMINATE"
BASELINE_RUNG = "M0"  # The baseline every delta is taken against.
# The rungs a ladder can contain, in ladder order (M1 < M2 < M3).
LADDER_RUNGS = tuple(r for r in MEMORY_RUNGS if r != BASELINE_RUNG)
BinKey = tuple[str, str, str]  # A bin key: (game, opponent, rendered legal_mask).

# The ladder shapes a delta sequence can take, as they appear in a note.
SHAPE_ONE_RUNG = "a single rung"
SHAPE_FALLING = "monotone non-increasing (deltas fall)"
SHAPE_RISING = "monotone non-decreasing (deltas rise)"
SHAPE_NON_MONOTONE = "non-monotone (deltas both rise and fall)"


def verify_rung(arm: ContrastArm) -> str | None:
    """
    Check an arm's config-declared rung against what its shards logged.

    The declared rung is what the ladder is built from, and the logged column
    is what the run actually rendered. If they disagree, every delta taken with
    this arm silently mislabels the condition, so the disagreement is returned
    as a problem string naming the arm and both values.
    :param arm: A loaded memory arm.
    :returns: None when the arm's non-null `memory_rung` values are exactly its
        declared rung; otherwise a human-readable problem string naming the
        arm.
    """
    if arm.rung is None:
        return (
            f"arm {arm.name!r}: role {arm.role!r} but the config declares no rung; a memory "
            f"arm without a rung cannot be placed on a ladder"
        )
    if arm.rung not in MEMORY_RUNGS:
        return (
            f"arm {arm.name!r}: declared rung {arm.rung!r} is not one of "
            f"{list(MEMORY_RUNGS)}"
        )
    if "memory_rung" not in arm.df.columns:
        return (
            f"arm {arm.name!r}: config declares rung {arm.rung!r} but the shards have no "
            f"'memory_rung' column (a legacy arm), so the declared rung is unverifiable"
        )
    logged = sorted(arm.df["memory_rung"].drop_nulls().unique().to_list())
    if logged != [arm.rung]:
        return (
            f"arm {arm.name!r}: config declares rung {arm.rung!r} but the logged "
            f"memory_rung values are {logged}"
        )
    return None


def comparability_problem(m0: ContrastArm, rung_arm: ContrastArm) -> str | None:
    """
    Refuse a rung arm that is not one prompt epoch with its M0 baseline.

    A prompt change makes previously collected arms incomparable, so a delta taken across epochs would report the prompt edit as
    a memory effect. `comparable` also treats an unknown epoch (a schema-v2 arm
    with no `prompt_spec_hash`) as never matching, not even another unknown.
    :returns: None when the two arms share one known prompt epoch; otherwise a
        problem string naming both arms and both hash sets.
    """
    if comparable(m0, rung_arm):
        return None
    return (
        f"arm {rung_arm.name!r} (rung {rung_arm.rung!r}) is not comparable "
        f"with its baseline {m0.name!r}: prompt_spec_hash sets "
        f"{list(rung_arm.prompt_spec_hashes)} against "
        f"{list(m0.prompt_spec_hashes)}"
    )


def _bin_cells(
    df: pl.DataFrame, *, memory_rung: str = BASELINE_RUNG
) -> tuple[dict[BinKey, tuple[bool, int]], set[BinKey]]:
    """Every `legal_mask` bin of a frame, and the subset that is populated.

    Returns `({(game, opponent, mask): (constant, rows)}, populated_keys)`.
    Both halves are needed: the population (E2) is decided on the pooled frame,
    while a per-seed share is taken over that fixed population even where a
    seed's slice of a bin held one prompt.
    """
    selected = condition_rows(df, memory_rung=memory_rung)
    lm = selected.filter(pl.col("prompt").is_not_null())
    cells: dict[BinKey, tuple[bool, int]] = {}
    populated: set[BinKey] = set()
    if lm.is_empty():
        return cells, populated
    # Grouped on matchup_id rather than a derived opponent column: the
    # matchup -> opponent map is injective (distinct values, identity on
    # unknown ids), so no two groups can collide on one key.
    for (game, matchup), cell in lm.group_by(["game", "matchup_id"]):
        opponent = parse_opponent(str(matchup))
        for stat in bin_stats(cell):
            key = (str(game), opponent, stat.mask)
            cells[key] = (stat.constant, stat.rows)
            if stat.multistate:
                populated.add(key)
    return cells, populated


def constant_share_by_bin(
    df: pl.DataFrame, *, memory_rung: str = BASELINE_RUNG
) -> dict[BinKey, tuple[bool, int]]:
    """
    The populated bins of one arm's frame, with each one's constancy.

    The population is E2's: bins of a (game, opponent) cell that held at least
    two distinct prompts (`bin_stats.multistate`). A single-prompt bin is
    dropped rather than counted as constant, because a constant action there is
    a determinism observation and not a state-blindness one
    (`state_sensitivity.py` module docstring).
    :param df: A logged decision-node frame.
    :param memory_rung: The rung to select, through `condition_rows` (which keeps
        null-rung rows at every rung; those are non-LM rows and the LM
        filter drops them here anyway).
    :returns: `{(game, opponent, mask): (constant, rows)}` over populated bins only.
    :raises ValueError: From `condition_rows`, if the rung is unknown or a non-M0
        rung is asked of a frame with no `memory_rung` column.
    """
    cells, populated = _bin_cells(df, memory_rung=memory_rung)
    return {key: value for key, value in cells.items() if key in populated}


def _share_over(
    bins: Mapping[BinKey, tuple[bool, int]], keys: list[BinKey]
) -> tuple[float | None, int]:
    """(constant-bin row share, row count) over `keys`; (None, 0) when empty.

    None rather than 0.0 on an empty population: 0.0 is this statistic's
    cleanest possible reading, so returning it from no evidence would assert
    perfect state-sensitivity (the same rule `multistate_share` follows).
    """
    total = sum(bins[k][1] for k in keys)
    if total == 0:
        return None, 0
    constant = sum(bins[k][1] for k in keys if bins[k][0])
    return constant / total, total


def _model_ids(arm: ContrastArm) -> list[str]:
    """Sorted unique non-null `model_id` values on the arm's LM rows."""
    if "model_id" not in arm.df.columns:
        return []
    return sorted(_lm_values(arm.df, "model_id", str))


def _lm_values(df: pl.DataFrame, column: str, kind: type) -> set:
    """
    the distinct non-null values of `column` on a frame's LM rows
    :param df: an arm's frame (pl.DataFrame)
    :param column: the column to read, e.g. "seed" or "game" (str)
    :param kind: the type each value is coerced to, e.g. `int` or `str` (type)
    :returns: the set of coerced values (set)
    """
    lm = lm_decision_rows(df)
    return {kind(v) for v in lm[column].drop_nulls().unique().to_list()}


def _selected_lm(df: pl.DataFrame, memory_rung: str) -> pl.DataFrame:
    """
    an arm's LM rows at the rung it is read at, the population the dropped count is over
    :param df: a logged decision-node frame (pl.DataFrame)
    :param memory_rung: the arm's rung, selected through `condition_rows` (str)
    :returns: the LM rows (non-null prompt) at that rung (pl.DataFrame)
    """
    return condition_rows(df, memory_rung=memory_rung).filter(
        pl.col("prompt").is_not_null()
    )


def _rung_of(arm: ContrastArm) -> str:
    """The arm's declared rung, defaulting to the baseline for a non-memory arm."""
    return arm.rung or BASELINE_RUNG


def _memory_row(
    *,
    model: str | None,
    game: str,
    rung: str,
    m0_name: str,
    rung_name: str,
    status: str,
    share_m0: float | None = None,
    share_rung: float | None = None,
    delta: float | None = None,
    n_bins: int = 0,
    n_rows_m0: int = 0,
    n_rows_rung: int = 0,
    seed_deltas: dict[int, float | None] | None = None,
    n_bins_seed: dict[int, int] | None = None,
    seed_spread: float | None = None,
    n_dropped_m0: int | None = None,
    n_total_m0: int | None = None,
    n_dropped_rung: int | None = None,
    n_total_rung: int | None = None,
    note: str = SIGN_NOTE,
) -> dict:
    """
    one E3 row, with the three strings that make it readable on its own
    The dropped counts are None on a row whose count is not defined (a refusal row: nothing is
    computed across prompt epochs); `dropped_rate_m0` and `dropped_rate_rung` are derived from
    the counts here so a row cannot report a rate that disagrees with them.
    :param n_dropped_m0: harness-fallback LM rows of the M0 arm at this game (int or None)
    :param n_total_m0: LM rows of the M0 arm at this game, before the drop (int or None)
    :param n_dropped_rung: the same over the rung arm (int or None)
    :param n_total_rung: the same over the rung arm (int or None)
    :returns: the row dict (dict)
    """
    return {
        "model": model,
        "game": game,
        "rung": rung,
        "arm_m0": m0_name,
        "arm_rung": rung_name,
        "share_m0": share_m0,
        "share_rung": share_rung,
        "delta": delta,
        "n_bins": n_bins,
        "n_rows_m0": n_rows_m0,
        "n_rows_rung": n_rows_rung,
        "n_dropped_m0": n_dropped_m0,
        "n_total_m0": n_total_m0,
        "dropped_rate_m0": _rate(n_dropped_m0, n_total_m0),
        "n_dropped_rung": n_dropped_rung,
        "n_total_rung": n_total_rung,
        "dropped_rate_rung": _rate(n_dropped_rung, n_total_rung),
        "dropped_unit": None if n_total_m0 is None else "rows",
        "seed_deltas": seed_deltas if seed_deltas is not None else {},
        "n_bins_seed": n_bins_seed if n_bins_seed is not None else {},
        "seed_spread": seed_spread,
        "status": status,
        "population_rule": POPULATION_RULE,
        "falsification_clause": FALSIFICATION_CLAUSE,
        "note": note,
    }


def e3_delta(m0: ContrastArm, rung_arm: ContrastArm, *, seeds: bool) -> list[dict]:
    """
    one E3 delta row per game, over the E2 intersection population
    The population is fixed once per (model, game): the bins populated in both pooled
    frames. Per-seed deltas use that key set, restricted to the keys the seed holds in both arms,
    and `n_bins_seed` beside `seed_deltas` shows a seed that saw fewer bins. Re-deriving the
    population per seed would let it move with the quantity being measured.
    A game gets a row when a bin is populated in either arm, or the fallback drop removed rows
    from it in either arm (so an all-fallback game reads `NO_INTERSECTION` with its counts). Each
    row carries the per-side dropped counts from `state_sensitivity.fallback_drop_report`.
    :param m0: the M0 baseline arm of this model's ladder (ContrastArm)
    :param rung_arm: the M1/M2/M3 arm being compared against it (ContrastArm)
    :param seeds: when True, also compute the per-seed deltas and their spread, which E4 compares
        the pooled delta against; seeds present in only one arm are not compared (bool)
    :returns: a list of row dicts, one per game reached by either arm. If the arms are not one
        prompt epoch every row is a `NOT_COMPARABLE` refusal with null numbers and
        null dropped counts, and both arm names in its `note` (list[dict])
    """
    m0_df = filter_excluded(m0)
    rung_df = filter_excluded(rung_arm)
    m0_rung, rung_rung = _rung_of(m0), _rung_of(rung_arm)
    model_ids = _model_ids(rung_arm)
    model = model_ids[0] if len(model_ids) == 1 else None

    incomparable = comparability_problem(m0, rung_arm)
    if incomparable is not None:
        return [
            _memory_row(
                model=model,
                game=game,
                rung=rung_rung,
                m0_name=m0.name,
                rung_name=rung_arm.name,
                status=STATUS_NOT_COMPARABLE,
                note=f"{incomparable}. {SIGN_NOTE}",
            )
            for game in sorted(
                _lm_values(m0_df, "game", str) | _lm_values(rung_df, "game", str)
            )
        ]

    m0_bins = constant_share_by_bin(m0_df, memory_rung=m0_rung)
    rung_bins = constant_share_by_bin(rung_df, memory_rung=rung_rung)

    # Hoisted out of the per-game loop: one pass per seed per arm, not one per
    # (game, seed). The population these are read over is the pooled one.
    shared_seeds = (
        sorted(_lm_values(m0_df, "seed", int) & _lm_values(rung_df, "seed", int))
        if seeds
        else []
    )
    seed_cells_m0 = {
        seed: _bin_cells(m0_df.filter(pl.col("seed") == seed), memory_rung=m0_rung)[0]
        for seed in shared_seeds
    }
    seed_cells_rung = {
        seed: _bin_cells(rung_df.filter(pl.col("seed") == seed), memory_rung=rung_rung)[
            0
        ]
        for seed in shared_seeds
    }

    rows: list[dict] = []
    lm_m0 = _selected_lm(m0_df, m0_rung)
    lm_rung = _selected_lm(rung_df, rung_rung)
    drops = {}
    for game in sorted(
        _lm_values(lm_m0, "game", str) | _lm_values(lm_rung, "game", str)
    ):
        drops[game] = (
            fallback_drop_report(lm_m0.filter(pl.col("game") == game)),
            fallback_drop_report(lm_rung.filter(pl.col("game") == game)),
        )
    # A game with a populated bin in either arm, or one the fallback drop removed rows from: a
    # game with neither has nothing to read and nothing that was dropped, so it emits no row.
    games = sorted(
        {k[0] for k in m0_bins}
        | {k[0] for k in rung_bins}
        | {g for g, (d0, d1) in drops.items() if d0.n_dropped or d1.n_dropped}
    )
    for game in games:
        drop_m0, drop_rung = drops[game]
        shared = sorted(k for k in m0_bins if k[0] == game and k in rung_bins)
        share_m0, rows_m0 = _share_over(m0_bins, shared)
        share_rung, rows_rung = _share_over(rung_bins, shared)
        delta = (
            None if share_m0 is None or share_rung is None else share_rung - share_m0
        )

        seed_deltas: dict[int, float | None] = {}
        n_bins_seed: dict[int, int] = {}
        seed_spread: float | None = None
        for seed in shared_seeds:
            cells_m0 = seed_cells_m0[seed]
            cells_rung = seed_cells_rung[seed]
            keys = [k for k in shared if k in cells_m0 and k in cells_rung]
            n_bins_seed[seed] = len(keys)
            seed_m0, _ = _share_over(cells_m0, keys)
            seed_rung, _ = _share_over(cells_rung, keys)
            seed_deltas[seed] = (
                None if seed_m0 is None or seed_rung is None else seed_rung - seed_m0
            )
        measured = [d for d in seed_deltas.values() if d is not None]
        if len(measured) >= 2:
            seed_spread = max(measured) - min(measured)

        if delta is None:
            status = STATUS_NO_INTERSECTION
        elif seeds and seed_spread is None:
            status = STATUS_SPREAD_UNMEASURABLE
        else:
            status = STATUS_OK

        rows.append(
            _memory_row(
                model=model,
                game=game,
                rung=rung_rung,
                m0_name=m0.name,
                rung_name=rung_arm.name,
                status=status,
                share_m0=share_m0,
                share_rung=share_rung,
                delta=delta,
                n_bins=len(shared),
                n_rows_m0=rows_m0,
                n_rows_rung=rows_rung,
                seed_deltas=seed_deltas,
                n_bins_seed=n_bins_seed,
                seed_spread=seed_spread,
                n_dropped_m0=drop_m0.n_dropped,
                n_total_m0=drop_m0.n_total,
                n_dropped_rung=drop_rung.n_dropped,
                n_total_rung=drop_rung.n_total,
                note=_row_note(m0.name, rung_arm.name, drop_m0, drop_rung),
            )
        )
    return rows


def _row_note(m0_name: str, rung_name: str, drop_m0, drop_rung) -> str:
    """
    the E3 row's note: the sign convention, plus what the fallback drop left out when it left
    anything out
    The counts are the arms' whole attrition at this game, not the intersection of bins the
    shares are read over, and the sentence says so, so no reader takes them for the shares'
    own denominator.
    :param m0_name: the M0 arm (str)
    :param rung_name: the rung arm (str)
    :param drop_m0: the M0 arm's `DropReport` at this game (DropReport)
    :param drop_rung: the rung arm's `DropReport` at this game (DropReport)
    :returns: the note text (str)
    """
    if not (drop_m0.n_dropped or drop_rung.n_dropped):
        return SIGN_NOTE
    return (
        f"{SIGN_NOTE} Harness-fallback LM rows are left out of every bin: "
        f"{drop_m0.n_dropped} of {drop_m0.n_total} in {m0_name!r} and "
        f"{drop_rung.n_dropped} of {drop_rung.n_total} in {rung_name!r}, counted over "
        "each arm's rows at this game, not over the intersection of bins the shares "
        "are read over."
    )


def _ladder_shape(deltas: list[float]) -> str:
    """Name the shape of a delta sequence, ordered M1 < M2 < M3.

    A note that calls a rising ladder "non-monotone" describes a shape the data
    does not have, so the three cases are kept apart.
    """
    if len(deltas) < 2:
        return SHAPE_ONE_RUNG
    # Consecutive pairs: the shifted copy is one shorter by design.
    pairs = list(zip(deltas, deltas[1:], strict=False))
    if all(b <= a for a, b in pairs):
        return SHAPE_FALLING
    if all(b >= a for a, b in pairs):
        return SHAPE_RISING
    return SHAPE_NON_MONOTONE


def _ladder_verdict(rows: list[dict]) -> tuple[str, str]:
    """Read one (model, game) ladder against E4; return (verdict, note).

    REFUTED when no rung's |delta| clears its between-seed spread. A clearing rung with a positive
    delta is CONTRARY. SUPPORTED needs a clearing negative rung on a monotone
    non-increasing ladder; any other negative-clearing ladder is INDETERMINATE.
    """
    ordered = sorted(rows, key=lambda r: LADDER_RUNGS.index(r["rung"]))
    incomparable = [r["rung"] for r in ordered if r["status"] == STATUS_NOT_COMPARABLE]
    if incomparable:
        return INDETERMINATE, (
            f"rung(s) {incomparable} are not one prompt epoch with the "
            f"{BASELINE_RUNG} baseline, so no delta can be taken across them"
        )
    blind_rows = [r for r in ordered if r["delta"] is None]
    blind = [r["rung"] for r in blind_rows]
    if blind:
        lost = "".join(
            f"; at {r['rung']} {r['n_dropped_m0']} of {r['n_total_m0']} {BASELINE_RUNG} "
            f"rows and {r['n_dropped_rung']} of {r['n_total_rung']} rung rows are "
            "harness-fallback rows"
            for r in blind_rows
            if r["n_dropped_m0"] or r["n_dropped_rung"]
        )
        return INDETERMINATE, (
            f"the E2 intersection population is empty at rung(s) {blind}, so "
            f"no delta exists to read against E4{lost}"
        )
    unspread = [r["rung"] for r in ordered if r["seed_spread"] is None]
    if unspread:
        return INDETERMINATE, (
            f"the between-seed spread is unmeasurable at rung(s) {unspread} "
            "(one seed), and E4 compares the delta against that spread"
        )

    deltas = [r["delta"] for r in ordered]
    shape = _ladder_shape(deltas)
    clearing = [r for r in ordered if abs(r["delta"]) > r["seed_spread"]]
    if not clearing:
        return REFUTED, (
            f"no rung of {[r['rung'] for r in ordered]} has |delta| above its "
            f"between-seed spread, which is E4's refutation condition; the "
            f"ladder is {shape}"
        )

    cleared = [r["rung"] for r in clearing]
    against = [r["rung"] for r in clearing if r["delta"] > 0]
    if against:
        return CONTRARY, (
            f"rung(s) {against} clear the between-seed spread with a POSITIVE "
            "delta, so memory RAISED the constant-action share, against E4's "
            f"hypothesis; the ladder is {shape}"
        )
    if shape in (SHAPE_ONE_RUNG, SHAPE_FALLING):
        return SUPPORTED, (
            f"rung(s) {cleared} clear the between-seed spread with a negative "
            f"delta and the ladder is {shape}, so E4 does not refute"
        )
    return INDETERMINATE, (
        f"rung(s) {cleared} clear the between-seed spread with a negative "
        f"delta but the ladder is {shape}; E4 refutes only when NO rung "
        "clears, so this is neither refuted nor supported"
    )


def _build_ladders(
    memory_arms: Mapping[str, ContrastArm], problems: list[str]
) -> dict[str, dict[str, ContrastArm]]:
    """Group verified arms into `{model: {rung: arm}}`, appending refusals.

    A rung declared by two arms is ambiguous, so neither is used and the rung is
    remembered as refused: a third arm at the same rung is refused too, rather
    than sliding into the hole the deletion left. Without that memory the
    ladder's rung would silently be "whichever duplicate was sorted last", which
    is the worst of the three possible answers.
    """
    ladders: dict[str, dict[str, ContrastArm]] = {}
    refused: dict[str, set[str]] = {}
    for name in sorted(memory_arms):
        arm = memory_arms[name]
        problem = verify_rung(arm)
        if problem is not None:
            problems.append(problem)
            continue
        model_ids = _model_ids(arm)
        if len(model_ids) != 1:
            problems.append(
                f"arm {name!r}: LM rows carry {len(model_ids)} model_id values "
                f"({model_ids}); a memory ladder is per model, so one arm must "
                "be one model"
            )
            continue
        model = model_ids[0]
        bucket = ladders.setdefault(model, {})
        already_refused = refused.setdefault(model, set())
        rung = cast(str, arm.rung)  # verify_rung refused a None rung above
        if rung in already_refused:
            problems.append(
                f"model {model!r}: arm {name!r} declares rung {rung!r}, "
                "which two earlier arms already made ambiguous; it is refused "
                "too, because a rung nobody can name uniquely cannot be read "
                "off any one of its claimants"
            )
            continue
        if rung in bucket:
            problems.append(
                f"model {model!r}: two arms declare rung {rung!r} "
                f"({bucket[rung].name!r} and {name!r}); the ladder is "
                "ambiguous and neither is used"
            )
            del bucket[rung]
            already_refused.add(rung)
            continue
        bucket[rung] = arm
    return ladders


def _verdict(model: str, game: str | None, verdict: str, note: str) -> dict:
    """One ladder verdict entry; `game` is None for a ladder that never reached a game."""
    return {"model": model, "game": game, "verdict": verdict, "note": note}


def evaluate_e3(memory_arms: Mapping[str, ContrastArm]) -> dict:
    """
    Read every memory ladder's E3 deltas against E4's falsification clause.

    Never raises on the empty path, and never drops an arm quietly: a
    mismatched rung, a two-model arm, a duplicated rung, a rung arm from
    another prompt epoch, and a ladder without an M0 baseline all land in
    `problems` and pull `status` to INDETERMINATE.
    :param memory_arms: `{arm name: ContrastArm}` for the role-`memory` arms the loader returned.
    :returns: `{"status", "rows", "verdict_note", "problems", "ladders"}`.
        `status` is `NOT_MEASURED` on an empty roster; otherwise
        `INDETERMINATE` if any problem was recorded or any ladder could not be
        read, else `CONTRARY` if any ladder moved the share the wrong way (a
        finding must not be masked by another ladder's support), else
        `SUPPORTED` if any ladder survives E4, else `REFUTED`.
        `rows` are the `e3_delta` rows per (model, game, rung); `ladders` is
        one verdict per (model, game).
    """
    if not memory_arms:
        return {
            "status": NOT_MEASURED,
            "rows": [],
            "verdict_note": NOT_MEASURED_LABEL,
            "problems": [],
            "ladders": [],
        }

    problems: list[str] = []
    ladders = _build_ladders(memory_arms, problems)

    rows: list[dict] = []
    verdicts: list[dict] = []
    for model in sorted(ladders):
        bucket = ladders[model]
        baseline = bucket.get(BASELINE_RUNG)
        if baseline is None:
            problems.append(
                f"model {model!r}: the memory ladder has no {BASELINE_RUNG} arm "
                f"(rungs present: {sorted(bucket)}); every E3 delta is taken "
                f"against the {BASELINE_RUNG} baseline, so this ladder measures "
                "nothing"
            )
            verdicts.append(
                _verdict(model, None, INDETERMINATE, f"no {BASELINE_RUNG} arm")
            )
            continue
        rung_arms = [bucket[r] for r in LADDER_RUNGS if r in bucket]
        if not rung_arms:
            verdicts.append(_verdict(model, None, INDETERMINATE, "no rung arms"))
            continue

        by_game: dict[str, list[dict]] = {}
        for arm in rung_arms:
            incomparable = comparability_problem(baseline, arm)
            if incomparable is not None:
                problems.append(incomparable)
            for row in e3_delta(baseline, arm, seeds=True):
                rows.append(row)
                by_game.setdefault(row["game"], []).append(row)
        for game in sorted(by_game):
            verdict, note = _ladder_verdict(by_game[game])
            verdicts.append(_verdict(model, game, verdict, note))

    counts = {v: 0 for v in (SUPPORTED, REFUTED, CONTRARY, INDETERMINATE)}
    for entry in verdicts:
        counts[entry["verdict"]] += 1
    if problems or counts[INDETERMINATE] or not verdicts:
        status = INDETERMINATE
    elif counts[CONTRARY]:
        status = CONTRARY
    elif counts[SUPPORTED]:
        status = SUPPORTED
    else:
        status = REFUTED

    note = (
        f"{len(verdicts)} (model, game) ladder(s) read against "
        f"{FALSIFICATION_CLAUSE}: {counts[SUPPORTED]} supported, "
        f"{counts[REFUTED]} refuted, {counts[CONTRARY]} contrary, "
        f"{counts[INDETERMINATE]} indeterminate; {len(problems)} problem(s). "
        f"Population: {POPULATION_RULE}. {SIGN_NOTE}"
    )
    flagged = [
        f"{v['verdict']} at ({v['model']}, {v['game']}): {v['note']}."
        for v in verdicts
        if v["verdict"] in (INDETERMINATE, CONTRARY)
    ]
    if flagged:
        note = f"{note} " + " ".join(flagged)

    return {
        "status": status,
        "rows": rows,
        "verdict_note": note,
        "problems": problems,
        "ladders": verdicts,
    }
