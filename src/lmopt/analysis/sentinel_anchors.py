"""
Sentinel anchors: the uniform-random agent's own score, per (game, opponent), as
the reference an LM's own score is measured against.

`lmopt.scoring.property3.score_property3(df, actor="random")` reports the regret
of random play at equilibrium value, per game: `<= 0`, visit-weighted. Random
play's regret differs by game and by which opponent it faces, so one anchor per
(game, opponent) cell is needed rather than one number for all fifteen P3
scoreboard cells. `lmopt.scoring.property6.score_property6(df, actor="random")`
reports the same agent's off-path total variation from the trembled reference,
and differs by game for the same reason, so P6's anchors have the same shape and
are built by the same functions.

This module holds the pure statistics and lookup over the anchor artefacts in
`configs/anchors/`.

Scoring is offline: every function here is a pure computation over
its arguments, no file I/O.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scipy.stats import t as _student_t

#: The provenance fields whose disagreement voids a sentinel anchor: each one
#: changes what the random agent's regret actually measures (a different
#: reference iteration count solves a different equilibrium, a different seat
#: mode changes which seat's regret table a row is looked up in, a different
#: pyspiel version can change the solved reference itself).
VOIDING_PROVENANCE_FIELDS = (
    "cfr_iters",
    "rl_iterations",
    "seat_mode",
    "pyspiel_version",
)

#: The fields whose disagreement voids an off-path (P6) anchor. The opening and
#: the seat mode select the subgame the distance is measured inside, and they
#: differ by game, so they are recorded on each game's own anchor and looked up
#: there; the tremble mass defines the reference the distance is taken from, and
#: a different mass is a different reference, and the estimand is what the mean averages
#: over. `rl_iterations` is absent because
#: the off-path run faces the CFR reference alone, so no trained policy enters
#: the measurement.
P6_VOIDING_PROVENANCE_FIELDS = (
    "cfr_iters",
    "seat_mode",
    "drive_openings",
    "tremble_eps",
    "estimand",
    "pyspiel_version",
)

#: What the sampled off-path mean averages over, recorded on the anchor and on the run. An
#: anchor built from one averaging is not an anchor for a value built from another, so a
#: disagreement voids it. The mean is taken over the reference's symmetry classes of the
#: logged info-sets, each class counted once (`lmopt.scoring.symmetry`).
P6_ESTIMAND = "symmetry_class_mean"


#: What the off-path anchors measure: a sampled-action distance, collected from a
#: uniform-random agent's logged moves. The distribution basis has its own baseline,
#: computed analytically at scoring time from the reference alone, and the two never
#: cross: an artefact that does not say which basis it holds cannot be told from one
#: built the other way, and a sampled anchor judging a distribution distance would
#: compare a point-mass estimator against a mixed policy.
SAMPLED_BASIS = "sampled"


def basis_refusal(artefact: Mapping[str, Any]) -> str | None:
    """
    the reason a sampled off-path verdict cannot be issued against an artefact,
    or None when the artefact states that it holds sampled-basis anchors
    :param artefact: the artefact `build_artefact` produced, or read back from the
        committed JSON (Mapping[str, Any])
    :returns: None when the artefact's provenance records basis `"sampled"`, else a
        str naming what it records instead, so every cell states why it carries no
        verdict rather than judging a sampled distance against an anchor whose basis
        nobody declared (str | None)
    """
    recorded = artefact.get("provenance", {}).get("basis")
    if recorded == SAMPLED_BASIS:
        return None
    if recorded is None:
        return (
            "the off-path sentinel anchor artefact records no basis, so it cannot be "
            f"read as the {SAMPLED_BASIS} anchors the sampled verdict needs"
        )
    return (
        f"the off-path sentinel anchor artefact records basis {recorded!r}, not "
        f"{SAMPLED_BASIS!r}, so it does not judge the sampled distance"
    )


def replicate_stats(per_seed_values: Sequence[float]) -> dict[str, Any]:
    """
    the replicate mean, sample standard deviation, count and two-sided 95%
    Student-t interval of one (game, opponent) cell's per-seed regret values
    :param per_seed_values: one regret value per seed, the unit of independent
        replication for this run (the seed, not the hand) (Sequence[float])
    :returns: dict with "mean", "sd", "n", "ci_low", "ci_high"; "mean" is None
        with zero seeds, and "sd"/"ci_low"/"ci_high" are None below two seeds
        rather than a fabricated width -- the caller states that, it never
        substitutes a wider or narrower interval in its place
    """
    n = len(per_seed_values)
    if n == 0:
        return {"mean": None, "sd": None, "n": 0, "ci_low": None, "ci_high": None}

    mean = sum(per_seed_values) / n
    if n < 2:
        return {"mean": mean, "sd": None, "n": n, "ci_low": None, "ci_high": None}

    variance = sum((v - mean) ** 2 for v in per_seed_values) / (n - 1)
    sd = math.sqrt(variance)
    standard_error = sd / math.sqrt(n)
    critical_value = float(_student_t.ppf(0.975, df=n - 1))
    half_width = critical_value * standard_error
    return {
        "mean": mean,
        "sd": sd,
        "n": n,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def build_artefact(
    per_seed_by_cell: Mapping[tuple[str, str], Mapping[str, Sequence]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """
    assemble the committed anchor artefact from every (game, opponent) cell's
    per-seed regret values and the run-level provenance that would void them
    :param per_seed_by_cell: `{(game, opponent): {"seeds": [...], "values": [...]}}`,
        one regret value per seed, in the same order as its seed. Any other key
        a cell carries is copied onto that cell's anchor verbatim, which is how a
        voiding field that differs by game (an off-path opening, a seat mode) is
        recorded where it belongs instead of at the run level, where one value
        would have to stand for every game
        (Mapping[tuple[str, str], Mapping[str, Sequence]])
    :param provenance: the run-level provenance to embed verbatim, read back
        by `anchor_for` against the run being scored (Mapping[str, Any])
    :returns: `{"provenance": ..., "anchors": {game: {opponent: {mean, sd, n,
        ci_low, ci_high, seeds, per_seed, ...}}}}`
    :raises ValueError: when a cell's seeds and values differ in length, or a cell's extra key
        is named like a computed field
    """
    anchors: dict[str, dict[str, dict[str, Any]]] = {}
    for (game, opponent), cell in per_seed_by_cell.items():
        if len(cell["seeds"]) != len(cell["values"]):
            raise ValueError(
                f"anchor cell (game={game!r}, opponent={opponent!r}) has "
                f"{len(cell['seeds'])} seeds and {len(cell['values'])} values; each value "
                "belongs to the seed in the same position, so pass one value per seed"
            )
        stats = replicate_stats(list(cell["values"]))
        extra = {k: v for k, v in cell.items() if k not in ("seeds", "values")}
        colliding = sorted(set(extra) & (set(stats) | {"per_seed"}))
        if colliding:
            raise ValueError(
                f"anchor cell (game={game!r}, opponent={opponent!r}) carries {colliding}, "
                "which would replace the computed fields of the same name; rename them"
            )
        anchors.setdefault(game, {})[opponent] = {
            **stats,
            "seeds": list(cell["seeds"]),
            "per_seed": list(cell["values"]),
            **extra,
        }
    return {"provenance": dict(provenance), "anchors": anchors}


def anchor_for(
    artefact: Mapping[str, Any],
    game: str,
    opponent: str,
    run_provenance: Mapping[str, Any],
    voiding_fields: Sequence[str] = VOIDING_PROVENANCE_FIELDS,
) -> dict[str, Any] | str:
    """
    the sentinel anchor for one (game, opponent) cell, or the reason it cannot
    be used against the run described by `run_provenance`
    A missing cell and a voided cell are both reported by name; neither ever
    falls back to another cell's anchor -- a cell uses its own (game, opponent)
    anchor or none at all.
    :param artefact: the artefact `build_artefact` produced, or read back from
        the committed JSON (Mapping[str, Any])
    :param game: the game key of the cell being scored (str)
    :param opponent: the opponent key of the cell being scored (str)
    :param run_provenance: the provenance of the run being scored, carrying
        the same `voiding_fields` keys as the artefact's own provenance. A
        field the run does not record is not a field that agrees: it voids the
        anchor, with the absence named as the reason, because an anchor solved
        under conditions the run cannot state is an anchor for something else.
        A field recorded as `{game: value}` is read at this cell's game
        (Mapping[str, Any])
    :param voiding_fields: the fields whose disagreement voids the anchor; each
        one's expected value comes from the CELL's own anchor when the anchor
        records it and from the run-level provenance otherwise, so a field that
        differs by game is compared against that game's value and never against
        one game's value standing for all five. A field neither records voids the
        anchor (Sequence[str])
    :returns: the anchor dict on a clean match, else a str naming why the cell
        is unusable
    """
    game_anchors = artefact.get("anchors", {}).get(game)
    if game_anchors is None or opponent not in game_anchors:
        return f"no sentinel anchor for (game={game!r}, opponent={opponent!r})"

    artefact_provenance = artefact.get("provenance", {})
    cell = game_anchors[opponent]
    for field in voiding_fields:
        if field not in cell and field not in artefact_provenance:
            return (
                f"sentinel anchor for (game={game!r}, opponent={opponent!r}) voided: "
                f"the anchor run records no {field}, so no run can be shown to agree with it"
            )
        expected = cell[field] if field in cell else artefact_provenance[field]
        if field not in run_provenance:
            return (
                f"sentinel anchor for (game={game!r}, opponent={opponent!r}) voided: "
                f"this run records no {field}, and the anchor run's is {expected!r}"
            )
        actual = run_provenance[field]
        if isinstance(actual, Mapping):
            # A game-disjoint composite records a per-game field as {game: value}; the anchor
            # side is already resolved per game, so the run side is too.
            if game not in actual:
                return (
                    f"sentinel anchor for (game={game!r}, opponent={opponent!r}) voided: "
                    f"this run records no {field} for {game!r}, and the anchor run's is "
                    f"{expected!r}"
                )
            actual = actual[game]
        if expected != actual:
            return (
                f"sentinel anchor for (game={game!r}, opponent={opponent!r}) voided: "
                f"{field} is {actual!r} on this run, {expected!r} on the anchor run"
            )

    return cell
