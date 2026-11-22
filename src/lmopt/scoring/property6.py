"""
P6, off-path rationality: how the agent plays inside a subgame the
equilibrium reference (almost) never reaches.

The estimand is a per-info-set distance between the agent's realised policy at
the driven info-sets and a reference that is defined there. Plain CFR is not:
at a zero-reach info-set the average policy is whatever the solver's regrets
happened to leave, so a distance from it would be partly a function of the
solver run. The tremble blend (`games/cfr.py`'s `tremble_policy`) gives every
info-set strictly positive reach and is a pure function of the CFR+ average
policy, so it is a single reproducible reference everywhere -- which is why
this scorer takes a trembled policy and refuses to invent one.

The rows this scores. Only a driven arm's (`probe_type="offpath_drive"`,
collected through `run_match(drive_openings=...)`). A frame holding any other
row is REFUSED rather than filtered: pooling driven and canonical rows would
produce one number over two populations, and P6's whole content is which
population it was measured on.

Scoring is offline: pure functions over a logged frame plus a
reference policy. The trembled policy is deliberately not picklable and not
cacheable (see `tremble_policy`), which is safe precisely because this path is
single-process.
"""

from __future__ import annotations

from typing import Optional

import polars as pl
import pyspiel

from lmopt.games.cfr import DEFAULT_TREMBLE_EPS
from lmopt.games.cfr import tremble_policy as _blend
from lmopt.logging.schema import OFFPATH_PROBE_TYPE
from lmopt.scoring import symmetry
from lmopt.scoring._actor import (
    LEGACY_PROMPT,
    actor_label,
    actor_rows,
    drop_fallback_rows,
    require_actor_column,
)

# The same two primitives the paired scorer (P1/P4) uses, imported rather than
# restated: "TV distance" and "the empirical policy at an info-set" must mean
# one thing across the properties, and a second copy is how they stop meaning
# it. `_empirical` also drops null actions and returns its own sample size.
from lmopt.scoring.property_paired import _empirical, _total_variation

_REQUIRED_COLUMNS = (
    "game",
    "info_set",
    "player_to_act",
    "parsed_action",
    "probe_type",
)


def offpath_reference(
    game: pyspiel.Game,
    iterations: int,
    *,
    tremble_eps: float = DEFAULT_TREMBLE_EPS,
    plus: bool = True,
    cache_dir: Optional[str] = None,
    require_hit: bool = False,
):
    """
    The P6 reference: the cached CFR(+) base policy, trembled at load.
    The trembled policy is an instance of a
    function-local class, so it neither pickles nor fits `policy_cache`'s
    TabularPolicy persistence; the cache therefore holds the base, under the
    ordinary CFR cache key that every other consumer shares, and the blend
    (cheap arithmetic on the base's output) is applied here on load. Solving
    once and blending is why this does not call `solve_cfr_tremble`, which
    would re-solve on every call and cache nothing.
    `tremble_eps` stays at the production default unless a caller has a reason:
    the reference should be the equilibrium wherever the equilibrium is
    defined.
    """
    from lmopt.games.policy_cache import DEFAULT_CFR_DIR, load_or_solve_cfr

    base = load_or_solve_cfr(
        game,
        iterations,
        plus=plus,
        cache_dir=DEFAULT_CFR_DIR if cache_dir is None else cache_dir,
        require_hit=require_hit,
    )
    return _blend(game, base, tremble_eps=tremble_eps)


def reference_policy_map(game: pyspiel.Game, policy) -> dict:
    """
    {(info_set, player): {action: probability}} for every decision node.

    A full walk of the game tree, which is the same tractability envelope the
    CFR reference itself has: an info-set string cannot be turned back into a
    state, so the only way to ask a Policy what it does at a logged info-set is
    to visit the tree once and index it.

    Keyed on (info_set, player), never on the string alone: two players can
    render the same information-state string, and scoring one seat's rows
    against the other seat's reference would be a silent mismatch.
    """
    out: dict[tuple[str, int], dict[int, float]] = {}
    stack = [game.new_initial_state()]
    while stack:
        state = stack.pop()
        if state.is_terminal():
            continue
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                stack.append(state.child(action))
            continue
        player = state.current_player()
        key = (state.information_state_string(player), player)
        if key not in out:
            out[key] = dict(policy.action_probabilities(state, player))
        for action in state.legal_actions():
            stack.append(state.child(action))
    return out


def uniform_baseline(reference: dict, info_sets, weights) -> dict:
    """
    the analytic distance a uniform-random policy stands at from the reference,
    over the info-sets a run actually reached and with that run's own weights
    A uniform-random agent's exact policy at an info-set is uniform over the legal
    actions, so its distance from the reference there is a pure function of the
    reference and needs no collection. The mean is visit-weighted, so the baseline
    describes the same population of decisions the measured distance does; an
    unweighted mean would compare a weighted number against an unweighted one.
    :param reference: `{(info_set, player): {action: probability}}`, as
        :func:`reference_policy_map` returns it (dict)
    :param info_sets: the `(info_set, player)` keys to average over, which are the
        keys the scored run reached (Iterable[tuple[str, int]])
    :param weights: `{info_set: visits}`, the same weights the measured distance
        uses (Mapping[str, int])
    :returns: dict with "value" (float, the visit-weighted mean, NaN when nothing
        carries weight) and "by_infoset" ({info_set: the uniform distance there})
    """
    by_infoset: dict[str, float] = {}
    numerator = 0.0
    denominator = 0
    for info_set, player in info_sets:
        ref = reference[(info_set, int(player))]
        uniform = {action: 1.0 / len(ref) for action in ref}
        by_infoset[info_set] = _total_variation(uniform, ref)
        weight = weights.get(info_set, 0)
        numerator += by_infoset[info_set] * weight
        denominator += weight
    return {
        "value": (numerator / denominator) if denominator else float("nan"),
        "by_infoset": by_infoset,
    }


def _walk(df, tremble_policy, game, actor, where: str):
    """
    the one per-info-set walk both P6 scorers take over a driven arm
    Shared rather than copied: the drop of the scored LM's fallback rows, the
    refusal of a frame that is not one driven arm of one game, the reference
    lookup and the visit weights are the same facts about the same rows whichever
    policy the distance is then taken between, and a second copy of the loop is
    how the two bases come to score different populations.
    :param df: one driven arm's rows (pl.DataFrame)
    :param tremble_policy: the reference from :func:`offpath_reference` (open_spiel Policy)
    :param game: the OpenSpiel game, or None to load the one the frame names
        (pyspiel.Game | None)
    :param actor: which agent's rows to score, None for the legacy population (str | None)
    :param where: the calling scorer's name, used in every refusal (str)
    :returns: (entries, reference, report, game_key, game), where an entry is
        (info_set, player, reference policy, empirical policy, visits, nulls dropped)
        in info-set order
    :raises ValueError: when the frame is not one driven arm of one game, when the
        reference has no entry for a logged info-set, or when two seats logged one
        info-set string
    """
    require_actor_column(df, actor, where)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{where}: frame is missing column(s) {missing}. P6 is a "
            f"distance at a (game, info_set, acting player), and it cannot be "
            f"reconstructed from a frame that lacks them."
        )

    # Nulls count as foreign. Dropping nulls here would let a frame whose
    # probe_type is null throughout pass the refusal and be scored as driven,
    # which is the fabricated-label hazard read backwards: the rows would
    # carry no claim at all and P6 would supply one. `run_match` cannot write
    # a null (the column is filled from a validated argument), which is
    # exactly why the guard costs nothing.
    foreign = sorted(
        {p for p in df["probe_type"].to_list() if p != OFFPATH_PROBE_TYPE},
        key=lambda p: (p is not None, str(p)),
    )
    if foreign:
        raise ValueError(
            f"{where}: the frame carries probe_type(s) {foreign} beside "
            f"{OFFPATH_PROBE_TYPE!r} (null counts as foreign: a row that makes no "
            f"claim about its arm is not evidence of a driven one). P6's estimand "
            f"is the policy INSIDE the driven subgame; pooling canonical rows in "
            f"would report one number over two populations. Score the driven "
            f"arm's own directory."
        )

    game_keys = sorted(set(df["game"].drop_nulls().to_list()))
    if len(game_keys) > 1:
        raise ValueError(
            f"{where}: the frame spans games {game_keys}. One reference "
            f"policy describes one game; score each game's arm separately."
        )
    game_key = game_keys[0] if game_keys else None
    if game is not None and game_key is not None:
        from lmopt.games.loaders import GAMES, load_game

        if game_key in GAMES and str(load_game(game_key)) != str(game):
            raise ValueError(
                f"{where}: the frame's rows are labelled game {game_key!r} "
                f"({load_game(game_key)}) but the game passed is {game}. Scoring one "
                f"game's rows against another game's reference reads every info-set as "
                f"unmatched or, worse, matches strings that mean different things."
            )
    if game is None:
        if game_key is None:
            raise ValueError(
                f"{where}: the frame names no game and none was passed, "
                f"so there is nothing to build a reference over."
            )
        from lmopt.games.loaders import load_game

        game = load_game(game_key)

    report = drop_fallback_rows(
        actor_rows(df, actor, legacy=pl.col("prompt").is_not_null())
    )
    rows = report.kept
    reference = reference_policy_map(game, tremble_policy)

    entries: list[tuple] = []
    seen: set[str] = set()
    unmatched: list[tuple[str, int]] = []
    split: list[str] = []
    keys = (
        rows.select("info_set", "player_to_act")
        .unique()
        .sort(["info_set", "player_to_act"])
        .rows()
    )
    for info_set, player in keys:
        if info_set in seen:
            # Two seats logging one info-set string: the output is keyed on the
            # string, so this would silently overwrite one seat's distance.
            split.append(info_set)
            continue
        ref = reference.get((info_set, int(player)))
        if ref is None:
            unmatched.append((info_set, int(player)))
            continue
        at = rows.filter(
            (pl.col("info_set") == info_set) & (pl.col("player_to_act") == player)
        )
        empirical, n, n_null = _empirical(at)
        seen.add(info_set)
        entries.append((info_set, int(player), ref, empirical, n, n_null))

    if unmatched:
        raise ValueError(
            f"{where}: the reference has no entry for {unmatched[:5]} "
            f"({len(unmatched)} in total). Every decision node of the game is in "
            f"the reference, so an info-set that is not means the frame and the "
            f"reference are not about the same game. Scoring the rest would "
            f"report a confident mean over whatever happened to match."
        )
    if split:
        raise ValueError(
            f"{where}: info-set(s) {split[:5]} were logged by more than "
            f"one player, so a distance keyed on the string alone would be two "
            f"seats' play under one key."
        )
    return entries, reference, report, game_key, game


def score_property6(
    df: pl.DataFrame,
    tremble_policy,
    game: Optional[pyspiel.Game] = None,
    actor: Optional[str] = None,
) -> dict:
    """
    per-info-set total variation between a driven arm and the trembled reference
    The scored LM's own harness-fallback rows are dropped before any distance is
    computed, so a substituted move is never read as the model's choice. An
    info-set left with no kept row leaves `tv_by_infoset`.
    :param df: one driven arm's rows, `probe_type="offpath_drive"` on every row (pl.DataFrame)
    :param tremble_policy: the reference from :func:`offpath_reference`, passed in
        so the caller owns the iteration count and the tremble mass, both of
        which are provenance (open_spiel Policy)
    :param game: the OpenSpiel game the frame was collected on, defaulting to the
        registry game named by the frame's own `game` column; when both are known and
        differ, the call raises (pyspiel.Game | None)
    :param actor: which agent's rows to score; `None` keeps the legacy
        prompt-not-null population, which is the LM's rows (str | None)
    :returns: dict with keys
        "tv_by_infoset" ({info_set: tv} at the driven info-sets),
        "visits_by_infoset" ({info_set: n}, the sample size behind each distance,
        because TV between two distributions is upward-biased at finite n),
        "tv_mean" (float, NaN when nothing was scored, since 0.0 would read as
        "played the reference exactly"; the mean over symmetry classes of each class's
        mean distance, so logged info-sets the reference treats as interchangeable count
        once, and over the logged info-sets themselves when the game has no such classes),
        "tv_mean_logged" (float, the mean over the logged info-sets, each counted once),
        "n_symmetry_classes" (int, the classes `tv_mean` averages over),
        "tv_mean_visit_weighted" (float, the same distances weighted by visits),
        "n_offpath_infosets" (int, info-sets reached inside the driven subgame),
        "n_rows_scored" (int),
        "n_null_dropped" (int, rows at a scored info-set whose `parsed_action`
        was null and which are therefore not in `n_rows_scored`),
        "game" (the frame's game key),
        "actor_filter" (the population every number was computed on),
        "dropped_rate" (float | None, fallback rows over the scored LM's rows;
        None when the population holds no LM row),
        "dropped_detail" (the counts behind that rate, `DropReport.detail()`)
    """
    entries, reference, report, game_key, game = _walk(
        df, tremble_policy, game, actor, "score_property6"
    )

    tv_by_infoset: dict[str, float] = {}
    visits_by_infoset: dict[str, int] = {}
    weighted_num = 0.0
    weighted_den = 0
    n_rows = 0
    n_null_dropped = 0

    for info_set, _player, ref, empirical, n, n_null in entries:
        # Summed into the result rather than discarded here. `n_rows_scored` counts rows that produced an action, so
        # without this a null-action row is invisible in both directions -- it is
        # neither scored nor reported as skipped.
        n_null_dropped += n_null
        tv_by_infoset[info_set] = _total_variation(empirical, ref)
        visits_by_infoset[info_set] = n
        weighted_num += tv_by_infoset[info_set] * n
        weighted_den += n
        n_rows += n

    values = list(tv_by_infoset.values())
    short_name = game.get_type().short_name
    classes = symmetry.card_classes(reference, short_name)
    by_class: dict[str, list[float]] = {}
    for info_set, tv in tv_by_infoset.items():
        by_class.setdefault(
            symmetry.class_key(info_set, classes, short_name), []
        ).append(tv)
    class_means = [sum(tvs) / len(tvs) for tvs in by_class.values()]
    return {
        "tv_by_infoset": tv_by_infoset,
        "visits_by_infoset": visits_by_infoset,
        "tv_mean": (
            (sum(class_means) / len(class_means)) if class_means else float("nan")
        ),
        "tv_mean_logged": (sum(values) / len(values)) if values else float("nan"),
        "n_symmetry_classes": len(class_means),
        "tv_mean_visit_weighted": (
            (weighted_num / weighted_den) if weighted_den else float("nan")
        ),
        "n_offpath_infosets": len(values),
        "n_rows_scored": n_rows,
        # Rows at a scored info-set whose `parsed_action` was null, so they are
        # Not in `n_rows_scored`. Present so a refusal-logging path cannot shrink the denominator
        # invisibly.
        "n_null_dropped": n_null_dropped,
        "game": game_key,
        "actor_filter": actor_label(actor, LEGACY_PROMPT),
        "dropped_rate": report.rate,
        "dropped_detail": report.detail(),
    }


def score_property6_exact(
    exact_policy: dict,
    df: pl.DataFrame,
    tremble_policy,
    game: Optional[pyspiel.Game] = None,
    actor: Optional[str] = None,
) -> dict:
    """
    P6 from the model's exact per-info-set policy rather than from its sampled
    actions, with the analytic uniform baseline over the same info-sets
    The estimand is the policy, and the sampled action is an estimator of it: at
    temperature 0 the sampled policy is a point mass at every node while the exact
    policy is mixed nearly everywhere, so a sampled distance of 0 or 1 per node
    misstates the policy distance and a temperature-0 board and a temperature-1
    board do not measure the same thing. The walk, the fallback drop, the refusals
    and the visit weights are :func:`score_property6`'s own; only the per-info-set
    policy differs.
    :param exact_policy: `{(game, info_set): {action: probability}}`, the exact
        policy pass's own distributions, as `analyse_stats.load_exact_policy`
        returns them; an info-set the pass does not cover is reported, never
        treated as an empty distribution (dict)
    :param df: one driven arm's rows, `probe_type="offpath_drive"` on every row
        (pl.DataFrame)
    :param tremble_policy: the reference from :func:`offpath_reference` (open_spiel Policy)
    :param game: the OpenSpiel game the frame was collected on, defaulting to the
        registry game named by the frame's own `game` column (pyspiel.Game | None)
    :param actor: which agent's rows to score; None keeps the legacy
        prompt-not-null population, which is the LM's rows (str | None)
    :returns: dict with keys
        "tv_by_infoset" ({info_set: exact tv} at the covered driven info-sets),
        "visits_by_infoset" ({info_set: n}, the LM's own visit counts, which are
        the weights both this distance and the baseline are averaged with),
        "tv_mean_visit_weighted" (float, the headline: the baseline is
        visit-weighted, and an unweighted distance beside a weighted baseline
        would compare two different populations),
        "tv_mean" (float, the unweighted mean, reported beside it),
        "baseline" (float, the visit-weighted mean of the uniform-random policy's
        own distance over the same info-sets with the same weights),
        "baseline_by_infoset" ({info_set: the uniform distance there}),
        "n_offpath_infosets" (int, the info-sets scored),
        "n_infosets_reached" (int, the driven info-sets the LM reached, covered or
        not, which is the denominator of "coverage"),
        "coverage" (float, scored over reached; NaN when the LM reached none),
        "missing_infosets" (list[str], the driven info-sets the pass does not
        cover, named rather than counted),
        "n_rows_scored" (int, rows behind the weights),
        "n_null_dropped" (int),
        "game" (the frame's game key),
        "actor_filter" (the population every number was computed on),
        "dropped_rate" (float | None) and "dropped_detail" (the counts behind it)
    """
    entries, reference, report, game_key, _game = _walk(
        df, tremble_policy, game, actor, "score_property6_exact"
    )

    tv_by_infoset: dict[str, float] = {}
    visits_by_infoset: dict[str, int] = {}
    scored_keys: list[tuple[str, int]] = []
    missing: list[str] = []
    weighted_num = 0.0
    weighted_den = 0
    n_rows = 0
    n_null_dropped = 0

    for info_set, player, ref, _empirical_policy, n, n_null in entries:
        exact = exact_policy.get((game_key, info_set))
        if exact is None:
            # Named, never scored as an empty distribution: an absent entry means
            # the pass did not cover this info-set, and an empty dict would read
            # as a policy at total-variation distance 1 from the reference.
            missing.append(info_set)
            continue
        n_null_dropped += n_null
        tv_by_infoset[info_set] = _total_variation(exact, ref)
        visits_by_infoset[info_set] = n
        scored_keys.append((info_set, player))
        weighted_num += tv_by_infoset[info_set] * n
        weighted_den += n
        n_rows += n

    baseline = uniform_baseline(reference, scored_keys, visits_by_infoset)
    values = list(tv_by_infoset.values())
    n_reached = len(entries)
    return {
        "tv_by_infoset": tv_by_infoset,
        "visits_by_infoset": visits_by_infoset,
        "tv_mean_visit_weighted": (
            (weighted_num / weighted_den) if weighted_den else float("nan")
        ),
        "tv_mean": (sum(values) / len(values)) if values else float("nan"),
        "baseline": baseline["value"],
        "baseline_by_infoset": baseline["by_infoset"],
        "n_offpath_infosets": len(values),
        "n_infosets_reached": n_reached,
        "coverage": (len(values) / n_reached) if n_reached else float("nan"),
        "missing_infosets": missing,
        "n_rows_scored": n_rows,
        "n_null_dropped": n_null_dropped,
        "game": game_key,
        "actor_filter": actor_label(actor, LEGACY_PROMPT),
        "dropped_rate": report.rate,
        "dropped_detail": report.detail(),
    }
