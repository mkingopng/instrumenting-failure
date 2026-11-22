"""Property 3: Randomisation - regret of the action taken vs. CFR's own
equilibrium value function, per info-set.

P3 is scored as regret, not as KL(LM action distribution ‖ CFR Nash) per
info-set. At temperature 0 the LM's empirical policy is a point mass, so a
distribution-shape statistic (KL) confounds every cell -- a point mass cannot
express mixing entropy, so KL would measure the wrong thing. `score_property3`
scores `P3_regret(info_set) = Q(action_taken) - max_a Q(a)`, via OpenSpiel's own
`BestResponsePolicy.q_value`: `<= 0` always, `0` means the taken action was
equilibrium-optimal (every action in the Nash support scores 0, by the
indifference principle). `score_property3_exact` (the exact-logprob KL
diagnostic) stays a KL statistic -- it answers a different question
(exact vs. empirical distribution under the same KL estimand).
"""

from __future__ import annotations

import dataclasses
import json
import math

import polars as pl

from lmopt.scoring._actor import (
    LEGACY_PROMPT,
    actor_label,
    actor_rows,
    drop_fallback_rows,
    refuse_foreign_probe_type,
    require_actor_column,
)

#: Project-standard CFR+ iteration count `score_property3` solves/loads its
#: own reference at when no caller-supplied reference covers a game (mirrors
#: `run_sweep.py`'s `cfr_iters` default). `score_property3` accepts `games`/`cfr_policies`,
#: like `score_property2`/7/8, so a caller that solved its own reference at
#: the arm's own recorded `cfr_iters`/`cache_dir` can pass it in rather than
#: rely on this module-wide default matching by coincidence. Call sites that
#: call it as `score_property3(df, actor=...)` use this default, and an
#: internal `load_or_solve_cfr`, as the fallback for any game the caller did
#: not supply a policy for. On any arm already scored via
#: `run_sweep.py` at the same default, this is a cache HIT
#: (fast), not a fresh solve.
_DEFAULT_CFR_ITERS = 1000


@dataclasses.dataclass(frozen=True)
class KLScore:
    """A P3 score value inseparable from the reference coverage it was computed on.

    Deliberately not a float. The class keeps its KL name although
    `score_property3` scores regret -- both are floats paired with a coverage
    fraction, and every reader reads through the generic `.value`/
    `.on_support_fraction` fields, none inspecting the class name itself. At
    3x3 the reference is partial (measured 57.5 % off-support under
    uniform-random play), and a bare `mean_kl`/`value` reaches its consumers
    with no denominator. Encoding the gap as NaN does not work: every
    NaN path in this repo is a skip path, so it makes the parity gate exit 0
    with the false reason "not applicable". This type makes those reads
    raise (or, once taught, report coverage explicitly) until each is
    taught.
    """

    value: float
    on_support_fraction: float


def _compute_regret_table(
    game, cfr_policy, lm_seat: int, info_map: dict | None = None
) -> dict[str, tuple[float, dict[int, float]]]:
    """For every info-set `lm_seat` reaches, return
    `{info_set_str: (max_q, {action: q_value})}`.

    `BestResponsePolicy(game, lm_seat, cfr_policy)` computes, for player
    `lm_seat`, the value of best-responding to `cfr_policy` -- constructed
    the opposite way round from property2.py's own use of the identical
    class (there, the opponent best-responds to the LM's empirical policy;
    here, the LM's seat is the responder and CFR is the fixed policy, so
    `q_value(state, action)` gives the value to the LM'S seat of taking
    `action`, assuming optimal play thereafter against the equilibrium).
    Imported CFR/OpenSpiel, never reimplemented.

    Per info-set, not per state: `build_infoset_map` gives one representative
    `State` per (player, info_set_str) key, used here only to enumerate keys
    and read `legal_actions()` (invariant across every state sharing an
    info-set). The actual `q_value`s are pooled over every state sharing that
    info-set (`br.infosets[info_str]`, the identical counterfactual-reach-
    weighted list `BestResponsePolicy.best_response_action` itself pools over
    internally), not read off the one representative state directly.

    On solved Kuhn CFR+ (5000 iterations, nash_conv 5.5e-5), reading `q_value` off a single representative state gives, e.g.,
    player 0's Jack-opening info-set "0" a Bet q-value of -0.0000438 against
    a Check/Pass q-value of -1.0 for one underlying deal (opponent holds
    Queen) but -1.999999 / -1.0 for the other underlying deal (opponent holds
    King) -- a ~1.0 spread depending purely on which arbitrary state the
    traversal happened to visit first, which does not satisfy "every
    equilibrium-support action scores 0" for either deal alone. Pooling by
    counterfactual reach (this function's actual computation) gives -1.00007
    and -1.0 respectively -- within nash_conv-scaled tolerance of each other,
    as the indifference principle requires.

    `info_map`: an already-built `build_infoset_map(game)` result, passed in
    by callers that need multiple seats' tables for the same game (the map
    is seat-independent -- it enumerates every `(player, info_str)` key
    regardless of `lm_seat`, so building it once per game and reusing it
    across seats avoids a redundant full-tree traversal per seat). `None`
    (the default) builds it locally, for direct callers (this module's own
    tests call this function standalone).
    """
    from open_spiel.python.algorithms.best_response import BestResponsePolicy

    from lmopt.scoring._tree import build_infoset_map

    br = BestResponsePolicy(game, lm_seat, cfr_policy)
    if info_map is None:
        info_map = build_infoset_map(game)

    table: dict[str, tuple[float, dict[int, float]]] = {}
    for (player, info_str), state in info_map.items():
        if player != lm_seat:
            continue
        # `br.infosets` is keyed by every info_str `BestResponsePolicy`'s own
        # `info_sets()` reaches from `game.new_initial_state()` -- the same
        # unconditional, complete full-tree DFS `build_infoset_map` performs
        # (neither prunes on reach probability), so the two traversals visit
        # an identical key set by construction; this lookup cannot KeyError
        # in practice, but is not a proof, hence this note rather than a
        # silent assumption.
        infoset = br.infosets[info_str]
        total_cf = sum(cf for _, cf in infoset)
        if total_cf <= 0.0:
            # Every path to this info-set requires a zero-probability
            # transition under `cfr_policy` -- genuinely unreachable under
            # this reference, not merely under-visited by the logged data.
            # No well-defined pooled Q value here (no silent fallbacks: skip
            # rather than divide by zero); a row whose
            # info_set matches this key is left unmatched by the caller,
            # exactly like any other info-set absent from this table.
            continue
        q_values = {
            a: sum(cf * br.q_value(s, a) for s, cf in infoset) / total_cf
            for a in state.legal_actions()
        }
        table[info_str] = (max(q_values.values()), q_values)
    return table


def score_property3(
    df: pl.DataFrame,
    actor: str | None = None,
    games: dict | None = None,
    cfr_policies: dict | None = None,
    require_hit: bool = False,
) -> dict:
    """Compute P3 (randomisation, restated): the regret of the action taken
    relative to the best action at that info-set, under CFR's own
    equilibrium value function.

    Regret, not a mean-KL(LM action dist || CFR Nash) statistic: a
    point-mass policy at temperature 0 cannot express mixing entropy, so a
    distribution-shape metric confounds every cell; a value-based
    metric answers "how costly was the one action taken" instead, which a
    point mass can honestly answer.

    P3_regret(info_set) = Q(action_taken) - max_a Q(a) via OpenSpiel's own
    BestResponsePolicy.q_value (pooled per info-set, see
    `_compute_regret_table`); 0 means the taken action was
    equilibrium-optimal (every action in the Nash support scores 0 by the
    indifference principle -- this is why regret, not surprisal, is used:
    surprisal would wrongly penalise a low-probability but equally-good
    in-support action).

    The scored LM's own harness-fallback rows (`fallback_used` on a row with a
    `prompt`) are dropped per game before the off-support filter, so
    `on_support_fraction` and the regret mean are both over kept rows, and
    the drop is reported in `dropped_rate` and `dropped_detail`.
    :param df: full match DataFrame, all games and all players (pl.DataFrame)
    :param actor: the `actor` column value to score; `None` (the default)
        keeps the legacy population, every row carrying a `prompt`, which is
        the LM's rows and the only thing a legacy arm can express (str | None)
    :param games: {game_name: pyspiel.Game}, for a game whose policy the
        caller already solved; a game absent from this dict is loaded
        internally (dict | None)
    :param cfr_policies: {game_name: policy}, solved at the CALLER'S own
        recorded `cfr_iters`/`cache_dir`; a game absent from this dict falls
        back to `_DEFAULT_CFR_ITERS`/`policy_cache.DEFAULT_CFR_DIR`, loaded or
        solved internally (dict | None)
    :param require_hit: for the internal fallback only (a game with no
        supplied policy): refuse a cold cache instead of solving one, the
        same "no silent fallback" the P6/P8 acceptance rows already use
        (bool)
    :returns: dict with keys
        "per_info_set": {game: {info_set: regret}}, regret per info-set per
        game, `<= 0`.
        "mean_kl": {game: KLScore}, visit-weighted mean regret per game (name
        kept for call-site compatibility; holds regret, not KL, see this
        module's KLScore docstring), paired with on_support_fraction. A game
        whose rows were all dropped is present with a NaN value.
        "global_mean_kl": float, mean over fully-covered games only (name
        kept for call-site compatibility).
        "global_mean_kl_partial_games": list[str], games excluded from
        global_mean_kl because their reference coverage is < 100%.
        "actor_filter": str, which population produced every number above,
        `"actor=<name>"` or `"legacy-prompt-null"`. A filter that is not
        reported is a silent fallback.
        "dropped_rate": {game: float | None}, the fraction of the game's LM
        rows dropped as harness fallback; None when the game had no LM rows.
        "dropped_detail": {game: dict}, the `DropReport.detail()` counts
        behind each rate (unit "rows"), including the info-sets left with no
        kept row.
        "n_unmatched_rows": {game: int}, the on-support rows whose info_set
        joined no entry of the solved regret table (skipped, not in the
        mean). A game the scorer never reached has no entry: absent means
        not measured, 0 means measured and none.
        "n_null_action": {game: int}, the kept rows whose parsed_action is
        null (skipped, not in the mean), with the same absent-versus-0 reading.
    """
    require_actor_column(df, actor, "score_property3")
    refuse_foreign_probe_type(df, "score_property3")
    label = actor_label(actor, LEGACY_PROMPT)
    lm_rows = actor_rows(df, actor, legacy=pl.col("prompt").is_not_null())
    if lm_rows.is_empty():
        return {
            "per_info_set": {},
            "mean_kl": {},
            "global_mean_kl": float("nan"),
            "global_mean_kl_partial_games": [],
            "actor_filter": label,
            "dropped_rate": {},
            "dropped_detail": {},
            "n_unmatched_rows": {},
            "n_null_action": {},
        }

    import sys

    from lmopt.games import policy_cache
    from lmopt.games.depth_reference import DEPTH_REFERENCES
    from lmopt.games.loaders import GAMES, load_game
    from lmopt.scoring._tree import build_infoset_map

    per_info_set: dict[str, dict[str, float]] = {}
    mean_kl: dict[str, KLScore] = {}
    partial_games: list[str] = []
    n_unmatched_rows: dict[str, int] = {}
    n_null_action: dict[str, int] = {}
    dropped_rate: dict[str, float | None] = {}
    dropped_detail: dict[str, dict] = {}

    for game_name in lm_rows["game"].unique().to_list():
        if game_name not in GAMES:
            # No silent fallbacks, but also no unnecessary crash. A
            # synthetic/foreign game name -- e.g. a "toygame" test fixture, whose
            # module docstring states its whole design intent: games named
            # "toygame" so gate-mechanics tests stay estimand-agnostic and
            # Every property (P3 included) reports "unmeasurable" rather
            # than depending on real CFR machinery -- must score gracefully
            # rather than crash. The regret estimand needs a real registered
            # game to solve/load a CFR reference against, so an unregistered
            # name is an explicit, early-and-named skip instead of a
            # `KeyError` surfacing from deep inside `load_game`. Skipped
            # Before any other bookkeeping, same as the DEPTH_REFERENCES skip
            # below.
            print(
                f"score_property3: skipping {game_name!r} -- not a registered "
                f"game ({sorted(GAMES)}); no CFR reference can be solved/loaded "
                "for it, so this game is not scored (P3 was never gated on the "
                "game registry before this estimand; a synthetic/foreign game "
                "name reads as unmeasurable, not a crash).",
                file=sys.stderr,
            )
            continue
        if game_name in DEPTH_REFERENCES:
            # No silent fallbacks. `depth_reference.py`'s own
            # module docstring: "Every scoring-path dispatch site must
            # consult [DEPTH_REFERENCES] before calling load_or_solve_cfr."
            # darkhex_3x3's real reference is a PARTIAL depth-limited MCCFR
            # policy (22,701 of ~1.11M info-sets), not a full CFR+ solve --
            # `load_or_solve_cfr` would attempt exactly the full CFR+ solve
            # this project's own depth-reference machinery exists to avoid
            # for this game (Python CFR does not finish at dark_hex 3x3:
            # wall-time bound on the full-tree traversal, not memory). Even the PARTIAL reference would not
            # help here: `_compute_regret_table`'s `BestResponsePolicy`
            # construction is also an unconditional full-tree traversal
            # (property2.py's own documented finding, `_is_partial_reference`'s
            # docstring/usage) -- intractable at this depth regardless of
            # which policy backs it. Skipped before any other bookkeeping
            # (never added to `partial_games` either -- it was never
            # attempted, so "partially covered" would misdescribe it) and
            # Named (never a bare silent absence): this game contributes no
            # key to "per_info_set"/"mean_kl" at all, exactly like any other
            # skipped-and-reported condition in this codebase
            # (policy_cache.py's own corrupt-cache WARNING is the pattern),
            # since the return dict's shape is pinned unchanged by every
            # existing call site and cannot carry a new "not_applicable" key.
            print(
                f"score_property3: skipping {game_name!r} -- its reference is a "
                "PARTIAL depth-limited policy, and score_property3's full-tree "
                "regret walk (BestResponsePolicy) is intractable at this depth "
                "regardless of which policy backs it; not scored for this game.",
                file=sys.stderr,
            )
            continue

        game_df = lm_rows.filter(pl.col("game") == game_name)

        # A harness `legal[0]` substitution row is not the model's own answer,
        # so it never contributes a regret value. Dropped before the
        # off-support filter below, so a fallback row that is also
        # off-support is removed and counted here rather than vanishing
        # first, and `on_support_fraction` is over the kept rows.
        report = drop_fallback_rows(game_df)
        dropped_rate[game_name] = report.rate
        dropped_detail[game_name] = report.detail()
        game_df = report.kept

        # Off-support rows carry the library uniform, not the equilibrium: scoring
        # regret against them would silently measure regret vs. a reference the
        # row was never actually played against. Exclude them from the mean,
        # but count them, so the coverage is reported rather than the rows
        # silently vanishing (the survivorship bias a null-column encoding
        # would introduce).
        n_all = len(game_df)
        # Correspondence-CHECK: refcoverage-scorerassumptions (PARTIAL: P3 only)
        if "cfr_ref_off_support" in game_df.columns:
            game_df = game_df.filter(~pl.col("cfr_ref_off_support"))
        n_on = len(game_df)
        # 0.0 (not NaN) when n_all == 0: a NaN fraction is both `>= 1.0` False
        # and `< 1.0` False, so the game would fall out of both global_mean_kl
        # and global_mean_kl_partial_games -- the exact silent drop this type
        # exists to remove. Reached when every row of the game was dropped as
        # fallback (`dropped_rate` is then 1.0).
        on_support_fraction = (n_on / n_all) if n_all else 0.0
        if on_support_fraction < 1.0:
            partial_games.append(game_name)

        # Every row was dropped as fallback or filtered out by the
        # off-support exclusion above (n_on == 0): nothing here can be
        # scored, so skip the CFR load/solve entirely rather than paying a
        # potentially expensive solve for a cell that will report NaN
        # regardless.
        if game_df.is_empty():
            per_info_set[game_name] = {}
            mean_kl[game_name] = KLScore(
                value=float("nan"), on_support_fraction=on_support_fraction
            )
            continue

        game_obj = (
            games[game_name] if games and game_name in games else load_game(game_name)
        )
        if cfr_policies is not None and game_name in cfr_policies:
            cfr_policy = cfr_policies[game_name]
        else:
            cfr_policy = policy_cache.load_or_solve_cfr(
                game_obj,
                _DEFAULT_CFR_ITERS,
                plus=True,
                cache_dir=policy_cache.DEFAULT_CFR_DIR,
                require_hit=require_hit,
            )

        # The seat the scored agent occupied in a hand is `player_to_act` on
        # Its own rows (`game_df`, already actor-filtered): an agent has a
        # decision row only at the seat it sat in, which also makes this
        # correct on legacy arms with no seat column at all (every row is
        # then assumed seat 0, byte-identical to every arm collected before
        # per-hand seat randomisation existed). Not `lm_seat`
        # (logging/schema.py: "the seat AGENT0 occupies", the OPPONENT's seat
        # whenever the scored actor is agent1 -- property2.py's
        # score_property2 states and relies on the identical rule, for the
        # identical reason). A `seat_mode="random"`/"alternate" cell has the
        # scored agent at both seats across hands, so a regret table is built
        # per seat actually encountered, and each row is looked up in its
        # Own seat's table. Checked once (`has_seat_col`), not per row.
        has_seat_col = "player_to_act" in game_df.columns
        if has_seat_col:
            seats = sorted({int(s) for s in game_df["player_to_act"].to_list()})
        else:
            seats = [0]
        # `build_infoset_map` is seat-independent (it enumerates every
        # (player, info_str) key regardless of which seat is queried), so
        # it is built once per game and shared across every seat's table,
        # not re-walked per seat.
        info_map = build_infoset_map(game_obj)
        regret_tables = {
            seat: _compute_regret_table(game_obj, cfr_policy, seat, info_map=info_map)
            for seat in seats
        }

        # Visit-weighted mean per info-set, not last-row-wins: with a plain
        # `game_regrets[info_str] = regret`, two rows at the same info-set
        # with different actions would silently overwrite each other, making
        # the reported value order-dependent on `iter_rows`' iteration order.
        # Unmatched rows (info-set absent from the solved table) and
        # unrecognised actions (present at the info-set but not among its
        # legal actions) are counted and reported to stderr rather than
        # silently dropped or scored as regret 0.0 -- an unrecognised action
        # reading as equilibrium-optimal is exactly the flattering silent
        # fallback this module forbids.
        regret_sums: dict[str, float] = {}
        regret_counts: dict[str, int] = {}
        weighted_sum = 0.0
        n_weighted = 0
        n_unmatched_game_rows = 0
        n_unrecognised_action = 0
        n_null_action[game_name] = 0
        for row in game_df.iter_rows(named=True):
            info_str = row["info_set"]
            action = row["parsed_action"]
            if action is None:
                n_null_action[game_name] += 1
                continue
            seat = int(row["player_to_act"]) if has_seat_col else 0
            table = regret_tables.get(seat)
            if table is None or info_str not in table:
                n_unmatched_game_rows += 1
                continue
            max_q, q_values = table[info_str]
            if action not in q_values:
                n_unrecognised_action += 1
                continue
            regret = q_values[action] - max_q
            regret_sums[info_str] = regret_sums.get(info_str, 0.0) + regret
            regret_counts[info_str] = regret_counts.get(info_str, 0) + 1
            weighted_sum += regret
            n_weighted += 1

        n_unmatched_rows[game_name] = n_unmatched_game_rows
        if n_unmatched_game_rows:
            # The exact survivorship bias
            # `KLScore`'s own docstring exists to prevent: `on_support_fraction`
            # is derived purely from `cfr_ref_off_support` above and says
            # nothing about whether an on-support row's `info_set` actually
            # Joined against the solved regret table. Named and counted here
            # so a reader cannot mistake "1.0 coverage" for "every on-support
            # row was scored".
            print(
                f"score_property3: {game_name!r} has {n_unmatched_game_rows} "
                "row(s) whose info_set does not match any entry in the "
                "solved regret table (skipped, not counted in the mean); "
                "on_support_fraction above reports cfr_ref_off_support "
                "coverage only, not this join.",
                file=sys.stderr,
            )
        if n_unrecognised_action:
            print(
                f"score_property3: {game_name!r} has {n_unrecognised_action} "
                "row(s) whose parsed_action is not a legal action at its "
                "info_set (skipped, not counted in the mean) -- scoring it "
                "as regret 0.0 would silently read a bad parse/log as "
                "equilibrium-optimal play.",
                file=sys.stderr,
            )

        game_regrets = {
            info_str: regret_sums[info_str] / regret_counts[info_str]
            for info_str in regret_sums
        }
        per_info_set[game_name] = game_regrets
        mean_kl[game_name] = KLScore(
            value=(weighted_sum / n_weighted) if n_weighted else float("nan"),
            on_support_fraction=on_support_fraction,
        )

    # global_mean_kl averages fully-covered, NON-NaN games only; both kinds
    # of exclusion are named, never silently dropped -- partially-covered
    # games in `partial_games` (returned), and a fully-covered game that
    # still scored NaN (e.g. every one of its logged info-sets fails to
    # match any entry in the solved regret table, as in a pilot fixture;
    # consumers read this value directly as a headline number, so one
    # unscorable game must not poison every other game's genuine signal)
    # stays fully visible in
    # `mean_kl[game].value` itself, just excluded from this average.
    # Mirrors `score_property3_exact`'s own identical double-filter below.
    full = [
        s
        for s in mean_kl.values()
        if not math.isnan(s.value) and s.on_support_fraction >= 1.0
    ]
    global_mean = sum(s.value for s in full) / len(full) if full else float("nan")

    return {
        "per_info_set": per_info_set,
        "mean_kl": mean_kl,
        "global_mean_kl": global_mean,
        "global_mean_kl_partial_games": partial_games,
        "actor_filter": label,
        "dropped_rate": dropped_rate,
        "dropped_detail": dropped_detail,
        "n_unmatched_rows": n_unmatched_rows,
        "n_null_action": n_null_action,
    }


def score_property3_exact(
    df: pl.DataFrame,
    exact_policy: dict[tuple[str, str], dict[int, float]],
    actor: str | None = None,
) -> dict:
    """Mirror of `score_property3`, with the exact per-info-set action
    distribution (`scripts/exact_policy_pass.py`'s offline echo-logprob
    pass) replacing the Laplace-smoothed empirical distribution built from
    `parsed_action` counts. Same LM-row population, same
    `cfr_ref_off_support` exclusion, same visit-weighted mean - only the
    distribution changes, which is what makes the two numbers comparable
    side by side.

    `parsed_action` (`score_property3`) is the primary scored quantity.
    This is a diagnostic reported beside it, and there is deliberately no
    `DECISION_RULE` entry for it (`scripts/analyse_stats.py`).

    Scoring is offline: a pure function over its arguments - no
    file I/O here. `exact_policy` is `{(game, info_set): {action: prob}}`;
    `scripts/analyse_stats.py`'s `_load_exact_policy` builds it from
    `exact_policy.csv` (`scripts/exact_policy_pass.py`'s own output),
    including only info-sets whose full legal-action set scored with no
    `error` and a non-null `prob` - a partial/untrusted distribution is
    treated as absent, never used half-renormalised.

    The scored LM's own harness-fallback rows are dropped per game before this
    function's off-support filter, so the visit weights, the matched and
    unmatched info-sets and `on_support_fraction` are all over kept rows; an
    info-set left with no kept row appears in none of them.
    :param df: full match DataFrame, all games and all players, same as
        `score_property3` (pl.DataFrame)
    :param exact_policy: `{(game, info_set): {action: prob}}`, already
        renormalised over the info-set's legal actions (this function does not
        renormalise again) (dict)
    :param actor: as `score_property3`, the `actor` column value to score,
        `None` keeping the legacy prompt-null population (str | None)
    :returns: dict with keys
        "per_info_set": {game: {info_set: kl}}, KL(exact || CFR) per info-set
        per game, for matched info-sets only (an info-set with no
        `exact_policy` entry contributes no key here).
        "mean_kl": {game: KLScore}, visit-weighted mean KL over matched
        info-sets per game, paired with the same `on_support_fraction`
        definition `score_property3` uses (the on-support fraction of the
        population this scorer would otherwise visit, not of the matched
        subset; an unmatched info-set is a different kind of gap from an
        off-support one, and conflating the two denominators would hide
        either).
        "global_mean_kl": float, mean over fully on-support-covered games
        only, mirroring `score_property3` exactly (reference coverage, not
        exact-policy coverage).
        "global_mean_kl_partial_games": list[str].
        "matched_fraction": {game: float}, of the distinct info-sets this
        scorer would otherwise visit for that game (on-support kept LM rows),
        the fraction that had an `exact_policy` entry (the join succeeded).
        Unmatched info-sets are SKIPPED and counted, never silently dropped
        from the denominator; a mean over a silent subset is exactly the
        defect this guards against. "Had an entry" is not the
        same claim as "contributed a KL value": `n_matched` increments the
        instant `exact_policy.get((game, info_set))` succeeds, before this
        info-set's own `cfr_policy` JSON is parsed, so an info-set with a real
        `exact_policy` entry but an unparseable or empty `cfr_policy` (a
        defect on the CFR/reference side, orthogonal to the exact-policy
        join) still counts as matched while contributing no key to
        "per_info_set" and no weight to "mean_kl". Read "matched" as
        "joined", not as "used".
        "n_matched": {game: int}, numerator of "matched_fraction" (join
        successes, per the same caveat above).
        "n_unmatched": {game: int}, denominator - numerator, i.e. the count
        of visited info-sets with no `exact_policy` entry.
        "actor_filter": str, which population produced the numbers,
        `"actor=<name>"` or `"legacy-prompt-null"`.
        "dropped_rate": {game: float | None}, the fraction of the game's LM
        rows dropped as harness fallback; None when the game had no LM rows.
        "dropped_detail": {game: dict}, the `DropReport.detail()` counts
        behind each rate (unit "rows").

    Over the real qwen2.5-14b-instruct arm's 185
    distinct (game, info_set) exact-policy entries, all joined against the
    same arm's `data/run_experiment/qwen2.5-14b-instruct/*.parquet`: every
    one of the 185 joined cleanly against the parquet's `info_set` column
    (including Dark Hex's multiline, CSV-quoted board strings), every one
    agreed exactly on its action key set with the corresponding
    `cfr_policy`, and NONE produced `kl == inf` (no joined info-set has a
    CFR probability of exactly 0 against a positive exact probability).
    This docstring states
    the measurement rather than assuming it generalises: a future arm's
    exact-policy pass could still disagree on keys or hit `inf`, and both
    are handled explicitly above (key mismatch: `.get(a, 0.0)` on either
    side, mirroring `score_property3`'s own convention; `inf`: propagated,
    never silently clipped - the same "no silent fallbacks" a bare `kl =
    float("inf")` already enforces for `score_property3`).
    """
    require_actor_column(df, actor, "score_property3_exact")
    refuse_foreign_probe_type(df, "score_property3_exact")
    label = actor_label(actor, LEGACY_PROMPT)
    lm_rows = actor_rows(df, actor, legacy=pl.col("prompt").is_not_null())
    if lm_rows.is_empty():
        return {
            "per_info_set": {},
            "mean_kl": {},
            "global_mean_kl": float("nan"),
            "global_mean_kl_partial_games": [],
            "matched_fraction": {},
            "n_matched": {},
            "n_unmatched": {},
            "actor_filter": label,
            "dropped_rate": {},
            "dropped_detail": {},
        }

    per_info_set: dict[str, dict[str, float]] = {}
    mean_kl: dict[str, KLScore] = {}
    matched_fraction: dict[str, float] = {}
    n_matched: dict[str, int] = {}
    n_unmatched: dict[str, int] = {}
    dropped_rate: dict[str, float | None] = {}
    dropped_detail: dict[str, dict] = {}

    for game_name in lm_rows["game"].unique().to_list():
        game_df = lm_rows.filter(pl.col("game") == game_name)

        # Same drop, same ordering as score_property3: fallback rows leave
        # before the off-support filter, so every count below is over kept
        # rows.
        report = drop_fallback_rows(game_df)
        dropped_rate[game_name] = report.rate
        dropped_detail[game_name] = report.detail()
        game_df = report.kept

        # Same off-support exclusion as score_property3, for the same
        # reason: an off-support row carries the library uniform, not the
        # equilibrium.
        n_all = len(game_df)
        # Correspondence-CHECK: refcoverage-scorerassumptions (PARTIAL: P3 only)
        if "cfr_ref_off_support" in game_df.columns:
            game_df = game_df.filter(~pl.col("cfr_ref_off_support"))
        n_on = len(game_df)
        on_support_fraction = (n_on / n_all) if n_all else 0.0

        game_kls: dict[str, float] = {}
        n_m = 0
        n_u = 0

        for info_set in game_df["info_set"].unique().to_list():
            exact = exact_policy.get((game_name, info_set))
            if exact is None:
                # Skipped and counted - never silently dropped from the
                # denominator (see "matched_fraction" above).
                n_u += 1
                continue
            # Counts the join succeeding, not the KL
            # actually being computed -- an unparseable/empty cfr_policy
            # below still counts as matched (see "matched_fraction"'s
            # docstring note above).
            n_m += 1

            rows = game_df.filter(pl.col("info_set") == info_set)
            cfr_raw = rows["cfr_policy"][0]
            try:
                cfr_policy: dict[int, float] = {
                    int(k): float(v) for k, v in json.loads(cfr_raw).items()
                }
            except (json.JSONDecodeError, TypeError):
                continue

            legal_actions = sorted(cfr_policy.keys())
            if not legal_actions:
                continue

            # KL(p_exact ‖ p_cfr); skip an action p_exact assigns 0 to
            # (contributes nothing), go to +inf if CFR assigns 0 to an
            # action p_exact assigns positive mass to - identical rule to
            # score_property3's KL loop, just with `exact` in place of the
            # Laplace-smoothed `p_emp`.
            kl = 0.0
            for a in legal_actions:
                pe = exact.get(a, 0.0)
                pc = cfr_policy.get(a, 0.0)
                if pe <= 0.0:
                    continue
                if pc <= 0.0:
                    kl = float("inf")
                    break
                kl += pe * math.log(pe / pc)

            game_kls[info_set] = kl

        per_info_set[game_name] = game_kls
        n_matched[game_name] = n_m
        n_unmatched[game_name] = n_u
        n_visited = n_m + n_u
        matched_fraction[game_name] = (n_m / n_visited) if n_visited else float("nan")

        # Visit-weighted mean KL, over matched info-sets only (game_kls'
        # keys) - identical mechanism to score_property3.
        total_visits = 0
        weighted_sum = 0.0
        for info_set, kl in game_kls.items():
            visits = len(game_df.filter(pl.col("info_set") == info_set))
            total_visits += visits
            weighted_sum += kl * visits
        mean_kl[game_name] = KLScore(
            value=weighted_sum / total_visits if total_visits > 0 else float("nan"),
            on_support_fraction=on_support_fraction,
        )

    full = [
        s
        for s in mean_kl.values()
        if not math.isnan(s.value) and s.on_support_fraction >= 1.0
    ]
    partial = sorted(g for g, s in mean_kl.items() if s.on_support_fraction < 1.0)
    global_mean = sum(s.value for s in full) / len(full) if full else float("nan")

    return {
        "per_info_set": per_info_set,
        "mean_kl": mean_kl,
        "global_mean_kl": global_mean,
        "global_mean_kl_partial_games": partial,
        "matched_fraction": matched_fraction,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "actor_filter": label,
        "dropped_rate": dropped_rate,
        "dropped_detail": dropped_detail,
    }
