"""Paired probe scoring: P1 (label invariance) and P4 (payoff-scale invariance).

The estimand both properties share is a policy difference between two arms of
the same matrix that differ in one rendered-prompt manipulation and nothing
else. Base rows and probe rows are joined on (game, canonical info-set), and
each paired info-set gets the total-variation distance between the two arms'
empirical action distributions there. Zero means the manipulation moved
nothing, which is what P1 and P4 assert should happen.

Why the join needs no seat key. Under every seat mode a probe pair plays the same seat sequence hand for
hand, for one of two reasons. Under ``random`` the pair shares its matchup seed
and the per-hand seat draw is a pinned pure function of that seed
(``orchestration/shard.py``'s STREAM_SEAT), so the draws coincide. Under
``alternate`` -- the mode the tier-1 recycle-v4 corpus is collected in -- the
seat is ``hand_idx % 2``, which draws nothing at all and is therefore shared by
construction, independent of seed and of probe labels; the fixed modes are
constant.
Seat is still carried on the row (``lm_seat``) for analysis.

Why the join key is the OpenSpiel info-set string. A relabel variant
renames what the prompt offers, never what the engine calls the information
state; the payoff-scale variant changes prose only. Asserted directly, over
the full kuhn tree, in ``tests/test_property_paired.py``.

What this module does not do. It never asserts that the two arms differ:
``score_paired`` is deliberately usable on an arm against itself (the exact
zero is the falsification anchor for everything else here). The assertions
that a pair is a real P1 or P4 pair -- base arm canonical, probe arm carrying
this property's probe type and its registry variant name -- live in
``score_property1`` / ``score_property4``, which is where a runner that
stamped the variant on the base rows too gets caught.

Scoring is offline: pure functions over logged frames plus the
variant registry.
"""

from __future__ import annotations

import polars as pl

from lmopt.prompting.probe_variants import (
    BASE_VARIANT_NAME,
    GAME_KEYS,
    coverage_rows,
    expected_variant_name,
)
from lmopt.scoring._actor import (
    LEGACY_PROMPT,
    DropReport,
    actor_label,
    actor_rows,
    drop_fallback_rows,
    require_actor_column,
)
from lmopt.scoring._belief_alignment import tvd

#: Per-game row states. A game is `measured` when the pair shares at least one
#: info-set, `awaiting-data` when the variant exists but an arm has no LM rows from the
#: scored actor for the boards the base arm played (collection is what is missing), `not-scored` when both arms have rows and
#: still no info-set paired (collection already happened, so more of it is not the fix),
#: and `recorded-absence` when the registry says no variant can exist. The four are
#: distinct on purpose: "not collected" and
#: "cannot be collected" are different facts about a property's coverage, and
#: collapsing them is how a table reports a property measurable on the strength
#: of the games that happen to have a variant. Prompt version 7 left no recorded
#: absence in the registry, so the third state is currently unreachable from a
#: real run; it stays because the distinction is the reason this field exists,
#: and a withdrawn variant must land on it rather than read as uncollected.
STATUS_MEASURED = "measured"
STATUS_AWAITING_DATA = "awaiting-data"
STATUS_RECORDED_ABSENCE = "recorded-absence"
STATUS_NOT_SCORED = "not-scored"

_REQUIRED_COLUMNS = (
    "game",
    "info_set",
    "parsed_action",
    "prompt_variant",
    "probe_type",
)


def _arm_label(df: pl.DataFrame, column: str, where: str) -> str:
    """The single value of a run-level label column across one arm.

    Raises on more than one value: an arm is one rendering, and taking the
    first row's value would let a base arm concatenated with a probe arm score
    as either. The column's presence is guaranteed by ``score_paired``'s
    ``_REQUIRED_COLUMNS`` gate, which is the only caller; a second absence
    check here would be dead code, because the gate raises first.
    """
    values = sorted(set(df[column].drop_nulls().to_list()))
    if len(values) != 1:
        raise ValueError(
            f"{where}: expected exactly one {column} in an arm, found {values}. "
            f"One arm is one rendering; score the arms separately."
        )
    return values[0]


def _empirical(rows: pl.DataFrame) -> tuple[dict[int, float], int, int]:
    """
    (action -> frequency, n, n_null_dropped) at one info-set.
    No smoothing: TV is a distance between the distributions actually
    observed, and smoothing would make the number depend on the action count
    as well as on the policy. `n` rides with the distribution because TV
    between two multinomials is upward-biased at finite n, so the number is
    uninterpretable without its sample size.

    Nulls are dropped rather than counted: a null `parsed_action` is not a
    choice, and a None key would take mass from every real action.

    The drop is counted. Returns a third
    element `n_null_dropped` so a caller can distinguish 40 of 40 from 40 of 400
    (an all-null info-set from a genuinely unvisited one). The reachability
    condition that makes this live: any nullable `parsed_action` path (a refusal
    logged as a row, a partial parse). Zero such rows exist in any collected
    arm, so the count is always 0 on collected data, but the count itself is
    recorded.

    The third element reaches a
    consumer: `score_paired` sums it per game into `n_null_dropped_by_game`,
    `_per_game_rows` pools it onto each family row, and the acceptance table
    prints it beside `n_base`/`n_probe`. A call site that unpacked it and threw
    it away would reintroduce the defect the count exists to fix, one layer out.
    """
    col = rows["parsed_action"]
    n_null = col.null_count()

    counts: dict[int, int] = {}
    for action in col.drop_nulls().to_list():
        counts[action] = counts.get(action, 0) + 1
    total = sum(counts.values())
    if not total:
        return {}, 0, n_null
    return {a: n / total for a, n in counts.items()}, total, n_null


def _total_variation(p: dict[int, float], q: dict[int, float]) -> float:
    """Half the L1 distance over the union of the two supports.

    The union, not the intersection: an action the probe arm took and the base
    arm never did is exactly the kind of movement P1 and P4 are looking for,
    and intersecting would score it as no movement at all.

    Delegates to the shared canonical `tvd` (also union-over-supports); its
    numpy sum agrees with a python sequential sum within 2 ulp
    on adversarial inputs and bit-identically on the paired golden
    fixtures.
    """
    return tvd(p, q)


def _pool_drop_details(base: DropReport, probe: DropReport) -> dict:
    """
    pool one game's two arms' fallback-drop counts into a single detail dict
    The pooled counts sit at the top level in the `DropReport.detail()` key set,
    and each arm's own `detail()` rides beside them so neither arm's rate can
    hide behind the other's.
    :param base: the base arm's drop report for the game (DropReport)
    :param probe: the probe arm's drop report for the game (DropReport)
    :returns: dict with the `detail()` keys pooled by summing counts over both
        arms (unit "rows"; `fallback_row_rate` is pooled n_fallback_rows over
        pooled n_lm_rows; `n_infosets` and `n_infosets_fully_dropped` are the
        per-arm counts summed), plus "base" and "probe" holding each arm's own
        `detail()`
    """
    base_detail, probe_detail = base.detail(), probe.detail()
    # Paired reports are built from LM rows, so the LM-side counts are set (None is the belief side).
    assert (
        base.n_fallback_rows is not None
        and probe.n_fallback_rows is not None
        and base.n_lm_rows is not None
        and probe.n_lm_rows is not None
    )
    n_fallback = base.n_fallback_rows + probe.n_fallback_rows
    n_lm = base.n_lm_rows + probe.n_lm_rows
    def _summed(key: str) -> int | None:
        if base_detail[key] is None or probe_detail[key] is None:
            return None
        return base_detail[key] + probe_detail[key]
    return {
        "unit": "rows",
        "n_dropped": base.n_dropped + probe.n_dropped,
        "n_total": base.n_total + probe.n_total,
        "n_fallback_rows": n_fallback,
        "n_lm_rows": n_lm,
        "fallback_row_rate": (n_fallback / n_lm) if n_lm else None,
        "n_infosets": _summed("n_infosets"),
        "n_infosets_fully_dropped": _summed("n_infosets_fully_dropped"),
        "base": base_detail,
        "probe": probe_detail,
    }


def score_paired(
    base_df: pl.DataFrame,
    probe_df: pl.DataFrame,
    actor: str | None = None,
) -> dict:
    """
    per-info-set total-variation distance between a base arm and a probe arm,
    computed after dropping each arm's own harness-fallback rows
    A fallback row is a harness substitution, not the model's answer, so it is
    removed from each arm's game frame before any distribution is built and the
    dropped share is reported beside the distance instead of refusing the game.
    Only rows with a non-null `prompt` are dropped, so a Bayesian or CFR-reference
    actor's rows pass through. An info-set left with no kept row in the base arm
    leaves the base set (it is counted in `n_infosets_fully_dropped`, not in
    `missing_infosets`); one left with no kept row in the probe arm is a base
    info-set the probe never reached, so it is named in `missing_infosets`.
    :param base_df: the base arm's rows (one Parquet, or several concatenated) (pl.DataFrame)
    :param probe_df: the probe arm's rows, same matrix and seeds (pl.DataFrame)
    :param actor: which agent's rows to score, on the `actor` column; `None`
        keeps the legacy prompt-not-null population, which is the LM's rows and
        the only thing a pre-actor arm can express (str | None)
    :returns: dict with keys:
        "tv_by_infoset": {game: {info_set: tv}} over paired info-sets only.
        "visits_by_infoset": {game: {info_set: {"n_base", "n_probe"}}} - the
            sample size behind each TV. TV between two multinomials is
            upward-biased at finite n, so a per-info-set distance without its
            counts cannot be told from sampling noise, and a consumer needs
            these to compute a null band.
        "tv_mean": float - mean TV over every paired info-set, NaN when none
            paired (0.0 would read as "the policy did not move"). The
            headline: the estimand treats each info-set as one comparison
            unit, so a few high-traffic info-sets must not speak for the tree.
        "tv_mean_visit_weighted": float - the same distances weighted by
            (n_base + n_probe). Secondary, reported because the two answer
            different questions: this one is closer to "how much did the
            realised play move", the unweighted one to "how many decision
            situations moved".
        "tv_mean_by_game": {game: float}.
        "n_null_dropped_by_game": {game: {"n_base", "n_probe"}} - how many rows
            at paired info-sets carried a null `parsed_action` and were therefore
            excluded from the frequencies and from `n_base`/`n_probe`. Zero on
            every collected arm; emitted so a future refusal-logging path cannot
            shrink a sample size invisibly. Unpaired base info-sets are named in
            `missing_infosets` and are not in this count.
        "paired_coverage": float - paired info-sets / base-arm info-sets.
        "coverage_by_game": {game: float}.
        "missing_infosets": {game: [info_set]} - the base info-sets the probe
            arm never reached, named rather than counted.
        "n_infosets_base" / "n_infosets_paired": int.
        "base_variant" / "probe_variant" / "probe_type": the arms' own labels,
            reported rather than assumed.
        "arms_differ": bool - whether the two arms carry different variants at
            all. False on a self-pair, whose 0.0 is an identity rather than a
            measurement.
        "actor_filter": str - the population every number above was computed
            on. A filter that is not reported is a silent fallback.
        "dropped_rate": {game: float | None} - fallback rows dropped from both
            arms over LM rows in both arms, pooled by counts (never an average
            of the two arms' rates); None where neither arm had an LM row.
        "probe_only_games": list[str]: games the probe arm played and the
            base arm did not; they are scored nowhere else, so they are named.
        "dropped_detail": {game: dict} - the counts behind that rate. The
            top level carries the pooled `DropReport.detail()` keys (unit
            "rows", n_dropped, n_total, n_fallback_rows, n_lm_rows,
            fallback_row_rate, n_infosets, n_infosets_fully_dropped, the last
            two summed over the arms), and "base" and "probe" hold each arm's
            own `detail()`.
        "per_game": list[dict] - one row per registered game, carrying the
            registry's recorded absence/caveat for the probe type alongside the
            measurement, so an acceptance table consumes data
            rather than prose. Each row also carries `dropped_rate`,
            `n_dropped` and `n_total`, pooled by counts over the family's frame
            keys and None where the family contributed no frame. A game that
            pairs no info-set reads `awaiting-data` when an arm has no LM rows
            to score, and `not-scored` when both arms have rows; the `reason`
            of a `not-scored` row states the counted cause.
    """
    require_actor_column(base_df, actor, "score_paired")
    require_actor_column(probe_df, actor, "score_paired")
    for name, df in (("base_df", base_df), ("probe_df", probe_df)):
        absent = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if absent:
            raise ValueError(
                f"score_paired: {name} is missing column(s) {absent}. The pairing "
                f"is defined by (game, info_set, prompt_variant) and cannot be "
                f"reconstructed from a frame that lacks them."
            )

    base_variant = _arm_label(base_df, "prompt_variant", "score_paired(base_df)")
    probe_variant = _arm_label(probe_df, "prompt_variant", "score_paired(probe_df)")
    probe_type = _arm_label(probe_df, "probe_type", "score_paired(probe_df)")

    legacy = pl.col("prompt").is_not_null()
    base_rows = actor_rows(base_df, actor, legacy=legacy)
    probe_rows = actor_rows(probe_df, actor, legacy=legacy)

    tv_by_infoset: dict[str, dict[str, float]] = {}
    visits_by_infoset: dict[str, dict[str, dict[str, int]]] = {}
    null_dropped_by_game: dict[str, dict[str, int]] = {}
    tv_mean_by_game: dict[str, float] = {}
    coverage_by_game: dict[str, float] = {}
    missing_infosets: dict[str, list[str]] = {}
    n_base_by_game: dict[str, int] = {}
    n_base_total = 0
    n_paired_total = 0
    all_tvs: list[float] = []
    weighted_num = 0.0
    weighted_den = 0
    dropped_rate: dict[str, float | None] = {}
    dropped_detail: dict[str, dict] = {}

    for game_name in sorted(base_rows["game"].unique().to_list()):
        base_report = drop_fallback_rows(base_rows.filter(pl.col("game") == game_name))
        probe_report = drop_fallback_rows(
            probe_rows.filter(pl.col("game") == game_name)
        )
        base_game = base_report.kept
        probe_game = probe_report.kept
        detail = _pool_drop_details(base_report, probe_report)
        dropped_detail[game_name] = detail
        dropped_rate[game_name] = (
            detail["n_dropped"] / detail["n_total"] if detail["n_total"] else None
        )

        probe_infosets = set(probe_game["info_set"].unique().to_list())

        game_tvs: dict[str, float] = {}
        game_visits: dict[str, dict[str, int]] = {}
        game_nulls: dict[str, int] = {"n_base": 0, "n_probe": 0}
        missing: list[str] = []
        base_infosets = sorted(base_game["info_set"].unique().to_list())
        for info_set in base_infosets:
            if info_set not in probe_infosets:
                # Named, not dropped: a silent inner join would report full
                # coverage on a probe arm that never reached half the tree.
                missing.append(info_set)
                continue
            p, n_p, null_p = _empirical(
                base_game.filter(pl.col("info_set") == info_set)
            )
            q, n_q, null_q = _empirical(
                probe_game.filter(pl.col("info_set") == info_set)
            )
            # The counted null drops are summed per
            # game here and pooled onto the per-game rows below, so a nonzero
            # count reaches a reader instead of dying in this unpacking. Scope,
            # which the number needs stated beside it: these are the nulls at
            # Paired info-sets only -- an unpaired base info-set is named into
            # `missing` above and never reaches `_empirical`, so its nulls are
            # not in this total.
            game_nulls["n_base"] += null_p
            game_nulls["n_probe"] += null_q
            game_tvs[info_set] = _total_variation(p, q)
            game_visits[info_set] = {"n_base": n_p, "n_probe": n_q}
            # Both arms' visits, because the estimand is symmetric between
            # them and both carry evidence about this info-set. The raw counts
            # are emitted too, so any other weighting is recomputable offline.
            weighted_num += game_tvs[info_set] * (n_p + n_q)
            weighted_den += n_p + n_q

        n_base = len(base_infosets)
        n_paired = len(game_tvs)
        n_base_total += n_base
        n_paired_total += n_paired
        tv_by_infoset[game_name] = game_tvs
        visits_by_infoset[game_name] = game_visits
        null_dropped_by_game[game_name] = game_nulls
        missing_infosets[game_name] = missing
        n_base_by_game[game_name] = n_base
        coverage_by_game[game_name] = (n_paired / n_base) if n_base else 0.0
        tv_mean_by_game[game_name] = (
            sum(game_tvs.values()) / n_paired if n_paired else float("nan")
        )
        all_tvs.extend(game_tvs.values())

    out = {
        "tv_by_infoset": tv_by_infoset,
        "visits_by_infoset": visits_by_infoset,
        "tv_mean": (sum(all_tvs) / len(all_tvs)) if all_tvs else float("nan"),
        # Secondary, never the headline: see the module docstring's note on
        # what each estimator answers.
        "tv_mean_visit_weighted": (
            (weighted_num / weighted_den) if weighted_den else float("nan")
        ),
        "tv_mean_by_game": tv_mean_by_game,
        # The counted null-`parsed_action` drops behind every distribution above
        # Present and all-zero on every collected arm; a nonzero
        # entry means some of the rows at a paired info-set carried no action, so
        # its `n_base`/`n_probe` is smaller than the rows observed there.
        "n_null_dropped_by_game": null_dropped_by_game,
        "paired_coverage": (n_paired_total / n_base_total) if n_base_total else 0.0,
        "coverage_by_game": coverage_by_game,
        # The denominator behind `coverage_by_game`, emitted so a consumer that
        # aggregates several frame game keys into one registry family (see
        # `_per_game_rows`) can pool the fractions correctly instead of
        # averaging averages.
        "n_infosets_base_by_game": n_base_by_game,
        "missing_infosets": missing_infosets,
        "n_infosets_base": n_base_total,
        "n_infosets_paired": n_paired_total,
        "base_variant": base_variant,
        "probe_variant": probe_variant,
        # Stated, not assumed: this function deliberately allows scoring an arm
        # against itself (the exact zero is the falsification anchor for every
        # other number here), so a caller that skips the P1/P4 wrappers must be
        # able to see that a 0.0 came from a self-pair rather than from an
        # invariant policy.
        "arms_differ": base_variant != probe_variant,
        "probe_type": probe_type,
        "actor_filter": actor_label(actor, LEGACY_PROMPT),
        "dropped_rate": dropped_rate,
        "dropped_detail": dropped_detail,
        # Games only the probe arm played: the loop above walks the base arm's games, so
        # these reach no other output and are named here instead.
        "probe_only_games": sorted(
            set(probe_rows["game"].unique().to_list())
            - set(base_rows["game"].unique().to_list())
        ),
    }
    out["per_game"] = _per_game_rows(out, probe_type)
    return out


def _unpaired_verdict(
    frame_keys: list[str], dropped_detail: dict
) -> tuple[str, str | None]:
    """
    the status and reason of an available game whose pair shares no info-set
    An arm with no LM rows from the scored actor for the boards the base arm played means
    collection is missing, so the game reads awaiting-data. When both arms have rows the cell is not-scored, and the reason states the
    counts, keeping apart an arm whose rows were all dropped as harness fallback and rows that
    remain but never meet.
    :param frame_keys: the frame game keys pooled into this game family (list[str])
    :param dropped_detail: the scorer's per-game drop counts, each with "base" and "probe" (dict)
    :returns: (status, reason); the reason is None for awaiting-data (tuple[str, str | None])
    """
    counts = {
        arm: (
            sum(dropped_detail[key][arm]["n_dropped"] for key in frame_keys),
            sum(dropped_detail[key][arm]["n_total"] for key in frame_keys),
        )
        for arm in ("base", "probe")
    }
    if any(total == 0 for _dropped, total in counts.values()):
        return STATUS_AWAITING_DATA, None
    fully_dropped = [
        arm for arm, (dropped, total) in counts.items() if dropped == total
    ]
    if fully_dropped:
        parts = " and ".join(
            f"all {counts[arm][1]:,} rows of the {arm} arm dropped as harness fallback"
            for arm in fully_dropped
        )
        return STATUS_NOT_SCORED, f"{parts}, so there is no paired row to score"
    (base_dropped, base_total), (probe_dropped, probe_total) = (
        counts["base"],
        counts["probe"],
    )
    return STATUS_NOT_SCORED, (
        f"rows remain in both arms (base {base_total - base_dropped:,} of {base_total:,}, "
        f"probe {probe_total - probe_dropped:,} of {probe_total:,} kept after the fallback "
        f"drop) but no info-set is present in both"
    )


def _per_game_rows(out: dict, probe_type: str) -> list[dict]:
    """One row per registered game family, measurement and registry record together.

    Built from ``coverage_rows()`` so the absence text, the `would_need` and
    the caveat cannot drift from the registry that owns them, and emitted for
    every game so a property cannot read as covered on the strength of the
    games that happen to have a variant. A probe type the registry does not
    describe (``none``, ``offpath_drive``) contributes no registry fields, only
    the measurement.

    **The row is keyed by family, and the measurement is pooled over every
    frame game key in that family** (`games/families.py`'s ``GAME_FAMILIES``).
    The registry is family-keyed (one relabel scheme serves every dark hex
    board) while the `game` column carries the registry key of the variant
    actually played, so a straight name match would report a measured depth
    variant as uncollected (a depth-variant relabel pair with 62 paired
    info-sets would read `awaiting-data` with `n_infosets_paired: 0`).

    Pooling is over the raw quantities, never over per-game summaries: the TV
    mean is the mean of the family's individual distances and the coverage is
    the family's paired count over its base count, so two boards of unequal
    size cannot weight each other by accident.

    The dropped-fallback counts are pooled the same way, by summing each frame
    key's `n_dropped` and `n_total` and dividing once, so a small contaminated
    board cannot outweigh a large clean one in the family's rate.
    """
    from lmopt.games.families import GAME_FAMILIES

    registry = {
        row["game"]: row for row in coverage_rows() if row["probe_type"] == probe_type
    }
    tv_by_infoset = out["tv_by_infoset"]
    visits_by_infoset = out["visits_by_infoset"]
    dropped_detail = out["dropped_detail"]
    frame_keys_by_family: dict[str, list[str]] = {}
    for frame_key in tv_by_infoset:
        family = GAME_FAMILIES.get(frame_key, frame_key)
        frame_keys_by_family.setdefault(family, []).append(frame_key)

    rows = []
    for game in GAME_KEYS:
        record = registry.get(
            game,
            {
                "game": game,
                "probe_type": probe_type,
                "available": True,
                "variant_name": None,
                "reason": None,
                "would_need": None,
                "caveat": None,
            },
        )
        frame_keys = sorted(frame_keys_by_family.get(game, []))
        tvs = [tv for key in frame_keys for tv in tv_by_infoset[key].values()]
        n_paired = len(tvs)
        n_base = sum(out["n_infosets_base_by_game"].get(key, 0) for key in frame_keys)
        missing = [
            info_set
            for key in frame_keys
            for info_set in out["missing_infosets"].get(key, [])
        ]
        # The row count behind this game's tv_mean, out of the same
        # visits_by_infoset the property-level `_visit_summary` already reads:
        # a TV without its n cannot be told from
        # sampling noise, per game as much as property-wide.
        visits = [
            counts
            for key in frame_keys
            for counts in visits_by_infoset.get(key, {}).values()
        ]
        n_visits_base = sum(v["n_base"] for v in visits)
        n_visits_probe = sum(v["n_probe"] for v in visits)
        # Pooled the same way as the visits, and reported beside them: the two
        # numbers are only interpretable together, since a null drop is exactly a
        # row that was observed and is not in n_visits_*.
        nulls = [out["n_null_dropped_by_game"].get(key, {}) for key in frame_keys]
        n_null_base = sum(v.get("n_base", 0) for v in nulls)
        n_null_probe = sum(v.get("n_probe", 0) for v in nulls)
        # Counts, never per-board rates: a family's rate is its total dropped
        # over its total LM rows, so the boards weight themselves by size.
        n_dropped = sum(dropped_detail[key]["n_dropped"] for key in frame_keys)
        n_total = sum(dropped_detail[key]["n_total"] for key in frame_keys)
        reason = None
        if not record["available"]:
            status = STATUS_RECORDED_ABSENCE
        elif n_paired:
            status = STATUS_MEASURED
        else:
            status, reason = _unpaired_verdict(frame_keys, dropped_detail)
        rows.append(
            {
                **record,
                "status": status,
                "reason": reason if reason is not None else record.get("reason"),
                # None (not NaN) when the family contributed no frame at all:
                # "no rows here" is not "rows here and nothing paired".
                "tv_mean": (
                    (sum(tvs) / n_paired)
                    if n_paired
                    else (float("nan") if frame_keys else None)
                ),
                "paired_coverage": (
                    (n_paired / n_base) if n_base else (0.0 if frame_keys else None)
                ),
                "n_infosets_paired": n_paired,
                "missing_infosets": missing if frame_keys else None,
                # Which registry keys this row pooled, so a reader of a multi-board
                # arm can tell one family row from the boards behind it.
                "frame_games": frame_keys,
                "n_visits_base": n_visits_base if frame_keys else None,
                "n_visits_probe": n_visits_probe if frame_keys else None,
                # Same None-vs-0 rule as the visits above: None = this family
                # contributed no frame at all, 0 = it did and nothing was dropped.
                "n_null_dropped_base": n_null_base if frame_keys else None,
                "n_null_dropped_probe": n_null_probe if frame_keys else None,
                # None = this family contributed no frame; a family with a frame
                # but no LM rows has counts of 0 and a rate that is undefined.
                "dropped_rate": (n_dropped / n_total) if n_total else None,
                "n_dropped": n_dropped if frame_keys else None,
                "n_total": n_total if frame_keys else None,
            }
        )
    return rows


def _score_property(
    base_df: pl.DataFrame,
    probe_df: pl.DataFrame,
    probe_type: str,
    actor: str | None,
    where: str,
) -> dict:
    """
    shared body of the P1/P4 entry points: assert the pair is that property's
    pair, then score it
    The three assertions are the ones a mislabelled collection would fail. The
    base arm must be canonical (a runner that stamped the variant on both arms
    leaves nothing for the difference to be attributed to); the probe arm must
    carry this property's probe type (a P4 arm scored as P1 measures the wrong
    manipulation); and the probe arm's variant name must be the registry's name
    for that probe type (the label and the rendering are one claim).
    :param base_df: the base arm's rows (pl.DataFrame)
    :param probe_df: the probe arm's rows (pl.DataFrame)
    :param probe_type: the probe type this property's pair must carry (str)
    :param actor: which agent's rows to score, or None for the legacy filter (str | None)
    :param where: the entry point's name, for error messages (str)
    :returns: `score_paired`'s output dict
    """
    out = score_paired(base_df, probe_df, actor=actor)
    if out["base_variant"] != BASE_VARIANT_NAME:
        raise ValueError(
            f"{where}: the base arm carries prompt_variant "
            f"{out['base_variant']!r}, expected {BASE_VARIANT_NAME!r}. A pair "
            f"whose base half is itself a probe has no canonical reference, so "
            f"any distance between the two is not this property's estimand."
        )
    if out["probe_type"] != probe_type:
        raise ValueError(
            f"{where}: the probe arm carries probe_type {out['probe_type']!r}, "
            f"expected {probe_type!r}."
        )
    expected = expected_variant_name(probe_type)
    if out["probe_variant"] != expected:
        raise ValueError(
            f"{where}: the probe arm carries prompt_variant "
            f"{out['probe_variant']!r}, expected {expected!r} for probe_type "
            f"{probe_type!r}."
        )
    return out


def score_property1(
    base_df: pl.DataFrame,
    probe_df: pl.DataFrame,
    actor: str | None = None,
) -> dict:
    """
    P1, label invariance: how far the policy moved when the action labels were
    renamed and nothing else was
    Coverage is 4 of 5 games (goofspiel's actions are semantically ordered bid
    numbers, so no presentation-only relabelling exists); the fifth game's row
    in ``per_game`` carries the registry's recorded reason. Each arm's own
    fallback rows are dropped before scoring and reported as `dropped_rate`.
    :param base_df: the canonical arm's rows (pl.DataFrame)
    :param probe_df: the relabel arm's rows (pl.DataFrame)
    :param actor: which agent's rows to score, or None for the legacy filter (str | None)
    :returns: `score_paired`'s output dict, including `dropped_rate` and `dropped_detail`
    """
    return _score_property(base_df, probe_df, "relabel", actor, "score_property1")


def score_property4(
    base_df: pl.DataFrame,
    probe_df: pl.DataFrame,
    actor: str | None = None,
) -> dict:
    """
    P4, payoff-scale invariance: how far the policy moved when the rendered
    stake magnitude was scaled and nothing else was
    Coverage is 1 of 5 games (only kuhn's prompt renders a payoff magnitude at
    all). The kuhn prompt states every chip magnitude and the variant scales
    all of them, so a model cannot read the probe arm as a pot-odds change
    rather than a change of units; the row reports whatever caveat the
    registry holds.
    Each arm's own fallback rows are dropped before scoring and reported as
    `dropped_rate`.
    :param base_df: the canonical arm's rows (pl.DataFrame)
    :param probe_df: the payoff-scale arm's rows (pl.DataFrame)
    :param actor: which agent's rows to score, or None for the legacy filter (str | None)
    :returns: `score_paired`'s output dict, including `dropped_rate` and `dropped_detail`
    """
    return _score_property(base_df, probe_df, "payoff_scale", actor, "score_property4")


def score_paired_exact(
    base_exact: dict[tuple[str, str], dict[int, float]] | None,
    probe_exact: dict[tuple[str, str], dict[int, float]] | None,
    base_df: pl.DataFrame,
    probe_df: pl.DataFrame,
    actor: str | None = None,
) -> dict:
    """
    the paired distance of a base arm and a probe arm taken on the two arms' exact per-info-set
    policies rather than on their sampled actions, over the same paired info-sets
    The estimand is the policy difference, and the sampled action is an estimator of it: at
    temperature 0 a sampled policy is a point mass at every node, so a sampled distance of 0 or 1
    per node misstates the policy distance. The pairing, the fallback drop, the visit counts and
    the coverage are `score_paired`'s own, taken from its output and never recomputed here; only
    the per-info-set distance is replaced, by the total-variation distance between the two arms'
    exact distributions at that info-set. An info-set either arm's pass does not cover is named,
    never scored as an empty distribution, which would read as a distance of 1 from every policy.
    The two arms' action keys are canonical action indices, so a relabel arm's distribution is
    comparable with its base arm's.
    :param base_exact: `{(game, info_set): {action: probability}}` for the base arm, as
        `analyse_stats.load_exact_policy` returns it; None is refused, because an arm with no pass has
        no distribution basis and the caller keeps its sampled headline (dict | None)
    :param probe_exact: the same for the probe arm (dict | None)
    :param base_df: the base arm's rows (pl.DataFrame)
    :param probe_df: the probe arm's rows, same matrix and seeds (pl.DataFrame)
    :param actor: which agent's rows to score, as `score_paired` takes it (str | None)
    :returns: dict with keys
        "paired": `score_paired`'s whole output, unchanged, which is the sampled basis beside this one.
        "tv_exact_by_infoset": {game: {info_set: exact tv}} over the paired info-sets both passes cover.
        "tv_mean_exact": float, mean over those info-sets, NaN when none (0.0 would read as "did not move").
        "tv_mean_visit_weighted_exact": float, the same distances weighted by `paired`'s own
            n_base + n_probe at each info-set.
        "tv_mean_exact_by_game": {game: float}.
        "exact_missing_infosets": {game: {"base": [info_set], "probe": [info_set]}}, the paired
            info-sets each pass does not cover.
        "n_infosets_exact" / "n_infosets_paired": int.
        "exact_coverage": float, exact-scored over paired info-sets, NaN when nothing paired.
        "exact_coverage_by_game": {game: float}.
        "exact_pass_infosets": {"base": int, "probe": int}, how many (game, info_set) keys each arm's
            pass holds, so a pass that ran and covered nothing (0) is told apart from every paired
            info-set being named missing for another reason; an arm with no pass at all is refused
            above and never reaches here.
        "per_game_exact": list[dict], one row per registered family in `paired["per_game"]` order,
            with `game`, `frame_games`, `n_infosets_exact`, `n_infosets_paired`, `tv_mean_exact`,
            `exact_coverage` and `n_missing_base` / `n_missing_probe`, pooled over the family's frame
            keys by counts and never by averaging averages.
    :raises ValueError: when either exact policy is None
    """
    if base_exact is None or probe_exact is None:
        raise ValueError(
            "score_paired_exact: both arms need an exact policy; an arm with no pass has no "
            "distribution basis and keeps its sampled headline, which the caller decides"
        )
    paired = score_paired(base_df, probe_df, actor=actor)
    tv_exact: dict[str, dict[str, float]] = {}
    missing: dict[str, dict[str, list[str]]] = {}
    mean_by_game: dict[str, float] = {}
    coverage_by_game: dict[str, float] = {}
    weighted_num = 0.0
    weighted_den = 0
    n_exact_total = 0
    n_paired_total = 0
    for game, game_tvs in paired["tv_by_infoset"].items():
        game_exact: dict[str, float] = {}
        game_missing: dict[str, list[str]] = {"base": [], "probe": []}
        for info_set in game_tvs:
            base_dist = base_exact.get((game, info_set))
            probe_dist = probe_exact.get((game, info_set))
            if base_dist is None:
                game_missing["base"].append(info_set)
            if probe_dist is None:
                game_missing["probe"].append(info_set)
            if base_dist is None or probe_dist is None:
                continue
            distance = tvd(base_dist, probe_dist)
            game_exact[info_set] = distance
            visits = paired["visits_by_infoset"][game][info_set]
            weight = visits["n_base"] + visits["n_probe"]
            weighted_num += distance * weight
            weighted_den += weight
        tv_exact[game] = game_exact
        missing[game] = game_missing
        mean_by_game[game] = (
            sum(game_exact.values()) / len(game_exact) if game_exact else float("nan")
        )
        coverage_by_game[game] = (
            len(game_exact) / len(game_tvs) if game_tvs else float("nan")
        )
        n_exact_total += len(game_exact)
        n_paired_total += len(game_tvs)
    all_values = [v for game_exact in tv_exact.values() for v in game_exact.values()]
    per_game_exact = []
    for row in paired["per_game"]:
        frames = row["frame_games"]
        values = [v for key in frames for v in tv_exact.get(key, {}).values()]
        n_paired = sum(len(paired["tv_by_infoset"].get(key, {})) for key in frames)
        per_game_exact.append(
            {
                "game": row["game"],
                "frame_games": frames,
                "n_infosets_exact": len(values),
                "n_infosets_paired": n_paired,
                "tv_mean_exact": (
                    (sum(values) / len(values)) if values else float("nan")
                ),
                "exact_coverage": (
                    (len(values) / n_paired) if n_paired else float("nan")
                ),
                "n_missing_base": sum(
                    len(missing.get(key, {}).get("base", [])) for key in frames
                ),
                "n_missing_probe": sum(
                    len(missing.get(key, {}).get("probe", [])) for key in frames
                ),
            }
        )
    return {
        "paired": paired,
        "tv_exact_by_infoset": tv_exact,
        "tv_mean_exact": (
            (sum(all_values) / len(all_values)) if all_values else float("nan")
        ),
        "tv_mean_visit_weighted_exact": (
            (weighted_num / weighted_den) if weighted_den else float("nan")
        ),
        "tv_mean_exact_by_game": mean_by_game,
        "exact_missing_infosets": missing,
        "n_infosets_exact": n_exact_total,
        "n_infosets_paired": n_paired_total,
        "exact_coverage": (
            (n_exact_total / n_paired_total) if n_paired_total else float("nan")
        ),
        "exact_coverage_by_game": coverage_by_game,
        "exact_pass_infosets": {"base": len(base_exact), "probe": len(probe_exact)},
        "per_game_exact": per_game_exact,
    }
