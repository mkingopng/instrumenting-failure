# src/lmopt/analysis/contrast/contrast_think.py
"""
Suite 1 of the contrast suites: instruct vs think.

Two readings, both pure functions over the logged Parquet plus the logged CFR
reference (`cfr_policy`), so scoring stays offline:

- `think_vs_instruct`: the think-vs-instruct contrast per (game, opponent) on
  paired arms, and pooled across the roster halves within one prompt epoch.
  The primary estimand is the CFR mass
  on the chosen action, `cfr_policy[parsed_action]` per decision node; mean
  reward per hand rides alongside and is never the headline.
- `trace_length_quality`: within think arms, quality per quartile bin of trace
  length in characters of `reasoning_trace`, which is the estimand by design.
  A token count is not the same measurement: `logging/schema.py` does log
  `completion_tokens`, but that counts the whole completion (trace plus the
  action the parser reads), so it cannot stand in for the length of the trace
  alone.

Four conventions this module is built around:

1. Sign convention. Every effect size is computed with `group_a="instruct"`
   and `group_b="think"`, so a positive value means instruct scored above
   think. That sentence travels in every row's `note`, refusal rows included,
   because a sign convention documented only in a docstring is a sign
   convention the reader of a CSV does not have.
2. Hand keys are per arm. `effect_sizes_ci` bootstraps through
   `resample_hands`, which keys a hand on `(matchup_id, game, seed, hand_idx)`.
   Two arms of the same roster share all four, so the combined frame rewrites
   `matchup_id` to `"<arm>:<matchup_id>"` before the call; without it one arm's
   hand and the other's would be resampled as a single hand.
3. No silent fallbacks. Absent or unmeasurable data is a row
   with a `status` naming the reason and null numbers, never a placeholder:
   `NOT_COMPARABLE (prompt epoch differs)`,
   `NOT_COMPARABLE (seat_mode differs)`, `PAIR_MISSING (<name> not loaded)`,
   `ROLE_UNSUPPORTED (think role, 0 traced rows)`,
   `NOT_PRESENT (no rows in either arm)`, `INSUFFICIENT (n<2 hands)`,
   `SPARSE (n<4, unbinned)`, `SPARSE (n<N)`, `UNDEFINED (...)`,
   `UNMEASURED (...)`.
4. A harness-fallback row is never the model's choice. The mean-reward
   cell drops every hand that holds a fallback row of the LM, the
   `cfr_mass_chosen` cell already leaves fallback rows unmeasured, and every
   row of a cell reports the count beside the number in the names the scorers
   use: `n_dropped`, `n_total`, `dropped_rate`, `dropped_unit`, with the rate
   again per side (`dropped_rate_instruct`, `dropped_rate_think`) because think
   arms are the ones that truncate.

`src/` never imports from `scripts/`, so the hand-level reward
reduction and the opponent parse are implemented here. The reduction flags
the hands holding a fallback row rather than dropping them.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, cast

import polars as pl

from lmopt.analysis.analysis_helpers import count_hands
from lmopt.analysis.contrast.contrast_arms import (
    ContrastArm,
    build_row,
    comparable,
    filter_excluded,
)
from lmopt.scoring.bootstrap import bootstrap_ci
from lmopt.scoring.robustness import effect_sizes_ci

TRACE_CAVEAT = "A trace is a self-report, not a mechanism."
SIGN_NOTE = "Group order is instruct then think, so positive means instruct > think."
ALL = "__all__"
STATUS_OK = "OK"
STATUS_INSUFFICIENT = "INSUFFICIENT (n<2 hands)"
STATUS_NOT_COMPARABLE = "NOT_COMPARABLE (prompt epoch differs)"
STATUS_NOT_COMPARABLE_SEAT = "NOT_COMPARABLE (seat_mode differs)"
STATUS_ROLE_UNSUPPORTED = "ROLE_UNSUPPORTED (think role, 0 traced rows)"
STATUS_ROLE_NOTE = "ROLE_NOTE (instruct role, reasoning_trace present)"
STATUS_TRACE_NOT_MEASURED = "NOT_MEASURED (no reasoning_trace column logged)"
STATUS_NOT_PRESENT = "NOT_PRESENT (no rows in either arm)"
STATUS_UNBINNED = "SPARSE (n<4, unbinned)"
STATUS_UNMEASURED_ALL_DROPPED = (
    "UNMEASURED (every {unit} of at least one side holds a harness-fallback row)"
)
STATUS_UNMEASURED_TRACE_GAME = (
    "UNMEASURED (no traced row of this game has a measurable CFR mass)"
)
_UNIT_SINGULAR = {"rows": "row", "hands": "hand"}
_FALLBACK_HAND = "fallback_hand"
METRICS = ("cfr_mass_chosen", "mean_reward")
DROP_REASONS = {
    "cfr_mass_chosen": (
        "null cfr_policy, null parsed_action, or an action the harness substituted"
    ),
    "reward": "null reward on the reduced hand row",
}
STATISTICS = ("hedges_g", "cliffs_delta", "mean_instruct", "mean_think")
_HAND_KEYS = ("matchup_id", "game", "seed", "hand_idx")
_QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]
_OPPONENT_BY_MATCHUP = {
    "lmvllm_vs_alwaysbet": "always_bet",
    "lmvllm_vs_bayesian": "bayesian",
    "lmvllm_vs_cfrref": "cfr_ref",
    "lmvllm_vs_rlselfplay": "rl",
}


def parse_opponent(matchup_id: str) -> str:
    """
    Opponent key for a matchup_id; unknown ids map to themselves.
    :param matchup_id:
    :returns:
    """
    return _OPPONENT_BY_MATCHUP.get(matchup_id, matchup_id)


def with_cfr_mass(df: pl.DataFrame) -> pl.DataFrame:
    """
    add `cfr_mass_chosen` and `cfr_mass_unmeasured` to a logged frame.
    On LM rows (`prompt` non-null) the mass is
    `json.loads(cfr_policy).get(str(parsed_action), 0.0)`. The 0.0 default is
    deliberate: choosing an action the
    equilibrium never plays is a measurement, and it measures zero.
    Three null cases are kept apart from that zero, because conflating them
    would report "we could not measure this" as "the equilibrium put no mass
    here":
    - a row with a null `cfr_policy` or a null `parsed_action` gets a NULL mass
      and `cfr_mass_unmeasured=True`;
    - an LM row whose action the harness substituted (`fallback_used`) gets a
      NULL mass and `cfr_mass_unmeasured=True`: its `parsed_action` is not what
      the model chose, so scoring it would credit or blame the model for the
      harness. A frame without a `fallback_used` column has none of these rows;
    - a non-LM row (opponent or chance node) gets a NULL mass and
      `cfr_mass_unmeasured=False`: it is outside the population, not unmeasured.
    :param df: a logged decision-node frame (pl.DataFrame)
    :returns: `df` plus `cfr_mass_chosen` (Float64) and `cfr_mass_unmeasured` (Boolean)
        (pl.DataFrame)
    :raises ValueError: if `prompt`, `cfr_policy` or `parsed_action` is absent
    :raises json.JSONDecodeError: if a `cfr_policy` value is not JSON (loud by design: a
        malformed reference is a finding, not a zero)
    """
    _require_columns(df, ("prompt", "cfr_policy", "parsed_action"))

    prompts = df["prompt"].to_list()
    policies = df["cfr_policy"].to_list()
    actions = df["parsed_action"].to_list()
    fallbacks = (
        df["fallback_used"].fill_null(True).to_list()
        if "fallback_used" in df.columns
        else [False] * df.height
    )

    mass: list[float | None] = []
    unmeasured: list[bool] = []
    for prompt, policy, action, fallback in zip(
        prompts, policies, actions, fallbacks, strict=True
    ):
        if prompt is None:
            mass.append(None)
            unmeasured.append(False)
        elif policy is None or action is None or fallback:
            mass.append(None)
            unmeasured.append(True)
        else:
            mass.append(float(json.loads(policy).get(str(action), 0.0)))
            unmeasured.append(False)

    return df.with_columns(
        pl.Series("cfr_mass_chosen", mass, dtype=pl.Float64),
        pl.Series("cfr_mass_unmeasured", unmeasured, dtype=pl.Boolean),
    )


def hand_level_lm_rewards(df: pl.DataFrame) -> pl.DataFrame:
    """
    one row per (seed, hand_idx, game, opponent) LM hand, with its reward and a flag for
    hands the harness played a move in
    `opponent` is derived from `matchup_id`, and `matchup_id` itself is carried through so
    the caller can disambiguate hand keys across arms. Flagged hands are kept and flagged,
    so the caller can count what it drops.
    The reward is the first reward in the hand: `reward` is the hand-level terminal return
    replicated onto every node of the hand, so first and last agree on a well-formed frame
    and the choice only matters on a malformed one.
    `fallback_hand` is true when any LM row of the hand has `fallback_used` set, false when
    the column is absent. Only LM rows count: an opponent row carries `fallback_used` too (a
    CFR-reference off-support move), and flagging from it would drop clean hands.
    :param df: a logged decision-node frame (pl.DataFrame)
    :returns: a frame with columns `seed` (when present), `hand_idx`, `game`, `opponent`,
        `reward`, `matchup_id` and `fallback_hand` (Boolean), sorted on the hand columns
        (pl.DataFrame)
    """
    _require_columns(df, ("prompt", "matchup_id", "game", "hand_idx", "reward"))
    lm = df.filter(pl.col("prompt").is_not_null()).with_columns(
        pl.col("matchup_id")
        .map_elements(parse_opponent, return_dtype=pl.Utf8)
        .alias("opponent")
    )
    if lm.is_empty():
        return pl.DataFrame(
            schema={
                "seed": pl.Int64,
                "hand_idx": pl.Int64,
                "game": pl.Utf8,
                "opponent": pl.Utf8,
                "reward": pl.Float64,
                "matchup_id": pl.Utf8,
                _FALLBACK_HAND: pl.Boolean,
            }
        )
    hand_cols = (
        ["seed", "hand_idx", "game", "opponent"]
        if "seed" in lm.columns
        else ["hand_idx", "game", "opponent"]
    )
    flagged = (
        pl.col("fallback_used").fill_null(True).any()
        if "fallback_used" in lm.columns
        else pl.lit(False)
    )
    return (
        lm.group_by(hand_cols)
        .agg(
            pl.col("reward").first(),
            pl.col("matchup_id").first(),
            flagged.alias(_FALLBACK_HAND),
        )
        .sort(hand_cols)
    )


def think_vs_instruct(
    arms: Mapping[str, ContrastArm], *, n_resamples: int = 1000, seed: int = 0
) -> list[dict]:
    """
    Contrast instruct arms against think arms, paired and pooled.
    Two contrast kinds:
    - `paired`: one contrast per `paired_with` edge, per (game, opponent) and
      per (game, `__all__`). The edge is canonicalised on the sorted name tuple
      and deduplicated, because the shipped roster declares `paired_with` on
      both sides of a pair; the think arm is always group "think" and the
      instruct arm always group "instruct", whichever side declared the edge.
    - `pooled`: within one partition of `(measured prompt_spec_hashes,
      seat_mode)`, every think arm against every instruct arm, per game, with
      `opponent="__all__"`. A cross-partition think x instruct pair contributes exactly
      one refusal row per game naming both arms, with the status naming which
      half of the key differed; no arm is dropped roster-wide, so one odd arm
      cannot empty the pooled contrast.
    - `role_evidence`: label rows only. A think arm with zero traced LM rows is
      refused from every contrast it would have joined
      (`ROLE_UNSUPPORTED (think role, 0 traced rows)`, one row per game), and an
      instruct arm that did log traces gets one `ROLE_NOTE` row.

    A harness-fallback unit is never the model's choice. On `mean_reward` a hand
    holding an LM fallback row is dropped whole (its reward was partly the
    harness's), on `cfr_mass_chosen` the fallback row is left unmeasured. Every
    row of a cell carries the count, so the reader sees what the number was
    computed without. The two metrics of one cell therefore report different
    `n_hands`, and the `INSUFFICIENT (n<2 hands)` floor can fire on one only. A
    cell in which every unit on at least one side is a fallback reads
    `STATUS_UNMEASURED_ALL_DROPPED`, checked before the floor.
    :param arms: loaded arms by name; roles other than think and instruct are ignored (Mapping[str, ContrastArm])
    :param n_resamples: bootstrap replicates handed to `effect_sizes_ci` (int)
    :param seed: base RNG seed for the bootstrap (int)
    :returns: rows built by `_think_row`, sorted on `(contrast_kind, game, opponent, metric,
        statistic, think_arms, instruct_arms)`. `partition` is the arm's measured prompt-hash
        prefix ("none" when the arm logs no hash), or "X vs Y" on a row that refuses a
        cross-partition pair. `n_instruct` / `n_think` count the values read (nodes for
        `cfr_mass_chosen`, hands for `mean_reward`); `n_hands_*` always count hands, which the
        `INSUFFICIENT (n<2 hands)` floor reads. The dropped counts are in `dropped_unit` and are
        None where no count is defined (list[dict])
    """
    roster = {n: a for n, a in arms.items() if a.role in ("think", "instruct")}
    traced = {name: traced_row_count(arm) for name, arm in roster.items()}
    unsupported = {
        name
        for name, arm in roster.items()
        # `None` (no reasoning_trace column) counts as `0`: the logs cannot support the role either
        # way.
        if arm.role == "think" and traced[name] in (0, None)
    }
    prepared = {name: _prepare(arm) for name, arm in roster.items()}

    rows: list[dict] = []
    rows.extend(_role_evidence_rows(roster, traced))
    rows.extend(
        _paired_rows(
            roster,
            prepared,
            unsupported=unsupported,
            n_resamples=n_resamples,
            seed=seed,
        )
    )
    rows.extend(
        _pooled_rows(
            roster,
            prepared,
            unsupported=unsupported,
            n_resamples=n_resamples,
            seed=seed,
        )
    )
    return sorted(
        rows,
        key=lambda r: (
            r["contrast_kind"],
            r["game"],
            r["opponent"],
            r["metric"],
            r["statistic"],
            r["think_arms"],
            r["instruct_arms"],
        ),
    )


def trace_length_quality(
    arms: Mapping[str, ContrastArm],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    min_rows_per_bin: int = 5,
) -> list[dict]:
    """Quality per quartile bin of trace length, within think arms.

    Trace length is `len(reasoning_trace)` in characters, by design: the logged
    `completion_tokens` counts the whole completion (trace plus action), not
    the trace alone, so it measures something else.
    Quality is the mean `cfr_mass_chosen` in the bin, with
    a hand-level bootstrap CI (`bootstrap_ci`, one call per (arm, game) with a
    stat_fn keyed by bin label).

    Sparse cells are labelled, never estimated:

    - an arm with no traced row at all yields one
      `ROLE_UNSUPPORTED (think role, 0 traced rows)` row and nothing else:
      the roster calls it a think arm and the logs do
      not support that;
    - a (arm, game) with fewer than 4 MEASURED traced rows cannot be quartiled at
      all and yields one `SPARSE (n<4, unbinned)` row carrying `n_rows` and no
      numbers;
    - a (arm, game) whose traced rows include none with a measurable CFR mass
      (every one a harness fallback, say) yields one
      `STATUS_UNMEASURED_TRACE_GAME` row carrying its dropped count;
    - a bin with fewer than `min_rows_per_bin` rows yields `SPARSE (n<N)` with a
      mean but a NULL CI.

    The quartile edges, `n_rows` and the sparse tests are over the traced rows
    that are measured; a harness-fallback row is never the model's choice. The
    role gate and the `__untraced__` summary stay over every LM row, because
    they are about traces, not about measurement. Every row of an (arm, game)
    carries that pair's dropped count and `n_dropped_longer_than_measured`, the
    fallback traced rows longer than the longest measured one: fallbacks are
    truncations, so the measured rows alone cut the long tail out of the top bin.

    LM rows on a think arm with an empty or null trace are never dropped
    silently: they are counted in one `__untraced__` summary row per arm.
    :param arms: loaded arms by name; arms whose role is not "think" are skipped (Mapping[str, ContrastArm])
    :param n_resamples: bootstrap replicates per (arm, game) (int)
    :param seed: base RNG seed for the bootstrap (int)
    :param min_rows_per_bin: floor below which a bin gets no CI (int, default 5)
    :returns: rows built by `_trace_row`, sorted on `(arm, game, bin)`. `value` is the mean
        `cfr_mass_chosen`; `bin_lo` / `bin_hi` are the observed trace-length bounds of the bin,
        not the quantile edges. `n_dropped` counts fallback traced rows of the (arm, game) and
        `n_total` all its traced rows; both are None on the arm-level rows (list[dict])
    """
    sparse_status = f"SPARSE (n<{min_rows_per_bin})"
    rows: list[dict] = []

    for name in sorted(arms):
        arm = arms[name]
        if arm.role != "think":
            continue
        df = filter_excluded(arm)
        lm = with_cfr_mass(df.filter(pl.col("prompt").is_not_null()))
        if "reasoning_trace" in lm.columns:
            lm = lm.with_columns(
                pl.col("reasoning_trace")
                .fill_null("")
                .str.len_chars()
                .alias("trace_chars")
            )
            traced = lm.filter(pl.col("trace_chars") > 0)
        else:
            traced = lm.clear()

        if traced.height == 0:
            rows.append(
                _trace_row(
                    arm=name,
                    game=ALL,
                    bin_label=ALL,
                    n_rows=lm.height,
                    n_hands=count_hands(lm, _HAND_KEYS),
                    n_dropped=_count_fallback_rows(lm),
                    n_total=lm.height,
                    dropped_unit="rows",
                    status=STATUS_ROLE_UNSUPPORTED,
                    note=(
                        f"All {lm.height} LM rows on this think arm have an empty or null "
                        f"reasoning_trace, so the role the roster declares is unsupported by "
                        f"the logs and no bin can be read. "
                        f"{_fallback_sentence(_count_fallback_rows(lm), lm.height)}"
                    ),
                )
            )
            continue

        untraced_rows = lm.filter(pl.col("trace_chars") == 0)
        untraced = untraced_rows.height
        n_untraced_fallback = _count_fallback_rows(untraced_rows)
        rows.append(
            _trace_row(
                arm=name,
                game=ALL,
                bin_label="__untraced__",
                n_rows=untraced,
                n_dropped=n_untraced_fallback,
                n_total=untraced,
                dropped_unit="rows",
                status="UNTRACED_SUMMARY",
                note=(
                    f"{untraced} of {lm.height} LM rows on this think arm have an empty or "
                    f"null reasoning_trace; they are excluded from the quartile bins and "
                    f"appear in no other row. "
                    f"{_fallback_sentence(n_untraced_fallback, untraced)}"
                ),
            )
        )

        for game in sorted(traced["game"].unique().to_list()):
            rows.extend(
                _game_bins(
                    arm_name=name,
                    game=game,
                    traced=traced.filter(pl.col("game") == game),
                    n_resamples=n_resamples,
                    seed=seed,
                    min_rows_per_bin=min_rows_per_bin,
                    sparse_status=sparse_status,
                )
            )

    return sorted(rows, key=lambda r: (r["arm"], r["game"], r["bin"]))


# --------------------------------------------------------------------------
# think_vs_instruct internals
# --------------------------------------------------------------------------


def traced_row_count(arm: ContrastArm) -> int | None:
    """
    LM rows on this arm whose `reasoning_trace` is neither null nor empty.
    This is the evidence the role gate reads: a property of the logs, not of the
    hand-typed roster label. Returns `None`, not a raise, when the log has no `reasoning_trace`
    column (a legacy schema): a think arm is unsupported either way, but an instruct arm's
    role-evidence check must say it is unmeasurable.
    :param arm:
    :return:
    """
    df = filter_excluded(arm)
    _require_columns(df, ("prompt",))
    lm = df.filter(pl.col("prompt").is_not_null())
    if "reasoning_trace" not in lm.columns:
        return None
    return lm.filter(pl.col("reasoning_trace").fill_null("").str.len_chars() > 0).height


def _role_evidence_rows(
    roster: Mapping[str, ContrastArm], traced: Mapping[str, int | None]
) -> list[dict]:
    """
    Label rows for instruct arms: traced or unmeasurable.
    :param roster:
    :param traced:
    :return:
    """
    rows: list[dict] = []
    for name in sorted(roster):
        arm = roster[name]
        if arm.role != "instruct":
            continue
        if traced[name] is None:
            rows.append(
                _label_row(
                    contrast_kind="role_evidence",
                    status=STATUS_TRACE_NOT_MEASURED,
                    partition=_arm_partition_label(arm),
                    instruct_arms=name,
                    think_arms="",
                    note=(
                        f"{SIGN_NOTE} Arm {name!r} carries no reasoning_trace column at all (a "
                        f"legacy log), so whether the roster's 'instruct' label agrees with "
                        f"the logged evidence cannot be checked. Its rows are still contrasted "
                        f"as configured; only this role-evidence check is unmeasurable."
                    ),
                )
            )
            continue
        if traced[name] == 0:
            continue
        rows.append(
            _label_row(
                contrast_kind="role_evidence",
                status=STATUS_ROLE_NOTE,
                partition=_arm_partition_label(arm),
                instruct_arms=name,
                think_arms="",
                n_instruct=traced[name],
                note=(
                    f"{SIGN_NOTE} The roster calls {name!r} an instruct arm, and "
                    f"{traced[name]} of its LM rows carry a non-empty reasoning_trace. Its "
                    f"rows are contrasted as configured; the disagreement between the role "
                    f"label and the logged evidence is reported, not resolved here."
                ),
            )
        )
    return rows


def _role_unsupported_rows(
    *,
    contrast_kind: str,
    games: list[str],
    think_arms: str,
    instruct_arms: str,
    partition: str,
    traced_names: str,
) -> list[dict]:
    """One ROLE_UNSUPPORTED row per game for a contrast a traceless arm blocks."""
    return [
        _label_row(
            contrast_kind=contrast_kind,
            game=game,
            status=STATUS_ROLE_UNSUPPORTED,
            partition=partition,
            instruct_arms=instruct_arms,
            think_arms=think_arms,
            n_think=0,
            note=(
                f"{SIGN_NOTE} Arm(s) {traced_names} are configured with role 'think' and carry "
                f"zero LM rows with a non-empty reasoning_trace, so nothing distinguishes them "
                f"from an instruct arm in the logs. This contrast is refused rather than "
                f"reported as a think-vs-instruct effect."
            ),
        )
        for game in games
    ]


def _seat_note(a: ContrastArm, b: ContrastArm) -> str:
    """
    The note fragment a contrast carries when its two arms differ in seat.
    Empty when they agree. Pooling across seat_mode is refused; a
    declared pair is still contrasted, because the roster asserted the pairing
    on purpose, but the reader is told what the pair mixes.
    :param a: ContrastArm
    :param b: ContrastArm
    :return: Seat's note
    """
    if _seat_key(a) == _seat_key(b):
        return ""
    return (
        f" The two arms differ in seat_mode ({a.name} {_seat_key(a)!r} vs "
        f"{b.name} {_seat_key(b)!r}), so this contrast mixes seat assignments; "
        "seat is a treatment, not a nuisance."
    )


def _prepare(arm: ContrastArm) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    (node frame, hand frame) for one arm, both arm-prefixed and role-tagged.
    The node frame is LM rows with `cfr_mass_chosen`; the hand frame is one row
    per LM hand with its reward. Both carry `role` (the group column) and a
    `matchup_id` rewritten to `"<arm>:<matchup_id>"`, which is what keeps hand
    keys disjoint across arms inside `resample_hands`.
    :param arm:
    :return:
    """
    df = filter_excluded(arm)
    nodes = with_cfr_mass(df.filter(pl.col("prompt").is_not_null())).with_columns(
        pl.col("matchup_id")
        .map_elements(parse_opponent, return_dtype=pl.Utf8)
        .alias("opponent"),
        pl.lit(arm.role).alias("role"),
    )
    hands = hand_level_lm_rewards(df).with_columns(pl.lit(arm.role).alias("role"))
    prefix = pl.lit(f"{arm.name}:") + pl.col("matchup_id")
    return (
        nodes.with_columns(prefix.alias("matchup_id")),
        hands.with_columns(prefix.alias("matchup_id")),
    )


def _paired_rows(
    roster: Mapping[str, ContrastArm],
    prepared: Mapping[str, tuple[pl.DataFrame, pl.DataFrame]],
    *,
    unsupported: set[str],
    n_resamples: int,
    seed: int,
) -> list[dict]:
    """
    One contrast per canonical `paired_with` edge (each edge formed once).
    `unsupported` names the think arms whose logged evidence does not support
    the role; a pair involving one is refused per game rather than
    contrasted, because a pair of two cannot proceed without one of them.
    :param roster:
    :param prepared:
    :param unsupported:
    :param n_resamples:
    :param seed:
    :return:
    """
    edges: dict[tuple[str, str], tuple[str, str]] = {}
    for name in sorted(roster):
        partner = roster[name].paired_with
        if not partner:
            continue
        low, high = sorted((name, partner))
        edges.setdefault((low, high), (name, partner))

    rows: list[dict] = []
    for _, (declarer, partner) in sorted(edges.items()):
        if partner not in roster:
            rows.append(
                _label_row(
                    contrast_kind="paired",
                    status=f"PAIR_MISSING ({partner} not loaded)",
                    instruct_arms="",
                    think_arms="",
                    partition=f"{_arm_partition_label(roster[declarer])} vs unknown",
                    note=(
                        f"{SIGN_NOTE} Arm {declarer!r} declares "
                        f"paired_with={partner!r}, which is not among the loaded "
                        "arms; the pair is reported as missing rather than dropped."
                    ),
                )
            )
            continue

        pair = [roster[declarer], roster[partner]]
        think = [a for a in pair if a.role == "think"]
        instruct = [a for a in pair if a.role == "instruct"]
        if len(think) != 1 or len(instruct) != 1:
            rows.append(
                _label_row(
                    contrast_kind="paired",
                    status=(
                        f"PAIR_INVALID (roles {pair[0].role}/{pair[1].role}; "
                        "a pair needs one think arm and one instruct arm)"
                    ),
                    instruct_arms=",".join(a.name for a in instruct),
                    think_arms=",".join(a.name for a in think),
                    partition=_partition_label(pair[0], pair[1]),
                    note=(
                        f"{SIGN_NOTE} No contrast is computed for a pair without "
                        "both roles."
                    ),
                )
            )
            continue

        refused = sorted({declarer, partner} & unsupported)
        if refused:
            rows.extend(
                _role_unsupported_rows(
                    contrast_kind="paired",
                    games=_games_of(prepared, (declarer, partner)),
                    think_arms=think[0].name,
                    instruct_arms=instruct[0].name,
                    partition=_partition_label(roster[declarer], roster[partner]),
                    traced_names=", ".join(refused),
                )
            )
            continue

        if not comparable(think[0], instruct[0]):
            rows.append(
                _label_row(
                    contrast_kind="paired",
                    status=STATUS_NOT_COMPARABLE,
                    instruct_arms=instruct[0].name,
                    think_arms=think[0].name,
                    partition=_partition_label(think[0], instruct[0]),
                    note=(
                        f"{SIGN_NOTE} The two arms' logged prompt_spec_hash sets "
                        "differ (or one is unknown), so they are never contrasted."
                    ),
                )
            )
            continue

        rows.extend(
            _cells(
                contrast_kind="paired",
                instruct=[instruct[0]],
                think=[think[0]],
                prepared=prepared,
                partition=_partition_label(think[0], instruct[0]),
                per_opponent=True,
                note_suffix=_seat_note(think[0], instruct[0]),
                n_resamples=n_resamples,
                seed=seed,
            )
        )
    return rows


def _pooled_rows(
    roster: Mapping[str, ContrastArm],
    prepared: Mapping[str, tuple[pl.DataFrame, pl.DataFrame]],
    *,
    unsupported: set[str],
    n_resamples: int,
    seed: int,
) -> list[dict]:
    """
    Pool the roster halves per game, within one (hash, seat_mode) partition.
    The partition is the arm's MEASURED
    `prompt_spec_hashes` tuple, never its `epoch` label, plus its `seat_mode`,
    and no arm is ever dropped roster-wide. Within a partition every think arm pools against every instruct
    arm; every cross-partition think x instruct pair contributes exactly one
    refusal row per game naming both arms, and the status says which half of the
    key differed: `NOT_COMPARABLE (prompt epoch differs)` or
    `NOT_COMPARABLE (seat_mode differs)`. Dropping both members of a failing
    pair from the whole epoch would let a single odd instruct arm empty the
    roster's pooled contrast.

    The hash half of the key is `comparable`'s equivalence class by
    construction: an arm with no logged `prompt_spec_hash` has an unknown epoch,
    which `comparable` never matches, not even against another unknown, so such
    an arm is keyed on its own name and is a partition of one.
    :param roster:
    :param prepared:
    :param unsupported:
    :param n_resamples:
    :param seed:
    :return:
    """
    all_think = sorted(
        (a for a in roster.values() if a.role == "think"), key=lambda a: a.name
    )
    think = [a for a in all_think if a.name not in unsupported]
    instruct = sorted(
        (a for a in roster.values() if a.role == "instruct"), key=lambda a: a.name
    )

    rows: list[dict] = []
    # A think arm without trace evidence is refused from the pool it
    # would have joined, and the pool proceeds without it.
    for t in all_think:
        if t.name not in unsupported:
            continue
        partners = [a for a in instruct if _partition_key(a) == _partition_key(t)]
        rows.extend(
            _role_unsupported_rows(
                contrast_kind="pooled",
                games=_games_of(prepared, (t.name,)),
                think_arms=t.name,
                instruct_arms=",".join(a.name for a in partners),
                partition=_arm_partition_label(t),
                traced_names=t.name,
            )
        )

    for t in think:
        for i in instruct:
            if _partition_key(t) == _partition_key(i):
                continue
            seat_only = _hash_key(t) == _hash_key(i)
            if seat_only:
                status = STATUS_NOT_COMPARABLE_SEAT
                why = (
                    f"share a prompt-hash partition but were seated differently "
                    f"(seat_mode {_seat_key(t)!r} vs {_seat_key(i)!r}), so they "
                    "are never pooled together"
                )
            else:
                status = STATUS_NOT_COMPARABLE
                why = (
                    "sit in different prompt-hash partitions, so they are never "
                    "pooled together"
                )
            # One row per game, not per opponent: the refusal is a property of
            # the arm pair, and one row per opponent would bury it.
            for game in _games_of(prepared, (t.name, i.name)):
                rows.append(
                    _label_row(
                        contrast_kind="pooled",
                        game=game,
                        status=status,
                        partition=_partition_label(t, i),
                        instruct_arms=i.name,
                        think_arms=t.name,
                        note=(
                            f"{SIGN_NOTE} Arms {t.name!r} and {i.name!r} {why}; "
                            "each still pools inside its own partition."
                        ),
                    )
                )

    for key in sorted({_partition_key(a) for a in think + instruct}):
        pooled_think = [a for a in think if _partition_key(a) == key]
        pooled_instruct = [a for a in instruct if _partition_key(a) == key]
        if not pooled_think or not pooled_instruct:
            continue
        rows.extend(
            _cells(
                contrast_kind="pooled",
                instruct=pooled_instruct,
                think=pooled_think,
                prepared=prepared,
                partition=_arm_partition_label(pooled_think[0]),
                per_opponent=False,
                n_resamples=n_resamples,
                seed=seed,
            )
        )
    return rows


def _cells(
    *,
    contrast_kind: str,
    instruct: list[ContrastArm],
    think: list[ContrastArm],
    prepared: Mapping[str, tuple[pl.DataFrame, pl.DataFrame]],
    partition: str,
    per_opponent: bool,
    n_resamples: int,
    seed: int,
    note_suffix: str = "",
) -> list[dict]:
    """Every (game, opponent, metric) cell for one instruct set vs one think set.

    `note_suffix` rides on every row of this contrast: it is where a caveat
    about the contrast (rather than about the cell) is written, e.g. a paired
    pair whose two arms were seated differently.
    """
    instruct_names = [a.name for a in instruct]
    think_names = [a.name for a in think]
    names = instruct_names + think_names

    instruct_nodes = _concat([prepared[n][0] for n in instruct_names])
    think_nodes = _concat([prepared[n][0] for n in think_names])
    instruct_hands = _concat([prepared[n][1] for n in instruct_names])
    think_hands = _concat([prepared[n][1] for n in think_names])

    # Null values are dropped inside `_cell_rows`, so the count it reports is
    # the count for that cell rather than an arm-wide total.
    frames = {
        "cfr_mass_chosen": (instruct_nodes, think_nodes, "cfr_mass_chosen"),
        "mean_reward": (instruct_hands, think_hands, "reward"),
    }

    opponents = [ALL]
    if per_opponent:
        opponents = sorted(
            set(instruct_nodes["opponent"].to_list())
            | set(think_nodes["opponent"].to_list())
        ) + [ALL]

    rows: list[dict] = []
    for game in _games_of(prepared, names):
        for opponent in opponents:
            selected = {
                metric: (
                    _select(frames[metric][0], game, opponent),
                    _select(frames[metric][1], game, opponent),
                )
                for metric in METRICS
            }
            # A cell neither arm played is NOT_PRESENT, and it is one row.
            # Reporting it as four INSUFFICIENT rows per metric would say
            # "too few hands" about a cell that was never observed at all.
            if not any(f.height for pair in selected.values() for f in pair):
                rows.append(
                    _label_row(
                        contrast_kind=contrast_kind,
                        game=game,
                        opponent=opponent,
                        partition=partition,
                        status=STATUS_NOT_PRESENT,
                        instruct_arms=",".join(instruct_names),
                        think_arms=",".join(think_names),
                        n_instruct=0,
                        n_think=0,
                        n_hands_instruct=0,
                        n_hands_think=0,
                        note=(
                            f"{SIGN_NOTE} Neither arm has a row on "
                            f"({game}, {opponent}), so there is nothing to "
                            "contrast; this is an absence of data, not a sample "
                            f"too small to read.{note_suffix}"
                        ),
                    )
                )
                continue
            for metric in METRICS:
                i_frame, t_frame = selected[metric]
                rows.extend(
                    _cell_rows(
                        contrast_kind=contrast_kind,
                        game=game,
                        opponent=opponent,
                        metric=metric,
                        partition=partition,
                        value_col=frames[metric][2],
                        instruct_frame=i_frame,
                        think_frame=t_frame,
                        instruct_arms=",".join(instruct_names),
                        think_arms=",".join(think_names),
                        note_suffix=note_suffix,
                        n_resamples=n_resamples,
                        seed=seed,
                    )
                )
    return rows


def _cell_rows(
    *,
    contrast_kind: str,
    game: str,
    opponent: str,
    metric: str,
    partition: str,
    value_col: str,
    instruct_frame: pl.DataFrame,
    think_frame: pl.DataFrame,
    instruct_arms: str,
    think_arms: str,
    n_resamples: int,
    seed: int,
    note_suffix: str = "",
) -> list[dict]:
    """
    the four statistic rows for one (game, opponent, metric) cell
    Harness-fallback units are dropped first and counted in the columns `n_dropped`,
    `n_total`, `dropped_rate` and `dropped_unit` (and per side): on `reward` a flagged hand
    is dropped whole, on `cfr_mass_chosen` a fallback row is already null and is counted from
    `fallback_used`. Rows with a null value are then dropped and counted in the note, so a
    cell never silently loses observations: on `cfr_mass_chosen` a null value is exactly a
    row `with_cfr_mass` flagged `cfr_mass_unmeasured`, and on `reward` it is a reduced hand
    with no logged terminal reward. The note names the reason for this metric's value
    column, from `DROP_REASONS`, and states the fallback count in the unit the columns use.
    A side whose every unit is a fallback makes the cell `STATUS_UNMEASURED_ALL_DROPPED`,
    judged before the two-hand floor.
    :param contrast_kind: "paired" or "pooled" (str)
    :param game: the game key of the cell (str)
    :param opponent: the opponent key, or `ALL` (str)
    :param metric: one of `METRICS` (str)
    :param partition: the partition label written into every row (str)
    :param value_col: the frame column the metric reads, `cfr_mass_chosen` or `reward` (str)
    :param instruct_frame: the instruct side's units for this cell (pl.DataFrame)
    :param think_frame: the think side's units for this cell (pl.DataFrame)
    :param instruct_arms: comma-joined instruct arm names (str)
    :param think_arms: comma-joined think arm names (str)
    :param n_resamples: bootstrap replicates for `effect_sizes_ci` (int)
    :param seed: bootstrap RNG seed (int)
    :param note_suffix: contrast-level caveat appended to every row's note (str)
    :returns: the four statistic rows of the cell (list[dict])
    """
    unit = "hands" if value_col == "reward" else "rows"
    instruct_frame, n_dropped_instruct, n_total_instruct = _drop_fallback_units(
        instruct_frame, value_col
    )
    think_frame, n_dropped_think, n_total_think = _drop_fallback_units(
        think_frame, value_col
    )
    n_dropped = n_dropped_instruct + n_dropped_think
    n_total = n_total_instruct + n_total_think

    kept_instruct = instruct_frame.filter(pl.col(value_col).is_not_null())
    kept_think = think_frame.filter(pl.col(value_col).is_not_null())
    dropped = (instruct_frame.height - kept_instruct.height) + (
        think_frame.height - kept_think.height
    )
    instruct_frame, think_frame = kept_instruct, kept_think

    n_hands_instruct = count_hands(instruct_frame, _HAND_KEYS)
    n_hands_think = count_hands(think_frame, _HAND_KEYS)
    note = SIGN_NOTE + note_suffix
    if dropped:
        reason = DROP_REASONS.get(value_col, f"null {value_col}")
        note += (
            f" {dropped} {unit} in this cell have no measurable {value_col} "
            f"({reason}) and are excluded."
        )
    if n_dropped:
        rates = (
            f"dropped_rate {_rate_note(n_dropped, n_total)}; instruct "
            f"{_rate_note(n_dropped_instruct, n_total_instruct)}, think "
            f"{_rate_note(n_dropped_think, n_total_think)}"
        )
        if value_col == "reward":
            note += (
                f" {n_dropped} of {n_total} hands in this cell hold a harness-fallback "
                f"row and are dropped whole ({rates}), before the null-reward "
                "exclusion above is counted."
            )
        else:
            note += (
                f" {n_dropped} of {n_total} rows in this cell are harness-fallback "
                f"rows ({rates}); they are among the rows excluded above."
            )

    common = dict(
        contrast_kind=contrast_kind,
        game=game,
        opponent=opponent,
        metric=metric,
        partition=partition,
        instruct_arms=instruct_arms,
        think_arms=think_arms,
        n_instruct=instruct_frame.height,
        n_think=think_frame.height,
        n_hands_instruct=n_hands_instruct,
        n_hands_think=n_hands_think,
        n_dropped=n_dropped,
        n_total=n_total,
        dropped_unit=unit,
        dropped_rate_instruct=_rate(n_dropped_instruct, n_total_instruct),
        dropped_rate_think=_rate(n_dropped_think, n_total_think),
        note=note,
    )

    emptied = [
        side
        for side, dropped_side, total_side in (
            ("instruct", n_dropped_instruct, n_total_instruct),
            ("think", n_dropped_think, n_total_think),
        )
        if total_side and dropped_side == total_side
    ]
    if emptied:
        common["note"] = (
            f"{note} Every {_UNIT_SINGULAR[unit]} on the {' and '.join(emptied)} side "
            "holds a harness-fallback row, so no unit the model chose is left to read."
        )
        status = STATUS_UNMEASURED_ALL_DROPPED.format(unit=_UNIT_SINGULAR[unit])
        return [
            _think_row(statistic=stat, status=status, **common) for stat in STATISTICS
        ]

    if n_hands_instruct < 2 or n_hands_think < 2:
        return [
            _think_row(statistic=stat, status=STATUS_INSUFFICIENT, **common)
            for stat in STATISTICS
        ]

    combined = pl.concat([instruct_frame, think_frame], how="diagonal_relaxed")
    effects = effect_sizes_ci(
        combined,
        "role",
        value_col,
        group_a="instruct",
        group_b="think",
        n_resamples=n_resamples,
        seed=seed,
    )

    rows = [
        _think_row(
            statistic=effect["metric"],
            value=effect["value"],
            ci_lo=effect["ci_lo"],
            ci_hi=effect["ci_hi"],
            status=_status_for(effect["value"]),
            **common,
        )
        for effect in effects
    ]
    mean_note = (
        f"{note} Mean rows are point estimates; the CIs are on the effect-size "
        "rows of the same cell."
    )
    for stat, frame in (
        ("mean_instruct", instruct_frame),
        ("mean_think", think_frame),
    ):
        mean = frame[value_col].mean()
        rows.append(
            _think_row(
                **{**common, "note": mean_note},
                statistic=stat,
                value=mean,
                status=_status_for(mean),
            )
        )
    return rows


_THINK_FIELDS = (
    "contrast_kind",
    "game",
    "opponent",
    "metric",
    "partition",
    "statistic",
    "value",
    "ci_lo",
    "ci_hi",
    "n_instruct",
    "n_think",
    "n_hands_instruct",
    "n_hands_think",
    "n_dropped",
    "n_total",
    "dropped_rate",
    "dropped_unit",
    "dropped_rate_instruct",
    "dropped_rate_think",
    "instruct_arms",
    "think_arms",
    "status",
    "note",
    "caveat",
)
_THINK_REQUIRED = (
    "contrast_kind",
    "game",
    "opponent",
    "metric",
    "statistic",
    "status",
    "partition",
    "instruct_arms",
    "think_arms",
    "note",
)
_TRACE_FIELDS = (
    "arm",
    "game",
    "bin",
    "bin_lo",
    "bin_hi",
    "n_rows",
    "n_hands",
    "n_dropped",
    "n_total",
    "dropped_rate",
    "dropped_unit",
    "n_dropped_longer_than_measured",
    "n_unmeasured_other",
    "value",
    "ci_lo",
    "ci_hi",
    "status",
    "note",
    "caveat",
)
_TRACE_REQUIRED = ("arm", "game", "bin", "status", "note")
_DERIVED = frozenset({"dropped_rate", "caveat"})
_CLEANED = frozenset({"value", "ci_lo", "ci_hi"})


def _finish_think_row(row: dict) -> dict:
    """
    derive the fields no caller may write: `dropped_rate` from the two counts, so a row cannot
    report a rate that disagrees with them, and the caveat, which is always `TRACE_CAVEAT`
    :param row: a row from `build_row` (dict)
    :returns: the same row (dict)
    """
    row["dropped_rate"] = _rate(row["n_dropped"], row["n_total"])
    row["caveat"] = TRACE_CAVEAT
    return row


def _think_row(**values: Any) -> dict:
    """
    one `think_vs_instruct` row
    :param values: the keys of `_THINK_FIELDS`; `_THINK_REQUIRED` must be present (Any)
    :returns: the row dict (dict)
    """
    return _finish_think_row(
        build_row(
            _THINK_FIELDS,
            values,
            derived=_DERIVED,
            required=_THINK_REQUIRED,
            cleaned=_CLEANED,
        )
    )


def _label_row(*, game: str = ALL, opponent: str = ALL, **values: Any) -> dict:
    """
    a refusal or label row: `metric` and `statistic` are `ALL` and the numbers stay None
    :param game: the game key, `ALL` for an arm-level row (str)
    :param opponent: the opponent key, `ALL` unless the row is for one opponent (str)
    :param values: the remaining keys of `_THINK_FIELDS` (Any)
    :returns: the row dict (dict)
    """
    return _think_row(game=game, opponent=opponent, metric=ALL, statistic=ALL, **values)


# --------------------------------------------------------------------------
# trace_length_quality internals
# --------------------------------------------------------------------------


def _game_bins(
    *,
    arm_name: str,
    game: str,
    traced: pl.DataFrame,
    n_resamples: int,
    seed: int,
    min_rows_per_bin: int,
    sparse_status: str,
) -> list[dict]:
    """
    quartile rows for one (arm, game), or one labelled row when it cannot be quartiled
    The bins are formed over the traced rows that are measured (a non-null
    `cfr_mass_chosen`); the harness-fallback rows among `traced` are counted, not binned, and
    every returned row carries that count. `n_dropped_longer_than_measured` is the fallback
    rows whose trace is longer than the longest measured trace, and None when no measured row
    exists to compare against. `n_unmeasured_other` counts the traced rows that are unmeasured
    for a reason other than a fallback (a null policy or action); they are not binned either,
    and the note says so.
    :param arm_name: the think arm (str)
    :param game: the game key (str)
    :param traced: every traced LM row of this (arm, game), with `cfr_mass_chosen` and `trace_chars` (pl.DataFrame)
    :param n_resamples: bootstrap replicates (int)
    :param seed: bootstrap RNG seed (int)
    :param min_rows_per_bin: floor below which a bin gets no CI (int)
    :param sparse_status: the label of a bin under that floor (str)
    :returns: the (arm, game)'s rows (list[dict])
    """
    measured = traced.filter(pl.col("cfr_mass_chosen").is_not_null())
    fallback = (
        traced.filter(pl.col("fallback_used").fill_null(True))
        if "fallback_used" in traced.columns
        else traced.clear()
    )
    counts = dict(
        n_dropped=fallback.height,
        n_total=traced.height,
        dropped_unit="rows",
        n_unmeasured_other=(
            traced.filter(pl.col("cfr_mass_chosen").is_null()).height - fallback.height
        ),
        n_dropped_longer_than_measured=(
            None
            if measured.is_empty()
            else fallback.filter(
                pl.col("trace_chars") > measured["trace_chars"].max()
            ).height
        ),
    )
    lost = ""
    if fallback.height:
        lost = (
            f" {fallback.height} of the {traced.height} traced LM rows of this "
            "(arm, game) are harness-fallback rows and are not binned"
        )
        if counts["n_dropped_longer_than_measured"] is not None:
            lost += (
                f"; {counts['n_dropped_longer_than_measured']} of them are longer "
                "than the longest measured trace"
            )
        lost += "."
    if counts["n_unmeasured_other"]:
        other = counts["n_unmeasured_other"]
        noun, be = ("row has", "is") if other == 1 else ("rows have", "are")
        lost += (
            f" {other} further traced {noun} no measurable CFR mass for another reason "
            f"(null cfr_policy or parsed_action) and {be} not binned."
        )

    if measured.is_empty():
        return [
            _trace_row(
                arm=arm_name,
                game=game,
                bin_label=ALL,
                n_rows=0,
                n_hands=0,
                status=STATUS_UNMEASURED_TRACE_GAME,
                note=(
                    f"None of the {traced.height} traced LM rows of this (arm, game) has "
                    f"a measurable CFR mass, so no bin can be read.{lost}"
                ),
                **counts,
            )
        ]

    if measured.height < len(_QUARTILE_LABELS):
        return [
            _trace_row(
                arm=arm_name,
                game=game,
                bin_label=ALL,
                n_rows=measured.height,
                n_hands=count_hands(measured, _HAND_KEYS),
                status=STATUS_UNBINNED,
                note=(
                    f"{measured.height} measured traced LM rows is fewer than the 4 "
                    f"quartiles, so this (arm, game) is not binned and carries no "
                    f"numbers.{lost}"
                ),
                **counts,
            )
        ]

    binned = measured.with_columns(
        pl.col("trace_chars")
        .qcut(4, labels=_QUARTILE_LABELS, allow_duplicates=True)
        .cast(pl.Utf8)
        .alias("trace_bin")
    )
    labels = [b for b in _QUARTILE_LABELS if b in binned["trace_bin"].to_list()]

    def _stat_fn(frame: pl.DataFrame) -> dict:
        out = {}
        for label in labels:
            values = frame.filter(pl.col("trace_bin") == label)["cfr_mass_chosen"]
            mean = values.mean()
            out[label] = float("nan") if mean is None else float(cast(float, mean))
        return out

    cis = bootstrap_ci(binned, _stat_fn, n_resamples=n_resamples, level=0.95, seed=seed)

    rows: list[dict] = []
    for label in labels:
        cell = binned.filter(pl.col("trace_bin") == label)
        mean = cell["cfr_mass_chosen"].mean()
        sparse = cell.height < min_rows_per_bin
        status = sparse_status if sparse else STATUS_OK
        note = (
            f"Quality is the mean CFR mass on the chosen action over the {cell.height} "
            f"rows of this bin. Bin bounds are the observed trace_chars range.{lost}"
        )
        if sparse:
            note += (
                f" Fewer than {min_rows_per_bin} rows: the CI is withheld rather "
                "than estimated."
            )
        rows.append(
            _trace_row(
                arm=arm_name,
                game=game,
                bin_label=label,
                bin_lo=cell["trace_chars"].min(),
                bin_hi=cell["trace_chars"].max(),
                n_rows=cell.height,
                n_hands=count_hands(cell, _HAND_KEYS),
                value=mean,
                ci_lo=None if sparse else cis[label]["lo"],
                ci_hi=None if sparse else cis[label]["hi"],
                status=status,
                note=note,
                **counts,
            )
        )
    return rows


def _trace_row(*, bin_label: str, **values: Any) -> dict:
    """
    one `trace_length_quality` row; `bin_label` is written as the key `bin`
    :param bin_label: the quartile label, `ALL` or `__untraced__` (str)
    :param values: the other keys of `_TRACE_FIELDS`; `arm`, `game`, `status` and `note` are
        required (Any)
    :returns: the row dict, `dropped_rate` derived from the two counts (dict)
    """
    return _finish_think_row(
        build_row(
            _TRACE_FIELDS,
            {"bin": bin_label, **values},
            derived=_DERIVED,
            required=_TRACE_REQUIRED,
            cleaned=_CLEANED,
        )
    )


# shared helpers


def _require_columns(df: pl.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"frame is missing required column(s) {missing}; "
            "this is not a decision-node log"
        )


def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")


def _select(frame: pl.DataFrame, game: str, opponent: str) -> pl.DataFrame:
    out = frame.filter(pl.col("game") == game)
    if opponent != ALL:
        out = out.filter(pl.col("opponent") == opponent)
    return out


def _games_of(
    prepared: Mapping[str, tuple[pl.DataFrame, pl.DataFrame]], names: Iterable[str]
) -> list[str]:
    games: set[str] = set()
    for name in names:
        games.update(prepared[name][0]["game"].unique().to_list())
    return sorted(games)


def _hash_key(arm: ContrastArm) -> tuple[str, ...]:
    """
    The prompt-epoch half of the partition key: the measured hash tuple.
    An arm with no logged `prompt_spec_hash` is keyed on its own name, so it is
    a partition of one and pools with nobody, not even another hashless arm.
    That is exactly what `comparable` says about an unknown epoch.
    :param arm: the arm
    :returns: the partition key
    """
    return arm.prompt_spec_hashes or ("__unknown__", arm.name)


def _seat_key(arm: ContrastArm) -> str:
    """
    The seat half of the partition key; an unread seat_mode is its own value.
    :param arm: the arm
    :returns: the seat half of the partition key
    """
    return arm.seat_mode or "unknown"


def _partition_key(arm: ContrastArm) -> tuple[tuple[str, ...], str]:
    """
    The pooling partition an arm belongs to: (hash tuple, seat_mode).
    Both halves, because pooling two arms that were seated
    differently reports one number for two populations. `unknown` is a seat
    value like any other, so an arm whose `run_meta.json` records no seat_mode
    never pools with one that does.
    :param arm: the arm
    :returns: the partition key
    """
    return (_hash_key(arm), _seat_key(arm))


def _hash_prefix(arm: ContrastArm) -> str:
    """
    Short label for an arm's prompt epoch: the hash prefixes, or 'none'.
    :param arm: the arm
    :returns: the short label for an arm's prompt epoch
    """
    if not arm.prompt_spec_hashes:
        return "none"
    return "+".join(h[:8] for h in arm.prompt_spec_hashes)


def _arm_partition_label(arm: ContrastArm) -> str:
    """One arm's partition, as it is written into a row: hashes and seat_mode."""
    return f"{_hash_prefix(arm)} seat={_seat_key(arm)}"


def _partition_label(a: ContrastArm, b: ContrastArm) -> str:
    """The shared partition label, or 'X vs Y' when the two arms differ."""
    if _partition_key(a) == _partition_key(b):
        return _arm_partition_label(a)
    return f"{_arm_partition_label(a)} vs {_arm_partition_label(b)}"


def _rate(n_dropped: int | None, n_total: int | None) -> float | None:
    """
    the dropped rate of a population
    :param n_dropped: units dropped, or None when the count is not defined (int or None)
    :param n_total: units before the drop, or None (int or None)
    :returns: `n_dropped / n_total`, None when either is unknown or the population is empty (float or None)
    """
    if n_dropped is None or not n_total:
        return None
    return n_dropped / n_total


def _rate_note(n_dropped: int, n_total: int) -> str:
    """
    a dropped rate as it reads inside a note
    :param n_dropped: units dropped (int)
    :param n_total: units before the drop (int)
    :returns: the rate to three places, or "n/a" on an empty population (str)
    """
    rate = _rate(n_dropped, n_total)
    return "n/a" if rate is None else f"{rate:.3f}"


def _count_fallback_rows(frame: pl.DataFrame) -> int:
    """
    the LM rows of a frame whose action the harness substituted
    :param frame: LM rows (pl.DataFrame)
    :returns: the count, 0 when the frame has no `fallback_used` column (int)
    """
    if "fallback_used" not in frame.columns:
        return 0
    return frame.filter(pl.col("fallback_used").fill_null(True)).height


def _fallback_sentence(n_dropped: int, n_total: int) -> str:
    """
    the note fragment that states how many of a row's rows are fallbacks
    :param n_dropped: fallback rows in the population (int)
    :param n_total: rows in the population (int)
    :returns: a leading-space sentence, or "" when there are none (str)
    """
    if not n_dropped:
        return ""
    return f" {n_dropped} of the {n_total} are harness-fallback rows."


def _drop_fallback_units(
    frame: pl.DataFrame, value_col: str
) -> tuple[pl.DataFrame, int, int]:
    """
    drop the harness-fallback units of one side of a cell and count them
    On `reward` the units are hands and a flagged hand leaves the frame whole; on
    `cfr_mass_chosen` the units are LM rows, whose mass `with_cfr_mass` has already nulled, so
    the frame is returned unchanged and the fallback rows are only counted. A frame without the
    flag column has no fallback units.
    :param frame: one side's units for one cell, hands (`reward`) or nodes (`cfr_mass_chosen`) (pl.DataFrame)
    :param value_col: the column the metric reads (str)
    :returns: (frame without the dropped units, n_dropped, n_total before the drop) (tuple)
    """
    n_total = frame.height
    if value_col == "reward":
        if _FALLBACK_HAND not in frame.columns:
            return frame, 0, n_total
        kept = frame.filter(~pl.col(_FALLBACK_HAND))
        return kept, n_total - kept.height, n_total
    return frame, _count_fallback_rows(frame), n_total


def _status_for(value: Any) -> str:
    """
    OK, or a named UNDEFINED status when the statistic has no value.
    :param value: the value to check
    :returns: the status string (str)
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNDEFINED (zero pooled variance or too few observations)"
    return STATUS_OK
