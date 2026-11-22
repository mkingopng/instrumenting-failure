"""Reading one collected arm directory back, verified against its own artefacts.

An arm is the directory one sweep wrote: Parquet shards, ``run_meta.json`` and
``shard_outcomes.json``. ``read_arm`` refuses an arm whose rows contradict its
provenance or that lost a shard, and ``discover_arms`` keys a pilot root's arms
by role. The scoreboard, the probe scorer and the acceptance gate all read arms
through here.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import pathlib
from typing import Optional

import polars as pl

from lmopt.analysis.condition import (
    ResolvedConfigShapeError,
    parse_drive_openings_field,
    require_seat_mode_field,
)

#: Arm roles, keyed by the `probe_type` the arm's own rows carry. "none" is the
#: base arm; the other three are the probe types.
ROLE_BASE = "base"
ROLE_RELABEL = "relabel"
ROLE_PAYOFF = "payoff_scale"
ROLE_DRIVEN = "offpath_drive"


class ScriptError(Exception):
    """The script cannot run at all (EXIT_SCRIPT_ERROR)."""


# Arms
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Arm:
    """One collected arm, with the provenance its own directory records."""

    role: str
    name: str
    path: str
    df: pl.DataFrame
    actor: str
    probe_type: str
    prompt_variant: str
    # A game-disjoint off-path composite records this as {game: [action, ...]}
    # instead of a flat list, since different games' pieces can legitimately
    # drive different openings. Every reader resolves per game and never
    # coerces this with `list(...)`, which on a dict returns its keys.
    drive_openings: Optional[list | dict]
    seat_mode: str | dict | None
    cfr_iters: int
    cfr_cache_dir: str
    n_rows: int

    configured_games: tuple[str, ...] = ()

    def games(self) -> list[str]:
        return sorted(self.df["game"].unique().to_list())

    def uncollected_games(self) -> list[str]:
        """Games this arm was configured for that wrote no row.

        P1 and P4 enumerate an expected game set (the variant registry), so a
        game that vanishes leaves a visible gap in their per-game lines. P3, P6
        and P8 enumerate the data -- `df["game"].unique()`, or a filter over it
        -- so a game whose rows were lost produces no line at all, and the
        headline is a mean over the survivors reported under the same name. The
        arm's own `run_meta.json` already records what it was asked to play, so
        `configured - observed` is exactly the silent-drop set and it costs a
        set difference. Empty when `run_meta` records no
        game list, so an empty result does not by itself mean nothing was lost:
        `_uncollected_line` says when the check could not run.
        """
        if not self.configured_games:
            return []
        return sorted(set(self.configured_games) - set(self.games()))


def _actor_of(resolved: dict) -> str:
    """Which agent this arm is a measurement of, from its own resolved config.

    The scorers take an `actor` column value, and getting it wrong scores the
    opponent under the agent's name. It is derived here rather than typed: a
    random-agent0 sweep's rows carry `match_runner._agent_id(RandomAgent())`,
    and an LM sweep's carry the served model id, so the same table works for a
    real collection arm without an edit.
    """
    from lmopt.agents.random_agent import RandomAgent
    from lmopt.harness.match_runner import _agent_id

    if resolved.get("agent0") == "random":
        return _agent_id(RandomAgent())
    if resolved.get("agent0") == "adaptive":
        from lmopt.agents.adaptive import AdaptiveAgent

        return _agent_id(AdaptiveAgent(prior_policy=None))
    model = resolved.get("model")
    if not model:
        raise ScriptError(
            "run_meta.json records agent0='lm' with no model, so there is no "
            "actor to score"
        )
    return model


def _driven_infosets(game_key: str, drive: list[int]) -> set[str]:
    """Every `information_state_string` reachable once `drive` is forced from
    the initial state, across every chance (deal) outcome.

    Mirrors `run_sweep._refuse_illegal_drive`'s replay of the forced plies, but
    does not stop at one chance draw: that function only needs to know the
    drive is legal, so the first chance outcome at each deal node is enough.
    A driven arm's logged rows span every hand dealt during collection, so
    verifying them needs the set over every hand, not one.

    `match_runner._apply_drive`'s docstring is why this is a sound check at
    all: "nothing forced is logged... the plies are still in the state's
    history", so every row downstream of the drive carries it inside the
    string this function also computes. No representation lets an info-set
    string be turned back into a state (`property6.reference_policy_map`'s
    docstring), so membership in this reachable set -- not a substring/prefix
    parse of the string itself, which is game-format-specific -- is the check.
    """
    from lmopt.games.loaders import load_game

    game = load_game(game_key)

    def walk(state, remaining: tuple[int, ...]) -> set[str]:
        if state.is_terminal():
            return set()
        if state.is_chance_node():
            out: set[str] = set()
            for action, _ in state.chance_outcomes():
                out |= walk(state.child(action), remaining)
            return out
        if remaining:
            action = remaining[0]
            if action not in state.legal_actions():
                return set()
            return walk(state.child(action), remaining[1:])
        out = {state.information_state_string(state.current_player())}
        for action in state.legal_actions():
            out |= walk(state.child(action), remaining)
        return out

    return walk(game.new_initial_state(), tuple(drive))


def _drive_mismatch_problems(
    name: str, df, drive: list[int] | dict[str, list[int]]
) -> list[str]:
    """Refusal text for every driven row whose `info_set` is unreachable under
    the arm's own recorded `drive_openings` -- the fabricated-provenance shape
    where `run_meta.json` says one forced opening while the rows were
    collected under another.

    Limit: this check detects over-claimed drive
    (rows unreachable under the claimed opening), but cannot detect under-claimed
    drive (claimed shallower than actual). A necessary-condition check (every
    reachable info-set has >=1 observed row) is unsound: with finite sampling,
    sparse chance coverage legitimately leaves reachable nodes unobserved, so the
    check would produce false positives. Validated against synthetic Kuhn data:
    drive=[1] (P0 bets) makes 3 info-sets reachable ('0b', '1b', '2b'), but with
    limited hands, not all 3 cards may be dealt to P1, so observing only 2 of 3
    is legitimate, not evidence of under-claiming.

    `drive` is a per-game dict on a game-disjoint off-path composite
    (`compose_arm_pieces._composite_drive_openings` records one when the arm's
    pieces drove different games with different openings) and a flat list
    otherwise. Resolved per `game_key` inside the loop below rather than
    coerced with `list(drive)` at the call site: `list()` on a dict returns its
    keys (game-name strings), which would then reach `_driven_infosets` as if
    they were action indices.
    """
    problems: list[str] = []
    for game_key in sorted(set(df["game"].to_list())):
        if isinstance(drive, dict):
            if game_key not in drive:
                problems.append(
                    f"{name}: drive_openings={drive!r} records no entry for "
                    f"{game_key!r}, but the arm has rows for it -- the forced "
                    "opening claim is incomplete for this game"
                )
                continue
            game_drive = drive[game_key]
        else:
            game_drive = drive
        reachable = _driven_infosets(game_key, game_drive)
        game_rows = df.filter(df["game"] == game_key)
        bad = sorted(set(game_rows["info_set"].to_list()) - reachable)
        if bad:
            problems.append(
                f"{name}: run_meta.json records drive_openings={game_drive!r} for "
                f"{game_key!r} but {len(bad)} row info_set(s) are unreachable "
                f"under it (e.g. {bad[:3]!r}); the forced opening claim "
                f"contradicts the data it is printed beside"
            )
    return problems


def read_arm(path: str) -> tuple[Optional[Arm], list[str]]:
    """One arm directory, verified against its own artefacts.

    Returns `(arm, problems)`; `arm` is None whenever anything below refuses
    it, and every refusal is a named problem printed with the table. The five
    checks, and what each stops:

    1. `run_meta.json` present and readable -- an arm with no provenance cannot
       say which manipulation produced it.
    2. `shard_outcomes.json` present, complete, nothing failed -- the file
       exists precisely so an incomplete arm is not read as a finished one
       (`run_sweep.SHARD_OUTCOMES_FILENAME`), and an ABSENT record is not
       evidence of completeness.
    3. At least one Parquet shard.
    4. The rows' own `probe_type` / `prompt_variant` agree with the meta -- the
       artifact check. Without it, a directory whose rows are canonical scores
       as a probe arm, reporting a manipulation that never reached the data.
    5. On a driven arm (`drive_openings` recorded), every row's `info_set` is
       reachable under that exact forced opening (`_drive_mismatch_problems`)
       -- otherwise `drive_openings` is a claim from the meta alone, printed
       beside a `tv_mean` scored against a subgame the rows never visited.
    """
    problems: list[str] = []
    name = os.path.basename(path.rstrip("/"))
    meta_path = os.path.join(path, "run_meta.json")
    if not os.path.exists(meta_path):
        return None, [f"{name}: no run_meta.json, so the arm cannot be identified"]
    try:
        resolved = (
            json.loads(pathlib.Path(meta_path).read_text()).get("resolved_config") or {}
        )
    except (OSError, ValueError) as exc:
        return None, [f"{name}: run_meta.json is unreadable ({exc})"]

    outcomes_path = os.path.join(path, "shard_outcomes.json")
    if not os.path.exists(outcomes_path):
        problems.append(
            f"{name}: no shard_outcomes.json, so this sweep's completeness is "
            "unrecorded; an absent record is not evidence of a complete arm"
        )
    else:
        try:
            outcomes = json.loads(pathlib.Path(outcomes_path).read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: shard_outcomes.json is unreadable ({exc})")
            outcomes = None
        if outcomes is not None:
            n_failed = int(outcomes.get("n_failed") or 0)
            n_done = int(outcomes.get("n_completed") or 0)
            n_req = int(outcomes.get("n_requested") or 0)
            if n_failed or n_done != n_req:
                problems.append(
                    f"{name}: {n_failed} shard(s) FAILED and {n_done} of {n_req} "
                    "completed; an incomplete arm must not be scored as a "
                    "finished one"
                )

    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        problems.append(f"{name}: no .parquet shards")
    if problems:
        return None, problems

    # A shard that cannot be read is an arm-level refusal, not a crash:
    # unguarded, a truncated or half-written Parquet would exit 1
    # with a raw traceback, a code this script's own docstring never names.
    # It is refused rather than skipped, because dropping the shard would
    # silently score a smaller arm than the one that was collected.
    try:
        from lmopt.analysis.shards import load_shards

        df = load_shards(files)
    except Exception as exc:  # - polars raises several types here
        return None, [
            f"{name}: a shard could not be read ({type(exc).__name__}: {exc}); "
            f"scoring the shards that DO read would report a smaller arm than "
            f"the one that was collected"
        ]

    probe = resolved.get("probe") or {}
    meta_probe = probe.get("type", "none")
    meta_variant = probe.get("prompt_variant")
    if meta_variant is None:
        from lmopt.prompting.probe_variants import expected_variant_name

        meta_variant = expected_variant_name(meta_probe)

    for column, expected in (
        ("probe_type", meta_probe),
        ("prompt_variant", meta_variant),
    ):
        if column not in df.columns:
            problems.append(
                f"{name}: rows carry no {column!r} column, so the arm's own "
                "provenance cannot be checked against them (an arm older than the column)"
            )
            continue
        found = sorted(set(df[column].to_list()), key=lambda v: (v is None, str(v)))
        if found != [expected]:
            problems.append(
                f"{name}: run_meta.json says {column}={expected!r} but the rows "
                f"carry {found}; the arm's provenance contradicts its data"
            )

    try:
        actor = _actor_of(resolved)
    except ScriptError as exc:
        return None, [f"{name}: {exc}"]
    if "actor" not in df.columns:
        problems.append(f"{name}: rows carry no 'actor' column (an arm older than the column)")
    elif actor not in set(df["actor"].to_list()):
        problems.append(
            f"{name}: no row was written by the scored agent {actor!r}; the "
            f"arm's actors are {sorted(set(df['actor'].to_list()))}"
        )

    # Shape-checked once here, so a garbage seat_mode/drive_openings is a
    # Named refusal rather than a value that flows unguarded into `Arm` and
    # from there into probe_scores.py's live P6 scoring path
    # (`_run_provenance` -> `sentinel_anchors.anchor_for`).
    try:
        seat_mode = require_seat_mode_field(resolved, context=name)
        drive = parse_drive_openings_field(resolved.get("drive_openings"), context=name)
    except ResolvedConfigShapeError as exc:
        return None, [f"{name}: {exc}"]
    # `if drive:` would treat a
    # Recorded-but-empty `drive_openings=[]` the same as `drive_openings`
    # never having been recorded at all (`None`), silently skipping this
    # whole check #5 -- the identical "claim from the meta alone, never
    # checked" shape the missing-columns branch just below refuses.
    # `is not None` distinguishes "recorded" from
    # "absent"; an empty list, once recorded, is refused explicitly rather
    # than silently read as "undriven".
    if drive is not None:
        if not drive:
            problems.append(
                f"{name}: run_meta.json records drive_openings={drive!r} -- an "
                "explicit empty value recorded as a forced opening is not a "
                "valid drive (indistinguishable from 'undriven' by a bare "
                "truthiness check, which is exactly the bypass this check "
                "refuses); refused rather than silently treated as no drive"
            )
        else:
            # Refuse loudly rather than
            # silently skip check #5 when a driven arm's rows lack the columns
            # the mismatch check reads: the same "claim from the meta alone"
            # shape #5's own docstring above describes for the case
            # `drive_openings` fires with nothing behind it. Mirrors the
            # `probe_type`/`prompt_variant` missing-column branch above.
            missing = [c for c in ("info_set", "game") if c not in df.columns]
            if missing:
                problems.append(
                    f"{name}: drive_openings={drive!r} recorded but rows "
                    f"carry no {missing} column(s), so the forced-opening claim "
                    "cannot be checked against the data"
                )
            else:
                problems.extend(_drive_mismatch_problems(name, df, drive))

    if problems:
        return None, problems

    return (
        Arm(
            role=ROLE_BASE if meta_probe == "none" else meta_probe,
            name=name,
            path=path,
            df=df,
            actor=actor,
            probe_type=meta_probe,
            prompt_variant=meta_variant,
            # `drive` cannot be a falsy-but-non-None value here: the empty-value
            # branch above already refused and returned before this point.
            drive_openings=drive,
            seat_mode=seat_mode,
            cfr_iters=int(resolved.get("cfr_iters", 1000)),
            cfr_cache_dir=resolved.get("cfr_cache_dir", "data/cfr_policies"),
            n_rows=df.height,
            configured_games=tuple(resolved.get("games") or ()),
        ),
        [],
    )


def discover_arms(pilot_dir: str) -> tuple[dict[str, Arm], list[str]]:
    """Every arm under `pilot_dir`, keyed by role.

    A directory with no `run_meta.json` at all is not an arm and is skipped in
    silence (a pilot root legitimately holds logs); an arm that is one and
    fails a check is refused loudly. Two arms claiming one role refuses both:
    the pair cannot be told apart, and scoring either would pick one manipulation
    over another by directory order.
    """
    if not os.path.isdir(pilot_dir):
        raise ScriptError(f"{pilot_dir} is not a directory")

    arms: dict[str, Arm] = {}
    problems: list[str] = []
    # A role, once contested, stays contested. Deleting the key and
    # continuing would refuse only the first two claimants: with three
    # directories claiming one role, the third would find the key absent and
    # be installed as that role's arm, scored, and printed in the header as
    # though it were the only candidate -- directory order deciding which
    # manipulation the table reports. The docstring's "refuses BOTH" holds
    # for all claimants.
    contested: set[str] = set()
    for entry in sorted(os.listdir(pilot_dir)):
        path = os.path.join(pilot_dir, entry)
        if not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, "run_meta.json")):
            continue
        arm, arm_problems = read_arm(path)
        problems.extend(arm_problems)
        if arm is None:
            continue
        if arm.role in contested:
            problems.append(
                f"{arm.name} also claims the already-contested {arm.role!r} "
                "role; every claimant is refused"
            )
            continue
        if arm.role in arms:
            problems.append(
                f"{arm.name} and {arms[arm.role].name} both claim the "
                f"{arm.role!r} role; one arm per role, or the table would pick "
                "one manipulation by directory order"
            )
            contested.add(arm.role)
            del arms[arm.role]
            continue
        arms[arm.role] = arm
    return arms, problems
