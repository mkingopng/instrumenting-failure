"""Match runner - Iterated-Kuhn play loop with per-node Parquet logging."""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import subprocess
import sys
import uuid
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


def _tqdm(it, **kw):
    """
    the iterable wrapped in a progress bar when tqdm is installed, else unchanged
    :param it: the iterable (iterable)
    :param kw: tqdm's keyword arguments (dict)
    :returns: the iterable, wrapped when tqdm is available
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return it
    return tqdm(it, **kw)


import numpy as np
import pyspiel

from lmopt.agents.base import Agent
from lmopt.agents.cfr_ref import CFRRefAgent
from lmopt.agents.lm_vllm import LMVllmAgent
from lmopt.elicitation.posterior import compute_true_posterior_with_provenance
from lmopt.logging.schema import (
    BASE_PROBE_TYPE,
    OFFPATH_PROBE_TYPE,
    PROBE_TYPE_DOMAIN,
    NodeRecord,
)
from lmopt.logging.writer import MatchWriter

# Which seat agent0 (the agent under test) occupies:
#   fixed0  - seat 0 every hand. The default, so arms collected under it
#             stay reproducible byte for byte.
#   fixed1  - seat 1 every hand. A driven diagnostic, not a collection mode.
#   random  - drawn per hand, Bernoulli(0.5), from the seat stream below.
#   alternate - seat 0 on even hands, seat 1 on odd (`hand_idx % 2`). The
#             seat mode every citable collection runs. Deterministic and exactly 50/50 at an even
#             n_hands, where `random`'s realised split is itself a nuisance
#             quantity of the arm; and it consumes no entropy, so the seat
#             stream stays untouched (see SEAT_SALT: that derivation is frozen
#             for the arms that do draw from it).
# An unknown value raises: a typo that quietly ran fixed0 would label a whole
# arm as alternating with nothing in the Parquet to contradict it.
SEAT_MODES = ("fixed0", "fixed1", "random", "alternate")

# The seat stream's own entropy space, and the one number here that is
# load-bearing rather than arbitrary. `seat_seed=None` roots the seat draw in
# the MATCHUP seed, and that integer is already two other things: the root
# orchestration/shard.py's `derive_stream_seeds` spawns its positional
# consumers off (deal, opponent, reference, agent0, seat_seed), and, on the
# direct-call path where `deal_seed` is None, the deal generator's own seed
# (`default_rng(seed)`). A second entropy word keeps the seat draw off both.
#
# Why A salt and not A spawn index. A spawn position pinned above shard.py's
# `N_STREAMS_PER_MATCHUP` would force the two constants to move together, and
# shard.py's is designed to grow: every new stream would force this number up
# as routine maintenance. A salt is a fixed word, so `N_STREAMS_PER_MATCHUP`
# can grow without touching anything here. The value below is the
# ASCII bytes of "SEAT", picked only to be nowhere near a small spawn index.
#
# What changing it costs, which is why it must not change: the seat
# sequence an arm reproduces from its recorded seed is a function of this
# derivation. Change the salt, or how the stream is rooted, and every arm
# already collected under seat_mode="random" replays a different seat sequence
# from the same logged seed, with a fully green suite endorsing it. Once a
# random-seat arm exists, this derivation is frozen, and a new seat behaviour
# needs a new named mode rather than a new number here.
SEAT_SALT = 0x53454154  # b"SEAT"


def _seat_stream(seat_seed: int) -> np.random.Generator:
    """Generator for the per-hand seat draw, rooted in its own entropy space.

    ``SeedSequence([seat_seed, SEAT_SALT])``: the salt is what makes this
    stream disjoint from the bare-root stream (the deal, on the direct-call
    path) and from every child ``shard.py`` spawns off the same root. See
    ``SEAT_SALT`` for why it is a salt rather than a spawn index, and for what
    changing it would cost.

    Integer ``seat_seed`` only. Accepting an already-spawned ``SeedSequence``
    would be wrong two ways: ``spawn``
    mutates the argument (it advances ``n_children_spawned``), so the function
    would not be idempotent and a second call on the same object would return a
    different stream; and the child would land at a nested spawn key rather
    than the documented position. Orchestration derives an
    integer seat seed from its own stream mechanism instead and passes
    that integer here.
    """
    return np.random.default_rng(np.random.SeedSequence([seat_seed, SEAT_SALT]))


def run_match(
    game: pyspiel.Game,
    agent0: Agent,
    agent1: Agent,
    cfr_ref: CFRRefAgent,
    n_hands: int,
    seed: int,
    writer: MatchWriter,
    run_id: Optional[str] = None,
    game_params: Optional[str] = None,
    oracle_policy=None,
    game_key: Optional[str] = None,
    deal_seed=None,
    seat_mode: str = "fixed0",
    seat_seed: Optional[int] = None,
    probe_type: str = BASE_PROBE_TYPE,
    prompt_variant: str = "base",
    drive_openings: Optional[Sequence[int]] = None,
    posterior_fallback_out: Optional[dict[str, dict[str, int]]] = None,
    driven_discards_out: Optional[dict[str, dict[str, int | list[int]]]] = None,
    git_sha: Optional[str] = None,
) -> list[NodeRecord]:
    """Play n_hands episodes and log one NodeRecord per decision node.

    cfr_ref is queried at every decision node to populate cfr_action/cfr_policy.

    oracle_policy: if provided, true_posterior is computed for all rows via
    compute_true_posterior(oracle_policy) - used when the LM is one of the
    agents. When None, true_posterior falls back to agent.belief().

    game_key: GAMES registry key for the Parquet 'game' column (depth variants,
    e.g. "darkhex_3x3"). When None, falls back to the short-name map - identical
    for the canonical five.

    seat_mode: which seat agent0 occupies (SEAT_MODES above). Under "random" it
    is redrawn every hand, and under "alternate" it is `hand_idx % 2`; either
    way it is logged per row as `lm_seat`, so a seat effect becomes a
    within-match factor instead of a constant of the arm. It also decides which
    seats' Goofspiel/Dark Hex true_posterior maps are precomputed:
    both under "random" and "alternate", the occupied seat alone under a fixed
    mode, since each map costs a full game-tree walk and only the maps built are
    ever read.

    seat_seed: integer entropy for the seat draws, defaulting to `seed`. Always
    spawned into its own stream (_seat_stream), never the deal rng, so a fixed0
    arm and a random arm at the same seed play the same deals. An orchestration
    shard derives this integer from its own stream mechanism and passes the
    integer, never a SeedSequence.

    probe_type / prompt_variant: the run-level labels written to every row
    (config over forks). probe_type is validated against
    PROBE_TYPE_DOMAIN here. prompt_variant stays a free string here and is
    validated where an arm is configured: `MatchupSpec.__post_init__` checks it
    against the imported registry (`prompting.probe_variants`) and against the
    probe_type it must agree with, and `run_sweep.py` checks the same at sweep
    startup. This function does not resolve variant specs
    either -- the agent it is handed already renders with whatever content the
    caller resolved, and re-deriving content from a label here would let the
    label and the rendering disagree.

    drive_openings: the P6 off-path driver. A list of
    forced opening actions applied at the start of every hand, before any agent
    chooses, so that the whole arm is played from inside one chosen subgame;
    the targets are low-reach openings under
    the equilibrium reference. Full semantics in
    `_apply_drive` below: chance nodes are still drawn from the match rng
    between the forced plies, the forced plies are not logged (no agent chose
    them), and an illegal or hand-ending drive raises rather than being skipped.
    Paired with `probe_type` in both directions -- driving without the label
    would collect a manipulated arm indistinguishable from canonical play, and
    the label without driving is the fabricated-probe hazard closed at the config
    layer.

    posterior_fallback_out: the sink for this match's per-game posterior-fallback
    tally, shape `{game: {"fell_back": int, "rows": int}}`. A caller that wants
    the tally passes a dict and reads it after the call; a caller that does not
    passes nothing and the tally is discarded. An explicit out-parameter, not a
    module global: a global is
    written in whichever process ran the match, so a shard dispatched into a
    `ProcessPoolExecutor` worker would leave the parent's copy empty and the
    counts would reach no run_meta.json at all. The out-parameter makes the read
    same-process by construction, and the crossing of the process boundary an
    explicit act at the layer that owns it (`orchestration/shard.py` writes the
    per-shard sidecar `run_sweep.py` aggregates). Updated once, at the end of a
    successful match, so a raising match leaves the caller's dict untouched
    rather than half-filled.

    driven_discards_out: the sink for the driven-hand discard tally, shape
    `{game: {"driven_hands": int, "discarded_hands": int, "discarded_hand_idx":
    [int, ...]}}`, same out-parameter contract as `posterior_fallback_out` above.
    `discarded_hand_idx` lists the `hand_idx` of each
    discarded hand, in order, so its length is `discarded_hands`; `hand_idx` restarts
    at 0 per shard, and a discarded hand has no row, so the shard, the game and this
    index are the only identity it has. A
    "discarded" hand is a driven hand that produced zero logged rows for the
    agent under test: `_apply_drive` is seat-agnostic in mechanism but
    seat-fixing in consequence (see its docstring),
    so under seat_mode="random" -- or a fixed seat on the wrong side of the
    drive's parity -- the agent's hands at the drive-consumed seat contribute
    no row at all (e.g. a driven pilot with 60/60 rows at seat 1 and 24
    seat-0 hands with no row); this tally makes that visible.
    Left empty on an undriven match ("not a driven arm" is a different fact
    from a driven arm's measured zero). Independent of the sink, a nonzero
    count prints one loud summary line (no silent fallbacks); the
    driving behaviour itself is unaffected.

    git_sha: the sha stamped onto every row's `model_revision` column. If
    each shard calls `_get_git_sha()` -- a `git rev-parse HEAD` subprocess --
    inside its own worker process, at whatever wall-clock moment it happens
    to run, a checkout whose HEAD moves mid-sweep splits one arm between
    shas (a 100-shard Qwen arm split 80/20 this way). None (the default, and
    every direct test):
    this function resolves its own sha via `_get_git_sha()`, which is correct
    for a single serial match but is exactly the per-call resolution that
    races under a multi-shard, multi-hour sweep. A caller that dispatches many
    shards (`orchestration/shard.py`'s `run_matchup_shard`) must instead
    resolve the sha once, before any shard is built, and pass the same value
    here for every shard -- the identical "resolved once at startup, threaded
    through `MatchupSpec`" pattern already used for `chat_template_hash` and
    `served_weights_revision`.
    """
    drive_openings = _validate_match_config(seat_mode, probe_type, drive_openings)
    run_id = run_id or str(uuid.uuid4())
    # Git-HEAD race: resolve only when the caller
    # did not already resolve one. A caller with many shards must pass the
    # Same sha to every one (see the `git_sha` docstring above); calling
    # `_get_git_sha()` fresh here on every invocation races.
    git_sha = git_sha if git_sha is not None else _get_git_sha()
    game_col, family = _resolve_game_column(game, game_key)
    maps = _precompute_posteriors(game, family, oracle_policy, seat_mode)

    # `seed` stays the matchup-seed identity and is logged as such
    # on every row, so it must remain the plain integer.
    # `deal_seed` is the entropy the chance nodes actually draw from -- a stream
    # spawned off that identity by orchestration/shard.py's derive_stream_seeds,
    # so the deal cannot share entropy with the opponent's mixing in any shard of
    # the sweep. Defaults to `seed`, so the
    # other call sites (gate parity runs, tests) need
    # not pass it; only a sweep shard does.
    ctx = _MatchContext(
        game=game,
        cfr_ref=cfr_ref,
        oracle_policy=oracle_policy,
        maps=maps,
        drive_openings=drive_openings,
        rng=np.random.default_rng(seed if deal_seed is None else deal_seed),
        game_col=game_col,
        run_id=run_id,
        matchup_id=f"{_agent_id(agent0)}_vs_{_agent_id(agent1)}",
        game_params=game_params or "{}",
        seed=seed,
        probe_type=probe_type,
        prompt_variant=prompt_variant,
        git_sha=git_sha,
    )
    all_records, discarded_hand_idx = _play_hands(
        ctx,
        (agent0, agent1),
        n_hands,
        seat_mode,
        seed if seat_seed is None else seat_seed,
        writer,
    )

    # Hand the tally to the
    # caller's own dict instead of a module global, so the count survives the
    # process boundary a sweep shard is dispatched across (see the
    # `posterior_fallback_out` docstring paragraph above).
    if posterior_fallback_out is not None:
        posterior_fallback_out.update(ctx.fallback_counts)
    if drive_openings is not None:
        _report_driven_discards(
            ctx, discarded_hand_idx, n_hands, seat_mode, driven_discards_out
        )
    return all_records


@dataclasses.dataclass(frozen=True)
class _PosteriorMaps:
    """
    the per-seat true-posterior maps precomputed once per match
    :param goofspiel: seat -> info string -> bid distribution, or None
    :param darkhex: seat -> info string -> cell-occupancy marginal, or None
    :param darkhex_partial: the Dark Hex reference is partial, so true_posterior stays null
    """

    goofspiel: Optional[dict[int, dict]]
    darkhex: Optional[dict[int, dict]]
    darkhex_partial: bool


@dataclasses.dataclass(frozen=True)
class _MatchContext:
    """
    everything constant across the hands of one match, plus its fallback tally
    :param game: the OpenSpiel game being played
    :param cfr_ref: the reference queried at every decision node
    :param oracle_policy: the policy true_posterior is computed against, or None
    :param maps: the precomputed per-seat posterior maps
    :param drive_openings: forced opening actions applied every hand, or None
    :param rng: the deal generator
    :param game_col: the value written to the game column
    :param run_id: the run id written to every row
    :param matchup_id: the matchup id written to every row
    :param game_params: the game_params string written to every row
    :param seed: the matchup-seed identity written to every row
    :param probe_type: the run-level probe label written to every row
    :param prompt_variant: the run-level prompt label written to every row
    :param git_sha: the sha written to every row's model_revision
    :param fallback_counts: per-game posterior-fallback tally, filled as rows are built
    """

    game: pyspiel.Game
    cfr_ref: CFRRefAgent
    oracle_policy: object
    maps: _PosteriorMaps
    drive_openings: Optional[list[int]]
    rng: np.random.Generator
    game_col: str
    run_id: str
    matchup_id: str
    game_params: str
    seed: int
    probe_type: str
    prompt_variant: str
    git_sha: str
    fallback_counts: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)


def _validate_match_config(
    seat_mode: str, probe_type: str, drive_openings: Optional[Sequence[int]]
) -> Optional[list[int]]:
    """
    refuse an unknown seat mode or probe type and a drive/label mismatch
    :param seat_mode: the requested seat mode, one of SEAT_MODES
    :param probe_type: the requested probe label, one of PROBE_TYPE_DOMAIN
    :param drive_openings: forced opening actions, or None; an empty list counts as None
    :returns: the opening actions as a list, or None when the match is not driven
    """
    if seat_mode not in SEAT_MODES:
        raise ValueError(
            f"unknown seat_mode {seat_mode!r}; expected one of {SEAT_MODES}"
        )
    if probe_type not in PROBE_TYPE_DOMAIN:
        raise ValueError(
            f"unknown probe_type {probe_type!r}; expected one of "
            f"{sorted(PROBE_TYPE_DOMAIN)}"
        )
    # The manipulation and its label are one claim, checked both ways. An empty
    # list counts as no drive: `[]` reading as "not driving" would relabel a
    # canonical arm through a config typo.
    if drive_openings is not None and not list(drive_openings):
        drive_openings = None
    if drive_openings is not None and probe_type != OFFPATH_PROBE_TYPE:
        raise ValueError(
            f"drive_openings={list(drive_openings)!r} was given with probe_type "
            f"{probe_type!r}; a driven match must be labelled "
            f"{OFFPATH_PROBE_TYPE!r} or its rows claim to be canonical play."
        )
    if drive_openings is None and probe_type == OFFPATH_PROBE_TYPE:
        raise ValueError(
            f"probe_type {OFFPATH_PROBE_TYPE!r} was given with no (or an empty) "
            f"drive_openings; that would collect an ordinary arm whose every row "
            f"asserts a manipulation that never happened."
        )
    return list(drive_openings) if drive_openings is not None else None


def _resolve_game_column(
    game: pyspiel.Game, game_key: Optional[str]
) -> tuple[str, str]:
    """
    resolve the game column value and the family the match's behaviour keys on
    :param game: the OpenSpiel game being played
    :param game_key: GAMES registry key, or None for the short-name fallback
    :returns: (game column value, game family)
    """
    from lmopt.games.families import GAME_FAMILIES

    # Normalise the game column to our registry key names. game_key (the GAMES
    # registry key) wins so depth variants are distinguishable in the Parquet;
    # the short-name map is the legacy fallback and
    # yields the same value for the canonical five. A provided game_key must be
    # a real registry key: a typo'd key would both mislabel the Parquet and -
    # via the family fallback below - silently skip its family's precompute.
    if game_key is not None and game_key not in GAME_FAMILIES:
        raise ValueError(
            f"unknown game_key {game_key!r}; expected a GAMES registry key "
            f"(one of {sorted(GAME_FAMILIES)})"
        )
    _short_name_map = {
        "kuhn_poker": "kuhn",
        "leduc_poker": "leduc",
        "turn_based_simultaneous_game": "goofspiel",
        "liars_dice": "liarsdice",
        "dark_hex": "darkhex",
    }
    game_col = game_key or _short_name_map.get(
        game.get_type().short_name, game.get_type().short_name
    )
    # Behavior branches key on the family, never the raw key - a variant key
    # ("darkhex_3x3") must not silently miss its family's precompute.
    # .get with fallback-to-self (not the raising game_family()) so a foreign
    # OpenSpiel game with no registry entry still runs; registry keys are safe:
    # game_key is validated above and the map covers every GAMES key.
    return game_col, GAME_FAMILIES.get(game_col, game_col)


def _posterior_seats(seat_mode: str) -> tuple[int, ...]:
    """
    the seats whose posterior maps a match in this seat mode needs
    :param seat_mode: one of SEAT_MODES
    :returns: the seats, (0, 1) when the agent under test can sit at either
    """
    # The maps are
    # keyed on one seat's information states, so a match in which the agent under
    # test can sit at either seat needs both; a fixed mode reads only the seat it
    # occupies, and the second walk is a full game-tree enumeration (the Dark Hex
    # comment below records a MemoryError after 35.6 s at 3x3), so it is not paid
    # where it would never be read. Under fixed0 this is the set {0}.
    #
    # Every member of SEAT_MODES is named explicitly and an unnamed one raises.
    # A bare `else` meaning fixed1 would run a newly appended mode
    # with seat-1's map alone and nothing in the Parquet to contradict the
    # label. Unreachable for a value outside SEAT_MODES (the membership check
    # in _validate_match_config already raised); reachable exactly for the defect it names.
    if seat_mode in ("random", "alternate"):
        return (0, 1)
    if seat_mode == "fixed0":
        return (0,)
    if seat_mode == "fixed1":
        return (1,)
    raise ValueError(
        f"seat_mode {seat_mode!r} is in SEAT_MODES but has no posterior-seat "
        "rule; add its branch here and in the per-hand seat rule in _seat_for_hand"
    )


def _precompute_posteriors(
    game: pyspiel.Game,
    family: str,
    oracle_policy,
    seat_mode: str,
) -> _PosteriorMaps:
    """
    walk the game tree once per seat for the families whose true posterior is a map
    :param game: the OpenSpiel game being played
    :param family: the game family the behaviour keys on
    :param oracle_policy: the reference policy, or None (nothing is precomputed)
    :param seat_mode: one of SEAT_MODES, deciding which seats' maps are built
    :returns: the Goofspiel and Dark Hex maps, each None when not built
    """
    posterior_seats = _posterior_seats(seat_mode)
    # Goofspiel P7 (Option A): the "true posterior" is a predictive distribution
    # over the opponent's current-round bid, not a dealt-card posterior. Precompute
    # every node's reach-weighted bid posterior in one tree walk per seat (per node
    # would re-walk the whole tree). Only when an oracle is supplied and P7 is
    # enabled.
    goofspiel_maps: Optional[dict[int, dict]] = None
    if family == "goofspiel" and oracle_policy is not None:
        from lmopt.scoring.goofspiel_posterior import (
            GOOFSPIEL_P7_ENABLED,
            goofspiel_all_posteriors,
        )

        if GOOFSPIEL_P7_ENABLED:
            goofspiel_maps = {
                s: goofspiel_all_posteriors(game, oracle_policy, seat=s)
                for s in posterior_seats
            }

    # Dark Hex P8: the "true posterior" is the per-cell opponent-occupancy marginal
    # (independent Bernoullis, not a simplex). Precompute per info-state in one
    # tree walk per seat; logged on that seat's nodes for the board belief / P8b.
    #
    # gather_worlds does a full game-tree walk
    # (worse than TabularPolicy - it also deserialises a full state object at
    # every observer node), so it is exactly as intractable as a CFR+ solve
    # of darkhex_3x3 (MemoryError after 35.6s under a 4GB cap). A darkhex_3x3
    # key reaches this branch: `family` resolves to "darkhex" (families.py)
    # and shard.py's own routing supplies a non-None oracle_policy
    # (a DepthReference). Guard on is_partial_reference the same way P7b/P8 do
    # - the same rule at a third call site, not a new policy.
    darkhex_maps: Optional[dict[int, dict]] = None
    darkhex_reference_is_partial = False
    if family == "darkhex" and oracle_policy is not None:
        from lmopt.scoring._reference import is_partial_reference

        if is_partial_reference(oracle_policy):
            # No silent fallbacks: true_posterior is left unset
            # in the hand loop (never fabricated, never NaN) - coherent with P7b/P8
            # being "not applicable at 3x3, reason recorded".
            darkhex_reference_is_partial = True
        else:
            from lmopt.scoring import darkhex_posterior as _dh

            darkhex_maps = {}
            for _seat in posterior_seats:
                _dh_worlds = _dh.gather_worlds(game, oracle_policy, seat=_seat)
                darkhex_maps[_seat] = {
                    info: _dh.cell_occupancy_posterior(wl, game, seat=_seat)
                    for info, wl in _dh_worlds.items()
                }
    return _PosteriorMaps(goofspiel_maps, darkhex_maps, darkhex_reference_is_partial)


def _hand_bar_kwargs() -> dict[str, float]:
    """
    tqdm keyword arguments for the per-hand progress bar
    :returns: a 10 s mininterval on a non-TTY stderr, else empty
    """
    # On an interactive terminal tqdm's default 0.1s
    # mininterval is fine, but a redirected (non-TTY) run -- the project's
    # `nohup ... > log 2>&1` cycle -- has no cursor to overwrite, so every
    # redraw is a literal `\r`-separated line and a multi-hour sweep buries
    # its per-shard timing line (the scheduling source of truth)
    # under megabytes of them. Throttle
    # Only the non-TTY case. NOTE this bar is `leave=False`: its close()
    # Clears the line rather than flushing a completed one (tqdm 4.67.3),
    # so no per-hand completed line reaches a non-TTY log
    # either way -- that survives-close guarantee holds only
    # for the `leave=True` matchup_bar below, which is the per-shard
    # timing source of truth. 10s: a hand loop runs seconds to minutes, so
    # this caps a long shard's log to a handful of lines while a rate
    # collapse mid-shard still surfaces within one interval.
    return {} if sys.stderr.isatty() else {"mininterval": 10.0}


def _set_call_log(agents: Sequence[Agent], *args) -> None:
    """
    point each agent that can log API calls at a call log, or detach it
    :param agents: the agents of the match
    :param args: the arguments `attach_call_log` takes: (call_log, run_id) to attach, (None,) to detach
    :returns: None; an agent with no `attach_call_log` is skipped
    """
    for attached in agents:
        attach = getattr(attached, "attach_call_log", None)
        if attach is not None:
            attach(*args)


def _seat_for_hand(
    seat_mode: str, hand_idx: int, seat_rng: Optional[np.random.Generator]
) -> int:
    """
    the seat agent0 occupies this hand
    :param seat_mode: one of SEAT_MODES
    :param hand_idx: the hand's index within the match
    :param seat_rng: the seat stream, present exactly when seat_mode is "random"
    :returns: 0 or 1
    """
    # This hand's seat for agent0 (the agent under test). `agents` is
    # indexed by player, so the swap happens in _play_hand and nowhere else: every
    # downstream use (act, actor, observe_hand_end, lm_seat) reads the map
    # rather than the argument order.
    # Every SEAT_MODES member named explicitly, no bare `else` (see
    # _posterior_seats for why): a mode with no rule here must
    # raise, not quietly play seat 1.
    if seat_mode == "random":
        assert seat_rng is not None  # built exactly when seat_mode == "random"
        return int(seat_rng.integers(2))
    if seat_mode == "alternate":
        return hand_idx % 2
    if seat_mode == "fixed0":
        return 0
    if seat_mode == "fixed1":
        return 1
    raise ValueError(
        f"seat_mode {seat_mode!r} is in SEAT_MODES but has no per-hand "
        "seat rule; add its branch here and in _posterior_seats"
    )


def _true_posterior(
    ctx: _MatchContext, agent: Agent, state: pyspiel.State, player: int
) -> tuple[Optional[str], Optional[bool]]:
    """
    the true_posterior column and its fallback flag for the acting player's node
    :param ctx: the match's constants and fallback tally
    :param agent: the acting agent, whose belief() is the posterior when no oracle exists
    :param state: the decision node
    :param player: the acting player
    :returns: (JSON posterior or None, fallback flag or None); the flag is set only on the oracle-policy branch
    """
    # Set only on the oracle-policy branch below, the one place a fallback
    # flag exists; null everywhere else, never a default False.
    posterior_fell_back: Optional[bool] = None
    true_posterior_str: Optional[str] = None
    if ctx.maps.goofspiel is not None:
        # Goofspiel: action-keyed bid posterior over the opponent of the
        # acting player. Keys are bid actions, aligning with the
        # action-keyed elicited_belief from prompting.goofspiel.parse_belief.
        #
        # The lookup is keyed on the acting player's own map,
        # never on agent0's seat. A map is present exactly for the seats
        # this match can seat the agent under test at, and each map's
        # keys are that seat's information strings, so a miss means the
        # seat was not precomputed and true_posterior stays None rather
        # than borrowing the other seat's population.
        seat_map = ctx.maps.goofspiel.get(player)
        if seat_map is not None:
            dist = seat_map.get(state.information_state_string(player))
            if dist:
                true_posterior_str = json.dumps(
                    {str(a): float(p) for a, p in dist.items()}
                )
    elif ctx.maps.darkhex is not None:
        # Dark Hex: cell-occupancy marginal for the acting player. Keys
        # are cell actions, aligning with prompting.darkhex.parse_belief.
        # Same per-seat lookup rule as the Goofspiel branch above.
        seat_map = ctx.maps.darkhex.get(player)
        if seat_map is not None:
            dist = seat_map.get(state.information_state_string(player))
            if dist:
                true_posterior_str = json.dumps(
                    {str(a): float(p) for a, p in dist.items()}
                )
    elif ctx.maps.darkhex_partial:
        # true_posterior stays None; must not fall
        # through to the generic oracle_policy branch below, which
        # assumes a dealt-card chance structure Dark Hex does not have
        # (compute_true_posterior calls state.chance_outcomes() on the
        # initial state, which raises for a game with zero leading
        # chance nodes; loaders.py's independent_deal docstring notes
        # Dark Hex has none). Skipping is coherent with P7b/P8 being
        # "not applicable at 3x3, reason recorded".
        pass
    elif ctx.oracle_policy is not None:
        tp, fell_back = compute_true_posterior_with_provenance(
            ctx.game, state, player, ctx.oracle_policy
        )
        # Tally the fallback per game
        tally = ctx.fallback_counts.setdefault(
            ctx.game_col, {"fell_back": 0, "rows": 0}
        )
        tally["rows"] += 1
        posterior_fell_back = bool(fell_back)
        if fell_back:
            tally["fell_back"] += 1

        # `compute_true_posterior`
        # substitutes a uniform array when every hypothesis's
        # likelihood is zero, and a driven opening is chosen precisely
        # for near-zero reach, so on a driven arm that substitution is
        # systematically reachable. P5 (ECE) and P7a (TVD) read this
        # column and already guard on null, so a driven fallback logs
        # None: a row that says nothing beats a row that says uniform
        # in the column labelled "exact Bayesian posterior"
        # (no silent fallbacks).
        #
        # Gated on driving, deliberately. On a non-driven hand the
        # fallback needs an opponent policy with an exact zero, which
        # the CFR+ reference does not produce on the collected games
        # (Kuhn min action probability 0.08),
        # so ungating it would change nothing measurable and would
        # still be a meaning change to a column every landed arm
        # carries. On non-driven hands the fallback is not nulled.
        if not (fell_back and ctx.drive_openings is not None):
            true_posterior_str = json.dumps(
                {str(i): float(p) for i, p in enumerate(tp)}
            )
    else:
        posterior = agent.belief(state)
        if posterior is not None:
            true_posterior_str = json.dumps(
                {str(i): float(p) for i, p in enumerate(posterior)}
            )
    return true_posterior_str, posterior_fell_back


def _node_record(
    ctx: _MatchContext,
    state: pyspiel.State,
    agent: Agent,
    action: int,
    agent0_seat: int,
    hand_idx: int,
    node_idx: int,
) -> NodeRecord:
    """
    build the logged row for one decision node
    :param ctx: the match's constants and fallback tally
    :param state: the decision node, before the action is applied
    :param agent: the acting agent
    :param action: the action it chose
    :param agent0_seat: the seat agent0 holds this hand
    :param hand_idx: the hand's index within the match
    :param node_idx: the node's index within the hand
    :returns: the row; reward is NaN until the hand ends
    """
    player = state.current_player()
    # LM metadata (non-LM agents return defaults via getattr).
    lm = _get_lm_metadata(agent)

    # CFR reference (logged regardless of which agent is acting).
    cfr_probs, cfr_off_support = ctx.cfr_ref.policy_at(state)
    # INVALID_ACTION (not an argmax) when off-support: a uniform
    # distribution's argmax is an arbitrary tie-break, and cfr_action is
    # documented as a deterministic reference. Fabricating one would be
    # the silent-fallback the flag exists to prevent.
    cfr_act = (
        pyspiel.INVALID_ACTION
        if cfr_off_support
        else int(max(cfr_probs, key=lambda a: cfr_probs[a]))
    )
    cfr_policy_str = json.dumps({str(k): v for k, v in cfr_probs.items()})
    # true_posterior: use oracle_policy when available, else agent.belief().
    true_posterior_str, posterior_fell_back = _true_posterior(ctx, agent, state, player)
    return NodeRecord(
        run_id=ctx.run_id,
        matchup_id=ctx.matchup_id,
        game=ctx.game_col,
        game_params=ctx.game_params,
        seed=ctx.seed,
        hand_idx=hand_idx,
        node_idx=node_idx,
        player_to_act=player,
        lm_seat=agent0_seat,
        actor=lm["model_id"] or _agent_id(agent),
        info_set=state.information_state_string(player),
        info_state_tensor=None,
        legal_mask=list(state.legal_actions_mask()),
        prompt=lm["prompt"],
        raw_completion=lm["raw_completion"],
        parsed_action=action,
        action_str=state.action_to_string(player, action),
        n_retries=lm["n_retries"],
        fallback_used=lm["fallback_used"],
        bet_aliased=lm["bet_aliased"],
        action_parse_vocab=lm["action_parse_vocab"],
        memory_rung=lm["memory_rung"],
        memory_truncated=lm["memory_truncated"],
        cfr_action=cfr_act,
        cfr_policy=cfr_policy_str,
        cfr_ref_off_support=cfr_off_support,
        elicited_belief=lm["elicited_belief"],
        true_posterior=true_posterior_str,
        belief_raw_completion=lm["belief_raw_completion"],
        belief_n_retries=lm["belief_n_retries"],
        belief_failed=lm["belief_failed"],
        reward=float("nan"),
        probe_type=ctx.probe_type,
        prompt_variant=ctx.prompt_variant,
        backend=getattr(agent, "backend", "cpu"),
        # model_revision is the harness git sha -- an established
        # meaning on 406k landed rows (schema.py's model_revision
        # comment). The served weights identity is the separate
        # served_weights_revision column below, never a redefinition
        # of this one.
        model_revision=ctx.git_sha,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_id=lm["model_id"],
        chat_template_hash=lm["chat_template_hash"],
        served_weights_revision=lm["served_weights_revision"],
        # Weights provenance: same thread, same
        # reasoning as served_weights_revision above.
        tokenizer_revision=lm["tokenizer_revision"],
        prompt_schema_version=lm["prompt_schema_version"],
        prompt_spec_hash=lm["prompt_spec_hash"],
        belief_protocol=lm["belief_protocol"],
        reasoning_trace=lm["reasoning_trace"],
        belief_reasoning_trace=lm["belief_reasoning_trace"],
        finish_reason=lm["finish_reason"],
        belief_finish_reason=lm["belief_finish_reason"],
        belief_prompt=lm["belief_prompt"],
        completion_tokens=lm["completion_tokens"],
        belief_completion_tokens=lm["belief_completion_tokens"],
        agent_node_idx=lm["agent_node_idx"],
        run_seed=lm["run_seed"],
        belief_salvage_kind=lm["belief_salvage_kind"],
        belief_salvage_attempt=lm["belief_salvage_attempt"],
        call_latency_s=lm["call_latency_s"],
        belief_call_latency_s=lm["belief_call_latency_s"],
        prompt_tokens=lm["prompt_tokens"],
        belief_prompt_tokens=lm["belief_prompt_tokens"],
        response_id=lm["response_id"],
        belief_response_id=lm["belief_response_id"],
        request_seed=lm["request_seed"],
        belief_request_seed=lm["belief_request_seed"],
        winning_attempt=lm["winning_attempt"],
        winning_raw_completion=lm["winning_raw_completion"],
        memory_chars_raw=lm["memory_chars_raw"],
        memory_chars_sent=lm["memory_chars_sent"],
        posterior_fell_back=posterior_fell_back,
    )


def _play_hands(
    ctx: _MatchContext,
    agents: tuple[Agent, Agent],
    n_hands: int,
    seat_mode: str,
    seat_seed: int,
    writer: MatchWriter,
) -> tuple[list[NodeRecord], list[int]]:
    """
    play every hand of the match, writing rows as each hand ends
    :param ctx: the match's constants and fallback tally
    :param agents: (the agent under test, its opponent)
    :param n_hands: hands to play
    :param seat_mode: one of SEAT_MODES
    :param seat_seed: integer entropy for the seat draws, used only under "random"
    :param writer: receives every row; flushed after the last hand
    :returns: (all rows, hand_idx of each driven hand with no row for the agent under test)
    """
    agent0, agent1 = agents
    drive_openings = ctx.drive_openings
    game_col, run_id = ctx.game_col, ctx.run_id
    # reset() stays a once-per-match call at seats 0/1, unchanged. It is not
    # the per-hand seat channel: LMVllmAgent.reset() clears the cross-hand
    # HandMemory and the RL/Bayesian agents rebuild per-match state there, so
    # calling it again each hand would silently make every LM agent memoryless.
    # begin_hand() below is the per-hand
    # channel, and it is called for both agents in every seat mode so the fixed
    # path exercises exactly the code the alternating path does.
    agent0.reset(ctx.game, seat=0, opponent_id=_agent_id(agent1))
    agent1.reset(ctx.game, seat=1, opponent_id=_agent_id(agent0))

    # Built only under "random": a fixed mode must not consume seat entropy at
    # all, which keeps fixed0 runs byte-reproducible.
    # "alternate" is a third non-drawing mode: its seat is a function of
    # hand_idx alone, so it touches the seat stream no more than fixed0 does.
    seat_rng = _seat_stream(seat_seed) if seat_mode == "random" else None
    all_records: list[NodeRecord] = []
    # Driven hands whose rows never include the agent
    # under test (see the driven_discards_out docstring paragraph above).
    discarded_hand_idx: list[int] = []

    # Direct each LM agent's API calls into the shard's call log under this run id. A
    # non-LM agent has nothing to attach.
    _set_call_log((agent0, agent1), writer.call_log, run_id)
    # The detach runs however the hands end, so an agent reused after a raising match
    # never keeps this shard's call log.
    try:
        for hand_idx in _tqdm(
            range(n_hands),
            desc=f"{game_col} hand",
            unit="hand",
            leave=False,
            **_hand_bar_kwargs(),
        ):
            agent0_seat = _seat_for_hand(seat_mode, hand_idx, seat_rng)
            hand_records = _play_hand(ctx, agent0, agent1, agent0_seat, hand_idx)
            # The exact detection is "a driven hand ending
            # with no rows for the tested agent's seat". Keyed on this hand's
            # agent0_seat, never a literal seat: under seat_mode="random" the
            # consumed seat is per-hand, and a fixed1 arm on the surviving side
            # of the parity must count zero.
            if drive_openings is not None and not any(
                rec.player_to_act == agent0_seat for rec in hand_records
            ):
                discarded_hand_idx.append(hand_idx)
            for rec in hand_records:
                writer.add(rec)
            all_records.extend(hand_records)

        writer.flush()
    finally:
        _set_call_log((agent0, agent1), None)
    return all_records, discarded_hand_idx


def _play_hand(
    ctx: _MatchContext,
    agent0: Agent,
    agent1: Agent,
    agent0_seat: int,
    hand_idx: int,
) -> list[NodeRecord]:
    """
    play one hand and return its rows with rewards filled in
    :param ctx: the match's constants and fallback tally
    :param agent0: the agent under test
    :param agent1: its opponent
    :param agent0_seat: the seat agent0 holds this hand
    :param hand_idx: the hand's index within the match
    :returns: one NodeRecord per decision node, in play order
    """
    agents = [agent0, agent1] if agent0_seat == 0 else [agent1, agent0]
    agent0.begin_hand(agent0_seat)
    agent1.begin_hand(1 - agent0_seat)

    state = ctx.game.new_initial_state()
    hand_records: list[NodeRecord] = []
    node_idx = 0

    # The forced opening, applied per hand (not once per match)
    # so every hand of the arm is played from the same subgame.
    if ctx.drive_openings is not None:
        _apply_drive(state, ctx.rng, ctx.drive_openings, hand_idx)

    while not state.is_terminal():
        if state.is_chance_node():
            state.apply_action(_draw_chance(state, ctx.rng))
            continue

        agent = agents[state.current_player()]
        action = agent.act(state)
        hand_records.append(
            _node_record(ctx, state, agent, action, agent0_seat, hand_idx, node_idx)
        )
        node_idx += 1
        state.apply_action(action)

    returns = state.returns()
    for rec in hand_records:
        rec.reward = float(returns[rec.player_to_act])

    action_history = _build_action_history(ctx.game, state)
    # Each agent gets its own terminal payoff.
    # `action_history` is shared and carries no payoff (it skips chance
    # nodes), while `returns` above is per-seat; passing `returns[0]` to
    # both would tell seat 1 it won when it lost, these being negatives of
    # each other in a zero-sum game.
    #
    # Indexed by the seat each agent HELD THIS hand, not by its
    # position in the argument list. Under a swap the literal returns[0]
    # would hand agent0 the exact negation of its own result, and the
    # cross-hand memory stores that number verbatim.
    agent0.observe_hand_end(action_history, returns[agent0_seat])
    agent1.observe_hand_end(action_history, returns[1 - agent0_seat])
    return hand_records


def _report_driven_discards(
    ctx: _MatchContext,
    discarded_hand_idx: list[int],
    n_hands: int,
    seat_mode: str,
    driven_discards_out: Optional[dict[str, dict[str, int | list[int]]]],
) -> None:
    """
    print the loud discard line and fill the sink for a driven match
    :param ctx: the match's constants; its game_col keys the sink and its drive_openings are named in the warning
    :param discarded_hand_idx: hand_idx of each driven hand with no row for the agent under test
    :param n_hands: hands played
    :param seat_mode: the match's seat mode, named in the warning
    :param driven_discards_out: the caller's sink, or None
    :returns: None
    """
    # Emit the driven-hand discard tally -- the count is
    # a per-match fact whether or not a caller passed a sink, so the loud line
    # prints regardless; the sink is only the machine-readable copy. Gated on
    # driving by the caller: an undriven match leaves the sink untouched, so {} keeps
    # meaning "not a driven arm" while a driven arm's zero is written out.
    n_discarded_hands = len(discarded_hand_idx)
    game_col = ctx.game_col
    if n_discarded_hands > 0:
        print(
            f"WARNING [driven-hand discard]: {n_discarded_hands}/{n_hands} "
            f"driven hands produced no row for the agent under test "
            f"(game={game_col}, seat_mode={seat_mode}, "
            f"drive_openings={ctx.drive_openings}): the drive's parity consumed "
            f"its seat, so the logged rows are a seat-conditioned subsample "
            f"(see match_runner._apply_drive).",
            flush=True,
        )
    if driven_discards_out is not None:
        driven_discards_out.update(
            {
                game_col: {
                    "driven_hands": n_hands,
                    "discarded_hands": n_discarded_hands,
                    "discarded_hand_idx": discarded_hand_idx,
                }
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def merge_posterior_fallback_counts(
    into: dict[str, dict[str, int]], other: dict[str, dict[str, int]]
) -> None:
    """Add `other`'s per-game tallies into `into`, in place.

    The aggregation step of the posterior-fallback sink: one
    function, used by `scripts/run_sweep.py` to pool every shard's sidecar into
    the run-level total. Additive per game and per key, so pooling N shards of
    one game gives that game's whole-sweep `fell_back` / `rows`, and a game only
    one shard played is not diluted by the games it did not.
    """
    for game, counts in other.items():
        target = into.setdefault(game, {"fell_back": 0, "rows": 0})
        for key, value in counts.items():
            target[key] = target.get(key, 0) + int(value)


def merge_driven_discard_counts(
    into: dict[str, dict[str, int]], other: dict[str, dict[str, int | list[int]]]
) -> None:
    """Add `other`'s per-game driven-discard tallies into `into`, in place.

    The discard twin of
    `merge_posterior_fallback_counts` above (a separate function because the
    seed keys differ, and seeding a discard tally with `fell_back`/`rows`
    would publish the wrong counters under the discard key). Additive per
    game and per key, so pooling a sweep's driven shards gives the arm's
    whole `discarded_hands` / `driven_hands`. A list-valued key
    (`discarded_hand_idx`) is not pooled: its indices are local to one shard, so a
    concatenation would name hands that two shards both call 0.
    """
    for game, counts in other.items():
        target = into.setdefault(game, {"driven_hands": 0, "discarded_hands": 0})
        for key, value in counts.items():
            if isinstance(value, list):
                continue
            target[key] = target.get(key, 0) + int(value)


def log_posterior_fallback_warning(counts: dict[str, dict[str, int]]) -> None:
    """Log a WARNING when any game had nonzero posterior fallbacks.

    Called by the sweep orchestrator (`scripts/run_sweep.py`,
    after the per-shard aggregation and before the run_meta update) to warn when
    the no-silent-fallbacks invariant was exercised. The fallback itself is not
    silent (it logs None on driven hands); this surfaces the count.
    """
    nonzero = {g: c for g, c in counts.items() if c.get("fell_back", 0) > 0}
    if nonzero:
        lines = [f"{g}: {c['fell_back']}/{c['rows']}" for g, c in nonzero.items()]
        logger.warning(
            f"Posterior fallback counts (fell_back/rows): {', '.join(lines)}"
        )


def log_driven_discard_warning(counts: dict[str, dict[str, int]]) -> None:
    """Log a WARNING when any game's driven arm discarded hands.

    The sweep-level twin of
    `log_posterior_fallback_warning` above: called by `scripts/run_sweep.py`
    after the per-shard aggregation, so a driven arm whose parity consumed the
    tested agent's seat announces the seat-conditioned subsample at the run
    level too, beside the per-match line `run_match` already printed.
    """
    nonzero = {g: c for g, c in counts.items() if c.get("discarded_hands", 0) > 0}
    if nonzero:
        lines = [
            f"{g}: {c['discarded_hands']}/{c['driven_hands']}"
            for g, c in nonzero.items()
        ]
        logger.warning(
            "Driven-hand discards (no row for the agent under test; "
            f"discarded/driven): {', '.join(lines)}"
        )


def _draw_chance(state: pyspiel.State, rng: np.random.Generator) -> int:
    """Sample one chance outcome at `state` from the match's deal stream.

    One function, two callers (the play loop and the off-path driver), so a
    driven hand cannot end up drawing its deal from a different mechanism than
    an undriven one.
    """
    actions, probs = zip(*state.chance_outcomes(), strict=True)
    return int(rng.choice(list(actions), p=list(probs)))


def _apply_drive(
    state: pyspiel.State,
    rng: np.random.Generator,
    drive_openings: list[int],
    hand_idx: int,
) -> None:
    """Force this hand's opening plies, in place, before any agent chooses.

    Semantics, in full:

    - Chance stays chance. Before each forced action the state is advanced
      through any chance nodes by drawing from the match's own deal stream, so
      a driven arm sees the ordinary deal distribution inside the subgame. Only
      player plies are forced; a drive that could pin the deal would make P6 a
      statement about one deal.
    - **Seat-agnostic in mechanism; seat-fixing in consequence. Read this
      before configuring a random-seat driven arm.** The plies are forced at
      whichever seat is to act, so nothing here hard-codes a player. But the
      opening's parity then fixes which seat acts first in the subgame, and
      that seat is the same on every hand because the drive is the same on
      every hand. Combined with "nothing forced is logged" below, a hand in
      which the agent under test occupied the seat the drive consumed
      contributes no row at all.

      For example, on a driven pilot at ``seat_mode="random"``: **all
      60 logged rows carry ``player_to_act == 1``**, and the 24 hands in which
      the agent sat seat 0 produced nothing. Each row records the seat it was
      played from, but without a count the discarded hands are silent, so
      the surviving rows are a seat-conditioned subsample with nothing in the
      Parquet to reveal it.

      A P6 arm must therefore either drive into a subgame deep enough
      that both seats still decide, or use a fixed seat and say so. If
      random-seat driving is used, the discarded-hand count has to be emitted.

      `run_match` counts the
      driven hands that produced no row for the agent under test, prints one
      loud summary line when the count is nonzero, and reports the tally
      through its `driven_discards_out` sink (sidecar-per-shard, pooled into
      run_meta.json, same surface as the posterior-fallback tally). The pooled
      counts are the integer keys only; `discarded_hand_idx` stays in each shard's
      sidecar, because `hand_idx` restarts per shard.
    - Nothing forced is logged. No agent chose these actions, so a row for
      them would be a fabricated decision; ``node_idx`` starts at 0 at the
      first post-drive decision node. The plies are still in the state's
      history, so terminal payoffs and the ``observe_hand_end`` action history
      are the whole hand's.
    - Illegal or hand-ending drives raise. Refused rather than defaulted:
      skipping an illegal ply would play the hand from a different
      subgame (or from the root) while every row asserts the drive, and on a
      game whose legality depends on the deal the surviving hands would be a
      deal-conditioned subsample -- a selection effect with nothing in the
      Parquet to reveal it. For the canonical five, action legality is a
      function of the action prefix and not of the deal, so a bad drive fails
      on hand 0 and costs nothing; the hand index rides in the message for the
      case where it does not.
    """
    for i, action in enumerate(drive_openings):
        while state.is_chance_node():
            state.apply_action(_draw_chance(state, rng))
        if state.is_terminal():
            raise ValueError(
                f"drive_openings={drive_openings!r} ran out of game at hand "
                f"{hand_idx}: the state is terminal after {i} forced action(s), "
                f"so the remaining forced actions cannot be applied."
            )
        legal = state.legal_actions()
        if action not in legal:
            raise ValueError(
                f"drive_openings={drive_openings!r} is not legal at hand "
                f"{hand_idx}: forced action {action} at position {i} is not in "
                f"the legal actions {legal} of player {state.current_player()}. "
                f"A forced opening is refused, never skipped: skipping it would "
                f"collect hands from a different subgame under the same label."
            )
        state.apply_action(action)

    while state.is_chance_node():
        state.apply_action(_draw_chance(state, rng))
    if state.is_terminal():
        raise ValueError(
            f"drive_openings={drive_openings!r} ends the hand at hand "
            f"{hand_idx}: the state is terminal once the forced actions are "
            f"applied, so no agent ever chooses and the arm would be an empty "
            f"Parquet that reads as 'the model was never asked'."
        )


#: Every agent attribute `_get_lm_metadata` reads, in the order it reads them. An `LMVllmAgent` sets
#: all of them when it is built, so for one a missing name is a rename, and the column it feeds would
#: log null on every row with nothing failing. The four non-LM agent types set none of them, and their
#: null is correct.
LM_AGENT_ATTRIBUTES: tuple[str, ...] = (
    "last_prompt",
    "last_raw_completion",
    "last_n_retries",
    "last_fallback_used",
    "last_bet_aliased",
    "last_action_parse_vocab",
    "memory_rung",
    "last_memory_truncated",
    "last_elicited_belief",
    "last_belief_raw_completion",
    "last_belief_n_retries",
    "last_belief_failed",
    "model_id",
    "chat_template_hash",
    "served_weights_revision",
    "tokenizer_revision",
    "prompt_schema_version",
    "prompt_spec_hash",
    "belief_protocol",
    "last_reasoning_trace",
    "last_belief_reasoning_trace",
    "last_finish_reason",
    "last_belief_finish_reason",
    "last_belief_prompt",
    "last_completion_tokens",
    "last_belief_completion_tokens",
    "last_agent_node_idx",
    "run_seed",
    "last_belief_salvage_kind",
    "last_belief_salvage_attempt",
    "last_call_latency_s",
    "last_belief_call_latency_s",
    "last_prompt_tokens",
    "last_belief_prompt_tokens",
    "last_response_id",
    "last_belief_response_id",
    "last_request_seed",
    "last_belief_request_seed",
    "last_winning_attempt",
    "last_winning_raw_completion",
    "last_memory_chars_raw",
    "last_memory_chars_sent",
)


def _get_lm_metadata(agent) -> dict:
    """Extract LM-specific logging fields. Non-LM agents return zero defaults.

    The belief-provenance and model/prompt-provenance keys:
    belief_n_retries/belief_failed default to 0/False for non-LM agents,
    mirroring n_retries/fallback_used above (never null). The rest
    (belief_raw_completion, model_id, chat_template_hash,
    prompt_schema_version) default to None, mirroring prompt/raw_completion/
    elicited_belief (null for non-LM agents; schema.py's PARQUET_SCHEMA
    docstrings for each column explain why). belief_protocol follows
    the same null-for-non-LM rule.

    An LM agent sets every attribute in `LM_AGENT_ATTRIBUTES` when it is built, so for one a
    missing attribute is a rename and raises here, naming each; a null column would pass for a
    legitimate null. A non-LM agent sets none of them and its null is correct. The values below
    stay plain ``getattr`` reads because the enumeration test recognises a column by that shape.
    """
    if isinstance(agent, LMVllmAgent):
        missing = [name for name in LM_AGENT_ATTRIBUTES if not hasattr(agent, name)]
        if missing:
            raise AttributeError(
                f"{type(agent).__name__} has no attribute(s) {missing}; each feeds a logged "
                "column that would read null on every row of this run"
            )
    return {
        "prompt": getattr(agent, "last_prompt", None),
        "raw_completion": getattr(agent, "last_raw_completion", None),
        "n_retries": getattr(agent, "last_n_retries", 0),
        "fallback_used": getattr(agent, "last_fallback_used", False),
        "bet_aliased": getattr(agent, "last_bet_aliased", False),
        "action_parse_vocab": getattr(agent, "last_action_parse_vocab", None),
        "memory_rung": getattr(agent, "memory_rung", None),
        "memory_truncated": getattr(agent, "last_memory_truncated", None),
        "elicited_belief": getattr(agent, "last_elicited_belief", None),
        "belief_raw_completion": getattr(agent, "last_belief_raw_completion", None),
        "belief_n_retries": getattr(agent, "last_belief_n_retries", 0),
        "belief_failed": getattr(agent, "last_belief_failed", False),
        "model_id": getattr(agent, "model_id", None),
        "chat_template_hash": getattr(agent, "chat_template_hash", None),
        # The served model's
        # HF weights revision, read off the agent (which carries the
        # launcher-resolved value from its construction) on the same
        # null-for-non-LM getattr default as chat_template_hash above --
        # never a run-level constant, so two agents in one run cannot be
        # conflated and a path that resolved nothing logs null.
        "served_weights_revision": getattr(agent, "served_weights_revision", None),
        # Weights provenance: same thread, same
        # null-for-non-LM getattr default as served_weights_revision above.
        "tokenizer_revision": getattr(agent, "tokenizer_revision", None),
        "prompt_schema_version": getattr(agent, "prompt_schema_version", None),
        # The digest of the prompt content this agent
        # renders with, read off the agent (which holds the spec object it
        # actually renders) rather than recomputed from any run-level label --
        # a label-derived hash would agree with the row's claim by
        # construction and could never detect a variant that failed to reach
        # the renderer. Null for non-LM agents on the same getattr default.
        "prompt_spec_hash": getattr(agent, "prompt_spec_hash", None),
        # The belief decoding protocol, null for non-LM agents on
        # the same getattr default as the keys above. Read off the agent
        # rather than off any run-level config, because the row must stay
        # interpretable when two agents in one run disagree -- see schema.py's
        # column comment for why the run-level config key is not sufficient.
        "belief_protocol": getattr(agent, "belief_protocol", None),
        # Seven additive columns: reasoning trace (action + belief),
        # finish reason (action + belief), belief prompt (the exact rendered
        # bytes), completion tokens (action + belief). Null for non-LM agents,
        # same rule as prompt/raw_completion and the other LM-only keys above.
        "reasoning_trace": getattr(agent, "last_reasoning_trace", None),
        "belief_reasoning_trace": getattr(agent, "last_belief_reasoning_trace", None),
        "finish_reason": getattr(agent, "last_finish_reason", None),
        "belief_finish_reason": getattr(agent, "last_belief_finish_reason", None),
        "belief_prompt": getattr(agent, "last_belief_prompt", None),
        "completion_tokens": getattr(agent, "last_completion_tokens", None),
        "belief_completion_tokens": getattr(
            agent, "last_belief_completion_tokens", None
        ),
        # The two quantities a Parquet reader needs to re-derive this row's request
        # seeds, both read off the acting agent and never off run-level values
        # -- the row's all-players node_idx is not the derivation's node
        # component (57.2% of the landed arm's LM rows differ), and the match
        # seed is not evidence a request seed was sent at all (schema.py's
        # run_seed comment). Null for non-LM agents on the same getattr
        # default as every key above.
        "agent_node_idx": getattr(agent, "last_agent_node_idx", None),
        "run_seed": getattr(agent, "run_seed", None),
        "belief_salvage_kind": getattr(agent, "last_belief_salvage_kind", None),
        "belief_salvage_attempt": getattr(agent, "last_belief_salvage_attempt", None),
        # Per-call reads of the first action and belief calls, plus the
        # winning attempt and the memory block's size. All read off the acting
        # agent on the null-for-non-LM getattr default of every key above.
        "call_latency_s": getattr(agent, "last_call_latency_s", None),
        "belief_call_latency_s": getattr(agent, "last_belief_call_latency_s", None),
        "prompt_tokens": getattr(agent, "last_prompt_tokens", None),
        "belief_prompt_tokens": getattr(agent, "last_belief_prompt_tokens", None),
        "response_id": getattr(agent, "last_response_id", None),
        "belief_response_id": getattr(agent, "last_belief_response_id", None),
        "request_seed": getattr(agent, "last_request_seed", None),
        "belief_request_seed": getattr(agent, "last_belief_request_seed", None),
        "winning_attempt": getattr(agent, "last_winning_attempt", None),
        "winning_raw_completion": getattr(agent, "last_winning_raw_completion", None),
        "memory_chars_raw": getattr(agent, "last_memory_chars_raw", None),
        "memory_chars_sent": getattr(agent, "last_memory_chars_sent", None),
    }


def _agent_id(agent: Agent) -> str:
    return type(agent).__name__.lower().replace("agent", "")


def get_git_sha(cwd: str | None = None) -> str:
    """Return the current git commit SHA, or "unknown" if unavailable.

    Public helper. Used by
    `run_match` to stamp run_meta.json provenance, and by
    `scripts/run_sweep.py` to stamp orchestration-level provenance.

    `cwd`: None (default) inherits the calling
    process's own cwd, which suits internal callers that run from the repo
    root. A script that supports cwd-independent invocation pins it to REPO_ROOT explicitly; without
    pinning, a non-repo-root invocation would silently stamp a wrong sha (a
    different repo's HEAD, not "unknown").

    Returns "unknown" on any failure (subprocess timeout, git not found, not in
    a repository) rather than raising, per "no silent fallbacks": the
    logged fallback is visible in the output artifact (run_meta.json carries it
    verbatim), not silent.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            # The exit code is read below, so a failure is handled, not raised.
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_git_sha() -> str:
    """Private alias for backward compatibility.

    Deprecated: use `get_git_sha()` instead. This private name remains to avoid
    breaking internal callers (run_sweep.py), but external scripts should import the public name.
    """
    return get_git_sha()


def _build_action_history(
    game: pyspiel.Game, terminal_state: pyspiel.State
) -> list[tuple[int, int]]:
    """Return (action, player) pairs for the completed hand, skipping chance nodes."""
    full_history = terminal_state.history()
    s = game.new_initial_state()
    result = []
    for action in full_history:
        if s.is_chance_node():
            s.apply_action(action)
        elif not s.is_terminal():
            result.append((action, s.current_player()))
            s.apply_action(action)
    return result
