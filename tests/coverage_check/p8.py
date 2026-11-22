"""
The planted coverage check for P8 (value of information) on the DEGENERATE 2x2 Dark Hex board.

The board_size=2 board is degenerate: four cells, no chance node, a tree of a few hundred nodes.
A pass here is evidence about the interval arithmetic of the P8 capture-rate statistic on this
tiny tree and about nothing else. It is not evidence about real Dark Hex.

The planted answer is an exact function of the tree, the solved reference and one planted
parameter. Nothing in this module's truth functions takes a sampled row or a frame.
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import polars as pl
import pyspiel

from lmopt.agents.base import SeatAware
from lmopt.agents.cfr_ref import CFRRefAgent
from lmopt.harness.match_runner import run_match
from lmopt.logging.writer import MatchWriter

from tests.coverage_check.coverage import Coverage, summarise

#: every report, verdict and docstring of this check carries this, so a pass is not misread
DEGENERATE_BOARD_NOTE = (
    "DEGENERATE BOARD: dark_hex(board_size=2) has four cells; a pass is not evidence about "
    "real Dark Hex."
)
#: the property key the production cell function publishes
P8_KEY = "P8_capture_rate"
#: the informativeness threshold and capture tolerance, equal to the scorer's defaults (a test
#: pins the equality, so a change to either default is loud)
VOI_THRESHOLD = 1e-2
EPSILON = 1e-6
#: the seat the prescribed agent occupies; the arm is played with a fixed seat
SEAT = 0


@dataclass(frozen=True)
class P8Node:
    """
    one information state of the acting seat, with its exact value table and exact visit weight
    :param info_state: the seat's information-state string (str)
    :param weight: expected visits per hand under the prescribed agent and the reference (float)
    :param voi: the expected value of perfect information at the state, V_pi - V_cfr, floored at 0 (float)
    :param best: the legal actions whose continuation value is within EPSILON of the best (tuple of int)
    :param n_legal: the number of legal actions (int)
    :param capture_prob: probability the prescribed agent plays an action in `best` here (float)
    """

    info_state: str
    weight: float
    voi: float
    best: tuple[int, ...]
    n_legal: int
    capture_prob: float


def _cfr_value(state: pyspiel.State, policy) -> float:
    """
    return to the acting seat of both players following the reference from `state`
    :param state: any non-terminal or terminal state (pyspiel.State)
    :param policy: the solved reference (object)
    :returns: the expected return to player 0 (float)
    """
    if state.is_terminal():
        return state.returns()[SEAT]
    player = state.current_player()
    probs = policy.action_probabilities(state, player)
    return sum(
        probs.get(a, 0.0) * _cfr_value(state.child(a), policy)
        for a in state.legal_actions()
        if probs.get(a, 0.0) > 0.0
    )


def _knowing_value(state: pyspiel.State, policy) -> float:
    """
    return to the seat when it best-responds knowing the world, against the reference
    :param state: a state of one concrete world (pyspiel.State)
    :param policy: the solved reference (object)
    :returns: the expected return to player 0 (float)
    """
    if state.is_terminal():
        return state.returns()[SEAT]
    player = state.current_player()
    if player == SEAT:
        return max(
            _knowing_value(state.child(a), policy) for a in state.legal_actions()
        )
    probs = policy.action_probabilities(state, player)
    return sum(
        probs.get(a, 0.0) * _knowing_value(state.child(a), policy)
        for a in state.legal_actions()
        if probs.get(a, 0.0) > 0.0
    )


def _value_tables(
    game: pyspiel.Game, policy
) -> dict[str, tuple[dict[int, float], float]]:
    """
    for each seat information state, V(a|I) per legal action and the expected value of information

    The worlds of a state are the histories that reach it, weighted by the opponent's reach alone
    (the seat's own reach cancels in the normalisation). This is the P8 value-of-information
    definition, written here from the tree and not from the scorer.
    :param game: the Dark Hex game (pyspiel.Game)
    :param policy: the solved reference (object)
    :returns: `{info_state: ({action: value}, voi)}` for every state with positive opponent reach (dict)
    """
    members: dict[str, list[tuple[pyspiel.State, float]]] = {}

    def walk(state: pyspiel.State, opp_reach: float) -> None:
        if state.is_terminal():
            return
        player = state.current_player()
        if player == SEAT:
            members.setdefault(state.information_state_string(player), []).append(
                (state.clone(), opp_reach)
            )
            for a in state.legal_actions():
                walk(state.child(a), opp_reach)
            return
        probs = policy.action_probabilities(state, player)
        for a in state.legal_actions():
            walk(state.child(a), opp_reach * probs.get(a, 0.0))

    walk(game.new_initial_state(), 1.0)
    tables = {}
    for info, worlds in members.items():
        total = sum(r for _, r in worlds)
        if total <= 0.0:
            continue
        legal = worlds[0][0].legal_actions()
        values = {a: 0.0 for a in legal}
        v_cfr = v_knowing = 0.0
        for state, reach in worlds:
            w = reach / total
            v_cfr += w * _cfr_value(state, policy)
            v_knowing += w * _knowing_value(state, policy)
            for a in legal:
                values[a] += w * _cfr_value(state.child(a), policy)
        tables[info] = (values, max(0.0, v_knowing - v_cfr))
    return tables


def _best_actions(values: dict[int, float]) -> tuple[int, ...]:
    """
    the actions whose value is within EPSILON of the best one
    :param values: `{action: V(a|I)}` (dict)
    :returns: the near-best actions, ascending (tuple of int)
    """
    top = max(values.values())
    return tuple(sorted(a for a, v in values.items() if top - v <= EPSILON))


def acquisition_policy(
    best: tuple[int, ...], legal: list[int], skill: float
) -> dict[int, float]:
    """
    the prescribed agent's action distribution at one state
    :param best: the near-best actions (tuple of int)
    :param legal: the legal actions (list of int)
    :param skill: the probability of choosing uniformly among `best` in place of uniformly among `legal` (float)
    :returns: `{action: probability}` over `legal` (dict)
    """
    return {
        a: skill * (1.0 / len(best) if a in best else 0.0) + (1.0 - skill) / len(legal)
        for a in legal
    }


def enumerate_p8_nodes(
    game: pyspiel.Game, reference_policy, skill: float
) -> list[P8Node]:
    """
    walk the whole tree once with the prescribed agent in the seat and the reference opposite

    The agent plays `acquisition_policy`; the visit weight of a state is the sum, over the
    histories in it, of the product of both players' action probabilities. The reference is the
    only input besides the planted `skill`; no sampled data enters.
    :param game: the Dark Hex game (pyspiel.Game)
    :param reference_policy: the solved reference, which the opponent follows (object)
    :param skill: the planted probability of choosing a near-best action deliberately, in [0, 1] (float)
    :returns: one P8Node per seat information state the agent can reach (list of P8Node)
    """
    if not 0.0 <= skill <= 1.0:
        raise ValueError(f"skill must be in [0, 1], got {skill!r}")
    tables = _value_tables(game, reference_policy)
    weights: dict[str, float] = {}

    def walk(state: pyspiel.State, reach: float) -> None:
        if state.is_terminal() or reach <= 0.0:
            return
        player = state.current_player()
        legal = state.legal_actions()
        if player == SEAT:
            info = state.information_state_string(player)
            weights[info] = weights.get(info, 0.0) + reach
            best = _best_actions(tables[info][0])
            probs = acquisition_policy(best, legal, skill)
        else:
            probs = reference_policy.action_probabilities(state, player)
        for a in legal:
            walk(state.child(a), reach * probs.get(a, 0.0))

    walk(game.new_initial_state(), 1.0)
    nodes = []
    for info, weight in weights.items():
        values, voi = tables[info]
        best = _best_actions(values)
        n_legal = len(values)
        nodes.append(
            P8Node(
                info,
                weight,
                voi,
                best,
                n_legal,
                skill + (1.0 - skill) * len(best) / n_legal,
            )
        )
    return nodes


def planted_p8_capture(nodes: list[P8Node]) -> float:
    """
    the exact expected capture rate: informative visits that play a near-best action, over informative visits
    :param nodes: the seat's information states with exact weights (list of P8Node)
    :returns: the visit-weighted capture probability over states with voi above VOI_THRESHOLD (float)
    """
    informative = [n for n in nodes if n.voi > VOI_THRESHOLD]
    total = sum(n.weight for n in informative)
    if total <= 0.0:
        raise ValueError(
            "no informative state is reachable under this agent, so the statistic has no "
            "scored row and no interval can be formed"
        )
    return sum(n.weight * n.capture_prob for n in informative) / total


def expected_scored_rows_per_hand(nodes: list[P8Node]) -> float:
    """
    the exact expected number of scored rows a hand contributes to a shard
    :param nodes: the seat's information states with exact weights (list of P8Node)
    :returns: the summed visit weight of states with voi above VOI_THRESHOLD (float)
    """
    return sum(n.weight for n in nodes if n.voi > VOI_THRESHOLD)


class PrescribedAcquirer(SeatAware):
    """
    plays a near-best action with probability `skill`, else a uniformly random legal action

    A near-best action is one whose continuation value V(a|I), under the reference, is within
    EPSILON of the best at the state. The scorer reads only the logged action, so this is the
    whole of what "acquires the available value" means to P8. It sets `last_prompt` to a fixed
    string because the scorer's population filter keeps rows that carry a prompt.
    :param game: the Dark Hex game (pyspiel.Game)
    :param reference_policy: the solved reference (object)
    :param skill: the planted probability of choosing a near-best action deliberately (float)
    :param seed: seeds the action choice (int)
    """

    backend = "cpu"
    requires_model = False
    deterministic_action = False

    def __init__(self, game, reference_policy, skill: float, seed: int = 0) -> None:
        if not 0.0 <= skill <= 1.0:
            raise ValueError(f"skill must be in [0, 1], got {skill!r}")
        self._skill = skill
        self._rng = np.random.default_rng(seed)
        self._best = {
            info: _best_actions(values)
            for info, (values, _) in _value_tables(game, reference_policy).items()
        }
        self.last_prompt = "prescribed acquirer"
        self.last_fallback_used = False

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        self._seat = seat

    def act(self, state: pyspiel.State) -> int:
        legal = state.legal_actions()
        best = self._best[state.information_state_string(state.current_player())]
        probs = acquisition_policy(best, legal, self._skill)
        return int(self._rng.choice(legal, p=[probs[a] for a in legal]))

    def belief(self, state: pyspiel.State):
        return None

    def observe_hand_end(self, history, own_payoff: float | None = None) -> None:
        pass


def simulate_p8_shard(
    game,
    game_key: str,
    reference_policy,
    skill: float,
    seed: int,
    n_hands: int,
    directory: str,
) -> pl.DataFrame:
    """
    play one shard through the production match runner and read its Parquet back
    :param game: the Dark Hex game (pyspiel.Game)
    :param game_key: the registry key written to the `game` column (str)
    :param reference_policy: the solved reference, the opponent and the oracle (object)
    :param skill: the planted near-best probability (float)
    :param seed: the shard seed (int)
    :param n_hands: hands to play (int)
    :param directory: where the shard is written (str)
    :returns: the shard as the scorers read it (pl.DataFrame)
    """
    path = os.path.join(directory, f"{game_key}_acquirer_vs_cfr_ref_seed{seed}.parquet")
    run_match(
        game,
        PrescribedAcquirer(game, reference_policy, skill, seed=seed),
        CFRRefAgent(reference_policy, seed=seed + 10_000),
        CFRRefAgent(reference_policy, seed=0),
        n_hands=n_hands,
        seed=seed,
        writer=MatchWriter(path),
        oracle_policy=reference_policy,
        game_key=game_key,
    )
    return pl.read_parquet(path)


def simulate_p8_arm(
    game, game_key: str, reference_policy, skill: float, seeds: list[int], n_hands: int
) -> list[tuple[int, pl.DataFrame]]:
    """
    one arm: a shard per seed, as the `(seed, frame)` pairs the cell scorers take
    :param game: the Dark Hex game (pyspiel.Game)
    :param game_key: the registry key (str)
    :param reference_policy: the solved reference (object)
    :param skill: the planted near-best probability (float)
    :param seeds: one seed per shard (list of int)
    :param n_hands: hands per shard (int)
    :returns: `(seed, frame)` per shard (list of tuple)
    """
    with tempfile.TemporaryDirectory() as directory:
        return [
            (
                s,
                simulate_p8_shard(
                    game, game_key, reference_policy, skill, s, n_hands, directory
                ),
            )
            for s in seeds
        ]


def _p8_arm_task(task: tuple) -> dict:
    """
    one arm scored by the production cell function; top level so a pool can run it
    :param task: (game, game_key, reference_policy, skill, seeds, n_hands) (tuple)
    :returns: `{point, lo, hi, n_replicates, n_distinct}` or `{}` when no interval is formable (dict)
    """
    os.environ.setdefault("TQDM_DISABLE", "1")
    from scripts.analyse_stats import _cell_replicate_cis

    game, game_key, reference_policy, skill, seeds, n_hands = task
    shards = simulate_p8_arm(game, game_key, reference_policy, skill, seeds, n_hands)
    means: dict = {}
    counts: dict = {}
    cis = _cell_replicate_cis(
        shards,
        game_key,
        "cfr_ref",
        {game_key: game},
        {game_key: reference_policy},
        means=means,
        counts=counts,
    )
    lo, hi = cis[P8_KEY]
    if lo is None:
        return {}
    n_replicates, n_distinct = counts[P8_KEY]
    return {
        "point": means[P8_KEY],
        "lo": lo,
        "hi": hi,
        "n_replicates": n_replicates,
        "n_distinct": n_distinct,
    }


def collect_p8_arms(
    game,
    game_key: str,
    reference_policy,
    skill: float,
    n_arms: int,
    n_seeds: int,
    n_hands: int,
    base_seed: int,
    workers: int = 1,
) -> list[dict]:
    """
    simulate arms and return the P8 capture-rate interval of each, from the production cell function
    :param game: the Dark Hex game (pyspiel.Game)
    :param game_key: the registry key (str)
    :param reference_policy: the solved reference (object)
    :param skill: the planted near-best probability (float)
    :param n_arms: simulated arms (int)
    :param n_seeds: shards per arm (int)
    :param n_hands: hands per shard (int)
    :param base_seed: the first shard seed; arms use disjoint seed ranges (int)
    :param workers: processes; patches applied in this process reach the arms only when this is 1 (int)
    :returns: one dict per arm, empty for an arm with no formable interval (list of dict)
    """
    tasks = [
        (
            game,
            game_key,
            reference_policy,
            skill,
            [base_seed + arm * n_seeds + i for i in range(n_seeds)],
            n_hands,
        )
        for arm in range(n_arms)
    ]
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            return list(pool.map(_p8_arm_task, tasks, chunksize=1))
    return [_p8_arm_task(t) for t in tasks]


def p8_coverage(
    arms: list[dict], truth: float, n_seeds: int
) -> tuple[Coverage | None, int]:
    """
    the coverage summary of the formable arms against the planted truth
    :param arms: the output of `collect_p8_arms` (list of dict)
    :param truth: the planted capture rate (float)
    :param n_seeds: shards per arm, which fixes the t critical value (int)
    :returns: the summary (None when fewer than 2 arms are formable) and the count of unformable arms (tuple)
    """
    from lmopt.scoring.replicate import t_crit_95

    formable = [a for a in arms if a]
    unformable = len(arms) - len(formable)
    if len(formable) < 2:
        return None, unformable
    return (
        summarise(
            truth,
            [a["point"] for a in formable],
            [a["lo"] for a in formable],
            [a["hi"] for a in formable],
            t_crit_95(n_seeds - 1),
        ),
        unformable,
    )
