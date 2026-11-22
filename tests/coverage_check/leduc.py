"""
The Leduc poker instance of the coverage check: a rank-space reporter and a rank-space planted truth.

Leduc differs from Kuhn in two ways that matter here. The runner logs `true_posterior` over six card
indices (two suits of three ranks) while the stated belief is a three-key rank simplex, and the
scorers compare them after rank-marginalising the posterior. And the observer's information state
includes a public community card that the opponent cannot hold. Everything below is derived from the
game tree with its own small rank marginalisation, so it can be compared with the scorers' helper
and not merely call it.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pyspiel

from tests.coverage_check.coverage import Coverage, require_all_arms, summarise
from tests.coverage_check.reporter import PrescribedReporter
from tests.coverage_check.truth import enumerate_nodes

#: the registry key of the game under test
GAME_KEY = "leduc"
#: copies of each rank in the deck (two suits)
CARDS_PER_RANK = 2
#: the property keys `_cell_replicate_cis` publishes for the two statistics under test
P7A_KEY = "P7a_belief_error"
P5_KEY = "P5_calibration_per_pair_margin"


def rank_marginal(card_posterior: np.ndarray) -> np.ndarray:
    """
    collapse a card-space distribution to rank space, summing the two suits of each rank
    :param card_posterior: one entry per card index, ranks adjacent (np.ndarray)
    :returns: one entry per rank (np.ndarray)
    """
    return card_posterior.reshape(-1, CARDS_PER_RANK).sum(axis=1)


def holdable_ranks(
    game: pyspiel.Game, history: tuple[int, ...], seat: int
) -> np.ndarray:
    """
    the ranks the opponent can still hold: a rank with at least one copy that is neither the
    observer's own card nor the revealed public card
    :param game: the Leduc game (pyspiel.Game)
    :param history: a history in the observer's information state (tuple of int)
    :param seat: the observer's seat (int)
    :returns: a boolean mask with one entry per rank (np.ndarray)
    """
    num_cards = len(game.new_initial_state().chance_outcomes())
    unavailable = {history[seat]}
    state = game.new_initial_state()
    for index, action in enumerate(history):
        if index >= 2 and state.is_chance_node():
            unavailable.add(action)
        state.apply_action(action)
    available = np.array([c not in unavailable for c in range(num_cards)], dtype=float)
    return available.reshape(-1, CARDS_PER_RANK).sum(axis=1) > 0


def stated_rank_belief(
    rank_posterior: np.ndarray, holdable: np.ndarray, lam: float
) -> np.ndarray:
    """
    the planted stated belief: the true rank posterior mixed toward uniform over holdable ranks
    :param rank_posterior: the exact posterior over ranks (np.ndarray)
    :param holdable: which ranks the opponent can still hold (np.ndarray of bool)
    :param lam: the weight on the uniform belief, in [0, 1] (float)
    :returns: a distribution over all ranks, zero on ranks that cannot be held (np.ndarray)
    """
    uniform = holdable / holdable.sum()
    return (1.0 - lam) * rank_posterior + lam * uniform


@dataclass(frozen=True)
class RankNode:
    """
    one information state of the observer in rank space
    :param info_state: the information-state string (str)
    :param weight: probability the observer reaches the state in a hand (float)
    :param card_posterior: the exact posterior over the six card indices (np.ndarray)
    :param rank_posterior: the same posterior summed to ranks (np.ndarray)
    :param holdable: ranks the opponent can still hold (np.ndarray of bool)
    :param history: one history in the information state (tuple of int)
    """

    info_state: str
    weight: float
    card_posterior: np.ndarray
    rank_posterior: np.ndarray
    holdable: np.ndarray
    history: tuple[int, ...]


def enumerate_rank_nodes(
    game: pyspiel.Game, reference_policy, seat: int = 0
) -> list[RankNode]:
    """
    the observer's information states with exact card and rank posteriors and the holdable ranks

    The deal is `history[0], history[1]` and the opponent's card is `history[1 - seat]`, which
    `enumerate_nodes` assumes; the public card is a later chance action and so never enters the
    posterior, because it is the same in every history of one information state.
    :param game: the Leduc game (pyspiel.Game)
    :param reference_policy: the opponent's policy (object)
    :param seat: the observer's seat (int)
    :returns: one RankNode per distinct information state (list of RankNode)
    """
    return [
        RankNode(
            n.info_state,
            n.weight,
            n.posterior,
            rank_marginal(n.posterior),
            holdable_ranks(game, n.history, seat),
            n.history,
        )
        for n in enumerate_nodes(game, reference_policy, seat)
    ]


def _pooled(nodes: list[RankNode], per_node) -> float:
    total = sum(n.weight for n in nodes)
    if total <= 0:
        raise ValueError(
            f"_pooled has no weight to pool over ({len(nodes)} nodes, total weight {total}); an empty or all-zero node list "
            "means the enumeration found nothing to score"
        )
    return sum(n.weight * per_node(n) for n in nodes) / total


def planted_leduc_p7a(nodes: list[RankNode], lam: float) -> float:
    """
    the exact mean total variation between the stated and the true rank posterior
    :param nodes: the observer's information states (list of RankNode)
    :param lam: the weight on the uniform belief (float)
    :returns: the visit-weighted mean of 0.5 * sum |stated - true| over ranks (float)
    """
    return _pooled(
        nodes,
        lambda n: 0.5
        * float(
            np.abs(
                stated_rank_belief(n.rank_posterior, n.holdable, lam) - n.rank_posterior
            ).sum()
        ),
    )


def planted_leduc_p5_margin(nodes: list[RankNode], lam: float) -> float:
    """
    the exact per-pair error minus the uninformed stater's, over the three rank keys of every row
    :param nodes: the observer's information states (list of RankNode)
    :param lam: the weight on the uniform belief (float)
    :returns: mean |stated - true| minus mean |1/3 - true| over rank keys (float)
    """

    def margin(n: RankNode) -> float:
        k = len(n.rank_posterior)
        stated = stated_rank_belief(n.rank_posterior, n.holdable, lam)
        return float(
            np.abs(stated - n.rank_posterior).mean()
            - np.abs(1.0 / k - n.rank_posterior).mean()
        )

    return _pooled(nodes, margin)


def leduc_truths(nodes: list[RankNode], lam: float) -> dict[str, float]:
    """
    the planted value of each property key, from the tree and `lam` alone
    :param nodes: the observer's information states (list of RankNode)
    :param lam: the weight on the uniform belief (float)
    :returns: `{property key: planted value}` (dict)
    """
    return {
        P7A_KEY: planted_leduc_p7a(nodes, lam),
        P5_KEY: planted_leduc_p5_margin(nodes, lam),
    }


class LeducReporter(PrescribedReporter):
    """
    the prescribed reporter with a rank-keyed belief, as the language-model agent states it

    It states `(1 - lam) * true_rank + lam * uniform_over_holdable_ranks` as a three-key JSON
    object, the shape `prompting.leduc.parse_belief` returns, and plays uniformly random actions.
    :param oracle_policy: the reference policy the logged true posterior is computed under (object)
    :param lam: the weight on the uniform belief, in [0, 1] (float)
    :param seed: seeds the action choice (int)
    """

    def act(self, state: pyspiel.State) -> int:
        from lmopt.elicitation.posterior import compute_true_posterior

        seat = state.current_player()
        true = compute_true_posterior(self._game, state, seat, self._oracle)
        stated = stated_rank_belief(
            rank_marginal(true),
            holdable_ranks(self._game, tuple(state.history()), seat),
            self._lam,
        )
        self.last_elicited_belief = json.dumps(
            {str(i): float(p) for i, p in enumerate(stated)}
        )
        return int(self._rng.choice(state.legal_actions()))


def _leduc_arm_task(task: tuple):
    """
    one simulated arm with the rank-keyed reporter swapped in for the card-keyed one

    A top-level function so a spawned worker can import it; the swap happens inside the worker,
    which is why it reaches arms run in a pool.
    :param task: the tuple `coverage._arm_task` takes (tuple)
    :returns: `{property: (point, lo, hi)}` (dict)
    """
    from tests.coverage_check import arm, coverage

    original = arm.PrescribedReporter
    arm.PrescribedReporter = LeducReporter
    try:
        return coverage._arm_task(task)
    finally:
        arm.PrescribedReporter = original


def run_leduc_coverage(
    game,
    reference_policy,
    lam: float,
    n_arms: int,
    n_seeds: int,
    n_hands: int,
    base_seed: int,
    truths: dict[str, float],
    workers: int = 1,
) -> dict[str, Coverage]:
    """
    simulate Leduc arms with the rank-keyed reporter and summarise each property's coverage
    :param game: the Leduc game (pyspiel.Game)
    :param reference_policy: the solved reference (object)
    :param lam: the weight on the uniform belief (float)
    :param n_arms: simulated arms (int)
    :param n_seeds: shards per arm (int)
    :param n_hands: hands per shard (int)
    :param base_seed: the first shard seed; arms use disjoint seed ranges (int)
    :param truths: planted truth per property key (dict)
    :param workers: processes; patches applied in this process reach arms only when this is 1 (int)
    :returns: the coverage summary per property key in `truths` (dict)
    """
    from lmopt.scoring.replicate import t_crit_95

    properties = tuple(truths)
    tasks = [
        (
            game,
            GAME_KEY,
            reference_policy,
            lam,
            [base_seed + arm * n_seeds + i for i in range(n_seeds)],
            n_hands,
            properties,
        )
        for arm in range(n_arms)
    ]
    os.environ.setdefault("TQDM_DISABLE", "1")
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            results = list(pool.map(_leduc_arm_task, tasks, chunksize=1))
    else:
        results = [_leduc_arm_task(t) for t in tasks]
    out = {}
    for prop in properties:
        rows = [r[prop] for r in results if prop in r]
        require_all_arms(prop, len(rows), n_arms)
        out[prop] = summarise(
            truths[prop],
            [r[0] for r in rows],
            [r[1] for r in rows],
            [r[2] for r in rows],
            t_crit_95(n_seeds - 1),
        )
    return out


__all__ = [
    "LeducReporter",
    "RankNode",
    "enumerate_rank_nodes",
    "holdable_ranks",
    "leduc_truths",
    "planted_leduc_p5_margin",
    "planted_leduc_p7a",
    "rank_marginal",
    "run_leduc_coverage",
    "stated_rank_belief",
]
