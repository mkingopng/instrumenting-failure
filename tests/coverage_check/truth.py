"""
The planted answers: exact functions of the game tree, the reference policy and the planted
parameters. Nothing here takes a sampled row or a frame, which is what makes a coverage figure
against it a test and not an identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyspiel


@dataclass(frozen=True)
class Node:
    """
    one information state of the observer, with its exact visit weight and exact posterior
    :param info_state: the observer's information-state string (str)
    :param weight: probability the observer reaches this state in a hand (float)
    :param posterior: P(hidden card | information state), one entry per card (np.ndarray)
    :param own_card: the observer's own card index (int)
    :param history: one history in the information state, to replay it (tuple of int)
    """

    info_state: str
    weight: float
    posterior: np.ndarray
    own_card: int
    history: tuple[int, ...]


def enumerate_nodes(game: pyspiel.Game, reference_policy, seat: int = 0) -> list[Node]:
    """
    walk the whole tree once and group the observer's decision nodes by information state

    The observer chooses uniformly among its legal actions and the opponent follows
    `reference_policy`. The posterior is the textbook definition: the joint reach of the histories
    in the information state, summed by the opponent's card and normalised. It does not replay a
    hypothesis the way the production posterior does, so the two can be compared.
    :param game: a two-player card game with a fixed private deal (pyspiel.Game)
    :param reference_policy: the opponent's policy, with `action_probabilities(state)` (object)
    :param seat: the observer's seat (int)
    :returns: one Node per distinct information state, in first-visit order (list of Node)
    """
    num_cards = len(game.new_initial_state().chance_outcomes())
    groups: dict[str, list[tuple[tuple[int, ...], float]]] = {}

    def walk(state: pyspiel.State, reach: float) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                walk(state.child(action), reach * prob)
            return
        player = state.current_player()
        legal = state.legal_actions()
        if player == seat:
            groups.setdefault(state.information_state_string(player), []).append(
                (tuple(state.history()), reach)
            )
            for action in legal:
                walk(state.child(action), reach / len(legal))
            return
        probs = reference_policy.action_probabilities(state)
        for action in legal:
            walk(state.child(action), reach * probs.get(action, 0.0))

    walk(game.new_initial_state(), 1.0)
    nodes = []
    for info_state, members in groups.items():
        total = sum(reach for _, reach in members)
        posterior = np.zeros(num_cards)
        for history, reach in members:
            posterior[history[1 - seat]] += reach
        nodes.append(
            Node(
                info_state, total, posterior / total, members[0][0][seat], members[0][0]
            )
        )
    return nodes


def _pooled(nodes: list[Node], per_node: Callable[[Node], float]) -> float:
    total = sum(n.weight for n in nodes)
    return sum(n.weight * per_node(n) for n in nodes) / total


def _stated(node: Node, lam: float) -> np.ndarray:
    uniform = np.full(len(node.posterior), 1.0 / (len(node.posterior) - 1))
    uniform[node.own_card] = 0.0
    return (1.0 - lam) * node.posterior + lam * uniform


def planted_p7a(nodes: list[Node], lam: float) -> float:
    """
    the exact mean total variation between the stated and the true posterior
    :param nodes: the observer's information states (list of Node)
    :param lam: the weight on the uniform belief (float)
    :returns: the visit-weighted mean of 0.5 * sum |stated - true| (float)
    """
    return _pooled(
        nodes, lambda n: 0.5 * float(np.abs(_stated(n, lam) - n.posterior).sum())
    )


def planted_p5_margin(nodes: list[Node], lam: float) -> float:
    """
    the exact per-pair error minus the uninformed stater's, over all (card, row) pairs
    :param nodes: the observer's information states (list of Node)
    :param lam: the weight on the uniform belief (float)
    :returns: mean |stated - true| minus mean |1/n - true| over the n card keys (float)
    """

    def margin(n: Node) -> float:
        k = len(n.posterior)
        observed = np.abs(_stated(n, lam) - n.posterior).mean()
        baseline = np.abs(1.0 / k - n.posterior).mean()
        return float(observed - baseline)

    return _pooled(nodes, margin)
