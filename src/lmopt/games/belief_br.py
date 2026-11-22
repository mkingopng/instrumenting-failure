"""Best response to a belief over the opponent's private card, for the dealt-card games.

One operator serves both the Bayesian agent (belief = its exact posterior) and the P7b
scorer (belief = the model's stated belief), so the agent and the yardstick it is scored
against cannot disagree about what a best response is.

Why pooled. Below the current information set the best-responding seat moves again.
At each of those later information sets it cannot see the opponent's card, so it must
choose one action for every history in the set. The action is chosen by the value pooled
over those histories, each weighted by belief(root card) times the opponent's and
chance's reach from the root. Taking the max separately in each state instead solves
every later decision with the opponent's card revealed. That is the same defect
``scoring/property2.py`` records fixing by importing OpenSpiel's ``BestResponsePolicy``.

``per_state_max_action_values`` keeps that per-state rule. It is not a best response. It
exists only so the Bayesian agent's play in already-collected data can be reproduced.
"""

from __future__ import annotations

import numpy as np
import pyspiel


def make_cf_state(game, seat: int, my_card: int, opp_card: int, actions: list[int]):
    """
    a fresh state with the hypothesised deal, replayed along a post-deal action sequence
    :param game: the pyspiel game (pyspiel.Game)
    :param seat: the seat holding `my_card` (int)
    :param my_card: the seat's own private card index (int)
    :param opp_card: the hypothesised opponent card index (int)
    :param actions: the post-deal actions to replay, public chance outcomes included (list)
    :returns: the counterfactual state; a public chance outcome the deal makes impossible
        is skipped, so callers that need the row's own information set check it
    """
    s = game.new_initial_state()
    if seat == 0:
        s.apply_action(my_card)
        s.apply_action(opp_card)
    else:
        s.apply_action(opp_card)
        s.apply_action(my_card)
    for a in actions:
        if s.is_terminal():
            break
        if s.is_chance_node():
            if a in [ca for ca, _ in s.chance_outcomes()]:
                s.apply_action(a)
        else:
            s.apply_action(a)
    return s


def _roots(state, seat: int, belief: np.ndarray) -> list:
    """
    one counterfactual state per opponent card the belief gives positive mass
    :param state: a representative state of the information set (pyspiel.State)
    :param seat: the best-responding seat (int)
    :param belief: mass over the opponent's card index (np.ndarray)
    :returns: list of (counterfactual state, belief weight)
    :raises ValueError: when a counterfactual state is not in the row's information set,
        so a hypothesis the history rules out is refused rather than valued at another node
    """
    history = list(state.history())
    game = state.get_game()
    s = game.new_initial_state()
    deal_end = 0
    while deal_end < len(history) and s.is_chance_node():
        s.apply_action(history[deal_end])
        deal_end += 1
    action_seq = history[deal_end:]
    target = state.information_state_string(seat)
    roots = []
    for card in range(len(belief)):
        if belief[card] <= 0.0:
            continue
        cf = make_cf_state(game, seat, history[seat], card, action_seq)
        if cf.information_state_string(seat) != target:
            raise ValueError(
                f"the hypothesis opp_card={card} does not reach the information set "
                f"{target!r} (it reaches {cf.information_state_string(seat)!r})"
            )
        roots.append((cf, float(belief[card])))
    return roots


class _PooledBR:
    """
    best response below one information set: each later own information set picks one
    action by its belief-and-reach-weighted pooled value
    """

    def __init__(self, seat: int, opp_policy, roots: list, first_actions: list):
        self._seat = seat
        self._pol = opp_policy
        self._members: dict = {}
        self._values: dict = {}
        self._actions: dict = {}
        for root, weight in roots:
            for a in first_actions:
                self._collect(root.child(a), weight)

    def _collect(self, s, weight: float) -> None:
        """
        record every own-decision state below `s` with its pooling weight
        :param s: the state to walk from (pyspiel.State)
        :param weight: belief(root card) times opponent and chance reach so far (float)
        """
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, p in s.chance_outcomes():
                self._collect(s.child(a), weight * p)
            return
        player = s.current_player()
        if player == self._seat:
            info = s.information_state_string(self._seat)
            self._members.setdefault(info, []).append((s, weight))
            for a in s.legal_actions():
                self._collect(s.child(a), weight)
            return
        probs = self._pol.action_probabilities(s, player)
        for a in s.legal_actions():
            self._collect(s.child(a), weight * probs.get(a, 0.0))

    def value(self, s) -> float:
        """
        expected return for the seat from `s`, own moves by the pooled best response
        :param s: the state (pyspiel.State)
        :returns: the expected return (float)
        """
        key = s.history_str()
        if key in self._values:
            return self._values[key]
        if s.is_terminal():
            v = s.returns()[self._seat]
        elif s.is_chance_node():
            v = sum(p * self.value(s.child(a)) for a, p in s.chance_outcomes())
        elif s.current_player() == self._seat:
            v = self.value(
                s.child(self._action(s.information_state_string(self._seat)))
            )
        else:
            probs = self._pol.action_probabilities(s, s.current_player())
            v = sum(
                probs.get(a, 0.0) * self.value(s.child(a))
                for a in s.legal_actions()
                if probs.get(a, 0.0) > 0.0
            )
        self._values[key] = v
        return v

    def _action(self, info: str) -> int:
        """
        the one action the seat takes at a later information set, the first argmax of the
        pooled value, as OpenSpiel's BestResponsePolicy breaks ties
        :param info: the information-state string (str)
        :returns: the chosen action (int)
        """
        if info in self._actions:
            return self._actions[info]
        members = self._members[info]
        legal = members[0][0].legal_actions()
        pooled = {a: sum(w * self.value(s.child(a)) for s, w in members) for a in legal}
        chosen = max(legal, key=pooled.__getitem__)
        self._actions[info] = chosen
        return chosen


def pooled_action_values(state, seat: int, belief: np.ndarray, opp_policy) -> dict:
    """
    the belief-weighted value of each legal action, every later own decision a best response
    pooled over the histories it cannot tell apart
    :param state: a representative state of the seat's information set (pyspiel.State)
    :param seat: the best-responding seat (int)
    :param belief: mass over the opponent's card index, not necessarily normalised; at
        least one entry must be positive (np.ndarray)
    :param opp_policy: the opponent's policy, `action_probabilities(state, player)` (object)
    :returns: {action: sum over cards of belief(card) * value} (dict)
    """
    legal = state.legal_actions()
    roots = _roots(state, seat, belief)
    solver = _PooledBR(seat, opp_policy, roots, legal)
    return {a: sum(w * solver.value(r.child(a)) for r, w in roots) for a in legal}


def _per_state_max_value(state, seat: int, opp_policy) -> float:
    """
    expected return with the seat's own later moves maximised state by state, which sees
    the opponent's card; the as-collected rule, not a best response
    :param state: the state (pyspiel.State)
    :param seat: the seat (int)
    :param opp_policy: the opponent's policy (object)
    :returns: the expected return (float)
    """
    if state.is_terminal():
        return state.returns()[seat]
    if state.is_chance_node():
        return sum(
            p * _per_state_max_value(state.child(a), seat, opp_policy)
            for a, p in state.chance_outcomes()
        )
    acting = state.current_player()
    if acting == seat:
        return max(
            _per_state_max_value(state.child(a), seat, opp_policy)
            for a in state.legal_actions()
        )
    probs = opp_policy.action_probabilities(state, acting)
    return sum(
        probs.get(a, 0.0) * _per_state_max_value(state.child(a), seat, opp_policy)
        for a in state.legal_actions()
        if probs.get(a, 0.0) > 0.0
    )


def per_state_max_action_values(
    state: pyspiel.State, seat: int, belief: np.ndarray, opp_policy
) -> dict:
    """
    the action values the Bayesian agent used before the pooled best response: not a best
    response, because each later own move is chosen with the opponent's card revealed. Kept
    only to reproduce the agent's play in data collected with it; never use it to score
    :param state: the current state (pyspiel.State)
    :param seat: the acting seat (int)
    :param belief: mass over the opponent's card index (np.ndarray)
    :param opp_policy: the opponent's policy (object)
    :returns: {action: sum over cards of belief(card) * per-state-max value} (dict)
    """
    history = list(state.history())
    game = state.get_game()
    s = game.new_initial_state()
    deal_end = 0
    while deal_end < len(history) and s.is_chance_node():
        s.apply_action(history[deal_end])
        deal_end += 1
    action_seq = history[deal_end:]
    values = {}
    for a in state.legal_actions():
        ev = 0.0
        for card in range(len(belief)):
            if belief[card] <= 0.0:
                continue
            cf = make_cf_state(game, seat, history[seat], card, action_seq + [a])
            ev += belief[card] * _per_state_max_value(cf, seat, opp_policy)
        values[a] = ev
    return values
