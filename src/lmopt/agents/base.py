"""Agent protocol shared by all harness participants."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pyspiel


class SeatAware:
    """Default `begin_hand` for every agent: remember this hand's seat.

    Seats became a per-hand property (`seat_mode="random"`
    in `run_match`), and the obvious way to express that -- calling `reset()`
    again with the new seat -- is exactly the thing that must not happen:
    `LMVllmAgent.reset()` clears the cross-hand `HandMemory`, and
    `RLSelfPlayAgent`/`BayesianAgent` rebuild per-match state there. So the
    per-hand channel is its own method, and it does the one thing a seat change
    means for an agent that reads `self._seat`.

    A mixin rather than seven copies of the same line: only `BayesianAgent` reads the
    attribute at all -- in `bayesian.py`, its posterior update (own card and
    opponent seat), its best-response own-card lookup, and its terminal-EV
    "which seat's return" read -- and it reads it fresh on every `act`/`belief` call, so storing the
    new value is sufficient. Every other agent keys off
    `state.current_player()` and writes `_seat` only for provenance. An agent
    that ever needs more than this (a cached seat-keyed table, say) overrides
    `begin_hand` and says so; inheriting silently is the correct default only
    because the survey says it is.
    """

    _seat: Optional[int] = None

    #: Whether `act()` can ever return a different action on two calls with
    #: the identical state, over every code path reachable from it -- a fact
    #: about the class, not about any one run's collected data. `True` is the
    #: correct default for this mixin's actual population: every agent here except
    #: `CFRRefAgent`/`RLSelfPlayAgent` computes `act()` by strict argmax with
    #: no RNG read anywhere in the reachable call chain, verified by reading
    #: the whole class, not inferred from its name. Consumed by the M0/M1/M2
    #: scoreboard's partial-cell classifier to decide whether a missing
    #: opponent's absence is structural (this flag true, game has no
    #: reachable chance node) or merely thin evidence (this flag false, or
    #: the game does have one) -- never a name-keyed lookup table.
    deterministic_action: bool = True

    def begin_hand(self, seat: int) -> None:
        """Called before every hand with this agent's seat for that hand.

        Called for both agents, in every seat mode, before any `act()` of the
        hand -- so the protocol is uniform and a fixed-seat run exercises the
        same code path as an alternating one.
        """
        self._seat = seat


@runtime_checkable
class Agent(Protocol):
    """Protocol all harness agents must satisfy.

    Non-LM agents set backend="cpu" and requires_model=False.
    LM agents set backend="cuda"/"mps"/... and requires_model=True.
    """

    backend: str
    requires_model: bool

    def reset(self, game: pyspiel.Game, seat: int, opponent_id: str) -> None:
        """Called once at the start of each match.

        `seat` here is the opening seat only, not a fact about the match:
        `begin_hand` supersedes it before every hand and may change it.
        So nothing seat-derived may be cached in `reset` -- an agent
        that precomputes a seat-keyed table here is reading a value that will
        be stale on the first swapped hand, silently.
        """
        ...

    def begin_hand(self, seat: int) -> None:
        """Called before every hand with this agent's seat for that hand.

        The default implementation is `SeatAware.begin_hand` above (store the
        seat); the seven concrete agents get it by inheriting `SeatAware`.
        Declared here as well because `run_match` calls it unconditionally, so
        an agent-like object that lacks it must fail loudly rather than have
        the runner guess a seat for it.
        """
        ...

    def act(self, state: pyspiel.State) -> int:
        """Return a legal action for the current state."""
        ...

    def belief(self, state: pyspiel.State) -> Optional[np.ndarray]:
        """Distribution over opponent's hidden state, shape (num_cards,).

        Returns None for agents that do not maintain explicit beliefs.
        Own-card entry is 0.0 by convention.
        """
        ...

    def observe_hand_end(
        self,
        history: list[tuple[int, int]],
        own_payoff: float | None = None,
    ) -> None:
        """Called after each hand with (action, player) pairs for the completed episode.

        `own_payoff` is this agent's terminal return for the hand, or None when
        the caller does not supply one. It is deliberately per-seat: `run_match` hands both agents the *same*
        `history` object, and in a zero-sum game the two seats' returns are
        negatives of each other, so a shared payload cannot express "yours".

        Why the parameter exists at all: `history` is built by
        `_build_action_history`, which returns `(action, player)` pairs and
        skips chance nodes, so it carries no payoff and no card. A
        cross-hand memory that states "you lost 2" is not constructible from it.
        `state.returns()` is computed a few lines above the hook calls in
        `run_match`; this parameter carries it to the agents as well as the
        Parquet rows.

        Optional with a default so the six non-LM agent classes need not use
        it; every implementer still accepts it, so
        the Protocol's implementers do not disagree in arity.
        """
        ...
