"""Loader for PARTIAL depth-axis reference policies.

Wraps OpenSpiel's own ``pyspiel.PartialTabularPolicy`` - imported, never
reimplemented. The 3x3 artefact is a co-resident MCCFR capture:
22,701 info-sets out of ~1.11 M reachable, so **off-support lookups are the
common case, not an edge case** (measured 57.5 % of decision nodes under
uniform-random play).

Why an adapter rather than the raw type (all measured):

- ``PartialTabularPolicy.action_probabilities`` accepts one argument;
  ``CFRRefAgent`` (``agents/cfr_ref.py``) calls the two-argument form, in
  ``act``.
- The string-path lookup ``get_state_policy(str)`` raises
  ``SpielError: GetStatePolicy(const std::string&) unimplemented`` on a missing
  key, and the State-path silently returns the library uniform. Neither is
  acceptable unlogged.
- ``policy_table()`` returns a fresh copy of the whole table per call
  (11-14 ms for 22,701 entries), so the on/off-support decision must come from
  a key set cached at load, not from a per-node table query.

The 2-arg ``PartialTabularPolicy(table, default_policy)`` overload is the single
off-support injection point: measured consulted exactly once on a miss and not
at all on a hit, and pybind keeps the Python default policy alive across the
boundary (verified with a non-uniform signature, so a GC'd default cannot
masquerade as uniform) - which is why this path cannot hit a
dangling-``average_policy`` SIGSEGV: the pickle is plain dicts and no
solver object exists at load time.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pyspiel

#: Off-support modes. "uniform" is the default (schema-preserving, crash-free,
#: information-preserving). "raise" is for
#: tests and for runs that must not tolerate any off-support lookup.
MODES = ("uniform", "raise")

#: Games whose reference is a PARTIAL depth-reference pickle rather than a
#: full CFR+ TabularPolicy. Every scoring-path dispatch site must consult this
#: before calling load_or_solve_cfr.
DEPTH_REFERENCES: dict[str, Path] = {
    "darkhex_3x3": Path("data/depth_solver/mccfr_darkhex_3x3_seed0_visited.pkl"),
}


class DepthReferenceError(RuntimeError):
    """Provenance mismatch, corrupt payload, or an off-support lookup in raise mode."""


class _OffSupportDefault(pyspiel.Policy):
    """Consulted only on a miss (measured: 1 call on miss, 0 on hit).

    Returns OpenSpiel's uniform-over-legal-actions, the same semantic
    ``nash_conv(..., use_state_get_policy=True)`` uses. Counts misses for debugging only - the reported coverage number
    comes from the Parquet ``cfr_ref_off_support`` column, because these
    in-process counters double-count (match_runner calls both ``agent.act`` and
    ``cfr_ref.policy_at`` per node) and can be bypassed by any consumer handed a
    naked policy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_miss = 0

    def get_state_policy(self, state):
        self.n_miss += 1
        legal = state.legal_actions()
        if not legal:
            raise DepthReferenceError(
                f"off-support lookup at a state with no legal actions ({str(state)!r}); "
                "a terminal or chance state has no policy, so look up decision states only"
            )
        p = 1.0 / len(legal)
        return [(int(a), p) for a in legal]


class DepthReference:
    """A partial reference policy with explicit, logged off-support behaviour."""

    #: (on-disk pyspiel version, running version) when they differ, else None; set by the loader.
    pyspiel_version_skew: tuple[str, str] | None

    def __init__(
        self, table: dict, game: pyspiel.Game, *, mode: str, meta: dict
    ) -> None:
        if mode not in MODES:
            raise DepthReferenceError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.meta = meta
        self._game = game
        # Cached at load: the flag is an O(1) membership test, never policy_table().
        self._keys = frozenset(table)
        self._default = _OffSupportDefault()
        self._policy = pyspiel.PartialTabularPolicy(
            {k: [(int(a), float(p)) for a, p in v.items()] for k, v in table.items()},
            self._default,
        )

    @property
    def n_infosets(self) -> int:
        return len(self._keys)

    def lookup(self, state) -> tuple[dict[int, float], bool]:
        """Return ``(action_probabilities, off_support)`` for ``state``."""
        off = state.information_state_string() not in self._keys
        if off and self.mode == "raise":
            raise DepthReferenceError(
                "off-support lookup at info-set "
                f"{state.information_state_string()!r} with mode='raise'"
            )
        probs = {int(a): float(p) for a, p in self._policy.get_state_policy(state)}
        return probs, off

    def action_probabilities(self, state, player=None) -> dict[int, float]:
        """2-arg signature CFRRefAgent expects; ``player`` is accepted and unused
        (the info-state string already identifies the acting player's view).

        WARNING - the ``off_support`` flag is discarded here. This method exists
        only to satisfy the 2-arg call signature ``CFRRefAgent`` requires (the raw
        ``pyspiel.PartialTabularPolicy.action_probabilities`` accepts one argument
        only - measured ``TypeError`` on two). It returns probabilities alone, so a
        caller on this path cannot tell an on-support answer from an off-support
        uniform substitute. Any caller that needs to know - logging, scoring, or
        anything that must not silently treat off-support play as a real reference
        answer - must call :meth:`lookup` directly and read its ``off_support``
        element instead of this method.
        """
        return self.lookup(state)[0]


def _live_infoset_keys(game: pyspiel.Game, *, budget: int = 200_000):
    """Breadth-first walk of ``game`` collecting live info-state strings.

    Returns ``(keys, exhausted)``. ``exhausted`` is True only when the walk ran
    out of states before it ran out of budget, i.e. ``keys`` is the COMPLETE set
    for this game. That distinction is what lets the caller pick an exact check
    over an overlap check without guessing a threshold.
    """
    import collections

    keys: set[str] = set()
    seen_hist: set[str] = set()
    queue = collections.deque([game.new_initial_state()])
    expanded = 0
    while queue:
        if expanded >= budget:
            return keys, False
        state = queue.popleft()
        expanded += 1
        if not state.is_chance_node():
            keys.add(state.information_state_string())
        for action in state.legal_actions():
            child = state.clone()
            child.apply_action(action)
            if child.is_terminal():
                continue
            h = child.history_str()
            if h in seen_hist:
                continue
            seen_hist.add(h)
            queue.append(child)
    return keys, True


def load_depth_reference(
    path, game: pyspiel.Game, *, mode: str = "uniform"
) -> DepthReference:
    """Load a pickled partial reference, validating provenance against ``game``."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (
        OSError,
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        AttributeError,
        ImportError,
        IndexError,
    ) as exc:
        raise DepthReferenceError(
            f"cannot load depth reference {path}: {exc!r}"
        ) from exc

    if not isinstance(payload, dict) or "policy" not in payload:
        raise DepthReferenceError(f"{path}: not a depth-reference payload")

    on_disk = payload.get("game_string")
    if on_disk != str(game):
        raise DepthReferenceError(
            f"{path}: game mismatch - pickle has {on_disk!r}, caller passed {str(game)!r}"
        )

    ref = DepthReference(payload["policy"], game, mode=mode, meta=payload)

    # The admissibility question is whether this table's keys still resolve
    # against the live game, and that is computed here rather than looked up
    # from a version string. An equality check on a `pyspiel_version` field
    # fails in both directions: across the 1.6.15 -> 2.0.2 upgrade it would
    # refuse artefacts whose keys are fully live (kuhn 12/12, darkhex_2x3
    # 7246/7246 and darkhex_3x2 12182/12182 enumerate with zero orphans; the
    # two 3x3 visited tables hit 43.8% and 46.5% of sampled decision states,
    # where a changed format gives 0.0% by construction), and it would exempt
    # pickles that carry no version field at all. A stale-key table reads as
    # 100% off-support, which at 3x3 is indistinguishable from genuine
    # off-support play, so the keys themselves are checked.
    root = game.new_initial_state()
    while root.is_chance_node():
        root.apply_action(root.legal_actions()[0])
    if root.information_state_string() not in ref._keys:
        raise DepthReferenceError(
            f"{path}: root info-set is off-support - the table does not match this game"
        )
    live_keys, exhausted = _live_infoset_keys(game)
    if exhausted:
        # The walk covered the whole game, so every table key must be live. This
        # is exact: it admits a legitimately tiny partial table (a truncated
        # fixture is not stale) and refuses one whose keys resolve nowhere.
        orphaned = ref._keys - live_keys
        verdict = (
            f"{len(orphaned)} of {len(ref._keys)} table keys are not info-sets of "
            f"this game (walk exhausted at {len(live_keys)} live info-sets)"
            if orphaned
            else None
        )
    else:
        # Too large to enumerate (3x3). The table is PARTIAL by construction, so
        # a subset check is not available; require only that it overlaps the
        # region actually explored. A stale table overlaps it in nothing, while
        # a genuine visited table covers the shallow region densely (measured
        # 43.8% and 46.5% of sampled states on the two 3x3 tables).
        verdict = (
            None
            if ref._keys & live_keys
            else f"none of {len(ref._keys)} table keys appear among the "
            f"{len(live_keys)} live info-sets reachable within the walk budget"
        )
    if verdict is not None:
        live_ver = getattr(pyspiel, "__version__", "unknown")
        raise DepthReferenceError(
            f"{path}: table keys do not resolve against this game - {verdict} "
            f"(pickle pyspiel {payload.get('pyspiel_version')!r}, live {live_ver!r}). "
            "Info-state string formats are version-dependent; a stale-key table "
            "would read as 100% off-support, which at 3x3 is indistinguishable "
            "from genuine off-support play."
        )

    # Version skew is provenance, not an admissibility criterion: recorded so a
    # consumer can see it, never silent, and never on its own a reason to refuse.
    on_disk_ver = payload.get("pyspiel_version")
    ref.pyspiel_version_skew = (
        None
        if on_disk_ver is None
        or on_disk_ver == getattr(pyspiel, "__version__", "unknown")
        else (on_disk_ver, getattr(pyspiel, "__version__", "unknown"))
    )
    return ref
