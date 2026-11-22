import pickle
import pytest

pyspiel = pytest.importorskip("pyspiel")

from lmopt.games.depth_reference import (
    DepthReferenceError,
    load_depth_reference,
)


@pytest.fixture
def kuhn_game():
    return pyspiel.load_game("kuhn_poker")


def _truncated_kuhn_payload(game, keep: int):
    """A deliberately PARTIAL table: solve kuhn, keep only `keep` info-sets.

    A fully covered table (as real small-board tables are, e.g. 7246/7246 and
    12182/12182) can never produce an off-support lookup. Truncation is what
    makes the off-support branch testable at all.

    The walk is breadth-first from the root so the kept info-sets are the
    shallowest ones - that guarantees the root is on-support (load_depth_reference
    asserts it) while deeper nodes fall off support.
    """
    import collections

    solver = pyspiel.CFRPlusSolver(game)
    for _ in range(50):
        solver.evaluate_and_update_policy()
    avg = solver.average_policy()

    table: dict[str, dict[int, float]] = {}
    queue = collections.deque([game.new_initial_state()])
    while queue and len(table) < keep:
        st = queue.popleft()
        if st.is_terminal():
            continue
        if st.is_chance_node():
            for a, _ in st.chance_outcomes():
                queue.append(st.child(a))
            continue
        iss = st.information_state_string()
        if iss not in table:
            table[iss] = {
                int(a): float(p) for a, p in avg.action_probabilities(st).items()
            }
        for a in st.legal_actions():
            queue.append(st.child(a))
    assert len(table) == keep, f"wanted {keep} info-sets, walked {len(table)}"
    return {
        "game_string": str(game),
        "solver": "mccfr_external",
        "seed": 0,
        "iterations": 50,
        "policy": table,
    }


def test_off_support_lookup_returns_uniform_and_is_flagged(tmp_path, kuhn_game):
    payload = _truncated_kuhn_payload(kuhn_game, keep=2)
    path = tmp_path / "trunc.pkl"
    path.write_bytes(pickle.dumps(payload))

    ref = load_depth_reference(path, kuhn_game, mode="uniform")

    # Walk until we hit an info-set the truncated table does not contain.
    st = kuhn_game.new_initial_state()
    while st.is_chance_node():
        st.apply_action(st.legal_actions()[0])
    seen_off = False
    cumulative_off = 0
    while not st.is_terminal():
        probs, off = ref.lookup(st)
        assert abs(sum(probs.values()) - 1.0) < 1e-9
        assert set(probs) == set(st.legal_actions())
        if off:
            seen_off = True
            cumulative_off += 1
            n = len(st.legal_actions())
            assert all(
                abs(p - 1.0 / n) < 1e-9 for p in probs.values()
            ), "off-support must be the library uniform default"
        # The injected default is the only observable that distinguishes "really
        # consulted the default" from "computed a uniform answer some other way"
        # (e.g. if self._default were dropped or GC'd, both worlds look uniform).
        # It must be consulted exactly once per off-support lookup and never on a
        # hit, so the running miss count must track cumulative off-support lookups.
        assert (
            ref._default.n_miss == cumulative_off
        ), f"expected default consulted {cumulative_off} times, got {ref._default.n_miss}"
        st.apply_action(st.legal_actions()[0])
    assert seen_off, "truncated table must produce at least one off-support lookup"


def test_on_support_lookup_is_not_flagged_and_is_not_uniform(tmp_path, kuhn_game):
    payload = _truncated_kuhn_payload(kuhn_game, keep=2)
    path = tmp_path / "trunc.pkl"
    path.write_bytes(pickle.dumps(payload))
    ref = load_depth_reference(path, kuhn_game, mode="uniform")

    st = kuhn_game.new_initial_state()
    while st.is_chance_node():
        st.apply_action(st.legal_actions()[0])
    probs, off = ref.lookup(st)
    assert off is False
    assert probs == pytest.approx(
        {a: p for a, p in payload["policy"][st.information_state_string()].items()}
    )
    # An on-support lookup must never consult the injected default (0 calls on a
    # hit) - this is the only observable that would catch the default
    # being wired to fire on every lookup instead of only on a miss.
    assert ref._default.n_miss == 0


def test_raise_mode_raises_on_off_support(tmp_path, kuhn_game):
    payload = _truncated_kuhn_payload(kuhn_game, keep=1)
    path = tmp_path / "trunc.pkl"
    path.write_bytes(pickle.dumps(payload))
    ref = load_depth_reference(path, kuhn_game, mode="raise")

    st = kuhn_game.new_initial_state()
    while st.is_chance_node():
        st.apply_action(st.legal_actions()[0])
    with pytest.raises(DepthReferenceError, match="off-support"):
        for _ in range(4):
            ref.lookup(st)
            st.apply_action(st.legal_actions()[0])


def test_two_arg_action_probabilities_signature_is_supplied(tmp_path, kuhn_game):
    """CFRRefAgent calls the 2-arg form (cfr_ref.py:30,43); the raw imported
    type accepts only 1 arg (a TypeError otherwise). The adapter must bridge it."""
    payload = _truncated_kuhn_payload(kuhn_game, keep=2)
    path = tmp_path / "trunc.pkl"
    path.write_bytes(pickle.dumps(payload))
    ref = load_depth_reference(path, kuhn_game, mode="uniform")
    st = kuhn_game.new_initial_state()
    while st.is_chance_node():
        st.apply_action(st.legal_actions()[0])
    assert ref.action_probabilities(
        st, st.current_player()
    ) == ref.action_probabilities(st)


# ---------------------------------------------------------------------------
# Provenance regression against the real artefacts.
#
# These tests validate loading and provenance - not the off-support decision,
# which the truncated synthetic table above covers.
# ---------------------------------------------------------------------------

from lmopt.games.loaders import GAMES

# No small, fully-covered, CFR+-cross-validatable Dark Hex board is in the
# registry: darkhex_3x3 is NOT CFR+-tractable (the whole reason it needs a
# PARTIAL MCCFR reference at all -- load_darkhex_3x3's own docstring:
# TabularPolicy enumeration MemoryErrors, 6 GB cap), so
# `load_or_solve_cfr(..., plus=True)` on it would OOM the same way
# `gather_worlds` does (exit 137). No MCCFR-vs-CFR+ agreement test exists for
# Dark Hex for that reason.


def test_game_string_mismatch_is_refused(tmp_path, kuhn_game):
    payload = _truncated_kuhn_payload(kuhn_game, keep=2)
    payload["game_string"] = "dark_hex(num_cols=3,num_rows=3)"
    path = tmp_path / "wrong.pkl"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(DepthReferenceError, match="game mismatch"):
        load_depth_reference(path, kuhn_game)


def test_pyspiel_version_mismatch_alone_is_not_refused(tmp_path, kuhn_game):
    """    This payload is a real truncated kuhn solve with a faked version string:
    every key in it resolves against the live game, so the only thing wrong
    with it is a label. It must load. What must still be refused is a table
    whose keys genuinely fail to resolve, and
    `test_stale_keys_are_refused_even_when_the_root_still_resolves` pins that --
    the two together are the contract.
    """
    payload = _truncated_kuhn_payload(kuhn_game, keep=2)
    payload["pyspiel_version"] = "0.0.0-not-this-build"
    path = tmp_path / "mislabelled.pkl"
    path.write_bytes(pickle.dumps(payload))
    ref = load_depth_reference(path, kuhn_game)
    assert ref.pyspiel_version_skew == ("0.0.0-not-this-build", pyspiel.__version__)


def test_payload_without_pyspiel_version_still_loads(tmp_path, kuhn_game):
    """The tractable-tier schema lacks the field; it must not be required."""
    payload = _truncated_kuhn_payload(kuhn_game, keep=2)
    assert "pyspiel_version" not in payload
    path = tmp_path / "old_schema.pkl"
    path.write_bytes(pickle.dumps(payload))
    assert load_depth_reference(path, kuhn_game).n_infosets == 2


def test_unstamped_fully_covered_table_is_probed_not_exempted(tmp_path, kuhn_game):
    """An unstamped table (no `pyspiel_version` field) must be probed, not
    exempted from checking: a gate that exempts any unstamped pickle skips
    exactly the oldest artefacts, which carry no field to compare. This
    proves the property on a synthetic, non-Dark-Hex fixture built on
    canonical `kuhn_poker`.

    What this test adds beyond the two neighbours above (`..._still_loads`,
    `..._fails_on_root_assertion`, both of which already exercise unstamped
    payloads via the same `_truncated_kuhn_payload` helper): full coverage.
    Both neighbours use `keep=2` (2 of kuhn's 12 info-sets), so neither can
    show the table resolving with zero off-support anywhere in the tree --
    the shape of a real fully covered unstamped artefact. `keep=12` is
    kuhn's full info-set count (an exhaustive walk of `kuhn_poker` visits
    exactly 12 information states).

    Honest limitation, stated rather than overclaimed: this specific
    fixture is fully covered and internally consistent, so it cannot by
    itself distinguish "probed and found admissible" from "exempted and
    assumed admissible" -- both produce the same observable
    (`n_infosets == 12`, `pyspiel_version_skew is None`, no off-support).
    Proving the check genuinely runs on an unstamped payload, rather than
    merely proving a clean one loads, is `test_foreign_table_fails_on_root_
    assertion`'s job (same unstamped construction, broken keys, must
    refuse) -- that test is the one an `if on_disk_ver is not None`
    gate would actually redden. This test's job is narrower and additive:
    confirm the admitted, fully-covered case reports the expected numbers.
    """
    payload = _truncated_kuhn_payload(kuhn_game, keep=12)
    assert "pyspiel_version" not in payload
    path = tmp_path / "unstamped_full.pkl"
    path.write_bytes(pickle.dumps(payload))

    ref = load_depth_reference(path, kuhn_game)
    assert ref.pyspiel_version_skew is None
    assert ref.n_infosets == 12

    def walk(state):
        if state.is_terminal():
            return False
        if state.is_chance_node():
            return any(walk(state.child(a)) for a, _ in state.chance_outcomes())
        _, off = ref.lookup(state)
        return off or any(walk(state.child(a)) for a in state.legal_actions())

    assert walk(kuhn_game.new_initial_state()) is False, (
        "a fully-covered table must produce zero off-support lookups anywhere "
        "in the tree -- an off-support hit here would mean keep=12 is not "
        "actually kuhn's full info-set count"
    )


def test_foreign_table_fails_on_root_assertion(tmp_path, kuhn_game):
    """A table whose keys do not match the game must fail loudly, not report
    itself as 100% off-support (at 3x3 there is no ground truth to catch it)."""
    payload = _truncated_kuhn_payload(kuhn_game, keep=2)
    payload["policy"] = {"bogus-info-set": {0: 0.5, 1: 0.5}}
    path = tmp_path / "foreign.pkl"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(DepthReferenceError, match="root info-set is off-support"):
        load_depth_reference(path, kuhn_game)


def test_precompute_refuses_darkhex_3x3():
    """Guards the expensive mistake: a CFR+ solve at 3x3 would thrash for hours.

    `--games` uses `choices=list(GAMES)`, so if the darkhex_3x3 registration
    were ever removed, argparse itself would raise SystemExit(2) before the
    guard is reached - `exc.value.code != 0` alone would still pass,
    reporting "refusal verified" when the guard was never exercised. Worse,
    if the guard regressed (not the registration), this test would not fail
    at all - it would hang for hours inside pytest, the exact failure the
    guard exists to prevent. Pin the message so a SystemExit(2) argparse rejection
    cannot masquerade as the guard firing.
    """
    from scripts.precompute_references import main as precompute_main
    import sys

    argv = sys.argv
    sys.argv = ["precompute_references.py", "--games", "darkhex_3x3", "--skip-rl"]
    try:
        with pytest.raises(SystemExit) as exc:
            precompute_main()
        assert exc.value.code != 0
        assert "has no full CFR+ reference" in str(exc.value)
    finally:
        sys.argv = argv


def test_precompute_refuses_darkhex_3x3_even_with_skip_cfr():
    """The RL-warming escape hatch: --skip-cfr must not be a safe way to reach
    XFPSolver for darkhex_3x3 (XFP's best-response step does the same full-tree get_all_states enumeration
    that OOMs TabularPolicy)."""
    from scripts.precompute_references import main as precompute_main
    import sys

    argv = sys.argv
    sys.argv = ["precompute_references.py", "--games", "darkhex_3x3", "--skip-cfr"]
    try:
        with pytest.raises(SystemExit) as exc:
            precompute_main()
        assert exc.value.code != 0
        assert "has no full CFR+ reference" in str(exc.value)
    finally:
        sys.argv = argv


def test_depth_references_registry_points_at_the_real_artefact():
    from lmopt.games.depth_reference import DEPTH_REFERENCES

    assert "darkhex_3x3" in DEPTH_REFERENCES


# ---------------------------------------------------------------------------
# Admissibility is a computed key-resolution probe, not a version-string
# guard. Across a pyspiel upgrade (1.6.15 -> 2.0.2) a version check fails in
# Both directions at once: it refuses artefacts whose keys are fully live, and
# it exempts unstamped pickles outright because they carry no field to compare.
# ---------------------------------------------------------------------------


def _root_key(game):
    st = game.new_initial_state()
    while st.is_chance_node():
        st.apply_action(st.legal_actions()[0])
    return st.information_state_string()


def _smoke_kuhn_payload(game):
    """
    the whole kuhn table (all 12 info-sets) stamped as written under pyspiel 1.6.15, as a real
    `smoke_kuhn` pickle was: a synthetic twin that pins the same claims, since kuhn has exactly
    12 info-sets and so the truncated table is the full one
    :param game: the kuhn game (pyspiel.Game)
    :returns: a depth-reference payload (dict)
    """
    payload = _truncated_kuhn_payload(game, keep=12)
    payload["pyspiel_version"] = "1.6.15"
    return payload


def test_version_skew_alone_does_not_refuse_a_table_whose_keys_resolve(
    tmp_path, kuhn_game
):
    """    The synthetic kuhn table is stamped pyspiel 1.6.15 (as a real smoke_kuhn pickle was) and loaded
    here under the live version, and all 12 of its keys are live (full enumeration, zero
    orphans). Version skew is provenance; it is not on its own a reason to refuse.
    """
    path = tmp_path / "smoke_kuhn.pkl"
    path.write_bytes(pickle.dumps(_smoke_kuhn_payload(kuhn_game)))
    ref = load_depth_reference(path, kuhn_game)
    assert ref.n_infosets == 12


def test_stale_keys_are_refused_even_when_the_root_still_resolves(tmp_path, kuhn_game):
    """    Mangles every key except the root, so the root-on-support smoke passes and
    only the probe can catch it. This is the case a version check would stand
    in for, and it must be refused without one.
    """
    payload = _smoke_kuhn_payload(kuhn_game)
    root = _root_key(kuhn_game)
    payload["policy"] = {
        (k if k == root else "STALE:" + k): v for k, v in payload["policy"].items()
    }
    stale = tmp_path / "stale.pkl"
    stale.write_bytes(pickle.dumps(payload))
    with pytest.raises(DepthReferenceError, match="do not resolve against this game"):
        load_depth_reference(stale, kuhn_game)


def test_a_legitimately_tiny_partial_table_is_admitted(tmp_path, kuhn_game):
    """    A one-info-set table is the root and nothing else, so it can never produce a
    hit past the root - yet it is perfectly valid, just small. A random-playout
    probe refuses it (which would make test_raise_mode_raises_on_off_support
    fail). "Tiny" and "stale" are
    different things and the gate must not conflate them.
    """
    payload = _truncated_kuhn_payload(kuhn_game, keep=1)
    path = tmp_path / "tiny.pkl"
    path.write_bytes(pickle.dumps(payload))
    ref = load_depth_reference(path, kuhn_game)
    assert ref.n_infosets == 1


def test_the_walk_reports_exhaustion_honestly(kuhn_game):
    """    Exhaustion is what selects the exact subset check over the weaker overlap
    check, so claiming it falsely would apply a subset check to a partial walk
    and refuse every legitimate 3x3 visited table. Kuhn exhausts at its full 12
    info-sets; darkhex_3x3 (~1.11 M reachable) must not.
    """
    from lmopt.games.depth_reference import _live_infoset_keys

    keys, exhausted = _live_infoset_keys(kuhn_game)
    assert exhausted is True and len(keys) == 12

    keys, exhausted = _live_infoset_keys(GAMES["darkhex_3x3"](), budget=5_000)
    assert exhausted is False, "a budget-limited walk must not claim exhaustion"


def test_version_skew_is_recorded_rather_than_silent(tmp_path, kuhn_game):
    """    Admitting a skewed artefact is only defensible if the skew stays visible to a
    consumer; a silent admission is the failure a version check over-corrects for.
    """
    path = tmp_path / "smoke_kuhn.pkl"
    path.write_bytes(pickle.dumps(_smoke_kuhn_payload(kuhn_game)))
    ref = load_depth_reference(path, kuhn_game)
    assert ref.pyspiel_version_skew == ("1.6.15", pyspiel.__version__)


def test_off_support_default_refuses_a_state_with_no_legal_actions(kuhn_game):
    """
    a state with no legal actions has no uniform policy, so the default refuses it with a
    DepthReferenceError naming the state instead of a ZeroDivisionError
    """
    from lmopt.games.depth_reference import _OffSupportDefault

    state = kuhn_game.new_initial_state()
    while not state.is_terminal():
        state.apply_action(state.legal_actions()[0])
    with pytest.raises(DepthReferenceError, match="no legal actions"):
        _OffSupportDefault().get_state_policy(state)


@pytest.mark.parametrize(
    "payload",
    [
        b"cnonexistent_module_for_depth_reference_test\nThing\n.",
        b"cbuiltins\nNoSuchAttributeForDepthReferenceTest\n.",
    ],
    ids=["missing-module", "missing-attribute"],
)
def test_a_pickle_naming_a_missing_class_is_a_depth_reference_error(
    tmp_path, kuhn_game, payload
):
    """
    a pickle that references a module or class this environment lacks (version skew) raises
    the loader's own DepthReferenceError, as a corrupt pickle does, not a bare ImportError or
    AttributeError
    """
    path = tmp_path / "skewed.pkl"
    path.write_bytes(payload)
    with pytest.raises(DepthReferenceError, match="cannot load depth reference"):
        load_depth_reference(path, kuhn_game)
