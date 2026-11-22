"""Correctness of the reach-weighted Goofspiel true posterior.

These tests exercise the module in isolation, so the estimand is checked
independently of the pipeline that consumes it.
"""

import pytest

pytest.importorskip("pyspiel")

from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_goofspiel
from lmopt.scoring.goofspiel_posterior import (
    GOOFSPIEL_P7_ENABLED,
    goofspiel_all_posteriors,
)


@pytest.fixture(scope="module")
def game():
    return load_goofspiel(num_cards=4)


@pytest.fixture(scope="module")
def policy(game):
    return solve_cfr(game, 300)


def _opening_mix(game, policy):
    """
    P1's CFR+ opening-bid distribution, read straight off the policy at P1's first node
    :param game: the wrapped Goofspiel game (pyspiel.Game)
    :param policy: the solved CFR+ policy (TabularPolicy)
    :returns: {bid action: probability} (dict)
    """
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    child = state.child(state.legal_actions()[0])
    return dict(policy.action_probabilities(child, 1))


def _seat0_posterior(game, policy, info):
    """
    seat 0's reach-weighted bid posterior at one info-state, from the walk the scorer uses
    :param game: the wrapped Goofspiel game (pyspiel.Game)
    :param policy: the solved CFR+ policy (TabularPolicy)
    :param info: ``information_state_string(0)`` of the P0 node (str)
    :returns: {bid action: probability}, empty when the info-state is not in the map (dict)
    """
    return goofspiel_all_posteriors(game, policy, seat=0).get(info, {})


def test_module_is_enabled():
    """Goofspiel P7 is live in the pipeline."""
    assert GOOFSPIEL_P7_ENABLED is True


def test_round1_posterior_equals_opening_mix(game, policy):
    """At the root (round 1, no evidence), the true posterior is P1's CFR+ opening
    mix - pure prediction, not a Bayesian update."""
    root_info = game.new_initial_state().information_state_string(0)
    post = _seat0_posterior(game, policy, root_info)
    opening = _opening_mix(game, policy)

    assert post, "round-1 posterior should be non-empty"
    assert sum(post.values()) == pytest.approx(1.0, abs=1e-6)
    for action, prob in opening.items():
        assert post.get(action, 0.0) == pytest.approx(prob, abs=1e-6)


def test_posterior_normalised_and_supported(game, policy):
    """Every P0 info-state's posterior is a normalised distribution over P1 bids."""
    root_info = game.new_initial_state().information_state_string(0)
    post = _seat0_posterior(game, policy, root_info)
    assert all(p >= -1e-12 for p in post.values())
    assert sum(post.values()) == pytest.approx(1.0, abs=1e-6)


def test_round2_posterior_shifts_from_round1(game, policy):
    """After a round-1 outcome is observed, the predictive distribution is
    conditioned on evidence and generally differs from the round-1 opening mix
    (genuine inference content from round 2)."""
    # Walk to a P0 round-2 decision node along a deterministic line.
    state = game.new_initial_state()
    plies = 0
    seen_round2_p0 = None
    while not state.is_terminal() and plies < 6:
        if state.is_chance_node():
            state.apply_action(state.legal_actions()[0])
            continue
        if (
            state.current_player() == 0
            and "Win sequence: \n" not in state.information_state_string(0)
        ):
            # A non-empty win sequence ⇒ at least one round resolved (round >= 2).
            seen_round2_p0 = state.information_state_string(0)
            break
        state.apply_action(state.legal_actions()[0])
        plies += 1

    assert seen_round2_p0 is not None, "deterministic line must reach a round-2 P0 node"

    post2 = _seat0_posterior(game, policy, seen_round2_p0)
    opening = _opening_mix(game, policy)
    assert sum(post2.values()) == pytest.approx(1.0, abs=1e-6)

    # The round-2 predictive distribution must be conditioned on the observed
    # round-1 outcome - a genuine shift from the unconditioned opening mix, not a
    # broadcast copy. P1 has also spent one card, so the support shrinks.
    tvd = 0.5 * sum(
        abs(post2.get(c, 0.0) - opening.get(c, 0.0)) for c in set(post2) | set(opening)
    )
    assert tvd > 1e-3, f"round-2 posterior did not shift from opening (TVD={tvd})"
    assert set(post2) != set(opening), "round-2 support should differ from round-1"


def test_num_cards_above_cap_raises(game, policy):
    with pytest.raises(ValueError):
        goofspiel_all_posteriors(game, policy, seat=0, num_cards=6)


def test_missing_posterior_keys_are_deducibly_unplayable_by_p1(game, policy):
    """The 'clairvoyant reference' check: the seat-0 posterior's support
    can omit deck cards at some P0 info-states. This test asserts those
    omissions are legitimate deduction: every card missing from the posterior
    must be unplayable by P1 (already spent) in every world consistent with
    the target info-state - checked over all such info-states, not one.

    With ``|worlds| == 1``, "missing from the posterior => illegal in every
    consistent world" collapses to "illegal in the one world the posterior
    was read from" - true by construction for any reference, including the
    clairvoyant per-world lookup - so a check that stops at the first
    narrow-support info-state (which can be a single-world info-set, e.g.
    opponent reach 7.4e-05) passes under exactly the hypothesis it exists to
    close. Hence this requires >= 2 consistent worlds and checks every
    qualifying info-state, not just the first: 24 multi-world narrow
    info-sets exist for this fixture and all satisfy the property. Checking
    all of them also removes the dependency on ``gather_p0_worlds``'s dict
    traversal order or the CFR iteration count silently changing which
    info-state is under test.

    "Round >= 2" is carried by the ``len(post) < 4`` narrow-support
    filter below, not by a win-sequence string guard - a `"Win sequence:
    \\n" in info` guard matches only 1 of this fixture's 81 info-sets
    and is not a meaningful round selector on its own.

    Uses ``gather_p0_worlds`` (goofspiel_p7b.py) for the independent
    enumeration of concrete states consistent with a P0 info-state, so this
    check does not lean on the posterior's own tree walk to
    grade itself.
    """
    from lmopt.scoring.goofspiel_p7b import gather_p0_worlds

    worlds = gather_p0_worlds(game, policy)
    posteriors = goofspiel_all_posteriors(game, policy, seat=0)

    checked = 0
    for info, states_reach in worlds.items():
        if len(states_reach) < 2:
            # A single-world info-set cannot distinguish legitimate
            # deduction from a clairvoyant per-world lookup - skip it.
            continue
        post = posteriors.get(info, {})
        if not post or len(post) >= 4:
            continue  # not narrow-support: not the case under test

        missing = set(range(4)) - set(post.keys())
        assert missing, "narrow post selected by len(post) < 4 must have missing keys"
        checked += 1

        for state, _reach in states_reach:
            assert state.information_state_string(0) == info
            # Apply any P0 action to reach P1's node: imp_info means P1's
            # info-state (and hence legal actions - P1's remaining hand) is
            # independent of which action P0 takes (goofspiel_posterior.py
            # relies on the same fact for the seat-0 walk itself).
            p0_action = state.legal_actions()[0]
            p1_state = state.child(p0_action)
            assert p1_state.current_player() == 1
            p1_legal = set(p1_state.legal_actions())
            for card in missing:
                assert card not in p1_legal, (
                    f"card {card} is missing from the seat-0 posterior's "
                    f"support at info-state {info!r} but IS a legal P1 bid "
                    "in a world consistent with that info-state — the "
                    "omission would be an action_probabilities artefact, "
                    "not legitimate deduction; the deduction premise is "
                    "falsified for this info-state"
                )

    assert checked > 0, (
        "expected at least one multi-world, narrow-support P0 info-state "
        "for this test to be meaningful"
    )


# ---------------------------------------------------------------------------
# The seat-general bid posterior
# ---------------------------------------------------------------------------


def _brute_force_seat1_posteriors(game, policy):
    """Independent ``{info_str(1): {bid: p}}``, built in the other formulation.

    The module computes seat 1's answer as a reach-weighted mixture of point
    masses: at a P1 node the opponent has already committed this round, so the
    world names its bid and the weighting over consistent worlds does the
    marginalisation. This function computes the same quantity the other way
    round, predictively: it groups the consistent worlds by the P0 node that
    precedes them, weights each group by the reach of that P0 prefix (the
    product of P0's probabilities at its EARLIER nodes only, excluding the
    current-round bid), and mixes P0's policy at that node.

    The two are algebraically the same object - reach(world) factorises as
    prefix_reach * pi_0(bid | prefix) - but they are different computations, so
    agreement is evidence and not a tautology. Nothing is imported from
    ``goofspiel_posterior``.

    Load-bearing fact, asserted below rather than assumed: at a P0 node every
    child P1 node carries the same ``information_state_string(1)``, i.e. P1
    does not observe P0's current bid (this is what makes the grouping legal).
    """
    groups: dict[str, list[tuple[float, dict[int, float]]]] = {}

    def _walk(state, prefix_reach):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _prob in state.chance_outcomes():
                _walk(state.child(action), prefix_reach)
            return
        player = state.current_player()
        if player == 0:
            probs = dict(policy.action_probabilities(state, 0))
            children = [state.child(a) for a in state.legal_actions()]
            infos = {c.information_state_string(1) for c in children}
            assert (
                len(infos) == 1
            ), "P1 observed P0's current bid; the predictive grouping is invalid"
            assert all(c.current_player() == 1 for c in children)
            groups.setdefault(infos.pop(), []).append((prefix_reach, probs))
            for action, child in zip(state.legal_actions(), children, strict=False):
                _walk(child, prefix_reach * probs.get(action, 0.0))
        else:
            for action in state.legal_actions():
                _walk(state.child(action), prefix_reach)  # P1's own reach cancels

    _walk(game.new_initial_state(), 1.0)

    out: dict[str, dict[int, float]] = {}
    for info, parts in groups.items():
        total = sum(w for w, _ in parts)
        if total <= 0.0:
            continue
        posterior: dict[int, float] = {}
        for w, dist in parts:
            for card, p in dist.items():
                posterior[card] = posterior.get(card, 0.0) + (w / total) * p
        out[info] = posterior
    return out


def test_seat_populations_are_the_seats_own_decision_nodes(game, policy):
    """Non-vacuity for the P-seat1-correct test below.

    Unlike Dark Hex, Goofspiel's two seats have the same number of info-sets
    (81 each), so a count cannot separate them; the
    strings can. Every key must be an info string read at that seat's own
    decision node, which Goofspiel renders as "Current player: <seat>" plus
    "P<seat> hand".

    The value halves matter too: two disjoint key sets carrying the same 81
    distributions would still be a relabelling.
    """
    seat0 = goofspiel_all_posteriors(game, policy, seat=0)
    seat1 = goofspiel_all_posteriors(game, policy, seat=1)
    assert seat0 and seat1
    assert not (set(seat0) & set(seat1)), "the two seats' info strings must not overlap"
    for seat, m in ((0, seat0), (1, seat1)):
        for info in m:
            assert info.startswith(f"Current player: {seat}\n")
            assert f"P{seat} hand:" in info
    assert sorted(map(sorted, (d.items() for d in seat0.values()))) != sorted(
        map(sorted, (d.items() for d in seat1.values()))
    ), "the two seats produced the same multiset of distributions"


def test_seat1_posteriors_match_the_independent_predictive_brute_force(game, policy):
    """P-seat1-correct for Goofspiel: every seat-1 info-set's bid posterior.

    Checked over every seat-1 info-set, not a spot check.
    """
    expected = _brute_force_seat1_posteriors(game, policy)
    got = goofspiel_all_posteriors(game, policy, seat=1)
    assert set(got) == set(expected)
    assert len(expected) > 10, "the seat-1 population must be non-trivial"
    shifted = 0
    for info, dist in expected.items():
        assert set(got[info]) == set(dist), f"support differs at {info!r}"
        for card, p in dist.items():
            assert got[info][card] == pytest.approx(p, abs=1e-12)
        assert sum(got[info].values()) == pytest.approx(1.0, abs=1e-9)
        shifted += int(max(dist.values()) > 0.999)
    assert shifted < len(
        expected
    ), "every seat-1 posterior is a point mass; P1 would be clairvoyant"


def test_seat1_round_one_posterior_is_p0s_opening_mix(game, policy):
    """The anchor a brute force cannot supply: a value known from the game.

    At P1's first decision node P1 has observed nothing, so its posterior over
    P0's just-committed bid is exactly P0's CFR+ opening mix, read straight off
    the policy at the root.
    """
    root = game.new_initial_state()
    while root.is_chance_node():
        root = root.child(root.legal_actions()[0])
    opening_p0 = dict(policy.action_probabilities(root, 0))
    first_p1_info = root.child(root.legal_actions()[0]).information_state_string(1)

    seat1 = goofspiel_all_posteriors(game, policy, seat=1)
    post = seat1[first_p1_info]
    # Every card P0 opens with must appear in the support. A subset, not an
    # equality: a card P0 never opens with is still a key carrying 0.0, because
    # the point-mass contributions are keyed by the action taken.
    assert {c for c, p in opening_p0.items() if p > 0.0} <= set(post)
    for card, p in opening_p0.items():
        assert post.get(card, 0.0) == pytest.approx(p, abs=1e-9)


def _brute_force_seat0_posteriors(game, policy):
    """
    independent ``{info_str(0): {bid: p}}``, built in the other formulation (point masses)
    :param game: the wrapped Goofspiel game (pyspiel.Game)
    :param policy: the solved CFR+ policy (TabularPolicy)
    :returns: {P0 info string: {bid action: probability}} (dict)

    The module computes seat 0's answer predictively: at a P0 node it reads P1's policy at the child node.
    This function enumerates the concrete world instead, a P0 state plus the bid P1 then makes, weights it
    by P1's reach including that bid (the opponent's probabilities at its earlier nodes times its
    probability of the bid), and puts a point mass on the bid. The reach-weighted point masses and the
    reach-weighted mixture are the same object, but they are different computations, so agreement is
    evidence. Nothing is imported from ``goofspiel_posterior``. P0's own reach cancels under perfect
    recall, and chance reach is 1 for descending points.
    """
    groups: dict[str, list[tuple[float, dict[int, float]]]] = {}

    def _walk(state, opp_reach):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _prob in state.chance_outcomes():
                _walk(state.child(action), opp_reach)
            return
        if state.current_player() == 0:
            p1_node = state.child(state.legal_actions()[0])
            probs = dict(policy.action_probabilities(p1_node, 1))
            info = state.information_state_string(0)
            for bid in p1_node.legal_actions():
                groups.setdefault(info, []).append(
                    (opp_reach * probs.get(bid, 0.0), {bid: 1.0})
                )
            for action in state.legal_actions():
                _walk(state.child(action), opp_reach)
        else:
            probs = dict(policy.action_probabilities(state, 1))
            for action in state.legal_actions():
                _walk(state.child(action), opp_reach * probs.get(action, 0.0))

    _walk(game.new_initial_state(), 1.0)

    out: dict[str, dict[int, float]] = {}
    for info, parts in groups.items():
        total = sum(w for w, _ in parts)
        if total <= 0.0:
            continue
        posterior: dict[int, float] = {}
        for w, dist in parts:
            for card, p in dist.items():
                posterior[card] = posterior.get(card, 0.0) + (w / total) * p
        out[info] = posterior
    return out


def test_seat0_posteriors_match_the_independent_per_node_brute_force(game, policy):
    """
    every seat-0 info-set's bid posterior agrees with an independent per-state weighting.

    This is the independent cross-check for the one-walk helper at seat 0
    (``goofspiel_all_posteriors`` is what the match runner calls).
    """
    expected = _brute_force_seat0_posteriors(game, policy)
    got = goofspiel_all_posteriors(game, policy, seat=0)
    assert set(got) == set(expected)
    assert len(expected) > 10, "the seat-0 population must be non-trivial"
    for info, dist in expected.items():
        assert set(got[info]) == set(dist), f"support differs at {info!r}"
        for card, p in dist.items():
            assert got[info][card] == pytest.approx(p, abs=1e-9)


def test_unknown_seat_raises(game, policy):
    """No silent fallbacks."""
    with pytest.raises(ValueError, match="seat"):
        goofspiel_all_posteriors(game, policy, seat=2)


def test_seat1_deck_cap_still_raises(game, policy):
    """The enumeration cap is not seat-specific."""
    with pytest.raises(ValueError):
        goofspiel_all_posteriors(game, policy, seat=1, num_cards=6)
