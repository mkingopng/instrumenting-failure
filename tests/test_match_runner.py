"""Integration tests for the match runner and per-node logging."""

from __future__ import annotations

import dataclasses
import io
import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("pyspiel")

import pyspiel

from lmopt.agents.bayesian import BayesianAgent
from lmopt.agents.cfr_ref import CFRRefAgent
from lmopt.agents.random_agent import RandomAgent
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_kuhn
from lmopt.harness.match_runner import run_match, get_git_sha, _get_git_sha
from lmopt.logging.schema import NodeRecord, REQUIRED_COLUMNS
from lmopt.logging.writer import MatchWriter


def _minimal_record(**overrides):
    """Construct a NodeRecord with every required field filled; overrides win."""
    base = dict(
        run_id="run0",
        matchup_id="a_vs_b",
        game="kuhn",
        game_params="{}",
        seed=0,
        hand_idx=0,
        node_idx=0,
        player_to_act=0,
        info_set="0",
        info_state_tensor=None,
        legal_mask=[1, 1],
        prompt=None,
        raw_completion=None,
        parsed_action=0,
        n_retries=0,
        fallback_used=False,
        bet_aliased=None,
        action_parse_vocab=None,
        memory_rung=None,
        memory_truncated=None,
        cfr_action=0,
        cfr_policy="{}",
        cfr_ref_off_support=False,
        elicited_belief=None,
        true_posterior=None,
        belief_raw_completion=None,
        belief_n_retries=0,
        belief_failed=False,
        reward=0.0,
        probe_type="none",
        backend="cpu",
        model_revision=None,
        timestamp="2026-07-26T00:00:00+00:00",
        model_id=None,
        chat_template_hash=None,
        prompt_schema_version=None,
        prompt_spec_hash=None,
        belief_protocol=None,
        reasoning_trace=None,
        belief_reasoning_trace=None,
        finish_reason=None,
        belief_finish_reason=None,
        belief_prompt=None,
        completion_tokens=None,
        belief_completion_tokens=None,
        lm_seat=0,
        actor="a",
        action_str="Pass",
        prompt_variant="base",
    )
    base.update(overrides)
    return NodeRecord(**base)


# Kuhn terminal payoffs (absolute values).
KUHN_PAYOFFS = {-2.0, -1.0, 1.0, 2.0}

N_HANDS = 10


@pytest.fixture(scope="module")
def game():
    return load_kuhn()


@pytest.fixture(scope="module")
def cfr_policy(game):
    return solve_cfr(game, 500)


@pytest.fixture(scope="module")
def cfr_ref(cfr_policy):
    return CFRRefAgent(cfr_policy, seed=0)


@pytest.fixture(scope="module")
def random_vs_cfr_records(game, cfr_policy, cfr_ref, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("match")
    writer = MatchWriter(str(tmp / "match.parquet"))
    records = run_match(
        game,
        RandomAgent(seed=1),
        cfr_ref,
        cfr_ref,
        n_hands=N_HANDS,
        seed=42,
        writer=writer,
    )
    return records


@pytest.fixture(scope="module")
def bayesian_vs_cfr_records(game, cfr_policy, cfr_ref, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bayesian_match")
    writer = MatchWriter(str(tmp / "bayesian.parquet"))
    bayesian = BayesianAgent(opponent_policy=cfr_policy, seed=2)
    records = run_match(
        game,
        bayesian,
        cfr_ref,
        cfr_ref,
        n_hands=N_HANDS,
        seed=42,
        writer=writer,
    )
    return records, bayesian


class TestMatchCompletion:
    def test_records_non_empty(self, random_vs_cfr_records):
        assert len(random_vs_cfr_records) > 0

    def test_hand_idx_range(self, random_vs_cfr_records):
        hand_idxs = {r.hand_idx for r in random_vs_cfr_records}
        assert hand_idxs == set(range(N_HANDS))

    def test_node_idx_resets_per_hand(self, random_vs_cfr_records):
        for hand_idx in range(N_HANDS):
            hand = [r for r in random_vs_cfr_records if r.hand_idx == hand_idx]
            node_idxs = [r.node_idx for r in hand]
            assert node_idxs == list(range(len(node_idxs)))

    def test_run_id_consistent(self, random_vs_cfr_records):
        run_ids = {r.run_id for r in random_vs_cfr_records}
        assert len(run_ids) == 1


class TestRequiredColumns:
    def test_no_nulls_in_required_columns(self, random_vs_cfr_records):
        for rec in random_vs_cfr_records:
            d = rec.__dict__
            for col in REQUIRED_COLUMNS:
                assert d[col] is not None, f"Null in required column {col!r}"

    def test_parsed_action_is_legal(self, random_vs_cfr_records):
        for rec in random_vs_cfr_records:
            mask = rec.legal_mask
            assert rec.parsed_action < len(mask)
            assert mask[rec.parsed_action] == 1

    def test_cfr_action_matches_policy(self, random_vs_cfr_records):
        """This fixture's cfr_ref wraps a fully-enumerated CFR+ policy, so
        cfr_ref_off_support is always False and an unconditional argmax check
        would pass here. It is FALSE for any off-support row (cfr_action is INVALID_ACTION there, not an argmax) - a
        partial-reference fixture (see test_off_support_flows_through_run_match_
        end_to_end below) would trip it. Branch on the flag so this assertion
        stays true regardless of which reference backs cfr_ref."""
        for rec in random_vs_cfr_records:
            probs = json.loads(rec.cfr_policy)
            if not rec.cfr_ref_off_support:
                expected_best = max(probs, key=probs.get)
                assert rec.cfr_action == int(expected_best)
            else:
                assert rec.cfr_action == pyspiel.INVALID_ACTION

    def test_reward_is_valid_kuhn_payoff(self, random_vs_cfr_records):
        for rec in random_vs_cfr_records:
            assert rec.reward in KUHN_PAYOFFS, f"Unexpected reward {rec.reward}"


class TestBayesianLogging:
    def test_true_posterior_non_null_for_bayesian(self, bayesian_vs_cfr_records):
        records, _ = bayesian_vs_cfr_records
        bayesian_rows = [r for r in records if r.player_to_act == 0]
        assert len(bayesian_rows) > 0
        for rec in bayesian_rows:
            assert rec.true_posterior is not None

    def test_true_posterior_valid_json(self, bayesian_vs_cfr_records):
        records, _ = bayesian_vs_cfr_records
        for rec in records:
            if rec.true_posterior is not None:
                d = json.loads(rec.true_posterior)
                total = sum(d.values())
                assert abs(total - 1.0) < 1e-5

    def test_true_posterior_null_for_cfr(self, bayesian_vs_cfr_records):
        records, _ = bayesian_vs_cfr_records
        cfr_rows = [r for r in records if r.player_to_act == 1]
        for rec in cfr_rows:
            assert rec.true_posterior is None


class TestAttributionColumns:
    """lm_seat/actor/action_str/prompt_variant let analysis (per-hand seat
    randomisation, a prompt-variant axis) attribute a row to a seat and an
    agent without re-deriving them from the game state or run_meta.json
    config after the fact."""

    def test_lm_seat_is_agent0_seat(self, random_vs_cfr_records):
        """lm_seat is agent0's seat for that hand. The value can be a
        per-hand draw; this fixture takes the default seat_mode ("fixed0"), so
        agent0 sits at 0 for every hand and the column is constant.
        """
        assert {r.lm_seat for r in random_vs_cfr_records} == {0}

    def test_actor_matches_acting_agent_id(self, bayesian_vs_cfr_records):
        records, _bayesian = bayesian_vs_cfr_records
        bayesian_rows = [r for r in records if r.player_to_act == 0]
        cfr_rows = [r for r in records if r.player_to_act == 1]
        assert bayesian_rows and cfr_rows
        assert all(r.actor == "bayesian" for r in bayesian_rows)
        assert all(r.actor == "cfrref" for r in cfr_rows)

    def test_action_str_matches_game_action_to_string(self, random_vs_cfr_records):
        seen = {r.action_str for r in random_vs_cfr_records}
        assert seen
        assert seen <= {"Pass", "Bet"}

    def test_prompt_variant_is_base(self, random_vs_cfr_records):
        assert {r.prompt_variant for r in random_vs_cfr_records} == {"base"}


class TestFixed0IsUnchanged:
    """Property P-fixed0: the per-hand seat machinery must leave the default
    path producing exactly the records it produced without that machinery.

    Why a stored golden and not a same-process comparison: the claim is about
    this code against code without seat machinery, and no assertion inside
    one process can reach that implementation.
    `tests/fixtures/fixed0_golden.json` was generated by running the
    match below on commit af679c1 (the last commit before any seat machinery
    existed) and dumping every NodeRecord field.

    Three columns are excluded, and each for a reason that is not "it was
    inconvenient": `run_id` is a fresh uuid per call, `timestamp` is wall clock,
    and `model_revision` is the git sha, which by construction differs from
    the golden's commit. Everything else -- including cfr_policy,
    true_posterior, reward, lm_seat and the action sequence -- must match
    exactly.

    Known false-positive mode: the fixture pins full-precision CFR floats
    (`cfr_policy` is a json dump of the solved distribution), so an OpenSpiel
    solver or numpy version bump can redden this while nothing in this repo
    changed. Diagnose by checking whether the divergence is confined to
    low-order digits of cfr_policy; if it is, re-record the fixture from the
    last green commit before the bump (never from the current tree, which is
    what the golden exists to check).
    """

    GOLDEN = Path(__file__).parent / "fixtures" / "fixed0_golden.json"
    EXCLUDED = ("run_id", "timestamp", "model_revision")

    # Columns that did not exist when the golden was recorded, with the value
    # each must carry on this path. Declared rather than excluded, and not by
    # re-recording the fixture: re-recording from the current tree is what the
    # golden exists to prevent, and dropping the key would stop pinning it. The
    # comparison below therefore checks the golden's own keys exactly and the
    # added keys against this table, so an additive column is a reviewable
    # one-line diff while an accidental change to any pre-existing column still
    # reddens.
    #
    # `prompt_spec_hash` is the digest of the prompt content a row was
    # rendered under. This match plays RandomAgent against CFRRef, so no
    # prompt is rendered at all and the column is null by the same
    # null-for-non-LM rule as `model_id` and `prompt_schema_version`.
    # reasoning_trace, belief_reasoning_trace, finish_reason,
    # belief_finish_reason, belief_prompt, completion_tokens and
    # belief_completion_tokens are null for non-LM rows.
    # `action_parse_vocab` records which vocabulary resolved the action
    # (label/verbose/coords/digit); null for non-LM agents.
    # served_weights_revision: this match has no launcher-resolved revision,
    # so null on every row by the same null-for-non-LM rule.
    # agent_node_idx + run_seed (seed re-derivability): no LM agent here, so
    # null on every row by the same rule.
    # The per-call columns (latency, prompt tokens, response id, request seed and their
    # belief twins, winning attempt and text, memory block sizes, posterior fallback flag)
    # are null on every row of this LM-free path.
    # tokenizer_revision is threaded like served_weights_revision; no LM
    # agent here, so null on every row by the same rule.
    ADDED_SINCE_GOLDEN = {
        "prompt_spec_hash": None,
        "reasoning_trace": None,
        "belief_reasoning_trace": None,
        "finish_reason": None,
        "belief_finish_reason": None,
        "belief_prompt": None,
        "completion_tokens": None,
        "belief_completion_tokens": None,
        "action_parse_vocab": None,
        "memory_rung": None,
        "memory_truncated": None,
        "served_weights_revision": None,
        "agent_node_idx": None,
        "run_seed": None,
        "tokenizer_revision": None,
        "belief_salvage_kind": None,
        "belief_salvage_attempt": None,
        "call_latency_s": None,
        "belief_call_latency_s": None,
        "prompt_tokens": None,
        "belief_prompt_tokens": None,
        "response_id": None,
        "belief_response_id": None,
        "request_seed": None,
        "belief_request_seed": None,
        "winning_attempt": None,
        "winning_raw_completion": None,
        "memory_chars_raw": None,
        "memory_chars_sent": None,
        "posterior_fell_back": None,
    }

    def _play(self, cfr_policy, tmp_path, **kw):
        from lmopt.games.loaders import load_kuhn

        writer = MatchWriter(str(tmp_path / "fixed0.parquet"))
        records = run_match(
            load_kuhn(),
            RandomAgent(seed=1),
            CFRRefAgent(cfr_policy, seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=20,
            seed=42,
            writer=writer,
            **kw,
        )
        rows = []
        for rec in records:
            d = dataclasses.asdict(rec)
            for col in self.EXCLUDED:
                d.pop(col)
            rows.append(d)
        return rows

    def test_default_call_matches_the_fixed0_golden(self, cfr_policy, tmp_path):
        """A default call reproduces the stored golden records.

        What this does not detect: merely consuming the seat stream without
        applying the value. The seat stream
        is rooted disjoint from the deal (match_runner.SEAT_SALT), so
        drawing from it perturbs no record.
        """
        golden = json.loads(self.GOLDEN.read_text())
        assert golden["excluded_columns"] == list(self.EXCLUDED)
        rows = self._play(cfr_policy, tmp_path)
        assert len(rows) == len(golden["records"])
        for i, (got, want) in enumerate(zip(rows, golden["records"], strict=False)):
            added = {k: v for k, v in got.items() if k not in want}
            assert added == self.ADDED_SINCE_GOLDEN, (
                f"record {i} carries an undeclared new column (or lost one): a "
                f"schema addition must be declared in ADDED_SINCE_GOLDEN, with "
                f"the value it takes on this LM-free path"
            )
            assert {
                k: v for k, v in got.items() if k in want
            } == want, f"record {i} diverged from the golden"

    def test_explicit_fixed0_equals_the_default(self, cfr_policy, tmp_path):
        """The default must be fixed0, not merely resemble it.
        """
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        assert self._play(cfr_policy, tmp_path / "a") == self._play(
            cfr_policy, tmp_path / "b", seat_mode="fixed0"
        )


class TestProbeAndVariantParams:
    """probe_type and prompt_variant are run-level config (config over forks)
    rather than literals in the NodeRecord fill. This tests only the
    labelling channel a probe writes through, not probe injection itself."""

    def _run(self, cfr_policy, game, tmp_path, **kw):
        return run_match(
            game,
            RandomAgent(seed=1),
            CFRRefAgent(cfr_policy, seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=3,
            seed=42,
            writer=MatchWriter(str(tmp_path / "probe.parquet")),
            **kw,
        )

    def test_probe_type_reaches_every_row(self, game, cfr_policy, tmp_path):
        records = self._run(cfr_policy, game, tmp_path, probe_type="relabel")
        assert {r.probe_type for r in records} == {"relabel"}

    def test_unknown_probe_type_raises(self, game, cfr_policy, tmp_path):
        """PROBE_TYPE_DOMAIN is the declared domain (schema.py).
        """
        with pytest.raises(ValueError, match="unknown probe_type"):
            self._run(cfr_policy, game, tmp_path, probe_type="relable")

    def test_prompt_variant_reaches_every_row(self, game, cfr_policy, tmp_path):
        records = self._run(cfr_policy, game, tmp_path, prompt_variant="terse")
        assert {r.prompt_variant for r in records} == {"terse"}


class TestGameKeyProvenance:
    """A depth variant's registry key must be the Parquet `game` value -
    without it, scorers grouping by (game, matchup_id) silently mix depth
    variants."""

    def test_game_key_flows_to_game_column(self, tmp_path):
        """This test is about depth-variant provenance in general, not Dark
        Hex specifically. It uses `goofspiel_5`, a depth variant cheap to
        full-solve at 5 iterations (~1.4s), unlike darkhex_3x3, which is NOT CFR+-tractable
        (`solve_cfr(..., plus=True)` on it would OOM the same way
        `gather_worlds` does -- TabularPolicy enumeration MemoryErrors per
        `load_darkhex_3x3`'s own docstring)."""
        import pandas as pd

        from lmopt.games.loaders import load_game

        game = load_game("goofspiel_5")
        policy = solve_cfr(game, 5, plus=True)
        ref = CFRRefAgent(policy, seed=0)
        out = tmp_path / "goofspiel5.parquet"
        writer = MatchWriter(str(out))
        run_match(
            game,
            CFRRefAgent(policy, seed=1),
            CFRRefAgent(policy, seed=2),
            ref,
            n_hands=2,
            seed=42,
            writer=writer,
            game_key="goofspiel_5",
        )
        df = pd.read_parquet(out)
        assert (df["game"] == "goofspiel_5").all()

    def test_unknown_game_key_raises(self, game, cfr_policy, tmp_path):
        """A typo'd game_key must fail loudly - otherwise it would mislabel the
        Parquet and silently skip its family's P7/P8 precompute."""
        with pytest.raises(ValueError, match="unknown game_key"):
            run_match(
                game,
                CFRRefAgent(cfr_policy, seed=1),
                CFRRefAgent(cfr_policy, seed=2),
                CFRRefAgent(cfr_policy, seed=0),
                n_hands=1,
                seed=42,
                writer=MatchWriter(str(tmp_path / "x.parquet")),
                game_key="darkhex_2x4",
            )

    def test_game_column_unchanged_without_game_key(self, game, cfr_policy, tmp_path):
        """Canonical path (no game_key) still derives the short key - zero diff."""
        import pandas as pd

        out = tmp_path / "kuhn.parquet"
        run_match(
            game,
            CFRRefAgent(cfr_policy, seed=1),
            CFRRefAgent(cfr_policy, seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=2,
            seed=42,
            writer=MatchWriter(str(out)),
        )
        df = pd.read_parquet(out)
        assert (df["game"] == "kuhn").all()


def test_cfr_action_is_invalid_action_iff_off_support():
    """Structural IFF sanity check on the NodeRecord schema itself: the sentinel
    and the flag must be able to encode the relation at all (field exists, the
    relation type-checks on hand-built records). This is not a production-path
    test - it would still pass if match_runner.py's flag-setting logic were
    deleted, because these NodeRecords are constructed by hand, not produced by
    run_match. The real hazard (legal_mask[cfr_action] with cfr_action == -1
    silently negative-indexing to the last legal-mask element instead of
    raising) and the production-path IFF are both exercised end-to-end by
    test_off_support_flows_through_run_match_end_to_end below.
    """
    fields = {f.name for f in dataclasses.fields(NodeRecord)}
    assert "cfr_ref_off_support" in fields

    rec_off = _minimal_record(
        cfr_action=pyspiel.INVALID_ACTION, cfr_ref_off_support=True
    )
    rec_on = _minimal_record(cfr_action=0, cfr_ref_off_support=False)
    for rec in (rec_off, rec_on):
        assert (rec.cfr_action == pyspiel.INVALID_ACTION) == rec.cfr_ref_off_support


def test_invalid_action_sentinel_is_negative():
    """If OpenSpiel ever changed this, the negative-indexing guard exercised in
    test_off_support_flows_through_run_match_end_to_end would silently stop
    guarding anything."""
    assert pyspiel.INVALID_ACTION == -1


def test_off_support_flows_through_run_match_end_to_end(tmp_path, game):
    """The IFF test above is tautological - it hand-builds the NodeRecords it
    then asserts a relation over, so it proves nothing about match_runner.py.
    This test drives a real partial reference (a truncated Kuhn table) through run_match and
    checks the flag rides durably through every logged row of the production
    path, not a synthetic pair.

    Also exercises the concrete negative-indexing hazard the flag exists to
    prevent: legal_mask[cfr_action] with cfr_action == -1 silently returns the
    last mask element instead of raising. On-support rows must bound-check
    cleanly; off-support rows must carry the sentinel so a consumer is forced
    to branch on cfr_ref_off_support before ever indexing legal_mask.
    """
    import pickle

    from lmopt.games.depth_reference import load_depth_reference
    from tests.test_depth_reference import _truncated_kuhn_payload

    payload = _truncated_kuhn_payload(game, keep=2)
    path = tmp_path / "trunc.pkl"
    path.write_bytes(pickle.dumps(payload))
    ref = load_depth_reference(path, game, mode="uniform")

    cfr_agent = CFRRefAgent(ref, seed=0)
    writer = MatchWriter(str(tmp_path / "trunc_match.parquet"))
    records = run_match(
        game,
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        cfr_agent,
        n_hands=30,
        seed=7,
        writer=writer,
    )

    assert len(records) > 0
    off_rows = [r for r in records if r.cfr_ref_off_support]
    on_rows = [r for r in records if not r.cfr_ref_off_support]
    assert (
        off_rows
    ), "expected at least one off-support row from the truncated reference"
    assert on_rows, "expected at least one on-support row"

    for rec in records:
        assert (
            rec.cfr_ref_off_support is not None
        ), "cfr_ref_off_support must never be null"
        # The IFF that matters: values produced by the production path, not
        # hand-built ones.
        assert (rec.cfr_action == pyspiel.INVALID_ACTION) == rec.cfr_ref_off_support
        if rec.cfr_ref_off_support:
            assert rec.cfr_action == pyspiel.INVALID_ACTION
        else:
            # The negative-indexing guard: a real bound check must hold before
            # any consumer does legal_mask[cfr_action].
            assert 0 <= rec.cfr_action < len(rec.legal_mask)
            assert rec.legal_mask[rec.cfr_action] == 1


def test_darkhex_partial_oracle_policy_skips_the_tree_walk(tmp_path):
    """A partial Dark Hex reference must skip the full tree walk.

    ``gather_p0_worlds`` (darkhex_posterior.py) is a full game-tree walk -
    worse than TabularPolicy, since it also deserialises a full state object
    at every P0 node. With darkhex_3x3 registered, this branch is reachable
    with a PARTIAL oracle_policy: `family` resolves to "darkhex" for any Dark Hex
    board (families.py keys on the short_name, not the size), and
    shard.py's own routing is what supplies a DepthReference as
    ``oracle_policy`` for a darkhex-family game. Exercised here on the small
    2x2 board (441 states, cheap) - the guard is duck-typed on the policy
    object, not the board size, so this proves the branch is SKIPPED without
    needing the expensive 3x3 tree.

    Also pins the crash the guard avoids: without the
    ``darkhex_reference_is_partial`` branch, execution would fall through to
    ``compute_true_posterior``, which calls ``state.chance_outcomes()`` on
    the initial state - Dark Hex has zero leading chance nodes, so that call
    raises. true_posterior must simply stay unset for every P0 row instead.
    """
    import pickle

    from lmopt.games.depth_reference import load_depth_reference
    from lmopt.games.loaders import load_darkhex
    from tests.test_depth_reference import _truncated_kuhn_payload

    dh_game = load_darkhex()  # 2x2 - small, tractable, same "no deal" shape as 3x3
    payload = _truncated_kuhn_payload(dh_game, keep=2)
    path = tmp_path / "trunc_dh.pkl"
    path.write_bytes(pickle.dumps(payload))
    partial_ref = load_depth_reference(path, dh_game, mode="uniform")

    cfr_agent = CFRRefAgent(partial_ref, seed=0)
    writer = MatchWriter(str(tmp_path / "trunc_dh_match.parquet"))
    records = run_match(
        dh_game,
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        cfr_agent,
        n_hands=5,
        seed=7,
        writer=writer,
        oracle_policy=partial_ref,
    )

    assert len(records) > 0
    p0_rows = [r for r in records if r.player_to_act == 0]
    assert p0_rows, "expected at least one P0 (player-under-test) row"
    assert all(r.true_posterior is None for r in p0_rows), (
        "a partial reference must not fabricate a true_posterior for Dark Hex — "
        "P8 is signed off 'not applicable at 3x3, reason recorded'"
    )


class TestHandEndPayoffIsPerSeat:
    """`observe_hand_end` must carry each agent its own terminal payoff.

    Why this is needed at all: `_build_action_history` returns `(action,
    player)` pairs and skips chance nodes, so the history argument contains
    no payoff and no card, while `returns = state.returns()` is computed a
    few lines above the hook calls. A cross-hand memory that states "you
    lost 2" is not buildable from the history alone.

    Why PER-seat specifically: `run_match` hands both agents the *same*
    `action_history` object, so a single shared payload cannot express "your"
    payoff in a zero-sum game where the two seats' returns are negatives of
    each other.
    """

    class _Recorder:
        """Minimal Agent that records what the hook delivered."""

        backend = "cpu"
        requires_model = False

        def __init__(self, inner):
            self._inner = inner
            self.payoffs: list = []

        def reset(self, game, seat, opponent_id=None):
            self.seat = seat
            self._inner.reset(game, seat, opponent_id)

        def begin_hand(self, seat):
            self.seat = seat
            self._inner.begin_hand(seat)

        def act(self, state):
            return self._inner.act(state)

        def belief(self, state):
            return self._inner.belief(state)

        def observe_hand_end(self, history, own_payoff=None):
            self.payoffs.append(own_payoff)

    def test_each_agent_receives_its_own_payoff_not_a_shared_one(
        self, game, cfr_policy, tmp_path
    ):
        """Zero-sum Kuhn: the two seats' recorded payoffs must be negatives.
        """
        a0 = self._Recorder(RandomAgent(seed=1))
        a1 = self._Recorder(RandomAgent(seed=2))
        cfr_ref = CFRRefAgent(cfr_policy)
        writer = MatchWriter(str(tmp_path / "hookpayoff.parquet"))
        run_match(game, a0, a1, cfr_ref, n_hands=25, seed=42, writer=writer)

        assert len(a0.payoffs) == 25
        assert len(a1.payoffs) == 25
        assert all(p is not None for p in a0.payoffs), "hook delivered no payoff"
        for p0, p1 in zip(a0.payoffs, a1.payoffs, strict=False):
            assert p0 == -p1, f"seats not zero-sum: {p0} vs {p1}"
        # Kuhn payoffs are non-zero at every terminal, so a shared payload
        # cannot coincidentally satisfy the above.
        assert all(p != 0 for p in a0.payoffs)


# ---------------------------------------------------------------------------
# The P6 off-path driver.
# ---------------------------------------------------------------------------


class _AlwaysPassPolicy:
    """An opponent policy that never bets, so a driven bet has zero likelihood.

    Not a CFR artefact: CFR+'s average policy puts strictly positive mass on
    both Kuhn actions at every info-set (min 0.08), so
    the uniform-posterior fallback is not reachable from the real reference on
    this game. A policy with an exact zero is what makes the driven-history
    fallback observable at all.
    """

    def action_probabilities(self, state, player_id=None):
        return {0: 1.0, 1: 0.0}


class TestOffPathDriving:
    """`drive_openings` forces the first plies of every
    hand so the agents choose from inside a subgame the reference reaches with
    (near) zero probability. The forced plies are not decisions, so they are
    not logged; everything after them is.
    """

    def _run(self, cfr_policy, game, tmp_path, name="drive", n_hands=6, **kw):
        return run_match(
            game,
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=n_hands,
            seed=42,
            writer=MatchWriter(str(tmp_path / f"{name}.parquet")),
            **kw,
        )

    def test_the_forced_plies_are_not_logged(self, game, cfr_policy, tmp_path):
        """P-drive. Driving Kuhn with [pass, bet] leaves exactly one decision
        per hand (seat 0 answering the bet), and every logged info-set carries
        the forced prefix "pb".
        """
        records = self._run(
            cfr_policy,
            game,
            tmp_path,
            drive_openings=[0, 1],
            probe_type="offpath_drive",
        )
        assert len(records) == 6
        assert {r.node_idx for r in records} == {0}
        assert {r.player_to_act for r in records} == {0}
        assert all(r.info_set.endswith("pb") for r in records), [
            r.info_set for r in records
        ]

    def test_every_driven_row_carries_the_offpath_probe_label(
        self, game, cfr_policy, tmp_path
    ):
        """P-offpath-flag.
        """
        records = self._run(
            cfr_policy,
            game,
            tmp_path,
            drive_openings=[0, 1],
            probe_type="offpath_drive",
        )
        assert {r.probe_type for r in records} == {"offpath_drive"}
        assert {r.prompt_variant for r in records} == {"base"}

    def test_driving_without_the_offpath_label_is_refused(
        self, game, cfr_policy, tmp_path
    ):
        """The label and the manipulation are one claim, enforced in both
        directions.
        """
        with pytest.raises(ValueError, match="drive_openings"):
            self._run(cfr_policy, game, tmp_path, drive_openings=[0, 1])

    def test_the_offpath_label_without_a_driver_is_refused(
        self, game, cfr_policy, tmp_path
    ):
        """The other direction, and the library-level twin of run_sweep's
        startup guard.
        """
        with pytest.raises(ValueError, match="offpath_drive"):
            self._run(cfr_policy, game, tmp_path, probe_type="offpath_drive")

    def test_an_empty_drive_is_refused(self, game, cfr_policy, tmp_path):
        with pytest.raises(ValueError, match="drive_openings"):
            self._run(
                cfr_policy,
                game,
                tmp_path,
                drive_openings=[],
                probe_type="offpath_drive",
            )

    def test_an_illegal_forced_action_refuses_loudly(self, game, cfr_policy, tmp_path):
        """An illegal forced action is refused, never skipped.
        """
        with pytest.raises(ValueError, match="not legal"):
            self._run(
                cfr_policy,
                game,
                tmp_path,
                drive_openings=[7],
                probe_type="offpath_drive",
            )

    def test_a_drive_that_ends_the_hand_refuses_loudly(
        self, game, cfr_policy, tmp_path
    ):
        """Kuhn's [pass, pass] is a terminal history: driving there leaves no
        decision for anyone to make.
        """
        with pytest.raises(ValueError, match="terminal"):
            self._run(
                cfr_policy,
                game,
                tmp_path,
                drive_openings=[0, 0],
                probe_type="offpath_drive",
            )

    def test_driven_hands_still_draw_their_deals_from_the_rng(
        self, game, cfr_policy, tmp_path
    ):
        """Chance nodes stay chance nodes: only the player plies are forced.
        """
        records = self._run(
            cfr_policy,
            game,
            tmp_path,
            name="deals",
            n_hands=20,
            drive_openings=[0, 1],
            probe_type="offpath_drive",
        )
        # The acting seat's own card is the first character of the Kuhn
        # info-set string.
        assert len({r.info_set[0] for r in records}) > 1

    def test_a_zero_reach_driven_row_logs_no_true_posterior(
        self, game, cfr_policy, tmp_path
    ):
        """`compute_true_posterior` substitutes a
        uniform posterior when every hypothesis's likelihood is zero, and a
        driven opening is chosen precisely for near-zero reach, so on a driven
        arm that fallback is systematically reachable. P5 (ECE) and P7a (TVD)
        consume this column.
        """
        records = self._run(
            cfr_policy,
            game,
            tmp_path,
            name="zeroreach",
            drive_openings=[1],
            probe_type="offpath_drive",
            oracle_policy=_AlwaysPassPolicy(),
        )
        assert len(records) == 6
        assert {r.true_posterior for r in records} == {None}

    def test_a_positive_reach_driven_row_still_logs_its_posterior(
        self, game, cfr_policy, tmp_path
    ):
        """Falsify the falsifier: the None above must come from the fallback,
        not from driving. Kuhn's CFR+ reference bets with positive probability
        at every info-set, so the same drive under the real reference has a
        computable posterior.
        """
        records = self._run(
            cfr_policy,
            game,
            tmp_path,
            name="posreach",
            drive_openings=[1],
            probe_type="offpath_drive",
            oracle_policy=cfr_policy,
        )
        assert all(r.true_posterior is not None for r in records)

    def test_the_uniform_fallback_is_unchanged_on_an_undriven_hand(
        self, game, cfr_policy, tmp_path
    ):
        """The null-posterior rule is scoped to driven hands, so that no
        collected arm's column changes meaning. AlwaysBet against an
        always-pass oracle reaches the same zero-likelihood history without
        any driving, and there the uniform fallback is what is logged.
        """
        from lmopt.agents.classical import AlwaysBetAgent

        records = run_match(
            game,
            AlwaysBetAgent(),
            RandomAgent(seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=4,
            seed=42,
            writer=MatchWriter(str(tmp_path / "undriven.parquet")),
            oracle_policy=_AlwaysPassPolicy(),
        )
        seat1 = [r for r in records if r.player_to_act == 1]
        assert seat1, "no seat-1 rows: the always-bet history was not reached"
        for rec in seat1:
            probs = json.loads(rec.true_posterior)
            assert sorted(round(p, 6) for p in probs.values() if p) == [0.5, 0.5]

    def test_a_driven_match_under_random_seats_carries_both_seats(
        self, game, cfr_policy, tmp_path
    ):
        """Driving and per-hand seat swaps compose.
        The drive forces the first plies regardless of who sits where, so the
        driven subgame's decisions land at both of agent0's seats across
        hands.
        """
        records = self._run(
            cfr_policy,
            game,
            tmp_path,
            name="seats",
            n_hands=24,
            drive_openings=[1],
            probe_type="offpath_drive",
            seat_mode="random",
        )
        assert {r.lm_seat for r in records} == {0, 1}
        assert {r.probe_type for r in records} == {"offpath_drive"}
        assert all(r.info_set.endswith("b") for r in records)


class _SlowRandomAgent(RandomAgent):
    """RandomAgent with an artificial per-decision delay.

    A CFR-vs-CFR (or plain RandomAgent) hand finishes in microseconds, faster
    than tqdm's own default mininterval, so an unthrottled and a throttled
    run would print about the same handful of redraws either way -- the
    difference the throttle makes only shows up once hands take real, LM-latency
    scale time, which this stand-in simulates without a network call.
    """

    def act(self, state):
        time.sleep(0.02)
        return super().act(state)


class _NonTTYStream(io.StringIO):
    def isatty(self):
        return False


class _TTYStream(io.StringIO):
    def isatty(self):
        return True


class TestHandBarNonTTYThrottle:
    """Unthrottled, run_match's per-hand tqdm bar buries the per-shard
    timing line under one `\\r`-redraw per hand when stderr is redirected to
    a log file (a `nohup ... > log 2>&1` run)."""

    def test_hand_bar_is_throttled_when_stderr_is_not_a_tty(
        self, game, cfr_ref, tmp_path, monkeypatch
    ):
        fake_stderr = _NonTTYStream()
        monkeypatch.setattr(sys, "stderr", fake_stderr)
        writer = MatchWriter(str(tmp_path / "throttled.parquet"))

        run_match(
            game,
            _SlowRandomAgent(seed=3),
            cfr_ref,
            cfr_ref,
            n_hands=20,
            seed=42,
            writer=writer,
        )

        redraws = fake_stderr.getvalue().count("\r")
        assert (
            redraws <= 4
        ), f"expected a throttled non-TTY hand bar, got {redraws} redraws"

    def test_hand_bar_keeps_its_default_cadence_on_a_tty(
        self, game, cfr_ref, tmp_path, monkeypatch
    ):
        """The other half of the same property: an interactive terminal must
        be unaffected by the non-TTY throttle."""
        pytest.importorskip("tqdm")
        fake_stderr = _TTYStream()
        monkeypatch.setattr(sys, "stderr", fake_stderr)
        writer = MatchWriter(str(tmp_path / "tty.parquet"))

        run_match(
            game,
            _SlowRandomAgent(seed=3),
            cfr_ref,
            cfr_ref,
            n_hands=20,
            seed=42,
            writer=writer,
        )

        redraws = fake_stderr.getvalue().count("\r")
        assert (
            redraws >= 6
        ), f"expected the default TTY redraw cadence, got only {redraws}"


class TestPosteriorFallbackCounting:
    """Posterior fallback tallies are counted per game and handed back
    through run_match's `posterior_fallback_out` sink, which
    orchestration/shard.py writes to a sidecar and run_sweep.py aggregates into
    run_meta.json. A module global would not work: it would be written in the
    shard's worker process and read in the sweep parent, so a pooled sweep
    would aggregate an empty dict.
    """

    @pytest.fixture
    def game(self):
        return load_kuhn()

    @pytest.fixture
    def cfr_policy(self, game):
        return solve_cfr(game, iterations=100)

    def test_fallback_counts_are_tallied_when_fallbacks_occur(
        self, game, cfr_policy, tmp_path
    ):
        """A sweep whose posterior computation falls back N>0 times records
        those counts in the shape {game: {"fell_back": int, "rows": int}}.
        """
        from lmopt.agents.classical import AlwaysBetAgent

        counts: dict[str, dict[str, int]] = {}
        # AlwaysBetAgent against an always-pass oracle reaches a zero-likelihood
        # history, forcing the fallback
        run_match(
            game,
            AlwaysBetAgent(),
            RandomAgent(seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=4,
            seed=42,
            writer=MatchWriter(str(tmp_path / "fallback.parquet")),
            oracle_policy=_AlwaysPassPolicy(),
            posterior_fallback_out=counts,
        )

        assert "kuhn" in counts
        assert counts["kuhn"]["fell_back"] > 0
        assert counts["kuhn"]["rows"] >= counts["kuhn"]["fell_back"]

    def test_zero_fallback_sweep_writes_zeros(self, game, cfr_policy, tmp_path):
        """A zero-fallback sweep writes zeros, so absence vs zero is
        distinguishable from an old run_meta (key always present when oracle
        path is taken).
        """
        counts: dict[str, dict[str, int]] = {}

        run_match(
            game,
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=4,
            seed=42,
            writer=MatchWriter(str(tmp_path / "nofallback.parquet")),
            oracle_policy=cfr_policy,
            posterior_fallback_out=counts,
        )

        assert "kuhn" in counts
        assert counts["kuhn"]["fell_back"] == 0
        assert counts["kuhn"]["rows"] > 0

    def test_serialized_true_posterior_unchanged_for_fallen_back_row(
        self, game, cfr_policy, tmp_path
    ):
        """The fallback BEHAVIOUR (when true_posterior is written) must be
        proven unchanged: the serialized true_posterior for a fallen-back
        undriven row is the uniform fallback, with or without counting.

        This is the "counting must not change behaviour" assertion.
        """
        from lmopt.agents.classical import AlwaysBetAgent

        records = run_match(
            game,
            AlwaysBetAgent(),
            RandomAgent(seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=4,
            seed=42,
            writer=MatchWriter(str(tmp_path / "unchanged.parquet")),
            oracle_policy=_AlwaysPassPolicy(),
        )

        # Seat 1 rows are the ones that hit the fallback
        seat1 = [r for r in records if r.player_to_act == 1]
        assert seat1, "no seat-1 rows: the fallback was not reached"

        for rec in seat1:
            probs = json.loads(rec.true_posterior)
            # The uniform fallback over kuhn's 3 cards (excluding own) = [0.5, 0.5]
            # for the two remaining cards
            assert sorted(round(p, 6) for p in probs.values() if p) == [0.5, 0.5]


def test_pooling_adds_the_counts_and_never_pools_the_shard_local_indices():
    """
    Two shards' sidecars, each discarding hands 0 and 1, pool to summed counts and no index list:
    `hand_idx` restarts per shard, so a concatenated list would name hands that two shards both
    call 0, and a row identity that is subtly wrong is worse than none.
    """
    from lmopt.harness.match_runner import merge_driven_discard_counts

    shard = {
        "kuhn": {
            "driven_hands": 2,
            "discarded_hands": 2,
            "discarded_hand_idx": [0, 1],
        }
    }
    pooled: dict[str, dict[str, int]] = {}
    merge_driven_discard_counts(pooled, shard)
    merge_driven_discard_counts(pooled, shard)
    assert pooled == {"kuhn": {"driven_hands": 4, "discarded_hands": 4}}


class TestDrivenDiscardCounting:
    """A driven hand whose drive-parity consumed the tested agent's seat
    contributes no row for that agent (e.g. a driven pilot: 60/60 rows
    at seat 1, the 24 seat-0 hands invisible). run_match counts those hands
    per game, hands the tally back through the `driven_discards_out` sink
    (the same out-parameter surface `posterior_fallback_out` uses), and
    prints one loud summary line when the count is nonzero, so the discard
    is never silent.
    """

    def _run(
        self, cfr_policy, game, tmp_path, name="discard", n_hands=24, sink=None, **kw
    ):
        return run_match(
            game,
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            CFRRefAgent(cfr_policy, seed=0),
            n_hands=n_hands,
            seed=42,
            writer=MatchWriter(str(tmp_path / f"{name}.parquet")),
            driven_discards_out=sink,
            **kw,
        )

    def test_random_seat_driving_counts_and_prints_the_discarded_hands(
        self, game, cfr_policy, tmp_path, capsys
    ):
        """A random-seat driven match in which some hands' surviving seat is
        not the agent's emits a nonzero discard count through the sink and
        prints one loud summary line carrying that count.

        Kuhn drive [1] (bet) consumes seat 0's only opening ply, so the one
        surviving decision per hand belongs to seat 1: every hand agent0 drew
        seat 0 is exactly a discarded hand, and at seed 42 / 24 hands both
        seats occur (pinned by test_a_driven_match_under_random_seats_
        carries_both_seats above).

        The identities are checked against the hands that logged no agent row.
        """
        sink: dict[str, dict[str, int | list[int]]] = {}
        records = self._run(
            cfr_policy,
            game,
            tmp_path,
            sink=sink,
            drive_openings=[1],
            probe_type="offpath_drive",
            seat_mode="random",
        )
        hands_with_agent0_row = {
            r.hand_idx for r in records if r.player_to_act == r.lm_seat
        }
        expected = 24 - len(hands_with_agent0_row)
        assert expected > 0, "fixture defect: no hand lost its agent0 row"
        assert sink["kuhn"]["discarded_hands"] == expected
        assert sink["kuhn"]["driven_hands"] == 24
        # The identities, from an oracle that does not use the counter: the hands that logged no
        # row for the agent under test. Both seats occur, so the list is neither empty nor every
        # hand, and an off-by-one or a swapped index cannot match it.
        assert sink["kuhn"]["discarded_hand_idx"] == sorted(
            set(range(24)) - hands_with_agent0_row
        )
        assert (
            len(sink["kuhn"]["discarded_hand_idx"]) == sink["kuhn"]["discarded_hands"]
        )
        out = capsys.readouterr().out
        assert f"{expected}/24" in out and "no row" in out, out

    def test_the_fixed_orientation_pair_discards_all_or_none(
        self, game, cfr_policy, tmp_path, capsys
    ):
        """The drive's parity fixes which seat survives, so under drive [1]
        fixed0 discards every hand (agent0's seat is consumed) and fixed1
        discards NONE -- and the zero is written, not omitted, so "measured
        zero" stays distinguishable from "not a driven match".

        Two literal-seat mutants, one per half. Every surviving row under
        drive [1] sits at seat 1, so:
        - literal 0 (`rec.player_to_act == 0`): the fixed0 half stays green
          (agent0_seat is 0 there), but the fixed1 half finds no seat-0 rows
          and reports 24/24 discarded, printing -- its zero-discard and
          no-print assertions redden.
        - literal 1 (`rec.player_to_act == 1`): the fixed1 half stays green (agent0_seat is 1
          there), but the fixed0 half finds seat-1 rows on every hand and
          reports 0/24 -- its `discarded_hands == 24` assertion reddens.
        """
        sink0: dict[str, dict[str, int | list[int]]] = {}
        self._run(
            cfr_policy,
            game,
            tmp_path,
            name="fixed0",
            sink=sink0,
            drive_openings=[1],
            probe_type="offpath_drive",
            seat_mode="fixed0",
        )
        assert sink0["kuhn"] == {
            "driven_hands": 24,
            "discarded_hands": 24,
            "discarded_hand_idx": list(range(24)),
        }
        assert "24/24" in capsys.readouterr().out

        sink1: dict[str, dict[str, int | list[int]]] = {}
        self._run(
            cfr_policy,
            game,
            tmp_path,
            name="fixed1",
            sink=sink1,
            drive_openings=[1],
            probe_type="offpath_drive",
            seat_mode="fixed1",
        )
        assert sink1["kuhn"] == {
            "driven_hands": 24,
            "discarded_hands": 0,
            "discarded_hand_idx": [],
        }
        out = capsys.readouterr().out
        assert "no row" not in out, f"a zero count must not print: {out!r}"

    def test_an_undriven_match_reports_no_driven_counts(
        self, game, cfr_policy, tmp_path, capsys
    ):
        """An undriven match leaves the sink empty: {} means "not a driven
        match", a different claim from a driven match's measured zero.
        """
        sink: dict[str, dict[str, int]] = {}
        self._run(cfr_policy, game, tmp_path, name="undriven", sink=sink)
        assert sink == {}
        assert "no row" not in capsys.readouterr().out


class TestPublicGitShaHelper:
    """Public get_git_sha() helper for external scripts."""

    def test_get_git_sha_returns_a_string(self):
        """The public helper returns a non-empty string (sha or "unknown").
        """
        sha = get_git_sha()
        assert isinstance(sha, str)
        assert len(sha) > 0
        # Either a 40-hex git sha or the "unknown" fallback
        assert sha == "unknown" or (
            len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
        )

    def test_private_alias_delegates_to_public(self):
        """The private _get_git_sha() delegates to the public get_git_sha().
        """
        assert _get_git_sha() == get_git_sha()


class TestRunMatchGitShaParameter:
    """Git-HEAD race: if `run_match` resolved `_get_git_sha()` internally on
    every invocation, a caller dispatching many shards into separate worker
    processes would get one independent git-subprocess resolution per shard,
    and if the checkout's HEAD moved mid-sweep, different shards of the same
    arm would record different `model_revision` values (a 100-shard Qwen arm
    once split 80/20 between two shas). An optional `git_sha` is threaded
    through `run_match`, so a multi-shard caller (`orchestration/shard.py`)
    can resolve it once and hand every shard the same value.
    """

    def test_given_git_sha_is_stamped_verbatim_and_not_re_resolved(
        self, game, cfr_policy, cfr_ref, tmp_path, monkeypatch
    ):
        import lmopt.harness.match_runner as match_runner_mod

        def _boom():
            raise AssertionError(
                "run_match must not resolve its own git sha when the "
                "caller already supplied one"
            )

        monkeypatch.setattr(match_runner_mod, "_get_git_sha", _boom)

        writer = MatchWriter(str(tmp_path / "explicit_sha.parquet"))
        records = run_match(
            game,
            RandomAgent(seed=1),
            cfr_ref,
            cfr_ref,
            n_hands=2,
            seed=42,
            writer=writer,
            git_sha="explicit-test-sha-0123456789",
        )

        assert records, "match produced no rows to check"
        assert {r.model_revision for r in records} == {"explicit-test-sha-0123456789"}

    def test_git_sha_defaults_to_resolving_its_own_when_not_given(
        self, game, cfr_ref, tmp_path, monkeypatch
    ):
        """Non-regression: a caller that omits `git_sha` (as every other test
        in this file does) and must keep getting the resolved-here value.
        """
        import lmopt.harness.match_runner as match_runner_mod

        monkeypatch.setattr(
            match_runner_mod, "_get_git_sha", lambda: "resolved-internally-sha"
        )

        writer = MatchWriter(str(tmp_path / "default_sha.parquet"))
        records = run_match(
            game,
            RandomAgent(seed=1),
            cfr_ref,
            cfr_ref,
            n_hands=2,
            seed=42,
            writer=writer,
        )

        assert records, "match produced no rows to check"
        assert {r.model_revision for r in records} == {"resolved-internally-sha"}


class _CallLogRecordingAgent(RandomAgent):
    """RandomAgent that records every attach_call_log call and can raise on a chosen act."""

    def __init__(self, seed, raise_on_act=None):
        super().__init__(seed=seed)
        self.attached = []
        self._raise_on_act = raise_on_act
        self._n_acts = 0

    def attach_call_log(self, call_log, run_id=None):
        self.attached.append(call_log)

    def act(self, state):
        self._n_acts += 1
        if self._raise_on_act is not None and self._n_acts == self._raise_on_act:
            raise RuntimeError("simulated agent failure")
        return super().act(state)


class TestCallLogDetach:
    def test_a_raising_match_still_detaches_the_call_log(self, game, cfr_ref, tmp_path):
        """
        an agent that raises mid-match leaves no handle to this shard's call log: the detach
        runs in a `finally`, so a reused agent cannot log into the wrong shard
        """
        writer = MatchWriter(str(tmp_path / "m.parquet"))
        agent = _CallLogRecordingAgent(seed=1, raise_on_act=3)
        with pytest.raises(RuntimeError, match="simulated agent failure"):
            run_match(game, agent, cfr_ref, cfr_ref, n_hands=5, seed=0, writer=writer)
        assert agent.attached[0] is writer.call_log
        assert agent.attached[-1] is None

    def test_a_clean_match_attaches_then_detaches_once(self, game, cfr_ref, tmp_path):
        """
        on the success path the call log is attached once and detached once, as before
        """
        writer = MatchWriter(str(tmp_path / "m.parquet"))
        agent = _CallLogRecordingAgent(seed=1)
        run_match(game, agent, cfr_ref, cfr_ref, n_hands=3, seed=0, writer=writer)
        assert agent.attached == [writer.call_log, None]
