"""Parquet round-trip and schema validation tests."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("polars")

import polars as pl

from lmopt.agents.bayesian import BayesianAgent
from lmopt.agents.cfr_ref import CFRRefAgent
from lmopt.games.cfr import solve_cfr
from lmopt.games.loaders import load_kuhn
from lmopt.harness.match_runner import run_match
from lmopt.logging.schema import (
    MEMORY_RUNG_DOMAIN,
    NodeRecord,
    OFFPATH_PROBE_TYPE,
    PARQUET_SCHEMA,
    PROBE_TYPE_DOMAIN,
    REQUIRED_COLUMNS,
)
from lmopt.prompting.memory import MEMORY_RUNGS
from lmopt.logging.writer import MatchWriter, read_parquet


def _full_record(**overrides) -> NodeRecord:
    """Construct a NodeRecord with every field filled; overrides win.

    Independent of test_match_runner.py's own `_minimal_record` (different
    file, deliberately not cross-imported) so this file's round-trip test
    exercises NodeRecord/PARQUET_SCHEMA agreement without depending on
    another test module's fixture staying in sync.
    """
    base = dict(
        run_id="run0",
        matchup_id="lm_vs_cfr_ref",
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
        raw_completion="0",
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
        model_revision="deadbeef",
        timestamp="2026-08-01T00:00:00+00:00",
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
        # served_weights_revision is deliberately not passed here: the base
        # record exercises NodeRecord's nullable default (the column-additive
        # constraint -- every pre-existing construction site must keep
        # working without the kwarg).
        lm_seat=0,
        actor="cfr_ref",
        action_str="Pass",
        prompt_variant="base",
    )
    base.update(overrides)
    return NodeRecord(**base)


# Columns that are intentionally null in a non-LM match (no LM, no tensor logging).
NULLABLE_IN_M1 = {
    "prompt",
    "raw_completion",
    "info_state_tensor",
    "elicited_belief",
    # LM-only provenance columns, null for the Bayesian+CFR-only
    # fixture below (no LMVllmAgent involved, so _get_lm_metadata's getattr
    # defaults apply).
    "belief_raw_completion",
    "model_id",
    "chat_template_hash",
    "prompt_schema_version",
    # The content hash of the prompt spec this row was
    # rendered under -- LM-only for the same reason (no prompt, no spec).
    "prompt_spec_hash",
    # The belief decoding protocol, null on the same rule (no LM
    # agent acted, so nothing was decoded).
    "belief_protocol",
    # Seven additive LM-only columns (reasoning traces, finish
    # reasons, belief prompt, token counts), null for non-LM agents.
    "reasoning_trace",
    "belief_reasoning_trace",
    "finish_reason",
    "belief_finish_reason",
    "belief_prompt",
    "completion_tokens",
    "belief_completion_tokens",
    # The served model's HF
    # weights revision, resolved at launch and threaded like
    # chat_template_hash -- LM-only, null for the non-LM fixture below.
    "served_weights_revision",
    # The served endpoint's HF
    # tokenizer revision, threaded like served_weights_revision -- LM-only,
    # null for the non-LM fixture below.
    "tokenizer_revision",
    # Seed re-derivability columns, read off the
    # acting agent -- null here because neither Bayesian nor CFRRef exposes a
    # per-decision ordinal or a run_seed.
    "agent_node_idx",
    "run_seed",
    # Belief salvage provenance: set only when a bare-integer-key recovery
    # supplied an LM row's belief, so null for the non-LM fixture below.
    "belief_salvage_kind",
    "belief_salvage_attempt",
}


@pytest.fixture(scope="module")
def parquet_df(tmp_path_factory):
    game = load_kuhn()
    policy = solve_cfr(game, 500)
    cfr_ref = CFRRefAgent(policy, seed=0)
    bayesian = BayesianAgent(opponent_policy=policy, seed=1)

    tmp = tmp_path_factory.mktemp("parquet")
    path = str(tmp / "test.parquet")
    writer = MatchWriter(path)
    run_match(game, bayesian, cfr_ref, cfr_ref, n_hands=5, seed=42, writer=writer)
    return read_parquet(path)


class TestSchemaPresence:
    def test_all_columns_present(self, parquet_df):
        for col in PARQUET_SCHEMA:
            assert col in parquet_df.columns, f"Missing column: {col!r}"

    def test_no_extra_columns(self, parquet_df):
        assert set(parquet_df.columns) == set(PARQUET_SCHEMA.keys())


class TestNullConstraints:
    def test_required_columns_non_null(self, parquet_df):
        for col in REQUIRED_COLUMNS:
            null_count = parquet_df[col].null_count()
            assert null_count == 0, f"Column {col!r} has {null_count} nulls"

    def test_nullable_m1_columns_are_null(self, parquet_df):
        for col in NULLABLE_IN_M1:
            assert parquet_df[col].null_count() == len(
                parquet_df
            ), f"Column {col!r} should be all-null in a non-LM match"


class TestValueValidity:
    def test_cfr_policy_valid_json_sums_to_one(self, parquet_df):
        for val in parquet_df["cfr_policy"].to_list():
            probs = json.loads(val)
            assert abs(sum(probs.values()) - 1.0) < 1e-5

    def test_true_posterior_valid_json(self, parquet_df):
        for val in parquet_df["true_posterior"].drop_nulls().to_list():
            d = json.loads(val)
            assert abs(sum(d.values()) - 1.0) < 1e-5

    def test_game_column(self, parquet_df):
        assert set(parquet_df["game"].unique().to_list()) == {"kuhn"}

    def test_probe_type_column(self, parquet_df):
        """Pins two facts at once: the schema's declared probe_type domain
        (PROBE_TYPE_DOMAIN, the probe modes)
        and that a base (non-probe) match emits only "none" -- relabel/
        payoff_scale/offpath_drive come from probe arms.
        """
        assert PROBE_TYPE_DOMAIN == {
            "none",
            "relabel",
            "payoff_scale",
            "offpath_drive",
            "option_order_reversed",
        }
        actual = set(parquet_df["probe_type"].unique().to_list())
        assert actual.issubset(PROBE_TYPE_DOMAIN)
        assert actual == {"none"}

    def test_the_offpath_label_is_a_member_of_the_declared_domain(self):
        """`OFFPATH_PROBE_TYPE` is the label three layers
        check the off-path manipulation against (run_match, MatchupSpec,
        run_sweep) and the one value `score_property6` accepts.
        """
        assert OFFPATH_PROBE_TYPE in PROBE_TYPE_DOMAIN

    def test_memory_rung_domain_does_not_drift_from_source_of_truth(self):
        """MEMORY_RUNG_DOMAIN (schema.py, the
        constraint for logged memory_rung values) must stay synchronized with
        MEMORY_RUNGS (prompting/memory.py, the source of truth).
        """
        assert MEMORY_RUNG_DOMAIN == set(MEMORY_RUNGS)

    def test_backend_column(self, parquet_df):
        assert set(parquet_df["backend"].unique().to_list()) == {"cpu"}


class TestSchemaInferenceRegression:
    """Writer must not rely on polars schema inference (m7_pass1 pythia bug).

    polars' default infer_schema_length=100 locks a column's dtype from the
    first 100 rows. A base-model run produced >100 leading None values in
    elicited_belief followed by a parsed JSON string -> "could not append
    value of type str to the builder" at pl.DataFrame(rows), killing the
    shard. The declared PARQUET_SCHEMA must govern construction itself, not
    only a post-hoc cast.
    """

    def _rows(self, n_none: int):
        base = {col: None for col in PARQUET_SCHEMA}
        base.update(
            game="kuhn",
            matchup_id="lm_vs_bayesian",
            seed=42,
            hand_idx=0,
            node_idx=0,
            player_to_act=0,
            parsed_action=1,
            backend="cpu",
            model_revision="deadbeef",
        )
        rows = []
        for i in range(n_none):
            r = dict(base)
            r["node_idx"] = i
            rows.append(r)
        late = dict(base)
        late["node_idx"] = n_none
        late["elicited_belief"] = json.dumps({"0": 0.0, "1": 0.5, "2": 0.5})
        rows.append(late)
        return rows

    def test_late_first_string_after_100_nones_survives(self):
        from lmopt.logging.writer import _build_dataframe

        df = _build_dataframe(self._rows(150))
        assert df["elicited_belief"].dtype == pl.Utf8
        assert df["elicited_belief"][150] is not None
        assert df["elicited_belief"][0] is None

    def test_all_none_belief_still_utf8(self):
        from lmopt.logging.writer import _build_dataframe

        rows = self._rows(5)[:-1]
        df = _build_dataframe(rows)
        assert df["elicited_belief"].dtype == pl.Utf8


class TestBeliefAndProvenanceColumnsRoundTrip:
    """Seven belief and provenance columns (including
    `belief_protocol`): NodeRecord and PARQUET_SCHEMA are two representations
    of the same added columns that must agree -- schema.py's own docstring
    names the defect shape ("two representations that
    must agree, where only one is ever checked"). Round-tripping through the
    real MatchWriter (not just constructing the dataclass) is the check that
    they actually do, rather than trusting the dataclass and the polars dict
    to stay in sync by inspection.
    """

    def test_all_added_provenance_columns_present_and_round_trip(self, tmp_path):
        record = _full_record(
            belief_raw_completion="garbage text",
            belief_n_retries=2,
            belief_failed=True,
            model_id="qwen2.5-14b-instruct",
            chat_template_hash="abc123",
            prompt_schema_version=1,
            belief_protocol="schema_reasoning_belief",
        )
        path = str(tmp_path / "roundtrip.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)

        expected = {
            "belief_raw_completion": "garbage text",
            "belief_n_retries": 2,
            "belief_failed": True,
            "model_id": "qwen2.5-14b-instruct",
            "chat_template_hash": "abc123",
            # Asserted as a literal, not against
            # BELIEF_PROTOCOL_REASONING_BELIEF: this is the round-trip test,
            # and its job is to prove the exact bytes an analyst will read out
            # of a landed Parquet, which a rename must therefore break here
            # loudly rather than follow silently.
            "belief_protocol": "schema_reasoning_belief",
            "prompt_schema_version": 1,
        }
        for col, value in expected.items():
            assert col in df.columns, f"missing column {col!r}"
            assert (
                df[col][0] == value
            ), f"{col!r}: expected {value!r}, got {df[col][0]!r}"

    def test_skipped_belief_round_trips_as_null_not_failed(self, tmp_path):
        record = _full_record(
            belief_raw_completion=None,
            belief_n_retries=0,
            belief_failed=False,
        )
        path = str(tmp_path / "skip_roundtrip.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert df["belief_raw_completion"][0] is None
        assert df["belief_n_retries"][0] == 0
        assert df["belief_failed"][0] is False


class TestAttributionColumnsRoundTrip:
    """Four additive columns (lm_seat, actor, action_str,
    prompt_variant) so analysis can attribute a row to a seat and an agent
    without re-deriving them from run_meta.json config. Round-trips through
    the real MatchWriter, the same reason TestBeliefAndProvenanceColumnsRoundTrip
    above does: NodeRecord and PARQUET_SCHEMA are two representations of the
    same added columns that must independently agree.
    """

    def test_all_four_columns_present_with_correct_types_and_round_trip(self, tmp_path):
        record = _full_record(
            lm_seat=0,
            actor="bayesian",
            action_str="Bet",
            prompt_variant="base",
        )
        path = str(tmp_path / "attribution_roundtrip.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)

        expected = {
            "lm_seat": (0, pl.Int64),
            "actor": ("bayesian", pl.Utf8),
            "action_str": ("Bet", pl.Utf8),
            "prompt_variant": ("base", pl.Utf8),
        }
        for col, (value, dtype) in expected.items():
            assert col in df.columns, f"missing column {col!r}"
            assert (
                df.schema[col] == dtype
            ), f"{col!r}: expected dtype {dtype}, got {df.schema[col]}"
            assert (
                df[col][0] == value
            ), f"{col!r}: expected {value!r}, got {df[col][0]!r}"

    def test_actor_none_violates_required_columns(self):
        assert "actor" in REQUIRED_COLUMNS
        record = _full_record(actor=None)
        violations = [col for col in REQUIRED_COLUMNS if getattr(record, col) is None]
        assert violations == ["actor"]


class TestServedWeightsRevisionColumn:
    """`served_weights_revision`
    is a nullable Utf8 column carrying the served model's HF weights
    revision, resolved at launch (never by run_sweep itself) and threaded
    exactly like chat_template_hash. Additive: `model_revision` keeps its
    established harness-git-sha meaning untouched (schema.py's own comment).
    Hardcodes the column name and dtype: an iteration over PARQUET_SCHEMA could never notice the entry
    vanishing.
    """

    def test_served_weights_revision_utf8_and_round_trips(self, tmp_path):
        assert PARQUET_SCHEMA["served_weights_revision"] == pl.Utf8
        record = _full_record(
            served_weights_revision="04d8a90549d23fc6bd7f642064003592df51e9b3",
        )
        path = str(tmp_path / "swr_roundtrip.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert "served_weights_revision" in df.columns
        assert df.schema["served_weights_revision"] == pl.Utf8
        assert df["served_weights_revision"][0] == (
            "04d8a90549d23fc6bd7f642064003592df51e9b3"
        )
        # Additive, not a redefinition: the same row still carries the
        # harness git sha in model_revision, untouched.
        assert df["model_revision"][0] == "deadbeef"

    def test_served_weights_revision_nullable_and_defaulted(self, tmp_path):
        """The column is nullable END TO END and NodeRecord defaults it to
        None, so every pre-existing construction site (tests and non-LM
        paths) keeps working unchanged.
        """
        record = _full_record()  # does not pass served_weights_revision
        assert record.served_weights_revision is None
        path = str(tmp_path / "swr_null.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert df["served_weights_revision"].null_count() == len(df)


class TestTokenizerRevisionColumn:
    """`tokenizer_revision` is a
    nullable Utf8 column carrying the served endpoint's HF tokenizer
    revision, threaded through the identical hop points as
    `served_weights_revision` (MatchupSpec -> shard -> LMVllmAgent ->
    match_runner getattr -> here). Additive, same rule as
    TestServedWeightsRevisionColumn above.
    """

    def test_tokenizer_revision_utf8_and_round_trips(self, tmp_path):
        assert PARQUET_SCHEMA["tokenizer_revision"] == pl.Utf8
        record = _full_record(tokenizer_revision="tok-abc123")
        path = str(tmp_path / "tokrev_roundtrip.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert "tokenizer_revision" in df.columns
        assert df.schema["tokenizer_revision"] == pl.Utf8
        assert df["tokenizer_revision"][0] == "tok-abc123"
        # Additive, not a substitute: the same row still carries its own
        # served_weights_revision, untouched.
        assert df["served_weights_revision"][0] is None

    def test_tokenizer_revision_nullable_and_defaulted(self, tmp_path):
        """The column is nullable END TO END and NodeRecord defaults it to
        None, so every pre-existing construction site (tests and non-LM
        paths) keeps working unchanged.
        """
        record = _full_record()  # does not pass tokenizer_revision
        assert record.tokenizer_revision is None
        path = str(tmp_path / "tokrev_null.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert df["tokenizer_revision"].null_count() == len(df)


class TestSeedProvenanceColumns:
    """`agent_node_idx` and `run_seed` are nullable Int64 columns carrying
    the acting agent's per-decision ordinal and the run-level seed its
    request seeds were derived from, additive on the same rule as
    served_weights_revision (defaulted, so
    every pre-existing construction site keeps working). Hardcodes the column
    names and dtypes, same rationale as TestServedWeightsRevisionColumn
    above: an iteration over PARQUET_SCHEMA could never notice an entry
    vanishing. These are the analogue of that class's dtype/value
    round-trip anchor.
    """

    def test_agent_node_idx_and_run_seed_int64_and_round_trip(self, tmp_path):
        assert PARQUET_SCHEMA["agent_node_idx"] == pl.Int64
        assert PARQUET_SCHEMA["run_seed"] == pl.Int64
        # agent_node_idx=1 against the base record's node_idx=0: distinct
        # values, so an aliasing writer (stamping node_idx into the new
        # column) cannot pass the value assertion.
        record = _full_record(agent_node_idx=1, run_seed=999)
        path = str(tmp_path / "seedcols_roundtrip.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert "agent_node_idx" in df.columns
        assert "run_seed" in df.columns
        assert df.schema["agent_node_idx"] == pl.Int64
        assert df.schema["run_seed"] == pl.Int64
        assert df["agent_node_idx"][0] == 1
        assert df["run_seed"][0] == 999
        # Additive, not a redefinition: the same row still carries the
        # all-players ordinal and the match seed, untouched.
        assert df["node_idx"][0] == 0
        assert df["seed"][0] == 0

    def test_agent_node_idx_and_run_seed_nullable_and_defaulted(self, tmp_path):
        """The columns are nullable END TO END and NodeRecord defaults both
        to None, so every pre-existing construction site (tests and non-LM
        paths) keeps working unchanged, and old arms null-fill on read.
        """
        record = _full_record()  # passes neither agent_node_idx nor run_seed
        assert record.agent_node_idx is None
        assert record.run_seed is None
        path = str(tmp_path / "seedcols_null.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert df["agent_node_idx"].null_count() == len(df)
        assert df["run_seed"].null_count() == len(df)


class TestBeliefSalvageColumns:
    """`belief_salvage_kind` (Utf8) and `belief_salvage_attempt` (Int64) are
    nullable, appended columns: null means no salvage, a non-LM row, a skipped
    belief, or an arm collected before the columns existed. The names and
    dtypes are hardcoded for the reason `TestSeedProvenanceColumns` gives:
    iterating PARQUET_SCHEMA could never notice an entry vanishing.
    """

    def test_kind_and_attempt_round_trip_with_their_dtypes(self, tmp_path):
        assert PARQUET_SCHEMA["belief_salvage_kind"] == pl.Utf8
        assert PARQUET_SCHEMA["belief_salvage_attempt"] == pl.Int64
        record = _full_record(
            belief_salvage_kind="key_quoted", belief_salvage_attempt=2
        )
        path = str(tmp_path / "salvagecols_roundtrip.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert df.schema["belief_salvage_kind"] == pl.Utf8
        assert df.schema["belief_salvage_attempt"] == pl.Int64
        assert df["belief_salvage_kind"][0] == "key_quoted"
        assert df["belief_salvage_attempt"][0] == 2

    def test_both_default_to_null_and_stay_out_of_required_columns(self, tmp_path):
        record = _full_record()
        assert record.belief_salvage_kind is None
        assert record.belief_salvage_attempt is None
        assert "belief_salvage_kind" not in REQUIRED_COLUMNS
        assert "belief_salvage_attempt" not in REQUIRED_COLUMNS
        path = str(tmp_path / "salvagecols_null.parquet")
        writer = MatchWriter(path)
        writer.add(record)
        writer.flush()
        df = read_parquet(path)
        assert df["belief_salvage_kind"].null_count() == len(df)
        assert df["belief_salvage_attempt"].null_count() == len(df)


def test_off_support_column_present_and_non_null(parquet_df):
    assert "cfr_ref_off_support" in parquet_df.columns
    assert parquet_df["cfr_ref_off_support"].null_count() == 0


def test_cfr_action_never_negative_on_fully_covered_games(parquet_df):
    """The fixture is Kuhn with a full CFR+ reference: no off-support rows, so
    no sentinel should appear. A -1 here means the flag logic misfired."""
    assert all(a >= 0 for a in parquet_df["cfr_action"].to_list())
