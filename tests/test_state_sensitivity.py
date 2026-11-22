"""Tests for the state-sensitivity decomposition."""

import polars as pl


def _frame(rows):
    """One cell's LM rows. `rows` is (prompt, info_set, mask, action)."""
    return pl.DataFrame(
        [
            {"prompt": p, "info_set": i, "legal_mask": m, "parsed_action": a}
            for p, i, m, a in rows
        ],
        schema={
            "prompt": pl.String,
            "info_set": pl.String,
            "legal_mask": pl.List(pl.Int64),
            "parsed_action": pl.Int64,
        },
    )


def test_bin_stats_marks_a_single_prompt_bin_as_not_multistate():
    """    A bin holding one distinct prompt supplied no state variation, so a
    constant action there is a determinism observation, not a state-blindness
    one. Over the landed corpus, at the pooled
    (arm, game, mask) unit: 8 one-prompt bins totalling 122,213 rows, of which
    6 hold exactly 20,000 rows, and one of those 6 (OLMo darkhex) is not
    constant. Counting only the constant ones would be selection on the
    dependent variable, the defect this whole diagnostic exists to correct.
    """
    from lmopt.analysis.state_sensitivity import bin_stats

    df = _frame([("P", "a", [1, 1], 0), ("P", "a", [1, 1], 0)])
    (b,) = bin_stats(df)
    assert b.n_prompt == 1
    assert b.multistate is False
    assert b.constant is True


def test_bin_stats_marks_a_two_prompt_bin_as_multistate():
    from lmopt.analysis.state_sensitivity import bin_stats

    df = _frame([("P", "a", [1, 1], 0), ("Q", "b", [1, 1], 0)])
    (b,) = bin_stats(df)
    assert (b.n_prompt, b.n_infoset, b.multistate, b.constant) == (2, 2, True, True)


def test_bin_stats_keys_multistate_on_prompts_not_infosets():
    """    Two info-sets that render to one prompt (leduc's suit marginalisation)
    are indistinguishable to the model, so the bin is not multi-state.
    """
    from lmopt.analysis.state_sensitivity import bin_stats

    df = _frame([("P", "a", [1, 1], 0), ("P", "b", [1, 1], 0)])
    (b,) = bin_stats(df)
    assert (b.n_prompt, b.n_infoset) == (1, 2)
    assert b.multistate is False


def test_bin_stats_separates_bins_by_mask():
    from lmopt.analysis.state_sensitivity import bin_stats

    df = _frame(
        [
            ("P", "a", [1, 1], 0),
            ("Q", "b", [1, 1], 0),
            ("R", "c", [1, 1, 1], 2),
            ("S", "d", [1, 1, 1], 1),
        ]
    )
    bins = {b.mask: b for b in bin_stats(df)}
    assert set(bins) == {"1,1", "1,1,1"}
    assert bins["1,1"].constant is True
    assert bins["1,1,1"].constant is False


def test_multistate_share_ignores_single_state_bins():
    """    The single-prompt bin below is constant, so including it would raise the
    share from 0.0 to 0.5.
    """
    from lmopt.analysis.state_sensitivity import multistate_share

    df = _frame(
        [
            ("P", "a", [1, 1], 0),
            ("P", "a", [1, 1], 0),
            ("R", "c", [1, 1, 1], 2),
            ("S", "d", [1, 1, 1], 1),
        ]
    )
    share, denom = multistate_share(df)
    assert share == 0.0
    assert denom == 2


def test_multistate_share_returns_none_not_zero_with_no_multistate_bin():
    """    0.0 is this statistic's CLEANEST reading ("no multi-state bin was
    constant"), so returning it for a cell with nothing to measure asserts
    perfect state-sensitivity from no evidence (the same defect class guarded
    against in `scripts/analyse_stats.py`).
    """
    from lmopt.analysis.state_sensitivity import multistate_share

    df = _frame([("P", "a", [1, 1], 0), ("P", "a", [1, 1], 0)])
    assert multistate_share(df) == (None, 0)


def test_multistate_share_excludes_opponent_rows():
    """    An opponent's action under the same mask is not evidence about the LM's
    policy; letting it in can only add spurious variation. Same rule
    `mask_constant_share` applies (`analyse_stats.py`).
    """
    from lmopt.analysis.state_sensitivity import multistate_share

    df = _frame([("P", "a", [1, 1], 0), ("Q", "b", [1, 1], 0)])
    df = pl.concat(
        [
            df,
            _frame([("X", "c", [1, 1], 1)]).with_columns(
                pl.lit(None, dtype=pl.String).alias("prompt")
            ),
        ]
    )
    share, denom = multistate_share(df)
    assert (share, denom) == (1.0, 2)


def _cell_frame(rows):
    """One cell's LM rows plus `cfr_policy`. `rows` is (prompt, info_set,
    mask, action, cfr_policy_json) -- `_frame` above with one more column,
    the JSON-string reference distribution `rows_for_cell` reads `p_lm` from.
    """
    return pl.DataFrame(
        [
            {
                "prompt": p,
                "info_set": i,
                "legal_mask": m,
                "parsed_action": a,
                "cfr_policy": c,
            }
            for p, i, m, a, c in rows
        ],
        schema={
            "prompt": pl.String,
            "info_set": pl.String,
            "legal_mask": pl.List(pl.Int64),
            "parsed_action": pl.Int64,
            "cfr_policy": pl.String,
        },
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _write_shard(tmp_path, name, rows):
    """Write one `<game>_lm_vs_<opponent>_seed<N>.parquet` shard.

    `rows` is `(prompt, info_set, mask, action, cfr_policy_json)`, the same
    shape `_cell_frame` builds, but on disk so `run_pass`'s own glob/`cell_of`
    parsing is exercised (not bypassed by calling `rows_for_cell` directly).
    """
    df = pl.DataFrame(
        [
            {
                "prompt": p,
                "info_set": i,
                "legal_mask": m,
                "parsed_action": a,
                "cfr_policy": c,
            }
            for p, i, m, a, c in rows
        ],
        schema={
            "prompt": pl.String,
            "info_set": pl.String,
            "legal_mask": pl.List(pl.Int64),
            "parsed_action": pl.Int64,
            "cfr_policy": pl.String,
        },
    )
    df.write_parquet(tmp_path / name)


# ---------------------------------------------------------------------------
# The within-seat multistate share.
# The pooled share counts a seat rendering as state variation, so under random
# seats a state-blind model can read as multi-state on prompts that differ only
# by which seat it occupies. The within-seat measure is the one that carries
# state-blindness claims; the pooled one is a census.
# ---------------------------------------------------------------------------


def _seated_frame(rows):
    """One cell's LM rows with `lm_seat`. `rows` is
    (prompt, info_set, mask, action, seat); `seat` may be `None`.
    """
    return pl.DataFrame(
        [
            {
                "prompt": p,
                "info_set": i,
                "legal_mask": m,
                "parsed_action": a,
                "lm_seat": s,
            }
            for p, i, m, a, s in rows
        ],
        schema={
            "prompt": pl.String,
            "info_set": pl.String,
            "legal_mask": pl.List(pl.Int64),
            "parsed_action": pl.Int64,
            "lm_seat": pl.Int64,
        },
    )


def test_bin_stats_reports_a_seat_only_bin_as_pooled_multistate_only():
    """    The two prompts below differ only in which seat the model occupies, one
    row per seat. Pooled, the bin holds two distinct prompts and is
    multi-state; within either seat it holds exactly one, so the model was
    never given a chance to respond to a state change while its seat was
    held fixed.
    """
    from lmopt.analysis.state_sensitivity import bin_stats

    df = _seated_frame(
        [
            ("You are player 0. J", "a", [1, 1], 0, 0),
            ("You are player 1. J", "a", [1, 1], 0, 1),
        ]
    )
    (b,) = bin_stats(df)
    assert (b.n_prompt, b.multistate) == (2, True)
    assert b.n_prompt_within_seat_max == 1
    assert b.multistate_within_seat is False


def test_multistate_share_within_seat_is_none_when_only_the_seat_varies():
    """    Same seat-only fixture as above, whose pooled share is a clean 1.0. The
    within-seat measure has nothing to divide by, and 0.0 would assert
    perfect within-seat state-sensitivity from no evidence (no silent
    fallbacks); the pooled 1.0 would re-import the very confound.
    """
    from lmopt.analysis.state_sensitivity import (
        multistate_share,
        multistate_share_within_seat,
        within_seat_reason,
    )

    df = _seated_frame(
        [
            ("You are player 0. J", "a", [1, 1], 0, 0),
            ("You are player 1. J", "a", [1, 1], 0, 1),
        ]
    )
    assert multistate_share(df) == (1.0, 2)
    assert multistate_share_within_seat(df) == (None, 0)
    reason = within_seat_reason(df)
    assert reason and "within that seat" in reason, reason


def test_multistate_share_within_seat_equals_the_pooled_share_on_a_fixed_seat():
    """    With the seat held fixed there is no seat confound to remove, so the two
    measures must agree; a within-seat share that differed here would be
    measuring something other than what its name says. `within_seat_reason`
    is `None` exactly when the share is measurable.
    """
    from lmopt.analysis.state_sensitivity import (
        multistate_share,
        multistate_share_within_seat,
        within_seat_reason,
    )

    df = _seated_frame(
        [
            ("P", "a", [1, 1], 0, 0),
            ("Q", "b", [1, 1], 0, 0),
            ("R", "c", [1, 1, 1], 2, 0),
            ("R", "c", [1, 1, 1], 2, 0),
        ]
    )
    assert multistate_share(df) == (1.0, 2)
    assert multistate_share_within_seat(df) == (1.0, 2)
    assert within_seat_reason(df) is None


def test_multistate_share_within_seat_refuses_a_frame_with_no_lm_seat_column():
    """    `lm_seat` is real only on schema-v3+ arms; older arms carry no such
    column at all. The pooled measure must be untouched by the new one.
    """
    from lmopt.analysis.state_sensitivity import (
        bin_stats,
        multistate_share,
        multistate_share_within_seat,
        within_seat_reason,
    )

    df = _frame([("P", "a", [1, 1], 0), ("Q", "b", [1, 1], 0)])
    assert "lm_seat" not in df.columns
    assert multistate_share(df) == (1.0, 2)
    assert multistate_share_within_seat(df) == (None, 0)
    reason = within_seat_reason(df)
    assert reason and "lm_seat" in reason, reason
    (b,) = bin_stats(df)
    assert b.n_prompt_within_seat_max is None
    assert b.multistate_within_seat is None


def test_rows_with_a_null_seat_count_toward_no_seat():
    """    A row that does not say which seat it was played from is evidence about
    no seat. The column is present here (schema-v3+ shape) but unfilled, so
    the refusal names the nulls rather than the missing column.
    """
    from lmopt.analysis.state_sensitivity import (
        bin_stats,
        multistate_share,
        multistate_share_within_seat,
        within_seat_reason,
    )

    df = _seated_frame([("P", "a", [1, 1], 0, None), ("Q", "b", [1, 1], 0, None)])
    assert multistate_share(df) == (1.0, 2)
    (b,) = bin_stats(df)
    assert b.n_prompt_within_seat_max == 0
    assert b.multistate_within_seat is False
    assert multistate_share_within_seat(df) == (None, 0)
    reason = within_seat_reason(df)
    assert reason and "null" in reason, reason


def test_a_null_seat_row_does_not_join_the_seat_it_sits_beside():
    """    Seat 0 holds one prompt; the null-seat row holds the other. If the null
    row were folded into seat 0 the bin would read within-seat multi-state
    (max 2) off one row whose seat is unknown.
    """
    from lmopt.analysis.state_sensitivity import bin_stats

    df = _seated_frame([("P", "a", [1, 1], 0, 0), ("Q", "b", [1, 1], 0, None)])
    (b,) = bin_stats(df)
    assert (b.n_prompt, b.multistate) == (2, True)
    assert b.n_prompt_within_seat_max == 1
    assert b.multistate_within_seat is False


# ---------------------------------------------------------------------------
# The within-seat statistic is computed
# at the (bin, seat) unit, constancy included. Selecting pairs by within-seat
# multi-stateness and then reading the pooled `constant` off the bin counts a
# seat-driven action switch as state-sensitivity, which is the confound the
# measure exists to remove.
# ---------------------------------------------------------------------------


def test_within_seat_share_judges_constancy_inside_the_seat_not_across_seats():
    """    One mask, two seats, each holding two distinct prompts, so both (bin,
    seat) pairs are within-seat multi-state. Seat 0 plays two different
    actions; seat 1 plays one action across both its prompts. Seat 1 is
    state-blind and its 2 rows must count as constant, giving 2/4 = 0.5.
    Pooled, the single bin holds two distinct actions and is non-constant, so
    the mutation scores every row state-sensitive and reads 0.0.

    This is not hypothetical: on shakedown-v3 olmo
    goofspiel/rl, bin `1,0,0,1` is pooled non-constant over 200 rows while
    seat 1 alone plays action 3 in all 94 of its rows across 4 prompts. The
    cell reads 0.735 under per-seat constancy against 0.5 under pooled.
    """
    from lmopt.analysis.state_sensitivity import (
        multistate_share,
        multistate_share_within_seat,
    )

    df = _seated_frame(
        [
            ("P0a", "a", [1, 1], 0, 0),
            ("P0b", "b", [1, 1], 1, 0),
            ("P1a", "c", [1, 1], 0, 1),
            ("P1b", "d", [1, 1], 0, 1),
        ]
    )
    assert multistate_share(df) == (0.0, 4)
    assert multistate_share_within_seat(df) == (0.5, 4)


def test_seat_bin_stats_reports_one_row_per_bin_and_seat():
    """    The per-pair fields are what `state_sensitivity` divides by, so each must
    be judged inside its own seat: seat 0 non-constant, seat 1 constant, both
    multi-state, two rows each.
    """
    from lmopt.analysis.state_sensitivity import seat_bin_stats

    df = _seated_frame(
        [
            ("P0a", "a", [1, 1], 0, 0),
            ("P0b", "b", [1, 1], 1, 0),
            ("P1a", "c", [1, 1], 0, 1),
            ("P1b", "d", [1, 1], 0, 1),
        ]
    )
    pairs = {sb.seat: sb for sb in seat_bin_stats(df)}
    assert set(pairs) == {0, 1}
    assert all(
        sb.mask == "1,1" and sb.rows == 2 and sb.multistate for sb in pairs.values()
    )
    assert (pairs[0].n_action, pairs[0].constant) == (2, False)
    assert (pairs[1].n_action, pairs[1].constant) == (1, True)


def test_seat_bin_stats_is_empty_without_a_seat_column():
    from lmopt.analysis.state_sensitivity import seat_bin_stats

    assert seat_bin_stats(_frame([("P", "a", [1, 1], 0)])) == []


def test_state_sensitivity_counts_bins_pooled_and_pairs_within_seat():
    """    Two masks. Mask `1,1` holds two prompts in each of two seats, so it is
    one pooled bin and two within-seat pairs. Mask `1,1,1` holds one prompt
    per seat, so it is a pooled bin that is not multi-state and contributes
    no pair. `bins_total` counts every bin, multi-state or not.
    """
    from lmopt.analysis.state_sensitivity import state_sensitivity

    df = _seated_frame(
        [
            ("P0a", "a", [1, 1], 0, 0),
            ("P0b", "b", [1, 1], 0, 0),
            ("P1a", "c", [1, 1], 0, 1),
            ("P1b", "d", [1, 1], 0, 1),
            ("Q", "e", [1, 1, 1], 0, 0),
            ("Q", "f", [1, 1, 1], 0, 1),
        ]
    )
    st = state_sensitivity(df)
    assert (st.bins_total, st.bins_multistate_pooled) == (2, 1)
    assert st.rows_multistate_pooled == 4
    assert st.bins_multistate_within_seat == 2
    assert st.rows_multistate_within_seat == 4
    assert st.share_within_seat == 1.0
    assert st.within_seat_reason is None


def test_state_sensitivity_within_seat_counts_are_none_only_without_the_column():
    """    Same asymmetry the pooled rows/share pair carries: a count over an empty set is a measured `0`, a ratio over
    one is undefined. The seat column's absence is the only thing that makes
    the counts unmeasurable.
    """
    from lmopt.analysis.state_sensitivity import state_sensitivity

    unseated = state_sensitivity(_frame([("P", "a", [1, 1], 0), ("Q", "b", [1, 1], 0)]))
    assert unseated.bins_multistate_within_seat is None
    assert unseated.rows_multistate_within_seat is None
    assert unseated.share_pooled == 1.0 and unseated.rows_multistate_pooled == 2

    seat_only = state_sensitivity(
        _seated_frame(
            [
                ("You are player 0. J", "a", [1, 1], 0, 0),
                ("You are player 1. J", "a", [1, 1], 0, 1),
            ]
        )
    )
    assert seat_only.bins_multistate_within_seat == 0
    assert seat_only.rows_multistate_within_seat == 0
    assert seat_only.share_within_seat is None
    assert seat_only.within_seat_reason is not None


def test_state_sensitivity_share_pooled_equals_the_published_multistate_share():
    """The two representations of the same
    pooled statistic must agree: `state_sensitivity(df).share_pooled` and
    `multistate_share(df)[0]`, the function the property table publishes.
    Pinning only the rows count would let the one-pass computation drift
    from the published share with the suite green.

    Enumerated over four shapes rather than one, so the agreement cannot rest
    on a fixture where every arithmetic path coincides: all-constant,
    all-varying, mixed, and a frame with no multi-state bin at all (both
    `None`, and `None == None` is asserted explicitly rather than through
    `==` on floats).
    """
    from lmopt.analysis.state_sensitivity import (
        multistate_share,
        state_sensitivity,
    )

    frames = {
        "all_constant": _frame([("P", "a", [1, 1], 0), ("Q", "a", [1, 1], 0)]),
        "all_varying": _frame([("P", "a", [1, 1], 0), ("Q", "a", [1, 1], 1)]),
        "mixed": _frame(
            [
                ("P", "a", [1, 1], 0),
                ("Q", "a", [1, 1], 0),
                ("R", "b", [1, 1, 1], 0),
                ("S", "b", [1, 1, 1], 1),
                ("T", "b", [1, 1, 1], 1),
            ]
        ),
        "no_multistate_bin": _frame([("P", "a", [1, 1], 0)]),
    }
    for name, df in frames.items():
        st = state_sensitivity(df)
        share, rows = multistate_share(df)
        assert st.rows_multistate_pooled == rows, name
        if share is None:
            assert st.share_pooled is None, name
        else:
            assert st.share_pooled == share, (name, st.share_pooled, share)
    # Anti-vacuity: the four shapes really do span the answer space, so the
    # equality above is not four copies of `None == None`.
    assert state_sensitivity(frames["all_constant"]).share_pooled == 1.0
    assert state_sensitivity(frames["all_varying"]).share_pooled == 0.0
    mixed = state_sensitivity(frames["mixed"]).share_pooled
    assert mixed is not None and 0.0 < mixed < 1.0, mixed
    assert state_sensitivity(frames["no_multistate_bin"]).share_pooled is None


def test_qualifying_masks_action_constant_excludes_cross_seat_only_mask():
    """A mask qualifying only by pooling two
    seats' single-prompt renderings must be absent from the returned dict."""
    from lmopt.analysis.state_sensitivity import qualifying_masks_action_constant

    df = _seated_frame(
        [
            ("You are player 0. J", "i0", [1, 1], 0, 0),
            ("You are player 1. J", "i1", [1, 1], 0, 1),
        ]
    )
    assert qualifying_masks_action_constant(df) == {}


def test_qualifying_masks_action_constant_true_on_genuine_within_seat_bin():
    """A real within-seat multistate, constant bin is present and `True`."""
    from lmopt.analysis.state_sensitivity import qualifying_masks_action_constant

    df = _seated_frame(
        [
            ("P", "i0", [1, 1], 0, 0),
            ("Q", "i1", [1, 1], 0, 0),
            ("R", "i2", [1, 1], 1, 1),
        ]
    )
    assert qualifying_masks_action_constant(df) == {"1,1": True}


def test_qualifying_masks_action_constant_false_when_qualifying_seat_varies_action():
    """A within-seat qualifying bin whose action varies within that seat is
    `False` (not action-constant)."""
    from lmopt.analysis.state_sensitivity import qualifying_masks_action_constant

    df = _seated_frame(
        [
            ("P", "i0", [1, 1], 0, 0),
            ("Q", "i1", [1, 1], 1, 0),
        ]
    )
    assert qualifying_masks_action_constant(df) == {"1,1": False}


def test_qualifying_masks_action_constant_conservative_any_across_seats():
    """A mask with two qualifying seats, one constant and one not, reads
    `True` (conservative: gate on any qualifying+constant evidence)."""
    from lmopt.analysis.state_sensitivity import qualifying_masks_action_constant

    df = _seated_frame(
        [
            ("P0", "i0", [1, 1], 0, 0),
            ("Q0", "i1", [1, 1], 0, 0),  # seat 0: constant
            ("P1", "i2", [1, 1], 0, 1),
            ("Q1", "i3", [1, 1], 1, 1),  # seat 1: varies
        ]
    )
    assert qualifying_masks_action_constant(df) == {"1,1": True}
