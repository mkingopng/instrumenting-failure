"""Tests for the Leduc card-index <-> rank conversions shared by P5/P7a/P7b.

Gating the collected-arm regression tests on `data/`'s presence
(gitignored, so untracked) would let a skip read as green under
`pyproject.toml`'s `addopts = "-q"` (no `-rs`). Instead: the collected
arm's (elicited_belief, true_posterior) distribution collapses to only 142
distinct pairs across all 53,145 rows, so it is committed as a small fixture
(`tests/fixtures/leduc_belief_posterior_pairs.csv`, ~15KB) and the tests
below always run - no skip, no dependency on `data/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl
import pytest

from lmopt.scoring.leduc_rank import (
    expand_rank_to_indices,
    leduc_num_ranks,
    marginalise_to_rank,
)

FIXTURE = Path(__file__).parent / "fixtures" / "leduc_belief_posterior_pairs.csv"


def _load_landed_leduc_fixture() -> pl.DataFrame:
    """Reconstruct the 53,145-row collected-arm leduc (elicited_belief,
    true_posterior) distribution from its 142 distinct pairs, each repeated
    `count` times.

    Command used to derive the fixture (from
    data/run_experiment/qwen2.5-14b-instruct/leduc_lm_vs_*.parquet, the
    collected qwen2.5-14b arm):

        uv run python - <<'EOF'
        ... Counter((row["elicited_belief"], row["true_posterior"])
                     for row in rows.iter_rows(named=True)) ...
        -> 142 distinct pairs, summing to 53145
        EOF

    Floats are stored via `repr()`, which Python guarantees round-trips to
    the exact same float64 bit pattern, so re-parsing reproduces the
    original JSON's numeric content exactly (the JSON text itself need not
    match byte-for-byte - only the parsed floats matter to the scorers).
    """
    rows = []
    with open(FIXTURE, newline="") as f:
        for r in csv.DictReader(f):
            eb = json.dumps(
                {"0": float(r["eb0"]), "1": float(r["eb1"]), "2": float(r["eb2"])}
            )
            tp = json.dumps({str(i): float(r[f"tp{i}"]) for i in range(6)})
            count = int(r["count"])
            rows.extend(
                [
                    {
                        "game": "leduc",
                        "elicited_belief": eb,
                        "true_posterior": tp,
                        "prompt": "p",
                    }
                ]
                * count
            )
    return pl.DataFrame(rows)


def test_leduc_num_ranks_for_canonical_leduc():
    pytest.importorskip("pyspiel")
    assert leduc_num_ranks("leduc") == 3


def test_marginalise_defaults_to_three_ranks_byte_identical_to_pre_param():
    post = {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.0, "4": 0.2, "5": 0.2}
    assert marginalise_to_rank(post) == marginalise_to_rank(post, n_ranks=3)


def test_marginalise_at_a_non_canonical_rank_count():
    """leduc_deep-shaped: 5 ranks x 2 suits = 10 card indices, collapsing to
    5 rank keys. Mass for ranks the canonical 3-key call would silently
    drop (index 6-9, ranks 3-4) must survive and land in the right bucket."""
    post = {str(i): 0.1 for i in range(10)}  # uniform over 10 card indices
    result = marginalise_to_rank(post, n_ranks=5)
    assert result == {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}
    assert abs(sum(result.values()) - 1.0) < 1e-12


def test_expand_defaults_to_three_ranks_byte_identical_to_pre_param():
    rank_belief = {"0": 0.4, "1": 0.2, "2": 0.4}
    assert expand_rank_to_indices(rank_belief) == expand_rank_to_indices(
        rank_belief, n_ranks=3
    )


def test_expand_at_a_non_canonical_rank_count():
    """leduc_deep-shaped: a 5-rank belief expands to 10 card indices, not 6 --
    a hardcoded range(N_RANKS * CARDS_PER_RANK) would silently
    produce only 6 keys here and drop ranks 3-4 entirely."""
    rank_belief = {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}
    expanded = expand_rank_to_indices(rank_belief, n_ranks=5)
    assert set(expanded.keys()) == {str(i) for i in range(10)}
    assert expanded["8"] == pytest.approx(0.1)  # rank 4 -> indices 8,9
    assert expanded["9"] == pytest.approx(0.1)
    assert sum(expanded.values()) == pytest.approx(1.0)


def test_marginalise_collapses_six_indices_to_three_ranks():
    # rank = idx // 2, so {0,1}=Jack, {2,3}=Queen, {4,5}=King
    post = {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.0, "4": 0.2, "5": 0.2}
    assert marginalise_to_rank(post) == {"0": 0.4, "1": 0.2, "2": 0.4}


def test_marginalise_preserves_total_mass():
    """Iterating only the 3 stated rank keys against a 6-key index
    posterior discards 55.48% of posterior mass per row."""
    post = {"0": 0.1, "1": 0.15, "2": 0.25, "3": 0.05, "4": 0.3, "5": 0.15}
    assert abs(sum(marginalise_to_rank(post).values()) - 1.0) < 1e-12


class TestExpandRankToIndices:
    """P7b needs the reverse direction of marginalise_to_rank: a rank
    belief expanded back to the 6 card indices. An unconditional 50/50
    split is wrong whenever one copy of a rank is a card the opponent
    provably cannot hold (the acting player's own card, or the revealed
    round-2 board card) - it crashes `_br_to_belief`'s counterfactual with
    an uncaught OpenSpiel SpielError.
    These are the pure, state-free cases; tests/test_scorer_property7.py's
    `TestParseBeliefArrayLeducExpansion` covers the same logic against real
    Leduc game states.
    """

    def test_even_split_when_nothing_excluded(self):
        rank_belief = {"0": 0.4, "1": 0.2, "2": 0.4}
        expanded = expand_rank_to_indices(rank_belief)
        assert expanded == {
            "0": 0.2,
            "1": 0.2,  # Jack
            "2": 0.1,
            "3": 0.1,  # Queen
            "4": 0.2,
            "5": 0.2,  # King
        }
        assert abs(sum(expanded.values()) - 1.0) < 1e-12

    def test_excluded_copy_gets_zero_and_the_other_copy_absorbs_the_mass(self):
        """Rank 0's mass must not split evenly when index 0 is excluded -
        it must go entirely to index 1, the only live copy."""
        rank_belief = {"0": 0.4, "1": 0.2, "2": 0.4}
        expanded = expand_rank_to_indices(rank_belief, excluded=frozenset({0}))
        assert expanded["0"] == pytest.approx(0.0)
        assert expanded["1"] == pytest.approx(0.4)
        assert expanded["2"] == pytest.approx(0.1)
        assert expanded["3"] == pytest.approx(0.1)
        assert expanded["4"] == pytest.approx(0.2)
        assert expanded["5"] == pytest.approx(0.2)
        assert sum(expanded.values()) == pytest.approx(1.0)

    def test_both_copies_excluded_drops_and_renormalises(self):
        """Only possible when the acting player's own card and the public
        board card share a rank. That rank's stated mass is unassignable
        and must be dropped; the remainder renormalises to sum to 1.0."""
        rank_belief = {"0": 0.5, "1": 0.3, "2": 0.2}
        expanded = expand_rank_to_indices(rank_belief, excluded=frozenset({0, 1}))
        assert expanded["0"] == pytest.approx(0.0)
        assert expanded["1"] == pytest.approx(0.0)
        # 0.3 + 0.2 = 0.5 of real mass remains; renormalised by /0.5.
        assert expanded["2"] == pytest.approx(0.3)
        assert expanded["3"] == pytest.approx(0.3)
        assert expanded["4"] == pytest.approx(0.2)
        assert expanded["5"] == pytest.approx(0.2)
        assert sum(expanded.values()) == pytest.approx(1.0)

    def test_all_mass_on_an_excluded_rank_sums_to_zero(self):
        """If the model's entire stated belief is on a rank with both
        copies excluded, there is nothing to renormalise - every value is
        0.0. Callers (`_parse_belief_array`) must treat sum==0 as
        uncomputable, not as a degenerate all-zero belief."""
        rank_belief = {"0": 1.0, "1": 0.0, "2": 0.0}
        expanded = expand_rank_to_indices(rank_belief, excluded=frozenset({0, 1}))
        assert sum(expanded.values()) == pytest.approx(0.0)


def test_rank_invariant_against_openspiel_showdown_returns():
    """Executable guard for `rank = card_index // 2`, so a future OpenSpiel
    deck-ordering change fails this test rather than silently invalidating
    every rank-space computation in leduc_rank.py. Checks the showdown
    facts directly: card0/card1 tie (same rank), card0 loses to
    card2 (different, higher rank), card2/card3 tie (same rank) - i.e. the
    showdown outcome depends only on floor(index / 2), never on which copy
    was dealt.
    """
    pytest.importorskip("pyspiel")
    from lmopt.games.loaders import load_leduc

    def _showdown(p0_card: int, p1_card: int) -> int:
        """+1 if P0 wins at an uncontested showdown (both check both rounds),
        0 for a tie, -1 if P1 wins. Picks a board card whose rank differs
        from both hole cards' ranks -- an incidental board pair would let a
        lower-rank hole card win outright, confounding the pure high-card
        comparison this guard checks (dealing whichever board card
        OpenSpiel iterates to first pairs card 0 in card 2's showdown and
        flips the expected loser into a winner)."""
        game = load_leduc()
        s = game.new_initial_state()
        s.apply_action(p0_card)
        s.apply_action(p1_card)
        while not s.is_terminal():
            if s.is_chance_node():
                outcomes = [a for a, _ in s.chance_outcomes()]
                board = next(
                    a
                    for a in outcomes
                    if a // 2 != p0_card // 2 and a // 2 != p1_card // 2
                )
                s.apply_action(board)
            else:
                s.apply_action(1)  # Call/Check: never fold or raise.
        returns = s.returns()
        if returns[0] > returns[1]:
            return 1
        if returns[0] < returns[1]:
            return -1
        return 0

    # card0/card1 tie: same rank (0 // 2 == 1 // 2 == 0).
    assert _showdown(0, 1) == 0
    # card0 loses to card2: different, higher rank (0 // 2 < 2 // 2).
    assert _showdown(0, 2) == -1
    # card2/card3 tie: same rank (2 // 2 == 3 // 2 == 1).
    assert _showdown(2, 3) == 0


def test_p7a_leduc_tvd_uses_rank_marginalised_posterior():
    """Pins the rank-marginalised value against the committed collected-arm
    fixture.

    Both figures derive directly from the real parquet shards:

        uv run python - <<'EOF'
        ... score_property7(df) on data/run_experiment/qwen2.5-14b-instruct ...
        EOF
        -> tvd_mean['leduc'] = 0.5858406986602719  (raw key-union)
        -> tvd_mean['leduc'] = 0.35416033133153496 (rank-marginalised)

    The committed 142-pair fixture reproduces 0.3541603313315349 (last
    digit differs only from floating-point summation-order, well within
    rel=1e-9) when loaded and passed to the real `score_property7`.
    """
    pytest.importorskip("pyspiel")
    from lmopt.scoring.property7 import score_property7

    df = _load_landed_leduc_fixture()
    assert df.height == 53145

    result = score_property7(df)
    tvd = result["tvd_mean"]["leduc"]
    assert tvd == pytest.approx(0.35416033133153496, rel=1e-9)
    assert tvd != pytest.approx(0.5858406986602719, abs=1e-3)
    assert result["n_belief_rows"]["leduc"] == 53145
