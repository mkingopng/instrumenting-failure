"""Leduc card-index <-> rank conversions, shared by P5, P7a, and P7b.

Leduc deals six card indices (two of each rank: {0,1}=Jack, {2,3}=Queen,
{4,5}=King), but the LM's elicited belief is a 3-simplex over ranks - a card's
identity beyond its rank never affects betting or showdown. `true_posterior`
is always logged in the 6-index space it is computed in. Comparing a 3-key
belief against a 6-key posterior directly is a category error, not a support
mismatch to paper over:

- P5 (`property5.py`): iterating only the 3 stated keys against the 6-key
  posterior silently discards 55.48% of posterior mass per row
  (measured on the landed arm, `data/run_experiment/qwen2.5-14b-instruct`).
- P7a (`property7.py`): unioning the two key sets inflates TVD because
  the union treats "index 3 has no stated mass" as a real disagreement
  instead of the stated belief simply not being expressed at that
  granularity - measured 0.5858 union-scored vs. 0.3542 rank-marginalised on
  the same arm (65% overstatement).
- P7b's `_parse_belief_array`: writing `arr[int(k)] = v` for k in
  {"0","1","2"} puts all the rank mass on card indices 0-2 and leaves
  indices 3-5 (half the deck) at zero, so the best-response counterfactual
  never considers half of the opponent's possible hands. An even 50/50
  split per rank is also wrong: it can put mass on the acting player's own
  card or the revealed board card - cards the opponent provably cannot
  hold - which crashes `_br_to_belief`'s counterfactual with an uncaught
  OpenSpiel `SpielError` ("deck_[move] != kInvalidCard") rather than merely
  mis-scoring. Measured: an even split puts mass on the acting player's own
  card on 40,317 of 53,145 rows (75.9%). `expand_rank_to_indices` is
  therefore availability-aware: it takes the set of card indices the
  opponent cannot hold and only splits a rank's mass across whichever of
  its two copies remain live.

`rank = card_index // 2`, verified against OpenSpiel Leduc showdown returns
(card0/card1 tie, card0 loses to card2, card2/card3 tie - i.e. showdown
outcome depends only on floor(index / 2), never on which copy was dealt).
"""

from __future__ import annotations

N_RANKS = 3
CARDS_PER_RANK = 2


def leduc_num_ranks(game_key: str) -> int:
    """The real `numRanks` for a `family == "leduc"` game registry key.

    `property5.py`/`property7.py`'s `family ==
    "leduc"` branches need the actual rank count per row's `game` column
    value, not the canonical 3 - `"leduc"` itself (native `leduc_poker`) has
    no `numRanks` parameter at all, `"leduc_deep"` (`universal_poker`) does.
    `.get("numRanks", N_RANKS)` handles both: the canonical game's live
    `pyspiel.Game.get_parameters()` genuinely lacks the key (confirmed:
    `leduc_poker` only exposes players/starting_player/
    action_mapping/suit_isomorphism), so it falls through to `N_RANKS`. Derived
    from the live game object rather than a hand-maintained
    `{"leduc": 3, "leduc_deep": 13}`-shaped table, so a future leduc-family
    depth variant needs no edit here at all, only its own `GAMES` entry.

    Constructing a `pyspiel.Game` per distinct game-column value (never
    per-row - callers do this once per `game_name` in their outer loop) is
    already the established pattern in this scoring layer, e.g.
    `darkhex_posterior.py`'s `darkhex_board_dims` and `property6.py`'s
    `load_game` uses.
    """
    from lmopt.games.loaders import GAMES

    game = GAMES[game_key]()
    return game.get_parameters().get("numRanks", N_RANKS)


def marginalise_to_rank(
    posterior: dict[str, float], n_ranks: int = N_RANKS
) -> dict[str, float]:
    """Collapse a card-index posterior to an `n_ranks`-key rank posterior.

    `t_rank[k] = t[2k] + t[2k+1]`. Total mass is preserved (the two rank
    keys partition the index keys), so this is a marginalisation, not a
    truncation.

    `n_ranks`: defaults to the canonical 3. `leduc_deep`'s frontier at a
    different rank count is the reason this is a parameter rather than a
    module constant - `CARDS_PER_RANK` (2) is not parametrized here, because leduc_deep keeps
    `numSuits=2` fixed and only varies `numRanks`; a future variant that
    also varied suit count would need this generalised further.
    """
    out = {str(k): 0.0 for k in range(n_ranks)}
    for key, mass in posterior.items():
        out[str(int(key) // CARDS_PER_RANK)] += float(mass)
    return out


def expand_rank_to_indices(
    rank_belief: dict[str, float],
    excluded: frozenset[int] = frozenset(),
    n_ranks: int = N_RANKS,
) -> dict[str, float]:
    """Inverse direction, for P7b: expand a 3-key rank belief to the 6 card
    indices (idx = 2*rank, 2*rank + 1), availability-aware.

    `excluded` names card indices the opponent provably cannot hold: the
    acting player's own dealt card, and - once round 2 has begun - the
    revealed public board card (`property7.py`'s `_leduc_excluded_indices`
    derives this set from the game state). An even 50/50 split across a
    rank's two copies is wrong whenever one copy is excluded: it hands the opponent a nonzero
    chance of holding a card that is provably in the acting player's hand or
    face-up on the board. `_br_to_belief`'s counterfactual then tries to
    deal that exact card index a second time, which is not a numerical
    distortion but an uncaught `SpielError` from OpenSpiel
    ("deck_[move] != kInvalidCard") - reproduced end-to-end via
    `score_property7` on a one-row fixture with `my_card == opp_card == 1`.
    Measured on the landed arm: an even split puts mass on the
    acting player's own card on 40,317 of 53,145 rows (75.9%).

    For each rank, mass goes entirely to whichever of its two copies is
    still live; it splits 50/50 only when both are live. If both copies of
    a rank are excluded (only possible when the acting player's own card
    and the public board card are the same rank), any stated mass on that
    rank is a belief about a card that cannot exist, so it is dropped and
    the remaining mass is renormalised. If dropping impossible mass leaves
    nothing - the model put all of its stated mass on a now-impossible rank
    - every returned value is 0.0 (mass sums to 0); callers must treat that
    as uncomputable (see `_parse_belief_array`), not as "no belief".
    """
    out = {str(i): 0.0 for i in range(n_ranks * CARDS_PER_RANK)}
    for key, mass in rank_belief.items():
        rank = int(key)
        idx_a, idx_b = CARDS_PER_RANK * rank, CARDS_PER_RANK * rank + 1
        live = [i for i in (idx_a, idx_b) if i not in excluded]
        if not live:
            continue  # both copies excluded: mass is on an impossible rank
        share = float(mass) / len(live)
        for i in live:
            out[str(i)] += share

    total = sum(out.values())
    if 0.0 < total < 1.0:
        out = {k: v / total for k, v in out.items()}
    return out
