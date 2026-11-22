"""Canonical OpenSpiel game loaders for the five tournament games.

This module is the single source of truth for the exact OpenSpiel game strings
used in the harness. Recording them here (and in tests) keeps the equilibrium
reference and the play environment provably identical.

The five games:

- Kuhn poker (``kuhn_poker``): one private card each, single check/bet round.
  Loaded here as the *single hand*; iteration is a match-level loop, not a
  different OpenSpiel game.
- Leduc poker (``leduc_poker``): two betting rounds with raises and one public
  card - the OpenSpiel default is the 6-card (J/Q/K x2) variant.
- Goofspiel (``goofspiel``): simultaneous-move bidding. We use the
  imperfect-information, point-total configuration and wrap it in
  ``turn_based_simultaneous_game`` so OpenSpiel's tabular CFR (which requires a
  sequential tree) can solve it.
- Liar's Dice (``liars_dice``): sequential bidding over a hidden die roll,
  1d6 per player. CFR-solvable directly without wrapping.
- Dark Hex (``dark_hex``): 2×2 imperfect-information connection game. Each
  player sees only their own stones; the opponent's placements are hidden until
  discovered by attempting to move there. CFR-solvable at 2×2 (441 states);
  3×3 is out of reach for tabular CFR at ~1.9×10¹⁰ information states per
  player (Tapkan 2022) and is separately a proven first-player forced win
  (Broline).
"""

from __future__ import annotations

import json
from typing import Callable

import pyspiel

# --- Canonical game strings ---------------------------------------------------

KUHN_GAME_STRING = "kuhn_poker"

# OpenSpiel default leduc_poker is the 6-card deck (J/Q/K x2), one public card,
# two betting rounds. We pin the default explicitly for provenance.
LEDUC_GAME_STRING = "leduc_poker"

# Imperfect-info, point-difference Goofspiel wrapped to a sequential tree for CFR.
# num_cards is the horizon dial; 4 keeps the CFR tree small and fast. points_order=descending fixes the prize sequence; imp_info=True
# hides the opponent's simultaneous bid; returns_type=point_difference scores by
# (your prizes) − (their prizes), making the game zero-sum so CFR converges.
# total_points would be general-sum (OpenSpiel Utility.GENERAL_SUM) - CFR has no
# convergence guarantee there and nash_conv oscillates in practice.
_GOOFSPIEL_INNER = (
    "goofspiel(imp_info=true,num_cards={num_cards},"
    "points_order=descending,returns_type=point_difference)"
)
GOOFSPIEL_GAME_STRING = (
    f"turn_based_simultaneous_game(game={_GOOFSPIEL_INNER.format(num_cards=4)})"
)

# Liar's Dice: 1 die per player, 6 sides. Two-player imperfect-information
# sequential bidding game. CFR-solvable without wrapping.
LIARSDICE_GAME_STRING = "liars_dice(numdice=1,dice_sides=6)"

# Dark Hex 2×2: imperfect-information connection game. Players see only their own
# stones; the opponent's placements are hidden. 441 states; 500 CFR+ iters in
# ~3s, nash_conv ≈ 1e-5. board_size=3 has ~1.9×10¹⁰ information states per
# player (Tapkan 2022) and is not tractable for tabular CFR on this hardware;
# it is separately a proven first-player forced win (Broline), so is gated out
# of the CFR-reference track on both grounds.
DARKHEX_GAME_STRING = "dark_hex(board_size=2)"


def load_kuhn() -> pyspiel.Game:
    """Load single-hand Kuhn poker (iteration is added at match level)."""
    return pyspiel.load_game(KUHN_GAME_STRING)


def load_leduc() -> pyspiel.Game:
    """Load Leduc poker (OpenSpiel 6-card default, one public card)."""
    return pyspiel.load_game(LEDUC_GAME_STRING)


def load_goofspiel(num_cards: int = 4) -> pyspiel.Game:
    """
    Load imperfect-info, point-total Goofspiel as a sequential (CFR-solvable) game.
    :param num_cards: deck size and horizon dial. Defaults to 4 to
        keep the CFR tree small; larger decks are exponentially
        more expensive to solve exactly.
    """
    inner = _GOOFSPIEL_INNER.format(num_cards=num_cards)
    return pyspiel.load_game(f"turn_based_simultaneous_game(game={inner})")


def load_liarsdice(numdice: int = 1, dice_sides: int = 6) -> pyspiel.Game:
    """
    Load Liar's Dice (default: 1d6 per player, 2 players).
    :param numdice: number of dice per player (default 1).
    :param dice_sides: number of faces per die (default 6).
    """
    return pyspiel.load_game(f"liars_dice(numdice={numdice},dice_sides={dice_sides})")


def load_darkhex(
    board_size: int = 2,
    num_rows: int | None = None,
    num_cols: int | None = None,
) -> pyspiel.Game:
    """
    Load Dark Hex (default: 2×2 board; rectangles via num_rows/num_cols).

    Dark Hex is imperfect-information Hex: each player sees only their own
    stones. Square boards use ``board_size``; the depth-axis rectangles
    (2×3, 3×2 - ~7k info-sets, tractable) pass ``num_rows``/``num_cols``
    (both required together). board_size=3 (~1.9×10¹⁰ information states per
    player, Tapkan 2022) is gated behind the depth-solver track and is
    separately a proven first-player forced win.

    Note: both rectangles are reward-degenerate (forced first/second-player
    win, v0 ≈ ±1); they contribute the P8/VoI and depth axes, not reward
    signal.
    :param board_size: side length of a square board (default 2). Ignored when
        num_rows/num_cols are given.
    :param num_rows: rectangle rows (requires num_cols).
    :param num_cols: rectangle cols (requires num_rows).
    """
    if (num_rows is None) != (num_cols is None):
        raise ValueError("num_rows and num_cols must be given together")
    if num_rows is not None:
        return pyspiel.load_game(f"dark_hex(num_rows={num_rows},num_cols={num_cols})")
    return pyspiel.load_game(f"dark_hex(board_size={board_size})")


def darkhex_board_dims(game: pyspiel.Game) -> tuple[int, int]:
    """(num_rows, num_cols) of a Dark Hex game from its parameters.

    Prefers num_rows/num_cols when both present: a rectangle's parameters also carry a misleading board_size (e.g.
    dark_hex(num_rows=3,num_cols=4) reports board_size=3). Falls back to
    board_size for the square form. Raises KeyError if neither form is
    present (loud, never a silent default).
    """
    params = game.get_parameters()
    if "num_rows" in params and "num_cols" in params:
        return params["num_rows"], params["num_cols"]
    return params["board_size"], params["board_size"]


def load_goofspiel_5() -> pyspiel.Game:
    """Depth-axis Goofspiel with a 5-card deck (6+ awaits cost data)."""
    return load_goofspiel(num_cards=5)


def load_goofspiel_6() -> pyspiel.Game:
    """Depth-axis Goofspiel with a 6-card deck (the CFR-tractability frontier)."""
    return load_goofspiel(num_cards=6)


def load_liarsdice_1d8() -> pyspiel.Game:
    """Depth-axis Liar's Dice with 8-sided dice (the CFR-tractability frontier). Holds numdice=1 -- numdice=2 needs
    a multi-deal posterior rewrite (see `independent_deal`)."""
    return load_liarsdice(numdice=1, dice_sides=8)


def load_kuhn_deep(num_ranks: int = 13) -> pyspiel.Game:
    """Kuhn-shaped universal_poker depth variant:
    same 1-round, no-board-card shape as native kuhn_poker, numRanks-parametric.
    Must load via the dict param form -- a string config mis-parses single-token
    numeric values as kInt instead of the required kString.

    numRanks=13 is an engine STRUCTURAL ceiling, not a CFR-tractability
    frontier (unlike goofspiel_6/liarsdice_1d8): universal_poker segfaults
    above 13 ranks. The canonical-shape
    (numRanks=3) exploitability matched native kuhn_poker exactly,
    independent_deal() confirmed False (without-replacement deal, same as
    native Kuhn/Leduc), the rank-char table confirmed "23456789TJQKA", and
    numRanks=13 converges trivially on CFR+ grounds alone (150 iterations,
    exploitability 7.1e-5, 3.28s) -- CFR was never the binding constraint here.
    """
    params = {
        "betting": "limit",
        "numPlayers": 2,
        "numRounds": 1,
        "numRanks": num_ranks,
        "numSuits": 1,
        "numHoleCards": 1,
        "numBoardCards": "0",
        "blind": "1 1",
        "raiseSize": "1",
        "maxRaises": "1",
        "firstPlayer": "1",
    }
    return pyspiel.load_game("universal_poker", params)


def load_leduc_deep(num_ranks: int = 13) -> pyspiel.Game:
    """Leduc-shaped universal_poker depth variant:
    same 2-round, 1-public-card, 2-suit shape as native leduc_poker,
    numRanks-parametric. Must load via the dict param form (same reason as
    load_kuhn_deep).

    numRanks=13 is an engine STRUCTURAL ceiling (universal_poker segfaults
    above 13 ranks, confirmed by source: MAX_RANKS=13 in the underlying ACPC
    engine, open_spiel/games/universal_poker/acpc/project_acpc_server/game.h),
    the same ceiling kuhn_deep hits. Unlike kuhn_deep, CFR+ convergence here
    was not trivial: canonical-shape (numRanks=3) exploitability matched
    native leduc_poker exactly, independent_deal() confirmed False, and a
    real timed solve at numRanks=13 took ~1h43m to converge (550 iterations,
    exploitability 9.88e-4) -- Leduc's 2-round, raise-capable betting
    structure is far more expensive per iteration than Kuhn's single-round
    shape, even at the identical max deck size.
    """
    params = {
        "betting": "limit",
        "numPlayers": 2,
        "numRounds": 2,
        "numRanks": num_ranks,
        "numSuits": 2,
        "numHoleCards": 1,
        "numBoardCards": "0 1",
        "blind": "1 1",
        "raiseSize": "2 4",
        "maxRaises": "2 2",
        "firstPlayer": "1 1",
    }
    return pyspiel.load_game("universal_poker", params)


def load_darkhex_3x3() -> pyspiel.Game:
    """Depth-axis Dark Hex 3x3.

    No full CFR+ reference exists or can exist: TabularPolicy's full-tree
    enumeration MemoryErrors (6 GB cap, 17.6 s; the 2x3 control completes in
    0.6 s). Its reference is the PARTIAL MCCFR capture in
    data/depth_solver/, served via lmopt.games.depth_reference. Excluded from
    CANONICAL_GAMES so no sweep or warm entrypoint picks it up implicitly.
    """
    return load_darkhex(num_rows=3, num_cols=3)


# The five canonical (reported) games. Depth variants are additive
# registry entries used by the depth-axis study; sweep/warm entrypoints
# default to this tuple so variants are always an explicit opt-in.
CANONICAL_GAMES: tuple[str, ...] = (
    "kuhn",
    "leduc",
    "goofspiel",
    "liarsdice",
    "darkhex",
)

# Registry mapping short keys to loaders.
# Depth-axis variants: darkhex_3x3 + goofspiel_5.
# darkhex_2x3 and darkhex_3x2 are not registered. The registry's Dark Hex keys
# are "darkhex", the 2x2 board, and "darkhex_3x3".
# goofspiel_6 is the CFR-tractability frontier; goofspiel_7 is out of reach (39.7GB RSS, no
# convergence). liarsdice_1d8 is the analogous frontier for Liar's Dice;
# dice_sides=9 is out of reach (RSS 927MB->3.1GB in 12s).
# Native leduc_poker has no deck/depth parameter (only players/
# starting_player/action_mapping/suit_isomorphism), so its depth axis is
# leduc_deep below. liarsdice numdice=2 needs the multi-deal posterior
# rewrite; do not add it here until that exists.
# kuhn_deep is Kuhn poker's own depth axis, loaded via universal_poker rather than kuhn_poker (which has no
# deck-size parameter, unlike leduc_poker's absence of any depth parameter
# noted above). numRanks=13 is different in kind from goofspiel_6/
# liarsdice_1d8's frontiers: it is an engine STRUCTURAL ceiling, not a
# CFR-tractability limit -- universal_poker segfaults above 13 ranks;
# CFR+ converges trivially at 13 ranks (150 iterations,
# exploitability 7.1e-5, 3.28s), so no tighter bound was needed on
# convergence grounds alone.
GAMES = {
    "kuhn": load_kuhn,
    "leduc": load_leduc,
    "goofspiel": load_goofspiel,
    "liarsdice": load_liarsdice,
    "darkhex": load_darkhex,
    "darkhex_3x3": load_darkhex_3x3,
    "goofspiel_5": load_goofspiel_5,
    "goofspiel_6": load_goofspiel_6,
    "liarsdice_1d8": load_liarsdice_1d8,
    "kuhn_deep": load_kuhn_deep,
    "leduc_deep": load_leduc_deep,
}

# Per-key extraction of the loader kwargs a game was actually built with, from
# its live pyspiel.Game's own get_parameters(). Never a raw get_parameters()
# dump: none of this registry's loader functions accept their own full raw
# parameter dict back as **kwargs (kuhn/leduc take none at all; goofspiel's
# raw params are nested under "game"; darkhex's board_size is misleading on
# rectangles -- prompting/__init__.py's own documented precedent). Dispatched
# by exact registry key, never by game_family(): kuhn_deep/leduc_deep share
# the kuhn/leduc family but are registered as their own parametric functions
# (num_ranks, a real signature parameter -- unlike goofspiel_5/goofspiel_6/
# liarsdice_1d8/darkhex_3x3, which are genuine zero-arg () -> Game pins).
# Family-level dispatch would silently drop num_ranks the moment anything
# ever varied it. An unrecognised key raises rather than silently returning
# {} or the raw dict (no silent fallbacks).
GAME_PARAMS_EXTRACTORS: dict[str, Callable[[dict], dict]] = {
    "kuhn": lambda params: {},
    "leduc": lambda params: {},
    "goofspiel": lambda params: {"num_cards": params["game"]["num_cards"]},
    "goofspiel_5": lambda params: {},
    "goofspiel_6": lambda params: {},
    "liarsdice": lambda params: {
        "numdice": params["numdice"],
        "dice_sides": params["dice_sides"],
    },
    "liarsdice_1d8": lambda params: {},
    "darkhex": lambda params: {
        "num_rows": params["num_rows"],
        "num_cols": params["num_cols"],
    },
    "darkhex_3x3": lambda params: {},
    # numRanks (raw pyspiel key, universal_poker camelCase) -> num_ranks (the
    # loader's own kwarg, snake_case). Mapped explicitly per key, never by a
    # generic case-transform: nothing guarantees every future loader's raw
    # key lowers to its exact kwarg name.
    "kuhn_deep": lambda params: {"num_ranks": params["numRanks"]},
    "leduc_deep": lambda params: {"num_ranks": params["numRanks"]},
}


def game_params_for(key: str, game: pyspiel.Game) -> str:
    """Canonical, sorted-key JSON of the loader kwargs `game` was built with.

    This is what a landed row's `game_params` column should carry: exactly
    what would need to be passed to ``GAMES[key](**kwargs)`` to reconstruct
    the same game, derived from the live game's own `get_parameters()` via
    `GAME_PARAMS_EXTRACTORS[key]` -- never a raw `get_parameters()` dump (see
    that dict's own docstring comment for why). Sorted keys: this column's
    purpose is to be read years from now by someone asking "what board was
    this?", so its serialisation must not depend on pyspiel's own dict
    ordering or drift across versions.

    Raises `KeyError` for a key with no registered extractor -- a new GAMES
    entry with no matching extractor is a defect to surface immediately, not
    a silent `{}` (no silent fallbacks).
    """
    extractor = GAME_PARAMS_EXTRACTORS[key]
    return json.dumps(extractor(game.get_parameters()), sort_keys=True)


def independent_deal(game: pyspiel.Game) -> bool:
    """True if the initial deal draws each player's hidden state independently.

    Distinguishes two deal models that the Bayesian posterior must treat
    differently:

    - Without replacement (Kuhn, Leduc): cards come from a shared deck, so
      the two players' private cards are *mutually exclusive* - the opponent
      cannot hold the card you hold. The second deal's support is smaller than
      the first (Kuhn 3->2, Leduc 6->5).
    - Independent (Liar's Dice): each player rolls their own die, so the
      opponent's die *can* equal yours. Both deals draw from the full support
      (dice 6->6).

    Detected from deck structure rather than a game-name branch ("config over
    forks"): an independent deal leaves the second deal's support
    equal to the first; a without-replacement deal shrinks it.

    Games with no initial deal (Goofspiel, Dark Hex - 0 leading chance nodes)
    return False; the Bayesian agent does not use the dealt-card posterior path
    for them.

    Precondition (validated game set only). This compares the first two
    leading chance nodes and assumes they are the two players' single private
    deals - true for Kuhn, Leduc, and Liar's Dice at ``numdice=1``. It is not
    valid for a game that deals multiple private cards to one player before the
    other (e.g. ``numdice>=2`` Liar's Dice, where the first two chance nodes are
    both P0's dice and would be read as ``[6,6] -> True``), nor for a public
    chance card of equal support interleaved with the private deals. Those
    configurations need a multi-deal posterior rewrite and must not rely on this helper. The dealt-card posterior in
    ``BayesianAgent`` / ``compute_true_posterior`` shares the same single-deal
    precondition (it indexes ``history[seat]``), so this helper is exactly as
    scoped as its only callers.
    """
    s = game.new_initial_state()
    counts: list[int] = []
    while s.is_chance_node() and len(counts) < 2:
        counts.append(len(s.chance_outcomes()))
        s.apply_action(s.legal_actions()[0])
    return len(counts) >= 2 and counts[0] == counts[1]


def load_game(name: str, **kwargs) -> pyspiel.Game:
    """Load a game by its ``GAMES`` registry key: the 5 canonical families
    (``"kuhn" | "leduc" | "goofspiel" | "liarsdice" | "darkhex"``) plus every
    depth/size variant (``"darkhex_3x3" | "goofspiel_5" | "goofspiel_6" |
    "liarsdice_1d8" | "kuhn_deep" | "leduc_deep"``). ``GAMES`` itself, not this
    docstring, is the authoritative key space.

    Extra keyword arguments are forwarded to the loader (e.g. ``num_cards`` for
    Goofspiel, ``numdice``/``dice_sides`` for Liar's Dice, ``board_size`` for
    Dark Hex). The depth-variant keys are genuine zero-argument pins and take
    no kwargs (``kuhn_deep``/``leduc_deep`` accept an optional ``num_ranks``).
    """
    try:
        loader = GAMES[name]
    except KeyError:
        raise ValueError(
            f"unknown game {name!r}; expected one of {sorted(GAMES)}"
        ) from None
    return loader(**kwargs)
