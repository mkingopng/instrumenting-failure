"""OpenSpiel game loaders and CFR helpers.

OpenSpiel is the single validated source for both the game environment and the
equilibrium reference. This package never hand-rolls a
game or a solver - it loads them from OpenSpiel and documents the exact game
strings used.
"""

from lmopt.games.families import GAME_FAMILIES, game_family
from lmopt.games.loaders import (
    CANONICAL_GAMES,
    DARKHEX_GAME_STRING,
    GAMES,
    GOOFSPIEL_GAME_STRING,
    KUHN_GAME_STRING,
    LEDUC_GAME_STRING,
    LIARSDICE_GAME_STRING,
    independent_deal,
    load_darkhex,
    load_game,
    load_goofspiel,
    load_goofspiel_5,
    load_goofspiel_6,
    load_kuhn,
    load_kuhn_deep,
    load_leduc,
    load_leduc_deep,
    load_liarsdice,
    load_liarsdice_1d8,
)

__all__ = [
    "CANONICAL_GAMES",
    "DARKHEX_GAME_STRING",
    "GAMES",
    "GAME_FAMILIES",
    "KUHN_GAME_STRING",
    "LEDUC_GAME_STRING",
    "GOOFSPIEL_GAME_STRING",
    "LIARSDICE_GAME_STRING",
    "game_family",
    "independent_deal",
    "load_darkhex",
    "load_game",
    "load_kuhn",
    "load_kuhn_deep",
    "load_leduc",
    "load_leduc_deep",
    "load_goofspiel",
    "load_goofspiel_5",
    "load_goofspiel_6",
    "load_liarsdice",
    "load_liarsdice_1d8",
]
