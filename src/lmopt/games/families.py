"""Game-family map: games registry key -> canonical family key.

Pure module (no pyspiel import) so offline scorers can branch on family
without a game-engine dependency. Behavior that differs by *family*
(P7/P8 posterior precomputes, the property7 scorer's dark-hex skip and
goofspiel branch, prompt serialisers) must derive the family from the
registry key via this map, never by string-comparing the key itself -
depth variants ("darkhex_3x3") would silently fall through an
``== "darkhex"`` check.
"""

from __future__ import annotations

GAME_FAMILIES: dict[str, str] = {
    "kuhn": "kuhn",
    "leduc": "leduc",
    "goofspiel": "goofspiel",
    "liarsdice": "liarsdice",
    "darkhex": "darkhex",
    "darkhex_3x3": "darkhex",
    "goofspiel_5": "goofspiel",
    "goofspiel_6": "goofspiel",
    "liarsdice_1d8": "liarsdice",
    "kuhn_deep": "kuhn",
    "leduc_deep": "leduc",
}


def game_family(key: str) -> str:
    """Family for a GAMES registry key; raises ValueError on unknown keys."""
    try:
        return GAME_FAMILIES[key]
    except KeyError:
        raise ValueError(
            f"unknown game key {key!r}; expected one of {sorted(GAME_FAMILIES)}"
        ) from None
