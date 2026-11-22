"""Stage B (B3): load-bearing Goofspiel game-string flags.

`imp_info=true` is the single flag that keeps Goofspiel an imperfect-information
game: without it, past and current bids are observable, the game becomes
effectively sequential, and P3 (randomisation) and P7 (belief) become meaningless.
`points_order=descending` removes the chance player (fixed prize sequence), which
the reach-weighted Goofspiel true-posterior relies on (chance reach = 1). This
test guards both so the flag cannot be silently dropped.
"""

import pytest

pytest.importorskip("pyspiel")

from lmopt.games.loaders import (
    GOOFSPIEL_GAME_STRING,
    load_goofspiel,
)


def test_goofspiel_string_has_imp_info_and_descending():
    assert "imp_info=true" in GOOFSPIEL_GAME_STRING
    assert "points_order=descending" in GOOFSPIEL_GAME_STRING
    assert "returns_type=point_difference" in GOOFSPIEL_GAME_STRING


def test_loaded_goofspiel_params():
    """The loaded game (any num_cards) carries imp_info and descending order."""
    game = load_goofspiel(num_cards=4)
    inner = game.get_parameters()["game"]  # wrapped inner game params
    assert inner["imp_info"] is True
    # points_order is an enum surfaced as a string by OpenSpiel.
    assert "descending" in str(inner["points_order"]).lower()


def test_darkhex_3x3_is_registered():
    from lmopt.games.loaders import GAMES, CANONICAL_GAMES

    assert "darkhex_3x3" in GAMES
    assert (
        "darkhex_3x3" not in CANONICAL_GAMES
    ), "depth variants must stay opt-in so sweeps never pick it up by default"
