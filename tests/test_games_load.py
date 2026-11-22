"""Games load, play to terminal, and expose the API the harness depends on
(legal-action masks, info-state tensors, chance nodes)."""

import pytest

pytest.importorskip("pyspiel")

from lmopt.games import GAMES, load_game

GAME_KEYS = [
    "kuhn",
    "leduc",
    "goofspiel",
    "liarsdice",
    "darkhex",
    "goofspiel_5",  # depth variant
    "goofspiel_6",  # game-depth-sweep depth variant
    "liarsdice_1d8",  # game-depth-sweep depth variant
    "kuhn_deep",  # game-depth-sweep depth variant
    "leduc_deep",  # game-depth-sweep depth variant
    "darkhex_3x3",  # depth-reference game (partial MCCFR reference only)
]


@pytest.mark.parametrize("name", GAME_KEYS)
def test_game_loads(name):
    game = load_game(name)
    assert game.num_players() == 2


@pytest.mark.parametrize("name", GAME_KEYS)
def test_decision_node_exposes_legal_actions_and_tensor(name):
    """Walk to the first decision node and confirm legal actions + info-state
    tensor are accessible - the inputs the prompt serialiser consumes."""
    game = load_game(name)
    state = game.new_initial_state()
    while state.is_chance_node():
        # Take the first chance outcome deterministically to reach a player node.
        state.apply_action(state.chance_outcomes()[0][0])

    assert not state.is_terminal()
    legal = state.legal_actions()
    assert len(legal) >= 1

    tensor = state.information_state_tensor()
    assert len(tensor) == game.information_state_tensor_size()


def test_registry_keys():
    assert set(GAMES) == set(GAME_KEYS)


def test_unknown_game_raises():
    with pytest.raises(ValueError):
        load_game("not_a_game")
