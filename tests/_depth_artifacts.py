"""Depth-solver reference fixtures.

``data/depth_solver/``'s captures are data, not source, so a fresh checkout does not carry
them. Tests that need a reference to load and route use a small synthetic one written here.
``DEPTH_REFERENCES`` (``src/lmopt/games/depth_reference.py``) is the single source for the path.
"""

from __future__ import annotations

import pytest


def _write_synthetic_darkhex_3x3_reference(path):
    """
    write a tiny, valid depth-reference pickle for darkhex_3x3 holding the root info-set only
    It is enough for every test that needs the reference to load and route (the loader's checks
    are the game string, the root key and an overlap with the live game's info-sets), and it
    asserts nothing about the real artefact's contents.
    :param path: where to write the pickle (Path)
    :returns: the path (Path)
    """
    import pickle

    import pyspiel

    from lmopt.games.loaders import load_darkhex_3x3

    game = load_darkhex_3x3()
    root = game.new_initial_state()
    legal = root.legal_actions()
    payload = {
        "policy": {
            root.information_state_string(): {int(a): 1.0 / len(legal) for a in legal}
        },
        "game_string": str(game),
        "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
    }
    path.write_bytes(pickle.dumps(payload))
    return path


@pytest.fixture
def synthetic_darkhex_3x3_reference(tmp_path, monkeypatch):
    """
    point `DEPTH_REFERENCES["darkhex_3x3"]` at a synthetic pickle for the test, so a test that
    only needs the reference to load and route runs without the gitignored real artefact
    """
    from lmopt.games.depth_reference import DEPTH_REFERENCES

    path = _write_synthetic_darkhex_3x3_reference(
        tmp_path / "synthetic_darkhex_3x3.pkl"
    )
    monkeypatch.setitem(DEPTH_REFERENCES, "darkhex_3x3", path)
    return path
