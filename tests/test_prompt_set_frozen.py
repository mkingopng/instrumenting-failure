"""
`load_set` is cached and its result shared by every caller, so its parsed JSON is read-only: a caller
that changed it in place would change it for every other caller in the process. Freezing must not
change what any reader sees, or the prompt spec hashes would move.
"""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from lmopt.prompts import _freeze, load_set


def test_nested_content_refuses_every_in_place_change():
    prompt_set = load_set(7)
    game = prompt_set.games["kuhn"]
    with pytest.raises(TypeError, match="read-only"):
        game["instruction"] = "changed"
    nested_list = next(v for v in _walk(prompt_set.games) if isinstance(v, list))
    with pytest.raises(TypeError, match="read-only"):
        nested_list.append("x")
    with pytest.raises(TypeError, match="read-only"):
        prompt_set.memory.update({"x": "y"})


def test_freezing_changes_nothing_a_reader_sees():
    """
    The frozen value serialises, compares and iterates exactly as the parsed JSON did, so a hash
    over `json.dumps` is unchanged.
    """
    raw = {"a": [1, {"b": [2, 3]}], "c": "d"}
    frozen = _freeze(raw)
    assert frozen == raw
    assert json.dumps(frozen, sort_keys=True) == json.dumps(raw, sort_keys=True)
    assert isinstance(frozen, dict) and isinstance(frozen["a"], list)


def test_a_deep_copy_and_a_pickle_give_an_ordinary_mutable_value():
    """
    The sites that copy before editing keep working, and a value sent to a worker process is a
    plain one.
    """
    frozen = _freeze({"a": [1, {"b": 2}]})
    copied = copy.deepcopy(frozen)
    copied["a"].append(3)
    copied["a"][1]["b"] = 9
    assert type(copied) is dict and type(copied["a"]) is list
    assert frozen == {"a": [1, {"b": 2}]}
    restored = pickle.loads(pickle.dumps(frozen))
    restored["new"] = 1
    assert type(restored) is dict


def _walk(value):
    if isinstance(value, dict):
        for v in value.values():
            yield v
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield v
            yield from _walk(v)
