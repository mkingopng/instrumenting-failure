"""Prompt set 7 as files: the loader, the move's byte identity, and the readable pins.

The move of set 7 from Python literals into ``src/lmopt/prompts/sets/7/`` changes no
rendered byte. Three digests say so and are pinned here as literals measured before the move:
the content hash ``PROMPT_SPEC_HASHES[7]``, the ten variant content hashes, and the memory
wording hash. The exhaustive per-info-set digests in ``tests/fixtures/prompt_render_pin/``
are the fourth pin and live in their own test.
"""

from __future__ import annotations

import fnmatch
import importlib.resources
import json
import shutil
import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pyspiel = pytest.importorskip("pyspiel")

import lmopt.prompts as prompts
from lmopt.agents.lm_vllm import (
    PROMPT_SCHEMA_VERSION,
    PROMPT_SPEC_HASHES,
    LMVllmAgent,
)
from lmopt.games.loaders import load_game
from lmopt.prompting import get_game_fns_for_game
from lmopt.prompting import spec as spec_module
from lmopt.prompting.memory import fixture_memories, memory_render_hash
from lmopt.prompting.probe_variants import PROBE_TYPES, variant_spec
from lmopt.prompting.spec import prompt_spec_hash
from lmopt.prompts import (
    GAMES,
    LIVE_SET,
    PromptSetError,
    compose,
    load_set,
    set_file_names,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


SET_DIR = REPO_ROOT / "src" / "lmopt" / "prompts" / "sets" / str(LIVE_SET)

# Measured against the Python literals, before any file existed.
SPEC_HASH_7 = "3dcdecc6b754d0e736d934ae5e9944d04477967e75289a188b35f637d5151f9a"
MEMORY_RENDER_HASH_7 = (
    "b5b3d4ec09b8d4d2e0ddbaf5b23e4ce4571914917cfd32842078cf518c0f67d0"
)
VARIANT_HASHES_7 = {
    (
        "kuhn",
        "relabel",
    ): "82ebf8ec3a13cea8bc4982925cb675dbcd937e13386aef88093645a7aa16488b",
    (
        "kuhn",
        "payoff_scale",
    ): "681aaac004a014796231a5d9c03bb9651ff6f22b931409087539c3cf7a285395",
    (
        "leduc",
        "relabel",
    ): "72ff47d3645c64e637a74a160512d2532264e1973a865717c595592f7deae13c",
    (
        "leduc",
        "payoff_scale",
    ): "51962393d1f2be65aa8e1c273ffcfb908cf6b6b41b1679cf28205b95823ae842",
    (
        "goofspiel",
        "relabel",
    ): "c296a00b2ad510efa67fc377de5ec54186d7b21e9dfb0a182598a4ccd9190ecf",
    (
        "goofspiel",
        "payoff_scale",
    ): "9a1bd1970801a9bd9e414b0c56318999fa0b6135ea13f86ece623fba2cc5c6eb",
    (
        "liarsdice",
        "relabel",
    ): "e240309e57cbc7cf7c282e63f90da92781f4bb8d63b7f0213a82783a42f98f54",
    (
        "liarsdice",
        "payoff_scale",
    ): "49bf1fa3c6d08dbafe5c5d9ce93feeb24e92b422ad69e8329fc3c685e98b9831",
    (
        "darkhex",
        "relabel",
    ): "c41230d838b2536189ad8cfa621a4ac6079209329d3e2380c8cc2a71f4131a06",
    (
        "darkhex",
        "payoff_scale",
    ): "44fe4dbd2062a12a8a76aad99c30e19e762dfcc259e06122c4f6692f6b332542",
}


SPEC_HASH_LIVE = SPEC_HASH_7
VARIANT_HASHES_LIVE = VARIANT_HASHES_7


def _load_uncached(number: int):
    """
    the set read fresh from its files, bypassing ``load_set``'s per-process cache so a
    monkeypatched root is honoured
    :param number: the set number (int)
    :returns: the set (PromptSet)
    """
    return load_set.__wrapped__(number)


def _copy_set(tmp_path: Path) -> Path:
    """
    a writable copy of the live set, for tests that break a file
    :param tmp_path: pytest's temporary directory (Path)
    :returns: the copy's directory (Path)
    """
    target = tmp_path / "sets" / str(LIVE_SET)
    shutil.copytree(SET_DIR, target)
    return target


# ---------------------------------------------------------------------------
# 1. The loader
# ---------------------------------------------------------------------------


def test_live_set_is_package_data_reachable_through_importlib_resources():
    """
    The live set is read from the installed package, not from a repo-relative path, and
    every file it holds matches a ``package-data`` glob in pyproject.toml so a wheel ships it.
    """
    root = importlib.resources.files("lmopt.prompts") / "sets" / str(LIVE_SET)
    assert root.is_dir()
    for rel in set_file_names(LIVE_SET):
        assert (root / rel).is_file(), rel
    loaded = load_set(LIVE_SET)
    assert loaded.number == LIVE_SET
    assert set(loaded.games) == set(GAMES)

    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    globs = pyproject["tool"]["setuptools"]["package-data"]["lmopt.prompts"]
    package_dir = REPO_ROOT / "src" / "lmopt" / "prompts"
    for path in SET_DIR.rglob("*"):
        if path.is_file():
            rel = path.relative_to(package_dir).as_posix()
            assert any(
                fnmatch.fnmatch(rel, g) for g in globs
            ), f"{rel} matches no package-data glob"


def test_loader_refuses_a_missing_set():
    """
    Pointing the code at a set that does not exist refuses with the number in the message;
    nothing defaults to the live set.
    """
    with pytest.raises(PromptSetError, match="no prompt set 999"):
        load_set(999)


def test_loader_refuses_a_missing_key(tmp_path, monkeypatch):
    """
    A game file missing one of the spec's fields refuses, naming the field.
    """
    copy = _copy_set(tmp_path)
    kuhn = json.loads((copy / "kuhn.json").read_text(encoding="utf-8"))
    del kuhn["belief_exemplar"]
    (copy / "kuhn.json").write_text(json.dumps(kuhn), encoding="utf-8")
    monkeypatch.setattr(prompts, "set_root", lambda number: copy)
    loaded = _load_uncached(LIVE_SET)
    with pytest.raises(PromptSetError, match="belief_exemplar"):
        loaded.spec("kuhn")


def test_loader_refuses_a_missing_file(tmp_path, monkeypatch):
    """
    A set without one of its files refuses, naming the file.
    """
    copy = _copy_set(tmp_path)
    (copy / "memory.json").unlink()
    monkeypatch.setattr(prompts, "set_root", lambda number: copy)
    with pytest.raises(PromptSetError, match="memory.json"):
        _load_uncached(LIVE_SET)


def test_compose_refuses_an_unknown_wrapper_and_fills_fields():
    """
    A part naming a wrapper the set lacks refuses; a part with fields fills them and keeps
    its prefix; a part without fields leaves the template's own placeholders for the game
    module to fill later.
    """
    wrappers = {
        "objective": "many {unit}s for {currency}",
        "head": "keys: {belief_keys}.",
    }
    parts = [
        "A.",
        {
            "wrapper": "objective",
            "prefix": " ",
            "fields": {"unit": "hand", "currency": "chips"},
        },
        {"wrapper": "head"},
    ]
    assert (
        compose(parts, wrappers, "test")
        == "A. many hands for chipskeys: {belief_keys}."
    )
    with pytest.raises(PromptSetError, match="no_such"):
        compose([{"wrapper": "no_such"}], wrappers, "test")


# ---------------------------------------------------------------------------
# 2. Byte identity of the move
# ---------------------------------------------------------------------------


def test_live_set_content_hash_equals_the_registry_entry():
    """
    The specs built from the files hash to the pinned value for the live set, which is what
    every row collected under it is stamped with (for set 7, what the Python literals hashed to).
    """
    assert PROMPT_SCHEMA_VERSION == LIVE_SET
    assert prompt_spec_hash() == PROMPT_SPEC_HASHES[LIVE_SET] == SPEC_HASH_LIVE


def test_spec_constants_are_the_loaded_set():
    """
    The five module constants the game modules bind to are the set's specs, field for field.
    """
    loaded = load_set(LIVE_SET)
    for constant in spec_module._ALL_SPECS:
        assert constant == loaded.spec(constant.game)
        assert isinstance(constant.instruction, dict)
        assert all(isinstance(v, str) for v in constant.instruction.values())
        assert isinstance(constant.belief_question, str)


def test_variant_content_hashes_are_pinned():
    """
    The ten (game, probe) variant specs built from ``variants/*.json`` hash to the values the
    Python tables produced.
    """
    measured = {
        (game, probe): prompt_spec_hash(variant_spec(game, probe))
        for game in GAMES
        for probe in PROBE_TYPES
    }
    assert measured == VARIANT_HASHES_LIVE


def test_memory_render_hash_is_pinned_for_set_7():
    """
    The memory wording rendered from ``memory.json`` is byte-identical to the Python wording:
    the digest recorded in every v7 run's ``run_meta.json`` is unchanged.
    """
    assert memory_render_hash() == MEMORY_RENDER_HASH_7


# ---------------------------------------------------------------------------
# 3. The readable rendered pins
# ---------------------------------------------------------------------------


def _response(content: str) -> MagicMock:
    """
    a chat completion whose first choice says ``content``
    :param content: the completion text (str)
    :returns: the response double (MagicMock)
    """
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _break(tmp_path, monkeypatch, relpath, edit):
    """
    load a fresh copy of the live set after one parsed file is edited
    :param tmp_path: pytest's temporary directory (Path)
    :param monkeypatch: pytest's monkeypatch (MonkeyPatch)
    :param relpath: the file inside the set to edit (str)
    :param edit: a function that mutates the parsed object in place (callable)
    :returns: the freshly loaded set (PromptSet)
    :raises PromptSetError: when the edited set is refused at load
    """
    copy = _copy_set(tmp_path)
    path = copy / relpath
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(prompts, "set_root", lambda number: copy)
    return _load_uncached(LIVE_SET)


@pytest.mark.parametrize("value", ["1", 1.0, True, 0, -1, None])
def test_a_display_scale_that_is_not_a_positive_integer_is_refused_at_load(
    tmp_path, monkeypatch, value
):
    """
    ``display_scale`` multiplies engine payoffs, so a string or a float would otherwise surface
    as a TypeError inside a game module at import, far from the file. It refuses at load, naming
    the game and the value.
    """
    with pytest.raises(PromptSetError, match="display_scale"):
        _break(tmp_path, monkeypatch, "kuhn.json", lambda d: d.update(display_scale=value))


@pytest.mark.parametrize(
    "key", ["legal_response_one", "legal_response_many", "legal_response_tail"]
)
def test_an_instruct_wrapper_missing_a_key_the_renderer_reads_is_refused_at_load(
    tmp_path, monkeypatch, key
):
    """
    The renderer reads these three templates by name, so a set without one would otherwise fail
    with a KeyError at the first render of a decision node; it fails at load naming the key.
    """
    with pytest.raises(PromptSetError, match=key):
        _break(tmp_path, monkeypatch, "wrappers/instruct.json", lambda d: d.pop(key))


def test_a_payoff_scale_factor_that_disagrees_with_a_replacement_is_refused_at_load(
    tmp_path, monkeypatch
):
    """
    ``factor`` and the literal replacements ("2 chips" -> "20 chips") are held independently,
    so editing one could leave the other stale. The loader checks that each replacement is the
    quantity times the factor, singular or plural.
    """
    def bad_number(data):
        data["games"]["leduc"]["phrases"]["2 chips"][0] = "30 chips"

    def bad_unit(data):
        data["games"]["leduc"]["phrases"]["2 chips"][0] = "20 points"

    def bad_factor(data):
        data["factor"] = 5

    for edit, why in ((bad_number, "30 chips"), (bad_unit, "20 points"), (bad_factor, "factor 5")):
        copy = tmp_path / why.replace(" ", "_")
        copy.mkdir()
        with pytest.raises(PromptSetError, match="scaled|factor"):
            _break(copy, monkeypatch, "variants/payoff_scale.json", edit)


def test_the_live_set_passes_every_load_time_check():
    """
    One anchor on the production path: the shipped set loads through ``load_set`` with the three
    checks on, so a check stricter than the data cannot land unnoticed.
    """
    assert set(load_set(LIVE_SET).games) == set(GAMES)


@pytest.mark.parametrize(
    "fields",
    [{"a": 1}, ["not", "a", "mapping"]],
)
def test_compose_raises_prompt_set_error_on_a_bad_wrapper_fields(fields):
    """
    a wrapper part whose fields miss a placeholder, or are not a mapping, raises
    PromptSetError as the docstring promises, not a bare KeyError or TypeError
    """
    with pytest.raises(PromptSetError, match="fields"):
        compose([{"wrapper": "w", "fields": fields}], {"w": "{b}"}, "test field")


def test_compose_raises_prompt_set_error_on_a_positional_placeholder():
    """
    a wrapper template with a positional placeholder raises PromptSetError, not IndexError
    """
    with pytest.raises(PromptSetError, match="fields"):
        compose([{"wrapper": "w", "fields": {"a": 1}}], {"w": "{0}"}, "test field")


def test_int_keyed_refuses_a_non_ascii_digit_key():
    """
    a table key made of a digit character ``int`` rejects (superscript two) raises
    PromptSetError, not ValueError
    """
    from lmopt.prompts import _int_keyed

    with pytest.raises(PromptSetError, match="not an action index"):
        _int_keyed({"²": "x"}, "corner_names")


# The fixed four-hand memory and window the committed rendered files were written with.
RENDER_PAYOFFS = (1.0, -2.0, None, 0.5)
MEMORY_WINDOW_K = 5


def _first_decision_state(game):
    """
    the game's first decision state, every chance node resolved to its first outcome
    :param game: the loaded pyspiel game (pyspiel.Game)
    :returns: the state (pyspiel.State)
    """
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    return state


def test_darkhex_corner_names_are_keyed_by_action_index():
    """
    ``corner_names`` reaches ``darkhex.make_fns`` keyed by int, as the factory indexes it, so
    the verbose corner names still parse to their cells.
    """
    corners = spec_module.DARKHEX_SPEC.action_vocabulary["corner_names"]
    assert list(corners) == [0, 1, 2, 3]
    fns = get_game_fns_for_game(load_game("darkhex"))
    state = _first_decision_state(load_game("darkhex"))
    info = state.information_state_string(state.current_player())
    assert fns.parse_action("top-left", info) == 0
    assert fns.parse_action("bottom right", info) == 3


def test_wrapper_sentences_reach_the_rendered_prompt(monkeypatch):
    """
    The objective paragraph and the legal-response sentence each game composes from
    ``wrappers/instruct.json`` are present in the prompt the production dispatch renders.
    """
    wrappers = load_set(LIVE_SET).wrapper("instruct")
    units = {
        "kuhn": ("hand", "chips"),
        "leduc": ("hand", "chips"),
        "goofspiel": ("game", "points"),
        "liarsdice": ("hand", "points"),
        "darkhex": ("game", "points"),
    }
    for key in GAMES:
        game = load_game(key)
        fns = get_game_fns_for_game(game)
        state = _first_decision_state(game)
        player = state.current_player()
        info = state.information_state_string(player)
        unit, currency = units[key]
        objective = wrappers["objective"].format(unit=unit, currency=currency)
        assert objective in fns.build_prompt(info, player), key
        if key != "darkhex":
            assert fns.action_suffix(info).endswith(
                wrappers["legal_response_tail"]
            ), key
    altered = spec_module._INSTRUCT["legal_response_tail"] + " ALTERED TAIL"
    # The loaded wrapper is read-only and shared, so the altered tail goes in a copy.
    monkeypatch.setattr(
        spec_module,
        "_INSTRUCT",
        {**spec_module._INSTRUCT, "legal_response_tail": altered},
    )
    kuhn = load_game("kuhn")
    state = _first_decision_state(kuhn)
    info = state.information_state_string(state.current_player())
    assert get_game_fns_for_game(kuhn).action_suffix(info).endswith(altered)


def test_agent_act_composes_the_pinned_m1_prompt():
    """
    ``LMVllmAgent.act`` at M1, holding the four-hand memory the rendered files were written
    with, sends exactly the committed ``kuhn.action_m1.txt``.
    """
    game = load_game("kuhn")
    client = MagicMock()
    agent = LMVllmAgent(
        client=client,
        model="test-model",
        memory_rung="M1",
        memory_window_k=MEMORY_WINDOW_K,
    )
    agent.reset(game, seat=0, opponent_id="fixed")
    state = _first_decision_state(game)
    info = state.information_state_string(state.current_player())
    legal = sorted(agent._fns.action_labels(info))
    agent._memory = fixture_memories(info, legal, RENDER_PAYOFFS)
    client.chat.completions.create.return_value = _response("CHECK")
    agent.act(state)
    pinned = (SET_DIR / "rendered" / "kuhn.action_m1.txt").read_text(encoding="utf-8")
    assert agent.last_prompt == pinned
