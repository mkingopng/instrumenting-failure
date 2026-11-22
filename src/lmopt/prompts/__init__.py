"""
Prompt sets as files.

A prompt set is everything a model reads that does not depend on the game state, as one
numbered, never-edited unit: the five games' content (``GamePromptSpec`` fields), the
cross-hand memory wording, the shared closing sentences (the "wrapper"), and the relabel and
payoff-scale substitutions. Set ``n`` lives in ``sets/<n>/`` beside this file and is read
through ``importlib.resources``, so an installed wheel and an editable checkout resolve it the
same way. ``LIVE_SET`` names the set the code renders; the recorded identifier in every row is
``prompt_schema_version`` and is the same number.

Content is data, rendering is code. A content string is either a literal or a list of parts,
where a part is a literal string or a reference to a wrapper template,
``{"wrapper": "<key>", "prefix": "<optional>", "fields": {...}}``; ``compose`` concatenates
the parts, filling the referenced template's fields when any are given. The parts form keeps
the sentences every game shares in one place without changing a rendered byte.

Nothing here defaults. A missing set, file or key raises ``PromptSetError`` naming what was
missing, because a prompt that silently fell back to another set's text would be collected
under the wrong identifier.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping, Sequence

LIVE_SET = 7

GAMES = ("kuhn", "leduc", "goofspiel", "liarsdice", "darkhex")
WRAPPERS = ("instruct",)
VARIANTS = ("relabel", "payoff_scale")

_SPEC_KEYS = (
    "game",
    "instruction",
    "action_vocabulary",
    "belief_question",
    "belief_exemplar",
    "display_scale",
)
# Vocabulary sub-tables keyed by an action index. JSON object keys are strings, so these are
# read back as integers; every other key is a symbolic name and stays a string.
_INT_KEYED_VOCABULARY = ("corner_names", "card_labels")


class PromptSetError(LookupError):
    """A prompt set, one of its files, or one of its keys is missing or malformed."""


def set_root(number: int):
    """
    the directory of one prompt set, as an ``importlib.resources`` traversable
    :param number: the set number (int)
    :returns: the traversable for ``sets/<number>`` (Traversable)
    :raises PromptSetError: when no such set directory exists
    """
    root = resources.files(__package__) / "sets" / str(number)
    if not root.is_dir():
        raise PromptSetError(f"no prompt set {number}: {root} is not a directory")
    return root


def read_set_text(number: int, relpath: str) -> str:
    """
    one file of a prompt set, as text
    :param number: the set number (int)
    :param relpath: the path inside the set directory, e.g. ``"wrappers/instruct.json"`` (str)
    :returns: the file's text, UTF-8 (str)
    :raises PromptSetError: when the file is absent
    """
    target = set_root(number)
    for piece in relpath.split("/"):
        target = target / piece
    if not target.is_file():
        raise PromptSetError(f"prompt set {number} has no file {relpath!r}")
    return target.read_text(encoding="utf-8")


def read_set_json(number: int, relpath: str) -> Any:
    """
    one JSON file of a prompt set, parsed with key order preserved
    :param number: the set number (int)
    :param relpath: the path inside the set directory (str)
    :returns: the parsed document (Any)
    :raises PromptSetError: when the file is absent or is not a JSON object
    """
    document = json.loads(read_set_text(number, relpath))
    if not isinstance(document, dict):
        raise PromptSetError(
            f"prompt set {number} file {relpath!r} is not a JSON object"
        )
    return document


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    """
    one required key of a set file
    :param mapping: the parsed object (Mapping)
    :param key: the key that must be present (str)
    :param where: what the mapping is, for the error (str)
    :returns: the value (Any)
    :raises PromptSetError: when the key is absent
    """
    if key not in mapping:
        raise PromptSetError(f"{where} has no key {key!r}; present: {sorted(mapping)}")
    return mapping[key]


def compose(content: Any, wrappers: Mapping[str, str], where: str) -> str:
    """
    a content field rendered to its string: a literal stays as it is, a list of parts is
    concatenated, and a wrapper part is its template, optionally prefixed and with its fields
    filled. The composed string is the template the game module later ``format``s per state,
    so a wrapper template's own ``{placeholders}`` survive composition untouched when no fields
    are given
    :param content: the field as the set file holds it (str | list)
    :param wrappers: the wrapper templates by key (Mapping[str, str])
    :param where: what the field is, for the error (str)
    :returns: the composed string (str)
    :raises PromptSetError: on an unknown wrapper key or a malformed part
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise PromptSetError(
            f"{where} must be a string or a list of parts, got {type(content).__name__}"
        )
    out: list[str] = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
            continue
        if not isinstance(part, dict) or "wrapper" not in part:
            raise PromptSetError(
                f"{where} has a part that is neither a string nor a wrapper reference: {part!r}"
            )
        unknown = set(part) - {"wrapper", "prefix", "fields"}
        if unknown:
            raise PromptSetError(
                f"{where} wrapper part has unknown keys {sorted(unknown)}"
            )
        name = part["wrapper"]
        if name not in wrappers:
            raise PromptSetError(
                f"{where} references wrapper {name!r}; present: {sorted(wrappers)}"
            )
        text = wrappers[name]
        fields = part.get("fields", {})
        if fields:
            try:
                text = text.format(**fields)
            except (KeyError, IndexError, TypeError) as exc:
                raise PromptSetError(
                    f"{where} wrapper {name!r} could not be filled from its fields "
                    f"{fields!r}: {exc!r}; fix the part's fields or the wrapper template"
                ) from exc
        out.append(part.get("prefix", "") + text)
    return "".join(out)


def _int_keyed(table: Mapping[str, Any], where: str) -> dict[int, Any]:
    """
    an action-indexed table with its JSON string keys read back as integers
    :param table: the table as the file holds it (Mapping[str, Any])
    :param where: what the table is, for the error (str)
    :returns: the same table keyed by int (dict[int, Any])
    :raises PromptSetError: when a key is not a decimal integer
    """
    out: dict[int, Any] = {}
    for key, value in table.items():
        if not isinstance(key, str) or not (key.isascii() and key.isdecimal()):
            raise PromptSetError(f"{where} key {key!r} is not an action index")
        out[int(key)] = value
    return out


def vocabulary_from_file(vocab: Mapping[str, Any], where: str) -> dict[str, Any]:
    """
    an action vocabulary as the code expects it: symbolic names to label literals, with the
    action-indexed sub-tables re-keyed by int
    :param vocab: the vocabulary as the file holds it (Mapping[str, Any])
    :param where: what the vocabulary is, for the error (str)
    :returns: the vocabulary (dict[str, Any])
    """
    out: dict[str, Any] = {}
    for key, value in vocab.items():
        if key in _INT_KEYED_VOCABULARY:
            out[key] = _int_keyed(value, f"{where}[{key!r}]")
        else:
            out[key] = value
    return out


def vocabulary_to_file(vocab: Mapping[str, Any]) -> dict[str, Any]:
    """
    the inverse of ``vocabulary_from_file``: an action vocabulary in the shape the set file
    holds, with int-keyed sub-tables keyed by decimal strings
    :param vocab: the vocabulary as the code holds it (Mapping[str, Any])
    :returns: the vocabulary for ``json.dumps`` (dict[str, Any])
    """
    out: dict[str, Any] = {}
    for key, value in vocab.items():
        if key in _INT_KEYED_VOCABULARY:
            out[key] = {str(k): v for k, v in value.items()}
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class PromptSet:
    """
    one prompt set, read from its files
    :param number: the set number (int)
    :param games: each game's content file, parsed, by game key (Mapping[str, Mapping])
    :param memory: the memory wording templates (Mapping[str, str])
    :param wrappers: each wrapper's templates, by wrapper name (Mapping[str, Mapping[str, str]])
    :param variants: each variant file, parsed, by probe type (Mapping[str, Mapping])
    :param readme: the set's README text (str)
    """

    number: int
    games: Mapping[str, Mapping[str, Any]]
    memory: Mapping[str, str]
    wrappers: Mapping[str, Mapping[str, str]]
    variants: Mapping[str, Mapping[str, Any]]
    readme: str

    def wrapper(self, name: str) -> Mapping[str, str]:
        """
        one wrapper's templates
        :param name: the wrapper name, e.g. ``"instruct"`` (str)
        :returns: template text by key (Mapping[str, str])
        :raises PromptSetError: when the set has no such wrapper
        """
        if name not in self.wrappers:
            raise PromptSetError(
                f"prompt set {self.number} has no wrapper {name!r}; present: {sorted(self.wrappers)}"
            )
        return self.wrappers[name]

    def variant(self, probe_type: str) -> Mapping[str, Any]:
        """
        one probe type's substitutions file: ``games`` maps each game to what the variant
        changes in it, and a payoff-scale file also carries its ``factor``
        :param probe_type: ``"relabel"`` or ``"payoff_scale"`` (str)
        :returns: the parsed file (Mapping[str, Any])
        :raises PromptSetError: when the set has no such variant
        """
        if probe_type not in self.variants:
            raise PromptSetError(
                f"prompt set {self.number} has no variant {probe_type!r}; present: {sorted(self.variants)}"
            )
        return self.variants[probe_type]

    def spec(self, game: str):
        """
        one game's content as the ``GamePromptSpec`` the game module renders from, every
        composed field composed through the instruct wrapper
        :param game: the game key (str)
        :returns: the spec (GamePromptSpec)
        :raises PromptSetError: when the set has no such game or the file lacks a field
        """
        # Imported here, not at module level: ``lmopt.prompting.spec`` builds its constants
        # from this method at import time, so a module-level import would be circular.
        from lmopt.prompting.spec import GamePromptSpec

        if game not in self.games:
            raise PromptSetError(
                f"prompt set {self.number} has no game {game!r}; present: {sorted(self.games)}"
            )
        data = self.games[game]
        where = f"prompt set {self.number} game {game!r}"
        for key in _SPEC_KEYS:
            _require(data, key, where)
        extra = set(data) - set(_SPEC_KEYS)
        if extra:
            raise PromptSetError(f"{where} has unexpected keys {sorted(extra)}")
        if data["game"] != game:
            raise PromptSetError(f"{where} names itself {data['game']!r}")
        wrappers = self.wrapper("instruct")
        instruction = {
            key: compose(value, wrappers, f"{where} instruction[{key!r}]")
            for key, value in data["instruction"].items()
        }
        return GamePromptSpec(
            game=game,
            instruction=instruction,
            action_vocabulary=vocabulary_from_file(
                data["action_vocabulary"], f"{where} action_vocabulary"
            ),
            belief_question=compose(
                data["belief_question"], wrappers, f"{where} belief_question"
            ),
            belief_exemplar=data["belief_exemplar"],
            display_scale=data["display_scale"],
        )


#: The wrapper templates the renderer reads by name (not through a composed part), so a set
#: that lacks one fails at load instead of at the first render.
_INSTRUCT_KEYS = ("legal_response_one", "legal_response_many", "legal_response_tail")
_QUANTITY = re.compile(r"(\d+) ([A-Za-z]+)")


def _check_display_scale(number: int, game: str, data: Mapping[str, Any]) -> None:
    """
    refuse a game file whose ``display_scale`` is not a positive integer: the game modules
    multiply engine payoffs by it, so a string or a float would fail far from the file
    :param number: the set number (int)
    :param game: the game key (str)
    :param data: the game's parsed content file (Mapping[str, Any])
    :raises PromptSetError: when the value is missing, not an int, a bool or below 1
    """
    where = f"prompt set {number} game {game!r}"
    value = _require(data, "display_scale", where)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PromptSetError(
            f"{where} display_scale must be a positive integer, got {value!r}"
        )


def _check_instruct_wrapper(number: int, table: Mapping[str, str]) -> None:
    """
    refuse an instruct wrapper that lacks a template the renderer reads by name
    :param number: the set number (int)
    :param table: the instruct wrapper's templates (Mapping[str, str])
    :raises PromptSetError: when a required key is absent
    """
    for key in _INSTRUCT_KEYS:
        _require(table, key, f"prompt set {number} wrapper 'instruct'")


def _check_payoff_scale(number: int, variant: Mapping[str, Any]) -> None:
    """
    refuse a payoff-scale variant whose ``factor`` disagrees with its phrase replacements:
    each phrase ``"<n> <unit>"`` must become ``"<n * factor> <unit>"``, singular or plural
    :param number: the set number (int)
    :param variant: the parsed ``variants/payoff_scale.json`` (Mapping[str, Any])
    :raises PromptSetError: when the factor is not an int above 1, or a replacement is not the
        scaled quantity
    """
    where = f"prompt set {number} variants/payoff_scale.json"
    factor = _require(variant, "factor", where)
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 2:
        raise PromptSetError(f"{where} factor must be an integer above 1, got {factor!r}")
    for game, entry in _require(variant, "games", where).items():
        for source, (target, _count) in _require(entry, "phrases", f"{where} {game!r}").items():
            before, after = _QUANTITY.fullmatch(source), _QUANTITY.fullmatch(target)
            ok = (
                before is not None
                and after is not None
                and int(after.group(1)) == int(before.group(1)) * factor
                and after.group(2).rstrip("s") == before.group(2).rstrip("s")
            )
            if not ok:
                raise PromptSetError(
                    f"{where} {game!r}: {source!r} -> {target!r} is not the quantity scaled "
                    f"by the factor {factor}"
                )


class _FrozenDict(dict):
    """
    a dict that refuses every in-place change. It stays a dict, so `json.dumps`, equality and
    every reader see exactly what they saw before, and the prompt spec hashes do not move. A deep
    copy or a pickle yields an ordinary, mutable dict.
    """

    def _refuse(self, *args, **kwargs):
        raise TypeError(
            "prompt set content is shared by every caller of load_set and is read-only; "
            "copy it (copy.deepcopy) before changing it"
        )

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self):
        return (dict, (dict(self),))


class _FrozenList(list):
    """a list that refuses every in-place change; see `_FrozenDict`"""

    _refuse = _FrozenDict._refuse
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _refuse
    append = clear = extend = insert = pop = remove = reverse = sort = _refuse

    def __reduce__(self):
        return (list, (list(self),))


def _freeze(value: Any) -> Any:
    """
    the same JSON value with every dict and list replaced by a read-only one, recursively
    :param value: parsed JSON (Any)
    :returns: the frozen value; strings, numbers, booleans and None are returned as they are (Any)
    """
    if isinstance(value, dict):
        return _FrozenDict((k, _freeze(v)) for k, v in value.items())
    if isinstance(value, list):
        return _FrozenList(_freeze(v) for v in value)
    return value


@lru_cache(maxsize=None)
def load_set(number: int) -> PromptSet:
    """
    read one prompt set from its files, once per process
    :param number: the set number (int)
    :returns: the set (PromptSet)
    :raises PromptSetError: when the set, or any of its files, is missing
    """
    games = {game: read_set_json(number, f"{game}.json") for game in GAMES}
    memory = read_set_json(number, "memory.json")
    for key, value in memory.items():
        if not isinstance(value, str):
            raise PromptSetError(
                f"prompt set {number} memory.json key {key!r} is not a string"
            )
    wrappers = {}
    for name in WRAPPERS:
        table = read_set_json(number, f"wrappers/{name}.json")
        for key, value in table.items():
            if not isinstance(value, str):
                raise PromptSetError(
                    f"prompt set {number} wrapper {name!r} key {key!r} is not a string"
                )
        wrappers[name] = table
    variants = {
        probe: read_set_json(number, f"variants/{probe}.json") for probe in VARIANTS
    }
    readme = read_set_text(number, "README.md")
    for game, data in games.items():
        _check_display_scale(number, game, data)
    _check_instruct_wrapper(number, wrappers["instruct"])
    _check_payoff_scale(number, variants["payoff_scale"])
    # Cached and handed to every caller, so frozen once: a caller that changed it in place would
    # change it for every other caller in the process.
    return PromptSet(
        number=number,
        games=_freeze(games),
        memory=_freeze(memory),
        wrappers=_freeze(wrappers),
        variants=_freeze(variants),
        readme=readme,
    )


def set_file_names(number: int) -> Sequence[str]:
    """
    every file a complete set holds, relative to its directory, in a fixed order
    :param number: the set number, unused today but kept so a later set can differ (int)
    :returns: the relative paths (Sequence[str])
    """
    del number
    return (
        *(f"{game}.json" for game in GAMES),
        "memory.json",
        *(f"wrappers/{name}.json" for name in WRAPPERS),
        *(f"variants/{probe}.json" for probe in VARIANTS),
        "README.md",
    )


__all__ = [
    "GAMES",
    "LIVE_SET",
    "VARIANTS",
    "WRAPPERS",
    "PromptSet",
    "PromptSetError",
    "compose",
    "load_set",
    "read_set_json",
    "read_set_text",
    "set_file_names",
    "set_root",
    "vocabulary_from_file",
    "vocabulary_to_file",
]
