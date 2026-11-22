"""NL prompt construction and response parsing for lmopt games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import lmopt.prompting.darkhex as _darkhex
import lmopt.prompting.goofspiel as _goof
import lmopt.prompting.kuhn as _kuhn
import lmopt.prompting.kuhn_deep as _kuhn_deep
import lmopt.prompting.leduc as _leduc
import lmopt.prompting.leduc_deep as _leduc_deep
import lmopt.prompting.liarsdice as _liars
from lmopt.games.loaders import darkhex_board_dims, load_game

# Re-export Kuhn symbols for backward compatibility.
from lmopt.prompting.kuhn import (
    action_labels,
    action_suffix,
    belief_suffix,
    build_prompt,
    parse_action,
    parse_belief,
)
from lmopt.prompting.memory import MemoryTerms
from lmopt.prompting.spec import split_reasoning


@dataclass
class GamePromptFns:
    """Prompt-building and parsing functions for one game."""

    build_prompt: Callable  # (info_str, current_player) -> str
    action_suffix: Callable  # (info_str) -> str
    # (info_str) -> dict[int, str]: legal action -> the exact literal
    # action_suffix offers for it, sourced from the same data action_suffix
    # itself renders from (never a re-parse of action_suffix's rendered
    # text, and never a hand-written scheme independent of it).
    # tests/test_prompt_fidelity.py enumerates the full tree of
    # every registered variant to verify every label appears verbatim in
    # action_suffix's own output and the label-key set equals legal_actions.
    action_labels: Callable
    belief_suffix: Callable  # (info_str) -> str | None
    parse_action: Callable  # (text, info_str) -> int | None
    parse_belief: Callable  # (text, info_str) -> str | None
    parse_own_card: Callable  # (info_str) -> int | None
    # Optional: (text, info_str) -> bool. True when parse_action resolved via a
    # vocabulary alias (Leduc BET->RAISE) so the normalisation is logged, not
    # silent. None for games with no alias path.
    parse_action_aliased: Optional[Callable] = None
    # Optional: (text, info_str) -> str | None. Returns which vocabulary resolved
    # the action ("label" | "verbose" | "coords" | "digit"). None for games without
    # multiple vocabularies.
    parse_action_vocab: Optional[Callable] = None
    # The words and payoff units the cross-hand memory block is written in. None only for a
    # hand-built instance; an agent at a memory level above M0 refuses one.
    memory_terms: Optional[MemoryTerms] = None


_KUHN_FNS = GamePromptFns(
    build_prompt=_kuhn.build_prompt,
    action_suffix=_kuhn.action_suffix,
    action_labels=_kuhn.action_labels,
    belief_suffix=_kuhn.belief_suffix,
    parse_action=_kuhn.parse_action,
    parse_belief=_kuhn.parse_belief,
    parse_own_card=_kuhn.parse_own_card,
    memory_terms=_kuhn.MEMORY_TERMS.with_labels(_kuhn.action_labels),
)

_LEDUC_FNS = GamePromptFns(
    build_prompt=_leduc.build_prompt,
    action_suffix=_leduc.action_suffix,
    action_labels=_leduc.action_labels,
    belief_suffix=_leduc.belief_suffix,
    parse_action=_leduc.parse_action,
    parse_belief=_leduc.parse_belief,
    parse_own_card=_leduc.parse_own_card,
    parse_action_aliased=_leduc.parse_action_aliased,
    memory_terms=_leduc.MEMORY_TERMS.with_labels(_leduc.action_labels),
)

_GOOFSPIEL_FNS = GamePromptFns(
    build_prompt=_goof.build_prompt,
    action_suffix=_goof.action_suffix,
    action_labels=_goof.action_labels,
    belief_suffix=_goof.belief_suffix,
    parse_action=_goof.parse_action,
    parse_belief=_goof.parse_belief,
    parse_own_card=_goof.parse_own_card,
    memory_terms=_goof.MEMORY_TERMS.with_labels(_goof.action_labels),
)

_LIARSDICE_FNS = GamePromptFns(
    build_prompt=_liars.build_prompt,
    action_suffix=_liars.action_suffix,
    action_labels=_liars.action_labels,
    belief_suffix=_liars.belief_suffix,
    parse_action=_liars.parse_action,
    parse_belief=_liars.parse_belief,
    parse_own_card=_liars.parse_own_card,
    memory_terms=_liars.MEMORY_TERMS.with_labels(_liars.action_labels),
)

_DARKHEX_FNS = GamePromptFns(
    build_prompt=_darkhex.build_prompt,
    action_suffix=_darkhex.action_suffix,
    action_labels=_darkhex.action_labels,
    belief_suffix=_darkhex.belief_suffix,
    parse_action=_darkhex.parse_action,
    parse_action_vocab=_darkhex.parse_action_vocab,
    parse_belief=_darkhex.parse_belief,
    parse_own_card=_darkhex.parse_own_card,
    memory_terms=_darkhex.MEMORY_TERMS.with_labels(_darkhex.action_labels),
)

# Maps our short game names and OpenSpiel short_name values to prompt fns.
_GAME_FNS_MAP: dict[str, GamePromptFns] = {
    "kuhn": _KUHN_FNS,
    "kuhn_poker": _KUHN_FNS,
    "leduc": _LEDUC_FNS,
    "leduc_poker": _LEDUC_FNS,
    "goofspiel": _GOOFSPIEL_FNS,
    "turn_based_simultaneous_game": _GOOFSPIEL_FNS,  # wrapped Goofspiel
    "liarsdice": _LIARSDICE_FNS,
    "liars_dice": _LIARSDICE_FNS,
    "darkhex": _DARKHEX_FNS,
    "dark_hex": _DARKHEX_FNS,
}


def get_game_fns(game_name: str) -> GamePromptFns:
    """Return prompt/parse functions for the named game.

    Accepts both our short keys ('kuhn', 'leduc', 'goofspiel') and the raw
    OpenSpiel short_name values ('kuhn_poker', 'leduc_poker',
    'turn_based_simultaneous_game').
    """
    try:
        return _GAME_FNS_MAP[game_name]
    except KeyError:
        raise ValueError(
            f"unknown game {game_name!r}; expected one of {sorted(_GAME_FNS_MAP)}"
        ) from None


# OpenSpiel short_name -> family (the live game object's self-description).
_SHORTNAME_FAMILY: dict[str, str] = {
    "kuhn_poker": "kuhn",
    "leduc_poker": "leduc",
    "turn_based_simultaneous_game": "goofspiel",
    "liars_dice": "liarsdice",
    "dark_hex": "darkhex",
}

# Family default instances (canonical sizes) for the fixed-size init path.
_FAMILY_DEFAULT_FNS: dict[str, GamePromptFns] = {
    "kuhn": _KUHN_FNS,
    "leduc": _LEDUC_FNS,
    "goofspiel": _GOOFSPIEL_FNS,
    "liarsdice": _LIARSDICE_FNS,
    "darkhex": _DARKHEX_FNS,
}

# Families whose factory takes the prompt spec as its only parameter.
# darkhex is absent because its factory takes board shape first and the
# spec last; it is handled in its own branch of ``get_game_fns_for_game`` and
# is spec-aware all the same. liarsdice is handled the same way: its factory
# takes a depth dimension (``dice_sides``) alongside its spec parameter, so it
# needs the live game's parameters read before dispatch, exactly like
# darkhex's board shape and goofspiel's num_cards.
_SPEC_FACTORIES: dict[str, Callable] = {
    "kuhn": _kuhn.make_fns,
    "leduc": _leduc.make_fns,
}

# Families that can honour a variant spec at all, which is all five.
# darkhex takes a trailing optional spec (the column-label alphabet is spec
# content there), and goofspiel's two probe types each have a variant to
# inject -- the bid cards spelled as number words for the relabel run, and a
# rendered-prize scale factor for the payoff-scale run. darkhex,
# goofspiel and liarsdice are listed explicitly rather than via
# ``_SPEC_FACTORIES`` because their factories take a dimension first and the
# spec last, so each has its own dispatch branch below.
_SPEC_AWARE_FAMILIES: frozenset[str] = frozenset(_SPEC_FACTORIES) | {
    "darkhex",
    "goofspiel",
    "liarsdice",
}


def get_game_fns_for_game(game, spec=None) -> GamePromptFns:
    """Authoritative prompt-fns dispatch from a live pyspiel game object.

    Family from ``game.get_type().short_name``; dimensions from
    ``game.get_parameters()``. This is the only correct dispatch for depth
    variants: every Dark Hex rectangle shares short_name ``dark_hex`` and
    every wrapped Goofspiel size shares ``turn_based_simultaneous_game``, so
    name-keyed lookups cannot distinguish sizes.

    Dark Hex dims: ``num_rows``/``num_cols`` win when both present:
    dark_hex(num_rows=3,num_cols=4) params also carry a
    misleading ``board_size: 3``. Falls back to ``board_size`` (canonical
    square loader form). Wrapped Goofspiel nests the real params under
    ``params["game"]``.

    :param game: a live pyspiel game object (pyspiel.Game)
    :param spec: an optional variant ``GamePromptSpec``, the injection point for the P1 and P4
        probe runs. ``None`` returns the canonical shared instance. A spec for a family whose
        factory cannot yet honour one raises rather than silently discarding it (no silent
        fallbacks): a discarded spec would let a probe arm log ``prompt_variant`` against
        canonical bytes, which is a fabricated measurement (Optional[GamePromptSpec])
    :returns: the prompt and parse functions for this game (GamePromptFns)
    """
    short_name = game.get_type().short_name
    family: str | None
    if short_name == "universal_poker":
        # universal_poker's short_name is shared by
        # every kuhn-shaped and leduc-shaped depth variant (both loaded via
        # the same OpenSpiel game string), so its family must be read from
        # numRounds, not the short_name-keyed map below (which cannot
        # distinguish them, the same class of bug _SHORTNAME_FAMILY's own
        # docstring names for Dark Hex rectangles/wrapped Goofspiel sizes).
        params = game.get_parameters()
        num_rounds = params["numRounds"]
        num_ranks = params.get("numRanks")
        if num_rounds == 1:
            family = "kuhn"
        elif num_rounds == 2:
            family = "leduc"
        else:
            raise ValueError(
                f"universal_poker with numRounds={num_rounds} matches no "
                f"registered depth-variant family (expected 1=kuhn-shaped or "
                f"2=leduc-shaped)"
            )
        if spec is not None and spec.game != family:
            raise ValueError(
                f"spec for game {spec.game!r} handed to the {family!r} "
                f"universal_poker dispatch"
            )
        if family == "kuhn":
            return GamePromptFns(**_kuhn_deep.make_fns(num_ranks, spec))
        return GamePromptFns(**_leduc_deep.make_fns(num_ranks, spec))
    family = _SHORTNAME_FAMILY.get(short_name)
    if family is None:
        raise ValueError(
            f"unknown game {short_name!r}; expected one of {sorted(_SHORTNAME_FAMILY)}"
        )
    if spec is not None and family not in _SPEC_AWARE_FAMILIES:
        raise ValueError(
            f"{family!r} does not accept a GamePromptSpec: its factory takes no "
            f"spec parameter, so honouring one here would silently discard it "
            f"and a probe arm would then log prompt_variant against canonical "
            f"bytes (no silent fallbacks). Spec-aware families: "
            f"{sorted(_SPEC_AWARE_FAMILIES)}."
        )
    if spec is not None and spec.game != family:
        raise ValueError(
            f"spec for game {spec.game!r} handed to the {family!r} dispatch. A "
            f"spec carries another game's prose and vocabulary, so rendering it "
            f"here would either raise deep inside a closure or emit the wrong "
            f"game's text under this game's name."
        )
    params = game.get_parameters()
    if family == "darkhex":
        rows, cols = darkhex_board_dims(game)
        return GamePromptFns(**_darkhex.make_fns(rows, cols, spec))
    if family == "goofspiel":
        inner = params.get("game", params)
        return GamePromptFns(**_goof.make_fns(inner["num_cards"], spec))
    if family == "liarsdice":
        # dice_sides varies by depth variant (liarsdice_1d8 and friends); a
        # raw `_FAMILY_DEFAULT_FNS["liarsdice"]`
        # shortcut here would silently render every variant with the
        # canonical 6-sided vocabulary regardless of the live game's actual
        # dice_sides (every info-set's action_labels() key set would disagree
        # with legal_actions()).
        dice_sides = params["dice_sides"]
        return GamePromptFns(**_liars.make_fns(spec, dice_sides))
    if spec is None:
        return _FAMILY_DEFAULT_FNS[family]
    return GamePromptFns(**_SPEC_FACTORIES[family](spec))


def get_game_fns_by_registry_key(key: str, spec=None) -> GamePromptFns:
    """Authoritative prompt-fns dispatch from a landed row's ``game`` column.

    ``get_game_fns(name)`` is keyed by OpenSpiel family/short_name, which
    cannot distinguish depth/size variants sharing one short_name:
    ``goofspiel_5``/``goofspiel_6`` both report short_name
    ``turn_based_simultaneous_game``, so ``get_game_fns("goofspiel_6")``
    raises ``ValueError`` (loudly -- ``_GAME_FNS_MAP`` has no such key).

    ``key`` is the registry key a landed row's ``game`` column actually
    carries (``loaders.GAMES``'s key space: the 5 canonical families plus
    ``darkhex_3x3``/``goofspiel_5``/``goofspiel_6``/``liarsdice_1d8``/
    ``kuhn_deep``/``leduc_deep``), not a ``game_params`` value: every depth
    variant this registry currently holds is a genuine zero-argument pin
    (``GAME_PARAMS_EXTRACTORS``'s own comment), so the key alone fully
    determines the live game object. ``game_params`` is not the depth axis:
    older landed rows carry the placeholder ``game_params == "{}"``.
    """
    game = load_game(key)
    return get_game_fns_for_game(game, spec=spec)


__all__ = [
    "GamePromptFns",
    "get_game_fns",
    "get_game_fns_for_game",
    "get_game_fns_by_registry_key",
    "split_reasoning",
    # Kuhn backward-compat exports
    "action_labels",
    "action_suffix",
    "belief_suffix",
    "build_prompt",
    "parse_action",
    "parse_belief",
]
