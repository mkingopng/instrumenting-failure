"""Probe-variant prompt specs for P1 (relabel) and P4 (payoff_scale).

What this module is. A probe arm renders the same game through a variant
``GamePromptSpec`` and asks whether the LM's policy moved. P1's variant renames
the action vocabulary; P4's variant scales the stake prose. Both are content
only: the engine, the seats, the seeds and the scoring reference are untouched,
so a policy difference between the base arm and the probe arm is attributable to
the prompt text and nothing else.

Why the registry records absences as data, with none live. Every one
of the ten (game, probe_type) pairs has a variant: ``VARIANT_ABSENT_REASONS``
is empty. The machinery stays because an absence is a
measurement result the acceptance table has to print per game rather than average
away. Absences are downstream of prompt content: a game with no relabelling that
leaves the bid order intact, or no rendered payoff magnitude for a
payoff scale to move, has none. Prompt version 7 states a payoff in every game, and the
goofspiel relabel spells the bid cards as number words, which carry their own
order without a sentence of added prose. ``variant_spec`` returns ``None`` only
where ``VARIANT_ABSENT_REASONS[game][probe_type]`` holds a ``VariantAbsence``
naming both the reason and what P1/P4 would need in that game, so a variant
withdrawn later re-enters as data rather than as silence. Unknown game or probe
keys RAISE: "no such variant" and "no such game" are different answers and the
caller must not confuse them.

Two kinds of payoff scaling, because two kinds of magnitude are rendered. A
magnitude stated in the template prose ("stake 1 chip", "wins 1 point") moves
through the whitelist below, each phrase carrying the count it must appear with
so a drift in the canonical prose raises here instead of shipping a partly
scaled variant. A magnitude computed from GAME state (Leduc's pot and chip
stacks, Goofspiel's prize values and scores) has no prose to whitelist, so it
travels on ``GamePromptSpec.display_scale`` and the game module multiplies by it
at render time. Goofspiel is the pure case of the second kind: its payoff-scale
variant changes no template byte at all.

Why some present variants also carry a caveat. ``VARIANT_CAVEATS`` records what a
variant that exists does not cover, in the same reachable-data shape. Live:
leduc's and kuhn's rules prose name the actions a second time in lowercase English
("check (free), bet, call (match a bet)"), which the relabel deliberately does not
rewrite (conjugation would change prose beyond the labels), and that is checked by
a test rather than merely asserted here; goofspiel's relabel moves the bid cards
and leaves prize values and scores as digits, so its prompt mixes two notations
for one deck; and dark hex's payoff scale has exactly one magnitude to move, which
makes it total and, for the same reason, the least informative of the five.
Kuhn's payoff scaling is total because prompt version 7 states every magnitude,
the ante included.

No silent fallbacks. A caller must never pass ``variant_spec(...)``
straight into ``get_game_fns_for_game(game, spec=...)`` without checking for
``None``: ``spec=None`` renders canonical bytes, so a probe arm stamped with a
``probe_type`` would then be a fabricated measurement rather than a missing one.
``require_variant_spec`` is the door for callers that must have a variant; it
raises with the recorded reason attached.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Optional

from lmopt.prompts import LIVE_SET, load_set
from lmopt.prompting.spec import (
    DARKHEX_SPEC,
    GOOFSPIEL_SPEC,
    KUHN_SPEC,
    LEDUC_SPEC,
    LIARSDICE_SPEC,
    OPTION_ORDER_REVERSED,
    GamePromptSpec,
)

# The five game families, and the two probe types this module serves.
GAME_KEYS = ("kuhn", "leduc", "goofspiel", "liarsdice", "darkhex")
PROBE_TYPES = ("relabel", "payoff_scale")

# Probe types that vary prompt content but measure no property: their result is a diagnostic
# table, never a scoreboard cell. They live in their own tuple, not in ``PROBE_TYPES``, because
# ``PROBE_TYPES`` is what the P1/P4 acceptance table iterates, and a diagnostic must never add a
# row to a table that feeds a cell.
OPTION_ORDER_PROBE_TYPE = "option_order_reversed"
DIAGNOSTIC_PROBE_TYPES = (OPTION_ORDER_PROBE_TYPE,)

# The substitutions each variant makes are data in the live prompt set:
# ``variants/relabel.json`` and ``variants/payoff_scale.json`` hold, per game, what the
# variant changes; the tables below are read from them and the builders further down apply
# them. The caveats and absences stay here: they are statements about the variants, not text
# a model reads.
_RELABEL = load_set(LIVE_SET).variant("relabel")["games"]
_PAYOFF_SCALE = load_set(LIVE_SET).variant("payoff_scale")

# Values for the row-level ``prompt_variant`` column. Named
# per probe type, not per game: a row already carries its game, and the paired
# scorer joins on (game, info-set, prompt_variant).
RELABEL_VARIANT_NAME = "relabel_v1"
PAYOFF_SCALE_FACTOR: int = _PAYOFF_SCALE["factor"]
PAYOFF_SCALE_VARIANT_NAME = f"payoff_scale_x{PAYOFF_SCALE_FACTOR}"
OPTION_ORDER_VARIANT_NAME = "option_order_reversed_v1"

# The ``prompt_variant`` value of an arm that renders canonical content. Named
# rather than typed at each site: ``MatchupSpec``, ``run_match``, the schema
# comment and the paired scorer all have to agree on the literal.
BASE_VARIANT_NAME = "base"

# Probe types that do not vary prompt content, and so must carry the base
# variant. Declared here rather than derived with a ``.get(..., "base")``
# default so that a probe type added to ``logging.schema.PROBE_TYPE_DOMAIN``
# without a decision about its prompt content raises in
# ``expected_variant_name`` instead of silently becoming a base arm
# (no silent fallbacks). ``offpath_drive`` drives the
# match into a chosen subgame and leaves the rendering untouched.
NON_PROMPT_PROBE_TYPES = frozenset({"none", "offpath_drive"})


@dataclass(frozen=True)
class VariantAbsence:
    """Why one (game, probe_type) pair has no variant, as reachable data.

    ``reason`` states the ESTIMAND problem (why building one would measure the
    wrong thing, or why there is nothing to vary). ``would_need`` states what
    would have to become true for the variant to exist, so a later reader can
    act on it instead of rediscovering the argument. The acceptance table
    prints both, per game, per property.
    """

    game: str
    probe_type: str
    reason: str
    would_need: str


@dataclass(frozen=True)
class VariantCaveat:
    """What a variant that does exist fails to cover, as reachable data.

    Same role as ``VariantAbsence`` for the present-but-partial case: an
    admissibility statement computed over what was measured, not a footnote.
    ``surfaces`` names the concrete prose the caveat is about, so a test can
    check the caveat against the rendering rather than trusting its wording.
    """

    game: str
    probe_type: str
    caveat: str
    surfaces: tuple[str, ...]


# ---------------------------------------------------------------------------
# Relabel (P1)
# ---------------------------------------------------------------------------

# New action-label literals, keyed by the symbolic vocabulary name. Chosen to
# collide with nothing the parsers already accept: not a canonical token
# (FOLD/CALL/RAISE/CHECK/BET/CHALLENGE) and not an existing ``parse_synonyms``
# entry (kuhn/leduc "PASS", liarsdice "LIAR"). A collision with a synonym is not
# cosmetic: it makes the parser resolve the new label through the old vocabulary
# entry, which is exactly what the label-agreement fidelity gate exists to catch.
_RELABEL_TOKENS: dict[str, dict[str, str]] = {
    game: dict(entry["tokens"]) for game, entry in _RELABEL.items() if "tokens" in entry
}

# Renamed parse-only synonyms. The variant keeps the same synonym structure as
# the canonical spec (one extra accepted literal for the same symbolic action),
# because parser permissiveness is a property of the arm: dropping the synonym
# would change the fallback rate between base and probe arms and confound the
# comparison with a parsing difference. The literals are renamed too, so the
# variant carries no canonical vocabulary anywhere.
_RELABEL_SYNONYMS: dict[str, dict[str, list[str]]] = {
    game: {sym: list(lits) for sym, lits in entry["synonyms"].items()}
    for game, entry in _RELABEL.items()
    if "synonyms" in entry
}

# Dark Hex's relabel is a cyclic rotation of the column-label alphabet by 15
# places (A->P, B->Q, ...), a column-label permutation. A rotation rather than a reversal, deliberately: reversal would
# put the labels out of left-to-right alphabetical order, which violates a prior
# the prompt never states and would confound label invariance with "does the
# model trust the rendered header over its own reading habit". A rotation keeps
# every geometric fact identical and changes only the letters.
DARKHEX_COLUMN_LABELS: str = _RELABEL["darkhex"]["column_labels"]

# Dark Hex prose that names a column label literally rather than through the
# generated enumeration. Both must move with the labels or the variant would
# ship a stale canonical letter ("e.g. A1" offers a label the variant board does
# not have). Each phrase's expected occurrence count is asserted at build time,
# so a later edit to the canonical prose fails loudly here instead of silently
# leaving the variant stale.
_DARKHEX_LABEL_PHRASES: dict[str, str] = dict(_RELABEL["darkhex"]["phrases"])

# Number words for the bid cards. They carry their own order, so the variant needs no
# sentence explaining it. Six covers every registered deck size.
GOOFSPIEL_CARD_LABELS: dict[int, str] = {
    int(card): word for card, word in _RELABEL["goofspiel"]["card_labels"].items()
}

# Goofspiel prose that names the card range by NUMBER rather than through the
# generated enumeration. Both must move with the labels or the variant would
# offer word labels while its rules described a deck of digits. Keys are the
# canonical spans as they appear in the templates (``{num_cards}`` included);
# the replacements introduce ``{first_card}``/``{last_card}``, which
# ``goofspiel.make_fns`` supplies at render time for both arms.
_GOOFSPIEL_CARD_RANGE_PHRASES: dict[str, str] = dict(_RELABEL["goofspiel"]["phrases"])

# ---------------------------------------------------------------------------
# Payoff scale (P4)
# ---------------------------------------------------------------------------

# Exact prose spans scaled, per game, with the count each must appear.
# Checked against prompt version 7's templates: every other
# number in every game's templates is a seat id, a round number, a board
# dimension, a die face or a card identity, none of which is a payoff.
# Goofspiel's entry is deliberately empty: every magnitude it renders is a prize
# value or a score computed from game state, so it scales through
# ``display_scale`` alone. Substring safety does not rest on the ordering of
# this dict -- ``_build_payoff_scale_spec`` substitutes every phrase in one
# pass, longest first, so no replacement can be rescaled by a later one.
_PAYOFF_SCALE_PHRASES: dict[str, dict[str, tuple[str, int]]] = {
    game: {
        phrase: (replacement, count)
        for phrase, (replacement, count) in entry["phrases"].items()
    }
    for game, entry in _PAYOFF_SCALE["games"].items()
}

# Games whose rendered-from-state magnitudes must scale too. Leduc renders the
# pot and both chip stacks from the info-state string; Goofspiel renders the
# prize sequence, the current prize and both scores. Kuhn, Liar's Dice and Dark
# Hex render no magnitude outside their templates, so the whitelist alone makes
# their scaling total.
_PAYOFF_SCALE_DISPLAY_GAMES: frozenset[str] = frozenset(
    game
    for game, entry in _PAYOFF_SCALE["games"].items()
    if entry["scale_rendered_magnitudes"]
)


def _replace_bare_token(text: str, old: str, new: str) -> str:
    """Replace whole-word `old` with `new`, never inside a ``{placeholder}``.

    The brace guard is load-bearing, not defensive. ``instruction``'s options
    lines are ``str.format`` templates keyed by the symbolic vocabulary name
    (``"{FOLD} (give up the pot) or {CALL} (...)"``), filled from
    ``action_vocabulary`` at render time. Renaming a placeholder would raise
    ``KeyError`` at the first render, because the game module passes
    ``FOLD=vocab["FOLD"]``. The rename belongs in the vocabulary values; the
    only prose this touches is the fixed rules text, which hand-types the
    canonical words.
    """
    return re.sub(r"(?<![{\w])" + re.escape(old) + r"(?![}\w])", new, text)


def _rename_all(text: str, literal_renames: dict[str, str]) -> str:
    for old, new in literal_renames.items():
        text = _replace_bare_token(text, old, new)
    return text


def _build_relabel_spec(spec: GamePromptSpec) -> GamePromptSpec:
    """The game's spec with its action vocabulary renamed everywhere it is stated.

    Everywhere means two places, and missing either one is the defect this
    function exists to prevent: the ``action_vocabulary`` values (which reach
    ``action_suffix``, ``action_labels``, ``parse_action`` and the options
    templates), and the fixed ``rules`` prose, which hand-types the canonical
    words in a copy no template fills (spec.py's KUHN/LIARSDICE rules lines).
    A relabel that renamed only the vocabulary would ship a prompt whose rules
    paragraph still named CHECK and BET while its offered labels said wait and
    STAKE.

    Raises if any canonical literal survives as a bare token in the result, so
    a later edit to the canonical prose cannot silently produce a stale variant.
    """
    game = spec.game
    renames = _RELABEL_TOKENS[game]
    canonical_vocab = spec.action_vocabulary
    literal_renames = {canonical_vocab[sym]: new for sym, new in renames.items()}

    vocab: dict[str, object] = {
        sym: renames.get(sym, lit)
        for sym, lit in canonical_vocab.items()
        if sym != "parse_synonyms"
    }
    vocab["parse_synonyms"] = {
        sym: list(lits) for sym, lits in _RELABEL_SYNONYMS[game].items()
    }

    instruction = {
        key: _rename_all(text, literal_renames)
        for key, text in spec.instruction.items()
    }
    belief_question = _rename_all(spec.belief_question, literal_renames)

    variant = dataclasses.replace(
        spec,
        instruction=instruction,
        action_vocabulary=vocab,
        belief_question=belief_question,
    )
    _reject_surviving_canonical_tokens(variant, literal_renames)
    _reject_ambiguous_vocabulary(variant)
    return variant


def _reject_surviving_canonical_tokens(
    variant: GamePromptSpec, literal_renames: dict[str, str]
) -> None:
    """No canonical literal may survive as a bare token in the variant's prose."""
    texts = {**variant.instruction, "belief_question": variant.belief_question}
    for key, text in texts.items():
        for old in literal_renames:
            if re.search(r"(?<![{\w])" + re.escape(old) + r"(?![}\w])", text):
                raise ValueError(
                    f"{variant.game} relabel variant still states the canonical "
                    f"token {old!r} in instruction key {key!r}; the rename would "
                    f"ship a prompt whose rules prose contradicts its own labels"
                )


def _reject_ambiguous_vocabulary(variant: GamePromptSpec) -> None:
    """Every accepted literal (label or synonym) must denote one action.

    A duplicate literal makes ``parse_action`` resolve a label to the wrong
    action, and makes an echo of ``action_suffix`` parse as an answer (two
    identical labels are one distinct label, so multi-label rejection does not
    fire). Both are caught by the standing fidelity gates over the full tree;
    this is the cheap construction-time check that keeps a broken variant from
    reaching them in the first place.
    """
    vocab = variant.action_vocabulary
    literals: list[str] = [v for k, v in vocab.items() if k != "parse_synonyms"]
    for lits in vocab.get("parse_synonyms", {}).values():
        literals.extend(lits)
    duplicates = {lit for lit in literals if literals.count(lit) > 1}
    if duplicates:
        raise ValueError(
            f"{variant.game} variant vocabulary is ambiguous: {sorted(duplicates)} "
            f"denotes more than one action"
        )


def _build_darkhex_relabel_spec(spec: GamePromptSpec) -> GamePromptSpec:
    """Dark Hex's relabel: the column-label alphabet rotated, prose included.

    The label enumeration is generated by ``darkhex.make_fns`` from the board
    shape (spec.py's module docstring keeps generated enumerations in code), so
    the spec carries the alphabet as ``action_vocabulary["column_labels"]`` and
    the factory reads it. The two prose spans that name a column letter
    literally are rewritten here.
    """
    instruction = dict(spec.instruction)
    for phrase, replacement in _DARKHEX_LABEL_PHRASES.items():
        hits = [key for key, text in instruction.items() if phrase in text]
        found = sum(text.count(phrase) for text in instruction.values())
        if found != 1:
            raise ValueError(
                f"darkhex relabel: expected exactly one occurrence of {phrase!r} "
                f"across the instruction templates, found {found} in keys {hits}; "
                f"the canonical prose changed and the variant would ship a stale "
                f"column letter"
            )
        key = hits[0]
        instruction[key] = instruction[key].replace(phrase, replacement)

    # copy.deepcopy, not dict(...): `corner_names` is a nested dict, and a
    # shallow copy would leave the variant's entry aliasing the canonical spec's
    # object. That would be safe only by a guard in another module (darkhex.make_fns
    # wraps it in a MappingProxyType), so a mutation of the
    # variant's corner names would reach canonical rendering the moment
    # that wrap were removed. A variant must not depend on a distant guard to
    # avoid corrupting the spec it was derived from.
    vocab = copy.deepcopy(dict(spec.action_vocabulary))
    vocab["column_labels"] = DARKHEX_COLUMN_LABELS
    return dataclasses.replace(spec, instruction=instruction, action_vocabulary=vocab)


def _build_goofspiel_relabel_spec(spec: GamePromptSpec) -> GamePromptSpec:
    """
    Goofspiel's relabel: the bid cards spelled as number words, prose included. The label
    enumeration is generated by ``goofspiel.make_fns`` from the deck size (spec.py's module
    docstring keeps generated enumerations in code), so the spec carries the alphabet as
    ``action_vocabulary["card_labels"]`` and the factory reads it. The two prose spans that
    name the card range by number are rewritten here, each guarded by an exactly-one-site
    check so a drift in the canonical wording raises instead of shipping a variant whose
    rules still describe a deck of digits
    :param spec: the canonical Goofspiel spec (GamePromptSpec)
    :returns: the relabel variant spec (GamePromptSpec)
    """
    instruction = dict(spec.instruction)
    belief_question = spec.belief_question
    for phrase, replacement in _GOOFSPIEL_CARD_RANGE_PHRASES.items():
        found = sum(text.count(phrase) for text in instruction.values())
        found += belief_question.count(phrase)
        if found != 1:
            raise ValueError(
                f"goofspiel relabel: expected exactly one occurrence of {phrase!r} "
                f"across the instruction templates and the belief question, found "
                f"{found}; the canonical prose changed and the variant would ship "
                f"a rules paragraph that still names the cards by number"
            )
        instruction = {
            key: text.replace(phrase, replacement) for key, text in instruction.items()
        }
        belief_question = belief_question.replace(phrase, replacement)

    vocab = copy.deepcopy(dict(spec.action_vocabulary))
    vocab["card_labels"] = dict(GOOFSPIEL_CARD_LABELS)
    return dataclasses.replace(
        spec,
        instruction=instruction,
        belief_question=belief_question,
        action_vocabulary=vocab,
    )


def _build_payoff_scale_spec(spec: GamePromptSpec) -> GamePromptSpec:
    """The game's spec with every stake magnitude in its templates scaled.

    Whitelist, not judgement: only the exact phrases in
    ``_PAYOFF_SCALE_PHRASES`` move, and each must appear exactly the recorded
    number of times or this raises. Anything else in the prose is out of the
    variant's lane by construction, which is what lets the test assert byte
    equality against the base rendering with the whitelist applied.

    The substitution is one simultaneous pass over an alternation ordered
    longest-phrase-first, not a sequence of ``str.replace`` calls: "1 chip" is a
    substring of "1 chips" and "2 chips" of "12 chips", so a second pass over a
    string an earlier pass has already rewritten is a live rescaling hazard
    rather than a hypothetical one.

    Magnitudes the game module computes from state rather than from prose are
    carried by ``display_scale`` instead; see ``_PAYOFF_SCALE_DISPLAY_GAMES``.
    """
    phrases = _PAYOFF_SCALE_PHRASES[spec.game]
    instruction = dict(spec.instruction)
    for phrase, (_replacement, expected_count) in phrases.items():
        found = sum(text.count(phrase) for text in instruction.values())
        if found != expected_count:
            raise ValueError(
                f"{spec.game} payoff_scale: expected {expected_count} occurrences "
                f"of {phrase!r} across the instruction templates, found {found}; "
                f"the canonical prose changed and the scaling is no longer total"
            )
    if phrases:
        pattern = re.compile(
            "|".join(
                re.escape(phrase) for phrase in sorted(phrases, key=len, reverse=True)
            )
        )
        instruction = {
            key: pattern.sub(lambda m: phrases[m.group(0)][0], text)
            for key, text in instruction.items()
        }
    display_scale = (
        PAYOFF_SCALE_FACTOR if spec.game in _PAYOFF_SCALE_DISPLAY_GAMES else 1
    )
    return dataclasses.replace(
        spec,
        instruction=instruction,
        display_scale=display_scale,
        memory_payoff_scale=PAYOFF_SCALE_FACTOR,
    )


# ---------------------------------------------------------------------------
# Option order (diagnostic)
# ---------------------------------------------------------------------------

# The option-list prose that lives in the spec as template text, per game, as (canonical
# literal, reversed literal). The canonical literal is checked for exact equality before the
# swap, so an edit to the canonical prose raises here instead of shipping a variant that
# reorders the old wording. Games whose option lists are built entirely in code (Goofspiel's
# hand line, Dark Hex's move list) carry no entry: ``option_order`` reaches them through the
# helpers in ``spec.py``. The reversed text uses exactly the canonical words.
_OPTION_ORDER_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "kuhn": {
        "options_facing_bet": (
            "{FOLD} (give up the pot) or {CALL} (match the bet and go to showdown)",
            "{CALL} (match the bet and go to showdown) or {FOLD} (give up the pot)",
        ),
        "options_first": (
            "{CHECK} (free, pass to opponent) or {BET} (stake 1 chip)",
            "{BET} (stake 1 chip) or {CHECK} (free, pass to opponent)",
        ),
        "options_after_check": (
            "{CHECK} (free, pass back) or {BET} (stake 1 chip)",
            "{BET} (stake 1 chip) or {CHECK} (free, pass back)",
        ),
    },
    "leduc": {
        "options_facing_raise": (
            "{FOLD} (give up), {CALL} (match the bet), or {RAISE} (re-raise)",
            "{RAISE} (re-raise), {CALL} (match the bet), or {FOLD} (give up)",
        ),
        "options_facing_final": (
            "{FOLD} (give up) or {CALL} (match the bet)",
            "{CALL} (match the bet) or {FOLD} (give up)",
        ),
        "options_first": (
            "{CHECK} (free) or {BET} (2 chips in round 1, 4 chips in round 2)",
            "{BET} (2 chips in round 1, 4 chips in round 2) or {CHECK} (free)",
        ),
    },
    "liarsdice": {
        "situation_facing_bid": (
            "The current bid is: at least {last_count} dice show {last_face}. "
            "You can raise this bid or {CHALLENGE} it.",
            "The current bid is: at least {last_count} dice show {last_face}. "
            "You can {CHALLENGE} this bid or raise it.",
        ),
    },
    "goofspiel": {},
    "darkhex": {},
}


def _words_of(text: str) -> list[str]:
    """
    :param text: template prose (str)
    :returns: its words, punctuation dropped, sorted, so two orderings of the same words
        compare equal (list[str])
    """
    return sorted(re.findall(r"[A-Za-z0-9_{}]+", text))


def _build_option_order_spec(spec: GamePromptSpec) -> GamePromptSpec:
    """
    the game's spec with its option-list prose reversed and ``option_order`` set, so the code-built
    lists reverse with it. Every rewritten template is checked twice: the canonical text must
    equal the expected literal exactly, and the reversed text must use exactly the same words
    :param spec: the canonical spec of one game (GamePromptSpec)
    :returns: the reversed-order variant of that spec (GamePromptSpec)
    :raises KeyError: when the game has no entry in ``_OPTION_ORDER_TEMPLATES``
    :raises ValueError: when a canonical template drifted or a reversal changed a word
    """
    entries = _OPTION_ORDER_TEMPLATES[spec.game]
    instruction = dict(spec.instruction)
    for key, (canonical, reversed_text) in entries.items():
        if instruction.get(key) != canonical:
            raise ValueError(
                f"{spec.game} option_order: instruction key {key!r} no longer reads "
                f"{canonical!r}; the canonical prose changed and the variant would reorder "
                f"the old wording"
            )
        if (
            _words_of(canonical) != _words_of(reversed_text)
            or canonical == reversed_text
        ):
            raise ValueError(
                f"{spec.game} option_order: the reversal of {key!r} must use the same words "
                f"as the canonical text in a different order"
            )
        instruction[key] = reversed_text
    return dataclasses.replace(
        spec, instruction=instruction, option_order=OPTION_ORDER_REVERSED
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

RELABEL_SPECS: dict[str, GamePromptSpec] = {
    "kuhn": _build_relabel_spec(KUHN_SPEC),
    "leduc": _build_relabel_spec(LEDUC_SPEC),
    "liarsdice": _build_relabel_spec(LIARSDICE_SPEC),
    "darkhex": _build_darkhex_relabel_spec(DARKHEX_SPEC),
    "goofspiel": _build_goofspiel_relabel_spec(GOOFSPIEL_SPEC),
}

PAYOFF_SCALE_SPECS: dict[str, GamePromptSpec] = {
    game: _build_payoff_scale_spec(spec)
    for game, spec in (
        ("kuhn", KUHN_SPEC),
        ("leduc", LEDUC_SPEC),
        ("goofspiel", GOOFSPIEL_SPEC),
        ("liarsdice", LIARSDICE_SPEC),
        ("darkhex", DARKHEX_SPEC),
    )
}

OPTION_ORDER_SPECS: dict[str, GamePromptSpec] = {
    game: _build_option_order_spec(spec)
    for game, spec in (
        ("kuhn", KUHN_SPEC),
        ("leduc", LEDUC_SPEC),
        ("goofspiel", GOOFSPIEL_SPEC),
        ("liarsdice", LIARSDICE_SPEC),
        ("darkhex", DARKHEX_SPEC),
    )
}

_SPECS_BY_PROBE: dict[str, dict[str, GamePromptSpec]] = {
    "relabel": RELABEL_SPECS,
    "payoff_scale": PAYOFF_SCALE_SPECS,
    OPTION_ORDER_PROBE_TYPE: OPTION_ORDER_SPECS,
}

# Public: the row-level ``prompt_variant`` a probe arm of each
# prompt-varying type must carry. Game-independent by construction -- one
# relabel scheme serves every game that has one, and a row already carries its
# game.
VARIANT_NAME_BY_PROBE: dict[str, str] = {
    "relabel": RELABEL_VARIANT_NAME,
    "payoff_scale": PAYOFF_SCALE_VARIANT_NAME,
    OPTION_ORDER_PROBE_TYPE: OPTION_ORDER_VARIANT_NAME,
}

# Every legal value of the ``prompt_variant`` column, base included. This is
# what MatchupSpec validates against; it is imported there, never re-typed, so
# a variant added here cannot leave the orchestration layer checking a stale
# copy.
PROMPT_VARIANT_NAMES: frozenset[str] = frozenset(
    {BASE_VARIANT_NAME, *VARIANT_NAME_BY_PROBE.values()}
)


# Empty under prompt version 7, and kept as a live registry rather than deleted.
# Entries are statements about prompt content: version 7 states a payoff in
# every game, and the goofspiel relabel spells the bid cards as number words,
# which preserve the bid order. A variant withdrawn later re-enters here as
# data, so ``coverage_rows`` prints a reason rather than a blank.
VARIANT_ABSENT_REASONS: dict[str, dict[str, VariantAbsence]] = {}

VARIANT_CAVEATS: dict[str, dict[str, VariantCaveat]] = {
    "kuhn": {
        "relabel": VariantCaveat(
            game="kuhn",
            probe_type="relabel",
            caveat="The rules prose still names the action in lowercase English "
            "('If facing a bet, ...', 'match the bet'). Only the uppercase LABEL "
            "tokens are renamed. These lowercase uses are DESCRIPTIONS of the "
            "situation, not offers of a label: rewriting 'facing a bet' as "
            "'facing a STAKE' would read as offering a label where none is "
            "offered, and the sentence would then name a token the prompt does "
            "not accept at that point. The rename stays where the prompt is "
            "telling the model what to say.",
            surfaces=("BET",),
        ),
    },
    "leduc": {
        "relabel": VariantCaveat(
            game="leduc",
            probe_type="relabel",
            caveat="Leduc's rules paragraph names every action in lowercase "
            "English, and the relabel does not rewrite it. The reason is "
            "paragraph-internal consistency, NOT that the words resist "
            "substitution: one sentence is a bare list that would substitute "
            "mechanically ('Betting: check (free), bet, call (match a bet), raise "
            "(re-raise), or fold (concede)'), but the two sentences flanking it in "
            "that same paragraph use conjugated forms that would not ('both players "
            "bet with only their private card', 'Maximum two bets/raises per "
            "round'). A third such use, 'how they have bet this hand', sits in the "
            "belief question rather than the rules paragraph, so it is a second "
            "instance of the problem and not part of the paragraph argument. "
            "Rewriting the list alone would leave the paragraph "
            "describing one game in two vocabularies, which is worse than "
            "describing it consistently in the canonical one. The offered labels "
            "and their inline glosses carry the mapping ('YIELD (give up), MATCH "
            "(match the bet), or PUSH (re-raise)'), so the variant is answerable, "
            "but a leduc P1 number is measured against a prompt that describes the "
            "mechanics in canonical poker words.",
            surfaces=("BET", "CALL", "CHECK", "FOLD", "RAISE"),
        )
    },
    "goofspiel": {
        "relabel": VariantCaveat(
            game="goofspiel",
            probe_type="relabel",
            caveat="Only the BID CARDS are relabelled. Prize values and both "
            "scores stay digits, by design and not by omission: a bid card is "
            "an action identity, a prize value is a payoff, and relabelling "
            "both would make P1's relabel run and P4's payoff-scale run vary "
            "the same surface. The consequence a reader has to carry is that "
            "the variant's prompt mixes two notations for what is, in the "
            "game's own terms, one deck of cards: the model is asked to bid "
            "THREE for the 3-point card. The order the words encode is the "
            "order the digits encoded, so the probe still measures label "
            "invariance rather than a change of game, but it measures it "
            "against a prompt no canonical arm renders.",
            surfaces=("bid card labels", "round history bid labels"),
        )
    },
    "darkhex": {
        "payoff_scale": VariantCaveat(
            game="darkhex",
            probe_type="payoff_scale",
            caveat="Dark Hex states exactly one magnitude, the win/lose stake "
            "of 1 point, and the variant scales that one phrase. Nothing else "
            "in the prompt is a payoff: the remaining numbers are board "
            "coordinates and the move count. The scaling is therefore TOTAL "
            "and also minimal, which makes this the weakest of the five "
            "payoff-scale runs: a model insensitive to the magnitude has only "
            "one number to have been insensitive to, so a null result here "
            "carries less evidence than the same null in a game that states "
            "an ante, a bet size and a settlement separately.",
            surfaces=("1 point",),
        )
    },
}


def _check_key(game_key: str, probe_type: str) -> None:
    if game_key not in GAME_KEYS:
        raise ValueError(
            f"unknown game key {game_key!r}; expected one of {sorted(GAME_KEYS)}"
        )
    known = (*PROBE_TYPES, *DIAGNOSTIC_PROBE_TYPES)
    if probe_type not in known:
        raise ValueError(
            f"unknown probe type {probe_type!r}; expected one of {sorted(known)}"
        )


def variant_spec(game_key: str, probe_type: str) -> Optional[GamePromptSpec]:
    """The variant spec for one (game, probe_type), or ``None`` if none exists.

    ``None`` always means "recorded absence": ``variant_absence(game_key,
    probe_type)`` then returns the ``VariantAbsence`` explaining it, and the
    registry's own test asserts the two are exact complements over all
    ``len(GAME_KEYS) * len(PROBE_TYPES)`` pairs. An unknown game or probe type
    raises instead, because "this game has no relabel variant" and "this is not
    one of our games" must not be the same answer to a caller.

    Callers that intend to RUN a probe arm should use ``require_variant_spec``:
    passing this function's ``None`` into ``get_game_fns_for_game(game,
    spec=...)`` silently renders canonical bytes.
    """
    _check_key(game_key, probe_type)
    return _SPECS_BY_PROBE[probe_type].get(game_key)


def variant_absence(game_key: str, probe_type: str) -> Optional[VariantAbsence]:
    """The recorded absence for one (game, probe_type), or ``None`` if a variant exists."""
    _check_key(game_key, probe_type)
    return VARIANT_ABSENT_REASONS.get(game_key, {}).get(probe_type)


def variant_caveat(game_key: str, probe_type: str) -> Optional[VariantCaveat]:
    """What the existing variant does not cover, or ``None`` if nothing is recorded."""
    _check_key(game_key, probe_type)
    return VARIANT_CAVEATS.get(game_key, {}).get(probe_type)


def variant_name(game_key: str, probe_type: str) -> Optional[str]:
    """The ``prompt_variant`` column value for this probe arm, or ``None`` if absent."""
    if variant_spec(game_key, probe_type) is None:
        return None
    return VARIANT_NAME_BY_PROBE[probe_type]


def expected_variant_name(probe_type: str) -> str:
    """The ``prompt_variant`` an arm of this probe type must carry.

    Game-independent, and total over ``logging.schema.PROBE_TYPE_DOMAIN`` by a
    standing test rather than by a permissive default: a probe type that is
    neither a registered prompt variant nor recorded in
    ``NON_PROMPT_PROBE_TYPES`` raises here. The alternative (defaulting to
    ``"base"``) would let a future prompt-varying probe collect an arm whose
    rows claim canonical content.

    Callers must validate ``probe_type`` against ``PROBE_TYPE_DOMAIN`` first
    (``MatchupSpec.__post_init__`` does, in that order): this function's raise
    is the second line of defence and names the classification, not the domain.
    """
    if probe_type in VARIANT_NAME_BY_PROBE:
        return VARIANT_NAME_BY_PROBE[probe_type]
    if probe_type in NON_PROMPT_PROBE_TYPES:
        return BASE_VARIANT_NAME
    raise ValueError(
        f"unclassified probe type {probe_type!r}: it is neither a prompt-variant "
        f"probe ({sorted(VARIANT_NAME_BY_PROBE)}) nor recorded as leaving the "
        f"prompt canonical ({sorted(NON_PROMPT_PROBE_TYPES)}). A new probe type "
        f"needs a decision about its prompt content before an arm can be labelled."
    )


def require_variant_spec(game_key: str, probe_type: str) -> GamePromptSpec:
    """``variant_spec`` for callers that must have one; raises the recorded reason.

    No silent fallbacks. This is the door a runner or sweep uses, so
    that "this game has no P1 variant" surfaces as a refusal to launch the arm,
    never as an arm that renders canonical bytes under a probe label.
    """
    spec = variant_spec(game_key, probe_type)
    if spec is None:
        absence = variant_absence(game_key, probe_type)
        if absence is None:
            raise ValueError(
                f"no {probe_type!r} variant exists for {game_key!r} and no absence is "
                f"recorded for it in VARIANT_ABSENT_REASONS; add the variant to its "
                f"registry or record why it is absent"
            )
        raise ValueError(
            f"no {probe_type!r} variant exists for {game_key!r}: {absence.reason} "
            f"What it would need: {absence.would_need}"
        )
    return spec


def coverage_rows() -> list[dict]:
    """One row per (game, probe_type) for the per-game acceptance table.

    Emitted for every pair, present or absent, so a property row cannot report
    "measurable" on the strength of the games that happen to have a variant.
    Keys: game, probe_type, available, variant_name,
    reason, would_need, caveat.
    """
    rows = []
    for game in GAME_KEYS:
        for probe in PROBE_TYPES:
            absence = variant_absence(game, probe)
            caveat = variant_caveat(game, probe)
            rows.append(
                {
                    "game": game,
                    "probe_type": probe,
                    "available": variant_spec(game, probe) is not None,
                    "variant_name": variant_name(game, probe),
                    "reason": absence.reason if absence else None,
                    "would_need": absence.would_need if absence else None,
                    "caveat": caveat.caveat if caveat else None,
                }
            )
    return rows


__all__ = [
    "GAME_KEYS",
    "PROBE_TYPES",
    "DIAGNOSTIC_PROBE_TYPES",
    "OPTION_ORDER_PROBE_TYPE",
    "OPTION_ORDER_VARIANT_NAME",
    "OPTION_ORDER_SPECS",
    "RELABEL_VARIANT_NAME",
    "PAYOFF_SCALE_VARIANT_NAME",
    "PAYOFF_SCALE_FACTOR",
    "BASE_VARIANT_NAME",
    "NON_PROMPT_PROBE_TYPES",
    "VARIANT_NAME_BY_PROBE",
    "PROMPT_VARIANT_NAMES",
    "expected_variant_name",
    "DARKHEX_COLUMN_LABELS",
    "GOOFSPIEL_CARD_LABELS",
    "RELABEL_SPECS",
    "PAYOFF_SCALE_SPECS",
    "VARIANT_ABSENT_REASONS",
    "VARIANT_CAVEATS",
    "VariantAbsence",
    "VariantCaveat",
    "variant_spec",
    "variant_absence",
    "variant_caveat",
    "variant_name",
    "require_variant_spec",
    "coverage_rows",
]
