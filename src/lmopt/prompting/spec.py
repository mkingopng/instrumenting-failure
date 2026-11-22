"""Per-game prompt content: instruction wording, action vocabulary, belief
question, and belief exemplar.

The content itself is data: the live prompt set under ``lmopt/prompts/sets/`` holds every
string, and this module reads it into one ``GamePromptSpec`` per game. Rendering logic --
info-set string to text, e.g. Leduc's betting-history description or Dark
Hex's board/discovery-state rendering -- stays in each game's own module;
this module holds only the content objects that logic renders around, plus the
parsing helpers every game shares.

Each game's ``action_vocabulary`` is the single source consumed by that
game's ``action_suffix``, ``action_labels``, and ``parse_action``: this
module already unifies ``action_suffix``/``action_labels``, and also sources
``parse_action``'s literal regex tokens from the same dict instead of a
separately hand-typed pattern.

For Dark Hex and Goofspiel (depth variants via ``make_fns``), the label
enumeration is generated from board/deck dimensions and stays code (module
docstrings of ``darkhex.py``/``goofspiel.py``); only the constant, dimension-
independent content -- instruction wording, belief question prose, the
belief exemplar's numeric format, and (Dark Hex) the fixed 2x2 corner-name
vocabulary -- lives here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from lmopt.prompts import LIVE_SET, load_set

_SPEC_HASH_VERSION = 1

# The two orders a prompt can list its legal options in. The canonical order is what every
# collected arm rendered; the reversed order is the same words in the opposite order, in every
# place the prompt lists the options.
OPTION_ORDER_CANONICAL = "canonical"
OPTION_ORDER_REVERSED = "reversed"
OPTION_ORDERS = (OPTION_ORDER_CANONICAL, OPTION_ORDER_REVERSED)


def count_matching_labels(text: str, labels: Optional[Iterable[str]]) -> int:
    """Count how many distinct strings among `labels` appear as whole-word
    (regex ``\\b...\\b``) matches in `text` (case-insensitive).

    Action parser hardening: a
    completion naming multiple distinct action labels -- exactly what an
    echo of ``action_suffix`` does, since the suffix typically names every
    legal label or an exemplar literal -- must not parse as a bare answer.
    This is the one place that rule lives; every game's ``parse_action``
    calls it against the labels ``action_labels(info_str)`` currently offers
    at that info-set (never the full, possibly branch-inapplicable,
    vocabulary), so the rule is exactly as permissive as the live prompt: a
    bare single label still parses (only 1 match), and a single label
    embedded in short prose ("I choose CHECK") still parses too (still only
    1 distinct label match) -- multi-label detection fires only on 2+
    distinct labels, not on extra prose around one.

    Multi-word labels (e.g. liarsdice's "1 4" bid literals) work the same
    way: the escaped label string can itself contain a literal space, and
    ``\\b`` still anchors correctly on either side of the whole phrase.

    Contract: ``labels=None`` returns 0, not a
    ``TypeError`` -- "no labels to check against" means "nothing can be
    multi-label", and callers (e.g. a game with
    no fixed word vocabulary) should not need a truthiness guard before
    calling this.
    """
    if not labels:
        return 0
    t = text.strip().upper()
    return sum(
        1
        for lab in {lab.strip().upper() for lab in labels}
        if re.search(r"\b" + re.escape(lab) + r"\b", t)
    )


def has_multiple_labels(text: str, labels: Optional[Iterable[str]]) -> bool:
    """True iff `text` names 2+ distinct strings from `labels` (see
    `count_matching_labels`). The single boolean check every game's
    `parse_action` applies before its normal branch logic. ``labels=None``
    (or empty) is False, never a ``TypeError`` -- see
    `count_matching_labels`'s contract note."""
    return count_matching_labels(text, labels) >= 2


def _opens_an_unterminated_object(text: str) -> bool:
    """True iff `text` opens a JSON object that never closes.

    That is exactly the shape of a truncated completion under a schema
    protocol (agents/lm_vllm.py's REASONING_BELIEF_JSON_SCHEMA comment): the
    outer object is cut off, so nothing after it can be trusted as a
    top-level answer.

    Braces inside JSON string literals are not structural and are skipped,
    with backslash escapes honoured. A complete response whose reasoning
    string mentions '{' must not read as truncated, and a free-generation
    completion that quotes a brace in prose must keep the belief it has
    always returned.
    """
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            # Clamp: a stray '}' in prose closes nothing.
            depth = max(0, depth - 1)
    return depth > 0


# The reasoning segment must be
# identified structurally (a `<think>...</think>` delimiter), never
# positionally. A "read the final line" rule is byte-identical on
# delimiter-free data and reintroduces a 3,078-info-set liarsdice
# echo defect: its
# `action_suffix` ends "Respond with exactly one bid (e.g. '1 4' = at least
# one die shows 4).", so a positional "last line" heuristic would hand that
# parenthetical straight to `parse_action` as if it were the model's answer.
_THINK_OPEN_RE = re.compile(r"^\s*<think>(.*?)</think>(.*)\Z", re.DOTALL)
_THINK_CLOSE_ONLY_RE = re.compile(r"^(.*?)</think>(.*)\Z", re.DOTALL)


def split_reasoning(text: str) -> tuple[Optional[str], str]:
    """Split `text` into (trace, answer) on a STRUCTURAL `</think>` delimiter.

    Mirrors vLLM's own olmo3 reasoning parser: two shapes are
    recognised, both anchored on the first (non-greedy) ``</think>``.

    1. ``<think>trace</think>answer`` -- a full block at the start of `text`
       (leading whitespace before ``<think>`` tolerated, e.g. a server that
       prepends a newline). `trace` is the text between the tags.
    2. ``trace</think>answer`` -- closing tag only, no ``<think>`` opener:
       the "pre-opened template" shape some servers emit when the opening
       tag was already part of the prompt they were given. `trace` is
       everything before the first ``</think>``.

    Anything else returns ``(None, text)`` unchanged, including:
    - text with no ``</think>`` anywhere (the common, non-reasoning-model
      case);
    - an unterminated ``<think>`` (opened, never closed) -- treating the
      open tag alone as a delimiter would silently discard everything
      before an answer that never actually arrived.

    No positional fallback of any kind: see the module comment above this
    function for why (the liarsdice defect a last-line rule reopens).

    This function does not read `reasoning_content` and does not decide
    whether to run -- callers apply it only when the server did not already
    supply a structured trace (`lm_vllm.py`'s call-site composition:
    ``if trace is None: trace, answer = split_reasoning(result.content)``).
    """
    match = _THINK_OPEN_RE.match(text)
    if match is not None:
        return match.group(1), match.group(2)
    match = _THINK_CLOSE_ONLY_RE.match(text)
    if match is not None:
        return match.group(1), match.group(2)
    return None, text


# Real Qwen liarsdice `belief_failed` rows (12 at memory level M1 under prompt
# version 6) share one shape -- the model fills the last bucket with an unevaluated
# "one minus the others" expression, e.g. `"6": 1.00 - (0.25 + 0.25 + 0.15 +
# 0.15 + 0.10)`, instead of the literal `0.10`. Strict `json.loads` rejects
# the whole six-key object over this one value, discarding five perfectly
# good numbers with it. `_repair_arithmetic_belief_values` below recovers
# that case; it does not fix every parse gap (an unquoted integer key, seen
# in OLMo goofspiel samples, is a different defect, handled by
# `salvage_belief_object`).
_SAFE_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_SAFE_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_arith_eval(expr: str) -> Optional[float]:
    """`expr`'s value if it is purely +, -, *, / over numeric literals
    (optionally parenthesised) -- None on anything else (a name, a call, an
    attribute, a comparison, a string). Never `eval()`: this runs over
    untrusted LM output, and a name or call node must refuse, not execute.
    """
    try:
        node = ast.parse(expr, mode="eval").body
    except (SyntaxError, ValueError, OverflowError, RecursionError):
        return None

    def _walk(n: ast.AST) -> float:
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)
        ):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_BINOPS:
            return _SAFE_BINOPS[type(n.op)](_walk(n.left), _walk(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_UNARYOPS:
            return _SAFE_UNARYOPS[type(n.op)](_walk(n.operand))
        raise ValueError("unsafe expression node")

    try:
        return _walk(node)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        return None


# Value spans after a bare or quoted integer key only -- every game's belief
# object keys are face/rank/card/action indices, never a "reasoning" string,
# so this cannot mistake a genuine string field elsewhere for a belief value.
_NUMERIC_KV_RE = re.compile(r'("?\d+"?)\s*:\s*([-+*/.() \d]+?)\s*(?=[,}])')


def _repair_arithmetic_belief_values(group: str) -> Optional[str]:
    """Rewrite `group`'s unevaluated arithmetic values into their computed
    numeric literal, so it becomes valid JSON; None if nothing needed
    repair (the failure is real) or if any matched value fails to evaluate
    (never return a partially repaired string -- fixing 5 of 6 keys and
    silently dropping the 6th would fabricate a belief the model never
    stated).
    """
    replacements: list[tuple[str, str]] = []
    for m in _NUMERIC_KV_RE.finditer(group):
        key, raw_value = m.group(1), m.group(2)
        stripped = raw_value.strip()
        try:
            float(stripped)
            continue  # already a plain literal, nothing to repair
        except ValueError:
            pass
        evaluated = _safe_arith_eval(stripped)
        if evaluated is None:
            return None
        replacements.append((m.group(0), f"{key}: {evaluated}"))
    if not replacements:
        return None
    repaired = group
    for old, new in replacements:
        repaired = repaired.replace(old, new, 1)
    return repaired


_BRACE_FREE_GROUP_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def extract_belief_object(text: str) -> Optional[str]:
    """
    return the JSON object in `text` that is the belief, or None
    Four rules apply in order. (1) If the whole response parses as JSON and has a "belief" key,
    return that field re-serialised when it is an object, and None when it is anything else
    (null, a list, a number, a string): this is the structured-output shape, where the reasoning
    is a sibling string that may itself contain braces. (2) Otherwise, if the text opens an object
    that never closes, return None: the response is truncated, and every group in it is interior
    to an answer the model never finished. (3) Otherwise scan the brace-free groups from last to
    first and return the first one that parses as a JSON object (dict), after repairing
    arithmetic values such as `1/6`. A model that reasons writes set notation ("{1, 2, 3}")
    before its answer, which is not valid JSON and is skipped. A model that answers first and
    narrates after may leave a later non-JSON fragment, which is skipped the same way, falling
    back to the earlier real answer. (4) Otherwise None.
    Rule (2) fails closed ahead of rule (3), so a truncated schema completion whose reasoning
    quotes a well-formed object is never returned as the belief. Rule (3) takes the last group
    that parses, not the last group, because the last group is not always JSON. Rule (2) fires
    only while an outer object is open: a free-generation completion truncated after a complete
    prose group still returns that group, and only the API's `finish_reason` closes that shape.
    :param text: the belief completion, with any reasoning trace already split off (str)
    :returns: the belief as a JSON object string, or None, which the caller records as a belief failure (Optional[str])
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "belief" in obj:
            belief = obj["belief"]
            # A belief that is not an object is no belief at all. Returning it would hand every
            # per-game parser a value with no `.items()`, so it reads as unparseable here, like
            # any other bad answer.
            return json.dumps(belief) if isinstance(belief, dict) else None
    except (ValueError, TypeError):
        pass
    # Rule (2) runs after rule (1) as defence in depth: no known input changes the answer if
    # the guard runs first, because the scan skips string literals, so anything `json.loads`
    # accepts also scans to depth 0. The order documents the precedence a less string-aware
    # guard would need.
    if _opens_an_unterminated_object(text):
        return None
    groups = _BRACE_FREE_GROUP_RE.findall(text)
    for group in reversed(groups):
        try:
            candidate = json.loads(group)
        except (ValueError, TypeError):
            repaired = _repair_arithmetic_belief_values(group)
            if repaired is None:
                continue
            try:
                candidate = json.loads(repaired)
            except (ValueError, TypeError):
                continue
            if isinstance(candidate, dict):
                return repaired
            continue
        if isinstance(candidate, dict):
            return group
    return None


def load_belief_object(text: str) -> Optional[dict]:
    """
    the belief in `text` as a parsed JSON object, or None when there is none. The one entry every
    per-game `parse_belief` reads its belief through, so the extraction rules and the refusal of
    anything that is not an object live in one place
    :param text: the belief completion, with any reasoning trace already split off (str)
    :returns: the belief object, or None, which the caller records as a belief failure (Optional[dict])
    """
    raw_obj = extract_belief_object(text)
    if raw_obj is None:
        return None
    try:
        raw = json.loads(raw_obj)
    except json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def finite_float(value: Any) -> Optional[float]:
    """
    a belief value read as a finite float, or None when it cannot be one. `json.loads`
    accepts an integer of any length and the constants NaN and Infinity; `float` raises
    OverflowError on the first and passes the second through, and neither is a probability,
    so each reads as a value the model did not give
    :param value: one value of a parsed belief object (any JSON value)
    :returns: the value as a finite float, or None (Optional[float])
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


BELIEF_SALVAGE_KEY_QUOTED = "key_quoted"

_JSON_NUMBER = r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
_SALVAGE_ITEM_RE = re.compile(
    rf'\s*(?P<key>"[^"]*"|(?P<bare>0|[1-9]\d*))\s*:\s*(?P<value>{_JSON_NUMBER})\s*'
)


def salvage_belief_object(text: str) -> Optional[str]:
    """
    Recover a belief object whose only defect is a bare integer key, by
    inserting quote characters around each bare key and changing nothing else.
    Runs on text the strict path (`extract_belief_object`) rejected. Fails
    closed: an unterminated object, more than one brace group, an item that
    is not a `key: number` pair, a bare key with a leading zero, a key that
    appears twice (a duplicate would let the last value silently win), fewer
    than two pairs, a value outside [0, 1], or no bare key at all yields None. A value's text is never
    altered, evaluated or dropped, so the result differs from the model's
    group only by the inserted quote characters.
    :param text: the belief completion, already stripped of any reasoning trace (str)
    :returns: the group as a JSON object string, or None (Optional[str])
    """
    if _opens_an_unterminated_object(text):
        return None
    groups = _BRACE_FREE_GROUP_RE.findall(text)
    if len(groups) != 1:
        return None
    items = groups[0][1:-1].split(",")
    if len(items) < 2:
        return None
    rebuilt: list[str] = []
    seen_keys: set[str] = set()
    n_bare = 0
    for item in items:
        match = _SALVAGE_ITEM_RE.fullmatch(item)
        if match is None or not 0.0 <= float(match.group("value")) <= 1.0:
            return None
        key = match.group("key").strip('"')
        if key.isdigit():
            # isdigit admits superscripts, which int() refuses: such a key is no belief key, and
            # every key int() accepts is canonicalised exactly as before.
            try:
                canonical_key = str(int(key))
            except ValueError:
                return None
        else:
            canonical_key = key
        if canonical_key in seen_keys:
            return None
        seen_keys.add(canonical_key)
        if match.group("bare") is None:
            rebuilt.append(item)
            continue
        n_bare += 1
        rebuilt.append(re.sub(r"\d+", lambda m: f'"{m.group(0)}"', item, count=1))
    if n_bare == 0:
        return None
    return "{" + ",".join(rebuilt) + "}"


def parse_belief_salvaging(
    parse_belief: Callable[[str, str], Optional[str]],
    text: str,
    info_str: str,
    finish_reason: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """
    Salvage `text` and feed the result through the strict per-game
    `parse_belief`. The single path every consumer uses to decide whether a
    rejected completion is recoverable, so none can disagree with the
    harness about which rows parsed. A completion cut off at the token cap
    (`finish_reason == "length"`) is never salvaged. It does not try the
    strict path first: the caller owns that ordering.
    :param parse_belief: the game's strict belief parser (Callable[[str, str], Optional[str]])
    :param text: the answer half of the belief completion (str)
    :param info_str: the info-state string the parser is keyed on (str)
    :param finish_reason: the completion's finish reason, or None when not known (Optional[str])
    :returns: (belief JSON string, salvage kind) or None (Optional[tuple[str, str]])
    """
    if finish_reason == "length":
        return None
    salvaged = salvage_belief_object(text)
    if salvaged is None:
        return None
    belief = parse_belief(salvaged, info_str)
    if belief is None:
        return None
    return belief, BELIEF_SALVAGE_KEY_QUOTED


@dataclass(frozen=True)
class GamePromptSpec:
    """Extracted prompt content for one game family.

    Fields:
      game: short game key ("kuhn", "leduc", "goofspiel", "liarsdice", "darkhex").
      instruction: the fixed rules prose ``build_prompt`` renders around the
          info-set, as a ``dict[str, str]`` of named ``str.format()``-style
          template lines. Every game uses this dict shape:
          a ``"rules"`` key always holds the fixed rules text; games whose
          prose also states an action-vocabulary token in situational
          "options" text (kuhn, leduc, liarsdice) carry additional
          ``"options_*"``/``"situation_*"`` template keys whose placeholders
          are filled from ``action_vocabulary`` at render time (e.g.
          ``"{FOLD} (give up the pot) or {CALL} (...)"``.format(FOLD=_VOCAB
          ["FOLD"], ...)), so a vocabulary rename propagates into the prose
          instead of leaving a stale, hand-typed copy. Dimension-parametric
          games (goofspiel, darkhex) additionally carry per-variant template
          lines filled by ``make_fns``'s closures.
      action_vocabulary: canonical action-label literals, keyed by symbolic
          name (plus an optional ``"parse_synonyms"`` sub-dict of extra
          parse-only literals per canonical name, e.g. Kuhn/Leduc's "PASS"
          for CHECK). ``action_suffix``, ``action_labels``, ``parse_action``,
          and (where applicable) ``instruction``'s options templates all read
          the same dict; none hand-types the literal a second time.
      belief_question: fixed belief-elicitation prose. A ``str.format()``
          template whose placeholders are the per-info-set values (own card
          name, deck counts, ...) the module already computes, plus a
          ``{belief_exemplar}`` placeholder for the rendered JSON example
          (every game uses the same key).
          Following a Qwen-14B calibration pilot,
          every game's question also states a precision/format
          instruction sentence -- "Use decimal numbers with at most 4
          decimal places (never fractions)", plus "the values should sum to
          1" for the four games whose answer is a normalised distribution
          (darkhex's independent-Bernoulli values already state they need
          not sum to 1, so its sentence omits that clause). The exemplar
          alone is not sufficient: with only an exemplar, the pilot measured
          Qwen emitting correct six-key liarsdice beliefs at
          17-significant-digit float precision (shown 4dp by example, never
          told) and leduc values as bare fractions (``"1/3"``, not a JSON
          number) -- both a model-behaviour gap an exemplar alone does not
          close.
      belief_exemplar: the belief-question JSON exemplar text, a literal
          ``str`` for all five games. Every key in every exemplar is a
          placeholder token (``"<...>"``-shaped) that cannot map to any real
          key in any game's ``parse_belief`` (own card name, rank name, die
          face, card number, or cell label) -- this guards against the
          verbatim-echo defect: an echo of the belief question ends with this
          exemplar, and ``parse_belief`` scans brace-free ``{...}`` groups
          from last to first and keeps the last parseable one, so a real (non-placeholder) exemplar key would let a pure
          echo parse as though it were an elicited belief. Because no key
          maps, every game's ``parse_belief`` returns ``None`` on an echo
          (each already returns ``None`` when the parsed key set is empty,
          rather than renormalising over a partial match -- verified, not
          assumed, per-game in ``tests/test_prompt_fidelity.py``).
          leduc/liarsdice/kuhn/goofspiel/darkhex all hold the full
          exemplar text (for kuhn/leduc/liarsdice this includes the outer
          ``{...}`` braces, matching their ``belief_question`` templates'
          single ``{belief_exemplar}`` placeholder; for goofspiel/darkhex it
          is the inner key:value content only, matching their templates'
          ``{{{belief_exemplar}}}`` double-brace wrapping that supplies the
          outer braces) -- dimension-independent for goofspiel/darkhex too,
          since a placeholder-keyed illustration does not need to track
          num_cards or the live legal-cell set.
          Preserved property (leduc): the worked example must
          never equal the true deck-conditional prior at any info-set --
          placeholder keys satisfy this by construction (the example is
          not a distribution over real keys at all, so it cannot coincide
          with any real prior under any permutation).
      display_scale: the factor a game module applies to the payoff magnitudes
          it renders from game state rather than from template text -- Leduc's
          pot and chip stacks, Goofspiel's prize values and scores. The
          whitelist in ``probe_variants`` can only move numbers that sit in the
          template prose; a magnitude parsed out of the info-state string and
          formatted by an f-string has no prose to whitelist, so the payoff
          scale has to travel with the spec instead. ``1`` is the canonical
          identity and is what every module-level spec carries.
      option_order: the order the game modules list the legal options in, in every place the
          prompt lists them: ``OPTION_ORDER_CANONICAL`` (the default, what every module-level
          spec carries) or ``OPTION_ORDER_REVERSED``. It is code that builds those lists, not
          template text, so it travels on the spec the way ``display_scale`` does, and it is
          emitted into the digest only when it is not canonical so that adding it moved no
          existing digest. Any other value raises at construction.
      memory_payoff_scale: the factor the game module applies to the payoffs its memory block
          shows, so a payoff-scale variant's block states every result in the units its
          prompt uses. ``display_scale`` cannot carry it, because Kuhn, Liar's Dice and Dark
          Hex scale their stakes in template prose and leave ``display_scale`` at 1. It is not
          in the digest: a payoff-scale arm records a prompt spec hash that run evidence
          recomputes from the variant, and the memory block is rendered only above M0, where
          its wording is pinned by the rendered-text tests instead. ``1`` is canonical.
    """

    game: str
    instruction: Any
    action_vocabulary: Mapping[str, Any]
    belief_question: str
    belief_exemplar: Any
    display_scale: int = 1
    option_order: str = OPTION_ORDER_CANONICAL
    memory_payoff_scale: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.display_scale, int)
            or isinstance(self.display_scale, bool)
            or self.display_scale < 1
        ):
            raise ValueError(
                f"display_scale must be a positive int, got {self.display_scale!r}: a zero, "
                f"negative or fractional scale would render wrong payoff magnitudes under a "
                f"variant label"
            )
        if self.option_order not in OPTION_ORDERS:
            raise ValueError(
                f"option_order must be one of {list(OPTION_ORDERS)}, got "
                f"{self.option_order!r}: an unrecognised order would render canonical bytes "
                f"under a variant label"
            )
        if (
            not isinstance(self.memory_payoff_scale, int)
            or isinstance(self.memory_payoff_scale, bool)
            or self.memory_payoff_scale < 1
        ):
            raise ValueError(
                f"memory_payoff_scale must be a positive int, got "
                f"{self.memory_payoff_scale!r}: a zero, negative or fractional scale would "
                f"show memory payoffs in units the prompt does not use"
            )


# ---------------------------------------------------------------------------
# The live prompt set
# ---------------------------------------------------------------------------
# Every string a model reads that does not depend on the game state lives in
# ``lmopt/prompts/sets/<LIVE_SET>/``: the five game files hold the four content fields
# below, ``wrappers/instruct.json`` holds the sentences every game shares (the objective
# paragraph, the belief steps, the belief format sentence, the legal-response sentence), and
# the set's README records what the set is and why its text reads as it does. This module
# turns those files into the ``GamePromptSpec`` each game module renders from, and composes
# the two closing sentences that depend on the live node (``legal_response_suffix``,
# ``quoted_keys``) from the wrapper's templates.

_PROMPT_SET = load_set(LIVE_SET)
_INSTRUCT = _PROMPT_SET.wrapper("instruct")


def legal_response_suffix(labels: Sequence[str], note: str = "") -> str:
    """
    the closing lines of every action call: the exact legal replies at this node, then one
    fixed instruction to answer with one of them and nothing else. Kept on one line because
    the echo-adversary wrapper quotes the longest line of the suffix as the line that
    names the legal labels
    :param labels: the exact reply strings legal at this node, in display order (Sequence[str])
    :param note: optional extra guidance placed after the list, e.g. a common mistake (str)
    :returns: the suffix text, starting with a blank line (str)
    """
    if not labels:
        raise ValueError(
            "legal_response_suffix: a decision node must offer at least one reply"
        )
    if len(labels) == 1:
        head = _INSTRUCT["legal_response_one"].format(label=labels[0])
    else:
        head = _INSTRUCT["legal_response_many"].format(labels=", ".join(labels))
    return f"\n\n{head}{note}{_INSTRUCT['legal_response_tail']}"


def in_display_order(spec: GamePromptSpec, items: Sequence[Any]) -> list[Any]:
    """
    the items of one option list in the order this spec displays them. Every game calls this
    after it has built the canonical list, so the reversed order is exactly the reverse of the
    canonical one in every place a prompt lists its options
    :param spec: the spec the prompt is rendered from (GamePromptSpec)
    :param items: the options in canonical order; never modified (Sequence[Any])
    :returns: a new list, in display order (list[Any])
    """
    if spec.option_order == OPTION_ORDER_REVERSED:
        return list(reversed(items))
    return list(items)


def labels_in_display_order(
    spec: GamePromptSpec, labels: Mapping[int, str]
) -> dict[int, str]:
    """
    an ``action_labels`` mapping with its insertion order equal to the order the prompt
    displays the labels in, so the keys, the closing list and the prose list agree
    :param spec: the spec the prompt is rendered from (GamePromptSpec)
    :param labels: action -> label, in canonical insertion order (Mapping[int, str])
    :returns: the same pairs, inserted in display order (dict[int, str])
    """
    return {action: labels[action] for action in in_display_order(spec, list(labels))}


def require_canonical_option_order(
    spec: Optional[GamePromptSpec], factory: str
) -> None:
    """
    refuse a spec that asks for an option order this factory cannot render. The template
    prose reads the spec on its own, so a factory that ignored ``option_order`` in its code-built
    lists would render a prompt whose prose and closing list disagree
    :param spec: the spec the factory was handed, or None for its module-level default
        (Optional[GamePromptSpec])
    :param factory: the factory's name, for the message (str)
    :raises ValueError: when the spec's option order is not canonical
    """
    if spec is not None and spec.option_order != OPTION_ORDER_CANONICAL:
        raise ValueError(
            f"{factory} cannot render option_order={spec.option_order!r}: its closing list is "
            f"built in code that does not reorder, so the prompt would contradict itself"
        )


def quoted_keys(keys: Sequence[str]) -> str:
    """
    the belief keys as a plain quoted list for the belief call's format sentence. Deliberately
    not JSON: no ``{...}`` group is formed, so an echo of the question cannot parse as a belief
    :param keys: the exact key strings ``parse_belief`` accepts at this node (Sequence[str])
    :returns: e.g. ``"Jack", "Queen"`` (str)
    """
    return ", ".join(f'"{k}"' for k in keys)


KUHN_SPEC = _PROMPT_SET.spec("kuhn")
LEDUC_SPEC = _PROMPT_SET.spec("leduc")
GOOFSPIEL_SPEC = _PROMPT_SET.spec("goofspiel")
LIARSDICE_SPEC = _PROMPT_SET.spec("liarsdice")
DARKHEX_SPEC = _PROMPT_SET.spec("darkhex")


_ALL_SPECS = (KUHN_SPEC, LEDUC_SPEC, GOOFSPIEL_SPEC, LIARSDICE_SPEC, DARKHEX_SPEC)


def _hash_fields(spec: GamePromptSpec) -> dict:
    """
    the content fields of one spec, in the shape the digest serialises. ``display_scale``
    is emitted only when it is not the canonical 1, and ``option_order`` only when it is not
    canonical, so that adding either field moved no existing digest while a spec that actually
    rescales its rendered magnitudes or reorders its options still hashes differently from the
    canonical one it was derived from
    :param spec: the spec to serialise (GamePromptSpec)
    :returns: the digest fields for that spec (dict)
    """
    fields = {
        "instruction": spec.instruction,
        "action_vocabulary": spec.action_vocabulary,
        "belief_question": spec.belief_question,
        "belief_exemplar": spec.belief_exemplar,
    }
    if spec.display_scale != 1:
        fields["display_scale"] = spec.display_scale
    if spec.option_order != OPTION_ORDER_CANONICAL:
        fields["option_order"] = spec.option_order
    return fields


def _canonical_payload(variant: Optional[GamePromptSpec] = None) -> dict:
    """Deterministic, sorted-key payload covering all five specs.

    ``variant`` replaces the entry for the game it names,
    so the payload describes the content a run is actually rendering rather
    than the module-level default. Substitution, not addition: the other four
    games' content stays in the digest, so the hash keeps meaning "the prompt
    content this harness would emit", with one game's content swapped.
    """
    specs = list(_ALL_SPECS)
    if variant is not None:
        games = [spec.game for spec in specs]
        if variant.game not in games:
            raise ValueError(
                f"variant prompt spec for unregistered game {variant.game!r}; "
                f"expected one of "
                f"{sorted(games)}. A spec that matches no registered game cannot "
                f"be substituted, and hashing it as if it were absent would emit "
                f"the CANONICAL digest for a run rendering something else."
            )
        specs[games.index(variant.game)] = variant
    return {
        "version": _SPEC_HASH_VERSION,
        "games": {spec.game: _hash_fields(spec) for spec in specs},
    }


def game_spec_hash(spec: GamePromptSpec) -> str:
    """SHA256 hex over one game's canonical, sorted, versioned serialisation."""
    payload = {
        "version": _SPEC_HASH_VERSION,
        "game": spec.game,
        **_hash_fields(spec),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prompt_spec_hash(variant: Optional[GamePromptSpec] = None) -> str:
    """SHA256 hex over a canonical, sorted, versioned serialisation of all
    five games' extracted prompt content.

    Deterministic across calls/processes: ``json.dumps(..., sort_keys=True)``
    removes any dependence on dict insertion order or Python's per-process
    ``repr()``/hash-randomisation behaviour. Not pinned to a literal digest
    value here (tests/test_prompt_spec.py), since prompt content changes
    deliberately between prompt versions and a pinned digest would make that
    necessary churn look like a regression.

    ``variant`` is the spec a probe arm is actually rendering with. Hashing
    the module-level specs unconditionally would make a run through
    ``get_game_fns_for_game(game, spec=variant)`` log (through
    ``lm_vllm.PROMPT_SPEC_HASHES``, which binds ``prompt_schema_version`` to
    this digest) a provenance positively asserting canonical content against
    variant bytes -- fabricated provenance, not missing provenance, which is
    the failure class ``prompting/__init__.py``'s dispatch raise exists to
    prevent. The no-argument call hashes the module-level specs, so a base
    arm's logged hash equals its version's registry entry exactly.
    """
    canonical = json.dumps(
        _canonical_payload(variant), sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "GamePromptSpec",
    "KUHN_SPEC",
    "LEDUC_SPEC",
    "GOOFSPIEL_SPEC",
    "LIARSDICE_SPEC",
    "DARKHEX_SPEC",
    "OPTION_ORDER_CANONICAL",
    "OPTION_ORDER_REVERSED",
    "OPTION_ORDERS",
    "in_display_order",
    "labels_in_display_order",
    "require_canonical_option_order",
    "game_spec_hash",
    "prompt_spec_hash",
    "count_matching_labels",
    "has_multiple_labels",
    "extract_belief_object",
    "salvage_belief_object",
    "parse_belief_salvaging",
    "BELIEF_SALVAGE_KEY_QUOTED",
    "split_reasoning",
]
