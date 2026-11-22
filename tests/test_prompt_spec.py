"""Tests for the extracted per-game prompt spec.

Pins two properties of `prompt_spec_hash()`/`game_spec_hash()`: they are
deterministic (no dict-order or repr/hash-randomisation dependence) and
sensitive (mutating any one field of any one game's spec changes the digest).
The literal digest value is deliberately never pinned -- prompt content
changes on purpose, and pinning the digest would make that necessary churn
look like a regression (see spec.py's own `prompt_spec_hash` docstring).

Also includes one production-entrypoint test: a spec-field mutation must be
observable through `get_game_fns_for_game` on a real loaded pyspiel game, not
only through a direct call into a prompting submodule -- otherwise a
regression where a game module falls back to a locally hardcoded literal
instead of reading the shared spec would pass every direct-module test while
silently breaking the "one source" property.

`test_mutating_any_spec_field_changes_both_hashes` loops over every (game,
field) pair and asserts a `dataclasses.replace` mutation changes both
`game_spec_hash` and `prompt_spec_hash`; a per-game-name check would be
vacuous, since the five `game` strings are already distinct. Subprocess
determinism tests cover both hashes. `pytest.importorskip("pyspiel")` is at
module scope because this file is transitively pyspiel-dependent via
`lmopt.prompting.__init__` -> `lmopt.games.loaders`.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

# Module-scoped, not function-local: `lmopt.prompting.__init__` (imported
# below) pulls in `darkhex_board_dims` from `lmopt.games.loaders`, which does
# `import pyspiel` at module scope -- so every import in this file is
# transitively pyspiel-dependent, and without pyspiel installed the file would
# hard-error at collection instead of skipping cleanly.
pyspiel = pytest.importorskip("pyspiel")

import lmopt.prompting.kuhn as kuhn_module
from lmopt.agents.lm_vllm import PROMPT_SCHEMA_VERSION, PROMPT_SPEC_HASHES
import lmopt.prompting.leduc as leduc_module
from lmopt.games.loaders import load_game
from lmopt.prompting import get_game_fns_for_game
from lmopt.prompting.fidelity import enumerate_infosets
from lmopt.prompting import spec as spec_module
from lmopt.prompting.spec import (
    DARKHEX_SPEC,
    count_matching_labels,
    extract_belief_object,
    game_spec_hash,
    has_multiple_labels,
    prompt_spec_hash,
    split_reasoning,
)


def test_kuhn_spec_rules_renders_byte_identical_at_canonical_three_cards():
    from lmopt.prompting import kuhn

    expected = (
        "You are playing Kuhn poker, a simplified card game.\n"
        "\n"
        "Rules:\n"
        "- Three cards: Jack (lowest), Queen, King (highest).\n"
        "- Each player is dealt one card secretly. The third card is not in play.\n"
        "- Before the deal each player puts 1 chip in the pot (the ante).\n"
        "- Player 0 acts first and may CHECK (free) or BET (stake 1 chip).\n"
        "- After a CHECK, the other player may CHECK or BET (stake 1 chip). After a BET, "
        "the other player may CALL (stake 1 chip to match, go to showdown) or FOLD "
        "(give up the pot).\n"
        "- If nobody bets, or a bet is matched, the higher card wins the pot.\n"
        "- The winner gains 1 chip if nobody bet or someone gave up the pot, and 2 chips "
        "if a bet was matched. The loser loses the same amount.\n"
        "\n"
        "This is a two-player zero-sum game: whatever you win, your opponent loses. "
        "You will play many hands. Your aim is to win as many chips as possible on "
        "average, not to win any single hand."
    )
    rendered = kuhn._SPEC.instruction["rules"].format(
        num_ranks="Three",
        rank_range_desc="Jack (lowest), Queen, King (highest)",
        n_spare="The third",
        spare_plural=" is",
    )
    assert rendered == expected


def test_leduc_spec_rules_renders_byte_identical_at_canonical_six_card_deck():
    from lmopt.prompting import leduc

    expected = (
        "You are playing Leduc poker, a two-player poker variant.\n"
        "\n"
        "Rules:\n"
        "- Six cards: Jack, Queen, King (two of each). Each player is dealt one private card.\n"
        "- Before the deal each player puts 1 chip in the pot (the ante).\n"
        "- Round 1: both players bet with only their private card. Every bet or raise in "
        "round 1 is 2 chips.\n"
        "- Round 2: one community card is revealed face-up; both players bet again. Every "
        "bet or raise in round 2 is 4 chips.\n"
        "- Betting: check (free, only if nobody has bet this round), bet, call (match the "
        "current bet), raise (re-raise), or fold (give up the pot).\n"
        "- Maximum two bets/raises per round (check, bet, raise — then no more raises).\n"
        "- Showdown: a private card that pairs the community card wins. Otherwise the "
        "higher private card wins. Equal cards split the pot.\n"
        "- You win or lose exactly the chips that went into the pot.\n"
        "\n"
        "This is a two-player zero-sum game: whatever you win, your opponent loses. "
        "You will play many hands. Your aim is to win as many chips as possible on "
        "average, not to win any single hand."
    )
    rendered = leduc._SPEC.instruction["rules"].format(
        num_cards="Six",
        rank_list_desc="Jack, Queen, King",
    )
    assert rendered == expected


def test_leduc_spec_belief_question_renders_byte_identical_at_canonical_three_ranks():
    from lmopt.prompting import leduc
    from lmopt.prompting.spec import quoted_keys

    expected = (
        "\n\nYour private card is Jack. The deck has two of each rank; of the "
        "5 cards you have not seen, 1 Jack; 2 Queens; 2 Kings remain unseen. What "
        "probability do you assign to your opponent's card being each rank? You cannot "
        "see their card.\n"
        "Work it out in three steps before you answer.\n"
        "1. Start from the deal alone: each unseen card is equally likely to be theirs, "
        "so before any betting a rank's chance is its unseen count divided by 5. A rank "
        "with no unseen cards gets 0.\n"
        "2. Update on what your opponent has DONE so far. For each possibility, ask how "
        "likely a strong opponent in that situation would have been to act exactly as "
        "they did. Possibilities that make their actions natural become more likely; "
        "ones that make their actions strange become less likely. If they have not acted "
        "yet, keep step 1's numbers.\n"
        "3. Make the numbers add up to about 1.\n"
        "Assume your opponent plays an optimal, unexploitable strategy. Infer their "
        "hidden state from their observed actions under that assumption.\n"
        "Give all three ranks. Reply with only a JSON object with exactly these keys: "
        '"Jack", "Queen", "King". Each value is a probability between 0 and 1, written '
        "as a decimal with 2 decimal places (never a fraction). The values should add up "
        "to about 1; we rescale them, so do not worry about rounding. Example of the "
        "FORMAT only, not a suggested answer (placeholder keys, made-up numbers): "
        '{"<rank_A>": 0.5, "<rank_B>": 0.3, "<rank_C>": 0.2}'
    )
    rendered = leduc._SPEC.belief_question.format(
        own_name="Jack",
        total_unseen=5,
        deck_desc="1 Jack; 2 Queens; 2 Kings",
        num_ranks_word="three",
        belief_exemplar=leduc._SPEC.belief_exemplar,
        belief_keys=quoted_keys(["Jack", "Queen", "King"]),
    )
    assert rendered == expected


def test_five_specs_registered_one_per_game():
    """Characterisation: asserts the feature is
    present (five specs, one per game family) rather than naming a mutation
    of finished code -- there is no partial-registration state to revert to
    that would still make sense as "finished" code to mutate against.
    """
    games = {s.game for s in spec_module._ALL_SPECS}
    assert games == {"kuhn", "leduc", "goofspiel", "liarsdice", "darkhex"}


def test_prompt_spec_hash_is_a_sha256_hex_digest():
    """Characterisation: pins the digest's shape
    (64 hex chars) as a basic sanity check, not a specific implementation
    choice to mutate against -- any sha256-based construction satisfies it
    equally; the determinism/sensitivity tests below are where the actual
    mutation-detection work happens.
    """
    digest = prompt_spec_hash()
    assert isinstance(digest, str)
    assert len(digest) == 64
    int(digest, 16)  # raises ValueError if not valid hex


def test_prompt_spec_hash_deterministic_within_process():
    assert prompt_spec_hash() == prompt_spec_hash() == prompt_spec_hash()


def test_prompt_spec_hash_deterministic_across_processes():
    cmd = [
        sys.executable,
        "-c",
        "from lmopt.prompting.spec import prompt_spec_hash; print(prompt_spec_hash())",
    ]
    out1 = subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()
    out2 = subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out1 == out2
    assert len(out1) == 64


def test_game_spec_hash_deterministic_across_processes():
    """Mirrors `test_prompt_spec_hash_deterministic_across_processes` for
    `game_spec_hash`: the two hashes share the same
    canonical-serialisation construction, so the process-randomisation
    mutation that could defeat one (``hash()``/``repr()`` instead of
    ``json.dumps(..., sort_keys=True)``) could equally defeat the other if
    `game_spec_hash` diverged from that construction independently.
    """
    cmd = [
        sys.executable,
        "-c",
        (
            "from lmopt.prompting.spec import KUHN_SPEC, game_spec_hash; "
            "print(game_spec_hash(KUHN_SPEC))"
        ),
    ]
    out1 = subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()
    out2 = subprocess.run(
        cmd, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out1 == out2
    assert len(out1) == 64


def _install_mutated_spec(monkeypatch, global_name: str, **field_overrides):
    """Swap one game's module-level spec instance for a field-mutated copy,
    keeping ``spec._ALL_SPECS`` (what `prompt_spec_hash` actually reads)
    consistent with the swap -- otherwise the mutation would be invisible
    to `prompt_spec_hash`, which iterates the tuple captured at import
    time, not the individual module globals.
    """
    original = getattr(spec_module, global_name)
    mutated = dataclasses.replace(original, **field_overrides)
    monkeypatch.setattr(spec_module, global_name, mutated)
    new_all = tuple(
        mutated if s.game == original.game else s for s in spec_module._ALL_SPECS
    )
    monkeypatch.setattr(spec_module, "_ALL_SPECS", new_all)
    return mutated


_GAME_TO_GLOBAL = {
    "kuhn": "KUHN_SPEC",
    "leduc": "LEDUC_SPEC",
    "goofspiel": "GOOFSPIEL_SPEC",
    "liarsdice": "LIARSDICE_SPEC",
    "darkhex": "DARKHEX_SPEC",
}

# The GamePromptSpec fields, "game" itself included (a `game_spec_hash`
# reading only this field is a defect, so it must be covered by the same loop
# that proves the other four are also covered, not asserted separately).
_SPEC_FIELDS = [
    "game",
    "instruction",
    "action_vocabulary",
    "belief_question",
    "belief_exemplar",
]


def _mutate_field_value(spec: "spec_module.GamePromptSpec", field: str):
    """Return a value for `field` guaranteed to differ from spec's current
    value, regardless of whether the field holds a str (game, belief_question,
    and belief_exemplar/instruction for some games) or a dict (action_vocabulary,
    and belief_exemplar/instruction for other games) -- GamePromptSpec fields
    are intentionally heterogeneous in shape across games (see spec.py's
    docstring), so this dispatches on the RUNTIME type, not the field name.
    """
    current = getattr(spec, field)
    if isinstance(current, str):
        return current + " __MUTATION_PROBE__"
    if isinstance(current, dict):
        return {**current, "__mutation_probe__": "MUTATED"}
    raise TypeError(f"unexpected type for spec field {field!r}: {type(current)!r}")


@pytest.mark.parametrize(
    "game_name,field",
    [(g, f) for g in _GAME_TO_GLOBAL for f in _SPEC_FIELDS],
)
def test_mutating_any_spec_field_changes_both_hashes(monkeypatch, game_name, field):
    """    A check that only compared different games' hashes would be vacuous --
    a `game_spec_hash` that hashed only `spec.game` would still pass it,
    because the five `game` strings are pairwise distinct by construction.
    Asserting on both `game_spec_hash` and `prompt_spec_hash` is the
    consistency clause: a field reachable by one hash and not the other
    would pass this test asymmetrically and expose the gap.
    """
    global_name = _GAME_TO_GLOBAL[game_name]
    original = getattr(spec_module, global_name)

    before_game_hash = game_spec_hash(original)
    before_global_hash = prompt_spec_hash()

    new_value = _mutate_field_value(original, field)
    mutated = _install_mutated_spec(monkeypatch, global_name, **{field: new_value})

    after_game_hash = game_spec_hash(mutated)
    after_global_hash = prompt_spec_hash()

    assert after_game_hash != before_game_hash
    assert after_global_hash != before_global_hash


_PRECISION_INSTRUCTION_STEM = (
    "written as a decimal with 2 decimal places (never a fraction)"
)


@pytest.mark.parametrize("game_name", sorted(_GAME_TO_GLOBAL))
def test_belief_question_states_precision_instruction(game_name):
    """Every game's `belief_question` states the format/precision
    instruction explicitly, in addition to the exemplar. Without it, a model
    is only shown one example, never told what format to use: a Qwen-14B
    calibration pilot emitted correct six-key liarsdice beliefs at
    full-float (17-significant-digit) precision -- roughly 4x longer than
    the 4-decimal-place precision BELIEF_MAX_TOKENS was sized for, so every
    liarsdice belief failed (717/717) -- and rendered leduc beliefs as
    fractions ("1/3") instead of JSON numbers.
    """
    spec = getattr(spec_module, _GAME_TO_GLOBAL[game_name])
    assert _PRECISION_INSTRUCTION_STEM in spec.belief_question, (
        f"{game_name}: belief_question is missing the precision/format "
        f"instruction sentence: {spec.belief_question!r}"
    )


def test_darkhex_precision_instruction_omits_sum_to_one_clause():
    """darkhex's belief values are independent Bernoullis and the question
    itself already states "need not sum to 1" (separate prose, hence why
    this test checks the specific clause below rather than the bare
    substring "sum to 1" -- that bare substring is, correctly, present in
    darkhex's question text). The shared precision instruction's own "; the
    values should sum to 1." clause (present verbatim in the other four
    games) would directly contradict the independent-Bernoulli framing, so
    darkhex's sentence is deliberately the shorter, adapted form (spec.py's
    `DARKHEX_SPEC.belief_question` comment)."""
    assert "the values should sum to 1" not in DARKHEX_SPEC.belief_question


def test_mutating_kuhn_spec_instruction_changes_production_build_prompt(monkeypatch):
    """Production anchor: reaches the spec-reading code
    through the same production entrypoint a live match uses
    (`get_game_fns_for_game` on a real loaded `pyspiel` game), not only via
    a direct call into `lmopt.prompting.kuhn`.

    `pyspiel` is bound by the module-level `importorskip` above the imports,
    since the whole file is transitively pyspiel-dependent.
    """
    game = pyspiel.load_game("kuhn_poker")
    fns = get_game_fns_for_game(game)
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    info = state.information_state_string(0)
    original_prompt = fns.build_prompt(info, 0)

    mutated_spec = dataclasses.replace(
        kuhn_module._SPEC,
        instruction={
            **kuhn_module._SPEC.instruction,
            "rules": "MUTATED INSTRUCTION TEXT PROBE",
        },
    )
    monkeypatch.setattr(kuhn_module, "_SPEC", mutated_spec)

    mutated_prompt = fns.build_prompt(info, 0)
    assert mutated_prompt != original_prompt
    assert "MUTATED INSTRUCTION TEXT PROBE" in mutated_prompt


class TestMultiLabelHelper:
    """`count_matching_labels` / `has_multiple_labels` -- the one shared rule
    every game's `parse_action` calls to reject an echo naming 2+ distinct
    labels. Each case below names a mutation that was actually run against
    the helper, not merely asserted.
    """

    def test_counts_distinct_labels_not_total_occurrences(self):
        """A mutation summing every individual regex match
        (`sum(len(re.findall(pattern, t)) for lab in labels)`) instead of
        counting distinct labels present would count "BET BET BET" as 3
        matches of the same label and incorrectly flag it multi-label -- a
        model repeating one word three times (a stutter) is not an echo of
        two different options and must still parse as one label."""
        assert count_matching_labels("BET BET BET", ["BET"]) == 1
        assert has_multiple_labels("BET BET BET", ["BET"]) is False

    def test_word_boundary_rejects_bet_inside_better(self):
        """A mutation dropping the `\\b...\\b` anchors (matching
        `re.escape(lab)` bare, substring-style) would count "BETTER" as
        containing label "BET" -- BETTER is not an echo of BET."""
        assert count_matching_labels("BETTER", ["BET"]) == 0

    def test_word_boundary_rejects_challenge_inside_challenger(self):
        """Word-boundary sibling case (liarsdice's CHALLENGE label)."""
        assert count_matching_labels("CHALLENGER", ["CHALLENGE"]) == 0

    def test_word_boundary_rejects_raise_inside_reraise(self):
        """Word-boundary sibling case (leduc's RAISE label)."""
        assert count_matching_labels("RERAISE", ["RAISE"]) == 0

    def test_word_boundary_rejects_check_inside_checkmate(self):
        """Word-boundary sibling case (kuhn/leduc's CHECK label)."""
        assert count_matching_labels("CHECKMATE", ["CHECK"]) == 0

    def test_case_insensitive_matching(self):
        """A mutation dropping the `.upper()` normalisation (matching
        case-sensitively) would fail to count lowercase "bet"/"call" against
        the canonical uppercase labels "BET"/"CALL" -- real completions are
        rarely all-caps, so case-sensitive matching would silently stop
        detecting the majority of real echoes."""
        assert count_matching_labels("bet call", ["BET", "CALL"]) == 2
        assert has_multiple_labels("bet call", ["BET", "CALL"]) is True

    def test_empty_labels_is_never_multi_label(self):
        """CHARACTERISATION: names no mutation because none reddens it -- an
        empty `labels` list already yields 0 matches from the natural
        `sum(... for lab in {lab.upper() for lab in labels} if ...)` logic
        even without the `if not labels: return 0` early-return guard (that
        guard exists for the `labels=None` case, covered separately by
        `test_none_labels_returns_false_not_typeerror`, which does name and
        reddens under a real mutation -- deleting the guard raises
        `TypeError` on `None` but stays green on `[]`). This test documents
        the empty-list behaviour for completeness, not as an independent
        regression guard."""
        assert count_matching_labels("BET or CALL", []) == 0
        assert has_multiple_labels("BET or CALL", []) is False

    def test_none_labels_returns_false_not_typeerror(self):
        """Contract: `has_multiple_labels(text, None)` returns `False` rather
        than raising `TypeError` (the set-comprehension iterating `None`) --
        "no labels to check against" means "nothing can be multi-label", not
        an error a caller must guard against."""
        assert count_matching_labels("anything", None) == 0
        assert has_multiple_labels("anything", None) is False

    def test_bet_or_raise_is_one_label_match_at_a_facing_bet_node(self):
        """The BET+RAISE-is-one-label decision, exercised
        through leduc's real production functions (not just the bare
        helper): at a facing-bet-with-raise-available info-set,
        `action_labels()` offers {FOLD, CALL, RAISE} -- "BET" is
        deliberately not one of them (leduc.py's `action_labels` docstring:
        the prompt itself says "do not say BET" there). So:

        - "BET or RAISE" has only one distinct label match (RAISE; "BET" is
          not in the checked set at all) -- not rejected as multi-label,
          parses via RAISE's own branch, and `parse_action_aliased` is
          correctly False (the alias only fires when FOLD/CALL/RAISE are
          all absent and only "BET" is said).
        - bare "BET" has zero label matches (also not rejected), parses via
          the BET->RAISE alias, and `parse_action_aliased` is True.

        This is the exact fixture `test_prompting_leduc.py` uses for its own
        facing-bet-with-raise coverage (`_R1_P1_FACING_BET`), reused here
        rather than hand-built, so this info-set's shape is independently
        verified elsewhere too.
        """
        info = "[Observer: 1][Private: 1][Round 1][Player: 1][Pot: 4][Money: 18 19][Round1: b][Round2: ]"
        labels = leduc_module.action_labels(info)
        assert labels == {0: "FOLD", 1: "CALL", 2: "RAISE"}

        assert has_multiple_labels("BET or RAISE", labels.values()) is False
        assert leduc_module.parse_action("BET or RAISE", info) == 2
        assert leduc_module.parse_action_aliased("BET or RAISE", info) is False

        assert has_multiple_labels("BET", labels.values()) is False
        assert leduc_module.parse_action("BET", info) == 2
        assert leduc_module.parse_action_aliased("BET", info) is True


# ---------------------------------------------------------------------------
# Shared belief-object extraction
# ---------------------------------------------------------------------------


class TestExtractBeliefObject:
    """Taking `re.search(r"\\{[^{}]*\\}", ...)`, the first brace-free-interior
    group, is a defect: a real gemma-4 goofspiel completion reasoned
    correctly, wrote set notation "{1, 2, 3}" twice in prose, then emitted
    the true belief last, and a first-match parser takes the prose and
    discards the belief."""

    def test_prefers_the_last_object_over_prose_braces(self):
        text = (
            "If the opponent has {1, 2, 3}, they bid low.\n\n"
            '{"1": 0.1, "2": 0.4, "3": 0.4, "4": 0.1}'
        )
        assert extract_belief_object(text) == '{"1": 0.1, "2": 0.4, "3": 0.4, "4": 0.1}'

    def test_falls_back_past_a_non_json_group_that_trails_the_real_answer(self):
        """Mirror image of the defect
        `test_prefers_the_last_object_over_prose_braces` covers.

        Concrete real row this fixture reproduces:
        `data/run_experiment/recycle-v4/M1/olmo-2-1124-13b-instruct/
        goofspiel_lm_vs_rl_seed4.parquet` row 916. The model answers first in
        a fenced json block, then narrates, and the narration itself
        contains a later brace group that is not JSON (`{2, 3, 4}`, set
        notation naming remaining cards, not a `key: value` object).
        Unconditional last-group selection returns that non-JSON fragment,
        which every caller's `json.loads` then rejects -- the real answer,
        sitting earlier in the text, is silently discarded exactly the way
        a first-match parser discards it, just from the other end. 1,350 of
        41,030 real multi-brace rows in the collected corpus hit this path
        under the unconditional rule."""
        text = (
            '```json\n{"1": 0.25, "2": 0.35, "3": 0.25, "4": 0.15}\n```\n\n'
            "Given the observed play, their remaining cards could be "
            "{2, 3, 4}."
        )
        assert (
            extract_belief_object(text)
            == '{"1": 0.25, "2": 0.35, "3": 0.25, "4": 0.15}'
        )

    def test_reads_the_belief_field_of_a_structured_response(self):
        """CHARACTERISATION. "Dropping the json.loads branch" is not a
        mutation this test catches: one might expect the response to fall
        through to regex and pick a brace out of the reasoning string, but it
        does not. In the reasoning-first key order the brace groups are
        ['{1, 2, 3, 4}', '{"1": 0.5, "2": 0.5}'] and the last-group rule (3)
        already returns the belief, so the json.loads branch can be
        deleted and this test stays green (4 passed). It documents the
        structured-output shape; the json branch's own regression guard is
        `test_reads_the_belief_field_when_reasoning_comes_last` below."""
        text = (
            '{"reasoning": "They could hold any of {1, 2, 3, 4}.", '
            '"belief": {"1": 0.5, "2": 0.5}}'
        )
        assert json.loads(extract_belief_object(text)) == {"1": 0.5, "2": 0.5}

    def test_reads_the_belief_field_when_reasoning_comes_last(self):
        text = (
            '{"belief": {"1": 0.5, "2": 0.5}, '
            '"reasoning": "They could hold any of {1, 2, 3, 4}."}'
        )
        assert json.loads(extract_belief_object(text)) == {"1": 0.5, "2": 0.5}

    def test_single_object_is_unchanged(self):
        assert (
            extract_belief_object('{"1": 0.25, "2": 0.75}') == '{"1": 0.25, "2": 0.75}'
        )

    def test_returns_none_when_no_object(self):
        assert extract_belief_object("I have no idea") is None

    def test_truncated_response_fails_closed_on_a_reasoning_object(self):
        """        The fixture is a SCHEMA completion cut mid-reasoning, and its
        reasoning quotes a well-formed belief object. That is the whole
        discriminating point: a truncated group that is not valid JSON (e.g.
        '{1, 2, 3}') already fails closed one level down in `parse_belief`
        (kuhn.py:97-100), so such a fixture leaves this mutation green.
        On the unguarded function this text returns
        '{"King":0.6,"Queen":0.4}' and `kuhn.parse_belief` turns it into
        '{"0": 0.0, "1": 0.4, "2": 0.6}' -- a plausible, silently wrong
        posterior logged as if the model had stated it."""
        text = '{"reasoning": "maybe {"King":0.6,"Queen":0.4} fits, and'
        assert extract_belief_object(text) is None

    def test_brace_inside_a_complete_reasoning_string_still_yields_belief(self):
        """        This response is complete and valid JSON, but its reasoning string
        holds an unbalanced '{', so a string-blind count reads it as
        truncated and fails it closed.

        "Applying the guard before rule (1)" is not a mutation this catches:
        the whole suite stays green under it, because the guard skips string
        literals, so
        anything `json.loads` accepts also scans to depth 0 and rule (1)
        still answers. The ordering in `spec.py` is defence in depth with no
        failing case behind it."""
        text = (
            '{"reasoning": "the exemplar looked like { and I lost track", '
            '"belief": {"1": 0.5, "2": 0.5}}'
        )
        assert json.loads(extract_belief_object(text)) == {"1": 0.5, "2": 0.5}

    def test_prose_brace_inside_quotes_does_not_trip_the_guard(self):
        text = 'The exemplar shows "{" then I answer {"1": 0.5, "2": 0.5}'
        assert json.loads(extract_belief_object(text)) == {"1": 0.5, "2": 0.5}

    def test_arithmetic_last_value_is_repaired_not_discarded(self):
        """12 real v6 M1 Qwen liarsdice `belief_failed` rows share this
        exact shape verbatim -- the model fills the last bucket with an
        unevaluated "one minus the others" expression. Strict `json.loads`
        discards the whole six-key object over this one value; the repair
        recovers all six.
        """
        text = (
            '{\n  "1": 0.25,\n  "2": 0.25,\n  "3": 0.15,\n  "4": 0.15,\n'
            '  "5": 0.10,\n  "6": 1.00 - (0.25 + 0.25 + 0.15 + 0.15 + 0.10)\n}'
        )
        result = extract_belief_object(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["6"] == pytest.approx(0.10)
        assert parsed == {
            "1": 0.25,
            "2": 0.25,
            "3": 0.15,
            "4": 0.15,
            "5": 0.10,
            "6": pytest.approx(0.10),
        }

    def test_repair_refuses_a_name_or_call_expression(self):
        """`_safe_arith_eval` must refuse anything that is not pure
        +, -, *, / over numeric literals -- a name or a call node is never
        evaluated, because this runs over untrusted LM output.
        """
        assert spec_module._safe_arith_eval("__import__('os')") is None
        assert spec_module._safe_arith_eval("some_name") is None
        assert spec_module._safe_arith_eval("1 + some_name") is None

    def test_repair_returns_none_rather_than_a_partial_fix(self):
        """A group where one value is a genuinely unparseable expression
        (not just an unevaluated arithmetic one) is not partially repaired
        -- `_repair_arithmetic_belief_values` returns None rather than
        silently dropping the bad key, so the caller's existing "try the
        next candidate group" logic runs, and if none work, the row
        correctly reads as unparseable rather than fabricated.
        """
        text = '{"1": 0.5, "2": some_undefined_name}'
        assert spec_module._repair_arithmetic_belief_values(text) is None
        assert extract_belief_object(text) is None

    def test_liarsdice_parse_belief_recovers_the_real_row_through_the_full_path(self):
        """Reachability anchor: the same text that reads `belief_failed=True`
        in the real v6 M1 Qwen data parses through `liarsdice.parse_
        belief` (the production entrypoint `act()` calls), not just through
        `extract_belief_object` in isolation.
        """
        import lmopt.prompting.liarsdice as liarsdice_module

        raw = (
            '{\n  "1": 0.25,\n  "2": 0.25,\n  "3": 0.15,\n  "4": 0.15,\n'
            '  "5": 0.10,\n  "6": 1.0 - (0.25 + 0.25 + 0.15 + 0.15 + 0.10)\n}'
        )
        result = liarsdice_module.parse_belief(raw, "irrelevant-info-set")
        assert result is not None
        parsed = json.loads(result)
        # liarsdice.parse_belief re-keys 1..6 (face) to "0".."5" (0-indexed
        # die action) and normalises to sum 1.0 -- see that function's own
        # docstring for why the reindex exists.
        assert set(parsed.keys()) == {"0", "1", "2", "3", "4", "5"}
        assert sum(parsed.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# split_reasoning
# ---------------------------------------------------------------------------


class TestSplitReasoning:
    """`split_reasoning` identifies a reasoning trace structurally (a
    `</think>` delimiter), never positionally (a "last line" rule
    reintroduces a liarsdice echo defect affecting 3,078 info-sets).
    `parse_action`/`extract_belief_object`
    themselves stay untouched here -- these tests only pin `split_reasoning`
    itself, not act()-level composition.
    """

    def test_full_think_block_splits_and_answer_parses(self):
        """Property 1a (shape 1, `<think>trace</think>answer`). The trace
        names both legal labels (CHECK and BET); only the answer segment is
        handed to the real `kuhn_module.parse_action`, and it resolves
        cleanly because the answer segment alone names a single label."""
        text = "<think>Between CHECK and BET, betting applies more pressure.</think>BET"
        trace, answer = split_reasoning(text)
        assert trace == "Between CHECK and BET, betting applies more pressure."
        assert answer == "BET"
        assert kuhn_module.parse_action(answer, "0") == 1

    def test_close_tag_only_splits_and_answer_parses(self):
        """Property 1b (shape 2, `trace</think>answer`, no `<think>`
        opener -- the "pre-opened template" shape some servers emit)."""
        text = "Between CHECK and BET, betting applies more pressure.</think>CHECK"
        trace, answer = split_reasoning(text)
        assert trace == "Between CHECK and BET, betting applies more pressure."
        assert answer == "CHECK"
        assert kuhn_module.parse_action(answer, "0") == 0

    def test_leading_whitespace_before_open_tag_tolerated(self):
        """Contract clause: leading whitespace before `<think>` is
        tolerated (a server that prepends a newline before the block)."""
        text = "\n   <think>weighing the options</think>CHECK"
        trace, answer = split_reasoning(text)
        assert trace == "weighing the options"
        assert answer == "CHECK"

    def test_first_close_tag_wins_over_a_quoted_one_in_the_answer(self):
        """Property 1c: the answer segment quotes a literal `</think>`
        after the real close."""
        text = '<think>CHECK or BET, I will pick</think>BET (not "</think>" for real)'
        trace, answer = split_reasoning(text)
        assert trace == "CHECK or BET, I will pick"
        assert answer == 'BET (not "</think>" for real)'
        assert kuhn_module.parse_action(answer, "0") == 1

    def test_no_delimiter_returns_none_and_text_unchanged(self):
        """Property 2: no `</think>` anywhere returns `(None, text)`
        BYTE-identical -- no positional fallback of any kind. Fixture is
        liarsdice's real `action_suffix` closing line (spec.py's
        LIARSDICE_SPEC):
        "Respond with exactly one bid (e.g. '1 4' = at least one die shows
        4)." names bid literal "1 4" in a parenthetical."""
        text = (
            "The opponent likely holds low dice given the round history so far.\n\n"
            "Respond with exactly one bid (e.g. '1 4' = at least one die shows 4)."
        )
        trace, answer = split_reasoning(text)
        assert trace is None
        assert answer == text

    def test_unterminated_think_tag_returns_none_and_text_unchanged(self):
        """Property 3: an opened `<think>` that never closes returns
        `(None, text)`, matching vLLM's own fallback."""
        text = "<think>I am still deciding whether to CHECK or BET"
        trace, answer = split_reasoning(text)
        assert trace is None
        assert answer == text

    def test_empty_segments_are_empty_strings_not_none(self):
        """Property 4: a delimiter with nothing on one side yields `''` on that
        side, never `None`. The distinction is load-bearing for the logged
        column: `None` means "no delimiter was present, so no trace exists",
        while `''` means "a delimiter was present and the trace was empty" --
        collapsing the two would make an empty-trace row indistinguishable from
        a no-parser row in `reasoning_trace`."""
        # Shape 2 (close tag only), empty trace.
        trace, answer = split_reasoning("</think>BET")
        assert trace == ""
        assert trace is not None
        assert answer == "BET"
        assert kuhn_module.parse_action(answer, "0") == 1

        # Shape 1 (full block), empty trace.
        trace, answer = split_reasoning("<think></think>BET")
        assert trace == ""
        assert trace is not None
        assert answer == "BET"

        # Shape 1, empty answer: the model reasoned and then said nothing.
        trace, answer = split_reasoning("<think>t</think>")
        assert trace == "t"
        assert answer == ""
        assert kuhn_module.parse_action(answer, "0") is None


# ---------------------------------------------------------------------------
# Prompt-version <-> prompt-content binding
# ---------------------------------------------------------------------------


class TestPromptVersionBinding:
    """`PROMPT_SCHEMA_VERSION` must actually denote the prompt content it
    claims to denote.

    The version is a hand-bumped integer, and a "bump on any content change"
    rule that lives only in a comment is easily broken: the hash can move
    (`7ca04b44` -> `a2855db1` -> `e3b19c46` -> `fb1bba1a`) while the constant
    reads `2` throughout.

    This is a TRIPWIRE, not a gate: see `PROMPT_SPEC_HASHES`'s own comment for
    the two ways it fails to bind (the pin is editable in place, and the hash
    is under-sensitive to rendering-code changes). It is kept anyway because
    it converts a convention into a diff nobody can make accidentally.
    """

    def test_current_hash_matches_the_pin_for_the_declared_version(self):
        """        Verified red before being trusted: a content change that moves the
        hash to `fb1bba1a...` while leaving the version at 2 fails this test.
        A pin first observed green proves nothing.
        """
        assert PROMPT_SCHEMA_VERSION in PROMPT_SPEC_HASHES, (
            f"PROMPT_SCHEMA_VERSION is {PROMPT_SCHEMA_VERSION} but PROMPT_SPEC_HASHES "
            f"has no entry for it. Bumping the version means ADDING an entry recording "
            f"the hash that version denotes, not just incrementing the integer."
        )
        assert prompt_spec_hash() == PROMPT_SPEC_HASHES[PROMPT_SCHEMA_VERSION], (
            f"Prompt content changed without a version bump.\n"
            f"  PROMPT_SCHEMA_VERSION = {PROMPT_SCHEMA_VERSION}\n"
            f"  pinned   = {PROMPT_SPEC_HASHES[PROMPT_SCHEMA_VERSION]}\n"
            f"  computed = {prompt_spec_hash()}\n"
            f"Two legitimate fixes: (a) revert the content change, or (b) bump "
            f"PROMPT_SCHEMA_VERSION and ADD a new PROMPT_SPEC_HASHES entry for it, "
            f"accepting that previously collected arms become incomparable. "
            f"Overwriting the existing "
            f"entry is NOT a fix: it relabels collected data."
        )

    def test_landed_v2_arms_hash_is_pinned_and_never_rewritten(self):
        """        The literal is duplicated here deliberately rather than read from
        `PROMPT_SPEC_HASHES`: a test that reads the value it is checking would
        be an algebraic identity, a check that cannot fail. This is the one place the collected
        arms' identity is written down twice on purpose.
        """
        assert PROMPT_SPEC_HASHES[2] == (
            "e3b19c4625607e5a97c512730ba597dce4ca0fe91472c4785b6edd5b11f87884"
        )


@pytest.mark.parametrize(
    "game_key,belief_json",
    [
        ("kuhn", '{"Jack": 0.5, "King": 0.5}'),
        ("leduc", '{"Jack": 0.5, "King": 0.5}'),
        ("goofspiel", '{"1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25}'),
        ("liarsdice", '{"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.1, "6": 0.1}'),
        ("darkhex", '{"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}'),
    ],
)
def test_reasoning_with_prose_braces_still_parses(game_key, belief_json):
    """Two anchors: this goes through the real `parse_belief` each game
    exports, not the helper.

    All five games are parametrised: the named mutation is "any one game",
    and a two-game parametrisation survives it on the other three. The
    info-set is taken from `enumerate_infosets` on a real loaded game rather
    than hand-typed, so the fixture is a node the harness actually reaches
    (kuhn's first P0 info-set is "0", not "1").
    """
    game = load_game(game_key)
    fns = get_game_fns_for_game(game)
    info = sorted(enumerate_infosets(game, player=0))[0]
    text = f"They might hold {{1, 2, 3}}, so:\n\n{belief_json}"
    assert fns.parse_belief(text, info) is not None


class TestSpecAwarePromptHash:
    """The content hash must be computed from the spec that actually renders.

    Hashing the five module-level specs unconditionally would make an arm
    rendered through `get_game_fns_for_game(game, spec=variant)` log (via
    `PROMPT_SPEC_HASHES`) a provenance positively asserting canonical content
    against variant bytes: fabricated, not missing. The optional argument
    prevents that, and the no-argument call stays byte-identical so every
    existing pin holds.
    """

    def test_no_argument_still_hashes_the_five_module_specs(self):
        """The byte-identity half."""
        assert prompt_spec_hash() == PROMPT_SPEC_HASHES[PROMPT_SCHEMA_VERSION]

    def test_every_registered_variant_changes_the_hash(self):
        """        Parametrised over every registered variant rather than kuhn's
        relabel alone: the claim the provenance column
        rests on is that any variant arm is distinguishable from canonical,
        and one game's variant cannot establish it for the other four. The
        count is the whole registry, five games times two probe types, and it
        is written against ``GAME_KEYS``/``PROBE_TYPES`` rather than as a
        literal so that the registry losing a variant reddens here instead of
        quietly shrinking what this test covers. Goofspiel's payoff scale is
        the case that makes the canonical-inequality non-trivial: its
        instruction, vocabulary, question and exemplar are byte-identical to
        canonical, so its digest moves only because ``display_scale`` enters
        the payload."""
        from lmopt.prompting.probe_variants import (
            GAME_KEYS,
            PAYOFF_SCALE_SPECS,
            PROBE_TYPES,
            RELABEL_SPECS,
        )

        canonical = prompt_spec_hash()
        variants = {f"relabel/{g}": s for g, s in RELABEL_SPECS.items()} | {
            f"payoff_scale/{g}": s for g, s in PAYOFF_SCALE_SPECS.items()
        }
        assert len(variants) == len(GAME_KEYS) * len(PROBE_TYPES)
        digests = {}
        for name, spec in variants.items():
            digest = prompt_spec_hash(spec)
            assert digest != canonical, f"{name} hashes to the canonical digest"
            digests[name] = digest
        # ... and mutually distinct, so a reader can tell which variant an arm
        # rendered, not merely that it was not canonical.
        assert len(set(digests.values())) == len(digests)

    def test_variants_of_different_games_hash_differently(self):
        """Two games' relabels must not collide, and neither may be merely the
        one-game digest `game_spec_hash` already provides.
        """
        from lmopt.prompting.probe_variants import RELABEL_SPECS
        from lmopt.prompting.spec import game_spec_hash

        kuhn = prompt_spec_hash(RELABEL_SPECS["kuhn"])
        leduc = prompt_spec_hash(RELABEL_SPECS["leduc"])
        assert kuhn != leduc
        # ... and neither is merely the one-game digest.
        assert kuhn != game_spec_hash(RELABEL_SPECS["kuhn"])

    @pytest.mark.parametrize(
        "variant_game,drifted_game", [("kuhn", "leduc"), ("leduc", "kuhn")]
    )
    def test_a_variant_digest_still_covers_the_other_four_games(
        self, monkeypatch, variant_game, drifted_game
    ):
        """SUBSTITUTION by `spec.game`, not replacement and not by position:
        the digest a variant arm logs must be the same five-game payload with
        one game swapped, because that is what a `PROMPT_SPEC_HASHES` entry
        means and what makes the base-arm equality check meaningful.

        Two mutations, one per parameter pair, both green under every
        other test in this class, which is why this test exists
        and why it is parametrised:

        - hash the variant alone (a one-game payload): every variant still
          differs from canonical and from every other variant, but no drift in
          another game moves it. Both pairs redden.
        - substitute by position (`specs[0] = variant`): kuhn happens to be at
          index 0, so the kuhn case is unaffected; the LEDUC variant overwrites
          kuhn's slot instead, and a kuhn drift then leaves the digest
          unmoved. The ("leduc", "kuhn") pair reddens.
        """
        import dataclasses
        import lmopt.prompting.spec as spec_mod
        from lmopt.prompting.probe_variants import RELABEL_SPECS

        variant = RELABEL_SPECS[variant_game]
        before = prompt_spec_hash(variant)
        drifted = tuple(
            (
                dataclasses.replace(s, belief_question=s.belief_question + " (drifted)")
                if s.game == drifted_game
                else s
            )
            for s in spec_mod._ALL_SPECS
        )
        monkeypatch.setattr(spec_mod, "_ALL_SPECS", drifted)
        assert prompt_spec_hash(variant) != before

    def test_a_spec_for_an_unregistered_game_is_refused(self):
        """No silent fallbacks: a spec whose `game` matches none of
        the five would otherwise hash to the canonical digest, which is the
        fabricated provenance this argument exists to prevent."""
        import dataclasses
        from lmopt.prompting.spec import KUHN_SPEC

        alien = dataclasses.replace(KUHN_SPEC, game="chess")
        with pytest.raises(ValueError, match="unregistered game 'chess'") as exc:
            prompt_spec_hash(alien)
        # The bare `chess` substring would also appear in the ValueError
        # `list.index` raises when the guard is deleted, so a bare substring
        # match would pass under its own named mutation. Asserting the guard's own wording, and that it names the five games
        # a caller may substitute, isolates it.
        assert "kuhn" in str(exc.value)


_BARE_KEY_FENCED = (
    "Here is my belief:\n```json\n"
    '{"1":0.25,"2":0.25,3:0.25,"4":0.25}'
    "\n```\nMy reasoning follows: the opponent has played twice."
)
_BARE_KEY_GROUP = '{"1":0.25,"2":0.25,3:0.25,"4":0.25}'


class TestSalvageBeliefObject:
    """`salvage_belief_object` recovers a belief whose only defect is a bare
    integer key, by inserting quote characters and changing nothing else.
    """

    def test_the_measured_shape_is_recovered_with_only_quotes_added(self):
        salvaged = spec_module.salvage_belief_object(_BARE_KEY_FENCED)
        assert salvaged is not None
        assert salvaged.replace('"', "") == _BARE_KEY_GROUP.replace('"', "")
        assert json.loads(salvaged) == {"1": 0.25, "2": 0.25, "3": 0.25, "4": 0.25}

    def test_a_trailing_zero_value_is_left_exactly_as_written(self):
        text = '{"1":0.20,"2":0.30,3:0.30,"4":0.20}'
        salvaged = spec_module.salvage_belief_object(text)
        assert salvaged is not None
        assert "0.20" in salvaged and "0.30" in salvaged

    def test_every_key_bare_is_recovered(self):
        salvaged = spec_module.salvage_belief_object("{1: 0.5, 2: 0.5}")
        assert salvaged is not None
        assert json.loads(salvaged) == {"1": 0.5, "2": 0.5}

    def test_the_recovered_string_reaches_the_production_parser(self):
        """Reachability anchor: the strict goofspiel `parse_belief`, the
        function `act()` calls, turns the recovered string into the belief
        the model stated."""
        from lmopt.prompting import goofspiel

        parse_belief = goofspiel.make_fns(4)["parse_belief"]
        assert parse_belief(_BARE_KEY_FENCED, "x") is None
        result = spec_module.parse_belief_salvaging(parse_belief, _BARE_KEY_FENCED, "x")
        assert result is not None
        belief, kind = result
        assert kind == spec_module.BELIEF_SALVAGE_KEY_QUOTED == "key_quoted"
        assert json.loads(belief) == {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param(
                '{"reasoning": "maybe {1: 0.4, 2: 0.6} fits, and',
                id="truncated-object-drops-the-unterminated-guard",
            ),
            pytest.param(
                "The running score is {1: 3, 2: 4} so far.",
                id="narrative-values-drop-the-unit-interval-guard",
            ),
            pytest.param(
                "Opponent has played {1: 1} already.",
                id="single-pair-drops-the-two-pair-minimum",
            ),
            pytest.param(
                '{"1": 0.5, 2: 0.5} and later {3: 0.4, 4: 0.6}',
                id="two-groups-drops-the-single-group-guard",
            ),
            pytest.param(
                '{1: "a", 2: 0.5}',
                id="non-numeric-value-drops-the-numeric-guard",
            ),
            pytest.param(
                '{"1":0.25,"1":0.25,2:0.25,"3":0.25}',
                id="duplicate-quoted-key-drops-the-uniqueness-guard",
            ),
            pytest.param(
                '{"1":0.5,1:0.5,"3":0.0,"4":0.0}',
                id="bare-and-quoted-same-key-drops-the-uniqueness-guard",
            ),
            pytest.param(
                '{01: 0.5, "2": 0.5}',
                id="leading-zero-key-drops-the-bare-key-shape",
            ),
        ],
    )
    def test_text_that_is_not_a_bare_key_belief_is_refused(self, text):
        """Each case is a shape the recovery must refuse; the case id names
        the guard whose deletion turns it green-to-red. With a guard-free
        quoting prototype, the narrative case became a belief of
        {0: .43, 1: .57} and the truncated case {0: .4, 1: .6}, values the
        model never stated as a belief."""
        assert spec_module.salvage_belief_object(text) is None

    def test_a_valid_group_is_not_salvage_material(self):
        assert spec_module.salvage_belief_object('{"1": 0.25, "2": 0.75}') is None
        assert spec_module.salvage_belief_object("I have no idea") is None

    def test_extract_belief_object_is_unchanged_by_the_recovery(self):
        """Standing invariant: the strict extractor is independent of the
        recovery, including the arithmetic shape
        it already repairs and the bare-key shape it still rejects."""
        arithmetic = (
            '{\n  "1": 0.25,\n  "2": 0.25,\n  "3": 0.15,\n  "4": 0.15,\n'
            '  "5": 0.10,\n  "6": 1.00 - (0.25 + 0.25 + 0.15 + 0.15 + 0.10)\n}'
        )
        expected = {
            '{"1": 0.25, "2": 0.75}': '{"1": 0.25, "2": 0.75}',
            "I have no idea": None,
            arithmetic: (
                '{\n  "1": 0.25,\n  "2": 0.25,\n  "3": 0.15,\n  "4": 0.15,\n'
                '  "5": 0.10,\n  "6": 0.09999999999999998}'
            ),
            '{"1": 0.5, "2": some_undefined_name}': None,
            _BARE_KEY_GROUP: None,
            _BARE_KEY_FENCED: None,
            "```json\n"
            + _BARE_KEY_GROUP
            + "\n```\nOther cards could be {2, 3, 4}.": None,
            '{"reasoning": "maybe {1: 0.4, 2: 0.6} fits, and': None,
        }
        for text, want in expected.items():
            assert extract_belief_object(text) == want, text

    def test_the_arithmetic_shape_is_not_a_salvage_case(self):
        """The Qwen liarsdice shape is recovered by the existing arithmetic
        repair inside `extract_belief_object`, so it must reach the belief
        without the recovery and the recovery must not claim it."""
        text = (
            '{"1":0.25,"2":0.25,"3":0.15,"4":0.15,"5":0.10,'
            '"6": 1.00 - (0.25+0.25+0.15+0.15+0.10)}'
        )
        assert extract_belief_object(text) is not None
        assert spec_module.salvage_belief_object(text) is None


def test_a_completion_cut_off_at_the_token_cap_is_never_salvaged_by_the_helper():
    """`parse_belief_salvaging` refuses a `finish_reason == "length"`
    completion itself, so no caller can forget the guard."""
    from lmopt.prompting import goofspiel

    parse_belief = goofspiel.make_fns(4)["parse_belief"]
    assert (
        spec_module.parse_belief_salvaging(
            parse_belief, _BARE_KEY_FENCED, "x", "length"
        )
        is None
    )
    assert (
        spec_module.parse_belief_salvaging(parse_belief, _BARE_KEY_FENCED, "x", "stop")
        is not None
    )
    assert (
        spec_module.parse_belief_salvaging(parse_belief, _BARE_KEY_FENCED, "x", None)
        is not None
    )


@pytest.mark.parametrize(
    "expr",
    ["0.5 * 1" + "0" * 400, "-" * 3000 + "1", "+".join(["1"] * 20000)],
    ids=["overflow", "unary_chain", "long_chain"],
)
def test_safe_arith_eval_returns_none_on_overflow_and_deep_nesting(expr):
    """
    an arithmetic value too large for a float, or nested too deeply to parse or walk, reads as
    None, the documented refusal, rather than raising out of the belief parse
    """
    from lmopt.prompting import spec as spec_module

    assert spec_module._safe_arith_eval(expr) is None


def test_salvage_refuses_a_digit_key_int_rejects():
    """
    a quoted key made of a digit character ``int`` rejects (superscript two) makes the salvage
    return None, as its fail-closed contract says, rather than raising ValueError; every key
    ``int`` accepts is canonicalised as before
    """
    from lmopt.prompting import spec as spec_module

    assert spec_module.salvage_belief_object('{"²": 0.5, 1: 0.5}') is None


@pytest.mark.parametrize("scale", [0, -2, 1.5, True])
def test_game_prompt_spec_refuses_a_non_positive_or_non_int_display_scale(scale):
    """
    ``GamePromptSpec`` refuses a display scale that is not a positive int, so no variant can
    render zeroed or fractional magnitudes under a payoff-scale label
    """
    from lmopt.prompting.spec import KUHN_SPEC

    with pytest.raises(ValueError, match="display_scale"):
        dataclasses.replace(KUHN_SPEC, display_scale=scale)
