"""Full-tree prompt-fidelity detector.

The detector rests on three requirements:

1. It enumerates the full tree per registered variant, not a sample of
   landed rows. A sample covers too little of the tree (liarsdice 32 of
   24,576 info-sets = 0.13%; darkhex 5 of 134 = 3.7%) to support
   game-level percentages.
2. Harm is judged by differing best-response value, computed via the
   imported ``BestResponsePolicy``. A criterion of "colliding info-sets have
   different CFR average policies" cannot fire under a uniform reference --
   darkhex is uniform on 121 of 134 (both-seat) info-sets (90.3%) -- while
   best-response value is non-degenerate wherever the states genuinely
   differ, uniform average policy or not.
3. Injectivity alone is structurally blind to one-to-*wrong* mapping (a
   confidently-mislabelled Leduc card is still injective), so a separate
   fidelity assertion checks rendered content against ground truth.

On the canonical 2x2 Dark Hex board, dropping the info-state's
move-count/history lines produces 16 collision groups that are nonetheless
harmless: every colliding pair has identical best-response value and
identical best-response action (checked at 500 and 5000 CFR+ iterations, so
this is not an under-converged reference). At 2x2 (4 cells total) the
move-count is a redundant encoding of "how many hidden opponent stones
remain among the unknown cells", so no decision-relevant information is
lost by dropping it at this size. The depth-axis rectangles
`darkhex_2x3`/`darkhex_3x2` did have real best-response harm from the same
dropped lines (20/146 and 32/171 harmful groups). The prompt renders the
move-count and own-move-history lines for every board size via one shared
parametric factory (``darkhex.py``'s ``make_fns``), justified at the larger
sizes on harm grounds and at the canonical size on faithful-serialisation
grounds. Leduc shows the criterion firing: before its rank-space and
full-history prompt fixes, 70 of its collision groups were harmful; after
them, 0 of 144 (see
``test_leduc_harmful_collisions_fixed_and_residual_collisions_are_suit_only``).
"""

import re
from collections import Counter

import pytest

pyspiel = pytest.importorskip("pyspiel")

from lmopt.games.loaders import load_game
from lmopt.prompting import get_game_fns_for_game
from lmopt.prompting.fidelity import (
    _br_values_differ,
    collision_report,
    enumerate_infosets,
    label_agreement_report,
    self_parse_report,
)


def test_enumerate_covers_the_whole_tree_not_a_sample():
    """The enumeration must cover the whole tree, not a sample of landed rows.
    These counts are P0 decision nodes only -- the nodes where build_prompt is
    actually called.

    Do not substitute both-player counts (kuhn 12 / leduc 936 / darkhex 134),
    which are ~2x these because they count both seats. Prompt fidelity
    concerns P0's decision nodes only.
    """
    assert len(enumerate_infosets(load_game("kuhn"))) == 6
    assert len(enumerate_infosets(load_game("leduc"))) == 468
    assert len(enumerate_infosets(load_game("darkhex"))) == 33


def test_harm_criterion_fires_for_a_hand_picked_genuinely_different_pair():
    """The harm criterion must be non-vacuous: it has to actually fire
    somewhere, or "best-response value" would be exactly as blind as v1's
    average-policy check.

    Asserting that ``collision_report("leduc").harmful`` is non-empty would
    not do: with a truthful, rank-space Leduc prompt, indices 2k and 2k+1
    are *meant* to render identically (Leduc is suit-symmetric), so the
    residual collisions have coinciding BR values and ``harmful`` is
    correctly empty. Testing ``_br_values_differ`` directly against a
    hand-picked pair that owes nothing to any particular game's prompt
    survives every prompt fix: Kuhn's two extreme info-sets, holding the
    worst card ("0", a Jack) versus the best ("2", a King) at the very first
    decision, are settled game theory with starkly different value -- BR
    value -1.0 (Jack, best action Pass) vs +1.168 (King, best action Bet).
    """
    game = load_game("kuhn")
    assert _br_values_differ(game, ["0", "2"], player=0), (
        "harm criterion is vacuous: Kuhn's Jack vs King opening info-sets "
        "have starkly different best-response value and action, and must "
        "be reported as differing"
    )


def test_leduc_prompt_states_the_true_card_rank():
    """An injectivity-only check cannot see a prompt where a Jack is called
    'Queen' and a Queen 'King': distinct-but-wrong is still injective.
    rank = card_index // 2, verified against OpenSpiel showdown returns.

    The rank-space prompt in src/lmopt/prompting/leduc.py keys private and
    public card names by rank = index // 2, not the raw 0-5 index; with it,
    0 of 468 P0 info-sets misstate the rank (390 of 468 without it).
    """
    game = load_game("leduc")
    RANK = {0: "Jack", 1: "Jack", 2: "Queen", 3: "Queen", 4: "King", 5: "King"}
    mapping = enumerate_infosets(game)
    assert (
        len(mapping) == 468
    ), f"expected 468 P0 info-sets from the full-tree walk, got {len(mapping)}"
    bad = []
    for info, prompt in mapping.items():
        m = re.search(r"Private: (\d+)", info)
        if not m:
            continue
        expected = RANK[int(m.group(1))]
        if f"private card: {expected}" not in prompt:
            bad.append((info, expected))
    assert not bad, (
        f"{len(bad)} leduc info-sets misstate or omit the card rank "
        f"(baseline without it: 390 of 468 P0 info-sets)"
    )


def test_leduc_prompt_states_the_true_betting_history():
    """Sibling of test_leduc_prompt_states_the_true_card_rank above, for the
    history rather than the card: collision-freedom and a handful of unit
    fixtures cannot see a prompt that is uniformly wrong but still injective
    and internally consistent -- this module docstring's point 3, applied to
    the betting-history line rather than the card-name line.

    Independent of production, deliberately: `_expected_history_line` below
    is a from-scratch parallel implementation of the contract
    `_describe_full_history` is supposed to satisfy -- fresh regex over the
    raw info-set string (not `_parse_field`), its own hardcoded verb map
    (not `_describe_history`'s `int_mapping`), and its own round-walk loop
    (not `_current_round_seq`/`_describe_full_history`). This test imports
    nothing from `lmopt.prompting.leduc` at all, unlike this file's suit-only
    collision test, which re-uses the production `_rank`/`_parse_field`.

    Reddens under three mutations to
    `_describe_full_history`, none of which any unit fixture or the
    collision-report test catches, because all three are collision-
    preserving (every info-set still maps to some consistent prompt) and
    only wrong relative to independently-known ground truth:
    - MUT D: round labels off by one (`f"Round {r+1}: ..."` while the content
      rendered is still round r's true sequence) -- a Round-2 node renders
      "Round 2: <round 1's real actions> Round 3: <round 2's real actions>".
    - MUT G: swap the two rounds' rendered contents with labels left correct
      -- a Round-2 node renders "Round 1: <round 2's actions> Round 2:
      <round 1's actions>".
    - MUT E: reverse the round-iteration order (content and labels both
      individually correct, sequence wrong) -- a Round-2 node renders
      "Round 2: ... Round 1: ..." instead of "Round 1: ... Round 2: ...".
    """
    game = load_game("leduc")
    mapping = enumerate_infosets(game)
    assert (
        len(mapping) == 468
    ), f"expected 468 P0 info-sets from the full-tree walk, got {len(mapping)}"

    verbs = {"0": "folded", "1": "checked/called", "2": "bet/raised"}

    def seq_desc(seq: str) -> str:
        seq = seq.strip()
        if not seq:
            return "No actions yet this round."
        toks = seq.split()
        parts = [f"Player {i % 2} {verbs[t]}" for i, t in enumerate(toks)]
        return ". ".join(parts) + "."

    def expected_history_line(info: str) -> str:
        rm = re.search(r"\[Round\s+(\d+)\]", info)
        current_round = int(rm.group(1)) if rm else 1
        if current_round <= 1:
            m = re.search(r"\[Round1:\s*([^\]]*)\]", info)
            seq = m.group(1) if m else ""
            return f"- History: {seq_desc(seq)}"
        parts = []
        for r in range(1, current_round + 1):
            m = re.search(r"\[Round" + str(r) + r":\s*([^\]]*)\]", info)
            seq = (m.group(1) if m else "").strip()
            if seq:
                parts.append(f"Round {r}: {seq_desc(seq)}")
            else:
                parts.append(f"Round {r}: no actions yet.")
        return "- History: " + " ".join(parts)

    mismatches = []
    for info, prompt in mapping.items():
        expected = expected_history_line(info)
        actual = next(
            (ln for ln in prompt.splitlines() if ln.startswith("- History")), None
        )
        if actual != expected:
            mismatches.append((info, expected, actual))

    assert not mismatches, (
        f"{len(mismatches)}/{len(mapping)} leduc info-sets render a "
        f"'- History' line that disagrees with the info-set's own "
        f"independently-reconstructed betting history: "
        f"{mismatches[:3]}"
    )


CLEAN_GAMES = ["kuhn", "goofspiel", "liarsdice"]


# Correspondence-CHECK: prompt-infoset
@pytest.mark.parametrize("game_name", CLEAN_GAMES)
def test_clean_games_have_no_harmful_collisions(game_name):
    """These three are injective on the full tree, P0-scoped (kuhn 6,
    goofspiel_4 81, goofspiel_5 1062, liarsdice 12288 -- all 1:1; not the
    both-seat figures ``test_enumerate_covers_the_whole_tree_not_a_sample``'s
    docstring warns against). Guards against regression.

    ``not rep.harmful`` alone is satisfied vacuously here -- these games have
    zero collisions, so ``harmful`` is empty by construction and
    ``_best_response_policy`` is never even called. So the actual
    injectivity claim is asserted too.
    """
    rep = collision_report(game_name)
    assert rep.n_prompts == rep.n_infosets, (
        f"{game_name}: expected full injectivity ({rep.n_prompts} prompts "
        f"for {rep.n_infosets} info-sets)"
    )
    assert not rep.harmful, f"{game_name}: {len(rep.harmful)} harmful collisions"


@pytest.mark.slow  # ~94s call (CFR+500 solve + full
# BestResponsePolicy construction over Leduc's collision groups). No shared
# fixture in this module, so marking only this test is isolated.
def test_leduc_harmful_collisions_fixed_and_residual_collisions_are_suit_only():
    """The rank-space card names and full-history line drive
    ``collision_report("leduc")``'s ``harmful`` set to zero. Without either
    fix: 468 info-sets / 129 prompts / 70 harmful; with the rank-space fix
    only: 468/90/35; with both: 468/144/0.

    Leduc is deliberately not asserted fully injective here (unlike
    ``test_clean_games_have_no_harmful_collisions``'s CLEAN_GAMES): 144
    prompts for 468 info-sets remain, all of them multi-member collision
    groups (zero singleton prompt groups). Rather than trust "not
    harmful" alone, this checks why each residual group collides: every one
    must differ only in the raw Private/Public card index (i.e. suit), never
    in rank, the Round1 sequence, the Round2 sequence, the round number, or
    the player -- the concrete, checkable form the "Leduc is suit-symmetric"
    premise the whole rank-space prompt design rests on must take if that
    premise is actually true, rather than merely asserted.
    """
    from lmopt.prompting.leduc import (
        _parse_field,
        _parse_public_card,
        _parse_round,
        _rank,
        parse_own_card,
    )

    rep = collision_report("leduc")
    assert rep.n_infosets == 468, f"expected 468 info-sets, got {rep.n_infosets}"
    assert rep.n_prompts == 144, f"expected 144 distinct prompts, got {rep.n_prompts}"
    assert (
        not rep.harmful
    ), f"{len(rep.harmful)} harmful collision groups: {list(rep.harmful)[:2]}"
    # Pin both n_prompts and non-emptiness of rep.collisions -- without these the suit-only loop below is quantifier-collapse-vacuous if a future
    # regression drove collisions to 0 (harmless by construction, no groups to check).
    assert rep.collisions, "expected leduc to still have residual (harmless) collisions"
    assert (
        len(rep.collisions) == 144
    ), f"expected 144 collision groups, got {len(rep.collisions)}"

    def _non_suit_features(info: str) -> tuple:
        priv = parse_own_card(info)
        pub = _parse_public_card(info)
        return (
            _rank(priv) if priv is not None else None,
            _rank(pub) if pub is not None else None,
            _parse_field(info, "Round1"),
            _parse_field(info, "Round2"),
            _parse_round(info),
            _parse_field(info, "Player"),
        )

    non_suit_only = [
        infos
        for infos in rep.collisions.values()
        if len({_non_suit_features(i) for i in infos}) > 1
    ]
    assert not non_suit_only, (
        f"{len(non_suit_only)} collision group(s) differ by more than suit "
        f"(rank/Round1/Round2/round/player) -- the suit-symmetry premise "
        f"this design rests on would be violated: {non_suit_only[:1]}"
    )


def test_darkhex_is_now_injective():
    """darkhex.py's shared parametric ``make_fns`` renders the move-count and
    own-move-history lines that follow the board rows in the info-state
    string, inside ``build_prompt`` itself (``_parse_board`` only reads the
    board rows). Without those lines the canonical board gives 17 prompts
    for 33 info-sets.

    darkhex is injective on the full P0 tree, the same standard
    kuhn/goofspiel/liarsdice meet (``CLEAN_GAMES``):
    ``collision_report("darkhex")`` gives n_infosets=33, n_prompts=33,
    0 collisions, 0 harmful (the collisions without those lines are
    0-harmful by best-response value/action too, so the injectivity
    assertion is the one with teeth).

    ``n_infosets`` is pinned to 33: without it, a
    walk regression that visits nothing would report ``n_infosets=0``,
    ``n_prompts=0``, both assertions below trivially true, and this test
    would read green on an empty tree -- exactly the failure mode
    ``_LABEL_AGREEMENT_EXPECTED_N_INFOSETS`` guards against elsewhere in this
    file.
    """
    rep = collision_report("darkhex")
    assert rep.n_infosets == 33, f"expected 33 info-sets, got {rep.n_infosets}"
    assert rep.n_prompts == rep.n_infosets, (
        f"darkhex still has {rep.n_infosets - rep.n_prompts} colliding info-sets "
        f"({rep.n_prompts} prompts for {rep.n_infosets} info-sets)"
    )
    assert not rep.harmful, f"darkhex: {len(rep.harmful)} harmful collisions"


# Standing invariant: action_labels() must agree with action_suffix() and
# legal_actions() over the full tree of every tractable registered variant
# (the class of defect includes darkhex candidate schemes and liarsdice
# regex-bleed candidates). darkhex_3x3 is excluded: no full-tree walk is
# tractable for it (~10^7 states; the same exclusion loaders.py's docstring
# and this file's collision_report tests make). Per-variant cost is seconds
# (kuhn_deep 26 info-sets, leduc_deep 9,828, goofspiel_6 17,241,
# liarsdice_1d8 262,144). darkhex_2x3/darkhex_3x2 are not registered, so
# the darkhex family's depth axis has no coverage here.
_LABEL_AGREEMENT_GAMES = [
    "kuhn",
    "leduc",
    "goofspiel",
    "liarsdice",
    "darkhex",
    "goofspiel_5",
    "liarsdice_1d8",
    "goofspiel_6",
    "kuhn_deep",
    "leduc_deep",
]

# Pinned per-variant counts, not just "> 0". Without this, label_agreement_report(game_name, player=7) (an out-of-range player
# - no state ever has current_player() == 7) returns n_infosets=0 with
# ok=True, so a walk regression that visits nothing reads as a clean pass;
# rep.n_infosets was computed and discarded, never asserted.
_LABEL_AGREEMENT_EXPECTED_N_INFOSETS = {
    "kuhn": 6,
    "leduc": 468,
    "goofspiel": 81,
    "liarsdice": 12288,
    "darkhex": 33,
    "goofspiel_5": 1062,
    "liarsdice_1d8": 262144,
    "goofspiel_6": 17241,
    "kuhn_deep": 26,
    "leduc_deep": 9828,
}


# Correspondence-CHECK: prompt-infoset
@pytest.mark.parametrize("game_name", _LABEL_AGREEMENT_GAMES)
def test_action_labels_agree_with_action_suffix_and_legal_actions(game_name):
    """    Also covers a permutation of the mapping (e.g. two labels swapped
    between two actions), which preserves both checks above -- the swapped
    labels are still verbatim in action_suffix, and the key SET is
    unchanged, only which key maps to which value moved. Without this check
    an injected permutation stays green for leduc/goofspiel/liarsdice.
    label_action_mismatch closes it: every action_labels() pair must
    round-trip through the game's own parse_action, an independent
    cross-CHECK on the label source (parse_action is never used as the
    source -- see LabelAgreementReport's docstring for "source versus
    check").

    n_infosets is pinned to the known count, not merely
    asserted non-zero, so a walk regression (not just an empty walk) fails
    here too.
    """
    rep = label_agreement_report(game_name)
    assert rep.n_infosets == _LABEL_AGREEMENT_EXPECTED_N_INFOSETS[game_name], (
        f"{game_name}: expected {_LABEL_AGREEMENT_EXPECTED_N_INFOSETS[game_name]} "
        f"info-sets from the full-tree walk, got {rep.n_infosets}"
    )
    assert not rep.label_not_in_suffix, (
        f"{game_name}: {len(rep.label_not_in_suffix)} info-set(s) have an "
        f"action_labels() literal missing from action_suffix()'s text: "
        f"{dict(list(rep.label_not_in_suffix.items())[:3])}"
    )
    assert not rep.label_set_mismatch, (
        f"{game_name}: {len(rep.label_set_mismatch)} info-set(s) have "
        f"action_labels() keys != legal_actions(): "
        f"{dict(list(rep.label_set_mismatch.items())[:3])}"
    )
    assert not rep.label_action_mismatch, (
        f"{game_name}: {len(rep.label_action_mismatch)} info-set(s) have an "
        f"action_labels() pair that does not round-trip through parse_action "
        f"(a permutation of the mapping): "
        f"{dict(list(rep.label_action_mismatch.items())[:3])}"
    )


# Standing coverage over the rest of the registered variants, so that "full
# tree per registered variant" holds beyond the five canonical games.
# ``lmopt.games.loaders.GAMES`` registers ten keys; darkhex_3x3 is
# deliberately excluded below -- it has no full CFR+ reference and cannot
# have one (TabularPolicy's full-tree enumeration MemoryErrors, ~6 GB / 17.6s,
# per loaders.py's docstring and scripts/validate_cfr.py's GAME_SPECS
# exclusion), so collision_report cannot run against it at all.
def test_registered_variant_goofspiel_5_is_injective():
    """Depth-axis variant: 1062 P0 info-sets, 1062 prompts, 0 collisions --
    same clean standard as goofspiel_4. Fast (~0.1s), so not marked slow."""
    rep = collision_report("goofspiel_5")
    assert rep.n_prompts == rep.n_infosets, (
        f"goofspiel_5: expected full injectivity ({rep.n_prompts} prompts "
        f"for {rep.n_infosets} info-sets)"
    )
    assert not rep.harmful, f"goofspiel_5: {len(rep.harmful)} harmful collisions"


def test_registered_variant_goofspiel_6_is_injective():
    """Depth-axis variant at the CFR-tractability frontier -- same clean standard as
    goofspiel_4/goofspiel_5."""
    rep = collision_report("goofspiel_6")
    assert rep.n_prompts == rep.n_infosets, (
        f"goofspiel_6: expected full injectivity ({rep.n_prompts} prompts "
        f"for {rep.n_infosets} info-sets)"
    )
    assert not rep.harmful, f"goofspiel_6: {len(rep.harmful)} harmful collisions"


def test_registered_variant_liarsdice_1d8_is_injective():
    """Depth-axis variant at the analogous CFR-tractability frontier for
    Liar's Dice -- same clean
    standard as canonical liarsdice. Own die value + full bid history make
    every info-set's rendered prompt unique by construction (no vocabulary
    collision), so (unlike a genuinely colliding game) this never needs to
    call ``_br_values_differ``/``solve_cfr``, and it runs fast (no CFR
    budget is registered for ``liars_dice`` at a 262,144-info-set scale)."""
    rep = collision_report("liarsdice_1d8")
    assert rep.n_prompts == rep.n_infosets, (
        f"liarsdice_1d8: expected full injectivity ({rep.n_prompts} prompts "
        f"for {rep.n_infosets} info-sets)"
    )
    assert not rep.harmful, f"liarsdice_1d8: {len(rep.harmful)} harmful collisions"


def test_registered_variant_kuhn_deep_is_injective():
    """Kuhn poker's own depth axis, loaded via universal_poker at
    numRanks=13. Unlike goofspiel_6/liarsdice_1d8, numRanks=13 is an engine
    STRUCTURAL ceiling (universal_poker segfaults above 13 ranks), not a
    CFR-tractability frontier -- but the injectivity standard is the same:
    26 P0 info-sets, 26 prompts, 0 collisions, same clean standard as
    canonical kuhn."""
    rep = collision_report("kuhn_deep")
    assert rep.n_prompts == rep.n_infosets, (
        f"kuhn_deep: expected full injectivity ({rep.n_prompts} prompts "
        f"for {rep.n_infosets} info-sets)"
    )
    assert not rep.harmful, f"kuhn_deep: {len(rep.harmful)} harmful collisions"


def test_universal_poker_cfr_budget_refuses_a_shape_it_was_not_justified_for():
    """The 30-iteration "universal_poker" budget in fidelity.py's _CFR_ITERATIONS is trustworthy
    only because leduc_deep's residual collisions are suit-only by
    construction, an argument that itself depends on numSuits == 2 -- but the
    budget is keyed by short_name, shared by every variant dispatched through
    universal_poker, so a future variant with more suits would silently
    inherit it. This is fast (no CFR runs -- the guard fires before
    solve_cfr is ever called) and needs no CFR_ITERATIONS registration of its
    own, unlike a real collision_report call.
    """
    from lmopt.prompting.fidelity import _best_response_policy

    params = {
        "betting": "limit",
        "numPlayers": 2,
        "numRounds": 2,
        "numRanks": 3,
        "numSuits": 3,
        "numHoleCards": 1,
        "numBoardCards": "0 1",
        "blind": "1 1",
        "raiseSize": "2 4",
        "maxRaises": "2 2",
        "firstPlayer": "1 1",
    }
    game = pyspiel.load_game("universal_poker", params)
    with pytest.raises(ValueError, match="numSuits"):
        _best_response_policy(str(game), "universal_poker", 0)


@pytest.mark.slow  # ~366s (30-iteration CFR+ reference
# solve against leduc_deep's 9828-info-set tree, ~11-13s/iter -- see
# fidelity.py's _CFR_ITERATIONS["universal_poker"] comment for why 30 is
# trustworthy despite being far short of leduc_poker-parity convergence).
def test_registered_variant_leduc_deep_residual_collisions_are_suit_only():
    """Leduc's own depth axis, loaded via universal_poker at numRanks=13. Unlike
    kuhn_deep, leduc_deep is not fully injective -- same as native leduc
    (``test_leduc_harmful_collisions_fixed_and_residual_collisions_are_suit_only``
    above), because leduc_deep.py's ``parse_own_card``/``_parse_public_card``
    discard the suit character exactly like leduc.py's own ``_rank`` do, and
    Leduc's hand-ranking rule (pair beats no-pair, else higher rank) never
    depends on suit. This test mirrors that native-leduc test's exact
    reasoning rather than asserting a stronger (full-injectivity) claim than
    native leduc itself makes: 9828 info-sets, 2574 prompts, 2574 collision
    groups, 0 harmful.
    """
    from lmopt.prompting.leduc_deep import (
        _parse_field,
        _parse_public_card,
        _parse_round,
        parse_own_card,
    )

    rep = collision_report("leduc_deep")
    assert rep.n_infosets == 9828, f"expected 9828 info-sets, got {rep.n_infosets}"
    assert rep.n_prompts == 2574, f"expected 2574 distinct prompts, got {rep.n_prompts}"
    assert (
        not rep.harmful
    ), f"leduc_deep: {len(rep.harmful)} harmful collision groups: {list(rep.harmful)[:2]}"
    # Pin both n_prompts and non-emptiness of rep.collisions -- without these
    # the suit-only loop below is quantifier-collapse-vacuous if a future
    # regression drove collisions to 0 (harmless by construction, no groups
    # to check), mirroring the native leduc test.
    assert (
        rep.collisions
    ), "expected leduc_deep to still have residual (harmless) collisions"
    assert (
        len(rep.collisions) == 2574
    ), f"expected 2574 collision groups, got {len(rep.collisions)}"

    def _non_suit_features(info: str) -> tuple:
        priv = parse_own_card(
            info
        )  # already a rank character, no index translation needed
        pub = _parse_public_card(info)
        return (
            priv,
            pub,
            _parse_field(info, "Sequences"),
            _parse_round(info),
            _parse_field(info, "Player"),
        )

    non_suit_only = [
        infos
        for infos in rep.collisions.values()
        if len({_non_suit_features(i) for i in infos}) > 1
    ]
    assert not non_suit_only, (
        f"{len(non_suit_only)} collision group(s) differ by more than suit "
        f"(rank/Sequences/round/player) -- the suit-symmetry premise this "
        f"design rests on would be violated: {non_suit_only[:1]}"
    )


# Standing invariant: a verbatim echo of the action/belief question must not
# itself parse as a valid answer, over the full tree of every tractable
# registered variant -- same _LABEL_AGREEMENT_GAMES list
# `label_agreement_report` walks (darkhex_3x3 excluded, same reason:
# intractable full-tree enumeration).
#
# Without the defences, every P0 info-set in every variant self-parses on
# Both action and belief (a verbatim echo parses as a belief, and as a
# "perfect" action on kuhn/leduc via keyword-in-echo):
#
#   game          n_infosets  action_self_parses  belief_self_parses
#   kuhn                   6                   6                   6
#   leduc                468                 468                 468
#   goofspiel             81                  81                  81
#   liarsdice          12288               12288               12288
#   darkhex               33                  33                  33
#   darkhex_2x3         2053                2053                2053
#   darkhex_3x2         2323                2323                2323
#   goofspiel_5         1062                1062                1062
#
# Placeholder-keyed belief exemplars drive belief_self_parses to 0 on every
# variant. Multi-label rejection plus the liarsdice legality gate drive
# action_self_parses to 0 everywhere except liarsdice, which stays at
# 6144/12288 (exactly half): every remaining case has
# |action_labels(info)| == 1 (the max-bid, CHALLENGE-only corner).
# `self_parse_report`'s |A|=1 exclusion (see `SelfParseReport`'s docstring
# in fidelity.py -- these are mathematically unavoidable, not a residual
# defect) drops liarsdice to 0 too.
_SELF_PARSE_GAMES = _LABEL_AGREEMENT_GAMES
_SELF_PARSE_EXPECTED_N_INFOSETS = _LABEL_AGREEMENT_EXPECTED_N_INFOSETS

# `n_infosets` alone cannot detect a regression that silently widens the |A|=1 exclusion (e.g.
# `>= 2` drifting to `>= 4` in fidelity.py) -- every excluded info-set is
# still "visited" by the walk and still counted in `n_infosets`, so a wider
# exclusion just makes `n_action_checked` (and therefore the number of
# info-sets the action check can possibly flag) silently shrink while
# `n_infosets` and `rep.ok` both stay exactly as green as before. Pinned per
# variant: every game except liarsdice checks
# All its info-sets (never reaches |A|=1 at P0); liarsdice checks exactly
# half (6,144 of 12,288 -- the other half is the max-bid, CHALLENGE-only
# corner, excluded by construction; see `SelfParseReport`'s docstring).
_SELF_PARSE_EXPECTED_N_ACTION_CHECKED = {
    "kuhn": 6,
    "leduc": 468,
    "goofspiel": 81,
    "liarsdice": 6144,
    "darkhex": 33,
    "goofspiel_5": 1062,
    # liarsdice_1d8: same "exactly half excluded" structural property as
    # canonical liarsdice -- 131072 of 262144.
    "liarsdice_1d8": 131072,
    # goofspiel_6/kuhn_deep/leduc_deep: none of the three ever reaches the
    # |A|=1 exclusion corner at P0 (same structural property as their
    # canonical counterparts -- leduc_deep's facing-final-bet branch offers
    # {Fold, Call}, |A|=2, never 1), so n_action_checked == n_infosets for
    # all three.
    "goofspiel_6": 17241,
    "kuhn_deep": 26,
    "leduc_deep": 9828,
}


# Correspondence-CHECK: prompt-infoset
@pytest.mark.parametrize("game_name", _SELF_PARSE_GAMES)
def test_self_parse_invariant_echo_never_parses(game_name):
    """    ``n_infosets`` is pinned to the same expected counts
    ``label_agreement_report`` uses (this is the identical full-tree walk,
    just a different per-info-set check) -- without it, a walk regression
    that visits nothing would report ``n_infosets=0``, both self-parse lists
    trivially empty, and ``rep.ok`` would read green on an empty tree.
    ``n_action_checked`` is pinned separately (see the constant's comment
    above) to close the exclusion-widening gap ``n_infosets`` alone cannot
    catch.
    """
    rep = self_parse_report(game_name)
    assert rep.n_infosets == _SELF_PARSE_EXPECTED_N_INFOSETS[game_name], (
        f"{game_name}: expected {_SELF_PARSE_EXPECTED_N_INFOSETS[game_name]} "
        f"info-sets from the full-tree walk, got {rep.n_infosets}"
    )
    assert rep.n_action_checked == _SELF_PARSE_EXPECTED_N_ACTION_CHECKED[game_name], (
        f"{game_name}: expected {_SELF_PARSE_EXPECTED_N_ACTION_CHECKED[game_name]} "
        f"info-sets actually checked for an action self-parse, got {rep.n_action_checked} "
        f"(n_infosets={rep.n_infosets}) -- the |A|=1 exclusion may have widened"
    )
    assert not rep.action_self_parses, (
        f"{game_name}: {len(rep.action_self_parses)} info-set(s) where an echo "
        f"of action_suffix(info) parsed as a legal action: "
        f"{rep.action_self_parses[:3]}"
    )
    assert not rep.belief_self_parses, (
        f"{game_name}: {len(rep.belief_self_parses)} info-set(s) where an echo "
        f"of belief_suffix(info) parsed as a belief: "
        f"{rep.belief_self_parses[:3]}"
    )
    assert rep.ok


# ---------------------------------------------------------------------------
# Offline half of the canonical-continuation guarantee.
#
# scripts/exact_policy_pass.py scores an action by concatenating token ids: a
# chat-template prefix (``render_chat_prefix_ids``, /tokenize on the
# messages, ``add_generation_prompt`` left at the server's default of True)
# then the action literal tokenised standalone (``tokenize_action``,
# ``add_special_tokens=False``). That concatenation rests on an unstated
# premise: it equals the tokeniser's own canonical encoding of the whole
# (user turn, assistant turn = literal) conversation
# (``canonical_assistant_ids``, ``continue_final_message=True``). The
# runtime detector (``assert_canonical_continuation``) checks this per call
# against the live vLLM endpoint. This is the OFFLINE half: a standing test
# over the whole action-literal vocabulary of every tractable registered
# variant, using the real Hub tokenizer configs rather than a fake, in the
# same enumerate-the-whole-space pattern the rest of this file uses.
#
# TOKENIZERS: Qwen/Qwen2.5-14B-Instruct, allenai/OLMo-2-1124-13B-Instruct,
# google/gemma-2-2b-it and google/gemma-4-12B-it.
# mistralai/Mistral-Nemo-Instruct-2407 is deliberately excluded: its served
# tokenizer is mistral-common under ``--tokenizer-mode mistral``, not the
# Hub ``AutoTokenizer`` -- the two disagree by construction (see
# ``render_chat_prefix_ids``'s docstring in ``scripts/exact_policy_pass.py``).
# A Hub-tokenizer check would test a different tokenizer than the one
# actually served, so it is not run as a stand-in for Mistral; this
# limitation is intentional, not a gap.
#
# The two Gemma tokenizers are not roster models (the roster is
# Qwen2.5-14B-Instruct, OLMo-2-1124-13B-Instruct and
# Mistral-Nemo-Instruct-2407); they are kept because their templates
# exercise cases the roster's do not. For gemma-2-2b-it, unlike Qwen and
# OLMo, its ``add_special_tokens`` flag is not inert -- see mutation (a) on
# ``test_action_literals_concat_equals_canonical_tokenisation`` below for
# what that buys this test.
#
# google/gemma-4-12B-it is the only tokenizer here whose chat template
# emits generation-prompt-only tokens -- its template opens and closes an
# empty thinking channel (``<|channel>thought\n<channel|>``) inside the
# generation prompt that the canonical (``continue_final_message=True,
# add_generation_prompt=False``) encoding never renders, because that
# channel belongs to the turn the model has not yet produced. It is
# therefore the only tokenizer in this list for which
# ``add_generation_prompt`` changes the prefix length, not just its content,
# and a bare ``canonical == prefix_ids + action_ids`` comparison cannot hold
# for it by construction. Production (``scripts/exact_policy_pass.py``)
# handles this by measuring the served generation-prompt suffix once per arm
# (``measure_generation_prompt_suffix``, ``G``) and factoring it out before
# comparing (``assert_canonical_continuation``'s ``gen_suffix``); this test
# performs the same decomposition offline, against the real cached Hub
# tokenizer, over the entire action-literal vocabulary rather than the
# handful of live probes a run can afford.
# ---------------------------------------------------------------------------

_TOKENIZER_MODEL_IDS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-4-12B-it",
]

# Derived against the real cached Hub tokenizers by running the per-pair
# derivation in ``test_action_literals_concat_equals_canonical_tokenisation``
# below (not copied from prose). Empty for every model whose generation
# prompt adds nothing beyond the canonical head. gemma-4's four ids decode to
# ``'<|channel>thought\n<channel|>'`` and are byte-identical to the value
# measured live over HTTP against a vLLM endpoint -- an independent
# offline/online cross-check.
_EXPECTED_GENERATION_PROMPT_SUFFIX = {
    "Qwen/Qwen2.5-14B-Instruct": [],
    "allenai/OLMo-2-1124-13B-Instruct": [],
    "google/gemma-2-2b-it": [],
    "google/gemma-4-12B-it": [100, 45518, 107, 101],
}

# Derived by running ``_action_literal_vocabulary()``, never copied from
# prose. The per-game breakdown is
# asserted alongside the total so a roster change that happens to preserve
# the total (one game losing a literal while another gains one) still
# reddens, and so a failure names which game moved.
_EXPECTED_LITERAL_PAIRS_PER_GAME = {
    "kuhn": 4,
    "leduc": 5,
    "goofspiel": 4,
    "liarsdice": 13,
    "darkhex": 4,
    "goofspiel_5": 5,
    # liarsdice_1d8: 16 bid literals ("1 1".."2 8", max_count=2 x
    # dice_sides=8) plus CHALLENGE.
    "liarsdice_1d8": 17,
    # goofspiel_6 has 6 bid literals ("1".."6", same shape as goofspiel_5's
    # 5); kuhn_deep has the same 4-literal FOLD/CALL/CHECK/BET vocabulary as
    # canonical kuhn; leduc_deep adds RAISE for its facing-bet branch, same
    # 5-literal vocabulary as canonical leduc.
    "goofspiel_6": 6,
    "kuhn_deep": 4,
    "leduc_deep": 5,
}
_EXPECTED_TOTAL_LITERAL_PAIRS = sum(
    _EXPECTED_LITERAL_PAIRS_PER_GAME.values()
)  # 67


def _action_literal_vocabulary() -> list[tuple[str, str, str]]:
    """One representative (game, literal, prefix) triple per distinct action
    literal actually offered within each tractable registered variant.

    Walks the full P0 tree of every ``_LABEL_AGREEMENT_GAMES`` variant (same
    10 games; same darkhex_3x3 exclusion as the rest of this file -- no
    full-tree walk is tractable for it) via ``enumerate_infosets``, and for
    every info-set pairs its rendered prompt (the exact ``build_prompt +
    action_suffix`` text a live match sends, i.e. the same ``user_content``
    ``render_chat_prefix_ids`` receives) with every literal
    ``action_labels`` offers there. The first prefix at which a literal is
    encountered, within its own game, is kept as that (game, literal)'s
    representative.

    Per-literal, not per-info-set, is deliberate, not a shortcut: the
    tokenisation-boundary risk this test checks depends on the tokeniser's
    own merge behaviour at the (prefix ending, literal start) boundary and
    on the literal's own characters, not on which specific info-set produced
    that prefix, and a game's ``action_suffix`` ends the same way (a fixed
    question tail) across its whole tree. A per-info-set version would
    multiply this to tens of thousands of tokenizer calls for no additional
    coverage of that risk (liarsdice alone has 12,288 P0 info-sets).
    The count is of distinct (game, literal) pairs (see
    ``_EXPECTED_LITERAL_PAIRS_PER_GAME``), not of distinct literal strings
    across all games (e.g. 'PASS' is one string shared by several games'
    vocabularies).
    """
    seen: dict[tuple[str, str], str] = {}
    for game_name in _LABEL_AGREEMENT_GAMES:
        game = load_game(game_name)
        fns = get_game_fns_for_game(game)
        mapping = enumerate_infosets(game)
        for info, prefix in mapping.items():
            for literal in fns.action_labels(info).values():
                seen.setdefault((game_name, literal), prefix)
    return [(g, lit, prefix) for (g, lit), prefix in seen.items()]


def _ids(chat_template_output):
    """Unwrap `apply_chat_template`'s return value to a plain id list.

    Depending on transformers version and arguments, ``tokenize=True``
    returns either a bare ``list[int]`` or a ``BatchEncoding`` (a dict-like
    object keyed by ``"input_ids"``/``"attention_mask"``); ``list(...)`` on
    the latter silently returns its KEY names, not the ids (skipping this
    unwrap compares ``['input_ids', 'attention_mask']``-length-2 lists and
    passes vacuously). ``hasattr(..., "keys")`` distinguishes the two without
    depending on the exact `BatchEncoding` class.
    """
    return (
        chat_template_output["input_ids"]
        if hasattr(chat_template_output, "keys")
        else chat_template_output
    )


@pytest.fixture(scope="module", params=_TOKENIZER_MODEL_IDS)
def served_tokenizer(request):
    """A real, offline Hub tokenizer for one of the four tokenizers
    in ``_TOKENIZER_MODEL_IDS``, cached locally.

    ``pytest.importorskip`` for `transformers`, then a per-model
    `local_files_only=True` load that skips (never fails) if the tokenizer
    config is not cached locally. Other machines and CI will not have this
    cache, and a red test on a fresh clone is worse than a skipped one.

    Only the specific failures are caught (OSError covers both local cache
    misses and network errors when local_files_only is violated; ImportError
    covers missing optional dependencies). A loaded tokenizer is asserted
    non-None, so tokenizer construction failures that aren't import/download
    issues fail the test rather than hide as skips.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            request.param, local_files_only=True
        )
    except (OSError, ImportError) as exc:
        pytest.skip(f"{request.param} tokenizer not cached locally ({exc})")

    # Assert the loaded tokenizer is not None - covers tokenizer construction
    # succeeding without raising but returning None or an invalid state.
    assert (
        tokenizer is not None
    ), f"AutoTokenizer.from_pretrained returned None for {request.param}"
    return tokenizer


def test_action_literals_concat_equals_canonical_tokenisation(served_tokenizer):
    """Every action literal, in every registered variant's vocabulary, must
    tokenise the same standalone-then-concatenated as it does inside the
    tokeniser's own canonical (user, assistant=literal) encoding.

    Mirrors production exactly, using the same functions
    ``scripts/exact_policy_pass.py`` actually calls as the contract:
    - `prefix_ids` = the chat template with ``add_generation_prompt=True``
      (matches ``render_chat_prefix_ids``'s ``/tokenize(messages)`` call,
      which does not pass ``add_generation_prompt`` and so gets the
      server's default of True for a messages-based request).
    - `action_ids` = the literal tokenised alone with
      ``add_special_tokens=False`` (matches ``tokenize_action``).
    - `canonical` = the same tokeniser's ``continue_final_message=True,
      add_generation_prompt=False`` encoding of the full (user,
      assistant=literal) turn (matches ``canonical_assistant_ids``, the
      runtime detector's own reference).

    A bare ``canonical == prefix_ids + action_ids`` does not hold for
    gemma-4 by construction. Its chat template opens and closes an empty
    thinking channel inside the generation prompt
    (``'<|channel>thought\\n<channel|>'``) that the canonical encoding never
    renders, since that channel belongs to a turn the model has not yet
    produced. This test performs the same served-side decomposition
    production performs (``measure_generation_prompt_suffix`` /
    ``assert_canonical_continuation``'s ``gen_suffix``), per pair: slice
    ``canonical`` into ``head`` + ``action_ids`` (erroring
    ``NonCanonicalActionTokenisation``-style if the literal does not survive
    the canonical encoding), confirm ``prefix_ids`` opens with that same
    ``head`` (erroring ``NonCanonicalPrefix``-style if not), and read
    whatever remains of ``prefix_ids`` past ``head`` as the observed
    generation-prompt suffix. Accumulating that observed suffix across all
    pairs and asserting the set has exactly one member is the offline
    analogue of production's cross-probe invariance check
    (``measure_generation_prompt_suffix``'s ``GenerationPromptSuffixUnstable``)
    -- and it is far stronger, since it covers the entire action-literal
    vocabulary instead of the handful of probes a live run can afford. The
    final assertion pins that single value against
    ``_EXPECTED_GENERATION_PROMPT_SUFFIX``, so a republished template
    reddens this test instead of silently changing what production scores.

    Named mutations (each run against this test -- a test only ever observed
    green is characterisation, not a gate):
    (a) flip the standalone action tokenisation to ``add_special_tokens=True``.
        This does not redden on Qwen2.5-14B-Instruct or
        OLMo-2-1124-13B-Instruct. Qwen2.5-14B-Instruct's
        tokenizer_config.json sets ``add_bos_token: false`` and
        ``bos_token: null`` -- there is no special token for the flag to
        add. OLMo-2-1124-13B-Instruct's GPT2-style tokenizer class does not
        auto-insert its BOS/EOS (both ``<|endoftext|>``) on a plain
        single-sequence encode regardless of the flag (``tok.encode("PASS",
        add_special_tokens=True) == tok.encode("PASS",
        add_special_tokens=False)`` on both).

        google/gemma-2-2b-it does differentiate: ``tok("PASS",
        add_special_tokens=True)`` = ``[2, 24524]`` vs
        ``add_special_tokens=False`` = ``[24524]`` -- id 2 is gemma's BOS,
        unconditionally prepended. The mutation reddens on every (game,
        literal) pair against gemma-2, first failure kuhn/'CHECK' (canonical
        192 ids vs mutated concat 193 ids). This offline layer therefore
        carries genuine red-side evidence for the ``add_special_tokens``
        class of defect.

        The live risk this mutation targets is specific to Mistral's served
        tokenizer (mistral-common under ``--tokenizer-mode mistral``, which
        does prepend a BOS -- see ``tokenize_action``'s docstring in
        scripts/exact_policy_pass.py). Mistral's own red-side proof lives
        only in the live check (``detector_action_failures == 127`` against
        Mistral under this same flip), not in this offline test -- a real
        limit on what this test alone can prove for that tokenizer, not a
        defect in it.

        This mutation does not redden on google/gemma-4-12B-it either: its
        ``add_special_tokens`` flag is inert, like Qwen's and OLMo's (the
        full parametrized run gives 1 failed (gemma-2-2b-it only), 3
        passed). gemma-4's tokenizer
        (``Gemma4UnifiedForConditionalGeneration``'s, distinct from
        gemma-2's despite the shared family name) does not unconditionally
        prepend a BOS the way gemma-2's does. gemma-2 remains the tokenizer
        that proves the class of defect is reachable offline at all, and
        gemma-4's own offline-reachable defect class is the
        suffix-invariance one covered by mutation (c) below.
    (b) append a trailing space to one action literal before comparing --
        reddens on all four tokenizers (kuhn's 'CHECK' literal gains one
        extra token from the leading-space BPE merge; canonical 192 ids vs
        mutated concat 193 ids). Only the standalone side fed to
        ``action_ids`` is mutated, leaving the canonical reference's literal
        unmutated; mutating both sides identically does not distinguish
        them and does not redden.
    (c) weaken ``assert len(suffixes) == 1`` to ``assert len(suffixes) >= 1``.
        This does not redden on any of the four tokenizers, because every
        one of them yields a suffix that is constant across the whole
        vocabulary (Qwen, OLMo and gemma-2 at ``()``, gemma-4 at its one
        4-id value) -- ``suffixes`` is a nonempty set of size exactly 1 in
        every run, so ``>= 1`` and ``== 1`` accept exactly the same data on
        this list.

        The invariance check is nonetheless tested by a different mutation
        of the same assertion: make `head` prompt-independent by forcing
        ``head = []`` unconditionally. The prefix-opens-with-head check then
        passes vacuously (an empty list is a prefix of anything) and
        ``observed_suffix`` absorbs the entire per-pair prompt instead of
        just its true generation-prompt tail; the walk's shared
        representative prefixes collapse into 11 distinct full prompts, so
        ``suffixes`` has 11 members and ``assert len(suffixes) == 1``
        reddens (``assert 11 == 1``) identically on all four tokenizers.
        ``>= 1`` would not catch this either (11 >= 1 is true), but ``== 1``
        genuinely does the work here.

        A second, subtler mutation shows which of the two closing
        assertions catches a boundary (rather than fully degenerate) defect:
        shift `head`'s slice by one element
        (``canonical[: len(canonical) - len(action_ids) - 1]``), which steals
        one token -- the chat template's own fixed assistant-turn opener,
        constant across every pair regardless of game or literal -- out of
        `head` and into the recorded suffix. This does not redden
        ``assert len(suffixes) == 1`` (the stolen token is the same control
        token on every pair, so the suffix stays constant, just wrong); it
        reddens the second assertion instead --
        ``list(suffixes.pop()) == _EXPECTED_GENERATION_PROMPT_SUFFIX[...]``
        -- on all four (e.g. gemma-4: observed ``[107, 100, 45518, 107,
        101]`` vs expected ``[100, 45518, 107, 101]``, an extra leading
        107). So the two closing assertions are complementary, not
        redundant: assertion 1 catches a head that varies within the
        vocabulary (including full degeneracy); assertion 2 catches a head
        that is wrong but happens to be constant. Without assertion 1 pinning
        `suffixes` to size 1 first, `suffixes.pop()` on a varying set would
        be nondeterministic (Python set iteration order is not the id
        sequence's own order), which would turn a real defect into an
        occasional flake rather than a reliable failure.

        Like ``scripts/exact_policy_pass.py``'s
        ``measure_generation_prompt_suffix`` (which raises
        ``NonCanonicalActionTokenisation`` up front whenever
        ``len(canonical) <= len(action_ids)``, so a degenerate head can never
        even be computed), the first assertion in the loop body is a
        dedicated guard on that exact condition, so a degenerate head is
        diagnosed at its own site instead of two assertions later as a side
        effect of the invariance check.

    The count is asserted, not merely described: without it an empty or
    truncated walk passes silently -- vacuously green over zero literals
    reads exactly like green over the full vocabulary.
    """
    vocabulary = _action_literal_vocabulary()
    assert len(vocabulary) == _EXPECTED_TOTAL_LITERAL_PAIRS
    per_game = Counter(game_name for game_name, _, _ in vocabulary)
    assert dict(per_game) == _EXPECTED_LITERAL_PAIRS_PER_GAME

    suffixes: set[tuple[int, ...]] = set()
    n_checked = 0
    for game_name, literal, prefix in vocabulary:
        n_checked += 1
        user_msg = {"role": "user", "content": prefix}
        action_ids = served_tokenizer(literal, add_special_tokens=False)["input_ids"]
        canonical = _ids(
            served_tokenizer.apply_chat_template(
                [user_msg, {"role": "assistant", "content": literal}],
                tokenize=True,
                continue_final_message=True,
                add_generation_prompt=False,
            )
        )
        prefix_ids = _ids(
            served_tokenizer.apply_chat_template(
                [user_msg], tokenize=True, add_generation_prompt=True
            )
        )

        # Mirrors scripts/exact_policy_pass.py's measure_generation_prompt_suffix,
        # which raises NonCanonicalActionTokenisation up front on this exact
        # condition so a degenerate head is never even computed (see named
        # Mutations (c) for what catches a degenerate head without this guard).
        assert len(canonical) > len(action_ids), (
            f"{game_name}/{literal!r}: canonical encoding ({len(canonical)} ids) is no "
            f"longer than the action alone ({len(action_ids)} ids) -- no head remains to "
            f"derive a generation-prompt suffix from"
        )
        assert list(canonical)[len(canonical) - len(action_ids) :] == list(
            action_ids
        ), f"{game_name}/{literal!r}: the literal does not survive the canonical encoding"
        head = list(canonical)[: len(canonical) - len(action_ids)]
        assert (
            list(prefix_ids)[: len(head)] == head
        ), f"{game_name}/{literal!r}: served prefix does not open with the canonical head"
        observed_suffix = tuple(prefix_ids[len(head) :])
        suffixes.add(observed_suffix)  # accumulate across all pairs

    # Belt and braces against a future `continue`/filter inside the loop
    # silently skipping literals while the vocabulary itself stays full length.
    assert n_checked == _EXPECTED_TOTAL_LITERAL_PAIRS

    # 1. The suffix is a property of the template, not of the prompt or
    #    literal. See the docstring's "NAMED MUTATIONS" (c) above for what
    #    `== 1` vs `>= 1` does and does not buy given the currently-cached
    #    tokenizer set.
    assert (
        len(suffixes) == 1
    ), f"generation-prompt suffix is not constant across the vocabulary: {suffixes}"
    # 2. It equals the value recorded for this model, so a republished
    #    template reddens instead of silently changing what production scores.
    assert (
        list(suffixes.pop())
        == _EXPECTED_GENERATION_PROMPT_SUFFIX[served_tokenizer.name_or_path]
    )
