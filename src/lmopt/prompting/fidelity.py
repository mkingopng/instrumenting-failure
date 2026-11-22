"""Full-tree prompt-fidelity auditing.

Detects the class of defect that produced the Leduc and Dark Hex prompt bugs:
information present in the info-set that never reaches the serialised prompt.
Enumerates the whole tree per game, not the rows a particular run happened to
visit -- a run's visited set can be a fraction of a percent of the space
(landed rows cover liarsdice 32/24,576 = 0.13%, darkhex 5/134 = 3.7%), so
game-level percentages computed from a sample are not audits.

Two independent checks, because injectivity alone is not enough:

1. Collision (this module's ``collision_report``): do two distinct
   info-sets render to the identical prompt string?
2. Fidelity (asserted directly in ``tests/test_prompt_fidelity.py``, not
   here): does the rendered prompt state the TRUE content of the info-set?
   An injectivity-only audit is structurally blind to one-to-*wrong*
   mappings -- a Leduc rendering that calls a Jack "Queen" and a Queen
   "King" is still injective, because distinct-but-wrong prompts are
   distinct.

The harm criterion for collisions is best-response value or action, not
CFR average-policy equality. "Colliding info-sets have different CFR average
policies" reports "0.0% harmful" for Dark Hex, because ``solve_cfr`` leaves
121 of its 134 info-sets exactly uniform (players effectively cannot
distinguish those states under the average policy) -- that criterion is
vacuous by construction, not evidence that the collisions are harmless.
Best-response value is non-degenerate wherever the
underlying states genuinely differ, uniform average policy or not. Action is
checked too: P3 scores the model's action
distribution against the reference policy, not the value, so two info-sets
whose values happen to coincide but whose best-response actions differ are
still "scored against a distinction it was never shown" -- value-only
under-detects that case.

The best response is computed by the imported
``open_spiel.python.algorithms.best_response.BestResponsePolicy``, never a
hand-rolled tree walk -- this is load-bearing for the project's research
claim that the equilibrium reference and the scoring yardstick come from the
same validated OpenSpiel source.

Scope across memory rungs: The collision, fidelity, and
self-parse audits range over the current-hand info-set corpus (what
``enumerate_infosets`` returns). The empty-render rule (memories empty renders
"") keeps hand-1 prompts byte-identical across rungs M0-M3, so the M0 audits
cover all rungs at hand 1. Beyond hand 1, the audited invariant weakens to
(info-set, memory-block): memory blocks are pinned separately, and the composition property (base +
memory_block + suffix) makes the combined pin sufficient. The one-to-one
info-set-to-prompt assumption therefore weakens for M1-M3 rows beyond hand 1,
and this weakening is stated rather than hidden.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Optional

import pyspiel
from open_spiel.python.algorithms.best_response import BestResponsePolicy

from lmopt.games.cfr import solve_cfr
from lmopt.prompting import get_game_fns_for_game
from lmopt.prompting.spec import GamePromptSpec


def enumerate_infosets(
    game: pyspiel.Game, player: int = 0, spec: Optional[GamePromptSpec] = None
) -> dict[str, str]:
    """Every info-set string reachable for `player`, mapped to its rendered prompt.

    Walks the full game tree (chance nodes expanded over every outcome, both
    players' every legal action), not a sampled run -- a run's visited set can
    be a fraction of a percent of the space (see module docstring). The
    rendered prompt is exactly what a live match sends the LM:
    ``build_prompt(info, player) + action_suffix(info)``, both parts,
    concatenated, from the same ``GamePromptFns`` a match uses.

    ``spec`` renders a probe variant instead of the canonical
    content. Threading it is load-bearing rather than convenience: with a
    hardcoded ``get_game_fns_for_game(game)``, a "per-variant" audit built on
    it would verify the canonical rendering while reporting under the
    variant's name. ``None`` renders the canonical spec.
    """
    fns = get_game_fns_for_game(game, spec=spec)
    out: dict[str, str] = {}

    def walk(state: pyspiel.State) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _p in state.chance_outcomes():
                walk(state.child(action))
            return
        if state.current_player() == player:
            info = state.information_state_string(player)
            if info not in out:
                out[info] = fns.build_prompt(info, player) + fns.action_suffix(info)
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return out


@dataclass
class CollisionReport:
    """Prompt-collision audit for one game's full P0 decision-node tree."""

    game: str
    n_infosets: int
    n_prompts: int
    collisions: dict[str, list[str]] = field(default_factory=dict)
    harmful: dict[str, list[str]] = field(default_factory=dict)


# Reference-solve iteration budgets, keyed by OpenSpiel's own
# ``game.get_type().short_name`` (stable across depth variants that share a
# short_name -- e.g. every Dark Hex rectangle is "dark_hex"). Mirrors
# scripts/validate_cfr.py's validated CFR+ convergence budgets, measured on
# a Linux box, so the harm criterion's reference is the same quality bar the
# CI gate already trusts. Deliberately no fallback default: an unbudgeted
# game must fail loudly (no silent fallbacks), not solve on a guessed count.
_CFR_ITERATIONS: dict[str, int] = {
    "kuhn_poker": 300,
    "leduc_poker": 1000,
    "turn_based_simultaneous_game": 500,
    "liars_dice": 500,
    "dark_hex": 500,
    # "universal_poker" covers both
    # kuhn_deep and leduc_deep (same OpenSpiel short_name) -- but unlike the
    # entries above, this is not a validated convergence budget mirroring
    # validate_cfr.py's own (leduc_deep's is 550 iters, ~1h43m; kuhn_deep
    # never reaches this code path at all, since it has 0 collisions and the
    # harm check short-circuits on an empty `collisions` dict). It is a
    # Cost-bounded compromise: leduc_deep's per-iteration cost is
    # ~11-13s/iter (vs. leduc_poker's ~0.19s/iter), so a
    # leduc_poker-parity 1000-iteration budget would take ~3+ hours. 30 is
    # trustworthy here for a different reason than convergence quality: the
    # residual collisions leduc_deep's own build_prompt produces are suit-only
    # by STRUCTURAL construction (parse_own_card/_parse_public_card in
    # leduc_deep.py discard the suit character unconditionally), and Leduc's
    # hand-ranking rule (pair beats no-pair, else higher rank) never depends
    # on suit at any numRanks -- so the true harmful count is 0 regardless of
    # how converged the reference policy is; a run at this budget finds
    # 0/2574 collision groups harmful at this budget, consistent with that
    # structural argument, not a coincidence of under-convergence.
    "universal_poker": 30,
}


@functools.lru_cache(maxsize=None)
def _best_response_policy(
    game_str: str, short_name: str, player: int
) -> BestResponsePolicy:
    """Cached ``BestResponsePolicy`` for `player` against a solved CFR(+) reference.

    Cached on the game's own canonical string (``str(game)``, which
    round-trips through ``pyspiel.load_game`` -- verified for all five
    canonical games) so that ``collision_report`` can call the harm check
    once per collision group without re-running CFR and re-walking the tree
    each time; the solve and the tree walk inside ``BestResponsePolicy`` are
    the expensive parts, not the per-group comparison.
    """
    if short_name not in _CFR_ITERATIONS:
        raise ValueError(
            f"no CFR iteration budget for short_name {short_name!r}; "
            f"expected one of {sorted(_CFR_ITERATIONS)} (no silent fallback)"
        )
    game = pyspiel.load_game(game_str)
    if short_name == "universal_poker":
        # The 30-iteration "universal_poker" budget above is trustworthy only
        # because leduc_deep's residual collisions are suit-only by
        # STRUCTURAL construction -- an argument that itself depends on
        # numSuits == 2 (native Leduc's own suit count; see the
        # _CFR_ITERATIONS comment). "universal_poker" is shared by every
        # variant dispatched through it (kuhn_deep, leduc_deep, and any
        # future one), so a variant with more suits would silently inherit
        # this budget without the argument that justifies it: keying by
        # short_name, not by registry key, means a wrong-but-quiet reuse is
        # possible. Fail loudly rather than let a future variant's harm
        # check run against a reference this budget was never justified for.
        num_suits = game.get_parameters().get("numSuits")
        if num_suits != 2:
            raise ValueError(
                f"universal_poker's 30-iteration CFR budget assumes "
                f"numSuits == 2 (the suit-symmetry argument this budget "
                f"relies on -- see _CFR_ITERATIONS's comment); got "
                f"numSuits={num_suits!r} for {game_str!r}. This budget has "
                f"not been justified for this game shape -- re-derive an "
                f"iteration count for it (or key _CFR_ITERATIONS by "
                f"registry key instead of short_name) before trusting a "
                f"harm check against this reference (no silent fallback)."
            )
    reference_policy = solve_cfr(game, _CFR_ITERATIONS[short_name])
    return BestResponsePolicy(game, player, reference_policy)


def _infoset_br_value(brp: BestResponsePolicy, info: str) -> float:
    """Best-response value of reaching info-set `info`, weighted by counterfactual reach.

    ``BestResponsePolicy.value(state)`` is a per-state value; an info-set can
    correspond to more than one history. We report the counterfactual-reach-
    weighted average across the info-set's ``(state, cf_prob)`` pairs -- the
    same weighting ``BestResponsePolicy`` itself uses internally to choose the
    single best-response action for the info-set (``best_response_action``),
    so this is the value consistent with that action, not an ad hoc average.
    """
    nodes = brp.infosets[info]
    total_p = sum(p for _, p in nodes)
    if total_p <= 0.0:
        # Zero-reach info-set under this reference: fall back to the
        # unweighted first state rather than dividing by zero. Rare (only
        # possible for unreached branches) and still a real BR value, not a
        # silent default -- value() always computes from actual game returns.
        return brp.value(nodes[0][0])
    return sum(p * brp.value(s) for s, p in nodes) / total_p


def _br_values_differ(
    game: pyspiel.Game, info_sets: list[str], player: int, tol: float = 1e-9
) -> bool:
    """True if best-response value or action differs materially across `info_sets`.

    Non-vacuous under a uniform CFR average policy (see module docstring):
    best-response value is computed from actual terminal returns via the
    imported ``BestResponsePolicy``, not from the average-policy probabilities
    that Dark Hex leaves uniform on 121/134 info-sets.

    Value equality alone under-detects. Two info-sets
    can have equal best-response value (e.g. two branches that are equally
    good by different routes) while their best-response action differs -- and
    P3 scores the model's chosen action distribution against the reference
    policy, not the value, so a collapsed prompt that hides an action-level
    distinction is exactly "scored against a distinction it was never shown"
    even when the value happens to coincide. Both checks use the same
    imported ``BestResponsePolicy`` instance -- no second reference is solved.
    """
    short_name = game.get_type().short_name
    brp = _best_response_policy(str(game), short_name, player)
    values = [_infoset_br_value(brp, info) for info in info_sets]
    if (max(values) - min(values)) > tol:
        return True
    actions = {brp.best_response_action(info) for info in info_sets}
    return len(actions) > 1


@dataclass
class LabelAgreementReport:
    """Action-label fidelity audit for one game's full P0 decision-node tree:
    do ``GamePromptFns.action_labels`` and
    ``action_suffix``/``legal_actions`` agree everywhere, not just on a
    sample? This is the standing-invariant answer to a class of defect hit
    repeatedly, one game at a time: a candidate/label scheme that is
    not actually derived from ``action_suffix``'s own rendering data can
    silently diverge from what the prompt offers while still round-tripping
    through ``parse_action``. Checking both representations, over the whole
    tree, is what makes drift in either direction fail loudly instead of
    surfacing only when someone measures a live endpoint by hand.

    ``label_action_mismatch``: a permutation of
    the action->label mapping (e.g. two labels swapped between two actions)
    preserves both the substring check and the key-set check above - the
    permuted labels are still present verbatim in ``action_suffix``, and the
    key SET is unchanged, only which key maps to which value moved. Measured
    reachable for leduc/goofspiel/liarsdice (176/185 info-sets, 543/574
    calls): all three passed both checks above under an injected
    permutation. This third check closes it: ``fns.parse_action(label, info)
    == action`` for every ``action: label`` pair.

    This is not the round-trip-as-source mistake, and the distinction is
    load-bearing: that defect uses the parse_action round-trip as the source
    of the candidate literal (search a hand-written list for
    anything that round-trips, keep the first hit) - a literal that
    round-trips to the right action can still be text the prompt never
    offered. Here the literal is already sourced from the renderer's own
    data (``action_labels``, never parsed, never hand-written), and
    ``parse_action`` is used only as a third independent cross-CHECK on top
    of that source, not as the mechanism that produces the literal. Source
    versus check.
    """

    game: str
    n_infosets: int
    # info-set -> labels present in action_labels() but not found verbatim
    # (substring) in action_suffix()'s rendered text.
    label_not_in_suffix: dict[str, list[str]] = field(default_factory=dict)
    # info-set -> (sorted(action_labels().keys()), sorted(legal_actions()))
    # whenever the two key sets disagree.
    label_set_mismatch: dict[str, tuple] = field(default_factory=dict)
    # info-set -> list of (claimed_action, label, fns.parse_action(label, info))
    # whenever action_labels() maps `label` to `claimed_action` but the
    # game's own parser disagrees about which action `label` denotes - the
    # permutation case described above.
    label_action_mismatch: dict[str, list[tuple]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            not self.label_not_in_suffix
            and not self.label_set_mismatch
            and not self.label_action_mismatch
        )


def label_agreement_report(
    game_name: str, player: int = 0, spec: Optional[GamePromptSpec] = None
) -> LabelAgreementReport:
    """Enumerate the full tree (not a sample) and check, at every P0
    decision node: every ``action_labels(info)`` literal appears verbatim in
    ``action_suffix(info)``, ``action_labels(info)``'s key set equals
    ``state.legal_actions()``, and every ``action_labels(info)`` pair
    round-trips through the game's own ``parse_action`` (see
    ``LabelAgreementReport``'s docstring for why this is a
    cross-check on the label source, not the mistake of using the round-trip
    as the source).

    Deliberately its own tree walk rather than reusing ``enumerate_infosets``
    (which discards ``legal_actions()`` and returns only the rendered
    prompt) - this needs the live state's legal actions and the separate
    ``action_suffix`` string (not the ``build_prompt``-concatenated one), so
    a label that happens to appear somewhere in ``build_prompt``'s body but
    not in ``action_suffix`` itself cannot pass by accident.

    ``spec`` audits a probe variant rather than the canonical
    content; see ``enumerate_infosets`` for why threading it matters. Every
    variant must pass this gate at both seats before a probe arm runs.
    """
    from lmopt.games.loaders import load_game

    game = load_game(game_name)
    fns = get_game_fns_for_game(game, spec=spec)

    label_not_in_suffix: dict[str, list[str]] = {}
    label_set_mismatch: dict[str, tuple] = {}
    label_action_mismatch: dict[str, list[tuple]] = {}
    seen: set[str] = set()

    def walk(state: pyspiel.State) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _p in state.chance_outcomes():
                walk(state.child(action))
            return
        if state.current_player() == player:
            info = state.information_state_string(player)
            if info not in seen:
                seen.add(info)
                legal = state.legal_actions()
                labels = fns.action_labels(info)
                suffix = fns.action_suffix(info)
                missing = [lab for lab in labels.values() if lab not in suffix]
                if missing:
                    label_not_in_suffix[info] = missing
                if set(labels.keys()) != set(legal):
                    label_set_mismatch[info] = (sorted(labels.keys()), sorted(legal))
                mismatches = [
                    (a, lab, fns.parse_action(lab, info))
                    for a, lab in labels.items()
                    if fns.parse_action(lab, info) != a
                ]
                if mismatches:
                    label_action_mismatch[info] = mismatches
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return LabelAgreementReport(
        game=game_name,
        n_infosets=len(seen),
        label_not_in_suffix=label_not_in_suffix,
        label_set_mismatch=label_set_mismatch,
        label_action_mismatch=label_action_mismatch,
    )


@dataclass
class SelfParseReport:
    """Self-parse (echo) audit for one game's full P0 decision-node tree.
    Mirrors ``LabelAgreementReport``'s shape.

    The defect this detects: a model
    that echoes the question back verbatim -- never reasoning about the
    actual info-set at all -- must not parse as a valid answer. Concretely,
    for every P0 info-set:

    - ``parse_action(action_suffix(info), info)`` must be ``None`` -- an
      echo of the action question (which typically names every legal label,
      or an exemplar literal) must not itself parse as a legal action.
    - ``parse_belief(belief_suffix(info), info)`` must be ``None`` -- an
      echo of the belief question (which always ends with a JSON exemplar)
      must not parse as a normalised belief.

    Deliberately checked against the suffix alone (not
    ``build_prompt(info) + suffix``), matching ``label_agreement_report``'s
    own reasoning: this is the narrowest, most adversarial echo (a model
    that reproduces only the question, not the surrounding rules prose) and
    it is exactly what ``action_suffix``/``belief_suffix`` return on their
    own -- the same strings ``LMVllmAgent.act`` sends as the tail of its
    prompt.

    Excluded by construction, not a loophole: an info-set where
    ``action_labels(info)`` offers exactly one
    label is never checked for an action self-parse. At |A|=1,
    ``action_suffix`` must name that one label somewhere (the model cannot
    be told to act without being told the word to use), so echoing the
    question trivially reproduces the only legal answer -- there is no
    hardening that closes this without withholding the answer from the
    question itself, which would break the actual game contract. Measured
    reachable only in liarsdice (6,144 of 12,288 P0 info-sets, exactly half:
    every info-set whose last bid is the maximum, 2x6, where only CHALLENGE
    remains legal) -- kuhn/leduc/goofspiel/darkhex never reach |A|=1 at P0
    (verified: `len(action_labels(info)) >= 2` at every P0 info-set in all 8
    tractable registered variants). This exclusion does not weaken the
    invariant for any |A|>=2 info-set, including liarsdice's own -- e.g. the
    exemplar-literal case ("e.g. '2 5'" leaking into a bogus action at
    that same |A|=1 corner) is handled in the parser itself (liarsdice.py's
    ``parse_action`` legality gate), not by widening this exclusion; the
    exclusion only stops flagging the mathematically-unavoidable case, it
    does not change what the parser accepts.
    """

    game: str
    n_infosets: int
    # info-sets where an echo of action_suffix(info) parsed as a legal action.
    action_self_parses: list = field(default_factory=list)
    # info-sets where an echo of belief_suffix(info) parsed as a belief.
    belief_self_parses: list = field(default_factory=list)
    # How many of
    # `n_infosets` were actually checked for an action self-parse (|A| >= 2;
    # see the |A|=1 exclusion above). Separate from `n_infosets` on purpose:
    # `n_infosets` counts info-sets the tree walk visited, not how many the
    # action check actually examined, so pinning only `n_infosets` cannot
    # detect a regression that silently widens the exclusion threshold (e.g.
    # `>= 2` drifting to `>= 4`) -- every excluded info-set is still
    # "visited" and counted in `n_infosets`, so `ok` could stay vacuously
    # True while the check itself examines fewer and fewer info-sets. Pin
    # This field per variant, not just `n_infosets`, to close that gap.
    n_action_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.action_self_parses and not self.belief_self_parses


def self_parse_report(
    game_name_or_game, player: int = 0, spec: Optional[GamePromptSpec] = None
) -> SelfParseReport:
    """Enumerate the full tree (not a sample) and check, at every P0
    decision node, that echoing the action/belief question back does not
    itself parse as a valid answer (see ``SelfParseReport``'s docstring for
    the exact defect and why the suffix alone, not the full prompt, is the
    adversarial input checked).

    Accepts either a registered game name (``str``, resolved via
    ``lmopt.games.loaders.load_game`` -- the same convention
    ``label_agreement_report``/``collision_report`` use) or an already-loaded
    ``pyspiel.Game`` object (so callers who already hold a game, e.g. a
    depth-variant instance not in the registry, can pass it directly). The
    reported ``game`` field is the string name when one was given, else
    ``str(game)`` (the canonical OpenSpiel game string).

    Own tree walk (not a reuse of ``enumerate_infosets``, same reasoning as
    ``label_agreement_report``): needs the standalone ``action_suffix``/
    ``belief_suffix`` strings, not ``build_prompt``'s concatenated output.

    ``spec`` audits a probe variant rather than the canonical
    content; see ``enumerate_infosets`` for why threading it matters. Every
    variant must pass this gate at both seats before a probe arm runs.
    """
    if isinstance(game_name_or_game, str):
        from lmopt.games.loaders import load_game

        game_name = game_name_or_game
        game = load_game(game_name)
    else:
        game = game_name_or_game
        game_name = str(game)

    fns = get_game_fns_for_game(game, spec=spec)

    action_self_parses: list[str] = []
    belief_self_parses: list[str] = []
    seen: set[str] = set()
    n_action_checked = 0

    def walk(state: pyspiel.State) -> None:
        nonlocal n_action_checked
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _p in state.chance_outcomes():
                walk(state.child(action))
            return
        if state.current_player() == player:
            info = state.information_state_string(player)
            if info not in seen:
                seen.add(info)
                # |A|=1 exclusion (see SelfParseReport's docstring): a
                # single-label info-set's action_suffix necessarily names
                # that one label, so an echo trivially reproduces the only
                # legal answer -- not a genuine ambiguity-collapse defect.
                if len(fns.action_labels(info)) >= 2:
                    n_action_checked += 1
                    act_sfx = fns.action_suffix(info)
                    if fns.parse_action(act_sfx, info) is not None:
                        action_self_parses.append(info)
                bel_sfx = fns.belief_suffix(info)
                # belief_suffix's contract is str | None (prompting/__init__.py);
                # None means no belief question is askable here, not that one
                # was asked and echoed -- nothing to self-parse-check.
                if bel_sfx is not None and fns.parse_belief(bel_sfx, info) is not None:
                    belief_self_parses.append(info)
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return SelfParseReport(
        game=game_name,
        n_infosets=len(seen),
        action_self_parses=action_self_parses,
        belief_self_parses=belief_self_parses,
        n_action_checked=n_action_checked,
    )


def collision_report(
    game_name: str, player: int = 0, spec: Optional[GamePromptSpec] = None
) -> CollisionReport:
    """Group info-sets by rendered prompt; flag groups the reference distinguishes.

    Harm is measured by best-response value, not by average-policy equality: a
    uniform reference (Dark Hex is uniform on 121/134 info-sets under the
    CFR+ average policy) makes policy equality trivially true and the
    criterion vacuous. See module docstring.

    ``spec`` audits a probe variant rather than the canonical
    content; see ``enumerate_infosets`` for why threading it matters. Scope of
    the variant gate, stated precisely: the harm criterion is anchored at seat 0,
    which is where every canonical harm baseline in
    ``tests/test_prompt_fidelity.py`` is asserted and where a variant's
    ``collision_report`` is therefore comparable to a canonical one. The
    seat-general half of the gate is the prompt partition,
    checked at both seats: a relabel that changed which info-sets share a prompt
    would change that partition, and ``_br_values_differ`` never sees a prompt,
    so an unchanged partition implies an unchanged harm set.

    **The partition claim above is not enforced by this module** (stated here
    explicitly rather than left implied).
    """
    from lmopt.games.loaders import load_game

    game = load_game(game_name)
    mapping = enumerate_infosets(game, player, spec=spec)

    by_prompt: dict[str, list[str]] = {}
    for info, prompt in mapping.items():
        by_prompt.setdefault(prompt, []).append(info)

    collisions = {p: sorted(v) for p, v in by_prompt.items() if len(v) > 1}
    harmful = {
        p: v for p, v in collisions.items() if _br_values_differ(game, v, player)
    }
    return CollisionReport(
        game=game_name,
        n_infosets=len(mapping),
        n_prompts=len(by_prompt),
        collisions=collisions,
        harmful=harmful,
    )
