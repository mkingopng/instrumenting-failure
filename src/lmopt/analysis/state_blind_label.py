"""The single canonical state-blindness label.

One text, one home. Two wordings that drift apart are worse than one that is
slightly wrong. Every emitter and the standing check import from here.

Over the three landed v2 arms, 496,659/661,644 = 75.1 % of LM
decision nodes carry an action that is a constant function of `legal_mask`
alone (Qwen 61.8 %, OLMo 71.8 %, Mistral 91.1 %).

This module must stay importable without matplotlib: `scripts/analyse_stats.py`
imports it and carries no matplotlib dependency. `draw_state_blind_note` takes
the already-constructed `fig` as an argument for exactly this reason -- it
never needs to import matplotlib itself.
"""

from __future__ import annotations

STATE_BLIND_COLUMN = "cell_lm_mask_constant_share"

LABEL_SENTENCE = (
    "Share of this cell's LM rows whose action is a constant function of "
    "`legal_mask` alone. Such rows record a response to prompt format rather "
    "than a policy over game states. These cells are reported, not suppressed. "
    "Whether the constancy is positional bias or "
    "label preference remains unresolved: both hypotheses predict the "
    "identical observation, and no analysis of this corpus can separate them."
)

# No Markdown backticks here.
# FIGURE_NOTE reaches matplotlib's fig.text (draw_state_blind_note below), which
# does not render Markdown, so a literal backtick would show up as a literal
# character on every figure. LABEL_SENTENCE
# above keeps its backticks: it has no production consumer (record-only),
# reserved for a future Markdown-rendered surface.
FIGURE_NOTE = (
    "State-blindness: a share of the LM rows behind these numbers carry an "
    "action constant in legal_mask alone. "
    "Positional bias vs label preference is unresolved."
)


def label_for_share(share: float | None, none_reason: str | None = None) -> str:
    """The per-cell label text, or an explicit naming of why it is absent.

    Never returns "" for an unmeasurable cell: 0.0 is this statistic's
    cleanest reading, so a silent fallback would read as a perfectly
    state-sensitive cell (no silent fallbacks).

    `none_reason`: `share is None` has two
    distinct causes that must not collapse onto one wording. The default
    ("no LM rows in this cell...") is correct only for the legitimate
    no-LM-rows case; a caller whose `None` came from something else (e.g.
    `mask_constant_share` raising) must pass the real reason here, or the
    label names a cause that did not occur. Ignored when `share` is not
    `None`.
    """
    if share is None:
        return none_reason or (
            "no LM rows in this cell, so the action's dependence on the "
            "legal mask could not be measured"
        )
    return f"action is constant in `legal_mask` on {share:.2%} of this cell's LM rows"


#: Fixed-inches reservation for FIGURE_NOTE, not a bare fraction of height.
#: A bare fraction grows in absolute inches as a figure grows, while a
#: caller's own chrome stays fixed-inches -- so a fraction-anchored note
#: collides with a caller's fixed-inches chrome past some height (e.g.
#: a figure 40.73in tall). Value derived from
#: `get_window_extent()` on FIGURE_NOTE at this module's real callers'
#: narrowest width (6.0in),
#: measured at 0.1845in, plus clearance -- not tuned to match a fraction at
#: one reference height (a value tuned that way at a small height would
#: reproduce the collision at a large one).
_STATE_BLIND_BASE_IN = 0.28


def draw_state_blind_note(fig) -> None:
    """Draw FIGURE_NOTE as figure chrome, reserving its own room additively.

    The single drawing path, so the standing check can count calls instead
    of grepping source.

    Self-enforcing, not caller-trusting: anchoring at a bare fraction of
    height would leave every caller responsible for leaving enough room,
    which a caller can silently fail to do.
    `fig.subplots_adjust(bottom=...)` is called here, additively on top of
    whatever the caller already set (never `max()`: a caller's existing
    `bottom` fraction is typically already filled edge-to-edge by its own
    axis labels via `tight_layout`, so `max()` computes room that already
    has something else drawn in it -- confirmed by rendering
    `render_reliability`'s real single-arm fixture and finding the note
    landing directly on its x-axis label at the `max()`-computed position).
    The note is then drawn near the absolute bottom of the newly-added
    sliver, not relative to the caller's old `bottom`, so it lands in the
    space actually carved out for it rather than wherever the caller's own
    content already sits. Verified safe against `subplots_adjust`/
    `tight_layout` call order: calling `subplots_adjust(bottom=...)` here
    without also passing `top=` does not disturb an already-set `top`, so
    this runs safely after any renderer's own layout calls (the existing
    call order: `render_registered` calls this after `entry.render(arms)`
    returns).
    """
    height = fig.get_size_inches()[1]
    added = _STATE_BLIND_BASE_IN / height
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + added)
    y = added * 0.1
    fig.text(
        0.5,
        y,
        FIGURE_NOTE,
        ha="center",
        va="bottom",
        fontsize=5,
        color="#555555",
        wrap=True,
    )
