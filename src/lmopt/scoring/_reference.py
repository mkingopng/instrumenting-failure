"""Shared duck-typing guard for PARTIAL vs full reference policies.

The guard exists in exactly one place: duplicated in `property7.py` and
`property8.py`, a later loosening applied to one copy would silently miss the
other. `src/lmopt/harness/match_runner.py` (a third call site) consumes it too.

No pyspiel import here - pure duck-typing keeps this module import-safe from
both `lmopt.scoring` (offline scorers; scoring is offline) and
`lmopt.harness` (which does depend on pyspiel, but should not need to import
`lmopt.games.depth_reference` just to ask "is this partial?").
"""

from __future__ import annotations


def is_partial_reference(pol) -> bool:
    """True for a DepthReference-style partial policy.

    Duck-typed on `lookup` rather than isinstance, so scoring keeps no import
    dependency on the games package (scoring is offline).

    Do not loosen these attribute names. OpenSpiel's TabularPolicy has
    `state_lookup` - verified: TabularPolicy.lookup False,
    .n_infosets False, .state_lookup True; DepthReference.lookup True,
    .n_infosets True. A substring check, or duck-typing on `state_lookup`,
    inverts this guard: every full CFR+ reference would read as partial and
    P7b/P8 would silently go "not applicable" for all five canonical games.
    Both attributes are required precisely so the test is unambiguous.
    """
    return hasattr(pol, "lookup") and hasattr(pol, "n_infosets")
