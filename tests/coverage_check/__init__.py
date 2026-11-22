"""
Semi-synthetic coverage check for the belief-channel confidence intervals.

A prescribed agent plays the real game tree against the real solved reference and reports a belief
that departs from the true posterior by a known amount. The planted answer is computed by exact
enumeration over the tree and the agent's known policy, never from sampled rows, so a confidence
interval that fails to bracket it is a defect in the interval and not an identity.
"""
