"""Test-session setup: fail at import if `lmopt` resolves outside this checkout."""

from __future__ import annotations

# Registered as a plugin: pytest collects fixtures by attribute on the plugin module.
pytest_plugins = ["_depth_artifacts"]

from lmopt._import_guard import assert_lmopt_is_local

assert_lmopt_is_local(__file__)
