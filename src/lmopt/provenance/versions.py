"""
The library versions a collection run records about the machine it ran on.

A run's `run_meta.json` (`lmopt.logging.run_meta.write_run_meta`) records them so the
run can later be compared against a reference measurement, such as the P3 sentinel
anchors, that is only valid for one OpenSpiel build. The reference cache key
(`lmopt.games.policy_cache.cache_components`) reads the same two versions on its own;
a standing test asserts the two spell each version identically, so a voided anchor
always means a real difference between the machines and never a formatting drift.
"""

from __future__ import annotations

#: The keys `library_versions` returns, and therefore the keys a consumer may
#: look for in a run's `environment` block.
LIBRARY_VERSION_KEYS = ("pyspiel_version", "numpy_version")


def library_versions() -> dict[str, str]:
    """
    the OpenSpiel and numpy versions of the environment this call runs in
    Imported inside the function, not at module scope, so a caller on a
    platform without the OpenSpiel wheel fails at the point it actually needs
    a game rather than at import of an unrelated provenance module.
    :returns: a dict carrying every key in `LIBRARY_VERSION_KEYS` (dict[str, str])
    """
    import numpy as np
    import pyspiel

    if not hasattr(pyspiel, "__version__"):
        raise RuntimeError(
            "pyspiel has no __version__, so the run cannot record which OpenSpiel build it "
            "ran on; install an OpenSpiel wheel that carries one"
        )
    return {
        "pyspiel_version": pyspiel.__version__,
        "numpy_version": np.__version__,
    }
