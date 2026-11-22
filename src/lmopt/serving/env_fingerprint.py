"""Serving-environment fingerprint for backends with no container to name.

The container lane records `image_ref`, a container digest read back from
`docker inspect`, as the token that makes two arms' decoding stacks comparable.
A job-scheduler backend (OpenPBS) has no container: the vendor
documentation records no container runtime, software arrives as environment
modules and venvs, and the PBS job script launches vLLM directly inside the
allocation. There `image_ref` is `None` as an observed fact, not as a missing
value, and this module supplies what carries the comparability instead.

A structured `serving_env` plus a hash of it, rather
than an opaque hash alone, so that a mismatch between two arms is diagnosable
(which component drifted) and not merely detectable.

Why the vLLM version is required rather than optional: the trace response field
name varies by vLLM version (`reasoning` on the pinned vLLM,
`reasoning_content` earlier). An unpinned stack can silently null every trace
an arm collects, which is exactly the failure a
fingerprint exists to catch, so a fingerprint that could not observe it would
be worse than none.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable
from typing import Any

from lmopt.serving.control import ServingError


class EnvFingerprintError(ServingError):
    """A required serving-environment component could not be observed."""


# Components whose drift changes decoding behaviour, so an arm that cannot
# observe them is refused rather than stamped (no silent fallbacks).
REQUIRED_COMPONENTS: tuple[str, ...] = ("vllm", "torch", "python")

# Components recorded for diagnosis. These may be null, but the key is always
# present: a dropped key and a null key hash differently, so a box where the
# driver version is unreadable must not collide with a box that has no such
# component at all.
OPTIONAL_COMPONENTS: tuple[str, ...] = (
    "cuda_runtime",
    "driver",
    "lockfile_sha256",
    "modules",
)


def fingerprint_hash(serving_env: dict[str, Any]) -> str:
    """sha256 of `serving_env` in a canonical form; 64-char lowercase hex.

    Canonical means: dict keys sorted at every depth, and every list sorted by
    its rendered form. Sorting lists is safe here and only here, because every
    list a serving_env carries is a set-like collection whose order is
    incidental (`modules` is whatever `module list` happened to print). Do not
    reuse this canonicaliser on `vllm_args`, where argv order is meaningful.
    """
    return hashlib.sha256(
        json.dumps(
            _canonical(serving_env), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _canonical(value: Any) -> Any:
    """Recursively normalise lists for hashing; dict key order is json's job.

    Deliberately does not sort dict keys: `json.dumps(sort_keys=True)` in the
    caller already normalises them at every depth, and a second overlapping
    defence here made that argument un-mutatable - dropping `sort_keys=True`
    left the whole suite green. One defence per property.
    """
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return sorted((_canonical(v) for v in value), key=repr)
    return value


def collect_serving_env(*, probe: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Assemble the serving-environment record, or raise.

    `probe` maps a component name to its observed value, returning None when it
    cannot be observed. It is injectable so the collection logic is testable
    with no GPU, no vLLM install and no cluster; the default probe does the
    real observation.

    Raises `EnvFingerprintError` when any REQUIRED_COMPONENTS entry is absent
    or empty. An empty string is refused exactly like absence, matching the
    precedent set by `PreservedServing.from_config`'s `image_ref` check and
    `run_sweep.py`'s served-weights gate: a supplied-but-empty value stamped
    on an arm is fabricated provenance, not partial provenance.
    """
    observe = probe if probe is not None else _default_probe

    env: dict[str, Any] = {}
    for name in REQUIRED_COMPONENTS:
        value = observe(name)
        if not value:
            raise EnvFingerprintError(
                f"serving-environment component {name!r} could not be observed "
                f"(got {value!r}). It is REQUIRED: two serving stacks that "
                "differ in it decode differently, so an arm stamped without it "
                "would claim a comparability it cannot support. Aborting "
                "before any shard rather than recording a null."
            )
        env[name] = value

    for name in OPTIONAL_COMPONENTS:
        value = observe(name)
        if not value and observe is not _default_probe:
            # driver/lockfile_sha256/modules are local-machine facts (nvidia-smi,
            # a fixed repo path, an env var), not properties of whichever
            # process answered the REQUIRED trio. `probe_from_interpreter`'s
            # remote query script never asks for these at all, so an arm
            # resolved via the server's own interpreter would record all three as
            # null -- honestly null, but a real completeness gap, since this
            # box can observe them locally regardless of which interpreter is
            # serving. Falls back to local self-inspection; skipped when
            # `observe` already is `_default_probe`, to avoid calling it twice.
            value = _default_probe(name)
        env[name] = value or None

    return env


def _default_probe(name: str) -> Any:
    """Observe one component in this process. Returns None when unobservable.

    This is the self-inspecting probe: it only sees packages installed in the
    interpreter that is running the harness. It is correct exactly when the
    harness and the server share one venv (this module's original design
    assumption). `probe_from_interpreter` below is for when they do not.
    """
    if name == "python":
        return platform.python_version()
    if name in ("vllm", "torch"):
        return _dist_version(name)
    if name == "cuda_runtime":
        return _torch_cuda_version()
    if name == "driver":
        return _nvidia_driver_version()
    if name == "lockfile_sha256":
        return _lockfile_sha256()
    if name == "modules":
        return _loaded_modules()
    return None


def probe_from_interpreter(
    python_exe: str, *, timeout_s: float = 15.0, run: Any = None
) -> Callable[[str], Any] | None:
    """Build a probe that queries a different Python interpreter than this one.

    `_default_probe` inspects the calling process's installed
    packages, which is wrong wherever the harness and the vLLM server come
    from separate environments -- as on the cluster, where vLLM is an
    environment module (`module load vllm`) with its own bundled venv,
    entirely distinct from the harness's own project venv. There,
    `_default_probe("torch")` raises `PackageNotFoundError` in the harness's
    venv even while vLLM is genuinely running torch in a different process --
    the harness would be asking the wrong interpreter about the server's
    environment. This queries `python_exe` itself instead, in one subprocess
    call for all components (not one call per component), and returns a probe
    closure over the parsed result.

    Returns None if `python_exe` cannot be run at all (subprocess creation
    fails, times out, or exits non-zero) -- a caller should treat that as
    "this component set is unobservable via this path" and fall back to
    another probe, not silently substitute an empty environment.

    `run` is a seam for tests (defaults to `subprocess.run`).
    """
    runner = run if run is not None else subprocess.run
    script = (
        "import json, platform\n"
        "out = {'python': platform.python_version()}\n"
        "try:\n"
        "    from importlib.metadata import version, PackageNotFoundError\n"
        "    for _d in ('vllm', 'torch'):\n"
        "        try:\n"
        "            out[_d] = version(_d)\n"
        "        except PackageNotFoundError:\n"
        "            out[_d] = None\n"
        "except ImportError:\n"
        "    pass\n"
        "try:\n"
        "    import torch as _t\n"
        "    out['cuda_runtime'] = _t.version.cuda\n"
        "except Exception:\n"
        "    out['cuda_runtime'] = None\n"
        "print(json.dumps(out))\n"
    )
    try:
        result = runner(
            [python_exe, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        observed = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    if not isinstance(observed, dict):
        return None
    return lambda name: observed.get(name)


def _dist_version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _torch_cuda_version() -> str | None:
    try:
        import torch

        return torch.version.cuda
    except (ImportError, AttributeError):
        # Narrow deliberately: a bare `except Exception`
        # would make a broken CUDA install indistinguishable from a CPU-only box, and
        # be inconsistent with _nvidia_driver_version's narrow catch directly
        # below for the same class of optional value.
        return None


def _nvidia_driver_version() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _lockfile_sha256() -> str | None:
    """sha256 of the harness repo's uv.lock.

    Scoped deliberately: this identifies the harness's
    own resolved dependency set, which equals the server's only when the job
    script runs both from one venv -- the shape this lane is built for, and the
    assumption `InJobServing.resolve()`'s declared-vs-observed check exists to
    trip on. It is not a general statement about the serving process.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "uv.lock"
    )
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _loaded_modules() -> list[str] | None:
    """Environment Modules exports its loaded set as $LOADEDMODULES, colon-separated."""
    raw = os.environ.get("LOADEDMODULES")
    if not raw:
        return None
    return [m for m in raw.split(":") if m]
