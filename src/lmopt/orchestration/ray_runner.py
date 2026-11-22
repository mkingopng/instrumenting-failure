"""Ray shard runner for multi-node fan-out.

``run_shards_ray`` is the Ray-backed sibling of ``run_shards_local``
(orchestration/local.py): same distribution unit (``MatchupSpec`` in,
``run_matchup_shard`` per task, one Parquet path out, results returned in
``specs`` order, any shard failure cancels the run and re-raises naming the
offending spec) but scheduled by a Ray cluster instead of a local
``ProcessPoolExecutor``.

Scope (serving alignment): Ray here distributes match shards only.
Model serving and cross-node routing belong to the separate serving setup
project's LiteLLM gateway layer - this
module builds no model-serving, node-discovery, or routing logic. LM shards
reach their model exactly as the serial/local-pool paths already do, via
``spec.vllm_base_url``. ``resources_per_shard`` (e.g. ``{"vllm_qwen": 1}``)
is a Ray custom-resource tag so shards *can* be steered toward a box near a
serving endpoint later; on a single-box cluster it is trivially satisfiable
by every worker (or omitted entirely).

Caller-managed init. Unlike ``run_shards_local`` (which owns its
``ProcessPoolExecutor`` for the duration of one call), this function does
not call ``ray.shutdown()`` - a Ray cluster is a longer-lived resource than
one sweep invocation, and the caller (``scripts/run_sweep.py``, or a test's
``ray.init`` fixture) is expected to control its lifetime. As a convenience
for standalone use, ``run_shards_ray`` will call ``ray.init(address=address)``
if no Ray runtime is active yet, but ownership (and therefore teardown)
stays with whoever actually wants the cluster torn down.

``ray`` is imported lazily (inside this function, not at module scope) so
importing ``lmopt.orchestration.ray_runner`` never requires Ray to be
installed - only calling ``run_shards_ray`` does.

uv-run hook opt-out (load-bearing). Ray >=2.43
ships ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` defaulting to *on*: if the driver
process was started via ``uv run`` (every entry point in this repo is), Ray
rewrites each worker's ``py_executable`` to re-launch under ``uv run`` with
a staged copy of the working directory. That staged copy has no ``.venv``
and uv re-resolves a *fresh, unpinned* environment from ``pyproject.toml``
base deps only - no ``orchestration``/``dev`` extras - so workers die on
``import ray`` (ModuleNotFoundError) and every ``ray.get`` hangs forever
(reproduced on-box). Beyond the hang, an
ephemeral per-worker re-resolution is exactly what a pinned-environment
research instrument must not do (FP determinism). Workers
are therefore pinned to the driver's own installed venv by disabling the
hook - ``setdefault`` so an operator who explicitly set the variable still
wins. The variable must be set **before the first ``import ray`` in the
process** (Ray bakes it into ``ray_constants`` at import time), hence the
module-top placement; ``run_shards_ray`` fail-loudly guards the too-late
case rather than silently proceeding into the hang.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from lmopt.orchestration.shard import MatchupSpec, run_matchup_shard

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def _guard_uv_run_hook() -> None:
    """Fail loudly if Ray baked the uv-run hook on and the driver is under uv run.

    Ray reads RAY_ENABLE_UV_RUN_RUNTIME_ENV once, at import time. If other
    code imported ray before this module set the opt-out (module top), the
    uv-run worker-relaunch hook is already active and every shard task would
    hang forever on workers that cannot import ray (module docstring).
    Raising converts that silent hang into an actionable error (no silent
    fallbacks).
    """
    from ray._private import ray_constants
    from ray._private.runtime_env.uv_runtime_env_hook import _get_uv_run_cmdline

    baked_on = bool(getattr(ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", False))
    if baked_on and os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV") != "1":
        if _get_uv_run_cmdline():
            raise RuntimeError(
                "Ray was imported (uv-run worker-relaunch hook baked ON) before "
                "lmopt.orchestration.ray_runner could disable it, and this driver "
                "runs under `uv run` - shard workers would be re-launched into a "
                "fresh uv environment without ray/lmopt installed and hang "
                "forever. Fix: import lmopt.orchestration.ray_runner before "
                "importing ray, or export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0. "
                "(Export RAY_ENABLE_UV_RUN_RUNTIME_ENV=1 to opt back in "
                "explicitly.)"
            )


def _shard_task(spec: MatchupSpec) -> str:
    """Plain (undecorated) wrapper so this module imports without Ray.

    ``ray.remote`` is applied to this function dynamically, inside
    ``run_shards_ray``, once Ray is known to be importable.
    """
    return run_matchup_shard(spec)


def run_shards_ray(
    specs: list[MatchupSpec],
    *,
    address: Optional[str] = None,
    resources_per_shard: Optional[dict] = None,
    on_result: Optional[Callable[[MatchupSpec, str], None]] = None,
) -> list[str]:
    """Run shards on a Ray cluster; returns parquet paths in ``specs`` order.

    Same contract as ``run_shards_local``: every spec is submitted as one
    Ray task (``.options(resources=resources_per_shard or {})``), results
    are gathered back in ``specs`` order (not completion order), and any
    shard exception cancels the run - outstanding tasks are cancelled and
    the exception is re-raised with the offending spec named, so a partial
    sweep failure is never silent. Parquet files already written by
    completed shards are left in place.

    ``on_result``, if given, is called ``(spec, out_path)`` as each result is
    collected - in ``specs`` order (this function ``ray.get``s tasks one at
    a time, in submission order), mirroring the serial (``max_workers=1``)
    reporting order of ``run_shards_local`` rather than the pool's true
    completion order.

    ``address``: passed to ``ray.init`` only if no Ray runtime is active yet
    (see module docstring on caller-managed init); ``None`` starts/attaches
    to a local single-box cluster, matching ``ray start --head`` with no
    remote address.
    """
    import ray

    _guard_uv_run_hook()

    if not ray.is_initialized():
        ray.init(address=address)

    remote_task = ray.remote(_shard_task)
    opts = resources_per_shard or {}
    futures = [remote_task.options(resources=opts).remote(spec) for spec in specs]

    results: list[str] = []
    try:
        for i, future in enumerate(futures):
            try:
                path = ray.get(future)
            except Exception as exc:
                raise RuntimeError(
                    f"Shard failed for spec={specs[i]!r}: {exc}"
                ) from exc
            results.append(path)
            if on_result is not None:
                on_result(specs[i], path)
    except Exception:
        for future in futures:
            ray.cancel(future, force=True)
        raise

    return results
