"""Local process-pool fan-out for matchup shards.

``run_shards_local`` is the local-hardware distribution strategy: one
``ProcessPoolExecutor`` worker per shard, each calling the picklable
``run_matchup_shard`` (lmopt.orchestration.shard) fresh in its own process -
no shared state crosses the process boundary beyond the frozen ``MatchupSpec``
itself. ``max_workers=1`` bypasses the pool entirely so the single-worker path
is a plain serial loop (no executor overhead, no
process-boundary (de)serialization of results).

Why ``spawn``, not the platform-default ``fork``. ``run_matchup_shard``
runs a per-hand ``tqdm`` progress bar (match_runner.py), and tqdm holds a
global lock around bar registration/refresh. Forking a worker while that
lock is held anywhere in the parent (e.g. a still-live bar from a *prior*
serial shard, or simply an unlucky interleaving) copies the lock in its
*locked* state into the child, whose only owning thread does not exist there
- the child then deadlocks forever on its first tqdm call, silently (no
exception, no timeout). This was reproduced directly (serial shards, then a
``fork``-context pool, hangs indefinitely) and is a known fork+tqdm/lock
interaction, not specific to this repo. ``spawn`` starts each worker from a
fresh interpreter with no inherited lock state, eliminating the hazard at
the cost of a slower worker start (a fresh ``import lmopt...`` per worker).
"""

from __future__ import annotations

import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Callable, Optional

from lmopt.orchestration.shard import MatchupSpec, run_matchup_shard

_POOL_GUARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@contextmanager
def _pool_signal_guard(executor: ProcessPoolExecutor):
    """Terminate every worker in ``executor`` if this process is killed.

    Root cause: ``with ProcessPoolExecutor(...)`` cleanup never runs when the
    parent is SIGKILLed or dies to an unhandled SIGTERM -- a worker blocked in
    an HTTP request has nothing left to reap it, reparents to systemd, and
    keeps running (holding GPU-serving load) for days. This installs
    SIGTERM/SIGINT handlers for the executor's lifetime that shut the pool
    down and terminate its worker processes directly before this process
    dies, restoring whatever disposition was in place before on every exit
    path (normal or signalled) so a caller's own handler, if any, is never
    silently dropped.

    Caller obligation: this context manager
    must stay entered for the executor's entire lifetime, including its own
    teardown -- i.e. wrap this around an explicit
    ``try/finally: executor.shutdown(wait=True)``, never around a bare
    ``with ProcessPoolExecutor(...)`` block where this guard is the inner
    context manager. Nested the other way, this guard's own ``finally``
    (restoring the ambient disposition) runs on the way out before
    ``ProcessPoolExecutor.__exit__``'s ``shutdown(wait=True)`` blocks joining
    in-flight workers with no timeout -- a signal delivered during exactly
    that block then hits the restored (pre-guard) disposition and orphans
    the workers this guard exists to kill. See ``run_shards_local`` for the
    correct shape.
    """
    previous: dict[int, Any] = {}

    def _restore() -> None:
        """
        put back every signal's earlier disposition. A disposition installed outside Python
        reads back as None, which signal.signal cannot reinstall; every other signal is
        restored first, then the guard refuses, naming the ones it could not put back
        :returns: None
        :raises RuntimeError: when an earlier disposition was None and so could not be restored
        """
        unrestorable = []
        for sig, old in previous.items():
            if old is None:
                unrestorable.append(signal.Signals(sig).name)
                continue
            signal.signal(sig, old)
        if unrestorable:
            raise RuntimeError(
                f"could not restore the handler for {', '.join(unrestorable)}: it was installed "
                f"outside Python, so signal.signal returned None for it. The pool guard's "
                f"handler is still installed there; install a Python handler before the pool "
                f"starts"
            )

    def _handle(signum: int, _frame) -> None:
        # Reentrancy: if a second guarded signal
        # arrives while this handler is still running (e.g. SIGINT then
        # SIGTERM before the first finishes), this function runs again,
        # nested, on the same thread. Verified safe: `executor.shutdown()`
        # re-acquires and releases `_shutdown_lock` per call (no deadlock,
        # since it isn't held across the blocking join) and is idempotent on
        # an already-shutting-down pool; `Process.terminate()` ->
        # `popen_fork.Popen._send_signal` checks `self.returncode is None`
        # first and catches `ProcessLookupError`, so re-terminating an
        # already-dead worker is a no-op, not an error. Nothing here holds a
        # lock across a call that could re-enter this handler.
        #
        # Capture the worker table before shutdown(), not after:
        # ProcessPoolExecutor.shutdown() unconditionally sets
        # self._processes = None as part of its own teardown (verified by
        # reading concurrent.futures.process.ProcessPoolExecutor.shutdown,
        # Python 3.12.11) regardless of `wait`, so
        # reading `_processes` after calling shutdown() silently loses every
        # live worker reference -- the exact defect this guard exists to
        # avoid, just moved one line down. `_processes` (pid ->
        # multiprocessing.Process) is ProcessPoolExecutor's private worker
        # table; concurrent.futures has no public accessor for live worker
        # processes as of this version. Prefer a public equivalent if a
        # future CPython adds one.
        processes = list((executor._processes or {}).values())
        # wait=False: a worker mid-request (the exact hazard above) must not
        # block this handler waiting for it to finish on its own.
        # cancel_futures=True: drop anything not yet started.
        executor.shutdown(wait=False, cancel_futures=True)
        # shutdown() stops the pool handing out new work; it does not touch a
        # worker already running a task, which is the orphan case -- hence the
        # explicit terminate() over the processes captured above.
        for process in processes:
            process.terminate()
        # Restore before re-raising: re-delivering the signal to a handler
        # still installed here would recurse into this function instead of
        # reaching the ambient disposition. For SIGTERM the ambient
        # disposition is ordinarily SIG_DFL, so this re-kill terminates the
        # process via the kernel and the exit code stays honest
        # (killed-by-signal, not "handled and exited 0/1"). SIGINT is not
        # symmetric: its ambient disposition is
        # ordinarily Python's own `signal.default_int_handler`, so re-raising
        # it here raises `KeyboardInterrupt` into whatever was running rather
        # than killing the process outright -- honest in the sense that the
        # interpreter still sees the interrupt and does not swallow it, but
        # not a true signal-death exit code.
        _restore()
        os.kill(os.getpid(), signum)

    for sig in _POOL_GUARDED_SIGNALS:
        previous[sig] = signal.signal(sig, _handle)
    try:
        yield
    finally:
        _restore()


def run_shards_local(
    specs: list[MatchupSpec],
    max_workers: int,
    on_result: Optional[Callable[[MatchupSpec, str], None]] = None,
    on_error: Optional[Callable[[MatchupSpec, BaseException], None]] = None,
) -> list[Optional[str]]:
    """Run shards locally; returns parquet paths in ``specs`` order.

    ``max_workers=1`` runs shards directly in this process, in order - no
    pool. For ``max_workers > 1``, shards are submitted to a
    ``ProcessPoolExecutor`` and results are gathered back into ``specs``
    order regardless of completion order.

    ``on_result``, if given, is called ``(spec, out_path)`` as each shard
    finishes - in true completion order for the pool, in spec order for the
    serial path - so a caller (e.g. a tqdm progress bar in run_sweep.py) can
    report real progress instead of jumping from 0 to done.

    Failure handling, and why it is opt-in. With
    ``on_error`` absent (the default), any
    shard exception cancels the run: outstanding futures are cancelled and
    the exception is re-raised with the offending spec named, so a
    partial-sweep failure is never silent. Parquet files already written by
    completed shards are left in place.

    With ``on_error`` supplied, a failing shard is instead reported to that
    callback, ``(spec, exception)``, and the run continues. This exists
    because the alternative has a measured cost: one goofspiel shard raising
    ``Request timed out`` aborted a 100-shard pilot at 71/100 and discarded
    roughly 3.5 h of remaining work. The callback receives the exception
    itself, not a count, so the caller can name the offender in its own
    record (no silent fallbacks).

    The failed shard's slot in the returned list is ``None``. It is not
    omitted: the list stays index-aligned with ``specs``, because
    run_sweep.py zips the two together to build its per-matchup summary and
    a shorter list would slide every later path onto the wrong spec -
    turning an incomplete arm into a complete-looking one. Continuing past a failure
    is therefore always paired with an explicit ``None``; there is no path
    on which a caller sees fewer entries than it asked for.

    Kept opt-in rather than made the default so ``run_shards_ray``'s "same
    contract as run_shards_local" docstring stays true (that runner still
    raises), and so callers that genuinely want all-or-nothing - parity
    runs - keep that behaviour.
    """
    if max_workers == 1:
        results: list[Optional[str]] = []
        for spec in specs:
            try:
                path = run_matchup_shard(spec)
            except Exception as exc:
                # Bare re-raise, not the pool branch's RuntimeError wrapper:
                # the serial loop propagates the shard's own exception, so a
                # non-opted-in caller sees exactly what the shard raised.
                if on_error is None:
                    raise
                on_error(spec, exc)
                results.append(None)
                continue
            results.append(path)
            if on_result is not None:
                on_result(spec, path)
        return results

    spawn_ctx = multiprocessing.get_context("spawn")
    # Built without `with ProcessPoolExecutor(...)` deliberately:
    # `_pool_signal_guard` must be the outer context manager,
    # so its signal handlers are still installed for the `executor.shutdown(
    # wait=True)` below -- the call that can block indefinitely joining
    # in-flight workers, and therefore the exact window a kill must still be
    # able to interrupt. Nesting the guard inside a `with ProcessPoolExecutor`
    # block would let the guard's own `finally` restore the ambient
    # disposition before that blocking shutdown ran, orphaning workers on
    # this function's own exception-unwind path.
    executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx)
    with _pool_signal_guard(executor):
        try:
            # Index (not the spec itself) is the map key: specs are plain
            # dataclasses and two shards could legitimately be field-equal,
            # which would collide as dict keys.
            future_to_index = {
                executor.submit(run_matchup_shard, spec): i
                for i, spec in enumerate(specs)
            }
            pool_results: dict[int, str] = {}
            try:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        path = future.result()
                    except Exception as exc:
                        if on_error is None:
                            raise RuntimeError(
                                f"Shard failed for spec={specs[index]!r}: {exc}"
                            ) from exc
                        on_error(specs[index], exc)
                        continue
                    pool_results[index] = path
                    if on_result is not None:
                        on_result(specs[index], path)
            except Exception:
                for future in future_to_index:
                    future.cancel()
                raise
        finally:
            # Mirrors ProcessPoolExecutor.__exit__ exactly (it calls
            # shutdown(wait=True)); done explicitly here, inside the guard's
            # scope, rather than via `with ProcessPoolExecutor(...)`, so this
            # blocking call is itself covered by the guard -- see the comment
            # above and `_pool_signal_guard`'s own docstring.
            executor.shutdown(wait=True)

    # `.get(i)` rather than `[i]`: a shard reported through on_error left no
    # entry, and its slot must be an explicit None in the aligned result list
    # (see this function's docstring on why the list is never shortened).
    return [pool_results.get(i) for i in range(len(specs))]
