"""In-job serving provider for job-scheduler backends (OpenPBS).

A `pbs-injob` provider performs no lifecycle actions: the PBS job script starts
vLLM inside the allocation, and the harness -- running in the same job -- only
reads back what is serving. Lifecycle is the job script's, and teardown is the
scheduler's, so this module never shells out.

It sits between the two other providers rather than beside them:

- `spark-systemd` (ServingController) has ssh + docker, so it reads the image
  digest and the container's real Cmd vector off the box.
- `preserved` has neither, so it records `image_ref` from config and reports
  `vllm_args=None` (with a reason), `container=None`, `unit=None` as documented "not observed".
- `pbs-injob` has no container to inspect but does share a machine with the
  server, so it observes the real argv off the process table and records the
  serving environment directly.

Consequently `image_ref` is None here as an OBSERVED fact -- the vendor
documentation records no container runtime on the cluster -- and the comparability
role it plays elsewhere is carried by `serving_env` + `serving_env_hash`
(see lmopt.serving.env_fingerprint).

The no-silent-fallbacks invariant is moved, not weakened: `preserved` refuses an
arm with no `image_ref`, and this provider refuses an arm with no `serving_env`,
on identical reasoning. An arm that can record neither has no serving-stack
provenance and must not be collected.

Config shape:
    serving:
      provider: pbs-injob
      serving_env:            # required, non-empty
        vllm: "0.31.0"
        torch: "2.13.0"
        python: "3.13"      # major.minor accepts any patch release
        ...                   # see env_fingerprint.collect_serving_env
      # serving_env_hash is derived, never read from config
      # ssh_host / image_ref / vllm_args are ignored if present
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from lmopt.serving.control import (
    ResolvedServing,
    ServingController,
    ServingError,
    parse_nvidia_smi_gpu_query,
)
from lmopt.serving.env_fingerprint import (
    REQUIRED_COMPONENTS,
    EnvFingerprintError,
    collect_serving_env,
    fingerprint_hash,
    probe_from_interpreter,
)


class InJobServingError(ServingError):
    """Failures specific to the in-job provider."""



def declaration_matches(component: str, declared: Any, observed: Any) -> bool:
    """
    whether an observed serving component satisfies the value a config declares for it. Every
    component compares exactly, with one exception: a Python declaration of major.minor only
    ("3.13") accepts any patch release of that minor version, because the patch release is
    whatever the installer picked and the observed full version is what the run records
    :param component: the component name, e.g. "vllm", "torch" or "python" (str)
    :param declared: the value the config declares (str)
    :param observed: the value read from the running server (str or None)
    :returns: True when the observed value satisfies the declaration (bool)
    """
    if observed == declared:
        return True
    if (
        component == "python"
        and isinstance(declared, str)
        and isinstance(observed, str)
        and declared.count(".") == 1
    ):
        return observed.startswith(declared + ".")
    return False


@dataclass(frozen=True)
class InJobServing:
    """An in-job resolver: no lifecycle, HTTP + process-table provenance."""

    model_id: str
    base_url: str
    # The config-declared environment. It is a tripwire, not the record: the
    # value stamped on the arm is what `resolve()` observes, and a disagreement
    # between the two is an abort. Same shape as the
    # `expected_chat_template_hash` pin, and for the same reason -- a declared
    # value that is never checked against the running system is a statement of
    # intent wearing a measurement's clothes.
    expected_serving_env: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls, model_id: str, base_url: str, serving_config: dict
    ) -> InJobServing:
        """Factory from the config's `serving:` block.

        Raises `InJobServingError` if `serving_env` is missing or empty. An
        empty mapping is refused exactly like absence, matching
        `PreservedServing.from_config`'s empty-`image_ref` rule and
        `run_sweep.py`'s served-weights gate: a supplied-but-empty value
        stamped on an arm is fabricated provenance, not partial provenance.

        A config-supplied `serving_env_hash` is deliberately not read. The hash
        is derived from `serving_env` at resolve time, so the two can never
        disagree -- a stamped hash that does not describe the environment
        recorded beside it is the provenance lie this whole surface exists to
        prevent.
        """
        # One check, not two: a separate presence check ("non-empty
        # serving_env") would be fully subsumed by the content check ("every
        # REQUIRED_COMPONENTS key present"), since an empty dict already fails
        # it, and a subsumed check cannot have a mutation that reddens.
        # `preserved` refuses an unusable image_ref; this provider refuses an
        # unusable serving_env on the same reasoning, in one place.
        serving_env = serving_config.get("serving_env") or {}
        missing = [k for k in REQUIRED_COMPONENTS if not serving_env.get(k)]
        if missing:
            raise InJobServingError(
                f"serving.provider=pbs-injob for {model_id} requires a "
                f"`serving_env` carrying every required component (got "
                f"{serving_config.get('serving_env')!r}, missing {missing!r}). "
                f"Required: {list(REQUIRED_COMPONENTS)}. There is no container "
                "to digest on a job-scheduler backend, so the fingerprint IS "
                "this arm's serving-stack provenance; without it the arm "
                "records none usable, which is exactly the `reasoning` vs "
                "`reasoning_content` drift class this "
                "check exists to catch."
            )
        return cls(
            model_id=model_id,
            base_url=base_url,
            expected_serving_env=dict(serving_env),
        )

    def resolve(
        self,
        *,
        argv_scan: Callable[[], tuple[str, ...] | ArgvNotObserved] | None = None,
        env_probe: Callable[[str], Any] | None = None,
        nvidia_smi_probe: Callable[[], subprocess.CompletedProcess] | None = None,
    ) -> ResolvedServing:
        """Confirm the served model, observe the environment and argv, build a
        ResolvedServing.

        The environment is OBSERVED here, not copied from config. Config
        intent is not trusted for the argv, and `serving_env` is the more
        load-bearing value -- the only serving-stack provenance a pbs-injob
        arm has, since `image_ref` is null by design -- so it cannot come
        straight from a hand-typed YAML block either.

        What is observed is, by preference, the SERVER'S own environment: its
        pid is found by the same /proc scan `argv_scan` uses
        (`resolve_server_python_exe`), and its interpreter is queried directly
        (`env_fingerprint.probe_from_interpreter`). Inspecting the calling
        process instead assumes the harness and the server share one venv
        (true for a simple PBS job script that activates one venv for both),
        which is wrong on the cluster, where vLLM is an environment module with
        its own bundled venv entirely separate from the harness's project
        venv, so the harness's own `importlib.metadata` sees no `torch` at all
        even while vLLM is genuinely running it. When the server's own
        interpreter cannot be resolved or queried (no match, or the subprocess
        query fails), this falls back to self-inspection, which is still
        correct for a shared-venv deployment. The config's declared
        `serving_env` is the tripwire either way -- if what actually gets
        observed does not match what was declared, this method aborts instead
        of stamping a fingerprint that describes the wrong process.

        `argv_scan`, `env_probe` and `nvidia_smi_probe` are injectable so the
        resolution logic is testable without a running vLLM; the defaults do
        the real observation.

        Exact-id check (tuple equality, never substring): mirrors
        `PreservedServing.resolve()` and `ServingController` semantics.
        """
        # Reuse ServingController.current() for the /v1/models read exactly as
        # PreservedServing does: it implements the exact-id membership check,
        # the unreachable-vs-parseable distinction and the exception semantics
        # this provider needs. ssh_host="" so no lifecycle method can be called.
        controller = ServingController(ssh_host="", base_url=self.base_url)
        served_model = controller.current()
        if served_model is None or self.model_id not in served_model.served_ids:
            served_desc = (
                served_model.served_ids
                if served_model is not None
                else "nothing (endpoint unreachable)"
            )
            raise InJobServingError(
                f"pbs-injob provider: {self.base_url} does not currently serve "
                f"{self.model_id!r} (serves {served_desc!r})"
            )

        from lmopt.serving.chat_template import resolve_chat_template_hash

        chat_template_hash = resolve_chat_template_hash(self.base_url, self.model_id)

        # Resolve which probe to use, preferring the server's own interpreter
        # over self-inspection (see the resolve() docstring).
        # `env_probe` (test injection) always wins outright.
        try:
            own_port = urlsplit(self.base_url).port
        except ValueError as exc:
            raise InJobServingError(
                f"pbs-injob provider: base_url {self.base_url!r} has an invalid "
                f"port ({exc})"
            ) from exc
        own_port_str = str(own_port) if own_port is not None else None
        if env_probe is not None:
            resolved_probe = env_probe
        else:
            resolved_probe = None
            # `own_port`: model_id alone cannot tell this job's own server
            # from a sibling array index serving the identical model on the
            # same node ("N different vllm serve command lines").
            # `self.base_url` is this job's own rendered endpoint, so its
            # port is the unambiguous disambiguator; `None` here (no port in
            # base_url) falls back to the model-id-only match.
            server_exe = resolve_server_python_exe(self.model_id, port=own_port_str)
            if server_exe is not None:
                resolved_probe = probe_from_interpreter(server_exe)
            # Still None here means: no server process found, or querying its
            # interpreter failed. collect_serving_env(probe=None) falls back
            # to self-inspection -- correct for a shared-venv deployment, and
            # the least-wrong default when the server's own env truly cannot
            # be reached.

        # Observe. collect_serving_env raises when a required component cannot
        # be observed, so an arm that could stamp nothing meaningful aborts here
        # rather than at read time (no silent fallbacks).
        try:
            observed_env = collect_serving_env(probe=resolved_probe)
        except EnvFingerprintError as env_exc:
            raise InJobServingError(
                f"pbs-injob provider: cannot observe the serving environment "
                f"for {self.model_id}: {env_exc}"
            ) from env_exc

        unknown = sorted(k for k in self.expected_serving_env if k not in observed_env)
        if unknown:
            raise InJobServingError(
                f"pbs-injob provider: the declared serving_env names {unknown} "
                f"which are not serving-environment components "
                f"({sorted(observed_env)}); an unchecked declaration is refused."
            )

        disagreements = sorted(
            f"{k}: declared {self.expected_serving_env[k]!r}, observed {observed_env.get(k)!r}"
            for k in self.expected_serving_env
            if k in observed_env
            and not declaration_matches(k, self.expected_serving_env[k], observed_env.get(k))
        )
        if disagreements:
            raise InJobServingError(
                f"pbs-injob provider: the declared serving_env does not describe "
                f"the observed environment for {self.model_id}: "
                + "; ".join(disagreements)
                + ". Either the job script served from a different environment "
                "than the config declares, or the config is stale. Aborting "
                "rather than stamping a fingerprint of the wrong process."
            )

        scan = (
            argv_scan
            if argv_scan is not None
            else lambda: scan_vllm_argv(self.model_id, port=own_port_str)
        )
        scanned = scan()
        if isinstance(scanned, ArgvNotObserved):
            vllm_args: tuple[str, ...] | None = None
            unobserved_reason: str | None = scanned.reason
        else:
            vllm_args = tuple(scanned)
            unobserved_reason = None

        # GPU-card provenance: local subprocess call, same physical node as
        # vLLM (this provider has no ssh_host at all -- the job is the node).
        # Mirrors env_fingerprint.py's _nvidia_driver_version() pattern.
        # nvidia_smi_probe is injectable for tests; the default runs the real
        # command.
        #
        # Resolved through the module-level `_default_nvidia_smi_probe`, not
        # a closure: a closure would be rebuilt on every call and expose no
        # module attribute, so nothing could spy on it and the default
        # path's wiring would have no reachability anchor. Module-level, it
        # is monkeypatchable exactly as `env_fingerprint._default_probe` is.
        probe = (
            nvidia_smi_probe
            if nvidia_smi_probe is not None
            else _default_nvidia_smi_probe
        )
        gpu_name: str | None = None
        gpu_memory_total_mib: int | None = None
        try:
            smi_result = probe()
            if smi_result.returncode == 0:
                gpu_name, gpu_memory_total_mib = parse_nvidia_smi_gpu_query(
                    smi_result.stdout
                )
        except (OSError, subprocess.SubprocessError):
            pass  # gpu_name/gpu_memory_total_mib stay explicitly None

        return ResolvedServing(
            model_id=self.model_id,
            base_url=self.base_url,
            # OBSERVED None: nothing is containerised on this backend. Not the
            # "unrecorded" None that PreservedServing refuses.
            image_ref=None,
            vllm_args=vllm_args,
            container=None,  # no container runtime on this backend
            max_model_len=served_model.max_model_len,
            chat_template_hash=chat_template_hash,
            unit=None,  # no systemd unit; the job is the lifetime
            # The OBSERVED environment, never the declared one.
            serving_env=dict(observed_env),
            serving_env_hash=fingerprint_hash(observed_env),
            gpu_name=gpu_name,
            gpu_memory_total_mib=gpu_memory_total_mib,
            vllm_args_unobserved_reason=unobserved_reason,
        )


def _default_nvidia_smi_probe() -> subprocess.CompletedProcess:
    """Ask the local machine which card it is serving on.

    Module-level, not a closure inside `InJobServing.resolve()`, for the same reason
    `env_fingerprint._default_probe` is module-level: a default that only
    exists inside the function that uses it cannot be spied on, so nothing
    can prove `resolve()` reaches the real probe when no fake is injected.
    `check=False` because a non-zero return is an ordinary outcome here (no
    GPU, no nvidia-smi on path) that `resolve()` degrades to a null card,
    not an error to raise.
    """
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@dataclass(frozen=True)
class ArgvNotObserved:
    """
    the argv scan could not name this job's server, and why
    :param reason: what the scan saw, in words (str)
    """

    reason: str


@dataclass(frozen=True)
class _ServerMatch:
    argv: tuple[str, ...]
    pid: str  # a representative pid for this argv; TP forks share one argv


def _argv_has_port(argv: tuple[str, ...], port: str) -> bool:
    """True if `argv` launches on `port`, matching both the space-separated
    (`--port 46781`) and `=`-joined (`--port=46781`) forms `vllm serve`
    accepts. Used to disambiguate sibling array indices serving the same
    model on the same node: model-id-only matching cannot tell one array
    index's own server from another index's, both serving the exact same
    `model_id` concurrently on a `default_shared` multi-GPU node, and the
    scan then reports "N different vllm serve command lines" (one-seed array
    indices produce N up to 6).
    """
    for i, tok in enumerate(argv):
        if tok == "--port" and i + 1 < len(argv) and argv[i + 1] == port:
            return True
        if tok == f"--port={port}":
            return True
    return False


def _find_server(
    model_id: str,
    *,
    proc_root: str = "/proc",
    log: Callable[[str], None] | None = None,
    port: str | None = None,
) -> _ServerMatch | ArgvNotObserved:
    """Shared /proc-scan core for `scan_vllm_argv` and `resolve_server_python_exe`.

    Shared so both reuse the identical matching semantics -- same
    dedupe-by-argv, same ambiguity handling -- rather than two scans that
    could silently drift apart. See `scan_vllm_argv`'s docstring for the
    three-outcomes contract this implements; this function is that contract,
    `scan_vllm_argv` is a thin wrapper over it.

    `port`, when given, additionally requires the
    matched argv to serve on that exact port -- see `_argv_has_port`. Without
    it, `model_id` alone cannot distinguish this job's own server from a
    sibling array index serving the identical model on the same node.
    """
    emit = log if log is not None else _warn
    self_pid = str(os.getpid())
    matches: dict[tuple[str, ...], str] = {}

    try:
        entries = os.listdir(proc_root)
    except OSError as exc:
        raise InJobServingError(
            f"pbs-injob provider: cannot read the process table at {proc_root!r} "
            f"({exc}). This provider runs inside the same job as the server it "
            "is describing, so an unreadable /proc means the environment is not "
            "what it assumes. Refusing to report 'no server found', which is a "
            "different fact."
        ) from exc

    for pid in (name for name in entries if name.isdigit()):
        if pid == self_pid:
            continue  # never mistake the harness for the server it is reading
        try:
            with open(os.path.join(proc_root, pid, "cmdline"), "rb") as fh:
                raw = fh.read()
        except OSError:
            # Per-entry failure is ordinary and stays quiet: the process exited
            # between listdir and open, or belongs to another user. Distinct
            # from the whole-tree failure above, which is not ordinary.
            continue
        argv = tuple(
            part for part in raw.decode("utf-8", "replace").split("\0") if part
        )
        if not _is_vllm_server(argv) or model_id not in argv:
            continue
        if port is not None and not _argv_has_port(argv, port):
            continue
        matches.setdefault(argv, pid)  # first pid seen for this argv wins

    # A SET of distinct argvs, not a list. vLLM does not run as one process:
    # under tensor parallelism the API server spawns engine-core/worker
    # children, and a forked child's cmdline is byte-identical to its
    # parent's. Counting those as rival candidates would make the ambiguity
    # guard fire and silently drop the argv in exactly the multi-GPU
    # configuration the job requests.
    # Identical argv vectors are one server observed several times.
    if len(matches) == 1:
        ((argv, pid),) = matches.items()
        return _ServerMatch(argv=argv, pid=pid)
    if not matches:
        reason = (
            f"no `vllm serve` process for {model_id!r}"
            + (f" on port {port}" if port is not None else "")
            + f" found under {proc_root}"
        )
    else:
        reason = (
            f"{len(matches)} DIFFERENT `vllm serve` command lines for {model_id!r}"
            + (f" on port {port}" if port is not None else "")
            + f" under {proc_root}; cannot say which one serves this endpoint"
        )
    emit(f"  serving: {reason}; vllm_args recorded as NOT OBSERVED.")
    return ArgvNotObserved(reason=reason)


def scan_vllm_argv(
    model_id: str,
    *,
    proc_root: str = "/proc",
    log: Callable[[str], None] | None = None,
    port: str | None = None,
) -> tuple[str, ...] | ArgvNotObserved:
    """The real argv of this ARM's vLLM server on this node, or `ArgvNotObserved`
    (carrying the reason) if it is not identifiable.

    Reads /proc rather than trusting config, because the serving block must
    name settings as actually applied.

    `model_id` scopes the match, and is not optional. The cluster's GPU nodes are `default_shared`, so another
    user's job on the same node can be running `vllm serve` too. An unscoped
    scan that missed our own server's launch form would find theirs, see
    exactly one match, and stamp their argv as this arm's provenance. Requiring
    our model id in the argv makes that misattribution impossible without also
    serving our model.

    `port`, when given, narrows further to this job's
    own port -- required once sibling jobs on the same node can serve the
    identical model (the one-seed-per-job array design), which `model_id`
    alone cannot disambiguate. See `_argv_has_port`.

    `proc_root` and `log` are seams for tests: this is the one part of the
    module that touches the real machine, and a fake /proc tree is the only way
    to exercise the dedupe, ambiguity and unreadable-entry branches.

    Three outcomes, kept distinct rather than collapsed into one silent `()`
    (as in `control.py:_find_container`, a failed query must never read as
    "the query succeeded and found nothing"):

    - the /proc tree is unreadable: raises (via `_find_server`). On this lane
      we are a process on a Linux node reading our own node's /proc, so a
      failure here means the environment is not what this provider assumes,
      which is worth an abort.
    - no match, or an ambiguous one: returns `ArgvNotObserved` with the reason,
      having said so on stderr. Never an empty tuple: an empty argv would read
      as a server launched with no arguments.
    - exactly one distinct match: returns it.
    """
    match = _find_server(model_id, proc_root=proc_root, log=log, port=port)
    return match if isinstance(match, ArgvNotObserved) else match.argv


def resolve_server_python_exe(
    model_id: str,
    *,
    proc_root: str = "/proc",
    log: Callable[[str], None] | None = None,
    port: str | None = None,
) -> str | None:
    """The path to the Python interpreter actually running this ARM's vLLM
    server, or None if the server cannot be identified.

    `_default_probe`'s `serving_env` collection inspects the calling process
    (the harness), which is wrong wherever the harness and the server come
    from separate environments -- as on the cluster, where vLLM is an environment module with its own bundled venv, entirely distinct
    from the harness's project venv. This resolves the SERVER's own
    interpreter so `env_fingerprint.probe_from_interpreter` can ask that
    process what it is running, using the exact same matching semantics as
    `scan_vllm_argv` (see `_find_server`) so the two can never disagree about
    which process is ours.

    Prefers argv[0] over `/proc/<pid>/exe`: venvs typically
    create `bin/python` as a SYMLINK to the base interpreter, and
    `os.readlink` follows it all the way through to that base install --
    which has neither vllm nor torch, since those live only in the venv's own
    site-packages. Python's own venv detection keys off the invocation path
    (`argv[0]`/`sys.executable`), not the resolved target, so argv[0] is the
    one that actually activates the right site-packages when re-invoked.
    argv[0] is the interpreter in every case this module's matcher produces
    except the one where the vllm-named element is argv[0] itself (a direct
    exec with no captured interpreter, `_is_vllm_server`'s "console script at
    argv[0]" branch) -- there, and only there, `/proc/<pid>/exe` is the
    fallback, and is likely correct precisely because that shape implies a
    real compiled binary rather than a venv-relative script.

    Same three-outcome shape as `scan_vllm_argv`: raises on an unreadable
    /proc tree, returns None on no-match/ambiguous (logged), returns the
    resolved path on exactly one match. A readlink failure on an otherwise
    successfully-matched pid (the process exited between the argv read and
    this call) also returns None, since "not observed" is the correct
    description of a process that is no longer there to describe.

    `port`, when given, disambiguates sibling array
    indices serving the identical model on the same node; see
    `_argv_has_port`.
    """
    match = _find_server(model_id, proc_root=proc_root, log=log, port=port)
    if isinstance(match, ArgvNotObserved):
        return None
    if match.argv and not os.path.basename(match.argv[0]).startswith("vllm"):
        return match.argv[0]
    try:
        return os.readlink(os.path.join(proc_root, match.pid, "exe"))
    except OSError:
        return None


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def _is_vllm_server(argv: tuple[str, ...]) -> bool:
    """True for a `vllm serve ...` process, however it was invoked.

    Matches the console-script form (`vllm serve MODEL`), the kernel-REWRITTEN
    shebang form (see below), and both module forms (`python -m vllm serve
    MODEL`, `python -m vllm.entrypoints.openai.api_server`). Excludes any shell
    whose command line merely mentions vllm. Excluding the harness process
    itself is `scan_vllm_argv`'s job, by pid, rather than this function's:
    relying on the harness happening to be argv[0]=python would be a guarantee
    this function does not actually implement.

    Checking the console-script form only at argv[0] is wrong for a
    shebang-invoked script. When the shell resolves `vllm` via path to a
    script starting `#!/path/to/python`, the kernel rewrites the exec so
    `/proc/<pid>/cmdline` shows the interpreter as argv[0] and the script
    path as argv[1] -- `['.../venv/bin/python3.13', '.../bin/vllm', 'serve',
    model, ...]` on the cluster's vllm module -- not `['vllm', 'serve', model,
    ...]`. Both positions are checked so either invocation style matches.
    """
    if not argv:
        return False
    # Console script, checked at argv[0] (direct exec, e.g. a shell alias or a
    # container ENTRYPOINT) and argv[1] (shebang rewrite, see above). `serve`
    # must immediately follow the vllm-named element, not merely be present
    # (`["vllm", "--version", "serve"]` must not match).
    for idx in (0, 1):
        if (
            len(argv) > idx + 1
            and os.path.basename(argv[idx]).startswith("vllm")
            and argv[idx + 1] == "serve"
        ):
            return True
    # Module forms: both `python -m vllm serve MODEL` and the api_server
    # entrypoint.
    for i, part in enumerate(argv):
        if part == "-m" and i + 1 < len(argv):
            target = argv[i + 1]
            if target == "vllm.entrypoints.openai.api_server":
                return True
            if target == "vllm" and argv[i + 2 : i + 3] == ("serve",):
                return True
    return False
