"""Thin serving-control client: swap which model the serving box
serves, from code, and read back what actually ended up running.

This module holds zero vLLM flags. Every vLLM flag (dtype, gpu-memory-
utilization, the image digest, ...) lives in the the serving setup's
on-box launch script; this module only drives the fixed contract that repo
exposes and verifies the result:

  - on-box script: /data/hf-cache/launch/launch-vllm-roster.sh <model-id>
  - user-scope templated unit vllm-roster@.service at
    ~/.config/systemd/user/, driven by `systemctl --user` (system-scope is
    not available: sudo/systemctl both require interactive auth on the box;
    user-scope systemd + linger works unprivileged)
  - instance name = `systemd-escape <model-id>`, invoked on THE BOX, never
    reimplemented locally (a naive '/'->'-' scheme addresses the wrong unit;
    see `escape_instance`)

No silent fallbacks: `swap_to` fails loudly
rather than proceeding against whatever happens to be listening. It never
returns while the endpoint serves anything other than the exact id
requested; every command's returncode is checked immediately rather than
relying on the poll to notice a failure later. `resolve` reads the serving
configuration back from the box (`docker inspect`, `/v1/models`) rather than
assembling it from the launch script or a config file, which would be a
record that does not correspond to the work actually done.
"""

from __future__ import annotations

import http.client
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

# Internal timeouts for the auxiliary commands/requests this module issues.
# Not part of the public API (ServingController.__init__ takes exactly
# ssh_host/base_url/unit_template/runner); swap_to/current do expose
# their own timeout/poll knobs, which is the caller-facing control surface.
_ESCAPE_TIMEOUT_S = 15.0
_SYSTEMCTL_TIMEOUT_S = 60.0
_DOCKER_TIMEOUT_S = 30.0
_HTTP_TIMEOUT_S = 30.0

# systemd's exit code for "unit not loaded" on `systemctl --user stop`
# (LSB "program not running"). On the live box a never-loaded instance of
# vllm-roster@ returns this, not 0 - see `ServingController.stop`'s docstring for why it is tolerated.
_SYSTEMCTL_NOT_LOADED_RC = 5

# The serving-side docker label carrying model identity, set at `docker
# run` time by the launch script (out of this module's scope; a fixed
# serving-side contract). Containers are
# named cosmetically (a lossy transform), so the label - which round-trips
# a literal '/' with no escaping - is the only lossless way to find the
# container actually serving `model_id`; never the container name.
_ROSTER_MODEL_LABEL = "roster.model"


def parse_nvidia_smi_gpu_query(stdout: str) -> tuple[str | None, int | None]:
    """Parse one line of `nvidia-smi --query-gpu=name,memory.total
    --format=csv,noheader` output (e.g. "NVIDIA H200, 143771 MiB") into
    (name, memory_total_mib).

    Returns (None, None) on anything that is not exactly two comma-
    separated fields with a numeric MiB memory value - an unparseable
    result is a provenance gap, never a guessed one (no silent
    fallbacks). Shared by every provider that obtains nvidia-smi CSV text
    via its own transport (SSH for ServingController, local subprocess for
    InJobServing) - this function only parses, it never executes anything.
    """
    text = stdout.strip()
    if not text:
        return None, None
    first_line = text.splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) != 2:
        return None, None
    name, mem_raw = parts
    mem_digits = mem_raw.replace("MiB", "").strip()
    if not name or not mem_digits.isdigit():
        return None, None
    return name, int(mem_digits)


class ServingError(RuntimeError):
    """Base class for this module's failures."""


class ServingControlError(ServingError):
    """A swap did not end with the endpoint serving the expected model."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Seam so tests drive the real entrypoint without touching a live box."""

    def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult: ...


@dataclass(frozen=True)
class SshCommandRunner:
    """Default runner: executes argv on `host` over ssh.

    `argv` is shell-quoted (`shlex.quote` per element) and joined into one
    remote command line, since ssh concatenates its trailing arguments with
    spaces and hands them to the remote user's shell as a single string.
    """

    host: str

    def run(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        remote_cmd = " ".join(shlex.quote(str(a)) for a in argv)
        full_argv = ["ssh", self.host, remote_cmd]
        try:
            proc = subprocess.run(
                full_argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                # The exit code is returned in the CommandResult.
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # CPython returns the partial output as bytes here even under text=True.
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", "replace")
            return CommandResult(
                returncode=124,
                stdout=partial,
                stderr=f"ssh {self.host} {remote_cmd!r} timed out after {timeout}s: {exc}",
            )
        return CommandResult(
            returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
        )


@dataclass(frozen=True)
class ServedModel:
    """What GET <base_url>/models reports right now.

    `model_id` is the first entry of `served_ids` (the box serves exactly
    one model in the measured deployment); `served_ids` is the full list so
    exact-id membership checks never rely on picking the "right" single
    entry out of several.
    """

    model_id: str
    max_model_len: int | None
    served_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedServing:
    """The serving configuration as actually applied, read back from the box.

    Never assembled from a launch script or a config file: `image_ref` and
    `vllm_args` come from `docker inspect` on a container VERIFIED running
    (`.State.Running` - a stale/dead container found only by label is never
    reported as live provenance); `max_model_len` comes from `/v1/models`;
    `unit` is verified with `systemctl --user is-active` and is `None`, not an unverified
    forward-assembled name, whenever that check does not confirm it. `container`/
    `image_ref`/`vllm_args` may be `None` (with `vllm_args_unobserved_reason`) when the container could not be
    unambiguously identified, or was found but not running (a documented gap,
    not a guessed value - see `ServingController._find_container`/`_docker_inspect`).

    For the pbs-injob provider (job-scheduler backends, e.g. OpenPBS):
    `serving_env`/`serving_env_hash` follow the class-level rule above
    without exception -- they are
    never assembled from config here either. `InJobServing.resolve()` calls
    `env_fingerprint.collect_serving_env()` to observe the real environment
    and stamps that; a config's declared expectation
    (`InJobServing.expected_serving_env`, not a field of this class) is only a
    tripwire `resolve()` cross-checks the observation against, aborting on
    disagreement before this dataclass is even constructed -- see
    `lmopt.serving.injob` and `lmopt.serving.env_fingerprint`. `image_ref:
    None` on this provider means "no container exists on this backend" (an
    observed fact), not "could not be unambiguously identified" (the meaning
    above); `serving_env`/`serving_env_hash` carry the comparability role
    `image_ref` plays for a container-backed provider.
    """

    model_id: str
    base_url: str
    image_ref: str | None  # With digest, from `docker inspect`
    vllm_args: (
        tuple[str, ...] | None
    )  # the real launch argv; None when it was not observed
    container: str | None
    max_model_len: int | None
    chat_template_hash: str | None
    unit: str | None  # verified `systemctl --user is-active`; None if not confirmed
    # PBS in-job lane. A job-scheduler backend launches vLLM inside the
    # allocation with no container, so `image_ref` is None there as an
    # Observed fact and these carry the comparability instead: a structured
    # record of the serving stack, plus a hash of it. Defaulted so the other
    # providers (spark-systemd, preserved) simply report two nulls -- "not
    # applicable to this provider", the same documented-gap convention
    # `container`/`unit` use.
    # See lmopt.serving.env_fingerprint for what goes in and why.
    serving_env: dict[str, Any] | None = None
    serving_env_hash: str | None = None
    # GPU-card provenance: not part of serving_env / serving_env_hash
    # deliberately - folding it into that hashed dict would change
    # serving_env_hash's meaning for every arm already collected. Defaulted
    # to None so a provider that does not probe it reports None. Each provider sets this explicitly on its own
    # resolve() path (never left as a dropped key) - see
    # ServingController.resolve(), InJobServing.resolve(),
    # PreservedServing.resolve().
    gpu_name: str | None = None
    gpu_memory_total_mib: int | None = None
    # Why `vllm_args` is None. Present exactly when `vllm_args` is None, absent otherwise.
    vllm_args_unobserved_reason: str | None = None

    def __post_init__(self) -> None:
        """
        refuse a launch argv that is neither an observation nor a reasoned absence
        :raises ValueError: when `vllm_args` is None without a reason, when it is observed and
            still carries a reason, or when it is an empty tuple (a launch always has at least
            `vllm serve`, so an empty argv says nothing was observed and must be None)
        """
        if self.vllm_args is None:
            if not self.vllm_args_unobserved_reason:
                raise ValueError(
                    "ResolvedServing.vllm_args is None (not observed) and no "
                    "vllm_args_unobserved_reason says why"
                )
        else:
            if not self.vllm_args:
                raise ValueError(
                    "ResolvedServing.vllm_args is an empty tuple; an empty argv is not an "
                    "observation, record None with a vllm_args_unobserved_reason instead"
                )
            if self.vllm_args_unobserved_reason is not None:
                raise ValueError(
                    "ResolvedServing.vllm_args was observed yet carries "
                    f"vllm_args_unobserved_reason={self.vllm_args_unobserved_reason!r}"
                )

    def to_run_meta(self) -> dict[str, Any]:
        """
        the serving block as `run_meta.json` records it
        :returns: the block; `vllm_args` is a list when observed, or null with
            `vllm_args_unobserved_reason` (a key present only then) saying why (dict)
        """
        meta = {
            "model_id": self.model_id,
            "base_url": self.base_url,
            "image_ref": self.image_ref,
            "vllm_args": list(self.vllm_args) if self.vllm_args is not None else None,
            "container": self.container,
            "max_model_len": self.max_model_len,
            "chat_template_hash": self.chat_template_hash,
            "unit": self.unit,
            # Defensive copy: the other values are
            # immutable, so without this a caller mutating the returned block
            # would reach back into a frozen record.
            "serving_env": (
                dict(self.serving_env) if self.serving_env is not None else None
            ),
            "serving_env_hash": self.serving_env_hash,
            "gpu_name": self.gpu_name,
            "gpu_memory_total_mib": self.gpu_memory_total_mib,
        }
        if self.vllm_args is None:
            meta["vllm_args_unobserved_reason"] = self.vllm_args_unobserved_reason
        return meta


class ServingController:
    def __init__(
        self,
        *,
        ssh_host: str,
        base_url: str,
        unit_template: str = "vllm-roster@",
        runner: CommandRunner | None = None,
    ) -> None:
        self.ssh_host = ssh_host
        self.base_url = base_url
        self.unit_template = unit_template
        self._runner: CommandRunner = (
            runner if runner is not None else SshCommandRunner(host=ssh_host)
        )

    # -- systemd-escape seam -------------------------------------------------

    def escape_instance(self, model_id: str) -> str:
        """Escaped systemd instance name for `model_id`, computed on THE BOX.

        Shells out to `systemd-escape` via the runner; never reimplemented
        locally. A naive '/'->'-' scheme silently addresses the wrong unit:
        `systemd-escape 'Qwen/Qwen2.5-14B-Instruct'` really produces
        `Qwen-Qwen2.5\\x2d14B\\x2dInstruct`, not `Qwen-Qwen2.5-14B-Instruct`
        (round-tripped for three model ids).
        """
        result = self._runner.run(
            ["systemd-escape", model_id], timeout=_ESCAPE_TIMEOUT_S
        )
        if result.returncode != 0:
            raise ServingControlError(
                f"systemd-escape {model_id!r} failed (rc={result.returncode}): {result.stderr}"
            )
        return result.stdout.strip()

    def unit_for(self, model_id: str) -> str:
        """The templated unit instance name for `model_id`."""
        return f"{self.unit_template}{self.escape_instance(model_id)}.service"

    # -- reading the endpoint -------------------------------------------------

    def current(self) -> ServedModel | None:
        """What GET <base_url>/models reports right now.

        Returns `None` only for an unreachable endpoint (connection refused,
        DNS failure, connect/read timeout). An endpoint that answers -
        including with an HTTP error status, or a 200 with an
        unparseable/unexpected body - raises `ServingControlError`; it is
        never silently treated as "nothing served." Mirrors
        `scripts/run_experiment.py`'s `preflight`, whose docstring
        draws the identical line (an unparseable body is a `PreflightError`,
        never a skip); membership there is likewise exact-id, matched here
        via `model_id in served.served_ids` tuple equality (never substring).
        """
        url = self.base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise ServingControlError(
                f"{url} answered with HTTP {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError:
            return None
        except OSError:
            return None
        except http.client.HTTPException as exc:
            raise ServingControlError(
                f"{url} body was cut off or malformed ({exc!r}); not ready, not absent"
            ) from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
            entries = payload["data"]
            served_ids = tuple(str(entry["id"]) for entry in entries)
            max_model_len: int | None = None
            if entries:
                mml = entries[0].get("max_model_len")
                max_model_len = int(mml) if mml is not None else None
            model_id = served_ids[0] if served_ids else ""
        except Exception as exc:
            raise ServingControlError(
                f"unexpected body shape from {url}: {exc!r}"
            ) from exc
        return ServedModel(
            model_id=model_id, max_model_len=max_model_len, served_ids=served_ids
        )

    # -- systemctl actions -------------------------------------------------

    def stop(self, model_id: str) -> None:
        """Stop `model_id`'s templated unit via `systemctl --user stop`.

        systemctl only guarantees rc=0 as a no-op for a unit that is loaded
        but inactive. A unit that was never loaded at all - the case that
        actually occurs here, since the box's current occupant is typically a
        bare `docker run` container started directly by the serving setup
        launch script, owned by no systemd unit - makes `systemctl --user
        stop` exit 5 ("unit not loaded"), as observed on the live box:
        `systemctl --user stop "vllm-roster@...service"` on an unloaded
        instance returns rc=5 with "Unit ... not loaded." on stderr, and
        this is indistinguishable from a genuinely unknown/misspelled unit
        name. "Not loaded" is nothing to stop, not a failure, so rc=5 is
        tolerated here (never any other non-zero code - those still raise).

        This is safe, not merely convenient: this method claims only "no
        systemd unit for `model_id` is left running," which rc=5 already
        guarantees (there was never a unit-managed process to leave
        running). It does not claim to have evicted a non-unit-managed
        occupant (e.g. a bare `docker run` container) - that eviction is
        the serving launch script's own port-occupant handling
        (the serving setup, out of this module's scope), which runs
        regardless of whether this method's systemctl call did anything.

        Raises `ServingControlError` on any other non-zero returncode;
        never swallowed to let the poll in `swap_to` notice later.
        """
        unit = self.unit_for(model_id)
        result = self._runner.run(
            ["systemctl", "--user", "stop", unit], timeout=_SYSTEMCTL_TIMEOUT_S
        )
        if result.returncode != 0 and result.returncode != _SYSTEMCTL_NOT_LOADED_RC:
            raise ServingControlError(
                f"systemctl --user stop {unit} failed (rc={result.returncode}): {result.stderr}"
            )

    def swap_to(
        self, model_id: str, *, timeout_s: float = 900.0, poll_s: float = 5.0
    ) -> ResolvedServing:
        """Stop whatever is being served, start `model_id`'s unit, and verify.

        No silent fallbacks: fails loudly rather than proceeding
        against whatever happens to be listening:

        1. Stop whatever `current()` reports is served (skipped if the
           endpoint is currently unreachable - nothing knowably to stop),
           then start the target unit via `systemctl --user`.
        2. A non-zero returncode from either command raises immediately with
           stderr attached; it is never swallowed to let the poll notice.
        3. Poll `/v1/models` until the requested id appears. Membership is
           exact-id: `model_id in served.served_ids` is tuple-element
           equality, never substring, so a superstring variant (e.g.
           'pythia-12b-deduped' when 'pythia-12b' was asked for) fails -
           matching `scripts/run_experiment.py`'s `preflight` rule.
        4. If the poll times out, or the endpoint keeps serving a different
           id, raises `ServingControlError` naming the requested id, what
           was last actually served (or that the endpoint was unreachable),
           the measured elapsed time and the configured budget, and the
           last transient polling error seen (if any). This is the central
           case: it never returns while the wrong thing (or nothing) is
           being served.
        5. A single transient error from `current()` during the poll (an
           HTTP 5xx while the model is still loading, an unparseable body
           mid-restart, ...) is treated as "not ready yet" and retried
           until the deadline, not surfaced immediately - an unattended
           cycle where each arm runs for hours should not die on one
           hiccup. It is never silently dropped: it becomes `last_error`
           in the eventual timeout message if the poll never recovers.
        """
        current = self.current()
        if current is not None:
            self.stop(current.model_id)

        unit = self.unit_for(model_id)
        start = self._runner.run(
            ["systemctl", "--user", "start", unit], timeout=_SYSTEMCTL_TIMEOUT_S
        )
        if start.returncode != 0:
            raise ServingControlError(
                f"systemctl --user start {unit} failed (rc={start.returncode}): {start.stderr}"
            )

        poll_start = time.monotonic()
        deadline = poll_start + timeout_s
        last_served: ServedModel | None = None
        last_error: ServingControlError | None = None
        while True:
            try:
                served = self.current()
            except ServingControlError as exc:
                # A single transient error from current() during the poll (an HTTP 5xx while the
                # model is still loading, say) is "not ready yet," not an
                # abort - an unattended cycle where each arm is hours long
                # should retry it, not die on one hiccup. The loud failure
                # still happens at the deadline below, and carries this as
                # `last_error` - never silently dropped.
                last_error = exc
                served = None
            if served is not None:
                last_served = served
                if model_id in served.served_ids:
                    return self.resolve(model_id)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_s, remaining))

        elapsed = time.monotonic() - poll_start
        # "nothing (endpoint unreachable)" is FALSE for a poll that never got
        # past a transient error - the endpoint was answering, just with
        # errors. Distinguish never-reachable (last_served is None and
        # last_error is None) from reachable-but-erroring (last_served is
        # None and last_error is not None); the error is appended via
        # `error_note`, but the leading clause must not itself claim
        # something false.
        served_desc: tuple[str, ...] | str
        if last_served is not None:
            served_desc = last_served.served_ids
        elif last_error is not None:
            served_desc = (
                "nothing confirmed (endpoint reachable but answered with errors)"
            )
        else:
            served_desc = "nothing (endpoint unreachable)"
        error_note = (
            f"; last error while polling: {last_error!r}"
            if last_error is not None
            else ""
        )
        raise ServingControlError(
            f"swap_to({model_id!r}) at {self.base_url} timed out after "
            f"{elapsed:.2f}s (budget {timeout_s:.2f}s): endpoint serves "
            f"{served_desc!r} instead of {model_id!r}{error_note}"
        )

    # -- reading configuration back from the box -------------------------------------------------

    def resolve(self, model_id: str) -> ResolvedServing:
        """Read the serving configuration as actually applied back from the box.

        Never assembled from a launch script or a config file: `image_ref`
        (with digest) and `vllm_args` (the real container `Cmd` vector) come
        from `docker inspect` on a container VERIFIED running (see
        `_docker_inspect`); `max_model_len` comes from `/v1/models`; `unit`
        is verified with `systemctl --user is-active` and set to `None` if
        it is not, rather than merely assembled from the template and the
        escape, because a swap can succeed via a non-unit-managed occupant
        that already happened to serve the target id - the dataclass's own
        docstring says "AS actually applied... never assembled," and an
        un-checked forward-built unit name would violate that for this
        field alone.
        `gpu_name` and `gpu_memory_total_mib` are observed the same way:
        `nvidia-smi` run through `self._runner`, so the card
        recorded is the serving box's, not the orchestrator's, and both stay
        `None` when the query fails - see the inline comment at that call for
        why the transport, not just the command, is the load-bearing part.
        Raises `ServingControlError` if `model_id` is not currently served
        (exact-id, same rule as `current`/`swap_to`), and propagates
        `ChatTemplateUnavailable` if the chat-template fingerprint cannot be
        resolved - this is deliberately not swallowed to a `None` field (no
        silent fallbacks): an otherwise-healthy endpoint that
        cannot answer `/tokenize` is itself a finding, not a null field.
        """
        served = self.current()
        if served is None or model_id not in served.served_ids:
            served_desc = (
                served.served_ids
                if served is not None
                else "nothing (endpoint unreachable)"
            )
            raise ServingControlError(
                f"resolve({model_id!r}): {self.base_url} does not currently serve "
                f"it (serves {served_desc!r})"
            )
        container = self._find_container(model_id)
        image_ref: str | None = None
        vllm_args: tuple[str, ...] | None = None
        unobserved = "no running container labelled for this model was found"
        if container is not None:
            inspected = self._docker_inspect(container)
            if inspected is None:
                # Found via the label, but not running (a stale/dead
                # container that lingered without --rm firing). Recording
                # its Image/Cmd as live provenance would be a FALSE record
                # - worse than the None a legitimately-absent container
                # already returns - so this degrades exactly like "not
                # found," never like "found and trusted."
                container = None
                unobserved = "the labelled container was found but is not running"
            else:
                image_ref, inspected_args = inspected
                if inspected_args:
                    vllm_args = inspected_args
                else:
                    unobserved = (
                        "docker inspect returned an empty Cmd for the container"
                    )

        # `container is not None` is a required precondition, checked before
        # `_unit_is_active`, not an independent fact checked alongside it.
        # `_unit_is_active` proves A unit by that name is active - not that
        # it owns the endpoint currently answering. Reachable on the very
        # first arm: under `Type=simple` the serving unit's
        # `systemctl --user start` returns in ~27ms, well before
        # eviction, `docker run`, or any model load; the box's actual
        # first-arm occupant is an unlabelled, non-roster container the
        # serving setup refuses to evict by default. So `is-active` can
        # legitimately say "active" for a unit that has not (yet, or
        # ever) served the model the endpoint is actually answering for.
        # Requiring a confirmed, running, label-matched container first is
        # a cheap ownership proxy: `unit` is never trusted in isolation
        # from `container`.
        unit_name = self.unit_for(model_id)
        unit = (
            unit_name
            if (container is not None and self._unit_is_active(unit_name))
            else None
        )

        # Local import: chat_template.py imports ServingError from this
        # module at its own top level, so importing chat_template back at control.py's
        # module top would be circular. Deferring to call time is safe -
        # by the time any ServingController method runs, this module has
        # finished initializing.
        from lmopt.serving.chat_template import resolve_chat_template_hash

        chat_template_hash = resolve_chat_template_hash(self.base_url, model_id)

        # GPU-card provenance: same SSH transport
        # this method already uses for docker inspect / systemctl - a probe
        # placed anywhere other than through self._runner would silently
        # query the ORCHESTRATOR's own GPU, not the serving box's, since
        # this method runs on the orchestrator for the ssh_host lane.
        gpu_name: str | None = None
        gpu_memory_total_mib: int | None = None
        smi_result = self._runner.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            timeout=_HTTP_TIMEOUT_S,
        )
        if smi_result.returncode == 0:
            gpu_name, gpu_memory_total_mib = parse_nvidia_smi_gpu_query(
                smi_result.stdout
            )

        return ResolvedServing(
            model_id=model_id,
            base_url=self.base_url,
            image_ref=image_ref,
            vllm_args=vllm_args,
            container=container,
            max_model_len=served.max_model_len,
            chat_template_hash=chat_template_hash,
            unit=unit,
            gpu_name=gpu_name,
            gpu_memory_total_mib=gpu_memory_total_mib,
            vllm_args_unobserved_reason=unobserved if vllm_args is None else None,
        )

    def _unit_is_active(self, unit: str) -> bool:
        """Verify `unit` with `systemctl --user is-active`, never assume it.

        Assembling `ResolvedServing.unit` forward from the template +
        `escape_instance`, with nothing checking the unit exists or is
        active, would contradict the dataclass's own "AS actually
        applied... never assembled" docstring.

        systemd exits 0 for `reloading`, not just `active`, so the
        returncode alone is not enough. Both `returncode == 0` and
        `stdout.strip() == "active"` (exactly, after stripping trailing
        whitespace/newline) are checked - `reloading` and any other
        rc=0-but-not-"active" output are treated as "not verified active,"
        matching what this method's name and callers assume it means.

        Any non-"active" outcome - inactive/failed/unknown/reloading, or a
        query-level failure such as an ssh hiccup - is treated uniformly
        as "not verified active"; `resolve()` reports `unit=None` rather
        than an unverified name. Deliberately not distinguished further (a
        query failure here vs a genuinely non-active unit): unlike
        `_find_container`'s `docker ps`, a failure here cannot
        silently manufacture a false-positive provenance record - the
        worst outcome is `unit=None`, already this method's honest "not
        verified" answer. NOTE: this method proves A unit
        by this name is active; it does not prove that unit owns whatever
        the endpoint is currently serving - `resolve()` additionally
        requires `container is not None` (a label-matched, running
        container was actually found) before trusting this method's
        answer at all. See the comment in `resolve()`.
        """
        result = self._runner.run(
            ["systemctl", "--user", "is-active", unit], timeout=_SYSTEMCTL_TIMEOUT_S
        )
        return result.returncode == 0 and result.stdout.strip() == "active"

    def _find_container(self, model_id: str) -> str | None:
        """Container-id lookup for `docker inspect`, via the serving setup label.

        The serving launch script names containers cosmetically (a lossy
        transform of the model id) but tags the container it starts with a
        docker label, `_ROSTER_MODEL_LABEL=<model_id>`, set at `docker run`
        time precisely so nothing here has to reconstruct a lossy transform -
        the label value round-trips a literal '/' with no escaping, unlike
        a container name. Matching on the escaped systemd instance name as a
        container name would match nothing against a real roster container
        (names and the systemd instance name are unrelated strings), so
        `image_ref`/`vllm_args` would come back `None`/`()` on every call in
        production while `resolve()` still looked healthy. Matching on the
        label is the serving-side contract, not an assumption.

        A non-zero returncode from the `docker ps` query itself (e.g. rc=1
        "Cannot connect to the Docker daemon", or rc=124 - literally what
        `SshCommandRunner.run` returns on its own ssh-level timeout) is not
        "the query succeeded and found nothing": treating the two alike
        would silently swallow a real failure, and `resolve()` would return
        a successful, provenance-free `ResolvedServing` instead of raising.
        A query failure raises `ServingControlError` with stderr attached.
        Only a successful query that legitimately
        finds zero or more than one labelled container returns `None` -
        that remains a documented gap, not a guessed value.
        """
        result = self._runner.run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label={_ROSTER_MODEL_LABEL}={model_id}",
            ],
            timeout=_DOCKER_TIMEOUT_S,
        )
        if result.returncode != 0:
            raise ServingControlError(
                f"docker ps --filter label={_ROSTER_MODEL_LABEL}={model_id} failed "
                f"(rc={result.returncode}): {result.stderr}"
            )
        matches = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(matches) == 1:
            return matches[0]
        return None

    def _docker_inspect(
        self, container: str
    ) -> tuple[str | None, tuple[str, ...]] | None:
        """`docker inspect` the label-matched container; `None` if it is not running.

        The serving setup removes all same-label containers before every launch
        (`--rm` is set, `ExecStop` delegates to the same cleanup), so a 2-match ambiguity in `_find_container` is
        unreachable in practice.
        The real residual is a single stale match - a container that died
        without `--rm` firing (a dockerd restart) or lingers in `created`.
        Recording a dead container's `Image`/`Cmd` as "AS actually APPLIED"
        provenance would be a FALSE record, which is worse than the `None`
        a legitimately-absent container already returns - so `.State.Running`
        is checked in the same inspect call (no extra round trip), and
        `None` is returned (not raised: a stale container is not a query
        failure) whenever it is not exactly `"true"`. `resolve()` then
        treats that identically to "no container found."
        """
        result = self._runner.run(
            [
                "docker",
                "inspect",
                container,
                "--format",
                "{{.Config.Image}}\t{{json .Config.Cmd}}\t{{.State.Running}}",
            ],
            timeout=_DOCKER_TIMEOUT_S,
        )
        if result.returncode != 0:
            raise ServingControlError(
                f"docker inspect {container} failed (rc={result.returncode}): {result.stderr}"
            )
        line = result.stdout.strip()
        parts = line.split("\t")
        if len(parts) != 3:
            raise ServingControlError(
                f"docker inspect {container}: unexpected output {result.stdout!r}"
            )
        image_ref, cmd_json, running = parts
        if running.strip().lower() != "true":
            return None
        try:
            cmd = tuple(json.loads(cmd_json))
        except Exception as exc:
            raise ServingControlError(
                f"docker inspect {container}: could not parse Cmd JSON {cmd_json!r}: {exc}"
            ) from exc
        return (image_ref or None), cmd
