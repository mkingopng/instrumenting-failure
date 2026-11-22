"""Preserved serving provider.

A `preserved` provider performs no lifecycle actions (no ssh, no systemctl):
the endpoint already exists, launched outside the harness. It reads `/v1/models`
to confirm the expected model is served (exact-id tuple matching), and builds
a serving manifest from config-supplied fields (primarily `image_ref`) plus
endpoint-reported fields (`max_model_len`, `chat_template_hash`).

No silent fallbacks: a missing `image_ref` on a preserved arm is
an ABORT, not a null. A served-id mismatch is an ABORT with the exact-id rule
(tuple equality, never substring), matching preflight and ServingController's
own semantics.

Config shape:
    serving:
      provider: preserved
      image_ref: <digest>  # sha-pinned form, required
      # ... other keys passed through (ssh_host ignored)

This module's resolver is invoked by `scripts/run_experiment.py` when
`serving.provider == "preserved"`.
"""

from __future__ import annotations

from dataclasses import dataclass

from lmopt.serving.control import ResolvedServing, ServingController, ServingError


class PreservedServingError(ServingError):
    """Failures specific to the preserved provider."""


@dataclass(frozen=True)
class PreservedServing:
    """A preserved-provider resolver: no lifecycle, config + HTTP provenance.

    Unlike ServingController (which swaps/stops/starts via ssh + systemctl),
    this resolver assumes the endpoint is already up and performs zero
    lifecycle actions. It confirms the expected model is served, then builds
    a ResolvedServing from:
      - config-supplied fields: `image_ref` (required), `model_id`, `base_url`
      - endpoint-reported fields: `max_model_len`, `chat_template_hash`
        (via the same HTTP calls ServingController uses)
      - box-side fields set to their documented "not observed" values:
        `vllm_args=None` with a reason, `container=None`, `unit=None`, `gpu_name=None`,
        `gpu_memory_total_mib=None` - the preserved provider has no ssh
        access to inspect them.

    No silent fallbacks: missing `image_ref` is an abort (a
    preserved arm without it cannot record serving-stack provenance).
    """

    model_id: str
    base_url: str
    image_ref: str

    @classmethod
    def from_config(
        cls, model_id: str, base_url: str, serving_config: dict
    ) -> PreservedServing:
        """Factory from the config's `serving:` block.

        Raises `PreservedServingError` if `image_ref` is missing or empty
        (an arm without it is refused, not defaulted to None).
        """
        image_ref = serving_config.get("image_ref")
        if not image_ref:
            raise PreservedServingError(
                f"serving.provider=preserved for {model_id} requires a truthy "
                f"`image_ref` (got {image_ref!r}). A preserved arm cannot record "
                "serving-stack provenance without it."
            )
        if not isinstance(image_ref, str) or "sha256:" not in image_ref:
            raise PreservedServingError(
                f"serving.provider=preserved for {model_id} requires a sha-pinned "
                f"string `image_ref` containing 'sha256:' (got {image_ref!r}). A tag "
                "or a non-string does not identify one serving stack."
            )
        return cls(model_id=model_id, base_url=base_url, image_ref=image_ref)

    def resolve(self) -> ResolvedServing:
        """Build a ResolvedServing: confirm the model is served, HTTP provenance only.

        Exact-id check (tuple equality, never substring): mirrors
        ServingController.current() / .resolve() / preflight semantics.
        Raises `PreservedServingError` if the endpoint is unreachable or
        serves a different model.

        Box-side fields (`vllm_args`, `container`, `unit`, `gpu_name`,
        `gpu_memory_total_mib`) are set to their "not observed" values: the
        preserved provider has no ssh access.
        """
        # Reuse ServingController.current() for the /v1/models read; it
        # implements the exact-id membership check, the unreachable-vs-
        # parseable distinction, and the exception semantics this provider
        # needs. Constructed with empty ssh_host so no lifecycle methods
        # can be called (they'd fail on missing ssh_host; this is a
        # programming error, not a config error, so loud failure is correct).
        controller = ServingController(ssh_host="", base_url=self.base_url)
        served_model = controller.current()
        if served_model is None or self.model_id not in served_model.served_ids:
            served_desc = (
                served_model.served_ids
                if served_model is not None
                else "nothing (endpoint unreachable)"
            )
            raise PreservedServingError(
                f"preserved provider: {self.base_url} does not currently serve "
                f"{self.model_id!r} (serves {served_desc!r})"
            )

        # Chat-template hash: reuse the same resolver ServingController.resolve()
        # uses (local import, same reason as control.py:445).
        from lmopt.serving.chat_template import resolve_chat_template_hash

        chat_template_hash = resolve_chat_template_hash(self.base_url, self.model_id)

        # Box-side fields: "not observed" values per ResolvedServing's own
        # contract (control.py:127-149). GPU fields follow the identical
        # contract: no ssh/lifecycle access means no
        # observation is possible, so both are set explicitly here rather
        # than left to the dataclass default - a dropped key and a null
        # key are not the same fact (env_fingerprint.py:42-47).
        return ResolvedServing(
            model_id=self.model_id,
            base_url=self.base_url,
            image_ref=self.image_ref,
            vllm_args=None,  # not observed (no ssh/docker access)
            container=None,  # not observed
            max_model_len=served_model.max_model_len,
            chat_template_hash=chat_template_hash,
            unit=None,  # not observed (no systemctl access)
            gpu_name=None,  # not observed (no ssh access)
            gpu_memory_total_mib=None,  # not observed (no ssh access)
            vllm_args_unobserved_reason="preserved provider: no ssh or docker access to the box",
        )
