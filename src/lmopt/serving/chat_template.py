"""Chat-template drift fingerprint, resolved from the served endpoint.

Reuse note: the tokenize/detokenize round trip and the `/v1`-stripping helper
below are a small, deliberate REIMPLEMENTATION of the approach
`scripts/exact_policy_pass.py` already validates (its
`render_chat_prefix_ids` / `_post_json` / `_root_url`), not an import
from it. `src/` importing from `scripts/` would be a new and awkward
dependency direction (scripts/ is not a package other src/ code depends on
anywhere in this repo), so the ~20-line pattern is duplicated here instead,
attributed rather than silently re-derived. If the two ever drift, that is a
consolidation task, not a defect in either.
"""

from __future__ import annotations

from typing import Any

import hashlib
import json
import urllib.error
import urllib.request

from lmopt.serving.control import ServingError


class ChatTemplateUnavailable(ServingError):
    """The served model's chat template could not be resolved from the endpoint.

    Raised rather than silently falling back to a hardcoded/naive template or
    returning a placeholder - the exact defect this exists to avoid (raw vs
    templated gave 0.9969 vs 0.6792 on the same node in the pass this reuses
    the mechanism from; a reported figure, not re-measured by this module).
    On any failure this module raises; it never
    returns None or a placeholder hash.
    """


# A fixed probe conversation with system + user + assistant turns, so more of
# the template is exercised than a single user turn would (system-prompt
# wrapping, turn separators, and the assistant-echo formatting all differ
# across templates). Pinned as a module constant so the resulting hash is
# stable release over release.
#
# Changing this tuple invalidates every previously recorded
# chat_template_hash value - the hash is a fingerprint of "this probe,
# rendered by the served template," not of the template in isolation.
# The probe MUST END WITH A user turn. vLLM renders
# it with add_generation_prompt=True, and Mistral's native renderer (reached
# via --tokenizer-mode mistral, which Mistral-NeMo's snapshot requires) raises
# on an assistant-final conversation:
#   renderers/mistral.py:47 safe_apply_chat_template ->
#   ValueError: Cannot set `add_generation_prompt` to True when the last
#   message is from the assistant. Consider using `continue_final_message`.
# With an assistant-final probe,
# a swap to `mistralai/Mistral-Nemo-Instruct-2407` exits
# non-zero against a healthy server (the model itself serves fine), so under
# `serving.auto_swap: true` the whole Mistral arm would abort. Qwen's
# and OLMo-2's HF-jinja templates tolerate assistant-final.
# The assistant turn is retained: the reason this probe has one (exercising
# assistant-echo formatting) still holds, and a trailing user turn satisfies
# the constraint without losing that coverage.
CHAT_TEMPLATE_PROBE: tuple[dict[str, str], ...] = (
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Reply with exactly one word: OK."},
    {"role": "assistant", "content": "OK."},
    {"role": "user", "content": "Reply with exactly one word: DONE."},
)


def _root_url(base_url: str) -> str:
    """Strip a trailing '/v1' so /tokenize and /detokenize (not under /v1,
    unlike /v1/completions) resolve correctly from a base_url that carries
    the '/v1' suffix (matches scripts/exact_policy_pass.py's `_root_url`)."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[: -len("/v1")]
    return b


def _post_json(root_url: str, path: str, payload: dict, timeout: float) -> dict:
    url = root_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_template_pin_key(model_id: str, chat_template_kwargs: dict | None) -> str:
    """Pin-key for expected_chat_template_hash registry lookups.

    Returns model_id unchanged when kwargs is None (the bare-template case),
    else appends a deterministic JSON-serialized query string. Two kwargs dicts
    with different key order produce the same key (sort_keys=True), so the pin
    lookup is stable regardless of dict insertion order.

    Example:
        chat_template_pin_key("Qwen/Qwen3-32B", None)
        -> "Qwen/Qwen3-32B"

        chat_template_pin_key("Qwen/Qwen3-32B", {"enable_thinking": False})
        -> 'Qwen/Qwen3-32B?chat_template_kwargs={"enable_thinking":false}'
    """
    if chat_template_kwargs is None:
        return model_id
    serialized = json.dumps(chat_template_kwargs, sort_keys=True, separators=(",", ":"))
    return f"{model_id}?chat_template_kwargs={serialized}"


def resolve_chat_template_hash(
    base_url: str,
    model_id: str,
    *,
    timeout: float = 30.0,
    chat_template_kwargs: dict | None = None,
) -> str:
    """A fingerprint of the served template as applied to `CHAT_TEMPLATE_PROBE`,
    with this run's chat_template_kwargs when set.

    Resolved from the endpoint's own `/tokenize` (messages -> token ids,
    applying the server's loaded chat template including
    add_generation_prompt) then `/detokenize` (token ids -> text) round trip,
    never from `tokenizer_config.json` on disk, because vLLM's
    `--chat-template` override (which EleutherAI/pythia-12b requires) makes
    the on-disk file non-authoritative for what is actually served.

    When `chat_template_kwargs` is set, the /tokenize request carries those
    kwargs, so the hash fingerprints the template as applied with those
    rendering parameters (e.g., Qwen3's enable_thinking=False arm hashes
    differently from enable_thinking omitted). When None, the /tokenize
    payload carries no kwargs field at all, so every hash recorded without
    kwargs keeps its meaning.

    Limitation, stated rather than glossed: this detects drift, it does not
    identify a template. Two different templates could render this one fixed
    probe identically and hash the same; a hash match is not proof of
    template identity, only an absence of detected drift against this probe.
    A second, narrower limitation: the hash is
    over the DETOKENIZED text, not the raw rendering itself, so a template
    difference that the tokenize/detokenize round trip normalises away
    (e.g. whitespace or Unicode-form differences collapsed by the
    tokenizer) is invisible to this fingerprint too - it detects drift
    visible to the tokenizer, not drift visible to a byte-level diff.

    Raises ChatTemplateUnavailable on any failure (unreachable endpoint, a
    non-answering endpoint, an empty/malformed /tokenize or /detokenize
    body) - never returns None or a placeholder.
    """
    root = _root_url(base_url)
    try:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": list(CHAT_TEMPLATE_PROBE),
        }
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = chat_template_kwargs
        tok = _post_json(root, "/tokenize", payload, timeout)
        token_ids = tok["tokens"]
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"/tokenize returned no tokens: {tok!r}")
        detok = _post_json(
            root, "/detokenize", {"model": model_id, "tokens": token_ids}, timeout
        )
        text = detok["prompt"]
        if not isinstance(text, str) or not text:
            raise ValueError(f"/detokenize returned no text: {detok!r}")
    except Exception as exc:
        raise ChatTemplateUnavailable(
            f"could not resolve the served chat template from {root} "
            f"for model {model_id!r}: {exc!r}"
        ) from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
