"""Weights-provenance verification mechanism.

This package is the home of weights-provenance verification, including the
declared-vs-observed revision comparison at launch.
"""

from __future__ import annotations


def parse_revision_flags(
    vllm_args: tuple[str, ...],
) -> tuple[str | None, str | None]:
    """Returns (revision, tokenizer_revision) parsed from a vLLM launch's own
    recorded argv.

    Mirrors `lmopt.serving.injob._argv_has_port`'s two-form handling exactly:
    `--revision value` (space-separated) and `--revision=value` (equals-
    joined) are both accepted, matched by exact token identity or an exact
    `--revision=` prefix, never by substring containment (a flag like
    `--revision-note` must not be mistaken for `--revision`).
    """
    revision: str | None = None
    tokenizer_revision: str | None = None

    if vllm_args is None:
        raise TypeError(
            "parse_revision_flags needs an observed argv; None means the launch was not "
            "observed, which is not a launch with no revision flags"
        )

    def _value_after(i: int, flag: str) -> str:
        """
        the token after a space-form revision flag, refused when there is none or it is a flag
        :param i: the index of the flag in `vllm_args` (int)
        :param flag: the flag, for the error (str)
        :returns: the flag's value (str)
        """
        if i + 1 >= len(vllm_args) or vllm_args[i + 1].startswith("--"):
            raise ValueError(
                f"{flag} has no value in the recorded launch argv {vllm_args!r}; the "
                f"launch's revision cannot be read, so fix the launch command"
            )
        return vllm_args[i + 1]

    for i, tok in enumerate(vllm_args):
        if tok == "--revision":
            revision = _value_after(i, tok)
        elif tok.startswith("--revision="):
            revision = tok[len("--revision=") :]
        elif tok == "--tokenizer-revision":
            tokenizer_revision = _value_after(i, tok)
        elif tok.startswith("--tokenizer-revision="):
            tokenizer_revision = tok[len("--tokenizer-revision=") :]

    return revision, tokenizer_revision
