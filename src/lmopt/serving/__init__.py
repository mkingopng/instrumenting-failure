"""Serving control for the LM Optimality Tournament harness.

Every vLLM flag lives in the the serving setup; every experiment
decision lives here. This package holds zero vLLM flags - it only drives
the fixed on-box contract (a launch script + a user-scope systemd template)
and reads back what actually ended up running, never what a config file
says should be running.
"""

from lmopt.serving.chat_template import (
    CHAT_TEMPLATE_PROBE,
    ChatTemplateUnavailable,
    resolve_chat_template_hash,
)
from lmopt.serving.env_fingerprint import (
    EnvFingerprintError,
    collect_serving_env,
    fingerprint_hash,
)
from lmopt.serving.control import (
    CommandResult,
    CommandRunner,
    ResolvedServing,
    ServedModel,
    ServingControlError,
    ServingController,
    ServingError,
    SshCommandRunner,
)
from lmopt.serving.injob import InJobServing, InJobServingError

__all__ = [
    "CHAT_TEMPLATE_PROBE",
    "EnvFingerprintError",
    "InJobServing",
    "InJobServingError",
    "collect_serving_env",
    "fingerprint_hash",
    "ChatTemplateUnavailable",
    "resolve_chat_template_hash",
    "CommandResult",
    "CommandRunner",
    "ResolvedServing",
    "ServedModel",
    "ServingControlError",
    "ServingController",
    "ServingError",
    "SshCommandRunner",
]
