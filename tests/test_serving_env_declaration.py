"""The declared serving environment in a job config is checked against the running server's."""

import glob

import pytest
import yaml

from lmopt.serving.injob import declaration_matches


@pytest.mark.parametrize(
    "component, declared, observed, expected",
    [
        ("vllm", "0.31.0", "0.31.0", True),
        ("vllm", "0.31.0", "0.31.1", False),
        ("vllm", "0.31", "0.31.0", False),
        ("torch", "2.13.0", "2.13.0+cu128", False),
        ("python", "3.13.7", "3.13.7", True),
        ("python", "3.13.7", "3.13.2", False),
        ("python", "3.13", "3.13.2", True),
        ("python", "3.13", "3.13.11", True),
        ("python", "3.13", "3.130.1", False),
        ("python", "3.13", "3.14.0", False),
        ("python", "3", "3.13.2", False),
        ("python", "3.13", None, False),
    ],
)
def test_declaration_matches(component, declared, observed, expected):
    """
    every component compares exactly, except that a major.minor Python declaration accepts any
    patch release of that minor version and nothing else
    """
    assert declaration_matches(component, declared, observed) is expected


def test_job_templates_declare_the_pinned_serving_stack():
    """
    the job templates declare the vLLM version the serve extra installs, and Python by minor
    version only, so a fresh install collects without editing a template
    """
    with open("pyproject.toml", encoding="utf-8") as fh:
        pinned = [line for line in fh if "vllm==" in line]
    assert len(pinned) == 1
    version = pinned[0].split("vllm==")[1].split(";")[0].strip()
    templates = sorted(glob.glob("configs/run_experiment_recycle_v*.yaml"))
    assert templates
    for path in templates:
        with open(path, encoding="utf-8") as fh:
            env = yaml.safe_load(fh)["serving"]["serving_env"]
        assert env["vllm"] == version, path
        assert env["python"] == "3.13", path


def _locked_version(package: str) -> str:
    """
    the version uv.lock pins for a package
    :param package: the package name as the lock spells it (str)
    :returns: the locked version (str)
    """
    with open("uv.lock", encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    at = lines.index(f'name = "{package}"')
    return lines[at + 1].split('"')[1]


def test_job_templates_declare_the_locked_torch():
    """
    the torch version the templates declare is the one the lock installs beside the pinned vLLM
    """
    torch = _locked_version("torch")
    for path in sorted(glob.glob("configs/run_experiment_recycle_v*.yaml")):
        with open(path, encoding="utf-8") as fh:
            env = yaml.safe_load(fh)["serving"]["serving_env"]
        assert env["torch"] == torch, path


def _resolve_against(monkeypatch, observed: dict):
    """
    run the in-job provider's resolve() on a real job template's serving block against a
    server reporting the given versions, with the HTTP reads replaced
    :param monkeypatch: the pytest fixture
    :param observed: component name to the value the server reports (dict)
    :returns: the resolved serving record (ResolvedServing)
    """
    import subprocess

    from lmopt.serving import chat_template
    from lmopt.serving.control import ServedModel, ServingController
    from lmopt.serving.injob import InJobServing

    model = "org/model"
    with open("configs/run_experiment_recycle_v8_m0.yaml", encoding="utf-8") as fh:
        serving = yaml.safe_load(fh)["serving"]
    monkeypatch.setattr(
        ServingController,
        "current",
        lambda self: ServedModel(model_id=model, max_model_len=8192, served_ids=(model,)),
    )
    monkeypatch.setattr(chat_template, "resolve_chat_template_hash", lambda *a, **k: "h")
    provider = InJobServing.from_config(model, "http://localhost:8000/v1", serving)
    return provider.resolve(
        argv_scan=lambda: ("vllm", "serve", model),
        env_probe=observed.get,
        nvidia_smi_probe=lambda: subprocess.CompletedProcess([], 1, "", ""),
    )


def test_resolve_accepts_any_patch_release_and_records_what_it_observed(monkeypatch):
    """
    a server on the declared vLLM and torch and any 3.13 patch release resolves, and the record
    carries the observed full Python version, not the declared minor version
    """
    observed = {
        "vllm": _locked_version("vllm"),
        "torch": _locked_version("torch"),
        "python": "3.13.5",
    }
    resolved = _resolve_against(monkeypatch, observed)
    assert resolved.serving_env["python"] == "3.13.5"
    assert resolved.serving_env["vllm"] == observed["vllm"]


@pytest.mark.parametrize(
    "component, value", [("vllm", "0.18.0"), ("torch", "2.10.0"), ("python", "3.12.9")]
)
def test_resolve_refuses_a_server_on_another_stack(monkeypatch, component, value):
    """
    a server whose vLLM, torch or Python minor version differs from the template is refused
    """
    from lmopt.serving.injob import InJobServingError

    observed = {
        "vllm": _locked_version("vllm"),
        "torch": _locked_version("torch"),
        "python": "3.13.5",
    }
    observed[component] = value
    with pytest.raises(InJobServingError, match=component):
        _resolve_against(monkeypatch, observed)
