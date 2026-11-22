"""
The job a collection run belongs to, and the attempts that came before it in the same directory.

Nothing here derives or defaults a value. A field the job environment does not provide is
None, a variable a cluster run cannot do without is refused, and an earlier attempt is carried
exactly as its own `run_meta.json` recorded it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

#: The marker the cluster job script exports. It is the only thing that says a run
#: is on the cluster: `PBS_JOBID` being present is not, so "on the cluster with a missing variable" stays
#: distinct from "not on the cluster".
ON_KATANA_ENV = "LMOPT_ON_KATANA"
#: The one variable a cluster run cannot be joined to its job without.
JOB_ID_ENV = "PBS_JOBID"
QUEUE_ENV = "PBS_QUEUE"
INDEX_ENV = "PBS_ARRAY_INDEX"
#: The walltime the submission asked for, which PBS does not put in the job's environment.
WALLTIME_ENV = "LMOPT_JOB_WALLTIME"
GPU_ENV = "CUDA_VISIBLE_DEVICES"
#: The card's own identity, exported by the cluster job script from `nvidia-smi` after
#: its GPU guard. `CUDA_VISIBLE_DEVICES` is not set on the cluster and a one-card cgroup reports index
#: 0 whichever physical card it is, so neither tells two jobs on one node apart; the UUID does.
GPU_UUID_ENV = "LMOPT_GPU_UUID"
GPU_BUS_ENV = "LMOPT_GPU_PCI_BUS_ID"


class JobIdentityUnavailable(RuntimeError):
    """
    Raised when a run marked as on the cluster lacks the variable that names its job.
    """


class PredecessorUnreadable(RuntimeError):
    """
    Raised when the `run_meta.json` already in a run directory cannot be read as a history.

    The file is written atomically, so this is never a half-written file from a killed attempt:
    something is wrong with that directory and nobody has looked. The run stops rather than
    overwrite the only copy.
    """


def is_on_katana(env: Mapping[str, str]) -> bool:
    """
    the one reader of the cluster job marker; every script that asks whether it runs on the cluster
    goes through this, so an off marker is never read as on and a typo is never read as off
    :param env: the job's environment
    :returns: True when the marker is exactly "1", False when it is unset, empty or "0"
    :raises ValueError: for any other value, so a typo is never read as "not on the cluster"
    """
    value = env.get(ON_KATANA_ENV)
    if value in (None, "", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{ON_KATANA_ENV} must be 1, 0 or unset, got {value!r}")


def job_block(env: Mapping[str, str], argv: Sequence[str], host: str) -> dict[str, Any]:
    """
    the `job` block for this attempt, read from the environment the job was given
    :param env: the process environment (mapping of str to str)
    :param argv: the launching command line (sequence of str)
    :param host: the host name (str)
    :returns: a dict with on_katana, pbs_job_id, queue, walltime_requested, index_label, host,
        gpu_index, gpu_uuid, gpu_pci_bus_id and argv. The two card fields are exactly what the
        job exported, None when it exported nothing; they are never derived or defaulted. Off the cluster the four PBS-derived fields are None, meaning not
        applicable. On the cluster a pilot kind has no array index, so index_label is None there.
        Identity facts refuse and submission facts may be absent: the job id says which job
        this is, so its absence refuses the run, while the requested walltime is a fact about
        the submission, so a bare `qsub` that did not export it leaves it None with
        on_katana still true. It is never defaulted to an estimate.
    :raises JobIdentityUnavailable: when the run is on the cluster and PBS_JOBID is missing or empty
    :raises ValueError: when the on-cluster marker holds anything but 1, 0 or nothing
    """
    on_katana = is_on_katana(env)
    pbs_fields: dict[str, str | None] = {
        "pbs_job_id": None,
        "queue": None,
        "walltime_requested": None,
        "index_label": None,
    }
    if on_katana:
        job_id = env.get(JOB_ID_ENV)
        if not job_id:
            raise JobIdentityUnavailable(
                f"this run is on the cluster ({ON_KATANA_ENV}=1) but {JOB_ID_ENV} is "
                f"{'empty' if job_id == '' else 'not set'}, so it cannot be joined to its job"
            )
        pbs_fields = {
            "pbs_job_id": job_id,
            "queue": env.get(QUEUE_ENV),
            "walltime_requested": env.get(WALLTIME_ENV),
            "index_label": env.get(INDEX_ENV),
        }
    return {
        "on_katana": on_katana,
        **pbs_fields,
        "host": host,
        "gpu_index": env.get(GPU_ENV),
        "gpu_uuid": env.get(GPU_UUID_ENV),
        "gpu_pci_bus_id": env.get(GPU_BUS_ENV),
        "argv": list(argv),
    }


def previous_attempts(
    meta_path: str, current: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """
    the earlier attempts recorded in a run directory's existing `run_meta.json`, oldest first
    :param meta_path: path of the run directory's `run_meta.json` (str); may not exist
    :param current: this attempt's `job` block, to recognise a rerun inside one job
    :returns: the old file's own `previous_jobs`, then one entry for the old file's attempt: its
        `job` block plus `created_utc` and `finished_utc` copied from that file (`finished_utc`
        None when the file recorded none). No label says why an attempt has no finish. An
        absent file, or a file from before the job block existed, adds nothing, and no entry is
        invented. An old attempt in the same job as `current`, compared only when both carry a
        job id, is not listed, since a resumed sweep inside one job is not an earlier job; its
        own history is still carried.
    :raises PredecessorUnreadable: when the file exists but is not a JSON object, or its `job`
        is not an object, or its `previous_jobs` is not a list of objects
    """
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path) as fh:
            old = json.load(fh)
    except (OSError, ValueError) as exc:
        raise PredecessorUnreadable(
            _refusal(meta_path, f"it cannot be read ({exc})")
        ) from exc
    if not isinstance(old, dict):
        raise PredecessorUnreadable(_refusal(meta_path, "it is not a JSON object"))
    history = old.get("previous_jobs", [])
    if not isinstance(history, list) or not all(isinstance(e, dict) for e in history):
        raise PredecessorUnreadable(
            _refusal(meta_path, "its previous_jobs is not a list of objects")
        )
    carried = [dict(e) for e in history]
    if "job" not in old:
        return carried
    old_job = old["job"]
    if not isinstance(old_job, dict):
        raise PredecessorUnreadable(_refusal(meta_path, "its job is not an object"))
    old_id, current_id = old_job.get("pbs_job_id"), current.get("pbs_job_id")
    if old_id is not None and old_id == current_id:
        return carried
    carried.append(
        {
            **old_job,
            "created_utc": old.get("created_utc"),
            "finished_utc": old.get("finished_utc"),
        }
    )
    return carried


def _refusal(meta_path: str, why: str) -> str:
    """
    :param meta_path: the file that cannot be read (str)
    :param why: the reason (str)
    :returns: the refusal text naming the file and the way forward (str)
    """
    return (
        f"{meta_path} already exists and {why}. This run would overwrite the only copy, so it "
        f"stops. Look at the directory, then move that file aside by hand; the job will then "
        f"start with no earlier attempts to carry."
    )
