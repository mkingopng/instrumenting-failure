"""
Reading the serving box's HF cache over ssh to name the weights revision it holds.

The helpers here are used by the launcher (`scripts/run_experiment.py`), which resolves the revision before each arm
launch.
"""

from __future__ import annotations

import datetime as dt
import subprocess

DEFAULT_SSH_HOST = "spark"
CACHE_ROOT = "/data/hf-cache/hub"


def ssh(script: str, host: str = DEFAULT_SSH_HOST) -> str:
    """Run a read-only shell snippet on the serving box.

    `host` lets a launcher with its own configured serving host
    (run_experiment.py's serving.ssh_host) reuse this transport; the
    default serves call sites that read the spark box.
    """
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, script],
        capture_output=True,
        text=True,
        timeout=120,
        # The exit code is read below, so a failure is handled, not raised.
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"ssh {host!r} failed rc={out.returncode}: {out.stderr[:400]}"
        )
    return out.stdout


def cache_evidence(
    model_id: str, host: str = DEFAULT_SSH_HOST, cache_root: str = CACHE_ROOT
) -> dict:
    """Read the four inference conditions off the serving box's HF cache.

    `cache_root` is parameterized to match the host parameter's convention
    -- CLI flag + env fallback, default = the module constant CACHE_ROOT."""
    slug = "models--" + model_id.replace("/", "--")
    d = f"{cache_root}/{slug}"
    raw = ssh(
        host=host,
        script=(
            f"d={d}; "
            f'if [ ! -d "$d" ]; then echo "MISSING"; exit 0; fi; '
            f'echo "snapshots=$(ls $d/snapshots 2>/dev/null | tr "\\n" ",")"; '
            f'echo "refs_main=$(cat $d/refs/main 2>/dev/null)"; '
            f'echo "n_blobs=$(ls $d/blobs 2>/dev/null | wc -l)"; '
            f'echo "n_incomplete=$(ls $d/blobs/*.incomplete 2>/dev/null | wc -l)"; '
            f'echo "snapshot_mtime_epoch=$(stat -c %Y $d/snapshots/* 2>/dev/null | tr "\\n" ",")"'
        ),
    )
    if raw.strip() == "MISSING":
        return {"present": False}

    fields = dict(
        line.split("=", 1) for line in raw.strip().splitlines() if "=" in line
    )
    snaps = [s for s in fields.get("snapshots", "").split(",") if s]
    epochs = [int(e) for e in fields.get("snapshot_mtime_epoch", "").split(",") if e]
    return {
        "present": True,
        "cache_path": d,
        "snapshots": snaps,
        "n_snapshots": len(snaps),
        "refs_main": fields.get("refs_main") or None,
        "n_blobs": int(fields.get("n_blobs", 0)),
        "n_incomplete": int(fields.get("n_incomplete", 0)),
        "snapshot_mtime_utc": [
            dt.datetime.fromtimestamp(e, dt.timezone.utc).isoformat() for e in epochs
        ],
    }


def cache_shape_conditions(ev: dict) -> dict:
    """The three ARM-independent cache-shape conditions of the inference,
    shared with main()'s conds so the
    per-arm capture record and `resolve_prelaunch_revision` below share one
    implementation, never a fork. The fourth condition,
    `snapshot_predates_collection`, is genuinely arm-specific (it needs the
    arm's `created_utc`) and stays in main().
    """
    return {
        "single_snapshot": ev.get("n_snapshots") == 1,
        "refs_main_agrees": bool(ev.get("refs_main"))
        and ev.get("refs_main") in (ev.get("snapshots") or []),
        "blobs_complete": ev.get("n_incomplete") == 0,
    }


def resolve_prelaunch_revision(
    model_id: str, *, host: str, cache_root: str = CACHE_ROOT
) -> str:
    """Resolve the revision the serving box will serve for `model_id`, for
    stamping at collection time (consumed by scripts/run_experiment.py per
    arm launch and handed to run_sweep as --served-weights-revision).

    `host` is a required keyword, deliberately without this module's
    DEFAULT_SSH_HOST default: a caller that omits it would otherwise
    silently read SPARK's cache and stamp spark's snapshot onto an arm
    served from some other box -- the exact wrong-box provenance lie
    collection-time stamping exists to prevent. The launcher passes its own
    configured serving.ssh_host.

    `cache_root` is parameterized to match the host parameter's convention
    -- default = the module constant CACHE_ROOT.

    Same evidence (`cache_evidence`) and same cache-shape conditions
    (`cache_shape_conditions`) as the per-arm capture in main() -- one
    implementation. STRICTER on the verdict, deliberately: main() records a
    revision under `single_snapshot` alone and lets its `confidence` field
    carry the ambiguity, but a value stamped onto every row of an arm has no
    confidence field to degrade, so all three conditions must hold here or
    this raises (no silent fallbacks).
    `snapshot_predates_collection` does not apply: collection has not
    started yet, so the snapshot on disk trivially predates it.

    Raises RuntimeError naming the model, the host and the failing
    conditions; never returns a guessed revision.
    """
    ev = cache_evidence(model_id, host=host, cache_root=cache_root)
    if not ev.get("present"):
        raise RuntimeError(
            f"served-weights revision unresolvable for {model_id!r}: no HF "
            f"cache dir for it on {host!r} ({cache_root})"
        )
    conds = cache_shape_conditions(ev)
    if not all(conds.values()):
        failing = ", ".join(sorted(k for k, v in conds.items() if not v))
        raise RuntimeError(
            f"served-weights revision unresolvable for {model_id!r} on "
            f"{host!r}: cache-shape condition(s) failed: {failing} "
            f"(snapshots={ev.get('snapshots')!r}, "
            f"refs_main={ev.get('refs_main')!r}, "
            f"n_incomplete={ev.get('n_incomplete')!r})"
        )
    return ev["snapshots"][0]
