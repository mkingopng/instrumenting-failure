"""Weights & Biases tracking hook.

Default-off, offline-capable, mirror-only: enabling W&B must not change
any logged Parquet content -- tracking is a mirror, not a source of truth. W&B never
sees a decision node until after it is already written to Parquet by
``lmopt.logging.writer.MatchWriter`` inside ``run_matchup_shard`` -- this
module only mirrors already-durable shard/score artifacts to a dashboard; it
never participates in scoring or in what gets logged to Parquet.

``init_tracking`` reads a single ``wandb:`` block from the (already-merged)
run config and returns a ``Tracker``: a ``NullTracker`` (every method a
no-op, never imports wandb) unless the config explicitly opts in with
``wandb.enabled: true``, in which case a ``WandbTracker`` is returned.
Callers (``scripts/run_sweep.py``) call the same
three methods regardless of which concrete tracker they got back.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol


def flatten_scores(scores: dict) -> dict:
    """Make a scores dict wandb-serialisable.

    KLScore is a (value, on_support_fraction) pair -- wandb cannot log a
    dataclass, and the caller swallows the resulting exception, so an unflattened
    KLScore silently drops P3 from the mirror. Expanding it to two explicit
    metrics also puts reference coverage on the dashboard, which is where a
    partially-covered game most needs to be visible.
    """
    from lmopt.scoring.property3 import KLScore

    def _f(v):
        if isinstance(v, KLScore):
            return {"value": v.value, "on_support_fraction": v.on_support_fraction}
        if isinstance(v, dict):
            return {k: _f(x) for k, x in v.items()}
        return v

    return _f(scores)


class Tracker(Protocol):
    """Structural interface both ``NullTracker`` and ``WandbTracker`` satisfy."""

    def log_shard(self, spec_dict: dict, parquet_path: str) -> None:
        """Mirror one completed matchup shard (spec fields + its Parquet path)."""
        ...

    def log_scores(self, scores: dict) -> None:
        """Mirror an offline scoring run's property-score dict."""
        ...

    def finish(self) -> None:
        """Close out the tracking run, if any."""
        ...


class NullTracker:
    """The default tracker: every method is a no-op.

    Never imports or touches wandb, so a run with tracking off has zero
    dependency on the (optional, ``tracking`` extra) wandb package.
    """

    def log_shard(self, spec_dict: dict, parquet_path: str) -> None:
        pass

    def log_scores(self, scores: dict) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbTracker:
    """Thin wrapper over a single wandb run.

    wandb is imported lazily, inside ``__init__`` -- not at module import
    time -- so ``import lmopt.logging.tracking`` (and building a
    ``NullTracker``) never requires the wandb package to be installed. Only
    actually enabling tracking (``wandb.enabled: true``) does.
    """

    def __init__(
        self,
        project: str = "lmopt",
        mode: Literal["online", "offline", "disabled", "shared"] = "offline",
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
    ) -> None:
        # A leftover empty directory named `wandb` on
        # sys.path (e.g. a stale editable-install remnant) makes `import
        # wandb` succeed as an implicit namespace package -- no ImportError,
        # but none of the real package's attributes. The `hasattr` check
        # below catches that case too, so both failure modes raise the same
        # worded ImportError instead of an AttributeError from wandb.init.
        try:
            import wandb

            if not hasattr(wandb, "init"):
                raise ImportError("wandb resolved to an empty namespace package")
        except ImportError as exc:
            raise ImportError(
                "W&B tracking is enabled (cfg['wandb']['enabled']=True) but the "
                "wandb package is not installed. Install the optional extra with: "
                "uv sync --extra tracking"
            ) from exc
        self._run = wandb.init(
            project=project, mode=mode, config=config, tags=tags, group=group
        )

    def log_shard(self, spec_dict: dict, parquet_path: str) -> None:
        payload = dict(spec_dict)
        payload["parquet_path"] = parquet_path
        self._run.log({"shard": payload})

    def log_scores(self, scores: dict) -> None:
        self._run.log({"scores": flatten_scores(scores)})

    def finish(self) -> None:
        self._run.finish()


def init_tracking(cfg: dict, run_meta: dict) -> Tracker:
    """Build the ``Tracker`` a run should use, from its (already-merged) config.

    ``cfg["wandb"]`` absent, or present with a falsy/missing ``enabled`` key,
    returns a ``NullTracker`` -- W&B is opt-in, never default.
    Otherwise returns a ``WandbTracker``, with ``project`` defaulting to
    ``"lmopt"`` and ``mode`` defaulting to ``"offline"`` (wandb writes a local
    run directory with no network calls required; ``wandb sync`` uploads it
    later, on request). ``run_meta`` -- run-level provenance (git sha, seed
    policy, resolved config, ...) -- is passed through as the wandb run's
    ``config`` when tracking is enabled.
    ``tags`` (a list of strings) and ``group`` (a single string) are read
    straight from the same ``wandb:`` block, undefined by default. This
    function provides the mechanism for tagging and grouping a run; which
    tags/group values a given config sets is a config-authoring decision,
    not something this function guesses.
    """
    wandb_cfg = cfg.get("wandb") or {}
    if not wandb_cfg.get("enabled", False):
        return NullTracker()
    project = wandb_cfg.get("project", "lmopt")
    mode = wandb_cfg.get("mode", "offline")
    tags = wandb_cfg.get("tags")
    group = wandb_cfg.get("group")
    return WandbTracker(
        project=project, mode=mode, config=run_meta, tags=tags, group=group
    )
