# src/biosignals/agent/feedback.py
"""
Feedback-driven experiment tracking and drift detection.

Reads the standardized artifact schema produced by biosignals.engine.trainer:
  - metrics.jsonl   (per-epoch records)
  - summary.json    (best/last pointers + monitor policy)
  - config_resolved.yaml (full Hydra config)

Provides:
  - RunRecord: structured view of a single completed run
  - FeedbackStore: accumulates runs, detects drift/stagnation, ranks experiments
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("biosignals.agent")


# ─────────────────────────────────────────────────
# Run artifact parsing
# ─────────────────────────────────────────────────


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml_as_dict(path: Path) -> Dict[str, Any]:
    """Read a YAML file. Uses OmegaConf if available, else pyyaml."""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(path))
        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


@dataclass
class EpochRecord:
    """One epoch from metrics.jsonl."""

    epoch: int
    global_step: int
    train: Dict[str, float]
    val: Optional[Dict[str, float]]
    monitor_metric: str
    monitor_mode: str
    monitor_value: float

    @classmethod
    def from_jsonl_record(cls, rec: Dict[str, Any]) -> EpochRecord:
        mon = rec.get("monitor", {})
        return cls(
            epoch=int(rec["epoch"]),
            global_step=int(rec.get("global_step", 0)),
            train=rec.get("train") or {},
            val=rec.get("val") or {},
            monitor_metric=str(mon.get("metric", "val/loss")),
            monitor_mode=str(mon.get("mode", "min")),
            monitor_value=float(mon.get("value", float("nan"))),
        )


@dataclass
class RunRecord:
    """
    Structured view of a single completed training run.

    Parsed from the standardized artifact schema in a Hydra run directory.
    This is the atomic unit the FeedbackStore tracks.
    """

    run_dir: str
    timestamp: str  # ISO format, from directory mtime or parsing

    # From summary.json
    monitor_metric: str
    monitor_mode: str  # "min" or "max"
    best_epoch: Optional[int]
    best_value: Optional[float]
    best_ckpt_path: Optional[str]
    last_epoch: int
    last_ckpt_path: Optional[str]

    # From metrics.jsonl
    epoch_history: List[EpochRecord]

    # From config_resolved.yaml (flattened key fields)
    task_name: str
    model_name: str
    dataset_name: str
    lr: float
    epochs_configured: int
    overrides: List[str] = field(default_factory=list)

    # Full resolved config (for agent inspection)
    config: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_epochs_completed(self) -> int:
        return len(self.epoch_history)

    @property
    def final_monitor_value(self) -> float:
        """The monitored metric at the last epoch."""
        if self.epoch_history:
            return self.epoch_history[-1].monitor_value
        return float("nan")

    @property
    def best_monitor_value(self) -> float:
        """The best monitored metric across all epochs."""
        if self.best_value is not None:
            return self.best_value
        return self.final_monitor_value

    @property
    def converged(self) -> bool:
        """
        Heuristic: did the run improve in the last 25% of epochs?
        If not, it likely converged or stagnated.
        """
        if len(self.epoch_history) < 4:
            return False
        quarter = max(1, len(self.epoch_history) // 4)
        late = self.epoch_history[-quarter:]
        early_end = self.epoch_history[-(quarter * 2) : -quarter]
        if not early_end:
            return False

        late_avg = sum(e.monitor_value for e in late) / len(late)
        early_avg = sum(e.monitor_value for e in early_end) / len(early_end)

        if self.monitor_mode == "min":
            return late_avg >= early_avg * 0.995  # not improving
        else:
            return late_avg <= early_avg * 1.005

    @property
    def failed(self) -> bool:
        return self.n_epochs_completed == 0

    def training_curve(self, metric: Optional[str] = None) -> List[Tuple[int, float]]:
        """
        Extract a training curve for a given metric.
        If metric is None, uses the monitor metric.
        Returns list of (epoch, value) tuples.
        """
        points = []
        for rec in self.epoch_history:
            if metric is None:
                val = rec.monitor_value
            else:
                # Try val/ first, then train/
                prefix, key = (metric.split("/", 1) + [""])[:2]
                if prefix == "val" and rec.val:
                    val = rec.val.get(key, float("nan"))
                elif prefix == "train":
                    val = rec.train.get(key, float("nan"))
                else:
                    val = rec.val.get(metric, rec.train.get(metric, float("nan")))
            points.append((rec.epoch, val))
        return points

    def summary_for_agent(self) -> str:
        """Compact text summary an LLM agent can reason about."""
        status = "FAILED" if self.failed else ("CONVERGED" if self.converged else "COMPLETED")
        lines = [
            f"Run: {self.run_dir}",
            f"Status: {status}",
            f"Task: {self.task_name} | Model: {self.model_name} | Dataset: {self.dataset_name}",
            f"LR: {self.lr} | Epochs: {self.n_epochs_completed}/{self.epochs_configured}",
            f"Monitor: {self.monitor_metric} ({self.monitor_mode})",
            f"Best: {self.best_monitor_value:.6f} @ epoch {self.best_epoch}",
            f"Final: {self.final_monitor_value:.6f}",
        ]
        if self.overrides:
            lines.append(f"Overrides: {self.overrides}")
        return "\n".join(lines)


def parse_run_dir(run_dir: str | Path, overrides: Optional[List[str]] = None) -> RunRecord:
    """
    Parse a completed Hydra run directory into a RunRecord.

    This is the bridge between your training system's file outputs
    and the feedback-driven agent layer.
    """
    run_path = Path(run_dir).resolve()

    # ── summary.json ──
    summary_path = run_path / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}

    monitor = summary.get("monitor", {})
    best = summary.get("best", {})
    last = summary.get("last", {})

    # ── metrics.jsonl ──
    metrics_path = run_path / "metrics.jsonl"
    epoch_history: List[EpochRecord] = []
    if metrics_path.exists():
        for rec in _read_jsonl(metrics_path):
            try:
                epoch_history.append(EpochRecord.from_jsonl_record(rec))
            except (KeyError, ValueError) as e:
                log.warning("Skipping malformed metrics record in %s: %s", run_dir, e)

    # ── config_resolved.yaml ──
    config_path = run_path / "config_resolved.yaml"
    config: Dict[str, Any] = {}
    task_name = "unknown"
    model_name = "unknown"
    dataset_name = "unknown"
    lr = 0.0
    epochs_configured = 0

    if config_path.exists():
        config = _read_yaml_as_dict(config_path)
        # Extract key fields from nested config
        task_cfg = config.get("task", {})
        model_cfg = config.get("model", {})
        dataset_cfg = config.get("dataset", {})
        trainer_cfg = config.get("trainer", {})

        task_name = str(task_cfg.get("name", task_cfg.get("_target_", "unknown"))).split(".")[-1]
        model_name = str(model_cfg.get("_target_", "unknown")).split(".")[-1]
        dataset_name = str(dataset_cfg.get("train", {}).get("_target_", "unknown")).split(".")[-1]
        lr = float(trainer_cfg.get("lr", 0.0))
        epochs_configured = int(trainer_cfg.get("epochs", 0))

    # ── Timestamp from directory ──
    try:
        ts = datetime.fromtimestamp(run_path.stat().st_mtime).isoformat()
    except OSError:
        ts = datetime.now().isoformat()

    return RunRecord(
        run_dir=str(run_path),
        timestamp=ts,
        monitor_metric=str(monitor.get("metric", "val/loss")),
        monitor_mode=str(monitor.get("mode", "min")),
        best_epoch=best.get("epoch"),
        best_value=best.get("value"),
        best_ckpt_path=best.get("ckpt_path"),
        last_epoch=int(last.get("epoch", len(epoch_history) - 1)),
        last_ckpt_path=last.get("ckpt_path"),
        epoch_history=epoch_history,
        task_name=task_name,
        model_name=model_name,
        dataset_name=dataset_name,
        lr=lr,
        epochs_configured=epochs_configured,
        overrides=overrides or [],
        config=config,
    )


# ─────────────────────────────────────────────────
# FeedbackStore — the core feedback-driven component
# ─────────────────────────────────────────────────


@dataclass
class DriftReport:
    """Actionable drift/stagnation diagnostics across experiment runs."""

    drift_detected: bool
    stagnation_detected: bool
    reasons: List[str]
    recommendation: str  # "continue", "change_lr", "change_model", "stop"
    best_run_dir: Optional[str]
    best_value: Optional[float]
    n_runs_analyzed: int
    trend_slope: Optional[float]  # negative = degrading for max metrics

    def summary_for_agent(self) -> str:
        lines = [
            f"Drift: {'YES' if self.drift_detected else 'no'}",
            f"Stagnation: {'YES' if self.stagnation_detected else 'no'}",
            f"Runs analyzed: {self.n_runs_analyzed}",
            f"Best value: {self.best_value:.6f}" if self.best_value is not None else "Best: N/A",
            f"Trend slope: {self.trend_slope:.6f}"
            if self.trend_slope is not None
            else "Trend: N/A",
            f"Recommendation: {self.recommendation}",
        ]
        if self.reasons:
            lines.append(f"Reasons: {'; '.join(self.reasons)}")
        return "\n".join(lines)


class FeedbackStore:
    """
    Accumulates RunRecords across experiments and provides
    feedback signals for the agent loop.

    This is the core of what makes the system "feedback-driven":
    it tracks HOW WELL experiments are going over time and produces
    actionable signals (drift, stagnation, recommendations).

    The store does NOT modify training code — it only reads artifacts.
    """

    def __init__(
        self,
        *,
        stagnation_window: int = 3,
        stagnation_tolerance: float = 0.005,
        max_history: int = 100,
    ):
        self.runs: List[RunRecord] = []
        self.stagnation_window = stagnation_window
        self.stagnation_tolerance = stagnation_tolerance
        self.max_history = max_history

    def add_run(self, run: RunRecord) -> None:
        """Register a completed run."""
        self.runs.append(run)
        if len(self.runs) > self.max_history:
            self.runs = self.runs[-self.max_history :]
        log.info(
            "FeedbackStore: added run %s (best=%s, %d total)",
            run.run_dir,
            run.best_monitor_value,
            len(self.runs),
        )

    def add_run_dir(self, run_dir: str | Path, overrides: Optional[List[str]] = None) -> RunRecord:
        """Parse a run directory and add it to the store."""
        run = parse_run_dir(run_dir, overrides=overrides)
        self.add_run(run)
        return run

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def successful_runs(self) -> List[RunRecord]:
        return [r for r in self.runs if not r.failed]

    def best_run(self) -> Optional[RunRecord]:
        """Return the run with the best monitor value."""
        successful = self.successful_runs
        if not successful:
            return None

        mode = successful[-1].monitor_mode  # use latest run's mode

        def key(r: RunRecord) -> float:
            v = r.best_monitor_value
            if math.isnan(v):
                return float("-inf") if mode == "max" else float("inf")
            return v if mode == "max" else -v

        return max(successful, key=key)

    def rank_runs(self) -> List[RunRecord]:
        """Return all successful runs ranked best-to-worst."""
        successful = self.successful_runs
        if not successful:
            return []

        mode = successful[-1].monitor_mode

        def key(r: RunRecord) -> float:
            v = r.best_monitor_value
            if math.isnan(v):
                return float("-inf") if mode == "max" else float("inf")
            return v if mode == "max" else -v

        return sorted(successful, key=key, reverse=True)

    def detect_drift(self) -> DriftReport:
        """
        Analyze experiment history for drift and stagnation.

        Drift: recent runs are WORSE than earlier runs.
        Stagnation: recent runs show no improvement despite config changes.

        This is the primary feedback signal for the agent.
        """
        successful = self.successful_runs
        n = len(successful)

        if n < 2:
            return DriftReport(
                drift_detected=False,
                stagnation_detected=False,
                reasons=["insufficient_history"],
                recommendation="continue",
                best_run_dir=successful[0].run_dir if successful else None,
                best_value=successful[0].best_monitor_value if successful else None,
                n_runs_analyzed=n,
                trend_slope=None,
            )

        mode = successful[-1].monitor_mode
        values = [r.best_monitor_value for r in successful if not math.isnan(r.best_monitor_value)]

        if len(values) < 2:
            return DriftReport(
                drift_detected=False,
                stagnation_detected=False,
                reasons=["no_valid_metrics"],
                recommendation="continue",
                best_run_dir=None,
                best_value=None,
                n_runs_analyzed=n,
                trend_slope=None,
            )

        # ── Trend analysis ──
        # Linear regression on run index vs metric value
        import numpy as np

        x = np.arange(len(values), dtype=float)
        y = np.array(values, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])

        # ── Drift detection ──
        # Is the trend going the WRONG direction?
        drift_detected = False
        reasons: List[str] = []

        if mode == "max" and slope < -self.stagnation_tolerance:
            drift_detected = True
            reasons.append(f"degrading_trend(slope={slope:.6f}, mode=max)")
        elif mode == "min" and slope > self.stagnation_tolerance:
            drift_detected = True
            reasons.append(f"degrading_trend(slope={slope:.6f}, mode=min)")

        # Is the latest run worse than the best by a significant margin?
        best_run = self.best_run()
        latest = successful[-1]
        if best_run and not math.isnan(latest.best_monitor_value):
            gap = abs(best_run.best_monitor_value - latest.best_monitor_value)
            ref = abs(best_run.best_monitor_value) + 1e-8
            if gap / ref > 0.1:  # >10% regression from best
                drift_detected = True
                reasons.append(
                    f"regression_from_best("
                    f"latest={latest.best_monitor_value:.6f}, "
                    f"best={best_run.best_monitor_value:.6f}, "
                    f"gap={gap / ref:.1%})"
                )

        # ── Stagnation detection ──
        stagnation_detected = False
        window = min(self.stagnation_window, len(values))
        if window >= 2:
            recent_values = values[-window:]
            spread = max(recent_values) - min(recent_values)
            ref = abs(sum(recent_values) / len(recent_values)) + 1e-8
            if spread / ref < self.stagnation_tolerance:
                stagnation_detected = True
                reasons.append(
                    f"stagnation(window={window}, spread={spread:.6f}, "
                    f"tolerance={self.stagnation_tolerance})"
                )

            # Also check: are all recent runs converged?
            recent_runs = successful[-window:]
            if all(r.converged for r in recent_runs):
                stagnation_detected = True
                reasons.append("all_recent_converged")

        # ── Recommendation ──
        recommendation = "continue"
        if stagnation_detected and drift_detected:
            recommendation = "change_model"
        elif drift_detected or stagnation_detected:
            recommendation = "change_lr"
        elif n >= 5 and not drift_detected and not stagnation_detected:
            recommendation = "continue"  # healthy exploration

        return DriftReport(
            drift_detected=drift_detected,
            stagnation_detected=stagnation_detected,
            reasons=reasons if reasons else ["healthy"],
            recommendation=recommendation,
            best_run_dir=best_run.run_dir if best_run else None,
            best_value=best_run.best_monitor_value if best_run else None,
            n_runs_analyzed=n,
            trend_slope=slope,
        )

    def history_for_agent(self) -> str:
        """
        Produce a compact text summary of ALL runs for the agent's context.
        Designed to fit in an LLM context window efficiently.
        """
        if not self.runs:
            return "No runs recorded yet."

        lines = [f"Experiment History ({len(self.runs)} runs):"]
        lines.append("-" * 60)

        for i, run in enumerate(self.runs):
            status = "FAIL" if run.failed else ("CONV" if run.converged else "OK")
            lines.append(
                f"  [{i}] {status} | {run.task_name}/{run.model_name} | "
                f"lr={run.lr} | best={run.best_monitor_value:.6f} "
                f"({run.monitor_metric} {run.monitor_mode}) | "
                f"epochs={run.n_epochs_completed}/{run.epochs_configured}"
            )

        # Add drift summary
        drift = self.detect_drift()
        lines.append("-" * 60)
        lines.append(drift.summary_for_agent())

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize store state (for persistence / ZenML artifact)."""
        return {
            "n_runs": self.n_runs,
            "runs": [
                {
                    "run_dir": r.run_dir,
                    "task": r.task_name,
                    "model": r.model_name,
                    "lr": r.lr,
                    "best_value": r.best_monitor_value,
                    "monitor": r.monitor_metric,
                    "mode": r.monitor_mode,
                    "epochs": r.n_epochs_completed,
                    "overrides": r.overrides,
                    "converged": r.converged,
                    "failed": r.failed,
                }
                for r in self.runs
            ],
            "drift": {
                "detected": self.detect_drift().drift_detected,
                "stagnation": self.detect_drift().stagnation_detected,
                "recommendation": self.detect_drift().recommendation,
            },
        }
