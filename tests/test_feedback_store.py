# tests/test_feedback_store.py
"""
Unit tests for FeedbackStore and RunRecord parsing.
These do NOT require API keys or network — they test the core feedback logic
against mock artifacts matching the real trainer.py output format.

Run from project root:
    PYTHONPATH=$PWD/src python -m pytest tests/test_feedback_store.py -v
"""
import json
import tempfile
from pathlib import Path

import pytest

from biosignals.agent.feedback import (
    FeedbackStore,
    RunRecord,
    DriftReport,
    EpochRecord,
    parse_run_dir,
)


# ─────────────────────────────────────────────────
# Fixtures: create mock run directories matching
# the exact artifact schema from trainer.py
# ─────────────────────────────────────────────────


def _make_run_dir(
    tmp_path: Path,
    name: str,
    epochs: list[dict],
    best_epoch: int,
    best_value: float,
    lr: float = 0.0003,
    monitor_metric: str = "val/mae",
    monitor_mode: str = "min",
    task_target: str = "biosignals.tasks.regression.RegressionTask",
    model_target: str = "biosignals.models.architectures.encoder_classifier",
    dataset_target: str = "biosignals.data.datasets.galaxyppg.GalaxyPPGDataset",
) -> Path:
    """Create a mock Hydra run directory with real artifact format."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)

    # metrics.jsonl — exact format from trainer.py lines 203-211
    with open(run_dir / "metrics.jsonl", "w") as f:
        for rec in epochs:
            f.write(json.dumps(rec) + "\n")

    # summary.json — exact format from trainer.py lines 242-249
    summary = {
        "monitor": {"metric": monitor_metric, "mode": monitor_mode},
        "best": {
            "epoch": best_epoch,
            "value": best_value,
            "ckpt_path": str(run_dir / "checkpoints" / "best.pt"),
        },
        "last": {
            "epoch": epochs[-1]["epoch"] if epochs else 0,
            "ckpt_path": str(run_dir / "checkpoints" / "last.pt"),
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f)

    # config_resolved.yaml
    config = f"""task:
  _target_: {task_target}
  name: regression
model:
  _target_: {model_target}
  in_channels: 1
  num_classes: 1
dataset:
  train:
    _target_: {dataset_target}
trainer:
  lr: {lr}
  epochs: {len(epochs)}
"""
    (run_dir / "config_resolved.yaml").write_text(config)

    return run_dir


def _make_epoch_records(
    n_epochs: int,
    start_mae: float = 10.0,
    improvement_per_epoch: float = 0.5,
    monitor_metric: str = "val/mae",
    monitor_mode: str = "min",
) -> list[dict]:
    """Generate synthetic epoch records with steady improvement."""
    records = []
    for i in range(n_epochs):
        val_mae = start_mae - (improvement_per_epoch * i)
        train_mae = val_mae - 0.5  # train slightly better than val
        records.append({
            "epoch": i,
            "global_step": (i + 1) * 100,
            "train": {"loss": 0.5 - (0.03 * i), "mae": train_mae},
            "val": {"loss": 0.4 - (0.02 * i), "mae": val_mae},
            "monitor": {
                "metric": monitor_metric,
                "mode": monitor_mode,
                "value": val_mae,
            },
            "data": None,
        })
    return records


# ─────────────────────────────────────────────────
# Tests: parse_run_dir
# ─────────────────────────────────────────────────


class TestParseRunDir:
    def test_parses_complete_run(self, tmp_path):
        epochs = _make_epoch_records(5)
        run_dir = _make_run_dir(tmp_path, "run_001", epochs, best_epoch=4, best_value=8.0)

        run = parse_run_dir(run_dir)

        assert run.task_name == "regression"
        assert run.model_name == "encoder_classifier"
        assert run.dataset_name == "GalaxyPPGDataset"
        assert run.lr == 0.0003
        assert run.n_epochs_completed == 5
        assert run.epochs_configured == 5
        assert run.best_epoch == 4
        assert run.best_value == 8.0
        assert run.monitor_metric == "val/mae"
        assert run.monitor_mode == "min"
        assert not run.failed

    def test_parses_with_overrides(self, tmp_path):
        epochs = _make_epoch_records(3)
        run_dir = _make_run_dir(tmp_path, "run_002", epochs, best_epoch=2, best_value=9.0)

        run = parse_run_dir(run_dir, overrides=["trainer.lr=0.001", "trainer.epochs=3"])

        assert run.overrides == ["trainer.lr=0.001", "trainer.epochs=3"]

    def test_handles_missing_summary(self, tmp_path):
        run_dir = tmp_path / "run_no_summary"
        run_dir.mkdir()
        (run_dir / "metrics.jsonl").write_text("")
        (run_dir / "config_resolved.yaml").write_text("trainer:\n  lr: 0.001\n  epochs: 10\n")

        run = parse_run_dir(run_dir)

        assert run.n_epochs_completed == 0
        assert run.failed

    def test_training_curve(self, tmp_path):
        epochs = _make_epoch_records(5, start_mae=10.0, improvement_per_epoch=1.0)
        run_dir = _make_run_dir(tmp_path, "run_curve", epochs, best_epoch=4, best_value=6.0)

        run = parse_run_dir(run_dir)
        curve = run.training_curve()

        assert len(curve) == 5
        assert curve[0] == (0, 10.0)
        assert curve[4] == (4, 6.0)

    def test_summary_for_agent_is_string(self, tmp_path):
        epochs = _make_epoch_records(3)
        run_dir = _make_run_dir(tmp_path, "run_summary", epochs, best_epoch=2, best_value=9.0)

        run = parse_run_dir(run_dir)
        summary = run.summary_for_agent()

        assert isinstance(summary, str)
        assert "regression" in summary
        assert "encoder_classifier" in summary


# ─────────────────────────────────────────────────
# Tests: FeedbackStore
# ─────────────────────────────────────────────────


class TestFeedbackStore:
    def test_add_and_rank_runs(self, tmp_path):
        store = FeedbackStore()

        # Run 1: baseline (mae=8.0)
        e1 = _make_epoch_records(5, start_mae=10.0, improvement_per_epoch=0.4)
        d1 = _make_run_dir(tmp_path, "run1", e1, best_epoch=4, best_value=8.0)
        store.add_run_dir(d1)

        # Run 2: better (mae=6.0)
        e2 = _make_epoch_records(5, start_mae=8.0, improvement_per_epoch=0.4)
        d2 = _make_run_dir(tmp_path, "run2", e2, best_epoch=4, best_value=6.0, lr=0.0001)
        store.add_run_dir(d2, overrides=["trainer.lr=0.0001"])

        assert store.n_runs == 2
        ranked = store.rank_runs()
        # For min mode, lower is better → run2 should be #1
        assert ranked[0].best_monitor_value == 6.0
        assert ranked[1].best_monitor_value == 8.0

    def test_best_run(self, tmp_path):
        store = FeedbackStore()

        e1 = _make_epoch_records(3, start_mae=10.0)
        d1 = _make_run_dir(tmp_path, "run_a", e1, best_epoch=2, best_value=9.0)
        store.add_run_dir(d1)

        e2 = _make_epoch_records(3, start_mae=7.0)
        d2 = _make_run_dir(tmp_path, "run_b", e2, best_epoch=2, best_value=6.0)
        store.add_run_dir(d2)

        best = store.best_run()
        assert best is not None
        assert best.best_monitor_value == 6.0

    def test_drift_detection_regression(self, tmp_path):
        """Drift should be detected when later runs are WORSE."""
        store = FeedbackStore()

        # Runs getting progressively worse (higher MAE for min metric)
        for i, (name, best_val) in enumerate([
            ("r1", 5.0),
            ("r2", 5.5),
            ("r3", 6.5),
            ("r4", 8.0),
        ]):
            e = _make_epoch_records(3, start_mae=best_val + 2)
            d = _make_run_dir(tmp_path, name, e, best_epoch=2, best_value=best_val)
            store.add_run_dir(d)

        report = store.detect_drift()
        assert report.drift_detected
        assert report.trend_slope > 0  # positive slope = getting worse for min metric
        assert report.recommendation in ("change_lr", "change_model")

    def test_no_drift_when_improving(self, tmp_path):
        """No drift when runs are getting better."""
        store = FeedbackStore()

        for name, best_val in [("r1", 10.0), ("r2", 8.0), ("r3", 6.0)]:
            e = _make_epoch_records(5, start_mae=best_val + 2)
            d = _make_run_dir(tmp_path, name, e, best_epoch=4, best_value=best_val)
            store.add_run_dir(d)

        report = store.detect_drift()
        assert not report.drift_detected

    def test_stagnation_detection(self, tmp_path):
        """Stagnation when multiple runs have nearly identical results."""
        store = FeedbackStore(stagnation_tolerance=0.01)

        for name, best_val in [("r1", 6.00), ("r2", 6.01), ("r3", 5.99)]:
            e = _make_epoch_records(5, start_mae=best_val + 2)
            d = _make_run_dir(tmp_path, name, e, best_epoch=4, best_value=best_val)
            store.add_run_dir(d)

        report = store.detect_drift()
        assert report.stagnation_detected

    def test_history_for_agent(self, tmp_path):
        store = FeedbackStore()

        e1 = _make_epoch_records(3)
        d1 = _make_run_dir(tmp_path, "run_hist", e1, best_epoch=2, best_value=9.0)
        store.add_run_dir(d1)

        history = store.history_for_agent()
        assert isinstance(history, str)
        assert "Experiment History" in history
        assert "regression" in history

    def test_to_dict_serializable(self, tmp_path):
        store = FeedbackStore()

        e1 = _make_epoch_records(3)
        d1 = _make_run_dir(tmp_path, "run_ser", e1, best_epoch=2, best_value=9.0)
        store.add_run_dir(d1)

        d = store.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert d["n_runs"] == 1

    def test_insufficient_history_recommendation(self, tmp_path):
        """With < 2 runs, should recommend 'continue'."""
        store = FeedbackStore()

        e1 = _make_epoch_records(3)
        d1 = _make_run_dir(tmp_path, "only_run", e1, best_epoch=2, best_value=9.0)
        store.add_run_dir(d1)

        report = store.detect_drift()
        assert not report.drift_detected
        assert not report.stagnation_detected
        assert report.recommendation == "continue"