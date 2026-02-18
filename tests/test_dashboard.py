# tests/test_dashboard.py
"""
Unit tests for campaign dashboard data export and HTML generation.
No API key needed — creates mock runs and verifies export structure.

Run:
    PYTHONPATH=$PWD/src python -m pytest tests/test_dashboard.py -v
"""
import json
from pathlib import Path

import pytest

from biosignals.agent.feedback import FeedbackStore, EpochRecord, RunRecord
from biosignals.agent.hooks import AutoApproveHook, ApprovalDecision, CallbackApprovalHook
from biosignals.agent.dashboard import export_campaign_data, generate_dashboard


# ─────────────────────────────────────────────────
# Helpers: create mock runs
# ─────────────────────────────────────────────────


def _make_run(
    run_dir: str = "/tmp/run1",
    model_name: str = "ResNet1D",
    lr: float = 0.0003,
    epochs: int = 10,
    best_value: float = 0.5,
    best_epoch: int = 8,
    monitor_metric: str = "val/mae",
    monitor_mode: str = "min",
    converged: bool = False,
    failed: bool = False,
    overrides: list = None,
) -> RunRecord:
    """Build a mock RunRecord with synthetic training curve."""
    import math
    import random

    history = []
    if not failed:
        for e in range(epochs):
            # Simulate decaying loss
            base = best_value + (1.0 - best_value) * math.exp(-0.3 * e)
            noise = random.uniform(-0.02, 0.02)
            val = max(0.01, base + noise)

            history.append(EpochRecord(
                epoch=e,
                global_step=e * 100,
                train={"loss": val * 0.9, "mae": val * 0.85},
                val={"loss": val, "mae": val},
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                monitor_value=val,
            ))

    return RunRecord(
        run_dir=run_dir,
        timestamp="2025-02-17T10:00:00",
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        best_epoch=best_epoch,
        best_value=best_value,
        best_ckpt_path=f"{run_dir}/checkpoints/best.pt",
        last_epoch=epochs - 1,
        last_ckpt_path=f"{run_dir}/checkpoints/last.pt",
        epoch_history=history,
        task_name="RegressionTask",
        model_name=model_name,
        dataset_name="GalaxyPPGDataset",
        lr=lr,
        epochs_configured=epochs,
        overrides=overrides or [],
        config={},
    )


def _make_campaign_store() -> FeedbackStore:
    """Create a realistic 5-run campaign."""
    store = FeedbackStore()
    store.add_run(_make_run(
        run_dir="/tmp/run1", model_name="EncoderClassifier",
        lr=0.0003, epochs=5, best_value=50.73, best_epoch=4,
        overrides=["experiment=galaxyppg_hr_ppg", "trainer=fast_dev"],
    ))
    store.add_run(_make_run(
        run_dir="/tmp/run2", model_name="EncoderClassifier",
        lr=0.001, epochs=10, best_value=10.94, best_epoch=9,
        overrides=["experiment=galaxyppg_hr_ppg", "trainer.lr=0.001", "trainer.epochs=10"],
    ))
    store.add_run(_make_run(
        run_dir="/tmp/run3", model_name="Transformer1D",
        lr=0.001, epochs=10, best_value=15.22, best_epoch=7,
        overrides=["model=transformer1d", "trainer.lr=0.001"],
    ))
    store.add_run(_make_run(
        run_dir="/tmp/run4", model_name="EncoderClassifier",
        lr=0.0005, epochs=20, best_value=8.31, best_epoch=17,
        converged=True,
        overrides=["experiment=galaxyppg_hr_ppg", "trainer.lr=0.0005", "trainer.epochs=20"],
    ))
    store.add_run(_make_run(
        run_dir="/tmp/run5", model_name="ResNet1DDeep",
        lr=0.0005, epochs=20, best_value=7.12, best_epoch=18,
        overrides=["model=resnet1d_deep", "trainer.lr=0.0005", "trainer.epochs=20"],
    ))
    return store


# ═════════════════════════════════════════════════
# Tests: Data Export
# ═════════════════════════════════════════════════


class TestExportCampaignData:
    def test_basic_export(self):
        store = _make_campaign_store()
        data = export_campaign_data(store, campaign_goal="Minimize val/mae")

        assert "meta" in data
        assert "drift" in data
        assert "runs" in data
        assert "approvals" in data
        assert data["meta"]["n_runs"] == 5
        assert data["meta"]["goal"] == "Minimize val/mae"

    def test_run_details(self):
        store = _make_campaign_store()
        data = export_campaign_data(store)

        runs = data["runs"]
        assert len(runs) == 5
        assert runs[0]["model"] == "EncoderClassifier"
        assert runs[0]["lr"] == 0.0003
        assert "training_curve" in runs[0]
        assert len(runs[0]["training_curve"]) == 5  # 5 epochs
        assert runs[0]["training_curve"][0]["epoch"] == 0

    def test_includes_approval_history(self):
        store = _make_campaign_store()
        hook = AutoApproveHook()
        # Simulate 3 approvals
        hook("run1", store)
        hook("run2", store)
        hook("run3", store)

        data = export_campaign_data(store, approval_hook=hook)
        assert len(data["approvals"]) == 3
        assert data["approvals"][0]["run_number"] == 1

    def test_drift_info(self):
        store = _make_campaign_store()
        data = export_campaign_data(store)

        drift = data["drift"]
        assert "detected" in drift
        assert "stagnation" in drift
        assert "recommendation" in drift
        assert "trend_slope" in drift

    def test_json_serializable(self):
        store = _make_campaign_store()
        data = export_campaign_data(store, campaign_goal="test", agent_summary="summary")
        # Should not raise
        json_str = json.dumps(data, default=str)
        assert len(json_str) > 100


# ═════════════════════════════════════════════════
# Tests: HTML Generation
# ═════════════════════════════════════════════════


class TestGenerateDashboard:
    def test_creates_html_file(self, tmp_path):
        store = _make_campaign_store()
        html_path = generate_dashboard(
            store,
            output_dir=tmp_path,
            campaign_goal="Minimize val/mae",
            agent_summary="Best run achieved 7.12 MAE",
            open_browser=False,
        )

        assert html_path.exists()
        assert html_path.suffix == ".html"

        content = html_path.read_text()
        assert "Experiment Campaign Dashboard" in content
        assert "Chart" in content  # Chart.js reference
        assert "EncoderClassifier" in content  # data embedded

    def test_creates_json_file(self, tmp_path):
        store = _make_campaign_store()
        generate_dashboard(
            store, output_dir=tmp_path,
            campaign_goal="test", open_browser=False,
        )

        json_path = tmp_path / "campaign_data.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["meta"]["n_runs"] == 5

    def test_includes_approval_data(self, tmp_path):
        store = _make_campaign_store()
        hook = AutoApproveHook()
        hook("run1", store)
        hook("run2", store)

        html_path = generate_dashboard(
            store, approval_hook=hook,
            output_dir=tmp_path, open_browser=False,
        )

        content = html_path.read_text()
        assert "Approvals" in content  # section title in HTML

    def test_empty_store(self, tmp_path):
        store = FeedbackStore()
        html_path = generate_dashboard(
            store, output_dir=tmp_path,
            campaign_goal="empty test", open_browser=False,
        )

        assert html_path.exists()
        content = html_path.read_text()
        assert "Experiment Campaign Dashboard" in content
