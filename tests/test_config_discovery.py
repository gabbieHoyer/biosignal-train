# tests/test_config_discovery.py
"""
Unit tests for Tier 1 config discovery tools.
No API key needed — tests config scanning against a mock config directory.

Run from project root:
    PYTHONPATH=$PWD/src python -m pytest tests/test_config_discovery.py -v
"""
import tempfile
from pathlib import Path

import pytest

from biosignals.agent.tools import (
    _find_config_root,
    _summarize_yaml_config,
    _read_yaml_safe,
)


# ─────────────────────────────────────────────────
# Fixtures: mock config directory
# ─────────────────────────────────────────────────


@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a mock Hydra config directory matching real project structure."""
    configs = tmp_path / "configs"
    configs.mkdir()

    # model/
    model_dir = configs / "model"
    model_dir.mkdir()
    (model_dir / "resnet1d.yaml").write_text(
        "_target_: biosignals.models.resnet.ResNet1D\n"
        "in_channels: 1\nnum_classes: 5\nbase_filters: 64\n"
    )
    (model_dir / "transformer1d.yaml").write_text(
        "_target_: biosignals.models.transformer.Transformer1D\n"
        "d_model: 128\nn_heads: 4\nn_layers: 3\n"
    )
    (model_dir / "encoder_classifier.yaml").write_text(
        "_target_: biosignals.models.architectures.EncoderClassifier\n"
        "in_channels: 1\nnum_classes: 1\n"
    )

    # experiment/
    exp_dir = configs / "experiment"
    exp_dir.mkdir()
    (exp_dir / "galaxyppg_hr_ppg.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - override /dataset: galaxyppg\n"
        "  - override /task: regression\n"
        "  - override /model: encoder_classifier\n"
        "  - override /trainer: fast_dev\n"
        "  - _self_\n"
    )
    (exp_dir / "ecg_arrhythmia.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - override /dataset: mitbih\n"
        "  - override /task: classification\n"
        "  - override /model: resnet1d\n"
        "  - _self_\n"
    )

    # trainer/
    trainer_dir = configs / "trainer"
    trainer_dir.mkdir()
    (trainer_dir / "default.yaml").write_text(
        "lr: 0.0003\nepochs: 20\nbatch_size: 64\namp: true\n"
    )
    (trainer_dir / "fast_dev.yaml").write_text(
        "lr: 0.0003\nepochs: 2\nbatch_size: 32\namp: false\n"
    )
    (trainer_dir / "long.yaml").write_text(
        "lr: 0.0001\nepochs: 100\nbatch_size: 64\namp: true\n"
    )

    # dataset/
    ds_dir = configs / "dataset"
    ds_dir.mkdir()
    (ds_dir / "ecg_npz.yaml").write_text(
        "_target_: biosignals.data.datasets.EcgNpzDataset\n"
    )
    (ds_dir / "galaxyppg.yaml").write_text(
        "_target_: biosignals.data.datasets.GalaxyPPGDataset\n"
    )

    # task/
    task_dir = configs / "task"
    task_dir.mkdir()
    (task_dir / "classification.yaml").write_text(
        "_target_: biosignals.tasks.ClassificationTask\n"
    )
    (task_dir / "regression.yaml").write_text(
        "_target_: biosignals.tasks.RegressionTask\n"
    )

    # transforms/ (nested)
    tr_dir = configs / "transforms"
    tr_dir.mkdir()
    ecg_tr = tr_dir / "ecg"
    ecg_tr.mkdir()
    (ecg_tr / "basic.yaml").write_text("_target_: biosignals.transforms.EcgBasic\n")
    ppg_tr = tr_dir / "galaxyppg"
    ppg_tr.mkdir()
    (ppg_tr / "hr_ppg.yaml").write_text("_target_: biosignals.transforms.HrPpg\n")

    return tmp_path


# ─────────────────────────────────────────────────
# Tests: config file parsing
# ─────────────────────────────────────────────────


class TestYamlParsing:
    def test_read_yaml_safe(self, mock_config_dir):
        model_yaml = mock_config_dir / "configs" / "model" / "resnet1d.yaml"
        data = _read_yaml_safe(model_yaml)
        assert data["_target_"] == "biosignals.models.resnet.ResNet1D"
        assert data["base_filters"] == 64

    def test_summarize_model_config(self, mock_config_dir):
        model_yaml = mock_config_dir / "configs" / "model" / "resnet1d.yaml"
        summary = _summarize_yaml_config(model_yaml, "model")
        assert "ResNet1D" in summary
        assert "resnet1d" in summary

    def test_summarize_experiment_config(self, mock_config_dir):
        exp_yaml = mock_config_dir / "configs" / "experiment" / "galaxyppg_hr_ppg.yaml"
        summary = _summarize_yaml_config(exp_yaml, "experiment")
        assert "galaxyppg_hr_ppg" in summary
        # Should mention what it overrides
        assert "galaxyppg" in summary or "override" in summary.lower() or "dataset" in summary

    def test_summarize_trainer_config(self, mock_config_dir):
        trainer_yaml = mock_config_dir / "configs" / "trainer" / "default.yaml"
        summary = _summarize_yaml_config(trainer_yaml, "trainer")
        assert "default" in summary
        assert "lr" in summary or "epochs" in summary


# ─────────────────────────────────────────────────
# Tests: config root discovery
# ─────────────────────────────────────────────────


class TestConfigRootDiscovery:
    def test_finds_configs_dir(self, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)
        root = _find_config_root()
        assert root is not None
        assert root.name == "configs"
        assert (root / "model").is_dir()

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty_project"
        empty.mkdir()
        monkeypatch.chdir(empty)
        root = _find_config_root()
        assert root is None


# ─────────────────────────────────────────────────
# Tests: list_available_configs tool (direct call)
# ─────────────────────────────────────────────────


class TestListAvailableConfigs:
    def test_lists_all_groups(self, mock_config_dir, monkeypatch):
        """Verify the tool discovers all config groups."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import list_available_configs
        # smolagents @tool creates a SimpleTool; .forward() calls the function
        result = list_available_configs.forward()

        assert "model/" in result
        assert "experiment/" in result
        assert "trainer/" in result
        assert "dataset/" in result
        assert "task/" in result
        assert "transforms/" in result

    def test_shows_model_names(self, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)
        from biosignals.agent.tools import list_available_configs
        result = list_available_configs.forward()

        assert "resnet1d" in result
        assert "transformer1d" in result
        assert "encoder_classifier" in result

    def test_shows_experiment_names(self, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)
        from biosignals.agent.tools import list_available_configs
        result = list_available_configs.forward()

        assert "galaxyppg_hr_ppg" in result
        assert "ecg_arrhythmia" in result

    def test_shows_nested_transforms(self, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)
        from biosignals.agent.tools import list_available_configs
        result = list_available_configs.forward()

        assert "ecg/basic" in result
        assert "galaxyppg/hr_ppg" in result

    def test_handles_missing_config_dir(self, tmp_path, monkeypatch):
        empty = tmp_path / "no_configs"
        empty.mkdir()
        monkeypatch.chdir(empty)
        from biosignals.agent.tools import list_available_configs
        result = list_available_configs.forward()
        assert "ERROR" in result


# ─────────────────────────────────────────────────
# Tests: suggest_search_space tool
# ─────────────────────────────────────────────────


class TestSuggestSearchSpace:
    def _make_run(self, tmp_path, mock_config_dir, best_val=8.0, lr=0.0003, epochs=5):
        """Create a mock run directory for suggestion testing."""
        import json
        run_dir = tmp_path / "run_for_suggest"
        run_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i in range(epochs):
            val = best_val + (epochs - i) * 0.5
            records.append({
                "epoch": i, "global_step": (i + 1) * 100,
                "train": {"loss": val - 0.5, "mae": val - 1},
                "val": {"loss": val, "mae": val},
                "monitor": {"metric": "val/mae", "mode": "min", "value": val},
            })

        with open(run_dir / "metrics.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        import json as j
        (run_dir / "summary.json").write_text(j.dumps({
            "monitor": {"metric": "val/mae", "mode": "min"},
            "best": {"epoch": epochs - 1, "value": best_val, "ckpt_path": "best.pt"},
            "last": {"epoch": epochs - 1, "ckpt_path": "last.pt"},
        }))

        (run_dir / "config_resolved.yaml").write_text(
            f"task:\n  _target_: biosignals.tasks.RegressionTask\n  name: regression\n"
            f"model:\n  _target_: biosignals.models.architectures.EncoderClassifier\n"
            f"  in_channels: 1\n  num_classes: 1\n"
            f"dataset:\n  train:\n    _target_: biosignals.data.datasets.GalaxyPPGDataset\n"
            f"trainer:\n  lr: {lr}\n  epochs: {epochs}\n"
        )
        return run_dir

    def test_suggests_alternative_models(self, tmp_path, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)
        run_dir = self._make_run(tmp_path, mock_config_dir)

        from biosignals.agent.tools import suggest_search_space
        result = suggest_search_space.forward(run_dir=str(run_dir))

        # Should suggest trying other models
        assert "model=" in result
        assert "resnet1d" in result or "transformer1d" in result

    def test_suggests_lr_variations(self, tmp_path, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)
        run_dir = self._make_run(tmp_path, mock_config_dir, lr=0.0003)

        from biosignals.agent.tools import suggest_search_space
        result = suggest_search_space.forward(run_dir=str(run_dir))

        assert "trainer.lr=" in result

    def test_suggests_more_epochs_when_not_converged(self, tmp_path, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)
        run_dir = self._make_run(tmp_path, mock_config_dir, epochs=3)

        from biosignals.agent.tools import suggest_search_space
        result = suggest_search_space.forward(run_dir=str(run_dir))

        assert "epochs" in result.lower()

    def test_shows_available_trainers(self, tmp_path, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)
        run_dir = self._make_run(tmp_path, mock_config_dir)

        from biosignals.agent.tools import suggest_search_space
        result = suggest_search_space.forward(run_dir=str(run_dir))

        assert "default" in result or "fast_dev" in result or "long" in result
