# tests/test_config_generation.py
"""
Unit tests for Tier 2 config-generation tools.
No API key needed — tests config writing, safety validation,
and experiment composition against mock config directories.

Run from project root:
    PYTHONPATH=$PWD/src python -m pytest tests/test_config_generation.py -v
"""

import json
import textwrap

import pytest

from biosignals.agent.tools import (
    _AGENT_GENERATED_MARKER,
    ConfigSafetyError,
    _parse_modifications,
    _read_yaml_safe,
    _validate_config_name,
    _validate_group,
    _validate_no_code_injection,
)

# ─────────────────────────────────────────────────
# Fixture: mock config directory (reused from Tier 1)
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
        "d_model: 128\nn_heads: 4\nn_layers: 3\ndropout: 0.1\n"
    )
    (model_dir / "encoder_classifier.yaml").write_text(
        "_target_: biosignals.models.architectures.EncoderClassifier\n"
        "in_channels: 1\nnum_classes: 1\n"
    )

    # experiment/
    exp_dir = configs / "experiment"
    exp_dir.mkdir()
    (exp_dir / "galaxyppg_hr_ppg.yaml").write_text(
        textwrap.dedent("""\
        # @package _global_
        defaults:
          - override /dataset: galaxyppg
          - override /task: regression
          - override /model: encoder_classifier
          - override /trainer: fast_dev
          - override /transforms: galaxyppg/hr_ppg
          - _self_

        model:
          in_channels: 1
          num_classes: 1

        data:
          num_workers: 0

        trainer:
          monitor_metric: val/loss
          monitor_mode: min
    """)
    )

    # trainer/
    trainer_dir = configs / "trainer"
    trainer_dir.mkdir()
    (trainer_dir / "default.yaml").write_text(
        "_target_: biosignals.engine.trainer.TrainerConfig\n"
        "lr: 0.0003\nepochs: 20\nbatch_size: 64\namp: true\n"
    )
    (trainer_dir / "fast_dev.yaml").write_text(
        "_target_: biosignals.engine.trainer.TrainerConfig\n"
        "lr: 0.0003\nepochs: 2\nbatch_size: 32\namp: false\n"
    )

    # dataset/
    ds_dir = configs / "dataset"
    ds_dir.mkdir()
    (ds_dir / "ecg_npz.yaml").write_text("_target_: biosignals.data.datasets.EcgNpzDataset\n")
    (ds_dir / "galaxyppg.yaml").write_text("_target_: biosignals.data.datasets.GalaxyPPGDataset\n")

    # task/
    task_dir = configs / "task"
    task_dir.mkdir()
    (task_dir / "classification.yaml").write_text("_target_: biosignals.tasks.ClassificationTask\n")
    (task_dir / "regression.yaml").write_text("_target_: biosignals.tasks.RegressionTask\n")

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


# ═════════════════════════════════════════════════
# Tests: Safety Validation
# ═════════════════════════════════════════════════


class TestConfigNameValidation:
    def test_valid_names(self):
        _validate_config_name("transformer1d_8head")
        _validate_config_name("resnet1d-big")
        _validate_config_name("a")
        _validate_config_name("model_v2_final")

    def test_rejects_empty(self):
        with pytest.raises(ConfigSafetyError):
            _validate_config_name("")

    def test_rejects_starting_with_number(self):
        with pytest.raises(ConfigSafetyError):
            _validate_config_name("8head_transformer")

    def test_rejects_special_chars(self):
        with pytest.raises(ConfigSafetyError):
            _validate_config_name("model/variant")
        with pytest.raises(ConfigSafetyError):
            _validate_config_name("model..yaml")
        with pytest.raises(ConfigSafetyError):
            _validate_config_name("../../etc/passwd")

    def test_rejects_spaces(self):
        with pytest.raises(ConfigSafetyError):
            _validate_config_name("model variant")

    def test_rejects_too_long(self):
        with pytest.raises(ConfigSafetyError):
            _validate_config_name("a" * 65)


class TestGroupValidation:
    def test_writable_groups(self):
        for group in ["model", "trainer", "experiment", "transforms", "dataset", "task"]:
            _validate_group(group)  # Should not raise

    def test_rejects_hydra(self):
        with pytest.raises(ConfigSafetyError, match="hydra"):
            _validate_group("hydra")

    def test_rejects_logger(self):
        with pytest.raises(ConfigSafetyError, match="logger"):
            _validate_group("logger")

    def test_rejects_arbitrary(self):
        with pytest.raises(ConfigSafetyError):
            _validate_group("malicious_group")


class TestCodeInjectionDetection:
    def test_safe_config(self):
        data = {
            "_target_": "biosignals.models.resnet.ResNet1D",
            "in_channels": 1,
            "num_classes": 5,
        }
        _validate_no_code_injection(data)  # Should not raise

    def test_rejects_unsafe_target(self):
        data = {
            "_target_": "os.system",
            "command": "rm -rf /",
        }
        with pytest.raises(ConfigSafetyError, match="Unsafe _target_"):
            _validate_no_code_injection(data)

    def test_rejects_exec_in_values(self):
        data = {
            "_target_": "biosignals.models.resnet.ResNet1D",
            "name": "exec('malicious()')",
        }
        with pytest.raises(ConfigSafetyError, match="exec"):
            _validate_no_code_injection(data)

    def test_rejects_subprocess_in_nested(self):
        data = {
            "_target_": "biosignals.models.resnet.ResNet1D",
            "encoder": {
                "init_cmd": "subprocess.run(['rm', '-rf', '/'])",
            },
        }
        with pytest.raises(ConfigSafetyError, match="subprocess"):
            _validate_no_code_injection(data)

    def test_allows_torch_target(self):
        data = {"_target_": "torch.nn.Linear", "in_features": 128, "out_features": 64}
        _validate_no_code_injection(data)  # Should not raise

    def test_rejects_import_in_list(self):
        data = {
            "_target_": "biosignals.models.resnet.ResNet1D",
            "transforms": ["import os", "normalize"],
        }
        with pytest.raises(ConfigSafetyError, match="import"):
            _validate_no_code_injection(data)


# ═════════════════════════════════════════════════
# Tests: Modification Parsing
# ═════════════════════════════════════════════════


class TestParseModifications:
    def test_basic_types(self):
        result = _parse_modifications("n_heads=8 d_model=256 dropout=0.1 amp=true name=mymodel")
        assert result["n_heads"] == 8
        assert result["d_model"] == 256
        assert result["dropout"] == 0.1
        assert result["amp"] is True
        assert result["name"] == "mymodel"

    def test_dotted_paths(self):
        result = _parse_modifications("model.in_channels=1 trainer.lr=0.001")
        assert result["model"]["in_channels"] == 1
        assert result["trainer"]["lr"] == 0.001

    def test_false_and_null(self):
        result = _parse_modifications("amp=false bias=null")
        assert result["amp"] is False
        assert result["bias"] is None

    def test_empty_string(self):
        assert _parse_modifications("") == {}
        assert _parse_modifications("   ") == {}

    def test_ignores_tokens_without_equals(self):
        result = _parse_modifications("n_heads=8 garbage lr=0.001")
        assert result == {"n_heads": 8, "lr": 0.001}


# ═════════════════════════════════════════════════
# Tests: create_config_variant Tool
# ═════════════════════════════════════════════════


class TestCreateConfigVariant:
    def test_basic_variant(self, mock_config_dir, monkeypatch):
        """Create a transformer variant with different head count."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant

        result_json = create_config_variant.forward(
            group="model",
            base_config="transformer1d",
            new_name="transformer1d_8head",
            modifications="n_heads=8 d_model=256",
        )
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["override_syntax"] == "model=transformer1d_8head"

        # Verify the file was written
        new_path = mock_config_dir / "configs" / "model" / "transformer1d_8head.yaml"
        assert new_path.exists()

        # Verify content
        data = _read_yaml_safe(new_path)
        assert data["n_heads"] == 8
        assert data["d_model"] == 256
        # _target_ must be preserved from base
        assert data["_target_"] == "biosignals.models.transformer.Transformer1D"

        # Verify agent marker
        content = new_path.read_text()
        assert _AGENT_GENERATED_MARKER.strip() in content

    def test_preserves_target(self, mock_config_dir, monkeypatch):
        """Agent cannot change _target_ field."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant

        result_json = create_config_variant.forward(
            group="model",
            base_config="transformer1d",
            new_name="transformer1d_bad",
            modifications="_target_=os.system n_heads=8",
        )
        result = json.loads(result_json)

        assert result["success"] is False
        assert "Cannot modify _target_" in result["error"]

    def test_rejects_invalid_group(self, mock_config_dir, monkeypatch):
        """Cannot write to restricted groups like hydra/."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant

        result_json = create_config_variant.forward(
            group="hydra",
            base_config="default",
            new_name="hacked",
            modifications="job.name=evil",
        )
        result = json.loads(result_json)

        assert result["success"] is False
        assert "SAFETY" in result["error"]

    def test_rejects_bad_name(self, mock_config_dir, monkeypatch):
        """Config names must be safe filesystem identifiers."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant

        result_json = create_config_variant.forward(
            group="model",
            base_config="transformer1d",
            new_name="../../etc/passwd",
            modifications="n_heads=8",
        )
        result = json.loads(result_json)

        assert result["success"] is False
        assert "SAFETY" in result["error"]

    def test_rejects_nonexistent_base(self, mock_config_dir, monkeypatch):
        """Helpful error when base config doesn't exist."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant

        result_json = create_config_variant.forward(
            group="model",
            base_config="nonexistent_model",
            new_name="variant",
            modifications="n_heads=8",
        )
        result = json.loads(result_json)

        assert result["success"] is False
        assert "not found" in result["error"]
        # Should list available configs
        assert "transformer1d" in result["error"] or "resnet1d" in result["error"]

    def test_overwrites_agent_generated(self, mock_config_dir, monkeypatch):
        """Can overwrite a previously agent-generated config."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant

        # Create first version
        create_config_variant.forward(
            group="model",
            base_config="transformer1d",
            new_name="transformer1d_v1",
            modifications="n_heads=4",
        )

        # Overwrite with second version — should succeed
        result_json = create_config_variant.forward(
            group="model",
            base_config="transformer1d",
            new_name="transformer1d_v1",
            modifications="n_heads=16",
        )
        result = json.loads(result_json)
        assert result["success"] is True

        # Verify updated content
        data = _read_yaml_safe(mock_config_dir / "configs" / "model" / "transformer1d_v1.yaml")
        assert data["n_heads"] == 16

    def test_refuses_overwrite_human_config(self, mock_config_dir, monkeypatch):
        """Cannot overwrite a human-authored config."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant

        result_json = create_config_variant.forward(
            group="model",
            base_config="transformer1d",
            new_name="resnet1d",  # Already exists, human-authored
            modifications="n_heads=8",
        )
        result = json.loads(result_json)

        assert result["success"] is False
        assert "NOT agent-generated" in result["error"]

    def test_trainer_variant(self, mock_config_dir, monkeypatch):
        """Create a trainer variant with different hyperparameters."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant

        result_json = create_config_variant.forward(
            group="trainer",
            base_config="default",
            new_name="slow_convergence",
            modifications="lr=0.00005 epochs=100",
        )
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["override_syntax"] == "trainer=slow_convergence"

        data = _read_yaml_safe(mock_config_dir / "configs" / "trainer" / "slow_convergence.yaml")
        assert data["lr"] == 5e-05
        assert data["epochs"] == 100
        # Preserved from base
        assert data["_target_"] == "biosignals.engine.trainer.TrainerConfig"
        assert data["batch_size"] == 64  # Unchanged from base


# ═════════════════════════════════════════════════
# Tests: compose_experiment_config Tool
# ═════════════════════════════════════════════════


class TestComposeExperimentConfig:
    def test_basic_composition(self, mock_config_dir, monkeypatch):
        """Compose a new experiment from existing components."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config

        result_json = compose_experiment_config.forward(
            name="ppg_transformer_experiment",
            components="model=transformer1d dataset=galaxyppg task=regression trainer=default",
            extra_overrides="model.in_channels=1 model.num_classes=1 trainer.monitor_metric=val/loss trainer.monitor_mode=min",
        )
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["override_syntax"] == "experiment=ppg_transformer_experiment"

        # Verify file
        exp_path = mock_config_dir / "configs" / "experiment" / "ppg_transformer_experiment.yaml"
        assert exp_path.exists()

        content = exp_path.read_text()
        # Must have @package _global_
        assert "@package _global_" in content
        # Must have override defaults
        assert "override /model: transformer1d" in content
        assert "override /dataset: galaxyppg" in content
        assert "override /task: regression" in content
        assert "override /trainer: default" in content
        # Must have agent marker
        assert _AGENT_GENERATED_MARKER.strip() in content
        # Must have extra overrides
        assert "in_channels" in content
        assert "monitor_metric" in content

    def test_rejects_nonexistent_component(self, mock_config_dir, monkeypatch):
        """Helpful error when a component config doesn't exist."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config

        result_json = compose_experiment_config.forward(
            name="bad_experiment",
            components="model=nonexistent_model dataset=galaxyppg",
            extra_overrides="",
        )
        result = json.loads(result_json)

        assert result["success"] is False
        assert "not found" in result["error"]
        assert "transformer1d" in result["error"] or "resnet1d" in result["error"]

    def test_no_extra_overrides(self, mock_config_dir, monkeypatch):
        """Works with empty extra_overrides."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config

        result_json = compose_experiment_config.forward(
            name="simple_experiment",
            components="model=resnet1d dataset=ecg_npz task=classification",
            extra_overrides="",
        )
        result = json.loads(result_json)
        assert result["success"] is True

    def test_rejects_bad_name(self, mock_config_dir, monkeypatch):
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config

        result_json = compose_experiment_config.forward(
            name="../../../etc/passwd",
            components="model=resnet1d",
            extra_overrides="",
        )
        result = json.loads(result_json)
        assert result["success"] is False
        assert "SAFETY" in result["error"]

    def test_refuses_overwrite_human_experiment(self, mock_config_dir, monkeypatch):
        """Cannot overwrite a human-authored experiment."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config

        result_json = compose_experiment_config.forward(
            name="galaxyppg_hr_ppg",  # Already exists, human-authored
            components="model=resnet1d dataset=galaxyppg task=regression",
            extra_overrides="",
        )
        result = json.loads(result_json)
        assert result["success"] is False
        assert "NOT agent-generated" in result["error"]

    def test_overwrites_agent_generated_experiment(self, mock_config_dir, monkeypatch):
        """Can overwrite a previously agent-generated experiment."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config

        # Create first version
        compose_experiment_config.forward(
            name="iterative_exp",
            components="model=resnet1d dataset=ecg_npz task=classification",
            extra_overrides="",
        )

        # Overwrite — should succeed
        result_json = compose_experiment_config.forward(
            name="iterative_exp",
            components="model=transformer1d dataset=galaxyppg task=regression",
            extra_overrides="model.in_channels=1",
        )
        result = json.loads(result_json)
        assert result["success"] is True

        # Verify updated content
        content = (mock_config_dir / "configs" / "experiment" / "iterative_exp.yaml").read_text()
        assert "transformer1d" in content
        assert "galaxyppg" in content

    def test_with_nested_transforms(self, mock_config_dir, monkeypatch):
        """Handles nested config paths like transforms=galaxyppg/hr_ppg."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config

        result_json = compose_experiment_config.forward(
            name="full_ppg_experiment",
            components="model=encoder_classifier dataset=galaxyppg task=regression trainer=fast_dev transforms=galaxyppg/hr_ppg",
            extra_overrides="model.in_channels=1 model.num_classes=1",
        )
        result = json.loads(result_json)
        assert result["success"] is True

        content = (
            mock_config_dir / "configs" / "experiment" / "full_ppg_experiment.yaml"
        ).read_text()
        assert "override /transforms: galaxyppg/hr_ppg" in content

    def test_rejects_code_injection_in_overrides(self, mock_config_dir, monkeypatch):
        """Safety scan catches code injection in extra_overrides."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config

        result_json = compose_experiment_config.forward(
            name="safe_experiment",
            components="model=resnet1d dataset=ecg_npz task=classification",
            extra_overrides="_target_=os.system",
        )
        result = json.loads(result_json)
        assert result["success"] is False
        assert "SAFETY" in result["error"]


# ═════════════════════════════════════════════════
# Tests: End-to-end workflow
# ═════════════════════════════════════════════════


class TestEndToEndWorkflow:
    def test_create_variant_then_compose_experiment(self, mock_config_dir, monkeypatch):
        """
        Full workflow: create model variant → compose experiment → verify.
        This is exactly what the agent does in a real campaign.
        """
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import compose_experiment_config, create_config_variant

        # Step 1: Create a model variant
        r1 = json.loads(
            create_config_variant.forward(
                group="model",
                base_config="transformer1d",
                new_name="transformer1d_wide",
                modifications="d_model=512 n_heads=8 n_layers=6",
            )
        )
        assert r1["success"] is True

        # Step 2: Create a trainer variant
        r2 = json.loads(
            create_config_variant.forward(
                group="trainer",
                base_config="default",
                new_name="long_train",
                modifications="epochs=50 lr=0.0001",
            )
        )
        assert r2["success"] is True

        # Step 3: Compose an experiment using the new variants
        r3 = json.loads(
            compose_experiment_config.forward(
                name="ppg_wide_transformer_long",
                components="model=transformer1d_wide dataset=galaxyppg task=regression trainer=long_train transforms=galaxyppg/hr_ppg",
                extra_overrides="model.in_channels=1 model.num_classes=1 trainer.monitor_metric=val/mae trainer.monitor_mode=min",
            )
        )
        assert r3["success"] is True
        assert r3["override_syntax"] == "experiment=ppg_wide_transformer_long"

        # Step 4: Verify the experiment file references our variants
        exp_content = (
            mock_config_dir / "configs" / "experiment" / "ppg_wide_transformer_long.yaml"
        ).read_text()
        assert "override /model: transformer1d_wide" in exp_content
        assert "override /trainer: long_train" in exp_content
        assert "in_channels" in exp_content

        # Step 5: Verify the model variant has correct params
        model_data = _read_yaml_safe(
            mock_config_dir / "configs" / "model" / "transformer1d_wide.yaml"
        )
        assert model_data["d_model"] == 512
        assert model_data["n_heads"] == 8
        assert model_data["_target_"] == "biosignals.models.transformer.Transformer1D"

    def test_list_shows_agent_generated_configs(self, mock_config_dir, monkeypatch):
        """After creating configs, list_available_configs() shows them."""
        monkeypatch.chdir(mock_config_dir)

        from biosignals.agent.tools import create_config_variant, list_available_configs

        # Create a variant
        create_config_variant.forward(
            group="model",
            base_config="transformer1d",
            new_name="transformer1d_big",
            modifications="d_model=1024",
        )

        # Verify it appears in listing
        listing = list_available_configs.forward()
        assert "transformer1d_big" in listing


# ---------------------------------------------------------

# # tests/test_config_generation.py
# """
# Unit tests for Tier 2 config-generation tools.
# No API key needed — tests config writing, safety validation,
# and experiment composition against mock config directories.

# Run from project root:
#     PYTHONPATH=$PWD/src python -m pytest tests/test_config_generation.py -v
# """
# import json
# import textwrap
# from pathlib import Path

# import pytest

# from biosignals.agent.tools import (
#     _find_config_root,
#     _read_yaml_safe,
#     _parse_modifications,
#     _validate_config_name,
#     _validate_group,
#     _validate_no_code_injection,
#     _AGENT_GENERATED_MARKER,
#     ConfigSafetyError,
# )


# # ─────────────────────────────────────────────────
# # Fixture: mock config directory (reused from Tier 1)
# # ─────────────────────────────────────────────────


# @pytest.fixture
# def mock_config_dir(tmp_path):
#     """Create a mock Hydra config directory matching real project structure."""
#     configs = tmp_path / "configs"
#     configs.mkdir()

#     # model/
#     model_dir = configs / "model"
#     model_dir.mkdir()
#     (model_dir / "resnet1d.yaml").write_text(
#         "_target_: biosignals.models.resnet.ResNet1D\n"
#         "in_channels: 1\nnum_classes: 5\nbase_filters: 64\n"
#     )
#     (model_dir / "transformer1d.yaml").write_text(
#         "_target_: biosignals.models.transformer.Transformer1D\n"
#         "d_model: 128\nn_heads: 4\nn_layers: 3\ndropout: 0.1\n"
#     )
#     (model_dir / "encoder_classifier.yaml").write_text(
#         "_target_: biosignals.models.architectures.EncoderClassifier\n"
#         "in_channels: 1\nnum_classes: 1\n"
#     )

#     # experiment/
#     exp_dir = configs / "experiment"
#     exp_dir.mkdir()
#     (exp_dir / "galaxyppg_hr_ppg.yaml").write_text(textwrap.dedent("""\
#         # @package _global_
#         defaults:
#           - override /dataset: galaxyppg
#           - override /task: regression
#           - override /model: encoder_classifier
#           - override /trainer: fast_dev
#           - override /transforms: galaxyppg/hr_ppg
#           - _self_

#         model:
#           in_channels: 1
#           num_classes: 1

#         data:
#           num_workers: 0

#         trainer:
#           monitor_metric: val/loss
#           monitor_mode: min
#     """))

#     # trainer/
#     trainer_dir = configs / "trainer"
#     trainer_dir.mkdir()
#     (trainer_dir / "default.yaml").write_text(
#         "_target_: biosignals.engine.trainer.TrainerConfig\n"
#         "lr: 0.0003\nepochs: 20\nbatch_size: 64\namp: true\n"
#     )
#     (trainer_dir / "fast_dev.yaml").write_text(
#         "_target_: biosignals.engine.trainer.TrainerConfig\n"
#         "lr: 0.0003\nepochs: 2\nbatch_size: 32\namp: false\n"
#     )

#     # dataset/
#     ds_dir = configs / "dataset"
#     ds_dir.mkdir()
#     (ds_dir / "ecg_npz.yaml").write_text(
#         "_target_: biosignals.data.datasets.EcgNpzDataset\n"
#     )
#     (ds_dir / "galaxyppg.yaml").write_text(
#         "_target_: biosignals.data.datasets.GalaxyPPGDataset\n"
#     )

#     # task/
#     task_dir = configs / "task"
#     task_dir.mkdir()
#     (task_dir / "classification.yaml").write_text(
#         "_target_: biosignals.tasks.ClassificationTask\n"
#     )
#     (task_dir / "regression.yaml").write_text(
#         "_target_: biosignals.tasks.RegressionTask\n"
#     )

#     # transforms/ (nested)
#     tr_dir = configs / "transforms"
#     tr_dir.mkdir()
#     ecg_tr = tr_dir / "ecg"
#     ecg_tr.mkdir()
#     (ecg_tr / "basic.yaml").write_text(
#         "_target_: biosignals.transforms.EcgBasic\n"
#     )
#     ppg_tr = tr_dir / "galaxyppg"
#     ppg_tr.mkdir()
#     (ppg_tr / "hr_ppg.yaml").write_text(
#         "_target_: biosignals.transforms.HrPpg\n"
#     )

#     return tmp_path


# # ═════════════════════════════════════════════════
# # Tests: Safety Validation
# # ═════════════════════════════════════════════════


# class TestConfigNameValidation:
#     def test_valid_names(self):
#         _validate_config_name("transformer1d_8head")
#         _validate_config_name("resnet1d-big")
#         _validate_config_name("a")
#         _validate_config_name("model_v2_final")

#     def test_rejects_empty(self):
#         with pytest.raises(ConfigSafetyError):
#             _validate_config_name("")

#     def test_rejects_starting_with_number(self):
#         with pytest.raises(ConfigSafetyError):
#             _validate_config_name("8head_transformer")

#     def test_rejects_special_chars(self):
#         with pytest.raises(ConfigSafetyError):
#             _validate_config_name("model/variant")
#         with pytest.raises(ConfigSafetyError):
#             _validate_config_name("model..yaml")
#         with pytest.raises(ConfigSafetyError):
#             _validate_config_name("../../etc/passwd")

#     def test_rejects_spaces(self):
#         with pytest.raises(ConfigSafetyError):
#             _validate_config_name("model variant")

#     def test_rejects_too_long(self):
#         with pytest.raises(ConfigSafetyError):
#             _validate_config_name("a" * 65)


# class TestGroupValidation:
#     def test_writable_groups(self):
#         for group in ["model", "trainer", "experiment", "transforms", "dataset", "task"]:
#             _validate_group(group)  # Should not raise

#     def test_rejects_hydra(self):
#         with pytest.raises(ConfigSafetyError, match="hydra"):
#             _validate_group("hydra")

#     def test_rejects_logger(self):
#         with pytest.raises(ConfigSafetyError, match="logger"):
#             _validate_group("logger")

#     def test_rejects_arbitrary(self):
#         with pytest.raises(ConfigSafetyError):
#             _validate_group("malicious_group")


# class TestCodeInjectionDetection:
#     def test_safe_config(self):
#         data = {
#             "_target_": "biosignals.models.resnet.ResNet1D",
#             "in_channels": 1,
#             "num_classes": 5,
#         }
#         _validate_no_code_injection(data)  # Should not raise

#     def test_rejects_unsafe_target(self):
#         data = {
#             "_target_": "os.system",
#             "command": "rm -rf /",
#         }
#         with pytest.raises(ConfigSafetyError, match="Unsafe _target_"):
#             _validate_no_code_injection(data)

#     def test_rejects_exec_in_values(self):
#         data = {
#             "_target_": "biosignals.models.resnet.ResNet1D",
#             "name": "exec('import os')",
#         }
#         with pytest.raises(ConfigSafetyError, match="exec"):
#             _validate_no_code_injection(data)

#     def test_rejects_subprocess_in_nested(self):
#         data = {
#             "_target_": "biosignals.models.resnet.ResNet1D",
#             "encoder": {
#                 "init_cmd": "subprocess.run(['rm', '-rf', '/'])",
#             },
#         }
#         with pytest.raises(ConfigSafetyError, match="subprocess"):
#             _validate_no_code_injection(data)

#     def test_allows_torch_target(self):
#         data = {"_target_": "torch.nn.Linear", "in_features": 128, "out_features": 64}
#         _validate_no_code_injection(data)  # Should not raise

#     def test_rejects_import_in_list(self):
#         data = {
#             "_target_": "biosignals.models.resnet.ResNet1D",
#             "transforms": ["import os", "normalize"],
#         }
#         with pytest.raises(ConfigSafetyError, match="import"):
#             _validate_no_code_injection(data)


# # ═════════════════════════════════════════════════
# # Tests: Modification Parsing
# # ═════════════════════════════════════════════════


# class TestParseModifications:
#     def test_basic_types(self):
#         result = _parse_modifications("n_heads=8 d_model=256 dropout=0.1 amp=true name=mymodel")
#         assert result["n_heads"] == 8
#         assert result["d_model"] == 256
#         assert result["dropout"] == 0.1
#         assert result["amp"] is True
#         assert result["name"] == "mymodel"

#     def test_dotted_paths(self):
#         result = _parse_modifications("model.in_channels=1 trainer.lr=0.001")
#         assert result["model"]["in_channels"] == 1
#         assert result["trainer"]["lr"] == 0.001

#     def test_false_and_null(self):
#         result = _parse_modifications("amp=false bias=null")
#         assert result["amp"] is False
#         assert result["bias"] is None

#     def test_empty_string(self):
#         assert _parse_modifications("") == {}
#         assert _parse_modifications("   ") == {}

#     def test_ignores_tokens_without_equals(self):
#         result = _parse_modifications("n_heads=8 garbage lr=0.001")
#         assert result == {"n_heads": 8, "lr": 0.001}


# # ═════════════════════════════════════════════════
# # Tests: create_config_variant Tool
# # ═════════════════════════════════════════════════


# class TestCreateConfigVariant:
#     def test_basic_variant(self, mock_config_dir, monkeypatch):
#         """Create a transformer variant with different head count."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant
#         result_json = create_config_variant.forward(
#             group="model",
#             base_config="transformer1d",
#             new_name="transformer1d_8head",
#             modifications="n_heads=8 d_model=256",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is True
#         assert result["override_syntax"] == "model=transformer1d_8head"

#         # Verify the file was written
#         new_path = mock_config_dir / "configs" / "model" / "transformer1d_8head.yaml"
#         assert new_path.exists()

#         # Verify content
#         data = _read_yaml_safe(new_path)
#         assert data["n_heads"] == 8
#         assert data["d_model"] == 256
#         # _target_ must be preserved from base
#         assert data["_target_"] == "biosignals.models.transformer.Transformer1D"

#         # Verify agent marker
#         content = new_path.read_text()
#         assert _AGENT_GENERATED_MARKER.strip() in content

#     def test_preserves_target(self, mock_config_dir, monkeypatch):
#         """Agent cannot change _target_ field."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant
#         result_json = create_config_variant.forward(
#             group="model",
#             base_config="transformer1d",
#             new_name="transformer1d_bad",
#             modifications="_target_=os.system n_heads=8",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is False
#         assert "Cannot modify _target_" in result["error"]

#     def test_rejects_invalid_group(self, mock_config_dir, monkeypatch):
#         """Cannot write to restricted groups like hydra/."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant
#         result_json = create_config_variant.forward(
#             group="hydra",
#             base_config="default",
#             new_name="hacked",
#             modifications="job.name=evil",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is False
#         assert "SAFETY" in result["error"]

#     def test_rejects_bad_name(self, mock_config_dir, monkeypatch):
#         """Config names must be safe filesystem identifiers."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant
#         result_json = create_config_variant.forward(
#             group="model",
#             base_config="transformer1d",
#             new_name="../../etc/passwd",
#             modifications="n_heads=8",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is False
#         assert "SAFETY" in result["error"]

#     def test_rejects_nonexistent_base(self, mock_config_dir, monkeypatch):
#         """Helpful error when base config doesn't exist."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant
#         result_json = create_config_variant.forward(
#             group="model",
#             base_config="nonexistent_model",
#             new_name="variant",
#             modifications="n_heads=8",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is False
#         assert "not found" in result["error"]
#         # Should list available configs
#         assert "transformer1d" in result["error"] or "resnet1d" in result["error"]

#     def test_overwrites_agent_generated(self, mock_config_dir, monkeypatch):
#         """Can overwrite a previously agent-generated config."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant

#         # Create first version
#         create_config_variant.forward(
#             group="model", base_config="transformer1d",
#             new_name="transformer1d_v1", modifications="n_heads=4",
#         )

#         # Overwrite with second version — should succeed
#         result_json = create_config_variant.forward(
#             group="model", base_config="transformer1d",
#             new_name="transformer1d_v1", modifications="n_heads=16",
#         )
#         result = json.loads(result_json)
#         assert result["success"] is True

#         # Verify updated content
#         data = _read_yaml_safe(mock_config_dir / "configs" / "model" / "transformer1d_v1.yaml")
#         assert data["n_heads"] == 16

#     def test_refuses_overwrite_human_config(self, mock_config_dir, monkeypatch):
#         """Cannot overwrite a human-authored config."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant
#         result_json = create_config_variant.forward(
#             group="model",
#             base_config="transformer1d",
#             new_name="resnet1d",  # Already exists, human-authored
#             modifications="n_heads=8",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is False
#         assert "NOT agent-generated" in result["error"]

#     def test_trainer_variant(self, mock_config_dir, monkeypatch):
#         """Create a trainer variant with different hyperparameters."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant
#         result_json = create_config_variant.forward(
#             group="trainer",
#             base_config="default",
#             new_name="slow_convergence",
#             modifications="lr=0.00005 epochs=100",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is True
#         assert result["override_syntax"] == "trainer=slow_convergence"

#         data = _read_yaml_safe(mock_config_dir / "configs" / "trainer" / "slow_convergence.yaml")
#         assert data["lr"] == 5e-05
#         assert data["epochs"] == 100
#         # Preserved from base
#         assert data["_target_"] == "biosignals.engine.trainer.TrainerConfig"
#         assert data["batch_size"] == 64  # Unchanged from base


# # ═════════════════════════════════════════════════
# # Tests: compose_experiment_config Tool
# # ═════════════════════════════════════════════════


# class TestComposeExperimentConfig:
#     def test_basic_composition(self, mock_config_dir, monkeypatch):
#         """Compose a new experiment from existing components."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import compose_experiment_config
#         result_json = compose_experiment_config.forward(
#             name="ppg_transformer_experiment",
#             components="model=transformer1d dataset=galaxyppg task=regression trainer=default",
#             extra_overrides="model.in_channels=1 model.num_classes=1 trainer.monitor_metric=val/loss trainer.monitor_mode=min",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is True
#         assert result["override_syntax"] == "experiment=ppg_transformer_experiment"

#         # Verify file
#         exp_path = mock_config_dir / "configs" / "experiment" / "ppg_transformer_experiment.yaml"
#         assert exp_path.exists()

#         content = exp_path.read_text()
#         # Must have @package _global_
#         assert "@package _global_" in content
#         # Must have override defaults
#         assert "override /model: transformer1d" in content
#         assert "override /dataset: galaxyppg" in content
#         assert "override /task: regression" in content
#         assert "override /trainer: default" in content
#         # Must have agent marker
#         assert _AGENT_GENERATED_MARKER.strip() in content
#         # Must have extra overrides
#         assert "in_channels" in content
#         assert "monitor_metric" in content

#     def test_rejects_nonexistent_component(self, mock_config_dir, monkeypatch):
#         """Helpful error when a component config doesn't exist."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import compose_experiment_config
#         result_json = compose_experiment_config.forward(
#             name="bad_experiment",
#             components="model=nonexistent_model dataset=galaxyppg",
#             extra_overrides="",
#         )
#         result = json.loads(result_json)

#         assert result["success"] is False
#         assert "not found" in result["error"]
#         assert "transformer1d" in result["error"] or "resnet1d" in result["error"]

#     def test_no_extra_overrides(self, mock_config_dir, monkeypatch):
#         """Works with empty extra_overrides."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import compose_experiment_config
#         result_json = compose_experiment_config.forward(
#             name="simple_experiment",
#             components="model=resnet1d dataset=ecg_npz task=classification",
#             extra_overrides="",
#         )
#         result = json.loads(result_json)
#         assert result["success"] is True

#     def test_rejects_bad_name(self, mock_config_dir, monkeypatch):
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import compose_experiment_config
#         result_json = compose_experiment_config.forward(
#             name="../../../etc/passwd",
#             components="model=resnet1d",
#             extra_overrides="",
#         )
#         result = json.loads(result_json)
#         assert result["success"] is False
#         assert "SAFETY" in result["error"]

#     def test_refuses_overwrite_human_experiment(self, mock_config_dir, monkeypatch):
#         """Cannot overwrite a human-authored experiment."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import compose_experiment_config
#         result_json = compose_experiment_config.forward(
#             name="galaxyppg_hr_ppg",  # Already exists, human-authored
#             components="model=resnet1d dataset=galaxyppg task=regression",
#             extra_overrides="",
#         )
#         result = json.loads(result_json)
#         assert result["success"] is False
#         assert "NOT agent-generated" in result["error"]

#     def test_overwrites_agent_generated_experiment(self, mock_config_dir, monkeypatch):
#         """Can overwrite a previously agent-generated experiment."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import compose_experiment_config

#         # Create first version
#         compose_experiment_config.forward(
#             name="iterative_exp",
#             components="model=resnet1d dataset=ecg_npz task=classification",
#             extra_overrides="",
#         )

#         # Overwrite — should succeed
#         result_json = compose_experiment_config.forward(
#             name="iterative_exp",
#             components="model=transformer1d dataset=galaxyppg task=regression",
#             extra_overrides="model.in_channels=1",
#         )
#         result = json.loads(result_json)
#         assert result["success"] is True

#         # Verify updated content
#         content = (mock_config_dir / "configs" / "experiment" / "iterative_exp.yaml").read_text()
#         assert "transformer1d" in content
#         assert "galaxyppg" in content

#     def test_with_nested_transforms(self, mock_config_dir, monkeypatch):
#         """Handles nested config paths like transforms=galaxyppg/hr_ppg."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import compose_experiment_config
#         result_json = compose_experiment_config.forward(
#             name="full_ppg_experiment",
#             components="model=encoder_classifier dataset=galaxyppg task=regression trainer=fast_dev transforms=galaxyppg/hr_ppg",
#             extra_overrides="model.in_channels=1 model.num_classes=1",
#         )
#         result = json.loads(result_json)
#         assert result["success"] is True

#         content = (mock_config_dir / "configs" / "experiment" / "full_ppg_experiment.yaml").read_text()
#         assert "override /transforms: galaxyppg/hr_ppg" in content

#     def test_rejects_code_injection_in_overrides(self, mock_config_dir, monkeypatch):
#         """Safety scan catches code injection in extra_overrides."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import compose_experiment_config
#         result_json = compose_experiment_config.forward(
#             name="safe_experiment",
#             components="model=resnet1d dataset=ecg_npz task=classification",
#             extra_overrides="_target_=os.system",
#         )
#         result = json.loads(result_json)
#         assert result["success"] is False
#         assert "SAFETY" in result["error"]


# # ═════════════════════════════════════════════════
# # Tests: End-to-end workflow
# # ═════════════════════════════════════════════════


# class TestEndToEndWorkflow:
#     def test_create_variant_then_compose_experiment(self, mock_config_dir, monkeypatch):
#         """
#         Full workflow: create model variant → compose experiment → verify.
#         This is exactly what the agent does in a real campaign.
#         """
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant, compose_experiment_config

#         # Step 1: Create a model variant
#         r1 = json.loads(create_config_variant.forward(
#             group="model",
#             base_config="transformer1d",
#             new_name="transformer1d_wide",
#             modifications="d_model=512 n_heads=8 n_layers=6",
#         ))
#         assert r1["success"] is True

#         # Step 2: Create a trainer variant
#         r2 = json.loads(create_config_variant.forward(
#             group="trainer",
#             base_config="default",
#             new_name="long_train",
#             modifications="epochs=50 lr=0.0001",
#         ))
#         assert r2["success"] is True

#         # Step 3: Compose an experiment using the new variants
#         r3 = json.loads(compose_experiment_config.forward(
#             name="ppg_wide_transformer_long",
#             components="model=transformer1d_wide dataset=galaxyppg task=regression trainer=long_train transforms=galaxyppg/hr_ppg",
#             extra_overrides="model.in_channels=1 model.num_classes=1 trainer.monitor_metric=val/mae trainer.monitor_mode=min",
#         ))
#         assert r3["success"] is True
#         assert r3["override_syntax"] == "experiment=ppg_wide_transformer_long"

#         # Step 4: Verify the experiment file references our variants
#         exp_content = (mock_config_dir / "configs" / "experiment" / "ppg_wide_transformer_long.yaml").read_text()
#         assert "override /model: transformer1d_wide" in exp_content
#         assert "override /trainer: long_train" in exp_content
#         assert "in_channels" in exp_content

#         # Step 5: Verify the model variant has correct params
#         model_data = _read_yaml_safe(mock_config_dir / "configs" / "model" / "transformer1d_wide.yaml")
#         assert model_data["d_model"] == 512
#         assert model_data["n_heads"] == 8
#         assert model_data["_target_"] == "biosignals.models.transformer.Transformer1D"

#     def test_list_shows_agent_generated_configs(self, mock_config_dir, monkeypatch):
#         """After creating configs, list_available_configs() shows them."""
#         monkeypatch.chdir(mock_config_dir)

#         from biosignals.agent.tools import create_config_variant, list_available_configs

#         # Create a variant
#         create_config_variant.forward(
#             group="model", base_config="transformer1d",
#             new_name="transformer1d_big", modifications="d_model=1024",
#         )

#         # Verify it appears in listing
#         listing = list_available_configs.forward()
#         assert "transformer1d_big" in listing
