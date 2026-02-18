# tests/test_code_generation.py
"""
Unit tests for Tier 3 code-generation tools.
No API key needed — tests AST validation, model registration,
and source reading against mock project directories.

Run from project root:
    PYTHONPATH=$PWD/src python -m pytest tests/test_code_generation.py -v
"""
import json
import textwrap
from pathlib import Path

import pytest

from biosignals.agent.tools import (
    _ast_validate_code,
    _resolve_target_to_source,
    _find_source_root,
    _AGENT_GENERATED_PY_MARKER,
    _AGENT_GENERATED_MARKER,
    _read_yaml_safe,
    CodeSafetyError,
)


# ─────────────────────────────────────────────────
# Sample model code (safe and unsafe variants)
# ─────────────────────────────────────────────────

SAFE_MODEL_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Optional

    class ResNet1DDeep(nn.Module):
        \"\"\"A deeper variant of ResNet1D.\"\"\"

        def __init__(self, in_channels: int = 1, num_classes: int = 5,
                     base_filters: int = 128, n_blocks: int = 8):
            super().__init__()
            self.in_channels = in_channels
            self.stem = nn.Conv1d(in_channels, base_filters, 7, padding=3)
            self.bn = nn.BatchNorm1d(base_filters)
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(base_filters, base_filters, 3, padding=1),
                    nn.BatchNorm1d(base_filters),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(base_filters, base_filters, 3, padding=1),
                    nn.BatchNorm1d(base_filters),
                )
                for _ in range(n_blocks)
            ])
            self.head = nn.Linear(base_filters, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.bn(self.stem(x)))
            for block in self.blocks:
                x = F.relu(x + block(x))
            x = x.mean(dim=-1)
            return self.head(x)
""")

SAFE_MINIMAL_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self, d: int = 32):
            super().__init__()
            self.fc = nn.Linear(d, d)
        def forward(self, x):
            return self.fc(x)
""")

UNSAFE_EXEC_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            exec("import os; os.system('rm -rf /')")
        def forward(self, x):
            return x
""")

UNSAFE_IMPORT_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    import subprocess

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x
""")

UNSAFE_OS_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    import os

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            os.system("whoami")
            return x
""")

UNSAFE_OPEN_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            open("/etc/passwd", "r")
            return x
""")

NO_MODULE_CODE = textwrap.dedent("""\
    import torch

    def my_function(x):
        return x * 2
""")

NO_FORWARD_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
""")

SYNTAX_ERROR_CODE = "def broken(\nclass foo:"

MULTIPLE_CLASSES_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.fc = nn.Linear(d, d)
        def forward(self, x):
            return self.fc(x)

    class Decoder(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.fc = nn.Linear(d, d)
        def forward(self, x):
            return self.fc(x)
""")

NUMPY_MODEL_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    import numpy as np
    from typing import Tuple

    class NumpyInitModel(nn.Module):
        def __init__(self, d: int = 64):
            super().__init__()
            self.fc = nn.Linear(d, d)
        def forward(self, x):
            return self.fc(x)
""")

BIOSIGNALS_IMPORT_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    from biosignals.models.resnet import ResNet1D

    class ExtendedResNet(nn.Module):
        def __init__(self, in_channels: int = 1, num_classes: int = 5):
            super().__init__()
            self.backbone = ResNet1D(in_channels, num_classes)
            self.extra_fc = nn.Linear(num_classes, num_classes)
        def forward(self, x):
            return self.extra_fc(self.backbone(x))
""")

GLOBAL_STATEMENT_CODE = textwrap.dedent("""\
    import torch
    import torch.nn as nn

    counter = 0

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            global counter
            counter += 1
            return x
""")


# ─────────────────────────────────────────────────
# Fixture: mock project directory with source files
# ─────────────────────────────────────────────────


@pytest.fixture
def mock_project(tmp_path):
    """Create a mock project with configs/ and src/biosignals/models/."""
    # Config root
    configs = tmp_path / "configs"
    configs.mkdir()
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

    # Source root
    src = tmp_path / "src"
    (src / "biosignals" / "models").mkdir(parents=True)
    (src / "biosignals" / "__init__.py").write_text("")
    (src / "biosignals" / "models" / "__init__.py").write_text("")
    (src / "biosignals" / "models" / "resnet.py").write_text(textwrap.dedent("""\
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class ResNet1D(nn.Module):
            def __init__(self, in_channels=1, num_classes=5, base_filters=64):
                super().__init__()
                self.conv1 = nn.Conv1d(in_channels, base_filters, 7, padding=3)
                self.bn1 = nn.BatchNorm1d(base_filters)
                self.fc = nn.Linear(base_filters, num_classes)

            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = x.mean(dim=-1)
                return self.fc(x)
    """))
    (src / "biosignals" / "models" / "transformer.py").write_text(textwrap.dedent("""\
        import torch
        import torch.nn as nn

        class Transformer1D(nn.Module):
            def __init__(self, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
                super().__init__()
                layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout)
                self.encoder = nn.TransformerEncoder(layer, n_layers)
                self.fc = nn.Linear(d_model, 1)

            def forward(self, x):
                return self.fc(self.encoder(x).mean(dim=1))
    """))

    # Experiment and trainer configs (for compose tests)
    exp_dir = configs / "experiment"
    exp_dir.mkdir()
    trainer_dir = configs / "trainer"
    trainer_dir.mkdir()
    (trainer_dir / "default.yaml").write_text(
        "_target_: biosignals.engine.trainer.TrainerConfig\nlr: 0.0003\nepochs: 20\n"
    )

    # Dataset / task
    (configs / "dataset").mkdir()
    (configs / "dataset" / "galaxyppg.yaml").write_text(
        "_target_: biosignals.data.datasets.GalaxyPPGDataset\n"
    )
    (configs / "task").mkdir()
    (configs / "task" / "regression.yaml").write_text(
        "_target_: biosignals.tasks.RegressionTask\n"
    )
    (configs / "transforms").mkdir()
    ppg_tr = configs / "transforms" / "galaxyppg"
    ppg_tr.mkdir()
    (ppg_tr / "hr_ppg.yaml").write_text(
        "_target_: biosignals.transforms.HrPpg\n"
    )

    return tmp_path


# ═════════════════════════════════════════════════
# Tests: AST Safety Validation
# ═════════════════════════════════════════════════


class TestASTValidation:
    """Tests for _ast_validate_code() safety scanner."""

    def test_safe_model_passes(self):
        result = _ast_validate_code(SAFE_MODEL_CODE)
        assert result["class_name"] == "ResNet1DDeep"
        assert result["has_forward"] is True
        assert len(result["constructor_params"]) == 4
        param_names = [p["name"] for p in result["constructor_params"]]
        assert param_names == ["in_channels", "num_classes", "base_filters", "n_blocks"]

    def test_extracts_defaults(self):
        result = _ast_validate_code(SAFE_MODEL_CODE)
        params = {p["name"]: p for p in result["constructor_params"]}
        assert params["in_channels"]["default"] == 1
        assert params["base_filters"]["default"] == 128
        assert params["n_blocks"]["default"] == 8

    def test_minimal_model_passes(self):
        result = _ast_validate_code(SAFE_MINIMAL_CODE)
        assert result["class_name"] == "TinyModel"
        assert result["has_forward"] is True

    def test_numpy_import_allowed(self):
        result = _ast_validate_code(NUMPY_MODEL_CODE)
        assert "numpy" in result["import_modules"]

    def test_biosignals_import_allowed(self):
        result = _ast_validate_code(BIOSIGNALS_IMPORT_CODE)
        assert result["class_name"] == "ExtendedResNet"

    def test_blocks_exec(self):
        with pytest.raises(CodeSafetyError, match="exec"):
            _ast_validate_code(UNSAFE_EXEC_CODE)

    def test_blocks_subprocess_import(self):
        with pytest.raises(CodeSafetyError, match="subprocess"):
            _ast_validate_code(UNSAFE_IMPORT_CODE)

    def test_blocks_os_import(self):
        with pytest.raises(CodeSafetyError, match="os"):
            _ast_validate_code(UNSAFE_OS_CODE)

    def test_blocks_open(self):
        with pytest.raises(CodeSafetyError, match="open"):
            _ast_validate_code(UNSAFE_OPEN_CODE)

    def test_blocks_global_statement(self):
        with pytest.raises(CodeSafetyError, match="global"):
            _ast_validate_code(GLOBAL_STATEMENT_CODE)

    def test_requires_nn_module(self):
        with pytest.raises(CodeSafetyError, match="nn.Module"):
            _ast_validate_code(NO_MODULE_CODE)

    def test_requires_forward(self):
        with pytest.raises(CodeSafetyError, match="forward"):
            _ast_validate_code(NO_FORWARD_CODE)

    def test_syntax_error(self):
        with pytest.raises(CodeSafetyError, match="Syntax error"):
            _ast_validate_code(SYNTAX_ERROR_CODE)

    def test_multiple_classes_uses_first(self):
        """When multiple nn.Module subclasses exist, uses the first."""
        result = _ast_validate_code(MULTIPLE_CLASSES_CODE)
        assert result["class_name"] == "Encoder"


# ═════════════════════════════════════════════════
# Tests: Source Resolution
# ═════════════════════════════════════════════════


class TestSourceResolution:
    def test_resolves_target_to_file(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        path = _resolve_target_to_source("biosignals.models.resnet.ResNet1D")
        assert path is not None
        assert path.name == "resnet.py"
        assert "ResNet1D" in path.read_text()

    def test_returns_none_for_missing(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        path = _resolve_target_to_source("biosignals.models.nonexistent.Foo")
        assert path is None

    def test_finds_source_root(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        root = _find_source_root()
        assert root is not None
        assert (root / "biosignals").is_dir()


# ═════════════════════════════════════════════════
# Tests: read_model_source Tool
# ═════════════════════════════════════════════════


class TestReadModelSource:
    def test_reads_resnet_source(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import read_model_source
        result = read_model_source.forward(model_name="resnet1d")

        assert "ResNet1D" in result
        assert "class ResNet1D" in result
        assert "forward" in result
        assert "biosignals.models.resnet.ResNet1D" in result

    def test_reads_transformer_source(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import read_model_source
        result = read_model_source.forward(model_name="transformer1d")

        assert "Transformer1D" in result
        assert "d_model" in result

    def test_error_for_missing_model(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import read_model_source
        result = read_model_source.forward(model_name="nonexistent_model")

        assert "ERROR" in result

    def test_shows_constructor_params(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import read_model_source
        result = read_model_source.forward(model_name="resnet1d")

        assert "__init__" in result
        assert "in_channels" in result


# ═════════════════════════════════════════════════
# Tests: register_generated_model Tool
# ═════════════════════════════════════════════════


class TestRegisterGeneratedModel:
    def test_registers_safe_model(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        result_json = register_generated_model.forward(
            name="resnet1d_deep",
            code=SAFE_MODEL_CODE,
            constructor_args="in_channels=1 num_classes=5 base_filters=128 n_blocks=8",
        )
        result = json.loads(result_json)

        assert result["success"] is True
        assert result["class_name"] == "ResNet1DDeep"
        assert result["override_syntax"] == "model=resnet1d_deep"

        # Verify .py file was written
        model_path = Path(result["model_path"])
        assert model_path.exists()
        source = model_path.read_text()
        assert _AGENT_GENERATED_PY_MARKER in source
        assert "class ResNet1DDeep" in source

        # Verify Hydra YAML config was written
        config_path = Path(result["config_path"])
        assert config_path.exists()
        config_data = _read_yaml_safe(config_path)
        assert config_data["_target_"] == "biosignals.models.generated.resnet1d_deep.ResNet1DDeep"
        assert config_data["in_channels"] == 1
        assert config_data["n_blocks"] == 8

        # Verify __init__.py was created
        gen_init = model_path.parent / "__init__.py"
        assert gen_init.exists()

    def test_rejects_unsafe_code(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        result = json.loads(register_generated_model.forward(
            name="bad_model", code=UNSAFE_EXEC_CODE, constructor_args="",
        ))
        assert result["success"] is False
        assert "SAFETY" in result["error"]

    def test_rejects_subprocess_import(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        result = json.loads(register_generated_model.forward(
            name="bad_model2", code=UNSAFE_IMPORT_CODE, constructor_args="",
        ))
        assert result["success"] is False
        assert "SAFETY" in result["error"]

    def test_rejects_code_without_forward(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        result = json.loads(register_generated_model.forward(
            name="no_forward", code=NO_FORWARD_CODE, constructor_args="",
        ))
        assert result["success"] is False
        assert "forward" in result["error"]

    def test_rejects_bad_name(self, mock_project, monkeypatch):
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        result = json.loads(register_generated_model.forward(
            name="../../etc/passwd",
            code=SAFE_MINIMAL_CODE,
            constructor_args="",
        ))
        assert result["success"] is False
        assert "SAFETY" in result["error"]

    def test_auto_extracts_defaults(self, mock_project, monkeypatch):
        """When no constructor_args given, extracts defaults from AST."""
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        result = json.loads(register_generated_model.forward(
            name="auto_defaults_model",
            code=SAFE_MINIMAL_CODE,
            constructor_args="",
        ))
        assert result["success"] is True

        config_data = _read_yaml_safe(Path(result["config_path"]))
        # TinyModel has d=32 as default
        assert config_data["d"] == 32

    def test_overwrites_agent_generated(self, mock_project, monkeypatch):
        """Can overwrite a previously agent-generated model."""
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        # Version 1
        register_generated_model.forward(
            name="evolving_model", code=SAFE_MINIMAL_CODE, constructor_args="d=32",
        )

        # Version 2 — different code
        result = json.loads(register_generated_model.forward(
            name="evolving_model", code=SAFE_MODEL_CODE,
            constructor_args="in_channels=1 num_classes=10 base_filters=64 n_blocks=4",
        ))
        assert result["success"] is True
        assert result["class_name"] == "ResNet1DDeep"

    def test_refuses_overwrite_human_config(self, mock_project, monkeypatch):
        """Cannot overwrite a human-authored model config (resnet1d)."""
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        result = json.loads(register_generated_model.forward(
            name="resnet1d",  # Already exists in configs/model/
            code=SAFE_MODEL_CODE,
            constructor_args="in_channels=1",
        ))
        assert result["success"] is False
        assert "NOT agent-generated" in result["error"]

    def test_model_visible_in_listings(self, mock_project, monkeypatch):
        """Registered model appears in list_available_configs."""
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model, list_available_configs

        register_generated_model.forward(
            name="custom_arch", code=SAFE_MINIMAL_CODE, constructor_args="d=64",
        )

        listing = list_available_configs.forward()
        assert "custom_arch" in listing


# ═════════════════════════════════════════════════
# Tests: End-to-end workflow
# ═════════════════════════════════════════════════


class TestEndToEndCodeGeneration:
    def test_read_modify_register_workflow(self, mock_project, monkeypatch):
        """
        Full agent workflow:
        1. Read existing model source
        2. (Agent modifies it — simulated here)
        3. Register the modified version
        4. Verify it's usable
        """
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import (
            read_model_source,
            register_generated_model,
            compose_experiment_config,
        )

        # Step 1: Read source
        source = read_model_source.forward(model_name="resnet1d")
        assert "class ResNet1D" in source

        # Step 2: "Agent" creates a deeper variant
        # (In real usage, the LLM generates this from reading the source)
        deeper_code = SAFE_MODEL_CODE  # Our pre-written deeper variant

        # Step 3: Register
        reg = json.loads(register_generated_model.forward(
            name="resnet1d_deep_v2",
            code=deeper_code,
            constructor_args="in_channels=1 num_classes=1 base_filters=128 n_blocks=8",
        ))
        assert reg["success"] is True

        # Step 4: Compose experiment using the generated model
        exp = json.loads(compose_experiment_config.forward(
            name="ppg_deep_resnet_experiment",
            components="model=resnet1d_deep_v2 dataset=galaxyppg task=regression trainer=default transforms=galaxyppg/hr_ppg",
            extra_overrides="model.in_channels=1 model.num_classes=1",
        ))
        assert exp["success"] is True
        assert exp["override_syntax"] == "experiment=ppg_deep_resnet_experiment"

        # Verify the experiment config points to the generated model
        exp_content = (mock_project / "configs" / "experiment" / "ppg_deep_resnet_experiment.yaml").read_text()
        assert "resnet1d_deep_v2" in exp_content

    def test_iterative_code_refinement(self, mock_project, monkeypatch):
        """
        Simulates the agent fixing a model after a failed attempt:
        1. Register model with bad code → fails safety check
        2. Fix the code → succeeds
        """
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        # Attempt 1: bad code (no forward method)
        r1 = json.loads(register_generated_model.forward(
            name="iter_model", code=NO_FORWARD_CODE, constructor_args="",
        ))
        assert r1["success"] is False
        assert "forward" in r1["error"]

        # Attempt 2: fixed code
        r2 = json.loads(register_generated_model.forward(
            name="iter_model", code=SAFE_MINIMAL_CODE, constructor_args="d=64",
        ))
        assert r2["success"] is True

    def test_generated_model_target_path(self, mock_project, monkeypatch):
        """Verify _target_ path follows the convention."""
        monkeypatch.chdir(mock_project)
        from biosignals.agent.tools import register_generated_model

        result = json.loads(register_generated_model.forward(
            name="my_custom_model", code=SAFE_MINIMAL_CODE, constructor_args="d=128",
        ))
        assert result["success"] is True

        config = _read_yaml_safe(Path(result["config_path"]))
        assert config["_target_"] == "biosignals.models.generated.my_custom_model.TinyModel"
