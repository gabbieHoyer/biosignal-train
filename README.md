# biosignal-train

Training and evaluation code for biosignal machine learning experiments, built around Hydra configs and a small, extensible PyTorch engine. The repo is set up for reproducible experiment runs, optional DDP via `torchrun`, and supports both single-signal and multi-modal pipelines (for example PPG + accelerometer) including fusion models and self-supervised pretraining.

## What’s inside

- **Hydra-based experiment configuration** (`configs/`) for datasets, transforms, models, tasks, and logging.
- **DDP-friendly training loop** with clear separation between task logic, model architecture, and training engine.
- **Tasks**: classification, regression, segmentation, detection, and masked-reconstruction SSL.
- **Models**: 1D ResNet, Transformer, TCN-style segmenter, ConvMAE1D (SSL), and multi-modal fusion variants including a Perceiver-style fusion module.
- **Logging**: config hooks for W&B and MLflow.

## Repository layout

- `configs/` Hydra configs:
  - `configs/train.yaml` top-level training config
  - `configs/experiment/` runnable experiment presets (MIT-BIH, GalaxyPPG)
  - `configs/model/`, `configs/task/`, `configs/dataset/`, `configs/transforms/`, `configs/logger/`
- `src/biosignals/` library code:
  - `cli/` train and eval entrypoints
  - `engine/` trainer and loops
  - `tasks/` task definitions (losses, step logic, metrics wiring)
  - `models/` backbones, heads, fusion modules, SSL models
  - `metrics/` metric implementations
  - `utils/` distributed helpers, checkpointing, reproducibility, paths
- `scripts/standardize/` dataset standardization scripts (raw -> standardized)
- `tests/` unit and smoke tests

## Install

Create an environment with Python 3.10+.

```bash
pip install -e .
pip install -e ".[dev]"
```

Optional logger installs:
```bash
pip install -e ".[wandb]"
pip install -e ".[mlflow]"
```

## Quickstart: run an experiment

Hydra composes configs from configs/. Most runs are driven by an experiment preset under configs/experiment/.

Example patterns (adjust to the exact preset you want):

```bash
python -m biosignals.cli.train experiment=galaxyppg/hr/ppg_baseline
python -m biosignals.cli.train experiment=galaxyppg/hr/perceiver_fusion
python -m biosignals.cli.train experiment=mitbih/arrhythmia/ecg_resnet
```

Hydra will write run artifacts under `outputs/` by default (ignored by git). Multi-run sweeps land in `multirun/`.

## Data

This repo does not ship datasets.

Datasets are expected to live under a standardized directory referenced by:

- `DATA_STD_DIR` (environment variable), falling back to `./data/standardized`

Example:

```bash
export DATA_STD_DIR=/path/to/standardized
```

Each dataset config under `configs/dataset/` defines the expected folder structure for that dataset (for example `galaxyppg/v1/...` and parquet “views”).

See `scripts/standardize/` for standardization scripts.

## Distributed training (DDP)

For multi-GPU:

```bash
torchrun --nproc_per_node=2 -m biosignals.cli.train experiment=galaxyppg/hr/perceiver_fusion
```

## Development

Run tests:
```bash
pytest -q
```

Lint:
```bash
ruff check .
```

Format:
```bash
ruff format .
```

Pre-commit:
```bash
pre-commit install
pre-commit run -a
```

## License

See `LICENSE`.