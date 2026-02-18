# biosignal-train

Training and evaluation code for biosignal machine learning experiments, built around Hydra configs and a small, extensible PyTorch engine. The repo supports reproducible runs, optional DDP via `torchrun`, and both single-signal and multi-modal pipelines (for example PPG + accelerometer), including fusion models and self-supervised pretraining. It also includes a feedback-driven experiment loop (agent + artifact contract) that can run, compare, and adapt experiments across multiple training runs without changing the training engine.

## Feedback-driven experiment loop (agent + artifact contract)

Agent code lives in [`src/biosignals/agent/`](src/biosignals/agent/).

The agent orchestrates training by invoking the existing Hydra CLI as a subprocess and then reading a small set of run artifacts to decide what to try next:

- `metrics.jsonl` (per-epoch metrics)
- `summary.json` (best checkpoint + monitor metric summary)
- `config_resolved.yaml` (the exact resolved Hydra config that ran)

Run a 3-run campaign:

```python
from biosignals.agent import run_experiment_loop

result = run_experiment_loop(
    goal=(
        "Minimize val/mae for HR regression on GalaxyPPG. "
        "Start with experiment=galaxyppg_hr_ppg as baseline. "
        "Budget: 3 runs."
    ),
    budget=3,
)
print(result)
```

## What's inside

- **Hydra-based experiment configuration** (`configs/`) for datasets, transforms, models, tasks, and logging.
- **DDP-friendly training loop** with clean separation between task logic, model architecture, and training engine.
- **Feedback-driven experiment loop** (`src/biosignals/agent/`) that runs training via the CLI, reads run artifacts, compares runs, and proposes the next config to try.
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
  - `agent/` feedback-driven experiment loop (tools, feedback store, orchestrator, optional pipeline wrapper)
  - `tasks/` task definitions (losses, step logic, metrics wiring)
  - `models/` backbones, heads, fusion modules, SSL models
  - `metrics/` metric implementations
  - `loggers/` W&B and MLflow adapters
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

---

## Reference
```bibtex
@misc{hoyer_biosignals_train_2026,
  author       = {Hoyer, Gabrielle},
  title        = {biosignals-train: Feedback-driven biosignal training and evaluation with an agent-orchestrated experiment loop},
  year         = {2026},
  howpublished = {\url{https://github.com/gabbieHoyer/biosignals-train}},
  note         = {Version: v0.1.0}
}
```