# src/biosignals/config/schema.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING, DictConfig, OmegaConf

log = logging.getLogger("biosignals")


# -----------------------
# Stable, typed sections
# -----------------------


@dataclass
class DistConfig:
    backend: str = "nccl"


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class DataConfig:
    # matches biosignals.data.datamodule.DataConfig
    _target_: str = "biosignals.data.datamodule.DataConfig"
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2


@dataclass
class TrainerConfig:
    # matches biosignals.engine.trainer.TrainerConfig
    _target_: str = "biosignals.engine.trainer.TrainerConfig"

    epochs: int = 20
    lr: float = 3e-4
    amp: bool = True
    grad_clip: float = 1.0
    log_every: int = 50

    # Outputs
    output_dir: str = "."
    ckpt_dir: str = "checkpoints"
    metrics_file: str = "metrics.jsonl"
    summary_file: str = "summary.json"

    # Checkpoint policy
    save_last: bool = True
    save_best: bool = True
    save_every: int = 1

    # Monitoring policy
    monitor_metric: str = "auto"
    monitor_mode: str = "auto"

    # Cache observability (optional)
    log_cache: bool = True
    cache_scan_every: int = 1


@dataclass
class InitConfig:
    # Optional SSL init block. If present, ckpt_path is required.
    ckpt_path: str = MISSING
    src_prefix: Optional[str] = None
    dst_prefix: Optional[str] = None


# -----------------------
# Flexible Hydra nodes
# -----------------------


@dataclass
class DatasetCacheConfig:
    train: Optional[str] = None
    val: Optional[str] = None
    test: Optional[str] = None


@dataclass
class DatasetGroupConfig:
    train: Any = MISSING
    val: Optional[Any] = None
    test: Optional[Any] = None
    cache: DatasetCacheConfig = field(default_factory=DatasetCacheConfig)


@dataclass
class TransformsGroupConfig:
    params: Dict[str, Any] = field(default_factory=dict)  # free-form
    train: Any = None
    val: Any = None
    test: Any = None


@dataclass
class TopConfig:
    seed: int = 42

    dist: DistConfig = field(default_factory=DistConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    dataset: DatasetGroupConfig = field(default_factory=DatasetGroupConfig)
    transforms: TransformsGroupConfig = field(default_factory=TransformsGroupConfig)

    task: Any = MISSING
    model: Any = MISSING
    logger: Any = None

    # Optional blocks that may exist in composed configs
    init: Optional[InitConfig] = None
    experiment: Any = None

    # Hydra injects this at runtime; include so strict mode can allow it
    hydra: Any = None


def _warn_or_raise(msg: str, *, strict: bool) -> None:
    if strict:
        raise ValueError(msg)
    log.warning(msg)


def _sanity_checks(cfg: DictConfig, *, strict: bool) -> None:
    # Required top-level sections
    if OmegaConf.is_missing(cfg, "dataset.train"):
        _warn_or_raise("Config missing required key: dataset.train", strict=strict)
    if OmegaConf.is_missing(cfg, "task"):
        _warn_or_raise("Config missing required key: task", strict=strict)
    if OmegaConf.is_missing(cfg, "model"):
        _warn_or_raise("Config missing required key: model", strict=strict)

    # monitor_mode validation
    mmode = str(cfg.trainer.monitor_mode).lower() if "trainer" in cfg else "auto"
    if mmode not in {"min", "max", "auto"}:
        _warn_or_raise(
            f"trainer.monitor_mode must be one of {{min,max,auto}}, got: {cfg.trainer.monitor_mode}",
            strict=strict,
        )

    # init validation: if init block exists, ckpt_path must be set
    init_cfg = cfg.get("init")
    if init_cfg is not None:
        if (
            OmegaConf.is_missing(init_cfg, "ckpt_path")
            or not str(init_cfg.get("ckpt_path", "")).strip()
        ):
            _warn_or_raise(
                "Config has an init block but init.ckpt_path is missing/empty. "
                "Either remove init: entirely, or set init.ckpt_path.",
                strict=strict,
            )


def validate_cfg(
    cfg: DictConfig,
    *,
    strict: bool = False,
    lock_struct: bool = False,
) -> DictConfig:
    """
    Validate/normalize a Hydra config by merging into a structured schema.

    strict=False:
      - allow unknown keys (recommended for rapid iteration / agentic workflows)
      - only sanity-check and warn

    strict=True:
      - unknown top-level keys become errors (schema must include hydra/experiment/etc)
      - sanity-check raises

    lock_struct=True:
      - prevents writing new keys into the returned cfg (useful in tests)
    """
    base = OmegaConf.structured(TopConfig)

    # Allow unknown keys unless strict
    OmegaConf.set_struct(base, bool(strict))

    merged = OmegaConf.merge(base, cfg)

    if lock_struct:
        OmegaConf.set_struct(merged, True)

    _sanity_checks(merged, strict=strict)
    return merged


# --------------------------------------------
