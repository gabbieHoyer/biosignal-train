# src/biosignals/cli/train.py
from __future__ import annotations

import os
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

from biosignals.config.schema import validate_cfg
from biosignals.utils.reproducibility import seed_everything, set_float32_matmul_precision
from biosignals.utils.distributed import init_distributed, cleanup_distributed, is_distributed
from biosignals.utils.distributed import get_rank, is_main_process
from biosignals.utils.checkpointing import load_partial_state_dict
from biosignals.engine.trainer import Trainer
from biosignals.data.datamodule import make_train_val_loaders
from biosignals.loggers.base import NoopLogger

log = logging.getLogger("biosignals")


@hydra.main(config_path="../../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = validate_cfg(cfg, strict=False, lock_struct=False)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    init_distributed(backend=cfg.dist.backend)
    seed_everything(int(cfg.seed) + get_rank(), deterministic=True)
    set_float32_matmul_precision("high")

    use_cuda = torch.cuda.is_available()
    device = f"cuda:{local_rank}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    rank = get_rank()

    # Optional: honor cfg.logging.level by setting root level
    # (Hydra controls handlers/format/file; this only adjusts verbosity.)
    level_name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.getLogger().setLevel(level)

    try:
        log.info("Run dir: %s", str(run_dir))
        log.info("Device: %s | distributed=%s | rank=%s", device, is_distributed(), rank)

        # Instantiate core objects
        task = instantiate(cfg.task)
        model = instantiate(cfg.model)
        data_cfg = instantiate(cfg.data)

        # Trainer config writes into Hydra run dir
        trainer_cfg = instantiate(cfg.trainer)
        trainer_cfg.output_dir = str(run_dir)
        trainer = Trainer(trainer_cfg)

        # # Optional partial init (SSL -> finetune)
        # if "init" in cfg and cfg.init.get("ckpt_path", None):
        #     stats = load_partial_state_dict(
        #         model=model,
        #         ckpt_path=cfg.init.ckpt_path,
        #         src_prefix=cfg.init.src_prefix,
        #         dst_prefix=cfg.init.dst_prefix,
        #     )
        #     if is_main_process():
        #         log.info("Init from SSL: %s", stats)

        # Optional partial init (SSL -> finetune)
        init_cfg = cfg.get("init")
        if init_cfg is not None and not OmegaConf.is_missing(init_cfg, "ckpt_path"):
            ckpt_path = init_cfg.get("ckpt_path")
            if ckpt_path:  # non-empty string
                stats = load_partial_state_dict(
                    model=model,
                    ckpt_path=ckpt_path,
                    src_prefix=init_cfg.get("src_prefix"),
                    dst_prefix=init_cfg.get("dst_prefix"),
                )
                if is_main_process():
                    log.info("Init from SSL: %s", stats)

        # DDP wrap
        if is_distributed():
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if use_cuda else None,
                output_device=local_rank if use_cuda else None,
            )

        # External logger (rank0 only)
        exp_logger = NoopLogger()
        if "logger" in cfg and cfg.logger is not None and is_main_process():
            exp_logger = instantiate(cfg.logger)
            exp_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

        # Build loaders
        train_loader, val_loader = make_train_val_loaders(
            dataset_cfg=cfg.dataset,
            transforms_cfg=cfg.transforms,
            task=task,
            data_cfg=data_cfg,
        )

        if is_main_process():
            resolved_path = run_dir / "config_resolved.yaml"
            resolved_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
            log.info("Wrote: %s", str(resolved_path))
            exp_logger.log_artifact(str(resolved_path), name="config_resolved")

        trainer.fit(model, task, train_loader, val_loader, device=device, logger=exp_logger)
        cleanup_distributed()

    except Exception:
        log.exception("Fatal error in training run")
        cleanup_distributed()
        raise


if __name__ == "__main__":
    main()

# -----------------------------------------------------
