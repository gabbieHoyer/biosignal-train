# src/biosignals/engine/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import math
import logging
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from biosignals.engine.loops import train_one_epoch, evaluate_one_epoch
from biosignals.utils.distributed import is_distributed, is_main_process
from biosignals.utils.io import append_jsonl, write_json
from biosignals.utils.checkpointing import save_checkpoint
from biosignals.loggers.base import ExperimentLogger, NoopLogger

log = logging.getLogger("biosignals")


@dataclass
class TrainerConfig:
    epochs: int = 10
    lr: float = 1e-3
    amp: bool = True
    grad_clip: float = 1.0
    log_every: int = 50

    output_dir: str = "."
    ckpt_dir: str = "checkpoints"
    metrics_file: str = "metrics.jsonl"
    summary_file: str = "summary.json"

    save_last: bool = True
    save_best: bool = True
    save_every: int = 1

    monitor_metric: str = "auto"
    monitor_mode: str = "auto"

    # NEW: cache observability
    log_cache: bool = True
    cache_scan_every: int = 1  # epochs


def _prefix_metrics(prefix: str, d: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}/{k}": float(v) for k, v in d.items()}


def _better(new: float, best: Optional[float], mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return new < best
    if mode == "max":
        return new > best
    raise ValueError(f"Unknown monitor_mode={mode}")


def _auto_monitor(task: Any, available: Dict[str, float]) -> Tuple[str, str]:
    name = str(getattr(task, "name", "unknown")).lower()
    if name == "regression":
        candidates = [("val/mae", "min"), ("val/loss", "min")]
    elif name == "classification":
        candidates = [("val/acc", "max"), ("val/loss", "min")]
    elif name == "segmentation":
        candidates = [("val/dice", "max"), ("val/loss", "min")]
    elif name == "detection":
        candidates = [("val/event_f1", "max"), ("val/loss", "min")]
    else:
        candidates = [("val/loss", "min")]

    for k, m in candidates:
        if k in available and (not math.isnan(float(available[k]))):
            return k, m
    return "val/loss", "min"


def _find_cache_dataset(ds: Any) -> Any:
    """
    Walk through wrappers (CacheDataset -> TransformDataset -> raw dataset)
    using duck typing to find an object exposing count_cached() + cache_dir/prefix.
    """
    cur = ds
    for _ in range(10):
        if hasattr(cur, "count_cached") and hasattr(cur, "cache_dir") and hasattr(cur, "prefix"):
            return cur
        if hasattr(cur, "dataset"):
            cur = getattr(cur, "dataset")
        else:
            break
    return None


def _cache_metrics(loader: DataLoader, *, tag: str) -> Dict[str, float]:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return {}
    cache_ds = _find_cache_dataset(ds)
    if cache_ds is None:
        return {}
    try:
        cached = int(cache_ds.count_cached())
        total = int(len(ds))
        frac = float(cached / total) if total > 0 else 0.0
        return {
            f"data/{tag}_cache_cached": float(cached),
            f"data/{tag}_cache_total": float(total),
            f"data/{tag}_cache_frac": float(frac),
        }
    except Exception:
        return {}


class Trainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.scaler = GradScaler(enabled=cfg.amp)

        self.global_step = 0
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_ckpt_path: Optional[str] = None

    def fit(
        self,
        model: torch.nn.Module,
        task: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        device: str = "cuda",
        logger: Optional[ExperimentLogger] = None,
    ) -> None:
        logger = logger or NoopLogger()

        run_dir = Path(self.cfg.output_dir).resolve()
        ckpt_dir = run_dir / self.cfg.ckpt_dir
        metrics_path = run_dir / self.cfg.metrics_file
        summary_path = run_dir / self.cfg.summary_file

        model.to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=float(self.cfg.lr))

        for epoch in range(int(self.cfg.epochs)):
            if is_distributed() and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Cache fill metrics (filesystem scan)
            cache_obs: Dict[str, float] = {}
            if (
                self.cfg.log_cache
                and is_main_process()
                and (epoch % int(self.cfg.cache_scan_every) == 0)
            ):
                cache_obs.update(_cache_metrics(train_loader, tag="train"))
                if val_loader is not None:
                    cache_obs.update(_cache_metrics(val_loader, tag="val"))

                if cache_obs:
                    log.info("Cache status @ epoch %d: %s", epoch, cache_obs)

            train_avg = train_one_epoch(
                model=model,
                task=task,
                loader=train_loader,
                optim=optim,
                scaler=self.scaler,
                epoch=epoch,
                amp=self.cfg.amp,
                grad_clip=self.cfg.grad_clip,
                log_every=self.cfg.log_every,
                device=device,
            )

            val_avg: Optional[Dict[str, float]] = None
            if val_loader is not None:
                val_avg = evaluate_one_epoch(
                    model=model,
                    task=task,
                    loader=val_loader,
                    device=device,
                )

            metrics: Dict[str, float] = {}
            metrics.update(_prefix_metrics("train", train_avg))
            if val_avg is not None:
                metrics.update(_prefix_metrics("val", val_avg))
            metrics.update(cache_obs)

            mm = str(self.cfg.monitor_metric)
            mode = str(self.cfg.monitor_mode)
            if mm.lower() == "auto" or mm.strip() == "":
                mm, mode = _auto_monitor(task, metrics)
            if mode.lower() == "auto" or mode.strip() == "":
                _, mode = _auto_monitor(task, metrics)

            monitor_value = metrics.get(mm, float("nan"))

            if is_main_process():
                record = {
                    "epoch": epoch,
                    "global_step": int(self.global_step),
                    "train": train_avg,
                    "val": val_avg,
                    "monitor": {"metric": mm, "mode": mode, "value": float(monitor_value)},
                    "data": cache_obs if cache_obs else None,
                }
                append_jsonl(metrics_path, record)
                logger.log_metrics(metrics, step=epoch)

            improved = (not math.isnan(float(monitor_value))) and _better(float(monitor_value), self.best_value, mode)

            if is_main_process():
                if self.cfg.save_last and ((epoch + 1) % int(self.cfg.save_every) == 0):
                    save_checkpoint(
                        ckpt_dir / "last.pt",
                        model=model,
                        optim=optim,
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=self.global_step,
                        extra={"monitor_metric": mm, "monitor_mode": mode, "monitor_value": float(monitor_value)},
                    )

                if self.cfg.save_best and improved:
                    self.best_value = float(monitor_value)
                    self.best_epoch = int(epoch)
                    self.best_ckpt_path = str((ckpt_dir / "best.pt").resolve())
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model=model,
                        optim=optim,
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=self.global_step,
                        extra={"monitor_metric": mm, "monitor_mode": mode, "monitor_value": float(monitor_value)},
                    )

                summary = {
                    "monitor": {"metric": mm, "mode": mode},
                    "best": {"epoch": self.best_epoch, "value": self.best_value, "ckpt_path": self.best_ckpt_path},
                    "last": {
                        "epoch": int(epoch),
                        "ckpt_path": str((ckpt_dir / "last.pt").resolve()) if self.cfg.save_last else None,
                    },
                }
                write_json(summary_path, summary)

            self.global_step += int(len(train_loader))

        if is_main_process():
            # log local artifacts at end (useful for remote tracking)
            logger.log_artifact(str(metrics_path), name="metrics_jsonl")
            logger.log_artifact(str(summary_path), name="summary_json")
            logger.finish()



# -------------------------------------------------------------
