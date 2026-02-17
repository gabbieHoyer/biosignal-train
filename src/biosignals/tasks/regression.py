# src/biosignals/tasks/regression.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from biosignals.data.types import Batch, Sample
from biosignals.data.collate import pad_stack_ct
from biosignals.tasks.base import Task
from biosignals.metrics.regression import mae as mae_metric, rmse as rmse_metric, r2 as r2_metric



@dataclass
class RegressionTask(Task):
    """
    Simple scalar regression on windows (e.g., HR).

    Expects dataset emits targets["y"] as float (or shape (1,)).
    Expects model outputs shape (B,) or (B,1).
    """
    name: str = "regression"
    primary_modality: str = "main"

    def collate_fn(self) -> Callable[[List[Sample]], Batch]:
        def collate(samples: List[Sample]) -> Batch:
            modalities = list(samples[0].signals.keys())
            signals: Dict[str, torch.Tensor] = {}
            masks: Dict[str, torch.Tensor] = {}
            lengths = None

            for m in modalities:
                xs = [torch.from_numpy(s.signals[m]) for s in samples]  # (C,T)
                x_pad, lengths, mask = pad_stack_ct(xs, pad_value=0.0)
                signals[m] = x_pad
                masks[m] = mask

            y = torch.tensor([float(s.targets["y"]) for s in samples], dtype=torch.float32)  # (B,)

            meta = {
                "lengths": lengths,
                "mask_by_modality": masks,
                "mask": masks.get(self.primary_modality, next(iter(masks.values()))),
                "ids": [s.meta.get("id") for s in samples],
                "fs": [s.meta.get("fs") for s in samples],
            }
            return Batch(signals=signals, targets={"y": y}, meta=meta)

        return collate


    def training_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        pred = model(batch.signals, batch.meta)  # (B,) or (B,1)
        y = batch.targets["y"]                  # (B,)

        if pred.ndim == 2 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        pred = pred.float()

        loss = F.mse_loss(pred, y)
        return {
            "loss": loss,
            "mae": mae_metric(pred, y),
            "rmse": rmse_metric(pred, y),
            "r2": r2_metric(pred, y),
        }


    @torch.no_grad()
    def validation_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        return self.training_step(model, batch)
