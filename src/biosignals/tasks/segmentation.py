# src/biosignals/tasks/segmentation.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from biosignals.data.collate import pad_labels_kt, pad_labels_t, pad_stack_ct
from biosignals.data.types import Batch, Sample
from biosignals.metrics.segmentation import dice_multiclass
from biosignals.tasks.base import Task


@dataclass
class SegmentationTask(Task):
    """
    Supports:
      - multiclass per-sample labels: y is (T,) int in [0..K-1]
      - multilabel per-sample labels: y is (K, T) float in {0,1}
    """

    name: str = "segmentation"
    num_classes: int = 2
    multilabel: bool = False
    ignore_index: int = 255
    primary_modality: str = "main"

    def collate_fn(self) -> Callable[[List[Sample]], Batch]:
        def collate(samples: List[Sample]) -> Batch:
            modalities = list(samples[0].signals.keys())

            signals: Dict[str, torch.Tensor] = {}
            masks: Dict[str, torch.Tensor] = {}
            lengths = None
            tmax = None

            for m in modalities:
                xs = [torch.from_numpy(s.signals[m]) for s in samples]  # (C,T)
                x_pad, lengths, mask = pad_stack_ct(xs, pad_value=0.0)
                signals[m] = x_pad
                masks[m] = mask
                tmax = x_pad.shape[-1]

            # Targets
            if self.multilabel:
                ys = [
                    torch.as_tensor(s.targets["y"], dtype=torch.float32) for s in samples
                ]  # (K,T)
                y_pad = pad_labels_kt(ys, tmax=int(tmax))
            else:
                ys = [torch.as_tensor(s.targets["y"], dtype=torch.long) for s in samples]  # (T,)
                y_pad = pad_labels_t(ys, tmax=int(tmax), ignore_index=self.ignore_index)

            meta = {
                "lengths": lengths,
                "mask": masks[self.primary_modality],  # (B,T)
                "ids": [s.meta.get("id") for s in samples],
                "fs": [s.meta.get("fs") for s in samples],
            }
            return Batch(signals=signals, targets={"y": y_pad}, meta=meta)

        return collate

    def training_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        mask = batch.meta["mask"]  # (B,T)
        logits = model(batch.signals, batch.meta)  # (B,K,T) for segmentation

        if self.multilabel:
            y = batch.targets["y"]  # (B,K,T)
            # mask broadcast to (B,1,T)
            w = mask.unsqueeze(1).float()
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction="sum") / (
                w.sum() + 1e-6
            )
            # simple metric: mean prob on positives (toy)
            prob = logits.sigmoid()
            metric = (
                prob[y > 0.5].mean() if (y > 0.5).any() else torch.tensor(0.0, device=logits.device)
            )
            return {"loss": loss, "pos_prob": metric}

        # multiclass
        y = batch.targets["y"]  # (B,T), padded with ignore_index
        loss = F.cross_entropy(logits, y, ignore_index=self.ignore_index)
        dice = dice_multiclass(
            logits, y, mask=mask & (y != self.ignore_index), num_classes=self.num_classes
        )
        return {"loss": loss, "dice": dice}

    @torch.no_grad()
    def validation_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        return self.training_step(model, batch)
