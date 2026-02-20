# src/biosignals/tasks/classification.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from biosignals.data.types import Batch, Sample
from biosignals.tasks.base import Task
from biosignals.data.collate import pad_stack_ct

from biosignals.metrics.classification import (
    multilabel_exact_match_accuracy,
    multilabel_f1_micro,
    multiclass_accuracy,
)

@dataclass
class ClassificationTask(Task):
    name: str = "classification"

    # IMPORTANT: match your model expectation (ResNet uses "main")
    primary_modality: str = "main"

    num_classes: int = 5
    multilabel: bool = True
    threshold: float = 0.5  # for multilabel accuracy

    def collate_fn(self) -> Callable[[List[Sample]], Batch]:
        def collate(samples: List[Sample]) -> Batch:
            modalities = list(samples[0].signals.keys())  # stable order
            signals: Dict[str, torch.Tensor] = {}
            masks: Dict[str, torch.Tensor] = {}
            lengths_by_modality: Dict[str, torch.Tensor] = {}

            for m in modalities:
                xs = [torch.from_numpy(s.signals[m]) for s in samples]  # (C,T)
                x_pad, lengths, mask = pad_stack_ct(xs, pad_value=0.0)
                signals[m] = x_pad
                masks[m] = mask                 # (B,T) valid mask
                lengths_by_modality[m] = lengths

            # Resolve primary modality (fallback to first available)
            pm = self.primary_modality if self.primary_modality in signals else modalities[0]

            # targets
            if self.multilabel:
                y = torch.stack(
                    [torch.as_tensor(s.targets["y"], dtype=torch.float32) for s in samples],
                    dim=0,
                )  # (B,K)
            else:
                y = torch.tensor([int(s.targets["y"]) for s in samples], dtype=torch.long)  # (B,)

            meta = {
                "lengths": lengths_by_modality[pm],         # (B,)
                "lengths_by_modality": lengths_by_modality, # Dict[str,(B,)]
                "mask_by_modality": masks,                  # Dict[str,(B,T)]
                "mask": masks[pm],                          # (B,T) for primary modality
                "ids": [s.meta.get("id") for s in samples],

                "sample_meta": [s.meta for s in samples],
                "subject_ids": [s.meta.get("subject_id") for s in samples],
                "record_ids": [s.meta.get("record_id") or s.meta.get("npz_id") for s in samples],
            }
            return Batch(signals=signals, targets={"y": y}, meta=meta)

        return collate

    def training_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        logits = model(batch.signals, batch.meta)
        y = batch.targets["y"]

        if self.multilabel:
            loss = F.binary_cross_entropy_with_logits(logits, y)
            acc = multilabel_exact_match_accuracy(logits, y, threshold=self.threshold)
            f1 = multilabel_f1_micro(logits, y, threshold=self.threshold)
            return {"loss": loss, "acc": acc, "f1_micro": f1}

        loss = F.cross_entropy(logits, y)
        acc = multiclass_accuracy(logits, y)
        return {"loss": loss, "acc": acc}

    @torch.no_grad()
    def validation_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        return self.training_step(model, batch)
