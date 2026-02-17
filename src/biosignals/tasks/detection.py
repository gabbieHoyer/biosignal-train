# src/biosignals/tasks/detection.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from biosignals.data.types import Batch, Sample
from biosignals.data.collate import pad_stack_ct
from biosignals.metrics.detection import (
    extract_events_from_probs,
    match_events_with_tolerance,
    f1_from_counts,
)
from biosignals.tasks.base import Task


@dataclass
class DetectionTask(Task):
    name: str = "detection"
    primary_modality: str = "main"

    threshold: float = 0.5
    tolerance_sec: float = 0.25
    min_event_sec: float = 0.05

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

            # Event lists (ragged) + dense supervision mask (B,1,T)
            events_list: List[List[dict]] = [list(s.targets.get("events", [])) for s in samples]
            fs_list = [float(s.meta.get("fs", 1.0)) for s in samples]

            y_dense = torch.zeros((len(samples), 1, int(tmax)), dtype=torch.float32)
            for i, (evts, fs) in enumerate(zip(events_list, fs_list)):
                # each event dict expected: {"start": int, "end": int} in sample indices
                for e in evts:
                    s_idx = int(e["start"])
                    e_idx = int(e["end"])
                    s_idx = max(0, min(s_idx, int(tmax)))
                    e_idx = max(0, min(e_idx, int(tmax)))
                    if e_idx > s_idx:
                        y_dense[i, 0, s_idx:e_idx] = 1.0

            meta = {
                "lengths": lengths,
                "mask": masks[self.primary_modality],  # (B,T)
                "events": events_list,
                "fs": fs_list,
                "ids": [s.meta.get("id") for s in samples],
            }
            return Batch(signals=signals, targets={"y_dense": y_dense}, meta=meta)

        return collate

    def training_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        mask = batch.meta["mask"]  # (B,T)
        logits = model(batch.signals, batch.meta)  # expected (B,1,T)

        y = batch.targets["y_dense"]  # (B,1,T)
        w = mask.unsqueeze(1).float()

        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction="sum") / (w.sum() + 1e-6)

        # quick proxy metric: average prob on positives
        probs = logits.sigmoid()
        pos = (y > 0.5) & (w > 0.0)
        pos_prob = probs[pos].mean() if pos.any() else torch.tensor(0.0, device=logits.device)

        return {"loss": loss, "pos_prob": pos_prob}

    @torch.no_grad()
    def validation_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        mask = batch.meta["mask"]  # (B,T)
        logits = model(batch.signals, batch.meta)  # (B,1,T)
        probs = logits.sigmoid()

        tp = fp = fn = 0

        for i in range(probs.shape[0]):
            fs = float(batch.meta["fs"][i])
            tol = int(round(self.tolerance_sec * fs))
            min_len = max(1, int(round(self.min_event_sec * fs)))

            pred_events = extract_events_from_probs(
                probs[i, 0].detach().cpu(),
                mask[i].detach().cpu(),
                threshold=self.threshold,
                min_len=min_len,
            )

            true_events_dicts = batch.meta["events"][i]
            true_events = [(int(e["start"]), int(e["end"])) for e in true_events_dicts]

            tpi, fpi, fni = match_events_with_tolerance(pred_events, true_events, tol=tol)
            tp += tpi
            fp += fpi
            fn += fni

        f1 = f1_from_counts(tp, fp, fn)

        # Return counts too (useful for global aggregation later)
        dev = logits.device
        return {
            "tp": torch.tensor(float(tp), device=dev),
            "fp": torch.tensor(float(fp), device=dev),
            "fn": torch.tensor(float(fn), device=dev),
            "event_f1": torch.tensor(float(f1), device=dev),
        }
