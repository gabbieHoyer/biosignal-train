# src/biosignals/tasks/ssl_masked_recon.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from biosignals.data.collate import pad_stack_ct
from biosignals.data.types import Batch, Sample
from biosignals.tasks.base import Task

"""
SSL task: masked reconstruction loss

Conventions:
  - batch.meta["mask"] is the valid padding mask (B,T) bool
  - optional batch.meta["ssl_mask"] is the reconstruction mask (B,T) bool
"""


@dataclass
class MaskedReconstructionTask(Task):
    name: str = "ssl_masked_recon"
    primary_modality: str = "main"
    mask_ratio: float = 0.3
    mask_value: float = 0.0
    ssl_mask_meta_key: str = "ssl_mask"  # produced by transforms if enabled

    def collate_fn(self) -> Callable[[List[Sample]], Batch]:
        def collate(samples: List[Sample]) -> Batch:
            modalities = list(samples[0].signals.keys())
            signals: Dict[str, torch.Tensor] = {}
            masks: Dict[str, torch.Tensor] = {}
            lengths = None

            for m in modalities:
                xs = [torch.from_numpy(s.signals[m]) for s in samples]  # (C,T)
                x_pad, lengths, valid_mask = pad_stack_ct(xs, pad_value=0.0)
                signals[m] = x_pad
                masks[m] = valid_mask  # (B,T) bool

            meta: Dict[str, any] = {
                "lengths": lengths,
                "mask": masks[self.primary_modality],
                "ids": [s.meta.get("id") for s in samples],
            }

            # Optional: bring precomputed ssl masks from Sample meta into batch meta
            # Expect per-sample shape (T,) bool numpy array
            if all(self.ssl_mask_meta_key in s.meta for s in samples):
                ssl_masks_np = [
                    np.asarray(s.meta[self.ssl_mask_meta_key]).astype(bool, copy=False)
                    for s in samples
                ]
                T_max = int(signals[self.primary_modality].shape[-1])
                ssl_pad = torch.zeros((len(samples), T_max), dtype=torch.bool)
                for i, msk in enumerate(ssl_masks_np):
                    t_i = min(int(msk.shape[0]), T_max)
                    ssl_pad[i, :t_i] = torch.from_numpy(msk[:t_i])
                # Ensure we never mask padded region
                ssl_pad = ssl_pad & meta["mask"]
                meta["ssl_mask"] = ssl_pad

            return Batch(signals=signals, targets={}, meta=meta)

        return collate

    def training_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        x = batch.signals[self.primary_modality]  # (B,C,T)
        valid = batch.meta["mask"]  # (B,T) bool
        b, c, t = x.shape

        # Prefer transform-provided ssl mask if present; else generate randomly
        if "ssl_mask" in batch.meta and batch.meta["ssl_mask"] is not None:
            mask = batch.meta["ssl_mask"].to(device=x.device)
            mask = mask.bool() & valid
        else:
            mask = (torch.rand((b, t), device=x.device) < self.mask_ratio).bool() & valid

        # Apply mask (correct broadcasting)
        x_masked = x.masked_fill(mask.unsqueeze(1), float(self.mask_value))

        # Forward: allow tensor or dict output
        out = model({**batch.signals, self.primary_modality: x_masked}, batch.meta)
        if isinstance(out, dict):
            if "recon" in out:
                x_hat = out["recon"]
            elif "x_hat" in out:
                x_hat = out["x_hat"]
            else:
                raise KeyError(
                    "MaskedReconstructionTask expects model to return tensor or dict with key 'recon'/'x_hat'."
                )
        else:
            x_hat = out

        # MSE on masked+valid positions
        w = mask.unsqueeze(1).float()  # (B,1,T)
        loss = ((x_hat - x) ** 2 * w).sum() / (w.sum() * c + 1e-6)
        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, model: torch.nn.Module, batch: Batch) -> Dict[str, torch.Tensor]:
        return self.training_step(model, batch)
