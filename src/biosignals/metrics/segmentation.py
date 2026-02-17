# src/biosignals/metrics/segmentation.py
from __future__ import annotations
from typing import Optional

import torch

@torch.no_grad()
def dice_multiclass(
    logits: torch.Tensor,          # (B, K, T)
    y: torch.Tensor,               # (B, T) int
    mask: torch.Tensor,            # (B, T) bool
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = logits.argmax(dim=1)  # (B, T)
    dice_sum = torch.zeros((), device=logits.device)
    valid_classes = 0

    for cls in range(num_classes):
        pred_c = (pred == cls) & mask
        y_c = (y == cls) & mask

        inter = (pred_c & y_c).sum().float()
        denom = pred_c.sum().float() + y_c.sum().float()
        if denom.item() == 0:
            continue

        dice_sum = dice_sum + (2.0 * inter + eps) / (denom + eps)
        valid_classes += 1

    if valid_classes == 0:
        return torch.tensor(0.0, device=logits.device)

    return dice_sum / float(valid_classes)



# requirements:

    # mask must be bool and on same device as logits/y.

    # With the updated batch_to_device() (recursive meta move), storing mask in batch.meta["mask"] is safe.