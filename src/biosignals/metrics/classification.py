# src/biosignals/metrics/classification.py
from __future__ import annotations
import torch


@torch.no_grad()
def multilabel_exact_match_accuracy(
    logits: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    # logits: (B,K)
    probs = logits.sigmoid()
    pred = (probs > threshold)
    true = (y > 0.5)
    return pred.eq(true).float().mean()


@torch.no_grad()
def multilabel_f1_micro(
    logits: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    probs = logits.sigmoid()
    pred = (probs > threshold)
    true = (y > 0.5)

    tp = (pred & true).sum().float()
    fp = (pred & (~true)).sum().float()
    fn = ((~pred) & true).sum().float()

    return (2 * tp) / (2 * tp + fp + fn + eps)


@torch.no_grad()
def multiclass_accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # logits: (B,C)  y: (B,)
    pred = logits.argmax(dim=-1)
    return pred.eq(y).float().mean()
