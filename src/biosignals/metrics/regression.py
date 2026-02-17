# src/biosignals/metrics/regression.py
from __future__ import annotations
import torch


@torch.no_grad()
def mae(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1).float()
    y = y.view(-1).float()
    return (pred - y).abs().mean()


@torch.no_grad()
def rmse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1).float()
    y = y.view(-1).float()
    return torch.sqrt(torch.mean((pred - y) ** 2) + 1e-12)


@torch.no_grad()
def r2(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1).float()
    y = y.view(-1).float()
    ss_res = torch.sum((y - pred) ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2) + 1e-12
    return 1.0 - (ss_res / ss_tot)
