# src/biosignals/models/heads/classification.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class MaskedMeanPool1D(nn.Module):
    """
    Pools (B,D,T) -> (B,D) using optional mask (B,T) where True=valid.
    """
    def forward(self, x: torch.Tensor, mask_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask_t is None:
            return x.mean(dim=-1)
        m = mask_t.to(device=x.device).bool().unsqueeze(1).float()  # (B,1,T)
        denom = m.sum(dim=-1).clamp_min(1.0)                        # (B,1)
        return (x * m).sum(dim=-1) / denom                          # (B,D)


@dataclass(eq=False)
class ClassificationHead(nn.Module):
    in_dim: int
    num_classes: int
    dropout: float = 0.1

    def __post_init__(self):
        super().__init__()
        self.pool = MaskedMeanPool1D()
        self.drop = nn.Dropout(float(self.dropout))
        self.fc = nn.Linear(int(self.in_dim), int(self.num_classes))

    def forward(self, feats: torch.Tensor, mask_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.pool(feats, mask_t=mask_t)  # (B,D)
        z = self.drop(z)
        return self.fc(z)                    # (B,K)
