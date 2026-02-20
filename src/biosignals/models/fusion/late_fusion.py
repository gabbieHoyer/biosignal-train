# src/biosignals/models/fusion/late_fusion.py
from __future__ import annotations

import torch
from torch import nn


class ConcatMLPFusion(nn.Module):
    """
    Fusion core:
      concat embeddings -> fused embedding

    Input:  z_cat (B, M*D)
    Output: z_fused (B, D_out)
    """

    def __init__(
        self, in_dim: int, out_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.ReLU(inplace=True),
        )
        self.out_dim = int(out_dim)

    def forward(self, z_cat: torch.Tensor) -> torch.Tensor:
        return self.net(z_cat)
