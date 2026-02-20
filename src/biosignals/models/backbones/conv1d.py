# src/biosignals/models/backbones/conv1d.py
from __future__ import annotations

import torch
from torch import nn


class ConvEncoder1D(nn.Module):
    """
    Simple conv encoder:
      (B,C,T) -> (B,D,T)
    """

    def __init__(self, in_channels: int, emb_dim: int = 256, depth: int = 4) -> None:
        super().__init__()
        layers = []
        c = int(in_channels)
        d = int(emb_dim)
        for _ in range(int(depth)):
            layers += [
                nn.Conv1d(c, d, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(d),
                nn.ReLU(inplace=True),
            ]
            c = d
        self.net = nn.Sequential(*layers)
        self.out_channels = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
