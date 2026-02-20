# src/biosignals/models/backbones/tcn_segmenter.py
from __future__ import annotations

import torch
from torch import nn


class TCNEncoder1D(nn.Module):
    """
    Encoder only (stride=1):
      (B,C,T) -> (B,D,T)
    """

    def __init__(self, in_channels: int, width: int = 128, depth: int = 4) -> None:
        super().__init__()
        layers = []
        c = int(in_channels)
        w = int(width)
        for _ in range(int(depth)):
            layers.append(nn.Conv1d(c, w, kernel_size=7, padding=3, bias=False))
            layers.append(nn.BatchNorm1d(w))
            layers.append(nn.ReLU(inplace=True))
            c = w
        self.backbone = nn.Sequential(*layers)
        self.out_channels = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B,width,T)
