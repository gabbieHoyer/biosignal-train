# src/biosignals/models/backbones/resnet1d.py
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 7, s: int = 1) -> None:
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet1DEncoder(nn.Module):
    """
    Encoder only:
      (B,C,T) -> (B,D,T')
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = ConvBlock(int(in_channels), 64, k=11, s=2)
        self.enc = nn.Sequential(
            ConvBlock(64, 128, k=7, s=2),
            ConvBlock(128, 256, k=7, s=2),
            ConvBlock(256, 256, k=3, s=1),
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.enc(x)
        return x  # (B,256,T')
