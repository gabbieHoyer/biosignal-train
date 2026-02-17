# src/biosignals/models/architectures/multimodal_fusion.py
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from biosignals.models.fusion.late_fusion import ConcatMLPFusion


def _masked_mean_pool(z: torch.Tensor, mask_t: Optional[torch.Tensor]) -> torch.Tensor:
    # z: (B,D,T)
    if mask_t is None:
        return z.mean(dim=-1)
    m = mask_t.to(device=z.device).bool().unsqueeze(1).float()  # (B,1,T)
    denom = m.sum(dim=-1).clamp_min(1.0)
    return (z * m).sum(dim=-1) / denom


def _resample_mask(mask_t: torch.Tensor, T_out: int) -> torch.Tensor:
    m = mask_t.float().unsqueeze(1)  # (B,1,T)
    m2 = F.interpolate(m, size=int(T_out), mode="nearest")
    return (m2.squeeze(1) > 0.5)


class LateFusionClassifier(nn.Module):
    """
    End-to-end multimodal classifier:
      per-modality encoders -> pooled embeddings -> concat -> fusion MLP -> logits

    Encoders may output:
      - (B,D) OR (B,D,T)
    """
    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        emb_dim: int,
        num_classes: int,
        fusion_hidden: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.modalities = list(encoders.keys())
        self.emb_dim = int(emb_dim)

        self.fusion = ConcatMLPFusion(
            in_dim=self.emb_dim * len(self.modalities),
            out_dim=self.emb_dim,
            hidden_dim=int(fusion_hidden),
            dropout=float(dropout),
        )
        self.head = nn.Linear(self.emb_dim, int(num_classes))

    def encode(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        mask_by_modality = None
        if meta is not None and isinstance(meta.get("mask_by_modality", None), dict):
            mask_by_modality = meta["mask_by_modality"]

        any_tensor = next(iter(signals.values()))
        B = int(any_tensor.shape[0])
        device = any_tensor.device
        dtype = any_tensor.dtype

        embs = []
        for m in self.modalities:
            if m not in signals:
                embs.append(torch.zeros((B, self.emb_dim), device=device, dtype=dtype))
                continue

            x = signals[m]  # (B,C,T)
            z = self.encoders[m](x)

            if z.ndim == 3:
                mask_t = None
                if mask_by_modality is not None and m in mask_by_modality:
                    mask_t = mask_by_modality[m]
                elif meta is not None and meta.get("mask", None) is not None:
                    mask_t = meta["mask"]
                if mask_t is not None and z.shape[-1] != mask_t.shape[-1]:
                    mask_t = _resample_mask(mask_t.to(device=device).bool(), z.shape[-1])
                embs.append(_masked_mean_pool(z, mask_t))
            elif z.ndim == 2:
                embs.append(z)
            else:
                raise ValueError(f"Encoder for modality '{m}' returned shape {tuple(z.shape)}; expected (B,D) or (B,D,T).")

        z_cat = torch.cat(embs, dim=-1)     # (B, M*D)
        z_fused = self.fusion(z_cat)        # (B, D)
        return z_fused

    def forward(self, signals: Dict[str, torch.Tensor], meta: Optional[dict] = None) -> torch.Tensor:
        z = self.encode(signals, meta)
        return self.head(z)  # (B,K)
