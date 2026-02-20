# src/biosignals/data/collate.py
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

import torch

from biosignals.data.types import Sample


def pad_stack_ct(
    xs: List[torch.Tensor],
    pad_value: float = 0.0,
    *,
    tmax: Optional[int] = None,
    pad_to_multiple: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads list of (C, T_i) tensors to (B, C, T_max).

    Returns:
      x_pad:   (B, C, T_max)
      lengths: (B,) long
      mask:    (B, T_max) bool, True for valid (non-padded) positions.

    Extra options:
      - tmax: pad to at least this length (must be >= max(T_i))
      - pad_to_multiple: round T_max up to a multiple (useful for patching)
    """
    if len(xs) == 0:
        # This shouldn't happen in normal DataLoader usage, but keep it safe.
        empty = torch.empty((0, 0, 0), dtype=torch.float32)
        lengths = torch.empty((0,), dtype=torch.long)
        mask = torch.empty((0, 0), dtype=torch.bool)
        return empty, lengths, mask

    # Basic validation
    c0 = int(xs[0].shape[0])
    for i, x in enumerate(xs):
        if x.ndim != 2:
            raise ValueError(
                f"pad_stack_ct expects (C,T) tensors; got shape {tuple(x.shape)} at index {i}"
            )
        if int(x.shape[0]) != c0:
            raise ValueError(
                f"All tensors must share C. Got C={int(x.shape[0])} vs C0={c0} at index {i}"
            )

    lengths = torch.tensor([int(x.shape[-1]) for x in xs], dtype=torch.long)
    tmax0 = int(lengths.max().item()) if len(xs) else 0

    T = tmax0
    if tmax is not None:
        if int(tmax) < tmax0:
            raise ValueError(f"tmax ({tmax}) must be >= max length ({tmax0})")
        T = int(tmax)

    if pad_to_multiple is not None and int(pad_to_multiple) > 1:
        m = int(pad_to_multiple)
        T = int(math.ceil(T / m) * m)

    b = len(xs)
    c = c0
    dtype = xs[0].dtype

    x_pad = torch.full((b, c, T), float(pad_value), dtype=dtype)
    mask = torch.zeros((b, T), dtype=torch.bool)

    for i, x in enumerate(xs):
        t = int(x.shape[-1])
        x_pad[i, :, :t] = x
        mask[i, :t] = True

    return x_pad, lengths, mask


def pad_labels_t(
    ys: List[torch.Tensor],
    tmax: int,
    ignore_index: int,
) -> torch.Tensor:
    """
    Pads list of (T_i,) integer label sequences to (B, T_max) with ignore_index.
    """
    b = len(ys)
    y_pad = torch.full((b, int(tmax)), int(ignore_index), dtype=torch.long)
    for i, y in enumerate(ys):
        t = int(y.shape[-1])
        y_pad[i, :t] = y
    return y_pad


def pad_labels_kt(
    ys: List[torch.Tensor],
    tmax: int,
) -> torch.Tensor:
    """
    Pads list of (K, T_i) float label sequences to (B, K, T_max) with zeros.
    """
    b = len(ys)
    k = int(ys[0].shape[0])
    y_pad = torch.zeros((b, k, int(tmax)), dtype=torch.float32)
    for i, y in enumerate(ys):
        t = int(y.shape[-1])
        y_pad[i, :, :t] = y
    return y_pad


def pad_stack_modalities(
    samples: List[Sample],
    *,
    modalities: Optional[Sequence[str]] = None,
    pad_value: float = 0.0,
    shared_tmax: bool = True,
    tmax: Optional[int] = None,
    pad_to_multiple: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Convenience helper for multimodal datasets.

    Returns:
      signals[m]  -> (B, C_m, T)
      lengths[m]  -> (B,)
      masks[m]    -> (B, T)

    If shared_tmax=True:
      - all modalities are padded to the same T (max over modalities), which is useful
        when your model expects time-aligned modalities.

    Note:
      - This does NOT resample. If modalities have different sampling rates,
        you should resample/align them in transforms first.
    """
    if len(samples) == 0:
        return {}, {}, {}

    mods = list(samples[0].signals.keys()) if modalities is None else list(modalities)

    # Determine global T if we want shared padding length
    global_tmax0 = 0
    if shared_tmax:
        for m in mods:
            Ts = [int(s.signals[m].shape[-1]) for s in samples]
            global_tmax0 = max(global_tmax0, max(Ts))
        global_T = global_tmax0
        if tmax is not None:
            if int(tmax) < global_tmax0:
                raise ValueError(f"tmax ({tmax}) must be >= global max length ({global_tmax0})")
            global_T = int(tmax)
        if pad_to_multiple is not None and int(pad_to_multiple) > 1:
            mm = int(pad_to_multiple)
            global_T = int(math.ceil(global_T / mm) * mm)
    else:
        global_T = None

    signals: Dict[str, torch.Tensor] = {}
    lengths: Dict[str, torch.Tensor] = {}
    masks: Dict[str, torch.Tensor] = {}

    for m in mods:
        xs = [torch.from_numpy(s.signals[m]) for s in samples]  # (C,T)
        x_pad, lens, mask = pad_stack_ct(
            xs,
            pad_value=pad_value,
            tmax=global_T if shared_tmax else tmax,
            pad_to_multiple=pad_to_multiple,
        )
        signals[m] = x_pad
        lengths[m] = lens
        masks[m] = mask

    return signals, lengths, masks
