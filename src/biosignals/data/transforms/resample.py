# src/biosignals/data/transforms/resample.py
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Optional, Sequence

import numpy as np
from biosignals.data.types import Sample

try:
    from scipy.signal import resample_poly
except Exception as e:  # pragma: no cover
    resample_poly = None  # type: ignore


def _copy_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(meta)


def _get_fs(meta: Dict[str, Any], modality: str, fs_key: str, default_fs: float) -> float:
    return float(meta.get(f"fs_{modality}", meta.get(fs_key, default_fs)))


def _pad_to_length(x: np.ndarray, T: int, pad_value: float) -> np.ndarray:
    """Pad/crop along last axis to exactly T."""
    cur = int(x.shape[-1])
    if cur == T:
        return x
    if cur > T:
        return x[..., :T]
    pad = T - cur
    return np.pad(x, pad_width=((0, 0), (0, pad)), mode="constant", constant_values=float(pad_value))


@dataclass
class ResampleToPrimary:
    """
    Resample all non-primary modalities to match the primary modality sampling rate.

    Common use case: PPG+ACC where fs_ppg != fs_acc.

    - Uses meta keys:
        fs_{modality} (preferred), else meta[fs_key], else default_fs
    - Optionally matches output length to primary (crop/pad), which is essential
      if later transforms assume same T across modalities (e.g. RandomCrop).

    Parameters:
      primary_modality: which modality defines target fs (e.g. "ppg")
      modalities: which modalities to resample; default = all except primary
      match_length: "primary" | "min" | "none"
      pad_value: used when padding to match length
    """
    primary_modality: str = "ppg"
    modalities: Optional[Sequence[str]] = None
    fs_key: str = "fs"
    default_fs: float = 100.0
    max_denominator: int = 1000
    match_length: str = "primary"  # "primary"|"min"|"none"
    pad_value: float = 0.0
    update_meta: bool = True
    store_orig_fs: bool = True

    def __call__(self, sample: Sample) -> Sample:
        if resample_poly is None:
            raise ImportError("scipy is required for ResampleToPrimary (scipy.signal.resample_poly).")

        if self.primary_modality not in sample.signals:
            return sample

        signals = dict(sample.signals)
        meta = _copy_meta(sample.meta)

        x_p = np.asarray(signals[self.primary_modality], dtype=np.float32)
        fs_p = _get_fs(meta, self.primary_modality, self.fs_key, self.default_fs)
        T_p = int(x_p.shape[-1])

        # Decide which modalities to resample
        mods = list(signals.keys()) if self.modalities is None else list(self.modalities)
        mods = [m for m in mods if m != self.primary_modality and m in signals]

        # Resample each modality to fs_p
        lengths_after: Dict[str, int] = {self.primary_modality: T_p}
        for m in mods:
            x = np.asarray(signals[m], dtype=np.float32)
            fs_m = _get_fs(meta, m, self.fs_key, self.default_fs)

            if self.store_orig_fs:
                meta[f"fs_{m}_orig"] = float(fs_m)

            # if already same fs, still allow length matching below
            if abs(fs_m - fs_p) > 1e-6 and fs_m > 0 and fs_p > 0:
                ratio = Fraction(fs_p / fs_m).limit_denominator(int(self.max_denominator))
                up, down = ratio.numerator, ratio.denominator
                y = resample_poly(x, up=up, down=down, axis=-1).astype(np.float32, copy=False)
            else:
                y = x

            signals[m] = y
            lengths_after[m] = int(y.shape[-1])

            if self.update_meta:
                meta[f"fs_{m}"] = float(fs_p)

        if self.update_meta:
            meta[f"fs_{self.primary_modality}"] = float(fs_p)
            meta["fs"] = float(fs_p)

        # Match lengths if requested (important for crop/mask transforms)
        if self.match_length == "primary":
            # crop/pad everyone to primary length
            signals[self.primary_modality] = _pad_to_length(x_p, T_p, self.pad_value)
            for m in mods:
                signals[m] = _pad_to_length(np.asarray(signals[m], dtype=np.float32), T_p, self.pad_value)
            meta["aligned_len"] = int(T_p)
            meta["aligned_to"] = self.primary_modality

        elif self.match_length == "min":
            Tmin = min(int(np.asarray(v).shape[-1]) for v in signals.values())
            for m, x in list(signals.items()):
                signals[m] = np.asarray(x, dtype=np.float32)[..., :Tmin]
            meta["aligned_len"] = int(Tmin)
            meta["aligned_to"] = "min"

        elif self.match_length == "none":
            pass
        else:
            raise ValueError("match_length must be one of: 'primary', 'min', 'none'")

        return Sample(signals=signals, targets=sample.targets, meta=meta)

