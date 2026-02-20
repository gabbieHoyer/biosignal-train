# src/biosignals/data/transforms/signal_ops.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from biosignals.data.types import Sample

try:
    from scipy.signal import butter, iirnotch, sosfiltfilt
except Exception:  # pragma: no cover
    butter = None  # type: ignore
    sosfiltfilt = None  # type: ignore
    iirnotch = None  # type: ignore


def _resolve_modalities(sample: Sample, modalities: Optional[Sequence[str]]) -> Sequence[str]:
    return list(sample.signals.keys()) if modalities is None else list(modalities)


def _get_fs(sample: Sample, modality: str, default_fs: float) -> float:
    # Prefer per-modality fs if present, otherwise global fs, otherwise default
    meta = sample.meta
    return float(meta.get(f"fs_{modality}", meta.get("fs", default_fs)))


def _tmin(sample: Sample) -> int:
    return min(int(x.shape[-1]) for x in sample.signals.values())


@dataclass
class EnsureFloat32:
    def __call__(self, sample: Sample) -> Sample:
        signals = {k: np.asarray(v, dtype=np.float32) for k, v in sample.signals.items()}
        return Sample(signals=signals, targets=sample.targets, meta=sample.meta)


@dataclass
class ZScorePerChannel:
    eps: float = 1e-6
    modalities: Optional[Sequence[str]] = None

    def __call__(self, sample: Sample) -> Sample:
        out: Dict[str, np.ndarray] = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)  # (C,T)
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True)
            out[m] = (x - mu) / (sd + float(self.eps))
        return Sample(signals=out, targets=sample.targets, meta=sample.meta)


@dataclass
class RandomCrop:
    window_size: int

    def __call__(self, sample: Sample) -> Sample:
        w = int(self.window_size)
        if w <= 0:
            return sample

        T = _tmin(sample)
        if w >= T:
            return sample

        start = int(np.random.randint(0, T - w + 1))
        signals = {m: x[..., start : start + w] for m, x in sample.signals.items()}
        meta = dict(sample.meta)
        meta["crop_start"] = start
        meta["crop_len"] = w
        return Sample(signals=signals, targets=sample.targets, meta=meta)


@dataclass
class CenterCrop:
    window_size: int

    def __call__(self, sample: Sample) -> Sample:
        w = int(self.window_size)
        if w <= 0:
            return sample

        T = _tmin(sample)
        if w >= T:
            return sample

        start = int((T - w) // 2)
        signals = {m: x[..., start : start + w] for m, x in sample.signals.items()}
        meta = dict(sample.meta)
        meta["crop_start"] = start
        meta["crop_len"] = w
        meta["crop_mode"] = "center"
        return Sample(signals=signals, targets=sample.targets, meta=meta)


@dataclass
class BandpassButter:
    """
    Apply a Butterworth bandpass filter (zero-phase via sosfiltfilt).

    Works per modality. Sampling rate comes from:
      meta["fs_{modality}"] or meta["fs"] or default_fs
    """

    low_hz: float = 0.5
    high_hz: float = 40.0
    order: int = 4
    default_fs: float = 500.0
    modalities: Optional[Sequence[str]] = None

    def __call__(self, sample: Sample) -> Sample:
        if butter is None or sosfiltfilt is None:
            raise ImportError(
                "scipy is required for BandpassButter (scipy.signal.butter, sosfiltfilt)."
            )

        out = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)  # (C,T)
            fs = _get_fs(sample, m, self.default_fs)

            nyq = 0.5 * fs
            lo = float(self.low_hz) / nyq
            hi = float(self.high_hz) / nyq

            # guard
            lo = max(lo, 1e-6)
            hi = min(hi, 0.999999)

            if lo >= hi:
                # degenerate request -> no-op
                continue

            sos = butter(int(self.order), [lo, hi], btype="bandpass", output="sos")
            y = sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)
            out[m] = y

        return Sample(signals=out, targets=sample.targets, meta=sample.meta)


@dataclass
class NotchFilter:
    """
    Apply an IIR notch filter (implemented via iirnotch + sosfiltfilt).

    Useful to remove line noise (50/60 Hz).
    """

    freq_hz: float = 60.0
    q: float = 30.0
    default_fs: float = 500.0
    modalities: Optional[Sequence[str]] = None

    def __call__(self, sample: Sample) -> Sample:
        if iirnotch is None or sosfiltfilt is None:
            raise ImportError(
                "scipy is required for NotchFilter (scipy.signal.iirnotch, sosfiltfilt)."
            )

        out = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)  # (C,T)
            fs = _get_fs(sample, m, self.default_fs)

            # iirnotch returns (b, a)
            b, a = iirnotch(w0=float(self.freq_hz), Q=float(self.q), fs=float(fs))

            # Convert (b,a) to SOS for stable filtfilt:
            # scipy doesn't directly provide sos here; simplest is use sosfiltfilt on a single section
            # by packing into shape (1,6): [b0,b1,b2,a0,a1,a2]
            if len(b) == 3 and len(a) == 3:
                sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]], dtype=np.float64)
                y = sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)
                out[m] = y

        return Sample(signals=out, targets=sample.targets, meta=sample.meta)


# # You’ll have dataset-specific configs picking the right transforms
# #  (e.g., bandpass for ECG/PPG, notch for EEG, motion magnitude features, etc.)
# #  without changing training code.
