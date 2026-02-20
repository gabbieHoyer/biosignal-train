# src/biosignals/data/transforms/aug.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from biosignals.data.types import Sample


def _resolve_modalities(sample: Sample, modalities: Optional[Sequence[str]]) -> Sequence[str]:
    return list(sample.signals.keys()) if modalities is None else list(modalities)


def _copy_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return dict(meta)


def _get_T(sample: Sample, primary_modality: Optional[str]) -> int:
    if primary_modality is not None and primary_modality in sample.signals:
        return int(sample.signals[primary_modality].shape[-1])
    return int(next(iter(sample.signals.values())).shape[-1])


@dataclass
class RandomGaussianNoise:
    sigma: float = 0.01
    p: float = 0.5
    modalities: Optional[Sequence[str]] = None

    def __call__(self, sample: Sample) -> Sample:
        if self.p <= 0 or np.random.rand() >= self.p:
            return sample
        out = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)
            noise = np.random.normal(0.0, self.sigma, size=x.shape).astype(np.float32)
            out[m] = (x + noise).astype(np.float32, copy=False)
        return Sample(signals=out, targets=sample.targets, meta=sample.meta)


@dataclass
class RandomScale:
    min_scale: float = 0.9
    max_scale: float = 1.1
    per_channel: bool = True
    p: float = 0.5
    modalities: Optional[Sequence[str]] = None

    def __call__(self, sample: Sample) -> Sample:
        if self.p <= 0 or np.random.rand() >= self.p:
            return sample
        out = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)
            c = int(x.shape[0])
            if self.per_channel:
                s = np.random.uniform(self.min_scale, self.max_scale, size=(c, 1)).astype(
                    np.float32
                )
                out[m] = (x * s).astype(np.float32, copy=False)
            else:
                s = np.float32(np.random.uniform(self.min_scale, self.max_scale))
                out[m] = (x * s).astype(np.float32, copy=False)
        return Sample(signals=out, targets=sample.targets, meta=sample.meta)


@dataclass
class RandomTimeShift:
    max_shift: int = 25
    p: float = 0.5
    mode: str = "constant"  # "constant" or "wrap"
    fill_value: float = 0.0
    modalities: Optional[Sequence[str]] = None

    def __call__(self, sample: Sample) -> Sample:
        if self.p <= 0 or np.random.rand() >= self.p or self.max_shift <= 0:
            return sample
        shift = int(np.random.randint(-self.max_shift, self.max_shift + 1))
        if shift == 0:
            return sample

        out = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)
            if self.mode == "wrap":
                y = np.roll(x, shift=shift, axis=-1)
            else:
                y = np.full_like(x, np.float32(self.fill_value))
                if shift > 0:
                    y[..., shift:] = x[..., :-shift]
                else:
                    s = -shift
                    y[..., :-s] = x[..., s:]
            out[m] = y

        meta = _copy_meta(sample.meta)
        meta["time_shift"] = shift
        return Sample(signals=out, targets=sample.targets, meta=meta)


@dataclass
class RandomChannelDropout:
    drop_prob: float = 0.1
    p: float = 0.5
    drop_value: float = 0.0
    modalities: Optional[Sequence[str]] = None

    def __call__(self, sample: Sample) -> Sample:
        if self.p <= 0 or np.random.rand() >= self.p:
            return sample
        out = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)
            c = int(x.shape[0])
            drop = np.random.rand(c) < self.drop_prob
            if drop.any():
                x2 = x.copy()
                x2[drop, :] = np.float32(self.drop_value)
                out[m] = x2
        return Sample(signals=out, targets=sample.targets, meta=sample.meta)


@dataclass
class BernoulliTimeMask:
    """
    Generate a time mask and store it in meta[meta_key] as (T,) bool.
    Optionally apply to selected modalities.

    primary_modality ensures T is taken from that stream (important for multimodal dict ordering).
    """

    mask_ratio: float = 0.3
    p: float = 1.0
    apply: bool = False
    mask_value: float = 0.0
    meta_key: Optional[str] = "ssl_mask"
    primary_modality: Optional[str] = None
    modalities: Optional[Sequence[str]] = None
    union_with_existing: bool = True

    def __call__(self, sample: Sample) -> Sample:
        if self.p <= 0 or np.random.rand() >= self.p:
            return sample

        T = _get_T(sample, self.primary_modality)
        mask = np.random.rand(T) < float(self.mask_ratio)

        meta = sample.meta
        if self.meta_key is not None:
            meta2 = _copy_meta(sample.meta)
            if self.union_with_existing and (self.meta_key in meta2):
                prev = np.asarray(meta2[self.meta_key]).astype(bool, copy=False)
                if prev.shape == mask.shape:
                    mask = mask | prev
            meta2[self.meta_key] = mask.astype(bool)
            meta = meta2

        if not self.apply:
            return Sample(signals=sample.signals, targets=sample.targets, meta=meta)

        out = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)
            x2 = x.copy()
            x2[:, mask] = np.float32(self.mask_value)
            out[m] = x2

        return Sample(signals=out, targets=sample.targets, meta=meta)


@dataclass
class DeterministicBernoulliTimeMask:
    """
    Deterministic time mask based on sample.meta[seed_key] (usually "id").
    Useful for stable SSL validation.
    """

    mask_ratio: float = 0.3
    apply: bool = False
    mask_value: float = 0.0
    meta_key: str = "ssl_mask"
    seed_key: str = "id"
    primary_modality: Optional[str] = None
    modalities: Optional[Sequence[str]] = None
    union_with_existing: bool = True

    def __call__(self, sample: Sample) -> Sample:
        sid = str(sample.meta.get(self.seed_key, ""))
        seed = np.uint32(abs(hash(sid)) % (2**32))
        rng = np.random.default_rng(seed)

        T = _get_T(sample, self.primary_modality)
        mask = rng.random(T) < float(self.mask_ratio)

        meta2 = _copy_meta(sample.meta)
        if self.union_with_existing and (self.meta_key in meta2):
            prev = np.asarray(meta2[self.meta_key]).astype(bool, copy=False)
            if prev.shape == mask.shape:
                mask = mask | prev
        meta2[self.meta_key] = mask.astype(bool)

        if not self.apply:
            return Sample(signals=sample.signals, targets=sample.targets, meta=meta2)

        out = dict(sample.signals)
        for m in _resolve_modalities(sample, self.modalities):
            x = np.asarray(out[m], dtype=np.float32)
            x2 = x.copy()
            x2[:, mask] = np.float32(self.mask_value)
            out[m] = x2

        return Sample(signals=out, targets=sample.targets, meta=meta2)


# ****Important note**** about your dataset caching:
# because BiosignalDataset caches after transforms,
# do not enable cache_dir when using random augmentations
# (or you’ll freeze one random augmentation per sample index).
