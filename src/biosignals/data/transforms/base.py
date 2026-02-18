# src/biosignals/data/transforms/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from biosignals.data.types import Sample

Transform = Callable[[Sample], Sample]


def _copy_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    # Sample is frozen, but meta dict is mutable; copy when writing.
    return dict(meta)


@dataclass
class Identity:
    """No-op transform."""
    def __call__(self, sample: Sample) -> Sample:
        return sample


@dataclass
class Lambda:
    """Wrap an arbitrary function(sample)->sample."""
    fn: Transform

    def __call__(self, sample: Sample) -> Sample:
        return self.fn(sample)


@dataclass
class RandomApply:
    """Apply transform with probability p."""
    transform: Transform
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if float(self.p) <= 0:
            return sample
        if np.random.rand() < float(self.p):
            return self.transform(sample)
        return sample


@dataclass
class OneOf:
    """
    Randomly choose one transform (optionally weighted) and apply it.
    """
    transforms: Sequence[Transform]
    weights: Optional[Sequence[float]] = None
    p: float = 1.0

    def __call__(self, sample: Sample) -> Sample:
        if len(self.transforms) == 0:
            return sample
        if float(self.p) < 1.0 and np.random.rand() >= float(self.p):
            return sample
        idx = int(np.random.choice(len(self.transforms), p=self._norm_weights()))
        return self.transforms[idx](sample)

    def _norm_weights(self) -> Optional[np.ndarray]:
        if self.weights is None:
            return None
        w = np.asarray(self.weights, dtype=np.float64)
        w = np.clip(w, 0, None)
        s = float(w.sum())
        if s <= 0:
            return None
        return w / s


@dataclass
class RenameModality:
    """
    Rename a modality key in sample.signals (e.g., "ecg" -> "main").
    """
    src: str
    dst: str
    overwrite: bool = True
    drop_src: bool = True

    def __call__(self, sample: Sample) -> Sample:
        if self.src not in sample.signals:
            return sample
        signals = dict(sample.signals)
        if (self.dst in signals) and (not self.overwrite):
            return sample
        signals[self.dst] = signals[self.src]
        if self.drop_src:
            signals.pop(self.src, None)
        return Sample(signals=signals, targets=sample.targets, meta=sample.meta)


@dataclass
class KeepModalities:
    """
    Keep only a subset of modalities. Safe as long as it is applied consistently
    to every sample (so collate sees stable keys).
    """
    modalities: Sequence[str]

    def __call__(self, sample: Sample) -> Sample:
        keep = {m: sample.signals[m] for m in self.modalities if m in sample.signals}
        return Sample(signals=keep, targets=sample.targets, meta=sample.meta)


@dataclass
class AddMeta:
    """Attach a constant key/value to sample.meta."""
    key: str
    value: Any

    def __call__(self, sample: Sample) -> Sample:
        meta = _copy_meta(sample.meta)
        meta[self.key] = self.value
        return Sample(signals=sample.signals, targets=sample.targets, meta=meta)
