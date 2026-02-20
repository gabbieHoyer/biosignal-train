# src/biosignals/data/transforms/compose.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from biosignals.data.types import Sample

Transform = Callable[[Sample], Sample]


@dataclass
class Compose:
    transforms: Iterable[Transform]

    def __post_init__(self) -> None:
        # materialize generator / ListConfig etc.
        self.transforms = list(self.transforms)
        for i, t in enumerate(self.transforms):
            if not callable(t):
                raise TypeError(
                    f"Compose: transforms[{i}] is not callable. " f"type={type(t)} value={t}"
                )

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample
