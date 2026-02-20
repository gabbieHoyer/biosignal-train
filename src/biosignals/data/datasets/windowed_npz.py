# src/biosignals/data/datasets/windowed_npz.py
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from biosignals.data.datasets.base import BiosignalDataset
from biosignals.data.types import Sample

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _as_ct(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)[None, :]
    if x.ndim == 2:
        x = x.astype(np.float32, copy=False)
        # Heuristic transpose if (T,C)
        if x.shape[0] > 16 and x.shape[1] <= 16:
            return x.T
        return x
    raise ValueError(f"{name}: expected 1D or 2D array, got shape={x.shape}")


def _scalar(x: Any, default: float) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (float, int)):
        return float(x)
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr.item())
    if arr.ndim == 1 and arr.size >= 1:
        return float(arr.reshape(-1)[0].item())
    return float(default)


@dataclass
class _LRUSubjectCache:
    max_items: int = 8

    def __post_init__(self) -> None:
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Dict[str, np.ndarray]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > int(self.max_items):
            self._cache.popitem(last=False)


class WindowedNpzDataset(BiosignalDataset):
    """
    Raw dataset:
      - subject/record-level signals stored as NPZ under root/subjects/{npz_id}.npz
      - window manifest stored under root/{view} (parquet/csv), one row per example

    Option B:
      - No transforms here
      - No disk caching here

    Key improvement:
      - Allow `npz_col` (file id) to differ from `subject_col` (grouping id).
        This is important for datasets like MIT-BIH where record_id != subject_id group.
    """

    def __init__(
        self,
        root: str,
        split: str,
        *,
        view: str = "views/windows.parquet",
        subjects_dir: str = "subjects",
        modalities: Sequence[str] = ("ppg", "acc"),
        label_col: Optional[str] = "hr",
        split_col: str = "split",
        subject_col: str = "subject_id",
        npz_col: Optional[str] = None,
        start_col: str = "start_idx",
        end_col: str = "end_idx",
        id_col: Optional[str] = None,
        dropna_labels: bool = True,
        subject_cache_size: int = 8,
        extra_meta_cols: Sequence[str] = (),
        **kwargs,
    ) -> None:
        # Catch config drift early
        if len(kwargs) > 0:
            raise TypeError(
                f"WindowedNpzDataset got unexpected keys: {list(kwargs.keys())}. "
                "Remove wrapper-only settings like cache_dir/transform from the dataset config; "
                "they are handled in the datamodule wrappers now."
            )

        super().__init__(split=split)

        if pd is None:
            raise ImportError("WindowedNpzDataset requires pandas (and pyarrow for parquet).")

        self.root = Path(root)
        self.view_path = self.root / view
        self.subjects_dir = self.root / subjects_dir

        self.modalities = list(modalities)
        self.label_col = label_col
        self.split_col = split_col

        # subject_col is the grouping id stored in meta
        self.subject_col = subject_col

        # npz_col is the file id used to locate NPZ. Defaults to subject_col for backward compatibility.
        self.npz_col = npz_col or subject_col

        self.start_col = start_col
        self.end_col = end_col
        self.id_col = id_col
        self.dropna_labels = bool(dropna_labels)
        self.extra_meta_cols = list(extra_meta_cols)

        if not self.view_path.exists():
            raise FileNotFoundError(f"Missing view file: {self.view_path}")
        if not self.subjects_dir.exists():
            raise FileNotFoundError(f"Missing subjects dir: {self.subjects_dir}")

        if self.view_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.view_path)
        else:
            df = pd.read_csv(self.view_path)

        # Required columns
        for c in [self.subject_col, self.npz_col, self.start_col, self.end_col]:
            if c not in df.columns:
                raise KeyError(f"View is missing required column '{c}'. Found: {df.columns.tolist()}")

        if self.split_col in df.columns:
            df = df[df[self.split_col].astype(str) == str(split)].reset_index(drop=True)

        if self.label_col is not None:
            if self.label_col not in df.columns:
                raise KeyError(
                    f"label_col='{self.label_col}' not in view columns. Found: {df.columns.tolist()}"
                )
            if self.dropna_labels:
                s = df[self.label_col]
                mask = s.notna()
                try:
                    if pd.api.types.is_numeric_dtype(s):
                        mask = mask & (s.astype(float) == s.astype(float))
                        if pd.api.types.is_integer_dtype(s):
                            mask = mask & (s.astype(int) >= 0)
                except Exception:
                    pass
                df = df[mask].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(f"No rows left after filtering view={self.view_path} for split='{split}'")

        self.rows: List[Dict[str, Any]] = df.to_dict(orient="records")
        self._subj_cache = _LRUSubjectCache(max_items=int(subject_cache_size))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_subject_npz(self, npz_id: str) -> Dict[str, np.ndarray]:
        cached = self._subj_cache.get(npz_id)
        if cached is not None:
            return cached

        npz_path = self.subjects_dir / f"{npz_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing NPZ: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as z:
            payload: Dict[str, np.ndarray] = {k: z[k] for k in z.files}

        self._subj_cache.put(npz_id, payload)
        return payload

    def _load_raw(self, idx: int) -> Sample:
        r = self.rows[idx]

        subject_id = str(r[self.subject_col])  # grouping id
        npz_id = str(r[self.npz_col])          # file id

        i0 = int(r[self.start_col])
        i1 = int(r[self.end_col])
        if i1 <= i0:
            raise ValueError(f"Bad window indices for npz_id={npz_id}: start={i0}, end={i1}")

        subj = self._load_subject_npz(npz_id)

        signals: Dict[str, np.ndarray] = {}
        for m in self.modalities:
            if m not in subj:
                raise KeyError(f"NPZ {npz_id} missing modality key '{m}'. Keys={list(subj.keys())}")
            x = _as_ct(subj[m], name=m)
            if i1 > x.shape[-1]:
                raise IndexError(
                    f"Window end_idx={i1} exceeds signal length T={x.shape[-1]} for npz_id={npz_id}, modality={m}"
                )
            signals[m] = x[:, i0:i1]

        targets: Dict[str, Any] = {}
        if self.label_col is not None:
            targets["y"] = r.get(self.label_col, None)

        # Prefer explicit example id if provided
        if self.id_col is not None and self.id_col in r and r[self.id_col] is not None:
            sample_id = str(r[self.id_col])
        else:
            # Use npz_id to ensure uniqueness when subject_id groups multiple records
            sample_id = f"{npz_id}:{i0}:{i1}"

        fs = _scalar(subj.get("fs", subj.get("fs_out", None)), default=0.0)

        meta: Dict[str, Any] = {
            "id": sample_id,
            "subject_id": subject_id,
            "record_id": npz_id,
            "fs": float(fs),
            "start_idx": i0,
            "end_idx": i1,
        }

        # Helpful when grouping != file id
        if npz_id != subject_id:
            meta["npz_id"] = npz_id

        # Common optional columns (kept from your original)
        for opt in ["session_id", "session", "tsst", "ssst", "t_start_ms", "t_end_ms", "t_center_ms"]:
            if opt in r:
                meta[opt] = r[opt]

        # User-configurable extra meta passthrough
        for k in self.extra_meta_cols:
            if k in r and k not in meta:
                meta[k] = r[k]

        return Sample(signals=signals, targets=targets, meta=meta)


# ---------------------------------------------
