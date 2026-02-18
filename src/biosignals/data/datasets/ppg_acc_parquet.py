# src/biosignals/data/datasets/ppg_acc_parquet.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from biosignals.data.datasets.base import BiosignalDataset
from biosignals.data.types import Sample

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

"""
Example dataset implementations:

Example 2: multi-modal PPG + ACC

You'd emit signals={"ppg": (C,T), "acc": (C,T)} and the rest of the pipeline remains unchanged.

"""

def _as_float32(a: Any) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype == object:
        # common when parquet stores nested lists
        arr = np.asarray(list(arr), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _to_ct(x: Any, *, name: str) -> np.ndarray:
    """
    Coerce array-like into float32 shape (C, T).
    Accepts (T,), (C,T), or (T,C). Uses a heuristic transpose if needed.
    """
    arr = _as_float32(x)

    if arr.ndim == 1:
        return arr[None, :]  # (1,T)

    if arr.ndim == 2:
        # Heuristic: channels are usually small (<=16). If first dim looks like time, transpose.
        if arr.shape[0] > 16 and arr.shape[1] <= 16:
            return arr.T
        return arr

    raise ValueError(f"Unsupported {name} shape {arr.shape}; expected 1D or 2D.")


def _stack_time_columns(df: "pd.DataFrame", cols: List[str]) -> np.ndarray:
    """
    Stack columns that represent channels with rows as time:
      df[cols] -> (T, C) -> transpose to (C, T)
    """
    mat = df[cols].to_numpy(dtype=np.float32, copy=False)  # (T, C)
    return mat.T  # (C, T)


def _find_prefixed_cols(columns: List[str], prefixes: Tuple[str, ...]) -> List[str]:
    cols = []
    for c in columns:
        cl = c.lower()
        if any(cl.startswith(p) for p in prefixes):
            cols.append(c)
    return cols


class PpgAccParquetDataset(BiosignalDataset):
    """
    Multi-modal PPG + ACC dataset stored as per-sample parquet.

    Expected structure:
      root/
        train/*.parquet
        val/*.parquet

    Emits:
      Sample(
        signals={"ppg": (C,T), "acc": (C,T)},
        targets={"y": ...} if present,
        meta={"id": ..., "fs": float, "fs_ppg": float, "fs_acc": float}
      )
    """

    def __init__(self, root: str, split: str, transform=None, cache_dir=None) -> None:
        super().__init__(split=split, transform=transform, cache_dir=cache_dir)
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.parquet"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .parquet files found under: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_raw(self, idx: int) -> Sample:
        if pd is None:
            raise ImportError(
                "Reading parquet requires pandas + a parquet engine (pyarrow or fastparquet). "
                "Install: pip install pandas pyarrow"
            )

        fp = self.files[idx]
        df = pd.read_parquet(fp)

        # ---- Load PPG ----
        ppg: Optional[np.ndarray] = None
        acc: Optional[np.ndarray] = None

        cols = list(df.columns)

        # Case 1: blob columns in a single row (or first row)
        if "ppg" in cols and "acc" in cols:
            # If multiple rows and these are numeric columns, treat as time columns.
            if len(df) > 1 and np.issubdtype(df["ppg"].dtype, np.number):
                ppg = _to_ct(df["ppg"].to_numpy(dtype=np.float32, copy=False), name="ppg")
            else:
                row0 = df.iloc[0].to_dict()
                ppg = _to_ct(row0["ppg"], name="ppg")

            if len(df) > 1 and np.issubdtype(df["acc"].dtype, np.number):
                acc = _to_ct(df["acc"].to_numpy(dtype=np.float32, copy=False), name="acc")
            else:
                row0 = df.iloc[0].to_dict()
                acc = _to_ct(row0["acc"], name="acc")

            row_meta = df.iloc[0].to_dict() if len(df) > 0 else {}
        else:
            # Case 2: columns represent channels with rows as time
            # Try common patterns
            ppg_cols = _find_prefixed_cols(cols, prefixes=("ppg_", "ppgch", "ppg0", "ppg1", "ppg2"))
            # If none found, allow a single "ppg" column handled above; here it doesn't exist.
            if len(ppg_cols) == 0:
                # also accept columns like ppg0, ppg1, ppg2
                ppg_cols = [c for c in cols if c.lower().startswith("ppg")]

            acc_cols = []
            # Prefer explicit axis naming
            for cand in ("acc_x", "acc_y", "acc_z", "ax", "ay", "az"):
                if cand in [c.lower() for c in cols]:
                    # map back to original case
                    for c in cols:
                        if c.lower() == cand:
                            acc_cols.append(c)
            if len(acc_cols) == 0:
                # fallback: any "acc" prefixed columns
                acc_cols = [c for c in cols if c.lower().startswith("acc")]

            if len(ppg_cols) == 0 or len(acc_cols) == 0:
                raise KeyError(
                    f"{fp.name}: could not infer PPG/ACC columns. "
                    f"Need either ('ppg','acc') blob columns or per-channel columns. "
                    f"Found columns: {cols}"
                )

            # Sort columns for stable ordering
            ppg_cols = sorted(ppg_cols)
            acc_cols = sorted(acc_cols)

            ppg = _stack_time_columns(df, ppg_cols)  # (C,T)
            acc = _stack_time_columns(df, acc_cols)  # (C,T)

            row_meta = df.iloc[0].to_dict() if len(df) > 0 else {}

        assert ppg is not None and acc is not None

        # ---- Targets (optional) ----
        targets: Dict[str, Any] = {}
        if "y" in row_meta and row_meta["y"] is not None:
            targets["y"] = row_meta["y"]
        elif "label" in row_meta and row_meta["label"] is not None:
            targets["y"] = row_meta["label"]

        # ---- Meta ----
        fs_ppg = float(row_meta.get("fs_ppg", row_meta.get("fs", 100.0)))
        fs_acc = float(row_meta.get("fs_acc", row_meta.get("fs", 50.0)))

        # IMPORTANT: keep meta["fs"] a float for compatibility with single-modality pipelines
        meta = {
            "id": fp.stem,
            "fs": fs_ppg,
            "fs_ppg": fs_ppg,
            "fs_acc": fs_acc,
        }

        return Sample(signals={"ppg": ppg, "acc": acc}, targets=targets, meta=meta)
