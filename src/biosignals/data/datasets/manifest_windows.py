# src/biosignals/data/datasets/manifest_windows.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from biosignals.data.datasets.base import BiosignalDataset
from biosignals.data.types import Sample

"""
A manifest-driven dataset skeleton:

This is the “reuse everywhere” dataset:
    EEG/ECG/PPG/ACC become different manifest builders + readers,
    not different trainers.
"""


@dataclass
class NpySignalReader:
    """
    Minimal reader for (C,T) stored as .npy.
    Replace/extend with WFDB/EDF/HDF5 readers as needed.
    """

    def read(self, path: str, start: int, length: int) -> np.ndarray:
        x = np.load(path, mmap_mode="r")  # expects (C,T)
        return np.asarray(x[:, start : start + length], dtype=np.float32)


class ManifestWindowDataset(BiosignalDataset):
    def __init__(
        self,
        records_path: str,
        windows_path: str,
        split: str,
    ) -> None:
        super().__init__(split=split)
        self.records = pd.read_parquet(records_path)
        self.windows = pd.read_parquet(windows_path)
        self.windows = self.windows[self.windows["split"] == split].reset_index(drop=True)

        # index records by record_id for fast join
        self.records = self.records.set_index("record_id", drop=False)

        self.reader = NpySignalReader()

    def __len__(self) -> int:
        return len(self.windows)

    def _load_raw(self, idx: int) -> Sample:
        row = self.windows.iloc[idx]
        record_id = str(row["record_id"])
        start = int(row["start"])
        length = int(row["length"])

        r = self.records.loc[record_id]
        paths = json.loads(r["paths_json"])
        fs_map = json.loads(r["fs_json"])

        signals: Dict[str, np.ndarray] = {}
        for modality, path in paths.items():
            signals[modality] = self.reader.read(path, start=start, length=length)

        targets = json.loads(row["target_json"])
        meta = {
            "id": str(row["example_id"]),
            "record_id": record_id,
            "subject_id": str(row["subject_id"]),
            "fs": float(list(fs_map.values())[0]),  # or keep dict if you want
            "fs_map": fs_map,
            "start": start,
            "length": length,
        }
        return Sample(signals=signals, targets=targets, meta=meta)
