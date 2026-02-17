# src/biosignals/utils/io.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, os.PathLike]


def ensure_parent(path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: PathLike, obj: Any, *, indent: int = 2) -> None:
    """
    Atomic-ish JSON write (write tmp then replace).
    """
    p = ensure_parent(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True)
    tmp.replace(p)


def append_jsonl(path: PathLike, record: Any) -> None:
    """
    Append a single JSON record to a .jsonl file.
    """
    p = ensure_parent(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
