# src/biosignals/utils/paths.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]


def project_root() -> Path:
    """
    Best-effort project root finder: finds a parent containing `src/`.
    Works well for src-layout repos.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "src").exists():
            return p
    return Path.cwd().resolve()


def expand(path: PathLike, *, base: Optional[PathLike] = None) -> Path:
    """Expand ~ and env vars; if relative and base is provided, join to base."""
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not p.is_absolute() and base is not None:
        p = Path(base) / p
    return p.resolve()


def ensure_dir(path: PathLike) -> Path:
    """Create directory and return resolved path."""
    p = expand(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return time.strftime(fmt, time.localtime())


def increment_path(path: PathLike, *, sep: str = "_", exist_ok: bool = False) -> Path:
    """
    If `path` exists and exist_ok=False, append _2, _3, ...
    """
    p = expand(path)
    if exist_ok or not p.exists():
        return p

    base = str(p)
    for i in range(2, 10_000):
        cand = Path(f"{base}{sep}{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find free incremented path for: {p}")
