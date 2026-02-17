# src/biosignals/utils/logging.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def is_rank_zero() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"
    try:
        return int(rank) == 0
    except ValueError:
        return True


def configure_logging(
    *,
    level: Union[int, str] = "INFO",
    log_file: Optional[PathLike] = None,
    fmt: str = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Add console + optional file logging.

    Important:
      - Does NOT force-reset Hydra’s logging configuration.
      - Adds a FileHandler if log_file is provided (and avoids duplicates).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Ensure we have a stream handler
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    # Optional file handler
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)

        # avoid duplicate file handlers to same path
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if Path(h.baseFilename).resolve() == p.resolve():
                        return
                except Exception:
                    pass

        fh = logging.FileHandler(p, mode="a")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_logger(name: str = "biosignals") -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------
