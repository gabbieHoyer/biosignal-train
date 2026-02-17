# src/biosignals/loggers/base.py
from __future__ import annotations

from typing import Any, Dict, Optional


class ExperimentLogger:
    def log_hparams(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str, *, name: Optional[str] = None) -> None:
        pass

    def finish(self) -> None:
        pass


class NoopLogger(ExperimentLogger):
    pass
