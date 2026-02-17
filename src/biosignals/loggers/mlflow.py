# src/biosignals/loggers/mlflow.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterable, Tuple
import json

from biosignals.loggers.base import ExperimentLogger


def _flatten_dict(
    d: Dict[str, Any],
    *,
    prefix: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


def _safe_param_value(v: Any, *, max_len: int = 500) -> str:
    # MLflow params must be string-ish.
    if v is None:
        s = "null"
    elif isinstance(v, (str, int, float, bool)):
        s = str(v)
    else:
        # lists/dicts/etc
        try:
            s = json.dumps(v, sort_keys=True)
        except Exception:
            s = str(v)

    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


@dataclass
class MLflowLogger(ExperimentLogger):
    tracking_uri: str
    experiment_name: str = "biosignals"
    run_name: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None

    _active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        import mlflow  # local import

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        self._active = True

        if self.tags:
            # tags must be str->str typically; best-effort stringify
            safe_tags = {str(k): _safe_param_value(v) for k, v in self.tags.items()}
            mlflow.set_tags(safe_tags)

    def log_hparams(self, params: Dict[str, Any]) -> None:
        if not self._active:
            return
        import mlflow

        flat = _flatten_dict(params)
        for k, v in flat.items():
            # MLflow param keys should be reasonably small; normalize slashes
            key = str(k).replace("/", ".")
            mlflow.log_param(key, _safe_param_value(v))

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        if not self._active:
            return
        import mlflow
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str, *, name: Optional[str] = None) -> None:
        if not self._active:
            return
        import mlflow
        mlflow.log_artifact(path)

    def finish(self) -> None:
        if self._active:
            import mlflow
            mlflow.end_run()
            self._active = False
