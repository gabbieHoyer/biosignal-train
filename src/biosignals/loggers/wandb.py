# src/biosignals/loggers/wandb.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from biosignals.loggers.base import ExperimentLogger


@dataclass
class WandbLogger(ExperimentLogger):
    project: str = "biosignals"
    name: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    tags: Sequence[str] = field(default_factory=list)

    # where wandb writes local files
    dir: Optional[str] = None

    # wandb modes: "online", "offline", "disabled"
    mode: str = "online"

    _run: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import wandb  # local import

        self._run = wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            group=self.group,
            job_type=self.job_type,
            tags=list(self.tags) if self.tags is not None else None,
            dir=self.dir,
            mode=self.mode,
        )

    def log_hparams(self, params: Dict[str, Any]) -> None:
        if self._run is None:
            return
        # wandb.config expects json-serializable
        self._run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        if self._run is None:
            return
        import wandb
        wandb.log(metrics, step=step)

    def log_artifact(self, path: str, *, name: Optional[str] = None) -> None:
        if self._run is None:
            return
        import wandb
        # simplest: upload file
        wandb.save(path)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
