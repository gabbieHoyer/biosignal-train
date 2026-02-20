# src/biosignals/cli/eval.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from biosignals.config.schema import validate_cfg
from biosignals.data.datamodule import make_split_loader
from biosignals.engine.loops import batch_to_device, evaluate_one_epoch
from biosignals.eval.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    nll_from_logits,
    softmax_np,
)
from biosignals.loggers.base import NoopLogger
from biosignals.utils.distributed import (
    cleanup_distributed,
    get_rank,
    init_distributed,
    is_distributed,
    is_main_process,
)
from biosignals.utils.reproducibility import seed_everything, set_float32_matmul_precision

try:
    import pandas as pd
except Exception:
    pd = None

log = logging.getLogger("biosignals")


def _extract_logits(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in ("logits", "pred", "yhat", "out"):
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v
    raise TypeError(f"Model output type not supported for prediction extraction: {type(out)}")


def _load_checkpoint_into_model(
    model: torch.nn.Module, ckpt_path: Path, strict: bool = True
) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # common checkpoint layouts
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt):
        # might already be a state_dict
        state = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format at {ckpt_path}")

    # try direct load
    try:
        missing, unexpected = model.load_state_dict(state, strict=strict)
        return {"missing": missing, "unexpected": unexpected, "ckpt_keys": list(state.keys())[:5]}
    except RuntimeError:
        # try stripping "module." prefix (DDP-saved)
        stripped = {}
        for k, v in state.items():
            nk = k[len("module.") :] if k.startswith("module.") else k
            stripped[nk] = v
        missing, unexpected = model.load_state_dict(stripped, strict=strict)
        return {"missing": missing, "unexpected": unexpected, "note": "stripped module. prefix"}


@torch.no_grad()
def _collect_logits_and_labels(
    model: torch.nn.Module,
    loader,
    *,
    device: str,
) -> Dict[str, Any]:
    model.eval()

    ids: list[str] = []
    subject_ids: list[Optional[str]] = []
    record_ids: list[Optional[str]] = []

    logits_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for batch in loader:
        batch = batch_to_device(batch, device)

        out = model(batch.signals, batch.meta)
        logits = _extract_logits(out).detach().float().cpu().numpy()

        y = batch.targets.get("y", None)
        if not torch.is_tensor(y):
            raise KeyError("Batch targets missing tensor 'y' (labels required for eval metrics).")
        y_np = y.detach().cpu().numpy()

        batch_ids = batch.meta.get("ids", None)
        if batch_ids is None:
            # fallback
            batch_ids = [f"idx_{len(ids)+i}" for i in range(int(logits.shape[0]))]

        ids.extend([str(x) for x in batch_ids])

        sids = batch.meta.get("subject_ids", [None] * int(logits.shape[0]))
        rids = batch.meta.get("record_ids", [None] * int(logits.shape[0]))
        subject_ids.extend([None if x is None else str(x) for x in sids])
        record_ids.extend([None if x is None else str(x) for x in rids])

        logits_list.append(logits.astype(np.float32, copy=False))
        y_list.append(y_np)

    logits_all = (
        np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, 0), dtype=np.float32)
    )
    y_all = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), dtype=np.int64)

    return {
        "ids": ids,
        "subject_ids": subject_ids,
        "record_ids": record_ids,
        "logits": logits_all,
        "y": y_all,
    }


def _ddp_gather_object(obj: Any) -> Any:
    if not is_distributed():
        return obj
    import torch.distributed as dist

    world = dist.get_world_size()
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, obj)
    return gathered


def _merge_gathered(packs: list[Dict[str, Any]]) -> Dict[str, Any]:
    ids: list[str] = []
    subject_ids: list[Optional[str]] = []
    record_ids: list[Optional[str]] = []
    logits: list[np.ndarray] = []
    y: list[np.ndarray] = []

    for p in packs:
        ids.extend(p["ids"])
        subject_ids.extend(p.get("subject_ids", [None] * len(p["ids"])))
        record_ids.extend(p.get("record_ids", [None] * len(p["ids"])))
        logits.append(np.asarray(p["logits"], dtype=np.float32))
        y.append(np.asarray(p["y"]))

    return {
        "ids": ids,
        "subject_ids": subject_ids,
        "record_ids": record_ids,
        "logits": np.concatenate(logits, axis=0) if logits else np.zeros((0, 0), dtype=np.float32),
        "y": np.concatenate(y, axis=0) if y else np.zeros((0,), dtype=np.int64),
    }


def _save_predictions_parquet(
    out_path: Path, pack: Dict[str, Any], *, temperature: Optional[float] = None
) -> None:
    if pd is None:
        raise ImportError("Saving predictions requires pandas+pyarrow.")

    logits = np.asarray(pack["logits"], dtype=np.float32)
    y = np.asarray(pack["y"])
    ids = pack["ids"]
    subject_ids = pack.get("subject_ids", [None] * len(ids))
    record_ids = pack.get("record_ids", [None] * len(ids))

    if temperature is not None:
        logits_used = apply_temperature(logits, float(temperature))
    else:
        logits_used = logits

    probs = softmax_np(logits_used)
    pred = probs.argmax(axis=1)

    df = pd.DataFrame(
        {
            "id": ids,
            "subject_id": subject_ids,
            "record_id": record_ids,
            "y_true": y.astype(np.int64, copy=False),
            "y_pred": pred.astype(np.int64, copy=False),
            "p_max": probs.max(axis=1).astype(np.float32, copy=False),
        }
    )
    # add per-class prob columns
    k = int(probs.shape[1]) if probs.ndim == 2 else 0
    for j in range(k):
        df[f"p_{j}"] = probs[:, j].astype(np.float32, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


@hydra.main(config_path="../../../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = validate_cfg(cfg, strict=False, lock_struct=False)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    init_distributed(backend=cfg.dist.backend)

    seed_everything(int(cfg.seed) + get_rank(), deterministic=True)
    set_float32_matmul_precision("high")

    use_cuda = torch.cuda.is_available()
    device = f"cuda:{local_rank}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()

    try:
        task = instantiate(cfg.task)
        model = instantiate(cfg.model)
        data_cfg = instantiate(cfg.data)

        # Resolve checkpoint path
        eval_cfg = cfg.get("eval", {})
        ckpt_path = eval_cfg.get("ckpt_path", None)
        run_dir_hint = eval_cfg.get("run_dir", None)
        ckpt_name = str(eval_cfg.get("ckpt_name", "best.pt"))
        strict = bool(eval_cfg.get("strict", True))

        if ckpt_path is None or str(ckpt_path).strip() == "":
            if run_dir_hint is None:
                raise ValueError(
                    "eval.ckpt_path is required (or provide eval.run_dir to auto-find checkpoints/)."
                )
            cand = Path(str(run_dir_hint)).expanduser().resolve() / "checkpoints" / ckpt_name
            ckpt_path = str(cand)

        ckpt_path_p = Path(str(ckpt_path)).expanduser().resolve()
        if not ckpt_path_p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path_p}")

        info = _load_checkpoint_into_model(model, ckpt_path_p, strict=strict)
        if is_main_process():
            log.info("Loaded checkpoint: %s | info=%s", str(ckpt_path_p), info)

        # Wrap DDP for distributed eval
        if is_distributed():
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if use_cuda else None,
                output_device=local_rank if use_cuda else None,
            )
        else:
            model.to(device)

        # External logger (rank0 only)
        exp_logger = NoopLogger()
        if "logger" in cfg and cfg.logger is not None and is_main_process():
            exp_logger = instantiate(cfg.logger)
            exp_logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

        split = str(eval_cfg.get("split", "test"))
        loader = make_split_loader(
            dataset_cfg=cfg.dataset,
            transforms_cfg=cfg.transforms,
            task=task,
            data_cfg=data_cfg,
            split=split,
        )

        # 1) aggregated task metrics (DDP-safe)
        avg = evaluate_one_epoch(model=model, task=task, loader=loader, device=device)
        if is_main_process():
            log.info("[eval/%s] %s", split, avg)
            exp_logger.log_metrics({f"eval/{split}/{k}": float(v) for k, v in avg.items()}, step=0)

        # 2) optional: save per-example predictions + calibration
        save_preds = bool(eval_cfg.get("save_predictions", True))
        do_calib = bool(eval_cfg.get("calibration", {}).get("enable", True))

        if save_preds or do_calib:
            pack_local = _collect_logits_and_labels(model, loader, device=device)
            gathered = _ddp_gather_object(pack_local)
            if is_distributed():
                if not isinstance(gathered, list):
                    raise RuntimeError("DDP gather returned unexpected type.")
                pack = _merge_gathered(gathered)
            else:
                pack = pack_local

            if is_main_process():
                # base predictions (uncalibrated)
                preds_path = run_dir / f"predictions_{split}.parquet"
                if save_preds:
                    _save_predictions_parquet(preds_path, pack, temperature=None)
                    log.info("Wrote predictions: %s", str(preds_path))
                    exp_logger.log_artifact(str(preds_path), name=f"predictions_{split}")

                # temperature scaling (fit on val, apply to eval split)
                if do_calib:
                    calib_cfg = eval_cfg.get("calibration", {})
                    fit_split = str(calib_cfg.get("fit_split", "val"))
                    n_bins = int(calib_cfg.get("n_bins", 15))

                    # Fit only if fit_split exists in dataset config
                    if fit_split in cfg.dataset and cfg.dataset.get(fit_split) is not None:
                        fit_loader = make_split_loader(
                            dataset_cfg=cfg.dataset,
                            transforms_cfg=cfg.transforms,
                            task=task,
                            data_cfg=data_cfg,
                            split=fit_split,
                        )
                        fit_pack_local = _collect_logits_and_labels(
                            model, fit_loader, device=device
                        )
                        fit_g = _ddp_gather_object(fit_pack_local)
                        fit_pack = _merge_gathered(fit_g) if is_distributed() else fit_pack_local

                        # Fit temperature
                        t_res = fit_temperature(
                            fit_pack["logits"],
                            fit_pack["y"],
                            max_iter=int(calib_cfg.get("max_iter", 50)),
                        )
                        temp = float(t_res.temperature)

                        # Metrics before/after on eval split
                        logits_eval = pack["logits"]
                        y_eval = pack["y"]
                        probs_before = softmax_np(logits_eval)
                        probs_after = softmax_np(apply_temperature(logits_eval, temp))

                        report = {
                            "fit_split": fit_split,
                            "eval_split": split,
                            "temperature": temp,
                            "nll_fit_before": float(t_res.nll_before),
                            "nll_fit_after": float(t_res.nll_after),
                            "eval_nll_before": float(nll_from_logits(logits_eval, y_eval)),
                            "eval_nll_after": float(
                                nll_from_logits(apply_temperature(logits_eval, temp), y_eval)
                            ),
                            "eval_ece_before": float(
                                expected_calibration_error(probs_before, y_eval, n_bins=n_bins)
                            ),
                            "eval_ece_after": float(
                                expected_calibration_error(probs_after, y_eval, n_bins=n_bins)
                            ),
                        }

                        calib_path = run_dir / f"calibration_{fit_split}_to_{split}.json"
                        calib_path.write_text(OmegaConf.to_yaml(report), encoding="utf-8")
                        log.info("Calibration report: %s", str(calib_path))
                        exp_logger.log_artifact(
                            str(calib_path), name=f"calibration_{fit_split}_to_{split}"
                        )

                        # Save calibrated predictions
                        if save_preds:
                            preds_cal_path = run_dir / f"predictions_{split}_calibrated.parquet"
                            _save_predictions_parquet(preds_cal_path, pack, temperature=temp)
                            log.info("Wrote calibrated predictions: %s", str(preds_cal_path))
                            exp_logger.log_artifact(
                                str(preds_cal_path), name=f"predictions_{split}_calibrated"
                            )
                    else:
                        log.info(
                            "Skipping calibration: fit_split='%s' not present in dataset config.",
                            fit_split,
                        )

                exp_logger.finish()

        cleanup_distributed()

    except Exception:
        log.exception("Fatal error in eval run")
        cleanup_distributed()
        raise


if __name__ == "__main__":
    main()


# Eval on test (auto-pick best checkpoint from a training run dir)

# python -m biosignals.cli.eval \
#   experiment=mitbih_aami3_cnn \
#   eval.run_dir=/ABS/PATH/TO/YOUR/TRAIN/RUN \
#   eval.split=test
