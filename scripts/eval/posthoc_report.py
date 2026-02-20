# scripts/eval/posthoc_report.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from biosignals.data.datamodule import build_dataset, make_loader


# -------------------------
# Metrics helpers
# -------------------------
def _confusion_matrix_mc(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=False):
        if 0 <= int(t) < k and 0 <= int(p) < k:
            cm[int(t), int(p)] += 1
    return cm


def _f1_from_cm(cm: np.ndarray) -> Tuple[float, List[float], List[float], List[float]]:
    k = int(cm.shape[0])
    per_f1: List[float] = []
    per_prec: List[float] = []
    per_rec: List[float] = []
    for i in range(k):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_prec.append(prec)
        per_rec.append(rec)
        per_f1.append(f1)
    macro_f1 = float(np.mean(per_f1)) if per_f1 else 0.0
    return macro_f1, per_f1, per_prec, per_rec


def _accuracy_mc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _brier_mc(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if probs.size == 0:
        return 0.0
    y_oh = np.zeros((y_true.size, k), dtype=np.float32)
    y_oh[np.arange(y_true.size), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs.astype(np.float32) - y_oh) ** 2, axis=1)))


def _reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # bin index in [0, n_bins-1]
    idx = np.digitize(conf, bins, right=True) - 1
    idx = np.clip(idx, 0, int(n_bins) - 1)

    bin_count = np.zeros((n_bins,), dtype=np.int64)
    bin_acc = np.zeros((n_bins,), dtype=np.float64)
    bin_conf = np.zeros((n_bins,), dtype=np.float64)

    for b in range(n_bins):
        m = idx == b
        c = int(np.sum(m))
        bin_count[b] = c
        if c > 0:
            bin_acc[b] = float(np.mean(correct[m]))
            bin_conf[b] = float(np.mean(conf[m]))

    n = float(conf.size) if conf.size > 0 else 1.0
    ece = float(np.sum((bin_count / n) * np.abs(bin_acc - bin_conf)))

    return {
        "bins": bins.tolist(),
        "bin_count": bin_count.tolist(),
        "bin_acc": bin_acc.tolist(),
        "bin_conf": bin_conf.tolist(),
        "ece": ece,
    }


def _plot_reliability(rel: Dict[str, Any], out_png: Path, title: str) -> None:
    bins = np.asarray(rel["bins"], dtype=np.float64)
    acc = np.asarray(rel["bin_acc"], dtype=np.float64)
    conf = np.asarray(rel["bin_conf"], dtype=np.float64)
    cnt = np.asarray(rel["bin_count"], dtype=np.int64)

    centers = 0.5 * (bins[:-1] + bins[1:])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot([0, 1], [0, 1])
    ax.plot(centers, acc, marker="o")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    # annotate counts lightly
    for x, y, c in zip(centers.tolist(), acc.tolist(), cnt.tolist(), strict=False):
        if c > 0:
            ax.annotate(
                str(c), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7
            )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -------------------------
# Calibration (temperature scaling)
# -------------------------
def _fit_temperature_mc(logits: np.ndarray, y_true: np.ndarray, *, max_iter: int = 100) -> float:
    """
    Fit a single scalar temperature on a calibration split by minimizing NLL.
    """
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(y_true.astype(np.int64), dtype=torch.long)

    log_t = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))  # temp = exp(log_t)

    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=int(max_iter), line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-3)
        loss = F.cross_entropy(x / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    t_final = float(torch.exp(log_t).detach().cpu().item())
    return t_final


# -------------------------
# IO / inference helpers
# -------------------------
def _load_ckpt_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # maybe already a state_dict-like mapping
        return ckpt  # type: ignore[return-value]
    raise ValueError(f"Unrecognized checkpoint format at: {path}")


def _maybe_strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state.keys())
    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def _infer_split(
    *,
    cfg: DictConfig,
    split: str,
    device: str,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Path]:
    """
    Returns (logits[N,K], y_true[N], ids[N], dataset_root)
    """
    task = instantiate(cfg.task)
    model = instantiate(cfg.model)

    # Load checkpoint
    ckpt_path = Path(cfg.eval.ckpt_path).expanduser().resolve()
    state = _maybe_strip_module_prefix(_load_ckpt_state(ckpt_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model.to(device)
    model.eval()

    # Choose dataset split cfg: prefer cfg.dataset.<split>, else reuse val and override split
    if split in cfg.dataset and cfg.dataset[split] is not None:
        split_cfg = cfg.dataset[split]
    else:
        base = (
            cfg.dataset.val
            if ("val" in cfg.dataset and cfg.dataset.val is not None)
            else cfg.dataset.train
        )
        split_cfg = OmegaConf.copy(base)
        split_cfg.split = str(split)

    # transforms: use cfg.transforms.<split> if exists else cfg.transforms.val
    if "transforms" in cfg and split in cfg.transforms and cfg.transforms[split] is not None:
        tf_cfg = cfg.transforms[split]
    else:
        tf_cfg = cfg.transforms.val

    # build dataset via your Option-B pipeline
    ds = build_dataset(
        split_cfg=split_cfg, transform_cfg=tf_cfg, cache_dir=None, cache_prefix=f"eval_{split}"
    )

    data_cfg = instantiate(cfg.data)
    if batch_size is not None:
        data_cfg.batch_size = int(batch_size)

    loader = make_loader(ds, task, data_cfg, shuffle=False)

    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            # move signals/targets/meta to device (lightweight)
            batch.signals = {k: v.to(device) for k, v in batch.signals.items()}
            batch.targets = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.targets.items()
            }
            batch.meta = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.meta.items()
            }

            logits = model(batch.signals, batch.meta)
            y = batch.targets["y"]

            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(y.detach().cpu().numpy())
            all_ids.extend([str(x) for x in batch.meta.get("ids", [])])

    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)

    # dataset root: assume WindowedNpzDataset underneath has .root
    dataset_root = getattr(getattr(ds, "dataset", ds), "root", None)
    dataset_root = Path(str(dataset_root)).resolve() if dataset_root is not None else Path(".")

    return logits_np, y_np, all_ids, dataset_root


def _ids_to_keys(ids: List[str]) -> pd.DataFrame:
    """
    Try to parse WindowedNpzDataset default ids: "{subject_id}:{start}:{end}"
    """
    pat = re.compile(r"^([^:]+):(\d+):(\d+)$")
    subj = []
    s0 = []
    s1 = []
    ok = []
    for x in ids:
        m = pat.match(str(x))
        if not m:
            subj.append(None)
            s0.append(None)
            s1.append(None)
            ok.append(False)
        else:
            subj.append(m.group(1))
            s0.append(int(m.group(2)))
            s1.append(int(m.group(3)))
            ok.append(True)
    return pd.DataFrame(
        {"id": ids, "subject_id": subj, "start_idx": s0, "end_idx": s1, "id_parsed": ok}
    )


def _metrics_mc_from_logits(logits: np.ndarray, y_true: np.ndarray, k: int) -> Dict[str, Any]:
    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    y_pred = np.argmax(probs, axis=1).astype(np.int64)

    acc = _accuracy_mc(y_true, y_pred)
    cm = _confusion_matrix_mc(y_true, y_pred, k=k)
    macro_f1, per_f1, per_prec, per_rec = _f1_from_cm(cm)

    conf = np.max(probs, axis=1)
    correct = (y_pred == y_true).astype(np.float32)

    rel = _reliability_bins(conf, correct, n_bins=15)
    brier = _brier_mc(probs, y_true, k=k)

    return {
        "n": int(y_true.size),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in per_f1],
        "per_class_precision": [float(x) for x in per_prec],
        "per_class_recall": [float(x) for x in per_rec],
        "confusion_matrix": cm.tolist(),
        "calibration": {"ece": float(rel["ece"]), "brier": float(brier), "reliability": rel},
    }


def main() -> None:
    ap = argparse.ArgumentParser("Post-hoc evaluation + calibration report (no training changes)")
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Hydra run dir containing config_resolved.yaml + checkpoints/",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Override checkpoint path (else use run_dir/checkpoints/best.pt)",
    )
    ap.add_argument("--split", type=str, default="test", help="Split to evaluate: train|val|test")
    ap.add_argument(
        "--calibrate",
        action="store_true",
        default=True,
        help="Fit temperature on val and evaluate calibrated test",
    )
    ap.add_argument("--no_calibrate", dest="calibrate", action="store_false")
    ap.add_argument(
        "--calibration_split", type=str, default="val", help="Split used to fit temperature scaling"
    )
    ap.add_argument(
        "--group_cols",
        type=str,
        default="sex,group_id",
        help="Comma-separated group columns for per-group metrics",
    )
    ap.add_argument(
        "--age_bins",
        type=str,
        default="0,40,60,80,200",
        help="Comma-separated bin edges for age_bin",
    )
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size for eval")
    ap.add_argument(
        "--out_dir", type=str, default=None, help="Output directory (default: run_dir/posthoc)"
    )

    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml at: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)

    # Attach eval config node so we can reuse _infer_split
    if "eval" not in cfg:
        cfg.eval = {}  # type: ignore[attr-defined]

    ckpt_path = (
        Path(args.ckpt).expanduser().resolve()
        if args.ckpt
        else (run_dir / "checkpoints" / "best.pt")
    )
    if not ckpt_path.exists():
        # fallback to last
        alt = run_dir / "checkpoints" / "last.pt"
        if alt.exists():
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"Missing checkpoint at {ckpt_path} and no last.pt found.")

    cfg.eval.ckpt_path = str(ckpt_path)  # type: ignore[attr-defined]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (run_dir / "posthoc")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- inference on requested split ----
    logits, y_true, ids, dataset_root = _infer_split(
        cfg=cfg, split=str(args.split), device=device, batch_size=args.batch_size
    )

    # figure out number of classes
    # prefer cfg.task.num_classes if present, else infer from logits
    k = int(getattr(cfg.task, "num_classes", logits.shape[1]))

    base_metrics = _metrics_mc_from_logits(logits, y_true.astype(np.int64), k=k)
    _plot_reliability(
        base_metrics["calibration"]["reliability"],
        out_dir / f"reliability_{args.split}_raw.png",
        title=f"Reliability ({args.split}) raw | ECE={base_metrics['calibration']['ece']:.4f}",
    )

    # ---- join predictions to windows/subjects for group slicing ----
    keys_df = _ids_to_keys(ids)
    pred_df = keys_df.copy()
    pred_df["y_true"] = y_true.astype(np.int64)

    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    pred_df["y_pred"] = np.argmax(probs, axis=1).astype(np.int64)
    pred_df["conf"] = np.max(probs, axis=1).astype(np.float32)

    windows_path = dataset_root / "views" / "windows.parquet"
    subjects_path = dataset_root / "views" / "subjects.parquet"

    if windows_path.exists():
        wdf = pd.read_parquet(windows_path)
        wdf = wdf[wdf["split"].astype(str) == str(args.split)].copy()
        # prefer example_id join
        if "example_id" in wdf.columns and "id" in pred_df.columns and pred_df["id"].notna().all():
            merged = pred_df.merge(
                wdf, left_on="id", right_on="example_id", how="left", suffixes=("", "_w")
            )
        else:
            merged = pred_df.merge(
                wdf, on=["subject_id", "start_idx", "end_idx"], how="left", suffixes=("", "_w")
            )
    else:
        merged = pred_df.copy()

    if subjects_path.exists() and "subject_id" in merged.columns:
        sdf = pd.read_parquet(subjects_path)
        merged = merged.merge(sdf, on="subject_id", how="left", suffixes=("", "_s"))

    # Add age_bin if age exists
    if "age" in merged.columns:
        try:
            edges = [int(x.strip()) for x in str(args.age_bins).split(",") if x.strip() != ""]
            if len(edges) >= 2:
                merged["age_bin"] = pd.cut(
                    merged["age"].astype(float), bins=edges, include_lowest=True
                ).astype(str)
        except Exception:
            pass

    # ---- per-group metrics ----
    group_cols = [c.strip() for c in str(args.group_cols).split(",") if c.strip() != ""]
    per_group: Dict[str, Any] = {}

    for gc in group_cols:
        if gc not in merged.columns:
            continue
        g_metrics: Dict[str, Any] = {}
        for gval, gdf in merged.groupby(gc, dropna=False):
            yt = gdf["y_true"].to_numpy(dtype=np.int64)
            yp = gdf["y_pred"].to_numpy(dtype=np.int64)
            cm = _confusion_matrix_mc(yt, yp, k=k)
            macro_f1, per_f1, _, _ = _f1_from_cm(cm)
            g_metrics[str(gval)] = {
                "n": int(len(gdf)),
                "acc": float(_accuracy_mc(yt, yp)),
                "macro_f1": float(macro_f1),
                "per_class_f1": [float(x) for x in per_f1],
            }
        per_group[gc] = g_metrics

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "metrics_raw": base_metrics,
        "per_group_raw": per_group,
    }

    # ---- temperature scaling calibration (optional) ----
    if bool(args.calibrate):
        logits_cal, y_cal, _, _ = _infer_split(
            cfg=cfg, split=str(args.calibration_split), device=device, batch_size=args.batch_size
        )
        t = _fit_temperature_mc(logits_cal, y_cal.astype(np.int64), max_iter=100)

        logits_scaled = logits / float(max(t, 1e-6))
        cal_metrics = _metrics_mc_from_logits(logits_scaled, y_true.astype(np.int64), k=k)

        _plot_reliability(
            cal_metrics["calibration"]["reliability"],
            out_dir / f"reliability_{args.split}_temp_scaled.png",
            title=f"Reliability ({args.split}) temp-scaled | T={t:.4f} | ECE={cal_metrics['calibration']['ece']:.4f}",
        )

        report["calibration"] = {
            "method": "temperature_scaling",
            "calibration_split": str(args.calibration_split),
            "temperature": float(t),
            "ece_raw": float(base_metrics["calibration"]["ece"]),
            "ece_scaled": float(cal_metrics["calibration"]["ece"]),
            "brier_raw": float(base_metrics["calibration"]["brier"]),
            "brier_scaled": float(cal_metrics["calibration"]["brier"]),
        }
        report["metrics_temp_scaled"] = cal_metrics

    # Save artifacts
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    merged.to_parquet(out_dir / f"predictions_{args.split}.parquet", index=False)

    print(f"[ok] wrote: {out_dir / 'report.json'}")
    print(f"[ok] wrote: {out_dir / f'predictions_{args.split}.parquet'}")
    print(f"[ok] wrote: {out_dir / f'reliability_{args.split}_raw.png'}")
    if bool(args.calibrate):
        print(f"[ok] wrote: {out_dir / f'reliability_{args.split}_temp_scaled.png'}")


if __name__ == "__main__":
    main()


# # ----- how to run ----- :
# after training (hydra run dir with config_resolved.yaml, checkpoints/best.pt):

# python scripts/eval/posthoc_report.py \
#   --run_dir /ABS/PATH/TO/your/hydra/run \
#   --split test \
#   --group_cols sex,age_bin,group_id \
#   --calibrate
