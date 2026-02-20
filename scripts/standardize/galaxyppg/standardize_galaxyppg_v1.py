# scripts/standardize/galaxyppg/standardize_galaxyppg_v1.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _ensure_monotonic(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=[tcol]).sort_values(tcol, kind="mergesort")
    out = out.drop_duplicates(subset=[tcol], keep="first").reset_index(drop=True)
    return out


def _read_ppg_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "ppg" not in df.columns:
        raise ValueError(f"Unexpected PPG columns in {path}: {df.columns.tolist()}")
    df = df.rename(columns={"timestamp": "t_ms"})
    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce").astype(np.int64)
    df["ppg"] = pd.to_numeric(df["ppg"], errors="coerce").astype(np.float32)
    return _ensure_monotonic(df, "t_ms")


def _read_acc_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"timestamp", "x", "y", "z"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Unexpected ACC columns in {path}: {df.columns.tolist()}")
    df = df.rename(columns={"timestamp": "t_ms"})
    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce").astype(np.int64)
    for c in ["x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    return _ensure_monotonic(df, "t_ms")


def _read_polar_hr_csv(path: Path, tz_offset_hours: float = 9.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "phoneTimestamp" not in df.columns:
        raise ValueError(f"Unexpected Polar HR columns in {path}: {df.columns.tolist()}")
    df = df.rename(columns={"phoneTimestamp": "t_ms"})
    df["t_ms"] = pd.to_numeric(df["t_ms"], errors="coerce").astype(np.float64)

    # IMPORTANT: GalaxyPPG README says Polar phoneTimestamp is UTC+9.
    # Convert to UTC epoch ms by subtracting offset.
    offset_ms = float(tz_offset_hours) * 60.0 * 60.0 * 1000.0
    df["t_ms"] = df["t_ms"] - offset_ms

    if "hr" in df.columns:
        df["hr"] = pd.to_numeric(df["hr"], errors="coerce").astype(np.float32)
    df = df.dropna(subset=["t_ms", "hr"])
    df = df[df["hr"] > 0].reset_index(drop=True)
    return _ensure_monotonic(df, "t_ms")


def _events_to_segments(events: pd.DataFrame) -> List[Tuple[float, float, str]]:
    """
    Event.csv format:
      timestamp (ms), session, status (ENTER/EXIT)
    Returns list of (t0_ms, t1_ms, session_name)
    """
    if events is None or len(events) == 0:
        return []

    df = events.copy()
    if "timestamp" not in df.columns or "session" not in df.columns or "status" not in df.columns:
        return []

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype(np.float64)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    segs: List[Tuple[float, float, str]] = []
    curr: Optional[str] = None
    t0: Optional[float] = None

    for _, r in df.iterrows():
        t = float(r["timestamp"])
        sess = str(r["session"])
        st = str(r["status"]).strip().upper()

        if st == "ENTER":
            curr = sess
            t0 = t
        elif st == "EXIT":
            if curr != sess or t0 is None:
                curr = None
                t0 = None
                continue
            if t > t0:
                segs.append((t0, t, sess))
            curr = None
            t0 = None

    return segs


def _interp_to_grid(t_grid: np.ndarray, t_obs: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    t_obs = np.asarray(t_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    if t_obs.size < 2:
        return np.full((t_grid.size,), np.nan, dtype=np.float32)

    order = np.argsort(t_obs)
    t_obs = t_obs[order]
    y_obs = y_obs[order]

    # drop duplicates
    keep = np.ones_like(t_obs, dtype=bool)
    keep[1:] = (t_obs[1:] != t_obs[:-1])
    t_obs = t_obs[keep]
    y_obs = y_obs[keep]

    if t_obs.size < 2:
        return np.full((t_grid.size,), np.nan, dtype=np.float32)

    y = np.interp(t_grid, t_obs, y_obs).astype(np.float32)
    # nan outside support
    y[(t_grid < t_obs[0]) | (t_grid > t_obs[-1])] = np.nan
    return y


def _interp_scalar(t_ms: float, t_obs: np.ndarray, y_obs: np.ndarray) -> float:
    t_obs = np.asarray(t_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    if t_obs.size < 2:
        return float("nan")
    if t_ms <= float(t_obs[0]):
        return float(y_obs[0])
    if t_ms >= float(t_obs[-1]):
        return float(y_obs[-1])
    return float(np.interp(float(t_ms), t_obs, y_obs))


def _resample_uniform(
    t_ms: np.ndarray,
    x: np.ndarray,
    fs_out: float,
    t_start_ms: float,
    t_end_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_ms, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(x)
    t = t[m]
    x = x[m]
    if t.size < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float32)

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    dt = 1000.0 / float(fs_out)
    t_grid = np.arange(float(t_start_ms), float(t_end_ms), dt, dtype=np.float64)
    y = np.interp(t_grid, t, x).astype(np.float32)
    return t_grid, y


def _subject_dirs(raw_root: Path) -> List[Path]:
    out = []
    for p in sorted(raw_root.glob("P*")):
        if p.is_dir() and p.name[1:].isdigit():
            out.append(p)
    return out


def _make_subject_splits(subjects: List[str], seed: int, train: float, val: float, test: float) -> Dict[str, str]:
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")
    rng = np.random.default_rng(int(seed))
    subs = sorted([str(s) for s in subjects])
    rng.shuffle(subs)
    n = len(subs)
    n_tr = int(round(train * n))
    n_va = int(round(val * n))
    n_te = n - n_tr - n_va

    m: Dict[str, str] = {}
    for s in subs[:n_tr]:
        m[s] = "train"
    for s in subs[n_tr : n_tr + n_va]:
        m[s] = "val"
    for s in subs[n_tr + n_va :]:
        m[s] = "test"
    return m


# -----------------------------
# Main standardizer
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Standardize GalaxyPPG -> subjects NPZ + views/windows.parquet (v1)")
    ap.add_argument("--raw_root", type=str, required=True, help="GalaxyPPG/Dataset directory containing Meta.csv and Pxx/")
    ap.add_argument("--out_root", type=str, default="data/standardized/galaxyppg/v1")

    ap.add_argument("--fs_out", type=float, default=64.0)
    ap.add_argument("--window_length_s", type=float, default=8.0)
    ap.add_argument("--window_shift_s", type=float, default=2.0)

    ap.add_argument("--polar_tz_offset_hours", type=float, default=9.0)

    ap.add_argument("--require_ppg", action="store_true", default=True)
    ap.add_argument("--require_acc", action="store_true", default=True)

    ap.add_argument("--require_polar_hr", action="store_true", default=False)
    ap.add_argument("--require_events", action="store_true", default=False)
    ap.add_argument("--require_same_session_window", action="store_true", default=False)

    ap.add_argument("--min_overlap_min", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)

    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    subj_out = out_root / "subjects"
    views_out = out_root / "views"
    subj_out.mkdir(parents=True, exist_ok=True)
    views_out.mkdir(parents=True, exist_ok=True)

    # Meta.csv (stress labels)
    meta_path = raw_root / "Meta.csv"
    meta_df = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    # Normalize UID column name
    if len(meta_df) > 0 and "UID" in meta_df.columns:
        meta_df["UID"] = meta_df["UID"].astype(str)
        meta_df = meta_df.set_index("UID", drop=False)

    # Pre-scan session names for stable id mapping
    segs_by_subj: Dict[str, List[Tuple[float, float, str]]] = {}
    session_names: set[str] = set()

    for sd in _subject_dirs(raw_root):
        subj = sd.name
        evp = sd / "Event.csv"
        if evp.exists():
            ev = pd.read_csv(evp)
            segs = _events_to_segments(ev)
            if segs:
                segs_by_subj[subj] = segs
                for _, _, nm in segs:
                    session_names.add(str(nm))

    session_name_list = sorted(list(session_names))
    session_name_to_id = {n: i for i, n in enumerate(session_name_list)}
    session_id_to_name = {int(i): str(n) for n, i in session_name_to_id.items()}

    # Process subjects
    window_len = int(round(float(args.window_length_s) * float(args.fs_out)))
    hop = int(round(float(args.window_shift_s) * float(args.fs_out)))
    if window_len <= 0 or hop <= 0:
        raise ValueError("Bad window spec")

    rows: List[Dict] = []
    subject_rows: List[Dict] = []
    kept_subjects: List[str] = []

    # Streaming stats accumulators
    stats = {
        "ppg": {"sum": None, "sumsq": None, "count": 0},
        "acc": {"sum": None, "sumsq": None, "count": 0},
    }

    for sd in _subject_dirs(raw_root):
        subj = sd.name
        gw = sd / "GalaxyWatch"
        po = sd / "PolarH10"

        ppg_path = gw / "PPG.csv"
        acc_path = gw / "ACC.csv"
        hr_path = po / "HR.csv"
        ev_path = sd / "Event.csv"

        if args.require_ppg and not ppg_path.exists():
            continue
        if args.require_acc and not acc_path.exists():
            continue
        if args.require_polar_hr and not hr_path.exists():
            continue
        if args.require_events and not ev_path.exists():
            continue

        # Load streams
        try:
            ppg = _read_ppg_csv(ppg_path) if ppg_path.exists() else None
            acc = _read_acc_csv(acc_path) if acc_path.exists() else None
            hr = _read_polar_hr_csv(hr_path, tz_offset_hours=float(args.polar_tz_offset_hours)) if hr_path.exists() else None
        except Exception as e:
            print(f"[skip] {subj}: load error: {e}")
            continue

        if ppg is None or acc is None:
            continue

        # Determine overlap
        t_start = max(float(ppg["t_ms"].iloc[0]), float(acc["t_ms"].iloc[0]))
        t_end = min(float(ppg["t_ms"].iloc[-1]), float(acc["t_ms"].iloc[-1]))

        # If HR present, include its overlap for hr labels
        if hr is not None and len(hr) >= 2:
            t_start = max(t_start, float(hr["t_ms"].iloc[0]))
            t_end = min(t_end, float(hr["t_ms"].iloc[-1]))

        # If session constraints enabled, constrain to session segment envelope
        segs = segs_by_subj.get(subj, [])
        if (args.require_events or args.require_same_session_window) and len(segs) == 0:
            continue
        if len(segs) > 0:
            seg_start = float(min(a for a, _, _ in segs))
            seg_end = float(max(b for _, b, _ in segs))
            t_start = max(t_start, seg_start)
            t_end = min(t_end, seg_end)

        if t_end <= t_start:
            continue

        overlap_min = (t_end - t_start) / 1000.0 / 60.0
        if overlap_min < float(args.min_overlap_min):
            continue

        # Resample onto uniform grid
        t_grid, ppg_u = _resample_uniform(ppg["t_ms"].to_numpy(), ppg["ppg"].to_numpy(), args.fs_out, t_start, t_end)
        if t_grid.size == 0:
            continue

        _, ax = _resample_uniform(acc["t_ms"].to_numpy(), acc["x"].to_numpy(), args.fs_out, t_start, t_end)
        _, ay = _resample_uniform(acc["t_ms"].to_numpy(), acc["y"].to_numpy(), args.fs_out, t_start, t_end)
        _, az = _resample_uniform(acc["t_ms"].to_numpy(), acc["z"].to_numpy(), args.fs_out, t_start, t_end)
        acc_u = np.stack([ax, ay, az], axis=0).astype(np.float32, copy=False)  # (3,T)

        # Build per-sample session_id on grid
        session_id_seq = np.full((t_grid.size,), -1, dtype=np.int16)
        if len(segs) > 0:
            tg = t_grid.astype(np.float64, copy=False)
            for a, b, nm in segs:
                sid = int(session_name_to_id.get(str(nm), -1))
                if sid < 0:
                    continue
                m = (tg >= float(a)) & (tg < float(b))
                session_id_seq[m] = np.int16(sid)

        # Save subject NPZ
        npz_path = subj_out / f"{subj}.npz"
        np.savez_compressed(
            npz_path,
            ppg=ppg_u.astype(np.float32, copy=False)[None, :],  # (1,T)
            acc=acc_u,                                          # (3,T)
            session_id=session_id_seq,
            t_ms=t_grid.astype(np.float64, copy=False),
            fs=np.array([float(args.fs_out)], dtype=np.float32),
        )

        # Update stats (simple sum/sumsq)
        # ppg
        ppg_ct = ppg_u.astype(np.float64, copy=False)[None, :]
        if stats["ppg"]["sum"] is None:
            stats["ppg"]["sum"] = np.zeros((1,), dtype=np.float64)
            stats["ppg"]["sumsq"] = np.zeros((1,), dtype=np.float64)
        stats["ppg"]["sum"] += np.sum(ppg_ct, axis=1)
        stats["ppg"]["sumsq"] += np.sum(ppg_ct * ppg_ct, axis=1)
        stats["ppg"]["count"] += int(ppg_ct.shape[1])

        # acc
        acc_ct = acc_u.astype(np.float64, copy=False)
        if stats["acc"]["sum"] is None:
            stats["acc"]["sum"] = np.zeros((3,), dtype=np.float64)
            stats["acc"]["sumsq"] = np.zeros((3,), dtype=np.float64)
        stats["acc"]["sum"] += np.sum(acc_ct, axis=1)
        stats["acc"]["sumsq"] += np.sum(acc_ct * acc_ct, axis=1)
        stats["acc"]["count"] += int(acc_ct.shape[1])

        # Subject-level labels (stress)
        tsst = None
        ssst = None
        if len(meta_df) > 0 and subj in meta_df.index:
            # README: TSST/SSST are 7-pt Likert (lower means less stress)
            try:
                tsst = int(meta_df.loc[subj]["TSST"])
            except Exception:
                tsst = None
            try:
                ssst = int(meta_df.loc[subj]["SSST"])
            except Exception:
                ssst = None

        # HR label arrays
        t_hr = hr["t_ms"].to_numpy(np.float64) if (hr is not None and len(hr) >= 2) else None
        y_hr = hr["hr"].to_numpy(np.float32) if (hr is not None and len(hr) >= 2) else None

        # Window rows
        T = int(t_grid.size)
        n_win = 0
        w_idx = 0
        i = 0
        while i + window_len <= T:
            i0 = int(i)
            i1 = int(i + window_len)
            ic = (i0 + i1 - 1) // 2

            t0 = float(t_grid[i0])
            t1 = float(t_grid[i1 - 1])
            tc = float(t_grid[ic])

            # session label rules
            sid_center = int(session_id_seq[ic])
            sid_start = int(session_id_seq[i0])
            sid_end = int(session_id_seq[i1 - 1])

            if args.require_same_session_window:
                if sid_start < 0 or sid_start != sid_end:
                    i += hop
                    w_idx += 1
                    continue
                sid = sid_start
            else:
                sid = sid_center

            sess_name = session_id_to_name.get(int(sid), "none") if sid >= 0 else "none"

            # hr label at center (can be NaN if HR missing)
            hr_y = float("nan")
            if t_hr is not None and y_hr is not None:
                hr_y = _interp_scalar(tc, t_hr, y_hr)

            rows.append(
                {
                    "dataset": "galaxyppg",
                    "version": "v1",
                    "subject_id": subj,
                    "window_idx": int(w_idx),
                    "start_idx": int(i0),
                    "end_idx": int(i1),
                    "t_start_ms": t0,
                    "t_end_ms": t1,
                    "t_center_ms": tc,
                    "hr": float(hr_y),
                    "session_id": int(sid),
                    "session": str(sess_name),
                    "tsst": (int(tsst) if tsst is not None else None),
                    "ssst": (int(ssst) if ssst is not None else None),
                    "fs": float(args.fs_out),
                }
            )
            n_win += 1
            i += hop
            w_idx += 1

        if n_win == 0:
            continue

        kept_subjects.append(subj)
        subject_rows.append(
            {
                "subject_id": subj,
                "fs": float(args.fs_out),
                "n_samples": int(T),
                "duration_s": float(T / float(args.fs_out)),
                "n_windows": int(n_win),
                "tsst": (int(tsst) if tsst is not None else None),
                "ssst": (int(ssst) if ssst is not None else None),
                "has_events": bool((sd / "Event.csv").exists()),
                "has_polar_hr": bool(hr_path.exists()),
            }
        )
        print(f"[ok] {subj}: T={T} windows={n_win} overlap_min={overlap_min:.2f}")

    if len(rows) == 0:
        raise RuntimeError("No windows produced. Check requirements/paths.")

    windows = pd.DataFrame(rows)
    subjects_df = pd.DataFrame(subject_rows)

    # Subject splits (only for kept subjects)
    split_map = _make_subject_splits(
        subjects=kept_subjects,
        seed=int(args.seed),
        train=float(args.train_frac),
        val=float(args.val_frac),
        test=float(args.test_frac),
    )
    windows["split"] = windows["subject_id"].astype(str).map(split_map)
    subjects_df["split"] = subjects_df["subject_id"].astype(str).map(split_map)

    # Write views
    windows_path = views_out / "windows.parquet"
    subjects_path = views_out / "subjects.parquet"
    splits_path = views_out / "splits.json"

    windows.to_parquet(windows_path, index=False)
    subjects_df.to_parquet(subjects_path, index=False)
    splits_path.write_text(json.dumps(split_map, indent=2))

    # Dataset meta + stats
    meta = {
        "dataset": "galaxyppg",
        "version": "v1",
        "raw_root": str(raw_root),
        "fs_out": float(args.fs_out),
        "window": {"length_s": float(args.window_length_s), "shift_s": float(args.window_shift_s)},
        "n_subjects": int(len(kept_subjects)),
        "n_windows": int(len(windows)),
        "splits": {
            "seed": int(args.seed),
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "test_frac": float(args.test_frac),
        },
        "session_name_to_id": session_name_to_id,
        "session_id_to_name": session_id_to_name,
        "notes": [
            "Subject NPZ stores aligned/resampled GalaxyWatch PPG+ACC on a uniform grid.",
            "Window labels include: hr (PolarH10 HR interpolated at window center), session_id/session, and TSST/SSST from Meta.csv.",
            "Polar phoneTimestamp is corrected from UTC+9 by subtracting 9 hours (configurable).",
        ],
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    stats_out = {}
    for m in ["ppg", "acc"]:
        cnt = max(1, int(stats[m]["count"]))
        s = stats[m]["sum"]
        ss = stats[m]["sumsq"]
        mean = (s / cnt).tolist() if s is not None else None
        var = (ss / cnt - (s / cnt) ** 2) if (s is not None and ss is not None) else None
        std = np.sqrt(np.maximum(var, 1e-12)).tolist() if var is not None else None
        stats_out[m] = {"count": int(cnt), "mean": mean, "std": std}

    (out_root / "stats.json").write_text(json.dumps(stats_out, indent=2))

    print(f"\nWrote standardized dataset to: {out_root}")
    print(f"  - {windows_path}")
    print(f"  - {subjects_path}")
    print(f"  - {splits_path}")


if __name__ == "__main__":
    main()


# ------------------------------

# Run standardization

# Example (adjust raw path):

# python scripts/standardize/galaxyppg/standardize_galaxyppg_v1.py \
#   --raw_root /ABS/PATH/GalaxyPPG/Dataset \
#   --out_root data/standardized/galaxyppg/v1 \
#   --fs_out 64 \
#   --window_length_s 8 \
#   --window_shift_s 2 \
#   --require_same_session_window
