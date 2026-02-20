# scripts/standardize/mitbih/standardize_mitbih_v1.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb


# -----------------------------
# AAMI label mapping (MIT-BIH beat symbols -> coarse classes)
# -----------------------------
_AAMI_N = {"N", "L", "R", "e", "j"}   # normal, BBB, escape beats
_AAMI_S = {"A", "a", "J", "S"}        # supraventricular ectopic
_AAMI_V = {"V", "E"}                  # ventricular ectopic
_AAMI_F = {"F"}                       # fusion
_AAMI_Q = {"/", "f", "Q"}             # paced / fusion paced / unclassifiable


def map_symbol(symbol: str, scheme: str) -> Optional[str]:
    """
    Map MIT-BIH beat annotation symbols to labels.
      - aami3: {N, S, V}
      - aami5: {N, S, V, F, Q}
    """
    if symbol in _AAMI_N:
        lbl = "N"
    elif symbol in _AAMI_S:
        lbl = "S"
    elif symbol in _AAMI_V:
        lbl = "V"
    elif symbol in _AAMI_F:
        lbl = "F"
    elif symbol in _AAMI_Q:
        lbl = "Q"
    else:
        return None

    scheme = str(scheme).lower().strip()
    if scheme == "aami3":
        return lbl if lbl in {"N", "S", "V"} else None
    if scheme == "aami5":
        return lbl
    raise ValueError(f"Unknown label scheme: {scheme}")


def list_mitbih_records(raw_root: Path) -> List[str]:
    """
    Prefer RECORDS file at dataset root; else infer from *.hea.
    Keep only numeric record IDs (ignore x_mitdb etc).
    """
    records_file = raw_root / "RECORDS"
    if records_file.exists():
        candidates = [ln.strip() for ln in records_file.read_text().splitlines() if ln.strip()]
    else:
        candidates = sorted([p.stem for p in raw_root.glob("*.hea")])

    records = [r for r in candidates if r.isdigit()]
    return sorted(records)


def group_id_for_record(record_id: str) -> str:
    # Documented special case for leakage-safe evaluation:
    # Records 201 and 202 are from the same subject.
    if record_id in {"201", "202"}:
        return "201_202"
    return record_id


def make_group_splits(
    group_ids: List[str],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Dict[str, str]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    groups = sorted(set([str(g) for g in group_ids]))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val

    out: Dict[str, str] = {}
    for g in groups[:n_train]:
        out[g] = "train"
    for g in groups[n_train : n_train + n_val]:
        out[g] = "val"
    for g in groups[n_train + n_val :]:
        out[g] = "test"
    assert len(out) == len(groups)
    return out


def _parse_age_sex_from_comments(comments: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Best-effort parse from WFDB header comments. Not guaranteed.
    """
    if not comments:
        return None, None

    text = " ".join([str(c).strip() for c in comments if str(c).strip()])
    if not text:
        return None, None

    age = None
    m = re.search(r"\bage\s*[:=]?\s*(\d{1,3})\b", text, flags=re.IGNORECASE)
    if m:
        try:
            age = int(m.group(1))
        except Exception:
            age = None

    sex = None
    m = re.search(r"\bsex\s*[:=]?\s*(male|female|m|f)\b", text, flags=re.IGNORECASE)
    if m:
        s = m.group(1).lower()
        if s in {"m", "male"}:
            sex = "M"
        elif s in {"f", "female"}:
            sex = "F"

    # fallback pattern: "69 M" or "83 F"
    if age is None or sex is None:
        m2 = re.search(r"\b(\d{2,3})\s*(M|F)\b", text)
        if m2:
            if age is None:
                try:
                    age = int(m2.group(1))
                except Exception:
                    pass
            if sex is None:
                sex = str(m2.group(2)).upper()

    return age, sex


def _strip_html(s: str) -> str:
    # very simple tag stripper for mitdbdir HTML files
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_mitdbdir_records_text(raw_root: Path) -> Optional[str]:
    """
    Try to load a local MIT-BIH directory 'records' document if present in the download.
    This is OPTIONAL. If absent, we fall back to header comments.
    """
    candidates = [
        raw_root / "mitdbdir" / "records",
        raw_root / "mitdbdir" / "records.txt",
        raw_root / "mitdbdir" / "records.htm",
        raw_root / "mitdbdir" / "records.html",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            txt = p.read_text(errors="ignore")
            if p.suffix.lower() in {".htm", ".html"}:
                txt = _strip_html(txt)
            return txt
    return None


def _parse_directory_record_meta(text: str, *, max_notes_chars: int = 2000) -> Dict[str, Dict[str, Any]]:
    """
    Parse a mitdbdir 'records' text into per-record metadata:
      - sex, age, medications, lead_config, record_notes
    """
    if not text:
        return {}

    # We look for blocks like:
    #   "Record 221 (MLII, V1; male, age 83) Medications: ..."
    pat = re.compile(r"\bRecord\s+(\d{3})\s*\(([^)]*)\)", flags=re.IGNORECASE)
    matches = list(pat.finditer(text))
    if not matches:
        return {}

    out: Dict[str, Dict[str, Any]] = {}

    for i, m in enumerate(matches):
        rec = str(m.group(1))
        inside = str(m.group(2)).strip()

        blk_start = m.end()
        blk_end = matches[i + 1].start() if (i + 1) < len(matches) else len(text)
        block = text[blk_start:blk_end].strip()

        # lead config: usually before ';'
        lead_config = None
        sex = None
        age = None

        parts = [p.strip() for p in inside.split(";") if p.strip()]
        if len(parts) >= 1:
            lead_config = parts[0]

        rest = "; ".join(parts[1:]) if len(parts) > 1 else ""
        if re.search(r"\bfemale\b", rest, flags=re.IGNORECASE):
            sex = "F"
        elif re.search(r"\bmale\b", rest, flags=re.IGNORECASE):
            sex = "M"

        ma = re.search(r"\bage\s*(\d{1,3})\b", rest, flags=re.IGNORECASE)
        if ma:
            try:
                age = int(ma.group(1))
            except Exception:
                age = None

        meds = None
        mm = re.search(r"\bMedications\s*:\s*([^\n\r]+)", block, flags=re.IGNORECASE)
        if mm:
            meds = str(mm.group(1)).strip()

        notes = re.sub(r"\s+", " ", block).strip()
        if max_notes_chars and len(notes) > int(max_notes_chars):
            notes = notes[: int(max_notes_chars)] + "..."

        out[rec] = {
            "age": age,
            "sex": sex,
            "medications": meds,
            "lead_config": lead_config,
            "record_notes": notes,
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser("Standardize MIT-BIH -> subjects NPZ + views/windows.parquet (v1)")
    ap.add_argument("--raw_root", type=str, required=True, help="Path to mit-bih-arrhythmia-database-1.0.0")
    ap.add_argument("--out_root", type=str, default="data/standardized/mitbih/v1")

    ap.add_argument("--ann_extension", type=str, default="atr", help="Annotation extension (e.g., atr)")
    ap.add_argument("--label_scheme", type=str, default="aami3", choices=["aami3", "aami5"])

    ap.add_argument("--window_left_s", type=float, default=0.30, help="Seconds left of beat sample")
    ap.add_argument("--window_right_s", type=float, default=0.50, help="Seconds right of beat sample")

    ap.add_argument("--fs_expected", type=float, default=360.0, help="Warn/skip if record fs differs (<=0 disables)")
    ap.add_argument("--lead_indices", type=str, default="0,1", help="Comma list of leads to store (default: both leads)")

    # "test period" in MIT-BIH directory tables typically excludes first 5 minutes
    ap.add_argument("--test_period_start_s", type=float, default=300.0, help="Mark beats after this time as in_test_period")

    # Compression flag pair (standard argparse pattern)
    ap.add_argument("--compress", dest="compress", action="store_true", help="Write subjects as np.savez_compressed")
    ap.add_argument("--no_compress", dest="compress", action="store_false", help="Write subjects as np.savez (uncompressed)")
    ap.set_defaults(compress=True)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)

    args = ap.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    subjects_dir = out_root / "subjects"
    views_dir = out_root / "views"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)

    records = list_mitbih_records(raw_root)
    if len(records) == 0:
        raise RuntimeError(f"No records found under {raw_root}. Expected RECORDS file or *.hea headers at root.")

    lead_indices = [int(x.strip()) for x in str(args.lead_indices).split(",") if x.strip() != ""]
    if len(lead_indices) == 0:
        raise ValueError("lead_indices parsed empty; expected something like '0' or '0,1'.")

    # Optional directory-based metadata (age/sex/meds/lead config)
    dir_text = _load_mitdbdir_records_text(raw_root)
    dir_meta = _parse_directory_record_meta(dir_text) if dir_text else {}

    # Streaming stats accumulators for modality "main"
    stats = {"sum": None, "sumsq": None, "count": 0}

    win_rows: List[Dict[str, Any]] = []
    subj_rows: List[Dict[str, Any]] = []

    for rec in records:
        rec_path = str(raw_root / rec)

        try:
            sig, fields = wfdb.rdsamp(rec_path)  # sig: (T, C)
            ann = wfdb.rdann(rec_path, str(args.ann_extension))
        except Exception as e:
            print(f"[skip] record {rec}: wfdb read error: {e}")
            continue

        fs = float(fields.get("fs", np.nan))
        T = int(sig.shape[0])
        C = int(sig.shape[1])

        if float(args.fs_expected) > 0 and np.isfinite(fs):
            if abs(fs - float(args.fs_expected)) > 1e-3:
                print(f"[skip] record {rec}: fs={fs} != expected={args.fs_expected}")
                continue

        if max(lead_indices) >= C:
            print(f"[skip] record {rec}: lead_indices {lead_indices} out of range for C={C}")
            continue

        # select leads
        x_tc = sig[:, lead_indices].astype(np.float32, copy=False)  # (T, C_sel)
        x_ct = x_tc.T  # (C_sel, T)

        sig_names = fields.get("sig_name", None)
        units = fields.get("units", None)
        comments = fields.get("comments", []) or []

        # demographic/clinical meta: prefer directory meta if present
        age_h, sex_h = _parse_age_sex_from_comments(list(comments))
        meta_dir = dir_meta.get(rec, {})

        age = meta_dir.get("age", age_h)
        sex = meta_dir.get("sex", sex_h)
        lead_config = meta_dir.get("lead_config", None)
        medications = meta_dir.get("medications", None)
        record_notes = meta_dir.get("record_notes", "")

        group_id = group_id_for_record(rec)

        # Save subject NPZ (record-level store)
        npz_path = subjects_dir / f"{rec}.npz"
        payload: Dict[str, Any] = {
            "main": x_ct,  # (C,T)
            "fs": np.array([fs], dtype=np.float32),
        }
        if sig_names is not None:
            payload["sig_names"] = np.asarray(list(sig_names), dtype=str)
        if units is not None:
            payload["units"] = np.asarray(list(units), dtype=str)

        if bool(args.compress):
            np.savez_compressed(npz_path, **payload)
        else:
            np.savez(npz_path, **payload)

        # Update dataset stats
        x64 = x_ct.astype(np.float64, copy=False)
        if stats["sum"] is None:
            stats["sum"] = np.zeros((x64.shape[0],), dtype=np.float64)
            stats["sumsq"] = np.zeros((x64.shape[0],), dtype=np.float64)
        stats["sum"] += np.sum(x64, axis=1)
        stats["sumsq"] += np.sum(x64 * x64, axis=1)
        stats["count"] += int(x64.shape[1])

        # Build beat windows
        left_n = int(round(float(args.window_left_s) * fs))
        right_n = int(round(float(args.window_right_s) * fs))
        if left_n <= 0 or right_n <= 0:
            raise ValueError("Bad window spec; left/right must be positive.")

        kept = 0
        for w_idx, (samp, sym) in enumerate(zip(ann.sample, ann.symbol)):
            lbl = map_symbol(str(sym), str(args.label_scheme))
            if lbl is None:
                continue

            start = int(samp) - left_n
            end = int(samp) + right_n  # exclusive
            if start < 0 or end > T:
                continue

            t_center_s = float(int(samp) / fs) if fs > 0 else float("nan")
            in_test_period = bool(t_center_s >= float(args.test_period_start_s))

            example_id = f"{rec}:{start}:{end}"

            win_rows.append(
                {
                    "dataset": "mitbih",
                    "version": "v1",
                    "example_id": example_id,
                    "subject_id": rec,      # NPZ key (record-level)
                    "record_id": rec,
                    "group_id": group_id,   # leakage-safe split group
                    "window_idx": int(w_idx),
                    "beat_sample": int(samp),
                    "start_idx": int(start),
                    "end_idx": int(end),
                    "t_center_s": t_center_s,
                    "in_test_period": in_test_period,
                    "symbol": str(sym),
                    "label": str(lbl),
                    "fs": float(fs),
                }
            )
            kept += 1

        subj_rows.append(
            {
                "dataset": "mitbih",
                "version": "v1",
                "subject_id": rec,
                "record_id": rec,
                "group_id": group_id,
                "fs": float(fs),
                "n_samples": int(T),
                "duration_s": float(T / fs) if fs > 0 else None,
                "n_channels": int(x_ct.shape[0]),
                "age": (int(age) if age is not None else None),
                "sex": (str(sex) if sex is not None else None),
                "lead_config": (str(lead_config) if lead_config is not None else ""),
                "medications": (str(medications) if medications is not None else ""),
                "record_notes": (str(record_notes) if record_notes else ""),
                "comments": " | ".join([str(c).strip() for c in comments if str(c).strip()]) if comments else "",
                "sig_names_json": json.dumps(list(sig_names)) if sig_names is not None else "",
                "units_json": json.dumps(list(units)) if units is not None else "",
                "n_windows": int(kept),
            }
        )

        print(f"[ok] record {rec}: T={T} C={x_ct.shape[0]} windows={kept}")

    if len(win_rows) == 0:
        raise RuntimeError("No windows produced. Check label_scheme, ann_extension, and window spec.")

    windows = pd.DataFrame(win_rows)

    # Stable label ids
    label_order = sorted(windows["label"].unique().tolist())
    label_to_id = {lbl: i for i, lbl in enumerate(label_order)}
    windows["label_id"] = windows["label"].map(label_to_id).astype(int)

    subjects_df = pd.DataFrame(subj_rows)

    # Group-based splits
    split_map = make_group_splits(
        group_ids=windows["group_id"].astype(str).tolist(),
        seed=int(args.seed),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
    )
    windows["split"] = windows["group_id"].astype(str).map(split_map)
    subjects_df["split"] = subjects_df["group_id"].astype(str).map(split_map)

    # Write views
    windows_path = views_dir / "windows.parquet"
    subjects_path = views_dir / "subjects.parquet"
    splits_path = views_dir / "splits.json"

    windows.to_parquet(windows_path, index=False)
    subjects_df.to_parquet(subjects_path, index=False)
    splits_path.write_text(json.dumps(split_map, indent=2))

    # Write labels + meta
    (out_root / "labels.json").write_text(json.dumps(label_to_id, indent=2))

    meta = {
        "dataset": "mitbih",
        "version": "v1",
        "raw_root": str(raw_root),
        "n_records": int(len(sorted(set(windows["record_id"].astype(str).tolist())))),
        "n_subject_ids": int(len(sorted(set(windows["subject_id"].astype(str).tolist())))),
        "n_groups": int(len(sorted(set(windows["group_id"].astype(str).tolist())))),
        "n_windows": int(len(windows)),
        "ann_extension": str(args.ann_extension),
        "label_scheme": str(args.label_scheme),
        "window": {"left_s": float(args.window_left_s), "right_s": float(args.window_right_s)},
        "lead_indices": lead_indices,
        "test_period": {"start_s": float(args.test_period_start_s)},
        "label_to_id": label_to_id,
        "splits": {
            "seed": int(args.seed),
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "test_frac": float(args.test_frac),
            "group_id_special_case": {"201": "201_202", "202": "201_202"},
        },
        "notes": [
            "Subjects store is record-level: subjects/{record_id}.npz contains main:(C,T), fs.",
            "Window manifest is beat-centered using annotation samples; labels are mapped via AAMI scheme.",
            "Splits are group-based using group_id to prevent leakage (records 201 and 202 grouped).",
            "example_id is a stable key for joining predictions back to windows.parquet.",
            "in_test_period marks beats occurring after test_period_start_s (commonly used to exclude first 5 minutes).",
        ],
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    # Write dataset-level stats
    cnt = max(1, int(stats["count"]))
    s = stats["sum"]
    ss = stats["sumsq"]
    mean = (s / cnt).tolist() if s is not None else None
    var = (ss / cnt - (s / cnt) ** 2) if (s is not None and ss is not None) else None
    std = np.sqrt(np.maximum(var, 1e-12)).tolist() if var is not None else None

    (out_root / "stats.json").write_text(
        json.dumps({"main": {"count": int(cnt), "mean": mean, "std": std}}, indent=2)
    )

    print(f"\nWrote standardized MIT-BIH dataset to: {out_root}")
    print(f"  - {windows_path}")
    print(f"  - {subjects_path}")
    print(f"  - {splits_path}")
    print("  - meta.json / labels.json / stats.json")


if __name__ == "__main__":
    main()


# run:

# python scripts/standardize/mitbih/standardize_mitbih_v1.py \
#   --raw_root /path/to/biosignals/MIT_BIH/mit-bih-arrhythmia-database-1.0.0 \
#   --out_root data/standardized/mitbih/v1 \
#   --ann_extension atr \
#   --label_scheme aami3 \
#   --window_left_s 0.30 \
#   --window_right_s 0.50 \
#   --lead_indices 0,1


