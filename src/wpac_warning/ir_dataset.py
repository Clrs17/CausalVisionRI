# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Phase 2: HURSAT IR frame pairing and preprocessing.

Input:  wpac_causal_phase1/data/raw/hursat/{SID}/*.nc  (3-hourly storm-centred
        HURSAT-B1 v07b frames) + wpac_tc_dataset_processed.parquet (the 6h
        analysis panel).
Output: data/frames/{SID}.npy  (T_i, 224, 224) float16 raw TBB (Kelvin),
        data/frames/{SID}_times.npy (T_i,) datetime64[s],
        data/frames_index.parquet (row per matched panel timestep),
        data/tbb_norm.json (train-period mean/std, fitted on 2000-2018 only).

Design decisions (R3 discussion): raw TBB (IRWIN, 10.2-11.4 µm) is the input
— NOT the Dvorak BD colour scale, which is a lossy display discretization for
human forecasters.  Frames are centre-cropped to 224x224 (SatMAE-compatible;
~15.7 deg across, covering the 0-8 deg outer annulus).  Pairing to the 6h
panel is by nearest frame within +/-1.5 h (frames are exactly on synoptic
hours for almost all rows).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None

P2_ROOT = Path(__file__).resolve().parents[2]
P1_ROOT = P2_ROOT / "data" / "phase1"
HURSAT_DIR = P1_ROOT / "raw" / "hursat"
PANEL_PATH = P1_ROOT / "processed" / "wpac_tc_dataset_processed.parquet"
OUT_DIR = P2_ROOT / "data" / "frames"
INDEX_PATH = P2_ROOT / "data" / "frames_index.parquet"
NORM_PATH = P2_ROOT / "data" / "tbb_norm.json"

CROP = 224          # central crop (301 -> 224)
GRID = 301
MATCH_TOL_H = 1.5   # hours
TRAIN_YEARS = (2000, 2018)

_FNAME_RE = re.compile(
    r"^(?P<sid>[0-9A-Z]+)\.(?P<name>[A-Za-z0-9_\-:]+)\."
    r"(?P<y>\d{4})\.(?P<m>\d{2})\.(?P<d>\d{2})\.(?P<hhmm>\d{4})\..*\.nc$")


def frame_time_from_name(path: Path) -> Optional[pd.Timestamp]:
    """Parse the frame time from a HURSAT v07b filename."""
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    return pd.Timestamp(
        f"{m['y']}-{m['m']}-{m['d']} {m['hhmm'][:2]}:{m['hhmm'][2:]}",
        tz="UTC",
    )


def build_frame_index() -> pd.DataFrame:
    """Map every downloaded frame to (sid, time) using filename parsing."""
    rows = []
    for storm_dir in sorted(HURSAT_DIR.iterdir()):
        if not storm_dir.is_dir():
            continue
        for f in storm_dir.glob("*.nc"):
            t = frame_time_from_name(f)
            if t is None:
                continue
            rows.append({"SID": storm_dir.name, "time": t, "path": str(f)})
    idx = pd.DataFrame(rows)
    if not idx.empty:
        idx = idx.sort_values(["SID", "time"]).reset_index(drop=True)
    logger.info(f"Frame index: {len(idx)} frames, {idx['SID'].nunique()} storms")
    return idx


def pair_to_panel(frame_idx: pd.DataFrame,
                  panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """For each 6h panel row (SID, ISO_TIME), find the nearest frame within
    MATCH_TOL_H hours."""
    if panel is None:
        panel = pd.read_parquet(PANEL_PATH)
    panel = panel.copy()
    panel["ISO_TIME"] = pd.to_datetime(panel["ISO_TIME"], utc=True)
    env = panel[panel["SHRD"].notna()] if "SHRD" in panel.columns else panel

    records = []
    for sid, grp in env.groupby("SID"):
        fs = frame_idx[frame_idx["SID"] == sid]
        if fs.empty:
            continue
        ftimes = fs["time"].values
        for _, row in grp.iterrows():
            t = row["ISO_TIME"]
            dt = np.abs((ftimes - np.datetime64(t)).astype("timedelta64[m]").astype(float)) / 60.0
            j = int(np.argmin(dt))
            if dt[j] <= MATCH_TOL_H:
                records.append({
                    "SID": sid, "ISO_TIME": t,
                    "frame_path": fs.iloc[j]["path"],
                    "frame_time": fs.iloc[j]["time"],
                    "dt_h": float(dt[j]),
                })
    out = pd.DataFrame(records)
    if not out.empty:
        logger.info(
            f"Paired {len(out)}/{len(env)} panel rows to frames "
            f"({out['SID'].nunique()} storms); mean |dt|={out['dt_h'].mean():.3f} h"
        )
    return out


def load_tbb_crop(path: str) -> np.ndarray:
    """Load IRWIN from a HURSAT frame file, centre-crop to CROP x CROP."""
    if xr is None:
        raise RuntimeError("xarray not installed")
    with xr.open_dataset(path) as ds:
        da = ds["IRWIN"]
        v = np.asarray(da.values, dtype=np.float32).squeeze()
    if v.ndim != 2:
        raise ValueError(f"unexpected IRWIN ndim={v.ndim} in {path}")
    h, w = v.shape
    s = (h - CROP) // 2
    t = (w - CROP) // 2
    return v[s:s + CROP, t:t + CROP]


def build_storm_store(sid: str, pairs: pd.DataFrame) -> Optional[dict]:
    """Build the per-storm float16 frame store from its paired panel rows."""
    sub = pairs[pairs["SID"] == sid].sort_values("ISO_TIME")
    if sub.empty:
        return None
    frames = []
    times = []
    iso_times = []
    for _, row in sub.iterrows():
        try:
            frames.append(load_tbb_crop(row["frame_path"]))
            times.append(np.datetime64(row["frame_time"].tz_localize(None), "s"))
            iso_times.append(np.datetime64(row["ISO_TIME"].tz_localize(None), "s"))
        except Exception as exc:
            logger.warning(f"{sid} {row['ISO_TIME']}: frame read failed ({exc})")
    if not frames:
        return None
    arr = np.stack(frames).astype(np.float16)
    np.save(OUT_DIR / f"{sid}.npy", arr)
    np.save(OUT_DIR / f"{sid}_times.npy", np.asarray(times, dtype="datetime64[s]"))
    np.save(OUT_DIR / f"{sid}_iso.npy", np.asarray(iso_times, dtype="datetime64[s]"))
    return {"SID": sid, "n_frames": len(frames),
            "t0": str(times[0]), "t1": str(times[-1])}


def fit_normalization(pairs: pd.DataFrame) -> dict:
    """Train-period (2000-2018) TBB mean/std over paired frames.

    Fitted ONLY on the training window to keep the validation years out of
    the transform (same discipline as phase 1).
    """
    train = pairs[pd.to_datetime(pairs["ISO_TIME"]).dt.year.between(*TRAIN_YEARS)]
    sums, sq_sums, n = 0.0, 0.0, 0
    for sid in train["SID"].unique():
        f = OUT_DIR / f"{sid}.npy"
        if not f.exists():
            continue
        arr = np.load(f).astype(np.float32)
        yrs = pd.to_datetime(np.load(OUT_DIR / f"{sid}_iso.npy")).year
        sel = (yrs >= TRAIN_YEARS[0]) & (yrs <= TRAIN_YEARS[1])
        arr = arr[sel]
        if len(arr) == 0:
            continue
        v = arr[~np.isnan(arr)]
        sums += float(v.sum())
        sq_sums += float((v * v).sum())
        n += v.size
    mean = sums / max(n, 1)
    var = sq_sums / max(n, 1) - mean ** 2
    std = float(np.sqrt(max(var, 1e-9)))
    stats = {"mean": float(mean), "std": std, "n_pixels": int(n),
             "train_years": list(TRAIN_YEARS)}
    NORM_PATH.write_text(json.dumps(stats, indent=2))
    logger.info(f"TBB normalization: mean={mean:.2f} K, std={std:.2f} K "
                f"(train {TRAIN_YEARS}, {n/1e6:.0f}M px)")
    return stats


def main() -> int:
    logger.add(str(P2_ROOT / "logs" / "ir_dataset.log"), level="INFO",
               rotation="10 MB")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_idx = build_frame_index()
    pairs = pair_to_panel(frame_idx)
    pairs.to_parquet(INDEX_PATH)
    logger.info(f"Index saved to {INDEX_PATH}")

    sids = pairs["SID"].unique()
    logger.info(f"Building frame stores for {len(sids)} storms")
    recs = []
    for i, sid in enumerate(sids, 1):
        rec = build_storm_store(sid, pairs)
        if rec:
            recs.append(rec)
        if i % 50 == 0 or i == len(sids):
            logger.info(f"store progress: {i}/{len(sids)}")
    mdf = pd.DataFrame(recs)
    logger.info(f"Stores built: {len(mdf)} storms, {mdf['n_frames'].sum()} frames")

    stats = fit_normalization(pairs)
    print(f"DONE: {len(mdf)} storms, {mdf['n_frames'].sum()} frames, "
          f"TBB mean={stats['mean']:.2f} K std={stats['std']:.2f} K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
