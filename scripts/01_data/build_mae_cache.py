# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""AUDIT-ONLY legacy converter: ALL raw WNP HURSAT .nc frames -> float16 memmap.

This path reads ``wpac_causal_phase1/data/raw/hursat`` directly and therefore
bypasses the QC v1 frame library (duplicate acquisition times, non-physical
fill).  It is kept for historical auditing only: the production WNP and global
MAE caches are built from ``data/global_hursat/frames_qc`` by
``scripts/merge_global_cache.py`` (``data/mae_cache_wnp_alltc`` /
``data/mae_cache_global_qc``).

Why a cache: xarray decoding of these files is pathologically slow (~18
s/file); direct netCDF4 reads are ~80 ms/file.  Streaming .nc reads during
MAE training would bottleneck IO, so we convert once.

Reads every ``../wpac_causal_phase1/data/raw/hursat/{SID}/*.nc`` (53,396
frames / 750 storms), centre-crops IRWIN to 224x224 and writes (legacy
directory; pass ``--out-dir`` to write elsewhere):

  data/mae_cache/frames.npy      (N, 224, 224) float16 memmap
  data/mae_cache/index.parquet   row per frame: SID, time, src_path
  data/mae_cache/progress.txt    resume cursor (next row to write)

Resumable: re-run to continue from progress.txt.  ``--budget-s`` stops
writing after N seconds (exits 2 when unfinished, 0 when complete) so the
job can be split across shell invocations.

Usage (from the release root):
    python scripts/01_data/build_mae_cache.py --workers 24
    python scripts/01_data/build_mae_cache.py --workers 24 --budget-s 280
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.ir_dataset import frame_time_from_name  # filename -> timestamp

HURSAT_DIR = P2_ROOT / "data" / "phase1" / "data" / "raw" / "hursat"
OUT_DIR = P2_ROOT / "data" / "mae_cache"
CROP = 224


def _read_one(args):
    """Worker: read one .nc, centre-crop, return (idx, float16 array|None)."""
    idx, path = args
    try:
        from netCDF4 import Dataset  # per-process import
        with Dataset(path) as ds:
            v = ds["IRWIN"][:]
        v = np.asarray(np.ma.filled(v, np.nan), dtype=np.float32).squeeze()
        if v.ndim != 2 or v.shape[0] < CROP or v.shape[1] < CROP:
            return idx, None
        s0 = (v.shape[0] - CROP) // 2
        s1 = (v.shape[1] - CROP) // 2
        return idx, v[s0:s0 + CROP, s1:s1 + CROP].astype(np.float16)
    except Exception:
        return idx, None


def build_manifest() -> pd.DataFrame:
    rows = []
    for storm_dir in sorted(HURSAT_DIR.iterdir()):
        if not storm_dir.is_dir():
            continue
        for f in sorted(storm_dir.glob("*.nc")):
            t = frame_time_from_name(f)
            if t is None:
                continue
            rows.append({"SID": storm_dir.name, "time": t, "src_path": str(f)})
    df = pd.DataFrame(rows)
    # deterministic global order: by storm, then time; frame_row is the row
    # number inside frames.npy and is REQUIRED by the holdout index filter
    df = df.sort_values(["SID", "time"]).reset_index(drop=True)
    df["frame_row"] = np.arange(len(df), dtype=np.int64)
    return df


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--budget-s", type=float, default=None,
                    help="stop after this many seconds (exit 2 if unfinished)")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="cache directory (legacy default data/mae_cache; the "
                         "production caches come from "
                         "scripts/merge_global_cache.py on frames_qc)")
    args = ap.parse_args()

    OUT_DIR = Path(args.out_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "index.parquet"
    frames_path = OUT_DIR / "frames.npy"
    progress_path = OUT_DIR / "progress.txt"

    if manifest_path.exists():
        df = pd.read_parquet(manifest_path)
    else:
        print("scanning raw HURSAT tree ...", flush=True)
        df = build_manifest()
        df.to_parquet(manifest_path)
    N = len(df)

    mm = np.lib.format.open_memmap(frames_path, mode="r+" if frames_path.exists() else "w+",
                                   dtype=np.float16, shape=(N, CROP, CROP))
    start_idx = int(progress_path.read_text().strip()) if progress_path.exists() else 0
    print(f"frames: {N}, resuming at {start_idx}", flush=True)
    if start_idx >= N:
        print("already complete")
        return 0

    t0 = time.time()
    done = start_idx
    n_bad = 0
    with Pool(args.workers) as pool:
        it = pool.imap(_read_one, ((i, df["src_path"].iloc[i]) for i in range(start_idx, N)),
                       chunksize=16)
        for idx, arr in it:
            if arr is None:
                mm[idx] = np.nan
                n_bad += 1
            else:
                mm[idx] = arr
            done = idx + 1
            if done % 2000 == 0:
                mm.flush()
                progress_path.write_text(str(done))
                el = time.time() - t0
                rate = (done - start_idx) / max(el, 1e-9)
                print(f"  {done}/{N} ({rate:.0f}/s, bad={n_bad})", flush=True)
            if args.budget_s and (time.time() - t0) > args.budget_s:
                mm.flush()
                progress_path.write_text(str(done))
                print(f"budget reached at {done}/{N}; rerun to continue", flush=True)
                return 2
    mm.flush()
    progress_path.write_text(str(done))
    print(f"DONE: {done}/{N} frames cached ({n_bad} unreadable -> NaN) -> {frames_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
