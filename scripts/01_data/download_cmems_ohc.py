# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""
CMEMS GLORYS12V1 Ocean Heat Content per-storm downloader.

Product: GLOBAL_MULTIYEAR_PHY_001_030
Dataset: cmems_mod_glo_phy_my_0.083deg_P1D-m (daily, 1/12°, 1993-present)
Variable: thetao (potential temperature, °C)

Strategy:
  - copernicusmarine.subset() per storm (spatial + temporal + depth subset)
  - Daily resolution (ocean changes slowly)
  - Upper 500m
  - Compute OHC relative to 26°C
  - Save compact per-storm NetCDF

Usage:
  python download_cmems_ohc.py --max 3     # test 3 storms
  python download_cmems_ohc.py             # all 387 storms
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
HYCOM_DIR = ROOT / "data" / "phase1" / "raw" / "hycom"
OHC_DIR = ROOT / "data" / "phase1" / "raw" / "ohc"  # new CMEMS-based OHC dir
QUEUE_PATH = ROOT / "data" / "phase1" / "processed" / "cds_queue_strong.csv"

# CMEMS config
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
DATASET_VERSION = "202311"
VARIABLE = "thetao"
DEPTH_MIN = 0.49  # m (shallowest CMEMS level)
DEPTH_MAX = 500.0  # m
SPATIAL_BUFFER = 15  # degrees
TEMPORAL_BUFFER_DAYS = 2  # extra days for safe coverage

OHC_RHO = 1025.0  # kg/m³
OHC_CP = 4000.0  # J/(kg·K)


def load_queue() -> pd.DataFrame:
    return pd.read_csv(QUEUE_PATH)


def storm_info(sid: str) -> Tuple[pd.Series, pd.DataFrame]:
    queue = load_queue()
    row = queue[queue["SID"] == sid].iloc[0]
    bt = pd.read_parquet(ROOT / "data" / "phase1" / "processed" / "wpac_tc_dataset.parquet")
    bt_storm = bt[bt["SID"] == sid].copy()
    bt_storm["ISO_TIME"] = pd.to_datetime(bt_storm["ISO_TIME"], utc=True)
    return row, bt_storm


def get_bbox(bt_storm: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Get [N, W, S, E] bbox with buffer (for CMEMS)."""
    lats = bt_storm["USA_LAT"].values
    lons = bt_storm["USA_LON"].values
    n = min(90, lats.max() + SPATIAL_BUFFER)
    s = max(-90, lats.min() - SPATIAL_BUFFER)
    e = min(360, lons.max() + SPATIAL_BUFFER)
    w = max(-180, lons.min() - SPATIAL_BUFFER)  # CMEMS uses -180 to 180
    return float(n), float(w), float(s), float(e)


def compute_ohc(da_temp: xr.DataArray, depths: np.ndarray) -> xr.DataArray:
    """Compute OHC (kJ/cm²) from (time, depth, lat, lon) temperature field.

    OHC = ρ * Cp * ∫ (T - 26) dz, integrated over depth dimension.
    Trapezoidal rule: sum over k of (T_k + T_{k+1})/2 * (z_{k+1} - z_k)
    """
    nd = da_temp.sizes["depth"]
    # Temperature excess above 26°C
    excess = da_temp.where(da_temp > 26) - 26.0
    excess = excess.fillna(0.0)

    # Trapezoidal integration over depth
    ohc = xr.zeros_like(excess.isel(depth=0))
    for k in range(nd - 1):
        dz = depths[k + 1] - depths[k]
        layer_mean = (excess.isel(depth=k) + excess.isel(depth=k + 1)) / 2.0
        ohc += layer_mean * dz

    ohc = ohc * OHC_RHO * OHC_CP / 1e7  # kJ/cm²
    ohc.name = "OHC"
    ohc.attrs = {
        "units": "kJ/cm²",
        "long_name": "Ocean Heat Content relative to 26°C",
        "integration_depth": f"{depths[0]:.1f}-{depths[-1]:.0f}m",
        "source": "CMEMS GLORYS12V1",
    }
    return ohc


def download_one_storm(sid: str) -> bool:
    out_dir = OHC_DIR / sid
    out_path = out_dir / f"{sid}_ohc.nc"

    if out_path.exists() and out_path.stat().st_size > 1024:
        logger.info(f"[{sid}] Already complete")
        return True

    row, bt_storm = storm_info(sid)
    n, w, s, e = get_bbox(bt_storm)
    t0 = pd.to_datetime(row["start"]) - pd.Timedelta(days=TEMPORAL_BUFFER_DAYS)
    t1 = pd.to_datetime(row["end"]) + pd.Timedelta(days=TEMPORAL_BUFFER_DAYS)

    logger.info(f"[{sid}] {row['name']} ({row['year']}): {t0.date()}-{t1.date()}, "
                f"bbox=[{n:.0f}N,{s:.0f}S,{w:.0f}W,{e:.0f}E]")

    temp_file = out_dir / f"{sid}_temp3d.nc"

    for attempt in range(1, 4):
        try:
            import copernicusmarine as cm
            cm.subset(
                dataset_id=DATASET_ID,
                dataset_version=DATASET_VERSION,
                variables=[VARIABLE],
                minimum_longitude=w,
                maximum_longitude=e,
                minimum_latitude=s,
                maximum_latitude=n,
                start_datetime=t0.strftime("%Y-%m-%dT00:00:00"),
                end_datetime=t1.strftime("%Y-%m-%dT00:00:00"),
                minimum_depth=DEPTH_MIN,
                maximum_depth=DEPTH_MAX,
                output_directory=str(out_dir),
                output_filename=temp_file.name,
                overwrite=True,
            )
            break
        except Exception as exc:
            logger.warning(f"[{sid}] Attempt {attempt} failed: {exc}")
            if attempt >= 3:
                logger.error(f"[{sid}] Failed after 3 attempts")
                return False
            time.sleep(30 * attempt)

    if not temp_file.exists():
        logger.error(f"[{sid}] Download produced no file")
        return False

    # Load, compute OHC, save compact
    ds = xr.open_dataset(temp_file)
    temp = ds[VARIABLE]  # (time, depth, lat, lon)
    depths = ds.depth.values
    ds.close()

    ohc = compute_ohc(temp, depths)
    ohc.to_netcdf(out_path)

    # Remove temp file
    temp_file.unlink()
    size_mb = out_path.stat().st_size / 1e6
    logger.success(f"[{sid}] Saved {out_path} ({size_mb:.1f} MB, {temp.sizes['time']} daily profiles)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--sid", default=None)
    args = parser.parse_args()

    logger.add(ROOT / "data" / "phase1" / "processed" / "logs" / "cmems_ohc.log",
               rotation="10 MB", retention="30 days", level="INFO")
    OHC_DIR.mkdir(parents=True, exist_ok=True)

    queue = load_queue()
    if args.sid:
        queue = queue[queue["SID"] == args.sid]

    success = 0
    for i, (_, row) in enumerate(queue.iterrows()):
        sid = row["SID"]
        ok = download_one_storm(sid)
        if ok:
            success += 1
        logger.info(f"Progress: {i+1}/{len(queue)} | ok={success}")

        if args.max and i + 1 >= args.max:
            break
        time.sleep(2)  # gentle rate limit

    logger.info(f"Done: {success}/{min(len(queue), args.max or len(queue))}")


if __name__ == "__main__":
    main()
