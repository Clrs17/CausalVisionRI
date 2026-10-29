# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""
Earth Data Hub (EDH) parallel downloader for ERA5 per-storm subsets.

Design:
- Multi-process: each worker handles one storm (process-safe, no dataset sharing)
- Bulk-load per batch: ~8 timesteps at a time to balance speed vs. connection stability
- Automatic retry with exponential backoff
- Skip existing files (resume support)
- Quota tracking (EDH monthly limit: 500k requests)
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

# ---------------------------------------------------------------------------
# Variable name mappings: CDS convention -> EDH native short names
# ---------------------------------------------------------------------------
PL_VAR_MAP: Dict[str, str] = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "vertical_velocity": "w",
    "divergence": "d",
    "potential_vorticity": "pv",
}

SL_VAR_MAP: Dict[str, str] = {
    "sea_surface_temperature": "sst",
    "mean_sea_level_pressure": "msl",
    "2m_temperature": "t2m",
    "surface_latent_heat_flux": "slhf",
    "top_net_thermal_radiation": "ttr",
}

# Reverse map for renaming after load
PL_RENAME = {v: k for k, v in PL_VAR_MAP.items()}
SL_RENAME = {v: k for k, v in SL_VAR_MAP.items()}


# ---------------------------------------------------------------------------
# Worker-level helpers (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _build_times(t0: pd.Timestamp, t1: pd.Timestamp) -> List[pd.Timestamp]:
    """Build 00/06/12/18 UTC timestamps between t0 and t1 inclusive."""
    times = []
    dr = pd.date_range(t0.floor("D"), t1.ceil("D"), freq="D")
    for d in dr:
        for h in (0, 6, 12, 18):
            t = pd.Timestamp(d) + pd.Timedelta(hours=h)
            if t0 <= t <= t1:
                times.append(t)
    return times


def _download_one_storm(
    sid: str,
    name: str,
    lats: List[float],
    lons: List[float],
    times_bt: List[pd.Timestamp],
    pl_url: str,
    sl_url: str,
    output_dir: Path,
    buffer_deg: float,
    buffer_days: int,
    pressure_levels: List[int],
    batch_size: int,
    retry_max: int,
) -> Dict[str, Any]:
    """
    Worker function: opens datasets in-process, downloads one storm.
    Returns status dict.
    """
    pl_file = output_dir / f"{sid}_pl.nc"
    sl_file = output_dir / f"{sid}_sl.nc"

    # Skip if both exist
    if pl_file.exists() and sl_file.exists():
        return {
            "sid": sid,
            "name": name,
            "status": "skipped",
            "pl": str(pl_file),
            "sl": str(sl_file),
            "requests": 0,
            "error": None,
        }

    # Compute bbox
    lats = np.array(lats, dtype=float)
    lons = np.array(lons, dtype=float)
    n = float(min(90.0, lats.max() + buffer_deg))
    s = float(max(-90.0, lats.min() - buffer_deg))
    e = float(min(360.0, lons.max() + buffer_deg))
    w = float(max(0.0, lons.min() - buffer_deg))

    # Time range with buffer
    t0 = pd.to_datetime(min(times_bt)) - pd.Timedelta(days=buffer_days)
    t1 = pd.to_datetime(max(times_bt)) + pd.Timedelta(days=buffer_days)
    times = _build_times(t0, t1)

    if not times:
        return {"sid": sid, "name": name, "status": "no_times", "error": "Empty time range"}

    request_count = 0
    error_msg = None

    def _open_dataset(url: str, label: str) -> xr.Dataset:
        """Open Zarr dataset with retry."""
        for attempt in range(1, retry_max + 1):
            try:
                t0 = time.time()
                ds = xr.open_dataset(url, chunks={}, engine="zarr")
                logger.info(f"[{sid}] Opened {label} dataset in {time.time()-t0:.1f}s")
                return ds
            except Exception as exc:
                logger.warning(f"[{sid}] Open {label} failed (attempt {attempt}): {exc}")
                if attempt >= retry_max:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"[{sid}] Cannot open {label} dataset")

    def _load_var(
        ds: xr.Dataset,
        var: str,
        time_str: str,
        levels: Optional[List[int]],
        sid: str,
        label: str,
    ) -> xr.DataArray:
        """Load one variable at one timestep with retry."""
        nonlocal request_count
        for attempt in range(1, retry_max + 1):
            try:
                kwargs = dict(
                    valid_time=time_str,
                    latitude=slice(n, s),
                    longitude=slice(w, e),
                )
                if levels is not None:
                    kwargs["isobaricInhPa"] = levels
                da = ds[var].sel(**kwargs)
                da = da.load()
                request_count += 1
                return da
            except Exception as exc:
                err_str = str(exc)
                is_network_err = any(k in err_str.lower() for k in [
                    "premature close", "timeout", "connection", "reset", "eof", "broken pipe",
                    "content length", "not enough data"
                ])
                if is_network_err:
                    wait = min(120, 10 * (2 ** attempt))
                    logger.warning(
                        f"[{sid}] {label} {var} {time_str} network error (attempt {attempt}): {exc} | retry in {wait}s"
                    )
                    time.sleep(wait)
                elif attempt >= retry_max:
                    raise RuntimeError(f"{label} {var} {time_str} failed after {retry_max} attempts: {exc}")
                else:
                    logger.warning(f"[{sid}] {label} {var} {time_str} error (attempt {attempt}): {exc}")
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"{label} {var} {time_str} exhausted retries")

    try:
        # ---- Pressure Levels ----
        if not pl_file.exists():
            pl_batches: List[xr.Dataset] = []
            pl_vars = list(PL_VAR_MAP.values())
            pl_rename = PL_RENAME
            ds_pl: Optional[xr.Dataset] = None
            reopen_every = 3

            for i, t in enumerate(times):
                if ds_pl is None or i % reopen_every == 0:
                    if ds_pl is not None:
                        ds_pl.close()
                        del ds_pl
                    ds_pl = _open_dataset(pl_url, "PL")

                time_str = t.strftime("%Y-%m-%dT%H:%M:%S")
                da_list = []
                for var in pl_vars:
                    da = _load_var(ds_pl, var, time_str, pressure_levels, sid, "PL")
                    da_list.append(da)
                ds_t = xr.merge(da_list)
                logger.info(f"[{sid}] PL {time_str} merged {len(pl_vars)} vars")
                pl_batches.append(ds_t)

            if ds_pl is not None:
                ds_pl.close()
                del ds_pl

            if pl_batches:
                ds_pl_storm = xr.concat(pl_batches, dim="valid_time")
                ds_pl_storm = ds_pl_storm.rename(pl_rename)
                ds_pl_storm.to_netcdf(pl_file)
                logger.success(f"[{sid}] Saved PL: {pl_file}")

        # ---- Single Levels ----
        if not sl_file.exists():
            sl_batches: List[xr.Dataset] = []
            sl_vars = list(SL_VAR_MAP.values())
            sl_rename = SL_RENAME
            ds_sl: Optional[xr.Dataset] = None

            for i, t in enumerate(times):
                if ds_sl is None or i % reopen_every == 0:
                    if ds_sl is not None:
                        ds_sl.close()
                        del ds_sl
                    ds_sl = _open_dataset(sl_url, "SL")

                time_str = t.strftime("%Y-%m-%dT%H:%M:%S")
                da_list = []
                for var in sl_vars:
                    da = _load_var(ds_sl, var, time_str, None, sid, "SL")
                    da_list.append(da)
                ds_t = xr.merge(da_list)
                logger.info(f"[{sid}] SL {time_str} merged {len(sl_vars)} vars")
                sl_batches.append(ds_t)

            if ds_sl is not None:
                ds_sl.close()
                del ds_sl

            if sl_batches:
                ds_sl_storm = xr.concat(sl_batches, dim="valid_time")
                ds_sl_storm = ds_sl_storm.rename(SL_RENAME)
                ds_sl_storm.to_netcdf(sl_file)
                logger.success(f"[{sid}] Saved SL: {sl_file}")

        status = "success"

    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        logger.error(f"[{sid}] Download failed: {error_msg}")

    return {
        "sid": sid,
        "name": name,
        "status": status,
        "pl": str(pl_file) if pl_file.exists() else None,
        "sl": str(sl_file) if sl_file.exists() else None,
        "requests": request_count,
        "timesteps": len(times),
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Main downloader class
# ---------------------------------------------------------------------------

class EDHDownloader:
    """Parallel ERA5 downloader via Earth Data Hub Zarr stores."""

    def __init__(
        self,
        token: str,
        output_dir: str = "data/phase1/raw/era5",
        buffer_deg: float = 15.0,
        buffer_days: int = 1,
        pressure_levels: Optional[List[int]] = None,
        max_workers: int = 4,
        batch_size: int = 8,
        retry_max: int = 3,
        skip_existing: bool = True,
    ) -> None:
        self.token = token
        self.pl_url = (
            f"https://edh:{token}@api.earthdatahub.destine.eu"
            "/era5/reanalysis-era5-pressure-levels-v0.zarr"
        )
        self.sl_url = (
            f"https://edh:{token}@api.earthdatahub.destine.eu"
            "/era5/reanalysis-era5-single-levels-v0.zarr"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_deg = buffer_deg
        self.buffer_days = buffer_days
        self.pressure_levels = pressure_levels or [1000, 925, 850, 700, 500, 300, 200]
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.retry_max = retry_max
        self.skip_existing = skip_existing

        # Stats
        self.stats: Dict[str, Any] = {
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "total_requests": 0,
            "total_timesteps": 0,
        }
        self.failed_storms: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_storm(self, sid: str, df_storm: pd.DataFrame) -> Dict[str, Any]:
        """Download a single storm (synchronous, for testing)."""
        lats = df_storm["USA_LAT"].tolist()
        lons = df_storm["USA_LON"].tolist()
        times = pd.to_datetime(df_storm["ISO_TIME"]).tolist()
        name = str(df_storm["NAME"].iloc[0]) if "NAME" in df_storm.columns else ""

        return _download_one_storm(
            sid=sid,
            name=name,
            lats=lats,
            lons=lons,
            times_bt=times,
            pl_url=self.pl_url,
            sl_url=self.sl_url,
            output_dir=self.output_dir,
            buffer_deg=self.buffer_deg,
            buffer_days=self.buffer_days,
            pressure_levels=self.pressure_levels,
            batch_size=self.batch_size,
            retry_max=self.retry_max,
        )

    def download_all(self, df_best_track: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Parallel download for all storms in best-track DataFrame.
        Uses ProcessPoolExecutor for true parallelism.
        """
        sid_col = "SID"
        if sid_col not in df_best_track.columns:
            raise ValueError(f"Best-track DataFrame missing '{sid_col}' column")

        sids = df_best_track[sid_col].unique()
        logger.info(f"Storms to download: {len(sids)}")
        logger.info(f"Workers: {self.max_workers} | Batch size: {self.batch_size} timesteps")
        logger.info(f"Output dir: {self.output_dir}")

        # Build task list
        tasks = []
        for sid in sids:
            storm_df = df_best_track[df_best_track[sid_col] == sid]
            tasks.append({
                "sid": sid,
                "name": str(storm_df["NAME"].iloc[0]) if "NAME" in storm_df.columns else "",
                "lats": storm_df["USA_LAT"].tolist(),
                "lons": storm_df["USA_LON"].tolist(),
                "times": pd.to_datetime(storm_df["ISO_TIME"]).tolist(),
            })

        # Filter already completed
        if self.skip_existing:
            pending = []
            for t in tasks:
                pl_f = self.output_dir / f"{t['sid']}_pl.nc"
                sl_f = self.output_dir / f"{t['sid']}_sl.nc"
                if pl_f.exists() and sl_f.exists():
                    logger.info(f"Skip existing: {t['sid']}")
                    self.stats["skipped"] += 1
                else:
                    pending.append(t)
            tasks = pending
            logger.info(f"Pending downloads: {len(tasks)}")

        if not tasks:
            logger.info("All storms already downloaded.")
            return []

        # Run workers: pure serial loop (Windows ProcessPoolExecutor is unstable)
        results: List[Dict[str, Any]] = []
        start_time = time.time()

        for i, t in enumerate(tasks):
            sid = t["sid"]
            try:
                result = _download_one_storm(
                    sid=sid,
                    name=t["name"],
                    lats=t["lats"],
                    lons=t["lons"],
                    times_bt=t["times"],
                    pl_url=self.pl_url,
                    sl_url=self.sl_url,
                    output_dir=self.output_dir,
                    buffer_deg=self.buffer_deg,
                    buffer_days=self.buffer_days,
                    pressure_levels=self.pressure_levels,
                    batch_size=self.batch_size,
                    retry_max=self.retry_max,
                )
                results.append(result)
                status = result["status"]
                if status == "success":
                    self.stats["success"] += 1
                elif status == "skipped":
                    self.stats["skipped"] += 1
                else:
                    self.stats["failed"] += 1
                    self.failed_storms.append(result)
                self.stats["total_requests"] += result.get("requests", 0)
                self.stats["total_timesteps"] += result.get("timesteps", 0)

                elapsed = time.time() - start_time
                done = self.stats["success"] + self.stats["failed"] + self.stats["skipped"]
                total = len(tasks)
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                logger.info(
                    f"Progress: {done}/{total} | "
                    f"Success: {self.stats['success']} | "
                    f"Failed: {self.stats['failed']} | "
                    f"Skipped: {self.stats['skipped']} | "
                    f"ETA: {eta/3600:.1f}h | "
                    f"Storm: {i+1}/{total}"
                )
            except Exception as exc:
                logger.error(f"[{sid}] Download crashed: {exc}")
                self.stats["failed"] += 1
                self.failed_storms.append({"sid": sid, "status": "crash", "error": str(exc)})

        # Summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"Total time: {elapsed/3600:.1f}h")
        logger.info(f"Success: {self.stats['success']} | Failed: {self.stats['failed']} | Skipped: {self.stats['skipped']}")
        logger.info(f"Total HTTP requests: {self.stats['total_requests']}")
        logger.info(f"Total timesteps: {self.stats['total_timesteps']}")
        quota_pct = (self.stats['total_requests'] / 500_000) * 100
        logger.info(f"EDH quota used: ~{quota_pct:.1f}% (of 500k/month)")
        if self.failed_storms:
            logger.warning(f"Failed storms: {len(self.failed_storms)}")
            for f in self.failed_storms:
                logger.warning(f"  - {f['sid']}: {f.get('error', 'unknown')}")
        logger.info("=" * 60)

        # Save retry JSON
        if self.failed_storms:
            retry_path = self.output_dir / "edh_failed.json"
            retry_path.write_text(json.dumps(self.failed_storms, indent=2, default=str), encoding="utf-8")
            logger.info(f"Failed storm log saved to {retry_path}")

        return results


# ---------------------------------------------------------------------------
# CLI / direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Simple CLI for testing one storm
    if len(sys.argv) >= 3:
        token = sys.argv[1]
        sid = sys.argv[2]
        # Expect a CSV with columns: ISO_TIME, USA_LAT, USA_LON, NAME
        csv_path = sys.argv[3] if len(sys.argv) > 3 else None

        downloader = EDHDownloader(token=token, max_workers=1, batch_size=8)

        if csv_path:
            df = pd.read_csv(csv_path)
            df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True)
            result = downloader.download_storm(sid, df)
        else:
            # Minimal test: Hagibis 2019
            print("Usage: python download_edh.py <token> <sid> <storm_csv>")
            print("Or import and use EDHDownloader class directly.")
            sys.exit(1)

        print(json.dumps(result, indent=2, default=str))
    else:
        print("Earth Data Hub ERA5 Downloader")
        print("Import EDHDownloader and call download_all() with best-track DataFrame.")
