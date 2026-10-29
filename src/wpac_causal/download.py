# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""ERA5 per-storm spatiotemporal subset download with resume and retry."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from wpac_causal.utils import Config, normalize_lon, phase_timer

try:
    import cdsapi
except Exception:  # pragma: no cover
    cdsapi = None  # type: ignore


class ERA5Downloader:
    """Download ERA5 pressure- and single-level reanalysis per storm."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.output_dir = Path(config.get("data.era5_download_dir", "data/phase1/raw/era5"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retry_json = Path(config.get("data.retry_json", "data/phase1/processed/retry.json"))
        self.retry_json.parent.mkdir(parents=True, exist_ok=True)

        # CDS datasets
        self.ds_pl = config.get("era5.cds_dataset_pl", "reanalysis-era5-pressure-levels")
        self.ds_sl = config.get("era5.cds_dataset_sl", "reanalysis-era5-single-levels")
        self.pl_vars = config.get("era5.pl_variables", [])
        self.sl_vars = config.get("era5.sl_variables", [])
        self.pressure_levels = config.get("era5.pressure_levels", [1000, 925, 850, 700, 500, 300, 200])
        self.buffer_deg = config.get("era5.spatial_buffer_deg", 15)
        self.buffer_days = config.get("era5.temporal_buffer_days", 1)
        self.retry_max = config.get("era5.download_retry", 3)
        self.timeout = config.get("era5.download_timeout", 600)
        self.sleep_sec = config.get("era5.download_sleep", 60)
        self.skip_existing = config.get("era5.skip_existing", True)

        self._client = None

    # ------------------------------------------------------------------
    # CDS client (lazy init)
    # ------------------------------------------------------------------

    def _get_client(self, wait_until_complete: bool = True):
        if self._client is None:
            if cdsapi is None:
                raise RuntimeError("cdsapi is not installed.")
            # Try project-root .cdsapirc first, then fallback to ~/.cdsapirc
            creds = self._read_cds_credentials()
            kwargs = {"timeout": self.timeout, "quiet": False}
            # New CDS-Beta API supports wait_until_complete; legacy API does not.
            try:
                import inspect
                sig = inspect.signature(cdsapi.Client.__init__)
                if "wait_until_complete" in sig.parameters:
                    kwargs["wait_until_complete"] = wait_until_complete
            except Exception:
                pass
            if creds:
                self._client = cdsapi.Client(
                    url=creds.get("url"),
                    key=creds.get("key"),
                    **kwargs,
                )
            else:
                self._client = cdsapi.Client(**kwargs)
        return self._client

    def _read_cds_credentials(self) -> Optional[Dict[str, str]]:
        """Read CDS API credentials from project root or home directory."""
        import os
        candidates = [
            Path(__file__).resolve().parents[2] / ".cdsapirc",
            Path(os.getcwd()) / ".cdsapirc",
            Path.home() / ".cdsapirc",
        ]
        for p in candidates:
            if p.exists():
                creds = {}
                for line in p.read_text(encoding="utf-8").splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        creds[k.strip()] = v.strip()
                if "url" in creds and "key" in creds:
                    logger.info(f"Using CDS credentials from {p}")
                    return creds
        return None

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _storm_bbox(self, df_storm: pd.DataFrame, lat_col: str, lon_col: str) -> Tuple[float, float, float, float]:
        """Return [N, W, S, E] bbox with buffer."""
        lats = df_storm[lat_col].astype(float)
        lons = df_storm[lon_col].astype(float) % 360.0  # pandas Series: min/max skip NaN
        n = min(90.0, lats.max() + self.buffer_deg)
        s = max(-90.0, lats.min() - self.buffer_deg)
        e = min(360.0, lons.max() + self.buffer_deg)
        w = max(0.0, lons.min() - self.buffer_deg)
        # Handle dateline crossing: if span > 180, split (rare in WPAC but possible)
        return float(n), float(w), float(s), float(e)

    def _storm_time_range(self, df_storm: pd.DataFrame, time_col: str) -> Tuple[str, str]:
        """Return start/end ISO date strings with buffer days."""
        t0 = pd.to_datetime(df_storm[time_col].min()) - pd.Timedelta(days=self.buffer_days)
        t1 = pd.to_datetime(df_storm[time_col].max()) + pd.Timedelta(days=self.buffer_days)
        return t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")

    def _build_times(self, start_date: str, end_date: str) -> List[str]:
        """Build 00:00/06:00/12:00/18:00 list between dates inclusive."""
        times = []
        dr = pd.date_range(start_date, end_date, freq="D")
        for d in dr:
            for h in ["00:00", "06:00", "12:00", "18:00"]:
                times.append(f"{d.strftime('%Y-%m-%d')}/{h}")
        return times

    def _build_request_pl(self, sid: str, bbox: Tuple[float, ...], times: List[str]) -> Dict:
        n, w, s, e = bbox
        return {
            "product_type": ["reanalysis"],
            "variable": self.pl_vars,
            "pressure_level": [str(p) for p in self.pressure_levels],
            # date range avoids the year×month×day cartesian inflation that
            # blows past CDS cost limits for cross-month/year storms
            "date": f"{min(t[:10] for t in times)}/{max(t[:10] for t in times)}",
            "time": sorted({t[11:] for t in times}),
            "area": [n, w, s, e],
            "format": "netcdf",
        }

    def _build_request_sl(self, sid: str, bbox: Tuple[float, ...], times: List[str]) -> Dict:
        n, w, s, e = bbox
        return {
            "product_type": ["reanalysis"],
            "variable": self.sl_vars,
            "date": f"{min(t[:10] for t in times)}/{max(t[:10] for t in times)}",
            "time": sorted({t[11:] for t in times}),
            "area": [n, w, s, e],
            "format": "netcdf",
        }

    # ------------------------------------------------------------------
    # Download logic
    # ------------------------------------------------------------------

    def _download_file(self, dataset: str, request: Dict, target: Path) -> bool:
        """Return True on success, False on failure.

        The current CDS API may return a ZIP archive (e.g. single-level
        requests mixing accumulated and instantaneous variables are split
        into stepType-instant/accum sub-files). Detect the PK header and
        unpack into the pipeline's expected `{sid}_sl_instant/accum.nc`
        layout; the raw zip is kept as `<name>.zip` (never deleted).
        """
        if self.skip_existing and target.exists() and target.stat().st_size > 0:
            logger.info(f"Skip existing: {target}")
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, self.retry_max + 1):
            try:
                client = self._get_client()
                logger.info(f"Downloading {dataset} -> {target} (attempt {attempt})")
                client.retrieve(dataset, request, str(target))
                if target.exists() and target.stat().st_size > 0:
                    logger.success(f"Saved {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)")
                    self._maybe_unpack_zip(target)
                    return True
            except Exception as exc:
                logger.warning(f"Download failed (attempt {attempt}): {exc}")
                if attempt < self.retry_max:
                    logger.info(f"Sleep {self.sleep_sec}s before retry...")
                    time.sleep(self.sleep_sec)
        return False

    @staticmethod
    def _maybe_unpack_zip(target: Path) -> None:
        """If `target` is a ZIP archive (CDS split-file response), extract its
        members and rename stepType-instant/accum NetCDFs to the pipeline's
        `<sid>_sl_instant.nc` / `<sid>_sl_accum.nc` convention. The archive is
        preserved as `<name>.zip`."""
        import zipfile
        if not zipfile.is_zipfile(target):
            return
        sid_dir = target.parent
        base = target.stem  # e.g. <sid>_sl
        zip_path = target.with_suffix(".zip")
        target.rename(zip_path)
        logger.info(f"CDS returned a ZIP archive; unpacking {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if not member.endswith(".nc"):
                    continue
                out = None
                if "stepType-instant" in member:
                    out = sid_dir / f"{base}_instant.nc"
                elif "stepType-accum" in member:
                    out = sid_dir / f"{base}_accum.nc"
                else:
                    out = sid_dir / Path(member).name
                with zf.open(member) as src, open(out, "wb") as dst:
                    dst.write(src.read())
                logger.info(f"  extracted -> {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    def download_storm(self, sid: str, df_storm: pd.DataFrame) -> Dict[str, Path]:
        """Download PL and SL NetCDFs for one storm."""
        lat_col = self.cfg.get("best_track.lat_col", "USA_LAT")
        lon_col = self.cfg.get("best_track.lon_col", "USA_LON")
        time_col = self.cfg.get("best_track.time_col", "ISO_TIME")

        bbox = self._storm_bbox(df_storm, lat_col, lon_col)
        t0, t1 = self._storm_time_range(df_storm, time_col)
        times = self._build_times(t0, t1)

        pl_file = self.output_dir / f"{sid}_pl.nc"
        sl_file = self.output_dir / f"{sid}_sl.nc"

        req_pl = self._build_request_pl(sid, bbox, times)
        req_sl = self._build_request_sl(sid, bbox, times)

        ok_pl = self._download_file(self.ds_pl, req_pl, pl_file)
        ok_sl = self._download_file(self.ds_sl, req_sl, sl_file)

        if not ok_pl or not ok_sl:
            self._record_retry(sid, {"bbox": bbox, "times": [t0, t1], "ok_pl": ok_pl, "ok_sl": ok_sl})

        return {"pl": pl_file if ok_pl else None, "sl": sl_file if ok_sl else None}

    def _record_retry(self, sid: str, info: Dict) -> None:
        retries = {}
        if self.retry_json.exists():
            retries = json.loads(self.retry_json.read_text(encoding="utf-8"))
        retries[sid] = info
        self.retry_json.write_text(json.dumps(retries, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.warning(f"Recorded retry for {sid}")

    # ------------------------------------------------------------------
    # Batch pipeline
    # ------------------------------------------------------------------

    def download_all(self, df_best_track: pd.DataFrame) -> Dict[str, Dict[str, Path]]:
        """Serial download for all storms in best-track DataFrame."""
        with phase_timer("ERA5Download"):
            sid_col = self.cfg.get("best_track.id_col", "SID")
            sids = df_best_track[sid_col].unique()
            logger.info(f"Storms to download: {len(sids)}")
            results = {}
            for sid in sids:
                storm_df = df_best_track[df_best_track[sid_col] == sid]
                results[sid] = self.download_storm(sid, storm_df)
            logger.info(f"Download batch complete. Retry log: {self.retry_json}")
        return results
