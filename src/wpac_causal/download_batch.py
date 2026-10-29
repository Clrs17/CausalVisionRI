# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""
Two-phase ERA5 batch downloader for WPAC causal inference.

Phase 1: Submit-only — dump all requests into CDS queue (fast).
Phase 2: Poll & download — check status and pull completed files.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from utils import Config, normalize_lon, phase_timer

try:
    from cads_api_client import ApiClient
except Exception:  # pragma: no cover
    ApiClient = None  # type: ignore


class BatchERA5Downloader:
    """Two-phase downloader: submit → poll & download."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.output_dir = Path(config.get("data.era5_download_dir", "data/phase1/raw/era5"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Submission log (request_uid per storm/level)
        self.submit_json = Path(config.get("data.submit_json", "data/phase1/processed/submit_log.json"))
        self.submit_json.parent.mkdir(parents=True, exist_ok=True)

        # CDS config
        self.ds_pl = config.get("era5.cds_dataset_pl", "reanalysis-era5-pressure-levels")
        self.ds_sl = config.get("era5.cds_dataset_sl", "reanalysis-era5-single-levels")
        self.pl_vars = config.get("era5.pl_variables", [])
        self.sl_vars = config.get("era5.sl_variables", [])
        self.pressure_levels = config.get("era5.pressure_levels", [1000, 925, 850, 700, 500, 300, 200])
        self.buffer_deg = config.get("era5.spatial_buffer_deg", 15)
        self.buffer_days = config.get("era5.temporal_buffer_days", 1)
        self.retry_max = config.get("era5.download_retry", 3)

        self._api_client: Any = None
        self._submit_log: Dict[str, Any] = {}
        self._load_submit_log()

    # ------------------------------------------------------------------
    # CDS client (cads-api-client)
    # ------------------------------------------------------------------

    def _get_api_client(self) -> Any:
        if self._api_client is None:
            if ApiClient is None:
                raise RuntimeError("cads-api-client is not installed. Run: pip install cads-api-client")
            creds = self._read_cds_credentials()
            url = creds.get("url", "https://cds.climate.copernicus.eu/api") if creds else "https://cds.climate.copernicus.eu/api"
            key = creds.get("key") if creds else None
            self._api_client = ApiClient(url=url, key=key)
            logger.info(f"CDS API client: {url}")
        return self._api_client

    def _read_cds_credentials(self) -> Optional[Dict[str, str]]:
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
                    return creds
        return None

    # ------------------------------------------------------------------
    # Submit log persistence
    # ------------------------------------------------------------------

    def _load_submit_log(self) -> None:
        if self.submit_json.exists():
            try:
                self._submit_log = json.loads(self.submit_json.read_text(encoding="utf-8"))
            except Exception:
                self._submit_log = {}

    def _save_submit_log(self) -> None:
        self.submit_json.write_text(json.dumps(self._submit_log, indent=2, ensure_ascii=False), encoding="utf-8")

    def _record_submit(self, sid: str, level: str, request_uid: str, req: Dict) -> None:
        self._submit_log.setdefault(sid, {})[level] = {
            "request_uid": request_uid,
            "status": "accepted",
            "ts_submit": time.time(),
            "request": req,
        }
        self._save_submit_log()

    def _update_status(self, sid: str, level: str, status: str) -> None:
        if sid in self._submit_log and level in self._submit_log[sid]:
            self._submit_log[sid][level]["status"] = status
            self._submit_log[sid][level]["ts_update"] = time.time()
            self._save_submit_log()

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _storm_bbox(self, df_storm: pd.DataFrame, lat_col: str, lon_col: str) -> Tuple[float, float, float, float]:
        lats = pd.to_numeric(df_storm[lat_col], errors="coerce").dropna()
        lons = pd.to_numeric(df_storm[lon_col], errors="coerce").dropna()
        lons = normalize_lon(lons)
        n = min(90.0, lats.max() + self.buffer_deg)
        s = max(-90.0, lats.min() - self.buffer_deg)
        e = min(360.0, lons.max() + self.buffer_deg)
        w = max(0.0, lons.min() - self.buffer_deg)
        return float(n), float(w), float(s), float(e)

    def _storm_time_range(self, df_storm: pd.DataFrame, time_col: str) -> Tuple[str, str]:
        t0 = pd.to_datetime(df_storm[time_col].min()) - pd.Timedelta(days=self.buffer_days)
        t1 = pd.to_datetime(df_storm[time_col].max()) + pd.Timedelta(days=self.buffer_days)
        return t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")

    def _build_request_pl(self, bbox: Tuple[float, ...], t0: str, t1: str) -> Dict:
        n, w, s, e = bbox
        return {
            "product_type": ["reanalysis"],
            "variable": self.pl_vars,
            "pressure_level": [str(p) for p in self.pressure_levels],
            "year": sorted({str(pd.Timestamp(d).year) for d in pd.date_range(t0, t1, freq="D")}),
            "month": sorted({f"{pd.Timestamp(d).month:02d}" for d in pd.date_range(t0, t1, freq="D")}),
            "day": sorted({f"{pd.Timestamp(d).day:02d}" for d in pd.date_range(t0, t1, freq="D")}),
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [n, w, s, e],
            "format": "netcdf",
        }

    def _build_request_sl(self, bbox: Tuple[float, ...], t0: str, t1: str) -> Dict:
        n, w, s, e = bbox
        return {
            "product_type": ["reanalysis"],
            "variable": self.sl_vars,
            "year": sorted({str(pd.Timestamp(d).year) for d in pd.date_range(t0, t1, freq="D")}),
            "month": sorted({f"{pd.Timestamp(d).month:02d}" for d in pd.date_range(t0, t1, freq="D")}),
            "day": sorted({f"{pd.Timestamp(d).day:02d}" for d in pd.date_range(t0, t1, freq="D")}),
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [n, w, s, e],
            "format": "netcdf",
        }

    # ------------------------------------------------------------------
    # Phase 1: Submit-only
    # ------------------------------------------------------------------

    def submit_storm(self, sid: str, df_storm: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Submit PL + SL requests for one storm; return request_uids."""
        lat_col = self.cfg.get("best_track.lat_col", "USA_LAT")
        lon_col = self.cfg.get("best_track.lon_col", "USA_LON")
        time_col = self.cfg.get("best_track.time_col", "ISO_TIME")

        bbox = self._storm_bbox(df_storm, lat_col, lon_col)
        t0, t1 = self._storm_time_range(df_storm, time_col)

        req_pl = self._build_request_pl(bbox, t0, t1)
        req_sl = self._build_request_sl(bbox, t0, t1)

        client = self._get_api_client()
        uids: Dict[str, Optional[str]] = {"pl": None, "sl": None}

        for level, ds_id, req in [("pl", self.ds_pl, req_pl), ("sl", self.ds_sl, req_sl)]:
            # Skip if already submitted and not failed
            existing = self._submit_log.get(sid, {}).get(level, {})
            if existing.get("status") in ("accepted", "running", "completed"):
                logger.info(f"[{sid}] {level} already submitted ({existing['request_uid']})")
                uids[level] = existing["request_uid"]
                continue

            for attempt in range(1, self.retry_max + 1):
                try:
                    remote = client.submit(collection_id=ds_id, **req)
                    uid = remote.request_uid
                    self._record_submit(sid, level, uid, req)
                    logger.info(f"[{sid}] {level} submitted: {uid}")
                    uids[level] = uid
                    break
                except Exception as exc:
                    logger.warning(f"[{sid}] {level} submit failed (attempt {attempt}): {exc}")
                    if attempt < self.retry_max:
                        time.sleep(5 * attempt)
                    else:
                        logger.error(f"[{sid}] {level} submit exhausted retries")
                        uids[level] = None

        return uids

    def submit_all(self, df_best_track: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
        """Phase 1: Submit all storm requests into CDS queue."""
        sid_col = self.cfg.get("best_track.id_col", "SID")
        sids = df_best_track[sid_col].unique()
        logger.info(f"Phase 1: Submitting {len(sids)} storms to CDS queue")

        results: Dict[str, Dict[str, Optional[str]]] = {}
        ok = 0
        fail = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for sid in sids:
                storm_df = df_best_track[df_best_track[sid_col] == sid]
                futures[executor.submit(self.submit_storm, sid, storm_df)] = sid

            for future in as_completed(futures):
                sid = futures[future]
                try:
                    res = future.result()
                    results[sid] = res
                    if res["pl"] and res["sl"]:
                        ok += 1
                    else:
                        fail += 1
                except Exception as exc:
                    logger.error(f"[{sid}] Submit exception: {exc}")
                    results[sid] = {"pl": None, "sl": None}
                    fail += 1
                logger.info(f"Submit progress: {ok+fail}/{len(sids)} (ok={ok}, fail={fail})")

        logger.info(f"Phase 1 done. Submitted: {ok}, Failed: {fail}")
        logger.info(f"Submit log: {self.submit_json}")
        return results

    # ------------------------------------------------------------------
    # Phase 2: Poll & download
    # ------------------------------------------------------------------

    def download_completed(self) -> Dict[str, Dict[str, Path]]:
        """Phase 2: Check status of all submitted requests and download completed ones."""
        client = self._get_api_client()
        results: Dict[str, Dict[str, Path]] = {}
        completed = 0
        failed = 0
        skipped = 0

        for sid, levels in list(self._submit_log.items()):
            storm_dir = self.output_dir / sid
            storm_dir.mkdir(parents=True, exist_ok=True)
            res = {"pl": None, "sl": None}

            for level in ("pl", "sl"):
                info = levels.get(level)
                if not info:
                    continue
                uid = info["request_uid"]
                target = storm_dir / f"{sid}_{level}.nc"

                # Skip if already on disk
                if target.exists() and target.stat().st_size > 1024:
                    logger.info(f"[{sid}] {level} already on disk, skip")
                    res[level] = target
                    skipped += 1
                    continue

                # Check status
                try:
                    from cads_api_client.processing import Remote
                    remote = Remote(
                        url=f"https://cds.climate.copernicus.eu/api/retrieve/v1/jobs/{uid}",
                        session=client._retrieve_api.session,
                        request_uid=uid,
                    )
                    remote.update()
                    status = remote.status
                    self._update_status(sid, level, status)
                    logger.info(f"[{sid}] {level} status: {status}")

                    if status == "completed":
                        remote.download(str(target))
                        if target.exists() and target.stat().st_size > 1024:
                            logger.success(f"[{sid}] {level} downloaded: {target}")
                            res[level] = target
                            completed += 1
                        else:
                            logger.warning(f"[{sid}] {level} download empty")
                            failed += 1
                    elif status in ("failed", "aborted"):
                        logger.error(f"[{sid}] {level} {status}")
                        failed += 1
                    else:
                        # still in queue
                        pass
                except Exception as exc:
                    logger.warning(f"[{sid}] {level} poll error: {exc}")

            results[sid] = res

        logger.info(f"Phase 2 done. Downloaded: {completed}, Failed: {failed}, Skipped: {skipped}")
        return results

    # ------------------------------------------------------------------
    # Legacy: blocking download (kept for compatibility)
    # ------------------------------------------------------------------

    def download_storm(self, sid: str, df_storm: pd.DataFrame) -> Dict[str, Path]:
        """Blocking download (uses cdsapi retrieve; may timeout on long queues)."""
        # ... keep existing implementation for backward compat ...
        lat_col = self.cfg.get("best_track.lat_col", "USA_LAT")
        lon_col = self.cfg.get("best_track.lon_col", "USA_LON")
        time_col = self.cfg.get("best_track.time_col", "ISO_TIME")

        bbox = self._storm_bbox(df_storm, lat_col, lon_col)
        t0, t1 = self._storm_time_range(df_storm, time_col)

        storm_dir = self.output_dir / sid
        storm_dir.mkdir(parents=True, exist_ok=True)
        pl_file = storm_dir / f"{sid}_pl.nc"
        sl_file = storm_dir / f"{sid}_sl.nc"

        req_pl = self._build_request_pl(bbox, t0, t1)
        req_sl = self._build_request_sl(bbox, t0, t1)

        try:
            import cdsapi
            client = cdsapi.Client(timeout=self.cfg.get("era5.download_timeout", 600), quiet=False)
            if not (pl_file.exists() and pl_file.stat().st_size > 1024):
                client.retrieve(self.ds_pl, req_pl, str(pl_file))
            if not (sl_file.exists() and sl_file.stat().st_size > 1024):
                client.retrieve(self.ds_sl, req_sl, str(sl_file))
            return {"pl": pl_file, "sl": sl_file}
        except Exception as exc:
            logger.error(f"[{sid}] Blocking download failed: {exc}")
            return {"pl": None, "sl": None}

    def download_all(self, df_best_track: pd.DataFrame) -> Dict[str, Dict[str, Path]]:
        """Blocking batch download (legacy)."""
        sid_col = self.cfg.get("best_track.id_col", "SID")
        sids = df_best_track[sid_col].unique()
        results = {}
        for sid in sids:
            storm_df = df_best_track[df_best_track[sid_col] == sid]
            results[sid] = self.download_storm(sid, storm_df)
        return results
