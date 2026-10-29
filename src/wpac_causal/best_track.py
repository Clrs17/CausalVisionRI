# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""IBTrACS best-track processing: read, filter, DV24, lifecycle labels, motion, land distance."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from wpac_causal.utils import Config, distance_to_land, haversine, normalize_lon, phase_timer


class BestTrackProcessor:
    """Process IBTrACS best track into analysis-ready DataFrame."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.intensity_col = config.get("best_track.intensity_col", "USA_WIND")
        self.lat_col = config.get("best_track.lat_col", "USA_LAT")
        self.lon_col = config.get("best_track.lon_col", "USA_LON")
        self.time_col = config.get("best_track.time_col", "ISO_TIME")
        self.id_col = config.get("best_track.id_col", "SID")
        self.name_col = config.get("best_track.name_col", "NAME")
        self.pres_col = config.get("best_track.pres_col", "USA_PRES")
        self.forecast_horizon = int(config.get("best_track.forecast_horizon", 4))
        self.min_intensity = config.get("best_track.min_intensity", 0)
        # Optional storm-level inclusion threshold (kt): when positive, keep
        # only systems whose lifecycle PEAK VMAX reaches it. Zero disables the
        # peak filter and retains the canonical all-system population.
        self.storm_min_peak = float(config.get("best_track.storm_min_peak_intensity", 0))
        self.years = config.get("data.years", [2000, 2025])
        self.basin = config.get("data.basin", "WP")
        self.stages = config.get("best_track.lifecycle_stages", {})

    # ------------------------------------------------------------------
    # Read & filter
    # ------------------------------------------------------------------

    def read_csv(self, path: str) -> pd.DataFrame:
        """Read IBTrACS CSV, coerce numeric columns, normalize time."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"IBTrACS file not found: {path}")
        logger.info(f"Reading IBTrACS from {path}")
        df = pd.read_csv(path, low_memory=False, na_values=["", " ", "NA", "NaN"])
        # Standard IBTrACS has header rows; skip if detected
        if "SID" not in df.columns:
            df = pd.read_csv(path, low_memory=False, skiprows=[1], na_values=["", " ", "NA", "NaN"])
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce", utc=True)
        for col in [self.intensity_col, self.lat_col, self.lon_col, self.pres_col]:
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def filter_basin_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep WP basin, year range, and non-null wind."""
        df = df.copy()
        df["YEAR"] = df[self.time_col].dt.year
        mask = (
            (df["BASIN"] == self.basin)
            & (df["YEAR"] >= self.years[0])
            & (df["YEAR"] <= self.years[1])
            & (df[self.intensity_col].notna())
        )
        n_before = len(df)
        df = df.loc[mask].copy()
        logger.info(f"Basin/year filter: {n_before} -> {len(df)} rows")
        return df

    def sort_by_storm_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by SID then ISO_TIME."""
        return df.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def compute_vmax_and_dv24(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VMAX, VMAX_F24, DV24; drop rows where 24h-ahead is unavailable.

        Fixes variable-timestep issue: instead of fixed shift(-4), we match
        the row whose ISO_TIME is closest to t + 24h within the same storm.
        """
        df = df.copy()
        df["VMAX"] = df[self.intensity_col].astype(float)
        df = df.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

        vmax_f24 = []
        f24_times = []
        for sid, group in df.groupby(self.id_col, sort=False):
            times = pd.DatetimeIndex(group[self.time_col])
            vmaxs = group["VMAX"].values
            for t in times:
                target = t + pd.Timedelta(hours=24)
                # searchsorted gives first index >= target
                idx_right = times.searchsorted(target, side="left")
                # Find the closest available time to target
                if idx_right < len(times):
                    # Check exact or near-exact match
                    diff_right = abs((times[idx_right] - target).total_seconds() / 3600)
                else:
                    diff_right = np.inf
                if idx_right > 0:
                    idx_left = idx_right - 1
                    diff_left = abs((times[idx_left] - target).total_seconds() / 3600)
                else:
                    diff_left = np.inf
                # Pick the closer one, requiring at most 6h tolerance
                if diff_right <= diff_left and diff_right <= 6.0:
                    vmax_f24.append(vmaxs[idx_right])
                    f24_times.append(times[idx_right])
                elif diff_left <= 6.0:
                    vmax_f24.append(vmaxs[idx_left])
                    f24_times.append(times[idx_left])
                else:
                    vmax_f24.append(np.nan)
                    f24_times.append(pd.NaT)

        df["VMAX_F24"] = vmax_f24
        df["F24_TIME"] = pd.to_datetime(f24_times, utc=True)
        df["DV24"] = df["VMAX_F24"] - df["VMAX"]
        ri_threshold = self.cfg.get("data.ri_threshold", 30)
        df["RI"] = (df["DV24"] >= ri_threshold).astype(int)
        df = df[df["VMAX_F24"].notna()].copy()
        logger.info(f"After DV24 truncation: {len(df)} rows | RI rate: {df['RI'].mean():.3f}")
        return df

    def filter_storm_peak_intensity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the optional lifecycle peak-VMAX threshold.

        A positive ``storm_min_peak`` drops weaker systems while retaining each
        selected storm's full lifecycle. A zero threshold performs no filtering
        and is the canonical all-system setting.
        """
        if self.storm_min_peak <= 0:
            return df
        peak = df.groupby(self.id_col)["VMAX"].transform("max")
        n_before = df[self.id_col].nunique()
        df = df.loc[peak >= self.storm_min_peak].copy()
        logger.info(
            f"Storm peak-intensity filter (>={self.storm_min_peak:.0f} kt): "
            f"{n_before} -> {df[self.id_col].nunique()} storms, {len(df)} rows"
        )
        return df

    def write_sample_manifest(self, df: pd.DataFrame, path: Optional[str] = None) -> pd.DataFrame:
        """Write the explicit sample manifest (R2-4): one row per storm with
        year, peak VMAX, inclusion flag and exclusion reason (if any).

        Called on the full-lifecycle frame before F24 truncation and peak
        filtering, so every canonical storm in the basin/year window is listed;
        label-eligible splits are derived separately by the holdout driver.
        """
        if path is None:
            path = self.cfg.get("best_track.sample_manifest_csv",
                                "data/phase1/processed/sample_manifest.csv")
        peak = df.groupby(self.id_col)["VMAX"].max().rename("PEAK_VMAX")
        year = df.groupby(self.id_col)[self.time_col].min().dt.year.rename("YEAR")
        name = df.groupby(self.id_col)[self.name_col].first().rename("NAME")
        man = pd.concat([name, year, peak], axis=1).reset_index()
        thr = self.storm_min_peak
        man["INCLUDED"] = man["PEAK_VMAX"] >= thr if thr > 0 else True
        man["EXCLUSION_REASON"] = ""
        if thr > 0:
            man.loc[~man["INCLUDED"], "EXCLUSION_REASON"] = (
                "peak < " + str(int(thr)) + " kt (" + man["PEAK_VMAX"].round(0).astype(int).astype(str) + " kt)"
            )
        # Provenance for systems whose ERA5/OHC inputs were backfilled in a
        # rerun; this flag does not change the configured population.
        backfilled = set(self.cfg.get("best_track.backfilled_sids", []) or [])
        man["added_in_rerun"] = man[self.id_col].isin(backfilled)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        man.to_csv(path, index=False)
        logger.info(
            f"Sample manifest written to {path}: {man['INCLUDED'].sum()}/{len(man)} "
            f"storms included (peak >= {thr:.0f} kt)"
        )
        return man

    def compute_lifecycle_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign LC_STAGE based on VMAX thresholds."""
        conditions = []
        choices = []
        for stage, (lo, hi) in self.stages.items():
            conditions.append((df["VMAX"] >= lo) & (df["VMAX"] < hi))
            choices.append(stage)
        df["LC_STAGE"] = np.select(conditions, choices, default="unknown")
        return df

    def compute_per(self, df: pd.DataFrame, lag_steps: int = 2) -> pd.DataFrame:
        """Past intensity trend: PER = VMAX - VMAX.shift(lag_steps).

        Computed on the NATIVE IBTrACS 3-hourly spacing with lag_steps=2,
        i.e. PER is the **6-hour** intensity trend (kt/6h). It is evaluated
        BEFORE ``resample_to_6h`` (see process()), so resampling only drops
        rows and does not change the values. An earlier docstring claimed a
        12-hour effective lag — that was wrong (review C-M2); the 6h trend
        also matches the RI pace convention (30 kt/24h = 7.5 kt/6h).
        """
        df["PER"] = df.groupby(self.id_col)["VMAX"].diff(lag_steps)
        return df

    def resample_to_6h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only observations at synoptic hours (00, 06, 12, 18 UTC).
        
        IBTrACS is natively 3-hourly (≈98.8 % of observations), but ERA5 is
        requested at 6-hourly resolution.  Without resampling, consecutive
        IBTrACS observations (e.g. 00Z and 03Z) map to the same ERA5 timestep,
        introducing duplicate environmental data that breaks the uniform-
        spacing assumption of PCMCI+.
        
        This method retains the 6-hourly subset so that each observation has
        a unique ERA5 counterpart and ``tau_max=4`` correctly corresponds to
        a 24-hour lookback window.
        """
        keep_hours = {0, 6, 12, 18}
        mask = df[self.time_col].dt.hour.isin(keep_hours)
        n_before = len(df)
        df = df.loc[mask].copy()
        logger.info(
            f"6-hourly resample: {n_before} -> {len(df)} rows "
            f"({len(df) / max(n_before, 1) * 100:.1f}%)"
        )
        return df

    def compute_motion(self, df: pd.DataFrame) -> pd.DataFrame:
        """TC motion: SPD (kt) and DIR (deg, 0=North) from haversine + bearing."""
        df = df.copy()
        # Shift within storm to get previous position
        df["LAT_prev"] = df.groupby(self.id_col)[self.lat_col].shift(1)
        df["LON_prev"] = df.groupby(self.id_col)[self.lon_col].shift(1)
        # Time delta in hours
        df["time_prev"] = df.groupby(self.id_col)[self.time_col].shift(1)
        dt_hours = (df[self.time_col] - df["time_prev"]).dt.total_seconds() / 3600.0
        # Distance km -> knots (1 km / 1.852 = 1 nmi)
        dist_km = haversine(
            df[self.lat_col].values,
            normalize_lon(df[self.lon_col].values),
            df["LAT_prev"].fillna(df[self.lat_col]).values,
            normalize_lon(df["LON_prev"].fillna(df[self.lon_col]).values),
        )
        dist_nmi = dist_km / 1.852
        # SPD in kt = nmi per hour
        df["SPD"] = np.where(dt_hours > 0, dist_nmi / dt_hours, np.nan)
        # DIR: bearing from previous to current
        df["DIR"] = np.degrees(
            np.arctan2(
                np.sin(np.radians(df[self.lon_col] - df["LON_prev"]))
                * np.cos(np.radians(df[self.lat_col])),
                np.cos(np.radians(df["LAT_prev"])) * np.sin(np.radians(df[self.lat_col]))
                - np.sin(np.radians(df["LAT_prev"]))
                * np.cos(np.radians(df[self.lat_col]))
                * np.cos(np.radians(df[self.lon_col] - df["LON_prev"])),
            )
        )
        df["DIR"] = (df["DIR"] + 360.0) % 360.0
        # Clean up
        df.drop(columns=["LAT_prev", "LON_prev", "time_prev"], inplace=True, errors="ignore")
        return df

    def compute_land_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute LAND_DIST (km) via cKDTree on coastline."""
        logger.info("Computing distance to land (this may take a moment)...")
        df["LAND_DIST"] = distance_to_land(
            df[self.lat_col].values, normalize_lon(df[self.lon_col].values)
        )
        return df

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def process(self, path: Optional[str] = None) -> pd.DataFrame:
        with phase_timer("BestTrack"):
            if path is None:
                path = self.cfg.get("data.ibtracs_path")
            df = self.read_csv(path)
            df = self.filter_basin_years(df)
            df = self.sort_by_storm_time(df)
            # Sample definition uses the FULL (untruncated) lifecycle peak:
            # compute_vmax_and_dv24 drops each storm's final ~24h rows, so a
            # peak computed afterwards could underestimate storms peaking late.
            df["VMAX"] = df[self.intensity_col].astype(float)
            # Explicit sample manifest BEFORE the peak filter (R2-4)
            self.write_sample_manifest(df)
            df = self.filter_storm_peak_intensity(df)
            df = self.compute_vmax_and_dv24(df)
            df = self.compute_lifecycle_stage(df)
            df = self.compute_per(df)
            df = self.compute_motion(df)
            df = self.compute_land_distance(df)
            df = self.resample_to_6h(df)
            # Final column selection / ordering
            core_cols = [
                self.id_col,
                self.name_col,
                self.time_col,
                self.lat_col,
                self.lon_col,
                "VMAX",
                "VMAX_F24",
                "F24_TIME",
                "DV24",
                "RI",
                "LC_STAGE",
                "PER",
                "SPD",
                "DIR",
                "LAND_DIST",
            ]
            if self.pres_col and self.pres_col in df.columns:
                core_cols.append(self.pres_col)
            # Keep any extra columns that exist
            keep = [c for c in core_cols if c in df.columns]
            df = df[keep].copy()
            logger.info(f"Best-track final shape: {df.shape}")
            logger.info(f"Storms: {df[self.id_col].nunique()}")
        return df


def download_ibtracs(url: str, output_path: str) -> None:
    """Download IBTrACS CSV from NOAA if not present."""
    import requests
    p = Path(output_path)
    if p.exists() and p.stat().st_size > 0:
        logger.info(f"IBTrACS already exists: {output_path}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading IBTrACS from {url}")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Saved IBTrACS to {output_path}")
