# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""
Preprocessing: ERA5 annulus extraction, derived variables,
Yeo-Johnson + Winsorize, Z-score standardization.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.preprocessing import PowerTransformer, StandardScaler

from wpac_causal.utils import Config, annulus_spatial_mean, normalize_lon, phase_timer, save_pickle

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import metpy.calc as mpcalc
    from metpy.units import units
except Exception:  # pragma: no cover
    mpcalc = None  # type: ignore
    units = None  # type: ignore


# CMEMS GLORYS12V1 "daily" fields are DAILY MEANS over a calendar day
# (00 UTC to next 00 UTC), not instantaneous snapshots stamped at 00 UTC.  A
# field labelled D is therefore only complete once D+1 00 UTC has arrived.
OHC_DAILY_WINDOW = pd.Timedelta(days=1)
FIT_LABEL_EMBARGO = pd.Timedelta(hours=30)
IMPUTATION_PREFIX = "__IMPUTED_"
IMPUTATION_KEY_COLUMNS = ["SID", "ISO_TIME"]


def strict_fit_row_mask(
    df: pd.DataFrame,
    fit_years,
    sid_col: str = "SID",
    time_col: str = "ISO_TIME",
    label_time_col: str = "F24_TIME",
) -> pd.Series:
    """Strict fit-only row mask shared by transforms, audits and causal priors.

    Both the canonical SID year and the row year must be inside ``fit_years``.
    The realised label time and the full t+30h label window must end strictly
    before the following split; NaT and boundary equality are rejected.
    """
    required = {sid_col, time_col, label_time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"strict fit mask missing required columns: {sorted(missing)}")
    if len(fit_years) != 2:
        raise ValueError("fit_years must contain [first_year, last_year]")
    first_year, last_year = (int(fit_years[0]), int(fit_years[1]))
    if first_year > last_year:
        raise ValueError("fit_years must be increasing")
    sid_year = pd.to_numeric(
        df[sid_col].astype(str).str[:4], errors="raise").astype(int)
    iso = pd.to_datetime(df[time_col], utc=True)
    f24 = pd.to_datetime(df[label_time_col], utc=True)
    if "storm_year" in df.columns:
        canonical_year = pd.to_numeric(df["storm_year"], errors="raise").astype(int)
        if not canonical_year.equals(sid_year):
            raise ValueError("storm_year disagrees with the canonical SID first year")
    boundary = pd.Timestamp(f"{last_year + 1}-01-01", tz="UTC")
    mask = (
        sid_year.between(first_year, last_year)
        & iso.dt.year.between(first_year, last_year)
        & iso.notna()
        & f24.notna()
        & (f24 < boundary)
        & ((iso + FIT_LABEL_EMBARGO) < boundary)
    )
    return pd.Series(mask.to_numpy(dtype=bool), index=df.index, name="fit_only")


def backward_time_reindex(da, times, window=None):
    """Select the field whose averaging window has already closed at each target.

    The GLORYS daily product is a daily mean over [D 00 UTC, D+1 00 UTC), so the
    field stamped D may only be used from D+1 00 UTC onwards:
    ``source_time + window <= target``.  Consequences:

    * 00/06/12/18 UTC on day D all use the field of day D-1 (never day D, whose
      window is still open);
    * day D+1 00 UTC is the first moment day D's complete window is available;
    * a target earlier than the first field's window end is NaN (never the
      nearest future field, and never a partially covered day).

    The old linear interpolation read the *next* day's field for 06/12/18 UTC.
    """
    import numpy as np
    import pandas as pd

    win = OHC_DAILY_WINDOW if window is None else pd.Timedelta(window)
    tv = pd.to_datetime(da["time"].values)
    order = np.argsort(tv.values)
    tv = tv[order]
    da = da.isel(time=order)
    if len(np.unique(tv.values)) != len(tv):
        # deterministic de-duplication: keep the first occurrence in time order
        _, keep = np.unique(tv.values, return_index=True)
        keep = np.sort(keep)
        tv = tv[keep]
        da = da.isel(time=keep)
    target = pd.to_datetime(times)
    window_end = tv.values + np.timedelta64(int(win.total_seconds()), "s")
    idx = np.searchsorted(window_end, target.values, side="right") - 1
    out = da.isel(time=np.clip(idx, 0, len(tv) - 1)).copy()
    data = np.asarray(out.values, dtype=np.float64).copy()
    if (idx < 0).any():
        data[idx < 0] = np.nan
    out = out.copy(data=data)
    out = out.assign_coords(time=target)
    return out


class ERA5Extractor:
    """Extract annulus means and derived variables from ERA5 NetCDF + best track."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.inner = tuple(config.get("extraction.inner_core_radius", [0, 2]))
        self.outer = tuple(config.get("extraction.outer_area_radius", [2, 8]))
        self.pl_vars = config.get("era5.pl_variables", [])
        self.sl_vars = config.get("era5.sl_variables", [])
        self.derived = config.get("extraction.derived_variables", [])
        self.optional = config.get("extraction.optional_variables", {})
        self.ohc_enabled = bool(self.optional.get("OHC", False))
        self.ohc_dir = config.get("data.ohc_dir", "data/phase1/raw/ohc")
        self._meshgrid_cache: Optional[Tuple[Any, Any, Any]] = None

    # ------------------------------------------------------------------
    # Load & align
    # ------------------------------------------------------------------

    def _time_coord(self, ds: Any) -> Optional[str]:
        """Return the time coordinate name (valid_time, time, etc.)."""
        for cand in ("valid_time", "time", "forecast_reference_time"):
            if cand in ds.coords:
                return cand
        return None

    def load_storm_netcdf(self, pl_path: Path, sl_path: Path) -> Optional[Any]:
        """Load and merge PL + SL datasets; return xarray Dataset or None.

        Supports split SL files (instant + accum) when sl_path stem ends in
        '_instant' or '_accum', e.g. <SID>_sl_instant.nc + <SID>_sl_accum.nc.
        """
        if xr is None:
            raise RuntimeError("xarray not installed")
        ds = None
        if pl_path.exists() and pl_path.stat().st_size > 0:
            if str(pl_path).lower().endswith(".grib"):
                ds = xr.open_dataset(pl_path, engine="cfgrib")
                # Harmonise grib coords with netcdf conventions
                rename_map = {}
                if "isobaricInhPa" in ds.coords:
                    rename_map["isobaricInhPa"] = "pressure_level"
                if "time" in ds.coords:
                    # cfgrib creates valid_time as a non-index coord; drop it first
                    if "valid_time" in ds.coords:
                        ds = ds.drop_vars("valid_time")
                    rename_map["time"] = "valid_time"
                if rename_map:
                    ds = ds.rename(rename_map)
                # Ensure valid_time is an index so .sel() works and merge aligns
                if "valid_time" in ds.coords and "valid_time" not in ds.indexes:
                    ds = ds.set_index(valid_time="valid_time")
            else:
                ds = xr.open_dataset(pl_path)

        # Collect SL files: single <SID>_sl.nc or split instant/accum pair
        sl_files = []
        if sl_path.exists() and sl_path.stat().st_size > 0:
            sl_files.append(sl_path)
        else:
            for suffix in ("_instant", "_accum"):
                candidate = sl_path.with_name(sl_path.stem.replace("_sl", f"_sl{suffix}") + sl_path.suffix)
                if candidate.exists() and candidate.stat().st_size > 0:
                    sl_files.append(candidate)

        for sl_file in sl_files:
            try:
                ds_sl = xr.open_dataset(sl_file)
            except Exception as exc:
                logger.warning(f"Could not open {sl_file}: {exc}")
                continue
            if ds is None:
                ds = ds_sl
            else:
                # Use inner join so spatial domains agree (e.g. PL global vs SL subset)
                ds = ds.merge(ds_sl, join="inner")
        return ds

    def nearest_time_select(self, ds: Any, target_time: pd.Timestamp) -> Any:
        """Select nearest time slice within 3h tolerance."""
        tname = self._time_coord(ds)
        if tname is None:
            return ds
        # Harmonise timezone (ERA5 CDS often uses UTC-aware datetimes)
        target = pd.Timestamp(target_time)
        if target.tz is not None:
            target = target.tz_convert("UTC").tz_localize(None)
        try:
            # xarray nearest method; pass naive datetime to avoid dtype mismatch
            result = ds.sel({tname: target}, method="nearest", tolerance=pd.Timedelta("3h"))
            # Ensure scalar time (drop length-1 time dim if present)
            if tname in result.dims and result.dims[tname] == 1:
                result = result.isel({tname: 0})
            return result
        except Exception:
            # fallback: manual find
            times = pd.to_datetime(ds[tname].values)
            if times.tz is not None:
                times = times.tz_convert("UTC").tz_localize(None)
            delta = np.abs(times - target)
            if delta.min() > pd.Timedelta("3h"):
                return None
            return ds.isel({tname: int(delta.argmin())})

    # ------------------------------------------------------------------
    # Annulus extraction helpers
    # ------------------------------------------------------------------

    def _extract_annulus_mean(
        self,
        da: Any,
        center_lon: float,
        center_lat: float,
        radius: Tuple[float, float],
    ) -> float:
        """Area-weighted mean over annulus with longitude cosine correction.

        The denominator must sum weights over VALID (finite) cells only:
        dividing by the full annulus weight biases the mean toward zero
        wherever the field has NaN cells (land/ice for SST etc.) — found
        2026-07-21, this biased SST low by the NaN-area fraction (up to
        unphysical -73 degC for land-heavy annuli).
        """
        if da is None:
            return np.nan
        lon = np.asarray(da.longitude.values if "longitude" in da.dims else da.lon.values)
        lat = np.asarray(da.latitude.values if "latitude" in da.dims else da.lat.values)
        lon2d, lat2d = np.meshgrid(lon, lat)
        # Correct longitude distance for spherical geometry:
        # 1 deg lon = cos(lat) * 111 km, 1 deg lat = 111 km
        cos_center = np.cos(np.radians(center_lat))
        dlon_eff = (lon2d - center_lon) * cos_center
        dist = np.sqrt(dlon_eff ** 2 + (lat2d - center_lat) ** 2)
        mask = (dist >= radius[0]) & (dist <= radius[1])
        if not np.any(mask):
            return np.nan
        data = np.asarray(da.values)
        if data.ndim > 2:
            data = data.squeeze()
        weights = np.cos(np.radians(lat2d)) * mask
        weights = weights * np.isfinite(data)  # only valid cells count
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            denom = np.nansum(weights)
            if denom == 0:
                return np.nan
            return float(np.nansum(data * weights) / denom)

    def _get_grid(self, da: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get or create cached meshgrid for a data array."""
        lon = np.asarray(da.longitude.values if "longitude" in da.dims else da.lon.values)
        lat = np.asarray(da.latitude.values if "latitude" in da.dims else da.lat.values)
        # Cache key: shape + first/last values
        cache_key = (lon.shape, lat.shape, lon[0], lon[-1], lat[0], lat[-1])
        if self._meshgrid_cache is not None and self._meshgrid_cache[0] == cache_key:
            return self._meshgrid_cache[1:]
        lon2d, lat2d = np.meshgrid(lon, lat)
        self._meshgrid_cache = (cache_key, lon, lat, lon2d, lat2d)
        return lon, lat, lon2d, lat2d

    # ------------------------------------------------------------------
    # Derived variables
    # ------------------------------------------------------------------

    def _get_var(self, ds: Any, name: str, level: Optional[int] = None) -> Any:
        """Safe variable extraction from xarray Dataset."""
        if ds is None:
            return None
        aliases = {
            "u": ["u", "u_component_of_wind", "u_wind"],
            "v": ["v", "v_component_of_wind", "v_wind"],
            "t": ["t", "temperature", "air_temperature"],
            "r": ["r", "relative_humidity", "rh"],
            "z": ["z", "geopotential", "geo"],
            "q": ["q", "specific_humidity"],
            "w": ["w", "vertical_velocity"],
            "d": ["d", "divergence"],
            "pv": ["pv", "potential_vorticity"],
            "vo": ["vo", "vorticity", "relative_vorticity"],
            "sst": ["sst", "sea_surface_temperature"],
            "msl": ["msl", "mean_sea_level_pressure"],
            "slhf": ["slhf", "surface_latent_heat_flux"],
            "olr": ["olr", "top_net_thermal_radiation", "tntr"],
        }
        candidates = aliases.get(name, [name])
        level_dims = ("level", "pressure_level", "isobaricInhPa")
        for c in candidates:
            if c in ds:
                da = ds[c]
                if level is not None:
                    for lev_dim in level_dims:
                        if lev_dim in da.dims:
                            try:
                                da = da.sel({lev_dim: level})
                            except Exception:
                                try:
                                    da = da.sel({lev_dim: float(level)})
                                except Exception:
                                    return None
                            break
                return da
        return None

    def compute_shear(self, ds: Any, center_lon: float, center_lat: float) -> Dict[str, float]:
        """SHRD, USHR, VSHR, SHEAR_DIR over Outer Area."""
        u850 = self._get_var(ds, "u", 850)
        v850 = self._get_var(ds, "v", 850)
        u200 = self._get_var(ds, "u", 200)
        v200 = self._get_var(ds, "v", 200)
        if any(v is None for v in [u850, v850, u200, v200]):
            return {"SHRD": np.nan, "USHR": np.nan, "VSHR": np.nan, "SHEAR_DIR": np.nan}

        du = self._extract_annulus_mean(u200, center_lon, center_lat, self.outer) - \
             self._extract_annulus_mean(u850, center_lon, center_lat, self.outer)
        dv = self._extract_annulus_mean(v200, center_lon, center_lat, self.outer) - \
             self._extract_annulus_mean(v850, center_lon, center_lat, self.outer)
        shrd = float(np.hypot(du, dv))
        shear_dir = float((np.degrees(np.arctan2(du, dv)) + 360.0) % 360.0)
        return {"SHRD": shrd, "USHR": float(du), "VSHR": float(dv), "SHEAR_DIR": shear_dir}

    def compute_d200(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """200 hPa divergence (10^-6 s^-1), OA.
        
        Prefers the directly downloaded divergence field 'd' (s^-1) from ERA5
        pressure levels, as documented in CAUSAL_FRAMEWORK.md §2.2.  
        Falls back to numerical differentiation of u,v when 'd' is unavailable.
        """
        # Try direct ERA5 divergence field first
        d200 = self._get_var(ds, "d", 200)
        if d200 is not None:
            raw_val = self._extract_annulus_mean(d200, center_lon, center_lat, self.outer)
            if np.isfinite(raw_val):
                return float(raw_val * 1e6)  # s^-1 → 10^-6 s^-1

        # Fallback: numerical differentiation of u, v
        u200 = self._get_var(ds, "u", 200)
        v200 = self._get_var(ds, "v", 200)
        if u200 is None or v200 is None:
            return np.nan
        lon, lat, lon2d, lat2d = self._get_grid(u200)
        mask = (
            (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) >= self.outer[0])
            & (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) <= self.outer[1])
        )
        if not np.any(mask):
            return np.nan
        u = np.asarray(u200.values).squeeze()
        v = np.asarray(v200.values).squeeze()
        R = 6371e3
        dlat = np.radians(np.abs(lat[1] - lat[0])) if len(lat) > 1 else 1e-3
        dlon = np.radians(np.abs(lon[1] - lon[0])) if len(lon) > 1 else 1e-3
        coslat = np.cos(np.radians(lat2d))
        du_dl = np.zeros_like(u)
        dv_dl = np.zeros_like(v)
        du_dl[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * R * coslat[:, 1:-1] * dlon)
        dv_dl[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * R * dlat)
        div = (du_dl + dv_dl) * 1e6
        weights = np.cos(np.radians(lat2d)) * mask
        return float(np.nansum(div * weights) / np.nansum(weights))

    def compute_rh_layer(self, ds: Any, center_lon: float, center_lat: float,
                         levels: List[int], annulus: Tuple[float, float]) -> float:
        """Mean RH over a layer and annulus."""
        vals = []
        for lev in levels:
            r = self._get_var(ds, "r", lev)
            if r is not None:
                vals.append(self._extract_annulus_mean(r, center_lon, center_lat, annulus))
        if not vals:
            return np.nan
        return float(np.nanmean(vals))

    def compute_vorticity(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """850 hPa relative vorticity (10^-5 s^-1), IC."""
        u850 = self._get_var(ds, "u", 850)
        v850 = self._get_var(ds, "v", 850)
        if u850 is None or v850 is None:
            return np.nan
        lon, lat, lon2d, lat2d = self._get_grid(u850)
        mask = (
            (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) >= self.inner[0])
            & (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) <= self.inner[1])
        )
        u = np.asarray(u850.values).squeeze()
        v = np.asarray(v850.values).squeeze()
        R = 6371e3
        dlat = np.radians(np.abs(lat[1] - lat[0])) if len(lat) > 1 else 1e-3
        dlon = np.radians(np.abs(lon[1] - lon[0])) if len(lon) > 1 else 1e-3
        coslat = np.cos(np.radians(lat2d))
        # dv/dx - du/dy
        dv_dx = np.zeros_like(v)
        du_dy = np.zeros_like(u)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * R * coslat[:, 1:-1] * dlon)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * R * dlat)
        vort = (dv_dx - du_dy) * 1e5
        weights = np.cos(np.radians(lat2d)) * mask
        return float(np.nansum(vort * weights) / np.nansum(weights))

    def compute_pv(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """500 hPa potential vorticity (PVU). IC."""
        pv500 = self._get_var(ds, "pv", 500)
        if pv500 is not None:
            # ERA5 PV is in K m^2 kg^-1 s^-1; 1 PVU = 1e-6 K m^2 kg^-1 s^-1
            val = self._extract_annulus_mean(pv500, center_lon, center_lat, self.inner)
            return float(val * 1e6) if np.isfinite(val) else np.nan
        # Fallback: metpy calculation if available
        if mpcalc is not None:
            return np.nan  # TODO: implement with metpy when available
        return np.nan

    def compute_tpw(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """Total precipitable water (kg/m^2) via specific humidity integral."""
        q = self._get_var(ds, "q")
        if q is None:
            return np.nan
        # Support both 'level' (netcdf) and 'pressure_level' (grib renamed)
        lev_dim = None
        for d in ("level", "pressure_level", "isobaricInhPa"):
            if d in q.dims:
                lev_dim = d
                break
        if q is None or lev_dim is None:
            return np.nan
        # q in kg/kg, pressure in Pa
        p_levels = np.asarray(q[lev_dim].values) * 100  # hPa -> Pa
        q_prof = q.values  # (level, lat, lon)
        # Integral q/g dp over OA
        # Simplified: use mean q per level and trapezoid
        lon, lat, lon2d, lat2d = self._get_grid(q)
        mask = (
            (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) >= self.outer[0])
            & (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) <= self.outer[1])
        )
        weights = np.cos(np.radians(lat2d)) * mask
        g = 9.81
        tpw_grid = np.zeros_like(q_prof[0])
        # Integrate from top (lowest p index is highest level = 200 hPa)
        # to bottom (highest p index is lowest level = 1000 hPa) so that dp > 0
        for i in range(len(p_levels) - 1, 0, -1):
            dp = p_levels[i - 1] - p_levels[i]  # positive downward
            q_mean = (q_prof[i] + q_prof[i - 1]) / 2.0
            tpw_grid += q_mean * dp / g
        return float(np.nansum(tpw_grid * weights) / np.nansum(weights))

    def compute_sst(self, ds: Any, center_lon: float, center_lat: float,
                    annulus: Tuple[float, float]) -> float:
        sst = self._get_var(ds, "sst")
        if sst is None:
            return np.nan
        raw = self._extract_annulus_mean(sst, center_lon, center_lat, annulus)
        if np.isnan(raw):
            return np.nan
        # ERA5 SST is in Kelvin. Convert to Celsius.
        # Heuristic: tropical SST in K ≈ 298–305; in C ≈ 25–32.
        # A value > 200 is unambiguously Kelvin.
        if raw > 200:
            return float(raw - 273.15)
        return float(raw)

    def _load_ohc(self, sid: str) -> Optional[Any]:
        """Load the per-storm OHC NetCDF (data/raw/ohc/{SID}/{SID}_ohc.nc).

        Returns the OHC DataArray with dims (time, lat, lon), or None when
        unavailable.  The field is the ocean heat content relative to the
        26 degC isotherm, unit kJ/cm^2, daily.
        """
        if xr is None:
            return None
        path = Path(self.ohc_dir) / sid / f"{sid}_ohc.nc"
        if not path.exists() or path.stat().st_size == 0:
            return None
        try:
            ds = xr.open_dataset(path)
            return ds["OHC"]
        except Exception as exc:
            logger.warning(f"Could not load OHC for {sid}: {exc}")
            return None

    def _prep_ohc_6h(self, ohc_da: Any, df_bt: pd.DataFrame) -> Optional[Tuple]:
        """Time-interpolate the daily OHC field to the storm's 6h track times.

        Done ONCE per storm (vectorized over all track times) instead of per
        timestep — per-timestep interp on the full grid is prohibitively slow.
        Also pre-subsets spatially to the track bounding box + annulus margin.

        Returns (ohc_6h DataArray (time6h, lat, lon), (lon2d, lat2d)) or None.
        """
        lat_col = self.cfg.get("best_track.lat_col")
        lon_col = self.cfg.get("best_track.lon_col")
        time_col = self.cfg.get("best_track.time_col")
        times = pd.to_datetime(df_bt[time_col])
        if getattr(times.dt, "tz", None) is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        margin = self.outer[1] + 1.0
        lat_c = df_bt[lat_col].astype(float)
        lon_c = normalize_lon(df_bt[lon_col].astype(float).values)

        lat_name = "latitude" if "latitude" in ohc_da.dims else "lat"
        lon_name = "longitude" if "longitude" in ohc_da.dims else "lon"
        da = ohc_da
        try:
            file_lon = da[lon_name].values
            # Harmonize storm longitudes to the file's convention
            if file_lon.max() > 180 and lon_c.min() < file_lon.min() - margin:
                lon_c = np.mod(lon_c, 360.0)
            lon_min, lon_max = lon_c.min() - margin, lon_c.max() + margin
            lat_min, lat_max = lat_c.min() - margin, lat_c.max() + margin
            crosses_dateline = lon_min < file_lon.min() or lon_max > file_lon.max()
            if not crosses_dateline:
                da = da.sel({lat_name: slice(lat_min, lat_max),
                             lon_name: slice(lon_min, lon_max)})
        except Exception:
            pass  # fall back to the full field

        try:
            # DAILY-MEAN semantics: GLORYS daily fields cover [D 00, D+1 00), so
            # 00/06/12/18 UTC on day D all use day D-1's completed field.  The
            # old linear interp blended in the following day.
            da6h = backward_time_reindex(da, times.values)
        except Exception as exc:
            logger.warning(f"OHC time selection failed: {exc}")
            return None
        lon = np.asarray(da6h[lon_name].values)
        lat = np.asarray(da6h[lat_name].values)
        lon2d, lat2d = np.meshgrid(lon, lat)
        return da6h, (lon2d, lat2d), times

    def compute_ohc(self, ohc_6h: Any, ohc_grid: Tuple, t_idx: int,
                    center_lon: float, center_lat: float,
                    radius: Tuple[float, float]) -> float:
        """Annulus mean of OHC (kJ/cm^2) around the track centre.

        Same inner/outer annulus convention as the ERA5 environmental
        variables; planar distance with cos(lat) correction and dateline-safe
        longitude wrap.
        """
        if ohc_6h is None or ohc_grid is None:
            return np.nan
        lon2d, lat2d = ohc_grid
        try:
            data = np.asarray(ohc_6h.isel(time=t_idx).values).squeeze()
        except Exception:
            return np.nan
        if data.ndim != 2:
            return np.nan
        if not np.isfinite(data).any():
            return np.nan  # e.g. time outside the OHC field's coverage
        cos_center = np.cos(np.radians(center_lat))
        dlon = (lon2d - center_lon + 180.0) % 360.0 - 180.0
        dist = np.sqrt((dlon * cos_center) ** 2 + (lat2d - center_lat) ** 2)
        mask = (dist >= radius[0]) & (dist <= radius[1])
        if not np.any(mask):
            return np.nan
        weights = np.cos(np.radians(lat2d)) * mask
        weights = weights * np.isfinite(data)  # only valid cells count (NaN-weight bug)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            denom = np.nansum(weights)
            if denom == 0:
                return np.nan
            return float(np.nansum(data * weights) / denom)

    def compute_pot(self, ds: Any, center_lon: float, center_lat: float, vmax_kt: float = 0.0) -> float:
        """Simplified potential intensity deficit (kt) = PI - VMAX.

        PI is estimated as 25 * (SST - 26) m/s (crude Emanuel-like proxy),
        then converted to kt for consistency with IBTrACS VMAX (1 kt = 0.5144 m/s).
        POT represents the available intensification potential.
        """
        sst = self.compute_sst(ds, center_lon, center_lat, self.outer)
        if np.isnan(sst):
            return np.nan
        pi_ms = max(0.0, (sst - 26.0) * 25.0)
        pi_kt = pi_ms / 0.5144  # m/s → kt
        return float(max(0.0, pi_kt - vmax_kt))

    def compute_warm_core(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """200 hPa temperature anomaly in IC vs OA (warm core signal strongest at 200 hPa)."""
        t200 = self._get_var(ds, "t", 200)
        if t200 is None:
            return np.nan
        ic_mean = self._extract_annulus_mean(t200, center_lon, center_lat, self.inner)
        oa_mean = self._extract_annulus_mean(t200, center_lon, center_lat, self.outer)
        if np.isnan(ic_mean) or np.isnan(oa_mean):
            return np.nan
        return float(ic_mean - oa_mean)

    def compute_mfc(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """850 hPa moisture flux convergence ~ -div(q*V) in OA."""
        q850 = self._get_var(ds, "q", 850)
        u850 = self._get_var(ds, "u", 850)
        v850 = self._get_var(ds, "v", 850)
        if any(v is None for v in [q850, u850, v850]):
            return np.nan
        qu = np.asarray(q850.values).squeeze() * np.asarray(u850.values).squeeze()
        qv = np.asarray(q850.values).squeeze() * np.asarray(v850.values).squeeze()
        lon, lat, lon2d, lat2d = self._get_grid(q850)
        mask = (
            (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) >= self.outer[0])
            & (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) <= self.outer[1])
        )
        R = 6371e3
        dlat = np.radians(np.abs(lat[1] - lat[0])) if len(lat) > 1 else 1e-3
        dlon = np.radians(np.abs(lon[1] - lon[0])) if len(lon) > 1 else 1e-3
        coslat = np.cos(np.radians(lat2d))
        dqu_dx = np.zeros_like(qu)
        dqv_dy = np.zeros_like(qv)
        dqu_dx[:, 1:-1] = (qu[:, 2:] - qu[:, :-2]) / (2 * R * coslat[:, 1:-1] * dlon)
        dqv_dy[1:-1, :] = (qv[2:, :] - qv[:-2, :]) / (2 * R * dlat)
        mfc = -(dqu_dx + dqv_dy) * 1e4  # scale for readability
        weights = np.cos(np.radians(lat2d)) * mask
        return float(np.nansum(mfc * weights) / np.nansum(weights))

    def compute_slhf_ic(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """Surface latent heat flux in W/m^2 (ocean -> atmosphere positive).

        ERA5 reanalysis 'surface_latent_heat_flux' (CDS dataset
        reanalysis-era5-single-levels) is the accumulation over the **1 hour
        preceding each valid time** (units: J/m^2) — NOT a continuously
        accumulating forecast field.  Differencing two 6-hourly values and
        dividing by 21600 s therefore yields ~(1h-flux difference)/6, i.e.
        near-noise ~6x too small (review C-H1).  The correct conversion is
        simply value / 3600 s.

        ERA5's sign convention is negative upward; we flip the sign so that
        positive = ocean releases latent heat to the atmosphere (the physical
        driver of intensification).  Verified on storm 2000125N06136: raw
        ~-3.4e5 J/m^2 -> ~95 W/m^2 upward, within the expected 50-200 W/m^2
        tropical range.
        """
        slhf = self._get_var(ds, "slhf")
        if slhf is None:
            return np.nan
        raw = self._extract_annulus_mean(slhf, center_lon, center_lat, self.inner)
        if not np.isfinite(raw):
            return np.nan
        return float(-raw / 3600.0)

    def compute_dry_air(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """Fraction of OA gridpoints with 700-500 hPa RH < 40%."""
        rh700 = self._get_var(ds, "r", 700)
        rh500 = self._get_var(ds, "r", 500)
        if rh700 is None or rh500 is None:
            return np.nan
        lon, lat, lon2d, lat2d = self._get_grid(rh700)
        mask = (
            (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) >= self.outer[0])
            & (np.sqrt((lon2d - center_lon) ** 2 + (lat2d - center_lat) ** 2) <= self.outer[1])
        )
        rh_mean = (np.asarray(rh700.values).squeeze() + np.asarray(rh500.values).squeeze()) / 2.0
        dry = (rh_mean < 40.0) & mask
        total = mask.sum()
        return float(dry.sum() / total) if total > 0 else np.nan

    def compute_w500(self, ds: Any, center_lon: float, center_lat: float) -> float:
        """500 hPa vertical velocity (Pa/s) over OA. Negative = ascent."""
        w500 = self._get_var(ds, "w", 500)
        if w500 is None:
            return np.nan
        return self._extract_annulus_mean(w500, center_lon, center_lat, self.outer)

    def compute_shear_vmax_angle(self, row: pd.Series, shear_dir: float) -> float:
        """Angle between shear vector and TC motion direction (0=aligned)."""
        tc_dir = row.get("DIR", np.nan)
        if np.isnan(tc_dir) or np.isnan(shear_dir):
            return np.nan
        angle = abs(shear_dir - tc_dir)
        if angle > 180.0:
            angle = 360.0 - angle
        return float(angle)

    # ------------------------------------------------------------------
    # Storm-level extraction pipeline
    # ------------------------------------------------------------------

    def extract_storm(self, sid: str, df_bt: pd.DataFrame, ds: Any) -> pd.DataFrame:
        """Return DataFrame of derived env vars for each timestep in df_bt."""
        records = []
        lat_col = self.cfg.get("best_track.lat_col")
        lon_col = self.cfg.get("best_track.lon_col")
        time_col = self.cfg.get("best_track.time_col")

        # OHC (optional): daily ocean heat content field for this storm,
        # time-interpolated once to all 6h track times.
        want_ohc = self.ohc_enabled and ("OHC_IC" in self.derived or "OHC_OA" in self.derived)
        ohc_6h, ohc_grid = None, None
        if want_ohc:
            ohc_da = self._load_ohc(sid)
            if ohc_da is None:
                logger.warning(f"OHC file missing/unreadable for {sid}; OHC_* will be NaN")
            else:
                prepped = self._prep_ohc_6h(ohc_da, df_bt)
                if prepped is not None:
                    ohc_6h, ohc_grid, _ = prepped

        for ridx, (_, row) in enumerate(df_bt.iterrows()):
            t = row[time_col]
            lat = row[lat_col]
            lon = normalize_lon(row[lon_col])
            ds_t = self.nearest_time_select(ds, pd.Timestamp(t))
            if ds_t is None:
                rec = {k: np.nan for k in self.derived}
                rec.update({"SID": sid, "ISO_TIME": t, "LAT": lat, "LON": lon})
                records.append(rec)
                continue
            rec = {"SID": sid, "ISO_TIME": t, "LAT": lat, "LON": lon}
            # Shear
            shear_res = self.compute_shear(ds_t, lon, lat)
            rec.update(shear_res)
            if "SHEAR_VMAX_ANGLE" in self.derived:
                rec["SHEAR_VMAX_ANGLE"] = self.compute_shear_vmax_angle(row, rec.get("SHEAR_DIR", np.nan))
            if "D200" in self.derived:
                rec["D200"] = self.compute_d200(ds_t, lon, lat)
            if "RHLO" in self.derived:
                rec["RHLO"] = self.compute_rh_layer(ds_t, lon, lat, [850, 700], self.outer)
            if "RHHI" in self.derived:
                rec["RHHI"] = self.compute_rh_layer(ds_t, lon, lat, [500, 300], self.outer)
            if "VORT" in self.derived:
                rec["VORT"] = self.compute_vorticity(ds_t, lon, lat)
            if "PV" in self.derived:
                rec["PV"] = self.compute_pv(ds_t, lon, lat)
            if "TPW" in self.derived:
                rec["TPW"] = self.compute_tpw(ds_t, lon, lat)
            if "SST_OA" in self.derived:
                rec["SST_OA"] = self.compute_sst(ds_t, lon, lat, self.outer)
            if "SST_IC" in self.derived:
                rec["SST_IC"] = self.compute_sst(ds_t, lon, lat, self.inner)
            if "OHC_IC" in self.derived:
                rec["OHC_IC"] = self.compute_ohc(ohc_6h, ohc_grid, ridx, lon, lat, self.inner)
            if "OHC_OA" in self.derived:
                rec["OHC_OA"] = self.compute_ohc(ohc_6h, ohc_grid, ridx, lon, lat, self.outer)
            if "POT" in self.derived:
                rec["POT"] = self.compute_pot(ds_t, lon, lat, vmax_kt=row.get("VMAX", 0.0))
            if "WARM_CORE" in self.derived:
                rec["WARM_CORE"] = self.compute_warm_core(ds_t, lon, lat)
            if "MFC" in self.derived:
                rec["MFC"] = self.compute_mfc(ds_t, lon, lat)
            if "SLHF_IC" in self.derived:
                # ERA5 reanalysis SLHF is the 1h accumulation preceding each
                # valid time; compute_slhf_ic converts to W/m^2 directly —
                # NO 6h differencing (review C-H1 / FIX_TASKS P0-4).
                rec["SLHF_IC"] = self.compute_slhf_ic(ds_t, lon, lat)
            if "DRY_AIR" in self.derived:
                rec["DRY_AIR"] = self.compute_dry_air(ds_t, lon, lat)
            if "W500" in self.derived:
                rec["W500"] = self.compute_w500(ds_t, lon, lat)
            records.append(rec)
        out = pd.DataFrame(records)
        return out


class Preprocessor:
    """Yeo-Johnson, Winsorize, Z-score standardization."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.yj_path = Path(config.get("preprocessing.yj_transformer_path"))
        self.scaler_path = Path(config.get("preprocessing.scaler_path"))
        self.winsor = config.get("preprocessing.winsorize_limits", [0.01, 0.99])
        self.scaler_fit_years = config.get("preprocessing.scaler_fit_years", [2000, 2018])
        mask_path = config.get("preprocessing.imputation_mask_path")
        self.imputation_mask_path = Path(mask_path) if mask_path else None
        self.imputation_mask_columns = list(
            config.get("preprocessing.imputation_mask_columns", []) or [])
        self.yj = None
        self.scaler = None

    def _train_mask(self, df: pd.DataFrame) -> pd.Series:
        """Rows eligible for every fitted preprocessing statistic.

        Winsorize quantiles, Yeo-Johnson lambda, scaler statistics and NaN fill
        values all use the canonical SID/row fit window, the realised F24_TIME
        boundary and the strict t+30h embargo.
        """
        return strict_fit_row_mask(df, self.scaler_fit_years)

    def fit_dv24_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit Yeo-Johnson on training-period DV24 only, then transform all."""
        self.yj = PowerTransformer(method="yeo-johnson", standardize=False)
        dv24_train = df.loc[self._train_mask(df), "DV24"].dropna().values.reshape(-1, 1)
        if len(dv24_train) == 0:
            raise ValueError("No training-period DV24 values to fit Yeo-Johnson on")
        self.yj.fit(dv24_train)
        save_pickle(self.yj, self.yj_path)
        out = np.full(len(df), np.nan)
        notna = df["DV24"].notna().values
        if notna.any():
            out[notna] = self.yj.transform(df.loc[notna, "DV24"].values.reshape(-1, 1)).ravel()
        df["DV24_YJ"] = out
        logger.info(f"Yeo-Johnson fitted on {len(dv24_train)} training-period DV24 obs; saved to {self.yj_path}")
        return df

    def winsorize_dv24(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize DV24 to [low, high] quantiles fitted on train years only."""
        lo, hi = self.winsor
        low_q = df.loc[self._train_mask(df), "DV24"].quantile(lo)
        high_q = df.loc[self._train_mask(df), "DV24"].quantile(hi)
        df["DV24"] = df["DV24"].clip(lower=low_q, upper=high_q)
        logger.info(f"Winsorized DV24 to [{low_q:.2f}, {high_q:.2f}] (quantiles from {self.scaler_fit_years} fit period)")
        return df

    def _write_imputation_mask(self, df: pd.DataFrame,
                               original_missing: pd.DataFrame,
                               feature_cols: List[str]) -> None:
        """Persist pre-fill missingness without changing the ATE input panel."""
        if self.imputation_mask_path is None:
            return
        if list(original_missing.columns) != list(feature_cols):
            raise RuntimeError("internal imputation-mask column mismatch")
        keys = df[IMPUTATION_KEY_COLUMNS]
        if keys.isna().any().any() or keys.duplicated().any():
            raise RuntimeError("imputation mask requires unique, non-null SID+ISO_TIME keys")
        mask = df[IMPUTATION_KEY_COLUMNS].copy()
        mask["ISO_TIME"] = pd.to_datetime(mask["ISO_TIME"], utc=True)
        for column in feature_cols:
            mask[IMPUTATION_PREFIX + column] = original_missing[column].to_numpy(dtype=bool)
        if len(mask) != len(df) or mask[IMPUTATION_KEY_COLUMNS].duplicated().any():
            raise RuntimeError("imputation mask does not cover each panel row exactly once")
        self.imputation_mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.to_parquet(self.imputation_mask_path, index=False)
        logger.info(
            f"Imputation mask saved to {self.imputation_mask_path}: "
            f"{len(mask)} rows, {len(feature_cols)} features, "
            f"{int(original_missing.to_numpy().sum())} originally missing cells")

    def fit_feature_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit Z-score scaler on training-year environmental features.

        Only rows with actual environmental data (non-NaN SHRD) are used for
        fitting and transformation. Rows without environmental data retain NaN.
        When configured, the original per-cell missingness is saved before any
        mean fill so downstream causal discovery can keep those cells masked.
        """
        train_mask = self._train_mask(df)
        # RI and best-track metadata are not environmental features. USA_PRES
        # must not mark rows without ERA5 as eligible for mean filling.
        exclude = {"SID", "ISO_TIME", "F24_TIME", "NAME", "DV24", "DV24_YJ", "LC_STAGE", "VMAX", "VMAX_F24",
                   "PER", "SPD", "DIR", "LAT", "LON", "LAND_DIST", "SHRD_RAW", "RI", "USA_PRES", "storm_year"}
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fi"]
        feature_cols = [c for c in feature_cols if df.loc[train_mask, c].notna().any()]
        if not feature_cols:
            if self.imputation_mask_path is not None:
                raise RuntimeError("no feature columns available for required imputation mask")
            logger.warning("No non-NaN feature columns found for scaler fitting; skipping")
            return df
        if self.imputation_mask_path is not None and self.imputation_mask_columns:
            expected = self.imputation_mask_columns
            if (len(expected) != len(set(expected))
                    or len(feature_cols) != len(expected)
                    or set(feature_cols) != set(expected)):
                raise RuntimeError(
                    "scaler feature columns do not match required imputation-mask schema: "
                    f"missing={sorted(set(expected) - set(feature_cols))} "
                    f"extra={sorted(set(feature_cols) - set(expected))}")

        if "SHRD" in df.columns:
            env_present = df["SHRD"].notna()
        else:
            env_present = df[feature_cols].notna().any(axis=1)
        train_vals = df.loc[train_mask & env_present, feature_cols].dropna()
        if len(train_vals) == 0:
            if self.imputation_mask_path is not None:
                raise RuntimeError("no complete fit rows available for required imputation mask")
            logger.warning("No training samples after dropna; skipping scaler fit")
            return df
        self.scaler = StandardScaler()
        self.scaler.fit(train_vals)
        save_pickle(self.scaler, self.scaler_path)
        # Preserve every original missing cell before fit-only mean imputation.
        original_missing = df.loc[:, feature_cols].isna().copy()
        if env_present.any():
            fill_vals = df.loc[train_mask & env_present, feature_cols].mean()
            vals = df.loc[env_present, feature_cols].fillna(fill_vals)
            df.loc[env_present, feature_cols] = self.scaler.transform(vals)
        self._write_imputation_mask(df, original_missing, feature_cols)
        logger.info(f"Z-score scaler fitted on {len(train_vals)} obs; {len(feature_cols)} features; saved to {self.scaler_path}")
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        with phase_timer("Preprocessing"):
            # Preserve raw (pre-scaling) SHRD for nonlinear binned analysis.
            # The bins [0, 5, 15, inf] are defined in m/s on raw values.
            if "SHRD" in df.columns:
                df["SHRD_RAW"] = df["SHRD"] * 1.94384  # m/s → kt for config bins
            df = self.winsorize_dv24(df)
            df = self.fit_dv24_transform(df)
            df = self.fit_feature_scaler(df)
        return df
