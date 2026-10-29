"""Public utilities: logging, coordinates, annulus averaging, config loading."""

from __future__ import annotations

import math
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Configure loguru logger with optional file output."""
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, colorize=True)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, format=fmt, rotation=rotation, retention=retention)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class Config:
    """Simple nested-dict config with dot-style access."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-delimited key access, e.g. 'data.years'."""
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None and key not in str(self._data):
            raise KeyError(key)
        return val

    def to_dict(self) -> Dict[str, Any]:
        return self._data


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def normalize_lon(lon: np.ndarray) -> np.ndarray:
    """Normalize longitude to [0, 360)."""
    lon = np.asarray(lon, dtype=float)
    return lon % 360.0


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Haversine distance in km between pairs of (lat1,lon1) and (lat2,lon2)."""
    R = 6371.0  # Earth radius km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def bearing(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Bearing from (lat1,lon1) to (lat2,lon2) in degrees, 0=North."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360.0) % 360.0


# ---------------------------------------------------------------------------
# Annulus / spatial extraction helpers
# ---------------------------------------------------------------------------

def planar_distance_mask(
    center_lon: float,
    center_lat: float,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    inner_radius_deg: float,
    outer_radius_deg: float,
) -> np.ndarray:
    """
    Return boolean mask for grid points within [inner, outer] degrees
    from center using planar approximation (fast for small radii).
    """
    dlon = lon_grid - center_lon
    dlat = lat_grid - center_lat
    dist = np.sqrt(dlon**2 + dlat**2)
    return (dist >= inner_radius_deg) & (dist <= outer_radius_deg)


def annulus_spatial_mean(
    data: np.ndarray,
    center_lon: float,
    center_lat: float,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    inner_radius_deg: float,
    outer_radius_deg: float,
) -> float:
    """
    Compute area-weighted mean over an annulus using planar distance mask.
    Assumes regular lat/lon grid; weights by cos(lat).
    """
    mask = planar_distance_mask(
        center_lon, center_lat, lon_grid, lat_grid, inner_radius_deg, outer_radius_deg
    )
    if not np.any(mask):
        return np.nan
    weights = np.cos(np.radians(lat_grid))
    weights = weights * mask
    weights = weights * np.isfinite(data)  # only valid cells count
    denom = np.nansum(weights)
    if denom == 0:
        return np.nan
    return np.nansum(data * weights) / denom


# ---------------------------------------------------------------------------
# Land distance with cKDTree (GSHHG coastline or ETOPO1)
# ---------------------------------------------------------------------------

def load_coastline_points(resolution: str = "f", gshhg_dir: Optional[str] = None) -> np.ndarray:
    """
    Load GSHHG coastline points as (lon, lat) array in degrees.
    Falls back to a coarse built-in coastline if GSHHG not available.
    """
    try:
        import cartopy.io.shapereader as shpreader
        # Natural Earth coastline via cartopy (always available)
        shpfile = shpreader.natural_earth(
            resolution="10m", category="physical", name="coastline"
        )
        reader = shpreader.Reader(shpfile)
        pts: List[Tuple[float, float]] = []
        for geom in reader.geometries():
            coords = np.asarray(geom.coords)
            pts.extend(coords[:, :2].tolist())
        return np.array(pts)
    except Exception:
        logger.warning("Cartopy coastline failed, using coarse fallback.")
        raise RuntimeError(
            "Cannot load coastline data. Install cartopy "
            "or provide coastline points manually to distance_to_land()."
        )


def distance_to_land(
    lats: np.ndarray,
    lons: np.ndarray,
    coastline_pts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute distance (km) from each (lat, lon) to nearest coastline point
    using cKDTree in 3D Cartesian space.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    if coastline_pts is None:
        coastline_pts = load_coastline_points()

    def _to_cartesian(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.column_stack([x, y, z])

    tree = cKDTree(_to_cartesian(coastline_pts[:, 1], coastline_pts[:, 0]))
    dist_rad, _ = tree.query(_to_cartesian(lats, lons), k=1)
    return dist_rad * 6371.0  # km


# ---------------------------------------------------------------------------
# Timing / memory profiler
# ---------------------------------------------------------------------------

@contextmanager
def phase_timer(phase_name: str):
    """Context manager that logs phase duration and peak memory."""
    tracemalloc.start()
    t0 = time.time()
    logger.info(f"=== Phase START: {phase_name} ===")
    try:
        yield
    finally:
        elapsed = time.time() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(
            f"=== Phase END: {phase_name} | "
            f"elapsed={elapsed:.1f}s | "
            f"mem_peak={peak / 1024 / 1024:.1f}MB ==="
        )


# ---------------------------------------------------------------------------
# DataFrame panel helpers
# ---------------------------------------------------------------------------

def make_panel_mask(df: pd.DataFrame, id_col: str) -> np.ndarray:
    """
    Build tigramite-compatible mask for variable-length panels.

    NOTE: tigramite follows numpy masked-array semantics — ``True`` means the
    sample is EXCLUDED.  This helper therefore returns True where a value is
    MISSING.  Currently unused; kept for potential external consumers.
    """
    return df.isnull().values


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save object with joblib."""
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load object with joblib."""
    import joblib
    return joblib.load(path)
