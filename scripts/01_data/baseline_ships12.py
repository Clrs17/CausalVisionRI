# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""WRPS1-style baseline: LDA on the 12 SHIPS predictors, recomputed on OUR panel.

Replicates the 12-predictor set of Horinouchi et al. (2025, WAF; WRPS1) Table 1
with SHIPS developmental-data definitions (CIRA ships_predictor_file.pdf), but
computed row-by-row on our own panel (all eligible WNP systems, 2000-2024,
6-hourly, paired HURSAT frames) so that the LDA comparison is apples-to-apples
with the DL arm: identical sample rows, RI labels and year-based holdout protocol.

Predictor definitions (WRPS1 Table 1; radial ranges = areal averages):
  VMAX  Max sustained wind (kt)                    panel VMAX (IBTrACS USA_WIND, 1-min)   [EXACT]
  INCV  Intensity change t-6h -> t0 (kt)           panel PER (native 3h IBTrACS diff(2))  [EXACT]
  POT   MPI - VMAX (kt); MPI = tcpyPI (Bister & Emanuel 2002, pyPI 1.4.1) from the
        ERA5 0-200 km mean sounding (1000-200 hPa + log-p extrapolated 150 hPa and
        isothermal 100/50 hPa TTL extension), CKCD=0.9, V_reduc=1.0, dissipative
        heating on.  SST/MSLP = 0-200 km ERA5 means.                              [APPROX:
        ERA5 per-storm archive tops out at 200 hPa; extension tested 108->178 kt
        on a sample sounding — treated as the best available local estimate]
  OHCD  Time-dependent OHC (kJ/cm2, rel. 26 C), CMEMS GLORYS12V1 daily product.
        Those "daily" fields are DAILY MEANS over [D 00, D+1 00), so OHCD uses
        the most recent field whose window has closed (source_time + 24 h <=
        t0): 00/06/12/18 UTC on day D all use day D-1's field, and day D+1 00
        UTC is the first moment day D's field is complete.  0-100 km areal mean.
        MAIN variant — WRPS1 had no time-dependent OHC in SHIPS and used
        climatology instead.                                                    [BETTER]
  COHC  Climatological OHC (kJ/cm2): monthly climatology pooled from the FIT-
        window GLORYS fields only (2000-01-01 <= t < 2018-01-01, applied per
        time step) on a 0.5-deg grid, same 0-100 km areal mean at
        (lat, lon, month(t0)).  WRPS1-spec variant (appendix).  [EXACT-ish:
        SHIPS uses NCODA 2005-2010; ours is GLORYS 2000-2017 sampled at TC times]
  SHRD  850-200 hPa shear magnitude (kt), 200-800 km annulus, ERA5 u/v        [EXACT]
  T150  150-hPa temperature (C), 200-800 km — 150 hPa NOT in local ERA5
        archive: log-p EXTRAPOLATION from 200/300 hPa                         [SUBSTITUTED]
  T200  200-hPa temperature (C), 200-800 km                                   [EXACT]
  T250  250-hPa temperature (C), 200-800 km — log-p interpolation 200/300 hPa [INTERP]
  R000  1000-hPa relative humidity (%), 200-800 km                            [EXACT]
  SDIR  Std dev of IR brightness temperature (C) within 0-200 km, from the
        paired HURSAT-B1 TBB frame (0.07-deg GridSat-B1 grid, storm-centred)  [EXACT-ish:
        SHIPS uses GOES; HURSAT intercalibrates geostationary IR to a common grid]
  Z850  850-hPa relative vorticity (1e-7 s-1), 0-1000 km, finite-differenced
        from ERA5 u/v (vo was not in the phase-1 download list)               [EXACT method]
  DTL   Distance to nearest land (km) — panel LAND_DIST (GSHHG coastline
        cKDTree); SHIPS uses "nearest MAJOR landmass" (small islands differ)  [APPROX]

Protocol:
  * HOLDOUT_MASKED_V1 (current, ALL WNP SYSTEMS — 725 canonical systems from
    the all-system panel, no peak-intensity filter): the row-level source table
    ``data/ships12_predictors_alltc_0024.parquet`` is built from the raw
    ERA5/GLORYS archives by ``--build-predictors`` (or ``--recompute``) over the
    QC v1 frame corpus declared by the config (``data.frames_dir`` =
    ``data/global_hursat/frames_qc``, audited by ``data.qc_manifest``); the raw
    download corpus is only reachable through an explicit ``--allow-raw-frames``
    audit build.  The table is only usable through its mandatory build manifest
    ``data/ships12_predictors_alltc_0024.manifest.json``: protocol/population,
    panel/frames-index/config/build-code SHA, every output row key, per-SID
    success/failure stats, every raw input path+content SHA, the QC v1 binding
    (``frames_qc_sha256`` / ``qc_manifest_sha256``, rule/version,
    ``stores_sha256``, frame-store and frames-index hashes), the OHCD
    algorithm/source window and the COHC derivation.  The fit-only feature
    tables are then built by ``build_ships12_features``; OHCD is recomputed on
    the backward daily window, COHC is a fit-window (2000-2017) content-hashed
    climatology.  No 375-storm >=65 kt branch exists.
    Row universe (shared with LR/LDA and the DL dataset): eligible panel minus
    the ``frame_availability_v1`` exclusions named by
    ``config.data.frame_availability``; the availability path+SHA+counts are
    pinned in every manifest.
  * legacy LOYO (archived, ``archive/pre_holdout_legacy/`` only, reachable only
    with an explicit ``--legacy``): the pre-holdout sample, held years
    2019-2024, train = years <= 2018 minus inner minus held; threshold =
    inner-split F1 max; per-fold median imputation + StandardScaler on the
    fold's train rows.  Kept for historical audit and never used by the holdout
    protocol (no row count is hard-coded here).
  * Any per-storm rebuild failure (missing/unreadable ERA5, GLORYS or frame
    input, or a row-level extraction error) aborts the build immediately with a
    non-zero exit; the all-system rebuild is expected to be 725/725, and the
    output SID/row_uid set is always checked against the expected panel.

Outputs (wpac_causal_phase2/):
  data/ships12_predictors_alltc_0024.parquet        per-row source table
  data/ships12_predictors_alltc_0024.manifest.json   mandatory build manifest
  data/ships12_raw_input_sha256.json                raw-input hash cache
  data/holdout_masked_v1/ships12_predictors_fit0017_{ohcd,cohc}.parquet
  data/ships12_cohc_clim_alltc_0024.npz             fit-only OHC climatology
  data/runs_holdout/baselines/baseline_ships12_*    versioned run dirs

Frame corpus: QC v1 only (``data/global_hursat/frames_qc`` + ``qc_manifest.json``
from the config).  ``--allow-raw-frames`` builds an audit-only table that the
feature cache refuses.

Usage (from the release root):
    python scripts/01_data/baseline_ships12.py --build-predictors   # source table + manifest
    python scripts/01_data/baseline_ships12.py                 # fit features + holdout eval
    python scripts/01_data/baseline_ships12.py --eval-only      # skip straight to eval
    python scripts/01_data/baseline_ships12.py --legacy --config <legacy yaml>  # archive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

P2_ROOT = Path(__file__).resolve().parents[2]
P1_ROOT = P2_ROOT / "data" / "phase1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.frame_store import frame_array_path, frame_time_path  # noqa: E402
from wpac_warning.protocol import sha256_file  # noqa: E402
from wpac_warning.train import bss_score, pick_threshold, safe_auc  # noqa: E402

# ALL-SYSTEM paths (725 WNP systems).  The 375-storm >=65 kt samples are not an
# analysis branch of HOLDOUT_MASKED_V1 and none of these defaults point at them.
PANEL_PARQUET = (P1_ROOT / "processed_masked" / "holdout_fit0017" /
                 "panel" / "wpac_tc_dataset_processed.parquet")
FRAMES_INDEX = P2_ROOT / "data" / "holdout_masked_v1" / "frames_index.parquet"
ERA5_DIR = P1_ROOT / "raw" / "era5"
OHC_DIR = P1_ROOT / "raw" / "ohc"
# Frame corpora: production reads the QC v1 corpus declared by
# config data.frames_dir; the raw download corpus is the QC/re-key input and is
# reachable only through an explicit --allow-raw-frames audit build.
RAW_FRAMES_DIR = P2_ROOT / "data" / "global_hursat" / "frames"
QC_FRAMES_DIR = P2_ROOT / "data" / "global_hursat" / "frames_qc"
FRAMES_DIR = QC_FRAMES_DIR
QC_MANIFEST = P2_ROOT / "data" / "global_hursat" / "qc_manifest.json"
DEFAULT_PRED_CACHE = P2_ROOT / "data" / "ships12_predictors_alltc_0024.parquet"
PRED_CACHE = DEFAULT_PRED_CACHE
DEFAULT_SOURCE_MANIFEST = DEFAULT_PRED_CACHE.with_suffix(".manifest.json")
RAW_SHA_CACHE = P2_ROOT / "data" / "ships12_raw_input_sha256.json"
COHC_CLIM_NPZ = P2_ROOT / "data" / "ships12_cohc_clim_alltc_0024.npz"
DEFAULT_CONFIG = P2_ROOT / "configs" / "phase2" / "holdout_masked_v1.yaml"

KM_PER_DEG = 111.195
KT_PER_MS = 1.0 / 0.5144
FRAME_CENTER_IDX = 112          # (301-224)//2 = 38 crop offset; 150-38 = 112
FRAME_KM_PER_PX = 0.07 * KM_PER_DEG  # GridSat-B1 0.07-deg equal-angle grid

LOYO_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
INNER_YEARS = [2016, 2017, 2018]

# 12 predictors; OHCD <-> COHC swap selects main vs WRPS1-spec variant
PREDICTORS_MAIN = ["VMAX", "INCV", "POT", "OHCD", "SHRD", "T150",
                   "T200", "T250", "R000", "SDIR", "Z850", "DTL"]


# ---------------------------------------------------------------------------
# per-storm worker: recompute SHIPS-defined predictors from raw fields
# ---------------------------------------------------------------------------

def _dist_km(lon2d: np.ndarray, lat2d: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Planar distance (km) with cos(lat0) longitude correction (phase-1 convention)."""
    cos0 = np.cos(np.radians(lat0))
    return np.sqrt(((lon2d - lon0) * cos0) ** 2 + (lat2d - lat0) ** 2) * KM_PER_DEG


def _amean(arr: np.ndarray, mask: np.ndarray, lat2d: np.ndarray) -> float:
    """Area-weighted mean over mask, valid (finite) cells only.

    ``mask``/``lat2d`` must be on the field's own grid: the ERA5 pressure-level
    and surface files are separate CDS requests and may not share a lat/lon box,
    so a mask built for one file must never be applied to the other.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim > 2:
        a = a.squeeze()
    if a.shape != mask.shape:
        raise ValueError(
            f"field {a.shape} and mask {mask.shape} are on different grids; "
            f"build the mask on the field's own lat/lon grid")
    w = np.cos(np.radians(lat2d)) * mask * np.isfinite(a)
    den = np.sum(w)
    if den == 0:
        return np.nan
    return float(np.sum(np.where(np.isfinite(a), a, 0.0) * w) / den)


def _vort850_1e7(u850: np.ndarray, v850: np.ndarray, lat: np.ndarray, mask: np.ndarray,
                 lat2d: np.ndarray) -> float:
    """Relative vorticity (1e-7 s-1) at 850 hPa, finite-differenced on the lat-lon
    grid, area-averaged over the 0-1000 km disk (SHIPS Z850)."""
    R = 6371e3
    # signed spacing: CDS ERA5 latitude is DESCENDING (N->S), so a signed dlat
    # keeps du/dy in true northward convention (abs() would flip its sign)
    dlat = np.radians(lat[1] - lat[0]) if len(lat) > 1 else 1e-3
    dlon = np.radians(abs(lat[1] - lat[0])) if len(lat) > 1 else 1e-3  # 0.25-deg equal-angle grid
    coslat = np.cos(np.radians(lat2d))
    dv_dx = np.zeros_like(v850)
    du_dy = np.zeros_like(u850)
    dv_dx[:, 1:-1] = (v850[:, 2:] - v850[:, :-2]) / (2 * R * coslat[:, 1:-1] * dlon)
    du_dy[1:-1, :] = (u850[2:, :] - u850[:-2, :]) / (2 * R * dlat)
    vort = (dv_dx - du_dy) * 1e7
    return _amean(vort, mask, lat2d)


def _mpi_kt(t_c: np.ndarray, q_kgkg: np.ndarray, levs_hpa: np.ndarray,
            sst_c: float, msl_hpa: float) -> float:
    """Emanuel MPI (kt) via tcpyPI on the 0-200 km mean sounding.

    Local ERA5 tops out at 200 hPa; append a TTL/stratosphere extension
    (150 hPa: log-p extrapolation of the 300->200 lapse; 100/50 hPa:
    isothermal at T150; half-saturation mixing ratios) so the parcel ascent
    reaches its level of neutral buoyancy (without it MPI is biased ~35% low).
    """
    from tcpyPI import EPS, es_cc, pi

    ok = np.isfinite(t_c) & np.isfinite(q_kgkg)
    if ok.sum() < 4 or not np.isfinite(sst_c) or not np.isfinite(msl_hpa):
        return np.nan
    p = levs_hpa[ok].astype(float)
    t = t_c[ok].astype(float)
    r = np.clip(q_kgkg[ok] / (1.0 - q_kgkg[ok]) * 1000.0, 1e-9, None)  # g/kg
    # TTL extension (see docstring)
    t150 = t[-1] + (t[-1] - t[-2]) * np.log(150.0 / p[-1]) / np.log(p[-2] / p[-1])
    p_ext = np.array([150.0, 100.0, 50.0])
    t_ext = np.array([t150, t150, t150])
    e_pa = np.array([es_cc(float(tt)) * 100.0 for tt in t_ext])  # es_cc: hPa -> Pa
    r_ext = np.clip(EPS * e_pa / (p_ext * 100.0 - e_pa) * 1000.0 * 0.5, 1e-9, None)
    p = np.concatenate([p, p_ext])
    t = np.concatenate([t, t_ext])
    r = np.concatenate([r, r_ext])
    try:
        vmax_ms, _pmin, ifl, _to, _otl = pi(
            float(sst_c), float(msl_hpa), p, t, r,
            CKCD=0.9, V_reduc=1.0, ptop=50, miss_handle=1)
    except Exception:
        return np.nan
    if ifl not in (1, 2) or not np.isfinite(vmax_ms):
        return np.nan
    return float(vmax_ms) * KT_PER_MS


def _ohc_backward_slice(ohc_da, t0: pd.Timestamp):
    """Daily-mean OHC field whose 24 h window has closed at t0 (None if none has).

    GLORYS daily fields are means over [D 00, D+1 00), so 00/06/12/18 UTC on day
    D all use day D-1's field and there is no nearest-future fallback.  Delegates
    to ``src.baseline_common.ohc_backward_slice`` so the legacy entrypoint and the
    holdout feature builder cannot drift apart.
    """
    from wpac_warning.baseline_common import ohc_backward_slice
    return ohc_backward_slice(ohc_da, t0)


def _ohc_at(ohc_da, ohc_grid, t0: pd.Timestamp, lon0: float, lat0: float) -> float:
    """Time-varying GLORYS OHC (kJ/cm2): previous day's completed daily mean,
    0-100 km areal mean around the centre."""
    if ohc_da is None:
        return np.nan
    a = _ohc_backward_slice(ohc_da, t0)
    if a is None or a.ndim != 2 or not np.isfinite(a).any():
        return np.nan
    lon2d, lat2d = ohc_grid
    mask = _dist_km(lon2d, lat2d, lon0, lat0) <= 100.0
    if not mask.any():
        return np.nan
    return _amean(a, mask, lat2d)


def _sdir_c(frame: np.ndarray, lat0: float) -> float:
    """Std dev (degC) of TBB within 0-200 km of the frame centre (SHIPS SDIR)."""
    f = np.asarray(frame, dtype=np.float64)
    n = f.shape[0]
    ii = (np.arange(n) - FRAME_CENTER_IDX) * FRAME_KM_PER_PX
    jj = (np.arange(n) - FRAME_CENTER_IDX) * FRAME_KM_PER_PX * np.cos(np.radians(lat0))
    d = np.sqrt(ii[:, None] ** 2 + jj[None, :] ** 2)
    vals = f[d <= 200.0]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return np.nan
    return float(np.std(vals, ddof=0))


def compute_storm(sid: str, rows: list, src: dict | None = None) -> list:
    """Worker: all SHIPS-12 field predictors for one storm's panel rows.

    rows: list of dicts with keys pos (panel-row index), ISO_TIME, LAT, LON.
    src:  {era5_dir, ohc_dir, frames_dir} overrides (Windows spawn re-imports
          this module at its defaults, so paths must travel with the task).
    Returns list of dicts (one per row) with the computed predictors.

    Every required raw input is checked before any row is computed, and a
    missing/unreadable ERA5, GLORYS or frame file -- or a row-level extraction
    error -- raises for this SID instead of returning NaN rows.  The caller
    aborts the whole rebuild on the first such failure.
    """
    import xarray as xr

    src = src or {}
    era5_dir = Path(src.get("era5_dir") or ERA5_DIR)
    ohc_dir = Path(src.get("ohc_dir") or OHC_DIR)
    frames_dir = Path(src.get("frames_dir") or FRAMES_DIR)
    out = []
    pl_path = era5_dir / sid / f"{sid}_pl.nc"
    sl_path = era5_dir / sid / f"{sid}_sl_instant.nc"
    ohc_path = ohc_dir / sid / f"{sid}_ohc.nc"
    f_arr = frame_array_path(frames_dir, sid)
    f_time = frame_time_path(frames_dir, sid)
    required = {"ERA5 pressure-level": pl_path, "ERA5 surface": sl_path,
                "GLORYS OHC": ohc_path, "HURSAT frames": f_arr}
    if f_time is not None:
        required["HURSAT time axis"] = f_time
    missing = [f"{k} ({p})" for k, p in required.items() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{sid}: missing required raw input(s): " + "; ".join(missing))

    ds = xr.open_dataset(pl_path)
    ds_sl = xr.open_dataset(sl_path)
    ohc_ds = xr.open_dataset(ohc_path)
    if "OHC" not in ohc_ds:
        ohc_ds.close()
        raise RuntimeError(f"{sid}: {ohc_path} has no OHC variable")
    ohc_da = ohc_ds["OHC"]
    ohc_grid = np.meshgrid(ohc_da.longitude.values, ohc_da.latitude.values)
    fr_arr = np.load(f_arr, mmap_mode="r")
    fr_iso = np.load(f_time)

    lon = ds.longitude.values
    lat = ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    # The surface archive is a separate CDS request and may span a different
    # lat/lon box than the pressure-level one (e.g. 2003098N03159: 217 vs 209
    # latitudes).  Keep SST/MSLP on the grid they were downloaded on instead of
    # assuming the two meshes are identical: the 0-200 km core mask for them is
    # built from the surface coordinates, and no cell is interpolated, clamped
    # or dropped.
    sl_lon2d, sl_lat2d = np.meshgrid(ds_sl.longitude.values,
                                     ds_sl.latitude.values)
    levs = ds.pressure_level.values.astype(float)

    for r in rows:
        rec = dict(r)
        rec["SID"] = str(sid)
        try:
            lat0, lon0 = float(r["LAT"]), float(r["LON"])
            t0 = pd.Timestamp(r["ISO_TIME"]).tz_convert(None)
            d = ds.sel(valid_time=t0, method="nearest", tolerance=pd.Timedelta("3h"))
            s = ds_sl.sel(valid_time=t0, method="nearest",
                          tolerance=pd.Timedelta("3h"))
            dist = _dist_km(lon2d, lat2d, lon0, lat0)
            m_env = (dist >= 200.0) & (dist <= 800.0)
            m_core = dist <= 200.0
            m_z850 = dist <= 1000.0
            m_core_sl = _dist_km(sl_lon2d, sl_lat2d, lon0, lat0) <= 200.0

            def lev(name, L):
                return np.asarray(d[name].sel(pressure_level=L).values, dtype=np.float64).squeeze()

            u850, v850 = lev("u", 850), lev("v", 850)
            u200, v200 = lev("u", 200), lev("v", 200)
            t200, t300 = lev("t", 200), lev("t", 300)
            r1000 = lev("r", 1000)

            du = _amean(u200, m_env, lat2d) - _amean(u850, m_env, lat2d)
            dv = _amean(v200, m_env, lat2d) - _amean(v850, m_env, lat2d)
            rec["SHRD"] = float(np.hypot(du, dv)) * KT_PER_MS
            rec["T200"] = _amean(t200, m_env, lat2d) - 273.15
            t250_k = np.exp((np.log(200.0) + np.log(300.0)) / 2)  # 244.9 hPa ~ 250
            rec["T250"] = (_amean(t200, m_env, lat2d) +
                           (_amean(t300, m_env, lat2d) - _amean(t200, m_env, lat2d)) *
                           np.log(t250_k / 200.0) / np.log(300.0 / 200.0)) - 273.15
            rec["T150"] = (_amean(t200, m_env, lat2d) +
                           (_amean(t200, m_env, lat2d) - _amean(t300, m_env, lat2d)) *
                           np.log(150.0 / 200.0) / np.log(200.0 / 300.0)) - 273.15
            rec["R000"] = _amean(r1000, m_env, lat2d)
            rec["Z850"] = _vort850_1e7(u850, v850, lat, m_z850, lat2d)

            sst_c = _amean(np.asarray(s.sst.values, dtype=np.float64).squeeze(),
                           m_core_sl, sl_lat2d) - 273.15
            msl_hpa = _amean(np.asarray(s.msl.values, dtype=np.float64).squeeze(),
                             m_core_sl, sl_lat2d) / 100.0
            t_prof = np.array([_amean(lev("t", L), m_core, lat2d) - 273.15 for L in levs])
            q_prof = np.array([_amean(lev("q", L), m_core, lat2d) for L in levs])
            rec["MPI"] = _mpi_kt(t_prof, q_prof, levs, sst_c, msl_hpa)
            rec["OHCD"] = _ohc_at(ohc_da, ohc_grid, pd.Timestamp(r["ISO_TIME"]), lon0, lat0)

            key = np.datetime64(pd.Timestamp(r["ISO_TIME"]).tz_convert(None), "s")
            j = int(np.searchsorted(fr_iso, key))
            if j < len(fr_iso) and fr_iso[j] == key:
                rec["SDIR"] = _sdir_c(fr_arr[j], lat0)
            else:
                rec["SDIR"] = np.nan
        except Exception as exc:
            raise RuntimeError(
                f"{sid} {r.get('ISO_TIME')}: predictor extraction failed "
                f"({type(exc).__name__}: {exc})") from exc
        out.append(rec)

    ds.close()
    ds_sl.close()
    ohc_ds.close()
    return out


# ---------------------------------------------------------------------------
# COHC: monthly OHC climatology pooled over FIT-window GLORYS fields only
# ---------------------------------------------------------------------------

def build_cohc_clim() -> dict:
    """Monthly climatology of GLORYS OHC on a 0.5-deg WNP grid.

    Delegates to ``src.baseline_common.build_cohc_clim`` so this entrypoint uses
    the same FIT-only, per-time-step window (2000-01-01 <= t < 2018-01-01) and
    the same content-SHA provenance as the holdout path; the previous local copy
    pooled every year present on disk (2000-2024), which leaked test-period OHC
    into the "climatology" of the WRPS1-spec COHC variant.
    """
    from wpac_warning.baseline_common import build_cohc_clim as _build_common
    logger.info("building COHC monthly climatology from per-storm GLORYS OHC files "
                "(fit years only)...")
    return _build_common((2000, 2017), OHC_DIR, COHC_CLIM_NPZ)


def cohc_lookup(clim: dict, lat0: float, lon0: float, month: int) -> float:
    """0-100 km areal mean of the monthly climatology around (lat0, lon0)."""
    step = clim["lat"][1] - clim["lat"][0]
    ilat = int(round((lat0 - clim["lat"][0]) / step))
    ilon = int(round((lon0 - clim["lon"][0]) / step))
    r = int(np.ceil(100.0 / KM_PER_DEG / step)) + 1
    a = clim["clim"][month - 1]
    lo_i, hi_i = max(0, ilat - r), min(a.shape[0], ilat + r + 1)
    lo_j, hi_j = max(0, ilon - r), min(a.shape[1], ilon + r + 1)
    lat2d, lon2d = np.meshgrid(clim["lat"][lo_i:hi_i], clim["lon"][lo_j:hi_j], indexing="ij")
    d = _dist_km(lon2d, lat2d, lon0, lat0)
    sub = a[lo_i:hi_i, lo_j:hi_j]
    vals = sub[d <= 100.0]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else np.nan


# ---------------------------------------------------------------------------
# predictor table assembly
# ---------------------------------------------------------------------------

def _as_config(config, config_path):
    """(Config, config path) from a path / Config / dict, path mandatory."""
    from wpac_warning.utils_cfg import Config, load_cfg
    if config is None:
        raise ValueError(
            "build_predictors requires the holdout config (and its path) so the "
            "source manifest can pin protocol, population and config SHA")
    if isinstance(config, (str, Path)):
        path = Path(config)
        return load_cfg(path), path
    cfg = config if isinstance(config, Config) else Config(dict(config))
    if config_path is None:
        raise ValueError("build_predictors requires config_path so the source "
                         "manifest can pin the config SHA")
    return cfg, Path(config_path)


def build_predictors(workers: int, panel_parquet: Path | None = None,
                     frames_index: Path | None = None,
                     pred_cache: Path | None = None,
                     src: dict | None = None,
                     config=None, config_path: Path | None = None,
                     manifest_path: Path | None = None,
                     qc_manifest: Path | None = None,
                     allow_raw_frames: bool = False,
                     verify_raw: bool = False,
                     raw_sha_cache: Path | None = None) -> pd.DataFrame:
    """All-system SHIPS-12 row-level source table (never the 375-storm sample).

    Reads the all-system panel, keeps panel rows that have a paired frame in the
    configured corpus, and recomputes the 12 predictors from the raw ERA5/GLORYS
    archives.

    The frame corpus is the QC v1 slice declared by the holdout config
    (``data.frames_dir`` = ``data/global_hursat/frames_qc``), audited by
    ``data.qc_manifest``.  The raw download corpus (``…/global_hursat/frames``)
    is refused unless ``allow_raw_frames=True``, which produces an explicitly
    audit-only table that ``require_ships12_source_manifest`` will never accept
    for a feature cache.

    Fail-closed contract: all required raw files (ERA5 pl/surface, GLORYS OHC,
    HURCAT frames) are checked up front **for the computed rows only** -- rows
    excluded by the ``frame_availability_v1`` manifest are not read and cannot
    fail the build.  The build aborts on the first computed storm whose inputs
    are missing/unreadable or whose row extraction raises, and the output
    SID/``row_uid`` set is verified against ``eligible panel - exclusions``
    before anything is written.  Every computed storm must succeed (725/725 in
    production), so a silently all-NaN predictor column or an MPI-length
    mismatch can no longer reach the fit feature cache.

    A mandatory build manifest (``<pred_cache>.manifest.json``) records
    protocol/population, panel/frames-index/config/build-code SHA-256, every
    output row key, per-SID success/failure statistics, the path+content SHA of
    every raw input, the QC v1 corpus binding (``frames_qc_sha256`` /
    ``qc_manifest_sha256``, rule/version, ``stores_sha256``, per-SID frame store
    and frames-index hashes), the OHCD algorithm/source window and the COHC
    derivation policy.  ``verify_raw=True`` re-hashes the raw inputs instead of
    trusting the size+mtime hash cache.
    """
    from wpac_warning.baseline_common import (OHCD_ALGORITHM, OHCD_ORIGIN_RECOMPUTED,
                                     OHCD_SELECTION_RULE, QC_FRAMES_REL,
                                     SHIPS_SOURCE_KIND, availability_record,
                                     eligible_panel_rows,
                                     frame_availability_binding,
                                     frame_stores_sha256,
                                     holdout_frames_binding,
                                     load_qc_binding,
                                     sha256_files_cached,
                                     ships12_raw_input_paths)
    from wpac_warning.protocol import (SplitProtocol, is_raw_hursat_frames,
                              portable_path, sha256_array, sha256_file,
                              sha256_json)

    cfg, config_path = _as_config(config, config_path)
    proto = SplitProtocol.from_cfg(cfg)
    proto.validate()
    population = proto.require_alltc()

    panel_parquet = Path(panel_parquet or PANEL_PARQUET)
    frames_index = Path(frames_index or FRAMES_INDEX)
    pred_cache = Path(pred_cache or PRED_CACHE)
    manifest_path = Path(manifest_path or pred_cache.with_suffix(".manifest.json"))
    src = src or {}
    era5_dir = Path(src.get("era5_dir") or ERA5_DIR)
    ohc_dir = Path(src.get("ohc_dir") or OHC_DIR)

    # QC v1 corpus: from the config unless explicitly overridden.  The raw
    # download corpus is audit-only and never a production input.
    frames_dir = Path(src["frames_dir"]) if src.get("frames_dir") else None
    qc_path = Path(qc_manifest) if qc_manifest else None
    raw_frames = frames_dir is not None and is_raw_hursat_frames(frames_dir)
    if frames_dir is None or (qc_path is None and not raw_frames):
        binding = holdout_frames_binding(cfg)
        frames_dir = frames_dir or binding["frames_dir"]
        qc_path = qc_path or binding["qc_manifest"]
    raw_frames = is_raw_hursat_frames(frames_dir)
    qc_info = None
    if raw_frames:
        if not allow_raw_frames:
            raise ValueError(
                f"frames_dir={frames_dir} is the RAW HURSAT download corpus; "
                f"the production SHIPS source table is built from the QC v1 "
                f"corpus declared by data.frames_dir ({QC_FRAMES_REL}).  Pass "
                f"--allow-raw-frames only for an explicit audit build (its "
                f"manifest is rejected by the feature cache)")
        if qc_path is not None and qc_manifest:
            raise ValueError(
                "--qc-manifest cannot be combined with --allow-raw-frames: the "
                "raw corpus has no QC v1 audit")
    else:
        if qc_path is None:
            raise ValueError(
                "no QC manifest: set data.qc_manifest in the holdout config or "
                "pass --qc-manifest; the source table must be pinned to the QC "
                "v1 frame corpus")
        qc_info = load_qc_binding(qc_path, frames_dir)
        if qc_info["sids"] is None or not qc_info["sids"]:
            raise ValueError(f"{qc_path}: QC manifest lists no stores")

    panel = pd.read_parquet(panel_parquet)
    panel["ISO_TIME"] = pd.to_datetime(panel["ISO_TIME"], utc=True)
    if panel.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError(f"{panel_parquet}: duplicate (SID, ISO_TIME) rows")
    panel = panel.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    index = pd.read_parquet(frames_index)
    index["ISO_TIME"] = pd.to_datetime(index["ISO_TIME"], utc=True)
    index_keys = (index["SID"].astype(str) + "|"
                  + index["ISO_TIME"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).tolist()

    # one row universe for DL and SHIPS: eligible panel - availability exclusions
    # (prefer the config panel the dataset reads; fall back to --panel)
    eligible = eligible_panel_rows(cfg)
    if eligible is None:
        eligible = eligible_panel_rows(cfg, panel)
    avail = frame_availability_binding(cfg, eligible=eligible,
                                       index_keys=index_keys, require=True)
    avail_rec = availability_record(avail)
    if eligible is not None:
        elig_keys = set(eligible["row_uid"].astype(str))
        base = panel.loc[(panel["SID"].astype(str) + "|"
                          + panel["ISO_TIME"].dt.strftime(
                              "%Y-%m-%dT%H:%M:%SZ")).isin(elig_keys)]
        expected_set = elig_keys - set(avail["excluded"])
    else:
        base = panel
        expected_set = None
        logger.warning("panel lacks the label-time/storm-year columns; the "
                       "eligible row set cannot be recomputed (availability "
                       "exclusions are still validated against the index)")
    df = base.merge(index[["SID", "ISO_TIME"]], on=["SID", "ISO_TIME"],
                    how="inner")
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no panel rows paired with the frames index")
    got_keys = set(df["SID"].astype(str) + "|"
                   + df["ISO_TIME"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    if expected_set is not None and got_keys != expected_set:
        missing = sorted(expected_set - got_keys)
        extra = sorted(got_keys - expected_set)
        raise RuntimeError(
            f"source-build row set differs from eligible panel - availability "
            f"exclusions ({len(missing)} missing, {len(extra)} extra; e.g. "
            f"missing {missing[:5]}, extra {extra[:5]}); refusing to build a "
            f"table that is not the DL row set")
    ri = df["DV24"].ge(30).astype(int)
    assert (df["RI"] == ri).all(), "RI column disagrees with DV24>=30"
    expected_uid = (df["SID"].astype(str) + "|"
                    + df["ISO_TIME"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).tolist()
    sids = [str(s) for s in df["SID"].unique()]
    logger.info(f"sample rows {len(df)} | storms {len(sids)} | "
                f"excluded rows {avail['n_excluded']} | "
                f"population {population} | RI base rate {ri.mean():.4f}")
    if qc_info is not None:
        missing_qc = sorted(set(sids) - set(qc_info["sids"]))
        if missing_qc:
            raise ValueError(
                f"{len(missing_qc)} panel SID(s) are absent from the QC v1 "
                f"manifest ({qc_path}): e.g. {missing_qc[:5]}; the frames index "
                f"and the QC corpus do not describe the same systems")
        logger.info(f"QC v1 corpus {qc_info['out_frames_dir']} "
                    f"({qc_info['n_stores']} stores, rule "
                    f"{qc_info['qc_rule_version']}, "
                    f"sha {qc_info['sha256'][:12]})")

    worker_src = {**src, "era5_dir": era5_dir, "ohc_dir": ohc_dir,
                  "frames_dir": frames_dir}

    # every raw input must exist before any compute starts (725/725 expected)
    raw_paths = {sid: ships12_raw_input_paths(sid, era5_dir, ohc_dir, frames_dir)
                 for sid in sids}
    missing = [f"{sid}: {group} {p}"
               for sid, paths in raw_paths.items()
               for group, p in paths.items() if not Path(p).is_file()]
    if missing:
        head = "; ".join(missing[:15])
        raise FileNotFoundError(
            f"{len(missing)} required raw input(s) missing for the all-system "
            f"source table; the production rebuild must be 725/725. First: "
            f"{head}")

    tasks = []
    for sid, g in df.groupby("SID", sort=False):
        rows = [{"pos": int(i), "ISO_TIME": r.ISO_TIME, "LAT": r.LAT, "LON": r.LON}
                for i, r in g.iterrows()]
        tasks.append((str(sid), rows))
    task_rows = {sid: rows for sid, rows in tasks}

    t0 = time.time()
    results: list = []
    per_sid: dict = {}
    failures: list = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(compute_storm, sid, rows, worker_src): sid
                for sid, rows in tasks}
        done = 0
        for fut in as_completed(futs):
            sid = futs[fut]
            done += 1
            try:
                res = fut.result()
            except Exception as exc:
                # fail fast: no partial table, no hours-long silent NaN run
                failures.append({"sid": sid,
                                 "error": f"{type(exc).__name__}: {exc}"})
                logger.error(f"{sid} FAILED, aborting rebuild: {exc}")
                for f in futs:
                    f.cancel()
                break
            if len(res) != len(task_rows[sid]):
                failures.append({"sid": sid,
                                 "error": f"returned {len(res)} rows, expected "
                                          f"{len(task_rows[sid])}"})
                for f in futs:
                    f.cancel()
                break
            results.extend(res)
            per_sid[sid] = {"n_rows": int(len(res)), "status": "ok"}
            if done % 25 == 0:
                el = time.time() - t0
                logger.info(f"  {done}/{len(tasks)} storms, {el:.0f}s elapsed, "
                            f"eta {el / done * (len(tasks) - done):.0f}s")
    if failures:
        raise RuntimeError(
            f"all-system SHIPS source build aborted: {len(failures)} storm "
            f"failure(s) out of {len(tasks)} after {len(per_sid)} successes; "
            + "; ".join(f"{f['sid']}: {f['error']}" for f in failures[:10]))

    recs = pd.DataFrame(results)
    if recs.empty:
        raise RuntimeError("SHIPS source build produced no rows")
    recs = recs.set_index("pos")
    if (not recs.index.is_unique or len(recs) != len(df)
            or set(recs.index.tolist()) != set(df.index.tolist())):
        raise RuntimeError(
            f"SHIPS source build returned {len(recs)} rows for {len(df)} "
            f"expected panel rows; row positions are missing/duplicated")
    recs = recs.loc[df.index]
    got_uid = (recs["SID"].astype(str) + "|"
               + pd.to_datetime(recs["ISO_TIME"], utc=True).dt.strftime(
                   "%Y-%m-%dT%H:%M:%SZ")).tolist()
    if got_uid != expected_uid:
        bad = next(i for i, (a, b) in enumerate(zip(got_uid, expected_uid))
                   if a != b)
        raise RuntimeError(
            f"SHIPS source build row keys do not match the expected panel: "
            f"position {bad} is {got_uid[bad]!r}, expected {expected_uid[bad]!r}")
    if "_err" in recs.columns and recs["_err"].notna().any():
        raise RuntimeError(f"{int(recs['_err'].notna().sum())} rows carry a "
                           f"per-row extraction error")
    counts = recs["SID"].astype(str).value_counts().to_dict()
    for sid in sids:
        if int(counts.get(sid, 0)) != per_sid[sid]["n_rows"]:
            raise RuntimeError(f"{sid}: {int(counts.get(sid, 0))} rows in the "
                               f"result table, {per_sid[sid]['n_rows']} reported")

    df["MPI"] = recs["MPI"].to_numpy(dtype=float)
    df["POT"] = df["MPI"] - df["VMAX"]
    for c in ["SHRD", "T150", "T200", "T250", "R000", "Z850", "OHCD", "SDIR"]:
        df[c + "_S12"] = recs[c].to_numpy(dtype=float)
    df["INCV"] = df["PER"]          # panel PER = 6h intensity change on native 3h track
    df["DTL"] = df["LAND_DIST"]     # km, GSHHG coastline distance
    for col in ("MPI", "OHCD_S12", "SHRD_S12", "T150_S12", "T200_S12",
                "T250_S12", "R000_S12", "Z850_S12", "SDIR_S12"):
        n_finite = int(np.isfinite(df[col].to_numpy(dtype=float)).sum())
        if n_finite == 0:
            raise RuntimeError(
                f"predictor {col} is all-NaN over {len(df)} rows; refusing to "
                f"write a silently empty all-system source table")
    keep = ["SID", "ISO_TIME", "VMAX", "INCV", "MPI", "POT", "OHCD_S12", "SHRD_S12",
            "T150_S12", "T200_S12", "T250_S12", "R000_S12", "SDIR_S12", "Z850_S12", "DTL"]
    out = df[keep].rename(columns={c + "_S12": c for c in
                                   ["OHCD", "SHRD", "T150", "T200", "T250", "R000", "SDIR", "Z850"]})
    pred_cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pred_cache, index=False)

    raw_inputs = {}
    for group in ("era5_pressure_levels", "era5_surface", "glorys_ohc",
                  "frames", "frames_time_axis"):
        hashed = sha256_files_cached(
            [raw_paths[sid][group] for sid in sids],
            raw_sha_cache or RAW_SHA_CACHE, verify=verify_raw)
        entries = []
        for sid in sids:
            entry = hashed.get(portable_path(raw_paths[sid][group], P2_ROOT))
            if entry is None:
                raise RuntimeError(f"{sid}: no content hash for {group} "
                                   f"{raw_paths[sid][group]}")
            entries.append({"sid": sid, **entry})
        raw_inputs[group] = entries

    def rel(p) -> str:
        return portable_path(p, P2_ROOT)

    out_uid = sorted((out["SID"].astype(str) + "|"
                      + pd.to_datetime(out["ISO_TIME"], utc=True).dt.strftime(
                          "%Y-%m-%dT%H:%M:%SZ")).tolist())
    if out_uid != sorted(expected_uid):
        raise RuntimeError("written source table row keys differ from the "
                           "verified panel row keys")
    table_by_split = {}
    if eligible is not None:
        split_of = dict(zip(eligible["row_uid"].astype(str),
                            eligible["split"].astype(str)))
        for key in out_uid:
            split = split_of.get(key)
            if split is not None:
                table_by_split[split] = table_by_split.get(split, 0) + 1
    manifest = {
        "kind": SHIPS_SOURCE_KIND,
        "generator": "scripts/baseline_ships12.py --build-predictors",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "parquet": {"path": rel(pred_cache), "rows": int(len(out)),
                    "sids": int(len(sids))},
        "output_sha256": sha256_file(pred_cache),
        "row_key_format": "SID|%Y-%m-%dT%H:%M:%SZ",
        "row_key_sha256": sha256_array(out_uid),
        "row_key_count": int(len(out_uid)),
        "sid_sha256": sha256_array(sorted(sids)),
        "population": population,
        "n_systems": proto.n_systems,
        "protocol": proto.as_dict(),
        "inputs": {
            "panel": {"path": rel(panel_parquet), "rows": int(len(panel)),
                      "sha256": sha256_file(panel_parquet)},
            "frames_index": {"path": rel(frames_index), "rows": int(len(index)),
                             "sha256": sha256_file(frames_index)},
            "config": {"path": rel(config_path),
                       "sha256": sha256_file(config_path)},
            "build_code": {
                "scripts/baseline_ships12.py": sha256_file(Path(__file__)),
                "src/baseline_common.py": sha256_file(
                    P2_ROOT / "src" / "baseline_common.py"),
            },
        },
        "raw_inputs": raw_inputs,
        "raw_inputs_sha256": sha256_json(
            {g: [{k: e[k] for k in ("sid", "path", "sha256")} for e in v]
             for g, v in raw_inputs.items()}),
        "raw_input_hash": {
            "method": "sha256 of file bytes; the manifest stores every path",
            "n_files": int(sum(len(v) for v in raw_inputs.values())),
            "hash_cache": rel(raw_sha_cache or RAW_SHA_CACHE),
            "rehashed_at_build": bool(verify_raw),
            "verify": "src.baseline_common.require_ships12_source_manifest("
                      "verify_raw=True)",
        },
        "storm_stats": {"n_storms": int(len(sids)), "n_ok": int(len(per_sid)),
                        "n_failed": 0, "failures": [], "per_sid": per_sid},
        "availability": avail_rec,
        "row_counts": {
            "eligible": (int(len(eligible)) if eligible is not None else None),
            "excluded": int(avail["n_excluded"]),
            "table": int(len(out)),
            "table_by_split": table_by_split,
        },
        "frames_qc": (
            {
                "production": True,
                "qc_rule_version": qc_info["qc_rule_version"],
                "qc_manifest": rel(qc_path),
                "qc_manifest_sha256": qc_info["sha256"],
                "frames_qc_sha256": qc_info["sha256"],
                "out_frames_dir": qc_info["out_frames_dir"],
                "source_frames_dir": qc_info["source_frames_dir"],
                "n_stores": int(qc_info["n_stores"]),
                "stores_sha256": qc_info["stores_sha256"],
                "frame_stores_sha256": frame_stores_sha256(raw_inputs, sids),
                "n_frame_stores": int(len(sids)),
                "frames_index_sha256": sha256_file(frames_index),
            }
            if qc_info is not None else
            {
                "production": False,
                "source": "raw_audit",
                "reason": "built with --allow-raw-frames from the raw HURCAT "
                          "download corpus; audit only, never a feature-cache "
                          "input",
                "out_frames_dir": rel(frames_dir),
            }
        ),
        "frames_qc_sha256": (qc_info["sha256"] if qc_info is not None else None),
        "qc_manifest_sha256": (qc_info["sha256"] if qc_info is not None else None),
        "ohcd": {
            "origin": OHCD_ORIGIN_RECOMPUTED,
            "algorithm": OHCD_ALGORITHM,
            "selection": OHCD_SELECTION_RULE,
            "window_hours": 24,
            "n_rows": int(len(out)),
            "n_rows_finite": int(np.isfinite(out["OHCD"].to_numpy(dtype=float)).sum()),
        },
        "cohc": {
            "in_source_table": False,
            "fit_only": True,
            "derivation": "monthly 0.5-deg OHC climatology pooled over the "
                          "FIT-window GLORYS source times only, applied per "
                          "time step at feature-build time "
                          "(src.baseline_common.build_cohc_clim)",
            "source_time_window": [f"{proto.fit_years[0]}-01-01T00:00:00",
                                   f"{proto.fit_years[1] + 1}-01-01T00:00:00"],
            "content_hash_required": True,
        },
        "predictors": [c for c in out.columns if c not in ("SID", "ISO_TIME")],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    logger.info(f"source table -> {pred_cache} ({time.time() - t0:.0f}s) | "
                f"manifest -> {manifest_path}")
    return out


def add_cohc(pred: pd.DataFrame) -> pd.DataFrame:
    clim = build_cohc_clim()
    panel = pd.read_parquet(PANEL_PARQUET, columns=["SID", "ISO_TIME", "LAT", "LON"])
    panel["ISO_TIME"] = pd.to_datetime(panel["ISO_TIME"], utc=True)
    m = pred.merge(panel, on=["SID", "ISO_TIME"], how="left", validate="1:1")
    pred = pred.copy()
    pred["COHC"] = [cohc_lookup(clim, la, lo, t.month)
                    for la, lo, t in zip(m["LAT"], m["LON"], m["ISO_TIME"])]
    return pred


# ---------------------------------------------------------------------------
# LOYO evaluation (identical protocol to scripts/baseline_lr.py)
# ---------------------------------------------------------------------------

def run_loyo(pred: pd.DataFrame, ohc_var: str, model: str) -> pd.DataFrame:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = pred.copy()
    feats = [ohc_var if f in ("OHCD", "COHC") else f for f in PREDICTORS_MAIN]
    sid_year = df.groupby("SID")["ISO_TIME"].min().dt.year
    years = df["SID"].map(sid_year).values
    y_all = df["RI"].astype(int).values
    X_raw = df[feats].values.astype(float)

    rows = []
    for held in LOYO_YEARS:
        va = years == held
        inner = np.isin(years, INNER_YEARS)
        tr = ~va & ~inner & (years <= 2018)
        med = np.nanmedian(X_raw[tr], axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X_raw), X_raw, np.tile(med, (len(X_raw), 1)))
        sc = StandardScaler().fit(X[tr])
        X = sc.transform(X)
        if model == "lda":
            clf = LinearDiscriminantAnalysis()
        else:
            clf = LogisticRegression(class_weight="balanced", max_iter=3000)
        clf.fit(X[tr], y_all[tr])
        thr = pick_threshold(clf.predict_proba(X[inner])[:, 1], y_all[inner])
        p = clf.predict_proba(X[va])[:, 1]
        y = y_all[va]
        pred_bin = p >= thr
        pod = float(((pred_bin == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        far = float(((pred_bin == 1) & (y == 0)).sum() / max((pred_bin == 1).sum(), 1))
        roc, pr = safe_auc(p, y)
        rows.append({"held_year": held, "POD": pod, "FAR": far,
                     "BSS": bss_score(p, y), "ROC_AUC": roc, "PR_AUC": pr,
                     "threshold": thr})
        logger.info(f"[{ohc_var}/{model} {held}] POD={pod:.3f} FAR={far:.3f} "
                    f"BSS={rows[-1]['BSS']:.3f} ROC={roc:.3f} PR={pr:.3f}")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="holdout config; any non-holdout config is refused "
                         "unless --legacy is given (default: "
                         "configs/phase2/holdout_masked_v1.yaml)")
    ap.add_argument("--legacy", action="store_true",
                    help="explicitly evaluate the archived pre-holdout LOYO "
                         "protocol (never the holdout)")
    ap.add_argument("--build-predictors", action="store_true",
                    help="build the all-system row-level source table + its "
                         "mandatory build manifest from the raw ERA5/GLORYS "
                         "archives, then exit")
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild the source table (same as --build-predictors, "
                         "then continue to the fit feature tables)")
    ap.add_argument("--verify-raw", action="store_true",
                    help="re-hash every raw ERA5/GLORYS input instead of using "
                         "the size+mtime hash cache")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-root", default=str(P2_ROOT / "data" / "runs_holdout" /
                                             "baselines"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", choices=["lda", "lr"], default="lda")
    ap.add_argument("--ohc-var", choices=["OHCD", "COHC", "both"],
                    default="OHCD",
                    help="OHCD = main arm; COHC = separate sensitivity")
    ap.add_argument("--features-dir",
                    default=str(P2_ROOT / "data" / "holdout_masked_v1"),
                    help="where the versioned fit-only predictor tables live")
    ap.add_argument("--predictor-parquet",
                    default=str(DEFAULT_PRED_CACHE),
                    help="ALL-SYSTEM SHIPS row-level source table, built from "
                         "the raw ERA5/GLORYS archives over all WNP systems "
                         "(default: data/ships12_predictors_alltc_0024.parquet)")
    ap.add_argument("--source-manifest",
                    default=None,
                    help="build manifest path (default: "
                         "<predictor-parquet>.manifest.json)")
    ap.add_argument("--raw-sha-cache", default=str(RAW_SHA_CACHE),
                    help="raw input sha256 cache (verifiable; --verify-raw "
                         "ignores it)")
    ap.add_argument("--panel",
                    default=str(PANEL_PARQUET),
                    help="phase-1 all-system processed panel for --recompute")
    ap.add_argument("--frames-index", default=None,
                    help="frame index parquet (default: config data.frames_index)")
    ap.add_argument("--frames-dir", default=None,
                    help="frame corpus (default: config data.frames_dir, the QC "
                         "v1 corpus data/global_hursat/frames_qc); the raw "
                         "corpus needs --allow-raw-frames")
    ap.add_argument("--qc-manifest", default=None,
                    help="QC v1 audit manifest (default: config data.qc_manifest)")
    ap.add_argument("--allow-raw-frames", action="store_true",
                    help="explicit AUDIT build from the raw HURCAT corpus; the "
                         "resulting manifest is rejected by the feature cache")
    ap.add_argument("--era5-dir", default=str(ERA5_DIR))
    ap.add_argument("--ohc-dir", default=str(OHC_DIR),
                    help="daily GLORYS fields; OHCD is recomputed BACKWARD from "
                         "these so the archived table's column is never reused")
    args = ap.parse_args()

    from wpac_warning.baseline_common import load_baseline_cfg
    try:
        cfg, is_holdout = load_baseline_cfg(args.config, legacy=args.legacy)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        return 2
    if is_holdout:
        return _run_holdout(args, cfg)

    logger.warning("LEGACY pre-holdout evaluation requested explicitly "
                   "(--legacy); this cannot touch the holdout protocol")
    if args.build_predictors or args.recompute:
        logger.error("the all-system source table is built under the holdout "
                     "config only; drop --legacy/--recompute")
        return 2

    pred_cache = Path(args.predictor_parquet)
    if args.eval_only or (pred_cache.is_file() and not args.recompute):
        pred = pd.read_parquet(pred_cache)
        pred["ISO_TIME"] = pd.to_datetime(pred["ISO_TIME"], utc=True)
        logger.info(f"loaded predictor cache {pred_cache} ({len(pred)} rows)")
    else:
        raise SystemExit("legacy predictor table missing: pass an existing "
                         "--predictor-parquet (legacy recompute is disabled)")
    # RI label for eval (row-identical to DL arm)
    panel = pd.read_parquet(Path(args.panel), columns=["SID", "ISO_TIME", "RI"])
    panel["ISO_TIME"] = pd.to_datetime(panel["ISO_TIME"], utc=True)
    pred = pred.merge(panel, on=["SID", "ISO_TIME"], how="left", validate="1:1")
    assert pred["RI"].notna().all()

    if "COHC" not in pred.columns or args.recompute:
        pred = add_cohc(pred)
        pred.drop(columns=["RI"]).to_parquet(pred_cache, index=False)

    for ohc_var, tag in [("OHCD", ""), ("COHC", "_cohc")]:
        for model in ("lda", "lr"):
            out = run_loyo(pred, ohc_var, model)
            path = P2_ROOT / "data" / f"loyo_results_baseline_ships12_{model}{tag}.csv"
            out.to_csv(path, index=False)
            logger.info(f"{ohc_var}/{model}: mean BSS={out.BSS.mean():.3f} "
                        f"ROC={out.ROC_AUC.mean():.3f} PR={out.PR_AUC.mean():.3f} -> {path}")
    return 0


def _run_holdout(args, cfg) -> int:
    """HOLDOUT_MASKED_V1: source table (+ manifest) then fit-only feature tables."""
    from wpac_warning.baseline_common import (build_ships12_features,
                                     holdout_frames_binding, holdout_path,
                                     make_pipeline,
                                     require_ships12_source_manifest,
                                     run_baseline_with_pipeline,
                                     ships12_source_manifest_for)

    src = Path(args.predictor_parquet)
    man_path = (Path(args.source_manifest) if args.source_manifest
                else ships12_source_manifest_for(src))
    frames_index = (Path(args.frames_index) if args.frames_index
                    else holdout_path(cfg.get("data.frames_index") or FRAMES_INDEX))
    binding = ({"frames_dir": Path(args.frames_dir),
                "qc_manifest": Path(args.qc_manifest)}
               if args.frames_dir and args.qc_manifest
               else holdout_frames_binding(cfg))
    if args.build_predictors or args.recompute:
        if args.eval_only:
            raise SystemExit("--eval-only cannot be combined with a source-table "
                             "rebuild")
        build_predictors(
            args.workers, panel_parquet=Path(args.panel),
            frames_index=frames_index, pred_cache=src,
            src={"era5_dir": args.era5_dir, "ohc_dir": args.ohc_dir,
                 **({"frames_dir": args.frames_dir} if args.frames_dir else {})},
            config=cfg, config_path=Path(args.config), manifest_path=man_path,
            qc_manifest=(Path(args.qc_manifest) if args.qc_manifest else None),
            allow_raw_frames=args.allow_raw_frames,
            verify_raw=args.verify_raw, raw_sha_cache=Path(args.raw_sha_cache))
        if args.build_predictors:
            return 0
    if not src.is_file():
        raise FileNotFoundError(
            f"all-system SHIPS predictor table not found: {src}.  Build it "
            f"first from the raw ERA5/GLORYS archives (this is a full "
            f"725-system rebuild and writes the mandatory manifest):\n"
            f"  python scripts/01_data/baseline_ships12.py --build-predictors "
            f"--config {args.config} --predictor-parquet {src}")
    if not man_path.is_file():
        raise FileNotFoundError(
            f"{src} has no build manifest ({man_path}); a table without one is "
            f"not usable.  Rebuild it explicitly with --build-predictors "
            f"--config {args.config}")
    require_ships12_source_manifest(
        src, where=str(src), qc_manifest=binding["qc_manifest"],
        frames_dir=binding["frames_dir"])

    variants = (["OHCD", "COHC"] if args.ohc_var == "both"
                else [args.ohc_var])
    if args.run_id and len(variants) > 1:
        raise SystemExit("--run-id cannot be combined with --ohc-var both")
    for ohc in variants:
        feat_path = (Path(args.features_dir) /
                     f"ships12_predictors_fit0017_{ohc.lower()}.parquet")
        feat_path = build_ships12_features(
            cfg, src, feat_path, ohc_var=ohc,
            ohc_dir=(Path(args.ohc_dir) if ohc == "OHCD" else None))
        features = pd.read_parquet(feat_path)
        feat_manifest = feat_path.with_suffix(".manifest.json")
        run_id = (args.run_id
                  or (f"baseline_ships12_{args.model}" if ohc == "OHCD"
                      else f"baseline_ships12_{args.model}_cohc"))
        pipe = make_pipeline(args.model)
        run_dir = run_baseline_with_pipeline(
            cfg, "ships12", features, pipe, args.out_root, run_id,
            feature_spec={"source": "SHIPS-12 predictors (all-system source "
                                    "table + build manifest), fit-only COHC if "
                                    "selected",
                          "predictors": [c for c in features.columns
                                         if c != "row_uid"],
                          "ohc_var": ohc,
                          "source_parquet": str(src),
                          "source_manifest": str(man_path),
                          "source_manifest_sha256": sha256_file(man_path),
                          "predictor_parquet": str(feat_path),
                          "features_parquet": str(feat_path),
                          "features_manifest": str(feat_manifest),
                          "features_parquet_sha256": sha256_file(feat_path),
                          "features_manifest_sha256":
                              sha256_file(feat_manifest)},
            extra_manifest={"ohc_var": ohc,
                            "main_or_sensitivity": (
                                "main" if ohc == "OHCD"
                                else "sensitivity")})
        logger.info(f"holdout baseline written ({ohc}): {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
