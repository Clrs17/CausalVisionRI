"""Build the RITCF-recon sample panel and its feature cache.

Pipeline (spec §1-3):

1. **Panel** -- IBTrACS WP v04r01, 2000-2021, 6-hourly synoptic rows.  RI is
   ``CMA_WIND(t+24h) - CMA_WIND(t) >= 25.27 kt`` with the +24 h endpoint
   required to exist in the same storm.  Samples must be over ocean
   (``DIST2LAND > 0``).  The funnel is printed and stored in the manifest.
2. **Features** -- per storm, in a process pool:

   * SAT  : HURSAT-B1 v07b, frames at t, -6, -12, -18, -24 h, 224x224 TBB.
            Read from the QC v1 store (``data/global_hursat/frames_qc``), which
            is the decoded form of the same tarballs with the multi-satellite
            pick and the ``100 K <= TBB <= 350 K`` mask already applied.
            Non-finite pixels are filled with that frame's mean before the
            float16 cache write (the pre-fill NaN fraction is recorded).
   * ERA5 : u, v, pv, d at 1000/850/500/200 hPa, plus SST, five times; a
            25 deg x 25 deg window around the t0 centre, block-averaged
            4x4 (0.25 deg -> 1 deg) to 25x25.  The local archive is 6-hourly,
            so every needed time is an exact step in the file.
   * HIS  : CMA wind + pressure at the five times (pressure falls back to
            USA_PRES, counted) and all ten pairwise differences -> 30 values.
3. **Cache** -- per-storm ``cache/feat/{SID}.npz`` (transient), then monolithic
   ``sat.npy`` / ``era5.npy`` / ``his.npy`` memmaps plus ``samples.parquet``,
   ``norm.json`` (train-split statistics only) and ``manifest.json``
   (config + source hashes, row keys and their hash, per-storm audit, funnel).
   A manifest whose fingerprint still matches is reused verbatim: the rebuild
   is skipped entirely.

Row order is identical everywhere: ``samples.parquet`` is sorted by
(time, SID) and the memmap row ``i`` is exactly row ``i`` of that table.

Usage (from the release root; this builder is numpy/xarray only, so it does
not need KMP_DUPLICATE_LIB_OK):

    python -m scripts.ritcf_recon.build_dataset --smoke          # 2 storms + 2010
    python -m scripts.ritcf_recon.build_dataset --workers 8      # full 2000-2021
    python -m scripts.ritcf_recon.build_dataset --report-only    # reprint report

Outputs land in ``data/ritcf_recon/dataset{_<tag>}/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))

from scripts.ritcf_recon import config as C  # noqa: E402

PANEL_COLUMNS = [
    "SID", "SEASON", "ISO_TIME", "NATURE", "DIST2LAND", "LAT", "LON",
    "CMA_WIND", "CMA_PRES", "USA_PRES",
]

SMOKE_STORMS = ("2000141N18116", "2013306N07162")  # 16-step track, Haiyan
SMOKE_YEARS = (2010,)

SMALL_KEYS = ("row_keys", "iso_time", "lat", "lon", "wind", "pres", "pres_source",
              "dv24", "ri", "season", "split")


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------
def load_panel(seasons=C.SEASON_RANGE, ri_thr_kt=C.RI_THR_KT) -> pd.DataFrame:
    """IBTrACS WP panel with the five-time columns and the RI label attached."""
    df = pd.read_csv(
        C.IBTRACS_CSV, usecols=PANEL_COLUMNS, skiprows=[1],
        na_values=[" ", ""], low_memory=False,
    )
    df["time"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    lo, hi = seasons
    df = df[(df["SEASON"] >= lo) & (df["SEASON"] <= hi)]
    df = df[df["time"].dt.hour.isin(C.SYNOPTIC_HOURS)]
    df = df.sort_values(["SID", "time"]).reset_index(drop=True)
    assert not df.duplicated(["SID", "time"]).any(), \
        "IBTrACS panel has duplicate (SID, time) rows"
    df["wind"] = pd.to_numeric(df["CMA_WIND"], errors="coerce")
    df["pres_cma"] = pd.to_numeric(df["CMA_PRES"], errors="coerce")
    df["pres_usa"] = pd.to_numeric(df["USA_PRES"], errors="coerce")
    df["pres"] = df["pres_cma"].where(df["pres_cma"].notna(), df["pres_usa"])
    df["pres_source"] = np.where(
        df["pres_cma"].notna(), "CMA",
        np.where(df["pres_usa"].notna(), "USA", "none"))
    df["dist2land"] = pd.to_numeric(df["DIST2LAND"], errors="coerce")

    # value at t + off for every offset of the five-time window
    base = df[["SID", "time", "wind", "pres"]]
    for off in C.TIME_OFFSETS_H:
        sub = base.copy()
        sub["time"] = sub["time"] - pd.Timedelta(hours=off)
        tag = f"o{abs(off):02d}"
        sub = sub.rename(columns={"wind": f"wind_{tag}", "pres": f"pres_{tag}"})
        df = df.merge(sub, on=["SID", "time"], how="left")

    later = base[["SID", "time", "wind"]].copy()
    later["time"] = later["time"] - pd.Timedelta(hours=24)
    later = later.rename(columns={"wind": "wind_p24"})
    df = df.merge(later, on=["SID", "time"], how="left")
    df["dv24"] = df["wind_p24"] - df["wind"]
    df["ri"] = (df["dv24"] >= ri_thr_kt).astype("int8")
    return df


def split_of(season: int) -> str:
    for lo, hi, name in C.SPLITS:
        if lo <= season <= hi:
            return name
    return "out"


def select_candidates(panel: pd.DataFrame, storms, years) -> tuple[pd.DataFrame, dict]:
    """Apply the sample rules in order and return the surviving rows + funnel."""
    funnel: dict[str, int] = {}
    df = panel
    funnel["1_season_rows"] = int(len(df))
    if years is not None or storms is not None:
        # a scoped build unions the two: named storms *and* whole seasons
        scope_mask = pd.Series(False, index=df.index)
        if storms is not None:
            scope_mask |= df["SID"].isin(list(storms))
        if years is not None:
            scope_mask |= df["SEASON"].isin(list(years))
        df = df[scope_mask]
        funnel["1b_scope_subset"] = int(len(df))

    df = df[df["wind"].notna()]
    funnel["2_cma_wind"] = int(len(df))
    df = df[df["dv24"].notna()]
    funnel["3_dv24_computable"] = int(len(df))
    df = df[df["dist2land"].notna()]
    funnel["4_dist2land_present"] = int(len(df))
    df = df[df["dist2land"] > 0]
    funnel["5_over_ocean"] = int(len(df))
    df = df[df["LAT"].notna() & df["LON"].notna()]
    funnel["6_has_centre"] = int(len(df))
    return df.reset_index(drop=True), funnel


def _num(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if v == v else float("nan")


def candidate_rows(df: pd.DataFrame) -> list[list[dict]]:
    """Group candidate samples by storm into plain dicts (picklable, spawn-safe)."""
    by_storm: dict[str, list[dict]] = {}
    for rec in df.itertuples(index=False):
        times = [rec.time + pd.Timedelta(hours=off) for off in C.TIME_OFFSETS_H]
        by_storm.setdefault(rec.SID, []).append({
            "row_key": f"{rec.SID}|{rec.time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "iso_time": rec.time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season": int(rec.SEASON),
            "split": split_of(int(rec.SEASON)),
            "lat": float(rec.LAT),
            "lon": float(rec.LON),
            "wind": float(rec.wind),
            "pres": _num(rec.pres),
            "pres_source": str(rec.pres_source),
            "dv24": float(rec.dv24),
            "ri": int(rec.ri),
            "times": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times],
            "wind5": [_num(getattr(rec, f"wind_o{abs(o):02d}")) for o in C.TIME_OFFSETS_H],
            "pres5": [_num(getattr(rec, f"pres_o{abs(o):02d}")) for o in C.TIME_OFFSETS_H],
        })
    return [by_storm[sid] for sid in sorted(by_storm)]


# --------------------------------------------------------------------------
# Per-storm feature extraction (worker)
# --------------------------------------------------------------------------
def _era5_window(arr: np.ndarray, ic: int, jc: int) -> tuple[np.ndarray, bool]:
    """Block-average a 25x25 one-degree window centred on (ic, jc).

    ``arr`` is (..., nlat, nlon) on the native 0.25 deg grid with latitude
    descending.  Out-of-domain indices are clamped (edge replication) so a
    storm near the archive boundary still yields a full window; the second
    return value says whether clamping happened.
    """
    n = C.ERA5_WINDOW_CELLS
    half = n // 2
    nlat, nlon = arr.shape[-2], arr.shape[-1]
    ilat = np.clip(np.arange(ic - half, ic + half), 0, nlat - 1)
    ilon = np.clip(np.arange(jc - half, jc + half), 0, nlon - 1)
    truncated = bool((ic - half < 0) or (ic + half > nlat - 1) or
                     (jc - half < 0) or (jc + half > nlon - 1))
    win = arr[..., ilat, :][..., :, ilon]
    block = C.ERA5_WINDOW_CELLS // C.GRID_SIZE  # 4 native cells per 1-deg block
    lead = win.shape[:-2]
    win = win.reshape(*lead, C.GRID_SIZE, block, C.GRID_SIZE, block)
    return win.mean(axis=(-3, -1)).astype(np.float32), truncated


def _nearest_index(coord: np.ndarray, value: float) -> int:
    return int(np.abs(np.asarray(coord, dtype=np.float64) - value).argmin())


def _sat_frame_indices(times_s: np.ndarray, need: list[str], tol_min: int) -> list[int] | None:
    """Index of the frame for each needed ISO time, or None if any is missing."""
    out = []
    for iso in need:
        t = np.datetime64(iso.replace("Z", ""), "s")
        exact = np.flatnonzero(times_s == t)
        if exact.size:
            out.append(int(exact[0]))
            continue
        if tol_min <= 0:
            return None
        delta = np.abs((times_s - t) / np.timedelta64(1, "m"))
        k = int(delta.argmin())
        if delta[k] <= tol_min:
            out.append(k)
        else:
            return None
    return out


def extract_storm(task: tuple[str, list[dict], str, int]) -> dict:
    """Extract every feature for one storm; writes ``cache/feat/{SID}.npz``."""
    sid, rows, feat_dir, sat_tolerance_min = task
    import xarray as xr

    counters = {"n_in": len(rows), "n_ok": 0, "n_era5_missing": 0,
                "n_era5_truncated": 0, "n_sat_missing": 0}
    started = time.time()

    pl_path = C.ERA5_DIR / sid / f"{sid}_pl.nc"
    sl_path = C.ERA5_DIR / sid / f"{sid}_sl_instant.nc"
    if not pl_path.is_file() or not sl_path.is_file():
        counters["n_era5_missing"] = len(rows)
        return {"sid": sid, "status": "no_era5_archive", "counters": counters,
                "seconds": round(time.time() - started, 2)}

    with xr.open_dataset(pl_path) as ds:
        levels = np.asarray(ds.pressure_level.values, dtype=np.float64)
        pl_times = np.asarray(ds.valid_time.values, dtype="datetime64[s]")
        if not all(np.abs(levels - need).min() < 1e-6 for need in C.ERA5_LEVELS):
            counters["n_era5_missing"] = len(rows)
            return {"sid": sid, "status": "pressure_levels_missing",
                    "counters": counters, "seconds": round(time.time() - started, 2)}
        lev_idx = [int(np.abs(levels - need).argmin()) for need in C.ERA5_LEVELS]
        fields = {var: np.asarray(ds[var].isel(pressure_level=lev_idx).load().values,
                                  dtype=np.float32)
                  for var in C.ERA5_VARS}
        lat_pl = np.asarray(ds.latitude.values, dtype=np.float64)
        lon_pl = np.asarray(ds.longitude.values, dtype=np.float64)
        assert np.all(np.diff(lon_pl) > 0), "pl grid longitude is not ascending"
        if lat_pl[0] < lat_pl[-1]:
            # canonicalise orientation: some archives store latitude ascending;
            # flip to descending so every storm shares the same map orientation
            lat_pl = lat_pl[::-1]
            fields = {var: arr[:, :, ::-1, :] if arr.ndim == 4 else arr
                      for var, arr in fields.items()}
    pl_index = {np.datetime64(t, "s"): i for i, t in enumerate(pl_times)}

    with xr.open_dataset(sl_path) as ds:
        sst = np.asarray(ds["sst"].load().values, dtype=np.float32)
        lat_sl = np.asarray(ds.latitude.values, dtype=np.float64)
        lon_sl = np.asarray(ds.longitude.values, dtype=np.float64)
        if lat_sl[0] < lat_sl[-1]:
            lat_sl = lat_sl[::-1]
            sst = sst[:, ::-1, :] if sst.ndim == 3 else sst
        sl_times = np.asarray(ds.valid_time.values, dtype="datetime64[s]")
    sl_index = {np.datetime64(t, "s"): i for i, t in enumerate(sl_times)}

    sat_path = C.FRAMES_DIR / f"{sid}.npy"
    sat_times_path = C.FRAMES_DIR / f"{sid}_times.npy"
    sat = None
    sat_times = None
    if sat_path.is_file() and sat_times_path.is_file():
        sat = np.load(sat_path, mmap_mode="r")
        sat_times = np.asarray(np.load(sat_times_path), dtype="datetime64[s]")

    kept, era5_list, sat_list, his_list = [], [], [], []
    nan_pixels = 0
    total_pixels = 0
    for row in rows:
        need = row["times"]
        need_s = [np.datetime64(t.replace("Z", ""), "s") for t in need]
        if any(np.datetime64(t, "s") not in pl_index for t in need_s) or \
                any(np.datetime64(t, "s") not in sl_index for t in need_s):
            counters["n_era5_missing"] += 1
            continue
        frame_idx = _sat_frame_indices(sat_times, need, sat_tolerance_min) if sat is not None else None
        if frame_idx is None:
            counters["n_sat_missing"] += 1
            continue

        grid = np.empty((C.GRID_CHANNELS, C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        ic = _nearest_index(lat_pl, row["lat"])
        jc = _nearest_index(lon_pl, row["lon"])
        ic_sl = _nearest_index(lat_sl, row["lat"])
        jc_sl = _nearest_index(lon_sl, row["lon"])
        truncated = False
        for vi, var in enumerate(C.ERA5_VARS):
            for ti, t in enumerate(need_s):
                ti_pl = pl_index[np.datetime64(t, "s")]
                win, tr = _era5_window(fields[var][ti_pl], ic, jc)
                truncated = truncated or tr
                for li in range(len(C.ERA5_LEVELS)):
                    grid[vi * len(C.ERA5_LEVELS) * C.N_TIMES + li * C.N_TIMES + ti] = win[li]
        for ti, t in enumerate(need_s):
            win, tr = _era5_window(sst[sl_index[np.datetime64(t, "s")]], ic_sl, jc_sl)
            truncated = truncated or tr
            grid[C.ERA5_CHANNELS + ti] = win
        if truncated:
            counters["n_era5_truncated"] += 1

        frames = np.array(sat[frame_idx], dtype=np.float32)  # (5, 224, 224)
        bad = ~np.isfinite(frames)
        nan_pixels += int(bad.sum())
        total_pixels += int(bad.size)
        for k in range(frames.shape[0]):
            m = bad[k]
            if m.any() and not m.all():
                frames[k][m] = float(frames[k][~m].mean())
            elif m.all():
                frames[k] = np.nan

        diffs = []
        for src in (row["wind5"], row["pres5"]):
            for i in range(C.N_TIMES):
                for j in range(i + 1, C.N_TIMES):
                    diffs.append(src[i] - src[j])
        his = np.asarray(row["wind5"] + row["pres5"] + diffs, dtype=np.float32)

        kept.append(row)
        era5_list.append(grid.astype(np.float16))
        sat_list.append(frames.astype(np.float16))
        his_list.append(his)

    counters["n_ok"] = len(kept)
    out = {"sid": sid, "status": "ok" if kept else "no_usable_sample",
           "counters": counters,
           "sat_nan_frac": (nan_pixels / total_pixels) if total_pixels else float("nan"),
           "seconds": round(time.time() - started, 2)}
    if kept:
        feat_dir = Path(feat_dir)
        feat_dir.mkdir(parents=True, exist_ok=True)
        path = feat_dir / f"{sid}.npz"
        np.savez(
            path,
            row_keys=np.array([r["row_key"] for r in kept]),
            iso_time=np.array([r["iso_time"] for r in kept]),
            era5=np.stack(era5_list),
            sat=np.stack(sat_list),
            his=np.stack(his_list),
            lat=np.array([r["lat"] for r in kept], dtype=np.float32),
            lon=np.array([r["lon"] for r in kept], dtype=np.float32),
            wind=np.array([r["wind"] for r in kept], dtype=np.float32),
            pres=np.array([r["pres"] for r in kept], dtype=np.float32),
            pres_source=np.array([r["pres_source"] for r in kept]),
            dv24=np.array([r["dv24"] for r in kept], dtype=np.float32),
            ri=np.array([r["ri"] for r in kept], dtype=np.int8),
            season=np.array([r["season"] for r in kept], dtype=np.int16),
            split=np.array([r["split"] for r in kept]),
        )
        out["path"] = str(path)
        out["sha256"] = C.sha256_file(path)
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def _per_storm_table(path: str, sid: str) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as z:
        n = int(z["ri"].shape[0])
        table = pd.DataFrame({
            "SID": np.repeat(sid, n),
            "row_key": z["row_keys"].astype(str),
            "iso_time": z["iso_time"].astype(str),
            "time": pd.to_datetime([str(x).rstrip("Z") for x in z["iso_time"]]),
            "season": z["season"].astype(int),
            "split": z["split"].astype(str),
            "lat": z["lat"].astype(float),
            "lon": z["lon"].astype(float),
            "wind_kt": z["wind"].astype(float),
            "pres_hpa": z["pres"].astype(float),
            "pres_source": z["pres_source"].astype(str),
            "dv24_kt": z["dv24"].astype(float),
            "ri": z["ri"].astype(int),
            "sat_nan_frac": np.full(n, np.nan, dtype=float),
        })
    table["storm_pos"] = np.arange(n, dtype=np.int64)
    return table


def assemble(feat_dir: Path, storms: list[dict], out_dir: Path, tag: str,
             keep_perstorm: bool, fingerprint: str, funnel: dict,
             extra_manifest: dict, ri_thr_kt: float = C.RI_THR_KT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    present = sorted([s for s in storms if s.get("status") == "ok"], key=lambda s: s["sid"])
    if not present:
        raise SystemExit("no storm produced any usable sample")

    # ---- pass 1: metadata only (small members), then the global row order
    parts = []
    for s in present:
        t = _per_storm_table(s["path"], s["sid"])
        t["sat_nan_frac"] = s.get("sat_nan_frac", float("nan"))
        t["sha256_perstorm"] = s.get("sha256", "")
        parts.append(t)
    samples = pd.concat(parts, ignore_index=True)
    samples = samples.sort_values(["time", "SID", "storm_pos"], kind="stable") \
                     .reset_index(drop=True)
    samples["sample_id"] = np.arange(len(samples), dtype=np.int64)
    total = len(samples)

    # ---- pass 2: fill the memmaps in exactly that row order
    era5 = np.lib.format.open_memmap(out_dir / "era5.npy", mode="w+", dtype=np.float16,
                                     shape=(total, C.GRID_CHANNELS, C.GRID_SIZE, C.GRID_SIZE))
    satmm = np.lib.format.open_memmap(out_dir / "sat.npy", mode="w+", dtype=np.float16,
                                      shape=(total, C.SAT_FRAMES, C.SAT_SIZE, C.SAT_SIZE))
    hismm = np.lib.format.open_memmap(out_dir / "his.npy", mode="w+", dtype=np.float32,
                                      shape=(total, C.HIS_DIM))
    sid_values = samples["SID"].to_numpy()
    for s in present:
        pos = np.flatnonzero(sid_values == s["sid"])
        with np.load(s["path"], allow_pickle=False) as z:
            if int(z["ri"].shape[0]) != pos.size:
                raise SystemExit(f"{s['sid']}: row count changed between passes")
            era5[pos] = z["era5"]
            satmm[pos] = z["sat"]
            hismm[pos] = z["his"]
    era5.flush(); satmm.flush(); hismm.flush()
    del era5, satmm, hismm

    samples.drop(columns=["storm_pos"]).to_parquet(out_dir / "samples.parquet", index=False)
    norm = compute_norm_stats(out_dir, samples)
    with open(out_dir / "norm.json", "w", encoding="utf-8") as f:
        json.dump(norm, f, indent=2)

    if not keep_perstorm:
        for s in present:
            Path(s["path"]).unlink(missing_ok=True)
        try:
            feat_dir.rmdir()
        except OSError:
            pass

    manifest = {
        "schema": "ritcf_recon_dataset_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": tag,
        "fingerprint": fingerprint,
        "code_sha256": C.code_fingerprint(),
        "feature_spec": C.feature_spec(ri_thr_kt=ri_thr_kt),
        "sources": {
            "ibtracs": {"path": str(C.IBTRACS_CSV), "sha256": C.sha256_file(C.IBTRACS_CSV)},
            "frames_dir": str(C.FRAMES_DIR),
            "frames_manifest": str(C.FRAMES_MANIFEST),
            "frames_manifest_sha256": (C.sha256_file(C.FRAMES_MANIFEST)
                                       if C.FRAMES_MANIFEST.is_file() else None),
            "era5_dir": str(C.ERA5_DIR),
        },
        "funnel": funnel,
        "counts": split_counts(samples),
        "storms": [{k: v for k, v in s.items() if k != "path"} for s in present] +
                  [{k: v for k, v in s.items() if k != "path"}
                   for s in storms if s.get("status") != "ok"],
        "row_keys": {"n": int(len(samples)),
                     "sha256": C.sha256_text("\n".join(samples["row_key"]))},
        "arrays": {
            "era5.npy": {"dtype": "float16",
                         "sha256": C.sha256_file(out_dir / "era5.npy")},
            "sat.npy": {"dtype": "float16",
                        "sha256": C.sha256_file(out_dir / "sat.npy")},
            "his.npy": {"dtype": "float32",
                        "sha256": C.sha256_file(out_dir / "his.npy")},
            "samples.parquet": {"sha256": C.sha256_file(out_dir / "samples.parquet")},
            "norm.json": {"sha256": C.sha256_file(out_dir / "norm.json")},
        },
        "extra": extra_manifest,
    }
    manifest["arrays"]["era5.npy"]["shape"] = [total, C.GRID_CHANNELS, C.GRID_SIZE, C.GRID_SIZE]
    manifest["arrays"]["sat.npy"]["shape"] = [total, C.SAT_FRAMES, C.SAT_SIZE, C.SAT_SIZE]
    manifest["arrays"]["his.npy"]["shape"] = [total, C.HIS_DIM]
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def split_counts(samples: pd.DataFrame) -> dict:
    out = {}
    for name in ("train", "val", "test", "out"):
        s = samples[samples["split"] == name]
        n = int(len(s))
        n_ri = int(s["ri"].sum()) if n else 0
        out[name] = {
            "n": n, "n_ri": n_ri, "n_nonri": n - n_ri,
            "base_rate": (n_ri / n) if n else None,
            "years": [int(s["season"].min()), int(s["season"].max())] if n else None,
            "n_storms": int(s["SID"].nunique()) if n else 0,
        }
    return out


def compute_norm_stats(out_dir: Path, samples: pd.DataFrame) -> dict:
    """Min/max (and NaN fill) per channel from the *train split only*."""
    train_idx = np.flatnonzero((samples["split"] == "train").to_numpy())
    if train_idx.size == 0:  # a smoke subset may hold no train-year row
        train_idx = np.arange(len(samples))
        scope = "all_rows_fallback"
    else:
        scope = "train_split"

    era5 = np.load(out_dir / "era5.npy", mmap_mode="r")
    his = np.load(out_dir / "his.npy", mmap_mode="r")
    ch_min = np.full(C.GRID_CHANNELS, np.inf)
    ch_max = np.full(C.GRID_CHANNELS, -np.inf)
    ch_sum = np.zeros(C.GRID_CHANNELS)
    ch_cnt = np.zeros(C.GRID_CHANNELS, dtype=np.int64)
    for start in range(0, train_idx.size, 512):
        block = np.asarray(era5[train_idx[start:start + 512]], dtype=np.float32)
        flat = block.reshape(block.shape[0], C.GRID_CHANNELS, -1)
        ch_min = np.fmin(ch_min, np.nanmin(flat, axis=(0, 2)))
        ch_max = np.fmax(ch_max, np.nanmax(flat, axis=(0, 2)))
        ch_sum += np.nansum(flat, axis=(0, 2)).astype(np.float64)
        ch_cnt += np.sum(np.isfinite(flat), axis=(0, 2))
    ch_mean = ch_sum / np.maximum(ch_cnt, 1)

    hism = np.asarray(his[train_idx], dtype=np.float64)
    his_min = np.nanmin(hism, axis=0)
    his_max = np.nanmax(hism, axis=0)
    his_mean = np.nanmean(hism, axis=0)
    fin = np.isfinite(his_min) & np.isfinite(his_max)
    era5_min = np.where(np.isfinite(ch_min), ch_min, 0.0)
    era5_max = np.where(np.isfinite(ch_max), ch_max, 0.0)
    era5_fill = np.where(np.isfinite(ch_mean), ch_mean, 0.0)
    for i in range(C.GRID_CHANNELS):  # a constant channel must not divide by zero
        if era5_max[i] - era5_min[i] < 1e-6:
            era5_max[i] = era5_min[i] + 1.0
    return {
        "scope": scope,
        "n_rows": int(train_idx.size),
        "era5": {
            "min": era5_min.tolist(), "max": era5_max.tolist(),
            "fill": era5_fill.tolist(),
            "directional": C.directional_mask(),
            "channel_names": C.era5_channel_names(),
        },
        "his": {
            "min": np.where(fin, his_min, 0.0).tolist(),
            "max": np.where(fin, his_max, 0.0).tolist(),
            "fill": np.where(np.isfinite(his_mean), his_mean, 0.0).tolist(),
            "channel_names": C.his_channel_names(),
        },
        "sat": {"min": C.SAT_MIN_K, "max": C.SAT_MAX_K,
                "fill_normalized": 0.5, "fill_rule": "per_frame_mean_then_0.5"},
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def report(out_dir: Path, manifest: dict) -> None:
    counts = manifest["counts"]
    print(f"\n=== RITCF-recon dataset [{manifest['tag'] or 'full'}] {out_dir}")
    print("funnel (rows, subset scope):")
    for key, value in manifest["funnel"].items():
        print(f"  {key:24s} {value:8d}")
    print("samples:")
    for name in ("train", "val", "test"):
        c = counts[name]
        if not c["n"]:
            print(f"  {name:5s} n=0")
            continue
        yr = c["years"]
        print(f"  {name:5s} n={c['n']:6d} RI={c['n_ri']:5d} nonRI={c['n_nonri']:6d} "
              f"base={100 * c['base_rate']:5.2f}%  years={yr[0]}-{yr[1]} "
              f"storms={c['n_storms']}")
    print("  paper: train 6795 (303 RI) | val 1472 (110) | test 1149 (78)")

    samples = pd.read_parquet(out_dir / "samples.parquet")
    cache_names = ("era5.npy", "sat.npy", "his.npy")
    if all((out_dir / name).is_file() for name in cache_names):
        era5 = np.load(out_dir / "era5.npy", mmap_mode="r")
        sat = np.load(out_dir / "sat.npy", mmap_mode="r")
        his = np.load(out_dir / "his.npy", mmap_mode="r")
        print(f"arrays: era5 {era5.shape} {era5.dtype} | sat {sat.shape} {sat.dtype} | "
              f"his {his.shape} {his.dtype}")
        probe = np.arange(min(len(samples), 256))
        e_nan = float(np.mean([np.isnan(np.asarray(era5[i], dtype=np.float32)).mean() for i in probe]))
        s_nan = float(np.mean([np.isnan(np.asarray(sat[i], dtype=np.float32)).mean() for i in probe]))
        h_nan = float(np.isnan(np.asarray(his[probe], dtype=np.float32)).mean())
        print(f"NaN rate (first {len(probe)} rows): era5 {e_nan:.4%} | sat {s_nan:.4%} "
              f"| his {h_nan:.4%}")
    else:
        print("array caches unavailable; using manifest shapes:")
        for name in cache_names:
            array = manifest.get("arrays", {}).get(name, {})
            print(f"  {name:10s} {array.get('shape', 'unknown')} {array.get('dtype', 'unknown')}")
        print("array NaN rates require rebuilt caches")
    print(f"SAT NaN before per-frame mean fill (per-storm mean): "
          f"{samples['sat_nan_frac'].mean():.4%}")
    print(f"pressure source: {samples['pres_source'].value_counts().to_dict()}")
    print(f"row-key hash: {manifest['row_keys']['sha256'][:16]}... "
          f"(n={manifest['row_keys']['n']})")
    ok = [s for s in manifest["storms"] if s.get("status") == "ok"]
    trunc = int(sum(s["counters"].get("n_era5_truncated", 0) for s in ok))
    print(f"storms: {len(ok)} with samples, {len(manifest['storms']) - len(ok)} without "
          f"| ERA5 windows clamped at the archive edge: {trunc} rows")

    summary = manifest.get("pairs_summary") or {}
    if summary:
        print("\npairs:")
        for name, c in summary.get("pairs", {}).items():
            print(f"  {name:5s} pairs={c['n_pairs']:6d} pos={c['n_pos']:5d} "
                  f"neg={c['n_neg']:6d} targets_without_pair={c['zero_ref_targets']}")
        print("  paper: 6363 pos + 6492 neg = 12855 train pairs (303 RI x 21)")
        print("inference reference pools (10 nearest known RI per target):")
        for kind, per_split in summary.get("refs", {}).items():
            for split, c in per_split.items():
                print(f"  {kind:12s} {split:5s} targets={c['n_targets']:5d} "
                      f"scoreable={c['n_scoreable']:5d} "
                      f"mean_refs={c['mean_refs']:5.2f}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build(args) -> dict:
    out_dir = C.dataset_dir(args.tag)
    storms, years = None, None
    if args.smoke:
        storms, years = SMOKE_STORMS, SMOKE_YEARS
    if args.storms:
        storms = tuple(args.storms)
    if args.years:
        years = tuple(args.years)

    panel = load_panel(seasons=C.SEASON_RANGE, ri_thr_kt=args.ri_thr_kt)
    candidates, funnel = select_candidates(panel, storms, years)
    print(f"candidates: {len(candidates)} rows over {candidates['SID'].nunique()} storms")

    scope = {"smoke": bool(args.smoke), "storms": list(storms) if storms else None,
             "years": list(years) if years else None}
    fingerprint = C.sha256_text(C.canonical_json({
        "spec": C.feature_spec(ri_thr_kt=args.ri_thr_kt),
        "scope": scope,
        "ibtracs_sha256": C.sha256_file(C.IBTRACS_CSV),
        "frames_manifest_sha256": (C.sha256_file(C.FRAMES_MANIFEST)
                                   if C.FRAMES_MANIFEST.is_file() else None),
        "sat_tolerance_min": args.sat_tolerance_min,
        "code": C.code_fingerprint(),
    }))

    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file() and not args.force:
        with open(manifest_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("fingerprint") == fingerprint:
            print(f"cache hit: {manifest_path} (fingerprint {fingerprint[:12]})")
            if not (out_dir / "pairs_train.parquet").is_file() or "pairs" not in old:
                finalize_pairs(out_dir, old)
            return old
        print(f"fingerprint changed ({old.get('fingerprint', '?')[:12]} -> "
              f"{fingerprint[:12]}), rebuilding")

    feat_dir = C.feature_cache_dir(args.tag)
    feat_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(rows[0]["row_key"].split("|")[0], rows, str(feat_dir),
              args.sat_tolerance_min) for rows in candidate_rows(candidates)]
    print(f"extracting {len(tasks)} storms with {args.workers} worker(s)")

    results = []
    t0 = time.time()
    if args.workers <= 1:
        for i, task in enumerate(tasks, 1):
            res = extract_storm(task)
            results.append(res)
            print(f"  [{i}/{len(tasks)}] {res['sid']} {res['status']} "
                  f"ok={res['counters']['n_ok']}/{res['counters']['n_in']} "
                  f"{res['seconds']}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(extract_storm, task): task[0] for task in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                if i % 25 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] {res['sid']} {res['status']} "
                          f"ok={res['counters']['n_ok']}/{res['counters']['n_in']}",
                          flush=True)
    elapsed = time.time() - t0
    print(f"extraction done in {elapsed:.1f}s")

    drop_keys = ("n_in", "n_ok", "n_era5_missing", "n_era5_truncated", "n_sat_missing")
    dropped = {k: int(sum(r["counters"].get(k, 0) for r in results)) for k in drop_keys}
    funnel["7_features_present"] = dropped["n_ok"]
    funnel["8a_dropped_no_era5"] = dropped["n_era5_missing"]
    funnel["8b_dropped_no_sat"] = dropped["n_sat_missing"]

    manifest = assemble(feat_dir, results, out_dir, args.tag, args.keep_perstorm,
                        fingerprint, funnel,
                        {"scope": scope, "args": vars(args),
                         "extraction_seconds": round(elapsed, 1),
                         "worker_counts": dropped},
                        ri_thr_kt=args.ri_thr_kt)
    finalize_pairs(out_dir, manifest)
    return manifest


def finalize_pairs(out_dir: Path, manifest: dict) -> dict:
    """Write the train/val pair tables next to the features and record them."""
    from scripts.ritcf_recon import pairs as P
    samples = P.load_samples(out_dir)
    tables = P.build_pair_tables(samples)
    meta = P.write_pair_tables(out_dir, tables)
    manifest["pairs"] = meta
    manifest["pairs_summary"] = P.summary(samples, tables)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help=f"tiny build: storms {SMOKE_STORMS} + season {SMOKE_YEARS}")
    parser.add_argument("--storms", nargs="*", default=None, help="restrict to these SIDs")
    parser.add_argument("--years", nargs="*", type=int, default=None,
                        help="restrict to these seasons")
    parser.add_argument("--tag", default="", help="dataset tag ('' = full build)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sat-tolerance-min", type=int, default=0,
                        help="allow a SAT frame within +/- N minutes (0 = exact time only)")
    parser.add_argument("--keep-perstorm", action="store_true",
                        help="keep cache/feat/{SID}.npz after assembly")
    parser.add_argument("--force", action="store_true", help="rebuild even on a cache hit")
    parser.add_argument("--ri-thr-kt", type=float, default=C.RI_THR_KT,
                        help="RI threshold in kt (default 25.27; use 30.0 for "
                             "the 30 kt comparison build)")
    parser.add_argument("--test-end-year", type=int, default=None,
                        help="extend the test split through this season "
                             "(default: config 2021; use 2024 for the "
                             "30 kt our-standard comparison)")
    parser.add_argument("--report-only", action="store_true", help="print the existing report")
    args = parser.parse_args()

    if args.test_end_year is not None:
        C.SPLITS = ((2000, 2017, "train"), (2018, 2019, "val"),
                    (2020, int(args.test_end_year), "test"))
        C.SEASON_RANGE = (C.SEASON_RANGE[0], max(C.SEASON_RANGE[1],
                                                 int(args.test_end_year)))

    if args.smoke and args.tag == "":
        args.tag = "smoke"
    if args.workers is None:
        n_cpu = max(1, (os.cpu_count() or 2) - 1)
        args.workers = min(2, n_cpu) if args.smoke else n_cpu

    out_dir = C.dataset_dir(args.tag)
    if args.report_only:
        with open(out_dir / "manifest.json", encoding="utf-8") as f:
            report(out_dir, json.load(f))
        return 0
    report(out_dir, build(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
