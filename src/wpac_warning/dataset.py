# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Phase-2 torch Dataset: joins the phase-1 panel (env vars + targets),
the HURSAT frame stores (raw TBB), and the causal prior graph.

One sample = one panel row (storm, 6h time):
  x_env : (H, N_env)  actual 6 h history t-24..t (H=5 by default), z-scored
          with TRAIN-period stats; internal gaps become standardized zero
  env_mask : (H,) 1=actual row, 0=pre-genesis padding or masked internal gap
  x_ir  : (1, 224, 224) raw TBB, normalized with train-period mean/std
          (with data.ir_lag_steps non-empty: (T, 1, 224, 224) frame stack,
          oldest lag first / t0 last, plus ir_mask (T,) 1=real 0=t0-fill)
  y_dv24, y_ri : targets (never inputs)
  meta  : sid, year, iso_time

The causal prior graph is loaded separately by the model (graph.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset

from wpac_warning.frame_store import (frame_array_path, frame_time_path,
                             has_frame_store, load_frame_times)

P2_ROOT = Path(__file__).resolve().parents[2]


def _resolved(path_like) -> Path:
    p = Path(path_like)
    return p.resolve() if p.is_absolute() else (P2_ROOT / p).resolve()


def _load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


class RIPanelDataset(Dataset):
    """Panel rows with paired IR frames.

    Two protocols are supported:

    * ``loyo`` (legacy default): no row-level year filtering; the split is
      built by storm-first-year inside ``src/train.py``.
    * ``holdout_masked_v1``: explicit fit/inner/test years, canonical
      SID -> storm_year from the phase-1 sample manifest, rows outside their
      split's calendar window deleted, and a 24+6 h label embargo on the tail
      of fit and inner.  ``split_indices`` exposes the three index sets and
      ``split_manifest`` the audit record.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.env_vars = list(cfg.get("data.env_vars"))
        self.hist = int(cfg.get("data.history_steps", 5))
        self.protocol_name = str(cfg.get("data.protocol", "loyo"))
        self.holdout = self.protocol_name == "holdout_masked_v1"
        self.train_years = cfg.get("data.train_years", [2000, 2018])
        if self.holdout:
            from wpac_warning.protocol import SplitProtocol
            self.split_protocol = SplitProtocol.from_cfg(cfg)
            self.split_protocol.validate()
            self.fit_years = list(self.split_protocol.fit_years)
        else:
            self.split_protocol = None
            self.fit_years = list(self.train_years)
        self.frames_qc_audit = None
        self.frame_availability_audit = None
        frames_dir = _resolved(cfg.get("data.frames_dir"))
        if self.holdout:
            from wpac_warning.protocol import (require_frames_corpus, require_population,
                                      sha256_file, validate_qc_manifest)
            data = cfg.to_dict().get("data", {})
            require_population(data)
            require_frames_corpus(data)
            qc_path_value = data.get("qc_manifest")
            if not qc_path_value:
                raise ValueError("data.qc_manifest is required for every holdout backend")
            qc_path = _resolved(qc_path_value)
            qc_audit = validate_qc_manifest(
                _load_json(qc_path), expected_out_frames=frames_dir)
            self.frames_qc_audit = {
                "path": str(qc_path), "sha256": sha256_file(qc_path),
                "qc_rule_version": qc_audit["qc_rule_version"],
                "stores_sha256": qc_audit["stores_sha256"],
                "out_frames_dir": qc_audit["out_frames_dir"],
            }

        panel = pd.read_parquet(cfg.get("data.panel_parquet"))
        panel["ISO_TIME"] = pd.to_datetime(panel["ISO_TIME"], utc=True)
        panel = panel.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

        # canonical SID -> storm_year (phase-1 sample manifest).  In holdout
        # mode this mapping is authoritative and must cover every panel SID.
        self.storm_year_source = str(cfg.get("data.storm_year_manifest", "") or "")
        if self.holdout or self.storm_year_source:
            if not self.storm_year_source:
                raise ValueError("data.storm_year_manifest is required")
            from wpac_warning.protocol import attach_storm_year, load_storm_year_manifest
            manifest = load_storm_year_manifest(self.storm_year_source)
            panel = attach_storm_year(panel, manifest)

        if self.holdout:
            from wpac_warning.protocol import assign_rows, split_summary
            panel = assign_rows(panel, self.split_protocol)
            # audit on the FULL panel (before the frames join), so dropped
            # out-of-window / embargo rows are still visible
            self.split_manifest = split_summary(panel, self.split_protocol)
            self.panel_split_stats = {
                "rows_in": int(len(panel)),
                "rows_eligible": int(panel["eligible"].sum()),
                "dropped_out_of_window": int((~panel["keep_window"]).sum()),
                "dropped_embargo": int(
                    (panel["keep_window"] & ~panel["keep_embargo"]).sum()),
            }
            # a storm belongs to exactly one split; rows outside that split's
            # calendar window are deleted (never re-assigned)
            panel = panel[panel["eligible"]].reset_index(drop=True)

        index = pd.read_parquet(cfg.get("data.frames_index"))
        index["ISO_TIME"] = pd.to_datetime(index["ISO_TIME"], utc=True)
        if self.holdout:
            from wpac_warning.protocol import sha256_array, sha256_file, validate_frame_availability
            availability_value = cfg.get("data.frame_availability")
            if not availability_value:
                raise ValueError("data.frame_availability is required for holdout")
            availability_path = _resolved(availability_value)
            availability = _load_json(availability_path)
            eligible_keys = set(panel["row_uid"].astype(str))
            index["row_uid"] = (index["SID"].astype(str) + "|"
                                + index["ISO_TIME"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            if index["row_uid"].duplicated().any():
                raise ValueError("duplicate row_uid in frames_index")
            index_keys = set(index["row_uid"].astype(str))
            audit = validate_frame_availability(
                availability,
                expected_qc={"sha256": self.frames_qc_audit["sha256"],
                             "stores_sha256": self.frames_qc_audit["stores_sha256"]},
                eligible_row_uids=eligible_keys,
                index_row_uids=index_keys)
            available_hash = sha256_array(sorted(index_keys))
            if audit["n_eligible_rows"] != len(eligible_keys) or \
                    audit["n_available_rows"] != len(index_keys) or \
                    audit["available_row_key_sha256"] != available_hash:
                raise ValueError(
                    "frame availability counts/available hash differ from the "
                    "actual eligible panel and frames_index")
            excluded_keys = {str(row["row_uid"]) for row in availability["rows"]}
            available_keys = eligible_keys - excluded_keys
            panel = panel[panel["row_uid"].isin(available_keys)].reset_index(drop=True)
            split_by_key = panel.set_index("row_uid")["split"]
            for split, entry in (availability.get("by_split") or {}).items():
                split_keys = {key for key in index_keys if split_by_key.get(key) == split}
                if int(entry.get("n_available", -1)) != len(split_keys) or \
                        entry.get("available_row_key_sha256") != sha256_array(
                            sorted(split_keys)):
                    raise ValueError(
                        f"frame availability by_split[{split}] available rows/hash "
                        f"do not match frames_index")
            for row in availability["rows"]:
                sid = str(row["SID"])
                reason = str(row["reason"])
                target = np.datetime64(
                    pd.to_datetime(row["ISO_TIME"], utc=True).tz_localize(None), "ns")
                if not has_frame_store(frames_dir, sid):
                    if reason != "missing_store":
                        raise ValueError(
                            f"frame availability row {row['row_uid']} has no QC store "
                            f"but reason={reason!r}")
                    continue
                if reason == "missing_store":
                    raise ValueError(
                        f"frame availability row {row['row_uid']} declares "
                        f"missing_store but the QC store exists")
                times = load_frame_times(frames_dir, sid)
                if bool((times == target).any()):
                    raise ValueError(
                        f"frame availability forged exclusion for {row['row_uid']}: "
                        f"an exact QC frame exists at t0")
                valid_reason = (
                    (reason == "before_first_frame" and target < times[0])
                    or (reason == "after_last_frame" and target > times[-1])
                    or (reason == "internal_frame_gap"
                        and times[0] < target < times[-1]))
                if not valid_reason:
                    raise ValueError(
                        f"frame availability row {row['row_uid']} reason={reason!r} "
                        f"is inconsistent with its QC store time axis")
            self.frame_availability_audit = {
                "path": str(availability_path),
                "sha256": sha256_file(availability_path),
                **{key: availability[key] for key in
                   ("rule_version", "n_eligible_rows", "n_available_rows",
                    "n_excluded_rows", "excluded_row_key_sha256",
                    "available_row_key_sha256", "by_split", "by_year", "by_storm")},
            }
            self.panel_split_stats.update({
                "dropped_no_frame": int(audit["n_excluded_rows"]),
                "excluded_row_key_sha256": audit["excluded_row_key_sha256"],
                "available_row_key_sha256": available_hash,
            })

        eligible_panel = panel.copy()
        # panel rows WITH a paired frame
        df = panel.merge(index[["SID", "ISO_TIME", "frame_path", "frame_time"]],
                         on=["SID", "ISO_TIME"], how="inner")
        if self.holdout:
            paired_keys = set(zip(df["SID"].astype(str), df["ISO_TIME"]))
            dropped = eligible_panel.loc[
                [key not in paired_keys for key in
                 zip(eligible_panel["SID"].astype(str), eligible_panel["ISO_TIME"])]]
            if not dropped.empty:
                audit = dropped[["SID", "iso_time_str", "split", "row_uid"]].head(
                    50).to_dict(orient="records")
                raise ValueError(
                    f"frames inner join deleted {len(dropped)} eligible holdout "
                    f"panel rows after availability filtering; row-key audit "
                    f"(first 50): {audit}")
            if len(df) != len(eligible_panel):
                raise ValueError(
                    f"frames join changed available row count from "
                    f"{len(eligible_panel)} to {len(df)}; check duplicate frame "
                    f"index keys")


        # per-storm frame stores: PATHS only at init.  Memmaps are opened
        # lazily inside each DataLoader worker (Windows spawn pickles the
        # dataset; an open numpy memmap serializes BY VALUE — ~1.2 GB per
        # worker — while paths keep the pickle tiny)
        self._paths = {}
        fdir = frames_dir
        panel_sids = set(panel["SID"].astype(str))
        for sid in df["SID"].unique():
            f_arr = frame_array_path(fdir, sid)
            f_time = frame_time_path(fdir, sid)
            if f_arr.is_file() and f_time is not None:
                self._paths[sid] = (f_arr, f_time)
        missing_stores = sorted(panel_sids - set(self._paths))
        if missing_stores:
            msg = (f"{len(missing_stores)} panel storms have no frame store "
                   f"under {fdir}: {missing_stores[:5]}")
            if self.holdout:
                raise FileNotFoundError(msg)
            logger.warning(msg + " — dropped")
        self._stores = {}
        df = df[df["SID"].isin(self._paths)].reset_index(drop=True)

        # fit-period feature stats for ALL env inputs (panel env vars are
        # already z-scored; VMAX/PER/SPD/LAT/LAND_DIST are raw — we recompute
        # per-feature stats so every input is ~N(0,1)).  In holdout mode this
        # is the fit window only (2000-2017), never inner/test.
        if self.holdout:
            tr_mask = (df["split"] == "fit").values
        else:
            yr_merged = df["ISO_TIME"].dt.year.values
            tr_mask = ((yr_merged >= self.fit_years[0])
                       & (yr_merged <= self.fit_years[1]))
        tr_rows = df[tr_mask]
        if tr_rows.empty:
            raise ValueError(f"no {self.protocol_name} fit rows for feature stats")
        self.feat_mean = tr_rows[self.env_vars].mean().values.astype(np.float32)
        self.feat_std = (tr_rows[self.env_vars].std().values + 1e-6).astype(np.float32)

        norm = _load_json(cfg.get("data.tbb_norm"))
        self.tbb_mean = float(norm["mean"])
        self.tbb_std = float(norm["std"])

        # Multi-frame temporal input (Plan A: per-frame encode + temporal
        # aggregation).  Extra frames at t0 + lag hours are looked up in the
        # storm's own frame store (nearest frame within +/-1.5 h); lags are
        # sorted ascending so the sequence runs oldest -> newest with t0 LAST.
        # Empty (default) = legacy single-frame behaviour, byte-identical.
        self.ir_lags = sorted(int(l) for l in (cfg.get("data.ir_lag_steps", []) or []))
        if any(l > 0 for l in self.ir_lags):
            raise ValueError(
                f"data.ir_lag_steps must all be <= 0 (a positive lag reads "
                f"frames AFTER t0 — future leakage): {self.ir_lags}")
        if any(l % 6 != 0 for l in self.ir_lags):
            logger.warning(
                f"data.ir_lag_steps {self.ir_lags}: not multiples of 6 h; the "
                f"frame store is 6-hourly, so the +/-1.5 h nearest lookup "
                f"snaps each lag to the closest synoptic frame or fills t0")

        # Offline frame-embedding cache (data.ir_embeds_dir): with
        # ir_lag_steps set, __getitem__ returns precomputed FROZEN backbone
        # embeddings x_ir_embed (T, D) instead of raw frames, using the SAME
        # frame indices and mask semantics as _ir_seq (the frame store's
        # frame time axis (_iso.npy or _times.npy) is still used; the big
        # image array is never opened).  The manifest and every required SID are verified before
        # use; legacy non-holdout mode may still drop uncovered storms.
        self.ir_embeds_dir = str(cfg.get("data.ir_embeds_dir", "") or "")
        self._embed_paths = {}
        self._embed_stores = {}
        self._use_embeds = bool(self.ir_embeds_dir and self.ir_lags)
        self.embedding_cache_audit = None
        if self._use_embeds:
            from wpac_warning.embedding_cache import validate_embedding_cache
            self.embedding_cache_audit = validate_embedding_cache(
                cfg, required_sids=sorted(self._paths) if self.holdout else [],
                allow_partial=not self.holdout)
            ed = Path(self.ir_embeds_dir)
            for sid, (_f_arr, f_iso) in self._paths.items():
                ep = ed / f"{sid}.npy"
                if ep.exists():
                    self._embed_paths[sid] = (ep, f_iso)
            missing_emb = sorted(set(self._paths) - set(self._embed_paths))
            if missing_emb:
                msg = (f"{len(missing_emb)} storms lack cached frame embeddings "
                       f"under {ed}: {missing_emb[:5]}")
                if self.holdout:
                    raise FileNotFoundError(msg)
                n_before = len(df)
                df = df[df["SID"].isin(self._embed_paths)].reset_index(drop=True)
                logger.warning(msg + f"; dropped {n_before - len(df)} rows")

        # Row identity: every prediction file must carry these keys so test
        # inference and evaluation can be re-merged and bootstrapped by storm.
        # Holdout rows already carry row_uid from protocol.assign_rows.
        if "storm_year" not in df.columns:
            # legacy LOYO convention: storm-first-year (row-level years would
            # let a cross-year storm straddle folds).  The holdout path sets
            # the canonical manifest year upstream.
            sid_year = df.groupby("SID")["ISO_TIME"].min().dt.year
            df["storm_year"] = df["SID"].map(sid_year).astype(int)
        df["storm_year"] = df["storm_year"].astype(int)
        if "iso_time_str" not in df.columns:
            df["iso_time_str"] = df["ISO_TIME"].dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        if "row_uid" not in df.columns:
            df["row_uid"] = df["SID"].astype(str) + "|" + df["iso_time_str"]
        if df["row_uid"].duplicated().any():
            raise ValueError("duplicate (SID, ISO_TIME) rows after the split filter")


        # optional smoke cap (explicitly recorded in the run manifest)
        limit = int(cfg.get("data.smoke_limit_rows", 0) or 0)
        self.smoke_limit_rows = limit
        if limit > 0:
            if self.holdout:
                keep = []
                for split in ("fit", "inner", "test"):
                    idx = df.index[df["split"] == split][:limit]
                    keep.extend(idx.tolist())
                df = df.loc[sorted(keep)].reset_index(drop=True)
            else:
                df = df.iloc[:limit].reset_index(drop=True)
            logger.warning(f"data.smoke_limit_rows={limit}: capped the dataset "
                           f"to {len(df)} rows (smoke only)")
        self.df = df

        if self.holdout:

            self.split_indices = {
                split: np.where((self.df["split"] == split).values)[0]
                for split in ("fit", "inner", "test")
            }
            self.split_manifest["panel_filter"] = getattr(
                self, "panel_split_stats", {})
            self.split_manifest["frame_availability"] = self.frame_availability_audit
            self.split_manifest["n_rows_with_frame"] = int(len(self.df))
            for split in ("fit", "inner", "test"):
                s = self.df["split"] == split
                self.split_manifest["splits"][split]["n_rows_with_frame"] = int(s.sum())
                self.split_manifest["splits"][split]["sids_with_frame"] = sorted(
                    self.df.loc[s, "SID"].astype(str).unique().tolist())
        else:
            self.split_indices = {}
            self.split_manifest = {}


        # NOTE (legacy loyo): no row-level year filtering here — LOYO splitting
        # is done BY STORM in train.py (row-level splits leak a storm across
        # folds).  Holdout rows were already filtered above.

        # Timestamp-indexed rows make the 6 h environmental history explicit.
        self._groups = {
            sid: g.set_index("ISO_TIME", drop=False)
            for sid, g in self.df.groupby("SID", sort=False)
        }


    def __len__(self) -> int:
        return len(self.df)

    def _env_seq_with_mask(self, row) -> tuple[np.ndarray, np.ndarray]:
        g = self._groups[row["SID"]]
        current = pd.Timestamp(row["ISO_TIME"])
        first = pd.Timestamp(g.index.min())
        seq, mask = [], []
        for slot in range(self.hist):
            target = current - pd.Timedelta(hours=6 * (self.hist - 1 - slot))
            if target in g.index:
                values = g.loc[target, self.env_vars].to_numpy(dtype=np.float32)
                mask.append(1.0)
            elif target < first:
                # Before the storm's first record: retain the historical
                # boundary-padding convention and expose a zero validity mask.
                values = g.iloc[0][self.env_vars].to_numpy(dtype=np.float32)
                mask.append(0.0)
            else:
                # Missing internal 6 h slots become standardized zero and are
                # masked out by CausalRINet; never join adjacent rows by count.
                values = np.full(len(self.env_vars), np.nan, dtype=np.float32)
                mask.append(0.0)
            seq.append(values)
        seq = np.stack(seq).astype(np.float32, copy=False)
        seq = (seq - self.feat_mean) / self.feat_std
        return (np.nan_to_num(seq, nan=0.0),
                np.asarray(mask, dtype=np.float32))

    def _env_seq(self, row) -> np.ndarray:
        return self._env_seq_with_mask(row)[0]

    def _open_store(self, sid: str):
        if sid not in self._stores:
            f_arr, f_iso = self._paths[sid]
            self._stores[sid] = (np.load(f_arr, mmap_mode="r"), np.load(f_iso))
        return self._stores[sid]

    def _norm_frame(self, arr, j: int) -> np.ndarray:
        v = np.asarray(arr[j], dtype=np.float32)
        v = (v - self.tbb_mean) / self.tbb_std
        return np.nan_to_num(v[None], nan=0.0)

    def _ir_frame(self, row) -> np.ndarray:
        sid = row["SID"]
        arr, iso = self._open_store(sid)
        key = np.datetime64(row["ISO_TIME"].tz_localize(None), "s")
        j = int(np.searchsorted(iso, key))
        if j >= len(iso) or iso[j] != key:
            # frame missing at this time: zeros (masked mean after norm)
            return np.zeros((1, 224, 224), dtype=np.float32)
        return self._norm_frame(arr, j)

    @staticmethod
    def _lag_lookup(iso, key, lag: int) -> int:
        """Store index of the frame nearest to key + lag hours within
        +/-1.5 h (frames are 3-hourly on exact synoptic times, so this is
        normally an exact match); -1 when no such frame exists (storm early
        life, lag before genesis).  Shared by the image and embedding paths
        so both see IDENTICAL frame indices."""
        tgt = key + np.timedelta64(lag * 3600, "s")
        tol = 1.5 * 3600.0  # seconds
        j = int(np.searchsorted(iso, tgt))
        best, best_dt = -1, tol + 1
        for cand in (j - 1, j):
            if 0 <= cand < len(iso):
                dt = abs(float((iso[cand] - tgt) / np.timedelta64(1, "s")))
                if dt < best_dt:
                    best, best_dt = cand, dt
        return best if best_dt <= tol else -1

    def _ir_seq(self, row) -> tuple:
        """(T, 1, 224, 224) frame stack + (T,) mask for data.ir_lag_steps.

        Frame order: lags sorted ascending (oldest first), t0 LAST; missing
        lag slots are filled with the t0 frame and masked 0.  The t0 slot
        keeps the legacy _ir_frame behaviour exactly (zeros on miss, mask
        stays 1: the frames-index inner join guarantees a paired frame).
        """
        sid = row["SID"]
        arr, iso = self._open_store(sid)
        key = np.datetime64(row["ISO_TIME"].tz_localize(None), "s")
        frames, masks = [], []
        for lag in self.ir_lags:
            best = self._lag_lookup(iso, key, lag)
            if best >= 0:
                frames.append(self._norm_frame(arr, best))
                masks.append(1.0)
            else:
                frames.append(None)  # filled with the t0 frame below
                masks.append(0.0)
        t0 = self._ir_frame(row)
        frames = [t0 if f is None else f for f in frames] + [t0]
        masks.append(1.0)
        return (np.stack(frames, axis=0),
                np.asarray(masks, dtype=np.float32))

    def _embed_seq(self, row) -> tuple:
        """(T, D) cached frozen backbone embeddings + (T,) mask — the exact
        embedding-path analogue of _ir_seq (same indices via _lag_lookup,
        same fill/mask semantics; t0 missing -> zeros, mask stays 1)."""
        sid = row["SID"]
        if sid not in self._embed_stores:
            ep, f_iso = self._embed_paths[sid]
            emb = np.load(ep, mmap_mode="r")
            iso = np.load(f_iso)
            if emb.shape[0] != len(iso):
                raise ValueError(
                    f"frame-embedding cache out of sync for {sid}: {ep} has "
                    f"{emb.shape[0]} rows but {f_iso} has {len(iso)} frame "
                    f"times — re-run scripts/precompute_frame_embeds.py")
            self._embed_stores[sid] = (emb, iso)
        emb, iso = self._embed_stores[sid]
        key = np.datetime64(row["ISO_TIME"].tz_localize(None), "s")
        rows, masks = [], []
        for lag in self.ir_lags:
            best = self._lag_lookup(iso, key, lag)
            if best >= 0:
                rows.append(np.asarray(emb[best], dtype=np.float32))
                masks.append(1.0)
            else:
                rows.append(None)  # filled with the t0 embedding below
                masks.append(0.0)
        j = int(np.searchsorted(iso, key))
        if j < len(iso) and iso[j] == key:
            t0 = np.asarray(emb[j], dtype=np.float32)
        else:
            t0 = np.zeros((emb.shape[1],), dtype=np.float32)
        rows = [t0 if r is None else r for r in rows] + [t0]
        masks.append(1.0)
        return (np.stack(rows, axis=0),
                np.asarray(masks, dtype=np.float32))

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        x_env, env_mask = self._env_seq_with_mask(row)
        item = {
            "x_env": torch.from_numpy(x_env),
            "env_mask": torch.from_numpy(env_mask),
            "y_dv24": torch.tensor(float(row["DV24"]), dtype=torch.float32),
            "y_ri": torch.tensor(float(row["RI"]), dtype=torch.float32),
            "sid": row["SID"],
            "year": int(row["ISO_TIME"].year),
            "iso_time": row["iso_time_str"],
            "storm_year": int(row["storm_year"]),
            "row_uid": row["row_uid"],
            "split": str(row["split"]) if "split" in self.df.columns else "",
        }
        if self.ir_lags:
            if self._use_embeds:
                x_e, ir_mask = self._embed_seq(row)
                item["x_ir_embed"] = torch.from_numpy(x_e)   # (T, D)
                item["ir_mask"] = torch.from_numpy(ir_mask)  # (T,) 1=real, 0=fill
            else:
                x_ir, ir_mask = self._ir_seq(row)
                item["x_ir"] = torch.from_numpy(x_ir)        # (T, 1, 224, 224)
                item["ir_mask"] = torch.from_numpy(ir_mask)  # (T,) 1=real, 0=fill
        else:
            item["x_ir"] = torch.from_numpy(self._ir_frame(row))  # (1, 224, 224)
        return item
