# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""MAE in-domain pretraining on ALL HURSAT TBB frames (self-supervised).

Data: <cache>/frames.npy + index.parquet, built by
scripts/merge_global_cache.py. Two production corpora, both fail-closed:

  * QC v1 (deduplicated): data/mae_cache_global_qc / data/mae_cache_wnp_alltc
    -> checkpoints .../mae_global_qc / mae_wnp_alltc  (audit, no longer the
    production default)
  * full multi-sensor (``--source-corpus raw_full``: every raw frame kept,
    duplicate acquisition times preserved, non-physical TBB masked to NaN at
    build): data/mae_cache_global_full / data/mae_cache_wnp_alltc_full
    -> checkpoints .../mae_global_full / mae_wnp_alltc_full

Pre-QC and raw/unmasked caches are rejected; the versioned out_dir must match
the verified cache corpus. The downstream frame corpus stays
data/global_hursat/frames_qc regardless. The split is STORM-disjoint by
canonical manifest storm year (with an IBTrACS SID-year fallback for
global-only storms), and frame-year windows plus the configured excluded
years are applied to the raw cache index.

Frames are normalized with data/tbb_norm.json (mean/std fitted on 2000-2018
by the pairing pipeline — kept for consistency with downstream inputs).

Features: AdamW + cosine lr with warmup (lr = base_lr * batch/256), mixed
precision, resume-from-checkpoint (model+optimizer+scaler+rng), periodic
checkpoints and original/masked/reconstruction visualization panels.

Usage (from the release root):
    KMP_DUPLICATE_LIB_OK=TRUE ~/anaconda3/python.exe scripts/pretrain_mae.py
    # smoke: 2 epochs on a small subset
    KMP_DUPLICATE_LIB_OK=TRUE ~/anaconda3/python.exe scripts/pretrain_mae.py \
        --epochs 2 --subset 8192 --batch-size 128 --warmup 1 --save-every 1
    # resume
    KMP_DUPLICATE_LIB_OK=TRUE ~/anaconda3/python.exe scripts/pretrain_mae.py \
        --resume checkpoints/mae_pretrain_last.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.mae import MaskedAutoencoderViT
from wpac_warning.mae_provenance import (MAE_CORPUS_FULL, checkpoint_dir_names,
                                compute_mae_manifest_fingerprint,
                                verify_cache_manifest,
                                verify_full_source_hashes,
                                verify_qc_for_config)
from wpac_warning.protocol import (load_storm_year_manifest, sha256_array, sha256_file,
                          sha256_json)
from wpac_warning.utils_cfg import load_cfg

DEFAULT_CONFIG = P2_ROOT / "configs" / "phase2" / "holdout_masked_v1.yaml"


class MAEFrameDataset(Dataset):
    """Memmap-backed HURSAT frames; normalization from tbb_norm.json.

    The memmap is opened LAZILY inside each worker (Windows spawn pickling
    of a >4 GB memmap object fails, so only the path crosses the process
    boundary)."""

    def __init__(self, cache_dir: Path, indices: np.ndarray,
                 mean: float, std: float):
        self._path = cache_dir / "frames.npy"
        self._frames = None
        self.idx = indices
        self.mean = mean
        self.std = std

    @property
    def frames(self):
        if self._frames is None:
            self._frames = np.load(self._path, mmap_mode="r")
        return self._frames

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        v = np.asarray(self.frames[self.idx[i]], dtype=np.float32)
        v = (v - self.mean) / self.std
        v = np.nan_to_num(v, nan=0.0)
        return torch.from_numpy(v[None])  # (1, 224, 224)


def _year_exclusion_mask(years: pd.Series,
                         excluded_years: Sequence[int]) -> pd.Series:
    if not excluded_years:
        return pd.Series(False, index=years.index)
    if len(excluded_years) == 2:
        lo, hi = int(excluded_years[0]), int(excluded_years[1])
        if lo > hi:
            raise ValueError(f"excluded year bounds are reversed: {[lo, hi]}")
        return years.between(lo, hi)
    return years.isin(excluded_years)


def _canonical_storm_year(sids: pd.Series, storm_year_manifest) -> pd.Series:
    if storm_year_manifest is None:
        mapping = pd.Series(dtype=np.int64)
    elif isinstance(storm_year_manifest, (str, Path)):
        mapping = load_storm_year_manifest(storm_year_manifest)
    else:
        mapping = pd.Series(storm_year_manifest).copy()
        if mapping.index.has_duplicates:
            raise ValueError("duplicate SIDs in storm-year manifest")
        mapping.index = mapping.index.astype(str)
        if mapping.isna().any():
            raise ValueError("missing years in storm-year manifest")
        years = pd.to_numeric(mapping, errors="raise").astype(int)
        mapping = pd.Series(years.values, index=mapping.index)

    sid = sids.astype(str)
    storm_year = sid.map(mapping)
    missing = storm_year.isna()
    if missing.any():
        prefix = sid.str[:4]
        valid_prefix = prefix.str.fullmatch(r"\d{4}")
        invalid = missing & ~valid_prefix
        if invalid.any():
            bad = sorted(sid[invalid].unique().tolist())
            raise ValueError(
                f"{len(bad)} SIDs have no canonical manifest year and no "
                f"IBTrACS year prefix (first 10: {bad[:10]})")
        storm_year.loc[missing] = pd.to_numeric(
            prefix[missing], errors="raise").astype(int)
    return storm_year.astype(int)


def cache_sid_year_audit(index: pd.DataFrame, storm_year_manifest=None,
                         source_hashes: dict | None = None) -> dict:
    """Record the raw cache's canonical SID/year universe without filtering it.

    ``source_hashes`` (full-multisensor caches): {sid: {"source_array_sha256",
    "source_times_sha256"}} verified against the raw corpus; every record must
    be covered.
    """
    if not {"SID", "time"}.issubset(index.columns):
        raise ValueError("MAE cache SID/year audit requires SID and time columns")
    work = index[["SID", "time"]].copy()
    work["SID"] = work["SID"].astype(str)
    work["time"] = pd.to_datetime(work["time"], errors="coerce", utc=True)
    if work["time"].isna().any():
        raise ValueError("MAE cache has invalid times")
    work["year"] = work["time"].dt.year.astype(int)
    work["storm_year"] = _canonical_storm_year(work["SID"], storm_year_manifest)
    records = []
    for sid, group in work.groupby("SID", sort=True):
        record = {
            "SID": str(sid),
            "storm_year": int(group["storm_year"].iloc[0]),
            "first_frame_year": int(group["year"].min()),
            "last_frame_year": int(group["year"].max()),
            "n_frames": int(len(group)),
        }
        if source_hashes is not None:
            entry = source_hashes.get(str(sid))
            if not entry:
                raise ValueError(
                    f"full-multisensor cache SID {sid} lacks its per-SID "
                    f"source hash audit")
            record.update(entry)
        records.append(record)
    return {
        "n_rows": int(len(work)),
        "n_sids": len(records),
        "records": records,
        "content_sha256": sha256_json(records),
        "sids_sha256": sha256_array(sorted(record["SID"] for record in records)),
    }


def verify_expected_sid_coverage(split_df: pd.DataFrame, expected_universe,
                                 train_years: Sequence[int],
                                 val_years: Sequence[int]) -> dict:
    """Require exact canonical SID coverage in both development splits."""
    if isinstance(expected_universe, (str, Path)):
        expected = load_storm_year_manifest(expected_universe)
    else:
        expected = pd.Series(expected_universe).copy()
    if expected.index.has_duplicates or expected.isna().any():
        raise ValueError("expected SID universe has duplicate SIDs or missing years")
    expected.index = expected.index.astype(str)
    expected = pd.Series(pd.to_numeric(expected, errors="raise").astype(int).values,
                         index=expected.index)
    actual = split_df.copy()
    actual["SID"] = actual["SID"].astype(str)
    audits = {}
    problems = {}
    for split, years in (("train", train_years), ("val", val_years)):
        lo, hi = int(years[0]), int(years[-1])
        expected_sids = set(expected.loc[expected.between(lo, hi)].index)
        actual_sids = set(actual.loc[actual["split"] == split, "SID"])
        missing = sorted(expected_sids - actual_sids)
        unexpected = sorted(actual_sids - expected_sids)
        audits[split] = {
            "years": [lo, hi],
            "expected_n_sids": len(expected_sids),
            "actual_n_sids": len(actual_sids),
            "expected_sids_sha256": sha256_json(sorted(expected_sids)),
            "actual_sids_sha256": sha256_json(sorted(actual_sids)),
            "missing_sids": missing,
            "unexpected_sids": unexpected,
        }
        if missing or unexpected:
            problems[split] = {"missing": missing[:20], "unexpected": unexpected[:20]}
    if problems:
        raise ValueError(f"MAE expected SID universe coverage failed: {problems}")
    return {"coverage_verified": True, "splits": audits}


def build_clean_index(index: pd.DataFrame, val_years: Sequence[int],
                      train_years: Sequence[int] | None = None,
                      excluded_years: Sequence[int] | None = None,
                      n_frames: int | None = None,
                      storm_year_manifest=None
                      ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Clean development-window index (HOLDOUT_MASKED_V1 §4).

    Splits use canonical SID storm years from ``storm_year_manifest``.  SIDs
    absent from the manifest use the four-digit IBTrACS SID year, never the
    first cached frame.  Frames are retained only when both their storm and
    frame years belong to the same split.  Two ``excluded_years`` values are
    inclusive bounds; those frames are filtered from the raw cache index.

    ``frame_row`` (the row number inside ``frames.npy``) is REQUIRED: it is
    never re-derived from the filtered position.  There is no random
    validation fallback.
    """
    if "frame_row" not in index.columns:
        raise ValueError(
            "MAE index has no 'frame_row' column: rebuild the cache (or run "
            "pretrain_mae.py --add-frame-row on a verified RangeIndex) — "
            "positional fallback is forbidden")
    if index.empty:
        raise ValueError("MAE index is empty")
    frame_row = index["frame_row"].to_numpy(dtype=np.int64)
    if len(np.unique(frame_row)) != len(frame_row):
        raise ValueError("frame_row values are not unique")
    if frame_row.min() < 0:
        raise ValueError("negative frame_row")
    if n_frames is not None and frame_row.max() >= n_frames:
        raise ValueError(f"frame_row {frame_row.max()} exceeds frames.npy "
                         f"length {n_frames}")

    val_years = [int(y) for y in val_years]
    excluded_years = [int(y) for y in (excluded_years or [])]
    if not val_years:
        raise ValueError("mae.val_years is empty")
    idx = index.copy()
    idx["SID"] = idx["SID"].astype(str)
    idx["time"] = pd.to_datetime(idx["time"])
    idx["year"] = idx["time"].dt.year
    if idx["year"].isna().any():
        raise ValueError("MAE index has missing or invalid frame times")
    idx["storm_year"] = _canonical_storm_year(
        idx["SID"], storm_year_manifest)

    if train_years is not None:
        if len(train_years) != 2:
            raise ValueError("mae.train_frame_years must be [first, last]")
        lo, hi = int(train_years[0]), int(train_years[1])
        if lo > hi:
            raise ValueError(f"training year bounds are reversed: {[lo, hi]}")
        train_sid_mask = idx["storm_year"].between(lo, hi)
        in_train_window = idx["year"].between(lo, hi)
    else:
        train_sid_mask = (~idx["storm_year"].isin(val_years)
                          & ~_year_exclusion_mask(idx["storm_year"],
                                                  excluded_years))
        in_train_window = ~idx["year"].isin(val_years)

    is_excluded = _year_exclusion_mask(idx["year"], excluded_years)
    val_sids = set(idx.loc[idx["storm_year"].isin(val_years), "SID"])
    if not val_sids:
        raise ValueError(
            f"no storm has a canonical year in validation years {val_years}; "
            f"refusing to fall back to a random split")
    train_sids = set(idx.loc[train_sid_mask, "SID"])
    leak = val_sids & train_sids
    if leak:
        raise ValueError(f"{len(leak)} storms appear in both splits")

    is_val = (idx["SID"].isin(val_sids) & idx["year"].isin(val_years)
              & ~is_excluded)
    in_train = (idx["SID"].isin(train_sids) & in_train_window
                & ~is_excluded)
    if in_train.sum() == 0 or is_val.sum() == 0:
        raise ValueError("empty MAE train or validation split")
    tr = np.where(in_train.values)[0]
    va = np.where(is_val.values)[0]
    split_df = idx.loc[in_train | is_val].copy()
    split_df["split"] = np.where(split_df["SID"].isin(val_sids), "val", "train")
    split_df["frame_row"] = split_df["frame_row"].astype(np.int64)
    n_excluded_left = int(_year_exclusion_mask(
        split_df["year"], excluded_years).sum())
    if n_excluded_left != 0:
        raise AssertionError(
            f"{n_excluded_left} excluded-year frames survived MAE filtering")
    return tr, va, split_df


def storm_disjoint_split(index: pd.DataFrame, val_years: Sequence[int],
                         train_years: Sequence[int] | None = None,
                         excluded_years: Sequence[int] | None = None,
                         n_frames: int | None = None,
                         storm_year_manifest=None
                         ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Backwards-compatible alias of :func:`build_clean_index`."""
    return build_clean_index(index, val_years, train_years, excluded_years,
                             n_frames, storm_year_manifest)




def adjust_lr(opt, epoch: int, base_lr: float, warmup: int, epochs: int):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(warmup, 1)
    else:
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup)
                                           / max(epochs - warmup, 1)))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            loss, _, _ = model(x)
        tot += float(loss) * len(x)
        n += len(x)
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def visualize(model, loader, device, out_path: Path, seed: int = 0):
    """Original / masked / reconstruction panel for the first val batch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    x = next(iter(loader))[:8].to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand(len(x), model.patch_embed.n_patches, generator=g).to(device)
    loss, pred, mask = model(x, noise=noise)
    rec = model.unpatchify(pred).float().cpu()
    orig = x.float().cpu()
    m = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, model.patch ** 2))
    m = m.float().cpu()
    masked = orig * (1 - m) + 0.0 * m  # masked patches -> 0 (norm mean)
    recon = orig * (1 - m) + rec * m   # paste reconstruction on masked only

    fig, axes = plt.subplots(8, 3, figsize=(9, 24))
    n_show = min(8, len(x))
    for i in range(8):
        for j, (img, title) in enumerate([(orig, "original"), (masked, "masked"),
                                          (recon, "reconstruction")]):
            axes[i, j].axis("off")
            if i >= n_show:
                continue
            axes[i, j].imshow(img[i, 0], cmap="gray_r")
            if i == 0:
                axes[i, j].set_title(title)
    fig.suptitle(f"loss={float(loss):.4f}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    model.train()


def validate_checkpoint_provenance(state: dict, protocol: str,
                                   manifest_fingerprint: str,
                                   fingerprint_key: str, label: str) -> None:
    if state.get("protocol") != protocol:
        raise RuntimeError(
            f"{label} protocol {state.get('protocol')!r} != current {protocol!r}")
    if state.get(fingerprint_key) != manifest_fingerprint:
        raise RuntimeError(
            f"{label} manifest fingerprint differs from the current MAE index; "
            f"refusing to reuse or repackage it")


def adopt_frame_row(index: pd.DataFrame, frames_path: Path) -> pd.DataFrame:
    """Adopt positional rows only after independent cache-contract checks."""
    if not isinstance(index.index, pd.RangeIndex) or \
            index.index.start != 0 or index.index.step != 1 or \
            index.index.stop != len(index):
        raise ValueError("--add-frame-row requires an exact 0..N-1 RangeIndex")
    n_frames = int(np.load(frames_path, mmap_mode="r").shape[0])
    if len(index) != n_frames:
        raise ValueError(
            f"--add-frame-row row-count mismatch: index has {len(index)} rows "
            f"but frames.npy has {n_frames}")
    if not {"SID", "time"}.issubset(index.columns):
        raise ValueError("--add-frame-row requires SID and time order metadata")
    if "src_storm" in index.columns:
        source_ok = index["src_storm"].astype(str) == index["SID"].astype(str)
    elif "src_path" in index.columns:
        checks = []
        for path, sid in zip(index["src_path"], index["SID"]):
            source = PureWindowsPath(str(path))
            sid = str(sid)
            checks.append(source.parent.name == sid
                          or source.name.startswith(sid + "."))
        source_ok = pd.Series(checks, index=index.index)
    else:
        raise ValueError("--add-frame-row requires source SID metadata")
    if not bool(source_ok.all()):
        raise ValueError("--add-frame-row source SID metadata is inconsistent")
    ordered = index.sort_values(["SID", "time"], kind="mergesort")
    if not np.array_equal(ordered.index.to_numpy(), np.arange(len(index))):
        raise ValueError(
            "--add-frame-row cannot verify frames.npy order: index rows are "
            "not in deterministic SID/time cache order")
    out = index.copy()
    out["frame_row"] = np.arange(len(index), dtype=np.int64)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None,
                    help="limit train frames (smoke tests)")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--early-stop-patience", type=int, default=None,
                    help="stop after N epochs without validation improvement")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="override mae.cache_dir (e.g. data/mae_cache_global)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="override mae.out_dir (checkpoints)")
    ap.add_argument("--vis-dir", type=str, default=None,
                    help="override mae.vis_dir")
    ap.add_argument("--tbb-norm", type=str, default=None,
                    help="override data.tbb_norm")
    ap.add_argument("--population", choices=("global", "alltc"), default=None,
                    help="MAE cache/training universe; global is never implicitly "
                         "restricted to the WNP canonical sample")
    ap.add_argument("--expected-sid-universe", type=str, default=None,
                    help="canonical SID/YEAR CSV; filter to this universe and "
                         "require exact fit+inner SID coverage")
    ap.add_argument("--verify-cache", action="store_true",
                    help="only verify the cache fingerprint and exit")
    ap.add_argument("--audit-legacy", action="store_true",
                    help="explicitly allow a non-holdout audit/legacy config")
    ap.add_argument("--add-frame-row", action="store_true",
                    help="adopt a verified 0..N-1 RangeIndex as frame_row and "
                         "rewrite index.parquet (no positional fallback)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    m = cfg.get("mae", {})
    protocol = str(cfg.get("data.protocol", "loyo"))
    if protocol != "holdout_masked_v1" and not args.audit_legacy:
        raise ValueError(
            "non-holdout MAE is audit/legacy only; pass --audit-legacy explicitly")
    population = str(args.population or m.get("population")
                     or cfg.get("data.population", "alltc")).strip().lower()
    if population in ("wnp", "wnp_alltc", "all_tc"):
        population = "alltc"
    if population not in ("global", "alltc"):
        raise ValueError("MAE population must be global or alltc")
    epochs = args.epochs or int(m.get("epochs", 400))
    batch_size = args.batch_size or int(m.get("batch_size", 256))
    warmup = args.warmup if args.warmup is not None else int(m.get("warmup_epochs", 40))
    save_every = args.save_every or int(m.get("save_every", 20))
    early_stop_patience = (args.early_stop_patience
                           if args.early_stop_patience is not None
                           else int(m.get("early_stop_patience", 0)))
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be >= 0")
    seed = int(m.get("seed", 42))
    global_paths = protocol == "holdout_masked_v1" and population == "global"
    default_out = (m.get("global_out_dir") if global_paths else m.get("out_dir"))
    default_cache = (m.get("global_cache_dir") if global_paths else m.get("cache_dir"))
    out_dir = P2_ROOT / (args.out_dir or default_out or "checkpoints")
    cache_dir = P2_ROOT / (args.cache_dir or default_cache or "data/mae_cache")
    vis_dir = P2_ROOT / (args.vis_dir or m.get("vis_dir", "logs/mae_vis"))
    # the versioned out_dir name is validated against the VERIFIED cache
    # corpus right after cache verification (fail-closed before any weights
    # are written); mkdir itself is harmless
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.add(str(P2_ROOT / "logs" / "pretrain_mae.log"), level="INFO",
               rotation="10 MB")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    frames_path = cache_dir / "frames.npy"
    index_path = cache_dir / "index.parquet"
    if not frames_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f"MAE cache missing under {cache_dir}")

    # ---- cache fingerprint and QC chain: no stale-cache short circuit ----
    cache_manifest_path = cache_dir / "cache_manifest.json"
    qc_audit = None
    cache_qc = None
    mae_corpus = None
    if protocol == "holdout_masked_v1":
        qc_audit = verify_qc_for_config(cfg)
        cache_qc = verify_cache_manifest(cache_dir, qc_audit, population)
        mae_corpus = cache_qc["mae_corpus"]
        expected_out = checkpoint_dir_names(mae_corpus)[population]
        if out_dir.name != expected_out:
            raise ValueError(
                f"holdout {population} MAE out_dir must be {expected_out} for "
                f"the {mae_corpus} corpus; raw/unversioned output paths are "
                f"forbidden")
        fingerprint = {
            "frames_npy_sha256": cache_qc["frames_npy_sha256"],
            "index_parquet_sha256": cache_qc["index_parquet_sha256"],
            "protocol": protocol,
        }
        old = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") not in (None, fingerprint):
            raise RuntimeError(
                f"MAE cache fingerprint changed under {cache_dir}: refusing to "
                f"reuse a stale cache/index")
        logger.info(f"MAE QC cache and provenance verified ({cache_dir})")
    else:
        fingerprint = {
            "frames_npy_sha256": sha256_file(frames_path),
            "index_parquet_sha256": sha256_file(index_path),
            "protocol": protocol,
        }
        if cache_manifest_path.is_file():
            old = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            if old.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"MAE cache fingerprint changed under {cache_dir}: refusing to "
                    f"reuse a stale cache/index. Rebuild it explicitly or point "
                    f"--cache-dir elsewhere.\n  old {old.get('fingerprint')}\n"
                    f"  new {fingerprint}")
            logger.info(f"MAE cache fingerprint verified ({cache_dir})")
        else:
            cache_manifest_path.write_text(json.dumps(
                {"fingerprint": fingerprint,
                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                indent=2, sort_keys=True), encoding="utf-8")
            logger.info(f"MAE cache fingerprint written ({cache_manifest_path})")
    if args.verify_cache:
        print(json.dumps(fingerprint, indent=2))
        return 0

    index = pd.read_parquet(index_path)
    index["time"] = pd.to_datetime(index["time"])
    n_frames = int(np.load(frames_path, mmap_mode="r").shape[0])
    native_frame_row = "frame_row" in index.columns
    if "frame_row" not in index.columns:
        if not args.add_frame_row:
            raise ValueError(
                f"{index_path} has no frame_row column; rebuild the cache with "
                f"the updated builder, or pass --add-frame-row only after "
                f"RangeIndex, row-count and cache-order verification")
        index = adopt_frame_row(index, frames_path)
        index.to_parquet(index_path, index=False)
        fingerprint["index_parquet_sha256"] = sha256_file(index_path)
        cache_record = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        cache_record["fingerprint"] = fingerprint
        cache_record["frame_row_source"] = "verified-rangeindex"
        if protocol == "holdout_masked_v1":
            cache_record["index_sha256"] = fingerprint["index_parquet_sha256"]
            cache_qc["index_parquet_sha256"] = fingerprint["index_parquet_sha256"]
        cache_record.setdefault(
            "created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        cache_manifest_path.write_text(json.dumps(cache_record, indent=2, sort_keys=True),
                                       encoding="utf-8")
        logger.warning("frame_row adopted after RangeIndex, row-count, source "
                       "SID and deterministic cache-order verification")
    val_years = m.get("val_years")
    if val_years is None:
        if m.get("val_year") is None:
            raise ValueError("mae.val_years (or legacy val_year) is required")
        val_years = [int(m["val_year"])]
    train_years = m.get("train_frame_years")
    excluded = m.get("excluded_years", [])
    storm_year_path = cfg.get("data.storm_year_manifest")
    if protocol == "holdout_masked_v1" and not storm_year_path:
        raise ValueError("data.storm_year_manifest is required for holdout MAE")
    storm_years = None
    if storm_year_path:
        storm_year_path = Path(storm_year_path)
        if not storm_year_path.is_absolute():
            storm_year_path = P2_ROOT / storm_year_path
        storm_years = load_storm_year_manifest(storm_year_path)

    source_hashes = None
    if mae_corpus == MAE_CORPUS_FULL:
        # the merge-time per-SID source audit is the provenance root for the
        # full-multisensor corpus: verify the audit file itself, then prove
        # its per-SID source hashes still match the raw corpus on disk
        cache_block = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        sid_entry = cache_block.get("sid_audit")
        if not isinstance(sid_entry, dict) or not sid_entry.get("path") \
                or not sid_entry.get("sha256"):
            raise ValueError(
                "full-multisensor cache manifest lacks its per-SID "
                "source/cache audit (sid_audit)")
        sid_audit_path = Path(str(sid_entry["path"]))
        if not sid_audit_path.is_file() and \
                (cache_dir / sid_audit_path.name).is_file():
            sid_audit_path = cache_dir / sid_audit_path.name
        if not sid_audit_path.is_file():
            raise FileNotFoundError(
                f"full-multisensor cache per-SID audit missing: {sid_audit_path}")
        if sha256_file(sid_audit_path) != str(sid_entry["sha256"]).lower():
            raise ValueError(
                "full-multisensor cache per-SID audit differs from the cache "
                "manifest (sid_audit)")
        merge_records = json.loads(
            sid_audit_path.read_text(encoding="utf-8")).get("records") or []
        verify_full_source_hashes(merge_records, cache_qc["frames_source_dir"])
        source_hashes = {
            str(r.get("sid", r.get("SID", ""))): {
                "source_array_sha256": r["source_array_sha256"],
                "source_times_sha256": r["source_times_sha256"]}
            for r in merge_records}

    cache_audit = cache_sid_year_audit(index, storm_years,
                                       source_hashes=source_hashes)
    if protocol == "holdout_masked_v1" and \
            cache_audit["sids_sha256"] != cache_qc["sids_sha256"]:
        raise ValueError("cache SID/year universe differs from cache_manifest.sids_sha256")
    # full-multisensor: keep the merge-time cache_sid_years.json (the cache
    # manifest's sid_audit) intact and write the holdout audit alongside it;
    # QC mode overwrites cache_sid_years.json exactly as before
    cache_sid_path = cache_dir / ("cache_sid_years_holdout.json"
                                  if mae_corpus == MAE_CORPUS_FULL
                                  else "cache_sid_years.json")
    cache_sid_path.write_text(json.dumps(cache_audit, indent=2, sort_keys=True),
                              encoding="utf-8")
    cache_sid_ref = {
        "path": str(cache_sid_path.resolve()),
        "sha256": sha256_file(cache_sid_path),
        "content_sha256": cache_audit["content_sha256"],
        "sids_sha256": cache_audit["sids_sha256"],
        "n_sids": cache_audit["n_sids"],
    }
    cache_record = {}
    if cache_manifest_path.is_file():
        cache_record = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    cache_record.update({
        "fingerprint": fingerprint,
        "frame_row_source": "native" if native_frame_row else "verified-rangeindex",
        "sid_years": cache_sid_ref,
    })
    cache_manifest_path.write_text(json.dumps(cache_record, indent=2, sort_keys=True),
                                   encoding="utf-8")
    cache_manifest_ref = {"path": str(cache_manifest_path.resolve()),
                          "sha256": sha256_file(cache_manifest_path)}
    qc_manifest_ref = None
    if qc_audit is not None:
        qc_manifest_ref = {
            "path": qc_audit["path"], "sha256": qc_audit["sha256"],
            "stores_sha256": qc_audit["stores_sha256"],
            "qc_rule_version": qc_audit["qc_rule_version"],
        }

    expected_value = args.expected_sid_universe
    if expected_value is None and population == "alltc":
        expected_value = m.get("expected_sid_universe")
    if protocol == "holdout_masked_v1" and population == "alltc":
        expected_value = expected_value or storm_year_path
    if protocol == "holdout_masked_v1" and population == "global" and expected_value:
        raise ValueError("global MAE cannot use a WNP/expected SID universe filter")
    expected_mapping = None
    expected_provenance = None
    if expected_value:
        expected_path = Path(expected_value)
        if not expected_path.is_absolute():
            expected_path = P2_ROOT / expected_path
        expected_mapping = load_storm_year_manifest(expected_path)
        expected_provenance = {"path": str(expected_path.resolve()),
                               "sha256": sha256_file(expected_path)}
        if protocol == "holdout_masked_v1" and population == "alltc":
            sids_from = Path(cache_qc["sids_from"])
            if not sids_from.is_absolute():
                sids_from = P2_ROOT / sids_from
            if not sids_from.is_file() or \
                    sha256_file(sids_from) != expected_provenance["sha256"]:
                raise ValueError(
                    "WNP cache sids_from differs from the current canonical SID universe")
        index = index[index["SID"].astype(str).isin(expected_mapping.index)].copy()
        if index.empty:
            raise ValueError("expected SID universe removed every MAE cache row")
    split_storm_years = expected_mapping if expected_mapping is not None else storm_years
    tr_idx, va_idx, split_df = build_clean_index(
        index, val_years, train_years=train_years, excluded_years=excluded,
        n_frames=n_frames, storm_year_manifest=split_storm_years)
    coverage = None
    if expected_mapping is not None:
        coverage = verify_expected_sid_coverage(
            split_df, expected_mapping, train_years, val_years)
        coverage["universe"] = expected_provenance

    tr_rows = split_df.loc[split_df["split"] == "train", "frame_row"].values
    va_rows = split_df.loc[split_df["split"] == "val", "frame_row"].values
    if args.subset:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(tr_rows), size=min(args.subset, len(tr_rows)),
                          replace=False)
        tr_rows = tr_rows[keep]
    split_path = cache_dir / "index_holdout.parquet"
    split_df.to_parquet(split_path, index=False)
    logger.info(f"frames: train={len(tr_rows)} val={len(va_rows)} "
                f"(storm-disjoint, val_years={val_years}, "
                f"train_window={train_years}); filtered index -> {split_path}")

    norm_path = args.tbb_norm or cfg.get("data.tbb_norm")
    if not norm_path:
        raise ValueError("data.tbb_norm is required")
    norm = json.loads(Path(norm_path).read_text())
    mean, std = float(norm["mean"]), float(norm["std"])
    tr_ds = MAEFrameDataset(cache_dir, tr_rows, mean, std)
    va_ds = MAEFrameDataset(cache_dir, va_rows, mean, std)

    val_sids = sorted(split_df.loc[split_df["split"] == "val", "SID"].unique().tolist())
    train_sids = sorted(split_df.loc[split_df["split"] == "train", "SID"].unique().tolist())
    leak = set(val_sids) & set(train_sids)
    if leak:
        raise ValueError(f"{len(leak)} storms in both MAE splits")
    mae_manifest = {
        "protocol": protocol,
        "population": population,
        "mae_corpus": mae_corpus,
        "qc_rule_version": qc_audit["qc_rule_version"] if qc_audit else None,
        "qc_manifest": qc_manifest_ref,
        "cache_dir": str(cache_dir.resolve()),
        "cache_manifest": cache_manifest_ref,
        "cache": fingerprint,
        "cache_sid_years": cache_sid_ref,
        "filtered_index": {"path": str(split_path.resolve()),
                           "sha256": sha256_file(split_path)},
        "expected_sid_universe": expected_provenance,
        "expected_sid_coverage": coverage,
        "train_frame_years": train_years,
        "val_years": val_years,
        "excluded_years": excluded,
        "n_train_frames": int(len(tr_rows)),
        "n_val_frames": int(len(va_rows)),
        "filtered_n_train_frames": int((split_df["split"] == "train").sum()),
        "filtered_n_val_frames": int((split_df["split"] == "val").sum()),
        "n_train_sids": len(train_sids),
        "n_val_sids": len(val_sids),
        "train_sids": train_sids,
        "val_sids": val_sids,
        "tbb_norm": {"path": str(Path(norm_path).resolve()),
                     "sha256": sha256_file(norm_path)},
        "config_sha256": sha256_json(cfg.to_dict()),
        "early_stop_patience": early_stop_patience,
        "seed": seed,
        "frame_row_source": "native" if native_frame_row else "verified-rangeindex",
    }
    manifest_fingerprint = compute_mae_manifest_fingerprint(mae_manifest)
    mae_manifest["fingerprint"] = manifest_fingerprint
    manifest_path = out_dir / "mae_manifest.json"
    nw = min(8, int(cfg.get("train.num_workers", 4)))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=nw, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

    model = MaskedAutoencoderViT(
        mask_ratio=float(m.get("mask_ratio", 0.75)),
        norm_pix_loss=bool(m.get("norm_pix_loss", True))).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MAE ViT-S/16 params: {n_params:,}")

    base_lr = float(m.get("base_lr", 1.5e-4)) * batch_size / 256.0
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            betas=(0.9, 0.95),
                            weight_decay=float(m.get("weight_decay", 0.05)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint_provenance(
            ckpt, protocol, manifest_fingerprint,
            "mae_manifest_fingerprint", "MAE resume checkpoint")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        np.random.set_state(ckpt["numpy_rng"])
        if device == "cuda" and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
        logger.info(f"resumed from {args.resume} at epoch {start_epoch}")

    best_val, best_ep = float("inf"), -1
    if (out_dir / "mae_best.pt").is_file():
        prior_best = torch.load(out_dir / "mae_best.pt", map_location="cpu",
                                weights_only=False)
        validate_checkpoint_provenance(
            prior_best, protocol, manifest_fingerprint,
            "manifest_fingerprint", "existing mae_best.pt")
        best_val = float(prior_best["val_loss"])
        best_ep = int(prior_best["epoch"])
        logger.info(f"existing best: val={best_val:.4f} @ ep{best_ep}")

    manifest_path.write_text(json.dumps(mae_manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    logger.info(f"MAE manifest: {manifest_path} "
                f"(fingerprint={manifest_fingerprint[:16]}…)")

    epochs_since_best = 0
    early_stopped = False
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        lr = adjust_lr(opt, ep, base_lr, warmup, epochs)
        model.train()
        tot, nb = 0.0, 0
        for x in tr_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss, _, _ = model(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach().cpu())
            nb += 1
        tr_loss = tot / max(nb, 1)
        va_loss = evaluate(model, va_loader, device)
        logger.info(f"ep{ep}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                    f"lr={lr:.2e} ({time.time() - t0:.0f}s)")
        if va_loss < best_val:
            best_val, best_ep = float(va_loss), ep
            epochs_since_best = 0
            torch.save({
                "kind": "mae_best",
                "model": model.state_dict(),
                "epoch": ep,
                "val_loss": float(va_loss),
                "protocol": protocol,
                "manifest_fingerprint": manifest_fingerprint,
                "train_frame_years": train_years,
                "val_years": val_years,
            }, out_dir / "mae_best.pt")
            logger.info(f"  new best val {best_val:.4f} @ ep{ep} -> mae_best.pt")
        else:
            epochs_since_best += 1

        if early_stop_patience and epochs_since_best >= early_stop_patience:
            early_stopped = True
            logger.info(f"early stop at ep{ep}: no val improvement for "
                        f"{epochs_since_best} epochs (best ep{best_ep})")

        last = early_stopped or ep == epochs - 1
        if (ep + 1) % save_every == 0 or last:
            state = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": ep,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
                "config": {k: m.get(k) for k in ("mask_ratio", "norm_pix_loss")},
                "protocol": protocol,
                "mae_manifest_fingerprint": manifest_fingerprint,
                "train_frame_years": train_years,
                "val_years": val_years,
            }
            torch.save(state, out_dir / "mae_pretrain_last.pt")
            torch.save({"model": model.state_dict(), "epoch": ep},
                       out_dir / f"mae_epoch{ep + 1:03d}.pt")
            # downstream-ready encoder weights (decoder stripped)
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "note": "full MAE state dict; ViTMAEEncoder loads "
                                "encoder keys and ignores dec_*/mask_token"},
                       out_dir / "mae_vit_s16_final.pt")
            visualize(model, va_loader, device,
                      vis_dir / f"epoch{ep + 1:03d}.png", seed=seed)
            logger.info(f"checkpoint + visualization saved at ep{ep}")
        if early_stopped:
            break

    if best_ep < 0:
        raise RuntimeError("MAE produced no valid best checkpoint; refusing success")
    if best_ep >= 0:
        best_state = torch.load(out_dir / "mae_best.pt", map_location="cpu",
                                weights_only=False)
        torch.save({"model": best_state["model"], "epoch": best_ep,
                    "val_loss": float(best_val),
                    "protocol": protocol,
                    "manifest_fingerprint": manifest_fingerprint,
                    "train_frame_years": train_years,
                    "val_years": val_years,
                    "excluded_years": excluded,
                    "population": population,
                    "early_stopped": bool(early_stopped),
                    "note": "best-val full MAE state dict; ViTMAEEncoder "
                            "loads encoder keys and ignores dec_*/mask_token"},
                   out_dir / "mae_vit_s16_best.pt")
        mae_manifest["best"] = {
            "epoch": int(best_ep),
            "val_loss": float(best_val),
            "early_stopped": bool(early_stopped),
            "checkpoint": "mae_best.pt",
            "checkpoint_sha256": sha256_file(out_dir / "mae_best.pt"),
            "encoder_checkpoint": "mae_vit_s16_best.pt",
            "encoder_sha256": sha256_file(out_dir / "mae_vit_s16_best.pt"),
        }
        manifest_path.write_text(
            json.dumps(mae_manifest, indent=2, sort_keys=True),
            encoding="utf-8")
        logger.info(f"best-val checkpoint: ep{best_ep} val={best_val:.4f} -> "
                    f"{out_dir / 'mae_vit_s16_best.pt'}")

    logger.info(f"MAE pretraining done -> {out_dir / 'mae_vit_s16_final.pt'} "
                f"(best -> {out_dir / 'mae_vit_s16_best.pt'})")
    return 0



if __name__ == "__main__":
    sys.exit(main())
