# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Merge per-storm frame stores into a single float16 memmap MAE cache.

Two explicit source corpora (``--source-corpus``):

* ``qc`` (default): the QC v1 frame library (``data/global_hursat/frames_qc``,
  rule ``hursat_qc_v1``) — one frame per acquisition time, already masked.
  Produces the versioned QC caches
      global clean  data/mae_cache_global_qc
      WNP all-TC    data/mae_cache_wnp_alltc   (``--sids-from`` the 725-system
                                                 canonical sample manifest)
* ``raw_full``: the RAW download corpus (``data/global_hursat/frames``) with
  NO deduplication — every frame is kept, including duplicate acquisition
  times from multiple satellites, so the self-supervised objective sees every
  sensor view.  Non-physical pixels (finite TBB < 100 K or > 350 K, and any
  non-finite pixel) are set to NaN frame-by-frame while the cache is written.
  Produces the versioned full-multisensor caches
      global full   data/mae_cache_global_full
      WNP all-TC    data/mae_cache_wnp_alltc_full (``--sids-from`` as above)

The raw_full cache is SELF-SUPERVISED PRETRAINING ONLY: the downstream
corpus stays ``data/global_hursat/frames_qc`` (frames_qc / frame availability
/ training row sets are untouched).

Layout (same as the legacy cache, so ``scripts/pretrain_mae.py`` works
unchanged):

    <out>/frames.npy     (N, 224, 224) float16
    <out>/index.parquet  row per frame: SID, time, src_storm, frame_row
    <out>/progress.txt   resume cursor (storms done)
    <out>/cache_manifest.json  corpus binding (see below)
    <out>/cache_sid_years.json per-SID source/cache audit

The raw_full manifest records ``mae_corpus=hursat_full_multisensor_v1``,
``source_corpus=raw_full``, ``deduplicate=false``, the mask range, masked
pixel count, duplicate-time statistics, input (per-SID source) and output
(frames/index) hashes and the per-SID audit reference.  The QC manifest is
unchanged (``qc_manifest`` binding, no ``mae_corpus`` field).

Resumable; resume/force/index checks stay fail-closed in both modes: a cache
built under one ``--source-corpus`` is never resumed or adopted under the
other.

Usage (from the release root):
    python scripts/01_data/merge_global_cache.py                       # global, QC
    python scripts/01_data/merge_global_cache.py --sids-from \\
        data/phase1/processed_masked/holdout_fit0017/panel/sample_manifest.csv \\
        --out-dir data/mae_cache_wnp_alltc                     # WNP all-TC, QC
    python scripts/01_data/merge_global_cache.py --source-corpus raw_full \\
        --out-dir data/mae_cache_global_full                   # global, full
    python scripts/01_data/merge_global_cache.py --source-corpus raw_full \\
        --sids-from <725 sample manifest> \\
        --out-dir data/mae_cache_wnp_alltc_full                # WNP all-TC, full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.frame_store import require_frame_time_path  # noqa: E402
from wpac_warning.protocol import (MAE_CORPUS_FULL_MULTISENSOR,  # noqa: E402
                          QC_MANIFEST_REL, QC_RULE_VERSION,
                          QC_HURSAT_FRAMES_REL, RAW_HURSAT_FRAMES_REL,
                          TBB_VALID_MAX_K, TBB_VALID_MIN_K,
                          is_qc_hursat_frames, is_raw_hursat_frames,
                          sha256_array, sha256_file, sha256_json,
                          validate_qc_manifest)

CROP = 224
SRC = P2_ROOT / QC_HURSAT_FRAMES_REL
RAW_SRC = P2_ROOT / RAW_HURSAT_FRAMES_REL
OUT = P2_ROOT / "data" / "mae_cache_global_qc"
OUT_FULL = P2_ROOT / "data" / "mae_cache_global_full"
QC_MANIFEST = P2_ROOT / QC_MANIFEST_REL
SOURCE_CORPORA = ("qc", "raw_full")


def _canonical_sids(path: Path) -> list:
    frame = pd.read_csv(path)
    cols = {c.lower(): c for c in frame.columns}
    if "sid" not in cols:
        raise ValueError(f"{path} has no SID column")
    if "included" in cols:
        inc = frame[cols["included"]]
        keep = (inc if inc.dtype == bool
                else inc.astype(str).str.lower().isin(("true", "1", "yes")))
        frame = frame[keep]
    return sorted(frame[cols["sid"]].astype(str).unique().tolist())


def _index_mismatches(existing: pd.DataFrame, expected: pd.DataFrame) -> dict:
    """Per-SID/per-row differences between a cached index and the QC corpus."""
    problems: dict = {}
    for col in ("SID", "time", "frame_row"):
        if col not in existing.columns:
            problems["missing_column"] = col
            return problems
    if len(existing) != len(expected):
        problems["row_count"] = [int(len(existing)), int(len(expected))]
        return problems
    # stable sort: duplicate acquisition times (raw_full) keep store order
    exp = expected.sort_values(["SID", "time"], kind="mergesort") \
        .reset_index(drop=True)
    got = existing.sort_values(["SID", "time"], kind="mergesort") \
        .reset_index(drop=True)
    if not (got["SID"].astype(str).to_numpy()
            == exp["SID"].astype(str).to_numpy()).all():
        bad = sorted(set(got["SID"].astype(str)) ^ set(exp["SID"].astype(str)))
        problems["sids"] = bad[:5]
    if not (pd.to_datetime(got["time"], utc=True).to_numpy()
            == pd.to_datetime(exp["time"], utc=True).to_numpy()).all():
        problems["times"] = "frame times differ"
    if not (got["frame_row"].to_numpy(dtype="int64")
            == exp["frame_row"].to_numpy(dtype="int64")).all():
        problems["frame_row"] = "frame_row mapping differs"
    return problems


def _expected_index(storms, src: Path) -> pd.DataFrame:

    """Index rows derived from the CURRENT stores (times + frame rows).

    Every store row is kept in BOTH modes: the QC corpus holds one frame per
    acquisition time, the raw corpus may repeat a time (multi-sensor views) —
    no deduplication happens here.
    """
    rows = []
    for p in storms:
        sid = p.stem
        n = int(np.load(p, mmap_mode="r").shape[0])
        t = np.load(require_frame_time_path(src, sid), mmap_mode="r")
        if len(t) != n:
            raise RuntimeError(f"{sid}: frames/times mismatch ({n} vs {len(t)})")
        rows.append(pd.DataFrame({"SID": sid, "time": pd.to_datetime(t),
                                  "src_storm": sid}))
    index = pd.concat(rows, ignore_index=True)
    index["frame_row"] = np.arange(len(index), dtype=np.int64)
    return index


def _verify_existing_cache(out: Path, index_path: Path, frames_path: Path,
                           expected: pd.DataFrame, qc: dict | None,
                           frames_dir: Path, source_corpus: str) -> dict:
    """Refuse a cache whose index/manifest no longer match this source run."""
    manifest_path = out / "cache_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"{out} holds an index/frames cache without cache_manifest.json; "
            f"the cache cannot be proven to come from this source corpus - "
            f"delete {index_path.name}/{frames_path.name} or pass --force to "
            f"rebuild")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "mae_frame_cache":
        raise SystemExit(f"{manifest_path}: not a mae_frame_cache manifest")
    built_corpus = str(manifest.get("source_corpus") or "qc")
    if built_corpus != source_corpus:
        raise SystemExit(
            f"{manifest_path}: cache was built with --source-corpus "
            f"{built_corpus} but this run is {source_corpus}; a cache is never "
            f"reused across corpora - rebuild with --force")
    if source_corpus == "raw_full":
        if manifest.get("mae_corpus") != MAE_CORPUS_FULL_MULTISENSOR:
            raise SystemExit(
                f"{manifest_path}: raw_full cache lacks "
                f"mae_corpus={MAE_CORPUS_FULL_MULTISENSOR!r}; rebuild with "
                f"--force")
        if manifest.get("deduplicate") is not False:
            raise SystemExit(
                f"{manifest_path}: raw_full cache must record "
                f"deduplicate=false; rebuild with --force")
        rng = [float(x) for x in manifest.get("tbb_mask_range_k", [])]
        if rng != [TBB_VALID_MIN_K, TBB_VALID_MAX_K]:
            raise SystemExit(
                f"{manifest_path}: raw_full cache tbb_mask_range_k={rng} != "
                f"{[TBB_VALID_MIN_K, TBB_VALID_MAX_K]}; rebuild with --force")
    bound = manifest.get("qc_manifest") or {}
    if qc is not None:
        if str(bound.get("sha256", "")).lower() != qc["sha256"].lower():
            raise SystemExit(
                f"{manifest_path}: cache was built from a different QC manifest "
                f"({bound.get('sha256')} != {qc['sha256']}); rebuild with "
                f"--force")
        if bound.get("stores_sha256") != qc["stores_sha256"]:
            raise SystemExit(
                f"{manifest_path}: cache QC stores hash differs from the current "
                f"QC corpus; rebuild with --force")
    if str(manifest.get("frames_source_dir", "")) != str(frames_dir):
        raise SystemExit(
            f"{manifest_path}: cache frames_source_dir="
            f"{manifest.get('frames_source_dir')!r} != {frames_dir!r}; rebuild "
            f"with --force")
    if manifest.get("index_sha256") != sha256_file(index_path):
        raise SystemExit(
            f"{index_path}: hash differs from {manifest_path} (the index was "
            f"edited or half-written); rebuild with --force")
    if sha256_array(sorted(expected["SID"].astype(str).unique())) != \
            manifest.get("sids_sha256"):
        raise SystemExit(
            f"{manifest_path}: cached SID set differs from the current source "
            f"corpus; rebuild with --force")
    existing = pd.read_parquet(index_path)
    for col in ("SID", "time", "frame_row"):
        if col not in existing.columns:
            raise SystemExit(f"{index_path}: lacks the '{col}' column; rebuild "
                             f"with --force")
    if len(existing) != len(expected):
        raise SystemExit(
            f"{index_path}: {len(existing)} rows but the current source corpus "
            f"yields {len(expected)}; rebuild with --force")
    exp = expected.sort_values(["SID", "time"], kind="mergesort") \
        .reset_index(drop=True)
    got = existing.sort_values(["SID", "time"], kind="mergesort") \
        .reset_index(drop=True)
    if not (got["SID"].astype(str).to_numpy()
            == exp["SID"].astype(str).to_numpy()).all():
        raise SystemExit(f"{index_path}: SID column differs from the current "
                         f"source corpus; rebuild with --force")
    if not (pd.to_datetime(got["time"], utc=True).to_numpy()
            == pd.to_datetime(exp["time"], utc=True).to_numpy()).all():
        raise SystemExit(
            f"{index_path}: frame times differ from the current source corpus; "
            f"rebuild with --force")
    if not (got["frame_row"].to_numpy(dtype="int64")
            == exp["frame_row"].to_numpy(dtype="int64")).all():
        raise SystemExit(f"{index_path}: frame_row mapping differs; rebuild "
                         f"with --force")
    return manifest


def _write_sid_audit(out: Path, index: pd.DataFrame, src: Path,
                     offsets: dict) -> tuple:
    """Per-SID source/cache hashes (resume and MAE provenance both read it)."""
    records = []
    for sid, grp in index.groupby("SID", sort=True):
        sid = str(sid)
        times = pd.to_datetime(grp["time"], utc=True)
        keys = [f"{sid}|{t.strftime('%Y-%m-%dT%H:%M:%SZ')}" for t in times]
        array = src / f"{sid}.npy"
        times_path = require_frame_time_path(src, sid)
        records.append({
            "sid": sid,
            "storm_year": int(sid[:4]),
            "n_frames": int(len(grp)),
            "row_first": int(offsets[sid]),
            "row_last": int(offsets[sid] + len(grp) - 1),
            "time_first": times.min().isoformat(),
            "time_last": times.max().isoformat(),
            "first_frame_year": int(times.min().year),
            "last_frame_year": int(times.max().year),
            "row_key_sha256": sha256_array(sorted(keys)),
            "source_array_sha256": sha256_file(array),
            "source_times_sha256": sha256_file(times_path),
        })
    payload = {"kind": "mae_cache_sid_audit", "n_sids": len(records),
               "sids_sha256": sha256_array(sorted(r["sid"] for r in records)),
               "records": records,
               "content_sha256": sha256_json(records)}
    path = out / "cache_sid_years.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path, records


def _source_stores_sha256(records: list) -> str:
    """Input-side hash: every per-SID raw source array/time hash, sorted."""
    return sha256_json([{"sid": r["sid"],
                         "source_array_sha256": r["source_array_sha256"],
                         "source_times_sha256": r["source_times_sha256"]}
                        for r in sorted(records, key=lambda r: r["sid"])])


def _masked_frame_block(arr) -> tuple:
    """float16 block with non-physical pixels set to NaN.

    Only finite out-of-range pixels (TBB < 100 K or > 350 K) are counted as
    masked; pre-existing NaN stays NaN.
    """
    block = np.asarray(arr, dtype=np.float32)
    bad = np.isfinite(block) & ((block < TBB_VALID_MIN_K)
                                | (block > TBB_VALID_MAX_K))
    n_bad = int(bad.sum())
    if n_bad:
        block[bad] = np.nan
    return block.astype(np.float16), n_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-corpus", choices=SOURCE_CORPORA, default="qc",
                    help="qc: deduplicated QC v1 library (default); raw_full: "
                         "every raw multi-sensor frame, non-physical TBB "
                         "masked to NaN, no deduplication (pretraining only)")
    ap.add_argument("--frames-dir", default=None,
                    help="frame corpus (default: data/global_hursat/frames_qc "
                         "for qc, data/global_hursat/frames for raw_full)")
    ap.add_argument("--out-dir", default=None,
                    help="cache directory (default data/mae_cache_global_qc / "
                         "data/mae_cache_global_full)")
    ap.add_argument("--qc-manifest", default=str(QC_MANIFEST))
    ap.add_argument("--sids-from", default=None,
                    help="sample manifest CSV; only its INCLUDED SIDs are "
                         "merged (used for the WNP all-TC cache)")
    ap.add_argument("--force", action="store_true",
                    help="discard an existing index/frames cache and rebuild")
    ap.add_argument("--allow-unqc", action="store_true",
                    help="audit-only escape hatch (qc mode only): accept a "
                         "non-QC frame corpus (the cache manifest then records "
                         "qc_manifest=null)")
    args = ap.parse_args()

    raw_full = args.source_corpus == "raw_full"
    src = Path(args.frames_dir
               or (RAW_SRC if raw_full else SRC)).resolve()
    out = Path(args.out_dir
               or (OUT_FULL if raw_full else OUT)).resolve()
    qc_path = Path(args.qc_manifest).resolve()

    qc_audit = None
    if raw_full:
        if not is_raw_hursat_frames(src):
            raise SystemExit(
                f"{src} is not the raw download corpus "
                f"({RAW_HURSAT_FRAMES_REL}); --source-corpus raw_full reads "
                f"the full multi-sensor raw frames, never frames_qc")
    elif not args.allow_unqc:
        if not is_qc_hursat_frames(src):
            raise SystemExit(f"{src} is not the QC v1 corpus "
                             f"({QC_HURSAT_FRAMES_REL}); pass --allow-unqc only "
                             f"for an audit-only cache")
        qc_audit = validate_qc_manifest(
            json.loads(qc_path.read_text(encoding="utf-8")),
            expected_out_frames=src)
        qc_audit = {**qc_audit, "sha256": sha256_file(qc_path)}

    wanted = None
    if args.sids_from:
        wanted = set(_canonical_sids(Path(args.sids_from)))

    out.mkdir(parents=True, exist_ok=True)
    storms = sorted(p for p in src.glob("*.npy")
                    if not p.name.endswith(("_times.npy", "_iso.npy")))
    if wanted is not None:
        storms = [p for p in storms if p.stem in wanted]
        missing = sorted(wanted - {p.stem for p in storms})
        if missing:
            raise SystemExit(f"{len(missing)} requested SIDs have no source "
                             f"store: {missing[:5]}")
    print(f"storms: {len(storms)} from {src} "
          f"(source_corpus={args.source_corpus})", flush=True)

    index_path = out / "index.parquet"
    progress_path = out / "progress.txt"
    frames_path = out / "frames.npy"

    expected = _expected_index(storms, src)
    if args.force:
        for name in ("index.parquet", "frames.npy", "progress.txt",
                     "cache_manifest.json", "cache_sid_years.json",
                     "mask_counts.json"):
            (out / name).unlink(missing_ok=True)
    manifest_path = out / "cache_manifest.json"
    if index_path.exists() and not manifest_path.exists():
        # interrupted first run (index + partial frames + progress, no manifest):
        # the index is derived data, so verify it against the current source
        # corpus per SID and continue; a complete cache still has to pass the
        # full gate
        current = pd.read_parquet(index_path)
        problems = _index_mismatches(current, expected)
        if problems:
            raise SystemExit(
                f"{out} holds an unfinished cache whose index no longer matches "
                f"the current source corpus ({problems}); rebuild with --force")
        print(f"resuming an interrupted cache ({len(current)} frames indexed, "
              f"no cache_manifest.json yet)", flush=True)
        index = current
        if not progress_path.exists():
            progress_path.write_text("", encoding="utf-8")
        if frames_path.is_file():
            shape = np.load(frames_path, mmap_mode="r").shape
            if int(shape[0]) != len(index):
                raise SystemExit(
                    f"{frames_path}: {shape[0]} rows but index.parquet has "
                    f"{len(index)}; rebuild with --force")
    elif index_path.exists():
        # never trust a completed cache: verify it against this source run
        _verify_existing_cache(out, index_path, frames_path, expected, qc_audit,
                               src, args.source_corpus)
        index = pd.read_parquet(index_path)
        if frames_path.is_file():
            shape = np.load(frames_path, mmap_mode="r").shape
            if int(shape[0]) != len(index):
                raise SystemExit(
                    f"{frames_path}: {shape[0]} rows but index.parquet has "
                    f"{len(index)}; rebuild with --force")
    else:
        index = expected
        index.to_parquet(index_path)
    N = len(index)
    print(f"frames total: {N}", flush=True)

    mm = np.lib.format.open_memmap(
        frames_path, mode="r+" if frames_path.exists() else "w+",
        dtype=np.float16, shape=(N, CROP, CROP))

    # per-storm resume cursor (+ per-storm masked-pixel counts in raw_full:
    # they survive an interrupted run so the manifest total stays exact)
    done_sids = set(progress_path.read_text().split()) if progress_path.exists() else set()
    mask_counts_path = out / "mask_counts.json"
    mask_counts: dict = {}
    if raw_full and mask_counts_path.is_file():
        mask_counts = {str(k): int(v) for k, v in json.loads(
            mask_counts_path.read_text(encoding="utf-8")).items()}
    t0 = time.time()
    offset = 0
    written = 0
    for sid, grp in index.groupby("SID", sort=True):
        n = len(grp)
        if sid in done_sids:
            offset += n
            continue
        arr = np.load(src / f"{sid}.npy", mmap_mode="r")
        if arr.shape[0] != n:
            raise RuntimeError(f"{sid}: times/frames mismatch ({n} vs {arr.shape[0]})")
        if raw_full:
            block, n_bad = _masked_frame_block(arr)
            mm[offset:offset + n] = block
            mask_counts[str(sid)] = n_bad
        else:
            mm[offset:offset + n] = np.asarray(arr)
        offset += n
        written += 1
        # flush the storm BEFORE its resume marker: a crash can never leave a
        # storm marked done while its rows are still only in the page cache
        mm.flush()
        if raw_full:
            mask_counts_path.write_text(json.dumps(mask_counts),
                                        encoding="utf-8")
        with open(progress_path, "a") as f:
            f.write(sid + "\n")
        if written % 100 == 0:
            el = time.time() - t0
            print(f"  {written} storms written this run, offset={offset}, "
                  f"{el:.0f}s", flush=True)
    mm.flush()
    progress_path.write_text("")  # keep file; done markers above suffice

    n_pixels_masked = 0
    if raw_full:
        missing_counts = sorted(
            set(index["SID"].astype(str).unique()) - set(mask_counts))
        if missing_counts:
            raise SystemExit(
                f"{mask_counts_path}: no masked-pixel count for "
                f"{len(missing_counts)} storms ({missing_counts[:5]}); the "
                f"manifest total would be inexact - rebuild with --force")
        n_pixels_masked = int(sum(mask_counts.values()))

    offsets: dict = {}
    cursor = 0
    for sid, grp in index.groupby("SID", sort=True):
        offsets[str(sid)] = cursor
        cursor += len(grp)
    sid_audit_path, sid_records = _write_sid_audit(out, index, src, offsets)

    sids = sorted(index["SID"].astype(str).unique().tolist())
    manifest = {
        "kind": "mae_frame_cache",
        "cache_dir": str(out),
        "frames_source_dir": str(src),
        "crop": CROP,
        "n_frames": int(N),
        "n_storms": len(sids),
        "sids_sha256": sha256_array(sids),
        "index_sha256": sha256_file(index_path),
        "sid_audit": {"path": str(sid_audit_path),
                      "sha256": sha256_file(sid_audit_path)},
        "frames_sha256": sha256_file(frames_path),
        "sids_from": str(Path(args.sids_from).resolve()) if args.sids_from else None,
        "qc_rule_version": QC_RULE_VERSION if qc_audit else None,
        "qc_manifest": ({"path": str(qc_path), "sha256": sha256_file(qc_path),
                         "stores_sha256": qc_audit["stores_sha256"],
                         "n_stores": qc_audit["n_stores"]}
                        if qc_audit else None),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "scripts/merge_global_cache.py",
    }
    if raw_full:
        dup_rows = index.duplicated(subset=["SID", "time"], keep=False)
        dup_extra = index.duplicated(subset=["SID", "time"], keep="first")
        manifest.update({
            "mae_corpus": MAE_CORPUS_FULL_MULTISENSOR,
            "source_corpus": "raw_full",
            "deduplicate": False,
            "tbb_mask_range_k": [TBB_VALID_MIN_K, TBB_VALID_MAX_K],
            "n_pixels_masked": int(n_pixels_masked),
            "duplicate_times": {
                "n_rows_in_duplicate_times": int(dup_rows.sum()),
                "n_duplicate_rows": int(dup_extra.sum()),
                "n_unique_sid_times": int(N - int(dup_extra.sum())),
                "n_stores_with_duplicate_times": int(
                    index.loc[dup_extra, "SID"].astype(str).nunique()),
            },
            "source_stores_sha256": _source_stores_sha256(sid_records),
        })
    (out / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"DONE: {frames_path} ({N} frames, {len(sids)} storms, "
          f"corpus={args.source_corpus}, qc={'yes' if qc_audit else 'NO'}"
          + (f", masked_pixels={n_pixels_masked}, "
             f"duplicate_rows={manifest['duplicate_times']['n_duplicate_rows']}"
             if raw_full else "") + ")", flush=True)
    print(f"manifest sha256={sha256_json(manifest)[:16]}…", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
