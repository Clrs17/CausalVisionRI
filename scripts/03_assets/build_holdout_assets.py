# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Build the HOLDOUT_MASKED_V1 asset bundle consumed by ``src.train``.

Inputs (produced by the phase-1 holdout build, never by this script):

    <phase1>/data/processed_masked/holdout_fit0017/panel/
        wpac_tc_dataset_processed.parquet     # full transformed panel
        dv24_yj_transformer.pkl               # fitted on 2000-2017 only
        feature_scaler.pkl
        sample_manifest.csv                   # canonical (725) SID, YEAR, INCLUDED
        split_manifest.csv                    # label-eligible SID, storm_year, split
        panel_manifest.json                   # rows / storms / row-key hashes
        boundary_audit.json                   # label-aware fit row-key hash
    <phase1>/data/processed_masked/holdout_fit0017/prior/
        pcmci_results.pkl
        causal_graph_edges.json
        ate_results.csv
        identification_status.csv
        identification_dag.dot                # the DAG DoWhy actually used
        identification_dag.json               # its provenance (+ dot_sha256)
        [trusted_signs.json]                  # optional; recomputed + asserted

Outputs (asset dir ``--out-dir``, default ``data/holdout_masked_v1``; the
availability manifest goes to ``data.frame_availability``):

    panel_fit0017.parquet         # copy/hardlink of the transformed panel
    dv24_yj_transformer.pkl       # copy/hardlink
    feature_scaler.pkl            # copy/hardlink
    tbb_norm_fit0017.json         # recomputed from fit-year paired frames
    frames_index.parquet          # exact frame pairing of every eligible row
    population_audit.json         # canonical 725 vs label-eligible binding
    frame_availability.json       # every excluded row + reason (rule v1)
    prior_manifest.json           # schema consumed by src.protocol.load_prior_bundle
    trusted_signs.json            # fail-closed recomputation
    build_manifest.json           # provenance of this build

Hard failures (no fallback, no tolerance):

  * any input path inside a frozen ``archive/`` snapshot;
  * ``data.population`` != alltc, a sample manifest that is not the canonical
    725-system set (e.g. the 375-system >=65 kt sample), or a frame corpus that
    is not the configured global HURSAT corpus;
  * a canonical sample system missing from / disagreeing with the
    label-eligible split manifest, or a label-ineligible system given rows;
  * missing hash keys in the phase-1 panel/prior manifests, a graph whose
    ``metadata.mask_type`` is not xyz, an identification status that was not
    produced from the packed DAG, or a trusted-sign mismatch;
  * duplicate (SID, ISO_TIME) row keys, duplicate frame timestamps, frame /
    time-axis length mismatch.

Real missing frames are NOT a hard failure: rule ``frame_availability_v1``
excludes exactly the eligible rows whose own t0 has no exact QC frame
(``missing_store`` / ``before_first_frame`` / ``after_last_frame`` /
``internal_frame_gap``), lists every one of them with its reason in
``data.frame_availability`` (``data/holdout_masked_v1/frame_availability.json``)
and publishes counts/hashes per split/year/storm plus the alias coverage.  The
frames index holds only exactly matched rows; internal environment-history gaps
are never exclusions (the dataset/model mask them).  ``prior_manifest.json``
inline-binds that manifest so ``load_prior_bundle`` re-validates it.

Usage (from the release root):
    python scripts/03_assets/build_holdout_assets.py
    python scripts/03_assets/build_holdout_assets.py --check-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.frame_store import (frame_array_path, has_frame_store,
                             require_frame_time_path)
from wpac_warning.protocol import (FRAME_AVAILABILITY_REASONS,
                          FRAME_AVAILABILITY_RULE_VERSION,
                          POPULATION_HASH_KEYS, QC_RULE_VERSION, SPLIT_NAMES,
                          SplitProtocol, assign_rows, is_legacy_wnp_frames,
                          is_qc_hursat_frames, is_raw_hursat_frames,
                          load_trusted_signs, portable_path,
                          population_expectations, require_frames_corpus,
                          require_not_archive, require_population,
                          sample_manifest_audit, sha256_array, sha256_file,
                          sha256_json, sha256_sid_years, sha256_sids,
                          validate_frame_availability, validate_qc_manifest,
                          verify_identification_dag,
                          verify_qc_store_files)
from wpac_warning.utils_cfg import load_cfg

PROTOCOL = "holdout_masked_v1"
POPULATION_AUDIT_NAME = "population_audit.json"


class FramePairingError(RuntimeError):
    """An eligible panel row has no exactly matching frame."""

    def __init__(self, message: str, audit: dict) -> None:
        self.audit = audit
        lines = [message]
        for split in SPLIT_NAMES:
            entry = audit["splits"].get(split) or {}
            if not entry.get("n_missing"):
                continue
            lines.append(
                f"  {split}: {entry['n_missing']} missing rows "
                f"(row_key_sha256={entry['missing_row_key_sha256']})")
            for row_key in entry["missing_rows"][:10]:
                lines.append(f"    {row_key}")
            if entry["n_missing"] > 10:
                lines.append(f"    … {entry['n_missing'] - 10} more")
        lines.append("  frames_dir: " + str(audit.get("frames_dir")))
        super().__init__("\n".join(lines))


def _p2(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (P2_ROOT / p)


def _portable(path: str | Path) -> str:
    """Phase-2-relative POSIX path (absolute on a different drive)."""
    return portable_path(path, P2_ROOT)


def _require(path: Path, what: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{what} missing or empty: {path}")
    require_not_archive(path, what)
    return path


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _frame_array_len(frames_dir: Path, sid: str) -> int:
    head = np.load(frame_array_path(frames_dir, sid), mmap_mode="r")
    return int(head.shape[0])


def load_frame_time_axis(frames_dir: Path, sid: str) -> tuple:
    """(times, n_duplicate_times) for a storm store.

    The global HURSAT corpus can hold the same acquisition time twice (a
    tarball that ships an image twice).  Times must still be sorted — the
    trainer's exact ``searchsorted`` lookup depends on it — and pairing always
    takes the leftmost frame, which is the frame the trainer reads.  Duplicates
    are counted and reported, never silently dropped.
    """
    path = require_frame_time_path(frames_dir, sid)
    raw = np.load(path)
    times = np.asarray(raw, dtype="datetime64[ns]").reshape(-1)
    if times.size == 0:
        raise ValueError(f"{sid}: empty frame time axis ({path})")
    if np.isnat(times).any():
        raise ValueError(f"{sid}: NaT in the frame time axis ({path})")
    if times.size > 1:
        d = np.diff(times).astype("timedelta64[ns]").astype(np.int64)
        if (d < 0).any():
            raise ValueError(
                f"{sid}: frame time axis is not sorted ({path}); exact row "
                f"pairing and the trainer's searchsorted lookup require an "
                f"ordered axis")
        n_dup = int((d == 0).sum())
    else:
        n_dup = 0
    n = _frame_array_len(frames_dir, sid)
    if n != times.size:
        raise ValueError(
            f"{sid}: {n} frames but {times.size} frame times ({path}); the "
            f"store is inconsistent")
    return times, n_dup


def load_split_manifest(path: Path, proto: SplitProtocol) -> pd.DataFrame:
    m = pd.read_csv(_require(path, "split manifest"))
    for col in ("SID", "storm_year", "split"):
        if col not in m.columns:
            raise ValueError(f"{path} missing column '{col}'")
    if m["SID"].astype(str).duplicated().any():
        dup = sorted(m.loc[m["SID"].astype(str).duplicated(), "SID"])[:5]
        raise ValueError(f"{path} lists duplicate SIDs: {dup}")
    unknown = sorted(set(m["split"].astype(str)) - set(SPLIT_NAMES)
                     - {"excluded"})
    if unknown:
        raise ValueError(f"{path} has unknown split labels: {unknown}")
    yrs = m["storm_year"].astype(int)
    bad = m[(m["split"] == "fit") & ~yrs.between(*proto.fit_years)]
    if len(bad):
        raise ValueError(f"fit split contains {len(bad)} SIDs outside the fit "
                         f"years: {bad.head(5).to_dict('records')}")
    all_years = set(yrs.tolist())
    if max(all_years) > proto.test_years[1] or min(all_years) < proto.fit_years[0]:
        raise ValueError(f"split manifest years {sorted(all_years)[:3]}…"
                         f"{sorted(all_years)[-3:]} escape the protocol window")
    m = m.copy()
    m["SID"] = m["SID"].astype(str)
    m["storm_year"] = yrs.values
    return m


def _canonical_aliases(cfg_data: dict) -> dict:
    """canonical SID -> archive SID from the re-key crosswalk.

    A config that declares ``data.rekey_crosswalk`` must point at the file: a
    missing crosswalk would silently degrade the alias coverage to 0/0.
    """
    rel = cfg_data.get("rekey_crosswalk")
    if not rel:
        return {}
    path = _p2(rel)
    if not path.is_file():
        raise FileNotFoundError(
            f"data.rekey_crosswalk={rel} is declared but the crosswalk file is "
            f"missing ({path}); run scripts/rekey_hursat_2024.py (28/28) before "
            f"building the assets")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(e["canonical_sid"]): str(e["archive_sid"])
            for e in payload.get("entries", [])}


def _iso(value) -> str:
    return pd.Timestamp(pd.to_datetime(value, utc=True)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def build_frame_availability(eligible: pd.DataFrame, frames_dir: Path, *,
                             qc_manifest_path: Path, qc_sha: str,
                             qc_stores_sha256: str, qc_store_rows: dict,
                             qc_counts: dict, qc_stores: list,
                             canonical_aliases: dict | None = None) -> dict:
    """Rule ``frame_availability_v1``: which eligible rows have an exact frame.

    An eligible row is excluded only when the QC corpus has no frame at its own
    ``t0`` timestamp (missing store / before the first frame / after the last
    frame / an internal gap).  Environment-history gaps are never exclusions —
    the dataset and model mask them.  Every excluded row is listed with its
    reason; nothing is dropped silently and nothing else is excluded.
    """
    if eligible["row_uid"].astype(str).duplicated().any():
        raise ValueError("duplicate eligible row keys; availability is ambiguous")
    frames_root = Path(frames_dir).resolve()
    excluded: list = []
    available_rows: list = []
    missing_stores: list = []
    for sid, grp in eligible.groupby("SID", sort=False):
        sid = str(sid)
        times = None
        if has_frame_store(frames_root, sid):
            times, _n_dup = load_frame_time_axis(frames_root, sid)
            declared = qc_store_rows.get(sid)
            if declared is not None and int(declared) != int(times.size):
                raise ValueError(
                    f"{sid}: QC manifest declares {declared} frames but the "
                    f"store holds {times.size}; the QC corpus and its manifest "
                    f"disagree")
        if times is None:
            missing_stores.append(sid)
            for _, row in grp.iterrows():
                excluded.append({"row_uid": str(row["row_uid"]), "SID": sid,
                                 "ISO_TIME": _iso(row["ISO_TIME"]),
                                 "split": str(row["split"]),
                                 "reason": "missing_store",
                                 "first_frame": None, "last_frame": None})
            continue
        first = pd.Timestamp(times[0])
        last = pd.Timestamp(times[-1])
        tgt = pd.to_datetime(grp["ISO_TIME"], utc=True).dt.tz_localize(None)
        tgt_ns = tgt.values.astype("datetime64[ns]")
        pos = np.minimum(np.searchsorted(times, tgt_ns), times.size - 1)
        hit = times[pos] == tgt_ns
        for (_, row), ok, j in zip(grp.iterrows(), hit, pos):
            if ok:
                available_rows.append({
                    "SID": sid, "ISO_TIME": row["ISO_TIME"],
                    "row_uid": str(row["row_uid"]), "split": str(row["split"]),
                    "frame_time": pd.Timestamp(times[j], tz="UTC")})
                continue
            t_naive = pd.Timestamp(pd.to_datetime(row["ISO_TIME"], utc=True)
                                   .tz_localize(None))
            if t_naive < first:
                reason = "before_first_frame"
            elif t_naive > last:
                reason = "after_last_frame"
            else:
                reason = "internal_frame_gap"
            excluded.append({"row_uid": str(row["row_uid"]), "SID": sid,
                             "ISO_TIME": _iso(row["ISO_TIME"]),
                             "split": str(row["split"]), "reason": reason,
                             "first_frame": first.isoformat(),
                             "last_frame": last.isoformat()})
    available = pd.DataFrame(available_rows,
                             columns=["SID", "ISO_TIME", "row_uid", "split",
                                      "frame_time"])
    if len(available):
        available = available.sort_values(["SID", "ISO_TIME"]).reset_index(
            drop=True)
    excluded = sorted(excluded, key=lambda r: r["row_uid"])
    n_eligible = int(len(eligible))
    n_available = int(len(available))
    n_excluded = len(excluded)
    if n_eligible != n_available + n_excluded:
        raise ValueError("frame availability accounting does not add up")

    def _reason_counts(rows) -> dict:
        counts = {reason: 0 for reason in FRAME_AVAILABILITY_REASONS}
        for row in rows:
            counts[row["reason"]] += 1
        return counts

    avail_keys = set(available["row_uid"].astype(str)) if n_available else set()
    by_split = {}
    for split in SPLIT_NAMES:
        rows_split = [r for r in excluded if r["split"] == split]
        avail_split_keys = sorted(
            available.loc[available["split"] == split, "row_uid"].astype(str)
        ) if n_available else []
        by_split[split] = {
            "n_eligible": int((eligible["split"] == split).sum()),
            "n_available": len(avail_split_keys),
            "n_excluded": len(rows_split),
            "available_row_key_sha256": sha256_array(avail_split_keys),
            "excluded_row_key_sha256": sha256_array(
                sorted(r["row_uid"] for r in rows_split)),
            "by_reason": _reason_counts(rows_split),
        }

    avail_year_counts: dict = {}
    if n_available:
        for year, count in pd.to_datetime(
                available["ISO_TIME"], utc=True).dt.year.value_counts().items():
            avail_year_counts[str(int(year))] = int(count)
    year_excluded: dict = {}
    for row in excluded:
        year_excluded.setdefault(str(pd.Timestamp(row["ISO_TIME"]).year),
                                 []).append(row["row_uid"])
    by_year = {}
    for year in sorted(set(avail_year_counts) | set(year_excluded)):
        keys = sorted(year_excluded.get(year, []))
        by_year[year] = {"n_available": avail_year_counts.get(year, 0),
                         "n_excluded": len(keys),
                         "excluded_row_key_sha256": sha256_array(keys)}

    sid_avail: dict = {}
    if n_available:
        for sid, count in available["SID"].astype(str).value_counts().items():
            sid_avail[str(sid)] = int(count)
    sid_excluded: dict = {}
    for row in excluded:
        sid_excluded.setdefault(row["SID"], []).append(row)
    by_storm = {}
    for sid in sorted(set(eligible["SID"].astype(str))):
        rows_sid = sid_excluded.get(sid, [])
        n_avail = sid_avail.get(sid, 0)
        by_storm[sid] = {
            "n_eligible": n_avail + len(rows_sid),
            "n_available": n_avail,
            "n_excluded": len(rows_sid),
            "by_reason": _reason_counts(rows_sid),
            "excluded_row_key_sha256": sha256_array(
                sorted(r["row_uid"] for r in rows_sid)),
            "first_excluded": rows_sid[0]["ISO_TIME"] if rows_sid else None,
            "last_excluded": rows_sid[-1]["ISO_TIME"] if rows_sid else None,
        }

    aliases = canonical_aliases or {}
    canon_keys = set(eligible.loc[eligible["SID"].astype(str).isin(set(aliases)),
                                  "row_uid"].astype(str))
    missing_canon = sorted(canon_keys - avail_keys)
    missing_times = sorted(pd.Timestamp(k.split("|")[1]) for k in missing_canon)
    alias_coverage = {
        "crosswalk_sids": len(aliases),
        "n_canonical_rows": len(canon_keys),
        "n_covered": len(canon_keys & avail_keys),
        "ratio": (len(canon_keys & avail_keys) / len(canon_keys))
        if canon_keys else None,
        "first_missing": _iso(missing_times[0]) if missing_times else None,
        "last_missing": _iso(missing_times[-1]) if missing_times else None,
        "excluded_row_key_sha256": sha256_array(missing_canon),
    }

    manifest = {
        "protocol": PROTOCOL,
        "rule_version": FRAME_AVAILABILITY_RULE_VERSION,
        "rule": (
            "exclude an eligible holdout row only when the QC frame corpus has "
            "no image at that row's own t0 timestamp (missing_store / "
            "before_first_frame / after_last_frame / internal_frame_gap); "
            "internal environment-history gaps are not exclusions (the dataset "
            "and model mask them); no RI/DV24/model/year-based hand picking"),
        "frames_dir": _portable(frames_root),
        "qc": {
            "path": _portable(qc_manifest_path),
            "sha256": qc_sha,
            "stores_sha256": qc_stores_sha256,
        },
        "qc_decode_failures": {
            "n_stores_with_decode_failures": int(
                (qc_counts or {}).get("n_stores_with_decode_failures", 0)),
            "n_stores_with_unparsed_members": int(
                (qc_counts or {}).get("n_stores_with_unparsed_members", 0)),
            "n_unrecovered_member_times": int(
                (qc_counts or {}).get(
                    "n_unrecovered_member_times",
                    sum(len(e.get("unrecovered_member_times") or [])
                        for e in qc_stores))),
            "n_unrecovered_member_times_per_store": int(sum(
                len(e.get("unrecovered_member_times") or [])
                for e in qc_stores)),
        },
        "n_eligible_rows": n_eligible,
        "n_available_rows": n_available,
        "n_excluded_rows": n_excluded,
        "n_dropped_no_frame": n_excluded,
        "available_row_key_sha256": sha256_array(sorted(avail_keys)),
        "excluded_row_key_sha256": sha256_array(
            sorted(r["row_uid"] for r in excluded)),
        "rows": excluded,
        "by_reason": _reason_counts(excluded),
        "by_split": by_split,
        "by_year": by_year,
        "by_storm": by_storm,
        "alias_coverage": alias_coverage,
        "missing_stores": sorted(missing_stores),
    }
    return {"available": available, "manifest": manifest,
            "excluded": excluded, "frames_root": frames_root}


def build_frames_index(panel: pd.DataFrame, frames_dir: Path,
                       strict: bool = True) -> pd.DataFrame:
    """Pair every panel row to the frame with the SAME timestamp.

    Exact matching only: the trainer looks a row up in the storm's frame store
    with an exact search, so a nearest-frame pairing (+/-1.5 h) would produce
    index rows whose ``ISO_TIME`` is absent from the store and the frame would
    silently fall back to zeros.  ``dt_h`` is therefore always exactly 0.

    Hard errors: duplicate (SID, ISO_TIME) row keys, duplicate/unsorted frame
    timestamps, frame array / time-axis length mismatch.  Unmatched rows are
    returned through ``attrs['unmatched']`` and, with ``strict=True`` (the
    default), raised as :class:`FramePairingError`.
    """
    frames_root = Path(frames_dir).resolve()
    require_not_archive(frames_root, "frames directory")
    try:
        stored_root = frames_root.relative_to(P2_ROOT.resolve()).as_posix()
    except ValueError:
        stored_root = frames_root.as_posix()

    sid_col = panel["SID"].astype(str)
    t = pd.to_datetime(panel["ISO_TIME"], utc=True)
    keys = sid_col + "|" + t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if keys.duplicated().any():
        dup = sorted(keys[keys.duplicated()])[:5]
        raise ValueError(
            f"{int(keys.duplicated().sum())} duplicate (SID, ISO_TIME) row keys "
            f"cannot be paired unambiguously: {dup}")

    records = []
    unmatched = []
    missing_stores = sorted({sid for sid in sid_col.unique()
                             if not has_frame_store(frames_root, sid)})
    if missing_stores:
        raise FileNotFoundError(
            f"{len(missing_stores)} panel storms have no frame store under "
            f"{frames_root} (first: {missing_stores[:5]})")
    dup_times = {}
    for sid, grp in panel.groupby("SID", sort=False):
        times, n_dup = load_frame_time_axis(frames_root, sid)
        if n_dup:
            dup_times[str(sid)] = n_dup
        tgt = t.loc[grp.index].dt.tz_convert(None).values.astype(
            "datetime64[ns]")
        pos = np.minimum(np.searchsorted(times, tgt), times.size - 1)
        hit = times[pos] == tgt
        for idx, ok, j in zip(grp.index, hit, pos):
            if not ok:
                unmatched.append({"SID": str(sid),
                                  "ISO_TIME": t.loc[idx],
                                  "row_key": f"{sid}|"
                                             f"{t.loc[idx].strftime('%Y-%m-%dT%H:%M:%SZ')}"})
                continue
            records.append({
                "SID": str(sid), "ISO_TIME": t.loc[idx],
                "frame_path": f"{stored_root}/{sid}.npy",
                "frame_time": pd.Timestamp(times[j], tz="UTC"),
                "dt_h": 0.0,
            })
    idx_df = (pd.DataFrame(records, columns=["SID", "ISO_TIME", "frame_path",
                                             "frame_time", "dt_h"])
              .sort_values(["SID", "ISO_TIME"]).reset_index(drop=True))
    idx_df.attrs["unmatched"] = unmatched
    idx_df.attrs["n_unmatched"] = len(unmatched)
    idx_df.attrs["frames_dir"] = str(frames_root)
    idx_df.attrs["match"] = "exact"
    idx_df.attrs["duplicate_frame_times"] = dup_times
    if strict and unmatched:
        raise FramePairingError(
            f"{len(unmatched)} panel rows have no exactly matching frame",
            {"splits": {}, "frames_dir": str(frames_root),
             "unmatched": unmatched})
    return idx_df


def _resolve_stored_path(value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else (P2_ROOT / p)


def verify_frames_index(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Strictly compare the exact pairing with an existing index.

    Duplicate row keys on either side are fatal (never ``drop_duplicates``);
    common rows must agree exactly on ``frame_time``, ``dt_h`` and on which
    frame file they point at.  Rows the old index paired but the new eligible
    set does not contain are reported (they were filtered out of the protocol,
    not silently dropped); rows only the new index has are fatal.
    """
    key = ["SID", "ISO_TIME"]
    a = old.copy()
    b = new.copy()
    for frame, label in ((a, "existing"), (b, "rebuilt")):
        frame["SID"] = frame["SID"].astype(str)
        frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], utc=True)
        if frame.duplicated(key).any():
            raise ValueError(
                f"{label} frames index has duplicate (SID, ISO_TIME) row keys "
                f"({int(frame.duplicated(key).sum())}); pairing is ambiguous "
                f"and must not be resolved by dropping rows")
    a = a.set_index(key).sort_index()
    b = b.set_index(key).sort_index()
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError(
            "rebuilt frames index has no overlap with the existing index — "
            "wrong frames_dir? Move the old frames_index.parquet aside if it "
            "belongs to a different corpus/protocol")
    dt_diff = (a.loc[common, "frame_time"] - b.loc[common, "frame_time"]).abs()
    n_time_bad = int((dt_diff > pd.Timedelta(0)).sum())
    if n_time_bad:
        raise ValueError(
            f"{n_time_bad} rows pair to a different frame time than the "
            f"existing index (nearest-frame pairing detected)")
    if "dt_h" in a.columns and "dt_h" in b.columns:
        dh = (a.loc[common, "dt_h"].astype(float)
              - b.loc[common, "dt_h"].astype(float)).abs()
        n_dt_bad = int((dh > 1e-9).sum())
        if n_dt_bad:
            raise ValueError(
                f"{n_dt_bad} rows have a different dt_h than the existing "
                f"index (the old index used a nearest-frame tolerance)")
    n_basename_bad = 0
    n_corpus_bad = 0
    n_unverified = 0
    for sid, iso in common:
        old_path = _resolve_stored_path(a.at[(sid, iso), "frame_path"])
        new_path = _resolve_stored_path(b.at[(sid, iso), "frame_path"])
        if old_path.is_file() and new_path.is_file():
            if old_path.resolve() != new_path.resolve():
                n_corpus_bad += 1
        elif old_path.name != new_path.name:
            n_basename_bad += 1
        else:
            n_unverified += 1
    if n_basename_bad:
        raise ValueError(f"{n_basename_bad} rows point at a different frame "
                         f"file than the existing index")
    if n_corpus_bad:
        raise ValueError(
            f"{n_corpus_bad} rows point at a different frame corpus than the "
            f"existing index; move the old index aside and rebuild it from the "
            f"configured corpus")
    return {
        "n_rows_old": int(len(a)),
        "n_rows_new": int(len(b)),
        "n_common": int(len(common)),
        "n_new_only": int(len(b) - len(common)),
        "n_old_only": int(len(a) - len(common)),
        "n_old_path_unverified": int(n_unverified),
        "max_frame_time_diff_s": float(dt_diff.max().total_seconds())
        if len(common) else 0.0,
        "match": "exact",
    }


def compute_tbb_norm(panel: pd.DataFrame, frames_dir: Path,
                     fit_years: tuple) -> dict:
    """TBB mean/std from the exactly matching frames of eligible FIT rows."""
    if panel.empty:
        raise ValueError("no fit rows for the TBB normalisation")
    if not panel["ISO_TIME"].dt.year.between(*fit_years).all():
        raise ValueError("TBB normalisation received a row outside the fit years")
    s = 0.0
    ss = 0.0
    n = 0
    n_frames = 0
    used_sids = set()
    for sid, grp in panel.groupby("SID", sort=False):
        if not has_frame_store(frames_dir, sid):
            raise FileNotFoundError(f"fit storm {sid} has no frame store")
        times, _n_dup = load_frame_time_axis(frames_dir, sid)
        arr = np.load(frame_array_path(frames_dir, sid), mmap_mode="r")
        for t in grp["ISO_TIME"]:
            tn = np.datetime64(t.tz_convert("UTC").tz_localize(None)
                               .to_datetime64(), "ns")
            j = int(np.searchsorted(times, tn))
            if j >= times.size or times[j] != tn:
                raise ValueError(
                    f"{sid}: fit row {t} has no exactly matching frame")
            v = np.asarray(arr[j], dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            s += float(v.sum())
            ss += float((v ** 2).sum())
            n += int(v.size)
            n_frames += 1
            used_sids.add(sid)
    if n == 0:
        raise ValueError("no finite fit-year pixels for the TBB normalisation")
    mean = s / n
    var = max(ss / n - mean ** 2, 0.0)
    return {
        "mean": mean, "std": float(np.sqrt(var)), "n_pixels": int(n),
        "n_frames": int(n_frames), "n_storms": len(used_sids),
        "train_years": list(fit_years), "match_tolerance_h": 0.0,
        "source": "eligible fit-row frames (exact timestamp match)",
    }


def _require_keys(block: dict, keys, what: str) -> None:
    missing = [k for k in keys if k not in block or block[k] is None]
    if missing:
        raise ValueError(f"{what} is missing required key(s): {missing}")


def _hash64(value, where: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{where}: missing or malformed sha256 ({value!r})")
    return text


def check_panel_manifest(pm: dict, panel: pd.DataFrame, fit_elig: pd.DataFrame,
                         panel_dir: Path) -> None:
    """Every hash/provenance key the panel manifest must carry."""
    _require_keys(pm, ("protocol", "fit_years", "inner_years", "test_years",
                       "rows", "storms", "population", "sample_universe",
                       "split_counts", "hashes", "row_key_sha256_fit_eligible",
                       "boundary_audit"), "panel_manifest.json")
    if str(pm["protocol"]).lower().replace("_", "") != "holdoutmaskedv1":
        raise ValueError(f"panel_manifest protocol={pm['protocol']!r}")
    if str(pm["population"]).lower() != "alltc":
        raise ValueError(
            f"panel_manifest population={pm['population']!r}; the holdout "
            f"protocol requires alltc")
    _hash64(pm["row_key_sha256_fit_eligible"],
            "panel_manifest row_key_sha256_fit_eligible")
    if not isinstance(pm["hashes"], dict) or not pm["hashes"]:
        raise ValueError("panel_manifest 'hashes' must be a non-empty mapping")
    for name in ("sample_manifest.csv", "split_manifest.csv",
                 "wpac_tc_dataset_processed.parquet",
                 "dv24_yj_transformer.pkl", "feature_scaler.pkl",
                 "boundary_audit.json"):
        if name not in pm["hashes"]:
            raise ValueError(f"panel_manifest hashes lack '{name}'")
        path = panel_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"panel file missing: {path}")
        if sha256_file(path) != pm["hashes"][name]:
            raise ValueError(f"panel_manifest hash mismatch for {name}")
    ba = pm.get("boundary_audit") or {}
    _require_keys(ba, ("path", "sha256"), "panel_manifest boundary_audit")
    if sha256_file(panel_dir / "boundary_audit.json") != ba["sha256"]:
        raise ValueError("panel_manifest boundary_audit hash mismatch")
    if int(pm["rows"]) != len(panel):
        raise ValueError(
            f"panel_manifest rows={pm['rows']} != panel rows={len(panel)}")
    if int(pm["storms"]) != int(panel["SID"].nunique()):
        raise ValueError("panel_manifest storms != panel storms")
    universe = pm.get("sample_universe") or {}
    if "n_included" in universe:
        got = int(universe["n_included"])
        if got < int(pm["storms"]):
            raise ValueError(
                f"panel_manifest sample_universe n_included={got} < panel "
                f"storms={pm['storms']}")


def check_boundary_audit(ba: dict, n_rows: int, fit_rows: int,
                         fit_sha: str) -> None:
    _require_keys(ba, ("protocol", "panel_rows", "fit_rows_with_f24_boundary",
                       "fit_row_key_sha256", "row_uid_format"),
                  "boundary_audit.json")
    if int(ba["panel_rows"]) != int(n_rows):
        raise ValueError("boundary_audit panel_rows != panel rows")
    if int(ba["fit_rows_with_f24_boundary"]) != int(fit_rows):
        raise ValueError("boundary_audit fit rows != fit_eligible rows")
    if str(ba["fit_row_key_sha256"]).lower() != str(fit_sha).lower():
        raise ValueError("boundary_audit fit row-key hash differs from fit rows")
    if str(ba["row_uid_format"]) != "SID|%Y-%m-%dT%H:%M:%SZ":
        raise ValueError("boundary_audit row_uid_format is not the canonical "
                         "SID|%Y-%m-%dT%H:%M:%SZ")


def build_population_audit(sample: dict, split: pd.DataFrame,
                           panel: pd.DataFrame, prior_panel: pd.DataFrame,
                           assigned: pd.DataFrame, expect: dict) -> dict:
    """Bind the canonical 725 sample to the label-eligible asset sets."""
    elig_mask = split["split"].isin(SPLIT_NAMES)
    split_sids = sorted(split.loc[elig_mask, "SID"].tolist())
    split_years = {s: int(y) for s, y in zip(split["SID"], split["storm_year"])}
    foreign = sorted(set(split["SID"]) - set(sample["sids"]))
    if foreign:
        raise ValueError(
            f"{len(foreign)} split-manifest SIDs are absent from the canonical "
            f"sample manifest: {foreign[:5]}")
    year_clash = sorted(s for s in split_sids
                        if int(sample["sid_years"][s]) != split_years[s])
    if year_clash:
        raise ValueError(
            f"{len(year_clash)} split SIDs disagree with the canonical sample "
            f"YEAR: {year_clash[:5]}")
    excluded = sorted(set(sample["sids"]) - set(split_sids))
    if len(excluded) != int(expect["n_label_ineligible"]):
        raise ValueError(
            f"canonical sample has {len(sample['sids'])} systems, the split "
            f"manifest {len(split_sids)} label-eligible; that leaves "
            f"{len(excluded)} label-ineligible systems but the protocol "
            f"declares {expect['n_label_ineligible']}: {excluded[:8]}")
    if len(sample["sids"]) != int(expect["n_systems"]):
        raise ValueError(
            f"canonical sample has {len(sample['sids'])} systems, expected "
            f"{expect['n_systems']}")

    panel_sids = sorted(panel["SID"].astype(str).unique().tolist())
    panel_years = dict(zip(assigned["SID"].astype(str),
                           assigned["storm_year"].astype(int)))
    # A label-less system may still appear in the raw panel (its rows carry no
    # F24 label), but it must never own an ELIGIBLE row — that would fabricate a
    # sample for a system the canonical sample manifest says has no label.
    panel_extra = sorted(set(panel_sids) - set(split_sids))
    not_in_sample = sorted(set(panel_extra) - set(excluded))
    if not_in_sample:
        raise ValueError(
            f"{len(not_in_sample)} panel SIDs are neither label-eligible nor in "
            f"the canonical sample: {not_in_sample[:5]}")
    assigned_sid = assigned["SID"].astype(str)
    fabricated = sorted(set(assigned_sid[assigned["eligible"].values])
                        & set(panel_extra))
    if fabricated:
        bad_rows = sorted(assigned.loc[
            assigned["eligible"].values & assigned_sid.isin(fabricated).values,
            "row_uid"].astype(str))[:5]
        raise ValueError(
            f"{len(fabricated)} label-ineligible systems have eligible panel "
            f"rows ({fabricated}); a system without an F24 label must not be "
            f"given samples (first rows: {bad_rows})")
    extra_rows = assigned[assigned_sid.isin(panel_extra)]
    panel_analysis_sids = [s for s in panel_sids if s not in set(panel_extra)]
    absent = sorted(set(split_sids) - set(panel_analysis_sids))
    if absent:
        raise ValueError(
            f"{len(absent)} label-eligible systems have no panel rows: "
            f"{absent[:5]}")
    prior_sids = sorted(prior_panel["SID"].astype(str).unique().tolist())
    prior_outside = sorted(set(prior_sids) - set(split_sids))
    if prior_outside:
        raise ValueError(
            f"{len(prior_outside)} fit-prior SIDs are not label-eligible: "
            f"{prior_outside[:5]}")

    fit_sids = sorted(split.loc[split["split"] == "fit", "SID"].tolist())

    def _section(sids, years, row_keys=None) -> dict:
        block = {
            "sids": sorted(sids),
            "sid_years": {s: int(years[s]) for s in sorted(sids)},
            "sids_sha256": sha256_sids(sids),
            "sid_years_sha256": sha256_array(
                sorted(f"{s}|{int(years[s])}" for s in sids)),
            "content_sha256": sha256_sid_years(
                {s: int(years[s]) for s in sids}),
        }
        if row_keys is not None:
            block["n_rows"] = len(row_keys)
            block["row_key_sha256"] = sha256_array(sorted(row_keys))
        return block

    audit = {
        "protocol": PROTOCOL,
        "population": "alltc",
        "n_systems": len(sample["sids"]),
        "n_label_eligible": len(split_sids),
        "n_label_ineligible": len(excluded),
        "label_ineligible_sids": excluded,
        "canonical_sample": {
            "path": _portable(sample["path"]),
            "sha256": sample["sha256"],
            "n_included": sample["n_included"],
            "sids": sorted(sample["sids"]),
            "sid_years": {s: int(sample["sid_years"][s])
                          for s in sorted(sample["sids"])},
            "sids_sha256": sample["sids_sha256"],
            "sid_years_sha256": sample["sid_years_sha256"],
            "content_sha256": sample["content_sha256"],
        },
        "label_eligible": _section(split_sids, split_years),
        "split": {
            **_section(split_sids, split_years),
            "n_fit_systems": len(fit_sids),
            "split_counts": {k: int(v) for k, v in
                             split.loc[elig_mask, "split"]
                             .value_counts().items()},
        },
        "panel": {
            **_section(panel_analysis_sids, panel_years,
                       panel.loc[panel["SID"].astype(str).isin(
                           panel_analysis_sids), "SID"].astype(str) + "|"
                       + panel.loc[panel["SID"].astype(str).isin(
                           panel_analysis_sids), "ISO_TIME"].dt.strftime(
                               "%Y-%m-%dT%H:%M:%SZ")),
            "eligible_row_key_sha256": sha256_array(sorted(
                assigned.loc[assigned["eligible"], "row_uid"].astype(str))),
            "n_eligible_rows": int(assigned["eligible"].sum()),
            "label_ineligible_present": panel_extra,
            "n_label_ineligible_rows": int(len(extra_rows)),
            "label_ineligible_row_key_sha256": sha256_array(sorted(
                extra_rows["row_uid"].astype(str))),
            "n_label_ineligible_eligible_rows": 0,
        },
        "prior": {
            **_section(prior_sids, panel_years,
                       prior_panel["SID"].astype(str) + "|"
                       + prior_panel["ISO_TIME"].dt.strftime(
                           "%Y-%m-%dT%H:%M:%SZ")),
            "n_fit_systems": len(prior_sids),
        },
    }
    for key in POPULATION_HASH_KEYS:
        missing = [h for h in POPULATION_HASH_KEYS[key] if h not in audit[key]]
        if missing:
            raise ValueError(f"population audit {key} lacks {missing}")
    return audit


IMPUTATION_VAR_CANDIDATES = ("variable", "var", "feature", "column", "name")
IMPUTATION_FLAG_CANDIDATES = ("imputed", "masked", "filled", "is_imputed",
                              "imputation_mask", "mask")
# the real mask column prefix (``__IMPUTED_SST_IC``); stripped once, then the
# bare variable name is normalised.  SHRD and SHEAR_VMAX_ANGLE are DIFFERENT
# upstream features and must never be merged: they would wrongly add their
# fill counts.
IMPUTATION_COLUMN_PREFIX = "__imputed_"


def _norm_var(name: Any) -> str:
    """Normalised imputation variable token (``__IMPUTED_X`` -> ``x``)."""
    text = str(name).strip()
    if text.lower().startswith(IMPUTATION_COLUMN_PREFIX):
        text = text[len(IMPUTATION_COLUMN_PREFIX):]
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _mask_long(path: Path) -> pd.DataFrame:
    """``(row_key, variable, imputed)`` rows of an imputation mask (any layout)."""
    mask = pd.read_parquet(path)
    cols = {str(c).lower(): c for c in mask.columns}
    var_col = next((cols[c] for c in IMPUTATION_VAR_CANDIDATES if c in cols), None)
    flag_col = next((cols[c] for c in IMPUTATION_FLAG_CANDIDATES if c in cols), None)
    time_cols = [c for c in mask.columns
                 if str(c).lower() in ("iso_time", "time", "timestamp")]
    sid_col = cols.get("sid")
    if var_col is not None and flag_col is not None:
        keep = [c for c in (sid_col, time_cols[0] if time_cols else None) if c]
        frame = mask[keep + [var_col, flag_col]].copy()
        frame = frame.rename(columns={var_col: "variable", flag_col: "imputed"})
        if time_cols:
            frame = frame.rename(columns={time_cols[0]: "ISO_TIME"})
        if sid_col:
            frame = frame.rename(columns={sid_col: "SID"})
        frame["imputed"] = frame["imputed"].fillna(False).astype(bool)
        frame["variable"] = frame["variable"].map(_norm_var)
        if "SID" in frame.columns and "ISO_TIME" in frame.columns:
            t = pd.to_datetime(frame["ISO_TIME"], utc=True)
            frame["row_uid"] = (frame["SID"].astype(str) + "|"
                                + t.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        return frame[["row_uid", "variable", "imputed"]]
    if time_cols:
        skip = set(time_cols) | ({sid_col} if sid_col else set())
        flags = [c for c in mask.columns if c not in skip]
        if not flags:
            raise ValueError(f"{path}: imputation mask has no per-variable "
                             f"columns")
        id_vars = [c for c in (sid_col, time_cols[0]) if c]
        long = mask.melt(id_vars=id_vars, value_vars=flags,
                         var_name="variable", value_name="imputed")
        long = long.rename(columns={time_cols[0]: "ISO_TIME"})
        if sid_col:
            long = long.rename(columns={sid_col: "SID"})
        long["imputed"] = long["imputed"].fillna(False).astype(bool)
        long["variable"] = long["variable"].map(_norm_var)
        if "SID" in long.columns and "ISO_TIME" in long.columns:
            t = pd.to_datetime(long["ISO_TIME"], utc=True)
            long["row_uid"] = (long["SID"].astype(str) + "|"
                               + t.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        return long[["row_uid", "variable", "imputed"]]
    raise ValueError(
        f"{path}: imputation mask is neither long (variable/imputed) nor wide "
        f"(SID/time + flag columns)")


def read_imputation_mask(path: Path, row_keys: Iterable[str] | None = None
                         ) -> dict:
    """Per-variable upstream-fill counts from ``imputation_mask.parquet``.

    ``row_keys`` restricts the counts to a row universe (the fit prior panel), so
    the file's whole-panel totals stay provenance while the fit counts are what
    the PCMCI metadata reports.
    """
    long = _mask_long(Path(path))
    scoped = long
    if row_keys is not None:
        wanted = {str(k) for k in row_keys}
        scoped = long[long["row_uid"].astype(str).isin(wanted)]
    counts = {str(v): int(n) for v, n in
              scoped.groupby("variable")["imputed"].sum().items()}
    manifest_path = Path(path).with_name(Path(path).stem + ".manifest.json")
    return {"path": Path(path), "counts": counts,
            "n_cells": int(len(scoped)),
            "n_masked_cells": int(sum(counts.values())),
            "full_counts": {str(v): int(n) for v, n in
                            long.groupby("variable")["imputed"].sum().items()},
            "full_n_cells": int(len(long)),
            "manifest": manifest_path if manifest_path.is_file() else None}


def _imputation_block(metadata: dict) -> tuple:
    """(path label, nested upstream-imputation dict) from the PCMCI metadata."""
    nested = metadata.get("upstream_imputation")
    if isinstance(nested, dict) and isinstance(
            nested.get("input_cells_by_variable"), dict):
        return "metadata.upstream_imputation", nested
    for key, value in metadata.items():
        if isinstance(value, dict) and isinstance(
                value.get("input_cells_by_variable"), dict):
            return f"metadata.{key}", value
    for key, value in metadata.items():
        if isinstance(value, dict):
            for sub, inner in value.items():
                if isinstance(inner, dict) and isinstance(
                        inner.get("input_cells_by_variable"), dict):
                    return f"metadata.{key}.{sub}", inner
    return None, None


def check_imputation_counts(metadata: dict, full: dict, fit: dict,
                            where: str) -> dict:
    """Reconcile the PCMCI metadata's fit fill counts with the mask file.

    The authoritative numbers are ``upstream_imputation.input_cells_by_variable``
    (cells that entered the fit prior) and they are compared with the mask
    restricted to the fit prior row keys.  The strong comparison covers only the
    variables the causal prior actually used (``upstream_imputation.used_columns``
    / ``metadata.used_columns``) plus the keys of ``input_cells_by_variable``; the
    mask's fills for non-causal variables (or any other file column) must not
    create a false difference and are recorded as ignored.  ``panel_cells_by_variable``
    (re-indexed panel counts) and the file's whole-panel totals are recorded but
    never held equal to the fit counts.
    """
    label, block = _imputation_block(metadata or {})
    if block is None:
        raise ValueError(
            f"{where}: the PCMCI metadata does not expose "
            f"upstream_imputation.input_cells_by_variable; the imputation mask "
            f"cannot be reconciled (metadata keys: "
            f"{sorted(metadata or {})[:8]})")
    input_counts = {_norm_var(k): int(v)
                    for k, v in block["input_cells_by_variable"].items()}
    fit_counts = {_norm_var(k): int(v) for k, v in fit["counts"].items()}
    used_raw = (block.get("used_columns") or (metadata or {}).get("used_columns")
                or [])
    used = sorted({_norm_var(c) for c in used_raw})
    strong = set(input_counts) | set(used)
    diffs = {v: (input_counts.get(v, 0), fit_counts.get(v, 0))
             for v in sorted(strong)
             if input_counts.get(v, 0) != fit_counts.get(v, 0)}
    if diffs:
        raise ValueError(
            f"{where}: PCMCI metadata {label}.input_cells_by_variable disagrees "
            f"with the fit-filtered imputation mask (metadata, file): {diffs}")
    ignored = {v: int(c) for v, c in fit_counts.items()
               if v not in strong and int(c)}
    panel_counts = block.get("panel_cells_by_variable")
    panel_match = None
    if isinstance(panel_counts, dict):
        panel_counts = {_norm_var(k): int(v) for k, v in panel_counts.items()}
        if panel_counts == {v: int(fit["counts"].get(v, 0))
                            for v in fit_counts}:
            panel_match = "fit_rows"
        elif panel_counts == {v: int(full["full_counts"].get(v, 0))
                              for v in full["full_counts"]}:
            panel_match = "full_panel"
    return {"metadata_key": f"{label}.input_cells_by_variable",
            "fit_counts": fit_counts,
            "strong_variables": sorted(strong),
            "used_columns": used,
            "ignored_file_variables": ignored,
            "full_panel_counts": {k: int(v) for k, v in full["full_counts"].items()},
            "n_fit_masked_cells": int(sum(fit_counts.values())),
            "n_full_masked_cells": int(sum(full["full_counts"].values())),
            "panel_cells_by_variable": (
                {k: int(v) for k, v in panel_counts.items()}
                if isinstance(panel_counts, dict) else None),
            "panel_cells_match": panel_match}


def check_phase1_prior_manifest(pm: dict, base: Path) -> None:
    """Required keys + every hash entry of the phase-1 prior manifest."""
    _require_keys(pm, ("protocol", "fit_years", "inner_years", "test_years",
                       "sids", "sid_years", "n_storms", "population", "rows",
                       "row_key_sha256", "row_uid_format", "max_feature_time",
                       "max_label_time", "mask_type", "config_snapshot",
                       "panel_manifest", "boundary_audit", "hashes", "inputs",
                       "outputs"), "phase-1 prior manifest")
    if str(pm["population"]).lower() != "alltc":
        raise ValueError(
            f"phase-1 prior manifest population={pm['population']!r}; the "
            f"holdout protocol requires alltc")
    if str(pm["mask_type"]) != "xyz":
        raise ValueError("phase-1 prior manifest mask_type != xyz")
    _hash64(pm["row_key_sha256"], "phase-1 prior manifest row_key_sha256")
    for section in ("hashes", "inputs", "outputs"):
        if not isinstance(pm[section], dict) or not pm[section]:
            raise ValueError(f"phase-1 prior manifest '{section}' must be a "
                             f"non-empty mapping")
        for rel, want in pm[section].items():
            path = Path(rel)
            if not path.is_absolute():
                path = base / path
            if not path.is_file():
                raise FileNotFoundError(
                    f"phase-1 prior manifest {section} entry missing: {path}")
            if sha256_file(path) != want:
                raise ValueError(
                    f"phase-1 prior manifest {section} hash mismatch: {rel}")
    for name in ("panel/sample_manifest.csv", "panel/split_manifest.csv",
                 "panel/population_audit.json", "panel/extraction_audit.json",
                 "panel/imputation_mask.parquet",
                 "panel/wpac_tc_dataset_processed.parquet",
                 "panel/dv24_yj_transformer.pkl", "panel/feature_scaler.pkl",
                 "prior/identification_dag.dot",
                 "prior/identification_dag.json",
                 "prior/fit_prior_panel.parquet"):
        if name not in pm["hashes"]:
            raise ValueError(f"phase-1 prior manifest hashes lack '{name}'")
    _require_keys(pm["config_snapshot"], ("path", "sha256"),
                  "phase-1 prior manifest config_snapshot")
    _require_keys(pm["panel_manifest"], ("path", "sha256"),
                  "phase-1 prior manifest panel_manifest")
    _require_keys(pm["boundary_audit"], ("path", "sha256"),
                  "phase-1 prior manifest boundary_audit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P2_ROOT / "configs" / "phase2" /
                                            "holdout_masked_v1.yaml"))
    ap.add_argument("--phase1-root",
                    default=str(P2_ROOT / "data" / "phase1"))
    ap.add_argument("--holdout-rel", default="data/processed_masked/holdout_fit0017")
    ap.add_argument("--out-dir", default=str(P2_ROOT / "data" /
                                             "holdout_masked_v1"))
    ap.add_argument("--frames-dir", default=None,
                    help="frame corpus; must equal data.frames_dir from "
                         "--config (the configured global HURSAT corpus)")
    ap.add_argument("--frames-index", default=None,
                    help="existing index to verify against "
                         "(default: <out-dir>/frames_index.parquet if present)")
    ap.add_argument("--check-only", action="store_true",
                    help="validate existing outputs without writing")
    ap.add_argument("--force", action="store_true",
                    help="explicitly replace a previous asset bundle")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    cfg_data = cfg.to_dict().get("data", {})
    proto = SplitProtocol.from_cfg(cfg)
    proto.validate()
    proto.require_alltc()
    population = require_population(cfg_data)
    require_frames_corpus(cfg_data)
    expect = population_expectations(cfg_data)

    declared = cfg_data.get("frames_dir")
    if not declared:
        raise ValueError(f"{args.config} does not set data.frames_dir")
    declared_frames = _p2(declared).resolve()
    if args.frames_dir:
        frames_dir = Path(args.frames_dir).resolve()
        if frames_dir != declared_frames:
            raise ValueError(
                f"--frames-dir {frames_dir} is not the corpus declared by "
                f"{args.config} ({declared_frames}); the protocol only reads "
                f"the configured global HURSAT corpus")
    else:
        frames_dir = declared_frames
    require_not_archive(frames_dir, "frames directory")
    if is_legacy_wnp_frames(frames_dir):
        raise ValueError(
            f"frames_dir {frames_dir} is the legacy WNP-only store; the "
            f"all-TC protocol reads the global HURSAT QC corpus ({declared})")
    if is_raw_hursat_frames(frames_dir):
        raise ValueError(
            f"frames_dir {frames_dir} is the RAW download corpus; the protocol "
            f"trains on the QC v1 corpus (scripts/build_hursat_qc.py)")
    if not is_qc_hursat_frames(frames_dir):
        raise ValueError(
            f"frames_dir {frames_dir} is not the QC v1 corpus "
            f"(data/global_hursat/frames_qc)")

    # ---- QC v1 binding: the frame corpus must be exactly the audited one ----
    qc_declared = cfg_data.get("qc_manifest")
    if not qc_declared:
        raise ValueError(f"{args.config} does not set data.qc_manifest; the "
                         f"frame corpus provenance cannot be bound")
    qc_manifest_path = _require(_p2(qc_declared), "QC manifest")
    qc_block = json.loads(qc_manifest_path.read_text(encoding="utf-8"))
    qc_audit = validate_qc_manifest(qc_block, expected_out_frames=frames_dir)
    qc_sha = sha256_file(qc_manifest_path)
    qc_sids = set(qc_audit["sids"])
    # re-hash every store on disk (arrays, time axes, and the QC inputs when
    # still present): a tampered or duplicated corpus is refused here
    qc_store_audit = verify_qc_store_files(
        qc_block, frames_dir, expected_out_frames=frames_dir)


    p1 = Path(args.phase1_root).resolve() / args.holdout_rel
    require_not_archive(p1, "phase-1 holdout root")
    panel_dir, prior_dir = p1 / "panel", p1 / "prior"
    out = Path(args.out_dir).resolve()
    require_not_archive(out, "asset output directory")

    fav_declared = cfg_data.get("frame_availability")
    if not fav_declared:
        raise ValueError(f"{args.config} does not set data.frame_availability; "
                         f"the row-exclusion manifest has no configured path")
    availability_path = _p2(fav_declared).resolve()
    require_not_archive(availability_path, "frame availability manifest")

    src = {
        "panel": _require(panel_dir / "wpac_tc_dataset_processed.parquet",
                          "transformed panel"),
        "transformer": _require(panel_dir / "dv24_yj_transformer.pkl",
                                "DV24 transformer"),
        "scaler": _require(panel_dir / "feature_scaler.pkl", "feature scaler"),
        "sample_manifest": _require(panel_dir / "sample_manifest.csv",
                                    "sample manifest"),
        "split_manifest": _require(panel_dir / "split_manifest.csv",
                                   "split manifest"),
        "panel_manifest": _require(panel_dir / "panel_manifest.json",
                                   "panel manifest"),
        "boundary_audit": _require(panel_dir / "boundary_audit.json",
                                   "boundary audit"),
        "imputation_mask": _require(panel_dir / "imputation_mask.parquet",
                                    "upstream imputation mask"),
        "prior_manifest": _require(
            prior_dir / "phase1_prior_manifest.json",
            "phase-1 prior manifest")
        if (prior_dir / "phase1_prior_manifest.json").is_file()
        else _require(prior_dir / "prior_manifest.json",
                      "phase-1 prior manifest (legacy name)"),
        "prior_panel": _require(prior_dir / "fit_prior_panel.parquet",
                                "fit-only prior panel"),
        "graph_edges": _require(prior_dir / "causal_graph_edges.json",
                                "prior graph"),
        "ate": _require(prior_dir / "ate_results.csv", "prior ATE"),
        "pcmci": _require(prior_dir / "pcmci_results.pkl", "prior PCMCI"),
        "identification": _require(prior_dir / "identification_status.csv",
                                   "prior identification"),
        "identification_dag_dot": _require(prior_dir / "identification_dag.dot",
                                           "identification DAG (.dot)"),
        "identification_dag_json": _require(
            prior_dir / "identification_dag.json",
            "identification DAG provenance (.json)"),
    }

    sample = sample_manifest_audit(src["sample_manifest"],
                                   expect_n_included=expect["n_systems"])
    panel = pd.read_parquet(src["panel"])
    panel["SID"] = panel["SID"].astype(str)
    panel["ISO_TIME"] = pd.to_datetime(panel["ISO_TIME"], utc=True)
    if proto.label_time_col not in panel.columns:
        raise ValueError(f"panel missing {proto.label_time_col}")
    panel[proto.label_time_col] = pd.to_datetime(panel[proto.label_time_col],
                                                 utc=True)
    split = load_split_manifest(src["split_manifest"], proto)
    panel_manifest = json.loads(src["panel_manifest"].read_text(encoding="utf-8"))
    prior_manifest = json.loads(src["prior_manifest"].read_text(encoding="utf-8"))
    boundary_audit = json.loads(src["boundary_audit"].read_text(encoding="utf-8"))

    # Two distinct fit sets (they differ by the systems whose rows are all lost
    # to the label boundary; the real 0024 panel has 523 nominal / 522 with rows):
    #   fit_split_sids -- the split manifest's fit SIDs (nominal, split audit)
    #   fit_sids       -- the fit SIDs that actually own a fit-eligible row
    #                     (= the phase-1 prior panel and the asset manifest)
    fit_split_sids = sorted(split.loc[split["split"] == "fit", "SID"].tolist())
    if not fit_split_sids:
        raise ValueError("split manifest has no fit SIDs")
    # A storm without a QC store is *real* missing data, not a reason to abort:
    # frame_availability_v1 excludes its rows with reason=missing_store and lists
    # every one of them (plus the SID in `missing_stores`).  It is still loud.
    missing_qc = sorted(set(panel["SID"]) - qc_sids)
    if missing_qc:
        print(f"WARNING: {len(missing_qc)} panel SIDs have no store in the QC "
              f"corpus; all their eligible rows will be excluded as "
              f"missing_store: {missing_qc[:5]}", flush=True)
    sid_year = dict(zip(split["SID"], split["storm_year"].astype(int)))
    missing = [s for s in panel["SID"].unique() if s not in sid_year]
    if missing:
        raise ValueError(f"{len(missing)} panel SIDs absent from the split "
                         f"manifest: {missing[:5]}")

    # prior-panel provenance: fit rows only, label time strictly < inner start
    fit_panel = panel[panel["SID"].isin(fit_split_sids)].copy()
    fit_panel["storm_year"] = fit_panel["SID"].map(sid_year)
    fit_panel = assign_rows(fit_panel, proto)
    fit_elig = fit_panel[fit_panel["eligible"]]
    fit_sids = sorted(fit_elig["SID"].astype(str).unique().tolist())
    if not fit_sids:
        raise ValueError("no fit SID owns a fit-eligible row")
    fit_without_rows = sorted(set(fit_split_sids) - set(fit_sids))
    bad_years = fit_elig["ISO_TIME"].dt.year[
        ~fit_elig["ISO_TIME"].dt.year.between(*proto.fit_years)]
    if len(bad_years):
        raise ValueError(f"{len(bad_years)} prior rows fall outside the fit years")
    inner_start = pd.Timestamp(f"{proto.inner_years[0]}-01-01", tz="UTC")
    if (fit_elig[proto.label_time_col] >= inner_start).any():
        raise ValueError("prior rows have label times at/after the inner split")

    # the phase-1 prior panel itself must be fit-only, label-safe and complete
    prior_panel = pd.read_parquet(src["prior_panel"])
    prior_panel["SID"] = prior_panel["SID"].astype(str)
    prior_panel["ISO_TIME"] = pd.to_datetime(prior_panel["ISO_TIME"], utc=True)
    if "storm_year" not in prior_panel.columns:
        raise ValueError("fit-only prior panel lacks storm_year")
    if not prior_panel["storm_year"].astype(int).between(*proto.fit_years).all():
        raise ValueError("fit-only prior panel contains SIDs outside fit")
    if not prior_panel["ISO_TIME"].dt.year.between(*proto.fit_years).all():
        raise ValueError("fit-only prior panel contains rows outside fit years")
    if proto.label_time_col in prior_panel.columns:
        lt = pd.to_datetime(prior_panel[proto.label_time_col], utc=True)
        if (lt >= inner_start).any():
            raise ValueError("fit-only prior panel has label times reaching "
                             "the inner split")
    prior_panel["row_uid"] = (prior_panel["SID"] + "|"
                              + prior_panel["ISO_TIME"].dt.strftime(
                                  "%Y-%m-%dT%H:%M:%SZ"))
    fit_panel_year = dict(zip(fit_panel["SID"], fit_panel["storm_year"]))
    if set(prior_panel["SID"]) - set(fit_panel_year):
        raise ValueError("fit-only prior panel contains a non-fit SID")

    # ---- hard consistency: prior rows == phase-1 prior rows == fit_elig ----
    fit_uid = sorted(fit_elig["row_uid"].astype(str).tolist())
    prior_uid = sorted(prior_panel["row_uid"].astype(str).tolist())
    fit_uid_sha = sha256_array(fit_uid)
    prior_uid_sha = sha256_array(prior_uid)
    if len(prior_panel) != len(fit_elig):
        raise ValueError(
            f"fit-only prior panel has {len(prior_panel)} rows but the phase-2 "
            f"fit-eligible set has {len(fit_elig)}; the prior row set differs "
            f"from the training boundary")
    if prior_uid_sha != fit_uid_sha:
        raise ValueError("prior panel and fit-eligible row-key hashes differ")
    if int(prior_manifest.get("rows", -1)) != len(fit_elig):
        raise ValueError(
            f"phase-1 prior manifest rows={prior_manifest.get('rows')} != "
            f"fit_eligible rows={len(fit_elig)}")
    if int(prior_manifest.get("n_storms", -1)) != len(fit_sids):
        raise ValueError("phase-1 prior manifest n_storms != fit SIDs")
    if sorted(prior_manifest.get("sids", [])) != fit_sids:
        raise ValueError("phase-1 prior manifest SIDs != split-manifest fit SIDs")
    if prior_manifest.get("row_key_sha256") != fit_uid_sha:
        raise ValueError(
            "phase-1 prior manifest row_key_sha256 differs from the fit rows")
    sid_years_phase1 = dict(zip(prior_manifest.get("sids", []),
                                prior_manifest.get("sid_years", [])))
    if any(int(sid_years_phase1.get(s, -1)) != int(sid_year[s])
           for s in fit_sids):
        raise ValueError("phase-1 prior manifest sid_years disagree with the "
                         "split manifest")

    # phase-1 manifests: required keys, hash completeness, DAG binding
    check_panel_manifest(panel_manifest, panel, fit_elig, panel_dir)
    check_boundary_audit(boundary_audit, len(panel), len(fit_elig), fit_uid_sha)
    check_phase1_prior_manifest(prior_manifest, p1)
    dag_audit = verify_identification_dag(src["identification_dag_dot"],
                                         src["identification_dag_json"],
                                         src["identification"])

    graph = json.loads(src["graph_edges"].read_text(encoding="utf-8"))
    if graph.get("metadata", {}).get("mask_type") != "xyz":
        raise ValueError("prior graph metadata.mask_type must be 'xyz'")

    # upstream imputation mask: the PCMCI metadata's FIT fill counts must
    # reconcile with the mask restricted to the fit prior rows (the file's
    # whole-panel totals are provenance only), and a mask-less prior is refused
    prior_keys = set(prior_panel["row_uid"].astype(str))
    imputation = read_imputation_mask(src["imputation_mask"])
    imputation_fit = read_imputation_mask(src["imputation_mask"],
                                          row_keys=prior_keys)
    imputation_audit = check_imputation_counts(
        graph.get("metadata") or {}, imputation, imputation_fit,
        f"prior graph metadata ({src['graph_edges']})")
    imputation_sha = sha256_file(src["imputation_mask"])
    if imputation["manifest"] is not None:
        mask_manifest = json.loads(
            imputation["manifest"].read_text(encoding="utf-8"))
        for key in ("content_sha256", "output_sha256", "parquet_sha256",
                    "sha256"):
            if mask_manifest.get(key):
                if str(mask_manifest[key]).lower() != imputation_sha:
                    raise ValueError(
                        f"{imputation['manifest']} {key} disagrees with the "
                        f"imputation mask file")
                break
        else:
            raise ValueError(
                f"{imputation['manifest']} does not pin the imputation mask "
                f"content hash")

    signs, audit = load_trusted_signs(src["ate"], src["identification"],
                                      exclude=("D200",))
    trusted_path = prior_dir / "trusted_signs.json"
    if trusted_path.is_file():
        old = json.loads(trusted_path.read_text(encoding="utf-8"))
        old_signs = {k: float(v) for k, v in old.get("signs", old).items()}
        if old_signs != {k: float(v) for k, v in signs.items()}:
            raise ValueError(
                "trusted_signs.json disagrees with the fail-closed "
                f"recomputation: file={old_signs} recomputed={signs}")
    if not signs:
        raise ValueError("no trusted treatment survived the fail-closed filter")

    # ---- population binding: canonical 725 vs label-eligible sets ----
    assigned = assign_rows(panel, proto)
    population_audit = build_population_audit(sample, split, panel, prior_panel,
                                              assigned, expect)

    # ---- frame availability v1: exclude only rows without an exact QC frame --
    eligible = assigned[assigned["eligible"]][["SID", "ISO_TIME", "split",
                                               "row_uid"]]
    availability = build_frame_availability(
        eligible, frames_dir, qc_manifest_path=qc_manifest_path, qc_sha=qc_sha,
        qc_stores_sha256=qc_audit["stores_sha256"],
        qc_store_rows={e["sid"]: int(e["output_rows"])
                       for e in qc_block["stores"]},
        qc_counts=qc_audit["counts"], qc_stores=qc_block["stores"],
        canonical_aliases=_canonical_aliases(cfg_data))
    available = availability["available"]
    idx = build_frames_index(available, frames_dir, strict=True)
    if len(idx) != len(available):
        raise ValueError(
            f"frames index lost rows: {len(idx)} paired of {len(available)} "
            f"available eligible rows")

    verify_stats = {}
    old_idx_path = (Path(args.frames_index).resolve() if args.frames_index
                    else (out / "frames_index.parquet"))
    if old_idx_path.is_file():
        verify_stats = verify_frames_index(pd.read_parquet(old_idx_path), idx)
        if verify_stats["n_new_only"]:
            raise ValueError(
                f"rebuilt frame row keys are absent from the verified index: "
                f"{verify_stats}")

    # per-split accounting: eligible == available + excluded (never silent)
    idx_uid = set(idx["SID"].astype(str) + "|"
                  + pd.to_datetime(idx["ISO_TIME"], utc=True).dt.strftime(
                      "%Y-%m-%dT%H:%M:%SZ"))
    split_stats = {}
    availability_manifest = availability["manifest"]
    for sp in SPLIT_NAMES:
        elig = assigned[(assigned["split"] == sp) & assigned["eligible"]]
        elig_keys = set(elig["row_uid"].astype(str))
        paired = elig_keys & idx_uid
        dropped = elig_keys - paired
        declared = availability_manifest["by_split"][sp]
        if len(paired) != declared["n_available"] or len(dropped) != \
                declared["n_excluded"]:
            raise ValueError(
                f"{sp}: frames index/availability disagree "
                f"(paired={len(paired)} vs {declared['n_available']}, "
                f"dropped={len(dropped)} vs {declared['n_excluded']})")
        split_stats[sp] = {
            "n_eligible": len(elig_keys),
            "n_available": len(paired),
            "n_paired": len(paired),
            "n_excluded_no_frame": len(dropped),
            "n_dropped_no_frame": len(dropped),
            "by_reason": declared["by_reason"],
            "eligible_row_key_sha256": sha256_array(sorted(elig_keys)),
            "paired_row_key_sha256": sha256_array(sorted(paired)),
            "dropped_row_key_sha256": sha256_array(sorted(dropped)),
        }
    n_panel_rows = len(panel)
    n_paired = len(idx)
    n_eligible = int(len(eligible))
    n_excluded = int(len(availability["excluded"]))
    if n_paired + n_excluded != n_eligible:
        raise ValueError(
            f"available ({n_paired}) + excluded ({n_excluded}) != eligible "
            f"({n_eligible})")

    # self-check the manifest against the exact eligible/index row sets before
    # anything is written (all missing rows and only those rows)
    availability_audit = validate_frame_availability(
        availability_manifest,
        expected_qc={"sha256": qc_sha,
                     "stores_sha256": qc_audit["stores_sha256"]},
        eligible_row_uids=eligible["row_uid"].astype(str),
        index_row_uids=idx_uid)

    if args.check_only:
        print(json.dumps({"fit_sids": len(fit_sids),
                          "nominal_fit_sids": len(fit_split_sids),
                          "fit_without_eligible_rows": fit_without_rows,
                          "population": population_audit,
                          "phase1_rows": len(panel),
                          "prior_rows": len(prior_panel),
                          "fit_elig_rows": len(fit_elig),
                          "row_key_sha256": fit_uid_sha,
                          "available_rows": n_paired,
                          "excluded_rows": n_excluded,
                          "eligible_rows": n_eligible,
                          "frame_availability": availability_audit,
                          "split_stats": split_stats,
                          "verify": verify_stats,
                          "identification_dag": dag_audit,
                          "trusted_signs": signs}, indent=2, default=str))
        return 0

    if (out / "prior_manifest.json").exists() and not args.force:
        raise FileExistsError(f"asset bundle already exists at {out}; pass --force to rebuild")
    out.mkdir(parents=True, exist_ok=True)
    _link_or_copy(src["panel"], out / "panel_fit0017.parquet")
    _link_or_copy(src["transformer"], out / "dv24_yj_transformer.pkl")
    _link_or_copy(src["scaler"], out / "feature_scaler.pkl")
    idx.to_parquet(out / "frames_index.parquet", index=False)

    audit_path = out / POPULATION_AUDIT_NAME
    audit_path.write_text(json.dumps(population_audit, indent=2, sort_keys=True),
                          encoding="utf-8")

    # bind the population audit into the row-exclusion manifest (same chain the
    # prior manifest publishes: prior -> availability -> population/QC)
    availability_manifest["population_audit"] = {
        "path": _portable(audit_path),
        "sha256": sha256_file(audit_path),
        "n_systems": population_audit["n_systems"],
        "n_label_eligible": population_audit["n_label_eligible"],
    }
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    availability_path.write_text(
        json.dumps(availability_manifest, indent=2, sort_keys=True),
        encoding="utf-8")
    availability_sha = sha256_file(availability_path)

    fit_available = available[available["split"] == "fit"]
    norm = compute_tbb_norm(fit_available, frames_dir, proto.fit_years)
    norm_path = out / "tbb_norm_fit0017.json"
    norm_path.write_text(json.dumps(norm, indent=2, sort_keys=True),
                         encoding="utf-8")

    trusted_out = {
        "protocol": PROTOCOL,
        "signs": signs,
        "rows": audit,
        "recomputed_from": {
            "ate": _portable(src["ate"]),
            "identification": _portable(src["identification"]),
            "identification_dag_dot": _portable(src["identification_dag_dot"]),
        },
    }
    (out / "trusted_signs.json").write_text(
        json.dumps(trusted_out, indent=2, sort_keys=True), encoding="utf-8")

    files = {
        "panel": {"path": _portable(out / "panel_fit0017.parquet"),
                  "sha256": sha256_file(out / "panel_fit0017.parquet")},
        "graph_edges": {"path": _portable(src["graph_edges"]),
                        "sha256": sha256_file(src["graph_edges"])},
        "ate": {"path": _portable(src["ate"]),
                "sha256": sha256_file(src["ate"])},
        "pcmci": {"path": _portable(src["pcmci"]),
                  "sha256": sha256_file(src["pcmci"])},
        "identification": {"path": _portable(src["identification"]),
                           "sha256": sha256_file(src["identification"])},
        "prior_panel": {"path": _portable(src["prior_panel"]),
                        "sha256": sha256_file(src["prior_panel"]),
                        "fit_rows": int(len(prior_panel)),
                        "row_key_sha256": sha256_array(
                            sorted(prior_panel["row_uid"].tolist()))},
    }

    assets = {
        "dv24_transformer": {
            "path": _portable(out / "dv24_yj_transformer.pkl"),
            "sha256": sha256_file(out / "dv24_yj_transformer.pkl"),
            "fit_years": list(proto.fit_years),
            "provenance": _portable(src["transformer"]),
        },
        "feature_scaler": {
            "path": _portable(out / "feature_scaler.pkl"),
            "sha256": sha256_file(out / "feature_scaler.pkl"),
            "fit_years": list(proto.fit_years),
            "provenance": _portable(src["scaler"]),
        },
        "tbb_norm": {
            "path": _portable(norm_path),
            "sha256": sha256_file(norm_path),
            "fit_years": list(proto.fit_years),
            "provenance": norm["source"],
        },
        "frames_index": {
            "path": _portable(out / "frames_index.parquet"),
            "sha256": sha256_file(out / "frames_index.parquet"),
            "fit_years": list(proto.fit_years),
            "provenance": _portable(frames_dir),
            "qc_manifest_sha256": qc_sha,
            "qc_rule_version": QC_RULE_VERSION,
        },
        "population_audit": {
            "path": _portable(audit_path),
            "sha256": sha256_file(audit_path),
            "population": population,
            "n_systems": population_audit["n_systems"],
            "n_label_eligible": population_audit["n_label_eligible"],
            "provenance": _portable(src["sample_manifest"]),
        },
        "frame_availability": {
            "path": _portable(availability_path),
            "sha256": availability_sha,
            "rule_version": FRAME_AVAILABILITY_RULE_VERSION,
            "n_eligible_rows": availability_manifest["n_eligible_rows"],
            "n_available_rows": availability_manifest["n_available_rows"],
            "n_excluded_rows": availability_manifest["n_excluded_rows"],
            "excluded_row_key_sha256":
                availability_manifest["excluded_row_key_sha256"],
        },
    }
    population_block = {
        **population_audit,
        "path": _portable(audit_path),
        "sha256": sha256_file(audit_path),
    }
    manifest = {
        "protocol": PROTOCOL,
        "years": list(range(proto.fit_years[0], proto.fit_years[1] + 1)),
        "mask_type": "xyz",
        "n_storms": len(fit_sids),
        "n_nominal_fit_sids": len(fit_split_sids),
        "fit_scope": {
            "note": "fit_split_sids is the split manifest's fit set (nominal); "
                    "prior_sids is the set that owns at least one fit-eligible "
                    "row (identical to the phase-1 prior panel). A fit SID in "
                    "the first set but not the second owns no eligible row: no "
                    "row is moved back into fit and no storm_year is changed.",
            "fit_split_sids": fit_split_sids,
            "n_fit_split_sids": len(fit_split_sids),
            "prior_sids": fit_sids,
            "n_prior_sids": len(fit_sids),
            "sid_years": {s: int(sid_year[s]) for s in fit_sids},
            "without_eligible_rows": [
                {"SID": sid, "storm_year": int(sid_year[sid]),
                 "panel_rows": int((panel["SID"] == sid).sum()),
                 "n_eligible_rows": 0}
                for sid in fit_without_rows],
        },
        "n_rows": int(len(fit_elig)),
        # fit rows that actually have an exact QC frame (eligible AFTER
        # frame_availability_v1); n_rows is the fit-eligible count before it
        "fit_rows_after_availability":
            int(availability_manifest["by_split"]["fit"]["n_available"]),
        "fit_eligible_row_count": int(len(fit_elig)),
        "n_eligible_rows_before_availability": int(n_eligible),
        "prior_rows": int(len(prior_panel)),
        "phase1_rows": int(len(panel)),
        "fit_elig_rows": int(len(fit_elig)),
        "n_panel_rows": int(n_panel_rows),
        "n_eligible_rows": int(n_eligible),
        "n_available_rows": int(n_paired),
        "n_paired_rows": int(n_paired),
        "n_excluded_no_frame": int(n_excluded),
        "n_dropped_no_frame": int(n_excluded),
        "n_excluded_by_reason": {
            k: int(v) for k, v in availability_manifest["by_reason"].items()},
        "sids": fit_sids,
        "sid_years": [int(sid_year[s]) for s in fit_sids],
        "row_key_sha256": fit_uid_sha,
        "prior_row_key_sha256": prior_uid_sha,
        "fit_elig_row_key_sha256": fit_uid_sha,
        "split_stats": split_stats,
        "population": population_block,
        "frame_availability": {
            "path": _portable(availability_path),
            "sha256": availability_sha,
            "rule_version": FRAME_AVAILABILITY_RULE_VERSION,
            "n_eligible_rows": availability_manifest["n_eligible_rows"],
            "n_available_rows": availability_manifest["n_available_rows"],
            "n_excluded_rows": availability_manifest["n_excluded_rows"],
            "excluded_row_key_sha256":
                availability_manifest["excluded_row_key_sha256"],
            "available_row_key_sha256":
                availability_manifest["available_row_key_sha256"],
        },
        "frames_qc": {
            "path": _portable(qc_manifest_path),
            "sha256": qc_sha,
            "qc_rule_version": QC_RULE_VERSION,
            "n_stores": qc_audit["n_stores"],
            "stores_sha256": qc_audit["stores_sha256"],
            "sids_sha256": qc_audit["sids_sha256"],
            "out_frames_dir": qc_audit["out_frames_dir"],
            "corpus": require_frames_corpus(cfg_data),
            "n_stores_with_decode_failures":
                qc_audit["n_stores_with_decode_failures"],
            "n_stores_with_unparsed_members":
                qc_audit["n_stores_with_unparsed_members"],
            "n_unrecovered_member_times":
                qc_audit["n_unrecovered_member_times"],
            "production": qc_audit["production"],
            "n_stores_verified": qc_store_audit["n_verified"],
        },
        "imputation": {
            "path": _portable(src["imputation_mask"]),
            "sha256": imputation_sha,
            # whole-panel totals: provenance only (never compared with the fit
            # counts the PCMCI metadata reports)
            "n_cells": imputation["full_n_cells"],
            "n_masked_cells": imputation["n_masked_cells"],
            "counts": imputation["full_counts"],
            "fit_n_cells": imputation_fit["n_cells"],
            "fit_counts": imputation_audit["fit_counts"],
            "n_fit_masked_cells": imputation_audit["n_fit_masked_cells"],
            "panel_cells_by_variable":
                imputation_audit["panel_cells_by_variable"],
            "panel_cells_match": imputation_audit["panel_cells_match"],
            "strong_variables": imputation_audit["strong_variables"],
            "used_columns": imputation_audit["used_columns"],
            "ignored_file_variables":
                imputation_audit["ignored_file_variables"],
            "pcmci_metadata_key": imputation_audit["metadata_key"],
            "manifest": (_portable(imputation["manifest"])
                         if imputation["manifest"] else None),
        },
        "identification_dag": {
            "dot": {"path": _portable(src["identification_dag_dot"]),
                    "sha256": sha256_file(src["identification_dag_dot"])},
            "json": {"path": _portable(src["identification_dag_json"]),
                     "sha256": sha256_file(src["identification_dag_json"])},
            "dot_sha256": dag_audit["dot"],
            "dot_name": dag_audit["dot_name"],
            "status_source": "identification_status.csv",
        },
        "max_feature_time": str(fit_elig["ISO_TIME"].max()),
        "max_label_time": str(fit_elig[proto.label_time_col].max()),
        "label_time_col": proto.label_time_col,
        "files": files,
        "assets": assets,
        "provenance": {
            "phase1_holdout": _portable(p1),
            "source_hashes": {k: sha256_file(v) for k, v in src.items()},
            "phase1_prior_manifest": {
                "path": _portable(src["prior_manifest"]),
                "sha256": sha256_file(src["prior_manifest"])},
            "panel_manifest": {
                "path": _portable(src["panel_manifest"]),
                "sha256": sha256_file(src["panel_manifest"])},
            "boundary_audit": {
                "path": _portable(src["boundary_audit"]),
                "sha256": sha256_file(src["boundary_audit"])},
            "frames_verify": verify_stats,
            "frames_match": "exact",
            "frames": {
                "dir": _portable(frames_dir),
                "match": "exact",
                "n_index_rows": int(len(idx)),
                "duplicate_frame_time_stores": len(
                    idx.attrs.get("duplicate_frame_times", {})),
                "duplicate_frame_times":
                    dict(idx.attrs.get("duplicate_frame_times", {})),
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "builder": "scripts/build_holdout_assets.py",
            "protocol_config": _portable(args.config),
        },
        "trusted_signs": {"path": _portable(out / "trusted_signs.json"),
                          "sha256": sha256_file(out / "trusted_signs.json")},
    }
    manifest_path = out / "prior_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    (out / "build_manifest.json").write_text(json.dumps({
        "prior_manifest": str(manifest_path),
        "prior_manifest_sha256": sha256_file(manifest_path),
        "build_fingerprint": sha256_json(manifest),
    }, indent=2, sort_keys=True), encoding="utf-8")

    print(f"population: {population} {population_audit['n_systems']} systems "
          f"({population_audit['n_label_eligible']} label-eligible, "
          f"{population_audit['n_label_ineligible']} excluded)")
    if fit_without_rows:
        print(f"NOTE: {len(fit_without_rows)} nominal fit SID(s) own no "
              f"fit-eligible row and are audited (not moved back into fit): "
              f"{fit_without_rows[:5]}")
    print(f"fit SIDs: {len(fit_sids)}/{len(fit_split_sids)} nominal  "
          f"prior rows: {len(fit_elig)}  "
          f"available rows: {n_paired}/{n_eligible} eligible "
          f"({n_panel_rows} panel rows)")
    print(f"frame availability: {n_excluded} eligible rows excluded "
          f"({availability_manifest['by_reason']}); alias coverage "
          f"{availability_manifest['alias_coverage']['n_covered']}/"
          f"{availability_manifest['alias_coverage']['n_canonical_rows']} "
          f"-> {availability_path}")
    if availability_manifest["missing_stores"]:
        print(f"WARNING: {len(availability_manifest['missing_stores'])} "
              f"label-eligible systems have no QC store; every one of their "
              f"rows is excluded and listed: "
              f"{availability_manifest['missing_stores'][:5]}")
    print(f"trusted signs: {signs}")
    dup_stores = idx.attrs.get("duplicate_frame_times", {})
    if dup_stores:
        print(f"WARNING: {len(dup_stores)} storm stores repeat a frame "
              f"timestamp (first-matching frame is used, per storm): "
              f"{dict(list(dup_stores.items())[:3])}")
    print(f"TBB norm: mean={norm['mean']:.3f} std={norm['std']:.3f} "
          f"({norm['n_pixels']} px, {norm['n_storms']} storms)")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
