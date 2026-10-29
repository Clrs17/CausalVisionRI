# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Deterministic QC v1 of the global HURSAT frame corpus (raw -> frames_qc).

The raw corpus (``data/global_hursat/frames``, written by
``scripts/download_global_hursat.py`` and re-keyed by
``scripts/rekey_hursat_2024.py``) has two defects that are not byte-level:

* the same acquisition time can hold several *different* frames from different
  satellites (e.g. ``2001106N04138`` 2001-04-17 18Z carries both GMS-5 and
  MET-5), so "take the leftmost row" is not a defensible choice; and
* some frames carry non-physical fill values (2.44 K) because decoding only
  honoured the netCDF masked value, plus the original decode silently skipped
  members (``2001106N04138`` lost 2001-04-16 09Z/12Z).

This tool produces an independent, reproducible QC corpus; the raw corpus is
never modified:

    data/global_hursat/frames_qc/{SID}.npy         float16 (n, 224, 224)
    data/global_hursat/frames_qc/{SID}_times.npy   datetime64[s] (n,)
    data/global_hursat/qc_manifest.json            per-store audit

QC rule ``hursat_qc_v1`` (no new calibration, one broad non-physical mask):

1. group the candidates of a storm by acquisition time;
2. per candidate count the pixels that are finite AND inside
   ``100 K <= TBB <= 350 K``; keep the candidate with the most such pixels,
   ties resolved to the lowest original index (deterministic);
3. set every pixel outside ``[100, 350] K`` (and every non-finite pixel) to
   NaN in the kept frame;
4. emit strictly ascending, unique times; acquisition times with no member in
   the tarball stay missing (never zero-filled).

Tarball repair: the parseable ``.nc`` members of the storm tarball are compared
with the raw store.  If the row count or the unique-time count differs (or the
tarball has a time the store lacks) the storm is re-decoded from the tarball and
the QC runs on that.  Every member that cannot be parsed or decoded is listed in
the manifest — the old decode's silent ``continue`` is gone.

Only a full run (all source stores) is marked ``complete: true``; ``--sids`` /
``--limit`` runs write ``complete: false`` plus a ``subset`` block and can never
be used as a production corpus.

Cost: every storm tarball is opened once to compare its members with the raw
store (the .tar.gz stream must be decompressed to walk it), and only the storms
whose row / unique-time counts differ are decoded again; a full-corpus run is
therefore tens of minutes, and ``--resume`` picks up where it stopped.

Order of operations (the re-key must run first):
    python scripts/rekey_hursat_2024.py
    python scripts/01_data/build_hursat_qc.py            # this tool
    python scripts/build_holdout_assets.py

Usage (from the release root):
    python scripts/01_data/build_hursat_qc.py --limit 3          # smoke
    python scripts/01_data/build_hursat_qc.py --resume           # full run
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.frame_store import (frame_array_path, load_frame_times,
                             require_frame_time_path)
from wpac_warning.ir_dataset import frame_time_from_name
from wpac_warning.protocol import (QC_MANIFEST_REL, QC_RULE_VERSION,
                          RAW_HURSAT_FRAMES_REL, REKEY_CROSSWALK_REL,
                          REKEY_N_ENTRIES,
                          TBB_VALID_MAX_K, TBB_VALID_MIN_K,
                          is_raw_hursat_frames, portable_path,
                          require_not_archive, sha256_array, sha256_file,
                          sha256_json)

SOURCE_FRAMES = P2_ROOT / RAW_HURSAT_FRAMES_REL
TARBALLS = P2_ROOT / "data" / "global_hursat" / "tarballs"
OUT_FRAMES = P2_ROOT / "data" / "global_hursat" / "frames_qc"
CROSSWALK = P2_ROOT / REKEY_CROSSWALK_REL
MANIFEST = P2_ROOT / QC_MANIFEST_REL
CROP = 224
TIME_SUFFIX = "_times"
ISO_SUFFIX = "_iso"
PARTIAL_EVERY = 25


def _p2(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (P2_ROOT / p)


def _portable(path: str | Path) -> str:
    return portable_path(path, P2_ROOT)


def _ns(times) -> np.ndarray:
    """tz-naive datetime64[ns] (the corpus convention)."""
    return np.asarray(times, dtype="datetime64[ns]").reshape(-1)


def valid_pixels(frame: np.ndarray) -> int:
    v = np.asarray(frame, dtype=np.float32)
    return int(np.count_nonzero(np.isfinite(v) & (v >= TBB_VALID_MIN_K)
                                & (v <= TBB_VALID_MAX_K)))


def member_sensor(name: str) -> str | None:
    """Satellite/sensor field of a HURSAT v07b member name (7th ``.`` field)."""
    parts = Path(name).name.split(".")
    # SID.NAME.YYYY.MM.DD.HHMM.QQ.SENSOR.VVV.hursat-b1.v07b.nc
    return parts[7] if len(parts) > 8 else None


def parse_tarball_members(tar_path: Path) -> dict:
    """(member times, parse failures) of a storm tarball, no decoding."""
    times, unparsed = [], []
    with tarfile.open(tar_path, "r:gz") as tf:
        for name in sorted(tf.getnames()):
            if not name.endswith(".nc"):
                continue
            t = frame_time_from_name(Path(name))
            if t is None:
                unparsed.append(name)
                continue
            times.append(t.tz_convert(None).to_datetime64())
    return {"times": _ns(times) if times else np.array([], dtype="datetime64[ns]"),
            "unparsed_members": unparsed}


def _decode_member(raw: bytes) -> np.ndarray:
    """Decode one HURSAT v07b ``.nc`` payload to a centred 224x224 float32."""
    from netCDF4 import Dataset
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "frame.nc"
        p.write_bytes(raw)
        with Dataset(p) as ds:
            v = ds["IRWIN"][:]
    arr = np.asarray(np.ma.filled(v, np.nan), dtype=np.float32).squeeze()
    if arr.ndim != 2 or arr.shape[0] < CROP or arr.shape[1] < CROP:
        raise ValueError(f"unusable IRWIN shape {arr.shape}")
    s0 = (arr.shape[0] - CROP) // 2
    s1 = (arr.shape[1] - CROP) // 2
    return arr[s0:s0 + CROP, s1:s1 + CROP]


def decode_tarball(tar_path: Path) -> dict:
    """Decode every ``.nc`` member; failures are reported, never swallowed."""
    candidates = []
    unparsed, failures = [], []
    n_members = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for name in sorted(tf.getnames()):
            if not name.endswith(".nc"):
                continue
            n_members += 1
            t = frame_time_from_name(Path(name))
            if t is None:
                unparsed.append(name)
                continue
            try:
                handle = tf.extractfile(name)
                if handle is None:
                    raise ValueError("member is not a regular file")
                frame = _decode_member(handle.read())
            except Exception as exc:   # noqa: BLE001 - recorded, not hidden
                failures.append({"member": str(name),
                                 "reason": f"{type(exc).__name__}: {exc}"})
                continue
            candidates.append({
                "time": t.tz_convert(None).to_datetime64(),
                "frame": frame,
                "member": Path(name).name,
                "sensor": member_sensor(name),
                "source_index": len(candidates),
            })
    return {"candidates": candidates, "members_total": n_members,
            "unparsed_members": unparsed, "failures": failures}


def qc_candidates(candidates: list) -> dict:
    """Rule ``hursat_qc_v1``: per time keep the most valid coverage, mask fill.

    ``candidates`` must be in source order (index = position); the return value
    carries the QC frames/times plus the duplicate-time audit.
    """
    if not candidates:
        raise ValueError("no candidate frames to QC")
    groups: dict = {}
    for i, cand in enumerate(candidates):
        groups.setdefault(np.datetime64(cand["time"], "ns"), []).append(i)
    out_frames, out_times, dup_groups = [], [], []
    n_masked = 0
    observed_min = np.inf
    observed_max = -np.inf
    for t in sorted(groups):
        idxs = groups[t]
        counts = []
        best_i, best_n = None, -1
        for i in idxs:
            n = valid_pixels(candidates[i]["frame"])
            counts.append(n)
            if n > best_n:          # strict: ties keep the lowest index
                best_n, best_i = n, i
        chosen = np.asarray(candidates[best_i]["frame"], dtype=np.float32)
        finite = np.isfinite(chosen)
        if finite.any():
            observed_min = min(observed_min, float(chosen[finite].min()))
            observed_max = max(observed_max, float(chosen[finite].max()))
        bad = ~finite | (chosen < TBB_VALID_MIN_K) | (chosen > TBB_VALID_MAX_K)
        n_masked += int(bad.sum())
        out = chosen.copy()
        out[bad] = np.nan
        out_frames.append(out.astype(np.float16))
        out_times.append(t)
        if len(idxs) > 1:
            dup_groups.append({
                "time": str(pd.Timestamp(t)),
                "n_candidates": len(idxs),
                "selected_index": int(candidates[best_i]["source_index"]),
                "selected_member": candidates[best_i]["member"],
                "selected_sensor": candidates[best_i]["sensor"],
                "selected_valid_pixels": int(best_n),
                "candidates_valid_pixels": counts,
                "candidate_members": [candidates[i]["member"] for i in idxs],
                "candidate_sensors": [candidates[i]["sensor"] for i in idxs],
                "candidate_arrays_identical": bool(
                    all(np.array_equal(np.asarray(candidates[i]["frame"]),
                                       np.asarray(candidates[idxs[0]]["frame"]))
                        for i in idxs[1:])),
            })
    frames = np.stack(out_frames)
    times = np.asarray(out_times, dtype="datetime64[s]")
    if times.size > 1 and not (np.diff(times).astype("timedelta64[s]")
                               .astype(np.int64) > 0).all():
        raise ValueError("QC output times are not strictly ascending")
    return {"frames": frames, "times": times, "duplicate_groups": dup_groups,
            "n_pixels_masked": int(n_masked),
            "observed_min_k": float(observed_min) if np.isfinite(observed_min) else None,
            "observed_max_k": float(observed_max) if np.isfinite(observed_max) else None,
            "n_duplicate_times": int(len(out_times) - len(groups))}


def _find_tarball(tarballs: Path, sid: str) -> Path | None:
    matches = sorted(Path(tarballs).glob(f"HURSAT_b1_v07b_{sid}_*.tar.gz"))
    if len(matches) > 1:
        raise ValueError(f"{sid}: {len(matches)} tarballs under {tarballs}")
    return matches[0] if matches else None


def needs_repair(source_times: np.ndarray, member_times: np.ndarray) -> bool:
    """True when the tarball holds members the raw store does not.

    Compared on rows, unique times and the member-time set (the original decode
    silently skipped members, so a row/unique-count match alone is not proof).
    """
    if member_times.size == 0:
        return False
    src = _ns(source_times)
    mem = _ns(member_times)
    if mem.size != src.size:
        return True
    src_u = set(np.unique(src).tolist())
    mem_u = set(np.unique(mem).tolist())
    if len(mem_u) != len(src_u):
        return True
    return bool(mem_u - src_u)


def _load_source(sid: str, source_frames: Path) -> dict:
    times = _ns(load_frame_times(source_frames, sid))
    arr = np.load(frame_array_path(source_frames, sid))
    if arr.shape[0] != times.size:
        raise ValueError(f"{sid}: {arr.shape[0]} frames vs {times.size} times")
    return {"array": arr, "times": times}


def _source_candidates(store: dict) -> list:
    return [{"time": np.datetime64(store["times"][i], "ns"),
             "frame": np.asarray(store["array"][i], dtype=np.float32),
             "member": None, "sensor": None, "source_index": i}
            for i in range(store["times"].size)]


def process_store(sid: str, *, source_frames: Path, tarballs: Path,
                  out_frames: Path, alias_of: str | None = None,
                  require_tarball: bool = True) -> dict:
    src_array = frame_array_path(source_frames, sid)
    src_times = require_frame_time_path(source_frames, sid)
    store = _load_source(sid, source_frames)
    source_rows = int(store["times"].size)
    source_unique = int(np.unique(store["times"]).size)

    tarball = _find_tarball(tarballs, sid)
    tarball_sid = sid
    if tarball is None and alias_of:
        # a re-keyed canonical SID shares the archive-SID tarball
        tarball = _find_tarball(tarballs, alias_of)
        tarball_sid = alias_of
    entry = {
        "sid": sid,
        "alias_of": alias_of,
        "source_rows": source_rows,
        "source_unique_times": source_unique,
        "n_duplicate_times": int(source_rows - source_unique),
        "source": {
            "array": _portable(src_array),
            "array_sha256": sha256_file(src_array),
            "times": _portable(src_times),
            "times_sha256": sha256_file(src_times),
        },
        "tarball": None,
        "source_kind": "raw",
        "decode_failures": [],
        "unparsed_members": [],
        "unrecovered_member_times": [],
    }
    if tarball is None:
        if require_tarball:
            raise FileNotFoundError(
                f"{sid}: no HURSAT v07b tarball under {tarballs}"
                + (f" (alias of {alias_of})" if alias_of else "")
                + "; the raw store cannot be verified")
        entry["tarball"] = {"name": None, "sha256": None, "status": "missing"}
        candidates = _source_candidates(store)
    else:
        members = parse_tarball_members(tarball)
        member_times = members["times"]
        repair = needs_repair(store["times"], member_times)
        entry["tarball"] = {
            "name": tarball.name,
            "path": _portable(tarball),
            "sha256": sha256_file(tarball),
            "sid": tarball_sid,
            "members_total": int(member_times.size) + len(
                members["unparsed_members"]),
            "members_parseable": int(member_times.size),
            "members_unique_times": int(np.unique(member_times).size),
            "unparsed_members": members["unparsed_members"],
            "repaired": bool(repair),
        }
        if repair:
            decoded = decode_tarball(tarball)
            candidates = decoded["candidates"]
            entry["source_kind"] = "tarball_redecode"
            entry["decode_failures"] = decoded["failures"]
            entry["unparsed_members"] = decoded["unparsed_members"]
            if not candidates:
                raise ValueError(f"{sid}: repair produced no decodable frames")
            have = {np.datetime64(c["time"], "ns") for c in candidates}
            entry["unrecovered_member_times"] = [
                str(pd.Timestamp(t)) for t in np.unique(member_times)
                if np.datetime64(t, "ns") not in have]
        else:
            candidates = _source_candidates(store)

    qc = qc_candidates(candidates)
    out_frames.mkdir(parents=True, exist_ok=True)
    out_arr = out_frames / f"{sid}.npy"
    out_times_path = out_frames / f"{sid}{TIME_SUFFIX}.npy"
    np.save(out_arr, qc["frames"])
    np.save(out_times_path, qc["times"])
    iso = out_frames / f"{sid}{ISO_SUFFIX}.npy"
    if iso.exists():
        raise ValueError(f"{sid}: a '_iso' time axis appeared in the QC corpus")
    entry.update({
        "output_rows": int(qc["times"].size),
        "output_unique_times": int(qc["times"].size),
        "output": {
            "array": _portable(out_arr),
            "array_sha256": sha256_file(out_arr),
            "times": _portable(out_times_path),
            "times_sha256": sha256_file(out_times_path),
        },
        "output_time_range": [str(pd.Timestamp(qc["times"][0])),
                              str(pd.Timestamp(qc["times"][-1]))]
        if qc["times"].size else [],
        "duplicate_groups": qc["duplicate_groups"],
        "qc": {
            "rule": QC_RULE_VERSION,
            "tbb_valid_range_k": [TBB_VALID_MIN_K, TBB_VALID_MAX_K],
            "observed_min_k": qc["observed_min_k"],
            "observed_max_k": qc["observed_max_k"],
            "n_pixels_masked": qc["n_pixels_masked"],
            "n_duplicate_times": qc["n_duplicate_times"],
        },
    })
    return entry


def _list_stores(frames_dir: Path) -> list:
    return sorted(p.stem for p in Path(frames_dir).glob("*.npy")
                  if not p.stem.endswith((ISO_SUFFIX, TIME_SUFFIX)))


def _check_dirs(source_frames: Path, out_frames: Path) -> None:
    src, out = source_frames.resolve(), out_frames.resolve()
    if src == out:
        raise ValueError("the QC output directory must differ from the raw "
                         "corpus (the raw stores are never modified)")
    if src in out.parents or out in src.parents:
        raise ValueError(f"QC output {out} must not nest with the raw corpus "
                         f"{src}")
    require_not_archive(src, "raw HURSAT frames")
    require_not_archive(out, "QC frame output")
    if not is_raw_hursat_frames(src):
        raise ValueError(
            f"{src} is not the raw global HURSAT corpus "
            f"({RAW_HURSAT_FRAMES_REL}); the QC reads the raw stores only")


def _rekey_block(crosswalk: Path, stores: set) -> dict:
    if not crosswalk.is_file():
        raise FileNotFoundError(
            f"{crosswalk} missing: run scripts/rekey_hursat_2024.py before the "
            f"QC so the 28 canonical 2024 aliases are part of the corpus")
    payload = json.loads(crosswalk.read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    canonical = [str(e["canonical_sid"]) for e in entries]
    if int(payload.get("n_entries", -1)) != len(entries):
        raise ValueError(f"{crosswalk}: n_entries disagrees with its entries")
    if len(entries) != REKEY_N_ENTRIES:
        raise ValueError(
            f"{crosswalk} carries {len(entries)} aliases; the 2024 re-key is "
            f"fixed at {REKEY_N_ENTRIES}")
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"{crosswalk}: duplicate canonical SIDs")
    missing = sorted(set(canonical) - stores)
    if missing:
        raise ValueError(
            f"{len(missing)} re-keyed canonical SID(s) are absent from the "
            f"source corpus ({missing[:5]}); the re-key writes into "
            f"{RAW_HURSAT_FRAMES_REL}, so run it before the QC")
    covered = len(set(canonical) & stores)
    if covered != REKEY_N_ENTRIES:
        raise ValueError(
            f"{covered}/{REKEY_N_ENTRIES} canonical aliases are present in the "
            f"source corpus; the re-key is incomplete")
    return {"path": _portable(crosswalk),
            "sha256": sha256_file(crosswalk),
            "n_entries": len(entries),
            "n_canonical_covered": covered,
            "canonical_sids_sha256": sha256_array(sorted(canonical))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-frames", default=str(SOURCE_FRAMES))
    ap.add_argument("--out-frames", default=str(OUT_FRAMES))
    ap.add_argument("--tarballs", default=str(TARBALLS))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--crosswalk", default=str(CROSSWALK))
    ap.add_argument("--sids", default=None,
                    help="comma-separated SIDs (subset runs / tests)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="replace an existing QC corpus")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the partial manifest and skip finished stores")
    ap.add_argument("--no-rekey-check", action="store_true",
                    help="skip the re-key coverage check; only for partial or "
                         "audit corpora (production requires the re-key first)")
    ap.add_argument("--allow-missing-tarball", action="store_true",
                    help="do not fail when a source store has no tarball (the "
                         "default is strict: every store must be verifiable)")
    args = ap.parse_args()

    source_frames = _p2(args.source_frames).resolve()
    out_frames = _p2(args.out_frames).resolve()
    tarballs = _p2(args.tarballs).resolve()
    manifest_path = _p2(args.manifest)
    crosswalk = _p2(args.crosswalk)
    _check_dirs(source_frames, out_frames)

    all_stores = _list_stores(source_frames)
    if not all_stores:
        raise SystemExit(f"no frame stores under {source_frames}")
    subset = bool(args.sids or args.limit)
    selected = all_stores
    if args.sids:
        wanted = [s.strip() for s in args.sids.split(",") if s.strip()]
        unknown = sorted(set(wanted) - set(all_stores))
        if unknown:
            raise SystemExit(f"unknown SIDs under {source_frames}: {unknown}")
        selected = wanted
    if args.limit:
        selected = selected[:args.limit]

    rekey = None
    alias_map: dict = {}
    if not subset and not args.no_rekey_check:
        rekey = _rekey_block(crosswalk, set(all_stores))
    if crosswalk.is_file():
        alias_map = {
            str(e["canonical_sid"]): str(e["archive_sid"])
            for e in json.loads(crosswalk.read_text(encoding="utf-8"))
            .get("entries", [])
        }

    if out_frames.exists() and any(out_frames.glob("*.npy")) and not (
            args.force or args.resume):
        raise SystemExit(f"{out_frames} already holds a QC corpus; pass "
                         f"--force to rebuild or --resume to continue")

    partial_path = manifest_path.with_suffix(".partial.json")
    done: dict = {}
    resume_context = {
        "qc_rule_version": QC_RULE_VERSION,
        "source_frames_dir": _portable(source_frames),
        "out_frames_dir": _portable(out_frames),
        "tarballs_dir": _portable(tarballs),
        "rekey_crosswalk_sha256": (rekey or {}).get("sha256"),
    }
    if args.resume:
        for cand in (manifest_path, partial_path):
            if cand.is_file():
                old = json.loads(cand.read_text(encoding="utf-8"))
                if old.get("qc_rule_version") != QC_RULE_VERSION:
                    raise SystemExit(f"{cand}: QC rule version changed; a new "
                                     f"corpus is required (--force)")
                old_ctx = old.get("run_context") or {}
                diffs = {k: (old_ctx.get(k), resume_context[k])
                         for k in resume_context
                         if (old_ctx.get(k) or None) != (resume_context[k] or None)}
                if diffs:
                    raise SystemExit(
                        f"{cand}: cannot resume — the run context changed "
                        f"(rule/corpus/tarballs/crosswalk): {diffs}; rebuild "
                        f"the QC corpus with --force")
                done = {e["sid"]: e for e in old.get("stores", [])}
                break

    def _reusable(sid: str) -> bool:
        entry = done.get(sid)
        if not entry:
            return False
        try:
            out_arr = out_frames / f"{sid}.npy"
            out_times = out_frames / f"{sid}{TIME_SUFFIX}.npy"
            if not (out_arr.is_file() and out_times.is_file()):
                return False
            if (sha256_file(out_arr) != entry["output"]["array_sha256"]
                    or sha256_file(out_times) != entry["output"]["times_sha256"]):
                return False
            if (entry["source"]["array_sha256"]
                    != sha256_file(frame_array_path(source_frames, sid))
                    or entry["source"]["times_sha256"]
                    != sha256_file(require_frame_time_path(source_frames, sid))):
                return False
            tarball = entry.get("tarball") or {}
            if tarball.get("name"):
                tar_path = tarballs / str(tarball["name"])
                if not tar_path.is_file():
                    return False
                if sha256_file(tar_path) != tarball.get("sha256"):
                    return False
            elif tarball.get("status") not in ("missing", None):
                return False
            return True
        except (KeyError, OSError):
            return False

    entries: dict = {}
    n_new = n_skipped = 0
    all_stores_set = set(all_stores)
    tarballs_without_store = sorted(
        sid for sid in (p.name.split("_")[3]
                        for p in tarballs.glob("HURSAT_b1_v07b_*.tar.gz"))
        if sid not in all_stores_set)

    full_run = set(selected) == set(all_stores)
    audit_flags = {
        "subset": not full_run,
        "no_rekey_check": bool(args.no_rekey_check),
        "allow_missing_tarball": bool(args.allow_missing_tarball),
    }
    production = bool(full_run and rekey is not None
                      and not audit_flags["no_rekey_check"]
                      and not audit_flags["allow_missing_tarball"])

    def _write_manifest(path: Path, complete: bool) -> dict:
        stores = [entries[s] for s in sorted(entries)]
        payload = {
            "protocol": "holdout_masked_v1",
            "population": "alltc",
            "qc_rule_version": QC_RULE_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool": "scripts/build_hursat_qc.py",
            # a partial corpus (subset/limit run) is never a production corpus
            "complete": bool(complete) and full_run,
            "production": bool(production and complete),
            "audit_flags": audit_flags,
            "run_context": resume_context,
            "subset": {
                "is_subset": not full_run,
                "n_selected": len(selected),
                "n_source": len(all_stores),
                "selected_sha256": sha256_array(sorted(selected)),
            },
            "source_frames_dir": _portable(source_frames),
            "out_frames_dir": _portable(out_frames),
            "tarballs_dir": _portable(tarballs),
            "tbb_valid_range_k": [TBB_VALID_MIN_K, TBB_VALID_MAX_K],
            "crop": CROP,
            "rule": (
                "group candidates by acquisition time; keep the candidate with "
                "the most finite pixels inside 100-350 K (ties -> lowest source "
                "index); mask every pixel outside that range to NaN; emit "
                "strictly ascending unique times; missing acquisition times are "
                "never zero-filled; a storm whose tarball has members the raw "
                "store lacks is re-decoded"),
            "rekey": rekey,
            "counts": {
                "n_source_stores": len(all_stores),
                "n_selected_stores": len(selected),
                "n_stores": len(stores),
                "n_missing_stores": len(set(selected) - set(entries)),
                "n_repaired": sum(1 for e in stores
                                  if e["source_kind"] == "tarball_redecode"),
                "n_stores_with_decode_failures": sum(
                    1 for e in stores if e["decode_failures"]),
                "n_stores_with_unparsed_members": sum(
                    1 for e in stores if e["unparsed_members"]),
                "n_stores_with_duplicate_times": sum(
                    1 for e in stores if e["duplicate_groups"]),
                "n_unrecovered_member_times": sum(
                    len(e.get("unrecovered_member_times") or [])
                    for e in stores),
                "n_source_rows": sum(e["source_rows"] for e in stores),
                "n_output_rows": sum(e["output_rows"] for e in stores),
            },
            "tarballs_without_store": tarballs_without_store,
            "missing_stores": sorted(set(selected) - set(entries)),
            "stores": stores,
            "stores_sha256": sha256_json(stores),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                        encoding="utf-8")
        return payload

    for sid in selected:
        if _reusable(sid):
            entries[sid] = done[sid]
            n_skipped += 1
            continue
        entry = process_store(sid, source_frames=source_frames,
                              tarballs=tarballs, out_frames=out_frames,
                              alias_of=alias_map.get(sid),
                              require_tarball=not args.allow_missing_tarball)
        entries[sid] = entry
        n_new += 1
        flag = entry["source_kind"]
        if entry["decode_failures"]:
            flag += f", {len(entry['decode_failures'])} decode failures"
        print(f"[{n_new + n_skipped}/{len(selected)}] {sid}: "
              f"{entry['source_rows']} -> {entry['output_rows']} rows, "
              f"{len(entry['duplicate_groups'])} duplicate times ({flag})",
              flush=True)
        if n_new % PARTIAL_EVERY == 0:
            _write_manifest(partial_path, complete=False)

    payload = _write_manifest(manifest_path, complete=True)
    if not payload["complete"]:
        print(f"NOTE: this is a SUBSET QC run ({len(selected)}/"
              f"{len(all_stores)} stores); the manifest is marked "
              f"complete=false and cannot be used for production assets",
              flush=True)
    if partial_path.is_file():
        partial_path.unlink()
    counts = payload["counts"]
    print(f"QC {QC_RULE_VERSION}: {counts['n_stores']} stores "
          f"({counts['n_repaired']} re-decoded from tarball, "
          f"{counts['n_stores_with_duplicate_times']} with duplicate times), "
          f"{counts['n_source_rows']} -> {counts['n_output_rows']} frames")
    if counts["n_stores_with_decode_failures"]:
        print(f"WARNING: {counts['n_stores_with_decode_failures']} stores have "
              f"undecodable members (listed in {manifest_path.name})")
    if counts["n_missing_stores"]:
        raise SystemExit(f"{counts['n_missing_stores']} source stores were not "
                         f"processed: {payload['missing_stores'][:5]}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
