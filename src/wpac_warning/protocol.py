# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""HOLDOUT_MASKED_V1 protocol: explicit splits, label embargo, prior bundle
validation, hashing and versioned run directories.

The protocol is defined in docs/PROTOCOL.md.  This module is the
single source of truth for

  * the canonical SID -> storm_year mapping (phase-1 sample manifest; never
    re-derived from a frames join or from the panel's first row);
  * split assignment (fit / inner / test) and the 30 h label embargo at the
    fit|inner and inner|test boundaries;
  * SHA-256 verification of every prior/data asset (no fallback paths, no
    silent re-fitting);
  * trusted ATE signs (fail-closed: a treatment without an identification
    record, or with no directed path, is never used as a causal constraint);
  * the versioned run directory and its manifests.

Nothing here touches the model or the training loop.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROTOCOL_NAME = "holdout_masked_v1"
DEFAULT_LABEL_HORIZON_H = 24
DEFAULT_LABEL_TOLERANCE_H = 6
SPLIT_NAMES = ("fit", "inner", "test")


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object (sorted keys, no NaN)."""
    txt = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str,
                     allow_nan=False)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def accepted_lock_sha256(lock: dict) -> set:
    """Lock digests a frozen artefact may legitimately carry.

    ``protocol_locks/final.json`` is re-issued when its path fields are
    normalised for release: the paths move, the numbers do not, and
    ``lock_sha256`` is recomputed over the normalised payload.  Artefacts that
    were frozen *before* the re-issue (the released ``data/test`` npz files,
    their sidecars and the receipt of the original run) carry the digest they
    were produced under.  The re-issue records that predecessor in
    ``release_reissue.superseded_lock_sha256``, so exactly two digests are
    accepted by a verification step and nothing else.
    """
    out = {str(lock["lock_sha256"])}
    prev = (lock.get("release_reissue") or {}).get("superseded_lock_sha256")
    if prev:
        out.add(str(prev))
    return out


def sha256_array(values: Sequence[Any]) -> str:
    """Hash of an ordered sequence of row keys / SIDs."""
    return hashlib.sha256(
        "\n".join(str(v) for v in values).encode("utf-8")).hexdigest()


def portable_path(path: str | Path, base: str | Path | None = None) -> str:
    """POSIX path relative to `base` (default: the phase-2 root).

    Falls back to an absolute POSIX path when the target lives on another
    Windows drive (``os.path.relpath`` raises ValueError in that case).
    """
    p = Path(path).resolve()
    b = (Path(base).resolve() if base is not None
         else Path(__file__).resolve().parents[2])
    try:
        return Path(os.path.relpath(p, b)).as_posix()
    except ValueError:
        return p.as_posix()


DATA_ASSET_PATH_KEYS = ("panel_parquet", "frames_index", "frames_dir",
                        "tbb_norm", "storm_year_manifest", "prior_manifest",
                        "ir_embeds_dir")


def canonical_data_assets(cfg_dict: Dict[str, Any],
                          base: str | Path | None = None) -> Dict[str, Any]:
    """Canonical portable paths so absolute and relative configs compare equal."""
    root = (Path(base).resolve() if base is not None
            else Path(__file__).resolve().parents[2])
    data = cfg_dict.get("data", {})
    assets = {}
    for key in DATA_ASSET_PATH_KEYS:
        value = data.get(key)
        if value in (None, ""):
            assets[key] = None
            continue
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        assets[key] = portable_path(resolved, root)
    return assets


PRODUCTION_TEST_OUTPUT_DIR = (Path(__file__).resolve().parents[2] /
                              "data" / "test" / "holdout_masked_v1")
TEST_RECEIPT_NAME = "test_receipt.json"


def resolve_protocol_test_output_dir(requested=None,
                                     allow_fixture: bool = False) -> Path:
    """Resolve the unique production test directory or an explicit fixture path."""
    root = Path(__file__).resolve().parents[2]
    canonical = PRODUCTION_TEST_OUTPUT_DIR.resolve()
    if requested is None:
        return canonical
    path = Path(requested)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not allow_fixture and resolved != canonical:
        raise ValueError(f"production test_output_dir is fixed at {canonical}; "
                         f"custom paths require explicit fixture mode")
    return resolved


def protocol_test_receipt(output_dir: str | Path) -> Path:
    return Path(output_dir).resolve() / TEST_RECEIPT_NAME


def check_test_receipt_available(receipt_path: str | Path,
                                 block_running: bool = True) -> Optional[dict]:
    """Fail when a protocol test is already underway or completed."""
    path = Path(receipt_path)
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"test receipt protocol mismatch: {path}")
    status = receipt.get("status")
    if status == "completed":
        raise RuntimeError(f"completed test receipt exists at {path}; "
                           f"selection/finalization/inference cannot rerun test")
    if status == "running" and block_running:
        raise RuntimeError(f"interrupted test receipt exists at {path}; recover "
                           f"it before re-selection or re-finalization")
    if status not in ("running", "completed"):
        raise ValueError(f"invalid test receipt status {status!r}: {path}")
    return receipt


def require_file(path: str | Path, what: str) -> Path:
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise FileNotFoundError(f"{what} not found or empty: {p}")
    return p


def verify_sha256(path: str | Path, expected: str, what: str) -> str:
    p = require_file(path, what)
    got = sha256_file(p)
    if str(expected).lower() != got:
        raise ValueError(
            f"{what} SHA-256 mismatch: {p}\n  expected {expected}\n  got      {got}")
    return got


def _as_year_list(cfg, key: str, default=None) -> List[int]:
    val = cfg.get(key, default)
    if val is None:
        raise ValueError(f"config key {key} is required")
    if isinstance(val, (int, np.integer)):
        val = [int(val), int(val)]
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise ValueError(f"{key} must be [start, end], got {val!r}")
    return [int(val[0]), int(val[1])]


def year_range(years: Sequence[int]) -> List[int]:
    return list(range(int(years[0]), int(years[1]) + 1))


# ----------------------------------------------------------------------
# ALL-SYSTEM population binding (725 WNP systems, never the 375-sample)
# ----------------------------------------------------------------------

ALLTC_POPULATION = "alltc"
DEFAULT_POPULATION_N_SYSTEMS = 725
DEFAULT_POPULATION_N_LABEL_INELIGIBLE = 5
# the training corpus is the QC v1 corpus; the raw download corpus is only the
# QC/re-key input and must never be configured as data.frames_dir
GLOBAL_HURSAT_CORPUS_RAW = "hursat_b1_v07b_global"
GLOBAL_HURSAT_CORPUS_QC = "hursat_b1_v07b_global_qc_v1"
GLOBAL_HURSAT_CORPUS = GLOBAL_HURSAT_CORPUS_QC
RAW_HURSAT_FRAMES_REL = "data/global_hursat/frames"
QC_HURSAT_FRAMES_REL = "data/global_hursat/frames_qc"
QC_MANIFEST_REL = "data/global_hursat/qc_manifest.json"
REKEY_CROSSWALK_REL = "data/global_hursat/rekey_crosswalk.json"
REKEY_N_ENTRIES = 28
QC_RULE_VERSION = "hursat_qc_v1"
TBB_VALID_MIN_K = 100.0
TBB_VALID_MAX_K = 350.0
# self-supervised MAE corpus identity: the QC corpus (deduplicated, the
# downstream training corpus) and the full multi-sensor raw corpus (every raw
# frame kept, duplicate acquisition times preserved, non-physical TBB masked
# to NaN at cache build time) — the latter is pretraining-only, never a
# downstream frame source
MAE_CORPUS_QC = "hursat_qc_v1"
MAE_CORPUS_FULL_MULTISENSOR = "hursat_full_multisensor_v1"
# frame availability (which eligible rows have an exact QC frame at their t0)
FRAME_AVAILABILITY_RULE_VERSION = "frame_availability_v1"
FRAME_AVAILABILITY_REL = "data/holdout_masked_v1/frame_availability.json"
FRAME_AVAILABILITY_REASONS = ("missing_store", "before_first_frame",
                              "after_last_frame", "internal_frame_gap")
LEGACY_WNP_FRAMES_REL = "data/frames"
ARCHIVE_SEGMENTS = ("archive", "pre_alltc_scope_v1", "pre_holdout_legacy",
                    "pre_ohc_backward_v1", "pre_ohc_daily_window_v1")


def _posix_segments(path: Any) -> List[str]:
    return [seg for seg in str(path).replace("\\", "/").lower().split("/")
            if seg not in ("", ".")]


def is_archive_path(path: Any) -> bool:
    """True when `path` lives inside a frozen ``archive/`` snapshot."""
    for seg in _posix_segments(path):
        if seg in ARCHIVE_SEGMENTS or seg.startswith("archive"):
            return True
    return False


def require_not_archive(path: Any, what: str = "path") -> Path:
    """Hard-reject archive snapshots as inputs (frozen 375-system products)."""
    if is_archive_path(path):
        raise ValueError(
            f"{what} resolves inside a frozen archive: {path} — archived "
            f"holdout/375-system products are never a protocol input")
    return Path(path)


def is_legacy_wnp_frames(path: Any) -> bool:
    """True for the WNP-only ``data/frames`` store the holdout protocol forbids."""
    return _posix_segments(path)[-2:] == ["data", "frames"]


def is_raw_hursat_frames(path: Any) -> bool:
    """True for the raw download corpus (``…/global_hursat/frames``)."""
    return _posix_segments(path)[-2:] == ["global_hursat", "frames"]


def is_qc_hursat_frames(path: Any) -> bool:
    """True for the QC v1 corpus (``…/global_hursat/frames_qc``)."""
    return _posix_segments(path)[-2:] == ["global_hursat", "frames_qc"]


QC_AUDIT_FLAGS = ("subset", "no_rekey_check", "allow_missing_tarball",
                  "allow_unqc")


def validate_qc_manifest(block: Mapping[str, Any], *,
                         expected_out_frames: Any | None = None,
                         require_complete: bool = True,
                         require_production: bool = True) -> Dict[str, Any]:
    """Fail-closed validation of ``qc_manifest.json`` (rule ``hursat_qc_v1``).

    Binds the QC rule version, the valid TBB window, the source/out corpora, the
    per-store row accounting and the store-set hash; a partial or rule-mismatched
    manifest can never be used as a training corpus.  With
    ``require_production`` (the default for the asset/pack paths) a subset run, a
    ``--no-rekey-check`` run, an ``--allow-missing-tarball`` run or an
    ``--allow-unqc`` cache is refused, and the re-key crosswalk must cover all 28
    aliases.
    """
    if not isinstance(block, Mapping):
        raise ValueError("QC manifest must be a mapping")
    if block.get("qc_rule_version") != QC_RULE_VERSION:
        raise ValueError(
            f"QC manifest rule version {block.get('qc_rule_version')!r} != "
            f"{QC_RULE_VERSION!r}")
    if require_complete and block.get("complete") is not True:
        raise ValueError("QC manifest is partial; finish the QC run "
                         "(scripts/build_hursat_qc.py) before using the corpus")
    audit_flags = {k: bool((block.get("audit_flags") or {}).get(k, False))
                   for k in QC_AUDIT_FLAGS}
    for key in QC_AUDIT_FLAGS:
        if block.get(key) is True:
            audit_flags[key] = True
    if require_production:
        if block.get("production") is False or any(audit_flags.values()):
            raise ValueError(
                "QC manifest is not production-grade "
                f"(production={block.get('production')!r}, audit flags="
                f"{[k for k, v in audit_flags.items() if v]}); subset / "
                "no-rekey-check / allow-missing-tarball / allow-unqc corpora "
                "are audit-only")
        crosswalk = block.get("rekey")
        if crosswalk is not None:
            n_entries = int(crosswalk.get("n_entries", 0))
            n_covered = int(crosswalk.get("n_canonical_covered", 0))
            if n_entries != REKEY_N_ENTRIES or n_covered != REKEY_N_ENTRIES:
                raise ValueError(
                    f"QC manifest re-key crosswalk covers {n_covered}/"
                    f"{n_entries} aliases; production requires "
                    f"{REKEY_N_ENTRIES}/{REKEY_N_ENTRIES}")
    rng = [float(x) for x in block.get("tbb_valid_range_k", [])]
    if rng != [TBB_VALID_MIN_K, TBB_VALID_MAX_K]:
        raise ValueError(
            f"QC manifest tbb_valid_range_k={rng} != "
            f"{[TBB_VALID_MIN_K, TBB_VALID_MAX_K]}")
    if not is_raw_hursat_frames(block.get("source_frames_dir", "")):
        raise ValueError(
            f"QC manifest source_frames_dir={block.get('source_frames_dir')!r} "
            f"is not the raw corpus ({RAW_HURSAT_FRAMES_REL})")
    if not is_qc_hursat_frames(block.get("out_frames_dir", "")):
        raise ValueError(
            f"QC manifest out_frames_dir={block.get('out_frames_dir')!r} is not "
            f"the QC corpus ({QC_HURSAT_FRAMES_REL})")
    if expected_out_frames is not None:
        root = Path(__file__).resolve().parents[2]
        want_abs = Path(str(expected_out_frames)).resolve()
        got_path = Path(str(block.get("out_frames_dir")))
        if not got_path.is_absolute():
            got_path = root / got_path
        got_abs = got_path.resolve()
        if got_abs.as_posix() != want_abs.as_posix():
            raise ValueError(
                f"QC manifest describes {block.get('out_frames_dir')}, but "
                f"data.frames_dir is {portable_path(want_abs)}; the configured "
                f"corpus is not this QC manifest's output")
    stores = block.get("stores")
    if not isinstance(stores, list) or not stores:
        raise ValueError("QC manifest has no per-store entries")
    if int(block.get("counts", {}).get("n_stores", -1)) != len(stores):
        raise ValueError("QC manifest counts.n_stores != its store entries")
    if int(block.get("counts", {}).get("n_missing_stores", -1)) != 0:
        raise ValueError("QC manifest reports unprocessed source stores")
    if block.get("missing_stores"):
        raise ValueError(f"QC manifest lists missing stores: "
                         f"{block['missing_stores'][:5]}")
    if block.get("stores_sha256") != sha256_json(stores):
        raise ValueError("QC manifest stores_sha256 does not match its entries")
    sids = [str(e.get("sid")) for e in stores]
    if len(set(sids)) != len(sids):
        raise ValueError("QC manifest repeats a SID")
    for entry in stores:
        for key in ("sid", "source", "output", "output_rows", "source_rows",
                    "qc"):
            if key not in entry:
                raise ValueError(
                    f"QC manifest store {entry.get('sid')} lacks '{key}'")
        source_rows = int(entry["source_rows"])
        output_rows = int(entry["output_rows"])
        tarball = entry.get("tarball") or {}
        kind = entry.get("source_kind")
        if kind is None:
            kind = ("tarball_redecode" if tarball.get("repaired") is True
                    else "raw")
        kind = str(kind)
        source_unique = entry.get("source_unique_times")
        n_dup = entry.get("n_duplicate_times")
        if source_unique is not None and n_dup is not None:
            source_unique = int(source_unique)
            if source_rows < source_unique or int(n_dup) != (
                    source_rows - source_unique):
                raise ValueError(
                    f"QC manifest store {entry['sid']}: duplicate-time "
                    f"accounting is inconsistent ({source_rows} rows, "
                    f"{source_unique} unique, {n_dup} duplicates)")
        if kind == "raw":
            if source_unique is not None and output_rows != int(source_unique):
                raise ValueError(
                    f"QC manifest store {entry['sid']}: an unrepaired store must "
                    f"emit one frame per unique source time "
                    f"({source_unique} unique vs {output_rows} output rows)")
            if output_rows > source_rows:
                raise ValueError(
                    f"QC manifest store {entry['sid']} grew rows without a "
                    f"re-decode ({source_rows} -> {output_rows})")
        elif kind == "tarball_redecode":
            if tarball.get("repaired") is not True:
                raise ValueError(
                    f"QC manifest store {entry['sid']} claims "
                    f"source_kind=tarball_redecode but the tarball block is not "
                    f"marked repaired")
            if not tarball.get("name") or not _is_sha256(tarball.get("sha256")):
                raise ValueError(
                    f"QC manifest store {entry['sid']} lacks its tarball "
                    f"provenance (name/sha256)")
            member_unique = tarball.get("members_unique_times")
            if member_unique is None:
                raise ValueError(
                    f"QC manifest store {entry['sid']} lacks "
                    f"tarball.members_unique_times; the repaired row count "
                    f"cannot be bounded")
            member_unique = int(member_unique)
            if output_rows > member_unique:
                raise ValueError(
                    f"QC manifest store {entry['sid']} emits {output_rows} rows "
                    f"but the tarball only holds {member_unique} unique member "
                    f"times")
            if source_unique is not None and output_rows < int(source_unique):
                raise ValueError(
                    f"QC manifest store {entry['sid']} lost rows during the "
                    f"re-decode ({source_unique} -> {output_rows}); a repair "
                    f"may only recover members")
            unrecovered = entry.get("unrecovered_member_times") or []
            if len(unrecovered) > member_unique - output_rows:
                raise ValueError(
                    f"QC manifest store {entry['sid']}: {len(unrecovered)} "
                    f"unrecovered member times do not fit the "
                    f"{member_unique - output_rows} member times the output "
                    f"does not cover")
            failed = entry.get("decode_failures") or []
            if len(failed) > member_unique - output_rows + len(unrecovered):
                raise ValueError(
                    f"QC manifest store {entry['sid']}: {len(failed)} decode "
                    f"failures exceed the unrecovered member times")
        else:
            raise ValueError(
                f"QC manifest store {entry['sid']} has unknown source_kind "
                f"{kind!r}")
        for side in ("source", "output"):
            for key in ("array", "array_sha256", "times", "times_sha256"):
                if key not in entry[side]:
                    raise ValueError(
                        f"QC manifest store {entry['sid']} {side} lacks {key}")
            if not _is_sha256(entry[side]["array_sha256"]) or not _is_sha256(
                    entry[side]["times_sha256"]):
                raise ValueError(
                    f"QC manifest store {entry['sid']} {side} has a malformed "
                    f"sha256")
        if entry["qc"].get("rule") != QC_RULE_VERSION:
            raise ValueError(
                f"QC manifest store {entry['sid']} rule "
                f"{entry['qc'].get('rule')!r} != {QC_RULE_VERSION!r}")
    return {
        "qc_rule_version": QC_RULE_VERSION,
        "complete": True,
        "production": block.get("production") is not False
        and not any(audit_flags.values()),
        "audit_flags": audit_flags,
        "rekey_crosswalk_sha256": (block.get("rekey") or {}).get("sha256"),
        "out_frames_dir": str(block.get("out_frames_dir")),
        "source_frames_dir": str(block.get("source_frames_dir")),
        "n_stores": len(stores),
        "stores_sha256": block["stores_sha256"],
        "sids": sorted(sids),
        "sids_sha256": sha256_array(sorted(sids)),
        "counts": dict(block.get("counts") or {}),
        "n_stores_with_decode_failures": int(
            (block.get("counts") or {}).get("n_stores_with_decode_failures", 0)),
        "n_stores_with_unparsed_members": int(
            (block.get("counts") or {}).get("n_stores_with_unparsed_members", 0)),
        "n_unrecovered_member_times": int(
            (block.get("counts") or {}).get("n_unrecovered_member_times", 0)),
        "rekey": block.get("rekey"),
    }


def _row_uid(sid: Any, iso_time: Any) -> str:
    t = pd.to_datetime(iso_time, utc=True)
    if pd.isna(t):
        raise ValueError(f"invalid ISO_TIME for {sid!r}: {iso_time!r}")
    return f"{sid}|{pd.Timestamp(t).strftime('%Y-%m-%dT%H:%M:%SZ')}"


def verify_qc_store_files(block: Mapping[str, Any], frames_dir: Any, *,
                          sids: Iterable[str] | None = None,
                          verify_sources: bool = True,
                          expected_out_frames: Any | None = None,
                          require_production: bool = True) -> Dict[str, Any]:
    """Re-hash the QC corpus on disk against the manifest (production gate).

    Every store must exist for every manifest SID (or exactly the requested
    subset), its time axis must be strictly ascending and unique, its row count
    must equal ``output_rows`` == ``unique times`` == ``len(times)``, and the
    array/time-axis SHA-256 must match the manifest.  When the original (raw or
    re-keyed) store is still present its SHA-256 is re-checked as well, so a
    tampered corpus or a duplicated timestamp is refused instead of silently
    trained on.
    """
    from wpac_warning.frame_store import (frame_array_path, list_store_sids,
                                 require_frame_time_path)
    frames_root = Path(frames_dir)
    audit = validate_qc_manifest(block, expected_out_frames=expected_out_frames,
                                 require_production=require_production)
    stores = {str(e["sid"]): e for e in block["stores"]}
    on_disk = set(list_store_sids(frames_root))
    if sids is None:
        if on_disk != set(stores):
            missing = sorted(set(stores) - on_disk)[:5]
            extra = sorted(on_disk - set(stores))[:5]
            raise ValueError(
                f"QC corpus and QC manifest disagree: {len(missing)} manifest "
                f"stores missing on disk ({missing}), {len(extra)} unlisted "
                f"stores ({extra})")
        targets = sorted(stores)
    else:
        targets = sorted({str(s) for s in sids})
        unknown = sorted(set(targets) - set(stores))
        if unknown:
            raise ValueError(
                f"{len(unknown)} requested SIDs are not in the QC manifest "
                f"({unknown[:5]})")
        absent = sorted(set(targets) - on_disk)
        if absent:
            raise ValueError(
                f"{len(absent)} QC stores are missing on disk ({absent[:5]})")
    verified = 0
    for sid in targets:
        entry = stores[sid]
        array = frame_array_path(frames_root, sid)
        times_path = require_frame_time_path(frames_root, sid)
        times = np.asarray(np.load(times_path), dtype="datetime64[ns]").reshape(-1)
        n_rows = int(np.load(array, mmap_mode="r").shape[0])
        if int(entry["output_rows"]) != n_rows or int(
                entry["output_rows"]) != times.size:
            raise ValueError(
                f"{sid}: QC manifest declares {entry['output_rows']} rows but "
                f"the corpus holds {n_rows} frames / {times.size} times")
        if times.size and (np.isnat(times).any()
                           or (np.diff(times).astype("timedelta64[ns]")
                               .astype(np.int64) <= 0).any()):
            raise ValueError(
                f"{sid}: QC time axis is not strictly ascending/unique "
                f"({times_path})")
        if sha256_file(array) != entry["output"]["array_sha256"]:
            raise ValueError(f"{sid}: QC frame array SHA-256 differs from the "
                             f"manifest ({array})")
        if sha256_file(times_path) != entry["output"]["times_sha256"]:
            raise ValueError(f"{sid}: QC time axis SHA-256 differs from the "
                             f"manifest ({times_path})")
        if verify_sources:
            source_array = Path(str(entry["source"]["array"]))
            source_times = Path(str(entry["source"]["times"]))
            if source_array.is_file():
                if sha256_file(source_array) != entry["source"]["array_sha256"]:
                    raise ValueError(
                        f"{sid}: QC input array changed since the QC run "
                        f"({source_array})")
            if source_times.is_file():
                if sha256_file(source_times) != entry["source"]["times_sha256"]:
                    raise ValueError(
                        f"{sid}: QC input time axis changed since the QC run "
                        f"({source_times})")
        verified += 1
    return {"n_stores": len(stores), "n_verified": verified,
            "sids_sha256": sha256_array(targets),
            "stores_sha256": audit["stores_sha256"],
            "production": audit["production"]}


def validate_frame_availability(
        block: Mapping[str, Any], *, expected_qc: Mapping[str, Any] | None = None,
        eligible_row_uids: Iterable[str] | None = None,
        index_row_uids: Iterable[str] | None = None) -> Dict[str, Any]:
    """Fail-closed validation of ``frame_availability.json``.

    Rule ``frame_availability_v1``: an eligible holdout row is excluded only when
    the QC corpus has no frame at its own t0 timestamp; internal environment
    history gaps are *not* exclusions (the dataset/model mask them).  The manifest
    must list **every** excluded row with its reason, and nothing else: given the
    eligible row keys (and the frames-index keys) the validator proves
    ``excluded ∪ indexed == eligible`` and ``excluded ∩ indexed == ∅``.
    """
    if not isinstance(block, Mapping):
        raise ValueError("frame availability manifest must be a mapping")
    if block.get("rule_version") != FRAME_AVAILABILITY_RULE_VERSION:
        raise ValueError(
            f"frame availability rule_version={block.get('rule_version')!r} != "
            f"{FRAME_AVAILABILITY_RULE_VERSION!r}")
    if not str(block.get("rule") or "").strip():
        raise ValueError("frame availability manifest lacks its rule text")
    qc = block.get("qc")
    if not isinstance(qc, Mapping):
        raise ValueError("frame availability manifest lacks its QC binding")
    for key in ("path", "sha256", "stores_sha256"):
        if not qc.get(key):
            raise ValueError(f"frame availability qc binding lacks '{key}'")
    if expected_qc is not None:
        if str(qc["sha256"]).lower() != str(expected_qc.get("sha256", "")).lower():
            raise ValueError(
                "frame availability QC manifest hash differs from the packed "
                "QC manifest")
        if qc["stores_sha256"] != expected_qc.get("stores_sha256"):
            raise ValueError(
                "frame availability QC stores hash differs from the packed QC "
                "manifest")

    rows = block.get("rows")
    if not isinstance(rows, list):
        raise ValueError("frame availability manifest has no 'rows' list")
    seen = set()
    parsed = []
    for row in rows:
        for key in ("row_uid", "SID", "ISO_TIME", "split", "reason",
                    "first_frame", "last_frame"):
            if key not in row:
                raise ValueError(
                    f"frame availability row {row.get('row_uid')!r} lacks '{key}'")
        reason = str(row["reason"])
        if reason not in FRAME_AVAILABILITY_REASONS:
            raise ValueError(
                f"frame availability row {row['row_uid']} has reason {reason!r}; "
                f"allowed: {list(FRAME_AVAILABILITY_REASONS)}")
        split = str(row["split"])
        if split not in SPLIT_NAMES:
            raise ValueError(
                f"frame availability row {row['row_uid']} has split {split!r}")
        want = _row_uid(row["SID"], row["ISO_TIME"])
        if str(row["row_uid"]) != want:
            raise ValueError(
                f"frame availability row key mismatch: {row['row_uid']} != {want}")
        if want in seen:
            raise ValueError(f"duplicate frame availability row key: {want}")
        seen.add(want)
        if reason == "missing_store":
            if row["first_frame"] is not None or row["last_frame"] is not None:
                raise ValueError(
                    f"{want}: a missing-store row must not declare frame times")
        else:
            if not row["first_frame"] or not row["last_frame"]:
                raise ValueError(
                    f"{want}: reason {reason} requires the store's first/last "
                    f"frame time")
        parsed.append({"row_uid": want, "split": split, "reason": reason,
                       "SID": str(row["SID"])})

    n_excluded = int(block.get("n_excluded_rows", -1))
    n_eligible = int(block.get("n_eligible_rows", -1))
    n_available = int(block.get("n_available_rows", -1))
    if n_excluded != len(rows):
        raise ValueError(
            f"frame availability n_excluded_rows={n_excluded} but lists "
            f"{len(rows)} rows")
    if n_eligible != n_available + n_excluded or n_eligible < 0:
        raise ValueError(
            f"frame availability counts do not add up: eligible={n_eligible} "
            f"available={n_available} excluded={n_excluded}")
    excluded_keys = sorted(r["row_uid"] for r in parsed)
    if block.get("excluded_row_key_sha256") != sha256_array(excluded_keys):
        raise ValueError(
            "frame availability excluded_row_key_sha256 does not match its rows")
    if not _is_sha256(block.get("available_row_key_sha256")):
        raise ValueError(
            "frame availability lacks available_row_key_sha256")
    if int(block.get("n_dropped_no_frame", -1)) != n_excluded:
        raise ValueError(
            "frame availability n_dropped_no_frame must equal n_excluded_rows")

    # per-reason / per-split / per-year / per-storm accounting
    def _counts(items, key) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for item in items:
            out[str(item[key])] = out.get(str(item[key]), 0) + 1
        return out

    def _reason_norm(value: Any, where: str) -> Dict[str, int]:
        if value is None:
            return {reason: 0 for reason in FRAME_AVAILABILITY_REASONS}
        if not isinstance(value, Mapping):
            raise ValueError(f"{where} must be a mapping")
        unknown = sorted(set(map(str, value)) - set(FRAME_AVAILABILITY_REASONS))
        if unknown:
            raise ValueError(f"{where} has unknown reasons {unknown}")
        out = {reason: int(value.get(reason, 0))
               for reason in FRAME_AVAILABILITY_REASONS}
        if any(v < 0 for v in out.values()):
            raise ValueError(f"{where} has a negative count")
        return out

    observed_reason = {reason: 0 for reason in FRAME_AVAILABILITY_REASONS}
    for item in parsed:
        observed_reason[item["reason"]] += 1
    declared_reason = _reason_norm(block.get("by_reason"),
                                   "frame availability by_reason")
    if declared_reason != observed_reason:
        raise ValueError("frame availability by_reason counts disagree with rows")

    by_split = block.get("by_split") or {}
    if not isinstance(by_split, Mapping) or not by_split:
        raise ValueError("frame availability lacks by_split accounting")
    for split in SPLIT_NAMES:
        entry = by_split.get(split)
        if entry is None:
            continue
        rows_split = [r for r in parsed if r["split"] == split]
        if int(entry.get("n_excluded", -1)) != len(rows_split):
            raise ValueError(
                f"frame availability by_split[{split}].n_excluded mismatch")
        if entry.get("excluded_row_key_sha256") != sha256_array(
                sorted(r["row_uid"] for r in rows_split)):
            raise ValueError(
                f"frame availability by_split[{split}] excluded hash mismatch")
        if int(entry.get("n_eligible", -1)) != int(entry.get("n_available", -2)) \
                + int(entry.get("n_excluded", -1)):
            raise ValueError(
                f"frame availability by_split[{split}] counts do not add up")
        split_reason = {reason: 0 for reason in FRAME_AVAILABILITY_REASONS}
        for item in rows_split:
            split_reason[item["reason"]] += 1
        if _reason_norm(entry.get("by_reason"),
                        f"frame availability by_split[{split}].by_reason") \
                != split_reason:
            raise ValueError(
                f"frame availability by_split[{split}].by_reason mismatch")
        if not _is_sha256(entry.get("available_row_key_sha256")):
            raise ValueError(
                f"frame availability by_split[{split}] lacks its available hash")
    if sum(int(e.get("n_eligible", 0)) for e in by_split.values()) != n_eligible:
        raise ValueError("frame availability by_split n_eligible sums != total")
    if sum(int(e.get("n_available", 0)) for e in by_split.values()) != n_available:
        raise ValueError("frame availability by_split n_available sums != total")
    if sum(int(e.get("n_excluded", 0)) for e in by_split.values()) != n_excluded:
        raise ValueError("frame availability by_split n_excluded sums != total")

    by_year = block.get("by_year") or {}
    if not isinstance(by_year, Mapping) or not by_year:
        raise ValueError("frame availability lacks by_year accounting")
    for year, entry in by_year.items():
        if len(str(year)) != 4 or not str(year).isdigit():
            raise ValueError(f"frame availability by_year key {year!r} is not a "
                             f"4-digit year")
        rows_year = [r for r in parsed
                     if pd.Timestamp(r["row_uid"].split("|")[1]).year == int(year)]
        if int(entry.get("n_excluded", -1)) != len(rows_year):
            raise ValueError(
                f"frame availability by_year[{year}].n_excluded mismatch")
        if entry.get("excluded_row_key_sha256") != sha256_array(
                sorted(r["row_uid"] for r in rows_year)):
            raise ValueError(
                f"frame availability by_year[{year}] excluded hash mismatch")
    if sum(int(e.get("n_excluded", 0)) for e in by_year.values()) != n_excluded:
        raise ValueError("frame availability by_year n_excluded sums != total")
    if sum(int(e.get("n_available", 0)) for e in by_year.values()) != n_available:
        raise ValueError("frame availability by_year n_available sums != total")

    by_storm = block.get("by_storm") or {}
    if not isinstance(by_storm, Mapping) or not by_storm:
        raise ValueError("frame availability lacks by_storm accounting")
    for sid, entry in by_storm.items():
        rows_sid = [r for r in parsed if r["SID"] == str(sid)]
        if int(entry.get("n_excluded", -1)) != len(rows_sid):
            raise ValueError(
                f"frame availability by_storm[{sid}].n_excluded mismatch")
        if int(entry.get("n_eligible", -1)) != int(entry.get("n_available", -2)) \
                + int(entry.get("n_excluded", -1)):
            raise ValueError(
                f"frame availability by_storm[{sid}] counts do not add up")
        if int(entry.get("n_available", -1)) < 0:
            raise ValueError(
                f"frame availability by_storm[{sid}] has a negative count")
        if entry.get("excluded_row_key_sha256") != sha256_array(
                sorted(r["row_uid"] for r in rows_sid)):
            raise ValueError(
                f"frame availability by_storm[{sid}] excluded hash mismatch")
        storm_reason = {reason: 0 for reason in FRAME_AVAILABILITY_REASONS}
        for item in rows_sid:
            storm_reason[item["reason"]] += 1
        if _reason_norm(entry.get("by_reason"),
                        f"frame availability by_storm[{sid}].by_reason") \
                != storm_reason:
            raise ValueError(
                f"frame availability by_storm[{sid}].by_reason mismatch")
    if sum(int(e.get("n_available", 0)) for e in by_storm.values()) != n_available:
        raise ValueError("frame availability by_storm n_available sums != total")
    if sum(int(e.get("n_excluded", 0)) for e in by_storm.values()) != n_excluded:
        raise ValueError("frame availability by_storm n_excluded sums != total")

    alias = block.get("alias_coverage")
    if not isinstance(alias, Mapping):
        raise ValueError("frame availability lacks alias_coverage")
    for key in ("n_canonical_rows", "n_covered", "ratio", "first_missing",
                "last_missing"):
        if key not in alias:
            raise ValueError(f"frame availability alias_coverage lacks '{key}'")
    n_canon = int(alias["n_canonical_rows"])
    n_cov = int(alias["n_covered"])
    if not 0 <= n_cov <= n_canon:
        raise ValueError("frame availability alias_coverage counts are invalid")
    want_ratio = (n_cov / n_canon) if n_canon else None
    if alias["ratio"] != want_ratio:
        raise ValueError(
            f"frame availability alias_coverage ratio={alias['ratio']!r} != "
            f"{want_ratio!r}")
    if n_canon == 0 and (alias["first_missing"] or alias["last_missing"]):
        raise ValueError("frame availability alias_coverage has no canonical "
                         "rows but reports a missing frame time")

    # "all missing rows and only those rows": tie the manifest to the eligible set
    if eligible_row_uids is not None:
        eligible = {str(k) for k in eligible_row_uids}
        extra = sorted(seen - eligible)
        if extra:
            raise ValueError(
                f"{len(extra)} frame availability rows are not eligible panel "
                f"rows: {extra[:5]}")
        if index_row_uids is not None:
            indexed = {str(k) for k in index_row_uids}
            clash = sorted(seen & indexed)
            if clash:
                raise ValueError(
                    f"{len(clash)} rows are both excluded and in the frames "
                    f"index: {clash[:5]}")
            if (indexed | seen) != eligible:
                missing = sorted(eligible - indexed - seen)
                unexpected = sorted((indexed | seen) - eligible)
                raise ValueError(
                    "frame availability + frames index do not cover the "
                    f"eligible set exactly: {len(missing)} eligible rows "
                    f"unaccounted (first {missing[:5]}), {len(unexpected)} rows "
                    f"outside the eligible set (first {unexpected[:5]})")
        else:
            if n_available != len(eligible) - len(seen):
                raise ValueError(
                    f"frame availability n_available_rows={n_available} but the "
                    f"eligible set minus exclusions has "
                    f"{len(eligible) - len(seen)} rows")
    return {
        "rule_version": FRAME_AVAILABILITY_RULE_VERSION,
        "n_eligible_rows": n_eligible,
        "n_available_rows": n_available,
        "n_excluded_rows": n_excluded,
        "excluded_row_key_sha256": block["excluded_row_key_sha256"],
        "available_row_key_sha256": block["available_row_key_sha256"],
        "by_reason": declared_reason,
        "by_split": {str(k): dict(v) for k, v in by_split.items()},
        "by_year": {str(k): dict(v) for k, v in by_year.items()},
        "by_storm": {str(k): dict(v) for k, v in by_storm.items()},
        "alias_coverage": dict(alias),
        "missing_stores": [str(s) for s in (block.get("missing_stores") or [])],
        "qc": dict(qc),
    }


def declared_population(cfg_data: Mapping[str, Any]) -> str:
    pop = cfg_data.get("population")
    if pop in (None, ""):
        raise ValueError(
            "config data.population is required (use 'alltc'): without it the "
            "375-system >=65 kt sample could silently become the analysis "
            "population")
    return str(pop).strip().lower()


def require_population(cfg_data: Mapping[str, Any],
                       expected: str | None = ALLTC_POPULATION) -> str:
    pop = declared_population(cfg_data)
    if expected is not None and pop != expected:
        raise ValueError(
            f"config data.population={pop!r} but this protocol requires "
            f"{expected!r}: the 725-system all-TC panel is the only allowed "
            f"population (the 375-system >=65 kt sample is not a branch)")
    return pop


def require_frames_corpus(cfg_data: Mapping[str, Any],
                          expected: str = GLOBAL_HURSAT_CORPUS) -> str:
    """The configured corpus must be the QC v1 corpus, never the raw one."""
    corpus = cfg_data.get("frames_corpus")
    if corpus in (None, ""):
        raise ValueError(
            "config data.frames_corpus is required (the all-TC protocol reads "
            f"the QC v1 global HURSAT corpus {expected!r}, not data/frames)")
    corpus = str(corpus).strip().lower()
    if corpus == GLOBAL_HURSAT_CORPUS_RAW:
        raise ValueError(
            f"config data.frames_corpus={corpus!r} is the RAW download corpus; "
            f"training reads the QC corpus {expected!r} "
            f"(scripts/build_hursat_qc.py)")
    if corpus != expected:
        raise ValueError(
            f"config data.frames_corpus={corpus!r} but the holdout protocol "
            f"requires the global HURSAT QC corpus {expected!r}")
    return corpus


def _int_cfg(cfg_data: Mapping[str, Any], key: str, default: int) -> int:
    val = cfg_data.get(key, default)
    if val is None:
        return int(default)
    return int(val)


def population_expectations(cfg_data: Mapping[str, Any]) -> Dict[str, int]:
    """(n_systems, n_label_ineligible) declared by the config."""
    return {
        "n_systems": _int_cfg(cfg_data, "population_n_systems",
                              DEFAULT_POPULATION_N_SYSTEMS),
        "n_label_ineligible": _int_cfg(cfg_data, "population_n_label_ineligible",
                                       DEFAULT_POPULATION_N_LABEL_INELIGIBLE),
    }


POPULATION_HASH_KEYS = {
    "canonical_sample": ("content_sha256", "sids_sha256", "sid_years_sha256"),
    "label_eligible": ("content_sha256", "sids_sha256", "sid_years_sha256"),
    "split": ("content_sha256", "sids_sha256"),
    "panel": ("content_sha256", "sids_sha256", "row_key_sha256"),
    "prior": ("content_sha256", "sids_sha256", "row_key_sha256"),
}


def sha256_sid_years(sid_years: Mapping[str, Any] | Iterable[tuple]) -> str:
    """Content hash over the normalised ``SID|year`` set (order independent)."""
    items = (sid_years.items() if isinstance(sid_years, Mapping)
             else sid_years)
    return sha256_array(sorted(f"{str(s)}|{int(y)}" for s, y in items))


def sha256_sids(sids: Iterable[Any]) -> str:
    return sha256_array(sorted(str(s) for s in sids))


def sid_year_frame(path: str | Path) -> pd.DataFrame:
    """Read a canonical sample manifest (SID/YEAR, optional INCLUDED)."""
    p = require_file(path, "sample manifest")
    require_not_archive(p, "sample manifest")
    frame = pd.read_csv(p)
    cols = {c.lower(): c for c in frame.columns}
    if "sid" not in cols:
        raise ValueError(f"sample manifest {p} must contain a SID column")
    if "year" not in cols:
        raise ValueError(f"sample manifest {p} must contain a YEAR column")
    out = pd.DataFrame({"SID": frame[cols["sid"]].astype(str),
                        "YEAR": pd.to_numeric(frame[cols["year"]],
                                              errors="raise").astype(int)})
    if "included" in cols:
        inc = frame[cols["included"]]
        out["INCLUDED"] = (inc if inc.dtype == bool
                           else inc.astype(str).str.lower().isin(
                               ("true", "1", "yes")))
    else:
        out["INCLUDED"] = True
    if out["SID"].duplicated().any():
        dup = sorted(out.loc[out["SID"].duplicated(), "SID"])[:5]
        raise ValueError(f"duplicate SIDs in sample manifest {p}: {dup}")
    sid_year = out["SID"].str[:4]
    if not sid_year.str.fullmatch(r"\d{4}").all():
        raise ValueError(f"sample manifest {p} has SIDs without a YYYY prefix")
    bad = out.loc[out["YEAR"] != sid_year.astype(int), ["SID", "YEAR"]]
    if len(bad):
        raise ValueError(
            f"{len(bad)} sample-manifest rows have YEAR != SID first year "
            f"(first: {bad.head(5).to_dict('records')})")
    return out


def sample_manifest_audit(path: str | Path,
                          expect_n_included: int | None = None) -> Dict[str, Any]:
    """Normalised INCLUDED canonical SID/year set + content hashes.

    This is the binding between the phase-1 canonical sample (725 systems) and
    every downstream asset: the count alone is not enough, the SID/year
    content hash is.
    """
    p = require_file(path, "sample manifest")
    frame = sid_year_frame(p)
    inc = frame[frame["INCLUDED"]]
    if inc.empty:
        raise ValueError(f"sample manifest {p} has no INCLUDED=True rows")
    sids = sorted(inc["SID"].tolist())
    sid_years = {s: int(y) for s, y in zip(inc["SID"], inc["YEAR"])}
    if expect_n_included is not None and len(sids) != int(expect_n_included):
        raise ValueError(
            f"sample manifest {p} lists {len(sids)} INCLUDED systems but the "
            f"protocol requires {int(expect_n_included)} (a partial/foreign "
            f"manifest — e.g. the 375-system >=65 kt sample — must not pass)")
    return {
        "path": p,
        "sha256": sha256_file(p),
        "n_rows": int(len(frame)),
        "n_included": int(len(sids)),
        "n_excluded": int(len(frame) - len(sids)),
        "sids": sids,
        "sid_years": sid_years,
        "sids_sha256": sha256_sids(sids),
        "sid_years_sha256": sha256_array(
            sorted(f"{s}|{y}" for s, y in sid_years.items())),
        "content_sha256": sha256_sid_years(sid_years),
        "columns": [str(c) for c in frame.columns],
    }


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def _require_hash(value: Any, where: str) -> str:
    text = str(value or "").strip().lower()
    if not _is_sha256(text):
        raise ValueError(f"{where}: missing or malformed sha256 ({value!r})")
    return text


def validate_population_audit(block: Mapping[str, Any], *,
                              population: str = ALLTC_POPULATION,
                              expected_n_systems: int | None = None,
                              expected_n_label_ineligible: int | None = None,
                              ) -> Dict[str, Any]:
    """Fail-closed validation of the population audit block.

    Binds the canonical sample content hash (725 SID/year rows) and the
    label-eligible split/panel/prior sets; a block that only carries counts or
    years is rejected.  ``label_ineligible_sids`` must be disjoint from the
    label-eligible set (the 5 systems without an F24 label are excluded, never
    given invented rows).
    """
    if not isinstance(block, Mapping):
        raise ValueError("population audit block must be a mapping")
    got_pop = str(block.get("population", "")).strip().lower()
    if got_pop != population:
        raise ValueError(
            f"population audit declares population={block.get('population')!r};"
            f" the protocol requires {population!r}")
    n_systems = int(block.get("n_systems", -1))
    n_eligible = int(block.get("n_label_eligible", -1))
    n_ineligible = int(block.get("n_label_ineligible", -1))
    if n_systems <= 0 or n_eligible <= 0 or n_ineligible < 0:
        raise ValueError(
            "population audit lacks n_systems / n_label_eligible / "
            f"n_label_ineligible ({n_systems}/{n_eligible}/{n_ineligible})")
    if n_eligible + n_ineligible != n_systems:
        raise ValueError(
            f"population audit counts do not add up: label_eligible="
            f"{n_eligible} + label_ineligible={n_ineligible} != "
            f"n_systems={n_systems}")
    if expected_n_systems is not None and n_systems != int(expected_n_systems):
        raise ValueError(
            f"population audit n_systems={n_systems} but the protocol requires "
            f"{int(expected_n_systems)}")
    if (expected_n_label_ineligible is not None
            and n_ineligible != int(expected_n_label_ineligible)):
        raise ValueError(
            f"population audit n_label_ineligible={n_ineligible} but the "
            f"protocol requires {int(expected_n_label_ineligible)} label-less "
            f"systems to be excluded")
    excluded = [str(s) for s in block.get("label_ineligible_sids", [])]
    if len(excluded) != n_ineligible:
        raise ValueError(
            f"population audit lists {len(excluded)} label-ineligible SIDs but "
            f"declares {n_ineligible}")
    if len(set(excluded)) != len(excluded):
        raise ValueError("population audit repeats label-ineligible SIDs")
    for key, keys in POPULATION_HASH_KEYS.items():
        section = block.get(key)
        if not isinstance(section, Mapping):
            raise ValueError(f"population audit missing section '{key}'")
        for hkey in keys:
            _require_hash(section.get(hkey), f"population audit {key}.{hkey}")
        expected_sha = sha256_sids(section["sids"]) if section.get("sids") \
            else None
        if expected_sha and expected_sha != section["sids_sha256"]:
            raise ValueError(
                f"population audit {key}.sids_sha256 does not match its own "
                f"SID list")
        if section.get("sid_years"):
            want = sha256_sid_years(dict(section["sid_years"]))
            if want != section["content_sha256"]:
                raise ValueError(
                    f"population audit {key}.content_sha256 does not match its "
                    f"own SID/year list")
    for key in ("label_eligible", "split", "panel", "prior"):
        listed = set(map(str, (block.get(key) or {}).get("sids") or []))
        clash = sorted(listed & set(excluded))
        if clash:
            raise ValueError(
                f"label-ineligible systems appear in the {key} set: "
                f"{clash[:5]} (the systems without an F24 label must have no "
                f"rows at all)")
    parts = set(map(str, (block.get("split") or {}).get("sids") or []))
    for key in ("panel", "prior"):
        listed = set(map(str, (block.get(key) or {}).get("sids") or []))
        if parts and listed and not listed <= parts:
            extra = sorted(listed - parts)[:5]
            raise ValueError(f"{key} SIDs outside the label-eligible split: "
                             f"{extra}")
    return {
        "population": got_pop,
        "n_systems": n_systems,
        "n_label_eligible": n_eligible,
        "n_label_ineligible": n_ineligible,
        "label_ineligible_sids": excluded,
        "canonical_sample_content_sha256":
            block["canonical_sample"]["content_sha256"],
        "label_eligible_content_sha256":
            block["label_eligible"]["content_sha256"],
        "split_sids_sha256": block["split"]["sids_sha256"],
        "panel_row_key_sha256": block["panel"]["row_key_sha256"],
        "prior_row_key_sha256": block["prior"]["row_key_sha256"],
    }


def verify_identification_dag(dot_path: str | Path, json_path: str | Path,
                              status_csv: str | Path) -> Dict[str, str]:
    """The DoWhy identification DAG must match its JSON and the status table.

    ``dot_sha256`` is accepted in either convention the producers use: the raw
    file bytes (``hashlib.sha256(open(..., "rb"))``) or the text-normalised
    content the phase-1 builder hashes (``read_text()`` -> UTF-8, which folds
    CRLF to LF on Windows).
    """
    dot = require_file(dot_path, "identification DAG (.dot)")
    js = require_file(json_path, "identification DAG provenance (.json)")
    payload = json.loads(js.read_text(encoding="utf-8"))
    dot_bytes = sha256_file(dot)
    dot_text = hashlib.sha256(
        dot.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    declared = payload.get("dot_sha256")
    if not declared:
        raise ValueError(f"{js} does not declare dot_sha256; the identification "
                         f"DAG provenance cannot be verified")
    if str(declared).lower() not in (dot_bytes, dot_text):
        raise ValueError(
            f"{js} dot_sha256={declared} does not match {dot} "
            f"(bytes {dot_bytes}, text {dot_text})")
    status = pd.read_csv(require_file(status_csv, "identification csv"))
    if "dag_source" not in status.columns:
        raise ValueError(
            f"{status_csv} lacks 'dag_source'; the identification status must "
            f"record which DAG produced it")
    want = Path(dot).name
    sources = {Path(str(v)).name for v in status["dag_source"].astype(str)}
    if sources != {want}:
        raise ValueError(
            f"{status_csv} was not produced from {want}: dag_source={sources}")
    return {"dot": dot_bytes, "dot_text_sha256": dot_text,
            "json": sha256_file(js), "dot_name": want,
            "hash_convention": ("bytes" if str(declared).lower() == dot_bytes
                                else "text")}


# ----------------------------------------------------------------------
# split protocol
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SplitProtocol:
    fit_years: Tuple[int, int]
    inner_years: Tuple[int, int]
    test_years: Tuple[int, int]
    label_horizon_h: int = DEFAULT_LABEL_HORIZON_H
    label_tolerance_h: int = DEFAULT_LABEL_TOLERANCE_H
    label_time_col: str = "F24_TIME"
    name: str = PROTOCOL_NAME
    population: str = ALLTC_POPULATION
    n_systems: Optional[int] = None
    n_label_ineligible: Optional[int] = None

    @classmethod
    def from_cfg(cls, cfg) -> "SplitProtocol":
        data = cfg.to_dict().get("data", {}) if hasattr(cfg, "to_dict") else {}
        expect = population_expectations(data)
        return cls(
            fit_years=tuple(_as_year_list(cfg, "data.fit_years")),
            inner_years=tuple(_as_year_list(cfg, "data.inner_years")),
            test_years=tuple(_as_year_list(cfg, "data.test_years")),
            label_horizon_h=int(cfg.get("data.label_horizon_h",
                                        DEFAULT_LABEL_HORIZON_H)),
            label_tolerance_h=int(cfg.get("data.label_tolerance_h",
                                          DEFAULT_LABEL_TOLERANCE_H)),
            label_time_col=str(cfg.get("data.label_time_col", "F24_TIME")),
            name=str(cfg.get("data.protocol", PROTOCOL_NAME)),
            population=str(data.get("population") or ALLTC_POPULATION).strip().lower(),
            n_systems=expect["n_systems"],
            n_label_ineligible=expect["n_label_ineligible"])


    @property
    def embargo_h(self) -> int:
        return self.label_horizon_h + self.label_tolerance_h

    def years_of(self, split: str) -> List[int]:
        return year_range({"fit": self.fit_years, "inner": self.inner_years,
                           "test": self.test_years}[split])

    def require_alltc(self) -> str:
        """Fail-closed population guard for the holdout protocol."""
        if self.population != ALLTC_POPULATION:
            raise ValueError(
                f"protocol population={self.population!r}; the holdout protocol "
                f"requires {ALLTC_POPULATION!r} (725 WNP systems). The "
                f"375-system >=65 kt sample is not a branch")
        return self.population

    def as_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.name,
            "fit_years": list(self.fit_years),
            "inner_years": list(self.inner_years),
            "test_years": list(self.test_years),
            "label_horizon_h": self.label_horizon_h,
            "label_tolerance_h": self.label_tolerance_h,
            "label_time_col": self.label_time_col,
            "embargo_h": self.embargo_h,
            "population": self.population,
            "n_systems": self.n_systems,
            "n_label_ineligible": self.n_label_ineligible,
        }

    def validate(self) -> None:
        if not (self.fit_years[1] < self.inner_years[0]
                <= self.inner_years[1] < self.test_years[0]):
            raise ValueError(
                "splits must be ordered and non-overlapping: "
                f"fit={self.fit_years} inner={self.inner_years} "
                f"test={self.test_years}")


def load_storm_year_manifest(path: str | Path) -> pd.Series:
    """SID -> canonical storm_year from the phase-1 sample manifest.

    Requires columns SID, YEAR and (when present) INCLUDED.  Only INCLUDED
    rows define the analysis sample; duplicate SIDs with conflicting years
    are a hard error.
    """
    p = require_file(path, "sample manifest")
    m = pd.read_csv(p)
    cols = {c.lower(): c for c in m.columns}
    if "sid" not in cols or "year" not in cols:
        raise ValueError(f"sample manifest {p} must contain SID and YEAR")
    sid_col, year_col = cols["sid"], cols["year"]
    if "included" in cols:
        inc = m[cols["included"]]
        inc = inc.astype(str).str.lower().isin(("true", "1", "yes"))
        m = m[inc]
    if m.empty:
        raise ValueError(f"sample manifest {p} has no INCLUDED=True rows")
    if m[sid_col].duplicated().any():
        dup = m.loc[m[sid_col].duplicated(), sid_col].tolist()
        raise ValueError(f"duplicate SIDs in sample manifest: {dup[:5]}")
    years = pd.to_numeric(m[year_col], errors="raise").astype(int)
    return pd.Series(years.values, index=m[sid_col].astype(str).values,
                     name="storm_year")


def attach_storm_year(df: pd.DataFrame, manifest: pd.Series,
                      sid_col: str = "SID") -> pd.DataFrame:
    """Attach the canonical storm_year; every panel SID must be present."""
    missing = sorted(set(df[sid_col].astype(str)) - set(manifest.index))
    if missing:
        raise ValueError(
            f"{len(missing)} panel SIDs absent from the sample manifest "
            f"(first 10: {missing[:10]})")
    out = df.copy()
    out["storm_year"] = out[sid_col].astype(str).map(manifest).astype(int)
    return out


def assign_split(storm_year: pd.Series | np.ndarray,
                 proto: SplitProtocol) -> np.ndarray:
    """SID-level split label; anything outside the protocol is 'excluded'."""
    sy = np.asarray(storm_year, dtype=int)
    out = np.full(sy.shape, "excluded", dtype=object)
    for split in SPLIT_NAMES:
        lo, hi = getattr(proto, f"{split}_years")
        out[(sy >= lo) & (sy <= hi)] = split
    return out


def _next_boundary(split: str, proto: SplitProtocol) -> Optional[pd.Timestamp]:
    """Timestamp at which the split after `split` begins (UTC)."""
    if split == "fit":
        return pd.Timestamp(f"{proto.inner_years[0]}-01-01", tz="UTC")
    if split == "inner":
        return pd.Timestamp(f"{proto.test_years[0]}-01-01", tz="UTC")
    return None


def window_mask(iso_time: pd.Series, split: np.ndarray,
                proto: SplitProtocol, row_year: np.ndarray | None = None) -> np.ndarray:
    """Keep a row only when BOTH the storm and the row fall in the split."""
    t = pd.to_datetime(iso_time, utc=True)
    if row_year is None:
        row_year = t.dt.year.values
    row_year = np.asarray(row_year, dtype=int)
    keep = np.zeros(len(t), dtype=bool)
    for name in SPLIT_NAMES:
        lo, hi = getattr(proto, f"{name}_years")
        in_split = np.asarray(split == name)
        keep |= in_split & (row_year >= lo) & (row_year <= hi)
    return keep


def embargo_keep(iso_time: pd.Series, split_labels: np.ndarray,
                 proto: SplitProtocol) -> np.ndarray:
    """30 h label embargo on the tail of fit and inner (STRICT inequality).

    The DV24/RI label at t looks at t + 24 h (+/-6 h tolerance).  A row is
    kept only if its label window cannot even touch the next split's first
    day: ``t + embargo_h < next_split_start``.  Equality is rejected.
    Test rows are never dropped.
    """
    t = pd.to_datetime(iso_time, utc=True)
    labels = np.asarray(split_labels)
    keep = np.ones(len(t), dtype=bool)
    delta = pd.Timedelta(hours=proto.embargo_h)
    for name in ("fit", "inner"):
        boundary = _next_boundary(name, proto)
        if boundary is None:
            continue
        sel = labels == name
        if sel.any():
            keep[sel] = ((t[sel] + delta) < boundary).values
    return keep


def label_time_keep(label_time: pd.Series, split_labels: np.ndarray,
                    proto: SplitProtocol) -> np.ndarray:
    """The row's ACTUAL label timestamp must be strictly inside its split.

    Uses the panel's ``F24_TIME`` (the timestamp of VMAX(t+24h)), not an
    assumed t+24h: a row whose realised label time reaches the next split's
    first day is deleted.  NaT counts as a violation (fail-closed).
    """
    lt = pd.to_datetime(label_time, utc=True)
    labels = np.asarray(split_labels)
    keep = np.ones(len(lt), dtype=bool)
    for name in ("fit", "inner"):
        boundary = _next_boundary(name, proto)
        if boundary is None:
            continue
        sel = labels == name
        if sel.any():
            keep[sel] = (lt[sel] < boundary).values
    return keep


def _iso_str(t: pd.Series) -> pd.Series:
    return t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def assign_rows(df: pd.DataFrame, proto: SplitProtocol,
                sid_col: str = "SID",
                time_col: str = "ISO_TIME",
                label_time_col: Optional[str] = None) -> pd.DataFrame:
    """Full row-level split assignment with window + embargo + label-time.

    Adds: storm_year, split, row_year, row_uid, iso_time_str, keep_window,
    keep_embargo, keep_label_time, eligible.  ``eligible`` requires ALL of

      * the SID's storm_year and the row's calendar year inside the split;
      * ``t + embargo_h < next_split_start``       (strict, fit/inner only);
      * the row's actual ``F24_TIME < next_split_start`` (strict).

    The label-time column is required under the holdout protocol: an assumed
    t+24h is not accepted.
    """
    out = df.copy()
    if "storm_year" not in out.columns:
        raise ValueError("assign_rows requires a canonical storm_year column")
    col = label_time_col or proto.label_time_col
    if col not in out.columns:
        raise ValueError(
            f"panel is missing the label-time column '{col}' required by "
            f"the holdout boundary rule")
    t = pd.to_datetime(out[time_col], utc=True)
    out["row_year"] = t.dt.year.astype(int)
    out["iso_time_str"] = _iso_str(t)
    out["row_uid"] = out[sid_col].astype(str) + "|" + out["iso_time_str"]
    if out["row_uid"].duplicated().any():
        raise ValueError("duplicate (SID, ISO_TIME) rows before split assignment")
    out["split"] = assign_split(out["storm_year"], proto)
    out["keep_window"] = window_mask(t, out["split"].values, proto,
                                     out["row_year"].values)
    out["keep_embargo"] = embargo_keep(t, out["split"].values, proto)
    out["keep_label_time"] = label_time_keep(out[col], out["split"].values, proto)
    out["eligible"] = (out["split"].isin(SPLIT_NAMES)
                       & out["keep_window"] & out["keep_embargo"]
                       & out["keep_label_time"])
    return out


def split_summary(df: pd.DataFrame, proto: SplitProtocol) -> Dict[str, Any]:
    """Counts and row-key audits; the test RI rate is never computed or stored."""
    t = pd.to_datetime(df["ISO_TIME"], utc=True)
    rows: Dict[str, Any] = {"protocol": proto.as_dict(), "splits": {}}
    all_keys: List[str] = []
    for split in SPLIT_NAMES:
        s = df["split"] == split
        e = s & df["eligible"]
        sids = sorted(df.loc[e, "SID"].astype(str).unique().tolist())
        keys = df.loc[e, "row_uid"].astype(str).tolist()
        all_keys.extend(keys)
        if proto.label_time_col in df.columns:
            lt = pd.to_datetime(df.loc[e, proto.label_time_col], utc=True)
            label_time_max = str(lt.max()) if e.any() else None
        else:
            label_time_max = None
        rows["splits"][split] = {
            "n_rows_raw": int(s.sum()),
            "n_rows_eligible": int(e.sum()),
            "n_sids": len(sids),
            "sids": sids,
            "sids_sha256": sha256_array(sids),
            "row_key_sha256": sha256_array(sorted(keys)),
            "ri_rate_eligible": (float(df.loc[e, "RI"].mean())
                                 if split != "test" and "RI" in df.columns
                                 and e.any() else None),
            "year_min": int(t[e].dt.year.min()) if e.any() else None,
            "year_max": int(t[e].dt.year.max()) if e.any() else None,
            "time_max": str(t[e].max()) if e.any() else None,
            "label_time_max": label_time_max,
            "dropped_out_of_window": int((s & ~df["keep_window"]).sum()),
            "dropped_embargo": int(
                (s & df["keep_window"] & ~df["keep_embargo"]).sum()),
            "dropped_label_time": int(
                (s & df["keep_window"] & df["keep_embargo"]
                 & ~df["keep_label_time"]).sum()),
        }
    rows["n_excluded_rows"] = int((df["split"] == "excluded").sum())
    rows["eligible_row_key_sha256"] = sha256_array(sorted(all_keys))
    overlap = {}
    sets = {sp: set(rows["splits"][sp]["sids"]) for sp in SPLIT_NAMES}
    for a, b in (("fit", "inner"), ("fit", "test"), ("inner", "test")):
        inter = sets[a] & sets[b]
        if inter:
            raise ValueError(f"SID leaked across {a}/{b}: {sorted(inter)[:10]}")
        overlap[f"{a}|{b}"] = 0
    rows["sid_overlap"] = overlap
    return rows



# ----------------------------------------------------------------------
# prior bundle (graph / ATE / identification / transforms)
# ----------------------------------------------------------------------

_PRIOR_FILE_KEYS = ("panel", "graph_edges", "ate", "pcmci", "identification")
_PRIOR_ASSET_KEYS = ("dv24_transformer", "tbb_norm")
_PRIOR_OPTIONAL_ASSET_KEYS = ("frames_index", "feature_scaler")


def _norm_entry(entry: Any, base: Path) -> Dict[str, str]:
    """Resolve a manifest file entry (absolute, manifest-relative or P2-relative).

    Archive snapshots are never a protocol input, so a manifest that points into
    ``archive/`` fails here rather than at load time.
    """
    if isinstance(entry, str):
        entry = {"path": entry}
    out = dict(entry)
    p = Path(out["path"])
    if not p.is_absolute():
        cand = (base / p)
        if not cand.exists():
            cand = (Path(__file__).resolve().parents[2] / p)
        out["path"] = str(cand.resolve())
    require_not_archive(out["path"], "prior manifest entry")
    return out



def load_prior_bundle(path: str | Path, proto: SplitProtocol,
                      allow_fixture: bool = False) -> Dict[str, Any]:
    """Load and fully validate the fit-only prior manifest.

    Checks (all hard errors):
      * protocol name, population == alltc and exact fit-year set;
      * mask_type == xyz;
      * every referenced file exists and its SHA-256 matches;
      * the manifest's SID list covers no year outside fit;
      * required identification/ATE/DAG files are present and consistent;
      * the population audit binds the canonical sample content hash and the
        label-eligible split/panel/prior sets (counts alone are not enough).
    Returns the manifest with resolved absolute paths and verified hashes.
    """
    mp = require_file(path, "prior manifest")
    raw = json.loads(mp.read_text(encoding="utf-8"))
    base = mp.parent
    if raw.get("fixture") and not allow_fixture:
        raise ValueError(f"prior manifest {mp} is marked fixture=true")

    manifest_protocol = raw.get("protocol")
    if manifest_protocol != proto.name:
        raise ValueError(f"prior manifest protocol {manifest_protocol!r} != "
                         f"{proto.name!r}")
    years = [int(y) for y in raw.get("years", [])]
    expected = year_range(proto.fit_years)
    if sorted(years) != expected:
        raise ValueError(
            f"prior manifest years must equal the fit years {expected}, "
            f"got {sorted(years)}")
    if raw.get("mask_type") != "xyz":
        raise ValueError(f"prior manifest mask_type must be 'xyz', "
                         f"got {raw.get('mask_type')!r}")
    proto.require_alltc()

    files = raw.get("files", {})
    resolved: Dict[str, Dict[str, str]] = {}
    for key in _PRIOR_FILE_KEYS:
        if key not in files:
            raise ValueError(f"prior manifest missing required file entry "
                             f"'{key}'")
        ent = _norm_entry(files[key], base)
        got = verify_sha256(ent["path"], ent["sha256"], f"prior {key}")
        resolved[key] = {"path": str(Path(ent["path"]).resolve()),
                         "sha256": got}
    assets: Dict[str, Dict[str, str]] = {}

    def _load_asset(key: str, required: bool) -> None:
        raw_assets = raw.get("assets", {})
        if key not in raw_assets:
            if required:
                raise ValueError(
                    f"prior manifest missing required asset entry '{key}'")
            return
        ent = _norm_entry(raw_assets[key], base)
        got = verify_sha256(ent["path"], ent["sha256"], f"prior {key}")
        assets[key] = {k: v for k, v in ent.items()
                       if k not in ("path", "sha256")}
        assets[key].update({"path": str(Path(ent["path"]).resolve()),
                            "sha256": got})
        if "fit_years" in ent:
            fy = [int(y) for y in ent["fit_years"]]
            if fy != list(proto.fit_years):
                raise ValueError(
                    f"prior asset {key} declares fit_years={fy}, expected "
                    f"{list(proto.fit_years)} — the asset was fitted on the "
                    f"wrong window")
            assets[key]["fit_years"] = fy

    for key in _PRIOR_ASSET_KEYS:
        _load_asset(key, required=True)
    for key in _PRIOR_OPTIONAL_ASSET_KEYS:
        _load_asset(key, required=False)

    sids = raw.get("sids")
    if sids is None:
        raise ValueError("prior manifest must list its SIDs ('sids')")
    sids = [str(s) for s in sids]
    if "n_storms" in raw and int(raw["n_storms"]) != len(sids):
        raise ValueError(
            f"prior manifest n_storms={raw['n_storms']} but lists "
            f"{len(sids)} SIDs")
    sy = pd.Series([int(str(s)[:4]) for s in sids])
    # storm SIDs are YYYYJJJ...; a cheap year guard, the authoritative check
    # is the year-column cross-check below when present.
    bad = sy[(sy < proto.fit_years[0]) | (sy > proto.fit_years[1])]
    if len(bad):
        raise ValueError(f"prior manifest lists {len(bad)} SIDs outside the "
                         f"fit years (first: {bad.head(5).tolist()})")
    if "sid_years" not in raw:
        raise ValueError(
            "prior manifest lacks 'sid_years'; the canonical SID->year mapping "
            "must be bound explicitly (a SID prefix is not enough)")
    if len(raw["sid_years"]) != len(sids):
        raise ValueError("prior manifest sid_years length != sids length")
    bad = [s for s, y in zip(raw["sids"], raw["sid_years"])
           if int(str(s)[:4]) != int(y) or int(y) > proto.fit_years[1]]
    if bad:
        raise ValueError(f"prior manifest SID years disagree with fit: {bad[:5]}")
    if "row_key_sha256" not in raw:
        raise ValueError("prior manifest lacks 'row_key_sha256'")
    _require_hash(raw["row_key_sha256"], "prior manifest row_key_sha256")

    # --- identification DAG (dot + provenance json + status) ---
    dag = raw.get("identification_dag")
    if not isinstance(dag, Mapping):
        raise ValueError(
            "prior manifest missing the 'identification_dag' section "
            "({'dot': ..., 'json': ...} with SHA-256 entries)")
    dag_paths: Dict[str, str] = {}
    for key in ("dot", "json"):
        ent = dag.get(key)
        if not isinstance(ent, Mapping) or "path" not in ent:
            raise ValueError(f"prior manifest identification_dag.{key} entry "
                             f"is missing its path")
        ent = _norm_entry(ent, base)
        got = verify_sha256(ent["path"], ent["sha256"],
                            f"prior identification_dag.{key}")
        dag_paths[key] = str(Path(ent["path"]).resolve())
        dag_paths[f"{key}_sha256"] = got
    dag_audit = verify_identification_dag(dag_paths["dot"], dag_paths["json"],
                                          resolved["identification"]["path"])

    # --- frames QC binding (the corpus the trainer reads) ---
    fqc = raw.get("frames_qc")
    if not isinstance(fqc, Mapping):
        raise ValueError(
            "prior manifest missing the 'frames_qc' block (QC v1 manifest "
            "path + SHA-256); the frame corpus provenance is not bound")
    fqc_ent = _norm_entry({"path": fqc.get("path"),
                           "sha256": fqc.get("sha256")}, base)
    fqc_sha = verify_sha256(fqc_ent["path"], fqc_ent["sha256"],
                            "frames QC manifest")
    if fqc.get("qc_rule_version") != QC_RULE_VERSION:
        raise ValueError(
            f"prior manifest frames_qc rule {fqc.get('qc_rule_version')!r} != "
            f"{QC_RULE_VERSION!r}")
    fqc_audit = validate_qc_manifest(
        json.loads(Path(fqc_ent["path"]).read_text(encoding="utf-8")))
    if fqc_audit["stores_sha256"] != fqc.get("stores_sha256"):
        raise ValueError("prior manifest frames_qc stores_sha256 disagrees "
                         "with the QC manifest")

    # --- frame availability (which eligible rows have an exact QC frame) ---
    fav = raw.get("frame_availability")
    if not isinstance(fav, Mapping):
        raise ValueError(
            "prior manifest missing the 'frame_availability' block (row-level "
            "exclusion manifest path + SHA-256)")
    fav_ent = _norm_entry({"path": fav.get("path"),
                           "sha256": fav.get("sha256")}, base)
    fav_sha = verify_sha256(fav_ent["path"], fav_ent["sha256"],
                            "frame availability manifest")
    fav_block = json.loads(Path(fav_ent["path"]).read_text(encoding="utf-8"))
    fav_audit = validate_frame_availability(
        fav_block, expected_qc={"sha256": fqc_sha,
                                "stores_sha256": fqc_audit["stores_sha256"]})
    for key in ("rule_version", "n_eligible_rows", "n_available_rows",
                "n_excluded_rows", "excluded_row_key_sha256",
                "available_row_key_sha256"):
        if fav.get(key) != fav_audit.get(key):
            raise ValueError(
                f"prior manifest frame_availability.{key} disagrees with the "
                f"manifest file")
    pop_ref = fav_block.get("population_audit")
    if isinstance(pop_ref, Mapping):
        pop_path = _norm_entry({"path": pop_ref.get("path"),
                                "sha256": pop_ref.get("sha256")}, base)
        verify_sha256(pop_path["path"], pop_path["sha256"],
                      "frame availability population audit")
        prior_pop = raw.get("population") or {}
        if prior_pop.get("sha256") and str(
                pop_ref.get("sha256", "")).lower() != str(
                    prior_pop["sha256"]).lower():
            raise ValueError(
                "frame availability population_audit differs from the prior "
                "manifest population audit")
    # every prior SID with at least one AVAILABLE row must have a QC store; a
    # storm whose rows are all excluded (missing_store) may stay in the prior
    # panel (the prior is row-level and built before availability)
    covered = {str(sid) for sid, entry in (fav_audit["by_storm"] or {}).items()
               if int(entry.get("n_available", 0)) > 0}
    missing = sorted(set(map(str, raw["sids"])) & covered
                     - set(fqc_audit["sids"]))
    if missing:
        raise ValueError(
            f"{len(missing)} prior SIDs have available rows but no QC frame "
            f"store: {missing[:5]}")
    fit_entry = (fav_audit["by_split"] or {}).get("fit") or {}
    if "rows" in raw and int(raw["rows"]) != int(fit_entry.get("n_eligible", -1)):
        raise ValueError(
            f"prior manifest rows={raw['rows']} != frame availability fit "
            f"n_eligible={fit_entry.get('n_eligible')}; the prior was built "
            f"from a different eligible set")

    # --- population binding (725 canonical vs label-eligible set) ---
    pop = raw.get("population")
    if not isinstance(pop, Mapping):
        raise ValueError(
            "prior manifest missing the 'population' block (canonical sample "
            "content hash + label-eligible split/panel/prior sets)")
    if "path" not in pop:
        raise ValueError("prior manifest population block lacks its audit path")
    pop_ent = _norm_entry({"path": pop["path"], "sha256": pop.get("sha256")},
                          base)
    pop_sha = verify_sha256(pop_ent["path"], pop_ent["sha256"],
                            "population audit")
    audit = json.loads(Path(pop_ent["path"]).read_text(encoding="utf-8"))
    summary = validate_population_audit(
        audit,
        expected_n_systems=proto.n_systems,
        expected_n_label_ineligible=proto.n_label_ineligible)
    if pop.get("canonical_sample") or pop.get("n_systems"):
        inline = {k: pop[k] for k in
                  ("population", "n_systems", "n_label_eligible",
                   "n_label_ineligible", "label_ineligible_sids",
                   "canonical_sample", "label_eligible", "split", "panel",
                   "prior") if k in pop}
        inline_summary = validate_population_audit(
            inline, expected_n_systems=proto.n_systems,
            expected_n_label_ineligible=proto.n_label_ineligible)
        for key in ("canonical_sample_content_sha256",
                    "label_eligible_content_sha256", "split_sids_sha256",
                    "panel_row_key_sha256", "prior_row_key_sha256"):
            if inline_summary[key] != summary[key]:
                raise ValueError(
                    f"prior manifest population block disagrees with its audit "
                    f"file on {key}")
    if pop.get("population") and str(pop["population"]).lower() != \
            ALLTC_POPULATION:
        raise ValueError(
            f"prior manifest population={pop.get('population')!r}; the "
            f"protocol requires {ALLTC_POPULATION!r}")

    out = dict(raw)
    out["manifest_path"] = str(mp.resolve())
    out["manifest_sha256"] = sha256_file(mp)
    out["files"] = resolved
    out["assets"] = assets
    out["years"] = expected
    out["identification_dag"] = {
        "dot": {"path": dag_paths["dot"], "sha256": dag_paths["dot_sha256"]},
        "json": {"path": dag_paths["json"], "sha256": dag_paths["json_sha256"]},
        "dot_name": dag_audit["dot_name"],
        "dot_sha256_text": dag_audit["dot_text_sha256"],
        "hash_convention": dag_audit["hash_convention"],
    }
    out["population"] = {"path": str(Path(pop_ent["path"]).resolve()),
                         "sha256": pop_sha, **summary}
    out["frames_qc"] = {"path": str(Path(fqc_ent["path"]).resolve()),
                        "sha256": fqc_sha, **fqc_audit}
    out["frame_availability"] = {"path": str(Path(fav_ent["path"]).resolve()),
                                 "sha256": fav_sha, **fav_audit}
    return out


def load_trusted_signs(ate_csv: str | Path,
                       identification_csv: str | Path,
                       exclude: Iterable[str] = (),
                       outcome: str = "DV24",
                       p_boot_max: float = 0.05,
                       min_refutations: int = 2,
                       strict: bool = True) -> Tuple[Dict[str, float],
                                                     List[Dict[str, Any]]]:
    """Fail-closed trusted sign set.

    A treatment enters only when ALL of the following hold:
      * ATE row for `outcome` with p_value_boot < p_boot_max;
      * refutation_passed >= min_refutations;
      * an identification row exists for (treatment, outcome)  [else drop];
      * identification says no_directed_path is False             [else drop];
      * treatment is not in the explicit exclude list.
    Returns ({treatment: +1/-1}, audit rows).
    """
    ate = pd.read_csv(require_file(ate_csv, "ATE csv"))
    for col in ("treatment", "outcome", "ate", "p_value_boot",
                "refutation_passed"):
        if col not in ate.columns:
            raise ValueError(f"ATE csv missing column '{col}'")
    ident = pd.read_csv(require_file(identification_csv, "identification csv"))
    for col in ("treatment", "outcome", "no_directed_path"):
        if col not in ident.columns:
            raise ValueError(f"identification csv missing column '{col}'")
    exclude = {str(t) for t in exclude}

    idmap: Dict[str, bool] = {}
    for _, r in ident[ident["outcome"] == outcome].iterrows():
        idmap[str(r["treatment"])] = bool(r["no_directed_path"])

    signs: Dict[str, float] = {}
    audit: List[Dict[str, Any]] = []
    dv = ate[ate["outcome"] == outcome]
    for _, r in dv.iterrows():
        trt = str(r["treatment"])
        row = {"treatment": trt, "ate": float(r["ate"]),
               "p_value_boot": float(r["p_value_boot"]),
               "refutation_passed": int(r["refutation_passed"]),
               "in_exclude": trt in exclude,
               "has_identification": trt in idmap,
               "no_directed_path": idmap.get(trt),
               "trusted": False, "reason": ""}
        if trt in exclude:
            row["reason"] = "explicit_exclude"
        elif not (row["p_value_boot"] < p_boot_max):
            row["reason"] = "bootstrap_ns"
        elif not (row["refutation_passed"] >= min_refutations):
            row["reason"] = "refutation_failed"
        elif trt not in idmap:
            row["reason"] = "identification_missing"
        elif idmap[trt]:
            row["reason"] = "no_directed_path"
        else:
            signs[trt] = float(np.sign(row["ate"]))
            row["trusted"] = True
            row["reason"] = "trusted"
        audit.append(row)
    if strict and not signs:
        raise ValueError(
            f"no trusted treatments for outcome={outcome}: all failed the "
            f"fail-closed filter ({[a['reason'] for a in audit]})")
    return signs, audit


# ----------------------------------------------------------------------
# versioned run directory
# ----------------------------------------------------------------------

@dataclass
class RunDir:
    root: Path
    run_id: str
    resume: bool = False
    path: Path = field(init=False)
    manifest_path: Path = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.path = self.root / self.run_id
        self.manifest_path = self.path / "run_manifest.json"

    def prepare(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Create (or explicitly resume) the run directory.

        A non-empty directory without `resume=True` is a hard error; with
        `resume=True` the stored fingerprint must match exactly.
        """
        fingerprint = sha256_json(dict(payload))
        if self.path.exists():
            if not self.resume:
                raise RuntimeError(
                    f"run directory already exists: {self.path} — pass "
                    f"--resume to continue it (implicit resume is disabled)")
            if not self.manifest_path.is_file():
                raise RuntimeError(
                    f"cannot resume {self.path}: run_manifest.json missing")
            old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if old.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "resume refused: run fingerprint changed "
                    f"(old {old.get('fingerprint')} != new {fingerprint}). "
                    "Use a new run_id for a different protocol/prior/config.")
            return old
        self.path.mkdir(parents=True, exist_ok=False)
        for sub in ("preds", "checkpoints", "logs"):
            (self.path / sub).mkdir(exist_ok=True)
        record = dict(payload)
        record["fingerprint"] = fingerprint
        record["run_id"] = self.run_id
        self.manifest_path.write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        return record

    def write_json(self, name: str, payload: Any) -> Path:
        p = self.path / name
        p.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                default=str), encoding="utf-8")
        return p

    def write_config(self, cfg_dict: Mapping[str, Any]) -> Path:
        import yaml
        p = self.path / "config.yaml"
        p.write_text(yaml.safe_dump(dict(cfg_dict), sort_keys=False),
                     encoding="utf-8")
        return p

    def file_hashes(self) -> Dict[str, str]:
        out = {}
        for p in sorted(self.path.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(self.path))] = sha256_file(p)
        return out


def resolve_path(root: str | Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (Path(root) / p)


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")
