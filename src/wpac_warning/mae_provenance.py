# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Fail-closed QC and provenance checks for downstream MAE encoder weights.

Two legal self-supervised corpora: the QC v1 deduplicated cache
(``mae_cache_*_qc`` / ``mae_*_qc`` weights) and the full multi-sensor cache
(``mae_corpus=hursat_full_multisensor_v1``: every raw frame kept, duplicate
acquisition times preserved, non-physical TBB masked to NaN at cache build;
``mae_cache_*_full`` / ``mae_*_full`` weights).  The downstream frame corpus
stays ``data/global_hursat/frames_qc`` in both cases; legacy raw/unmasked
caches are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from wpac_warning.protocol import (MAE_CORPUS_FULL_MULTISENSOR, MAE_CORPUS_QC,
                          QC_RULE_VERSION, TBB_VALID_MAX_K, TBB_VALID_MIN_K,
                          is_qc_hursat_frames, is_raw_hursat_frames,
                          load_storm_year_manifest, require_frames_corpus,
                          sha256_array, sha256_file, sha256_json,
                          validate_qc_manifest)

P2_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_PROTOCOL = "holdout_masked_v1"
TRAIN_YEARS = [2000, 2017]
VAL_YEARS = [2018, 2019]
EXCLUDED_YEARS = [2020, 2024]
TEST_YEARS = [2020, 2024]
# QC corpus (deduplicated; legacy cache manifests carry no mae_corpus field)
CACHE_DIR_NAMES = {"global": "mae_cache_global_qc", "alltc": "mae_cache_wnp_alltc"}
CHECKPOINT_DIR_NAMES = {"global": "mae_global_qc", "alltc": "mae_wnp_alltc"}
# full multi-sensor corpus (raw frames, duplicate times kept, masked at build)
FULL_CACHE_DIR_NAMES = {"global": "mae_cache_global_full",
                        "alltc": "mae_cache_wnp_alltc_full"}
FULL_CHECKPOINT_DIR_NAMES = {"global": "mae_global_full",
                             "alltc": "mae_wnp_alltc_full"}
MAE_CORPUS_FULL = MAE_CORPUS_FULL_MULTISENSOR


def compute_mae_manifest_fingerprint(manifest: dict) -> str:
    """Canonical fingerprint shared by pretraining and downstream verification."""
    return sha256_json({key: value for key, value in manifest.items()
                        if key not in ("best", "fingerprint")})


def _resolved(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (P2_ROOT / p).resolve()


def _cfg_get(cfg, key: str, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    value = cfg
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _population(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {"wnp": "alltc", "wnp_alltc": "alltc", "all_tc": "alltc"}
    return aliases.get(text, text)


def _years(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"MAE manifest {name} must be [first, last]")
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MAE manifest {name} contains invalid years") from exc


def _reference_path(entry: Any, what: str, fallback_dir: Path | None = None) -> Path:
    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
        raise ValueError(f"{what} must declare path and SHA-256")
    declared = Path(str(entry["path"]))
    candidates = [_resolved(declared)]
    if fallback_dir is not None:
        candidates.append(fallback_dir / declared.name)
    for candidate in candidates:
        if candidate.is_file():
            got = sha256_file(candidate)
            if got != str(entry["sha256"]).lower():
                raise ValueError(f"{what} SHA-256 mismatch: expected {entry['sha256']}, got {got}")
            return candidate.resolve()
    raise FileNotFoundError(f"{what} not found: {declared}")


def _require_versioned_dir(path: Path, population: str, names: dict,
                           what: str, corpus: str = MAE_CORPUS_QC) -> None:
    expected = names[population]
    if path.name == expected:
        return
    if corpus == MAE_CORPUS_FULL:
        raise ValueError(
            f"{what}={path} is not the full-multisensor {population} artifact "
            f"directory {expected}; unversioned or raw/unmasked caches and "
            f"weights are forbidden")
    raise ValueError(
        f"{what}={path} is not the QC {population} artifact directory "
        f"{expected}; raw/pre-QC caches and weights are forbidden")


def cache_dir_names(corpus: str) -> dict:
    return (FULL_CACHE_DIR_NAMES if corpus == MAE_CORPUS_FULL
            else CACHE_DIR_NAMES)


def checkpoint_dir_names(corpus: str) -> dict:
    return (FULL_CHECKPOINT_DIR_NAMES if corpus == MAE_CORPUS_FULL
            else CHECKPOINT_DIR_NAMES)


def mae_corpus_of_cache_manifest(manifest: dict) -> str:
    """Corpus identity of a merged cache (legacy QC manifests lack the field)."""
    corpus = manifest.get("mae_corpus")
    if corpus in (None, ""):
        return MAE_CORPUS_QC
    if str(corpus) != MAE_CORPUS_FULL:
        raise ValueError(f"unsupported MAE corpus {corpus!r}")
    return MAE_CORPUS_FULL


def verify_qc_assets(qc_manifest: str | Path, frames_dir: str | Path,
                     frames_corpus: str) -> dict:
    """Validate the current config's QC manifest and frame-corpus identity."""
    require_frames_corpus({"frames_corpus": frames_corpus})
    qc_path = _resolved(qc_manifest)
    frames_path = _resolved(frames_dir)
    if not qc_path.is_file():
        raise FileNotFoundError(f"QC manifest not found: {qc_path}")
    block = json.loads(qc_path.read_text(encoding="utf-8"))
    audit = validate_qc_manifest(block, expected_out_frames=frames_path)
    return {
        "path": str(qc_path),
        "sha256": sha256_file(qc_path),
        "qc_rule_version": audit["qc_rule_version"],
        "stores_sha256": audit["stores_sha256"],
        "out_frames_dir": audit["out_frames_dir"],
        "frames_dir": str(frames_path),
        "n_stores": int(audit["n_stores"]),
    }


def verify_qc_for_config(cfg) -> dict:
    return verify_qc_assets(
        _cfg_get(cfg, "data.qc_manifest"),
        _cfg_get(cfg, "data.frames_dir"),
        _cfg_get(cfg, "data.frames_corpus"),
    )


def _check_qc_reference(entry: Any, qc: dict, what: str) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{what} is missing")
    if entry.get("qc_rule_version") not in (None, QC_RULE_VERSION):
        raise ValueError(f"{what} QC rule/version mismatch")
    if entry.get("sha256") != qc["sha256"]:
        raise ValueError(f"{what} QC manifest SHA-256 differs from current config")
    if entry.get("stores_sha256") != qc["stores_sha256"]:
        raise ValueError(f"{what} QC stores hash differs from current config")
    if not entry.get("path"):
        raise ValueError(f"{what} lacks its QC manifest path")


def verify_cache_manifest(cache_dir: str | Path, qc: dict, population: str) -> dict:
    """Verify a merged MAE cache and its direct frames/index/store bindings.

    Two legal corpora: the QC v1 deduplicated cache (downstream-compatible,
    frames_source_dir == frames_qc) and the full multi-sensor cache
    (``mae_corpus=hursat_full_multisensor_v1``: raw frames, duplicate times
    kept, non-physical TBB masked to NaN at build, ``deduplicate=false``).
    A raw/unmasked cache that does not declare the full-multisensor corpus is
    rejected, as is a full cache whose mask range or dedup flag is wrong.
    """
    population = _population(population)
    if population not in CACHE_DIR_NAMES:
        raise ValueError(f"unsupported MAE population {population!r}")
    cache_path = _resolved(cache_dir)
    frames_path = cache_path / "frames.npy"
    index_path = cache_path / "index.parquet"
    manifest_path = cache_path / "cache_manifest.json"
    for path, what in ((frames_path, "MAE cache frames"),
                       (index_path, "MAE cache index"),
                       (manifest_path, "MAE cache manifest")):
        if not path.is_file():
            raise FileNotFoundError(f"{what} not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "mae_frame_cache":
        raise ValueError("MAE cache manifest is not a mae_frame_cache")
    corpus = mae_corpus_of_cache_manifest(manifest)
    _require_versioned_dir(cache_path, population, cache_dir_names(corpus),
                           "MAE cache_dir", corpus)
    source_dir = str(manifest.get("frames_source_dir", ""))
    if corpus == MAE_CORPUS_FULL:
        if manifest.get("source_corpus") != "raw_full":
            raise ValueError(
                "full-multisensor MAE cache must record source_corpus=raw_full")
        if manifest.get("deduplicate") is not False:
            raise ValueError(
                "full-multisensor MAE cache must record deduplicate=false "
                "(multi-sensor duplicate acquisition times are kept)")
        rng = [float(x) for x in manifest.get("tbb_mask_range_k", [])]
        if rng != [TBB_VALID_MIN_K, TBB_VALID_MAX_K]:
            raise ValueError(
                f"full-multisensor MAE cache tbb_mask_range_k={rng} != "
                f"{[TBB_VALID_MIN_K, TBB_VALID_MAX_K]}; an unmasked raw cache "
                f"is forbidden")
        if not is_raw_hursat_frames(source_dir):
            raise ValueError(
                "full-multisensor MAE cache frames_source_dir is not the raw "
                "download corpus (data/global_hursat/frames)")
        if manifest.get("qc_rule_version") is not None or \
                manifest.get("qc_manifest") is not None:
            raise ValueError(
                "full-multisensor MAE cache must not claim a QC binding "
                "(it is a raw-source, mask-at-build corpus)")
        dup = manifest.get("duplicate_times")
        if not isinstance(dup, dict) or int(dup.get("n_duplicate_rows", -1)) < 0:
            raise ValueError(
                "full-multisensor MAE cache lacks its duplicate-time statistics")
        if not isinstance(manifest.get("n_pixels_masked"), int) or \
                manifest["n_pixels_masked"] < 0:
            raise ValueError(
                "full-multisensor MAE cache lacks its masked-pixel count")
    else:
        if manifest.get("qc_rule_version") != QC_RULE_VERSION:
            raise ValueError(
                "MAE cache manifest QC rule/version is missing or wrong")
        _check_qc_reference(manifest.get("qc_manifest"), qc,
                            "MAE cache qc_manifest")
        if not is_qc_hursat_frames(source_dir):
            raise ValueError(
                "MAE cache frames_source_dir is raw/pre-QC, not frames_qc")
        if _resolved(source_dir) != _resolved(qc["out_frames_dir"]):
            raise ValueError(
                "MAE cache source frames differ from the configured QC corpus")
    frames_sha = sha256_file(frames_path)
    if manifest.get("frames_sha256") != frames_sha:
        raise ValueError("MAE cache fingerprint changed: frames.npy SHA-256 mismatch")
    index_sha = sha256_file(index_path)
    if manifest.get("index_sha256") != index_sha:
        raise ValueError("MAE cache fingerprint changed: index.parquet SHA-256 mismatch")
    if not manifest.get("sids_sha256"):
        raise ValueError("MAE cache manifest lacks sids_sha256")
    sids_from = manifest.get("sids_from")
    if population == "global" and sids_from not in (None, ""):
        raise ValueError("global MAE cache was restricted by sids_from; rebuild from full QC global")
    if population == "alltc" and not sids_from:
        raise ValueError("WNP alltc MAE cache lacks its canonical sids_from provenance")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "cache_dir": str(cache_path),
        "mae_corpus": corpus,
        "frames_source_dir": source_dir,
        "frames_npy_sha256": frames_sha,
        "index_parquet_sha256": index_sha,
        "sids_sha256": manifest["sids_sha256"],
        "sids_from": sids_from,
    }


def _filtered_index_path(entry: dict, checkpoint: Path, cache_dir: Path) -> Path:
    return _reference_path(entry, "MAE filtered index",
                           fallback_dir=cache_dir if cache_dir else checkpoint.parent)


def _is_sha256_text(value) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def verify_full_source_hashes(records: list, source_dir: str | Path) -> dict:
    """Re-hash every per-SID raw source store against the audit records."""
    from wpac_warning.frame_store import frame_array_path, require_frame_time_path
    src = Path(source_dir)
    if not is_raw_hursat_frames(src):
        raise ValueError(
            f"full-multisensor MAE source dir {src} is not the raw download "
            f"corpus (data/global_hursat/frames)")
    for record in records:
        sid = str(record.get("sid", record.get("SID", "")))
        array_sha = record.get("source_array_sha256")
        times_sha = record.get("source_times_sha256")
        if not _is_sha256_text(array_sha) or not _is_sha256_text(times_sha):
            raise ValueError(
                f"full-multisensor MAE SID/year audit record {sid} lacks its "
                f"per-SID source hashes")
        array = frame_array_path(src, sid)
        times_path = require_frame_time_path(src, sid)
        if not array.is_file():
            raise ValueError(
                f"full-multisensor MAE raw source store missing: {array}")
        if sha256_file(array) != str(array_sha).lower():
            raise ValueError(
                f"full-multisensor MAE raw source array {array} differs from "
                f"the cache SID audit; the raw corpus changed since the cache "
                f"was built")
        if sha256_file(times_path) != str(times_sha).lower():
            raise ValueError(
                f"full-multisensor MAE raw source time axis {times_path} "
                f"differs from the cache SID audit")
    return {"n_sources_verified": len(records)}


def _verify_cache_sid_years(manifest: dict, cache: dict) -> dict:
    entry = manifest.get("cache_sid_years")
    path = _reference_path(entry, "MAE cache SID/year audit",
                           fallback_dir=Path(cache["cache_dir"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("MAE cache SID/year audit has no records")
    if payload.get("content_sha256") != sha256_json(records):
        raise ValueError("MAE cache SID/year content hash mismatch")
    sids = sorted(str(record.get("sid", record.get("SID", ""))) for record in records)
    if not all(sids) or len(set(sids)) != len(sids):
        raise ValueError("MAE cache SID/year audit has missing or duplicate SIDs")
    if sha256_array(sids) != cache["sids_sha256"]:
        raise ValueError("MAE cache SID/year universe differs from cache manifest")
    out = {"path": str(path), "sha256": sha256_file(path), "sids": sids}
    if cache.get("mae_corpus") == MAE_CORPUS_FULL:
        out["source_hashes"] = verify_full_source_hashes(
            records, cache["frames_source_dir"])
    return out


def _verify_wnp_coverage(manifest: dict, expected_path_value, actual_index: pd.DataFrame,
                         cache_sids: list[str]) -> dict:
    entry = manifest.get("expected_sid_universe")
    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
        raise ValueError("MAE expected SID universe must declare path and SHA-256")
    if not expected_path_value:
        raise ValueError("WNP alltc requires data.storm_year_manifest for coverage")
    current_path = _resolved(expected_path_value)
    if not current_path.is_file():
        raise FileNotFoundError(f"canonical SID universe not found: {current_path}")
    if sha256_file(current_path) != entry["sha256"]:
        raise ValueError("WNP expected SID universe differs from current canonical manifest")
    expected = load_storm_year_manifest(current_path)
    if set(cache_sids) - set(expected.index.astype(str)):
        raise ValueError("WNP alltc cache contains SIDs outside the canonical universe")
    coverage = manifest.get("expected_sid_coverage")
    if not isinstance(coverage, dict) or coverage.get("coverage_verified") is not True:
        raise ValueError("WNP alltc lacks verified expected SID coverage")
    if (coverage.get("universe") or {}).get("sha256") != entry["sha256"]:
        raise ValueError("WNP coverage universe hash differs from expected SID universe")
    actual = actual_index.copy()
    actual["SID"] = actual["SID"].astype(str)
    result = {}
    for split, years in (("train", TRAIN_YEARS), ("val", VAL_YEARS)):
        expected_sids = set(expected.loc[expected.between(years[0], years[1])].index)
        actual_sids = set(actual.loc[actual["split"] == split, "SID"])
        if actual_sids != expected_sids:
            missing = sorted(expected_sids - actual_sids)
            unexpected = sorted(actual_sids - expected_sids)
            raise ValueError(
                f"WNP canonical {split} SID coverage failed: missing={missing[:20]} "
                f"unexpected={unexpected[:20]}")
        result[split] = {"n_sids": len(actual_sids),
                         "sids_sha256": sha256_array(sorted(actual_sids))}
    return result


def _verify_tbb_norm(manifest: dict, tbb_norm, prior_manifest) -> dict:
    if not tbb_norm or not prior_manifest:
        raise ValueError("current data.tbb_norm and prior_manifest are required")
    current = _resolved(tbb_norm)
    entry = manifest.get("tbb_norm")
    manifest_path = _reference_path(entry, "MAE manifest tbb_norm",
                                    fallback_dir=current.parent)
    if manifest_path != current:
        raise ValueError("MAE tbb_norm path differs from current config")
    prior_path = _resolved(prior_manifest)
    if not prior_path.is_file():
        raise FileNotFoundError(f"prior manifest not found: {prior_path}")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    asset = (prior.get("assets") or {}).get("tbb_norm")
    if not isinstance(asset, dict):
        raise ValueError("prior manifest lacks assets.tbb_norm")
    if [int(y) for y in asset.get("fit_years", [])] != TRAIN_YEARS:
        raise ValueError("prior tbb_norm is not a fit-only 2000-2017 asset")
    asset_path = _reference_path(asset, "prior fit-only tbb_norm",
                                 fallback_dir=prior_path.parent)
    if asset_path != current:
        raise ValueError("current tbb_norm is not the same file as the prior asset")
    if asset["sha256"] != entry["sha256"]:
        raise ValueError("MAE tbb_norm SHA differs from the prior fit-only asset")
    return {"path": str(current), "sha256": sha256_file(current)}


def _verify_internal_checkpoints(ckpt: Path, manifest: dict) -> dict:
    if manifest.get("fingerprint") != compute_mae_manifest_fingerprint(manifest):
        raise ValueError("MAE external manifest fingerprint does not match its contents")
    best_meta = manifest["best"]
    best_path = ckpt.parent / "mae_best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"MAE best checkpoint missing: {best_path}")
    if best_meta.get("checkpoint") != best_path.name:
        raise ValueError("MAE manifest best.checkpoint must be mae_best.pt")
    best_sha = sha256_file(best_path)
    if best_meta.get("checkpoint_sha256") != best_sha:
        raise ValueError("MAE mae_best.pt SHA-256 differs from manifest best")
    downstream = torch.load(ckpt, map_location="cpu", weights_only=False)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    for label, state in (("downstream encoder", downstream), ("mae_best", best)):
        if state.get("protocol") != HOLDOUT_PROTOCOL:
            raise ValueError(f"{label} internal protocol is not holdout_masked_v1")
        if state.get("manifest_fingerprint") != manifest["fingerprint"]:
            raise ValueError(f"{label} internal manifest_fingerprint mismatch")
        for key, expected in (("train_frame_years", TRAIN_YEARS),
                              ("val_years", VAL_YEARS)):
            value = state.get(key)
            if not isinstance(value, (list, tuple)) or \
                    [int(y) for y in value] != expected:
                raise ValueError(f"{label} internal {key} mismatch")
        if not isinstance(state.get("model"), dict):
            raise ValueError(f"{label} lacks its model state dict")
    if best.get("kind") != "mae_best":
        raise ValueError("mae_best.pt has the wrong internal kind")
    if downstream.get("epoch") != best.get("epoch") or \
            int(best.get("epoch", -1)) != int(best_meta.get("epoch", -2)):
        raise ValueError("downstream/mae_best/manifest epochs disagree")
    left, right = downstream["model"], best["model"]
    if set(left) != set(right) or any(not torch.equal(left[key], right[key])
                                      for key in left):
        raise ValueError("downstream encoder weights differ from mae_best.pt")
    from wpac_warning.mae import ViTMAEEncoder
    ViTMAEEncoder(out_dim=256, checkpoint=str(ckpt), freeze=False)
    return {"best_checkpoint": str(best_path), "best_checkpoint_sha256": best_sha}


def verify_mae_provenance(checkpoint: str | Path, *, protocol: str,
                          downstream_population: str = "alltc",
                          expected_mae_population: str | None = None,
                          qc_manifest: str | Path | None = None,
                          frames_dir: str | Path | None = None,
                          frames_corpus: str | None = None,
                          expected_sid_universe: str | Path | None = None,
                          global_cache_dir: str | Path | None = None,
                          alltc_cache_dir: str | Path | None = None,
                          tbb_norm: str | Path | None = None,
                          prior_manifest: str | Path | None = None) -> dict:
    """Verify QC, cache, index, population and encoder before MAE loading."""
    ckpt = _resolved(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"MAE checkpoint not found: {ckpt}")
    if protocol != HOLDOUT_PROTOCOL:
        raise ValueError(f"MAE provenance requires {HOLDOUT_PROTOCOL}, got {protocol!r}")
    if _population(downstream_population) != "alltc":
        raise ValueError(f"downstream population {downstream_population!r} is not alltc")
    if not qc_manifest or not frames_dir or not frames_corpus:
        raise ValueError("current data.qc_manifest/frames_dir/frames_corpus are required")
    qc = verify_qc_assets(qc_manifest, frames_dir, frames_corpus)

    manifest_path = ckpt.parent / "mae_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MAE manifest missing beside checkpoint: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != protocol:
        raise ValueError(f"MAE manifest protocol {manifest.get('protocol')!r} != {protocol!r}")
    population = _population(manifest.get("population"))
    if population not in CHECKPOINT_DIR_NAMES:
        raise ValueError("MAE manifest requires explicit population=global or alltc")
    if expected_mae_population is not None and population != _population(expected_mae_population):
        raise ValueError(f"MAE population {population!r} != configured {expected_mae_population!r}")
    if manifest.get("qc_rule_version") != QC_RULE_VERSION:
        raise ValueError("MAE manifest QC rule/version is missing or wrong")
    _check_qc_reference(manifest.get("qc_manifest"), qc, "MAE qc_manifest")

    configured_cache = (global_cache_dir if population == "global" else alltc_cache_dir)
    cache_dir_value = configured_cache or manifest.get("cache_dir")
    if not cache_dir_value:
        raise ValueError("MAE manifest/config lacks the population-specific cache_dir")
    cache = verify_cache_manifest(cache_dir_value, qc, population)
    corpus = cache["mae_corpus"]
    declared_corpus = manifest.get("mae_corpus")
    if corpus == MAE_CORPUS_FULL:
        if declared_corpus != MAE_CORPUS_FULL:
            raise ValueError(
                "MAE manifest does not declare "
                f"mae_corpus={MAE_CORPUS_FULL!r} for a full-multisensor cache")
    elif declared_corpus not in (None, "", MAE_CORPUS_QC):
        raise ValueError(
            f"MAE manifest declares mae_corpus={declared_corpus!r} but the "
            f"cache is the QC corpus; a QC cache cannot back a full-multisensor "
            f"run")
    _require_versioned_dir(ckpt.parent, population,
                           checkpoint_dir_names(corpus),
                           "MAE checkpoint directory", corpus)
    cache_ref = _reference_path(
        manifest.get("cache_manifest"), "MAE cache manifest reference",
        fallback_dir=Path(cache["cache_dir"]))
    if cache_ref != Path(cache["path"]):
        raise ValueError("MAE manifest points at a different cache manifest")
    if manifest["cache_manifest"].get("sha256") != cache["sha256"]:
        raise ValueError("MAE cache manifest hash differs from MAE manifest")
    cache_fingerprint = manifest.get("cache")
    if not isinstance(cache_fingerprint, dict):
        raise ValueError("MAE manifest lacks cache frames/index hashes")
    if cache_fingerprint.get("frames_npy_sha256") != cache["frames_npy_sha256"]:
        raise ValueError("MAE manifest cache frames hash mismatch")
    if cache_fingerprint.get("index_parquet_sha256") != cache["index_parquet_sha256"]:
        raise ValueError("MAE manifest cache index hash mismatch")
    sid_audit = _verify_cache_sid_years(manifest, cache)

    if population == "global":
        if manifest.get("expected_sid_universe") is not None or \
                manifest.get("expected_sid_coverage") is not None:
            raise ValueError("global MAE must not be restricted to a WNP expected SID universe")
    else:
        coverage_universe = expected_sid_universe

    for key, expected in (("train_frame_years", TRAIN_YEARS),
                          ("val_years", VAL_YEARS),
                          ("excluded_years", EXCLUDED_YEARS)):
        got = _years(manifest.get(key), key) if key != "val_years" else manifest.get(key)
        if key == "val_years":
            if not isinstance(got, (list, tuple)):
                raise ValueError("MAE manifest val_years must be a list")
            got = [int(y) for y in got]
        if list(got) != expected:
            raise ValueError(f"MAE manifest {key}={got}, required {expected}")

    index_path = _filtered_index_path(manifest.get("filtered_index"), ckpt,
                                      Path(cache["cache_dir"]))
    index = pd.read_parquet(index_path)
    required = {"SID", "time", "frame_row", "split"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"MAE filtered index missing columns {missing}")
    if index.empty:
        raise ValueError("MAE filtered index is empty")
    if index["frame_row"].isna().any():
        raise ValueError("MAE filtered index has null frame_row values")
    frame_rows = index["frame_row"].to_numpy(dtype="int64")
    if (frame_rows < 0).any() or len(set(frame_rows.tolist())) != len(frame_rows):
        raise ValueError("MAE filtered index frame_row must be native, unique and nonnegative")
    times = pd.to_datetime(index["time"], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError("MAE filtered index has invalid frame times")
    frame_year = times.dt.year.astype(int)
    if "year" in index.columns:
        recorded_year = pd.to_numeric(index["year"], errors="coerce")
        if recorded_year.isna().any() or not (recorded_year.astype(int) == frame_year).all():
            raise ValueError("MAE filtered index year disagrees with ISO frame time")
    test_mask = frame_year.between(TEST_YEARS[0], TEST_YEARS[1])
    if bool(test_mask.any()):
        rows = index.loc[test_mask, ["SID", "time", "frame_row"]].head(20)
        raise ValueError(
            f"MAE filtered index contains {int(test_mask.sum())} test-year "
            f"frames ({TEST_YEARS}); row audit: {rows.to_dict(orient='records')}")
    if "storm_year" in index.columns:
        storm_year = pd.to_numeric(index["storm_year"], errors="coerce")
        if storm_year.isna().any() or bool(storm_year.between(*TEST_YEARS).any()):
            raise ValueError("MAE filtered index contains invalid or test-year storms")

    raw_index = pd.read_parquet(Path(cache["cache_dir"]) / "index.parquet")
    raw_required = {"SID", "time", "frame_row"}
    if not raw_required.issubset(raw_index.columns) or raw_index.empty:
        raise ValueError("MAE raw cache index lacks native SID/time/frame_row rows")
    raw_rows = raw_index["frame_row"].to_numpy(dtype="int64")
    if len(set(raw_rows.tolist())) != len(raw_rows):
        raise ValueError("MAE raw cache index has duplicate frame_row values")
    raw_sids = sorted(raw_index["SID"].astype(str).unique())
    if sha256_array(raw_sids) != cache["sids_sha256"]:
        raise ValueError("MAE raw cache index SID set differs from cache manifest")
    if not set(frame_rows.tolist()).issubset(set(raw_rows.tolist())):
        raise ValueError("MAE filtered index contains frame_row values absent from QC cache")
    raw_index = raw_index.assign(
        _raw_time=pd.to_datetime(raw_index["time"], errors="coerce", utc=True))
    if raw_index["_raw_time"].isna().any():
        raise ValueError("MAE raw cache index has invalid frame times")
    aligned = raw_index.set_index("frame_row").loc[frame_rows]
    if not (aligned["SID"].astype(str).to_numpy()
            == index["SID"].astype(str).to_numpy()).all() or not (
            aligned["_raw_time"].to_numpy() == times.to_numpy()).all():
        raise ValueError(
            "MAE filtered index frame_row does not point to the same QC cache SID/time")

    split = index["split"].astype(str)
    if not set(split).issubset({"train", "val"}):
        raise ValueError("MAE filtered index has unknown split labels")
    for label, years in (("train", TRAIN_YEARS), ("val", VAL_YEARS)):
        selected = split == label
        if not selected.any():
            raise ValueError(f"MAE filtered index has no {label} rows")
        if not bool(frame_year[selected].between(years[0], years[1]).all()):
            raise ValueError(f"MAE {label} rows fall outside {years}")
        if "storm_year" in index.columns and not bool(
                storm_year[selected].between(years[0], years[1]).all()):
            raise ValueError(f"MAE {label} storm years fall outside {years}")
    train_sids = set(index.loc[split == "train", "SID"].astype(str))
    val_sids = set(index.loc[split == "val", "SID"].astype(str))
    if train_sids & val_sids:
        raise ValueError("MAE filtered index has SIDs in both train and val")
    coverage = None
    if population == "alltc":
        coverage = _verify_wnp_coverage(manifest, coverage_universe, index,
                                        sid_audit["sids"])

    best = manifest.get("best")
    if not isinstance(best, dict) or not best.get("encoder_sha256"):
        raise ValueError("MAE manifest lacks best.encoder_sha256")
    encoder_name = best.get("encoder_checkpoint")
    if encoder_name and Path(str(encoder_name)).name != ckpt.name:
        raise ValueError(f"MAE checkpoint {ckpt.name} is not manifest best encoder")
    encoder_sha = sha256_file(ckpt)
    if encoder_sha != str(best["encoder_sha256"]).lower():
        raise ValueError("MAE encoder SHA-256 differs from manifest best")
    tbb_audit = _verify_tbb_norm(manifest, tbb_norm, prior_manifest)
    internal_audit = _verify_internal_checkpoints(ckpt, manifest)

    n_train = int((split == "train").sum())
    n_val = int((split == "val").sum())
    manifest_train = int(manifest.get("filtered_n_train_frames",
                                      manifest.get("n_train_frames", -1)))
    manifest_val = int(manifest.get("filtered_n_val_frames",
                                    manifest.get("n_val_frames", -1)))
    if (n_train, n_val) != (manifest_train, manifest_val):
        raise ValueError(
            f"MAE manifest frame counts {(manifest_train, manifest_val)} != "
            f"filtered index {(n_train, n_val)}")
    return {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": encoder_sha,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "tbb_norm": tbb_audit,
        "internal_checkpoints": internal_audit,
        "filtered_index": str(index_path),
        "filtered_index_sha256": manifest["filtered_index"]["sha256"],
        "cache_manifest": cache["path"],
        "cache_manifest_sha256": cache["sha256"],
        "qc_manifest": qc["path"],
        "qc_manifest_sha256": qc["sha256"],
        "qc_rule_version": QC_RULE_VERSION,
        "qc_stores_sha256": qc["stores_sha256"],
        "protocol": protocol,
        "population": population,
        "mae_corpus": corpus,
        "downstream_population": "alltc",
        "train_frame_years": TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "excluded_years": EXCLUDED_YEARS,
        "n_train_frames": n_train,
        "n_val_frames": n_val,
        "n_sids": int(index["SID"].nunique()),
        "canonical_coverage": coverage,
        "best_epoch": int(best.get("epoch", -1)),
    }


def verify_mae_for_config(cfg) -> dict | None:
    """Validate the configured MAE immediately before CausalRINet construction."""
    if str(_cfg_get(cfg, "model.vision.backend", "")) != "vit_mae":
        return None
    checkpoint = _cfg_get(cfg, "model.vision.mae_checkpoint")
    if not checkpoint:
        raise ValueError("holdout vit_mae requires model.vision.mae_checkpoint")
    return verify_mae_provenance(
        checkpoint,
        protocol=str(_cfg_get(cfg, "data.protocol", "")),
        downstream_population=str(_cfg_get(cfg, "data.population", "alltc")),
        expected_mae_population=_cfg_get(cfg, "model.vision.mae_population"),
        qc_manifest=_cfg_get(cfg, "data.qc_manifest"),
        frames_dir=_cfg_get(cfg, "data.frames_dir"),
        frames_corpus=_cfg_get(cfg, "data.frames_corpus"),
        expected_sid_universe=_cfg_get(cfg, "data.storm_year_manifest"),
        global_cache_dir=_cfg_get(cfg, "mae.global_cache_dir"),
        alltc_cache_dir=_cfg_get(cfg, "mae.cache_dir"),
        tbb_norm=_cfg_get(cfg, "data.tbb_norm"),
        prior_manifest=_cfg_get(cfg, "data.prior_manifest"),
    )
