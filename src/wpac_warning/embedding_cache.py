# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Provenance construction and fail-closed validation for frame embeddings."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wpac_warning.protocol import sha256_array, sha256_file, sha256_json

P2_ROOT = Path(__file__).resolve().parents[2]


def resolved(path, base=None) -> Path:
    p = Path(path)
    root = Path(base) if base is not None else P2_ROOT
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def list_frame_sids(frames_dir: Path) -> list[str]:
    return sorted(p.stem for p in Path(frames_dir).glob("*.npy")
                  if not p.stem.endswith(("_iso", "_times")))


def _file_meta(path) -> dict:
    p = resolved(path)
    if not p.is_file():
        raise FileNotFoundError(f"embedding provenance input missing: {p}")
    return {"path": str(p), "sha256": sha256_file(p)}


def fit_only_norm_sha256(cfg, norm_path, base=None) -> str:
    got = sha256_file(resolved(norm_path, base))
    if str(cfg.get("data.protocol", "loyo")) != "holdout_masked_v1":
        return got
    prior_path = resolved(cfg.get("data.prior_manifest"), base)
    raw = json.loads(prior_path.read_text(encoding="utf-8"))
    try:
        want = raw["assets"]["tbb_norm"]["sha256"]
    except KeyError as exc:
        raise ValueError("holdout prior manifest lacks assets.tbb_norm.sha256") from exc
    if got != want:
        raise ValueError(
            f"TBB norm is not the fit-only asset pinned by {prior_path}: "
            f"expected {want}, got {got}")
    return got


def make_provenance(backend: str, tag: str, variant: str | None,
                    pool: str | None, dim: int | None, frames_dir,
                    frames_index, weights_path, tbb_norm,
                    qc_manifest=None, mae_corpus: str | None = None) -> dict:
    """Frozen-backbone provenance; pins the QC frame corpus when configured.

    ``mae_corpus`` records the self-supervised corpus the MAE weights were
    pretrained on (e.g. ``hursat_full_multisensor_v1``); it is provenance
    only — the ENCODED frame corpus stays the configured QC corpus.
    """
    prov = {
        "backend": str(backend),
        "tag": str(tag),
        "variant": variant,
        "pool": pool,
        "dim": int(dim) if dim is not None else None,
        "frames_dir": str(resolved(frames_dir)),
        "frames_index": _file_meta(frames_index),
        "weights": _file_meta(weights_path),
        "tbb_norm": _file_meta(tbb_norm),
    }
    if mae_corpus:
        prov["mae_corpus"] = str(mae_corpus)
    if qc_manifest:
        from wpac_warning.protocol import QC_RULE_VERSION
        meta = _file_meta(qc_manifest)
        meta["qc_rule_version"] = QC_RULE_VERSION
        prov["frames_qc"] = meta
    return prov


def frame_row_key_sha256(frames_dir, sid: str) -> tuple[int, str]:
    from wpac_warning.frame_store import load_frame_times
    iso = load_frame_times(frames_dir, sid)
    keys = [f"{sid}|{np.datetime_as_string(t, unit='s')}Z" for t in iso]
    return len(keys), sha256_array(keys)


def source_content_sha256(frames_dir, sid: str) -> dict:
    """Content hashes of the QC frame store a cache row is aligned to."""
    from wpac_warning.frame_store import frame_array_path, require_frame_time_path
    array = frame_array_path(frames_dir, sid)
    times = require_frame_time_path(frames_dir, sid)
    array_sha = sha256_file(array)
    times_sha = sha256_file(times)
    return {"array_sha256": array_sha, "times_sha256": times_sha,
            "content_sha256": sha256_json({"array_sha256": array_sha,
                                           "times_sha256": times_sha})}


def store_metadata(frames_dir, embed_dir, sid: str) -> dict:
    ep = Path(embed_dir) / f"{sid}.npy"
    if not ep.is_file():
        raise FileNotFoundError(f"embedding store missing: {ep}")
    emb = np.load(ep, mmap_mode="r")
    if emb.ndim != 2:
        raise ValueError(f"embedding store is not a rank-2 array: {ep}")
    n_rows, row_hash = frame_row_key_sha256(frames_dir, sid)
    if emb.shape[0] != n_rows:
        raise ValueError(
            f"embedding/frame row mismatch for {sid}: {emb.shape[0]} vs {n_rows}")
    source = source_content_sha256(frames_dir, sid)
    return {
        "n_rows": int(n_rows),
        "dim": int(emb.shape[1]),
        "row_key_sha256": row_hash,
        "content_sha256": sha256_file(ep),
        "source": source,
    }


def build_manifest(provenance: dict, frames_dir, embed_dir,
                   adopted_existing: bool = False) -> dict:
    embed_dir = Path(embed_dir)
    present = sorted(p.stem for p in embed_dir.glob("*.npy")
                     if not p.name.startswith("."))
    stores = {sid: store_metadata(frames_dir, embed_dir, sid)
              for sid in present}
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance,
        "adopted_existing": bool(adopted_existing),
        "n_storms": len(present),
        "sids_sha256": sha256_array(present),
        "stores": stores,
    }


def write_manifest(provenance: dict, frames_dir, embed_dir,
                   adopted_existing: bool = False) -> dict:
    manifest = build_manifest(provenance, frames_dir, embed_dir,
                              adopted_existing=adopted_existing)
    path = Path(embed_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def contract_from_config(cfg, base=None) -> dict:
    embed_dir = resolved(cfg.get("data.ir_embeds_dir"), base)
    frames_dir = resolved(cfg.get("data.frames_dir"), base)
    backend = str(cfg.get("model.vision.backend", ""))
    if backend == "vit_mae":
        pool = str(cfg.get("model.vision.mae_pool", "cls"))
        variant = None
        tag = f"vit_mae_{pool}"
        weights = resolved(cfg.get("model.vision.mae_checkpoint"), base)
    elif backend == "lingbot":
        pool = None
        variant = str(cfg.get("model.vision.lingbot_variant", "small"))
        tag = f"lingbot_{variant}"
        weights = resolved(cfg.get("model.vision.lingbot_dir"), base) / "model.pt"
    else:
        raise ValueError(f"embedding cache is unsupported for backend {backend!r}")
    norm = resolved(cfg.get("data.tbb_norm"), base)
    fit_only_norm_sha256(cfg, norm, base=base)
    qc_manifest = cfg.get("data.qc_manifest")
    if str(cfg.get("data.protocol", "loyo")) == "holdout_masked_v1" and             not qc_manifest:
        raise ValueError(
            "holdout embedding caches require data.qc_manifest: the cache must "
            "pin the QC frame corpus it was encoded from")
    provenance = make_provenance(
        backend, tag, variant, pool, None, frames_dir,
        resolved(cfg.get("data.frames_index"), base), weights, norm,
        qc_manifest=resolved(qc_manifest, base) if qc_manifest else None)
    return {
        "provenance": provenance,
        "frames_dir": frames_dir,
        "embed_dir": embed_dir,
        "holdout": str(cfg.get("data.protocol", "loyo")) == "holdout_masked_v1",
    }


def validate_contract(contract: dict, required_sids=None,
                      allow_partial: bool = False) -> dict:
    embed_dir = Path(contract["embed_dir"])
    frames_dir = Path(contract["frames_dir"])
    holdout = bool(contract.get("holdout"))
    manifest_path = embed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"embedding cache provenance manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if holdout and manifest.get("adopted_existing") is not False:
        raise ValueError(
            f"holdout embedding cache must be fully encoded with current inputs; "
            f"adopted or legacy provenance is forbidden: {manifest_path}")
    actual = manifest.get("provenance")
    expected = contract["provenance"]
    if not isinstance(actual, dict):
        raise ValueError(f"embedding cache has no provenance: {manifest_path}")
    for key in ("backend", "tag", "variant", "pool"):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"embedding cache provenance {key} mismatch: "
                f"{actual.get(key)!r} != {expected.get(key)!r}")
    if not isinstance(actual.get("dim"), int) or actual["dim"] <= 0:
        raise ValueError("embedding cache provenance has an invalid dimension")
    if expected.get("dim") is not None and actual["dim"] != expected["dim"]:
        raise ValueError("embedding cache dimension differs from the encoder")
    if Path(str(actual.get("frames_dir"))).as_posix() != Path(
            str(expected.get("frames_dir"))).as_posix():
        raise ValueError(
            f"embedding cache frames_dir mismatch: {actual.get('frames_dir')} "
            f"!= {expected.get('frames_dir')}; the cache was encoded from "
            f"another frame corpus")
    for key in ("weights", "tbb_norm", "frames_index"):
        got = actual.get(key, {}).get("sha256")
        want = expected[key]["sha256"]
        if not got or got != want:
            raise ValueError(
                f"embedding cache {key} SHA-256 mismatch: {got!r} != {want!r}")
    if expected.get("frames_qc") or holdout:
        want = (expected.get("frames_qc") or {}).get("sha256")
        if not want:
            raise ValueError(
                "holdout embedding cache has no frames_qc provenance binding "
                "(data.qc_manifest missing?)")
        got = (actual.get("frames_qc") or {}).get("sha256")
        if not got or got != want:
            raise ValueError(
                f"embedding cache frames QC manifest SHA-256 mismatch: "
                f"{got!r} != {want!r}; the cache was encoded against another "
                f"frame corpus")

    stores = manifest.get("stores")
    if not isinstance(stores, dict):
        raise ValueError("embedding cache lacks per-SID row/content fingerprints")
    present = sorted(p.stem for p in embed_dir.glob("*.npy")
                     if not p.name.startswith("."))
    if set(present) != set(stores):
        raise ValueError("embedding cache files and per-SID manifest entries differ")
    if required_sids is None:
        required_sids = list_frame_sids(frames_dir) if holdout else []
    missing = sorted(set(required_sids) - set(present))
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"{len(missing)} required storms lack cached embeddings "
            f"(first 10: {missing[:10]})")
    if not present and not allow_partial:
        raise FileNotFoundError(f"embedding cache is empty: {embed_dir}")
    for sid in present:
        got = store_metadata(frames_dir, embed_dir, sid)
        if stores[sid] != got:
            raise ValueError(
                f"embedding row-key/content fingerprint mismatch for {sid}")
        if got["dim"] != actual["dim"]:
            raise ValueError(f"embedding dimension mismatch for {sid}")
    if int(manifest.get("n_storms", -1)) != len(present):
        raise ValueError("embedding cache n_storms does not match its files")
    if manifest.get("sids_sha256") != sha256_array(present):
        raise ValueError("embedding cache SID-set fingerprint mismatch")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "n_storms": len(present),
        "sids_sha256": sha256_array(present),
        "stores_sha256": sha256_json(stores),
        "weights_sha256": actual["weights"]["sha256"],
        "tbb_norm_sha256": actual["tbb_norm"]["sha256"],
        "frames_index_sha256": actual["frames_index"]["sha256"],
        "frames_qc_sha256": (actual.get("frames_qc") or {}).get("sha256"),
    }


def validate_embedding_cache(cfg, required_sids=None,
                             allow_partial: bool = False) -> dict:
    return validate_contract(contract_from_config(cfg), required_sids,
                             allow_partial=allow_partial)
