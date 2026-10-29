# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Training initialization and completion artifacts for holdout runs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from wpac_warning.protocol import sha256_file, sha256_json

INITIAL_KIND = "holdout_masked_v1_initial"
INITIAL_CHECKPOINT = Path("checkpoints") / "initial.pt"
INITIAL_PROVENANCE = "initialization.json"
COMPLETED_MARKER = "completed.json"
REQUIRED_RUN_ARTIFACTS = (
    "config.yaml",
    "run_manifest.json",
    "data_manifest.json",
    "weight_manifest.json",
    INITIAL_PROVENANCE,
    "results.csv",
    INITIAL_CHECKPOINT.as_posix(),
    (Path("checkpoints") / "best_ri.pt").as_posix(),
    (Path("preds") / "inner.npz").as_posix(),
)


def atomic_write_json(path: str | Path, payload: dict, *, overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_save_torch(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def save_initial_checkpoint(path: str | Path, model, *, run_id: str, seed: int,
                            config_sha256: str, source: dict | None = None) -> dict:
    path = Path(path)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    meta = {
        "run_id": str(run_id),
        "seed": int(seed),
        "config_sha256": str(config_sha256),
        "gnn_enabled": bool(getattr(model, "gnn_enabled", False)),
        "source": source or {"kind": "fresh_seeded_model"},
    }
    atomic_save_torch(path, {"kind": INITIAL_KIND, "model": state, "meta": meta})
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "meta": meta}


def load_initial_checkpoint(model, path: str | Path, *,
                            allow_structural: bool = False) -> dict:
    """Load every compatible source tensor and explicitly audit all others.

    Structural mode is for C2. It still performs a strict full-state load after
    merging the compatible source tensors into the target's own state; no
    partial ``strict=False`` load can silently omit a shared tensor.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"shared initialization not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != INITIAL_KIND or not isinstance(payload.get("model"), dict):
        raise ValueError(f"{path} is not a holdout initial checkpoint")
    source = payload["model"]
    target = model.state_dict()
    source_keys, target_keys = set(source), set(target)
    shape_mismatches = sorted(
        name for name in source_keys & target_keys
        if tuple(source[name].shape) != tuple(target[name].shape))
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    if not allow_structural and (missing or unexpected or shape_mismatches):
        raise RuntimeError(
            f"initial/model structure mismatch: missing={missing[:10]} "
            f"unexpected={unexpected[:10]} shape_mismatches={shape_mismatches[:10]}")
    if allow_structural:
        meta = payload.get("meta", {})
        if not meta.get("gnn_enabled") or getattr(model, "gnn_enabled", False):
            raise RuntimeError("structural initial loading is restricted to C2 (GNN on -> off)")
        bad_missing = [n for n in missing if not n.startswith(("gnn.", "gnn_proj."))]
        bad_unexpected = [n for n in unexpected
                          if not n.startswith(("gnn.", "gnn_proj."))]
        bad_shapes = [n for n in shape_mismatches
                      if not n.startswith("fusion.gate.")]
        if bad_missing or bad_unexpected or bad_shapes:
            raise RuntimeError(
                f"C2 initial differs outside the registered GNN/fusion structure: "
                f"missing={bad_missing[:10]} unexpected={bad_unexpected[:10]} "
                f"shape_mismatches={bad_shapes[:10]}")

    loaded_names = sorted((source_keys & target_keys) - set(shape_mismatches))
    if not loaded_names:
        raise RuntimeError("initial checkpoint has no name+shape-compatible tensors")
    merged = dict(target)
    for name in loaded_names:
        merged[name] = source[name]
    model.load_state_dict(merged, strict=True)
    mismatched_values = [
        name for name in loaded_names
        if not torch.equal(model.state_dict()[name].detach().cpu(),
                           source[name].detach().cpu())
    ]
    if mismatched_values:
        raise RuntimeError(
            f"shared initial tensors differ after loading: {mismatched_values[:10]}")
    return {
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "source_meta": payload.get("meta", {}),
        "structural_load": bool(allow_structural),
        "loaded_tensors": loaded_names,
        "loaded_tensor_list_sha256": sha256_json(loaded_names),
        "n_loaded_tensors": len(loaded_names),
        "missing_in_source": missing,
        "unexpected_in_source": unexpected,
        "shape_mismatches": shape_mismatches,
        "shared_values_verified": True,
        "strict_full_state_load": True,
    }


def write_completed_marker(run_dir: str | Path, run_id: str) -> dict:
    run_dir = Path(run_dir)
    artifacts = {}
    for relative in REQUIRED_RUN_ARTIFACTS:
        path = run_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"cannot complete {run_id}: required artifact missing or empty: {path}")
        artifacts[relative] = sha256_file(path)
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("inner_only") is not True:
        raise ValueError(f"cannot complete {run_id}: run_manifest.inner_only is not true")
    marker = {
        "kind": "holdout_training_completed",
        "protocol": "holdout_masked_v1",
        "status": "completed",
        "run_id": str(run_id),
        "inner_only": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    marker["artifact_catalog_sha256"] = sha256_json(artifacts)
    atomic_write_json(run_dir / COMPLETED_MARKER, marker, overwrite=True)
    return marker


def load_completed_marker(run_dir: str | Path, *, verify_hashes: bool = True) -> dict:
    run_dir = Path(run_dir)
    path = run_dir / COMPLETED_MARKER
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir}: missing {COMPLETED_MARKER}; run is not completed")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if (marker.get("kind") != "holdout_training_completed"
            or marker.get("status") != "completed"):
        raise ValueError(f"{run_dir}: training completed marker is not completed")
    if marker.get("run_id") != run_dir.name:
        raise ValueError(f"{run_dir}: completed marker run_id mismatch")
    if marker.get("inner_only") is not True:
        raise ValueError(f"{run_dir}: completed marker is not inner-only")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_RUN_ARTIFACTS):
        raise ValueError(f"{run_dir}: completed marker artifact catalog is incomplete")
    if marker.get("artifact_catalog_sha256") != sha256_json(artifacts):
        raise ValueError(f"{run_dir}: completed marker artifact catalog hash mismatch")
    for relative, expected in artifacts.items():
        artifact = run_dir / relative
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise FileNotFoundError(f"{run_dir}: completed artifact missing: {relative}")
        if verify_hashes and sha256_file(artifact) != expected:
            raise ValueError(f"{run_dir}: completed artifact hash differs: {relative}")
    return marker
