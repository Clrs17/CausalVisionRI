# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""HOLDOUT_MASKED_V1 test inference (runs ONCE, after the lock is frozen).

Refuses to run unless the final lock and every artifact hash are intact.
Rebuilds every member from the lock's main/ablation root, predicts ``test``,
and atomically writes keyed npz + manifest outputs under the locked
``test_output_dir``.  A fixed receipt prevents a second run through another
output path; only explicit ``--resume`` can recover an interrupted receipt.

No training happens here and no selection metric is read.

Usage (from the release root):
    python scripts/05_eval/test_infer.py --lock protocol_locks/final.json \
        --out-dir data/test/holdout_masked_v1  # must equal the final lock
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.baseline_common import predict_baseline        # noqa: E402
from wpac_warning.checkpoint import load_inference_checkpoint   # noqa: E402
from wpac_warning.dataset import RIPanelDataset                 # noqa: E402
from wpac_warning.graph import load_prior_graph, shuffle_prior_graph  # noqa: E402
from wpac_warning.models import CausalRINet                     # noqa: E402
from wpac_warning.protocol import (SplitProtocol, canonical_data_assets,  # noqa: E402
                          check_test_receipt_available, load_prior_bundle,
                          protocol_test_receipt, resolve_protocol_test_output_dir,
                          sha256_array, sha256_file, sha256_json)
from wpac_warning.train import (predict_probs_gates, verify_canonical_sample_manifest)  # noqa: E402
from wpac_warning.utils_cfg import Config                       # noqa: E402

LOCK_KIND = "holdout_masked_v1"


def _resolved(path_like) -> Path:
    p = Path(path_like)
    return p.resolve() if p.is_absolute() else (P2_ROOT / p).resolve()


def _atomic_write_bytes(path: Path, payload: bytes, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite protocol output: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite protocol output: {path}")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_write_json(path: Path, payload: dict, overwrite: bool = False) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(Path(path), data, overwrite=overwrite)


def _atomic_savez(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite protocol output: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite protocol output: {path}")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _data_assets(cfg: dict) -> dict:
    return canonical_data_assets(cfg, P2_ROOT)


def _valid_probabilities(p, what: str) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    if not len(p) or not np.isfinite(p).all():
        raise ValueError(f"{what}: probabilities are empty or contain NaN/Inf")
    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"{what}: probabilities outside [0,1]")
    return p


def verify_lock(lock: dict, allow_fixture: bool = False) -> None:
    if lock.get("protocol") != LOCK_KIND:
        raise ValueError(f"lock protocol {lock.get('protocol')!r} != {LOCK_KIND}")
    if lock.get("kind") != "final_protocol_lock" or not lock.get("finalized"):
        raise ValueError("an intermediate selection cannot unlock test inference")
    if lock.get("fixture") and not allow_fixture:
        raise ValueError("fixture lock requires explicit --allow-fixture")
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    got = sha256_json(payload)
    if got != lock.get("lock_sha256"):
        raise ValueError("protocol lock hash mismatch — the lock was modified "
                         "after freezing")
    if not lock.get("entities"):
        raise ValueError("final protocol lock has no inference entities")
    names = ([e.get("name") for e in lock["entities"]]
             + [b.get("name") for b in lock.get("baselines", [])])
    if None in names or len(names) != len(set(names)):
        raise ValueError(f"duplicate or missing entity names in the lock: "
                         f"{names}")
    if not lock.get("test_output_dir") or not lock.get("test_receipt"):
        raise ValueError("final lock lacks test_output_dir/test_receipt")
    if not allow_fixture:
        canonical = resolve_protocol_test_output_dir()
        check_test_receipt_available(protocol_test_receipt(canonical),
                                     block_running=False)
    test_output_dir = resolve_protocol_test_output_dir(
        lock["test_output_dir"], allow_fixture=allow_fixture)
    if _resolved(lock["test_receipt"]) != protocol_test_receipt(test_output_dir):
        raise ValueError("test receipt must be <test_output_dir>/test_receipt.json")
    if any(e.get("kind") == "ablation" for e in lock["entities"]) and not \
            lock.get("ablations_root"):
        raise ValueError("ablation entities require a locked ablations_root")
    if lock.get("baselines") and not lock.get("baseline_manifest"):
        raise ValueError("baselines require a locked baseline manifest")


def build_member(entry: dict, test_years: tuple, run_dir: Path | None = None):
    run_dir = _resolved(run_dir or entry["run_dir"])
    cfg_path = run_dir / "config.yaml"
    if not entry.get("config_sha256") or \
            sha256_file(cfg_path) != entry["config_sha256"]:
        raise ValueError(f"{entry['run_id']}: config hash differs from lock")
    cfg_dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config_json_sha256 = sha256_json(cfg_dict)
    if entry.get("config_json_sha256") != config_json_sha256:
        raise ValueError(f"{entry['run_id']}: semantic config hash differs from lock")
    if entry.get("data_assets") != _data_assets(cfg_dict):
        raise ValueError(f"{entry['run_id']}: data/cache paths differ from lock")
    if entry.get("run_manifest_sha256") and sha256_file(
            run_dir / "run_manifest.json") != entry["run_manifest_sha256"]:
        raise ValueError(f"{entry['run_id']}: run manifest hash differs from lock")
    cfg = Config(cfg_dict)
    proto = SplitProtocol.from_cfg(cfg)
    if tuple(proto.test_years) != tuple(test_years):
        raise ValueError(f"{entry['run_id']}: protocol test years "
                         f"{proto.test_years} != lock {test_years}")
    bundle = load_prior_bundle(
        _resolved(cfg.get("data.prior_manifest")), proto)
    if not entry.get("prior_manifest_sha256") or \
            bundle["manifest_sha256"] != entry["prior_manifest_sha256"]:
        raise ValueError(f"{entry['run_id']}: prior manifest hash differs from "
                         f"the lock")
    verify_canonical_sample_manifest(cfg, bundle)

    def check_path(cfg_key, locked_path, what):
        configured = cfg.get(cfg_key)
        if not configured or _resolved(configured) != Path(locked_path).resolve():
            raise ValueError(f"{entry['run_id']}: {what} path differs from the "
                             f"verified prior bundle")

    check_path("data.panel_parquet", bundle["files"]["panel"]["path"], "panel")
    check_path("data.tbb_norm", bundle["assets"]["tbb_norm"]["path"], "TBB norm")
    if "frames_index" in bundle["assets"]:
        check_path("data.frames_index", bundle["assets"]["frames_index"]["path"],
                   "frames index")
    cfg_dict["data"]["graph_edges_json"] = bundle["files"]["graph_edges"]["path"]
    prior = load_prior_graph(
        Config(cfg_dict),
        expected_sha256=bundle["files"]["graph_edges"]["sha256"],
        require_metadata=True)
    shuffle_seed = cfg.get("model.gnn.prior_shuffle_seed", None)
    if shuffle_seed is not None and int(shuffle_seed) > 0:
        prior = shuffle_prior_graph(prior, int(shuffle_seed))
    from wpac_warning.mae_provenance import verify_mae_for_config
    mae_provenance = verify_mae_for_config(cfg)
    model = CausalRINet(Config(cfg_dict), prior=prior)
    ckpt = run_dir / "checkpoints" / "best_ri.pt"
    if not entry.get("checkpoint_sha256") or \
            sha256_file(ckpt) != entry["checkpoint_sha256"]:
        raise ValueError(f"{entry['run_id']}: checkpoint hash differs from lock")
    info, bb = load_inference_checkpoint(ckpt, model, verify_backbone=True)
    if info.get("config_sha256") != config_json_sha256:
        raise ValueError(f"{entry['run_id']}: checkpoint config_sha256 does not "
                         f"match the locked resolved config")
    if info.get("prior_manifest_sha256") not in (None, entry["prior_manifest_sha256"]):
        raise ValueError(f"{entry['run_id']}: checkpoint prior hash differs from lock")
    model.eval()
    return {
        "run_id": entry["run_id"],
        "cfg": Config(cfg_dict),
        "cfg_dict": cfg_dict,
        "prior": prior,
        "model": model,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "ckpt_info": info,
        "backbone": bb,
        "mae_provenance": mae_provenance,
        "weight": float(entry.get("weight", 0.0)),
    }


def _check_test_keys(lock: dict, keys: dict) -> None:
    want = lock.get("evaluation", {}).get("test_row_key_sha256")
    if not want:
        raise ValueError("final lock has no test_row_key_sha256")
    got = sha256_array(sorted(keys["row_uid"].astype(str).tolist()))
    if got != want:
        raise ValueError("test rows differ from the frozen final-lock row keys")
    n_want = lock["evaluation"].get("n_test_rows")
    if n_want is not None and int(n_want) != len(keys["row_uid"]):
        raise ValueError("test row count differs from the final lock")


def infer_member(entry: dict, test_years: tuple, runs_root: Path,
                 batch_size: int, device: str) -> dict:
    """Run one member through its OWN data pipeline, then release the model."""
    member = build_member(entry, test_years, runs_root / entry["run_id"])
    ds = RIPanelDataset(member["cfg"])
    if (member["cfg"].get("data.ir_embeds_dir")
            and member["cfg"].get("data.ir_lag_steps")
            and ds.embedding_cache_audit is None):
        raise ValueError(f"{entry['run_id']}: unaudited embedding cache")
    test_idx = ds.split_indices.get("test", np.array([], dtype=int))
    if not len(test_idx):
        raise RuntimeError(f"{entry['run_id']}: empty test split")
    rows = ds.df.iloc[test_idx].reset_index(drop=True)
    keys = {
        "sid": rows["SID"].astype(str).values,
        "iso_time": rows["iso_time_str"].values,
        "storm_year": rows["storm_year"].astype(np.int32).values,
        "row_uid": rows["row_uid"].values,
        "split": np.full(len(rows), "test", dtype=object),
    }
    y = rows["RI"].astype(int).values
    loader = DataLoader(torch.utils.data.Subset(ds, test_idx),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info(f"inferring {entry['run_id']} ({member['checkpoint_sha256'][:12]}…)")
    p, y_m, gate_mean, gate_std, causal_share = predict_probs_gates(
        member["model"].to(device), loader, member["prior"], device)
    p = _valid_probabilities(p, entry["run_id"])
    if not np.array_equal(y_m.astype(int), y):
        raise RuntimeError(f"{entry['run_id']}: label order mismatch on test")
    meta = {
        "run_id": entry["run_id"], "checkpoint": member["checkpoint"],
        "checkpoint_sha256": member["checkpoint_sha256"],
        "config_sha256": entry["config_sha256"],
        "config_json_sha256": entry["config_json_sha256"],
        "prior_manifest_sha256": entry["prior_manifest_sha256"],
        "ckpt_mode": member["ckpt_info"].get("mode"),
        "shuffle_seed": int(member["ckpt_info"].get("prior_shuffle_seed", 0) or 0),
        "weight": float(entry.get("weight", 0.0)),
        "ir_embeds": ds.embedding_cache_audit,
        "mae_provenance": member["mae_provenance"],
    }
    del member, ds, loader
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"p": p, "y": y, "keys": keys, "meta": meta,
            "gate_mean": gate_mean, "gate_std": gate_std,
            "causal_share": causal_share}


def _prepare_receipt(lock: dict, output_dir: Path, names: list,
                     outputs: list, resume: bool) -> tuple:
    receipt_path = _resolved(lock["test_receipt"])
    expected = {name: _resolved(out) for name, out in zip(names, outputs)}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("protocol") != LOCK_KIND:
            raise ValueError("test receipt belongs to a different protocol")
        if receipt.get("status") == "completed":
            if receipt.get("lock_sha256") != lock["lock_sha256"]:
                raise SystemExit("test inference already completed under another "
                                 "final lock; the protocol-level receipt cannot be reused")
            raise SystemExit("test inference already completed — it runs once; "
                             "changing --out/--out-dir cannot repeat it")
        if receipt.get("lock_sha256") != lock["lock_sha256"]:
            raise ValueError("test receipt belongs to a different final lock")
        if _resolved(receipt.get("test_output_dir", "")) != output_dir:
            raise ValueError("test receipt output directory differs from the lock")
        if receipt.get("status") != "running":
            raise ValueError("test receipt has an invalid status")
        if not resume:
            raise SystemExit("interrupted test inference receipt exists; fault "
                             "recovery requires explicit --resume")
    else:
        if resume:
            raise SystemExit("--resume requires an existing interrupted test receipt")
        existing = [p for out in outputs
                    for p in (out, out.with_suffix(".manifest.json")) if p.exists()]
        if existing:
            raise SystemExit(f"test output exists without a receipt: {existing[0]} — "
                             f"this is not a fresh output and cannot be rerun safely")
        now = datetime.now(timezone.utc).isoformat()
        receipt = {
            "protocol": LOCK_KIND, "status": "running",
            "lock_sha256": lock["lock_sha256"],
            "test_output_dir": str(output_dir),
            "created_at": now, "updated_at": now, "outputs": {},
        }
        _atomic_write_json(receipt_path, receipt)

    recorded = receipt.get("outputs", {})
    if not isinstance(recorded, dict) or set(recorded) - set(names):
        raise ValueError("test receipt contains unknown or invalid outputs")
    completed = {}
    for name, record in recorded.items():
        out = expected[name]
        manifest_path = out.with_suffix(".manifest.json")
        if _resolved(record.get("output", "")) != out:
            raise ValueError(f"{name}: receipt output path changed")
        if not out.is_file() or not manifest_path.is_file():
            raise ValueError(f"{name}: receipt output or manifest is missing")
        if sha256_file(out) != record.get("output_sha256"):
            raise ValueError(f"{name}: output hash differs from receipt")
        if sha256_file(manifest_path) != record.get("manifest_sha256"):
            raise ValueError(f"{name}: manifest hash differs from receipt")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("lock_sha256") != lock["lock_sha256"]
                or manifest.get("entity") != name
                or manifest.get("output_sha256") != record["output_sha256"]):
            raise ValueError(f"{name}: output manifest does not match receipt/lock")
        completed[name] = out
    for name, out in expected.items():
        if name not in completed and (out.exists() or
                out.with_suffix(".manifest.json").exists()):
            raise SystemExit(f"{name}: partial output is not registered in the "
                             f"receipt; remove the incomplete files before --resume")
    return receipt, completed


def _record_output(receipt: dict, receipt_path: Path, name: str,
                   out: Path) -> None:
    manifest_path = out.with_suffix(".manifest.json")
    receipt["outputs"][name] = {
        "output": str(out.resolve()), "output_sha256": sha256_file(out),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }
    receipt["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(receipt_path, receipt, overwrite=True)


def _completed_result(out: Path) -> dict:
    with np.load(out, allow_pickle=True) as z:
        _valid_probabilities(z["p"], str(out))
        return {
            "y": z["y"], "p": z["p"],
            "keys": {key: z[key] for key in
                     ("sid", "iso_time", "storm_year", "row_uid", "split")},
        }


def _merge_reference(reference, result: dict, run_id: str):
    if reference is None:
        return result["y"], result["keys"]
    y_ref, keys_ref = reference
    if not np.array_equal(result["y"], y_ref):
        raise RuntimeError(f"{run_id}: labels differ across members")
    for key in ("sid", "iso_time", "storm_year", "row_uid"):
        if not np.array_equal(result["keys"][key], keys_ref[key]):
            raise RuntimeError(f"{run_id}: test row order differs for {key}")
    return reference


def _verify_baselines(lock: dict) -> None:
    baselines = lock.get("baselines", [])
    if not baselines:
        return
    info = lock["baseline_manifest"]
    manifest_path = _resolved(info["path"])
    if info.get("type") == "file":
        if sha256_file(manifest_path) != info.get("sha256"):
            raise ValueError("baseline manifest hash differs from final lock")
    else:
        current = {b["name"]: sha256_file(_resolved(b["run_dir"]) /
                                          "run_manifest.json") for b in baselines}
        if sha256_json(current) != info.get("sha256"):
            raise ValueError("baseline directory manifest hashes differ from lock")
    for baseline in baselines:
        run_dir = _resolved(baseline["run_dir"])
        for filename, field in (("model.pkl", "model_sha256"),
                                ("features.parquet", "features_sha256"),
                                ("config.yaml", "config_sha256"),
                                ("inner.npz", "inner_npz_sha256"),
                                ("run_manifest.json", "run_manifest_sha256")):
            if not baseline.get(field) or sha256_file(run_dir / filename) != baseline[field]:
                raise ValueError(f"{baseline['name']}: baseline {filename} hash "
                                 f"differs from lock")
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if sha256_json(cfg) != baseline.get("config_json_sha256"):
            raise ValueError(f"{baseline['name']}: semantic config hash differs")
        if _data_assets(cfg) != baseline.get("data_assets"):
            raise ValueError(f"{baseline['name']}: baseline data/cache paths differ")
        if baseline.get("kind") == "ships12":
            # reject an OHCD arm inherited from the pre-fix archived table
            from wpac_warning.baseline_common import require_ships12_run_provenance
            require_ships12_run_provenance(run_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--out", default=None, help="single-entity output npz")
    ap.add_argument("--out-dir", default=None, help="multi-entity output directory")
    ap.add_argument("--runs-root", default=None,
                    help="must match the final lock; cannot redirect inference")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--allow-fixture", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="recover an interrupted receipt without recomputing completed entities")
    ap.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if bool(args.out) == bool(args.out_dir):
        raise SystemExit("specify exactly one of --out or --out-dir")
    if args.force:
        raise SystemExit("--force cannot override the once-only test protocol")

    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    verify_lock(lock, allow_fixture=args.allow_fixture)
    if "frozen_at" not in lock:
        raise ValueError("lock has no frozen_at timestamp")
    test_years = tuple(lock["protocol_years"]["test_years"])
    entities = lock["entities"]
    baselines = lock.get("baselines", [])
    output_dir = _resolved(lock["test_output_dir"])
    if args.out and (len(entities) != 1 or baselines):
        raise SystemExit("--out is only valid for a single-entity lock with no "
                         "baselines; use --out-dir")
    if args.out:
        requested = _resolved(args.out)
        if requested.parent != output_dir:
            raise SystemExit(f"--out must be inside locked test_output_dir: {output_dir}")
        outputs = [requested]
    else:
        if _resolved(args.out_dir) != output_dir:
            raise SystemExit(f"--out-dir differs from locked test_output_dir: {output_dir}")
        outputs = [output_dir / f"{e['name']}.npz" for e in entities]
        outputs += [output_dir / f"{b['name']}.npz" for b in baselines]
    names = [e["name"] for e in entities] + [b["name"] for b in baselines]
    receipt_path = _resolved(lock["test_receipt"])
    _verify_baselines(lock)
    runs_root = _resolved(lock["runs_root"])
    if args.runs_root and _resolved(args.runs_root) != runs_root:
        raise ValueError("--runs-root differs from the final lock")
    ablations_root = (_resolved(lock["ablations_root"])
                      if lock.get("ablations_root") else None)
    receipt, completed = _prepare_receipt(
        lock, output_dir, names, outputs, resume=args.resume)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {}
    reference = None
    for entity, out in zip(entities, outputs):
        if entity["name"] in completed:
            result = _completed_result(out)
            _check_test_keys(lock, result["keys"])
            reference = _merge_reference(reference, result, entity["name"])
            logger.info(f"{entity['name']}: verified completed output; skipping")
            continue
        probs, metas, member_results = [], [], []
        entity_root = (ablations_root if entity.get("kind") == "ablation"
                       else runs_root)
        for entry in entity["members"]:
            run_id = entry["run_id"]
            cache_key = (str(entity_root), run_id, entry.get("config_sha256"),
                         entry.get("checkpoint_sha256"))
            if cache_key not in cache:
                cache[cache_key] = infer_member(entry, test_years, entity_root,
                                                args.batch_size, device)
            result = cache[cache_key]
            reference = _merge_reference(reference, result, run_id)
            probs.append(result["p"])
            member_results.append(result)
            meta = dict(result["meta"])
            meta["weight"] = float(entry.get("weight", 0.0))
            metas.append(meta)
        p_mat = np.stack(probs, axis=1)
        gate_mean_mat = np.stack([r["gate_mean"] for r in member_results],
                                 axis=1)
        gate_std_mat = np.stack([r["gate_std"] for r in member_results],
                                axis=1)
        causal_share_mat = np.stack([r["causal_share"] for r in member_results],
                                    axis=1)
        assert gate_mean_mat.shape == (len(reference[0]), len(metas)), (
            f"gate matrix shape {gate_mean_mat.shape} != "
            f"({len(reference[0])}, {len(metas)})")
        weights = np.array([m["weight"] for m in metas], dtype=float)
        if abs(weights.sum() - 1.0) > 1e-6:
            raise ValueError(f"{entity['name']}: member weights sum to {weights.sum()}")
        cond = entity.get("condition") or ""
        want_seed = int(cond[3:]) if cond.startswith("C3s") else 0
        for meta in metas:
            if int(meta.get("shuffle_seed", 0)) != want_seed:
                raise ValueError(
                    f"{entity['name']}: condition {cond or 'M0'} requires "
                    f"prior_shuffle_seed={want_seed}, member "
                    f"{meta['run_id']} has {meta.get('shuffle_seed', 0)}")
        p_ens = _valid_probabilities(p_mat @ weights, entity["name"])
        y, keys = reference
        _check_test_keys(lock, keys)
        _atomic_savez(
            out, p=p_ens, p_members=p_mat, y=y,
            member_names=np.array([m["run_id"] for m in metas], dtype=object),
            weights=weights, threshold=float(entity["threshold"]),
            gate_mean_members=gate_mean_mat.astype(np.float32),
            gate_std_members=gate_std_mat.astype(np.float32),
            causal_share_members=causal_share_mat.astype(np.float32),
            entity=entity["name"], lock_sha256=lock["lock_sha256"], **keys)
        manifest = {
            "protocol": LOCK_KIND, "entity": entity["name"], "kind": entity["kind"],
            "condition": entity.get("condition"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "lock": str(Path(args.lock).resolve()), "lock_sha256": lock["lock_sha256"],
            "n_rows": int(len(y)), "n_members": len(metas),
            "threshold": float(entity["threshold"]), "members": metas,
            "row_key_sha256": sha256_json([str(x) for x in keys["row_uid"]]),
            "output": str(out.resolve()), "output_sha256": sha256_file(out),
        }
        _atomic_write_json(out.with_suffix(".manifest.json"), manifest)
        _record_output(receipt, receipt_path, entity["name"], out)
        logger.info(f"{entity['name']}: {len(y)} rows -> {out}")

    if baselines:
        if reference is None:
            raise RuntimeError("baseline inference needs a model entity to fix "
                               "the test row order")
        y, keys = reference
        _check_test_keys(lock, keys)
        for baseline, out in zip(baselines, outputs[len(entities):]):
            if baseline["name"] in completed:
                logger.info(f"{baseline['name']}: verified completed output; skipping")
                continue
            run_dir = _resolved(baseline["run_dir"])
            p = _valid_probabilities(
                predict_baseline(run_dir, keys["row_uid"]), baseline["name"])
            _atomic_savez(
                out, p=p, p_members=p[:, None], y=y,
                member_names=np.array([baseline["name"]], dtype=object),
                weights=np.array([1.0]), threshold=float(baseline["threshold"]),
                entity=baseline["name"], kind="baseline",
                lock_sha256=lock["lock_sha256"], **keys)
            manifest = {
                "protocol": LOCK_KIND, "entity": baseline["name"], "kind": "baseline",
                "baseline_kind": baseline["kind"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "lock": str(Path(args.lock).resolve()),
                "lock_sha256": lock["lock_sha256"],
                "n_rows": int(len(y)), "threshold": float(baseline["threshold"]),
                "model_sha256": baseline["model_sha256"],
                "features_sha256": baseline["features_sha256"],
                "config_sha256": baseline["config_sha256"],
                "run_manifest_sha256": baseline["run_manifest_sha256"],
                "row_key_sha256": sha256_json([str(x) for x in keys["row_uid"]]),
                "output": str(out.resolve()), "output_sha256": sha256_file(out),
            }
            _atomic_write_json(out.with_suffix(".manifest.json"), manifest)
            _record_output(receipt, receipt_path, baseline["name"], out)
            logger.info(f"{baseline['name']}: {len(y)} rows -> {out}")
    receipt["status"] = "completed"
    receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
    receipt["updated_at"] = receipt["completed_at"]
    _atomic_write_json(receipt_path, receipt, overwrite=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
