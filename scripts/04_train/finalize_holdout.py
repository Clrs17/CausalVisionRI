# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Create the FINAL HOLDOUT_MASKED_V1 lock that alone unlocks test inference.

The intermediate selection lock only freezes the 22-arm inner selection.  A
final lock additionally requires the 40 inner-only ablation runs and the
inner-only statistical baselines (LR / LDA / SHIPS-12), reconciles every main
and sensitivity artifact with selection, checks one global prior/data set and
M0-plus-shared-initialization-plus-condition ablation configs, and freezes development/test keys,
statistical policy and the once-only test output/receipt paths.

Production (no ``--fixture``) is strictly checked:
  * 22 arms in the selection lock (registry enforced),
  * exactly 40 ablation run directories (C1,C2,C3s1-3,C4adj,C4sign,C4cf x 5),
  * 49 entities: main + k4/k6/all22 + 5 per-member M0 + 40 ablations,
  * at least one LR, one LDA and one SHIPS-12 baseline,
  * one shared prior manifest and identical shared data assets across all runs.

``--fixture`` is a synthetic smoke mode and must be paired with
``--allow-fixture``; consumers also need ``--allow-fixture``.

Usage (from the release root):
    python scripts/04_train/finalize_holdout.py --selection protocol_locks/selection.json \
        --runs-root data/runs/holdout_masked_v1 \
        --ablations-root data/runs_holdout/ablations \
        --baselines data/runs_holdout/baselines/baselines_manifest.json \
        --out protocol_locks/final.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.arms import arm_names, load_registry                  # noqa: E402
from wpac_warning.dataset import RIPanelDataset                        # noqa: E402
from wpac_warning.protocol import (SplitProtocol, canonical_data_assets,  # noqa: E402
                          check_test_receipt_available, portable_path,
                          protocol_test_receipt, resolve_protocol_test_output_dir,
                          sha256_array, sha256_file, sha256_json)
from wpac_warning.run_integrity import load_completed_marker         # noqa: E402
from wpac_warning.utils_cfg import Config, load_cfg                    # noqa: E402
from run_holdout_ablations import condition_overrides  # noqa: E402

CONDITIONS = ("C1", "C2", "C3s1", "C3s2", "C3s3", "C4adj", "C4sign", "C4cf")
C3_CONDITIONS = ("C3s1", "C3s2", "C3s3")
BASELINE_KINDS_REQUIRED = ("lr", "lda", "ships12")
N_EXPECTED_ENTITIES = 49          # main + k4 + k6 + all22 + 5 M0 + 40 ablations


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (P2_ROOT / p).resolve()


def _verify_lock(lock: dict, kind: str) -> None:
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if sha256_json(payload) != lock.get("lock_sha256"):
        raise ValueError(f"{kind} lock hash mismatch")
    if lock.get("kind") != kind:
        raise ValueError(f"expected a {kind} lock, got {lock.get('kind')!r}")


def _read_inner(path: Path) -> tuple:
    with np.load(path, allow_pickle=True) as z:
        required = ("p_inner", "y_inner", "row_uid", "threshold")
        missing = [key for key in required if key not in z.files]
        if missing:
            raise ValueError(f"{path}: missing inner fields {missing}")
        p = np.asarray(z["p_inner"], dtype=float).ravel()
        y = np.asarray(z["y_inner"], dtype=float).ravel()
        keys = np.asarray(z["row_uid"]).astype(str).ravel()
        threshold = float(z["threshold"])
    if not len(keys) or not (len(keys) == len(p) == len(y)):
        raise ValueError(f"{path}: empty or inconsistent inner arrays")
    if len(np.unique(keys)) != len(keys):
        raise ValueError(f"{path}: duplicate inner row_uid values")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{path}: p_inner must be finite and within [0,1]")
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{path}: y_inner must be finite and binary")
    if not np.isfinite(threshold):
        raise ValueError(f"{path}: inner threshold is NaN/Inf")
    return threshold, keys, y.astype(int)


def _public(value):
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items()
                if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, tuple):
        return [_public(item) for item in value]
    return value


def _run_meta(run_dir: Path, weight: float) -> dict:
    run_dir = Path(run_dir)
    ckpt = run_dir / "checkpoints" / "best_ri.pt"
    cfg_path = run_dir / "config.yaml"
    pred = run_dir / "preds" / "inner.npz"
    manifest_path = run_dir / "run_manifest.json"
    for path in (ckpt, cfg_path, pred, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"{run_dir.name}: required artifact "
                                    f"missing: {path}")
    completed = load_completed_marker(run_dir)
    initialization = _load_json(run_dir / "initialization.json")
    initial_sha = (initialization.get("initial_checkpoint") or {}).get("sha256")
    if not initial_sha or completed["artifacts"]["checkpoints/initial.pt"] != initial_sha:
        raise ValueError(f"{run_dir.name}: initialization/initial.pt/completed hashes disagree")
    fit_shuffle = (initialization.get("fit_shuffle") or {}).get(
        "first_epoch_order_sha256")
    if not fit_shuffle:
        raise ValueError(f"{run_dir.name}: initialization lacks fit shuffle hash")
    load_audit = initialization.get("load_audit")
    initial_source_sha = (load_audit or {}).get("source_sha256") \
        if isinstance(load_audit, dict) else None
    manifest = _load_json(manifest_path)
    if manifest.get("protocol") is not None:
        proto_meta = manifest.get("protocol")
        if isinstance(proto_meta, dict):
            if str(proto_meta.get("protocol")) != "holdout_masked_v1":
                raise ValueError(f"{run_dir.name}: not a holdout run")
    threshold, inner_keys, inner_labels = _read_inner(pred)
    cfg_dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if str(cfg_dict.get("data", {}).get("protocol")) != "holdout_masked_v1":
        raise ValueError(f"{run_dir.name}: run config is not holdout_masked_v1")
    config_json_sha256 = sha256_json(cfg_dict)
    if manifest.get("config_sha256") != config_json_sha256:
        raise ValueError(f"{run_dir.name}: run manifest config_sha256 does not "
                         f"match run config.yaml")
    return {
        "run_id": run_dir.name,
        "_run_dir": str(run_dir.resolve()),
        "weight": float(weight),
        "initial_sha256": initial_sha,
        "initial_source_sha256": initial_source_sha,
        "fit_shuffle_sha256": fit_shuffle,
        "completed_artifact_catalog_sha256": completed["artifact_catalog_sha256"],
        "inner_threshold": threshold,
        "checkpoint_sha256": sha256_file(ckpt),
        "config_sha256": sha256_file(cfg_path),
        "config_json_sha256": config_json_sha256,
        "run_manifest_sha256": sha256_file(manifest_path),
        "prior_manifest_sha256": manifest.get("prior_manifest_sha256"),
        "inner_npz_sha256": sha256_file(pred),
        "inner_row_key_sha256": sha256_array(sorted(inner_keys.tolist())),
        "inner_row_label_sha256": sha256_array(
            [f"{key}|{int(label)}" for key, label in zip(inner_keys, inner_labels)]),
        "_inner_keys": inner_keys,
        "_inner_labels": inner_labels,
        "data_assets": _data_assets(cfg_dict),
        "shuffle_seed": int(cfg_dict.get("model", {}).get("gnn", {})
                            .get("prior_shuffle_seed", 0) or 0),
    }


def _data_assets(cfg_dict: dict) -> dict:
    return canonical_data_assets(cfg_dict, P2_ROOT)


def _shared_data_assets(cfg_dict: dict) -> dict:
    assets = _data_assets(cfg_dict)
    assets.pop("ir_embeds_dir", None)  # backbone-specific, pinned per member
    return assets


def _check_data_assets(metas: list, what: str) -> None:
    assets = {json.dumps(_shared_data_assets(m["cfg_dict"]), sort_keys=True)
              for m in metas}
    if len(assets) != 1:
        raise ValueError(f"{what}: runs use different data assets: {assets}")


def _check_inner_rows(metas: list, what: str) -> None:
    reference = metas[0]
    for meta in metas[1:]:
        name = meta.get("run_id") or meta.get("name")
        if meta["inner_row_key_sha256"] != reference["inner_row_key_sha256"]:
            raise ValueError(f"{what}: {name}: inner_row_key_sha256 differs from "
                             f"the main sample")
        if not np.array_equal(meta["_inner_keys"], reference["_inner_keys"]):
            raise ValueError(f"{what}: {name}: inner row keys/order differ from "
                             f"the main sample")
        if not np.array_equal(meta["_inner_labels"], reference["_inner_labels"]):
            raise ValueError(f"{what}: {name}: inner labels differ row-by-row "
                             f"from the main sample")


def _check_shared(metas: list, what: str) -> None:
    priors = {m["prior_manifest_sha256"] for m in metas}
    if None in priors or len(priors) != 1:
        raise ValueError(f"{what}: runs use different or missing prior "
                         f"manifests: {priors}")
    _check_inner_rows(metas, what)
    _check_data_assets(metas, what)


def _member(runs_root: Path, run_id: str, weight: float) -> dict:
    meta = _run_meta(runs_root / run_id, weight)
    with open(runs_root / run_id / "config.yaml", "r", encoding="utf-8") as f:
        meta["cfg_dict"] = yaml.safe_load(f)
    return meta


def _reconcile_artifact(meta: dict, selected: dict, what: str) -> None:
    fields = ("checkpoint_sha256", "config_sha256", "config_json_sha256",
              "prior_manifest_sha256", "inner_threshold", "data_assets")
    for field in fields:
        if field not in selected or selected[field] is None:
            raise ValueError(f"{what}: selection artifact lacks {field}")
        if meta[field] != selected[field]:
            raise ValueError(f"{what}: {field} differs from selection")
    selected_inner_hash = (selected.get("inner_npz_sha256")
                           or selected.get("npz_sha256"))
    if not selected_inner_hash:
        raise ValueError(f"{what}: selection artifact lacks inner npz hash")
    if meta["inner_npz_sha256"] != selected_inner_hash:
        raise ValueError(f"{what}: inner_npz_sha256 differs from selection")


def _reconcile_selection(selection: dict, metas: list) -> None:
    catalog = selection.get("arm_artifacts")
    if not isinstance(catalog, dict):
        raise ValueError("selection lock lacks the per-arm artifact catalog")
    for meta in metas:
        run_id = meta["run_id"]
        if run_id not in catalog:
            raise ValueError(f"{run_id}: absent from selection artifact catalog")
        _reconcile_artifact(meta, catalog[run_id], run_id)
    selected_members = {m["run_id"]: m for m in selection["members"]}
    for meta in metas:
        if meta["run_id"] in selected_members:
            _reconcile_artifact(meta, selected_members[meta["run_id"]],
                                f"selected member {meta['run_id']}")


def _development_reference(cfg) -> dict:
    ds = RIPanelDataset(cfg)
    fit_idx = ds.split_indices["fit"]
    inner_idx = ds.split_indices["inner"]
    test_idx = ds.split_indices["test"]
    rows = ds.df.iloc[np.concatenate([fit_idx, inner_idx])]
    if not len(rows):
        raise ValueError("development reference has no eligible fit+inner rows")
    if not len(test_idx):
        raise ValueError("test split is empty")
    test_rows = ds.df.iloc[test_idx]
    availability = ds.frame_availability_audit
    availability_summary = {
        key: availability[key] for key in
        ("path", "sha256", "rule_version", "n_eligible_rows",
         "n_available_rows", "n_excluded_rows", "excluded_row_key_sha256",
         "available_row_key_sha256", "by_split")
    }
    return {
        "frame_availability": availability_summary,
        "dev_rate": float(rows["RI"].mean()),
        "n_dev_rows": int(len(rows)),
        "n_fit_rows": int(len(fit_idx)),
        "n_inner_rows": int(len(inner_idx)),
        "dev_row_key_sha256": sha256_array(
            sorted(rows["row_uid"].astype(str).tolist())),
        "n_test_rows": int(len(test_idx)),
        "test_row_key_sha256": sha256_array(
            sorted(test_rows["row_uid"].astype(str).tolist())),
        "fit_years": list(ds.split_protocol.fit_years),
        "inner_years": list(ds.split_protocol.inner_years),
        "test_years": list(ds.split_protocol.test_years),
        "embargo_h": ds.split_protocol.embargo_h,
    }


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    target = cfg
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _flat_config(value, prefix="") -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flat_config(item, dotted))
        return out
    return {prefix: value}


def _check_ablation(meta: dict, condition: str, base: dict) -> None:
    seed = int(condition[3:]) if condition.startswith("C3s") else 0
    if condition in C3_CONDITIONS:
        if meta["shuffle_seed"] != seed:
            raise ValueError(
                f"{meta['run_id']}: {condition} requires prior_shuffle_seed="
                f"{seed}, config has {meta['shuffle_seed']}")
    elif meta["shuffle_seed"] != 0:
        raise ValueError(f"{meta['run_id']}: {condition} must not shuffle the "
                         f"prior (seed={meta['shuffle_seed']})")
    if meta["initial_source_sha256"] != base["initial_sha256"]:
        raise ValueError(
            f"{meta['run_id']}: ablation initial source hash differs from "
            f"{base['run_id']} initial.pt")
    if meta["fit_shuffle_sha256"] != base["fit_shuffle_sha256"]:
        raise ValueError(
            f"{meta['run_id']}: ablation fit shuffle differs from {base['run_id']}")

    expected = copy.deepcopy(base["cfg_dict"])
    _set_dotted(expected, "train.run_id", meta["run_id"])
    expected_initial = _resolve(base["_run_dir"]) / "checkpoints" / "initial.pt"
    _set_dotted(expected, "train.init_from", str(expected_initial))
    for key, value in condition_overrides(condition).items():
        _set_dotted(expected, key, value)
    actual = copy.deepcopy(meta["cfg_dict"])
    # Output location is provenance, not a modelling degree of freedom.
    expected.get("train", {}).pop("output_root", None)
    actual.get("train", {}).pop("output_root", None)
    if actual != expected:
        flat_expected = _flat_config(expected)
        flat_actual = _flat_config(actual)
        differences = {
            key: {"expected_m0_plus_condition": flat_expected.get(key, "<missing>"),
                  "actual": flat_actual.get(key, "<missing>")}
            for key in sorted(set(flat_expected) | set(flat_actual))
            if flat_actual.get(key, "<missing>") != flat_expected.get(key, "<missing>")
        }
        raise ValueError(f"{meta['run_id']}: {condition} config differs from its "
                         f"M0 beyond the condition overrides: "
                         f"{json.dumps(differences, sort_keys=True, default=str)}")


def _baseline_entries(manifest_path: Path) -> list:
    if manifest_path.is_dir():
        entries = []
        for d in sorted(manifest_path.iterdir()):
            man = d / "run_manifest.json"
            if man.is_file():
                m = _load_json(man)
                if m.get("kind") in BASELINE_KINDS_REQUIRED:
                    entries.append({"name": d.name, "kind": m["kind"],
                                    "run_dir": str(d)})
        raw = {"baselines": entries}
    else:
        raw = _load_json(manifest_path)
    baselines = raw.get("baselines", [])
    if not baselines:
        raise ValueError("baseline manifest is empty")
    out = []
    seen = set()
    for b in baselines:
        for key in ("name", "kind", "run_dir"):
            if key not in b:
                raise ValueError(f"baseline entry missing {key}: {b}")
        run_dir = _resolve(b["run_dir"])
        for fname in ("inner.npz", "config.yaml", "model.pkl",
                      "features.parquet", "run_manifest.json"):
            if not (run_dir / fname).is_file():
                raise FileNotFoundError(f"{run_dir}: missing {fname}")
        meta = _load_json(run_dir / "run_manifest.json")
        if meta.get("protocol") != "holdout_masked_v1" or not bool(
                meta.get("inner_only", False)):
            raise ValueError(f"{run_dir}: baseline run is not an inner-only "
                             f"holdout run (protocol={meta.get('protocol')!r})")
        cfg_dict = yaml.safe_load(
            (run_dir / "config.yaml").read_text(encoding="utf-8"))
        config_json_sha256 = sha256_json(cfg_dict)
        if meta.get("config_sha256") != config_json_sha256:
            raise ValueError(f"{run_dir}: baseline manifest config_sha256 does "
                             f"not match config.yaml")
        if meta.get("kind") == "ships12":
            # reject an OHCD arm inherited from the pre-fix archived table
            from wpac_warning.baseline_common import require_ships12_run_provenance
            require_ships12_run_provenance(run_dir)
        thr, keys, labels = _read_inner(run_dir / "inner.npz")
        entry = {
            "name": str(b["name"]),
            "kind": str(b["kind"]),
            "run_dir": portable_path(run_dir),
            "threshold": thr,
            "inner_npz_sha256": sha256_file(run_dir / "inner.npz"),
            "config_sha256": sha256_file(run_dir / "config.yaml"),
            "config_json_sha256": config_json_sha256,
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "model_sha256": sha256_file(run_dir / "model.pkl"),
            "features_sha256": sha256_file(run_dir / "features.parquet"),
            "inner_row_key_sha256": sha256_array(sorted(keys.tolist())),
            "inner_row_label_sha256": sha256_array(
                [f"{key}|{int(label)}" for key, label in zip(keys, labels)]),
            "_inner_keys": keys,
            "_inner_labels": labels,
            "data_assets": _data_assets(cfg_dict),
            "protocol": meta.get("protocol"),
            "ohc_var": meta.get("ohc_var"),
            "main_or_sensitivity": meta.get("main_or_sensitivity"),
        }
        if entry["kind"] not in BASELINE_KINDS_REQUIRED:
            raise ValueError(f"{entry['name']}: unsupported baseline kind "
                             f"{entry['kind']!r}")
        if entry["name"] in seen:
            raise ValueError(f"duplicate baseline name {entry['name']!r}")
        seen.add(entry["name"])
        out.append(entry)
    kinds = {b["kind"] for b in out}
    missing = set(BASELINE_KINDS_REQUIRED) - kinds
    if missing:
        raise ValueError(f"production finalization requires baselines for "
                         f"{sorted(BASELINE_KINDS_REQUIRED)}; missing "
                         f"{sorted(missing)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--ablations-root", default=None)
    ap.add_argument("--baselines", default=None,
                    help="JSON manifest of completed inner-only baselines")
    ap.add_argument("--test-output-dir", default=None,
                    help="must match the selection-locked test output directory")
    ap.add_argument("--fixture", action="store_true",
                    help="synthetic smoke only: main entity only")
    ap.add_argument("--allow-fixture", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.fixture and not args.allow_fixture:
        raise SystemExit("--fixture requires an explicit --allow-fixture")
    if not args.fixture and not args.allow_fixture:
        canonical = resolve_protocol_test_output_dir()
        check_test_receipt_available(protocol_test_receipt(canonical),
                                     block_running=True)

    out = Path(args.out)
    if out.exists() and not args.force:
        raise FileExistsError(f"final lock already exists: {out}")

    selection = _load_json(args.selection)
    _verify_lock(selection, "selection")
    if selection.get("kind") != "selection":
        raise ValueError("finalization requires the intermediate selection lock")
    if selection.get("protocol") != "holdout_masked_v1":
        raise ValueError("selection lock protocol mismatch")
    if selection.get("fixture") and not args.fixture:
        raise ValueError("fixture selection can only create a fixture final lock")
    reg = load_registry(_resolve(selection["registry"]["path"]))
    if selection["registry"]["sha256"] != sha256_file(reg["_path"]):
        raise ValueError("registry changed since selection")
    if len(selection["members"]) != int(selection["selection_rule"]["k"]):
        raise ValueError("selection members != k")
    if not selection.get("test_output_dir") or not selection.get("test_receipt"):
        raise ValueError("selection lock lacks protocol-fixed test output/receipt")
    allow_fixture_paths = bool(args.allow_fixture or args.fixture)
    test_output_dir = resolve_protocol_test_output_dir(
        selection["test_output_dir"], allow_fixture=allow_fixture_paths)
    if args.test_output_dir:
        requested = resolve_protocol_test_output_dir(
            args.test_output_dir, allow_fixture=allow_fixture_paths)
        if requested != test_output_dir:
            raise ValueError("--test-output-dir cannot differ from the selection lock")
    test_receipt = protocol_test_receipt(test_output_dir)
    if _resolve(selection["test_receipt"]) != test_receipt:
        raise ValueError("selection test_receipt must be inside test_output_dir")
    if not args.fixture:
        check_test_receipt_available(test_receipt, block_running=True)

    runs_root = _resolve(args.runs_root or selection["runs_root"])
    if args.runs_root and runs_root != _resolve(selection["runs_root"]):
        raise ValueError("--runs-root differs from the selection lock")
    main_members = [_member(runs_root, m["run_id"], m["weight"])
                    for m in selection["members"]]
    _check_shared(main_members, "main members")
    entities = [{
        "name": "main", "kind": "ensemble", "members": main_members,
        "threshold": float(selection["threshold"]),
    }]

    sensitivity = {}
    sensitivity_metas = []
    baseline_manifest = None
    if not args.fixture:
        for key in ("k4", "k6", "all22"):
            spec = selection["sensitivity"][key]
            weight = 1.0 / len(spec["members"])
            members = [_member(runs_root, rid, weight)
                       for rid in spec["members"]]
            sensitivity_metas.extend(members)
            sensitivity[key] = {
                "members": [m["run_id"] for m in members],
                "threshold": float(spec["threshold"]),
            }
            entities.append({
                "name": key, "kind": "sensitivity", "members": members,
                "threshold": float(spec["threshold"]),
            })
        if len(entities) != 4:
            raise RuntimeError("main + k4/k6/all22 entity construction failed")
        _reconcile_selection(selection, main_members + sensitivity_metas)

        for member in main_members:
            m0 = copy.deepcopy(member)
            m0["weight"] = 1.0
            entities.append({
                "name": f"M0_{member['run_id']}", "kind": "m0",
                "members": [m0], "threshold": member["inner_threshold"],
                "condition": "M0", "member_run_id": member["run_id"],
            })

        if not args.ablations_root:
            raise ValueError("production finalization requires --ablations-root")
        abl_root = _resolve(args.ablations_root)
        abl_metas = []
        for member in main_members:
            base = member["run_id"]
            for condition in CONDITIONS:
                run_id = f"abl_{condition}_{base}"
                meta = _member(abl_root, run_id, 1.0)
                _check_ablation(meta, condition, member)
                abl_metas.append(meta)
                entities.append({
                    "name": run_id, "kind": "ablation",
                    "members": [meta], "threshold": meta["inner_threshold"],
                    "condition": condition, "member_run_id": base,
                })
        _check_shared(main_members + sensitivity_metas + abl_metas,
                      "main + sensitivity + ablations")
        if len(abl_metas) != 40:
            raise RuntimeError(f"expected 40 ablation runs, got {len(abl_metas)}")
        if len(entities) != N_EXPECTED_ENTITIES:
            raise RuntimeError(f"expected {N_EXPECTED_ENTITIES} inference "
                               f"entities, got {len(entities)}")

        if not args.baselines:
            raise ValueError("production finalization requires --baselines")
        baseline_manifest_path = _resolve(args.baselines)
        baselines = _baseline_entries(baseline_manifest_path)
        baseline_manifest = {
            "path": portable_path(baseline_manifest_path, P2_ROOT),
            "type": "file" if baseline_manifest_path.is_file() else "directory",
            "sha256": (sha256_file(baseline_manifest_path)
                       if baseline_manifest_path.is_file() else
                       sha256_json({b["name"]: b["run_manifest_sha256"]
                                    for b in baselines})),
        }
        base_cfgs = [{"cfg_dict": yaml.safe_load(
            (_resolve(b["run_dir"]) / "config.yaml").read_text(encoding="utf-8"))}
            for b in baselines]
        _check_inner_rows(main_members + baselines, "baselines vs main sample")
        _check_data_assets(main_members + base_cfgs, "baselines")
    else:
        _reconcile_selection(selection, main_members)
        baselines = []

    cfg = load_cfg(str(runs_root / main_members[0]["run_id"] / "config.yaml"))
    proto = SplitProtocol.from_cfg(cfg)
    if tuple(proto.test_years) != tuple(selection["protocol_years"]["test_years"]):
        raise ValueError("selection lock / run config test years differ")
    dev = _development_reference(cfg)
    availability = dev.pop("frame_availability")
    eval_cfg = cfg.get("evaluation", {}) or {}
    holm_families = {"main": [b["name"] for b in baselines]}
    if not args.fixture:
        for member in main_members:
            base = member["run_id"]
            holm_families[f"M0_{base}"] = [
                f"abl_{condition}_{base}" for condition in CONDITIONS]

    lock = {
        "kind": "final_protocol_lock",
        "protocol": "holdout_masked_v1",
        "finalized": True,
        "fixture": bool(args.fixture),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_years": selection["protocol_years"],
        "registry": selection["registry"],
        "runs_root": portable_path(runs_root, P2_ROOT),
        "ablations_root": (portable_path(_resolve(args.ablations_root), P2_ROOT)
                           if args.ablations_root else None),
        "selection_lock": portable_path(Path(args.selection), P2_ROOT),
        "selection_sha256": selection["lock_sha256"],
        "n_arms": len(arm_names(reg)),
        "test_output_dir": portable_path(test_output_dir, P2_ROOT),
        "test_receipt": portable_path(test_receipt, P2_ROOT),
        "members": _public(main_members),
        "threshold": float(selection["threshold"]),
        "sensitivity": sensitivity,
        "entities": _public(entities),
        "baselines": _public(baselines),
        "baseline_manifest": baseline_manifest,
        "frame_availability": availability,
        "evaluation": {
            **dev,
            "bootstrap": int(eval_cfg.get("bootstrap", 2000)),
            "alpha": float(eval_cfg.get("alpha", 0.10)),
            "seed": int(eval_cfg.get("seed", 42)),
            "multiplicity": "holm_bonferroni",
            "multiplicity_family": "complete_per_primary_holm_families",
            "holm_families": holm_families,
            "primary_contrast": "C2_vs_M0_BSS_dev",
            "other_comparisons": "exploratory",
            "bss_primary": "BSS_dev (fixed 2000-2019 rate)",
            "bss_secondary": "BSS_sample (external comparability)",
        },
    }
    lock["lock_sha256"] = sha256_json(lock)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print(f"final lock: {out}")
    print(f"  entities={len(entities)}  baselines={len(baselines)}  "
          f"fixture={bool(args.fixture)}")
    print(f"  dev_rate={dev['dev_rate']:.4f} ({dev['n_dev_rows']} rows)  "
          f"test rows={dev['n_test_rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
