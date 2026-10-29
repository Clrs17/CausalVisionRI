# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""HOLDOUT_MASKED_V1 evaluation from keyed test predictions.

Reports BOTH BSS conventions side by side:

  * ``BSS_dev``    — fixed 2000-2019 development RI rate (primary);
  * ``BSS_sample`` — the evaluated sample's own RI rate (external
                     comparability only).

Primary and comparison predictions must carry keys, the same final-lock hash
and a hash-verified manifest sidecar.  The final lock's test receipt must be
completed with the same lock hash.  Bootstrap/alpha/seed come only from the
lock.  Year filters apply to both sides before alignment.  Comparisons must
exactly cover the primary's complete Holm family declared in the final lock.

Usage (from the release root):
    python scripts/05_eval/eval_holdout.py --config configs/phase2/holdout_masked_v1.yaml \
        --pred data/test/holdout_masked_v1/main.npz \
        --lock protocol_locks/final.json \
        --compare data/test/holdout_masked_v1/baseline_lda.npz --out /tmp/eval_main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.evaluate import (bss, holm_adjust, metrics_row,  # noqa: E402
                          paired_cluster_bootstrap, reliability_table)
from wpac_warning.protocol import (SplitProtocol, accepted_lock_sha256,  # noqa: E402
                          sha256_array, sha256_file, sha256_json)
from wpac_warning.utils_cfg import load_cfg                      # noqa: E402

KEYS = ("sid", "iso_time", "storm_year", "row_uid", "y")


def _resolved(path_like) -> Path:
    p = Path(path_like)
    return p.resolve() if p.is_absolute() else (P2_ROOT / p).resolve()


def verify_completed_test_receipt(lock: dict) -> dict:
    """Require the one completed test receipt belonging to this exact lock."""
    if not lock.get("test_receipt") or not lock.get("test_output_dir"):
        raise ValueError("final lock lacks test_output_dir/test_receipt")
    path = _resolved(lock["test_receipt"])
    if not path.is_file():
        raise FileNotFoundError(
            f"completed test receipt is required before evaluation: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("protocol") != "holdout_masked_v1":
        raise ValueError(f"test receipt protocol mismatch: {path}")
    status = receipt.get("status")
    if status != "completed":
        raise RuntimeError(
            f"test receipt status={status!r}; evaluation requires completed: {path}")
    if receipt.get("lock_sha256") != lock.get("lock_sha256"):
        raise ValueError("test receipt lock hash differs from the final lock")
    if _resolved(receipt.get("test_output_dir", "")) != _resolved(
            lock["test_output_dir"]):
        raise ValueError("test receipt output directory differs from the final lock")
    return receipt


def load_pred(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"prediction file not found: {p}")
    with np.load(p, allow_pickle=True) as z:
        missing = [k for k in KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{p} is missing row keys {missing}; predictions must "
                             f"come from test_infer under the final lock")
        out = {k: z[k] for k in z.files}
    out["_path"] = str(p)
    out["_sha256"] = sha256_file(p)
    manifest_path = p.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"{p}: missing prediction manifest sidecar")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != out["_sha256"]:
        raise ValueError(f"{p}: output hash differs from its manifest sidecar")
    out["_manifest"] = manifest
    out["_manifest_path"] = str(manifest_path)
    return out


def align_comparison(primary_uid, compare_uid, *arrays):
    """Align comparison arrays to the primary row order without duplicating keys."""
    primary_uid = np.asarray(primary_uid).astype(str)
    compare_uid = np.asarray(compare_uid).astype(str)
    if len(primary_uid) != len(compare_uid):
        raise ValueError("comparison predictions have different row counts")
    if len(np.unique(primary_uid)) != len(primary_uid) or len(np.unique(compare_uid)) != len(compare_uid):
        raise ValueError("comparison row_uid values must be unique")
    if set(primary_uid) != set(compare_uid):
        raise ValueError("comparison row keys do not match the primary predictions")
    order = pd.Index(compare_uid).get_indexer(primary_uid)
    if len(order) != len(primary_uid) or (order < 0).any():
        raise ValueError("comparison row keys cannot be aligned")
    aligned = [np.asarray(arr)[order] for arr in arrays]
    return order, aligned


def _entity_threshold(lock: dict, pred: dict) -> tuple:
    """(entity_name, threshold) from the final lock; the npz must agree."""
    name = pred.get("entity")
    if name is None:
        raise ValueError("prediction npz has no entity name; it must come from "
                         "scripts/test_infer.py under the final lock")
    name = str(name)
    entries = {e["name"]: e for e in lock.get("entities", [])}
    entries.update({b["name"]: b for b in lock.get("baselines", [])})
    if name not in entries:
        raise ValueError(f"entity {name!r} is not registered in the final lock")
    thr = float(entries[name]["threshold"])
    if pred.get("threshold") is None or not np.isfinite(float(pred["threshold"])):
        raise ValueError(f"{name}: prediction threshold is missing or NaN/Inf")
    if abs(float(pred["threshold"]) - thr) > 1e-9:
        raise ValueError(f"{name}: prediction threshold differs from the lock")
    return name, thr


def _validate_pred(lock: dict, pred: dict) -> tuple:
    path = pred["_path"]
    accepted = accepted_lock_sha256(lock)
    if pred.get("lock_sha256") is None or \
            str(pred["lock_sha256"]) not in accepted:
        raise ValueError(f"{path}: prediction was not produced under this final lock")
    manifest = pred["_manifest"]
    if str(manifest.get("lock_sha256")) not in accepted:
        raise ValueError(f"{path}: sidecar belongs to a different final lock")
    entity_name, threshold = _entity_threshold(lock, pred)
    if manifest.get("entity") != entity_name:
        raise ValueError(f"{path}: sidecar entity differs from prediction")
    ruid = np.asarray(pred["row_uid"]).astype(str).ravel()
    n = len(ruid)
    if not n or len(np.unique(ruid)) != n:
        raise ValueError(f"{path}: row_uid must be non-empty and unique")
    for key in (*KEYS, "p"):
        if len(np.asarray(pred[key]).ravel()) != n:
            raise ValueError(f"{path}: inconsistent prediction/key row counts")
    y = np.asarray(pred["y"], dtype=float).ravel()
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{path}: labels must be finite and binary")
    p = np.asarray(pred["p"], dtype=float).ravel()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{path}: probabilities must be finite and within [0,1]")
    evaluation = lock.get("evaluation", {})
    test_hash = sha256_array(sorted(ruid.tolist()))
    if test_hash != evaluation.get("test_row_key_sha256"):
        raise ValueError(f"{path}: prediction rows are not the frozen test rows")
    if int(evaluation.get("n_test_rows", -1)) != n:
        raise ValueError(f"{path}: prediction row count differs from final lock")
    if int(manifest.get("n_rows", -1)) != n:
        raise ValueError(f"{path}: sidecar row count differs from prediction")
    if manifest.get("row_key_sha256") != sha256_json(ruid.tolist()):
        raise ValueError(f"{path}: sidecar row-key hash differs from prediction")
    return entity_name, threshold


def _year_subset(pred: dict, year_min, year_max) -> dict:
    keep = np.ones(len(np.asarray(pred["y"]).ravel()), dtype=bool)
    years = np.asarray(pred["storm_year"]).astype(int).ravel()
    if year_min is not None:
        keep &= years >= year_min
    if year_max is not None:
        keep &= years <= year_max
    return {
        "y": np.asarray(pred["y"]).astype(int).ravel()[keep],
        "p": np.asarray(pred["p"], dtype=float).ravel()[keep],
        "sy": years[keep],
        "ruid": np.asarray(pred["row_uid"]).astype(str).ravel()[keep],
        "sid": np.asarray(pred["sid"]).astype(str).ravel()[keep],
        "iso": np.asarray(pred["iso_time"]).astype(str).ravel()[keep],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True,
                    help="output prefix (writes <out>.csv, <out>.json)")
    ap.add_argument("--lock", required=True,
                    help="final protocol lock (the frozen development rate)")
    ap.add_argument("--compare", action="append", default=[],
                    help="comparison prediction npz (repeatable); paired "
                         "storm-cluster bootstrap vs --pred")
    ap.add_argument("--dev-rate", type=float, default=None,
                    help="forbidden: the development rate is frozen in the lock")
    ap.add_argument("--allow-fixture", action="store_true")
    ap.add_argument("--year-min", type=int, default=None)
    ap.add_argument("--year-max", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=None,
                    help="forbidden: frozen in the final lock")
    ap.add_argument("--alpha", type=float, default=None,
                    help="forbidden: frozen in the final lock")
    ap.add_argument("--seed", type=int, default=None,
                    help="forbidden: frozen in the final lock")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    proto = SplitProtocol.from_cfg(cfg)
    if args.dev_rate is not None:
        raise ValueError("--dev-rate cannot override the frozen development rate")
    if any(value is not None for value in
           (args.bootstrap, args.alpha, args.seed)):
        raise ValueError("--bootstrap/--alpha/--seed are read-only from the final lock")
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    if sha256_json(payload) != lock.get("lock_sha256"):
        raise ValueError("protocol lock hash mismatch")
    if lock.get("kind") != "final_protocol_lock" or not lock.get("finalized"):
        raise ValueError("primary evaluation requires the FINAL protocol lock")
    if lock.get("fixture") and not args.allow_fixture:
        raise ValueError("fixture lock requires explicit --allow-fixture")
    if tuple(lock["protocol_years"]["test_years"]) != tuple(proto.test_years):
        raise ValueError("lock test years != config test years")
    test_receipt = verify_completed_test_receipt(lock)
    dev = dict(lock.get("evaluation", {}))
    if "dev_rate" not in dev or "dev_row_key_sha256" not in dev:
        raise ValueError("final lock lacks the frozen development reference")
    if dev.get("multiplicity") != "holm_bonferroni":
        raise ValueError("final lock has an unsupported multiplicity policy")
    bootstrap = int(dev["bootstrap"])
    alpha = float(dev["alpha"])
    seed = int(dev["seed"])
    if bootstrap <= 0 or not 0 < alpha < 1:
        raise ValueError("final lock has invalid bootstrap/alpha policy")

    pred = load_pred(args.pred)
    entity_name, thr = _validate_pred(lock, pred)
    dev_rate = float(dev["dev_rate"])
    if not np.isfinite(dev_rate):
        raise ValueError("final lock development rate is NaN/Inf")
    primary = _year_subset(pred, args.year_min, args.year_max)
    y, p, sy = primary["y"], primary["p"], primary["sy"]
    ruid, sid = primary["ruid"], primary["sid"]
    if len(np.unique(y)) < 2:
        raise SystemExit("evaluated subset has a single class; BSS undefined")

    rows = []
    main_row = metrics_row(p, y, thr, dev_rate)
    main_row.update({"subset": "all", "pred": Path(args.pred).stem})
    rows.append(main_row)
    for yr in sorted(np.unique(sy)):
        sel = sy == yr
        if not sel.any():
            continue
        r = metrics_row(p[sel], y[sel], thr, dev_rate)
        r.update({"subset": f"year{yr}", "pred": Path(args.pred).stem})
        rows.append(r)
    table = pd.DataFrame(rows)

    out = Path(args.out)
    report = {
        "protocol": proto.as_dict(),
        "pred": str(_resolved(args.pred)),
        "pred_sha256": pred["_sha256"],
        "lock": str(_resolved(args.lock)),
        "lock_sha256": lock["lock_sha256"],
        "test_receipt": str(_resolved(lock["test_receipt"])),
        "test_receipt_sha256": sha256_file(_resolved(lock["test_receipt"])),
        "test_receipt_status": test_receipt["status"],
        "entity": entity_name,
        "threshold": thr,
        "threshold_source": "final_lock",
        "dev_rate": dev,
        "multiplicity": dev["multiplicity"],
        "alpha": alpha,
        "statistical_policy": {"bootstrap": bootstrap, "alpha": alpha,
                               "seed": seed, "source": "final_lock"},
        "year_filter": {"min": args.year_min, "max": args.year_max},
        "n_rows": int(len(y)),
        "base_rate": float(y.mean()),
        "metrics": table.set_index("subset").to_dict(orient="index"),
        "reliability": reliability_table(p, y),
        "paired_bootstrap": {},
    }

    comparison_inputs = []
    compare_names = []
    for cmp_path in args.compare:
        cmp = load_pred(cmp_path)
        compare_entity, _ = _validate_pred(lock, cmp)
        if compare_entity in compare_names:
            raise ValueError(f"duplicate comparison entity in Holm family: "
                             f"{compare_entity}")
        compare_names.append(compare_entity)
        comparison_inputs.append((cmp_path, cmp, compare_entity))
    if args.compare:
        families = dev.get("holm_families")
        if not isinstance(families, dict) or entity_name not in families:
            raise ValueError(f"final lock has no declared Holm family for {entity_name}")
        expected_family = list(families[entity_name])
        if len(expected_family) != len(set(expected_family)):
            raise ValueError("final lock declares a duplicate Holm family member")
        if (len(compare_names) != len(expected_family)
                or set(compare_names) != set(expected_family)):
            raise ValueError(f"comparison family does not match final lock for "
                             f"{entity_name}: expected {sorted(expected_family)}, "
                             f"got {sorted(compare_names)}")

    for cmp_path, cmp, compare_entity in comparison_inputs:
        comparison = _year_subset(cmp, args.year_min, args.year_max)
        cy, cp, csy = comparison["y"], comparison["p"], comparison["sy"]
        cruid, csid, ciso = (comparison["ruid"], comparison["sid"],
                             comparison["iso"])
        if not np.array_equal(cruid, ruid):
            _, aligned = align_comparison(ruid, cruid, cy, cp, csid, ciso, csy)
            cy, cp, csid, ciso, csy = aligned
        if not np.array_equal(cy, y):
            raise ValueError(f"{cmp_path}: labels differ from the primary "
                             f"predictions")
        if not (np.array_equal(csid, sid) and np.array_equal(csy, sy)
                and np.array_equal(ciso, primary["iso"])):
            raise ValueError(f"{cmp_path}: aligned row keys differ from primary")
        stats = paired_cluster_bootstrap(
            y, p, cp, csid, bss, n_boot=bootstrap, seed=seed,
            alpha=alpha, fixed_rate=dev_rate)
        stats["compare"] = str(_resolved(cmp_path))
        stats["compare_entity"] = compare_entity
        stats["compare_sha256"] = cmp["_sha256"]
        report["paired_bootstrap"][compare_entity] = stats
        print(f"ΔBSS_dev vs {compare_entity}: {stats['delta']:+.4f} "
              f"[{stats['ci_lo']:+.4f}, {stats['ci_hi']:+.4f}] "
              f"({stats['n_clusters']} storm clusters)")

    if report["paired_bootstrap"]:
        pvals = {k: v["p_boot"] for k, v in report["paired_bootstrap"].items()}
        if not np.isfinite(np.asarray(list(pvals.values()), dtype=float)).all():
            raise ValueError("Holm family contains NaN/Inf p-values")
        adjusted = holm_adjust(pvals)
        for name, adj in adjusted.items():
            stats = report["paired_bootstrap"][name]
            stats["p_holm"] = float(adj)
            stats["reject_holm"] = bool(adj <= alpha)
        report["multiplicity_applied"] = {
            "method": "holm_bonferroni",
            "primary": entity_name,
            "family": sorted(adjusted),
            "declared_by_final_lock": True,
            "alpha": alpha,
            "n_comparisons": len(adjusted),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float),
        encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
