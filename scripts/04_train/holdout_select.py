# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""HOLDOUT_MASKED_V1 ensemble selection on the two inner years (2018, 2019).

Runs BEFORE any test inference.  It reads only ``preds/inner.npz`` files, so
test rows cannot influence membership, K, weights or the threshold.

Rules (defined in docs/PROTOCOL.md §3):

* every arm that completed the inner-only run participates — there is no
  ``max_arms`` truncation and no NNLS;
* per-arm score = two-year worst-year BSS (each inner year against its own
  climatology), called ``two_year_worst`` (NOT LOIO);
* ranking ties break deterministically on: inner mean BSS, worst year, the
  pre-specified family order, then arm name;
* final rule = quality-gated stratified family champion (gate = global top-8),
  uniform weights, K=5 primary; K=4/6 and all-22 uniform are sensitivity;
* the decision threshold is F1-tuned on the selected members' blended inner
  probability and frozen into the protocol lock.

Output: an intermediate selection lock (+ a CSV table) with a per-arm artifact
catalog and the protocol-fixed test output/receipt paths.  This lock alone
cannot unlock test inference.

Usage (from wpac_causal_phase2/):
    python scripts/04_train/holdout_select.py --runs-root data/runs/holdout_masked_v1 \
        --out protocol_locks/selection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.arms import (DEFAULT_REGISTRY, arm_family as registry_family,
                      arm_names, arm_overrides, load_registry)  # noqa: E402
from wpac_warning.evaluate import bss, f1_threshold          # noqa: E402
from wpac_warning.protocol import (canonical_data_assets, check_test_receipt_available,  # noqa: E402
                          protocol_test_receipt, resolve_protocol_test_output_dir,
                          sha256_file, sha256_json)
from wpac_warning.run_integrity import load_completed_marker  # noqa: E402

INNER_YEARS = (2018, 2019)
FAMILY_ORDER = ["lingbot_sf", "lingbot_mf", "mae_sf", "mae_mf", "other"]
FAMILY_GATE = 8
K_MAIN = 5
K_SENSITIVITY = (4, 6)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(P2_ROOT))
    except ValueError:
        return str(Path(p).resolve())


def arm_family(arm: str) -> str:
    is_lingbot = arm.startswith(("ens_lingbot", "ens_lb", "rit_lingbot", "rit_lb"))
    is_mae = arm.startswith(("ens_ep40", "ens_v1", "rit_ep40", "rit_v1"))
    mf = re.search(r"_temp\d", arm) is not None
    if is_lingbot:
        return "lingbot_mf" if mf else "lingbot_sf"
    if is_mae:
        return "mae_mf" if mf else "mae_sf"
    return "other"


def discover_arms(root: Path) -> dict:
    arms = {}
    for npz in sorted(root.glob("*/preds/inner.npz")):
        run_dir = npz.parent.parent
        load_completed_marker(run_dir, verify_hashes=False)
        arm = run_dir.name
        arms[arm] = {"run_dir": run_dir, "npz": npz}
    return arms


def _cfg_value(cfg: dict, dotted: str):
    value = cfg
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def validate_arm_config(arm: str, cfg: dict, reg: dict,
                        allow_fixture: bool = False) -> None:
    is_mini_fixture = (allow_fixture and cfg.get("model", {}).get(
        "vision", {}).get("backend") == "mini_cnn")
    expected = ({"train.run_id": arm} if is_mini_fixture
                else arm_overrides(reg, arm))
    mismatches = {}
    for key, want in expected.items():
        try:
            got = _cfg_value(cfg, key)
        except KeyError:
            got = "<missing>"
        if got != want:
            mismatches[key] = {"expected": want, "actual": got}
    if mismatches:
        raise ValueError(f"{arm}: config does not match registry overrides: "
                         f"{json.dumps(mismatches, sort_keys=True, default=str)}")
    shuffle_seed = int(_cfg_value(cfg, "model.gnn.prior_shuffle_seed")
                       if _contains_key(cfg, "model.gnn.prior_shuffle_seed") else 0)
    if shuffle_seed != 0:
        raise ValueError(f"{arm}: main arms require prior_shuffle_seed=0, "
                         f"got {shuffle_seed}")


def _contains_key(cfg: dict, dotted: str) -> bool:
    try:
        _cfg_value(cfg, dotted)
        return True
    except KeyError:
        return False


def _data_assets(cfg: dict) -> dict:
    return canonical_data_assets(cfg, P2_ROOT)


def load_arm(entry: dict) -> dict:
    with np.load(entry["npz"], allow_pickle=True) as z:
        required = ("p_inner", "y_inner", "sid", "iso_time", "storm_year", "row_uid")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"{entry['npz']} missing row-key fields {missing}; "
                             f"re-run the inner-only training with the holdout "
                             f"trainer")
        out = {k: z[k] for k in required}
        out["threshold"] = (float(z["threshold"]) if "threshold" in z.files
                            else np.nan)
    p = np.asarray(out["p_inner"], dtype=float)
    if not np.isfinite(p).all():
        raise ValueError(f"{entry['npz']}: inner probabilities contain NaN/Inf")
    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"{entry['npz']}: inner probabilities outside [0,1]")
    y = np.asarray(out["y_inner"], dtype=float)
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{entry['npz']}: inner labels must be finite and binary")
    if not np.isfinite(out["threshold"]):
        raise ValueError(f"{entry['npz']}: inner threshold is NaN/Inf")
    out["run_dir"] = entry["run_dir"]
    out["completed"] = load_completed_marker(entry["run_dir"])
    out["run_manifest"] = json.loads(
        (entry["run_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
    if out["run_manifest"].get("inner_only") is not True:
        raise ValueError(f"{entry['run_dir']}: run_manifest.inner_only must be true")
    cfg_path = entry["run_dir"] / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{entry['run_dir']}: missing config.yaml")
    out["cfg_dict"] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out["config_sha256"] = sha256_file(cfg_path)
    out["config_json_sha256"] = sha256_json(out["cfg_dict"])
    recorded_cfg = out["run_manifest"].get("config_sha256")
    if recorded_cfg != out["config_json_sha256"]:
        raise ValueError(f"{entry['run_dir']}: run manifest config_sha256 does "
                         f"not match resolved config.yaml")
    out["data_assets"] = _data_assets(out["cfg_dict"])
    out["npz_sha256"] = sha256_file(entry["npz"])
    ckpt = entry["run_dir"] / "checkpoints" / "best_ri.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"{entry['run_dir']}: missing checkpoints/best_ri.pt — the arm "
            f"cannot be re-inferred on test and must not enter selection")
    out["checkpoint_sha256"] = sha256_file(ckpt)
    out["checkpoint"] = str(ckpt)
    return out


def align(arms: dict) -> tuple:
    """All arms must carry the same inner rows in the same order."""
    ref_name = sorted(arms)[0]
    ref = arms[ref_name]
    ref_keys = np.asarray(ref["row_uid"]).astype(str)
    for name, a in arms.items():
        keys = np.asarray(a["row_uid"]).astype(str)
        if len(keys) != len(ref_keys) or not np.array_equal(keys, ref_keys):
            raise ValueError(
                f"{name}: inner row keys differ from {ref_name} — arms must "
                f"share an identical inner split")
        if not np.array_equal(np.asarray(a["y_inner"]).astype(int),
                              np.asarray(ref["y_inner"]).astype(int)):
            raise ValueError(f"{name}: inner labels differ from {ref_name}")
    return np.asarray(ref["y_inner"]).astype(int), ref_keys


def arm_scores(a: dict) -> dict:
    y = np.asarray(a["y_inner"]).astype(int)
    p = np.asarray(a["p_inner"], dtype=float)
    years = np.asarray(a["storm_year"]).astype(int)
    per_year = {}
    for yr in INNER_YEARS:
        sel = years == yr
        if not sel.any():
            raise ValueError(f"inner year {yr} has no rows")
        if len(np.unique(y[sel])) < 2:
            raise ValueError(f"inner year {yr} has a single class; the "
                             f"two-year worst-year score is undefined")
        per_year[int(yr)] = float(bss(p[sel], y[sel]))
    scores = {
        "two_year_worst": float(min(per_year.values())),
        "inner_mean": float(np.mean(list(per_year.values()))),
        "pooled": float(bss(p, y)),
        "per_year": {str(k): v for k, v in per_year.items()},
    }
    values = list(per_year.values()) + [scores[key] for key in
                                       ("two_year_worst", "inner_mean", "pooled")]
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError("inner selection scores contain NaN/Inf")
    return scores


def rank_arms(table: dict) -> list:
    def key(arm):
        s = table[arm]
        fam = FAMILY_ORDER.index(s["family"]) if s["family"] in FAMILY_ORDER else 99
        return (-s["two_year_worst"], -s["inner_mean"], fam, arm)
    return sorted(table, key=key)


def family_select(ranking: list, families: dict, k: int,
                  gate: int = FAMILY_GATE) -> tuple:
    """Quality-gated stratified family champion + global fill."""
    chosen, seen, gated = [], set(), []
    for i, arm in enumerate(ranking, start=1):
        fam = families[arm]
        if fam in FAMILY_ORDER[:-1] and fam not in seen:
            seen.add(fam)
            if i <= gate:
                chosen.append(arm)
            else:
                gated.append(fam)
    chosen = chosen[:k]
    for arm in ranking:
        if len(chosen) >= k:
            break
        if arm not in chosen:
            chosen.append(arm)
    return chosen[:k], gated


def blend(arms: dict, members: list, years=None) -> tuple:
    p = np.mean([np.asarray(arms[m]["p_inner"], dtype=float) for m in members],
                axis=0)
    if not np.isfinite(p).all():
        raise ValueError("blended inner probabilities contain NaN/Inf")
    y = np.asarray(arms[members[0]]["y_inner"]).astype(int)
    sy = np.asarray(arms[members[0]]["storm_year"]).astype(int)
    if years is not None:
        sel = np.isin(sy, list(years))
        return p[sel], y[sel]
    return p, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True,
                    help="directory of <arm>/preds/inner.npz run dirs")
    ap.add_argument("--out", required=True, help="selection lock JSON path")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                    help="frozen arm registry; selection requires ALL arms")
    ap.add_argument("--csv", default=None,
                    help="selection table CSV (default: <out>.csv)")
    ap.add_argument("--k", type=int, default=K_MAIN)
    ap.add_argument("--gate", type=int, default=FAMILY_GATE)
    ap.add_argument("--expected-arms", type=int, default=None,
                    help="compatibility override; must equal the registry "
                         "size when given")
    ap.add_argument("--test-output-dir",
                    default=str(P2_ROOT / "data" / "test" / "holdout_masked_v1"),
                    help="production-fixed; custom paths require --allow-fixture")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing lock (test must not have run)")
    ap.add_argument("--allow-fixture", action="store_true",
                    help="synthetic mini_cnn smoke only; recorded in the lock")
    args = ap.parse_args()

    if not args.allow_fixture and (args.k != K_MAIN or args.gate != FAMILY_GATE):
        raise SystemExit(
            f"production selection is fixed at k={K_MAIN}, gate={FAMILY_GATE}; "
            f"K4/K6/all22 are available only through the sensitivity catalog")
    if not args.allow_fixture:
        canonical = resolve_protocol_test_output_dir()
        check_test_receipt_available(protocol_test_receipt(canonical),
                                     block_running=True)
    test_output_dir = resolve_protocol_test_output_dir(
        args.test_output_dir, allow_fixture=args.allow_fixture)
    test_receipt = protocol_test_receipt(test_output_dir)

    root = Path(args.runs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"runs root not found: {root}")
    entries = discover_arms(root)
    reg = load_registry(args.registry)
    expected = set(arm_names(reg))
    found = set(entries)
    if args.expected_arms is not None and args.expected_arms != len(reg["arms"]):
        raise SystemExit(f"--expected-arms {args.expected_arms} != registry "
                         f"size {len(reg['arms'])}")
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise SystemExit(
            f"completed arms do not match the frozen registry: "
            f"{len(found)} found / {len(expected)} registered; "
            f"missing={missing}; unregistered={extra}")
    if len(entries) < 2:
        raise SystemExit(f"need at least 2 completed arms, found {len(entries)}")
    arms = {}
    for name, entry in entries.items():
        arms[name] = load_arm(entry)
        validate_arm_config(name, arms[name]["cfg_dict"], reg,
                            allow_fixture=args.allow_fixture)
    y, keys = align(arms)
    print(f"arms discovered: {len(arms)}")
    print("  " + ", ".join(sorted(arms)))

    table = {}
    for name in sorted(arms):
        s = arm_scores(arms[name])
        s["family"] = registry_family(reg, name)
        s["run_id"] = name
        s["run_dir"] = _rel(arms[name]["run_dir"])
        s["npz_sha256"] = arms[name]["npz_sha256"]
        s["checkpoint_sha256"] = arms[name]["checkpoint_sha256"]
        s["config_sha256"] = arms[name]["config_sha256"]
        s["config_json_sha256"] = arms[name]["config_json_sha256"]
        s["inner_threshold"] = arms[name]["threshold"]
        table[name] = s
    protocols = {name: a["run_manifest"].get("protocol", {}) for name, a in arms.items()}
    ref_proto = protocols[sorted(protocols)[0]]
    for name, proto in protocols.items():
        if proto != ref_proto:
            raise ValueError(f"{name}: run protocol differs from "
                             f"{sorted(protocols)[0]} — refusing to mix runs")
    priors = {name: a["run_manifest"].get("prior_manifest_sha256")
              for name, a in arms.items()}
    if None in priors.values():
        raise ValueError("a run manifest is missing prior_manifest_sha256")
    if len(set(priors.values())) != 1:
        raise ValueError(
            "arms were trained under different prior manifests: "
            f"{ {k: v[:12] for k, v in priors.items()} } — selection would "
            f"average models with different causal priors")
    for key in ("fit_years", "inner_years", "test_years"):
        if key not in ref_proto:
            raise ValueError(f"run manifest protocol missing {key}")
    families = {a: registry_family(reg, a) for a in table}
    for a in table:
        table[a]["family"] = families[a]
    ranking = rank_arms(table)

    members, gated = family_select(ranking, families, args.k, args.gate)
    p_sel, y_sel = blend(arms, members)
    threshold = f1_threshold(p_sel, y_sel)
    members_bss = bss(p_sel, y_sel)

    sensitivity = {}
    for k in K_SENSITIVITY:
        ms, _ = family_select(ranking, families, k, args.gate)
        p_k, y_k = blend(arms, ms)
        sensitivity[f"k{k}"] = {
            "members": ms, "threshold": f1_threshold(p_k, y_k),
            "pooled_inner_bss": bss(p_k, y_k),
        }
    p_all, y_all = blend(arms, ranking)
    sensitivity["all22"] = {
        "members": ranking, "threshold": f1_threshold(p_all, y_all),
        "pooled_inner_bss": bss(p_all, y_all),
    }
    score_values = [threshold, members_bss]
    for spec in sensitivity.values():
        score_values.extend([spec["threshold"], spec["pooled_inner_bss"]])
    if not np.isfinite(np.asarray(score_values, dtype=float)).all():
        raise ValueError("selected/sensitivity thresholds or scores contain NaN/Inf")

    arm_artifacts = {
        name: {
            "run_id": name,
            "run_dir": table[name]["run_dir"],
            "npz_sha256": arms[name]["npz_sha256"],
            "checkpoint_sha256": arms[name]["checkpoint_sha256"],
            "config_sha256": arms[name]["config_sha256"],
            "config_json_sha256": arms[name]["config_json_sha256"],
            "prior_manifest_sha256": arms[name]["run_manifest"].get(
                "prior_manifest_sha256"),
            "inner_threshold": arms[name]["threshold"],
            "data_assets": arms[name]["data_assets"],
        }
        for name in sorted(arms)
    }
    historical_duplicates = []
    if {"ens_ep40", "ens_ep40_srv"}.issubset(found):
        historical_duplicates.append({
            "arms": ["ens_ep40", "ens_ep40_srv"],
            "reason": "identical technical configuration; historical re-run",
            "policy": "retained as separate registered arms; no deduplication",
        })

    lock = {
        "kind": "selection",
        "protocol": "holdout_masked_v1",
        "fixture": bool(args.allow_fixture),
        "lock_version": 3,
        "registry": {"path": _rel(Path(reg["_path"])),
                     "sha256": sha256_file(reg["_path"]),
                     "n_arms": len(reg["arms"]),
                     "family_order": reg["family_order"]},
        "runs_root": _rel(root),
        "test_output_dir": _rel(test_output_dir),
        "test_receipt": _rel(test_receipt),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_years": {
            "fit_years": ref_proto["fit_years"],
            "inner_years": ref_proto["inner_years"],
            "test_years": ref_proto["test_years"],
        },
        "selection_rule": {
            "score": "two_year_worst_year_bss",
            "inner_years": list(INNER_YEARS),
            "tie_break": ["inner_mean_bss_desc", "worst_year_bss",
                          "family_order", "arm_name"],
            "family_order": list(reg["family_order"]),
            "family_gate": int(args.gate),
            "k": int(args.k),
            "weights": "uniform",
            "max_arms_truncation": False,
            "nnls": False,
            "threshold": "f1_on_selected_blended_inner_probability",
            "n_arms": len(reg["arms"]),
        },
        "members": [
            {
                "run_id": m,
                "run_dir": table[m]["run_dir"],
                "family": table[m]["family"],
                "weight": 1.0 / len(members),
                "inner_threshold": table[m]["inner_threshold"],
                "npz_sha256": table[m]["npz_sha256"],
                "checkpoint_sha256": table[m]["checkpoint_sha256"],
                "config_sha256": arms[m]["config_sha256"],
                "config_json_sha256": arms[m]["config_json_sha256"],
                "prior_manifest_sha256": arms[m]["run_manifest"].get(
                    "prior_manifest_sha256"),
                "data_assets": arms[m]["data_assets"],
            }
            for m in members
        ],
        "arm_artifacts": arm_artifacts,
        "historical_duplicates": historical_duplicates,
        "threshold": float(threshold),
        "inner_pooled_bss_selected": float(members_bss),
        "gated_families": gated,
        "sensitivity": sensitivity,
        "ranking": [
            {
                "rank": i + 1, "arm": a, "family": table[a]["family"],
                "two_year_worst": table[a]["two_year_worst"],
                "inner_mean": table[a]["inner_mean"],
                "pooled_inner_bss": table[a]["pooled"],
                "per_year": table[a]["per_year"],
            }
            for i, a in enumerate(ranking)
        ],
    }
    payload_for_hash = {k: v for k, v in lock.items() if k != "lock_sha256"}
    lock["lock_sha256"] = sha256_json(payload_for_hash)

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(
            f"protocol lock already exists: {out} — refusing to re-lock "
            f"(test inference may already have run). Use --force only if no "
            f"test prediction exists.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")
    pd.DataFrame(lock["ranking"]).to_csv(csv_path, index=False)
    print(f"\nselected K={args.k}: {members}")
    print(f"threshold={threshold:.3f}  pooled inner BSS={members_bss:.4f}")
    print(f"lock written: {out}\n  sha256={lock['lock_sha256'][:16]}…")
    print(f"ranking table: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
