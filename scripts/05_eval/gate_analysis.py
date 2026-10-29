# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
#!/usr/bin/env python
"""Causal-branch participation (gate) analysis over final-lock test outputs.

Reads the per-entity prediction npz files produced by the single test_infer
run (they carry ``gate_mean_members`` / ``gate_std_members`` /
``causal_share_members`` for every model entity) and summarises how strongly
each member and each ablation arm engaged the causal prior branch on the
2020-2024 test set.

For every entity/member it reports:
  * gate_mean statistics (overall, per year, RI vs non-RI) and gate_std
  * causal_attn_share statistics (head-averaged attention on the causal token)
  * Spearman correlation between gate_mean and the predicted probability
  * a descriptive RI-vs-non-RI separation of gate_mean (mean delta + its
    descriptive AUC; explicitly not a model)

Ablation entities are additionally pivoted against their own member's M0
(main-arm) statistics so the causal participation shift per condition is
directly readable.

Usage (from the release root):
    python scripts/05_eval/gate_analysis.py \
        --lock protocol_locks/final.json \
        --out data/analysis/gate_participation.json \
        --csv data/analysis/gate_participation.csv
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

from wpac_warning.evaluate import auc_metrics  # noqa: E402
from wpac_warning.protocol import accepted_lock_sha256  # noqa: E402


def _resolved(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q).resolve()


def _stats(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "std": float(np.std(x)),
            "p05": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95))}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _member_block(y: np.ndarray, years: np.ndarray, p: np.ndarray,
                  g: np.ndarray, gs: np.ndarray, cs: np.ndarray) -> dict:
    out = {"gate_mean": _stats(g), "gate_dim_std_mean": float(np.mean(gs)),
           "causal_attn_share": _stats(cs),
           "causal_attn_share_minus_0p5":
               float(np.mean(cs) - 0.5),
           "causal_influence_gate_x_share": float(np.mean(g * cs)),
           "spearman_gate_vs_prob": _spearman(g, p),
           "spearman_share_vs_prob": _spearman(cs, p),
           "by_year": {}}
    ri = y.astype(int) == 1
    for flag, mask in (("ri", ri), ("non_ri", ~ri)):
        if mask.sum():
            out[f"gate_mean_{flag}"] = float(np.mean(g[mask]))
            out[f"causal_share_{flag}"] = float(np.mean(cs[mask]))
    if ri.sum() and (~ri).sum():
        out["gate_mean_ri_delta"] = float(np.mean(g[ri]) - np.mean(g[~ri]))
        out["gate_mean_descriptive_auc"] = auc_metrics(g, y)["ROC_AUC"]
        out["causal_share_descriptive_auc"] = auc_metrics(cs, y)["ROC_AUC"]
    for yr in sorted(set(int(v) for v in years)):
        mask = years == yr
        if mask.sum():
            out["by_year"][str(yr)] = {
                "n": int(mask.sum()),
                "gate_mean": float(np.mean(g[mask])),
                "causal_attn_share": float(np.mean(cs[mask]))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    lock_path = Path(args.lock).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in lock.items() if k != "lock_sha256"}
    from wpac_warning.protocol import sha256_json
    if sha256_json(payload) != lock.get("lock_sha256"):
        raise SystemExit("final lock hash mismatch")
    entities = lock.get("entities", [])
    test_dir = _resolved(P2_ROOT, lock["test_output_dir"])

    rows = []
    skipped = []
    report = {"kind": "gate_participation_analysis",
              "lock": str(lock_path), "lock_sha256": lock.get("lock_sha256"),
              "entities": {}, "skipped": skipped}
    for entity in entities:
        name = entity["name"]
        path = test_dir / f"{name}.npz"
        if not path.is_file():
            skipped.append({"entity": name, "reason": "npz missing"})
            continue
        with np.load(path, allow_pickle=True) as z:
            if "gate_mean_members" not in z.files:
                skipped.append({"entity": name,
                                "reason": "no gate arrays (baseline or "
                                          "pre-gate-dump output)"})
                continue
            if str(z["lock_sha256"]) not in accepted_lock_sha256(lock):
                raise SystemExit(f"{path}: output was produced under a "
                                 f"different lock")
            y = z["y"].astype(int)
            years = (z["storm_year"].astype(int)
                     if "storm_year" in z.files
                     else np.zeros(len(y), dtype=int))
            p_mat = z["p_members"]
            g_mat = z["gate_mean_members"]
            gs_mat = z["gate_std_members"]
            cs_mat = z["causal_share_members"]
            member_names = [str(m) for m in z["member_names"]]
            weights = z["weights"].astype(float)
        n_rows, n_members = len(y), len(member_names)
        for arr, an in ((p_mat, "p_members"), (g_mat, "gate_mean_members"),
                        (gs_mat, "gate_std_members"),
                        (cs_mat, "causal_share_members")):
            if arr.shape != (n_rows, n_members):
                raise SystemExit(f"{path}: {an} shape {arr.shape} != "
                                 f"({n_rows}, {n_members})")
        if len(weights) != n_members:
            raise SystemExit(f"{path}: weights length mismatch")
        kind = entity.get("kind", "?")
        condition = entity.get("condition") or "M0"
        member_block = {}
        for j, mn in enumerate(member_names):
            block = _member_block(y, years, p_mat[:, j], g_mat[:, j],
                                  gs_mat[:, j], cs_mat[:, j])
            member_block[mn] = block
            rows.append({
                "entity": name, "kind": kind, "condition": condition,
                "member": mn, "weight": float(weights[j]),
                "n_rows": int(len(y)),
                "gate_mean": block["gate_mean"]["mean"],
                "gate_mean_ri": block.get("gate_mean_ri"),
                "gate_mean_non_ri": block.get("gate_mean_non_ri"),
                "gate_mean_ri_delta": block.get("gate_mean_ri_delta"),
                "gate_descriptive_auc": block.get("gate_mean_descriptive_auc"),
                "causal_attn_share": block["causal_attn_share"]["mean"],
                "causal_attn_share_minus_0p5":
                    block["causal_attn_share_minus_0p5"],
                "causal_influence_gate_x_share":
                    block["causal_influence_gate_x_share"],
                "causal_share_descriptive_auc":
                    block.get("causal_share_descriptive_auc"),
                "spearman_gate_vs_prob": block["spearman_gate_vs_prob"]})
        report["entities"][name] = {
            "kind": kind, "condition": condition, "n_rows": int(len(y)),
            "members": member_block}

    if not report["entities"]:
        raise SystemExit("no model entities with gate arrays found; the "
                         "analysis ran on an empty set")

    # ablation pivot: delta of each condition vs the member's own M0
    m0 = report["entities"].get("main", {}).get("members", {})
    member_of = {e["name"]: e.get("member_run_id") for e in entities}
    pivot = []
    for name, ent in report["entities"].items():
        if ent["kind"] != "ablation":
            continue
        cond = ent["condition"]
        base_name = member_of.get(name)
        member_block = (ent["members"].get(base_name or "")
                        or next(iter(ent["members"].values()), None))
        if member_block is None or base_name not in m0:
            continue
        mn = base_name
        base = m0[base_name]
        pivot.append({
            "member": mn, "condition": cond,
            "gate_mean": member_block["gate_mean"]["mean"],
            "gate_mean_delta_vs_m0": (member_block["gate_mean"]["mean"]
                                      - base["gate_mean"]["mean"]),
            "causal_attn_share": member_block["causal_attn_share"]["mean"],
            "causal_attn_share_delta_vs_m0":
                (member_block["causal_attn_share"]["mean"]
                 - base["causal_attn_share"]["mean"])})
    report["ablation_pivot"] = pivot

    # condition-level aggregation: C3 shuffle seeds pooled per member
    c3 = {}
    for row in pivot:
        if not row["condition"].startswith("C3s"):
            continue
        c3.setdefault(row["member"], []).append(row)
    report["c3_seed_aggregate"] = [
        {"member": mn,
         "n_seeds": len(rs),
         "gate_mean_delta_vs_m0_mean": float(
             np.mean([r["gate_mean_delta_vs_m0"] for r in rs])),
         "gate_mean_delta_vs_m0_range": [
             float(min(r["gate_mean_delta_vs_m0"] for r in rs)),
             float(max(r["gate_mean_delta_vs_m0"] for r in rs))],
         "causal_attn_share_delta_vs_m0_mean": float(
             np.mean([r["causal_attn_share_delta_vs_m0"] for r in rs]))}
        for mn, rs in sorted(c3.items())]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True),
                        encoding="utf-8")
    print(f"gate analysis -> {out_path}")
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"gate table   -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
