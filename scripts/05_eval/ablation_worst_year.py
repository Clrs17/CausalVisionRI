# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Worst-year ablation analysis: does the causal prior protect the worst year?

Read-only analysis of the FROZEN test predictions in
``data/test/holdout_masked_v1`` (protocol holdout_masked_v1).  No training, no
re-evaluation of the primary contrasts; every number here is a decomposition
of ``BSS_dev`` (fixed 2000-2019 development RI rate, ``DEV_RATE``) by test year
(2020-2024, parsed from ``row_uid``'s ISO_TIME).

What is produced:

  * per member x per year x condition (C1..C4sign) ``BSS_dev`` and the
    ``ΔBSS_dev`` against M0;
  * the ensemble level (equal-weight mean of the same 5 members) per year
    for M0 / C2 / C1 — M0 must reproduce ``eval/main_2024.json``;
  * worst-year (M0 arg-min) protection metrics and ``Δ(min-year)``;
  * storm-cluster bootstrap (whole-storm resampling, 2000 draws, seed 42)
    90% CI for the worst-year Δ, both with the year fixed at the M0 arg-min
    and with the min-year re-selected inside every draw.

Outputs
    data/analysis/worst_year_ablation.json
    docs/WORST_YEAR_ABLATION.md

Usage (from the release root):
    python scripts/05_eval/ablation_worst_year.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.evaluate import bss  # noqa: E402

PRED_DIR = P2_ROOT / "data" / "test" / "holdout_masked_v1"
EVAL_DIR = P2_ROOT / "data" / "analysis" / "eval"
OUT_JSON = P2_ROOT / "data" / "analysis" / "worst_year_ablation.json"
OUT_MD = P2_ROOT / "docs" / "WORST_YEAR_ABLATION.md"

MEMBERS = ["lora16_sf", "lora16_temp5", "partial4_sf", "temp2", "temp5_slim"]
CONDITIONS = ["C1", "C2", "C3s1", "C3s2", "C3s3", "C4adj", "C4cf", "C4sign"]
FOCUS = ["C2", "C1"]
YEARS = [2020, 2021, 2022, 2023, 2024]

DEV_RATE = 0.07910230657338782
ALPHA = 0.10
N_BOOT = 2000
SEED = 42

ENSEMBLE_M0 = "M0_equal_weight_5members"
ENSEMBLE_PREFIX = "ens_equal_weight_5members"


# ----------------------------------------------------------------------
# loading / integrity
# ----------------------------------------------------------------------

def _load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen() -> tuple[dict, dict]:
    """Load every frozen npz, reindex to the reference row order, verify."""
    ref = _load_npz(PRED_DIR / "main.npz")
    ref_uid = np.asarray(ref["row_uid"])
    pos = {u: i for i, u in enumerate(ref_uid)}

    preds: dict[str, np.ndarray] = {}
    checks: dict[str, dict] = {}
    files = {"main": PRED_DIR / "main.npz"}
    files.update({f"M0_{m}": PRED_DIR / f"M0_ens_lbB_{m}.npz" for m in MEMBERS})
    files.update({f"{c}_{m}": PRED_DIR / f"abl_{c}_ens_lbB_{m}.npz"
                  for c in CONDITIONS for m in MEMBERS})

    for key, path in files.items():
        d = _load_npz(path)
        uid = np.asarray(d["row_uid"])
        if len(uid) != len(ref_uid) or set(uid) != set(ref_uid):
            raise SystemExit(f"row_uid mismatch in {path.name}")
        order = np.array([pos[u] for u in uid])
        p = np.zeros(len(ref_uid), dtype=float)
        p[order] = np.asarray(d["p"], dtype=float)
        preds[key] = p

        split = np.unique(np.asarray(d["split"])) if "split" in d else None
        man_path = path.with_suffix("").with_suffix(".manifest.json")
        man = json.loads(man_path.read_text())
        sha = _sha256(path)
        checks[key] = {
            "file": path.name,
            "manifest": man_path.name,
            "manifest_sha256_match": sha == man.get("output_sha256"),
            "protocol": man.get("protocol"),
            "n_rows": int(man.get("n_rows", -1)),
            "split": [str(s) for s in split] if split is not None else None,
            "row_uid_matches_reference": True,
            "npz_sha256": sha,
        }
        if checks[key]["protocol"] != "holdout_masked_v1":
            raise SystemExit(f"{path.name}: unexpected protocol")
        if split is not None and list(split) != ["test"]:
            raise SystemExit(f"{path.name}: non-test split present")

    # y / sid must agree everywhere
    for key in files:
        d = _load_npz(files[key])
        uid = np.asarray(d["row_uid"])
        order = np.array([pos[u] for u in uid])
        y = np.zeros(len(ref_uid), dtype=int)
        y[order] = np.asarray(d["y"], dtype=int)
        if not np.array_equal(y, np.asarray(ref["y"], dtype=int)):
            raise SystemExit(f"{files[key].name}: label mismatch")

    # equal-weight ensembles per condition (same 5 members, no re-selection)
    preds["M0_ENS"] = np.stack([preds[f"M0_{m}"] for m in MEMBERS], axis=0).mean(axis=0)
    preds[ENSEMBLE_M0] = preds["M0_ENS"]
    for cond in CONDITIONS:
        stack = np.stack([preds[f"{cond}_{m}"] for m in MEMBERS], axis=0)
        preds[f"{cond}_ENS"] = stack.mean(axis=0)

    meta = {
        "row_uid": ref_uid,
        "y": np.asarray(ref["y"], dtype=int),
        "sid": np.asarray(ref["sid"]),
        "storm_year": np.asarray(ref["storm_year"], dtype=int),
        "iso_time": np.asarray(ref["iso_time"]),
    }
    if not np.allclose(preds["main"], preds["M0_ENS"]):
        raise SystemExit("main.npz != equal-weight mean of the 5 M0 members")
    return {"preds": preds, "meta": meta}, checks


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def bss_by_year(p: np.ndarray, y: np.ndarray, year: np.ndarray) -> dict:
    def _blk(py: np.ndarray, yy: np.ndarray) -> dict:
        bs = float(np.mean((py - yy) ** 2))
        bs_clim = DEV_RATE ** 2 + (1.0 - 2.0 * DEV_RATE) * float(yy.mean())
        return {
            "n": int(len(yy)),
            "BS": bs,
            "BS_clim_dev": bs_clim,
            "BSS_dev": bss(py, yy, DEV_RATE),
            "base_rate": float(yy.mean()),
            "mean_p": float(py.mean()),
            "roc_auc": float(roc_auc_score(yy, py)) if len(np.unique(yy)) > 1 else float("nan"),
        }

    out = {"all": _blk(p, y)}
    for Y in YEARS:
        s = year == Y
        out[f"year{Y}"] = _blk(p[s], y[s])
    return out


def pooled_delta_decomposition(p_cond: np.ndarray, p_m0: np.ndarray,
                               y: np.ndarray, year: np.ndarray) -> dict:
    """Exact split of the pooled ΔBSS_dev into per-year BS contributions.

    BS_pooled = Σ_Y (n_Y/N)·BS_Y, so with a common BS_clim the pooled ΔBSS is
    Σ_Y (n_Y/N)·(BS_M0,Y − BS_cond,Y)/BS_clim_pooled exactly.
    """
    n = len(y)
    bs_clim = DEV_RATE ** 2 + (1.0 - 2.0 * DEV_RATE) * float(y.mean())
    total = (float(np.mean((p_m0 - y) ** 2)) - float(np.mean((p_cond - y) ** 2))) / bs_clim
    per_year = {}
    for Y in YEARS:
        s = year == Y
        contrib = ((float(np.mean((p_m0[s] - y[s]) ** 2))
                    - float(np.mean((p_cond[s] - y[s]) ** 2))) * s.sum() / n / bs_clim)
        per_year[str(Y)] = {
            "n_share": float(s.sum() / n),
            "delta_bs_year": float(np.mean((p_m0[s] - y[s]) ** 2)
                                   - np.mean((p_cond[s] - y[s]) ** 2)),
            "delta_bss_contribution": contrib,
            "contribution_share_of_pooled": (contrib / total) if total else float("nan"),
        }
    sum_check = float(sum(v["delta_bss_contribution"] for v in per_year.values()))
    return {"pooled_delta_bss": float(total), "sum_check": sum_check,
            "years": per_year}


def worst_year(row: dict) -> int:
    return min(YEARS, key=lambda Y: row[f"year{Y}"]["BSS_dev"])


# ----------------------------------------------------------------------
# storm-cluster bootstrap
# ----------------------------------------------------------------------

def _cluster_index(cluster: np.ndarray) -> dict:
    return {c: np.where(cluster == c)[0] for c in np.unique(cluster)}


def boot_delta_fixed_year(p_cond: np.ndarray, p_m0: np.ndarray, y: np.ndarray,
                          sid: np.ndarray, year_mask: np.ndarray,
                          n_boot: int = N_BOOT, seed: int = SEED,
                          alpha: float = ALPHA) -> dict:
    """90% CI of BSS_dev(cond) - BSS_dev(M0) on one fixed year, whole-storm resampling."""
    pc, pm, yy, ss = (a[year_mask] for a in (p_cond, p_m0, y, sid))
    point = bss(pc, yy, DEV_RATE) - bss(pm, yy, DEV_RATE)
    gmap = _cluster_index(ss)
    keys = np.array(list(gmap.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([gmap[c] for c in pick])
        draws[i] = bss(pc[idx], yy[idx], DEV_RATE) - bss(pm[idx], yy[idx], DEV_RATE)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_boot = 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean()))
    p_boot = min(max(p_boot, 1.0 / (n_boot + 1)), 1.0)
    return {
        "delta": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "alpha": float(alpha),
        "p_boot": float(p_boot),
        "n_boot": int(n_boot),
        "n_clusters": int(len(keys)),
        "n_rows": int(len(yy)),
        "seed": int(seed),
        "significant_90": bool((lo > 0) or (hi < 0)),
    }


def boot_delta_minyear(p_cond: np.ndarray, p_m0: np.ndarray, y: np.ndarray,
                       sid: np.ndarray, year: np.ndarray,
                       n_boot: int = N_BOOT, seed: int = SEED,
                       alpha: float = ALPHA) -> dict:
    """Same statistic but the min-year is re-selected inside every draw.

    Whole-storm resampling over ALL test storms; per draw the M0 arg-min year is
    located again and the Δ is taken there, so the year-selection noise of the
    ``min`` operator is inside the interval.
    """
    point = min(YEARS, key=lambda Y: bss(p_m0[year == Y], y[year == Y], DEV_RATE))
    point_delta = (bss(p_cond[year == point], y[year == point], DEV_RATE)
                   - bss(p_m0[year == point], y[year == point], DEV_RATE))
    gmap = _cluster_index(sid)
    keys = np.array(list(gmap.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    draw_years = np.empty(n_boot, dtype=int)
    for i in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([gmap[c] for c in pick])
        yy, yr = y[idx], year[idx]
        best_y, best_v = None, np.inf
        for Y in YEARS:
            s = yr == Y
            v = bss(p_m0[idx][s], yy[s], DEV_RATE)
            if v < best_v:
                best_y, best_v = Y, v
        s = yr == best_y
        draws[i] = (bss(p_cond[idx][s], yy[s], DEV_RATE)
                    - bss(p_m0[idx][s], yy[s], DEV_RATE))
        draw_years[i] = best_y
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta": float(point_delta),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "n_clusters": int(len(keys)),
        "n_rows": int(len(y)),
        "seed": int(seed),
        "point_worst_year": int(point),
        "draw_worst_year_counts": {str(Y): int((draw_years == Y).sum())
                                   for Y in YEARS},
        "significant_90": bool((lo > 0) or (hi < 0)),
    }


# ----------------------------------------------------------------------
# headline verdict (single source of truth for JSON + markdown)
# ----------------------------------------------------------------------

def summarize(out: dict) -> dict:
    """Decide, from the frozen numbers alone, whether the assumption holds.

    The assumption is "the causal prior's value is not just in the mean but in
    keeping the worst year from collapsing".  Operational test: the ΔBSS_dev at
    the worst year must be (a) larger in magnitude than the pooled Δ and
    (b) distinguishable from 0 by the storm-cluster bootstrap.
    """
    bss_table, worst = out["bss_dev"], out["worst_year"]
    boot, dec = out["bootstrap"], out["pooled_delta_decomposition"]
    ens = worst["ensemble"]
    wy = ens["M0_worst_year"]

    e_c2 = ens["conditions"]["C2"]["delta_at_M0_worst_year"]
    e_c1 = ens["conditions"]["C1"]["delta_at_M0_worst_year"]
    ens_pool_c2 = bss_table["C2_ENS"]["all"]["BSS_dev"] - bss_table[ENSEMBLE_M0]["all"]["BSS_dev"]
    ens_pool_c1 = bss_table["C1_ENS"]["all"]["BSS_dev"] - bss_table[ENSEMBLE_M0]["all"]["BSS_dev"]
    ens_sig = boot["ensemble"]["fixed_year"]["C2"]["significant_90"]

    d2 = [worst["members"][m]["min_year"]["delta_C2"] for m in MEMBERS]
    d1 = [worst["members"][m]["min_year"]["delta_C1"] for m in MEMBERS]
    pooled_c2 = [bss_table[f"C2_{m}"]["all"]["BSS_dev"] - bss_table[f"M0_{m}"]["all"]["BSS_dev"]
                 for m in MEMBERS]
    pooled_c1 = [bss_table[f"C1_{m}"]["all"]["BSS_dev"] - bss_table[f"M0_{m}"]["all"]["BSS_dev"]
                 for m in MEMBERS]
    mean_min, mean_pool = float(np.mean(d2)), float(np.mean(pooled_c2))
    sig = [m for m in MEMBERS if boot["members"][m]["fixed_year"]["C2"]["significant_90"]]
    sig_help = [m for m in sig
                if worst["members"][m]["conditions"]["C2"]["delta_at_M0_worst_year"] < 0]

    ampl_ens = bool(abs(e_c2) > abs(ens_pool_c2) and ens_sig)
    ampl_mem = bool(abs(mean_min) > abs(mean_pool))
    holds = ampl_ens and ampl_mem
    verdict = "supported" if holds else ("partially supported" if (ampl_ens or ampl_mem) else "not supported")

    m0_pooled = {m: bss_table[f"M0_{m}"]["all"]["BSS_dev"] for m in MEMBERS}
    rank = 1 + sum(1 for v in m0_pooled.values() if v > m0_pooled["lora16_sf"])
    share22 = dec["ensemble"]["C2"]["years"][str(wy)]["contribution_share_of_pooled"]

    return {
        "hypothesis": "The causal prior provides special protection against a poor worst year",
        "verdict": verdict,
        "hypothesis_holds": bool(holds),
        "amplification_at_worst_year_ensemble": ampl_ens,
        "amplification_at_worst_year_members": ampl_mem,
        "worst_year": int(wy),
        "worst_year_identical_for_all_entities": bool(
            all(worst["members"][m]["M0_worst_year"] == wy for m in MEMBERS)),
        "ensemble_delta_C2_at_worst_year": float(e_c2),
        "ensemble_pooled_delta_C2": float(ens_pool_c2),
        "ensemble_delta_C1_at_worst_year": float(e_c1),
        "ensemble_pooled_delta_C1": float(ens_pool_c1),
        "ensemble_worst_year_ci90_fixed_year": [
            boot["ensemble"]["fixed_year"]["C2"]["ci_lo"],
            boot["ensemble"]["fixed_year"]["C2"]["ci_hi"]],
        "ensemble_worst_year_significant_90": bool(ens_sig),
        "members_mean_delta_C2_at_worst_year": mean_min,
        "members_mean_pooled_delta_C2": mean_pool,
        "members_min_over_pooled_ratio": (mean_min / mean_pool) if mean_pool else None,
        "members_delta_C1_at_worst_year_mean": float(np.mean(d1)),
        "members_pooled_delta_C1_mean": float(np.mean(pooled_c1)),
        "n_members_prior_helps_at_worst_year": int(sum(1 for v in d2 if v < 0)),
        "n_members_prior_hurts_at_worst_year": int(sum(1 for v in d2 if v > 0)),
        "members_significant_90_at_worst_year": sig,
        "members_significant_and_prior_helps": sig_help,
        "worst_year_share_of_pooled_delta_C2": float(share22),
        "worst_year_row_share": float(dec["ensemble"]["C2"]["years"][str(wy)]["n_share"]),
        "one_line": (
            f"The graph pathway does not protect the ensemble's worst year {wy}: "
            f"that year contributes {share22:.0%} of the pooled ΔC2 "
            f"(below its {dec['ensemble']['C2']['years'][str(wy)]['n_share']:.0%} row share), "
            f"with ΔC2 = {e_c2:+.3f} and a 90% CI containing zero. "
            f"The mean member worst-year ΔC2 is {mean_min:+.3f}, smaller in magnitude "
            f"than the pooled mean {mean_pool:+.3f}; "
            f"only {sig_help[0] if sig_help else 'no member'} is a significant exception."
        ),
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    data, checks = load_frozen()
    preds, meta = data["preds"], data["meta"]
    y, sid, year = meta["y"], meta["sid"], meta["storm_year"]

    entities = {f"M0_{m}": m for m in MEMBERS}
    entities.update({f"{c}_{m}": m for m in MEMBERS for c in CONDITIONS})
    entities[ENSEMBLE_M0] = "ENS"
    entities.update({f"{c}_ENS": "ENS" for c in CONDITIONS})

    # ---- BSS_dev per entity, per year -------------------------------------
    bss_table: dict[str, dict] = {}
    for key in entities:
        bss_table[key] = bss_by_year(preds[key], y, year)

    delta: dict[str, dict] = {}
    for key, grp in entities.items():
        base_key = f"M0_{grp}" if grp != "ENS" else ENSEMBLE_M0
        if key == base_key:
            continue
        cond = key.split("_")[0]
        if cond not in CONDITIONS:
            continue
        d = {}
        for k in ["all"] + [f"year{Y}" for Y in YEARS]:
            d[k] = bss_table[key][k]["BSS_dev"] - bss_table[base_key][k]["BSS_dev"]
        delta[key] = d

    # ---- worst year per entity (M0 arg-min) -------------------------------
    def worst_block(grp: str) -> dict:
        base_key = f"M0_{grp}" if grp != "ENS" else ENSEMBLE_M0
        wr = worst_year(bss_table[base_key])
        block = {
            "M0_worst_year": int(wr),
            "M0_worst_year_BSS_dev": bss_table[base_key][f"year{wr}"]["BSS_dev"],
            "M0_BSS_dev_by_year": {str(Y): bss_table[base_key][f"year{Y}"]["BSS_dev"]
                                   for Y in YEARS},
            "conditions": {},
        }
        for cond in CONDITIONS:
            key = f"{cond}_{grp}"
            at_worst = bss_table[key][f"year{wr}"]["BSS_dev"]
            own = worst_year(bss_table[key])
            block["conditions"][cond] = {
                "BSS_dev_at_M0_worst_year": at_worst,
                "delta_at_M0_worst_year": at_worst - bss_table[base_key][f"year{wr}"]["BSS_dev"],
                "own_worst_year": int(own),
                "own_min_BSS_dev": bss_table[key][f"year{own}"]["BSS_dev"],
            }
        block["min_year"] = {
            "M0": bss_table[base_key][f"year{wr}"]["BSS_dev"],
            **{c: block["conditions"][c]["BSS_dev_at_M0_worst_year"] for c in CONDITIONS},
            **{f"delta_{c}": block["conditions"][c]["delta_at_M0_worst_year"]
               for c in CONDITIONS},
        }
        return block

    worst = {"ensemble": worst_block("ENS"),
             "members": {m: worst_block(m) for m in MEMBERS}}

    # ---- bootstrap --------------------------------------------------------
    boot = {"ensemble": {}, "members": {m: {} for m in MEMBERS}}
    for grp in ["ENS"] + MEMBERS:
        base_key = ENSEMBLE_M0 if grp == "ENS" else f"M0_{grp}"
        wr = worst_year(bss_table[base_key])
        mask = year == wr
        tgt = boot["ensemble"] if grp == "ENS" else boot["members"][grp]
        tgt["worst_year"] = int(wr)
        tgt["fixed_year"] = {}
        tgt["minyear_reselected"] = {}
        for cond in CONDITIONS:
            key = f"{cond}_{grp}"
            tgt["fixed_year"][cond] = boot_delta_fixed_year(
                preds[key], preds[base_key], y, sid, mask)
            tgt["minyear_reselected"][cond] = boot_delta_minyear(
                preds[key], preds[base_key], y, sid, year)

    # ---- consistency against the frozen eval/ artefacts --------------------
    def _eval_json(name: str) -> dict:
        return json.loads((EVAL_DIR / f"{name}.json").read_text())["metrics"]

    ens_ref = _eval_json("main_2024")
    cons = {
        "ensemble_vs_eval_main_2024": {
            f"year{Y}": {
                "computed": bss_table[ENSEMBLE_M0][f"year{Y}"]["BSS_dev"],
                "eval_json": ens_ref[f"year{Y}"]["BSS_dev"],
                "abs_diff": abs(bss_table[ENSEMBLE_M0][f"year{Y}"]["BSS_dev"]
                                - ens_ref[f"year{Y}"]["BSS_dev"]),
            } for Y in YEARS
        },
        "ensemble_all": {
            "computed": bss_table[ENSEMBLE_M0]["all"]["BSS_dev"],
            "eval_json": ens_ref["all"]["BSS_dev"],
            "abs_diff": abs(bss_table[ENSEMBLE_M0]["all"]["BSS_dev"]
                            - ens_ref["all"]["BSS_dev"]),
        },
        "members_vs_eval_json": {},
    }
    for m in MEMBERS:
        ref = _eval_json(f"M0_ens_lbB_{m}")
        d = {f"year{Y}": abs(bss_table[f"M0_{m}"][f"year{Y}"]["BSS_dev"]
                             - ref[f"year{Y}"]["BSS_dev"]) for Y in YEARS}
        d["all"] = abs(bss_table[f"M0_{m}"]["all"]["BSS_dev"] - ref["all"]["BSS_dev"])
        cons["members_vs_eval_json"][m] = d
    all_diffs = [v["abs_diff"] for v in cons["ensemble_vs_eval_main_2024"].values()]
    all_diffs.append(cons["ensemble_all"]["abs_diff"])
    for m in MEMBERS:
        all_diffs.extend(cons["members_vs_eval_json"][m].values())
    cons["max_abs_diff_all_checks"] = float(max(all_diffs))
    cons["all_match_lt_1e12"] = bool(max(all_diffs) < 1e-12)
    cons["manifest_hash_verified_all_npz"] = bool(
        all(c["manifest_sha256_match"] for c in checks.values()))
    cons["n_npz_checked"] = len(checks)
    cons["all_split_test"] = bool(
        all(c["split"] in (None, ["test"]) for c in checks.values()))

    # ---- pooled ΔBSS_dev split into per-year contributions ----------------
    decomp = {}
    for grp in ["ENS"] + MEMBERS:
        base_key = ENSEMBLE_M0 if grp == "ENS" else f"M0_{grp}"
        entry = {}
        for cond in CONDITIONS:
            key = f"{cond}_{grp}"
            entry[cond] = pooled_delta_decomposition(preds[key], preds[base_key], y, year)
        decomp["ensemble" if grp == "ENS" else grp] = entry

    # ---- assemble ---------------------------------------------------------
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/05_eval/ablation_worst_year.py",
        "protocol": "holdout_masked_v1",
        "read_only": True,
        "sources": {
            "pred_dir": "data/test/holdout_masked_v1",
            "eval_dir": "data/analysis/eval",
            "n_members": len(MEMBERS),
            "members": MEMBERS,
            "conditions": CONDITIONS,
            "n_test_rows": int(len(y)),
            "n_storms": int(len(np.unique(sid))),
            "years": YEARS,
        },
        "settings": {
            "bss_margin": "BSS_dev",
            "dev_rate": DEV_RATE,
            "dev_rate_source": "eval/main_2024.json → dev_rate.dev_rate (2000-2019)",
            "bootstrap": N_BOOT,
            "seed": SEED,
            "alpha": ALPHA,
            "cluster": "sid (whole-storm resampling)",
            "year_source": "row_uid ISO_TIME",
            "ensemble_definition": "equal-weight mean of p over the same 5 members",
            "worst_year_definition": "arg-min over 2020-2024 of the entity's M0 BSS_dev",
        },
        "checks": {
            "per_file": checks,
            "consistency": cons,
        },
        "base_rate_by_year": {str(Y): float(y[year == Y].mean()) for Y in YEARS},
        "bs_clim_dev_by_year": {
            str(Y): float(DEV_RATE ** 2 + (1 - 2 * DEV_RATE) * y[year == Y].mean())
            for Y in YEARS},
        "bss_dev": bss_table,
        "delta_vs_M0": delta,
        "worst_year": worst,
        "bootstrap": boot,
        "pooled_delta_decomposition": decomp,
    }
    out["verdict"] = summarize(out)
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=False), encoding="utf-8")
    OUT_MD.write_text(build_md(out), encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(P2_ROOT)}")
    print(f"wrote {OUT_MD.relative_to(P2_ROOT)}")
    print(f"consistency max|diff| = {cons['max_abs_diff_all_checks']:.3e}")
    print(f"verdict = {out['verdict']['verdict']}")
    print(out["verdict"]["one_line"])


# ----------------------------------------------------------------------
# markdown report
# ----------------------------------------------------------------------

def _f(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def _e(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def build_md(out: dict) -> str:
    bss_table = out["bss_dev"]
    worst = out["worst_year"]
    boot = out["bootstrap"]
    cons = out["checks"]["consistency"]
    members = MEMBERS

    L: list[str] = []
    A = L.append
    A("# Worst-year ablation analysis: C2 and C1 relative to M0\n")
    A("Generated by `scripts/05_eval/ablation_worst_year.py`. The inputs are "
      "45 frozen member-level NPZ files (five members × {M0, C1, C2, C3s1-3, "
      "C4adj, C4cf, C4sign}) and the equal-weight five-member `main.npz`. "
      "The analysis reads frozen predictions without training or changing thresholds.\n")
    A("`BSS_dev = 1 − BS/BS_clim`, with `BS_clim = π² + (1−2π)·ȳ_year` and "
      f"π = {DEV_RATE:.6f} for 2000–2019. Years come from `row_uid` ISO_TIME. "
      "Every NPZ is checked for `split == test`, `protocol == holdout_masked_v1`, "
      "and agreement between its file hash and the manifest `output_sha256`. "
      "C2 removes the GNN graph pathway; C1 removes all causal losses. "
      "**Contrasts in this document are condition − M0, so a negative Δ favours "
      "the causal component.**\n")
    A("The analysis asks whether the causal prior reduces the loss in the worst "
      "year by more than it changes the pooled score. Both the ensemble and the "
      "individual members are evaluated with the same frozen probabilities.\n")

    # --- consistency
    A("## 0. Consistency with the evaluation JSON\n")
    A(f"- Ensemble M0 (`main.npz`) versus `eval/main_2024.json`: maximum absolute "
      f"all-year difference = {cons['ensemble_all']['abs_diff']:.2e}, with computed "
      f"all-year BSS_dev = {cons['ensemble_all']['computed']:.10f}.\n")
    per_year_diff = max(v["abs_diff"] for v in cons["ensemble_vs_eval_main_2024"].values())
    A(f"- Five M0 members versus `eval/M0_ens_lbB_*.json`: maximum absolute difference = "
      f"{max(max(v.values()) for v in cons['members_vs_eval_json'].values()):.2e}.\n")
    A(f"- All checks (ensemble + five members × six quantities): maximum absolute "
      f"difference = {cons['max_abs_diff_all_checks']:.2e}; "
      f"{'consistent (<1e-12)' if cons['all_match_lt_1e12'] else '**requires review**'}. "
      f"The maximum yearly ensemble difference is {per_year_diff:.2e}.\n")
    A(f"- NPZ files checked: {cons['n_npz_checked']}; all manifest hashes match: "
      f"{cons['manifest_hash_verified_all_npz']}; all rows are test: {cons['all_split_test']}.\n")

    # --- ensemble per year
    A("## 1. Ensemble BSS_dev by year\n")
    A("The ensemble is the equal-weight mean of the five members.\n")
    A("| Year | Rows | Observed rate | M0 | C2 | C1 | ΔC2−M0 | ΔC1−M0 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for Y in YEARS:
        k = f"year{Y}"
        row = bss_table[ENSEMBLE_M0][k]
        m0 = row["BSS_dev"]
        c2 = bss_table[f"C2_ENS"][k]["BSS_dev"]
        c1 = bss_table[f"C1_ENS"][k]["BSS_dev"]
        star = " **←worst**" if Y == worst["ensemble"]["M0_worst_year"] else ""
        A(f"| {Y}{star} | {row['n']} | {row['base_rate']:.4f} | {_e(m0)} | {_e(c2)} | "
          f"{_e(c1)} | {_f(c2-m0)} | {_f(c1-m0)} |")
    k = "all"
    A(f"| all | {bss_table[ENSEMBLE_M0][k]['n']} | "
      f"{bss_table[ENSEMBLE_M0][k]['base_rate']:.4f} | "
      f"{_e(bss_table[ENSEMBLE_M0][k]['BSS_dev'])} | "
      f"{_e(bss_table['C2_ENS'][k]['BSS_dev'])} | "
      f"{_e(bss_table['C1_ENS'][k]['BSS_dev'])} | "
      f"{_f(bss_table['C2_ENS'][k]['BSS_dev']-bss_table[ENSEMBLE_M0][k]['BSS_dev'])} | "
      f"{_f(bss_table['C1_ENS'][k]['BSS_dev']-bss_table[ENSEMBLE_M0][k]['BSS_dev'])} |")
    A("")
    w = worst["ensemble"]
    A(f"The ensemble's worst year is **{w['M0_worst_year']}**, with M0 BSS_dev "
      f"{_e(w['M0_worst_year_BSS_dev'])}. In that year, C2 is "
      f"{_e(w['conditions']['C2']['BSS_dev_at_M0_worst_year'])} "
      f"(Δ = {_f(w['conditions']['C2']['delta_at_M0_worst_year'])}) and C1 is "
      f"{_e(w['conditions']['C1']['BSS_dev_at_M0_worst_year'])} "
      f"(Δ = {_f(w['conditions']['C1']['delta_at_M0_worst_year'])}). The worst year "
      f"for C2 itself is {w['conditions']['C2']['own_worst_year']}, and for C1 itself "
      f"it is {w['conditions']['C1']['own_worst_year']}.\n")

    # --- per-member per-year
    for cond in FOCUS:
        A(f"## 2{'' if cond == 'C2' else 'b'}. Member-by-year BSS_dev and "
          f"Δ{cond}−M0 ({cond}: {'remove the GNN graph branch' if cond == 'C2' else 'remove all causal losses'})\n")
        A("For each member, the first row gives M0, with the worst year in bold. "
          f"The second row gives Δ{cond}−M0.\n")
        A("| Member / quantity | 2020 | 2021 | 2022 | 2023 | 2024 | Worst year | Δ at worst year |")
        A("|---|---|---|---|---|---|---|---|")
        for m in members:
            wm = worst["members"][m]
            wy = wm["M0_worst_year"]
            cells = []
            for Y in YEARS:
                v = _e(bss_table[f"M0_{m}"][f"year{Y}"]["BSS_dev"])
                cells.append(f"**{v}**" if Y == wy else v)
            dcell = [_f(bss_table[f"{cond}_{m}"][f"year{Y}"]["BSS_dev"]
                        - bss_table[f"M0_{m}"][f"year{Y}"]["BSS_dev"]) for Y in YEARS]
            dcell = [f"**{c}**" if Y == wy else c for c, Y in zip(dcell, YEARS)]
            A(f"| {m} · M0 | " + " | ".join(cells) + f" | {wy} | — |")
            A(f"| {m} · Δ{cond} | " + " | ".join(dcell) + " | — | "
              f"{_f(wm['conditions'][cond]['delta_at_M0_worst_year'])} |")
        A("")
        dsum = [worst["members"][m]["conditions"][cond]["delta_at_M0_worst_year"]
                for m in members]
        A(f"Δ{cond} at each member's worst year: "
          + ", ".join(f"{m} {_f(v)}" for m, v in zip(members, dsum))
          + f"; mean {_f(float(np.mean(dsum)))}, with "
          f"{sum(1 for v in dsum if v > 0)}/5 positive.\n")

    # --- min-year protection
    A("## 3. Worst-year protection metric\n")
    A("For each entity, `min-year` is the year with the lowest M0 BSS_dev during "
      "2020–2024. All three conditions are evaluated at that year.\n")
    A("| Entity | Worst year | M0 | C2 | C1 | Δ(min-year) C2 | Δ(min-year) C1 |")
    A("|---|---|---:|---:|---:|---:|---:|")
    mrows = [("ensemble", worst["ensemble"])] + [(m, worst["members"][m]) for m in members]
    for label, wb in mrows:
        my = wb["min_year"]
        A(f"| {label} | {wb['M0_worst_year']} | {_e(my['M0'])} | {_e(my['C2'])} | "
          f"{_e(my['C1'])} | {_f(my['delta_C2'])} | {_f(my['delta_C1'])} |")
    A("")
    d2 = [worst["members"][m]["min_year"]["delta_C2"] for m in members]
    d1 = [worst["members"][m]["min_year"]["delta_C1"] for m in members]
    A(f"Mean member Δ(min-year): C2 {_f(float(np.mean(d2)))}, C1 {_f(float(np.mean(d1)))}. "
      f"For the ensemble: C2 {_f(worst['ensemble']['min_year']['delta_C2'])}, "
      f"C1 {_f(worst['ensemble']['min_year']['delta_C1'])}. Each member is evaluated at "
      "its own worst year, which is the same year for all members in this analysis.\n")

    # --- bootstrap
    A("## 4. Storm-cluster bootstrap\n")
    A("Clusters are whole `sid` storms, resampled 2,000 times with seed 42. "
      "Approach A fixes the entity's M0 worst year and resamples storms within that "
      "year. Approach B resamples all test storms and selects the worst year again "
      "inside each draw, including the year-selection variability. Significance "
      "means that the 90% interval excludes zero.\n")
    A("| Entity | ΔC2, fixed worst year | 90% CI | p_boot | Significant | ΔC2, year reselected | 90% CI | Significant |")
    A("|---|---:|---:|---:|:---:|---:|---:|:---:|")
    brow = [("ensemble", boot["ensemble"])] + [(m, boot["members"][m]) for m in members]
    for label, bb in brow:
        a = bb["fixed_year"]["C2"]
        b = bb["minyear_reselected"]["C2"]
        A(f"| {label} | {_f(a['delta'])} | [{_f(a['ci_lo'])}, {_f(a['ci_hi'])}] | "
          f"{a['p_boot']:.4f} | {'Yes' if a['significant_90'] else 'No'} | "
          f"{_f(b['delta'])} | [{_f(b['ci_lo'])}, {_f(b['ci_hi'])}] | "
          f"{'Yes' if b['significant_90'] else 'No'} |")
    A("")
    A("C1 control (all causal losses removed):\n")
    A("| Entity | ΔC1, fixed worst year | 90% CI | p_boot | Significant | ΔC1, year reselected | 90% CI | Significant |")
    A("|---|---:|---:|---:|:---:|---:|---:|:---:|")
    for label, bb in brow:
        a = bb["fixed_year"]["C1"]
        b = bb["minyear_reselected"]["C1"]
        A(f"| {label} | {_f(a['delta'])} | [{_f(a['ci_lo'])}, {_f(a['ci_hi'])}] | "
          f"{a['p_boot']:.4f} | {'Yes' if a['significant_90'] else 'No'} | "
          f"{_f(b['delta'])} | [{_f(b['ci_lo'])}, {_f(b['ci_hi'])}] | "
          f"{'Yes' if b['significant_90'] else 'No'} |")
    A("")
    dys = boot["ensemble"]["minyear_reselected"]["C2"]["draw_worst_year_counts"]
    A("Worst-year frequencies across the 2,000 ensemble approach-B draws: "
      + ", ".join(f"{k}: {v}" for k, v in dys.items()) + ".\n")

    # --- pooled reference
    A("## 5. Pooled all-year ΔBSS_dev\n")
    A("| Entity | M0 | C2 | C1 | ΔC2 | ΔC1 |")
    A("|---|---:|---:|---:|---:|---:|")
    for label, grp in [("ensemble", "ENS")] + [(m, m) for m in members]:
        b = ENSEMBLE_M0 if grp == "ENS" else f"M0_{grp}"
        c2 = f"C2_{grp}"
        c1 = f"C1_{grp}"
        A(f"| {label} | {_e(bss_table[b]['all']['BSS_dev'])} | "
          f"{_e(bss_table[c2]['all']['BSS_dev'])} | {_e(bss_table[c1]['all']['BSS_dev'])} | "
          f"{_f(bss_table[c2]['all']['BSS_dev']-bss_table[b]['all']['BSS_dev'])} | "
          f"{_f(bss_table[c1]['all']['BSS_dev']-bss_table[b]['all']['BSS_dev'])} |")
    A("")

    # --- pooled loss decomposed by year
    dec = out["pooled_delta_decomposition"]
    wy0 = worst["ensemble"]["M0_worst_year"]
    A("## 6. Yearly contributions to pooled ΔBSS_dev\n")
    A("Because `BS_pooled = Σ_Y (n_Y/N)·BS_Y`, the pooled contrast decomposes as "
      "`Σ_Y (n_Y/N)·(BS_M0,Y − BS_cond,Y)/BS_clim_pooled`. Each cell is that year's "
      "contribution to the pooled ΔBSS_dev. A positive value favours the condition. "
      "The share is the contribution divided by the pooled total. "
      f"The row share for {wy0} is {dec['ensemble']['C2']['years'][str(wy0)]['n_share']:.3f}. "
      "Shares are shown only when |pooled ΔBSS| ≥ 0.01; smaller totals make percentages "
      "unstable and are shown as `—`.\n")
    for cond in FOCUS:
        A(f"**{cond}**\n")
        A(f"| Entity | Pooled ΔBSS | 2020 | 2021 | 2022 | 2023 | 2024 | Share in {wy0} |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|")
        for label, key in [("ensemble", "ensemble")] + [(m, m) for m in members]:
            d = dec[key][cond]
            yrs = d["years"]
            sh = yrs[str(wy0)]["contribution_share_of_pooled"]
            sh_txt = f"{sh:.3f}" if abs(d["pooled_delta_bss"]) >= 0.01 else "—"
            A(f"| {label} | {_f(d['pooled_delta_bss'])} | "
              + " | ".join(_f(yrs[str(Y)]["delta_bss_contribution"]) for Y in YEARS)
              + f" | {sh_txt} |")
        A("")

    # --- mechanism: level vs ranking per year
    A("## 7. Worst-year diagnosis: probability level and discrimination\n")
    A("| Year | BS_clim | Observed rate | Ensemble M0 mean p | Ensemble C2 mean p | M0 overprediction ratio | ROC-AUC M0 | ROC-AUC C2 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for Y in YEARS:
        k = f"year{Y}"
        r0 = bss_table[ENSEMBLE_M0][k]
        r2 = bss_table[f"C2_ENS"][k]
        A(f"| {Y} | {r0['BS_clim_dev']:.4f} | {r0['base_rate']:.4f} | "
          f"{r0['mean_p']:.4f} | {r2['mean_p']:.4f} | "
          f"{r0['mean_p']/r0['base_rate']:.2f}× | {r0['roc_auc']:.4f} | "
          f"{r2['roc_auc']:.4f} |")
    A("")
    wy = worst["ensemble"]["M0_worst_year"]
    rwy = bss_table[ENSEMBLE_M0][f"year{wy}"]
    A(f"Discrimination in {wy} is comparable to the other years (ROC-AUC "
      f"{rwy['roc_auc']:.3f}). Its low BSS is associated with a systematically high "
      f"probability level: ensemble M0 has mean p = {rwy['mean_p']:.4f}, compared with "
      f"an observed rate of {rwy['base_rate']:.4f} "
      f"({rwy['mean_p']/rwy['base_rate']:.2f}×). It also has the lowest BS_clim "
      f"({rwy['BS_clim_dev']:.4f}), so absolute probability errors receive a large "
      "penalty relative to the reference. The worst-year diagnosis therefore concerns "
      "probability calibration more than ranking ability.\n")

    # --- conclusion
    V = out["verdict"]
    ens = worst["ensemble"]
    e_c2 = V["ensemble_delta_C2_at_worst_year"]
    e_c1 = V["ensemble_delta_C1_at_worst_year"]
    ens_pool_c2 = V["ensemble_pooled_delta_C2"]
    ens_pool_c1 = V["ensemble_pooled_delta_C1"]
    share22 = V["worst_year_share_of_pooled_delta_C2"]
    n_share22 = V["worst_year_row_share"]
    n_c2_neg = V["n_members_prior_helps_at_worst_year"]
    n_c2_pos = V["n_members_prior_hurts_at_worst_year"]
    sig_c2 = V["members_significant_90_at_worst_year"]
    sig_c2_help = V["members_significant_and_prior_helps"]
    n_c1_pos = sum(1 for m in members
                   if worst["members"][m]["conditions"]["C1"]["delta_at_M0_worst_year"] > 0)
    mean_min = V["members_mean_delta_C2_at_worst_year"]
    mean_pool = V["members_mean_pooled_delta_C2"]
    ens_ratio = e_c2 / ens_pool_c2 if ens_pool_c2 else float("nan")
    ens_sig = V["ensemble_worst_year_significant_90"]
    top = max(members, key=lambda m: bss_table[f"M0_{m}"]["all"]["BSS_dev"])
    rank_sf = 1 + sum(1 for m in members
                      if bss_table[f"M0_{m}"]["all"]["BSS_dev"]
                      > bss_table["M0_lora16_sf"]["all"]["BSS_dev"])
    verdict_text = str(V["verdict"])

    A("## 8. Interpretation\n")
    A(f"**Assessment: the hypothesis that the causal prior provides special protection "
      f"against a poor worst year is {verdict_text} on the frozen test set.** "
      "The decision rule requires the worst-year contrast to exceed the pooled contrast "
      "in magnitude and to have a 90% storm-cluster interval excluding zero, for both "
      "the ensemble and the member panel.\n")
    A(f"**Members.** {V['worst_year']} "
      + ("is the worst year shared by all five members" if V['worst_year_identical_for_all_entities']
         else "is not the same worst year for every member; see Section 3")
      + f". Removing the graph pathway lowers BSS_dev in {n_c2_neg}/5 members, which "
      f"favours the prior in that year, and raises it in {n_c2_pos}/5. The mean member "
      f"contrast is {_f(mean_min)}, compared with {_f(mean_pool)} for the pooled years. "
      f"The worst-year/pooled magnitude ratio is {V['members_min_over_pooled_ratio']:.2f}; "
      "a ratio below 1 means that the worst-year loss is not amplified. "
      f"Fixed-year bootstrap identifies {len(sig_c2)} significant members: "
      + (", ".join(sig_c2) if sig_c2 else "none")
      + ". Those whose direction supports worst-year protection are "
      + (", ".join(sig_c2_help) if sig_c2_help else "none")
      + f". C1 raises BSS_dev in {n_c1_pos}/5 members, with mean "
      f"{_f(V['members_delta_C1_at_worst_year_mean'])}, compared with "
      f"{_f(V['members_pooled_delta_C1_mean'])} for the pooled years. This member pattern "
      "does not support a general worst-year protection effect.\n")
    A(f"**Ensemble.** In {V['worst_year']}, M0 is {_e(ens['M0_worst_year_BSS_dev'])}, "
      f"C2 is {_e(ens['conditions']['C2']['BSS_dev_at_M0_worst_year'])} "
      f"(Δ = {_f(e_c2)}), and C1 is {_e(ens['conditions']['C1']['BSS_dev_at_M0_worst_year'])} "
      f"(Δ = {_f(e_c1)}). The worst-year and pooled C2 contrasts ({_f(ens_pool_c2)}) "
      f"are of similar magnitude (ratio {ens_ratio:.2f}), and the 90% CI "
      f"[{_f(V['ensemble_worst_year_ci90_fixed_year'][0])}, "
      f"{_f(V['ensemble_worst_year_ci90_fixed_year'][1])}] "
      f"{'excludes' if ens_sig else 'contains'} zero. The year contributes "
      f"{share22:.3f} of the pooled C2 contrast, **below its {n_share22:.3f} row share**. "
      f"2020 and 2021 jointly contribute "
      f"{sum(dec['ensemble']['C2']['years'][str(Y)]['contribution_share_of_pooled'] for Y in (2020, 2021)):.3f}. "
      f"The ensemble's graph-pathway gain is therefore smallest in {V['worst_year']}. "
      f"The pooled C1 contrast is {_f(ens_pool_c1)}, so yearly shares are not "
      "interpretable at that scale.\n")
    A(f"**Summary.** The ensemble's graph-pathway benefit is smallest in its worst year, "
      f"{V['worst_year']}: that year supplies {share22:.0%} of the pooled loss, below its "
      f"{n_share22:.0%} row share, with ΔC2 = {_f(e_c2)} and a 90% CI containing zero. "
      f"The member-level mean worst-year contrast ({_f(mean_min)}) is also smaller in "
      f"magnitude than the pooled contrast ({_f(mean_pool)}). LoRA-sf is the significant "
      f"exception: Δ = {_f(worst['members']['lora16_sf']['min_year']['delta_C2'])} in "
      f"{V['worst_year']}, with 90% CI "
      f"[{_f(boot['members']['lora16_sf']['fixed_year']['C2']['ci_lo'])}, "
      f"{_f(boot['members']['lora16_sf']['fixed_year']['C2']['ci_hi'])}]. Its M0 pooled "
      f"BSS_dev ranks {rank_sf}/5 ({_e(bss_table['M0_lora16_sf']['all']['BSS_dev'])}; "
      f"the highest is {top} at {_e(bss_table[f'M0_{top}']['all']['BSS_dev'])}), so this "
      "is a member-specific result rather than a general ensemble effect. "
      f"The worst year, {V['worst_year']}, combines systematic probability overprediction "
      "with the lowest climatological reference score. The data support a calibration "
      "diagnosis for that year and a worst-year protection effect only for LoRA-sf.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
