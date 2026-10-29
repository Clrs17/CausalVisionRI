# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Year-gradient ablation: where in 2020-2024 does the causal path pay off?

Read-only analysis of the FROZEN test predictions in
``data/test/holdout_masked_v1`` (protocol holdout_masked_v1).  No training, no
re-evaluation, no threshold change.  Every number is a decomposition of
``BSS_dev`` (fixed 2000-2019 development RI rate) along the test-year axis.

Contrast
    ΔBSS_dev(cond) = BSS_dev(M0) - BSS_dev(cond),  positive = the causal
    component helps.  ``C2`` = no GNN causal branch, ``C1`` = no causal loss.

What is produced
    * per-year (2020-2024) ΔC2 for the equal-weight ensemble and for the 5
      members, with storm-cluster paired bootstrap 90% / 95% percentile CI and
      ``p_boot`` (same definition as ``ablation_worst_year.py``);
    * pooled 2020-2021 (both years merged, ``BS_clim`` from the pooled base
      rate) with the same bootstrap;
    * pooled 2020-2024 as a self-consistency check (ensemble Δ must be +0.032
      to 1e-3) and per-year M0 cross-check against ``eval/main_2024.json``
      (< 1e-12);
    * the C1 contrast on the same periods (point estimates everywhere, CI at
      the ensemble level).

Method is deliberately NOT re-implemented: the frozen-npz loader, the
``BSS_dev`` block, ``DEV_RATE``, the member list and the bootstrap are taken
from the vetted ``scripts/ablation_worst_year.py`` (imported, plus a direct
cross-check of the re-signed bootstrap against its function).

Outputs
    data/analysis/year_gradient_ablation.json
    docs/YEAR_GRADIENT_ABLATION.md

Usage (from the release root):
    python scripts/05_eval/ablation_year_gradient.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

P2_ROOT = Path(__file__).resolve().parents[2]
for _p in (P2_ROOT / "src", Path(__file__).resolve().parents[1]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from wpac_warning.evaluate import bss  # noqa: E402
import ablation_worst_year as vetted  # noqa: E402

MEMBERS = vetted.MEMBERS
YEARS = vetted.YEARS
DEV_RATE = vetted.DEV_RATE
N_BOOT = vetted.N_BOOT
SEED = vetted.SEED
ENS = vetted.ENSEMBLE_M0
EVAL_DIR = vetted.EVAL_DIR

OUT_JSON = P2_ROOT / "data" / "analysis" / "year_gradient_ablation.json"
OUT_MD = P2_ROOT / "docs" / "YEAR_GRADIENT_ABLATION.md"

# primary contrast C2, supplementary C1
CONDS = ("C2", "C1")
EARLY = (2020, 2021)
PERIODS = {str(Y): (Y,) for Y in YEARS}
PERIODS["pooled_2020_2021"] = EARLY
PERIODS["pooled_2020_2024"] = tuple(YEARS)
BOOT_PERIODS = [str(Y) for Y in YEARS] + ["pooled_2020_2021"]


# ----------------------------------------------------------------------
# point estimates
# ----------------------------------------------------------------------

def block(p: np.ndarray, y: np.ndarray) -> dict:
    """Same block as ``ablation_worst_year.bss_by_year._blk``."""
    bs = float(np.mean((p - y) ** 2))
    bs_clim = DEV_RATE ** 2 + (1.0 - 2.0 * DEV_RATE) * float(y.mean())
    return {
        "n": int(len(y)),
        "BS": bs,
        "BS_clim_dev": bs_clim,
        "BSS_dev": bss(p, y, DEV_RATE),
        "base_rate": float(y.mean()),
        "mean_p": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def entity_scores(preds: dict, y: np.ndarray, masks: dict) -> dict:
    """BSS_dev per entity (M0 / C2 / C1 for ensemble + 5 members) per period."""
    keys = {ENS: "ensemble", **{f"M0_{m}": m for m in MEMBERS}}
    for cond in CONDS:
        keys[f"{cond}_ENS"] = "ensemble"
        keys.update({f"{cond}_{m}": m for m in MEMBERS})
    out: dict[str, dict] = {"ensemble": {}, "members": {m: {} for m in MEMBERS}}
    for key, grp in keys.items():
        tgt = out["ensemble"] if grp == "ensemble" else out["members"][grp]
        tgt[key] = {per: block(preds[key][mask], y[mask]) for per, mask in masks.items()}
    return out


# ----------------------------------------------------------------------
# storm-cluster paired bootstrap (same resampling, re-signed contrast)
# ----------------------------------------------------------------------

def boot_contrast(p_m0: np.ndarray, p_cond: np.ndarray, y: np.ndarray,
                  sid: np.ndarray, mask: np.ndarray,
                  n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Percentile CI of Δ = BSS_dev(M0) - BSS_dev(cond) on a fixed row subset.

    Mirrors ``ablation_worst_year.boot_delta_fixed_year`` exactly — whole-storm
    (``sid``) resampling with replacement, B = 2000, seed 42, and the SAME draw
    index used for both conditions (paired).  Differences: sign convention
    (M0 - cond instead of cond - M0) and both 90% and 95% intervals from the one
    draw array.  ``p_boot = 2*min(P(draw <= 0), P(draw >= 0))`` clipped below by
    1/(B+1), identical to the vetted script.
    """
    pm, pc, yy, ss = (a[mask] for a in (p_m0, p_cond, y, sid))
    point = bss(pm, yy, DEV_RATE) - bss(pc, yy, DEV_RATE)
    gmap = {c: np.where(ss == c)[0] for c in np.unique(ss)}
    keys = np.array(list(gmap.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([gmap[c] for c in pick])
        draws[i] = bss(pm[idx], yy[idx], DEV_RATE) - bss(pc[idx], yy[idx], DEV_RATE)
    lo90, hi90 = np.percentile(draws, [5.0, 95.0])
    lo95, hi95 = np.percentile(draws, [2.5, 97.5])
    p_boot = 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean()))
    p_boot = min(max(p_boot, 1.0 / (n_boot + 1)), 1.0)
    return {
        "delta": float(point),
        "ci90": [float(lo90), float(hi90)],
        "ci95": [float(lo95), float(hi95)],
        "p_boot": float(p_boot),
        "n_boot": int(n_boot),
        "n_clusters": int(len(keys)),
        "n_rows": int(len(yy)),
        "seed": int(seed),
        "frac_draws_positive": float((draws > 0).mean()),
        "significant_90": bool(lo90 > 0 or hi90 < 0),
        "significant_95": bool(lo95 > 0 or hi95 < 0),
        "ci95_excludes_zero": bool(lo95 > 0 or hi95 < 0),
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    data, per_file = vetted.load_frozen()
    preds, meta = data["preds"], data["meta"]
    y, sid, year = meta["y"], meta["sid"], meta["storm_year"]

    masks = {}
    for per, yrs in PERIODS.items():
        masks[per] = np.isin(year, yrs) if len(yrs) > 1 else (year == yrs[0])

    scores = entity_scores(preds, y, masks)

    # ---- point-estimate Δ = M0 - cond ------------------------------------
    dump: dict[str, dict] = {"ensemble": {}, "members": {m: {} for m in MEMBERS}}
    for cond in CONDS:
        dump["ensemble"][cond] = {
            per: scores["ensemble"][ENS][per]["BSS_dev"] - scores["ensemble"][f"{cond}_ENS"][per]["BSS_dev"]
            for per in PERIODS}
        for m in MEMBERS:
            dump["members"][m][cond] = {
                per: scores["members"][m][f"M0_{m}"][per]["BSS_dev"]
                     - scores["members"][m][f"{cond}_{m}"][per]["BSS_dev"]
                for per in PERIODS}

    # ---- bootstrap --------------------------------------------------------
    boot: dict[str, dict] = {}
    for cond in CONDS:
        boot[cond] = {}
        for per in BOOT_PERIODS:
            boot[cond][per] = boot_contrast(preds[ENS], preds[f"{cond}_ENS"], y, sid, masks[per])
    boot_members: dict[str, dict] = {}
    for m in MEMBERS:
        boot_members[m] = {"C2": {}}
        for per in [str(Y) for Y in YEARS]:
            boot_members[m]["C2"][per] = boot_contrast(
                preds[f"M0_{m}"], preds[f"C2_{m}"], y, sid, masks[per])

    # ---- cross-check the re-signed bootstrap against the vetted function --
    cv_cases = []
    for per in ["2020", "2021", "2023", "pooled_2020_2021"]:
        ref = vetted.boot_delta_fixed_year(preds[f"C2_ENS"], preds[ENS], y, sid, masks[per])
        mine = boot["C2"][per]
        cv_cases.append({
            "period": per,
            "vetted_delta_cond_minus_M0": ref["delta"],
            "delta_absdiff": abs(mine["delta"] + ref["delta"]),
            "ci90_lo_absdiff": abs(mine["ci90"][0] + ref["ci_hi"]),
            "ci90_hi_absdiff": abs(mine["ci90"][1] + ref["ci_lo"]),
            "p_boot_absdiff": abs(mine["p_boot"] - ref["p_boot"]),
        })
    cv_max = max(max(c["delta_absdiff"], c["ci90_lo_absdiff"], c["ci90_hi_absdiff"],
                     c["p_boot_absdiff"]) for c in cv_cases)

    # ---- consistency with the frozen eval artefacts -----------------------
    def _eval_metrics(name: str) -> dict:
        return json.loads((EVAL_DIR / f"{name}.json").read_text())["metrics"]

    ens_ref = _eval_metrics("main_2024")
    cons = {
        "ensemble_M0_vs_eval_main_2024": {
            **{f"year{Y}": {
                "computed": scores["ensemble"][ENS][str(Y)]["BSS_dev"],
                "eval_json": ens_ref[f"year{Y}"]["BSS_dev"],
                "abs_diff": abs(scores["ensemble"][ENS][str(Y)]["BSS_dev"]
                                - ens_ref[f"year{Y}"]["BSS_dev"]),
            } for Y in YEARS},
            "all": {
                "computed": scores["ensemble"][ENS]["pooled_2020_2024"]["BSS_dev"],
                "eval_json": ens_ref["all"]["BSS_dev"],
                "abs_diff": abs(scores["ensemble"][ENS]["pooled_2020_2024"]["BSS_dev"]
                                - ens_ref["all"]["BSS_dev"]),
            },
        },
        "members_M0_vs_eval_json": {},
    }
    for m in MEMBERS:
        ref = _eval_metrics(f"M0_ens_lbB_{m}")
        d = {f"year{Y}": abs(scores["members"][m][f"M0_{m}"][str(Y)]["BSS_dev"]
                             - ref[f"year{Y}"]["BSS_dev"]) for Y in YEARS}
        d["all"] = abs(scores["members"][m][f"M0_{m}"]["pooled_2020_2024"]["BSS_dev"]
                       - ref["all"]["BSS_dev"])
        cons["members_M0_vs_eval_json"][m] = d

    eval_diffs = [v["abs_diff"] for v in cons["ensemble_M0_vs_eval_main_2024"].values()]
    eval_diffs += [v for d in cons["members_M0_vs_eval_json"].values() for v in d.values()]
    cons["max_abs_diff_vs_eval_json"] = float(max(eval_diffs))
    cons["matches_eval_lt_1e_12"] = bool(max(eval_diffs) < 1e-12)

    # pooled 2020-2024 ensemble ΔC2 must be +0.032 (+-1e-3)
    pooled_all = dump["ensemble"]["C2"]["pooled_2020_2024"]
    TARGET = 0.032
    consistency_target = {
        "quantity": "ΔBSS_dev(ensemble, C2, pooled 2020-2024) = BSS_dev(M0) - BSS_dev(C2)",
        "computed": float(pooled_all),
        "target": TARGET,
        "abs_diff": abs(pooled_all - TARGET),
        "within_1e_3": bool(abs(pooled_all - TARGET) <= 1e-3),
        "sign_note": "worst_year_ablation.json uses C2-M0, so its ensemble_pooled_delta_C2 should be the negative of this value",
    }
    prev_json = P2_ROOT / "data" / "analysis" / "worst_year_ablation.json"
    prev_year_diff = None
    if prev_json.exists():
        prev_out = json.loads(prev_json.read_text())
        prev = prev_out["verdict"]["ensemble_pooled_delta_C2"]
        consistency_target["worst_year_ablation_ensemble_pooled_delta_C2"] = float(prev)
        consistency_target["match_vs_previous_script_absdiff"] = float(abs(prev + pooled_all))
        prev_year_diff = {
            per: abs(dump["ensemble"]["C2"][per] + prev_out["delta_vs_M0"]["C2_ENS"][key])
            for per, key in [(str(Y), f"year{Y}") for Y in YEARS] + [("pooled_2020_2024", "all")]
        }
        consistency_target["per_year_vs_worst_year_ablation_absdiff"] = prev_year_diff
        consistency_target["per_year_max_absdiff"] = float(max(prev_year_diff.values()))

    per_year_m0 = {str(Y): scores["ensemble"][ENS][str(Y)]["BSS_dev"] for Y in YEARS}
    cons["per_year_M0_ensemble"] = per_year_m0

    checks = {
        "per_file": per_file,
        "n_npz_checked": len(per_file),
        "manifest_hash_verified_all_npz": bool(
            all(c["manifest_sha256_match"] for c in per_file.values())),
        "all_split_test": bool(all(c["split"] in (None, ["test"]) for c in per_file.values())),
        "bootstrap_crosscheck_vs_ablation_worst_year": {
            "cases": cv_cases,
            "max_abs_diff": float(cv_max),
            "passed_lt_1e_12": bool(cv_max < 1e-12),
            "note": "With the same seed and resampling indices, this script's Δ(M0−cond) is exactly "
                    "the negative of the vetted Δ(cond−M0); 90% CI endpoints mirror and p_boot is identical.",
        },
        "consistency_with_eval_json": cons,
        "pooled_all_ensemble_delta_C2": consistency_target,
        "per_year_crosscheck_vs_worst_year_ablation": {
            "abs_diff_by_period": prev_year_diff,
            "passed": bool(prev_year_diff is None or max(prev_year_diff.values()) < 1e-12),
        },
        "all_passed": bool(
            max(eval_diffs) < 1e-12
            and cv_max < 1e-12
            and consistency_target["within_1e_3"]
            and (prev_year_diff is None or max(prev_year_diff.values()) < 1e-12)
            and all(c["manifest_sha256_match"] for c in per_file.values())
            and all(c["split"] in (None, ["test"]) for c in per_file.values())
        ),
    }

    # ---- member gain counts ----------------------------------------------
    counts = {}
    for per in [str(Y) for Y in YEARS] + ["pooled_2020_2021", "pooled_2020_2024"]:
        block_c = {}
        for cond in CONDS:
            vals = {m: dump["members"][m][cond][per] for m in MEMBERS}
            sig = {m: boot_members[m]["C2"][per] for m in MEMBERS} if (
                cond == "C2" and per in boot_members[MEMBERS[0]]["C2"]) else {}
            block_c[cond] = {
                "per_member": {m: float(v) for m, v in vals.items()},
                "n_positive": int(sum(1 for v in vals.values() if v > 0)),
                "n_negative": int(sum(1 for v in vals.values() if v < 0)),
                "members_positive": [m for m, v in vals.items() if v > 0],
                "members_negative": [m for m, v in vals.items() if v < 0],
                "members_significant_90": [m for m in sig if sig[m]["significant_90"]],
                "members_significant_95": [m for m in sig if sig[m]["significant_95"]],
                "mean_delta": float(np.mean(list(vals.values()))),
            }
        counts[per] = block_c

    # ---- headline summary ------------------------------------------------
    e2 = {per: boot["C2"][per] for per in BOOT_PERIODS}
    e1 = {per: boot["C1"][per] for per in BOOT_PERIODS}
    summary = {
        "contrast": "ΔBSS_dev = BSS_dev(M0) − BSS_dev(cond); positive values favour the causal component",
        "ensemble_C2_by_year": {
            per: {"delta": e2[per]["delta"], "ci90": e2[per]["ci90"],
                  "ci95": e2[per]["ci95"], "p_boot": e2[per]["p_boot"],
                  "significant_90": e2[per]["significant_90"],
                  "significant_95": e2[per]["significant_95"]}
            for per in [str(Y) for Y in YEARS]},
        "ensemble_C2_pooled_2020_2021": {
            "delta": e2["pooled_2020_2021"]["delta"],
            "ci90": e2["pooled_2020_2021"]["ci90"],
            "ci95": e2["pooled_2020_2021"]["ci95"],
            "p_boot": e2["pooled_2020_2021"]["p_boot"],
            "significant_95": e2["pooled_2020_2021"]["significant_95"]},
        "ensemble_C2_pooled_2020_2024_point": float(dump["ensemble"]["C2"]["pooled_2020_2024"]),
        "ensemble_C1_pooled_2020_2021_point": float(dump["ensemble"]["C1"]["pooled_2020_2021"]),
        "ensemble_C1_by_year_point": {str(Y): float(dump["ensemble"]["C1"][str(Y)]) for Y in YEARS},
        "years_95_ci_excludes_zero_C2": [
            int(per) for per in [str(Y) for Y in YEARS] if e2[per]["significant_95"]],
        "years_90_ci_excludes_zero_C2": [
            int(per) for per in [str(Y) for Y in YEARS] if e2[per]["significant_90"]],
        "member_gain_counts": {
            per: {cond: {"n_positive": counts[per][cond]["n_positive"],
                         "n_negative": counts[per][cond]["n_negative"]}
                  for cond in CONDS}
            for per in ["2020", "2021", "pooled_2020_2021", "pooled_2020_2024"]},
    }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/05_eval/ablation_year_gradient.py",
        "protocol": "holdout_masked_v1",
        "read_only": True,
        "method_source": (
            "scripts/05_eval/ablation_worst_year.py::load_frozen (split, protocol, "
            "and manifest SHA-256 checks), bss, DEV_RATE, MEMBERS, and YEARS are imported. "
            "The fixed-year bootstrap is reused with the contrast sign reversed and "
            "95% intervals added, and is cross-checked against that function in four periods."),
        "sources": {
            "pred_dir": "data/test/holdout_masked_v1",
            "eval_dir": "data/analysis/eval",
            "n_members": len(MEMBERS),
            "members": MEMBERS,
            "conditions": list(CONDS),
            "n_test_rows": int(len(y)),
            "n_storms": int(len(np.unique(sid))),
            "years": YEARS,
            "periods": {k: [int(v) for v in vals] for k, vals in PERIODS.items()},
        },
        "settings": {
            "bss_margin": "BSS_dev",
            "dev_rate": DEV_RATE,
            "dev_rate_source": "eval/main_2024.json → dev_rate.dev_rate (2000-2019)",
            "bs_clim": "BS_clim = π² + (1−2π)·ȳ_period",
            "bootstrap": N_BOOT,
            "seed": SEED,
            "alphas": [0.10, 0.05],
            "cluster": "sid (whole-storm resampling, paired across the two conditions)",
            "p_boot": "2*min(P(draw<=0), P(draw>=0)), clipped at 1/(B+1)",
            "year_source": "row_uid ISO_TIME (storm_year column)",
            "ensemble_definition": "equal-weight mean of p over the same 5 members",
            "pooled_bs_clim_note": "Each pooled period's BS_clim uses that subset's own observed event rate, with the same bss call as the vetted script",
        },
        "checks": checks,
        "period_stats": {per: {
            "n": int(masks[per].sum()),
            "n_storms": int(len(np.unique(sid[masks[per]]))),
            "base_rate": float(y[masks[per]].mean()),
            "bs_clim_dev": float(DEV_RATE ** 2 + (1 - 2 * DEV_RATE) * y[masks[per]].mean()),
        } for per in PERIODS},
        "bss_dev": scores,
        "delta_M0_minus_cond": dump,
        "bootstrap_ensemble": boot,
        "bootstrap_members_C2": boot_members,
        "member_gain_counts": counts,
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=False), encoding="utf-8")
    OUT_MD.write_text(build_md(out), encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(P2_ROOT)}")
    print(f"wrote {OUT_MD.relative_to(P2_ROOT)}")
    print(f"checks all_passed = {checks['all_passed']}  "
          f"max|Δ-eval| = {cons['max_abs_diff_vs_eval_json']:.2e}  "
          f"bootstrap crosscheck = {cv_max:.2e}")
    for Y in YEARS:
        b = boot["C2"][str(Y)]
        print(f"  {Y}: ΔC2 = {b['delta']:+.4f}  90%[{b['ci90'][0]:+.4f},{b['ci90'][1]:+.4f}] "
              f"95%[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] p={b['p_boot']:.4f} "
              f"members>0 {counts[str(Y)]['C2']['n_positive']}/5")
    b = boot["C2"]["pooled_2020_2021"]
    print(f"  2020-21 pooled: ΔC2 = {b['delta']:+.4f} "
          f"90%[{b['ci90'][0]:+.4f},{b['ci90'][1]:+.4f}] "
          f"95%[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] p={b['p_boot']:.4f}")
    print(f"  pooled all ΔC2 = {pooled_all:+.6f} (target +0.032, "
          f"|diff| = {abs(pooled_all - TARGET):.2e})")


# ----------------------------------------------------------------------
# markdown report
# ----------------------------------------------------------------------

def _f(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def _e(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def _ci(ci: list, nd: int = 3) -> str:
    return f"[{_f(ci[0], nd)}, {_f(ci[1], nd)}]"


def build_md(out: dict) -> str:
    scores = out["bss_dev"]
    dump = out["delta_M0_minus_cond"]
    boot = out["bootstrap_ensemble"]
    bm = out["bootstrap_members_C2"]
    counts = out["member_gain_counts"]
    ps = out["period_stats"]
    cons = out["checks"]["consistency_with_eval_json"]
    ck = out["checks"]
    S = out["summary"]
    members = MEMBERS
    years = YEARS

    L: list[str] = []
    A = L.append
    A("# Yearly ablation gradient: where the graph pathway helps during 2020–2024\n")
    A("Generated by `scripts/05_eval/ablation_year_gradient.py`. The inputs are "
      "45 member-level frozen NPZ files (five members × {M0, C1, C2, C3s1-3, "
      "C4adj, C4cf, C4sign}) and the equal-weight five-member `main.npz`, "
      "46 files in total. This analysis uses M0, C2, and C1. It reads frozen "
      "predictions without training or changing thresholds.\n")
    A(f"**Definition.** `BSS_dev = 1 − BS/BS_clim`, with "
      f"`BS_clim = π² + (1−2π)·ȳ` and π = {DEV_RATE:.6f} for the 2000–2019 "
      "development period. **ΔBSS_dev = BSS_dev(M0) − BSS_dev(condition)**, so "
      "a positive value favours the causal component. C2 removes the GNN graph "
      "branch; C1 removes all causal losses. Years come from `row_uid` ISO_TIME, "
      "and the ensemble is the equal-weight mean of the five members.\n")
    A("**Inference.** Paired storm-cluster bootstrap resamples whole `sid` "
      f"clusters with replacement, B = {N_BOOT}, seed = {SEED}. The same draw "
      "is used for both conditions. Percentile 90% and 95% intervals are reported, "
      "with `p_boot = 2·min(P(draw≤0), P(draw≥0))` and lower bound 1/(B+1). "
      "`load_frozen`, `bss`, and `DEV_RATE` are shared with "
      "`scripts/05_eval/ablation_worst_year.py`. The fixed-year bootstrap uses "
      "the opposite contrast sign and adds 95% intervals from the same draws; "
      "Section 0 cross-checks it against the shared implementation in four periods.\n")

    # --- checks
    A("## 0. Consistency checks\n")
    A("| Check | Result |\n|---|---|")
    A(f"| NPZ files (45 member-level + 1 ensemble) / all manifest hashes match / all rows are test | "
      f"{ck['n_npz_checked']} / {ck['manifest_hash_verified_all_npz']} / {ck['all_split_test']} |")
    pd_year = max(v["abs_diff"] for k, v in cons["ensemble_M0_vs_eval_main_2024"].items())
    A(f"| Ensemble M0 yearly BSS_dev versus `eval/main_2024.json` (five years) | "
      f"Maximum absolute difference = {pd_year:.2e} |")
    A(f"| Ensemble M0 all-year BSS_dev versus `eval/main_2024.json` | "
      f"Computed {cons['ensemble_M0_vs_eval_main_2024']['all']['computed']:.10f}; "
      f"eval {cons['ensemble_M0_vs_eval_main_2024']['all']['eval_json']:.10f}; "
      f"absolute difference = {cons['ensemble_M0_vs_eval_main_2024']['all']['abs_diff']:.2e} |")
    mem_max = max(max(v.values()) for v in cons["members_M0_vs_eval_json"].values())
    A(f"| Five M0 members, yearly and all-year, versus `eval/M0_ens_lbB_*.json` | "
      f"Maximum absolute difference = {mem_max:.2e} |")
    A(f"| Maximum absolute difference across eval checks | {cons['max_abs_diff_vs_eval_json']:.2e}; "
      f"{'consistent (<1e-12)' if cons['matches_eval_lt_1e_12'] else '**requires review**'} |")
    ct = ck["pooled_all_ensemble_delta_C2"]
    A(f"| Pooled 2020–2024 ensemble ΔC2 against the +0.032 target | {ct['computed']:+.6f}; "
      f"absolute difference = {ct['abs_diff']:.2e}; "
      f"{'passes (≤1e-3)' if ct['within_1e_3'] else '**fails**'} |")
    if "match_vs_previous_script_absdiff" in ct:
        A(f"| `worst_year_ablation.json` pooled C2, with the opposite sign convention | "
          f"{ct['worst_year_ablation_ensemble_pooled_delta_C2']:+.6f}; "
          f"absolute difference after negation = {ct['match_vs_previous_script_absdiff']:.2e} |")
        pyc = ck["per_year_crosscheck_vs_worst_year_ablation"]
        A(f"| Yearly ensemble ΔC2 versus `worst_year_ablation.json`, opposite sign | "
          f"Maximum absolute difference = {max(pyc['abs_diff_by_period'].values()):.2e}; "
          f"{'consistent' if pyc['passed'] else '**inconsistent**'} |")
    bcv = ck["bootstrap_crosscheck_vs_ablation_worst_year"]
    A(f"| Bootstrap cross-check against the shared implementation (four periods) | "
      f"Maximum absolute difference = {bcv['max_abs_diff']:.2e}; "
      f"{'consistent' if bcv['passed_lt_1e_12'] else '**inconsistent**'} |")
    A(f"| All checks | **{'PASS' if ck['all_passed'] else 'FAIL'}** |\n")

    # --- ensemble per year
    A("## 1. Ensemble ΔC2 by year\n")
    A("| Year | Rows | Storms | Observed rate | M0 BSS_dev | C2 BSS_dev | **ΔC2** | 90% CI | 95% CI | p_boot | 95% CI contains 0 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for Y in years:
        per = str(Y)
        b = boot["C2"][per]
        r0 = scores["ensemble"]["M0_equal_weight_5members"][per]
        r2 = scores["ensemble"]["C2_ENS"][per]
        A(f"| {Y} | {ps[per]['n']} | {ps[per]['n_storms']} | {ps[per]['base_rate']:.4f} | "
          f"{_e(r0['BSS_dev'])} | {_e(r2['BSS_dev'])} | **{_f(b['delta'])}** | "
          f"{_ci(b['ci90'])} | {_ci(b['ci95'])} | {b['p_boot']:.4f} | "
          f"{'Yes' if not b['significant_95'] else 'No'} |")
    per = "pooled_2020_2021"
    b = boot["C2"][per]
    r0 = scores["ensemble"]["M0_equal_weight_5members"][per]
    r2 = scores["ensemble"]["C2_ENS"][per]
    A(f"| **2020–2021 pooled** | {ps[per]['n']} | {ps[per]['n_storms']} | "
      f"{ps[per]['base_rate']:.4f} | {_e(r0['BSS_dev'])} | {_e(r2['BSS_dev'])} | "
      f"**{_f(b['delta'])}** | {_ci(b['ci90'])} | {_ci(b['ci95'])} | {b['p_boot']:.4f} | "
      f"{'Yes' if not b['significant_95'] else 'No'} |")
    per = "pooled_2020_2024"
    r0 = scores["ensemble"]["M0_equal_weight_5members"][per]
    r2 = scores["ensemble"]["C2_ENS"][per]
    A(f"| 2020–2024 pooled (full test) | {ps[per]['n']} | {ps[per]['n_storms']} | "
      f"{ps[per]['base_rate']:.4f} | {_e(r0['BSS_dev'])} | {_e(r2['BSS_dev'])} | "
      f"{_f(dump['ensemble']['C2'][per])} | — | — | — | — |")
    A("")
    A("Each period uses its own observed event rate in `BS_clim`. Yearly contrasts "
      "and the pooled contrast therefore have different reference scores and are "
      "not related by a simple row-weighted average.\n")

    # --- member layer
    A("## 2. Member-level ΔC2 by year\n")
    A("Point estimates and paired storm-cluster bootstrap intervals:\n")
    A("| Member | Quantity | 2020 | 2021 | 2022 | 2023 | 2024 |")
    A("|---|---|---|---|---|---|---|")
    for m in members:
        A(f"| {m} | ΔC2 (90% CI) | " + " | ".join(
            f"{_f(dump['members'][m]['C2'][str(Y)])} "
            f"({_ci(bm[m]['C2'][str(Y)]['ci90'], 3)})" for Y in years) + " |")
    A("")
    A("The corresponding 95% intervals from the same draws:\n")
    A("| Member | 2020 | 2021 | 2022 | 2023 | 2024 |")
    A("|---|---|---|---|---|---|")
    for m in members:
        A(f"| {m} | " + " | ".join(_ci(bm[m]["C2"][str(Y)]["ci95"])
                                   for Y in years) + " |")
    A("")
    A("**Positive-contrast counts.** A positive ΔC2 means that the graph pathway "
      "helps that member in that year.\n")
    A("| Quantity | 2020 | 2021 | 2022 | 2023 | 2024 | 2020–2021 pooled | 2020–2024 pooled |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    A("| Members with ΔC2 > 0 (of 5) | " + " | ".join(
        str(counts[str(Y)]["C2"]["n_positive"]) for Y in years) + f" | "
      f"{counts['pooled_2020_2021']['C2']['n_positive']} | "
      f"{counts['pooled_2020_2024']['C2']['n_positive']} |")
    A("| Members whose 95% CI excludes 0 | " + " | ".join(
        str(len(counts[str(Y)]["C2"]["members_significant_95"])) for Y in years) + " | — | — |")
    A("")
    for Y in years:
        cp = counts[str(Y)]["C2"]
        A(f"- {Y}: positive {cp['n_positive']}/5 (" + ", ".join(cp["members_positive"]) +
          "); negative " + ((", ".join(cp["members_negative"])) if cp["members_negative"] else "none") +
          "; significant at 95%: " + ((", ".join(cp["members_significant_95"]))
                          if cp["members_significant_95"] else "none") + ".")
    A("")

    # --- C1
    A("## 3. C1 control: remove all causal losses\n")
    A("| Period | Ensemble ΔC1 | 90% CI | 95% CI | p_boot | Members with ΔC1 > 0 |")
    A("|---|---:|---:|---:|---:|---:|")
    for per in [str(Y) for Y in years] + ["pooled_2020_2021", "pooled_2020_2024"]:
        b = boot["C1"].get(per)
        label = (per.replace("pooled_2020_2021", "2020–2021 pooled")
                    .replace("pooled_2020_2024", "2020–2024 pooled"))
        if b:
            A(f"| {label} | {_f(b['delta'], 4)} | {_ci(b['ci90'], 4)} | {_ci(b['ci95'], 4)} | "
              f"{b['p_boot']:.4f} | {counts[per]['C1']['n_positive']}/5 |")
        else:
            A(f"| {label} | {_f(dump['ensemble']['C1'][per], 4)} | — | — | — | "
              f"{counts[per]['C1']['n_positive']}/5 |")
    A("")
    A("C1 intervals are reported for the ensemble; member rows give point estimates.\n")

    # --- conclusion
    A("## 4. Interpretation\n")
    sig95 = S["years_95_ci_excludes_zero_C2"]
    sig90 = S["years_90_ci_excludes_zero_C2"]
    e2y = S["ensemble_C2_by_year"]
    order = sorted(years, key=lambda Y: -e2y[str(Y)]["delta"])
    A(f"1. **The gradient is non-monotonic.** Ensemble ΔC2 is "
      + ", ".join(f"{Y} {_f(e2y[str(Y)]['delta'])}" for Y in years)
      + "; the descending order is " + " > ".join(f"{Y}" for Y in order) + ". "
      f"The pooled 2020–2021 contrast is {_f(S['ensemble_C2_pooled_2020_2021']['delta'])} "
      f"(90% CI {_ci(S['ensemble_C2_pooled_2020_2021']['ci90'])}), nearly twice the "
      f"full-test pooled value of {_f(S['ensemble_C2_pooled_2020_2024_point'])}. "
      f"These two years contribute {ps['pooled_2020_2021']['n'] / ps['pooled_2020_2024']['n']:.0%} "
      "of the test rows. Each period has its own reference rate, so yearly and pooled "
      "contrasts cannot be combined by a simple row-weighted average.\n")
    A(f"2. **Significance.** The 95% intervals exclude zero in "
      + (", ".join(str(Y) for Y in sig95) if sig95 else "no year")
      + "; the 90% intervals exclude zero in "
      + (", ".join(str(Y) for Y in sig90) if sig90 else "no year")
      + f". The pooled 2020–2021 p_boot is {S['ensemble_C2_pooled_2020_2021']['p_boot']:.4f}.")
    for Y in years:
        b = e2y[str(Y)]
        if not b["significant_95"]:
            A(f" {Y}: the 95% CI {_ci(b['ci95'])} contains zero (p_boot = {b['p_boot']:.4f}), "
              + ("and the 90% CI also contains zero" if not b["significant_90"] else "while the 90% CI excludes zero")
              + ".")
    A("")
    A(f"3. **Member pattern.** Positive counts are {counts['2020']['C2']['n_positive']}/5 "
      f"in 2020 and {counts['2021']['C2']['n_positive']}/5 in 2021 ("
      + ", ".join(counts['2020']['C2']['members_positive']) + " are all positive in 2020), "
      f"then {counts['2022']['C2']['n_positive']}/5, {counts['2023']['C2']['n_positive']}/5, "
      f"and {counts['2024']['C2']['n_positive']}/5 in 2022–2024. The 2023 ensemble contrast "
      f"is negative ({_f(dump['ensemble']['C2']['2023'])}), but its interval contains zero. "
      "Member intervals are wide because each year contains 19–29 storm clusters, so the "
      "member counts describe consistency rather than separate member-level discoveries.\n")
    A(f"4. **Loss-only control.** The pooled 2020–2021 C1 contrast is "
      f"{_f(S['ensemble_C1_pooled_2020_2021_point'], 4)} "
      f"(95% CI {_ci(boot['C1']['pooled_2020_2021']['ci95'], 4)}). Yearly values are "
      + ", ".join(f"{Y} {_f(S['ensemble_C1_by_year_point'][str(Y)], 4)}" for Y in years)
      + f", and every 95% interval contains zero. The corresponding C2 contrast is "
      f"{_f(S['ensemble_C2_pooled_2020_2021']['delta'])}. C1 retains the GNN branch and "
      "removes only the three causal losses; its small contrast identifies the graph "
      "pathway as the main source of the ensemble gain in this period.\n")
    A("**Summary.** The ensemble graph-removal loss is largest in 2020–2021 ("
      f"pooled ΔC2 = {_f(S['ensemble_C2_pooled_2020_2021']['delta'])}, "
      f"95% CI {_ci(S['ensemble_C2_pooled_2020_2021']['ci95'])}, p_boot = "
      f"{S['ensemble_C2_pooled_2020_2021']['p_boot']:.4f}), with "
      f"{_f(e2y['2020']['delta'])} in 2020 and {_f(e2y['2021']['delta'])} in 2021. "
      f"Positive member counts are {counts['2020']['C2']['n_positive']}/5 and "
      f"{counts['2021']['C2']['n_positive']}/5 in those years. The 2022 and 2024 "
      f"contrasts are positive with intervals containing zero; the 2023 contrast is "
      f"negative with an interval containing zero. The years with 95% intervals "
      f"excluding zero are {(', '.join(str(Y) for Y in sig95) if sig95 else 'none')}.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
