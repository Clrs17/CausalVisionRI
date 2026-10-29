# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Subset ablation panel: five members x eight conditions on a test-year subset.

Read-only analysis of the FROZEN test predictions in
``data/test/holdout_masked_v1`` (protocol holdout_masked_v1).  No training, no
re-evaluation, no threshold change.  Every number is a decomposition of
``BSS_dev`` (fixed 2000-2019 development RI rate) restricted to the test rows
whose ``row_uid`` ISO_TIME year is in the requested year set — 2020-2021 pooled
(the two years nearest the 2000-2017 estimation window) by default, or a single
year (2020, 2021) with ``--years``.

Contrast
    ΔBSS_dev(cond) = BSS_dev(M0) - BSS_dev(cond),  positive = the causal
    component helps.  Identical to the pooled forest panel
    (``fig_ablation_forest``), only the row subset changes.

Method is not re-implemented: the frozen-npz loader (with split / protocol /
manifest-sha256 verification), ``DEV_RATE``, the member list and the bootstrap
are taken from the vetted ``scripts/ablation_worst_year.py`` and
``src/evaluate.py`` (``paired_cluster_bootstrap`` = the exact function used by
``scripts/eval_holdout.py`` to produce the published ``paired_bootstrap``
blocks, hence the same ``p_boot`` definition).  Holm-Bonferroni is applied per
member over the eight-condition family at alpha = 0.10, exactly as the final
lock declares for the full-test evaluation.

Outputs
    data/analysis/ablation_2020_2021_panel.json   (default: both years pooled)
    data/analysis/ablation_2020_panel.json        (--years 2020)
    data/analysis/ablation_2021_panel.json        (--years 2021)

Usage (from the release root):
    python scripts/05_eval/ablation_subset_panel.py                  # 2020-2021 pooled
    python scripts/05_eval/ablation_subset_panel.py --years 2020
    python scripts/05_eval/ablation_subset_panel.py --years 2021
"""

from __future__ import annotations

import argparse
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

import ablation_worst_year as vetted  # noqa: E402
from wpac_warning.evaluate import bss, holm_adjust, paired_cluster_bootstrap  # noqa: E402

MEMBERS = vetted.MEMBERS
DEV_RATE = vetted.DEV_RATE
N_BOOT = vetted.N_BOOT
SEED = vetted.SEED
ALPHA = vetted.ALPHA
EVAL_DIR = vetted.EVAL_DIR
ENS = vetted.ENSEMBLE_M0

# figure row order (same as the pooled forest panel)
CONDS = ["C1", "C2", "C3s1", "C3s2", "C3s3", "C4adj", "C4sign", "C4cf"]

SUBSET_YEARS = (2020, 2021)          # default run: the two years pooled
YEAR_ROWS = {2020: 559, 2021: 757, 2022: 571, 2023: 606, 2024: 511}
YEAR_STORMS = {2020: 26, 2021: 29}   # frozen test storms per year

ANALYSIS_DIR = P2_ROOT / "data" / "analysis"
YEAR_GRADIENT_JSON = ANALYSIS_DIR / "year_gradient_ablation.json"


def year_slug(years) -> str:
    return "_".join(str(y) for y in years)


def years_suffix(years) -> str:
    """Suffix used in the check keys ($YEAR_GRADIENT comparison)."""
    return f"{years[0]}" if len(years) == 1 else year_slug(years)


def period_key(years) -> str:
    """Key of the same period in ``year_gradient_ablation.json``."""
    return str(years[0]) if len(years) == 1 else f"pooled_{years[0]}_{years[-1]}"


def period_label(years) -> str:
    """Figure corner label for the period."""
    if len(years) == 1:
        return (f"{years[0]} (test year nearest the estimation window)"
                if years[0] == 2020 else f"{years[0]}")
    return f"{years[0]}\u2013{years[-1]} (years nearest the estimation window)"


def expected_counts(years) -> tuple[int, int | None]:
    rows = sum(YEAR_ROWS[y] for y in years)
    storms = (sum(YEAR_STORMS[y] for y in years)
              if all(y in YEAR_STORMS for y in years) else None)
    return rows, storms


def out_json_for(years) -> Path:
    return ANALYSIS_DIR / f"ablation_{year_slug(years)}_panel.json"



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


def contrast(p_m0: np.ndarray, p_cond: np.ndarray, y: np.ndarray,
             sid: np.ndarray) -> dict:
    """Δ = BSS_dev(M0) - BSS_dev(cond) with the eval-script paired bootstrap."""
    return paired_cluster_bootstrap(y, p_m0, p_cond, sid, bss,
                                    n_boot=N_BOOT, seed=SEED, alpha=ALPHA,
                                    fixed_rate=DEV_RATE)


def main(years=SUBSET_YEARS, out_json: Path | None = None) -> None:
    years = tuple(sorted(int(y) for y in years))
    out_json = Path(out_json) if out_json is not None else out_json_for(years)
    exp_rows, exp_storms = expected_counts(years)

    data, per_file = vetted.load_frozen()
    preds, meta = data["preds"], data["meta"]
    y, sid, year = meta["y"], meta["sid"], meta["storm_year"]

    mask = np.isin(year, years)
    n_rows = int(mask.sum())
    n_storms = int(len(np.unique(sid[mask])))
    if n_rows != exp_rows:
        raise SystemExit(f"subset row count {n_rows} != expected {exp_rows}")
    if exp_storms is not None and n_storms != exp_storms:
        raise SystemExit(f"subset storm count {n_storms} != expected {exp_storms}")

    ys, ss = y[mask], sid[mask]
    sub = {k: v[mask] for k, v in preds.items()}

    # ---- point estimates --------------------------------------------------
    m0_block = {m: block(sub[f"M0_{m}"], ys) for m in MEMBERS}
    m0_block[ENS] = block(sub[ENS], ys)
    cond_block = {m: {c: block(sub[f"{c}_{m}"], ys) for c in CONDS} for m in MEMBERS}

    # ---- bootstrap + Holm (per member, 8-condition family) ----------------
    panel: dict[str, dict] = {}
    for m in MEMBERS:
        stats = {c: contrast(sub[f"M0_{m}"], sub[f"{c}_{m}"], ys, ss) for c in CONDS}
        adjusted = holm_adjust({c: stats[c]["p_boot"] for c in CONDS})
        panel[m] = {}
        for c in CONDS:
            s = stats[c]
            panel[m][c] = {
                "delta": s["delta"],
                "ci_lo": s["ci_lo"],
                "ci_hi": s["ci_hi"],
                "p_boot": s["p_boot"],
                "p_holm": float(adjusted[c]),
                "reject_holm": bool(adjusted[c] <= ALPHA),
                "ci_excludes_zero": bool(s["ci_lo"] > 0 or s["ci_hi"] < 0),
                "bss_dev_M0": m0_block[m]["BSS_dev"],
                "bss_dev_cond": cond_block[m][c]["BSS_dev"],
                "n_boot": int(s["n_boot"]),
                "n_clusters": int(s["n_clusters"]),
                "n_rows": int(s["n_rows"]),
                "alpha": float(s["alpha"]),
            }

    # ---- checks -----------------------------------------------------------
    pk = period_key(years)
    yg = json.loads(YEAR_GRADIENT_JSON.read_text()) if YEAR_GRADIENT_JSON.exists() else None
    checks: dict = {
        "n_npz_checked": len(per_file),
        "npz_kinds": "1 main.npz + 5 M0 + 40 abl = 46",
        "manifest_hash_verified_all_npz": bool(
            all(c["manifest_sha256_match"] for c in per_file.values())),
        "all_split_test": bool(all(c["split"] in (None, ["test"]) for c in per_file.values())),
        "all_protocol_holdout_masked_v1": bool(
            all(c["protocol"] == "holdout_masked_v1" for c in per_file.values())),
        "subset": {
            "years": list(years),
            "label": period_label(years),
            "period_key_in_year_gradient": pk,
            "year_source": "row_uid ISO_TIME (storm_year column of main.npz)",
            "n_rows": n_rows,
            "expected_rows": exp_rows,
            "rows_match": bool(n_rows == exp_rows),
            "n_storms": n_storms,
            "expected_storms": exp_storms,
            "storms_match": bool(exp_storms is None or n_storms == exp_storms),
            "base_rate": float(ys.mean()),
            "bs_clim_dev": float(DEV_RATE ** 2 + (1.0 - 2.0 * DEV_RATE) * ys.mean()),
        },
    }

    if yg is not None:
        ref = yg["bss_dev"]
        m0_vs_yg = {
            m: {
                "computed": float(m0_block[m]["BSS_dev"]),
                f"year_gradient_{pk}": float(
                    ref["members"][m][f"M0_{m}"][pk]["BSS_dev"]),
            } for m in MEMBERS
        }
        for m in MEMBERS:
            m0_vs_yg[m]["abs_diff"] = abs(
                m0_vs_yg[m]["computed"] - m0_vs_yg[m][f"year_gradient_{pk}"])
        ens_vs_yg = {
            "computed": float(m0_block[ENS]["BSS_dev"]),
            f"year_gradient_{pk}": float(
                ref["ensemble"][ENS][pk]["BSS_dev"]),
        }
        ens_vs_yg["abs_diff"] = abs(ens_vs_yg["computed"]
                                    - ens_vs_yg[f"year_gradient_{pk}"])
        yg_conds = [c for c in CONDS
                    if c in yg["delta_M0_minus_cond"]["members"][MEMBERS[0]]]
        point_vs_yg = {
            m: {c: abs(panel[m][c]["delta"]
                       - float(yg["delta_M0_minus_cond"]["members"][m][c][pk]))
                for c in yg_conds} for m in MEMBERS
        }
        max_point = max(max(v.values()) for v in point_vs_yg.values())
        max_m0 = max(max(v["abs_diff"] for v in m0_vs_yg.values()),
                     ens_vs_yg["abs_diff"])
        checks[f"member_M0_bss_vs_year_gradient_{years_suffix(years)}"] = {
            "period_key": pk,
            "members": m0_vs_yg,
            "ensemble": ens_vs_yg,
            "max_abs_diff": float(max_m0),
            "tolerance": 1e-3,
            "passed": bool(max_m0 <= 1e-3),
        }
        checks[f"point_delta_vs_year_gradient_{years_suffix(years)}"] = {
            "period_key": pk,
            "conditions": yg_conds,
            "abs_diff_by_member_condition": point_vs_yg,
            "max_abs_diff": float(max_point),
            "passed_lt_1e_12": bool(max_point < 1e-12),
            "note": "year_gradient_ablation.json provides only C2/C1 point estimates for this period. "
                    "With the same subset, BS_clim, and bss call, this script's Δ(M0−cond) should "
                    "match those point estimates exactly (confidence intervals are not compared).",
        }
    else:
        checks[f"member_M0_bss_vs_year_gradient_{years_suffix(years)}"] = {
            "skipped": True, "reason": "year_gradient_ablation.json not found"}

    # ---- cross-check the bootstrap against the vetted function ------------
    cv_cases = []
    for m, c in [("lora16_sf", "C2"), ("temp2", "C1")]:
        ref = vetted.boot_delta_fixed_year(sub[f"{c}_{m}"], sub[f"M0_{m}"], ys, ss,
                                           np.ones(len(ys), dtype=bool))
        mine = panel[m][c]
        cv_cases.append({
            "member": m,
            "condition": c,
            "vetted_delta_cond_minus_M0": ref["delta"],
            "delta_absdiff": abs(mine["delta"] + ref["delta"]),
            "ci_lo_absdiff": abs(mine["ci_lo"] + ref["ci_hi"]),
            "ci_hi_absdiff": abs(mine["ci_hi"] + ref["ci_lo"]),
            "p_boot_absdiff": abs(mine["p_boot"] - ref["p_boot"]),
        })
    cv_max = max(max(c["delta_absdiff"], c["ci_lo_absdiff"], c["ci_hi_absdiff"],
                     c["p_boot_absdiff"]) for c in cv_cases)
    checks["bootstrap_crosscheck_vs_ablation_worst_year"] = {
        "cases": cv_cases,
        "max_abs_diff": float(cv_max),
        "passed_lt_1e_12": bool(cv_max < 1e-12),
        "note": "With the same seed and resampling indices, this script's Δ(M0−cond) is exactly "
                "the negative of the vetted Δ(cond−M0); 90% CI endpoints mirror and p_boot is identical.",
    }

    # ---- pooled (full 3,004-row test) reference, read from eval/ ----------
    pooled_ref: dict[str, dict] = {}
    for m in MEMBERS:
        e = json.loads((EVAL_DIR / f"M0_ens_lbB_{m}.json").read_text())
        pooled_ref[m] = {}
        for c in CONDS:
            st = e["paired_bootstrap"][f"abl_{c}_ens_lbB_{m}"]
            pooled_ref[m][c] = {
                "delta": float(st["delta"]),
                "ci_lo": float(st["ci_lo"]),
                "ci_hi": float(st["ci_hi"]),
                "p_boot": float(st["p_boot"]),
                "p_holm": float(st["p_holm"]),
                "reject_holm": bool(st["reject_holm"]),
                "n_rows": int(st["n_rows"]),
                "n_clusters": int(st["n_clusters"]),
            }

    # ---- summary ----------------------------------------------------------
    def _counts(table: dict, key_holm: str = "reject_holm") -> dict:
        pos = [f"{m}:{c}" for m in MEMBERS for c in CONDS
               if table[m][c][key_holm] and table[m][c]["delta"] > 0]
        neg = [f"{m}:{c}" for m in MEMBERS for c in CONDS
               if table[m][c][key_holm] and table[m][c]["delta"] < 0]
        return {"n_sig_positive": len(pos), "n_sig_negative": len(neg),
                "n_sig_total": len(pos) + len(neg),
                "significant_positive": pos,
                "significant_negative": neg}

    by_cond = {}
    for c in CONDS:
        by_cond[c] = {
            "mean_delta": float(np.mean([panel[m][c]["delta"] for m in MEMBERS])),
            "n_positive": int(sum(1 for m in MEMBERS if panel[m][c]["delta"] > 0)),
            "n_sig_positive": int(sum(1 for m in MEMBERS
                                     if panel[m][c]["reject_holm"] and panel[m][c]["delta"] > 0)),
            "n_sig_negative": int(sum(1 for m in MEMBERS
                                     if panel[m][c]["reject_holm"] and panel[m][c]["delta"] < 0)),
            "per_member_delta": {m: panel[m][c]["delta"] for m in MEMBERS},
            "per_member_ci": {m: [panel[m][c]["ci_lo"], panel[m][c]["ci_hi"]]
                              for m in MEMBERS},
            "per_member_p_holm": {m: panel[m][c]["p_holm"] for m in MEMBERS},
        }

    summary = {
        "contrast": "ΔBSS_dev = BSS_dev(M0) − BSS_dev(cond); positive values favour the causal component",
        "n_contrasts": len(MEMBERS) * len(CONDS),
        "holm_family": "per member, the 8 conditions; alpha = 0.10",
        "subset_all_conditions": _counts(panel),
        "pooled_all_conditions": _counts(pooled_ref),
        "subset_by_condition": by_cond,
        "C2": {
            "per_member_subset": {
                m: {k: panel[m]["C2"][k] for k in
                    ("delta", "ci_lo", "ci_hi", "p_boot", "p_holm", "reject_holm")}
                for m in MEMBERS},
            "per_member_pooled": {
                m: {k: pooled_ref[m]["C2"][k] for k in
                    ("delta", "ci_lo", "ci_hi", "p_boot", "p_holm", "reject_holm")}
                for m in MEMBERS},
            "mean_delta_subset": by_cond["C2"]["mean_delta"],
            "mean_delta_pooled": float(np.mean([pooled_ref[m]["C2"]["delta"]
                                                for m in MEMBERS])),
        },
    }

    checks["all_passed"] = bool(
        checks["manifest_hash_verified_all_npz"]
        and checks["all_split_test"]
        and checks["all_protocol_holdout_masked_v1"]
        and checks["subset"]["rows_match"]
        and checks["subset"]["storms_match"]
        and checks[f"member_M0_bss_vs_year_gradient_{years_suffix(years)}"].get(
            "passed", False)
        and checks["bootstrap_crosscheck_vs_ablation_worst_year"]["passed_lt_1e_12"]
    )

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/05_eval/ablation_subset_panel.py",
        "protocol": "holdout_masked_v1",
        "read_only": True,
        "method_source": (
            "scripts/05_eval/ablation_worst_year.py::load_frozen (split, protocol, "
            "and manifest SHA-256 checks), DEV_RATE, and MEMBERS; "
            "src/wpac_warning/evaluate.py::bss, paired_cluster_bootstrap, and holm_adjust. "
            "The latter is the same function used by scripts/05_eval/eval_holdout.py to "
            "create eval/M0_*.json → paired_bootstrap, so p_boot has the same definition."),
        "sources": {
            "pred_dir": "data/test/holdout_masked_v1",
            "eval_dir": "data/analysis/eval",
            "year_gradient": "data/analysis/year_gradient_ablation.json",
            "n_members": len(MEMBERS),
            "members": MEMBERS,
            "conditions": CONDS,
            "n_test_rows_full": int(len(y)),
            "subset_years": list(years),
            "period_label": period_label(years),
        },
        "settings": {
            "bss_margin": "BSS_dev",
            "dev_rate": DEV_RATE,
            "dev_rate_source": "eval/main_2024.json → dev_rate.dev_rate (2000-2019)",
            "bs_clim": ("BS_clim = π² + (1−2π)·ȳ_subset, ȳ_subset = "
                        f"{'-'.join(str(v) for v in years)} observed rate"),
            "bootstrap": N_BOOT,
            "seed": SEED,
            "alpha": ALPHA,
            "multiplicity": "holm_bonferroni, family = the 8 conditions of one member",
            "cluster": "sid (whole-storm resampling, paired across the two conditions)",
            "p_boot": "2*min(P(draw<=0), P(draw>=0)), clipped at 1/(B+1)",
            "y_reference": "main.npz (row order of the frozen test set)",
            "ensemble_definition": "equal-weight mean of p over the same 5 members",
        },
        "checks": checks,
        "subset": checks["subset"],
        "member_M0_bss_subset": {m: m0_block[m] for m in MEMBERS},
        "ensemble_M0_bss_subset": m0_block[ENS],
        "member_cond_bss_subset": cond_block,
        "panel": panel,
        "reference_pooled_full_test": pooled_ref,
        "summary": summary,
    }
    out_json.write_text(json.dumps(out, indent=2, sort_keys=False), encoding="utf-8")
    print(f"wrote {out_json.relative_to(P2_ROOT)}  (years {list(years)}, key {pk})")
    print(f"subset rows = {n_rows} (expected {exp_rows}), storms = {n_storms}, "
          f"rate = {ys.mean():.4f}, BS_clim = {checks['subset']['bs_clim_dev']:.4f}")
    print(f"checks: hashes={checks['manifest_hash_verified_all_npz']} "
          f"split={checks['all_split_test']} "
          f"M0-vs-year_gradient="
          f"{checks[f'member_M0_bss_vs_year_gradient_{years_suffix(years)}'].get('max_abs_diff'):.2e} "
          f"bootstrap-xcheck={cv_max:.2e} all_passed={checks['all_passed']}")
    sc, sp = summary["subset_all_conditions"], summary["pooled_all_conditions"]
    print(f"Holm-significant contrasts: subset {sc['n_sig_total']}/40 "
          f"(+{sc['n_sig_positive']}/-{sc['n_sig_negative']}), "
          f"pooled {sp['n_sig_total']}/40 (+{sp['n_sig_positive']}/-{sp['n_sig_negative']})")
    for m in MEMBERS:
        p = panel[m]["C2"]
        q = pooled_ref[m]["C2"]
        print(f"  {m:18s} C2 subset {p['delta']:+.4f} "
              f"[{p['ci_lo']:+.4f},{p['ci_hi']:+.4f}] p_holm={p['p_holm']:.4f} "
              f"{'SIG' if p['reject_holm'] else 'n.s.'} | "
              f"pooled {q['delta']:+.4f} [{q['ci_lo']:+.4f},{q['ci_hi']:+.4f}] "
              f"{'SIG' if q['reject_holm'] else 'n.s.'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", default=",".join(str(v) for v in SUBSET_YEARS),
                    help="comma-separated ISO_TIME years to pool, e.g. 2020 or 2020,2021 "
                         "(default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: data/analysis/ablation_<years>_panel.json)")
    args = ap.parse_args()
    main(tuple(int(v) for v in args.years.split(",")), args.out)
