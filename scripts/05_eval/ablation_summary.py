# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Legacy fold-based ablation summary for the pre-holdout prediction dumps.

This entry point reads ``data/preds`` and ``data/preds_server``, which are not
part of the current release. It writes member and ensemble matrices under
``data/``. For the current 40-contrast holdout panel, use
``scripts/05_eval/eval_holdout.py`` and ``scripts/05_eval/make_sm_tables.py``.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED_DIRS = [ROOT / "data" / "preds", ROOT / "data" / "preds_server"]
LOG_DIR = ROOT / "logs" / "server"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

MEMBERS = ["ep40srv", "lingbotL", "lbBtemp2", "v1", "lbB_lora16sf"]
M0_RUN = {
    "ep40srv": "abl_M0_ep40srv",
    "lingbotL": "abl_M0_lingbotL",
    "lbBtemp2": "ens_lbB_temp2",
    "v1": "abl_M0_v1",
    "lbB_lora16sf": "ens_lbB_lora16_sf",
}
CONDITIONS = ["C1", "C2", "C3s1", "C3s2", "C3s3", "C4adj", "C4sign", "C4cf"]


def bss(p: np.ndarray, y: np.ndarray) -> float:
    bs = float(np.mean((p - y) ** 2))
    bs_clim = float(np.mean((y.mean() - y) ** 2))
    return 1.0 - bs / max(bs_clim, 1e-9)


def load_run_probs(run: str):
    """Return {year: (p, y)} for a run across the two pred dirs."""
    out = {}
    for y_ in YEARS:
        for d in PRED_DIRS:
            f = d / f"{run}_fold{y_}.npz"
            if f.exists():
                z = np.load(f)
                keys = z.files
                p = z["p_ri"] if "p_ri" in keys else z["prob"] if "prob" in keys else z[keys[0]]
                y = z["y_ri"] if "y_ri" in keys else z["y"] if "y" in keys else z[keys[1]]
                out[y_] = (np.asarray(p, float), np.asarray(y, float))
                break
    return out


def arm_fold_bss(run: str) -> dict:
    return {y: bss(p, yv) for y, (p, yv) in load_run_probs(run).items()}


def summarize(bss_by_year: dict) -> tuple[float, float]:
    vals = [bss_by_year[y] for y in YEARS if y in bss_by_year]
    return float(np.mean(vals)), float(np.min(vals))


def gate_mean(run: str) -> float | None:
    f = LOG_DIR / f"train_{run}.log"
    if not f.exists():
        return None
    vals = [float(m.group(1)) for m in
            re.finditer(r"finetune ep\d+: .*gate=([0-9.]+)", f.read_text(errors="ignore"))]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    rows = []
    for m in MEMBERS:
        m0_run = M0_RUN[m]
        m0 = arm_fold_bss(m0_run)
        m0_mean, m0_worst = summarize(m0)
        row = {"member": m, "cond": "M0", "mean": m0_mean, "worst": m0_worst,
               "dmean": 0.0, "dworst": 0.0, "gate": gate_mean(m0_run)}
        rows.append(row)
        for c in CONDITIONS:
            run = f"abl_{c}_{m}"
            b = arm_fold_bss(run)
            if not b:
                continue
            mean, worst = summarize(b)
            rows.append({"member": m, "cond": c, "mean": mean, "worst": worst,
                         "dmean": mean - m0_mean, "dworst": worst - m0_worst,
                         "gate": gate_mean(run)})
    df = pd.DataFrame(rows)

    # ensemble level: uniform average of the same 5 members per condition
    ens_rows = []
    for cond in ["M0"] + CONDITIONS:
        per_year = {}
        for y in YEARS:
            ps, ys = [], []
            for m in MEMBERS:
                run = M0_RUN[m] if cond == "M0" else f"abl_{cond}_{m}"
                probs = load_run_probs(run)
                if y in probs:
                    ps.append(probs[y][0])
                    ys.append(probs[y][1])
            if len(ps) == len(MEMBERS):
                y_ref = ys[0]
                assert all((yv == y_ref).all() for yv in ys[1:])
                per_year[y] = bss(np.mean(ps, axis=0), y_ref)
        mean, worst = summarize(per_year)
        ens_rows.append({"cond": cond, "mean": mean, "worst": worst})
    ens = pd.DataFrame(ens_rows)
    m0e = float(ens.loc[ens.cond == "M0", "mean"].iloc[0])
    m0ew = float(ens.loc[ens.cond == "M0", "worst"].iloc[0])
    ens["dmean"] = ens["mean"] - m0e
    ens["dworst"] = ens["worst"] - m0ew

    pd.set_option("display.width", 200)
    print("=== member x condition (mean BSS / worst fold; deltas vs M0) ===")
    for m in MEMBERS:
        sub = df[df.member == m]
        print(f"\n--- {m} (M0 {sub.iloc[0]['mean']:.3f}/{sub.iloc[0]['worst']:.3f}) ---")
        for _, r in sub.iterrows():
            print(f"  {r['cond']:7s} mean={r['mean']:.3f} worst={r['worst']:.3f} "
                  f"d={r['dmean']:+.3f}/{r['dworst']:+.3f} gate={r['gate'] if pd.notna(r['gate']) else float('nan'):.3f}")
    print("\n=== ensemble level (same 5 members, uniform) ===")
    for _, r in ens.iterrows():
        print(f"  {r['cond']:7s} mean={r['mean']:.3f} worst={r['worst']:.3f} "
              f"d={r['dmean']:+.3f}/{r['dworst']:+.3f}")

    df.to_csv(ROOT / "data" / "ablation_member_matrix.csv", index=False)
    ens.to_csv(ROOT / "data" / "ablation_ensemble_matrix.csv", index=False)
    print("\nwrote data/ablation_member_matrix.csv + data/ablation_ensemble_matrix.csv")


if __name__ == "__main__":
    main()
