# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Offline threshold policy + calibration analysis on saved ensemble probs.

For every LOYO fold, loads the 4 arms' saved probs (data/preds/*.npz),
averages them (held + inner), then compares threshold SELECTION strategies —
all fitted on INNER only, evaluated on held once:

  f1      : maximize F1 on inner (current production rule)
  peirce  : maximize Peirce score (POD - FARate) on inner
  far45   : maximize POD subject to FARatio <= 0.45 on inner
  far50   : maximize POD subject to FARatio <= 0.50 on inner

Plus a calibration study: Platt (logistic) and isotonic calibration fitted on
inner-ens probs, applied to held-ens probs -> BSS before/after.

FAR for RI is currently high (~0.55-0.6) because F1-optimal thresholds are
aggressive for rare events; this quantifies what a BSS/constraint-oriented
policy buys.  CPU-only, offline.

Usage (from the release root):
    python scripts/04_train/threshold_opt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]
RUNS = ["ens_v1", "ens_lingbot", "ens_ep40", "ens_ep40ft"]


def counts(p, y, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def metrics(p, y, thr):
    tp, fp, fn, tn = counts(p, y, thr)
    pod = tp / max(tp + fn, 1)
    faratio = fp / max(tp + fp, 1)
    farate = fp / max(fp + tn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    peirce = pod - farate
    return pod, faratio, farate, f1, peirce


def pick(p_in, y_in, rule):
    grid = np.linspace(0.03, 0.9, 88)
    if rule == "f1":
        vals = [metrics(p_in, y_in, t)[3] for t in grid]
    elif rule == "peirce":
        vals = [metrics(p_in, y_in, t)[4] for t in grid]
    elif rule in ("far45", "far50"):
        cap = 0.45 if rule == "far45" else 0.50
        vals = [m[0] if m[1] <= cap else -1 for m in (metrics(p_in, y_in, t) for t in grid)]
    return float(grid[int(np.argmax(vals))])


def platt(p, y):
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000).fit(x, y)
    return lr


def main() -> int:
    pred_dir = P2_ROOT / "data" / "preds"
    years = sorted({int(p.stem.split("fold")[1]) for p in pred_dir.glob(f"{RUNS[0]}_fold*.npz")})
    rows, cal_rows = [], []
    for yr in years:
        ps, ys, pins, yins = [], [], [], []
        for run in RUNS:
            d = np.load(pred_dir / f"{run}_fold{yr}.npz")
            ps.append(d["p_held"]); ys.append(d["y_held"])
            pins.append(d["p_inner"]); yins.append(d["y_inner"])
        y, yi = ys[0], yins[0]
        p_ens = np.mean(ps, axis=0)
        p_in_ens = np.mean(pins, axis=0)
        rec = {"held_year": yr, "n": len(y), "ri_rate": round(float(y.mean()), 3)}
        for rule in ("f1", "peirce", "far45", "far50"):
            thr = pick(p_in_ens, yi, rule)
            pod, faratio, farate, f1, peirce = metrics(p_ens, y, thr)
            rec[f"{rule}_thr"] = round(thr, 2)
            rec[f"{rule}_pod"] = round(pod, 3)
            rec[f"{rule}_far"] = round(faratio, 3)
            rec[f"{rule}_peirce"] = round(peirce, 3)
        rows.append(rec)

        # calibration (fit on inner only)
        bs_raw = float(np.mean((p_ens - y) ** 2))
        clim = float(np.mean((y.mean() - y) ** 2))
        lr = platt(p_in_ens, yi)
        x_h = np.log(np.clip(p_ens, 1e-6, 1 - 1e-6) / (1 - np.clip(p_ens, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        p_cal = lr.predict_proba(x_h)[:, 1]
        bs_cal = float(np.mean((p_cal - y) ** 2))
        cal_rows.append({"held_year": yr,
                         "bss_raw": round(1 - bs_raw / clim, 4),
                         "bss_platt": round(1 - bs_cal / clim, 4)})

    df = pd.DataFrame(rows).set_index("held_year")
    cal = pd.DataFrame(cal_rows).set_index("held_year")
    print("=== threshold policies (fit on inner, eval on held) ===")
    print(df.to_string())
    print("\nmeans:")
    print(df.mean().round(3).to_string())
    print("\n=== calibration (Platt on inner) ===")
    print(cal.to_string())
    print("means:", cal.mean().round(4).to_dict())
    out = P2_ROOT / "data" / "threshold_opt.csv"
    df.join(cal).to_csv(out)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
