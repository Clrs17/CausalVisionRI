# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""WRPS1-style comparison: evaluate the RITCF FINAL ensemble on a sample
filtered per Horinouchi et al. (2025, WAF) §2a:
  * drop each storm's first record (INCV unavailable, their filter)
  * ocean-only at t0: drop rows with LAND_DIST <= 0 (their INCV-missing proxy)
  * (their last-18h trim is subsumed by our stricter -24h label horizon)
Sample stats + POD / FARate / FARatio / BSS per test year and pooled.
Reference: WRPS1 CV BSS = 0.225 +/- 0.035 (WNP 2002-21 all-TC, 9648 recs,
RI 10.6%); their LDA CV BSS = 0.140.

Run from the release root.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P2 = str(Path(__file__).resolve().parents[2])
ARMS = ["fin_lbB_lora16_sf", "fin_ep40_temp5", "fin_v1",
        "fin_lbB_lora16_temp5", "fin_lbL_lora16_sf"]
YEARS = [2020, 2021]

import sys
sys.path.insert(0, f"{P2}/src")
from wpac_warning.utils_cfg import load_cfg
from wpac_warning.dataset import RIPanelDataset

cfg = load_cfg(f"{P2}/configs/phase2/config_ritcf_final.yaml")
ds = RIPanelDataset(cfg)
df = ds.df.copy()  # exact row order used by train.py
df["Y"] = pd.to_datetime(df["ISO_TIME"], utc=True).dt.year
print(f"dataset rows: {len(df)}, cols ok: {'LAND_DIST' in df.columns}")


def bss(p, y):
    bs = float(np.mean((p - y) ** 2))
    return 1.0 - bs / max(float(np.mean((y.mean() - y) ** 2)), 1e-9)


def pod_fares(p, y, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp / max(tp + fn, 1), fp / max(fp + tn, 1), fp / max(tp + fp, 1), (tp, fp, fn, tn)


def f1_thr(p, y):
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.025):
        pred = p >= t
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)


# per-year ensemble probabilities in dataset order
results = []
for y in YEARS:
    d = np.load(f"{P2}/data/preds/fin_v1_fold{y}.npz")
    n_held = len(d["y_held"])
    sub = df[df.Y == y].reset_index(drop=True)
    assert len(sub) == n_held, f"row mismatch {y}: {len(sub)} vs {n_held}"
    p_ens = np.mean([np.load(f"{P2}/data/preds/{a}_fold{y}.npz")["p_held"].astype(float)
                     for a in ARMS], axis=0)
    p_in = np.mean([np.load(f"{P2}/data/preds/{a}_fold{y}.npz")["p_inner"].astype(float)
                    for a in ARMS], axis=0)
    y_in = np.load(f"{P2}/data/preds/fin_v1_fold{y}.npz")["y_inner"].astype(float)
    thr = f1_thr(p_in, y_in)
    sub["p"] = p_ens
    sub["y"] = d["y_held"].astype(float)

    # WRPS1 filters
    first_idx = sub.groupby("SID")["ISO_TIME"].transform("min") == sub["ISO_TIME"]
    wr = sub[(~first_idx) & (sub["LAND_DIST"] > 0)]
    results.append((y, thr, sub, wr))

print("\n== sample stats (WRPS1-filtered) ==")
for y, thr, sub, wr in results:
    print(f"{y}: raw n={len(sub)} (RI {sub.y.mean():.3f}) -> filtered n={len(wr)} "
          f"(RI {wr.y.mean():.3f})  [dropped first-record {(sub.groupby('SID')['ISO_TIME'].transform('min') == sub['ISO_TIME']).sum()}, land {(sub['LAND_DIST'] <= 0).sum()}]")

print("\n== FINAL ensemble on WRPS1-filtered sample ==")
print("year   thr    POD  FARate FARatio   BSS   PR-AUC   n    ri_rate")
from sklearn.metrics import average_precision_score
pooled_p, pooled_y = [], []
for y, thr, sub, wr in results:
    pod, frate, fratio, _ = pod_fares(wr.p.values, wr.y.values, thr)
    print(f"{y}  {thr:.3f}  {pod:.3f}  {frate:.3f}  {fratio:.3f}  "
          f"{bss(wr.p.values, wr.y.values):.3f}  "
          f"{average_precision_score(wr.y.values, wr.p.values):.3f}  {len(wr)}  {wr.y.mean():.3f}")
    pooled_p.append(wr.p.values); pooled_y.append(wr.y.values)

p_all = np.concatenate(pooled_p); y_all = np.concatenate(pooled_y)
p_in_all = np.concatenate([
    np.mean([np.load(f"{P2}/data/preds/{a}_fold{y}.npz")["p_inner"].astype(float) for a in ARMS], axis=0)
    for y in YEARS])
y_in_all = np.concatenate([np.load(f"{P2}/data/preds/fin_v1_fold{y}.npz")["y_inner"].astype(float) for y in YEARS])
thr = f1_thr(p_in_all, y_in_all)
pod, frate, fratio, counts = pod_fares(p_all, y_all, thr)
print(f"\nPOOLED (thr={thr:.3f}, n={len(y_all)}, RI={y_all.mean():.3f}): "
      f"POD={pod:.3f} FARate={frate:.3f} FARatio={fratio:.3f} BSS={bss(p_all, y_all):.3f} "
      f"PR-AUC={average_precision_score(y_all, p_all):.3f}")
print(f"counts TP/FP/FN/TN = {counts}")
print("\nWRPS1 reference: CV BSS=0.225+/-0.035, LDA CV BSS=0.140 "
      "(WNP 2002-21, 9648 recs, RI 10.6%)")

pd.DataFrame({"p": p_all, "y": y_all}).to_csv(
    f"{P2}/data/ritcf_final_wrps1style_pooled.csv", index=False)
print("saved data/ritcf_final_wrps1style_pooled.csv")
