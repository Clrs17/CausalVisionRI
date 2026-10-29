# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""RITCF FINAL ensemble evaluation on test years 2020/2021 (evaluate once).

Ensemble = uniform average of the 5 selected final arms' probabilities.
Per test year y: members' fold-y models predict y; threshold F1-tuned on the
ensemble's inner (2017-2019) probabilities of the same fold (training
convention). Metrics: POD, FARate FP/(FP+TN), FARatio FP/(TP+FP),
BSS (year's own climatology), PR-AUC, ROC-AUC.
Sensitivity: (a) per-arm scores; (b) cross-fold — year y evaluated with the
OTHER seed replicate (fold trained with the other test year's seed; both
share the identical 00-19 training set).
RITCF (PNAS 2025) reference: POD 0.91 / FARate 0.089 (BSS ≈ -0.49).

Run from the release root.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PREDS = Path(__file__).resolve().parents[2] / "data" / "preds"
ARMS = ["fin_lbB_lora16_sf", "fin_ep40_temp5", "fin_v1",
        "fin_lbB_lora16_temp5", "fin_lbL_lora16_sf"]
YEARS = [2020, 2021]


def bss(p, y):
    bs = float(np.mean((p - y) ** 2))
    return 1.0 - bs / max(float(np.mean((y.mean() - y) ** 2)), 1e-9)


def pod_fares(p, y, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    pod = tp / max(tp + fn, 1)
    farate = fp / max(fp + tn, 1)
    faratio = fp / max(tp + fp, 1)
    return pod, farate, faratio, (tp, fp, fn, tn)


def f1_thr(p, y):
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.025):
        pred = p >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)


data = {}
for a in ARMS:
    for y in YEARS:
        d = np.load(PREDS / f"{a}_fold{y}.npz")
        data[(a, y)] = {k: d[k].astype(float) for k in ("p_held", "y_held", "p_inner", "y_inner")}

print(f"arms: {len(ARMS)}, years: {YEARS}\n")

# ---- per-arm table ----
print("== per-arm held-year scores (own fold model, arm's own threshold) ==")
rows = []
for a in ARMS:
    for y in YEARS:
        d = data[(a, y)]
        thr = f1_thr(d["p_inner"], d["y_inner"])
        pod, frate, fratio, _ = pod_fares(d["p_held"], d["y_held"], thr)
        rows.append(dict(arm=a, year=y, POD=pod, FARate=frate, FARatio=fratio,
                         BSS=bss(d["p_held"], d["y_held"]), thr=thr))
pa = pd.DataFrame(rows)
print(pa.round(3).to_string(index=False))

# ---- ensemble per year + pooled + cross-fold sensitivity ----
print("\n== FINAL ensemble (uniform 5 arms) ==")
ens_rows = []
for y in YEARS:
    other = YEARS[1] - YEARS.index(y) if False else (2021 if y == 2020 else 2020)
    for tag, fy in (("own-fold", y), ("cross-fold(sens)", other)):
        p_ens = np.mean([data[(a, fy)]["p_held"] for a in ARMS], axis=0)
        y_true = data[(ARMS[0], fy)]["y_held"]
        p_in = np.mean([data[(a, fy)]["p_inner"] for a in ARMS], axis=0)
        y_in = data[(ARMS[0], fy)]["y_inner"]
        thr = f1_thr(p_in, y_in)
        pod, frate, fratio, counts = pod_fares(p_ens, y_true, thr)
        ens_rows.append(dict(year=y, model=tag, POD=pod, FARate=frate,
                             FARatio=fratio, BSS=bss(p_ens, y_true),
                             PR_AUC=average_precision_score(y_true, p_ens),
                             ROC_AUC=roc_auc_score(y_true, p_ens),
                             thr=thr, n=len(y_true), ri=y_true.mean(),
                             tpfpfn_tn=counts))
ed = pd.DataFrame(ens_rows)
print(ed.drop(columns=["tpfpfn_tn"]).round(3).to_string(index=False))

# pooled both years (own-fold models)
p_all = np.concatenate([np.mean([data[(a, y)]["p_held"] for a in ARMS], axis=0) for y in YEARS])
y_all = np.concatenate([data[(ARMS[0], y)]["y_held"] for y in YEARS])
p_in_all = np.concatenate([np.mean([data[(a, y)]["p_inner"] for a in ARMS], axis=0) for y in YEARS])
y_in_all = np.concatenate([data[(ARMS[0], y)]["y_inner"] for y in YEARS])
thr = f1_thr(p_in_all, y_in_all)
pod, frate, fratio, counts = pod_fares(p_all, y_all, thr)
print(f"\n== POOLED 2020+2021 (n={len(y_all)}, RI={y_all.mean():.3f}, thr={thr}) ==")
print(f"POD={pod:.3f}  FARate={frate:.3f}  FARatio={fratio:.3f}  "
      f"BSS={bss(p_all, y_all):.3f}  PR-AUC={average_precision_score(y_all, p_all):.3f}  "
      f"ROC-AUC={roc_auc_score(y_all, p_all):.3f}")
print(f"counts TP/FP/FN/TN = {counts}")
print("\nRITCF (PNAS 2025) reference: POD=0.910 FARate=0.089 (BSS≈-0.49)")

out = pd.concat([pa.assign(section="per_arm"),
                 ed.assign(section="ensemble")], ignore_index=True, sort=False)
out.to_csv(Path(__file__).resolve().parents[2] / "data" / "ritcf_final_eval.csv", index=False)
print("\nsaved data/ritcf_final_eval.csv")
