# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
#!/usr/bin/env python
"""Same-sample WRPS1 (Horinouchi et al. 2025, WAF 40(10)) comparison.

Scores the published WRPS1 knowledge chunks (Supplementary Tables S2/S3,
NF-type t=0-24 h, 20-yr fit) directly on our SHIPS-12 COHC feature table
and evaluates with our metric suite.  WRPS1's fitted coefficients come from
their data; nothing is fitted on our data, so this touches no protocol
asset.  The evaluation subset is rows where all 12 predictors are finite;
the frozen Top-5 ensemble is additionally evaluated on the same subset.

Units: the SHIPS developmental data carries SHRD in 0.1 kt, T150/T200/T250
and SDIR in 0.1 degC, Z850 in 1e-7 s-1.  Our table carries SHRD in kt,
temperatures and SDIR in degC, Z850 in 1e-7 s-1 (same).  Chunk thresholds
below are already converted to our units (division by 10 where needed).

Supplement anomalies recorded: printed row 55 of Table S3 is an exact
duplicate of row 1 (excluded); two rows are both numbered 36 (both kept,
they are distinct conditions).  Negative set = 55 distinct rows
(73 chunks total with the 18 positive ones).

WRPS1 formulas (paper):
  logit(pu) = sum_j w_j x_j + c,  c = -0.3536
  pl = alpha * pu / (1 - (1-alpha) * pu),  alpha = 0.275
  P  = 1.134*pl - 0.296*pl^2     (Eq. 8, K=2)
  deterministic RI when pu >= 0.5  (equivalently pl >= 0.215)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P2))

from wpac_warning.evaluate import auc_metrics, brier  # noqa: E402

C_OFFSET = -0.3536
ALPHA = 0.275
PROB_COEF = (1.134, -0.296)

# --- Table S2: 18 positive chunks (weight > 0) ----------------------------
POS = [
    (0.731, [("COHC", 63, None), ("VMAX", 48, 92), ("T200", -52.5, -49.8),
             ("SHRD", None, 17.0)]),
    (0.405, [("COHC", 106, None), ("VMAX", 28, 112), ("INCV", 2, None),
             ("SHRD", None, 17.0)]),
    (0.283, [("COHC", 97, 131), ("VMAX", 28, 128), ("Z850", None, 181),
             ("SDIR", None, 11.2)]),
    (0.205, [("COHC", 97, 131), ("VMAX", 48, 112), ("T250", -39.6, None),
             ("SDIR", None, 19.0)]),
    (0.196, [("COHC", 97, None), ("POT", 34, 100), ("Z850", None, 181),
             ("SDIR", None, 19.0)]),
    (0.191, [("COHC", 63, 131), ("POT", 34, None), ("INCV", 2, None),
             ("SDIR", None, 11.2)]),
    (0.189, [("COHC", 63, 131), ("VMAX", 48, 112), ("T150", None, -66.3),
             ("SHRD", None, 17.0)]),
    (0.187, [("COHC", 106, None), ("POT", 34, 128), ("SHRD", None, 18.6),
             ("SDIR", None, 19.0)]),
    (0.163, [("COHC", 106, None), ("POT", 34, 128), ("SHRD", None, 17.0),
             ("SDIR", None, 19.0)]),
    (0.160, [("COHC", 63, None), ("POT", 34, 100), ("SHRD", None, 17.0),
             ("SDIR", None, 11.2)]),
    (0.137, [("COHC", 97, 131), ("DTL", 142, None), ("VMAX", 48, 112),
             ("T250", -39.6, None)]),
    (0.113, [("COHC", 63, 131), ("VMAX", 48, 112), ("INCV", 2, None),
             ("SDIR", None, 11.2)]),
    (0.107, [("COHC", 63, None), ("VMAX", 48, 112), ("T150", None, -66.3),
             ("SHRD", None, 17.0)]),
    (0.104, [("COHC", 97, 131), ("VMAX", 28, 112), ("T250", -39.6, None),
             ("SDIR", None, 11.2)]),
    (0.098, [("COHC", 63, 131), ("POT", 34, 100), ("INCV", 2, None),
             ("SDIR", None, 11.2)]),
    (0.049, [("COHC", 97, 131), ("VMAX", 48, 112), ("R000", None, 79),
             ("T250", -39.6, None)]),
    (0.038, [("COHC", 97, 131), ("VMAX", 48, 112), ("Z850", None, 181),
             ("T250", -39.6, -36.2)]),
    (0.033, [("COHC", 63, 131), ("POT", 34, 128), ("INCV", 2, None),
             ("SDIR", None, 11.2)]),
]

# --- Table S3: 55 distinct negative chunks (duplicate row 55 excluded) ---
NEG = [
    (-0.581, [("COHC", None, 37)]),
    (-0.497, [("INCV", None, -2)]),
    (-0.475, [("COHC", None, 37), ("SDIR", 2.5, None)]),
    (-0.405, [("COHC", None, 63), ("INCV", None, 8), ("T150", -64.6, None)]),
    (-0.294, [("INCV", None, 2)]),
    (-0.275, [("COHC", None, 106), ("POT", None, 34), ("T250", -39.6, None)]),
    (-0.273, [("COHC", None, 63), ("INCV", None, 2), ("Z850", None, 181),
              ("SDIR", 11.2, None)]),
    (-0.272, [("INCV", None, 2), ("SHRD", 11.5, None)]),
    (-0.250, [("COHC", None, 97), ("DTL", None, 525), ("Z850", None, 181),
              ("T200", -53.3, None)]),
    (-0.238, [("INCV", None, 8)]),
    (-0.200, [("COHC", None, 37), ("INCV", None, 8)]),
    (-0.197, [("COHC", None, 97), ("DTL", None, 525), ("Z850", None, 181)]),
    (-0.196, [("COHC", None, 106), ("INCV", None, 8), ("T150", -66.3, None),
              ("SDIR", 11.2, None)]),
    (-0.194, [("COHC", None, 106), ("POT", None, 34), ("INCV", None, 22),
              ("T250", -39.6, None)]),
    (-0.185, [("COHC", None, 131), ("POT", None, 34)]),
    (-0.181, [("COHC", None, 37), ("T200", -52.5, None)]),
    (-0.177, [("COHC", None, 63), ("Z850", None, 181), ("T150", -66.3, None),
              ("SHRD", 11.5, None)]),
    (-0.170, [("DTL", None, 1509), ("INCV", None, 8), ("SHRD", 11.5, None)]),
    (-0.164, [("COHC", None, 106), ("POT", None, 34)]),
    (-0.153, [("DTL", None, 1223), ("POT", None, 59), ("INCV", None, 8),
              ("SDIR", 2.5, None)]),
    (-0.148, [("COHC", None, 106), ("SHRD", 11.5, None)]),
    (-0.146, [("POT", None, 34)]),
    (-0.145, [("COHC", None, 106), ("INCV", None, 8), ("T150", -64.6, None)]),
    (-0.143, [("COHC", None, 131), ("POT", None, 34), ("T200", -52.5, None)]),
    (-0.124, [("DTL", None, 1223), ("INCV", None, 2), ("Z850", None, 181),
              ("SHRD", 11.5, None)]),
    (-0.124, [("DTL", None, 1515), ("POT", None, 34), ("INCV", None, 22),
              ("T150", -66.3, None)]),
    (-0.121, [("DTL", None, 1509), ("POT", None, 34), ("INCV", None, 22),
              ("T150", -66.3, None)]),
    (-0.116, [("INCV", None, 2), ("SDIR", 11.2, None)]),
    (-0.106, [("COHC", None, 106), ("POT", None, 34), ("T200", -52.5, None)]),
    (-0.088, [("POT", None, 34), ("T200", -52.5, None)]),
    (-0.086, [("COHC", None, 106), ("POT", None, 34), ("INCV", None, 22)]),
    (-0.081, [("DTL", None, 525), ("INCV", None, 2)]),
    (-0.080, [("COHC", None, 106), ("INCV", None, -2)]),
    (-0.070, [("COHC", None, 97), ("DTL", None, 525), ("T200", -53.3, None)]),
    (-0.067, [("DTL", None, 1223), ("INCV", None, 8), ("Z850", None, 181),
              ("SHRD", 11.5, None)]),
    (-0.065, [("COHC", None, 131), ("POT", None, 34), ("T250", -39.6, None)]),
    (-0.058, [("POT", None, 128), ("INCV", None, 8), ("Z850", None, 181),
              ("T150", -65.3, None)]),
    (-0.053, [("COHC", None, 106), ("DTL", None, 1223), ("POT", None, 34),
              ("INCV", None, 22)]),
    (-0.049, [("COHC", None, 131), ("POT", None, 34), ("INCV", None, 22)]),
    (-0.034, [("COHC", None, 106), ("POT", None, 34), ("INCV", None, 22),
              ("T200", -52.5, None)]),
    (-0.028, [("COHC", None, 106), ("POT", None, 34), ("T250", -39.6, None),
              ("SDIR", 2.5, None)]),
    (-0.027, [("POT", None, 34), ("T150", -66.3, None)]),
    (-0.025, [("COHC", None, 106), ("POT", None, 34), ("T250", -39.6, None),
              ("SDIR", None, 26.4)]),
    (-0.025, [("DTL", None, 1515), ("POT", None, 34), ("T200", -52.5, None),
              ("SDIR", 2.5, None)]),
    (-0.025, [("COHC", None, 37), ("INCV", None, 8), ("T200", -52.5, None)]),
    (-0.022, [("DTL", None, 1509), ("POT", None, 34), ("T200", -52.5, None),
              ("SDIR", 2.5, None)]),
    (-0.019, [("DTL", None, 1515), ("POT", None, 34), ("T200", -52.5, None),
              ("SDIR", None, 26.4)]),
    (-0.018, [("COHC", None, 106), ("DTL", None, 525), ("Z850", None, 181),
              ("T200", -53.3, None)]),
    (-0.017, [("DTL", None, 1223), ("POT", None, 34), ("INCV", None, 22)]),
    (-0.016, [("POT", None, 34), ("T250", -39.6, None)]),
    (-0.016, [("DTL", None, 1509), ("POT", None, 34), ("T200", -52.5, None),
              ("SDIR", None, 26.4)]),
    (-0.012, [("DTL", None, 1509), ("POT", None, 34), ("INCV", None, 22),
              ("T200", -52.5, None)]),
    (-0.008, [("DTL", None, 1515), ("POT", None, 34), ("INCV", None, 22),
              ("T200", -52.5, None)]),
    (-0.005, [("COHC", None, 131), ("POT", None, 34), ("INCV", None, 22),
              ("T200", -52.5, None)]),
    (-0.001, [("POT", None, 34), ("INCV", None, 22)]),
]

CHUNKS = POS + NEG
PREDICTORS = ("COHC", "DTL", "VMAX", "POT", "INCV", "SDIR", "SHRD",
              "T150", "T200", "T250", "Z850", "R000")


def wrps1_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return pu, pl, P (probability) for each row (NaN-free predictors)."""
    logit = np.full(len(df), C_OFFSET, dtype=float)
    fired = np.zeros((len(df), len(CHUNKS)), dtype=bool)
    for j, (w, conds) in enumerate(CHUNKS):
        ok = np.ones(len(df), dtype=bool)
        for var, lo, hi in conds:
            v = df[var].to_numpy(dtype=float)
            if lo is not None:
                ok &= v >= lo
            if hi is not None:
                ok &= v < hi
        fired[:, j] = ok
        logit += w * ok
    pu = 1.0 / (1.0 + np.exp(-logit))
    pl = ALPHA * pu / (1.0 - (1.0 - ALPHA) * pu)
    p = PROB_COEF[0] * pl + PROB_COEF[1] * pl ** 2
    p = np.clip(p, 0.0, 1.0)
    return pd.DataFrame({"pu": pu, "pl": pl, "p": p,
                         "n_chunks_fired": fired.sum(axis=1)})


def _metrics(p, y, thr, dev_rate):
    from wpac_warning.evaluate import classification_metrics, bss
    m = classification_metrics(p, y, thr)
    m["BSS_dev"] = bss(p, y, clim_rate=dev_rate)
    m.update(auc_metrics(p, y))
    return m


def main() -> int:
    dev_rate = 0.0791
    df = pd.read_parquet(
        P2 / "data" / "holdout_masked_v1" /
        "ships12_predictors_fit0017_cohc.parquet")
    panel = pd.read_parquet(
        P2 / "data" / "holdout_masked_v1" / "panel_fit0017.parquet",
        columns=["SID", "ISO_TIME", "storm_year", "RI"])
    panel["ISO_TIME"] = pd.to_datetime(panel["ISO_TIME"], utc=True)
    panel["row_uid"] = (panel["SID"].astype(str) + "|" +
                        panel["ISO_TIME"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    df["row_uid"] = df["row_uid"].astype(str)
    m = panel.merge(df, on="row_uid", how="inner", validate="1:1")

    finite = np.ones(len(m), dtype=bool)
    for v in PREDICTORS:
        finite &= np.isfinite(m[v].to_numpy(dtype=float))
    print(f"rows total {len(m)}, all-12-finite {int(finite.sum())}, "
          f"excluded by NaN {int((~finite).sum())}")
    ev = m[finite].reset_index(drop=True)

    sc = wrps1_score(ev)
    y = ev["RI"].astype(int).to_numpy()
    years = ev["storm_year"].astype(int).to_numpy()

    # frozen Top-5 ensemble probabilities on the same rows
    z = np.load(P2 / "data" / "test" / "holdout_masked_v1" / "main.npz",
                allow_pickle=True)
    zmap = {str(k): i for i, k in enumerate(z["row_uid"])}
    ens_p = np.full(len(ev), np.nan)
    for i, k in enumerate(ev["row_uid"].astype(str)):
        j = zmap.get(k)
        if j is not None:
            ens_p[i] = z["p"][j]
    has_ens = np.isfinite(ens_p)

    # inner-F1 threshold for WRPS1 (same rule as our protocol: inner only)
    inner_mask = (years >= 2018) & (years <= 2019)
    from wpac_warning.evaluate import f1_threshold
    thr_f1 = float(f1_threshold(sc["p"].to_numpy()[inner_mask],
                                y[inner_mask]))

    out = {"kind": "wrps1_same_sample_comparison",
           "n_rows": int(len(ev)),
           "n_with_ensemble": int(has_ens.sum()),
           "ensemble_coverage": "test rows only (frozen predictions)",
           "thresholds": {"wrps1_own_rule_pl_on_pl": 0.215,
                          "inner_f1_on_P": float(thr_f1)}}
    for tag, mask in (("all", np.ones(len(ev), bool)),
                      ("fit", years <= 2017),
                      ("inner", (years >= 2018) & (years <= 2019)),
                      ("test", years >= 2020)):
        row = {"n": int(mask.sum()), "base_rate": float(y[mask].mean())}
        # WRPS1's own deterministic rule: pu >= 0.5 == pl >= 0.215 (on pl,
        # not on the recalibrated P)
        w_own = _metrics(sc["pl"].to_numpy()[mask], y[mask], 0.215, dev_rate)
        w_own.update(auc_metrics(sc["p"].to_numpy()[mask], y[mask]))
        row["wrps1_pu0.5"] = w_own
        row["wrps1_innerF1"] = _metrics(
            sc["p"].to_numpy()[mask], y[mask], thr_f1, dev_rate)
        row["ensemble_innerF1"] = _metrics(
            ens_p[mask & has_ens], y[mask & has_ens], 0.225, dev_rate)
        out[tag] = row
    # test years breakdown
    out["test_by_year"] = {}
    for yr in range(2020, 2025):
        mask = (years == yr) & (years >= 2020)
        if mask.sum():
            out["test_by_year"][str(yr)] = {
                "n": int(mask.sum()),
                "wrps1_innerF1": _metrics(sc["p"].to_numpy()[mask], y[mask],
                                          thr_f1, dev_rate),
                "ensemble": _metrics(ens_p[mask & has_ens],
                                     y[mask & has_ens], 0.225, dev_rate)}

    # predictor distribution alignment vs paper Fig 7 (descriptive)
    align = {}
    for v in PREDICTORS:
        x = ev[v].to_numpy(dtype=float)
        align[v] = {"mean": float(np.mean(x)), "std": float(np.std(x)),
                    "p05": float(np.percentile(x, 5)),
                    "p50": float(np.percentile(x, 50)),
                    "p95": float(np.percentile(x, 95)),
                    "ri_mean": float(np.mean(x[y == 1])),
                    "nonri_mean": float(np.mean(x[y == 0]))}
    out["predictor_alignment"] = align

    op = P2 / "data" / "analysis" / "wrps1_same_sample.json"
    op.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"-> {op}")
    for tag in ("all", "fit", "inner", "test"):
        r = out[tag]
        w = r["wrps1_pu0.5"]
        e = r["ensemble_innerF1"]
        print(f"{tag:<6} n={r['n']:<6} base={r['base_rate']:.3f} | "
              f"WRPS1 BSS={w['BSS_dev']:.4f} ROC={w['ROC_AUC']:.3f} "
              f"POD={w['POD']:.3f} CSI={w['CSI']:.3f} | "
              f"ENS BSS={e['BSS_dev']:.4f} ROC={e['ROC_AUC']:.3f} "
              f"POD={e['POD']:.3f} CSI={e['CSI']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
