# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""RI threshold ("switch") effect analysis.

Scientific question: which factors have a *disproportionate* effect on the
binary RI outcome relative to their mean effect on continuous DV24?  Slow-
intensification samples dominate the mean, so a factor can look modest on
DV24 yet act as an RI "switch".

Important subtlety: RI = 1{DV24 >= 30} is a deterministic coarsening of DV24,
so the Baron-Kenny "direct effect on RI controlling DV24" is IDENTICALLY ~0
under a correctly specified model; the LPM-based mediation numbers are
functional-form artefacts and must not be interpreted mechanistically.  The
correct evidence for a threshold effect is DISTRIBUTIONAL:

  1. Quantile regression of DV24 on X (+ graph adjustment) at tau = 0.75/0.90
     vs the OLS mean effect.  |q90/mean| > 1  => the effect is amplified in
     the upper (RI-prone) tail -> switch character.
  2. AME/implied ratio: the actual RI probability effect (average marginal
     effect from the GLM logit) divided by the probability shift implied by
     the mean DV24 shift under a normal approximation.  > 1 => the RI effect
     exceeds what a uniform distribution shift predicts.

Outputs: data/processed_sensitivity/threshold_effect.csv + console.

Usage (from the release root):
    ~/anaconda3/python.exe scripts/threshold_effect_analysis.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wpac_causal.utils import setup_logging

OUT = Path(os.environ.get("WPAC_THRESHOLD_OUT", "data/phase1/processed_sensitivity/threshold_effect.csv"))
PANEL = Path(os.environ.get("WPAC_PANEL", "data/phase1/processed/wpac_tc_dataset_processed.parquet"))
ATE = Path(os.environ.get("WPAC_BASELINE_ATE", "data/phase1/processed/ate_results.csv"))
RI_THRESHOLD = 30.0


def parse_adjust(value) -> list[str]:
    if pd.isna(value) or str(value).strip() in ("", "(empty)"):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main() -> int:
    setup_logging(level="INFO")
    df = pd.read_parquet(PANEL)
    ate = pd.read_csv(ATE)

    mu, sd = float(df["DV24"].mean()), float(df["DV24"].std())
    ri_rate = float((df["DV24"] >= RI_THRESHOLD).mean())
    logger.info(f"DV24 mean={mu:.2f} sd={sd:.2f} P(RI)={ri_rate:.4f}")

    rows = []
    for trt in ate["treatment"].unique():
        r_dv = ate[(ate.treatment == trt) & (ate.outcome == "DV24")].iloc[0]
        adj_d = parse_adjust(r_dv.adjustment_set)
        sub = df[[trt, "DV24"] + adj_d].dropna()
        if len(sub) < 200:
            continue
        X = sm.add_constant(sub[[trt] + adj_d])
        mean_c = float(r_dv["ate"])
        q75 = float(sm.QuantReg(sub["DV24"], X).fit(q=0.75, max_iter=2000).params[trt])
        q90 = float(sm.QuantReg(sub["DV24"], X).fit(q=0.90, max_iter=2000).params[trt])

        # implied dP (normal approx) vs actual AME from the RI GLM
        d = mean_c
        impl = (1 - stats.norm.cdf((RI_THRESHOLD - mu - d) / sd)) - \
               (1 - stats.norm.cdf((RI_THRESHOLD - mu) / sd))
        r_ri = ate[(ate.treatment == trt) & (ate.outcome == "RI")].iloc[0]
        adj_r = parse_adjust(r_ri.adjustment_set)
        subr = df[[trt, "RI"] + adj_r].dropna()
        Xr = sm.add_constant(subr[[trt] + adj_r])
        g1 = sm.GLM(subr["RI"], Xr, family=sm.families.Binomial()).fit(disp=0, maxiter=200)
        p1 = g1.predict(Xr)
        ame = float((p1 * (1 - p1)).mean() * g1.params[trt])

        rows.append({
            "treatment": trt,
            "ate_dv24_mean": mean_c,
            "q75": q75,
            "q90": q90,
            "tail_ratio_abs_q90_over_mean": abs(q90 / mean_c) if abs(mean_c) > 1e-6 else np.nan,
            "ri_ame": ame,
            "implied_dp_mean_shift": impl,
            "ame_over_implied": ame / impl if abs(impl) > 1e-6 else np.nan,
            "switch_evidence": "yes" if (abs(q90 / mean_c) > 1.1 if abs(mean_c) > 1e-6 else False)
                              or (abs(ame / impl) > 1.3 if abs(impl) > 1e-6 else False) else "no",
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print("\n=== RI threshold (switch) effect analysis ===")
    print(res.to_string(index=False))
    print(f"\nSaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
