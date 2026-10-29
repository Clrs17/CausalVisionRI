# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Selection-bias sensitivity for the >=65kt peak-intensity sample definition.

Scientific context (user decision, 2026-07-21): the analysis population is
DELIBERATELY WPAC TCs whose peak intensity reaches >=65 kt, because (1) RI is
disaster-relevant mainly at typhoon strength and (2) weaker systems may obey
different mechanisms and are out of scope.

This is a legitimate domain restriction (estimand = ATE among typhoon-strength
TCs).  The residual concern (E report section 3) is that selecting on LIFECYCLE
PEAK intensity conditions on a post-treatment outcome relative to early-lifecycle
observations: storms that never reach 65 kt contribute no weak-phase rows, which
can distort the pooled env -> DV24 relationship (range restriction).  The
distortion should appear mainly in WEAK-PHASE rows.

This script therefore re-estimates every headline ATE (graph-derived adjustment
sets, statsmodels, same lean dropna) on three row subsets of the SAME processed
panel of >=65kt-peaking storms:

    ALL      : all lifecycle rows (the headline estimand)
    WEAK     : rows with VMAX < 65 kt (pre-typhoon phase)
    STRONG   : rows with VMAX >= 65 kt (typhoon phase)

Material ALL-vs-WEAK differences indicate selection distortion (+ genuine stage
effect modification, which cannot be separated here but bounds the bias).

Outputs: data/processed_sensitivity/selection_bias_check.csv + console.

Usage (from the release root):
    ~/anaconda3/python.exe scripts/selection_bias_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wpac_causal.effect_estimation import EffectEstimator
from wpac_causal.utils import Config, setup_logging

OUT = Path(os.environ.get("WPAC_SELECTION_OUT", "data/phase1/processed_sensitivity/selection_bias_check.csv"))
PANEL = Path(os.environ.get("WPAC_PANEL", "data/phase1/processed/wpac_tc_dataset_processed.parquet"))
ATE = Path(os.environ.get("WPAC_BASELINE_ATE", "data/phase1/processed/ate_results.csv"))


def main() -> int:
    setup_logging(level="INFO")
    cfg = Config.from_yaml("configs/phase1/config.yaml")
    est = EffectEstimator(cfg)

    df = pd.read_parquet(PANEL)
    ate = pd.read_csv(ATE)

    subsets = {
        "ALL": df,
        "WEAK": df[df["VMAX"] < 65],
        "STRONG": df[df["VMAX"] >= 65],
    }
    logger.info(
        "rows: ALL=%d, WEAK=%d, STRONG=%d"
        % tuple(len(s) for s in subsets.values())
    )

    rows = []
    for _, r in ate.iterrows():
        trt, oc = r["treatment"], r["outcome"]
        raw_adjust = r["adjustment_set"]
        adjust = [] if pd.isna(raw_adjust) or str(raw_adjust).strip() in ("", "(empty)") else [
            part.strip() for part in str(raw_adjust).split(",") if part.strip()
        ]
        rec = {"treatment": trt, "outcome": oc,
               "adjust": ",".join(adjust) if adjust else "(empty)"}
        for name, sub_df in subsets.items():
            cols = [trt, oc] + adjust
            sub = sub_df[cols].dropna()
            if len(sub) < 100 or sub[trt].nunique() < 2:
                rec[f"ate_{name}"] = None
                rec[f"n_{name}"] = len(sub)
                continue
            try:
                rec[f"ate_{name}"] = est._fit_ate_statsmodels(sub, trt, oc, adjust)
            except Exception:
                rec[f"ate_{name}"] = None
            rec[f"n_{name}"] = len(sub)
        rows.append(rec)
    res = pd.DataFrame(rows)
    for name in ("WEAK", "STRONG"):
        res[f"sign_flip_{name}"] = (res[f"ate_{name}"] * res["ate_ALL"]) < 0
        res[f"rel_diff_{name}_pct"] = (
            (res[f"ate_{name}"] - res["ate_ALL"]).abs()
            / res["ate_ALL"].abs().clip(lower=1e-9) * 100
        ).round(1)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 240)
    print("\n=== Selection-bias sensitivity (ATE by lifecycle phase) ===")
    print(res.to_string(index=False))
    print(f"\nSaved to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
