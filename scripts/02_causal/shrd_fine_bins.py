# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Fine-grained SHRD response curve (0-15 kt in 3.75 kt steps) with HC3-robust
SEs, to document the plateau->decline response shape (supports the FINDINGS
statement that low-to-moderate shear has no distinguishable effect and
suppression emerges around ~11-15 kt). Writes
data/processed_sensitivity/shrd_fine_bins.csv.

Usage (from the release root):
    ~/anaconda3/python.exe scripts/shrd_fine_bins.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

CONF = ["VMAX", "PER", "SPD", "LAT", "LAND_DIST", "SST_OA", "SST_IC", "RHLO", "W500"]


def contrast(sub: pd.DataFrame, lo: float, hi: float, rlo: float, rhi: float,
             label: str, conf: list) -> dict:
    a = sub[(sub.SHRD_RAW >= lo) & (sub.SHRD_RAW < hi)]
    b = sub[(sub.SHRD_RAW >= rlo) & (sub.SHRD_RAW < rhi)]
    d = pd.concat([a.assign(T=1), b.assign(T=0)])
    X = sm.add_constant(d[["T"] + conf].astype(float))
    o = sm.OLS(d["DV24"], X).fit(cov_type="HC3")
    g = sm.GLM(d["RI"], X, family=sm.families.Binomial()).fit(disp=0, maxiter=200)
    return {
        "contrast": label,
        "n": len(d),
        "dv24_coef": float(o.params["T"]),
        "dv24_p": float(o.pvalues["T"]),
        "ri_logit": float(g.params["T"]),
        "ri_p": float(g.pvalues["T"]),
    }


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    df = pd.read_parquet("data/phase1/processed/wpac_tc_dataset_processed.parquet")
    conf = [c for c in CONF if c in df.columns]
    sub = df[["SHRD_RAW", "DV24", "RI"] + conf].dropna().copy()
    rows = [
        contrast(sub, 0, 3.75, 3.75, 7.5, "0-3.75 vs 3.75-7.5", conf),
        contrast(sub, 3.75, 7.5, 7.5, 11.25, "3.75-7.5 vs 7.5-11.25", conf),
        contrast(sub, 7.5, 11.25, 11.25, 15, "7.5-11.25 vs 11.25-15", conf),
        contrast(sub, 0, 7.5, 7.5, 15, "0-7.5 vs 7.5-15", conf),
        contrast(sub, 0, 5, 5, 15, "0-5 vs 5-15", conf),
        contrast(sub, 15, 100, 5, 15, ">=15 vs 5-15", conf),
    ]
    out = pd.DataFrame(rows)
    out_path = Path("data/phase1/processed_sensitivity/shrd_fine_bins.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
