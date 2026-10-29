# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Sensitivity check: does MJO/ENSO confounding materially change the phase-1 ATEs?

Scientific question (user request, 2026-07-21): MJO/ENSO are BETWEEN-STORM
exogenous drivers (they modulate the large-scale environment, hence the local
field variables).  Before spending another ~2h PCMCI+ run to add them as graph
nodes, this script answers the decisive cheap question: re-estimate every
headline ATE with MJO (OMI1/OMI2, daily) and ENSO (Nino3.4 anomaly, monthly)
ADDED to the graph-derived adjustment set.  If the shifts are small, the
current conclusions are robust and graph inclusion is optional; if large,
MJO/ENSO nodes are necessary.

Outputs: data/processed_sensitivity/mjo_enso_confounding_check.csv + console.

Usage (from the release root):
    ~/anaconda3/python.exe scripts/mjo_enso_confounding_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wpac_causal.effect_estimation import EffectEstimator
from wpac_causal.utils import Config, setup_logging

RAW = Path("data/phase1/raw")
OUT = Path(os.environ.get("WPAC_MJO_OUT", "data/phase1/processed_sensitivity/mjo_enso_confounding_check.csv"))
PANEL = Path(os.environ.get("WPAC_PANEL", "data/phase1/processed/wpac_tc_dataset_processed.parquet"))
ATE = Path(os.environ.get("WPAC_BASELINE_ATE", "data/phase1/processed/ate_results.csv"))


def load_omi(path: Path) -> pd.DataFrame:
    """Wheeler-Hendon OMI: YR MON DAY OMI1 OMI2 AMPLITUDE -> daily frame."""
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["YR", "MON", "DAY", "OMI1", "OMI2", "AMPLITUDE"])
    df["date"] = pd.to_datetime(df[["YR", "MON", "DAY"]].rename(
        columns={"YR": "year", "MON": "month", "DAY": "day"}))
    return df[["date", "OMI1", "OMI2", "AMPLITUDE"]]


def load_nino34(path: Path) -> pd.Series:
    """CPC sstoi.indices: monthly Nino3.4 ANOM -> daily linear-interp series."""
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None,
                     names=["YR", "MON", "N12", "A12", "N3", "A3", "N4", "A4", "N34", "ANOM"])
    df["date"] = pd.to_datetime(dict(year=df["YR"], month=df["MON"], day=15))
    s = df.set_index("date")["ANOM"].astype(float)
    daily = s.resample("D").interpolate("time")
    return daily


def main() -> int:
    setup_logging(level="INFO")
    cfg = Config.from_yaml("configs/phase1/config.yaml")
    est = EffectEstimator(cfg)

    omi = load_omi(RAW / "omi_indices.txt")
    nino = load_nino34(RAW / "nino34_indices.txt")

    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["ISO_TIME"]).dt.tz_localize(None).dt.normalize()
    df = df.merge(omi[["date", "OMI1", "OMI2"]], on="date", how="left")
    nino_map = df["date"].map(pd.Series(nino.values, index=nino.index.normalize()))
    df["NINO34"] = nino_map.values
    n_cov = df[["OMI1", "OMI2", "NINO34"]].notna().all(axis=1).sum()
    logger.info(f"rows with MJO+ENSO coverage: {n_cov}/{len(df)}")

    # Between-storm vs within-storm variance (identification source)
    for c in ("OMI1", "NINO34"):
        b = df.groupby("SID")[c].mean().var()
        w = df.groupby("SID")[c].transform(lambda s: s - s.mean()).var()
        logger.info(f"{c}: between-storm var={b:.3f}, within-storm var={w:.3f}")

    ate = pd.read_csv(ATE)
    rows = []
    for _, r in ate.iterrows():
        trt, oc = r["treatment"], r["outcome"]
        raw_adjust = r["adjustment_set"]
        adjust = [] if pd.isna(raw_adjust) or str(raw_adjust).strip() in ("", "(empty)") else [
            part.strip() for part in str(raw_adjust).split(",") if part.strip()
        ]
        base_ate = float(r["ate"])
        sens_cols = ["OMI1", "OMI2", "NINO34"]
        adjust_s = list(dict.fromkeys(adjust + sens_cols))
        cols = [trt, oc] + adjust_s
        sub = df[cols].dropna()
        if len(sub) < 50:
            continue
        try:
            sens_ate = est._fit_ate_statsmodels(sub, trt, oc, adjust_s)
        except Exception as exc:
            logger.warning(f"{trt}->{oc}: {exc}")
            continue
        rows.append({
            "treatment": trt,
            "outcome": oc,
            "ate_base": base_ate,
            "ate_mjo_enso": sens_ate,
            "delta": sens_ate - base_ate,
            "rel_change_pct": abs(sens_ate - base_ate) / max(abs(base_ate), 1e-9) * 100,
            "sign_flip": (sens_ate * base_ate) < 0,
            "adjust_base": ",".join(adjust) if adjust else "(empty)",
            "n_obs": len(sub),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print("\n=== MJO/ENSO confounding sensitivity (ATE with indices added to adjustment) ===")
    print(res.to_string(index=False))
    print(f"\nSaved to {OUT}")

    # Marginal associations of the indices with outcome and treatments
    print("\n=== confounding channels (Spearman corr, env rows) ===")
    env = df[df["SHRD"].notna()]
    for c in ("OMI1", "OMI2", "NINO34"):
        corr_y = env[[c, "DV24"]].corr(method="spearman").iloc[0, 1]
        tops = env[["SHRD", "RHLO", "SST_OA", "SST_IC", "W500", "VORT"]].corrwith(
            env[c], method="spearman").abs().sort_values(ascending=False).head(2)
        print(f"{c}: corr with DV24={corr_y:+.3f}; strongest env links: "
              + ", ".join(f"{k}={v:.2f}" for k, v in tops.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
