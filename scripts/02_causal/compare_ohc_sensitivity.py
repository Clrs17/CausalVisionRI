# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Run or compare the with/without-OHC effect specifications."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from main import run_causal_discovery, run_effect_estimation
from wpac_causal.utils import Config, setup_logging

OUT_DIR = Path(os.environ.get("WPAC_NO_OHC_DIR", "data/phase1/processed_sensitivity/no_ohc"))
FULL_ATE = Path(os.environ.get("WPAC_BASELINE_ATE", "data/phase1/processed/ate_results.csv"))
OUT_CSV = Path(os.environ.get("WPAC_OHC_COMPARE_OUT", "data/phase1/processed_sensitivity/ohc_confounding_comparison.csv"))
COMPARE_TREATMENTS = ["SHRD", "RHLO", "VORT", "PV", "D200", "WARM_CORE", "MFC", "W500"]


def build_comparison() -> pd.DataFrame:
    key = ["treatment", "outcome"]
    ate_with = pd.read_csv(FULL_ATE)
    ate_without = pd.read_csv(OUT_DIR / "ate_results.csv")
    merged = ate_with[key + ["ate", "adjustment_set"]].merge(
        ate_without[key + ["ate", "adjustment_set"]],
        on=key, how="inner", suffixes=("_with_ohc", "_without_ohc"),
    )
    merged = merged[merged.treatment.isin(COMPARE_TREATMENTS)].copy()
    merged["sign_flip"] = merged.ate_with_ohc * merged.ate_without_ohc < 0
    merged["rel_change_pct"] = (
        (merged.ate_with_ohc - merged.ate_without_ohc).abs()
        / merged.ate_without_ohc.abs().clip(lower=1e-9) * 100
    ).round(1)
    columns = [
        "treatment", "outcome", "ate_with_ohc", "adjustment_set_with_ohc",
        "ate_without_ohc", "adjustment_set_without_ohc", "sign_flip",
        "rel_change_pct",
    ]
    return merged[columns].sort_values(["treatment", "outcome"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    setup_logging(level="INFO")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.compare_only:
        cfg_dict = Config.from_yaml("configs/phase1/config.yaml").to_dict()
        vars_no_ohc = [
            v for v in cfg_dict["causal_discovery"]["variables"]
            if not v.startswith("OHC")
        ]
        cfg_dict["causal_discovery"]["variables"] = vars_no_ohc
        cfg_dict["effect_estimation"]["treatments"] = [
            v for v in cfg_dict["effect_estimation"]["treatments"]
            if not v.startswith("OHC")
        ]
        cfg_dict["effect_estimation"]["headroom"]["treatments"] = [
            v for v in cfg_dict["effect_estimation"]["headroom"]["treatments"]
            if not v.startswith("OHC")
        ]
        cfg_dict["causal_discovery"]["results_pkl"] = str(OUT_DIR / "pcmci_results.pkl")
        cfg_dict["causal_discovery"]["graph_png"] = str(OUT_DIR / "causal_graph.png")
        cfg_dict["causal_discovery"]["graph_dot"] = str(OUT_DIR / "causal_graph.dot")
        cfg_dict["effect_estimation"]["ate_csv"] = str(OUT_DIR / "ate_results.csv")
        cfg_dict["effect_estimation"]["cate_csv"] = str(OUT_DIR / "cate_results.csv")
        cfg_dict["effect_estimation"]["refutation_md"] = str(OUT_DIR / "refutation_report.md")
        cfg_dict["knowledge_base"]["output_yaml"] = str(OUT_DIR / "causal_effects_db.yaml")
        cfg_dict["knowledge_base"]["graph_edges_json"] = str(OUT_DIR / "causal_graph_edges.json")
        df = pd.read_parquet("data/phase1/processed/wpac_tc_dataset_processed.parquet")
        logger.info(f"Panel: {df.shape}; running WITHOUT OHC nodes ({len(vars_no_ohc)} vars)")
        run_causal_discovery(Config(cfg_dict), df)
        run_effect_estimation(Config(cfg_dict), df)

    comparison = build_comparison()
    comparison.to_csv(OUT_CSV, index=False)
    logger.info(f"Comparison saved to {OUT_CSV}")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
