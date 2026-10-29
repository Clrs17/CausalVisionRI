# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Local rerun of refutations only (R3-2): rebuild refutation_report.md and
the per-test verdict columns in ate_results.csv without re-running ATEs,
bootstraps, PCMCI or headroom.

Usage (from the release root):
    ~/anaconda3/python.exe scripts/rerun_refutations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wpac_causal.effect_estimation import EffectEstimator
from wpac_causal.utils import Config, load_pickle, setup_logging

META_KEYS = ("total_passed", "total_tests", "refutation_seed",
             "reference_method", "reference_ate_lpm")


def main() -> int:
    setup_logging(level="INFO")
    cfg = Config.from_yaml("configs/phase1/config.yaml")
    est = EffectEstimator(cfg)
    df = pd.read_parquet("data/phase1/processed/wpac_tc_dataset_processed.parquet")
    pcmci = load_pickle("data/phase1/processed/pcmci_results.pkl")
    gdot = est.build_graph_dot(df, pcmci)

    ate_path = Path("data/phase1/processed/ate_results.csv")
    ate = pd.read_csv(ate_path)

    report = ["# Refutation Report\n"]
    report.append(
        f"\nRandom seed for all stochastic refuters: **{est.ref_seed}** "
        "(placebo random_state, subset random_state, unobserved-confounder "
        "shuffle_random_seed). Reference effects are LPM-scale on both "
        "outcomes. Empty-backdoor pairs use statsmodels-native "
        "placebo/subset (R3-2); SKIPPED = engineering limitation, not a "
        "robustness verdict.\n"
    )

    for trt in ate["treatment"].unique():
        logger.info(f"Refutations for {trt}")
        ref_by_outcome = {}
        for outcome in (est.target_cont, est.target_bin):
            row = ate[(ate.treatment == trt) & (ate.outcome == outcome)]
            if row.empty:
                continue
            row = row.iloc[0]
            ref = est.run_refutations(df, trt, outcome, gdot, {"ate": row["ate"]})
            ref_by_outcome[outcome] = ref
            ate.loc[(ate.treatment == trt) & (ate.outcome == outcome), "refutation_passed"] = ref.get("total_passed", 0)
            ate.loc[(ate.treatment == trt) & (ate.outcome == outcome), "refutation_total"] = ref.get("total_tests", 0)
            for test in ("placebo", "subset", "unobserved_confounder", "offshore"):
                ate.loc[(ate.treatment == trt) & (ate.outcome == outcome), f"ref_{test}_passed"] = ref.get(test, {}).get("passed")

        report.append(f"\n## Treatment: {trt}\n")
        for outcome, label in ((est.target_cont, "DV24"), (est.target_bin, "RI")):
            ref = ref_by_outcome.get(outcome)
            if ref is None:
                continue
            report.append(f"### {label}\n")
            for test_name, test_res in ref.items():
                if test_name in META_KEYS or not isinstance(test_res, dict):
                    continue
                if test_res.get("skipped"):
                    status = "SKIP"
                else:
                    status = "PASS" if test_res.get("passed") else "FAIL"
                report.append(f"- **{test_name}**: {status} ({test_res})\n")

    ate.to_csv(ate_path, index=False)
    Path("data/phase1/processed/refutation_report.md").write_text("".join(report), encoding="utf-8")
    logger.info("refutation_report.md and ate_results.csv verdict columns rewritten")
    print(ate[["treatment", "outcome", "refutation_passed", "refutation_total",
               "ref_placebo_passed", "ref_subset_passed",
               "ref_unobserved_confounder_passed", "ref_offshore_passed"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
