# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Local rerun of the headroom CATE only (R3-1): no PCMCI re-run needed.
Writes data/processed/headroom_cate_results.csv.

Usage (from the release root):
    ~/anaconda3/python.exe scripts/rerun_headroom.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wpac_causal.effect_estimation import EffectEstimator
from wpac_causal.utils import Config, load_pickle, setup_logging


def main() -> int:
    setup_logging(level="INFO")
    cfg = Config.from_yaml("configs/phase1/config.yaml")
    est = EffectEstimator(cfg)
    df = pd.read_parquet("data/phase1/processed/wpac_tc_dataset_processed.parquet")
    pcmci = load_pickle("data/phase1/processed/pcmci_results.pkl")
    gdot = est.build_graph_dot(df, pcmci)
    rows = est.estimate_headroom_cate(df, gdot)
    out = pd.DataFrame(rows)
    out_path = Path("data/phase1/processed/headroom_cate_results.csv")
    out.to_csv(out_path, index=False)
    logger.info(f"Headroom CATE rewritten: {len(out)} rows -> {out_path}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
