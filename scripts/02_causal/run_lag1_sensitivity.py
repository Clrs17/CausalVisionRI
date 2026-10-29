# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Legacy lag>=1 sensitivity run with the VMAX lag-0 exception.

Uses allow_target_lag0=false and the default target_lag0_exceptions=["VMAX"].
For the strict configuration with no target lag-0 exceptions, use
scripts/run_lag1_strict_sensitivity.py.

Runs on the CURRENT processed panel and writes all outputs with `_nolag0`
suffix under data/processed_sensitivity/nolag0/.  Prints a comparison of the
target-parent sets vs the main (lag0-allowed) run.

Usage (from the release root):
    ~/anaconda3/python.exe scripts/run_lag1_sensitivity.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from main import run_causal_discovery, run_effect_estimation
from wpac_causal.utils import Config, setup_logging

OUT_DIR = Path("data/phase1/processed_sensitivity/nolag0")


def main() -> int:
    setup_logging(level="INFO")
    cfg_dict = Config.from_yaml("configs/phase1/config.yaml").to_dict()
    cfg_dict["causal_discovery"]["allow_target_lag0"] = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dict["causal_discovery"]["results_pkl"] = str(OUT_DIR / "pcmci_results_nolag0.pkl")
    cfg_dict["causal_discovery"]["graph_png"] = str(OUT_DIR / "causal_graph_nolag0.png")
    cfg_dict["causal_discovery"]["graph_dot"] = str(OUT_DIR / "causal_graph_nolag0.dot")
    cfg_dict["effect_estimation"]["ate_csv"] = str(OUT_DIR / "ate_results_nolag0.csv")
    cfg_dict["effect_estimation"]["cate_csv"] = str(OUT_DIR / "cate_results_nolag0.csv")
    cfg_dict["effect_estimation"]["refutation_md"] = str(OUT_DIR / "refutation_report_nolag0.md")
    cfg_dict["knowledge_base"]["output_yaml"] = str(OUT_DIR / "causal_effects_db_nolag0.yaml")
    cfg_dict["knowledge_base"]["graph_edges_json"] = str(OUT_DIR / "causal_graph_edges_nolag0.json")
    cfg = Config(cfg_dict)

    df = pd.read_parquet("data/phase1/processed/wpac_tc_dataset_processed.parquet")
    logger.info("Running lag>=1 sensitivity (allow_target_lag0=false)")
    run_causal_discovery(cfg, df)
    run_effect_estimation(cfg, df)

    # Comparison of target parents vs the main run
    def _parents(pkl_path):
        res = pickle.load(open(pkl_path, "rb"))
        g, vn, v = res["graph"], res["var_names"], res["val_matrix"]
        t = vn.index("DV24_YJ")
        out = {}
        for i in range(len(vn)):
            for tau in range(5):
                if i != t and g[i, t, tau] != "":
                    out[f"{vn[i]} lag{tau}"] = float(v[i, t, tau])
        return out

    main_parents = _parents("data/phase1/processed/pcmci_results.pkl")
    nolag0_parents = _parents(OUT_DIR / "pcmci_results_nolag0.pkl")
    keys = sorted(set(main_parents) | set(nolag0_parents))
    print("\n=== DV24_YJ parents: main (lag0 allowed) vs lag>=1 ===")
    print(f"{'edge':20s} {'main':>10s} {'lag>=1':>10s}")
    for k in keys:
        m = main_parents.get(k)
        n = nolag0_parents.get(k)
        print(f"{k:20s} {('%+.3f' % m) if m is not None else '   -':>10s} "
              f"{('%+.3f' % n) if n is not None else '   -':>10s}")
    logger.info("lag>=1 sensitivity done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
