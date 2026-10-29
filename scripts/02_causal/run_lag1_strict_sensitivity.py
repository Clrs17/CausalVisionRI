# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Strict lag>=1 sensitivity: no non-self contemporaneous discovery links into DV24_YJ.

Runs PCMCI+ and effect estimation on the current 375-storm panel. Outputs are
kept separate from both the baseline and the legacy nolag0 sensitivity run.
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

OUT_DIR = Path("data/phase1/processed_sensitivity/nolag0_strict")
PKL_PATH = OUT_DIR / "pcmci_results_nolag0_strict.pkl"


def _target_parents(pkl_path: Path) -> dict[str, float]:
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    graph, names, values = res["graph"], res["var_names"], res["val_matrix"]
    target = names.index("DV24_YJ")
    out = {}
    for i, name in enumerate(names):
        if i == target:
            continue
        for tau in range(graph.shape[2]):
            if graph[i, target, tau] != "":
                out[f"{name} lag{tau}"] = float(values[i, target, tau])
    return out


def _assert_strict_target_lag1(pkl_path: Path) -> None:
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    graph, names = res["graph"], res["var_names"]
    target = names.index("DV24_YJ")
    violations = []
    for i, name in enumerate(names):
        if i == target:
            continue
        if graph[i, target, 0] != "" or graph[target, i, 0] != "":
            violations.append(name)
    if violations:
        raise RuntimeError(f"Non-self target lag-0 links remain: {violations}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(OUT_DIR / "run.log"))
    cfg_dict = Config.from_yaml("configs/phase1/config.yaml").to_dict()
    cfg_dict["causal_discovery"]["allow_target_lag0"] = False
    cfg_dict["causal_discovery"]["target_lag0_exceptions"] = []
    cfg_dict["causal_discovery"]["results_pkl"] = str(PKL_PATH)
    cfg_dict["causal_discovery"]["graph_png"] = str(OUT_DIR / "causal_graph_nolag0_strict.png")
    cfg_dict["causal_discovery"]["graph_dot"] = str(OUT_DIR / "causal_graph_nolag0_strict.dot")
    cfg_dict["effect_estimation"]["ate_csv"] = str(OUT_DIR / "ate_results_nolag0_strict.csv")
    cfg_dict["effect_estimation"]["cate_csv"] = str(OUT_DIR / "cate_results_nolag0_strict.csv")
    cfg_dict["effect_estimation"]["refutation_md"] = str(OUT_DIR / "refutation_report_nolag0_strict.md")
    cfg = Config(cfg_dict)

    baseline_pkl = Path("data/phase1/processed/pcmci_results.pkl")
    if not baseline_pkl.exists():
        raise FileNotFoundError(f"Baseline PCMCI result required for comparison: {baseline_pkl}")
    df = pd.read_parquet("data/phase1/processed/wpac_tc_dataset_processed.parquet")
    logger.info("Running strict lag>=1 sensitivity with zero target lag-0 exceptions")
    run_causal_discovery(cfg, df)
    _assert_strict_target_lag1(PKL_PATH)
    logger.info("Strict target lag-0 assertion passed; starting effect estimation")
    run_effect_estimation(cfg, df)

    baseline = _target_parents(Path("data/phase1/processed/pcmci_results.pkl"))
    strict = _target_parents(PKL_PATH)
    print("\n=== DV24_YJ parents: baseline vs strict lag>=1 ===")
    print(f"{'edge':28s} {'baseline':>12s} {'strict':>12s}")
    for key in sorted(set(baseline) | set(strict)):
        b = baseline.get(key)
        s = strict.get(key)
        btxt = f"{b:+.4f}" if b is not None else "-"
        stxt = f"{s:+.4f}" if s is not None else "-"
        print(f"{key:28s} {btxt:>12s} {stxt:>12s}")
    logger.info("Strict lag>=1 sensitivity complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
