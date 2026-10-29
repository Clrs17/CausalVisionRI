# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Run masked PCMCI+ and downstream estimation in an isolated directory."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from main import run_causal_discovery, run_effect_estimation, run_knowledge_base
from wpac_causal.utils import Config, setup_logging


def configure(stage: str, out_dir: Path) -> Config:
    cfg_dict = Config.from_yaml("configs/phase1/config.yaml").to_dict()
    if stage == "strict":
        cfg_dict["causal_discovery"]["allow_target_lag0"] = False
        cfg_dict["causal_discovery"]["target_lag0_exceptions"] = []
    elif stage == "no_ohc":
        cfg_dict["causal_discovery"]["variables"] = [
            name for name in cfg_dict["causal_discovery"]["variables"]
            if not name.startswith("OHC")
        ]
        cfg_dict["effect_estimation"]["treatments"] = [
            name for name in cfg_dict["effect_estimation"]["treatments"]
            if not name.startswith("OHC")
        ]
        cfg_dict["effect_estimation"]["headroom"]["treatments"] = [
            name for name in cfg_dict["effect_estimation"]["headroom"]["treatments"]
            if not name.startswith("OHC")
        ]
    cfg_dict["causal_discovery"]["results_pkl"] = str(out_dir / "pcmci_results.pkl")
    cfg_dict["causal_discovery"]["graph_png"] = str(out_dir / "causal_graph.png")
    cfg_dict["causal_discovery"]["graph_dot"] = str(out_dir / "causal_graph.dot")
    cfg_dict["effect_estimation"]["ate_csv"] = str(out_dir / "ate_results.csv")
    cfg_dict["effect_estimation"]["cate_csv"] = str(out_dir / "cate_results.csv")
    cfg_dict["effect_estimation"]["refutation_md"] = str(out_dir / "refutation_report.md")
    cfg_dict["knowledge_base"]["output_yaml"] = str(out_dir / "causal_effects_db.yaml")
    cfg_dict["knowledge_base"]["graph_edges_json"] = str(out_dir / "causal_graph_edges.json")
    return Config(cfg_dict)


def validate_graph(stage: str, pkl_path: Path) -> None:
    with open(pkl_path, "rb") as handle:
        result = pickle.load(handle)
    metadata = result["metadata"]
    if metadata.get("mask_type") != "xyz":
        raise RuntimeError(f"Unexpected mask_type: {metadata.get('mask_type')}")
    if stage != "strict":
        return
    graph, names = result["graph"], result["var_names"]
    target = names.index("DV24_YJ")
    violations = [
        name for i, name in enumerate(names)
        if i != target and (graph[i, target, 0] != "" or graph[target, i, 0] != "")
    ]
    if violations:
        raise RuntimeError(f"Non-self target lag-0 links remain: {violations}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["baseline", "strict", "no_ohc"])
    parser.add_argument("--output-root", default="data/phase1/processed_masked")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--effects-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / "run.log"))
    cfg = configure(args.stage, out_dir)
    df = pd.read_parquet("data/phase1/processed/wpac_tc_dataset_processed.parquet")
    logger.info(f"Running masked {args.stage} pipeline into {out_dir}")
    if not args.effects_only:
        run_causal_discovery(cfg, df)
    pkl_path = out_dir / "pcmci_results.pkl"
    validate_graph(args.stage, pkl_path)
    if not args.discovery_only:
        run_effect_estimation(cfg, df)
        run_knowledge_base(cfg, df)
    logger.info(f"Masked {args.stage} pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
