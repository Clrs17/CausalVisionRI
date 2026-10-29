# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Deterministically rebuild the identification DAG (NO ATE re-estimation).

The ATE/CATE/refutation results already on disk were produced with
``EffectEstimator.build_graph_dot``.  That DOT was not persisted in the early
runs, so this script rebuilds it from the same PCMCI results and verifies that
DoWhy identification from the rebuilt DOT reproduces the existing
``identification_status.csv`` adjustment sets exactly.  It never re-estimates
an effect.

Outputs (in --prior-dir):
    identification_dag.dot
    identification_dag.json          # provenance edge table + dot hash
    identification_dag_verification.json

Usage (from the release root):
    python scripts/02_causal/rebuild_identification_dag.py
    python scripts/02_causal/rebuild_identification_dag.py --prior-dir ... --allow-diff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from wpac_causal.effect_estimation import EffectEstimator
from wpac_causal.utils import Config

HOLDOUT = ROOT / "data" / "phase1" / "processed_masked" / "holdout_fit0017"
DEFAULT_PRIOR = HOLDOUT / "prior"
DEFAULT_PANEL = HOLDOUT / "panel" / "wpac_tc_dataset_processed.parquet"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(prior_dir: Path) -> Config:
    cfg = Config.from_yaml("configs/phase1/config.yaml").to_dict()
    cfg["causal_discovery"]["results_pkl"] = str(prior_dir / "pcmci_results.pkl")
    cfg["effect_estimation"]["ate_csv"] = str(prior_dir / "ate_results.csv")
    cfg["effect_estimation"]["cate_csv"] = str(prior_dir / "cate_results.csv")
    cfg["effect_estimation"]["refutation_md"] = str(
        prior_dir / "refutation_report.md")
    cfg["effect_estimation"]["identification_dot"] = str(
        prior_dir / "identification_dag.dot")
    cfg["effect_estimation"]["identification_json"] = str(
        prior_dir / "identification_dag.json")
    return Config(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-dir", default=str(DEFAULT_PRIOR))
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--allow-diff", action="store_true",
                    help="write the verification report even if adjustment "
                         "sets differ (does not change the existing CSV)")
    args = ap.parse_args()

    prior = Path(args.prior_dir)
    fit = pd.read_parquet(prior / "fit_prior_panel.parquet")
    with (prior / "pcmci_results.pkl").open("rb") as handle:
        result = pickle.load(handle)
    if result["metadata"].get("mask_type") != "xyz":
        raise RuntimeError("PCMCI results are not masked (mask_type != xyz)")

    cfg = build_config(prior)
    estimator = EffectEstimator(cfg)
    dag = estimator.build_identification_dag(fit, result)
    estimator._write_identification_dag(dag, fit, result)

    # Verify identification from the rebuilt DOT against the existing CSV.
    env = os.environ.copy()
    env.update({
        "WPAC_PCMCI_PKL": str(prior / "pcmci_results.pkl"),
        "WPAC_PANEL": str(prior / "fit_prior_panel.parquet"),
        "WPAC_IDENTIFICATION_DOT": str(prior / "identification_dag.dot"),
        "WPAC_IDENTIFICATION_JSON": str(prior / "identification_dag.json"),
        "WPAC_ATE_RESULTS": str(prior / "ate_results.csv"),
        "WPAC_IDENTIFICATION_OUT": str(prior / "_identification_check.csv"),
    })
    subprocess.run(
        [sys.executable, str(ROOT / "paper_supplementary/code/identification_status.py")],
        cwd=ROOT, env=env, check=True)
    fresh = pd.read_csv(prior / "_identification_check.csv")
    existing_path = prior / "identification_status.csv"
    dot_sha = json.loads(
        (prior / "identification_dag.json").read_text(encoding="utf-8")
    )["dot_sha256"]
    report = {
        "dot_sha256": dot_sha,
        "check_vs_existing": str(existing_path),
        "n_rows_checked": int(len(fresh)),
        "adjustment_sets": "copied from ate_results.csv (never re-derived)",
        "no_directed_path": "deterministic reachability on identification_dag.dot",
    }
    ate_path = prior / "ate_results.csv"
    if ate_path.is_file():
        report["ate_sha256"] = _sha256_file(ate_path)
    if existing_path.is_file():
        old = pd.read_csv(existing_path)
        cols = ["treatment", "outcome", "no_directed_path", "adjustment_set"]
        merged = old[cols].merge(fresh[cols], on=["treatment", "outcome"],
                                 suffixes=("_old", "_new"), validate="one_to_one")
        def norm(v):
            return "" if pd.isna(v) or str(v) == "(empty)" else str(v)
        diffs = []
        for row in merged.itertuples():
            # adjustment sets are read from the ATE on both sides; only the
            # graph-derived reachability flag can legitimately differ here
            if (bool(row.no_directed_path_old) != bool(row.no_directed_path_new)
                    or norm(row.adjustment_set_old) != norm(row.adjustment_set_new)):
                diffs.append({
                    "treatment": row.treatment, "outcome": row.outcome,
                    "old": [bool(row.no_directed_path_old),
                            str(row.adjustment_set_old)],
                    "new": [bool(row.no_directed_path_new),
                            str(row.adjustment_set_new)],
                })
        report["n_differences"] = len(diffs)
        report["differences"] = diffs[:20]
        if diffs and not args.allow_diff:
            (prior / "identification_dag_verification.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            raise SystemExit(
                f"rebuilt DOT changes {len(diffs)} identification rows; the "
                f"existing status/ATE pair is not reproducible from this DOT. "
                f"See identification_dag_verification.json")
    (prior / "_identification_check.csv").unlink(missing_ok=True)
    (prior / "identification_dag_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
