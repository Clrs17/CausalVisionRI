# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Trivial statistical baseline for the RI task: logistic regression / LDA on
the panel env predictors (current time step only).

Production entry is the HOLDOUT_MASKED_V1 path (``configs/phase2/holdout_masked_v1.yaml``
is the default; any non-holdout config exits non-zero unless ``--legacy`` is
given explicitly).  The archived pre-holdout LOYO evaluation is reachable only
through ``--legacy`` and can never affect the holdout.  CPU-only.

Usage (from the release root):
    python scripts/04_train/baseline_lr.py                          # holdout (default)
    python scripts/04_train/baseline_lr.py --config configs/phase2/holdout_masked_v1.yaml
    python scripts/04_train/baseline_lr.py --legacy --config configs/phase2/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.baseline_common import DEFAULT_HOLDOUT_CONFIG  # noqa: E402
from wpac_warning.dataset import RIPanelDataset  # noqa: E402
from wpac_warning.train import bss_score, pick_threshold, safe_auc  # noqa: E402

DEFAULT_CONFIG = DEFAULT_HOLDOUT_CONFIG
LOYO_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


def main() -> int:
    import argparse
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from wpac_warning.baseline_common import load_baseline_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lr", "lda"], default="lr",
                    help="lr = logistic regression; lda = linear discriminant analysis (WRPS1-style)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="holdout config; a non-holdout config is refused "
                         "unless --legacy is given (default: "
                         "configs/phase2/holdout_masked_v1.yaml)")
    ap.add_argument("--legacy", action="store_true",
                    help="explicitly evaluate the archived pre-holdout LOYO "
                         "protocol (never the holdout)")
    ap.add_argument("--out-root", default=str(P2_ROOT / "data" / "runs_holdout" /
                                             "baselines"))
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    try:
        cfg, is_holdout = load_baseline_cfg(args.config, legacy=args.legacy)
    except (ValueError, FileNotFoundError) as exc:
        logger.error(str(exc))
        return 2
    if is_holdout:
        from wpac_warning.baseline_common import run_lr_lda
        run_id = args.run_id or f"baseline_{args.model}"
        run_dir = run_lr_lda(cfg, args.model, args.out_root, run_id)
        logger.info(f"holdout baseline written: {run_dir}")
        return 0

    logger.warning("LEGACY pre-holdout evaluation requested explicitly "
                   "(--legacy); this cannot touch the holdout protocol")
    ds = RIPanelDataset(cfg)
    df = ds.df
    sid_year = df.groupby("SID")["ISO_TIME"].min().dt.year
    # features: current (last) step of the env window, z-scored on train window
    X_all = np.stack([ds[i]["x_env"][-1].numpy() for i in range(len(ds))])
    y_all = df["DV24"].ge(30).astype(int).values
    years = df["SID"].map(sid_year).values
    tr_global = (years >= 2000) & (years <= 2018)
    sc = StandardScaler().fit(X_all[tr_global])
    X_all = sc.transform(X_all)

    rows = []
    for held in LOYO_YEARS:
        va = years == held
        inner = np.isin(years, [2016, 2017, 2018])
        tr = ~va & ~inner & (years <= 2018)
        if args.model == "lda":
            clf = LinearDiscriminantAnalysis()
        else:
            clf = LogisticRegression(class_weight="balanced", max_iter=3000)
        clf.fit(X_all[tr], y_all[tr])
        p_in = clf.predict_proba(X_all[inner])[:, 1]
        thr = pick_threshold(p_in, y_all[inner])
        p = clf.predict_proba(X_all[va])[:, 1]
        y = y_all[va]
        pred = p >= thr
        pod = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        far = float(((pred == 1) & (y == 0)).sum() / max((pred == 1).sum(), 1))
        roc, pr = safe_auc(p, y)
        rows.append({"held_year": held, "POD": pod, "FAR": far,
                     "BSS": bss_score(p, y), "ROC_AUC": roc, "PR_AUC": pr,
                     "threshold": thr})
        logger.info(f"[{held}] POD={pod:.3f} FAR={far:.3f} "
                    f"BSS={rows[-1]['BSS']:.3f} ROC={roc:.3f} PR={pr:.3f}")
    out = pd.DataFrame(rows)
    out_path = P2_ROOT / "data" / f"loyo_results_baseline_{args.model}.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"baseline done, mean BSS={out.BSS.mean():.3f} "
                f"ROC={out.ROC_AUC.mean():.3f} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
