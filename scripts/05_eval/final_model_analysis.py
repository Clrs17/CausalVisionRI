# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Final all-model output analysis (analysis-only; never touches training code).

Recomputes every metric from the raw per-sample predictions in
``data/preds/<arm>_fold<year>.npz`` (keys: p_held, y_held, p_inner, y_inner),
reconciles them against the LOYO CSVs, and produces

  * analysis/figures/<prefix>_*.png   (dpi=150)
  * docs/<PREFIX>_ANALYSIS.md

Supports multiple preds directories (server arms pulled later) and is
tolerant of missing folds: arms with n_folds < 6 are still reported and
flagged.  Arms that have a LOYO CSV but no preds (ablations, baselines)
are reported from CSV numbers only and flagged as provenance=csv.

Usage (from the release root):
    python scripts/05_eval/final_model_analysis.py --preds-dirs data/preds \
        [--preds-dirs data/preds_server] --out-prefix final
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# display / analysis order: main arms, then server arms, seeds, ablations, baselines
MAIN_ORDER = [
    "ens_v1", "ens_lingbot", "ens_lingbotB", "ens_lingbotL", "ens_lingbotG",
    "ens_ep40", "ens_ep40ft", "ens_ep40_srv", "union4",
]
UNION4_MEMBERS = ["ens_v1", "ens_lingbotB", "ens_ep40", "ens_ep40ft"]

# npz arm name -> loyo csv stem, when they differ
CSV_ALIAS = {"ens_lingbot_oldp": "ens_lingbot_oldprotocol"}

# arms expected to exist only as CSVs (no per-sample preds)
CSV_ONLY_PATTERNS = ["ablate_noloss", "ablate_nognn", "ablate_frozen",
                     "baseline_lr", "baseline_lda"]

# arms excluded from the report entirely (seed variants no longer tracked,
# superseded protocol runs)
EXCLUDE_PATTERNS = [r"_s\d+$", r"_oldp$"]

RECON_TOL = 1e-3


# --------------------------------------------------------------------------
# metric primitives (mirror src/train.py conventions: p >= thr; BSS vs the
# fold's own climatology)
# --------------------------------------------------------------------------

def bss_score(p: np.ndarray, y: np.ndarray) -> float:
    bs = float(np.mean((p - y) ** 2))
    bs_clim = float(np.mean((y.mean() - y) ** 2))
    return 1.0 - bs / max(bs_clim, 1e-9)


def pod_far(p: np.ndarray, y: np.ndarray, thr: float) -> tuple[float, float]:
    pred = p >= thr
    pod = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    far = float(((pred == 1) & (y == 0)).sum() / max((pred == 1).sum(), 1))
    return pod, far


def safe_auc(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    from sklearn.metrics import average_precision_score, roc_auc_score
    return (float(roc_auc_score(y, p)), float(average_precision_score(y, p)))


def pick_threshold(p: np.ndarray, y: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = p >= t
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def murphy_decomposition(p: np.ndarray, y: np.ndarray, bins: int = 10
                         ) -> dict[str, float]:
    """REL - RES + UNC decomposition of the Brier score (10-bin Murphy).

    Uses the mean forecast probability inside each bin, so
    BS ~= REL - RES + UNC up to within-bin variance.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    n = len(y)
    obar = float(y.mean())
    rel = res = 0.0
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        fbar = float(p[m].mean())
        obark = float(y[m].mean())
        rel += nk * (fbar - obark) ** 2
        res += nk * (obark - obar) ** 2
    rel /= n
    res /= n
    unc = obar * (1.0 - obar)
    bs = float(np.mean((p - y) ** 2))
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "brier": bs, "check": rel - res + unc}


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int = 10
                      ) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        rows.append({
            "bin": k,
            "p_mean": float(p[m].mean()) if nk else float(edges[k] + edges[k + 1]) / 2,
            "obs_freq": float(y[m].mean()) if nk else np.nan,
            "n": nk,
        })
    return pd.DataFrame(rows)


def calibration_slope(p: np.ndarray, y: np.ndarray) -> float:
    """Slope of logistic regression y ~ logit(p); 1.0 = perfectly calibrated."""
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.linear_model import LogisticRegression
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    x = np.log(pc / (1.0 - pc)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(x, y)
    return float(lr.coef_[0][0])


def event_focused_bss(p: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """BSS restricted to the high-risk subset (p > 0.5 or y == 1),
    vs that subset's own climatology."""
    m = (p > 0.5) | (y == 1)
    n = int(m.sum())
    if n < 10 or len(np.unique(y[m])) < 2:
        return float("nan"), n
    return bss_score(p[m], y[m]), n


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def discover_arms(preds_dirs: list[Path]) -> dict[str, dict[int, Path]]:
    """{arm: {year: npz path}} from every preds dir."""
    arms: dict[str, dict[int, Path]] = {}
    for d in preds_dirs:
        if not d.is_dir():
            print(f"WARN: preds dir missing: {d}")
            continue
        for f in sorted(d.glob("*_fold*.npz")):
            m = re.match(r"(.+)_fold(\d{4})\.npz$", f.name)
            if not m:
                continue
            arm, year = m.group(1), int(m.group(2))
            if any(re.search(p, arm) for p in EXCLUDE_PATTERNS):
                continue
            arms.setdefault(arm, {})[year] = f
    return arms


def load_csv_for_arm(arm: str, data_dir: Path) -> pd.DataFrame | None:
    stem = CSV_ALIAS.get(arm, arm)
    path = data_dir / f"loyo_results_{stem}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "held_year" in df.columns:
        df = df.set_index("held_year")
    return df


# --------------------------------------------------------------------------
# per-arm analysis
# --------------------------------------------------------------------------

def analyze_arm(arm: str, fold_paths: dict[int, Path],
                csv: pd.DataFrame | None, bins: int,
                warns: list[str]) -> dict:
    folds = {}
    p_pool, y_pool = [], []
    for year in sorted(fold_paths):
        d = np.load(fold_paths[year])
        p = np.asarray(d["p_held"], dtype=float)
        y = np.asarray(d["y_held"], dtype=float)
        thr = None
        if csv is not None and year in csv.index and "threshold" in csv.columns:
            thr = float(csv.loc[year, "threshold"])
        if thr is None or not math.isfinite(thr):
            warns.append(f"{arm}/{year}: no CSV threshold -> POD/FAR skipped")
        pod, far = (pod_far(p, y, thr) if thr is not None
                    else (float("nan"), float("nan")))
        roc, pr = safe_auc(p, y)
        folds[year] = {
            "n": len(y), "ri_rate": float(y.mean()), "threshold": thr,
            "POD": pod, "FAR": far, "BSS": bss_score(p, y),
            "ROC_AUC": roc, "PR_AUC": pr,
            **murphy_decomposition(p, y, bins),
        }
        p_pool.append(p)
        y_pool.append(y)

    pooled = None
    if p_pool:
        pp = np.concatenate(p_pool)
        yy = np.concatenate(y_pool)
        pooled = {
            "n": len(yy), "ri_rate": float(yy.mean()),
            **murphy_decomposition(pp, yy, bins),
            "calib_slope": calibration_slope(pp, yy),
            "rel_curve": reliability_curve(pp, yy, bins),
            "event_bss": event_focused_bss(pp, yy),
        }

    bss = [f["BSS"] for f in folds.values()]
    inner_bss = (float(csv["inner_bss"].mean())
                 if csv is not None and "inner_bss" in csv.columns else float("nan"))
    return {
        "arm": arm, "folds": folds, "n_folds": len(folds), "csv": csv,
        "bss_mean": float(np.mean(bss)) if bss else float("nan"),
        "bss_worst": float(np.min(bss)) if bss else float("nan"),
        "bss_std": float(np.std(bss)) if bss else float("nan"),
        "inner_bss": inner_bss, "pooled": pooled,
        "provenance": "preds" if folds else "csv",
    }


def analyze_csv_only_arm(arm: str, csv: pd.DataFrame) -> dict:
    folds = {}
    for year, r in csv.iterrows():
        folds[int(year)] = {
            "n": int(r.get("n", 0)), "ri_rate": float(r.get("ri_rate", np.nan)),
            "threshold": float(r.get("threshold", np.nan)),
            "POD": float(r.get("POD", np.nan)), "FAR": float(r.get("FAR", np.nan)),
            "BSS": float(r.get("BSS", np.nan)),
            "ROC_AUC": float(r.get("ROC_AUC", np.nan)),
            "PR_AUC": float(r.get("PR_AUC", np.nan)),
        }
    bss = [f["BSS"] for f in folds.values()]
    return {
        "arm": arm, "folds": folds, "n_folds": len(folds), "csv": csv,
        "bss_mean": float(np.nanmean(bss)), "bss_worst": float(np.nanmin(bss)),
        "bss_std": float(np.nanstd(bss)),
        "inner_bss": float(csv["inner_bss"].mean())
        if "inner_bss" in csv.columns else float("nan"),
        "pooled": None, "provenance": "csv",
    }


def build_union4(analyses: dict[str, dict], bins: int,
                 warns: list[str]) -> dict | None:
    members = [a for a in UNION4_MEMBERS
               if a in analyses and analyses[a]["provenance"] == "preds"]
    if len(members) < 2:
        warns.append(f"union4 skipped: only {len(members)} member arms with preds")
        return None
    common = set.intersection(*[set(analyses[a]["folds"]) for a in members])
    folds = {}
    p_pool, y_pool = [], []
    for year in sorted(common):
        ps, ys, pins, yins = [], [], [], []
        for a in members:
            d = np.load(analyses[a]["fold_paths"][year])
            ps.append(np.asarray(d["p_held"], float))
            ys.append(np.asarray(d["y_held"], float))
            pins.append(np.asarray(d["p_inner"], float))
            yins.append(np.asarray(d["y_inner"], float))
        if not all(np.array_equal(ys[0], o) for o in ys[1:]):
            warns.append(f"union4/{year}: held labels differ across arms -> fold skipped")
            continue
        y = ys[0]
        p = np.mean(ps, axis=0)
        if all(np.array_equal(yins[0], o) for o in yins[1:]):
            thr = pick_threshold(np.mean(pins, axis=0), yins[0])
        else:
            thrs = [analyses[a]["folds"][year]["threshold"] for a in members]
            thrs = [t for t in thrs if t is not None and math.isfinite(t)]
            if not thrs:
                warns.append(f"union4/{year}: inner sets differ and no member thresholds")
                continue
            thr = float(np.mean(thrs))
        pod, far = pod_far(p, y, thr)
        roc, pr = safe_auc(p, y)
        folds[year] = {
            "n": len(y), "ri_rate": float(y.mean()), "threshold": thr,
            "POD": pod, "FAR": far, "BSS": bss_score(p, y),
            "ROC_AUC": roc, "PR_AUC": pr,
            **murphy_decomposition(p, y, bins),
        }
        p_pool.append(p)
        y_pool.append(y)
    if not folds:
        warns.append("union4 skipped: no aligned folds")
        return None
    pp, yy = np.concatenate(p_pool), np.concatenate(y_pool)
    bss = [f["BSS"] for f in folds.values()]
    return {
        "arm": "union4", "folds": folds, "n_folds": len(folds), "csv": None,
        "bss_mean": float(np.mean(bss)), "bss_worst": float(np.min(bss)),
        "bss_std": float(np.std(bss)), "inner_bss": float("nan"),
        "pooled": {
            "n": len(yy), "ri_rate": float(yy.mean()),
            **murphy_decomposition(pp, yy, bins),
            "calib_slope": calibration_slope(pp, yy),
            "rel_curve": reliability_curve(pp, yy, bins),
            "event_bss": event_focused_bss(pp, yy),
        },
        "provenance": "preds",
        "note": f"members: {', '.join(members)}",
    }


# --------------------------------------------------------------------------
# reconciliation vs CSV
# --------------------------------------------------------------------------

def reconcile(res: dict, warns: list[str]) -> None:
    csv = res["csv"]
    if csv is None or res["provenance"] != "preds":
        return
    arm = res["arm"]
    for year, f in res["folds"].items():
        if year not in csv.index:
            warns.append(f"{arm}/{year}: fold absent from CSV")
            continue
        for key in ("POD", "FAR", "BSS"):
            if key not in csv.columns:
                continue
            ref = float(csv.loc[year, key])
            got = f[key]
            if math.isfinite(got) and abs(got - ref) > RECON_TOL:
                warns.append(f"{arm}/{year}: recomputed {key}={got:.6f} "
                             f"vs CSV {ref:.6f} (|d|={abs(got-ref):.2e})")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def arm_sort_key(name: str):
    if name in MAIN_ORDER:
        return (0, MAIN_ORDER.index(name), name)
    if re.search(r"_s\d+$", name):
        base = re.sub(r"_s\d+$", "", name)
        bi = MAIN_ORDER.index(base) if base in MAIN_ORDER else 99
        return (1, bi, name)
    if name.startswith("ablate"):
        return (2, 0, name)
    if name.startswith("baseline"):
        return (3, 0, name)
    return (4, 0, name)


def fig_heatmap(analyses: list[dict], fig_dir: Path, prefix: str) -> Path:
    arms = [a for a in analyses if a["folds"]]
    arms.sort(key=lambda a: arm_sort_key(a["arm"]))
    years = YEARS
    mat = np.full((len(arms), len(years) + 1), np.nan)
    for i, a in enumerate(arms):
        for j, y in enumerate(years):
            if y in a["folds"]:
                mat[i, j] = a["folds"][y]["BSS"]
        mat[i, -1] = a["bss_mean"]
    fig, ax = plt.subplots(figsize=(1.2 * (len(years) + 2), 0.42 * len(arms) + 1.5))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 0.3
    vmax = max(vmax, 0.05)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(years) + 1), [str(y) for y in years] + ["MEAN"])
    labels = [a["arm"] + (f" ({a['n_folds']} folds)" if a["n_folds"] < 6 else "")
              for a in arms]
    ax.set_yticks(range(len(arms)), labels)
    for i in range(len(arms)):
        for j in range(len(years) + 1):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="gray")
    ax.set_title("Held-year BSS (vs fold climatology)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="BSS")
    fig.tight_layout()
    out = fig_dir / f"{prefix}_bss_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_reliability_grid(analyses: dict[str, dict], fig_dir: Path,
                         prefix: str) -> Path | None:
    wanted = ["ens_v1", "ens_lingbotB", "ens_lingbotL", "ens_lingbotG",
              "ens_ep40", "ens_ep40ft", "union4"]
    arms = [a for a in wanted
            if a in analyses and analyses[a].get("pooled")]
    if not arms:
        return None
    ncol = 3
    nrow = math.ceil(len(arms) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.0 * nrow),
                             squeeze=False)
    for k, name in enumerate(arms):
        ax = axes[k // ncol][k % ncol]
        rel = analyses[name]["pooled"]["rel_curve"]
        clim = analyses[name]["pooled"]["ri_rate"]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.axhline(clim, color="gray", ls=":", lw=1,
                   label=f"clim={clim:.3f}")
        m = rel["n"] > 0
        ax.plot(rel.loc[m, "p_mean"], rel.loc[m, "obs_freq"], "o-",
                color="tab:blue", label="model")
        ax.set_title(f"{name} (N={analyses[name]['pooled']['n']})")
        ax.set_xlabel("forecast probability")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    for k in range(len(arms), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Reliability diagrams (pooled folds, 10 bins)")
    fig.tight_layout()
    out = fig_dir / f"{prefix}_reliability_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_pod_far(analyses: list[dict], fig_dir: Path, prefix: str) -> Path | None:
    arms = [a for a in analyses
            if a["folds"] and all("POD" in f for f in a["folds"].values())]
    if not arms:
        return None
    arms.sort(key=lambda a: arm_sort_key(a["arm"]))
    fig, ax = plt.subplots(figsize=(8, 7))
    all_ri = [f["ri_rate"] for a in arms for f in a["folds"].values()
              if math.isfinite(f.get("ri_rate", np.nan))]
    for a in arms:
        pods = [f["POD"] for f in a["folds"].values()]
        fars = [f["FAR"] for f in a["folds"].values()]
        if not all(math.isfinite(v) for v in pods + fars):
            continue
        ax.errorbar(np.mean(fars), np.mean(pods),
                    xerr=np.std(fars), yerr=np.std(pods),
                    fmt="o", capsize=3)
        ax.annotate(a["arm"], (np.mean(fars), np.mean(pods)),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    if all_ri:
        clim_far = 1.0 - float(np.mean(all_ri))
        ax.axvline(clim_far, color="red", ls="--", lw=1,
                   label=f"climatology FAR=1-mean_ri_rate={clim_far:.3f}")
    ax.set_xlabel("FAR (mean ± fold std)")
    ax.set_ylabel("POD (mean ± fold std)")
    ax.set_title("POD–FAR per arm (LOYO folds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / f"{prefix}_pod_far.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_bss_bars(analyses: list[dict], fig_dir: Path, prefix: str) -> Path:
    arms = [a for a in analyses if a["folds"]]
    arms.sort(key=lambda a: -a["bss_mean"])
    names = [a["arm"] for a in arms]
    means = [a["bss_mean"] for a in arms]
    worst = [a["bss_worst"] for a in arms]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(names)), 5))
    ax.bar(x - 0.2, means, width=0.4, label="mean BSS", color="tab:blue")
    ax.bar(x + 0.2, worst, width=0.4, label="worst-fold BSS", color="tab:orange")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("BSS")
    ax.set_title("BSS mean vs worst fold (sorted by mean)")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / f"{prefix}_bss_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_lingbot_ladder(analyses: dict[str, dict], fig_dir: Path,
                       prefix: str) -> Path | None:
    """LingBot scale ladder S/B/L/G: inner_bss vs held BSS."""
    steps = [("S", "ens_lingbot"), ("B", "ens_lingbotB"),
             ("L", "ens_lingbotL"), ("G", "ens_lingbotG")]
    have = [(lab, analyses[arm]) for lab, arm in steps if arm in analyses]
    if len(have) < 2:
        return None
    x = np.arange(len(have))
    inner = [a["inner_bss"] for _, a in have]
    held = [a["bss_mean"] for _, a in have]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, inner, "o-", label="inner_bss (val, CSV mean)", color="tab:green")
    ax.plot(x, held, "s-", label="held BSS (mean over folds)", color="tab:blue")
    ax.set_xticks(x, [lab for lab, _ in have])
    ax.set_xlabel("LingBot model scale")
    ax.set_ylabel("BSS")
    ax.set_title("LingBot scale ladder")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / f"{prefix}_lingbot_ladder.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_calib_slopes(analyses: list[dict], fig_dir: Path,
                     prefix: str) -> Path | None:
    arms = [a for a in analyses
            if a.get("pooled") and math.isfinite(a["pooled"]["calib_slope"])]
    if not arms:
        return None
    arms.sort(key=lambda a: arm_sort_key(a["arm"]))
    names = [a["arm"] for a in arms]
    vals = [a["pooled"]["calib_slope"] for a in arms]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(names)), 4.5))
    colors = ["tab:blue" if 0.8 <= v <= 1.25 else "tab:orange" for v in vals]
    ax.bar(np.arange(len(names)), vals, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="ideal slope = 1")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right",
                  fontsize=8)
    ax.set_ylabel("calibration slope")
    ax.set_title("Calibration slope of y ~ logit(p) (pooled folds)")
    ax.legend()
    fig.tight_layout()
    out = fig_dir / f"{prefix}_calib_slopes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def f3(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) and math.isfinite(v) else "—"


def summary_table(analyses: list[dict]) -> pd.DataFrame:
    rows = []
    for a in analyses:
        pods = [f["POD"] for f in a["folds"].values()]
        fars = [f["FAR"] for f in a["folds"].values()]
        prs = [f["PR_AUC"] for f in a["folds"].values()]
        rocs = [f["ROC_AUC"] for f in a["folds"].values()]
        pooled = a.get("pooled") or {}
        rows.append({
            "arm": a["arm"], "n_folds": a["n_folds"],
            "BSS_mean": a["bss_mean"], "BSS_worst": a["bss_worst"],
            "BSS_std": a["bss_std"],
            "PR_AUC": float(np.nanmean(prs)) if prs else np.nan,
            "ROC_AUC": float(np.nanmean(rocs)) if rocs else np.nan,
            "POD": float(np.nanmean(pods)) if pods else np.nan,
            "FAR": float(np.nanmean(fars)) if fars else np.nan,
            "calib_slope": pooled.get("calib_slope", np.nan),
            "resolution": pooled.get("resolution", np.nan),
            "reliability": pooled.get("reliability", np.nan),
            "event_bss": (pooled.get("event_bss") or (np.nan, 0))[0],
            "src": a["provenance"],
        })
    df = pd.DataFrame(rows)
    df["_k"] = df["arm"].map(lambda n: arm_sort_key(n))
    return df.sort_values("_k").drop(columns="_k").reset_index(drop=True)


def per_fold_table(analyses: list[dict]) -> pd.DataFrame:
    rows = []
    for a in analyses:
        for y in YEARS:
            f = a["folds"].get(y)
            if f is None:
                continue
            rows.append({"arm": a["arm"], "year": y, "n": f["n"],
                         "ri_rate": f["ri_rate"], "thr": f["threshold"],
                         "POD": f["POD"], "FAR": f["FAR"], "BSS": f["BSS"],
                         "ROC_AUC": f.get("ROC_AUC", np.nan),
                         "PR_AUC": f.get("PR_AUC", np.nan),
                         "REL": f.get("reliability", np.nan),
                         "RES": f.get("resolution", np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_k"] = df["arm"].map(lambda n: arm_sort_key(n))
    return df.sort_values(["_k", "year"]).drop(columns="_k").reset_index(drop=True)


def md_table(df: pd.DataFrame, float_cols: list[str]) -> str:
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.3f}"
                              if isinstance(v, (int, float))
                              and math.isfinite(v) else "—")
    return df.to_markdown(index=False)


def build_report(analyses: dict[str, dict], ordered: list[dict],
                 warns: list[str], figures: list[Path | None],
                 args) -> str:
    L: list[str] = []
    sdf = summary_table(ordered)
    preds_arms = [a for a in ordered if a["provenance"] == "preds"]
    best_bss = max(ordered, key=lambda a: a["bss_mean"])
    full_arms = [a for a in ordered if a["n_folds"] >= 6] or ordered
    best_worst = max(full_arms, key=lambda a: a["bss_worst"])
    best_pr = max(ordered, key=lambda a: float(np.nanmean(
        [f["PR_AUC"] for f in a["folds"].values()] or [np.nan])))
    calib = {a["arm"]: a["pooled"]["calib_slope"]
             for a in preds_arms if a.get("pooled")}
    best_cal = min(calib, key=lambda k: abs(calib[k] - 1.0)) if calib else None
    union = analyses.get("union4")

    L.append(f"# {args.out_prefix.upper()} — final full-model output analysis")
    L.append("")
    L.append(f"Generation command: `python scripts/05_eval/final_model_analysis.py "
             f"{' '.join(sys.argv[1:])}`  ")
    L.append(f"Prediction directories: {', '.join(str(d) for d in args.preds_dirs)}  ")
    L.append(f"Decomposition bins: {args.bins}; reconciliation tolerance: {RECON_TOL}")
    L.append("")
    L.append("## TL;DR")
    L.append("")
    L.append(f"- **Highest mean BSS**: `{best_bss['arm']}` = "
             f"**{best_bss['bss_mean']:.4f}** (worst fold {best_bss['bss_worst']:.4f})")
    L.append(f"- **Best worst-fold score (most robust among arms with all 6 folds)**: "
             f"`{best_worst['arm']}`, worst-fold BSS = **{best_worst['bss_worst']:.4f}**")
    L.append(f"- **Highest PR-AUC**: `{best_pr['arm']}` = "
             f"{np.nanmean([f['PR_AUC'] for f in best_pr['folds'].values()]):.4f}")
    if best_cal:
        L.append(f"- **Calibration slope closest to 1**: `{best_cal}` = {calib[best_cal]:.3f}")
    if union:
        singles = [a for a in preds_arms if a["arm"] != "union4"]
        best_single = max(singles, key=lambda a: a["bss_mean"])
        d = union["bss_mean"] - best_single["bss_mean"]
        L.append(f"- **union4** (mean probability of {'+'.join(UNION4_MEMBERS)}) mean BSS = "
                 f"{union['bss_mean']:.4f}, versus best single arm "
                 f"`{best_single['arm']}` {d:+.4f}")
    csv_only = [a for a in ordered if a["provenance"] == "csv"]
    if csv_only:
        bl = next((a for a in csv_only if a["arm"] == "baseline_lr"), None)
        if bl:
            L.append(f"- Baseline logistic mean BSS = {bl['bss_mean']:.4f}; "
                     f"improvement of the best model arm "
                     f"{best_bss['bss_mean'] - bl['bss_mean']:+.4f}")
    missing = [a for a in ordered if a["n_folds"] < 6]
    if missing:
        L.append(f"- WARNING: arms with missing folds: "
                 + ", ".join(f"`{a['arm']}`({a['n_folds']}/6)" for a in missing))
    L.append("")

    L.append("## Summary (arm x metrics; recomputed from raw NPZ predictions except src=csv)")
    L.append("")
    L.append(md_table(sdf, ["BSS_mean", "BSS_worst", "BSS_std", "PR_AUC",
                            "ROC_AUC", "POD", "FAR", "calib_slope",
                            "resolution", "reliability", "event_bss"]))
    L.append("")

    L.append("## Per-fold details")
    L.append("")
    fdf = per_fold_table(ordered)
    if not fdf.empty:
        L.append(md_table(fdf, ["ri_rate", "thr", "POD", "FAR", "BSS",
                                "ROC_AUC", "PR_AUC", "REL", "RES"]))
    L.append("")

    # --- ablation section
    abl = [a for a in ordered if a["arm"].startswith("ablate")]
    if abl:
        L.append("## Ablation comparison (main arm vs ablate_*)")
        L.append("")
        ref = next((a for a in ordered if a["arm"] == "ens_v1"), None)
        if ref is not None:
            L.append(f"Reference main arm `ens_v1`: mean BSS {ref['bss_mean']:.4f}")
            L.append("")
            for a in abl:
                d = a["bss_mean"] - ref["bss_mean"]
                L.append(f"- `{a['arm']}`: mean BSS {a['bss_mean']:.4f} "
                         f"({d:+.4f} vs ens_v1), "
                         f"PR-AUC {np.nanmean([f['PR_AUC'] for f in a['folds'].values()]):.4f}")
        else:
            for a in abl:
                L.append(f"- `{a['arm']}`: mean BSS {a['bss_mean']:.4f}")
        L.append("")

    bl = [a for a in ordered if a["arm"].startswith("baseline")]
    if bl:
        L.append("## Baseline comparison (baseline_lr / baseline_lda)")
        L.append("")
        for a in bl:
            L.append(f"- `{a['arm']}`: mean BSS {a['bss_mean']:.4f}, "
                     f"POD {np.nanmean([f['POD'] for f in a['folds'].values()]):.3f}, "
                     f"FAR {np.nanmean([f['FAR'] for f in a['folds'].values()]):.3f}, "
                     f"PR-AUC {np.nanmean([f['PR_AUC'] for f in a['folds'].values()]):.4f}")
        L.append("")

    # --- findings
    L.append("## Automated findings")
    L.append("")
    findings = []
    if best_worst["arm"] != best_bss["arm"]:
        findings.append(f"The highest-mean arm ({best_bss['arm']}) differs from the "
                        f"best worst-fold arm ({best_worst['arm']}): the former has "
                        f"the higher peak, whereas the latter is more robust.")
    else:
        findings.append(f"`{best_bss['arm']}` has both the highest mean and the best "
                        f"worst-fold score.")
    if best_cal:
        findings.append(f"`{best_cal}` has the calibration slope closest to 1 "
                        f"({calib[best_cal]:.3f}).")
        worst_cal = max(calib, key=lambda k: abs(calib[k] - 1.0))
        findings.append(f"`{worst_cal}` has the worst calibration ({calib[worst_cal]:.3f}); "
                        f"a slope <1 indicates overconfidence, and >1 underconfidence.")
    res_map = {a["arm"]: a["pooled"]["resolution"] for a in preds_arms
               if a.get("pooled")}
    if res_map:
        top_res = max(res_map, key=res_map.get)
        nfold = analyses[top_res]["n_folds"]
        note = f" (only {nfold} folds)" if nfold < 6 else ""
        findings.append(f"`{top_res}` has the highest Murphy resolution "
                        f"({res_map[top_res]:.4f}){note}, indicating the strongest "
                        f"event/non-event discrimination.")
    if union:
        singles = {a["arm"]: a for a in preds_arms if a["arm"] != "union4"}
        n_win = sum(1 for a in singles.values()
                    if union["bss_mean"] >= a["bss_mean"])
        findings.append(f"union4's mean BSS ({union['bss_mean']:.4f}) exceeds "
                        f"{n_win}/{len(singles)} single arms.")
    seed_groups: dict[str, list[dict]] = {}
    for a in ordered:
        m = re.match(r"(.+)_s(\d+)$", a["arm"])
        if m:
            seed_groups.setdefault(m.group(1), []).append(a)
    for base, group in seed_groups.items():
        base_a = analyses.get(base)
        if base_a is None:
            continue
        allv = [base_a["bss_mean"]] + [g["bss_mean"] for g in group]
        findings.append(f"`{base}` mean BSS across seeds ranges from "
                        f"{min(allv):.4f} to {max(allv):.4f}; "
                        f"seed sensitivity +/-{(max(allv)-min(allv))/2:.4f}.")
    for s in findings:
        L.append(f"- {s}")
    L.append("")

    # --- missing data
    L.append("## Missing-data annotations")
    L.append("")
    if missing:
        for a in missing:
            have = sorted(a["folds"])
            miss = [y for y in YEARS if y not in a["folds"]]
            L.append(f"- `{a['arm']}`: only {a['n_folds']}/6 folds "
                     f"(available {have}, missing {miss})")
    else:
        L.append("- All arms have complete six-fold results.")
    for a in csv_only:
        L.append(f"- `{a['arm']}`: row-level predictions unavailable; metrics come "
                 f"from CSV (calibration/decomposition/event analysis unavailable).")
    expected_absent = [a for a in ["ens_lingbotL", "ens_lingbotG", "ens_ep40_srv"]
                       if a not in analyses]
    if expected_absent:
        L.append(f"- Expected server arms not yet available: "
                 + ", ".join(f"`{a}`" for a in expected_absent)
                 + "; place them in the prediction directories and rerun this script.")
    L.append("")

    # --- figures
    L.append("## Figures")
    L.append("")
    for fp in figures:
        if fp is not None:
            try:
                L.append(f"- `{fp.relative_to(P2_ROOT)}`")
            except ValueError:
                L.append(f"- `{fp}`")
    L.append("")

    # --- reconciliation
    L.append("## Reconciliation (NPZ recomputation vs CSV, tolerance "
             f"{RECON_TOL})")
    L.append("")
    if warns:
        L.append(f"{len(warns)} WARN messages:")
        L.append("")
        for w in warns:
            L.append(f"- WARN: {w}")
    else:
        L.append("All values agree; no WARN messages.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dirs", nargs="+", action="extend",
                    default=[], required=True,
                    help="one or more dirs with <arm>_fold<year>.npz")
    ap.add_argument("--out-prefix", default="final")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--data-dir", default=str(P2_ROOT / "data"))
    ap.add_argument("--fig-dir", default=str(P2_ROOT / "analysis" / "figures"))
    ap.add_argument("--doc-dir", default=str(P2_ROOT / "docs"))
    args = ap.parse_args()

    preds_dirs = [Path(d) for d in args.preds_dirs]
    data_dir = Path(args.data_dir)
    fig_dir = Path(args.fig_dir)
    doc_dir = Path(args.doc_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    warns: list[str] = []
    fold_map = discover_arms(preds_dirs)
    print(f"discovered {len(fold_map)} arms with preds: "
          f"{sorted(fold_map)}")

    analyses: dict[str, dict] = {}
    for arm, paths in sorted(fold_map.items()):
        csv = load_csv_for_arm(arm, data_dir)
        if csv is None:
            warns.append(f"{arm}: no LOYO CSV found -> POD/FAR thresholds unavailable")
        res = analyze_arm(arm, paths, csv, args.bins, warns)
        res["fold_paths"] = paths
        analyses[arm] = res
        reconcile(res, warns)

    # CSV-only arms (ablations / baselines)
    for name in CSV_ONLY_PATTERNS:
        csv = load_csv_for_arm(name, data_dir)
        if csv is not None and name not in analyses:
            analyses[name] = analyze_csv_only_arm(name, csv)

    u4 = build_union4(analyses, args.bins, warns)
    if u4 is not None:
        analyses["union4"] = u4

    ordered = sorted(analyses.values(), key=lambda a: arm_sort_key(a["arm"]))

    figures = [
        fig_heatmap(ordered, fig_dir, args.out_prefix),
        fig_reliability_grid(analyses, fig_dir, args.out_prefix),
        fig_pod_far(ordered, fig_dir, args.out_prefix),
        fig_bss_bars(ordered, fig_dir, args.out_prefix),
        fig_lingbot_ladder(analyses, fig_dir, args.out_prefix),
        fig_calib_slopes(ordered, fig_dir, args.out_prefix),
    ]
    for fp in figures:
        if fp is not None:
            print(f"wrote {fp}")

    report = build_report(analyses, ordered, warns, figures, args)
    out_md = doc_dir / f"{args.out_prefix.upper()}_ANALYSIS.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"wrote {out_md}")

    sdf = summary_table(ordered)
    print("\n=== summary ===")
    print(sdf.round(4).to_string(index=False))
    if warns:
        print(f"\n=== {len(warns)} WARN ===")
        for w in warns:
            print("WARN:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
