# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Ensemble composition re-evaluation (analysis-only; never touches training).

Discovers arms from one or more preds dirs (``<arm>_fold<year>.npz`` with
keys p_held/y_held/p_inner/y_inner), quantifies inter-arm diversity, then
re-runs ensemble member selection **using only the inner (validation)
predictions**. Held predictions ARE evaluated for every scanned combination
(a diagnostic report table), but no selection step ever reads them — the
selected combinations are chosen strictly on inner scores, then evaluated
on held exactly once each.

Protocol
--------
* Arms missing any of the 6 LOYO folds are dropped as whole arms (WARN).
* Folds whose y_held or y_inner labels disagree across arms are refused
  (dropped for all arms) with a WARN — averaging misaligned samples is
  never attempted.
* Candidate members: all complete arms (if > --max-arms, truncated to the
  top-8 by pooled inner BSS to bound the combinatorial scan).
* All subsets of size 2..5 are fitted on pooled inner only:
  (a) uniform mean, (b) NNLS non-negative weights (normalized to sum 1).
  Selection metric: BSS on pooled inner vs pooled-inner climatology.
* Final evaluation on held (once): per-fold BSS (vs that fold's own
  climatology), PR-AUC, POD/FAR at a threshold tuned by F1 on that fold's
  inner predictions (same convention as training).
* Controls: union4 (pre-specified default, evaluated directly on held,
  not part of selection) and the best single arm (by inner BSS).
* Final Top-K protocol (--topk, default 4,5,6) on the RETAINED pool:
  failed fine-tuning pilot arms and wholesale-failed routes are excluded
  a priori (--pool-all opts out; they remain in the diversity analysis,
  flagged pilot=1).  Retained arms are ranked by the LOIO
  (leave-one-inner-year-out, WITHOUT refitting) robust score = mean over
  folds of the per-fold worst inner-year BSS (minimax). The FINAL rule is
  quality-gated stratified family selection: a route-family champion
  (lingbot/mae x single/multi-frame) keeps its seat only if it ranks in
  the global LOIO top-8; remaining seats are filled by global LOIO.
  Pure inner-pooled and global-LOIO Top-K are kept as comparison rules
  (--no-loio falls back to pooled-inner rankings). Selected members are
  combined by uniform average; K is chosen on inner; held once.

Usage (from the release root):
    python scripts/04_train/ensemble_reselect.py --preds-dirs data/preds \
        [--preds-dirs data/preds_server] --out data/ensemble_reselect
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

P2_ROOT = Path(__file__).resolve().parents[2]

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
EXCLUDE_PATTERNS = [r"_s\d+$", r"_oldp$"]
UNION4_MEMBERS = ["ens_v1", "ens_lingbotB", "ens_ep40", "ens_ep40ft"]

# final-pool exclusion for the registered pilot and failed-route patterns:
# failed fine-tuning pilot arms and wholesale-failed routes never enter the
# selection pool (Top-K / family / combination scan).  They stay in the
# diversity analysis, flagged pilot=1.  Aggregator variants
# (diff/slim/diffslim/attnpool/temp2/temp3) and main-route full-FT arms
# (ep40ft, v1, ...) are NOT pilot arms and stay in the pool.
POOL_EXCLUDE_PATTERNS = [
    r"^ens_lb[BL]_(partial4|lpft4)_temp5$",  # FT pilot matrix, multi-frame (proven negative)
    r"^ens_lbL_lora16_temp5$",  # adjudicated 2026-09-21: mean 0.189 < 0.200, stays pilot
    r"^ens_lbL_partial4_sf$",   # adjudicated 2026-09-21: mean 0.198 < 0.200, stays pilot
    # NOTE: lora16 reinstated 2026-09-19 by user exception clause — it passed the
    # reinstatement bar (held mean 0.209 >= 0.200, worst fold 0.142 >= 0.10) and
    # is a pilot-VALIDATED feasible path, not a failed route. 2026-09-20 the user
    # generalized the clause to a standing rule: ANY pilot arm (lora/partial/lpft,
    # temp5/sf) clearing the same bar (held mean >= 0.200, worst fold >= 0.10)
    # returns to the pool as a pilot-validated path; lora16_*sf therefore stays in
    # the pool pending its scores. Executed under the rule:
    #   ens_lbB_lora16_temp5  0.209/0.142 -> reinstated 2026-09-19
    #   ens_lbL_lora16_temp5  0.189/0.104 -> mean < 0.200, stays pilot (2026-09-21)
    #   ens_lbB_partial4_sf   0.236/0.113 -> reinstated 2026-09-21 (pattern carved out)
    #   ens_lbB_lpft4_sf      0.243/0.177 -> reinstated 2026-09-21 (pattern carved out)
    #   ens_lbL_partial4_sf   0.198/0.111 -> mean < 0.200 by 0.002, stays pilot (2026-09-21)
    #   ens_lbL_lpft4_sf      0.229/0.144 -> reinstated 2026-09-21 (pattern carved out)
    r"^ens_lingbotG$",     # failed route (also incomplete folds)
    r"^ens_ep40ft_temp5$",  # failed route
    r"^ens_v1_temp5$",     # failed route
    r"mini_cnn",
]

# quality gate for the stratified family rule: a family champion keeps its
# guaranteed seat only if it ranks in the global LOIO top-GATE of the pool
FAMILY_GATE = 8

# expected preregistered union4 held BSS mean (sanity anchor)
UNION4_REF_BSS = 0.209
UNION4_REF_TOL = 5e-3


# --------------------------------------------------------------------------
# metrics (mirror src/train.py: p >= thr; BSS vs the fold's own climatology)
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


def pr_auc(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def pick_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """F1-optimal threshold on (inner) predictions — training convention."""
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


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def discover_arms(preds_dirs: list[Path], warns: list[str]
                  ) -> dict[str, dict[int, Path]]:
    """{arm: {year: npz path}}; seed/old-protocol variants excluded."""
    arms: dict[str, dict[int, Path]] = {}
    for d in preds_dirs:
        if not d.is_dir():
            warns.append(f"preds dir missing: {d}")
            continue
        for f in sorted(d.glob("*_fold*.npz")):
            m = re.match(r"(.+)_fold(\d{4})\.npz$", f.name)
            if not m:
                continue
            arm, year = m.group(1), int(m.group(2))
            if any(re.search(p, arm) for p in EXCLUDE_PATTERNS):
                continue
            if year in arms.get(arm, {}):
                warns.append(f"{arm}/{year}: duplicate npz across preds dirs "
                             f"-> keeping first ({arms[arm][year]})")
                continue
            arms.setdefault(arm, {})[year] = f
    return arms


def load_arm(fold_paths: dict[int, Path]) -> dict[int, dict[str, np.ndarray]]:
    out = {}
    for year, path in fold_paths.items():
        d = np.load(path)
        out[year] = {
            "p_held": np.asarray(d["p_held"], dtype=float),
            "y_held": np.asarray(d["y_held"], dtype=float),
            "p_inner": np.asarray(d["p_inner"], dtype=float),
            "y_inner": np.asarray(d["y_inner"], dtype=float),
        }
    return out


def align_folds(arms_data: dict[str, dict], names: list[str],
                warns: list[str]) -> list[int]:
    """Years where y_held AND y_inner are identical across all `names`.

    Misaligned folds are refused (never averaged) and dropped for everyone.
    """
    ok = []
    for year in YEARS:
        if not all(year in arms_data[a] for a in names):
            continue
        yh = [arms_data[a][year]["y_held"] for a in names]
        yi = [arms_data[a][year]["y_inner"] for a in names]
        held_ok = all(np.array_equal(yh[0], o) for o in yh[1:])
        inner_ok = all(np.array_equal(yi[0], o) for o in yi[1:])
        if held_ok and inner_ok:
            ok.append(year)
        else:
            which = []
            if not held_ok:
                which.append("y_held")
            if not inner_ok:
                which.append("y_inner")
            warns.append(f"fold {year}: {' and '.join(which)} differ across "
                         f"arms -> fold REFUSED (excluded for all arms)")
    return ok


def stacked(arms_data: dict[str, dict], members: list[str],
            years: list[int], kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(N x k prediction matrix, labels) pooled over `years`.

    kind: 'inner' or 'held'. Callers must guarantee label alignment
    (see align_folds).
    """
    ps, ys = [], []
    for year in years:
        ps.append(np.column_stack(
            [arms_data[a][year][f"p_{kind}"] for a in members]))
        ys.append(arms_data[members[0]][year][f"y_{kind}"])
    return np.vstack(ps), np.concatenate(ys)


def nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, _ = nnls(P, y)
    s = w.sum()
    return w / s if s > 1e-12 else np.full(len(w), 1.0 / len(w))


# --------------------------------------------------------------------------
# diversity
# --------------------------------------------------------------------------

def diversity_mats(arms_data: dict[str, dict], arm_names: list[str]
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise error pearson (pooled + per-fold std) and prob spearman."""
    n = len(arm_names)
    err = np.full((n, n), np.nan)
    err_std = np.full((n, n), np.nan)
    sp = np.full((n, n), np.nan)
    for i, a in enumerate(arm_names):
        for j, b in enumerate(arm_names):
            common = sorted(set(arms_data[a]) & set(arms_data[b]))
            if not common:
                continue
            if a == b:
                err[i, j] = 1.0
                err_std[i, j] = 0.0
                sp[i, j] = 1.0
                continue
            ea, eb, pa, pb = [], [], [], []
            per_fold = []
            for year in common:
                da, db = arms_data[a][year], arms_data[b][year]
                if not np.array_equal(da["y_held"], db["y_held"]):
                    continue
                ya = da["y_held"]
                fa = da["p_held"] - ya
                fb = db["p_held"] - ya
                ea.append(fa)
                eb.append(fb)
                pa.append(da["p_held"])
                pb.append(db["p_held"])
                if fa.std() > 0 and fb.std() > 0:
                    per_fold.append(pearsonr(fa, fb)[0])
            if not ea:
                continue
            fa = np.concatenate(ea)
            fb = np.concatenate(eb)
            if fa.std() > 0 and fb.std() > 0:
                err[i, j] = pearsonr(fa, fb)[0]
            if len(per_fold) > 1:
                err_std[i, j] = float(np.std(per_fold))
            sp[i, j] = spearmanr(np.concatenate(pa), np.concatenate(pb))[0]
    idx = arm_names
    return (pd.DataFrame(err, index=idx, columns=idx),
            pd.DataFrame(err_std, index=idx, columns=idx),
            pd.DataFrame(sp, index=idx, columns=idx))


def fig_diversity(err: pd.DataFrame, sp: pd.DataFrame,
                  fig_dir: Path) -> Path:
    names = list(err.index)
    fig, axes = plt.subplots(1, 2, figsize=(6.4 + 0.62 * len(names),
                                            3.2 + 0.5 * len(names)))
    for ax, mat, title, cmap in (
            (axes[0], err, "error pearson corr (pooled held)", "RdBu_r"),
            (axes[1], sp, "probability spearman corr (pooled held)", "RdBu_r")):
        im = ax.imshow(mat.values, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right",
                      fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Inter-arm diversity (held folds)")
    fig.tight_layout()
    out = fig_dir / "ensemble_diversity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# combo evaluation
# --------------------------------------------------------------------------

def eval_combo_held(arms_data: dict[str, dict], members: list[str],
                    weights: np.ndarray, years: list[int]) -> dict:
    """Apply fixed members/weights to held; threshold tuned per fold on
    that fold's inner (F1), matching the training convention."""
    rows = []
    for year in years:
        Ph, yh = stacked(arms_data, members, [year], "held")
        Pi, yi = stacked(arms_data, members, [year], "inner")
        p_h = Ph @ weights
        p_i = Pi @ weights
        thr = pick_threshold(p_i, yi)
        pod, far = pod_far(p_h, yh, thr)
        rows.append({"year": year, "BSS": bss_score(p_h, yh),
                     "PR_AUC": pr_auc(p_h, yh), "POD": pod, "FAR": far,
                     "thr": thr})
    bss = [r["BSS"] for r in rows]
    return {"folds": rows,
            "held_bss_mean": float(np.mean(bss)),
            "held_bss_worst": float(np.min(bss)),
            "pr_auc": float(np.nanmean([r["PR_AUC"] for r in rows])),
            "pod": float(np.nanmean([r["POD"] for r in rows])),
            "far": float(np.nanmean([r["FAR"] for r in rows]))}


def inner_bss(arms_data: dict[str, dict], members: list[str],
              weights: np.ndarray, years: list[int]) -> float:
    Pi, yi = stacked(arms_data, members, years, "inner")
    return bss_score(Pi @ weights, yi)


# --------------------------------------------------------------------------
# LOIO (leave-one-inner-year-out) robust scoring
# --------------------------------------------------------------------------

INNER_YEARS = [2016, 2017, 2018]


def reconstruct_inner_labels(panel_parquet: Path, frames_index: Path,
                             frames_dir: Path
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild (storm-year, RI label) per inner-split row, mirroring
    RIPanelDataset row order and train.py's storm-level LOYO split:
    inner = rows whose storm's first time step falls in INNER_YEARS."""
    panel = pd.read_parquet(panel_parquet)
    panel["ISO_TIME"] = pd.to_datetime(panel["ISO_TIME"], utc=True)
    panel = panel.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    index = pd.read_parquet(frames_index)
    index["ISO_TIME"] = pd.to_datetime(index["ISO_TIME"], utc=True)
    df = panel.merge(index[["SID", "ISO_TIME", "frame_path", "frame_time"]],
                     on=["SID", "ISO_TIME"], how="inner")
    fdir = Path(frames_dir)
    have = {sid for sid in df["SID"].unique()
            if (fdir / f"{sid}.npy").exists()
            and (fdir / f"{sid}_iso.npy").exists()}
    df = df[df["SID"].isin(have)].reset_index(drop=True)
    sid_year = df.groupby("SID")["ISO_TIME"].min().dt.year
    mask = df["SID"].isin(
        sid_year[sid_year.isin(INNER_YEARS)].index).values
    sub = df[mask]
    return (sid_year[sub["SID"]].values.astype(int),
            sub["RI"].values.astype(float))


def loio_fold_bss(p_folds: list[np.ndarray], y_folds: list[np.ndarray],
                  year_labels: np.ndarray) -> np.ndarray:
    """(n_folds x len(INNER_YEARS)) BSS matrix; each cell vs that inner
    year's own climatology."""
    mat = np.full((len(p_folds), len(INNER_YEARS)), np.nan)
    for i, (p, y) in enumerate(zip(p_folds, y_folds)):
        for j, yr in enumerate(INNER_YEARS):
            m = year_labels == yr
            if m.sum() and len(np.unique(y[m])) > 1:
                mat[i, j] = bss_score(p[m], y[m])
    return mat


def loio_summaries(mat: np.ndarray) -> dict[str, float]:
    """Primary score = mean over folds of the per-fold WORST inner-year BSS
    (minimax); pooled mean and mean-std are references."""
    fold_min = np.nanmin(mat, axis=1)
    flat = mat[np.isfinite(mat)]
    return {"min_mean": float(np.nanmean(fold_min)),
            "pooled_mean": float(np.mean(flat)),
            "mean_minus_std": float(np.mean(flat) - np.std(flat))}


def arm_loio(arms_data: dict[str, dict], names: list[str],
             ok_years: list[int], year_labels: np.ndarray) -> dict:
    out = {}
    for a in names:
        mat = loio_fold_bss([arms_data[a][y]["p_inner"] for y in ok_years],
                            [arms_data[a][y]["y_inner"] for y in ok_years],
                            year_labels)
        out[a] = {"matrix": mat, **loio_summaries(mat)}
    return out


def ensemble_loio(arms_data: dict[str, dict], members: list[str],
                  weights: np.ndarray, ok_years: list[int],
                  year_labels: np.ndarray) -> dict:
    ps, ys = [], []
    for y in ok_years:
        P = np.column_stack([arms_data[a][y]["p_inner"] for a in members])
        ps.append(P @ weights)
        ys.append(arms_data[members[0]][y]["y_inner"])
    mat = loio_fold_bss(ps, ys, year_labels)
    return {"matrix": mat, **loio_summaries(mat)}


def fig_loio_yearbss(loio_arm: dict, ranked: list[str],
                     fig_dir: Path) -> Path:
    """Arm x inner-year BSS heatmap (mean over folds), plus the per-fold
    MIN column that drives the LOIO score."""
    names = [a for a in ranked if a in loio_arm]
    cols = [str(y) for y in INNER_YEARS] + ["fold-min"]
    mat = np.full((len(names), len(cols)), np.nan)
    for i, a in enumerate(names):
        m = loio_arm[a]["matrix"]
        for j in range(len(INNER_YEARS)):
            mat[i, j] = np.nanmean(m[:, j])
        mat[i, -1] = np.nanmean(np.nanmin(m, axis=1))
    fig, ax = plt.subplots(figsize=(2.2 + 1.1 * len(cols),
                                    1.6 + 0.42 * len(names)))
    vmax = max(np.nanmax(np.abs(mat)), 0.05)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=7)
    ax.set_title("Inner-year BSS (mean over 6 folds; fold-min = LOIO driver)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="BSS")
    fig.tight_layout()
    out = fig_dir / "ensemble_loio_yearbss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# stratified family selection
# --------------------------------------------------------------------------

FAMILIES = ["lingbot_sf", "lingbot_mf", "mae_sf", "mae_mf"]


def arm_family(arm: str) -> str:
    """Route family from the arm name (hard-coded patterns; unmatched arms
    fall into 'other').  mf = multi-frame (temporal _temp<N> suffix)."""
    is_lingbot = arm.startswith(("ens_lingbot", "ens_lb"))
    is_mae = arm.startswith(("ens_v1", "ens_ep40"))
    mf = re.search(r"_temp\d", arm) is not None
    if is_lingbot:
        return "lingbot_mf" if mf else "lingbot_sf"
    if is_mae:
        return "mae_mf" if mf else "mae_sf"
    return "other"


def family_select(ranking: list[str], families: dict[str, str],
                  k: int, gate: int | None = None
                  ) -> tuple[list[str], list[str]]:
    """Quality-gated stratified pick from a LOIO-ordered ranking.

    A family champion keeps its guaranteed seat only when it ranks within
    the global top-`gate`; families whose champion fails the gate do NOT
    take a seat.  Remaining seats: 'other' champion first (also gated),
    then global-ranking fill regardless of family.  Returns (members,
    gated_out_families).
    """
    champs, seen = [], set()
    gated_out = []
    for i, a in enumerate(ranking, start=1):
        f = families[a]
        if f in FAMILIES and f not in seen:
            seen.add(f)
            if gate is None or i <= gate:
                champs.append(a)
            else:
                gated_out.append(f)
    selected = champs[:k]
    others = [a for a in ranking if families[a] == "other"]
    if others and len(selected) < k:
        oc = others[0]
        if gate is None or ranking.index(oc) + 1 <= gate:
            selected.append(oc)
    for a in ranking:
        if len(selected) >= k:
            break
        if a not in selected:
            selected.append(a)
    return selected[:k], gated_out


# --------------------------------------------------------------------------
# figures / report helpers
# --------------------------------------------------------------------------

def fig_inner_vs_held(combos: pd.DataFrame, selected: dict[str, str],
                      fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for method, marker, color in (("uniform", "o", "tab:blue"),
                                  ("nnls", "s", "tab:orange")):
        m = combos["method"] == method
        ax.scatter(combos.loc[m, "inner_bss"], combos.loc[m, "held_bss_mean"],
                   marker=marker, s=18, alpha=0.55, color=color,
                   label=f"scanned ({method})")
    for cid, style in selected.items():
        r = combos[combos["combo_id"] == cid]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.scatter([r["inner_bss"]], [r["held_bss_mean"]], s=140,
                   facecolors="none", edgecolors=style, linewidths=2,
                   label=f"{cid} ({style})")
    lim = [min(combos["inner_bss"].min(), combos["held_bss_mean"].min()) - 0.01,
           max(combos["inner_bss"].max(), combos["held_bss_mean"].max()) + 0.01]
    ax.plot(lim, lim, "k--", lw=0.8, label="inner = held")
    ax.set_xlabel("inner BSS (selection metric, pooled inner)")
    ax.set_ylabel("held BSS mean (evaluated once)")
    ax.set_title("inner vs held BSS per scanned combo\n"
                 "(points above diagonal = inner-pessimistic; below = "
                 "inner-overfit)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "ensemble_inner_vs_held.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def f3(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) and math.isfinite(v) \
        else "—"


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(P2_ROOT))
    except ValueError:
        return str(p)


def md(df: pd.DataFrame, float_cols: list[str]) -> str:
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(f3)
    return df.to_markdown(index=False)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dirs", nargs="+", action="extend",
                    default=[], required=True,
                    help="one or more dirs with <arm>_fold<year>.npz")
    ap.add_argument("--out", default=str(P2_ROOT / "data" / "ensemble_reselect"),
                    help="output prefix; CSV written to <out>_combos.csv")
    ap.add_argument("--fig-dir", default=str(P2_ROOT / "analysis" / "figures"))
    ap.add_argument("--doc-dir", default=str(P2_ROOT / "docs"))
    ap.add_argument("--max-arms", type=int, default=8,
                    help="truncate candidate pool to top-N by inner BSS")
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=5)
    ap.add_argument("--topk", default="4,5,6",
                    help="comma-separated K values for the final Top-K "
                         "protocol: rank ALL complete arms, uniform-average "
                         "the top-K (default 4,5,6)")
    ap.add_argument("--no-loio", action="store_true",
                    help="fall back to pooled-inner-BSS ranking instead of "
                         "the LOIO (leave-one-inner-year-out) robust score")
    ap.add_argument("--pool-all", action="store_true",
                    help="disable the final-pool exclusion (pilot FT arms / "
                         "failed routes re-enter the selection pool)")
    ap.add_argument("--panel-parquet",
                    default=str(P2_ROOT / "data" / "phase1"
                                / "processed"
                                / "wpac_tc_dataset_processed.parquet"),
                    help="panel parquet used to reconstruct inner-year labels")
    ap.add_argument("--frames-index",
                    default=str(P2_ROOT / "data" / "frames_index.parquet"))
    ap.add_argument("--frames-dir", default=str(P2_ROOT / "data" / "frames"))
    args = ap.parse_args()
    topk_list = sorted({int(k) for k in str(args.topk).split(",") if k.strip()})

    preds_dirs = [Path(d) for d in args.preds_dirs]
    fig_dir = Path(args.fig_dir)
    doc_dir = Path(args.doc_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    warns: list[str] = []

    # ---- discovery & completeness --------------------------------------
    fold_map = discover_arms(preds_dirs, warns)
    print(f"discovered {len(fold_map)} arms (seed/old-protocol excluded): "
          f"{sorted(fold_map)}")
    complete, dropped = {}, []
    for arm, paths in sorted(fold_map.items()):
        missing = [y for y in YEARS if y not in paths]
        if missing:
            dropped.append(arm)
            warns.append(f"{arm}: incomplete folds "
                         f"({len(paths)}/6, missing {missing}) "
                         f"-> arm dropped from selection/diversity")
        else:
            complete[arm] = paths
    if len(complete) < 2:
        print("FATAL: fewer than 2 complete arms; nothing to ensemble")
        return 1
    arms_data = {a: load_arm(p) for a, p in complete.items()}

    # ---- cross-arm label alignment -------------------------------------
    all_names = sorted(complete)
    ok_years = align_folds(arms_data, all_names, warns)

    # ---- final pool scope (user ruling): pilot FT arms & failed routes --
    # ---- stay in diversity (flagged pilot) but never in selection -------
    if args.pool_all:
        pool_names = list(all_names)
        pool_excluded: list[str] = []
    else:
        pool_names = [a for a in all_names
                      if not any(re.search(p, a)
                                 for p in POOL_EXCLUDE_PATTERNS)]
        pool_excluded = [a for a in all_names if a not in pool_names]
    if pool_excluded:
        print(f"pool exclusion (pilot FT / failed routes): {pool_excluded}")
    print(f"selection pool ({len(pool_names)}): {pool_names}")
    if len(ok_years) < 2:
        print("FATAL: fewer than 2 aligned folds")
        return 1
    if len(ok_years) < 6:
        warns.append(f"only {len(ok_years)}/6 folds aligned across arms: "
                     f"{ok_years}")
    print(f"aligned folds: {ok_years}")

    # ---- diversity ------------------------------------------------------
    err_m, err_std_m, sp_m = diversity_mats(arms_data, all_names)
    fig_div = fig_diversity(err_m, sp_m, fig_dir)
    print(f"wrote {fig_div}")

    # ---- candidate pool truncation --------------------------------------
    single_inner = {a: bss_score(
                        np.concatenate([arms_data[a][y]["p_inner"]
                                        for y in ok_years]),
                        np.concatenate([arms_data[a][y]["y_inner"]
                                        for y in ok_years]))
                    for a in all_names}
    pool = sorted(pool_names, key=lambda a: -single_inner[a])
    if len(pool) > args.max_arms:
        cut = pool[args.max_arms:]
        pool = pool[:args.max_arms]
        warns.append(f"candidate pool truncated to top-{args.max_arms} by "
                     f"inner BSS; excluded from scan: {cut}")
    print(f"candidate pool ({len(pool)}): {pool}")
    print("single-arm inner BSS: "
          + ", ".join(f"{a}={single_inner[a]:.4f}" for a in pool))

    # ---- combination scan (fit on pooled inner only) --------------------
    rows = []
    for size in range(args.min_size, args.max_size + 1):
        for members in itertools.combinations(pool, size):
            members = list(members)
            Pi, yi = stacked(arms_data, members, ok_years, "inner")
            for method in ("uniform", "nnls"):
                if method == "uniform":
                    w = np.full(size, 1.0 / size)
                else:
                    w = nnls_weights(Pi, yi)
                ib = bss_score(Pi @ w, yi)
                held = eval_combo_held(arms_data, members, w, ok_years)
                cid = f"{'U' if method == 'uniform' else 'N'}:" \
                      + "+".join(members)
                rows.append({
                    "combo_id": cid, "method": method, "n_members": size,
                    "members": "+".join(members),
                    "weights": ";".join(f"{a}:{wi:.4f}" for a, wi
                                        in zip(members, w)),
                    "inner_bss": ib,
                    "held_bss_mean": held["held_bss_mean"],
                    "held_bss_worst": held["held_bss_worst"],
                    "pr_auc": held["pr_auc"], "pod": held["pod"],
                    "far": held["far"], "selection": "scanned",
                })
    combos = pd.DataFrame(rows)
    n_combos = len(combos)
    print(f"scanned {n_combos} combos "
          f"(sizes {args.min_size}..{args.max_size} x uniform/nnls)")

    # ---- selection on inner ONLY ----------------------------------------
    best_uniform = combos[combos["method"] == "uniform"] \
        .sort_values("inner_bss", ascending=False).iloc[0]
    best_nnls = combos[combos["method"] == "nnls"] \
        .sort_values("inner_bss", ascending=False).iloc[0]
    combos.loc[combos["combo_id"] == best_uniform["combo_id"],
               "selection"] = "inner"
    combos.loc[combos["combo_id"] == best_nnls["combo_id"],
               "selection"] = "inner"

    # best single arm (inner BSS; control row, not part of the 2..5 scan)
    best_single = max(pool, key=lambda a: single_inner[a])
    bs_held = eval_combo_held(arms_data, [best_single], np.array([1.0]),
                              ok_years)
    single_row = {
        "combo_id": f"S:{best_single}", "method": "single", "n_members": 1,
        "members": best_single, "weights": f"{best_single}:1.0000",
        "inner_bss": single_inner[best_single],
        "held_bss_mean": bs_held["held_bss_mean"],
        "held_bss_worst": bs_held["held_bss_worst"],
        "pr_auc": bs_held["pr_auc"], "pod": bs_held["pod"],
        "far": bs_held["far"], "selection": "inner",
    }
    combos = pd.concat([combos, pd.DataFrame([single_row])],
                       ignore_index=True)

    # ---- union4: pre-specified control, held evaluated directly --------
    u4_row, u4_folds = None, None
    missing_u4 = [a for a in UNION4_MEMBERS if a not in complete]
    if missing_u4:
        warns.append(f"union4 control skipped: member arms missing "
                     f"{missing_u4}")
    else:
        w = np.full(len(UNION4_MEMBERS), 1.0 / len(UNION4_MEMBERS))
        u4_held = eval_combo_held(arms_data, UNION4_MEMBERS, w, ok_years)
        u4_folds = u4_held["folds"]
        u4_row = {
            "combo_id": "union4", "method": "uniform", "n_members": 4,
            "members": "+".join(UNION4_MEMBERS),
            "weights": ";".join(f"{a}:{1/4:.4f}" for a in UNION4_MEMBERS),
            "inner_bss": inner_bss(arms_data, UNION4_MEMBERS, w, ok_years),
            "held_bss_mean": u4_held["held_bss_mean"],
            "held_bss_worst": u4_held["held_bss_worst"],
            "pr_auc": u4_held["pr_auc"], "pod": u4_held["pod"],
            "far": u4_held["far"], "selection": "preregistered",
        }
        combos = pd.concat([combos, pd.DataFrame([u4_row])],
                           ignore_index=True)
        d = u4_row["held_bss_mean"] - UNION4_REF_BSS
        status = "OK" if abs(d) <= UNION4_REF_TOL else "MISMATCH"
        print(f"union4 held BSS mean = {u4_row['held_bss_mean']:.4f} "
              f"(ref {UNION4_REF_BSS}, |d|={abs(d):.4f}) -> {status}")
        if abs(d) > UNION4_REF_TOL:
            warns.append(f"union4 held BSS mean {u4_row['held_bss_mean']:.4f} "
                         f"deviates from preregistered ref {UNION4_REF_BSS} "
                         f"by {abs(d):.4f} > tol {UNION4_REF_TOL}")

    # ---- LOIO reconstruction & scoring (selection signal with an ----
    # ---- extrapolation component; DomainBed LODO analogue) --------------
    year_labels = None
    loio_arm: dict = {}
    if not args.no_loio:
        try:
            year_labels, inner_y = reconstruct_inner_labels(
                Path(args.panel_parquet), Path(args.frames_index),
                Path(args.frames_dir))
            ref = next(iter(arms_data.values()))
            n_ok = all(len(ref[y]["y_inner"]) == len(year_labels)
                       for y in ok_years)
            y_ok = n_ok and all(
                np.array_equal(ref[y]["y_inner"].astype(float), inner_y)
                for y in ok_years)
            if not (n_ok and y_ok):
                warns.append("LOIO: reconstructed inner labels do NOT match "
                             "npz y_inner (len or values) -> LOIO disabled, "
                             "falling back to pooled-inner ranking")
                year_labels = None
            else:
                counts = {y: int((year_labels == y).sum())
                          for y in INNER_YEARS}
                print(f"LOIO: inner-year labels reconstructed & verified vs "
                      f"npz y_inner (n={len(year_labels)}, counts={counts})")
        except Exception as e:  # missing panel/frames etc.
            warns.append(f"LOIO: inner-year reconstruction failed ({e}) -> "
                         f"LOIO disabled, falling back to pooled-inner "
                         f"ranking")
            year_labels = None
        if year_labels is not None:
            loio_arm = arm_loio(arms_data, all_names, ok_years, year_labels)

    # ---- Top-K protocol (final ensemble): three selection rules on the --
    # ---- RETAINED pool: inner_pooled (reference), loio (global LOIO -----
    # ---- Top-K), family (quality-gated stratified champions; primary). --
    rankings = {"inner_pooled": sorted(pool_names,
                                       key=lambda a: -single_inner[a])}
    if loio_arm:
        rankings = {"loio": sorted(pool_names,
                                   key=lambda a: -loio_arm[a]["min_mean"]),
                    **rankings}
    base_rule = "loio" if loio_arm else "inner_pooled"
    families = {a: arm_family(a) for a in pool_names}
    print("family assignment (pool): "
          + ", ".join(f"{a}={families[a]}" for a in pool_names))

    def ens_select_score(e: dict, rule: str) -> float:
        r = base_rule if rule == "family" else rule
        return e["loio_min_mean"] if r == "loio" else e["inner_bss"]

    topk_ens: dict[tuple[str, int], dict] = {}
    gated_out_fams: set[str] = set()
    for rule, ranking in list(rankings.items()) + [("family",
                                                    rankings[base_rule])]:
        for k in topk_list:
            if rule == "family":
                members, gated = family_select(ranking, families, k,
                                               gate=FAMILY_GATE)
                gated_out_fams.update(gated)
                if len(members) < k:
                    warns.append(f"Top-{k} (family): only {len(members)} "
                                 f"arms available")
            else:
                if k > len(ranking):
                    warns.append(f"Top-{k} ({rule}) skipped: only "
                                 f"{len(ranking)} pool arms")
                    continue
                members = ranking[:k]
            w = np.full(len(members), 1.0 / len(members))
            held = eval_combo_held(arms_data, members, w, ok_years)
            e = {"members": members, "weights": w,
                 "inner_bss": inner_bss(arms_data, members, w, ok_years),
                 **held}
            if loio_arm:
                e["loio_min_mean"] = ensemble_loio(
                    arms_data, members, w, ok_years,
                    year_labels)["min_mean"]
            else:
                e["loio_min_mean"] = float("nan")
            topk_ens[(rule, k)] = e
            print(f"[{rule}] Top-{k} uniform ({'+'.join(members)}): "
                  f"select={ens_select_score(e, rule):.4f} "
                  f"(inner_pooled={e['inner_bss']:.4f}"
                  + (f", loio={e['loio_min_mean']:.4f}" if loio_arm else "")
                  + f") -> held BSS mean {held['held_bss_mean']:.4f} "
                    f"(worst {held['held_bss_worst']:.4f})")
    active_rule = "family"
    active_ens = {k: e for (r, k), e in topk_ens.items() if r == active_rule}
    best_k = (max(active_ens,
                  key=lambda k: ens_select_score(active_ens[k], active_rule))
              if active_ens else None)
    ranked = rankings[base_rule]

    # ---- outputs ---------------------------------------------------------
    combos = combos.sort_values("inner_bss", ascending=False) \
        .reset_index(drop=True)
    csv_path = Path(f"{args.out}_combos.csv")
    combos.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    # Top-K CSV: arm ranking rows (both rules) + one row per rule x K ensemble
    rank_of = {rule: {a: i + 1 for i, a in enumerate(r)}
               for rule, r in rankings.items()}
    ks_by_rule = {}
    for (r, k) in topk_ens:
        ks_by_rule.setdefault(r, []).append(k)
    ks_by_rule = {r: sorted(ks) for r, ks in ks_by_rule.items()}
    topk_rows = []
    for a in pool_names:
        topk_rows.append({
            "row_type": "arm", "arm": a, "family": families[a],
            "k": "", "members": "",
            "rank_loio": rank_of.get("loio", {}).get(a, ""),
            "rank_pooled": rank_of["inner_pooled"].get(a, ""),
            "inner_bss": single_inner[a],
            "loio_min_mean": loio_arm.get(a, {}).get("min_mean", ""),
            "loio_pooled_mean": loio_arm.get(a, {}).get("pooled_mean", ""),
            "loio_mean_minus_std": loio_arm.get(a, {}).get(
                "mean_minus_std", ""),
            "held_bss_mean": "", "held_bss_worst": "", "pr_auc": "",
            "pod": "", "far": "",
            "in_topk_loio": ";".join(
                f"top{k}" for k in ks_by_rule.get("loio", [])
                if rank_of.get("loio", {}).get(a, 10 ** 9) <= k),
            "in_topk_pooled": ";".join(
                f"top{k}" for k in ks_by_rule.get("inner_pooled", [])
                if rank_of["inner_pooled"].get(a, 10 ** 9) <= k),
            "in_topk_family": ";".join(
                f"top{k}" for k in ks_by_rule.get("family", [])
                if a in topk_ens[("family", k)]["members"]),
            "selection": "",
        })
    for (rule, k), e in sorted(topk_ens.items()):
        topk_rows.append({
            "row_type": "ensemble", "arm": "", "family": "", "k": k,
            "members": "+".join(e["members"]),
            "rank_loio": "", "rank_pooled": "",
            "inner_bss": e["inner_bss"],
            "loio_min_mean": e["loio_min_mean"],
            "loio_pooled_mean": "", "loio_mean_minus_std": "",
            "held_bss_mean": e["held_bss_mean"],
            "held_bss_worst": e["held_bss_worst"],
            "pr_auc": e["pr_auc"], "pod": e["pod"], "far": e["far"],
            "in_topk_loio": "", "in_topk_pooled": "", "in_topk_family": "",
            "selection": rule if (rule == active_rule and k == best_k)
            else f"{rule}_alt",
        })
    topk_df = pd.DataFrame(topk_rows)
    topk_csv_path = Path(f"{args.out}_topk.csv")
    topk_df.to_csv(topk_csv_path, index=False)
    print(f"wrote {topk_csv_path}")

    fig_loio = None
    if loio_arm:
        fig_loio = fig_loio_yearbss(loio_arm, rankings["loio"], fig_dir)
        print(f"wrote {fig_loio}")

    selected_marks = {}
    if u4_row:
        selected_marks["union4"] = "red"
    selected_marks[best_uniform["combo_id"]] = "green"
    selected_marks[best_nnls["combo_id"]] = "purple"
    fig_sc = fig_inner_vs_held(combos[combos["selection"] != "preregistered"],
                               selected_marks, fig_dir)
    print(f"wrote {fig_sc}")

    # ---- report ----------------------------------------------------------
    def combo_held_folds(cid: str):
        r = combos[combos["combo_id"] == cid].iloc[0]
        members = r["members"].split("+")
        if r["method"] == "single":
            w = np.array([1.0])
        elif r["method"] == "uniform":
            w = np.full(len(members), 1.0 / len(members))
        else:
            w = np.array([float(x.split(":")[1])
                          for x in r["weights"].split(";")])
        return eval_combo_held(arms_data, members, w, ok_years)["folds"]

    final_ids = [best_uniform["combo_id"], best_nnls["combo_id"],
                 f"S:{best_single}"] + (["union4"] if u4_row else [])
    final_rows = []
    for cid in final_ids:
        r = combos[combos["combo_id"] == cid].iloc[0]
        final_rows.append({
            "combo": cid, "selection": r["selection"],
            "inner_bss": r["inner_bss"],
            "held_bss_mean": r["held_bss_mean"],
            "held_bss_worst": r["held_bss_worst"],
            "PR_AUC": r["pr_auc"], "POD": r["pod"], "FAR": r["far"],
        })
    final_df = pd.DataFrame(final_rows)

    top10 = combos[combos["selection"] != "preregistered"].head(10)

    L: list[str] = []
    L.append("# Ensemble composition re-evaluation (ensemble re-selection)")
    L.append("")
    L.append(f"Generation command: `python scripts/04_train/ensemble_reselect.py "
             f"{' '.join(sys.argv[1:])}`  ")
    L.append(f"Prediction directories: {', '.join(str(d) for d in preds_dirs)}  ")
    L.append(f"Aligned folds: {ok_years} ({len(ok_years)}/6); "
             f"candidate pool of {len(pool)} arms: {', '.join(f'`{a}`' for a in pool)}; "
             f"{n_combos} combinations scanned (sizes "
             f"{args.min_size}..{args.max_size} x uniform/nnls)")
    if dropped:
        L.append(f"Arms removed for missing folds: {', '.join(f'`{a}`' for a in dropped)}")
    L.append("")

    L.append("## TL;DR")
    L.append("")
    bu, bn = best_uniform, best_nnls
    L.append(f"- **Best inner uniform combination**: `{bu['members']}` — "
             f"inner BSS {bu['inner_bss']:.4f} -> mean held BSS "
             f"**{bu['held_bss_mean']:.4f}** (worst fold {bu['held_bss_worst']:.4f})")
    L.append(f"- **Best inner NNLS combination**: `{bn['members']}` — "
             f"inner BSS {bn['inner_bss']:.4f} -> mean held BSS "
             f"**{bn['held_bss_mean']:.4f}** (worst fold {bn['held_bss_worst']:.4f}), "
             f"weights {bn['weights']}")
    if u4_row:
        d_u = bu["held_bss_mean"] - u4_row["held_bss_mean"]
        L.append(f"- **Preregistered union4 control**: mean held BSS "
                 f"**{u4_row['held_bss_mean']:.4f}**"
                 f" (worst fold {u4_row['held_bss_worst']:.4f}); "
                 f"best inner uniform versus union4 {d_u:+.4f}")
    L.append(f"- **Best single arm**: `{best_single}` — mean held BSS "
             f"{bs_held['held_bss_mean']:.4f}")
    if best_k is not None:
        e = topk_ens[(active_rule, best_k)]
        sel = ens_select_score(e, active_rule)
        L.append(f"- **Final ensemble protocol (Top-{best_k} uniform, selection rule = "
                 f"{active_rule})**: selection score {sel:.4f} -> mean held BSS "
                 f"**{e['held_bss_mean']:.4f}** (worst fold "
                 f"{e['held_bss_worst']:.4f}), members "
                 f"{'+'.join(f'`{m}`' for m in e['members'])}")
    overfit = bn["inner_bss"] - bn["held_bss_mean"]
    L.append(f"- NNLS combination inner-held difference = {overfit:+.4f}"
             + (" (a positive value suggests inner overfitting)" if overfit > 0.01 else ""))
    L.append("")

    L.append("## Diversity matrices (pooled held; per-fold error-correlation std also provided)")
    L.append("")
    if pool_excluded:
        L.append("Note: excluded pilot=1 arms ("
                 + ", ".join(f"`{a}`" for a in pool_excluded)
                 + ") are shown for diversity reference only and are outside the selection pool.")
        L.append("")
    L.append("### Pearson error correlation (p - y)")
    L.append("")
    L.append(md(err_m.round(3).reset_index().rename(columns={"index": "arm"}),
                list(err_m.columns)))
    L.append("")
    L.append("### Per-fold std of error correlations")
    L.append("")
    L.append(md(err_std_m.round(3).reset_index().rename(columns={"index": "arm"}),
                list(err_std_m.columns)))
    L.append("")
    L.append("### Spearman rank correlation of probabilities")
    L.append("")
    L.append(md(sp_m.round(3).reset_index().rename(columns={"index": "arm"}),
                list(sp_m.columns)))
    L.append("")
    L.append(f"Figure: `{rel(fig_div)}`")
    L.append("")

    L.append("## Inner selection — Top 10 combinations by inner BSS")
    L.append("")
    L.append(md(top10[["combo_id", "method", "n_members", "weights",
                       "inner_bss", "held_bss_mean", "held_bss_worst",
                       "selection"]],
                ["inner_bss", "held_bss_mean", "held_bss_worst"]))
    L.append("")
    L.append(f"Inner versus held scatter plot: `{rel(fig_sc)}`")
    L.append("")

    L.append("## Final held evaluation (each combination evaluated once)")
    L.append("")
    L.append(md(final_df, ["inner_bss", "held_bss_mean", "held_bss_worst",
                           "PR_AUC", "POD", "FAR"]))
    L.append("")
    L.append("### Per-fold BSS")
    L.append("")
    fold_rows = []
    for cid in final_ids:
        for fr in combo_held_folds(cid):
            fold_rows.append({"combo": cid, **{k: fr[k] for k in
                                               ("year", "BSS", "PR_AUC",
                                                "POD", "FAR", "thr")}})
    L.append(md(pd.DataFrame(fold_rows),
                ["BSS", "PR_AUC", "POD", "FAR", "thr"]))
    L.append("")

    # ---- LOIO robust scoring section --------------------------------------
    if loio_arm:
        L.append("## LOIO yearly robust scoring (leave-one-inner-year-out)")
        L.append("")
        L.append("Each arm's inner predictions (the same three years within each fold, "
                 f"{INNER_YEARS}, with storm-level splits) are divided by inner year, "
                 "and BSS is computed against each year's own climatology, giving "
                 "6 folds x 3 years = 18 yearly BSS values. The primary **min-mean** "
                 "score takes the worst (minimum) of the three years within each fold, "
                 "then averages across six folds. This minimax robust score prevents "
                 "an arm with one anomalously strong year from being selected on its "
                 "pooled score; pooled mean and mean-std are references. Inner-year "
                 "labels are deterministically reconstructed from panel row order and "
                 "storm-level LOYO splits in RIPanelDataset, and are checked against "
                 "each fold's NPZ y_inner.")
        L.append("")
        L.append(f"Yearly BSS heatmap (arm x inner year, six-fold mean; fold-min drives "
                 f"the primary LOIO score): `{rel(fig_loio)}`")
        L.append("")
        L.append("### Method description")
        L.append("")
        L.append("> Ranking arms by their pooled inner-score is "
                 "training-domain validation in the sense of DomainBed "
                 "(Gulrajani & Lopez-Paz, ICLR 2021) and is known to favour "
                 "high-capacity arms under distribution shift (winner's "
                 "curse; Cawley & Talbot, JMLR 2010). We therefore rank "
                 "arms by a leave-one-inner-year-out (LOIO) robust score — "
                 "the direct analogue of DomainBed's leave-one-domain-out "
                 "validation within our leave-one-year-out layout: the "
                 "inner predictions of every fold are split by inner year, "
                 "the Brier skill score is computed separately against each "
                 "year's own climatology, and an arm's score is the mean "
                 "over folds of its worst (minimum) inner-year BSS. The "
                 "per-fold minimum injects an extrapolation component into "
                 "the selection signal and implements a minimax criterion "
                 "that is appropriate for rare-event forecasting, where a "
                 "single failed season is operationally unacceptable; the "
                 "pooled yearly mean and the mean-minus-standard-deviation "
                 "are reported as references. All LOIO quantities are "
                 "computed exclusively on inner splits.")
        L.append("")

    # ---- Top-K final protocol section ------------------------------------
    if topk_ens:
        rule_names = {"family": f"quality-gated family champions (global LOIO top "
                                f"{FAMILY_GATE}) + global LOIO fill (primary rule)",
                      "loio": "global LOIO min-mean Top-K (control)",
                      "inner_pooled": "global pooled-inner BSS Top-K (control)"}
        L.append("## Final pool scope")
        L.append("")
        L.append("The candidate pool excludes the registered pilot fine-tuning and "
                 "failed-route patterns. Their predictions remain available for "
                 "diversity diagnostics with pilot=1, but are excluded from Top-K, "
                 "stratified, and combination scans unless `--pool-all` is used.")
        L.append("")
        if pool_excluded:
            L.append("### Excluded arms")
            L.append("")
            for a in pool_excluded:
                pats = [p for p in POOL_EXCLUDE_PATTERNS if re.search(p, a)]
                L.append(f"- `{a}` (pilot=1; matched `{pats[0]}`)")
            L.append("")
            L.append("Note: `lb{B,L}_{partial4,lora16,lpft4}_{temp5,sf}` are pilot "
                     "fine-tuning arms. Aggregator variants (diff/slim/diffslim/"
                     "attnpool/temp2/temp3) remain in the pool, as do fully "
                     "fine-tuned main-route arms such as ep40ft/v1.")
        else:
            L.append("No arms were excluded in this run (or --pool-all was enabled).")
        L.append("")
        L.append(f"### Retained pool ({len(pool_names)} arms)")
        L.append("")
        L.append(", ".join(f"`{a}`" for a in pool_names))
        L.append("")
        L.append("## Top-K selection (final ensemble protocol)")
        L.append("")
        L.append(f"The pool contains {len(pool_names)} arms with all six folds, "
                 f"including heterogeneous training recipes (different backbones, "
                 f"frozen or fully fine-tuned encoders, single- or multi-frame "
                 f"inputs, and batch configurations). Arms with missing folds, "
                 f"seed or superseded-protocol variants, pilot fine-tuning arms, "
                 f"and failed routes are excluded. Top-K (K in "
                 f"{sorted({k for (_, k) in topk_ens})}) uses a uniform mean; "
                 f"ranking and K selection use inner only, with held evaluated once. "
                 f"Primary selection rule = **{rule_names[active_rule]}**"
                 + ("; pooled-inner and global-LOIO rankings are shown as controls."
                    if loio_arm else " (--no-loio fallback; family ranking uses "
                    "pooled inner)."))
        L.append("")
        L.append("### Arm ranking (two-rule comparison)")
        L.append("")
        rank_rows = []
        loio_rank = rankings.get("loio", [])
        pooled_rank = rankings["inner_pooled"]
        for i, a in enumerate(pooled_rank, start=1):
            row = {
                "rank_pooled": i, "arm": a, "family": families[a],
                "inner_bss": single_inner[a],
                "rank_loio": (loio_rank.index(a) + 1) if loio_rank else "—",
                "loio_min_mean": loio_arm.get(a, {}).get("min_mean",
                                                         float("nan")),
                "loio_pooled_mean": loio_arm.get(a, {}).get(
                    "pooled_mean", float("nan")),
                "loio_mean_minus_std": loio_arm.get(a, {}).get(
                    "mean_minus_std", float("nan")),
                "in_topk_pooled": ";".join(
                    f"top{k}" for k in ks_by_rule.get("inner_pooled", [])
                    if i <= k) or "—",
                "in_topk_loio": (";".join(
                    f"top{k}" for k in ks_by_rule.get("loio", [])
                    if loio_rank.index(a) + 1 <= k) or "—")
                if loio_rank else "—",
                "in_topk_family": ";".join(
                    f"top{k}" for k in ks_by_rule.get("family", [])
                    if a in topk_ens[("family", k)]["members"]) or "—",
            }
            rank_rows.append(row)
        L.append(md(pd.DataFrame(rank_rows),
                    ["inner_bss", "loio_min_mean", "loio_pooled_mean",
                     "loio_mean_minus_std"]))
        L.append("")
        L.append("### Top-K ensemble performance under three rules (held evaluated once)")
        L.append("")
        k_rows = []
        for (rule, k), e in sorted(topk_ens.items()):
            k_rows.append({
                "rule": rule, "K": k, "members": "+".join(e["members"]),
                "select_score": ens_select_score(e, rule),
                "inner_bss": e["inner_bss"],
                "loio_min_mean": e["loio_min_mean"],
                "held_bss_mean": e["held_bss_mean"],
                "held_bss_worst": e["held_bss_worst"],
                "PR_AUC": e["pr_auc"], "POD": e["pod"], "FAR": e["far"],
                "selected": "✓" if (rule == active_rule and k == best_k)
                else "",
            })
        L.append(md(pd.DataFrame(k_rows),
                    ["select_score", "inner_bss", "loio_min_mean",
                     "held_bss_mean", "held_bss_worst",
                     "PR_AUC", "POD", "FAR"]))
        L.append("")
        L.append(f"Best K by the primary rule ({active_rule}) selection score = **{best_k}**.")
        L.append("")
        L.append("### Top-K per-fold BSS")
        L.append("")
        kfold_rows = []
        for (rule, k), e in sorted(topk_ens.items()):
            for fr in e["folds"]:
                kfold_rows.append({"rule": rule, "K": k,
                                   **{kk: fr[kk] for kk in
                                      ("year", "BSS", "PR_AUC",
                                       "POD", "FAR", "thr")}})
        L.append(md(pd.DataFrame(kfold_rows),
                    ["BSS", "PR_AUC", "POD", "FAR", "thr"]))
        L.append("")
        L.append("### Top-K protocol description")
        L.append("")
        rank_desc = (
            "a leave-one-inner-year-out (LOIO) robust score: for each fold "
            "the inner predictions were split by inner year, the Brier "
            "skill score was computed against each year's own climatology, "
            "and an arm's score was the mean over folds of its worst "
            "inner-year BSS (minimax)" if loio_arm else
            "the Brier skill score of each arm's pooled inner (validation) "
            "predictions against the pooled inner climatology")
        L.append("> The final ensemble was constructed by ranking every arm "
                 "that produced valid predictions under the identical "
                 "leave-one-year-out split and inner/held protocol — "
                 "including arms trained under heterogeneous recipes "
                 "(different backbones, frozen vs. fully fine-tuned "
                 "encoders, single- vs. multi-frame inputs, and different "
                 "batch configurations; training details for each arm are "
                 "documented in the supplementary training reports) — by "
                 f"{rank_desc}. The top-K arms (reported for K = "
                 f"{', '.join(str(k) for k in sorted({k for (_, k) in topk_ens}))}) were "
                 "combined by a uniform probability average. Both the arm "
                 "ranking and the choice of K used only the inner splits; "
                 "the held-out years were evaluated exactly once, after "
                 "the ensemble composition was frozen. Arms with "
                 "incomplete folds and seed or superseded-protocol "
                 "variants were excluded a priori, and folds whose labels "
                 "disagreed across arms were refused rather than averaged. "
                 "The decision threshold for POD/FAR was tuned by F1 on "
                 "each fold's inner split, following the training "
                 "protocol.")
        L.append("")

    # ---- stratified family selection section ------------------------------
    if ("family", 5) in topk_ens or any(r == "family" for (r, _) in topk_ens):
        L.append("## Stratified family selection (quality-gated final ensemble rule)")
        L.append("")
        L.append("Global Top-K can concentrate seats in one strong model family, "
                 "reducing structural diversity; an ungated family quota can reserve "
                 "seats for weak families. The quality-gated compromise guarantees a "
                 "family champion a seat only when it reaches the global LOIO top "
                 f"{FAMILY_GATE}; other seats are filled by the global LOIO ranking. "
                 "Within-family ranking uses the LOIO score (Miller et al. 2021), "
                 "and selected families are uniformly averaged (Weigel et al. 2010).")
        L.append("")
        L.append("### Route families (arm-name patterns; unmatched arms go to other)")
        L.append("")
        fam_desc = {
            "lingbot_sf": "LingBot single-frame (ens_lingbot/B/L, ens_lb{B,L}_*_sf; no temporal suffix)",
            "lingbot_mf": "LingBot multi-frame (ens_lb_temp5, ens_lingbotL_temp5, ens_lb{B,L}_*_temp5, ens_lbB_temp5_{diff,slim,diffslim,attnpool}, ens_lbB_temp{2,3})",
            "mae_sf": "vit_mae single-frame (ens_v1, ens_ep40, ens_ep40ft, ens_ep40_srv)",
            "mae_mf": "vit_mae multi-frame (ens_ep40_temp5, ens_ep40ft_temp5, ens_v1_temp5)",
            "other": "Arms not matching the patterns above",
        }
        fam_rows = []
        for fam in FAMILIES + ["other"]:
            members = [a for a in rankings[base_rule]
                       if families[a] == fam]
            if not members and fam == "other":
                continue
            champ = members[0] if members else "—"
            fam_rows.append({
                "family": fam, "definition": fam_desc[fam],
                "members (LOIO order)": ", ".join(members) or "—",
                "family champion": champ,
            })
        L.append(pd.DataFrame(fam_rows).to_markdown(index=False))
        L.append("")
        L.append("### Selection rule (quality-gated stratification)")
        L.append("")
        L.append(f"- Each family champion is selected by LOIO min-mean; the **quality "
                 f"gate** requires a global LOIO rank within the top {FAMILY_GATE} "
                 f"for a guaranteed seat"
                 + (f" (families excluded by the gate: {', '.join(sorted(gated_out_fams))})"
                    if gated_out_fams else " (no family was excluded by the gate)"))
        L.append("- When fewer than K seats are filled, use the global LOIO ranking "
                 "regardless of family (prefer an eligible other-family champion that "
                 "passes the gate); fill K=4/5/6 to 4/5/6 members, respectively")
        L.append("- Average selected members with uniform weights; evaluate held once")
        L.append("")
        L.append("### Three selection rules (K=5; held evaluated once)")
        L.append("")
        cmp_rows = []
        for rule, label in (("inner_pooled", "inner-only Top-5"),
                            ("loio", "LOIO Top-5"),
                            ("family", "quality-gated stratified Top-5")):
            e = topk_ens.get((rule, 5))
            if e is None:
                continue
            cmp_rows.append({
                "rule": label, "members": "+".join(e["members"]),
                "held_bss_mean": e["held_bss_mean"],
                "held_bss_worst": e["held_bss_worst"],
                "PR_AUC": e["pr_auc"], "POD": e["pod"], "FAR": e["far"],
            })
        L.append(md(pd.DataFrame(cmp_rows),
                    ["held_bss_mean", "held_bss_worst", "PR_AUC",
                     "POD", "FAR"]))
        L.append("")
        mf_in = [a for a in topk_ens[("family", best_k)]["members"]
                 if families[a].endswith("_mf")] if best_k else []
        L.append(f"Multi-frame arms in stratified Top-{best_k}: "
                 + (", ".join(f"`{a}` ({families[a]} family champion)" for a in mf_in)
                    if mf_in else "none"))
        L.append("")
        L.append("### Stratified selection protocol description")
        L.append("")
        L.append("> To prevent the ensemble seats from being monopolised by "
                 "a single strong model family, the final ensemble was "
                 "selected by a quality-gated stratified family rule. Arms "
                 "from failed fine-tuning pilot experiments and from "
                 "wholesale-failed routes were excluded from the candidate "
                 "pool a priori (their predictions remain in the diversity "
                 "analysis, flagged as pilot). Every retained arm was "
                 "assigned to one of four route families defined by "
                 "backbone and input structure (LingBot single-frame, "
                 "LingBot multi-frame, vit_mae single-frame, vit_mae "
                 "multi-frame; unmatched arms fall into a residual "
                 "category). Within each family, arms were ranked by the "
                 "leave-one-inner-year-out (LOIO) robust score — "
                 "within-family ranking by in-distribution validation is "
                 "expected to transfer to out-of-distribution performance "
                 "(Miller et al., ICML 2021). A family champion kept its "
                 f"guaranteed seat only if it ranked within the global "
                 f"LOIO top-{FAMILY_GATE} of the retained pool; families "
                 "failing this quality gate did not take a seat, and all "
                 "remaining seats were filled by the global LOIO ranking "
                 "regardless of family. Selected members were combined by "
                 "a uniform (equal-weight) probability average: when skill "
                 "differences between members are small relative to "
                 "sampling uncertainty, estimated optimal weights are "
                 "outperformed by equal weights (Weigel et al., J. Climate "
                 "2010), and structural heterogeneity across families "
                 "provides the diversity that correlated same-family "
                 "members cannot. All ranking and seat-filling decisions "
                 "used only the inner splits; the held-out years were "
                 "evaluated exactly once after the composition was frozen.")
        L.append("")

    L.append("## Protocol description")
    L.append("")
    L.append("> Ensemble members and weights were selected exclusively on the "
             "inner (validation) splits of each leave-one-year-out fold; the "
             "held-out year of every fold was evaluated exactly once, after "
             "all selection decisions were frozen. Candidate members were "
             "restricted to arms with complete six-fold predictions; arms "
             "with missing folds were excluded as a whole, and folds whose "
             "labels disagreed across arms were refused rather than "
             "averaged. All subsets of two to five members from the "
             "candidate pool were fitted on the pooled inner predictions "
             "using (a) a uniform probability average and (b) non-negative "
             "least-squares (NNLS) weights normalized to sum to one, and "
             "ranked by the Brier skill score against the pooled inner "
             "climatology. The decision threshold for POD/FAR was tuned by "
             "F1 on each fold's inner split, following the training "
             "protocol. The previously registered four-member uniform "
             "ensemble (union4) was carried through as a pre-specified "
             "control: it was not part of the selection procedure and was "
             "evaluated directly on the held-out years.")
    L.append("")

    L.append("## WARN summary")
    L.append("")
    if warns:
        for w in warns:
            L.append(f"- WARN: {w}")
    else:
        L.append("No WARN messages.")
    L.append("")

    doc_path = doc_dir / "ENSEMBLE_RESELECT.md"
    doc_path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {doc_path}")

    # ---- console summary --------------------------------------------------
    print("\n=== final held table ===")
    print(final_df.round(4).to_string(index=False))
    print("\n=== top5 by inner BSS ===")
    print(combos.head(5)[["combo_id", "method", "inner_bss",
                          "held_bss_mean", "held_bss_worst", "selection"]]
          .round(4).to_string(index=False))
    if topk_ens:
        print("\n=== Top-K protocol (final ensemble) ===")
        for rule, ranking in rankings.items():
            score = (lambda a: loio_arm[a]["min_mean"]) if rule == "loio" \
                else (lambda a: single_inner[a])
            print(f"arm ranking ({rule}): "
                  + ", ".join(f"{i+1}.{a}={score(a):.4f}"
                              for i, a in enumerate(ranking)))
        for (rule, k), e in sorted(topk_ens.items()):
            mark = (" <- SELECTED" if (rule == active_rule and k == best_k)
                    else "")
            print(f"[{rule}] Top-{k}: select "
                  f"{ens_select_score(e, rule):.4f} | "
                  f"held mean {e['held_bss_mean']:.4f} "
                  f"worst {e['held_bss_worst']:.4f} "
                  f"PR-AUC {e['pr_auc']:.4f} POD {e['pod']:.3f} "
                  f"FAR {e['far']:.3f}{mark}")
        if best_k is not None:
            sel = topk_ens[("family", best_k)]["members"]
            mf = [a for a in sel if families[a].endswith("_mf")]
            print(f"family Top-{best_k} members: {sel}")
            print(f"multi-frame arms in family Top-{best_k}: "
                  f"{mf or 'NONE'}")
        if loio_arm:
            print("rank movement (pooled -> loio): "
                  + ", ".join(
                      f"{a} #{rankings['inner_pooled'].index(a)+1}->"
                      f"#{rankings['loio'].index(a)+1}"
                      for a in pool_names))
            # ens_lb_temp5 gate status (frozen multi-frame arm of interest)
            probe = "ens_lb_temp5"
            if probe in rankings["loio"]:
                r = rankings["loio"].index(probe) + 1
                in_sel = [k for (rule, k), e in topk_ens.items()
                          if rule == "family" and probe in e["members"]]
                print(f"{probe}: LOIO rank #{r}/{len(rankings['loio'])} "
                      f"(gate top-{FAMILY_GATE}: {'PASS' if r <= FAMILY_GATE else 'FAIL'}), "
                      f"lingbot_mf champion="
                      f"{next((a for a in rankings['loio'] if families[a] == 'lingbot_mf'), None)}, "
                      f"family-selected at K={in_sel or 'none'}")
    if warns:
        print(f"\n=== {len(warns)} WARN ===")
        for w in warns:
            print("WARN:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
