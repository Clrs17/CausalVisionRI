"""Metrics for HOLDOUT_MASKED_V1.

Two BSS columns are always reported separately:

* ``BSS_dev``    — Brier skill against a FIXED development-period RI rate
                   (2000-2019, computed once and passed in; never re-estimated
                   inside a bootstrap draw).  This is the primary score.
* ``BSS_sample`` — Brier skill against the evaluated sample's own RI rate.
                   Kept only for external comparability.

The module also implements the paired storm-cluster bootstrap used for every
main metric and ablation delta: storms are resampled with replacement and all
rows of a sampled storm are kept together, so within-storm label correlation
is respected.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_int(y) -> np.ndarray:
    return np.asarray(y).astype(int).ravel()


def brier(p, y) -> float:
    p = np.asarray(p, dtype=float).ravel()
    y = _as_int(y)
    return float(np.mean((p - y) ** 2))


def bss(p, y, clim_rate: Optional[float] = None) -> float:
    """Brier skill score with the EXACT climatological reference.

    ``BS_clim = mean((pi - y)^2) = pi^2 + (1 - 2*pi) * mean(y)`` — not
    ``pi*(1-pi)``: the two coincide only when the evaluated sample's base
    rate equals ``pi``.  With a fixed development rate they differ, and using
    ``pi*(1-pi)`` would inflate the score whenever the test base rate is lower
    than the development rate.
    """
    y = _as_int(y)
    if len(y) == 0:
        return float("nan")
    if clim_rate is None:
        clim_rate = float(y.mean())
    clim_rate = float(min(max(clim_rate, 1e-6), 1 - 1e-6))
    y_bar = float(y.mean())
    bs_clim = clim_rate ** 2 + (1.0 - 2.0 * clim_rate) * y_bar
    return 1.0 - brier(p, y) / max(bs_clim, 1e-12)



def bss_components(p, y, clim_rate: float) -> Dict[str, float]:
    """BS, the exact climatological BS, and their skill score."""
    y = _as_int(y)
    bs = brier(p, y)
    bs_clim = float(clim_rate) ** 2 + (1.0 - 2.0 * float(clim_rate)) * float(y.mean())
    return {"BS": bs, "BS_clim": bs_clim,
            "BSS": 1.0 - bs / max(bs_clim, 1e-12)}


def classification_metrics(p, y, threshold: float) -> Dict[str, float]:
    p = np.asarray(p, dtype=float).ravel()
    y = _as_int(y)
    pred = p >= float(threshold)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "POD": tp / max(tp + fn, 1),
        "FARate": fp / max(fp + tn, 1),      # POFD
        "FARatio": fp / max(tp + fp, 1),     # 1 - precision
        "CSI": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "threshold": float(threshold),
    }


def auc_metrics(p, y) -> Dict[str, float]:
    p = np.asarray(p, dtype=float).ravel()
    y = _as_int(y)
    if len(np.unique(y)) < 2:
        return {"ROC_AUC": float("nan"), "PR_AUC": float("nan")}
    return {"ROC_AUC": float(roc_auc_score(y, p)),
            "PR_AUC": float(average_precision_score(y, p))}


def f1_threshold(p, y, lo: float = 0.05, hi: float = 0.95,
                 step: float = 0.025) -> float:
    """F1-maximising threshold (same convention as the training protocol)."""
    p = np.asarray(p, dtype=float).ravel()
    y = _as_int(y)
    best_t, best_f = 0.5, -1.0
    for t in np.arange(lo, hi + 1e-9, step):
        pred = p >= t
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best_f:
            best_f, best_t = f1, float(t)
    return best_t


def reliability_table(p, y, bins: int = 10) -> Dict[str, list]:
    p = np.asarray(p, dtype=float).ravel()
    y = _as_int(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = {"bin_lo": [], "bin_hi": [], "n": [], "mean_p": [],
           "obs_rate": []}
    for b in range(bins):
        sel = idx == b
        out["bin_lo"].append(float(edges[b]))
        out["bin_hi"].append(float(edges[b + 1]))
        out["n"].append(int(sel.sum()))
        out["mean_p"].append(float(p[sel].mean()) if sel.any() else float("nan"))
        out["obs_rate"].append(float(y[sel].mean()) if sel.any() else float("nan"))
    return out


def metrics_row(p, y, threshold: float,
                dev_rate: float) -> Dict[str, float]:
    """Full metric row with BOTH BSS conventions, separated."""
    y = _as_int(y)
    p = np.asarray(p, dtype=float).ravel()
    dev = bss_components(p, y, dev_rate)
    sample = bss_components(p, y, float(y.mean()) if len(y) else 0.5)
    row = {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "dev_rate": float(dev_rate),
        "BS": dev["BS"],
        "BS_clim_dev": dev["BS_clim"],
        "BSS_dev": dev["BSS"],
        "BS_clim_sample": sample["BS_clim"],
        "BSS_sample": sample["BSS"],
    }
    row.update(auc_metrics(p, y))
    row.update(classification_metrics(p, y, threshold))
    return row



# ----------------------------------------------------------------------
# paired storm-cluster bootstrap
# ----------------------------------------------------------------------

def _cluster_indices(cluster: np.ndarray) -> Dict[object, np.ndarray]:
    out: Dict[object, np.ndarray] = {}
    for c in np.unique(cluster):
        out[c] = np.where(cluster == c)[0]
    return out


def paired_cluster_bootstrap(
        y, p_a, p_b, cluster,
        metric: Callable[..., float],
        n_boot: int = 2000,
        seed: int = 42,
        alpha: float = 0.10,
        fixed_rate: Optional[float] = None,
        row_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Bootstrap CI of metric(A) - metric(B), resampling whole storms.

    ``metric(p, y)`` when ``fixed_rate is None`` else ``metric(p, y, fixed_rate)``.
    Rows must be aligned across p_a, p_b, y and cluster.
    """
    y = _as_int(y)
    p_a = np.asarray(p_a, dtype=float).ravel()
    p_b = np.asarray(p_b, dtype=float).ravel()
    cluster = np.asarray(cluster)
    if not (len(y) == len(p_a) == len(p_b) == len(cluster)):
        raise ValueError("paired bootstrap requires equal-length, aligned arrays")

    def _m(p, yy):
        return metric(p, yy) if fixed_rate is None else metric(p, yy, fixed_rate)

    point = _m(p_a, y) - _m(p_b, y)
    gmap = _cluster_indices(cluster)
    keys = np.array(list(gmap.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([gmap[c] for c in pick])
        draws[i] = _m(p_a[idx], y[idx]) - _m(p_b[idx], y[idx])
    lo, hi = np.percentile(draws, [100.0 * alpha / 2, 100.0 * (1 - alpha / 2)])
    p_boot = 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean()))
    p_boot = min(max(p_boot, 1.0 / (n_boot + 1)), 1.0)
    return {
        "delta": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "alpha": float(alpha),
        "p_boot": float(p_boot),
        "n_boot": int(n_boot),
        "n_clusters": int(len(keys)),
        "n_rows": int(len(y)),
    }


def holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    names = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m = len(names)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(names):
        val = min(1.0, (m - rank) * float(pvalues[name]))
        running = max(running, val)
        adjusted[name] = running
    return adjusted


def paired_bss_delta_dev(y, p_a, p_b, cluster, dev_rate: float,
                         **kw) -> Dict[str, float]:
    return paired_cluster_bootstrap(y, p_a, p_b, cluster, bss,
                                    fixed_rate=dev_rate, **kw)
