# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""fig_compare_overview: the SOTA-narrative 2x2 on the consistent Fig-A sample
(2,178 rows, USA-30 kt): (a) PR, (b) ROC, (c) reliability, (d) yearly BSS,
with ours + baselines + WRPS1 + RITCF recon variants all present.

Publication conventions (Arial, 7 pt, bold a/b/c/d, Okabe-Ito, 600 dpi + PDF).
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "data" / "test" / "holdout_masked_v1"
RR = ROOT / "data" / "ritcf_recon"
FIG = ROOT / "data" / "analysis" / "figures_pub"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
MM = 1 / 25.4
C_OURS, C_GREY, C_ORANGE, C_SKY, C_BG = ("#0072B2", "#999999", "#E69F00",
                                         "#56B4E9", "#009E73")
C_VERM, C_PURP, C_BROWN = "#D55E00", "#CC79A7", "#4B2E2A"
C_YELLOW = "#B79F00"


def _save(fig, stem):
    import os
    import tempfile
    import time
    for suffix in (".png", ".pdf"):
        path = FIG / f"{stem}{suffix}"
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=str(path.parent))
        os.close(fd)
        try:
            fig.savefig(tmp, dpi=600 if suffix == ".png" else None)
            for _ in range(20):
                try:
                    os.replace(tmp, path)
                    break
                except OSError:
                    time.sleep(1.5)
            else:
                raise SystemExit(f"could not replace {path}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    print("wrote", stem)


def panel_label(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


# ---- assemble per-model scores on the Fig-A sample -----------------------
_z = np.load(TD / "main.npz", allow_pickle=True)
ours_uids = _z["row_uid"].astype(str)
Y_USA = dict(zip(ours_uids, _z["y"].astype(int)))
YEAR = dict(zip(ours_uids, _z["storm_year"].astype(int)))

rec = pd.read_parquet(RR / "dataset_kt30" / "samples.parquet")
rec_test = set(rec.loc[rec.split == "test", "row_key"].astype(str))
df = pd.read_parquet(ROOT / "data/holdout_masked_v1" /
                     "ships12_predictors_fit0017_cohc.parquet")
finite = df.drop(columns=["row_uid"]).notna().all(axis=1)
ships = set(df["row_uid"].astype(str)[finite])
FIGA = set(ours_uids) & rec_test & ships
print(f"compare sample: {len(FIGA)} rows")


def subset_npz(name):
    z = np.load(TD / f"{name}.npz", allow_pickle=True)
    uids = z["row_uid"].astype(str)
    keep = np.array([u in FIGA for u in uids])
    u, s = uids[keep], z["p"][keep]
    y = np.array([Y_USA[x] for x in u])
    yr = np.array([YEAR[x] for x in u])
    return s, y, yr


def subset_recon(variant):
    fname = f"sweep_{variant}_kt30" + os.environ.get("RECON_TAG_KIND", "") \
        + ".npz"
    z = np.load(RR / fname, allow_pickle=True)
    uids = z["test_row_uid"].astype(str)
    keep = np.array([u in FIGA for u in uids])
    u, s = uids[keep], z["test_score"][keep]
    y = np.array([Y_USA[x] for x in u])
    yr = np.array([YEAR[x] for x in u])
    return s, y, yr


def subset_wrps1():
    z = np.load(ROOT / "data/analysis/wrps1_scores_2024.npz", allow_pickle=True)
    uids = z["row_uid"].astype(str)
    keep = np.array([u in FIGA for u in uids])
    u, s = uids[keep], z["p"][keep]
    y = np.array([Y_USA[x] for x in u])
    yr = np.array([YEAR[x] for x in u])
    return s, y, yr


MODELS = [
    ("CVRI-Ens", lambda: subset_npz("main"), C_OURS, 1.8),
    ("LR", lambda: subset_npz("baseline_lr"), C_GREY, 0.9),
    ("LDA", lambda: subset_npz("baseline_lda"), C_ORANGE, 0.9),
    ("SHIPS-12 OHCD", lambda: subset_npz("baseline_ships12_lda"), C_SKY, 0.9),
    ("SHIPS-12 COHC", lambda: subset_npz("baseline_ships12_lda_cohc"), C_YELLOW,
     0.9),
    ("WRPS1", subset_wrps1, C_BG, 1.1),
    ("RITCF faithful", lambda: subset_recon("V1"), C_VERM, 0.9),
    ("RITCF repo skeleton", lambda: subset_recon("V3"), C_PURP, 0.9),
    ("RITCF direct clf", lambda: subset_recon("V4direct"), C_BROWN, 0.9),
]

fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 150 * MM))

from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

ax = axes[0, 0]
for label, fn, c, lw in MODELS:
    s, y, _ = fn()
    pr, rc, _ = precision_recall_curve(y, s)
    ax.plot(rc, pr, color=c, lw=lw, label=label)
base = np.mean([Y_USA[u] for u in FIGA])
ax.axhline(base, color="k", ls=":", lw=0.6)
ax.text(0.5, base + 0.015, f"base rate {base:.3f}", fontsize=6,
        color="#444444")
ax.set_xlabel("Recall (POD)")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
panel_label(ax, "a")

ax = axes[0, 1]
ax.plot([0, 1], [0, 1], "k--", lw=0.6)
for label, fn, c, lw in MODELS:
    s, y, _ = fn()
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=c, lw=lw, label=label)
ax.set_xlabel("False-detection rate (POFD)")
ax.set_ylabel("Probability of detection")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
panel_label(ax, "b")

ax = axes[1, 0]
ax.plot([0, 1], [0, 1], "k--", lw=0.6)
for label, fn, c, lw in MODELS:
    s, y, _ = fn()
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(s, bins) - 1
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 20:
            xs.append(s[m].mean())
            ys.append(y[m].mean())
    ax.plot(xs, ys, "o-", color=c, ms=3, lw=lw, label=label)
ax.set_xlabel("score (probability scale per model)")
ax.set_ylabel("Observed frequency")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
panel_label(ax, "c")

ax = axes[1, 1]
years = list(range(2020, 2025))
for label, fn, c, lw in MODELS:
    s, y, yr = fn()
    vals = []
    for yy in years:
        m = yr == yy
        if m.sum() < 30 or len(np.unique(y[m])) < 2:
            vals.append(np.nan)
            continue
        bs = np.mean((s[m] - y[m]) ** 2)
        bss = 1 - bs / max(np.mean((y[m].mean() - y[m]) ** 2), 1e-9)
        vals.append(bss)
    vals = np.array(vals)
    on = np.clip(vals, -1.4, None)
    ax.plot(years, on, "o-", color=c, lw=lw, ms=3, label=label)
    off = vals < -1.4
    if off.any():
        ax.scatter(np.array(years)[off], np.full(int(off.sum()), -1.36),
                   marker="v", color=c, s=22, zorder=4)
ax.set_ylim(-1.5, 0.45)
ax.set_xticks(years)
ax.set_ylabel("BSS (year's own climatology)")
ax.set_xlabel("year")
ax.text(0.02, 0.02, "triangles: value below −1.4 (off scale)",
        transform=ax.transAxes, fontsize=6, color="#444444")

panel_label(ax, "d")
handles = [plt.Line2D([], [], color=c, lw=lw, label=label)
           for label, _, c, lw in MODELS]
fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3,
           fontsize=6.2, bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(rect=(0, 0.08, 1, 1))
_save(fig, "fig_compare_overview")
plt.close(fig)
print("DONE")
