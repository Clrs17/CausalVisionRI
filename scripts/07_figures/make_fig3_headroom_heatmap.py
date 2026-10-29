# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
# Figure 3 — headroom (PI − VMAX) effect-modification heatmap
# Quantitative grid: two panels (a: DV24, b: RI), 5 treatments x 3 headroom regimes.
# Cells show descriptive point estimates; regime contrasts and their uncertainty
# are evaluated in data/processed_sensitivity/headroom_contrast_ci.csv.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from pathlib import Path
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("WPAC_HEADROOM_CATE", ROOT / "data" / "phase1" / "processed" / "headroom_cate_results.csv"))
OUT_BASE = str(Path(os.environ.get("WPAC_FIG3_OUT", ROOT / "paper" / "fig3_headroom_heatmap")))

TREAT_ORDER = ["SST_IC", "OHC_IC", "SHRD", "RHLO"]
ROW_LABELS = {
    "SST_OA": r"SST$_{\mathrm{OA}}^{\,\mathrm{a}}$",
    "SST_IC": r"SST$_{\mathrm{IC}}$",
    "OHC_IC": r"OHC$_{\mathrm{IC}}$",
    "SHRD": "SHRD",
    "RHLO": "RHLO",
}
REGIME_ORDER = ["low(<30kt)", "mid(30-60kt)", "high(>60kt)"]
COL_LABELS = ["Low\n(<30 kt)", "Mid\n(30\u201360 kt)", "High\n(>60 kt)"]

# Low-saturation diverging map: dynamic blue (negative) -> white -> thermal red (positive)
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "headroom_div", ["#4A7A96", "#BFD3DD", "#FFFFFF", "#E8C6C7", "#C1666B"]
)

df = pd.read_csv(DATA)
assert set(df["low_n_flag"].unique()) == {False}, "unexpected low_n cells present"


def build_matrix(outcome):
    sub = df[df["outcome"] == outcome]
    cate = np.full((len(TREAT_ORDER), len(REGIME_ORDER)), np.nan)
    nobs = np.full_like(cate, np.nan)
    for i, t in enumerate(TREAT_ORDER):
        for j, r in enumerate(REGIME_ORDER):
            row = sub[(sub["treatment"] == t) & (sub["headroom_regime"] == r)]
            assert len(row) == 1, (outcome, t, r)
            cate[i, j] = row["cate"].iloc[0]
            nobs[i, j] = row["n_obs"].iloc[0]
    return cate, nobs


def luminance(rgba):
    r, g, b = rgba[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def draw_panel(ax, cate, nobs, cbar_label, panel_tag, show_ylabels=True,
               value_fmt="{:+.2f}"):
    vmax = np.nanmax(np.abs(cate))
    # symmetric limits, padded slightly so extremes are not clipped visually
    vmax = np.ceil(vmax * 10) / 10
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(cate, cmap=CMAP, norm=norm, aspect="auto")

    # white cell separators
    for edge in np.arange(0.5, cate.shape[0] - 0.5 + 1, 1):
        ax.axhline(edge, color="white", lw=1.2)
    for edge in np.arange(0.5, cate.shape[1] - 0.5 + 1, 1):
        ax.axvline(edge, color="white", lw=1.2)

    for i in range(cate.shape[0]):
        for j in range(cate.shape[1]):
            rgba = CMAP(norm(cate[i, j]))
            txt_col = "white" if luminance(rgba) < 0.45 else "#272727"
            ax.text(j, i - 0.13, value_fmt.format(cate[i, j]),
                    ha="center", va="center", fontsize=7, color=txt_col,
                    fontweight="bold")
            ax.text(j, i + 0.24, f"n={int(nobs[i, j]):,}",
                    ha="center", va="center", fontsize=6.2, color=txt_col,
                    alpha=0.85)

    ax.set_xticks(range(len(COL_LABELS)))
    ax.set_xticklabels(COL_LABELS, fontsize=7)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks(range(len(TREAT_ORDER)))
    if show_ylabels:
        ax.set_yticklabels([ROW_LABELS[t] for t in TREAT_ORDER], fontsize=7)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(-0.02, 1.10, panel_tag, transform=ax.transAxes,
            fontsize=8, fontweight="bold", ha="left", va="bottom")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(cbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.6, length=2)
    cbar.outline.set_linewidth(0.6)
    return im


cate_dv, n_dv = build_matrix("DV24")
cate_ri, n_ri = build_matrix("RI")

MM = 1 / 25.4
fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 68 * MM),
                         gridspec_kw={"wspace": 0.30, "width_ratios": [1, 1]})

draw_panel(axes[0], cate_dv, n_dv,
           r"State-conditional estimate on $\Delta V_{24}$ (kt 24 h$^{-1}$ per SD)",
           "a", show_ylabels=True)
draw_panel(axes[1], cate_ri, n_ri,
           "State-conditional estimate on RI (logit per SD)",
           "b", show_ylabels=False)

fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.14)

fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
fig.savefig(OUT_BASE + ".png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved", OUT_BASE + ".pdf/.png")
print("DV24 symmetric limit:", np.ceil(np.nanmax(np.abs(cate_dv)) * 10) / 10)
print("RI   symmetric limit:", np.ceil(np.nanmax(np.abs(cate_ri)) * 10) / 10)
