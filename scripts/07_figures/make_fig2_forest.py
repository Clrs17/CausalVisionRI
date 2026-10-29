# Figure 2 — forest plot of graph-adjusted ATEs (DV24 and RI panels).
# Data: ../data/processed/ate_results.csv (block-bootstrap CIs, BH q-values).
import numpy as np
import os
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
})

MM = 1 / 25.4
CSV = os.environ.get("WPAC_ATE_CSV", "../data/processed/ate_results.csv")
OUT = Path(os.environ.get("WPAC_FIG2_OUT", "fig2_ate_forest"))
DV24_SD = 23.69032833176526   # SD of DV24 (kt/24 h) over the analysis sample
SHIPS_STD = -0.26             # standardized SHIPS 24-h shear coefficient (DeMaria & Kaplan 1994)
SHIPS_KT = SHIPS_STD * DV24_SD  # -> -6.16 kt/24 h per SD

CAT = {  # physical category per treatment
    "SHRD": "dyn", "VORT": "dyn", "PV": "dyn", "D200": "dyn", "W500": "dyn",
    "RHLO": "thermo", "WARM_CORE": "thermo", "MFC": "thermo",
    "OHC_IC": "ocean", "OHC_OA": "ocean",
}
COL = {"dyn": "#4A7A96", "thermo": "#C1666B", "ocean": "#6A8E7F"}
CAT_LABEL = {"dyn": "Dynamical", "thermo": "Thermodynamic / moisture", "ocean": "Ocean state"}

df = pd.read_csv(CSV)
df = df[df.treatment != "D200"].copy()
dv = df[df.outcome == "DV24"].copy().sort_values("ate", ascending=True)  # bottom->top
order = dv["treatment"].tolist()           # most positive on top
ylabels = {t: t.replace("_", " ") for t in order}
y = {t: i for i, t in enumerate(order)}

fig, axes = plt.subplots(1, 2, figsize=(89 * MM, 82 * MM), sharey=True)
panels = [("DV24", "a", "DV24 ATE (kt 24 h$^{-1}$ per SD)"),
          ("RI", "b", "RI adjusted logit (per SD)")]

for ax, (outcome, tag, xlabel) in zip(axes, panels):
    sub = df[df.outcome == outcome]
    # shade rows with empty adjustment set
    for _, r in sub.iterrows():
        if str(r["adjustment_set"]).strip() in ("(empty)", "nan", ""):
            ax.axhspan(y[r["treatment"]] - 0.5, y[r["treatment"]] + 0.5,
                       color="0.92", zorder=0)
    ax.axvline(0, color="0.45", lw=0.7, ls=(0, (4, 3)), zorder=1)
    for _, r in sub.iterrows():
        yy = y[r["treatment"]]
        sig = r["q_value_bh"] < 0.05
        c = COL[CAT[r["treatment"]]]
        ax.plot([r["ci_boot_low"], r["ci_boot_high"]], [yy, yy],
                color=c, lw=1.1, solid_capstyle="round", zorder=2)
        ax.plot([r["ci_boot_low"], r["ci_boot_low"]], [yy - 0.13, yy + 0.13], color=c, lw=0.8, zorder=2)
        ax.plot([r["ci_boot_high"], r["ci_boot_high"]], [yy - 0.13, yy + 0.13], color=c, lw=0.8, zorder=2)
        ax.plot(r["ate_boot"], yy, "o", ms=3.6, mfc=c if sig else "white",
                mec=c, mew=0.9, zorder=3)
    if outcome == "DV24":
        yy = y["SHRD"]
        ax.plot([SHIPS_KT, SHIPS_KT], [yy - 0.42, yy + 0.42],
                color="0.35", lw=0.9, ls=(0, (2, 2)), zorder=2)
        ax.annotate("SHIPS\n$-0.26$ SD", xy=(SHIPS_KT, yy + 0.38),
                    xytext=(-2.0, yy + 0.65), fontsize=5.2,
                    color="0.25", ha="center", va="center", linespacing=1.1,
                    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.5,
                                    connectionstyle="arc3,rad=0.3"))
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.04, 1.04, tag, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")

axes[0].set_yticks([y[t] for t in order])
axes[0].set_yticklabels([ylabels[t] for t in order], fontsize=6.5)
axes[0].set_xlim(-11.5, 12.0)
axes[0].set_xticks([-10, -5, 0, 5, 10])
axes[1].set_xlim(-0.85, 1.05)
axes[1].set_xticks([-0.5, 0, 0.5, 1.0])
axes[1].tick_params(axis="x", labelsize=6.5)
axes[0].tick_params(axis="x", labelsize=6.5)

handles = [Line2D([], [], marker="o", ls="", ms=3.6, mfc=COL[k], mec=COL[k],
                  mew=0.9, label=CAT_LABEL[k]) for k in ("dyn", "thermo", "ocean")]
handles += [
    Line2D([], [], marker="o", ls="", ms=3.6, mfc="0.35", mec="0.35", label="$q_{BH}<0.05$"),
    Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec="0.35", mew=0.9,
           label="n.s. after FDR"),
    Patch(fc="0.92", ec="none", label="empty adjustment set"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.4,
           handlelength=1.2, handletextpad=0.45, columnspacing=0.8,
           borderaxespad=0.1, frameon=False)

fig.subplots_adjust(left=0.13, right=0.985, top=0.95, bottom=0.24, wspace=0.16)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
print("saved", OUT, "; SHIPS band at", round(SHIPS_KT, 2), "kt")
