# Figure 5 — conditional RI frequency by SHRD bin (storm-block-bootstrap 95% CI).
# Data: ../data/processed_sensitivity/shrd_ri_rate_by_bin.csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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
CSV = "../data/processed_sensitivity/shrd_ri_rate_by_bin.csv"
C_DYN = "#4A7A96"          # dynamical-factor family colour (same as fig. 2)

df = pd.read_csv(CSV)
nbin = len(df)
x = np.arange(nbin)
rate = df["ri_rate"] * 100.0
ci = pd.read_csv("../data/processed_sensitivity/shrd_ri_rate_stormblock_ci.csv")
lo = ci["ci_lo"].values * 100.0
hi = ci["ci_hi"].values * 100.0
binlab = (df["shrd_bin"].str.replace(r"[\[\)\s]", "", regex=True)
          .str.replace(",", "–").str.replace(r"\.0(?=–|$)", "", regex=True))

fig = plt.figure(figsize=(89 * MM, 68 * MM))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.4], hspace=0.06)
axn = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharex=axn)

# ---- top strip: sample size per bin ----
axn.bar(x, df["n"], width=0.62, color="0.82", ec="none", zorder=1)
for xi, ni in zip(x, df["n"]):
    axn.text(xi, ni + 60, f"{ni:,}", ha="center", va="bottom",
             fontsize=4.8, color="0.35")
axn.set_ylim(0, df["n"].max() * 1.28)
axn.set_xlim(-0.55, nbin - 0.45)
axn.spines[["top", "right", "left"]].set_visible(False)
axn.set_yticks([])
axn.set_xticks(x)
plt.setp(axn.get_xticklabels(), visible=False)
axn.tick_params(axis="x", length=0)
axn.set_ylabel("n", fontsize=6, rotation=0, labelpad=6, va="center", color="0.35")
axn.text(-0.06, 1.02, "a", transform=axn.transAxes, fontsize=8,
         fontweight="bold", va="bottom", ha="left")

# ---- main panel: conditional RI rate with storm-block bootstrap 95% CI ----
ax.axvline(4.5, color="0.45", lw=0.7, ls=(0, (4, 3)), zorder=0)
ax.axhspan(0, 0, color="none")  # keep origin visible
ax.plot(x, rate, "-", color=C_DYN, lw=1.0, zorder=2, alpha=0.85)
for xi, r, l, h in zip(x, rate, lo, hi):
    ax.plot([xi, xi], [l, h], color=C_DYN, lw=1.1,
            solid_capstyle="round", zorder=2)
    ax.plot([xi - 0.10, xi + 0.10], [l, l], color=C_DYN, lw=0.8, zorder=2)
    ax.plot([xi - 0.10, xi + 0.10], [h, h], color=C_DYN, lw=0.8, zorder=2)
ax.plot(x, rate, "o", ms=3.6, mfc=C_DYN, mec=C_DYN, mew=0.9, zorder=3)

# point value labels (offset to avoid CI caps and the threshold line)
for xi, r in zip(x, rate):
    if xi == 4:  # 12.5–15 bin sits left of the ≈15 kt threshold line
        ax.annotate(f"{r:.1f}", xy=(xi, r), xytext=(xi - 0.34, r + 0.85),
                    fontsize=5.2, color=C_DYN, ha="right", va="bottom")
    else:
        ax.annotate(f"{r:.1f}", xy=(xi, r), xytext=(xi + 0.26, r + 0.55),
                    fontsize=5.2, color=C_DYN, ha="left", va="bottom")

ax.annotate("≈15 kt\nthreshold", xy=(4.5, 17.3), fontsize=5.4,
            color="0.30", ha="center", va="top", linespacing=1.15)

ax.set_xticks(x)
ax.set_xticklabels(binlab, fontsize=6)
ax.set_xlabel("Deep-layer vertical wind shear SHRD (kt)", fontsize=7)
ax.set_ylabel("RI conditional frequency (%)", fontsize=7)
ax.set_ylim(0, 19.5)
ax.set_yticks([0, 5, 10, 15])
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="both", labelsize=6.5)
ax.text(-0.06, 1.02, "b", transform=ax.transAxes, fontsize=8,
        fontweight="bold", va="bottom", ha="left")

fig.subplots_adjust(left=0.115, right=0.985, top=0.965, bottom=0.13)
fig.savefig("fig5_shrd_ri_rate.pdf", bbox_inches="tight")
fig.savefig("fig5_shrd_ri_rate.png", dpi=300, bbox_inches="tight")
print("saved fig5_shrd_ri_rate.pdf/.png ; total n =", df["n"].sum())
