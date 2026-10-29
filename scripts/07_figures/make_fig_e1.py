# Fig. E1 — forecast utility of shear causal threshold (Experiment E1).
# Data: ../analysis/forecast_utility/e1_threshold_utility.csv
# Panel a: RandomForest POD / FAR across shear strata for S0/S1/S2 (95% CI).
# Panel b: overall BSS and AUC for both models x three schemes (95% CI).
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
CSV = "../analysis/forecast_utility/e1_threshold_utility.csv"
OUT = "fig_e1_threshold_utility"

C_SCHEME = {"S0": "0.55", "S1": "#4A7A96", "S2": "#B0764A"}
LAB_SCHEME = {"S0": "S0  no threshold", "S1": "S1  SHRD<10 kt",
              "S2": "S2  SHRD<15 kt"}
STRATA = ["<10", "10–15", "15–20", "≥20"]
XLAB = ["<10", "10–15", "15–20", r"$\geq$20"]

df = pd.read_csv(CSV)
df = df[df["scheme"].isin(["S0", "S1", "S2"])]

fig = plt.figure(figsize=(178 * MM, 72 * MM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30,
                      left=0.055, right=0.985, top=0.86, bottom=0.16)

# ---------------- panel a: RF POD & FAR across strata ----------------
gsa = gs[0].subgridspec(2, 1, hspace=0.55)
rf = df[df["model"] == "RandomForest"].set_index(["scheme", "stratum"])
x = np.arange(len(STRATA))
offsets = {"S0": -0.24, "S1": 0.0, "S2": 0.24}

for row, (met, ylab, ylim) in enumerate([
        ("POD", "POD", (-0.02, 0.75)),
        ("FAR", "FAR", (0.30, 0.80))]):
    ax = fig.add_subplot(gsa[row])
    ax.axvspan(0.62, 1.38, color="#B0764A", alpha=0.10, lw=0, zorder=0)
    for sch in ["S0", "S1", "S2"]:
        xs, ys, lo, hi = [], [], [], []
        for xi, st in zip(x, STRATA):
            r = rf.loc[(sch, st)]
            xs.append(xi + offsets[sch]); ys.append(r[met])
            lo.append(r[f"{met}_lo"]); hi.append(r[f"{met}_hi"])
        ys = np.array(ys, float); lo = np.array(lo, float); hi = np.array(hi, float)
        err = np.vstack([np.clip(ys - lo, 0, None), np.clip(hi - ys, 0, None)])
        ax.errorbar(xs, ys, yerr=err, fmt="o", ms=2.8, lw=0.9,
                    color=C_SCHEME[sch], mfc="white", mec=C_SCHEME[sch],
                    mew=0.8, elinewidth=0.7, capsize=1.6, capthick=0.6,
                    label=LAB_SCHEME[sch] if row == 0 else None, zorder=3)
    ax.set_xlim(-0.55, len(STRATA) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB if row == 1 else [])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(ylab, fontsize=7)
    if row == 1:
        ax.set_xlabel("Deep-layer shear stratum (kt)", fontsize=7)
    if row == 0:
        ax.text(1.0, 0.98, "transition band", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5.2, color="#B0764A")
        ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.32), ncols=3,
                  fontsize=6, handletextpad=0.4, columnspacing=1.0,
                  borderaxespad=0.0)
        ax.text(-0.09, 1.34, "a", transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")

# ---------------- panel b: overall BSS & AUC ----------------
gsb = gs[1].subgridspec(1, 2, wspace=0.45)
ov = df[df["stratum"] == "overall"].set_index(["model", "scheme"])
models = ["Logistic_SHIPS", "RandomForest"]
mlab = {"Logistic_SHIPS": "Logistic", "RandomForest": "RF"}
xb = np.arange(len(models))

for col, (met, ttl, ylim) in enumerate([
        ("BSS", "BSS (overall)", (-1.35, 0.35)),
        ("AUC", "AUC (overall)", (0.65, 0.95))]):
    ax = fig.add_subplot(gsb[col])
    w = 0.24
    for j, sch in enumerate(["S0", "S1", "S2"]):
        xs, ys, lo, hi = [], [], [], []
        for xi, m in zip(xb, models):
            r = ov.loc[(m, sch)]
            xs.append(xi + (j - 1) * (w + 0.03)); ys.append(r[met])
            lo.append(r[f"{met}_lo"]); hi.append(r[f"{met}_hi"])
        ys = np.array(ys, float)
        err = np.vstack([np.clip(ys - np.array(lo, float), 0, None),
                         np.clip(np.array(hi, float) - ys, 0, None)])
        ax.bar(xs, ys, width=w, color=C_SCHEME[sch], alpha=0.85, ec="none",
               zorder=2, label=sch if col == 0 else None)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="0.25",
                    elinewidth=0.7, capsize=1.6, capthick=0.6, zorder=3)
    ax.axhline(0, color="0.35", lw=0.6, zorder=1)
    ax.set_xticks(xb)
    ax.set_xticklabels([mlab[m] for m in models], fontsize=7)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(ttl, fontsize=7, pad=3)
    if col == 0:
        ax.text(-0.30, 1.06, "b", transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=300)
print("saved:", OUT + ".pdf / .png")
