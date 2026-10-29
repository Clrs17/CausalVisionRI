# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
# Figure 1 — prior-constrained PCMCI+ causal graph (375-storm WPac panel)
# Panel a: contemporaneous (lag-0) links; panel b: lagged (lag 1-4) links.
# Data: wpac_causal_phase1/data/processed/causal_graph_edges.json (134 edges)
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7,
    "legend.frameon": False,
})

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("WPAC_GRAPH_JSON", ROOT.parent / "data" / "processed" / "causal_graph_edges.json"))
OUT = Path(os.environ.get("WPAC_FIG1_OUT", ROOT / "fig1_causal_graph"))

with open(DATA) as f:
    graph = json.load(f)
EDGES = graph["edges"]

# ---------------------------------------------------------------- groups ----
GROUPS = {
    "state":    ["VMAX", "PER", "SPD", "LAT", "LAND_DIST"],
    "dynamics": ["SHRD", "SHEAR_VMAX_ANGLE", "VORT", "PV", "D200", "W500"],
    "moisture": ["RHLO", "RHHI", "MFC", "DRY_AIR"],
    "ocean":    ["SST_IC", "SST_OA", "OHC_IC", "OHC_OA", "SLHF_IC", "WARM_CORE"],
}
GROUP_OF = {v: g for g, vs in GROUPS.items() for v in vs}

# family colours (border) / light tints (fill) — colour-blind safe, low saturation
FAM = {
    "state":    {"ec": "#6A8E7F", "fc": "#DCE8E2"},
    "dynamics": {"ec": "#4A7A96", "fc": "#D5E1E9"},
    "moisture": {"ec": "#8C8C8C", "fc": "#E4E4E4"},
    "ocean":    {"ec": "#C1666B", "fc": "#EFD9DA"},
    "target":   {"ec": "#272727", "fc": "#272727"},
}
EDGE_POS = "#A34E54"   # darker dusty red  = positive partial corr
EDGE_NEG = "#3B627A"   # darker steel blue = negative partial corr

# ------------------------------------------------------------- node layout ---
# shared layout for both panels; DV24_YJ right, ocean nodes at the bottom
POS = {
    # storm state & track (left column)
    "PER":       (10, 86), "VMAX": (12, 64), "LAT": (9, 44),
    "SPD":       (12, 24), "LAND_DIST": (9, 6),
    # atmospheric dynamics (top centre/right)
    "SHRD":      (37, 92), "SHEAR_VMAX_ANGLE": (61, 93),
    "VORT":      (38, 71), "PV": (63, 77), "D200": (73, 61), "W500": (52, 59),
    # moisture (middle band)
    "RHLO":      (30, 48), "RHHI": (43, 50), "MFC": (33, 38), "DRY_AIR": (48, 40),
    # ocean & surface heat (bottom)
    "WARM_CORE": (72, 40),
    "SLHF_IC":   (44, 10), "SST_IC": (60, 8), "OHC_IC": (72, 16),
    "SST_OA":    (82, 6),  "OHC_OA": (93, 14),
    # target
    "DV24_YJ":   (92, 52),
}
LABEL = {v: v for v in POS}
LABEL["DV24_YJ"] = "$\\Delta V_{24}$"

# (x0, y0, x1, y1, name, label position)
ZONES = [
    (1.5, 1.5, 25.5, 96.5, "Storm state & track", "tl"),
    (29, 56, 80, 98.5, "Atmospheric dynamics", "tl"),
    (27, 33, 56, 54, "Moisture", "tl"),
    (38, 1.5, 98.5, 21, "Ocean & surface heat", "bl"),
]

MAXC = max(abs(e["coef"]) for e in EDGES)  # 0.407

# manual curvature overrides for legibility: (src, dst, lag) -> rad
RAD_OVR = {
    ("VMAX", "DV24_YJ", 0): 0.12,   # arc over W500
    ("PER", "DV24_YJ", 0): 0.08,    # arc over VORT
    ("VORT", "DV24_YJ", 0): -0.05,  # dip below D200
    ("W500", "MFC", 0): 0.25,       # swing clear of RHHI
    ("PER", "VMAX", 1): 0.18,       # bow inward (right)
    ("VMAX", "PER", 4): -0.18,
    ("VMAX", "OHC_IC", 1): -0.18,   # cold wake: bow through open corridor
    ("VMAX", "OHC_IC", 2): -0.30,
    ("VMAX", "DV24_YJ", 3): 0.12,   # arc over W500
    ("LAT", "DV24_YJ", 1): -0.05,   # dip below RHLO
}

def edge_style(coef):
    f = abs(coef) / MAXC
    lw = 0.30 + 1.5 * f
    alpha = 0.08 + 0.26 * f
    color = EDGE_POS if coef > 0 else EDGE_NEG
    return color, lw, alpha

def compute_rads(edges):
    groups = defaultdict(list)
    for e in edges:
        groups[(e["src"], e["dst"])].append(e)
    rads = {}
    for (s, t), es in groups.items():
        es = sorted(es, key=lambda x: x["lag"])
        n = len(es)
        recip = (t, s) in groups
        for i, e in enumerate(es):
            r = 0.18 * (i - (n - 1) / 2)
            if recip:
                direction = 1 if s < t else -1
                r += direction * 0.14
            key = (e["src"], e["dst"], e["lag"])
            rads[id(e)] = RAD_OVR.get(key, r)
    return rads

def draw_nodes(ax):
    for v, (x, y) in POS.items():
        g = "target" if v == "DV24_YJ" else GROUP_OF[v]
        st = FAM[g]
        tc = "white" if g == "target" else "#1a1a1a"
        fw = "bold" if g == "target" else "normal"
        fs = 7.0 if g == "target" else 6.0
        ax.text(x, y, LABEL[v], ha="center", va="center", fontsize=fs,
                fontweight=fw, color=tc, zorder=5,
                bbox=dict(boxstyle="round,pad=0.34", fc=st["fc"], ec=st["ec"],
                          lw=1.0 if g == "target" else 0.7))

def draw_zones(ax):
    for x0, y0, x1, y1, name, lp in ZONES:
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.8,rounding_size=2.2",
            fc="#FAFAFA", ec="#D0D0D0", lw=0.5, ls=(0, (3, 2)), zorder=0))
        if lp == "tl":
            ax.text(x0 + 1.2, y1 - 1.0, name, fontsize=6.0, style="italic",
                    color="#666666", ha="left", va="top", zorder=1)
        else:
            ax.text(x0 + 1.2, y0 + 1.0, name, fontsize=6.0, style="italic",
                    color="#666666", ha="left", va="bottom", zorder=1)

def draw_edge(ax, e, directed, rad, lw_scale=1.0, alpha_boost=0.0, zorder=2):
    color, lw, alpha = edge_style(e["coef"])
    x0, y0 = POS[e["src"]]
    x1, y1 = POS[e["dst"]]
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>" if directed else "-",
        mutation_scale=5.5,
        connectionstyle=f"arc3,rad={rad:.3f}",
        shrinkA=11, shrinkB=11,
        color=color, lw=lw * lw_scale,
        alpha=min(1.0, alpha + alpha_boost),
        zorder=zorder, capstyle="round")
    ax.add_patch(arr)
    return arr

# ------------------------------------------------------------------ figure ---
fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 100 / 25.4))

for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_zones(ax)

# ---- panel a: contemporaneous (lag 0) ---------------------------------------
axA = axes[0]
lag0 = [e for e in EDGES if e["lag"] == 0]
radsA = compute_rads(lag0)
for e in lag0:  # predictor-predictor links (undirected)
    if e["dst"] != "DV24_YJ":
        draw_edge(axA, e, directed=False, rad=radsA[id(e)], lw_scale=1.15,
                  alpha_boost=0.45, zorder=2)
for e in lag0:  # prior-oriented links into DV24
    if e["dst"] == "DV24_YJ":
        draw_edge(axA, e, directed=True, rad=radsA[id(e)], lw_scale=1.7,
                  alpha_boost=0.62, zorder=3)
draw_nodes(axA)
axA.set_title("Contemporaneous links (lag 0)", fontsize=7.5, pad=3)
axA.text(-0.035, 1.015, "a", transform=axA.transAxes, fontsize=8,
         fontweight="bold", ha="left", va="bottom")

# ---- panel b: lagged (lag >= 1) ---------------------------------------------
axB = axes[1]
lagged = [e for e in EDGES if e["lag"] >= 1]
radsB = compute_rads(lagged)

KEY = {  # emphasised physical paths
    ("VMAX", "OHC_IC", 1),   # cold-wake feedback
    ("SHRD", "W500", 1),     # shear-ascent link (negative = enhanced ascent)
    ("SHRD", "VMAX", 1),     # shear suppresses intensity
    ("PER", "VMAX", 1),      # persistence
    ("SPD", "OHC_IC", 1),    # translation-ocean coupling
}
into_dv = [e for e in lagged if e["dst"] == "DV24_YJ"]
key_edges, bg_edges = [], []
for e in lagged:
    if e["dst"] == "DV24_YJ" or (e["src"], e["dst"], e["lag"]) in KEY:
        key_edges.append(e)
    else:
        bg_edges.append(e)

for e in bg_edges:  # faint background fabric
    draw_edge(axB, e, directed=True, rad=radsB[id(e)], zorder=1.5)
for e in key_edges:  # emphasised structure
    draw_edge(axB, e, directed=True, rad=radsB[id(e)], lw_scale=2.4,
              alpha_boost=0.62, zorder=3)

LAG_TAG = {
    ("OHC_OA", 1): (50, 34.6),
    ("VMAX", 1): (60.5, 65.5),
    ("VMAX", 3): (70, 55.0),
}
for e in into_dv:
    key = (e["src"], e["lag"])
    if key not in LAG_TAG:
        continue
    tx, ty = LAG_TAG[key]
    tag = f"{e['src']} lag {e['lag']}" if e["src"] == "VMAX" else f"lag {e['lag']}"
    axB.text(tx, ty, tag, fontsize=5.6, color="#333333",
             ha="center", va="center", zorder=4,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#999999",
                       lw=0.4, alpha=0.92))
cold = next((e for e in EDGES
             if e["src"] == "VMAX" and e["dst"] == "OHC_IC" and e["lag"] == 1), None)
if cold is not None:
    axB.annotate(f"cold-wake feedback\nVMAX$\\rightarrow$OHC_IC, lag 1 ({cold['coef']:+.2f})",
                 xy=(36.5, 35.5), xytext=(33.5, 29.8), fontsize=6.0,
                 color="#7A3B40", ha="center", va="center", zorder=4,
                 arrowprops=dict(arrowstyle="-", color="#7A3B40", lw=0.6,
                                 shrinkA=1, shrinkB=2),
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#C1666B",
                           lw=0.5, alpha=0.95))
draw_nodes(axB)
axB.set_title("Lagged links (lag 1–4, 6 h per lag)", fontsize=7.5, pad=3)
axB.text(-0.035, 1.015, "b", transform=axB.transAxes, fontsize=8,
         fontweight="bold", ha="left", va="bottom")

# ------------------------------------------------------ summary line + legend -
fig.text(0.5, 0.098,
         f"a: {len(lag0)} contemporaneous links; target directions are prior-fixed and other "
         f"directions are PCMCI+-oriented but suppressed for readability.   "
         f"b: {len(lagged)} lagged links; parents of $\\Delta V_{{24}}$ and key physical "
         "paths emphasised.",
         fontsize=6.0, color="#444444", ha="center", va="center")
ocean_direct = [e for e in EDGES
                if e["dst"] == "DV24_YJ" and e["src"] in GROUPS["ocean"]]
if ocean_direct:
    ocean_text = "Direct ocean links into $\\Delta V_{24}$: " + ", ".join(
        f"{e['src']} lag {e['lag']}" for e in ocean_direct) + "."
else:
    ocean_text = "No direct ocean link into $\\Delta V_{24}$; ocean influence is indirect."
fig.text(0.5, 0.072, ocean_text,
         fontsize=6.0, color="#7A3B40", ha="center", va="center")
fig.text(0.5, 0.054, "Six additional unorientable adjacencies are listed in Appendix A.",
         fontsize=5.8, color="#444444", ha="center", va="center")

handles = [
    Patch(fc=FAM["state"]["fc"], ec=FAM["state"]["ec"], lw=0.7,
          label="Storm state & track"),
    Patch(fc=FAM["dynamics"]["fc"], ec=FAM["dynamics"]["ec"], lw=0.7,
          label="Atmospheric dynamics"),
    Patch(fc=FAM["moisture"]["fc"], ec=FAM["moisture"]["ec"], lw=0.7,
          label="Moisture"),
    Patch(fc=FAM["ocean"]["fc"], ec=FAM["ocean"]["ec"], lw=0.7,
          label="Ocean & surface heat"),
    Patch(fc=FAM["target"]["fc"], ec=FAM["target"]["ec"], lw=0.7,
          label="Target $\\Delta V_{24}$"),
    Line2D([0], [0], color=EDGE_POS, lw=1.4, label="Positive link"),
    Line2D([0], [0], color=EDGE_NEG, lw=1.4, label="Negative link"),
    Line2D([0], [0], color="#888888", lw=0.6,
           label="Width / opacity $\\propto$ |partial corr.|"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.0,
           handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, -0.028))

fig.subplots_adjust(left=0.015, right=0.985, top=0.93, bottom=0.165,
                    wspace=0.06)
fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
fig.savefig(str(OUT) + ".png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved:", OUT.with_suffix(".pdf"))
print("edges drawn:", len(lag0), "+", len(lagged), "=", len(lag0) + len(lagged))
