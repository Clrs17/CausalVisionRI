# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""All release figures, one consistent figure-publication style.

Style: Arial, 5-7 pt text, bold lowercase panel letters, 89/183 mm widths,
600 dpi PNG + vector PDF, RGB, Okabe-Ito colorblind-safe palette, no embedded
figure titles (panel letters + caption text instead).

Outputs to data/analysis/figures_pub/ (superseding data/analysis/figures/).

Figures:
  fig_main_overview      2x3: (a) PR, (b) ROC, (c) yearly BSS/CSI,
                         (d) reliability — official 3,004-row test, ours + baselines.
  fig_ablation_forest    causal ablation deltas with 90% CI.
  fig_ablation_forest_2020_2021 / _2020 / _2021
                         same panel on the 2020-2021 subset and on each year
                         alone (independent entries, FIG_ONLY=<stem>).
  fig_gate_members       (a) gate mean, (b) causal attention share.
  fig_selection_landscape  22 arms ranked by inner two-year-worst BSS.
  fig_member_ensemble_gain  M0 members vs Top-5 ensemble, inner and test BSS.
  fig_score_distributions  ours predicted-probability density by outcome.
  fig_cma_label_control  operating points on the Fig-A rows under CMA-30 kt.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    roc_curve

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "data" / "analysis" / "eval"
TD = ROOT / "data" / "test" / "holdout_masked_v1"
CMP = ROOT / "data" / "analysis" / "compare"
FIG = ROOT / "data" / "analysis" / "figures_pub"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

C_OURS = "#0072B2"
C_GREY = "#999999"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_BG = "#009E73"
C_VERM = "#D55E00"
C_PURP = "#CC79A7"
C_YELLOW = "#B79F00"
C_WRPS1 = "#009E73"
C_RITCF = "#D55E00"
C_V2 = "#E69F00"
C_V3 = "#CC79A7"
C_V4D = "#4B2E2A"

MM = 1 / 25.4


def _ordered_unique(items):
    """De-duplicate (colour, marker, label) triples, preserving first-seen order.

    A ``set`` comprehension here would order the legend by hash seed, so the
    published figure would differ between interpreters; the legend order must
    follow the declaration order of ``STYLE`` instead.
    """
    seen = []
    for item in items:
        triple = (item[0], item[1], item[2])
        if triple not in seen:
            seen.append(triple)
    return seen


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
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left")


def load_test(name):
    z = np.load(TD / f"{name}.npz", allow_pickle=True)
    return z["p"], z["y"].astype(int), z["storm_year"].astype(int)


MODELS = [("CVRI-Ens", "main", C_OURS, 1.8),
          ("LR", "baseline_lr", C_GREY, 0.9),
          ("LDA", "baseline_lda", C_ORANGE, 0.9),
          ("SHIPS-12 OHCD", "baseline_ships12_lda", C_SKY, 0.9),
          ("SHIPS-12 COHC", "baseline_ships12_lda_cohc", C_BG, 0.9)]

# ablation panel entities, shared by §2 (pooled) and §2b (2020-2021 subset)
MEMBERS = ["ens_lbB_lora16_sf", "ens_lbB_lora16_temp5", "ens_lbB_partial4_sf",
           "ens_lbB_temp2", "ens_lbB_temp5_slim"]
SHORT = {"ens_lbB_lora16_sf": "LoRA-sf", "ens_lbB_lora16_temp5": "LoRA-T5",
         "ens_lbB_partial4_sf": "Part4", "ens_lbB_temp2": "T2",
         "ens_lbB_temp5_slim": "T5"}
CONDS = ["C1", "C2", "C3s1", "C3s2", "C3s3", "C4adj", "C4sign", "C4cf"]
# the ablation-panel JSON (scripts/ablation_subset_panel.py) keys members by the
# vetted short run id, the eval JSONs by the full ens_lbB_* entity name
PANEL_KEY = {m: m[len("ens_lbB_"):] for m in MEMBERS}

# ---------------------------------------------------------------------------
# 2b. ablation forest on a test-year subset (companion to the pooled forest of §2)
#
#     Same five members x eight conditions and the same style, but the frozen
#     test rows are restricted to a year subset — 2020-2021 together (the two
#     years nearest the 2000-2017 estimation window) or one year alone.
#     Numbers come from data/analysis/ablation_<years>_panel.json, written by
#     scripts/ablation_subset_panel.py (--years).  The pooled forest of §2 is
#     untouched.  This block is defined ahead of §1 only so that the standalone
#     entry below can draw one panel without regenerating the others; §2 draws
#     the whole table in the normal all-figures run.
#         FIG_ONLY=ablation_forest_2020 python scripts/07_figures/make_all_figures_pub.py
#     (any stem of FOREST_SUBSETS, with or without the fig_ prefix)
# ---------------------------------------------------------------------------
ANALYSIS = ROOT / "data" / "analysis"
C_NS = "#9aa5ad"

# (panel JSON stem, figure stem) for the subset forests
FOREST_SUBSETS = [
    ("ablation_2020_2021_panel", "fig_ablation_forest_2020_2021"),
    ("ablation_2020_panel", "fig_ablation_forest_2020"),
    ("ablation_2021_panel", "fig_ablation_forest_2021"),
]


def _text_layout_check(fig, ax, extra_texts=()):
    """Rendered text boxes: overlaps, clipping, plotting-area intrusion, row gap.

    Covers all four text classes in the forest panel: the 40 y row labels, the
    x tick labels, the x axis label and the two free annotations.  A text may
    not intersect any other text, may not be clipped by the canvas, and the row
    labels / annotations must stay outside the data rectangle.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def box(t):
        return t.get_window_extent(renderer)

    rows = [t for t in ax.get_yticklabels() if t.get_text().strip()]
    xlo, xhi = ax.get_xlim()
    xticks = [t.label1 for t in ax.xaxis.get_major_ticks()
              if t.label1.get_visible() and t.label1.get_text().strip()
              and xlo <= t.get_loc() <= xhi]
    notes = [t for t in extra_texts if t.get_text().strip()]
    texts = rows + xticks + [ax.xaxis.label] + notes
    overlaps = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = box(texts[i]), box(texts[j])
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5
                    and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5):
                overlaps.append([texts[i].get_text(), texts[j].get_text()])
    canvas = fig.bbox
    clipped = [t.get_text() for t in texts
               if (box(t).x0 < canvas.x0 - 0.5 or box(t).x1 > canvas.x1 + 0.5
                   or box(t).y0 < canvas.y0 - 0.5 or box(t).y1 > canvas.y1 + 0.5)]
    ab = ax.bbox
    intruders = [t.get_text() for t in rows + notes
                 if (box(t).x1 > ab.x0 + 0.5 and box(t).x0 < ab.x1 - 0.5
                     and box(t).y0 < ab.y1 - 0.5 and box(t).y1 > ab.y0 + 0.5)]
    ys = sorted((box(t).y0, box(t).y1) for t in rows)
    gaps = [ys[i + 1][0] - ys[i][1] for i in range(len(ys) - 1)]
    min_gap = min(gaps) * 72.0 / fig.dpi if gaps else None
    min_xgap = None
    if len(xticks) > 1:
        xs = sorted((box(t).x0, box(t).x1) for t in xticks)
        min_xgap = min(xs[i + 1][0] - xs[i][1] for i in range(len(xs) - 1))
    return {"n_texts": len(texts), "n_row_labels": len(rows),
            "overlaps": overlaps, "clipped": clipped,
            "intrudes_plot_area": intruders,
            "min_vertical_gap_pt": min_gap,
            "min_xtick_gap_px": min_xgap,
            "passed": bool(not overlaps and not clipped and not intruders
                           and (min_gap is None or min_gap > 0)
                           and (min_xgap is None or min_xgap > 0))}


def fig_ablation_forest_subset(panel_stem, fig_stem):
    """Forest of ΔBSS_dev (M0 − ablation) on a test-year subset panel.

    ``panel_stem`` is a stem of ``FOREST_SUBSETS`` (JSON written by
    ``scripts/ablation_subset_panel.py``); the corner label, row count and
    storm count come from the panel's own ``subset`` block.
    """
    data = json.loads((ANALYSIS / f"{panel_stem}.json").read_text(encoding="utf-8"))
    panel, subset = data["panel"], data["subset"]
    fig, ax = plt.subplots(figsize=(120 * MM, 168 * MM))
    pos = 0.0
    ypos, ylabels = [], []
    for m in MEMBERS:
        for c in CONDS:
            st = panel[PANEL_KEY[m]][c]
            sig = bool(st["reject_holm"])
            color = (C_OURS if st["delta"] > 0 else C_VERM) if sig else C_NS
            ax.errorbar(st["delta"], pos,
                        xerr=[[st["delta"] - st["ci_lo"]],
                              [st["ci_hi"] - st["delta"]]],
                        fmt="o", ms=3, capsize=1.6, lw=0.8, color=color)
            ypos.append(pos)
            ylabels.append(f"{SHORT[m]}  {c}")
            pos += 1.0
        pos += 0.9                      # blank row between member blocks
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_yticks(ypos, ylabels, fontsize=6.3)
    ax.set_xlabel("ΔBSS (M0 − ablation; >0 = causal prior helps), 90% CI",
                  fontsize=8.5)
    lo = min(panel[PANEL_KEY[m]][c]["ci_lo"] for m in MEMBERS for c in CONDS)
    hi = max(panel[PANEL_KEY[m]][c]["ci_hi"] for m in MEMBERS for c in CONDS)
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    note = ax.text(0.0, 1.006,
                   "blue/orange = Holm-adjusted significant; grey = n.s.",
                   transform=ax.transAxes, fontsize=8, va="bottom",
                   color="#333333")
    sub = fig.text(0.5, 0.006,
                   f"{subset['label']} \u00b7 n = {subset['n_rows']:,} rows / "
                   f"{subset['n_storms']} storms \u00b7 Holm \u03b1 = 0.10",
                   ha="center", va="bottom", fontsize=7.2, color="#333333")
    fig.tight_layout(rect=(0, 0.030, 1, 0.972))
    check = _text_layout_check(fig, ax, (note, sub))
    print(f"layout self-check ({fig_stem}): "
          f"texts={check['n_texts']} (rows={check['n_row_labels']}) "
          f"overlaps={len(check['overlaps'])} clipped={len(check['clipped'])} "
          f"in-plot={len(check['intrudes_plot_area'])} "
          f"min row gap={check['min_vertical_gap_pt']:.1f} pt -> "
          f"{'PASS' if check['passed'] else 'FAIL'}")
    for pair in check["overlaps"]:
        print("  overlap:", pair)
    for txt in check["clipped"] + check["intrudes_plot_area"]:
        print("  misplaced:", txt)
    _save(fig, fig_stem)
    return fig


def fig_ablation_forest_2020_2021():
    """Combined 2020-2021 subset (kept as a named entry point)."""
    return fig_ablation_forest_subset(*FOREST_SUBSETS[0])


ONLY = os.environ.get("FIG_ONLY", "").strip()
if ONLY:
    ONLY = ONLY if ONLY.startswith("fig_") else f"fig_{ONLY}"
    stems = [stem for _, stem in FOREST_SUBSETS]
    if ONLY not in stems:
        raise SystemExit(f"unknown FIG_ONLY={ONLY!r} "
                         f"(available: {', '.join(stems)})")
    plt.close(fig_ablation_forest_subset(*FOREST_SUBSETS[stems.index(ONLY)]))
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 1. main overview 2x3 (official 3,004 rows)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 128 * MM))
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "legend.fontsize": 8})
ax = axes[0, 0]
for label, name, c, lw in MODELS:
    p, y, _ = load_test(name)
    pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, color=c, lw=lw, label=label)
p, y, _ = load_test("main")
ax.axhline(y.mean(), color="k", ls=":", lw=0.6)
ax.text(0.35, y.mean() + 0.015, f"base rate {y.mean():.3f}", fontsize=8,
        color="#444444")
ax.set_xlabel("Recall (POD)")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
panel_label(ax, "a")

ax = axes[0, 1]
ax.plot([0, 1], [0, 1], "k--", lw=0.6)
for label, name, c, lw in MODELS:
    p, y, _ = load_test(name)
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=c, lw=lw, label=label)
ax.set_xlabel("False-detection rate (POFD)")
ax.set_ylabel("Probability of detection")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

panel_label(ax, "b")

ax = axes[1, 0]
ax.plot([0, 1], [0, 1], "k--", lw=0.6)
for label, name, c, lw in MODELS:
    p, y, _ = load_test(name)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 20:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    ax.plot(xs, ys, "o-", color=c, ms=3.2, lw=lw, label=label)
ax.set_xlabel("Forecast probability")
ax.set_ylabel("Observed frequency")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
panel_label(ax, "d")

ax = axes[1, 1]
p, y, _ = load_test("main")
bins = np.linspace(0, 1, 40)
ax.hist(p[y == 0], bins=bins, density=True, color=C_GREY, alpha=0.75,
        label="non-RI")
ax.hist(p[y == 1], bins=bins, density=True, color=C_VERM, alpha=0.75,
        label="RI")
ax.axvline(0.225, color="k", ls="--", lw=0.7)
ax.set_xlabel("CVRI-Ens predicted RI probability")
ax.set_ylabel("density")
ax.legend(frameon=False, loc="upper right", fontsize=8)
panel_label(ax, "e")

ax = axes[0, 2]
r = json.load(open(EV / "main_2024.json"))
years = list(range(2020, 2025))
bss = [r["metrics"][f"year{y}"]["BSS_dev"] for y in years]
csi = [r["metrics"][f"year{y}"]["CSI"] for y in years]
x = np.arange(5)
ax.bar(x - 0.19, bss, 0.38, color=C_OURS, label="BSS (dev clim.)")
ax.bar(x + 0.19, csi, 0.38, color=C_ORANGE, label="CSI")
ax.set_xticks(x, [str(y) for y in years])
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_ylim(0, 0.52)
ax.set_ylabel("score")
panel_label(ax, "c")

ax = axes[1, 2]
names_b = ["baseline_lr", "baseline_lda", "baseline_ships12_lda",
           "baseline_ships12_lda_cohc"]
labels_b = ["LR", "LDA", "SHIPS OHCD", "SHIPS COHC"]
bvals = [json.load(open(EV / f"{n}.json"))["metrics"]["all"]["BSS_dev"]
         for n in names_b]
bvals.append(json.load(open(EV / "main_2024.json"))["metrics"]["all"]
             ["BSS_dev"])
labels_b.append("CVRI-Ens")
cols_b = [C_GREY, C_ORANGE, C_SKY, C_BG, C_OURS]
ypos = np.arange(len(labels_b))
clip = -0.35
shown = [max(v, clip) for v in bvals]
ax.barh(ypos, shown, color=cols_b, height=0.62)
for i, v in enumerate(bvals):
    if v < clip:
        ax.annotate(f"off scale ({v:.1f})", (clip + 0.01, i), fontsize=7.5,
                    va="center", color="#444444")
ax.set_yticks(ypos, labels_b, fontsize=8.5)
ax.set_xlabel("BSS (dev clim.), test 2020–2024")
ax.set_xlim(clip, 0.32)
ax.axvline(0, color="k", lw=0.6)
panel_label(ax, "f")
handles = [plt.Line2D([], [], color=c, lw=lw, label=label)
           for label, _, c, lw in MODELS]
fig.legend(handles=handles, frameon=False, loc="lower center", ncol=5,
           fontsize=8.5, bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(rect=(0, 0.10, 1, 1))
_save(fig, "fig_main_overview")
plt.close(fig)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5})

# ---------------------------------------------------------------------------
# 2. ablation forest (restyled)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(120 * MM, 126 * MM))
pos = 0
ylabels, ypos = [], []
for m in MEMBERS:
    r = json.load(open(EV / f"M0_{m}.json"))
    for c in CONDS:
        st = r["paired_bootstrap"].get(f"abl_{c}_{m}")
        if st is None:
            continue
        sig = st.get("reject_holm", False)
        color = (C_OURS if st["delta"] > 0 else C_VERM) if sig else "#9aa5ad"
        ax.errorbar(st["delta"], pos,
                    xerr=[[st["delta"] - st["ci_lo"]],
                          [st["ci_hi"] - st["delta"]]],
                    fmt="o", ms=3, capsize=1.6, lw=0.8, color=color)
        ypos.append(pos)
        ylabels.append(f"{SHORT[m]}  {c}")
        pos += 1
    pos += 1.5
ax.axvline(0, color="k", lw=0.7, ls="--")
ax.set_yticks(ypos, ylabels, fontsize=6.5)
ax.set_xlabel("ΔBSS (M0 − ablation; >0 = causal prior helps), 90% CI", fontsize=8.5)
ax.text(0.02, 0.985, "blue/orange = Holm-adjusted significant; grey = n.s.",
        transform=ax.transAxes, fontsize=8, va="top", color="#333333")
fig.tight_layout()
_save(fig, "fig_ablation_forest")
plt.close(fig)

# 2b. same panel on the year subsets (2020-2021, 2020, 2021); the function is
# defined near the top so FIG_ONLY can draw one of them on its own
for _panel_stem, _fig_stem in FOREST_SUBSETS:
    plt.close(fig_ablation_forest_subset(_panel_stem, _fig_stem))

# ---------------------------------------------------------------------------
# 3. gate members
# ---------------------------------------------------------------------------
g = json.load(open(ROOT / "data/analysis/gate_participation.json"))
names = list(g["entities"]["main"]["members"].keys())
short = [SHORT[m] for m in names]
gm = [g["entities"]["main"]["members"][m]["gate_mean"]["mean"] for m in names]
cs = [g["entities"]["main"]["members"][m]["causal_attn_share"]["mean"]
      for m in names]
fig, axes = plt.subplots(1, 2, figsize=(120 * MM, 52 * MM))
axes[0].bar(range(len(names)), gm, color=C_OURS)
axes[0].set_xticks(range(len(names)), short, rotation=20, ha="right",
                   fontsize=6)
axes[0].set_ylabel("gate mean")
panel_label(axes[0], "a")
axes[1].bar(range(len(names)), cs, color=C_ORANGE)
axes[1].axhline(0.5, color="k", ls="--", lw=0.6)
axes[1].set_xticks(range(len(names)), short, rotation=20, ha="right",
                   fontsize=6)
axes[1].set_ylabel("causal attention share")
axes[1].text(0.98, 0.52, "no-preference line 0.5", transform=axes[1].transAxes,
             fontsize=6, ha="right", color="#444444")
panel_label(axes[1], "b")
fig.tight_layout()
_save(fig, "fig_gate_members")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. selection landscape (22 arms)
# ---------------------------------------------------------------------------
sel = pd.read_csv(ROOT / "protocol_locks" / "selection.csv").sort_values(
    "two_year_worst", ascending=False)
fig, ax = plt.subplots(figsize=(183 * MM, 60 * MM))


def origin(row) -> str:
    if row["family"].startswith("lingbot"):
        return "LingBot backbone"
    if row["arm"].startswith("ens_ep40"):
        return "MAE global"
    return "MAE WNP"


sel["origin"] = sel.apply(origin, axis=1)
sel["temporal"] = sel["family"].str.endswith("_mf")
origin_color = {"LingBot backbone": C_BG, "MAE global": C_ORANGE,
                "MAE WNP": C_PURP}
top5 = set(sel.nsmallest(5, "rank")["arm"] if "rank" in sel else
           sel.head(5)["arm"])
for i, (_, row) in enumerate(sel.iterrows()):
    is_top5 = row["arm"] in top5
    ax.bar(i, row["two_year_worst"], color=origin_color[row["origin"]],
           hatch="//" if row["temporal"] else "",
           edgecolor="k" if is_top5 else ("#333333" if row["temporal"] else "none"),
           lw=1.4 if is_top5 else 0)
ax.set_ylabel("inner BSS (two-year worst)")
ax.set_xlabel("22 candidate arms (ranked)")
handles = [plt.Line2D([], [], marker="s", color=c, ls="", label=o)
           for o, c in origin_color.items()]
handles.append(plt.Line2D([], [], marker="s", color="w", ls="", mec="k",
                          mew=1.4, label="selected Top-5 (framed)"))
handles.append(Patch(facecolor="w", edgecolor="#333333", hatch="//",
                     lw=0, label="multi-frame (hatched)"))
ax.legend(handles=handles, frameon=False, fontsize=6, ncol=5,
          loc="upper right")
fig.tight_layout()
_save(fig, "fig_selection_landscape")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. member vs ensemble gain
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(89 * MM, 66 * MM))
names = [f"M0_{m}" for m in MEMBERS]
m0_bss = [json.load(open(EV / f"{n}.json"))["metrics"]["all"]["BSS_dev"]
          for n in names]
main_bss = json.load(open(EV / "main_2024.json"))["metrics"]["all"]["BSS_dev"]
x = np.arange(len(MEMBERS) + 1)
vals = m0_bss + [main_bss]
labels = [SHORT[m] for m in MEMBERS] + ["Top-5\nensemble"]
colors = [C_GREY] * 5 + [C_OURS]
ax.bar(x, vals, color=colors)
ax.axhline(max(m0_bss), color="k", ls=":", lw=0.7)
ax.text(4.6, max(m0_bss) + 0.006, "best single member",
        fontsize=6, ha="right", color="#444444")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6, rotation=25, ha="right")
ax.set_ylabel("BSS (test 2020–2024)")
fig.tight_layout()
_save(fig, "fig_member_ensemble_gain")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. score distributions (ours)
# ---------------------------------------------------------------------------
p, y, _ = load_test("main")
fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
bins = np.linspace(0, 1, 40)
ax.hist(p[y == 0], bins=bins, density=True, color=C_GREY, alpha=0.75,
        label="non-RI")
ax.hist(p[y == 1], bins=bins, density=True, color=C_VERM, alpha=0.75,
        label="RI")
ax.axvline(0.225, color="k", ls="--", lw=0.7)
ax.text(0.235, ax.get_ylim()[1] * 0.9, "inner-F1 threshold 0.225",
        fontsize=6, color="#444444")
ax.set_xlabel("predicted RI probability")
ax.set_ylabel("density")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
_save(fig, "fig_score_distributions")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. CMA-30 kt label-control operating points (Fig A rows, CMA labels)
# ---------------------------------------------------------------------------
tA = pd.read_csv(CMP / f"table_figA{os.environ.get('CMP_TABLE_SUFFIX', '_best_auc')}.csv")
sub = tA[tA.figure == "A_cma30"]
fig, ax = plt.subplots(figsize=(100 * MM, 92 * MM))
STYLE = {"ours": (C_OURS, "*", "CVRI-Ens"),
         "baseline_lr": (C_GREY, "o", "LR"),
         "baseline_lda": (C_ORANGE, "o", "LDA"),
         "baseline_ships12_lda": (C_SKY, "o", "SHIPS OHCD"),
         "baseline_ships12_lda_cohc": (C_BG, "o", "SHIPS COHC"),
         "wrps1_f1": (C_WRPS1, "s", "WRPS1-F1"),
         "wrps1_own": (C_WRPS1, "s", "WRPS1"),
         "ritcf_V1": (C_RITCF, "D", "RITCF V1"),
         "ritcf_V2": (C_V2, "D", "RITCF V2"),
         "ritcf_V3": (C_V3, "D", "RITCF V3"),
         "ritcf_V4direct": (C_V4D, "v", "RITCF direct")}
for _, r in sub.iterrows():
    st = STYLE.get(r["model"])
    if st is None:
        continue
    color, marker, lbl = st
    fill = "val-F1" not in r["rule"]
    ax.scatter(r["FARate"] * 100, r["POD"] * 100, marker=marker,
               s=95 if marker == "*" else 42, edgecolors=color,
               facecolors=color if fill else "none",
               linewidths=0.9 if fill else 1.2, zorder=3)
ax.set_xlabel("False-detection rate, POFD (%)")
ax.set_ylabel("POD (%)")
ax.set_xlim(0, 70)
ax.set_ylim(0, 103)
ax.grid(alpha=0.22, lw=0.35)
fill_styles = {}
for _, r in sub.iterrows():
    if r["model"] in STYLE:
        fill_styles.setdefault(STYLE[r["model"]][2], set()).add(
            "val-F1" not in r["rule"])
hh = []
for c, m, l in _ordered_unique(STYLE.values()):
    for fill in (True, False):
        if fill not in fill_styles[l]:
            continue
        suffix = f" ({'6/10' if fill else 'val-F1'})" if len(fill_styles[l]) > 1 else ""
        hh.append(plt.Line2D([], [], marker=m, color=c, ls="",
                             ms=12 if m == "*" else 6, label=l + suffix,
                             mfc=c if fill else "none"))
assert len(hh) == sum(len(styles) for styles in fill_styles.values())
fig.legend(handles=hh, frameon=False, fontsize=6.2, loc="lower center",
           ncol=4, bbox_to_anchor=(0.5, 0.005), handletextpad=0.4,
           columnspacing=0.9)
fig.tight_layout(rect=(0, 0.17, 1, 1))
_save(fig, "fig_cma_label_control")
plt.close(fig)

print("ALL FIGURES DONE")
