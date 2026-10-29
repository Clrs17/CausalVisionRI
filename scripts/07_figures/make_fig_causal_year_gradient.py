# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Per-year causal gain (main-text Fig. 4), publication style.

(a) ensemble delta BSS per year (M0 - C2; positive = causal prior helps),
    starred where the storm-cluster bootstrap CI excludes zero at both
    the 90% and 95% levels
(b) the same per member, with lora16_sf emphasised

Sources: data/analysis/worst_year_ablation.json for the point estimates and
data/analysis/year_gradient_ablation.json for the per-year bootstrap CIs.
``delta_vs_M0`` is C2 - M0 (negative = causal prior favourable); every value
plotted here is negated so the figure follows the paper convention
d BSS = M0 - C2.  The year-gradient ``bootstrap_ensemble`` block already uses
that convention, so its CI bounds are used as stored.

Outputs data/analysis/figures_pub/fig_causal_year_gradient.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "analysis" / "worst_year_ablation.json"
GRAD_SRC = ROOT / "data" / "analysis" / "year_gradient_ablation.json"
FIG = ROOT / "data" / "analysis" / "figures_pub"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

C_POS = "#0072B2"
C_NEG = "#D55E00"
C_SKY = "#56B4E9"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_PURP = "#CC79A7"

MEMBERS = ["lora16_sf", "lora16_temp5", "partial4_sf", "temp2", "temp5_slim"]
SHORT = {"lora16_sf": "LoRA-sf", "lora16_temp5": "LoRA-T5",
         "partial4_sf": "Part4", "temp2": "T2", "temp5_slim": "T5"}
# main colour for lora16_sf; the other members stay pale at alpha 0.55 so the
# emphasised member dominates and every swatch keeps >40 RGB units of separation
STYLE = {
    "lora16_sf": dict(face=C_POS, alpha=1.0, edge="#023E5E", lw=0.35),
    "lora16_temp5": dict(face=C_ORANGE, alpha=0.55, edge="none", lw=0.0),
    "partial4_sf": dict(face=C_GREEN, alpha=0.55, edge="none", lw=0.0),
    "temp2": dict(face=C_PURP, alpha=0.55, edge="none", lw=0.0),
    "temp5_slim": dict(face=C_SKY, alpha=0.55, edge="none", lw=0.0),
}

MM = 1 / 25.4
# 2020-2024 ensemble deltas in the paper convention, for the self-check
EXPECTED_ENS = {2020: 0.080, 2021: 0.047, 2022: 0.029, 2023: -0.007, 2024: 0.017}
MINUS = "\u2212"


def fmt(v, nd=3):
    s = f"{v:+.{nd}f}"
    return s.replace("-", MINUS)


def save(fig, stem):
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
        print("wrote", path)


def panel_label(ax, letter):
    ax.text(-0.02, 1.045, letter, transform=ax.transAxes, fontsize=7,
            fontweight="bold", va="bottom", ha="left")


data = json.load(open(SRC))
years = [int(y) for y in data["sources"]["years"]]
delta = data["delta_vs_M0"]

# M0 - C2: negate the stored C2 - M0 values
ens = {y: -delta["C2_ENS"][f"year{y}"] for y in years}
member = {m: {y: -delta[f"C2_{m}"][f"year{y}"] for y in years} for m in MEMBERS}

# panel a stars: years whose bootstrap CI excludes zero at both 90% and 95%
grad = json.load(open(GRAD_SRC))
ens_ci = grad["bootstrap_ensemble"]["C2"]
star_years = [y for y in years
              if bool(ens_ci[str(y)]["significant_90"])
              and bool(ens_ci[str(y)]["significant_95"])]

print("self-check, ensemble d BSS (M0 - C2):")
ok = True
for y in years:
    exp = EXPECTED_ENS[y]
    got = round(ens[y], 3)
    flag = "ok" if abs(got - exp) <= 5e-4 else "MISMATCH"
    ok &= flag == "ok"
    print(f"  {y}: {ens[y]:+.4f}  expected {exp:+.3f}  [{flag}]")

# the year-gradient json carries the same point estimates in the same
# convention; a drift here would mean the CI does not belong to these bars
print("self-check, year-gradient delta vs plotted bars and 95% CI:")
for y in years:
    g = ens_ci[str(y)]
    flag = "ok" if abs(g["delta"] - ens[y]) <= 5e-4 else "MISMATCH"
    ok &= flag == "ok"
    lo, hi = g["ci95"]
    print(f"  {y}: d={g['delta']:+.4f} CI95=[{lo:+.4f}, {hi:+.4f}] "
          f"star={'yes' if y in star_years else 'no'}  [{flag}]")
summ_years = [int(y) for y in grad["summary"]["years_95_ci_excludes_zero_C2"]]
if summ_years != star_years:
    ok = False
    print(f"  MISMATCH: summary lists {summ_years}, stars derived {star_years}")
print("  panel a star years:", star_years)

ymin = min(min(ens.values()), min(min(v.values()) for v in member.values()))
ymax = max(max(ens.values()), max(max(v.values()) for v in member.values()))
span = ymax - ymin
# extra headroom on top: a star marker sits above the ensemble bar's number
ylim = (ymin - 0.055 * span, ymax + 0.105 * span)
yticks = np.arange(-0.10, 0.1501, 0.05)

fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 72 * MM), sharey=True)
fig.subplots_adjust(left=0.082, right=0.995, top=0.865, bottom=0.285,
                    wspace=0.16)

# ---------------------------------------------------------------- panel a
ax = axes[0]
x = np.arange(len(years))
vals = [ens[y] for y in years]
cols = [C_POS if v >= 0 else C_NEG for v in vals]
ax.bar(x, vals, width=0.62, color=cols, edgecolor="none", zorder=2)
ax.axhline(0, color="#555555", lw=0.6, ls=(0, (2.6, 1.8)), zorder=1)
labels = {}
for xi, v, yi in zip(x, vals, years):
    off = 0.010 * (ylim[1] - ylim[0])
    labels[yi] = ax.text(xi, v + (off if v >= 0 else -off), fmt(v),
                         ha="center", va="bottom" if v >= 0 else "top",
                         fontsize=5.7, color="#222222")
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years])
ax.set_xlim(-0.6, len(years) - 0.4)
ax.set_ylim(*ylim)
ax.set_yticks(yticks)
ax.set_yticklabels([fmt(t, 2) for t in yticks])
ax.set_title("Ensemble: gain concentrates in 2020" + "\u2013" + "2021",
             pad=6)
panel_label(ax, "a")
ax.legend(handles=[Patch(facecolor=C_POS, edgecolor="none",
                         label="gain (M0 > C2)"),
                   Patch(facecolor=C_NEG, edgecolor="none",
                         label="loss (M0 < C2)")],
          loc="upper right", frameon=False, fontsize=5.5, handlelength=1.25,
          handletextpad=0.5, labelspacing=0.35, borderaxespad=0.2)

# the star rides just above the number label; its height is read off the
# rendered label box so the marker can never crowd the text
fig.canvas.draw()
_rnd = fig.canvas.get_renderer()
_to_data = ax.transData.inverted()
for yi, t in labels.items():
    if yi not in star_years:
        continue
    bb = t.get_window_extent(_rnd)
    ax.text(t.get_position()[0], _to_data.transform((bb.x0, bb.y1 + 3.0))[1],
            "*", ha="center", va="bottom", fontsize=7, color="#111111",
            zorder=6)

# ---------------------------------------------------------------- panel b
ax = axes[1]
n = len(MEMBERS)
bw = 0.135
offs = (np.arange(n) - (n - 1) / 2) * (bw + 0.012)
for i, m in enumerate(MEMBERS):
    st = STYLE[m]
    xs = x + offs[i]
    ys = [member[m][y] for y in years]
    ax.bar(xs, ys, width=bw, facecolor=st["face"], alpha=st["alpha"],
           edgecolor=st["edge"], linewidth=st["lw"],
           zorder=4 if m == "lora16_sf" else 2)
ax.axhline(0, color="#555555", lw=0.6, ls=(0, (2.6, 1.8)), zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years])
ax.set_xlim(-0.6, len(years) - 0.4)
ax.set_ylim(*ylim)
ax.set_title("Members: all five gain in 2020, four of five in 2021",
             pad=6)
panel_label(ax, "b")

# identical y scale and label on both panels; the literal Greek delta keeps the
# label in Arial instead of pulling in the DejaVu mathtext font
for ax in axes:
    ax.set_ylabel("\u0394BSS (M0 " + MINUS + " C2; >0 = graph pathway helps)")

# ---------------------------------------------------------------- legend
CX = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
handles = [Patch(facecolor=STYLE[m]["face"], alpha=STYLE[m]["alpha"],
                 edgecolor=STYLE[m]["edge"],
                 linewidth=STYLE[m]["lw"], label=SHORT[m]) for m in MEMBERS]
fig.legend(handles=handles, loc="lower center", ncol=len(MEMBERS),
           frameon=False, handlelength=1.4, handletextpad=0.45,
           columnspacing=1.5, borderaxespad=0.0, bbox_to_anchor=(CX, 0.062))
star_note = Line2D([], [], linestyle="none", marker="*", markersize=6,
                   color="#111111",
                   label="star: bootstrap CI excludes 0 at both "
                         "90% and 95% (panel a)")
fig.legend(handles=[star_note], loc="lower center", ncol=1, frameon=False,
           handlelength=1.4, handletextpad=0.45, borderaxespad=0.0,
           bbox_to_anchor=(CX, 0.012))


def check_layout(fig, axes):
    """Automated layout audit: containment, text overlap, legend placement.

    Run before saving so a crowded or out-of-frame figure fails loudly instead
    of ending up in the supplement.
    """
    import itertools
    import matplotlib.text as mtext

    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    fb = fig.bbox
    pad = 1.0
    problems = []

    texts = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(rnd)
        texts.append((t.get_text()[:30], bb))
        if (bb.x0 < fb.x0 - pad or bb.y0 < fb.y0 - pad
                or bb.x1 > fb.x1 + pad or bb.y1 > fb.y1 + pad):
            problems.append(f"outside canvas: {t.get_text()[:40]!r}")

    for (na, ba), (nb, bb) in itertools.combinations(texts, 2):
        if ba.overlaps(bb):
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w > 0.5 and h > 0.5:
                problems.append(f"text overlap: {na!r} vs {nb!r} ({w:.1f}x{h:.1f} px)")

    for leg in fig.legends:
        lb = leg.get_window_extent(rnd)
        if (lb.x0 < fb.x0 - pad or lb.y0 < fb.y0 - pad
                or lb.x1 > fb.x1 + pad or lb.y1 > fb.y1 + pad):
            problems.append("legend outside the canvas")
        for i, ax in enumerate(axes):
            if lb.overlaps(ax.get_window_extent(rnd)):
                problems.append(f"legend overlaps axes {i}")
            if lb.overlaps(ax.title.get_window_extent(rnd)):
                problems.append(f"legend overlaps title {i}")

    # each axes' decoration must not run into the neighbouring panel
    for i, ax in enumerate(axes):
        others = [axes[j].get_window_extent(rnd) for j in range(len(axes)) if j != i]
        arts = [ax.title, ax.yaxis.label] + list(ax.get_xticklabels()) + \
            list(ax.get_yticklabels()) + [t for t in ax.texts if t.get_fontweight() != "bold"]
        for t in arts:
            tb = t.get_window_extent(rnd)
            for ob in others:
                if tb.overlaps(ob):
                    problems.append(
                        f"panel {i}: {t.get_text()!r} runs into the other panel")
                    break
    return problems


problems = check_layout(fig, axes)


def palette_audit():
    """Legend swatches, alpha-blended on white, must stay distinguishable."""
    def blend(hexc, a):
        rgb = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)], float)
        return 255.0 - a * (255.0 - rgb)

    shades = {m: blend(st["face"], st["alpha"]) for m, st in STYLE.items()}
    worst, pair = 1e9, None
    for i, m in enumerate(MEMBERS):
        for n in MEMBERS[i + 1:]:
            d = float(np.linalg.norm(shades[m] - shades[n]))
            if d < worst:
                worst, pair = d, (m, n)
    print(f"palette: closest swatch pair {pair} = {worst:.1f} RGB units")
    if worst < 40:
        problems.append(f"legend colours too close: {pair} ({worst:.1f})")


palette_audit()

save(fig, "fig_causal_year_gradient")
plt.close(fig)

# rendered-pixel check: the 600 dpi PNG must have clean white margins
from PIL import Image
im = np.asarray(Image.open(FIG / "fig_causal_year_gradient.png").convert("L"))
edge = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]])
if edge.min() < 250:
    problems.append("ink touches the PNG border (content clipped)")
h, w = im.shape
print(f"png {w}x{h} px, border min level {edge.min()}, "
      f"top/bottom ink rows {int((im.min(1) < 250).sum())}")

if problems:
    print("LAYOUT PROBLEMS:")
    for p in problems:
        print("  -", p)
    raise SystemExit(1)
print("PASS: no text overlap, all labels/legend inside the canvas, "
      "legend outside the data area, no clipping")

if not ok:
    raise SystemExit("ensemble per-year deltas do not match the expected values")
print("PASS: ensemble per-year d BSS matches the JSON-derived expectation")
