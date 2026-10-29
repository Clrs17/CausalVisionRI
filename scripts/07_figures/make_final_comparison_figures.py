# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Final comparison figures — figure-publication conventions.

Style applied: Arial, 5-7 pt text, bold lowercase panel letters (a, b),
183 mm double-column width, 600 dpi PNG + vector PDF, RGB, colorblind-safe
Okabe-Ito palette, no embedded figure titles, legend kept outside data areas.

Fig 1 (points only): fig_operating_points.{png,pdf} — operating points per
model, panels a (our standard: USA-30 kt, 2020-2024, n=2,178) and b (their
standard: CMA-25.27 kt, 2020-2021, n=949; paper claim as hollow anchor).
Fig 2 (curves only): fig_operating_envelopes.{png,pdf} — score-sweep
envelopes on the Fig-A sample (USA-30 kt labels), panels a (POD-FARate) and
b (POD-FARatio).

Data: data/analysis/compare/table_figA.csv / table_figB.csv + recon per-row
score dumps + our/baseline npz + WRPS1 score dump.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Affine2D, Bbox
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CMP = ROOT / "data" / "analysis" / "compare"
TD = ROOT / "data" / "test" / "holdout_masked_v1"
RR = ROOT / "data" / "ritcf_recon"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,          # editable TrueType in PDF
    "svg.fonttype": "none",
})

# Okabe-Ito colorblind-safe assignments
C_OURS = "#0072B2"     # blue
C_LR = "#999999"       # gray
C_LDA = "#E69F00"      # orange
C_OHCD = "#56B4E9"     # sky blue
C_COHC = "#009E73"     # bluish green
C_WRPS1 = "#009E73"    # bluish green (WRPS1 squares)
C_RITCF = "#D55E00"    # vermillion
C_V2 = "#E69F00"       # orange
C_V3 = "#CC79A7"       # reddish purple
C_V4D = "#4B2E2A"      # dark brown
C_CLAIM = "#000000"

_TABLE_SUFFIX = os.environ.get("CMP_TABLE_SUFFIX", "")
_RECON_TAG_KIND = os.environ.get("RECON_TAG_KIND", "")
tA = pd.read_csv(CMP / f"table_figA{_TABLE_SUFFIX}.csv")
tB = pd.read_csv(CMP / f"table_figB{_TABLE_SUFFIX}.csv")


def get(t, model, rule, figure=None):
    sub = t[(t.model == model) & (t.rule == rule)]
    if figure is not None and "figure" in t.columns:
        sub = sub[sub.figure == figure]
    return sub.iloc[0] if len(sub) else None


def _save(fig, path):
    import os
    import tempfile
    import time
    path = Path(path)
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    try:
        fig.savefig(tmp, dpi=600 if path.suffix == ".png" else None)
        for _ in range(20):
            try:
                os.replace(tmp, path)
                break
            except OSError:
                time.sleep(1.5)
        else:
            raise SystemExit(f"could not replace {path}")
        print("wrote", path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_all(fig, stem):
    _save(fig, CMP / f"{stem}.png")
    _save(fig, CMP / f"{stem}.pdf")


def panel_label(ax, letter):
    return ax.text(-0.16, 1.02, letter, transform=ax.transAxes, fontsize=8,
                   fontweight="bold", va="bottom", ha="left")


def _pt(ax, r, color, marker, fill=True):
    size = {"*": 80, "X": 95}.get(marker, 42)
    return ax.scatter(r["FARate"] * 100, r["POD"] * 100, marker=marker,
                      s=size, edgecolors=color, zorder=3,
                      facecolors=color if fill else "none",
                      linewidths=0.9 if fill else 1.2)


LABEL_POSITIONS = {
    "A": {"CVRI-Ens": (3, 81), "LR": (15, 73), "LDA": (26, 82),
          "SHIPS OHCD": (13, 50), "SHIPS COHC": (12, 28),
          "WRPS1": (1, 20), "WRPS1-F1": (1, 54),
          "RITCF V1": (47, 80), "RITCF V2": (56, 87), "RITCF V3": (25, 58),
          "RITCF direct": (11, 12)},
    "B": {"CVRI-Ens": (1, 74), "LR": (20, 69), "LDA": (26, 76),
          "SHIPS OHCD": (1, 79), "SHIPS COHC": (10, 49),
          "WRPS1": (1, 21), "WRPS1-F1": (11, 62),
          "RITCF V1": (47, 82), "RITCF V2": (57, 87), "RITCF V3": (36, 76),
          "RITCF direct": (2, 15)},
}


def _padded(bb, pad=1.5):
    return Bbox([[bb.x0 - pad, bb.y0 - pad], [bb.x1 + pad, bb.y1 + pad]])


def _segment_hits_box(p0, p1, bb):
    t0, t1 = 0.0, 1.0
    for p, d, lo, hi in ((p0[0], p1[0] - p0[0], bb.x0, bb.x1),
                         (p0[1], p1[1] - p0[1], bb.y0, bb.y1)):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return False
        else:
            a, b = (lo - p) / d, (hi - p) / d
            if a > b:
                a, b = b, a
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return False
    return t1 >= 0 and t0 <= 1


def _point_segment_distance(p, a, b):
    d = b - a
    denom = float(np.dot(d, d))
    t = 0.0 if denom == 0 else np.clip(np.dot(p - a, d) / denom, 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))


def _segments_close(a, b, c, d, tol=0.75):
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b)) < tol


def _leader_segments(ann):
    path = ann.arrow_patch.get_path().transformed(ann.arrow_patch.get_transform())
    return list(zip(path.vertices[:-1], path.vertices[1:]))


def _marker_paths(ax, obstacle):
    if not hasattr(obstacle, "get_offsets"):
        return []
    points = ax.transData.transform(np.asarray(obstacle.get_offsets()))
    paths = obstacle.get_paths()
    transforms = obstacle.get_transforms()
    out = []
    for i, (x, y) in enumerate(points):
        path = paths[min(i, len(paths) - 1)]
        raw = transforms[min(i, len(transforms) - 1)] if len(transforms) else None
        transform = (Affine2D(raw.get_matrix() if hasattr(raw, "get_matrix")
                              else np.asarray(raw)) if raw is not None else Affine2D())
        out.append(path.transformed(transform.translate(x, y)))
    return out


def _obstacle_bbox(ax, obstacle):
    if not hasattr(obstacle, "get_offsets"):
        return obstacle.get_window_extent(ax.figure.canvas.get_renderer())
    boxes = [path.get_extents() for path in _marker_paths(ax, obstacle)]
    widths = np.asarray(obstacle.get_linewidth(), dtype=float)
    stroke = 0.5 * float(widths.max(initial=0.5)) * ax.figure.dpi / 72
    return _padded(Bbox.union(boxes), stroke)


def _label_clear(ax, ann, target, obstacles, placed):
    renderer = ax.figure.canvas.get_renderer()
    bb = Text.get_window_extent(ann, renderer)
    ab = ax.get_window_extent(renderer)
    if (bb.x0 < ab.x0 + 2 or bb.x1 > ab.x1 - 2 or
            bb.y0 < ab.y0 + 2 or bb.y1 > ab.y1 - 2):
        return None
    segments = _leader_segments(ann)
    leader_path = ann.arrow_patch.get_path().transformed(ann.arrow_patch.get_transform())
    target_display = ax.transData.transform(target)
    for obstacle in obstacles:
        own_target = False
        marker_paths = _marker_paths(ax, obstacle)
        if marker_paths:
            own_target = bool(np.allclose(
                ax.transData.transform(np.asarray(obstacle.get_offsets())[0]),
                target_display))
        ob = _padded(_obstacle_bbox(ax, obstacle))
        if bb.overlaps(ob):
            return None
        if own_target:
            continue
        if marker_paths:
            if any(path.intersects_path(leader_path, filled=True)
                   for path in marker_paths):
                return None
        elif any(_segment_hits_box(a, b, ob) for a, b in segments):
            return None
    for other_bb, other_segments in placed:
        other_bb = _padded(other_bb)
        if bb.overlaps(other_bb):
            return None
        if any(_segment_hits_box(a, b, other_bb) for a, b in segments):
            return None
        if any(_segment_hits_box(a, b, _padded(bb)) for a, b in other_segments):
            return None
        if any(_segments_close(a, b, c, d) for a, b in segments
               for c, d in other_segments):
            return None
    return bb, segments


def _place_labels(ax, items, obstacles=(), tag=""):
    """Place labels deterministically, checking rendered text and leader geometry."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    placed = []
    shifts = [(8, 0), (-8, 0), (0, 8), (0, -8), (14, 8), (-14, 8),
              (14, -8), (-14, -8), (24, 0), (-24, 0), (0, 18), (0, -18)]
    near = [(8, 8), (-8, 8), (8, -8), (-8, -8), (18, 0), (-18, 0),
            (0, 18), (0, -18), (30, 12), (-30, 12), (30, -12), (-30, -12),
            (44, 0), (-44, 0), (0, 36), (0, -36)]
    n_adjusted = 0
    for x, y, label, tx, ty in items:
        anchor = ax.transData.transform((tx, ty))
        point = ax.transData.transform((x, y))
        candidates = [(tx, ty)]
        candidates += [tuple(inv.transform(anchor + (dx, dy))) for dx, dy in shifts]
        candidates += [tuple(inv.transform(point + (dx, dy))) for dx, dy in near]
        grid = [(gx, gy) for gx in np.linspace(3, 67, 8)
                for gy in np.linspace(8, 96, 7)]
        candidates += sorted(grid, key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)
        seen = set()
        accepted = None
        for cx, cy in candidates:
            key = (round(cx, 5), round(cy, 5))
            if key in seen:
                continue
            seen.add(key)
            ann = ax.annotate(label, xy=(x, y), xytext=(cx, cy), fontsize=6.5,
                              ha="left", va="center", zorder=5,
                              arrowprops=dict(arrowstyle="-", lw=0.4, color="#999999",
                                              shrinkA=2, shrinkB=6))
            fig.canvas.draw()
            checked = _label_clear(ax, ann, (x, y), obstacles, placed)
            if checked is not None:
                accepted = ann, checked
                break
            ann.remove()
        if accepted is None:
            raise RuntimeError(f"{tag}/{label}: no collision-free label position")
        ann, checked = accepted
        n_adjusted += int(abs(ann.get_position()[0] - tx) > 1e-6 or
                          abs(ann.get_position()[1] - ty) > 1e-6)
        placed.append(checked)
    for i, (bb, segments) in enumerate(placed):
        ann = ax.texts[-len(placed) + i]
        if _label_clear(ax, ann, ann.xy, obstacles, placed[:i]) is None:
            raise RuntimeError(f"{tag}: rendered label validation failed at index {i}")
    print(f"label layout {tag}: {len(placed)} labels, {n_adjusted} adjusted; "
          "text/marker/leader checks PASS")


# ---------------------------------------------------------------------------
# Fig 1: operating points (points only)
# ---------------------------------------------------------------------------
def op_panel(ax, t, figure, with_claim=False):
    locked = ("native inner-F1 (locked 0.225)" if figure == "A"
              else "val-F1 re-derived under this label")
    # per-entry: (model, rule, short label, color, marker, fill)
    puts = [
        ("ours", locked, "CVRI-Ens", C_OURS, "*", True),
        ("baseline_lr", "native inner-F1 (locked)" if figure == "A" else locked,
         "LR", C_LR, "o", True),
        ("baseline_lda", "native inner-F1 (locked)" if figure == "A" else locked,
         "LDA", C_LDA, "o", True),
        ("baseline_ships12_lda",
         "native inner-F1 (locked)" if figure == "A" else locked,
         "SHIPS OHCD", C_OHCD, "o", True),
        ("baseline_ships12_lda_cohc",
         "native inner-F1 (locked)" if figure == "A" else locked,
         "SHIPS COHC", C_COHC, "o", True),
        ("wrps1_own", "public rule pu>=0.5 (== pl>=0.215)", "WRPS1", C_WRPS1,
         "s", True),
        ("wrps1_f1", "inner-F1 on P (2018-19)" if figure == "A"
         else "val-F1 on P (re-derived)", "WRPS1-F1", C_WRPS1, "s", False),
        ("ritcf_V1", "public rule 6/10 votes", "RITCF V1", C_RITCF, "D", True),
        ("ritcf_V2", "public rule 6/10 votes", "RITCF V2", C_RITCF, "D", True),
        ("ritcf_V3", "public rule 6/10 votes", "RITCF V3", C_RITCF, "D", True),
        ("ritcf_V4direct", "val-F1 optimum (their val sweep)", "RITCF direct",
         C_V4D, "v", True),
        ("ritcf_V1", "val-F1 optimum (their val sweep)", None, C_RITCF, "D",
         False),
        ("ritcf_V2", "val-F1 optimum (their val sweep)", None, C_RITCF, "D",
         False),
        ("ritcf_V3", "val-F1 optimum (their val sweep)", None, C_RITCF, "D",
         False),
    ]
    items, points = [], []
    for model, rule, short, color, marker, fill in puts:
        r = get(t, model, rule, figure)
        if r is None:
            print(f"  [miss] {figure}/{model}/{rule}")
            continue
        points.append(_pt(ax, r, color, marker, fill=fill))
        if short:
            items.append((r["FARate"] * 100, r["POD"] * 100, short,
                          *LABEL_POSITIONS[figure][short]))
    header = ax.text(0.97, 0.03,
                     ("our standard: USA-30 kt, 2020–2024\n"
                      "n = 2,178 shared records"
                      if figure == "A" else
                      "their standard: CMA-25.27 kt, 2020–2021\n"
                      "n = 949 shared records"),
                     transform=ax.transAxes, fontsize=6.5, va="bottom",
                     ha="right", color="#333333")
    obstacles = [header, *points]
    if with_claim:
        claim_pt = ax.scatter(9.4, 91.0, marker="X", s=120, facecolors="none",
                              edgecolors=C_CLAIM, linewidths=1.6, zorder=4)
        ctxt = ax.annotate("paper claim\n91.0 / 9.4 / 58.7\n(external, "
                           "n=1,149)", (9.4, 91.0), fontsize=6.5,
                           xytext=(14, 10), textcoords="offset points",
                           ha="left", va="center")
        obstacles.extend([claim_pt, ctxt])
    _pending = (items, obstacles)
    ax.set_xlabel("False-detection rate, POFD (%)")
    ax.set_ylabel("POD (%)")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 103)
    ax.grid(alpha=0.22, lw=0.35)
    return _pending


MM = 1 / 25.4
fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 78 * MM))
pend_a = op_panel(axes[0], tA, "A")
pend_b = op_panel(axes[1], tB, "B", with_claim=True)
handles = [
    plt.Line2D([], [], marker="*", color=C_OURS, ls="", ms=12,
               label="CVRI-Ens (ours)"),
    plt.Line2D([], [], marker="o", color=C_LR, ls="", ms=6,
               label="LR / LDA / SHIPS-12 (circle)"),
    plt.Line2D([], [], marker="s", color=C_WRPS1, ls="", ms=6,
               label="WRPS1 (public rule)"),
    plt.Line2D([], [], marker="s", color=C_WRPS1, ls="", ms=6, mfc="none",
               label="WRPS1-F1 (F1-derived)"),
    plt.Line2D([], [], marker="D", color=C_RITCF, ls="", ms=6,
               label="RITCF recon (6/10 rule)"),
    plt.Line2D([], [], marker="D", color=C_RITCF, ls="", ms=6, mfc="none",
               label="RITCF recon (val-F1 optimum)"),
    plt.Line2D([], [], marker="v", color=C_V4D, ls="", ms=6,
               label="RITCF direct (direct classifier)"),
    plt.Line2D([], [], marker="X", color=C_CLAIM, ls="", ms=9, mfc="none",
               label="RITCF paper claim"),
]
fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.14, 1, 1))
pa = panel_label(axes[0], "a")
pb = panel_label(axes[1], "b")
_place_labels(axes[0], pend_a[0], obstacles=pend_a[1] + [pa], tag="A")
_place_labels(axes[1], pend_b[0], obstacles=pend_b[1] + [pb], tag="B")
save_all(fig, "fig_operating_points")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2: operating envelopes (curves only, Fig-A sample / USA-30 kt)
# ---------------------------------------------------------------------------
def figA_rows():
    z = np.load(TD / "main.npz", allow_pickle=True)
    ours = set(z["row_uid"].astype(str))
    rec = pd.read_parquet(RR / "dataset_kt30" / "samples.parquet")
    rec_test = set(rec.loc[rec.split == "test", "row_key"].astype(str))
    df = pd.read_parquet(ROOT / "data/holdout_masked_v1" /
                         "ships12_predictors_fit0017_cohc.parquet")
    finite = df.drop(columns=["row_uid"]).notna().all(axis=1)
    ships = set(df["row_uid"].astype(str)[finite])
    return ours & rec_test & ships


FIGA_UIDS = figA_rows()
_z = np.load(TD / "main.npz", allow_pickle=True)
Y_USA = dict(zip(_z["row_uid"].astype(str), _z["y"].astype(int)))
print(f"envelope sample: {len(FIGA_UIDS)} rows")


def curve_from_scores(uids, score, y):
    finite = np.isfinite(score)
    score, y = score[finite], y[finite]
    order = np.argsort(-score)
    s_sorted, y_sorted = score[order], y[order]
    out = []
    for thr in np.unique(s_sorted):
        pred = s_sorted >= thr
        tp = int((pred & (y_sorted == 1)).sum())
        fp = int((pred & (y_sorted == 0)).sum())
        fn = int((~pred & (y_sorted == 1)).sum())
        tn = int((~pred & (y_sorted == 0)).sum())
        out.append({"FARate": fp / (fp + tn) if fp + tn else np.nan,
                    "FARatio": fp / (tp + fp) if tp + fp else np.nan,
                    "POD": tp / (tp + fn) if tp + fn else np.nan})
    return pd.DataFrame(out)


def npz_curve(name):
    z = np.load(TD / f"{'main' if name == 'ours' else name}.npz",
                allow_pickle=True)
    uids = z["row_uid"].astype(str)
    keep = np.array([u in FIGA_UIDS for u in uids])
    y = np.array([Y_USA[u] for u in uids[keep]])
    return curve_from_scores(uids[keep], z["p"][keep], y)


def recon_curve(variant):
    z = np.load(RR / f"sweep_{variant}_kt30{_RECON_TAG_KIND}.npz",
                allow_pickle=True)
    uids = z["test_row_uid"].astype(str)
    keep = np.array([u in FIGA_UIDS for u in uids])
    y = np.array([Y_USA[u] for u in uids[keep]])
    return curve_from_scores(uids[keep], z["test_score"][keep], y)


def wrps1_curve():
    z = np.load(ROOT / "data/analysis/wrps1_scores_2024.npz", allow_pickle=True)
    uids = z["row_uid"].astype(str)
    keep = np.array([u in FIGA_UIDS for u in uids])
    y = np.array([Y_USA[u] for u in uids[keep]])
    return curve_from_scores(uids[keep], z["p"][keep], y)


ENV = [
    ("ours", C_OURS, "-", 1.8, "CVRI-Ens (ours)"),
    ("baseline_lr", C_LR, "-", 0.8, "LR"),
    ("baseline_lda", C_LDA, "-", 0.8, "LDA"),
    ("baseline_ships12_lda", C_OHCD, "-", 0.8, "SHIPS-12 OHCD"),
    ("baseline_ships12_lda_cohc", C_COHC, "-", 0.8, "SHIPS-12 COHC"),
    ("wrps1", C_WRPS1, "-", 1.1, "WRPS1 (same sample)"),
    ("recon_V1", C_RITCF, "-", 0.9, "RITCF faithful"),
    ("recon_V2", C_V2, "-.", 0.9, "RITCF clean pool"),
    ("recon_V3", C_V3, "-", 0.9, "RITCF repo skeleton"),
    ("recon_V4direct", C_V4D, "--", 1.1, "RITCF direct clf"),
]

fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 82 * MM))
for ax, (key, xlabel, letter) in zip(axes, (("FARate", "False-detection rate, "
                                             "POFD (%)", "a"),
                                            ("FARatio", "False-alarm ratio, "
                                             "1 − precision (%)", "b"))):
    for name, color, ls, lw, lbl in ENV:
        try:
            if name.startswith("recon_"):
                cur = recon_curve(name[6:])
            elif name == "wrps1":
                cur = wrps1_curve()
            else:
                cur = npz_curve(name)
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
            continue
        ax.plot(cur[key] * 100, cur["POD"] * 100, color=color, ls=ls, lw=lw,
                label=lbl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("POD (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.grid(alpha=0.22, lw=0.35)
    panel_label(ax, letter)
axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
               handletextpad=0.5)
fig.tight_layout()
save_all(fig, "fig_operating_envelopes")
plt.close(fig)
