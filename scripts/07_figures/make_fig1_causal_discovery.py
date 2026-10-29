# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
# Figure 1 -- causal discovery: prior-constrained PCMCI+ graph and graph-adjusted ATEs
#
# a  contemporaneous (lag 0) sub-graph
# b  lagged (lag 1-4, 6 h per lag) sub-graph, strongest links only
# c  bootstrap ATE forest for the 24-h intensity change (DV24)
# d  bootstrap ATE forest for rapid intensification (RI)
#
# Data (all local, read-only):
#   wpac_causal_phase1/data/processed_masked/holdout_fit0017/prior/
#       causal_graph_edges.json   (97 edges, q_value, coef, lag, sign)
#       ate_results.csv           (bootstrap ATEs, 10 treatments x 2 outcomes)
#       trusted_signs.json        (8 bootstrap-stable signs; D200, MFC excluded)
#
# Layout and visual grammar follow the phase-1 companion figure script
# scripts/07_figures/make_fig1.py in this release
# (POS / ZONES / FAM tables), so Fig. 1 of both papers is directly comparable.
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.path import Path as MPath
from matplotlib.text import Text
from matplotlib.transforms import Bbox, blended_transform_factory

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "ps.fonttype": 42,
    "font.size": 8.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "savefig.transparent": False,
})

MM = 1 / 25.4
BASE = Path(__file__).resolve().parents[2]
PRIOR = BASE / "data" / "phase1" / "processed_masked" / "holdout_fit0017" / "prior"
OUT = Path(__file__).resolve().parents[2] / "figures" / "fig1_causal_discovery"
OUT.parent.mkdir(parents=True, exist_ok=True)

Q_MAX = 0.05          # edges above this FDR level are not drawn
MAX_LAGGED = 32       # lagged links kept by |coef| rank (readability), parents always kept
TARGET = "DV24_YJ"
PARENTS = ("VMAX", "OHC_OA")

# ------------------------------------------------------------------- data ---
EDGES_ALL = json.loads((PRIOR / "causal_graph_edges.json").read_text())["edges"]
TRUSTED = json.loads((PRIOR / "trusted_signs.json").read_text())
ATE = pd.read_csv(PRIOR / "ate_results.csv").set_index(["outcome", "treatment"])

EDGES = [e for e in EDGES_ALL if e["q_value"] <= Q_MAX]
PANEL_A = [e for e in EDGES if e["lag"] == 0]
LAGGED_SIG = [e for e in EDGES if e["lag"] >= 1]
_ranked = sorted(LAGGED_SIG, key=lambda e: -abs(e["coef"]))[:MAX_LAGGED]
_keep = {id(e) for e in _ranked} | {id(e) for e in LAGGED_SIG if e["dst"] == TARGET}
PANEL_B = sorted((e for e in LAGGED_SIG if id(e) in _keep), key=lambda e: abs(e["coef"]))
COV_CUT = min(abs(e["coef"]) for e in PANEL_B)   # |coef| cutoff actually applied

# ------------------------------------------------------- grouping / style ---
GROUPS = {
    "state":    ["VMAX", "PER", "SPD", "LAT", "LAND_DIST"],
    "dynamics": ["SHRD", "SHEAR_VMAX_ANGLE", "VORT", "PV", "D200", "W500"],
    "moisture": ["RHLO", "RHHI", "MFC", "DRY_AIR"],
    "ocean":    ["SST_IC", "SST_OA", "OHC_IC", "OHC_OA", "SLHF_IC", "WARM_CORE"],
}
GROUP_OF = {v: g for g, vs in GROUPS.items() for v in vs}

FAM = {  # colour-blind safe, low-saturation family palette (border / tint)
    "state":    {"ec": "#6A8E7F", "fc": "#DCE8E2"},
    "dynamics": {"ec": "#4A7A96", "fc": "#D5E1E9"},
    "moisture": {"ec": "#8C8C8C", "fc": "#E4E4E4"},
    "ocean":    {"ec": "#C1666B", "fc": "#EFD9DA"},
    "target":   {"ec": "#272727", "fc": "#272727"},
}
FAM_LABEL = {
    "state": "Storm state & track",
    "dynamics": "Atmospheric dynamics",
    "moisture": "Moisture",
    "ocean": "Ocean & heat",
}
EDGE_POS = "#A34E54"   # dusty red   -> positive partial correlation / positive ATE
EDGE_NEG = "#3B627A"   # steel blue  -> negative partial correlation / negative ATE
GREY = "#8C8C8C"
HALO = "#3A3A3A"

POS = {
    "PER":       (10, 86), "VMAX": (12, 64), "LAT": (9, 44),
    "SPD":       (12, 24), "LAND_DIST": (13, 6),
    "SHRD":      (35, 90), "SHEAR_VMAX_ANGLE": (61, 90),
    "VORT":      (38, 71), "PV": (63, 77), "D200": (73, 61), "W500": (52, 61),
    "RHLO":      (33, 48), "RHHI": (50, 50), "MFC": (33, 38), "DRY_AIR": (48, 40),
    "WARM_CORE": (64, 25),
    "SLHF_IC":   (47, 13), "SST_IC": (62, 4), "OHC_IC": (68, 17),
    "SST_OA":    (82, 6),  "OHC_OA": (88, 17),
    "DV24_YJ":   (92, 52),
}
LABEL = {v: v for v in POS}
LABEL[TARGET] = "$\\Delta V_{24}$"

ZONES = [  # x0, y0, x1, y1, name, label position
    (1.5, 1.5, 25.5, 96.5, "Storm state & track", "tl"),
    (28, 58, 80, 98.5, "Atmospheric dynamics", "tl"),
    (27, 35, 56.5, 55.5, "Moisture", "tl"),
    (32, 0.8, 98.5, 30, "Ocean & heat", "bl"),
]
ZONE_LABEL_POS = {
    "Storm state & track": (1.6, 94.5, "top"),
    "Atmospheric dynamics": (30.2, 97.0, "top"),
    "Moisture": (28.2, 54.5, "top"),
    "Ocean & heat": (33.2, 28.5, "top"),
}

MAXC = max(abs(e["coef"]) for e in EDGES)

DRAWN = {}   # tag -> [(src, dst, lag, sampled polyline)] for the layout self-check

RAD_OVR = {  # seeding curvature for long / crossing-prone links (rad units in arc3)
    ("VMAX", TARGET, 0): 0.20,
    ("VMAX", TARGET, 1): 0.02,
    ("VMAX", TARGET, 3): -0.16,
    ("OHC_OA", TARGET, 1): -0.10,
    ("W500", "D200", 0): 0.12,
    ("WARM_CORE", "PV", 0): -0.10,
    ("DRY_AIR", "RHHI", 0): 0.00,
    ("DRY_AIR", "RHHI", 1): 0.12,
    ("RHHI", "DRY_AIR", 1): 0.12,
    ("W500", "MFC", 0): 0.18,
    ("RHLO", "MFC", 0): -0.12,
}
CUBIC_OVR = {
    ("OHC_IC", "PER", 1): (-54.0, 0.0),
    ("W500", "SST_IC", 3): (-60.0, 76.0),
    ("W500", "SST_OA", 3): (-30.0, 2.0),
    ("PV", "SLHF_IC", 1): (24.0, 56.0),
    ("VORT", "MFC", 1): (-4.0, 24.0),
}


def edge_style(coef, a_lo=0.20, a_hi=0.50, w_lo=0.50, w_hi=1.90):
    """Colour by sign; width and opacity scale with |coef| within [lo, hi] floors."""
    f = abs(coef) / MAXC
    return (EDGE_POS if coef > 0 else EDGE_NEG), w_lo + (w_hi - w_lo) * f, \
        a_lo + (a_hi - a_lo) * f


# per-panel ink floors: panel a carries only 6 links, so they must read at a glance
STYLE = {
    "a": dict(a_lo=0.35, a_hi=0.70, w_lo=0.70, w_hi=2.10),
    "b": dict(a_lo=0.20, a_hi=0.55, w_lo=0.55, w_hi=1.90),
}


def compute_rads(edges):
    """Fan out parallel links and keep reciprocal paths on opposite sides."""
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
                r += 0.14
            rads[id(e)] = RAD_OVR.get((e["src"], e["dst"], e["lag"]), r)
    return rads


# ---------------------------------------------------- arc3 path utilities ---
def arc3_control(p0, p1, rad):
    """Control point of matplotlib's arc3 connection (identical formula)."""
    x12, y12 = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return x12 + rad * dy, y12 - rad * dx


def bezier(p0, c, p1, t):
    t = np.asarray(t, dtype=float)[:, None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(c) + t ** 2 * np.asarray(p1)


DELTAS = [0.0] + [s * d for d in (0.04, 0.08, 0.12, 0.16, 0.20, 0.26,
                                  0.32, 0.38, 0.44, 0.50, 0.56) for s in (1, -1)]


def count_hits(pts, boxes, skip, clear=0.5):
    """Number of node/zone-label boxes (inflated by `clear`) hit by a sampled path."""
    n = 0
    for name, (x0, y0, x1, y1) in boxes.items():
        if name in skip:
            continue
        if np.any((pts[:, 0] > x0 - clear) & (pts[:, 0] < x1 + clear) &
                  (pts[:, 1] > y0 - clear) & (pts[:, 1] < y1 + clear)):
            n += 1
    return n


def route(p0, p1, base, boxes, skip, clear=0.5):
    """Pick the smallest curvature perturbation that avoids every obstacle box.

    Boxes are inflated by `clear` data units so that near-misses (arcs grazing a
    label) are rejected, not just strict intersections.
    """
    best, best_score = base, None
    for d in DELTAS:
        rad = base + d
        c = arc3_control(p0, p1, rad)
        pts = bezier(p0, c, p1, np.linspace(0.0, 1.0, 240))
        hits = count_hits(pts, boxes, skip, clear)
        full = bezier(p0, c, p1, np.linspace(0, 1, 80))
        outside = int(np.sum((full[:, 0] < 0.4) | (full[:, 0] > 99.6) |
                             (full[:, 1] < 0.4) | (full[:, 1] > 99.6)))
        extra = float(np.sum(np.hypot(*np.diff(full, axis=0).T))) - float(
            np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        score = 1000 * hits + 400 * outside + 4 * abs(d) + 0.05 * extra
        if best_score is None or score < best_score:
            best, best_score = rad, score
    return best


def cubic_ctrl(p0, p1, r1, r2):
    """Two free handles perpendicular to the chord: an S-capable connector."""
    d = np.subtract(p1, p0, dtype=float)
    nrm = np.array([d[1], -d[0]]) / np.hypot(*d)
    c1 = np.asarray(p0, float) + d / 3.0 + nrm * r1
    c2 = np.asarray(p0, float) + 2 * d / 3.0 + nrm * r2
    return c1, c2


def cubic_pts(p0, c1, c2, p1, t):
    t = np.asarray(t, dtype=float)[:, None]
    return ((1 - t) ** 3 * np.asarray(p0, float) +
            3 * (1 - t) ** 2 * t * np.asarray(c1, float) +
            3 * (1 - t) * t ** 2 * np.asarray(c2, float) +
            t ** 3 * np.asarray(p1, float))


GRID = (-72.0, -54.0, -36.0, -24.0, -18.0, -12.0, -7.0, -3.0, 0.0, 3.0, 7.0, 12.0, 18.0, 24.0, 36.0, 54.0, 72.0)


def route_cubic(p0, p1, boxes, skip, clear=0.5):
    """Fallback for corridors an arc cannot thread: sweep both Bezier handles."""
    best, best_score = None, None
    for r1 in GRID:
        for r2 in GRID:
            c1, c2 = cubic_ctrl(p0, p1, r1, r2)
            pts = cubic_pts(p0, c1, c2, p1, np.linspace(0.0, 1.0, 240))
            hits = count_hits(pts, boxes, skip, clear)
            full = cubic_pts(p0, c1, c2, p1, np.linspace(0, 1, 120))
            outside = int(np.sum((full[:, 0] < 0.4) | (full[:, 0] > 99.6) |
                                 (full[:, 1] < 0.4) | (full[:, 1] > 99.6)))
            extra = float(np.sum(np.hypot(*np.diff(full, axis=0).T))) - float(
                np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            score = (1000 * hits + 400 * outside + 0.05 * extra +
                     0.8 * (abs(r1) + abs(r2)))
            if best_score is None or score < best_score:
                best, best_score = (r1, r2), score
    return best if best_score < 1000 else None


def segment_clear(p0, p1, boxes, skip):
    """Whether an open segment avoids every non-endpoint box interior."""
    for name, (x0, y0, x1, y1) in boxes.items():
        if name in skip:
            continue
        t0, t1 = 0.0, 1.0
        for p, d, lo, hi in ((p0[0], p1[0] - p0[0], x0, x1),
                             (p0[1], p1[1] - p0[1], y0, y1)):
            if abs(d) < 1e-12:
                if p <= lo or p >= hi:
                    break
            else:
                a, b = (lo - p) / d, (hi - p) / d
                if a > b:
                    a, b = b, a
                t0, t1 = max(t0, a), min(t1, b)
                if t0 >= t1:
                    break
        else:
            return False
    return True


def visibility_route(p0, p1, boxes, skip, clear=0.5):
    """Shortest visible-corner route when a single Bezier cannot fit."""
    obstacles = {v: (x0 - clear, y0 - clear, x1 + clear, y1 + clear)
                 for v, (x0, y0, x1, y1) in boxes.items()}
    points = [np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)]
    for x0, y0, x1, y1 in obstacles.values():
        for x in (x0, x1):
            for y in (y0, y1):
                if 0.4 <= x <= 99.6 and 0.4 <= y <= 99.6:
                    points.append(np.array([x, y]))
    n = len(points)
    dist = [np.inf] * n
    previous = {}
    seen = set()
    dist[0] = 0.0
    for _ in range(n):
        u = min((i for i in range(n) if i not in seen),
                key=lambda i: dist[i], default=None)
        if u is None or not np.isfinite(dist[u]):
            break
        if u == 1:
            break
        seen.add(u)
        for v in range(n):
            if v in seen or not segment_clear(points[u], points[v], obstacles, skip):
                continue
            candidate = dist[u] + float(np.hypot(*(points[v] - points[u])))
            if candidate < dist[v] - 1e-12:
                dist[v] = candidate
                previous[v] = u
    if not np.isfinite(dist[1]):
        raise RuntimeError("no visible route through the node layout")
    indices = [1]
    while indices[-1] != 0:
        indices.append(previous[indices[-1]])
    return np.array([points[i] for i in reversed(indices)])


def trim_polyline(pts, d0, d1, n=180):
    """Cut a sampled curve at arc-length distances d0 / d1 from its two ends."""
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    s = np.clip(np.linspace(d0, cum[-1] - d1, n), 0.0, cum[-1])
    return np.column_stack([np.interp(s, cum, pts[:, 0]),
                            np.interp(s, cum, pts[:, 1])])


def box_interval(p0, p1, box):
    """Fractions at which a segment enters and leaves a rectangle."""
    x0, y0, x1, y1 = box
    t0, t1 = 0.0, 1.0
    for p, d, lo, hi in ((p0[0], p1[0] - p0[0], x0, x1),
                         (p0[1], p1[1] - p0[1], y0, y1)):
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return None
        else:
            a, b = (lo - p) / d, (hi - p) / d
            if a > b:
                a, b = b, a
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return None
    return t0, t1


def clip_to_node_boxes(pts, source_box, target_box, margin=0.12):
    """Clip a connector at its actual intersections with the two node boxes."""
    pts = np.asarray(pts, dtype=float)
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    source_inside = ((pts[:, 0] >= sx0) & (pts[:, 0] <= sx1) &
                     (pts[:, 1] >= sy0) & (pts[:, 1] <= sy1))
    target_inside = ((pts[:, 0] >= tx0) & (pts[:, 0] <= tx1) &
                     (pts[:, 1] >= ty0) & (pts[:, 1] <= ty1))
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    start = None
    for i in range(len(pts) - 1):
        if source_inside[i] and not source_inside[i + 1]:
            interval = box_interval(pts[i], pts[i + 1], source_box)
            if interval is None:
                raise RuntimeError("source boundary intersection not found")
            start = cum[i] + interval[1] * (cum[i + 1] - cum[i])
            break
    stop = None
    for i in range(len(pts) - 2, -1, -1):
        if not target_inside[i] and target_inside[i + 1]:
            interval = box_interval(pts[i], pts[i + 1], target_box)
            if interval is None:
                raise RuntimeError("target boundary intersection not found")
            stop = cum[i] + interval[0] * (cum[i + 1] - cum[i])
            break
    if start is None or stop is None or stop <= start:
        raise RuntimeError("connector cannot be clipped to its endpoint boxes")
    return trim_polyline(pts, start + margin, cum[-1] - stop + margin)


# ----------------------------------------------------------------- figure ---
FIG_W, FIG_H = 183 * MM, 168 * MM
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")

# --- explicit 2x2 grid, laid out in millimetres so the rows cannot overlap ----
# vertical budget (168 mm): bottom 0.8 | footnote 6.6 | legend 8.6 | axis labels 3 |
#                           forests 38 | gutter 6 | graphs 86 | titles 5.8
GRAPH_MM, GUTTER_MM, FOREST_MM = 86, 6, 38
FOREST_BOTTOM_MM = 24.2
GRAPH_BOTTOM_MM = FOREST_BOTTOM_MM + FOREST_MM + GUTTER_MM
LAYOUT = {
    "a": (3, GRAPH_BOTTOM_MM, GRAPH_MM, GRAPH_MM),
    "b": (94, GRAPH_BOTTOM_MM, GRAPH_MM, GRAPH_MM),
    "c": (28, FOREST_BOTTOM_MM, 62, FOREST_MM),
    "d": (113, FOREST_BOTTOM_MM, 62, FOREST_MM),
}
AX = {k: fig.add_axes([x / (FIG_W / MM), y / (FIG_H / MM),
                           w / (FIG_W / MM), h / (FIG_H / MM)])
      for k, (x, y, w, h) in LAYOUT.items()}
axA, axB, axC, axD = AX["a"], AX["b"], AX["c"], AX["d"]
ROW_GAP_MM = LAYOUT["a"][1] - (LAYOUT["c"][1] + LAYOUT["c"][3])   # a bottom - c top


def draw_zones(ax):
    labels = {}
    for x0, y0, x1, y1, name, _ in ZONES:
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.8,rounding_size=2.2",
            fc="#FAFAFA", ec="#D2D2D2", lw=0.5, ls=(0, (3, 2)), zorder=0,
            clip_on=True))
        tx, ty, va = ZONE_LABEL_POS[name]
        labels[name] = ax.text(tx, ty, name, fontsize=6.5, style="italic",
                               color="#6A6A6A", ha="left", va=va, zorder=1, clip_on=True)
    return labels


def draw_nodes(ax):
    texts = {}
    for v, (x, y) in POS.items():
        g = "target" if v == TARGET else GROUP_OF[v]
        st = FAM[g]
        tc = "white" if g == "target" else "#1a1a1a"
        is_t = g == "target"
        texts[v] = ax.text(
            x, y, LABEL[v], ha="center", va="center",
            fontsize=8.5 if is_t else 8.0,
            fontweight="bold" if is_t else "normal", color=tc, zorder=5,
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.34", fc=st["fc"], ec=st["ec"],
                      lw=1.5 if (is_t or v in PARENTS) else 0.7))
    return texts


def measure(ax, texts, fig_):
    """Data-space bounds of the rendered labels, including any box padding."""
    fig_.canvas.draw()
    renderer = fig_.canvas.get_renderer()
    inv = ax.transData.inverted()
    boxes = {}
    for v, t in texts.items():
        patch = t.get_bbox_patch()
        bb = (patch.get_window_extent(renderer) if patch is not None
              else t.get_window_extent(renderer))
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        boxes[v] = (x0, y0, x1, y1)
    return boxes


def draw_edges(ax, edges, rads, boxes, end_boxes, emphasis=(), tag=""):
    n_in, n_cubic = 0, 0
    DRAWN[tag] = []
    for e in edges:
        s, t = e["src"], e["dst"]
        p0, p1 = POS[s], POS[t]
        rad = route(p0, p1, rads[id(e)], boxes, {s, t})
        rads[id(e)] = rad
        color, lw, alpha = edge_style(e["coef"], **STYLE[tag])
        key = (s, t, e["lag"]) in emphasis
        if key:
            lw *= 1.7
            alpha = min(1.0, alpha + 0.34)
        style = dict(arrowstyle="-|>", mutation_scale=5.0, color=color, lw=lw,
                     alpha=alpha, zorder=3 if key else 2,
                     capstyle="round", joinstyle="round")
        cubic = CUBIC_OVR.get((s, t, e["lag"]))
        poly = None
        if cubic is None:
            c = arc3_control(p0, p1, rad)
            candidate = bezier(p0, c, p1, np.linspace(0, 1, 400))
            if count_hits(candidate, boxes, {s, t}) == 0:
                poly = clip_to_node_boxes(candidate, end_boxes[s], end_boxes[t])
        routed = cubic is not None
        if poly is None:
            routed = True
            handles = cubic if cubic is not None else route_cubic(p0, p1, boxes, {s, t})
            if handles is not None:
                c1, c2 = cubic_ctrl(p0, p1, *handles)
                candidate = cubic_pts(p0, c1, c2, p1, np.linspace(0, 1, 400))
                if count_hits(candidate, boxes, {s, t}) == 0:
                    poly = clip_to_node_boxes(candidate, end_boxes[s], end_boxes[t])
            if poly is None:
                waypoints = visibility_route(p0, p1, boxes, {s, t})
                poly = clip_to_node_boxes(waypoints, end_boxes[s], end_boxes[t])
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(poly) - 1)
        ax.add_patch(FancyArrowPatch(path=MPath(poly, codes), shrinkA=0.0,
                                     shrinkB=0.0, **style))
        n_cubic += int(routed)
        DRAWN[tag].append((s, t, e["lag"], poly))
        if t == TARGET:
            n_in += 1
    return n_in, n_cubic


def draw_panel(edges, ax, tag, title, emphasis=()):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    zone_texts = draw_zones(ax)
    texts = draw_nodes(ax)
    end_boxes = measure(ax, texts, fig)
    boxes = {k: (x0 - 0.6, y0 - 0.6, x1 + 0.6, y1 + 0.6) for k, (x0, y0, x1, y1) in end_boxes.items()}
    obstacles = dict(boxes)
    obstacles.update({f"zone:{name}": box
                      for name, box in measure(ax, zone_texts, fig).items()})
    rads = compute_rads(edges)
    n_in, n_cubic = draw_edges(ax, edges, rads, obstacles, end_boxes, emphasis, tag)
    for v in PARENTS:  # halo: direct parent of DV24
        x0, y0, x1, y1 = boxes[v]
        ax.add_patch(FancyBboxPatch(
            (x0 - 0.7, y0 - 0.7), (x1 - x0) + 1.4, (y1 - y0) + 1.4,
            boxstyle="round,pad=0.45,rounding_size=1.6",
            fc="none", ec=HALO, lw=1.2, zorder=4))
    ax.set_title(title, fontsize=8.5, pad=3.5, color="#1a1a1a")
    ax.text(-0.02, 1.005, tag, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")
    return len(edges), n_in, boxes, n_cubic


_KEY_PHYS = {("OHC_IC", "VMAX", 1)}   # ocean-state -> intensity channel
EMPH_B = {(e["src"], e["dst"], e["lag"]) for e in PANEL_B
          if e["dst"] == TARGET or (e["src"], e["dst"], e["lag"]) in _KEY_PHYS}

nA, nA_in, boxesA, cA = draw_panel(
    PANEL_A, axA, "a", "Contemporaneous links (lag 0)",
    {(e["src"], e["dst"], e["lag"]) for e in PANEL_A if e["dst"] == TARGET})
nB, nB_in, boxesB, cB = draw_panel(
    PANEL_B, axB, "b", "Lagged links (lag 1–4, 6 h per lag)", EMPH_B)

# --------------------------------------------------------- panels c and d ---
def draw_forest(ax, outcome, tag, xlabel, xlim, xticks, d200_text_xy):
    sub = ATE.loc[outcome]
    order = sub["ate_boot"].sort_values().index.tolist()
    ypos = {t: i for i, t in enumerate(order)}
    d200_xy = (float(sub.loc["D200", "ate_boot"]), ypos["D200"])

    ax.axvline(0, color="0.45", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.75, len(order) - 0.25)

    trusted = set(TRUSTED["signs"])
    for t in order:
        r = sub.loc[t]
        y = ypos[t]
        lo, hi, pt = r["ci_boot_low"], r["ci_boot_high"], r["ate_boot"]
        if t == "D200":
            color, mk, mfc, ms, mew = GREY, "D", "white", 3.8, 1.0
        elif r["p_value_boot"] >= 0.05:
            color, mk, mfc, ms, mew = GREY, "o", "white", 3.2, 1.0
        else:
            color = EDGE_POS if pt > 0 else EDGE_NEG
            mk, mfc, ms, mew = "o", color, 3.2, 0.8
        ax.plot([lo, hi], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=2)
        for xc in (lo, hi):
            ax.plot([xc, xc], [y - 0.11, y + 0.11], color=color, lw=0.7, zorder=2)
        ax.plot([pt], [y], marker=mk, ms=ms, mfc=mfc, mec=color, mew=mew,
                ls="", zorder=3)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8.0)
    ax.tick_params(axis="y", length=0, pad=8.5)
    ax.tick_params(axis="x", labelsize=7.5, pad=2)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, fontsize=8.0, labelpad=3)
    ax.spines[["top", "right"]].set_visible(False)

    # trusted-sign markers in the label gutter
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for t in order:
        if t in trusted:
            ax.plot([-0.021], [ypos[t]], marker="s", ms=2.5, mfc="#2B2B2B",
                    mec="#2B2B2B", mew=0.4, ls="", transform=tr,
                    clip_on=False, zorder=6)

    # inline significance tags
    for t in order:
        r = sub.loc[t]
        if t != "D200" and r["p_value_boot"] >= 0.05:
            ax.text(r["ci_boot_high"] + 0.012 * (xlim[1] - xlim[0]), ypos[t],
                    "n.s.", fontsize=6.5, color=GREY, style="italic",
                    ha="left", va="center", zorder=4)

    ax.annotate("association only", xy=d200_xy, xytext=d200_text_xy,
                textcoords="data", fontsize=6.5, style="italic",
                color="#5E5E5E", ha="left", va="center", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.65,
                                shrinkA=2, shrinkB=3.5,
                                connectionstyle="arc3,rad=0.16"), zorder=4)
    ax.text(-0.02, 1.005, tag, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")
    return len(order)


nC = draw_forest(axC, "DV24", "c", "Effect on 24-h intensity change (kt per SD)",
                 (-6.9, 9.6), [-6, -4, -2, 0, 2, 4, 6, 8],
                 d200_text_xy=(4.3, 4.5))
nD = draw_forest(axD, "RI", "d", "Effect on RI (log-odds per SD)",
                 (-1.00, 1.30), [-1.0, -0.5, 0.0, 0.5, 1.0],
                 d200_text_xy=(0.60, 5.0))

# ---------------------------------------------------------------- legend ----
handles = [Patch(fc=FAM[g]["fc"], ec=FAM[g]["ec"], lw=0.7, label=FAM_LABEL[g])
           for g in ("state", "dynamics", "moisture", "ocean")]
handles += [
    Patch(fc=FAM["target"]["fc"], ec=FAM["target"]["ec"], lw=0.7,
          label="Target $\\Delta V_{24}$"),
    Patch(fc="none", ec=HALO, lw=1.2, label="Halo = direct parent"),
    Line2D([], [], color=EDGE_POS, lw=1.4, label="Positive link / effect"),
    Line2D([], [], color=EDGE_NEG, lw=1.4, label="Negative link / effect"),
    Line2D([], [], color="#888888", lw=0.6,
           label="Width/opacity: |partial corr.|"),
    Line2D([], [], marker="s", ls="", ms=2.5, mfc="#2B2B2B", mec="#2B2B2B",
           label="Trusted sign"),
    Line2D([], [], marker="D", ls="", ms=3.6, mfc="white", mec=GREY, mew=1.0,
           label="D200: association only"),
    Line2D([], [], marker="o", ls="", ms=3.0, mfc="white", mec=GREY, mew=1.0,
           label="n.s. ($p\\geq0.05$)"),
]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6.2,
           handlelength=1.4, handletextpad=0.45, columnspacing=0.9,
           labelspacing=0.18, borderpad=0.0, borderaxespad=0.0,
           bbox_to_anchor=(0.5, 8.8 / (FIG_H / MM)))

n_sig_lag = len(LAGGED_SIG)
fig.text(0.5, 0.8 / (FIG_H / MM),
         f"a, lag-0 links at unadjusted $p\\leq0.05$; b, {n_sig_lag} lagged "
         f"links at BH $q\\leq${Q_MAX:g} ({nB} shown); "
         "c, d, graph-adjusted effects with bootstrap 95% CIs.",
         fontsize=6.0, color="#3F3F3F", ha="center", va="bottom",
         linespacing=1.45)

# ------------------------------------------------------------ self-check ----
PROBLEMS = []


def report(msgs, text, ok):
    """Record one check, pass/fail, and remember failures for the exit code."""
    msgs.append(("OK   " if ok else "FAIL ") + text)
    if not ok:
        PROBLEMS.append(text)


def artist_bboxes(ax, r):
    """Every rendered artist of an axes with its display bbox."""
    out = []

    def add(name, bb):
        out.append((name, Bbox([[bb.x0, bb.y0], [bb.x1, bb.y1]])))

    for t in ax.texts:
        if t.get_text():
            add("text:" + t.get_text()[:16].replace("\n", "/"), t.get_window_extent(r))
    for pa in ax.patches:
        add("patch:" + type(pa).__name__, pa.get_window_extent(r))
    for ln in ax.lines:
        pts = ln.get_transform().transform(ln.get_xydata())
        add("line:marker=" + str(ln.get_marker()),
            Bbox([[pts[:, 0].min() - 1, pts[:, 1].min() - 1],
                  [pts[:, 0].max() + 1, pts[:, 1].max() + 1]]))
    for t, name in ((ax.title, "title"), (ax.xaxis.label, "xlabel"),
                    (ax.yaxis.label, "ylabel")):
        if t.get_text():
            add(name, t.get_window_extent(r))
    if ax.axison:
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if t.get_text():
                add("tick:" + t.get_text(), t.get_window_extent(r))
    return out


def layout_report():
    """Cross-panel geometry: axes must not overlap and nothing may leak out of its box."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = {k: ax.get_window_extent(r) for k, ax in AX.items()}
    px_per_mm = ab["a"].height / GRAPH_MM
    msgs = []
    gutter = (ab["a"].y0 - ab["c"].y1) / px_per_mm
    report(msgs, "geometry (mm, x/y/w/h): " +
           "; ".join(f"{k} {LAYOUT[k][0]},{LAYOUT[k][1]},{LAYOUT[k][2]},{LAYOUT[k][3]}"
                     for k in "abcd") +
           f" | row gutter a-c / b-d = {gutter:.1f} mm", gutter >= GUTTER_MM - 0.1)
    pairs = [(k, o) for k in AX for o in AX if k < o and ab[k].overlaps(ab[o])]
    report(msgs, f"axes pairs overlapping = {pairs or 'none'}", not pairs)
    for k in "ab":                       # nothing may hang below its own panel
        worst = min(artist_bboxes(AX[k], r), key=lambda nb: nb[1].y0)
        below = (ab[k].y0 - worst[1].y0) / px_per_mm
        report(msgs, f"{k}: lowest artist '{worst[0]}' sits {below:+.2f} mm relative "
                     f"to the axes bottom", below <= 0)
    leaks = []
    for k, ax in AX.items():
        for other, bb in ab.items():
            if other == k:
                continue
            for name, b in artist_bboxes(ax, r):
                if b.overlaps(bb):
                    leaks.append(f"{k}:{name}->{other}")
    report(msgs, f"artists entering another panel = {leaks or 'none'}", not leaks)
    graph_texts = [t for ax in (axA, axB) for t in ax.texts if t.get_text() not in ("a", "b")]
    clip_ok = (all(t.get_clip_on() for t in graph_texts) and
               all(pa.get_clip_on() for pa in axA.patches + axB.patches))
    report(msgs, "graph-panel texts and patches clipped to own axes", clip_ok)
    for name, art in (("footnote", fig.texts[-1]), ("legend", fig.legends[0]),
                      ("forest axes", axC), ("forest xlabel", axC.xaxis.label)):
        bb = art.get_window_extent(r)
        msgs.append(f"info  {name:13s} y = {bb.y0 / px_per_mm:5.1f} .. "
                    f"{bb.y1 / px_per_mm:5.1f} mm")
    return msgs


def text_only_bbox(text, renderer):
    """Text extent without an annotation leader's bounding box."""
    if getattr(text, "arrow_patch", None) is None:
        return text.get_window_extent(renderer)
    return Text.get_window_extent(text, renderer)


def overlap_report():
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    msgs = []
    for ax, tag in ((axA, "a"), (axB, "b")):          # node-label collisions
        tbs = [(t.get_bbox_patch() if t.get_bbox_patch() is not None else t)
               .get_window_extent(r)
               for t in ax.texts if t.get_text() in LABEL.values()]
        bad = sum(1 for i in range(len(tbs)) for j in range(i + 1, len(tbs))
                  if tbs[i].overlaps(tbs[j]))
        report(msgs, f"{tag}: node-label overlaps = {bad}/{len(tbs)}", bad == 0)
    zone_names = set(FAM_LABEL.values())
    zone_group = {label: group for group, label in FAM_LABEL.items()}
    zone_overlaps = []
    for i, a in enumerate(ZONES):
        for b in ZONES[i + 1:]:
            if (max(a[0], b[0]) < min(a[2], b[2])
                    and max(a[1], b[1]) < min(a[3], b[3])):
                zone_overlaps.append(f"{a[4]} / {b[4]}")
    report(msgs, f"zone overlaps = {zone_overlaps or 'none'}", not zone_overlaps)
    for ax, tag in ((axA, "a"), (axB, "b")):
        zt = {t.get_text(): t for t in ax.texts if t.get_text() in zone_names}
        zb = measure(ax, zt, fig)
        outside = []
        for x0, y0, x1, y1, name, _ in ZONES:
            bx0, by0, bx1, by1 = zb[name]
            if bx0 < x0 or by0 < y0 or bx1 > x1 or by1 > y1:
                outside.append(name)
        report(msgs, f"{tag}: zone labels outside zones = {outside or 'none'}",
               not outside)
        nt = {t.get_text(): t for t in ax.texts if t.get_text() in LABEL.values()}
        nb = measure(ax, nt, fig)
        misplaced = []
        for x0, y0, x1, y1, name, _ in ZONES:
            group = zone_group[name]
            for v, bb in nb.items():
                if v not in GROUP_OF or GROUP_OF[v] != group:
                    continue
                if bb[0] < x0 or bb[1] < y0 or bb[2] > x1 or bb[3] > y1:
                    misplaced.append(f"{v} ({name})")
        report(msgs, f"{tag}: nodes outside family zones = {misplaced or 'none'}",
               not misplaced)
        collisions = [name for name, bb in zb.items()
                      if any(Bbox([bb[:2], bb[2:]]).overlaps(
                          Bbox([nb[v][:2], nb[v][2:]])) for v in nb)]
        report(msgs, f"{tag}: zone-label overlaps = {collisions or 'none'}",
               not collisions)
        edge_zone_hits = [
            f"{s}->{t}(lag {lag}) / {name}"
            for s, t, lag, poly in DRAWN[tag]
            for name, bb in zb.items()
            if any(box_interval(p0, p1, bb) is not None
                   for p0, p1 in zip(poly[:-1], poly[1:]))]
        report(msgs, f"{tag}: edge/zone-label crossings = "
                     f"{edge_zone_hits or 'none'}", not edge_zone_hits)
        end_hits = []
        for s, t, lag, poly in DRAWN[tag]:
            for v, point in ((s, poly[0]), (t, poly[-1])):
                x0, y0, x1, y1 = nb[LABEL[v]]
                if (x0 - 1e-7 <= point[0] <= x1 + 1e-7 and
                        y0 - 1e-7 <= point[1] <= y1 + 1e-7):
                    end_hits.append(f"{s}->{t}(lag {lag})")
                    break
        report(msgs, f"{tag}: edge endpoints inside node boxes = "
                     f"{end_hits or 'none'}", not end_hits)
        seen = {}
        coincident = []
        for s, t, lag, poly in DRAWN[tag]:
            key = (min(s, t), max(s, t), lag)
            if key in seen and np.allclose(poly, seen[key][::-1], rtol=0, atol=1e-6):
                coincident.append(f"{key[0]}<->{key[1]}(lag {lag})")
            else:
                seen[key] = poly
        report(msgs, f"{tag}: reciprocal edges with identical reversed paths = "
                     f"{coincident or 'none'}", not coincident)
    lb = fig.legends[0].get_window_extent(r)
    hit = [n for n, ax in AX.items() if lb.overlaps(ax.get_window_extent(r))]
    report(msgs, f"legend overlaps axes = {hit or 'none'}", not hit)
    fb = fig.get_window_extent(r)
    out = []
    for n, ax in AX.items():
        arts = [ax.title, ax.xaxis.label] + list(ax.texts)
        if ax in (axC, axD):
            arts += list(ax.get_yticklabels())
        for t in arts:
            if not t.get_text():
                continue
            b = t.get_window_extent(r)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > fb.x1 + 1 or b.y1 > fb.y1 + 1:
                out.append(f"{n}:{t.get_text()!r}")
    report(msgs, f"artists outside canvas = {out or 'none'}", not out)
    esc = [f"{s}->{t}" for tag in ("a", "b") for s, t, lag, poly in DRAWN[tag]
           if np.any((poly[:, 0] < 0) | (poly[:, 0] > 100) |
                     (poly[:, 1] < 0) | (poly[:, 1] > 100))]
    report(msgs, f"edges leaving the graph canvas = {len(esc)} {esc or ''}".strip(),
           not esc)
    for tag, boxes in (("a", boxesA), ("b", boxesB)):
        hits = [f"{s}->{t}(lag {lag})"
                for s, t, lag, poly in DRAWN[tag]
                if any(v not in (s, t) for v in boxes
                       if np.any((poly[:, 0] > boxes[v][0]) & (poly[:, 0] < boxes[v][2]) &
                                 (poly[:, 1] > boxes[v][1]) & (poly[:, 1] < boxes[v][3])))]
        report(msgs, f"{tag}: edge/node-label crossings = {len(hits)} {hits or ''}".strip(),
               not hits)
    for ax, tag in ((axC, "c"), (axD, "d")):
        texts = list(ax.texts) + [ax.xaxis.label] + [t for t in ax.get_yticklabels()
                                                     if t.get_text()]
        tb = [(t.get_text(), text_only_bbox(t, r)) for t in texts]
        t_over = [f"{a!r}x{b!r}" for i, (a, ba) in enumerate(tb)
                  for b, bb in tb[i + 1:] if ba.overlaps(bb)]
        segs = []
        for ln in ax.lines:
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if len(xd) == 2 and list(yd) == [0, 1]:
                continue                  # zero reference line (blended transform)
            segs.append(ln.get_transform().transform(np.column_stack([xd, yd])))
        s_over = []
        for txt, bb in tb:
            for pts in segs:
                sb = Bbox([[pts[:, 0].min() - 1, pts[:, 1].min() - 4],
                           [pts[:, 0].max() + 1, pts[:, 1].max() + 4]])
                if bb.overlaps(sb):
                    s_over.append(txt)
        report(msgs, f"{tag}: text/text overlaps = {t_over or 'none'}; "
                     f"text/CI-bar overlaps = {s_over or 'none'}",
               not t_over and not s_over)
        sq = [ln.get_transform().transform(ln.get_xydata())[0] for ln in ax.lines
              if ln.get_marker() == "s"]
        lab = [t.get_window_extent(r) for t in ax.get_yticklabels() if t.get_text()]
        sq_over = sum(1 for pt in sq for b in lab
                      if Bbox([[pt[0] - 2, pt[1] - 2], [pt[0] + 2, pt[1] + 2]]).overlaps(b))
        report(msgs, f"{tag}: trusted squares = {len(sq)}, overlapping y labels = {sq_over}",
               sq_over == 0 and len(sq) == 8)
        xlo = min(b.x0 for _, b in tb)
        report(msgs, f"{tag}: leftmost label/annotation x = {xlo:.1f} px", xlo >= 0)
    for name, art in (("legend", fig.legends[0]), ("footnote", fig.texts[-1])):
        b = art.get_window_extent(r)
        report(msgs, f"{name} inside canvas",
               bool(b.x0 >= -0.5 and b.y0 >= -0.5 and b.x1 <= fb.x1 + 0.5))
    blockers = []
    for n, ax in (("c", axC), ("d", axD)):
        for t in [ax.xaxis.label] + list(ax.get_yticklabels()) + list(ax.texts):
            if t.get_text() and lb.overlaps(t.get_window_extent(r)):
                blockers.append(f"{n}:{t.get_text()!r}")
    report(msgs, f"legend over forest labels = {blockers or 'none'}", not blockers)
    report(msgs, "legend/footnote separated",
           not lb.overlaps(fig.texts[-1].get_window_extent(r)))
    return msgs


for m in layout_report():
    print("layout:", m)
for m in overlap_report():
    print("check:", m)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=600)
plt.close(fig)

print(f"saved: {OUT.with_suffix('.pdf')}")
print(f"saved: {OUT.with_suffix('.png')}")
print(f"panel a: {nA} edges ({nA_in} into target); panel b: {nB} edges of "
      f"{n_sig_lag} significant lagged ({nB_in} into target), |coef| >= {COV_CUT:.4f}")
print(f"panel c: {nC} points; panel d: {nD} points")
print(f"routed connectors (cubic or visible-corner): panel a {cA}, panel b {cB}")
print(f"SELF-CHECK: {'PASS' if not PROBLEMS else 'FAIL -> ' + '; '.join(PROBLEMS)}")
if PROBLEMS:
    raise SystemExit(1)
