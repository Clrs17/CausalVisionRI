# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Figure 2: CausalVisionRI architecture schematic (figure-publication style).

Pure matplotlib patches (FancyBboxPatch blocks + FancyArrowPatch connectors),
183 mm double-column width, 125 mm tall, no data dependency.

Layout, left to right: three input branches (TBB IR sequence, environmental
state, discovered causal graph) -> one encoder per branch -> gated
cross-attention fusion -> 24 h RI head.  Only the data -> branch role ->
fusion -> output narrative is drawn; hyper-parameters (LoRA mode, widths,
heads, loss weights, member names) are deliberately left to the text and
tables.  Per-record environmental history supplies both the MLP and GNN.
The adjacency loss feeds the GNN, while sign-gradient and counterfactual losses
feed the auxiliary DV24 training head; a bottom strip carries the top-5
equal-weight ensemble.

Outputs (600 dpi PNG + vector PDF with pdf.fonttype 42):
    figures/fig2_architecture.{png,pdf}

A built-in layout audit runs after drawing and reports (a) text escaping its
block or the canvas, (b) pairwise text collisions, (c) connector endpoints that
do not touch a block, (d) connectors crossing a block interior.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"

MM = 1 / 25.4
W, H = 183.0, 125.0                       # canvas in mm

# --- palette: low-saturation, colour-blind safe; causal branch = dusty red ---
INK, INK_SOFT, INK_FAINT = "#2B2B2B", "#4C4C4C", "#8C8C8C"
GREY_FC, GREY_EC = "#ECECE6", "#9C9C9C"
V_EC, V_FC = "#4F7CA6", "#E5EDF5"          # vision branch (dusty blue)
E_EC, E_FC = "#4C8672", "#E2EEE9"          # environment branch (dusty green)
C_EC, C_FC = "#A9605A", "#F4E3E0"          # causal branch (dusty red)
FUS_FC, FUS_EC = "#3F4A56", "#2C343D"      # fusion (dark slate)
OUT_EC, OUT_FC = "#8A6E33", "#F6EEDA"      # output head (muted ochre)
FLOW = "#555555"

SZ_TITLE, SZ_BODY = 9.0, 7.6              # block title / detail line, pt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

BOXES = {}      # name -> (x, y, w, h)
TEXTS = []      # (artist, parent box name or None)
ARROWS = []     # (label, p0, p1, (anchor_start, anchor_end))


def rbox(name, x, y, w, h, fc, ec, lw=0.9, r=1.4, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, joinstyle="round"))
    BOXES[name] = (x, y, w, h)


def text(x, y, s, size=7.6, color=INK_SOFT, weight="normal", ha="center",
         va="center", parent=None, style="normal", z=6):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, fontstyle=style, zorder=z)
    TEXTS.append((t, parent))
    return t


def block(name, title, lines, tcolor=INK, lcolor=INK_SOFT,
          tsize=SZ_TITLE, lsize=SZ_BODY, lead=5.6, tlead=6.4):
    x, y, w, h = BOXES[name]
    yc = y + h / 2
    if not lines:                          # title-only block (loss chips)
        text(x + w / 2, yc, title, size=tsize, color=tcolor,
             weight="bold", parent=name)
        return
    y_title = yc + (len(lines) - 1) * lead / 2 + 2.6
    text(x + w / 2, y_title, title, size=tsize, color=tcolor,
         weight="bold", parent=name)
    for i, ln in enumerate(lines):
        text(x + w / 2, y_title - tlead - i * lead, ln, size=lsize,
             color=lcolor, parent=name)


def arrow(label, p0, p1, color=FLOW, lw=0.9, ms=6.5, ls="-", z=4,
          anchor=(True, True)):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        linestyle=ls, shrinkA=0, shrinkB=0, zorder=z, capstyle="butt"))
    ARROWS.append((label, p0, p1, anchor))


def shaft(label, p0, p1, color=FLOW, lw=0.9, ls="-", anchor=(True, False)):
    """Headless line used for the dashed part of a polyline connector."""
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=ls,
            zorder=3, solid_capstyle="butt")
    ARROWS.append((label, p0, p1, anchor))


# ----------------------------------------------------------------- geometry --
ROWS = {"vision": 108.0, "env": 79.0, "causal": 50.0}
IN_X, ENC_X, FUS_X, OUT_X = 4.0, 52.0, 100.0, 150.0
BW, BH = 43.0, 21.0
FUS = (FUS_X, 45.0, 40.0, 68.0)          # 45..113, centre 79
OUT = (OUT_X, 69.0, 29.0, 20.0)          # centre 79
DV_OUT = (OUT_X, 47.0, 29.0, 13.0)

rbox("v_in", IN_X, ROWS["vision"] - BH / 2, BW, BH, V_FC, V_EC)
rbox("e_in", IN_X, ROWS["env"] - BH / 2, BW, BH, E_FC, E_EC)
rbox("c_in", IN_X, ROWS["causal"] - BH / 2, BW, BH, C_FC, C_EC)

rbox("v_enc", ENC_X, ROWS["vision"] - BH / 2, BW, BH, V_FC, V_EC)
rbox("e_enc", ENC_X, ROWS["env"] - BH / 2, BW, BH, E_FC, E_EC)
rbox("c_enc", ENC_X, ROWS["causal"] - BH / 2, BW, BH, C_FC, C_EC)

rbox("fus", *FUS, FUS_FC, FUS_EC, lw=1.1, r=2.0)
rbox("out", *OUT, OUT_FC, OUT_EC, lw=1.0)
rbox("dv_out", *DV_OUT, OUT_FC, OUT_EC, lw=0.9, ls=(0, (2.0, 1.6)))

# -------------------------------------------------------------- input blocks --
block("v_in", "TBB IR sequence",
      ["HURSAT-B1 TBB", "≤ 5 frames (t−24 … t, 6-h)"])
block("e_in", "Environmental state",
      ["21 variables: ERA5 · OHC · track", "× 5 history steps (24 h)"])
block("c_in", "Discovered causal graph",
      ["87 directed edges (PCMCI+)", "eight sign-verified ΔV24 effects"])

# ------------------------------------------------------------ encoder blocks --
block("v_enc", "Vision backbone",
      ["LingBot-Vision (ViT-B/16)", "→ pooled visual token"])
block("e_enc", "MLP", ["→ env token"])
block("c_enc", "GNN", ["record history → causal token"])

# --------------------------------------------------------------- fusion block
fx, fw = FUS[0], FUS[2]
fxc = fx + fw / 2
text(fxc, 100.1, "Gated cross-attention", size=9.0, color="#FFFFFF",
     weight="bold", parent="fus")
ax.plot([fx + 4, fx + fw - 4], [93.1, 93.1], color="#78838F", lw=0.5,
        zorder=3)
for yy, s in [(85.2, "pooled visual token queries"),
              (79.6, "env + causal tokens")]:
    text(fxc, yy, s, size=7.6, color="#E8ECF0", parent="fus")
ax.plot([fx + 4, fx + fw - 4], [71.6, 71.6], color="#78838F", lw=0.5,
        zorder=3)
for yy, s in [(63.2, "per-record gate scales"),
              (57.6, "the fused attention output")]:
    text(fxc, yy, s, size=7.4, color="#C9D1D9", parent="fus")

# ---------------------------------------------------------------- output head --
block("out", "P(RI) at +24 h", ["RI head"], tsize=9.0)
block("dv_out", "ΔV24 head", ["training only"], tsize=8.0, lsize=6.6)

# ------------------------------------------------------------ causal loss row --
CHIP_Y, CHIP_W, CHIP_H = 22.0, 26.0, 11.0
chip_x = [27.0, 58.0, 89.0]
for i, title in enumerate(["Adjacency CE", "Sign–gradient", "Counterfactual"]):
    name = f"chip{i}"
    rbox(name, chip_x[i], CHIP_Y, CHIP_W, CHIP_H, C_FC, C_EC, lw=0.7, r=1.0)
    block(name, title, [], tcolor=C_EC, tsize=8.0)

# ----------------------------------------------------------- ensemble strip ---
ST_X, ST_Y, ST_W, ST_H = 6.0, 4.0, 170.0, 15.0
rbox("strip", ST_X, ST_Y, ST_W, ST_H, GREY_FC, GREY_EC, lw=0.7, r=1.2)
text(ST_X + ST_W / 2, 14.4, "Final prediction: top-5 equal-weight ensemble",
     size=8.0, color=INK, weight="bold", parent="strip")
text(ST_X + ST_W / 2, 8.5, "members selected on inner 2018–19 (per-year BSS, worst-year first)",
     size=7.2, color=INK_SOFT, parent="strip")

# ------------------------------------------------------------------ connectors --
arrow("in->enc vision", (IN_X + BW, ROWS["vision"]), (ENC_X, ROWS["vision"]))
arrow("in->enc env", (IN_X + BW, ROWS["env"]), (ENC_X, ROWS["env"]))
shaft("env history->gnn a", (IN_X + BW, 74.0), (49.5, 74.0), color=E_EC,
      ls=(0, (2.0, 1.6)), anchor=(True, False))
shaft("env history->gnn b", (49.5, 74.0), (49.5, 56.0), color=E_EC,
      ls=(0, (2.0, 1.6)), anchor=(False, False))
arrow("env history->gnn c", (49.5, 56.0), (ENC_X, 56.0), color=E_EC,
      ls=(0, (2.0, 1.6)), anchor=(False, True))
arrow("in->enc causal", (IN_X + BW, ROWS["causal"]), (ENC_X, ROWS["causal"]))

arrow("enc->fus vision", (ENC_X + BW, ROWS["vision"]), (FUS_X, 104.0))
arrow("enc->fus env", (ENC_X + BW, ROWS["env"]), (FUS_X, 79.0))
arrow("enc->fus causal", (ENC_X + BW, ROWS["causal"]), (FUS_X, 54.0))

arrow("fus->out", (FUS_X + FUS[2], 79.0), (OUT_X, 79.0))
text((FUS_X + FUS[2] + OUT_X) / 2, 81.4, "z", size=7.4, color=INK_SOFT,
     style="italic")

# The latent state feeds both the RI head and the training-only DV24 head.
shaft("latent->dv a", (144.0, 79.0), (144.0, 53.5), ls=(0, (2.0, 1.6)),
      lw=0.8, anchor=(False, False))
arrow("latent->dv head", (144.0, 53.5), (OUT_X, 53.5), ls=(0, (2.0, 1.6)),
      lw=0.8, ms=5.5, anchor=(False, True))

# Adjacency constrains the GNN; sign and counterfactual constrain the DV24 head.
GNN_BOTTOM = ROWS["causal"] - BH / 2
RAIL_Y = 34.2
chip_cx = [c + CHIP_W / 2 for c in chip_x]
shaft("adjacency rail", (chip_cx[0], RAIL_Y), (54.0, RAIL_Y),
      color=C_EC, lw=0.8, anchor=(False, False))
shaft("response rail", (chip_cx[1], RAIL_Y), (OUT_X + OUT[2] / 2, RAIL_Y),
      color=C_EC, lw=0.8, anchor=(False, False))
for i, cx in enumerate(chip_cx):
    shaft(f"chip{i}->rail", (cx, CHIP_Y + CHIP_H), (cx, RAIL_Y),
          color=C_EC, lw=0.8)
arrow("adjacency->gnn", (54.0, RAIL_Y), (54.0, GNN_BOTTOM), color=C_EC,
      lw=0.8, ms=5.5, anchor=(False, True))
arrow("sign+cf->dv", (OUT_X + OUT[2] / 2, RAIL_Y),
      (OUT_X + OUT[2] / 2, DV_OUT[1]), color=C_EC, lw=0.8, ms=5.5,
      anchor=(False, True))
text(ENC_X + BW / 2, 36.85,
     "training-only causal constraints", size=6.6, color=C_EC)

# ensemble: five trained copies of the pipeline
shaft("out->strip right", (OUT_X + OUT[2], 79.0), (W - 2.0, 79.0),
      ls=(0, (2.0, 1.6)), lw=0.8, anchor=(True, False))
shaft("out->strip down", (W - 2.0, 79.0), (W - 2.0, 23.0),
      ls=(0, (2.0, 1.6)), lw=0.8, anchor=(False, False))
shaft("out->strip across", (W - 2.0, 23.0), (145.0, 23.0),
      ls=(0, (2.0, 1.6)), lw=0.8, anchor=(False, False))
arrow("out->strip head", (145.0, 23.0), (145.0, ST_Y + ST_H), lw=0.8, ms=5.5,
      anchor=(False, True))
text(W - 7.0, 30.0, "× 5 members", size=6.0, color=INK_FAINT,
     ha="right")

# ------------------------------------------------------------- column headers --
for xc, lab in [(IN_X + BW / 2, "INPUT"), (ENC_X + BW / 2, "ENCODER"),
                (FUS_X + FUS[2] / 2, "FUSION"), (OUT_X + OUT[2] / 2, "OUTPUT")]:
    text(xc, 121.5, lab, size=6.4, color=INK_FAINT)


# --------------------------------------------------------------- layout audit --
def _rect_dist(x, y, r):
    """Distance from a point to a rectangle's outline (0 inside the edges)."""
    rx, ry, rw, rh = r
    dx = max(rx - x, 0.0, x - (rx + rw))
    dy = max(ry - y, 0.0, y - (ry + rh))
    if dx == 0 and dy == 0:
        return min(x - rx, rx + rw - x, y - ry, ry + rh - y)
    return (dx ** 2 + dy ** 2) ** 0.5


def audit():
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    problems, ext = [], []
    for art, parent in TEXTS:
        bb = art.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        r = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        ext.append((art.get_text(), r, parent))
        if r[0] < 0.2 or r[1] < 0.2 or r[2] > W - 0.2 or r[3] > H - 0.2:
            problems.append(f"text off canvas: {art.get_text()!r} {r}")
        if parent:
            px, py, pw, ph = BOXES[parent]
            if (r[0] < px - 0.4 or r[1] < py - 0.4
                    or r[2] > px + pw + 0.4 or r[3] > py + ph + 0.4):
                problems.append(f"text overflows {parent}: {art.get_text()!r}")
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            a, b = ext[i][1], ext[j][1]
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            if ox > 0.5 and oy > 0.5:
                problems.append(
                    f"text overlap: {ext[i][0]!r} / {ext[j][0]!r}")
    for label, p0, p1, (a0, a1) in ARROWS:
        for tag, p, on in (("start", p0, a0), ("end", p1, a1)):
            if on and min(_rect_dist(p[0], p[1], b)
                          for b in BOXES.values()) > 0.35:
                problems.append(f"connector {label} {tag} touches no block")
        for k in range(1, 24):
            t = k / 24
            x, y = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
            for name, (bx, by, bw_, bh_) in BOXES.items():
                if (bx + 0.8 < x < bx + bw_ - 0.8
                        and by + 0.8 < y < by + bh_ - 0.8):
                    problems.append(f"connector {label} crosses block {name}")
    return problems


problems = audit()
for p in problems:
    print("LAYOUT:", p)
print(f"layout audit: {len(TEXTS)} text runs, {len(ARROWS)} connectors, "
      f"{len(problems)} problem(s)")

def save(stem):
    """Write PNG+PDF via temp files, then swap in (survives an open viewer)."""
    import os
    import tempfile
    import time
    for suffix, kw in ((".png", {"dpi": 600}), (".pdf", {})):
        dst = FIG / f"{stem}{suffix}"
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=str(FIG))
        os.close(fd)
        try:
            fig.savefig(tmp, facecolor="white", **kw)
            for _ in range(10):
                try:
                    os.replace(tmp, dst)
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise SystemExit(f"could not write {dst}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        print("wrote", dst)


FIG.mkdir(parents=True, exist_ok=True)
save("fig2_architecture")
