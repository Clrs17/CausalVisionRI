# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
#!/usr/bin/env python
"""Consistent-sample comparison engine (secondary analysis).

Spec: ``docs/COMPARISON.md`` (v1) + ``docs/SM_MAP.md``.

Consistency axiom: inside one figure every model shares

  1. the same row set (an explicit intersection),
  2. the same RI label definition,
  3. the same evaluation window,
  4. the same metric recipes (a public operating rule plus a val/inner F1 point
     re-derived *under that figure's label*).

Two figures plus one control cell:

  fig A        our standard   : USA-30kt labels, test 2020-2024, three-way
                                intersection (recon kt30 test x our frozen test
                                rows x SHIPS-12 all-finite), n=2,178.
  fig A (CMA)  control cell   : the *same rows*, CMA-30kt labels (label swap,
                                no new inference).
  fig B        their standard : CMA-25.27kt labels, test 2020-2021, three-way
                                intersection (recon 25.27 test x our test rows
                                x SHIPS-12 all-finite), n=949.

Nothing is trained or re-inferred here: our probabilities are the frozen
holdout predictions subset by ``row_uid`` (once-only protocol untouched), and
the only new numbers are metrics, thresholds and coverage counts.

Inputs: the RITCF per-row sweep scores of ``data/ritcf_recon/`` under one
checkpoint selection.  ``--recon-kind`` picks the checkpoint (``best_loss``,
``best_auc``, ``last``) and ``--recon-tag-suffix`` the run tag, so the file
names follow ``sweep_<V>[_kt30]<tag_suffix>_<kind>.npz`` and
``sweep_table[_kt30]<tag_suffix>_<kind>.csv`` (``--recon-tag-suffix ""`` walks
back to the pre-``kind`` legacy names ``sweep_<V>_main.npz`` /
``sweep_table.csv``).  Which file each variant actually read, and every
fallback that had to be taken, is recorded in the lock under
``ritcf_sweep_inputs``.

Outputs (``data/analysis/compare/``), each carrying the kind marker so that
runs with different checkpoint selections sit side by side::

  table_figA_<kind>.csv, table_figB_<kind>.csv
                                   model x metric x operating rule
  iso_pofd_<kind>.csv              POD at FARate = 9.4% (both figs)
  waterfall_<kind>.csv             sample waterfall with drop reasons
  coverage_<kind>.csv              RI-row / storm coverage and drop reasons
  fig_compare_pofd_<kind>.png      fig A, POD vs FARate
  fig_compare_faratio_<kind>.png   fig A, FARatio vs POD
  fig_compare_cma_pofd_<kind>.png  fig A rows, CMA-30kt labels
  fig_compare_pofd_2021_<kind>.png fig B, POD vs FARate (+ paper anchor)
  fig_compare_faratio_2021_<kind>.png
                                   fig B, FARatio vs POD (+ paper anchor)
  fig_compare_pr_<kind>.png        fig A, precision-recall, scored models
  fig_compare_roc_<kind>.png       fig A, ROC, scored models

and the lock ``protocol_locks/comparison_secondary_<kind>.json``.  The
unsuffixed tables and lock of earlier runs are left untouched.  Use
``--figures-only`` to redraw the figures without changing the CSV tables or
lock; ``--skip-figures`` writes only the tables and lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.dpi": 200, "axes.titlesize": 8.6})

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "06_compare"))
from wpac_warning.evaluate import (  # noqa: E402
    auc_metrics, bss, classification_metrics, f1_threshold)
from wrps1_same_sample import PREDICTORS, wrps1_score  # noqa: E402

DATA = ROOT / "data"
TD = DATA / "test" / "holdout_masked_v1"
OUT = DATA / "analysis" / "compare"
LOCK_DIR = ROOT / "protocol_locks"
RECON_DIR = DATA / "ritcf_recon"

RECON_KINDS = ("best_loss", "best_auc", "last")
# Which RITCF reconstruction sweep feeds the RITCF rows.  ``_aucfix`` is the
# tag of the run that re-evaluated all three checkpoints; the kind names the
# checkpoint file (see scripts/ritcf_recon/operating_sweep.py).
RECON = {"kind": "best_auc", "tag_suffix": "_aucfix"}
RECON_INPUTS: dict[str, dict] = {}      # variant -> file actually read
RECON_WARNINGS: list[str] = []          # fallbacks and missing artifacts
SKIP_FIGURES = False
SKIP_TABLES = False

DEV_RATE = 0.0791            # fixed development-period RI rate (BSS reference)
FARATE_TARGET = 0.094        # paper claim's FARate (= POFD)
CLAIM = {"pod": 0.910, "farate": 0.094, "faratio": 0.587, "n": 1149,
         "source": "Wang, Yang & Li (2025) PNAS Table 1, reanalysis row"}
MEMBERS = ("ens_lbB_partial4_sf", "ens_lbB_lora16_temp5", "ens_lbB_temp2",
           "ens_lbB_lora16_sf", "ens_lbB_temp5_slim")
BASELINES = ("baseline_lr", "baseline_lda", "baseline_ships12_lda",
             "baseline_ships12_lda_cohc")
V_KT30 = ("V1", "V2", "V3", "V4direct")      # no V4 (mean dist) at 30 kt
V_2527 = ("V1", "V2", "V3", "V4", "V4direct")

COLORS = {
    "ours": "#0b6fa4",
    "baseline_lr": "#8d99a6",
    "baseline_lda": "#4d8a3d",
    "baseline_ships12_lda": "#b8860b",
    "baseline_ships12_lda_cohc": "#c9a227",
    "wrps1_own": "#1a7a4a",
    "wrps1_f1": "#57a773",
    "ritcf_V1": "#b03a2e",
    "ritcf_V2": "#d48a7c",
    "ritcf_V3": "#e07b5f",
    "ritcf_V4": "#7d3c98",
    "ritcf_V4direct": "#5e2018",
}
PRETTY = {
    "ours": "CVRI-Ens",
    "baseline_lr": "LR",
    "baseline_lda": "LDA",
    "baseline_ships12_lda": "SHIPS-12 OHCD",
    "baseline_ships12_lda_cohc": "SHIPS-12 COHC",
    "wrps1_own": "WRPS1 pu>=0.5",
    "wrps1_f1": "WRPS1 inner-F1",
    "ritcf_V1": "RITCF V1 (rolling pool)",
    "ritcf_V2": "RITCF V2 (clean pool)",
    "ritcf_V3": "RITCF V3 (repo skeleton)",
    "ritcf_V4": "RITCF V4 (mean dist)",
    "ritcf_V4direct": "RITCF V4direct (prob)",
}
RITCF_NAMES = PRETTY


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def row_hash(uids) -> str:
    """sha256 over the sorted, newline-joined row_uids."""
    return hashlib.sha256("\n".join(sorted(map(str, uids))).encode()).hexdigest()


def f1_from_counts(tp, fp, fn):
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1e-9)


def pofd_curve(score, y):
    """Staircase (FARate, POD, threshold) from the highest threshold down."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y).astype(int)
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    us, inv = np.unique(score, return_inverse=True)
    tp_by = np.bincount(inv, weights=y, minlength=len(us))
    fp_by = np.bincount(inv, weights=(1 - y), minlength=len(us))
    tp = np.cumsum(tp_by[::-1])
    fp = np.cumsum(fp_by[::-1])
    farate = np.r_[0.0, fp / max(nneg, 1)]
    pod = np.r_[0.0, tp / max(npos, 1)]
    thr = us[::-1]
    return farate, pod, thr


def faratio_curve(score, y):
    farate, pod, thr = pofd_curve(score, y)
    npos = int(np.asarray(y).astype(int).sum())
    tp = pod * npos
    fp = farate * (len(y) - npos)
    with np.errstate(invalid="ignore", divide="ignore"):
        faratio = np.where(tp + fp > 0, fp / np.maximum(tp + fp, 1e-12), 0.0)
    return pod, faratio, thr


def pod_at_farate(score, y, target=FARATE_TARGET):
    """Interpolated POD at a fixed FARate plus the nearest discrete point."""
    farate, pod, thr = pofd_curve(score, y)
    val = float(np.interp(target, farate, pod))
    j = int(np.argmin(np.abs(farate[1:] - target))) + 1
    return {"POD_at_farate": val, "target_farate": target,
            "nearest_farate": float(farate[j]), "nearest_pod": float(pod[j]),
            "nearest_threshold": float(thr[j - 1])}


def _save(fig, path):
    """Save via a temp file + replace, and run the layout self-checks.

    The file name picks up the checkpoint-selection marker, so the figures of
    two runs never overwrite each other.
    """
    path = path.with_name(out_name(path.name))
    if SKIP_FIGURES:
        plt.close(fig)
        print(f"  skipped {path.relative_to(ROOT)}")
        return {"file": path.name, "saved": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax = fig.axes[0]
    axbox = ax.get_window_extent(rend)
    checks = {"file": path.name, "axes": [round(v, 1) for v in
                                          (axbox.x0, axbox.y0, axbox.x1,
                                           axbox.y1)]}
    legend = ax.get_legend()
    if legend is not None:
        lb = legend.get_window_extent(rend)
        checks["legend_entries"] = len(legend.get_texts())
        checks["legend_outside_axes"] = bool(lb.x0 >= axbox.x1 - 1.0)
        checks["legend_box"] = [round(v, 1) for v in
                                (lb.x0, lb.y0, lb.x1, lb.y1)]
    texts = [t for t in ax.texts]
    checks["text_annotations"] = len(texts)
    checks["texts_inside_figure"] = bool(all(
        t.get_window_extent(rend).x1 <= fig.bbox.x1 + 1 and
        t.get_window_extent(rend).y0 >= 0 for t in texts))
    outside = 0
    for c in ax.collections:
        off = np.asarray(c.get_offsets(), dtype=float)
        if off.size:
            pts = ax.transData.transform(off)
            outside += int(((pts[:, 0] < axbox.x0 - 1) |
                            (pts[:, 0] > axbox.x1 + 1) |
                            (pts[:, 1] < axbox.y0 - 1) |
                            (pts[:, 1] > axbox.y1 + 1)).sum())
    lines = 0
    for ln in ax.lines:
        if ln.get_gid() == "axes_fraction_line":
            continue                      # axvline lives in axes fractions
        xy = ax.transData.transform(np.column_stack(ln.get_data()))
        if xy.size:
            outside += int(((xy[:, 0] < axbox.x0 - 1) |
                            (xy[:, 0] > axbox.x1 + 1) |
                            (xy[:, 1] < axbox.y0 - 1) |
                            (xy[:, 1] > axbox.y1 + 1)).sum())
            lines += 1
    checks["drawn_lines"] = lines
    checks["artists_outside_axes"] = outside
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=str(path.parent))
    os.close(fd)
    try:
        fig.savefig(tmp, bbox_inches="tight", dpi=200)
        for _ in range(20):
            try:
                os.replace(tmp, path)
                break
            except OSError:
                time.sleep(1.5)
        else:
            raise SystemExit(f"could not replace {path} (held open?)")
        print(f"  wrote {path.relative_to(ROOT)}  layout={checks}")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return checks


def _place_labels(ax, items, fontsize=7.0):
    """Greedy label placement: first offset whose bbox hits nothing.

    Returns the number of labels that had to be placed on the fallback offset
    (i.e. that could not be placed without a collision).
    """
    cands = [(6, 5), (6, -9), (-6, 5), (-6, -9), (12, 9), (-12, -9),
             (0, 12), (0, -13), (14, -1), (-14, -1), (18, 12), (-18, 12),
             (24, 18), (-24, 18), (24, -20), (-24, -20)]
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(rend)
    taken = []
    fallbacks = 0
    for x, y, text, color in items:
        chosen = None
        for dx, dy in cands:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            ann = ax.annotate(text, (x, y), fontsize=fontsize, color=color,
                              xytext=(dx, dy), textcoords="offset points",
                              ha=ha, va="bottom" if dy >= 0 else "top",
                              zorder=6)
            bb = ann.get_window_extent(rend)
            inside = (bb.x0 >= axbox.x0 - 2 and bb.x1 <= axbox.x1 + 2 and
                      bb.y0 >= axbox.y0 - 2 and bb.y1 <= axbox.y1 + 2)
            clash = any(bb.overlaps(o) for o in taken)
            if inside and not clash:
                chosen = bb
                break
            ann.remove()
        if chosen is not None:
            taken.append(chosen)
        else:
            fallbacks += 1
            ann = ax.annotate(text, (x, y), fontsize=fontsize, color=color,
                              xytext=(6, 5), textcoords="offset points",
                              zorder=6)
            taken.append(ann.get_window_extent(rend))
    return fallbacks


# --------------------------------------------------------------------------
# loading (frozen assets only)
# --------------------------------------------------------------------------
def load_ours():
    z = np.load(TD / "main.npz", allow_pickle=True)
    df = pd.DataFrame({
        "row_uid": z["row_uid"].astype(str),
        "sid": z["sid"].astype(str),
        "storm_year": z["storm_year"].astype(int),
        "y_usa": z["y"].astype(int),
        "p": z["p"].astype(float),
    })
    meta = {"threshold": float(z["threshold"]),
            "lock_sha256": str(z["lock_sha256"]),
            "member_names": [str(m) for m in z["member_names"]],
            "member_weights": [float(w) for w in z["weights"]]}
    return df, meta


def load_baselines():
    out, thr = {}, {}
    for name in BASELINES:
        z = np.load(TD / f"{name}.npz", allow_pickle=True)
        out[name] = pd.DataFrame({"row_uid": z["row_uid"].astype(str),
                                  "p": z["p"].astype(float),
                                  "y_usa": z["y"].astype(int)})
        meta = json.loads((DATA / "analysis" / "eval" /
                           f"{name}.json").read_text(encoding="utf-8"))
        thr[name] = float(meta["threshold"])
    return out, thr


def load_ships():
    df = pd.read_parquet(DATA / "holdout_masked_v1" /
                         "ships12_predictors_fit0017_cohc.parquet")
    df["row_uid"] = df["row_uid"].astype(str)
    finite = np.ones(len(df), dtype=bool)
    for v in PREDICTORS:
        finite &= np.isfinite(df[v].to_numpy(dtype=float))
    return df, finite


def recon_stem(head, tag):
    """``head`` under the selected run tag + checkpoint kind.

    ``head`` is ``sweep_<V>`` for the per-row scores and ``sweep_table`` for
    the operating table; ``tag`` in {'kt30', '2527'} adds the dataset part.
    """
    ds = "kt30" if tag == "kt30" else ""
    stem = "_".join(p for p in (head, ds) if p) + RECON["tag_suffix"]
    return f"{stem}_{RECON['kind']}"


def legacy_stem(head, tag):
    """Pre-``--checkpoint-kind`` name of the same artifact."""
    if head == "sweep_table":
        # the legacy operating tables never carried the ``main`` marker
        return "sweep_table_kt30" if tag == "kt30" else "sweep_table"
    return f"{head}_{'kt30' if tag == 'kt30' else 'main'}"


def resolve_recon(head, tag, variant=None, ext=".npz"):
    """Path of one recon artifact, with the fallback chain spelled out.

    Direct hit first; then V2 borrowing V1 (V2 is V1's checkpoint scored
    through the clean inference pool, so V1's file is the same model); then
    the legacy kind-less sweep of that run tag, which predates the three
    checkpoints and therefore carries the best-loss checkpoint.
    """
    cands = [(recon_stem(head, tag), None)]
    if variant == "V2":
        cands.append((recon_stem("sweep_V1", tag),
                      "V2 has no file of its own here; read V1's (V2 reuses "
                      "the V1 checkpoint)"))
    cands.append((legacy_stem(head, tag),
                  "no file for this checkpoint kind; read the legacy "
                  "kind-less sweep (best-loss checkpoint)"))
    for name, why in cands:
        path = RECON_DIR / f"{name}{ext}"
        if path.is_file():
            return path, name, why
    raise SystemExit(
        f"no RITCF sweep for {head!r} tag={tag!r}: tried "
        + ", ".join(n + ext for n, _ in cands))


def out_name(stem):
    """Output artifact name carrying the checkpoint-selection marker."""
    p = Path(stem)
    return f"{p.stem}_{RECON['kind']}{p.suffix}"


def load_recon(tag):
    """tag in {'kt30', '2527'} -> test/val frames with row_key, ri, SID."""
    rel = "dataset_kt30/samples.parquet" if tag == "kt30" else \
        "dataset/samples.parquet"
    df = pd.read_parquet(DATA / "ritcf_recon" / rel)
    df["row_uid"] = df["row_key"].astype(str)
    return {"test": df[df.split == "test"].reset_index(drop=True),
            "val": df[df.split == "val"].reset_index(drop=True),
            "all": df}


def load_sweep(tag, variant):
    path, name, why = resolve_recon(f"sweep_{variant}", tag, variant)
    key = f"{tag}:{variant}"
    RECON_INPUTS[key] = {"file": f"data/ritcf_recon/{name}.npz",
                         "requested_kind": RECON["kind"],
                         "requested_tag_suffix": RECON["tag_suffix"],
                         "fallback": why}
    if why:
        msg = f"RITCF {variant} [{tag}] -> {name}.npz ({why})"
        print(f"  NOTE {msg}")
        RECON_WARNINGS.append(msg)
    z = np.load(path, allow_pickle=True)
    RECON_INPUTS[key].update({"n_test": int(len(z["test_row_uid"])),
                              "n_val": int(len(z["val_row_uid"]))})
    test = pd.DataFrame({"row_uid": z["test_row_uid"].astype(str),
                         "p": z["test_score"].astype(float),
                         "y_recon": z["test_y"].astype(int)})
    val = pd.DataFrame({"row_uid": z["val_row_uid"].astype(str),
                        "p": z["val_score"].astype(float),
                        "y_recon": z["val_y"].astype(int)})
    return test, val


def load_inner(model_key):
    """Frozen inner (2018-19) predictions; ensemble-averaged for ours."""
    if model_key == "ours":
        pres, ref = [], None
        for m in MEMBERS:
            z = np.load(DATA / "runs" / "holdout_masked_v1" / m /
                        "preds" / "inner.npz", allow_pickle=True)
            uid = z["row_uid"].astype(str)
            if ref is None:
                ref = uid
                y = z["y_inner"].astype(int)
            elif not np.array_equal(ref, uid):
                raise SystemExit(f"inner row order differs for {m}")
            pres.append(z["p_inner"].astype(float))
        return pd.DataFrame({"row_uid": ref, "p_inner": np.mean(pres, axis=0),
                             "y_usa": y})
    z = np.load(DATA / "runs_holdout" / "baselines" / model_key /
                "inner.npz", allow_pickle=True)
    return pd.DataFrame({"row_uid": z["row_uid"].astype(str),
                         "p_inner": z["p_inner"].astype(float),
                         "y_usa": z["y_inner"].astype(int)})


def load_frame_availability():
    fav = json.loads((DATA / "holdout_masked_v1" /
                      "frame_availability.json").read_text(encoding="utf-8"))
    rows = {r["row_uid"]: r for r in fav["rows"]}
    return rows, fav


def load_panel():
    """Phase-1 panel rows (unfiltered) for exclusion-reason attribution."""
    p = pd.read_parquet(DATA / "holdout_masked_v1" / "panel_fit0017.parquet",
                        columns=["SID", "ISO_TIME", "VMAX", "VMAX_F24",
                                 "F24_TIME", "DV24", "storm_year"])
    p["ISO_TIME"] = pd.to_datetime(p["ISO_TIME"], utc=True)
    p["row_uid"] = (p["SID"].astype(str) + "|" +
                    p["ISO_TIME"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return p.set_index("row_uid")


# --------------------------------------------------------------------------
# samples: waterfall + coverage
# --------------------------------------------------------------------------
def build_sample_A(ours, ships, finite, recon30):
    ships_uids = set(ships["row_uid"].to_numpy()[finite])
    recon_test = recon30["test"]
    recon_any = set(recon30["all"]["row_uid"])
    stages = []

    def stage(name, mask_src, extra=None):
        df = mask_src
        row = {"stage": name, "n": len(df),
               "n_RI_USA30": int(df["y_usa"].sum()) if "y_usa" in df else None,
               "n_RI_CMA30": int(df["ri"].sum()) if "ri" in df else None}
        if extra:
            row.update(extra)
        stages.append(row)
        return row

    base = ours.copy()
    stage("0_our_frozen_test_rows_official", base)
    step1 = base[base.row_uid.isin(ships_uids)]
    stage("1_and_ships12_all_finite", step1,
          {"dropped": len(base) - len(step1),
           "drop_reason": "SHIPS-12 predictor not all finite"})
    r30 = recon_test[["row_uid", "ri", "SID"]]
    step2 = step1.merge(r30, on="row_uid", how="inner")
    dropped = step1[~step1.row_uid.isin(set(r30["row_uid"]))]
    by_year = {str(k): int(v) for k, v in
               dropped["storm_year"].value_counts().sort_index().items()}
    n_in_recon_any = int(dropped["row_uid"].isin(recon_any).sum())
    stage("2_and_recon_kt30_test_rows", step2,
          {"dropped": len(step1) - len(step2),
           "drop_reason":
               "row absent from the recon kt30 dataset in every split (their "
               "eligibility: CMA wind / frame availability / storm filter); "
               "not a wind-source join failure",
           "dropped_by_year": by_year,
           "dropped_present_in_any_recon_split": n_in_recon_any,
           "recon_kt30_test_rows": int(len(recon_test)),
           "recon_kt30_test_window": "2020-2024"})
    uids = sorted(step2["row_uid"].tolist())
    return {"uids": uids, "stages": stages, "step2": step2}


def build_sample_B(ours, ships, finite, recon25, fav_rows, panel):
    ships_uids = set(ships["row_uid"].to_numpy()[finite])
    ours_uids = set(ours["row_uid"])
    r25 = recon25["test"][["row_uid", "ri", "SID"]]
    y_usa_map = dict(zip(ours["row_uid"], ours["y_usa"]))
    stages = []
    n0 = len(r25)
    stages.append({"stage": "0_recon_25.27_test_rows", "n": n0,
                   "n_RI_CMA25": int(r25["ri"].sum())})
    step1 = r25[r25.row_uid.isin(ours_uids)]
    stages.append({"stage": "1_and_our_frozen_test_rows", "n": len(step1),
                   "n_RI_CMA25": int(step1["ri"].sum()),
                   "n_RI_USA30": int(sum(y_usa_map.get(u, 0)
                                         for u in step1["row_uid"])),
                   "dropped": n0 - len(step1),
                   "drop_reason": "row has no frozen prediction (new "
                                  "inference not allowed); see reason table"})
    step2 = step1[step1.row_uid.isin(ships_uids)]
    stages.append({"stage": "2_and_ships12_all_finite", "n": len(step2),
                   "n_RI_CMA25": int(step2["ri"].sum()),
                   "n_RI_USA30": int(sum(y_usa_map.get(u, 0)
                                         for u in step2["row_uid"])),
                   "dropped": len(step1) - len(step2),
                   "drop_reason": "SHIPS-12 predictor not all finite"})
    uids = sorted(step2["row_uid"].tolist())

    # why the 103 recon test rows carry no locked prediction
    span = panel.reset_index().groupby("SID")["ISO_TIME"].agg(["min", "max"])
    elig_test = ours_uids | {u for u, r in fav_rows.items()
                             if r["split"] == "test"}
    uncovered = sorted(set(r25["row_uid"]) - ours_uids)
    reasons = {"frame_availability:*": 0,
               "eligible_filter (embargo / F24 target-window boundary)": 0}
    for uid in uncovered:
        if uid in fav_rows:
            key = f"frame_availability:{fav_rows[uid]['reason']}"
        elif uid in panel.index and uid not in elig_test:
            key = "eligible_filter (embargo / F24 target-window boundary)"
        elif uid in panel.index:
            key = "present_in_panel_and_eligible_but_missing_from_frozen_table"
        else:
            sid, stamp = uid.split("|")
            if sid not in span.index:
                key = "usa_side_row_absent:storm_not_in_phase1_panel"
            else:
                t = pd.Timestamp(stamp)
                if t < span.loc[sid, "min"]:
                    key = "usa_side_row_absent:before_first_panel_row"
                elif t > span.loc[sid, "max"]:
                    key = "usa_side_row_absent:after_last_panel_row"
                else:
                    key = "usa_side_row_absent:internal_panel_gap"
        reasons[key] = reasons.get(key, 0) + 1
    return {"uids": uids, "stages": stages,
            "uncovered_reasons": reasons, "n_uncovered": len(uncovered),
            "step2": step2}


def coverage_block(uids, recon_test, ri_label):
    """RI-row / storm coverage of a figure sample inside the recon test set.

    The RI denominator is the recon dataset's own label (``ri_label``); the
    rows themselves are label-agnostic, so this is a pure coverage count.
    """
    sel = recon_test[recon_test.row_uid.isin(set(uids))]
    storms_all = set(recon_test["SID"].astype(str))
    storms_in = set(sel["SID"].astype(str))
    ri_all = int(recon_test["ri"].sum())
    ri_in = int(sel["ri"].sum())
    return {"ri_denominator_label": ri_label, "rows": len(uids),
            "ri_rows_total": ri_all, "ri_rows_covered": ri_in,
            "ri_row_coverage": ri_in / max(ri_all, 1),
            "storms_total": len(storms_all), "storms_covered": len(storms_in),
            "storm_coverage": len(storms_in) / max(len(storms_all), 1)}


# --------------------------------------------------------------------------
# score assembly
# --------------------------------------------------------------------------
def assemble(uids, ours, baselines, ships, finite, sweeps, wrps1_cache):
    idx = pd.Index(uids, name="row_uid")
    df = pd.DataFrame(index=idx)
    o = ours.set_index("row_uid")
    df["y_usa"] = o.loc[uids, "y_usa"].to_numpy()
    df["p_ours"] = o.loc[uids, "p"].to_numpy()
    df["sid"] = o.loc[uids, "sid"].to_numpy()
    for name, b in baselines.items():
        df[f"p_{name}"] = b.set_index("row_uid").loc[uids, "p"].to_numpy()
    sc = wrps1_cache["scores"]
    df["wrps1_pu"] = sc["pu"].reindex(idx).to_numpy()
    df["wrps1_pl"] = sc["pl"].reindex(idx).to_numpy()
    df["wrps1_p"] = sc["p"].reindex(idx).to_numpy()
    for var, (test_df, _val_df) in sweeps.items():
        df[f"ritcf_{var}"] = test_df.set_index("row_uid")["p"].reindex(
            idx).to_numpy()
    return df


def wrps1_table(ships, finite):
    ev = ships[finite].copy()
    sc = wrps1_score(ev)
    sc.index = pd.Index(ev["row_uid"].astype(str), name="row_uid")
    return {"scores": sc}


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------
def ritcf_val_f1_picks(tag):
    """Per-variant val-F1 optimum from the recon operating sweep.

    The sweep CSV is the model's own val split (CMA labels, their protocol);
    the operating knob is ``vote_frac`` for the voting variants and the raw
    scan for V4 / V4direct.  ``tag`` selects the dataset (and therefore the
    label definition) exactly as in :func:`load_sweep`.
    """
    path, name, why = resolve_recon("sweep_table", tag, ext=".csv")
    key = f"{tag}:sweep_table"
    RECON_INPUTS[key] = {"file": f"data/ritcf_recon/{name}.csv",
                         "requested_kind": RECON["kind"],
                         "requested_tag_suffix": RECON["tag_suffix"],
                         "fallback": why}
    if why:
        msg = f"RITCF operating table [{tag}] -> {name}.csv ({why})"
        print(f"  NOTE {msg}")
        RECON_WARNINGS.append(msg)
    sw = pd.read_csv(path)
    RECON_INPUTS[key]["variants"] = sorted(sw.variant.unique().tolist())
    val = sw[sw.split == "val"]
    picks = {}
    for var, g in val.groupby("variant"):
        g = g.copy()
        g["F1"] = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
        i = g.F1.idxmax()
        r = g.loc[i]
        picks[var] = {"knob": str(r.knob), "threshold": float(r.value),
                      "val_F1": float(r.F1), "val_POD": float(r.POD),
                      "val_FARate": float(r.FARate),
                      "val_n": int(r.tp + r.fp + r.fn + r.tn),
                      "val_n_RI": int(r.tp + r.fn),
                      "source": f"data/ritcf_recon/{name}.csv (val split)"}
    return picks


def ritcf_val_f1_pick_fallback(tag, variant):
    """Val-F1 pick for a variant the selected kind's table does not carry.

    The score column of that variant already falls back to the legacy sweep
    (:func:`resolve_recon`), so the operating point has to come from the same
    legacy table; the sweep and its threshold then stay on one protocol.
    """
    name = legacy_stem("sweep_table", tag)
    legacy = RECON_DIR / f"{name}.csv"
    if not legacy.is_file():
        raise SystemExit(f"{variant} is missing from the {tag} operating table "
                         f"and {legacy} does not exist either")
    sw = pd.read_csv(legacy)
    g = sw[(sw.split == "val") & (sw.variant == variant)].copy()
    if g.empty:
        raise SystemExit(f"{variant} is missing from the {tag} operating table "
                         f"and from data/ritcf_recon/{name}.csv; cannot place "
                         f"its val-F1 point")
    g["F1"] = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
    r = g.loc[g.F1.idxmax()]
    src = f"data/ritcf_recon/{name}.csv"
    msg = (f"RITCF {variant} [{tag}] val-F1 point taken from {src} "
           f"(no row in the {RECON['kind']} table)")
    print(f"  NOTE {msg}")
    RECON_WARNINGS.append(msg)
    return {"knob": str(r.knob), "threshold": float(r.value),
            "val_F1": float(r.F1), "val_POD": float(r.POD),
            "val_FARate": float(r.FARate),
            "val_n": int(r.tp + r.fp + r.fn + r.tn),
            "val_n_RI": int(r.tp + r.fn),
            "source": f"{src} (val split; legacy checkpoint selection)"}


def ritcf_val_f1_usa_relabel(sweeps, inner_map):
    """Sensitivity: RITCF's val-F1 operating point re-derived on USA-30kt.

    The primary val-F1 point (spec) comes from the recon val sweep, whose
    labels are CMA-30kt.  The val scores themselves are frozen, so the same
    scan can be re-read against the USA-30kt labels; the difference shows how
    much of the operating point is label-driven.
    """
    out = {}
    for var, (_test, val) in sweeps.items():
        y = val["row_uid"].map(inner_map)
        ok = y.notna().to_numpy()
        p = val["p"].to_numpy(dtype=float)[ok]
        yy = y.to_numpy()[ok].astype(int)
        if var in ("V1", "V2", "V3"):
            rows = []
            for k in range(0, 11):
                thr = k / 10.0
                pred = p >= thr
                tp = int((pred & (yy == 1)).sum())
                fp = int((pred & (yy == 0)).sum())
                fn = int((~pred & (yy == 1)).sum())
                rows.append((f1_from_counts(tp, fp, fn), thr, tp, fp, fn))
            f1, thr, tp, fp, fn = max(rows)
        else:
            best = (-1.0, None, 0, 0, 0)
            for thr in np.unique(p):
                pred = p >= thr
                tp = int((pred & (yy == 1)).sum())
                fp = int((pred & (yy == 0)).sum())
                fn = int((~pred & (yy == 1)).sum())
                cand = (f1_from_counts(tp, fp, fn), float(thr), tp, fp, fn)
                if cand[0] > best[0]:
                    best = cand
            f1, thr, tp, fp, fn = best
        out[var] = {"threshold": thr, "val_F1": f1, "val_n": int(len(yy)),
                    "val_n_RI": int(yy.sum()), "tp": tp, "fp": fp, "fn": fn,
                    "note": "same frozen val scores, USA-30kt labels "
                            "(sensitivity only, not plotted)"}
    return out


def derive_thresholds(fig_key, data, val):
    """Operating rules for one figure; ``val`` carries the relabelled val set."""
    rules = {}

    if fig_key == "A":
        # our standard: native locked thresholds (USA-30kt, our protocol)
        rules["ours"] = [{"rule": "native inner-F1 (locked 0.225)", "family":
                          "locked", "score": "p_ours",
                          "threshold": data["ours_meta"]["threshold"]}]
        for name in BASELINES:
            rules[name] = [{"rule": "native inner-F1 (locked)",
                            "family": "locked", "score": f"p_{name}",
                            "threshold": data["baseline_thr"][name]}]
        rules["wrps1_own"] = [{"rule": "public rule pu>=0.5 (== pl>=0.215)",
                               "family": "public", "score": "wrps1_pl",
                               "threshold": 0.215}]
        rules["wrps1_f1"] = [{"rule": "inner-F1 on P (2018-19)",
                              "family": "rederived", "score": "wrps1_p",
                              "threshold": data["wrps1_inner_f1_usa"]}]
        picks = data["picks_kt30"]
        for var in V_KT30:
            pts = []
            if var in ("V1", "V2", "V3"):
                pts.append({"rule": "public rule 6/10 votes",
                            "family": "public", "score": f"ritcf_{var}",
                            "threshold": 0.6})
            pts.append({"rule": "val-F1 optimum (their val sweep)",
                        "family": "rederived", "score": f"ritcf_{var}",
                        "threshold": picks[var]["threshold"]})
            rules[f"ritcf_{var}"] = pts
        return rules

    # fig A control cell (CMA-30kt) and fig B (CMA-25.27): every model's
    # F1 threshold is re-derived on val/inner rows *under this label*.
    label_col = "y_cma30" if fig_key == "A_cma30" else "y_cma25"
    for key, score_col, inner_key in (
            [("ours", "p_ours", "ours")] +
            [(n, f"p_{n}", n) for n in BASELINES] +
            [("wrps1", "wrps1_p", None)]):
        if inner_key is None:
            p = val[score_col].to_numpy()
            tag = "val-F1 on P (re-derived)"
        else:
            p = val[inner_key].to_numpy()
            tag = "val-F1 re-derived under this label"
        y = val[label_col].to_numpy()
        thr = float(f1_threshold(p, y))
        if key == "wrps1":
            rules["wrps1_f1"] = [{"rule": tag, "family": "rederived",
                                  "score": score_col, "threshold": thr,
                                  "derivation_n": int(len(y)),
                                  "derivation_n_RI": int(y.sum())}]
            rules["wrps1_own"] = [{"rule": "public rule pu>=0.5 (== pl>=0.215)",
                                   "family": "public", "score": "wrps1_pl",
                                   "threshold": 0.215}]
        else:
            rules[key] = [{"rule": tag, "family": "rederived",
                           "score": score_col, "threshold": thr,
                           "derivation_n": int(len(y)),
                           "derivation_n_RI": int(y.sum())}]
    picks = data["picks_2527"] if fig_key == "B" else data["picks_kt30"]
    variants = V_2527 if fig_key == "B" else V_KT30
    for var in variants:
        pts = []
        if var in ("V1", "V2", "V3"):
            pts.append({"rule": "public rule 6/10 votes", "family": "public",
                        "score": f"ritcf_{var}", "threshold": 0.6})
        pts.append({"rule": "val-F1 optimum (their val sweep)",
                    "family": "rederived", "score": f"ritcf_{var}",
                    "threshold": picks[var]["threshold"]})
        rules[f"ritcf_{var}"] = pts
    return rules


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_figure(fig, data):
    scores, label_col = fig["scores"], fig["label_col"]
    y = scores[label_col].to_numpy().astype(int)
    rows = []
    for model, pts in fig["rules"].items():
        for pt in pts:
            p = scores[pt["score"]].to_numpy(dtype=float)
            ok = np.isfinite(p)
            if not ok.all():
                raise SystemExit(f"{fig['key']}: {model} has "
                                 f"{int((~ok).sum())} NaN scores")
            cm = classification_metrics(p, y, pt["threshold"])
            am = auc_metrics(p, y)
            iso = pod_at_farate(p, y)
            row = {"figure": fig["key"], "label_definition": fig["label"],
                   "window": fig["window"], "n": int(len(y)),
                   "n_RI": int(y.sum()), "base_rate": float(y.mean()),
                   "model": model, "model_name": PRETTY.get(model, model),
                   "rule": pt["rule"], "rule_family": pt["family"],
                   "threshold": float(pt["threshold"]),
                   "score_column": pt["score"], "tp": cm["tp"], "fp": cm["fp"],
                   "fn": cm["fn"], "tn": cm["tn"],
                   "POD": cm["POD"], "FARate": cm["FARate"],
                   "FARatio": cm["FARatio"], "CSI": cm["CSI"],
                   "precision": cm["precision"],
                   "ROC_AUC": am["ROC_AUC"], "PR_AUC": am["PR_AUC"],
                   "BSS_dev": bss(p, y, clim_rate=DEV_RATE),
                   "POD_at_FARate9.4": iso["POD_at_farate"],
                   "iso_nearest_FARate": iso["nearest_farate"],
                   "iso_nearest_POD": iso["nearest_pod"]}
            rows.append(row)
    return pd.DataFrame(rows)


def curve_models(fig):
    """Models whose full threshold sweep is drawn as a curve."""
    return [m for m in fig["rules"] if m.startswith("ritcf_")]


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def _caption(ax, text, width=104, size=6.3, y=-0.135):
    """Wrap the caption and put it under the axes (outside the data area)."""
    lines = []
    for raw in text.split("\n"):
        lines.extend(textwrap.wrap(raw, width=width) or [""])
    ax.text(0.0, y, "\n".join(lines), transform=ax.transAxes,
            fontsize=size, color="#444444", va="top", ha="left")


def draw_panel(fig, tab, xkey, title, caption, outfile, claim=False):
    """Panel with operating points + RITCF sweeps.

    ``xkey='farate'``  -> x = FARate, y = POD (detection view, ROC-like).
    ``xkey='faratio'`` -> x = FARate, y = FARatio = 1 - precision (alarm
    quality view: at the same false-alarm rate, how much of the alarm set is
    wrong).  The two panels share the x axis so they can be read side by side.
    """
    scores, label_col = fig["scores"], fig["label_col"]
    y = scores[label_col].to_numpy().astype(int)
    f, ax = plt.subplots(figsize=(7.3, 5.1))

    if xkey == "farate":
        def project(pod, farate, faratio):
            return farate * 100, pod * 100
        ylabel = "POD = TP/(TP+FN)  (%)"
        ylim = (0, 104)
    else:
        def project(pod, farate, faratio):
            return farate * 100, faratio * 100
        ylabel = "FARatio = FP/(TP+FP) = 1 - precision  (%)"
        ylim = (25, 104)
    xlim = (0, 100)

    handles, legend_labels = [], []
    for model in curve_models(fig):
        p = scores[fig["rules"][model][-1]["score"]].to_numpy(dtype=float)
        pod, farate, thr = pofd_curve(p, y)
        _p, faratio, _t = faratio_curve(p, y)
        xs, ys = project(pod, farate, faratio)
        keep = ((xs >= xlim[0]) & (xs <= xlim[1]) &
                (ys >= ylim[0]) & (ys <= ylim[1]))
        ln, = ax.plot(xs[keep], ys[keep], color=COLORS[model], lw=1.15,
                      alpha=0.9)
        handles.append(ln)
        legend_labels.append(f"{RITCF_NAMES[model]} sweep")

    labels = []
    n_public = n_rederived = 0
    for _, r in tab.iterrows():
        model = r["model"]
        col = COLORS[model]
        public = r["rule"].startswith("public")
        if model.startswith("ritcf_"):
            marker, size, lbl = ("D" if public else "v"), 40, None
            mfc = col if public else "none"
            n_public += int(public)
            n_rederived += int(not public)
        elif model.startswith("wrps1"):
            marker = "^" if public else "s"
            size, lbl, mfc = 52, PRETTY[model], col
        else:
            marker = "*" if model == "ours" else "o"
            size = 150 if model == "ours" else 46
            lbl, mfc = PRETTY[model], col
        x, yv = project(r["POD"], r["FARate"], r["FARatio"])
        ax.scatter(x, yv, marker=marker, s=size, color=col,
                   facecolors=mfc, edgecolors="white" if mfc != "none" else col,
                   linewidths=0.6 if mfc != "none" else 1.2, zorder=5)
        if lbl:
            handles.append(plt.Line2D([], [], marker=marker, ls="",
                                      color=col, markersize=5))
            legend_labels.append(lbl)

    if n_public:
        handles.append(plt.Line2D([], [], marker="D", ls="", color="grey",
                                  markersize=5))
        legend_labels.append("RITCF public rule (6/10 votes)")
    if n_rederived:
        handles.append(plt.Line2D([], [], marker="v", ls="", mfc="none",
                                  color="grey", markersize=5))
        legend_labels.append("RITCF val-F1 optimum (re-derived)")

    if claim:
        x, yv = project(CLAIM["pod"], CLAIM["farate"], CLAIM["faratio"])
        ax.scatter(x, yv, marker="X", s=170, facecolors="none",
                   edgecolors="black", linewidths=1.4, zorder=7)
        handles.append(plt.Line2D([], [], marker="X", ls="", mfc="none",
                                  color="black", markersize=7))
        legend_labels.append("RITCF paper claim 91.0/9.4/58.7 "
                             "(n=1,149 external anchor)")

    ax.axvline(FARATE_TARGET * 100, color="grey", ls=":", lw=1.0,
               zorder=1).set_gid("axes_fraction_line")
    ax.annotate(f"FARate = {FARATE_TARGET*100:.1f}%",
                (FARATE_TARGET * 100, 4), fontsize=6.8, color="grey",
                rotation=90, va="bottom", ha="right")
    ax.set_xlabel("FARate = POFD = FP/(FP+TN)  (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(title, fontsize=8.6, loc="left")
    _caption(ax, caption)
    ax.legend(handles, legend_labels, fontsize=6.3, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fb = _place_labels(ax, labels)
    f.subplots_adjust(left=0.09, right=0.60, top=0.91, bottom=0.40)
    checks = _save(f, outfile)
    checks["labels_unplaced_fallback"] = fb
    plt.close(f)
    return checks


def draw_curve_panel(fig, kind, outfile, title, caption):
    scores, label_col = fig["scores"], fig["label_col"]
    y = scores[label_col].to_numpy().astype(int)
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    f, ax = plt.subplots(figsize=(9.0, 5.6))
    for model in fig["rules"]:
        if model.startswith("wrps1_own"):
            continue
        col = COLORS[model]
        p = scores[fig["rules"][model][-1]["score"]].to_numpy(dtype=float)
        farate, pod, thr = pofd_curve(p, y)
        if kind == "roc":
            xs, ys = farate * 100, pod * 100
            lbl = f"{PRETTY[model]}  (AUC {auc_metrics(p, y)['ROC_AUC']:.3f})"
        else:
            rec = pod
            tp = pod * npos
            fp = farate * nneg
            prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 1.0)
            xs, ys = rec * 100, prec * 100
            lbl = (f"{PRETTY[model]}  "
                   f"(AP {auc_metrics(p, y)['PR_AUC']:.3f})")
        ax.plot(xs, ys, color=col, lw=1.2, alpha=0.9, label=lbl)
        # operating point marker
        for pt in fig["rules"][model]:
            r = classification_metrics(
                p, y, pt["threshold"])
            x = r["FARate"] * 100 if kind == "roc" else r["POD"] * 100
            yv = r["POD"] * 100 if kind == "roc" else r["precision"] * 100
            ax.scatter(x, yv, marker="*" if model == "ours" else "o", s=90 if
                       model == "ours" else 22, color=col, edgecolors="white",
                       linewidths=0.5, zorder=5)
    if kind == "roc":
        ax.plot([0, 100], [0, 100], ls=":", color="grey", lw=0.9)
        ax.set_xlabel("FARate = POFD (%)")
        ax.set_ylabel("POD (%)")
        ax.set_xlim(0, 100)
    else:
        ax.set_xlabel("Recall = POD (%)")
        ax.set_ylabel("Precision (%)")
        ax.set_xlim(0, 100)
    ax.set_ylim(0, 104)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.set_title(title, fontsize=12, loc="left")
    _caption(ax, caption, width=96, size=9.5, y=-0.24)
    ax.legend(fontsize=10, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    f.subplots_adjust(left=0.09, right=0.58, top=0.91, bottom=0.46)
    checks = _save(f, outfile)
    plt.close(f)
    return checks


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------
def print_table(tab, title):
    print(f"\n{title}")
    print(f"  n={int(tab['n'].iloc[0])}  n_RI={int(tab['n_RI'].iloc[0])}  "
          f"base_rate={tab['base_rate'].iloc[0]:.4f}  "
          f"label={tab['label_definition'].iloc[0]}")
    hdr = (f"  {'model':<26}{'rule':<34}{'thr':>9}  {'tp':>4}{'fp':>5}"
           f"{'fn':>4}{'tn':>5}  {'POD':>6}{'FARate':>7}{'FARatio':>8}"
           f"{'ROC':>6}{'PR':>6}{'BSS':>7}{'POD@9.4':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in tab.iterrows():
        print(f"  {PRETTY.get(r['model'], r['model'])[:25]:<26}"
              f"{r['rule'][:33]:<34}"
              f"{r['threshold']:>9.4f}  {r['tp']:>4}{r['fp']:>5}{r['fn']:>4}"
              f"{r['tn']:>5}  {r['POD']:>6.3f}{r['FARate']:>7.4f}"
              f"{r['FARatio']:>8.3f}{r['ROC_AUC']:>6.3f}{r['PR_AUC']:>6.3f}"
              f"{r['BSS_dev']:>7.3f}{r['POD_at_FARate9.4']:>8.3f}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Consistent-sample comparison engine; the RITCF rows are "
                    "read from one checkpoint-selection sweep.")
    ap.add_argument("--recon-kind", default=RECON["kind"],
                    choices=list(RECON_KINDS),
                    help="RITCF checkpoint selection to read (default: %(default)s)")
    ap.add_argument("--recon-tag-suffix", default=RECON["tag_suffix"],
                    help="run tag between the dataset part and the kind in "
                         "the sweep file names; '' walks back to the legacy "
                         "kind-less names (default: %(default)s)")
    ap.add_argument("--skip-figures", action="store_true",
                    help="write tables and the lock only")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw figures without changing tables or the lock")
    args = ap.parse_args(argv)
    suffix = args.recon_tag_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    RECON.update(kind=args.recon_kind, tag_suffix=suffix)
    global SKIP_FIGURES, SKIP_TABLES
    SKIP_FIGURES = bool(args.skip_figures)
    SKIP_TABLES = bool(args.figures_only)
    return args


def main(argv=None) -> int:
    parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"RITCF recon inputs: checkpoint kind = {RECON['kind']}, "
          f"tag suffix = {RECON['tag_suffix'] or '(none)'}")
    print("loading frozen assets ...")
    ours, ours_meta = load_ours()
    baselines, baseline_thr = load_baselines()
    ships, finite = load_ships()
    recon30 = load_recon("kt30")
    recon25 = load_recon("2527")
    fav_rows, fav = load_frame_availability()
    panel = load_panel()
    sweeps_kt30 = {v: load_sweep("kt30", v) for v in V_KT30}
    sweeps_2527 = {v: load_sweep("2527", v) for v in V_2527}
    w1 = wrps1_table(ships, finite)

    # ---- samples ---------------------------------------------------------
    sA = build_sample_A(ours, ships, finite, recon30)
    sB = build_sample_B(ours, ships, finite, recon25, fav_rows, panel)
    uidsA, uidsB = sA["uids"], sB["uids"]
    print(f"\nfig A sample n={len(uidsA)}   fig B sample n={len(uidsB)}")

    # labels per sample
    yA_usa = ours.set_index("row_uid").loc[uidsA, "y_usa"].to_numpy()
    r30 = recon30["test"].set_index("row_uid")
    r25 = recon25["test"].set_index("row_uid")
    yA_cma = r30.loc[uidsA, "ri"].to_numpy()
    yB_cma = r25.loc[uidsB, "ri"].to_numpy()
    yB_usa = ours.set_index("row_uid").loc[uidsB, "y_usa"].to_numpy()
    inter = int(((yA_usa == 1) & (yA_cma == 1)).sum())
    union = int(((yA_usa == 1) | (yA_cma == 1)).sum())
    jaccard = inter / max(union, 1)
    agree = float((yA_usa == yA_cma).mean())

    # ---- WRPS1 inner-F1 threshold (USA labels, native protocol) ----------
    inner_ours = load_inner("ours")
    inner_base = {n: load_inner(n) for n in BASELINES}
    wrps1_inner_f1 = float(f1_threshold(
        w1["scores"]["p"].reindex(inner_ours["row_uid"]).to_numpy(),
        inner_ours["y_usa"].to_numpy()))
    ours_inner_f1 = float(f1_threshold(inner_ours["p_inner"].to_numpy(),
                                       inner_ours["y_usa"].to_numpy()))
    base_inner_f1 = {n: float(f1_threshold(inner_base[n]["p_inner"].to_numpy(),
                                           inner_base[n]["y_usa"].to_numpy()))
                     for n in BASELINES}

    # ---- val derivation samples (for the relabelled figures) -------------
    inner_uid = set(inner_ours["row_uid"])
    ships_uid = set(ships["row_uid"].to_numpy()[finite])

    def val_block(recon, label_series, label_name):
        val_uids = [u for u in recon["val"]["row_uid"]
                    if u in inner_uid and u in ships_uid]
        idx = pd.Index(val_uids, name="row_uid")
        block = pd.DataFrame(index=idx)
        for n in ("ours",) + BASELINES:
            src = inner_ours if n == "ours" else inner_base[n]
            block[n] = src.set_index("row_uid")["p_inner"].loc[val_uids]\
                .to_numpy()
        block["wrps1_p"] = w1["scores"]["p"].reindex(idx).to_numpy()
        block[label_name] = label_series.reindex(idx).to_numpy().astype(int)
        block["y_usa"] = inner_ours.set_index("row_uid")["y_usa"]\
            .loc[val_uids].to_numpy()
        return block

    val_A = val_block(recon30, recon30["val"].set_index("row_uid")["ri"]
                      .astype(int), "y_cma30")
    val_B = val_block(recon25, recon25["val"].set_index("row_uid")["ri"]
                      .astype(int), "y_cma25")

    data = {"ours_meta": ours_meta, "baseline_thr": baseline_thr,
            "wrps1_inner_f1_usa": wrps1_inner_f1,
            "picks_kt30": ritcf_val_f1_picks("kt30"),
            "picks_2527": ritcf_val_f1_picks("2527")}

    # A variant whose score column had to fall back to the legacy sweep is
    # usually absent from the selected operating table as well; its val-F1
    # point then comes from the same legacy table.
    for tag, key, variants in (("kt30", "picks_kt30", V_KT30),
                               ("2527", "picks_2527", V_2527)):
        for var in variants:
            if var not in data[key]:
                data[key][var] = ritcf_val_f1_pick_fallback(tag, var)

    # ---- score matrices --------------------------------------------------
    scA = assemble(uidsA, ours, baselines, ships, finite, sweeps_kt30, w1)
    scA["y_cma30"] = yA_cma
    scB = assemble(uidsB, ours, baselines, ships, finite, sweeps_2527, w1)
    scB["y_cma25"] = yB_cma
    scB["y_usa"] = yB_usa

    figA = {"key": "A", "window": "test 2020-2024",
            "label": "USA-30kt (IBTrACS USA DV24>=30 kt)",
            "label_col": "y_usa", "scores": scA}
    figA_cma = {"key": "A_cma30", "window": "test 2020-2024",
                "label": "CMA-30kt (CMA DV24>=30 kt)",
                "label_col": "y_cma30", "scores": scA}
    figB = {"key": "B", "window": "test 2020-2021",
            "label": "CMA-25.27kt (CMA DV24>=25.27 kt)",
            "label_col": "y_cma25", "scores": scB}
    val_for = {"A": val_A, "A_cma30": val_A, "B": val_B}
    for fg in (figA, figA_cma, figB):
        fg["val"] = val_for[fg["key"]]
        fg["rules"] = derive_thresholds(fg["key"], data, fg["val"])

    tabA = evaluate_figure(figA, data)
    tabAC = evaluate_figure(figA_cma, data)
    tabB = evaluate_figure(figB, data)

    print_table(tabA, "FIG A · our standard · USA-30kt labels")
    print_table(tabAC, "FIG A control cell · same rows · CMA-30kt labels")
    print_table(tabB, "FIG B · their standard · CMA-25.27kt labels")

    # ---- waterfall + coverage -------------------------------------------
    print("\nwaterfall A:")
    for st in sA["stages"]:
        print("  " + json.dumps(st, ensure_ascii=False))
    print("\nwaterfall B:")
    for st in sB["stages"]:
        print("  " + json.dumps(st, ensure_ascii=False))
    print("\nfig B uncovered recon-test rows by reason:")
    for k, v in sorted(sB["uncovered_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {v:>4}  {k}")

    covA = coverage_block(uidsA, recon30["test"], "CMA-30kt (recon label)")
    covB = coverage_block(uidsB, recon25["test"],
                          "CMA-25.27kt (recon label)")
    print(f"\nfig A coverage (recon kt30 test): RI rows "
          f"{covA['ri_rows_covered']}/{covA['ri_rows_total']}, storms "
          f"{covA['storms_covered']}/{covA['storms_total']}")
    print(f"fig B coverage (recon 25.27 test): RI rows "
          f"{covB['ri_rows_covered']}/{covB['ri_rows_total']}, storms "
          f"{covB['storms_covered']}/{covB['storms_total']}")

    # ---- self checks ----------------------------------------------------
    checks = {}

    def pod_of(tab, model, rule_prefix):
        sel = tab[(tab.model == model) &
                  (tab.rule.str.startswith(rule_prefix))]
        return None if sel.empty else float(sel.POD.iloc[0])

    o = tabA[(tabA.model == "ours")].iloc[0]
    checks["figA_ours_native_POD"] = float(o["POD"])
    checks["figA_ours_native_POD_vs_official_0.68"] = float(o["POD"] - 0.68)
    checks["ours_inner_f1_reproduces_locked_0.225"] = ours_inner_f1
    checks["baseline_inner_f1_vs_locked"] = {
        n: {"derived": base_inner_f1[n], "locked": baseline_thr[n],
            "delta": base_inner_f1[n] - baseline_thr[n]} for n in BASELINES}
    checks["wrps1_inner_f1_usa"] = wrps1_inner_f1
    checks["figB_ours_threshold_not_transplanted"] = {
        "figA_locked": ours_meta["threshold"],
        "figB_rederived": [p["threshold"] for p in figB["rules"]["ours"]]}
    checks["ritcf_610_POD_figA_USA"] = {
        m: pod_of(tabA, m, "public") for m in curve_models(figA)}
    checks["ritcf_610_POD_figA_CMA"] = {
        m: pod_of(tabAC, m, "public") for m in curve_models(figA_cma)}
    checks["ritcf_610_POD_USA_lt_CMA"] = {
        m: (None if checks["ritcf_610_POD_figA_CMA"][m] is None else
            bool(checks["ritcf_610_POD_figA_USA"][m] <
                 checks["ritcf_610_POD_figA_CMA"][m]))
        for m in checks["ritcf_610_POD_figA_USA"]}
    checks["ritcf_valF1_POD_figA_USA"] = {
        m: pod_of(tabA, m, "val-F1") for m in curve_models(figA)}
    checks["jaccard_USA_vs_CMA_RI_rows_figA"] = jaccard
    checks["label_agreement_figA"] = agree
    checks["n_RI_figA_USA"] = int(yA_usa.sum())
    checks["n_RI_figA_CMA30"] = int(yA_cma.sum())
    checks["n_RI_figB_CMA25"] = int(yB_cma.sum())
    checks["n_figA"] = len(uidsA)
    checks["n_figB"] = len(uidsB)
    checks["figB_rows_subset_of_figA"] = bool(set(uidsB) <= set(uidsA))
    checks["figB_val_derivation"] = {"n": int(len(val_B)),
                                     "n_RI": int(val_B["y_cma25"].sum())}
    checks["figA_cma_val_derivation"] = {"n": int(len(val_A)),
                                         "n_RI": int(val_A["y_cma30"].sum())}
    # cross-check: fig A subset vs the official 3,004-row table
    official = {}
    for key, json_name in (("ours", "main_2024"),
                           ("baseline_lr", "baseline_lr"),
                           ("baseline_lda", "baseline_lda"),
                           ("baseline_ships12_lda", "baseline_ships12_lda"),
                           ("baseline_ships12_lda_cohc",
                            "baseline_ships12_lda_cohc")):
        m = json.loads((DATA / "analysis" / "eval" /
                        f"{json_name}.json").read_text(encoding="utf-8")
                       )["metrics"]["all"]
        row = tabA[tabA.model == key].iloc[0]
        official[key] = {
            k: {"official": round(m[k], 4), "figA_subset": round(float(row[k]), 4),
                "delta": round(float(row[k]) - m[k], 4)}
            for k in ("POD", "FARate", "ROC_AUC", "PR_AUC", "BSS_dev")}
    checks["figA_subset_vs_official_3004"] = official
    checks["figA_subset_max_abs_delta_POD"] = max(
        abs(v["POD"]["delta"]) for v in official.values())

    # the operating table and the per-row sweep of one variant must describe
    # the same val split, otherwise score and threshold come from two
    # different runs
    pick_consistency = {}
    for tag, key, sweeps, variants in (
            ("kt30", "picks_kt30", sweeps_kt30, V_KT30),
            ("2527", "picks_2527", sweeps_2527, V_2527)):
        for var in variants:
            pick = data[key][var]
            val = sweeps[var][1]
            ok = (pick["val_n"] == len(val) and
                  pick["val_n_RI"] == int(val["y_recon"].sum()))
            pick_consistency[f"{tag}:{var}"] = {
                "table_source": pick["source"],
                "val_n": pick["val_n"], "sweep_val_n": int(len(val)),
                "val_n_RI": pick["val_n_RI"],
                "sweep_val_n_RI": int(val["y_recon"].sum()),
                "val_split_matches": bool(ok)}
    checks["ritcf_val_f1_pick_vs_sweep_val_split"] = pick_consistency
    checks["ritcf_val_f1_pick_val_split_all_match"] = all(
        v["val_split_matches"] for v in pick_consistency.values())
    # the sweep carries its own labels while the engine takes the RI label
    # from the recon dataset directory; the two must agree row by row
    label_mismatch = {}
    for tag, sweeps, recon in (("kt30", sweeps_kt30, recon30),
                               ("2527", sweeps_2527, recon25)):
        ri = recon["all"].set_index("row_uid")["ri"]
        for var, (test, _val) in sweeps.items():
            y = test.set_index("row_uid")["y_recon"]
            pair = pd.concat([y.rename("sweep"),
                              ri.reindex(y.index).rename("dataset")], axis=1)
            label_mismatch[f"{tag}:{var}"] = int(
                (pair["sweep"] != pair["dataset"]).sum())
    checks["ritcf_sweep_label_vs_dataset_mismatches"] = label_mismatch
    checks["ritcf_sweep_inputs"] = RECON_INPUTS
    checks["ritcf_sweep_input_warnings"] = RECON_WARNINGS

    # RITCF val-F1 operating point re-read under USA-30kt (sensitivity only)
    inner_map = dict(zip(inner_ours["row_uid"], inner_ours["y_usa"]))
    sens_usa = ritcf_val_f1_usa_relabel(sweeps_kt30, inner_map)
    checks["ritcf_val_f1_label_sensitivity_kt30"] = sens_usa

    print("\nself-checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    # ---- iso_pofd --------------------------------------------------------
    iso_rows = []
    for fg, tab in ((figA, tabA), (figA_cma, tabAC), (figB, tabB)):
        for _, r in tab.iterrows():
            iso_rows.append({"figure": fg["key"], "label": fg["label"],
                             "model": r["model"], "rule": r["rule"],
                             "POD_at_FARate9.4": r["POD_at_FARate9.4"],
                             "nearest_FARate": r["iso_nearest_FARate"],
                             "nearest_POD": r["iso_nearest_POD"]})
    iso = pd.DataFrame(iso_rows)
    print(f"\niso operating point table (POD at FARate = "
          f"{FARATE_TARGET*100:.1f}%, interpolated):")
    for fig_label in ("A", "A_cma30", "B"):
        sub = iso[iso.figure == fig_label]
        print(f"  [{fig_label}] {sub.label.iloc[0]}")
        for model, g in sub.groupby("model", sort=False):
            for _, r in g.iterrows():
                print(f"    {PRETTY.get(model, model):<24}"
                      f"{r['rule'][:32]:<34}"
                      f"POD@9.4%={r['POD_at_FARate9.4']:.3f}  "
                      f"(nearest {r['nearest_FARate']*100:5.2f}% -> "
                      f"POD {r['nearest_POD']:.3f})")

    # ---- CSVs ------------------------------------------------------------
    cols = ["figure", "label_definition", "window", "n", "n_RI", "base_rate",
            "model", "model_name", "rule", "rule_family", "threshold",
            "tp", "fp", "fn", "tn", "POD", "FARate", "FARatio", "CSI",
            "precision", "ROC_AUC", "PR_AUC", "BSS_dev", "POD_at_FARate9.4"]
    wf = []
    for name, s in (("A", sA), ("B", sB)):
        for st in s["stages"]:
            wf.append({"figure": name, **st})
    cov_rows = [{"figure": "A", "n_RI_eval_label": int(yA_usa.sum()), **covA},
                {"figure": "B", "n_RI_eval_label": int(yB_cma.sum()), **covB}]
    for k, v in sB["uncovered_reasons"].items():
        cov_rows.append({"figure": "B", "label": f"uncovered:{k}",
                         "rows": v})
    written = []
    for name, frame in (("table_figA.csv", pd.concat([tabA, tabAC])[cols]),
                        ("table_figB.csv", tabB[cols]),
                        ("iso_pofd.csv", iso),
                        ("waterfall.csv", pd.DataFrame(wf)),
                        ("coverage.csv", pd.DataFrame(cov_rows))):
        target = OUT / out_name(name)
        if not SKIP_TABLES:
            frame.to_csv(target, index=False)
            written.append(target.name)

    # ---- figures ---------------------------------------------------------
    print("\nfigures:")
    layout = {}
    nA, riA, riC = len(uidsA), int(yA_usa.sum()), int(yA_cma.sum())
    capA = (f"Same {nA} rows for every model (recon-kt30 test 2020-2024 x our "
            f"frozen test rows x SHIPS-12 all-finite); label = USA-30kt "
            f"({riA} RI rows). Curves = RITCF kt30 variants reswept on these "
            f"rows; diamonds = their 6/10 rule, triangles = their val-F1 "
            f"optimum.\n"
            f"This is a 2,178-row subset of our official 3,004-row test table "
            f"(the 714 dropped rows are absent from the recon dataset in every "
            f"split, i.e. their CMA-wind eligibility); the official "
            f"POD/FARate stay in the main table. Wind-source gap on the same "
            f"rows: USA+ {riA} vs CMA+ {riC} RI rows, Jaccard {jaccard:.2f}, "
            f"label agreement {agree*100:.1f}% — the CMA-trained RITCF "
            f"variants face a higher base rate here "
            f"({riA/nA*100:.1f}% vs {riC/nA*100:.1f}%).")
    layout["fig_compare_pofd.png"] = draw_panel(
        figA, tabA, "farate",
        f"Fig A — our standard · test 2020-2024 · label USA-30kt "
        f"· n={nA} (RI {riA})", capA, OUT / "fig_compare_pofd.png")
    layout["fig_compare_faratio.png"] = draw_panel(
        figA, tabA, "faratio",
        f"Fig A — our standard · test 2020-2024 · label USA-30kt "
        f"· n={nA} (RI {riA})",
        capA + "\nx = FARate, y = FARatio = 1 - precision: at the same "
        "false-alarm rate a lower curve has the cleaner alarm set; the dotted "
        "line is the paper's 9.4% FARate.",
        OUT / "fig_compare_faratio.png")
    capC = (f"Label-swap control cell: the same {nA} rows and the same model "
            f"scores, only the label changes to CMA-30kt ({riC} RI rows). "
            f"Our/baseline/WRPS1 thresholds are re-derived on val 2018-19 "
            f"rows under CMA-30kt labels; RITCF keeps its own val sweep.")
    layout["fig_compare_cma_pofd.png"] = draw_panel(
        figA_cma, tabAC, "farate",
        f"Fig A control — same rows, label CMA-30kt · test 2020-2024 "
        f"· n={nA} (RI {riC})", capC, OUT / "fig_compare_cma_pofd.png")

    nB, riB = len(uidsB), int(yB_cma.sum())
    capB = (f"Shared sample: recon 25.27kt test 2020-2021 (1,057) x "
            f"our frozen test rows (954) x SHIPS-12 all-finite -> n={nB} "
            f"(RI {riB}; RI-row coverage {covB['ri_rows_covered']}/"
            f"{covB['ri_rows_total']}, storms "
            f"{covB['storms_covered']}/{covB['storms_total']}); the 103 "
            f"uncovered rows have no locked prediction and were NOT "
            f"re-inferred.\n"
            f"All thresholds re-derived on val 2018-19 under CMA-25.27kt. "
            f"CVRI-Ens and the statistical baselines use USA-30kt labels in fit "
            f"2000-2017 and inner 2018-2019; WRPS1 uses its published "
            f"2002-2021 development window. RITCF V1's reference pool is "
            f"rolling inside the test years (their protocol). RI count is only {riB}: the "
            f"binomial s.d. of POD is about "
            f"{100*np.sqrt(0.25/riB):.0f} pt (95% +-{200*np.sqrt(0.25/riB):.0f}"
            f" pt), so smaller differences are not resolvable.")
    layout["fig_compare_pofd_2021.png"] = draw_panel(
        figB, tabB, "farate",
        f"Fig B — their standard · test 2020-2021 · label CMA-25.27kt "
        f"· n={nB} (RI {riB})", capB,
        OUT / "fig_compare_pofd_2021.png", claim=True)
    layout["fig_compare_faratio_2021.png"] = draw_panel(
        figB, tabB, "faratio",
        f"Fig B — their standard · test 2020-2021 · label CMA-25.27kt "
        f"· n={nB} (RI {riB})", capB,
        OUT / "fig_compare_faratio_2021.png", claim=True)

    capPR = (f"Fig A rows (n={nA}, USA-30kt); every model with a continuous "
             f"score. Markers = operating points (RITCF: 6/10 and val-F1; "
             f"WRPS1: inner-F1 on P). The WRPS1 pu>=0.5 rule lives on the pl "
             f"scale and is drawn in the operating panels only. RITCF variants "
             f"are their kt30 recon sweeps re-scored on these rows; WRPS1 uses "
             f"its published formula on SHIPS-12.")
    layout["fig_compare_roc.png"] = draw_curve_panel(
        figA, "roc", OUT / "fig_compare_roc.png",
        f"ROC · Fig A rows · test 2020-2024 · label USA-30kt · "
        f"n={nA} (RI {riA})", capPR)
    layout["fig_compare_pr.png"] = draw_curve_panel(
        figA, "pr", OUT / "fig_compare_pr.png",
        f"Precision-recall · Fig A rows · test 2020-2024 · label "
        f"USA-30kt · n={nA} (RI {riA})", capPR)
    # keys of the layout block are the file names actually written
    layout = {c["file"]: c for c in layout.values()}
    written.extend(n for n, c in layout.items() if c.get("saved", True))

    print("\nfigure layout self-checks (legend must sit outside the axes, no "
          "artist outside the axes):")
    for name, c in layout.items():
        print(f"  {name}: legend_outside={c.get('legend_outside_axes')} "
              f"artists_outside={c.get('artists_outside_axes')} "
              f"texts_inside={c.get('texts_inside_figure')} "
              f"label_fallback={c.get('labels_unplaced_fallback', 0)}")

    # ---- lock ------------------------------------------------------------
    lock = {
        "kind": "comparison_secondary",
        "protocol": "holdout_masked_v1",
        "recon_checkpoint_selection": {
            "kind": RECON["kind"],
            "tag_suffix": RECON["tag_suffix"] or None,
            "requested": (f"data/ritcf_recon/sweep_<V>[_kt30]"
                          f"{RECON['tag_suffix']}_{RECON['kind']}.npz"),
            "inputs": RECON_INPUTS,
            "warnings": RECON_WARNINGS,
        },
        "analysis_class": (
            "secondary: subset and/or relabelled evaluation of FROZEN "
            "predictions; no training, no new inference"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": {"script": "scripts/06_compare/compare_models_consistent.py",
                   "spec": "docs/COMPARISON.md",
                   "design": "docs/SM_MAP.md",
                   "sha256": hashlib.sha256(
                       Path(__file__).read_bytes()).hexdigest()},
        "row_key_hash_rule": "sha256 of '\\n'.join(sorted(row_uid))",
        "figures": {
            "A": {
                "window": figA["window"],
                "label_definition": figA["label"],
                "sample_definition":
                    "recon dataset_kt30 test rows ∩ our frozen test rows ∩ "
                    "SHIPS-12 (12 predictors all finite) rows",
                "n": nA, "n_RI": riA, "base_rate": riA / nA,
                "row_key_sha256": row_hash(uidsA),
                "official_protocol_rows": 3004,
                "relation_to_official":
                    "subset of the official 3,004-row test table; official "
                    "headline numbers stay in the main table",
            },
            "A_cma30_control_cell": {
                "window": figA_cma["window"],
                "label_definition": figA_cma["label"],
                "sample_definition": "identical rows to figure A",
                "n": nA, "n_RI": riC, "base_rate": riC / nA,
                "row_key_sha256": row_hash(uidsA),
            },
            "B": {
                "window": figB["window"],
                "label_definition": figB["label"],
                "sample_definition":
                    "recon dataset (25.27 kt) test rows ∩ our frozen test "
                    "rows ∩ SHIPS-12 all finite",
                "n": nB, "n_RI": riB, "base_rate": riB / nB,
                "row_key_sha256": row_hash(uidsB),
            },
        },
        "waterfall": {"A": sA["stages"], "B": sB["stages"],
                      "B_uncovered_reasons": sB["uncovered_reasons"]},
        "coverage": {"A": covA, "B": covB,
                     "B_uncovered_reason_counts":
                         sB["uncovered_reasons"]},
        "wind_source_decomposition": {
            "figure": "A",
            "n_RI_USA30": int(yA_usa.sum()),
            "n_RI_CMA30": int(yA_cma.sum()),
            "jaccard_RI_rows": jaccard,
            "label_agreement": agree,
            "note": "same rows, two wind sources; the CMA-labelled subset "
                    "carries a much lower base rate, so CMA-trained models "
                    "face a distribution shift in figure A"},
        "thresholds": {
            "policy": ("public/locked operating rule for the model's own "
                       "protocol + an F1 point re-derived under the figure's "
                       "label on val/inner rows (2018-19); figure B never "
                       "reuses the USA-30kt 0.225"),
            "figures": {
                "A": figA["rules"],
                "A_cma30_control_cell": figA_cma["rules"],
                "B": figB["rules"],
            },
            "derivation_samples": {
                "A_native": {"ours_inner_n": int(len(inner_ours)),
                             "ours_inner_n_RI": int(
                                 inner_ours["y_usa"].sum()),
                             "rule": "locked thresholds reused as-is"},
                "A_cma30_val": {"n": int(len(val_A)),
                                "n_RI": int(val_A["y_cma30"].sum())},
                "B_val": {"n": int(len(val_B)),
                          "n_RI": int(val_B["y_cma25"].sum())},
                "ritcf_val_sweeps": {
                    "A_and_A_cma30": "the kt30 recon operating table "
                                     "(varying suffix, val split; CMA-30kt "
                                     "labels, their protocol)",
                    "B": "the 25.27kt recon operating table (val split; "
                         "CMA-25.27kt labels)"},
            },
            "ritcf_val_f1_picks": {"kt30": data["picks_kt30"],
                                   "25.27": data["picks_2527"]},
            "ritcf_val_f1_label_sensitivity_kt30_usa": sens_usa,
            "special_cases": (
                "RITCF V4 / V4direct are distance and probability heads; only "
                "the val-F1 point exists for them (no 6/10 vote rule). In "
                "figure A the RITCF val-F1 point is the one threshold that is "
                "NOT label-matched: the spec takes it from their own val sweep, "
                "whose labels are CMA-30kt. Their frozen val scores are "
                "relabelable, so the sensitivity block "
                "ritcf_val_f1_label_sensitivity_kt30_usa prices that choice; "
                "the plotted point stays with the spec (their own val labels)."),
        },
        "labels": {
            "A": "USA-30kt = our locked label (IBTrACS USA DV24>=30 kt)",
            "A_cma30": "CMA-30kt: recon dataset_kt30 ri, same rows",
            "B": "CMA-25.27kt: recon dataset ri (their definition)",
        },
        "metrics": ["POD", "FARate (POFD)", "FARatio (=1-precision)", "CSI",
                    "precision", "ROC_AUC", "PR_AUC", "BSS_dev",
                    "POD_at_FARate9.4"],
        "bss_reference": {"clim_rate": DEV_RATE,
                          "note": "fixed development-period rate, never "
                                  "re-estimated per sample"},
        "iso_operating_point": {"farate": FARATE_TARGET,
                                "method": "linear interpolation of the "
                                          "empirical POD-FARate staircase"},
        "paper_claim_anchor": CLAIM,
        "artifacts": sorted(written),
        "figures_written": not SKIP_FIGURES,
        "figure_layout_checks": layout,
        "declarations": [
            "Secondary analysis only: our probabilities are the frozen "
            "holdout predictions, evaluated on a subset of rows and/or "
            "against a swapped label; no new forward pass, once-only "
            "protocol intact.",
            "Every number inside one figure uses one row set, one label, one "
            "window and one metric recipe; model input/training differences "
            "are the object of comparison.",
            "Figure A is a 3-way-intersection subset of the official 3,004-"
            "row test table; the official headline numbers are unchanged and "
            "remain in the main table.",
            "Figure B thresholds are re-derived under CMA-25.27kt on val "
            "2018-19 rows; the USA-30kt 0.225 threshold is not transplanted.",
            "CVRI-Ens and the statistical baselines are evaluated across label "
            "sources in figure B: USA-30kt fit-period labels (2000-2017), "
            "inner-period labels (2018-2019), and CMA-25.27kt evaluation labels "
            "(2020-2021). RITCF uses CMA labels and is evaluated across label "
            "sources in figure A.",
            "WRPS1's final coefficients and probability mapping were fitted on "
            "2002-2021, overlapping figure B's two evaluation years and two "
            "years of figure A and the 2,892-row comparison.",
            "RITCF variant V1 uses a rolling reference pool that admits "
            "within-test-year labels; this is the model's own protocol, "
            "carried here unchanged and flagged wherever V1 is read.",
            "The paper claim (POD 91.0 / FARate 9.4 / FARatio 58.7, n=1,149) "
            "appears as an open external anchor only: different sample, "
            "different label pipeline, different evaluation code; it is not "
            "connected to any curve.",
            "Figure B has only 66 RI rows: the binomial s.d. of POD is about "
            "6 points at POD 0.5 (95% +-12 points), so smaller differences "
            "are not resolvable and no significance is claimed for them.",
            "BSS uses the fixed development-period climatology 0.0791 as the "
            "reference rate.",
        ],
        "self_checks": checks,
        "tables": {key: f"data/analysis/compare/{out_name(name)}"
                   for key, name in (
                       ("figA", "table_figA.csv"),
                       ("figB", "table_figB.csv"),
                       ("iso_pofd", "iso_pofd.csv"),
                       ("waterfall", "waterfall.csv"),
                       ("coverage", "coverage.csv"))},
    }
    lock_path = LOCK_DIR / out_name("comparison_secondary.json")
    if not SKIP_TABLES:
        lock_path.write_text(json.dumps(lock, indent=1, ensure_ascii=False,
                                        default=float), encoding="utf-8")
        print(f"\nwrote {lock_path.relative_to(ROOT)}")
    print(f"wrote {OUT.relative_to(ROOT)}/: " + ", ".join(sorted(written)))
    if RECON_WARNINGS:
        print("recon input warnings:")
        for w in RECON_WARNINGS:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
