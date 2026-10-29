"""Evaluate one trained RITCF-recon variant (spec §7).

What is computed, in this order:

1. **pair threshold** -- the Part-C distance is binarised on the *validation*
   pair table (label 0 = RI-RI, 1 = cross-class); the distance threshold
   maximises pair-level F1.  The paper gives no pair-level threshold.
2. **evaluation split** -- each row of the split is paired with the 10 nearest
   RI samples that were already known at its forecast time, the reference pool
   being the variant's choice (``rolling`` keeps test-year RIs, ``capped_2019``
   drops them).  Votes ``d < threshold``; >= 60 % of the available references
   -> RI.  V4 skips the vote and uses the mean distance as the score; V4direct
   uses ``sigmoid(logit)``.
3. **scores** -- POD, FARate (POFD), FARatio (1-precision), the BSS-equivalent
   (0/1 Brier against the split's own base rate, plus the soft Brier when a
   probability exists), ROC-AUC / PR-AUC, a per-year breakdown, and the sample
   frame next to the paper's 6795 / 1472 / 1149.

Nothing here reads ``data/test`` or ``protocol_locks``; the test years are the
IBTrACS 2020-2021 rows of this reconstruction's own panel, and the test split is
scored exactly once per variant.

Usage (from the release root):

    set KMP_DUPLICATE_LIB_OK=TRUE
    python -m scripts.ritcf_recon.infer_eval --variant V1
    python -m scripts.ritcf_recon.infer_eval --variant V1 --tag smoke --eval-split val
    python -m scripts.ritcf_recon.infer_eval --all
    python -m scripts.ritcf_recon.infer_eval --all --tag kt30_aucfix \\
        --checkpoint-kind best_loss best_auc last

Checkpoint selection: ``--checkpoint-kind`` picks the file written by
``train.py`` -- ``best_loss`` (``checkpoint.pt``, the historical validation-loss
argmin), ``best_auc`` (``checkpoint_auc.pt``, the validation ROC-AUC argmax) or
``last`` (``checkpoint_last.pt``).  Give several kinds to score the same test
split once per checkpoint; every artifact carries the kind in its name
(``eval_best_auc.json``), and ``best_loss`` additionally keeps the historical
``eval.json`` alias.

Outputs: ``runs{_<tag>}/<variant>/eval[_<kind>].json`` + ``.md``, and for
``--all`` ``data/ritcf_recon/summary[_<tag>]_<kind>.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

P2_ROOT = Path(__file__).resolve().parents[2]
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))

from scripts.ritcf_recon import config as C  # noqa: E402
from scripts.ritcf_recon import model as M  # noqa: E402
from scripts.ritcf_recon import pairs as P  # noqa: E402
from scripts.ritcf_recon.features import FeatureStore  # noqa: E402

PAPER = {"POD": 91.0, "FARate": 9.4, "FARatio": 58.7,
         "note": "reanalysis version, test 2020-2021 (Table 1)"}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def contingency(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y).astype(bool)
    p = np.asarray(pred).astype(bool)
    return {"tp": int(np.sum(y & p)), "fp": int(np.sum(~y & p)),
            "fn": int(np.sum(y & ~p)), "tn": int(np.sum(~y & ~p))}


def scores(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    c = contingency(y, pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    base = float(np.mean(y)) if n else float("nan")
    out = {
        "n": n, "n_ri": tp + fn, "base_rate": base, **c,
        "POD": tp / max(tp + fn, 1),
        "FARate": fp / max(fp + tn, 1),
        "FARatio": fp / max(tp + fp, 1),
        "BSS_hard": 1.0 - float(np.mean((np.asarray(pred, dtype=float) - y) ** 2)) /
                    max(base * (1.0 - base), 1e-9) if n else float("nan"),
    }
    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        out["BSS_soft"] = 1.0 - float(np.mean((prob - y) ** 2)) / \
            max(base * (1.0 - base), 1e-9)
        if len(np.unique(y)) > 1:
            out["ROC_AUC"] = float(roc_auc_score(y, prob))
            out["PR_AUC"] = float(average_precision_score(y, prob))
        else:
            out["ROC_AUC"] = float("nan")
            out["PR_AUC"] = float("nan")
    return out


def f1_score_of(pred: np.ndarray, y: np.ndarray) -> float:
    c = contingency(y, pred)
    return 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1e-9)


def pick_pair_threshold(dist: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """Distance threshold maximising pair-level F1 (positive = label 0)."""
    positive = label < 0.5
    lo, hi = float(np.min(dist)), float(np.max(dist))
    best_t, best_f1 = 0.5, -1.0
    for thr in np.linspace(lo, hi, 201):
        f1 = f1_score_of(dist < thr, positive)
        if f1 > best_f1:
            best_f1, best_t = f1, float(thr)
    return best_t, best_f1


def pick_score_threshold(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for thr in np.linspace(float(np.min(prob)), float(np.max(prob)), 201):
        f1 = f1_score_of(prob >= thr, y)
        if f1 > best_f1:
            best_f1, best_t = f1, float(thr)
    return best_t, best_f1


def pick_vote_threshold_by_sample_f1(dist2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pair threshold that maximises *sample-level* F1 after the 6/10 vote.

    Alternative to the spec's pair-level F1 rule (``--threshold-rule sample_f1``),
    because a pair-F1 threshold sits at a very high-POD / high-FAR corner once
    it is combined with a 6-of-10 vote.  Both are selected on validation only.
    """
    per_row = np.isfinite(dist2).sum(axis=1)
    need = np.maximum(1, np.ceil(C.VOTE_MIN_FRACTION * np.maximum(per_row, 1)))
    finite = dist2[np.isfinite(dist2)]
    lo, hi = float(finite.min()), float(finite.max())
    best_t, best_f1 = 0.5, -1.0
    for thr in np.linspace(lo, hi, 201):
        votes = np.nansum(dist2 < thr, axis=1)
        f1 = f1_score_of(votes >= need, y)
        if f1 > best_f1:
            best_f1, best_t = f1, float(thr)
    return best_t, best_f1


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def load_model(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = M.build_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def checkpoint_path(variant: str, tag: str,
                    kind: str = C.DEFAULT_CHECKPOINT_KIND,
                    explicit: str | Path | None = None) -> Path:
    """File of one (variant, selection criterion) written by ``train.py``.

    ``best_loss`` -> ``checkpoint.pt``, ``best_auc`` -> ``checkpoint_auc.pt``,
    ``last`` -> ``checkpoint_last.pt``.  V2 shares V1's trained weights (only
    the reference pool differs), so it falls back to the V1 run directory for
    the same criterion.
    """
    if explicit:
        return Path(explicit)
    name = C.CHECKPOINT_FILES[C.checkpoint_kind(kind)]
    path = C.run_dir(variant, tag) / name
    if not path.is_file() and variant == "V2":
        path = C.run_dir("V1", tag) / name
    return path


def checkpoint_marker(kind: str, explicit: str | Path | None = None) -> str:
    """Filename marker carried by every artifact scored from one checkpoint."""
    return "custom" if explicit else C.checkpoint_kind(kind)


@torch.no_grad()
def pair_distances(model: torch.nn.Module, store: FeatureStore,
                   a_idx: np.ndarray, b_idx: np.ndarray, device: torch.device,
                   batch: int = 64) -> np.ndarray:
    out = np.empty(len(a_idx), dtype=np.float32)
    for start in range(0, len(a_idx), batch):
        sl = slice(start, start + batch)
        a = store.batch(a_idx[sl], device)
        b = store.batch(b_idx[sl], device)
        out[sl] = model(a, b).detach().cpu().numpy().reshape(-1)
    return out


@torch.no_grad()
def sample_probs(model: torch.nn.Module, store: FeatureStore, rows: np.ndarray,
                 device: torch.device, batch: int = 64) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch):
        sl = slice(start, start + batch)
        x = store.batch(rows[sl], device)
        out[sl] = torch.sigmoid(model(x)).detach().cpu().numpy().reshape(-1)
    return out


def score_with_refs(model, store, target_idx: np.ndarray, refs: np.ndarray,
                    device: torch.device, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Per-target mean distance and vote fraction over its known-RI references."""
    n, k = refs.shape
    flat_t = np.repeat(np.arange(n), k)
    flat_r = refs.reshape(-1)
    valid = flat_r >= 0
    dist = np.full(n * k, np.nan, dtype=np.float32)
    if valid.any():
        d = pair_distances(model, store, flat_r[valid], target_idx[flat_t[valid]],
                           device, batch=batch)
        dist[valid] = d
    dist2 = dist.reshape(n, k)
    with np.errstate(invalid="ignore"):
        mean_d = np.nanmean(dist2, axis=1) if valid.any() else np.full(n, np.nan)
    return dist2, mean_d


def evaluate_split(model, store, split: str, pool_kind: str, device: torch.device,
                   pair_thr: float, sample_thr: float | None,
                   direct: bool) -> dict:
    """Score one split; returns raw per-row arrays plus the operating-point metadata.

    ``raw`` is the monotone score a probability can be calibrated from: the vote
    fraction, ``1 - mean distance``, or the sigmoid probability.  It is not a
    probability in the first two cases, so the calibrated Brier is filled in by
    :func:`finalize` once a validation-fitted calibrator exists.
    """
    samples = store.samples
    target_mask = (samples["split"] == split).to_numpy()
    target_idx = np.flatnonzero(target_mask)
    if target_idx.size == 0:
        return {"n": 0, "note": f"no rows in split {split!r}"}
    y = samples["ri"].to_numpy()[target_idx].astype(int)

    meta = {"split": split, "reference_pool": pool_kind,
            "seasons": samples["season"].to_numpy()[target_idx]}
    if direct:
        prob = sample_probs(model, store, target_idx, device)
        thr = sample_thr if sample_thr is not None else 0.5
        pred = (prob >= thr).astype(int)
        raw = prob
        meta.update({"threshold": float(thr), "agg": "direct", "calibrated": False})
    else:
        ref_mask = P.pool_mask(samples, pool_kind)
        refs = P.nearest_refs(samples, target_mask, ref_mask, C.TEST_REFS)
        per_row = (refs >= 0).sum(axis=1)
        dist2, mean_d = score_with_refs(model, store, target_idx, refs, device)
        if sample_thr is None:  # 6/10 vote
            votes = np.nansum((dist2 < pair_thr).astype(float), axis=1)
            need = np.maximum(1, np.ceil(C.VOTE_MIN_FRACTION * np.maximum(per_row, 1)))
            pred = (votes >= need).astype(int)
            raw = votes / np.maximum(per_row, 1)
            meta.update({"threshold": float(pair_thr), "agg": "vote",
                         "vote_min_fraction": C.VOTE_MIN_FRACTION,
                         "mean_votes": float(votes.mean()), "calibrated": True})
        else:  # mean pair distance used directly as the score
            pred = (mean_d < sample_thr).astype(int)
            raw = np.clip(1.0 - mean_d, 0.0, 1.0)
            meta.update({"threshold": float(sample_thr), "agg": "mean_distance",
                         "calibrated": True})
        meta.update({"mean_refs": float(per_row.mean()),
                     "frac_full_refs": float((per_row == C.TEST_REFS).mean()),
                     "mean_distance": float(np.nanmean(mean_d))})
    meta.update({"y": y, "pred": pred, "raw": np.asarray(raw, dtype=float)})
    return meta


def fit_calibrator(raw: np.ndarray, y: np.ndarray):
    """Validation-fitted Platt scaling: raw score -> P(RI).  None if degenerate."""
    ok = np.isfinite(raw)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return None
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(raw[ok].reshape(-1, 1), y[ok])
    return lambda z: lr.predict_proba(np.asarray(z, dtype=float).reshape(-1, 1))[:, 1]


def finalize(rec: dict, calibrator) -> dict:
    """Fill the metric block of one split (calibrated probability if available)."""
    y = rec.get("y")
    if y is None or np.asarray(y).size == 0:
        rec["n"] = 0
        return rec
    pred, raw = rec["pred"], rec["raw"]
    n = int(y.size)
    if calibrator is not None:
        finite = np.isfinite(raw)
        fill = float(np.nanmean(raw[finite])) if finite.any() else 0.0
        prob = calibrator(np.where(finite, raw, fill))
        prob = np.where(np.isfinite(prob), prob, float(np.mean(y)))
        prob_source = "val_platt"
    elif rec.get("agg") == "direct":
        prob = raw
        prob_source = "model_sigmoid"
    else:
        prob = np.clip(raw, 0.0, 1.0)   # monotone, not calibrated: AUC/PR are valid
        prob_source = "raw"
    out = scores(y, pred, prob)
    base = max(float(np.mean(y)) * (1 - float(np.mean(y))), 1e-9)
    out["BSS_soft_raw"] = 1.0 - float(np.mean((np.clip(raw, 0, 1) - y) ** 2)) / base
    out["prob_source"] = prob_source
    out.update({k: v for k, v in rec.items() if k not in ("y", "pred", "raw", "seasons")})
    out["n"] = n
    by_year = {}
    seasons = rec["seasons"]
    for season in sorted(set(seasons.tolist())):
        m = seasons == season
        sub = scores(y[m], pred[m], prob[m] if prob is not None else None)
        by_year[str(int(season))] = sub
    out["by_year"] = by_year
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def format_row(name: str, m: dict) -> str:
    def pct(key):
        v = m.get(key)
        return "  n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:5.1f}"
    bss = m.get("BSS_hard")
    bss_s = "  n/a" if bss is None else f"{bss:+.3f}"
    bss_soft = m.get("BSS_soft")
    bss_soft_s = "" if bss_soft is None else f" (soft {bss_soft:+.3f})"
    auc = m.get("ROC_AUC")
    auc_s = "  n/a" if auc is None or (isinstance(auc, float) and math.isnan(auc)) else f"{auc:.3f}"
    pr = m.get("PR_AUC")
    pr_s = "  n/a" if pr is None or (isinstance(pr, float) and math.isnan(pr)) else f"{pr:.3f}"
    return (f"{name:16s} n={m.get('n', 0):5d} RI={m.get('n_ri', 0):4d} "
            f"POD={pct('POD')} FARate={pct('FARate')} FARatio={pct('FARatio')} "
            f"BSS={bss_s}{bss_soft_s} AUC={auc_s} PR={pr_s}")


def write_eval(run_dir: Path, payload: dict, marker: str = C.DEFAULT_CHECKPOINT_KIND) -> None:
    """Write ``eval_<marker>.json/.md``; ``best_loss`` also keeps ``eval.*``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# RITCF-recon evaluation -- {payload['variant']}",
             "",
             f"checkpoint [{payload['checkpoint_kind']}]: `{payload['checkpoint']}`  ",
             f"checkpoint epoch {payload.get('checkpoint_epoch')}, val loss "
             f"{payload.get('checkpoint_val_loss')}, val AUC "
             f"{payload.get('checkpoint_val_auc')}  ",
             f"dataset: `{payload['dataset']['dataset_dir']}`  ",
             f"seed {payload['seed']}, operating point fixed on the validation split "
             f"(rule `{payload['thresholds'].get('rule')}`, "
             f"threshold {payload['thresholds']['pair_or_sample']:.4f})",
             "",
             "| set | n | RI | POD % | FARate % | FARatio % | BSS | BSS soft | "
             "ROC-AUC | PR-AUC |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def fmt(v, spec="{:.3f}"):
        return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else spec.format(v)

    for split in ("val", "test"):
        m = payload["metrics"].get(split)
        if not m or not m.get("n"):
            continue
        lines.append(
            f"| {split} | {m['n']} | {m['n_ri']} | {100 * m['POD']:.1f} | "
            f"{100 * m['FARate']:.1f} | {100 * m['FARatio']:.1f} | "
            f"{m['BSS_hard']:+.3f} | {fmt(m.get('BSS_soft'), '{:+.3f}')} | "
            f"{fmt(m.get('ROC_AUC'))} | {fmt(m.get('PR_AUC'))} |")
        for year, y_m in m.get("by_year", {}).items():
            lines.append(
                f"| {split} {year} | {y_m['n']} | {y_m['n_ri']} | {100 * y_m['POD']:.1f} | "
                f"{100 * y_m['FARate']:.1f} | {100 * y_m['FARatio']:.1f} | "
                f"{y_m['BSS_hard']:+.3f} | {fmt(y_m.get('BSS_soft'), '{:+.3f}')} | - | - |")
    lines += ["",
              "BSS is the 0/1 Brier score against the split's own base rate; the soft "
              "column uses the score itself (Platt-calibrated on validation for the "
              "test split, raw on validation).",
              "",
              f"paper (reanalysis, test 2020-2021): POD {PAPER['POD']:.1f} / "
              f"FARate {PAPER['FARate']:.1f} / FARatio {PAPER['FARatio']:.1f}; "
              f"their test frame 1149 rows (78 RI).",
              ""]
    stems = [f"eval_{marker}"] + (["eval"] if marker == C.DEFAULT_CHECKPOINT_KIND else [])
    for stem in stems:
        (run_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2),
                                              encoding="utf-8")
        (run_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def evaluate(args) -> dict:
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    v = C.variant(args.variant)
    dataset_dir = C.dataset_dir(args.dataset_tag if args.dataset_tag is not None else args.tag)
    store = FeatureStore(dataset_dir)
    run_dir = C.run_dir(v.name, args.tag)
    kind_arg = getattr(args, "checkpoint_kind", C.DEFAULT_CHECKPOINT_KIND)
    if isinstance(kind_arg, (list, tuple)):
        raise SystemExit("evaluate() takes one checkpoint kind; main() loops over "
                         "the --checkpoint-kind values")
    kind = C.checkpoint_kind(kind_arg)
    explicit = getattr(args, "checkpoint", None)
    marker = checkpoint_marker(kind, explicit)
    checkpoint = checkpoint_path(v.name, args.tag, kind, explicit)
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {checkpoint} [{marker}]; train this "
                         f"variant with the current train.py or pick another "
                         f"--checkpoint-kind")
    model, ckpt = load_model(checkpoint, device)
    ck_variant = ckpt.get("variant")
    if ck_variant and ck_variant != v.name and v.name != "V2":
        raise SystemExit(f"checkpoint variant {ck_variant} != requested {v.name}")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(
        encoding="utf-8"))
    if (ckpt.get("dataset_fingerprint")
            and ckpt["dataset_fingerprint"] != manifest.get("fingerprint")):
        raise SystemExit("checkpoint was trained on a different dataset "
                         "fingerprint than the current manifest")
    print(f"{v.name}: {v.description}")
    print(f"checkpoint [{marker}] {checkpoint.name} epoch {ckpt.get('epoch')} "
          f"val loss {ckpt.get('val_loss')} val AUC {ckpt.get('val_auc')}")

    pair_thr, pair_f1 = 0.5, float("nan")
    sample_thr = None
    thresholds = {"pair_f1": None,
                  "rule": (args.threshold_rule if not v.direct_classifier
                           else "sample_f1")}
    if not v.direct_classifier:
        # validation rows against their own known-RI references (10 nearest)
        val_mask = (store.samples["split"] == "val").to_numpy()
        val_idx = np.flatnonzero(val_mask)
        val_refs = P.nearest_refs(store.samples, val_mask,
                                  P.pool_mask(store.samples, v.pool), C.TEST_REFS)
        val_dist2, val_mean_d = (score_with_refs(model, store, val_idx, val_refs, device)
                                 if val_idx.size else (np.zeros((0, C.TEST_REFS)),
                                                       np.zeros(0)))
        val_y = store.samples["ri"].to_numpy()[val_idx].astype(int)
        val_pairs = pd.read_parquet(dataset_dir / "pairs_val.parquet")
        if len(val_pairs) == 0 or val_idx.size == 0:
            print("WARNING: empty validation data (smoke build); using threshold 0.5")
        elif args.threshold_rule == "sample_f1":
            pair_thr, pair_f1 = pick_vote_threshold_by_sample_f1(val_dist2, val_y)
            print(f"pair threshold {pair_thr:.4f} (val sample F1 {pair_f1:.3f}, "
                  f"n={val_idx.size})")
            thresholds["pair_f1"] = pair_f1
        else:
            d = pair_distances(model, store, val_pairs["a_idx"].to_numpy(),
                               val_pairs["b_idx"].to_numpy(), device)
            pair_thr, pair_f1 = pick_pair_threshold(d, val_pairs["label"].to_numpy())
            print(f"pair threshold {pair_thr:.4f} (val pair F1 {pair_f1:.3f}, "
                  f"n={len(d)})")
            thresholds["pair_f1"] = pair_f1
        thresholds["pair"] = pair_thr
        if v.aggregation == "mean_distance":
            ok = np.isfinite(val_mean_d)
            sample_thr, sample_f1 = pick_score_threshold(-val_mean_d[ok], val_y[ok])
            sample_thr = -sample_thr
            thresholds["sample_f1"] = sample_f1
            print(f"sample threshold {sample_thr:.4f} (val F1 {sample_f1:.3f})")
    else:
        val_rows = np.flatnonzero((store.samples["split"] == "val").to_numpy())
        if val_rows.size:
            p = sample_probs(model, store, val_rows, device)
            y_val = store.samples["ri"].to_numpy()[val_rows]
            sample_thr, sample_f1 = pick_score_threshold(p, y_val)
            thresholds["sample_f1"] = sample_f1
            print(f"sample threshold {sample_thr:.3f} (val F1 {sample_f1:.3f})")
        else:
            sample_thr = 0.5

    metrics = {}
    raw_recs = {}
    splits = ("val",) if args.eval_split == "val" else ("val", "test")
    for split in splits:
        rec = evaluate_split(model, store, split, v.pool, device, pair_thr,
                             sample_thr if v.aggregation == "mean_distance" else None,
                             v.direct_classifier)
        raw_recs[split] = rec
    if "test" not in raw_recs:
        print("eval split = val only (test rows untouched)")
    elif raw_recs["test"].get("y") is None:
        print("WARNING: the test split is empty (smoke build?); "
              "score the validation split instead with --eval-split val")

    # probabilities are calibrated on the validation split only
    calibrator = None
    val_rec = raw_recs["val"]
    if val_rec.get("y") is not None and val_rec.get("calibrated"):
        calibrator = fit_calibrator(val_rec["raw"], val_rec["y"])
        print(f"validation Platt calibration: "
              f"{'fitted' if calibrator else 'degenerate, reporting raw Brier only'}")
    for split, rec in raw_recs.items():
        # the calibrator is fitted on val, so val itself keeps the raw score
        m = finalize(rec, calibrator if split == "test" else None)
        metrics[split] = m
        print(format_row(split, m) if m.get("n") else f"{split}: no rows")
        for year, ym in (m.get("by_year") or {}).items():
            if ym.get("n"):
                print("  " + format_row(f"  {split} {year}", ym))

    frame = {"ours": {}, "paper": {"train": 6795, "val": 1472, "test": 1149,
                                   "n_ri": {"train": 303, "val": 110, "test": 78}}}
    for split in ("train", "val", "test"):
        s = store.samples[store.samples["split"] == split]
        n = int(len(s))
        n_ri = int(s["ri"].sum()) if n else 0
        frame["ours"][split] = {"n": n, "n_ri": n_ri,
                                "base_rate": (n_ri / n) if n else None}
    frame["paper"]["base_rate"] = {k: v / frame["paper"][k]
                                   for k, v in frame["paper"]["n_ri"].items()}
    print(f"sample frame ours: {frame['ours']}")
    print(f"sample frame paper: {frame['paper']}")
    payload = {
        "schema": "ritcf_recon_eval_v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": v.name,
        "variant_spec": C.variant_spec(v.name),
        "checkpoint": str(checkpoint),
        "checkpoint_kind": marker,
        "checkpoint_selection": ckpt.get("selection"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
        "checkpoint_val_auc": ckpt.get("val_auc"),
        "seed": seed,
        "dataset": store.provenance(),
        "dataset_fingerprint": ckpt.get("dataset_fingerprint"),
        "row_key_sha256": ckpt.get("row_key_sha256"),
        "thresholds": {"pair_or_sample": (sample_thr if sample_thr is not None
                                          else pair_thr), **thresholds},
        "metrics": metrics,
        "sample_frame": frame,
        "paper": PAPER,
        "eval_split": args.eval_split,
        "args": vars(args),
    }
    write_eval(run_dir, payload, marker)
    print(f"wrote {run_dir / f'eval_{marker}.json'}")
    return payload


def write_summary(out: dict, tag: str, marker: str, threshold_rule: str,
                  seed: int) -> Path:
    """One `--all` summary table (``summary[_<tag>]_<kind>.md``)."""
    lines = ["| variant | split | n | RI | POD % | FARate % | FARatio % | BSS | "
             "BSS soft | AUC | PR-AUC |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, payload in out.items():
        for split in ("val", "test"):
            m = payload["metrics"].get(split) or {}
            if not m.get("n"):
                continue
            soft = m.get("BSS_soft")
            soft_s = ("-" if soft is None or
                      (isinstance(soft, float) and math.isnan(soft))
                      else f"{soft:+.3f}")
            auc = m.get("ROC_AUC")
            auc_s = ("-" if auc is None or
                     (isinstance(auc, float) and math.isnan(auc)) else f"{auc:.3f}")
            pr = m.get("PR_AUC")
            pr_s = ("-" if pr is None or
                    (isinstance(pr, float) and math.isnan(pr)) else f"{pr:.3f}")
            lines.append(
                f"| {name} | {split} | {m['n']} | {m['n_ri']} | "
                f"{100 * m['POD']:.1f} | {100 * m['FARate']:.1f} | "
                f"{100 * m['FARatio']:.1f} | {m['BSS_hard']:+.3f} | "
                f"{soft_s} | {auc_s} | {pr_s} |")
    lines += ["",
              f"| paper (reanalysis 2020-2021) | test | 1149 | 78 | "
              f"{PAPER['POD']:.1f} | {PAPER['FARate']:.1f} | "
              f"{PAPER['FARatio']:.1f} | -0.485 | - | - | - |",
              "",
              f"checkpoint selection: `{marker}` "
              f"(files `{C.CHECKPOINT_FILES.get(marker, 'explicit path')}`); "
              f"threshold rule: `{threshold_rule}`; base seed {seed}."]
    path = C.OUT_ROOT / f"summary_{tag or 'main'}_{marker}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None, help=f"one of {sorted(C.VARIANTS)}")
    parser.add_argument("--all", action="store_true", help="evaluate every trained variant")
    parser.add_argument("--tag", default="", help="run tag")
    parser.add_argument("--dataset-tag", default=None,
                        help="dataset tag when it differs from the run tag "
                             "(defaults to --tag)")
    parser.add_argument("--checkpoint", default=None,
                        help="override the checkpoint path (one file, kind `custom`)")
    parser.add_argument("--checkpoint-kind", nargs="+", default=None,
                        choices=sorted(C.CHECKPOINT_FILES), metavar="KIND",
                        help="which selection criterion to score: best_loss "
                             "(checkpoint.pt, default), best_auc "
                             "(checkpoint_auc.pt), last (checkpoint_last.pt); "
                             "several values score each checkpoint in turn")
    parser.add_argument("--eval-split", default="test", choices=("test", "val"),
                        help="split to score (test is the paper's protocol; val is "
                             "for smoke builds with no 2020-2021 rows)")
    parser.add_argument("--threshold-rule", default="pair_f1",
                        choices=("pair_f1", "sample_f1"),
                        help="pair_f1 (spec §5: val pair-level F1) or sample_f1 "
                             "(pair threshold maximising val sample F1 after voting)")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    kinds = args.checkpoint_kind or [C.DEFAULT_CHECKPOINT_KIND]
    if args.checkpoint and len(kinds) > 1:
        raise SystemExit("--checkpoint names a single file; drop --checkpoint-kind")

    if args.all:
        for kind in kinds:
            out = {}
            for name in C.VARIANTS:
                ckpt = checkpoint_path(name, args.tag, kind, args.checkpoint)
                if not ckpt.is_file():
                    print(f"skip {name}: no checkpoint {ckpt.name}")
                    continue
                sub = argparse.Namespace(**vars(args))
                sub.variant = name
                sub.all = False
                sub.checkpoint_kind = kind
                out[name] = evaluate(sub)
            if out:
                marker = checkpoint_marker(kind, args.checkpoint)
                path = write_summary(out, args.tag, marker, args.threshold_rule,
                                     args.seed)
                print(f"\nwrote {path}")
        return 0

    if not args.variant:
        raise SystemExit("--variant or --all is required")
    for kind in kinds:
        sub = argparse.Namespace(**vars(args))
        sub.checkpoint_kind = kind
        evaluate(sub)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    raise SystemExit(main())
