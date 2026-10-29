"""Operating-point sweep for the RITCF reconstruction variants.

Computes per-sample scores on val and test once per variant, then sweeps the
decision knob (vote fraction k/10 for vote variants; continuous score
threshold for V4 / V4direct) and tabulates POD / FARate / FARatio at every
operating point.  Read-only with respect to training: checkpoints are loaded,
nothing is refit.  Deterministic given the checkpoint.

``--checkpoint-kind`` selects the file written by ``train.py`` (``best_loss``,
``best_auc`` or ``last``); every artifact carries the kind in its name so the
three operating-point tables can sit side by side.

Outputs: data/ritcf_recon/sweep_<variant>_<tag|main>_<kind>.npz (per-row scores)
and data/ritcf_recon/sweep_table_<tag|main>_<kind>.csv (variant x knob x split
metrics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

P2 = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(P2))

from scripts.ritcf_recon import config as C  # noqa: E402
from scripts.ritcf_recon import pairs as P  # noqa: E402
from scripts.ritcf_recon.features import FeatureStore  # noqa: E402
from scripts.ritcf_recon.infer_eval import (checkpoint_marker, checkpoint_path,
                                            load_model, pair_distances,
                                            pick_pair_threshold,
                                            sample_probs, score_with_refs)
# noqa: E402


def metrics_at(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {"POD": tp / (tp + fn) if tp + fn else float("nan"),
            "FARate": fp / (fp + tn) if fp + tn else float("nan"),
            "FARatio": fp / (tp + fp) if tp + fp else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sweep_variant(name: str, tag: str, dataset_tag: str | None, device: str,
                  kind: str = C.DEFAULT_CHECKPOINT_KIND,
                  explicit: str | Path | None = None) -> list[dict]:
    v = C.variant(name)
    ds_tag = dataset_tag if dataset_tag is not None else tag
    store = FeatureStore(C.dataset_dir(ds_tag))
    marker = checkpoint_marker(kind, explicit)
    ckpt_path = checkpoint_path(v.name, tag, kind, explicit)
    if not ckpt_path.is_file():
        # same behaviour as ``infer_eval --all``: an arm without this checkpoint
        # is skipped, the rest of the sweep still runs
        print(f"skip {name} [{marker}]: no checkpoint {ckpt_path.name} "
              f"in {ckpt_path.parent}")
        return []
    model, ckpt = load_model(ckpt_path, device=torch.device(device))
    manifest = json.loads((store.dir / "manifest.json").read_text(encoding="utf-8"))
    if (ckpt.get("dataset_fingerprint")
            and ckpt["dataset_fingerprint"] != manifest.get("fingerprint")):
        raise SystemExit(f"checkpoint {ckpt_path} was trained on a different "
                         f"dataset fingerprint than {store.dir}/manifest.json; "
                         f"retrain it against this dataset")
    for split in ("val", "test"):
        n = int((store.samples["split"] == split).sum())
        if n == 0:
            raise SystemExit(f"dataset {store.dir} has no {split} rows; the sweep "
                             f"needs both splits (build a dataset with val and "
                             f"test seasons)")
    print(f"checkpoint [{marker}] {ckpt_path.name} epoch {ckpt.get('epoch')}  "
          f"val loss {ckpt.get('val_loss')} val AUC {ckpt.get('val_auc')}")

    rows = []
    scores = {}
    for split in ("val", "test"):
        mask = (store.samples["split"] == split).to_numpy()
        idx = np.flatnonzero(mask)
        y = store.samples["ri"].to_numpy()[idx].astype(int)
        row_uids = store.samples["row_key"].astype(str).to_numpy()[idx] \
            if "row_key" in store.samples.columns else None
        scores[split] = {"y": y, "row_uid": row_uids}
        if v.direct_classifier:
            raw = sample_probs(model, store, idx, torch.device(device))
            knobs = np.quantile(raw, np.linspace(0.02, 0.98, 49))
            for thr in knobs:
                m = metrics_at(y, raw >= thr)
                m.update({"variant": name, "split": split, "knob": "prob_thr",
                          "value": float(thr)})
                rows.append(m)
        else:
            refs = P.nearest_refs(store.samples, mask,
                                  P.pool_mask(store.samples, v.pool),
                                  C.TEST_REFS)
            dist2, mean_d = score_with_refs(model, store, idx, refs,
                                            torch.device(device))
            if v.aggregation == "mean_distance":
                raw = -mean_d
                finite = np.isfinite(raw)
                knobs = np.quantile(raw[finite], np.linspace(0.02, 0.98, 49))
                for thr in knobs:
                    m = metrics_at(y[finite], raw[finite] >= thr)
                    m.update({"variant": name, "split": split,
                              "knob": "mean_dist_thr", "value": float(thr)})
                    rows.append(m)
            else:
                val_pairs = pd.read_parquet(C.dataset_dir(ds_tag) /
                                            "pairs_val.parquet")
                d_val = pair_distances(model, store,
                                       val_pairs["a_idx"].to_numpy(),
                                       val_pairs["b_idx"].to_numpy(),
                                       torch.device(device))
                pair_thr, _ = pick_pair_threshold(d_val,
                                                  val_pairs["label"].to_numpy())
                votes = np.isfinite(dist2) & (dist2 < pair_thr)
                n_refs = np.isfinite(dist2).sum(axis=1)
                frac = votes.sum(axis=1) / np.maximum(n_refs, 1)
                frac[n_refs == 0] = np.nan
                for k in range(1, C.TEST_REFS + 1):
                    frac_level = k / C.TEST_REFS
                    need = np.ceil(frac_level * n_refs).astype(int)
                    need = np.maximum(need, 1)
                    pred = votes.sum(axis=1) >= need
                    pred[n_refs == 0] = False
                    m = metrics_at(y, pred)
                    m.update({"variant": name, "split": split,
                              "knob": "vote_frac", "value": frac_level,
                              "pair_thr": float(pair_thr)})
                    rows.append(m)
        scores[split]["y"] = y
        if v.direct_classifier:
            scores[split]["score"] = raw
        elif v.aggregation == "mean_distance":
            scores[split]["score"] = -mean_d
        else:
            scores[split]["score"] = frac
    out_npz = C.OUT_ROOT / f"sweep_{name}_{tag or 'main'}_{marker}.npz"
    np.savez_compressed(out_npz,
                        **{f"{k}_{f}": v[f] for k, v in scores.items()
                           for f in ("y", "score", "row_uid") if f in v
                           and v[f] is not None})
    print(f"sweep scores -> {out_npz.name}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*",
                    default=["V1", "V2", "V3", "V4", "V4direct"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset-tag", default=None)
    ap.add_argument("--checkpoint-kind", default=C.DEFAULT_CHECKPOINT_KIND,
                    choices=sorted(C.CHECKPOINT_FILES),
                    help="checkpoint.pt (best_loss), checkpoint_auc.pt "
                         "(best_auc) or checkpoint_last.pt (last)")
    ap.add_argument("--checkpoint", default=None,
                    help="override the checkpoint path (kind `custom`)")
    ap.add_argument("--device",
                    default="cuda" if __import__("torch").cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    marker = checkpoint_marker(args.checkpoint_kind, args.checkpoint)
    rows = []
    for name in args.variants:
        print(f"=== sweep {name} [{marker}]", flush=True)
        rows.extend(sweep_variant(name, args.tag, args.dataset_tag,
                                  args.device, args.checkpoint_kind,
                                  args.checkpoint))
    table = pd.DataFrame(rows)
    if table.empty:
        print("no variant produced a sweep (every checkpoint missing); "
              "no table written")
        return 0
    out = Path(args.out) if args.out else (
        C.OUT_ROOT / f"sweep_table_{args.tag or 'main'}_{marker}.csv")
    table.to_csv(out, index=False)
    print(f"sweep table -> {out} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
