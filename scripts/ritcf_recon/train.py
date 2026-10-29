"""Train one RITCF-recon variant.

Protocol (spec §4-5, §8):

* Adam, lr 1e-3, MSE on the Part-C distance (BCE for the direct control),
  batch 256 *pairs*, fixed 60-epoch budget (``--epochs 60 --patience 60``, so
  the loss-based early stop never fires), seed 2025.
* The validation set is used for checkpoint selection only; nothing here looks
  at the test years.
* 8 GB VRAM: the effective batch stays 256, but the micro-batch is probed at
  start-up (or capped with ``--max-micro-batch``) and the shortfall is covered
  by gradient accumulation.  The probe result is written to the run manifest.

Checkpoints: every epoch is scored twice on the validation pairs, by the loss
and by the ROC-AUC of the pair distance.  Three files are kept, all from the
same trajectory:

* ``checkpoint.pt``      argmin of the validation pair loss (historical rule)
* ``checkpoint_auc.pt``  argmax of the validation ROC-AUC (audit fix: the
                         downstream analysis only uses the pair-distance
                         *ranking*, and the MSE simultaneously penalises
                         ranking errors and calibration drift, so its argmin
                         lands on the first epoch -- an undertrained model)
* ``checkpoint_last.pt`` the final epoch trained (epoch 60 of the fixed budget,
                         or wherever ``--patience`` cut it), for the last-epoch
                         reference row

Model structure, loss, optimiser and hyper-parameters are untouched: the three
checkpoints come from the same run, so the comparison isolates the selection
criterion.  The stopping rule is the fixed 60-epoch budget used for the reported
runs (``--epochs 60 --patience 60``), so the AUC argmax cannot be truncated by
a loss-based early stop; ``--patience`` still works for cheap probes.

Usage (from the release root):

    set KMP_DUPLICATE_LIB_OK=TRUE
    python -m scripts.ritcf_recon.train --variant V1 --epochs 60 --patience 60
    python -m scripts.ritcf_recon.train --variant V3 --tag smoke --epochs 2

Outputs: ``data/ritcf_recon/runs{_<tag>}/<variant>/{checkpoint.pt,
checkpoint_auc.pt, checkpoint_last.pt, history.json, run_manifest.json}``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

P2_ROOT = Path(__file__).resolve().parents[2]
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))

from scripts.ritcf_recon import config as C  # noqa: E402
from scripts.ritcf_recon import model as M  # noqa: E402
from scripts.ritcf_recon.features import FeatureStore  # noqa: E402


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class PairBatch(Dataset):
    """(known RI, judged sample, label) rows of a prebuilt pair table."""

    def __init__(self, store: FeatureStore, pairs: pd.DataFrame) -> None:
        self.store = store
        self.a_idx = pairs["a_idx"].to_numpy(dtype=np.int64)
        self.b_idx = pairs["b_idx"].to_numpy(dtype=np.int64)
        self.label = pairs["label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return self.label.size

    def __getitem__(self, i: int):
        out = {}
        for side, idx in (("a", self.a_idx[i]), ("b", self.b_idx[i])):
            for key, value in self.store.raw_batch(np.asarray([idx])).items():
                out[f"{side}_{key}"] = value[0]
        return out, np.float32(self.label[i])


class DirectBatch(Dataset):
    """Single samples with their RI label (V4 direct control)."""

    def __init__(self, store: FeatureStore, rows: np.ndarray) -> None:
        self.store = store
        self.rows = np.asarray(rows, dtype=np.int64)
        self.label = store.samples["ri"].to_numpy(dtype=np.float32)[self.rows]

    def __len__(self) -> int:
        return self.rows.size

    def __getitem__(self, i: int):
        batch = {k: v[0] for k, v in self.store.raw_batch(self.rows[i:i + 1]).items()}
        return batch, np.float32(self.label[i])


def collate_pair(items):
    batch, labels = zip(*items)
    out = {key: np.stack([b[key] for b in batch]) for key in batch[0]}
    return out, torch.from_numpy(np.asarray(labels, dtype=np.float32)).reshape(-1, 1)


def collate_direct(items):
    batch, labels = zip(*items)
    out = {key: np.stack([b[key] for b in batch]) for key in batch[0]}
    return out, torch.from_numpy(np.asarray(labels, dtype=np.float32)).reshape(-1, 1)


def split_pair(dict_batch) -> tuple[dict, dict]:
    a = {k[2:]: v for k, v in dict_batch.items() if k.startswith("a_")}
    b = {k[2:]: v for k, v in dict_batch.items() if k.startswith("b_")}
    return a, b


# --------------------------------------------------------------------------
# Batching / memory
# --------------------------------------------------------------------------
def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def probe_micro_batch(model: nn.Module, store: FeatureStore, pairs: pd.DataFrame,
                      device: torch.device, direct: bool, start: int,
                      floor: int = 2) -> int:
    """Largest micro-batch that survives one forward+backward pass."""
    if device.type != "cuda":
        return start
    criterion = nn.MSELoss() if not direct else nn.BCEWithLogitsLoss()
    b = min(start, len(pairs))
    while b >= floor:
        try:
            if direct:
                rows = pairs["b_idx"].to_numpy()[:b]
                x = store.batch(rows, device)
                out = model(x)
                target = torch.zeros_like(out)
            else:
                a_idx = pairs["a_idx"].to_numpy()[:b]
                b_idx = pairs["b_idx"].to_numpy()[:b]
                a = store.batch(a_idx, device)
                bb = store.batch(b_idx, device)
                out = model(a, bb)
                target = torch.zeros_like(out)
            loss = criterion(out, target)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del loss, out, target
            torch.cuda.empty_cache()
            return b
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_oom(exc):
                raise
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            b //= 2
    raise SystemExit("even the smallest micro-batch does not fit; lower the model size")


# --------------------------------------------------------------------------
# Epoch
# --------------------------------------------------------------------------
def run_epoch(model: nn.Module, loader: DataLoader, criterion, device: torch.device,
              direct: bool, store: FeatureStore, optimizer=None, accum: int = 1) -> dict:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    preds, labels = [], []
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    steps_since_step = 0
    for raw_batch, y in loader:
        y = y.to(device)
        with torch.set_grad_enabled(training):
            if direct:
                out = model(store.normalize(raw_batch, device))
            else:
                a_raw, b_raw = split_pair(raw_batch)
                out = model(store.normalize(a_raw, device),
                            store.normalize(b_raw, device))
            loss = criterion(out, y)
            if training:
                (loss / accum).backward()
                steps_since_step += 1
                if steps_since_step == accum:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps_since_step = 0
        total += float(loss.detach()) * y.shape[0]
        count += int(y.shape[0])
        shown = torch.sigmoid(out) if direct else out
        preds.append(shown.detach().float().cpu().numpy().reshape(-1))
        labels.append(y.detach().float().cpu().numpy().reshape(-1))
    if training and steps_since_step:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    pred = np.concatenate(preds) if preds else np.zeros(0)
    lab = np.concatenate(labels) if labels else np.zeros(0)
    return {"loss": total / max(count, 1), "pred": pred, "label": lab}


def pair_diagnostics(pred: np.ndarray, label: np.ndarray, thr: float = 0.5,
                     higher_is_positive: bool = False) -> dict:
    """Monitoring only: POD / false-alarm rate at a fixed 0.5 threshold.

    The positive class depends on the arm.  For the siamese models ``label`` 0
    is the RI-RI pair and ``pred`` is the Part-C distance (RI when *below* the
    threshold); for the direct control ``label`` 1 is RI and ``pred`` is the
    sigmoid probability (RI when *above* it).
    """
    true_ri = label >= 0.5 if higher_is_positive else label < 0.5
    pred_ri = pred >= thr if higher_is_positive else pred < thr
    tp = int(np.sum(true_ri & pred_ri))
    fn = int(np.sum(true_ri & ~pred_ri))
    fp = int(np.sum(~true_ri & pred_ri))
    tn = int(np.sum(~true_ri & ~pred_ri))
    return {"POD": tp / max(tp + fn, 1), "FARate": fp / max(fp + tn, 1),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def pair_auc(pred: np.ndarray, label: np.ndarray,
             higher_is_positive: bool = False) -> float:
    """ROC-AUC of the pair decision (the criterion ``checkpoint_auc`` uses).

    For the siamese models ``pred`` is the Part-C distance and the positive
    class is label 0 (RI-RI), so the RI-likelihood score is ``-pred``; for the
    direct control ``pred`` is the sigmoid probability and the positive class is
    label 1.  Threshold-free, hence blind to the calibration drift that drags
    the MSE upwards while the ordering keeps improving.
    """
    pred = np.asarray(pred, dtype=float)
    label = np.asarray(label, dtype=float)
    ok = np.isfinite(pred)
    if ok.sum() < 2:
        return float("nan")
    y = (label[ok] >= 0.5).astype(int) if higher_is_positive \
        else (label[ok] < 0.5).astype(int)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, pred[ok] if higher_is_positive else -pred[ok]))


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(path: Path, *, cfg: dict, model: nn.Module, variant: str,
                    epoch: int, val_loss: float, val_auc: float, selection: str,
                    seed: int, args, provenance: dict, fingerprint, row_key_sha) -> None:
    """One checkpoint file tagged with the criterion it is the argmax/argmin of."""
    torch.save({
        "model_config": cfg,
        "state_dict": model.state_dict(),
        "variant": variant,
        "epoch": epoch,
        "val_loss": float(val_loss),
        "val_auc": None if val_auc != val_auc else float(val_auc),
        "selection": selection,
        "seed": seed,
        "dataset": provenance,
        "dataset_fingerprint": fingerprint,
        "row_key_sha256": row_key_sha,
        "args": vars(args),
    }, path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def make_loader(dataset: Dataset, collate, batch_size: int, shuffle: bool,
                seed: int) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=0, drop_last=False, generator=gen if shuffle else None)


def train(args) -> dict:
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    v = C.variant(args.variant)
    dataset_dir = C.dataset_dir(args.dataset_tag if args.dataset_tag is not None else args.tag)
    store = FeatureStore(dataset_dir)
    out_dir = C.run_dir(v.name, args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = {}
    for name in ("train", "val"):
        path = dataset_dir / f"pairs_{name}.parquet"
        if not path.is_file():
            raise SystemExit(f"missing {path}; run "
                             f"`python -m scripts.ritcf_recon.pairs --tag {args.tag}`")
        pairs[name] = pd.read_parquet(path)
    print(f"pairs: train {len(pairs['train'])} | val {len(pairs['val'])}")

    device = torch.device(args.device)
    cfg = M.model_config(v.name, use_bn=args.use_bn)
    model = M.build_model(cfg).to(device)
    print(f"model: {type(model).__name__} shared={cfg['shared_weights']} "
          f"embed={cfg['embed_dim']} params={M.count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss() if v.direct_classifier else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # -- micro batch: cap, probe, then cover the rest with accumulation
    probe_pairs = pairs["train"]
    cap = args.max_micro_batch
    micro = probe_micro_batch(model, store, probe_pairs, device,
                              v.direct_classifier, start=cap)
    micro = min(micro, cap)
    accum = max(1, int(round(args.effective_batch / micro)))
    print(f"micro batch {micro} x accum {accum} = effective {micro * accum} "
          f"({'pairs' if not v.direct_classifier else 'samples'})")

    if v.direct_classifier:
        train_rows = np.flatnonzero((store.samples["split"] == "train").to_numpy())
        val_rows = np.flatnonzero((store.samples["split"] == "val").to_numpy())
        if val_rows.size == 0:
            print("WARNING: no validation rows in this dataset; validating on the "
                  "train rows (smoke builds only)")
            val_rows = train_rows[: max(1, train_rows.size // 5)]
        train_ds = DirectBatch(store, train_rows)
        val_ds = DirectBatch(store, val_rows)
        collate = collate_direct
    else:
        train_ds = PairBatch(store, pairs["train"])
        val_pairs = pairs["val"]
        if len(val_pairs) == 0:
            print("WARNING: no validation pairs in this dataset; validating on the "
                  "train pairs (smoke builds only)")
            val_pairs = pairs["train"]
        val_ds = PairBatch(store, val_pairs)
        collate = collate_pair
    train_loader = make_loader(train_ds, collate, micro, True, seed)
    val_loader = make_loader(val_ds, collate, micro, False, seed)
    print(f"batches: train {len(train_loader)} | val {len(val_loader)}")

    history = []
    best_loss = {"val_loss": float("inf"), "epoch": -1, "val_auc": float("nan")}
    best_auc = {"val_auc": float("-inf"), "epoch": -1, "val_loss": float("nan")}
    provenance = store.provenance()
    fingerprint = _manifest_field(dataset_dir, "fingerprint")
    row_key_sha = _manifest_field(dataset_dir, "row_keys", "sha256")
    bad_epochs = 0
    stopped_early = False
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, v.direct_classifier,
                       store=store, optimizer=optimizer, accum=accum)
        va = run_epoch(model, val_loader, criterion, device, v.direct_classifier,
                       store=store)
        diag = pair_diagnostics(va["pred"], va["label"],
                                higher_is_positive=v.direct_classifier)
        auc = pair_auc(va["pred"], va["label"],
                       higher_is_positive=v.direct_classifier)
        record = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"],
                  "val_auc": auc,
                  "val_POD_pair": diag["POD"], "val_FARate_pair": diag["FARate"],
                  "seconds": round(time.time() - t0, 1)}
        history.append(record)
        print(f"epoch {epoch:03d} train {tr['loss']:.6f} val {va['loss']:.6f} "
              f"val AUC {auc:.4f} "
              f"pair POD {diag['POD']:.3f} FARate {diag['FARate']:.3f} "
              f"({record['seconds']}s)", flush=True)

        common = dict(cfg=cfg, model=model, variant=v.name, epoch=epoch,
                      seed=seed, args=args, provenance=provenance,
                      fingerprint=fingerprint, row_key_sha=row_key_sha)
        if va["loss"] < best_loss["val_loss"] - 1e-6:
            best_loss = {"val_loss": va["loss"], "epoch": epoch, "val_auc": auc}
            bad_epochs = 0
            save_checkpoint(out_dir / C.CHECKPOINT_FILES["best_loss"],
                            val_loss=va["loss"], val_auc=auc,
                            selection="best_loss", **common)
        else:
            bad_epochs += 1
        if np.isfinite(auc) and auc > best_auc["val_auc"] + 1e-6:
            best_auc = {"val_auc": auc, "epoch": epoch, "val_loss": va["loss"]}
            save_checkpoint(out_dir / C.CHECKPOINT_FILES["best_auc"],
                            val_loss=va["loss"], val_auc=auc,
                            selection="best_auc", **common)
        if bad_epochs >= args.patience:
            print(f"early stop at epoch {epoch} (patience {args.patience}; "
                  f"best loss epoch {best_loss['epoch']}, best AUC epoch "
                  f"{best_auc['epoch']})")
            stopped_early = True
            break

    if not history:
        raise SystemExit("no epoch ran; --epochs must be >= 1")
    last = history[-1]
    save_checkpoint(out_dir / C.CHECKPOINT_FILES["last"], cfg=cfg, model=model,
                    variant=v.name, epoch=last["epoch"], val_loss=last["val_loss"],
                    val_auc=last["val_auc"], selection="last", seed=seed,
                    args=args, provenance=provenance, fingerprint=fingerprint,
                    row_key_sha=row_key_sha)
    prior_auc = [r["val_auc"] for r in history[:-1] if np.isfinite(r["val_auc"])]
    auc_still_improving = bool(np.isfinite(last["val_auc"]) and
                               last["val_auc"] > max(prior_auc, default=float("-inf")))

    elapsed = time.time() - started
    checkpoints = {
        "best_loss": {"file": C.CHECKPOINT_FILES["best_loss"], **best_loss},
        "best_auc": {"file": C.CHECKPOINT_FILES["best_auc"], **best_auc},
        "last": {"file": C.CHECKPOINT_FILES["last"],
                 "epoch": last["epoch"], "val_loss": last["val_loss"],
                 "val_auc": last["val_auc"]},
    }
    manifest = {
        "schema": "ritcf_recon_run_v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variant": C.variant_spec(v.name),
        "model_config": cfg,
        "code_sha256": C.code_fingerprint(),
        "seed": seed,
        "device": str(device),
        "torch": torch.__version__,
        "micro_batch": micro,
        "grad_accum": accum,
        "effective_batch": micro * accum,
        "n_parameters": M.count_parameters(model),
        "epochs_run": len(history),
        "best": best_loss,            # historical field: argmin of the val loss
        "best_loss": best_loss,
        "best_auc": best_auc,
        "checkpoints": checkpoints,
        "stopped_early": stopped_early,
        "auc_still_improving_at_stop": auc_still_improving,
        "selection_metric": "val ROC-AUC of the pair decision (siamese: -distance "
                            "vs label; direct: sigmoid prob)",
        "seconds": round(elapsed, 1),
        "dataset": provenance,
        "dataset_fingerprint": fingerprint,
        "row_key_sha256": row_key_sha,
        "pairs": {name: int(len(frame)) for name, frame in pairs.items()},
        "args": vars(args),
        "history": history,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best-loss epoch {best_loss['epoch']} val loss "
          f"{best_loss['val_loss']:.6f} -> "
          f"{out_dir / C.CHECKPOINT_FILES['best_loss']}")
    print(f"best-AUC  epoch {best_auc['epoch']} val AUC "
          f"{best_auc['val_auc']:.4f} -> "
          f"{out_dir / C.CHECKPOINT_FILES['best_auc']}")
    print(f"last      epoch {last['epoch']} -> "
          f"{out_dir / C.CHECKPOINT_FILES['last']} ({elapsed / 60:.1f} min)")
    return manifest


def _manifest_field(dataset_dir: Path, *keys):
    path = Path(dataset_dir) / "manifest.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        node = json.load(f)
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True,
                        help=f"one of {sorted(C.VARIANTS)}")
    parser.add_argument("--tag", default="", help="run tag ('' = full build)")
    parser.add_argument("--dataset-tag", default=None,
                        help="dataset tag when it differs from the run tag "
                             "(defaults to --tag)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10,
                        help="loss-based early-stop patience; the reported runs "
                             "use --epochs 60 --patience 60 (fixed budget, early "
                             "stop off) so the AUC argmax is not truncated")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--max-micro-batch", type=int, default=64,
                        help="cap for the probed micro-batch (VRAM guard)")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use-bn", action="store_true",
                        help="add BatchNorm (off by default: not in the paper)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    raise SystemExit(main())
