"""Pair construction for RITCF-recon (spec §5).

Three different things share the word "pair" here, and they are kept apart:

* **train pairs** -- every train RI sample pairs with its ``TRAIN_POS_REFS``
  (21) previous nearest RI samples (label 0); every train non-RI sample pairs
  with its single nearest previous RI (label 1).  ``Input-A`` is always the
  known RI, ``Input-B`` the sample being judged, and the two are at least
  ``PAIR_MIN_GAP_H`` (24 h) apart -- the paper's "no later than 30 September
  2019" rule is an example of that separation, not a pool cap.
* **val pairs** -- the same construction inside the validation years.  The
  paper says "follows the same procedure" yet reports 110 x 12 positives; the
  spec takes x21 and records the contradiction.  These pairs drive early
  stopping and the pair-level threshold.
* **inference references** -- each evaluated sample is judged against the
  ``TEST_REFS`` (10) nearest RI samples that were *already known* at its
  forecast time.  Which RI samples count as "known" is the one thing the
  variants disagree about (``rolling`` vs ``capped_2019``).

Row indices are positions in ``samples.parquet``, which is time-sorted and
identical to the row order of the feature memmaps.

Usage (from the release root):
    python -m scripts.ritcf_recon.pairs --tag smoke      # write + print counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P2_ROOT = Path(__file__).resolve().parents[2]
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))

from scripts.ritcf_recon import config as C  # noqa: E402

POOL_CAP = pd.Timestamp("2019-12-31 23:59:59")


def load_samples(dataset_dir: Path) -> pd.DataFrame:
    path = Path(dataset_dir) / "samples.parquet"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run build_dataset first")
    samples = pd.read_parquet(path)
    samples["time"] = pd.to_datetime(samples["time"], utc=True).dt.tz_localize(None)
    return samples.sort_values(["time", "SID"]).reset_index(drop=True)


def _epoch(samples: pd.DataFrame) -> np.ndarray:
    return samples["time"].to_numpy(dtype="datetime64[s]").astype(np.int64)


def pool_mask(samples: pd.DataFrame, kind: str) -> np.ndarray:
    """RI rows that count as "known" for a reference pool."""
    ri = samples["ri"].to_numpy() == 1
    if kind == "rolling":
        return ri
    if kind == "capped_2019":
        return ri & (samples["time"] <= POOL_CAP).to_numpy()
    raise SystemExit(f"unknown reference-pool kind {kind!r}")


def nearest_refs(samples: pd.DataFrame, target_mask: np.ndarray, ref_mask: np.ndarray,
                 k: int, min_gap_h: int = C.PAIR_MIN_GAP_H) -> np.ndarray:
    """(n_target, k) row indices of the nearest known RI samples, -1 padded.

    "Nearest" means most recent in time: the pool is walked backwards from
    ``t - min_gap_h`` and the first ``k`` entries are taken (nearest first).
    """
    epoch = _epoch(samples)
    pool = np.flatnonzero(ref_mask)
    pool_times = epoch[pool]
    targets = np.flatnonzero(target_mask)
    out = np.full((targets.size, k), -1, dtype=np.int64)
    gap = min_gap_h * 3600
    for row, t in enumerate(targets):
        stop = int(np.searchsorted(pool_times, epoch[t] - gap, side="right"))
        take = pool[max(0, stop - k):stop]
        out[row, :take.size] = take[::-1]
    return out


def pair_frame(samples: pd.DataFrame, split: str, pos_refs: int,
               min_gap_h: int = C.PAIR_MIN_GAP_H) -> pd.DataFrame:
    """Training-style pairs inside one split (RI-relative, label 0/1)."""
    in_split = (samples["split"] == split).to_numpy()
    targets = np.flatnonzero(in_split)
    if targets.size == 0:
        return pd.DataFrame(columns=["a_idx", "b_idx", "label", "refs", "split"])
    # inference-style pool: every known RI with time <= t-gap, across splits
    # (the temporal guard alone prevents any future reference; for val targets
    # this also admits train-era RIs, matching the paper's implied val counts)
    pool = samples["ri"].to_numpy() == 1
    epoch = _epoch(samples)
    pool_idx = np.flatnonzero(pool)
    pool_times = epoch[pool_idx]
    gap = min_gap_h * 3600

    rows_a, rows_b, labels, refs = [], [], [], []
    for t in targets:
        stop = int(np.searchsorted(pool_times, epoch[t] - gap, side="right"))
        if stop == 0:
            continue
        if samples["ri"].iat[t] == 1:
            take = pool_idx[max(0, stop - pos_refs):stop][::-1]
            label = 0
        else:
            take = pool_idx[stop - 1:stop]
            label = 1
        for ref in take:
            rows_a.append(int(ref))
            rows_b.append(int(t))
            labels.append(label)
            refs.append(len(take))
    frame = pd.DataFrame({"a_idx": rows_a, "b_idx": rows_b,
                          "label": np.asarray(labels, dtype=np.int8),
                          "refs": refs})
    frame["split"] = split
    return frame


def build_pair_tables(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": pair_frame(samples, "train", C.TRAIN_POS_REFS),
        "val": pair_frame(samples, "val", C.VAL_POS_REFS),
    }


def write_pair_tables(dataset_dir: Path, tables: dict[str, pd.DataFrame]) -> dict:
    meta = {}
    for name, frame in tables.items():
        path = Path(dataset_dir) / f"pairs_{name}.parquet"
        frame.to_parquet(path, index=False)
        meta[name] = {
            "n_pairs": int(len(frame)),
            "n_pos": int((frame["label"] == 0).sum()),
            "n_neg": int((frame["label"] == 1).sum()),
            "n_unique_targets": int(frame["b_idx"].nunique()),
            "sha256": C.sha256_file(path),
        }
    return meta


def summary(samples: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> dict:
    """Counts for the report: sample frame plus pair frame per split."""
    out = {"samples": {}, "pairs": {}}
    for name in ("train", "val", "test"):
        s = samples[samples["split"] == name]
        out["samples"][name] = {
            "n": int(len(s)), "n_ri": int(s["ri"].sum()),
            "n_nonri": int((s["ri"] == 0).sum()),
            "base_rate": float(s["ri"].mean()) if len(s) else None,
            "n_storms": int(s["SID"].nunique()),
        }
    for name, frame in tables.items():
        out["pairs"][name] = {
            "n_pairs": int(len(frame)),
            "n_pos": int((frame["label"] == 0).sum()),
            "n_neg": int((frame["label"] == 1).sum()),
            "zero_ref_targets": int(len(samples[samples["split"] == name]) -
                                    frame["b_idx"].nunique()) if len(frame) else 0,
        }
    # how many test rows can be scored at all (needs >=1 known reference)
    for kind in ("rolling", "capped_2019"):
        for split in ("val", "test"):
            tgt = (samples["split"] == split).to_numpy()
            refs = nearest_refs(samples, tgt, pool_mask(samples, kind), C.TEST_REFS)
            have = int((refs >= 0).sum(axis=1).sum())
            per_row = (refs >= 0).sum(axis=1)
            out.setdefault("refs", {}).setdefault(kind, {})[split] = {
                "n_targets": int(tgt.sum()),
                "n_scoreable": int((per_row > 0).sum()),
                "mean_refs": float(per_row.mean()) if per_row.size else 0.0,
                "n_ref_links": have,
            }
    return out


def report(dataset_dir: Path) -> dict:
    samples = load_samples(dataset_dir)
    tables = build_pair_tables(samples)
    info = summary(samples, tables)
    print("\npairs:")
    for name, c in info["pairs"].items():
        print(f"  {name:5s} pairs={c['n_pairs']:6d} pos={c['n_pos']:5d} "
              f"neg={c['n_neg']:6d} targets_without_pair={c['zero_ref_targets']}")
    print("  paper: 6363 pos + 6492 neg = 12855 train pairs (303 x 21)")
    print("inference reference pools (10 nearest known RI per target):")
    for kind, per_split in info["refs"].items():
        for split, c in per_split.items():
            print(f"  {kind:12s} {split:5s} targets={c['n_targets']:5d} "
                  f"scoreable={c['n_scoreable']:5d} mean_refs={c['mean_refs']:5.2f}")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="dataset tag")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    dataset_dir = C.dataset_dir(args.tag)
    samples = load_samples(dataset_dir)
    tables = build_pair_tables(samples)
    meta = write_pair_tables(dataset_dir, tables)
    print(f"pairs written to {dataset_dir}")
    info = report(dataset_dir)
    if not args.report_only:
        path = dataset_dir / "pairs_manifest.json"
        path.write_text(json.dumps({"files": meta, "summary": info}, indent=2),
                        encoding="utf-8")
        print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
