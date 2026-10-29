# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Shared configuration for the RITCF reconstruction (scripts/ritcf_recon).

The reconstruction follows ``docs/RITCF_RECON_SPEC.md``; ``README.md`` names
every point the spec left open and every deviation from the paper.  This module
holds what all four variants share: the feature layout, the sample-panel rules,
the split boundaries, the RI threshold and the variant registry.

Isolation: nothing here reads ``data/test`` or ``protocol_locks``.  The only
inputs are IBTrACS WP v04r01, the per-storm ERA5 archive, the QC v1 HURSAT-B1
frame corpus and this package's own cache under ``data/ritcf_recon``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
P2_ROOT = Path(__file__).resolve().parents[2]
P1_ROOT = P2_ROOT / "data" / "phase1"
PACKAGE_DIR = Path(__file__).resolve().parent

OUT_ROOT = P2_ROOT / "data" / "ritcf_recon"

IBTRACS_CSV = P1_ROOT / "raw" / "ibtracs" / "ibtracs.WP.list.v04r01.csv"
ERA5_DIR = P1_ROOT / "raw" / "era5"
RAW_HURSAT_DIR = P1_ROOT / "raw" / "hursat"

# QC v1 corpus of HURSAT-B1 v07b frames (scripts/build_hursat_qc.py): one
# 224x224 float16 TBB crop per acquisition time, non-physical pixels set to
# NaN, multi-satellite collisions resolved.  This is the decoded form of the
# same tarballs under RAW_HURSAT_DIR, and for RITCF it replaces 300k per-frame
# netCDF opens; see README "SAT source".
FRAMES_DIR = P2_ROOT / "data" / "global_hursat" / "frames_qc"
FRAMES_MANIFEST = P2_ROOT / "data" / "global_hursat" / "qc_manifest.json"


def dataset_dir(tag: str = "") -> Path:
    """Directory of one built dataset (``tag=""`` is the full build)."""
    return OUT_ROOT / (f"dataset_{tag}" if tag else "dataset")


def run_dir(variant: str, tag: str = "") -> Path:
    base = OUT_ROOT / (f"runs_{tag}" if tag else "runs")
    return base / variant


def feature_cache_dir(tag: str = "") -> Path:
    return dataset_dir(tag) / "cache" / "feat"


# --------------------------------------------------------------------------
# Sample panel / label
# --------------------------------------------------------------------------
SEASON_RANGE = (2000, 2021)
SYNOPTIC_HOURS = (0, 6, 12, 18)

# RI = 24-h CMA intensity change >= 25.27 kt (= 13 m/s).  The paper states
# 13 m/s and 25.26 kt in different places; 13 m/s * 1.943844 = 25.26997 kt, so
# the two agree to the stated precision.  Keep the knot value explicit.
RI_THR_KT = 25.27
MPS_TO_KT = 1.943844

# Sample filter, in evaluation order (the build reports the funnel):
#   CMA wind present -> exact +24 h CMA wind in the same storm -> over ocean.
# "Over ocean" uses IBTrACS DIST2LAND > 0 (km); rows with a missing DIST2LAND
# are dropped and counted (spec §1 allows either LAND_DIST>0 or the IBTrACS
# sea/land flag -- we take the distance field and record it).
OCEAN_RULE = "DIST2LAND_gt_0"

SPLITS = (
    (2000, 2017, "train"),
    (2018, 2019, "val"),
    (2020, 2021, "test"),
)

# --------------------------------------------------------------------------
# Feature layout
# --------------------------------------------------------------------------
# ERA5: 4 variables x 4 levels x 5 times = 80 channels, then SST x 5 times = 5.
ERA5_VARS = ("u", "v", "pv", "d")
ERA5_LEVELS = (1000, 850, 500, 200)
DIRECTIONAL_VARS = ("u", "v")          # min-max to [-1, 1]; all else to [0, 1]
N_TIMES = 5
TIME_OFFSETS_H = (0, -6, -12, -18, -24)

ERA5_CHANNELS = len(ERA5_VARS) * len(ERA5_LEVELS) * N_TIMES   # 80
SST_CHANNELS = N_TIMES                                        # 5
GRID_CHANNELS = ERA5_CHANNELS + SST_CHANNELS                  # 85
GRID_SIZE = 25                                                # 1-deg blocks
ERA5_BLOCK_DEG = 1.0
ERA5_NATIVE_DEG = 0.25
ERA5_WINDOW_CELLS = int(GRID_SIZE * ERA5_BLOCK_DEG / ERA5_NATIVE_DEG)  # 100

SAT_FRAMES = N_TIMES
SAT_SIZE = 224
SAT_MIN_K = 100.0
SAT_MAX_K = 350.0

HIS_DIM = 30   # 5 wind + 5 pressure + 10 + 10 pairwise differences


def era5_channel_names() -> list[str]:
    names = []
    for var in ERA5_VARS:
        for lev in ERA5_LEVELS:
            for off in TIME_OFFSETS_H:
                names.append(f"{var}{lev}_t{off:+d}h")
    for off in TIME_OFFSETS_H:
        names.append(f"sst_t{off:+d}h")
    return names


def his_channel_names() -> list[str]:
    which = ["wind", "pres"]
    names = [f"{w}_t{off:+d}h" for w in which for off in TIME_OFFSETS_H]
    pairs = [(i, j) for i in range(N_TIMES) for j in range(i + 1, N_TIMES)]
    names += [f"d{w}_{TIME_OFFSETS_H[i]:+d}_{TIME_OFFSETS_H[j]:+d}h"
              for w in which for i, j in pairs]
    return names


def directional_mask() -> list[bool]:
    """True where the channel is a signed wind component (min-max to [-1, 1])."""
    return [name.startswith(tuple(DIRECTIONAL_VARS)) for name in era5_channel_names()]


# --------------------------------------------------------------------------
# Pair construction (spec §5)
# --------------------------------------------------------------------------
PAIR_MIN_GAP_H = 24      # paper: Input-A and Input-B at least 24 h apart
TRAIN_POS_REFS = 21      # each RI sample pairs with its 21 previous nearest RIs
VAL_POS_REFS = 21        # paper says "same procedure" but reports 110x12; spec §5
                         # takes x21 and records the contradiction
TEST_REFS = 10           # each test sample pairs with the 10 nearest known RIs
VOTE_MIN_FRACTION = 0.6  # >=6 of 10 votes -> RI


# --------------------------------------------------------------------------
# Variants (spec §6)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    name: str
    pool: str                 # "rolling" | "capped_2019"
    shared_weights: bool
    embed_dim: int
    scalar_distance: bool     # True -> |f(a)-f(b)| (author repo skeleton)
    aggregation: str          # "vote" | "mean_distance"
    direct_classifier: bool   # True -> no siamese path at all
    description: str


VARIANTS = {
    "V1": Variant(
        name="V1", pool="rolling", shared_weights=True, embed_dim=32,
        scalar_distance=False, aggregation="vote", direct_classifier=False,
        description="faithful: rolling reference pool (test-year RIs are usable "
                    "references), shared siamese weights, 32-d embedding, 6/10 vote",
    ),
    "V2": Variant(
        name="V2", pool="capped_2019", shared_weights=True, embed_dim=32,
        scalar_distance=False, aggregation="vote", direct_classifier=False,
        description="clean pool: references limited to <=2019-12-31 (V1-V2 = "
                    "contribution of rolling label circulation)",
    ),
    "V3": Variant(
        name="V3", pool="rolling", shared_weights=False, embed_dim=1,
        scalar_distance=True, aggregation="vote", direct_classifier=False,
        description="author-repo skeleton: unshared branches, scalar embeddings, "
                    "distance = |f(a)-f(b)|, everything else as V1",
    ),
    "V4": Variant(
        name="V4", pool="rolling", shared_weights=True, embed_dim=32,
        scalar_distance=False, aggregation="mean_distance", direct_classifier=False,
        description="vote ablation: mean pair distance used directly as the score "
                    "(no 6/10 hard vote); probability metrics",
    ),
    "V4direct": Variant(
        name="V4direct", pool="rolling", shared_weights=True, embed_dim=32,
        scalar_distance=False, aggregation="mean_distance", direct_classifier=True,
        description="same features, direct binary classifier (no siamese, no "
                    "pairing) -- measures the contrastive-learning increment",
    ),
}

SIAMESE_VARIANTS = tuple(k for k, v in VARIANTS.items() if not v.direct_classifier)


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
# One file per selection criterion, all written by train.py.  ``best_loss`` is
# the original argmin of the validation pair loss (the historical default);
# ``best_auc`` is the argmax of the validation ROC-AUC, the criterion the
# downstream vote/ROC analysis actually needs (see README "Checkpoint
# selection"); ``last`` is the final epoch trained, i.e. wherever the
# loss-based early stop fired.
CHECKPOINT_FILES = {
    "best_loss": "checkpoint.pt",
    "best_auc": "checkpoint_auc.pt",
    "last": "checkpoint_last.pt",
}
DEFAULT_CHECKPOINT_KIND = "best_loss"


def checkpoint_kind(name: str) -> str:
    key = name.strip()
    if key not in CHECKPOINT_FILES:
        raise SystemExit(f"unknown checkpoint kind {name!r}; choose from "
                         f"{sorted(CHECKPOINT_FILES)}")
    return key


def variant(name: str) -> Variant:
    key = name.strip()
    if key not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[key]


# --------------------------------------------------------------------------
# Hashing helpers
# --------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def code_fingerprint() -> str:
    """sha256 over the package sources, so a manifest names the code that built it."""
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def feature_spec(ri_thr_kt: float = RI_THR_KT) -> dict:
    """The feature/label layout, with the RI threshold actually in force.

    ``build_dataset`` carries ``--ri-thr-kt`` (25.27 by default, 30.0 for the
    our-standard comparison build) and the panel labels rows with that value,
    so the manifest must record the argument rather than the module constant.
    """
    return {
        "ri_thr_kt": float(ri_thr_kt),
        "season_range": list(SEASON_RANGE),
        "synoptic_hours": list(SYNOPTIC_HOURS),
        "ocean_rule": OCEAN_RULE,
        "time_offsets_h": list(TIME_OFFSETS_H),
        "era5_vars": list(ERA5_VARS),
        "era5_levels": list(ERA5_LEVELS),
        "era5_native_deg": ERA5_NATIVE_DEG,
        "era5_block_deg": ERA5_BLOCK_DEG,
        "era5_window_cells": ERA5_WINDOW_CELLS,
        "grid_size": GRID_SIZE,
        "grid_channels": GRID_CHANNELS,
        "sat_frames": SAT_FRAMES,
        "sat_size": SAT_SIZE,
        "sat_valid_k": [SAT_MIN_K, SAT_MAX_K],
        "his_dim": HIS_DIM,
        "era5_channel_names": era5_channel_names(),
        "his_channel_names": his_channel_names(),
        "pair_min_gap_h": PAIR_MIN_GAP_H,
        "train_pos_refs": TRAIN_POS_REFS,
        "val_pos_refs": VAL_POS_REFS,
        "test_refs": TEST_REFS,
        "vote_min_fraction": VOTE_MIN_FRACTION,
        "splits": [{"years": [lo, hi], "name": name} for lo, hi, name in SPLITS],
    }


def variant_spec(name: str) -> dict:
    return asdict(variant(name))
