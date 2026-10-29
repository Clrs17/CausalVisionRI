"""Feature store: memmap access + normalization for RITCF-recon.

The build writes raw values (TBB in K, ERA5 in SI units, HIS in kt/hPa) and
``norm.json`` holds the min/max and NaN-fill statistics computed on the train
split only (spec §3).  One cache therefore serves every variant.

Normalization (paper Eq.1 + §3 of the spec):

* U, V channels -> min-max to [-1, 1]
* every other ERA5 / SST / HIS value -> min-max to [0, 1]
* SAT TBB -> linear over [100 K, 350 K] to [0, 1]

NaN policy: ERA5 and HIS take the train-split mean (per channel); SAT pixels
are per-frame-mean filled at build time, and an entirely missing frame becomes
0.5 after normalization.

The work is done on the GPU: the batches leave the loader in cache dtype
(float16) and are scaled there, which is ~10x cheaper than scaling 224x224
frames on the CPU.  ``raw_batch`` + ``normalize`` are the only two entry points
training and inference need.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.ritcf_recon import config as C


class FeatureStore:
    def __init__(self, dataset_dir: str | Path) -> None:
        self.dir = Path(dataset_dir)
        self.samples = pd.read_parquet(self.dir / "samples.parquet")
        self.samples["time"] = (pd.to_datetime(self.samples["time"], utc=True)
                               .dt.tz_localize(None))
        self.samples = self.samples.sort_values(["time", "SID"]).reset_index(drop=True)
        if "sample_id" in self.samples.columns:
            assert self.samples["sample_id"].is_monotonic_increasing, \
                "sample_id order disagrees with the (time, SID) row order"
        assert not self.samples.duplicated(["SID", "time"]).any()
        with open(self.dir / "norm.json", encoding="utf-8") as f:
            self.norm = json.load(f)
        self._open()
        self._stats: dict[str, dict[str, torch.Tensor]] = {}

    # -- memmap handles (re-opened per worker process) ---------------------
    def _open(self) -> None:
        self.era5 = np.load(self.dir / "era5.npy", mmap_mode="r")
        self.sat = np.load(self.dir / "sat.npy", mmap_mode="r")
        self.his = np.load(self.dir / "his.npy", mmap_mode="r")
        n = self.norm["era5"]
        self.era5_min = np.asarray(n["min"], dtype=np.float32)
        self.era5_max = np.asarray(n["max"], dtype=np.float32)
        self.era5_span = np.maximum(self.era5_max - self.era5_min, 1e-6)
        self.era5_dir = np.asarray(n["directional"], dtype=bool)
        fill = (np.asarray(n["fill"], dtype=np.float32) - self.era5_min) / self.era5_span
        self.era5_fill = np.where(self.era5_dir, fill * 2.0 - 1.0, fill).astype(np.float32)
        h = self.norm["his"]
        self.his_min = np.asarray(h["min"], dtype=np.float32)
        self.his_max = np.asarray(h["max"], dtype=np.float32)
        self.his_span = np.maximum(self.his_max - self.his_min, 1e-6)
        self.his_fill = ((np.asarray(h["fill"], dtype=np.float32) - self.his_min)
                         / self.his_span).astype(np.float32)
        self.sat_min = float(self.norm["sat"]["min"])
        self.sat_max = float(self.norm["sat"]["max"])
        self.sat_fill = float(self.norm["sat"].get("fill_normalized", 0.5))

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("era5", "sat", "his"):
            state.pop(key, None)
        state["_stats"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()
        self._stats = {}

    # -- batches ----------------------------------------------------------
    def raw_batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Cache-dtype arrays for the given rows (float16 / float16 / float32)."""
        idx = np.asarray(idx)
        return {"era5": np.asarray(self.era5[idx]), "sat": np.asarray(self.sat[idx]),
                "his": np.asarray(self.his[idx])}

    def _device_stats(self, device: torch.device) -> dict[str, torch.Tensor]:
        key = str(device)
        if key not in self._stats:
            mk = lambda arr, shape: torch.from_numpy(np.ascontiguousarray(arr)
                                                     .reshape(shape)).to(device)
            self._stats[key] = {
                "era5_min": mk(self.era5_min, (1, -1, 1, 1)),
                "era5_span": mk(self.era5_span, (1, -1, 1, 1)),
                "era5_dir": mk(self.era5_dir, (1, -1, 1, 1)),
                "era5_fill": mk(self.era5_fill, (1, -1, 1, 1)),
                "his_min": mk(self.his_min, (1, -1)),
                "his_span": mk(self.his_span, (1, -1)),
                "his_fill": mk(self.his_fill, (1, -1)),
            }
        return self._stats[key]

    def normalize(self, raw: dict[str, np.ndarray],
                  device: torch.device) -> dict[str, torch.Tensor]:
        s = self._device_stats(device)
        era5 = torch.from_numpy(raw["era5"]).to(device, non_blocking=True).float()
        era5 = (era5 - s["era5_min"]) / s["era5_span"]
        era5 = torch.where(s["era5_dir"], era5 * 2.0 - 1.0, era5)
        era5 = torch.where(torch.isfinite(era5), era5, s["era5_fill"])

        sat = torch.from_numpy(raw["sat"]).to(device, non_blocking=True).float()
        sat = (sat - self.sat_min) / (self.sat_max - self.sat_min)
        sat = torch.where(torch.isfinite(sat), sat,
                          torch.full_like(sat, self.sat_fill))

        his = torch.from_numpy(raw["his"]).to(device, non_blocking=True).float()
        his = (his - s["his_min"]) / s["his_span"]
        his = torch.where(torch.isfinite(his), his, s["his_fill"])
        return {"era5": era5, "sat": sat, "his": his}

    def batch(self, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        return self.normalize(self.raw_batch(idx), device)

    # -- bookkeeping -------------------------------------------------------
    def provenance(self) -> dict:
        return {
            "dataset_dir": str(self.dir),
            "n_rows": int(len(self.samples)),
            "norm_scope": self.norm.get("scope"),
            "norm_train_rows": self.norm.get("n_rows"),
            "samples_sha256": C.sha256_file(self.dir / "samples.parquet"),
        }
