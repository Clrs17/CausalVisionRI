"""Per-storm HURSAT frame stores: array + time axis, two naming conventions.

Two corpora are in use:

* ``data/frames`` (WNP, built by phase-1/``src.ir_dataset``):
  ``{SID}.npy`` + ``{SID}_iso.npy`` — one timestamp per frame, at the panel row
  times.
* ``data/global_hursat/frames_qc`` (all basins, the training corpus): the QC v1
  slice of the download corpus, one frame per acquisition time with
  non-physical TBB masked to NaN (``scripts/build_hursat_qc.py``);
  ``{SID}.npy`` + ``{SID}_times.npy`` — the acquisition time of each image.
  Its raw input ``data/global_hursat/frames`` is re-key/QC input and the
  full-multisensor MAE pretraining corpus (``--source-corpus raw_full``); it
  is never configured as ``data.frames_dir``.

The row keys are identical in both cases (``SID|<ISO>Z``); only the time-axis
file name differs, so every reader goes through this module instead of assuming
``_iso``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TIME_SUFFIXES = ("_iso", "_times")


def frame_array_path(frames_dir, sid: str) -> Path:
    return Path(frames_dir) / f"{sid}.npy"


def frame_time_path(frames_dir, sid: str) -> Path | None:
    """Existing time-axis file for ``sid`` (``_iso`` preferred), else None."""
    base = Path(frames_dir)
    for suffix in TIME_SUFFIXES:
        candidate = base / f"{sid}{suffix}.npy"
        if candidate.is_file():
            return candidate
    return None


def require_frame_time_path(frames_dir, sid: str) -> Path:
    path = frame_time_path(frames_dir, sid)
    if path is None:
        raise FileNotFoundError(
            f"{sid}: no frame time axis ('{sid}_iso.npy' or '{sid}_times.npy') "
            f"under {Path(frames_dir)}")
    return path


def list_store_sids(frames_dir) -> list:
    """SIDs with a frame array in ``frames_dir`` (time axes excluded)."""
    return sorted(p.stem for p in Path(frames_dir).glob("*.npy")
                  if not p.stem.endswith(TIME_SUFFIXES))


def has_frame_store(frames_dir, sid: str) -> bool:
    return (frame_array_path(frames_dir, sid).is_file()
            and frame_time_path(frames_dir, sid) is not None)


def load_frame_times(frames_dir, sid: str) -> np.ndarray:
    """Frame timestamps as naive UTC ``datetime64[ns]``."""
    raw = np.load(require_frame_time_path(frames_dir, sid))
    return np.asarray(raw, dtype="datetime64[ns]")
