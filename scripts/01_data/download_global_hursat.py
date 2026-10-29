# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Download global HURSAT-B1 v07b storm tarballs (all basins) and convert each
to the phase-2 per-storm float16 frame format.

This writes the RAW corpus.  It is not the training corpus: the same acquisition
time can carry frames from different satellites with non-physical fill, so the
raw stores feed (1) the 2024 re-key (``scripts/rekey_hursat_2024.py``) and
(2) the QC pass (``scripts/build_hursat_qc.py``) that produces
``data/global_hursat/frames_qc``.  The holdout config points ``data.frames_dir``
at the QC corpus, and both MAE caches are built from it; only the
self-supervised objective itself is basin-agnostic (the predictive mapping is
still calibrated on WNP rows alone).

Source: https://www.ncei.noaa.gov/pub/data/gridsat/collaborate/hursat_tarballs/v07b/
One tarball per storm: HURSAT_b1_v07b_{SID}_{NAME}_c{date}.tar.gz containing
flat *.nc frames (IRWIN ~301x301, K).

Layout produced (under data/global_hursat/):
    tarballs/{name}.tar.gz     kept archive (re-processable)
    frames/{SID}.npy           float16 (n, 224, 224), NaN = unreadable frame
    frames/{SID}_times.npy     datetime64[s] (n,)  (frame_time_from_name)
    done/{SID}.ok              resume marker

Resumable (re-run freely).  Disk guard: pauses while free space < 25 GB.
Network + CPU only; safe to run alongside GPU training.

Usage (from the release root):
    python scripts/01_data/download_global_hursat.py --limit 3          # smoke
    python scripts/01_data/download_global_hursat.py --workers 3        # full run
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# HDF5/netCDF4 is NOT thread-safe in this build — serialize all decode calls
# (decode is CPU-light relative to download, so the lock is not a bottleneck).
_NC_LOCK = threading.Lock()

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.ir_dataset import frame_time_from_name

BASE_URL = "https://www.ncei.noaa.gov/pub/data/gridsat/collaborate/hursat_tarballs/v07b/"
OUT_ROOT = P2_ROOT / "data" / "global_hursat"
CROP = 224
MIN_FREE_GB = 25.0

_NAME_RE = re.compile(r"HURSAT_b1_v07b_([0-9A-Z]+)_([A-Za-z0-9_\-]+)_c\d+\.tar\.gz")


def fetch_listing() -> pd.DataFrame:
    html = requests.get(BASE_URL, timeout=120).text
    rows = []
    for m in re.finditer(r'href="(HURSAT_b1_v07b_[^"]+\.tar\.gz)"', html):
        name = m.group(1)
        mm = _NAME_RE.match(name)
        if not mm:
            continue
        sid = mm.group(1)
        rows.append({"name": name, "SID": sid, "year": int(sid[:4])})
    return pd.DataFrame(rows).drop_duplicates("name").sort_values("year").reset_index(drop=True)


def free_gb(path: Path) -> float:
    import shutil
    return shutil.disk_usage(path).free / 1e9


def disk_guard():
    while free_gb(OUT_ROOT) < MIN_FREE_GB:
        print(f"[disk] free {free_gb(OUT_ROOT):.0f} GB < {MIN_FREE_GB:.0f} GB — "
              f"pausing 10 min (clear space to resume)", flush=True)
        time.sleep(600)


def download(url: str, dest: Path, expected: int | None = None) -> Path:
    """Resumable download to dest (.part while in flight)."""
    part = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    mode = "wb"
    got = 0
    if part.exists():
        got = part.stat().st_size
        headers["Range"] = f"bytes={got}-"
        mode = "ab"
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        if r.status_code == 416:
            pass  # range unsatisfiable: .part already holds the full file
        elif got and r.status_code == 200:   # server ignored Range -> restart
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        elif r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        else:
            with open(part, mode) as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    # Windows AV may hold the freshly written .part open; retry the rename
    # patiently (Defender scans of 100 MB files can take minutes under load).
    for _ in range(60):
        try:
            part.rename(dest)
            break
        except OSError:
            time.sleep(5)
    else:
        raise RuntimeError(f"cannot rename {part} (locked)")
    return dest


def decode_tarball(tar_path: Path, sid: str):
    """Return (frames float16 (n,224,224), times datetime64[s] (n,))."""
    frames, times = [], []
    with tempfile.TemporaryDirectory() as td, tarfile.open(tar_path, "r:gz") as tf:
        for member in sorted(tf.getnames()):
            if not member.endswith(".nc"):
                continue
            t = frame_time_from_name(Path(member))
            if t is None:
                continue
            try:
                raw = tf.extractfile(member).read()
                p = Path(td) / "f.nc"
                p.write_bytes(raw)
                with _NC_LOCK:
                    from netCDF4 import Dataset
                    with Dataset(p) as ds:
                        v = ds["IRWIN"][:]
                v = np.asarray(np.ma.filled(v, np.nan), dtype=np.float32).squeeze()
                if v.ndim != 2 or v.shape[0] < CROP or v.shape[1] < CROP:
                    continue
                s0 = (v.shape[0] - CROP) // 2
                s1 = (v.shape[1] - CROP) // 2
                frames.append(v[s0:s0 + CROP, s1:s1 + CROP].astype(np.float16))
                times.append(t.tz_convert(None))
            except Exception:
                continue
    if not frames:
        raise RuntimeError(f"no decodable frames in {tar_path.name}")
    return np.stack(frames), np.array(times, dtype="datetime64[s]")


def process(row, dirs) -> str:
    sid = row["SID"]
    done_f = dirs["done"] / f"{sid}.ok"
    if done_f.exists():
        return "skip"
    tar = dirs["tarballs"] / row["name"]
    if not tar.exists():
        disk_guard()
        download(BASE_URL + row["name"], tar)
    frames = times = None
    for attempt in (1, 2):  # retry once: WinError 32 AV races are transient
        try:
            frames, times = decode_tarball(tar, sid)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    np.save(dirs["frames"] / f"{sid}.npy", frames)
    np.save(dirs["frames"] / f"{sid}_times.npy", times)
    done_f.write_text(f"{len(frames)} frames\n")
    return f"ok ({len(frames)} fr)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-min", type=int, default=2000)
    ap.add_argument("--year-max", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dirs = {k: OUT_ROOT / k for k in ("tarballs", "frames", "done")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    listing = fetch_listing()
    sel = listing[(listing.year >= args.year_min) & (listing.year <= args.year_max)]
    if args.limit:
        sel = sel.head(args.limit)
    todo = sel[~sel["SID"].isin({p.stem for p in dirs["done"].glob("*.ok")})]
    print(f"listing: {len(listing)} storms total; selected {len(sel)} "
          f"({args.year_min}-{args.year_max}); todo {len(todo)}", flush=True)

    t0 = time.time()
    done_n = fail_n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, row, dirs): row for row in sel.to_dict("records")
                if not (dirs["done"] / f"{row['SID']}.ok").exists()}
        for fut in as_completed(futs):
            row = futs[fut]
            try:
                status = fut.result()
                done_n += 1
            except Exception as e:
                fail_n += 1
                status = f"FAIL: {e}"
            el = time.time() - t0
            rate = done_n / max(el, 1e-9) * 3600
            print(f"[{done_n + fail_n}/{len(futs)}] {row['SID']} {row['name'][:52]} "
                  f"-> {status} | {rate:.0f} storms/h, {fail_n} failed", flush=True)
    print(f"DONE: {done_n} ok, {fail_n} failed in {(time.time()-t0)/3600:.1f} h")
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
