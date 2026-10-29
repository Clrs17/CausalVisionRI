# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""HURSAT-B1 v07b per-storm downloader for the phase-2 IR imagery dataset.

Archive layout (v07b): a FLAT directory of per-storm tarballs
    HURSAT_b1_v07b_{SID}_{NAME}_c20240926.tar.gz
covering 1978-2024, where SID uses the IBTrACS convention. We match our
375-storm sample (data/processed/sample_manifest.csv, INCLUDED==True) by SID.

- Downloads go to data/raw/hursat/ (tarballs kept; NEVER deleted).
- Extracted to data/raw/hursat/{SID}/.
- Resumable (skip existing), few parallel workers, manifest + log written.

Usage (from the release root):
    ~/anaconda3/python.exe scripts/download_hursat.py            # all sample storms
    ~/anaconda3/python.exe scripts/download_hursat.py --max 2    # probe 2 storms
    ~/anaconda3/python.exe scripts/download_hursat.py --dry-run  # listing/coverage only
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.ncei.noaa.gov/pub/data/gridsat/collaborate/hursat_tarballs/v07b/"
OUT_DIR = ROOT / "data" / "phase1" / "raw" / "hursat"
LOG = ROOT / "data" / "phase1" / "processed" / "logs" / "hursat_download.log"
MANIFEST = ROOT / "data" / "phase1" / "processed" / "hursat_manifest.csv"


def fetch_listing() -> dict:
    """Return {SID: (filename, size_bytes)} from the v07b archive index."""
    logger.info(f"Fetching archive index from {BASE_URL}")
    r = requests.get(BASE_URL, timeout=120)
    r.raise_for_status()
    pat = re.compile(r'href="(HURSAT_b1_v07b_([0-9A-Z]+)_([A-Za-z0-9_\-]+)_c[0-9]+\.tar\.gz)"')
    out = {}
    for m in pat.finditer(r.text):
        fname, sid = m.group(1), m.group(2)
        out[sid] = fname
    logger.info(f"Archive lists {len(out)} storms")
    return out


def download_one(sid: str, fname: str, extract: bool = True,
                 max_retries: int = 5) -> dict:
    url = BASE_URL + fname
    tar_path = OUT_DIR / fname
    rec = {"SID": sid, "file": fname, "url": url, "status": "ok",
           "n_nc": 0, "size_mb": 0.0, "error": ""}
    try:
        need_dl = not (tar_path.exists() and tar_path.stat().st_size > 0)
        if need_dl:
            # Retry with resume: NOAA's server drops long connections
            # (IncompleteRead); resume from the .part byte offset via Range.
            tmp = tar_path.with_suffix(".part")
            expected = None
            try:
                head = requests.head(url, timeout=60, allow_redirects=True)
                if head.ok and "Content-Length" in head.headers:
                    expected = int(head.headers["Content-Length"])
            except Exception:
                pass
            for attempt in range(1, max_retries + 1):
                try:
                    have = tmp.stat().st_size if tmp.exists() else 0
                    if expected is not None and have >= expected and have > 0:
                        tmp.rename(tar_path)
                        break
                    headers = {"Range": f"bytes={have}-"} if have > 0 else {}
                    with requests.get(url, stream=True, timeout=(60, 900),
                                      headers=headers) as r:
                        r.raise_for_status()
                        mode = "ab" if (have > 0 and r.status_code == 206) else "wb"
                        if r.status_code == 200 and have > 0:
                            have = 0  # server ignored Range; restart
                        with open(tmp, mode) as f:
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                f.write(chunk)
                    if expected is None or tmp.stat().st_size >= expected:
                        tmp.rename(tar_path)
                        break
                except Exception as exc:
                    wait = min(30 * attempt, 180)
                    logger.warning(f"{sid}: attempt {attempt} failed ({exc}); sleep {wait}s")
                    time.sleep(wait)
            if not tar_path.exists():
                raise RuntimeError(f"download incomplete after {max_retries} attempts")
        rec["size_mb"] = round(tar_path.stat().st_size / 1e6, 2)
        if extract:
            storm_dir = OUT_DIR / sid
            storm_dir.mkdir(parents=True, exist_ok=True)
            existing = list(storm_dir.glob("*.nc"))
            if not existing:
                with tarfile.open(tar_path, "r:gz") as tf:
                    nc_members = [m for m in tf.getmembers() if m.name.endswith(".nc")]
                    for m in nc_members:
                        m.name = Path(m.name).name  # flatten any inner dirs
                        tf.extract(m, path=storm_dir, filter="data")
            rec["n_nc"] = len(list(storm_dir.glob("*.nc")))
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = str(exc)
        logger.warning(f"{sid}: {exc}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    logger.add(str(LOG), level="INFO", rotation="10 MB", retention="90 days")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    listing = fetch_listing()

    man = pd.read_csv(ROOT / "data" / "phase1" / "processed" / "sample_manifest.csv")
    sids = man.loc[man["INCLUDED"], "SID"].tolist()
    logger.info(f"Sample storms: {len(sids)}")

    matched = [(sid, listing[sid]) for sid in sids if sid in listing]
    missing = [sid for sid in sids if sid not in listing]
    logger.info(f"Matched {len(matched)}/{len(sids)}; missing from archive: {missing}")
    for sid in missing:
        print(f"MISSING: {sid}")

    if args.dry_run:
        print(f"coverage: {len(matched)}/{len(sids)}")
        return 0

    tasks = matched[: args.max] if args.max else matched
    logger.info(f"Downloading {len(tasks)} storms with {args.workers} workers")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, sid, fname, not args.no_extract): sid
                for sid, fname in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            records.append(rec)
            if i % 25 == 0 or i == len(tasks):
                logger.info(f"Progress: {i}/{len(tasks)}")

    mdf = pd.DataFrame(records)
    mdf.to_csv(MANIFEST, index=False)
    ok = (mdf["status"] == "ok").sum()
    n_nc = int(mdf["n_nc"].sum())
    gb = mdf["size_mb"].sum() / 1024
    logger.info(f"Done: {ok}/{len(mdf)} ok, {n_nc} nc files extracted, {gb:.2f} GB")
    print(f"DONE ok={ok}/{len(mdf)} nc_files={n_nc} size={gb:.2f}GB manifest={MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
