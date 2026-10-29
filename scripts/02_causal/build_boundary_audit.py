# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Deterministic boundary audit for the HOLDOUT_MASKED_V1 fit panel.

Records, with the same row key (`SID|%Y-%m-%dT%H:%M:%SZ`) and the same
sorted newline-joined SHA-256 convention as the phase-2 protocol:

  * ISO-only and realised-F24-only candidate rows;
  * the final strict mask (canonical SID/row fit years, F24_TIME<2018 and
    ISO_TIME+30h<2018, with equality rejected);
  * separate F24/embargo removal sets, row-key hashes and per-year counts;
  * the fit row-key hash actually used by the prior panel.

The output is deterministic (no timestamps) so re-running on the same panel
reproduces the file byte-for-byte.

Usage (from the release root):
    python scripts/02_causal/build_boundary_audit.py
    python scripts/02_causal/build_boundary_audit.py --panel ... --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wpac_causal.preprocessing import FIT_LABEL_EMBARGO, strict_fit_row_mask  # noqa: E402
DEFAULT_PANEL = (ROOT / "data" / "phase1" / "processed_masked" / "holdout_fit0017" /
                 "panel" / "wpac_tc_dataset_processed.parquet")
DEFAULT_OUT = (ROOT / "data" / "phase1" / "processed_masked" / "holdout_fit0017" /
               "panel" / "boundary_audit.json")
FIT_END = "2018-01-01"


def row_keys(frame: pd.DataFrame) -> pd.Series:
    iso = pd.to_datetime(frame["ISO_TIME"], utc=True)
    return (frame["SID"].astype(str) + "|"
            + iso.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


def sha256_array(values) -> str:
    text = "\n".join(sorted(str(v) for v in values))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit(panel: pd.DataFrame, fit_end: str = FIT_END) -> dict:
    iso = pd.to_datetime(panel["ISO_TIME"], utc=True)
    f24 = pd.to_datetime(panel["F24_TIME"], utc=True)
    if "storm_year" not in panel.columns:
        raise ValueError("panel must carry the canonical storm_year column")
    boundary = pd.Timestamp(fit_end, tz="UTC")
    last_fit_year = boundary.year - 1
    sid_year = pd.to_numeric(panel["SID"].astype(str).str[:4], errors="raise").astype(int)
    storm = sid_year.between(2000, last_fit_year)
    iso_only = storm & iso.dt.year.between(2000, last_fit_year) & (iso < boundary)
    f24_only = iso_only & f24.notna() & (f24 < boundary)
    strict = strict_fit_row_mask(panel, [2000, last_fit_year])
    removed_f24 = iso_only & ~f24_only
    removed_embargo = f24_only & ~strict
    diff = iso_only & ~strict
    keys_strict = row_keys(panel.loc[strict])
    keys_diff = row_keys(panel.loc[diff])
    return {
        "protocol": "HOLDOUT_MASKED_V1",
        "fit_end": str(boundary),
        "embargo_hours": FIT_LABEL_EMBARGO.total_seconds() / 3600,
        "strict_boundary": "F24_TIME < fit_end and ISO_TIME + 30h < fit_end",
        "row_uid_format": "SID|%Y-%m-%dT%H:%M:%SZ",
        "sha256_method": "sha256 of sorted row keys joined by newline",
        "panel_rows": int(len(panel)),
        "panel_storms": int(panel["SID"].nunique()),
        "fit_rows_iso_only": int(iso_only.sum()),
        "fit_rows_with_f24_only": int(f24_only.sum()),
        "fit_rows_with_f24_boundary": int(strict.sum()),
        "fit_rows_strict_30h": int(strict.sum()),
        "fit_rows_removed_by_f24": int(removed_f24.sum()),
        "fit_rows_removed_by_embargo": int(removed_embargo.sum()),
        "fit_mask_difference_rows": int(diff.sum()),
        "fit_sids_iso_only": int(panel.loc[iso_only, "SID"].nunique()),
        "fit_sids_with_f24_boundary": int(panel.loc[strict, "SID"].nunique()),
        "fit_ri_rows_with_f24_boundary": int(panel.loc[strict, "RI"].sum()),
        "fit_row_key_sha256": sha256_array(keys_strict),
        "fit_iso_only_row_key_sha256": sha256_array(row_keys(panel.loc[iso_only])),
        "fit_f24_only_row_key_sha256": sha256_array(row_keys(panel.loc[f24_only])),
        "f24_removed_row_key_sha256": sha256_array(row_keys(panel.loc[removed_f24])),
        "embargo_removed_row_key_sha256": sha256_array(row_keys(panel.loc[removed_embargo])),
        "difference_row_key_sha256": sha256_array(keys_diff),
        "fit_rows_by_year": {
            str(int(y)): int(n)
            for y, n in panel.loc[strict].groupby(iso[strict].dt.year).size().items()
        },
        "max_feature_time": str(iso[strict].max()),
        "max_label_time": str(f24[strict].max()),
        "conclusion": (
            "The final fit mask requires canonical SID/row fit years, realised "
            "F24_TIME < fit_end and ISO_TIME + 30h < fit_end; equality is removed."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--fit-end", default=FIT_END)
    args = ap.parse_args()
    panel = pd.read_parquet(args.panel)
    result = audit(panel, args.fit_end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
