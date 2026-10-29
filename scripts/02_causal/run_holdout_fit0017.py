# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Build the fit-only panel and masked causal prior for HOLDOUT_MASKED_V1.

Stages:
    panel     best-track + ERA5 extraction + fit-only transforms + boundary audit
    prior     PCMCI+ / DoWhy / knowledge base / identification / trusted signs
    effects   re-run ONLY the effect stage on the existing panel +
              prior/pcmci_results.pkl: ATE -> knowledge base -> identification
              status -> trusted signs -> prior manifest.  PCMCI+ and the panel
              are never recomputed; missing panel/pkl hard-fails.
    manifest  re-emit panel/boundary/prior manifests from EXISTING artifacts
              (no PCMCI/ATE recomputation; used after schema changes)
    all       panel + prior
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from main import run_best_track, run_causal_discovery, run_effect_estimation, run_knowledge_base, run_preprocessing
from wpac_causal.effect_estimation import normalize_text_lf, sha256_text_lf
from wpac_causal.identification_policy import POLICY_ID
from wpac_causal.preprocessing import (
    IMPUTATION_KEY_COLUMNS, IMPUTATION_PREFIX, strict_fit_row_mask,
)
from wpac_causal.utils import Config, setup_logging

DEFAULT_BASE_CONFIG = "configs/phase1/config_alltc_0024.yaml"
DEFAULT_OUT_ROOT = "data/phase1/processed_masked/holdout_fit0017"
OUT_ROOT = Path(DEFAULT_OUT_ROOT)
PANEL_DIR = OUT_ROOT / "panel"
PRIOR_DIR = OUT_ROOT / "prior"

# ALL-TC scope: every WNP system in the base config's window (725 for
# config_alltc_0024), no peak-intensity filter, no storm-count branch anywhere in
# this pipeline.  The 375 >=65 kt sample is NOT an analysis branch of
# HOLDOUT_MASKED_V1.
REQUIRED_BEST_TRACK_KEYS = {
    "storm_min_peak_intensity": 0,
    "min_intensity": 0,
}
# sample manifest the base (all-TC) run wrote; used to prove the panel under
# OUT_ROOT really covers the same storm universe as the base config.
BASE_SAMPLE_MANIFEST: Path | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text_lf(path: Path) -> str:
    return normalize_text_lf(path.read_bytes().decode("utf-8"))


def identification_dot_hashes(dot_path: Path) -> dict:
    return {
        "sha256_text_lf": sha256_text_lf(read_text_lf(dot_path)),
        "sha256_file_bytes": sha256(dot_path),
    }


def verify_declared_dot_hash(declared, hashes: dict) -> str:
    value = str(declared or "").lower()
    if value == hashes["sha256_text_lf"]:
        return "utf8_lf_text"
    if value == hashes["sha256_file_bytes"]:
        return "legacy_file_bytes"
    raise RuntimeError("identification_dag.dot does not match its JSON hash")


def row_keys(frame: pd.DataFrame) -> pd.Series:
    iso = pd.to_datetime(frame["ISO_TIME"], utc=True)
    return (frame["SID"].astype(str) + "|"
            + iso.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


def sha256_array(values) -> str:
    text = "\n".join(sorted(str(v) for v in values))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _posix_hashes(paths) -> dict:
    hashes = {}
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"required manifest artifact missing: {path}")
        rel = path.resolve().relative_to(OUT_ROOT.resolve()).as_posix()
        hashes[rel] = sha256(path)
    return hashes


def storm_years(frame: pd.DataFrame) -> pd.Series:
    years = pd.to_numeric(frame["SID"].astype(str).str[:4], errors="raise").astype(int)
    return pd.Series(years.values, index=frame.index, name="storm_year")


def included_sample_rows(sample_manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"SID", "YEAR", "INCLUDED"}
    missing = required - set(sample_manifest.columns)
    if missing:
        raise RuntimeError(f"sample manifest missing columns: {sorted(missing)}")
    included = sample_manifest["INCLUDED"]
    if included.dtype != bool:
        included = included.astype(str).str.lower().isin({"true", "1"})
    frame = sample_manifest.loc[included].copy()
    if frame["SID"].duplicated().any():
        raise RuntimeError("Duplicated SID in canonical sample manifest")
    return frame


def sid_year_provenance(frame: pd.DataFrame, year_col: str = "YEAR") -> dict:
    if frame["SID"].duplicated().any():
        raise RuntimeError("SID/year provenance requires unique SIDs")
    sid_years = {str(sid): int(year) for sid, year in
                 zip(frame["SID"], frame[year_col])}
    sids = sorted(sid_years)
    content = sha256_array(f"{sid}|{sid_years[sid]}" for sid in sids)
    return {
        "sids": sids,
        "sid_years": {sid: sid_years[sid] for sid in sids},
        "sids_sha256": sha256_array(sids),
        "sid_years_sha256": content,
        "content_sha256": content,
    }


def find_label_eligible_sids(frame: pd.DataFrame) -> set[str]:
    """SIDs with at least one realised, non-null 24h best-track label."""
    required = {"SID", "VMAX_F24", "F24_TIME", "DV24"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"best-track label audit missing columns: {sorted(missing)}")
    valid = (
        frame["VMAX_F24"].notna()
        & frame["F24_TIME"].notna()
        & pd.to_numeric(frame["DV24"], errors="coerce").notna()
    )
    return set(frame.loc[valid, "SID"].astype(str))


def build_split_manifest(sample_manifest: pd.DataFrame,
                         label_eligible_sids: set[str]) -> pd.DataFrame:
    """Split only canonical systems with a real F24 label, never all 725."""
    frame = included_sample_rows(sample_manifest)
    canonical = set(frame["SID"].astype(str))
    eligible = set(map(str, label_eligible_sids))
    foreign = sorted(eligible - canonical)
    if foreign:
        raise RuntimeError(f"Label-eligible SIDs outside canonical sample: {foreign}")
    frame = frame.loc[frame["SID"].astype(str).isin(eligible)].copy()
    frame["storm_year"] = storm_years(frame)
    if not (frame["storm_year"] == frame["YEAR"].astype(int)).all():
        raise RuntimeError("Sample-manifest YEAR disagrees with the canonical SID first year")
    frame["split"] = pd.cut(
        frame["storm_year"], bins=[1999, 2017, 2019, 2024], labels=["fit", "inner", "test"]
    ).astype(str)
    if frame["split"].isna().any() or frame["SID"].duplicated().any():
        raise RuntimeError("Invalid or duplicated SID in split manifest")
    return frame[["SID", "NAME", "storm_year", "split"]].sort_values("SID").reset_index(drop=True)


def select_fit_prior_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"SID", "ISO_TIME", "F24_TIME"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    mask = strict_fit_row_mask(frame, [2000, 2017])
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise RuntimeError("No fit rows selected")
    if not strict_fit_row_mask(selected, [2000, 2017]).all():
        raise RuntimeError("Non-fit or boundary-crossing row entered the causal prior")
    return selected


def set_out_root(path: str | Path) -> Path:
    """Point every output path at ``path`` (module-level, used by all stages)."""
    global OUT_ROOT, PANEL_DIR, PRIOR_DIR
    OUT_ROOT = Path(path)
    PANEL_DIR = OUT_ROOT / "panel"
    PRIOR_DIR = OUT_ROOT / "prior"
    return OUT_ROOT


def configure(base_config: str | Path = DEFAULT_BASE_CONFIG) -> Config:
    """Holdout config derived from an ALL-TC base config (never the 375 one).

    The base config supplies the sample universe (``config_alltc_0024.yaml``:
    725 WNP systems, ``storm_min_peak_intensity=0``) and the variable list; this
    function rewrites the paths, fixes the fit-only transformation window and
    enforces complete extraction without a storm-duration filter.
    """
    base = Config.from_yaml(str(base_config)).to_dict()
    cfg = deepcopy(base)
    global BASE_SAMPLE_MANIFEST
    base_sample = base.get("best_track", {}).get("sample_manifest_csv")
    BASE_SAMPLE_MANIFEST = Path(base_sample) if base_sample else None
    cfg["data"]["output_dir"] = str(PANEL_DIR)
    cfg["data"]["log_dir"] = str(OUT_ROOT / "logs")
    cfg["data"]["retry_json"] = str(OUT_ROOT / "retry.json")
    cfg["best_track"]["sample_manifest_csv"] = str(PANEL_DIR / "sample_manifest.csv")
    for key, want in REQUIRED_BEST_TRACK_KEYS.items():
        have = cfg["best_track"].get(key)
        if int(have) != want:
            raise RuntimeError(
                f"{base_config}: best_track.{key}={have!r} but the all-system "
                f"HOLDOUT_MASKED_V1 pipeline requires {want} (no peak-intensity "
                f"filter and full lifecycle)")
    expected_treatments = {
        "SHRD", "RHLO", "VORT", "PV", "D200", "WARM_CORE", "MFC",
        "W500", "OHC_IC", "OHC_OA",
    }
    treatments = list(cfg["effect_estimation"].get("treatments", []))
    if len(treatments) != len(expected_treatments) or set(treatments) != expected_treatments:
        raise RuntimeError(
            "holdout effect_estimation.treatments must be the 10 A1 treatments, "
            "including OHC_OA")
    cfg["preprocessing"]["scaler_fit_years"] = [2000, 2017]
    cfg["preprocessing"]["min_obs_per_storm"] = 1
    cfg["preprocessing"]["require_complete_extraction"] = True
    cfg["preprocessing"]["extraction_audit_json"] = str(PANEL_DIR / "extraction_audit.json")
    cfg["preprocessing"]["yj_transformer_path"] = str(PANEL_DIR / "dv24_yj_transformer.pkl")
    cfg["preprocessing"]["scaler_path"] = str(PANEL_DIR / "feature_scaler.pkl")
    mask_features = list(dict.fromkeys([
        *cfg["extraction"].get("derived_variables", []),
        cfg["best_track"].get("lat_col", "USA_LAT"),
        cfg["best_track"].get("lon_col", "USA_LON"),
    ]))
    mask_path = PANEL_DIR / "imputation_mask.parquet"
    cfg["preprocessing"]["imputation_mask_columns"] = mask_features
    cfg["preprocessing"]["imputation_mask_path"] = str(mask_path)
    cfg["causal_discovery"]["imputation_mask_path"] = str(mask_path)
    cfg["causal_discovery"]["require_imputation_mask"] = True
    cfg["causal_discovery"]["results_pkl"] = str(PRIOR_DIR / "pcmci_results.pkl")
    cfg["causal_discovery"]["graph_png"] = str(PRIOR_DIR / "causal_graph.png")
    cfg["causal_discovery"]["graph_dot"] = str(PRIOR_DIR / "causal_graph.dot")
    cfg["effect_estimation"]["ate_csv"] = str(PRIOR_DIR / "ate_results.csv")
    cfg["effect_estimation"]["cate_csv"] = str(PRIOR_DIR / "cate_results.csv")
    cfg["effect_estimation"]["refutation_md"] = str(PRIOR_DIR / "refutation_report.md")
    cfg["effect_estimation"]["identification_dot"] = str(PRIOR_DIR / "identification_dag.dot")
    cfg["effect_estimation"]["identification_json"] = str(PRIOR_DIR / "identification_dag.json")
    cfg["knowledge_base"]["output_yaml"] = str(PRIOR_DIR / "causal_effects_db.yaml")
    cfg["knowledge_base"]["graph_edges_json"] = str(PRIOR_DIR / "causal_graph_edges.json")
    cfg["logging"]["file"] = str(OUT_ROOT / "run.log")
    return Config(cfg)


def write_config(cfg: Config) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, allow_unicode=True)


def assert_sample_universe(sample_manifest: str | Path) -> dict:
    """Require the exact canonical SID/year universe from the base config."""
    sample_path = Path(sample_manifest)
    if not sample_path.is_file():
        raise FileNotFoundError(f"sample manifest missing: {sample_path}")
    if BASE_SAMPLE_MANIFEST is None:
        raise RuntimeError("base sample manifest is not configured")
    if not BASE_SAMPLE_MANIFEST.is_file():
        raise FileNotFoundError(f"base sample manifest missing: {BASE_SAMPLE_MANIFEST}")
    sample = included_sample_rows(pd.read_csv(sample_path))
    base = included_sample_rows(pd.read_csv(BASE_SAMPLE_MANIFEST))
    got_prov = sid_year_provenance(sample)
    want_prov = sid_year_provenance(base)
    got, want = set(got_prov["sids"]), set(want_prov["sids"])
    if got != want:
        raise RuntimeError(
            f"{sample_path} covers {len(got)} storms but the base config's "
            f"manifest {BASE_SAMPLE_MANIFEST} has {len(want)} "
            f"(missing={len(want - got)}, extra={len(got - want)}); rebuild the "
            f"panel with --base-config instead of refreshing a foreign panel")
    if got_prov["content_sha256"] != want_prov["content_sha256"]:
        raise RuntimeError("sample manifest SID/year content differs from the base manifest")
    return {
        "n_included": len(got),
        "base_sample_manifest": str(BASE_SAMPLE_MANIFEST),
        "canonical_content_sha256": got_prov["content_sha256"],
        "canonical_sids_sha256": got_prov["sids_sha256"],
    }


def _configured_imputation_features() -> list[str]:
    snapshot_path = OUT_ROOT / "config_snapshot.yaml"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"config snapshot missing: {snapshot_path}")
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    features = list(snapshot.get("preprocessing", {}).get(
        "imputation_mask_columns", []) or [])
    if not features or len(features) != len(set(features)):
        raise RuntimeError("config_snapshot lacks a valid imputation-mask feature schema")
    return features


def _imputation_mask_schema(panel: pd.DataFrame,
                            expected_features: list[str]) -> dict:
    """Validate the full panel mask and return its manifest/schema block."""
    path = PANEL_DIR / "imputation_mask.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"imputation mask missing: {path}")
    mask = pd.read_parquet(path)
    if not set(IMPUTATION_KEY_COLUMNS) <= set(mask.columns):
        raise RuntimeError("imputation mask lacks SID+ISO_TIME key columns")
    data_columns = [c for c in mask.columns if c not in IMPUTATION_KEY_COLUMNS]
    expected_columns = [IMPUTATION_PREFIX + name for name in expected_features]
    if (len(data_columns) != len(expected_columns)
            or set(data_columns) != set(expected_columns)):
        raise RuntimeError(
            "imputation mask columns disagree with config_snapshot: "
            f"missing={sorted(set(expected_columns) - set(data_columns))} "
            f"extra={sorted(set(data_columns) - set(expected_columns))}")
    for frame, name in ((panel, "panel"), (mask, "imputation mask")):
        if not set(IMPUTATION_KEY_COLUMNS) <= set(frame.columns):
            raise RuntimeError(f"{name} lacks SID+ISO_TIME keys")
        keys = frame[IMPUTATION_KEY_COLUMNS]
        if keys.isna().any().any() or keys.duplicated().any():
            raise RuntimeError(f"{name} SID+ISO_TIME keys must be non-null and unique")
    panel_keys = set(row_keys(panel))
    mask_keys = set(row_keys(mask))
    if panel_keys != mask_keys:
        raise RuntimeError(
            "imputation mask row keys differ from the panel: "
            f"missing={len(panel_keys - mask_keys)} extra={len(mask_keys - panel_keys)}")
    for column in data_columns:
        if not pd.api.types.is_bool_dtype(mask[column].dtype):
            raise RuntimeError(f"imputation mask column {column} is not boolean")
    values = mask[data_columns].to_numpy(dtype=bool)
    return {
        "path": "panel/imputation_mask.parquet",
        "sha256": sha256(path),
        "schema": {
            "key_columns": list(IMPUTATION_KEY_COLUMNS),
            "prefix": IMPUTATION_PREFIX,
            "row_uid_format": "SID|%Y-%m-%dT%H:%M:%SZ",
        },
        "columns": data_columns,
        "feature_columns": [c[len(IMPUTATION_PREFIX):] for c in data_columns],
        "rows": len(mask),
        "row_key_sha256": sha256_array(row_keys(mask)),
        "masked_cells": int(values.sum()),
        "masked_cells_by_variable": {
            column: int(mask[column].sum()) for column in data_columns},
    }


def _fit_imputation_counts(fit: pd.DataFrame,
                           used_columns: list[str]) -> dict:
    """Recompute the mask counts for exactly the fit rows used by PCMCI."""
    mask_path = PANEL_DIR / "imputation_mask.parquet"
    mask = pd.read_parquet(mask_path)
    missing_columns = sorted(set(used_columns) - set(mask.columns))
    if missing_columns:
        raise RuntimeError(
            f"fit imputation reconciliation lacks mask columns: {missing_columns}")
    merged = fit[IMPUTATION_KEY_COLUMNS].merge(
        mask[IMPUTATION_KEY_COLUMNS + used_columns],
        on=IMPUTATION_KEY_COLUMNS, how="left", validate="one_to_one",
        indicator=True, sort=False)
    if (merged["_merge"] != "both").any():
        raise RuntimeError("imputation mask lacks rows used by the fit prior")
    by_variable = {
        column[len(IMPUTATION_PREFIX):]: int(merged[column].sum())
        for column in used_columns}
    rows = int(merged[used_columns].any(axis=1).sum()) if used_columns else 0
    return {
        "rows": len(fit),
        "rows_with_imputation": rows,
        "cells": int(sum(by_variable.values())),
        "cells_by_variable": by_variable,
    }


def _write_population_audit(raw_path: Path, split_path: Path,
                            processed_path: Path | None = None,
                            failure: dict | None = None) -> dict:
    """Bind canonical, label-eligible, panel and prior populations by content."""
    sample_path = PANEL_DIR / "sample_manifest.csv"
    extraction_path = PANEL_DIR / "extraction_audit.json"
    sample = included_sample_rows(pd.read_csv(sample_path))
    split = pd.read_csv(split_path)
    canonical = sid_year_provenance(sample)
    eligible = sid_year_provenance(split, "storm_year")
    canonical_years = canonical["sid_years"]
    if not set(eligible["sids"]) <= set(canonical["sids"]):
        raise RuntimeError("split manifest contains SIDs outside the canonical sample")
    if any(canonical_years[sid] != eligible["sid_years"][sid]
           for sid in eligible["sids"]):
        raise RuntimeError("split manifest disagrees with canonical SID/year content")

    raw = pd.read_parquet(raw_path)
    raw_eligible = find_label_eligible_sids(raw)
    if raw_eligible != set(eligible["sids"]):
        raise RuntimeError(
            "split manifest is not exactly the best-track label-eligible SID set: "
            f"missing={sorted(set(eligible['sids']) - raw_eligible)} "
            f"extra={sorted(raw_eligible - set(eligible['sids']))}")
    excluded = sorted(set(canonical["sids"]) - raw_eligible)
    sample_by_sid = sample.set_index(sample["SID"].astype(str))
    exclusion_rows = [{
        "SID": sid,
        "YEAR": int(sample_by_sid.loc[sid, "YEAR"]),
        "NAME": str(sample_by_sid.loc[sid, "NAME"]) if "NAME" in sample_by_sid else "",
        "reason": "no_valid_f24_label_after_best_track",
    } for sid in excluded]

    if extraction_path.is_file():
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        extraction_sha = sha256(extraction_path)
    else:
        extraction = {"status": "missing", "failures": []}
        extraction_sha = None
    split_counts = {str(k): int(v) for k, v in split["split"].value_counts().items()}
    audit = {
        "protocol": "HOLDOUT_MASKED_V1",
        "status": "failed" if failure else "complete",
        "population": "alltc",
        "n_systems": len(canonical["sids"]),
        "n_label_eligible": len(eligible["sids"]),
        "n_label_ineligible": len(excluded),
        "label_ineligible_sids": excluded,
        "label_ineligible_reasons": exclusion_rows,
        "canonical_sample": {
            "path": "panel/sample_manifest.csv",
            "sha256": sha256(sample_path),
            "n_included": len(canonical["sids"]),
            **canonical,
        },
        "label_eligible": eligible,
        "split": {
            **eligible,
            "n_fit_systems": int((split["split"] == "fit").sum()),
            "split_counts": split_counts,
        },
        "extraction": {
            "path": "panel/extraction_audit.json",
            "sha256": extraction_sha,
            **extraction,
        },
    }
    if failure:
        audit["failure"] = failure

    if processed_path is not None:
        processed = pd.read_parquet(processed_path)
        processed_sids = set(processed["SID"].astype(str))
        if processed_sids != raw_eligible:
            raise RuntimeError(
                "processed panel does not preserve every label-eligible SID: "
                f"missing={sorted(raw_eligible - processed_sids)} "
                f"extra={sorted(processed_sids - raw_eligible)}")
        fit = select_fit_prior_rows(processed)
        fit_sids = set(fit["SID"].astype(str))
        fit_split = split.loc[split["SID"].astype(str).isin(fit_sids)]
        audit["panel"] = {
            **eligible,
            "n_rows": len(processed),
            "row_key_sha256": sha256_array(row_keys(processed)),
        }
        audit["prior"] = {
            **sid_year_provenance(fit_split, "storm_year"),
            "n_rows": len(fit),
            "n_fit_systems": len(fit_sids),
            "row_key_sha256": sha256_array(row_keys(fit)),
        }
        if not failure:
            audit["imputation_mask"] = _imputation_mask_schema(
                processed, _configured_imputation_features())
            if not extraction_path.is_file():
                raise FileNotFoundError(f"extraction audit missing: {extraction_path}")
            required_extraction = {
                "status", "expected_sids", "successful_sids", "failures",
                "min_obs_exclusions", "postprocessing_missing_sids",
                "postprocessing_extra_sids",
            }
            if not required_extraction <= set(extraction):
                raise RuntimeError("extraction audit lacks required fail-closed fields")
            if (extraction["status"] != "complete"
                    or set(extraction["expected_sids"]) != raw_eligible
                    or set(extraction["successful_sids"]) != raw_eligible
                    or extraction["failures"]
                    or extraction["min_obs_exclusions"]
                    or extraction["postprocessing_missing_sids"]
                    or extraction["postprocessing_extra_sids"]):
                raise RuntimeError("population audit cannot bless incomplete extraction")

    out = PANEL_DIR / "population_audit.json"
    out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def _write_panel_manifest(raw_path: Path, processed_path: Path,
                          split_path: Path) -> None:
    """Panel manifest + population/boundary audits from existing panel files."""
    df_proc = pd.read_parquet(processed_path)
    split = pd.read_csv(split_path)
    universe = assert_sample_universe(PANEL_DIR / "sample_manifest.csv")
    population = _write_population_audit(raw_path, split_path, processed_path)
    if universe["canonical_content_sha256"] != population["canonical_sample"]["content_sha256"]:
        raise RuntimeError("panel population audit differs from the base sample universe")
    population_path = PANEL_DIR / "population_audit.json"
    extraction_path = PANEL_DIR / "extraction_audit.json"
    mask_path = PANEL_DIR / "imputation_mask.parquet"
    audit_path = PANEL_DIR / "boundary_audit.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_boundary_audit.py"),
         "--panel", str(processed_path), "--out", str(audit_path)],
        cwd=ROOT, check=True)
    manifest_paths = [raw_path, processed_path, split_path,
                      PANEL_DIR / "sample_manifest.csv", population_path,
                      extraction_path, mask_path,
                      PANEL_DIR / "dv24_yj_transformer.pkl",
                      PANEL_DIR / "feature_scaler.pkl"]
    manifest = {
        "protocol": "HOLDOUT_MASKED_V1",
        "fit_years": [2000, 2017],
        "inner_years": [2018, 2019],
        "test_years": [2020, 2024],
        "rows": len(df_proc),
        "storms": int(df_proc["SID"].nunique()),
        "population": "alltc",
        "sample_universe": universe,
        "canonical_sample": {
            key: population["canonical_sample"][key]
            for key in ("content_sha256", "sids_sha256", "sid_years_sha256")},
        "label_eligible": {
            key: population["label_eligible"][key]
            for key in ("content_sha256", "sids_sha256", "sid_years_sha256")},
        "population_audit": {
            "path": "panel/population_audit.json",
            "sha256": sha256(population_path)},
        "imputation_mask": population["imputation_mask"],
        "split_counts": {str(k): int(v) for k, v in split["split"].value_counts().items()},
        "row_key_sha256_fit_eligible": population["prior"]["row_key_sha256"],
        "hashes": {
            **{p.name: sha256(p) for p in manifest_paths},
            "boundary_audit.json": sha256(audit_path),
        },
        "boundary_audit": {
            "path": "panel/boundary_audit.json",
            "sha256": sha256(audit_path)},
    }
    with (PANEL_DIR / "panel_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def run_panel(cfg: Config, resume: bool) -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = PANEL_DIR / "wpac_tc_dataset.parquet"
    processed_path = PANEL_DIR / "wpac_tc_dataset_processed.parquet"
    split_path = PANEL_DIR / "split_manifest.csv"
    if processed_path.exists() and split_path.exists() and resume:
        logger.info("Panel already complete; explicit resume skips regeneration")
        _write_panel_manifest(raw_path, processed_path, split_path)
        return
    if processed_path.exists() and not resume:
        raise RuntimeError(f"Panel exists at {processed_path}; pass --resume to continue")
    logger.info("Running best-track processing")
    df_bt = run_best_track(cfg)
    df_bt["storm_year"] = storm_years(df_bt)
    df_bt.to_parquet(raw_path)
    sample = pd.read_csv(PANEL_DIR / "sample_manifest.csv")
    universe = assert_sample_universe(PANEL_DIR / "sample_manifest.csv")
    logger.info(f"All-system canonical universe: {universe['n_included']} storms")
    eligible_sids = find_label_eligible_sids(df_bt)
    split = build_split_manifest(sample, eligible_sids)
    split.to_csv(split_path, index=False)
    logger.info(
        f"Label-eligible universe: {len(eligible_sids)} storms; "
        f"label-ineligible: {universe['n_included'] - len(eligible_sids)}")
    logger.info(f"Split counts:\n{split.groupby(['split', 'storm_year']).size()}")
    logger.info("Running ERA5/GLORYS extraction and fit-only transforms")
    try:
        df_proc = run_preprocessing(cfg, df_bt)
        if "F24_TIME" not in df_proc.columns:
            raise RuntimeError("F24_TIME was not preserved through preprocessing")
        df_proc["storm_year"] = storm_years(df_proc)
        processed_sids = set(df_proc["SID"].astype(str))
        if processed_sids != eligible_sids:
            raise RuntimeError(
                "processed panel changed the label-eligible population: "
                f"missing={sorted(eligible_sids - processed_sids)} "
                f"extra={sorted(processed_sids - eligible_sids)}")
        df_proc.to_parquet(processed_path)
    except Exception as exc:
        _write_population_audit(
            raw_path, split_path,
            failure={"stage": "preprocessing", "type": type(exc).__name__,
                     "message": str(exc)})
        raise
    _write_panel_manifest(raw_path, processed_path, split_path)


def _write_prior_manifest(fit: pd.DataFrame, result: Dict) -> None:
    """Phase-1 prior manifest with mandatory population and DAG hash chains."""
    boundary = json.loads((PANEL_DIR / "boundary_audit.json").read_text(encoding="utf-8"))
    panel_manifest = json.loads((PANEL_DIR / "panel_manifest.json").read_text(encoding="utf-8"))
    population_path = PANEL_DIR / "population_audit.json"
    population = json.loads(population_path.read_text(encoding="utf-8"))
    fit_uid_sha = sha256_array(row_keys(fit))
    if boundary["fit_rows_with_f24_boundary"] != len(fit):
        raise RuntimeError(
            "prior rows disagree with boundary_audit.fit_rows_with_f24_boundary")
    if boundary["fit_row_key_sha256"] != fit_uid_sha:
        raise RuntimeError("prior row-key hash disagrees with boundary_audit")
    if int(panel_manifest["rows"]) < len(fit):
        raise RuntimeError("panel_manifest rows disagree with the panel")
    if panel_manifest.get("population_audit", {}).get("sha256") != sha256(population_path):
        raise RuntimeError("panel_manifest population_audit hash is stale")
    if population["prior"]["row_key_sha256"] != fit_uid_sha:
        raise RuntimeError("prior row-key hash disagrees with population_audit")
    if set(population["prior"]["sids"]) != set(fit["SID"].astype(str)):
        raise RuntimeError("prior SIDs disagree with population_audit")
    processed = pd.read_parquet(PANEL_DIR / "wpac_tc_dataset_processed.parquet")
    mask_schema = _imputation_mask_schema(processed, _configured_imputation_features())
    if population.get("imputation_mask") != mask_schema:
        raise RuntimeError("population_audit imputation-mask schema or hash is stale")
    if panel_manifest.get("imputation_mask") != mask_schema:
        raise RuntimeError("panel_manifest imputation-mask schema or hash is stale")
    upstream = result.get("metadata", {}).get("upstream_imputation")
    if not isinstance(upstream, dict) or upstream.get("status") != "applied":
        raise RuntimeError("PCMCI metadata lacks the required upstream imputation-mask audit")
    if int(upstream.get("input_cells", -1)) < 0:
        raise RuntimeError("PCMCI upstream imputation-mask count is invalid")
    if (upstream.get("sha256") != mask_schema["sha256"]
            or upstream.get("row_key_sha256") != mask_schema["row_key_sha256"]
            or list(upstream.get("mask_columns", [])) != mask_schema["columns"]):
        raise RuntimeError("PCMCI metadata is not bound to the current imputation mask")
    if Path(upstream.get("path", "")).resolve() != (
            PANEL_DIR / "imputation_mask.parquet").resolve():
        raise RuntimeError("PCMCI upstream imputation-mask path differs from the panel")
    used_columns = list(upstream.get("used_columns", []))
    fit_imputation_counts = _fit_imputation_counts(fit, used_columns)
    upstream_counts = {
        str(key): int(value) for key, value in
        upstream.get("input_cells_by_variable", {}).items()}
    if (int(upstream.get("input_rows", -1)) != fit_imputation_counts["rows"]
            or int(upstream.get("input_cells", -1)) != fit_imputation_counts["cells"]
            or int(upstream.get("input_rows_with_imputation", -1))
            != fit_imputation_counts["rows_with_imputation"]
            or upstream_counts != fit_imputation_counts["cells_by_variable"]):
        raise RuntimeError(
            "PCMCI fit imputation counts do not match the mask file: "
            f"metadata={upstream_counts}/{upstream.get('input_cells')} vs "
            f"mask={fit_imputation_counts['cells_by_variable']}/"
            f"{fit_imputation_counts['cells']}")

    dot_path = PRIOR_DIR / "identification_dag.dot"
    dag_json_path = PRIOR_DIR / "identification_dag.json"
    if not dot_path.is_file() or not dag_json_path.is_file():
        raise FileNotFoundError("identification_dag.dot/json are required prior outputs")
    dag_payload = json.loads(dag_json_path.read_text(encoding="utf-8"))
    dot_hashes = identification_dot_hashes(dot_path)
    dot_hash_convention = verify_declared_dot_hash(
        dag_payload.get("dot_sha256"), dot_hashes)
    snapshot = yaml.safe_load((OUT_ROOT / "config_snapshot.yaml").read_text(encoding="utf-8"))
    snapshot_treatments = list(snapshot["effect_estimation"]["treatments"])
    if "OHC_OA" not in snapshot_treatments or len(snapshot_treatments) != 10:
        raise RuntimeError("config_snapshot does not contain all 10 treatments including OHC_OA")
    if list(dag_payload.get("config", {}).get("treatments", [])) != snapshot_treatments:
        raise RuntimeError("identification DAG treatments differ from config_snapshot")

    outputs = [
        PRIOR_DIR / "fit_prior_panel.parquet", PRIOR_DIR / "pcmci_results.pkl",
        PRIOR_DIR / "causal_graph_edges.json", PRIOR_DIR / "causal_graph.dot",
        PRIOR_DIR / "causal_effects_db.yaml", PRIOR_DIR / "ate_results.csv",
        PRIOR_DIR / "cate_results.csv", dot_path, dag_json_path,
        PRIOR_DIR / "identification_status.csv", PRIOR_DIR / "trusted_signs.json",
    ]
    inputs = [
        PANEL_DIR / "sample_manifest.csv", PANEL_DIR / "split_manifest.csv",
        PANEL_DIR / "panel_manifest.json", PANEL_DIR / "boundary_audit.json",
        population_path, PANEL_DIR / "extraction_audit.json",
        PANEL_DIR / "imputation_mask.parquet",
        PANEL_DIR / "wpac_tc_dataset_processed.parquet",
        PANEL_DIR / "dv24_yj_transformer.pkl",
        PANEL_DIR / "feature_scaler.pkl", OUT_ROOT / "config_snapshot.yaml",
    ]
    input_hashes = _posix_hashes(inputs)
    output_hashes = _posix_hashes(outputs)
    manifest = {
        "protocol": "HOLDOUT_MASKED_V1",
        "fit_years": [2000, 2017],
        "inner_years": [2018, 2019],
        "test_years": [2020, 2024],
        "sids": sorted(fit["SID"].astype(str).unique().tolist()),
        "sid_years": [int(y) for y in
                      fit.drop_duplicates("SID").sort_values("SID")["storm_year"]],
        "n_storms": int(fit["SID"].nunique()),
        "population": "alltc",
        "canonical_sample": {
            key: population["canonical_sample"][key]
            for key in ("content_sha256", "sids_sha256", "sid_years_sha256")},
        "label_eligible": {
            key: population["label_eligible"][key]
            for key in ("content_sha256", "sids_sha256", "sid_years_sha256")},
        "population_audit": {
            "path": "panel/population_audit.json",
            "sha256": sha256(population_path)},
        "imputation_mask": mask_schema,
        "upstream_imputation": upstream,
        "upstream_imputed_cells": int(upstream["input_cells"]),
        "upstream_imputed_rows": int(upstream["input_rows_with_imputation"]),
        "fit_imputation_reconciliation": fit_imputation_counts,
        "rows": len(fit),
        "ri_rows": int(fit["RI"].sum()),
        "row_key_sha256": fit_uid_sha,
        "row_uid_format": "SID|%Y-%m-%dT%H:%M:%SZ",
        "max_feature_time": str(pd.to_datetime(fit["ISO_TIME"], utc=True).max()),
        "max_label_time": str(pd.to_datetime(fit["F24_TIME"], utc=True).max()),
        "mask_type": result["metadata"].get("mask_type"),
        "config_snapshot": {
            "path": "config_snapshot.yaml",
            "sha256": sha256(OUT_ROOT / "config_snapshot.yaml")},
        "panel_manifest": {
            "path": "panel/panel_manifest.json",
            "sha256": sha256(PANEL_DIR / "panel_manifest.json")},
        "boundary_audit": {
            "path": "panel/boundary_audit.json",
            "sha256": sha256(PANEL_DIR / "boundary_audit.json")},
        "identification_dag": {
            "dot": {
                "path": "prior/identification_dag.dot",
                "sha256": output_hashes["prior/identification_dag.dot"],
                "sha256_text_lf": dot_hashes["sha256_text_lf"],
                "sha256_file_bytes": dot_hashes["sha256_file_bytes"],
                "json_hash_convention": dot_hash_convention,
            },
            "json": {"path": "prior/identification_dag.json",
                     "sha256": output_hashes["prior/identification_dag.json"]},
        },
        "identification_policy": {
            "id": POLICY_ID,
            "rule_documented_in": "src/identification_policy.py",
            "per_pair_audit": {
                "ate_column": "identification_candidates",
                "ate_policy_column": "identification_policy",
                "dag_json_block": "adjustment_policy",
            },
            "ate": {"path": "prior/ate_results.csv",
                    "sha256": output_hashes["prior/ate_results.csv"]},
        },
        "hashes": {**output_hashes, **input_hashes},
        "inputs": input_hashes,
        "outputs": output_hashes,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True)
    for name in ("phase1_prior_manifest.json", "prior_manifest.json"):
        (PRIOR_DIR / name).write_text(text, encoding="utf-8")


def run_identification() -> None:
    """Read the SAME DOT that EffectEstimator used; verify its hash first."""
    dot_path = PRIOR_DIR / "identification_dag.dot"
    json_path = PRIOR_DIR / "identification_dag.json"
    ate_path = PRIOR_DIR / "ate_results.csv"
    if not dot_path.is_file() or not json_path.is_file():
        raise RuntimeError(
            "EffectEstimator did not write identification_dag.{dot,json}")
    if not ate_path.is_file():
        raise RuntimeError(
            "ate_results.csv is missing; identification status must copy the "
            "adjustment sets the ATE actually used")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    dot_hashes = identification_dot_hashes(dot_path)
    dot_hash_convention = verify_declared_dot_hash(
        payload.get("dot_sha256"), dot_hashes)
    if dot_hash_convention == "legacy_file_bytes":
        logger.warning("identification_dag.json uses the legacy raw-byte DOT hash")
    script = ROOT / "paper_supplementary/code/identification_status.py"
    env = os.environ.copy()
    env.update({
        "WPAC_PCMCI_PKL": str(PRIOR_DIR / "pcmci_results.pkl"),
        "WPAC_PANEL": str(PRIOR_DIR / "fit_prior_panel.parquet"),
        "WPAC_IDENTIFICATION_OUT": str(PRIOR_DIR / "identification_status.csv"),
        "WPAC_IDENTIFICATION_DOT": str(dot_path),
        "WPAC_IDENTIFICATION_JSON": str(json_path),
        "WPAC_ATE_RESULTS": str(ate_path),
        "WPAC_A1_OUT": str(PRIOR_DIR / "dag_audit"),
    })
    subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, check=True)
    status = pd.read_csv(PRIOR_DIR / "identification_status.csv")
    if "dag_source" not in status.columns or set(status["dag_source"]) != {
            "identification_dag.dot"}:
        raise RuntimeError(
            "identification_status did not use identification_dag.dot")
    ate = pd.read_csv(ate_path)
    _verify_adjustment_sets_match_ate(status, ate)
    _verify_identification_policy(status, ate)


def _verify_adjustment_sets_match_ate(status: pd.DataFrame,
                                      ate: pd.DataFrame) -> None:
    """Fail-closed: every row must carry the set the ATE actually used."""
    def norm(value) -> tuple:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ()
        text = str(value).strip()
        if text in ("", "(empty)", "empty", "None", "nan"):
            return ()
        return tuple(sorted(p.strip() for p in text.split(",") if p.strip()))

    wanted = {(str(r.treatment), str(r.outcome)): norm(r.adjustment_set)
              for r in ate.itertuples()}
    got = {(str(r.treatment), str(r.outcome)): norm(r.adjustment_set)
           for r in status.itertuples()}
    if len(got) != len(status):
        raise RuntimeError("identification_status.csv has duplicate rows")
    missing = sorted(set(got) - set(wanted))
    if missing:
        raise RuntimeError(
            f"identification_status rows absent from the ATE table: {missing}")
    diffs = {key: (got[key], wanted[key]) for key in got
             if got[key] != wanted[key]}
    if diffs:
        raise RuntimeError(
            "identification_status adjustment sets differ from the ATE: "
            + json.dumps({f"{k[0]}->{k[1]}": list(v) for k, v in
                          list(diffs.items())[:5]}))


def _verify_identification_policy(status: pd.DataFrame,
                                  ate: pd.DataFrame) -> None:
    """Fail-closed: both tables must record the deterministic policy id."""
    for frame, name in ((ate, "ate_results.csv"),
                        (status, "identification_status.csv")):
        if "identification_policy" not in frame.columns:
            raise RuntimeError(
                f"{name} has no identification_policy column; rerun the "
                "effects stage so the ATE carries the deterministic policy audit")
        observed = {str(value).strip() for value in frame["identification_policy"]}
        if observed != {POLICY_ID}:
            raise RuntimeError(
                f"{name} was not produced by {POLICY_ID}: "
                f"observed={sorted(observed)[:5]}")


def write_trusted_signs() -> None:
    ate = pd.read_csv(PRIOR_DIR / "ate_results.csv")
    ident = pd.read_csv(PRIOR_DIR / "identification_status.csv")
    dv24 = ate.loc[ate["outcome"] == "DV24"].merge(
        ident.loc[ident["outcome"] == "DV24"],
        on=["treatment", "outcome"], how="left", validate="one_to_one",
    )
    eligible = dv24.loc[
        (dv24["treatment"] != "D200")
        & (dv24["p_value_boot"] < 0.05)
        & (dv24["refutation_passed"] >= 2)
        & (dv24["no_directed_path"] == False)  # noqa: E712
    ].copy()
    signs = {}
    for row in eligible.itertuples():
        if row.ate > 0:
            signs[row.treatment] = 1
        elif row.ate < 0:
            signs[row.treatment] = -1
    payload = {
        "rule": ("treatment!=D200 and p_value_boot<0.05 and "
                 "refutation_passed>=2 and no_directed_path=False and ate!=0"),
        "explicit_exclusions": ["D200"],
        "signs": signs,
        "excluded": sorted(set(dv24["treatment"]) - set(signs)),
    }
    with (PRIOR_DIR / "trusted_signs.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_prior(cfg: Config) -> None:
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PANEL_DIR / "wpac_tc_dataset_processed.parquet"
    if not processed_path.exists():
        raise RuntimeError("Run the panel stage first")
    frame = pd.read_parquet(processed_path)
    fit = select_fit_prior_rows(frame)
    fit.to_parquet(PRIOR_DIR / "fit_prior_panel.parquet")
    logger.info(f"Fit-only prior: {fit['SID'].nunique()} SID, {len(fit)} rows, {fit['RI'].sum()} RI")
    run_causal_discovery(cfg, fit)
    with (PRIOR_DIR / "pcmci_results.pkl").open("rb") as handle:
        result = pickle.load(handle)
    if result["metadata"].get("mask_type") != "xyz":
        raise RuntimeError("PCMCI+ mask_type is not xyz")
    run_effect_estimation(cfg, fit)
    run_knowledge_base(cfg, fit)
    run_identification()
    write_trusted_signs()
    _write_prior_manifest(fit, result)


def run_effects(cfg: Config) -> None:
    """Re-run effects only: ATE -> KB -> identification -> signs -> manifest.

    Requires the existing panel and the existing ``prior/pcmci_results.pkl``;
    PCMCI+ and the panel are NEVER recomputed.  Missing panel/pkl or a fit
    panel that disagrees with the current panel hard-fails before any output is
    overwritten.  Every output keeps its canonical path, so the hash chains in
    ``phase1_prior_manifest.json`` (DOT/ATE/status/signs) are re-verified and
    re-written exactly as in the ``prior`` stage.
    """
    PRIOR_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PANEL_DIR / "wpac_tc_dataset_processed.parquet"
    if not processed_path.is_file():
        raise RuntimeError(
            f"effects stage requires the panel ({processed_path}); run --stage panel first")
    pkl_path = PRIOR_DIR / "pcmci_results.pkl"
    if not pkl_path.is_file():
        raise RuntimeError(
            f"effects stage requires an existing PCMCI+ result ({pkl_path}); "
            "run --stage prior first (PCMCI is not recomputed here)")
    fit_path = PRIOR_DIR / "fit_prior_panel.parquet"
    if not fit_path.is_file():
        raise RuntimeError(
            f"effects stage requires the fit-only prior panel ({fit_path})")
    expected = select_fit_prior_rows(pd.read_parquet(processed_path))
    fit = pd.read_parquet(fit_path)
    if sha256_array(row_keys(fit)) != sha256_array(row_keys(expected)):
        raise RuntimeError(
            "fit_prior_panel.parquet disagrees with select_fit_prior_rows on the "
            "current panel; rerun the prior stage instead of the effects stage")
    with pkl_path.open("rb") as handle:
        result = pickle.load(handle)
    if result["metadata"].get("mask_type") != "xyz":
        raise RuntimeError("PCMCI+ mask_type is not xyz")
    logger.info(
        f"Effects-only rerun on {len(fit)} rows / {fit['SID'].nunique()} SID "
        f"(PCMCI reused from {pkl_path}, policy={POLICY_ID})")
    run_effect_estimation(cfg, fit)
    run_knowledge_base(cfg, fit)
    run_identification()
    write_trusted_signs()
    _write_prior_manifest(fit, result)
    logger.info("Effects stage complete (PCMCI and panel untouched)")


def refresh_manifests(cfg: Config) -> None:
    """Regenerate manifests/audits from EXISTING artifacts (no PCMCI/ATE rerun)."""
    processed_path = PANEL_DIR / "wpac_tc_dataset_processed.parquet"
    if not processed_path.exists():
        raise RuntimeError("panel processed parquet is missing")
    if not (OUT_ROOT / "config_snapshot.yaml").exists():
        write_config(cfg)
    raw_path = PANEL_DIR / "wpac_tc_dataset.parquet"
    split_path = PANEL_DIR / "split_manifest.csv"
    if not split_path.exists():
        raise RuntimeError("split manifest is missing")
    _write_panel_manifest(raw_path, processed_path, split_path)
    frame = pd.read_parquet(processed_path)
    expected = select_fit_prior_rows(frame)
    fit = pd.read_parquet(PRIOR_DIR / "fit_prior_panel.parquet")
    if sha256_array(row_keys(fit)) != sha256_array(row_keys(expected)):
        raise RuntimeError(
            "fit_prior_panel.parquet disagrees with select_fit_prior_rows on the "
            "current panel; rerun the prior stage instead of the manifest stage")
    with (PRIOR_DIR / "pcmci_results.pkl").open("rb") as handle:
        result = pickle.load(handle)
    _write_prior_manifest(fit, result)
    logger.info("Manifests regenerated from existing artifacts (no recomputation)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",
                        choices=["panel", "prior", "effects", "manifest", "all"],
                        default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--base-config", default=DEFAULT_BASE_CONFIG,
        help="ALL-TC phase-1 config that defines the sample universe "
             "(default: configs/phase1/config_alltc_0024.yaml — 725 WNP systems, no "
             "peak-intensity filter).  The >=65 kt 375-system config is not a "
             "valid base for this protocol.")
    parser.add_argument(
        "--out-root", default=DEFAULT_OUT_ROOT,
        help="versioned output directory (panel/, prior/, manifests)")
    args = parser.parse_args()
    base = Path(args.base_config)
    if not base.is_file():
        raise FileNotFoundError(f"base config not found: {base}")
    set_out_root(args.out_root)
    cfg = configure(base)
    setup_logging(level="INFO", log_file=cfg.get("logging.file"))
    write_config(cfg)
    if args.stage in {"panel", "all"}:
        run_panel(cfg, args.resume)
    if args.stage in {"prior", "all"}:
        run_prior(cfg)
    if args.stage == "effects":
        run_effects(cfg)
    if args.stage == "manifest":
        refresh_manifests(cfg)
    logger.info("HOLDOUT_MASKED_V1 causal-data pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
