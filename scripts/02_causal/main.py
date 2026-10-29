# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""One-click execution of WPAC Causal Inference Phase 1 pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from wpac_causal.best_track import BestTrackProcessor, download_ibtracs
from wpac_causal.causal_discovery import CausalDiscovery
from wpac_causal.download import ERA5Downloader
from wpac_causal.download_edh import EDHDownloader
from wpac_causal.effect_estimation import EffectEstimator
from wpac_causal.knowledge_base import KnowledgeBaseBuilder
from wpac_causal.preprocessing import ERA5Extractor, Preprocessor
from wpac_causal.utils import Config, phase_timer, setup_logging


def run_best_track(cfg: Config) -> pd.DataFrame:
    """Run Phase 1: Best track processing."""
    ibtracs_path = cfg.get("data.ibtracs_path")
    ibtracs_url = cfg.get("data.ibtracs_url")
    p = Path(ibtracs_path)
    if not p.exists() or p.stat().st_size == 0:
        logger.info("IBTrACS not found locally; attempting download...")
        download_ibtracs(ibtracs_url, ibtracs_path)
    proc = BestTrackProcessor(cfg)
    df = proc.process(ibtracs_path)
    return df


def run_download(cfg: Config, df_bt: pd.DataFrame, use_edh: bool = False, edh_token: str = "") -> None:
    """Run Phase 2: ERA5 per-storm download (CDS or EDH)."""
    if use_edh:
        if not edh_token:
            raise ValueError("EDH token required for Earth Data Hub download. Set EDH_TOKEN env var or pass --edh-token.")
        downloader = EDHDownloader(
            token=edh_token,
            output_dir=cfg.get("data.era5_download_dir", "data/phase1/raw/era5"),
            buffer_deg=cfg.get("era5.spatial_buffer_deg", 15),
            buffer_days=cfg.get("era5.temporal_buffer_days", 1),
            pressure_levels=cfg.get("era5.pressure_levels", [1000, 925, 850, 700, 500, 300, 200]),
            max_workers=cfg.get("execution.n_jobs", 4) if cfg.get("execution.n_jobs", -1) > 0 else 4,
            batch_size=8,
            retry_max=3,
            skip_existing=True,
        )
        downloader.download_all(df_bt)
    else:
        downloader = ERA5Downloader(cfg)
        downloader.download_all(df_bt)


def _has_complete_era5(storm_dir: Path, sid: str) -> bool:
    """Check whether storm_dir contains the required PL + SL files."""
    if not storm_dir.is_dir():
        return False
    has_pl = (storm_dir / f"{sid}_pl.nc").exists()
    # Accept either a single SL file or the split instant/accum pair
    has_sl_single = (storm_dir / f"{sid}_sl.nc").exists()
    has_sl_split = (
        (storm_dir / f"{sid}_sl_instant.nc").exists()
        and (storm_dir / f"{sid}_sl_accum.nc").exists()
    )
    return has_pl and (has_sl_single or has_sl_split)


def _extract_one_storm(args: Tuple) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """Worker for parallel per-storm extraction (must be top-level for spawn)."""
    sid, storm_df, pl_file, sl_file, cfg_dict = args
    from wpac_causal.preprocessing import ERA5Extractor
    from wpac_causal.utils import Config
    extractor = ERA5Extractor(Config(cfg_dict))
    try:
        ds = extractor.load_storm_netcdf(pl_file, sl_file)
        if ds is None:
            return sid, None, "netcdf_load_returned_none"
        extracted = extractor.extract_storm(sid, storm_df, ds)
        if extracted is None or extracted.empty:
            return sid, None, "empty_extraction_result"
        return sid, extracted, None
    except Exception as exc:
        logger.warning(f"Extraction failed for {sid}: {exc}")
        return sid, None, f"{type(exc).__name__}: {exc}"


def _write_extraction_report(cfg: Config, report: dict) -> None:
    path = cfg.get("preprocessing.extraction_audit_json")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def run_preprocessing(cfg: Config, df_bt: pd.DataFrame) -> pd.DataFrame:
    """Run ERA5 extraction, derived variables and fit-only transforms.

    Legacy runs may process only locally available storms.  The holdout sets
    ``require_complete_extraction`` and instead fails on any missing/failed
    storm, after recording a deterministic per-storm extraction audit.
    """
    sid_col = cfg.get("best_track.id_col", "SID")
    era5_dir = Path(cfg.get("data.era5_download_dir"))
    strict = bool(cfg.get("preprocessing.require_complete_extraction", False))
    min_obs = int(cfg.get("preprocessing.min_obs_per_storm", 0))
    expected_sids = sorted(df_bt[sid_col].astype(str).unique().tolist())
    report = {
        "protocol": "HOLDOUT_MASKED_V1" if strict else "LEGACY",
        "status": "running",
        "expected_sids": expected_sids,
        "n_expected": len(expected_sids),
        "min_obs_per_storm": min_obs,
        "min_obs_exclusions": [],
        "postprocessing_missing_sids": [],
        "postprocessing_extra_sids": [],
    }

    available_sids = [
        sid for sid in df_bt[sid_col].unique()
        if _has_complete_era5(era5_dir / sid, sid)
    ]
    available_set = set(map(str, available_sids))
    missing_sids = sorted(set(expected_sids) - available_set)
    logger.info(f"Preprocessing {len(available_sids)} storms with local ERA5 files")
    tasks = [
        (sid, df_bt[df_bt[sid_col] == sid],
         era5_dir / sid / f"{sid}_pl.nc", era5_dir / sid / f"{sid}_sl.nc",
         cfg.to_dict())
        for sid in available_sids
    ]

    n_jobs = int(cfg.get("execution.n_jobs", 1))
    if n_jobs == 1:
        results = [_extract_one_storm(t) for t in tasks]
    else:
        max_workers = os.cpu_count() if n_jobs < 0 else n_jobs
        max_workers = min(max_workers, int(cfg.get("execution.extraction_max_workers", 8)))
        logger.info(f"Parallel extraction with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_extract_one_storm, tasks))

    failures = [{"SID": sid, "reason": "missing_era5_pl_or_sl"}
                for sid in missing_sids]
    failures.extend({"SID": str(sid), "reason": reason or "unknown_extraction_failure"}
                    for sid, env, reason in results if env is None)
    failures = sorted(failures, key=lambda row: row["SID"])
    report.update({
        "available_sids": sorted(available_set),
        "successful_sids": sorted(str(sid) for sid, env, _ in results if env is not None),
        "failures": failures,
        "n_failures": len(failures),
    })
    report["status"] = "failed" if strict and failures else "extracted"
    _write_extraction_report(cfg, report)
    if strict and failures:
        raise RuntimeError(
            f"ERA5 extraction failed for {len(failures)} label-eligible storms: "
            f"{[row['SID'] for row in failures]}")

    env_frames = [env for _, env, _ in results if env is not None]
    if env_frames:
        env_df = pd.concat(env_frames, ignore_index=True)
        time_col = cfg.get("best_track.time_col", "ISO_TIME")
        df_bt[time_col] = pd.to_datetime(df_bt[time_col], utc=True)
        env_df["ISO_TIME"] = pd.to_datetime(env_df["ISO_TIME"], utc=True)
        for col in ("LAT", "LON"):
            if col in env_df.columns:
                env_df = env_df.drop(columns=col)
        df_merged = pd.merge(df_bt, env_df, on=[sid_col, "ISO_TIME"], how="left")
    else:
        if strict:
            raise RuntimeError("strict extraction produced no environmental rows")
        logger.warning("No ERA5 extractions succeeded; proceeding with best-track only")
        df_merged = df_bt.copy()

    if min_obs > 0:
        counts = df_merged.groupby(sid_col).size()
        keep_sids = counts[counts >= min_obs].index
        dropped = counts[counts < min_obs]
        report["min_obs_exclusions"] = [
            {"SID": str(sid), "rows": int(count),
             "reason": f"rows<{min_obs}"}
            for sid, count in dropped.sort_index().items()
        ]
        if len(dropped):
            logger.info(f"Dropping {len(dropped)} storms with < {min_obs} obs: {list(dropped.index)}")
        df_merged = df_merged[df_merged[sid_col].isin(keep_sids)].copy()
    if strict and report["min_obs_exclusions"]:
        report["status"] = "failed"
        _write_extraction_report(cfg, report)
        raise RuntimeError("min_obs_per_storm removed label-eligible systems under holdout")

    lat_col = cfg.get("best_track.lat_col", "USA_LAT")
    lon_col = cfg.get("best_track.lon_col", "USA_LON")
    df_merged["LAT"] = df_merged[lat_col]
    df_merged["LON"] = df_merged[lon_col]
    preprocessor = Preprocessor(cfg)
    df_merged = preprocessor.process(df_merged)

    output_sids = set(df_merged[sid_col].astype(str).unique())
    report["postprocessing_missing_sids"] = sorted(set(expected_sids) - output_sids)
    report["postprocessing_extra_sids"] = sorted(output_sids - set(expected_sids))
    report["n_output_storms"] = len(output_sids)
    report["status"] = "complete"
    if strict and (report["postprocessing_missing_sids"] or report["postprocessing_extra_sids"]):
        report["status"] = "failed"
    _write_extraction_report(cfg, report)
    if report["status"] == "failed":
        raise RuntimeError(
            "preprocessing changed the label-eligible storm population: "
            f"missing={report['postprocessing_missing_sids']} "
            f"extra={report['postprocessing_extra_sids']}")
    return df_merged


def run_causal_discovery(cfg: Config, df: pd.DataFrame) -> None:
    """Run Phase 4: PCMCI+ causal discovery."""
    cd = CausalDiscovery(cfg)
    cd.run(df)


def run_effect_estimation(cfg: Config, df: pd.DataFrame) -> None:
    """Run Phase 5: DoWhy ATE/CATE + refutation + regime analysis."""
    pcmci_path = Path(cfg.get("causal_discovery.results_pkl"))
    if not pcmci_path.exists():
        logger.warning("PCMCI+ results not found; skipping effect estimation")
        return
    import pickle
    with open(pcmci_path, "rb") as f:
        pcmci_results = pickle.load(f)
    estimator = EffectEstimator(cfg)
    ate_df, qual_df, med_df, cate_df, regime_df, shrd_df, _ = estimator.run(df, pcmci_results)
    logger.info(
        f"Effect estimation complete: ATE={len(ate_df)}, Qualitative={len(qual_df)}, "
        f"Mediation={len(med_df)}, CATE={len(cate_df)}, Regime={len(regime_df)}, SHRD_bins={len(shrd_df)}"
    )


def run_knowledge_base(cfg: Config, df: pd.DataFrame) -> None:
    """Run Phase 6: YAML knowledge base export."""
    pcmci_path = Path(cfg.get("causal_discovery.results_pkl"))
    ate_path = Path(cfg.get("effect_estimation.ate_csv"))
    cate_path = Path(cfg.get("effect_estimation.cate_csv"))
    med_path = ate_path.parent / "mediation_results.csv"
    regime_path = ate_path.parent / "regime_cate_results.csv"
    if not all(p.exists() for p in [pcmci_path, ate_path, cate_path]):
        logger.warning("Missing upstream outputs; skipping knowledge base")
        return
    builder = KnowledgeBaseBuilder(cfg)
    builder.build(df, pcmci_path, ate_path, cate_path, med_path, regime_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="WPAC Causal Inference Phase 1")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Skip ERA5 download phase")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip ERA5 extraction")
    parser.add_argument("--phases", nargs="+", default=None,
                        help="Run specific phases: best_track, download, preprocessing, causal_discovery, effect_estimation, knowledge_base")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    setup_logging(
        level=cfg.get("logging.level", "INFO"),
        log_file=cfg.get("logging.file"),
        rotation=cfg.get("logging.rotation", "10 MB"),
        retention=cfg.get("logging.retention", "30 days"),
    )

    phases = args.phases or cfg.get("execution.phases", [])
    logger.info(f"Running phases: {phases}")

    # Phase 1: Best Track
    if "best_track" in phases:
        df_bt = run_best_track(cfg)
        out_dir = Path(cfg.get("data.output_dir"))
        out_dir.mkdir(parents=True, exist_ok=True)
        df_bt.to_parquet(out_dir / "wpac_tc_dataset.parquet")
        logger.info(f"Best-track dataset saved to {out_dir / 'wpac_tc_dataset.parquet'}")
    else:
        df_bt = pd.read_parquet(Path(cfg.get("data.output_dir")) / "wpac_tc_dataset.parquet")

    # Phase 2: Download
    if "download" in phases and not args.skip_download:
        run_download(cfg, df_bt)

    # Phase 3: Preprocessing
    if "preprocessing" in phases and not args.skip_extraction:
        df_proc = run_preprocessing(cfg, df_bt)
        df_proc.to_parquet(Path(cfg.get("data.output_dir")) / "wpac_tc_dataset_processed.parquet")
        logger.info("Preprocessed dataset saved")
    else:
        proc_path = Path(cfg.get("data.output_dir")) / "wpac_tc_dataset_processed.parquet"
        if proc_path.exists():
            df_proc = pd.read_parquet(proc_path)
        else:
            df_proc = df_bt.copy()

    # Phase 4: Causal Discovery
    if "causal_discovery" in phases:
        run_causal_discovery(cfg, df_proc)

    # Phase 5: Effect Estimation
    if "effect_estimation" in phases:
        run_effect_estimation(cfg, df_proc)

    # Phase 6: Knowledge Base
    if "knowledge_base" in phases:
        run_knowledge_base(cfg, df_proc)

    logger.info("=== All requested phases complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
