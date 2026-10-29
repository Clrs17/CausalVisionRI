# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Shared HOLDOUT_MASKED_V1 baseline machinery (LR / LDA / SHIPS-12).

Every baseline is trained on the FIT split only, thresholded on INNER, and
written to a versioned run directory that ``scripts/finalize_holdout.py`` and
``scripts/test_infer.py`` can consume:

    <out_root>/<run_id>/
        config.yaml            # the holdout config the baseline ran under
        run_manifest.json      # kind, protocol, hashes, feature spec
        model.pkl              # sklearn Pipeline (impute -> scale -> clf)
        features.parquet       # row_uid + feature columns for ALL rows
        inner.npz              # p_inner / y_inner / row keys / threshold
        results.csv            # inner metrics (selection only)

No test label or test prediction is produced here.  ``predict_baseline`` is
used by the single test-inference step after the final lock is frozen.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import yaml

P2_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P2_ROOT / "src"))

from wpac_warning.dataset import RIPanelDataset          # noqa: E402
from wpac_warning.evaluate import f1_threshold, metrics_row  # noqa: E402
from wpac_warning.frame_store import (frame_array_path,  # noqa: E402
                             require_frame_time_path)
from wpac_warning.protocol import (FRAME_AVAILABILITY_REL,  # noqa: E402
                          FRAME_AVAILABILITY_RULE_VERSION, QC_MANIFEST_REL,
                          QC_RULE_VERSION, SplitProtocol, is_qc_hursat_frames,
                          is_raw_hursat_frames, portable_path, sha256_array,
                          sha256_file, sha256_json, validate_frame_availability,
                          validate_qc_manifest)
from wpac_warning.utils_cfg import Config, load_cfg              # noqa: E402

BASELINE_KINDS = ("lr", "lda", "ships12")
INNER_YEARS = (2018, 2019)

# the only protocol a production baseline may run under, and its config
HOLDOUT_PROTOCOL = "holdout_masked_v1"
DEFAULT_HOLDOUT_CONFIG = P2_ROOT / "configs" / "phase2" / "holdout_masked_v1.yaml"


def load_baseline_cfg(config_path: str | Path | None = None,
                      *, legacy: bool = False):
    """Load a baseline config, refusing any silent legacy fallback.

    Production baselines (`scripts/baseline_lr.py`, `scripts/baseline_ships12.py`)
    run ONLY under ``config/holdout_masked_v1.yaml`` (the default).  Any other
    config -- in particular the legacy ``config/config.yaml`` -- exits non-zero
    unless the caller asks for the archived pre-holdout evaluation with an
    explicit ``legacy=True`` (CLI: ``--legacy``).  A holdout config combined with
    the legacy flag is contradictory and also refused.
    """
    path = Path(config_path) if config_path else DEFAULT_HOLDOUT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"baseline config not found: {path}")
    cfg = load_cfg(path)
    protocol = str(cfg.get("data.protocol") or "")
    if protocol == HOLDOUT_PROTOCOL:
        if legacy:
            raise ValueError(
                f"{path} is the {HOLDOUT_PROTOCOL} config; --legacy evaluates "
                f"the archived pre-holdout LOYO protocol and must be given a "
                f"legacy config instead")
        return cfg, True
    if not legacy:
        raise ValueError(
            f"{path}: data.protocol={protocol!r} is not {HOLDOUT_PROTOCOL!r}. "
            f"Production baselines run only under "
            f"{DEFAULT_HOLDOUT_CONFIG}; pass --legacy to re-run the archived "
            f"pre-holdout evaluation explicitly (it cannot touch holdout)")
    return cfg, False


def _as_cfg(cfg):
    return cfg if isinstance(cfg, Config) else Config(dict(cfg))


def _rel(p: str | Path) -> str:
    return portable_path(p, P2_ROOT)


def _feature_frame(ds: RIPanelDataset, columns) -> pd.DataFrame:
    """Raw feature values for every dataset row (no scaling here)."""
    missing = [c for c in columns if c not in ds.df.columns]
    if missing:
        raise ValueError(f"panel missing baseline features: {missing}")
    out = ds.df[["row_uid"] + list(columns)].copy()
    if out["row_uid"].duplicated().any():
        raise ValueError("duplicate row_uid in the baseline feature matrix")
    return out


def make_pipeline(kind: str):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "lda":
        clf = LinearDiscriminantAnalysis()
    elif kind == "lr":
        clf = LogisticRegression(class_weight="balanced", max_iter=3000)
    else:
        raise ValueError(f"make_pipeline: unsupported kind {kind!r}")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def run_baseline(cfg, kind: str, features: pd.DataFrame, out_root: str | Path,
                 run_id: str, feature_spec: Dict[str, Any],
                 extra_manifest: Optional[Dict[str, Any]] = None) -> Path:
    """Fit one baseline on FIT, threshold on INNER, write the run directory."""
    if kind not in BASELINE_KINDS:
        raise ValueError(f"unknown baseline kind {kind!r}")
    if kind == "ships12":
        raise ValueError("ships12 must pass its fitted pipeline via "
                         "run_baseline_with_pipeline")
    return _run_baseline_impl(cfg, kind, features, out_root, run_id,
                              feature_spec, None, extra_manifest)


def run_baseline_with_pipeline(cfg, kind: str, features: pd.DataFrame,
                               pipeline, out_root: str | Path, run_id: str,
                               feature_spec: Dict[str, Any],
                               extra_manifest: Optional[Dict[str, Any]] = None
                               ) -> Path:
    return _run_baseline_impl(cfg, kind, features, out_root, run_id,
                              feature_spec, pipeline, extra_manifest)


def _run_baseline_impl(cfg, kind, features, out_root, run_id, feature_spec,
                       pipeline, extra_manifest) -> Path:
    cfg = _as_cfg(cfg)
    proto = SplitProtocol.from_cfg(cfg)
    proto.validate()
    # one row universe for DL and every baseline: eligible panel minus
    # frame-availability exclusions (never "all eligible rows")
    eligible = eligible_panel_rows(cfg)
    index_keys = None
    if cfg.get("data.frames_index"):
        idx_path = holdout_path(cfg.get("data.frames_index"))
        if idx_path.is_file():
            index_keys = frame_index_row_keys(idx_path)
    avail = frame_availability_binding(cfg, eligible=eligible,
                                       index_keys=index_keys)
    avail_rec = availability_record(avail)
    ds = RIPanelDataset(cfg)
    if not ds.holdout:
        raise ValueError("baseline holdout path requires data.protocol="
                         "holdout_masked_v1")
    check_dataset_availability(ds, avail, eligible=eligible,
                               where=f"baseline {kind}")
    fit_idx = ds.split_indices["fit"]
    inner_idx = ds.split_indices["inner"]
    if len(fit_idx) == 0 or len(inner_idx) == 0:
        raise ValueError("empty baseline fit/inner split")

    if "row_uid" not in features.columns:
        raise ValueError("baseline features must carry a row_uid column")
    _check_feature_columns([c for c in features.columns if c != "row_uid"],
                           kind)
    rows = ds.df[["row_uid", "RI", "SID", "storm_year", "iso_time_str"]]
    merged = rows.merge(features, on="row_uid", how="left", validate="1:1")
    if merged["row_uid"].duplicated().any():
        raise ValueError("feature matrix is not unique per row_uid")
    feat_cols = [c for c in merged.columns
                 if c not in ("row_uid", "RI", "SID", "storm_year",
                              "iso_time_str")]
    if not feat_cols:
        raise ValueError("no baseline feature columns")
    _check_feature_columns(feat_cols, kind)
    X = merged[feat_cols].to_numpy(dtype=float)
    y = merged["RI"].to_numpy(dtype=int)
    if np.isnan(X[fit_idx]).all(axis=0).any():
        bad = [c for j, c in enumerate(feat_cols)
               if np.isnan(X[fit_idx][:, j]).all()]
        raise ValueError(f"fit rows are all-NaN for features: {bad}")

    pipe = pipeline if pipeline is not None else make_pipeline(kind)
    pipe.fit(X[fit_idx], y[fit_idx])
    p_inner = pipe.predict_proba(X[inner_idx])[:, 1]
    thr = f1_threshold(p_inner, y[inner_idx])

    run_dir = Path(out_root) / run_id
    if run_dir.exists():
        raise FileExistsError(f"baseline run directory exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(pipe, f)
    features.to_parquet(run_dir / "features.parquet", index=False)

    inner_rows = merged.iloc[inner_idx].reset_index(drop=True)
    np.savez(run_dir / "inner.npz",
             p_inner=p_inner, y_inner=y[inner_idx], threshold=float(thr),
             inner_bss=float(metrics_row(p_inner, y[inner_idx], thr,
                                         float(y[fit_idx].mean()))["BSS_dev"]),
             sid=inner_rows["SID"].astype(str).values,
             iso_time=inner_rows["iso_time_str"].values,
             storm_year=inner_rows["storm_year"].astype(np.int32).values,
             row_uid=inner_rows["row_uid"].values,
             split=np.full(len(inner_rows), "inner", dtype=object))
    pd.DataFrame([{
        "run_id": run_id, "kind": kind, "n_fit": len(fit_idx),
        "n_inner": len(inner_idx), "inner_threshold": thr,
        "n_features": len(feat_cols),
    }]).to_csv(run_dir / "results.csv", index=False)

    manifest = {
        "kind": kind,
        "protocol": proto.name,
        "inner_only": True,
        "run_id": run_id,
        "config_sha256": sha256_json(cfg.to_dict()),
        "feature_spec": feature_spec,
        "n_features": len(feat_cols),
        "features": feat_cols,
        "n_fit": int(len(fit_idx)),
        "n_inner": int(len(inner_idx)),
        "inner_threshold": float(thr),
        "model_sha256": sha256_file(run_dir / "model.pkl"),
        "features_parquet_sha256": sha256_file(run_dir / "features.parquet"),
        "inner_npz_sha256": sha256_file(run_dir / "inner.npz"),
        "prior_manifest_sha256": None,
        "protocol_years": proto.as_dict(),
        "availability": avail_rec,
        **(extra_manifest or {}),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return run_dir


def run_lr_lda(cfg, kind: str, out_root: str | Path, run_id: str) -> Path:
    cfg = _as_cfg(cfg)
    eligible = eligible_panel_rows(cfg)
    index_keys = None
    if cfg.get("data.frames_index"):
        idx_path = holdout_path(cfg.get("data.frames_index"))
        if idx_path.is_file():
            index_keys = frame_index_row_keys(idx_path)
    avail = frame_availability_binding(cfg, eligible=eligible,
                                       index_keys=index_keys)
    ds = RIPanelDataset(cfg)
    check_dataset_availability(ds, avail, eligible=eligible,
                               where=f"baseline {kind} features")
    env_vars = list(cfg.get("data.env_vars"))
    features = _feature_frame(ds, env_vars)
    if list(features.columns) != ["row_uid"] + env_vars:
        raise ValueError("LR/LDA features must be exactly data.env_vars")
    return run_baseline(
        cfg, kind, features, out_root, run_id,
        feature_spec={"source": "panel env_vars (raw)", "columns": env_vars})


def predict_baseline(run_dir: str | Path, row_uids) -> np.ndarray:
    """Predict probabilities for the given row keys (test inference)."""
    run_dir = Path(run_dir)
    with open(run_dir / "model.pkl", "rb") as f:
        pipe = pickle.load(f)
    feats = pd.read_parquet(run_dir / "features.parquet")
    if feats["row_uid"].duplicated().any():
        raise ValueError(f"{run_dir}: duplicate row_uid in features.parquet")
    idx = pd.Index(feats["row_uid"].astype(str)).get_indexer(
        pd.Index(np.asarray(row_uids).astype(str)))
    if (idx < 0).any():
        raise ValueError(f"{run_dir}: missing feature rows for test keys")
    spec = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    cols = spec["features"]
    X = feats.iloc[idx][cols].to_numpy(dtype=float)
    return pipe.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------
# SHIPS-12 predictor table with FIT-ONLY COHC climatology
# ----------------------------------------------------------------------

# The 12 predictors of Horinouchi et al. (2025, WRPS1) Table 1, in the order
# used by the historical baseline.  OHCD is the MAIN ocean predictor; COHC is a
# separate pre-specified sensitivity and never enters together with OHCD.
SHIPS_PREDICTORS = ("VMAX", "INCV", "POT", "OHCD", "SHRD", "T150", "T200",
                    "T250", "R000", "SDIR", "Z850", "DTL")
SHIPS_PREDICTORS_COHC = ("VMAX", "INCV", "POT", "COHC", "SHRD", "T150", "T200",
                         "T250", "R000", "SDIR", "Z850", "DTL")
KM_PER_DEG = 111.195

# labels / future information that must never become a baseline feature
FORBIDDEN_FEATURES = {"RI", "DV24", "DV24_YJ", "VMAX_F24", "F24_TIME",
                      "USA_PRES", "LC_STAGE"}


def _check_feature_columns(cols, kind: str) -> None:
    bad = sorted(set(cols) & FORBIDDEN_FEATURES)
    if bad:
        raise ValueError(f"{kind}: labels/future columns leaked into the "
                         f"baseline features: {bad}")
    if kind == "ships12" and frozenset(cols) not in (
            frozenset(SHIPS_PREDICTORS), frozenset(SHIPS_PREDICTORS_COHC)):
        raise ValueError(
            "ships12 must use exactly the 12 WRPS1 predictors (OHCD main or "
            f"COHC sensitivity), got {sorted(cols)}")


def _dist_km(lon2d, lat2d, lon0: float, lat0: float):
    """Planar distance (km) with cos(lat0) correction (phase-1 convention)."""
    cos0 = np.cos(np.radians(lat0))
    return np.sqrt(((lon2d - lon0) * cos0) ** 2
                   + (lat2d - lat0) ** 2) * KM_PER_DEG


# CMEMS GLORYS12V1 daily fields are daily MEANS over a calendar day
# (00 UTC to next 00 UTC), so a field labelled D is complete only at D+1 00 UTC.
OHC_DAILY_WINDOW = pd.Timedelta(days=1)

# provenance marker: OHCD selected by window end (source_time + 24 h <= target)
OHCD_ALGORITHM = "backward-daily-window-end-24h"
# human-readable form of the same rule, recorded in every source/feature manifest
OHCD_SELECTION_RULE = "source_time + 24 h <= ISO_TIME"
# only this origin is acceptable for the holdout OHCD main arm
OHCD_ORIGIN_RECOMPUTED = "recomputed_backward"
# the origin the pre-fix archived predictor table carries (future-peeking OHC)
OHCD_ORIGIN_SOURCE_TABLE = "source_table_column"


def ships12_feature_manifest_for(features_parquet: str | Path) -> Path:
    """Provenance manifest next to a fit-only SHIPS feature table."""
    return Path(features_parquet).with_suffix(".manifest.json")


# ----------------------------------------------------------------------
# ALL-SYSTEM SHIPS source table: mandatory build manifest
# ----------------------------------------------------------------------
# The row-level source table is rebuilt from the raw ERA5/GLORYS archives over
# the 725-system panel.  The fit-only feature cache may only consume it through
# the build manifest next to it -- the file name/hash alone is not provenance.
SHIPS_SOURCE_KIND = "ships12_source_predictors"
SHIPS_RAW_GROUPS = ("era5_pressure_levels", "era5_surface", "glorys_ohc")
SHIPS_RAW_SHA_CACHE = P2_ROOT / "data" / "ships12_raw_input_sha256.json"


def ships12_source_manifest_for(source_parquet: str | Path) -> Path:
    """Fixed build-manifest path (``<table>.manifest.json``).

    Canonical name is ``data/<table stem>.manifest.json``.  A manifest at the
    also-seen ``<table>.parquet.manifest.json`` spelling is accepted on lookup
    so a packaging whitelist using that spelling still resolves; the builder
    always writes the canonical name.
    """
    source_parquet = Path(source_parquet)
    canonical = source_parquet.with_suffix(".manifest.json")
    if canonical.is_file():
        return canonical
    appended = Path(f"{source_parquet}.manifest.json")
    return appended if appended.is_file() else canonical


def ships12_raw_input_paths(sid: str, era5_dir: str | Path,
                            ohc_dir: str | Path,
                            frames_dir: str | Path) -> Dict[str, Path]:
    """Every raw file one storm's predictors are built from (per SID)."""
    sid = str(sid)
    era5_dir, ohc_dir, frames_dir = Path(era5_dir), Path(ohc_dir), Path(frames_dir)
    return {
        "era5_pressure_levels": era5_dir / sid / f"{sid}_pl.nc",
        "era5_surface": era5_dir / sid / f"{sid}_sl_instant.nc",
        "glorys_ohc": ohc_dir / sid / f"{sid}_ohc.nc",
        "frames": frame_array_path(frames_dir, sid),
        "frames_time_axis": require_frame_time_path(frames_dir, sid),
    }


# ----------------------------------------------------------------------
# QC v1 frame corpus binding (rule hursat_qc_v1)
# ----------------------------------------------------------------------
# The production frame corpus is the QC v1 slice (data/global_hursat/frames_qc),
# audited by data/global_hursat/qc_manifest.json.  The raw download corpus
# (data/global_hursat/frames) is a QC/re-key input only and may never be the
# source of a production SHIPS table.
QC_FRAMES_REL = "data/global_hursat/frames_qc"
RAW_FRAMES_REL = "data/global_hursat/frames"


def holdout_path(value: str | Path) -> Path:
    """Config-relative holdout path (absolute paths are kept as given)."""
    p = Path(value)
    return p if p.is_absolute() else (P2_ROOT / p)


def holdout_frames_binding(cfg) -> Dict[str, Path]:
    """``data.frames_dir`` / ``data.qc_manifest`` declared by the holdout config.

    The source table is built over the QC v1 corpus; neither path may be
    inferred from a module default, and the raw download corpus is refused.
    """
    cfg = _as_cfg(cfg)
    frames = cfg.get("data.frames_dir")
    qc = cfg.get("data.qc_manifest")
    if not frames:
        raise ValueError(
            "config data.frames_dir is required: the SHIPS source table is "
            f"built from the QC v1 corpus ({QC_FRAMES_REL})")
    if not qc:
        raise ValueError(
            "config data.qc_manifest is required: the SHIPS source table must "
            f"be pinned to the QC v1 audit ({QC_MANIFEST_REL})")
    frames_dir = holdout_path(frames)
    if is_raw_hursat_frames(frames_dir):
        raise ValueError(
            f"config data.frames_dir resolves to the RAW HURSAT corpus "
            f"({frames_dir}); the production source table uses the QC v1 "
            f"corpus ({QC_FRAMES_REL}).  Use --frames-dir + --allow-raw-frames "
            f"only for an explicit audit build (never packed)")
    if not is_qc_hursat_frames(frames_dir):
        raise ValueError(
            f"config data.frames_dir={frames!r} is not the QC v1 corpus "
            f"({QC_FRAMES_REL})")
    return {"frames_dir": frames_dir, "qc_manifest": holdout_path(qc)}


def load_qc_binding(qc_manifest: str | Path, frames_dir: str | Path | None = None
                    ) -> Dict[str, Any]:
    """Validated QC v1 manifest + its content SHA-256.

    Reuses ``src.protocol.validate_qc_manifest`` (rule version, completeness,
    TBB window, source/out corpora, per-store accounting, stores hash) so the
    builder and the packer cannot drift apart.  A missing or partial manifest is
    a hard error: without it the frame corpus is not the QC corpus.
    """
    path = Path(qc_manifest)
    if not path.is_file():
        raise FileNotFoundError(
            f"QC v1 manifest missing: {path}; run scripts/build_hursat_qc.py "
            f"before building the all-system SHIPS source table")
    block = json.loads(path.read_text(encoding="utf-8"))
    audit = validate_qc_manifest(
        block, expected_out_frames=(Path(frames_dir) if frames_dir else None))
    return {"path": str(path), "sha256": sha256_file(path), **audit}


def frame_store_tokens(raw_inputs: Dict[str, Any], sids) -> list:
    """Per-SID (array, time axis) content hash tokens, sorted by SID."""
    frames = {str(e.get("sid")): e for e in raw_inputs.get("frames", [])}
    times = {str(e.get("sid")): e for e in raw_inputs.get("frames_time_axis", [])}
    tokens = []
    for sid in sorted(str(s) for s in sids):
        if sid not in frames or sid not in times:
            raise ValueError(f"frame store provenance missing for SID {sid}")
        tokens.append({"sid": sid,
                       "array_sha256": str(frames[sid].get("sha256") or ""),
                       "times_sha256": str(times[sid].get("sha256") or "")})
    return tokens


def frame_stores_sha256(raw_inputs: Dict[str, Any], sids) -> str:
    """Content hash of the frame stores the source table was built from."""
    return sha256_json(frame_store_tokens(raw_inputs, sids))


# ----------------------------------------------------------------------
# Frame availability (rule frame_availability_v1): the ONE row universe
# ----------------------------------------------------------------------
# The DL dataset, the LR/LDA baselines and the SHIPS source table must use
# exactly ``eligible panel - availability exclusions``.  The exclusions are the
# eligible rows whose t0 has no exact QC frame; the asset step publishes them in
# ``data/holdout_masked_v1/frame_availability.json``.  A config that does not
# declare the file (non-production/fixture configs) falls back to the same rule
# derived from the frames index, and every manifest records which source was
# used, so a table can never silently drift from the DL row set.
FRAME_AVAILABILITY_RULE = FRAME_AVAILABILITY_RULE_VERSION


def frame_availability_path(cfg) -> Path | None:
    """``data.frame_availability`` resolved against the repo (None if unset)."""
    rel = _as_cfg(cfg).get("data.frame_availability")
    return holdout_path(rel) if rel else None


def eligible_panel_rows(cfg, panel: pd.DataFrame | None = None):
    """Label-eligible holdout panel rows (``row_uid``/``SID``/``ISO_TIME``/``split``).

    Mirrors the dataset's split assignment: storm_year from
    ``data.storm_year_manifest``, ``assign_rows`` with the configured protocol,
    keep ``eligible``.  Returns ``None`` when the config cannot define it
    (non-holdout protocol, missing label-time column, storm-year manifest or
    panel), in which case callers fall back to the panel row set.
    """
    cfg = _as_cfg(cfg)
    if str(cfg.get("data.protocol") or "") != HOLDOUT_PROTOCOL:
        return None
    proto = SplitProtocol.from_cfg(cfg)
    proto.validate()
    label_col = proto.label_time_col
    if panel is None:
        panel_path = cfg.get("data.panel_parquet")
        if not panel_path:
            return None
        panel_path = holdout_path(panel_path)
        if not panel_path.is_file():
            return None
        panel = pd.read_parquet(panel_path)
    if not {"SID", "ISO_TIME", label_col}.issubset(panel.columns):
        return None
    year_src = cfg.get("data.storm_year_manifest")
    if not year_src:
        return None
    year_src = holdout_path(year_src)
    if not year_src.is_file():
        return None
    from wpac_warning.protocol import (assign_rows, attach_storm_year,
                              load_storm_year_manifest)
    frame = panel.copy()
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], utc=True)
    frame = frame.drop_duplicates(["SID", "ISO_TIME"])
    frame = attach_storm_year(frame, load_storm_year_manifest(year_src))
    out = assign_rows(frame, proto)
    return out.loc[out["eligible"],
                   ["row_uid", "SID", "ISO_TIME", "split"]].reset_index(drop=True)


def frame_index_row_keys(frames_index: str | Path) -> list:
    """``SID|ISO_TIME`` row keys of a frames index (unique, sorted)."""
    df = pd.read_parquet(frames_index, columns=["SID", "ISO_TIME"])
    t = pd.to_datetime(df["ISO_TIME"], utc=True)
    keys = df["SID"].astype(str) + "|" + t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if keys.duplicated().any():
        raise ValueError(f"{frames_index}: duplicate (SID, ISO_TIME) frame keys")
    return sorted(keys.tolist())


def _availability_counts(rows) -> tuple:
    """Flat per-split/year/storm counts of the excluded rows."""
    splits: Dict[str, int] = {}
    years: Dict[str, int] = {}
    storms: Dict[str, int] = {}
    for row in rows:
        uid = str(row["row_uid"])
        sid = str(row.get("SID") or uid.split("|", 1)[0])
        year = str(pd.Timestamp(row["ISO_TIME"]).year)
        split = str(row["split"])
        splits[split] = splits.get(split, 0) + 1
        years[year] = years.get(year, 0) + 1
        storms[sid] = storms.get(sid, 0) + 1
    return splits, years, storms


def load_frame_availability(cfg, *, eligible=None, index_keys=None
                            ) -> Dict[str, Any]:
    """Validated frame-availability manifest (rule ``frame_availability_v1``).

    Delegates to ``src.protocol.validate_frame_availability``, the single
    fail-closed rule shared with the asset builder, the dataset and the packer:
    rule/version text, every excluded row with its reason, all hashes,
    per-reason/split/year/storm accounting, the QC manifest/store binding, and
    (when the eligible and index keys are given) the identity
    ``excluded | indexed == eligible`` with ``excluded & indexed == {}``.
    """
    cfg = _as_cfg(cfg)
    path = frame_availability_path(cfg)
    if path is None:
        raise ValueError(
            "config data.frame_availability is not set; the baseline row set "
            f"must come from {FRAME_AVAILABILITY_REL}")
    if not path.is_file():
        raise FileNotFoundError(
            f"frame-availability manifest missing: {path} (rule "
            f"{FRAME_AVAILABILITY_RULE!r}); the baseline cannot decide its row "
            f"set without it")
    block = json.loads(path.read_text(encoding="utf-8"))
    expected_qc = None
    if cfg.get("data.qc_manifest"):
        qc_path = holdout_frames_binding(cfg)["qc_manifest"]
        if not qc_path.is_file():
            raise FileNotFoundError(f"QC v1 manifest missing: {qc_path}")
        qc_block = json.loads(qc_path.read_text(encoding="utf-8"))
        expected_qc = {"sha256": sha256_file(qc_path),
                       "stores_sha256": qc_block.get("stores_sha256")}
    if eligible is not None:
        elig_set = set(eligible["row_uid"].astype(str))
        if index_keys is not None:
            # the row universe is eligible & indexed; index rows outside the
            # eligible set can never reach the dataset, so they are not part of
            # the availability identity
            index_keys = [k for k in index_keys if str(k) in elig_set]
    audit = validate_frame_availability(
        block, expected_qc=expected_qc,
        eligible_row_uids=(eligible["row_uid"].astype(str).tolist()
                           if eligible is not None else None),
        index_row_uids=index_keys)
    rows = block["rows"]
    splits, years, storms = _availability_counts(rows)
    return {
        "source": "manifest",
        "path": str(path),
        "sha256": sha256_file(path),
        "rule_version": audit["rule_version"],
        "rows": rows,
        "excluded": sorted(str(r["row_uid"]) for r in rows),
        "excluded_row_key_sha256": audit["excluded_row_key_sha256"],
        "available_row_key_sha256": audit["available_row_key_sha256"],
        "n_eligible": int(audit["n_eligible_rows"]),
        "n_available": int(audit["n_available_rows"]),
        "n_excluded": int(audit["n_excluded_rows"]),
        "by_split": splits,
        "by_year": years,
        "by_storm": storms,
        "qc_manifest_sha256": str(audit["qc"]["sha256"]).lower(),
        "stores_sha256": str(audit["qc"]["stores_sha256"]).lower(),
        "protocol": block.get("protocol"),
        "population": block.get("population"),
    }


def frame_availability_binding(cfg, *, panel: pd.DataFrame | None = None,
                               eligible=None, index_keys=None,
                               require: bool = False) -> Dict[str, Any]:
    """Frame-availability exclusions used by this baseline build.

    ``require=True`` (the production source-table build) demands the declared
    manifest.  A config without ``data.frame_availability`` (non-production /
    fixture configs) derives the same rule from the frames index and says so in
    ``source``.
    """
    cfg = _as_cfg(cfg)
    if eligible is None:
        eligible = eligible_panel_rows(cfg, panel)
    if frame_availability_path(cfg) is not None:
        return load_frame_availability(cfg, eligible=eligible,
                                       index_keys=index_keys)
    if require:
        raise ValueError(
            "config data.frame_availability is required for the all-system "
            f"source table ({FRAME_AVAILABILITY_REL}); without it the SHIPS row "
            f"set cannot be proven equal to the DL row set")
    if index_keys is None:
        raise ValueError("cannot derive frame availability without the frames "
                         "index keys")
    index_set = set(str(k) for k in index_keys)
    if eligible is not None:
        index_set &= set(eligible["row_uid"].astype(str))
    if eligible is None:
        excluded_keys: list = []
        splits: Dict[str, int] = {}
        years: Dict[str, int] = {}
        storms: Dict[str, int] = {}
        n_eligible = n_available = None
    else:
        frame = eligible.copy()
        frame["row_uid"] = frame["row_uid"].astype(str)
        n_eligible = int(len(frame))
        frame = frame.loc[~frame["row_uid"].isin(index_set)]
        excluded_keys = sorted(frame["row_uid"].tolist())
        splits, years, storms = _availability_counts(
            frame[["row_uid", "SID", "ISO_TIME", "split"]].to_dict("records"))
        n_available = n_eligible - len(excluded_keys)
    available_keys = sorted(index_set) if eligible is not None else []
    return {
        "source": "derived_frames_index",
        "path": None,
        "sha256": None,
        "rule_version": FRAME_AVAILABILITY_RULE,
        "rows": [],
        "excluded": excluded_keys,
        "excluded_row_key_sha256": sha256_array(excluded_keys),
        "available_row_key_sha256": sha256_array(available_keys),
        "n_eligible": n_eligible,
        "n_available": n_available,
        "n_excluded": len(excluded_keys),
        "by_split": splits,
        "by_year": years,
        "by_storm": storms,
        "qc_manifest_sha256": None,
        "stores_sha256": None,
        "protocol": None,
        "population": None,
    }


def availability_record(binding: Dict[str, Any]) -> Dict[str, Any]:
    """Manifest block for a frame-availability binding (flat exclusion counts)."""
    return {
        "source": binding.get("source"),
        "path": _rel(binding["path"]) if binding.get("path") else None,
        "sha256": binding.get("sha256"),
        "rule_version": binding.get("rule_version"),
        "n_eligible_rows": binding.get("n_eligible"),
        "n_available_rows": binding.get("n_available"),
        "n_excluded_rows": int(binding.get("n_excluded", 0)),
        "excluded_row_key_sha256": binding.get("excluded_row_key_sha256"),
        "available_row_key_sha256": binding.get("available_row_key_sha256"),
        "by_split": dict(binding.get("by_split") or {}),
        "by_year": dict(binding.get("by_year") or {}),
        "by_storm": dict(binding.get("by_storm") or {}),
        "qc_manifest_sha256": binding.get("qc_manifest_sha256"),
        "stores_sha256": binding.get("stores_sha256"),
    }


def expected_availability_rows(eligible, binding: Dict[str, Any]) -> list | None:
    """``eligible panel - availability exclusions`` row_uid list (None if unknown)."""
    if eligible is None:
        return None
    keys = set(eligible["row_uid"].astype(str)) - set(binding["excluded"])
    return sorted(keys)


def check_dataset_availability(ds, binding: Dict[str, Any],
                               eligible=None, where: str = "baseline") -> None:
    """The dataset row set must equal eligible panel - availability exclusions."""
    keys = set(ds.df["row_uid"].astype(str))
    overlap = sorted(keys & set(binding["excluded"]))
    if overlap:
        raise ValueError(
            f"{where}: {len(overlap)} availability-excluded row(s) are still in "
            f"the dataset (e.g. {overlap[:5]}); the dataset and the "
            f"availability manifest disagree")
    expected = expected_availability_rows(eligible, binding)
    if expected is None:
        return
    want = set(expected)
    if keys != want:
        missing = sorted(want - keys)
        extra = sorted(keys - want)
        raise ValueError(
            f"{where}: dataset row set differs from the eligible panel minus "
            f"availability exclusions ({len(missing)} missing, {len(extra)} "
            f"extra; e.g. missing {missing[:5]}, extra {extra[:5]})")


def sha256_files_cached(paths, cache_path: str | Path | None = None,
                        *, verify: bool = False) -> Dict[str, Dict[str, Any]]:
    """Content SHA-256 of every file, with an optional size+mtime cache.

    ``verify=True`` ignores the cache and re-hashes everything, so a precomputed
    cache can never hide a changed raw input.  The cache only saves time; the
    manifest always carries the resulting ``sha256`` values.
    """
    cache_path = Path(cache_path) if cache_path else None
    cache: Dict[str, Any] = {}
    if cache_path is not None and cache_path.is_file() and not verify:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:          # a corrupt cache is rebuilt, never trusted
            cache = {}
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        path = Path(path)
        st = path.stat()
        key = _rel(path)
        old = cache.get(key) or {}
        if (not verify and old.get("size") == st.st_size
                and old.get("mtime_ns") == st.st_mtime_ns and old.get("sha256")):
            sha = old["sha256"]
        else:
            sha = sha256_file(path)
        out[key] = {"path": key, "size": int(st.st_size), "sha256": sha}
        cache[key] = {**out[key], "mtime_ns": int(st.st_mtime_ns)}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        live = {k: v for k, v in cache.items()
                if (P2_ROOT / k).is_file() or Path(k).is_file()}
        cache_path.write_text(json.dumps(live, indent=2, sort_keys=True),
                              encoding="utf-8")
    return out


def source_table_row_keys(source_parquet: str | Path) -> list:
    """``SID|ISO_TIME`` row keys of a SHIPS source table (unique, sorted)."""
    df = pd.read_parquet(source_parquet, columns=["SID", "ISO_TIME"])
    t = pd.to_datetime(df["ISO_TIME"], utc=True)
    keys = (df["SID"].astype(str) + "|"
            + t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")).tolist()
    if len(set(keys)) != len(keys):
        raise ValueError(f"{source_parquet}: duplicate (SID, ISO_TIME) rows")
    return sorted(keys)


def _is_sha256(value: Any) -> bool:
    s = str(value or "")
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s.lower())


def _require_entry_sha(entry: Dict[str, Any], where: str,
                       what: str) -> None:
    sha = str(entry.get("sha256") or "")
    if (len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha.lower())):
        raise ValueError(
            f"{where}: {what} entry {entry.get('path')!r} has no content "
            f"sha256 (got {sha!r}); raw input content hashes are required")


def _require_availability(man: Dict[str, Any], table_keys, where: str
                          ) -> Dict[str, Any]:
    """Fail unless the table's row set is the pinned availability exclusion set.

    The source table may only contain ``eligible panel - exclusions``; the
    availability block must therefore pin the exclusion hash/counts, and (when
    the exclusions come from the published manifest) the manifest file must
    still hash-match and must not exclude any row the table contains.
    """
    av = man.get("availability")
    if not isinstance(av, dict) or not av:
        raise ValueError(
            f"{where}: build manifest has no availability block; the source "
            f"table row set must be 'eligible panel - frame-availability "
            f"exclusions' (rule {FRAME_AVAILABILITY_RULE!r})")
    source = av.get("source")
    if source not in ("manifest", "derived_frames_index"):
        raise ValueError(f"{where}: availability source is {source!r}")
    if av.get("rule_version") != FRAME_AVAILABILITY_RULE:
        raise ValueError(f"{where}: availability rule is "
                         f"{av.get('rule_version')!r}, expected "
                         f"{FRAME_AVAILABILITY_RULE!r}")
    for key in ("excluded_row_key_sha256", "available_row_key_sha256"):
        if not _is_sha256(av.get(key)):
            raise ValueError(f"{where}: availability block lacks {key}")
    n_excl_raw = av.get("n_excluded_rows", av.get("n_excluded"))
    n_excl = -1 if n_excl_raw is None else int(n_excl_raw)
    if n_excl < 0:
        raise ValueError(f"{where}: availability block lacks n_excluded_rows")
    elig_raw = av.get("n_eligible_rows")
    avail_raw = av.get("n_available_rows")
    n_elig = -1 if elig_raw is None else int(elig_raw)
    n_avail = -1 if avail_raw is None else int(avail_raw)
    if n_elig >= 0 and n_avail >= 0 and n_elig != n_avail + n_excl:
        raise ValueError(
            f"{where}: availability counts do not add up (eligible={n_elig}, "
            f"available={n_avail}, excluded={n_excl})")
    counts = {}
    for name in ("by_split", "by_year", "by_storm"):
        block = av.get(name) or {}
        if not isinstance(block, dict):
            raise ValueError(f"{where}: availability {name} must be a mapping")
        vals = {str(k): int(v) for k, v in block.items()}
        if sum(vals.values()) != n_excl:
            raise ValueError(
                f"{where}: availability {name} sums to {sum(vals.values())}, "
                f"but n_excluded_rows={n_excl}")
        counts[name] = vals

    if source == "manifest":
        if not _is_sha256(av.get("sha256")) or not av.get("path"):
            raise ValueError(f"{where}: availability source='manifest' but the "
                             f"path/sha256 are missing")
        path = holdout_path(av["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"{where}: the frame-availability manifest this table was built "
                f"from is missing: {path}")
        if sha256_file(path).lower() != str(av["sha256"]).lower():
            raise ValueError(
                f"{where}: the frame-availability manifest changed since the "
                f"source table was built ({path}); the row set can no longer be "
                f"proven")
        block = json.loads(path.read_text(encoding="utf-8"))
        audit = validate_frame_availability(block)
        if audit["excluded_row_key_sha256"] != av["excluded_row_key_sha256"]:
            raise ValueError(f"{where}: availability excluded_row_key_sha256 "
                             f"differs from the manifest file")
        if audit["available_row_key_sha256"] != av["available_row_key_sha256"]:
            raise ValueError(f"{where}: availability available_row_key_sha256 "
                             f"differs from the manifest file")
        if int(audit["n_excluded_rows"]) != n_excl:
            raise ValueError(f"{where}: availability n_excluded_rows="
                             f"{n_excl} but the manifest lists "
                             f"{audit['n_excluded_rows']}")
        splits, years, storms = _availability_counts(block["rows"])
        for name, computed in (("by_split", splits), ("by_year", years),
                               ("by_storm", storms)):
            if counts[name] != {str(k): int(v) for k, v in computed.items()}:
                raise ValueError(f"{where}: availability {name} counts differ "
                                 f"from the manifest rows")
        both = sorted(set(str(r["row_uid"]) for r in block["rows"])
                      & set(str(k) for k in table_keys))
        if both:
            raise ValueError(
                f"{where}: the source table contains {len(both)} row(s) the "
                f"availability manifest excludes (e.g. {both[:5]})")
    else:
        if av.get("path") or av.get("sha256"):
            raise ValueError(f"{where}: derived availability must not claim a "
                             f"manifest path/sha256")
    return av


def _require_frames_qc(man: Dict[str, Any], sids, where: str, *,
                       qc_manifest: str | Path | None = None,
                       frames_dir: str | Path | None = None) -> Dict[str, Any]:
    """Fail unless the source table is pinned to the QC v1 frame corpus.

    Requires the ``frames_qc`` block (rule version, QC manifest path/sha,
    ``stores_sha256``, per-SID frame-store content hash, frames-index hash) and
    the top-level ``frames_qc_sha256`` / ``qc_manifest_sha256`` aliases the
    packer reads.  The recorded QC manifest is re-hashed and re-validated, so a
    swapped or partial corpus cannot be smuggled in through the manifest.
    """
    fq = man.get("frames_qc")
    if not isinstance(fq, dict) or not fq:
        raise ValueError(
            f"{where}: build manifest has no frames_qc block; the source table "
            f"must be pinned to the QC v1 frame corpus (rule "
            f"{QC_RULE_VERSION!r}, {QC_FRAMES_REL})")
    if fq.get("production") is False or fq.get("source") == "raw_audit":
        raise ValueError(
            f"{where}: source table was built from the RAW HURSAT corpus "
            f"(audit-only build); a production source table must use the QC v1 "
            f"corpus ({QC_FRAMES_REL})")
    for key in ("frames_qc_sha256", "qc_manifest_sha256"):
        if not _is_sha256(man.get(key)):
            raise ValueError(
                f"{where}: build manifest lacks {key} (the SHA-256 of the QC v1 "
                f"manifest); the source table is not pinned to a QC corpus")
    top_sha = str(man["frames_qc_sha256"]).lower()
    if str(man["qc_manifest_sha256"]).lower() != top_sha:
        raise ValueError(
            f"{where}: frames_qc_sha256 and qc_manifest_sha256 disagree; they "
            f"must both be the SHA-256 of the QC v1 manifest")
    if fq.get("qc_rule_version") != QC_RULE_VERSION:
        raise ValueError(
            f"{where}: frames_qc rule version is {fq.get('qc_rule_version')!r}, "
            f"expected {QC_RULE_VERSION!r}")
    for key in ("stores_sha256", "frame_stores_sha256", "frames_index_sha256"):
        if not _is_sha256(fq.get(key)):
            raise ValueError(
                f"{where}: frames_qc lacks {key}; the QC store set, the frame "
                f"store contents and the frames index must all be hashed")
    if not fq.get("qc_manifest"):
        raise ValueError(f"{where}: frames_qc does not name the QC manifest it "
                         f"was built from")
    if not fq.get("out_frames_dir"):
        raise ValueError(f"{where}: frames_qc does not record out_frames_dir")

    rec_path = holdout_path(fq["qc_manifest"])
    if not rec_path.is_file():
        raise FileNotFoundError(
            f"{where}: the QC v1 manifest this source table was built from is "
            f"missing: {rec_path} (it must travel with the source table)")
    if sha256_file(rec_path).lower() != top_sha:
        raise ValueError(
            f"{where}: the QC v1 manifest changed since the source table was "
            f"built ({rec_path} re-hashes differently); rebuild the source "
            f"table or restore the QC manifest")
    qc_info = load_qc_binding(rec_path)
    if fq.get("stores_sha256") != qc_info["stores_sha256"]:
        raise ValueError(
            f"{where}: frames_qc.stores_sha256 differs from the QC manifest's "
            f"store set")
    if int(fq.get("n_stores", -1)) != int(qc_info["n_stores"]):
        raise ValueError(
            f"{where}: frames_qc.n_stores={fq.get('n_stores')!r} but the QC "
            f"manifest lists {qc_info['n_stores']} stores")
    if (_canon_path(fq["out_frames_dir"])
            != _canon_path(qc_info["out_frames_dir"])):
        raise ValueError(
            f"{where}: frames_qc.out_frames_dir="
            f"{fq['out_frames_dir']!r} but the QC manifest describes "
            f"{qc_info['out_frames_dir']!r}")

    raw = man.get("raw_inputs") or {}
    if not raw.get("frames") or not raw.get("frames_time_axis"):
        raise ValueError(f"{where}: build manifest has no frame store "
                         f"provenance; every frame array and time axis needs a "
                         f"path and a content sha256")
    if frame_stores_sha256(raw, sids) != fq.get("frame_stores_sha256"):
        raise ValueError(
            f"{where}: frames_qc.frame_stores_sha256 does not match the "
            f"recorded per-SID frame store hashes")
    missing_qc = sorted(set(str(s) for s in sids) - set(qc_info["sids"]))
    if missing_qc:
        raise ValueError(
            f"{where}: {len(missing_qc)} source SID(s) are not in the QC "
            f"manifest (e.g. {missing_qc[:5]}); the table was not built over "
            f"this QC corpus")
    idx_sha = ((man.get("inputs") or {}).get("frames_index") or {}).get("sha256")
    if idx_sha != fq["frames_index_sha256"]:
        raise ValueError(
            f"{where}: frames_qc.frames_index_sha256 differs from "
            f"inputs.frames_index.sha256")

    if qc_manifest is not None:
        expected = Path(qc_manifest)
        if not expected.is_file():
            raise FileNotFoundError(
                f"{where}: configured QC manifest is missing: {expected}")
        if sha256_file(expected).lower() != top_sha:
            raise ValueError(
                f"{where}: the configured QC manifest ({expected}) differs from "
                f"the one the source table was built from; the source table is "
                f"not a build over this corpus")
    if frames_dir is not None:
        if _canon_path(frames_dir) != _canon_path(qc_info["out_frames_dir"]):
            raise ValueError(
                f"{where}: the configured frames_dir ({frames_dir}) is not "
                f"the QC manifest's corpus ({qc_info['out_frames_dir']})")
    return fq


def _canon_path(path) -> Path:
    """Canonical absolute path; relative paths anchor at the phase-2 root."""
    pp = Path(str(path))
    return pp.resolve() if pp.is_absolute() else (P2_ROOT / pp).resolve()


def require_ships12_source_manifest(source_parquet: str | Path, *,
                                    protocol: SplitProtocol | None = None,
                                    qc_manifest: str | Path | None = None,
                                    frames_dir: str | Path | None = None,
                                    verify_raw: bool = False,
                                    where: str | None = None) -> Dict[str, Any]:
    """Fail unless the source table is pinned by a complete build manifest.

    The gates are the ones a fit-only feature cache cannot infer from the
    parquet itself: protocol/population, input (panel/frames/config/build-code)
    hashes, every output row key, per-SID success/failure statistics, the
    content SHA-256 of every raw ERA5/GLORYS/frame input, the OHCD algorithm and
    source window, the COHC derivation policy, and the QC v1 frame-corpus
    binding (``frames_qc_sha256`` / ``qc_manifest_sha256``, rule version,
    ``stores_sha256``, per-SID frame store and frames-index hashes).  A table
    built from the raw (pre-QC) corpus is refused.  ``verify_raw=True`` also
    re-hashes the raw inputs, so a precomputed hash cache stays verifiable.
    """
    src = Path(source_parquet)
    where = where or str(src)
    if not src.is_file():
        raise FileNotFoundError(f"{where}: SHIPS source table missing")
    man_path = ships12_source_manifest_for(src)
    if not man_path.is_file():
        raise FileNotFoundError(
            f"{where}: source-build manifest missing ({man_path}).  The fit "
            f"feature cache may only be built from a manifest-pinned "
            f"all-system source table; rebuild it with "
            f"`python scripts/baseline_ships12.py --build-predictors`")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("kind") != SHIPS_SOURCE_KIND:
        raise ValueError(f"{where}: {man_path.name} is not a "
                         f"{SHIPS_SOURCE_KIND!r} manifest (kind="
                         f"{man.get('kind')!r})")
    if man.get("output_sha256") != sha256_file(src):
        raise ValueError(f"{where}: source table hash differs from its build "
                         f"manifest (file was rebuilt or edited after the "
                         f"manifest was written)")

    keys = source_table_row_keys(src)
    if int(man.get("row_key_count", -1)) != len(keys):
        raise ValueError(f"{where}: build manifest records "
                         f"{man.get('row_key_count')} row keys, table has "
                         f"{len(keys)}")
    if man.get("row_key_sha256") != sha256_array(keys):
        raise ValueError(
            f"{where}: row keys differ from the build manifest "
            f"(row_key_sha256 mismatch); the table is not the one the "
            f"manifest describes")
    sids = sorted({k.split("|", 1)[0] for k in keys})
    if man.get("sid_sha256") and man["sid_sha256"] != sha256_array(sids):
        raise ValueError(f"{where}: SID set differs from the build manifest")

    block = man.get("protocol") or {}
    populations = [str(p) for p in (man.get("population"),
                                    block.get("population")) if p]
    if not populations:
        raise ValueError(f"{where}: build manifest does not declare its "
                         f"population; the all-system (alltc, 725 canonical "
                         f"systems) population is required")
    for pop in populations:
        if pop != "alltc":
            raise ValueError(
                f"{where}: build manifest population is {pop!r}; the all-system "
                f"(alltc, 725 canonical systems) population is required")
    if protocol is not None:
        if block.get("protocol") != protocol.name:
            raise ValueError(f"{where}: build manifest protocol is "
                             f"{block.get('protocol')!r}, expected "
                             f"{protocol.name!r}")
        for name in ("fit_years", "inner_years", "test_years"):
            if list(block.get(name) or []) != list(getattr(protocol, name)):
                raise ValueError(
                    f"{where}: build manifest {name}="
                    f"{block.get(name)!r} differs from the configured "
                    f"{list(getattr(protocol, name))}")

    stats = man.get("storm_stats") or {}
    failures = stats.get("failures") or []
    if failures or int(stats.get("n_failed", -1)) != 0:
        raise ValueError(f"{where}: source build recorded per-storm failures "
                         f"({failures}); the all-system table must be 725/725")
    per_sid = stats.get("per_sid") or {}
    if (int(stats.get("n_ok", -1)) != int(stats.get("n_storms", -2))
            or set(per_sid) != set(sids)):
        raise ValueError(
            f"{where}: per-SID success/failure stats cover "
            f"{len(per_sid)} SIDs, the table has {len(sids)} (n_ok="
            f"{stats.get('n_ok')}, n_storms={stats.get('n_storms')})")
    for sid in sids:
        rec = per_sid.get(sid) or {}
        if rec.get("status") != "ok" or int(rec.get("n_rows", -1)) < 1:
            raise ValueError(f"{where}: SID {sid} is not recorded as a complete "
                             f"success ({rec})")

    raw = man.get("raw_inputs") or {}
    for group in SHIPS_RAW_GROUPS:
        entries = raw.get(group)
        if not entries:
            raise ValueError(
                f"{where}: build manifest has no {group} provenance; every raw "
                f"ERA5/GLORYS input needs a path and a content sha256")
        for entry in entries:
            _require_entry_sha(entry, where, group)
        group_sids = {str(e.get("sid")) for e in entries}
        if group_sids != set(sids):
            raise ValueError(
                f"{where}: {group} provenance covers {len(group_sids)} SIDs, "
                f"the table has {len(sids)}")
        if verify_raw:
            for entry in entries:
                p = Path(entry["path"])
                p = p if p.is_absolute() else P2_ROOT / p
                if not p.is_file():
                    raise FileNotFoundError(f"{where}: raw input recorded in the "
                                            f"manifest is missing: {p}")
                if sha256_file(p) != entry["sha256"]:
                    raise ValueError(f"{where}: raw input changed since the "
                                     f"source table was built: {entry['path']}")

    ohcd = man.get("ohcd") or {}
    if ohcd.get("algorithm") != OHCD_ALGORITHM:
        raise ValueError(f"{where}: build manifest OHCD algorithm is "
                         f"{ohcd.get('algorithm')!r}, expected "
                         f"{OHCD_ALGORITHM!r}")
    if ohcd.get("origin") != OHCD_ORIGIN_RECOMPUTED:
        raise ValueError(f"{where}: build manifest OHCD origin is "
                         f"{ohcd.get('origin')!r}, expected "
                         f"{OHCD_ORIGIN_RECOMPUTED!r}")
    if int(ohcd.get("window_hours", -1)) != 24:
        raise ValueError(f"{where}: build manifest OHCD window is "
                         f"{ohcd.get('window_hours')!r} h, expected 24")
    if ohcd.get("selection") != OHCD_SELECTION_RULE:
        raise ValueError(f"{where}: build manifest OHCD selection rule is "
                         f"{ohcd.get('selection')!r}, expected "
                         f"{OHCD_SELECTION_RULE!r}")

    cohc = man.get("cohc") or {}
    for key in ("derivation", "source_time_window"):
        if not cohc.get(key):
            raise ValueError(f"{where}: build manifest lacks COHC provenance "
                             f"({key}); COHC is a fit-window climatology with "
                             f"content hashes and must be recorded")
    if cohc.get("in_source_table"):
        raise ValueError(f"{where}: build manifest says the source table "
                         f"carries a COHC column; the source table must not "
                         f"contain a climatological OHC predictor")

    _require_frames_qc(man, sids, where, qc_manifest=qc_manifest,
                       frames_dir=frames_dir)
    _require_availability(man, keys, where)
    return man


def require_ohcd_origin(manifest: Dict[str, Any], *,
                        features_parquet: str | Path | None = None,
                        where: str = "ships12 features") -> Dict[str, Any]:
    """Fail unless OHCD was recomputed with the BACKWARD rule.

    ``source_table_column`` is the pre-fix origin: the archived predictor table
    sampled the daily OHC field by interpolation, so 06/12/18 UTC read part of
    the next day.  It is rejected here rather than warned about.
    """
    origin = manifest.get("ohcd_origin")
    if origin == OHCD_ORIGIN_SOURCE_TABLE:
        raise ValueError(
            f"{where}: OHCD was inherited from the archived predictor table "
            f"(ohcd_origin={OHCD_ORIGIN_SOURCE_TABLE!r}); its daily OHC field "
            f"was time-interpolated, so 06/12/18 UTC leaked the next day. "
            f"Rebuild with scripts/baseline_ships12.py (OHCD is recomputed "
            f"backward from the raw GLORYS fields).")
    if origin != OHCD_ORIGIN_RECOMPUTED:
        raise ValueError(f"{where}: OHCD origin is {origin!r}, expected "
                         f"{OHCD_ORIGIN_RECOMPUTED!r}")
    ohcd = manifest.get("ohcd") or {}
    if ohcd.get("algorithm") != OHCD_ALGORITHM:
        raise ValueError(f"{where}: OHCD algorithm is "
                         f"{ohcd.get('algorithm')!r}, expected "
                         f"{OHCD_ALGORITHM!r}")
    if features_parquet is not None:
        table = Path(features_parquet)
        want = manifest.get("output_sha256")
        if not want:
            raise ValueError(f"{where}: feature manifest lacks output_sha256")
        got = sha256_file(table)
        if got != want:
            raise ValueError(f"{where}: {table.name} hash differs from its "
                             f"manifest ({want} != {got})")
    return ohcd


def require_cohc_provenance(manifest: Dict[str, Any],
                            where: str = "ships12 features") -> Dict[str, Any]:
    """Fail unless the COHC climatology records a fit-window content fingerprint."""
    if manifest.get("ohc_var") != "COHC":
        raise ValueError(f"{where}: expected the COHC sensitivity, manifest "
                         f"declares ohc_var={manifest.get('ohc_var')!r}")
    cohc = manifest.get("cohc") or {}
    source = cohc.get("source") or {}
    for key in ("content_sha256", "file_list_sha256", "time_window"):
        if not source.get(key):
            raise ValueError(f"{where}: COHC manifest lacks source.{key}; the "
                             f"climatology is not pinned to fit-window content")
    if not cohc.get("sha256"):
        raise ValueError(f"{where}: COHC manifest lacks the cache sha256")
    return cohc


def _resolve_manifest_spelling(path: Path) -> Path:
    """Existing manifest file under either fixed spelling (else the input)."""
    if path.is_file():
        return path
    name = path.name
    if name.endswith(".parquet.manifest.json"):
        alt = path.with_name(name[: -len(".parquet.manifest.json")]
                            + ".manifest.json")
    elif name.endswith(".manifest.json"):
        alt = path.with_name(name[: -len(".manifest.json")]
                            + ".parquet.manifest.json")
    else:
        return path
    return alt if alt.is_file() else path


def require_ships12_run_provenance(run_dir: str | Path) -> Dict[str, Any]:
    """Validate a finished ships12 baseline run's OHCD/COHC provenance.

    Used by ``scripts/finalize_holdout.py`` (lock) and ``scripts/test_infer.py``
    (inference): a run whose feature table came from the leaky archived OHCD
    column, or that cannot be traced to a pinned feature table, is rejected.
    """
    run_dir = Path(run_dir)
    man_path = run_dir / "run_manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"{run_dir}: missing run_manifest.json")
    meta = json.loads(man_path.read_text(encoding="utf-8"))
    if meta.get("kind") != "ships12":
        return {}
    spec = meta.get("feature_spec") or {}
    ohc_var = meta.get("ohc_var") or spec.get("ohc_var")
    if ohc_var not in ("OHCD", "COHC"):
        raise ValueError(f"{run_dir}: ships12 run does not record which OHC "
                         f"variable it used (ohc_var={ohc_var!r})")
    table = spec.get("features_parquet") or spec.get("predictor_parquet")
    if not table:
        raise ValueError(f"{run_dir}: run_manifest.feature_spec does not name "
                         f"the fit-only SHIPS feature table")
    table = Path(table)
    if not table.is_absolute():
        from_cwd = Path.cwd() / table
        table = from_cwd if from_cwd.is_file() else (run_dir / table)
    fman_path = ships12_feature_manifest_for(table)
    if not fman_path.is_file():
        raise FileNotFoundError(f"{run_dir}: feature table provenance manifest "
                                f"missing ({fman_path})")
    fman = json.loads(fman_path.read_text(encoding="utf-8"))
    if fman.get("ohc_var") != ohc_var:
        raise ValueError(f"{run_dir}: feature manifest declares "
                         f"ohc_var={fman.get('ohc_var')!r}, run used "
                         f"{ohc_var!r}")
    pinned = spec.get("features_parquet_sha256")
    if pinned and pinned != fman.get("output_sha256"):
        raise ValueError(f"{run_dir}: feature table hash recorded in the run "
                         f"differs from the feature manifest")
    if ohc_var == "OHCD":
        require_ohcd_origin(fman, features_parquet=table, where=str(run_dir))
    else:
        require_cohc_provenance(fman, where=str(run_dir))
    source_man = fman.get("source_manifest") or {}
    if source_man:
        sm_path = Path(source_man.get("path") or "")
        if not sm_path.is_absolute():
            sm_path = P2_ROOT / sm_path
        sm_path = _resolve_manifest_spelling(sm_path)
        if not sm_path.is_file():
            raise FileNotFoundError(
                f"{run_dir}: the source-build manifest the feature table was "
                f"derived from is missing ({sm_path}); pack it with the "
                f"all-system source table")
        got = sha256_file(sm_path)
        if got != source_man.get("sha256"):
            raise ValueError(f"{run_dir}: source-build manifest changed since "
                             f"the fit feature table was built")
    return {"ohc_var": ohc_var,
            "ohcd_origin": fman.get("ohcd_origin"),
            "features_manifest_sha256": sha256_file(fman_path),
            "features_parquet_sha256": sha256_file(table),
            "source_manifest_sha256": (source_man or {}).get("sha256")}


def ohc_backward_selection(ohc_da, t0: pd.Timestamp, window=None):
    """(sorted position, source_time) of the field whose window has closed.

    Same rule as ``ohc_backward_slice``; the source timestamp is returned so a
    caller can record/audit the expected (``ISO_TIME - 24 h``) against the
    actually selected daily field.
    """
    win = OHC_DAILY_WINDOW if window is None else pd.Timedelta(window)
    t = pd.Timestamp(t0)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    tt = pd.to_datetime(ohc_da["time"].values)
    order = np.argsort(tt.values)
    tt = tt[order]
    window_end = tt.values + np.timedelta64(int(win.total_seconds()), "s")
    idx = int(np.searchsorted(window_end, np.datetime64(t), side="right")) - 1
    if idx < 0:
        return None, None
    return idx, pd.Timestamp(tt[idx])


def ohc_backward_slice(ohc_da, t0: pd.Timestamp, window=None):
    """OHC field whose daily averaging window has closed at t0 (None if none has).

    CMEMS GLORYS12V1 daily fields are daily MEANS over [D 00 UTC, D+1 00 UTC),
    so the field stamped D may only be used from D+1 00 UTC: the selection rule
    is ``source_time + window <= target``, not ``source_time <= target``.
    Hence 00/06/12/18 UTC on day D all use day D-1's field, and day D+1 00 UTC
    is the first moment day D's field is complete.  There is deliberately no
    nearest-future fallback, and no partially covered day: a target earlier than
    the first field's window end returns None.
    """
    idx, _src = ohc_backward_selection(ohc_da, t0, window)
    if idx is None:
        return None
    order = np.argsort(pd.to_datetime(ohc_da["time"].values).values)
    return np.asarray(ohc_da.isel(time=order).isel(time=idx).values,
                      dtype=np.float64).squeeze()


def _geo_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Pick the storm-centre lat/lon columns that are in DEGREES.

    The fit-only processed panel carries both the raw IBTrACS coordinates
    (``LAT``/``LON``) and ``USA_LAT``/``USA_LON``, and the latter have been
    standardised by the feature scaler (mean 0, sd 1).  Using those for a 100 km
    disc yields an empty mask and turns every OHCD into NaN, so a standardised
    pair is rejected explicitly instead of being silently accepted.
    """
    candidates = (("LAT", "LON"), ("USA_LAT", "USA_LON"))
    rejected = []
    for lat_col, lon_col in candidates:
        if lat_col not in frame.columns or lon_col not in frame.columns:
            continue
        lat = pd.to_numeric(frame[lat_col], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(frame[lon_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            rejected.append(f"{lat_col}/{lon_col}: all NaN")
            continue
        lat_v, lon_v = lat[ok], lon[ok]
        looks_scaled = (abs(float(lat_v.mean())) < 0.5
                        and abs(float(lat_v.std()) - 1.0) < 0.05
                        and abs(float(lon_v.mean())) < 0.5
                        and abs(float(lon_v.std()) - 1.0) < 0.05)
        in_degrees = (np.abs(lat_v).max() <= 90.0
                      and np.abs(lon_v).max() <= 540.0)
        if looks_scaled:
            rejected.append(f"{lat_col}/{lon_col}: standardised (not degrees)")
            continue
        if not in_degrees:
            rejected.append(
                f"{lat_col}/{lon_col}: lat=[{lat_v.min():.2f},{lat_v.max():.2f}] "
                f"lon=[{lon_v.min():.2f},{lon_v.max():.2f}] not degrees")
            continue
        return lat_col, lon_col
    raise ValueError(
        "no usable storm-centre coordinates in the panel for the OHC 100 km "
        "disc; candidates rejected: " + "; ".join(rejected or ["none present"]))


def ohc_areal_mean(ohc_da, t0: pd.Timestamp, lon0: float, lat0: float,
                   radius_km: float = 100.0) -> float:
    """D-1 daily-mean OHC (kJ/cm2), cos(lat) area-weighted mean over the disc."""
    if ohc_da is None:
        return np.nan
    a = ohc_backward_slice(ohc_da, t0)
    if a is None or a.ndim != 2 or not np.isfinite(a).any():
        return np.nan
    lon2d, lat2d = np.meshgrid(np.asarray(ohc_da.longitude.values, dtype=float),
                               np.asarray(ohc_da.latitude.values, dtype=float))
    mask = _dist_km(lon2d, lat2d, float(lon0), float(lat0)) <= radius_km
    ok = np.isfinite(a)
    w = np.cos(np.radians(lat2d)) * mask * ok
    den = float(np.sum(w))
    if den == 0.0:
        return np.nan
    return float(np.sum(np.where(ok, a, 0.0) * w) / den)


def recompute_ohcd(df: pd.DataFrame, ohc_dir: str | Path,
                   ohc_path_for: Callable[[str], Path] | None = None
                   ) -> tuple[pd.Series, Dict[str, Any]]:
    """Recompute OHCD row-by-row from the daily GLORYS fields.

    The daily field is a daily mean, so the selection uses
    ``source_time + 24 h <= ISO_TIME`` (see ``ohc_backward_selection``).  Fields
    are opened once per storm.

    Every OHC file the panel needs must open and must yield at least one closed
    daily window; a missing/unreadable file, a row whose selected
    ``source_time`` is not closed at ``ISO_TIME``, or a storm with no value at
    all is collected per SID and raised.  Nothing is skipped with ``continue``:
    an all-NaN OHCD column can therefore never reach the feature cache.  The
    returned stats record, per SID, the expected file, its content SHA-256, the
    number of rows, the latest selected ``source_time`` and the smallest
    ``ISO_TIME - source_time`` lead, so the window rule is auditable.
    """
    import xarray as xr

    ohc_dir = Path(ohc_dir)
    if not ohc_dir.is_dir():
        raise FileNotFoundError(f"OHC directory missing: {ohc_dir}")
    path_for = ohc_path_for or (lambda sid: ohc_dir / str(sid) / f"{sid}_ohc.nc")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    failures: list = []
    per_storm: Dict[str, Any] = {}
    n_open = n_values = 0
    for sid, idx in df.groupby("SID", sort=False).groups.items():
        sid = str(sid)
        p = path_for(sid)
        rec: Dict[str, Any] = {"file": str(p), "n_rows": int(len(idx)),
                               "n_values": 0, "n_without_closed_window": 0,
                               "max_source_time": None, "min_lead_hours": None,
                               "n_source_steps": 0, "file_sha256": None}
        if not p.is_file():
            failures.append(f"{sid}: expected OHC file {p} is missing")
            per_storm[sid] = {**rec, "status": "missing"}
            continue
        try:
            da = xr.open_dataset(p)["OHC"]
        except Exception as exc:
            failures.append(
                f"{sid}: cannot open expected OHC file {p} "
                f"({type(exc).__name__}: {exc})")
            per_storm[sid] = {**rec, "status": "unreadable"}
            continue
        n_open += 1
        try:
            rec["file_sha256"] = sha256_file(p)
            stamps = pd.to_datetime(da["time"].values)
            rec["n_source_steps"] = int(len(stamps))
            for i in idx:
                t0 = pd.Timestamp(df.at[i, "ISO_TIME"])
                if t0.tz is not None:
                    t0 = t0.tz_convert("UTC").tz_localize(None)
                _pos, src_time = ohc_backward_selection(da, t0)
                if src_time is None:
                    rec["n_without_closed_window"] += 1
                    continue
                if src_time + OHC_DAILY_WINDOW > t0:
                    failures.append(
                        f"{sid}: selected source_time {src_time} is not closed "
                        f"at ISO_TIME {t0} (source_time + 24 h > ISO_TIME)")
                    continue
                lead = (t0 - src_time).total_seconds() / 3600.0
                rec["min_lead_hours"] = (lead if rec["min_lead_hours"] is None
                                         else min(rec["min_lead_hours"], lead))
                if (rec["max_source_time"] is None
                        or str(src_time) > rec["max_source_time"]):
                    rec["max_source_time"] = str(src_time)
                val = ohc_areal_mean(da, t0, float(df.at[i, "USA_LON"]),
                                     float(df.at[i, "USA_LAT"]))
                if np.isfinite(val):
                    out.loc[i] = val
                    rec["n_values"] += 1
                    n_values += 1
        except Exception as exc:
            failures.append(
                f"{sid}: cannot read the daily OHC field of {p} "
                f"({type(exc).__name__}: {exc})")
            per_storm[sid] = {**rec, "status": "unreadable"}
            continue
        finally:
            da.close()
        if rec["n_values"] == 0:
            span = ("none" if rec["n_source_steps"] == 0 else
                    f"{str(stamps.min())}..{str(stamps.max())}")
            failures.append(
                f"{sid}: no panel row has a closed daily window in {p} "
                f"(expected source_time <= ISO_TIME - 24 h; file stamps {span})")
            rec["status"] = "no_closed_window"
        else:
            rec["status"] = "ok"
        per_storm[sid] = rec
    if failures:
        head = "; ".join(failures[:25])
        more = (f" (+{len(failures) - 25} more)" if len(failures) > 25 else "")
        raise RuntimeError(
            f"{len(failures)} OHC source-time/SID failure(s) while recomputing "
            f"OHCD from {ohc_dir}: {head}{more}")
    stats = {"algorithm": OHCD_ALGORITHM,
             "selection": OHCD_SELECTION_RULE,
             "ohc_dir": str(ohc_dir),
             "window_hours": int(OHC_DAILY_WINDOW.total_seconds() // 3600),
             "n_storms_opened": int(n_open),
             "n_storms_without_ohc": 0,
             "n_rows_without_value": int(len(df) - n_values),
             "n_rows_with_value": int(n_values),
             "per_storm": per_storm,
             "lat_span": [float(df["USA_LAT"].min()), float(df["USA_LAT"].max())],
             "lon_span": [float(df["USA_LON"].min()), float(df["USA_LON"].max())]}
    return out, stats


def _fill_clim(clim: np.ndarray) -> np.ndarray:
    """Original baseline_ships12 fill: 40x 3x3 nanmean, then month mean.

    Numerically identical to ``generic_filter(np.nanmean, size=3,
    mode="nearest")`` but vectorized with ``uniform_filter`` over valid-cell
    counts (the Python-callback version is prohibitively slow on the 141x181
    grid).  This is the same local 3x3 neighbourhood mean, NOT a global-mean
    fallback; a window with no valid cell stays NaN.
    """
    from scipy.ndimage import uniform_filter

    for mo in range(12):
        a = clim[mo]
        for _ in range(40):
            if np.isfinite(a).all():
                break
            valid = np.isfinite(a)
            if not valid.any():
                break
            sums = uniform_filter(np.where(valid, a, 0.0), size=3,
                                  mode="nearest")
            cnts = uniform_filter(valid.astype(np.float64), size=3,
                                  mode="nearest")
            with np.errstate(invalid="ignore", divide="ignore"):
                filled = sums / cnts
            a = np.where(valid, a, filled)
        m = np.nanmean(a)
        clim[mo] = np.where(np.isfinite(a), a, m)
    return clim


def build_cohc_clim(fit_years, ohc_dir: str | Path,
                    cache_path: str | Path,
                    manifest_path: str | Path | None = None,
                    grid: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Monthly OHC climatology pooled over FIT-year data only.

    Two independent guards:

    * the FIT-YEAR window is applied **per time step**
      (``lo <= time < hi`` with lo=2000-01-01, hi=2018-01-01), not only to the
      file list: a 2017 storm file that also contains 2018 days contributes
      only its in-window timestamps.  Out-of-window steps are counted and
      reported;
    * the cache provenance carries a CONTENT SHA-256 of every source file plus
      the cache file's own SHA-256.  Reuse requires both to match, so a file
      whose name is unchanged but whose content changed is detected.
    """
    import xarray as xr

    ohc_dir = Path(ohc_dir)
    files = sorted(ohc_dir.glob("*/*_ohc.nc"))
    if not files:
        raise FileNotFoundError(f"no *_ohc.nc under {ohc_dir}")
    lo_year, hi_year = int(fit_years[0]), int(fit_years[1])
    fit_files = [f for f in files
                 if lo_year <= int(f.parent.name[:4]) <= hi_year]
    if not fit_files:
        raise ValueError("no OHC files inside the fit years")

    # source CONTENT fingerprint (not just names)
    rel_names = [f"{f.parent.name}/{f.name}" for f in fit_files]
    content = hashlib.sha256()
    for rel, f in zip(rel_names, fit_files):
        content.update(rel.encode("utf-8"))
        content.update(sha256_file(f).encode("utf-8"))
    source = {
        "n_files": len(fit_files),
        "fit_years": [lo_year, hi_year],
        "time_window": [f"{lo_year}-01-01T00:00:00", f"{hi_year + 1}-01-01T00:00:00"],
        "file_list_sha256": sha256_array(rel_names),
        "content_sha256": content.hexdigest(),
        "content_hash_method": "sha256 of sorted 'relpath:file_sha256' tokens",
    }
    cache_path = Path(cache_path)
    manifest_path = (Path(manifest_path) if manifest_path
                     else cache_path.with_suffix(".manifest.json"))
    if cache_path.is_file():
        if not manifest_path.is_file():
            raise ValueError(f"{cache_path} exists without a provenance "
                             f"manifest; delete it and rebuild")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("source") != source:
            raise ValueError("OHC climatology cache provenance changed "
                             "(source content or window); rebuild it explicitly")
        want = old.get("cache_sha256")
        if not want:
            raise ValueError("OHC climatology cache manifest lacks cache_sha256")
        got = sha256_file(cache_path)
        if got != want:
            raise ValueError(f"OHC climatology cache file is corrupt/changed "
                             f"({want} != {got}); rebuild it")
        z = np.load(cache_path)
        return {"lat": z["lat"], "lon": z["lon"], "clim": z["clim"],
                "source": old["source"], "stats": old.get("stats", {}),
                "sha256": got}

    if grid is None:
        step = 0.5
        lat_g = np.arange(-10.0, 60.0 + step, step)
        lon_g = np.arange(95.0, 185.0 + step, step)
    else:
        lat_g = np.asarray(grid["lat"], dtype=float)
        lon_g = np.asarray(grid["lon"], dtype=float)
        step = float(lat_g[1] - lat_g[0])
        if len(lon_g) < 2 or abs((lon_g[1] - lon_g[0]) - step) > 1e-9:
            raise ValueError("test/override grid must be uniform with equal "
                             "lat/lon spacing")
    sums = np.zeros((12, len(lat_g), len(lon_g)), dtype=np.float64)
    cnts = np.zeros_like(sums)
    n_steps_total = 0
    n_steps_used = 0
    n_steps_out = 0
    unreachable = []
    for k, f in enumerate(fit_files):
        try:
            da = xr.open_dataset(f)["OHC"]
        except Exception as exc:
            # never skipped silently: a fit source file that should be read but
            # cannot be is a hard failure of the climatology
            raise RuntimeError(
                f"COHC climatology: cannot open fit-window OHC file {f} "
                f"({type(exc).__name__}: {exc}); expected every "
                f"{lo_year}-{hi_year} storm file to be readable") from exc
        if len(da["time"].values) == 0:
            da.close()
            unreachable.append(f"{f.parent.name}/{f.name}: empty time axis")
            continue
        latv, lonv = da.latitude.values, da.longitude.values
        ilat = np.clip(np.round((latv - lat_g[0]) / step).astype(int),
                       0, len(lat_g) - 1)
        ilon = np.clip(np.round((lonv - lon_g[0]) / step).astype(int),
                       0, len(lon_g) - 1)
        times = pd.to_datetime(da["time"].values)
        tz = getattr(times, "tz", None)
        lo = pd.Timestamp(f"{lo_year}-01-01", tz=tz)
        hi = pd.Timestamp(f"{hi_year + 1}-01-01", tz=tz)
        in_window = (times >= lo) & (times < hi)
        n_steps_total += int(len(times))
        n_steps_used += int(in_window.sum())
        n_steps_out += int((~in_window).sum())
        if not in_window.any():
            da.close()
            unreachable.append(f"{f.parent.name}/{f.name}: no fit-window step "
                               f"({str(times.min())}..{str(times.max())})")
            continue
        months = times.month.values
        vals = np.asarray(da.values, dtype=np.float64)
        for ti, mo in enumerate(months):
            if not in_window[ti]:
                continue
            a = vals[ti]
            valid = np.isfinite(a)
            if not valid.any():
                continue
            ii, jj = np.meshgrid(ilat, ilon, indexing="ij")
            np.add.at(sums[mo - 1], (ii[valid], jj[valid]), a[valid])
            np.add.at(cnts[mo - 1], (ii[valid], jj[valid]), 1.0)
        if (k + 1) % 200 == 0:
            print(f"  COHC clim: {k + 1}/{len(fit_files)}", flush=True)
        da.close()
    if n_steps_used == 0:
        raise RuntimeError(
            "COHC climatology: no fit-window time step contributed "
            f"(window {lo_year}-01-01 <= t < {hi_year + 1}-01-01); "
            f"unusable files: {unreachable[:10]}")
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = sums / cnts
    clim = _fill_clim(clim)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, lat=lat_g, lon=lon_g, clim=clim)
    stats = {"n_time_steps_total": n_steps_total,
             "n_time_steps_used": n_steps_used,
             "n_time_steps_out_of_window": n_steps_out,
             "n_files_without_fit_window": len(unreachable),
             "files_without_fit_window": unreachable}
    cache_sha = sha256_file(cache_path)
    manifest_path.write_text(json.dumps(
        {"source": source, "stats": stats, "cache_sha256": cache_sha,
         "sha256": cache_sha,  # compatibility alias
         "method": {"algorithm": "fit-window-monthly-ohc-climatology",
                    "fit_only": True,
                    "source_time_window": source["time_window"],
                    "window_applied_per_time_step": True,
                    "content_hash": "source.content_sha256"},
         "created_at": pd.Timestamp.utcnow().isoformat()},
        indent=2, sort_keys=True), encoding="utf-8")
    return {"lat": lat_g, "lon": lon_g, "clim": clim, "source": source,
            "stats": stats, "sha256": cache_sha}


def cohc_lookup(clim: Dict[str, Any], lat0: float, lon0: float,
                month: int) -> float:
    """0-100 km areal mean of the monthly climatology around (lat0, lon0)."""
    step = float(clim["lat"][1] - clim["lat"][0])
    ilat = int(round((lat0 - clim["lat"][0]) / step))
    ilon = int(round((lon0 - clim["lon"][0]) / step))
    r = int(np.ceil(100.0 / KM_PER_DEG / step)) + 1
    a = clim["clim"][month - 1]
    lo_i, hi_i = max(0, ilat - r), min(a.shape[0], ilat + r + 1)
    lo_j, hi_j = max(0, ilon - r), min(a.shape[1], ilon + r + 1)
    lat2d, lon2d = np.meshgrid(clim["lat"][lo_i:hi_i],
                               clim["lon"][lo_j:hi_j], indexing="ij")
    d = _dist_km(lon2d, lat2d, lon0, lat0)
    sub = a[lo_i:hi_i, lo_j:hi_j]
    vals = sub[d <= 100.0]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def build_ships12_features(cfg, predictor_parquet: str | Path,
                           out_path: str | Path, ohc_var: str = "OHCD",
                           cohc_builder: Optional[Callable] = None,
                           ohc_dir: str | Path | None = None) -> Path:
    """Fit-only SHIPS-12 predictor table.

    ``ohc_var="OHCD"`` is the main arm; ``"COHC"`` is a separate sensitivity in
    which the fit-only COHC climatology replaces OHCD.  The two never coexist,
    and the source table's old COHC/MPI columns are never used.

    ``ohc_dir`` is REQUIRED for OHCD: the archived predictor table's OHCD column
    was built with the old (future-peeking) time selection, so inheriting it is
    refused outright.  OHCD is always recomputed from the daily GLORYS fields
    with the backward rule and the manifest records
    ``ohcd_origin="recomputed_backward"``.

    The source table must carry its mandatory build manifest
    (``<table>.manifest.json``, see ``require_ships12_source_manifest``): the
    row keys, per-SID success/failure stats, raw ERA5/GLORYS content hashes, the
    QC v1 frame-corpus binding (``frames_qc_sha256`` / ``qc_manifest_sha256``)
    and the OHCD/COHC policy of the all-system build are re-verified, and the
    manifest SHA-256 is pinned in the feature manifest (and in cache reuse).
    When the config declares ``data.qc_manifest`` / ``data.frames_dir`` they are
    cross-checked against the source build, so a QC-corpus swap is refused.
    """
    if ohc_var not in ("OHCD", "COHC"):
        raise ValueError(f"ohc_var must be OHCD or COHC, got {ohc_var!r}")
    if ohc_var == "OHCD" and ohc_dir is None:
        raise ValueError(
            "OHCD must be recomputed from the daily GLORYS fields: pass "
            "ohc_dir.  The archived predictor table's OHCD column came from "
            "linear time interpolation, i.e. 06/12/18 UTC read part of the "
            "next day; inheriting it is not allowed.")
    cfg = _as_cfg(cfg)
    proto = SplitProtocol.from_cfg(cfg)
    qc_binding = (holdout_frames_binding(cfg)
                  if cfg.get("data.qc_manifest") else {})
    # the row universe is eligible panel - frame-availability exclusions, the
    # same set the DL dataset uses; never "all eligible rows"
    eligible = eligible_panel_rows(cfg)
    index_keys = None
    if cfg.get("data.frames_index"):
        idx_path = holdout_path(cfg.get("data.frames_index"))
        if idx_path.is_file():
            index_keys = frame_index_row_keys(idx_path)
    avail = frame_availability_binding(cfg, eligible=eligible,
                                       index_keys=index_keys)
    avail_rec = availability_record(avail)
    src = Path(predictor_parquet)
    out_path = Path(out_path)
    manifest_path = out_path.with_suffix(".manifest.json")
    source_man_path = ships12_source_manifest_for(src)
    source_man_sha = (sha256_file(source_man_path)
                      if source_man_path.is_file() else None)
    ohcd_source = (OHCD_ORIGIN_RECOMPUTED if ohc_var == "OHCD" else None)
    if out_path.is_file():
        if not manifest_path.is_file():
            raise ValueError(f"{out_path} exists without a provenance manifest")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (ohc_var == "OHCD"
                and old.get("ohcd_origin") == OHCD_ORIGIN_SOURCE_TABLE):
            # a cache built from the archived column must never be reused
            require_ohcd_origin(old, features_parquet=out_path,
                                where=str(out_path))
        qc_cfg_sha = (sha256_file(qc_binding["qc_manifest"])
                      if qc_binding.get("qc_manifest") else None)
        qc_mismatch = bool(qc_cfg_sha) and (
            old.get("frames_qc_sha256") != qc_cfg_sha)
        avail_old = old.get("availability") or {}
        avail_mismatch = (
            str(avail_old.get("sha256")) != str(avail_rec.get("sha256"))
            or str(avail_old.get("excluded_row_key_sha256"))
            != str(avail_rec.get("excluded_row_key_sha256"))
            or int(avail_old.get("n_excluded_rows",
                                 avail_old.get("n_excluded", -1)))
            != int(avail_rec["n_excluded_rows"]))
        old_src_man_sha = ((old.get("source_manifest") or {}).get("sha256")
                           or old.get("source_manifest_sha256"))
        if (old.get("source_sha256") != sha256_file(src)
                or old_src_man_sha != source_man_sha
                or old.get("fit_years") != list(proto.fit_years)
                or old.get("ohc_var") != ohc_var
                or old.get("ohcd_origin") != ohcd_source
                or qc_mismatch
                or avail_mismatch
                or old.get("output_sha256") != sha256_file(out_path)):
            drifted = [
                name for name, bad in (
                    ("source_sha256",
                     old.get("source_sha256") != sha256_file(src)),
                    ("source_manifest_sha256",
                     old_src_man_sha != source_man_sha),
                    ("fit_years",
                     old.get("fit_years") != list(proto.fit_years)),
                    ("ohc_var", old.get("ohc_var") != ohc_var),
                    ("ohcd_origin", old.get("ohcd_origin") != ohcd_source),
                    ("qc_binding", qc_mismatch),
                    ("availability", avail_mismatch),
                    ("output_sha256",
                     old.get("output_sha256") != sha256_file(out_path)),
                ) if bad]
            raise ValueError("SHIPS feature cache provenance changed "
                             f"({', '.join(drifted)}); rebuild it "
                             f"explicitly")
        require_ships12_source_manifest(
            src, protocol=proto, where=str(src),
            qc_manifest=qc_binding.get("qc_manifest"),
            frames_dir=qc_binding.get("frames_dir"))
        if ohc_var == "OHCD":
            require_ohcd_origin(old, features_parquet=out_path,
                                where=str(out_path))
        else:
            require_cohc_provenance(old, where=str(out_path))
        return out_path
    if not src.is_file():
        raise FileNotFoundError(f"SHIPS predictor table missing: {src}")
    # the source table is only usable through its build manifest (input hashes,
    # row keys, per-SID stats, QC corpus binding, OHCD/COHC provenance) -- never
    # on file name alone
    source_man = require_ships12_source_manifest(
        src, protocol=proto, where=str(src),
        qc_manifest=qc_binding.get("qc_manifest"),
        frames_dir=qc_binding.get("frames_dir"))
    pred = pd.read_parquet(src)
    for col in ("SID", "ISO_TIME"):
        if col not in pred.columns:
            raise ValueError(f"{src} missing {col}")
    needed = [p for p in SHIPS_PREDICTORS if p != "OHCD" or ohc_var == "OHCD"]
    missing = [c for c in needed if c not in pred.columns]
    if missing:
        raise ValueError(f"{src} missing SHIPS predictors: {missing}")
    if pred.duplicated(["SID", "ISO_TIME"]).any():
        raise ValueError(f"{src} has duplicate (SID, ISO_TIME) rows")
    pred["ISO_TIME"] = pd.to_datetime(pred["ISO_TIME"], utc=True)
    if (set(pred["SID"].astype(str) + "|" + pred["ISO_TIME"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ")) & set(avail["excluded"])):
        raise ValueError(
            f"{src} still contains frame-availability-excluded rows; the source "
            f"table was built over a different row universe")
    ds = RIPanelDataset(cfg)
    check_dataset_availability(ds, avail, eligible=eligible, where=str(out_path))
    lat_col, lon_col = _geo_columns(ds.df)
    key = ds.df[["row_uid", "SID", "ISO_TIME", lat_col, lon_col]].rename(
        columns={lat_col: "USA_LAT", lon_col: "USA_LON"})
    merged = key.merge(pred, on=["SID", "ISO_TIME"], how="left",
                       suffixes=("", "_src"), indicator=True)
    if (merged["_merge"] != "both").any():
        n_missing = int((merged["_merge"] != "both").sum())
        raise ValueError(f"{n_missing} dataset rows have no SHIPS row "
                         f"(SID/ISO_TIME key mismatch)")
    merged = merged.drop(columns=["_merge"])
    if len(merged) != len(ds.df) or merged["row_uid"].duplicated().any():
        raise ValueError("SHIPS merge changed the row keys")
    if not np.array_equal(merged["row_uid"].to_numpy(),
                          ds.df["row_uid"].to_numpy()):
        raise ValueError("SHIPS merge reordered rows relative to the dataset")

    predictors = list(SHIPS_PREDICTORS)
    ohcd_meta = None
    if ohc_var == "COHC":
        builder = cohc_builder or (lambda years: build_cohc_clim(
            years, P2_ROOT / "data" / "phase1" / "raw" /
            "ohc", Path(out_path).with_suffix(".cohc.npz")))
        clim = builder(proto.fit_years)
        cohc_meta = {"source": clim.get("source"),
                     "sha256": clim.get("sha256")}
        # fit-only content-hashed climatology or nothing: a COHC built without
        # a per-time fit window / source fingerprint is refused here
        require_cohc_provenance({"ohc_var": "COHC", "cohc": cohc_meta},
                                where=f"{out_path}: COHC climatology")
        want_window = [f"{proto.fit_years[0]}-01-01T00:00:00",
                       f"{proto.fit_years[1] + 1}-01-01T00:00:00"]
        got_window = list(cohc_meta["source"].get("time_window") or [])
        if got_window != want_window:
            raise ValueError(
                f"{out_path}: COHC climatology source window is {got_window}, "
                f"expected the fit window {want_window}")
        months = merged["ISO_TIME"].dt.month.values
        merged["COHC"] = [cohc_lookup(clim, float(la), float(lo), int(mo))
                          for la, lo, mo in zip(merged["USA_LAT"],
                                                merged["USA_LON"], months)]
        predictors = list(SHIPS_PREDICTORS_COHC)
    else:
        cohc_meta = None
        ohcd, ohcd_meta = recompute_ohcd(merged, ohc_dir)
        merged = merged.copy()
        merged["OHCD"] = ohcd.to_numpy()
        n_finite = int(np.isfinite(merged["OHCD"].to_numpy()).sum())
        if n_finite == 0:
            raise ValueError(
                "OHCD recomputation produced no finite value for any row: "
                f"storms_opened={ohcd_meta['n_storms_opened']}, "
                f"storms_without_ohc={ohcd_meta['n_storms_without_ohc']}, "
                f"lat_span={ohcd_meta['lat_span']}, "
                f"lon_span={ohcd_meta['lon_span']} (coordinates must be "
                f"degrees, not standardised), ohc_dir={ohcd_meta['ohc_dir']}")
        frac_missing = 1.0 - n_finite / max(len(merged), 1)
        if frac_missing > 0.5:
            # a few leading rows can legitimately precede the first completed
            # daily window; losing half the table means something systematic
            raise ValueError(
                f"OHCD recomputation left {len(merged) - n_finite} of "
                f"{len(merged)} rows without a value ({frac_missing:.0%}) "
                f"although {ohcd_meta['n_storms_opened']} storms were opened "
                f"and {ohcd_meta['n_storms_without_ohc']} had no OHC file; "
                f"lat_span={ohcd_meta['lat_span']}, "
                f"lon_span={ohcd_meta['lon_span']}")

    _check_feature_columns(predictors, "ships12")
    out = merged[["row_uid"] + predictors].copy()
    if out[predictors].notna().sum().sum() == 0:
        raise ValueError("SHIPS feature table is entirely NaN")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    source_cols = [c for c in pred.columns if c not in ("SID", "ISO_TIME")]
    manifest_path.write_text(json.dumps({
        "kind": "ships12_features",
        "source": _rel(src),
        "source_sha256": sha256_file(src),
        "source_manifest": {
            "path": _rel(source_man_path),
            "sha256": sha256_file(source_man_path),
            "row_key_sha256": source_man.get("row_key_sha256"),
            "raw_inputs_sha256": source_man.get("raw_inputs_sha256"),
            "protocol": source_man.get("protocol"),
            "frames_qc_sha256": source_man.get("frames_qc_sha256"),
            "qc_manifest_sha256": source_man.get("qc_manifest_sha256"),
            "frames_qc": source_man.get("frames_qc"),
            "availability": source_man.get("availability"),
            "storm_stats": {
                "n_storms": source_man.get("storm_stats", {}).get("n_storms"),
                "n_ok": source_man.get("storm_stats", {}).get("n_ok"),
                "n_failed": source_man.get("storm_stats", {}).get("n_failed"),
            },
        },
        "availability": avail_rec,
        "frames_qc_sha256": source_man.get("frames_qc_sha256"),
        "qc_manifest_sha256": source_man.get("qc_manifest_sha256"),
        "qc_rule_version": (source_man.get("frames_qc") or {}).get(
            "qc_rule_version"),
        "source_columns": source_cols,
        "ignored_source_columns": sorted(set(source_cols) - set(predictors)),
        "fit_years": list(proto.fit_years),
        "ohc_var": ohc_var,
        "main_or_sensitivity": "main" if ohc_var == "OHCD" else "sensitivity",
        "ohcd_origin": ohcd_source,
        "ohcd": ohcd_meta,
        "predictors": predictors,
        "cohc": cohc_meta,
        "output_sha256": sha256_file(out_path),
        "row_key_sha256": sha256_array(
            sorted(out["row_uid"].astype(str).tolist())),
    }, indent=2, sort_keys=True), encoding="utf-8")
    return out_path



