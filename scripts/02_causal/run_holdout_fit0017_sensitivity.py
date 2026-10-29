# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Sensitivity variants for the all-TC fit-0017 (HOLDOUT_MASKED_V1) prior.

The 375-storm masked arm has four sensitivity analyses that the all-TC fit-only
prior does not have.  This driver adds them without touching the existing
phase-1 entry points and without ever writing into ``panel/`` or ``prior/``:

    strict      allow_target_lag0=false + target_lag0_exceptions=[] on the
                fit-only panel -> PCMCI+ discovery + effect estimation
    no_ohc      OHC_IC/OHC_OA removed from the causal variables, the treatments
                and the headroom treatments -> PCMCI+ discovery + effects
    mjo_enso    baseline ``prior/ate_results.csv`` adjustment sets + OMI1/OMI2
                (daily OMI) + NINO34 (Nino-3.4 anomaly); no discovery
    lat         OHC_OA adjustment set + LAT; no discovery

Reused phase-1 code
-------------------
    scripts/run_holdout_fit0017.py  configure() / select_fit_prior_rows() /
                                    row_keys() / sha256_array() -- the holdout
                                    config (all-TC universe, fit-only window,
                                    fit-only scaler/YJ, imputation-mask wiring)
    main.py                         run_causal_discovery() /
                                    run_effect_estimation() / run_knowledge_base()
    scripts/mjo_enso_confounding_check.py
                                    adjustment-set re-estimation for MJO/ENSO
                                    (subprocess: the module resolves its
                                    WPAC_* paths at import time)
    analysis/forecast_utility/ohc_oa_lat_sensitivity.py
                                    adjustment-set re-estimation for LAT
                                    (same subprocess convention)

Fit-window rule: every fitted quantity here is computed from the fit-only panel
(2000-2017, feature time <= 2017-12-25 18Z, realised F24 label < 2018-01-01).
Rows are re-derived with ``select_fit_prior_rows`` and hard-failed if they differ
from ``prior/fit_prior_panel.parquet`` by a single row key; the two panel files
are additionally bound byte-for-byte to the sha256 values recorded in
``prior/prior_manifest.json``.

Fail-closed guards
------------------
* ``assert_config_whitelist`` -- each variant may change only its declared
  config keys (strict: the two ``allow_target_lag0`` keys; no_ohc: the three
  variable/treatment lists; adjustment variants: nothing).  The assertion runs
  before any expensive stage.
* ``guard_snapshot`` / ``assert_untouched`` -- every baseline prior/panel
  artifact, including all effects-stage products and the ``dag_audit_all/``
  tree, is hashed before and after each variant.  A guard failure never hides
  the original exception: both are reported together.
* ``validate_adjust`` -- the re-estimation must reproduce the baseline ATE
  table's treatment/outcome pairs exactly, and must leave no PCMCI+ result in
  its directory (the adjustment variants never run discovery).

Usage (from the release root):
    # plan only -- prints the full-run commands, starts nothing
    python scripts/02_causal/run_holdout_fit0017_sensitivity.py --plan

    # smoke: ~25 storms stratified over every fit year (2000-2017), one
    # treatment, writes under _smoke_sensitivity/
    python scripts/02_causal/run_holdout_fit0017_sensitivity.py --variant all \
        --smoke --smoke-sids 25 --smoke-treatments SHRD

    # full run (strict/no_ohc ~3 h each, mjo_enso/lat minutes)
    python scripts/02_causal/run_holdout_fit0017_sensitivity.py --variant all

    # recovery: reuse a stored PCMCI+ result instead of re-running discovery
    python scripts/02_causal/run_holdout_fit0017_sensitivity.py --variant strict \
        --effects-only --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _entry in (str(ROOT / "scripts" / "02_causal"), str(ROOT / "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
os.chdir(ROOT)

import run_holdout_fit0017 as holdout
from main import run_causal_discovery, run_effect_estimation, run_knowledge_base
from wpac_causal.knowledge_base import KnowledgeBaseBuilder
from wpac_causal.utils import Config, setup_logging

HOLDOUT_ROOT = ROOT / "data/phase1/processed_masked/holdout_fit0017"
PANEL_DIR = HOLDOUT_ROOT / "panel"
PRIOR_DIR = HOLDOUT_ROOT / "prior"
DEFAULT_SENS_ROOT = HOLDOUT_ROOT / "sensitivity"
DEFAULT_SMOKE_ROOT = HOLDOUT_ROOT / "_smoke_sensitivity"
BASE_CONFIG = ROOT / "configs/phase1/config_alltc_0024.yaml"

DISCOVERY_VARIANTS = ("strict", "no_ohc")
ADJUST_VARIANTS = ("mjo_enso", "lat")
ALL_VARIANTS = ("strict", "no_ohc", "mjo_enso", "lat")

FIT_YEARS = (2000, 2017)
FIT_FEATURE_END = pd.Timestamp("2017-12-25 18:00:00", tz="UTC")
FIT_LABEL_END = pd.Timestamp("2018-01-01 00:00:00", tz="UTC")

MJO_ENSO_OUT = "ate_mjo_enso_sensitivity.csv"
LAT_OUT = "ohc_oa_lat_sensitivity.csv"

# Read-only artifacts: hashed before and after every run so an accidental write
# into the established prior/panel artifacts fails loudly instead of silently
# replacing the paper's baseline.  The list covers the prior's headline outputs
# AND every effects-stage product, because EffectEstimator writes its derived
# CSVs into ``ate_csv.parent`` -- a mis-redirected path would otherwise only
# show up as a silent overwrite of the baseline tables.
GUARDED_FILES = [
    PRIOR_DIR / "pcmci_results.pkl",
    PRIOR_DIR / "ate_results.csv",
    PRIOR_DIR / "cate_results.csv",
    PRIOR_DIR / "causal_graph_edges.json",
    PRIOR_DIR / "causal_graph.dot",
    PRIOR_DIR / "causal_graph.png",
    PRIOR_DIR / "causal_effects_db.yaml",
    PRIOR_DIR / "identification_dag.dot",
    PRIOR_DIR / "identification_dag.json",
    PRIOR_DIR / "identification_status.csv",
    PRIOR_DIR / "trusted_signs.json",
    PRIOR_DIR / "fit_prior_panel.parquet",
    PRIOR_DIR / "prior_manifest.json",
    PRIOR_DIR / "phase1_prior_manifest.json",
    # effects-stage products (EffectEstimator writes them next to ate_csv)
    PRIOR_DIR / "headroom_cate_results.csv",
    PRIOR_DIR / "mediation_results.csv",
    PRIOR_DIR / "regime_cate_results.csv",
    PRIOR_DIR / "positivity_diagnostics.csv",
    PRIOR_DIR / "qualitative_classification.csv",
    PRIOR_DIR / "shrd_binned_results.csv",
    PRIOR_DIR / "refutation_report.md",
    HOLDOUT_ROOT / "config_snapshot.yaml",
    PANEL_DIR / "wpac_tc_dataset_processed.parquet",
    PANEL_DIR / "imputation_mask.parquet",
    PANEL_DIR / "feature_scaler.pkl",
    PANEL_DIR / "dv24_yj_transformer.pkl",
    PANEL_DIR / "sample_manifest.csv",
    PANEL_DIR / "split_manifest.csv",
    PANEL_DIR / "panel_manifest.json",
    PANEL_DIR / "boundary_audit.json",
    PANEL_DIR / "population_audit.json",
    PANEL_DIR / "extraction_audit.json",
]
# Directories are guarded as a whole (path list + per-file digest), so adding,
# deleting or editing any audit table inside them trips the guard.
GUARDED_DIRS = [
    PRIOR_DIR / "dag_audit_all",
]

# Leaf keys that only hold output/input paths (they are *supposed* to differ
# between the prior and a sensitivity variant) plus whole sections that are
# irrelevant to the comparison.
PATH_LEAVES = {
    "results_pkl", "graph_png", "graph_dot", "ate_csv", "cate_csv",
    "refutation_md", "identification_dot", "identification_json",
    "output_yaml", "graph_edges_json", "file", "output_dir", "log_dir",
    "retry_json", "sample_manifest_csv", "extraction_audit_json",
    "yj_transformer_path", "scaler_path", "imputation_mask_path",
}
SKIP_SECTIONS = ("logging.", "data.")
LOG_SUFFIXES = (".log", ".log.stdout")

DESCRIPTIONS = {
    "strict": ("Removes every non-self contemporaneous (lag-0) link into "
               "DV24_YJ (`allow_target_lag0=false`, `target_lag0_exceptions=[]`) "
               "and re-runs PCMCI+ discovery + effect estimation on the fit-only "
               "panel."),
    "no_ohc": ("Drops OHC_IC/OHC_OA from the causal variables, the treatment "
               "list and the headroom treatments, then re-runs PCMCI+ discovery "
               "+ effect estimation on the fit-only panel."),
    "mjo_enso": ("No discovery. Re-estimates every baseline ATE after appending "
                 "OMI1, OMI2 (daily Wheeler-Hendon OMI) and NINO34 (CPC Nino-3.4 "
                 "anomaly, daily-interpolated) to the adjustment set recorded in "
                 "`prior/ate_results.csv`."),
    "lat": ("No discovery. Re-estimates the OHC_OA effect after appending LAT to "
            "its recorded adjustment set, with a 200-draw storm-level block "
            "bootstrap on the change."),
}


# ---------------------------------------------------------------------------
# Hash / guard helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return holdout.sha256(Path(path))


def rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(HOLDOUT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def hash_dir(path: Path) -> str:
    """Content digest of a directory tree (path list + per-file sha256)."""
    path = Path(path)
    entries = sorted(p for p in path.rglob("*") if p.is_file())
    if not entries:
        raise RuntimeError(f"guarded directory holds no files: {path}")
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(entry).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def guard_snapshot() -> Dict[str, str]:
    missing = [str(p) for p in GUARDED_FILES if not Path(p).is_file()]
    missing += [str(p) for p in GUARDED_DIRS if not Path(p).is_dir()]
    if missing:
        raise FileNotFoundError(
            "required fit-0017 artifact(s) missing: " + ", ".join(missing))
    snapshot = {rel(p): sha256(p) for p in GUARDED_FILES}
    for directory in GUARDED_DIRS:
        snapshot[rel(directory) + "/"] = hash_dir(directory)
    return snapshot


def assert_untouched(before: Dict[str, str]) -> None:
    after = guard_snapshot()
    changed = {key: (before[key], after[key])
               for key in before if before[key] != after[key]}
    if changed:
        raise RuntimeError(
            "sensitivity run modified protected fit-0017 artifacts: "
            + json.dumps(changed, indent=2))


# ---------------------------------------------------------------------------
# Fit panel
# ---------------------------------------------------------------------------

def _assert_fit_window(fit: pd.DataFrame) -> pd.DataFrame:
    iso = pd.to_datetime(fit["ISO_TIME"], utc=True)
    f24 = pd.to_datetime(fit["F24_TIME"], utc=True)
    if iso.max() > FIT_FEATURE_END:
        raise RuntimeError(
            f"feature time {iso.max()} exceeds the fit boundary {FIT_FEATURE_END}")
    if f24.max() >= FIT_LABEL_END:
        raise RuntimeError(
            f"realised F24 label {f24.max()} reaches the holdout window")
    years = pd.to_numeric(fit["SID"].astype(str).str[:4], errors="raise")
    if int(years.max()) > FIT_YEARS[1] or int(years.min()) < FIT_YEARS[0]:
        raise RuntimeError(
            f"storm years outside {FIT_YEARS}: {sorted(years.unique())[:3]}...")
    return fit


# Files whose bytes bind every variant result to the exact baseline panel.
PANEL_BINDING = {
    "panel/wpac_tc_dataset_processed.parquet":
        PANEL_DIR / "wpac_tc_dataset_processed.parquet",
    "panel/imputation_mask.parquet": PANEL_DIR / "imputation_mask.parquet",
}


def assert_panel_matches_prior_manifest() -> Dict[str, str]:
    """Byte-level binding to the panel the prior was fitted on.

    Row-key equality (below) proves the *selection* is unchanged; the sha256
    comparison against ``prior/prior_manifest.json`` proves the *content* is
    the same file the baseline was estimated from -- an edited column or a
    rescaled feature would otherwise slip through the row-key check.
    """
    manifest = json.loads((PRIOR_DIR / "prior_manifest.json").read_text("utf-8"))
    declared = manifest.get("inputs", {})
    observed: Dict[str, str] = {}
    for key, path in PANEL_BINDING.items():
        if key not in declared:
            raise RuntimeError(
                f"prior_manifest.json does not declare {key}; refusing to fit "
                "against an unbound panel")
        digest = sha256(path)
        if digest != declared[key]:
            raise RuntimeError(
                f"{key} sha256={digest[:16]}... differs from the prior manifest "
                f"{declared[key][:16]}...; the baseline panel changed")
        observed[key] = digest
    logger.info(
        "Panel bound to prior_manifest.json: "
        + ", ".join(f"{key}={value[:12]}..." for key, value in observed.items()))
    return observed


def stratified_smoke_sids(fit: pd.DataFrame, n: int) -> List[str]:
    """Smoke subset that covers every fit year instead of the first N SIDs.

    The first-N-by-SID subset is dominated by early years and was too weak to
    reproduce the baseline's VMAX -> DV24_YJ edges, so it could not exercise the
    strict lag-0 removal.  This picks at least one storm per year 2000-2017 and
    spreads the remaining budget round-robin across years.
    """
    storms = (fit.drop_duplicates("SID")[["SID", "storm_year"]]
              .assign(SID=lambda d: d["SID"].astype(str))
              .sort_values(["storm_year", "SID"]))
    years = sorted(storms["storm_year"].unique())
    per_year = max(1, int(n) // max(1, len(years)))
    picks: List[str] = []
    leftovers: List[List[str]] = []
    for year in years:
        group = storms.loc[storms["storm_year"] == year, "SID"].tolist()
        picks.extend(group[:per_year])
        leftovers.append(group[per_year:])
    for group in leftovers:
        if len(picks) >= int(n):
            break
        if group:
            picks.append(group[0])
            group[:] = group[1:]
    return sorted(set(picks))


def load_fit_panel(smoke_sids: Optional[int] = None) -> pd.DataFrame:
    """Fit-only panel, proven to be exactly the panel the prior was fitted on."""
    assert_panel_matches_prior_manifest()
    processed = PANEL_DIR / "wpac_tc_dataset_processed.parquet"
    fit = holdout.select_fit_prior_rows(pd.read_parquet(processed))
    prior_fit = pd.read_parquet(PRIOR_DIR / "fit_prior_panel.parquet")
    got = holdout.sha256_array(holdout.row_keys(fit))
    want = holdout.sha256_array(holdout.row_keys(prior_fit))
    if got != want:
        raise RuntimeError(
            "select_fit_prior_rows() no longer reproduces prior/fit_prior_panel.parquet")
    fit = _assert_fit_window(fit)
    logger.info(
        f"Fit-only panel: {len(fit)} rows, {fit['SID'].nunique()} SIDs, "
        f"max feature time {pd.to_datetime(fit['ISO_TIME'], utc=True).max()}")
    if smoke_sids:
        total = fit["SID"].nunique()
        keep = stratified_smoke_sids(fit, int(smoke_sids))
        fit = fit.loc[fit["SID"].astype(str).isin(keep)].copy()
        fit = _assert_fit_window(fit)
        years = sorted(pd.to_numeric(fit["SID"].astype(str).str[:4]).unique())
        logger.warning(
            f"SMOKE MODE: fit panel restricted to {len(keep)} of {total} SIDs "
            f"({len(fit)} rows) covering {len(years)} fit years "
            f"{years[0]}-{years[-1]}")
    return fit


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------

def _redirect_outputs(cfg_dict: Dict[str, Any], out_dir: Path) -> None:
    """Point every output of the prior stages at the variant directory."""
    cd = cfg_dict["causal_discovery"]
    cd["results_pkl"] = str(out_dir / "pcmci_results.pkl")
    cd["graph_png"] = str(out_dir / "causal_graph.png")
    cd["graph_dot"] = str(out_dir / "causal_graph.dot")
    ee = cfg_dict["effect_estimation"]
    ee["ate_csv"] = str(out_dir / "ate_results.csv")
    ee["cate_csv"] = str(out_dir / "cate_results.csv")
    ee["refutation_md"] = str(out_dir / "refutation_report.md")
    # EffectEstimator falls back to ate_csv.parent when these are absent, but
    # the holdout config sets them explicitly to prior/, so they MUST be
    # overridden or the variant would rewrite the paper's identification DAG.
    ee["identification_dot"] = str(out_dir / "identification_dag.dot")
    ee["identification_json"] = str(out_dir / "identification_dag.json")
    kb = cfg_dict["knowledge_base"]
    kb["output_yaml"] = str(out_dir / "causal_effects_db.yaml")
    kb["graph_edges_json"] = str(out_dir / "causal_graph_edges.json")
    cfg_dict["logging"]["file"] = str(out_dir / "run.log")


def build_variant_config(variant: str, out_dir: Path,
                         treatments: Optional[List[str]] = None) -> Config:
    """Holdout config + the single change that defines the variant."""
    if variant not in DISCOVERY_VARIANTS:
        raise ValueError(f"{variant} is not a discovery variant")
    cfg_dict = holdout.configure(str(BASE_CONFIG)).to_dict()
    if variant == "strict":
        cd = cfg_dict["causal_discovery"]
        cd["allow_target_lag0"] = False
        cd["target_lag0_exceptions"] = []
    elif variant == "no_ohc":
        cd = cfg_dict["causal_discovery"]
        cd["variables"] = [v for v in cd["variables"] if not v.startswith("OHC")]
        ee = cfg_dict["effect_estimation"]
        ee["treatments"] = [t for t in ee["treatments"] if not t.startswith("OHC")]
        ee["headroom"]["treatments"] = [
            t for t in ee["headroom"]["treatments"] if not t.startswith("OHC")]
        if any(v.startswith("OHC") for v in cd["variables"]):
            raise RuntimeError("no_ohc still has OHC causal variables")
    if treatments:
        kept = [t for t in treatments
                if t in cfg_dict["effect_estimation"]["treatments"]]
        if not kept:
            raise ValueError(
                f"{variant}: none of the smoke treatments {treatments} survive "
                f"the variant's treatment list")
        cfg_dict["effect_estimation"]["treatments"] = kept
        cfg_dict["effect_estimation"]["headroom"]["treatments"] = [
            t for t in cfg_dict["effect_estimation"]["headroom"]["treatments"]
            if t in kept]
        logger.warning(f"SMOKE MODE: effect estimation restricted to {kept}")
    _redirect_outputs(cfg_dict, out_dir)
    cfg = Config(cfg_dict)
    # Refuse before any expensive stage if the variant changed anything beyond
    # its declared keys.
    assert_config_whitelist(
        variant, semantic_diff(prior_config_snapshot(), cfg_dict),
        smoke=bool(treatments))
    return cfg


def _flat(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, name + "."))
        else:
            out[name] = value
    return out


def semantic_diff(prior_cfg: Dict[str, Any],
                  variant_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Config differences that matter scientifically (paths/sections dropped)."""
    a, b = _flat(prior_cfg), _flat(variant_cfg)
    rows = []
    for key in sorted(set(a) | set(b)):
        if key.startswith(SKIP_SECTIONS):
            continue
        if key.split(".")[-1] in PATH_LEAVES:
            continue
        if a.get(key) != b.get(key):
            rows.append({"key": key, "prior": a.get(key), "variant": b.get(key)})
    return rows


# Exactly which configuration keys each variant is allowed to change.  Anything
# outside this set means the variant silently mutated a discovery/effect
# parameter it has no business touching (e.g. tau_max, pc_alpha, variables), so
# the run is refused BEFORE the expensive stages start.
VARIANT_CONFIG_WHITELIST = {
    "strict": {
        "causal_discovery.allow_target_lag0",
        "causal_discovery.target_lag0_exceptions",
    },
    "no_ohc": {
        "causal_discovery.variables",
        "effect_estimation.treatments",
        "effect_estimation.headroom.treatments",
    },
    # Adjustment-set variants must not change ANY discovery/effect configuration:
    # they reuse prior/ate_results.csv and the stored panel.
    "mjo_enso": set(),
    "lat": set(),
}
# `--smoke --smoke-treatments` narrows the treatment lists; allowed on top.
SMOKE_CONFIG_WHITELIST = {
    "effect_estimation.treatments",
    "effect_estimation.headroom.treatments",
}


def assert_config_whitelist(variant: str, diff_rows: List[Dict[str, Any]],
                            smoke: bool) -> List[str]:
    """Fail-closed: only the variant's declared keys may differ from the prior."""
    if variant not in VARIANT_CONFIG_WHITELIST:
        raise ValueError(f"{variant} has no config whitelist")
    allowed = set(VARIANT_CONFIG_WHITELIST[variant])
    if smoke:
        allowed |= SMOKE_CONFIG_WHITELIST
    keys = sorted({str(row["key"]) for row in diff_rows})
    extra = [key for key in keys if key not in allowed]
    if extra:
        raise RuntimeError(
            f"{variant}: configuration differs from the baseline prior in "
            f"non-whitelisted key(s) {extra}; allowed={sorted(allowed)}")
    logger.info(
        f"{variant}: config whitelist satisfied "
        f"({keys if keys else 'no semantic difference'})")
    return keys


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _load_result(pkl_path: Path) -> Dict[str, Any]:
    with Path(pkl_path).open("rb") as handle:
        return pickle.load(handle)


def count_edges(result: Dict[str, Any]) -> Dict[str, Any]:
    """Edge diagnostic using the knowledge base's own edge extractor.

    ``n_edges`` therefore equals ``n_edges`` in ``causal_graph_edges.json``
    (97 for the baseline prior); ``n_edges_q_le_alpha`` counts the directed
    links whose BH-adjusted q-value is <= pc_alpha.
    """
    edges = KnowledgeBaseBuilder._extract_edges(result)
    q = [e["q_value"] for e in edges if e["q_value"] is not None]
    return {
        "n_edges": len(edges),
        "n_edges_q_le_alpha": int(sum(1 for value in q if value <= 0.01)),
        "target_parents": sorted(
            f"{e['src']} lag{e['lag']}" for e in edges if e["dst"] == "DV24_YJ"),
    }


def validate_discovery(variant: str, out_dir: Path, fit: pd.DataFrame,
                       full_scale: bool) -> Dict[str, Any]:
    """Fail-closed checks on the variant's PCMCI+ result versus the baseline."""
    pkl_path = out_dir / "pcmci_results.pkl"
    if not pkl_path.is_file():
        raise RuntimeError(f"PCMCI+ result missing: {pkl_path}")
    result = _load_result(pkl_path)
    md = result["metadata"]
    if md.get("mask_type") != "xyz":
        raise RuntimeError(f"{variant}: mask_type={md.get('mask_type')!r} != 'xyz'")
    if int(md.get("tau_max", -1)) != 4:
        raise RuntimeError(f"{variant}: tau_max={md.get('tau_max')!r} != 4")
    if float(md.get("pc_alpha", -1)) != 0.01:
        raise RuntimeError(f"{variant}: pc_alpha={md.get('pc_alpha')!r} != 0.01")
    if not str(md.get("independence_test", "")).startswith("ParCorr"):
        raise RuntimeError(
            f"{variant}: independence_test={md.get('independence_test')!r}")
    upstream = md.get("upstream_imputation", {})
    if upstream.get("status") != "applied":
        raise RuntimeError(f"{variant}: imputation mask was not applied")
    if int(upstream.get("input_rows", -1)) != len(fit):
        raise RuntimeError(
            f"{variant}: PCMCI saw {upstream.get('input_rows')} rows, "
            f"the fit panel has {len(fit)}")

    names = list(result["var_names"])
    graph = result["graph"]
    target = names.index("DV24_YJ")
    if variant == "strict":
        if md.get("allow_target_lag0") is not False:
            raise RuntimeError("strict: allow_target_lag0 was not disabled")
        if list(md.get("target_lag0_exceptions", ["?"])):
            raise RuntimeError("strict: target_lag0_exceptions is not empty")
        violations = [
            name for i, name in enumerate(names)
            if i != target and (graph[i, target, 0] != "" or graph[target, i, 0] != "")
        ]
        if violations:
            raise RuntimeError(
                f"strict: non-self target lag-0 links remain: {violations}")
        if md.get("lag0_target_edges_sync_risk"):
            raise RuntimeError("strict: lag-0 target edges still flagged")
    if variant == "no_ohc":
        leftover = [name for name in names if name.startswith("OHC")]
        if leftover:
            raise RuntimeError(f"no_ohc: OHC nodes remain: {leftover}")
    if full_scale:
        prior_md = _load_result(PRIOR_DIR / "pcmci_results.pkl")["metadata"]
        if md.get("n_storms") != prior_md.get("n_storms"):
            raise RuntimeError(
                f"{variant}: n_storms={md.get('n_storms')} != baseline "
                f"{prior_md.get('n_storms')}")
        if int(md.get("n_valid_cells", -1)) <= 0:
            raise RuntimeError(f"{variant}: no valid cells")

    edges = count_edges(result)
    baseline = count_edges(_load_result(PRIOR_DIR / "pcmci_results.pkl"))
    logger.info(
        f"{variant}: PCMCI+ OK -- {len(names)} nodes, {edges['n_edges']} edges "
        f"(baseline {baseline['n_edges']}), {edges['n_edges_q_le_alpha']} at "
        f"q<=0.01 (baseline {baseline['n_edges_q_le_alpha']}); "
        f"target parents: {edges['target_parents']}")
    return {
        "var_names": names,
        "n_edges": edges["n_edges"],
        "n_edges_q_le_alpha": edges["n_edges_q_le_alpha"],
        "target_parents": edges["target_parents"],
        "baseline": baseline,
        "metadata": {key: md.get(key) for key in
                     ("mask_type", "tau_max", "pc_alpha", "independence_test",
                      "allow_target_lag0", "target_lag0_exceptions", "n_storms",
                      "n_vars", "n_valid_cells", "tigramite_version")},
    }


def validate_effects(variant: str, out_dir: Path,
                     full_scale: bool) -> Dict[str, Any]:
    ate_path = out_dir / "ate_results.csv"
    if not ate_path.is_file():
        raise RuntimeError(f"{variant}: ate_results.csv missing")
    ate = pd.read_csv(ate_path)
    if ate.empty:
        raise RuntimeError(f"{variant}: ate_results.csv is empty")
    required = {"treatment", "outcome", "ate", "adjustment_set",
                "identification_policy"}
    missing = required - set(ate.columns)
    if missing:
        raise RuntimeError(f"{variant}: ate_results.csv lacks {sorted(missing)}")
    if variant == "no_ohc" and any(
            str(t).startswith("OHC") for t in ate["treatment"]):
        raise RuntimeError("no_ohc: OHC treatments survived in the ATE table")
    expected = 16 if variant == "no_ohc" else 20
    row_count_ok = len(ate) == expected
    if full_scale and not row_count_ok:
        logger.warning(
            f"{variant}: {len(ate)} ATE rows, expected {expected} "
            "(treatments x {DV24, RI}); a treatment/outcome pair without a "
            "valid estimate is recorded, not fatal")
    for name in ("causal_graph_edges.json", "identification_dag.json",
                 "causal_effects_db.yaml"):
        if not (out_dir / name).is_file():
            logger.warning(f"{variant}: expected output missing: {name}")
    logger.info(
        f"{variant}: effects OK -- {len(ate)} ATE rows, "
        f"treatments={sorted(ate['treatment'].unique())}")
    return {
        "n_ate_rows": int(len(ate)),
        "expected_ate_rows": int(expected),
        "row_count_ok": bool(row_count_ok),
        "treatments": sorted(ate["treatment"].unique().tolist()),
        "outcomes": sorted(ate["outcome"].unique().tolist()),
    }


MJO_ADDED_COVARIATES = ("OMI1", "OMI2", "NINO34")


def _parse_adjust(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if text in ("", "(empty)", "empty", "None", "nan"):
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def annotate_mjo_csv(out_csv: Path) -> None:
    """Record the expanded adjustment set next to the 375-protocol columns.

    The reused script only exports ``adjust_base``; the artifact is only
    self-describing once the three appended indices are visible per row.
    """
    frame = pd.read_csv(out_csv)
    frame["adjust_mjo_enso"] = [
        ",".join(_parse_adjust(value) + list(MJO_ADDED_COVARIATES)) or "(empty)"
        for value in frame["adjust_base"]
    ]
    frame["added_covariates"] = ",".join(MJO_ADDED_COVARIATES)
    frame.to_csv(out_csv, index=False)


def baseline_pairs() -> pd.DataFrame:
    """(treatment, outcome) pairs of the baseline prior ATE table."""
    ate = pd.read_csv(PRIOR_DIR / "ate_results.csv")
    pairs = ate[["treatment", "outcome"]].drop_duplicates().reset_index(drop=True)
    if pairs.empty:
        raise RuntimeError("prior/ate_results.csv has no treatment/outcome pairs")
    return pairs


def validate_adjust(variant: str, out_dir: Path) -> Dict[str, Any]:
    out_csv = out_dir / (MJO_ENSO_OUT if variant == "mjo_enso" else LAT_OUT)
    if not out_csv.is_file():
        raise RuntimeError(f"{variant}: expected output missing: {out_csv}")
    if (out_dir / "pcmci_results.pkl").exists():
        raise RuntimeError(
            f"{variant} must not run discovery, but {out_dir/'pcmci_results.pkl'} "
            "exists")
    res = pd.read_csv(out_csv)
    if res.empty:
        raise RuntimeError(f"{variant}: output is empty")
    pairs = baseline_pairs()
    if variant == "mjo_enso":
        wanted = {"treatment", "outcome", "ate_base", "ate_mjo_enso",
                  "delta", "rel_change_pct", "sign_flip", "adjust_base",
                  "adjust_mjo_enso", "added_covariates", "n_obs"}
        expected = len(pairs)
        have = set(zip(res["treatment"], res["outcome"]))
        want = set(zip(pairs["treatment"], pairs["outcome"]))
    else:
        wanted = {"outcome", "adjust_base", "adjust_lat", "ate_base", "ate_lat",
                  "change_lat_minus_base", "n_obs", "n_storms", "boot_B"}
        expected = int((pairs["treatment"] == "OHC_OA").sum())
        have = set(res["outcome"])
        want = set(pairs.loc[pairs["treatment"] == "OHC_OA", "outcome"])
    missing = wanted - set(res.columns)
    if missing:
        raise RuntimeError(f"{variant}: output lacks {sorted(missing)}")
    if len(res) != expected:
        raise RuntimeError(
            f"{variant}: {len(res)} rows, but the baseline ATE table has "
            f"{expected} matching pair(s); the re-estimation dropped rows")
    if have != want:
        raise RuntimeError(
            f"{variant}: pairs differ from the baseline ATE table "
            f"(missing={sorted(want - have)}, extra={sorted(have - want)})")
    if variant == "mjo_enso":
        for row in res.itertuples():
            expanded = _parse_adjust(row.adjust_mjo_enso)
            base = _parse_adjust(row.adjust_base)
            if expanded != base + list(MJO_ADDED_COVARIATES):
                raise RuntimeError(
                    f"mjo_enso: adjust_mjo_enso {expanded} is not "
                    f"adjust_base + {list(MJO_ADDED_COVARIATES)}")
    else:
        if not res["adjust_lat"].astype(str).str.split(",").apply(
                lambda parts: "LAT" in [p.strip() for p in parts]).all():
            raise RuntimeError("lat: LAT is not in every adjustment set")
        if int(res["boot_B"].max()) != 200:
            raise RuntimeError(
                f"lat: unexpected bootstrap B={int(res['boot_B'].max())}")
        if not (res["outcome"] == "RI").any():
            raise RuntimeError("lat: RI outcome missing")
    logger.info(
        f"{variant}: adjustment-set re-estimation OK ({len(res)} rows, "
        f"baseline pairs {expected})")
    return {"rows": int(len(res)),
            "expected_rows": int(expected),
            "baseline_pairs": pairs.to_dict("records"),
            "added_covariates": (list(MJO_ADDED_COVARIATES)
                                 if variant == "mjo_enso" else ["LAT"]),
            "discovery_rerun": False,
            "columns": list(res.columns),
            "head": json.loads(res.head(3).to_json(orient="records"))}


def run_validation_only(variant: str, out_dir: Path,
                        fit: pd.DataFrame) -> Dict[str, Any]:
    """Validate artifacts of an earlier run (no computation)."""
    if variant in DISCOVERY_VARIANTS:
        return {"discovery": validate_discovery(variant, out_dir, fit, True),
                "effects": validate_effects(variant, out_dir, True)}
    return {"adjust": validate_adjust(variant, out_dir)}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def write_config_snapshot(cfg: Config, out_dir: Path) -> Path:
    path = out_dir / "config_snapshot.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, allow_unicode=True)
    return path


def run_discovery_variant(variant: str, out_dir: Path, fit: pd.DataFrame,
                          treatments: Optional[List[str]],
                          discovery_only: bool,
                          effects_only: bool = False) -> Dict[str, Any]:
    cfg = build_variant_config(variant, out_dir, treatments)
    write_config_snapshot(cfg, out_dir)
    started = time.time()
    if effects_only:
        pkl_path = out_dir / "pcmci_results.pkl"
        if not pkl_path.is_file():
            raise RuntimeError(
                f"--effects-only needs an existing {pkl_path}; run discovery first")
        if treatments:
            raise RuntimeError(
                "--effects-only cannot be combined with a smoke treatment subset: "
                "the stored PCMCI+ result covers the full treatment list")
        logger.info(f"[{variant}] effects-only: PCMCI+ reused from {pkl_path}")
    else:
        logger.info(
            f"[{variant}] PCMCI+ discovery on {len(fit)} fit rows / "
            f"{fit['SID'].nunique()} SIDs -> {out_dir}")
        run_causal_discovery(cfg, fit)
    discovery = validate_discovery(variant, out_dir, fit,
                                   full_scale=not treatments)
    effects: Dict[str, Any] = {}
    if not discovery_only:
        logger.info(f"[{variant}] effect estimation (same fit panel)")
        run_effect_estimation(cfg, fit)
        run_knowledge_base(cfg, fit)
        effects = validate_effects(variant, out_dir, full_scale=not treatments)
    elapsed = time.time() - started
    logger.info(f"[{variant}] finished in {elapsed / 60:.1f} min")
    return {"discovery": discovery, "effects": effects,
            "runtime_seconds": round(elapsed, 1), "config": cfg.to_dict(),
            "config_diff": semantic_diff(prior_config_snapshot(),
                                         cfg.to_dict()),
            "effects_only": effects_only}


def _run_subprocess(script: Path, env_extra: Dict[str, str], log_path: Path) -> None:
    env = os.environ.copy()
    env.update(env_extra)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("command: " + " ".join([sys.executable, str(script)]) + "\n")
        handle.write("env: " + json.dumps(env_extra, indent=2) + "\n\n")
        handle.flush()
        subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env,
                       check=True, stdout=handle, stderr=subprocess.STDOUT)


def smoke_panel_view(fit: pd.DataFrame, out_dir: Path) -> Path:
    """Smoke runs need a truncated panel file; full runs read prior/ directly."""
    path = out_dir / "fit_panel_smoke_subset.parquet"
    fit.to_parquet(path)
    return path


def run_adjust_variant(variant: str, out_dir: Path, fit: pd.DataFrame,
                       smoke: bool) -> Dict[str, Any]:
    """Re-estimate effects with an extended adjustment set (no discovery)."""
    started = time.time()
    baseline_ate = PRIOR_DIR / "ate_results.csv"
    panel = smoke_panel_view(fit, out_dir) if smoke else PRIOR_DIR / "fit_prior_panel.parquet"
    if variant == "mjo_enso":
        script = ROOT / "scripts" / "mjo_enso_confounding_check.py"
        out_csv = out_dir / MJO_ENSO_OUT
        env_extra = {
            "WPAC_MJO_OUT": str(out_csv),
            "WPAC_PANEL": str(panel),
            "WPAC_BASELINE_ATE": str(baseline_ate),
        }
    elif variant == "lat":
        script = ROOT / "analysis" / "forecast_utility" / "ohc_oa_lat_sensitivity.py"
        out_csv = out_dir / LAT_OUT
        env_extra = {
            "WPAC_OHC_OA_LAT_OUT": str(out_csv),
            "WPAC_PANEL": str(panel),
            "WPAC_BASELINE_ATE": str(baseline_ate),
            "WPAC_SCALER": str(PANEL_DIR / "feature_scaler.pkl"),
        }
    else:
        raise ValueError(f"unknown adjustment variant {variant}")
    if not script.is_file():
        raise FileNotFoundError(f"reused script missing: {script}")
    logger.info(f"[{variant}] {script.name} -> {out_csv} (panel {rel(panel)})")
    _run_subprocess(script, env_extra, out_dir / "run.log.stdout")
    if variant == "mjo_enso":
        annotate_mjo_csv(out_csv)
        logger.info(
            f"[mjo_enso] recorded expanded adjustment sets "
            f"(+{list(MJO_ADDED_COVARIATES)}) in {out_csv.name}")
    result = validate_adjust(variant, out_dir)
    elapsed = time.time() - started
    logger.info(f"[{variant}] finished in {elapsed:.1f} s")
    return {"adjust": result, "runtime_seconds": round(elapsed, 1),
            "reused_script": script.relative_to(ROOT).as_posix(),
            "env": env_extra, "panel": str(Path(panel).resolve())}


# ---------------------------------------------------------------------------
# Manifest / README
# ---------------------------------------------------------------------------

def prior_config_snapshot() -> Dict[str, Any]:
    return yaml.safe_load((HOLDOUT_ROOT / "config_snapshot.yaml").read_text("utf-8"))


def prior_phase_timings() -> Dict[str, float]:
    """Measured stage durations of the baseline fit-0017 run (seconds)."""
    log = HOLDOUT_ROOT / "run.log"
    timings: Dict[str, float] = {}
    if not log.is_file():
        return timings
    text = log.read_text(encoding="utf-8", errors="replace")
    for match in re.finditer(r"Phase END: (\w+) \| elapsed=([0-9.]+)s", text):
        timings.setdefault(match.group(1), float(match.group(2)))
    return timings


def variant_inputs(variant: str) -> Dict[str, str]:
    inputs = [
        PANEL_DIR / "wpac_tc_dataset_processed.parquet",
        PANEL_DIR / "imputation_mask.parquet",
        PANEL_DIR / "feature_scaler.pkl",
        PANEL_DIR / "dv24_yj_transformer.pkl",
        PRIOR_DIR / "fit_prior_panel.parquet",
        PRIOR_DIR / "ate_results.csv",
        HOLDOUT_ROOT / "config_snapshot.yaml",
    ]
    if variant in DISCOVERY_VARIANTS:
        inputs.append(PRIOR_DIR / "pcmci_results.pkl")
    hashes = {rel(p): sha256(p) for p in inputs}
    if variant == "mjo_enso":
        for name in ("omi_indices.txt", "nino34_indices.txt"):
            path = ROOT / "data/phase1/raw" / name
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    return hashes


def variant_outputs(out_dir: Path) -> Dict[str, str]:
    out = {}
    for path in sorted(Path(out_dir).iterdir()):
        if not path.is_file():
            continue
        if path.name in {"MANIFEST.json", "SENSITIVITY_README.md"}:
            continue
        if path.name.endswith(LOG_SUFFIXES):
            continue
        if path.name == "fit_panel_smoke_subset.parquet":
            continue  # smoke-only input view, hashed under "inputs"
        out[path.name] = sha256(path)
    return out


def write_manifest(out_dir: Path, record: Dict[str, Any]) -> Path:
    path = out_dir / "MANIFEST.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "`(unset)`"
    if isinstance(value, (list, tuple, dict)):
        return "`" + json.dumps(value) + "`"
    return f"`{value}`"


def readme_section(record: Dict[str, Any]) -> str:
    lines = [f"## {record['variant']}  ({record['kind']})", ""]
    lines.append(record["description"])
    lines.append("")
    lines.append(f"- output directory: `{record['out_dir']}`")
    lines.append(f"- mode: {'SMOKE (not a result)' if record['smoke'] else 'full scale'}"
                 f" -- {record['n_rows']} rows / {record['n_sids']} SIDs")
    lines.append(f"- reused code: `{record['reused_code']}`")
    if record.get("panel_used"):
        lines.append(f"- panel read by the reused script: `{record['panel_used']}`")
    if record.get("added_covariates"):
        lines.append("- added covariates: "
                     + ", ".join(f"`{name}`" for name in record["added_covariates"]))
        lines.append(f"- discovery re-run: {record.get('discovery_rerun')}")
    lines.append(f"- runtime: {record['runtime_seconds']} s")
    lines.append("")
    lines.append("### Configuration difference vs the baseline prior")
    lines.append("")
    diffs = record.get("config_diff", [])
    if diffs:
        lines.append("| key | baseline prior | this variant |")
        lines.append("| --- | --- | --- |")
        for row in diffs:
            lines.append(f"| `{row['key']}` | {_fmt(row['prior'])} | "
                         f"{_fmt(row['variant'])} |")
    else:
        lines.append(record.get("config_diff_note", "no semantic difference"))
    lines.append("")
    if record.get("env"):
        lines.append("### Environment overrides passed to the reused script")
        lines.append("")
        for key, value in record["env"].items():
            lines.append(f"- `{key}` = `{value}`")
        lines.append("")
    lines.append("### Input hashes (sha256)")
    lines.append("")
    for key, value in record["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("### Outputs (sha256; logs excluded)")
    lines.append("")
    for key, value in record["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("### Run logs")
    lines.append("")
    for key in record["logs"]:
        lines.append(f"- `{key}`")
    lines.append("")
    if record.get("checks"):
        lines.append("### Checks")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(record["checks"], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


README_HEADER = (
    "# fit-0017 sensitivity variants\n\n"
    "Sensitivity analyses for the all-TC `HOLDOUT_MASKED_V1` prior "
    "(`data/processed_masked/holdout_fit0017/prior/`). Generated by "
    "`scripts/run_holdout_fit0017_sensitivity.py`.\n\n"
    "Every variant reads the fit-only panel (2000-2017, feature time <= "
    "2017-12-25 18Z) and the established `prior/` artifacts; nothing in "
    "`panel/` or `prior/` is written. Discovery parameters are identical to "
    "the baseline (ParCorr, tau_max=4, pc_alpha=0.01, mask_type=xyz).\n\n"
    "Not produced here (deliberately out of scope, and available from the "
    "baseline `prior/`): `identification_status.csv`, `dag_audit/`, "
    "`trusted_signs.json`, `prior_manifest.json`. They can be regenerated for a "
    "variant by pointing `paper_supplementary/code/identification_status.py` at "
    "the variant's `pcmci_results.pkl` / `ate_results.csv` / "
    "`identification_dag.{dot,json}` through its `WPAC_*` environment "
    "variables.\n"
)


def write_readmes(root: Path, record: Dict[str, Any]) -> None:
    (root / record["variant"] / "SENSITIVITY_README.md").write_text(
        README_HEADER + "\n" + readme_section(record), encoding="utf-8")
    sections = [readme_section(json.loads(manifest.read_text("utf-8")))
                for manifest in sorted(root.glob("*/MANIFEST.json"))]
    (root / "SENSITIVITY_README.md").write_text(
        README_HEADER + "\n" + "\n".join(sections), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def plan_commands(variants: List[str], root: Path) -> str:
    timings = prior_phase_timings()
    disc = timings.get("CausalDiscovery", 0.0)
    eff = timings.get("EffectEstimation", 0.0)
    lines = ["Full-run commands (--plan executes nothing):", ""]
    exe = sys.executable
    for variant in variants:
        lines.append(f"{exe} scripts/run_holdout_fit0017_sensitivity.py "
                     f"--variant {variant}")
        if variant in DISCOVERY_VARIANTS:
            hours = (disc + eff) / 3600.0
            note = ""
            if variant == "no_ohc":
                note = ("  (upper bound: no_ohc has 20 nodes / 8 treatments, "
                        "so ~2.5-2.7 h expected)")
            lines.append(f"    measured baseline: PCMCI+ {disc / 60:.1f} min + "
                         f"effects {eff / 60:.1f} min -> ~{hours:.1f} h "
                         f"({'EXCEEDS 2 h' if hours > 2 else 'under 2 h'}){note}")
        else:
            lines.append("    adjustment-set re-estimation only (no discovery): "
                         "minutes")
        lines.append(f"    outputs: {root / variant}")
        lines.append("")
    lines.append("Smoke equivalent: --smoke --smoke-sids 25 "
                 f"--smoke-treatments SHRD --out-root {DEFAULT_SMOKE_ROOT} "
                 "(storms stratified over 2000-2017)")
    lines.append("Recovery after a failure: --variant <v> --effects-only --force "
                 "(reuses the stored pcmci_results.pkl)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sensitivity variants for the all-TC fit-0017 causal prior")
    parser.add_argument("--variant", default="all",
                        choices=list(ALL_VARIANTS) + ["all"])
    parser.add_argument("--out-root", default=str(DEFAULT_SENS_ROOT),
                        help="root for the variant directories "
                             f"(default: {DEFAULT_SENS_ROOT})")
    parser.add_argument("--smoke", action="store_true",
                        help="small-sample smoke run (see --smoke-sids)")
    parser.add_argument("--smoke-sids", type=int, default=25,
                        help="number of SIDs kept in smoke mode (default 25)")
    parser.add_argument("--smoke-treatments", default="SHRD",
                        help="comma-separated treatments for smoke effects "
                             "(default SHRD)")
    parser.add_argument("--discovery-only", action="store_true",
                        help="discovery variants: stop after PCMCI+/graph export")
    parser.add_argument("--effects-only", action="store_true",
                        help="discovery variants: reuse the stored PCMCI+ result "
                             "and re-run only effects (needs --force)")
    parser.add_argument("--validate-only", action="store_true",
                        help="re-check artifacts of an existing run; compute nothing")
    parser.add_argument("--force", action="store_true",
                        help="allow writing into an existing variant directory")
    parser.add_argument("--plan", action="store_true",
                        help="print the full-run commands and exit")
    args = parser.parse_args()

    variants = list(ALL_VARIANTS) if args.variant == "all" else [args.variant]
    root = Path(args.out_root)
    if args.smoke and root == DEFAULT_SENS_ROOT and not args.validate_only:
        root = DEFAULT_SMOKE_ROOT
        logger.warning(f"smoke mode: writing under {root}")

    if args.plan:
        print(plan_commands(variants, root))
        return 0
    if args.effects_only and (args.smoke or args.discovery_only):
        raise SystemExit(
            "--effects-only reuses a full-scale PCMCI+ result: it cannot be "
            "combined with --smoke or --discovery-only")

    before = guard_snapshot()
    treatments = None
    if args.smoke and args.smoke_treatments and not args.validate_only:
        treatments = [t.strip() for t in args.smoke_treatments.split(",")
                      if t.strip()]
    fit = load_fit_panel(args.smoke_sids if args.smoke and not args.validate_only
                         else None)

    for variant in variants:
        original: Optional[BaseException] = None
        try:
            _run_variant(variant, root, args, fit, treatments)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            original = exc
        guard_failure: Optional[BaseException] = None
        try:
            assert_untouched(before)
            logger.info("Protected prior/panel artifacts are unchanged")
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            guard_failure = exc
        if guard_failure is not None and original is not None:
            raise RuntimeError(
                f"{guard_failure}\n[also] variant {variant} failed first with "
                f"{type(original).__name__}: {original}") from original
        if guard_failure is not None:
            raise guard_failure
        if original is not None:
            raise original
    return 0


def _run_variant(variant: str, root: Path, args: argparse.Namespace,
                 fit: pd.DataFrame,
                 treatments: Optional[List[str]]) -> Dict[str, Any]:
    out_dir = root / variant
    out_resolved = out_dir.resolve()
    for guarded_dir in (PANEL_DIR.resolve(), PRIOR_DIR.resolve()):
        if out_resolved == guarded_dir or guarded_dir in out_resolved.parents:
            raise RuntimeError(
                f"refusing to write {variant} outputs inside {guarded_dir}")
    if (not args.validate_only and out_dir.exists()
            and any(out_dir.iterdir()) and not args.force):
        raise RuntimeError(
            f"{out_dir} already holds outputs; pass --force to overwrite "
            "(the baseline prior/panel are never touched)")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        # Read-only verification: never rewrite MANIFEST/README, so a smoke
        # artifact keeps its own provenance (rows/SIDs/smoke flag).
        setup_logging(level="INFO")
        logger.info(f"=== validate-only {variant} -> {out_dir} (no writes) ===")
        checks = run_validation_only(variant, out_dir, fit)
        snap = out_dir / "config_snapshot.yaml"
        cfg_dict = yaml.safe_load(snap.read_text("utf-8")) if snap.is_file() else None
        previous = out_dir / "MANIFEST.json"
        previously_smoke = (bool(json.loads(previous.read_text("utf-8")).get("smoke"))
                            if previous.is_file() else False)
        assert_config_whitelist(
            variant,
            semantic_diff(prior_config_snapshot(), cfg_dict) if cfg_dict else [],
            smoke=previously_smoke)
        logger.info(f"=== validate-only {variant} passed (no files written) ===")
        return {"variant": variant, "validate_only": True, "checks": checks}

    setup_logging(level="INFO", log_file=str(out_dir / "run.log"))
    logger.info(f"=== variant {variant} -> {out_dir} ===")
    started = time.time()
    checks: Dict[str, Any]
    panel_used = ""
    panel_abs: Optional[Path] = None
    if variant in DISCOVERY_VARIANTS:
        out = run_discovery_variant(variant, out_dir, fit, treatments,
                                    args.discovery_only, args.effects_only)
        checks = {k: out[k] for k in ("discovery", "effects") if out.get(k)}
        cfg_dict = out["config"]
        diff_rows = out["config_diff"]
        reused = "scripts/run_holdout_fit0017.py + main.py"
        env: Dict[str, str] = {}
    else:
        # Adjustment-set variants carry no discovery config at all: any
        # non-empty diff would mean a config path leaked into the run.
        out = run_adjust_variant(variant, out_dir, fit, args.smoke)
        checks = {"adjust": out["adjust"]}
        cfg_dict = None
        diff_rows = []
        assert_config_whitelist(variant, diff_rows, smoke=bool(args.smoke))
        reused = out["reused_script"]
        env = out["env"]
        panel_abs = Path(out["panel"])
        panel_used = rel(panel_abs)

    record = {
        "protocol": "HOLDOUT_MASKED_V1",
        "variant": variant,
        "kind": ("discovery + effects" if variant in DISCOVERY_VARIANTS
                 else "adjustment-set re-estimation, no discovery"),
        "description": DESCRIPTIONS[variant],
        "out_dir": rel(out_dir),
        "smoke": bool(args.smoke),
        "effects_only": bool(args.effects_only),
        "n_rows": int(len(fit)),
        "n_sids": int(fit["SID"].nunique()),
        "fit_window": {
            "fit_years": list(FIT_YEARS),
            "feature_boundary_inclusive": str(FIT_FEATURE_END),
            "label_boundary_exclusive": str(FIT_LABEL_END),
        },
        "reused_code": reused,
        "panel_used": panel_used,
        "env": env,
        "config_diff": diff_rows,
        "config_diff_note": (
            "no configuration change: the variant only perturbs the "
            "adjustment set recorded in prior/ate_results.csv"),
        "inputs": variant_inputs(variant),
        "logs": [rel(out_dir / "run.log")]
                 + ([rel(out_dir / "run.log.stdout")]
                    if (out_dir / "run.log.stdout").is_file() else []),
        "checks": checks,
        "runtime_seconds": round(time.time() - started, 1),
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "created_by": "scripts/run_holdout_fit0017_sensitivity.py",
        "python": sys.version.split()[0],
    }
    if variant in ADJUST_VARIANTS:
        record["added_covariates"] = checks["adjust"]["added_covariates"]
        record["baseline_pairs"] = checks["adjust"]["baseline_pairs"]
        record["discovery_rerun"] = False
    if args.smoke:
        covered = sorted(pd.to_numeric(
            fit["SID"].astype(str).str[:4]).unique().tolist())
        record["smoke_selection"] = {
            "method": "stratified_by_fit_year (>=1 storm per fit year, "
                      "remaining budget round-robin over years)",
            "requested_sids": int(args.smoke_sids),
            "selected_sids": int(fit["SID"].nunique()),
            "fit_years_covered": covered,
        }
    if panel_abs is not None and panel_used != rel(
            PRIOR_DIR / "fit_prior_panel.parquet"):
        record["inputs"][panel_used] = sha256(panel_abs)
    record["outputs"] = variant_outputs(out_dir)
    write_manifest(out_dir, record)
    write_readmes(root, record)
    logger.info(f"=== variant {variant} done ({record['runtime_seconds']} s) ===")
    return record


if __name__ == "__main__":
    sys.exit(main())
