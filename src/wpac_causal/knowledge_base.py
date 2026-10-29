# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""Build structured causal knowledge base YAML from all phase outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from wpac_causal.utils import Config, phase_timer


class KnowledgeBaseBuilder:
    """Serialize causal graph, ATE/CATE, and metadata into YAML."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.output_yaml = Path(config.get("knowledge_base.output_yaml"))
        self.schema_version = config.get("knowledge_base.schema_version", "wpac_dv24_v1.0")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_pcmci(self, path: Path) -> Dict[str, Any]:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_ate(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _load_cate(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _build_sample_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        ri_threshold = self.cfg.get("data.ri_threshold", 30)
        stats = {
            "total_observations": int(len(df)),
            "total_storms": int(df[self.cfg.get("best_track.id_col", "SID")].nunique()),
            "mean_dv24": float(df["DV24"].mean()),
            "std_dv24": float(df["DV24"].std()),
        }
        if "RI" in df.columns:
            # Best-track RI label from the RAW (pre-winsorize) DV24 — the
            # authoritative RI rate.
            stats["ri_rate"] = float(df["RI"].mean())
            stats["ri_rate_note"] = "from best-track RI label (raw DV24 >= threshold)"
        else:
            stats["ri_rate"] = float((df["DV24"] >= ri_threshold).mean())
            stats["ri_rate_note"] = "from winsorized DV24 (approximation)"
        return stats

    @staticmethod
    def _extract_edges(pcmci: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Machine-readable edge list for stage-2 (GNN prior) consumption.

        Each record keeps ``p_value`` for backward compatibility but the value
        stored by tigramite after ``run_pcmciplus(fdr_method=...)`` is the
        Benjamini-Hochberg **adjusted q-value**; it is therefore also exported
        as ``q_value`` with ``p_value_type="bh_adjusted_q"``.  ``link_type`` is
        the raw PCMCI+ orientation (all exported links are directed ``-->`` in
        the current priors; the field exists so a future orientation type is
        never silently dropped).
        """
        graph = pcmci["graph"]
        var_names = pcmci["var_names"]
        val_matrix = pcmci["val_matrix"]
        p_matrix = pcmci.get("p_matrix")
        tau_max = graph.shape[2] - 1
        edges = []
        for i, src in enumerate(var_names):
            for j, dst in enumerate(var_names):
                if i == j:
                    continue
                for tau in range(tau_max + 1):
                    link = graph[i, j, tau]
                    if link in ("-->", "->", "o->"):
                        coef = float(val_matrix[i, j, tau])
                        q = (float(p_matrix[i, j, tau])
                             if p_matrix is not None else None)
                        edges.append({
                            "src": src,
                            "dst": dst,
                            "lag": int(tau),
                            "coef": coef,
                            # p_value retained for compatibility; it IS the BH q
                            "p_value": q,
                            "q_value": q,
                            "p_value_type": "bh_adjusted_q",
                            "link_type": str(link),
                            "sign": "+" if coef > 0 else ("-" if coef < 0 else "0"),
                        })
        return edges

    def _build_graph_section(self, pcmci: Dict[str, Any]) -> Dict[str, Any]:
        target = self.cfg.get("causal_discovery.target", "DV24_YJ")
        edges = self._extract_edges(pcmci)
        edge_strings = [
            f"{e['src']} (lag {e['lag']}) -> {e['dst']} "
            f"[coef={e['coef']:.3f}, p={e['p_value']:.2e}]"
            for e in edges
        ]
        link_types: Dict[str, int] = {}
        for e in edges:
            link_types[e["link_type"]] = link_types.get(e["link_type"], 0) + 1
        return {
            "dot_path": str(self.cfg.get("causal_discovery.graph_dot")),
            "png_path": str(self.cfg.get("causal_discovery.graph_png")),
            "target_node": target,
            "edges": edges,                       # machine-readable (stage-2)
            "significant_edges": edge_strings,    # human-readable
            "n_edges": len(edges),
            "link_type_counts": link_types,
            "p_value_semantics": (
                "p_value/q_value are Benjamini-Hochberg adjusted q-values "
                "produced by tigramite run_pcmciplus(fdr_method=fdr_bh); "
                "p_value is kept as a compatibility alias for q_value"),
            "metadata": pcmci.get("metadata", {}),
        }

    def _build_ate_section(self, ate_df: pd.DataFrame) -> Dict[str, Any]:
        """Build ATE entries keyed by treatment, reporting DV24 and RI separately."""
        out = {}
        if ate_df.empty or "treatment" not in ate_df.columns:
            return out
        for trt in ate_df["treatment"].unique():
            sub = ate_df[ate_df["treatment"] == trt]
            entry: Dict[str, Any] = {
                "description": self._treatment_desc(trt),
                # ATE is per +1 fit-period z-score SD of the treatment
                "unit": self._treatment_unit(trt),
                "unit_note": ("treatments are z-scored with the fit-period "
                              "(2000-2017) scaler; effects are per +1 SD"),
                "ate_dv24_unit": "kt per fit-period SD",
                "ate_ri_unit": "probability per fit-period SD",
            }

            # DV24 (continuous target) — includes refutation
            dv24_row = sub[sub["outcome"] == "DV24"]
            if not dv24_row.empty:
                row = dv24_row.iloc[0]
                ref_passed = int(row["refutation_passed"]) if pd.notna(row.get("refutation_passed")) else 0
                ref_total = int(row["refutation_total"]) if pd.notna(row.get("refutation_total")) else 0
                entry["ate_dv24"] = float(row["ate"])
                entry["ci_95_dv24"] = [float(row.get("ci_low", np.nan)), float(row.get("ci_high", np.nan))]
                entry["p_value_dv24"] = float(row.get("p_value", np.nan))
                entry["refutation_passed"] = f"{ref_passed}/{ref_total}"
                # REAL per-test verdicts stored by effect_estimation
                # (FIX_TASKS P0-7; the previous aggregate-count mapping could
                # mislabel which test passed, D report §2.4.1).
                details = {}
                for test in ("placebo", "subset", "unobserved_confounder", "offshore"):
                    v = row.get(f"ref_{test}_passed")
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        details[test] = "skipped"
                    elif v is True or str(v) == "True":
                        details[test] = "passed"
                    else:
                        details[test] = "failed"
                entry["refutation_details"] = details

            # RI (binary target) — no refutation
            ri_row = sub[sub["outcome"] == "RI"]
            if not ri_row.empty:
                row = ri_row.iloc[0]
                entry["ate_ri"] = float(row["ate"])
                entry["ci_95_ri"] = [float(row.get("ci_low", np.nan)), float(row.get("ci_high", np.nan))]
                entry["p_value_ri"] = float(row.get("p_value", np.nan))

            if dv24_row.empty and ri_row.empty:
                continue
            out[trt] = entry
        return out

    def _build_cate_section(self, cate_df: pd.DataFrame) -> Dict[str, Any]:
        if cate_df.empty:
            return {"dimensions": self.cfg.get("effect_estimation.cate_dimensions", []), "lookup_table": {}}
        lookup = {}
        for _, row in cate_df.iterrows():
            key = f"{row['LC_STAGE']}_{row['LAT_BIN']}_{row['VMAX_BIN']}"
            key = key.replace("(", "").replace(")", "").replace(", ", "-").replace(" ", "_")
            if key not in lookup:
                lookup[key] = {"sample_size": int(row["n_obs"])}
            lookup[key][f"{row['treatment']}_ate"] = float(row["cate"])
        return {
            "dimensions": self.cfg.get("effect_estimation.cate_dimensions", []),
            "lookup_table": lookup,
        }

    def _build_feature_ranking(self, ate_df: pd.DataFrame, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rank by |ATE_DV24| * std(factor) / std(DV24). Uses DV24 outcome only."""
        if ate_df.empty or "DV24" not in df.columns:
            return []
        dv24_rows = ate_df[ate_df["outcome"] == "DV24"]
        if dv24_rows.empty:
            return []
        rankings = []
        dv24_std = df["DV24"].std()
        for _, row in dv24_rows.iterrows():
            trt = row["treatment"]
            if trt not in df.columns:
                continue
            ate = abs(float(row["ate"]))
            f_std = df[trt].std()
            score = (ate * f_std) / (dv24_std + 1e-9)
            rankings.append({
                "name": trt,
                "importance_score": round(float(score), 4),
                "rationale": self._treatment_rationale(trt),
            })
        rankings.sort(key=lambda x: x["importance_score"], reverse=True)
        return rankings[:10]

    @staticmethod
    def _treatment_desc(name: str) -> str:
        mapping = {
            "SHRD": "200-850hPa vertical wind shear",
            "RHLO": "850-700hPa mean relative humidity",
            "RHHI": "500-300hPa mean relative humidity",
            "VORT": "850hPa relative vorticity",
            "PV": "500hPa potential vorticity",
            "D200": "200hPa divergence",
            "WARM_CORE": "Warm core strength (200hPa T anomaly, IC minus OA)",
            "MFC": "Moisture flux convergence",
            "SLHF_IC": "Surface latent heat flux (inner core, ocean->atmosphere positive)",
            "W500": "500hPa vertical velocity",
            "OHC_IC": "Ocean heat content rel. 26C isotherm (inner core)",
            "OHC_OA": "Ocean heat content rel. 26C isotherm (outer area)",
        }
        return mapping.get(name, name)

    @staticmethod
    def _treatment_unit(name: str) -> str:
        """All treatments share the same effect scale: fit-period z-score SD.

        The raw physical units differ per variable, but the DoWhy estimator
        runs on the z-scored panel (fit-period scaler), so an ATE is the
        DV24 (kt) or RI (probability) change per +1 SD of the treatment.
        """
        return "fit-period z-score SD"

    @staticmethod
    def _treatment_rationale(name: str) -> str:
        mapping = {
            "SHRD": "Vertical wind shear disrupts vortex alignment",
            "RHLO": "Mid-level moisture fuels eyewall convection",
            "VORT": "Low-level vorticity seeds core rotation",
            "PV": "Mid-level PV tracks balanced vortex strength",
            "D200": "Upper-level divergence ventilates outflow",
            "WARM_CORE": "Warm core reflects ongoing convective organization",
            "MFC": "Moisture convergence feeds deep convection",
            "W500": "Mid-level ascent couples environment to core convection",
            "OHC_IC": "Subsurface ocean heat sustains flux against cold-wake feedback",
            "OHC_OA": "Environmental ocean heat content sets the thermodynamic reservoir",
        }
        return mapping.get(name, "Significant causal parent")

    # ------------------------------------------------------------------
    # Build YAML
    # ------------------------------------------------------------------

    def _build_mediation_section(self, med_df: pd.DataFrame) -> Dict[str, Any]:
        out = {}
        for _, row in med_df.iterrows():
            trt = row["treatment"]
            prop = row.get("mediation_proportion", np.nan)
            if prop > 0.7:
                tag = "primarily_quantitative"
            elif prop < 0.4:
                tag = "primarily_qualitative"
            else:
                tag = "mixed"
            out[trt] = {
                "path_x_mediator_coef": float(row.get("path_x_mediator_coef", np.nan)),
                "total_effect_ri": float(row.get("path_total_effect", np.nan)),
                "natural_direct_effect": float(row.get("path_direct_effect", np.nan)),
                "natural_indirect_effect": float(row.get("path_indirect_effect", np.nan)),
                "mediation_proportion": f"{prop:.1%}" if np.isfinite(prop) else "N/A",
                "mechanism_tag": tag,
            }
        return out

    def _build_regime_section(self, regime_df: pd.DataFrame) -> Dict[str, Any]:
        if regime_df.empty or "REGIME" not in regime_df.columns:
            return {}
        ri_specific = []
        regime_effects = {}
        baseline = self.cfg.get("effect_estimation.regime.comparison_baseline", "moderate")
        threshold = self.cfg.get("effect_estimation.regime.ri_specific_threshold", 3.0)

        for trt in regime_df["treatment"].unique():
            sub = regime_df[regime_df["treatment"] == trt]
            # Focus on DV24 outcome for regime comparison
            sub_dv24 = sub[sub["outcome"] == "DV24"]
            if sub_dv24.empty:
                continue
            regime_vals = {}
            for _, row in sub_dv24.iterrows():
                regime_vals[row["REGIME"]] = {
                    "cate": float(row["cate"]),
                    "n_obs": int(row["n_obs"]),
                }
            regime_effects[trt] = regime_vals

            # Check RI-specific: |CATE_RI| > threshold * |CATE_baseline|
            if baseline in regime_vals and "RI" in regime_vals:
                base_abs = abs(regime_vals[baseline]["cate"])
                ri_abs = abs(regime_vals["RI"]["cate"])
                if base_abs > 1e-9 and ri_abs / base_abs > threshold:
                    ri_specific.append(trt)

        return {
            "comparison_baseline": baseline,
            "ri_specific_threshold": threshold,
            "ri_specific_drivers": ri_specific,
            "regime_effects": regime_effects,
        }

    def build(
        self,
        df: pd.DataFrame,
        pcmci_path: Path,
        ate_path: Path,
        cate_path: Path,
        med_path: Optional[Path] = None,
        regime_cate_path: Optional[Path] = None,
    ) -> Path:
        with phase_timer("KnowledgeBase"):
            # Defensive: ensure Path objects (cfg.get may return str)
            pcmci_path = Path(pcmci_path)
            ate_path = Path(ate_path)
            cate_path = Path(cate_path)
            med_path = Path(med_path) if med_path else None
            regime_cate_path = Path(regime_cate_path) if regime_cate_path else None
            pcmci = self._load_pcmci(pcmci_path)
            ate_df = self._load_ate(ate_path)
            cate_df = self._load_cate(cate_path)
            med_df = self._load_ate(med_path) if med_path and med_path.exists() else pd.DataFrame()
            regime_df = self._load_cate(regime_cate_path) if regime_cate_path and regime_cate_path.exists() else pd.DataFrame()

            iso = pd.to_datetime(df["ISO_TIME"], utc=True)
            period_start, period_end = iso.min(), iso.max()
            db = {
                "version": self.schema_version,
                # actual panel range, not the configured download window
                "period": f"{period_start.year}-{period_end.year}",
                "period_start": str(period_start),
                "period_end": str(period_end),
                "target_variable": "DV24 (future 24h intensity change, kt)",
                "transformation": "yeo_johnson",
                "winsorize": "1%_99%",
                "sample_statistics": self._build_sample_stats(df),
                "causal_graph": self._build_graph_section(pcmci),
                "average_treatment_effects": self._build_ate_section(ate_df),
                "mediation_analysis": self._build_mediation_section(med_df),
                "conditional_average_treatment_effects": self._build_cate_section(cate_df),
                "regime_specific_effects": self._build_regime_section(regime_df),
                "feature_importance_ranking": self._build_feature_ranking(ate_df, df),
                "model_metadata": {
                    "pcmci_version": "tigramite>=5.0",
                    "dowhy_version": "dowhy>=0.9",
                    "tau_max": self.cfg.get("causal_discovery.tau_max"),
                    "pc_alpha": self.cfg.get("causal_discovery.pc_alpha"),
                    "fdr_method": self.cfg.get("causal_discovery.fdr_method"),
                    "refutation_seed": self.cfg.get("effect_estimation.refutation.random_seed", 42),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            }

            self.output_yaml.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_yaml, "w", encoding="utf-8") as f:
                yaml.dump(db, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            logger.info(f"Causal knowledge base saved to {self.output_yaml}")

            # Stage-2 export: machine-readable causal graph (adjacency edges
            # with lag, sign, coefficient, p/FDR value) for the GNN prior.
            edges_json = Path(self.cfg.get(
                "knowledge_base.graph_edges_json",
                "data/phase1/processed/causal_graph_edges.json",
            ))
            payload = {
                "var_names": pcmci["var_names"],
                "edges": self._extract_edges(pcmci),
                "metadata": pcmci.get("metadata", {}),
            }
            edges_json.parent.mkdir(parents=True, exist_ok=True)
            edges_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Machine-readable causal graph edges saved to {edges_json}")
        return self.output_yaml
