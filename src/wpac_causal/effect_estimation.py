# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""DoWhy ATE/CATE estimation for dual targets (DV24 continuous + RI binary) + mediation."""

from __future__ import annotations

import copy
import hashlib
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

from wpac_causal.identification_policy import (
    POLICY_ID,
    POLICY_RULE,
    IdentificationPolicyError,
    identify_adjustment_set,
    override_backdoor_variables,
)
from wpac_causal.utils import Config, phase_timer

try:
    from dowhy import CausalModel
except Exception:  # pragma: no cover
    CausalModel = None  # type: ignore


def normalize_text_lf(text: str) -> str:
    """Canonical text form for portable UTF-8/LF hashes."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_text_lf(text: str) -> str:
    return hashlib.sha256(normalize_text_lf(text).encode("utf-8")).hexdigest()


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values; NaN entries preserved."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def _e_value(ate: float, outcome_std: float, binary: bool,
             base_rate: float = 0.08017711710287025) -> float:
    """E-value (VanderWeele & Ding 2017): the minimum strength of association
    an unmeasured confounder would need with both treatment and outcome to
    explain away the observed effect.

    base_rate is the RI base rate of the fit panel (1014/12647); recompute it
    from the panel if that panel is rebuilt, since a stale rate changes RI
    E-values but nothing in training.

    Continuous outcome: standardized mean difference d -> RR ~ exp(0.91*|d|)
    (Chinn 2000 conversion), then E = RR + sqrt(RR*(RR-1)).
    Binary outcome: logit coefficient -> OR -> RR via
    RR = OR/(1-p0+p0*OR) with the observed base rate p0 (VanderWeele & Ding);
    for protective effects RR<1 the inverse is used in the same formula.
    Returns NaN for null/tiny effects.
    """
    try:
        if binary:
            or_val = float(np.exp(ate))
            rr = or_val / (1.0 - base_rate + base_rate * or_val)
            if rr < 1.0:
                rr = 1.0 / rr
        else:
            d = abs(ate) / (outcome_std + 1e-12)
            rr = float(np.exp(0.91 * d))
        if rr <= 1.0:
            return 1.0
        return float(rr + np.sqrt(rr * (rr - 1.0)))
    except Exception:
        return np.nan


def _e_value_ci_limit(ate: float, ci_low: float, ci_high: float,
                      outcome_std: float, binary: bool,
                      base_rate: float = 0.08017711710287025) -> float:
    """E-value at the nearer-to-null confidence limit; 1.0 when the CI crosses zero."""
    try:
        if (ate >= 0 and ci_low <= 0) or (ate < 0 and ci_high >= 0):
            return 1.0
        bound = ci_low if ate >= 0 else ci_high
        return _e_value(bound, outcome_std, binary, base_rate)
    except Exception:
        return np.nan


class EffectEstimator:
    """Estimate ATE/CATE for DV24 (continuous) and RI (binary) with mediation."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.treatments = config.get("effect_estimation.treatments", [])
        self.target_cont = config.get("effect_estimation.target_original", "DV24")
        self.target_bin = config.get("effect_estimation.target_binary", "RI")
        self.mediator = config.get("effect_estimation.mediator", "DV24")
        self.ate_method = config.get("effect_estimation.ate_method", "backdoor.linear_regression")
        self.ate_method_nonlinear = config.get("effect_estimation.ate_method_nonlinear", "backdoor.generalized_linear_model")
        self.cate_dims = config.get("effect_estimation.cate_dimensions", ["LC_STAGE", "LAT_BIN", "VMAX_BIN"])
        self.cate_bins = config.get("effect_estimation.cate_bins", {})
        self.ref_cfg = config.get("effect_estimation.refutation", {})

        self.ate_csv = Path(config.get("effect_estimation.ate_csv"))
        self.cate_csv = Path(config.get("effect_estimation.cate_csv"))
        self.refutation_md = Path(config.get("effect_estimation.refutation_md"))

        # Regime stratification config
        self.regime_cfg = config.get("effect_estimation.regime", {})
        self.use_regime = self.regime_cfg.get("enabled", True)
        # Stratification variable for regime CATE. MUST be observable at time t
        # (pre-treatment). Stratifying by the outcome DV24 itself conditions on
        # the outcome and truncates its distribution (review B-C2). Default PER =
        # past 6h intensity trend; 7.5 kt/6h == 30 kt/24h (ongoing-RI pace).
        self.regime_var = self.regime_cfg.get("stratify_by", "PER")
        self.regime_bins = self.regime_cfg.get("bins", [-float("inf"), 0.0, 7.5, float("inf")])
        self.regime_bins = [float("-inf") if str(b).lower() == "-inf" else float("inf") if str(b).lower() == "inf" else b for b in self.regime_bins]
        self.regime_labels = self.regime_cfg.get("labels", ["weakening", "moderate", "RI"])
        self.regime_baseline = self.regime_cfg.get("comparison_baseline", "moderate")
        self.ri_specific_threshold = self.regime_cfg.get("ri_specific_threshold", 3.0)

        # SHRD nonlinear (binned) analysis
        self.shrd_cfg = config.get("effect_estimation.shrd_nonlinear", {})
        self.use_shrd_bins = self.shrd_cfg.get("enabled", True)
        self.shrd_bins = self.shrd_cfg.get("bins", [0, 5, 15, float("inf")])
        self.shrd_bins = [float("inf") if str(b).lower() == "inf" else b for b in self.shrd_bins]
        self.shrd_labels = self.shrd_cfg.get("labels", ["low", "moderate", "high"])
        self.shrd_ref = self.shrd_cfg.get("reference", "moderate")

        # Refutation reproducibility (FIX_TASKS P0-7)
        self.ref_seed = int(self.ref_cfg.get("random_seed", 42))

        # Block (cluster) bootstrap inference by SID (FIX_TASKS P1-2):
        # overlapping 6h windows and within-storm autocorrelation make the
        # i.i.d. analytic p-values anti-conservative; bootstrap the storms.
        self.boot_cfg = config.get("effect_estimation.bootstrap", {})
        self.use_boot = bool(self.boot_cfg.get("enabled", True))
        self.boot_B = int(self.boot_cfg.get("B", 200))
        self.boot_seed = int(self.boot_cfg.get("seed", 42))

        # Headroom (PI - VMAX) effect-modification analysis (R2-1)
        self.hr_cfg = config.get("effect_estimation.headroom", {})
        self.use_headroom = bool(self.hr_cfg.get("enabled", False))
        self.hr_bins = self.hr_cfg.get("bins", [-float("inf"), 30, 60, float("inf")])
        self.hr_bins = [float("-inf") if str(b).lower() == "-inf" else float("inf") if str(b).lower() == "inf" else float(b) for b in self.hr_bins]
        self.hr_labels = self.hr_cfg.get("labels", ["low", "mid", "high"])
        self.hr_treatments = self.hr_cfg.get("treatments", ["SST_OA", "SST_IC", "OHC_IC", "RHLO", "SHRD"])

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_identification_dag(self, df: pd.DataFrame,
                                 pcmci_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build the identification DAG actually passed to DoWhy, with provenance.

        Returns ``{dot, nodes, final_edges, discovery_edges, cycle_break,
        hard_add}``.  ``build_graph_dot`` keeps returning just the DOT string.

        Provenance distinguishes:
          * ``discovery_edges`` — one row per directed (src, dst) pair from
            PCMCI+ (strongest lag kept; lag/coef/q recorded);
          * ``cycle_break`` — edges removed to make the graph acyclic;
          * ``hard_add`` — prior/structural edges added (or skipped because
            they would create a cycle).
        """
        graph = pcmci_results["graph"]
        var_names = pcmci_results["var_names"]
        val_matrix = pcmci_results.get("val_matrix")
        p_matrix = pcmci_results.get("p_matrix")
        tau_max = graph.shape[2] - 1

        # Map DV24_YJ -> DV24
        name_map = {v: v for v in var_names}
        if "DV24_YJ" in name_map and self.target_cont == "DV24":
            name_map["DV24_YJ"] = "DV24"

        # Collect directed edges; keep the strongest lag per (src, dst).
        edge_detail: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for i, src in enumerate(var_names):
            for j, dst in enumerate(var_names):
                if i == j:
                    continue
                for tau in range(tau_max + 1):
                    link = graph[i, j, tau]
                    if link not in ("-->", "->", "o->"):
                        continue
                    s, d = name_map.get(src, src), name_map.get(dst, dst)
                    if s == d:
                        continue
                    val = float(val_matrix[i, j, tau]) if val_matrix is not None else 1.0
                    new_val = abs(val)
                    key = (s, d)
                    if key not in edge_detail or new_val > edge_detail[key]["abs_val"]:
                        edge_detail[key] = {
                            "src": s, "dst": d, "lag": int(tau),
                            "coef": val, "abs_val": new_val,
                            "q_value": (float(p_matrix[i, j, tau])
                                        if p_matrix is not None else None),
                            "link_type": str(link),
                        }
        edges = set(edge_detail)

        try:
            import networkx as nx
        except ImportError:
            nx = None

        cycle_break: List[Dict[str, Any]] = []
        if nx is not None:
            G = nx.DiGraph()
            G.add_edges_from(sorted(edges))
            while True:
                try:
                    cycle = list(nx.find_cycle(G, orientation="original"))
                except nx.NetworkXNoCycle:
                    break
                weakest = min(cycle, key=lambda e: edge_detail.get(
                    (e[0], e[1]), {}).get("abs_val", 0.0))
                G.remove_edge(weakest[0], weakest[1])
                edges.discard((weakest[0], weakest[1]))
                cycle_break.append({
                    "src": weakest[0], "dst": weakest[1],
                    "abs_val": float(edge_detail.get(
                        (weakest[0], weakest[1]), {}).get("abs_val", 0.0)),
                })
                logger.info(f"Cycle-break: removed edge {weakest[0]} -> "
                            f"{weakest[1]} (|val|="
                            f"{edge_detail.get((weakest[0], weakest[1]), {}).get('abs_val', 0.0):.3f})")

        lines = ["digraph {"]
        nodes = set()
        for s, d in edges:
            nodes.add(s)
            nodes.add(d)
        for n in list(nodes) + self.treatments + [self.target_cont, self.target_bin, self.mediator]:
            if n and n in df.columns:
                nodes.add(n)
        # Sorted emission: set-iteration order is hash-dependent across
        # processes, which makes DoWhy's identification (and thus adjustment
        # sets and ATEs) non-reproducible between runs (found comparing the
        # 2026-07-21 reruns). Keep the DOT fully deterministic.
        for n in sorted(nodes):
            lines.append(f'    "{n}";')
        for s, d in sorted(edges):
            lines.append(f'    "{s}" -> "{d}";')

        # Hard-add mediation and structural edges (doc §5.3), with kinds.
        extra_edges: List[Tuple[str, str, str]] = []
        if self.target_bin in df.columns and self.mediator in df.columns:
            extra_edges.append((self.mediator, self.target_bin, "mediator->outcome"))
        for trt in self.treatments:
            if trt in df.columns and self.target_bin in df.columns:
                extra_edges.append((trt, self.target_bin, "treatment->outcome"))
        if "SST_OA" in df.columns and "W500" in df.columns:
            extra_edges.append(("SST_OA", "W500", "prior:SST_OA->W500"))
        if "W500" in df.columns and self.target_cont in df.columns:
            extra_edges.append(("W500", self.target_cont, "prior:W500->DV24"))
        if "W500" in df.columns and self.target_bin in df.columns:
            extra_edges.append(("W500", self.target_bin, "prior:W500->RI"))

        # Deduplicate by (src, dst), keeping the first kind, and drop edges
        # already present in the collapsed discovery graph.
        seen = set()
        extra_dedup = []
        for s, d, kind in extra_edges:
            if (s, d) in seen or (s, d) in edges:
                seen.add((s, d))
                continue
            seen.add((s, d))
            extra_dedup.append((s, d, kind))

        hard_added: List[Dict[str, Any]] = []
        hard_skipped: List[Dict[str, Any]] = []
        if nx is not None:
            G2 = nx.DiGraph()
            G2.add_edges_from(sorted(edges))
            for s, d, kind in extra_dedup:
                G2.add_edge(s, d)
                if not nx.is_directed_acyclic_graph(G2):
                    G2.remove_edge(s, d)
                    hard_skipped.append({"src": s, "dst": d, "kind": kind,
                                         "reason": "would_create_cycle"})
                    logger.info(f"Hard-added edge {s} -> {d} skipped (would create cycle)")
                else:
                    lines.append(f'    "{s}" -> "{d}";')
                    hard_added.append({"src": s, "dst": d, "kind": kind})
        else:
            for s, d, kind in extra_dedup:
                lines.append(f'    "{s}" -> "{d}";')
                hard_added.append({"src": s, "dst": d, "kind": kind})

        lines.append("}")
        final_edges = sorted(
            {(e[0], e[1]) for e in edges}
            | {(e["src"], e["dst"]) for e in hard_added})
        return {
            "dot": "\n".join(lines),
            "nodes": sorted(nodes),
            "final_edges": [{"src": s, "dst": d} for s, d in final_edges],
            "discovery_edges": [
                {k: v for k, v in edge_detail[key].items() if k != "abs_val"}
                for key in sorted(edge_detail)
            ],
            "cycle_break": cycle_break,
            "hard_add": {"added": hard_added, "skipped": hard_skipped},
        }

    def build_graph_dot(self, df: pd.DataFrame, pcmci_results: Dict[str, Any]) -> str:
        """Backward-compatible accessor for the identification DOT."""
        return self.build_identification_dag(df, pcmci_results)["dot"]

    def _write_identification_dag(self, dag: Dict[str, Any], df: pd.DataFrame,
                                  pcmci_results: Dict[str, Any]) -> None:
        """Persist the exact DOT used for ATE plus its provenance edge table."""
        import json

        dot_path = Path(self.cfg.get(
            "effect_estimation.identification_dot",
            self.ate_csv.parent / "identification_dag.dot"))
        json_path = Path(self.cfg.get(
            "effect_estimation.identification_json",
            self.ate_csv.parent / "identification_dag.json"))
        dot_path.parent.mkdir(parents=True, exist_ok=True)
        dot_text = normalize_text_lf(dag["dot"])
        dot_path.write_bytes(dot_text.encode("utf-8"))
        pcmci_path = Path(self.cfg.get("causal_discovery.results_pkl"))
        pcmci_sha = ""
        if pcmci_path.is_file():
            pcmci_sha = hashlib.sha256(pcmci_path.read_bytes()).hexdigest()
        payload = {
            "dot_sha256": sha256_text_lf(dot_text),
            "dot_path": str(dot_path),
            "n_nodes": len(dag["nodes"]),
            "n_final_edges": len(dag["final_edges"]),
            "final_edges": dag["final_edges"],
            "discovery_edges": dag["discovery_edges"],
            "cycle_break": dag["cycle_break"],
            "hard_add": dag["hard_add"],
            "q_value_note": ("discovery_edges[].q_value is the PCMCI+ "
                             "BH-adjusted q-value (tigramite p_matrix)"),
            "adjustment_policy": {
                "id": POLICY_ID,
                "rule": POLICY_RULE,
                "uses_ate_or_pvalue": False,
                "per_pair_audit": ("ate_results.csv columns "
                                   "identification_policy / "
                                   "identification_source / "
                                   "identification_candidates"),
            },
            "inputs": {
                "pcmci_results": str(pcmci_path),
                "pcmci_sha256": pcmci_sha,
                "n_rows": int(len(df)),
            },
            "config": {
                "treatments": list(self.treatments),
                "target_cont": self.target_cont,
                "target_bin": self.target_bin,
                "mediator": self.mediator,
                "fdr_method": self.cfg.get("causal_discovery.fdr_method"),
            },
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                             encoding="utf-8")
        logger.info(f"identification DAG saved: {dot_path} (sha256="
                    f"{payload['dot_sha256'][:16]}…) + {json_path}")

    # ------------------------------------------------------------------
    # Parents identification
    # ------------------------------------------------------------------

    def identify_parents(self, pcmci_results: Dict[str, Any]) -> List[str]:
        graph = pcmci_results["graph"]
        var_names = pcmci_results["var_names"]
        n_vars = len(var_names)
        tau_max = graph.shape[2] - 1
        target = self.cfg.get("causal_discovery.target", "DV24_YJ")
        if target not in var_names:
            logger.warning(f"Target {target} not in causal graph variables")
            return []
        t_idx = var_names.index(target)
        parents = []
        for i, name in enumerate(var_names):
            if i == t_idx:
                continue
            for tau in range(tau_max + 1):
                if graph[i, t_idx, tau] in ("-->", "->", "o->"):
                    parents.append(name)
                    break
        logger.info(f"Identified parents of {target}: {parents}")
        return parents

    def _compute_headroom(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Intensification room = max(0, PI - VMAX) in kt, from RAW SST_OA.

        The panel's SST_OA is z-scored; raw values are recovered with the
        saved feature scaler (PI proxy: 25*(SST-26) m/s, /0.5144 -> kt, the
        same construction as preprocessing.compute_pot).
        """
        from wpac_causal.utils import load_pickle
        if "SST_OA" not in df.columns or "VMAX" not in df.columns:
            return None
        scaler_path = Path(self.cfg.get("preprocessing.scaler_path", "data/phase1/processed/feature_scaler.pkl"))
        sst_raw = None
        if scaler_path.exists():
            try:
                scaler = load_pickle(scaler_path)
                names = list(getattr(scaler, "feature_names_in_", []))
                if "SST_OA" in names:
                    i = names.index("SST_OA")
                    sst_raw = df["SST_OA"].values * scaler.scale_[i] + scaler.mean_[i]
            except Exception as exc:
                logger.warning(f"Could not invert SST_OA z-score via scaler: {exc}")
        if sst_raw is None:
            logger.warning("Headroom unavailable (scaler missing SST_OA); skipping")
            return None
        pi_kt = np.maximum(0.0, (sst_raw - 26.0) * 25.0) / 0.5144
        return np.maximum(0.0, pi_kt - df["VMAX"].astype(float).values)

    def estimate_headroom_cate(self, df: pd.DataFrame, graph_dot: str) -> List[Dict[str, Any]]:
        """CATE of environmental treatments on DV24/RI stratified by headroom.

        Reported as "how environmental effects vary with intensification
        room" — NEVER as "deficit causes RI" (R2-1).
        """
        records: List[Dict[str, Any]] = []
        hr = self._compute_headroom(df)
        if hr is None:
            return records
        df = df.copy()
        df["HEADROOM"] = hr
        df["HEADROOM_BIN"] = pd.cut(df["HEADROOM"], bins=self.hr_bins,
                                    labels=self.hr_labels, include_lowest=True)
        for trt in self.hr_treatments:
            if trt not in df.columns:
                continue
            for outcome in (self.target_cont, self.target_bin):
                if outcome not in df.columns:
                    continue
                adjust = self._probe_adjustment(df, trt, outcome, graph_dot)
                # Lean dropna must include the ADJUSTMENT columns used by the
                # fit (R3-1): dropping only [treatment, outcome] lets NaNs in
                # the adjustment set crash the fit downstream (SPD has 368
                # NaNs, which silently killed all OHC_IC/SHRD cells).
                drop_cols = [trt, outcome] + [c for c in adjust if c in df.columns]
                for bin_label in df["HEADROOM_BIN"].dropna().unique():
                    sub = df[df["HEADROOM_BIN"] == bin_label].dropna(subset=drop_cols)
                    if len(sub) < 100:
                        logger.warning(
                            f"Headroom CATE skipped (low n={len(sub)}): "
                            f"{trt}->{outcome} [{bin_label}]"
                        )
                        continue
                    val = self._stratum_ate(sub, trt, outcome, adjust,
                                            context=f"{trt}->{outcome} [{bin_label}]")
                    if val is None:
                        continue
                    records.append({
                        "treatment": trt,
                        "outcome": outcome,
                        "headroom_regime": str(bin_label),
                        "cate": val,
                        "n_obs": len(sub),
                        "low_n_flag": bool(len(sub) < 200),
                        "adjustment_set": ",".join(adjust) if adjust else "(empty)",
                    })
        return records

    # ------------------------------------------------------------------
    # Positivity / common-support diagnostics (P1-3)
    # ------------------------------------------------------------------

    def positivity_diagnostics(self, df: pd.DataFrame, treatments: List[str]) -> pd.DataFrame:
        """Light-weight positivity / common-support check per treatment.

        For continuous treatments: report the empirical support (quantiles)
        in the estimation sample — regression adjustment extrapolates badly
        outside it.  Flags treatments with degenerate support (IQR ~ 0) or
        small n.  AIPW/DML as a replacement for single regression adjustment
        remains future work (B report §3); this table is the prerequisite
        diagnostic either way.
        """
        rows = []
        for trt in treatments:
            if trt not in df.columns:
                continue
            s = df[trt].dropna()
            if len(s) == 0:
                continue
            q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            iqr = q.loc[0.75] - q.loc[0.25]
            rows.append({
                "treatment": trt,
                "n": int(len(s)),
                "min": float(s.min()),
                "p1": float(q.loc[0.01]),
                "median": float(q.loc[0.5]),
                "p99": float(q.loc[0.99]),
                "max": float(s.max()),
                "iqr": float(iqr),
                "positivity_flag": "OK" if (iqr > 1e-8 and len(s) >= 100) else "CHECK",
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Block bootstrap inference (cluster by SID)
    # ------------------------------------------------------------------

    def _fit_ate_statsmodels(self, sub: pd.DataFrame, treatment: str,
                             outcome: str, adjust: List[str]) -> float:
        """Fast statsmodels ATE point estimate (OLS / Binomial GLM coefficient)."""
        adjust = [c for c in adjust if c in sub.columns and c != treatment]
        X = sm.add_constant(sub[[treatment] + adjust])
        y = sub[outcome]
        if y.nunique() <= 2:
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit(disp=0, maxiter=100)
        else:
            res = sm.OLS(y, X).fit()
        return float(res.params[treatment])

    def block_bootstrap_ate(self, df: pd.DataFrame, treatment: str, outcome: str,
                            adjust: List[str]) -> Dict[str, Any]:
        """Cluster bootstrap of the ATE by resampling whole storms (SIDs).

        Returns percentile CI and a two-sided sign-based p-value. This
        replaces i.i.d. analytic p-values, which are anti-conservative under
        overlapping 24h windows and within-storm autocorrelation
        (FIX_TASKS P1-2, reviews A-M4 / B-H5 / E §5).
        """
        sid_col = self.cfg.get("best_track.id_col", "SID")
        cols = [c for c in {sid_col, treatment, outcome, *adjust} if c in df.columns]
        sub = df[cols].dropna()
        if sid_col not in sub.columns or len(sub) < 50:
            return {}
        groups = {sid: g for sid, g in sub.groupby(sid_col)}
        sids = np.array(list(groups.keys()))
        if len(sids) < 5:
            return {}
        try:
            point = self._fit_ate_statsmodels(sub, treatment, outcome, adjust)
        except Exception as exc:
            logger.debug(f"Bootstrap point fit failed for {treatment}->{outcome}: {exc}")
            return {}
        rng = np.random.default_rng(self.boot_seed)
        ests = []
        for _ in range(self.boot_B):
            draw = rng.choice(sids, size=len(sids), replace=True)
            boot = pd.concat([groups[s] for s in draw], ignore_index=True)
            try:
                ests.append(self._fit_ate_statsmodels(boot, treatment, outcome, adjust))
            except Exception:
                continue  # skip degenerate resamples (e.g. perfect separation)
        if len(ests) < self.boot_B * 0.5:
            return {}
        ests = np.asarray(ests)
        ci_low, ci_high = np.percentile(ests, [2.5, 97.5])
        # Two-sided sign-based p-value around 0, with the 1/(B+1) resolution
        # floor (R2-7.4): never report an exact 0.
        p_val = float(2.0 * min((ests <= 0).mean(), (ests >= 0).mean()))
        p_val = max(p_val, 1.0 / (len(ests) + 1))
        return {
            "ate_boot": point,
            "ci_boot_low": float(ci_low),
            "ci_boot_high": float(ci_high),
            "p_value_boot": min(p_val, 1.0),
            "boot_B": int(len(ests)),
        }

    # ------------------------------------------------------------------
    # ATE estimation: single target
    # ------------------------------------------------------------------

    def estimate_ate_target(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        graph_dot: str,
    ) -> Dict[str, Any]:
        """Estimate ATE for a given treatment and outcome.

        DoWhy is used for IDENTIFICATION ONLY (backdoor adjustment set from the
        causal graph).  Estimation itself uses statsmodels directly:
        - continuous outcome -> OLS (equivalent to DoWhy backdoor.linear_regression)
        - binary outcome -> Binomial GLM logit (equivalent to the DoWhy GLM path,
          without the fragile fit_kwargs forwarding, review B-H1)

        DoWhy's `get_confidence_intervals` / `test_stat_significance` are NOT
        used: in DoWhy 0.14 they default to i.i.d. bootstrap resampling of the
        whole estimator (very slow with many covariates, and statistically
        inappropriate for autocorrelated panel data).  Uncertainty instead
        comes from (a) statsmodels analytic SE (reference) and (b) the SID
        block bootstrap (primary, FIX_TASKS P1-2).

        The adjustment set does NOT come from ``get_backdoor_variables()``:
        DoWhy 0.14 breaks ties between equally valid minimal backdoor sets in
        PYTHONHASHSEED-dependent order (see src/identification_policy.py), so
        both DV24 and RI go through the deterministic policy.  Its per-pair
        audit is returned as ``identification_*`` row fields.
        """
        if CausalModel is None:
            raise RuntimeError("dowhy not installed")
        if treatment not in df.columns or outcome not in df.columns:
            return {}
        sid_col = self.cfg.get("best_track.id_col", "SID")
        try:
            # Identification needs the FULL covariate set present: with only
            # [treatment, outcome] columns DoWhy cannot find any valid
            # backdoor set ("No such variable(s) found!") and silently yields
            # empty adjustments — i.e. unadjusted associations (found in the
            # 2026-07-20 full rerun).
            decision = identify_adjustment_set(
                df, treatment, outcome, graph_dot,
                forbidden=(treatment, outcome, sid_col))
        except IdentificationPolicyError:
            raise
        except Exception as exc:
            logger.warning(f"Identification failed for {treatment}->{outcome}: {exc}")
            return {}
        adjust = list(decision.selected)

        # Lean dropna on the variables actually used (review B-H4: dropping on
        # ALL columns needlessly shrinks n and makes samples inconsistent
        # across treatments).  The policy already rejected candidates with any
        # NaN, so an adjustment variable never changes n here.
        cols = [treatment, outcome] + adjust
        sub = df[cols].dropna()
        if len(sub) < 50:
            return {}

        is_binary = sub[outcome].nunique() <= 2
        method = self.ate_method_nonlinear if is_binary else self.ate_method
        ame = np.nan
        try:
            X = sm.add_constant(sub[[treatment] + adjust])
            if is_binary:
                fit = sm.GLM(sub[outcome], X, family=sm.families.Binomial()).fit(disp=0, maxiter=200)
                # Average marginal effect on the probability scale (B-H2 /
                # R2-7.3): the headline RI effect stays a logit coefficient,
                # but magnitude thresholds must be judged on AME, not log-odds.
                p_hat = fit.predict(X)
                ame = float((p_hat * (1 - p_hat)).mean() * fit.params[treatment])
            else:
                fit = sm.OLS(sub[outcome], X).fit()
            ate = float(fit.params[treatment])
            ci = fit.conf_int(alpha=0.05).loc[treatment]
            ci_low, ci_high = float(ci[0]), float(ci[1])
            p_value = float(fit.pvalues[treatment])
        except Exception as exc:
            logger.warning(f"ATE statsmodels fit failed for {treatment}->{outcome}: {exc}")
            return {}

        result = {
            "treatment": treatment,
            "outcome": outcome,
            "ate": ate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,          # analytic (i.i.d.) — reference only
            "n_obs": len(sub),
            "method": method,
            "adjustment_set": ",".join(adjust) if adjust else "(empty)",
        }
        result.update(decision.to_row_fields())
        if is_binary:
            result["ame"] = ame  # probability-scale average marginal effect
        # Block bootstrap by SID — the primary uncertainty measure (P1-2)
        if self.use_boot:
            boot = self.block_bootstrap_ate(df, treatment, outcome, adjust)
            result.update(boot)
        return result

    # ------------------------------------------------------------------
    # Mediation analysis (Baron-Kenny simplified via LPM)
    # ------------------------------------------------------------------

    def _mediation_baron_kenny(self, sub: pd.DataFrame, treatment: str, confounders: List[str]) -> Dict[str, Any]:
        """Baron-Kenny simplified using OLS/LPM. Robust for all sample sizes."""
        X = sm.add_constant(sub[[treatment] + [c for c in confounders if c in sub.columns]])
        # Path 1: X -> DV24 (OLS)
        model1 = sm.OLS(sub[self.mediator], X).fit()
        a = model1.params[treatment]

        # Path 2: X -> RI (LPM total effect)
        model2 = sm.OLS(sub[self.target_bin], X).fit()
        c = model2.params[treatment]

        # Path 3: X + DV24 -> RI (LPM direct effect)
        X3 = sm.add_constant(sub[[treatment, self.mediator] + [c for c in confounders if c in sub.columns]])
        model3 = sm.OLS(sub[self.target_bin], X3).fit()
        cp = model3.params[treatment]

        nie = c - cp
        nde = cp
        total = c
        prop = nie / total if abs(total) > 1e-9 else np.nan

        return {
            "treatment": treatment,
            "path_x_mediator_coef": float(a),
            "path_total_effect": float(total),
            "path_direct_effect": float(nde),
            "path_indirect_effect": float(nie),
            "mediation_proportion": float(prop),
            "n_obs": len(sub),
            "method": "baron_kenny_lpm",
        }

    def _mediation_dowhy(self, sub: pd.DataFrame, treatment: str, confounders: List[str], graph_dot: str) -> Dict[str, Any]:
        """DoWhy two-stage regression mediation. Preferred for large samples (n>=1000)."""
        if CausalModel is None:
            raise RuntimeError("dowhy not installed")

        model = CausalModel(
            data=sub,
            treatment=treatment,
            outcome=self.target_bin,
            graph=graph_dot,
            proceed_when_unidentifiable=True,
        )
        identified = model.identify_effect()

        # Natural direct effect
        nde_est = model.estimate_effect(
            identified,
            method_name="mediation.two_stage_regression",
            target_units="ate",
            test_significance=True,
            method_params={
                "first_stage_model": sm.OLS,
                "second_stage_model": sm.OLS,
                "mediation_identifier": {self.mediator: self.mediator},
            },
        )

        # Natural indirect effect
        nie_est = model.estimate_effect(
            identified,
            method_name="mediation.two_stage_regression",
            target_units="ate",
            test_significance=True,
            method_params={
                "first_stage_model": sm.OLS,
                "second_stage_model": sm.OLS,
                "mediation_identifier": {self.mediator: self.mediator},
                "estimand_type": "indirect_effect",
            },
        )

        total = float(getattr(nde_est, "value", 0)) + float(getattr(nie_est, "value", 0))
        nde = float(getattr(nde_est, "value", 0))
        nie = float(getattr(nie_est, "value", 0))
        prop = nie / total if abs(total) > 1e-9 else np.nan

        # Also get X->DV24 coefficient from first stage
        X = sm.add_constant(sub[[treatment] + [c for c in confounders if c in sub.columns]])
        model1 = sm.OLS(sub[self.mediator], X).fit()
        a = model1.params[treatment]

        return {
            "treatment": treatment,
            "path_x_mediator_coef": float(a),
            "path_total_effect": float(total),
            "path_direct_effect": float(nde),
            "path_indirect_effect": float(nie),
            "mediation_proportion": float(prop),
            "n_obs": len(sub),
            "method": "dowhy_two_stage",
        }

    def estimate_mediation(self, df: pd.DataFrame, treatment: str, confounders: List[str], graph_dot: str = "") -> Dict[str, Any]:
        """
        Mediation decomposition: NDE (direct) vs NIE (indirect via DV24).
        - n >= 1000: Try DoWhy two_stage_regression (preferred for large samples)
        - n < 1000 or DoWhy fails: Fallback to Baron-Kenny LPM
        """
        if treatment not in df.columns or self.mediator not in df.columns or self.target_bin not in df.columns:
            return {}
        cols = [treatment, self.mediator, self.target_bin] + [c for c in confounders if c in df.columns]
        sub = df[cols].dropna()
        if len(sub) < 50:
            return {}

        # Large sample: try DoWhy first
        if len(sub) >= 1000 and graph_dot and CausalModel is not None:
            try:
                result = self._mediation_dowhy(sub, treatment, confounders, graph_dot)
                logger.info(f"DoWhy mediation succeeded for {treatment} (n={len(sub)})")
                return result
            except Exception as exc:
                logger.warning(f"DoWhy mediation failed for {treatment}: {exc}. Falling back to Baron-Kenny.")

        # Fallback: Baron-Kenny LPM (works for all sample sizes)
        try:
            result = self._mediation_baron_kenny(sub, treatment, confounders)
            logger.info(f"Baron-Kenny mediation for {treatment} (n={len(sub)})")
            return result
        except Exception as exc:
            logger.warning(f"Mediation failed for {treatment}: {exc}")
            return {}

    def _placebo_permute_sm(self, sub: pd.DataFrame, treatment: str, outcome: str,
                            adjust: List[str], original_ate: float,
                            n_sim: int = 20) -> Tuple[Dict[str, Any], bool]:
        """Statsmodels-native placebo-treatment refutation (permute treatment).

        Used when DoWhy's refuters cannot be constructed (empty backdoor set,
        R3-2). Same pass rule as the DoWhy path: |placebo| < 0.30*|original|.
        """
        rng = np.random.default_rng(self.ref_seed)
        sims = []
        for _ in range(n_sim):
            perm = sub.copy()
            perm[treatment] = rng.permutation(perm[treatment].values)
            try:
                sims.append(self._fit_ate_statsmodels(perm, treatment, outcome, adjust))
            except Exception:
                continue
        placebo_ate = float(np.mean(sims)) if sims else np.nan
        if not np.isfinite(placebo_ate):
            return {"passed": None, "skipped": True, "note": "all placebo fits failed"}, False
        ratio = abs(placebo_ate / original_ate) if abs(original_ate) > 1e-4 else abs(placebo_ate)
        ok = bool(ratio < self.ref_cfg.get("placebo", {}).get("pass_threshold", 0.30))
        return {"placebo_ate": placebo_ate, "ratio": ratio, "passed": ok}, ok

    def _data_subset_sm(self, sub: pd.DataFrame, treatment: str, outcome: str,
                        adjust: List[str], original_ate: float) -> Tuple[Dict[str, Any], bool]:
        """Statsmodels-native data-subset refutation (R3-2)."""
        rng = np.random.default_rng(self.ref_seed)
        frac = self.ref_cfg.get("subset", {}).get("fraction", 0.9)
        n_runs = int(self.ref_cfg.get("subset", {}).get("num_runs", 3))
        ests = []
        for _ in range(n_runs):
            boot = sub.sample(frac=frac, replace=False, random_state=int(rng.integers(0, 10 ** 9)))
            try:
                ests.append(self._fit_ate_statsmodels(boot, treatment, outcome, adjust))
            except Exception:
                continue
        subset_ate = float(np.mean(ests)) if ests else np.nan
        if not np.isfinite(subset_ate):
            return {"passed": None, "skipped": True, "note": "all subset fits failed"}, False
        diff = abs(subset_ate - original_ate) / abs(original_ate) if abs(original_ate) > 1e-4 else abs(subset_ate - original_ate)
        ok = bool(diff < self.ref_cfg.get("subset", {}).get("pass_threshold", 0.20))
        return {"subset_ate": subset_ate, "diff": diff, "passed": ok}, ok

    # ------------------------------------------------------------------
    # Refutation tests
    # ------------------------------------------------------------------

    def run_refutations(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        graph_dot: str,
        ate_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Refute an ATE estimate on *outcome* (DV24 or RI).

        All refuters re-estimate with LPM (``self.ate_method``) for speed.
        The reference effect is therefore also taken from the SAME LPM
        estimate (``estimate.value``), NOT from ``ate_result["ate"]``: for the
        binary outcome RI the headline ATE comes from a GLM logit (log-odds
        scale), and comparing probability-scale refutations against a
        log-odds reference mixes units (review B-C3).  For the continuous
        outcome the two are identical anyway.

        Reproducibility: every stochastic refuter receives ``self.ref_seed``
        (FIX_TASKS P0-7); the seed is recorded in the returned dict and in
        the refutation report.
        """
        if CausalModel is None or not ate_result:
            return {}
        # Lean dropna on the variables in the graph's adjustment set (same
        # sample as the headline ATE, review B-H4). Identification uses the
        # full covariate set; estimation/refutation uses the lean sample.
        # Same deterministic policy as the headline ATE (src/identification_policy):
        # DoWhy's own tie-break is PYTHONHASHSEED-dependent and would make the
        # refutation verdicts (hence trusted signs) irreproducible.
        _adjust = self._probe_adjustment(df, treatment, outcome, graph_dot)
        sub = df[[treatment, outcome] + _adjust].dropna()
        method = self.ate_method  # LPM for refutation (scale-matched, see docstring)

        model = CausalModel(data=sub, treatment=treatment, outcome=outcome,
                            graph=graph_dot, proceed_when_unidentifiable=True)
        identified = model.identify_effect()
        # Refuters read the estimand's backdoor set; pin it to the policy set.
        override_backdoor_variables(identified, _adjust)
        results = {"refutation_seed": self.ref_seed, "reference_method": method}
        passed = 0
        n_tests = 0

        try:
            estimate = model.estimate_effect(identified, method_name=method, target_units="ate")
        except Exception as exc:
            logger.warning(f"DoWhy LPM estimate raised for {treatment}->{outcome}: {exc}")
            estimate = None

        if estimate is None or getattr(estimate, "value", None) is None:
            # DoWhy could not produce a valid estimate on this sample/graph.
            # The three stochastic refuters REQUIRE a valid DoWhy estimate
            # object, so they are marked SKIPPED (not FAIL) — an engineering
            # limitation, not a robustness failure (FIX_TASKS_R2 R2-3).
            # Offshore below still runs via statsmodels.
            logger.warning(
                f"DoWhy LPM estimate unavailable for {treatment}->{outcome}; "
                "stochastic refuters marked SKIPPED"
            )
            try:
                original_ate = self._fit_ate_statsmodels(sub, treatment, outcome, _adjust)
            except Exception:
                original_ate = None
            results["reference_ate_lpm"] = original_ate
            for _test in ("placebo", "subset", "unobserved_confounder"):
                results[_test] = {"passed": None, "skipped": True,
                                  "note": "DoWhy estimate unavailable (None)"}
        elif not _adjust:
            # R3-2: DoWhy 0.14 cannot construct its refuters for an EMPTY
            # backdoor set — CausalRefuter.get_adjustment_set() crashes on
            # backdoor_variables=None ('NoneType' object is not subscriptable),
            # and its LPM estimate is degenerate (literal 0).  Use the
            # statsmodels-native placebo/subset for REAL results and mark the
            # unobserved-confounder refuter SKIPPED (DoWhy-only test).
            logger.info(
                f"Empty backdoor set for {treatment}->{outcome}; using "
                "statsmodels-native placebo/subset refutations"
            )
            original_ate = self._fit_ate_statsmodels(sub, treatment, outcome, _adjust)
            results["reference_ate_lpm"] = original_ate
            if original_ate is None:
                for _test in ("placebo", "subset", "unobserved_confounder"):
                    results[_test] = {"passed": None, "skipped": True,
                                      "note": "reference statsmodels fit failed"}
            else:
                results["placebo"], _ok = self._placebo_permute_sm(
                    sub, treatment, outcome, _adjust, original_ate)
                passed += int(bool(_ok))
                n_tests += 1
                results["subset"], _ok = self._data_subset_sm(
                    sub, treatment, outcome, _adjust, original_ate)
                passed += int(bool(_ok))
                n_tests += 1
                results["unobserved_confounder"] = {
                    "passed": None, "skipped": True,
                    "note": "DoWhy UC refuter unsupported for empty backdoor set (R3-2)"}
        else:
            original_ate = float(estimate.value)  # LPM scale, same as refuters
            results["reference_ate_lpm"] = original_ate

            # 1. Placebo (num_simulations reduced from default 100 to 20 for speed)
            try:
                placebo = model.refute_estimate(identified, estimate, method_name="placebo_treatment_refuter",
                                                placebo_type="permute", num_simulations=20, n_jobs=-1,
                                                random_state=self.ref_seed)
                placebo_ate = float(getattr(placebo, "new_effect", 0))
                # Use absolute difference when original ATE is near zero (avoids division blowup)
                if abs(original_ate) > 1e-4:
                    ratio = abs(placebo_ate / original_ate)
                else:
                    ratio = abs(placebo_ate)  # both near zero → ratio = |placebo|
                ok = ratio < self.ref_cfg.get("placebo", {}).get("pass_threshold", 0.30)
                results["placebo"] = {"placebo_ate": placebo_ate, "ratio": ratio, "passed": ok}
                passed += int(ok)
                n_tests += 1
            except Exception as exc:
                results["placebo"] = {"passed": False, "error": str(exc)}
                n_tests += 1

            # 2. Data Subset (single call with num_simulations instead of manual loop)
            try:
                subset_ref = model.refute_estimate(identified, estimate, method_name="data_subset_refuter",
                                                    subset_fraction=self.ref_cfg.get("subset", {}).get("fraction", 0.9),
                                                    num_simulations=self.ref_cfg.get("subset", {}).get("num_runs", 3),
                                                    n_jobs=-1, random_state=self.ref_seed)
                subset_ate = float(getattr(subset_ref, "new_effect", 0))
                # Use absolute difference when original ATE is near zero
                if abs(original_ate) > 1e-4:
                    diff = abs(subset_ate - original_ate) / abs(original_ate)
                else:
                    diff = abs(subset_ate - original_ate)  # absolute diff, no normalization
                ok = diff < self.ref_cfg.get("subset", {}).get("pass_threshold", 0.20)
                results["subset"] = {"subset_ate": subset_ate, "diff": diff, "passed": ok}
                passed += int(ok)
                n_tests += 1
            except Exception as exc:
                results["subset"] = {"passed": False, "error": str(exc)}
                n_tests += 1

            # 3. Unobserved Confounder (reduce num_simulations)
            try:
                uc = model.refute_estimate(identified, estimate, method_name="add_unobserved_common_cause",
                                           effect_strength_on_treatment=self.ref_cfg.get("unobserved_confounder", {}).get("effect_strength_on_treatment", 0.3),
                                           effect_strength_on_outcome=self.ref_cfg.get("unobserved_confounder", {}).get("effect_strength_on_outcome", 0.3),
                                           num_simulations=20, n_jobs=-1,
                                           shuffle_random_seed=self.ref_seed)
                uc_ate = float(getattr(uc, "new_effect", 0))
                if abs(original_ate) > 1e-4:
                    diff = abs(uc_ate - original_ate) / abs(original_ate)
                else:
                    diff = abs(uc_ate - original_ate)
                ok = diff < self.ref_cfg.get("unobserved_confounder", {}).get("pass_threshold", 0.25)
                results["unobserved_confounder"] = {"uc_ate": uc_ate, "diff": diff, "passed": ok}
                passed += int(ok)
                n_tests += 1
            except Exception as exc:
                results["unobserved_confounder"] = {"passed": False, "error": str(exc)}
                n_tests += 1

        # 4. Offshore
        try:
            if original_ate is None:
                results["offshore"] = {"passed": None, "skipped": True,
                                       "note": "reference estimate unavailable"}
            elif "LAND_DIST" in df.columns:
                # Same adjustment set as the headline estimate — the previous
                # ">80% non-NaN columns" filter could drop graph nodes and make
                # DoWhy return a None estimate.
                keep = ["LAND_DIST", treatment, outcome] + [c for c in _adjust if c != "LAND_DIST"]
                offshore_df = df[[c for c in keep if c in df.columns]].copy()
                offshore_df = offshore_df[offshore_df["LAND_DIST"] > self.ref_cfg.get("offshore", {}).get("land_dist_threshold", 100)]
                offshore_df = offshore_df.replace([np.inf, -np.inf], np.nan).dropna()
                if len(offshore_df) > 50:
                    off_ate = self._fit_ate_statsmodels(offshore_df, treatment, outcome, _adjust)
                    if abs(original_ate) > 1e-4:
                        diff = abs(off_ate - original_ate) / abs(original_ate)
                    else:
                        diff = abs(off_ate - original_ate)
                    ok = diff < self.ref_cfg.get("offshore", {}).get("pass_threshold", 0.20)
                    results["offshore"] = {"offshore_ate": off_ate, "diff": diff, "passed": ok}
                    passed += int(ok)
                    n_tests += 1
                else:
                    # Skipped tests are NOT counted as passed (review B-M1)
                    results["offshore"] = {"passed": None, "skipped": True, "note": "too few offshore samples"}
            else:
                results["offshore"] = {"passed": None, "skipped": True, "note": "LAND_DIST unavailable"}
        except Exception as exc:
            results["offshore"] = {"passed": False, "error": str(exc)}
            n_tests += 1

        results["total_passed"] = passed
        results["total_tests"] = n_tests
        return results

    # ------------------------------------------------------------------
    # CATE (manual stratification)
    # ------------------------------------------------------------------

    def _probe_adjustment_decision(self, df: pd.DataFrame, treatment: str,
                                   outcome: str, graph_dot: str):
        """Deterministic policy decision for (treatment, outcome).

        Shared by the headline ATE, refutations, CATE and mediation so every
        downstream number uses one adjustment set per pair (see
        src/identification_policy).  Fail-closed policy errors propagate; a
        DoWhy crash logs and yields ``None``.  The FULL covariate set must be
        present in df — with only [treatment, outcome] columns DoWhy finds no
        valid set at all."""
        if treatment not in df.columns or outcome not in df.columns:
            raise IdentificationPolicyError(
                f"{treatment}->{outcome}: treatment or outcome missing from the panel")
        if CausalModel is None:
            raise RuntimeError("dowhy not installed")
        sid_col = self.cfg.get("best_track.id_col", "SID")
        try:
            return identify_adjustment_set(
                df, treatment, outcome, graph_dot,
                forbidden=(treatment, outcome, sid_col))
        except IdentificationPolicyError:
            raise
        except Exception as exc:
            logger.warning(f"probe_adjustment failed for {treatment}->{outcome}: {exc}")
            return None

    def _probe_adjustment(self, df: pd.DataFrame, treatment: str, outcome: str,
                          graph_dot: str) -> List[str]:
        """Backdoor adjustment set for (treatment, outcome) via the
        deterministic policy (never DoWhy's hash-dependent default pick)."""
        decision = self._probe_adjustment_decision(
            df, treatment, outcome, graph_dot)
        return [] if decision is None else list(decision.selected)

    def _stratum_ate(self, sub: pd.DataFrame, treatment: str, outcome: str,
                     adjust: List[str], context: str = "") -> Optional[float]:
        """ATE within one stratum via OLS (or Binomial GLM for binary outcome).

        Failures are logged (never silently swallowed, R3-1)."""
        if len(sub) < 30 or sub[treatment].nunique() < 2:
            logger.warning(f"_stratum_ate skipped (degenerate stratum): {context or f'{treatment}->{outcome}'}")
            return None
        try:
            adj = [c for c in adjust if c in sub.columns and sub[c].nunique() > 1]
            X = sm.add_constant(sub[[treatment] + adj])
            if sub[outcome].nunique() <= 2:
                fit = sm.GLM(sub[outcome], X, family=sm.families.Binomial()).fit(disp=0, maxiter=200)
            else:
                fit = sm.OLS(sub[outcome], X).fit()
            return float(fit.params[treatment])
        except Exception as exc:
            logger.warning(f"_stratum_ate fit failed for {context or f'{treatment}->{outcome}'}: {exc}")
            return None

    def estimate_cate(self, df: pd.DataFrame, treatment: str, outcome: str, graph_dot: str) -> List[Dict[str, Any]]:
        records = []
        if treatment not in df.columns or outcome not in df.columns:
            return records
        if "LC_STAGE" not in df.columns:
            return records
        df = df.copy()
        lat_col = self.cfg.get("best_track.lat_col", "LAT")
        vmax_col = "VMAX"
        if lat_col not in df.columns:
            lat_col = "LAT"  # fallback to env column name
        lat_bins = self.cate_bins.get("lat", [0, 15, 25, 45])
        vmax_bins = self.cate_bins.get("vmax", [0, 25, 65, 100])
        df["LAT_BIN"] = pd.cut(df[lat_col], bins=lat_bins, include_lowest=True)
        df["VMAX_BIN"] = pd.cut(df[vmax_col], bins=vmax_bins, include_lowest=True)
        adjust = self._probe_adjustment(df, treatment, outcome, graph_dot)

        for lc in df["LC_STAGE"].dropna().unique():
            for lat_bin in df["LAT_BIN"].dropna().unique():
                for vmax_bin in df["VMAX_BIN"].dropna().unique():
                    sub = df[(df["LC_STAGE"] == lc) & (df["LAT_BIN"] == lat_bin) & (df["VMAX_BIN"] == vmax_bin)]
                    sub = sub.dropna(subset=[treatment, outcome])
                    if len(sub) < 30:
                        continue
                    val = self._stratum_ate(sub, treatment, outcome, adjust)
                    if val is None:
                        continue
                    records.append({
                        "treatment": treatment,
                        "outcome": outcome,
                        "LC_STAGE": lc,
                        "LAT_BIN": str(lat_bin),
                        "VMAX_BIN": str(vmax_bin),
                        "cate": val,
                        "n_obs": len(sub),
                    })
        return records

    def estimate_regime_cate(self, df: pd.DataFrame, treatment: str, outcome: str, graph_dot: str) -> List[Dict[str, Any]]:
        """Estimate CATE stratified by the PRE-TREATMENT regime variable."""
        records = []
        if treatment not in df.columns or outcome not in df.columns:
            return records
        if "REGIME" not in df.columns:
            return records
        adjust = self._probe_adjustment(df, treatment, outcome, graph_dot)

        for regime in df["REGIME"].dropna().unique():
            sub = df[df["REGIME"] == regime].dropna(subset=[treatment, outcome])
            if len(sub) < 30:
                continue
            val = self._stratum_ate(sub, treatment, outcome, adjust)
            if val is None:
                continue
            records.append({
                "treatment": treatment,
                "outcome": outcome,
                "REGIME": str(regime),
                "cate": val,
                "n_obs": len(sub),
            })
        return records

    def estimate_shrd_binned_ate(self, df: pd.DataFrame, graph_dot: str) -> List[Dict[str, Any]]:
        """
        Estimate SHRD bin contrasts with nine prespecified covariates.

        Uses SHRD_RAW (pre-scaling original values in kt) if available, because
        the bin thresholds [0, 5, 15, inf] are defined in kt.
        """
        shrd_col = "SHRD_RAW" if "SHRD_RAW" in df.columns else "SHRD"
        if shrd_col not in df.columns or not self.use_shrd_bins:
            return []
        df = df.copy()
        df["SHRD_BIN"] = pd.cut(
            df[shrd_col],
            bins=self.shrd_bins,
            labels=self.shrd_labels,
            include_lowest=True,
            right=False,
        )
        ref_label = self.shrd_ref
        records = []
        for outcome in [self.target_cont, self.target_bin]:
            for label in self.shrd_labels:
                if label == ref_label:
                    continue
                sub = df[df["SHRD_BIN"].isin([label, ref_label])].copy()
                if len(sub) < 30:
                    continue
                sub["SHRD_BIN_TREAT"] = (sub["SHRD_BIN"] == label).astype(int)
                confounders = [
                    c for c in ("VMAX", "PER", "SPD", "LAT", "LAND_DIST",
                                "SST_OA", "SST_IC", "RHLO", "W500")
                    if c in sub.columns
                ]
                keep_cols = ["SHRD_BIN_TREAT", outcome] + confounders
                sub_clean = sub[keep_cols].dropna()
                if len(sub_clean) < 30:
                    continue
                try:
                    X = sm.add_constant(sub_clean[["SHRD_BIN_TREAT"] + confounders])
                    if outcome == self.target_bin:
                        # Binary outcome: use Logit
                        try:
                            glm_model = sm.GLM(sub_clean[outcome], X, family=sm.families.Binomial())
                            glm_result = glm_model.fit(maxiter=200, disp=0)
                            ate_val = float(glm_result.params["SHRD_BIN_TREAT"])
                        except Exception:
                            # Fallback to LPM if GLM fails
                            ols_model = sm.OLS(sub_clean[outcome], X).fit()
                            ate_val = float(ols_model.params["SHRD_BIN_TREAT"])
                    else:
                        ols_model = sm.OLS(sub_clean[outcome], X).fit()
                        ate_val = float(ols_model.params["SHRD_BIN_TREAT"])
                    records.append({
                        "treatment": f"SHRD_{label}_vs_{ref_label}",
                        "outcome": outcome,
                        "ate": ate_val,
                        "n_obs": len(sub_clean),
                    })
                except Exception as exc:
                    logger.debug(f"SHRD bin ATE failed for {label} vs {ref_label}: {exc}")
        return records

    def mediation_confounders(self, df: pd.DataFrame, treatments: List[str]) -> List[str]:
        """Numeric confounders for the mediation regressions.

        VMAX_F24 (future 24h intensity) and DV24_YJ (deterministic transform
        of the mediator/outcome) must NEVER enter the adjustment set —
        temporal leakage + controlling an outcome descendant (review B-C1 /
        FIX_TASKS P0-3).
        """
        excluded = treatments + [self.target_cont, self.target_bin, self.mediator,
                                 "VMAX_F24", "DV24_YJ",
                                 "SID", "ISO_TIME", "NAME", "LAT_BIN", "VMAX_BIN", "REGIME"]
        return [
            c for c in df.columns
            if c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    # ------------------------------------------------------------------
    # Regime assignment (pre-treatment stratification)
    # ------------------------------------------------------------------

    def assign_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign REGIME labels from a PRE-TREATMENT variable (default PER).

        Stratifying by the outcome DV24 itself is conditioning on the outcome
        (truncation bias, review B-C2), so the stratifier must be observable
        at time t. Default PER = past-6h intensity trend; bins [0, 7.5] kt/6h
        correspond to steady / intensifying / ongoing-RI pace (30 kt/24h).
        """
        df = df.copy()
        if self.use_regime and self.regime_var in df.columns:
            df["REGIME"] = pd.cut(
                df[self.regime_var],
                bins=self.regime_bins,
                labels=self.regime_labels,
                include_lowest=True,
            )
        else:
            df["REGIME"] = np.nan
        return df

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, pcmci_results: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        with phase_timer("EffectEstimation"):
            dag = self.build_identification_dag(df, pcmci_results)
            graph_dot = dag["dot"]
            self._write_identification_dag(dag, df, pcmci_results)
            parents = self.identify_parents(pcmci_results)
            treatments_in_graph = [t for t in self.treatments if t in parents]
            # Always include all configured treatments — the causal graph from PCMCI+
            # identifies direct parents, but DoWhy's backdoor adjustment can estimate
            # effects for non-parents too (controlling for confounders).
            treatments = [t for t in self.treatments if t in df.columns]
            logger.info(f"Parents in graph: {parents}")
            logger.info(f"Treatments for effect estimation: {treatments}")

            # Regime labels from a PRE-TREATMENT variable (see assign_regime;
            # stratifying by the outcome DV24 itself is invalid, review B-C2).
            df = self.assign_regime(df)

            ate_rows = []
            qualitative_rows = []
            mediation_rows = []
            cate_rows = []
            regime_rows = []
            refutation_report = ["# Refutation Report\n"]
            refutation_report.append(
                f"\nRandom seed for all stochastic refuters: **{self.ref_seed}** "
                "(placebo random_state, subset random_state, unobserved-confounder "
                "shuffle_random_seed). Reference effects are LPM-scale on both "
                "outcomes (see run_refutations docstring).\n"
            )

            def _flatten_ref(ate_row: Dict[str, Any], ref: Dict[str, Any]) -> None:
                """Store per-test refutation booleans as flat columns so the KB
                can record REAL per-test verdicts (FIX_TASKS P0-7)."""
                ate_row["refutation_passed"] = ref.get("total_passed", 0)
                ate_row["refutation_total"] = ref.get("total_tests", 0)
                for test in ("placebo", "subset", "unobserved_confounder", "offshore"):
                    ate_row[f"ref_{test}_passed"] = ref.get(test, {}).get("passed")

            # Thresholds for qualitative trigger detection (effect magnitude in absolute value)
            qual_threshold_dv24 = self.cfg.get("effect_estimation.qualitative.threshold_dv24", 2.0)
            qual_threshold_ri = self.cfg.get("effect_estimation.qualitative.threshold_ri", 0.02)

            # Process treatments sequentially. Each refute_estimate call uses
            # joblib n_jobs=-1 internally for parallelism on simulations.
            for trt in treatments:
                logger.info(f"Estimating effects for {trt}")
                ref_dv24: Dict[str, Any] = {}
                ref_ri: Dict[str, Any] = {}

                ate_dv24 = self.estimate_ate_target(df, trt, self.target_cont, graph_dot)
                if ate_dv24:
                    ref_dv24 = self.run_refutations(df, trt, self.target_cont, graph_dot, ate_dv24)
                    _flatten_ref(ate_dv24, ref_dv24)
                    ate_rows.append(ate_dv24)

                ate_ri = self.estimate_ate_target(df, trt, self.target_bin, graph_dot)
                if ate_ri:
                    ref_ri = self.run_refutations(df, trt, self.target_bin, graph_dot, ate_ri)
                    ate_ri["outcome"] = self.target_bin
                    _flatten_ref(ate_ri, ref_ri)
                    ate_rows.append(ate_ri)

                def _sig(res, thr):
                    # Magnitude on the probability scale for binary outcomes
                    # (R2-7.3 / B-H2): use the stored AME, fall back to the
                    # raw coefficient; p prefers block bootstrap.
                    mag = res.get("ame", np.nan)
                    if mag is None or (isinstance(mag, float) and np.isnan(mag)):
                        mag = res.get("ate", 0)
                    p = res.get("p_value_boot", np.nan)
                    if p is None or (isinstance(p, float) and np.isnan(p)):
                        p = res.get("p_value", np.nan)
                    return abs(mag) > thr or (not np.isnan(p) and p < 0.05)

                if ate_dv24 and ate_ri:
                    dv24_sig = _sig(ate_dv24, qual_threshold_dv24)
                    ri_sig = _sig(ate_ri, qual_threshold_ri)
                    if not dv24_sig and ri_sig:
                        factor_type = "qualitative_trigger"
                    elif dv24_sig and ri_sig:
                        factor_type = "mixed"
                    elif dv24_sig and not ri_sig:
                        factor_type = "quantitative_only"
                    else:
                        factor_type = "none"
                    qualitative_rows.append({
                        "treatment": trt,
                        "dv24_ate": ate_dv24.get("ate"),
                        "dv24_pvalue": ate_dv24.get("p_value"),
                        "ri_ate": ate_ri.get("ate"),
                        "ri_pvalue": ate_ri.get("p_value"),
                        "factor_type": factor_type,
                        "n_obs_dv24": ate_dv24.get("n_obs"),
                        "n_obs_ri": ate_ri.get("n_obs"),
                    })
                    logger.info(
                        f"[{trt}] Factor type: {factor_type} "
                        f"(DV24_ATE={ate_dv24.get('ate', 0):.3f}, DV24_p={ate_dv24.get('p_value', 1):.3f}, "
                        f"RI_ATE={ate_ri.get('ate', 0):.3f}, RI_p={ate_ri.get('p_value', 1):.3f})"
                    )

                # Mediation with the SAME graph-driven backdoor set as the
                # ATE on the binary outcome (R2-7.5 / B-H3), replacing the
                # all-numeric confounder list.
                med_adjust = self._probe_adjustment(df, trt, self.target_bin, graph_dot)
                med = self.estimate_mediation(df, trt, med_adjust, graph_dot)
                if med:
                    mediation_rows.append(med)

                cate_rows.extend(self.estimate_cate(df, trt, self.target_cont, graph_dot))
                if self.target_bin in df.columns:
                    cate_rows.extend(self.estimate_cate(df, trt, self.target_bin, graph_dot))

                if self.use_regime:
                    regime_rows.extend(self.estimate_regime_cate(df, trt, self.target_cont, graph_dot))
                    if self.target_bin in df.columns:
                        regime_rows.extend(self.estimate_regime_cate(df, trt, self.target_bin, graph_dot))

                refutation_report.append(f"\n## Treatment: {trt}\n")
                ref_meta_keys = ("total_passed", "total_tests", "refutation_seed",
                                 "reference_method", "reference_ate_lpm")
                refutation_report.append("### DV24\n")
                for test_name, test_res in ref_dv24.items():
                    if test_name in ref_meta_keys or not isinstance(test_res, dict):
                        continue
                    if test_res.get("skipped"):
                        status = "SKIP"
                    else:
                        status = "PASS" if test_res.get("passed") else "FAIL"
                    refutation_report.append(f"- **{test_name}**: {status} ({test_res})\n")
                refutation_report.append("### RI\n")
                for test_name, test_res in ref_ri.items():
                    if test_name in ref_meta_keys or not isinstance(test_res, dict):
                        continue
                    if test_res.get("skipped"):
                        status = "SKIP"
                    else:
                        status = "PASS" if test_res.get("passed") else "FAIL"
                    refutation_report.append(f"- **{test_name}**: {status} ({test_res})\n")

            # 6. SHRD nonlinear (binned) analysis
            shrd_rows = []
            if self.use_shrd_bins:
                shrd_rows = self.estimate_shrd_binned_ate(df, graph_dot)

            # 7. Positivity / common-support diagnostics (P1-3)
            pos_df = self.positivity_diagnostics(df, treatments)

            # 8. Headroom (PI - VMAX) effect modification (R2-1)
            hr_rows = []
            if self.use_headroom:
                hr_rows = self.estimate_headroom_cate(df, graph_dot)
            hr_df = pd.DataFrame(hr_rows)

            ate_df = pd.DataFrame(ate_rows)
            # FDR (Benjamini-Hochberg) across the treatment family, per outcome
            # (A-M3 / FIX_TASKS P1-1). Uses the block-bootstrap p-values where
            # available (P1-2); analytic p-values are the fallback.
            if not ate_df.empty and "p_value" in ate_df.columns:
                if "p_value_boot" in ate_df.columns:
                    p_for_fdr = ate_df["p_value_boot"].where(
                        ate_df["p_value_boot"].notna(), ate_df["p_value"])
                else:
                    p_for_fdr = ate_df["p_value"]
                ate_df["q_value_bh"] = np.nan
                for oc in ate_df["outcome"].unique():
                    m = (ate_df["outcome"] == oc).values
                    ate_df.loc[m, "q_value_bh"] = _bh_qvalues(p_for_fdr.loc[m].values)
                # E-values (VanderWeele & Ding 2017, methodology review E §7.1):
                # robustness of each ATE against unmeasured confounding.
                dv24_std = float(df["DV24"].std()) if "DV24" in df.columns else 1.0
                ate_df["e_value"] = ate_df.apply(
                    lambda r: _e_value(float(r["ate"]), dv24_std,
                                       r["outcome"] == self.target_bin),
                    axis=1,
                )

                def _ci_limit_evalue(r):
                    lo = r.get("ci_boot_low", np.nan)
                    hi = r.get("ci_boot_high", np.nan)
                    if pd.isna(lo) or pd.isna(hi):
                        lo, hi = r["ci_low"], r["ci_high"]
                    return _e_value_ci_limit(float(r["ate"]), float(lo), float(hi),
                                             dv24_std, r["outcome"] == self.target_bin)

                ate_df["e_value_ci_limit"] = ate_df.apply(_ci_limit_evalue, axis=1)
            qual_df = pd.DataFrame(qualitative_rows)
            med_df = pd.DataFrame(mediation_rows)
            cate_df = pd.DataFrame(cate_rows)
            regime_df = pd.DataFrame(regime_rows)
            shrd_df = pd.DataFrame(shrd_rows)

            out_dir = self.ate_csv.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            ate_df.to_csv(self.ate_csv, index=False)
            qual_df.to_csv(out_dir / "qualitative_classification.csv", index=False)
            med_df.to_csv(out_dir / "mediation_results.csv", index=False)
            cate_df.to_csv(self.cate_csv, index=False)
            regime_df.to_csv(out_dir / "regime_cate_results.csv", index=False)
            shrd_df.to_csv(out_dir / "shrd_binned_results.csv", index=False)
            pos_df.to_csv(out_dir / "positivity_diagnostics.csv", index=False)
            hr_df.to_csv(out_dir / "headroom_cate_results.csv", index=False)
            self.refutation_md.write_text("".join(refutation_report), encoding="utf-8")

            logger.info(
                f"ATE {len(ate_df)} rows, Qualitative {len(qual_df)} rows, "
                f"Mediation {len(med_df)} rows, CATE {len(cate_df)} rows, "
                f"Regime {len(regime_df)} rows, SHRD_bins {len(shrd_df)} rows, "
                f"Headroom {len(hr_df)} rows"
            )
        return ate_df, qual_df, med_df, cate_df, regime_df, shrd_df, "".join(refutation_report)
