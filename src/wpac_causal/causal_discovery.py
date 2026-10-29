# [release] path normalization applied (build-machine absolute roots -> Path(__file__).resolve().parents[N]); see docs/KNOWN_ISSUES.md
"""TiGraMITe PCMCI+ causal discovery with meteorological prior constraints."""

from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from wpac_causal.preprocessing import IMPUTATION_KEY_COLUMNS, IMPUTATION_PREFIX
from wpac_causal.utils import Config, phase_timer

try:
    from tigramite.data_processing import DataFrame as TigramiteDataFrame
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite import plotting as tp
except Exception:  # pragma: no cover
    TigramiteDataFrame = None  # type: ignore
    PCMCI = None  # type: ignore
    ParCorr = None  # type: ignore
    tp = None  # type: ignore


class CausalDiscovery:
    """Run PCMCI+ on panel TC data and apply domain-knowledge priors."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.tau_max = int(config.get("causal_discovery.tau_max", 4))
        self.tau_min = int(config.get("causal_discovery.tau_min", 0))
        self.pc_alpha = config.get("causal_discovery.pc_alpha", 0.01)
        self.indep_test = config.get("causal_discovery.independence_test", "ParCorr")
        self.mask_type = str(config.get("causal_discovery.mask_type", "xyz"))
        self.variables = config.get("causal_discovery.variables", [])
        self.target = config.get("causal_discovery.target", "DV24_YJ")
        self.fdr_method = config.get("causal_discovery.fdr_method", "fdr_bh")
        # Whether X -> DV24_YJ contemporaneous (lag-0) links are allowed.
        # False enables the lag>=1 sensitivity mode; target_lag0_exceptions
        # preserves the legacy VMAX exception unless explicitly set to [].
        self.allow_target_lag0 = bool(config.get("causal_discovery.allow_target_lag0", True))
        self.target_lag0_exceptions = list(config.get(
            "causal_discovery.target_lag0_exceptions", ["VMAX"]
        ))
        self.results_pkl = Path(config.get("causal_discovery.results_pkl", "data/phase1/processed/pcmci_results.pkl"))
        self.graph_png = Path(config.get("causal_discovery.graph_png", "data/phase1/processed/causal_graph.png"))
        self.graph_dot = Path(config.get("causal_discovery.graph_dot", "data/phase1/processed/causal_graph.dot"))
        mask_path = (config.get("causal_discovery.imputation_mask_path")
                     or config.get("preprocessing.imputation_mask_path"))
        self.imputation_mask_path = Path(mask_path) if mask_path else None
        self.require_imputation_mask = bool(config.get(
            "causal_discovery.require_imputation_mask", bool(mask_path)))
        self.imputation_mask_columns = list(
            config.get("preprocessing.imputation_mask_columns", []) or [])
        self._last_imputation_audit = {
            "status": "disabled", "input_cells": 0,
            "input_cells_by_variable": {}, "panel_cells": 0,
        }

    # ------------------------------------------------------------------
    # Prior constraints
    # ------------------------------------------------------------------

    def _build_link_assumptions(self, var_names: List[str]) -> Dict[int, Dict[Tuple[int, int], str]]:
        """
        Build tigramite link_assumptions dict for PCMCI+.
        Format: {j: {(i, -tau): link_type, ...}, ...}
        link_type: '-?>' = link may exist (orientation i -> j if present);
                   'o?o' = contemporaneous adjacency may exist (orientation free);
                   '<?-' = required mirror of a '-?>' contemporaneous entry.
        Omitted links are forbidden (tigramite treats them as absent).

        Order of operations (fixes review A-C2: previously the contemporaneous
        tightening ran before the lag-restriction rules, silently killing ALL
        lag-0 edges into the target):
          1. allow everything;
          2. forbid target-outgoing and POT-outgoing edges;
          3. apply §4.4 lag-restriction rules (these only remove);
          4. contemporaneous tightening: lag-0 pairs must be whitelisted;
             target lag-0 pairs are whitelisted with FIXED orientation
             X -> DV24_YJ ('-?>' + '<-?' mirror), so PCMCI+ tests their
             existence but cannot orient DV24_YJ -> X.
        """
        idx = {name: i for i, name in enumerate(var_names)}
        n_vars = len(var_names)
        if self.target not in idx:
            raise ValueError(
                f"Target '{self.target}' not among causal variables: {var_names}"
            )
        target_idx = idx[self.target]

        links: Dict[int, Dict[Tuple[int, int], str]] = {}

        def _set(i: int, j: int, tau: int, kind: str) -> None:
            links.setdefault(j, {})[(i, -tau)] = kind

        def _remove(i: int, j: int, tau: int) -> None:
            if j in links:
                links[j].pop((i, -tau), None)

        # ---- 1) Allow all edges initially (autocorrelation at lag>0 included) ----
        for j in range(n_vars):
            for i in range(n_vars):
                for tau in range(self.tau_min, self.tau_max + 1):
                    if i == j and tau == 0:
                        continue  # no self-loop at lag 0
                    _set(i, j, tau, "o?o" if tau == 0 else "-?>")

        # ---- 2b) Forbidden: DV24_YJ must not parent anything ----
        # (POT was removed from the causal variable set in R2-1 as a
        # deterministic derived variable; its prior rules went with it.)
        for j in range(n_vars):
            if j == target_idx:
                continue
            for tau in range(self.tau_max + 1):
                _remove(target_idx, j, tau)

        # ---- 3) §4.4 lag-restriction rules (removal-only, applied BEFORE the
        #        contemporaneous tightening so whitelisted target lag-0 edges
        #        survive; review A-C2) ----
        def _restrict_lags(parent: str, child: str, allowed: set) -> None:
            if parent not in idx or child not in idx:
                return
            j = idx[child]
            for tau in range(self.tau_max + 1):
                if tau not in allowed:
                    _remove(idx[parent], j, tau)

        # RHLO -> DV24 only lags 1-3
        _restrict_lags("RHLO", self.target, {1, 2, 3})
        if self.allow_target_lag0:
            # §4.4 design table
            _restrict_lags("PER", self.target, {0})
            for _drv in ("SHRD", "VORT", "PV"):
                _restrict_lags(_drv, self.target, {0, 1})
        else:
            # Sensitivity mode: every X -> DV24 link starts at lag >= 1
            _restrict_lags("PER", self.target, {1})
            for _drv in ("SHRD", "VORT", "PV"):
                _restrict_lags(_drv, self.target, {1})

        # ---- 4) Contemporaneous tightening (lag 0): keep only whitelisted pairs ----
        allowed_contemp: set = set()
        directed_contemp: Dict[frozenset, Tuple[int, int]] = {}

        def _add_contemp(a: str, b: str, direction: Optional[Tuple[str, str]] = None):
            """Allow a contemporaneous edge between a and b.

            direction=(src, dst) additionally fixes orientation src -> dst
            ('-?>' on dst's entry plus '<-?' mirror on src's entry): existence
            is still TESTED, only the direction is constrained.
            """
            if a in idx and b in idx:
                pair = frozenset({idx[a], idx[b]})
                allowed_contemp.add(pair)
                if direction is not None:
                    directed_contemp[pair] = (idx[direction[0]], idx[direction[1]])

        _add_contemp("SST_OA", "SLHF_IC")
        _add_contemp("SST_IC", "SLHF_IC")
        _add_contemp("SHRD", "SHEAR_VMAX_ANGLE")
        _add_contemp("RHLO", "RHHI")
        _add_contemp("W500", "RHHI")
        _add_contemp("VORT", "PV")
        _add_contemp("WARM_CORE", "PV")
        _add_contemp("MFC", "W500")
        # Physical bidirectional feedback at lag 0:
        _add_contemp("VMAX", "WARM_CORE")  # thermal wind balance [Stern & Nolan 2012]
        _add_contemp("VORT", "WARM_CORE")  # low-level vorticity ↔ warm core coupling
        _add_contemp("D200", "W500")       # upper divergence ↔ mid-level ascent
        _add_contemp("MFC", "RHLO")        # moisture convergence ↔ low-level humidity
        _add_contemp("DRY_AIR", "RHLO")    # dry air intrusion modulates humidity
        _add_contemp("DRY_AIR", "RHHI")
        # VMAX ↔ SHRD: strong TC outflow/anticyclone reduces local environmental
        # shear simultaneously [Ryglicki et al. 2019, MWR; Dai et al. 2021, JAS].
        _add_contemp("VMAX", "SHRD")
        # VMAX ↔ SST contemporaneous excluded: cold wake takes days [Price 1981].
        # VMAX → SST lagged edges ARE allowed for slow-moving TCs.
        # OHC contemporaneous: SST and OHC are tightly coupled ocean state variables
        _add_contemp("SST_OA", "OHC_OA")   # SST determines upper ocean thermal structure
        _add_contemp("SST_IC", "OHC_IC")
        _add_contemp("OHC_OA", "OHC_IC")   # spatial correlation of OHC field
        # All other contemporaneous edges involving OHC are forbidden:
        # ocean responds slowly, atmosphere responds fast, current atmospheric
        # state cannot influence current OHC at 6h scale.

        # Target lag-0 edges per §4.4 + review A-C2: X -> DV24_YJ allowed with
        # FIXED orientation into the target (environment leads the future-24h
        # intensity-change window by at most 24h — the key anticipatory signal
        # for RI). These edges carry window-overlap synchronization risk and are
        # flagged in the results metadata (see run()); use
        # allow_target_lag0=false for the lag>=1 sensitivity run.
        if self.allow_target_lag0:
            for _drv in ("VMAX", "SHRD", "VORT", "PV", "PER",
                         "RHLO", "D200", "W500", "MFC", "WARM_CORE"):
                _add_contemp(_drv, self.target, direction=(_drv, self.target))
        else:
            for _drv in self.target_lag0_exceptions:
                _add_contemp(_drv, self.target, direction=(_drv, self.target))

        for j in range(n_vars):
            for (i, tau) in list(links.get(j, {}).keys()):
                if tau == 0 and i != j and frozenset({i, j}) not in allowed_contemp:
                    links[j].pop((i, 0), None)

        # Fix orientation of directed contemporaneous pairs and add the mirrors
        # tigramite requires ('-?>' on (src,0)->dst needs '<?-' on (dst,0)->src;
        # cf. tigramite _reverse_link / valid_types).
        # Only applied where the dst entry survived the rules above (e.g. RHLO's
        # lag-0 edge into the target was removed by its lag restriction).
        for pair, (src, dst) in directed_contemp.items():
            if (src, 0) in links.get(dst, {}):
                links[dst][(src, 0)] = "-?>"
                links.setdefault(src, {})[(dst, 0)] = "<?-"

        # Clean empty dicts
        links = {k: v for k, v in links.items() if v}
        return links

    # ------------------------------------------------------------------
    # Data formatting
    # ------------------------------------------------------------------

    def _merge_upstream_imputation_mask(self, df: pd.DataFrame,
                                        var_names: List[str]) -> pd.DataFrame:
        """Join the full-panel pre-fill mask onto this (possibly fit-only) input.

        Extra mask rows are expected because the file covers fit+inner+test.
        Every input row and the configured full feature schema are mandatory.
        """
        self._last_imputation_audit = {
            "status": "disabled", "input_cells": 0,
            "input_cells_by_variable": {}, "panel_cells": 0,
            "panel_cells_by_variable": {},
        }
        if not self.require_imputation_mask:
            return df
        if self.imputation_mask_path is None:
            raise RuntimeError("holdout requires an upstream imputation-mask path")
        if not self.imputation_mask_path.is_file():
            raise FileNotFoundError(
                f"upstream imputation mask missing: {self.imputation_mask_path}")
        sid_col = self.cfg.get("best_track.id_col", "SID")
        time_col = self.cfg.get("best_track.time_col", "ISO_TIME")
        if [sid_col, time_col] != IMPUTATION_KEY_COLUMNS:
            raise RuntimeError("upstream imputation mask requires SID+ISO_TIME keys")
        mask = pd.read_parquet(self.imputation_mask_path)
        if not set(IMPUTATION_KEY_COLUMNS) <= set(mask.columns):
            raise RuntimeError("upstream imputation mask lacks SID+ISO_TIME columns")
        data_columns = [c for c in mask.columns if c not in IMPUTATION_KEY_COLUMNS]
        if self.imputation_mask_columns:
            expected = [IMPUTATION_PREFIX + c for c in self.imputation_mask_columns]
            if (len(data_columns) != len(expected)
                    or set(data_columns) != set(expected)):
                raise RuntimeError(
                    "upstream imputation-mask columns are incomplete: "
                    f"missing={sorted(set(expected) - set(data_columns))} "
                    f"extra={sorted(set(data_columns) - set(expected))}")
        if (not data_columns
                or any(not c.startswith(IMPUTATION_PREFIX) for c in data_columns)):
            raise RuntimeError("upstream imputation mask has an invalid column schema")
        for frame, name in ((df, "causal input"), (mask, "upstream imputation mask")):
            keys = frame[IMPUTATION_KEY_COLUMNS]
            if keys.isna().any().any() or keys.duplicated().any():
                raise RuntimeError(f"{name} keys must be unique and non-null")
        for column in data_columns:
            if not pd.api.types.is_bool_dtype(mask[column].dtype):
                raise RuntimeError(f"upstream imputation-mask column {column} is not boolean")
        used_columns = [IMPUTATION_PREFIX + name for name in var_names
                        if IMPUTATION_PREFIX + name in data_columns]
        if set(used_columns) & set(df.columns):
            raise RuntimeError("causal input already contains reserved imputation-mask columns")
        merged = df[IMPUTATION_KEY_COLUMNS].merge(
            mask[IMPUTATION_KEY_COLUMNS + used_columns],
            on=IMPUTATION_KEY_COLUMNS, how="left", validate="one_to_one",
            indicator=True, sort=False)
        missing = merged["_merge"] != "both"
        if missing.any():
            missing_keys = merged.loc[missing, IMPUTATION_KEY_COLUMNS].head(5)
            raise RuntimeError(
                "upstream imputation mask is missing input row keys: "
                f"{missing_keys.to_dict('records')}")
        out = df.copy()
        for column in used_columns:
            out[column] = merged[column].to_numpy(dtype=bool)
        counts = {
            column[len(IMPUTATION_PREFIX):]: int(merged[column].sum())
            for column in used_columns}
        if used_columns:
            rows_with_imputation = int(merged[used_columns].any(axis=1).sum())
        else:
            rows_with_imputation = 0
        iso = pd.to_datetime(mask["ISO_TIME"], utc=True)
        mask_row_keys = (mask["SID"].astype(str) + "|"
                         + iso.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        row_key_text = "\n".join(sorted(mask_row_keys.astype(str)))
        self._last_imputation_audit = {
            "status": "applied",
            "path": str(self.imputation_mask_path),
            "sha256": hashlib.sha256(self.imputation_mask_path.read_bytes()).hexdigest(),
            "row_key_sha256": hashlib.sha256(row_key_text.encode("utf-8")).hexdigest(),
            "input_rows": len(df),
            "input_rows_with_imputation": rows_with_imputation,
            "input_cells": int(sum(counts.values())),
            "input_cells_by_variable": counts,
            "mask_columns": data_columns,
            "used_columns": used_columns,
            "panel_cells": 0,
            "panel_cells_by_variable": {},
        }
        return out

    def _to_tigramite_array(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert panel DataFrame to tigramite 3D array (sample, time, variable)
        and mask.

        MASK CONVENTION (tigramite / numpy masked-array semantics):
        ``mask=True`` means the sample is EXCLUDED from all analyses.
        Real observations must therefore be ``mask=False`` and padding /
        imputed positions ``mask=True``. (Fixes review A-C1, which had the
        semantics inverted so PCMCI+ ran on the padding constants.)

        Imputation policy:
        - When configured, the scaler's pre-fill SID+ISO_TIME mask is joined
          one-to-one before reindexing and ORed into this mask; synthesized 6h
          gap rows inherit masked status. Holdout requires that complete file.
        - Feature columns: short gaps may be linearly interpolated and longer
          gaps may use bfill/ffill to keep the array numeric, but EVERY filled
          cell remains masked because interpolation can use a future endpoint.
          The global-mean catch-all is masked too; only real observations are
          unmasked.
        - Target column (DV24_YJ): never interpolated — NaN targets are
          masked out directly (review C-M4).
        """
        sid_col = self.cfg.get("best_track.id_col", "SID")
        time_col = self.cfg.get("best_track.time_col", "ISO_TIME")
        var_names = [v for v in self.variables if v in df.columns]
        missing = [v for v in self.variables if v not in df.columns]
        if missing:
            logger.warning(f"Causal variables absent from DataFrame, excluded: {missing}")
        if not var_names:
            raise ValueError("No causal variables found in DataFrame")
        if self.target not in var_names:
            raise ValueError(
                f"Target '{self.target}' not found in DataFrame columns "
                f"(available causal vars: {var_names})"
            )
        if time_col in df.columns and df[[sid_col, time_col]].duplicated().any():
            duplicated = df.loc[
                df[[sid_col, time_col]].duplicated(keep=False),
                [sid_col, time_col]].head(5).to_dict("records")
            raise RuntimeError(
                f"duplicate (SID, time) rows before causal reindexing: {duplicated}")

        df = self._merge_upstream_imputation_mask(df, var_names)

        # Drop storms with NO environmental data at all (e.g. ERA5 not
        # downloaded): every feature would be global-mean-filled and fully
        # masked anyway, so they only waste memory and distort diagnostics.
        env_proxy = "SHRD" if "SHRD" in df.columns else var_names[0]
        good_sids = df.groupby(df[sid_col])[env_proxy].apply(lambda s: s.notna().any())
        good_sids = good_sids[good_sids].index
        n_drop = df[sid_col].nunique() - len(good_sids)
        if n_drop:
            logger.info(f"Dropping {n_drop} storms without any environmental data")
            df = df[df[sid_col].isin(good_sids)].copy()

        # Defensive ordering: lag structures assume time-sorted rows within
        # each storm (review A-M7). If no time column exists, upstream row
        # order is trusted as-is.
        if time_col in df.columns:
            df = df.sort_values([sid_col, time_col]).reset_index(drop=True)
            # Reindex each storm onto a COMPLETE 6h grid: a missing synoptic
            # time must leave a NaN row, otherwise rows after the gap are
            # stacked one position up and lag-1 silently becomes 12h
            # (review C-M3). Every synthesized gap row stays masked below,
            # whether its numeric placeholder comes from interpolation or fill.
            df = df.drop_duplicates(subset=[sid_col, time_col])
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                parts = []
                for sid, g in df.groupby(sid_col, sort=False):
                    g = g.set_index(time_col)
                    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="6h")
                    if len(full_idx) == len(g):
                        g_out = g.reset_index()
                    else:
                        g_out = g.reindex(full_idx)
                        g_out[sid_col] = sid
                        g_out = g_out.reset_index().rename(columns={"index": time_col})
                    parts.append(g_out)
                df = pd.concat(parts, ignore_index=True)
        sids = df[sid_col].unique()
        n_vars = len(var_names)
        target_idx = var_names.index(self.target)
        feature_vars = [v for v in var_names if v != self.target]

        df = df.copy()
        imputed = np.zeros((len(df), n_vars), dtype=bool)
        upstream_columns = list(self._last_imputation_audit.get("used_columns", []))
        for column in upstream_columns:
            name = column[len(IMPUTATION_PREFIX):]
            values = df[column]
            # Reindex-created 6h rows have NaN mask cells: exclude them too.
            imputed[:, var_names.index(name)] = (
                values.isna().to_numpy()
                | values.eq(True).to_numpy(dtype=bool, na_value=False))
        if upstream_columns:
            df = df.drop(columns=upstream_columns)
        panel_counts = {
            column[len(IMPUTATION_PREFIX):]: int(
                imputed[:, var_names.index(column[len(IMPUTATION_PREFIX):])].sum())
            for column in upstream_columns}
        self._last_imputation_audit["panel_cells"] = int(imputed.sum())
        self._last_imputation_audit["panel_cells_by_variable"] = panel_counts
        self._last_imputation_audit["panel_rows"] = len(df)
        self._last_imputation_audit["panel_rows_with_imputation"] = int(
            imputed.any(axis=1).sum())

        # Locally synthesized values are also excluded; OR preserves the
        # upstream mask even where preprocessing already mean-filled a value.
        n_interpolated = 0
        for sid in sids:
            rows = np.where((df[sid_col] == sid).values)[0]
            sub = df.iloc[rows][feature_vars]
            was_na = sub.isna().values
            sub_i = sub.interpolate(method="linear", limit_direction="forward", limit=3)
            interp_filled = was_na & sub_i.notna().values
            n_interpolated += int(interp_filled.sum())
            sub_f = sub_i.bfill().ffill()
            filled = was_na & sub_f.notna().values
            imputed[np.ix_(rows, [var_names.index(v) for v in feature_vars])] |= filled
            df.iloc[rows, df.columns.get_indexer(feature_vars)] = sub_f

        # Find max length
        groups = df.groupby(sid_col, sort=False).indices
        max_t = max(len(v) for v in groups.values())
        n_s = len(sids)

        data = np.full((n_s, max_t, n_vars), np.nan, dtype=float)
        # Start fully masked (=excluded); unmask only genuine observations.
        mask = np.ones((n_s, max_t, n_vars), dtype=bool)

        n_real_rows = 0
        for s, sid in enumerate(sids):
            rows = groups[sid]
            t = len(rows)
            n_real_rows += t
            data[s, :t, :] = df.iloc[rows][var_names].values
            mask[s, :t, :] = False                      # real observations are valid
            mask[s, :t, :] |= imputed[rows]             # ... except every filled cell
            mask[s, :t, target_idx] |= np.isnan(data[s, :t, target_idx])  # NaN target excluded

        # Global-mean catch-all for any remaining NaNs (e.g. whole-variable
        # missing). Filled positions are masked out so they never enter tests.
        for v in range(n_vars):
            col = data[:, :, v]
            bad = ~np.isfinite(col)
            if bad.any():
                valid = np.isfinite(col) & (~mask[:, :, v])
                fill = float(np.nanmean(col[valid])) if valid.any() else 0.0
                col[bad] = fill
                mask[:, :, v][bad] = True

        # Assertions (review A-C1 fix verification)
        n_valid = int((~mask).sum())
        assert n_valid <= n_real_rows * n_vars, "more valid cells than real rows"
        assert np.isfinite(data[~mask]).all(), "NaN/inf in unmasked samples"
        logger.info(
            f"Tigramite panel: {n_s} storms, max_t={max_t}, {n_vars} vars | "
            f"real rows={n_real_rows}, valid cells={n_valid} "
            f"({n_valid / max(n_real_rows * n_vars, 1):.1%}), "
            f"short-gap interpolated cells={n_interpolated}"
        )
        return data, mask, var_names

    # ------------------------------------------------------------------
    # Run PCMCI+
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        if PCMCI is None:
            raise RuntimeError("tigramite not installed")

        with phase_timer("CausalDiscovery"):
            data, mask, var_names = self._to_tigramite_array(df)
            if (self.require_imputation_mask
                    and self._last_imputation_audit.get("status") != "applied"):
                raise RuntimeError(
                    "holdout PCMCI requires an applied upstream imputation mask")
            n_vars = len(var_names)

            dataframe = TigramiteDataFrame(
                data=data,
                mask=mask,
                var_names=var_names,
                analysis_mode="multiple",
            )

            indep_test = ParCorr(significance="analytic", mask_type=self.mask_type)
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=indep_test, verbosity=0)

            link_assumptions = self._build_link_assumptions(var_names)

            results = pcmci.run_pcmciplus(
                tau_min=self.tau_min,
                tau_max=self.tau_max,
                pc_alpha=self.pc_alpha,
                link_assumptions=link_assumptions,
                fdr_method=self.fdr_method,
            )

            # Post-process: hard-remove forbidden outgoing edges from the target.
            # Lagged (tau>0): graph[target, j, tau] is target-outgoing by time
            # order -> remove. Lag 0: only remove when ORIENTED target -> j
            # ('-->'); a '<--' mirror of j -> target must be KEPT, otherwise
            # the symmetrization below would also kill the incoming edge
            # (found in smoke testing).
            target_idx = var_names.index(self.target)
            graph = results["graph"].copy()
            val_matrix = results["val_matrix"].copy()
            p_matrix = results["p_matrix"].copy()
            for j in range(n_vars):
                if j == target_idx:
                    continue
                for tau in range(1, self.tau_max + 1):
                    if graph[target_idx, j, tau] != "":
                        graph[target_idx, j, tau] = ""
                        val_matrix[target_idx, j, tau] = 0.0
                        p_matrix[target_idx, j, tau] = 1.0
                if graph[target_idx, j, 0] == "-->":
                    graph[target_idx, j, 0] = ""
                    graph[j, target_idx, 0] = ""
                    val_matrix[target_idx, j, 0] = 0.0
                    val_matrix[j, target_idx, 0] = 0.0
                    p_matrix[target_idx, j, 0] = 1.0
                    p_matrix[j, target_idx, 0] = 1.0

            # Enforce symmetric lag-0 entries after editing the graph:
            # tigramite requires graph[i,j,0] and graph[j,i,0] to be consistent.
            # Mirror val/p clearing as well (review A-m4).
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    a, b = graph[i, j, 0], graph[j, i, 0]
                    if a == "" or b == "":
                        if a != "" or b != "":
                            graph[i, j, 0] = ""
                            graph[j, i, 0] = ""
                            val_matrix[i, j, 0] = 0.0
                            val_matrix[j, i, 0] = 0.0
                            p_matrix[i, j, 0] = 1.0
                            p_matrix[j, i, 0] = 1.0
                    elif a == "x-x" or b == "x-x":
                        graph[i, j, 0] = "x-x"
                        graph[j, i, 0] = "x-x"
            results["graph"] = graph
            results["val_matrix"] = val_matrix
            results["p_matrix"] = p_matrix
            results["var_names"] = var_names

            # Lag-0 edges into the target: retained per CAUSAL_FRAMEWORK §4.4
            # but flagged — X(t) overlaps the first 6h of the DV24 window, so
            # these links carry synchronization-contamination risk
            # (review A-C2 / E §4.2). Use allow_target_lag0=false for the
            # lag>=1 sensitivity run.
            lag0_target_edges = []
            for i, name in enumerate(var_names):
                if i == target_idx:
                    continue
                if graph[i, target_idx, 0] != "":
                    lag0_target_edges.append({
                        "src": name,
                        "dst": self.target,
                        "lag": 0,
                        "val": float(val_matrix[i, target_idx, 0]),
                        "p": float(p_matrix[i, target_idx, 0]),
                    })
            results["lag0_target_edges"] = lag0_target_edges

            # Save pickle with metadata for stage-2 reproducibility (review A-m5)
            try:
                import importlib.metadata as _md
                _tig_ver = _md.version("tigramite")
            except Exception:
                _tig_ver = "unknown"
            metadata = {
                "pc_alpha": self.pc_alpha,
                "tau_min": self.tau_min,
                "tau_max": self.tau_max,
                "fdr_method": self.fdr_method,
                "allow_target_lag0": self.allow_target_lag0,
                "target_lag0_exceptions": list(self.target_lag0_exceptions),
                "independence_test": "ParCorr(analytic)",
                "mask_type": indep_test.mask_type,
                "n_storms": int(data.shape[0]),
                "max_t": int(data.shape[1]),
                "n_vars": int(n_vars),
                "n_valid_cells": int((~mask).sum()),
                "upstream_imputation": self._last_imputation_audit.copy(),
                "upstream_imputed_cells": int(
                    self._last_imputation_audit.get("input_cells", 0)),
                "tigramite_version": _tig_ver,
                "created_at": pd.Timestamp.utcnow().isoformat(),
                "lag0_target_edges_sync_risk": lag0_target_edges,
            }
            results["metadata"] = metadata
            self.results_pkl.parent.mkdir(parents=True, exist_ok=True)
            with open(self.results_pkl, "wb") as f:
                pickle.dump({
                    "val_matrix": results["val_matrix"],
                    "p_matrix": results["p_matrix"],
                    "graph": results["graph"],
                    "var_names": var_names,
                    "metadata": metadata,
                }, f)
            logger.info(f"PCMCI+ results saved to {self.results_pkl}")

            # Plot graph
            self._plot_graph(results, var_names)
            self._export_dot(results, var_names)

        return results

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _plot_graph(self, results: Dict, var_names: List[str]) -> None:
        """Save PNG of causal graph."""
        if tp is None:
            logger.warning("tigramite.plotting unavailable; skipping graph PNG")
            return
        try:
            self.graph_png.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = tp.plot_graph(
                val_matrix=results["val_matrix"],
                graph=results["graph"],
                var_names=var_names,
                figsize=(14, 14),
            )
            fig.savefig(self.graph_png, dpi=150, bbox_inches="tight")
            logger.info(f"Causal graph PNG saved to {self.graph_png}")
        except Exception as exc:
            logger.warning(f"Graph plotting failed: {exc}")

    def _export_dot(self, results: Dict, var_names: List[str]) -> None:
        """Export a lag-collapsed DOT graph with every lag in each edge label."""
        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx unavailable; skipping DOT export")
            return

        graph = results["graph"]  # shape (n_vars, n_vars, tau_max+1)
        n_vars = len(var_names)
        G = nx.DiGraph()
        for i, name_i in enumerate(var_names):
            G.add_node(name_i)
            for j, name_j in enumerate(var_names):
                if i == j:
                    continue
                for tau in range(graph.shape[2]):
                    link = graph[i, j, tau]
                    # tigramite graph codes: 'o-o', 'o->', '->', '<-', 'x-x', ''
                    if link in ("-->", "o->", "->"):
                        val = results["val_matrix"][i, j, tau]
                        label = f"lag{tau}|{val:.6f}"
                        if G.has_edge(name_i, name_j):
                            G[name_i][name_j]["label"] += f";{label}"
                        else:
                            G.add_edge(name_i, name_j, label=label)

        dot_str = nx.nx_pydot.to_pydot(G).to_string()
        self.graph_dot.parent.mkdir(parents=True, exist_ok=True)
        self.graph_dot.write_text(dot_str, encoding="utf-8")
        logger.info(f"Causal graph DOT saved to {self.graph_dot}")
