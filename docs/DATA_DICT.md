# Data dictionary

The Zenodo packages contain derived products released under CC BY 4.0. The GitHub checkout contains the code and documentation; extracting `record1_core.zip` and `record2_ritcf_recon.zip` into its root populates `data/`.

Row identity is `row_uid = "<SID>|<ISO_TIME>"`, where `SID` is the IBTrACS storm identifier and `ISO_TIME` is the UTC timestamp. Row-set fingerprints use `sha256("\n".join(sorted(row_uid)))` and appear in locks, manifests, and prediction files.

## 1. Phase-2 assets: `data/holdout_masked_v1/`

| File | Contents |
|---|---|
| `panel_fit0017.parquet` | Analysis panel with 17,655 rows and 42 columns |
| `frames_index.parquet` | The 17,441 exact `t0` row-to-frame matches, with `SID`, `ISO_TIME`, `frame_path`, `frame_time`, and `dt_h`; `dt_h=0` certifies exact-time matching |
| `frame_availability.json` | Available and excluded row keys, exclusion reasons, counts, and split fingerprints |
| `tbb_norm_fit0017.json` | Fit-only TBB normalisation statistics |
| `prior_manifest.json` | Paths and SHA-256 values for the phase-1 prior bundle and phase-2 assets |
| `trusted_signs.json` | Eight effect signs used by the training constraints, plus exclusions and generation rules |
| `dv24_yj_transformer.pkl` | Fit-only Yeo–Johnson transformation for 24-h intensity change |
| `feature_scaler.pkl` | Fit-only environmental feature scaler |
| `ships12_predictors_fit0017_{cohc,ohcd}.parquet` | Twelve SHIPS predictors under the two ocean variants |
| `ships12_predictors_fit0017_cohc.npz` and manifests | Array representation and provenance for the predictor table |
| `population_audit.json`, `build_manifest.json` | Population accounting and build fingerprints |

`frame_path` identifies the source frame used for image loading, and `frame_time` is the satellite acquisition time. This file indexes the panel-aligned rows rather than providing the full 153,295-frame inventory.

### 1.1 Panel columns

| Group | Columns |
|---|---|
| Identity and time | `SID`, `NAME`, `ISO_TIME`, `storm_year` |
| Track | `USA_LAT`, `USA_LON`, `LAT`, `LON`, `PER`, `SPD`, `DIR`, `LAND_DIST`, `LC_STAGE` |
| Intensity | `VMAX`, `VMAX_F24`, `F24_TIME`, `DV24`, `RI`, `USA_PRES`, `POT` |
| Environment | `SHRD`, `USHR`, `VSHR`, `SHEAR_DIR`, `SHEAR_VMAX_ANGLE`, `D200`, `RHLO`, `RHHI`, `VORT`, `PV`, `TPW`, `W500`, `DRY_AIR`, `MFC`, `SLHF_IC` |
| Ocean | `SST_IC`, `SST_OA`, `OHC_IC`, `OHC_OA`, `WARM_CORE` |
| Derived targets | `DV24_YJ`, `SHRD_RAW` |

`DV24 = VMAX(t+24h) − VMAX(t)` and `RI = (DV24 >= 30 kt)`. `F24_TIME` is the realised endpoint time used by the split embargo.

`data/phase1/processed_masked/holdout_fit0017/panel/imputation_mask.parquet` is a 17,655 × 25 Boolean table. Each `__IMPUTED_<VAR>` column records whether a value was filled. The 725-row `sample_manifest.csv` is the canonical storm inventory before label-eligibility filtering. The 720-row `split_manifest.csv` contains the eligible storms' split assignments; the five-storm difference is the label-ineligible set.

## 2. Causal prior: `data/phase1/processed_masked/holdout_fit0017/prior/`

The prior is estimated from 522 fit-period storms and 12,647 observations using masked PCMCI+ with ParCorr, `tau_max=4`, `pc_alpha=0.01`, and `mask_type=xyz`.

| File or directory | Contents |
|---|---|
| `causal_graph_edges.json` | Ninety-seven candidate directed edges with `src`, `dst`, `lag`, `coef`, `p_value`, `q_value`, `p_value_type`, `link_type`, and `sign`; 87 edges are retained for the training graph |
| `pcmci_results.pkl` | Raw TiGraMITe result object |
| `ate_results.csv` | Twenty treatment–outcome rows with effects, confidence intervals, bootstrap results, adjustment sets, refutation outcomes, and E-values |
| `cate_results.csv` | Effect estimates by lifecycle stage, latitude, and intensity strata |
| `headroom_cate_results.csv` | Effect modification by potential-intensity headroom |
| `regime_cate_results.csv` | Effect modification by SST and shear regimes |
| `shrd_binned_results.csv` | Effect estimates by shear bin |
| `identification_status.csv` | Directed-path status, adjustment sets, identification sources, and ATE-row hashes |
| `identification_dag.json`, `.dot` | Deterministic 23-node identification DAG |
| `dag_audit_all/` | Cycle breaking, structural additions, adjacency, construction sensitivity, and summary |
| `refutation_report.md` | Placebo, subset, synthetic-confounder, and offshore refutation results |
| `trusted_signs.json` | Eight retained effect signs: `SHRD −`, `RHLO +`, `VORT −`, `PV −`, `WARM_CORE −`, `W500 −`, `OHC_IC +`, `OHC_OA +` |
| `causal_effects_db.yaml` | Effects in the phase-1 knowledge-base schema |
| `positivity_diagnostics.csv`, `qualitative_classification.csv`, `mediation_results.csv` | Overlap, qualitative classification, and mediation diagnostics |
| `fit_prior_panel.parquet`, `prior_manifest.json`, `phase1_prior_manifest.json` | Estimation rows and prior provenance |

D200 is excluded from the sign and counterfactual constraints by the pre-specified `ate_exclude_treatments` rule. The identification filter applies this exclusion.

## 3. Sensitivity variants: `data/phase1/.../sensitivity/`

| Variant | Change |
|---|---|
| `strict/` | Re-run discovery after excluding non-self contemporaneous target links |
| `no_ohc/` | Remove `OHC_IC` and `OHC_OA` from graph variables, treatments, and effect analyses |
| `mjo_enso/` | Add `OMI1`, `OMI2`, and `NINO34` to every baseline adjustment set |
| `lat/` | Add `LAT` to the `OHC_OA` adjustment set |

Each variant includes its own `MANIFEST.json`, `SENSITIVITY_README.md`, and run log. Full `config_snapshot.yaml` files are included for the re-run variants (`strict` and `no_ohc`); the adjustment-only variants (`lat` and `mjo_enso`) record an empty `config_diff` in their manifests. The manifest records the graph and effect outputs for that variant. Candidate-edge counts and retained significant-edge counts are distinct quantities.

## 4. Inner predictions and run metadata

The data record includes 66 inner-prediction files for 22 development arms, 40 ablations, and four statistical baselines.

| Path | Contents |
|---|---|
| `data/runs/holdout_masked_v1/<arm>/` | Development-arm configuration, run manifest, inner predictions, and related audit files |
| `data/runs_holdout/ablations/<condition>_<arm>/` | Ablation configuration, shared-initialization provenance, inner predictions, and audit files |
| `data/runs_holdout/baselines/` | LR, LDA, and SHIPS-12 baseline manifests and inner predictions |

The 62 development and ablation configuration snapshots have issued, source-verified, and shipped-copy hashes in `protocol_locks/asset_sha256_manifest.json`. Deep-learning `*.pt` checkpoints are not included in the data packages; fitted LR, LDA, and SHIPS-12 `model.pkl` baseline models are included.

## 5. Frozen test predictions: `data/test/holdout_masked_v1/`

The record contains 53 NPZ files, 53 manifest sidecars, and `test_receipt.json`.

Entities include `main`, `k4`, `k6`, `all22`, five `M0_<arm>` members, 40 `abl_<condition>_<arm>` ablations, and four baselines. The `_2024` suffix in evaluation output names refers to the complete 2020–2024 test period.

| NPZ key | Meaning |
|---|---|
| `p` | Entity probability for each of the 3,004 aligned test rows |
| `p_members` | Member probabilities with shape `(n_rows, n_members)`: 5 for `main`, 4 for `k4`, 6 for `k6`, 22 for `all22`, and 1 for M0 and the statistical baselines |
| `y` | USA 30-kt RI label |
| `member_names`, `weights` | Ensemble membership and uniform weights |
| `threshold` | Inner-derived decision threshold, 0.225 for `main` |
| `gate_mean_members`, `gate_std_members`, `causal_share_members` | Fusion and causal-attention diagnostics |
| `entity`, `lock_sha256`, `split` | Entity identity and protocol provenance |
| `sid`, `iso_time`, `storm_year`, `row_uid` | Join keys |

Each sidecar records the entity, threshold, NPZ hash, row-key hash, and member configuration hashes. The receipt records the completed once-only production inference.

## 6. Analysis products

| Path | Contents |
|---|---|
| `data/analysis/eval/` | Per-entity sample counts, event rates, BS, BSS, ROC/PR-AUC, threshold metrics, reliability bins, yearly results, and paired bootstrap results |
| `data/analysis/gate_participation.*` | Gate and causal-attention summaries by member, year, and RI stratum |
| `data/analysis/compare/` | Comparison tables, score-sweep interpolations, coverage and waterfall tables, and comparison figures |
| `data/analysis/figures_pub/` | Generated analysis and publication-formatted figures; exact exhibit paths are listed in `SM_MAP.md` |
| `data/analysis/worst_year_ablation.json` | Worst-year scores, contrasts, and bootstrap analysis |
| `data/analysis/year_gradient_ablation.json` | Yearly and pooled graph-removal and loss-removal contrasts |
| `data/analysis/ablation_2020_2021_panel.json` | Forty contrasts on the 1,316-row two-year subset |
| `data/analysis/sm_tables.tex` | Generated supplementary table body |
| `data/analysis/wrps1_*` | WRPS1 shared-sample scores, sample definition, and bootstrap results |
| `data/protocol_locks/` | Data-record copies of the root protocol locks |

`docs/WORST_YEAR_ABLATION.md` and `docs/YEAR_GRADIENT_ABLATION.md` are generated from the corresponding JSON files by the scripts named in `SM_MAP.md`.

## 7. RITCF reconstruction: `data/ritcf_recon/`

The second package contains the sample frame, pair tables, normalisation, manifests, run audit trails, score sweeps, and evaluation summaries.

| Path | Contents |
|---|---|
| `dataset/` | `samples.parquet`, `pairs_{train,val}.parquet`, `norm.json`, `manifest.json`, and `his.npy` |
| `dataset_kt30/`, `dataset_kt2524/` | Alternative label and test-window definitions |
| `runs*/` | Run manifests, training histories, and evaluation outputs without model weights |
| `sweep_*.npz`, `sweep_table*.csv` | Per-row scores and operating-point sweeps for the checkpoint-selection kinds |
| `summary_*.md` | Variant-by-checkpoint evaluation summaries |

`era5.npy` and `sat.npy` are excluded because of their size. `data/ritcf_recon/ACQUIRE.md` gives their shapes, dtypes, and the rebuild command.

## 8. Inputs that must be obtained separately

| Input | Acquisition route |
|---|---|
| Original ERA5, GLORYS12V1, IBTrACS, and HURSAT-B1 fields | `scripts/01_data/download_hursat.py`, `download_cmems_ohc.py`, and `src/wpac_causal/download*.py` |
| Global frame store and quality-controlled frames | `scripts/01_data/download_global_hursat.py`, then `build_hursat_qc.py` |
| MAE memory-mapped caches | `scripts/01_data/build_mae_cache.py`, `merge_global_cache.py` |
| Model checkpoints | `scripts/03_assets/pretrain_mae.py`, `scripts/04_train/run_holdout_arms.py`, `run_holdout_ablations.py` |
| RITCF `era5.npy` and `sat.npy` | `python -m scripts.ritcf_recon.build_dataset` |
| External LingBot checkpoints | The external checkpoint locations described in `KNOWN_ISSUES.md` |

The released `frames_index.parquet`, `frame_availability.json`, `qc_manifest.json`, and `rekey_crosswalk.json` record the identity, time, and store provenance of the frames used by the analysis. Raw pixels and model weights are obtained through the routes above.
