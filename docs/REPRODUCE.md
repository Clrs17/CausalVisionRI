# Reproduction guide

This guide covers two routes. CPU verification recomputes the reported results from released artifacts. Full reproduction rebuilds the predictors, causal priors, model checkpoints, and frozen predictions from source data.

## 1. Set up the code and data

Clone the repository and enter its root directory:

```bash
git clone https://github.com/Clrs17/CausalVisionRI.git
cd CausalVisionRI
```

Install the declared dependencies in an appropriate environment:

```bash
python -m pip install -r requirements/phase1.txt -r requirements/phase2.txt
```

Download `record1_core.zip` and `record2_ritcf_recon.zip` from [Zenodo record 23768442](https://doi.org/10.5281/zenodo.23768442), then extract both into the repository root:

```bash
unzip /path/to/record1_core.zip
unzip /path/to/record2_ritcf_recon.zip
sha256sum -c DATA_MANIFEST_record1_core.sha256
sha256sum -c DATA_MANIFEST_record2_ritcf_recon.sha256
```

The packages contain derived quantities only. [DATA_DICT.md](DATA_DICT.md) describes their contents and [KNOWN_ISSUES.md](KNOWN_ISSUES.md) lists the inputs that must be obtained separately for full reproduction.

A code-only checkout can run the layout and configuration tests before the data are downloaded. Eight data-dependent tests are skipped in that state.

## 2. Verify the release on CPU

Run each command from the repository root:

```bash
# Lock self-hashes, receipt linkage, pinned artifact hashes, and path screening.
python tests/verify_protocol_locks.py

# Recompute the main metrics from the frozen probabilities.
python scripts/05_eval/eval_holdout.py \
    --config configs/phase2/holdout_masked_v1.yaml \
    --pred  data/test/holdout_masked_v1/main.npz \
    --lock  protocol_locks/final.json \
    --out   /tmp/eval_main

# Evaluate the shared-sample comparison tables.
python scripts/06_compare/compare_models_consistent.py

# Regenerate the analysis figures and supplementary tables.
python scripts/07_figures/make_all_figures_pub.py
python scripts/05_eval/make_sm_tables.py

# Run the release tests.
python -m pytest tests/ -q
```

The main evaluation should produce:

```text
n=3004  BSS_dev 0.258973  ROC_AUC 0.899712  PR_AUC 0.517783
POD 0.680  FARate 0.095  FARatio 0.635  CSI 0.311  threshold 0.225
```

With both data packages extracted, the test suite reports **11 passed**. The lock and prediction checks run on CPU and do not train a model.

`make_all_figures_pub.py` uses `table_figA_best_auc.csv` for the CMA label-control panel by default. Keep that setting when checking the shipped figure. `CMP_TABLE_SUFFIX` selects a different comparison table when another analysis is required. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) explains the figure defaults and environment requirements.

## 3. Rebuild the full chain

Full reproduction requires the original ERA5, GLORYS12V1, IBTrACS, and HURSAT-B1 inputs, the external LingBot checkpoints for the corresponding arms, and suitable GPU resources. The release does not include raw fields, the large frame store, or model weights.

The following stages define the complete route. Paths are relative to the repository root.

| Stage | Command or entry point | Main product |
|---|---|---|
| Source data | `scripts/01_data/download_hursat.py`, `download_cmems_ohc.py`, `src/wpac_causal/download*.py` | Original storm, atmospheric, ocean, and satellite inputs |
| Best track and panel | `scripts/02_causal/main.py --config configs/phase1/config_alltc_0024.yaml` | Storm-aligned analysis panel |
| Global frame store | `scripts/01_data/download_global_hursat.py`, then `build_hursat_qc.py` | Quality-controlled frame store and index |
| Causal priors | `scripts/02_causal/run_holdout_fit0017.py` | Fit-only graph, effects, identification, and trusted signs |
| Prior sensitivity | `scripts/02_causal/run_holdout_fit0017_sensitivity.py` | Strict-lag, no-OHC, MJO–ENSO, and latitude variants |
| Phase-2 assets | `scripts/03_assets/build_holdout_assets.py` | Validated model inputs and manifests |
| SHIPS predictors | `scripts/01_data/baseline_ships12.py --recompute` | COHC and OHCD predictor tables |
| MAE caches | `scripts/01_data/build_mae_cache.py`, `merge_global_cache.py` | Memory-mapped pretraining corpora |
| MAE pretraining | `scripts/03_assets/pretrain_mae.py --config configs/phase2/holdout_masked_v1.yaml` for WNP; append `--population global` for the global route | WNP and global visual encoders |
| Embedding caches | `scripts/03_assets/precompute_frame_embeds.py` | Frozen-backbone embeddings for the registered routes |
| Development arms | `scripts/04_train/run_holdout_arms.py --jobs 4` | Twenty-two trained arms and inner predictions |
| Ensemble selection | `scripts/04_train/holdout_select.py` | `protocol_locks/selection.json` |
| Ablations | `scripts/04_train/run_holdout_ablations.py` | Forty member–condition models |
| Statistical baselines | `scripts/04_train/baseline_lr.py` | LR, LDA, and SHIPS-12 variants |
| Final lock | `scripts/04_train/finalize_holdout.py --selection protocol_locks/selection.json --out protocol_locks/final.json` | Locked entities, thresholds, and evaluation policy |
| Locked test inference | `scripts/05_eval/test_infer.py --lock protocol_locks/final.json --out-dir data/test/holdout_masked_v1` | Frozen test predictions and completion receipt |
| Evaluation | `scripts/05_eval/eval_holdout.py`, `gate_analysis.py`, `ablation_worst_year.py`, `ablation_year_gradient.py`, `ablation_subset_panel.py` | Metrics, uncertainty, and ablation analyses |
| External comparisons | `scripts/06_compare/compare_models_consistent.py` and the WRPS1 drivers | Shared-sample tables and comparison locks |
| Figures and tables | `scripts/07_figures/*`, `scripts/05_eval/make_sm_tables.py` | Release artwork and supplementary tables |

The selection, finalisation, and inference stages enforce the production guards described in [PROTOCOL.md](PROTOCOL.md). A completed test receipt prevents another production inference pass. Use separate fixture directories for software tests and keep them outside the production paths.

The producing setup required approximately three GPU-days plus 30 CPU-hours for the full chain, excluding source-data download time. Actual run time depends on the GPU, storage, and download service. `--jobs 4` was used on a 96 GB GPU; choose a smaller value on GPUs with less memory.

## 4. Reconstruct RITCF separately

The RITCF reconstruction has its own data pipeline and training commands. See [RITCF_RECON_SPEC.md](RITCF_RECON_SPEC.md) and [../scripts/ritcf_recon/README.md](../scripts/ritcf_recon/README.md). Its large `era5.npy` and `sat.npy` caches are rebuilt locally and are not included in the Zenodo packages.

The consistent-comparison engine evaluates the frozen CVRI-Ens predictions alongside the reconstructed RITCF score sweeps. [COMPARISON.md](COMPARISON.md) defines the sample intersections, label sources, thresholds, checkpoint choices, and output files.

## 5. Windows and headless systems

For Windows/conda environments that load PyTorch and MKL-based libraries in one process, use:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Figure scripts select the `Agg` backend before importing `pyplot`. Keep that ordering on headless systems. The scripts request Arial; a missing font changes the fallback font rather than the analysis. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for optional dependencies and platform-specific requirements.
