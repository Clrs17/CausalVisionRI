# Reproduction requirements and known limitations

This document records input requirements, unsupported entry points, and artifact differences that affect reproduction. The numerical results are reported in [RESULTS.md](RESULTS.md).

## 1. Repository layout and path normalisation

The release combines two development projects into one root:

| Development location | Release location |
|---|---|
| `wpac_causal_phase1/src/*.py` | `src/wpac_causal/*.py` |
| `wpac_causal_phase2/src/*.py` | `src/wpac_warning/*.py` |
| `wpac_causal_phase2/scripts/*.py` | `scripts/<stage>/*.py` |
| `wpac_causal_phase1/data/**` | `data/phase1/**` |
| `wpac_causal_phase2/config/*.yaml` | `configs/phase2/*.yaml` |

Build-machine absolute roots were replaced by release-relative paths. Script bootstraps place `src/` on `sys.path`, and imports use `wpac_causal` or `wpac_warning`. Run commands from the release root so that `configs/` and `data/` resolve correctly.

`tests/verify_protocol_locks.py` checks text and supported binary formats for remaining build-machine paths. [PROTOCOL.md](PROTOCOL.md) explains the corresponding lock re-issue.

## 2. Unsupported pre-holdout comparison drivers

`ritcf_final_eval.py`, `ritcf_select.py`, and `ritcf_wrps1_eval.py` read per-arm prediction dumps from `data/preds/`. Those dumps are not included in this release. These entry points are retained for code inspection and are not part of the current reproduction route. `scripts/05_eval/ablation_summary.py` also requires the pre-holdout prediction dumps; the current 40-contrast panel is produced by `eval_holdout.py` and `make_sm_tables.py`.

Use `scripts/06_compare/compare_models_consistent.py` and the current comparison locks for the shared-sample analyses. `configs/phase2/config.yaml`, `config_ritcf.yaml`, and `config_ritcf_final.yaml` are explicit legacy variants whose development inputs are not included in this release; they are not production configurations.

Free-form runner logs under `data/runs/holdout_masked_v1/runner_logs/` and `data/runs_holdout/ablations/runner_logs/` are retained in local audit storage and excluded from the public data packages. Structured run manifests and training histories remain available, including `ablation_runner_manifest.json` and `ablation_runner_rerun1.json`.

## 3. Phase-1 companion scripts and unavailable inputs

The phase-1 companion scripts have development-tree defaults and are retained for code inspection rather than as current reproduction entry points. `make_fig5_shrd_ri_rate.py` and `make_fig_e1.py` additionally require `shrd_ri_rate_by_bin.csv`, `shrd_ri_rate_stormblock_ci.csv`, and `e1_threshold_utility.csv`, which are not included in the data packages.

The current paper figures use the release entry points listed in `SM_MAP.md` and the released prior bundle. [REPRODUCE.md](REPRODUCE.md) gives the supported commands.

## 4. Large or external inputs

### RITCF feature caches

`data/ritcf_recon/dataset/era5.npy` is 1.34 GB and `sat.npy` is 6.32 GB. They are excluded from the Zenodo record. `data/ritcf_recon/ACQUIRE.md` records their shapes, dtypes, and the rebuild command:

```bash
python -m scripts.ritcf_recon.build_dataset
```

The score sweeps, evaluation summaries, and manifests needed to check the reported RITCF results are included.

### Satellite frame store

The full frame store and its quality-controlled derivative are not included. The release provides the frame index, availability ledger, quality-control manifest, and acquisition scripts. Full retraining requires the HURSAT-B1 v07b source archives and a new frame-store build.

### Model checkpoints

Deep-learning `*.pt` checkpoints are not included. `run_holdout_arms.py`, `run_holdout_ablations.py`, and `test_infer.py` require the corresponding trained checkpoints. The frozen NPZ predictions support metric recomputation without those checkpoints. Fitted LR, LDA, and SHIPS-12 `model.pkl` baseline models are included in the data record.

The external LingBot checkpoint directories, `external/lingbot`, `external/lingbot-base`, and `external/lingbot-large`, must be supplied separately for the LingBot routes. LingBot is an external checkpoint rather than a pip dependency.

## 5. Configuration hashes after path normalisation

The release contains 62 development and ablation configuration snapshots. Their issued `config_sha256` values refer to the training-host files, whose paths differ from the shipped copies.

`protocol_locks/asset_sha256_manifest.json` records three values for each snapshot:

- `sha256_as_issued`, the original training-time digest;
- `sha256_verified_against_source`, the digest verified against the source file;
- `sha256_of_shipped_copy`, the digest of the path-normalised release file.

The lock verifier checks the shipped copy and preserves the other two values as provenance. `config_json_sha256` is a different quantity: it hashes the resolved in-memory configuration after arm overrides and is not expected to equal the hash of a released YAML file.

## 6. Environment and optional dependencies

- **Windows/conda:** set `KMP_DUPLICATE_LIB_OK=TRUE` when PyTorch and MKL-based libraries are loaded in the same process.
- **Headless figures:** the figure scripts select `matplotlib.use("Agg")` before importing `pyplot`.
- **Fonts:** the scripts request Arial. Install `ttf-mscorefonts` on Linux or accept the matplotlib fallback. Layout and analysis do not depend on the font choice; byte-identical rendering is only expected in the original environment.
- **`adjustText`:** not required by the current figure scripts.
- **`tcpyPI`:** optional except when recomputing the SHIPS potential-intensity predictor. The released predictor table can be used without it.
- **`CMP_TABLE_SUFFIX`:** selects the comparison table used for `fig_cma_label_control`. The default is `_best_auc`, which matches the shipped panel.

## 7. RITCF input and checkpoint provenance

The published RITCF sample counts are not reproduced by the independent frame definition. The reconstruction reports its own 10,144 training, 1,387 validation, and 1,057 test rows and presents the published counts separately.

Three checkpoint choices are reported: `best_loss`, `best_auc`, and `last`. Each comparison lock records the actual input file for every variant and every fallback.

For the 25.27-kt protocol, the best-AUC operating table contains V4direct and omits V1–V4. The V1–V4 validation-F1 points therefore use `sweep_table.csv`, while their best-AUC score sweeps use the corresponding `_aucfix_best_auc.npz` files. V4direct uses its best-AUC sweep and operating table. [COMPARISON.md](COMPARISON.md) gives the file-selection rule for each figure.

D200 is excluded from the causal sign and counterfactual constraints by the pre-specified identification rule.

## 8. RI E-value correction

The RI E-values in the released `ate_results.csv` use the fit panel's actual base rate, `1014/12647 = 0.080177`. An earlier calculation used 0.113. The corrected CSV and the re-derived phase-2 manifests describe the same shipped values.

The training-time manifest and identification-status table preserve the original ATE pointer as provenance. The phase-2 `prior_manifest.json` and sensitivity manifests carry the hashes of the corrected CSV and its updated prior manifest. The protocol locks and frozen predictions retain their original verification chain because these E-value columns are not model inputs.

The two artifact generations are distinguished explicitly in the asset manifest. [PROTOCOL.md](PROTOCOL.md) describes the accepted lock hashes and the path re-issue.

## 9. Graph-edge boundary recomputation

Recomputing Benjamini–Hochberg q-values from the raw p-value matrix in `pcmci_results.pkl` gives 86 edges at `q ≤ 0.05`, one fewer than the 87 edges in `causal_graph_edges.json`. The affected edge lies at the decision boundary, with a recomputed q-value of approximately 0.0527–0.053.

The locked `causal_graph_edges.json` is the graph used to train the 22 arms and to derive the identification DAG and released effects. The release preserves that graph and records the recomputation difference rather than replacing the training prior.

## 10. Verified reproduction scope

With both data packages extracted, the following checks run on CPU:

| Check | Command |
|---|---|
| Lock, receipt, path, and pinned-hash verification | `python tests/verify_protocol_locks.py` |
| Main metrics from frozen probabilities | `scripts/05_eval/eval_holdout.py` with `main.npz` |
| Ten analysis figures | `python scripts/07_figures/make_all_figures_pub.py` |
| Causal-discovery figure and its layout checks | `python scripts/07_figures/make_fig1_causal_discovery.py` |
| Supplementary tables | `python scripts/05_eval/make_sm_tables.py` |
| Release tests | `python -m pytest tests/ -q` |

`make_all_figures_pub.py` reads `table_figA_best_auc.csv` for the CMA label-control panel by default. Its ten PNGs are byte-identical to the shipped versions in the original environment. The PDF creation timestamps can differ.

Full retraining requires the original input fields, external checkpoints, and GPU resources described above. The frozen predictions and derived tables are sufficient for the CPU verification route.
