# RITCF reconstruction pipeline

This package builds and evaluates an independent reconstruction of the RITCF contrastive-learning scheme published by Wang, Yang and Li (2025), *PNAS* 122(4), e2415501122.

The package constructs RITCF samples, trains the registered variants, and produces per-row score sweeps. `scripts/06_compare/compare_models_consistent.py` performs the separate shared-sample comparison with the frozen CVRI-Ens predictions.

See [../../docs/RITCF_RECON_SPEC.md](../../docs/RITCF_RECON_SPEC.md) for labels, inputs, model definitions, and evaluation frames. See [../../docs/COMPARISON.md](../../docs/COMPARISON.md) for the common-standard figures and tables.

## 1. Modules

| File | Role |
|---|---|
| `config.py` | Paths, feature layout, split years, RI threshold, pair rules, variant registry, and hashing |
| `build_dataset.py` | Sample-frame construction, per-storm feature extraction, memory maps, normalisation, manifests, and pair tables |
| `pairs.py` | Training and validation pairs plus rolling and capped inference reference pools |
| `features.py` | Memory-map reader and batch normalisation using training-split statistics |
| `model.py` | Shared-weight Siamese encoder, scalar unshared skeleton, and direct-classifier control |
| `train.py` | Fixed-budget training, validation metrics, three checkpoint types, and run manifests |
| `infer_eval.py` | Validation thresholds, reference-pool inference, metrics, and checkpoint evaluation |
| `operating_sweep.py` | Per-row score output and vote or continuous-threshold sweeps |

Run module commands from the release root using `python -m scripts.ritcf_recon.<module>`.

## 2. Data products

A full build produces:

```text
data/ritcf_recon/
  dataset/
    samples.parquet
    era5.npy
    sat.npy
    his.npy
    norm.json
    pairs_train.parquet
    pairs_val.parquet
    manifest.json
  runs/<variant>/
    checkpoint.pt
    checkpoint_auc.pt
    checkpoint_last.pt
    history.json
    run_manifest.json
    eval[_<kind>].json
    eval[_<kind>].md
  summary_<tag|main>_<kind>.md
  sweep_<variant>_<tag|main>_<kind>.npz
  sweep_table_<tag|main>_<kind>.csv
```

The Zenodo package includes the sample frame, pair tables, normalisation, manifests, histories, evaluations, and score sweeps. It excludes deep-learning `*.pt` checkpoints and the large `era5.npy` and `sat.npy` caches. `data/ritcf_recon/ACQUIRE.md` gives the cache shapes and rebuild instructions.

The primary frame has 10,144 training, 1,387 validation, and 1,057 test rows. The corresponding pair counts are 21,461 training pairs and 3,247 validation pairs. The manifests record the source rows, labels, split, and hashes.

## 3. Build the dataset

```bash
# Full dataset with a worker pool.
python -m scripts.ritcf_recon.build_dataset --workers 8

# Small build for exercising the pipeline.
python -m scripts.ritcf_recon.build_dataset --smoke

# Reprint the existing build report without rebuilding.
python -m scripts.ritcf_recon.build_dataset --report-only
```

The builder fingerprints the input data and code. A rerun reuses an unchanged dataset. `norm.json` uses training-split statistics, and the integrity checks verify finite normalised arrays, pair labels, reference pools, and the minimum 24 h separation.

The frame filter requires CMA wind at both endpoints, a valid centre, `DIST2LAND > 0`, and complete ERA5 and SAT windows. Exact satellite times are required by default (`--sat-tolerance-min 0`).

## 4. Train and evaluate

For Windows/conda environments that require the MKL workaround, set:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Train the registered variants with the fixed 60-epoch budget:

```bash
python -m scripts.ritcf_recon.train --variant V1 --epochs 60 --patience 60
python -m scripts.ritcf_recon.train --variant V3 --epochs 60 --patience 60
python -m scripts.ritcf_recon.train --variant V4 --epochs 60 --patience 60
python -m scripts.ritcf_recon.train --variant V4direct --epochs 60 --patience 60
```

V2 changes only the inference reference pool. Reuse the V1 checkpoint under the capped pool instead of repeating the same training:

```bash
python -m scripts.ritcf_recon.infer_eval --variant V2 \
    --checkpoint data/ritcf_recon/runs/V1/checkpoint.pt
```

Evaluate one run or all registered runs:

```bash
python -m scripts.ritcf_recon.infer_eval --variant V1
python -m scripts.ritcf_recon.infer_eval --all
python -m scripts.ritcf_recon.infer_eval --variant V1 --threshold-rule sample_f1
```

Evaluate all three checkpoint types in one command:

```bash
python -m scripts.ritcf_recon.infer_eval --all --tag kt30_aucfix \
    --checkpoint-kind best_loss best_auc last
```

Generate an operating-point sweep for a selected checkpoint:

```bash
python -m scripts.ritcf_recon.operating_sweep --tag kt30_aucfix \
    --checkpoint-kind best_auc
```

Every run records its configuration, code and dataset hashes, row-key fingerprints, seed, micro-batch and accumulation choices, parameter count, and validation history.

## 5. Checkpoint selection

Three checkpoints are saved from each trajectory:

| File | Selection |
|---|---|
| `checkpoint.pt` | Minimum validation loss |
| `checkpoint_auc.pt` | Maximum validation ROC-AUC |
| `checkpoint_last.pt` | Last epoch of the fixed training budget |

`history.json` and `run_manifest.json` record the loss and AUC series and the selected epochs. The fixed budget is `--epochs 60 --patience 60`, so the training run covers all 60 epochs rather than stopping after the first loss minimum.

`infer_eval.py` and `operating_sweep.py` accept `--checkpoint-kind best_loss`, `best_auc`, or `last`. The kind is included in output filenames. An explicit checkpoint supplied with `--checkpoint` is labelled `custom`.

The comparison engine accepts `--recon-kind` with the same three values and `--recon-tag-suffix`, which defaults to `_aucfix`. Every input and fallback is recorded in `protocol_locks/comparison_secondary_<kind>.json`. [../../docs/COMPARISON.md](../../docs/COMPARISON.md) gives the source used by each paper figure.

## 6. Implementation choices and deviations

The following choices are recorded so that the reconstruction can be distinguished from the published pipeline:

1. **Satellite source.** Inputs come from the quality-controlled HURSAT store built from the source tarballs. Multi-satellite choices are deterministic, and nonphysical pixels are masked before frame-level filling.
2. **ERA5 cadence.** The local archive is already six-hourly, so required times are exact steps.
3. **Window boundaries.** ERA5 indices outside a storm archive box are clamped to the nearest edge. The number of affected rows is recorded.
4. **Sample frame.** The independent filter produces its own row counts. Published counts are reported separately because the published filtering information does not reproduce them.
5. **Cache precision.** SAT and ERA5 are cached as float16. The manifest preserves the information needed to rebuild the arrays from source fields.
6. **Missing values.** ERA5 and history channels use training means. Satellite frames use a per-frame mean, with the pre-fill fraction recorded.
7. **Optimisation.** The effective batch approximates 256 pairs through a memory-probed micro-batch and gradient accumulation. The fixed budget, learning rate, and layer choices are recorded in each run manifest.
8. **Pair threshold.** The default pair threshold maximizes validation pair-level F1. A sample-level F1 rule is available as a separate validation-only sensitivity analysis.
9. **Probability metrics.** Soft Brier scores use validation-fitted Platt scaling on the test split. ROC-AUC and PR-AUC are threshold-free.
10. **Direct control.** V4direct uses the same input features with a linear logit and binary cross-entropy. Its natural-sample training distribution differs from the Siamese pair table, so it is a pipeline-level control rather than an isolated estimate of the contrastive objective.

The published description and released implementation differ in model details and normalisation. The reconstruction follows the published description for those choices and records the remaining unspecified architecture choices in [../../docs/RITCF_RECON_SPEC.md](../../docs/RITCF_RECON_SPEC.md).

## 7. Verification and limitations

Dataset checks cover normalisation ranges, missing values, pair labels, reference-pool dates, time separation, and zero distance for a sample paired with itself. Run manifests bind each evaluation to its code and dataset fingerprints.

A code edit changes the package fingerprint. Rebuilding the dataset after such an edit creates a new fingerprint, and old checkpoints are rejected by the evaluation guards. Reuse the existing dataset when only its code has changed, retrain as required, and rebuild only when the feature definition or source data has changed.

Samples with an incomplete five-time ERA5 or satellite window are excluded and counted. The reconstruction uses one split and one seed, matching the period structure of the published comparison rather than adding cross-validation.

The test split contains 66 RI events, giving approximate 95% binomial uncertainty of ±11 percentage points for POD. The independent inputs and sample filtering differ from the published pipeline, so the results describe this reconstruction under its stated frame. [../../docs/RESULTS.md](../../docs/RESULTS.md) gives the shared-panel results, where every model within a panel uses the same rows and labels.

## 8. Related documentation

- [../../docs/RITCF_RECON_SPEC.md](../../docs/RITCF_RECON_SPEC.md): input and model specification.
- [../../docs/COMPARISON.md](../../docs/COMPARISON.md): shared samples, thresholds, checkpoints, and output tables.
- [../../docs/REPRODUCE.md](../../docs/REPRODUCE.md): release setup and full-chain context.
- [../../docs/KNOWN_ISSUES.md](../../docs/KNOWN_ISSUES.md): excluded caches, external inputs, and checkpoint fallback rules.
