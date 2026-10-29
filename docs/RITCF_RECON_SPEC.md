# RITCF reconstruction

This document defines the independent reconstruction of the RITCF contrastive-learning scheme published by Wang, Yang and Li (2025), *PNAS* 122(4), e2415501122. The reconstruction evaluates the described architecture and a direct-classifier control on independently constructed samples.

The implementation is in `scripts/ritcf_recon/`. The consistent comparison with CVRI-Ens is handled separately by `scripts/06_compare/compare_models_consistent.py`, as described in [COMPARISON.md](COMPARISON.md).

## 1. Labels, population, and splits

The primary reconstruction label is:

```text
CMA_WIND(t+24h) − CMA_WIND(t) >= 25.27 kt
```

The wind is the CMA two-minute sustained-wind column in IBTrACS. The endpoint must belong to the same storm. The split follows the published period: 2000–2017 for training, 2018–2019 for validation, and 2020–2021 for testing.

The independent sample frame requires CMA wind at both endpoints, a valid centre, `DIST2LAND > 0`, and complete ERA5 and satellite windows. Rows with incomplete five-time inputs are excluded and counted.

| Split | Rows | RI events | RI rate |
|---|---:|---:|---:|
| Train, 2000–2017 | 10,144 | 579 | 5.71% |
| Validation, 2018–2019 | 1,387 | 93 | 6.71% |
| Test, 2020–2021 | 1,057 | 66 | 6.24% |

The publication reports 6,795/303, 1,472/110, and 1,149/78 for the corresponding splits. Its filtering information does not reproduce those counts under the independent rule, so the two sample frames are reported separately.

The `kt30` variant uses a CMA 30-kt label and a 2020–2024 test window. The `kt2524` variant retains the 25.27-kt threshold with the longer test window. These variants are recorded in their dataset manifests and are not interchangeable with the primary frame.

## 2. Input features

### Satellite

HURSAT-B1 v07b TBB frames are taken at `t0`, `−6`, `−12`, `−18`, and `−24 h`. Each frame is a 224×224 centre crop. The quality-controlled store resolves multi-satellite collisions deterministically and masks nonphysical pixels below 100 K or above 350 K. Frame times must match exactly.

### ERA5

The atmospheric input contains `u`, `v`, potential vorticity, and divergence at 1000, 850, 500, and 200 hPa, plus SST. All fields use the same five times as the satellite input.

A 25°×25° window is centred on the storm at `t0` and block-averaged to 1°. The local archive is already six-hourly, so no temporal interpolation is needed. Indices outside a storm's archive box are clamped to the nearest edge, and the number of affected rows is recorded.

### Track history

CMA wind and pressure are sampled at the five times. The five values and all ten pairwise differences for each variable give 30 history features. USA pressure is used when CMA pressure is unavailable, and the fallback is counted.

## 3. Normalisation

Wind components are mapped to `[−1,1]`. Other continuous fields are mapped to `[0,1]`. TBB is mapped linearly from 100–350 K to `[0,1]`.

Normalisation statistics use the training split only. ERA5 and history channels are filled with training means. Satellite frames use a per-frame mean at build time, with the pre-fill missing fraction recorded.

SAT and ERA5 caches use float16 to control storage. The cache manifest preserves the source and extraction information needed to rebuild them. The large `era5.npy` and `sat.npy` arrays are not included in the Zenodo package; `data/ritcf_recon/ACQUIRE.md` gives the rebuild route.

## 4. Model and training

The published-style model uses two weight-sharing branches and a 32-dimensional embedding. The distance between embeddings is trained with mean squared error, with positive pairs assigned target 0 and negative pairs target 1.

The publication and released implementation differ in embedding and normalisation details. The reconstruction follows the published description for those choices. Layer widths and some optimisation details are unspecified in the publication, so the implementation records the following local choices:

- SAT branch: three `3×3` convolution, ReLU, and max-pooling blocks, followed by a fully connected width of 128;
- ERA5 branch: two convolutional blocks, followed by width 128;
- history branch: multilayer perceptron with width 64;
- merged representation: hidden width 64, followed by the 32-dimensional embedding;
- Adam with learning rate `1e-3`, no weight decay, ReLU hidden activations, and a linear embedding output;
- fixed seed 2025 and a fixed 60-epoch budget;
- effective batch size approximating 256 pairs through micro-batching and gradient accumulation.

Each run stores three checkpoints from the same trajectory: minimum validation loss, maximum validation ROC-AUC, and the last epoch. The run manifest records the selected epochs, configuration, code and dataset hashes, and full training history.

## 5. Pair construction and inference

For training, each RI sample is paired with up to 21 earlier nearest RI samples to form positive pairs. Each non-RI sample is paired with its nearest earlier RI sample to form a negative pair. The two inputs are at least 24 h apart. Validation uses the same construction.

At inference, each target is paired with the ten nearest RI references known at the forecast time. The pair threshold is selected on validation by pair-level F1 by default. At least six positive votes among ten pairs produce an RI decision. A sample-level F1 threshold rule is also available as a sensitivity analysis.

| Variant | Definition |
|---|---|
| V1 | Shared-weight Siamese model with a rolling reference pool, including test-year labels known at forecast time |
| V2 | V1 with the reference pool capped at 2019 |
| V3 | Released-repository skeleton with unshared branches and scalar outputs |
| V4 | Direct use of the learned pair distance without the 6-of-10 vote |
| V4direct | Direct classifier with the same input features, trained with binary cross-entropy |

V4direct trains on the natural sample distribution. The Siamese arms train on a pair table with implicit resampling, so their comparison with V4direct measures the complete modelling pipelines rather than an isolated contrastive-learning effect.

## 6. Evaluation

The evaluation reports POD, FARate (POFD), FARatio, ROC-AUC, PR-AUC, yearly results, and Brier-based scores. `FARate` is `FP/(FP+TN)` and `FARatio` is `1 − precision`.

Hard-vote Brier scores use the 0/1 decision. Soft scores use validation-fitted Platt scaling for the test split. Vote fractions and distances are scores until a probability calibration is specified. ROC-AUC and PR-AUC do not depend on that calibration.

Under the best-AUC checkpoint criterion on the 1,057-row independent test frame:

| Variant | POD | FARate | ROC-AUC |
|---|---:|---:|---:|
| V1 | 0.879 | 0.465 | 0.725 |
| V2 | 0.924 | 0.549 | 0.709 |
| V3 | 0.909 | 0.518 | 0.739 |
| V4direct | — | — | 0.854 |

The published operating point is POD 0.910 at FARate 0.094. Across the full vote-threshold sweep, no point reproduces both values. An equal-variance binormal approximation requires an AUC of about 0.97 to attain that combination.

The test split contains 66 RI events, which gives an approximate 95% binomial uncertainty of ±11 percentage points for POD. The reconstruction uses one split and one seed; it does not provide a storm-cluster confidence interval for these standalone RITCF metrics.

## 7. Outputs and reproduction

The released products include:

- `samples.parquet`, pair tables, normalisation, and dataset manifests;
- run manifests and training histories without model weights;
- per-row score sweeps and operating tables for `best_loss`, `best_auc`, and `last`;
- per-variant evaluation JSON and Markdown summaries;
- comparison locks that record checkpoint and fallback choices.

[../scripts/ritcf_recon/README.md](../scripts/ritcf_recon/README.md) gives module-level commands. [COMPARISON.md](COMPARISON.md) defines the shared panels with CVRI-Ens and the source table for each figure. [RESULTS.md](RESULTS.md) reports the common-standard results.

The reconstruction's independent test frame and the shared comparison frame are distinct. The shared Panel B contains 949 of the 1,057 independent test rows after intersection with the locked CVRI-Ens predictions and finite SHIPS-12 predictors.
