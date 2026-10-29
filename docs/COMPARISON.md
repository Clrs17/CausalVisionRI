# Common-standard comparisons

The comparison engine evaluates frozen predictions under a common sample, label source, time window, and metric definition within each panel. Subset and relabelled analyses are secondary evaluations of the locked predictions. They do not run another CVRI-Ens inference pass.

The entry point is:

```bash
python scripts/06_compare/compare_models_consistent.py
```

Outputs are written to `data/analysis/compare/` and `protocol_locks/comparison_secondary_<kind>.json`. Each lock records the sample definition, row-key hashes, threshold rules, checkpoint choices, input files, and fallbacks.

## 1. Evaluation panels

| Panel | Label and window | Sample definition | Rows and RI events |
|---|---|---|---:|
| Main holdout | USA 30 kt, 2020–2024 | Locked test predictions | 3,004; 222 |
| WRPS1 shared table | USA 30 kt, 2020–2024 | Main rows with all 12 SHIPS predictors finite | 2,892; 221 |
| Panel A | USA 30 kt, 2020–2024 | RITCF CMA-30 eligibility ∩ main holdout ∩ finite SHIPS-12 rows | 2,178; 209 |
| Panel A label control | CMA 30 kt, 2020–2024 | The Panel A rows, re-evaluated with CMA labels | 2,178; 120 |
| Panel B | CMA 25.27 kt, 2020–2021 | RITCF CMA-25.27 eligibility ∩ main holdout ∩ finite SHIPS-12 rows | 949; 66 |
| Independent RITCF frame | CMA 25.27 kt, 2020–2021 | Reconstructed RITCF sample frame before intersection with the main holdout | 1,057; 66 |
| Published RITCF anchor | Published CMA threshold, 2020–2021 | Original publication's reported frame | 1,149; 78 |

The 2,178-row USA panel contains 209 USA-labelled RI events and 120 CMA-labelled RI events. The two label assignments agree on 95.4% of those rows and have a Jaccard index of 0.530 on RI-positive rows.

CVRI-Ens and the statistical baselines were developed with USA labels. RITCF was reconstructed with CMA labels. Each label-control panel therefore measures performance under a stated cross-label condition.

## 2. Metrics and thresholds

`POD` is `TP/(TP+FN)`. `POFD`, also called `FARate`, is `FP/(FP+TN)`. `FARatio` is `1 − precision`, and `CSI` is the critical success index. ROC-AUC and PR-AUC use continuous scores.

`BSS_dev` uses the fixed development RI rate 0.0791023 for every model. Its reference Brier score is computed on the evaluated rows with that fixed rate, so models are compared under the same climatological reference.

Thresholds follow the evaluation label and the development split:

- **Panel A:** CVRI-Ens uses its locked 0.225 threshold; the other statistical baselines use their native inner-F1 thresholds. WRPS1 includes both its public `p_u ≥ 0.5` rule (`p_l ≥ 0.215`) and the inner-F1 rule at 0.15. RITCF includes its public 6-of-10 vote and validation-F1 points.
- **Panel B:** F1 thresholds are re-derived on 2018–2019 data using CMA 25.27-kt labels. CVRI-Ens uses 0.325 for the five-member inner ensemble. The same re-derivation rule applies to the baselines and WRPS1. RITCF retains its public 6-of-10 vote alongside its validation-derived point.
- **POD at 9.4% FARate:** the value is linearly interpolated from each model's empirical POD–FARate score sweep. Discrete markers show public or inner-F1 operating rules.

The published RITCF point is an external anchor with `n=1,149`. Its sample and production pipeline are separate from the shared-panel evaluation.

## 3. Checkpoint selection and file naming

The comparison engine supports three RITCF checkpoint choices:

```bash
python scripts/06_compare/compare_models_consistent.py --recon-kind best_auc
python scripts/06_compare/compare_models_consistent.py --recon-kind best_loss
python scripts/06_compare/compare_models_consistent.py --recon-kind last
```

`--recon-tag-suffix` defaults to `_aucfix`. Named outputs use `table_figA_<kind>.csv`, `table_figB_<kind>.csv`, `iso_pofd_<kind>.csv`, `coverage_<kind>.csv`, and `waterfall_<kind>.csv`, together with the corresponding figures and comparison locks.

The current paper products use these sources:

| Product | Source |
|---|---|
| Main Fig. 5 and the primary score-sweep comparison | Unsuffixed tables and `_best_loss` products, including `table_figB_best_loss.csv` |
| Supplementary Fig. S3, shared USA-30 curves | `_best_auc` score sweeps |
| Supplementary Fig. S4, CMA-30 label control | `table_figA_best_auc.csv` and `_best_auc` score sweeps |

Unsuffixed comparison tables preserve the primary best-loss products. Named files make the checkpoint criterion explicit and prevent one analysis from overwriting another.

For the 25.27-kt protocol, `sweep_table_aucfix_best_auc.csv` contains V4direct and omits V1–V4. The V1–V4 validation-F1 operating points therefore come from `sweep_table.csv`, while their best-AUC score sweeps come from the corresponding `_aucfix_best_auc.npz` files. V4direct has a best-AUC score sweep and operating table. The comparison lock records these choices and every fallback; it is the authoritative provenance for each plotted point.

Use `--figures-only` to redraw figures without changing the comparison tables or locks:

```bash
python scripts/06_compare/compare_models_consistent.py --figures-only
```

## 4. WRPS1 reconstruction

`scripts/06_compare/wrps1_same_sample.py` evaluates the published WRPS1 knowledge blocks on the released SHIPS-12 feature table. The reconstruction uses 73 published blocks, the stated offset, coefficient, probability mapping, and unit conventions.

Two typographical irregularities are handled explicitly. Printed row 55 duplicates row 1 and is counted once. Block number 36 occurs twice with different conditions, and both are retained. Shear values use 0.1 kt and temperature values use 0.1°C where required by the published formulation.

The continuous score `P` is used for the shared-sample BSS comparison. The public `p_u ≥ 0.5` rule supplies the corresponding operating point. The evaluation also reports a separate inner-F1 point.

WRPS1 was developed on a different data stream and its development window extends through 2021. Its sample overlap is recorded with the results rather than treated as a fully independent forecast comparison.

## 5. RITCF reconstruction variants

The independent RITCF pipeline uses HURSAT-B1 frames, ERA5 fields, and CMA best-track labels. [RITCF_RECON_SPEC.md](RITCF_RECON_SPEC.md) and [../scripts/ritcf_recon/README.md](../scripts/ritcf_recon/README.md) define its inputs and training commands.

| Variant | Purpose |
|---|---|
| V1 | Published-style shared-weight Siamese model with a rolling RI reference pool |
| V2 | V1 with the reference pool capped at 2019 |
| V3 | Released-repository skeleton with unshared branches and scalar outputs |
| V4 | Direct use of the learned pair distance without the 6-of-10 decision rule |
| V4direct | Direct classifier using the same input features |

The published description and released implementation differ in model details and normalization. The reconstruction follows the published description for those choices and records unspecified choices explicitly. [RITCF_RECON_SPEC.md](RITCF_RECON_SPEC.md) gives the deviations and the resulting sample frames.

## 6. Output map

| Output | Contents |
|---|---|
| `table_figA*.csv` | Panel A metrics, label-control metrics, thresholds, confusion counts, ROC/PR-AUC, and POD at 9.4% FARate |
| `table_figB*.csv` | Panel B metrics under CMA 25.27-kt labels |
| `iso_pofd*.csv` | Interpolated POD at the reference false-detection rate |
| `coverage*.csv` | RI and storm coverage by sample definition |
| `waterfall*.csv` | Row counts and exclusion reasons through each intersection |
| `fig_operating_points.*` | Main operating-point panels |
| `fig_compare_pr_best_auc.png`, `fig_compare_roc_best_auc.png` | Shared USA-30 precision–recall and ROC curves |
| `protocol_locks/comparison_secondary*.json` | Sample, row-key, checkpoint, input, and metric provenance |

[RESULTS.md](RESULTS.md) reports the resulting numbers. [SM_MAP.md](SM_MAP.md) links each paper figure to the corresponding table and generator.
