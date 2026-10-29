# Current holdout results

All values in this document refer to `holdout_masked_v1`. The main evaluation contains 3,004 frozen test rows from 2020–2024. Model fitting uses 2000–2017, and early stopping, thresholds, and ensemble selection use 2018–2019.

The primary metric is `BSS_dev`, with the fixed development RI rate 0.0791023. The main threshold is 0.225, selected by inner-period F1. Confidence intervals use 2,000 paired storm-cluster bootstrap resamples with seed 42. Holm adjustment is applied within the locked comparison families.

## 1. Main test performance

CVRI-Ens is the equal-weight ensemble of five members selected on inner data. Its pooled inner BSS is 0.332, compared with 0.316 for the best individual arm. On the test set, the five members score 0.151–0.214, while the ensemble scores 0.259.

| Model | BSS_dev | ROC-AUC | PR-AUC | POD | POFD | FARatio | CSI |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CVRI-Ens** | **0.259** | **0.900** | **0.518** | 0.680 | 0.095 | 0.635 | **0.311** |
| SHIPS-12 LDA, COHC | 0.105 | 0.820 | 0.276 | 0.347 | 0.060 | 0.686 | 0.197 |
| SHIPS-12 LDA, OHCD | 0.095 | 0.813 | 0.266 | 0.473 | 0.095 | 0.715 | 0.216 |
| 21-predictor LDA | 0.098 | 0.813 | 0.278 | 0.730 | 0.250 | 0.811 | 0.176 |
| Logistic regression | −2.074 | 0.826 | 0.298 | 0.649 | 0.173 | 0.769 | 0.205 |

The logistic-regression row is an uncalibrated reference. Its negative BSS is distinct from its discrimination metrics.

Paired bootstrap differences are positive for CVRI-Ens against every reference system:

| Contrast | ΔBSS_dev | 90% CI | Holm-adjusted p |
|---|---:|---:|---:|
| CVRI-Ens − SHIPS-12 COHC | +0.154 | [+0.079, +0.214] | 0.003 |
| CVRI-Ens − SHIPS-12 OHCD | +0.164 | [+0.095, +0.222] | 0.002 |
| CVRI-Ens − 21-predictor LDA | +0.161 | [+0.097, +0.215] | 0.002 |
| CVRI-Ens − logistic regression | +2.333 | [+1.935, +2.866] | 0.002 |

The source files are `data/analysis/eval/main_2024.json` and the corresponding baseline JSON files. `data/analysis/sm_tables.tex` contains the complete yearly and per-model tables.

## 2. Yearly results and ensemble sensitivity

| Year | Rows | BSS_dev | ROC-AUC | PR-AUC | POD | POFD | FARatio | CSI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 559 | 0.276 | 0.887 | 0.552 | 0.674 | 0.097 | 0.633 | 0.312 |
| 2021 | 757 | 0.298 | 0.897 | 0.594 | 0.659 | 0.073 | 0.642 | 0.302 |
| 2022 | 571 | 0.159 | 0.904 | 0.425 | 0.639 | 0.099 | 0.697 | 0.258 |
| 2023 | 606 | 0.287 | 0.901 | 0.543 | 0.660 | 0.076 | 0.545 | 0.368 |
| 2024 | 511 | 0.252 | 0.907 | 0.490 | 0.761 | 0.142 | 0.653 | 0.313 |

BSS_dev is positive in every test year and ranges from 0.159 in 2022 to 0.298 in 2021.

| Ensemble | BSS_dev | ROC-AUC | PR-AUC | POD | POFD | FARatio | CSI |
|---|---:|---:|---:|---:|---:|---:|---:|
| K4 | 0.259 | 0.898 | 0.513 | 0.667 | 0.087 | 0.621 | 0.319 |
| **Top-5, primary** | **0.259** | **0.900** | **0.518** | 0.680 | 0.095 | 0.635 | 0.311 |
| K6 | 0.279 | 0.902 | 0.527 | 0.662 | 0.076 | 0.589 | 0.339 |
| All 22 | 0.244 | 0.905 | 0.509 | 0.743 | 0.128 | 0.684 | 0.285 |

K4, K6, and all-22 are sensitivity analyses. The locked primary ensemble remains Top-5.

The lowest reliability bin, `[0, 0.1)`, contains 2,019 of 3,004 rows (67.2%). Its mean prediction is 0.049 and its observed RI rate is 0.011. The 0.225 threshold lies above this bin, so the bin does not change the thresholded metrics. The complete reliability table is in `main_2024.json`.

## 3. Causal-pathway ablation

The panel contains five members under eight conditions. C1 removes the causal losses while retaining the graph branch. C2 removes the graph neural-network branch and disables the causal losses. C3 replaces the discovered graph with degree-preserving shuffles under three seeds. C4 retains one of the three causal losses at a time.

Across the 40 member–condition contrasts, ten pass Holm correction and seven favour the causal configuration. Three significant contrasts are negative, showing that the effect is member-specific.

For the ensemble, removing the graph pathway reduces pooled BSS_dev by **0.032** over all 3,004 test rows. The yearly and pooled contrasts are:

| Period | M0 − C2, ΔBSS_dev | 90% CI | p_boot |
|---|---:|---:|---:|
| 2020 | +0.080 | [+0.043, +0.140] | <0.001 |
| 2021 | +0.047 | [+0.023, +0.076] | 0.003 |
| 2022 | +0.029 | [−0.010, +0.089] | 0.258 |
| 2023 | −0.007 | [−0.045, +0.036] | 0.808 |
| 2024 | +0.017 | [−0.029, +0.053] | 0.489 |
| **2020–2021 pooled** | **+0.063** | **[+0.042, +0.090]** | **<0.001** |

All five members improve in 2020, and four improve in 2021. The 2020–2021 C1 contrast is +0.007 with a 90% CI of [−0.013, +0.026], so removing the losses alone leaves ensemble skill statistically unchanged in that period. These contrasts separate the graph pathway from the auxiliary losses and from the shuffled-topology controls.

At member level, LoRA-sf has the largest C2 effect, +0.0796 with Holm-adjusted `p=0.004`. The other four C2 intervals include zero. The complete 40-row panel is in Supplementary Table S5 and in `data/analysis/eval/M0_*.json`. `YEAR_GRADIENT_ABLATION.md` gives the ensemble and member contrasts by year.

Gate diagnostics describe how the members use the prior. Part4 assigns 68.9% of its context attention to the causal token, while LoRA-sf assigns 23.0%. These are descriptive fusion measurements, not estimates of mediation. `WORST_YEAR_ABLATION.md` gives the complementary worst-year analysis.

## 4. Common-standard comparisons

Every comparison within a panel uses the same rows, evaluation labels, time window, and metric definitions. [COMPARISON.md](COMPARISON.md) documents the threshold and checkpoint rules.

### WRPS1 on the shared feature table

On 2,892 test rows with all twelve SHIPS predictors finite, CVRI-Ens has BSS_dev 0.260, compared with 0.143 for the continuous WRPS1 knowledge-block score. The paired difference is +0.117 with a 90% CI of [+0.045, +0.174] and `p=0.010`. WRPS1's development window extends through 2021 and is reported with this comparison.

### USA 30-kt labels, 2020–2024

The shared panel contains 2,178 rows and 209 RI events, a base rate of 9.60%. At a 9.4% false-detection rate, interpolated POD is **0.670** for CVRI-Ens and **0.517** for the next-best entry. These values come from `iso_pofd.csv`.

The CMA 30-kt label control retains the same 2,178 rows and contains 120 RI events. Under the best-AUC comparison table, CVRI-Ens has the highest interpolated POD at 9.4% FARate, 0.675 versus 0.567 for the next-best entry, and the highest CSI, 0.260. Its uncalibrated BSS_dev is 0.092, compared with 0.134 for WRPS1. The source is `table_figA_best_auc.csv`.

### CMA 25.27-kt labels, 2020–2021

The shared panel contains 949 rows and 66 RI events, a base rate of 6.95%. CVRI-Ens ranks first in ROC-AUC at 0.896 and PR-AUC at 0.562. Interpolated POD at 9.4% FARate is **0.727**, compared with **0.636** for the next-best entry. These values come from `table_figB_best_loss.csv` and the corresponding score sweeps.

The published RITCF operating point is POD 0.910, FARate 0.094, and FARatio 0.587 on 1,149 rows. It is an external reference point with a different sample frame, as [COMPARISON.md](COMPARISON.md) explains.

## 5. Sample frames

| Frame | Rows | Purpose |
|---|---:|---|
| Main holdout | 3,004 | Primary test metrics |
| WRPS1 shared feature table | 2,892 | WRPS1 and CVRI-Ens comparison with finite predictors |
| RITCF CMA 30-kt reconstruction | 2,385 | Independent reconstruction frame, 2020–2024 |
| USA 30-kt common panel | 2,178 | Common-row comparison under the project standard |
| 2020–2021 subset | 1,316 | Supplementary two-year ablation analysis |
| Published RITCF operating point | 1,149 | External reference from the original publication |
| RITCF CMA 25.27-kt reconstruction | 1,057 | Independent reconstruction frame, 2020–2021 |
| CMA 25.27-kt common panel | 949 | Common-row comparison under the published threshold |

These counts describe different evaluation frames and are not interchangeable. [COMPARISON.md](COMPARISON.md) gives the intersections and label sources, while `data/analysis/compare/coverage*.csv` and `waterfall*.csv` record row-level coverage.
