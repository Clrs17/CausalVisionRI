# Evaluation protocol and audit chain

The current protocol is `holdout_masked_v1`. It fixes the sample boundaries, causal priors, model selection, test inference, and statistical evaluation used in the paper.

## 1. Data and training boundaries

| Stage | Years | Purpose |
|---|---|---|
| Fit | 2000–2017 | Causal discovery, effect estimation, preprocessing, pretraining, and model fitting |
| Inner | 2018–2019 | Early stopping, thresholds, and ensemble selection |
| Test | 2020–2024 | One locked inference pass and subsequent metric evaluation |

Storm identity, not individual row time, determines the split. Each SID belongs to one storm year and one split. A 30 h label embargo at each boundary excludes rows whose `F24_TIME` reaches the next split.

The canonical population contains 725 western North Pacific systems, with 720 label-eligible systems. The fit-only causal analysis uses 522 systems and 12,647 six-hourly observations. After frame-availability filtering, the model datasets contain:

| Split | Systems | Rows |
|---|---:|---:|
| Fit | 522 | 12,619 |
| Inner | 66 | 1,818 |
| Test | 131 | 3,004 |

The frame-availability rule excludes 202 rows with a recorded reason. It excludes a row only when its exact `t0` frame is unavailable in the quality-controlled store. Earlier environmental gaps are handled through the model's environment mask rather than by an additional row exclusion.

## 2. Fit-only causal and input artifacts

Masked PCMCI+ uses `mask_type=xyz`, `tau_max=4`, and `pc_alpha=0.01`. The discovery output contains 97 directed links. The retained training graph contains 87 links at the declared significance threshold: 6 contemporaneous and 81 lagged links.

The identification DAG includes all 97 candidates when adjustment sets are derived. Eight effects enter the sign and counterfactual constraints after passing the storm-block bootstrap, refutation, and directed-path requirements. D200 is excluded from those constraints by the pre-specified identification rule.

The following transformations are fitted on 2000–2017 and then applied to the inner and test data:

- intensity-change transformation and environmental scaling;
- median imputation and the recorded imputation mask;
- TBB normalisation;
- daily OHC selection using completed source windows;
- fit-only monthly, gridded COHC climatology;
- SHIPS predictor imputation and scaling.

Every prior package carries input and output hashes. A missing or inconsistent manifest stops asset construction or model loading.

## 3. Visual pretraining and model selection

The self-supervised routes use fit-period frames for training and inner-period frames for validation. Test frames are excluded from the pretraining index. The multi-satellite raw corpus contains 304,518 frames; the downstream quality-controlled set contains 153,295 unique `(storm, time)` frames.

The WNP MAE route reaches its best validation checkpoint at epoch 96 and the global route at epoch 75. External LingBot checkpoints provide the other registered initialization routes.

Twenty-two development arms are ranked by the worse of their 2018 and 2019 inner BSS values. A family gate protects a family champion only when it ranks within the overall top eight. Remaining places are filled by overall rank. The top five members are averaged with equal weights to form CVRI-Ens. K4, K6, and all-22 are sensitivity analyses.

The primary threshold, 0.225, maximizes inner-period F1. The 40 ablation entities combine the five members with eight conditions while holding member initialization and shuffle generators fixed.

## 4. Lock and inference sequence

The production sequence is:

```text
22 development arms
        ↓
Inner-only ensemble selection
        ↓
40 ablations and 4 statistical baselines
        ↓
protocol_locks/final.json
        ↓
One production test inference and signed receipt
        ↓
Metrics, ablations, comparisons, figures, and tables
```

`protocol_locks/selection.json` records the inner ranking, family gate, selected members, and thresholds. `protocol_locks/final.json` fixes 49 inference entities, four baselines, the primary threshold, the evaluation policy, and the primary contrast `C2_vs_M0_BSS_dev`.

`data/test/holdout_masked_v1/test_receipt.json` records `status: completed`, one entry per inference entity, and the hash of every prediction file. Production selection, finalisation, and inference refuse to rerun once their completion guards are satisfied. A custom output directory does not reset the production guard. Fixtures are supported only through the explicit fixture mechanism.

## 5. Evaluation and uncertainty

The primary score is:

```text
BSS_dev = 1 − BS / BS_clim
BS_clim = π² + (1 − 2π) × observed_event_frequency
π = 0.07910230657338782
```

The development rate is fixed across all evaluated models. ROC-AUC, PR-AUC, POD, POFD, FARatio, and CSI are reported alongside BSS. `FARate` is synonymous with POFD; `FARatio` equals `1 − precision`.

Uncertainty uses 2,000 paired storm-cluster bootstrap resamples with seed 42. The main intervals are 90% confidence intervals, with Holm adjustment at `alpha=0.10` inside complete, pre-declared comparison families. The evaluator rejects incomplete families so that contrasts cannot be removed after the results are known.

The threshold and ensemble are fixed before test evaluation. Test-period analysis consists of scoring the frozen predictions, secondary subset or relabelling analyses, and descriptive diagnostics.

## 6. Lock files and receipt

| Artifact | Role |
|---|---|
| `protocol_locks/selection.json` | Inner ranking, selection rule, members, and thresholds |
| `protocol_locks/final.json` | Entity list, primary threshold, evaluation contract, and artifact hashes |
| `protocol_locks/test_receipt.json` | Completion status and one receipt per prediction file |
| `protocol_locks/comparison_secondary*.json` | Secondary sample definitions, row keys, comparison rules, and input provenance |
| `protocol_locks/asset_sha256_manifest.json` | Verification of released and non-redistributed artifacts |

The lock self-hash is computed from the JSON payload with `lock_sha256` removed, using the canonicalisation in `src/wpac_warning/protocol.py`. `final.json` records the selection lock's hash, and the receipt records the final lock's hash. This creates a checkable selection → final → receipt chain.

## 7. Release path normalisation

The locks were created with absolute paths on the training host. The release converts path fields to release-relative paths and records the re-issue. It preserves the numerical evaluation contract and the hashes of non-redistributed artifacts.

| File | Normalised path strings | Re-issued lock SHA-256 |
|---|---:|---|
| `final.json` | 393 | `78c522b320584de0d687dcb56d6c7e44c9f2626bd7f124ea9f9739caf8a1baec` |
| `selection.json` | 27 | `f5d8e4622da4d7ff1c25e03f0b1d8904569a219f98a29d3c9b96893b77c0491f` |
| `test_receipt.json` | 107 | Bound to the re-issued final lock |

`release_reissue` records the predecessor hashes and rewrite counts. `protocol.accepted_lock_sha256(lock)` accepts exactly the current lock hash and its recorded predecessor. Frozen NPZ files embed the predecessor and therefore continue to verify without changing their contents.

The 62 released per-arm configuration snapshots have three hashes in the asset manifest: the originally issued hash, the hash verified against the source file, and the hash of the shipped path-normalised file. `config_json_sha256` is a separate hash of the resolved in-memory configuration and is preserved verbatim.

Path normalisation also covers binary metadata, including the pandas and Arrow schema metadata in `frames_index.parquet`. `tests/verify_protocol_locks.py` checks textual and binary path references, lock self-hashes, receipt linkage, and artifact hashes.

`data/protocol_locks/` is the data record's copy of the lock directory. The selection lock and receipt are byte-identical to the root copies. Comparison locks are JSON-equivalent and may differ in line endings, so the data record remains self-describing.

## 8. Secondary comparisons

`scripts/06_compare/compare_models_consistent.py` evaluates subset and relabelled analyses over frozen predictions. It does not perform another CVRI-Ens inference pass. Comparison locks record the sample row-key hashes, label sources, threshold rules, checkpoint inputs, and relationship to the official 3,004-row table.

The release entry point is `scripts/06_compare/compare_models_consistent.py`. The original engine script and hash remain provenance, while `spec` and `design` point to the current documentation. `--figures-only` redraws figures without changing comparison tables or the existing lock.

[COMPARISON.md](COMPARISON.md) defines the panel intersections, threshold re-derivation, checkpoint choices, and fallback rules. [REPRODUCE.md](REPRODUCE.md) gives the commands for verifying the chain.

## 9. Verification commands

```bash
python tests/verify_protocol_locks.py

python scripts/05_eval/eval_holdout.py \
    --config configs/phase2/holdout_masked_v1.yaml \
    --pred data/test/holdout_masked_v1/main.npz \
    --lock protocol_locks/final.json \
    --out /tmp/eval_main
```

The second command recomputes `BSS_dev = 0.258973`, `ROC_AUC = 0.899712`, and `PR_AUC = 0.517783` on 3,004 rows from the frozen probabilities.
