# CausalVisionRI documentation

This release provides the code and derived artifacts for a 24-h rapid-intensification (RI) warning system in the western North Pacific. CausalVisionRI combines satellite infrared imagery, atmospheric and oceanic fields, and an observationally discovered causal graph.

The code repository is [Clrs17/CausalVisionRI](https://github.com/Clrs17/CausalVisionRI). The two data packages are published in [Zenodo record 23768442](https://doi.org/10.5281/zenodo.23768442).

## Start here

| Task | Document |
|---|---|
| Install the data packages and verify the release | [REPRODUCE.md](REPRODUCE.md) |
| Read the current holdout results | [RESULTS.md](RESULTS.md) |
| Check the sample boundaries and protocol locks | [PROTOCOL.md](PROTOCOL.md) |
| Inspect released files and columns | [DATA_DICT.md](DATA_DICT.md) |
| Trace a figure or table to its source and generator | [SM_MAP.md](SM_MAP.md) |
| Understand the common-standard comparisons | [COMPARISON.md](COMPARISON.md) |
| Rebuild the SHIPS-12 predictor table | [SHIPS12_BASELINE.md](SHIPS12_BASELINE.md) |
| Reconstruct the RITCF experiments | [RITCF_RECON_SPEC.md](RITCF_RECON_SPEC.md) and [../scripts/ritcf_recon/README.md](../scripts/ritcf_recon/README.md) |
| Resolve missing inputs and other practical limitations | [KNOWN_ISSUES.md](KNOWN_ISSUES.md) |

## Current analysis frame

The current protocol is `holdout_masked_v1`:

- **Fit, 2000–2017:** causal discovery, effect estimation, preprocessing, and model fitting.
- **Inner, 2018–2019:** early stopping, thresholds, and ensemble selection.
- **Test, 2020–2024:** one locked inference pass followed by metric evaluation.

The main test set contains 3,004 rows from 131 storms. Its RI rate is 7.39%. The primary score is `BSS_dev`, which uses the fixed 2000–2019 development RI rate of 0.0791023 as its reference.

## Reading conventions

Run commands from the repository root unless a document states otherwise. Paths of the form `data/...` become available after the two Zenodo packages are extracted there. [REPRODUCE.md](REPRODUCE.md) gives the exact setup steps.

`POD` is the probability of detection. `POFD` is the probability of false detection and is labelled `FARate` in the comparison figures. `FARatio` equals `1 − precision`. These are distinct quantities and appear in separate columns.

The two ablation analyses use explicit sign conventions:

- `YEAR_GRADIENT_ABLATION.md` reports `M0 − condition`, so a positive value favours the causal component.
- `WORST_YEAR_ABLATION.md` reports `condition − M0`, so a negative value favours the causal component.

`SM_MAP.md` links every paper exhibit to its source data and generator. `RESULTS.md` gives the corresponding numerical summary.
