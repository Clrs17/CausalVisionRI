# `data/` — populated from the Zenodo record

This directory is empty in a plain checkout.  Extract the two record packages
here:

```bash
cd <release root>                 # the directory holding src/, scripts/, configs/
unzip /path/to/record1_core.zip
unzip /path/to/record2_ritcf_recon.zip
sha256sum -c DATA_MANIFEST_record1_core.sha256
sha256sum -c DATA_MANIFEST_record2_ritcf_recon.sha256
```

After extraction you should see:

```
data/
├─ holdout_masked_v1/          phase-2 assets: panel, frames index, prior manifest,
│                              trusted signs, TBB norm, SHIPS-12 feature tables
├─ phase1/
│  └─ processed_masked/holdout_fit0017/
│     ├─ panel/                processed panel, split/sample manifests,
│     │                        imputation mask, DV24 + feature transforms
│     ├─ prior/                causal graph, ATE/CATE/headroom/regime, identification
│     │                        status, refutation report, DAG audits, trusted signs
│     └─ sensitivity/          strict / no_ohc / mjo_enso / lat variants
├─ test/holdout_masked_v1/     53 frozen npz + 53 manifest sidecars + test_receipt.json
├─ analysis/                   eval json/csv, compare tables, figures_pub, gate products
├─ protocol_locks/             identical copy of ../protocol_locks (self-describing record)
├─ runs_holdout/               baselines_manifest.json
├─ global_hursat/              QC and rekey manifests (index only, no pixels)
├─ ships12_predictors_alltc_0024.parquet + manifest + raw input hashes
└─ ritcf_recon/                RITCF sample frame, pairs, norm, sweeps, summaries
```

What is *not* here, and why: see `docs/DATA_DICT.md` §8 and
`docs/KNOWN_ISSUES.md` §4-§6 — raw ERA5/GLORYS/HURSAT archives, the 195 GB frame
store, deep-learning `*.pt` checkpoints, and the two multi-GB RITCF caches.
Fitted LR, LDA, and SHIPS-12 `model.pkl` baselines are included.

Licence: CC BY 4.0 (`LICENSE-DATA` at the release root).
