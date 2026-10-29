# `era5.npy` and `sat.npy` are not in this record

Both are large binary feature caches rebuilt deterministically from primary
sources; they are deliberately excluded from the Zenodo record.

| file | shape | dtype | size on disk | included |
|---|---|---|---|---|
| `era5.npy` | [12588, 85, 25, 25] | float16 | 1337.48 MB | no |
| `sat.npy` | [12588, 5, 224, 224] | float16 | 6316.16 MB | no |

`sat.npy` exceeds 6 GB and `era5.npy` 1.3 GB; neither is needed to read any
reported number.  Everything that *is* needed is here: `samples.parquet`
(row keys, labels, split years), `pairs_train.parquet` / `pairs_val.parquet`,
`norm.json` (train-split normalisation statistics) and `manifest.json`
(per-storm funnel counts, clamped-window counts, NaN rates, and the
`array_sha256` / `row_key_sha256` fingerprints of the two caches).

## How to rebuild them

Run from the root of the `typhoon-ri` source release, after placing the raw
archives under `data/phase1/raw/` (`era5/<SID>/<SID>_{pl,sl_instant}.nc` from
the Copernicus CDS downloader, `hursat/<SID>/*.nc` from the HURSAT-B1 v07b
downloader — see `docs/DATA_DICT.md`):

```bash
python -m scripts.ritcf_recon.build_dataset --workers 8
```

The builder is fingerprint-checked: it recomputes the dataset fingerprint from
`config.py` + the source archives and refuses to reuse a directory whose
fingerprint moved, so a rebuild on the same inputs reproduces the same arrays.
`--report-only` reprints the funnel without touching the caches.

Reference values recorded when the released caches were built:

* `era5.npy` sha256 `ebd554bbb4839af93e8bb1e2cf6a58de25c20c883ff5fb4d66a2e3abb3dd6884`
* `sat.npy` sha256 `5b8f906a6e7b351de23eb40c0625359335e75a7d0ac98e14f1368ed17009e11a`

Because the caches are float16 and are normalised per channel with the released
`norm.json` statistics, any consumer that needs the *learned* model outputs can
stop here: the released `runs/*/eval*.json` and `sweep_*` files carry every
reported metric.
