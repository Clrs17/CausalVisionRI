# tests/

Two entry points, both CPU-only and offline.

| file | what it does | cost |
|---|---|---|
| `verify_protocol_locks.py` | audits the whole chain, in 14 verification lines across eight logical groups: `final.json` and `selection.json` self-hashes; the `final.selection_sha256` link; the receipt binding and `completed` status; the data-record `selection.json` (self-hash, byte-identity with the root copy, no residual `../wpac_causal_phase1`); the populated `asset_sha256_manifest.json` re-verified against the shipped files; every `*_sha256` in the locks and data sidecars that names a shipped file, with path-normalised mismatches accepted only when the asset manifest declares them; and the absence of a build-machine absolute path in **both** the text and the binary files (parquet metadata, pickle headers, npz member names) | ~3-20 s |
| `test_release.py` | pytest wrapper: directory layout, no absolute paths in `src/`/`scripts/`/`configs/`, YAML parses, the RITCF report without large caches, the headline BSS_dev recomputed from `main.npz` (asserts 0.258973 ± 1e-5), the frozen npz accepted-lock set, panel/manifest agreement, the selection-lock self-consistency and chain, the asset manifest re-verification, and a byte-level sweep of `parquet/npz/pkl/npy` for build paths | ~15 s |

```bash
python tests/verify_protocol_locks.py          # readable report, exit 0 = pass
python -m pytest tests/ -q
```

`test_release.py` **skips** the main data-dependent tests when
`data/test/holdout_masked_v1/main.npz` is absent. The RITCF report test is
skipped separately when its released `samples.parquet` is absent, so a
code-only checkout still passes. `verify_protocol_locks.py` needs the data
record for its hash and
binary checks — without it you get the lock self-hashes and the layout checks,
and the pinned hashes report "describe artefacts not in the record" instead of
a mismatch.

Neither file needs network access, a GPU, or any package beyond numpy / pandas /
PyYAML, plus `matplotlib`/`sklearn` if you go on to run the figure and
comparison scripts.

## What "normalised" means in the asset manifest

`protocol_locks/asset_sha256_manifest.json` splits pinned hashes into three
buckets, and the verifier treats them differently:

* `entries` — the recorded digest equals the shipped file.  Re-verified.
* `normalised` — the recorded digest is the hash of the file **as produced**,
  and it was re-verified against the working tree at build time; the shipped
  copy differs only because absolute paths inside it were rewritten.  The
  manifest carries both hashes, and the verifier checks the shipped one.
  Today this covers the 62 per-arm `config.yaml` snapshots (22 development arms
  + 40 ablations).
* `not_redistributed` — model weights, raw field archives and the two multi-GB
  RITCF caches.  Counted, never failed.
