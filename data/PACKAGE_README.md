# TyphoonRI data record

This archive is one record of the TyphoonRI / CausalVisionRI publication
package.  Unzip it **into the root of the `typhoon-ri` source release**
(the directory that contains `src/`, `scripts/`, `configs/`, `docs/`), so that
the `data/...` paths written in the code and in the protocol locks resolve.

Every file is listed in the accompanying `DATA_MANIFEST_<record>.sha256`
file (`<sha256>  <relative path>`, `sha256sum -c` compatible); the matching
`PACKAGE_<record>.json` carries the same inventory as JSON.

Scope and licence: see `LICENSE-DATA` (CC BY 4.0).  Redistributed here are
**derived quantities only** — processed panels, causal-effect estimates,
frozen test-set predictions and figure sources.  Raw ERA5 / GLORYS12 /
HURSAT-B1 / IBTrACS fields are **not** redistributed; they are re-downloadable
through `scripts/01_data/` (see `DATA_DICT.md` §8).
