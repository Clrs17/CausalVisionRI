# SHIPS-12 baselines

The SHIPS-12 baselines use the twelve predictors listed in WRPS1 Table 1, reconstructed on this project's storm-aligned panel. Two ocean variants are evaluated independently: time-varying daily OHC (OHCD) and fit-only climatological OHC (COHC).

The predictor definitions follow WRPS1 and the SHIPS predictor-file conventions. The fitted LDA baselines use the same fit, inner, and test boundaries as CVRI-Ens. WRPS1's separate knowledge-block scheme is described in [COMPARISON.md](COMPARISON.md).

## 1. Data and evaluation frame

The source predictor table is `data/ships12_predictors_alltc_0024.parquet`. Its manifest is `data/ships12_predictors_alltc_0024.manifest.json`.

The table covers the canonical 725-system population and the quality-controlled frame-availability rules used by the deep-learning models. `frame_availability_v1` excludes only rows whose exact `t0` frame is unavailable. Excluded rows are recorded with their SID, time, split, and reason.

Fit-period rows supply medians for missing predictors and statistics for scaling. Inner-period rows select the F1 threshold. Test-period scoring uses the locked evaluation chain described in [PROTOCOL.md](PROTOCOL.md).

On the 3,004-row main holdout, the two LDA variants give:

| Variant | BSS_dev | ROC-AUC | PR-AUC | POD | POFD | FARatio | CSI |
|---|---:|---:|---:|---:|---:|---:|---:|
| SHIPS-12 LDA, COHC | 0.105 | 0.820 | 0.276 | 0.347 | 0.060 | 0.686 | 0.197 |
| SHIPS-12 LDA, OHCD | 0.095 | 0.813 | 0.266 | 0.473 | 0.095 | 0.715 | 0.216 |

The complete metrics and yearly values are in `data/analysis/eval/baseline_ships12_lda*.json` and [RESULTS.md](RESULTS.md).

## 2. Predictor definitions

Environmental fields are matched to the row's `t0` synoptic time. Area averages use great-circle distance with latitude-adjusted grid geometry and valid grid cells. Radii follow WRPS1 Table 1.

| # | Predictor | Definition and radius | Source in this release | Fidelity |
|---|---|---|---|---|
| 1 | `VMAX` | Maximum sustained wind, kt | `USA_WIND`, 1-minute wind | EXACT |
| 2 | `INCV` | Intensity change from `t−6h` to `t0`, kt | Panel `PER`, calculated before six-hourly resampling | EXACT |
| 3 | `POT` | Potential intensity minus `VMAX`, kt | `tcpyPI` 1.4.1 with the Bister–Emanuel algorithm, using the local ERA5 profile and sea-surface fields | APPROX |
| 4 | `OHCD` | Time-varying ocean heat content, kJ cm⁻², 0–100 km | Daily GLORYS12V1 above the 26°C isotherm | BETTER |
| 4′ | `COHC` | Climatological ocean heat content, kJ cm⁻², 0–100 km | Fit-only GLORYS12V1 month-by-0.5° climatology | NEAR-EXACT |
| 5 | `SHRD` | 850–200 hPa shear magnitude, kt, 200–800 km | Annular ERA5 `u` and `v` means | EXACT |
| 6 | `T150` | 150 hPa temperature, °C, 200–800 km | Log-pressure extrapolation using 200 and 300 hPa | SUBSTITUTED |
| 7 | `T200` | 200 hPa temperature, °C, 200–800 km | ERA5 temperature at 200 hPa | EXACT |
| 8 | `T250` | 250 hPa temperature, °C, 200–800 km | Log-pressure interpolation using 200 and 300 hPa | INTERP |
| 9 | `R000` | 1000 hPa relative humidity, %, 200–800 km | ERA5 relative humidity at 1000 hPa | EXACT |
| 10 | `SDIR` | Infrared brightness-temperature standard deviation, °C, 0–200 km | Quality-controlled HURSAT-B1 TBB crop | NEAR-EXACT |
| 11 | `Z850` | 850 hPa relative vorticity, 10⁻⁷ s⁻¹, 0–1000 km | Spherical central differences of ERA5 `u` and `v` at 850 hPa | EXACT |
| 12 | `DTL` | Great-circle distance to the nearest coastline, km | Panel `LAND_DIST` from the GSHHG coastline | APPROX |

Fidelity labels have the following meanings:

- **EXACT:** direct field or an equivalent computed with the stated operator.
- **NEAR-EXACT:** close correspondence with a documented source difference.
- **INTERP:** interpolation in log-pressure coordinates.
- **APPROX:** a local approximation to the SHIPS definition.
- **SUBSTITUTED:** the requested level is unavailable locally and is derived from neighbouring levels.
- **BETTER:** time-varying daily OHC replaces the climatological WRPS1 input.

The potential-intensity calculation uses `tcpyPI` 1.4.1 with `CKCD=0.9`, `V_reduc=1.0`, and dissipative heating enabled. The local ERA5 top level is 200 hPa, so the profile is extended in log-pressure coordinates and with an isothermal tropical-tropopause-layer extension. This produces the best locally available estimate rather than an exact SHIPS input field.

OHCD uses the daily GLORYS mean for `[D 00:00, D+1 00:00)`. A daily field is eligible only when `source_time + 24 h <= ISO_TIME`, ensuring that the complete day is available at the row time. The selected source time is recorded.

COHC pools only fit-period GLORYS fields into a month-by-0.5° climatology. It differs from the NCODA 2005–2010 climatology used by SHIPS, and the source files are hashed.

`T150` is extrapolated because the local archive has no 150 hPa level. `T250` is interpolated in log-pressure coordinates. `SDIR` uses HURSAT-B1 rather than the native GOES product used by SHIPS. `DTL` measures distance to the full coastline, which can differ from the SHIPS major-landmass convention.

## 3. Build and validation

Recompute the source table with:

```bash
python scripts/01_data/baseline_ships12.py --recompute
```

The build validates the population, frame-availability ledger, quality-controlled frame store, and raw-input hashes. A missing or unreadable required input stops the build. A frame-availability exclusion does not trigger a storm failure because those rows are outside the computation set.

The source manifest records:

- canonical population and protocol identifiers;
- row keys and per-system counts;
- SHA-256 values for the panel, frame index, build code, and raw inputs;
- the quality-control rule and frame-store fingerprints;
- OHCD source-window rules and COHC fit-only provenance;
- the frame-availability file and its row-key hashes.

Fit-period feature tables are produced separately for OHCD and COHC. The two ocean variants never enter the same fitted model. The 21-predictor panel LDA and class-balanced logistic regression use the same evaluation boundaries and are reported separately in [RESULTS.md](RESULTS.md).

## 4. Practical limitations

`POT` depends on the local vertical-profile extension. `T150` and `T250` depend on extrapolation or interpolation. `SDIR` measures a different satellite product from the SHIPS development data. `COHC` uses a different ocean climatology and sampling distribution. `DTL` uses all coastlines rather than only major landmasses.

The class-balanced logistic-regression probabilities are uncalibrated. Its negative BSS_dev is reported alongside its discrimination metrics in [RESULTS.md](RESULTS.md).

The source-table and feature-table manifests are required for reproduction. `tcpyPI` is needed only when recomputing `POT`; the released predictor tables already contain it.
