# Usage

Run commands from your project directory. All input paths are explicit. New calculations require new output locations; collection alone supports an explicit `--resume`.

## Small executable example

This generates synthetic inputs to exercise installation and the workflow. It is not a calibration benchmark.

```bash
python examples/make_demo.py runs/demo
stitch prepare --metadata runs/demo/metadata.parquet --labels runs/demo/labels.parquet \
  --splits runs/demo/splits.parquet --out runs/demo/prepared
stitch train-hgb --prepared runs/demo/prepared --identity-catalog runs/demo/identity.parquet \
  --config configs/hgb_smoke.json --out runs/demo/model
stitch predict-hgb --model runs/demo/model --features runs/demo/prepared/features.parquet \
  --prepared --curves runs/demo/curves.parquet --out runs/demo/prediction
stitch dashboard --predictions runs/demo/prediction/predictions.parquet \
  --curves runs/demo/prediction/curves.parquet --out runs/demo/dashboard.html
```

## Download frozen assets

```bash
python tools/download_models.py --bundle hgb --bundle cnf --bundle geometry --out artifacts
stitch verify-model artifacts/hgb
stitch verify-model artifacts/cnf
```

The [manifest](../models/manifest.json) pins an immutable Git commit and bundle checksums. Downloads are limited to two at a time; existing verified assets are reused. Model assets occupy approximately 343 MiB, including the PRF calibration grids. These are separate from light-curve data. The downloader retains a 4 GiB free-disk reserve.

## 1. Collect light curves

Create `requests.parquet` or `requests.csv` with:

| Column | Meaning |
| --- | --- |
| `source_row` | Optional unique nonnegative row ID; generated if omitted |
| `tic_id`, `sector` | Exact requested identity |
| `data_uri` | MAST archive URI or HTTPS archive product URL |
| `cached_path` | Optional existing FITS or gzipped FITS path |
| `expected_raw_sha256` | Optional checksum of uncompressed FITS bytes |
| `systematic_score` | Optional individually matched TARS sector score |

Select the intended extraction family explicitly when constructing requests. The frozen HGB uses TESS-SPOC PDCSAP; collected SAP is a separate measurement product, not a validated substitute for that model's inputs.

```bash
stitch collect --requests requests.parquet --out runs/collection --workers 2
# Resume the same immutable request table after an interruption:
stitch collect --requests requests.parquet --out runs/collection --workers 2 --resume
stitch curves --collection runs/collection --out runs/curves.parquet
```

The collector verifies FITS TIC/sector identity, records separate SAP and PDCSAP measurements for `QUALITY == 0` and the default bitmask `17087`, and persists cadence tables, header/aperture metadata and receipts before releasing raw bytes. Neither SAP nor PDCSAP should be assumed to have all bad-quality cadences removed automatically. `stitch curves` exports finite positive PDCSAP measurements with `QUALITY == 0`.

At most two raw products are in progress. No new FITS cache is retained. A 4 GiB output limit and 4 GiB free-disk reserve apply by default. For larger surveys, use bounded request batches and retain their receipts. Existing caches are never deleted. A failed product is recorded under `failures/`; a successful receipt takes precedence on resume.

## Physical features

Provide a TIC catalog cone table named `<TIC>_cone.ecsv` for each target. It must contain `ID`, `GAIA`, `ra`, `dec`, `RA_orig`, `Dec_orig`, `POSflag`, `pmRA`, `pmDEC`, `disposition`, `Tmag`, `gaiarp`, `gaiabp`, `contratio`, and `numcont`. Include neighboring catalog sources, not only the target. Native Gaia positions and proper motions are propagated to each sector. Catalog acquisition and completeness remain the caller's responsibility.

```bash
stitch features --collection runs/collection --catalog-dir catalogs \
  --thermal artifacts/hgb/thermal.parquet --prf-dir artifacts/geometry/prf \
  --strap-path artifacts/geometry/strap_columns.csv --out runs/physical.parquet
```

The thermal table requires `sector`, `cam`, `ccd`, `focal_plane_temp`, and `focal_plane_temp_std`. Missing thermal measurements remain missing for fitted imputation; they are not invented. PRF grids must cover the requested camera, CCD and era. The bundle contains the verified historical grids; missing coverage fails explicitly. The PRF interpolation follows TESS-PRF 0.1.3; its [MIT attribution](third_party/TESS_PRF_LICENSE.txt) is retained.

## 2. Frozen HGB inference

```bash
stitch predict-hgb --model artifacts/hgb --features runs/physical.parquet \
  --catalog-dir catalogs --curves runs/curves.parquet --out runs/hgb
```

New-source inference checks catalog identity against the historical cohort and attaches the frozen source-excluded references. It never fits a target from query flux. An existing cohort star must be evaluated through its correctly prepared historical features rather than passed off as a new source.

For verified, already prepared feature tables, use `--prepared` instead of `--catalog-dir`. Such tables must contain the exact feature definitions and reference values appropriate to the model; column names alone do not establish scientific validity.

The output contains `predictions.parquet`, optional corrected `curves.parquet`, and `provenance.json` with all-sector/qualified coverage, missing-anchor counts, reference support, imputation counts and extraction-family flags. HGB also reports its historical range checks. Its inherited `science_supported` column is only a basic range/reference screen, not validated science accuracy. Unknown product families are flagged as unverified transfers. Corrections divide the observed flux. Scaled formal cadence errors exclude calibration and extraction uncertainty.

## 3. Frozen CNF inference

```bash
stitch predict-cnf --model artifacts/cnf --features runs/physical.parquet \
  --catalog-dir catalogs --curves runs/curves.parquet --out runs/cnf
```

The supplied CNF is the historical pairwise NSF, with its exact conditional-median estimator, saved target scaling, beta and `[0.85, 1.15]` clipping. Its application reference bank is separate from HGB's. It was selected among four archived candidates on eight calibration stars and is not a newly trained Gaia-RP model.

## Train new models

Training requires three matched Parquet tables with unique `source_row`, `tic_id`, `sector` keys:

* **Metadata:** physically verified predictors, including the derived fields produced by `stitch features`; no flux target is consumed as a predictor.
* **Labels:** `source_row`, `tic_id`, `sector`, `sector_median`, measured with a documented product and quality rule.
* **Splits:** the keys plus `role` (`train`, `validation`, or `test`), `physical_group_id`, and the sector's `systematic_score`. Resolve catalog aliases before assigning these groups. Optional metadata `training_reference_eligible` can quarantine additional observations.

Source groups occurring in validation/test are excluded from training references. Training uses individually qualified sectors from eligible stars with at least two such observations. Five outer folds exclude query groups from both reference statistics and the color transfer. The validation groups are split deterministically into calibration and confirmation; fitting choices use calibration only. No test labels enter fitting or references.

```bash
stitch prepare --family hgb --metadata metadata.parquet --labels labels.parquet \
  --splits splits.parquet --out runs/prepared_hgb
stitch train-hgb --prepared runs/prepared_hgb --identity-catalog identity.parquet \
  --out runs/new_hgb

stitch prepare --family cnf --metadata metadata.parquet --labels labels.parquet \
  --splits splits.parquet --out runs/prepared_cnf
stitch train-cnf --prepared runs/prepared_cnf --identity-catalog identity.parquet \
  --epochs 60 --out runs/new_cnf
```

`identity.parquet` contains `ID`, `GAIA`, `ra`, and `dec` for the full historical/training cohort used by the new model. It is retained for future source-collision checks. Each preparation must have enough supported, high-crowding training stars for its color transfer; a tiny arbitrary sample may fail this requirement.

New HGB and CNF targets default to Gaia RP. Missing anchors use an explicitly recorded relative fallback; HGB retains the Gaia anchor in every one of five rounds. `train-cnf --target relative` is an explicit control, never an implicit fallback for an entire run. New CNF training uses the published architecture and training-only preprocessing, but is not claimed to reproduce the historical checkpoint's exact training procedure or performance. The historical checkpoint remains unchanged.

## 4. Dashboard

```bash
stitch dashboard --predictions runs/hgb/predictions.parquet \
  --curves runs/hgb/curves.parquet --model-spec artifacts/hgb/selected_model.json \
  --out runs/hgb_dashboard.html
```

Open the standalone HTML file. Click a star to see the sector summary plus input and corrected full-light-curve panels. Qualification applies per sector; unknown scores remain explicitly unqualified. Display normalization is global to each plotted series and does not alter model targets or stored fluxes. Absolute-flux display is available. The renderer performs no downloads or model fitting and emits a separate coverage CSV.
