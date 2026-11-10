# Applying the frozen physical STITCH model

Use `infer_physical.py` for the October 12, 2026 five-round Gaia-RP physical
HGB. `infer_star.py` is the historical normalizing-flow application. The HGB
remains a research artifact; this entry point does not promote or retrain it.

## Local exact-product application

From the repository root:

```bash
python3 infer_physical.py \
  --products eval/prospective_ground_validation_20261012/products.parquet \
  --catalog-dir eval/prospective_ground_validation_20261012 \
  --tars eval/prospective_ground_validation_20261012/full_tars_lookup.parquet \
  --tics 230384382 233547261 219773818 \
  --out eval/my_new_science_application
```

Choose a **new output directory** for every application. Existing directories
are rejected, including interrupted runs. Repeatedly examined examples above
are development cases, not fresh confirmation sources.

The input Parquet manifest needs `tic_id`, `sector`, `path`, and `sha256`.
`path` points to local uncompressed or losslessly gzip-compressed FITS files;
relative product paths resolve against the repository root. Hashes describe
the actual local bytes. There must be exactly one product per TIC/sector in
one extraction family. Each source needs its full TIC scene catalog at
`<catalog-dir>/<TIC>_cone.ecsv`, using the existing geometry catalog schema.
The repository's PRF grids, strap map, full feature table, historical identity
catalog, and selected model bundle must be available locally. No downloads occur.

Omit `--tics` to apply every source in the manifest. Omit `--tars` only when
qualification is unavailable: all rows will explicitly be `unknown`.
Use `--product-family spoc120` for a separate standard two-minute control;
this remains an experimental extraction transfer. Joined TARS scores describe
the primary catalog and do not certify standard products independently.

## Output contract

- `predictions.parquet`: one row per usable source/sector, with correction,
  raw/corrected sector median, reference count/radii, anchor/fallback status,
  qualification, imputed feature names/count, and simple training-range flags.
- `<TIC>/curves.parquet`: accepted Q0 finite-positive PDCSAP cadences, raw flux,
  corrected flux, SAP diagnostic, and formal PDCSAP errors before/after scaling.
- `<TIC>/features.parquet`: the complete prepared feature rows.
- `cadence_attrition.csv`: original and accepted cadence counts for every
  input sector, including sectors with no valid cadences.
- `coverage.csv`: all-sector and individually TARS-qualified coverage and
  fallbacks. No variable-star scatter improvement is interpreted as accuracy.
- `identity_audit.json`, `protocol.json`, `input_products.parquet`,
  `source_snapshot/`, `complete.json`: source checks, model and input hashes,
  frozen run settings, copied application code and output hashes.

`corrected = raw / correction`. Formal `corrected_error` scales the FITS
cadence error by the same factor; it does **not** include uncertainty in the
predicted correction, extraction/background calibration, or stellar variability.
`science_supported` is only a basic magnitude/color/sector-range and reference
screen, **not** a scientifically validated quality certificate. Missing inputs
still require the saved training imputer, with names reported for every sector.

Missing Gaia RP/color uses explicit flagged frozen-imputation/reference
inference. It never normalizes using the query flux; such output is not
Gaia-anchored science validation. The Gaia target is never recentered or refitted.

## Identity and temporal scope

New-source application rejects any historical cohort match by TIC, Gaia ID or
less than two arcseconds. This conservative restriction prevents aliases from
entering the frozen reference/color preprocessing. Historical sources should
use the archived full application table rather than this external-source path.
The reference bank is verified against its fitting rows and physical groups;
validation/test groups are forbidden, and the query group/fold is excluded.

Metadata context uses **all supplied sectors**. This is retrospective,
transductive application: adding sectors can change earlier metadata deviations
and corrections. It is not a future-only forecasting benchmark. The selected
sector feature has no validated extrapolation beyond the frozen reference era.

## Science validation

The historical 0.6196% all-sector and 0.5811% qualified median CV describe quiet
development stars. They are not uncertainty bars for new variables or a noise
floor. Multiplication preserves within-sector fractional shape and locally
normalized eclipse depths algebraically. Long-term mean levels and amplitudes
against a common reference require independent measurements.

The initial application/measurement audit is saved in
`eval/science_readiness_20261013/`. Its ground-only phase-adjusted seasonal
diagnostics preserve the original prospective benchmark and explicitly identify
few-season passband uncertainty. A fresh confirmation cohort must be selected
and its scoring protocol frozen before correction outcomes are examined.

## Checks

```bash
python3 -m unittest discover -s eval -p test_frozen_physical.py
python3 -m unittest discover -s eval -p test_seasonal_science_readiness.py
```

The first suite checks all 570 archived variable corrections, query-flux
invariance, reference-label/source-group isolation, missing-anchor behavior,
and schema failures. The second checks phase-level recovery under strongly
asymmetric sampling and rejection of rank-deficient phase coverage.

## CNF102 with per-sector sigma (`infer_cnf.py`, added 2026-11-02)

`infer_cnf.py` applies the frozen pdcsap-branch CNF102
(`eval/sap_gaia_cnf_20261017/pdcsap`, the model behind every 2026-10/11
quiet-star and join benchmark) with the same inputs and identity rules as
`infer_physical.py`, and adds the correction uncertainty that the HGB path
cannot provide:

```bash
python3 data/download_flat_models.py   # once; 16 public flats, 568 MB, in tess_cache/flat_models
python3 infer_cnf.py --products study/products.parquet --catalog-dir study \
    --tars study/tars.parquet --tics 230384382 --out new_run
```

Per sector `predictions.parquet` carries `pred_knn` (pdcsap reference base),
`hgb_pdcsap` (pdcsap HGB control), `cnf102` (point estimate, the analytic
conditional median), `cnf102_sigma_pct` (standard deviation of the log
correction over 256 seeded flow samples), the 16/84 and 2.5/97.5 percentile
factors, and two frozen quality flags. `sigma_keep80` and `sigma_keep50` mark
sectors whose sigma is below the 80 % and 50 % quantiles of the qualified quiet
confirmation population (1.33 % and 0.88 %, frozen in
`eval/cnf_sigma_release_20261102/thresholds.json`). They behave like TARS
qualification: a coverage/precision trade reported with attrition, never a
headline. Cadence files add `level_sigma = corrected x sigma`, the per-sector
level uncertainty; `corrected_error` stays the scaled formal error. Neither
includes stellar variability. On qualified quiet confirmation stars the sigma
is calibrated in scale (|z| < 1 for 80 %, |z| < 2 for 97 %) with heavy tails,
and ranks per-star residual scatter with Spearman 0.65; use the percentile
columns rather than a Gaussian when the tails matter.

The module `stitch/cnf.py` (`PdcsapCNF.apply`) is the same code path used by
the untouched-star join benchmark (`eval/untouched_join_benchmark_20261102`),
which is its acceptance test on stars outside every split.

### Point-source flat correction (added 2026-11-05)

`infer_cnf.py` now also corrects an imprint of the 2018 flat field on stellar
flux. Each sector gets `flat_phi_pct`, `flat_factor = exp(-phi/100)` and
`total_factor = cnf102 x flat_factor`, and cadence files use
`corrected = raw / total_factor`. `phi` depends only on camera, CCD and
position. It has no fitted amplitude and reads no stellar flux, so it cannot
respond to a star's variability, and it rescales each sector by a constant.
Use `--no-flat-correction` to reproduce outputs from before this date.

`cnf102`, its percentile factors, `cnf102_sigma_pct`, the sigma flags and
`hgb_pdcsap` remain model-only. To apply the flat term to them, multiply by
`flat_factor`. The sigma does not include the flat term.

On qualified confirmation stars this lowers the median step from 0.572 % to
0.518 % for CNF102 and from 0.550 % to 0.498 % for HGB. It is not demonstrated
on boundary joins (642 quiet joins: 0.789 % to 0.784 %, 95 % interval of the
change -0.064 to +0.036) or on pulsator amplitudes, and most of the step
remains. Evidence and limits: `eval/flat_imprint_20261105/REPORT.md`.
