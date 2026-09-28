# STITCH

**Sector-To-sector Intercalibration via Trained Conditional Homogenization**

STITCH studies instrumental flux offsets between TESS sectors using repeated
observations of quiet stars. We compare spatial reference-star corrections,
robust global calibration and conditional normalizing flows. This is research
software; astrophysical signal preservation and correction uncertainty remain
active validation tasks.

## Current results — September 28, 2026

A new matched diagnostic split uses a separate reference catalog and disjoint
training, validation and test TICs. All methods below are evaluated on the same
2,970 stars using the coefficient of variation of corrected sector fluxes.

| Method | Median CV | Reduction versus raw |
|---|---:|---:|
| Raw | 1.439% | — |
| Median KNN | 1.040% | 27.7% |
| Validation-tuned KNN | 1.029% | 28.5% |
| Residual HGB | 0.941% | 34.6% |
| Robust global HGB | 0.937% | 34.9% |

The global model has **8.97% lower residual scatter than tuned KNN** (paired-star
95% bootstrap interval 7.79–10.64%). Its 0.45% advantage over residual HGB is
inconclusive (−0.49 to 1.68%). These are single-seed diagnostic results on a new
partition of previously examined data, not a pristine final scientific holdout.
No new NSF was trained in this comparison.

A separate compact/full NSF pilot found 28.4%/26.1% reductions versus 29.3% for
tuned KNN on 2,969 stars. Different cohorts and feature sets must not be ranked
by comparing their absolute scatter. Historical v4 feature construction and
some legacy evaluations have known correctness issues; see [model status](MODEL_VERSIONS.md).

The first background-scaling diagnostic is inconclusive after controlling for
star and observing-group effects. There is no established irreducible noise floor.
See [current research status and limitations](docs/STATUS.md).

## Setup and reproduction

```bash
pip install -r requirements.txt

# Requires the local training_data_v3.parquet; no downloads are performed.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 eval/matched_global_benchmark.py

# Requires the existing cache-native cohort and quiet-star catalog.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 eval/background_bias_diagnostic.py

# Focused numerical and leakage-control checks.
python3 -m unittest discover -s eval -p 'test_matched_global_benchmark.py'
python3 -m unittest discover -s eval -p 'test_background_bias_diagnostic.py'
```

Source datasets, FITS caches and model weights are not bundled. Aggregate JSON
results are versioned; scripts record source hashes, split policy and limitations.
Generated per-star tables and split manifests remain local. Both scripts accept
`--help` and `--out` to keep subsequent runs separate.

The cache-native cohort can be extracted from locally downloaded products with:

```bash
python3 eval/spoc_cohort_experiment.py --n-tics 1000 --workers 1 --extract-only
```

That extraction includes all readable selected cache stars; the background
analysis subsequently intersects the result with `tars_quiet_tics_v2.csv` and
requires at least five quality-supported sectors. Product families and cadence
must be recorded explicitly: the cache is not guaranteed to contain exclusively
2-minute products.

## Research priorities

1. Reproduce improvements across additional splits and future-sector tests.
2. Measure residual background independently before adopting additive corrections.
3. Audit WCS/PRF coordinates and calculate aperture-integrated response to motion
   and PRF shape on a bounded sample.
4. Test physical scene/color features, long-timescale signal preservation and
   calibrated uncertainty against the matched baselines.

Private correspondence, handover notes and internal research documents are
excluded from version control. Public Markdown is explicitly allowlisted.
