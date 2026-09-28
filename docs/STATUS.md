# Research status — September 28, 2026

## Completed in this update

### Matched reference-disjoint calibration benchmark

`eval/matched_global_benchmark.py` applies the original value-benchmark metadata
selection (valid normal sectors, positive finite sector flux, Tmag <= 13, Gaia RP
available, at least five retained sectors). It recomputes LOO labels from retained
observations and saves explicit TIC roles with seed 20260928.

There are 16,275 reference stars, 5,966 supported training stars, 1,493 validation
stars and 2,970 common test stars. Spatial features come only from the separate
reference catalog. No subject flux anchor or CDPP/PDC variability input is used.
Residual and global learners receive identical metadata and reference features.
Both use absolute-error histogram gradient boosting with the same model capacity.
Correction strength is selected on validation stars. Predictions are bounded to
[0.5, 2] for every non-raw method; observations are not selected by target value.

| Method | Median corrected-flux CV | Stars worse than raw |
|---|---:|---:|
| Raw | 1.439% | — |
| Median KNN | 1.040% | 13.94% |
| Tuned KNN | 1.029% | 11.48% |
| Residual HGB | 0.941% | 8.92% |
| Global HGB | 0.937% | 8.42% |

Global versus tuned KNN: 8.97% relative residual reduction, paired-star 95% interval
7.79–10.64%. Global versus residual HGB: 0.45%, interval −0.49 to 1.68%.
Thus learnable residual structure survives the new setup, but the global objective
does not establish a significant increment over a matched residual learner here.
The earlier 0.980% global result came from a different cohort/reference policy.

This is one seed on a new partition of previously examined data. The bootstrap
conditions on trained models and the reference catalog. It does not measure
uncertainty across processing eras or training seeds. No new NSF was trained;
the older NSF pilot is a separate experiment, not a matched architecture comparison.

Machine-readable output: [matched results](../eval/matched_global_results/results.json).

### Background susceptibility and spatial residuals

`eval/background_bias_diagnostic.py` intersects the existing 1,000-star cache cohort
with the independent TARS quiet-star catalog and requires five retained sectors,
Tmag <= 13, valid aperture fractions, and at least 50 quality-zero cadences per
observation. The resulting sample has 994 stars and 9,654 observations.

Five-fold TIC cross-fitting produces global-model predictions without the subject
star in its fit. Signed residuals are relative to each star's corrected median;
that normalization defines the diagnostic response, not a model predictor.

The tested susceptibility is

```
x = aperture_pixels * CROWDSAP / (FLFRCSAP * external_expected_flux)
external_expected_flux = 15000 * 10**((10 - Tmag)/2.5)
```

The denominator never uses observed subject flux. The partial regression absorbs
TIC and sector/camera/CCD fixed effects and controls for magnitude, crowding,
aperture capture fraction and detector position. Star-constant/redundant nuisance
directions are removed for numerical stability. Groups require eight stars.

Across 8,864 observations from 966 stars in 348 observing groups, the coefficient
is −0.00106 fractional residual per 0.001 susceptibility, with conditional
TIC-bootstrap interval **−0.00302 to +0.00137**. The interval spans zero and fold
results vary: this does not establish a common additive background error.
It also does not rule out spatially varying additive errors with opposing signs.

The equal-cell cross-star residual-product statistic has a one-sided within-TIC
permutation p-value of approximately 0.046. This is exploratory and weak evidence:
the null does not preserve temporal correlation or heteroscedasticity, and shared
model errors can produce apparent coherence. It is not an estimated physical floor.

Raw/corrected median CV is 1.548%/0.979% for this separate diagnostic cohort.
Do not compare it directly with the larger matched benchmark.

The earlier one-way group-only diagnostic suggested an association; including
star fixed effects removes the clear result. The more controlled result above is
the one retained for interpretation. No independent residual-background estimate
was available. SAP_BKG is not a substitute for that estimate, and a susceptibility
association alone would not establish a causal background correction.

Machine-readable output: [background results](../eval/background_bias_results/results.json).

## Historical benchmark and audit status

The compact/full NSF pilot uses a separate reference pool and corrected-flux CV.
Its compact/full reductions are 28.4%/26.1% versus 29.3% for tuned KNN, on 2,969
common test stars. Full NSF's relative deficit against tuned KNN is 4.6% with
interval 3.3–6.7%; compact NSF has no established advantage. The pilot includes
a subject-derived Gaia anchor and cannot validate blind metadata-only inference.

Historical v4 evaluations are withdrawn because their test population overlaps
checkpoint training and the robust features were misaligned. Local comparison
found 849,078 median-feature values differ between original and fixed parquets;
this is not an exact count of all incorrectly assigned feature values.

The current tests do not establish an irreducible noise floor. Observed scatter
divided by sqrt(N−1) is not an independently validated calibration ceiling.
Astrophysical preservation and calibrated uncertainty remain unproven.

## Next bounded experiments

1. Obtain independent residual-background measurements for a matched subset;
   check susceptibility-by-region effects before fitting a gain-plus-offset model.
2. Verify image/PRF WCS conventions and mask semantics, then compute aperture-
   integrated PRF capture and derivatives with respect to position/shape.
3. Evaluate the physical features against both tuned KNN and residual/global
   models on identical rows and references; keep a new time-based holdout closed.
4. Test multiple seeds and long-timescale signal injections before deployment.

Private correspondence and internal working notes are excluded from this public
status. Source data, caches and model binaries are not included in Git.
