# Matched Gaia-RP CNF

The October 17 model uses exactly the physical HGB's 457,123 qualified training
sectors from 36,507 stars, 120 predictors, source-excluded references and Gaia RP
anchor. The training-only color transfer remains fixed. No target centering by
star is added. There are no missing training anchors; 21 all-sector calibration
rows retain the explicit historical relative fallback.

The conditional neural spline flow has four transforms, two 128-unit hidden
layers per transform and 12 bins. Training uses conditional negative log
likelihood, Adam at 3e-4 and seed 2026101201. HGB uses absolute-error boosting;
matching the inputs and target does not match loss or model capacity.

Training stopped after 28 epochs under the prespecified 12-epoch calibration
patience rule. Epoch 16 had the lowest qualified calibration median per-star CV
and was frozen before confirmation/test evaluation. Inference uses the
conditional median, the HGB `pred_knn` reference prior, beta 1 and no clipping.

| Population | Sectors | HGB median CV | CNF median CV |
|---|---|---:|---:|
| Confirmation | All | 0.628095% | 0.659050% |
| Confirmation | Individually TARS-qualified | 0.580546% | 0.610775% |
| Development test | All | 0.619550% | 0.657290% |
| Development test | Individually TARS-qualified | 0.581111% | 0.613743% |

Qualification means each sector has `systematic_score > 0.95`. Confirmation
coverage falls from 39,021 sectors / 2,296 metric-eligible stars to 28,296 / 2,256;
test coverage falls from 79,873 / 4,662 to 56,872 / 4,576. These are historically
examined development cohorts, not pristine holdouts.

Paired qualified confirmation results favor HGB by a median 0.02844 percentage
points (whole-star 95% bootstrap interval 0.02347–0.03499). This configuration
does not improve quiet-star calibration. One configuration and seed cannot
establish which architecture is generally better. Long-baseline signal recovery
has not been evaluated for this checkpoint. No model is automatically promoted.

## Download and apply

```bash
python tools/download_models.py --bundle cnf_gaia --out artifacts
stitch verify-model artifacts/cnf_gaia
stitch predict-cnf --model artifacts/cnf_gaia --features runs/physical.parquet \
  --catalog-dir catalogs --curves runs/curves.parquet --out runs/cnf_gaia
```

Use the physical-feature workflow in [usage](usage.md). Already prepared inputs
must use HGB's 120-input feature/reference construction, not the legacy compact
CNF construction. The historical relative-target checkpoint remains available
under the separate bundle name `cnf`.

The bundle includes the frozen training protocol, original checkpoint hash,
reference assets and training-only scaling. The [selection record](../models/provenance/cnf_gaia_selection.json)
and [metric table](../models/provenance/cnf_gaia_metrics.csv) document the result.
Independent research checks replayed the original HGB targets and scaled inputs,
CNF predictions and all 18 metric groups. Published-bundle inference is checked
against the saved predictions before release.
