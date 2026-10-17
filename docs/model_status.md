# Model status

The release provides reproducible research workflows and verified frozen models. It does not designate a new production calibration or certify long-baseline science performance.

| Model | Target and application |
| --- | --- |
| Frozen physical HGB | Gaia RP with a training-only color/passband transfer; five rounds, 120 predictors, source-group-safe reference banks and retained external anchor. Trained on 457,123 qualified sectors from 36,507 stars. The original `default_model_promoted: false` remains in its manifest. |
| Frozen pairwise CNF | Historical relative-target NSF; exact conditional median, saved beta 0.95 and clipping. Selected among four archived candidates using eight calibration stars. Application references were repaired and intersected with its historical training membership. This does not repair historical training preprocessing or make it a global best model. |
| New training commands | HGB and CNF default to Gaia RP. New fits create new directories and are never automatically promoted. Relative CNF targets require an explicit option. |

Lower scatter in quiet-star sector medians does not by itself show that true stellar variability is preserved. Inputs related to crowding, extraction, PDC or flux-weighted centroids can respond to stellar signals. The models do not explicitly force sector fluxes to match, but a learned correction can still alter genuine variability.

The frozen HGB uses metadata summaries across all supplied visits. The effect of limiting this history to nearby sectors or downweighting distant visits is being tested separately. Those temporal-context models are not part of this frozen release. No behavior during an unobserved gap is reconstructed.

The repeatedly examined test stars are a development benchmark. Report all retained sectors and individually TARS-qualified sectors separately, including lost rows and missing anchors. A selected quiet TIC does not make every collected sector quiet or individually qualified. A missing score is unknown, not a passing score.

Sampling precision, extraction/calibration uncertainty and stellar variability are different contributions. Neither model residual scatter nor short-timescale precision establishes a photometric noise floor. The formal light-curve error bars scaled by a correction factor do not include calibration uncertainty.

The supplied HGB was developed for TESS-SPOC PDCSAP. Collecting SAP or adjusted SAP does not validate transferring that model to a different extraction. Independent science checks and extraction-aware signal tests remain necessary.
