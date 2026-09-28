# STITCH model status

Updated September 28, 2026. This registry supersedes historical performance and
feature claims. Checkpoints and training data are not distributed in this repository.

| Model/experiment | Status |
|---|---|
| Matched global and residual HGB, seed 20260928 | New matched diagnostic benchmark; global 0.937%, residual 0.941%, tuned KNN 1.029% median CV on 2,970 stars |
| Compact NSF pilot | 1.037% median CV, 28.4% reduction; tuned KNN 1.024%, 29.3%, on 2,969 common stars |
| Full NSF pilot | 1.071% median CV, 26.1% reduction on the same pilot; worse than tuned KNN |
| Historical v3 thermal checkpoint | Inspected feature list does not include temperature despite its name |
| v4 exp1 and exp2 | Trained with misaligned robust spatial features; legacy evaluator also overlaps training stars; headline results withdrawn |
| v5 uncertainty checkpoint | Zero-filled missing-feature evaluation is invalid; polynomial and previous-LOO feature construction need methodological review |

The matched HGB experiment omits the subject-derived Gaia anchor and uses a
separate reference-star catalog. The older NSF pilot includes the subject's mean
Gaia flux ratio and a different conditioning vector; comparisons across these
experiments cannot establish an architecture advantage.

The v4 builder reset filtered row indices before joining features back. A separate
fixed parquet exists locally, but merely changing evaluation inputs cannot repair
training. The legacy trainers/evaluators remain historical research code and are
not the recommended path for new scientific claims.

A star-level split alone is insufficient: all target-derived reference features
must also respect the partition. Exact flow likelihood does not guarantee
calibrated uncertainty or preservation of astrophysical variability.

See [research status](docs/STATUS.md) and the aggregate JSON results under
`eval/matched_global_results/`, `eval/value_benchmark_results/`, and
`eval/value_benchmark_full_results/`.
