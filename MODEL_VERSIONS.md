# STITCH Model Registry

## Current canonical model

| Field | Value |
|---|---|
| **File** | `stitch_nsf_v3_thermal.pt` |
| **Date** | 2026-08-24 |
| **Training script** | `training/train_flow_nsf.py` |
| **Training data** | `training_data_v3.parquet` (97 MB, 850K rows, sectors 1–100, includes Gaia labels + focal plane temps) |
| **Target** | `flux_offset` (Gaia-calibrated label; LOO fallback where Gaia unavailable) |
| **Context dim** | 24D |
| **Stars** | 64,378 (TARS quiet, Tmag ≤ 13, ≥ 4 sectors) |
| **Test MAE** | 0.0086 (0.86%) |
| **Baseline MAE** | 0.0164 (global mean) |
| **Improvement** | 47.8% over global mean baseline |
| **Median efficiency** | 37.4% (vs. oracle ceiling set by label noise) |
| **Val NLL** | 0.5747 (early stop epoch 40) |
| **Architecture** | NSF, 8 transforms, 16 bins, hidden=[256,256], 674K params |

### Feature set (24D)
```
Continuous (18): col, row, delta_sub_col, delta_sub_row, sector,
                 tmag, crowdsap, cdpp1_0, pdcvar, jitter_rms,
                 pdc_noi, pr_wght2, gaiarp, perstar_gaia_offset,
                 focal_plane_temp,
                 sector_ccd_mean_loo, spatial_knn_mean_loo
Categorical (6): cam (1–4 one-hot), ccd (1–4 one-hot)
```

---

## Ablation models

### `stitch_nsf_original.pt` — original conditioning vector only
- **Date**: 2026-08-24
- **Script**: `training/train_flow_nsf_original.py`
- **Data**: `training_data_v3.parquet`
- **Target**: `flux_offset`
- **Context dim**: 16D
- **Features**: `col, row, delta_sub_col, delta_sub_row, tmag, crowdsap, focal_plane_temp, jitter_rms` + cam/ccd one-hot (no spatial, no Gaia, no PDC)
- **Median efficiency**: 18.1%
- **Val NLL**: 0.9979 (early stop epoch 45)
- **Purpose**: Establishes baseline for original proposal c = [row,col,Δsub-pixel,cam,CCD,Tmag,ρ_crowd,T_thermal,σ_jitter]

---

## Experimental / intermediate models

These were stepping stones during development. **Do not use for evaluation or comparison** — they use different data, targets, or filters, so metrics are not comparable to the canonical model.

| File | Date | Data | Target | Key difference | Status |
|---|---|---|---|---|---|
| `stitch_nsf_intrasector.pt` | 2026-08-13 | `training_data_topup_pdc_intrasector.parquet` | intra-sector LOO | Different target (within-sector), y_std=0.012 vs 0.024 — metrics not comparable | Experimental |
| `stitch_nsf_v5_v3data.pt` | 2026-08-18 | `training_data_v3.parquet` | `flux_offset_loo` | v5 architecture + extra features (spatial_poly_pred, loo_prev); LOO target | Superseded |
| `stitch_nsf_knn_v3data.pt` | 2026-08-18 | `training_data_v3.parquet` | unknown | KNN features, v3 data | Superseded |
| `stitch_nsf_gaia_feat.pt` | 2026-08-13 | `training_data_topup_pdc.parquet` | unknown | Added Gaia features | Superseded |
| `stitch_nsf_gaia.pt` | 2026-08-13 | `training_data_topup_pdc.parquet` | unknown | Gaia only | Superseded |
| `stitch_nsf_v5.pt` | 2026-08-15 | `training_data_topup_pdc.parquet` | `flux_offset_loo` | v5 architecture, older data | Superseded |
| `stitch_nsf_pdc_full.pt` | 2026-08-06 | `training_data_topup_pdc_spatial.parquet` | unknown | Full PDC features | Superseded |
| `stitch_nsf_spatial.pt` | 2026-08-06 | `training_data_topup_pdc_spatial.parquet` | unknown | First with spatial KNN | Superseded |
| `stitch_nsf_tmag12.pt` | 2026-08-06 | unknown | unknown | Tmag ≤ 12 cut | Superseded |
| `stitch_nsf_v4_pdc.pt` | 2026-07-25 | `training_data_topup_bkg.parquet` | unknown | v4 + PDC | Superseded |
| `stitch_nsf_v4_crossstar.pt` | 2026-07-23 | `training_data_topup_crossstar.parquet` | unknown | v4 + cross-star features | Superseded |
| `stitch_nsf_nbr.pt` | 2026-08-14 | unknown | unknown | Neighbor features | Superseded |
| `stitch_nsf_v3.pt` | 2026-07-22 | `training_data_topup.parquet` | unknown | v3 architecture | Superseded |
| `stitch_nsf_v2.pt` | 2026-07-22 | `training_data_topup.parquet` | unknown | v2 architecture | Superseded |
| `stitch_nsf_topup.pt` | 2026-07-22 | `training_data_topup.parquet` | unknown | First with top-up stars | Superseded |
| `stitch_nsf.pt` | 2026-07-21 | `training_data.parquet` | unknown | First full NSF | Superseded |
| `stitch_nsf_weighted_v1.pt` | 2026-06-19 | unknown | unknown | Weighted sampling | Superseded |
| `stitch_nsf_per_ccd.pt` | 2026-06-16 | unknown | unknown | Separate model per CCD (7.8 MB — different architecture) | Superseded |
| `stitch_v1.pt` | 2026-06-11 | unknown | unknown | Very early prototype (12 KB) | Superseded |

---

## Metrics reference — current canonical model

All metrics on held-out test set (never seen during training):

| Camera | n stars | MAE | Baseline MAE | Improvement |
|---|---|---|---|---|
| Cam 1 (ecliptic) | 1,253 | 0.0106 | 0.0178 | 40.4% |
| Cam 2 | 5,849 | 0.0102 | 0.0200 | 49.1% |
| Cam 3 | 16,308 | 0.0087 | 0.0157 | 44.3% |
| Cam 4 (ecliptic pole) | 56,992 | 0.0083 | 0.0162 | 48.6% |
| **Overall** | **80,402** | **0.0086** | **0.0164** | **47.8%** |

**Baseline**: predict the global mean flux offset for every observation.  
**Improvement %**: `(baseline_MAE − STITCH_MAE) / baseline_MAE`. Note: baseline is weak (global mean); improvement over a per-CCD or per-sector mean would be smaller.

---

## Data versions

| File | Date | Size | Key additions vs. predecessor |
|---|---|---|---|
| `training_data_v3.parquet` | 2026-08-24 | 97 MB | Focal plane temps (98.3% coverage), Gaia-calibrated labels, PDC metrics, 850K rows |
| `training_data_topup_pdc_newstars.parquet` | 2026-08-24 | 96 MB | New stars top-up batch |
| `training_data_topup_pdc_v2.parquet` | 2026-08-17 | 62 MB | PDC v2 |
| `training_data_topup_pdc.parquet` | 2026-08-13 | 57 MB | PDC features added |
| `training_data_topup_pdc_spatial.parquet` | 2026-08-06 | 52 MB | Spatial features |
| `training_data_topup.parquet` | 2026-07-22 | 49 MB | Top-up stars added |
| `training_data.parquet` | 2026-07-07 | 29 MB | Original dataset |

**Canonical data**: `training_data_v3.parquet`
