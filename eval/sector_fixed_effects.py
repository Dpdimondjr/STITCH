"""
Sector fixed-effects post-hoc calibration.

After HGB v3 prediction, add per-sector mean-residual correction learned from
training data.  The logic:

  residual_s(i) = log(actual_s(i) / pred_s(i))

  per-sector bias α_s = mean over train stars: residual_s

  corrected_pred_s(i) = pred_s(i) × exp(α_s)

If prediction errors have a systematic per-sector component, this removes it
and reduces within-star CV.

We also test per-(sector, cam) and per-(sector, cam, ccd) variants.

Usage (from STITCH root):
  python3 eval/sector_fixed_effects.py
"""

import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

STAGE2_DIR = "models/hgb_v3/stage2_data"

def cv_metric(df, pred_col, n_min=2):
    """Median per-star CV of corrected flux."""
    cvs = []
    for _, g in df[df.groupby("tic_id")["sector"].transform("nunique") >= n_min].groupby("tic_id"):
        ratio = g.sector_median.values / g[pred_col].values
        cvs.append(ratio.std() / ratio.mean() * 100)
    return float(np.median(cvs))

# ── Load ──────────────────────────────────────────────────────────────────────
log("Loading ...")
train = pd.read_parquet(os.path.join(STAGE2_DIR, "train.parquet"))
val   = pd.read_parquet(os.path.join(STAGE2_DIR, "val.parquet"))
test  = pd.read_parquet(os.path.join(STAGE2_DIR, "test.parquet"))

for df in [train, val, test]:
    df["residual"] = np.log(np.clip(df.flux_offset_loo, 0.85, 1.15)) \
                   - np.log(df.pred_hgb_v3)

log(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

# Baseline
log("\nBaseline CVs (no sector FE):")
log(f"  Val:  {cv_metric(val, 'pred_hgb_v3'):.4f}%")
log(f"  Test: {cv_metric(test, 'pred_hgb_v3'):.4f}%")

# ── Variant 1: per-sector ─────────────────────────────────────────────────────
log("\n--- Variant 1: per-sector bias ---")

def apply_sector_fe(source_df, target_df, group_cols, pred_col="pred_hgb_v3", out_col="pred_fe"):
    """Learn mean residual per group from source, apply to target."""
    bias = source_df.groupby(group_cols)["residual"].mean().rename("alpha")
    merged = target_df.merge(bias, on=group_cols, how="left")
    merged["alpha"] = merged["alpha"].fillna(0.0)
    merged[out_col] = target_df[pred_col].values * np.exp(merged["alpha"].values)
    return merged[out_col].values

# Calibrate on train, evaluate on val
val["pred_fe_sec"]  = apply_sector_fe(train, val, ["sector"])
test["pred_fe_sec"] = apply_sector_fe(train, test, ["sector"])
val_cv  = cv_metric(val,  "pred_fe_sec")
test_cv = cv_metric(test, "pred_fe_sec")
log(f"  Val:  {val_cv:.4f}%   Test: {test_cv:.4f}%")

# Calibrate on train+val for final test
trainval = pd.concat([train, val], ignore_index=True)
trainval["residual"] = np.log(np.clip(trainval.flux_offset_loo, 0.85, 1.15)) \
                     - np.log(trainval.pred_hgb_v3)
test["pred_fe_sec_tv"] = apply_sector_fe(trainval, test, ["sector"])
log(f"  Test (train+val calibration): {cv_metric(test, 'pred_fe_sec_tv'):.4f}%")

# ── Variant 2: per-(sector, cam) ──────────────────────────────────────────────
log("\n--- Variant 2: per-(sector, cam) ---")
val["pred_fe_seccam"]  = apply_sector_fe(train, val, ["sector","cam"])
test["pred_fe_seccam"] = apply_sector_fe(train, test, ["sector","cam"])
val_cv  = cv_metric(val,  "pred_fe_seccam")
test_cv = cv_metric(test, "pred_fe_seccam")
log(f"  Val:  {val_cv:.4f}%   Test: {test_cv:.4f}%")

test["pred_fe_seccam_tv"] = apply_sector_fe(trainval, test, ["sector","cam"])
log(f"  Test (train+val calibration): {cv_metric(test, 'pred_fe_seccam_tv'):.4f}%")

# ── Variant 3: per-(sector, cam, ccd) ─────────────────────────────────────────
log("\n--- Variant 3: per-(sector, cam, ccd) ---")
val["pred_fe_full"]  = apply_sector_fe(train, val, ["sector","cam","ccd"])
test["pred_fe_full"] = apply_sector_fe(train, test, ["sector","cam","ccd"])
val_cv  = cv_metric(val,  "pred_fe_full")
test_cv = cv_metric(test, "pred_fe_full")
log(f"  Val:  {val_cv:.4f}%   Test: {test_cv:.4f}%")

test["pred_fe_full_tv"] = apply_sector_fe(trainval, test, ["sector","cam","ccd"])
log(f"  Test (train+val calibration): {cv_metric(test, 'pred_fe_full_tv'):.4f}%")

# ── Sector-level diagnostics ──────────────────────────────────────────────────
log("\n--- Per-sector bias (train) vs effect on test ---")
sector_bias = train.groupby("sector")["residual"].mean().sort_values(key=abs, ascending=False)
log(f"  Top 10 biased sectors by |alpha|:")
for sec, alpha in sector_bias.head(10).items():
    n_test = (test.sector == sec).sum()
    log(f"    Sector {int(sec):>3}: alpha={alpha:+.5f}  n_test={n_test}")

# ── Combined: sector FE + β from v2 ─────────────────────────────────────────
# What if we use v2's predictions (slightly better single model) + sector FE?
# We don't have v2 preds in this file, but we can note the concept.
log("\n--- Summary ---")
log(f"  Baseline (HGB v3):                 {cv_metric(test, 'pred_hgb_v3'):.4f}%")
log(f"  + per-sector FE (train-cal):       {cv_metric(test, 'pred_fe_sec'):.4f}%")
log(f"  + per-sector FE (train+val-cal):   {cv_metric(test, 'pred_fe_sec_tv'):.4f}%")
log(f"  + per-(sec,cam) FE (train-cal):    {cv_metric(test, 'pred_fe_seccam'):.4f}%")
log(f"  + per-(sec,cam) FE (tv-cal):       {cv_metric(test, 'pred_fe_seccam_tv'):.4f}%")
log(f"  + per-(sec,cam,ccd) FE (tv-cal):   {cv_metric(test, 'pred_fe_full_tv'):.4f}%")
log(f"\n  Run complete  {T()}")
