"""
Add robust spatial KNN features to an existing STITCH parquet.

Adds four new columns to any parquet that already has flux_offset_loo,
col, row, sector, cam, ccd:

  spatial_knn_median_loo  — median of K nearest neighbours' LOO offset
  spatial_knn_mad_loo     — MAD of K nearest neighbours (spread / reliability)
  spatial_knn_q25_loo     — 25th percentile of neighbours (lower tail)
  spatial_knn_q75_loo     — 75th percentile of neighbours (upper tail)

All four are computed with LOO exclusion (a star is never its own neighbour).
Existing columns (spatial_knn_mean_loo, sector_ccd_mean_loo) are preserved.

Usage:
  python3 data/add_robust_spatial_features.py \\
      training_data_v3.parquet training_data_v4.parquet

Or just:
  python3 data/add_robust_spatial_features.py   # uses defaults below
"""

import sys, time, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

IN_PARQUET  = next((a for a in sys.argv[1:] if a.endswith(".parquet") and "_v4" not in a),
                   "training_data_v3.parquet")
OUT_PARQUET = next((a for a in sys.argv[1:] if a.endswith(".parquet") and a != IN_PARQUET),
                   IN_PARQUET.replace(".parquet", "_v4_fixed.parquet")
                   if "_v4" not in IN_PARQUET
                   else IN_PARQUET)

K       = 20    # neighbours (same as canonical model)
TARGET  = "flux_offset_loo"

print(f"Input:  {IN_PARQUET}")
print(f"Output: {OUT_PARQUET}")
print(f"K = {K}, target = {TARGET}")

t0 = time.time()
df = pd.read_parquet(IN_PARQUET)
print(f"Loaded {len(df):,} records from {df['tic_id'].nunique():,} stars  "
      f"({time.time()-t0:.1f}s)")

# ── Validate ───────────────────────────────────────────────────────────────────
required = ["tic_id", "sector", "cam", "ccd", "col", "row", TARGET]
missing  = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Input parquet missing columns: {missing}")

# Warn if columns already exist (will be overwritten)
new_cols = ["spatial_knn_median_loo", "spatial_knn_mad_loo",
            "spatial_knn_q25_loo", "spatial_knn_q75_loo"]
existing = [c for c in new_cols if c in df.columns]
if existing:
    print(f"  WARNING: overwriting existing columns: {existing}")

# ── Filter to quality rows for computing features ─────────────────────────────
# We compute features on the full dataset (all rows act as neighbours for each other)
# matching the same quality filter used in training so the features are consistent.
df_clean = df.copy()
df_clean["_source_row"] = np.arange(len(df_clean))
df_clean = df_clean.dropna(subset=["col", "row", TARGET])
df_clean = df_clean[(df_clean[TARGET] > 0.85) & (df_clean[TARGET] < 1.15)]
df_clean = df_clean.reset_index(drop=True)
print(f"Quality-filtered: {len(df_clean):,} records for KNN computation")

# ── Compute robust KNN features ────────────────────────────────────────────────
print(f"Computing K={K} spatial KNN features per (sector, cam, ccd) …")
n_groups = 0
n_small  = 0

knn_median = np.full(len(df_clean), np.nan, dtype=np.float32)
knn_mad    = np.full(len(df_clean), np.nan, dtype=np.float32)
knn_q25    = np.full(len(df_clean), np.nan, dtype=np.float32)
knn_q75    = np.full(len(df_clean), np.nan, dtype=np.float32)

for (sec, cam, ccd), grp in df_clean.groupby(["sector", "cam", "ccd"]):
    n = len(grp)
    n_groups += 1
    if n < 2:
        n_small += 1
        continue

    coords = grp[["col", "row"]].values.astype(np.float32)
    vals   = grp[TARGET].values
    idx    = grp.index.values
    k      = min(K, n - 1)

    # query k+1 neighbours; first result is self (distance=0), skip it
    _, candidates = cKDTree(coords).query(coords, k=k + 1)
    # Tied coordinates do not guarantee that self is the first result.
    local = np.arange(n)
    nn = np.asarray([row[row != i][:k] for i, row in zip(local, candidates)])
    nbr_vals = vals[nn]

    knn_median[idx] = np.median(nbr_vals, axis=1)
    knn_mad[idx]    = np.median(np.abs(nbr_vals - np.median(nbr_vals, axis=1, keepdims=True)), axis=1)
    knn_q25[idx]    = np.percentile(nbr_vals, 25, axis=1)
    knn_q75[idx]    = np.percentile(nbr_vals, 75, axis=1)

print(f"  Groups: {n_groups:,}  small (<2 stars, skipped): {n_small:,}")

# Fill singletons with sector_ccd_median (robust fallback)
ccd_med_map = df_clean.groupby(["sector", "cam", "ccd"])[TARGET].median()
df_clean["_ccd_med"] = df_clean.set_index(["sector","cam","ccd"]).index.map(ccd_med_map)

def fill_nan(arr, fallback_col):
    series = pd.Series(arr.astype(np.float64), index=df_clean.index)
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        series[nan_mask] = df_clean.loc[nan_mask, fallback_col].values.astype(np.float64)
    return series.values.astype(np.float32)

knn_median = fill_nan(knn_median, "_ccd_med")
knn_mad    = np.where(np.isnan(knn_mad), 0.0, knn_mad)
knn_q25    = fill_nan(knn_q25, "_ccd_med")
knn_q75    = fill_nan(knn_q75, "_ccd_med")

df_clean["spatial_knn_median_loo"] = knn_median
df_clean["spatial_knn_mad_loo"]    = knn_mad
df_clean["spatial_knn_q25_loo"]    = knn_q25
df_clean["spatial_knn_q75_loo"]    = knn_q75

# ── Diagnostics ────────────────────────────────────────────────────────────────
print("\nNew feature diagnostics:")
r_mean   = df_clean["spatial_knn_median_loo"].corr(df_clean[TARGET]) if "spatial_knn_mean_loo" in df_clean.columns else None
r_median = df_clean["spatial_knn_median_loo"].corr(df_clean[TARGET])
r_mad    = df_clean["spatial_knn_mad_loo"].corr(df_clean[TARGET])

print(f"  spatial_knn_median_loo: mean={df_clean['spatial_knn_median_loo'].mean():.5f}  "
      f"std={df_clean['spatial_knn_median_loo'].std():.5f}  r={r_median:.4f}")
print(f"  spatial_knn_mad_loo:    mean={df_clean['spatial_knn_mad_loo'].mean():.5f}  "
      f"std={df_clean['spatial_knn_mad_loo'].std():.5f}  r_target={r_mad:.4f}")
print(f"  spatial_knn_q25_loo:    mean={df_clean['spatial_knn_q25_loo'].mean():.5f}")
print(f"  spatial_knn_q75_loo:    mean={df_clean['spatial_knn_q75_loo'].mean():.5f}")
if r_mean is not None:
    r_old = df_clean["spatial_knn_mean_loo"].corr(df_clean[TARGET])
    print(f"  (mean r={r_old:.4f} vs median r={r_median:.4f} — "
          f"{'median better' if r_median > r_old else 'mean better'})")

# ── Merge back to original (all rows, including those filtered out) ─────────────
# Rows that didn't pass quality filter get NaN for positional columns (median, q25, q75)
# but 0 for MAD — undefined spread should not be confused with "large spread".
for col in new_cols:
    df[col] = np.nan
    if col in df_clean.columns:
        df.loc[df_clean["_source_row"].to_numpy(), col] = df_clean[col].to_numpy()

# MAD NaN → 0 (singleton / quality-filtered rows have no defined neighbourhood spread)
df["spatial_knn_mad_loo"] = df["spatial_knn_mad_loo"].fillna(0.0)

# ── Provenance metadata ───────────────────────────────────────────────────────
meta = {
    "source":     IN_PARQUET,
    "script":     "data/add_robust_spatial_features.py",
    "date":       time.strftime("%Y-%m-%d"),
    "k":          K,
    "target":     TARGET,
    "new_cols":   new_cols,
    "n_records":  len(df),
    "n_stars":    int(df["tic_id"].nunique()),
}
print(f"\nProvenance: {json.dumps(meta, indent=2)}")

# ── Save ───────────────────────────────────────────────────────────────────────
df.to_parquet(OUT_PARQUET, index=False)
elapsed = time.time() - t0
print(f"\nSaved → {OUT_PARQUET}  ({len(df):,} records, {elapsed:.0f}s total)")
print(f"Columns added: {new_cols}")
