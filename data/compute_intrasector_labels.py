"""
Compute intra-sector normalization labels.

Instead of LOO: flux_offset = sector_median(star, s) / mean_j≠s[sector_median(star, j)]
Use intra-sector: flux_offset = sector_median(star, s) / median_peers(cam, ccd, sector)

where median_peers is the spatial median of sector_median values across all stars
on the same (cam, ccd, sector), optionally refined to spatial bins within the CCD.

Key advantage over LOO:
  - No cross-sector history required (works for single-sector stars)
  - Less contaminated by astrophysical variability (variable stars don't inflate
    their own label; the reference is their neighbors, not their own history)
  - Directly measures the detector-level systematic

Two label resolutions produced:
  flux_offset_intrasec    : single value per (cam, ccd, sector) — CCD-level
  flux_offset_intrasec_4x4: 4x4 spatial bins per CCD — finer spatial resolution

Usage:
  python3 data/compute_intrasector_labels.py [input.parquet] [output.parquet]
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

IN  = next((s for s in sys.argv[1:] if s.endswith(".parquet") and "output" not in s),
           "training_data_topup_pdc.parquet")
OUT = next((s for s in sys.argv[1:] if "output" in s),
           "training_data_topup_pdc_intrasector.parquet")
print(f"Input:  {IN}")
print(f"Output: {OUT}")

df = pd.read_parquet(IN)
print(f"Loaded {len(df):,} rows, {df['tic_id'].nunique():,} unique stars")
print(f"  flux_offset valid: {df['flux_offset'].notna().sum():,} "
      f"({df['flux_offset'].notna().mean()*100:.1f}%)")

# Only sigma-clip extreme outliers before computing group statistics
valid = df[(df["flux_offset"] > 0.80) & (df["flux_offset"] < 1.20)].copy()
print(f"  After outlier filter: {len(valid):,} rows")

# ── Label 1: CCD-sector median (one value per cam/ccd/sector) ─────────────────
# For each (cam, ccd, sector) group: what is the typical offset on this detector
# chip this sector?
group_med = (valid
    .groupby(["cam", "ccd", "sector"])["flux_offset"]
    .transform("median"))
valid["flux_offset_intrasec"] = group_med

# Sigma-clip obvious variable stars before group stats (they inflate the median)
# Use per-star deviation from their own group median
residual = valid["flux_offset"] - group_med
sigma_g  = residual.groupby([valid["cam"], valid["ccd"], valid["sector"]]).transform("std")
mask_ok  = np.abs(residual) < 3.0 * sigma_g
n_clipped = (~mask_ok).sum()
print(f"  Sigma-clipped {n_clipped:,} star-sector records as potential variable star outliers")

group_med_clipped = (valid[mask_ok]
    .groupby(["cam", "ccd", "sector"])["flux_offset"]
    .median()
    .rename("flux_offset_intrasec_clipped"))
valid = valid.join(group_med_clipped, on=["cam", "ccd", "sector"])
# Fall back to unclipped group median if group has too few after clipping
valid["flux_offset_intrasec"] = valid["flux_offset_intrasec_clipped"].fillna(group_med)

# ── Label 2: 4×4 spatial bins per CCD ────────────────────────────────────────
# Finer resolution: within each (cam, ccd, sector), bin by col and row (4×4)
# and compute the bin median. Falls back to CCD-sector median for sparse bins.
NBINS = 4

print("\nComputing 4×4 spatial bin labels …")
bin_labels = []
for (cam, ccd, sec), g in valid.groupby(["cam", "ccd", "sector"]):
    col_lo, col_hi = g["col"].quantile(0.01), g["col"].quantile(0.99)
    row_lo, row_hi = g["row"].quantile(0.01), g["row"].quantile(0.99)
    ccd_med = g["flux_offset_intrasec"].iloc[0]

    # Skip spatial binning if group too small or degenerate range
    if len(g) < 10 or (col_hi - col_lo) < 1 or (row_hi - row_lo) < 1:
        bin_labels.append(pd.Series(ccd_med, index=g.index))
        continue

    col_edges = np.linspace(col_lo, col_hi, NBINS + 1)
    row_edges = np.linspace(row_lo, row_hi, NBINS + 1)

    try:
        stat, ce, re, bnum = binned_statistic_2d(
            g["col"], g["row"], g["flux_offset"],
            statistic="median", bins=[col_edges, row_edges]
        )
        cnt, _, _, _ = binned_statistic_2d(
            g["col"], g["row"], g["flux_offset"],
            statistic="count",  bins=[col_edges, row_edges]
        )
        ci = np.clip(np.searchsorted(col_edges[1:], g["col"].values), 0, NBINS - 1)
        ri = np.clip(np.searchsorted(row_edges[1:], g["row"].values), 0, NBINS - 1)
        bin_val = stat[ci, ri].copy()
        sparse  = cnt[ci, ri] < 5
        bin_val[sparse | np.isnan(bin_val)] = ccd_med
        bin_labels.append(pd.Series(bin_val, index=g.index))
    except Exception:
        bin_labels.append(pd.Series(ccd_med, index=g.index))

valid["flux_offset_intrasec_4x4"] = pd.concat(bin_labels).sort_index()

# ── Diagnostics ───────────────────────────────────────────────────────────────
loo    = valid["flux_offset"].dropna()
intra  = valid["flux_offset_intrasec"]
fine   = valid["flux_offset_intrasec_4x4"]

print(f"\nLabel statistics (std ↔ variance retained):")
print(f"  LOO raw:         std={loo.std():.5f}")
print(f"  Intra CCD-level: std={intra.std():.5f}  "
      f"({intra.std()**2/loo.std()**2*100:.1f}% variance retained)")
print(f"  Intra 4×4 bins:  std={fine.std():.5f}  "
      f"({fine.std()**2/loo.std()**2*100:.1f}% variance retained)")

# Per-group spread comparison
g_spread = (valid.groupby(["cam", "ccd", "sector"])
    .agg(
        loo_std=("flux_offset", "std"),
        intra_std=("flux_offset_intrasec_4x4", "std"),
    ))
print(f"\n  Median within-group std — LOO: {g_spread.loo_std.median():.5f}  "
      f"  4×4 intra: {g_spread.intra_std.median():.5f}")
print(f"  (lower intra std means smoother within-group labels)")

# ── Merge back onto full dataframe and save ────────────────────────────────────
out = df.copy()
out["flux_offset_raw"] = out["flux_offset"]  # keep original LOO

for col in ["flux_offset_intrasec", "flux_offset_intrasec_4x4"]:
    out[col] = np.nan
    out.loc[valid.index, col] = valid[col]

# Primary label column: use 4×4 for the training parquet
out["flux_offset"] = out["flux_offset_intrasec_4x4"]

# For single-sector stars (no LOO), also assign CCD-sector median
n_filled = out["flux_offset"].notna().sum() - df["flux_offset"].notna().sum()
print(f"\n  Records with new label (incl. prev NaN LOO): "
      f"{out['flux_offset'].notna().sum():,}  (+{max(n_filled,0):,} newly labelled)")

out.to_parquet(OUT, index=False)
print(f"\nSaved → {OUT}")
print(f"  Columns: {out.columns.tolist()}")
