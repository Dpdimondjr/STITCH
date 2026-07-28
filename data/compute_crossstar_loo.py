"""
Recompute training labels using cross-star ensemble LOO.

Current label (temporal LOO):
  offset(star, sector_i) = median(star, sector_i) / mean(star, other_sectors)
  → 96% within-star temporal variance; spatial signal is tiny

New label (cross-star LOO):
  norm_med(star, sector) = sector_median / mean_across_all_sectors_for_that_star
  ref(star, sector)      = mean(norm_med(Y, sector) for Y on same CCD, Y ≠ star)
  offset(star, sector)   = norm_med(star, sector) / ref(star, sector)
  → signal is the shared CCD systematic; star-specific temporal drift removed

Efficient computation: for each (ccd, sector) group,
  ref_i = (sum_all - norm_med_i) / (n - 1)
which avoids O(N²) loops.
"""

import numpy as np, pandas as pd, sys

PARQUET_IN  = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                   "training_data_topup.parquet")
PARQUET_OUT = PARQUET_IN.replace(".parquet", "_crossstar.parquet")
MIN_REF     = 5   # min other stars on same CCD/sector to compute a label

print(f"Input:  {PARQUET_IN}")
print(f"Output: {PARQUET_OUT}")

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading...")
df = pd.read_parquet(PARQUET_IN)
df = df.dropna(subset=["sector_median", "col", "row", "cam", "ccd", "sector"])
df = df[df["sector_median"] > 0].copy()
print(f"  {len(df):,} records, {df['tic_id'].nunique():,} stars")

# ── Step 1: long-run median per star ──────────────────────────────────────────
print("Computing per-star long-run medians...")
star_lrm = (df.groupby("tic_id")["sector_median"]
              .mean()
              .rename("long_run_med"))
df = df.join(star_lrm, on="tic_id")

# ── Step 2: normalised sector median (removes star brightness) ────────────────
df["norm_med"] = df["sector_median"] / df["long_run_med"]

# ── Step 3: cross-star reference per (ccd, sector) ───────────────────────────
print("Computing cross-star ensemble references...")

# Use integer keys to avoid float precision issues
df["ccd_int"]    = df["ccd"].astype(int)
df["sector_int"] = df["sector"].astype(int)

cross_refs = []
n_skipped  = 0

for (ccd, sec), g in df.groupby(["ccd_int", "sector_int"]):
    n = len(g)
    if n < MIN_REF + 1:
        # too few stars — mark as NaN, will be dropped later
        cross_refs.append(pd.Series(np.nan, index=g.index))
        n_skipped += n
        continue

    total = g["norm_med"].sum()
    # leave-one-out ensemble reference for each star
    ref = (total - g["norm_med"]) / (n - 1)
    cross_refs.append(ref)

df["cross_ref"] = pd.concat(cross_refs).reindex(df.index)

# ── Step 4: cross-star offset ─────────────────────────────────────────────────
df["flux_offset_crossstar"] = df["norm_med"] / df["cross_ref"]

before = len(df)
df = df.dropna(subset=["flux_offset_crossstar"])
df = df[(df["flux_offset_crossstar"] > 0.85) & (df["flux_offset_crossstar"] < 1.15)]
print(f"  Dropped {before - len(df):,} records (too few neighbors or out of range)")
print(f"  Remaining: {len(df):,} records, {df['tic_id'].nunique():,} stars")

# ── Step 5: compare distributions ────────────────────────────────────────────
print("\nLabel distribution comparison:")
orig = df["flux_offset"].dropna() if "flux_offset" in df.columns else None
new  = df["flux_offset_crossstar"]

if orig is not None:
    print(f"  Temporal LOO  std={orig.std()*100:.4f}%  "
          f"mean={orig.mean():.6f}  kurtosis={orig.kurt():.2f}")
print(f"  Cross-star    std={new.std()*100:.4f}%  "
      f"mean={new.mean():.6f}  kurtosis={new.kurt():.2f}")

# Per-star scatter of new labels vs old
print("\nPer-star scatter (std of offsets across sectors):")
for label, col in [("Temporal LOO", "flux_offset"), ("Cross-star", "flux_offset_crossstar")]:
    if col not in df.columns: continue
    sc = df.groupby("tic_id")[col].std() * 100
    print(f"  {label}: median={sc.median():.4f}%  mean={sc.mean():.4f}%  "
          f"p90={sc.quantile(.9):.4f}%")

# ── Step 6: write output with flux_offset replaced ───────────────────────────
print(f"\nWriting {PARQUET_OUT}...")
out = df.copy()
# Keep original as backup column, replace flux_offset with new label
out["flux_offset_temporal"] = out["flux_offset"].copy()
out["flux_offset"]          = out["flux_offset_crossstar"]
# Drop working columns
out = out.drop(columns=["norm_med","cross_ref","flux_offset_crossstar",
                         "long_run_med","ccd_int","sector_int"], errors="ignore")
out.to_parquet(PARQUET_OUT, index=False)
print(f"Done. {len(out):,} records saved.")
print(f"\nTo retrain:")
print(f"  python3 training/train_flow_nsf.py {PARQUET_OUT} stitch_nsf_v4_crossstar.pt")
