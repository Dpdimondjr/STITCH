"""
Compute spatially-smoothed LOO labels per (cam, ccd, sector).

For each (cam, ccd, sector) group, fits a 2D degree-2 polynomial to the
LOO labels (flux_offset) as a function of (col, row).  The fitted surface
value replaces flux_offset — so the training label for each star comes from
the population of stars at similar detector positions in the same sector,
not from that star's own cross-sector history.

Stars in groups with < MIN_STARS fall back to their raw LOO label.
One round of 3-sigma clipping before the second-pass fit to suppress outliers.

Input:  training_data_topup_pdc.parquet
Output: training_data_topup_pdc_spatial.parquet
  flux_offset      -> smoothed spatial label (used for training)
  flux_offset_raw  -> original LOO label     (kept for diagnostics)
"""

import sys
import numpy as np
import pandas as pd

PARQUET    = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                  "training_data_topup_pdc.parquet")
OUT        = next((s for s in sys.argv[1:] if s.endswith("_spatial.parquet")),
                  PARQUET.replace(".parquet", "_spatial.parquet"))
DEGREE     = 2      # quadratic surface: 6 coefficients per (cam, ccd, sector)
MIN_STARS  = 15     # groups smaller than this fall back to raw LOO
SIGMA_CLIP = 3.0    # sigma-clipping threshold for outlier LOO labels

print(f"Input:  {PARQUET}")
print(f"Output: {OUT}")

# ── Load & filter ─────────────────────────────────────────────────────────────
print("\nLoading …")
df = pd.read_parquet(PARQUET)
print(f"  {len(df):,} records, {df['tic_id'].nunique():,} stars")

# Match training script quality filters exactly
df = df.dropna(subset=["col", "row", "flux_offset"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[df["n_sectors_total"] >= 4]
df = df.reset_index(drop=True)
print(f"  After quality filter: {len(df):,} records")

df["flux_offset_raw"] = df["flux_offset"].copy()


# ── 2D polynomial smoother ────────────────────────────────────────────────────
def poly2d_design(c, r, degree=2):
    """Design matrix: [1, c, r, c², cr, r², ...]"""
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            terms.append(c ** (d - i) * r ** i)
    return np.column_stack(terms)


def smooth_group(col, row, y):
    n = len(y)
    if n < MIN_STARS:
        return y.copy(), True   # fallback

    # Centre coordinates for numerical stability
    c = col - col.mean()
    r = row - row.mean()
    X = poly2d_design(c, r, DEGREE)

    try:
        # First pass
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coeffs
        sigma = resid.std()

        # Sigma-clip, then refit on inliers
        mask = np.abs(resid) <= SIGMA_CLIP * sigma
        if mask.sum() < MIN_STARS:
            mask = np.ones(n, dtype=bool)
        coeffs2, _, _, _ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        return X @ coeffs2, False
    except Exception:
        return y.copy(), True


# ── Apply per (cam, ccd, sector) ─────────────────────────────────────────────
print("Fitting 2D polynomial per (cam, ccd, sector) …")
smoothed   = df["flux_offset_raw"].values.copy()
n_groups   = 0
n_fallback = 0

for (cam, ccd, sector), grp in df.groupby(["cam", "ccd", "sector"]):
    idx = grp.index.values
    col = grp["col"].values.astype("float64")
    row = grp["row"].values.astype("float64")
    y   = grp["flux_offset"].values.astype("float64")

    fitted, fell_back = smooth_group(col, row, y)
    smoothed[idx] = fitted
    n_groups   += 1
    n_fallback += int(fell_back)

print(f"  Groups: {n_groups:,}   fallback (< {MIN_STARS} stars): "
      f"{n_fallback:,}  ({n_fallback/n_groups*100:.1f}%)")

# Clip to valid range in case polynomial extrapolates slightly outside [0.85, 1.15]
smoothed = np.clip(smoothed, 0.85, 1.15)
df["flux_offset"] = smoothed

# ── Diagnostics ───────────────────────────────────────────────────────────────
delta = np.abs(df["flux_offset"] - df["flux_offset_raw"])
print(f"\nLabel change |smooth − raw|:")
print(f"  Median:   {np.median(delta):.5f}")
print(f"  Mean:     {np.mean(delta):.5f}")
print(f"  95th pct: {np.percentile(delta, 95):.5f}")
print(f"  Max:      {np.max(delta):.5f}")

print(f"\nRaw LOO std:      {df['flux_offset_raw'].std():.5f}")
print(f"Smoothed LOO std: {df['flux_offset'].std():.5f}")
print(f"Variance retained: "
      f"{df['flux_offset'].var() / df['flux_offset_raw'].var() * 100:.1f}%")

# Per-camera breakdown
print(f"\nPer-camera label change:")
for cam, g in df.groupby("cam"):
    d = np.abs(g["flux_offset"] - g["flux_offset_raw"])
    print(f"  Cam{int(cam)}  n={len(g):>6,}  median |Δ|={np.median(d):.5f}  "
          f"raw_std={g['flux_offset_raw'].std():.5f}  "
          f"smooth_std={g['flux_offset'].std():.5f}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUT, index=False)
print(f"\nSaved → {OUT}  ({len(df):,} records)")
