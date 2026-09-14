"""
Replace flux_offset with a Gaia RP-based absolute label.

  gaia_label = sector_median / expected_flux_from_gaia_rp

where expected_flux is calibrated per-camera via a log-linear fit
between gaiarp and the per-star median sector_median.

Old LOO labels are preserved as flux_offset_loo.
Stars with no Gaia RP match keep their LOO label (flux_offset unchanged).

Usage:
  python3 data/compute_gaia_labels.py [parquet]
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import linregress
from astroquery.mast import Catalogs

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
CHUNK   = 1000   # TIC query chunk size
OUT     = PARQUET  # overwrite in place (with backup column)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
print(f"  {len(df):,} rows  |  {df.tic_id.nunique():,} unique TIC IDs")

# ── Query TIC for gaiarp in chunks ────────────────────────────────────────────
tic_ids = df["tic_id"].dropna().astype(int).unique().tolist()
chunks  = [tic_ids[i:i+CHUNK] for i in range(0, len(tic_ids), CHUNK)]
print(f"\nQuerying TIC in {len(chunks)} chunks of ≤{CHUNK} …")

records = []
for k, chunk in enumerate(chunks):
    try:
        res = Catalogs.query_criteria(catalog="TIC", ID=chunk)
        sub = res[["ID", "gaiarp"]].to_pandas()
        sub["ID"]     = pd.to_numeric(sub["ID"],     errors="coerce").astype("Int64")
        sub["gaiarp"] = pd.to_numeric(sub["gaiarp"], errors="coerce")
        records.append(sub.dropna(subset=["ID"]))
    except Exception as e:
        print(f"  chunk {k+1}/{len(chunks)} FAILED: {e}")
        continue
    if (k + 1) % 10 == 0 or k == len(chunks) - 1:
        print(f"  {k+1}/{len(chunks)} done  ({sum(len(r) for r in records):,} rows so far)")

tic_df = pd.concat(records, ignore_index=True)
tic_df = tic_df.rename(columns={"ID": "tic_id"})
tic_df["tic_id"] = tic_df["tic_id"].astype("Int64")
tic_df = tic_df.drop_duplicates("tic_id")
n_with_rp = tic_df["gaiarp"].notna().sum()
print(f"\nGot {n_with_rp:,} / {len(tic_ids):,} stars with Gaia RP "
      f"({n_with_rp/len(tic_ids)*100:.1f}%)")

# ── Per-camera ZP calibration ─────────────────────────────────────────────────
# Use per-star median flux to fit: log10(median_flux) = slope * gaiarp + intercept
print("\nCalibrating ZP per camera …")
df["tic_id_int"] = df["tic_id"].astype("Int64")
star_med = (df[df.sector_median > 0]
            .groupby(["tic_id_int", "cam"])["sector_median"]
            .median()
            .reset_index()
            .rename(columns={"tic_id_int": "tic_id", "sector_median": "median_flux"}))

cal = star_med.merge(tic_df, on="tic_id").dropna(subset=["gaiarp", "median_flux"])
cal = cal[(cal.median_flux > 0) & cal.gaiarp.notna()]
cal["log_flux"] = np.log10(cal["median_flux"])

zp = {}  # cam -> (slope, intercept)
for cam, grp in cal.groupby("cam"):
    if len(grp) < 20:
        print(f"  cam {cam}: too few stars ({len(grp)}), skipping")
        continue
    slope, intercept, r, *_ = linregress(grp["gaiarp"], grp["log_flux"])
    zp[cam] = (slope, intercept)
    print(f"  cam {cam}: n={len(grp):,}  slope={slope:.4f}  "
          f"intercept={intercept:.4f}  r={r:.4f}")

if not zp:
    # Fall back to global calibration
    slope, intercept, r, *_ = linregress(cal["gaiarp"], cal["log_flux"])
    for cam in df["cam"].dropna().unique():
        zp[cam] = (slope, intercept)
    print(f"  global fallback: slope={slope:.4f}  intercept={intercept:.4f}  r={r:.4f}")

# ── Compute Gaia label ────────────────────────────────────────────────────────
print("\nComputing Gaia labels …")
# Drop any stale gaiarp / _tic columns so the merge is clean and non-duplicate
stale = [c for c in df.columns if "gaiarp" in c or c == "tic_id_tic"]
df = df.drop(columns=stale, errors="ignore")
df_out = df.merge(tic_df[["tic_id", "gaiarp"]], left_on="tic_id_int",
                  right_on="tic_id", how="left", suffixes=("", "_tic"))
df_out = df_out.drop(columns=["tic_id_tic"], errors="ignore")

def expected_flux(row):
    cam = row["cam"]
    rp  = row["gaiarp"]
    if pd.isna(rp) or cam not in zp:
        return np.nan
    s, i = zp[cam]
    return 10 ** (s * rp + i)

df_out["expected_flux_gaia"] = df_out.apply(expected_flux, axis=1)
df_out["gaia_label_raw"] = df_out["sector_median"] / df_out["expected_flux_gaia"]

# Keep LOO, replace flux_offset where we have a valid Gaia label
df_out["flux_offset_loo"] = df_out["flux_offset"].copy()
has_gaia = df_out["gaia_label_raw"].notna() & (df_out["sector_median"] > 0)
df_out.loc[has_gaia, "flux_offset"] = df_out.loc[has_gaia, "gaia_label_raw"]

n_replaced = has_gaia.sum()
n_kept_loo = (~has_gaia).sum()
print(f"  Gaia label applied: {n_replaced:,} rows ({n_replaced/len(df_out)*100:.1f}%)")
print(f"  LOO kept (no Gaia): {n_kept_loo:,} rows ({n_kept_loo/len(df_out)*100:.1f}%)")

# ── Summary stats ─────────────────────────────────────────────────────────────
clean = df_out[has_gaia & (df_out.flux_offset > 0.7) & (df_out.flux_offset < 1.3)]
print(f"\nGaia label stats (clean range 0.7–1.3):")
print(f"  n={len(clean):,}  mean={clean.flux_offset.mean():.4f}  "
      f"std={clean.flux_offset.std():.4f}  "
      f"MAD={clean.flux_offset.sub(1).abs().median():.4f}")

print(f"\nLOO label stats (same rows for comparison):")
print(f"  n={len(clean):,}  mean={clean.flux_offset_loo.mean():.4f}  "
      f"std={clean.flux_offset_loo.std():.4f}  "
      f"MAD={clean.flux_offset_loo.sub(1).abs().median():.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
drop_cols = ["tic_id_int", "expected_flux_gaia", "gaia_label_raw", "gaiarp_tic", "tic_id_tic"]
df_out = df_out.drop(columns=[c for c in drop_cols if c in df_out.columns])

print(f"\nSaving to {OUT} …")
df_out.to_parquet(OUT, index=False)
print(f"Done. Columns: {list(df_out.columns)}")
