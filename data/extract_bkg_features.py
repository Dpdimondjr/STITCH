"""
Extract background / scattered-light features from cached TESS SPOC LC FITS files.

New features (one value per star per sector):
  median_sap_bkg      — median background e-/s (scattered light floor)
  p90_sap_bkg         — 90th percentile background (peak scatter severity)
  bkg_rms             — std of SAP_BKG (how dynamic the scattered light was)
  scatter_flag_frac   — fraction of cadences with QUALITY bit 13 set (SPOC scattered-light flag)
  flfrcsap            — fraction of target flux captured in aperture (complements crowdsap)
  teff                — stellar effective temperature from TIC (K)

Reads from: tess_cache/**/*_lc.fits   (~47K files)
Writes to:  training_data_topup.parquet  (adds columns in-place, backs up original)
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, re, sys

CACHE_DIR   = "./tess_cache"
PARQUET_IN  = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                   "training_data_topup.parquet")
PARQUET_OUT = PARQUET_IN.replace(".parquet", "_bkg.parquet")
N_WORKERS   = 8

# ── Scan for all LC files ──────────────────────────────────────────────────────
print(f"Scanning {CACHE_DIR} for LC FITS files...")
lc_files = []
for root, dirs, files in os.walk(CACHE_DIR):
    for fn in files:
        if fn.endswith("_lc.fits"):
            lc_files.append(os.path.join(root, fn))
print(f"Found {len(lc_files):,} files")

# ── Extract features from one file ────────────────────────────────────────────
SCATTER_BIT = 4096   # QUALITY bit 13

def extract(path):
    try:
        with fits.open(path, memmap=True) as hdul:
            # TIC ID and sector from primary header (more reliable than filename)
            tic_id = int(hdul[0].header.get("TICID", 0))
            sector = int(hdul[0].header.get("SECTOR", 0))
            teff   = hdul[0].header.get("TEFF")
            try:
                teff = float(teff) if teff is not None else np.nan
            except (TypeError, ValueError):
                teff = np.nan

            data    = hdul[1].data
            bkg     = np.array(data["SAP_BKG"],  dtype=np.float64)
            quality = np.array(data["QUALITY"],   dtype=np.int32)
            h1 = hdul[1].header
            def _hflt(key):
                v = h1.get(key)
                try:    return float(v)
                except: return np.nan

            flfrcsap = _hflt("FLFRCSAP")
            pdc_noi  = _hflt("PDC_NOI")
            pdc_corp = _hflt("PDC_CORP")
            pdc_totp = _hflt("PDC_TOTP")
            pr_wght2 = _hflt("PR_WGHT2")

        finite_bkg = bkg[np.isfinite(bkg)]
        if len(finite_bkg) == 0:
            return None

        median_bkg        = float(np.median(finite_bkg))
        p90_bkg           = float(np.percentile(finite_bkg, 90))
        bkg_rms           = float(np.std(finite_bkg))
        scatter_flag_frac = float(np.mean((quality & SCATTER_BIT) > 0))

        return {
            "tic_id":             tic_id,
            "sector":             sector,
            "median_sap_bkg":     median_bkg,
            "p90_sap_bkg":        p90_bkg,
            "bkg_rms":            bkg_rms,
            "scatter_flag_frac":  scatter_flag_frac,
            "flfrcsap":           flfrcsap,
            "teff":               teff,
            "pdc_noi":            pdc_noi,
            "pdc_corp":           pdc_corp,
            "pdc_totp":           pdc_totp,
            "pr_wght2":           pr_wght2,
        }
    except Exception as e:
        return None

# ── Parallel extraction ────────────────────────────────────────────────────────
print(f"\nExtracting features ({N_WORKERS} workers)...")
rows = []
done = 0
errors = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(extract, p): p for p in lc_files}
    for f in as_completed(futs):
        done += 1
        r = f.result()
        if r is not None:
            rows.append(r)
        else:
            errors += 1
        if done % 2000 == 0:
            print(f"  {done:>6,}/{len(lc_files):,}  extracted={len(rows):,}  errors={errors}")

print(f"\nDone: {len(rows):,} records extracted, {errors} errors")

feat_df = pd.DataFrame(rows)
feat_df["tic_id"] = feat_df["tic_id"].astype(np.int64)
feat_df["sector"] = feat_df["sector"].astype(np.int32)

print(f"\nFeature summary:")
for col in ["median_sap_bkg","p90_sap_bkg","bkg_rms","scatter_flag_frac","flfrcsap","teff",
            "pdc_noi","pdc_corp","pdc_totp","pr_wght2"]:
    s = feat_df[col]
    print(f"  {col:<22}  median={s.median():.3f}  std={s.std():.3f}  nan%={s.isna().mean()*100:.1f}%")

# ── Join onto parquet ──────────────────────────────────────────────────────────
print(f"\nLoading {PARQUET_IN}...")
df = pd.read_parquet(PARQUET_IN)
print(f"  {len(df):,} records, {df['tic_id'].nunique():,} stars")

df["tic_id_int"] = df["tic_id"].astype(np.int64)
df["sector_int"] = df["sector"].astype(np.int32)
feat_df = feat_df.rename(columns={"tic_id": "tic_id_int", "sector": "sector_int"})

before_cols = set(df.columns)
df = df.merge(feat_df, on=["tic_id_int","sector_int"], how="left")
df = df.drop(columns=["tic_id_int","sector_int"])

new_cols = [c for c in df.columns if c not in before_cols]
print(f"\nNew columns added: {new_cols}")
print(f"Match rate:")
for col in new_cols:
    filled = df[col].notna().mean() * 100
    print(f"  {col:<22}  {filled:.1f}% non-null")

print(f"\nSaving → {PARQUET_OUT}")
df.to_parquet(PARQUET_OUT, index=False)
print(f"Done. {len(df):,} records saved.")
print(f"\nTo retrain with new features, update CONTINUOUS in train_flow_nsf.py:")
print(f"  add: 'pr_wght2', 'pdc_noi', 'pdc_corp', 'pdc_totp', 'flfrcsap', 'teff'")
print(f"  (skip sap_bkg features — near-zero correlation with flux_offset)")
