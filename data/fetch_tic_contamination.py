"""
Fetch contratio, numcont, gaiabp from MAST TIC for all stars in
training_data_v3.parquet, then write training_data_v3_tic.parquet with
those columns joined in.

  contratio  — TIC fixed-aperture contamination ratio (fraction of
                aperture flux from non-target sources)
  numcont    — number of contaminating TIC sources in aperture
  gaiabp     — Gaia Bp magnitude
  bp_rp      — gaiabp − gaiarp (colour index; gaiarp already in parquet)

Usage:
  python3 data/fetch_tic_contamination.py
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astroquery.mast import Catalogs

PARQUET  = "training_data_v3.parquet"
CACHE    = "/private/tmp/tic_contamination_cache.parquet"
OUT      = "training_data_v3_tic.parquet"
CHUNK    = 1000
FIELDS   = ["ID", "contratio", "numcont", "gaiaBPmag"]   # TIC column names

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

log(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
tic_ids = df["tic_id"].dropna().astype(int).unique().tolist()
log(f"  {len(df):,} rows  |  {len(tic_ids):,} unique TIC IDs")

# ── Resume from cache if available ───────────────────────────────────────────
if os.path.exists(CACHE):
    cached = pd.read_parquet(CACHE)
    done_ids = set(cached["tic_id"].astype(int))
    remaining = [t for t in tic_ids if t not in done_ids]
    log(f"Resuming: {len(done_ids):,} already cached, {len(remaining):,} remaining")
    records = [cached]
else:
    remaining = tic_ids
    records = []

# ── Query in chunks ───────────────────────────────────────────────────────────
chunks = [remaining[i:i+CHUNK] for i in range(0, len(remaining), CHUNK)]
log(f"Querying MAST TIC in {len(chunks)} chunks of ≤{CHUNK} …")

for k, chunk in enumerate(chunks):
    try:
        res = Catalogs.query_criteria(catalog="TIC", ID=chunk)
        # TIC returns gaiaBPmag, not gaiabp
        avail = [c for c in FIELDS if c in res.colnames]
        sub = res[avail].to_pandas()
        for col in avail:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub["ID"] = sub["ID"].astype("Int64")
        sub = sub.rename(columns={
            "ID": "tic_id",
            "gaiaBPmag": "gaiabp",
        })
        sub = sub.drop_duplicates("tic_id")
        records.append(sub)
    except Exception as e:
        log(f"  chunk {k+1}/{len(chunks)} FAILED: {e}")
        continue

    if (k + 1) % 20 == 0 or k == len(chunks) - 1:
        tic_df = pd.concat(records, ignore_index=True)
        tic_df.to_parquet(CACHE, index=False)
        nn = tic_df["contratio"].notna().sum()
        log(f"  {k+1}/{len(chunks)} done  ({len(tic_df):,} rows, {nn:,} with contratio)  {T()}")

# ── Final assembly ────────────────────────────────────────────────────────────
tic_df = pd.concat(records, ignore_index=True).drop_duplicates("tic_id")
tic_df["tic_id"] = tic_df["tic_id"].astype("Int64")

log(f"\nTIC query complete: {len(tic_df):,} rows")
for col in ["contratio", "numcont", "gaiabp"]:
    if col in tic_df.columns:
        nn = tic_df[col].notna().sum()
        log(f"  {col}: {nn:,} / {len(tic_df):,} non-null ({nn/len(tic_df)*100:.1f}%)")

# Compute bp_rp
if "gaiabp" in tic_df.columns and "gaiarp" in df.columns:
    # gaiarp is per-observation in df; get per-star value
    gaiarp_star = df.groupby("tic_id")["gaiarp"].first().reset_index()
    gaiarp_star["tic_id"] = gaiarp_star["tic_id"].astype("Int64")
    tic_df = tic_df.merge(gaiarp_star, on="tic_id", how="left")
    tic_df["bp_rp"] = tic_df["gaiabp"] - tic_df["gaiarp"]
    log(f"  bp_rp: {tic_df['bp_rp'].notna().sum():,} non-null")

# ── Join to main parquet ──────────────────────────────────────────────────────
log(f"\nJoining to {PARQUET} …")
df["tic_id_int"] = df["tic_id"].astype("Int64")
tic_df["tic_id_int"] = tic_df["tic_id"].astype("Int64")
join_cols = [c for c in ["tic_id_int", "contratio", "numcont", "gaiabp", "bp_rp"]
             if c in tic_df.columns]
out = df.merge(tic_df[join_cols], on="tic_id_int", how="left")
out = out.drop(columns=["tic_id_int"])

log(f"  Output shape: {out.shape}")
for col in ["contratio", "numcont", "gaiabp", "bp_rp"]:
    if col in out.columns:
        nn = out[col].notna().sum()
        log(f"  {col}: {nn:,} / {len(out):,} non-null ({nn/len(out)*100:.1f}%)")

out.to_parquet(OUT, index=False)
log(f"\nSaved → {OUT}  {T()}")
log("DONE")
