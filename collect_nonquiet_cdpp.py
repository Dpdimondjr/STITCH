"""
Collect CDPP1_0 for a sample of non-quiet TESS-SPOC stars.

Non-quiet TIC IDs come from TARS table 2 (sys_score < 0.3) intersected
with TIC IDs confirmed present in TESS-SPOC sectors 1-10.
"""

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import lightkurve as lk
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

CACHE_FILE = "nonquiet_cdpp_cache.parquet"
N_SAMPLE   = 300   # non-quiet stars to attempt
N_WORKERS  = 8

# ── Load TARS table 2 ──────────────────────────────────────────────────────
print("Loading TARS table 2...", flush=True)
tars2 = feather.read_table("/Users/daviddimond/Documents/STITCH/tars_table_2.feather").to_pandas()

# All stars in table 2 have sys_score near 0 (non-quiet)
print(f"Non-quiet catalog: {len(tars2):,} stars", flush=True)

# ── Load TESS-SPOC confirmed TIC set (sectors 1-10) ───────────────────────
with open("/tmp/tess_spoc_tics_s1_10.pkl", "rb") as f:
    spoc_tics = pickle.load(f)

# ── Intersect: non-quiet with confirmed TESS-SPOC coverage ────────────────
nq_spoc = tars2[tars2["TICID"].isin(spoc_tics)].copy()
print(f"Non-quiet with TESS-SPOC s1-10: {len(nq_spoc):,}", flush=True)

# Load quiet stars for Tmag reference
train = pd.read_parquet("/Users/daviddimond/Documents/STITCH/training_data.parquet")
quiet_tmag = train.groupby("tic_id")["tmag"].median()
quiet_tmag_range = (quiet_tmag.min(), quiet_tmag.max())
print(f"Quiet star Tmag range: {quiet_tmag_range[0]:.1f} – {quiet_tmag_range[1]:.1f}", flush=True)

# Sample non-quiet stars in same Tmag range, stratified by Tmag
mask = (nq_spoc["Tmag"] >= quiet_tmag_range[0]) & (nq_spoc["Tmag"] <= quiet_tmag_range[1])
nq_matched = nq_spoc[mask].copy()
print(f"Tmag-matched non-quiet stars: {len(nq_matched):,}", flush=True)

# Stratified sample by Tmag bins
rng = np.random.default_rng(42)
bins = [(7,9), (9,11), (11,13)]
sample_per_bin = N_SAMPLE // len(bins)
sample_tics = []
for lo, hi in bins:
    sub = nq_matched[(nq_matched["Tmag"]>=lo) & (nq_matched["Tmag"]<hi)]
    n = min(sample_per_bin, len(sub))
    idx = rng.choice(len(sub), size=n, replace=False)
    sample_tics.extend(sub.iloc[idx]["TICID"].tolist())

print(f"Sample size: {len(sample_tics)} TICs", flush=True)

# ── Download CDPP for each star ───────────────────────────────────────────

def fetch_cdpp(tic_id):
    try:
        sr = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS",
            author="TESS-SPOC",
            cadence=1800,
        )
        if len(sr) == 0:
            return None
        # Use first available sector
        lc = sr[0].download(quality_bitmask="default")
        if lc is None:
            return None
        lc = lc.normalize().remove_nans().remove_outliers()
        cdpp = float(lc.estimate_cdpp(transit_duration=1))  # 1-hour CDPP in ppm
        flux_med = float(np.nanmedian(lc.flux.value))
        # Get Tmag
        tmag_row = nq_matched[nq_matched["TICID"] == tic_id]["Tmag"]
        tmag = float(tmag_row.iloc[0]) if len(tmag_row) > 0 else np.nan
        return {"tic_id": tic_id, "cdpp1_0": cdpp, "flux_norm": flux_med, "tmag": tmag}
    except Exception:
        return None

results = []
done = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(fetch_cdpp, tic): tic for tic in sample_tics}
    for fut in as_completed(futures):
        done += 1
        r = fut.result()
        if r is not None and 0 < r["cdpp1_0"] < 1e5:
            results.append(r)
        if done % 20 == 0:
            print(f"  {done}/{len(sample_tics)}  valid={len(results)}", flush=True)

df_nq = pd.DataFrame(results)
print(f"\nCollected: {len(df_nq)} non-quiet stars with valid CDPP")
df_nq.to_parquet(CACHE_FILE)
print(f"Saved → {CACHE_FILE}")
print(df_nq[["tmag","cdpp1_0"]].describe())
