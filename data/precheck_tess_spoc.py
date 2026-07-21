"""
Pre-check which TICs in tars_quiet_tics.csv actually have TESS-SPOC products.

MAST metadata queries are fast (no download). Running 16 workers in parallel,
this should scan 8000 TICs in ~20-30 minutes and produce tars_spoc_valid.csv
containing only stars with confirmed TESS-SPOC coverage.
"""

import os, threading, warnings
import pandas as pd
import lightkurve as lk
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

STARS_CSV   = "tars_quiet_tics.csv"
OUT_CSV     = "tars_spoc_valid.csv"
N_WORKERS   = 16
MAST_SEM    = threading.Semaphore(16)
CACHE_FILE  = "precheck_cache.csv"   # resume support

def log(msg):
    print(msg, flush=True)

# ── Load catalog ──────────────────────────────────────────────────────────────

stars = pd.read_csv(STARS_CSV)
all_tics = stars["tic_id"].tolist()
log(f"Loaded {len(all_tics):,} TICs from {STARS_CSV}")

# ── Resume: skip already-checked TICs ────────────────────────────────────────

if os.path.exists(CACHE_FILE):
    cache = pd.read_csv(CACHE_FILE)
    checked = set(cache["tic_id"].tolist())
    log(f"Resuming: {len(checked):,} already checked, {len(all_tics)-len(checked):,} remaining")
else:
    cache = pd.DataFrame(columns=["tic_id", "has_spoc", "n_spoc_sectors"])
    checked = set()

remaining = [t for t in all_tics if t not in checked]
log(f"Checking {len(remaining):,} TICs with {N_WORKERS} workers...\n")

# ── Per-TIC check ─────────────────────────────────────────────────────────────

_lock = threading.Lock()
results = []
FLUSH_EVERY = 200

def check_tic(tic_id):
    with MAST_SEM:
        try:
            sr = lk.search_lightcurve(
                f"TIC {tic_id}", mission="TESS", author="TESS-SPOC"
            )
            n = len(sr)
            return {"tic_id": tic_id, "has_spoc": n > 0, "n_spoc_sectors": n}
        except Exception:
            return {"tic_id": tic_id, "has_spoc": False, "n_spoc_sectors": 0}

n_done = 0
n_hits = 0

with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(check_tic, tic): tic for tic in remaining}
    for future in as_completed(futures):
        res = future.result()
        results.append(res)
        n_done += 1
        if res["has_spoc"]:
            n_hits += 1

        if n_done % 50 == 0:
            log(f"  [{n_done:5d}/{len(remaining)}]  hits so far: {n_hits} ({n_hits/n_done*100:.0f}%)")

        if n_done % FLUSH_EVERY == 0:
            with _lock:
                new_rows = pd.DataFrame(results[-FLUSH_EVERY:])
                updated = pd.concat([cache, new_rows], ignore_index=True)
                updated.to_csv(CACHE_FILE, index=False)

# ── Final flush ───────────────────────────────────────────────────────────────

all_results = pd.concat(
    [cache, pd.DataFrame(results)], ignore_index=True
).drop_duplicates(subset="tic_id")

all_results.to_csv(CACHE_FILE, index=False)

# ── Build valid list and save ─────────────────────────────────────────────────

valid_tics = all_results[all_results["has_spoc"]]["tic_id"].tolist()

# Merge back with catalog to keep cam/ccd/n_quiet_sectors columns
valid_df = stars[stars["tic_id"].isin(set(valid_tics))].copy()
valid_df = valid_df.merge(
    all_results[["tic_id", "n_spoc_sectors"]], on="tic_id", how="left"
)
valid_df.to_csv(OUT_CSV, index=False)

total_checked = len(all_results)
log(f"\n=== Pre-check Complete ===")
log(f"  Checked:          {total_checked:,} TICs")
log(f"  Have TESS-SPOC:   {len(valid_tics):,} ({len(valid_tics)/total_checked*100:.0f}%)")
log(f"  No TESS-SPOC:     {total_checked-len(valid_tics):,}")
log(f"\nPer cam-CCD breakdown:")
for (cam, ccd), g in valid_df.groupby(["cam", "ccd"]):
    log(f"  Cam{int(cam)}/CCD{int(ccd)}: {len(g):4d} stars  avg {g['n_spoc_sectors'].mean():.1f} SPOC sectors")
log(f"\nSaved → {OUT_CSV}")
log("Next: run collect_training_data_v2.py with STARS_CSV = 'tars_spoc_valid.csv'")
