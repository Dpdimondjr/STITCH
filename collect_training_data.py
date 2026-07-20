"""
Collect training data for STITCH from top-ranked stars in training_stars.csv.

For each star:
  1. Download all SPOC PDCSAP light curves via lightkurve (one call per star).
     All stars in training_stars.csv are guaranteed to have SPOC products.
  2. Compute leave-one-out flux offset per sector:
       flux_offset(S) = median(LC_S) / mean(medians of all other sectors)
  3. Extract scalar features from each sector's LC (header + POS_CORR cols).
  4. Apply stability filter (CDPP, PDCVAR, CROWDSAP).
  5. Write each star's results immediately to its own CSV in STAR_CACHE_DIR.
     (Crash-safe: partially completed runs resume automatically.)
  6. Consolidate all per-star CSVs into training_data.parquet.

col/row come from training_pairs.csv (tess-point, no TPF needed).
delta_sub_col/row = (col + median(POS_CORR)) % 1  — fractional pixel position.
"""

import os, sys, warnings, threading
import numpy as np
import pandas as pd
import lightkurve as lk
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

STARS_CSV       = "training_stars.csv"
PAIRS_CSV       = "training_pairs.csv"
OUT_FILE        = "training_data.parquet"
CACHE_DIR       = "./tess_cache"
STAR_CACHE_DIR  = "./tess_cache/star_records"  # per-star CSVs go here

N_PER_CAM  = 150   # top stars per camera — 4 cams × 150 = up to 600, camera-balanced
TEFF_MIN   = 4500  # exclude cool active stars (M/K dwarfs dominate PDCVAR failures)
N_WORKERS  = 2
MAST_SEM   = threading.Semaphore(2)

CDPP_MAX     = 1000.0
PDCVAR_MAX   = 1.5    # loosened from 1.0; stars slightly above 1.0 still have reliable sector medians
CROWDSAP_MIN = 0.5

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STAR_CACHE_DIR, exist_ok=True)


# ── Load catalogs ─────────────────────────────────────────────────────────────

def log(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass   # stdout pipe closed — keep running, don't crash


log("Loading catalogs...")
stars = pd.read_csv(STARS_CSV)
pairs = pd.read_csv(PAIRS_CSV)

# Apply Teff pre-filter if available (removes cool active stars that fail PDCVAR)
if "Teff" in stars.columns:
    n_before = stars["tic_id"].nunique()
    stars = stars[stars["Teff"].isna() | (stars["Teff"] >= TEFF_MIN)]
    log(f"  Teff filter (>={TEFF_MIN}K): {stars['tic_id'].nunique()} / {n_before} unique stars kept")
else:
    log("  Teff not in catalog — skipping temperature filter (run enrich_catalog.py first)")

# Per-camera quota: top N_PER_CAM stars per camera, then deduplicate by tic_id
# Guarantees Camera 1/2/3 representation instead of all-Camera-4
top_tics = (
    stars.sort_values("training_score", ascending=False)
         .groupby("cam", group_keys=False)
         .head(N_PER_CAM)
         .drop_duplicates(subset="tic_id")["tic_id"]
         .tolist()
)
log(f"  Per-camera quota ({N_PER_CAM}/cam): {len(top_tics)} unique stars selected")
cam_counts = (stars[stars["tic_id"].isin(top_tics)]
              .drop_duplicates(subset="tic_id")
              .groupby("cam")["tic_id"].nunique())
for cam, n in cam_counts.items():
    log(f"    Cam{int(cam)}: {n} stars")

pairs_dedup = pairs.drop_duplicates(subset=["tic_id", "sector"])
pos_lookup = (
    pairs_dedup
    .set_index(["tic_id", "sector"])[["col", "row", "cam", "ccd"]]
    .to_dict("index")
)

# Stars whose per-star CSV already exists → skip
done_tics = {
    int(f.replace("tic_", "").replace(".csv", ""))
    for f in os.listdir(STAR_CACHE_DIR)
    if f.startswith("tic_") and f.endswith(".csv")
}
if done_tics:
    log(f"  {len(done_tics)} stars already collected — skipping")
top_tics = [t for t in top_tics if t not in done_tics]
log(f"  {len(top_tics)} stars to collect\n")


# ── Per-star processing ───────────────────────────────────────────────────────

def process_star(tic_id):
    star_csv = os.path.join(STAR_CACHE_DIR, f"tic_{tic_id}.csv")

    with MAST_SEM:
        try:
            sr = lk.search_lightcurve(
                f"TIC {tic_id}", mission="TESS", author="SPOC",
            )
            if len(sr) == 0:
                return f"TIC {tic_id}: no SPOC light curves"
            # Drop 20-second fast-cadence products (exptime≈20s); keep 2-min (120s) and 10-min (600s)
            try:
                sr = sr[sr.exptime.value >= 100]
            except Exception:
                pass  # exptime not available — proceed with full set
            if len(sr) == 0:
                return f"TIC {tic_id}: no 2-min/10-min SPOC light curves"
            # Download one at a time so a single bad FITS file doesn't abort the whole star
            lc_list = []
            for i in range(len(sr)):
                try:
                    lc = sr[i].download(download_dir=CACHE_DIR)
                    if lc is not None:
                        lc_list.append(lc)
                except Exception:
                    pass
        except Exception as e:
            return f"TIC {tic_id}: search error — {e}"

    if not lc_list:
        return f"TIC {tic_id}: no LCs successfully downloaded"

    # Per-sector medians (for leave-one-out reference)
    sector_meds = {}
    for lc in lc_list:
        try:
            sec = lc.meta.get("SECTOR")
            if sec is None:
                continue
            flux = lc.flux.value
            med = np.nanmedian(flux)
            if np.isfinite(med) and med > 0:
                sector_meds[sec] = med
        except Exception:
            continue

    if len(sector_meds) < 2:
        return f"TIC {tic_id}: <2 usable sectors ({len(sector_meds)} found)"

    # Debug counters (only used in 0-record case)
    n_lookup_hits = sum(1 for s in sector_meds if pos_lookup.get((tic_id, s)) is not None)
    n_cdpp_fail = n_pdcvar_fail = n_crowd_fail = 0

    records = []
    for lc in lc_list:
        try:
            sec = lc.meta.get("SECTOR")
            cam = lc.meta.get("CAMERA")
            ccd = lc.meta.get("CCD")

            if sec is None or sec not in sector_meds:
                continue

            ref_vals = [v for s, v in sector_meds.items() if s != sec]
            if not ref_vals:
                continue
            global_med   = float(np.mean(ref_vals))
            flux_offset  = sector_meds[sec] / global_med

            cdpp1_0  = lc.meta.get("CDPP1_0")
            pdcvar   = lc.meta.get("PDCVAR")
            crowdsap = lc.meta.get("CROWDSAP")
            tmag     = lc.meta.get("TESSMAG")

            if cdpp1_0  is not None and cdpp1_0  > CDPP_MAX:
                n_cdpp_fail += 1; continue
            if pdcvar   is not None and pdcvar   > PDCVAR_MAX:
                n_pdcvar_fail += 1; continue
            if crowdsap is not None and crowdsap < CROWDSAP_MIN:
                n_crowd_fail += 1; continue

            try:
                pc1 = lc["pos_corr1"].value.astype(float)
                pc2 = lc["pos_corr2"].value.astype(float)
                jitter_rms = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))
                pc1_med, pc2_med = float(np.nanmedian(pc1)), float(np.nanmedian(pc2))
            except Exception:
                jitter_rms = pc1_med = pc2_med = np.nan

            pos = pos_lookup.get((tic_id, sec))
            if pos is None:
                continue
            col, row = pos["col"], pos["row"]

            records.append({
                "tic_id":         tic_id,
                "sector":         sec,
                "cam":            cam,
                "ccd":            ccd,
                "col":            col,
                "row":            row,
                "delta_sub_col":  (col + pc1_med) % 1.0 if np.isfinite(pc1_med) else np.nan,
                "delta_sub_row":  (row + pc2_med) % 1.0 if np.isfinite(pc2_med) else np.nan,
                "tmag":           tmag,
                "crowdsap":       crowdsap,
                "cdpp1_0":        cdpp1_0,
                "pdcvar":         pdcvar,
                "jitter_rms":     jitter_rms,
                "flux_offset":    flux_offset,
                "n_sectors_total": len(sector_meds),
            })
        except Exception:
            continue

    if records:
        pd.DataFrame(records).to_csv(star_csv, index=False)

    debug = (f" (lookup={n_lookup_hits}/{len(sector_meds)}"
             f" cdpp_fail={n_cdpp_fail} pdcvar_fail={n_pdcvar_fail}"
             f" crowd_fail={n_crowd_fail})") if not records else ""
    return (f"TIC {tic_id}: {len(records)} records "
            f"from {len(sector_meds)} sectors{debug}")


# ── Main collection loop ──────────────────────────────────────────────────────

log(f"Collecting ({N_WORKERS} workers, SPOC only)...\n")

with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
    futures = {pool.submit(process_star, tic): tic for tic in top_tics}
    n_done = 0
    for future in as_completed(futures):
        msg = future.result()
        n_done += 1
        log(f"  [{n_done:3d}/{len(top_tics)}] {msg}")


# ── Consolidate all per-star CSVs into parquet ────────────────────────────────

log("\nConsolidating per-star CSVs → parquet...")

csv_files = [
    os.path.join(STAR_CACHE_DIR, f)
    for f in os.listdir(STAR_CACHE_DIR)
    if f.startswith("tic_") and f.endswith(".csv")
]

if not csv_files:
    log("No per-star CSVs found. Check MAST connectivity.")
    sys.exit(1)

frames = []
for p in csv_files:
    try:
        frames.append(pd.read_csv(p))
    except Exception:
        pass

final = (pd.concat(frames, ignore_index=True)
           .drop_duplicates(subset=["tic_id", "sector"])
           .sort_values(["tic_id", "sector"])
           .reset_index(drop=True))

final.to_parquet(OUT_FILE, index=False)

log(f"\n=== Training Data Summary ===")
log(f"  Total (star, sector) records:  {len(final)}")
log(f"  Unique stars:                  {final['tic_id'].nunique()}")
log(f"  Sectors covered:               {final['sector'].min()}–{final['sector'].max()}")
log(f"  Cam/CCD coverage:")
for (cam, ccd), g in final.groupby(["cam", "ccd"]):
    log(f"    Cam{cam}/CCD{ccd}: {len(g):4d} records")
log(f"  flux_offset range:  {final['flux_offset'].min():.4f} – "
    f"{final['flux_offset'].max():.4f}")
log(f"  flux_offset std:    {final['flux_offset'].std():.4f}")
log(f"\nSaved → {OUT_FILE}  ({os.path.getsize(OUT_FILE)/1024:.1f} KB)")
log("Next: visualize flux_offset distribution, then train the flow.")
