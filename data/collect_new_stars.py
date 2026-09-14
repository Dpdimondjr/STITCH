"""
Phase-2 data collection: discover NEW stars (not in current parquet) with many TESS sectors.

Strategy:
  1. Query MAST TESS-SPOC observations at the CVZ regions (both ecliptic poles).
  2. Find TIC IDs with >= MIN_MAST_SECTORS available SPOC sectors.
  3. Cross-match against TIC catalog to get Tmag; keep tmag <= TMAG_MAX.
  4. Exclude TICs already in the current parquet.
  5. Sort by n_mast_sectors descending (most-observed first → lowest label noise).
  6. Download ALL sectors for each new star; compute LOO from scratch.

Usage:
    python3 data/collect_new_stars.py [parquet] [--min-sectors N] [--shard 0 4]
    python3 data/collect_new_stars.py training_data_topup_pdc.parquet --min-sectors 15
"""

import os, sys, warnings, glob, shutil, socket, traceback
socket.setdefaulttimeout(60)
import numpy as np
import pandas as pd
import lightkurve as lk
from tess_stars2px import tess_stars2px_function_entry

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_IN      = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                       "training_data_topup_pdc.parquet")
PARQUET_OUT     = PARQUET_IN.replace(".parquet", "_newstars.parquet")

CACHE_DIR       = "./tess_cache"
STARS_DIR       = "./tess_cache/star_records_new_stars"

N_WORKERS        = 3
CHECKPOINT_EVERY = 25
CDPP_MAX         = 2000.0
CROWDSAP_MIN     = 0.3
TMAG_MAX         = 13.0
MIN_SECTORS      = 2        # minimum sectors to keep a star in output

MIN_MAST_SECTORS = 15
if "--min-sectors" in sys.argv:
    p = sys.argv.index("--min-sectors")
    MIN_MAST_SECTORS = int(sys.argv[p + 1])

_shard_idx, _shard_n = 0, 1
if "--shard" in sys.argv:
    p = sys.argv.index("--shard")
    _shard_idx, _shard_n = int(sys.argv[p + 1]), int(sys.argv[p + 2])

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STARS_DIR, exist_ok=True)

LOG_PATH = f"/tmp/stitch_new_stars_s{_shard_idx}.log"
_log_fh = open(LOG_PATH, "a", buffering=1)

def log(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass
    try:
        _log_fh.write(msg + "\n"); _log_fh.flush()
    except Exception:
        pass

TARS_CSV = "tars_quiet_tics_v2.csv"

# ── TARS discovery ─────────────────────────────────────────────────────────────
def discover_candidates(existing_tics):
    """
    Load TARS-vetted quiet TIC IDs (systematic_score > 0.95) and filter to
    new stars not already in the parquet. Sort by n_quiet_sectors descending
    so the most-observed quiet stars are processed first.
    """
    if not os.path.exists(TARS_CSV):
        log(f"  ERROR: {TARS_CSV} not found. Run data/build_tars_catalog.py first.")
        return pd.DataFrame(columns=["tic_id", "n_quiet_sectors"])

    tars = pd.read_csv(TARS_CSV)
    if "TICID" in tars.columns:
        tars = tars.rename(columns={"TICID": "tic_id"})
    tars["tic_id"] = tars["tic_id"].astype(int)
    log(f"  TARS catalog: {len(tars):,} quiet TIC entries")

    tars = tars[~tars["tic_id"].isin(existing_tics)].reset_index(drop=True)
    log(f"  {len(tars):,} TICs not already in parquet")

    # Sort by n_quiet_sectors descending — best TARS proxy for many sectors.
    # No pre-filter: TARS only covers through ~sector 50, so a star with few
    # TARS quiet sectors may still have 80+ sectors on MAST.
    # Post-download filter (MIN_MAST_SECTORS) handles actual sector count.
    if "n_quiet_sectors" in tars.columns:
        tars = tars.sort_values("n_quiet_sectors", ascending=False)

    return tars


# ── Load existing parquet ─────────────────────────────────────────────────────
log(f"Loading {PARQUET_IN} …")
existing = pd.read_parquet(PARQUET_IN)
existing = existing[existing["sector"] <= 200]
existing_tics = set(existing["tic_id"].unique().astype(int))
log(f"  {len(existing):,} records  ·  {len(existing_tics):,} existing TICs")

# Discover candidates
candidates = discover_candidates(existing_tics)
if len(candidates) == 0:
    log("No new candidates found. Exiting.")
    sys.exit(0)

log(f"\n  {len(candidates):,} candidates to process (TARS-quiet, sorted by n_quiet_sectors desc)")

all_tics = candidates["tic_id"].tolist()

# Resume: skip already-processed TICs
done_tics = {
    int(f.replace("tic_", "").replace(".csv", "").replace(".nodata", ""))
    for f in os.listdir(STARS_DIR)
    if f.startswith("tic_") and (f.endswith(".csv") or f.endswith(".nodata"))
}
if done_tics:
    log(f"\n  {len(done_tics):,} TICs already processed — skipping")
all_tics = [t for t in all_tics if t not in done_tics]
all_tics = all_tics[_shard_idx::_shard_n]
log(f"  {len(all_tics):,} TICs to process (shard {_shard_idx}/{_shard_n})\n")


# ── Per-star processing ───────────────────────────────────────────────────────
def process_star(tic_id):
    out_csv = os.path.join(STARS_DIR, f"tic_{tic_id}.csv")
    out_nod = os.path.join(STARS_DIR, f"tic_{tic_id}.nodata")

    def mark_nodata():
        open(out_nod, "w").close()

    try:
        sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
        try:
            sr = sr[sr.exptime.value >= 100]
        except Exception:
            pass
        if len(sr) == 0:
            mark_nodata()
            return f"TIC {tic_id}: no SPOC LCs"
    except Exception as e:
        return f"TIC {tic_id}: MAST error — {e}"

    def _sector_num(r):
        try:
            m = str(r.mission[0])
            parts = m.split()
            return int(parts[-1]) if parts[-1].isdigit() else None
        except Exception:
            return None

    # Download ALL available sectors
    all_lcs = []
    for idx in range(len(sr)):
        try:
            lc = sr[idx].download(download_dir=CACHE_DIR)
            if lc is not None:
                all_lcs.append(lc)
        except Exception:
            pass

    if not all_lcs:
        mark_nodata()
        return f"TIC {tic_id}: download failed"

    ra  = all_lcs[0].meta.get("RA_OBJ")
    dec = all_lcs[0].meta.get("DEC_OBJ")
    if ra is None or dec is None:
        mark_nodata()
        return f"TIC {tic_id}: no RA/DEC"

    try:
        _, _, _, outSec, outCam, outCcd, outCol, outRow, _ = \
            tess_stars2px_function_entry(int(tic_id), float(ra), float(dec))
        pos_lookup = {int(s): (int(c1), int(c2), float(cl), float(rw))
                      for s, c1, c2, cl, rw in zip(outSec, outCam, outCcd, outCol, outRow)}
    except Exception as e:
        mark_nodata()
        return f"TIC {tic_id}: tess_stars2px error — {e}"

    sec_meds = {}
    sec_meta = {}

    for lc in all_lcs:
        try:
            sec  = int(lc.meta.get("SECTOR"))
            flux = lc.flux.value
            med  = float(np.nanmedian(flux))
            if not (np.isfinite(med) and med > 0):
                continue

            def _flt(v):
                try: return float(v)
                except: return None

            cdpp1_0  = _flt(lc.meta.get("CDPP1_0"))
            crowdsap = _flt(lc.meta.get("CROWDSAP"))
            tmag     = _flt(lc.meta.get("TESSMAG"))
            pdcvar   = _flt(lc.meta.get("PDCVAR"))

            if tmag is not None and tmag > TMAG_MAX: continue
            if cdpp1_0  is not None and cdpp1_0  > CDPP_MAX: continue
            if crowdsap is not None and crowdsap < CROWDSAP_MIN: continue

            try:
                pc1 = lc["pos_corr1"].value.astype(float)
                pc2 = lc["pos_corr2"].value.astype(float)
                jitter_rms = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))
                pc1_med = float(np.nanmedian(pc1))
                pc2_med = float(np.nanmedian(pc2))
            except Exception:
                jitter_rms = pc1_med = pc2_med = np.nan

            pos = pos_lookup.get(sec)
            cam_tp = lc.meta.get("CAMERA")
            ccd_tp = lc.meta.get("CCD")
            if pos is not None:
                cam_tp, ccd_tp, col, row = pos
            else:
                col = row = np.nan

            try:
                bkg_arr = lc["sap_bkg"].value.astype(float)
                qual    = lc.quality.value.astype(int)
                finite_bkg = bkg_arr[np.isfinite(bkg_arr)]
                median_sap_bkg    = float(np.median(finite_bkg)) if len(finite_bkg) else np.nan
                p90_sap_bkg       = float(np.percentile(finite_bkg, 90)) if len(finite_bkg) else np.nan
                bkg_rms           = float(np.std(finite_bkg)) if len(finite_bkg) else np.nan
                scatter_flag_frac = float(np.mean((qual & 4096) > 0))
            except Exception:
                median_sap_bkg = p90_sap_bkg = bkg_rms = scatter_flag_frac = np.nan

            sec_meds[sec] = med
            sec_meta[sec] = {
                "tic_id":            tic_id,
                "sector":            sec,
                "cam":               cam_tp,
                "ccd":               ccd_tp,
                "col":               col,
                "row":               row,
                "delta_sub_col":     (col + pc1_med) % 1.0 if (np.isfinite(col) and np.isfinite(pc1_med)) else np.nan,
                "delta_sub_row":     (row + pc2_med) % 1.0 if (np.isfinite(row) and np.isfinite(pc2_med)) else np.nan,
                "tmag":              tmag,
                "crowdsap":          crowdsap,
                "cdpp1_0":           cdpp1_0,
                "pdcvar":            pdcvar,
                "jitter_rms":        jitter_rms,
                "sector_median":     med,
                "ra":                ra,
                "dec":               dec,
                "median_sap_bkg":    median_sap_bkg,
                "p90_sap_bkg":       p90_sap_bkg,
                "bkg_rms":           bkg_rms,
                "scatter_flag_frac": scatter_flag_frac,
                "flfrcsap":          _flt(lc.meta.get("FLFRCSAP")),
                "teff":              _flt(lc.meta.get("TEFF")),
                "pdc_noi":           _flt(lc.meta.get("PDC_NOI")),
                "pdc_corp":          _flt(lc.meta.get("PDC_CORP")),
                "pdc_totp":          _flt(lc.meta.get("PDC_TOTP")),
                "pr_wght2":          _flt(lc.meta.get("PR_WGHT2")),
            }
        except Exception:
            continue

    total_sectors = len(sec_meds)
    if total_sectors < MIN_MAST_SECTORS:
        mark_nodata()
        return f"TIC {tic_id}: only {total_sectors} valid sectors (need {MIN_MAST_SECTORS})"

    # Compute LOO labels from scratch
    records = []
    for sec, meta in sec_meta.items():
        ref_vals = [v for s, v in sec_meds.items() if s != sec]
        if not ref_vals:
            continue
        ref_mean = float(np.mean(ref_vals))
        meta["flux_offset"]     = float(sec_meds[sec]) / ref_mean
        meta["flux_offset_loo"] = float(sec_meds[sec]) / ref_mean
        meta["ref_mean"]        = ref_mean
        meta["n_sectors_total"] = total_sectors
        records.append(meta)

    if not records:
        mark_nodata()
        return f"TIC {tic_id}: no valid records after LOO"

    pd.DataFrame(records).to_csv(out_csv, index=False)

    # Clean up MAST cache (both TESS/ and HLSP/ subdirs)
    for subdir in ["TESS", "HLSP"]:
        for d in glob.glob(os.path.join(CACHE_DIR, "mastDownload", subdir,
                                        f"*{int(tic_id):016d}*")):
            try: shutil.rmtree(d)
            except Exception: pass
        for d in glob.glob(os.path.join(CACHE_DIR, "mastDownload", subdir,
                                        f"*-{int(tic_id):010d}-*")):
            try: shutil.rmtree(d)
            except Exception: pass

    return f"TIC {tic_id}: {total_sectors} sectors collected"


# ── Consolidation ─────────────────────────────────────────────────────────────
def consolidate(label=""):
    frames = []
    for f in os.listdir(STARS_DIR):
        if f.startswith("tic_") and f.endswith(".csv"):
            try: frames.append(pd.read_csv(os.path.join(STARS_DIR, f)))
            except Exception: pass

    if not frames:
        log(f"  No new star records yet.")
        return

    new_stars = pd.concat(frames, ignore_index=True)
    # Merge with existing parquet (new stars don't overlap by construction)
    final = pd.concat([existing, new_stars], ignore_index=True)
    final = (final.drop_duplicates(subset=["tic_id", "sector"])
                  .sort_values(["tic_id", "sector"])
                  .reset_index(drop=True))
    final.to_parquet(PARQUET_OUT, index=False)
    tag = f" [{label}]" if label else ""
    n_new = new_stars["tic_id"].nunique()
    log(f"  ✓ Checkpoint{tag}: {len(final):,} records, {n_new:,} new stars → {PARQUET_OUT}")


# ── Main loop ─────────────────────────────────────────────────────────────────
import queue as _queue
from concurrent.futures import ThreadPoolExecutor as _TPE

log(f"Starting new-star collection ({N_WORKERS} workers)…\n")

_result_q = _queue.Queue()

def _worker(tic_id):
    try:
        _result_q.put(process_star(tic_id))
    except Exception as e:
        _result_q.put(f"TIC {tic_id}: unhandled — {e}")

n_done = 0
try:
    with _TPE(max_workers=N_WORKERS) as pool:
        tic_iter   = iter(all_tics)
        in_flight  = 0
        MAX_PENDING = max(N_WORKERS * 2, 2)

        for tic in tic_iter:
            pool.submit(_worker, tic)
            in_flight += 1
            if in_flight >= MAX_PENDING:
                break

        while in_flight > 0:
            try:
                msg = _result_q.get(timeout=120)
            except _queue.Empty:
                log("  [WARN] 120s no result")
                continue
            in_flight -= 1
            n_done += 1
            log(f"  [{n_done:5d}/{len(all_tics)}] {msg}")
            if n_done % CHECKPOINT_EVERY == 0:
                consolidate(f"{n_done}/{len(all_tics)}")
            try:
                pool.submit(_worker, next(tic_iter))
                in_flight += 1
            except StopIteration:
                pass
except Exception as e:
    log(f"\n[FATAL] {e}\n{traceback.format_exc()}")

log("\nFinal consolidation…")
consolidate("final")

final = pd.read_parquet(PARQUET_OUT)
log(f"\n=== New-Star Collection Summary ===")
log(f"  Total records: {len(final):,}")
log(f"  Total stars:   {final['tic_id'].nunique():,}")
log(f"  New stars:     {final['tic_id'].nunique() - len(existing_tics):,}")
log(f"  Sectors:       {int(final['sector'].min())}–{int(final['sector'].max())}")
nsec_dist = final.groupby("tic_id")["sector"].nunique()
for t in [10, 15, 20, 30, 50]:
    n = (nsec_dist >= t).sum()
    log(f"  >= {t:2d} sectors: {n:,} ({n/len(nsec_dist)*100:.1f}%)")
log(f"\nOutput → {PARQUET_OUT}")
