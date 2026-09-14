"""
Top-up data collection focused on quiet, multi-sector stars.

Strategy:
  1. For each star in the current parquet, check MAST for sectors not yet collected.
  2. Prioritize stars by their estimated oracle ceiling (loo_std / sqrt(N-1)):
       - quietest stars (low loo_std) get collected first
       - stars with many sectors already get new ones first
  3. Skip stars flagged as non-quiet (pdcvar > PDCVAR_MAX) unless they have many sectors.

This is adapted from collect_topup.py to work with the current
training_data_topup_pdc.parquet (which has flux_offset_loo as the target).

Usage:
    python3 data/collect_quiet_topup.py [parquet] [--shard 0 4] [--min-sectors N]
    python3 data/collect_quiet_topup.py training_data_topup_pdc.parquet --min-sectors 8
"""

import os, sys, warnings, glob, shutil, socket, threading
socket.setdefaulttimeout(60)
import numpy as np
import pandas as pd
import lightkurve as lk
from tess_stars2px import tess_stars2px_function_entry

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_IN     = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                      "training_data_topup_pdc.parquet")
PARQUET_OUT    = PARQUET_IN.replace(".parquet", "_v2.parquet")

CACHE_DIR      = "./tess_cache"
TOPUP_DIR      = "./tess_cache/star_records_quiet_topup"

N_WORKERS       = 3
CHECKPOINT_EVERY = 50
CDPP_MAX        = 2000.0
CROWDSAP_MIN    = 0.3
MIN_SECTORS     = 2

# Minimum sectors already observed to bother collecting new ones.
# Stars with fewer existing sectors are skipped (their labels are noisiest).
MIN_EXISTING = 8
if "--min-sectors" in sys.argv:
    p = sys.argv.index("--min-sectors")
    MIN_EXISTING = int(sys.argv[p + 1])
print(f"Collecting new sectors for stars with >= {MIN_EXISTING} existing sectors")

_shard_idx, _shard_n = 0, 1
if "--shard" in sys.argv:
    p = sys.argv.index("--shard")
    _shard_idx, _shard_n = int(sys.argv[p + 1]), int(sys.argv[p + 2])

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TOPUP_DIR, exist_ok=True)

LOG_PATH = f"/tmp/stitch_quiet_topup_s{_shard_idx}.log"
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

# ── Load parquet ──────────────────────────────────────────────────────────────
log(f"Loading {PARQUET_IN} …")
existing = pd.read_parquet(PARQUET_IN)

# Detect which column is the LOO target
target_col = "flux_offset_loo" if "flux_offset_loo" in existing.columns else "flux_offset"
log(f"  LOO target column: {target_col}")
log(f"  {len(existing):,} records  ·  {existing['tic_id'].nunique():,} TICs")

# Remove obviously bad sectors (sector=1751 etc.)
existing = existing[existing["sector"] <= 200]

# Per-star stats for prioritization
star_stats = (existing
    .dropna(subset=[target_col])
    .query(f"{target_col} > 0.85 and {target_col} < 1.15")
    .groupby("tic_id")
    .agg(
        n_obs     = ("sector",   "nunique"),
        loo_std   = (target_col, "std"),
        max_sec   = ("sector",   "max"),
        ra        = ("ra",       "first"),
        dec       = ("dec",      "first"),
    )
    .reset_index()
)
star_stats["loo_std"] = star_stats["loo_std"].fillna(0.02)
star_stats["oracle_ceil"] = (
    star_stats["loo_std"] / np.sqrt((star_stats["n_obs"] - 1).clip(lower=1))
)

# Filter to stars that qualify for top-up
qualify = star_stats[star_stats["n_obs"] >= MIN_EXISTING].copy()
# Sort: quietest (lowest oracle_ceil) first, then most sectors
qualify = qualify.sort_values(["oracle_ceil", "n_obs"], ascending=[True, False])
log(f"\n  Stars qualifying for top-up (>= {MIN_EXISTING} sectors): {len(qualify):,}")
log(f"  Oracle ceiling distribution:")
for p in [10, 25, 50, 75, 90]:
    log(f"    p{p}: {np.percentile(qualify['oracle_ceil'], p)*100:.4f}%")

# Per-star sector cache
parquet_secs = (existing
    .groupby("tic_id")
    .apply(lambda g: dict(zip(g["sector"].astype(int), g["sector_median"].astype(float))))
    .to_dict()
)

all_tics = qualify["tic_id"].tolist()

# Resume: skip already-processed TICs
done_tics = {
    int(f.replace("tic_", "").replace(".csv", "").replace(".nodata", ""))
    for f in os.listdir(TOPUP_DIR)
    if f.startswith("tic_") and (f.endswith(".csv") or f.endswith(".nodata"))
}
if done_tics:
    log(f"\n  {len(done_tics):,} TICs already processed — skipping")
all_tics = [t for t in all_tics if t not in done_tics]
all_tics = all_tics[_shard_idx::_shard_n]
log(f"  {len(all_tics):,} TICs to process (shard {_shard_idx}/{_shard_n})\n")

# ── Per-star processing ───────────────────────────────────────────────────────
def process_star(tic_id):
    out_csv = os.path.join(TOPUP_DIR, f"tic_{tic_id}.csv")
    out_nod = os.path.join(TOPUP_DIR, f"tic_{tic_id}.nodata")

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

    mast_sectors = set(s for r in sr if (s := _sector_num(r)) is not None)
    existing_meds = parquet_secs.get(int(tic_id), {})
    existing_sec_set = set(existing_meds.keys())
    new_mast_sectors = mast_sectors - existing_sec_set

    if not new_mast_sectors:
        open(out_nod, "w").close()
        return f"TIC {tic_id}: up-to-date ({len(existing_sec_set)} sectors)"

    # Download only new sectors
    try:
        sr_new = sr[[i for i, r in enumerate(sr) if _sector_num(r) in new_mast_sectors]]
    except Exception:
        sr_new = sr

    new_lcs = []
    for idx in range(len(sr_new)):
        try:
            lc = sr_new[idx].download(download_dir=CACHE_DIR)
            if lc is not None:
                new_lcs.append(lc)
        except Exception:
            pass

    if not new_lcs:
        mark_nodata()
        return f"TIC {tic_id}: download failed"

    # RA/DEC
    ra = dec = None
    ra  = new_lcs[0].meta.get("RA_OBJ")
    dec = new_lcs[0].meta.get("DEC_OBJ")
    if ra is None or dec is None:
        ex_rows = existing[existing["tic_id"] == tic_id]
        if len(ex_rows):
            ra  = float(ex_rows.iloc[0]["ra"])
            dec = float(ex_rows.iloc[0]["dec"])
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

    # Extract new sector medians
    new_sec_meds = {}
    new_sec_meta = {}

    for lc in new_lcs:
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

            new_sec_meds[sec] = med
            new_sec_meta[sec] = {
                "tic_id":             tic_id,
                "sector":             sec,
                "cam":                cam_tp,
                "ccd":                ccd_tp,
                "col":                col,
                "row":                row,
                "delta_sub_col":      (col + pc1_med) % 1.0 if (np.isfinite(col) and np.isfinite(pc1_med)) else np.nan,
                "delta_sub_row":      (row + pc2_med) % 1.0 if (np.isfinite(row) and np.isfinite(pc2_med)) else np.nan,
                "tmag":               tmag,
                "crowdsap":           crowdsap,
                "cdpp1_0":            cdpp1_0,
                "pdcvar":             pdcvar,
                "jitter_rms":         jitter_rms,
                "sector_median":      med,
                "ra":                 ra,
                "dec":                dec,
                "median_sap_bkg":     median_sap_bkg,
                "p90_sap_bkg":        p90_sap_bkg,
                "bkg_rms":            bkg_rms,
                "scatter_flag_frac":  scatter_flag_frac,
                "flfrcsap":           _flt(lc.meta.get("FLFRCSAP")),
                "teff":               _flt(lc.meta.get("TEFF")),
                "pdc_noi":            _flt(lc.meta.get("PDC_NOI")),
                "pdc_corp":           _flt(lc.meta.get("PDC_CORP")),
                "pdc_totp":           _flt(lc.meta.get("PDC_TOTP")),
                "pr_wght2":           _flt(lc.meta.get("PR_WGHT2")),
            }
        except Exception:
            continue

    # Combine old + new sector medians; recompute LOO
    all_meds = {**existing_meds, **new_sec_meds}
    total_sectors = len(all_meds)
    if total_sectors < MIN_SECTORS:
        mark_nodata()
        return f"TIC {tic_id}: only {total_sectors} sectors"

    records = []

    # Old sectors: carry forward metadata, update LOO + n_sectors_total
    old_rows = existing[existing["tic_id"] == tic_id].copy()
    for _, row_data in old_rows.iterrows():
        sec = int(row_data["sector"])
        if sec not in all_meds: continue
        ref_vals    = [v for s, v in all_meds.items() if s != sec]
        if not ref_vals: continue
        ref_mean    = float(np.mean(ref_vals))
        rec = row_data.to_dict()
        rec["flux_offset"]     = float(all_meds[sec]) / ref_mean
        rec["flux_offset_loo"] = float(all_meds[sec]) / ref_mean
        rec["ref_mean"]        = ref_mean
        rec["n_sectors_total"] = total_sectors
        records.append(rec)

    # New sectors
    for sec, meta in new_sec_meta.items():
        ref_vals = [v for s, v in all_meds.items() if s != sec]
        if not ref_vals: continue
        ref_mean    = float(np.mean(ref_vals))
        meta["flux_offset"]     = float(all_meds[sec]) / ref_mean
        meta["flux_offset_loo"] = float(all_meds[sec]) / ref_mean
        meta["ref_mean"]        = ref_mean
        meta["n_sectors_total"] = total_sectors
        records.append(meta)

    if not records:
        mark_nodata()
        return f"TIC {tic_id}: no valid records"

    pd.DataFrame(records).to_csv(out_csv, index=False)

    for d in glob.glob(os.path.join(CACHE_DIR, "mastDownload", "TESS",
                                    f"*-{int(tic_id):016d}-*")):
        try: shutil.rmtree(d)
        except Exception: pass

    return (f"TIC {tic_id}: {total_sectors} total "
            f"({len(existing_meds)} carried, {len(new_sec_meds)} new)")


# ── Consolidation ─────────────────────────────────────────────────────────────
def consolidate(label=""):
    frames = []
    for f in os.listdir(TOPUP_DIR):
        if f.startswith("tic_") and f.endswith(".csv"):
            try: frames.append(pd.read_csv(os.path.join(TOPUP_DIR, f)))
            except Exception: pass

    orig = existing.copy()
    if frames:
        topup = pd.concat(frames, ignore_index=True)
        topup_tics = set(topup["tic_id"].unique())
        orig_keep  = orig[~orig["tic_id"].isin(topup_tics)]
        final = pd.concat([orig_keep, topup], ignore_index=True)
    else:
        final = orig

    final = (final.drop_duplicates(subset=["tic_id", "sector"])
                  .sort_values(["tic_id", "sector"])
                  .reset_index(drop=True))
    final.to_parquet(PARQUET_OUT, index=False)
    tag = f" [{label}]" if label else ""
    log(f"  ✓ Checkpoint{tag}: {len(final):,} records from {final['tic_id'].nunique():,} TICs → {PARQUET_OUT}")


# ── Main loop ─────────────────────────────────────────────────────────────────
import queue as _queue
from concurrent.futures import ThreadPoolExecutor as _TPE

log(f"Starting quiet top-up collection ({N_WORKERS} workers)…\n")

_result_q = _queue.Queue()

def _worker(tic_id):
    try:
        _result_q.put(process_star(tic_id))
    except Exception as e:
        _result_q.put(f"TIC {tic_id}: unhandled — {e}")

n_done = 0
try:
    with _TPE(max_workers=N_WORKERS) as pool:
        tic_iter  = iter(all_tics)
        in_flight = 0
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
    import traceback
    log(f"\n[FATAL] {e}\n{traceback.format_exc()}")

log("\nFinal consolidation…")
consolidate("final")

final = pd.read_parquet(PARQUET_OUT)
log(f"\n=== Quiet Top-Up Summary ===")
log(f"  Records: {len(final):,}")
log(f"  Stars:   {final['tic_id'].nunique():,}")
log(f"  Sectors: {int(final['sector'].min())}–{int(final['sector'].max())}")
nsec_dist = final.groupby("tic_id")["sector"].nunique()
for t in [8, 12, 15, 20, 25, 30]:
    n = (nsec_dist >= t).sum()
    log(f"  >= {t:2d} sectors: {n:,} ({n/len(nsec_dist)*100:.1f}%)")
log(f"\nOutput → {PARQUET_OUT}")
