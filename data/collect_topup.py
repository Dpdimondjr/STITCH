"""
STITCH training data top-up pass.

For every TIC already in training_data.parquet, checks whether MAST has more
SPOC 2-min sectors than are recorded.  If so:
  - Carries forward existing sector_median values (no re-download for old sectors)
  - Downloads ONLY the new sectors
  - Recomputes leave-one-out flux_offset for ALL sectors of that star
    (necessary because ref_mean changes when new sectors are added)
  - Writes an updated record set to TOPUP_CACHE_DIR

Also collects any TICs from the TARS catalog that are NOT yet in the parquet
(same logic as the original script, but without the MAX_SECTORS cap).

Key change from collect_training_data_v2.py:
  MAX_SECTORS cap removed entirely — we want every sector for every star.

Run with optional sharding:
    python3 data/collect_topup.py --shard 0 4
    python3 data/collect_topup.py --shard 1 4
    ...
"""

import os, sys, warnings, glob, shutil, socket, threading
socket.setdefaulttimeout(60)
import numpy as np
import pandas as pd
import lightkurve as lk
from tess_stars2px import tess_stars2px_function_entry

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

PARQUET_IN     = "training_data.parquet"
PARQUET_OUT    = "training_data_topup.parquet"  # merged output; rename to training_data.parquet when done
STARS_CSV      = "tars_quiet_tics_v2.csv"
CACHE_DIR      = "./tess_cache"
TOPUP_DIR      = "./tess_cache/star_records_topup"   # separate from original star_records_v2

N_WORKERS       = 3
CHECKPOINT_EVERY = 50
CDPP_MAX        = 2000.0
CROWDSAP_MIN    = 0.3
MIN_SECTORS     = 2   # minimum usable sectors to write any records

# --focus: only process existing TICs at the sector cap (n_sectors >= this)
# skips new TICs entirely so collection goes where the data already is.
# Use --focus 10 to prioritise stars likely to have been capped.
FOCUS_MIN_SECTORS = None
if "--focus" in sys.argv:
    p = sys.argv.index("--focus")
    FOCUS_MIN_SECTORS = int(sys.argv[p + 1])

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TOPUP_DIR, exist_ok=True)

# ── Shard support ─────────────────────────────────────────────────────────────

_shard_idx, _shard_n = 0, 1
if "--shard" in sys.argv:
    p = sys.argv.index("--shard")
    _shard_idx, _shard_n = int(sys.argv[p + 1]), int(sys.argv[p + 2])


LOG_PATH = f"/tmp/stitch_topup_s{_shard_idx}.log"
_log_fh = open(LOG_PATH, "a", buffering=1)   # line-buffered

def log(msg):
    try:
        _log_fh.write(msg + "\n")
        _log_fh.flush()
    except Exception:
        pass


# ── Load existing parquet: build {tic_id → {sector → sector_median}} ──────────

log(f"Loading existing parquet: {PARQUET_IN}")
existing = pd.read_parquet(PARQUET_IN)
log(f"  {len(existing):,} records  ·  {existing['tic_id'].nunique():,} TICs")

# Per-TIC: which sectors we already have + their raw medians
parquet_secs = (existing.groupby("tic_id")
                .apply(lambda g: dict(zip(g["sector"].astype(int), g["sector_median"].astype(float))))
                .to_dict())
parquet_n = {tic: len(secs) for tic, secs in parquet_secs.items()}

# ── Build TIC work list ───────────────────────────────────────────────────────

log(f"\nBuilding work list...")

# 1. Existing TICs — sorted by descending sector count so capped stars come first.
#    With --focus N, only include stars with >= N sectors (almost certainly capped).
existing_tics_sorted = sorted(parquet_secs.keys(), key=lambda t: -parquet_n.get(t, 0))
if FOCUS_MIN_SECTORS is not None:
    topup_tics = [t for t in existing_tics_sorted if parquet_n.get(t, 0) >= FOCUS_MIN_SECTORS]
    log(f"  Focus mode: {len(topup_tics):,} existing TICs with >= {FOCUS_MIN_SECTORS} sectors")
    new_tics = []   # skip new-TIC collection in focus mode
else:
    topup_tics = existing_tics_sorted
    # 2. TICs from TARS catalog not yet in parquet (appended after existing)
    new_tics = []
    if os.path.exists(STARS_CSV):
        tars = pd.read_csv(STARS_CSV)
        parquet_tic_set = set(parquet_secs.keys())
        new_tics = [t for t in tars["tic_id"].astype(int).tolist() if t not in parquet_tic_set]
        log(f"  {len(new_tics):,} new TICs from TARS not yet in parquet")

all_tics = topup_tics + new_tics
log(f"  {len(topup_tics):,} existing TICs to check for new sectors")
log(f"  {len(all_tics):,} total TICs to process")

# ── Resume: skip TICs already handled by this top-up script ──────────────────

done_tics = {
    int(f.replace("tic_", "").replace(".csv", "").replace(".nodata", ""))
    for f in os.listdir(TOPUP_DIR)
    if f.startswith("tic_") and (f.endswith(".csv") or f.endswith(".nodata"))
}
if done_tics:
    log(f"  {len(done_tics):,} TICs already processed by top-up — skipping")
all_tics = [t for t in all_tics if t not in done_tics]
all_tics = all_tics[_shard_idx::_shard_n]
log(f"  {len(all_tics):,} TICs to process (shard {_shard_idx}/{_shard_n})\n")


# ── Per-star processing ───────────────────────────────────────────────────────

def process_star(tic_id):
    out_csv  = os.path.join(TOPUP_DIR, f"tic_{tic_id}.csv")
    out_nod  = os.path.join(TOPUP_DIR, f"tic_{tic_id}.nodata")

    def mark_nodata():
        open(out_nod, "w").close()

    # ── Step 1: search MAST for available sectors ─────────────────────────────
    try:
        sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
        try:
            sr = sr[sr.exptime.value >= 100]
        except Exception:
            pass
        if len(sr) == 0:
            mark_nodata()
            return f"TIC {tic_id}: no SPOC LCs on MAST"
    except Exception as e:
        return f"TIC {tic_id}: MAST search error — {e}"

    # Parse sector numbers from search results.
    # r.mission is a numpy array like array(['TESS Sector 11'], dtype=object),
    # so we index [0] before calling .split().
    def _sector_num(r):
        try:
            m = str(r.mission[0])
            parts = m.split()
            return int(parts[-1]) if parts[-1].isdigit() else None
        except Exception:
            return None

    mast_sectors = set(s for r in sr if (s := _sector_num(r)) is not None)

    # ── Step 2: decide whether to process ────────────────────────────────────
    existing_meds = parquet_secs.get(int(tic_id), {})   # {sector_int: median_float}
    existing_sec_set = set(existing_meds.keys())
    new_mast_sectors = mast_sectors - existing_sec_set

    is_new_tic = int(tic_id) not in parquet_secs

    if not is_new_tic and not new_mast_sectors:
        # Parquet already has everything MAST offers — nothing to do
        open(out_nod, "w").close()   # mark as checked so we skip next resume
        return f"TIC {tic_id}: up-to-date ({len(existing_sec_set)} sectors, none new)"

    # ── Step 3: download ONLY new sectors ────────────────────────────────────
    # For new TICs download everything; for existing TICs only new sectors
    if is_new_tic:
        sr_to_download = sr   # download all
    else:
        # Filter search results to only new sectors
        try:
            sr_to_download = sr[[
                i for i, r in enumerate(sr)
                if _sector_num(r) in new_mast_sectors
            ]]
        except Exception:
            sr_to_download = sr   # fallback: download all

    new_lcs = []
    for idx in range(len(sr_to_download)):
        try:
            lc = sr_to_download[idx].download(download_dir=CACHE_DIR)
            if lc is not None:
                new_lcs.append(lc)
        except Exception:
            pass

    if not new_lcs and is_new_tic:
        mark_nodata()
        return f"TIC {tic_id}: download failed"

    # ── Step 4: get RA/DEC + pixel positions ─────────────────────────────────
    ra = dec = None
    if new_lcs:
        ra  = new_lcs[0].meta.get("RA_OBJ")
        dec = new_lcs[0].meta.get("DEC_OBJ")
    if (ra is None or dec is None) and not is_new_tic:
        # Fall back to existing parquet values
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

    # ── Step 5: compute medians for new sectors ───────────────────────────────
    new_sec_meds = {}   # {sector_int: float}
    new_sec_meta = {}   # {sector_int: metadata dict}

    for lc in new_lcs:
        try:
            sec = lc.meta.get("SECTOR")
            if sec is None:
                continue
            sec = int(sec)
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

            if cdpp1_0  is not None and cdpp1_0  > CDPP_MAX:
                continue
            if crowdsap is not None and crowdsap < CROWDSAP_MIN:
                continue

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

            new_sec_meds[sec] = med
            new_sec_meta[sec] = {
                "tic_id":        tic_id,
                "sector":        sec,
                "cam":           cam_tp,
                "ccd":           ccd_tp,
                "col":           col,
                "row":           row,
                "delta_sub_col": (col + pc1_med) % 1.0 if (np.isfinite(col) and np.isfinite(pc1_med)) else np.nan,
                "delta_sub_row": (row + pc2_med) % 1.0 if (np.isfinite(row) and np.isfinite(pc2_med)) else np.nan,
                "tmag":          tmag,
                "crowdsap":      crowdsap,
                "cdpp1_0":       cdpp1_0,
                "pdcvar":        pdcvar,
                "jitter_rms":    jitter_rms,
                "sector_median": med,
                "ra":            ra,
                "dec":           dec,
            }
        except Exception:
            continue

    # ── Step 6: combine old + new sector medians; recompute LOO ──────────────
    all_meds = {**existing_meds, **new_sec_meds}   # new values override old on conflict

    total_sectors = len(all_meds)
    if total_sectors < MIN_SECTORS:
        mark_nodata()
        return f"TIC {tic_id}: only {total_sectors} usable sectors total"

    # Build records for ALL sectors (old metadata from parquet + new from download)
    records = []

    # Old sectors: carry forward metadata, update flux_offset + n_sectors_total
    if not is_new_tic:
        old_rows = existing[existing["tic_id"] == tic_id].copy()
        for _, row_data in old_rows.iterrows():
            sec = int(row_data["sector"])
            if sec not in all_meds:
                continue
            ref_vals = [v for s, v in all_meds.items() if s != sec]
            if not ref_vals:
                continue
            ref_mean    = float(np.mean(ref_vals))
            flux_offset = float(all_meds[sec]) / ref_mean
            rec = row_data.to_dict()
            rec["flux_offset"]     = flux_offset
            rec["ref_mean"]        = ref_mean
            rec["n_sectors_total"] = total_sectors
            records.append(rec)

    # New sectors: use freshly computed metadata
    for sec, meta in new_sec_meta.items():
        ref_vals = [v for s, v in all_meds.items() if s != sec]
        if not ref_vals:
            continue
        ref_mean    = float(np.mean(ref_vals))
        flux_offset = float(all_meds[sec]) / ref_mean
        meta["flux_offset"]     = flux_offset
        meta["ref_mean"]        = ref_mean
        meta["n_sectors_total"] = total_sectors
        records.append(meta)

    if not records:
        mark_nodata()
        return f"TIC {tic_id}: no valid records assembled"

    pd.DataFrame(records).to_csv(out_csv, index=False)

    # Clean downloaded FITS
    for d in glob.glob(os.path.join(CACHE_DIR, "mastDownload", "TESS",
                                    f"*-{int(tic_id):016d}-*")):
        try:
            shutil.rmtree(d)
        except Exception:
            pass

    n_new = len(new_sec_meds)
    n_carried = len(existing_meds)
    return (f"TIC {tic_id}: {total_sectors} sectors total "
            f"({n_carried} carried, {n_new} new)  →  {len(records)} records")


# ── Consolidation ─────────────────────────────────────────────────────────────

def consolidate(label=""):
    # Load top-up records (these have updated LOO for ALL their sectors)
    topup_frames = []
    for f in os.listdir(TOPUP_DIR):
        if f.startswith("tic_") and f.endswith(".csv"):
            try:
                topup_frames.append(pd.read_csv(os.path.join(TOPUP_DIR, f)))
            except Exception:
                pass

    # Load original records
    orig = pd.read_parquet(PARQUET_IN) if os.path.exists(PARQUET_IN) else pd.DataFrame()

    if topup_frames:
        topup = pd.concat(topup_frames, ignore_index=True)
        # Top-up records win: drop original rows for any TIC that has top-up data
        topup_tic_set = set(topup["tic_id"].unique())
        orig_keep = orig[~orig["tic_id"].isin(topup_tic_set)]
        final = pd.concat([orig_keep, topup], ignore_index=True)
    else:
        final = orig

    final = (final.drop_duplicates(subset=["tic_id", "sector"])
                  .sort_values(["tic_id", "sector"])
                  .reset_index(drop=True))
    final.to_parquet(PARQUET_OUT, index=False)
    tag = f" [{label}]" if label else ""
    log(f"  ✓ Checkpoint{tag}: {len(final):,} records "
        f"from {final['tic_id'].nunique():,} TICs → {PARQUET_OUT}")


# ── Main loop ─────────────────────────────────────────────────────────────────

import queue as _queue
from concurrent.futures import ThreadPoolExecutor as _TPE

log(f"Starting top-up collection ({N_WORKERS} worker, no sector cap)...\n")

_result_q = _queue.Queue()
MAX_PENDING = max(N_WORKERS * 2, 2)

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

        for tic in tic_iter:
            pool.submit(_worker, tic)
            in_flight += 1
            if in_flight >= MAX_PENDING:
                break

        while in_flight > 0:
            try:
                msg = _result_q.get(timeout=120)
            except _queue.Empty:
                log("  [WARN] 120s no result — workers may be hung")
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
    log(f"\n[FATAL] {e}")
    log(traceback.format_exc())

# ── Final consolidation ───────────────────────────────────────────────────────

log("\nFinal consolidation...")
consolidate("final")

final = pd.read_parquet(PARQUET_OUT)
log(f"\n=== Top-Up Summary ===")
log(f"  Records:       {len(final):,}")
log(f"  Unique TICs:   {final['tic_id'].nunique():,}")
log(f"  Sector range:  {int(final['sector'].min())}–{int(final['sector'].max())}")

n_dist = final.groupby("tic_id")["sector"].nunique()
for thresh in [2, 3, 4, 5, 8, 12, 20]:
    n = (n_dist >= thresh).sum()
    log(f"  TICs with >={thresh:2d} sectors: {n:,} ({n/len(n_dist)*100:.1f}%)")

log(f"\nOutput → {PARQUET_OUT}")
log("Rename to training_data.parquet when satisfied, then retrain.")
