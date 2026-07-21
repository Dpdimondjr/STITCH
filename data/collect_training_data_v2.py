"""
STITCH training data collector — v2, TARS-based.

Key differences from v1:
  - Input: tars_quiet_tics.csv (from build_tars_catalog.py) instead of training_stars.csv.
  - No training_pairs.csv: col/row computed on-the-fly via tess_stars2px using
    RA_OBJ/DEC_OBJ from the SPOC FITS header.
  - No PDCVAR filter: TARS systematic_score > 0.99 already pre-screened for variability.
  - Appends to existing training_data.parquet rather than overwriting.
  - Higher quota per cam-CCD to exploit the larger TARS catalog.

Flow per star:
  1. Download all SPOC 2-min LCs (lightkurve).
  2. Get RA/DEC from first LC header.
  3. Call tess_stars2px(RA, DEC) to get sector → (cam, ccd, col, row) mapping.
  4. Compute leave-one-out flux_offset per sector.
  5. Write per-star CSV to STAR_CACHE_DIR.
  6. Consolidate CSVs → training_data.parquet (appended, deduplicated).
"""

import os, sys, warnings, threading, glob, shutil, socket
socket.setdefaulttimeout(45)
import numpy as np
import pandas as pd
import lightkurve as lk
from tess_stars2px import tess_stars2px_function_entry

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

STARS_CSV      = "tars_quiet_tics_v2.csv"
OUT_FILE       = "training_data.parquet"
CACHE_DIR      = "./tess_cache"
STAR_CACHE_DIR = "./tess_cache/star_records_v2"  # separate from v1 cache

N_PER_CAM_CCD = 5000   # stars per camera-CCD combo (16 combos → up to 80000 total)
N_WORKERS     = 1      # 1 worker per process; run multiple sharded processes for parallelism

CDPP_MAX      = 2000.0   # keep loose — TARS already filtered variability
CROWDSAP_MIN  = 0.3      # loosen crowding filter slightly for more coverage
MIN_SECTORS   = 2        # star must have >=2 usable sectors to contribute
MAX_SECTORS   = 12       # cap downloads per star — LOO is stable at ~10 sectors

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STAR_CACHE_DIR, exist_ok=True)

# ── Shard support: --shard I N  processes every Nth star starting at index I ──
# Run 4 independent instances: --shard 0 4, --shard 1 4, --shard 2 4, --shard 3 4
_shard_idx, _shard_n = 0, 1
if "--shard" in sys.argv:
    _p = sys.argv.index("--shard")
    _shard_idx, _shard_n = int(sys.argv[_p + 1]), int(sys.argv[_p + 2])


# ── Load TARS quiet-star catalog ──────────────────────────────────────────────

def log(msg):
    try:
        print(msg, flush=True)
    except Exception:
        pass


log("Loading TARS quiet-star catalog...")
if not os.path.exists(STARS_CSV):
    log(f"ERROR: {STARS_CSV} not found. Run build_tars_catalog.py first.")
    sys.exit(1)

stars = pd.read_csv(STARS_CSV)

# Expect columns: tic_id, cam, ccd, mean_sys_score, n_quiet_sectors
log(f"  Loaded {len(stars):,} TICs from {STARS_CSV}")
log(f"  Columns: {list(stars.columns)}")

# Select top N_PER_CAM_CCD per cam-CCD.
# Primary sort: n_quiet_sectors desc — more sectors → more records per download → denser training data.
# Secondary sort: mean_sys_score desc — tiebreak on quietness.
sort_cols = [c for c in ["n_quiet_sectors", "mean_sys_score"] if c in stars.columns]
top_tics_df = (
    stars
    .sort_values(sort_cols, ascending=False)
    .groupby(["cam", "ccd"], group_keys=False)
    .head(N_PER_CAM_CCD)
    .drop_duplicates(subset="tic_id")
    .reset_index(drop=True)
)
top_tics = top_tics_df["tic_id"].tolist()
log(f"  After cap ({N_PER_CAM_CCD}/cam-ccd): {len(top_tics):,} stars selected")
for (cam, ccd), g in top_tics_df.groupby(["cam", "ccd"]):
    log(f"    Cam{int(cam)}/CCD{int(ccd)}: {len(g):4d} stars")


# ── Resume: skip already-processed stars (success or confirmed no-data) ───────

done_tics = {
    int(f.replace("tic_", "").replace(".csv", "").replace(".nodata", ""))
    for f in os.listdir(STAR_CACHE_DIR)
    if f.startswith("tic_") and (f.endswith(".csv") or f.endswith(".nodata"))
}
if done_tics:
    log(f"\n  {len(done_tics)} stars already processed — skipping")
top_tics = [t for t in top_tics if t not in done_tics]
# Apply shard slice AFTER resume filter so each shard covers distinct stars
top_tics = top_tics[_shard_idx::_shard_n]
log(f"  {len(top_tics)} stars to collect (shard {_shard_idx}/{_shard_n})\n")


# ── Per-star processing ───────────────────────────────────────────────────────

def process_star(tic_id):
    star_csv = os.path.join(STAR_CACHE_DIR, f"tic_{tic_id}.csv")

    try:
        sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
        if len(sr) == 0:
            open(os.path.join(STAR_CACHE_DIR, f"tic_{tic_id}.nodata"), "w").close()
            return f"TIC {tic_id}: no SPOC light curves"
        try:
            sr = sr[sr.exptime.value >= 100]
        except Exception:
            pass
        if len(sr) == 0:
            open(os.path.join(STAR_CACHE_DIR, f"tic_{tic_id}.nodata"), "w").close()
            return f"TIC {tic_id}: no TESS-SPOC LCs found"
        if len(sr) > MAX_SECTORS:
            sr = sr[-MAX_SECTORS:]
    except Exception as e:
        msg = str(e).lower()
        if "no data" in msg or "not found" in msg or "nodataerror" in msg:
            open(os.path.join(STAR_CACHE_DIR, f"tic_{tic_id}.nodata"), "w").close()
        return f"TIC {tic_id}: search error — {e}"

    # Phase 2: downloads — no semaphore, runs concurrently with other searches
    lc_list = []
    for idx in range(len(sr)):
        try:
            lc = sr[idx].download(download_dir=CACHE_DIR)
            if lc is not None:
                lc_list.append(lc)
        except Exception:
            pass

    _nd = lambda: open(os.path.join(STAR_CACHE_DIR, f"tic_{tic_id}.nodata"), "w").close()

    if not lc_list:
        _nd(); return f"TIC {tic_id}: no LCs downloaded"

    # Get RA/DEC from first LC header
    ra  = lc_list[0].meta.get("RA_OBJ")
    dec = lc_list[0].meta.get("DEC_OBJ")
    if ra is None or dec is None:
        _nd(); return f"TIC {tic_id}: no RA/DEC in header"

    # Compute pixel positions for all sectors using tess_stars2px
    try:
        outID, outRa, outDec, outSec, outCam, outCcd, outCol, outRow, _ = \
            tess_stars2px_function_entry(int(tic_id), float(ra), float(dec))
        # Key by sector only — a star can only be on one camera per sector
        pos_lookup = {}
        for sec, cam, ccd, col, row in zip(outSec, outCam, outCcd, outCol, outRow):
            pos_lookup[int(sec)] = (int(cam), int(ccd), float(col), float(row))
    except Exception as e:
        _nd(); return f"TIC {tic_id}: tess_stars2px error — {e}"

    # Compute per-sector medians
    sector_meds = {}
    for lc in lc_list:
        try:
            sec = lc.meta.get("SECTOR")
            if sec is None or not (1 <= int(sec) <= 200):
                continue
            flux = lc.flux.value
            med = np.nanmedian(flux)
            if np.isfinite(med) and med > 0:
                sector_meds[sec] = med
        except Exception:
            continue

    if len(sector_meds) < MIN_SECTORS:
        _nd(); return f"TIC {tic_id}: <{MIN_SECTORS} usable sectors ({len(sector_meds)} found)"

    records = []
    for lc in lc_list:
        try:
            sec = lc.meta.get("SECTOR")
            cam = lc.meta.get("CAMERA")
            ccd = lc.meta.get("CCD")

            if sec is None or not (1 <= int(sec) <= 200) or sec not in sector_meds:
                continue

            ref_vals = [v for s, v in sector_meds.items() if s != sec]
            if not ref_vals:
                continue
            this_med    = float(sector_meds[sec])
            ref_mean    = float(np.mean(ref_vals))
            flux_offset = this_med / ref_mean   # leave-one-out offset

            def _flt(val):
                try: return float(val)
                except (TypeError, ValueError): return None

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
                pc1_med, pc2_med = float(np.nanmedian(pc1)), float(np.nanmedian(pc2))
            except Exception:
                jitter_rms = pc1_med = pc2_med = np.nan

            pos = pos_lookup.get(int(sec))
            if pos is None:
                cam_tp = cam; ccd_tp = ccd; col = row = np.nan
            else:
                cam_tp, ccd_tp, col, row = pos
                # Use tess-point cam/ccd as authoritative if SPOC header is missing
                if cam is None: cam = cam_tp
                if ccd is None: ccd = ccd_tp

            records.append({
                "tic_id":           tic_id,
                "sector":           sec,
                "cam":              cam,
                "ccd":              ccd,
                "col":              col,
                "row":              row,
                "delta_sub_col":    (col + pc1_med) % 1.0 if (np.isfinite(col) and np.isfinite(pc1_med)) else np.nan,
                "delta_sub_row":    (row + pc2_med) % 1.0 if (np.isfinite(row) and np.isfinite(pc2_med)) else np.nan,
                "tmag":             tmag,
                "crowdsap":         crowdsap,
                "cdpp1_0":          cdpp1_0,
                "pdcvar":           pdcvar,
                "jitter_rms":       jitter_rms,
                "flux_offset":      flux_offset,    # leave-one-out: this_med / mean(other meds)
                "sector_median":    this_med,       # raw median flux (e-/s); recompute any normalisation
                "ref_mean":         ref_mean,       # mean of other-sector medians used for LOO
                "n_sectors_total":  len(sector_meds),
                "ra":               ra,
                "dec":              dec,
            })
        except Exception:
            continue

    if records:
        pd.DataFrame(records).to_csv(star_csv, index=False)

    # Delete downloaded FITS files for this TIC to keep disk usage bounded.
    # Pattern: mastDownload/TESS/tess*-{tic:016d}-*
    for d in glob.glob(os.path.join(CACHE_DIR, "mastDownload", "TESS",
                                    f"*-{int(tic_id):016d}-*")):
        try:
            shutil.rmtree(d)
        except Exception:
            pass

    return f"TIC {tic_id}: {len(records)} records from {len(sector_meds)} sectors"


# ── Checkpoint: consolidate per-star CSVs → parquet ─────────────────────────
# Called periodically during collection so a crash doesn't lose all progress.
# Per-star CSVs are the durable checkpoint; parquet is a convenience merge.

CHECKPOINT_EVERY = 50   # write parquet after every N completed stars
_checkpoint_lock = threading.Lock()

ALL_CACHE_DIRS = [
    os.path.join(CACHE_DIR, "star_records"),   # v1
    STAR_CACHE_DIR,                             # v2
]

def consolidate(label: str = "") -> None:
    frames = []
    for d in ALL_CACHE_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("tic_") and f.endswith(".csv"):
                try:
                    frames.append(pd.read_csv(os.path.join(d, f)))
                except Exception:
                    pass
    if not frames:
        return
    final = (pd.concat(frames, ignore_index=True)
               .drop_duplicates(subset=["tic_id", "sector"])
               .sort_values(["tic_id", "sector"])
               .reset_index(drop=True))
    final.to_parquet(OUT_FILE, index=False)
    tag = f" [{label}]" if label else ""
    log(f"  ✓ Checkpoint{tag}: {len(final):,} records from {final['tic_id'].nunique():,} stars → {OUT_FILE}")


# ── Main collection loop ──────────────────────────────────────────────────────

log(f"Collecting ({N_WORKERS} workers, SPOC 2-min, TARS pre-filtered)...\n")
log(f"  Checkpointing every {CHECKPOINT_EVERY} stars. Resume-safe: already-collected")
log(f"  stars are skipped based on files in {STAR_CACHE_DIR}\n")

import traceback as _tb
import queue as _queue
from concurrent.futures import ThreadPoolExecutor as _TPE

# Use a plain queue.Queue to collect results — bypasses concurrent.futures
# internals (as_completed / wait) that deadlock on Python 3.9/macOS.
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
        tic_iter  = iter(top_tics)
        in_flight = 0

        # Seed initial batch
        for tic in tic_iter:
            pool.submit(_worker, tic)
            in_flight += 1
            if in_flight >= MAX_PENDING:
                break

        while in_flight > 0:
            try:
                msg = _result_q.get(timeout=90)
            except _queue.Empty:
                log(f"  [WARN] 90s with no result — workers may be hung, continuing wait")
                continue
            in_flight -= 1
            n_done += 1
            log(f"  [{n_done:4d}/{len(top_tics)}] {msg}")
            if n_done % CHECKPOINT_EVERY == 0:
                consolidate(f"{n_done}/{len(top_tics)} stars")
            # Refill
            try:
                pool.submit(_worker, next(tic_iter))
                in_flight += 1
            except StopIteration:
                pass

except Exception as _main_e:
    log(f"\n[FATAL] Main loop crashed: {_main_e}")
    log(_tb.format_exc())


# ── Final consolidation ───────────────────────────────────────────────────────

log("\nFinal consolidation...")
with _checkpoint_lock:
    consolidate("final")

# Print summary from the saved parquet
import pyarrow.parquet as pq
final = pq.read_table(OUT_FILE).to_pandas()
log(f"\n=== Training Data Summary ===")
log(f"  Total (star, sector) records:  {len(final):,}")
log(f"  Unique stars:                  {final['tic_id'].nunique():,}")
log(f"  Sectors covered:               {final['sector'].min()}–{final['sector'].max()}")
log(f"  Cam/CCD coverage:")
for (cam, ccd), g in final.groupby(["cam", "ccd"]):
    log(f"    Cam{cam}/CCD{ccd}: {len(g):5d} records")
log(f"  flux_offset range:  {final['flux_offset'].min():.4f} – {final['flux_offset'].max():.4f}")
log(f"  flux_offset std:    {final['flux_offset'].std():.4f}")
log(f"\nSaved → {OUT_FILE}")
log("Next: re-run train_flow.py to train on the expanded dataset.")
