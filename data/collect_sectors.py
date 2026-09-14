"""
Targeted sector collection — fetch specific sectors for all stars in the parquet.

For each TIC in the parquet that is MISSING any of the target sectors,
downloads those sector LCs from MAST and appends the records to a new parquet.

Usage:
    python3 data/collect_sectors.py --sectors 66 92
    python3 data/collect_sectors.py --sectors 92 training_data_topup.parquet
    python3 data/collect_sectors.py --sectors 66 92 --workers 4
"""

import os, sys, warnings, glob, shutil, socket, queue, traceback
socket.setdefaulttimeout(90)
import numpy as np
import pandas as pd
import lightkurve as lk
from tess_stars2px import tess_stars2px_function_entry
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

# ── Args ──────────────────────────────────────────────────────────────────────
PARQUET_IN = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                  "training_data_topup.parquet")
PARQUET_OUT = PARQUET_IN.replace(".parquet", "_sectors.parquet")

if "--sectors" not in sys.argv:
    print("Usage: python3 data/collect_sectors.py --sectors 66 92")
    sys.exit(1)

sec_start = sys.argv.index("--sectors") + 1
TARGET_SECTORS = set()
for s in sys.argv[sec_start:]:
    if s.startswith("--"): break
    try: TARGET_SECTORS.add(int(s))
    except ValueError: break

N_WORKERS = 4
if "--workers" in sys.argv:
    p = sys.argv.index("--workers")
    N_WORKERS = int(sys.argv[p + 1])

CACHE_DIR  = "./tess_cache"
OUT_DIR    = "./tess_cache/sector_targeted"
CDPP_MAX   = 2000.0
CROWDSAP_MIN = 0.3
CHECKPOINT_EVERY = 100

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = f"/tmp/stitch_collect_sectors.log"
_log_fh  = open(LOG_PATH, "a", buffering=1)

def log(msg):
    try: print(msg, flush=True)
    except: pass
    try: _log_fh.write(msg + "\n"); _log_fh.flush()
    except: pass

# ── Load parquet ──────────────────────────────────────────────────────────────
log(f"Loading {PARQUET_IN} …")
existing = pd.read_parquet(PARQUET_IN)
existing = existing[existing["sector"] <= 200]   # drop sector 1751 etc.
log(f"  {len(existing):,} records · {existing['tic_id'].nunique():,} TICs")
log(f"  Target sectors: {sorted(TARGET_SECTORS)}")

# Find which TIC×sector pairs are already present
have = set(zip(existing["tic_id"].astype(int), existing["sector"].astype(int)))

# Per-star: all current sector medians (for LOO recomputation)
parquet_secs = (
    existing
    .groupby("tic_id")
    .apply(lambda g: dict(zip(g["sector"].astype(int), g["sector_median"].astype(float))))
    .to_dict()
)

# Build work list: TICs missing at least one target sector
all_tics = existing["tic_id"].dropna().astype(int).unique().tolist()
work = []
for tic in all_tics:
    missing = TARGET_SECTORS - {s for (t, s) in have if t == tic}
    if missing:
        work.append((tic, missing))

log(f"  TICs missing ≥1 target sector: {len(work):,}\n")

# Resume
done_tics = {
    int(f.replace("tic_", "").replace(".csv", "").replace(".nodata", ""))
    for f in os.listdir(OUT_DIR)
    if f.startswith("tic_") and (f.endswith(".csv") or f.endswith(".nodata"))
}
if done_tics:
    log(f"  {len(done_tics):,} TICs already processed — skipping")
work = [(t, m) for t, m in work if t not in done_tics]
log(f"  {len(work):,} TICs remaining\n")

# ── Per-star processing ───────────────────────────────────────────────────────
def process_star(tic_id, missing_sectors):
    out_csv = os.path.join(OUT_DIR, f"tic_{tic_id}.csv")
    out_nod = os.path.join(OUT_DIR, f"tic_{tic_id}.nodata")

    def mark_nodata(reason=""):
        open(out_nod, "w").close()
        return f"TIC {tic_id}: nodata — {reason}"

    try:
        sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
        try:
            sr = sr[sr.exptime.value >= 100]
        except Exception:
            pass
        if len(sr) == 0:
            return mark_nodata("no SPOC LCs on MAST")
    except Exception as e:
        return f"TIC {tic_id}: MAST error — {e}"

    def _sector_num(r):
        try:
            m = str(r.mission[0])
            parts = m.split()
            return int(parts[-1]) if parts[-1].isdigit() else None
        except Exception:
            return None

    # Only download the missing target sectors
    mast_sec_map = {_sector_num(r): i for i, r in enumerate(sr)}
    to_download  = [i for s, i in mast_sec_map.items()
                    if s is not None and s in missing_sectors]

    if not to_download:
        return mark_nodata(f"sectors {missing_sectors} not on MAST")

    # Get RA/DEC for tess_stars2px
    ex_rows = existing[existing["tic_id"] == tic_id]
    ra  = float(ex_rows.iloc[0]["ra"])  if len(ex_rows) else None
    dec = float(ex_rows.iloc[0]["dec"]) if len(ex_rows) else None

    try:
        _, _, _, outSec, outCam, outCcd, outCol, outRow, _ = \
            tess_stars2px_function_entry(int(tic_id), float(ra), float(dec))
        pos_lookup = {int(s): (int(c1), int(c2), float(cl), float(rw))
                      for s, c1, c2, cl, rw in zip(outSec, outCam, outCcd, outCol, outRow)}
    except Exception as e:
        pos_lookup = {}

    new_sec_meds = {}
    new_sec_meta = {}

    for idx in to_download:
        try:
            lc = sr[idx].download(download_dir=CACHE_DIR)
            if lc is None: continue
            sec  = int(lc.meta.get("SECTOR"))
            flux = lc.flux.value
            med  = float(np.nanmedian(flux))
            if not (np.isfinite(med) and med > 0): continue

            def _flt(v):
                try: return float(v)
                except: return None

            cdpp1_0  = _flt(lc.meta.get("CDPP1_0"))
            crowdsap = _flt(lc.meta.get("CROWDSAP"))
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

            if ra is None:
                ra  = _flt(lc.meta.get("RA_OBJ"))
                dec = _flt(lc.meta.get("DEC_OBJ"))

            new_sec_meds[sec] = med
            new_sec_meta[sec] = {
                "tic_id": tic_id, "sector": sec,
                "cam": cam_tp, "ccd": ccd_tp, "col": col, "row": row,
                "delta_sub_col": (col + pc1_med) % 1.0 if (np.isfinite(col if col is not None else np.nan) and np.isfinite(pc1_med)) else np.nan,
                "delta_sub_row": (row + pc2_med) % 1.0 if (np.isfinite(row if row is not None else np.nan) and np.isfinite(pc2_med)) else np.nan,
                "tmag": _flt(lc.meta.get("TESSMAG")),
                "crowdsap": crowdsap, "cdpp1_0": cdpp1_0,
                "pdcvar": _flt(lc.meta.get("PDCVAR")),
                "jitter_rms": jitter_rms,
                "sector_median": med, "ra": ra, "dec": dec,
                "median_sap_bkg": median_sap_bkg, "p90_sap_bkg": p90_sap_bkg,
                "bkg_rms": bkg_rms, "scatter_flag_frac": scatter_flag_frac,
                "flfrcsap": _flt(lc.meta.get("FLFRCSAP")),
                "teff": _flt(lc.meta.get("TEFF")),
                "pdc_noi": _flt(lc.meta.get("PDC_NOI")),
                "pdc_corp": _flt(lc.meta.get("PDC_CORP")),
                "pdc_totp": _flt(lc.meta.get("PDC_TOTP")),
                "pr_wght2": _flt(lc.meta.get("PR_WGHT2")),
            }
        except Exception:
            continue

    if not new_sec_meds:
        return mark_nodata("download failed or filtered out")

    # LOO using all known medians for this star (existing + new)
    all_meds = {**parquet_secs.get(tic_id, {}), **new_sec_meds}
    total_sectors = len(all_meds)

    records = []
    for sec, meta in new_sec_meta.items():
        ref_vals = [v for s, v in all_meds.items() if s != sec]
        if not ref_vals: continue
        ref_mean = float(np.mean(ref_vals))
        meta["flux_offset"]     = float(all_meds[sec]) / ref_mean
        meta["flux_offset_loo"] = float(all_meds[sec]) / ref_mean
        meta["ref_mean"]        = ref_mean
        meta["n_sectors_total"] = total_sectors
        records.append(meta)

    if not records:
        return mark_nodata("no valid records built")

    pd.DataFrame(records).to_csv(out_csv, index=False)

    for d in glob.glob(os.path.join(CACHE_DIR, "mastDownload", "TESS",
                                    f"*-{int(tic_id):016d}-*")):
        try: shutil.rmtree(d)
        except: pass

    return f"TIC {tic_id}: added sectors {sorted(new_sec_meds.keys())}"


# ── Consolidation ─────────────────────────────────────────────────────────────
def consolidate(label=""):
    frames = []
    for f in os.listdir(OUT_DIR):
        if f.startswith("tic_") and f.endswith(".csv"):
            try: frames.append(pd.read_csv(os.path.join(OUT_DIR, f)))
            except: pass

    if not frames:
        log(f"  Nothing to consolidate yet.")
        return

    new_rows = pd.concat(frames, ignore_index=True)
    final = pd.concat([existing, new_rows], ignore_index=True)
    final = (final.drop_duplicates(subset=["tic_id", "sector"])
                  .sort_values(["tic_id", "sector"])
                  .reset_index(drop=True))
    final.to_parquet(PARQUET_OUT, index=False)
    tag = f" [{label}]" if label else ""
    log(f"  ✓ Checkpoint{tag}: {len(final):,} records · {final['tic_id'].nunique():,} TICs → {PARQUET_OUT}")
    for sec in sorted(TARGET_SECTORS):
        n = (final["sector"] == sec).sum()
        log(f"    sector {sec}: {n:,} records")


# ── Main loop ─────────────────────────────────────────────────────────────────
log(f"Starting targeted collection ({N_WORKERS} workers) …\n")

result_q = queue.Queue()

def _worker(args):
    tic_id, missing = args
    try:
        result_q.put(process_star(tic_id, missing))
    except Exception as e:
        result_q.put(f"TIC {tic_id}: unhandled — {e}\n{traceback.format_exc()}")

n_done = 0
try:
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        work_iter = iter(work)
        in_flight = 0
        MAX_PENDING = N_WORKERS * 2

        for item in work_iter:
            pool.submit(_worker, item)
            in_flight += 1
            if in_flight >= MAX_PENDING:
                break

        while in_flight > 0:
            try:
                msg = result_q.get(timeout=120)
            except queue.Empty:
                log("  [WARN] 120s no result")
                continue
            in_flight -= 1
            n_done += 1
            log(f"  [{n_done:5d}/{len(work)}] {msg}")
            if n_done % CHECKPOINT_EVERY == 0:
                consolidate(f"{n_done}/{len(work)}")
            try:
                pool.submit(_worker, next(work_iter))
                in_flight += 1
            except StopIteration:
                pass
except Exception as e:
    log(f"\n[FATAL] {e}\n{traceback.format_exc()}")

log("\nFinal consolidation …")
consolidate("final")
log("\nDone.")
