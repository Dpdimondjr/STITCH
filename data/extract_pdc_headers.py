"""
Targeted PDC header extraction — Option C.

For each star in the parquet, downloads its SPOC light curve FITS files,
reads only the headers, then deletes the files.

New features per (tic_id, sector):
  pr_wght2   — PDC prior weight       (Pearson r=+0.48 with flux_offset)
  pdc_noi    — PDC noise goodness     (Pearson r=-0.30 with flux_offset)
  pdc_corp   — PDC correlation goodness percentile
  pdc_totp   — PDC total goodness percentile
  flfrcsap   — aperture flux fraction
  teff       — stellar effective temperature (K)

Resume-safe: per-TIC results written to PDC_HEADERS_DIR/tic_{id}.csv
Run with optional sharding:
  python3 data/extract_pdc_headers.py --shard 0 4
  python3 data/extract_pdc_headers.py --shard 1 4
"""

import os, sys, glob, shutil, socket, warnings
socket.setdefaulttimeout(60)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightkurve as lk
lk.log.setLevel("ERROR")   # suppress quality-mask warnings printed via lk.log
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────────────
PARQUET_IN       = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                        "training_data_topup.parquet")
PARQUET_OUT      = PARQUET_IN.replace(".parquet", "_pdc.parquet")
CACHE_DIR        = "./tess_cache"
PDC_HEADERS_DIR  = os.path.join(CACHE_DIR, "pdc_headers")
N_WORKERS        = 3
CHECKPOINT_EVERY = 50

_shard_idx, _shard_n = 0, 1
if "--shard" in sys.argv:
    p = sys.argv.index("--shard")
    _shard_idx, _shard_n = int(sys.argv[p + 1]), int(sys.argv[p + 2])

LOG_PATH = f"/tmp/stitch_pdc_s{_shard_idx}.log"
os.makedirs(PDC_HEADERS_DIR, exist_ok=True)

_log = open(LOG_PATH, "a", buffering=1)
def log(msg):
    _log.write(msg + "\n"); _log.flush()
    print(msg, flush=True)

PDC_KEYS = ["PR_WGHT2", "PDC_NOI", "PDC_CORP", "PDC_TOTP", "FLFRCSAP"]
STAR_KEYS = ["TEFF"]   # constant per star, only need one sector

# ── Load parquet — get unique TICs and their sectors ──────────────────────────
log(f"Loading {PARQUET_IN}...")
df = pd.read_parquet(PARQUET_IN)
log(f"  {len(df):,} records, {df['tic_id'].nunique():,} unique stars")

# Map tic_id → list of sectors we need
tic_sectors = (df.groupby("tic_id")["sector"]
                 .apply(lambda x: sorted(x.astype(int).unique().tolist()))
                 .to_dict())
all_tics = sorted(tic_sectors.keys(), key=lambda t: -len(tic_sectors[t]))

# ── Resume: skip already-done TICs ────────────────────────────────────────────
done_tics = set()
for f in os.listdir(PDC_HEADERS_DIR):
    if f.startswith("tic_") and (f.endswith(".csv") or f.endswith(".nodata")):
        try:
            done_tics.add(int(float(f.replace("tic_", "").replace(".csv", "").replace(".nodata", ""))))
        except ValueError:
            pass
remaining = [t for t in all_tics if t not in done_tics]
remaining  = remaining[_shard_idx::_shard_n]
log(f"  {len(done_tics):,} already done, {len(remaining):,} to process (shard {_shard_idx}/{_shard_n})\n")

# ── Per-star extraction ────────────────────────────────────────────────────────
def _flt(v):
    try:    return float(v)
    except: return np.nan

def process_star(tic_id):
    out_csv = os.path.join(PDC_HEADERS_DIR, f"tic_{int(tic_id)}.csv")
    out_nod = os.path.join(PDC_HEADERS_DIR, f"tic_{int(tic_id)}.nodata")
    need_sectors = set(tic_sectors.get(int(tic_id), []))

    import time
    sr = None
    for attempt in range(5):
        try:
            sr = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS", author="TESS-SPOC")
            try:
                sr = sr[sr.exptime.value >= 100]
            except Exception:
                pass
            break
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                time.sleep(10 * (attempt + 1))
            else:
                return f"TIC {int(tic_id)}: search error — {e}"
    if sr is None or len(sr) == 0:
        open(out_nod, "w").close()
        return f"TIC {int(tic_id)}: no SPOC LCs"

    # Parse sector from search result
    def _sec(r):
        try:
            m = str(r.mission[0])
            parts = m.split()
            return int(parts[-1]) if parts[-1].isdigit() else None
        except:
            return None

    rows = []
    teff_star = np.nan
    downloaded_paths = []   # track every file/dir we touch for cleanup

    for i in range(len(sr)):
        sec = _sec(sr[i])
        if sec is None or sec not in need_sectors:
            continue

        import time as _time
        lc = None
        for _attempt in range(3):
            try:
                lc = sr[i].download(download_dir=CACHE_DIR, quality_bitmask=0)
                break
            except Exception as _e:
                if "429" in str(_e) and _attempt < 2:
                    _time.sleep(15 * (_attempt + 1))
                else:
                    break

        if lc is None:
            continue

        # Record the file path for cleanup
        lc_path = getattr(lc, "filename", None)
        if lc_path and os.path.exists(lc_path):
            downloaded_paths.append(lc_path)

        try:
            if lc_path and os.path.exists(lc_path):
                with fits.open(lc_path, memmap=False) as hdul:
                    if np.isnan(teff_star):
                        teff_star = _flt(hdul[0].header.get("TEFF"))
                    h1  = hdul[1].header
                    row = {"tic_id": int(tic_id), "sector": sec, "teff": teff_star}
                    for k in PDC_KEYS:
                        row[k.lower()] = _flt(h1.get(k))
                    rows.append(row)
            else:
                # Fallback: read from lightkurve metadata
                row = {"tic_id": int(tic_id), "sector": sec}
                if np.isnan(teff_star):
                    teff_star = _flt(lc.meta.get("TEFF"))
                row["teff"] = teff_star
                for k in PDC_KEYS:
                    row[k.lower()] = _flt(lc.meta.get(k))
                rows.append(row)
        except Exception:
            pass

    # ── Cleanup: delete everything we touched for this TIC ─────────────────────
    # 1. Direct file paths we recorded
    for p in downloaded_paths:
        parent = os.path.dirname(p)
        try:
            if os.path.isdir(parent):
                shutil.rmtree(parent)   # removes the _tp/ dir containing the fits
            elif os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    # 2. Broad sweep: any HLSP dir containing this TIC ID (catches edge cases)
    tic_str = f"{int(tic_id):016d}"
    for d in glob.glob(os.path.join(CACHE_DIR, "mastDownload", "HLSP",
                                    f"*{tic_str}*")):
        try:
            if os.path.isdir(d): shutil.rmtree(d)
            else:                os.remove(d)
        except Exception:
            pass

    if not rows:
        open(out_nod, "w").close()
        return f"TIC {tic_id}: no data extracted"

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return f"TIC {tic_id}: {len(rows)} sectors"

# ── Parallel processing ────────────────────────────────────────────────────────
log(f"Starting extraction ({N_WORKERS} workers)...\n")
done = 0
ok   = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(process_star, t): t for t in remaining}
    for f in as_completed(futs):
        done += 1
        result = f.result()
        if "sectors" in result:
            ok += 1
        if done % CHECKPOINT_EVERY == 0 or done <= 5:
            log(f"  [{done:>6}/{len(remaining):,}]  ok={ok}  {result}")

log(f"\nExtraction complete: {ok}/{done} stars yielded data")

# ── Merge all per-TIC CSVs ────────────────────────────────────────────────────
log(f"\nMerging per-TIC CSVs from {PDC_HEADERS_DIR}...")
csv_files = glob.glob(os.path.join(PDC_HEADERS_DIR, "tic_*.csv"))
log(f"  {len(csv_files):,} CSV files found")

parts = []
for path in csv_files:
    try:
        parts.append(pd.read_csv(path))
    except Exception:
        pass

feat = pd.concat(parts, ignore_index=True)
feat["tic_id"] = feat["tic_id"].astype("int64")
feat["sector"] = feat["sector"].astype("int32")

# Rename to lowercase to match parquet convention
feat = feat.rename(columns={k.lower(): k.lower() for k in PDC_KEYS})

log(f"  {len(feat):,} total (tic, sector) records")
log(f"\nFeature summary:")
for col in ["pr_wght2", "pdc_noi", "pdc_corp", "pdc_totp", "flfrcsap", "teff"]:
    if col not in feat.columns:
        continue
    s = feat[col]
    log(f"  {col:<12}  median={s.median():.3f}  nan%={s.isna().mean()*100:.1f}%")

# ── Join to parquet ────────────────────────────────────────────────────────────
log(f"\nJoining onto {PARQUET_IN}...")
df["tic_id"] = df["tic_id"].astype("int64")
df["sector"] = df["sector"].astype("int32")

new_cols = [c for c in feat.columns if c not in ("tic_id", "sector")]
before = set(df.columns)
df = df.merge(feat[["tic_id", "sector"] + new_cols], on=["tic_id", "sector"], how="left")

log(f"\nMatch rate:")
for col in new_cols:
    if col in df.columns:
        log(f"  {col:<12}  {df[col].notna().mean()*100:.1f}% non-null")

log(f"\nSaving → {PARQUET_OUT}")
df.to_parquet(PARQUET_OUT, index=False)
log(f"Done. {len(df):,} records, {df['tic_id'].nunique():,} stars.")
log(f"\nNext: update CONTINUOUS in train_flow_nsf.py to include:")
log(f"  'pr_wght2', 'pdc_noi', 'pdc_corp', 'pdc_totp', 'flfrcsap', 'teff'")
