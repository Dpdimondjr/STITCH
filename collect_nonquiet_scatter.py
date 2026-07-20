"""
Collect between-sector scatter for non-quiet TARS stars.

Downloads all available sectors (1-10) per star, computes sector median,
then measures std of normalized medians — same metric as quiet star validation.
"""

import re, io, pickle, warnings, requests, numpy as np, pandas as pd
import pyarrow.feather as feather
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.io import fits

warnings.filterwarnings("ignore")

CACHE_FILE = "nonquiet_scatter_cache.parquet"
N_SAMPLE   = 150
N_WORKERS  = 12
MAST_BASE  = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-spoc"

def tic_url(tic_id, sector):
    t = f"{int(tic_id):016d}"
    g1, g2, g3, g4 = t[0:4], t[4:8], t[8:12], t[12:16]
    fn = f"hlsp_tess-spoc_tess_phot_{t}-s{sector:04d}_tess_v1_lc.fits"
    return f"{MAST_BASE}/s{sector:04d}/target/{g1}/{g2}/{g3}/{g4}/{fn}"

def sector_median_from_url(url):
    """Download FITS, return median PDCSAP_FLUX (electrons/s, non-normalized)."""
    r = requests.get(url, timeout=30, stream=True)
    if r.status_code != 200:
        return None
    with fits.open(io.BytesIO(r.content)) as hdul:
        flux = hdul[1].data["PDCSAP_FLUX"].astype(float)
        qual = hdul[1].data["QUALITY"].astype(int)
        flux[qual != 0] = np.nan
        flux[flux <= 0] = np.nan
    good = flux[np.isfinite(flux)]
    if len(good) < 100:
        return None
    return float(np.nanmedian(good))

# ── Build TIC → sectors mapping ──────────────────────────────────────────
print("Loading TIC→sector map...", flush=True)
with open("/tmp/tess_spoc_tics_s1_10.pkl", "rb") as f:
    spoc_tics = pickle.load(f)

# Rebuild per-sector TIC sets from disk (already downloaded)
# We need TIC → list_of_sectors, not just membership
# Re-scrape to get per-sector TIC lists
import concurrent.futures

def fetch_tic_list(sector):
    url = (f"https://archive.stsci.edu/hlsps/tess-spoc/download_scripts/"
           f"hlsp_tess-spoc_tess_phot_s{sector:04d}_tess_v1_dl-lc.sh")
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return sector, set()
    return sector, {int(m.group(1)) for m in re.finditer(r'phot_(\d{16})-s', r.text)}

tic_to_sectors = {}
with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(fetch_tic_list, s): s for s in range(1, 11)}
    for fut in as_completed(futures):
        sec, tics = fut.result()
        for tic in tics:
            tic_to_sectors.setdefault(tic, []).append(sec)
        print(f"  Sector {sec}: {len(tics):,}", flush=True)

# Keep only TICs with ≥3 sectors (meaningful scatter estimate)
multi_sector = {t: sorted(s) for t, s in tic_to_sectors.items() if len(s) >= 3}
print(f"TICs with ≥3 sectors: {len(multi_sector):,}", flush=True)

# ── Load non-quiet catalog ────────────────────────────────────────────────
tars2 = feather.read_table("/Users/daviddimond/Documents/STITCH/tars_table_2.feather").to_pandas()
nq_spoc = tars2[tars2["TICID"].isin(multi_sector)].copy()
print(f"Non-quiet with ≥3 TESS-SPOC sectors: {len(nq_spoc):,}", flush=True)

# Load quiet star Tmag range
train = pd.read_parquet("/Users/daviddimond/Documents/STITCH/training_data.parquet")
tmag_lo, tmag_hi = train["tmag"].min(), train["tmag"].max()

# Stratified sample
rng = np.random.default_rng(42)
bins = [(7,9), (9,11), (11,13)]
sample_rows = []
for lo, hi in bins:
    sub = nq_spoc[(nq_spoc["Tmag"]>=lo) & (nq_spoc["Tmag"]<hi)]
    n = min(N_SAMPLE // len(bins), len(sub))
    idx = rng.choice(len(sub), size=n, replace=False)
    sample_rows.extend(sub.iloc[idx][["TICID","Tmag"]].itertuples(index=False))

print(f"Sample: {len(sample_rows)} TICs", flush=True)

# ── Collect sector medians per star ───────────────────────────────────────

def fetch_star_scatter(row):
    tic_id, tmag = int(row.TICID), float(row.Tmag)
    sectors = tic_to_sectors.get(tic_id, [])
    medians = {}

    def dl_sector(s):
        med = sector_median_from_url(tic_url(tic_id, s))
        if med is not None:
            medians[s] = med

    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(dl_sector, sectors))

    if len(medians) < 3:
        return None

    vals = np.array(list(medians.values()))
    global_med = vals.mean()
    if global_med <= 0:
        return None
    norm = vals / global_med
    scatter = float(np.std(norm))
    return {
        "tic_id": tic_id, "tmag": tmag,
        "n_sectors": len(medians),
        "scatter": scatter,
        "global_median": global_med,
    }

results = []
done = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(fetch_star_scatter, row): row for row in sample_rows}
    for fut in as_completed(futures):
        done += 1
        r = fut.result()
        if r is not None:
            results.append(r)
        if done % 10 == 0 or done == len(sample_rows):
            print(f"  {done}/{len(sample_rows)}  valid={len(results)}", flush=True)

df_nq = pd.DataFrame(results)
print(f"\nCollected {len(df_nq)} non-quiet stars with scatter")
df_nq.to_parquet(CACHE_FILE)
print(f"Saved → {CACHE_FILE}")
if len(df_nq) > 0:
    print(df_nq[["tmag","n_sectors","scatter"]].describe())
