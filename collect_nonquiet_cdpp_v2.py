"""
Fast CDPP collection for non-quiet stars using direct MAST URL construction.

We know the TESS-SPOC URL pattern from the download scripts:
  mast:HLSP/tess-spoc/s{SSSS}/target/{G1}/{G2}/{G3}/{G4}/
  hlsp_tess-spoc_tess_phot_{TIC16}-s{SSSS}_tess_v1_lc.fits

We also know which TICs appear in which sectors (from the sector download scripts).
This lets us skip the slow lightkurve search step and go straight to download.
"""

import re, io, pickle, warnings, requests, numpy as np, pandas as pd
import pyarrow.feather as feather
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.io import fits

warnings.filterwarnings("ignore")

CACHE_FILE = "nonquiet_cdpp_cache.parquet"
N_SAMPLE   = 200
N_WORKERS  = 12
MAST_BASE  = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-spoc"

def tic_url(tic_id, sector):
    t = f"{int(tic_id):016d}"
    g1, g2, g3, g4 = t[0:4], t[4:8], t[8:12], t[12:16]
    fn = f"hlsp_tess-spoc_tess_phot_{t}-s{sector:04d}_tess_v1_lc.fits"
    return f"{MAST_BASE}/s{sector:04d}/target/{g1}/{g2}/{g3}/{g4}/{fn}"

# ── Build TIC → sector mapping from scripts ──────────────────────────────

print("Building TIC→sector map from download scripts...", flush=True)
tic_to_sectors = {}
with ThreadPoolExecutor(max_workers=10) as ex:
    def fetch_script(sec):
        url = (f"https://archive.stsci.edu/hlsps/tess-spoc/download_scripts/"
               f"hlsp_tess-spoc_tess_phot_s{sec:04d}_tess_v1_dl-lc.sh")
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return sec, {}
        return sec, {int(m.group(1)): sec
                     for m in re.finditer(r'phot_(\d{16})-s', r.text)}
    futures = {ex.submit(fetch_script, s): s for s in range(1, 11)}
    for fut in as_completed(futures):
        sec, mapping = fut.result()
        for tic, s in mapping.items():
            if tic not in tic_to_sectors:
                tic_to_sectors[tic] = []
            tic_to_sectors[tic].append(s)
        print(f"  Sector {sec}: {len(mapping):,} TICs", flush=True)

print(f"Total TICs mapped: {len(tic_to_sectors):,}", flush=True)

# ── Load non-quiet catalog ────────────────────────────────────────────────
print("Loading TARS table 2...", flush=True)
tars2 = feather.read_table("/Users/daviddimond/Documents/STITCH/tars_table_2.feather").to_pandas()

# Intersect with TESS-SPOC confirmed TICs
spoc_set = set(tic_to_sectors.keys())
nq_spoc  = tars2[tars2["TICID"].isin(spoc_set)].copy()
print(f"Non-quiet with TESS-SPOC (s1-10): {len(nq_spoc):,}", flush=True)

# Load quiet star Tmag range
train = pd.read_parquet("/Users/daviddimond/Documents/STITCH/training_data.parquet")
tmag_lo = train["tmag"].min()
tmag_hi = train["tmag"].max()

# Stratified sample by Tmag
rng = np.random.default_rng(42)
mask = (nq_spoc["Tmag"] >= tmag_lo) & (nq_spoc["Tmag"] <= tmag_hi)
nq_matched = nq_spoc[mask].copy()

bins = [(7,9), (9,11), (11,13)]
sample_tics = []
for lo, hi in bins:
    sub = nq_matched[(nq_matched["Tmag"]>=lo) & (nq_matched["Tmag"]<hi)]
    n = min(N_SAMPLE // len(bins), len(sub))
    idx = rng.choice(len(sub), size=n, replace=False)
    sample_tics.extend(sub.iloc[idx][["TICID","Tmag"]].itertuples(index=False))

print(f"Sample: {len(sample_tics)} TICs", flush=True)

# ── Download FITS and compute CDPP ────────────────────────────────────────

def compute_cdpp_from_url(url, transit_dur_cadences=2):
    """Download FITS LC, compute 1-h CDPP (ppm) from PDCSAP_FLUX."""
    r = requests.get(url, timeout=30, stream=True)
    if r.status_code != 200:
        return None
    data = r.content
    with fits.open(io.BytesIO(data)) as hdul:
        lc_ext = hdul[1]
        flux   = lc_ext.data["PDCSAP_FLUX"].astype(float)
        qual   = lc_ext.data["QUALITY"].astype(int)
        flux[qual != 0] = np.nan
        flux[flux <= 0] = np.nan
    good = flux[np.isfinite(flux)]
    if len(good) < 50:
        return None
    med = np.nanmedian(good)
    norm = good / med  # normalize
    # CDPP: rms of transit-box-filtered light curve (simple box = transit_dur_cadences)
    n = len(norm)
    box = transit_dur_cadences
    n_full = n - box + 1
    if n_full < 20:
        return None
    rms_vals = [np.std(norm[i:i+box]) for i in range(0, n_full, box)]
    cdpp_ppm = float(np.median(rms_vals)) * 1e6 / np.sqrt(box)
    return cdpp_ppm, float(med)

def fetch_star(row):
    tic_id, tmag = int(row.TICID), float(row.Tmag)
    sectors = tic_to_sectors.get(tic_id, [])
    if not sectors:
        return None
    # Try each sector until we get a good download
    for sec in sorted(sectors)[:3]:
        url = tic_url(tic_id, sec)
        result = compute_cdpp_from_url(url)
        if result is not None:
            cdpp, flux_med = result
            if 0 < cdpp < 1e5:
                return {"tic_id": tic_id, "tmag": tmag, "cdpp1_0": cdpp,
                        "flux_med_norm": flux_med, "sector": sec}
    return None

results = []
done = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(fetch_star, row): row for row in sample_tics}
    for fut in as_completed(futures):
        done += 1
        r = fut.result()
        if r is not None:
            results.append(r)
        if done % 20 == 0 or done == len(sample_tics):
            print(f"  {done}/{len(sample_tics)}  valid={len(results)}", flush=True)

df_nq = pd.DataFrame(results)
print(f"\nCollected {len(df_nq)} non-quiet stars with CDPP")
df_nq.to_parquet(CACHE_FILE)
print(f"Saved → {CACHE_FILE}")
if len(df_nq) > 0:
    print(df_nq[["tmag","cdpp1_0"]].describe())
