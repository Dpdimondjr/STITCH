"""
Small experiment: download ~15 TESS SPOC light curve FITS files spanning
the Tmag range and inspect what background information is available.

Checks:
  - Header keywords related to background (BACKAPP, SKYAPP, etc.)
  - Whether SAP_BKG column exists in the binary table
  - Median SAP_BKG vs sector_median -> background fraction

Cleans up all downloaded files afterward.
"""

import os, shutil, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import lightkurve as lk
lk.log.setLevel("ERROR")
from astropy.io import fits

CACHE_DIR = "./tess_cache_experiment"
os.makedirs(CACHE_DIR, exist_ok=True)

# Stars sampled across Tmag 7.5–13, from training_data_topup.parquet
TARGETS = [
    {"tic_id": 167721564, "sector": 61, "tmag": 7.70,  "crowdsap": 0.986, "sector_median": 130026},
    {"tic_id": 284575942, "sector": 43, "tmag": 8.79,  "crowdsap": 0.984, "sector_median": 49149},
    {"tic_id": 351872819, "sector": 18, "tmag": 8.49,  "crowdsap": 0.960, "sector_median": 63924},
    {"tic_id": 16330502,  "sector": 24, "tmag": 9.97,  "crowdsap": 0.995, "sector_median": 16640},
    {"tic_id": 31412505,  "sector":  1, "tmag": 9.51,  "crowdsap": 0.946, "sector_median": 25832},
    {"tic_id": 258032232, "sector": 42, "tmag": 10.85, "crowdsap": 0.986, "sector_median": 7334},
    {"tic_id": 31943791,  "sector": 61, "tmag": 10.12, "crowdsap": 0.986, "sector_median": 14419},
    {"tic_id": 453079512, "sector": 27, "tmag": 11.50, "crowdsap": 0.930, "sector_median": 4100},
    {"tic_id": 410450228, "sector":  1, "tmag": 12.10, "crowdsap": 0.960, "sector_median": 2200},
    {"tic_id": 262400835, "sector":  1, "tmag": 12.50, "crowdsap": 0.950, "sector_median": 1400},
]

results = []

for t in TARGETS:
    tic, sec, tmag = t["tic_id"], t["sector"], t["tmag"]
    print(f"\nTIC {tic}  Tmag={tmag}  sector={sec}  crowdsap={t['crowdsap']}")

    try:
        sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="TESS-SPOC")
        # Filter to this sector
        try:
            sr = sr[sr.exptime.value >= 100]
        except Exception:
            pass

        # Find the right sector
        match = None
        for i in range(len(sr)):
            try:
                m = str(sr[i].mission[0])
                parts = m.split()
                s = int(parts[-1]) if parts[-1].isdigit() else None
                if s == sec:
                    match = sr[i]
                    break
            except Exception:
                continue

        if match is None:
            print(f"  → sector {sec} not found in search results")
            continue

        lc = match.download(download_dir=CACHE_DIR, quality_bitmask=0)
        lc_path = getattr(lc, "filename", None)

        if not lc_path or not os.path.exists(lc_path):
            print(f"  → no file path available")
            continue

        with fits.open(lc_path, memmap=False) as hdul:
            h0 = hdul[0].header
            h1 = hdul[1].header
            cols = [c.name for c in hdul[1].columns]
            data = hdul[1].data

            # Header background keywords
            bg_keys = {k: h1.get(k) for k in h1.keys()
                       if any(x in k.upper() for x in ["BACK","SKY","BKG","APP"])}
            h0_bg  = {k: h0.get(k) for k in h0.keys()
                      if any(x in k.upper() for x in ["BACK","SKY","BKG","APP"])}

            print(f"  Header (ext1) bg keys: {bg_keys}")
            print(f"  Header (ext0) bg keys: {h0_bg}")
            print(f"  Columns: {cols}")

            row = {"tic_id": tic, "sector": sec, "tmag": tmag,
                   "crowdsap": t["crowdsap"],
                   "sector_median_parquet": t["sector_median"]}

            # SAP_BKG column — per-cadence background in e-/s
            if "SAP_BKG" in cols:
                bkg = data["SAP_BKG"]
                bkg = bkg[np.isfinite(bkg)]
                med_bkg = float(np.median(bkg))
                row["sap_bkg_median"] = med_bkg
                row["bkg_fraction"]   = med_bkg / t["sector_median"]
                print(f"  SAP_BKG median = {med_bkg:.1f} e-/s")
                print(f"  bkg / star flux = {med_bkg/t['sector_median']*100:.2f}%")
            else:
                row["sap_bkg_median"] = None
                row["bkg_fraction"]   = None
                print(f"  SAP_BKG column NOT present")

            # FLUX_BKG column (alternative name in some versions)
            if "FLUX_BKG" in cols:
                bkg2 = data["FLUX_BKG"]
                bkg2 = bkg2[np.isfinite(bkg2)]
                print(f"  FLUX_BKG median = {np.median(bkg2):.1f}")

            results.append(row)

        # Cleanup
        parent = os.path.dirname(lc_path)
        if os.path.isdir(parent):
            shutil.rmtree(parent)
        elif os.path.exists(lc_path):
            os.remove(lc_path)

    except Exception as e:
        print(f"  → ERROR: {e}")

# Cleanup cache dir
shutil.rmtree(CACHE_DIR, ignore_errors=True)

# Summary table
print("\n" + "="*70)
print("SUMMARY: background fraction by Tmag")
print("="*70)
print(f"{'TIC':<12} {'Tmag':>5} {'crowdsap':>9} {'bkg (e/s)':>11} {'bkg %':>7}")
print("-"*70)
for r in results:
    bkg   = r.get("sap_bkg_median")
    bfrac = r.get("bkg_fraction")
    bkg_s   = f"{bkg:>11.1f}" if bkg is not None else "        N/A"
    bfrac_s = f"{bfrac*100:>6.2f}%" if bfrac is not None else "    N/A"
    print(f"{r['tic_id']:<12} {r['tmag']:>5.2f} {r['crowdsap']:>9.3f} {bkg_s} {bfrac_s}")
