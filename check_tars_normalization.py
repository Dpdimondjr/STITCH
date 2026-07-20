"""
Download one TARS light curve and check:
  1. Is flux normalized per sector (median ~1.0) or in absolute units?
  2. What columns are available?
  3. How do sector-to-sector medians compare to SPOC for the same star?
"""

import numpy as np
import warnings
from astropy.io import fits
from astroquery.mast import Observations
import lightkurve as lk

warnings.filterwarnings("ignore")

TIC = 349683816  # star we already have 37 SPOC records for

# ── 1. Get TARS products for this star ────────────────────────────────────────

print(f"Querying TARS products for TIC {TIC}...")
obs = Observations.query_criteria(
    provenance_name="TARS",
    target_name=str(TIC),
)
print(f"Found {len(obs)} TARS observations (sectors)")

# Download just the first 3 sectors
products = Observations.get_product_list(obs[:3])
fits_products = products[
    [str(uri).endswith("_lc.fits") for uri in products["dataURI"]]
]
print(f"Downloading {len(fits_products)} FITS files...\n")
manifest = Observations.download_products(fits_products, download_dir="/tmp/tars_check")

# ── 2. Inspect flux normalization ─────────────────────────────────────────────

tars_meds = {}
for row in manifest:
    fpath = row["Local Path"]
    with fits.open(fpath) as hdul:
        print(f"File: {fpath.split('/')[-1]}")
        print(f"  Extensions: {[h.name for h in hdul]}")
        hdr  = hdul[1].header
        data = hdul[1].data
        print(f"  Columns: {data.names}")
        sec  = hdr.get("SECTOR", "?")
        flux = data["flux"].astype(float)
        med  = np.nanmedian(flux)
        tars_meds[sec] = med
        print(f"  Sector {sec}: median={med:.6f}  std={np.nanstd(flux):.6f}")
        # Check for any normalization header keywords
        for key in ["NORMALIZED", "FLUXTYPE", "FLUXUNIT", "BUNIT"]:
            if key in hdr:
                print(f"  {key}: {hdr[key]}")
        print()

# ── 3. Compare sector-to-sector ratios ────────────────────────────────────────

secs  = sorted(tars_meds.keys())
vals  = [tars_meds[s] for s in secs]
print("TARS sector medians:")
for s, v in zip(secs, vals):
    print(f"  Sector {s}: {v:.6f}")

if len(vals) >= 2:
    ratios = [vals[i]/vals[i+1] for i in range(len(vals)-1)]
    print(f"\nSector-to-sector ratios: {[f'{r:.4f}' for r in ratios]}")
    if all(abs(r - 1.0) < 0.005 for r in ratios):
        print("\n→ TARS flux is NORMALIZED per sector (all ratios ≈ 1.0)")
        print("  Cannot use TARS LCs directly for STITCH training.")
        print("  But TARS Table 4 systematic_score is still valuable as a star filter.")
    else:
        print("\n→ TARS flux preserves inter-sector offsets (ratios differ from 1.0)")
        print("  TARS LCs may be usable for STITCH training directly.")

# ── 4. Compare against SPOC for the same star/sectors ────────────────────────

print("\nFor comparison — SPOC medians for same sectors:")
sr = lk.search_lightcurve(f"TIC {TIC}", mission="TESS", author="SPOC")
sr = sr[sr.exptime.value == 120]
for sec in secs[:3]:
    matches = [i for i in range(len(sr)) if hasattr(sr[i], 'mission')]
    try:
        lc = sr[list(secs).index(sec)].download()
        med = np.nanmedian(lc.flux.value)
        print(f"  Sector {sec}: {med:.2f} e-/s")
    except Exception:
        pass
