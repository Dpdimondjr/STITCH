"""
Download and display a full TESS CCD FFI (2048×2048 pixels, compressed for display).

Downloads one calibrated FFI FITS file for a given sector/cam/CCD from MAST
and renders the full detector as a single compressed image.

Usage:
  python3 eval/ffi_full_ccd.py [sector=29] [cam=3] [ccd=3]
"""
import sys, os, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

SECTOR = int(next((s for s in sys.argv[1:] if s.lstrip("-").isdigit()), "29"))
CAM    = int(next((s for s in sys.argv[1:] if s in "1234" and len(s)==1), "3"))
CCD    = 3

OUT = f"eval/ffi_full_ccd_s{SECTOR}.png"

# ── Try to find a cached FFI FITS file ────────────────────────────────────────
def find_cached(sector, cam, ccd):
    """Search tess_cache for an existing ffic FITS for this sector/cam/ccd."""
    pat = re.compile(rf"tess\d+-s{sector:04d}-{cam}-{ccd}-\d+-s_ffic\.fits")
    for root, _, files in os.walk("tess_cache"):
        for f in files:
            if pat.match(f):
                return os.path.join(root, f)
    return None

fits_path = find_cached(SECTOR, CAM, CCD)

# ── Download if not cached ─────────────────────────────────────────────────────
if fits_path is None:
    print(f"No cached FFI found for S{SECTOR} Cam{CAM} CCD{CCD} — querying MAST …")
    from astroquery.mast import Observations

    obs = Observations.query_criteria(
        obs_collection="TESS",
        sequence_number=SECTOR,
        dataproduct_type="image",
    )
    print(f"Found {len(obs)} observations")

    products = Observations.get_product_list(obs[:20])  # first 20 obs to limit scope
    cam_pat = re.compile(rf"-{CAM}-{CCD}-")
    keep = [p for p in products
            if "_ffic.fits" in str(p["productFilename"])
            and cam_pat.search(str(p["productFilename"]))]
    print(f"Matching calibrated FFI products: {len(keep)}")

    if not keep:
        # Broaden search
        all_prod = Observations.get_product_list(obs)
        keep = [p for p in all_prod
                if "_ffic.fits" in str(p["productFilename"])
                and cam_pat.search(str(p["productFilename"]))]
        print(f"Broadened search: {len(keep)} products")

    if not keep:
        print("No matching FFI products found — exiting")
        sys.exit(1)

    from astropy.table import Table
    keep_table = Table(rows=keep[:1])
    dl = Observations.download_products(keep_table, download_dir="tess_cache")
    fits_path = dl["Local Path"][0]
    print(f"Downloaded → {fits_path}")
else:
    print(f"Using cached FFI: {fits_path}")

# ── Load the FITS image ────────────────────────────────────────────────────────
print("Loading FITS …")
with fits.open(fits_path, memmap=True) as hdul:
    # Calibrated FFI: primary + science extension
    # Extension 0 is empty header; ext 1 is the calibrated science image
    for ext in range(len(hdul)):
        if hdul[ext].data is not None and hdul[ext].data.ndim == 2:
            img = hdul[ext].data.astype("float32")
            hdr = hdul[ext].header
            print(f"  Using extension {ext}: shape={img.shape}  "
                  f"NAXIS1={hdr.get('NAXIS1')} NAXIS2={hdr.get('NAXIS2')}")
            break
    else:
        print("No 2D image extension found")
        sys.exit(1)

# ── Render ─────────────────────────────────────────────────────────────────────
img[img <= 0] = np.nan
img[~np.isfinite(img)] = np.nan

vlo, vhi = np.nanpercentile(img, [2, 99])
display = np.log1p(np.clip(img - vlo, 0, None))

BG = "#0b0f1a"
fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
ax.set_facecolor("#020408")
for sp in ax.spines.values():
    sp.set_edgecolor("#1c2d45")

ax.imshow(display, cmap="Greys_r", origin="lower",
          interpolation="bilinear", aspect="equal")

ax.set_xlabel("Column (px)", color="#7a9cc4", fontsize=9)
ax.set_ylabel("Row (px)", color="#7a9cc4", fontsize=9)
ax.tick_params(colors="#3a5070", labelsize=8)

ax.set_title(
    f"TESS Sector {SECTOR} — Cam {CAM} CCD {CCD}  ({img.shape[1]}×{img.shape[0]} px)",
    color="#c4d4ee", fontsize=10, pad=8
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT}")
