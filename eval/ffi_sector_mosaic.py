"""
Download one real TESS FFI per sector (cam 3, CCD 3, sectors 1-10) from MAST
and render them as a mosaic showing the full 2048×2048 detector.

Files are cached in tess_cache/ffi/ so re-runs don't re-download.

Usage:
  python3 eval/ffi_sector_mosaic.py
"""

import os, re, warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

CAM      = 3
CCD      = 3
SECTORS  = list(range(1, 11))
CACHE    = "tess_cache/ffi"
MAST_BASE = "https://archive.stsci.edu/missions/tess/ffi"
OUT      = "eval/ffi_sector_mosaic.png"

os.makedirs(CACHE, exist_ok=True)

# ── Find one FFI FITS file per sector via MAST directory listing ───────────────
def find_ffi_url(sector, cam, ccd):
    """Walk the MAST archive tree to find the first FFI file for this sector/cam/ccd."""
    sector_url = f"{MAST_BASE}/s{sector:04d}/"
    r = requests.get(sector_url, timeout=15)
    years = re.findall(r'href="(\d{4}/)"', r.text)
    if not years:
        return None
    target_dir = f"{cam}-{ccd}/"
    for year in years:
        year_url = sector_url + year
        r = requests.get(year_url, timeout=15)
        doys = re.findall(r'href="(\d{3}/)"', r.text)
        for doy in doys:
            doy_url = year_url + doy
            r = requests.get(doy_url, timeout=15)
            cam_dirs = re.findall(r'href="(\d+-\d+/)"', r.text)
            if target_dir not in cam_dirs:
                continue
            file_url = doy_url + target_dir
            r = requests.get(file_url, timeout=15)
            files = re.findall(r'href="(tess[^"]+_ffic\.fits)"', r.text)
            if files:
                return file_url + files[0]
    return None

def download_ffi(sector, cam, ccd):
    """Return local path to a cached FFI FITS file, downloading if needed."""
    fname = f"s{sector:04d}_cam{cam}_ccd{ccd}_ffic.fits"
    local = os.path.join(CACHE, fname)
    if os.path.exists(local):
        print(f"  S{sector:02d}: cached → {local}")
        return local
    url = find_ffi_url(sector, cam, ccd)
    if url is None:
        print(f"  S{sector:02d}: could not find FFI URL")
        return None
    print(f"  S{sector:02d}: downloading {url.split('/')[-1]} ({requests.head(url, timeout=10).headers.get('Content-Length','?')} bytes)…")
    r = requests.get(url, timeout=120, stream=True)
    with open(local, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    print(f"       saved → {local}")
    return local

# ── Load image from FITS ──────────────────────────────────────────────────────
def load_image(path):
    with fits.open(path, memmap=True) as hdul:
        for ext in range(len(hdul)):
            if hdul[ext].data is not None and hdul[ext].data.ndim == 2:
                img = hdul[ext].data.astype("float32")
                return img
    return None

def render(img, vlo_pct=1, vhi_pct=99.5):
    img = img.copy()
    img[img <= 0] = np.nan
    vlo, vhi = np.nanpercentile(img, [vlo_pct, vhi_pct])
    return np.log1p(np.clip(img - vlo, 0, None))

# ── Download ──────────────────────────────────────────────────────────────────
print(f"Fetching cam{CAM}/CCD{CCD} FFIs for sectors {SECTORS[0]}–{SECTORS[-1]} …\n")
images = {}
for sec in SECTORS:
    path = download_ffi(sec, CAM, CCD)
    if path:
        img = load_image(path)
        if img is not None:
            images[sec] = img
            print(f"       shape={img.shape}  dtype={img.dtype}")

if not images:
    print("No images loaded — exiting")
    import sys; sys.exit(1)

# ── Plot ──────────────────────────────────────────────────────────────────────
BG   = "#0b0f1a"
MUTE = "#7a9cc4"
TEXT = "#c4d4ee"

ncols = 5
nrows = (len(images) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 3.2, nrows * 3.4),
                         facecolor=BG)
axes = np.array(axes).reshape(nrows, ncols)

for i, sec in enumerate(sorted(images)):
    ax = axes[i // ncols, i % ncols]
    ax.set_facecolor("#020408")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1c2d45")

    display = render(images[sec])
    ax.imshow(display, cmap="Greys_r", origin="lower",
              interpolation="bilinear", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Sector {sec}", color=TEXT, fontsize=10, pad=4)

# Hide unused axes
for j in range(len(images), nrows * ncols):
    axes[j // ncols, j % ncols].set_visible(False)

fig.suptitle(
    f"TESS full CCD — Cam {CAM}, CCD {CCD}  (2048 × 2048 px each)",
    color=TEXT, fontsize=12, y=1.01
)
plt.tight_layout(pad=0.4)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {OUT}")
