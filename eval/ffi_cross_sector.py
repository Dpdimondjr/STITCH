"""
Cross-sector spatial LOO pattern — Cam 4 southern CVZ stars.

For each of 8 southern sectors, shows the real TESS CCD pixel image
(Cam 4, dominant CCD for that sector) with CVZ stars overlaid at their
detector (col/row) positions, coloured by LOO flux_offset.

Usage:
  python3 eval/ffi_cross_sector.py [parquet]
"""

import sys, os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from astropy.io import fits

PARQUET   = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                 "training_data_topup_pdc.parquet")
CAM       = 4
CACHE     = "tess_cache/ffi"
MAST_BASE = "https://archive.stsci.edu/missions/tess/ffi"

# Southern sectors only — Cam 4 points at the southern ecliptic pole
SOUTH_SECTORS = set(range(1, 14)) | set(range(27, 40)) | set(range(56, 70))

os.makedirs(CACHE, exist_ok=True)

# ── Load & filter ─────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset_loo"])
df = df[(df["flux_offset_loo"] > 0.85) & (df["flux_offset_loo"] < 1.15)]

cam4 = df[(df["cam"] == CAM) & (df["sector"].isin(SOUTH_SECTORS))].copy()
n_sec = cam4.groupby("tic_id")["sector"].nunique()
rich  = n_sec[n_sec >= 6].index          # relax threshold — southern only = fewer sectors
cam4  = cam4[cam4["tic_id"].isin(rich)].copy()
print(f"Cam{CAM} southern: {cam4['tic_id'].nunique():,} stars (≥6 southern sectors)")

# ── Pick 8 evenly-spaced southern sectors ────────────────────────────────────
avail_secs = sorted(cam4["sector"].unique())
idx = np.linspace(0, len(avail_secs) - 1, 8, dtype=int)
show_secs  = [avail_secs[i] for i in idx]
print(f"Candidate sectors: {show_secs}")

# ── Dominant CCD per sector ──────────────────────────────────────────────────
dom_ccd = {}
for sec in show_secs:
    g = cam4[cam4["sector"] == sec]
    dom_ccd[sec] = int(g["ccd"].value_counts().idxmax()) if len(g) else 4
print("Dominant CCD per sector:", {s: dom_ccd[s] for s in show_secs})

# ── FFI helpers ───────────────────────────────────────────────────────────────
def find_ffi_url(sector, cam, ccd):
    sector_url = f"{MAST_BASE}/s{sector:04d}/"
    r = requests.get(sector_url, timeout=15)
    years = re.findall(r'href="(\d{4}/)"', r.text)
    target_dir = f"{cam}-{ccd}/"
    for year in years:
        year_url = sector_url + year
        r = requests.get(year_url, timeout=15)
        doys = re.findall(r'href="(\d{3}/)"', r.text)
        for doy in doys:
            doy_url = year_url + doy
            r = requests.get(doy_url, timeout=15)
            if target_dir not in re.findall(r'href="(\d+-\d+/)"', r.text):
                continue
            file_url = doy_url + target_dir
            r = requests.get(file_url, timeout=15)
            files = re.findall(r'href="(tess[^"]+_ffic\.fits)"', r.text)
            if files:
                return file_url + files[0]
    return None

def get_ffi(sector, cam, ccd):
    fname = f"s{sector:04d}_cam{cam}_ccd{ccd}_ffic.fits"
    local = os.path.join(CACHE, fname)
    if os.path.exists(local):
        print(f"  S{sector:02d} CCD{ccd}: cached")
        return local
    url = find_ffi_url(sector, cam, ccd)
    if url is None:
        print(f"  S{sector:02d} CCD{ccd}: not found on MAST")
        return None
    size = int(requests.head(url, timeout=10).headers.get("Content-Length", 0))
    print(f"  S{sector:02d} CCD{ccd}: downloading ({size/1e6:.0f} MB) …")
    r = requests.get(url, timeout=300, stream=True)
    with open(local, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return local

def load_science(path):
    """Return the 2048×2048 science region, or None if the file is bad."""
    with fits.open(path, memmap=True) as hdul:
        for ext in range(len(hdul)):
            d = hdul[ext].data
            if d is None or d.ndim != 2:
                continue
            img = d.astype("float32")
            # Detect science column range (>10 mean count)
            col_means = np.nanmean(np.where(img > 0, img, np.nan), axis=0)
            sci_cols = np.where(col_means > 10)[0]
            if len(sci_cols) < 1800:
                print(f"    *** only {len(sci_cols)} science cols — bad amplifier, skipping")
                return None
            c0, c1 = sci_cols[0], sci_cols[0] + 2048
            return img[:2048, c0:c1]   # 2048×2048 science pixels
    return None

# ── Download FFIs, skip bad ones ──────────────────────────────────────────────
print(f"\nFetching Cam{CAM} FFIs …")
ffis = {}
for sec in show_secs:
    ccd = dom_ccd[sec]
    path = get_ffi(sec, CAM, ccd)
    if path:
        sci = load_science(path)
        if sci is not None:
            ffis[sec] = sci
            print(f"    science shape: {sci.shape}")

# If any sectors failed, warn but continue
missing = [s for s in show_secs if s not in ffis]
if missing:
    print(f"\nWarning: no valid FFI for sectors {missing} — panels will show text placeholder.")

# ── Colour scale — shared across all sectors ──────────────────────────────────
all_offsets = pd.concat([
    cam4[(cam4["sector"] == sec) & (cam4["ccd"] == dom_ccd[sec])]["flux_offset_loo"]
    for sec in show_secs
])
lo = np.percentile(all_offsets, 1)
hi = np.percentile(all_offsets, 99)
extent_c = max(abs(lo - 1.0), abs(hi - 1.0), 0.010)
NORM = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - extent_c, vmax=1.0 + extent_c)
CMAP = "RdBu_r"

# ── Figure ────────────────────────────────────────────────────────────────────
BG   = "#0b0f1a"
MUTE = "#7a9cc4"
TEXT = "#c4d4ee"

NCOLS = 4
NROWS = (len(show_secs) + NCOLS - 1) // NCOLS
fig, axes = plt.subplots(NROWS, NCOLS,
                         figsize=(NCOLS * 3.5, NROWS * 3.8 + 0.7),
                         facecolor=BG)
axes = np.array(axes).reshape(NROWS, NCOLS)

for i, sec in enumerate(show_secs):
    ax = axes[i // NCOLS, i % NCOLS]
    ax.set_facecolor("#020408")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1c2d45")

    ccd = dom_ccd[sec]

    # ── Pixel image — science region, col/row coords 1–2048 ──────────────────
    if sec in ffis:
        img = ffis[sec]
        vlo, vhi = np.nanpercentile(img[img > 0], [1, 99.5])
        display  = np.log1p(np.clip(img - vlo, 0, None))
        # extent aligns with SPOC col/row coords (1-indexed, 1–2048)
        ax.imshow(display, cmap="Greys_r", origin="lower",
                  interpolation="bilinear", aspect="equal",
                  extent=[1, 2049, 1, 2049])
        ax.set_xlim(1, 2049)
        ax.set_ylim(1, 2049)
    else:
        ax.text(0.5, 0.5, "FFI unavailable", ha="center", va="center",
                transform=ax.transAxes, color=MUTE, fontsize=9)
        ax.set_xlim(1, 2049)
        ax.set_ylim(1, 2049)

    # ── Star overlay — stars on the dominant CCD for this sector ─────────────
    g = cam4[(cam4["sector"] == sec) & (cam4["ccd"] == ccd)]
    if len(g):
        ax.scatter(g["col"], g["row"],
                   c=g["flux_offset_loo"], cmap=CMAP, norm=NORM,
                   s=8, alpha=0.85, linewidths=0, edgecolors="none",
                   rasterized=True, zorder=3)

    med = g["flux_offset_loo"].median() if len(g) else float("nan")
    ax.set_title(f"Sector {sec}  (CCD{ccd})", color=TEXT, fontsize=10,
                 fontweight="bold", pad=4)
    ax.text(0.03, 0.97, f"n={len(g)}  med={med:.4f}",
            transform=ax.transAxes, color=MUTE, fontsize=7.5, va="top")
    ax.set_xlabel("col (px)", color=MUTE, fontsize=7)
    ax.set_ylabel("row (px)", color=MUTE, fontsize=7)
    ax.tick_params(colors="#3a5070", labelsize=6)

for j in range(len(show_secs), NROWS * NCOLS):
    axes[j // NCOLS, j % NCOLS].set_visible(False)

# ── Shared colourbar — keep it inside the figure, not overlapping panels ──────
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("LOO flux_offset", color=TEXT, fontsize=9)
cbar.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)
cbar.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

fig.suptitle(
    f"TESS Cam{CAM} — southern CVZ across {len(show_secs)} sectors\n"
    "Full CCD pixel image (log e⁻/s).  Dots: stars coloured by LOO flux_offset.  "
    "LMC visible at lower-right of early sectors.",
    color=TEXT, fontsize=10, y=1.01
)
plt.tight_layout(pad=0.5, rect=[0, 0, 0.91, 1.0])
plt.savefig("eval/ffi_cross_sector.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
print("\nSaved → eval/ffi_cross_sector.png")
