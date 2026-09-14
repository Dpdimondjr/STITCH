"""
FFI actual image + LOO offset overlay + cross-sector trend.

Downloads a real TESS FFI cutout from MAST (via lightkurve/TESScut),
overlays training stars colored by their LOO flux_offset, and shows
how the spatial offset pattern changes across sectors for the same sky patch.

Usage:
  python3 eval/ffi_actual_image.py [sector=29] [parquet]
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightkurve as lk
from astropy.coordinates import SkyCoord
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binned_statistic_2d

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
SECTOR  = int(next((s for s in sys.argv[1:] if s.lstrip("-").isdigit()), "29"))
CUTOUT  = 200   # pixel width/height of downloaded FFI cutout

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "ra", "dec"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]

# ── Pick region: cam 4 CCD 3 (densest, most consistent across sectors) ────────
# Cam 4 observes the CVZ so we can get many sectors for the cross-sector panel
CAM, CCD = 4, 3
focus = df[(df.cam == CAM) & (df.ccd == CCD)].copy()
anchor = df[(df.sector == SECTOR) & (df.cam == CAM) & (df.ccd == CCD)].copy()
print(f"Cam{CAM} CCD{CCD}: {focus['tic_id'].nunique():,} unique stars, "
      f"{focus['sector'].nunique()} sectors")
print(f"Sector {SECTOR}: {len(anchor):,} stars")

ra_c  = anchor["ra"].median()
dec_c = anchor["dec"].median()
print(f"Cutout centre: RA={ra_c:.4f}  Dec={dec_c:.4f}")

# ── Download real TESS FFI cutout ─────────────────────────────────────────────
print(f"\nDownloading {CUTOUT}×{CUTOUT} px TESScut from MAST (sector {SECTOR}) …")
coords = SkyCoord(ra=ra_c, dec=dec_c, unit="deg")
ffi_ok = False
try:
    sr   = lk.search_tesscut(coords, sector=SECTOR)
    tpf  = sr.download(cutout_size=CUTOUT, quality_bitmask=0)
    flux = tpf.flux.value.copy().astype("float32")
    flux[flux <= 0] = np.nan
    median_img = np.nanmedian(flux, axis=0)   # (CUTOUT, CUTOUT) e-/s
    wcs = tpf.wcs

    # Map training stars to cutout pixel coords
    sky  = SkyCoord(ra=anchor["ra"].values, dec=anchor["dec"].values, unit="deg")
    px, py = wcs.world_to_pixel(sky)
    anchor = anchor.copy()
    anchor["px"] = px
    anchor["py"] = py
    mask = (anchor["px"] >= 0) & (anchor["px"] < CUTOUT) & \
           (anchor["py"] >= 0) & (anchor["py"] < CUTOUT)
    in_cut = anchor[mask]
    print(f"  OK — {tpf.flux.shape}  |  {len(in_cut)} training stars within cutout")
    ffi_ok = True
except Exception as e:
    print(f"  Download failed: {e}")
    median_img = None

# ── Cross-sector: same sky patch, multiple sectors ────────────────────────────
# Find all sectors that observed the same small RA/Dec patch
RA_HALF, DEC_HALF = 2.0, 1.5
patch = focus[
    (focus["ra"]  >= ra_c  - RA_HALF)  & (focus["ra"]  <= ra_c  + RA_HALF) &
    (focus["dec"] >= dec_c - DEC_HALF) & (focus["dec"] <= dec_c + DEC_HALF)
].copy()

sector_counts = patch.groupby("sector")["tic_id"].count()
good_sectors  = sector_counts[sector_counts >= 10].index.sort_values()
# Pick up to 6 sectors spread across the range
step = max(1, len(good_sectors) // 6)
show_sectors = list(good_sectors[::step])[:6]
if SECTOR not in show_sectors:
    show_sectors = [SECTOR] + show_sectors[:5]
show_sectors = sorted(set(show_sectors))[:6]
print(f"\nCross-sector patch ({RA_HALF*2:.1f}°×{DEC_HALF*2:.1f}°): "
      f"{patch['tic_id'].nunique()} stars, {len(good_sectors)} sectors")
print(f"  Showing sectors: {show_sectors}")

# Color scale — shared across all panels
all_offsets = patch[patch["sector"].isin(show_sectors)]["flux_offset"]
lo = np.percentile(all_offsets, 1)
hi = np.percentile(all_offsets, 99)
extent = max(abs(lo - 1.0), abs(hi - 1.0), 0.008)
VMIN, VMAX = 1.0 - extent, 1.0 + extent
NORM = TwoSlopeNorm(vcenter=1.0, vmin=VMIN, vmax=VMAX)
CMAP = "RdBu_r"

# ── Figure ────────────────────────────────────────────────────────────────────
n_cross = len(show_sectors)
fig_cols = 2 + n_cross          # [FFI raw] [FFI+overlay] [sector panels…]
fig, axes = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 5))
fig.patch.set_facecolor("#0b0f1a")

def dark_ax(ax):
    ax.set_facecolor("#060a12")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1c2d45")
    ax.tick_params(colors="#3a5070", labelsize=7)

# ── Panel 0: Raw FFI image ────────────────────────────────────────────────────
ax = axes[0]
dark_ax(ax)
if ffi_ok and median_img is not None:
    vlo, vhi = np.nanpercentile(median_img, [2, 98])
    ax.imshow(np.log1p(np.clip(median_img - vlo, 0, None)),
              origin="lower", cmap="Greys_r",
              interpolation="nearest")
    ax.set_title(f"Actual TESS FFI\nSector {SECTOR} Cam{CAM} CCD{CCD}\n"
                 f"({CUTOUT}×{CUTOUT} px cutout, log-scaled)",
                 color="#c4d4ee", fontsize=9, pad=4)
    ax.set_xlabel("Cutout col (px)", color="#7a9cc4", fontsize=8)
    ax.set_ylabel("Cutout row (px)", color="#7a9cc4", fontsize=8)
else:
    ax.text(0.5, 0.5, "FFI download\nfailed", ha="center", va="center",
            transform=ax.transAxes, color="#5a7090", fontsize=10)
    ax.set_title("Actual TESS FFI\n(unavailable)", color="#7a9cc4", fontsize=9)

# ── Panel 1: FFI + LOO overlay ────────────────────────────────────────────────
ax = axes[1]
dark_ax(ax)
if ffi_ok and median_img is not None:
    vlo, vhi = np.nanpercentile(median_img, [2, 98])
    ax.imshow(np.log1p(np.clip(median_img - vlo, 0, None)),
              origin="lower", cmap="Greys_r", alpha=0.6,
              interpolation="nearest")
    sc = ax.scatter(in_cut["px"], in_cut["py"],
                    c=in_cut["flux_offset"], cmap=CMAP, norm=NORM,
                    s=max(4, 60000 // max(len(in_cut), 1)),
                    alpha=0.9, linewidths=0.4, edgecolors="none",
                    zorder=3)
    ax.set_title(f"FFI + LOO offset\n({len(in_cut)} stars in cutout)\n"
                 f"Red=flux>{VMAX:.3f}  Blue=flux<{VMIN:.3f}",
                 color="#c4d4ee", fontsize=9, pad=4)
    ax.set_xlabel("Cutout col (px)", color="#7a9cc4", fontsize=8)
else:
    ax.text(0.5, 0.5, "FFI download\nfailed", ha="center", va="center",
            transform=ax.transAxes, color="#5a7090", fontsize=10)
    ax.set_title("FFI + LOO offset\n(unavailable)", color="#7a9cc4", fontsize=9)

# ── Panels 2+: Cross-sector trend for same sky patch ─────────────────────────
for i, sec in enumerate(show_sectors):
    ax = axes[2 + i]
    dark_ax(ax)

    g = patch[patch["sector"] == sec]
    if len(g) < 5:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#3a5070", fontsize=9)
        ax.set_title(f"Sector {sec}", color="#7a9cc4", fontsize=9)
        continue

    sc = ax.scatter(g["ra"], g["dec"],
                    c=g["flux_offset"], cmap=CMAP, norm=NORM,
                    s=max(3, 1200 // len(g)),
                    alpha=0.85, linewidths=0, rasterized=True)

    med = g["flux_offset"].median()
    std = g["flux_offset"].std()
    cam_str = f"Cam{int(g['cam'].mode()[0])}"
    ax.set_title(f"Sector {sec}  ({cam_str})\n"
                 f"n={len(g)}  med={med:.4f}  σ={std:.4f}",
                 color="#c4d4ee", fontsize=8.5, pad=4)
    ax.set_xlabel("RA (°)", color="#7a9cc4", fontsize=7)
    if i == 0:
        ax.set_ylabel("Dec (°)", color="#7a9cc4", fontsize=7)
    ax.invert_xaxis()   # RA increases right-to-left on sky

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="vertical",
                    fraction=0.012, pad=0.01, shrink=0.7)
cbar.set_label("LOO flux_offset", color="#c4d4ee", fontsize=9)
cbar.ax.yaxis.set_tick_params(labelcolor="#7a9cc4", labelsize=7)
cbar.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

fig.suptitle(
    f"TESS Cam{CAM} CCD{CCD}  —  Actual FFI image vs LOO offset vs cross-sector variation\n"
    f"Same {RA_HALF*2:.0f}°×{DEC_HALF*2:.0f}° sky patch across {len(show_sectors)} sectors",
    color="#c4d4ee", fontsize=11, y=1.01
)

plt.tight_layout()
out = f"eval/ffi_actual_sector{SECTOR}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"\nSaved → {out}")
