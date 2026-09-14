"""
Show actual TESS detector pixel images for stars with different LOO offsets.

Downloads SPOC TPFs (pre-made TESS pixel cutouts ~11×11 px) for 8 stars
spread across cam 3 CCD 3 in sector 29, ranging from low to high flux_offset.
Demonstrates that nearby stars on the same CCD have correlated systematics.

Usage:
  python3 eval/ffi_pixel_images.py [parquet]
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
SECTOR = 29
CAM, CCD = 3, 3

# Stars pre-selected: spread across col/row on cam3 CCD3, sorted by flux_offset
TIC_IDS = [24706007, 50311880, 431479636, 220523249,
           32091856,  32035972, 238181271, 234305600]

print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","ra","dec"])
df = df[(df.flux_offset > 0.85) & (df.flux_offset < 1.15)]
stars = df[(df.sector == SECTOR) & (df.cam == CAM) & (df.ccd == CCD) &
           (df.tic_id.isin(TIC_IDS))].set_index("tic_id")

# ── Download SPOC TPFs ────────────────────────────────────────────────────────
print(f"Downloading {len(TIC_IDS)} SPOC TPFs from MAST …")
tpfs = {}
for tic in TIC_IDS:
    try:
        sr  = lk.search_targetpixelfile(f"TIC {tic}", sector=SECTOR, mission="TESS")
        tpf = sr.download(quality_bitmask=0)
        flux = tpf.flux.value.astype("float32")
        flux[flux == 0] = np.nan
        tpfs[tic] = {
            "img":  np.nanmedian(flux, axis=0),
            "col":  stars.loc[tic, "col"],
            "row":  stars.loc[tic, "row"],
            "fo":   stars.loc[tic, "flux_offset"],
            "nsec": int(stars.loc[tic, "n_sectors_total"]),
            "shape": flux.shape,
        }
        print(f"  TIC {tic:>10}  col={tpfs[tic]['col']:>6.0f} "
              f"row={tpfs[tic]['row']:>6.0f}  "
              f"flux_offset={tpfs[tic]['fo']:.4f}  OK {flux.shape}")
    except Exception as e:
        print(f"  TIC {tic:>10}  FAIL: {e}")

if not tpfs:
    print("No TPFs downloaded — exiting")
    sys.exit(1)

# Sort by flux_offset
ordered = sorted(tpfs.items(), key=lambda kv: kv[1]["fo"])

# ── Figure 1: pixel stamp row ─────────────────────────────────────────────────
N = len(ordered)
fig = plt.figure(figsize=(2.5 * N, 3.8), facecolor="#0b0f1a")
gs = fig.add_gridspec(1, N, hspace=0.1, wspace=0.15)

offsets = np.array([v["fo"] for _, v in ordered])
ext = max(abs(offsets - 1.0).max(), 0.008)
VMIN, VMAX = 1.0 - ext, 1.0 + ext
NORM = TwoSlopeNorm(vcenter=1.0, vmin=VMIN, vmax=VMAX)
CMAP = "RdBu_r"

# ── Top row: actual pixel images ──────────────────────────────────────────────
for i, (tic, info) in enumerate(ordered):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor("#020408")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1c2d45")

    img = info["img"]
    vlo, vhi = np.nanpercentile(img, [5, 98])
    # Show in greyscale
    ax.imshow(np.log1p(np.clip(img - vlo, 0, None)),
              cmap="Greys_r", origin="lower", interpolation="nearest")

    # Tint the border by LOO offset colour
    fo = info["fo"]
    fo_norm = NORM(fo)
    import matplotlib.cm as cm
    border_color = cm.RdBu_r(fo_norm)
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color)
        sp.set_linewidth(3)

    # Flux offset badge
    badge_color = "#ef4444" if fo > 1.002 else ("#3b82f6" if fo < 0.998 else "#6b7280")
    ax.text(0.5, -0.06, f"{fo:.4f}", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=badge_color)
    ax.text(0.5, 1.01,
            f"col={info['col']:.0f}\nrow={info['row']:.0f}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color="#7a9cc4")
    ax.set_xticks([]); ax.set_yticks([])

    if i == 0:
        ax.set_ylabel("Actual TESS pixels\n(log-scaled)", color="#7a9cc4", fontsize=8)

fig.suptitle(
    f"Cam{CAM} CCD{CCD} Sector {SECTOR} — actual TESS pixel stamps\n"
    f"Each cutout: real e⁻/s pixel data (log-scaled). "
    f"Number below = LOO flux_offset.",
    color="#c4d4ee", fontsize=10, y=1.04
)
out = f"eval/ffi_pixel_stamps_s{SECTOR}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"Saved → {out}")

# ── Figure 2: square CCD map ──────────────────────────────────────────────────
fig2, ax_ccd = plt.subplots(figsize=(6, 6), facecolor="#0b0f1a")
ax_ccd.set_facecolor("#060a12")
for sp in ax_ccd.spines.values():
    sp.set_edgecolor("#1c2d45")

all_ccd = df[(df.sector == SECTOR) & (df.cam == CAM) & (df.ccd == CCD)]
ax_ccd.scatter(all_ccd["col"], all_ccd["row"],
               c=all_ccd["flux_offset"], cmap=CMAP, norm=NORM,
               s=2, alpha=0.5, linewidths=0, rasterized=True)

for i, (tic, info) in enumerate(ordered):
    fo = info["fo"]
    fo_norm = NORM(fo)
    color = cm.RdBu_r(fo_norm)
    ax_ccd.scatter(info["col"], info["row"],
                   s=120, c=[color], edgecolors="white", linewidths=1.2,
                   zorder=5)
    ax_ccd.text(info["col"] + 40, info["row"] + 40,
                f"{fo:.3f}", fontsize=7, color="white", zorder=6)

ax_ccd.set_xlim(0, 2048)
ax_ccd.set_ylim(0, 2048)
ax_ccd.set_aspect("equal")
ax_ccd.set_xlabel("col (px)", color="#7a9cc4", fontsize=9)
ax_ccd.set_ylabel("row (px)", color="#7a9cc4", fontsize=9)
ax_ccd.tick_params(colors="#3a5070", labelsize=8)
ax_ccd.set_title(
    f"Cam{CAM} CCD{CCD} Sector {SECTOR} — all {len(all_ccd):,} stars\n"
    f"Colour = LOO flux_offset   ●  = 8 stars shown above",
    color="#c4d4ee", fontsize=9, pad=6
)

sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig2.colorbar(sm, ax=ax_ccd, fraction=0.046, pad=0.04)
cbar.set_label("LOO flux_offset", color="#c4d4ee", fontsize=9)
cbar.ax.yaxis.set_tick_params(labelcolor="#7a9cc4", labelsize=8)
cbar.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

plt.tight_layout()
out2 = f"eval/ffi_ccd_map_s{SECTOR}.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"Saved → {out2}")
