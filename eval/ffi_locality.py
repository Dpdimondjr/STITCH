"""
FFI spatial locality: are nearby stars' flux offsets correlated?

Plots all stars from a single TESS sector in sky coordinates (ra/dec),
colored by LOO flux_offset. Smooth color gradients = spatial locality =
the offset is a detector/instrumental effect, not per-star astrophysics.

Also runs a Moran's I spatial autocorrelation test as a numeric answer.

Usage:
  python3 eval/ffi_locality.py [sector] [parquet]
  python3 eval/ffi_locality.py 29
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

# ── Args ──────────────────────────────────────────────────────────────────────
PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup.parquet")
_sec_arg = next((s for s in sys.argv[1:] if s.lstrip("-").isdigit()), None)

df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["ra", "dec", "flux_offset"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]

SECTOR = int(float(_sec_arg)) if _sec_arg else int(df["sector"].value_counts().idxmax())
sec = df[df["sector"] == SECTOR].copy()
print(f"Sector {SECTOR}: {len(sec):,} stars")

# ── Color scale (global across all cameras) ───────────────────────────────────
offsets_all = sec["flux_offset"].values
lo, hi = np.percentile(offsets_all, [1, 99])
extent  = max(abs(lo - 1.0), abs(hi - 1.0), 0.008)
VMIN, VMAX = 1.0 - extent, 1.0 + extent
NORM = TwoSlopeNorm(vcenter=1.0, vmin=VMIN, vmax=VMAX)
CMAP = "RdBu_r"

# ── Helper: gnomonic projection centred on a group ────────────────────────────
def gnomonic(ra_deg, dec_deg):
    ra0  = np.deg2rad(np.median(ra_deg))
    dec0 = np.deg2rad(np.median(dec_deg))
    ra   = np.deg2rad(ra_deg)
    dec  = np.deg2rad(dec_deg)
    cos_c = np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0)
    x = np.rad2deg(np.cos(dec)*np.sin(ra - ra0) / cos_c)
    y = np.rad2deg((np.cos(dec0)*np.sin(dec) - np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0)) / cos_c)
    return x, y

# ── Moran's I (vectorised) ────────────────────────────────────────────────────
def morans_i(x, y, values, k=15):
    k = min(k, len(values) - 1)
    coords = np.column_stack([x, y])
    _, idxs = cKDTree(coords).query(coords, k=k + 1)
    idxs = idxs[:, 1:]
    n, z = len(values), values - values.mean()
    num = (z * z[idxs].sum(axis=1)).sum() / k
    I   = (n / (n * k)) * (num / (z**2).sum())
    return float(I)

# ── Neighbour correlation ──────────────────────────────────────────────────────
def nbr_corr(x, y, values, k=15):
    k = min(k, len(values) - 1)
    coords = np.column_stack([x, y])
    _, idxs = cKDTree(coords).query(coords, k=k + 1)
    nbr_mean = values[idxs[:, 1:]].mean(axis=1)
    return float(np.corrcoef(values, nbr_mean)[0, 1]), nbr_mean

# ── Figure: 2×2 grid, one camera per panel ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor("#0b0f1a")
fig.suptitle(
    f"TESS Sector {SECTOR}  —  {len(sec):,} stars, one panel per camera\n"
    f"Colour = LOO flux offset.  Smooth gradients → spatial locality → instrumental origin.",
    color="#c4d4ee", fontsize=13, y=0.995
)

cam_layout = [(0,0,1),(0,1,2),(1,0,3),(1,1,4)]
all_nbr_r = []

for (ri, ci, cam) in cam_layout:
    ax = axes[ri][ci]
    ax.set_facecolor("#060a12")
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")

    g = sec[sec["cam"] == cam].copy()
    if len(g) < 10:
        ax.text(0.5, 0.5, f"Cam {cam}\nno data",
                ha="center", va="center", transform=ax.transAxes,
                color="#3a5070", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        continue

    x, y = gnomonic(g["ra"].values, g["dec"].values)
    off  = g["flux_offset"].values
    msize = max(2, min(20, 15000 // len(g)))

    sc = ax.scatter(x, y, c=off, cmap=CMAP, norm=NORM,
                    s=msize, alpha=0.8, linewidths=0, rasterized=True)

    I = morans_i(x, y, off)
    r, nbr_mean = nbr_corr(x, y, off)
    all_nbr_r.append(r)
    sigma_raw = off.std()
    sigma_nbr = nbr_mean.std()

    ax.set_title(f"Camera {cam}", color="#c4d4ee", fontsize=11, pad=6)
    ax.set_xlabel("ξ (deg)", color="#3a5070", fontsize=9)
    ax.set_ylabel("η (deg)", color="#3a5070", fontsize=9)
    ax.tick_params(colors="#3a5070", labelsize=8)

    r_color = "#34c580" if r > 0.35 else "#f59e0b" if r > 0.15 else "#ef4444"
    stats = (f"n = {len(g):,}\n"
             f"Moran's I = {I:.3f}\n"
             f"nbr r = {r:.3f}   σ ratio = {sigma_nbr/sigma_raw:.2f}")
    ax.text(0.03, 0.03, stats, transform=ax.transAxes,
            color=r_color, fontsize=9, va="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#0f1c30", ec="#1c2d45", alpha=0.9))

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.012, pad=0.01, shrink=0.55)
cbar.set_label("LOO flux_offset", color="#c4d4ee", fontsize=11)
cbar.ax.yaxis.set_tick_params(color="#3a5070", labelcolor="#7a9cc4", labelsize=9)
cbar.ax.axhline(y=1.0, color="white", lw=1, alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.97])

med_r = float(np.median(all_nbr_r))
print(f"\nMedian neighbour r across cameras: {med_r:.3f}")
if med_r > 0.4:
    print("→ Strong spatial locality — offset is likely detector/instrumental.")
elif med_r > 0.2:
    print("→ Moderate spatial locality — mixed signal.")
else:
    print("→ Weak spatial locality.")

# ── Save ──────────────────────────────────────────────────────────────────────
out = f"eval/ffi_locality_sector{SECTOR}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"Saved → {out}")
