"""
Test whether the CVZ systematic is a static CCD response or time-variable.

If static: plot all sectors in detector (col, row) space → one consistent map.
If time-variable: the same (col, row) cell has different offsets across sectors.

Produces:
  ffi_cvz_detector_frame.png
    Panel A: mean flux_offset in detector (col,row) space — the "detector flat"
    Panel B: std of flux_offset across sectors per detector cell — time variability
    Panel C: coefficient of variation (std/mean) — relative time variability
    Panel D: mean vs std per cell — separates systematic from noise

Usage:
  python3 eval/ffi_cvz_detector_frame.py [parquet]
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binned_statistic_2d

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
print(f"Parquet: {PARQUET}")

df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]

# All cam 4 stars with ≥10 sectors — same population as CVZ analysis
cam4  = df[df["cam"] == 4].copy()
n_sec = cam4.groupby("tic_id")["sector"].nunique()
cam4  = cam4[cam4["tic_id"].isin(n_sec[n_sec >= 10].index)].copy()
print(f"Cam 4 (≥10 sectors): {cam4['tic_id'].nunique():,} stars, "
      f"{cam4['sector'].nunique()} sectors, {len(cam4):,} records")

# ── Per-detector-cell statistics across all sectors ───────────────────────────
NBINS = 24
col_edges = np.linspace(cam4["col"].quantile(0.01), cam4["col"].quantile(0.99), NBINS+1)
row_edges = np.linspace(cam4["row"].quantile(0.01), cam4["row"].quantile(0.99), NBINS+1)

mean_map, _, _, _ = binned_statistic_2d(
    cam4["col"], cam4["row"], cam4["flux_offset"],
    statistic="mean", bins=[col_edges, row_edges])
std_map,  _, _, _ = binned_statistic_2d(
    cam4["col"], cam4["row"], cam4["flux_offset"],
    statistic="std",  bins=[col_edges, row_edges])
cnt_map,  _, _, _ = binned_statistic_2d(
    cam4["col"], cam4["row"], cam4["flux_offset"],
    statistic="count", bins=[col_edges, row_edges])

# Per-cell cross-sector std: for each star, get its sector-to-sector std,
# then bin those stds by detector position
star_stats = (cam4.groupby("tic_id")
              .agg(col_med=("col","median"), row_med=("row","median"),
                   loo_mean=("flux_offset","mean"),
                   loo_std=("flux_offset","std"),
                   n_sec=("sector","nunique")))
star_stats = star_stats[star_stats["n_sec"] >= 10].dropna()

xsec_std_map, _, _, _ = binned_statistic_2d(
    star_stats["col_med"], star_stats["row_med"], star_stats["loo_std"],
    statistic="median", bins=[col_edges, row_edges])
xsec_mean_map, _, _, _ = binned_statistic_2d(
    star_stats["col_med"], star_stats["row_med"], star_stats["loo_mean"],
    statistic="median", bins=[col_edges, row_edges])

# mask sparse cells
sparse = cnt_map < 5
mean_map[sparse] = np.nan
std_map[sparse]  = np.nan
xsec_std_map[sparse] = np.nan

# fraction of LOO std that is cross-sector (temporal) vs all-time
# high ratio → mostly time-variable; low → mostly sector-averaged
spatial_bias = np.abs(mean_map - 1.0)   # how far from 1 on average

print(f"\nDetector map statistics (cam 4, {NBINS}×{NBINS} grid):")
print(f"  Mean offset range:   [{np.nanmin(mean_map):.4f}, {np.nanmax(mean_map):.4f}]")
print(f"  Spatial std of mean: {np.nanstd(mean_map):.5f}  "
      f"(spatial variation of the mean detector response)")
print(f"  Median cross-sec std per star: {np.nanmedian(xsec_std_map):.5f}  "
      f"(temporal variation at fixed detector position)")
print(f"  Ratio (temporal/total): "
      f"{np.nanmedian(xsec_std_map) / np.nanstd(mean_map):.2f}x")

# ── Figure ─────────────────────────────────────────────────────────────────────
BG, AX, TEXT, MUTE = "#0b0f1a", "#060a12", "#c4d4ee", "#7a9cc4"
CMAP_DIV = "RdBu_r"
CMAP_SEQ = "plasma"

ext_mean = max(abs(np.nanmin(mean_map)-1), abs(np.nanmax(mean_map)-1), 0.008)
NORM_MEAN = TwoSlopeNorm(vcenter=1.0, vmin=1-ext_mean, vmax=1+ext_mean)

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
fig.patch.set_facecolor(BG)

extent = [col_edges[0], col_edges[-1], row_edges[0], row_edges[-1]]

def style(ax, title):
    ax.set_facecolor(AX)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=9.5, pad=5)
    ax.set_xlabel("col (px)", color=MUTE, fontsize=8)
    ax.set_ylabel("row (px)", color=MUTE, fontsize=8)

# Panel A: mean detector map
ax = axes[0]
style(ax, "A. Mean LOO in detector frame\n(all sectors stacked)")
im = ax.imshow(mean_map.T, origin="lower", extent=extent, aspect="auto",
               cmap=CMAP_DIV, norm=NORM_MEAN, interpolation="bilinear")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("mean flux_offset", color=TEXT, fontsize=7)
cbar.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)
cbar.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)
ax.text(0.03, 0.97, "Static CCD response", transform=ax.transAxes,
        color=MUTE, fontsize=7.5, va="top")

# Panel B: cross-sector std per star in detector frame
ax = axes[1]
style(ax, "B. Sector-to-sector σ per star\n(temporal variability at fixed detector pos)")
vmax_std = np.nanpercentile(xsec_std_map, 97)
im2 = ax.imshow(xsec_std_map.T, origin="lower", extent=extent, aspect="auto",
                cmap=CMAP_SEQ, vmin=0, vmax=vmax_std, interpolation="bilinear")
cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cbar2.set_label("median σ(LOO) across sectors", color=TEXT, fontsize=7)
cbar2.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)
ax.text(0.03, 0.97, "Time-variable component", transform=ax.transAxes,
        color=MUTE, fontsize=7.5, va="top")

# Panel C: static component magnitude vs temporal component
ax = axes[2]
style(ax, "C. |Mean offset − 1| in detector frame\n(amplitude of static systematic)")
im3 = ax.imshow(spatial_bias.T, origin="lower", extent=extent, aspect="auto",
                cmap="YlOrRd", vmin=0, vmax=np.nanpercentile(spatial_bias, 97),
                interpolation="bilinear")
cbar3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
cbar3.set_label("|mean − 1|", color=TEXT, fontsize=7)
cbar3.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)
ax.text(0.03, 0.97, "Static amplitude", transform=ax.transAxes,
        color=MUTE, fontsize=7.5, va="top")

# Panel D: scatter — static bias vs temporal std, one dot per star
ax = axes[3]
ax.set_facecolor(AX)
for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")
ax.tick_params(colors=MUTE, labelsize=8)
ax.set_title("D. Static bias vs temporal variability\n(one point per star)",
             color=TEXT, fontsize=9.5, pad=5)
static_per_star = np.abs(star_stats["loo_mean"] - 1.0)
temporal_per_star = star_stats["loo_std"]
sc = ax.scatter(static_per_star * 100, temporal_per_star * 100,
                c=np.log10(star_stats["n_sec"].clip(4, 60)),
                cmap="viridis", s=1.5, alpha=0.35, rasterized=True, linewidths=0)
# Diagonal: if temporal = static, points fall on this line
lim = max(np.percentile(static_per_star*100, 99),
          np.percentile(temporal_per_star*100, 99))
ax.plot([0, lim], [0, lim], color="white", lw=0.8, alpha=0.3, ls="--")
ax.set_xlabel("|mean LOO − 1| × 100 (static bias %)", color=MUTE, fontsize=8)
ax.set_ylabel("σ(LOO across sectors) × 100 (temporal %)", color=MUTE, fontsize=8)
cbar4 = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar4.set_label("log₁₀(N sectors)", color=TEXT, fontsize=7)
cbar4.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)

# Median lines
ax.axvline(static_per_star.median()*100, color="#f59e0b", lw=1, ls=":", alpha=0.7)
ax.axhline(temporal_per_star.median()*100, color="#3b8ef0", lw=1, ls=":", alpha=0.7)
ax.text(0.97, 0.97,
        f"Median static:   {static_per_star.median()*100:.3f}%\n"
        f"Median temporal: {temporal_per_star.median()*100:.3f}%",
        transform=ax.transAxes, ha="right", va="top",
        color=TEXT, fontsize=7.5, family="monospace")

spatial_std = np.nanstd(mean_map)
temporal_med = np.nanmedian(xsec_std_map)
fig.suptitle(
    f"Is the CVZ systematic a static detector response or time-variable?\n"
    f"Cam 4 in detector (col, row) frame — "
    f"spatial σ of mean = {spatial_std*100:.3f}%   "
    f"median temporal σ = {temporal_med*100:.3f}%   "
    f"ratio = {temporal_med/spatial_std:.1f}×",
    color=TEXT, fontsize=10, y=1.02)

plt.tight_layout()
out = "eval/ffi_cvz_detector_frame.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
