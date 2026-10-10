"""
Roland-style spatial flux-offset grid.

Reproduces the 8×2 all-CCD focal-plane grid in Roland's SPOC images:
  - Viridis colormap
  - 80×80 bins per CCD, Gaussian-smoothed
  - Focal-plane panel ordering (matches Roland's layout)
  - Time-averaged per star across all sectors

Usage:
  python3 eval/ffi_roland_style.py                          # LOO label, all sectors
  python3 eval/ffi_roland_style.py --gaia                   # Gaia label
  python3 eval/ffi_roland_style.py 1 60 --loo               # sectors 1-60
  python3 eval/ffi_roland_style.py --vlim=0.03              # fixed ±3% scale
  python3 eval/ffi_roland_style.py training_data_v4.parquet
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d

# ── Args ──────────────────────────────────────────────────────────────────────
PARQUET   = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                 "training_data_topup.parquet")
USE_GAIA  = "--gaia" in sys.argv
USE_LOO   = "--loo" in sys.argv or not USE_GAIA
int_args  = [int(s) for s in sys.argv[1:] if s.lstrip("-").isdigit()]
S_MIN     = int_args[0] if len(int_args) >= 1 else None
S_MAX     = int_args[1] if len(int_args) >= 2 else None
_vlim_arg = next((s for s in sys.argv[1:] if s.startswith("--vlim=")), None)
FIXED_VLIM = float(_vlim_arg.split("=")[1]) if _vlim_arg else None

NBINS  = 80
SIGMA  = 0.8   # minimal smoothing — let real star-density structure show

# ── Focal-plane panel ordering (matches Roland's image) ───────────────────────
# Top row / bottom row, 8 panels each
PANEL_ORDER = [
    # top row
    (1, 4), (1, 1), (2, 4), (2, 1), (3, 2), (3, 3), (4, 2), (4, 3),
    # bottom row
    (1, 3), (1, 2), (2, 3), (2, 2), (3, 1), (3, 4), (4, 1), (4, 4),
]

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)

if USE_GAIA and "gaiarp" in df.columns:
    label_col = "flux_offset"
    df = df[df["gaiarp"].notna()].copy()
    label_name = "Gaia Rₚ flux_offset"
else:
    label_col = "flux_offset_loo" if "flux_offset_loo" in df.columns else "flux_offset"
    label_name = "LOO flux_offset"

df = df.dropna(subset=["col", "row", label_col, "cam", "ccd", "tic_id"])
df = df[(df[label_col] > 0.85) & (df[label_col] < 1.15)]

if S_MIN is not None:
    df = df[df["sector"] >= S_MIN]
if S_MAX is not None:
    df = df[df["sector"] <= S_MAX]

s_lo = int(df["sector"].min())
s_hi = int(df["sector"].max())
print(f"  {len(df):,} records · sectors {s_lo}–{s_hi} · {df['tic_id'].nunique():,} unique stars")

# ── Use all (star, sector) observations for dense bin coverage ────────────────
agg = df.rename(columns={label_col: "offset"})
n_total = agg["tic_id"].nunique()
print(f"  {len(agg):,} observations · {n_total:,} unique stars")

# ── Colour scale ──────────────────────────────────────────────────────────────
# One-sided auto scale (not forced symmetric) — matches old flux_offset_grid_hires style
if FIXED_VLIM is not None:
    vlo = 1.0 - FIXED_VLIM
    vhi = 1.0 + FIXED_VLIM
    print(f"  Fixed scale: {vlo:.3f}–{vhi:.3f}  (--vlim={FIXED_VLIM})")
else:
    vlo, vhi = np.nanpercentile(agg["offset"], [5, 95])
    print(f"  Auto scale: {vlo:.4f}–{vhi:.4f}  (p5–p95; use --vlim=N for fixed)")

CMAP = "viridis"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 8, figsize=(24, 6.5))
fig.patch.set_facecolor("#0a0d14")

range_str = f"s{s_lo}–{s_hi}"
fig.suptitle(
    f"STITCH training target — {label_name}  ·  all 16 CCDs\n"
    f"{NBINS}×{NBINS} bins  ·  σ={SIGMA}  ·  {n_total:,} stars  ({range_str})",
    fontsize=11, color="#c4d4ee", y=1.01
)

for idx, (cam, ccd) in enumerate(PANEL_ORDER):
    row_i = idx // 8
    col_i = idx %  8
    ax = axes[row_i][col_i]
    ax.set_facecolor("#05070d")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a2535")

    g = agg[(agg["cam"] == cam) & (agg["ccd"] == ccd)]

    if len(g) < 10:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#3a5070", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{cam}.{ccd}", color="#3a5070", fontsize=9, pad=3)
        continue

    col_arr = g["col"].values
    row_arr = g["row"].values
    off_arr = g["offset"].values

    c0, c1 = np.percentile(col_arr, [1, 99])
    r0, r1 = np.percentile(row_arr, [1, 99])

    stat, c_edges, r_edges, _ = binned_statistic_2d(
        col_arr, row_arr, off_arr,
        statistic="mean",
        bins=NBINS,
        range=[[c0, c1], [r0, r1]],
    )
    valid = np.isfinite(stat)
    if valid.sum() < 4:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{cam}.{ccd}", color="#3a5070", fontsize=9, pad=3)
        continue
    # Fill empty bins with global mean so Gaussian smoothing creates
    # natural transitions — occupied regions float on an average background
    stat[~valid] = np.nanmean(stat)
    stat_t = gaussian_filter(stat.T, sigma=SIGMA)

    ax.imshow(
        stat_t,
        origin="lower",
        aspect="auto",
        extent=[c_edges[0], c_edges[-1], r_edges[0], r_edges[-1]],
        cmap=CMAP,
        vmin=vlo, vmax=vhi,
        interpolation="bilinear",
    )

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{cam}.{ccd}", color="white", fontsize=9.5,
                 fontweight="bold", pad=3)

# ── Colorbar ──────────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=CMAP)
sm.set_clim(vlo, vhi)
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.008, pad=0.01, shrink=0.85)
cbar.set_label(label_name, color="#c4d4ee", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#3a5070", labelcolor="#7a9cc4", labelsize=7)

# ── Save ──────────────────────────────────────────────────────────────────────
label_tag  = "_gaia" if USE_GAIA else "_loo"
sector_tag = f"_{range_str.replace('–','-')}" if S_MIN or S_MAX else ""
out = f"eval/ffi_roland_style{sector_tag}{label_tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0a0d14")
print(f"\nSaved → {out}")
