"""
Time-averaged spatial flux-offset map.

For each star, average its Gaia RP (or LOO) flux_offset across a sector range,
then plot by detector position (col, row) per cam × CCD.  This shows persistent
detector-level structure rather than any single sector's snapshot — if someone
found a blob in cam 3 by averaging s1–96, this will reproduce it.

Usage:
  python3 eval/ffi_time_avg.py                        # all sectors, Gaia label
  python3 eval/ffi_time_avg.py 1 30                   # sectors 1–30 inclusive
  python3 eval/ffi_time_avg.py 1 96 --loo             # LOO label instead of Gaia
  python3 eval/ffi_time_avg.py 1 30 training_data_topup.parquet
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator

# ── Args ──────────────────────────────────────────────────────────────────────
PARQUET  = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                "training_data_topup.parquet")
USE_LOO  = "--loo" in sys.argv
int_args = [int(s) for s in sys.argv[1:] if s.lstrip("-").isdigit()]
S_MIN    = int_args[0] if len(int_args) >= 1 else None
S_MAX    = int_args[1] if len(int_args) >= 2 else None

LABEL_COL  = "flux_offset"
LABEL_NAME = "LOO flux_offset" if USE_LOO else "Gaia Rₚ flux_offset"

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)

if USE_LOO:
    label_col = "flux_offset_loo" if "flux_offset_loo" in df.columns else "flux_offset"
else:
    label_col = LABEL_COL
    if "gaiarp" in df.columns:
        df = df[df["gaiarp"].notna()].copy()

df = df.dropna(subset=["col", "row", label_col, "cam", "ccd", "tic_id"])
df = df[(df[label_col] > 0.85) & (df[label_col] < 1.15)]

if S_MIN is not None:
    df = df[df["sector"] >= S_MIN]
if S_MAX is not None:
    df = df[df["sector"] <= S_MAX]

s_lo = int(df["sector"].min())
s_hi = int(df["sector"].max())
range_str = f"s{s_lo}–{s_hi}"
print(f"  {len(df):,} records · sectors {s_lo}–{s_hi} · {df['tic_id'].nunique():,} unique stars")

# ── Time-average per star ──────────────────────────────────────────────────────
# Use mean col/row across sectors (usually nearly identical for 2-min cadence).
# Mean flux_offset across all sectors in the selected range.
agg = df.groupby("tic_id").agg(
    col      = ("col",      "mean"),
    row      = ("row",      "mean"),
    cam      = ("cam",      "first"),
    ccd      = ("ccd",      "first"),
    offset   = (label_col,  "mean"),
    n_sectors= ("sector",   "nunique"),
).reset_index()

# Require at least 3 sectors for a reliable average
agg = agg[agg["n_sectors"] >= 3].copy()
print(f"  {len(agg):,} stars with ≥ 3 sectors after averaging")

# ── Color scale: centered on 1.0 ─────────────────────────────────────────────
lo, hi = np.percentile(agg["offset"], [0.5, 99.5])
extent = max(abs(lo - 1.0), abs(hi - 1.0), 0.005)
VMIN, VMAX = 1.0 - extent, 1.0 + extent
CMAP = "RdBu_r"
NORM = TwoSlopeNorm(vcenter=1.0, vmin=VMIN, vmax=VMAX)

# ── Helper: neighbor spatial correlation ──────────────────────────────────────
def neighbor_corr(col, row, offsets, k=10):
    if len(col) < k + 2:
        return np.nan
    coords = np.column_stack([col, row])
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=k + 1)
    neighbor_mean = offsets[idxs[:, 1:]].mean(axis=1)
    mask = np.isfinite(neighbor_mean) & np.isfinite(offsets)
    if mask.sum() < 5:
        return np.nan
    return float(np.corrcoef(offsets[mask], neighbor_mean[mask])[0, 1])

# ── Helper: binned smooth heatmap ─────────────────────────────────────────────
NBINS = 28

def smooth_heatmap(col, row, offsets, bins=NBINS):
    c_edges = np.linspace(col.min(), col.max(), bins + 1)
    r_edges = np.linspace(row.min(), row.max(), bins + 1)
    Z = np.full((bins, bins), np.nan)
    for ci in range(bins):
        for ri in range(bins):
            mask = ((col >= c_edges[ci]) & (col < c_edges[ci + 1]) &
                    (row >= r_edges[ri]) & (row < r_edges[ri + 1]))
            if mask.sum() >= 2:
                Z[ri, ci] = offsets[mask].mean()
    valid = ~np.isnan(Z)
    if valid.sum() < 4:
        return None, None, None
    yy, xx = np.mgrid[0:bins, 0:bins]
    interp = NearestNDInterpolator(np.column_stack([xx[valid], yy[valid]]), Z[valid])
    Z_filled = interp(xx, yy)
    Z_smooth = gaussian_filter(Z_filled, sigma=1.8)
    return c_edges, r_edges, Z_smooth

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 18))
fig.patch.set_facecolor("#0b0f1a")
fig.suptitle(
    f"TESS  —  Time-averaged {LABEL_NAME}  ({range_str})  per cam × CCD\n"
    f"Each star = mean offset across all its sectors in this range (≥ 3 sectors required)",
    fontsize=13, color="#c4d4ee", y=0.995
)

corr_results = []

for cam_idx, cam in enumerate([1, 2, 3, 4]):
    for ccd_idx, ccd in enumerate([1, 2, 3, 4]):
        ax = axes[cam_idx][ccd_idx]
        ax.set_facecolor("#080c18")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1c2d45")

        g = agg[(agg["cam"] == cam) & (agg["ccd"] == ccd)].copy()

        if len(g) < 5:
            ax.text(0.5, 0.5, f"Cam {cam} CCD {ccd}\nno data",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#3a5070", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        col_arr = g["col"].values
        row_arr = g["row"].values
        off_arr = g["offset"].values

        # Smoothed heatmap background
        n = len(g)
        heatmap_alpha = min(0.70, 0.1 + np.sqrt(n) / 25)
        c_edges, r_edges, Z = smooth_heatmap(col_arr, row_arr, off_arr) if n >= 25 else (None, None, None)
        if Z is not None:
            ax.imshow(
                Z, origin="lower", aspect="auto",
                extent=[c_edges[0], c_edges[-1], r_edges[0], r_edges[-1]],
                cmap=CMAP, norm=NORM, alpha=heatmap_alpha, interpolation="bilinear"
            )

        # Star scatter on top
        pt_size = max(2, min(14, 5000 // max(n, 1)))
        ax.scatter(
            col_arr, row_arr,
            c=off_arr, cmap=CMAP, norm=NORM,
            s=pt_size, alpha=0.85, linewidths=0, rasterized=True
        )

        r = neighbor_corr(col_arr, row_arr, off_arr, k=min(10, len(g) - 2))
        corr_results.append(dict(cam=cam, ccd=ccd, n=n, r=r))

        r_str = f"r={r:.3f}" if not np.isnan(r) else "r=n/a"
        ax.text(0.03, 0.96, f"Cam {cam} / CCD {ccd}",
                transform=ax.transAxes, color="#c4d4ee",
                fontsize=8.5, fontweight="bold", va="top")
        ax.text(0.03, 0.87, f"n={n:,}  σ={off_arr.std():.4f}",
                transform=ax.transAxes, color="#7a9cc4",
                fontsize=7.5, va="top")
        ax.text(0.03, 0.79, f"nbr-corr {r_str}",
                transform=ax.transAxes,
                color="#34c580" if (not np.isnan(r) and r > 0.3) else "#f59e0b",
                fontsize=7.5, va="top")

        ax.tick_params(labelsize=6, colors="#3a5070")
        if cam_idx == 3:
            ax.set_xlabel("col (px)", fontsize=7, color="#3a5070")
        if ccd_idx == 0:
            ax.set_ylabel("row (px)", fontsize=7, color="#3a5070")

# Colorbar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.012, pad=0.01, shrink=0.6)
cbar.set_label(f"Mean {LABEL_NAME}", color="#c4d4ee", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="#3a5070", labelcolor="#7a9cc4", labelsize=8)
cbar.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

# ── Summary ───────────────────────────────────────────────────────────────────
corr_df = pd.DataFrame(corr_results)
print("\nNeighbor spatial correlation (time-averaged offsets):")
print(corr_df.to_string(index=False, float_format="%.3f"))
if corr_df["r"].notna().any():
    med_r = corr_df["r"].median()
    print(f"\n  Median r: {med_r:.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
label_tag = "_loo" if USE_LOO else "_gaia"
out = f"eval/ffi_time_avg_{range_str.replace('–','-')}{label_tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"\nSaved → {out}")
