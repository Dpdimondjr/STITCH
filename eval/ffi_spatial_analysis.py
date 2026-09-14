"""
FFI spatial smoothness analysis.

For a given TESS sector, scatter all stars by (col, row) per cam×CCD,
colored by their LOO flux_offset. If the offset varies smoothly over the
detector, it's instrumental (detector-level systematic), not astrophysical.

Also computes a nearest-neighbor spatial correlation (Moran-like) per CCD
to give a numeric answer to Daniel's question.

Usage:
  python3 eval/ffi_spatial_analysis.py [sector=N] [parquet_path]
  python3 eval/ffi_spatial_analysis.py 26
  python3 eval/ffi_spatial_analysis.py 26 training_data_topup.parquet
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# ── Args ──────────────────────────────────────────────────────────────────────
PARQUET    = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                  "training_data_topup.parquet")
_sector_arg = next((s for s in sys.argv[1:] if s.lstrip("-").isdigit()), None)
USE_GAIA   = "--gaia" in sys.argv
LABEL_COL  = "flux_offset"   # flux_offset = Gaia label post-compute_gaia_labels
LABEL_NAME = "Gaia Rₚ label" if USE_GAIA else "LOO flux_offset"

df = pd.read_parquet(PARQUET)
if USE_GAIA:
    # Keep only rows with a real Gaia label (gaiarp was filled)
    df = df[df["gaiarp"].notna()].copy()
    df = df.dropna(subset=["col", "row", "flux_offset"])
else:
    LABEL_COL = "flux_offset_loo" if "flux_offset_loo" in df.columns else "flux_offset"
    df = df.dropna(subset=["col", "row", LABEL_COL])
df = df[(df[LABEL_COL] > 0.85) & (df[LABEL_COL] < 1.15)]

SECTOR = int(float(_sector_arg)) if _sector_arg else int(df["sector"].value_counts().idxmax())
print(f"Sector: {SECTOR}  (parquet: {PARQUET})")

sec = df[df["sector"] == SECTOR].copy()
print(f"  {len(sec):,} records across {sec['tic_id'].nunique():,} stars after quality filter")

CAMS = sorted(sec["cam"].dropna().unique().astype(int))
CCDS = sorted(sec["ccd"].dropna().unique().astype(int))

# ── Color scale ───────────────────────────────────────────────────────────────
# Clip to central 99th percentile range, centered on 1.0
lo, hi = np.percentile(sec[LABEL_COL], [0.5, 99.5])
extent = max(abs(lo - 1.0), abs(hi - 1.0), 0.01)
VMIN, VMAX = 1.0 - extent, 1.0 + extent
CMAP = "RdBu_r"
NORM = TwoSlopeNorm(vcenter=1.0, vmin=VMIN, vmax=VMAX)

# ── Helper: neighbor spatial correlation (Moran-like) ─────────────────────────
def neighbor_corr(col, row, offsets, k=10):
    """Pearson r between each star's offset and the mean of its k nearest neighbors."""
    if len(col) < k + 2:
        return np.nan
    coords = np.column_stack([col, row])
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)   # +1 because first result is self
    neighbor_mean = offsets[idxs[:, 1:]].mean(axis=1)
    mask = np.isfinite(neighbor_mean) & np.isfinite(offsets)
    if mask.sum() < 5:
        return np.nan
    return float(np.corrcoef(offsets[mask], neighbor_mean[mask])[0, 1])

# ── Helper: binned smooth heatmap ─────────────────────────────────────────────
NBINS = 24

def smooth_heatmap(col, row, offsets, bins=NBINS):
    """Returns (X, Y, Z_smooth) for imshow."""
    c_edges = np.linspace(col.min(), col.max(), bins + 1)
    r_edges = np.linspace(row.min(), row.max(), bins + 1)
    Z = np.full((bins, bins), np.nan)
    for ci in range(bins):
        for ri in range(bins):
            mask = ((col >= c_edges[ci]) & (col < c_edges[ci + 1]) &
                    (row >= r_edges[ri]) & (row < r_edges[ri + 1]))
            if mask.sum() >= 2:
                Z[ri, ci] = offsets[mask].mean()
    # Interpolate NaN cells with nearest neighbor before smoothing
    from scipy.interpolate import NearestNDInterpolator
    valid = ~np.isnan(Z)
    if valid.sum() < 4:
        return None, None, None
    yy, xx = np.mgrid[0:bins, 0:bins]
    interp = NearestNDInterpolator(np.column_stack([xx[valid], yy[valid]]), Z[valid])
    Z_filled = interp(xx, yy)
    Z_smooth = gaussian_filter(Z_filled, sigma=1.5)
    return c_edges, r_edges, Z_smooth

# ── Figure: 4×4 grid (cam rows, ccd cols) ────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(20, 18))
fig.patch.set_facecolor("#0b0f1a")
fig.suptitle(
    f"TESS Sector {SECTOR}  —  Spatial distribution of {LABEL_NAME} per cam × CCD\n"
    f"Smooth gradient = instrumental; random speckle = astrophysical noise",
    fontsize=13, color="#c4d4ee", y=0.995
)

corr_results = []

for cam_idx, cam in enumerate([1, 2, 3, 4]):
    for ccd_idx, ccd in enumerate([1, 2, 3, 4]):
        ax = axes[cam_idx][ccd_idx]
        ax.set_facecolor("#080c18")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1c2d45")

        g = sec[(sec["cam"] == cam) & (sec["ccd"] == ccd)].copy()

        if len(g) < 5:
            ax.text(0.5, 0.5, f"Cam {cam} CCD {ccd}\nno data",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#3a5070", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        col_arr = g["col"].values
        row_arr = g["row"].values
        off_arr = g[LABEL_COL].values

        # Smoothed heatmap background — opacity scales with √n so sparse panels
        # don't show a confident-looking gradient over just a few stars
        n = len(g)
        heatmap_alpha = min(0.65, 0.08 + np.sqrt(n) / 30)
        c_edges, r_edges, Z = smooth_heatmap(col_arr, row_arr, off_arr) if n >= 25 else (None, None, None)
        if Z is not None:
            ax.imshow(
                Z, origin="lower", aspect="auto",
                extent=[c_edges[0], c_edges[-1], r_edges[0], r_edges[-1]],
                cmap=CMAP, norm=NORM, alpha=heatmap_alpha, interpolation="bilinear"
            )
        elif n < 25:
            ax.text(0.5, 0.42, "too sparse\nfor heatmap",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#3a5070", fontsize=8, style="italic")

        # Raw star scatter on top
        pt_size = max(2, min(12, 4000 // max(n, 1)))
        sc = ax.scatter(
            col_arr, row_arr,
            c=off_arr, cmap=CMAP, norm=NORM,
            s=pt_size, alpha=0.85, linewidths=0, rasterized=True
        )

        # Neighbor correlation
        r = neighbor_corr(col_arr, row_arr, off_arr, k=min(10, len(g) - 2))
        corr_results.append(dict(cam=cam, ccd=ccd, n=len(g), r=r))

        # Stats overlay
        r_str = f"r={r:.3f}" if not np.isnan(r) else "r=n/a"
        offset_std = off_arr.std()
        ax.text(0.03, 0.96, f"Cam {cam} / CCD {ccd}",
                transform=ax.transAxes, color="#c4d4ee",
                fontsize=8.5, fontweight="bold", va="top")
        ax.text(0.03, 0.87, f"n={len(g):,}  σ={offset_std:.4f}",
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
            ax.set_ylabel(f"row (px)", fontsize=7, color="#3a5070")

# Shared colorbar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                    fraction=0.012, pad=0.01, shrink=0.6)
cbar.set_label(LABEL_NAME, color="#c4d4ee", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="#3a5070", labelcolor="#7a9cc4", labelsize=8)
cbar.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

# ── Summary table ─────────────────────────────────────────────────────────────
corr_df = pd.DataFrame(corr_results)
print("\nNeighbor spatial correlation (Pearson r between a star's offset and mean of 10 nearest neighbors):")
print("  r > 0.5  → strong spatial structure (instrumental)\n  r ~ 0   → no spatial correlation (random)\n")
print(corr_df.to_string(index=False, float_format="%.3f"))

if corr_df["r"].notna().any():
    med_r = corr_df["r"].median()
    print(f"\n  Median r across all CCD panels: {med_r:.3f}")
    if med_r > 0.5:
        print("  → Strong spatial structure: offset is likely instrumental / detector-level.")
    elif med_r > 0.2:
        print("  → Moderate spatial structure: mixed signal, may be partially instrumental.")
    else:
        print("  → Weak spatial structure: offset may be dominated by noise or astrophysical variation.")

# ── Save ──────────────────────────────────────────────────────────────────────
_suffix = "_gaia" if USE_GAIA else ""
out = f"eval/ffi_spatial_sector{SECTOR}{_suffix}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"\nSaved → {out}")
