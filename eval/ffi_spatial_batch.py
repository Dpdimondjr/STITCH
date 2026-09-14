"""
Batch-generate spatial heatmap PNGs for sectors 1-30 (or specified range),
both with and without the smooth heatmap background. Loads parquet once.

Usage:
  python3 eval/ffi_spatial_batch.py [s_lo] [s_hi]
  python3 eval/ffi_spatial_batch.py 1 30
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator

S_LO = int(sys.argv[1]) if len(sys.argv) > 1 else 1
S_HI = int(sys.argv[2]) if len(sys.argv) > 2 else 30
PARQUET = "training_data_v3.parquet"
OUTDIR  = "eval"

LABEL_COL  = "flux_offset"
LABEL_NAME = "Gaia Rₚ label"
CMAP = "RdBu_r"
NBINS = 24

print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df[df["gaiarp"].notna()].copy()
df = df.dropna(subset=["col", "row", LABEL_COL])
df = df[(df[LABEL_COL] > 0.85) & (df[LABEL_COL] < 1.15)]
print(f"  {len(df):,} rows after filter")

sectors = [s for s in range(S_LO, S_HI + 1) if s in df["sector"].unique()]
print(f"  Sectors with data: {sectors}")


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
    Z_smooth = gaussian_filter(Z_filled, sigma=1.5)
    return c_edges, r_edges, Z_smooth


def render_sector(sec_df, sector, show_heatmap, norm):
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))
    fig.patch.set_facecolor("#0b0f1a")
    mode_str = "with heatmap" if show_heatmap else "scatter only"
    fig.suptitle(
        f"TESS Sector {sector}  —  Spatial distribution of {LABEL_NAME} per cam × CCD\n"
        f"Smooth gradient = instrumental; random speckle = astrophysical noise  [{mode_str}]",
        fontsize=13, color="#c4d4ee", y=0.995
    )

    for cam_idx, cam in enumerate([1, 2, 3, 4]):
        for ccd_idx, ccd in enumerate([1, 2, 3, 4]):
            ax = axes[cam_idx][ccd_idx]
            ax.set_facecolor("#080c18")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1c2d45")

            g = sec_df[(sec_df["cam"] == cam) & (sec_df["ccd"] == ccd)]
            if len(g) < 5:
                ax.text(0.5, 0.5, f"Cam {cam} CCD {ccd}\nno data",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#3a5070", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            col_arr = g["col"].values
            row_arr = g["row"].values
            off_arr = g[LABEL_COL].values
            n = len(g)

            if show_heatmap:
                heatmap_alpha = min(0.65, 0.08 + np.sqrt(n) / 30)
                c_edges, r_edges, Z = smooth_heatmap(col_arr, row_arr, off_arr) if n >= 25 else (None, None, None)
                if Z is not None:
                    ax.imshow(Z, origin="lower", aspect="auto",
                              extent=[c_edges[0], c_edges[-1], r_edges[0], r_edges[-1]],
                              cmap=CMAP, norm=norm, alpha=heatmap_alpha,
                              interpolation="bilinear")
                elif n < 25:
                    ax.text(0.5, 0.42, "too sparse\nfor heatmap",
                            ha="center", va="center", transform=ax.transAxes,
                            color="#3a5070", fontsize=8, style="italic")

            pt_size = max(2, min(12, 4000 // max(n, 1)))
            ax.scatter(col_arr, row_arr, c=off_arr, cmap=CMAP, norm=norm,
                       s=pt_size, alpha=0.85, linewidths=0, rasterized=True)

            r = neighbor_corr(col_arr, row_arr, off_arr, k=min(10, n - 2))
            r_str = f"r={r:.3f}" if not np.isnan(r) else "r=n/a"
            ax.text(0.03, 0.96, f"Cam {cam} / CCD {ccd}",
                    transform=ax.transAxes, color="#c4d4ee",
                    fontsize=8.5, fontweight="bold", va="top")
            ax.text(0.03, 0.87, f"n={n:,}  σ={off_arr.std():.4f}",
                    transform=ax.transAxes, color="#7a9cc4", fontsize=7.5, va="top")
            ax.text(0.03, 0.79, f"nbr-corr {r_str}",
                    transform=ax.transAxes,
                    color="#34c580" if (not np.isnan(r) and r > 0.3) else "#f59e0b",
                    fontsize=7.5, va="top")

            ax.tick_params(labelsize=6, colors="#3a5070")
            if cam_idx == 3:
                ax.set_xlabel("col (px)", fontsize=7, color="#3a5070")
            if ccd_idx == 0:
                ax.set_ylabel("row (px)", fontsize=7, color="#3a5070")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.012, pad=0.01, shrink=0.6)
    cbar.set_label(LABEL_NAME, color="#c4d4ee", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#3a5070", labelcolor="#7a9cc4", labelsize=8)
    cbar.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)
    return fig


# ── Compute shared color scale across all sectors ─────────────────────────────
lo, hi = np.percentile(df[LABEL_COL], [0.5, 99.5])
extent = max(abs(lo - 1.0), abs(hi - 1.0), 0.01)
NORM = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - extent, vmax=1.0 + extent)

# ── Render ────────────────────────────────────────────────────────────────────
for sector in sectors:
    sec_df = df[df["sector"] == sector]
    n = len(sec_df)
    print(f"Sector {sector:2d}: {n:,} stars", flush=True)

    for show_heatmap in [True, False]:
        suffix = "_gaia" if show_heatmap else "_gaia_raw"
        out = f"{OUTDIR}/ffi_spatial_sector{sector}{suffix}.png"
        if os.path.exists(out):
            print(f"  {'heat' if show_heatmap else 'raw '} → exists, skip", flush=True)
            continue
        fig = render_sector(sec_df, sector, show_heatmap, NORM)
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
        plt.close(fig)
        print(f"  {'heat' if show_heatmap else 'raw '} → {out}", flush=True)

print("Done.")
