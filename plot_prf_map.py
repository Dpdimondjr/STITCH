"""
Plot STITCH's learned PRF offset map across the TESS focal plane.

For each cam/CCD, sweep a grid of (col, row) positions and show the
predicted flux_offset as a 2D contour map. All other features are held
at their training-set medians. This reveals the spatial structure of the
PRF-induced systematics that the model has learned.
"""

import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

# ── Load model ────────────────────────────────────────────────────────────────

ckpt = torch.load("stitch_nsf.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(**cfg)
flow.load_state_dict(ckpt["model_state"])
flow.eval()

means      = ckpt["means"]
stds       = ckpt["stds"]
y_mean     = ckpt["y_mean"]
y_std      = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS   = ckpt["cam_cols"]
CCD_COLS   = ckpt["ccd_cols"]

K = 0.0  # no shrinkage — show raw model predictions at full amplitude

# ── Load training data for median feature values ───────────────────────────────

df = pd.read_parquet("training_data.parquet")
df = df[(df.flux_offset > 0.85) & (df.flux_offset < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
for col in CONTINUOUS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

feature_medians = df[CONTINUOUS].median()

# ── Grid resolution ───────────────────────────────────────────────────────────

GRID = 60          # grid points per axis
COL_RANGE = (45, 2048)
ROW_RANGE = (0, 2048)

cols_grid = np.linspace(*COL_RANGE, GRID)
rows_grid = np.linspace(*ROW_RANGE, GRID)
CC, RR = np.meshgrid(cols_grid, rows_grid)  # (GRID, GRID)

def predict_grid(cam, ccd):
    n = GRID * GRID
    rows_df = pd.DataFrame({col: [feature_medians[col]] * n for col in CONTINUOUS})
    rows_df["col"] = CC.ravel()
    rows_df["row"] = RR.ravel()
    rows_df["delta_sub_col"] = 0.5
    rows_df["delta_sub_row"] = 0.5

    cont   = (rows_df[CONTINUOUS] - means) / stds
    cam_oh = pd.DataFrame(0, index=range(n), columns=CAM_COLS)
    ccd_oh = pd.DataFrame(0, index=range(n), columns=CCD_COLS)
    if f"cam_{cam}" in CAM_COLS: cam_oh[f"cam_{cam}"] = 1
    if f"ccd_{ccd}" in CCD_COLS: ccd_oh[f"ccd_{ccd}"] = 1

    C = pd.concat([cont.reset_index(drop=True),
                   cam_oh.reset_index(drop=True),
                   ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

    with torch.no_grad():
        s = flow(torch.tensor(C)).sample((300,)).squeeze(-1)
        pred_z     = s.mean(0).numpy()
        pred_z_std = s.std(0).numpy()

    pred_raw = pred_z * y_std + y_mean
    pred_std = pred_z_std * y_std
    w        = 1.0 / (1.0 + K * pred_std)
    pred     = 1.0 * (1 - w) + pred_raw * w

    return pred.reshape(GRID, GRID), pred_std.reshape(GRID, GRID)

# ── Figure: 4 cameras × 4 CCDs ───────────────────────────────────────────────

CAMS = [1, 2, 3, 4]
CCDS = [1, 2, 3, 4]

# Two figures: predicted offset and uncertainty
for mode in ("offset", "uncertainty"):
    fig, axes = plt.subplots(4, 4, figsize=(14, 13),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.08, "wspace": 0.08})

    # Collect all values first to set a shared colour scale
    grids = {}
    for cam in CAMS:
        for ccd in CCDS:
            off_grid, std_grid = predict_grid(cam, ccd)
            grids[(cam, ccd)] = (off_grid, std_grid)
            print(f"  Cam{cam}/CCD{ccd}: offset {off_grid.min():.4f}–{off_grid.max():.4f}  "
                  f"std {std_grid.mean():.4f}")

    if mode == "offset":
        all_vals = np.concatenate([g[0].ravel() for g in grids.values()])
        vmax = max(abs(all_vals - 1.0).max(), 0.005)
        norm = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - vmax, vmax=1.0 + vmax)
        cmap = "RdBu_r"
        cbar_label = "Predicted flux offset"
        title = "STITCH — Learned PRF Offset Map"
    else:
        all_stds = np.concatenate([g[1].ravel() for g in grids.values()])
        norm = plt.Normalize(vmin=0, vmax=np.percentile(all_stds, 97))
        cmap = "YlOrRd"
        cbar_label = "Posterior std (flux offset units)"
        title = "STITCH — Model Uncertainty Map"

    for i, cam in enumerate(CAMS):
        for j, ccd in enumerate(CCDS):
            ax = axes[i][j]
            off_grid, std_grid = grids[(cam, ccd)]
            data = off_grid if mode == "offset" else std_grid

            im = ax.contourf(CC, RR, data, levels=30, cmap=cmap, norm=norm)
            ax.contour(CC, RR, data, levels=8, colors="k", linewidths=0.3, alpha=0.4)

            # Training data density overlay
            sub = df[(df.cam == cam) & (df.ccd == ccd)]
            if len(sub) > 0:
                ax.scatter(sub["col"], sub["row"], s=0.3, c="white", alpha=0.15,
                           rasterized=True)

            # Label
            ax.text(0.04, 0.96, f"Cam{cam}/CCD{ccd}", transform=ax.transAxes,
                    fontsize=7.5, va="top", color="white",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4))

            if i == 3: ax.set_xlabel("Column", fontsize=8)
            if j == 0: ax.set_ylabel("Row", fontsize=8)
            ax.tick_params(labelsize=6)

    # Shared colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    if mode == "offset":
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    fig.suptitle(title, fontsize=13, y=1.005)

    outfile = f"stitch_prf_map_{mode}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved → {outfile}")
    plt.close()
