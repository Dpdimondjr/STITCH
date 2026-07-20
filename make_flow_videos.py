"""
Animated focal-plane offset maps, sweeping one feature at a time.

Produces three MP4s:
  stitch_flow_sector.gif    — sector 1 → 83 (PRF pattern over mission lifetime)
  stitch_flow_subpixel.gif  — delta_sub_col 0 → 1 (intra-pixel sensitivity)
  stitch_flow_tmag.gif      — Tmag 6 → 13 (brightness dependence)

Each frame is a 4×4 grid of cam/CCD offset maps. All frames per cam/CCD are
batched into a single flow call (vectorised over frames × grid points).
"""

import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker

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

# ── Load training medians ─────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
df = df[(df.flux_offset > 0.85) & (df.flux_offset < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
for col in CONTINUOUS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
feature_medians = df[CONTINUOUS].median()

# ── Grid config ───────────────────────────────────────────────────────────────

GRID = 40          # grid points per axis (lower than PRF map for speed)
N_SAMPLES = 150    # flow samples per prediction
K = 0.0            # no shrinkage — show raw model amplitude

COL_RANGE = (45, 2048)
ROW_RANGE = (0, 2048)
cols_grid = np.linspace(*COL_RANGE, GRID)
rows_grid = np.linspace(*ROW_RANGE, GRID)
CC, RR = np.meshgrid(cols_grid, rows_grid)

CAMS = [1, 2, 3, 4]
CCDS = [1, 2, 3, 4]


def predict_all_frames(cam, ccd, sweep_col, sweep_vals):
    """
    Predict flux_offset for all (frame, grid_point) pairs in one flow call.
    Returns array of shape (n_frames, GRID, GRID).
    """
    n_frames = len(sweep_vals)
    n_pts    = GRID * GRID

    # Base context repeated for all frames × grid points
    base = pd.DataFrame({col: [float(feature_medians[col])] * (n_frames * n_pts)
                         for col in CONTINUOUS})

    # Tile grid positions across frames
    base["col"] = np.tile(CC.ravel(), n_frames)
    base["row"] = np.tile(RR.ravel(), n_frames)
    base["delta_sub_col"] = 0.5
    base["delta_sub_row"] = 0.5

    # Sweep the target feature
    base[sweep_col] = np.repeat(sweep_vals, n_pts)

    # Normalise
    cont   = (base[CONTINUOUS] - means) / stds
    cam_oh = pd.DataFrame(0, index=range(len(base)), columns=CAM_COLS)
    ccd_oh = pd.DataFrame(0, index=range(len(base)), columns=CCD_COLS)
    if f"cam_{cam}" in CAM_COLS: cam_oh[f"cam_{cam}"] = 1
    if f"ccd_{ccd}" in CCD_COLS: ccd_oh[f"ccd_{ccd}"] = 1

    C = pd.concat([cont.reset_index(drop=True),
                   cam_oh.reset_index(drop=True),
                   ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

    with torch.no_grad():
        s       = flow(torch.tensor(C)).sample((N_SAMPLES,)).squeeze(-1)
        pred_z     = s.mean(0).numpy()
        pred_z_std = s.std(0).numpy()

    pred_raw = pred_z * y_std + y_mean
    pred_std = pred_z_std * y_std
    w        = 1.0 / (1.0 + K * pred_std)
    pred     = 1.0 * (1 - w) + pred_raw * w

    return pred.reshape(n_frames, GRID, GRID)


def make_video(sweep_col, sweep_vals, sweep_labels, out_file, title_prefix, fps=10):
    print(f"\nBuilding {out_file}  ({len(sweep_vals)} frames, all cam/CCDs)...")

    # Precompute all frames for all 16 cam/CCDs
    all_grids = {}
    for cam in CAMS:
        for ccd in CCDS:
            print(f"  Cam{cam}/CCD{ccd}...", end=" ", flush=True)
            all_grids[(cam, ccd)] = predict_all_frames(cam, ccd, sweep_col, sweep_vals)
            print("done")

    # Shared colour scale across all frames and all panels
    all_vals = np.concatenate([g.ravel() for g in all_grids.values()])
    vmax = max(abs(all_vals - 1.0).max(), 0.005)
    norm = TwoSlopeNorm(vcenter=1.0, vmin=1.0 - vmax, vmax=1.0 + vmax)
    cmap = "RdBu_r"

    # Build figure
    fig, axes = plt.subplots(4, 4, figsize=(13, 12),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.06, "wspace": 0.06})

    # Initial contourf plots
    contours = {}
    for i, cam in enumerate(CAMS):
        for j, ccd in enumerate(CCDS):
            ax = axes[i][j]
            data = all_grids[(cam, ccd)][0]
            cf = ax.contourf(CC, RR, data, levels=30, cmap=cmap, norm=norm)
            contours[(cam, ccd)] = cf
            ax.text(0.04, 0.96, f"Cam{cam}/CCD{ccd}", transform=ax.transAxes,
                    fontsize=7, va="top", color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4))
            if i == 3: ax.set_xlabel("Column", fontsize=8)
            if j == 0: ax.set_ylabel("Row", fontsize=8)
            ax.tick_params(labelsize=6)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Predicted flux offset", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    suptitle = fig.suptitle("", fontsize=12, y=1.005)

    def update(frame_idx):
        label = sweep_labels[frame_idx]
        suptitle.set_text(f"STITCH — {title_prefix} = {label}")
        for i, cam in enumerate(CAMS):
            for j, ccd in enumerate(CCDS):
                ax = axes[i][j]
                for coll in ax.collections:
                    coll.remove()
                data = all_grids[(cam, ccd)][frame_idx]
                ax.contourf(CC, RR, data, levels=30, cmap=cmap, norm=norm)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(sweep_vals),
                                  interval=1000//fps, blit=False)
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_file, writer=writer, dpi=100)
    plt.close(fig)
    print(f"  Saved → {out_file}")


# ── 1. Sector sweep (1 → 83, step 2) ─────────────────────────────────────────

sector_vals   = np.arange(1, 84, 2).astype(float)
sector_labels = [f"{int(s):02d}" for s in sector_vals]
make_video("sector", sector_vals, sector_labels,
           "stitch_flow_sector.gif", "Sector", fps=8)

# ── 2. Sub-pixel position sweep (delta_sub_col 0 → 1) ────────────────────────

subpix_vals   = np.linspace(0, 1, 50)
subpix_labels = [f"{v:.2f}" for v in subpix_vals]
make_video("delta_sub_col", subpix_vals, subpix_labels,
           "stitch_flow_subpixel.gif", "Sub-pixel col offset", fps=10)

# ── 3. Tmag sweep (6 → 13) ───────────────────────────────────────────────────

tmag_vals   = np.linspace(6, 13, 50)
tmag_labels = [f"{v:.1f}" for v in tmag_vals]
make_video("tmag", tmag_vals, tmag_labels,
           "stitch_flow_tmag.gif", "Tmag", fps=10)

print("\nAll videos done.")
