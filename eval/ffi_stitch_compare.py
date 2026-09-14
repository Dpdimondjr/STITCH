"""
Compare LOO labels vs STITCH predictions for the same sky patch across sectors.

Uses the same cam 4 CVZ patch as ffi_cross_sector.py.
Top row: LOO flux_offset heatmap per sector.
Bottom row: STITCH predicted flux_offset heatmap (same colour scale).

Usage:
  python3 eval/ffi_stitch_compare.py [model.pt] [parquet]
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),
               "stitch_nsf_pdc_full.pt")
print(f"Model:   {MODEL}")
print(f"Parquet: {PARQUET}")

# ── Load STITCH model ─────────────────────────────────────────────────────────
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(
    features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"],
    bins=cfg["bins"],
)
flow.load_state_dict(ckpt["model_state"])
flow.eval()
means, stds = ckpt["means"], ckpt["stds"]
y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
CONTINUOUS, CAM_COLS, CCD_COLS = ckpt["continuous_cols"], ckpt["cam_cols"], ckpt["ccd_cols"]

# ── Load & clean data ─────────────────────────────────────────────────────────
print("Loading parquet …")
df = pd.read_parquet(PARQUET)
# Use LOO label for display; compute Gaia features if model needs them
if "flux_offset_loo" not in df.columns and "flux_offset" in df.columns:
    df["flux_offset_loo"] = df["flux_offset"]
if "gaiarp" in df.columns and "flux_offset" in df.columns:
    df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")
df = df.dropna(subset=["col", "row", "flux_offset_loo", "ra", "dec"])
df = df[(df["flux_offset_loo"] > 0.85) & (df["flux_offset_loo"] < 1.15)]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Same cam 4 patch as ffi_cross_sector ─────────────────────────────────────
SOUTH_SECTORS = set(range(1, 14)) | set(range(27, 40)) | set(range(56, 70))
cam4  = df[(df["cam"] == 4) & (df["sector"].isin(SOUTH_SECTORS))].copy()
n_sec = cam4.groupby("tic_id")["sector"].nunique()
rich  = n_sec[n_sec >= 6].index
cam4  = cam4[cam4["tic_id"].isin(rich)].copy()

ra_c  = cam4.groupby("tic_id")["ra"].first().median()
dec_c = cam4.groupby("tic_id")["dec"].first().median()
HALF_RA, HALF_DEC = 5.0, 3.5
patch = cam4[
    (cam4["ra"]  >= ra_c  - HALF_RA)  & (cam4["ra"]  <= ra_c  + HALF_RA) &
    (cam4["dec"] >= dec_c - HALF_DEC) & (cam4["dec"] <= dec_c + HALF_DEC)
].copy().reset_index(drop=True)
print(f"Patch: {patch['tic_id'].nunique():,} stars, "
      f"{patch['sector'].nunique()} sectors, {len(patch):,} records")

# ── STITCH inference on all patch records ─────────────────────────────────────
def make_context(d):
    cont   = (d[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(
                 columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(
                 columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype("float32")

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
flow = flow.to(device)
print(f"Running inference on {len(patch):,} records ({device}) …")

C = torch.tensor(make_context(patch)).to(device)
with torch.no_grad():
    samples  = flow(C).sample((200,)).squeeze(-1)
    mu_z     = samples.mean(0).cpu().numpy()
    sig_z    = samples.std(0).cpu().numpy()

pred_raw = mu_z * y_std + y_mean
pred_std = sig_z * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)
patch["pred_offset"] = weight * pred_raw + (1 - weight) * 1.0
patch["residual"]    = patch["flux_offset_loo"] - patch["pred_offset"]

print(f"  Mean weight: {weight.mean():.3f}   "
      f"Pred range: [{patch['pred_offset'].min():.4f}, {patch['pred_offset'].max():.4f}]")

# ── Pick 4 most informative sectors ──────────────────────────────────────────
SHOW_SECS = [1, 4, 9, 13]

# ── Shared colour scale across LOO and prediction ─────────────────────────────
shown = patch[patch["sector"].isin(SHOW_SECS)]
lo = np.percentile(shown["flux_offset_loo"], 1)
hi = np.percentile(shown["flux_offset_loo"], 99)
extent = max(abs(lo - 1.0), abs(hi - 1.0), 0.010)
VMIN, VMAX = 1.0 - extent, 1.0 + extent
NORM  = TwoSlopeNorm(vcenter=1.0, vmin=VMIN, vmax=VMAX)
CMAP  = "RdBu_r"

# Residual uses its own tighter scale
res_ext = max(abs(shown["residual"]).quantile(0.98), 0.005)
RNORM = TwoSlopeNorm(vcenter=0.0, vmin=-res_ext, vmax=res_ext)

ra_lo,  ra_hi  = patch["ra"].min(),  patch["ra"].max()
dec_lo, dec_hi = patch["dec"].min(), patch["dec"].max()

# ── Figure: 3 rows × 4 cols ──────────────────────────────────────────────────
# Row 0: LOO   Row 1: STITCH prediction   Row 2: Residual (LOO − pred)
N = len(SHOW_SECS)
fig, axes = plt.subplots(3, N, figsize=(5 * N, 13),
                         gridspec_kw={"hspace": 0.35, "wspace": 0.08})
fig.patch.set_facecolor("#0b0f1a")

row_labels = ["LOO flux_offset", "STITCH prediction", "Residual (LOO − pred)"]
row_norms  = [NORM, NORM, RNORM]

for col_i, sec in enumerate(SHOW_SECS):
    g = patch[patch["sector"] == sec]

    for row_i, (data_col, norm) in enumerate(
            zip(["flux_offset_loo", "pred_offset", "residual"], row_norms)):

        ax = axes[row_i, col_i]
        ax.set_facecolor("#060a12")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1c2d45")

        ax.scatter(
            g["ra"], g["dec"],
            c=g[data_col], cmap=CMAP, norm=norm,
            s=18, alpha=0.85, linewidths=0, rasterized=True
        )
        ax.set_xlim(ra_hi + 0.3, ra_lo - 0.3)
        ax.set_ylim(dec_lo - 0.3, dec_hi + 0.3)
        ax.tick_params(colors="#3a5070", labelsize=7)

        if col_i == 0:
            ax.set_ylabel(row_labels[row_i], color="#c4d4ee", fontsize=9)
        if row_i == 0:
            med    = g["flux_offset_loo"].median()
            spread = g["flux_offset_loo"].quantile(0.9) - g["flux_offset_loo"].quantile(0.1)
            ax.set_title(f"Sector {sec}\nn={len(g)}  med={med:.4f}  p10-p90={spread:.4f}",
                         color="#c4d4ee", fontsize=10, pad=4)
        if row_i == 2:
            res_med = g["residual"].median()
            res_std = g["residual"].std()
            ax.text(0.03, 0.04,
                    f"res med={res_med:+.4f}  σ={res_std:.4f}",
                    transform=ax.transAxes, color="#f59e0b",
                    fontsize=7.5, va="bottom")
        if row_i == 2:
            ax.set_xlabel("RA (°)", color="#7a9cc4", fontsize=8)

# Colourbars
sm_main = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm_main.set_array([])
cbar1 = fig.colorbar(sm_main, ax=axes[:2, :].ravel().tolist(),
                     orientation="vertical", fraction=0.012, pad=0.01, shrink=0.6)
cbar1.set_label("flux_offset", color="#c4d4ee", fontsize=9)
cbar1.ax.yaxis.set_tick_params(labelcolor="#7a9cc4", labelsize=7)
cbar1.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

sm_res = plt.cm.ScalarMappable(cmap=CMAP, norm=RNORM)
sm_res.set_array([])
cbar2 = fig.colorbar(sm_res, ax=axes[2, :].ravel().tolist(),
                     orientation="vertical", fraction=0.012, pad=0.01, shrink=0.9)
cbar2.set_label("LOO − prediction", color="#c4d4ee", fontsize=9)
cbar2.ax.yaxis.set_tick_params(labelcolor="#7a9cc4", labelsize=7)
cbar2.ax.axhline(y=0.0, color="white", lw=0.8, alpha=0.5)

fig.suptitle(
    f"STITCH vs LOO — Cam 4 CVZ patch ({HALF_RA*2:.0f}°×{HALF_DEC*2:.0f}°)\n"
    f"Model: {MODEL}   |   Top: LOO label   Middle: STITCH prediction   "
    f"Bottom: residual",
    color="#c4d4ee", fontsize=11, y=1.005
)

out = "eval/ffi_stitch_compare.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"\nSaved → {out}")
