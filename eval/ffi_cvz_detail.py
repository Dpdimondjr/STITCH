"""
Extended CVZ visualisations for the same cam 4 patch used in ffi_cross_sector.py.

Produces three figures:
  ffi_cvz_all_sectors.png   — compact grid of ALL sectors showing gradient rotation
  ffi_cvz_detector_trace.png — where does the CVZ patch land on cam 4 across sectors?
  ffi_cvz_star_timeline.png  — per-star LOO timeline with STITCH prediction overlay

Usage:
  python3 eval/ffi_cvz_detail.py [model.pt] [parquet]
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch, zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),
               "stitch_nsf_pdc_full.pt")
print(f"Model:   {MODEL}\nParquet: {PARQUET}")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(features=cfg["features"], context=cfg["context"],
                      transforms=cfg["transforms"],
                      hidden_features=cfg["hidden_features"], bins=cfg["bins"])
flow.load_state_dict(ckpt["model_state"])
flow.eval()
means, stds   = ckpt["means"], ckpt["stds"]
y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
CONTINUOUS, CAM_COLS, CCD_COLS = ckpt["continuous_cols"], ckpt["cam_cols"], ckpt["ccd_cols"]

# ── Load data & build patch ───────────────────────────────────────────────────
print("Loading parquet …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "ra", "dec"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

cam4  = df[df["cam"] == 4].copy()
n_sec = cam4.groupby("tic_id")["sector"].nunique()
cam4  = cam4[cam4["tic_id"].isin(n_sec[n_sec >= 10].index)].copy()
ra_c  = cam4.groupby("tic_id")["ra"].first().median()
dec_c = cam4.groupby("tic_id")["dec"].first().median()
HALF_RA, HALF_DEC = 5.0, 3.5
patch = cam4[
    (cam4["ra"]  >= ra_c - HALF_RA)  & (cam4["ra"]  <= ra_c + HALF_RA) &
    (cam4["dec"] >= dec_c - HALF_DEC) & (cam4["dec"] <= dec_c + HALF_DEC)
].copy().reset_index(drop=True)
print(f"Patch: {patch['tic_id'].nunique():,} stars, "
      f"{patch['sector'].nunique()} sectors, {len(patch):,} records")

# ── STITCH inference ───────────────────────────────────────────────────────────
def make_context(d):
    cont   = (d[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(
                 columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(
                 columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True), cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype("float32")

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
flow = flow.to(device)
print(f"Running inference on {len(patch):,} records ({device}) …")
BATCH = 8192
mus, sigs = [], []
for i in range(0, len(patch), BATCH):
    C = torch.tensor(make_context(patch.iloc[i:i+BATCH])).to(device)
    with torch.no_grad():
        s = flow(C).sample((200,)).squeeze(-1)
        mus.append(s.mean(0).cpu().numpy())
        sigs.append(s.std(0).cpu().numpy())
pred_raw  = np.concatenate(mus)  * y_std + y_mean
pred_std  = np.concatenate(sigs) * y_std
weight    = 1.0 / (1.0 + 5.0 * pred_std)
patch["pred"] = weight * pred_raw + (1 - weight) * 1.0
patch["pred_std"] = pred_std

# ── Shared colour scale ───────────────────────────────────────────────────────
lo = np.percentile(patch["flux_offset"], 1)
hi = np.percentile(patch["flux_offset"], 99)
ext = max(abs(lo-1), abs(hi-1), 0.010)
NORM = TwoSlopeNorm(vcenter=1.0, vmin=1-ext, vmax=1+ext)
CMAP = "RdBu_r"
BG, AX, TEXT, MUTE = "#0b0f1a", "#060a12", "#c4d4ee", "#7a9cc4"

ra_lo,  ra_hi  = patch["ra"].min(),  patch["ra"].max()
dec_lo, dec_hi = patch["dec"].min(), patch["dec"].max()

def make_heatmap(g, col="flux_offset", nbins=7):
    re = np.linspace(ra_lo,  ra_hi,  nbins+1)
    de = np.linspace(dec_lo, dec_hi, nbins+1)
    stat, _, _, _ = binned_statistic_2d(g["ra"], g["dec"], g[col],
                                        statistic="median", bins=[re, de])
    cnt,  _, _, _ = binned_statistic_2d(g["ra"], g["dec"], g[col],
                                        statistic="count",  bins=[re, de])
    rc = 0.5*(re[:-1]+re[1:]); dc = 0.5*(de[:-1]+de[1:])
    RR, DD = np.meshgrid(rc, dc, indexing="ij")
    obs = cnt >= 1
    if obs.sum() >= 4:
        filled = griddata(np.column_stack([RR[obs], DD[obs]]), stat[obs],
                          (RR, DD), method="linear")
        filled[~binary_dilation(obs, iterations=1)] = np.nan
    else:
        filled = np.where(obs, stat, np.nan)
    return filled, re, de

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: All-sectors compact grid
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: all-sectors grid …")
sec_stats = (patch.groupby("sector")["flux_offset"]
             .agg(n="count", spread=lambda x: x.quantile(.9)-x.quantile(.1)))
show_secs = sorted(sec_stats[sec_stats["n"] >= 20].index.tolist())
NCOLS = 6
NROWS = int(np.ceil(len(show_secs) / NCOLS))
fig1, axes1 = plt.subplots(NROWS, NCOLS,
                            figsize=(NCOLS*2.8, NROWS*2.6 + 0.6))
fig1.patch.set_facecolor(BG)
axes1_flat = axes1.flatten()

for i, sec in enumerate(show_secs):
    ax = axes1_flat[i]
    ax.set_facecolor(AX)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")
    g = patch[patch["sector"] == sec]
    filled, re, de = make_heatmap(g, nbins=6)
    ax.imshow(filled.T, extent=[re[-1], re[0], de[0], de[-1]],
              origin="lower", aspect="auto",
              cmap=CMAP, norm=NORM, alpha=0.85, interpolation="bilinear")
    ax.scatter(g["ra"], g["dec"], c=g["flux_offset"], cmap=CMAP, norm=NORM,
               s=3, alpha=0.65, linewidths=0, rasterized=True)
    ax.invert_xaxis()
    ax.set_xticks([]); ax.set_yticks([])
    spread = sec_stats.loc[sec, "spread"]
    med    = g["flux_offset"].median()
    ax.set_title(f"S{sec}", color=TEXT, fontsize=8, pad=2)
    # Small spread indicator bar at bottom
    bar_col = "#ef4444" if med > 1.003 else ("#3b82f6" if med < 0.997 else "#4a5568")
    ax.axhline(dec_lo + 0.02*(dec_hi-dec_lo), color=bar_col,
               lw=max(1, spread/ext*3), alpha=0.7)

for j in range(len(show_secs), len(axes1_flat)):
    axes1_flat[j].set_visible(False)

sm1 = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM); sm1.set_array([])
cbar1 = fig1.colorbar(sm1, ax=axes1_flat[:len(show_secs)].tolist(),
                      orientation="vertical", fraction=0.008, pad=0.01, shrink=0.7)
cbar1.set_label("LOO flux_offset", color=TEXT, fontsize=9)
cbar1.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)
cbar1.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

fig1.suptitle(
    f"Cam 4 CVZ — all {len(show_secs)} sectors  |  same 10°×7° sky patch\n"
    "Colour bar at bottom of each panel = median offset direction & spread",
    color=TEXT, fontsize=10, y=1.005)
plt.tight_layout(h_pad=0.4, w_pad=0.3)
out1 = "eval/ffi_cvz_all_sectors.png"
plt.savefig(out1, dpi=140, bbox_inches="tight", facecolor=BG)
print(f"  → {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Detector position trace
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: detector position trace …")

# For each sector, compute the patch centroid in col/row space
patch_center = (patch.groupby("sector")
                .agg(col_med=("col","median"), row_med=("row","median"),
                     med_offset=("flux_offset","median"),
                     n=("tic_id","count"))
                .reset_index())

# Also get all-cam4 centroid for context
all_cam4_center = cam4.groupby("sector").agg(
    col_med=("col","median"), row_med=("row","median")).reset_index()

secs_sorted = sorted(show_secs)
n_s = len(secs_sorted)
colors_sec = plt.cm.plasma(np.linspace(0.1, 0.9, n_s))
sec_color = {s: colors_sec[i] for i, s in enumerate(secs_sorted)}

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.patch.set_facecolor(BG)

# Left: col/row trace of patch centroid coloured by sector
ax2L = axes2[0]
ax2L.set_facecolor(AX)
for sp in ax2L.spines.values(): sp.set_edgecolor("#1c2d45")

# Draw all cam4 stars (sector 1 as reference density)
s1 = cam4[cam4["sector"] == show_secs[len(show_secs)//2]]
ax2L.scatter(s1["col"], s1["row"], c="#1a2a44", s=0.3, alpha=0.4,
             rasterized=True, linewidths=0)

# Draw the patch centroid trace
pc = patch_center[patch_center["sector"].isin(show_secs)].sort_values("sector")
sc2 = ax2L.scatter(pc["col_med"], pc["row_med"],
                   c=[sec_color[s] for s in pc["sector"]],
                   s=50, zorder=5, linewidths=0.5, edgecolors="white")

# Connect with a thin line in sector order
ax2L.plot(pc["col_med"], pc["row_med"], color="white", lw=0.5, alpha=0.25, zorder=4)

# Annotate a few key sectors
for _, row in pc[pc["sector"].isin([1, 4, 9, 13, 27, 40])].iterrows():
    if row["sector"] in show_secs:
        ax2L.annotate(f"S{int(row['sector'])}",
                      (row["col_med"], row["row_med"]),
                      xytext=(6, 4), textcoords="offset points",
                      color=TEXT, fontsize=7.5, zorder=6)

ax2L.set_xlabel("col (px)", color=MUTE, fontsize=9)
ax2L.set_ylabel("row (px)", color=MUTE, fontsize=9)
ax2L.tick_params(colors=MUTE, labelsize=7)
ax2L.set_title("CVZ patch centroid trace across sectors\n(cam 4 detector space)",
               color=TEXT, fontsize=10, pad=5)
ax2L.set_xlim(0, 2100); ax2L.set_ylim(0, 2100)

# Right: col and row vs sector number, coloured by median LOO
ax2R = axes2[1]
ax2R.set_facecolor(AX)
for sp in ax2R.spines.values(): sp.set_edgecolor("#1c2d45")

norm_off = TwoSlopeNorm(vcenter=1.0, vmin=1-ext, vmax=1+ext)
sc2R = ax2R.scatter(pc["sector"], pc["col_med"],
                    c=pc["med_offset"], cmap=CMAP, norm=norm_off,
                    s=60, zorder=5, linewidths=0, label="col centroid")
ax2R.plot(pc["sector"], pc["col_med"], color="#3b8ef0", lw=1, alpha=0.4)
ax2Ra = ax2R.twinx()
ax2Ra.set_facecolor("none")
ax2Ra.scatter(pc["sector"], pc["row_med"],
              c=pc["med_offset"], cmap=CMAP, norm=norm_off,
              s=60, marker="s", zorder=5, linewidths=0, label="row centroid")
ax2Ra.plot(pc["sector"], pc["row_med"], color="#f59e0b", lw=1, alpha=0.4)
ax2Ra.set_ylabel("row centroid (px)", color="#f59e0b", fontsize=9)
ax2Ra.tick_params(colors=MUTE, labelsize=7)

ax2R.set_xlabel("Sector", color=MUTE, fontsize=9)
ax2R.set_ylabel("col centroid (px)", color="#3b8ef0", fontsize=9)
ax2R.tick_params(colors=MUTE, labelsize=7)
ax2R.set_title("Detector position vs sector\n(colour = median LOO offset)",
               color=TEXT, fontsize=10, pad=5)

sm2 = plt.cm.ScalarMappable(cmap=CMAP, norm=norm_off); sm2.set_array([])
cbar2 = fig2.colorbar(sm2, ax=[ax2R, ax2Ra], fraction=0.03, pad=0.12, shrink=0.85)
cbar2.set_label("median LOO offset", color=TEXT, fontsize=8)
cbar2.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)
cbar2.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

# Legend manually
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], color="#3b8ef0", lw=1.5, label="col centroid"),
       Line2D([0],[0], color="#f59e0b", lw=1.5, label="row centroid")]
ax2R.legend(handles=leg, fontsize=7.5, labelcolor=TEXT,
            facecolor="#0d1a30", edgecolor="#1c2d45")

fig2.suptitle(
    "Why the systematic rotates: the CVZ patch moves across the cam 4 detector each sector",
    color=TEXT, fontsize=11, y=1.02)
plt.tight_layout()
out2 = "eval/ffi_cvz_detector_trace.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"  → {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Per-star LOO timeline with STITCH overlay
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: per-star timelines …")

# Pick 8 stars: sample across the range of mean LOO offset and spread
star_stats = (patch.groupby("tic_id")
              .agg(mean_loo=("flux_offset","mean"),
                   std_loo=("flux_offset","std"),
                   n_sec=("sector","nunique"),
                   ra=("ra","first"), dec=("dec","first"))
              .dropna())
star_stats = star_stats[star_stats["n_sec"] >= 10]
# Stratify by mean_loo into 8 bins and pick the most representative
star_stats["bin"] = pd.qcut(star_stats["mean_loo"], 8, labels=False, duplicates="drop")
chosen = (star_stats.groupby("bin", observed=True)
          .apply(lambda g: g.loc[g["std_loo"].sub(g["std_loo"].median()).abs().idxmin()])
          .reset_index(drop=True))
chosen_tids = chosen["tic_id"].tolist() if "tic_id" in chosen.columns else chosen.index.tolist()
if "tic_id" not in chosen.columns:
    chosen_tids = list(chosen.index)
print(f"  Selected {len(chosen_tids)} stars for timeline")

N_STARS = len(chosen_tids)
NCOLS_T = 4
NROWS_T = int(np.ceil(N_STARS / NCOLS_T))
fig3, axes3 = plt.subplots(NROWS_T, NCOLS_T,
                            figsize=(NCOLS_T*4.5, NROWS_T*3.2))
fig3.patch.set_facecolor(BG)
axes3_flat = axes3.flatten()

for i, tid in enumerate(chosen_tids):
    ax = axes3_flat[i]
    ax.set_facecolor(AX)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")

    star_data = patch[patch["tic_id"] == tid].sort_values("sector")
    if len(star_data) < 2:
        ax.set_visible(False)
        continue
    secs = star_data["sector"].values
    loo  = star_data["flux_offset"].values
    pred = star_data["pred"].values
    psig = star_data["pred_std"].values

    # LOO as scatter + line
    ax.plot(secs, loo, color="#7a9cc4", lw=1.2, alpha=0.7, zorder=3)
    ax.scatter(secs, loo, c=loo, cmap=CMAP, norm=NORM,
               s=18, zorder=4, linewidths=0)

    # STITCH prediction + uncertainty band
    ax.plot(secs, pred, color="#f59e0b", lw=1.8, alpha=0.9, zorder=5, label="STITCH")
    ax.fill_between(secs, pred - psig, pred + psig,
                    color="#f59e0b", alpha=0.15, zorder=2)

    ax.axhline(1.0, color="white", lw=0.6, alpha=0.3, ls="--")
    ax.tick_params(colors=MUTE, labelsize=7)
    ax.set_xlim(secs.min()-1, secs.max()+1)

    mean_l = loo.mean()
    std_l  = loo.std()
    bar_col = "#ef4444" if mean_l > 1.003 else ("#3b82f6" if mean_l < 0.997 else "#4a5568")
    ax.set_title(f"TIC {tid}\nmean={mean_l:.4f}  σ={std_l:.4f}",
                 color=TEXT, fontsize=8, pad=3)
    ax.set_xlabel("Sector", color=MUTE, fontsize=7)
    if i % NCOLS_T == 0:
        ax.set_ylabel("flux_offset", color=MUTE, fontsize=7)

# Legend on last used panel
axes3_flat[N_STARS-1].plot([], [], color="#7a9cc4", lw=1.5, label="LOO label")
axes3_flat[N_STARS-1].plot([], [], color="#f59e0b", lw=1.8, label="STITCH prediction")
axes3_flat[N_STARS-1].legend(fontsize=7.5, labelcolor=TEXT,
                              facecolor="#0d1a30", edgecolor="#1c2d45",
                              loc="upper right")

for j in range(N_STARS, len(axes3_flat)):
    axes3_flat[j].set_visible(False)

fig3.suptitle(
    "Per-star LOO timeline vs STITCH prediction — 8 CVZ stars sampled across offset range\n"
    "Blue/red dots = LOO offset per sector   |   Amber line = STITCH prediction ± 1σ",
    color=TEXT, fontsize=10, y=1.02)
plt.tight_layout(h_pad=0.8, w_pad=0.4)
out3 = "eval/ffi_cvz_star_timeline.png"
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"  → {out3}")
print("\nDone.")
