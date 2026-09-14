"""
Diagnostic: which stars does STITCH hurt, and what's the performance ceiling?

Three panels:
  A) Feature distributions — hurt vs improved stars (violin plots)
  B) Spatial position of hurt stars within each cam/CCD
  C) Ceiling analysis — per-star neighbor LOO correlation vs actual improvement

Ceiling logic:
  For each star, compute r = Pearson correlation between its LOO values and the
  mean LOO of its K nearest col/row neighbours (same cam, ccd), across shared
  sectors.  r² estimates the fraction of per-star LOO variance that is a spatially
  coherent systematic (and therefore learnable).  A perfect model could achieve:
    scatter_reduction_ceiling ≈ 1 - sqrt(1 - r²)
  The residual (1 - r²) is star-specific noise or astrophysical variability that
  no context-based model can correct.

Usage:
  python3 eval/hurt_star_analysis.py [model.pt] [parquet]
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),
               "stitch_nsf_pdc_full.pt")
K_NEIGHBOURS = 15   # nearest neighbours for ceiling estimate
MIN_SHARED   = 5    # minimum shared sectors for a valid correlation estimate
print(f"Model:   {MODEL}")
print(f"Parquet: {PARQUET}")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(
    features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"],
    bins=cfg["bins"],
)
flow.load_state_dict(ckpt["model_state"])
flow.eval()
means, stds   = ckpt["means"], ckpt["stds"]
y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
CONTINUOUS, CAM_COLS, CCD_COLS = ckpt["continuous_cols"], ckpt["cam_cols"], ckpt["ccd_cols"]

# ── Load & clean data ─────────────────────────────────────────────────────────
print("Loading parquet …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Require ≥4 sectors for a meaningful scatter estimate
n_sec = df.groupby("tic_id")["sector"].nunique()
df = df[df["tic_id"].isin(n_sec[n_sec >= 4].index)].copy().reset_index(drop=True)
print(f"  {df['tic_id'].nunique():,} stars, {len(df):,} records after filtering")

# ── STITCH inference ───────────────────────────────────────────────────────────
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
print(f"Running inference on {len(df):,} records ({device}) …")

BATCH = 8192
preds, pred_stds = [], []
for i in range(0, len(df), BATCH):
    chunk = df.iloc[i:i+BATCH]
    C = torch.tensor(make_context(chunk)).to(device)
    with torch.no_grad():
        samp = flow(C).sample((200,)).squeeze(-1)
        preds.append(samp.mean(0).cpu().numpy())
        pred_stds.append(samp.std(0).cpu().numpy())

mu_z  = np.concatenate(preds)
sig_z = np.concatenate(pred_stds)
pred_raw = mu_z * y_std + y_mean
pred_std  = sig_z * y_std
weight    = 1.0 / (1.0 + 5.0 * pred_std)
df["pred"]   = weight * pred_raw + (1 - weight) * 1.0
df["pred_std"] = pred_std
df["corrected"] = df["flux_offset"] / df["pred"]
print(f"  Pred range [{df['pred'].min():.4f}, {df['pred'].max():.4f}]  "
      f"mean weight {weight.mean():.3f}")

# ── Per-star scatter eval ──────────────────────────────────────────────────────
def sector_std(x): return x.std(ddof=1) if len(x) >= 3 else np.nan

g = df.groupby("tic_id")
star_raw = g["flux_offset"].agg(sector_std).rename("raw_std")
star_cor = g["corrected"].agg(sector_std).rename("cor_std")
star_meta = g.agg(
    n_sectors=("sector", "nunique"),
    tmag=("tmag", "median"),
    cdpp1_0=("cdpp1_0", "median"),
    pdcvar=("pdcvar", "median"),
    crowdsap=("crowdsap", "median"),
    cam=("cam", "first"),
    ccd=("ccd", "first"),
    col=("col", "median"),
    row=("row", "median"),
    mean_weight=("pred_std", lambda x: (1/(1+5*x)).mean()),
    mean_pred_std=("pred_std", "mean"),
)
stars = pd.concat([star_raw, star_cor, star_meta], axis=1).dropna(subset=["raw_std","cor_std"])
stars["improvement"] = 1.0 - stars["cor_std"] / stars["raw_std"]
stars["hurt"]        = stars["improvement"] < 0

n_hurt     = stars["hurt"].sum()
n_improved = (~stars["hurt"]).sum()
med_imp    = stars.loc[~stars["hurt"], "improvement"].median()
print(f"\nScatter eval: {n_improved:,} improved ({n_improved/len(stars)*100:.1f}%)  "
      f"{n_hurt:,} hurt ({n_hurt/len(stars)*100:.1f}%)")
print(f"  Median improvement (improved stars): {med_imp*100:.1f}%")
print(f"  Overall median improvement: {stars['improvement'].median()*100:.1f}%")

# ── Ceiling: per-star neighbour LOO correlation ────────────────────────────────
print(f"\nComputing neighbour LOO correlation (K={K_NEIGHBOURS}) …")
star_loo = df.pivot_table(index="tic_id", columns="sector", values="flux_offset")

# Build KD-tree per (cam, ccd) in col/row space using per-star median position
ceiling_r = {}
for (cam, ccd), grp in star_meta.groupby(["cam", "ccd"]):
    tids = grp.index.tolist()
    if len(tids) < K_NEIGHBOURS + 1:
        continue
    coords = grp[["col", "row"]].values
    # Normalise col/row to roughly equal scale
    col_scale = coords[:, 0].std() + 1e-6
    row_scale = coords[:, 1].std() + 1e-6
    norm_coords = coords / np.array([col_scale, row_scale])
    tree = cKDTree(norm_coords)
    _, idx = tree.query(norm_coords, k=K_NEIGHBOURS + 1)  # +1 includes self
    for i, tid in enumerate(tids):
        neighbour_tids = [tids[j] for j in idx[i, 1:]]  # exclude self
        if tid not in star_loo.index:
            continue
        focal = star_loo.loc[tid].dropna()
        nb_loos = star_loo.loc[[t for t in neighbour_tids if t in star_loo.index]]
        if len(nb_loos) == 0:
            continue
        nb_mean = nb_loos.mean(axis=0).dropna()
        shared  = focal.index.intersection(nb_mean.index)
        if len(shared) < MIN_SHARED:
            continue
        r = np.corrcoef(focal[shared].values, nb_mean[shared].values)[0, 1]
        ceiling_r[tid] = r

stars["neighbour_r"]  = stars.index.map(ceiling_r)
stars["ceiling_reduction"] = 1.0 - np.sqrt(np.maximum(0, 1 - stars["neighbour_r"]**2))
valid_ceil = stars.dropna(subset=["neighbour_r"])
print(f"  Ceiling computed for {len(valid_ceil):,} stars")
print(f"  Median neighbour r: {valid_ceil['neighbour_r'].median():.3f}")
print(f"  Median ceiling reduction: {valid_ceil['ceiling_reduction'].median()*100:.1f}%")

# Ceiling by quiet vs noisy stars
quiet_mask = valid_ceil["cdpp1_0"] < valid_ceil["cdpp1_0"].quantile(0.33)
noisy_mask = valid_ceil["cdpp1_0"] > valid_ceil["cdpp1_0"].quantile(0.67)
print(f"  Quiet star ceiling (low cdpp):  {valid_ceil.loc[quiet_mask,'ceiling_reduction'].median()*100:.1f}%")
print(f"  Noisy star ceiling (high cdpp): {valid_ceil.loc[noisy_mask,'ceiling_reduction'].median()*100:.1f}%")
print(f"  Quiet star actual improvement:  {valid_ceil.loc[quiet_mask,'improvement'].median()*100:.1f}%")
print(f"  Noisy star actual improvement:  {valid_ceil.loc[noisy_mask,'improvement'].median()*100:.1f}%")

# ── Figure ─────────────────────────────────────────────────────────────────────
BG   = "#0b0f1a"
AX   = "#060a12"
TEXT = "#c4d4ee"
MUTE = "#7a9cc4"
RED  = "#ef4444"
BLUE = "#3b82f6"
GOLD = "#f59e0b"
GRN  = "#22c55e"

fig = plt.figure(figsize=(20, 18), facecolor=BG)
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        hspace=0.45, wspace=0.35,
                        left=0.06, right=0.97, top=0.93, bottom=0.06)

def style_ax(ax):
    ax.set_facecolor(AX)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1c2d45")
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.yaxis.label.set_color(MUTE)
    ax.xaxis.label.set_color(MUTE)
    ax.title.set_color(TEXT)

# ── Row A: feature violin plots ───────────────────────────────────────────────
FEATURES = [
    ("n_sectors",    "N sectors observed"),
    ("cdpp1_0",      "CDPP 1-hr (ppm)"),
    ("pdcvar",       "PDC variability"),
    ("crowdsap",     "Crowding (CROWDSAP)"),
    ("mean_weight",  "Mean inference weight"),
]
for fi, (feat, label) in enumerate(FEATURES):
    if fi >= 4:
        break
    ax = fig.add_subplot(gs[0, fi])
    style_ax(ax)
    hurt_vals = stars.loc[stars["hurt"],     feat].dropna().clip(
        *np.percentile(stars[feat].dropna(), [1, 99]))
    good_vals = stars.loc[~stars["hurt"],    feat].dropna().clip(
        *np.percentile(stars[feat].dropna(), [1, 99]))
    parts = ax.violinplot([good_vals, hurt_vals], positions=[0, 1],
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.55)
    parts["bodies"][0].set_facecolor(BLUE)
    parts["bodies"][1].set_facecolor(RED)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Improved\n(n={n_improved:,})", f"Hurt\n(n={n_hurt:,})"],
                       color=TEXT, fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    med_g = good_vals.median()
    med_h = hurt_vals.median()
    ax.set_title(f"{label}\nmed improved={med_g:.2f}  hurt={med_h:.2f}",
                 fontsize=8, pad=3)

# Fifth violin in a merged cell
ax5 = fig.add_subplot(gs[0, 3])
style_ax(ax5)
hurt_vals5 = stars.loc[stars["hurt"],  "mean_weight"].dropna()
good_vals5 = stars.loc[~stars["hurt"], "mean_weight"].dropna()
parts5 = ax5.violinplot([good_vals5, hurt_vals5], positions=[0, 1],
                        showmedians=True, showextrema=False)
for pc in parts5["bodies"]:
    pc.set_alpha(0.55)
parts5["bodies"][0].set_facecolor(BLUE)
parts5["bodies"][1].set_facecolor(RED)
parts5["cmedians"].set_color("white")
parts5["cmedians"].set_linewidth(2)
ax5.set_xticks([0, 1])
ax5.set_xticklabels([f"Improved\n(n={n_improved:,})", f"Hurt\n(n={n_hurt:,})"],
                    color=TEXT, fontsize=8)
ax5.set_ylabel("Mean inference weight", fontsize=8)
med_g5 = good_vals5.median()
med_h5 = hurt_vals5.median()
ax5.set_title(f"Inference weight\nmed improved={med_g5:.3f}  hurt={med_h5:.3f}",
              fontsize=8, pad=3)

# ── Row B: spatial positions of hurt stars on detector ────────────────────────
CAM_CCD_PAIRS = [(c, d) for c in [1,2,3,4] for d in [1,2,3,4]]
# Show one row of 4 cam panels, each with CCD subplots
ax_sp = [fig.add_subplot(gs[1, ci]) for ci in range(4)]
for ci, cam in enumerate([1,2,3,4]):
    ax = ax_sp[ci]
    style_ax(ax)
    cam_stars = stars[stars["cam"] == cam]
    hurt_s    = cam_stars[cam_stars["hurt"]]
    good_s    = cam_stars[~cam_stars["hurt"]]
    # Colour by improvement, red for hurt
    scatter_all = ax.scatter(good_s["col"], good_s["row"],
                             c=good_s["improvement"].clip(-0.5, 0.8),
                             cmap="RdYlGn", vmin=-0.5, vmax=0.8,
                             s=1.5, alpha=0.4, rasterized=True, linewidths=0)
    ax.scatter(hurt_s["col"], hurt_s["row"],
               c=RED, s=4, alpha=0.7, rasterized=True, linewidths=0, zorder=5)
    ax.set_xlabel("col (px)", fontsize=8)
    if ci == 0:
        ax.set_ylabel("row (px)", fontsize=8)
    frac_hurt = len(hurt_s) / max(len(cam_stars), 1)
    ax.set_title(f"Cam {cam}  ({frac_hurt*100:.1f}% hurt)", fontsize=9, pad=3)

# Shared colorbar for row B
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(-0.5, 0.8))
sm.set_array([])
cbar_b = fig.colorbar(sm, ax=ax_sp, orientation="vertical",
                      fraction=0.012, pad=0.01, shrink=0.85)
cbar_b.set_label("scatter improvement", color=TEXT, fontsize=8)
cbar_b.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)
cbar_b.ax.axhline(y=0, color="white", lw=1.0, alpha=0.5)

# ── Row C: ceiling analysis ───────────────────────────────────────────────────
# C1: neighbour_r distribution split by quiet/noisy
ax_c1 = fig.add_subplot(gs[2, 0])
style_ax(ax_c1)
bins_r = np.linspace(-0.3, 1.0, 40)
ax_c1.hist(valid_ceil.loc[quiet_mask, "neighbour_r"], bins=bins_r,
           color=GRN, alpha=0.65, label="Quiet (low CDPP)", density=True)
ax_c1.hist(valid_ceil.loc[noisy_mask, "neighbour_r"], bins=bins_r,
           color=GOLD, alpha=0.65, label="Noisy (high CDPP)", density=True)
ax_c1.axvline(valid_ceil.loc[quiet_mask, "neighbour_r"].median(),
              color=GRN, lw=1.5, ls="--")
ax_c1.axvline(valid_ceil.loc[noisy_mask, "neighbour_r"].median(),
              color=GOLD, lw=1.5, ls="--")
ax_c1.set_xlabel("Neighbour LOO correlation  r", fontsize=8)
ax_c1.set_ylabel("Density", fontsize=8)
ax_c1.set_title("Spatial LOO correlation\n(higher r = more correctable)", fontsize=9, pad=3)
ax_c1.legend(fontsize=7.5, labelcolor=TEXT, facecolor="#0d1a30",
             edgecolor="#1c2d45")

# C2: ceiling vs actual improvement scatter
ax_c2 = fig.add_subplot(gs[2, 1])
style_ax(ax_c2)
vc = valid_ceil.dropna(subset=["ceiling_reduction", "improvement"])
# Colour by cdpp1_0 (log scale)
cdpp_log = np.log10(vc["cdpp1_0"].clip(10, 10000))
sc = ax_c2.scatter(vc["ceiling_reduction"] * 100, vc["improvement"] * 100,
                   c=cdpp_log, cmap="plasma_r",
                   vmin=cdpp_log.quantile(0.05), vmax=cdpp_log.quantile(0.95),
                   s=2, alpha=0.35, rasterized=True, linewidths=0)
ax_c2.axline((0,0), slope=1, color="white", lw=0.8, alpha=0.4, ls="--",
             label="ceiling = actual")
ax_c2.axhline(0, color=MUTE, lw=0.5, alpha=0.4)
ax_c2.set_xlabel("Ceiling reduction (%) from neighbour r", fontsize=8)
ax_c2.set_ylabel("Actual scatter improvement (%)", fontsize=8)
ax_c2.set_title("Ceiling vs actual improvement\n(colour = log CDPP)", fontsize=9, pad=3)
cbar_c = fig.colorbar(sc, ax=ax_c2, fraction=0.04, pad=0.02)
cbar_c.set_label("log₁₀(CDPP)", color=TEXT, fontsize=7)
cbar_c.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)

# C3: improvement vs cdpp, split hurt/good, with ceiling overlay
ax_c3 = fig.add_subplot(gs[2, 2])
style_ax(ax_c3)
cdpp_bins = np.logspace(np.log10(stars["cdpp1_0"].quantile(0.01)),
                        np.log10(stars["cdpp1_0"].quantile(0.99)), 20)
for mask, colour, label in [
    (~stars["hurt"], BLUE, "Improved"),
    (stars["hurt"],  RED,  "Hurt"),
]:
    sub = stars[mask].copy()
    sub["cdpp_bin"] = pd.cut(sub["cdpp1_0"], bins=cdpp_bins)
    med = sub.groupby("cdpp_bin", observed=True)["improvement"].median()
    mids = np.array([b.mid for b in med.index])
    ax_c3.plot(mids, med.values * 100, color=colour, lw=2, label=label)

# Ceiling line for quiet stars
if valid_ceil is not None:
    vc2 = valid_ceil.dropna(subset=["ceiling_reduction","cdpp1_0"])
    vc2["cdpp_bin"] = pd.cut(vc2["cdpp1_0"], bins=cdpp_bins)
    ceil_med = vc2.groupby("cdpp_bin", observed=True)["ceiling_reduction"].median()
    mids_c   = np.array([b.mid for b in ceil_med.index])
    ax_c3.plot(mids_c, ceil_med.values * 100, color="white", lw=1.5,
               ls=":", alpha=0.7, label="Ceiling (neighbour r)")

ax_c3.set_xscale("log")
ax_c3.axhline(0, color=MUTE, lw=0.5, alpha=0.4)
ax_c3.set_xlabel("CDPP 1-hr (ppm)", fontsize=8)
ax_c3.set_ylabel("Scatter improvement (%)", fontsize=8)
ax_c3.set_title("Improvement vs CDPP\nwith ceiling estimate", fontsize=9, pad=3)
ax_c3.legend(fontsize=7.5, labelcolor=TEXT, facecolor="#0d1a30",
             edgecolor="#1c2d45")

# C4: improvement vs n_sectors
ax_c4 = fig.add_subplot(gs[2, 3])
style_ax(ax_c4)
sec_bins = np.arange(4, stars["n_sectors"].max() + 2, 3)
for mask, colour, label in [
    (~stars["hurt"], BLUE, "Improved"),
    (stars["hurt"],  RED,  "Hurt"),
]:
    sub = stars[mask].copy()
    sub["sec_bin"] = pd.cut(sub["n_sectors"], bins=sec_bins)
    med = sub.groupby("sec_bin", observed=True)["improvement"].median()
    mids = np.array([b.mid for b in med.index])
    ax_c4.plot(mids, med.values * 100, color=colour, lw=2, label=label)

if valid_ceil is not None:
    vc3 = valid_ceil.dropna(subset=["ceiling_reduction","n_sectors"])
    vc3["sec_bin"] = pd.cut(vc3["n_sectors"], bins=sec_bins)
    ceil_med3 = vc3.groupby("sec_bin", observed=True)["ceiling_reduction"].median()
    mids_s    = np.array([b.mid for b in ceil_med3.index])
    ax_c4.plot(mids_s, ceil_med3.values * 100, color="white", lw=1.5,
               ls=":", alpha=0.7, label="Ceiling")

ax_c4.axhline(0, color=MUTE, lw=0.5, alpha=0.4)
ax_c4.set_xlabel("N sectors observed", fontsize=8)
ax_c4.set_ylabel("Scatter improvement (%)", fontsize=8)
ax_c4.set_title("Improvement vs N sectors\nwith ceiling estimate", fontsize=9, pad=3)
ax_c4.legend(fontsize=7.5, labelcolor=TEXT, facecolor="#0d1a30",
             edgecolor="#1c2d45")

fig.suptitle(
    f"STITCH hurt-star analysis & performance ceiling — {MODEL}\n"
    f"Total: {len(stars):,} stars  |  {n_improved:,} improved ({n_improved/len(stars)*100:.1f}%)  "
    f"|  {n_hurt:,} hurt ({n_hurt/len(stars)*100:.1f}%)  "
    f"|  Overall median improvement: {stars['improvement'].median()*100:.1f}%",
    color=TEXT, fontsize=11, y=0.97
)

out = "eval/hurt_star_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
