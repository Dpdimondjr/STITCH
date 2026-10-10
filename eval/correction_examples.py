"""
Visual explainer: what does each correction actually apply to a star?

For a handful of example test stars with good sector coverage, plots:
  - Raw flux_offset_loo across sectors  (the cross-sector drift to correct)
  - sector_ccd_mean_loo                 (whole-CCD average — the coarse correction)
  - spatial_knn_mean_loo                (20 nearest neighbours — finer correction)
  - STITCH pred_offset                  (flow prediction — what is actually applied)
  - The corrected series after dividing by each

Uses the same train/test split and training-only KNN as gaia_locus_validation.py.

Usage:
  python3 eval/correction_examples.py stitch_nsf_knn.pt training_data_topup_pdc.parquet
"""

import sys, warnings
import numpy as np
import pandas as pd
import torch, zuko
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_knn.pt")
PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")

# ── Load model ─────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"], bins=cfg["bins"])
flow.load_state_dict(ckpt["model_state"]); flow.eval()
means=ckpt["means"]; stds=ckpt["stds"]; y_mean=ckpt["y_mean"]; y_std=ckpt["y_std"]
CONTINUOUS=ckpt["continuous_cols"]; CAM_COLS=ckpt["cam_cols"]; CCD_COLS=ckpt["ccd_cols"]
device=(torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
flow = flow.to(device)

# ── Load parquet ───────────────────────────────────────────────────────────────
print("Loading …", flush=True)
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median","gaiarp"])
df = df[(df["flux_offset"]>0.85)&(df["flux_offset"]<1.15)]
df = df[df["sector_median"]>0]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
df = df.reset_index(drop=True)
df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")

# ── Split (same seed as validation) ───────────────────────────────────────────
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index().rename(columns={"cam":"dominant_cam"}))
tr, tmp = train_test_split(star_cam["tic_id"], test_size=0.2,
                            stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_, te   = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)

train_df = df[df["tic_id"].isin(tr)].copy().reset_index(drop=True)
test_pool = df[df["tic_id"].isin(te)].copy()

_t = "flux_offset_loo"
K  = 20

# ── Build training-star features (for KNN reference pool) ─────────────────────
print("Building training KNN …")
grp_sum   = train_df.groupby(["sector","cam","ccd"])[_t].transform("sum")
grp_count = train_df.groupby(["sector","cam","ccd"])[_t].transform("count")
train_df["sector_ccd_mean_loo"] = (grp_sum - train_df[_t]) / (grp_count - 1).clip(lower=1)

_knn_tr = np.full(len(train_df), np.nan, dtype=np.float32)
for (sec, cam, ccd), grp in train_df.groupby(["sector","cam","ccd"]):
    n = len(grp)
    if n < 2: continue
    co = grp[["col","row"]].values.astype(np.float32)
    v  = grp[_t].values
    k  = min(K, n-1)
    _, nn = cKDTree(co).query(co, k=k+1)
    _knn_tr[grp.index.values] = v[nn[:, 1:]].mean(axis=1)
train_df["spatial_knn_mean_loo"] = _knn_tr
train_df["spatial_knn_mean_loo"] = train_df["spatial_knn_mean_loo"].fillna(
    train_df["sector_ccd_mean_loo"])

# ── Pick example stars: test stars with enough sectors and moderate scatter ───────
ns = test_pool.groupby("tic_id")["sector"].count()
print(f"Test pool: {len(ns)} stars, sector count distribution: "
      f"max={ns.max()} p90={np.percentile(ns.values,90):.0f} "
      f"p75={np.percentile(ns.values,75):.0f} median={np.median(ns.values):.0f}")
# Prefer ≥20, fall back to ≥10, then ≥5
for min_sec in [20, 10, 5]:
    candidates = ns[ns >= min_sec].index.tolist()
    if len(candidates) >= 4:
        break
print(f"Candidates (≥{min_sec} sectors): {len(candidates)} stars")

star_scatter = (test_pool[test_pool["tic_id"].isin(candidates)]
                .groupby("tic_id")[_t].std()
                .dropna()
                .sort_values())
print(f"star_scatter has {len(star_scatter)} entries")
# Pick spread: low, medium-low, medium-high, high scatter
quartiles = np.percentile(star_scatter.values, [25, 45, 65, 85])
chosen = []
for q in quartiles:
    idx = (star_scatter - q).abs().idxmin()
    if idx not in chosen:
        chosen.append(int(idx))
chosen = chosen[:4]
print(f"Example stars: {chosen}")

# ── Build features for chosen test stars ───────────────────────────────────────
test_df = test_pool[test_pool["tic_id"].isin(chosen)].copy().reset_index(drop=True)

tr_grp_mean = (train_df.groupby(["sector","cam","ccd"])[_t].mean()
               .reset_index().rename(columns={_t: "sector_ccd_mean_loo"}))
test_df = test_df.merge(tr_grp_mean, on=["sector","cam","ccd"], how="left")
test_df["sector_ccd_mean_loo"] = test_df["sector_ccd_mean_loo"].fillna(test_df[_t])

_knn_te = np.full(len(test_df), np.nan, dtype=np.float32)
for (sec, cam, ccd), q_grp in test_df.groupby(["sector","cam","ccd"]):
    ref = train_df[(train_df["sector"]==sec)&
                   (train_df["cam"]==cam)&
                   (train_df["ccd"]==ccd)]
    if len(ref) < 2: continue
    r_co = ref[["col","row"]].values.astype(np.float32)
    r_v  = ref[_t].values
    q_co = q_grp[["col","row"]].values.astype(np.float32)
    k    = min(K, len(ref))
    _, nn_idx = cKDTree(r_co).query(q_co, k=k)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, np.newaxis]
    _knn_te[q_grp.index.values] = r_v[nn_idx].mean(axis=1)
test_df["spatial_knn_mean_loo"] = _knn_te
test_df["spatial_knn_mean_loo"] = test_df["spatial_knn_mean_loo"].fillna(
    test_df["sector_ccd_mean_loo"])

for col in CONTINUOUS:
    if col in test_df.columns and test_df[col].isna().any():
        test_df[col] = test_df[col].fillna(test_df[col].median())

# ── STITCH inference ───────────────────────────────────────────────────────────
def make_context(d):
    cont  = (d[CONTINUOUS].reindex(columns=CONTINUOUS, fill_value=0) - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(
                columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(
                columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype("float32")

print("Running STITCH inference …")
C = torch.tensor(make_context(test_df)).to(device)
with torch.no_grad():
    samp    = flow(C).sample((300,)).squeeze(-1)
    mu_z    = samp.mean(0).cpu().numpy()
    sig_z   = samp.std(0).cpu().numpy()
pred_raw = mu_z * y_std + y_mean
pred_std = sig_z * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)
test_df["pred_offset"] = weight * pred_raw + (1 - weight) * 1.0

# ── Figure ─────────────────────────────────────────────────────────────────────
BG,SURF,TEXT,MUT,DIM="#0b0f1a","#0f1826","#c4d4ee","#7a9cc4","#1c2d45"
C_RAW  = "#ef4444"
C_CCD  = "#f59e0b"
C_KNN  = "#a78bfa"
C_ST   = "#4f8ef7"

n_stars = len(chosen)
# Two rows per star: top = correction vectors, bottom = corrected series
fig = plt.figure(figsize=(18, 4 * n_stars), facecolor=BG)
gs  = GridSpec(n_stars * 2, 1, figure=fig, hspace=0.08)

def style(ax, ylabel, title=None, show_xlabel=False):
    ax.set_facecolor(SURF)
    if show_xlabel:
        ax.set_xlabel("TESS sector", color=MUT, fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, color=MUT, fontsize=8)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, pad=5)
    ax.tick_params(colors=MUT, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True, color=DIM, lw=0.3, zorder=0)
    ax.set_axisbelow(True)

legend_added = False

for i, tic in enumerate(chosen):
    star = test_df[test_df["tic_id"]==tic].sort_values("sector").copy()
    secs = star["sector"].values
    raw  = star[_t].values
    ccd  = star["sector_ccd_mean_loo"].values
    knn  = star["spatial_knn_mean_loo"].values
    st   = star["pred_offset"].values

    raw_std  = raw.std()
    ccd_std  = (raw / ccd).std()
    knn_std  = (raw / knn).std()
    st_std   = (raw / st).std()

    cam_val = int(star["cam"].iloc[0])
    ccd_val = int(star["ccd"].iloc[0])

    # Top: correction vectors overlaid on raw
    ax_top = fig.add_subplot(gs[2*i])
    ax_top.plot(secs, raw, color=C_RAW, lw=1.2, alpha=0.9,
                label=f"Raw flux_offset_loo  (σ={raw_std*100:.2f}%)", zorder=4)
    ax_top.plot(secs, ccd, color=C_CCD, lw=1.5, ls="--", alpha=0.9,
                label=f"sector_ccd_mean  — whole-CCD average for this sector", zorder=3)
    ax_top.plot(secs, knn, color=C_KNN, lw=1.2, ls=":",  alpha=0.9,
                label=f"spatial_knn_mean  — 20 nearest neighbours this sector", zorder=3)
    ax_top.plot(secs, st,  color=C_ST,  lw=1.8, alpha=1.0,
                label=f"STITCH pred_offset  — applied correction", zorder=5)
    ax_top.axhline(1.0, color=DIM, lw=0.7, ls="--")
    title = (f"TIC {tic}  ·  Cam {cam_val} CCD {ccd_val}  ·  "
             f"{len(secs)} sectors  ·  "
             f"raw σ={raw_std*100:.2f}%")
    style(ax_top, "flux_offset_loo / correction value", title=title)
    if not legend_added:
        ax_top.legend(fontsize=8, framealpha=0.25, labelcolor=TEXT,
                      facecolor="#0f1c30", edgecolor=DIM, ncol=2, loc="upper right")
        legend_added = True

    # Bottom: corrected series
    ax_bot = fig.add_subplot(gs[2*i + 1])
    ax_bot.axhline(1.0, color=DIM, lw=0.7, ls="--")
    ax_bot.plot(secs, raw / raw.mean(),  color=C_RAW, lw=1.0, alpha=0.6,
                label=f"Raw (normalised)  σ={raw_std*100:.2f}%")
    ax_bot.plot(secs, raw / ccd,         color=C_CCD, lw=1.4, alpha=0.85,
                label=f"÷ sector_ccd_mean  σ={ccd_std*100:.2f}%")
    ax_bot.plot(secs, raw / knn,         color=C_KNN, lw=1.2, alpha=0.85,
                label=f"÷ knn_mean         σ={knn_std*100:.2f}%")
    ax_bot.plot(secs, raw / st,          color=C_ST,  lw=1.8, alpha=1.0,
                label=f"÷ STITCH           σ={st_std*100:.2f}%")
    show_x = (i == n_stars - 1)
    style(ax_bot, "corrected value", show_xlabel=show_x)
    ax_bot.legend(fontsize=8, framealpha=0.25, labelcolor=TEXT,
                  facecolor="#0f1c30", edgecolor=DIM, ncol=4, loc="upper right")

    print(f"TIC {tic}: raw={raw_std*100:.2f}%  ccd={ccd_std*100:.2f}%  "
          f"knn={knn_std*100:.2f}%  stitch={st_std*100:.2f}%")

fig.suptitle(
    "What each correction actually does — four example test stars\n"
    "Top: raw flux_offset_loo (red) with correction vectors overlaid  ·  "
    "Bottom: the corrected series after dividing by each method",
    color=TEXT, fontsize=10, y=1.01)

out = "eval/correction_examples.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
