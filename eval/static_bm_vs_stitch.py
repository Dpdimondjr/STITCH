"""
Static BM surface correction vs STITCH — scatter comparison.

Roland's image is a time-averaged BM surface (BM = sector_median / TIC-expected).
Applying it as a static correction removes the global spatial systematic but NOT
the per-sector temporal variation. This script measures how much that static
correction actually reduces scatter, and compares it to STITCH.

Three baselines:
  1. Raw            — no correction (std of flux_offset_loo)
  2. Static BM      — Roland's approach: divide by time-averaged BM surface per CCD
  3. Per-star mean  — divide each obs by that star's own mean flux_offset (oracle upper bound
                      for any static correction)
  4. STITCH         — divide by per-sector pred_offset

Usage:
  python3 eval/static_bm_vs_stitch.py stitch_nsf_knn.pt training_data_topup_pdc.parquet
"""

import sys, warnings
import numpy as np
import pandas as pd
import torch, zuko
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_knn.pt")
PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")

print(f"Model: {MODEL}  Parquet: {PARQUET}\n")

# ── Load model ──────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"], bins=cfg["bins"])
flow.load_state_dict(ckpt["model_state"]); flow.eval()
means=ckpt["means"]; stds=ckpt["stds"]; y_mean=ckpt["y_mean"]; y_std=ckpt["y_std"]
CONTINUOUS=ckpt["continuous_cols"]; CAM_COLS=ckpt["cam_cols"]; CCD_COLS=ckpt["ccd_cols"]
device=(torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
flow=flow.to(device)

# ── Load parquet ────────────────────────────────────────────────────────────────
print("Loading parquet …", flush=True)
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median"])
df = df[(df["flux_offset"]>0.85)&(df["flux_offset"]<1.15)]
df = df[df["sector_median"]>0]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col]=df[col].fillna(df[col].median())
df = df.reset_index(drop=True)
print(f"  {df['tic_id'].nunique():,} stars, {len(df):,} sector-records")

# ── Derived features (needed for STITCH) ───────────────────────────────────────
df["perstar_gaia_offset"]=df.groupby("tic_id")["flux_offset"].transform("mean")
_t="flux_offset_loo"
_s=df.groupby(["sector","cam","ccd"])[_t].transform("sum")
_c=df.groupby(["sector","cam","ccd"])[_t].transform("count")
df["sector_ccd_mean_loo"]=(_s-df[_t])/(_c-1).clip(lower=1)
_K=20; _knn=np.full(len(df),np.nan,dtype=np.float32)
for (_sec,_cam,_ccd),_g in df.groupby(["sector","cam","ccd"]):
    _n=len(_g)
    if _n<2: continue
    _co=_g[["col","row"]].values.astype(np.float32); _v=_g[_t].values; _k=min(_K,_n-1)
    _,_nn=cKDTree(_co).query(_co,k=_k+1); _knn[_g.index.values]=_v[_nn[:,1:]].mean(1)
df["spatial_knn_mean_loo"]=_knn
df["spatial_knn_mean_loo"]=df["spatial_knn_mean_loo"].fillna(df["sector_ccd_mean_loo"])
print("Derived features computed.")

# ── Train/test split (same as gaia_locus_validation.py) ────────────────────────
star_cam=(df.groupby("tic_id")["cam"].agg(lambda x:x.mode()[0])
           .reset_index().rename(columns={"cam":"dominant_cam"}))
tr,tmp=train_test_split(star_cam["tic_id"],test_size=0.2,
                        stratify=star_cam["dominant_cam"],random_state=42)
tmp_cam=star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_,te=train_test_split(tmp,test_size=0.5,stratify=tmp_cam.values,random_state=42)

# Held-out test set, ≥3 sectors each
test_df = df[df["tic_id"].isin(te)].copy()
ns = test_df.groupby("tic_id")["sector"].count()
test_df = test_df[test_df["tic_id"].isin(ns[ns>=3].index)].copy().reset_index(drop=True)
print(f"Test set: {test_df['tic_id'].nunique():,} stars, {len(test_df):,} records\n")

# ── Static BM surface (Roland's approach) ──────────────────────────────────────
# Build the surface from TRAINING data only (so test stars are truly held out).
train_df = df[df["tic_id"].isin(tr)].copy()

print("Building static BM surfaces per CCD from training data …")
bm_surfaces = {}  # (cam, ccd) -> interpolator
NBINS = 32

for (cam, ccd), grp in train_df.groupby(["cam","ccd"]):
    # Time-average BM for each star (their per-star mean flux_offset ≈ static position-based BM)
    # bin into 2D grid of col×row
    col_edges = np.linspace(0, 2048, NBINS+1)
    row_edges = np.linspace(0, 2048, NBINS+1)
    col_c = 0.5*(col_edges[:-1]+col_edges[1:])
    row_c = 0.5*(row_edges[:-1]+row_edges[1:])

    # Use ALL observations in each bin (across sectors) — Roland's time-averaged map
    bm_vals = grp["flux_offset"].values   # flux_offset ≈ BM (Gaia-based, same idea as TIC-based)
    col_v   = grp["col"].values
    row_v   = grp["row"].values

    grid = np.full((NBINS, NBINS), np.nan)
    for ci in range(NBINS):
        for ri in range(NBINS):
            mask = ((col_v>=col_edges[ci])&(col_v<col_edges[ci+1])&
                    (row_v>=row_edges[ri])&(row_v<row_edges[ri+1]))
            if mask.sum() >= 3:
                grid[ri, ci] = np.nanmedian(bm_vals[mask])

    # Fill NaN bins with global mean
    global_mean = np.nanmedian(bm_vals)
    grid = np.where(np.isnan(grid), global_mean, grid)

    # Smooth 2D spline interpolator
    spl = RectBivariateSpline(row_c, col_c, grid, kx=3, ky=3)
    bm_surfaces[(cam, ccd)] = (spl, global_mean)

print(f"  Built surfaces for {len(bm_surfaces)} cam/CCD pairs.")

# Apply static BM correction to test set
def apply_static_bm(row):
    key = (int(row["cam"]), int(row["ccd"]))
    if key not in bm_surfaces:
        return 1.0
    spl, gm = bm_surfaces[key]
    pred = float(spl(row["row"], row["col"])[0,0])
    return np.clip(pred, 0.85, 1.15)

print("Applying static BM correction to test set …")
test_df["static_bm"] = test_df.apply(apply_static_bm, axis=1)
test_df["flux_offset_static"] = test_df["flux_offset_loo"] / test_df["static_bm"]

# Per-star mean correction (oracle ceiling for static methods)
test_df["perstar_mean"] = test_df.groupby("tic_id")["flux_offset_loo"].transform("mean")
test_df["flux_offset_perstar"] = test_df["flux_offset_loo"] / test_df["perstar_mean"]

# ── STITCH inference ────────────────────────────────────────────────────────────
print("Running STITCH inference …")
def make_context(d):
    cont=(d[CONTINUOUS].reindex(columns=CONTINUOUS,fill_value=0)-means)/stds
    cam_oh=pd.get_dummies(d["cam"].astype(int),prefix="cam").reindex(columns=CAM_COLS,fill_value=0)
    ccd_oh=pd.get_dummies(d["ccd"].astype(int),prefix="ccd").reindex(columns=CCD_COLS,fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)],axis=1).values.astype("float32")

C=torch.tensor(make_context(test_df)).to(device)
with torch.no_grad():
    samp=flow(C).sample((300,)).squeeze(-1)
    mu_z=samp.mean(0).cpu().numpy(); sig_z=samp.std(0).cpu().numpy()
pred_raw=mu_z*y_std+y_mean; pred_std=sig_z*y_std
weight=1.0/(1.0+5.0*pred_std)
test_df["pred_offset"]=weight*pred_raw+(1-weight)*1.0
test_df["flux_offset_stitch"]=test_df["flux_offset_loo"]/test_df["pred_offset"]
print("Inference done.\n")

# ── Per-star scatter comparison ─────────────────────────────────────────────────
star_stats = test_df.groupby("tic_id").agg(
    std_raw     =("flux_offset_loo",    "std"),
    std_static  =("flux_offset_static", "std"),
    std_perstar =("flux_offset_perstar","std"),
    std_stitch  =("flux_offset_stitch", "std"),
    n_sectors   =("sector","count"),
).reset_index().dropna()
star_stats = star_stats[star_stats["n_sectors"]>=3]

# Global per-observation scatter
obs_raw     = test_df["flux_offset_loo"].values
obs_static  = test_df["flux_offset_static"].values
obs_perstar = test_df["flux_offset_perstar"].values
obs_stitch  = test_df["flux_offset_stitch"].values

def pct_improved(before, after):
    return ((before>after).sum()/len(before)*100)

def reduction(before_vals, after_vals):
    return (1 - np.median(after_vals)/np.median(before_vals))*100

print("="*65)
print(f"{'Method':<22} {'Obs σ%':>8} {'Star med σ%':>12} {'Stars better%':>14}")
print("="*65)
for label, obs, star_col in [
    ("Raw (no correction)", obs_raw,     "std_raw"),
    ("Static BM (Roland)", obs_static,   "std_static"),
    ("Per-star mean",       obs_perstar,  "std_perstar"),
    ("STITCH",             obs_stitch,   "std_stitch"),
]:
    obs_pct   = obs.std()*100
    star_med  = star_stats[star_col].median()*100
    better    = pct_improved(star_stats["std_raw"], star_stats[star_col]) if star_col!="std_raw" else float('nan')
    print(f"  {label:<20} {obs_pct:>8.3f}  {star_med:>12.3f}  {better:>13.1f}%")
print()

print("Scatter reduction vs raw:")
for label, col in [("Static BM (Roland)", "std_static"),
                   ("Per-star mean",       "std_perstar"),
                   ("STITCH",             "std_stitch")]:
    red = reduction(star_stats["std_raw"], star_stats[col])
    print(f"  {label:<22}: {red:.1f}% median reduction in per-star sector scatter")

# ── Figure ──────────────────────────────────────────────────────────────────────
BG,SURF,TEXT,MUT,DIM="#0b0f1a","#0f1826","#c4d4ee","#7a9cc4","#1c2d45"
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF); ax.set_xlabel(xlabel,color=MUT,fontsize=9)
    ax.set_ylabel(ylabel,color=MUT,fontsize=9); ax.set_title(title,color=TEXT,fontsize=10,pad=7)
    ax.tick_params(colors=MUT,labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True,color=DIM,lw=0.3,zorder=0); ax.set_axisbelow(True)

# Panel 1: global obs distribution
ax = axes[0]
bins = np.linspace(0.88, 1.12, 100)
for label, vals, clr, lw in [
    ("Raw",            obs_raw,     "#ef4444", 1.5),
    ("Static BM",      obs_static,  "#f59e0b", 1.5),
    ("Per-star mean",  obs_perstar, "#a78bfa", 1.5),
    ("STITCH",         obs_stitch,  "#4f8ef7", 1.8),
]:
    ax.hist(vals, bins=bins, density=True, histtype="step",
            lw=lw, color=clr, label=f"{label}  σ={vals.std()*100:.3f}%")
ax.axvline(1, color=TEXT, lw=0.8, ls="--", alpha=0.5)
ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
style(ax, "flux_offset_loo / correction", "Density",
      "All observations\n(held-out test set)")

# Panel 2: per-star scatter CDF
ax = axes[1]
for label, col, clr in [
    ("Raw",            "std_raw",     "#ef4444"),
    ("Static BM",      "std_static",  "#f59e0b"),
    ("Per-star mean",  "std_perstar", "#a78bfa"),
    ("STITCH",         "std_stitch",  "#4f8ef7"),
]:
    vals = np.sort(star_stats[col].values)*100
    ax.plot(vals, np.linspace(0,1,len(vals)), color=clr, lw=1.8,
            label=f"{label}  med={np.median(vals):.3f}%")
ax.set_xlim(0, 3.5)
ax.axvline(star_stats["std_raw"].median()*100, color="#ef4444", lw=0.6, ls="--", alpha=0.5)
ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
style(ax, "Per-star sector-to-sector σ (%)", "CDF",
      f"Per-star scatter CDF\n({len(star_stats):,} stars, ≥3 sectors)")

# Panel 3: scatter reduction bar chart
ax = axes[2]
methods = ["Static BM\n(Roland)", "Per-star\nmean", "STITCH"]
colors  = ["#f59e0b", "#a78bfa", "#4f8ef7"]
reductions = []
for col in ["std_static", "std_perstar", "std_stitch"]:
    reductions.append(reduction(star_stats["std_raw"], star_stats[col]))

bars = ax.bar(methods, reductions, color=colors, width=0.5, zorder=3)
for bar, val in zip(bars, reductions):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.3, f"{val:.1f}%",
            ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
ax.set_ylim(0, max(reductions)*1.25)
style(ax, "", "Reduction in per-star σ (%)",
      "Scatter reduction vs raw\n(median across all test stars)")

fig.suptitle(
    "Static BM surface (Roland) vs STITCH — scatter reduction on held-out test stars\n"
    "Static BM: time-averaged 2D surface fit per CCD (no sector knowledge)  ·  "
    "STITCH: per-sector conditional flow prediction",
    color=TEXT, fontsize=9.5, y=1.02)
plt.tight_layout()
out = "eval/static_bm_vs_stitch.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
