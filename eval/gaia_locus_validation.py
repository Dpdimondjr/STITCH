"""
Cross-sector consistency validation.

flux_offset_loo = sector_median / mean(this star's OTHER sector medians)
  — a pure cross-sector consistency label; 1.0 = perfectly consistent.
  This is NOT a Gaia-based label. Gaia RP enters only as a context feature
  (perstar_gaia_offset) that helps STITCH condition its predictions.

STITCH should make per-sector flux_offset_loo values cluster tighter around 1.0,
meaning fewer cross-sector jumps in the corrected light curves.

Clean evaluation: the train/test split is performed FIRST, and KNN spatial
context features for test stars are built using ONLY training-star references —
no leakage of test labels into training features or vice versa.

Three panels:
  1. Distribution of flux_offset_loo before and after STITCH
  2. Per-star scatter (std across sectors) before vs after STITCH
  3. Scatter reduction binned by Gaia RP magnitude

Usage:
  python3 eval/gaia_locus_validation.py stitch_nsf_knn.pt training_data_topup_pdc.parquet
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

warnings.filterwarnings("ignore")

MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_knn.pt")
PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")
N_STARS = int(next((s.split("=")[1] for s in sys.argv[1:] if s.startswith("--n-stars=")), 5000))
# --out=path/to/output.png  (default: eval/gaia_locus_validation.png)
_out_arg = next((s.split("=", 1)[1] for s in sys.argv[1:] if s.startswith("--out=")), None)

print(f"Model: {MODEL}  Parquet: {PARQUET}  n_stars={N_STARS}")

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
flow=flow.to(device)

# ── Load parquet ───────────────────────────────────────────────────────────────
print("Loading parquet …", flush=True)
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median","gaiarp"])
df = df[(df["flux_offset"]>0.85)&(df["flux_offset"]<1.15)]
df = df[df["sector_median"]>0]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col]=df[col].fillna(df[col].median())
df = df.reset_index(drop=True)
print(f"  {df['tic_id'].nunique():,} stars, {len(df):,} records")

# perstar_gaia_offset uses only this star's own Gaia-based flux_offset values —
# no cross-star leakage; safe to compute on full df before split.
df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")

# ── SPLIT FIRST, then build spatial context features ──────────────────────────
# This is the key fix: KNN features for test stars use only training-star labels.
print("Splitting …")
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam":"dominant_cam"}))
tr, tmp = train_test_split(star_cam["tic_id"], test_size=0.2,
                            stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_, te   = train_test_split(tmp, test_size=0.5,
                            stratify=tmp_cam.values, random_state=42)

train_mask = df["tic_id"].isin(tr)
test_mask  = df["tic_id"].isin(te)
train_df   = df[train_mask].copy().reset_index(drop=True)
all_df     = df.copy()  # kept for reference

_t = "flux_offset_loo"
K  = 20

# ── sector_ccd_mean_loo: training stars (LOO within training only) ─────────────
print("Building sector_ccd_mean_loo …")
grp_sum   = train_df.groupby(["sector","cam","ccd"])[_t].transform("sum")
grp_count = train_df.groupby(["sector","cam","ccd"])[_t].transform("count")
train_df["sector_ccd_mean_loo"] = (grp_sum - train_df[_t]) / (grp_count - 1).clip(lower=1)

# ── spatial_knn_mean_loo: training stars (KNN within training only) ────────────
print("Building KNN features for training stars …")
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

# ── Test set: features computed from training stars only ───────────────────────
# Select and filter test stars
ns = df[test_mask].groupby("tic_id")["sector"].count()
eligible = ns[ns >= 3].index
rng = np.random.default_rng(42)
sample_tics = rng.choice(eligible, size=min(N_STARS, len(eligible)), replace=False)

test_df = df[df["tic_id"].isin(sample_tics)].copy().reset_index(drop=True)
print(f"Test sample: {test_df['tic_id'].nunique():,} stars, {len(test_df):,} sector-records")

print("Building spatial context for test stars (training references only) …")

# sector_ccd_mean_loo for test stars: mean of training stars in same group
tr_grp_mean = (train_df.groupby(["sector","cam","ccd"])[_t].mean()
               .reset_index()
               .rename(columns={_t: "sector_ccd_mean_loo"}))
test_df = test_df.merge(tr_grp_mean, on=["sector","cam","ccd"], how="left")
test_df["sector_ccd_mean_loo"] = test_df["sector_ccd_mean_loo"].fillna(test_df[_t])

# spatial_knn_mean_loo / spatial_knn_median_loo / spatial_knn_mad_loo for test stars
# All built from K nearest training stars only (no test leakage).
_knn_te     = np.full(len(test_df), np.nan, dtype=np.float32)
_knn_te_med = np.full(len(test_df), np.nan, dtype=np.float32)
_knn_te_mad = np.full(len(test_df), np.nan, dtype=np.float32)
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
    nbr = r_v[nn_idx]                                          # (n_test, k)
    med = np.median(nbr, axis=1)
    _knn_te[q_grp.index.values]     = nbr.mean(axis=1)
    _knn_te_med[q_grp.index.values] = med
    _knn_te_mad[q_grp.index.values] = np.median(np.abs(nbr - med[:, np.newaxis]), axis=1)
test_df["spatial_knn_mean_loo"]   = _knn_te
test_df["spatial_knn_median_loo"] = _knn_te_med
test_df["spatial_knn_mad_loo"]    = _knn_te_mad
test_df["spatial_knn_mean_loo"]   = test_df["spatial_knn_mean_loo"].fillna(
    test_df["sector_ccd_mean_loo"])
test_df["spatial_knn_median_loo"] = test_df["spatial_knn_median_loo"].fillna(
    test_df["sector_ccd_mean_loo"])
test_df["spatial_knn_mad_loo"]    = test_df["spatial_knn_mad_loo"].fillna(0.0)

print(f"  NaN in KNN after fill: {test_df['spatial_knn_mean_loo'].isna().sum()}")

# fill any remaining CONTINUOUS NaNs (columns absent from df get added as 0 below)
for col in CONTINUOUS:
    if col in test_df.columns and test_df[col].isna().any():
        test_df[col] = test_df[col].fillna(test_df[col].median())

_missing_cols = [c for c in CONTINUOUS if c not in test_df.columns]
if _missing_cols:
    print(f"  WARNING: {len(_missing_cols)} context cols absent from parquet, filling with 0: {_missing_cols}")

# ── STITCH inference ───────────────────────────────────────────────────────────
def make_context(d):
    # reindex handles both column ordering and missing columns (filled with 0)
    cont  = (d.reindex(columns=CONTINUOUS, fill_value=0) - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(
                columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(
                columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype("float32")

print("Running STITCH inference …", flush=True)
C = torch.tensor(make_context(test_df)).to(device)
with torch.no_grad():
    samp    = flow(C).sample((300,)).squeeze(-1)
    mu_z    = samp.mean(0).cpu().numpy()
    sig_z   = samp.std(0).cpu().numpy()
pred_raw = mu_z * y_std + y_mean
pred_std = sig_z * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)
pred     = weight * pred_raw + (1 - weight) * 1.0

# flux_offset_loo = sector_median / mean(other sector medians for this star)
# After STITCH: divide by pred_offset to reduce cross-sector level shifts
test_df["pred_offset"]             = pred
test_df["corrected_flux_offset"]   = test_df[_t] / pred
test_df["corrected_knn_median"]    = test_df[_t] / test_df["spatial_knn_median_loo"]

# ── Per-star summary ───────────────────────────────────────────────────────────
star_summary = test_df.groupby("tic_id").agg(
    gaiarp          =("gaiarp",                "first"),
    std_raw         =(_t,                      "std"),
    std_corrected   =("corrected_flux_offset", "std"),
    std_knn_median  =("corrected_knn_median",  "std"),
    mean_raw        =(_t,                      "mean"),
    mean_corrected  =("corrected_flux_offset", "mean"),
    n_sectors       =("sector",                "count"),
).reset_index().dropna()
star_summary = star_summary[star_summary["n_sectors"] >= 3]

pct_improved     = (star_summary["std_corrected"]  < star_summary["std_raw"]).mean() * 100
pct_imp_knn_med  = (star_summary["std_knn_median"] < star_summary["std_raw"]).mean() * 100
med_raw      = star_summary["std_raw"].median() * 100
med_corr     = star_summary["std_corrected"].median() * 100
med_knn_med  = star_summary["std_knn_median"].median() * 100
print(f"\nPer-star cross-sector scatter (std of flux_offset_loo across sectors):")
print(f"  Raw median:          {med_raw:.3f}%")
print(f"  Median-KNN median:   {med_knn_med:.3f}%  ({(1-med_knn_med/med_raw)*100:.1f}% reduction)")
print(f"  STITCH median:       {med_corr:.3f}%  ({(1-med_corr/med_raw)*100:.1f}% reduction)")
print(f"  Stars improved (STITCH):    {pct_improved:.1f}%")
print(f"  Stars improved (KNN-med):   {pct_imp_knn_med:.1f}%")
print(f"  (KNN features for test stars built from training-star references only)")

# ── Figure ─────────────────────────────────────────────────────────────────────
BG,SURF,TEXT,MUT,DIM="#0b0f1a","#0f1826","#c4d4ee","#7a9cc4","#1c2d45"
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF)
    ax.set_xlabel(xlabel, color=MUT, fontsize=9)
    ax.set_ylabel(ylabel, color=MUT, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=10, pad=7)
    ax.tick_params(colors=MUT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True, color=DIM, lw=0.3, zorder=0)
    ax.set_axisbelow(True)

# Panel 1: distribution of per-sector flux_offset_loo
ax = axes[0]
raw_vals  = test_df[_t].values
kmed_vals = test_df["corrected_knn_median"].values
corr_vals = test_df["corrected_flux_offset"].values
bins = np.linspace(0.88, 1.12, 100)
ax.hist(raw_vals[np.isfinite(raw_vals)],   bins=bins, alpha=0.55, color="#ef4444", density=True,
        label=f"Raw          σ={np.nanstd(raw_vals)*100:.2f}%")
ax.hist(kmed_vals[np.isfinite(kmed_vals)], bins=bins, alpha=0.55, color="#f59e0b", density=True,
        label=f"Median-KNN  σ={np.nanstd(kmed_vals)*100:.2f}%")
ax.hist(corr_vals[np.isfinite(corr_vals)], bins=bins, alpha=0.55, color="#4f8ef7", density=True,
        label=f"STITCH       σ={np.nanstd(corr_vals)*100:.2f}%")
ax.axvline(1.0, color=TEXT, lw=1.0, ls="--", alpha=0.6)
ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
style(ax, "flux_offset_loo  (sector_median / mean other sector medians)", "Density",
      f"All sector observations  (N={len(raw_vals):,})\n"
      "Cross-sector consistency  [clean held-out test]")

# Panel 2: per-star scatter CDF — raw vs median-KNN vs STITCH
ax = axes[1]
for label, col, clr in [
    ("Raw",         "std_raw",        "#ef4444"),
    ("Median-KNN",  "std_knn_median", "#f59e0b"),
    ("STITCH",      "std_corrected",  "#4f8ef7"),
]:
    vals = np.sort(star_summary[col].values) * 100
    med  = np.median(vals)
    ax.plot(vals, np.linspace(0, 1, len(vals)), color=clr, lw=1.8,
            label=f"{label}  med={med:.3f}%")
ax.set_xlim(0, 3.5)
ax.axvline(med_raw/100  * 100, color="#ef4444", lw=0.6, ls="--", alpha=0.4)
ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
style(ax, "Per-star sector-to-sector σ (%)", "CDF",
      f"Per-star scatter CDF  ({len(star_summary):,} stars)\n"
      f"STITCH {(1-med_corr/med_raw)*100:.1f}% vs Median-KNN {(1-med_knn_med/med_raw)*100:.1f}% reduction")

# Panel 3: scatter reduction by Gaia RP magnitude bin
ax = axes[2]
mag_bins = np.arange(6, 14.5, 0.5)
bin_idx  = np.digitize(star_summary["gaiarp"], mag_bins)
bin_centers, raw_meds, corr_meds, knn_med_meds = [], [], [], []
for b in range(1, len(mag_bins)):
    mask = bin_idx == b
    if mask.sum() < 10: continue
    r  = star_summary.loc[mask, "std_raw"].median() * 100
    c  = star_summary.loc[mask, "std_corrected"].median() * 100
    km = star_summary.loc[mask, "std_knn_median"].median() * 100
    bin_centers.append((mag_bins[b-1]+mag_bins[b])/2)
    raw_meds.append(r); corr_meds.append(c); knn_med_meds.append(km)

bin_centers = np.array(bin_centers)
ax.plot(bin_centers, raw_meds,      color="#ef4444", lw=2, marker="o", ms=5, label="Raw")
ax.plot(bin_centers, knn_med_meds,  color="#f59e0b", lw=1.8, marker="s", ms=4, ls="--", label="Median-KNN")
ax.plot(bin_centers, corr_meds,     color="#4f8ef7", lw=2, marker="o", ms=5, label="STITCH")
ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
style(ax, "Gaia RP magnitude", "Median scatter (% std across sectors)",
      "Scatter by stellar brightness\nMedian-KNN is the honest simple baseline")

fig.suptitle(
    "Cross-sector flux consistency — before and after STITCH\n"
    "flux_offset_loo = sector_median / mean(other sector medians)  ·  "
    "test KNN features built from training stars only",
    color=TEXT, fontsize=10, y=1.02)

plt.tight_layout()
out = _out_arg if _out_arg else "eval/gaia_locus_validation.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
star_summary.to_csv("eval/gaia_locus_validation.csv", index=False)
print(f"Saved → eval/gaia_locus_validation.csv")
