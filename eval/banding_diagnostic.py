"""
Banding diagnostic: after HGB v2 inference, look for horizontal/vertical
stripe structure in the sectors the model still gets badly wrong.

Key idea: KNN-100 is radially symmetric — it can't learn vertical column
stripes or horizontal row stripes that come from read noise, bias rows, or
charge-transfer inefficiency. This script looks for that structure.

Failure definition: per-(tic_id, sector), compute within-star z-score of
the post-model log-residual. High |z| = this sector is anomalous relative
to the star's own correction history across all its other sectors. That
definition is independent of which stars are intrinsically variable.

Usage (from STITCH root):
  python3 eval/banding_diagnostic.py [parquet]
Output: eval/banding_results/
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic

PARQUET   = sys.argv[1] if len(sys.argv) > 1 else "training_data_v3.parquet"
MODEL_DIR = "models/hgb_v2"
OUT_DIR   = "eval/banding_results"
Z_THRESH  = 2.0          # |z| > this flagged as a failure sector
MIN_SECS  = 4            # minimum sectors per star for reliable z-score
N_BINS    = 60           # col/row profile bins

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model artifacts ───────────────────────────────────────────────────────
log("Loading HGB v2 artifacts ...")
model     = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
ref_tics  = set(np.load(os.path.join(MODEL_DIR, "ref_tic_ids.npy")).tolist())
te_tics   = set(np.load(os.path.join(MODEL_DIR, "test_tic_ids.npy")).tolist())
sec_stats = pd.read_parquet(os.path.join(MODEL_DIR, "sec_stats.parquet"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)
best_beta = meta["best_beta"]
FEATS     = meta["features"]
KNN_KS    = meta.get("knn_ks", [5, 10, 20, 50])
log(f"  β={best_beta:.2f}  features={len(FEATS)}D")

# ── Load data ─────────────────────────────────────────────────────────────────
log(f"Loading {PARQUET} ...")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset_loo","sector_median"])
df = df[(df.flux_offset_loo > 0.85) & (df.flux_offset_loo < 1.15)]
df["tic_id"] = df["tic_id"].astype("int64")
df["sector"] = df["sector"].astype("int64")
df = df.reset_index(drop=True)
log(f"  {len(df):,} records · {df.tic_id.nunique():,} stars")

ref_df  = df[ df.tic_id.isin(ref_tics)].copy().reset_index(drop=True)
test_df = df[ df.tic_id.isin(te_tics) ].copy().reset_index(drop=True)
log(f"  Reference: {len(ref_df):,} obs / {ref_df.tic_id.nunique():,} stars")
log(f"  Test:      {len(test_df):,} obs / {test_df.tic_id.nunique():,} stars")

# ── Sector aggregates ─────────────────────────────────────────────────────────
test_df.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"],
             errors="ignore", inplace=True)
merged = test_df.merge(sec_stats, on=["sector","cam","ccd"], how="left")
test_df["sector_mean_loo"]     = merged["sector_mean_loo"].fillna(1.0).values
test_df["sector_std_loo"]      = merged["sector_std_loo"].fillna(0.0).values
test_df["sector_frac_extreme"] = merged["sector_frac_extreme"].fillna(0.0).values

# ── KNN-100 baseline ──────────────────────────────────────────────────────────
log(f"KNN-100 baseline ...  {T()}")
knn_te = np.ones(len(test_df), dtype=np.float64)
for (s, c, q), qg in test_df.groupby(["sector","cam","ccd"]):
    r = ref_df[(ref_df.sector == s) & (ref_df.cam == c) & (ref_df.ccd == q)]
    if len(r) < 2: continue
    coords  = r[["col","row"]].to_numpy(np.float64)
    qcoords = qg[["col","row"]].to_numpy(np.float64)
    rids    = r.tic_id.to_numpy()
    lvals   = r.flux_offset_loo.to_numpy()
    k       = 100
    take    = min(k + int(r.tic_id.value_counts().max()), len(r))
    _, idxs = cKDTree(coords).query(qcoords, k=take)
    if take == 1: idxs = idxs[:, None]
    qids = qg.tic_id.to_numpy()
    for i, (q_id, nn) in enumerate(zip(qids, idxs)):
        keep = rids[nn] != q_id
        vals = lvals[nn[keep][:k]]
        if len(vals) >= 1:
            knn_te[qg.index[i]] = np.median(vals)
test_df["pred_knn"] = knn_te
log(f"  Done  {T()}")

# ── Spatial KNN features ──────────────────────────────────────────────────────
log(f"Spatial features ...  {T()}")
SPATIAL = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]
           + ["ref_count"])
arrs = {c: np.full(len(test_df), np.nan, dtype=np.float32) for c in SPATIAL}
for (s, c, q), gr in ref_df.groupby(["sector","cam","ccd"]):
    ge = test_df[(test_df.sector == s) & (test_df.cam == c) & (test_df.ccd == q)]
    if not len(ge): continue
    xyr = gr[["col","row"]].values.astype(np.float32)
    xye = ge[["col","row"]].values.astype(np.float32)
    vr  = gr.flux_offset_loo.values.astype(np.float32)
    tree = cKDTree(xyr)
    arrs["ref_count"][ge.index.values] = len(gr)
    kk = min(max(KNN_KS), len(gr))
    if kk < 1: continue
    d_all, nn_all = tree.query(xye, k=kk)
    if kk == 1: nn_all = nn_all[:, None]; d_all = d_all[:, None]
    for k in KNN_KS:
        ak = min(k, kk)
        nn_k = nn_all[:, :ak]; d_k = d_all[:, :ak]
        nbv = vr[nn_k]; med = np.median(nbv, axis=1)
        arrs[f"knn{k}_mean"][ge.index.values]   = nbv.mean(axis=1)
        arrs[f"knn{k}_median"][ge.index.values] = med
        arrs[f"knn{k}_mad"][ge.index.values]    = np.abs(nbv - med[:,None]).mean(axis=1)
        arrs[f"knn{k}_radius"][ge.index.values] = d_k.max(axis=1)
for cn, arr in arrs.items():
    test_df[cn] = arr
for cn in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
    test_df[cn] = test_df[cn].fillna(test_df["sector_mean_loo"])
test_df["ref_count"] = test_df["ref_count"].fillna(0.0)
log(f"  Done  {T()}")

# ── Derived features ──────────────────────────────────────────────────────────
fp = test_df.get("focal_plane_temp", pd.Series(np.nan, index=test_df.index))
test_df["temp_x_col"]         = fp.values * (test_df["col"].values / 2200.0)
test_df["temp_x_row"]         = fp.values * (test_df["row"].values / 2200.0)
test_df["temp_x_crowdsap"]    = fp.values * test_df["crowdsap"].values
pt = test_df.get("pdc_totp", pd.Series(np.nan, index=test_df.index))
test_df["pdctotp_x_crowdsap"] = pt.values * test_df["crowdsap"].values
test_df["n_sectors_obs"]      = test_df.groupby("tic_id")["sector"].transform("nunique").astype(float)

X = test_df[FEATS].copy()
for c in X.columns:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())

# ── Inference ─────────────────────────────────────────────────────────────────
log(f"Inference ...  {T()}")
delta = model.predict(X.values.astype(np.float32))
test_df["pred"] = knn_te * np.exp(best_beta * delta)
log(f"  Done  {T()}")

# ── Post-model residuals & within-star z-scores ───────────────────────────────
log("Computing within-star z-scores ...")
test_df["resid"] = np.log(test_df["flux_offset_loo"].values) - np.log(test_df["pred"].values)

def within_star_zscore(df, col="resid", min_s=MIN_SECS):
    z = np.full(len(df), np.nan)
    for tic, grp in df.groupby("tic_id"):
        if grp["sector"].nunique() < min_s:
            continue
        r = grp[col].values
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-8:
            continue
        z[grp.index] = (r - med) / (1.4826 * mad)
    return z

test_df["z"] = within_star_zscore(test_df)
valid = test_df.dropna(subset=["z"])
log(f"  Valid obs: {len(valid):,}  ({valid.tic_id.nunique():,} stars, ≥{MIN_SECS} sectors)")
log(f"  |z| > {Z_THRESH}: {(valid['z'].abs() > Z_THRESH).sum():,}  ({(valid['z'].abs() > Z_THRESH).mean()*100:.1f}%)")

# ── Plot 1: Spatial scatter per cam ──────────────────────────────────────────
log("Plotting spatial scatter ...")

cmap_div = plt.cm.RdBu_r
vmax = 4.0

cams = sorted(valid["cam"].unique())
ccds = sorted(valid["ccd"].unique())
fig, axes = plt.subplots(len(cams), len(ccds), figsize=(14, 12),
                         sharex=False, sharey=False)
fig.patch.set_facecolor("#0D1828")
plt.suptitle("Within-star post-model residual z-score by detector position\n"
             "Look for horizontal (row) or vertical (col) stripe patterns",
             color="white", fontsize=12, y=0.98)

for ci, cam in enumerate(cams):
    for qi, ccd in enumerate(ccds):
        ax = axes[ci][qi]
        ax.set_facecolor("#0D1828")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E2E42")

        sub = valid[(valid.cam == cam) & (valid.ccd == ccd)]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        # all points (grey)
        ax.scatter(sub["col"], sub["row"], c="#2A3D5A", s=1.5, alpha=0.3,
                   rasterized=True, zorder=1)

        # flagged points (colored by z)
        flagged = sub[sub["z"].abs() > Z_THRESH]
        if len(flagged):
            sc = ax.scatter(flagged["col"], flagged["row"],
                            c=flagged["z"], cmap=cmap_div,
                            vmin=-vmax, vmax=vmax, s=4, alpha=0.7,
                            rasterized=True, zorder=2)

        ax.set_title(f"Cam {cam} · CCD {ccd}  (n={len(sub):,})",
                     color="#8A9BB5", fontsize=8, pad=3)
        ax.tick_params(colors="#4E607A", labelsize=7)
        ax.set_xlabel("col", color="#4E607A", fontsize=7)
        ax.set_ylabel("row", color="#4E607A", fontsize=7)

fig.subplots_adjust(hspace=0.35, wspace=0.3)
# shared colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_div, norm=mcolors.Normalize(-vmax, vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
cbar.set_label("within-star z-score", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

out1 = os.path.join(OUT_DIR, "spatial_zscore.png")
fig.savefig(out1, dpi=140, bbox_inches="tight", facecolor="#0D1828")
plt.close(fig)
log(f"  Saved → {out1}")

# ── Plot 2: 1D col and row profiles of mean |z| ──────────────────────────────
log("Plotting col/row banding profiles ...")

fig, axes = plt.subplots(2, len(cams), figsize=(14, 7), sharey=False)
fig.patch.set_facecolor("#0D1828")
plt.suptitle("Mean |z-score| along detector columns and rows\n"
             "Elevated peaks = banding structure the model doesn't capture",
             color="white", fontsize=12, y=0.99)

for ci, cam in enumerate(cams):
    sub = valid[valid.cam == cam]
    if len(sub) == 0:
        continue

    absz = sub["z"].abs().values

    for axis_idx, axis_name, coord in [(0, "Column (col)", "col"),
                                        (1, "Row (row)", "row")]:
        ax = axes[axis_idx][ci]
        ax.set_facecolor("#0D1828")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E2E42")

        coords = sub[coord].values
        cmin, cmax = coords.min(), coords.max()

        # mean |z| per bin
        stat, edges, _ = binned_statistic(coords, absz, statistic="mean", bins=N_BINS,
                                           range=(cmin, cmax))
        cnt,  _,     _ = binned_statistic(coords, absz, statistic="count", bins=N_BINS,
                                           range=(cmin, cmax))
        centers = 0.5 * (edges[:-1] + edges[1:])
        # mask low-count bins
        stat[cnt < 5] = np.nan

        # baseline (overall mean)
        baseline = np.nanmean(stat)
        ax.axhline(baseline, color="#4E607A", lw=1, ls="--", zorder=1)

        # shade region above baseline + 0.5σ (peaks)
        sigma = np.nanstd(stat)
        ax.fill_between(centers, baseline, np.where(stat > baseline + sigma, stat, baseline),
                        alpha=0.35, color="#C97800", zorder=2)
        ax.plot(centers, stat, color="#5090D8", lw=1.2, zorder=3)
        ax.scatter(centers[stat > baseline + sigma], stat[stat > baseline + sigma],
                   color="#C97800", s=12, zorder=4)

        ax.set_title(f"Cam {cam} · {axis_name}", color="#8A9BB5", fontsize=9, pad=3)
        ax.set_xlabel(coord, color="#4E607A", fontsize=8)
        ax.set_ylabel("mean |z|" if ci == 0 else "", color="#4E607A", fontsize=8)
        ax.tick_params(colors="#4E607A", labelsize=7)
        ax.set_xlim(cmin, cmax)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out2 = os.path.join(OUT_DIR, "banding_profiles.png")
fig.savefig(out2, dpi=140, bbox_inches="tight", facecolor="#0D1828")
plt.close(fig)
log(f"  Saved → {out2}")

# ── Plot 3: 2D heatmap of mean |z| across the full focal plane ───────────────
log("Plotting 2D focal-plane z heatmap ...")

fig, axes = plt.subplots(len(cams), len(ccds), figsize=(14, 12))
fig.patch.set_facecolor("#0D1828")
plt.suptitle("2D heatmap of mean |z| — red = model consistently wrong in that region\n"
             "Horizontal or vertical bands indicate structure KNN can't see",
             color="white", fontsize=12, y=0.98)

for ci, cam in enumerate(cams):
    for qi, ccd in enumerate(ccds):
        ax = axes[ci][qi]
        ax.set_facecolor("#0D1828")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E2E42")

        sub = valid[(valid.cam == cam) & (valid.ccd == ccd)]
        if len(sub) < 10:
            ax.set_visible(False)
            continue

        # 2D bin: col × row → mean |z|
        H, xedges, yedges = np.histogram2d(
            sub["col"], sub["row"], bins=30,
            weights=sub["z"].abs()
        )
        N, _, _ = np.histogram2d(sub["col"], sub["row"], bins=30)
        with np.errstate(invalid="ignore"):
            Hmean = np.where(N > 2, H / N, np.nan)

        im = ax.imshow(Hmean.T, origin="lower", aspect="auto",
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap="YlOrRd", vmin=0, vmax=vmax * 0.7, interpolation="nearest")
        ax.set_title(f"Cam {cam} · CCD {ccd}", color="#8A9BB5", fontsize=8, pad=3)
        ax.tick_params(colors="#4E607A", labelsize=7)
        ax.set_xlabel("col", color="#4E607A", fontsize=7)
        ax.set_ylabel("row", color="#4E607A", fontsize=7)

fig.subplots_adjust(hspace=0.35, wspace=0.3)
sm2 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=mcolors.Normalize(0, vmax * 0.7))
sm2.set_array([])
cbar2 = fig.colorbar(sm2, ax=axes, shrink=0.6, pad=0.02)
cbar2.set_label("mean |z|", color="white", fontsize=9)
cbar2.ax.yaxis.set_tick_params(color="white", labelcolor="white")

out3 = os.path.join(OUT_DIR, "2d_heatmap.png")
fig.savefig(out3, dpi=140, bbox_inches="tight", facecolor="#0D1828")
plt.close(fig)
log(f"  Saved → {out3}")

# ── Summary stats ─────────────────────────────────────────────────────────────
log("\n=== BANDING SUMMARY ===")
for cam in cams:
    sub = valid[valid.cam == cam]
    flagged_frac = (sub["z"].abs() > Z_THRESH).mean()
    log(f"  Cam {cam}: {len(sub):,} obs, {flagged_frac*100:.1f}% flagged |z|>{Z_THRESH}")
    # Check if col or row is predictive of |z|
    from scipy.stats import pearsonr
    r_col, p_col = pearsonr(sub["col"], sub["z"].abs())
    r_row, p_row = pearsonr(sub["row"], sub["z"].abs())
    log(f"    r(col, |z|) = {r_col:+.3f}  p={p_col:.3f}")
    log(f"    r(row, |z|) = {r_row:+.3f}  p={p_row:.3f}")

# Save summary
summary = {
    "n_valid_obs": int(len(valid)),
    "n_valid_stars": int(valid.tic_id.nunique()),
    "flagged_frac": float((valid["z"].abs() > Z_THRESH).mean()),
    "z_thresh": Z_THRESH,
    "per_cam": {}
}
for cam in cams:
    sub = valid[valid.cam == cam]
    r_col, _ = pearsonr(sub["col"], sub["z"].abs())
    r_row, _ = pearsonr(sub["row"], sub["z"].abs())
    summary["per_cam"][str(cam)] = {
        "n_obs": int(len(sub)),
        "flagged_frac": float((sub["z"].abs() > Z_THRESH).mean()),
        "r_col": float(r_col),
        "r_row": float(r_row),
    }

import json as _json
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    _json.dump(summary, f, indent=2)

log(f"\nOutputs → {OUT_DIR}/")
log("  spatial_zscore.png  — scatter by cam/ccd, colored by z")
log("  banding_profiles.png — 1D col/row profiles of mean |z|")
log("  2d_heatmap.png       — 2D focal plane mean |z| heatmap")
log("  summary.json         — pearson r(col,|z|) and r(row,|z|) per cam")
log(f"\nDONE  {T()}")
