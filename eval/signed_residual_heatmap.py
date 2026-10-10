"""
Signed residual heatmaps per sector.

Shows mean SIGNED residual (not |z|) in (col, row) space, broken out by
individual sector, for Cam 4 CCD 3 and Cam 3. If the banding structure
carries a consistent direction in specific sectors, it shows up here as a
coherent red/blue region rather than noise.

Usage:
  python3 eval/signed_residual_heatmap.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
from scipy.spatial import cKDTree

MODEL_DIR = "models/hgb_v2"
OUT_DIR   = "eval/banding_results"
PARQUET   = "training_data_v3_tic.parquet"
PARQUET2  = "training_data_v3.parquet"
N_BINS    = 20
MIN_STARS = 10   # minimum stars per sector to plot it

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
print("Loading artifacts ...")
model    = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
ref_tics = set(np.load(os.path.join(MODEL_DIR, "ref_tic_ids.npy")).tolist())
te_tics  = set(np.load(os.path.join(MODEL_DIR, "test_tic_ids.npy")).tolist())
vl_tics  = set(np.load(os.path.join(MODEL_DIR, "val_tic_ids.npy")).tolist())
sec_stats = pd.read_parquet(os.path.join(MODEL_DIR, "sec_stats.parquet"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)
best_beta = meta["best_beta"]
FEATS     = meta["features"]
KNN_KS    = [5, 10, 20, 50]

# ── Load data ─────────────────────────────────────────────────────────────────
parquet = PARQUET if os.path.exists(PARQUET) else PARQUET2
print(f"Loading {parquet} ...")
df = pd.read_parquet(parquet)
df = df.dropna(subset=["col","row","flux_offset_loo"])
df = df[(df.flux_offset_loo > 0.85) & (df.flux_offset_loo < 1.15)]
df["tic_id"] = df["tic_id"].astype("int64")
df["sector"] = df["sector"].astype("int64")
df = df.reset_index(drop=True)

ref_df  = df[df.tic_id.isin(ref_tics)].copy().reset_index(drop=True)
# use val+test combined for more stars per cell
eval_df = df[df.tic_id.isin(vl_tics | te_tics)].copy().reset_index(drop=True)
print(f"  ref={len(ref_df):,}  eval={len(eval_df):,}")

# ── Inference ─────────────────────────────────────────────────────────────────
print("Running inference ...")

# sector stats
eval_df.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"],
             errors="ignore", inplace=True)
m = eval_df.merge(sec_stats, on=["sector","cam","ccd"], how="left")
eval_df["sector_mean_loo"]     = m["sector_mean_loo"].fillna(1.0).values
eval_df["sector_std_loo"]      = m["sector_std_loo"].fillna(0.0).values
eval_df["sector_frac_extreme"] = m["sector_frac_extreme"].fillna(0.0).values

# KNN-100
knn = np.ones(len(eval_df), dtype=np.float64)
for (s, c, ccd_), qg in eval_df.groupby(["sector","cam","ccd"]):
    r = ref_df[(ref_df.sector == s) & (ref_df.cam == c) & (ref_df.ccd == ccd_)]
    if len(r) < 2: continue
    rids = r.tic_id.to_numpy(); lvals = r.flux_offset_loo.to_numpy()
    k = min(101, len(r))
    _, idxs = cKDTree(r[["col","row"]].to_numpy(np.float64)).query(
        qg[["col","row"]].to_numpy(np.float64), k=k)
    if k == 1: idxs = idxs[:, None]
    for i, nn in enumerate(zip(idxs)):
        nn = nn[0]
        keep = rids[nn] != qg.tic_id.iloc[i]
        vals = lvals[nn[keep][:100]]
        if len(vals) >= 1: knn[qg.index[i]] = np.median(vals)

# spatial features
SPATIAL = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]
           + ["ref_count"])
arrs = {c: np.full(len(eval_df), np.nan, dtype=np.float32) for c in SPATIAL}
for (s, c, ccd_), gr in ref_df.groupby(["sector","cam","ccd"]):
    ge = eval_df[(eval_df.sector == s) & (eval_df.cam == c) & (eval_df.ccd == ccd_)]
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
        ak = min(k, kk); nn_k = nn_all[:, :ak]; d_k = d_all[:, :ak]
        nbv = vr[nn_k]; med = np.median(nbv, axis=1)
        arrs[f"knn{k}_mean"][ge.index.values]   = nbv.mean(axis=1)
        arrs[f"knn{k}_median"][ge.index.values] = med
        arrs[f"knn{k}_mad"][ge.index.values]    = np.abs(nbv - med[:,None]).mean(axis=1)
        arrs[f"knn{k}_radius"][ge.index.values] = d_k.max(axis=1)
for cn, arr in arrs.items():
    eval_df[cn] = arr
for cn in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
    eval_df[cn] = eval_df[cn].fillna(eval_df["sector_mean_loo"])
eval_df["ref_count"] = eval_df["ref_count"].fillna(0.0)

fp = eval_df.get("focal_plane_temp", pd.Series(np.nan, index=eval_df.index))
eval_df["temp_x_col"]         = fp.values * (eval_df["col"].values / 2200.0)
eval_df["temp_x_row"]         = fp.values * (eval_df["row"].values / 2200.0)
eval_df["temp_x_crowdsap"]    = fp.values * eval_df["crowdsap"].values
pt = eval_df.get("pdc_totp", pd.Series(np.nan, index=eval_df.index))
eval_df["pdctotp_x_crowdsap"] = pt.values * eval_df["crowdsap"].values
eval_df["n_sectors_obs"]      = eval_df.groupby("tic_id")["sector"].transform("nunique").astype(float)

X = eval_df[FEATS].copy()
for col in X.columns:
    if X[col].isna().any(): X[col] = X[col].fillna(X[col].median())
delta = model.predict(X.values.astype(np.float32))
pred  = knn * np.exp(best_beta * delta)
eval_df["pred"]  = pred
eval_df["resid"] = np.log(eval_df["flux_offset_loo"].values) - np.log(pred)
print(f"  Done. mean resid={eval_df['resid'].mean():.4f}  std={eval_df['resid'].std():.4f}")

# ── Plotting helper ────────────────────────────────────────────────────────────
def sector_signed_heatmaps(sub, label, fname, n_bins=N_BINS):
    """
    For each sector in sub, plot a 2D heatmap of mean signed residual
    in (col, row) bins.  Diverging scale centered at 0.
    """
    sectors = sorted(sub["sector"].unique())
    # filter to sectors with enough stars
    sectors = [s for s in sectors if sub[sub.sector == s]["tic_id"].nunique() >= MIN_STARS]
    if not sectors:
        print(f"  {label}: no sectors with ≥{MIN_STARS} stars")
        return

    ncols = 5
    nrows = int(np.ceil(len(sectors) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    # global scale: 2nd/98th percentile of all residuals in this subgroup
    vmax = np.percentile(np.abs(sub["resid"].values), 98)
    vmax = max(vmax, 0.005)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = "RdBu_r"   # red = over-correction (resid < 0), blue = under-correction

    for ax, sector in zip(axes, sectors):
        s = sub[sub.sector == sector]
        col_edges = np.linspace(s["col"].min(), s["col"].max(), n_bins + 1)
        row_edges = np.linspace(s["row"].min(), s["row"].max(), n_bins + 1)

        col_bins = pd.cut(s["col"], bins=col_edges, labels=False)
        row_bins = pd.cut(s["row"], bins=row_edges, labels=False)
        tmp = s.copy()
        tmp["col_bin"] = col_bins.fillna(0).astype(int)
        tmp["row_bin"] = row_bins.fillna(0).astype(int)
        grid = tmp.groupby(["row_bin","col_bin"])["resid"].mean().unstack(fill_value=np.nan)
        # ensure full grid
        full = pd.DataFrame(np.nan, index=range(n_bins), columns=range(n_bins))
        full.update(grid)
        img = full.values

        im = ax.imshow(img, origin="lower", aspect="auto",
                       cmap=cmap, norm=norm,
                       extent=[col_edges[0], col_edges[-1],
                               row_edges[0], row_edges[-1]])
        n_stars = s["tic_id"].nunique()
        ax.set_title(f"S{sector}  n={n_stars}", fontsize=8)
        ax.set_xlabel("col", fontsize=7)
        ax.set_ylabel("row", fontsize=7)
        ax.tick_params(labelsize=6)

    # hide unused axes
    for ax in axes[len(sectors):]:
        ax.set_visible(False)

    # colorbar
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes[:len(sectors)], fraction=0.02, pad=0.02,
                 label="mean signed residual  (blue=under-corrected, red=over-corrected)")
    fig.suptitle(f"{label} — signed residual per sector", fontsize=11, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)
    print(f"  Saved → {os.path.join(OUT_DIR, fname)}  ({len(sectors)} sectors)")

def row_profile_per_sector(sub, label, fname, n_bins=30):
    """
    For each sector: bin by row, compute mean signed residual.
    Overlay all sectors as colored lines to show consistency.
    """
    sectors = sorted(sub["sector"].unique())
    sectors = [s for s in sectors if sub[sub.sector == s]["tic_id"].nunique() >= MIN_STARS]
    if not sectors: return

    fig, ax = plt.subplots(figsize=(9, 4))
    cmap_lines = plt.cm.viridis(np.linspace(0, 1, len(sectors)))
    for i, sector in enumerate(sectors):
        s = sub[sub.sector == sector]
        row_edges = np.linspace(s["row"].quantile(0.01), s["row"].quantile(0.99), n_bins + 1)
        centers = 0.5 * (row_edges[:-1] + row_edges[1:])
        bins = pd.cut(s["row"], bins=row_edges, labels=False)
        means = s["resid"].groupby(bins).mean().reindex(range(n_bins))
        ax.plot(centers, means.values, color=cmap_lines[i], alpha=0.7,
                linewidth=1.2, label=f"S{sector}")

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("row position", fontsize=10)
    ax.set_ylabel("mean signed residual", fontsize=10)
    ax.set_title(f"{label} — row-profile per sector", fontsize=11, fontweight="bold")
    if len(sectors) <= 20:
        ax.legend(fontsize=6, ncol=4, loc="upper right")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=mcolors.Normalize(vmin=min(sectors), vmax=max(sectors)))
    fig.colorbar(sm, ax=ax, label="sector")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)
    print(f"  Saved → {os.path.join(OUT_DIR, fname)}")

# ── Cam 4 CCD 3 ──────────────────────────────────────────────────────────────
print("\nCam 4 CCD 3 ...")
c4c3 = eval_df[(eval_df.cam == 4) & (eval_df.ccd == 3)]
sector_signed_heatmaps(c4c3, "Cam 4 CCD 3", "cam4ccd3_signed_heatmaps.png")
row_profile_per_sector(c4c3, "Cam 4 CCD 3", "cam4ccd3_row_profiles.png")

# ── Cam 3 all CCDs ────────────────────────────────────────────────────────────
print("\nCam 3 (all CCDs) ...")
for ccd_ in sorted(eval_df[eval_df.cam == 3]["ccd"].unique()):
    sub = eval_df[(eval_df.cam == 3) & (eval_df.ccd == ccd_)]
    sector_signed_heatmaps(sub, f"Cam 3 CCD {ccd_:.0f}", f"cam3ccd{ccd_:.0f}_signed_heatmaps.png")
    row_profile_per_sector(sub, f"Cam 3 CCD {ccd_:.0f}", f"cam3ccd{ccd_:.0f}_row_profiles.png")

# ── Summary stats: per-sector mean residual at high-|z| rows ──────────────────
print("\nPer-sector signed residual stats (Cam 4 CCD 3):")
for sector in sorted(c4c3["sector"].unique()):
    s = c4c3[c4c3.sector == sector]
    if s["tic_id"].nunique() < MIN_STARS: continue
    # find rows with visually-banded structure: top/bottom quartile of row
    q25, q75 = s["row"].quantile(0.25), s["row"].quantile(0.75)
    low  = s[s["row"] < q25]["resid"].mean()
    mid  = s[(s["row"] >= q25) & (s["row"] <= q75)]["resid"].mean()
    high = s[s["row"] > q75]["resid"].mean()
    spread = max(abs(low - mid), abs(high - mid), abs(high - low))
    print(f"  S{sector:2d}: n={s['tic_id'].nunique():3d}  "
          f"low={low:+.4f}  mid={mid:+.4f}  high={high:+.4f}  "
          f"spread={spread:.4f}{'  <<' if spread > 0.003 else ''}")

print("\nDONE")
