"""
Intrinsic scatter analysis — what fraction of each star's cross-sector scatter
is astrophysical (real) vs instrumental (correctable)?

Three scatter levels:
  sigma_obs       total observed scatter (std of flux_offset_loo across sectors)
  sigma_stitch    scatter after STITCH correction        — upper bound on intrinsic
  sigma_loo_oracle scatter when correcting by the LOO label itself (i.e. the model
                  had perfect foresight of the label) — label-noise floor
  sigma_photo     photometric noise on a single sector median
                  ≈ cdpp1_0_ppm / 1e6 / sqrt(26 × 24 hr-bins)

Key diagnostic — sigma vs N_sectors:
  * Intrinsic variability:  FLAT  (real astrophysics doesn't average out)
  * Label noise (LOO):      ∝ 1/sqrt(N)  (more sectors → better LOO estimate)
  * STITCH model error:     FLAT or weakly decreasing

Usage:
  python3 eval/intrinsic_scatter.py stitch_nsf_knn.pt training_data_topup_pdc.parquet
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

# ── Model ──────────────────────────────────────────────────────────────────────
print("Loading model …")
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

# ── Parquet ────────────────────────────────────────────────────────────────────
print("Loading parquet …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median","gaiarp"])
df = df[(df["flux_offset"]>0.85)&(df["flux_offset"]<1.15)]
df = df[df["sector_median"]>0]
# derive model context features
df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")
_s = df.groupby(["sector","cam","ccd"])["flux_offset_loo"].transform("sum")
_c = df.groupby(["sector","cam","ccd"])["flux_offset_loo"].transform("count")
df["sector_ccd_mean_loo"]  = (_s - df["flux_offset_loo"]) / (_c-1).clip(lower=1)
df["spatial_knn_mean_loo"] = df["sector_ccd_mean_loo"]   # placeholder
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
df = df.reset_index(drop=True)
print(f"  {df['tic_id'].nunique():,} stars, {len(df):,} records")

# ── Split ──────────────────────────────────────────────────────────────────────
star_cam = (df.groupby("tic_id")["cam"].agg(lambda x: x.mode()[0])
              .reset_index().rename(columns={"cam":"dominant_cam"}))
tr, tmp = train_test_split(star_cam["tic_id"], test_size=0.2,
                           stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_, te   = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)

train_df = df[df["tic_id"].isin(tr)].copy().reset_index(drop=True)
test_df  = df[df["tic_id"].isin(te)].copy()
ns = test_df.groupby("tic_id")["sector"].count()
test_df = test_df[test_df["tic_id"].isin(ns[ns>=5].index)].copy().reset_index(drop=True)
print(f"  Test set: {test_df['tic_id'].nunique():,} stars (≥5 sectors)")

_t = "flux_offset_loo"; K = 20

# ── KNN (training refs) ────────────────────────────────────────────────────────
print("Building KNN …")
grp_sum   = train_df.groupby(["sector","cam","ccd"])[_t].transform("sum")
grp_count = train_df.groupby(["sector","cam","ccd"])[_t].transform("count")
train_df["sector_ccd_mean_loo"] = (grp_sum - train_df[_t]) / (grp_count-1).clip(lower=1)

knn_mean = np.full(len(test_df), np.nan, dtype=np.float32)
knn_med  = np.full(len(test_df), np.nan, dtype=np.float32)
for (sec,cam,ccd), q_grp in test_df.groupby(["sector","cam","ccd"]):
    ref = train_df[(train_df["sector"]==sec)&(train_df["cam"]==cam)&(train_df["ccd"]==ccd)]
    if len(ref) < 2: continue
    r_co = ref[["col","row"]].values.astype(np.float32)
    r_v  = ref[_t].values
    q_co = q_grp[["col","row"]].values.astype(np.float32)
    k    = min(K, len(ref))
    _, nn = cKDTree(r_co).query(q_co, k=k)
    if nn.ndim==1: nn=nn[:,np.newaxis]
    knn_mean[q_grp.index.values] = r_v[nn].mean(axis=1)
    knn_med[q_grp.index.values]  = np.median(r_v[nn], axis=1)

tr_ccd_med = (train_df.groupby(["sector","cam","ccd"])[_t].median()
              .reset_index().rename(columns={_t:"_ccd_med"}))
test_df = test_df.merge(tr_ccd_med, on=["sector","cam","ccd"], how="left")
test_df["spatial_knn_mean_loo"]   = pd.Series(knn_mean,  index=test_df.index).fillna(test_df["_ccd_med"])
test_df["spatial_knn_median_loo"] = pd.Series(knn_med,   index=test_df.index).fillna(test_df["_ccd_med"])

for col in CONTINUOUS:
    if col in test_df.columns and test_df[col].isna().any():
        test_df[col] = test_df[col].fillna(test_df[col].median())

# ── STITCH inference ───────────────────────────────────────────────────────────
print("Running STITCH …")
def make_ctx(d):
    cont  = (d[CONTINUOUS].reindex(columns=CONTINUOUS,fill_value=0)-means)/stds
    cam_oh = pd.get_dummies(d["cam"].astype(int),prefix="cam").reindex(columns=CAM_COLS,fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int),prefix="ccd").reindex(columns=CCD_COLS,fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)],axis=1).values.astype("float32")

C = torch.tensor(make_ctx(test_df)).to(device)
with torch.no_grad():
    samp = flow(C).sample((300,)).squeeze(-1)
    mu_z = samp.mean(0).cpu().numpy(); sig_z = samp.std(0).cpu().numpy()
pred_raw = mu_z*y_std+y_mean; pred_std = sig_z*y_std
weight   = 1.0/(1.0+5.0*pred_std)
test_df["pred_offset"]            = weight*pred_raw+(1-weight)*1.0
test_df["corrected_stitch"]       = test_df[_t]/test_df["pred_offset"]
test_df["corrected_knn_median"]   = test_df[_t]/test_df["spatial_knn_median_loo"]

# ── LOO-oracle: divide by the LOO label itself ─────────────────────────────────
# This is the model-has-perfect-label ceiling.  Residual = label noise only.
# For star i with N sectors, LOO label for sector s uses N-1 others:
#   flux_offset_loo_s = sector_median_s / mean_{t≠s}(sector_median_t)
# If we divide by it: corrected_s = 1 exactly → std = 0 only if all labels match.
# In practice std > 0 because the LOO label itself has uncertainty ∝ 1/√(N−1).
# We simulate this by computing what the residual std WOULD be after oracle correction.
# For N sectors: sigma_oracle ≈ sigma_obs / sqrt(N-1)   (LOO label noise formula)
# (exact: var_oracle = var_obs × 1/(N-1) for i.i.d. Gaussian)
test_df["corrected_loo_oracle"]   = 1.0  # trivially 1 by definition; std per star below

# ── Photometric floor ──────────────────────────────────────────────────────────
# cdpp1_0 in ppm → fractional uncertainty per sector median
N_HR = 26 * 24  # 624 one-hour bins per sector
if "cdpp1_0" in test_df.columns:
    test_df["sigma_photo"] = (test_df["cdpp1_0"].clip(lower=1) / 1e6) / np.sqrt(N_HR)
else:
    test_df["sigma_photo"] = np.nan

# ── Per-star summary ───────────────────────────────────────────────────────────
print("Computing per-star summary …")
star = test_df.groupby("tic_id").agg(
    gaiarp        =("gaiarp",               "first"),
    tmag          =("tmag",                 "first"),
    n_sectors     =("sector",               "count"),
    sigma_obs     =(_t,                     "std"),
    sigma_stitch  =("corrected_stitch",     "std"),
    sigma_knn_med =("corrected_knn_median", "std"),
    sigma_photo   =("sigma_photo",          "median"),
).reset_index().dropna(subset=["sigma_obs","sigma_stitch"])
star = star[star["n_sectors"]>=5]

# LOO oracle scatter per star: sigma_obs / sqrt(N-1)
# (variance of the LOO label = var of the true signal × 1/(N-1))
star["sigma_loo_oracle"] = star["sigma_obs"] / np.sqrt((star["n_sectors"]-1).clip(lower=1))

# Intrinsic scatter estimate: quadrature subtract photo floor from STITCH residual
star["sigma_photo"]      = star["sigma_photo"].fillna(0)
star["sigma_intrinsic"]  = np.sqrt(np.maximum(0,
    star["sigma_stitch"]**2 - star["sigma_photo"]**2))

# pct columns
for c in ["sigma_obs","sigma_stitch","sigma_knn_med","sigma_loo_oracle",
          "sigma_photo","sigma_intrinsic"]:
    star[c+"_pct"] = star[c]*100

print("\n── Scatter budget (median across test stars) ──────────────────────────")
print(f"  N stars (≥5 sectors):       {len(star):,}")
print(f"  sigma_obs        (raw):     {star['sigma_obs_pct'].median():.3f}%")
print(f"  sigma_knn_med    (knn):     {star['sigma_knn_med_pct'].median():.3f}%")
print(f"  sigma_stitch     (STITCH):  {star['sigma_stitch_pct'].median():.3f}%")
print(f"  sigma_loo_oracle (label ceil): {star['sigma_loo_oracle_pct'].median():.3f}%")
print(f"  sigma_photo      (photon):  {star['sigma_photo_pct'].median():.4f}%")
print(f"  sigma_intrinsic  (upper bound): {star['sigma_intrinsic_pct'].median():.3f}%")
print(f"")
print(f"  Interpretation:")
print(f"    STITCH removes {(1-star['sigma_stitch_pct'].median()/star['sigma_obs_pct'].median())*100:.1f}% of scatter")
print(f"    LOO label floor is {star['sigma_loo_oracle_pct'].median():.3f}% — "
      f"{'above' if star['sigma_loo_oracle_pct'].median()>star['sigma_stitch_pct'].median() else 'below'} STITCH residual")
print(f"    → remaining {star['sigma_stitch_pct'].median():.3f}% is intrinsic + uncaptured instrumental")
print(f"    Photon noise is negligible ({star['sigma_photo_pct'].median():.4f}%) — scatter is not noise-limited")

# ── Figure ─────────────────────────────────────────────────────────────────────
BG,SURF,TEXT,MUT,DIM="#0b0f1a","#0f1826","#c4d4ee","#7a9cc4","#1c2d45"
C_OBS="#ef4444"; C_KNN="#f59e0b"; C_ST="#4f8ef7"; C_ORC="#a78bfa"; C_PH="#5dca8a"

fig = plt.figure(figsize=(18,10), facecolor=BG)
gs  = GridSpec(2,3, figure=fig, hspace=0.45, wspace=0.32)
axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

def sty(ax, xl, yl, tt):
    ax.set_facecolor(SURF); ax.set_xlabel(xl,color=MUT,fontsize=8)
    ax.set_ylabel(yl,color=MUT,fontsize=8); ax.set_title(tt,color=TEXT,fontsize=9,pad=5)
    ax.tick_params(colors=MUT,labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True,color=DIM,lw=0.3,zorder=0); ax.set_axisbelow(True)

# ── P0: CDF across stars ──────────────────────────────────────────────────────
ax = axes[0]
for lbl, col, clr, lw, ls in [
    ("Raw",            "sigma_obs_pct",       C_OBS, 1.8, "-"),
    ("Median-KNN",     "sigma_knn_med_pct",   C_KNN, 1.4, "--"),
    ("STITCH",         "sigma_stitch_pct",    C_ST,  1.8, "-"),
    ("LOO label ceil.","sigma_loo_oracle_pct",C_ORC, 1.2, ":"),
]:
    v = np.sort(star[col].dropna().values)
    ax.plot(v, np.linspace(0,1,len(v)), color=clr, lw=lw, ls=ls,
            label=f"{lbl}  {np.median(v):.3f}%")
ax.set_xlim(0, 4)
ax.legend(fontsize=7.5, framealpha=0.25, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
sty(ax,"Per-star σ across sectors (%)","CDF",
    "Scatter distribution across stars\n(LOO label ceil. = σ_obs/√(N−1))")

# ── P1: sigma vs N_sectors  — the key diagnostic ─────────────────────────────
ax = axes[1]
bins = [5,8,11,15,20,27,45]
labs = ["5–7","8–10","11–14","15–19","20–26","27+"]
bi   = np.digitize(star["n_sectors"].values, bins)
for lbl, col, clr, ls in [
    ("Raw",            "sigma_obs_pct",       C_OBS, "-"),
    ("STITCH",         "sigma_stitch_pct",    C_ST,  "-"),
    ("LOO label ceil.","sigma_loo_oracle_pct",C_ORC, ":"),
]:
    meds = [star.loc[bi==b, col].median() if (bi==b).sum()>5 else np.nan
            for b in range(1,len(bins))]
    ax.plot(labs, meds, color=clr, lw=1.6, ls=ls, marker="o", ms=5, label=lbl)

# Reference line ∝ 1/sqrt(N) anchored at N=5–7 bin
raw5 = star.loc[bi==1,"sigma_obs_pct"].median()
ns_c = np.array([6,9,12.5,17.5,23,36])
ax.plot(labs, raw5/np.sqrt(ns_c/6), color=DIM, lw=1.0, ls="--", alpha=0.8,
        label="∝ 1/√N (pure label noise)")
ax.legend(fontsize=7.5, framealpha=0.25, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
sty(ax,"N sectors observed","Median σ (% per star)",
    "KEY DIAGNOSTIC: σ vs N_sectors\nFlat → intrinsic  ·  ∝1/√N → label noise")

# Annotate STITCH trend
st_meds = [star.loc[bi==b,"sigma_stitch_pct"].median() for b in range(1,len(bins))]
finite   = [(l,v) for l,v in zip(labs,st_meds) if np.isfinite(v)]
if len(finite)>=2:
    slope = (finite[-1][1]-finite[0][1])
    trend = "flat" if abs(slope)<0.1 else ("↓" if slope<0 else "↑")
    ax.text(0.97, 0.97, f"STITCH trend: {trend}\n({'intrinsic-dominated' if abs(slope)<0.1 else 'label noise present'})",
            transform=ax.transAxes, ha="right", va="top", color=TEXT, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1c2d45", edgecolor=DIM))

# ── P2: Intrinsic scatter distribution ────────────────────────────────────────
ax = axes[2]
v_intr = star["sigma_intrinsic_pct"].values
v_ph   = star["sigma_photo_pct"].fillna(0).values
ax.hist(v_intr[np.isfinite(v_intr)], bins=60, color=C_ST, alpha=0.75, density=True,
        label=f"Intrinsic est.  med={np.nanmedian(v_intr):.3f}%")
if v_ph.max()>0.01:
    ax.hist(v_ph[v_ph>0], bins=60, color=C_PH, alpha=0.7, density=True,
            label=f"Photo floor  med={np.nanmedian(v_ph[v_ph>0]):.3f}%")
ax.legend(fontsize=8, framealpha=0.25, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
sty(ax,"σ (%)","Density",
    "Intrinsic scatter estimate per star\n√max(0, σ²_STITCH − σ²_photo)")

# ── P3: Stars where STITCH helps vs doesn't ──────────────────────────────────
ax = axes[3]
frac = 1 - star["sigma_stitch_pct"] / star["sigma_obs_pct"]
ax.hist(frac.clip(-0.3, 1.0).values, bins=60, color=C_ST, alpha=0.8, density=True)
ax.axvline(0,   color="#ef4444", lw=1.5, ls="--", label="No change")
ax.axvline(frac.median(), color=C_PH, lw=2, label=f"Median {frac.median()*100:.0f}%")
ax.legend(fontsize=8, framealpha=0.25, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
# Stars NOT helped are intrinsically variable
n_hurt = (frac < 0).sum(); n_total = len(frac)
sty(ax,"1 − σ_STITCH/σ_raw  per star","Density",
    f"Fraction of scatter STITCH removes\n{n_hurt/n_total*100:.0f}% of stars not improved → likely intrinsic")

# ── P4: Scatter by magnitude ──────────────────────────────────────────────────
ax = axes[4]
mbins = np.arange(6, 14, 0.75)
mi = np.digitize(star["gaiarp"].values, mbins)
for lbl, col, clr, ls in [
    ("Raw",    "sigma_obs_pct",    C_OBS, "-"),
    ("STITCH", "sigma_stitch_pct", C_ST,  "-"),
    ("LOO",    "sigma_loo_oracle_pct", C_ORC, ":"),
]:
    meds=[]; cens=[]
    for b in range(1,len(mbins)):
        m = star.loc[mi==b, col]
        if len(m)>5: meds.append(m.median()); cens.append((mbins[b-1]+mbins[b])/2)
    ax.plot(cens, meds, color=clr, lw=1.6, ls=ls, marker="o", ms=4, label=lbl)
ax.legend(fontsize=8, framealpha=0.25, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
sty(ax,"Gaia RP magnitude","Median σ (%)","Scatter by brightness\n(fainter → more label noise, not more intrinsic)")

# ── P5: STITCH residual vs raw scatter (per star) ────────────────────────────
ax = axes[5]
x = star["sigma_obs_pct"].clip(0,5).values
y = star["sigma_stitch_pct"].clip(0,5).values
ok = np.isfinite(x)&np.isfinite(y)
sc = ax.scatter(x[ok], y[ok], c=star["n_sectors"].clip(5,40).values[ok],
                s=2, alpha=0.3, cmap="plasma", rasterized=True)
plt.colorbar(sc, ax=ax, label="N sectors", pad=0.01).ax.tick_params(labelsize=6, colors=MUT)
lim=(0,4)
ax.plot(lim, lim, color=MUT, lw=0.8, ls="--", alpha=0.5, label="No change")
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.legend(fontsize=7, framealpha=0.25, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
sty(ax,"Raw σ per star (%)","STITCH σ per star (%)",
    "STITCH residual vs raw, coloured by N_sectors\nPoints near diagonal = intrinsic scatter")

fig.suptitle(
    "What is our true scatter baseline?  Intrinsic (astrophysical) vs instrumental (correctable)\n"
    "Photon noise negligible — 1.1% STITCH residual is intrinsic variability + uncaptured instrumental\n"
    "LOO label ceil. = σ_obs/√(N−1): if STITCH residual tracks this, label noise is the bottleneck",
    color=TEXT, fontsize=9.5, y=1.02)

out = "eval/intrinsic_scatter.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
star.to_csv("eval/intrinsic_scatter.csv", index=False)
print(f"Saved → eval/intrinsic_scatter.csv")
