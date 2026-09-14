"""
Quick within-star CV comparison: KNN model vs v5 model.
Uses the same test split and sector_median correction methodology as eval_ceiling.py.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import zuko
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree

PARQUET  = "training_data_topup_pdc.parquet"
TARGET   = "flux_offset_loo"
MIN_SECTS = 4
TMAG_MAX  = 13.0
N_SAMPLES = 200

# ── Load & filter ──────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", TARGET])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[(df[TARGET] > 0.85) & (df[TARGET] < 1.15)]
df = df[df["n_sectors_total"] >= MIN_SECTS]
df = df[df["tmag"] <= TMAG_MAX]
df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")
df = df.reset_index(drop=True)
print(f"  {len(df):,} records, {df['tic_id'].nunique():,} stars")

for col in ["col", "row", "delta_sub_col", "delta_sub_row", "sector", "tmag",
            "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms", "pdc_noi", "pr_wght2"]:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Shared features (both models) ─────────────────────────────────────────────
_grp = df.groupby(["sector", "cam", "ccd"])[TARGET]
df["sector_ccd_mean_loo"] = (
    (_grp.transform("sum") - df[TARGET]) / (_grp.transform("count") - 1).clip(lower=1)
)

print("Computing KNN …", flush=True)
_K = 20
_knn = np.full(len(df), np.nan, dtype=np.float32)
for (_s, _c, _q), _g in df.groupby(["sector", "cam", "ccd"]):
    _n = len(_g)
    if _n < 2: continue
    _xy = _g[["col", "row"]].values.astype(np.float32)
    _v  = _g[TARGET].values
    _k  = min(_K, _n - 1)
    _, _nn = cKDTree(_xy).query(_xy, k=_k + 1)
    _knn[_g.index.values] = _v[_nn[:, 1:]].mean(axis=1)
df["spatial_knn_mean_loo"] = np.where(np.isnan(_knn), df["sector_ccd_mean_loo"], _knn)
print(f"  spatial_knn_mean_loo: r={df['spatial_knn_mean_loo'].corr(df[TARGET]):.3f}")

# ── v5-only features ──────────────────────────────────────────────────────────
def _poly2d_design(c, r, degree=3):
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            terms.append(c ** (d - i) * r ** i)
    return np.column_stack(terms)

print("Computing polynomial spatial field …", flush=True)
_poly = np.full(len(df), np.nan, dtype=np.float32)
for (_s, _c, _q), _g in df.groupby(["sector", "cam", "ccd"]):
    _idx = _g.index.values
    if len(_g) < 15:
        _poly[_idx] = _g["sector_ccd_mean_loo"].values.astype(np.float32)
        continue
    _cc = _g["col"].values.astype(np.float64) - _g["col"].mean()
    _rr = _g["row"].values.astype(np.float64) - _g["row"].mean()
    _yy = _g[TARGET].values.astype(np.float64)
    _X  = _poly2d_design(_cc, _rr, degree=3)
    try:
        _cf, _, _, _ = np.linalg.lstsq(_X, _yy, rcond=None)
        _res = _yy - _X @ _cf
        _msk = np.abs(_res) <= 3 * (_res.std() or 1e-9)
        if _msk.sum() >= 15:
            _cf, _, _, _ = np.linalg.lstsq(_X[_msk], _yy[_msk], rcond=None)
        _poly[_idx] = np.clip(_X @ _cf, 0.85, 1.15).astype(np.float32)
    except Exception:
        _poly[_idx] = _g["sector_ccd_mean_loo"].values.astype(np.float32)
df["spatial_poly_pred"] = _poly
print(f"  spatial_poly_pred:    r={df['spatial_poly_pred'].corr(df[TARGET]):.3f}")

print("Computing temporal feature …", flush=True)
_lp  = np.full(len(df), np.nan, dtype=np.float64)
_sg  = np.full(len(df), np.nan, dtype=np.float64)
for _tic, _g in df.groupby("tic_id"):
    _gs  = _g.sort_values("sector")
    _idx = _gs.index.values
    if len(_idx) < 2: continue
    _lp[_idx[1:]] = _gs[TARGET].values[:-1]
    _sg[_idx[1:]] = np.diff(_gs["sector"].values)
df["loo_prev"]   = np.where(np.isnan(_lp), df["sector_ccd_mean_loo"].values, _lp)
df["sector_gap"] = np.where(np.isnan(_sg), 99.0, np.clip(_sg, 1, 99))
print(f"  loo_prev:             r={df['loo_prev'].corr(df[TARGET]):.3f}")

# ── Same split as training ─────────────────────────────────────────────────────
sc = (df.groupby("tic_id")["cam"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index().rename(columns={"cam": "dom"}))
tr, tmp = train_test_split(sc["tic_id"], test_size=0.2, stratify=sc["dom"], random_state=42)
tmp_cam = sc[sc["tic_id"].isin(tmp)]["dom"]
_, te   = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)

test_df = df[df["tic_id"].isin(te)].copy().reset_index(drop=True)
print(f"\nTest set: {test_df['tic_id'].nunique():,} stars, {len(test_df):,} rows")

# ── Inference ──────────────────────────────────────────────────────────────────
device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

def run_model(path, sdf):
    ck   = torch.load(path, map_location="cpu", weights_only=False)
    flow = zuko.flows.NSF(**ck["flow_config"])
    flow.load_state_dict(ck["model_state"]); flow.to(device).eval()
    cont   = (sdf[ck["continuous_cols"]] - ck["means"]) / ck["stds"]
    cam_oh = pd.get_dummies(sdf["cam"].astype(int), prefix="cam").reindex(
                 columns=ck["cam_cols"], fill_value=0)
    ccd_oh = pd.get_dummies(sdf["ccd"].astype(int), prefix="ccd").reindex(
                 columns=ck["ccd_cols"], fill_value=0)
    C = torch.tensor(
        pd.concat([cont.reset_index(drop=True),
                   cam_oh.reset_index(drop=True),
                   ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)
    ).to(device)
    with torch.no_grad():
        mu = flow(C).sample((N_SAMPLES,)).squeeze(-1).mean(0).cpu().numpy()
    return mu * ck["y_std"] + ck["y_mean"]

print("\nRunning KNN model …", flush=True)
pred_knn = run_model("stitch_nsf_knn.pt", test_df)

print("Running v5 model …", flush=True)
pred_v5  = run_model("stitch_nsf_v5.pt", test_df)

# ── CV computation ─────────────────────────────────────────────────────────────
test_df["_raw"]    = test_df["sector_median"]
test_df["_oracle"] = test_df["sector_median"] / test_df[TARGET]
test_df["_knn"]    = test_df["sector_median"] / pred_knn
test_df["_v5"]     = test_df["sector_median"] / pred_v5

def cv_stars(sdf, col, min_n=2):
    return sdf.groupby("tic_id")[col].apply(
        lambda x: float(x.std() / x.mean()) if len(x) >= min_n else np.nan)

cv_raw    = cv_stars(test_df, "_raw")
cv_oracle = cv_stars(test_df, "_oracle")
cv_knn    = cv_stars(test_df, "_knn")
cv_v5     = cv_stars(test_df, "_v5")

common = (cv_raw.dropna().index
          .intersection(cv_oracle.dropna().index)
          .intersection(cv_knn.dropna().index)
          .intersection(cv_v5.dropna().index))

cv_raw    = cv_raw[common];    cv_oracle = cv_oracle[common]
cv_knn    = cv_knn[common];    cv_v5     = cv_v5[common]

def eff(cv_m, cv_r=cv_raw, cv_o=cv_oracle):
    return float(((cv_r - cv_m) / (cv_r - cv_o).clip(lower=1e-6)).median()) * 100

print(f"\n{'─'*52}")
print(f"  {'Model':<14}  {'Median CV':>10}  {'Reduction':>10}  {'Efficiency':>10}")
print(f"{'─'*52}")
print(f"  {'Raw':<14}  {cv_raw.median()*100:>9.3f}%  {'—':>10}  {'—':>10}")
print(f"  {'Oracle':<14}  {cv_oracle.median()*100:>9.3f}%  {'—':>10}  {'—':>10}")
print(f"  {'KNN (v4)':<14}  {cv_knn.median()*100:>9.3f}%  "
      f"{(1-cv_knn.median()/cv_raw.median())*100:>9.1f}%  {eff(cv_knn):>9.1f}%")
print(f"  {'v5 (poly+temp)':<14}  {cv_v5.median()*100:>9.3f}%  "
      f"{(1-cv_v5.median()/cv_raw.median())*100:>9.1f}%  {eff(cv_v5):>9.1f}%")
print(f"{'─'*52}")
delta = (cv_knn.median() - cv_v5.median()) * 100
print(f"\n  v5 vs KNN delta: {delta:+.4f}% CV  ({len(common):,} stars)")

# Breakdown by n_sectors
nsec = test_df.groupby("tic_id")["sector"].nunique()[common]
print(f"\n  By n_sectors:")
for lo, hi, label in [(4,6,"4–6"), (7,10,"7–10"), (11,99,"11+")]:
    m = nsec.between(lo, hi)
    tics = nsec[m].index
    if len(tics) < 5: continue
    r = cv_raw[tics].median()*100
    o = cv_oracle[tics].median()*100
    k = cv_knn[tics].median()*100
    v = cv_v5[tics].median()*100
    ek = eff(cv_knn[tics], cv_raw[tics], cv_oracle[tics])
    ev = eff(cv_v5[tics],  cv_raw[tics], cv_oracle[tics])
    print(f"  n={label:<4} ({len(tics):>4} stars): raw={r:.3f}% oracle={o:.3f}%  "
          f"knn={k:.3f}% ({ek:.0f}%)  v5={v:.3f}% ({ev:.0f}%)")

# By camera
print(f"\n  By camera:")
cam_mode = test_df.groupby("tic_id")["cam"].agg(lambda x: x.mode()[0])[common]
for cam in sorted(cam_mode.unique()):
    tics = cam_mode[cam_mode==cam].index
    r = cv_raw[tics].median()*100
    k = cv_knn[tics].median()*100
    v = cv_v5[tics].median()*100
    print(f"  Cam{int(cam)}: n={len(tics):>4}  raw={r:.3f}%  knn={k:.3f}%  v5={v:.3f}%  "
          f"Δ={k-v:+.4f}%")
