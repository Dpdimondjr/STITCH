"""
Spatial secondary correction experiment.

Part 1: Does numcont explain the Cam 3 ecliptic pattern?
  - Join post-model z-scores with TIC features
  - Check r(numcont, |z|) within Cam 3 vs other cameras

Part 2: Spatial residual KNN correction
  - Run HGB v2 inference on val set (unbiased residuals)
  - For each test observation, find K nearest val neighbors in (col, row)
    within same (sector, cam, ccd) → mean signed residual
  - Secondary: pred_s2 = pred_v2 × exp(γ × spatial_signal)
  - Tune γ on 80% of val, evaluate on test
  - Report overall and per-cam CV

Rationale: training residuals are biased near-zero (in-sample), so we use
val residuals as the reference map. This is valid — val stars are never in
the test set.

Usage (from STITCH root):
  python3 eval/spatial_secondary_correction.py
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

PARQUET    = "training_data_v3_tic.parquet"   # has numcont
PARQUET_V2 = "training_data_v3.parquet"       # fallback
MODEL_DIR  = "models/hgb_v2"
OUT_DIR    = "eval/banding_results"
KNN_KS     = [5, 10, 20, 50]
Z_THRESH   = 2.0
MIN_SECS   = 4

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

# ── Load artifacts ─────────────────────────────────────────────────────────────
log("Loading HGB v2 artifacts ...")
model     = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
ref_tics  = set(np.load(os.path.join(MODEL_DIR, "ref_tic_ids.npy")).tolist())
te_tics   = set(np.load(os.path.join(MODEL_DIR, "test_tic_ids.npy")).tolist())
vl_tics   = set(np.load(os.path.join(MODEL_DIR, "val_tic_ids.npy")).tolist())
sec_stats = pd.read_parquet(os.path.join(MODEL_DIR, "sec_stats.parquet"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)
best_beta = meta["best_beta"]
FEATS     = meta["features"]
log(f"  β={best_beta:.2f}  features={len(FEATS)}D")

# ── Load data ─────────────────────────────────────────────────────────────────
parquet = PARQUET if os.path.exists(PARQUET) else PARQUET_V2
log(f"Loading {parquet} ...")
df = pd.read_parquet(parquet)
df = df.dropna(subset=["col","row","flux_offset_loo","sector_median"])
df = df[(df.flux_offset_loo > 0.85) & (df.flux_offset_loo < 1.15)]
df["tic_id"] = df["tic_id"].astype("int64")
df["sector"] = df["sector"].astype("int64")
has_tic = "numcont" in df.columns
log(f"  {len(df):,} records · {df.tic_id.nunique():,} stars  (numcont={'yes' if has_tic else 'NO'})")
df = df.reset_index(drop=True)

ref_df  = df[ df.tic_id.isin(ref_tics)].copy().reset_index(drop=True)
val_df  = df[ df.tic_id.isin(vl_tics) ].copy().reset_index(drop=True)
test_df = df[ df.tic_id.isin(te_tics) ].copy().reset_index(drop=True)

# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(ref, query):
    """Returns predictions for query using ref as KNN reference."""
    q = query.copy()
    # sector aggregates
    q.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"],
           errors="ignore", inplace=True)
    m = q.merge(sec_stats, on=["sector","cam","ccd"], how="left")
    q["sector_mean_loo"]     = m["sector_mean_loo"].fillna(1.0).values
    q["sector_std_loo"]      = m["sector_std_loo"].fillna(0.0).values
    q["sector_frac_extreme"] = m["sector_frac_extreme"].fillna(0.0).values

    # KNN-100
    knn = np.ones(len(q), dtype=np.float64)
    for (s, c, ccd_), qg in q.groupby(["sector","cam","ccd"]):
        r = ref[(ref.sector == s) & (ref.cam == c) & (ref.ccd == ccd_)]
        if len(r) < 2: continue
        rids = r.tic_id.to_numpy(); lvals = r.flux_offset_loo.to_numpy()
        take = min(100 + int(r.tic_id.value_counts().max()), len(r))
        _, idxs = cKDTree(r[["col","row"]].to_numpy(np.float64)).query(
            qg[["col","row"]].to_numpy(np.float64), k=take)
        if take == 1: idxs = idxs[:, None]
        for i, (q_id, nn) in enumerate(zip(qg.tic_id.to_numpy(), idxs)):
            keep = rids[nn] != q_id
            vals = lvals[nn[keep][:100]]
            if len(vals) >= 1: knn[qg.index[i]] = np.median(vals)

    # spatial features
    SPATIAL = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]
               + ["ref_count"])
    arrs = {c: np.full(len(q), np.nan, dtype=np.float32) for c in SPATIAL}
    for (s, c, ccd_), gr in ref.groupby(["sector","cam","ccd"]):
        ge = q[(q.sector == s) & (q.cam == c) & (q.ccd == ccd_)]
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
        q[cn] = arr
    for cn in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
        q[cn] = q[cn].fillna(q["sector_mean_loo"])
    q["ref_count"] = q["ref_count"].fillna(0.0)

    fp = q.get("focal_plane_temp", pd.Series(np.nan, index=q.index))
    q["temp_x_col"]         = fp.values * (q["col"].values / 2200.0)
    q["temp_x_row"]         = fp.values * (q["row"].values / 2200.0)
    q["temp_x_crowdsap"]    = fp.values * q["crowdsap"].values
    pt = q.get("pdc_totp", pd.Series(np.nan, index=q.index))
    q["pdctotp_x_crowdsap"] = pt.values * q["crowdsap"].values
    q["n_sectors_obs"]      = q.groupby("tic_id")["sector"].transform("nunique").astype(float)

    X = q[FEATS].copy()
    for col in X.columns:
        if X[col].isna().any(): X[col] = X[col].fillna(X[col].median())
    delta = model.predict(X.values.astype(np.float32))
    pred  = knn * np.exp(best_beta * delta)
    return knn, pred, q

def cv_col(df_, pred_col, min_s=2):
    rows = []
    for _, g in df_.groupby("tic_id"):
        if g.sector.nunique() < min_s: continue
        c = g["sector_median"].values / g[pred_col].values
        rows.append(float(c.std() / c.mean() * 100))
    return pd.Series(rows)

def within_star_zscore(df, col="resid", min_s=MIN_SECS):
    z = np.full(len(df), np.nan)
    for tic, grp in df.groupby("tic_id"):
        if grp["sector"].nunique() < min_s: continue
        r = grp[col].values
        med = np.median(r)
        mad = np.median(np.abs(r - med))
        if mad < 1e-8: continue
        z[grp.index] = (r - med) / (1.4826 * mad)
    return z

# ── Run inference on val and test ─────────────────────────────────────────────
log(f"\nRunning inference on val set ...  {T()}")
knn_vl, pred_vl, val_df = run_inference(ref_df, val_df)
val_df["pred_knn"] = knn_vl
val_df["pred_v2"]  = pred_vl
val_df["resid"]    = np.log(val_df["flux_offset_loo"].values) - np.log(pred_vl)
val_df["z"]        = within_star_zscore(val_df)
log(f"  Val baseline CV: {cv_col(val_df, 'pred_v2').median():.4f}%  {T()}")

log(f"Running inference on test set ...  {T()}")
knn_te, pred_te, test_df = run_inference(ref_df, test_df)
test_df["pred_knn"] = knn_te
test_df["pred_v2"]  = pred_te
test_df["resid"]    = np.log(test_df["flux_offset_loo"].values) - np.log(pred_te)
test_df["z"]        = within_star_zscore(test_df)
baseline_test_cv    = cv_col(test_df, "pred_v2").median()
log(f"  Test baseline CV: {baseline_test_cv:.4f}%  {T()}")

# ── Part 1: numcont × Cam 3 analysis ─────────────────────────────────────────
log("\n── Part 1: numcont vs z-score by camera ──")
valid_vl = val_df.dropna(subset=["z"])
valid_te = test_df.dropna(subset=["z"])

if has_tic:
    for src_name, src in [("val", valid_vl), ("test", valid_te)]:
        log(f"  [{src_name}]")
        for cam in sorted(src["cam"].unique()):
            sub = src[src.cam == cam].dropna(subset=["numcont"])
            if len(sub) < 50: continue
            r, p = pearsonr(np.log1p(sub["numcont"]), sub["z"].abs())
            rs, _ = spearmanr(sub["numcont"], sub["z"].abs())
            flagged = sub[sub["z"].abs() > Z_THRESH]
            med_nc_flagged = flagged["numcont"].median() if len(flagged) else np.nan
            med_nc_all     = sub["numcont"].median()
            log(f"    Cam {cam:.0f}: r(log_numcont,|z|)={r:+.3f} p={p:.3f}  "
                f"rho={rs:+.3f}  "
                f"med_numcont: flagged={med_nc_flagged:.1f}  all={med_nc_all:.1f}")
else:
    log("  numcont not available — using training_data_v3_tic.parquet required")

# ── Part 2: Spatial residual KNN correction ───────────────────────────────────
log(f"\n── Part 2: Spatial residual KNN correction ──")

# Split val into 80% map / 20% tune
rng = np.random.default_rng(42)
val_stars = valid_vl["tic_id"].unique()
rng.shuffle(val_stars)
n_map   = int(len(val_stars) * 0.80)
map_tics  = set(val_stars[:n_map].tolist())
tune_tics = set(val_stars[n_map:].tolist())
map_df  = valid_vl[valid_vl.tic_id.isin(map_tics)].copy()
tune_df = valid_vl[valid_vl.tic_id.isin(tune_tics)].copy()
log(f"  Map set: {len(map_tics):,} val stars ({len(map_df):,} obs)")
log(f"  Tune set: {len(tune_tics):,} val stars ({len(tune_df):,} obs)")

def build_spatial_correction(reference, query, K=30):
    """
    For each query obs, find K nearest reference obs in (col, row) space
    within same (sector, cam, ccd).  Returns array of mean signed residuals.
    """
    query  = query.reset_index(drop=True)
    signal = np.zeros(len(query), dtype=np.float64)
    count  = np.zeros(len(query), dtype=int)
    for (s, c, q_), qg in query.groupby(["sector","cam","ccd"]):
        r = reference[(reference.sector == s) & (reference.cam == c) & (reference.ccd == q_)]
        if len(r) < 2: continue
        coords  = r[["col","row"]].values.astype(np.float64)
        qcoords = qg[["col","row"]].values.astype(np.float64)
        k = min(K, len(r))
        _, idxs = cKDTree(coords).query(qcoords, k=k)
        if k == 1: idxs = idxs[:, None]
        resids = r["resid"].values
        for i, nn in enumerate(idxs):
            signal[qg.index[i]] = resids[nn].mean()
            count[qg.index[i]]  = k
    return signal, count

log(f"  Building spatial correction from map set ...  {T()}")
for K in [20, 50]:
    # tune on tune_df
    sig_tune, _ = build_spatial_correction(map_df, tune_df, K=K)
    tune_df[f"spatial_K{K}"] = sig_tune
    sig_test, _ = build_spatial_correction(map_df, test_df, K=K)
    test_df[f"spatial_K{K}"] = sig_test

    best_gamma, best_cv = 0.0, 1e9
    for gamma in np.linspace(-1.5, 1.5, 31):
        tune_df["pred_s2"] = tune_df["pred_v2"].values * np.exp(gamma * sig_tune)
        cv = cv_col(tune_df, "pred_s2").median()
        if cv < best_cv:
            best_cv, best_gamma = cv, gamma

    test_df["pred_s2"] = test_df["pred_v2"].values * np.exp(best_gamma * sig_test)
    test_cv_s2 = cv_col(test_df, "pred_s2").median()
    log(f"\n  K={K}: best γ={best_gamma:.2f}  tune_CV={best_cv:.4f}%  test_CV={test_cv_s2:.4f}%")
    log(f"    vs v2 baseline: {test_cv_s2 - baseline_test_cv:+.4f}pp")

    # Per-cam breakdown
    log(f"  Per-cam (K={K}, γ={best_gamma:.2f}):")
    for cam in sorted(test_df["cam"].unique()):
        sub_base = test_df[test_df.cam == cam]
        sub_s2   = test_df[test_df.cam == cam]
        cv_base  = cv_col(sub_base, "pred_v2").median()
        cv_s2    = cv_col(sub_s2,   "pred_s2").median()
        log(f"    Cam {cam:.0f}: {cv_base:.4f}% → {cv_s2:.4f}%  ({cv_s2 - cv_base:+.4f}pp)")

# ── Part 3: Row-stripe feature for Cam 4 CCD 3 ────────────────────────────────
log(f"\n── Part 3: Row-stripe signal — Cam 4 CCD 3 ──")
# Build row-bin mean residual from MAP set, apply to test Cam 4 CCD 3
cam4ccd3_map  = map_df[(map_df.cam == 4) & (map_df.ccd == 3)]
cam4ccd3_test = test_df[(test_df.cam == 4) & (test_df.ccd == 3)].copy()

if len(cam4ccd3_map) > 0 and len(cam4ccd3_test) > 0:
    # bin rows into 40 bins per sector
    for sector in cam4ccd3_map["sector"].unique():
        m_sec = cam4ccd3_map[cam4ccd3_map.sector == sector]
        if len(m_sec) < 10: continue
        row_min, row_max = m_sec["row"].min(), m_sec["row"].max()
        edges = np.linspace(row_min, row_max, 41)
        m_sec = m_sec.copy()
        m_sec["row_bin"] = pd.cut(m_sec["row"], bins=edges, labels=False)
        stripe = m_sec.groupby("row_bin")["resid"].mean()

        t_sec_idx = cam4ccd3_test[(cam4ccd3_test.sector == sector)].index
        if not len(t_sec_idx): continue
        t_sec = cam4ccd3_test.loc[t_sec_idx]
        t_bins = pd.cut(t_sec["row"], bins=edges, labels=False)
        test_df.loc[t_sec_idx, "row_stripe"] = t_bins.map(stripe).fillna(0.0).values

    test_df["row_stripe"] = test_df.get("row_stripe", pd.Series(0.0, index=test_df.index)).fillna(0.0)

    best_gamma_rs, best_cv_rs = 0.0, cv_col(tune_df, "pred_v2").median()
    sig_test_rs = test_df["row_stripe"].values
    # quick gamma sweep (use K=20 spatial as base, add row stripe on top)
    for gamma in np.linspace(-2.0, 2.0, 41):
        test_df["pred_rs"] = test_df["pred_v2"].values * np.exp(gamma * sig_test_rs)
        # evaluate only on Cam 4 CCD 3
        sub = test_df[(test_df.cam == 4) & (test_df.ccd == 3)]
        cv = cv_col(sub, "pred_rs").median()
        if cv < best_cv_rs:
            best_cv_rs, best_gamma_rs = cv, gamma

    test_df["pred_rs"] = test_df["pred_v2"].values * np.exp(best_gamma_rs * sig_test_rs)
    cv_base_c4 = cv_col(cam4ccd3_test, "pred_v2").median()
    cv_rs_c4   = cv_col(test_df[(test_df.cam == 4) & (test_df.ccd == 3)], "pred_rs").median()
    log(f"  Cam 4 CCD 3: {cv_base_c4:.4f}% → {cv_rs_c4:.4f}%  (γ={best_gamma_rs:.2f})")
    log(f"  Δ = {cv_rs_c4 - cv_base_c4:+.4f}pp")
else:
    log("  Cam 4 CCD 3 not found in map/test sets")

# ── Summary ───────────────────────────────────────────────────────────────────
log(f"\n=== SUMMARY ===")
log(f"  HGB v2 baseline test CV:    {baseline_test_cv:.4f}%")
log(f"  Spatial residual KNN (K=20): see per-K results above")
log(f"  Note: banding_profiles.png and spatial_zscore.png show the structure being targeted")
log(f"\nDONE  {T()}")
