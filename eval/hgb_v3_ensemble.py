"""
HGB v3 Ensemble: 5-seed variance reduction.

Trains 5 independent HGB v3 models (same features, different seeds),
averages their predictions. Run AFTER hgb_v3.py has completed (reuses
the split TIC IDs and pre-computed spatial features).

Typically improves CV by 3-8% over single model via variance reduction.
Also gives a ±σ confidence interval on the 0.944% result.

Usage (from STITCH root):
  python3 eval/hgb_v3_ensemble.py [parquet]
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.spatial import cKDTree

PARQUET    = sys.argv[1] if len(sys.argv) > 1 else "training_data_v3.parquet"
OUT_MODEL  = "models/hgb_v3_ensemble"
MODEL_DIR  = "models/hgb_v3"
N_ENSEMBLE = 5
MIN_OBS_SECTORS = 4

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

# ── Load v3 artifacts to reuse exact split ────────────────────────────────────
log("Loading HGB v3 artifacts ...")
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)
SEED     = meta["seed"]
FEATS    = meta["features"]
KNN_KS   = meta.get("knn_ks", [5, 10, 20, 50])
LSA_KS   = meta.get("lsa_ks", [20, 50, 100])
best_beta = meta["best_beta"]

tr_tics = set(np.load(os.path.join(MODEL_DIR, "train_tic_ids.npy")).tolist())
vl_tics = set(np.load(os.path.join(MODEL_DIR, "val_tic_ids.npy")).tolist())
te_tics = set(np.load(os.path.join(MODEL_DIR, "test_tic_ids.npy")).tolist())
log(f"  Split: train={len(tr_tics):,}  val={len(vl_tics):,}  test={len(te_tics):,}")

# ── Load data (exact same pipeline as hgb_v3) ─────────────────────────────────
log(f"Loading {PARQUET} ...")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","flux_offset_loo","sector_median"])
df = df[(df.flux_offset     > 0.85) & (df.flux_offset     < 1.15)]
df = df[(df.flux_offset_loo > 0.85) & (df.flux_offset_loo < 1.15)]
df = df[df.tmag <= 13.0]
df["tic_id"] = df["tic_id"].astype("int64")
df["sector"] = df["sector"].astype("int64")
if "n_sectors_total" in df.columns:
    df = df[df["n_sectors_total"] >= MIN_OBS_SECTORS]
else:
    n_sec = df.groupby("tic_id")["sector"].nunique()
    df = df[df.tic_id.isin(n_sec[n_sec >= MIN_OBS_SECTORS].index)]
df = df.reset_index(drop=True)
log(f"  {len(df):,} records · {df.tic_id.nunique():,} stars")

te_set = te_tics
train_df = df[ df.tic_id.isin(tr_tics)].copy().reset_index(drop=True)
val_df   = df[ df.tic_id.isin(vl_tics)].copy().reset_index(drop=True)
test_df  = df[ df.tic_id.isin(te_tics)].copy().reset_index(drop=True)
ref_full = df[~df.tic_id.isin(te_set)  ].copy().reset_index(drop=True)

# Sector aggregates
sec_stats = pd.read_parquet(os.path.join(MODEL_DIR, "sec_stats.parquet"))
for df_ in [train_df, val_df, test_df]:
    df_.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"],
             errors="ignore", inplace=True)
    merged = df_.merge(sec_stats, on=["sector","cam","ccd"], how="left")
    df_["sector_mean_loo"]     = merged["sector_mean_loo"].fillna(1.0).values
    df_["sector_std_loo"]      = merged["sector_std_loo"].fillna(0.0).values
    df_["sector_frac_extreme"] = merged["sector_frac_extreme"].fillna(0.0).values

# ── KNN-100 baseline ──────────────────────────────────────────────────────────
log("KNN-100 baseline ...")
def knn_median_loo(ref, query, k=100):
    result = np.ones(len(query), dtype=np.float64)
    for (s, c, q), qg in query.groupby(["sector","cam","ccd"]):
        r = ref[(ref.sector == s) & (ref.cam == c) & (ref.ccd == q)]
        if len(r) < 2: continue
        coords   = r[["col","row"]].to_numpy(np.float64)
        qcoords  = qg[["col","row"]].to_numpy(np.float64)
        rids     = r.tic_id.to_numpy()
        loo_vals = r.flux_offset_loo.to_numpy()
        take = min(k + int(r.tic_id.value_counts().max()), len(r))
        _, idxs = cKDTree(coords).query(qcoords, k=take)
        if take == 1: idxs = idxs[:, None]
        for i, (q_id, nn) in enumerate(zip(qg.tic_id.to_numpy(), idxs)):
            keep = rids[nn] != q_id
            vals = loo_vals[nn[keep][:k]]
            if len(vals) >= 1:
                result[qg.index[i]] = np.median(vals)
    return result

knn_tr = knn_median_loo(ref_full, train_df)
knn_vl = knn_median_loo(ref_full, val_df)
knn_te = knn_median_loo(ref_full, test_df)
train_df["pred_knn"] = knn_tr; val_df["pred_knn"] = knn_vl; test_df["pred_knn"] = knn_te
log(f"  Done  {T()}")

# ── Spatial + LSA + poly features ─────────────────────────────────────────────
log("Spatial + LSA + poly features ...")
SPATIAL   = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]] + ["ref_count"])
LSA_FEATS = ([f"lsa{k}_{s}" for k in LSA_KS for s in ["mean","median","std"]] + ["poly_val"])

def add_all_spatial(ref_df, eval_df):
    all_cols = SPATIAL + LSA_FEATS
    arrays = {c: np.full(len(eval_df), np.nan, dtype=np.float32) for c in all_cols}
    max_k = max(max(KNN_KS), max(LSA_KS))
    for (s, c, q), gr in ref_df.groupby(["sector","cam","ccd"]):
        ge = eval_df[(eval_df.sector == s) & (eval_df.cam == c) & (eval_df.ccd == q)]
        if not len(ge): continue
        xyr = gr[["col","row"]].values.astype(np.float32)
        xye = ge[["col","row"]].values.astype(np.float32)
        vr  = gr.flux_offset_loo.values.astype(np.float32)
        delta_r = (gr.flux_offset_loo.values - gr.sector_median.values).astype(np.float32)
        tree = cKDTree(xyr)
        arrays["ref_count"][ge.index.values] = len(gr)
        kk = min(max_k, len(gr))
        if kk < 1: continue
        d_all, nn_all = tree.query(xye, k=kk)
        if kk == 1: nn_all = nn_all[:, None]; d_all = d_all[:, None]
        for k in KNN_KS:
            ak = min(k, kk); nn_k = nn_all[:, :ak]; d_k = d_all[:, :ak]
            nbv = vr[nn_k]; med = np.median(nbv, axis=1)
            arrays[f"knn{k}_mean"][ge.index.values]   = nbv.mean(axis=1)
            arrays[f"knn{k}_median"][ge.index.values] = med
            arrays[f"knn{k}_mad"][ge.index.values]    = np.abs(nbv - med[:, None]).mean(axis=1)
            arrays[f"knn{k}_radius"][ge.index.values] = d_k.max(axis=1)
        for k in LSA_KS:
            ak = min(k, kk); nn_k = nn_all[:, :ak]
            nbv_d = delta_r[nn_k]; med_d = np.median(nbv_d, axis=1)
            arrays[f"lsa{k}_mean"][ge.index.values]   = nbv_d.mean(axis=1)
            arrays[f"lsa{k}_median"][ge.index.values] = med_d
            arrays[f"lsa{k}_std"][ge.index.values]    = nbv_d.std(axis=1)
        if len(gr) >= 6:
            col_n = (gr.col.values - 1100.0) / 1100.0; row_n = (gr.row.values - 1100.0) / 1100.0
            A = np.column_stack([np.ones_like(col_n), col_n, row_n, col_n**2, col_n*row_n, row_n**2])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, vr, rcond=None)
                qcn = (ge.col.values - 1100.0) / 1100.0; qrn = (ge.row.values - 1100.0) / 1100.0
                Aq = np.column_stack([np.ones_like(qcn), qcn, qrn, qcn**2, qcn*qrn, qrn**2])
                arrays["poly_val"][ge.index.values] = (Aq @ coeffs).astype(np.float32)
            except: pass
    for c_name, arr in arrays.items():
        eval_df[c_name] = arr
    for cn in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
        eval_df[cn] = eval_df[cn].fillna(eval_df["sector_mean_loo"])
    eval_df["ref_count"] = eval_df["ref_count"].fillna(0.0)
    for cn in [f"lsa{k}_{s}" for k in LSA_KS for s in ["mean","median","std"]]:
        eval_df[cn] = eval_df[cn].fillna(0.0)
    eval_df["poly_val"] = eval_df["poly_val"].fillna(eval_df["sector_mean_loo"])

add_all_spatial(ref_full, train_df)
add_all_spatial(ref_full, val_df)
add_all_spatial(ref_full, test_df)
log(f"  Done  {T()}")

# Derived features
for df_ in [train_df, val_df, test_df]:
    fp = df_.get("focal_plane_temp", pd.Series(np.nan, index=df_.index))
    df_["temp_x_col"]         = fp.values * (df_["col"].values / 2200.0)
    df_["temp_x_row"]         = fp.values * (df_["row"].values / 2200.0)
    df_["temp_x_crowdsap"]    = fp.values * df_["crowdsap"].values
    pt = df_.get("pdc_totp",  pd.Series(np.nan, index=df_.index))
    df_["pdctotp_x_crowdsap"] = pt.values * df_["crowdsap"].values
    df_["n_sectors_obs"]      = (df_.groupby("tic_id")["sector"].transform("nunique").astype(float))

def get_X(df_):
    X = df_[FEATS].copy()
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X.values.astype(np.float32)

def cv_col(df_, col, min_s=2):
    rows = []
    for _, g in df_.groupby("tic_id"):
        if g.sector.nunique() < min_s: continue
        c = g["sector_median"].values / g[col].values
        rows.append(float(c.std() / c.mean() * 100))
    return pd.Series(rows)

X_tr = get_X(train_df); X_vl = get_X(val_df); X_te = get_X(test_df)

# ── Train N_ENSEMBLE models ────────────────────────────────────────────────────
log(f"\nTraining {N_ENSEMBLE} ensemble members ...")
resp   = np.log(train_df.sector_median.values) - np.log(knn_tr)
ids_tr = train_df.tic_id.to_numpy()

def group_median(vals, ids):
    return pd.Series(vals).groupby(pd.Series(ids)).transform("median").to_numpy()

models = []
val_preds_list  = []
test_preds_list = []

for e in range(N_ENSEMBLE):
    seed_e = SEED + e * 100
    log(f"\n  Ensemble member {e+1}/{N_ENSEMBLE} (seed base={seed_e}) ...")
    intercept = group_median(resp, ids_tr)
    m = None
    for i in range(5):
        m = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.06, max_iter=250,
            max_leaf_nodes=1023, min_samples_leaf=40, l2_regularization=0.1,
            random_state=seed_e + i, early_stopping=False)
        m.fit(X_tr, resp - intercept)
        intercept = group_median(resp - m.predict(X_tr), ids_tr)
        m = m
        log(f"    Round {i+1}/5  {T()}")
    models.append(m)
    val_preds_list.append(m.predict(X_vl))
    test_preds_list.append(m.predict(X_te))

# Ensemble average deltas
val_delta_ens  = np.mean(val_preds_list, axis=0)
test_delta_ens = np.mean(test_preds_list, axis=0)

# ── Beta sweep on ensemble val predictions ────────────────────────────────────
log("\nBeta sweep (ensemble) ...")
best_beta_e, best_val_cv_e = 1.0, 1e9
for beta in np.linspace(0.5, 2.0, 31):
    pred = np.clip(knn_vl * np.exp(float(beta) * val_delta_ens), 0.85, 1.15)
    val_df["_tmp"] = pred
    cv = cv_col(val_df, "_tmp").median()
    log(f"  β={beta:.2f}  val_CV={cv:.4f}%")
    if cv < best_val_cv_e:
        best_val_cv_e = cv
        best_beta_e   = float(beta)
log(f"  → Best β={best_beta_e:.2f}  val_CV={best_val_cv_e:.4f}%")
val_df.drop(columns=["_tmp"], inplace=True)

# Also report individual model val CVs for variance estimate
log("\nIndividual member val CVs (±σ estimate):")
ind_val_cvs = []
for e, vp in enumerate(val_preds_list):
    pred = np.clip(knn_vl * np.exp(best_beta * vp), 0.85, 1.15)
    val_df["_tmp"] = pred
    cv_e = cv_col(val_df, "_tmp").median()
    val_df.drop(columns=["_tmp"], inplace=True)
    ind_val_cvs.append(cv_e)
    log(f"  Member {e+1}: val_CV={cv_e:.4f}%")
log(f"  Mean={np.mean(ind_val_cvs):.4f}%  Std={np.std(ind_val_cvs):.4f}%")

# ── Test evaluation ───────────────────────────────────────────────────────────
log("\n=== TEST RESULTS ===")
test_df["pred_ens"] = np.clip(knn_te * np.exp(best_beta_e * test_delta_ens), 0.85, 1.15)
cv_ens = cv_col(test_df, "pred_ens").median()

# Individual test CVs for ±σ
ind_test_cvs = []
for e, tp in enumerate(test_preds_list):
    pred = np.clip(knn_te * np.exp(best_beta * tp), 0.85, 1.15)
    test_df["_tmp"] = pred
    cv_e = cv_col(test_df, "_tmp").median()
    test_df.drop(columns=["_tmp"], inplace=True)
    ind_test_cvs.append(cv_e)

log(f"  Ensemble CV:          {cv_ens:.4f}%  (β={best_beta_e:.2f})")
log(f"  Individual test CVs:  {[f'{v:.4f}' for v in ind_test_cvs]}")
log(f"  Mean ± Std:           {np.mean(ind_test_cvs):.4f}% ± {np.std(ind_test_cvs):.4f}%")
log(f"  vs HGB v2 baseline:   {cv_ens - 0.9444:+.4f}pp")
log(f"  vs single HGB v3:     see above")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(OUT_MODEL, exist_ok=True)
for e, m in enumerate(models):
    joblib.dump(m, os.path.join(OUT_MODEL, f"model_{e}.pkl"))
ens_meta = {
    "n_ensemble":    N_ENSEMBLE,
    "best_beta":     best_beta_e,
    "val_cv":        float(best_val_cv_e),
    "test_cv_ens":   float(cv_ens),
    "test_cv_mean":  float(np.mean(ind_test_cvs)),
    "test_cv_std":   float(np.std(ind_test_cvs)),
    "test_cv_each":  ind_test_cvs,
    "features":      FEATS,
    "date":          "2026-10-07",
}
with open(os.path.join(OUT_MODEL, "meta.json"), "w") as f:
    json.dump(ens_meta, f, indent=2)
log(f"\nSaved → {OUT_MODEL}/  {T()}")
log("DONE")
