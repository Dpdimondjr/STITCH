"""
HGB v3: local-sector-anomaly (LSA) + polynomial surface features.

New vs HGB v2:
  - LSA features: for each (star, sector), KNN-aggregate of reference
    neighbors' DEVIATION from their own sector_median in this specific sector.
    Captures "is this sector unusually high/low for stars at this position?"
    lsa{20,50,100}_{mean,median,std}  →  9 new features
  - Polynomial surface: degree-2 poly fitted to ref corrections per
    (sector,cam,ccd); poly_val at query (col,row).  →  1 new feature
  - Total: 57D features (vs 47D in v2)
  - Variance decomposition: oracle (perfect per-star prediction) shows
    irreducible per-sector variation floor.

Usage (from STITCH root):
  python3 eval/hgb_v3.py [parquet]
"""

import sys, os, re, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.spatial import cKDTree

PARQUET   = sys.argv[1] if len(sys.argv) > 1 else "training_data_v3.parquet"
OUT       = sys.argv[2] if len(sys.argv) > 2 else "stitch_hgb_v3_dashboard.html"
TARS_CSV  = "tars_quiet_tics_v2.csv"
MODEL_DIR = "models/hgb_v3"
SEED      = 42
KNN_KS    = [5, 10, 20, 50]
LSA_KS    = [20, 50, 100]
MIN_OBS_SECTORS = 4

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

# ── Load & filter ─────────────────────────────────────────────────────────────
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

# ── Method A split (80/10/10, stratified by dominant cam) ─────────────────────
star_cam = df.groupby("tic_id")["cam"].agg(lambda x: int(x.mode()[0]))
tr_tics, tmp = train_test_split(
    star_cam.index.to_numpy(), test_size=0.2,
    stratify=star_cam.values, random_state=SEED)
tmp_cam = star_cam.loc[tmp].values
vl_tics, te_tics = train_test_split(
    tmp, test_size=0.5, stratify=tmp_cam, random_state=SEED)
log(f"  Train {len(tr_tics):,}  Val {len(vl_tics):,}  Test {len(te_tics):,}")

te_set = set(te_tics)
train_df = df[ df.tic_id.isin(tr_tics)].copy().reset_index(drop=True)
val_df   = df[ df.tic_id.isin(vl_tics)].copy().reset_index(drop=True)
test_df  = df[ df.tic_id.isin(te_tics)].copy().reset_index(drop=True)
ref_full = df[~df.tic_id.isin(te_set) ].copy().reset_index(drop=True)

# ── Sector aggregates (from ref_full) ─────────────────────────────────────────
log("Sector aggregates ...")
sec_stats = (
    ref_full[ref_full.flux_offset_loo.between(0.5, 1.5)]
    .groupby(["sector","cam","ccd"])["flux_offset_loo"]
    .agg(sector_mean_loo="mean", sector_std_loo="std",
         sector_frac_extreme=lambda x: (np.abs(x - 1) > 0.02).mean())
    .reset_index()
)
for df_ in [train_df, val_df, test_df]:
    df_.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"],
             errors="ignore", inplace=True)
    merged = df_.merge(sec_stats, on=["sector","cam","ccd"], how="left")
    df_["sector_mean_loo"]     = merged["sector_mean_loo"].fillna(1.0).values
    df_["sector_std_loo"]      = merged["sector_std_loo"].fillna(0.0).values
    df_["sector_frac_extreme"] = merged["sector_frac_extreme"].fillna(0.0).values

# ── KNN-100 baseline (LOO from ref_full) ──────────────────────────────────────
log("KNN-100 baseline ...")

def knn_median_loo(ref, query, k=100):
    result = np.ones(len(query), dtype=np.float64)
    for (s, c, q), qg in query.groupby(["sector","cam","ccd"]):
        r = ref[(ref.sector == s) & (ref.cam == c) & (ref.ccd == q)]
        if len(r) < 2:
            continue
        coords   = r[["col","row"]].to_numpy(np.float64)
        qcoords  = qg[["col","row"]].to_numpy(np.float64)
        qids     = qg.tic_id.to_numpy()
        rids     = r.tic_id.to_numpy()
        loo_vals = r.flux_offset_loo.to_numpy()
        take = min(k + int(r.tic_id.value_counts().max()), len(r))
        _, idxs = cKDTree(coords).query(qcoords, k=take)
        if take == 1:
            idxs = idxs[:, None]
        for i, (q_id, nn) in enumerate(zip(qids, idxs)):
            keep = rids[nn] != q_id
            vals = loo_vals[nn[keep][:k]]
            if len(vals) >= 1:
                result[qg.index[i]] = np.median(vals)
    return result

knn_tr = knn_median_loo(ref_full, train_df)
knn_vl = knn_median_loo(ref_full, val_df)
knn_te = knn_median_loo(ref_full, test_df)
train_df["pred_knn"] = knn_tr
val_df["pred_knn"]   = knn_vl
test_df["pred_knn"]  = knn_te
log(f"  Done  {T()}")

# ── Spatial KNN + Local Sector Anomaly + Polynomial Surface ───────────────────
log("Spatial KNN + LSA + polynomial features ...")

SPATIAL  = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]
            + ["ref_count"])
LSA_FEATS = ([f"lsa{k}_{s}" for k in LSA_KS for s in ["mean","median","std"]]
             + ["poly_val"])

def add_all_spatial(ref_df, eval_df):
    """Compute spatial KNN features, LSA features, and polynomial surface in one pass."""
    all_cols = SPATIAL + LSA_FEATS
    arrays = {c: np.full(len(eval_df), np.nan, dtype=np.float32) for c in all_cols}

    max_k = max(max(KNN_KS), max(LSA_KS))  # 100

    for (s, c, q), gr in ref_df.groupby(["sector","cam","ccd"]):
        ge = eval_df[(eval_df.sector == s) & (eval_df.cam == c) & (eval_df.ccd == q)]
        if not len(ge):
            continue

        xyr = gr[["col","row"]].values.astype(np.float32)
        xye = ge[["col","row"]].values.astype(np.float32)
        vr  = gr.flux_offset_loo.values.astype(np.float32)

        # Delta = ref star's deviation from its own sector_median in THIS sector
        delta_r = (gr.flux_offset_loo.values - gr.sector_median.values).astype(np.float32)

        tree = cKDTree(xyr)
        arrays["ref_count"][ge.index.values] = len(gr)

        # Single KNN query for all k-values
        kk = min(max_k, len(gr))
        if kk < 1:
            continue
        d_all, nn_all = tree.query(xye, k=kk)
        if kk == 1:
            nn_all = nn_all[:, None]
            d_all  = d_all[:, None]

        # Spatial KNN features (flux_offset_loo)
        for k in KNN_KS:
            actual_k = min(k, kk)
            nn_k = nn_all[:, :actual_k]
            d_k  = d_all[:, :actual_k]
            nbv  = vr[nn_k]
            med  = np.median(nbv, axis=1)
            arrays[f"knn{k}_mean"]  [ge.index.values] = nbv.mean(axis=1)
            arrays[f"knn{k}_median"][ge.index.values] = med
            arrays[f"knn{k}_mad"]   [ge.index.values] = np.abs(nbv - med[:, None]).mean(axis=1)
            arrays[f"knn{k}_radius"][ge.index.values] = d_k.max(axis=1)

        # Local sector anomaly (deviation of neighbors from their own averages)
        for k in LSA_KS:
            actual_k = min(k, kk)
            nn_k = nn_all[:, :actual_k]
            nbv_d = delta_r[nn_k]
            med_d = np.median(nbv_d, axis=1)
            arrays[f"lsa{k}_mean"]  [ge.index.values] = nbv_d.mean(axis=1)
            arrays[f"lsa{k}_median"][ge.index.values] = med_d
            arrays[f"lsa{k}_std"]   [ge.index.values] = nbv_d.std(axis=1)

        # Polynomial surface (degree-2) fit to ref corrections in this group
        if len(gr) >= 6:
            col_n = (gr.col.values - 1100.0) / 1100.0
            row_n = (gr.row.values - 1100.0) / 1100.0
            A = np.column_stack([np.ones_like(col_n), col_n, row_n,
                                  col_n**2, col_n * row_n, row_n**2])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, vr, rcond=None)
                qcol_n = (ge.col.values - 1100.0) / 1100.0
                qrow_n = (ge.row.values - 1100.0) / 1100.0
                Aq = np.column_stack([np.ones_like(qcol_n), qcol_n, qrow_n,
                                       qcol_n**2, qcol_n * qrow_n, qrow_n**2])
                arrays["poly_val"][ge.index.values] = (Aq @ coeffs).astype(np.float32)
            except Exception:
                pass

    for c_name, arr in arrays.items():
        eval_df[c_name] = arr

    # Fill NaNs
    for c_name in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
        eval_df[c_name] = eval_df[c_name].fillna(eval_df["sector_mean_loo"])
    eval_df["ref_count"] = eval_df["ref_count"].fillna(0.0)
    for c_name in [f"lsa{k}_{s}" for k in LSA_KS for s in ["mean","median","std"]]:
        eval_df[c_name] = eval_df[c_name].fillna(0.0)
    eval_df["poly_val"] = eval_df["poly_val"].fillna(eval_df["sector_mean_loo"])

add_all_spatial(ref_full, train_df)
add_all_spatial(ref_full, val_df)
add_all_spatial(ref_full, test_df)
log(f"  Done  {T()}")

# ── Derived features ──────────────────────────────────────────────────────────
for df_ in [train_df, val_df, test_df]:
    fp = df_.get("focal_plane_temp", pd.Series(np.nan, index=df_.index))
    df_["temp_x_col"]         = fp.values * (df_["col"].values / 2200.0)
    df_["temp_x_row"]         = fp.values * (df_["row"].values / 2200.0)
    df_["temp_x_crowdsap"]    = fp.values * df_["crowdsap"].values
    pt = df_.get("pdc_totp",  pd.Series(np.nan, index=df_.index))
    df_["pdctotp_x_crowdsap"] = pt.values * df_["crowdsap"].values
    df_["n_sectors_obs"]      = (df_.groupby("tic_id")["sector"]
                                   .transform("nunique").astype(float))

# ── Feature list ──────────────────────────────────────────────────────────────
BASE    = ["col","row","delta_sub_col","delta_sub_row","sector","cam","ccd",
           "tmag","crowdsap","cdpp1_0","pdcvar","jitter_rms","pdc_noi","pr_wght2","gaiarp"]
THERMAL = ["focal_plane_temp","focal_plane_temp_std",
           "temp_x_col","temp_x_row","temp_x_crowdsap"]
PDC_F   = ["pdc_corp","pdc_totp","pdctotp_x_crowdsap","teff","bkg_rms","scatter_flag_frac"]
MISC    = ["n_sectors_obs"]
SEC_AGG = ["sector_mean_loo","sector_std_loo","sector_frac_extreme"]
TAB     = [c for c in BASE + THERMAL + PDC_F + MISC + SEC_AGG if c in train_df.columns]
FEATS   = TAB + SPATIAL + LSA_FEATS
log(f"Features: {len(FEATS)}D ({len(TAB)} tab + {len(SPATIAL)} spatial + {len(LSA_FEATS)} lsa/poly)")

def get_X(df_):
    X = df_[FEATS].copy()
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X.values.astype(np.float32)

def cv_col(df_, col, min_s=2):
    rows = []
    for _, g in df_.groupby("tic_id"):
        if g.sector.nunique() < min_s:
            continue
        c = g["sector_median"].values / g[col].values
        rows.append(float(c.std() / c.mean() * 100))
    return pd.Series(rows)

# ── 5-round progressive HGB ────────────────────────────────────────────────────
log("\nTraining HGB (5-round progressive) ...")
X_tr = get_X(train_df)
X_vl = get_X(val_df)
X_te = get_X(test_df)

resp   = np.log(train_df.sector_median.values) - np.log(knn_tr)
ids_tr = train_df.tic_id.to_numpy()

def group_median(vals, ids):
    return pd.Series(vals).groupby(pd.Series(ids)).transform("median").to_numpy()

intercept = group_median(resp, ids_tr)
model = None
for i in range(5):
    m = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=250,
        max_leaf_nodes=1023, min_samples_leaf=40, l2_regularization=0.1,
        random_state=SEED + i, early_stopping=False)
    m.fit(X_tr, resp - intercept)
    intercept = group_median(resp - m.predict(X_tr), ids_tr)
    model = m
    log(f"  Round {i+1}/5  {T()}")

# ── Beta sweep on val set ─────────────────────────────────────────────────────
log("Beta sweep on val set ...")
val_delta = model.predict(X_vl)
best_beta, best_val_cv = 1.0, 1e9
for beta in np.linspace(-0.5, 2.5, 61):
    pred = np.clip(knn_vl * np.exp(float(beta) * val_delta), 0.85, 1.15)
    val_df["_tmp"] = pred
    cv = cv_col(val_df, "_tmp").median()
    log(f"  β={beta:.2f}  val_CV={cv:.4f}%")
    if cv < best_val_cv:
        best_val_cv = cv
        best_beta   = float(beta)
log(f"  → Best β={best_beta:.2f}  val_CV={best_val_cv:.4f}%")
val_df.drop(columns=["_tmp"], inplace=True)

# ── Save model artifacts ──────────────────────────────────────────────────────
log(f"Saving model to {MODEL_DIR}/ ...")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
np.save(os.path.join(MODEL_DIR, "ref_tic_ids.npy"), ref_full["tic_id"].unique())
sec_stats.to_parquet(os.path.join(MODEL_DIR, "sec_stats.parquet"), index=False)
# Save per-star sector_median from ref_full for LSA computation at inference time
ref_star_medians = (ref_full.groupby("tic_id")["sector_median"].first()
                    .reset_index().rename(columns={"sector_median": "ref_sector_median"}))
ref_star_medians.to_parquet(os.path.join(MODEL_DIR, "ref_star_medians.parquet"), index=False)
meta_out = {
    "parquet":       PARQUET,
    "seed":          SEED,
    "best_beta":     best_beta,
    "val_cv":        float(best_val_cv),
    "features":      FEATS,
    "n_train_stars": int(len(tr_tics)),
    "n_val_stars":   int(len(vl_tics)),
    "n_test_stars":  int(len(te_tics)),
    "min_obs_sectors": MIN_OBS_SECTORS,
    "knn_ks":        KNN_KS,
    "lsa_ks":        LSA_KS,
    "date":          "2026-10-07",
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta_out, f, indent=2)
np.save(os.path.join(MODEL_DIR, "test_tic_ids.npy"),  np.array(list(te_tics),  dtype=np.int64))
np.save(os.path.join(MODEL_DIR, "val_tic_ids.npy"),   np.array(list(vl_tics),  dtype=np.int64))
np.save(os.path.join(MODEL_DIR, "train_tic_ids.npy"), np.array(list(tr_tics),  dtype=np.int64))
log(f"  Saved  {T()}")

# ── Test evaluation ───────────────────────────────────────────────────────────
log("\n=== VARIANCE DECOMPOSITION ===")
# Oracle: what if we could predict each star's sector_median EXACTLY?
# (perfect per-star, zero prediction error, only per-sector variation remains)
test_df["pred_oracle"] = test_df["sector_median"]
cv_oracle = cv_col(test_df, "pred_oracle").median()
log(f"  Oracle (perfect per-star prediction): {cv_oracle:.4f}%")
log(f"  KNN baseline:                         see below")
log(f"  This is the irreducible floor from within-star sector-to-sector variation.")
log(f"  (Cannot be reduced without per-sector features)")
test_df.drop(columns=["pred_oracle"], inplace=True)

log("\n=== TEST RESULTS ===")
test_delta          = model.predict(X_te)
test_df["pred_hgb"] = np.clip(knn_te * np.exp(best_beta * test_delta), 0.85, 1.15)
test_df["pred_knn_clip"] = np.clip(knn_te, 0.85, 1.15)
test_df["pred_raw"]  = 1.0

cv_raw = cv_col(test_df, "pred_raw").median()
cv_knn = cv_col(test_df, "pred_knn_clip").median()
cv_hgb = cv_col(test_df, "pred_hgb").median()

# Residual analysis vs v2
log(f"  Raw:    {cv_raw:.4f}%")
log(f"  KNN:    {cv_knn:.4f}%")
log(f"  HGB v3: {cv_hgb:.4f}%  β={best_beta:.2f}")
log(f"  Oracle: {cv_oracle:.4f}%  (per-sector variation floor)")
log(f"  HGB v3 vs KNN:  {cv_hgb - cv_knn:+.4f}pp")
log(f"  HGB v3 vs Raw:  {(cv_hgb/cv_raw - 1)*100:+.1f}%")
log(f"  HGB v3 vs v2 baseline (0.9444%): {cv_hgb - 0.9444:+.4f}pp")

# Save predictions for stage-2 training
test_df["hgb_v3_delta"] = test_delta
test_df["pred_hgb_v3"]  = test_df["pred_hgb"].copy()
# Also save val predictions for stage-2 training
val_df["hgb_v3_delta"]  = val_delta
val_df["pred_hgb_v3"]   = np.clip(knn_vl * np.exp(best_beta * val_delta), 0.85, 1.15)
train_delta = model.predict(X_tr)
train_df["hgb_v3_delta"] = train_delta
train_df["pred_hgb_v3"]  = np.clip(knn_tr * np.exp(best_beta * train_delta), 0.85, 1.15)
# Compute per-sector stage-2 targets for train/val
train_df["stage2_target"] = np.log(np.clip(train_df.flux_offset_loo.values, 0.85, 1.15)) \
                           - np.log(train_df["pred_hgb_v3"].values)
val_df["stage2_target"]   = np.log(np.clip(val_df.flux_offset_loo.values, 0.85, 1.15)) \
                           - np.log(val_df["pred_hgb_v3"].values)
log(f"\n  Stage-2 target std (train): {train_df['stage2_target'].std():.5f}")
log(f"  Stage-2 target std (val):   {val_df['stage2_target'].std():.5f}")

# Save train/val for stage-2
stage2_dir = os.path.join(MODEL_DIR, "stage2_data")
os.makedirs(stage2_dir, exist_ok=True)
cols_s2 = (["tic_id","sector","cam","ccd","flux_offset_loo","sector_median",
             "pred_hgb_v3","stage2_target"]
           + LSA_FEATS
           + [c for c in ["focal_plane_temp","n_sectors_obs","sector_mean_loo",
                           "sector_std_loo","sector_frac_extreme"] if c in train_df.columns])
cols_s2 = [c for c in cols_s2 if c in train_df.columns]
train_df[cols_s2].to_parquet(os.path.join(stage2_dir, "train.parquet"), index=False)
val_df[cols_s2].to_parquet(os.path.join(stage2_dir, "val.parquet"), index=False)
# For test, save for stage-2 eval (without stage2_target since it's the test set)
cols_s2_te = [c for c in cols_s2 if c != "stage2_target"] + ["pred_hgb_v3"]
cols_s2_te = list(dict.fromkeys(cols_s2_te))
cols_s2_te = [c for c in cols_s2_te if c in test_df.columns]
test_df[cols_s2_te].to_parquet(os.path.join(stage2_dir, "test.parquet"), index=False)
log(f"  Stage-2 data saved to {stage2_dir}/  {T()}")

log("\nBy tmag:")
log(f"  {'Bin':>8}  {'n':>5}  {'raw':>7}  {'knn':>7}  {'hgb':>7}  {'oracle':>8}")
for lo, hi in [(6,9),(9,10),(10,11),(11,12),(12,13)]:
    sub = test_df[test_df.tmag.between(lo, hi)]
    if not len(sub): continue
    n = sub.tic_id.nunique()
    r = cv_col(sub, "pred_raw").median()
    k = cv_col(sub, "pred_knn_clip").median()
    h = cv_col(sub, "pred_hgb").median()
    # Oracle per-tmag bin
    sub["_or"] = sub["sector_median"]
    o = cv_col(sub, "_or").median()
    sub.drop(columns=["_or"], inplace=True)
    log(f"  T {lo:2d}–{hi:2d}  {n:>5}  {r:>7.3f}  {k:>7.3f}  {h:>7.3f}  {o:>8.3f}")

log("\nBy camera:")
log(f"  {'Cam':>4}  {'n':>5}  {'raw':>7}  {'knn':>7}  {'hgb':>7}")
for cam in sorted(test_df.cam.unique()):
    sub = test_df[test_df.cam == cam]
    n = sub.tic_id.nunique()
    r = cv_col(sub, "pred_raw").median()
    k = cv_col(sub, "pred_knn_clip").median()
    h = cv_col(sub, "pred_hgb").median()
    log(f"  Cam{int(cam)}  {n:>5}  {r:>7.3f}  {k:>7.3f}  {h:>7.3f}")

log("\nHarm-case analysis:")
harmed_tics = set()
for tic, g in test_df.groupby("tic_id"):
    if g.sector.nunique() < 2: continue
    raw_a = g["sector_median"].values
    cor_a = g["sector_median"].values / g["pred_hgb"].values
    if cor_a.std()/cor_a.mean()*100 > raw_a.std()/raw_a.mean()*100:
        harmed_tics.add(tic)
harmed_df = test_df[test_df.tic_id.isin(harmed_tics)]
helped_df = test_df[~test_df.tic_id.isin(harmed_tics)]
log(f"  Harmed: {harmed_df.tic_id.nunique()} stars  Helped: {helped_df.tic_id.nunique()} stars")
for label, sub in [("Harmed", harmed_df), ("Helped", helped_df)]:
    if not len(sub): continue
    log(f"  {label}:")
    log(f"    tmag   mean={sub.tmag.mean():.2f}  median={sub.tmag.median():.2f}")
    log(f"    n_sec  mean={sub.groupby('tic_id').sector.nunique().mean():.1f}")
    if "crowdsap" in sub.columns:
        log(f"    crowdsap  mean={sub.crowdsap.mean():.3f}")

log(f"\nDone eval  {T()}")

# ── Per-star dashboard data ────────────────────────────────────────────────────
log("\nBuilding per-star data ...")
tars_map = {}
if os.path.exists(TARS_CSV):
    tars = pd.read_csv(TARS_CSV)
    tars_map = {int(r.tic_id): float(r.mean_sys_score) for r in tars.itertuples()}

stars_out = []
n_harmed = n_harmed_n5 = n_cvz = 0
improvs = []

for tic, g in test_df.sort_values(["tic_id","sector"]).groupby("tic_id"):
    tic      = int(tic)
    raw_meds = g["sector_median"].tolist()
    cor_meds = (g["sector_median"] / g["pred_hgb"]).tolist()
    sectors  = [int(s) for s in g["sector"].tolist()]
    n        = len(sectors)
    raw_a    = np.array(raw_meds)
    cor_a    = np.array(cor_meds)
    sc_bef   = float(raw_a.std()/raw_a.mean()*100) if n >= 2 else 0.0
    sc_aft   = float(cor_a.std()/cor_a.mean()*100) if n >= 2 else 0.0
    improv   = float((sc_bef-sc_aft)/sc_bef*100) if sc_bef > 0 else 0.0
    harmed   = sc_aft > sc_bef
    if harmed:          n_harmed   += 1
    if harmed and n>=5: n_harmed_n5 += 1
    if n >= 30:         n_cvz      += 1
    improvs.append(improv)
    entry = {
        "tic_id":   tic,
        "tmag":     round(float(g["tmag"].mean()), 2),
        "n":        n,
        "cam":      int(g["cam"].mode()[0]),
        "ccd":      int(g["ccd"].mode()[0]),
        "sc_before":round(sc_bef, 3),
        "sc_after": round(sc_aft, 3),
        "improv":   round(improv, 1),
        "harmed":   harmed,
        "sectors":  sectors,
        "raw_meds": [round(v,5) for v in raw_meds],
        "cor_meds": [round(v,5) for v in cor_meds],
    }
    ss = tars_map.get(tic)
    if ss is not None: entry["sys_score"] = round(ss, 4)
    stars_out.append(entry)

imp_arr = [s["improv"] for s in stars_out if abs(s["improv"]) < 1000]
summary = {
    "model":         f"HGB-v3 β={best_beta:.2f} LSA+poly",
    "total_stars":   len(stars_out),
    "total_records": len(test_df),
    "harm_rate_all": round(n_harmed/len(stars_out)*100, 1),
    "harm_rate_n5":  round(n_harmed_n5/len(stars_out)*100, 1),
    "median_improv": round(float(np.median(imp_arr)), 1),
    "n_harmed":      n_harmed,
    "n_harmed_n5":   n_harmed_n5,
    "n_cvz":         n_cvz,
    "cv_raw_pct":    round(float(cv_raw), 4),
    "cv_hgb_pct":    round(float(cv_hgb), 4),
    "cv_oracle_pct": round(float(cv_oracle), 4),
    "beta":          best_beta,
}
log(f"  {len(stars_out):,} stars  median_improv={summary['median_improv']}%  "
    f"harm_n5={summary['harm_rate_n5']}%  n_cvz={n_cvz}")

split_of = {int(t): 0 for t in tr_tics}
split_of.update({int(t): 1 for t in vl_tics})
split_of.update({int(t): 2 for t in te_tics})
md_split=[]; md_cam=[]; md_ccd=[]; md_n=[]; md_tmag=[]; md_sc=[]; md_sys=[]
for tic, g in df.groupby("tic_id"):
    sp = split_of.get(int(tic))
    if sp is None: continue
    g    = g.sort_values("sector")
    sm   = g["sector_median"].values
    sc   = float(sm.std()/sm.mean()*100) if len(sm) >= 2 else 0.0
    tmag = float(g["tmag"].iloc[0]) if "tmag" in g.columns else None
    md_split.append(sp); md_cam.append(int(g["cam"].mode()[0]))
    md_ccd.append(int(g["ccd"].mode()[0])); md_n.append(int(len(g)))
    md_tmag.append(round(tmag, 2) if tmag else None)
    md_sc.append(round(sc, 3))
    ss = tars_map.get(int(tic))
    md_sys.append(round(ss, 4) if ss else None)

meta = {
    "tmag_min":   round(min(v for v in md_tmag if v), 1),
    "tmag_max":   round(max(v for v in md_tmag if v), 1),
    "n_min":      int(min(md_n)), "n_max": int(max(md_n)),
    "sc_max":     round(float(np.percentile(md_sc, 99)), 1),
    "improv_min": round(float(np.percentile(imp_arr, 1)), 0),
}

log("Building HTML ...")
lc_payload = json.dumps({"summary":summary,"stars":stars_out,"meta":meta},
                        separators=(",",":"))
md_payload = json.dumps({"split":md_split,"cam":md_cam,"ccd":md_ccd,
                         "n":md_n,"tmag":md_tmag,"sc":md_sc,"sys":md_sys},
                        separators=(",",":"))
log(f"  LC {len(lc_payload)//1024}KB  MD {len(md_payload)//1024}KB")

_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_lightcurve_dashboard.py")
with open(_script) as f:
    _src = f.read()
m = re.search(r'TEMPLATE\s*=\s*r"""(.*?)"""', _src, re.DOTALL)
if not m:
    raise RuntimeError("Could not extract TEMPLATE from build_lightcurve_dashboard.py")
html = m.group(1)

SUBS = {
    "%%TOTAL%%":     f"{len(stars_out):,}",
    "%%MEDIMPROV%%": f"{summary['median_improv']}%",
    "%%HARMRATE%%":  f"{summary['harm_rate_n5']}%",
    "%%NCVZ%%":      str(n_cvz),
    "%%MODEL%%":     summary["model"],
    "%%N_MAX%%":     str(meta["n_max"]),
    "%%TMAG_MIN%%":  str(meta["tmag_min"]),
    "%%TMAG_MAX%%":  str(meta["tmag_max"]),
    "%%SC_MAX%%":    str(meta["sc_max"]),
    "%%IMP_MIN%%":   str(int(meta["improv_min"])),
    "%%LC_JSON%%":   lc_payload,
    "%%MD_JSON%%":   md_payload,
}
for k, v in SUBS.items():
    html = html.replace(k, v)

with open(OUT, "w") as f:
    f.write(html)
log(f"\nWritten → {OUT}  ({os.path.getsize(OUT)/1024/1024:.1f} MB)  {T()}")
log("DONE")
