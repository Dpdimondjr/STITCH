"""
HGB + TIC + destripe features: adds contratio, numcont (TIC) AND two new
1D-marginal destriping features on top of HGB v2's 47D base set.

Why: KNN-100 is an isotropic (2D disk) estimator. Diagnostics (eval/strap_*.py)
found two real effects it structurally can't clean up:
  1. A smooth, large-scale column/row gradient present in every (cam,ccd),
     confirmed to SURVIVE the current KNN+HGB correction (per-sector stellar
     density is often too low for K=100 to stay local, diluting a trend that
     spans the full 2048px range).
  2. A genuine localized ~3-5% artifact at Cam3 CCD3, column ~1245 — confirmed
     to reproduce across independent random star-halves AND to persist in the
     current model's test residual, i.e. currently completely uncorrected.

Fix: compute per-(cam,ccd) 1D MARGINAL lookups (collapse over the orthogonal
axis AND over all sectors) from reference stars only. This is fixed-detector,
sector-independent information — safe to use as an exogenous feature, and by
construction immune to the isotropic-dilution problem since it's built from a
1D marginal, not a 2D radius search.

Otherwise identical to hgb_tic_features.py: same split, same progressive
boosting, same beta sweep.

Baseline to beat: HGB+TIC ensemble = 0.899% (current best)

Usage (from STITCH root):
  python3 eval/hgb_destripe_features.py
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.spatial import cKDTree

PARQUET    = "training_data_v3_tic_pointing.parquet"
OUT_DIR    = "models/hgb_final_combined_ensemble"
V2_DIR     = "models/hgb_v2"
N_ENSEMBLE = 5
MIN_OBS_SECTORS = 4
NEW_FEATS  = ["contratio", "numcont", "bp_rp"]
DESTRIPE_FEATS = ["col_destripe_loo", "row_destripe_loo"]
POINTING_FEATS = ["perr_col", "perr_row"]

# Destripe lookup construction params
N_BINS        = 128   # ~16px bins across a ~2048px range
SMOOTH_WINDOW = 3      # light smoothing (~48px) — preserves localized spikes
MIN_N_PER_BIN = 20
SHRINK_PRIOR  = 30     # pseudo-count pulling sparse bins toward 1.0

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

# ── Load v2 artifacts (split + feature list + sector stats) ───────────────────
log("Loading HGB v2 artifacts ...")
with open(os.path.join(V2_DIR, "meta.json")) as f:
    meta = json.load(f)
BASE_FEATS = meta["features"]
KNN_KS     = meta.get("knn_ks", [5, 10, 20, 50])

tr_tics = set(np.load(os.path.join(V2_DIR, "train_tic_ids.npy")).tolist())
vl_tics = set(np.load(os.path.join(V2_DIR, "val_tic_ids.npy")).tolist())
te_tics = set(np.load(os.path.join(V2_DIR, "test_tic_ids.npy")).tolist())
log(f"  Split: train={len(tr_tics):,}  val={len(vl_tics):,}  test={len(te_tics):,}")

# ── Load + filter data ────────────────────────────────────────────────────────
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

# Coverage check
for feat in NEW_FEATS:
    if feat in df.columns:
        nn = df[feat].notna().sum()
        log(f"  {feat}: {nn:,}/{len(df):,} non-null ({nn/len(df)*100:.1f}%)")
    else:
        log(f"  WARNING: {feat} not in parquet — skipping")
        NEW_FEATS = [f for f in NEW_FEATS if f != feat]

# log1p(contratio): overnight failure analysis found r=0.24 vs r=0.20 for raw
# contratio against |residual| — the relationship is flat below ~3% contratio
# then accelerates, so the log compression should give trees cleaner splits.
df["log1p_contratio"] = np.log1p(df["contratio"])
LOG_FEATS = ["log1p_contratio"]

FEATS = BASE_FEATS + [f for f in NEW_FEATS if f not in BASE_FEATS] + DESTRIPE_FEATS + POINTING_FEATS + LOG_FEATS
log(f"  Features: {len(FEATS)}D ({len(BASE_FEATS)}D base + {len(NEW_FEATS)} TIC + {len(DESTRIPE_FEATS)} destripe + {len(POINTING_FEATS)} pointing + {len(LOG_FEATS)} log)")
log(f"  {len(df):,} records · {df.tic_id.nunique():,} stars")

train_df = df[ df.tic_id.isin(tr_tics)].copy().reset_index(drop=True)
val_df   = df[ df.tic_id.isin(vl_tics)].copy().reset_index(drop=True)
test_df  = df[ df.tic_id.isin(te_tics)].copy().reset_index(drop=True)
ref_full = df[~df.tic_id.isin(te_tics)  ].copy().reset_index(drop=True)

# Sector aggregates
sec_stats = pd.read_parquet(os.path.join(V2_DIR, "sec_stats.parquet"))
for df_ in [train_df, val_df, test_df]:
    df_.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"],
             errors="ignore", inplace=True)
    merged = df_.merge(sec_stats, on=["sector","cam","ccd"], how="left")
    df_["sector_mean_loo"]     = merged["sector_mean_loo"].fillna(1.0).values
    df_["sector_std_loo"]      = merged["sector_std_loo"].fillna(0.0).values
    df_["sector_frac_extreme"] = merged["sector_frac_extreme"].fillna(0.0).values

# ── Destripe features (1D marginal, fixed per cam/ccd, train+val only) ───────
log("\nBuilding destripe lookups (1D marginal, ref_full = train+val only) ...")

def build_destripe_profile(ref_df, axis, n_bins=N_BINS, min_n=MIN_N_PER_BIN,
                            smooth_window=SMOOTH_WINDOW, prior=SHRINK_PRIOR):
    profiles = {}
    for (cam, ccd), g in ref_df.groupby(["cam", "ccd"]):
        lo, hi = g[axis].min(), g[axis].max()
        bins = np.linspace(lo, hi, n_bins + 1)
        gb = g.copy()
        gb["_bin"] = np.clip(np.digitize(gb[axis].values, bins) - 1, 0, n_bins - 1)
        prof = gb.groupby("_bin")["flux_offset_loo"].agg(["median", "count"])
        prof = prof.reindex(range(n_bins))
        # shrink sparse bins toward 1.0
        cnt = prof["count"].fillna(0)
        med = prof["median"].fillna(1.0)
        shrunk = (med * cnt + 1.0 * prior) / (cnt + prior)
        smoothed = shrunk.rolling(smooth_window, center=True, min_periods=1).median()
        profiles[(cam, ccd)] = (bins, smoothed.values)
    return profiles

def apply_destripe(df_, profiles, axis, colname):
    vals = np.ones(len(df_), dtype=np.float32)
    for (cam, ccd), (bins, prof_vals) in profiles.items():
        mask = (df_["cam"].values == cam) & (df_["ccd"].values == ccd)
        if not mask.any():
            continue
        idx = np.clip(np.digitize(df_.loc[mask, axis].values, bins) - 1, 0, len(prof_vals) - 1)
        vals[mask] = prof_vals[idx]
    df_[colname] = vals

col_profiles = build_destripe_profile(ref_full, "col")
row_profiles = build_destripe_profile(ref_full, "row")

for df_ in [train_df, val_df, test_df]:
    apply_destripe(df_, col_profiles, "col", "col_destripe_loo")
    apply_destripe(df_, row_profiles, "row", "row_destripe_loo")

for name, prof in [("col_destripe_loo", col_profiles), ("row_destripe_loo", row_profiles)]:
    amps = [v.max() - v.min() for (_, v) in prof.values()]
    log(f"  {name}: amplitude range across (cam,ccd) = {min(amps):.4f} to {max(amps):.4f}")
    r_tr = train_df[name].corr(train_df["flux_offset_loo"])
    log(f"    corr with flux_offset_loo (train): r={r_tr:.3f}")

log(f"  Done  {T()}")

# ── KNN-100 baseline ──────────────────────────────────────────────────────────
log("\nKNN-100 baseline ...")
def knn_median_loo(ref, query, k=100):
    result = np.ones(len(query), dtype=np.float64)
    for (s, c, q), qg in query.groupby(["sector","cam","ccd"]):
        r = ref[(ref.sector == s) & (ref.cam == c) & (ref.ccd == q)]
        if len(r) < 2: continue
        coords  = r[["col","row"]].to_numpy(np.float64)
        qcoords = qg[["col","row"]].to_numpy(np.float64)
        rids    = r.tic_id.to_numpy()
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

# ── Spatial features ──────────────────────────────────────────────────────────
log("Spatial features ...")
SPATIAL = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]
           + ["ref_count"])

def add_spatial(ref_df, eval_df):
    arrays = {c: np.full(len(eval_df), np.nan, dtype=np.float32) for c in SPATIAL}
    for (s, c, q), gr in ref_df.groupby(["sector","cam","ccd"]):
        ge = eval_df[(eval_df.sector == s) & (eval_df.cam == c) & (eval_df.ccd == q)]
        if not len(ge): continue
        xyr  = gr[["col","row"]].values.astype(np.float32)
        xye  = ge[["col","row"]].values.astype(np.float32)
        vr   = gr.flux_offset_loo.values.astype(np.float32)
        tree = cKDTree(xyr)
        arrays["ref_count"][ge.index.values] = len(gr)
        kk = min(max(KNN_KS), len(gr))
        if kk < 1: continue
        d_all, nn_all = tree.query(xye, k=kk)
        if kk == 1: nn_all = nn_all[:, None]; d_all = d_all[:, None]
        for k in KNN_KS:
            ak = min(k, kk); nn_k = nn_all[:, :ak]; d_k = d_all[:, :ak]
            nbv = vr[nn_k]; med = np.median(nbv, axis=1)
            arrays[f"knn{k}_mean"][ge.index.values]   = nbv.mean(axis=1)
            arrays[f"knn{k}_median"][ge.index.values] = med
            arrays[f"knn{k}_mad"][ge.index.values]    = np.abs(nbv - med[:,None]).mean(axis=1)
            arrays[f"knn{k}_radius"][ge.index.values] = d_k.max(axis=1)
    for c_name, arr in arrays.items():
        eval_df[c_name] = arr
    for cn in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
        eval_df[cn] = eval_df[cn].fillna(eval_df["sector_mean_loo"])
    eval_df["ref_count"] = eval_df["ref_count"].fillna(0.0)

add_spatial(ref_full, train_df)
add_spatial(ref_full, val_df)
add_spatial(ref_full, test_df)
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

# ── Train ensemble ────────────────────────────────────────────────────────────
log(f"\nTraining {N_ENSEMBLE} ensemble members ...")
resp_base = np.log(train_df.sector_median.values) - np.log(knn_tr)
ids_tr    = train_df.tic_id.to_numpy()

def group_median(vals, ids):
    return pd.Series(vals).groupby(pd.Series(ids)).transform("median").to_numpy()

N_ROUNDS = 5
HGB_PARAMS = dict(
    loss="absolute_error", learning_rate=0.06,
    max_iter=250, max_leaf_nodes=1023,
    min_samples_leaf=40, l2_regularization=0.1,
    early_stopping=False, verbose=0
)

os.makedirs(OUT_DIR, exist_ok=True)
val_preds_list  = []
test_preds_list = []
val_cvs  = []
test_cvs = []

for e in range(N_ENSEMBLE):
    seed_base = 42 + e * 100
    log(f"\n  Ensemble member {e+1}/{N_ENSEMBLE} (seed base={seed_base}) ...")
    r    = resp_base.copy()
    d_vl = np.zeros(len(val_df))
    d_te = np.zeros(len(test_df))
    for rnd in range(N_ROUNDS):
        log(f"    Round {rnd+1}/{N_ROUNDS}  {T()}")
        r -= group_median(r, ids_tr)
        m = HistGradientBoostingRegressor(
            random_state=seed_base + rnd, **HGB_PARAMS
        )
        m.fit(X_tr, r)
        d_vl += m.predict(X_vl)
        d_te += m.predict(X_te)
        r     = r - m.predict(X_tr)
    with open(os.path.join(OUT_DIR, f"meta_{e}.json"), "w") as f:
        json.dump({"rounds": N_ROUNDS, "seed_base": seed_base, "n_feats": len(FEATS)}, f)
    val_preds_list.append(knn_vl * np.exp(d_vl))
    test_preds_list.append(knn_te * np.exp(d_te))

log("\nBeta sweep (ensemble) ...")
val_delta_avg  = np.mean([np.log(p / knn_vl) for p in val_preds_list], axis=0)
test_delta_avg = np.mean([np.log(p / knn_te) for p in test_preds_list], axis=0)

best_beta, best_val = 1.0, 1e9
for beta in np.linspace(0.5, 2.0, 31):
    val_df["pred_ens"] = knn_vl * np.exp(beta * val_delta_avg)
    cv = cv_col(val_df, "pred_ens").median()
    log(f"  β={beta:.2f}  val_CV={cv:.4f}%")
    if cv < best_val:
        best_val, best_beta = cv, beta

log(f"  → Best β={best_beta:.2f}  val_CV={best_val:.4f}%")

log("\nIndividual member val/test CVs:")
for e_i, (vp, tp) in enumerate(zip(val_preds_list, test_preds_list)):
    val_df["_tmp"] = vp
    vc = cv_col(val_df, "_tmp").median()
    test_df["_tmp"] = tp
    tc = cv_col(test_df, "_tmp").median()
    val_cvs.append(vc); test_cvs.append(tc)
    log(f"  Member {e_i+1}: val_CV={vc:.4f}%  test_CV={tc:.4f}%")
log(f"  Mean val ± std:  {np.mean(val_cvs):.4f}% ± {np.std(val_cvs):.4f}%")
log(f"  Mean test ± std: {np.mean(test_cvs):.4f}% ± {np.std(test_cvs):.4f}%")

val_df["pred_ens"]  = knn_vl * np.exp(best_beta * val_delta_avg)
test_df["pred_ens"] = knn_te * np.exp(best_beta * test_delta_avg)
final_val  = cv_col(val_df,  "pred_ens").median()
final_test = cv_col(test_df, "pred_ens").median()

log(f"\n=== TEST RESULTS ===")
log(f"  Features: {len(FEATS)}D ({len(BASE_FEATS)} base + {len(NEW_FEATS)} TIC + {len(DESTRIPE_FEATS)} destripe + {len(POINTING_FEATS)} pointing + {len(LOG_FEATS)} log)")
log(f"  New features: {NEW_FEATS + DESTRIPE_FEATS + POINTING_FEATS + LOG_FEATS}")
log(f"  Ensemble val CV:  {final_val:.4f}%  (β={best_beta:.2f})")
log(f"  Ensemble test CV: {final_test:.4f}%")
log(f"  vs HGB+TIC ensemble (0.899% original best): {final_test - 0.899:+.4f}pp")
log(f"  vs destripe+pointing (0.8926% overnight best): {final_test - 0.8926:+.4f}pp")

meta_out = {
    "test_cv": final_test, "val_cv": final_val,
    "best_beta": best_beta,
    "test_cv_members": test_cvs,
    "test_cv_std": float(np.std(test_cvs)),
    "features": FEATS,
    "new_features": NEW_FEATS,
    "destripe_features": DESTRIPE_FEATS,
    "n_ensemble": N_ENSEMBLE,
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta_out, f, indent=2)
log(f"\nSaved → {OUT_DIR}/  {T()}")
log("DONE")
