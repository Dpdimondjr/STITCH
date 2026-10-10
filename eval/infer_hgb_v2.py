"""
HGB v2 inference: apply saved model to new observations.

Loads pre-trained artifacts from models/hgb_v2/:
  model.pkl          — trained HGB
  ref_tic_ids.npy    — TIC IDs that form the KNN reference set
  sec_stats.parquet  — sector-level LOO aggregates
  meta.json          — best_beta, feature list, etc.

Usage (from STITCH root):
  python3 eval/infer_hgb_v2.py [parquet] [output.parquet]

Outputs a parquet with one row per input observation, adding:
  pred_knn     — KNN-100 baseline
  pred_hgb_v2  — final HGB v2 correction
  hgb_delta    — log-residual prediction (raw model output)
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from scipy.spatial import cKDTree

PARQUET   = sys.argv[1] if len(sys.argv) > 1 else "training_data_v3.parquet"
OUT       = sys.argv[2] if len(sys.argv) > 2 else "hgb_v2_predictions.parquet"
MODEL_DIR = "models/hgb_v2"
KNN_KS    = [5, 10, 20, 50]

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

# ── Load artifacts ─────────────────────────────────────────────────────────────
log(f"Loading model from {MODEL_DIR}/ ...")
model      = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
ref_tics   = set(np.load(os.path.join(MODEL_DIR, "ref_tic_ids.npy")).tolist())
sec_stats  = pd.read_parquet(os.path.join(MODEL_DIR, "sec_stats.parquet"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)
best_beta = meta["best_beta"]
FEATS     = meta["features"]
log(f"  β={best_beta:.2f}  features={len(FEATS)}D")

# ── Load query data ───────────────────────────────────────────────────────────
log(f"Loading {PARQUET} ...")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","sector_median"])
df["tic_id"] = df["tic_id"].astype("int64")
df["sector"] = df["sector"].astype("int64")
df = df.reset_index(drop=True)
log(f"  {len(df):,} records · {df.tic_id.nunique():,} stars")

# ── Separate reference (for KNN) from query ────────────────────────────────────
ref_df   = df[df.tic_id.isin(ref_tics) & df.flux_offset_loo.between(0.85, 1.15)].copy().reset_index(drop=True)
query_df = df.copy().reset_index(drop=True)
log(f"  Reference: {ref_df.tic_id.nunique():,} stars  Query: {query_df.tic_id.nunique():,} stars")

# ── Sector aggregates ─────────────────────────────────────────────────────────
log("Attaching sector aggregates ...")
query_df.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"],
              errors="ignore", inplace=True)
merged = query_df.merge(sec_stats, on=["sector","cam","ccd"], how="left")
query_df["sector_mean_loo"]     = merged["sector_mean_loo"].fillna(1.0).values
query_df["sector_std_loo"]      = merged["sector_std_loo"].fillna(0.0).values
query_df["sector_frac_extreme"] = merged["sector_frac_extreme"].fillna(0.0).values

# ── KNN-100 baseline ──────────────────────────────────────────────────────────
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

knn = knn_median_loo(ref_df, query_df)
query_df["pred_knn"] = knn
log(f"  Done  {T()}")

# ── Spatial KNN features ──────────────────────────────────────────────────────
log("Spatial KNN features ...")
SPATIAL = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]
           + ["ref_count"])

cols = {c: np.full(len(query_df), np.nan, dtype=np.float32) for c in SPATIAL}
for (s, c, q), gr in ref_df.groupby(["sector","cam","ccd"]):
    ge = query_df[(query_df.sector == s) & (query_df.cam == c) & (query_df.ccd == q)]
    if not len(ge):
        continue
    xyr  = gr[["col","row"]].values.astype(np.float32)
    xye  = ge[["col","row"]].values.astype(np.float32)
    vr   = gr.flux_offset_loo.values.astype(np.float32)
    tree = cKDTree(xyr)
    cols["ref_count"][ge.index.values] = len(gr)
    for k in KNN_KS:
        kk = min(k, len(gr))
        if kk < 1:
            continue
        d, nn = tree.query(xye, k=kk)
        if kk == 1:
            nn = nn[:, None]; d = d[:, None]
        nbv = vr[nn]
        med = np.median(nbv, axis=1)
        cols[f"knn{k}_mean"]  [ge.index.values] = nbv.mean(axis=1)
        cols[f"knn{k}_median"][ge.index.values] = med
        cols[f"knn{k}_mad"]   [ge.index.values] = np.abs(nbv - med[:, None]).mean(axis=1)
        cols[f"knn{k}_radius"][ge.index.values] = d.max(axis=1)
for c_name, arr in cols.items():
    query_df[c_name] = arr
for c_name in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
    query_df[c_name] = query_df[c_name].fillna(query_df["sector_mean_loo"])
query_df["ref_count"] = query_df["ref_count"].fillna(0.0)
log(f"  Done  {T()}")

# ── Derived features ──────────────────────────────────────────────────────────
fp = query_df.get("focal_plane_temp", pd.Series(np.nan, index=query_df.index))
query_df["temp_x_col"]         = fp.values * (query_df["col"].values / 2200.0)
query_df["temp_x_row"]         = fp.values * (query_df["row"].values / 2200.0)
query_df["temp_x_crowdsap"]    = fp.values * query_df["crowdsap"].values
pt = query_df.get("pdc_totp", pd.Series(np.nan, index=query_df.index))
query_df["pdctotp_x_crowdsap"] = pt.values * query_df["crowdsap"].values
query_df["n_sectors_obs"]      = (query_df.groupby("tic_id")["sector"]
                                    .transform("nunique").astype(float))

# ── Feature matrix & prediction ────────────────────────────────────────────────
log("Predicting ...")
X = query_df[FEATS].copy()
for c in X.columns:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())
delta = model.predict(X.values.astype(np.float32))
query_df["hgb_delta"]    = delta
query_df["pred_hgb_v2"]  = np.clip(knn * np.exp(best_beta * delta), 0.85, 1.15)
log(f"  Done  {T()}")

# ── Write output ──────────────────────────────────────────────────────────────
out_cols = ["tic_id","sector","cam","ccd","pred_knn","hgb_delta","pred_hgb_v2"]
out_cols = [c for c in out_cols if c in query_df.columns]
query_df[out_cols].to_parquet(OUT, index=False)
log(f"\nWritten → {OUT}  ({os.path.getsize(OUT)/1024/1024:.1f} MB)  {T()}")
log("DONE")
