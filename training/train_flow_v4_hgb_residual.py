"""
STITCH NSF v4: flow trained on HGB v2 residuals.

Hypothesis: HGB v2 captures the systematic/spatial component cleanly.
The remaining residual log(actual/hgb_v2) should be smaller, closer to
Gaussian, and easier for the flow to model. The flow then learns either:
  (a) a calibration correction — where HGB v2 is systematically biased, and
  (b) the conditional uncertainty — how confident HGB is for each context.

Pipeline:
  1. Load HGB v2 artifacts (model, sec_stats, meta) from models/hgb_v2/
  2. Apply HGB v2 to val observations (out-of-sample for HGB)
  3. Train NSF on log(flux_offset_loo / pred_hgb_v2) using val as train set
  4. Evaluate combined prediction on test set

Combined prediction at test time:
  final = hgb_v2_pred * exp(flow_mean_correction(context))

If flow_mean_correction ≈ 0 everywhere, the flow adds nothing.
If non-trivial, the combined model beats pure HGB v2.

Requires: models/hgb_v2/ artifacts (run eval/hgb_v2.py first)

Usage (from STITCH root):
  python3 training/train_flow_v4_hgb_residual.py [parquet] [output.pt]
"""

import sys as _flush_sys
_flush_sys.stdout.reconfigure(line_buffering=True)

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zuko
import joblib
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PARQUET   = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_v3.parquet")
OUT_PT    = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_v4_residual.pt")
MODEL_DIR = "models/hgb_v2"
KNN_KS    = [5, 10, 20, 50]
SEED      = 42

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

# ── Load HGB v2 artifacts ──────────────────────────────────────────────────────
log(f"Loading HGB v2 from {MODEL_DIR}/ ...")
hgb_model  = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
sec_stats  = pd.read_parquet(os.path.join(MODEL_DIR, "sec_stats.parquet"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)
best_beta   = meta["best_beta"]
FEATS_HGB   = meta["features"]
ref_tics    = set(np.load(os.path.join(MODEL_DIR, "ref_tic_ids.npy")).tolist())
val_tics    = set(np.load(os.path.join(MODEL_DIR, "val_tic_ids.npy")).tolist())
te_tics     = set(np.load(os.path.join(MODEL_DIR, "test_tic_ids.npy")).tolist())
tr_tics     = set(np.load(os.path.join(MODEL_DIR, "train_tic_ids.npy")).tolist())
log(f"  β={best_beta:.2f}  features={len(FEATS_HGB)}D")
log(f"  Train {len(tr_tics):,}  Val {len(val_tics):,}  Test {len(te_tics):,}")

# ── Load data (same filters as hgb_v2.py) ─────────────────────────────────────
log(f"Loading {PARQUET} ...")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","flux_offset_loo","sector_median"])
df = df[(df.flux_offset > 0.85) & (df.flux_offset < 1.15)]
df = df[(df.flux_offset_loo > 0.85) & (df.flux_offset_loo < 1.15)]
df = df[df.tmag <= 13.0]
df["tic_id"] = df["tic_id"].astype("int64")
df["sector"] = df["sector"].astype("int64")
if "n_sectors_total" in df.columns:
    df = df[df["n_sectors_total"] >= 4]
df = df.reset_index(drop=True)
log(f"  {len(df):,} records · {df.tic_id.nunique():,} stars")

ref_full = df[~df.tic_id.isin(te_tics)].copy().reset_index(drop=True)
val_df   = df[ df.tic_id.isin(val_tics)].copy().reset_index(drop=True)
test_df  = df[ df.tic_id.isin(te_tics)].copy().reset_index(drop=True)

# ── Sector aggregates ─────────────────────────────────────────────────────────
def attach_sec_stats(df_, ss):
    df_.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"],
             errors="ignore", inplace=True)
    m = df_.merge(ss, on=["sector","cam","ccd"], how="left")
    df_["sector_mean_loo"]     = m["sector_mean_loo"].fillna(1.0).values
    df_["sector_std_loo"]      = m["sector_std_loo"].fillna(0.0).values
    df_["sector_frac_extreme"] = m["sector_frac_extreme"].fillna(0.0).values

attach_sec_stats(val_df, sec_stats)
attach_sec_stats(test_df, sec_stats)

# ── KNN-100 + spatial for val and test (from ref_full) ────────────────────────
log("KNN-100 baseline ...")

def knn_median_loo(ref, query, k=100):
    result = np.ones(len(query), dtype=np.float64)
    for (s, c, q), qg in query.groupby(["sector","cam","ccd"]):
        r = ref[(ref.sector==s)&(ref.cam==c)&(ref.ccd==q)]
        if len(r) < 2: continue
        coords = r[["col","row"]].to_numpy(np.float64)
        qcoords = qg[["col","row"]].to_numpy(np.float64)
        qids = qg.tic_id.to_numpy(); rids = r.tic_id.to_numpy()
        loo_vals = r.flux_offset_loo.to_numpy()
        take = min(k + int(r.tic_id.value_counts().max()), len(r))
        _, idxs = cKDTree(coords).query(qcoords, k=take)
        if take == 1: idxs = idxs[:, None]
        for i, (q_id, nn) in enumerate(zip(qids, idxs)):
            keep = rids[nn] != q_id
            vals = loo_vals[nn[keep][:k]]
            if len(vals) >= 1: result[qg.index[i]] = np.median(vals)
    return result

SPATIAL = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]
           + ["ref_count"])

def add_spatial(ref_df, eval_df):
    cols = {c: np.full(len(eval_df), np.nan, dtype=np.float32) for c in SPATIAL}
    for (s,c,q), gr in ref_df.groupby(["sector","cam","ccd"]):
        ge = eval_df[(eval_df.sector==s)&(eval_df.cam==c)&(eval_df.ccd==q)]
        if not len(ge): continue
        xyr = gr[["col","row"]].values.astype(np.float32)
        xye = ge[["col","row"]].values.astype(np.float32)
        vr  = gr.flux_offset_loo.values.astype(np.float32)
        tree = cKDTree(xyr)
        cols["ref_count"][ge.index.values] = len(gr)
        for k in KNN_KS:
            kk = min(k, len(gr))
            if kk < 1: continue
            d, nn = tree.query(xye, k=kk)
            if kk == 1: nn=nn[:,None]; d=d[:,None]
            nbv = vr[nn]; med = np.median(nbv, axis=1)
            cols[f"knn{k}_mean"]  [ge.index.values] = nbv.mean(axis=1)
            cols[f"knn{k}_median"][ge.index.values] = med
            cols[f"knn{k}_mad"]   [ge.index.values] = np.abs(nbv-med[:,None]).mean(axis=1)
            cols[f"knn{k}_radius"][ge.index.values] = d.max(axis=1)
    for c_name, arr in cols.items():
        eval_df[c_name] = arr
    for c_name in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
        eval_df[c_name] = eval_df[c_name].fillna(eval_df["sector_mean_loo"])
    eval_df["ref_count"] = eval_df["ref_count"].fillna(0.0)

for df_ in [val_df, test_df]:
    fp = df_.get("focal_plane_temp", pd.Series(np.nan, index=df_.index))
    df_["temp_x_col"]         = fp.values * (df_["col"].values / 2200.0)
    df_["temp_x_row"]         = fp.values * (df_["row"].values / 2200.0)
    df_["temp_x_crowdsap"]    = fp.values * df_["crowdsap"].values
    pt = df_.get("pdc_totp", pd.Series(np.nan, index=df_.index))
    df_["pdctotp_x_crowdsap"] = pt.values * df_["crowdsap"].values
    df_["n_sectors_obs"]      = df_.groupby("tic_id")["sector"].transform("nunique").astype(float)

knn_vl = knn_median_loo(ref_full, val_df)
knn_te = knn_median_loo(ref_full, test_df)
val_df["pred_knn"] = knn_vl; test_df["pred_knn"] = knn_te
add_spatial(ref_full, val_df); add_spatial(ref_full, test_df)
log(f"  Done  {T()}")

def get_X_hgb(df_):
    X = df_[FEATS_HGB].copy()
    for c in X.columns:
        if X[c].isna().any(): X[c] = X[c].fillna(X[c].median())
    return X.values.astype(np.float32)

# ── HGB v2 predictions (val and test — both out-of-sample for HGB) ────────────
log("Applying HGB v2 ...")
val_hgb_delta  = hgb_model.predict(get_X_hgb(val_df))
test_hgb_delta = hgb_model.predict(get_X_hgb(test_df))
val_df["pred_hgb_v2"]  = np.clip(knn_vl  * np.exp(best_beta * val_hgb_delta),  0.85, 1.15)
test_df["pred_hgb_v2"] = np.clip(knn_te  * np.exp(best_beta * test_hgb_delta), 0.85, 1.15)

def cv_col(df_, col, min_s=2):
    rows = []
    for _, g in df_.groupby("tic_id"):
        if g.sector.nunique() < min_s: continue
        c = g["sector_median"].values / g[col].values
        rows.append(float(c.std()/c.mean()*100))
    return pd.Series(rows)

cv_knn_val  = cv_col(val_df,  "pred_knn").median()
cv_hgb_val  = cv_col(val_df,  "pred_hgb_v2").median()
cv_knn_te   = cv_col(test_df, "pred_knn").median()
cv_hgb_te   = cv_col(test_df, "pred_hgb_v2").median()
log(f"  Val:  KNN={cv_knn_val:.4f}%  HGB={cv_hgb_val:.4f}%")
log(f"  Test: KNN={cv_knn_te:.4f}%  HGB={cv_hgb_te:.4f}%")

# ── Residual target for flow: log(actual / hgb_v2_pred) ───────────────────────
val_df["flow_target"]  = np.log(val_df["flux_offset_loo"].values) - np.log(val_df["pred_hgb_v2"].values)
test_df["flow_target"] = np.log(test_df["flux_offset_loo"].values) - np.log(test_df["pred_hgb_v2"].values)
log(f"  Residual std: val={val_df['flow_target'].std():.5f}  test={test_df['flow_target'].std():.5f}")
log(f"  (compare to raw log(loo) std ≈ {np.log(df['flux_offset_loo']).std():.5f})")

# ── Flow context features (18D production set) ────────────────────────────────
CONTINUOUS = ["col","row","delta_sub_col","delta_sub_row","sector","tmag",
              "crowdsap","cdpp1_0","pdcvar","jitter_rms","pdc_noi","pr_wght2"]

def build_context(df_):
    cam_oh = pd.get_dummies(df_["cam"].astype(int), prefix="cam")
    ccd_oh = pd.get_dummies(df_["ccd"].astype(int), prefix="ccd")
    cont   = df_[CONTINUOUS].copy()
    for c in cont.columns:
        if cont[c].isna().any(): cont[c] = cont[c].fillna(cont[c].median())
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1)

ctx_val  = build_context(val_df)
ctx_test = build_context(test_df)
# Align columns
all_ctx_cols = sorted(set(ctx_val.columns) | set(ctx_test.columns))
for df_, ctx in [(val_df, ctx_val), (test_df, ctx_test)]:
    for c in all_ctx_cols:
        if c not in ctx.columns: ctx[c] = 0.0

ctx_val  = ctx_val[all_ctx_cols]
ctx_test = ctx_test[all_ctx_cols]

# Normalise using val statistics (val is the training set for the flow)
c_mean = ctx_val.mean(); c_std = ctx_val.std().replace(0, 1)
C_val  = torch.tensor(((ctx_val  - c_mean) / c_std).values, dtype=torch.float32)
C_test = torch.tensor(((ctx_test - c_mean) / c_std).values, dtype=torch.float32)
y_val  = torch.tensor(val_df["flow_target"].values,  dtype=torch.float32)
y_test = torch.tensor(test_df["flow_target"].values, dtype=torch.float32)
log(f"  Context: {C_val.shape[1]}D")

# ── Train NSF on residuals ─────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"\nTraining NSF on residuals (device={device}) ...")

y_mean = float(y_val.mean()); y_std = float(y_val.std())
y_val_n  = (y_val  - y_mean) / y_std
y_test_n = (y_test - y_mean) / y_std

flow = zuko.flows.NSF(1, context=C_val.shape[1],
                      transforms=6, hidden_features=[128,128], bins=8).to(device)
optimizer = torch.optim.AdamW(flow.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

dataset = TensorDataset(C_val, y_val_n.unsqueeze(1))
loader  = DataLoader(dataset, batch_size=512, shuffle=True)

best_val_nll = float("inf"); best_state = None; patience = 0
for epoch in range(1, 81):
    flow.train()
    train_nll = 0.0; n = 0
    for cx, yb in loader:
        cx, yb = cx.to(device), yb.to(device)
        loss = -flow(cx).log_prob(yb).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_nll += float(loss) * len(yb); n += len(yb)
    scheduler.step()
    flow.eval()
    with torch.no_grad():
        val_nll = float(-flow(C_val.to(device)).log_prob(y_val_n.unsqueeze(1).to(device)).mean())
    if val_nll < best_val_nll:
        best_val_nll = val_nll; best_state = {k: v.cpu().clone() for k,v in flow.state_dict().items()}; patience = 0
    else:
        patience += 1
    if epoch % 10 == 0:
        log(f"  Epoch {epoch:3d}  train={train_nll/n:.4f}  val={val_nll:.4f}  patience={patience}")
    if patience >= 20:
        log(f"  Early stop at epoch {epoch}")
        break

flow.load_state_dict(best_state)
log(f"  Best val NLL = {best_val_nll:.4f}  {T()}")

# ── Combined test evaluation ───────────────────────────────────────────────────
log("\n=== COMBINED TEST RESULTS ===")
flow.eval()
with torch.no_grad():
    # Flow mean correction (MC estimate)
    n_mc = 1000
    C_te_dev = C_test.to(device)
    samples = flow(C_te_dev).sample((n_mc,)).squeeze(-1)  # (n_mc, n_obs)
    flow_correction_n = samples.mean(dim=0).cpu().numpy()  # normalised space
flow_correction = flow_correction_n * y_std + y_mean   # back to log space

test_df["pred_hgb_v2"]    = np.clip(knn_te * np.exp(best_beta * test_hgb_delta), 0.85, 1.15)
test_df["pred_combined"]  = np.clip(test_df["pred_hgb_v2"].values * np.exp(flow_correction), 0.85, 1.15)
test_df["pred_raw"]       = 1.0

cv_raw  = cv_col(test_df, "pred_raw").median()
cv_knn  = cv_col(test_df, "pred_knn").median()
cv_hgb  = cv_col(test_df, "pred_hgb_v2").median()
cv_comb = cv_col(test_df, "pred_combined").median()
log(f"  Raw:      {cv_raw:.4f}%")
log(f"  KNN:      {cv_knn:.4f}%")
log(f"  HGB v2:   {cv_hgb:.4f}%")
log(f"  Combined: {cv_comb:.4f}%  (Δ={cv_comb-cv_hgb:+.4f}pp vs HGB v2)")

# ── Save flow ─────────────────────────────────────────────────────────────────
log(f"\nSaving → {OUT_PT} ...")
torch.save({
    "flow_state": best_state,
    "context_mean": c_mean.to_dict(),
    "context_std":  c_std.to_dict(),
    "context_cols": list(all_ctx_cols),
    "y_mean": y_mean, "y_std": y_std,
    "val_nll": best_val_nll,
    "cv_hgb_test": float(cv_hgb),
    "cv_combined_test": float(cv_comb),
    "hgb_model_dir": MODEL_DIR,
}, OUT_PT)
log(f"Done  {T()}")
