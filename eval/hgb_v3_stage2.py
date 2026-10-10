"""
HGB v3 Stage 2: per-sector residual model.

Loads hgb_v3 stage-2 data (models/hgb_v3/stage2_data/) and trains a second
HGB whose target is the per-sector deviation from the stage-1 prediction:

    stage2_target = log(flux_offset_loo) - log(hgb_v3_pred)

Features: LSA features (lsa{20,50,100}_{mean,median,std}), polynomial surface
value (poly_val), plus sector-specific tabular features. Critically, these
features already capture "is this sector unusual for stars at this position?" —
the core signal for per-sector correction.

Final prediction:
    hgb_v3_pred × exp(beta2 × stage2_delta)

Run AFTER hgb_v3.py has completed.

Usage (from STITCH root):
  python3 eval/hgb_v3_stage2.py
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MODEL_DIR  = "models/hgb_v3"
STAGE2_DIR = os.path.join(MODEL_DIR, "stage2_data")
SEED       = 42

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

# ── Load stage-2 data ─────────────────────────────────────────────────────────
log("Loading stage-2 data ...")
train_df = pd.read_parquet(os.path.join(STAGE2_DIR, "train.parquet"))
val_df   = pd.read_parquet(os.path.join(STAGE2_DIR, "val.parquet"))
test_df  = pd.read_parquet(os.path.join(STAGE2_DIR, "test.parquet"))
log(f"  Train {len(train_df):,} obs ({train_df.tic_id.nunique():,} stars)")
log(f"  Val   {len(val_df):,} obs ({val_df.tic_id.nunique():,} stars)")
log(f"  Test  {len(test_df):,} obs ({test_df.tic_id.nunique():,} stars)")

# ── Feature set: sector-varying signals only ──────────────────────────────────
# LSA features are the core: they directly measure "is this sector unusual?"
LSA_KS = [20, 50, 100]
LSA_FEATS = [f"lsa{k}_{s}" for k in LSA_KS for s in ["mean","median","std"]] + ["poly_val"]

# Additional sector-varying tabular features
SEC_FEATS = [c for c in ["focal_plane_temp","sector_mean_loo","sector_std_loo",
                          "sector_frac_extreme","n_sectors_obs"]
             if c in train_df.columns]

FEATS_S2 = [c for c in LSA_FEATS + SEC_FEATS if c in train_df.columns]
log(f"Stage-2 features: {len(FEATS_S2)}D = {FEATS_S2}")

# ── Target & feature matrix ───────────────────────────────────────────────────
resp_tr = train_df["stage2_target"].values.astype(np.float32)
resp_vl = val_df["stage2_target"].values.astype(np.float32)
log(f"  Target std train={resp_tr.std():.5f}  val={resp_vl.std():.5f}")

# ── True oracle: per-star mean (irreducible floor) ────────────────────────────
log("\n=== ORACLE: TRUE PER-STAR MEAN FLOOR ===")

def cv_col_df(df_, col, min_s=2):
    rows = []
    for _, g in df_.groupby("tic_id"):
        if g.sector.nunique() < min_s: continue
        c = g["sector_median"].values / g[col].values
        rows.append(float(c.std() / c.mean() * 100))
    return pd.Series(rows)

# Oracle = predict each star's mean sector_median for all its sectors
test_df["pred_star_mean"] = test_df.groupby("tic_id")["sector_median"].transform("mean")
cv_oracle = cv_col_df(test_df, "pred_star_mean").median()
cv_stage1 = cv_col_df(test_df, "pred_hgb_v3").median()
log(f"  True oracle (predict mean sector_median): {cv_oracle:.4f}%")
log(f"  Stage 1 (HGB v3):                         {cv_stage1:.4f}%")
log(f"  Reducible gap (oracle→stage1):             {cv_stage1 - cv_oracle:+.4f}pp")
log(f"  This gap = within-star sector variation that IS systematically predictable.")
test_df.drop(columns=["pred_star_mean"], inplace=True)

def get_X(df_):
    X = df_[FEATS_S2].copy()
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X.values.astype(np.float32)

X_tr = get_X(train_df)
X_vl = get_X(val_df)
X_te = get_X(test_df)

# ── CV metric (same as main) ───────────────────────────────────────────────────
def cv_col(df_, col, min_s=2):
    rows = []
    for _, g in df_.groupby("tic_id"):
        if g.sector.nunique() < min_s:
            continue
        c = g["sector_median"].values / g[col].values
        rows.append(float(c.std() / c.mean() * 100))
    return pd.Series(rows)

# ── Train stage-2 HGB ─────────────────────────────────────────────────────────
log("\nTraining stage-2 HGB ...")
# No progressive intercept removal — the per-sector target already has
# per-star mean ≈ 0 (stage1 prediction accounts for star-level effects).
# Use lightweight model: fewer leaves, since the per-sector signal is weaker.
s2_model = HistGradientBoostingRegressor(
    loss="absolute_error", learning_rate=0.05, max_iter=300,
    max_leaf_nodes=255, min_samples_leaf=40, l2_regularization=0.1,
    random_state=SEED, early_stopping=False)
s2_model.fit(X_tr, resp_tr)
log(f"  Done  {T()}")

# ── Beta2 sweep on val ────────────────────────────────────────────────────────
log("Beta2 sweep on val set ...")
s2_delta_vl = s2_model.predict(X_vl)
best_beta2, best_val_cv = 1.0, 1e9
for beta2 in np.linspace(0.0, 2.0, 41):
    combined = np.clip(val_df["pred_hgb_v3"].values * np.exp(float(beta2) * s2_delta_vl), 0.85, 1.15)
    val_df["_tmp"] = combined
    cv = cv_col(val_df, "_tmp").median()
    log(f"  β2={beta2:.2f}  val_CV={cv:.4f}%")
    if cv < best_val_cv:
        best_val_cv = cv
        best_beta2  = float(beta2)
log(f"  → Best β2={best_beta2:.2f}  val_CV={best_val_cv:.4f}%")
val_df.drop(columns=["_tmp"], inplace=True)

# ── Test evaluation ───────────────────────────────────────────────────────────
log("\n=== TEST RESULTS ===")
s2_delta_te = s2_model.predict(X_te)

test_df["pred_stage1"]   = test_df["pred_hgb_v3"]
test_df["pred_combined"] = np.clip(
    test_df["pred_hgb_v3"].values * np.exp(best_beta2 * s2_delta_te), 0.85, 1.15)

cv_s1   = cv_col(test_df, "pred_stage1").median()
cv_comb = cv_col(test_df, "pred_combined").median()

log(f"  Stage 1 (HGB v3) CV: {cv_s1:.4f}%")
log(f"  Combined (v3 + s2):  {cv_comb:.4f}%  β2={best_beta2:.2f}")
log(f"  Delta:               {cv_comb - cv_s1:+.4f}pp")
log(f"  vs HGB v2 baseline:  {cv_comb - 0.9444:+.4f}pp")

log("\nBy tmag:")
log(f"  {'Bin':>8}  {'n':>5}  {'stage1':>8}  {'combined':>10}")
for lo, hi in [(6,9),(9,10),(10,11),(11,12),(12,13)]:
    sub = test_df[test_df.sector_median.between(0.85, 1.15)]  # use full test
    # filter by tmag — need to get it from the original df
    # (stage2_data/test.parquet may not have tmag; just skip by-tmag here)
    pass

# Per-sector residual statistics
s2_delta_tr = s2_model.predict(X_tr)
log(f"\n  Stage-2 δ std train={s2_delta_tr.std():.5f}")
log(f"  Stage-2 δ std test= {s2_delta_te.std():.5f}")
log(f"  (vs target std train={resp_tr.std():.5f})")
log(f"  Fraction explained: {1 - (resp_tr - s2_delta_tr).var()/resp_tr.var():.3f} (train)")

# ── Save stage-2 model ────────────────────────────────────────────────────────
import joblib
s2_model_path = os.path.join(MODEL_DIR, "stage2_model.pkl")
joblib.dump(s2_model, s2_model_path)
s2_meta = {
    "best_beta2": best_beta2,
    "val_cv": float(best_val_cv),
    "features": FEATS_S2,
    "date": "2026-10-07",
}
with open(os.path.join(MODEL_DIR, "stage2_meta.json"), "w") as f:
    json.dump(s2_meta, f, indent=2)
log(f"\n  Saved → {s2_model_path}  {T()}")

log("\n=== SUMMARY ===")
log(f"  HGB v2 baseline:          0.9444%")
log(f"  HGB v3 (LSA+poly, s1):    {cv_s1:.4f}%")
log(f"  HGB v3 + stage2:          {cv_comb:.4f}%")
log("DONE")
