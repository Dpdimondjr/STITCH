"""
STITCH normalizing flow — v5.

New features over v4/KNN model:
  - spatial_poly_pred : 2D degree-3 polynomial fitted per (sector, cam, ccd) on
                        flux_offset_loo values. Uses all ~1000-3000 stars on the CCD
                        rather than K=20 neighbours.  r=0.624 vs KNN r=0.604.
  - loo_prev          : Previous (chronologically) sector's flux_offset_loo for the
                        same star. Captures temporal autocorrelation (r=0.207 direct;
                        r=0.077 partial after KNN). Falls back to sector_ccd_mean_loo
                        for first-observed sectors.
  - sector_gap        : Sector-number gap since the previous observation.  Tells the
                        model how much to trust loo_prev (gap>4 → near-zero signal).
                        Set to 99 for first observations.
  - n_sectors_total   : Number of sectors the star was observed in. Calibrates how
                        reliable the spatial mean features are.

Context dimension: 28D (was 24D in KNN model).
"""

import sys as _flush_sys
_flush_sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zuko
from torch.utils.data import DataLoader, TensorDataset

# ── 1. Load data ──────────────────────────────────────────────────────────────

import sys as _sys
_parquet   = next((s for s in _sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")
_out_pt    = next((s for s in _sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_v5.pt")
_tmag_arg  = next((s for s in _sys.argv[1:] if s.startswith("--tmag=")), None)
_target    = next((s.split("=")[1] for s in _sys.argv[1:] if s.startswith("--target=")), "flux_offset_loo")
TMAG_MAX   = float(_tmag_arg.split("=")[1]) if _tmag_arg else 13.0
print(f"Parquet: {_parquet}  →  {_out_pt}  (tmag <= {TMAG_MAX}, target={_target})")

df = pd.read_parquet(_parquet)
print(f"Loaded {len(df):,} records from {df['tic_id'].nunique():,} stars")

# ── 2. Feature engineering ────────────────────────────────────────────────────

cam_dummies = pd.get_dummies(df["cam"].astype(int), prefix="cam")
ccd_dummies = pd.get_dummies(df["ccd"].astype(int), prefix="ccd")

df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))

if "flux_offset" in df.columns and "gaiarp" in df.columns:
    df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")
    GAIA_FEATURES = ["gaiarp", "perstar_gaia_offset"]
    print("Gaia features: gaiarp + perstar_gaia_offset")
else:
    GAIA_FEATURES = []
    print("No Gaia features available")

CONTINUOUS = ["col", "row", "delta_sub_col", "delta_sub_row",
              "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
              "pdc_noi", "pr_wght2"] + GAIA_FEATURES

# ── 3. Clean data ─────────────────────────────────────────────────────────────

MIN_SECTORS = 4

df = df.dropna(subset=["col", "row", _target])
df = df[(df[_target] > 0.85) & (df[_target] < 1.15)]
df = df[df["n_sectors_total"] >= MIN_SECTORS]
df = df[df["tmag"] <= TMAG_MAX]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
print(f"After cleaning (n_sectors >= {MIN_SECTORS}, tmag <= {TMAG_MAX}): {len(df):,} records")

# Sector-CCD leave-one-out mean
_grp = df.groupby(["sector", "cam", "ccd"])[_target]
_grp_sum   = _grp.transform("sum")
_grp_count = _grp.transform("count")
df["sector_ccd_mean_loo"] = (_grp_sum - df[_target]) / (_grp_count - 1).clip(lower=1)
_r = df["sector_ccd_mean_loo"].corr(df[_target])
print(f"sector_ccd_mean_loo: std={df['sector_ccd_mean_loo'].std():.5f}  r={_r:.3f}")

# Spatial KNN leave-one-out mean (K=20)
from scipy.spatial import cKDTree as _cKDTree
_K = 20
print(f"Computing spatial KNN mean LOO (k={_K}) …", flush=True)
df = df.reset_index(drop=True)
_knn_vals = np.full(len(df), np.nan, dtype=np.float32)
for (_sec, _cam, _ccd), _g in df.groupby(["sector", "cam", "ccd"]):
    _n = len(_g)
    if _n < 2:
        continue
    _coords = _g[["col", "row"]].values.astype(np.float32)
    _vals   = _g[_target].values
    _k      = min(_K, _n - 1)
    _, _nn  = _cKDTree(_coords).query(_coords, k=_k + 1)
    _knn_vals[_g.index.values] = _vals[_nn[:, 1:]].mean(axis=1)
df["spatial_knn_mean_loo"] = _knn_vals
df["spatial_knn_mean_loo"] = df["spatial_knn_mean_loo"].fillna(df["sector_ccd_mean_loo"])
_r2 = df["spatial_knn_mean_loo"].corr(df[_target])
print(f"spatial_knn_mean_loo: std={df['spatial_knn_mean_loo'].std():.5f}  r={_r2:.3f}")

# 2D polynomial spatial field (degree 3): fits all stars on a CCD per sector,
# capturing large-scale gradients better than K=20 neighbours.
def _poly2d_design(c, r, degree=3):
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            terms.append(c ** (d - i) * r ** i)
    return np.column_stack(terms)

print("Computing 2D polynomial spatial field (degree=3) …", flush=True)
_poly_vals  = np.full(len(df), np.nan, dtype=np.float32)
_n_fallback = 0
for (_sec, _cam, _ccd), _g in df.groupby(["sector", "cam", "ccd"]):
    _idx = _g.index.values
    _n   = len(_g)
    if _n < 15:
        _poly_vals[_idx] = _g["sector_ccd_mean_loo"].values.astype(np.float32)
        _n_fallback += 1
        continue
    _c = _g["col"].values.astype(np.float64) - _g["col"].mean()
    _r = _g["row"].values.astype(np.float64) - _g["row"].mean()
    _y = _g[_target].values.astype(np.float64)
    _X = _poly2d_design(_c, _r, degree=3)
    try:
        _coef, _, _, _ = np.linalg.lstsq(_X, _y, rcond=None)
        _res  = _y - _X @ _coef
        _mask = np.abs(_res) <= 3.0 * (_res.std() or 1e-9)
        if _mask.sum() >= 15:
            _coef, _, _, _ = np.linalg.lstsq(_X[_mask], _y[_mask], rcond=None)
        _poly_vals[_idx] = np.clip(_X @ _coef, 0.85, 1.15).astype(np.float32)
    except Exception:
        _poly_vals[_idx] = _g["sector_ccd_mean_loo"].values.astype(np.float32)
        _n_fallback += 1
df["spatial_poly_pred"] = _poly_vals
_r3 = df["spatial_poly_pred"].corr(df[_target])
print(f"spatial_poly_pred: std={df['spatial_poly_pred'].std():.5f}  r={_r3:.3f}  fallbacks={_n_fallback}")

# Temporal feature: previous sector's LOO value for the same star.
# Captures autocorrelation across consecutive sectors (r=0.28 for gap=1).
print("Computing temporal LOO feature (previous sector) …", flush=True)
_loo_prev_arr  = np.full(len(df), np.nan, dtype=np.float32)
_sector_gap_arr = np.full(len(df), np.nan, dtype=np.float32)
for _tic, _g in df.groupby("tic_id"):
    _gs  = _g.sort_values("sector")
    _idx = _gs.index.values
    if len(_idx) < 2:
        continue
    _loo_prev_arr[_idx[1:]]   = _gs[_target].values[:-1].astype(np.float32)
    _sector_gap_arr[_idx[1:]] = np.diff(_gs["sector"].values).astype(np.float32)
df["loo_prev"]   = _loo_prev_arr
df["sector_gap"] = _sector_gap_arr
# No previous sector: fall back to spatial mean; mark gap as 99
_no_prev = df["loo_prev"].isna()
df.loc[_no_prev, "loo_prev"]   = df.loc[_no_prev, "sector_ccd_mean_loo"]
df.loc[_no_prev, "sector_gap"] = 99.0
df["sector_gap"] = df["sector_gap"].clip(upper=99)
_r4  = df["loo_prev"].corr(df[_target])
_cov = (df["sector_gap"] <= 3).mean()
print(f"loo_prev:   std={df['loo_prev'].std():.5f}  r={_r4:.3f}")
print(f"sector_gap: mean={df['sector_gap'].mean():.1f}  gap<=3: {_cov*100:.1f}% of rows")

CONTINUOUS = CONTINUOUS + ["sector_ccd_mean_loo", "spatial_knn_mean_loo",
                            "spatial_poly_pred", "loo_prev", "sector_gap", "n_sectors_total"]

# Weight by inverse oracle ceiling: quiet stars with many sectors dominate the loss.
# oracle_ceil_i = loo_std_i / sqrt(N_i - 1)  — lower = cleaner label.
# weight_i = 1 / oracle_ceil_i, bounded to [1, 20] then normalized to mean 1.
_star_stats = (
    df.groupby("tic_id")[_target]
    .agg(["std", "count"])
    .rename(columns={"std": "_loo_std", "count": "_n"})
    .reset_index()
)
_star_stats["_loo_std"] = _star_stats["_loo_std"].fillna(0.02).clip(lower=1e-5)
_star_stats["_oracle"]  = _star_stats["_loo_std"] / np.sqrt((_star_stats["_n"] - 1).clip(lower=1))
# Soft inverse weighting: weight = (median_oracle / oracle_ceil), clipped to [0.2, 5].
# Quietest stars (oracle_ceil = p10) get ~5x weight; noisiest (p90) get ~0.2x.
_med_oracle = float(_star_stats["_oracle"].median())
_star_stats["_w"] = (_med_oracle / _star_stats["_oracle"]).clip(lower=0.2, upper=5.0)
_star_stats["_w"] /= _star_stats["_w"].mean()   # normalize to mean 1
df = df.merge(_star_stats[["tic_id", "_w"]], on="tic_id", how="left")
df["sample_weight"] = df["_w"].fillna(1.0).astype(np.float32)
df = df.drop(columns=["_w"])

_wmed = df["sample_weight"].median()
_wmax = df["sample_weight"].max()
print(f"  Oracle-ceiling weights: median={_wmed:.2f}  max={_wmax:.2f}")
print(f"  n>=5: {(df['n_sectors_total']>=5).mean()*100:.1f}%  "
      f"n>=8: {(df['n_sectors_total']>=8).mean()*100:.1f}%")

# ── 4. Stratified star-level train/val/test split ─────────────────────────────

from sklearn.model_selection import train_test_split

star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))

train_tics, temp_tics = train_test_split(
    star_cam["tic_id"], test_size=0.2,
    stratify=star_cam["dominant_cam"], random_state=42,
)
temp_cam = star_cam[star_cam["tic_id"].isin(temp_tics)]["dominant_cam"]
val_tics, test_tics = train_test_split(
    temp_tics, test_size=0.5, stratify=temp_cam.values, random_state=42,
)

train_df = df[df["tic_id"].isin(train_tics)]
val_df   = df[df["tic_id"].isin(val_tics)]
test_df  = df[df["tic_id"].isin(test_tics)]

print(f"\nSplit (by star, stratified by cam):")
print(f"  Train: {len(train_df):5,} records, {train_df['tic_id'].nunique()} stars")
print(f"  Val:   {len(val_df):5,} records, {val_df['tic_id'].nunique()} stars")
print(f"  Test:  {len(test_df):5,} records, {test_df['tic_id'].nunique()} stars")

# ── 5. Normalisation ──────────────────────────────────────────────────────────

means = train_df[CONTINUOUS].mean()
stds  = train_df[CONTINUOUS].std().replace(0, 1)

def make_context(split_df):
    cont   = (split_df[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(split_df["cam"].astype(int), prefix="cam").reindex(
                 columns=cam_dummies.columns, fill_value=0)
    ccd_oh = pd.get_dummies(split_df["ccd"].astype(int), prefix="ccd").reindex(
                 columns=ccd_dummies.columns, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

y_mean = float(train_df[_target].mean())
y_std  = float(train_df[_target].std())

def make_target(split_df):
    return ((split_df[_target].values - y_mean) / y_std).astype(np.float32)

C_train = make_context(train_df);  y_train = make_target(train_df)
C_val   = make_context(val_df);    y_val   = make_target(val_df)
C_test  = make_context(test_df);   y_test  = make_target(test_df)

w_train = train_df["sample_weight"].values.astype(np.float32)

context_dim = C_train.shape[1]
print(f"\nContext dimension: {context_dim}")
print(f"Target y_mean={y_mean:.5f}  y_std={y_std:.5f}")

# ── 6. NSF model ──────────────────────────────────────────────────────────────

TRANSFORMS    = 8
HIDDEN        = [256, 256]
BINS          = 16

flow = zuko.flows.NSF(
    features=1,
    context=context_dim,
    transforms=TRANSFORMS,
    hidden_features=HIDDEN,
    bins=BINS,
)

n_params = sum(p.numel() for p in flow.parameters())
print(f"NSF parameters: {n_params:,}  ({TRANSFORMS} transforms, {BINS} bins, hidden={HIDDEN})")

# ── 7. Training ───────────────────────────────────────────────────────────────

BATCH_SIZE = 512
LR         = 3e-4
MAX_EPOCHS = 300
PATIENCE   = 25

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
print(f"Device: {device}")
flow   = flow.to(device)
opt    = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=1e-5)
sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

train_loader = DataLoader(
    TensorDataset(torch.tensor(C_train),
                  torch.tensor(y_train).unsqueeze(-1),
                  torch.tensor(w_train)),
    batch_size=BATCH_SIZE, shuffle=True,
)

C_val_t  = torch.tensor(C_val).to(device)
y_val_t  = torch.tensor(y_val).unsqueeze(-1).to(device)
C_test_t = torch.tensor(C_test).to(device)
y_test_t = torch.tensor(y_test).unsqueeze(-1).to(device)

best_val_nll  = float("inf")
best_state    = None
patience_count = 0

print("\nTraining NSF...\n  epoch  train_nll   val_nll    lr")
for epoch in range(1, MAX_EPOCHS + 1):
    flow.train()
    train_nlls = []
    for cb, yb, wb in train_loader:
        cb, yb, wb = cb.to(device), yb.to(device), wb.to(device)
        log_probs = flow(cb).log_prob(yb)
        nll = -(log_probs * wb).sum() / wb.sum()
        opt.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
        train_nlls.append(nll.item())

    flow.eval()
    with torch.no_grad():
        val_nll = -flow(C_val_t).log_prob(y_val_t).mean().item()

    sched.step(val_nll)
    train_nll = np.mean(train_nlls)

    if epoch % 10 == 0:
        lr_now = opt.param_groups[0]["lr"]
        print(f"  {epoch:5d}  {train_nll:.4f}      {val_nll:.4f}    {lr_now:.2e}")

    if val_nll < best_val_nll:
        best_val_nll  = val_nll
        best_state    = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

flow.load_state_dict(best_state)
print(f"\nBest val NLL: {best_val_nll:.4f}")

# ── 8. Evaluation ─────────────────────────────────────────────────────────────

flow.eval()
with torch.no_grad():
    samples = flow(C_test_t).sample((200,)).squeeze(-1)
    mu_test = samples.mean(0).cpu().numpy()

y_test_fo  = y_test  * y_std + y_mean
mu_test_fo = mu_test * y_std + y_mean

residuals = y_test_fo - mu_test_fo
baseline_mae = np.abs(y_test_fo - y_test_fo.mean()).mean()
model_mae    = np.abs(residuals).mean()

print(f"\n=== Test Set Evaluation ===")
print(f"  MAE:              {model_mae:.4f}")
print(f"  Residual std:     {residuals.std():.4f}")
print(f"  Residual mean:    {residuals.mean():.4f}")
print(f"  Baseline MAE:     {baseline_mae:.4f}")
print(f"  Improvement:      {(1 - model_mae/baseline_mae)*100:.1f}%")

print(f"\n  Per-camera breakdown (test set):")
print(f"  {'Cam':<6} {'n':>5} {'MAE':>8} {'baseline':>10} {'improvement':>12}")
print(f"  {'─'*50}")
test_df_reset = test_df.reset_index(drop=True)
for cam, cam_df in test_df_reset.groupby("cam"):
    pos    = cam_df.index.tolist()
    y_cam  = y_test_fo[pos]
    mu_cam = mu_test_fo[pos]
    mae    = np.abs(y_cam - mu_cam).mean()
    base   = np.abs(y_cam - y_cam.mean()).mean()
    print(f"  Cam{int(cam):<3} {len(y_cam):>5} {mae:>8.4f} {base:>10.4f} {(1-mae/base)*100:>11.1f}%")

# Within-star CV on test set
print(f"\n  Within-star CV (test set):")
test_df_reset["pred_fo"] = mu_test_fo
test_df_reset["raw_fo"]  = y_test_fo
cvs = []
for tic, g in test_df_reset.groupby("tic_id"):
    if len(g) < 2:
        continue
    raw_cv  = g["raw_fo"].std()  / g["raw_fo"].mean()
    pred_cv = (g["raw_fo"] / g["pred_fo"]).std() / (g["raw_fo"] / g["pred_fo"]).mean()
    cvs.append((raw_cv, pred_cv))
cv_raw  = np.median([c[0] for c in cvs]) * 100
cv_pred = np.median([c[1] for c in cvs]) * 100
print(f"  Raw median CV:   {cv_raw:.3f}%")
print(f"  Model median CV: {cv_pred:.3f}%")
print(f"  Reduction:       {(1 - cv_pred/cv_raw)*100:.1f}%")

# ── 9. Save ───────────────────────────────────────────────────────────────────

torch.save({
    "model_state":      best_state,
    "means":            means,
    "stds":             stds,
    "y_mean":           y_mean,
    "y_std":            y_std,
    "continuous_cols":  CONTINUOUS,
    "cam_cols":         list(cam_dummies.columns),
    "ccd_cols":         list(ccd_dummies.columns),
    "flow_config": {
        "features":        1,
        "context":         context_dim,
        "transforms":      TRANSFORMS,
        "hidden_features": HIDDEN,
        "bins":            BINS,
    },
}, _out_pt)
print(f"\nSaved → {_out_pt}")
