"""
STITCH normalizing flow — v2, Neural Spline Flow (NSF).

Uses zuko's NSF implementation: a stack of rational-quadratic spline coupling
transforms conditioned on the detector/star context vector c.

Why NSF over the Gaussian baseline (train_flow.py):
  - flux_offset residuals have kurtosis ~9 (vs 0 for Gaussian) and are
    negatively skewed — the Gaussian assumption is clearly violated.
  - NSF learns an invertible monotone transformation from N(0,1) to the
    true conditional distribution p(flux_offset | c), capturing heavy tails
    and asymmetry without assuming a parametric form.

Architecture:
  - Context c: 18D (10 continuous + 4 cam-OHE + 4 ccd-OHE), z-scored
  - Flow: zuko.flows.NSF with T=8 transforms, hidden=[128,128], K=8 spline bins
  - Base: standard Gaussian (transformed by the spline stack)
  - Output: samples and log-probabilities under p(flux_offset | c)
"""

import sys as _flush_sys
_flush_sys.stdout.reconfigure(line_buffering=True)   # flush every print immediately

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zuko
from torch.utils.data import DataLoader, TensorDataset

# Matched initialization and minibatch order across feature variants.
np.random.seed(20260915)
torch.manual_seed(20260915)

# ── 1. Load data ──────────────────────────────────────────────────────────────

import sys as _sys
_parquet   = next((s for s in _sys.argv[1:] if s.endswith(".parquet")), "training_data.parquet")
_out_pt    = next((s for s in _sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf.pt")
_tmag_arg  = next((s for s in _sys.argv[1:] if s.startswith("--tmag=")), None)
_target    = next((s.split("=")[1] for s in _sys.argv[1:] if s.startswith("--target=")), "flux_offset_loo")
# --weight-mode=legacy   clips n_sectors to 20, divides by 20  (original)
# --weight-mode=sharp    uses (n_sectors-1) directly, rewards well-observed stars more
_weight_mode = next((s.split("=")[1] for s in _sys.argv[1:] if s.startswith("--weight-mode=")), "legacy")
TMAG_MAX   = float(_tmag_arg.split("=")[1]) if _tmag_arg else 13.0
print(f"Parquet: {_parquet}  →  {_out_pt}  "
      f"(tmag <= {TMAG_MAX}, target={_target}, weight-mode={_weight_mode})")

df = pd.read_parquet(_parquet)
print(f"Loaded {len(df):,} records from {df['tic_id'].nunique():,} stars")

# ── 2. Feature engineering ────────────────────────────────────────────────────

cam_dummies = pd.get_dummies(df["cam"].astype(int), prefix="cam")
ccd_dummies = pd.get_dummies(df["ccd"].astype(int), prefix="ccd")

# log_sector_median: absolute flux level (e-/s) spans 4–800K, log-compress.
df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))

# per-star mean Gaia label: how much this star deviates from its Gaia RP
# expected flux on average across sectors — an absolute calibration anchor.
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

# Sector-CCD leave-one-out mean: mean LOO offset of all OTHER stars on same
# chip in the same sector. Directly captures the local spatial systematic.
_grp = df.groupby(["sector", "cam", "ccd"])[_target]
_grp_sum   = _grp.transform("sum")
_grp_count = _grp.transform("count")
df["sector_ccd_mean_loo"] = (_grp_sum - df[_target]) / (_grp_count - 1).clip(lower=1)
_r = df["sector_ccd_mean_loo"].corr(df[_target])
print(f"sector_ccd_mean_loo: std={df['sector_ccd_mean_loo'].std():.5f}  r={_r:.3f}")

# Spatial KNN features.
# If the parquet already has robust features (v4+), use them directly.
# Otherwise compute mean KNN on-the-fly (legacy behaviour for v3 and earlier).
from scipy.spatial import cKDTree as _cKDTree
_K = 20
df = df.reset_index(drop=True)

_HAS_ROBUST = ("spatial_knn_median_loo" in df.columns and
               df["spatial_knn_median_loo"].notna().mean() > 0.8)

if _HAS_ROBUST:
    print("Using pre-computed robust KNN features from parquet (v4+).")
    # median_loo fills NaN with sector_ccd_mean_loo (also ~1.0, same scale — correct)
    # mad_loo fills NaN with 0 — singletons have undefined spread, safest value is 0
    # (filling MAD with sector_ccd_mean_loo ≈ 1.0 would corrupt the feature badly)
    df["spatial_knn_median_loo"] = df["spatial_knn_median_loo"].fillna(df["sector_ccd_mean_loo"])
    df["spatial_knn_mad_loo"]    = df["spatial_knn_mad_loo"].fillna(0.0)
    _r_med = df["spatial_knn_median_loo"].corr(df[_target])
    _r_mad = df["spatial_knn_mad_loo"].corr(df[_target])
    print(f"  spatial_knn_median_loo: r={_r_med:.4f}  std={df['spatial_knn_median_loo'].std():.5f}")
    print(f"  spatial_knn_mad_loo:    r={_r_mad:.4f}  std={df['spatial_knn_mad_loo'].std():.5f}")
    # Retain the legacy mean so this is an additive feature ablation.
    if "spatial_knn_mean_loo" not in df.columns:
        df["spatial_knn_mean_loo"] = df["spatial_knn_median_loo"]
    SPATIAL_FEATURES = ["sector_ccd_mean_loo", "spatial_knn_mean_loo",
                        "spatial_knn_median_loo", "spatial_knn_mad_loo"]
else:
    # Legacy: compute mean KNN on-the-fly (v3 and earlier parquets)
    print(f"Computing spatial KNN mean LOO (k={_K}) …", flush=True)
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
    SPATIAL_FEATURES = ["sector_ccd_mean_loo", "spatial_knn_mean_loo"]

CONTINUOUS = CONTINUOUS + SPATIAL_FEATURES
print(f"Spatial features: {SPATIAL_FEATURES}")

# Sample weighting — rewards well-observed stars with cleaner LOO labels.
# legacy: clip(n,20)/20    — original behaviour
# sharp:  (n-1)/max(n-1)  — proportional to label variance reduction
if _weight_mode == "sharp":
    _n_obs = df["n_sectors_total"].clip(lower=2)
    _raw_w = (_n_obs - 1).astype(np.float32)
    df["sample_weight"] = (_raw_w / _raw_w.max()).astype(np.float32)
    print(f"Weight mode: sharp (∝ N−1)  mean={df['sample_weight'].mean():.3f}")
else:
    df["sample_weight"] = (df["n_sectors_total"].clip(upper=20) / 20.0).astype(np.float32)
    print(f"Weight mode: legacy (clip(N,20)/20)  mean={df['sample_weight'].mean():.3f}")
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

# NSF works best when the target is ~N(0,1), so also standardise the target.
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
# zuko.flows.NSF(features, context, transforms, hidden_features, bins)
# features=1  : target is 1D (flux_offset)
# context=D   : conditioning vector dimension
# transforms  : number of spline coupling layers
# bins        : number of rational-quadratic spline bins per layer

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
        log_probs = flow(cb).log_prob(yb)           # (batch,)
        nll = -(log_probs * wb).sum() / wb.sum()    # weighted mean NLL
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
    # Point estimate: use mean of the learned distribution (via samples)
    samples = flow(C_test_t).sample((200,)).squeeze(-1)  # (200, N_test)
    mu_test = samples.mean(0).cpu().numpy()               # (N_test,)

# Convert back to flux_offset units
y_test_fo  = y_test  * y_std + y_mean   # already numpy (built from pandas)
mu_test_fo = mu_test * y_std + y_mean

residuals = y_test_fo - mu_test_fo
baseline_mae = np.abs(y_test_fo - y_test_fo.mean()).mean()
model_mae    = np.abs(residuals).mean()

print(f"\n=== Test Set Evaluation ===")
print(f"  Mean absolute error:  {model_mae:.4f}")
print(f"  Residual std:         {residuals.std():.4f}")
print(f"  Residual mean:        {residuals.mean():.4f}  (should be ~0)")
print(f"\n  Baseline (global mean): MAE = {baseline_mae:.4f}")
print(f"  Improvement over baseline:  {(1 - model_mae/baseline_mae)*100:.1f}%")

print(f"\n  Per-camera breakdown (test set):")
print(f"  {'Cam':<6} {'n':>5} {'MAE':>8} {'baseline':>10} {'improvement':>12}")
print(f"  {'─'*50}")
train_cam_counts = train_df.groupby("cam").size()
test_df_reset = test_df.reset_index(drop=True)
for cam, cam_df in test_df_reset.groupby("cam"):
    pos     = cam_df.index.tolist()
    y_cam   = y_test_fo[pos]
    mu_cam  = mu_test_fo[pos]
    mae     = np.abs(y_cam - mu_cam).mean()
    base    = np.abs(y_cam - y_cam.mean()).mean()
    improv  = (1 - mae / base) * 100
    n_train = train_cam_counts.get(cam, 0)
    print(f"  Cam{int(cam):<3} {len(y_cam):>5} {mae:>8.4f} {base:>10.4f} {improv:>11.1f}%")

# ── 9. Save ───────────────────────────────────────────────────────────────────

import time as _time
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
    "training_provenance": {
        "parquet":         _parquet,
        "target":          _target,
        "weight_mode":     _weight_mode,
        "spatial_features": SPATIAL_FEATURES,
        "has_robust_knn":  _HAS_ROBUST,
        "tmag_max":        TMAG_MAX,
        "n_train_stars":   int(train_df["tic_id"].nunique()),
        "n_test_stars":    int(test_df["tic_id"].nunique()),
        "date":            _time.strftime("%Y-%m-%d"),
    },
}, _out_pt)
print(f"\nSaved → {_out_pt}")
