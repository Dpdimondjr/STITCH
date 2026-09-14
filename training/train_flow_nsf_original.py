"""
STITCH NSF — original proposed conditioning vector ablation.

Features: col, row, delta_sub_col, delta_sub_row, tmag, crowdsap,
          focal_plane_temp (T_thermal), jitter_rms
          + cam/CCD one-hot (8D)
Total: 8 continuous + 8 one-hot = 16D context

No KNN spatial features, no PDC extras, no Gaia features.
Meant for direct comparison vs full conditioning vector.
"""

import sys as _flush_sys
_flush_sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import zuko
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

_parquet = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                "training_data_v3.parquet") if False else "training_data_v3.parquet"

import sys
_parquet = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_v3.parquet")
_out_pt  = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_original.pt")
_target  = next((s.split("=")[1] for s in sys.argv[1:] if s.startswith("--target=")), "flux_offset")
TMAG_MAX = 13.0

print(f"Parquet: {_parquet}  →  {_out_pt}  (target={_target})")

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_parquet(_parquet)
print(f"Loaded {len(df):,} records from {df['tic_id'].nunique():,} stars")

# ── 2. Features ───────────────────────────────────────────────────────────────
cam_dummies = pd.get_dummies(df["cam"].astype(int), prefix="cam")
ccd_dummies = pd.get_dummies(df["ccd"].astype(int), prefix="ccd")

# Original conditioning vector (eq. 1):
# [row, col, Δsub-pixel, cam, CCD, Tmag, ρ_crowd, T_thermal, σ_jitter, A_ap]
# A_ap skipped (68% missing, redundant with Tmag)
CONTINUOUS = ["col", "row", "delta_sub_col", "delta_sub_row",
              "tmag", "crowdsap", "focal_plane_temp", "jitter_rms"]

# ── 3. Clean ──────────────────────────────────────────────────────────────────
df = df.dropna(subset=["col", "row", _target])
df = df[(df[_target] > 0.85) & (df[_target] < 1.15)]
df = df[df["n_sectors_total"] >= 4]
df = df[df["tmag"] <= TMAG_MAX]

for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        med = df[col].median()
        n_fill = df[col].isna().sum()
        df[col] = df[col].fillna(med)
        if n_fill > 0:
            print(f"  Imputed {n_fill:,} missing values in {col} with median {med:.4f}")

print(f"After cleaning: {len(df):,} records")

df["sample_weight"] = (df["n_sectors_total"].clip(upper=20) / 20.0).astype(np.float32)

# ── 4. Split ──────────────────────────────────────────────────────────────────
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))

train_tics, temp_tics = train_test_split(
    star_cam["tic_id"], test_size=0.2,
    stratify=star_cam["dominant_cam"], random_state=42)
temp_cam = star_cam[star_cam["tic_id"].isin(temp_tics)]["dominant_cam"]
val_tics, test_tics = train_test_split(
    temp_tics, test_size=0.5, stratify=temp_cam.values, random_state=42)

train_df = df[df["tic_id"].isin(train_tics)]
val_df   = df[df["tic_id"].isin(val_tics)]
test_df  = df[df["tic_id"].isin(test_tics)]

print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

# ── 5. Normalise ──────────────────────────────────────────────────────────────
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

C_train = make_context(train_df); y_train = make_target(train_df)
C_val   = make_context(val_df);   y_val   = make_target(val_df)
C_test  = make_context(test_df);  y_test  = make_target(test_df)
w_train = train_df["sample_weight"].values.astype(np.float32)

context_dim = C_train.shape[1]
print(f"Context dim: {context_dim}  (should be 16)")

# ── 6. Model ──────────────────────────────────────────────────────────────────
TRANSFORMS = 8
HIDDEN     = [256, 256]
BINS       = 16

flow = zuko.flows.NSF(features=1, context=context_dim,
                      transforms=TRANSFORMS, hidden_features=HIDDEN, bins=BINS)
print(f"NSF params: {sum(p.numel() for p in flow.parameters()):,}")

# ── 7. Train ──────────────────────────────────────────────────────────────────
device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
print(f"Device: {device}")

flow = flow.to(device)
opt  = torch.optim.Adam(flow.parameters(), lr=3e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

train_loader = DataLoader(
    TensorDataset(torch.tensor(C_train),
                  torch.tensor(y_train).unsqueeze(-1),
                  torch.tensor(w_train)),
    batch_size=512, shuffle=True)

C_val_t  = torch.tensor(C_val).to(device)
y_val_t  = torch.tensor(y_val).unsqueeze(-1).to(device)
C_test_t = torch.tensor(C_test).to(device)

best_val_nll = float("inf")
best_state   = None
patience_cnt = 0

print("\nTraining...\n  epoch  train_nll   val_nll    lr")
for epoch in range(1, 301):
    flow.train()
    nlls = []
    for cb, yb, wb in train_loader:
        cb, yb, wb = cb.to(device), yb.to(device), wb.to(device)
        nll = -(flow(cb).log_prob(yb) * wb).sum() / wb.sum()
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
        nlls.append(nll.item())

    flow.eval()
    with torch.no_grad():
        val_nll = -flow(C_val_t).log_prob(y_val_t).mean().item()

    sched.step(val_nll)
    if epoch % 10 == 0:
        print(f"  {epoch:5d}  {np.mean(nlls):.4f}      {val_nll:.4f}    {opt.param_groups[0]['lr']:.2e}")

    if val_nll < best_val_nll:
        best_val_nll = val_nll
        best_state   = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= 25:
            print(f"\n  Early stopping at epoch {epoch}")
            break

flow.load_state_dict(best_state)
print(f"\nBest val NLL: {best_val_nll:.4f}")

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
flow.eval()
with torch.no_grad():
    samples  = flow(C_test_t).sample((200,)).squeeze(-1)
    mu_test  = samples.mean(0).cpu().numpy()

y_test_fo  = y_test  * y_std + y_mean
mu_test_fo = mu_test * y_std + y_mean
residuals  = y_test_fo - mu_test_fo

print(f"\n=== Test Set ===")
print(f"  MAE:          {np.abs(residuals).mean():.4f}")
print(f"  Residual std: {residuals.std():.4f}")

# Efficiency vs oracle
test_df_r = test_df.reset_index(drop=True)
if "flux_offset_loo" in test_df_r.columns:
    eff_rows = []
    for tic, g in test_df_r.groupby("tic_id"):
        n = len(g)
        if n < 2: continue
        pos     = g.index.tolist()
        y_raw   = g[_target].values
        y_pred  = mu_test_fo[pos]
        loo_std = g["flux_offset_loo"].std() if "flux_offset_loo" in g else y_raw.std()
        cv_raw    = y_raw.std() / y_raw.mean() if y_raw.mean() != 0 else np.nan
        cv_model  = (y_raw - y_pred).std() / y_raw.mean() if y_raw.mean() != 0 else np.nan
        cv_oracle = (loo_std / np.sqrt(n - 1)) / y_raw.mean() if y_raw.mean() != 0 else np.nan
        denom = cv_raw - cv_oracle
        if np.isfinite(denom) and abs(denom) > 1e-8:
            eff_rows.append((cv_raw - cv_model) / denom)
    if eff_rows:
        print(f"\n  Efficiency (median): {np.median(eff_rows)*100:.1f}%")
        print(f"  Efficiency (mean):   {np.mean(eff_rows)*100:.1f}%")

print(f"\n  Per-camera:")
for cam, cg in test_df_r.groupby("cam"):
    pos    = cg.index.tolist()
    y_c    = y_test_fo[pos]
    mu_c   = mu_test_fo[pos]
    mae    = np.abs(y_c - mu_c).mean()
    base   = np.abs(y_c - y_c.mean()).mean()
    print(f"  Cam{int(cam)}: n={len(y_c):,}  MAE={mae:.4f}  base={base:.4f}  "
          f"improv={100*(1-mae/base):.1f}%")

# ── 9. Save ───────────────────────────────────────────────────────────────────
torch.save({
    "model_state":     best_state,
    "means":           means,
    "stds":            stds,
    "y_mean":          y_mean,
    "y_std":           y_std,
    "continuous_cols": CONTINUOUS,
    "cam_cols":        list(cam_dummies.columns),
    "ccd_cols":        list(ccd_dummies.columns),
    "flow_config": {
        "features": 1, "context": context_dim,
        "transforms": TRANSFORMS, "hidden_features": HIDDEN, "bins": BINS,
    },
}, _out_pt)
print(f"\nSaved → {_out_pt}")
