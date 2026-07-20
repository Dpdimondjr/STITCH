"""
STITCH normalizing flow — v1 (conditional Gaussian baseline).

Architecture: MLP conditioned on c → (mu, log_sigma) → N(mu, sigma^2)
This is the simplest possible "flow" — one transformation, Gaussian output.
If residuals show non-Gaussian structure, upgrade to NSF/MAF.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 1. Load data ───────────────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
print(f"Loaded {len(df)} records from {df['tic_id'].nunique()} stars")

# ── 2. Feature engineering ────────────────────────────────────────────────────

# One-hot encode cam and ccd (4 cameras × 4 CCDs = 8 binary columns)
cam_dummies = pd.get_dummies(df["cam"].astype(int), prefix="cam")
ccd_dummies = pd.get_dummies(df["ccd"].astype(int), prefix="ccd")

# Continuous features
CONTINUOUS = ["col", "row", "delta_sub_col", "delta_sub_row",
              "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms"]

# Drop rows missing the target or position features (critical).
# Impute remaining NaNs with training-set median — cdpp1_0/pdcvar are often
# unpopulated in TESS-SPOC FFI headers; dropping them loses 40%+ of records.
CRITICAL = ["col", "row", "flux_offset"]
df = df.dropna(subset=CRITICAL)
for col in CONTINUOUS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Remove photometry failures: offsets this far from 1.0 are bad data, not PRF effects.
# Keeps 99%+ of records while eliminating contaminated sectors and download artifacts.
n_before = len(df)
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
print(f"After outlier clip [0.85, 1.15]: {len(df)} records ({n_before - len(df)} removed)")
print(f"After dropping NaNs: {len(df)} records")

# ── 3. Stratified star-level train/val/test split ─────────────────────────────
# Split by star (tic_id), not by record, to prevent data leakage.
# Stratify by dominant cam to ensure Camera 1/2 representation in all splits.

# Each star gets one "dominant cam" label (the cam with most records for that star)
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))

# 80/10/10 split — first cut off 20% (val+test), then split that in half
from sklearn.model_selection import train_test_split

train_tics, temp_tics = train_test_split(
    star_cam["tic_id"],
    test_size=0.2,
    stratify=star_cam["dominant_cam"],
    random_state=42,
)
temp_cam = star_cam[star_cam["tic_id"].isin(temp_tics)]["dominant_cam"]
val_tics, test_tics = train_test_split(
    temp_tics,
    test_size=0.5,
    stratify=temp_cam.values,
    random_state=42,
)

train_df = df[df["tic_id"].isin(train_tics)]
val_df   = df[df["tic_id"].isin(val_tics)]
test_df  = df[df["tic_id"].isin(test_tics)]

print(f"\nSplit (by star, stratified by cam):")
print(f"  Train: {len(train_df):5d} records, {train_df['tic_id'].nunique()} stars")
print(f"  Val:   {len(val_df):5d} records, {val_df['tic_id'].nunique()} stars")
print(f"  Test:  {len(test_df):5d} records, {test_df['tic_id'].nunique()} stars")
print(f"\n  Cam distribution in train:")
for cam, g in train_df.groupby("cam"):
    print(f"    Cam{int(cam)}: {len(g)} records")

# ── 4. Z-score normalization (fit on train only, apply to all) ─────────────────
# Fitting on train only prevents test information leaking into normalization.

means = train_df[CONTINUOUS].mean()
stds  = train_df[CONTINUOUS].std().replace(0, 1)  # avoid div-by-zero

def normalize(split_df):
    cont = (split_df[CONTINUOUS] - means) / stds
    # Re-index one-hot columns to match training set (handles missing cam/ccd combos)
    cam_oh  = pd.get_dummies(split_df["cam"].astype(int), prefix="cam").reindex(
                  columns=cam_dummies.columns, fill_value=0)
    ccd_oh  = pd.get_dummies(split_df["ccd"].astype(int), prefix="ccd").reindex(
                  columns=ccd_dummies.columns, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1)

X_train = normalize(train_df).values.astype(np.float32)
X_val   = normalize(val_df).values.astype(np.float32)
X_test  = normalize(test_df).values.astype(np.float32)

y_train = train_df["flux_offset"].values.astype(np.float32)
y_val   = val_df["flux_offset"].values.astype(np.float32)
y_test  = test_df["flux_offset"].values.astype(np.float32)

print(f"\nConditioning vector dimension: {X_train.shape[1]}")

# ── 5. Model: conditional Gaussian MLP ────────────────────────────────────────
# Takes conditioning vector c, outputs (mu, log_sigma).
# Loss = negative log-likelihood under N(mu, sigma^2).

class ConditionalGaussianFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=4):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.mu_head       = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, c):
        h = self.net(c)
        mu        = self.mu_head(h).squeeze(-1)
        log_sigma = self.log_sigma_head(h).squeeze(-1).clamp(-6, 2)
        return mu, log_sigma

    def log_prob(self, c, y):
        mu, log_sigma = self.forward(c)
        sigma = log_sigma.exp()
        # Log-likelihood of y under N(mu, sigma^2)
        return -0.5 * ((y - mu) / sigma) ** 2 - log_sigma - 0.5 * np.log(2 * np.pi)

    def nll_loss(self, c, y):
        return -self.log_prob(c, y).mean()


# ── 6. Training ───────────────────────────────────────────────────────────────

BATCH_SIZE  = 256
LR          = 1e-3
MAX_EPOCHS  = 200
PATIENCE    = 20      # stop if val loss doesn't improve for this many epochs

device = torch.device("cpu")

model = ConditionalGaussianFlow(input_dim=X_train.shape[1], hidden_dim=32, n_layers=2).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True,
)

best_val_loss = float("inf")
patience_count = 0
best_state = None

print("\nTraining...\n  epoch  train_nll   val_nll")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = model.nll_loss(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = model.nll_loss(
            torch.tensor(X_val).to(device),
            torch.tensor(y_val).to(device),
        ).item()

    train_loss = np.mean(train_losses)
    if epoch % 10 == 0:
        print(f"  {epoch:5d}  {train_loss:.4f}      {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)
print(f"\nBest val NLL: {best_val_loss:.4f}")

# ── 7. Evaluation ─────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    mu_test, log_sigma_test = model(torch.tensor(X_test).to(device))
    mu_test       = mu_test.numpy()
    sigma_test    = log_sigma_test.exp().numpy()

residuals = y_test - mu_test

print(f"\n=== Test Set Evaluation ===")
print(f"  Mean absolute error:  {np.abs(residuals).mean():.4f}")
print(f"  Residual std:         {residuals.std():.4f}")
print(f"  Residual mean:        {residuals.mean():.4f}  (should be ~0)")
print(f"  Mean predicted sigma: {sigma_test.mean():.4f}")
print(f"\n  Baseline (predict global mean): MAE = {np.abs(y_test - y_test.mean()).mean():.4f}")
print(f"  Improvement over baseline:      {(1 - np.abs(residuals).mean() / np.abs(y_test - y_test.mean()).mean())*100:.1f}%")

# Per-camera breakdown
print(f"\n  Per-camera breakdown (test set):")
print(f"  {'Cam':<6} {'n':>5} {'train_n':>8} {'MAE':>8} {'baseline':>10} {'improvement':>12} {'pred_sigma':>11}")
print(f"  {'─'*65}")
train_cam_counts = train_df.groupby('cam').size()
for cam, cam_idx in test_df.groupby('cam').groups.items():
    mask = test_df.index.isin(cam_idx)
    # map test_df rows to positional indices in test arrays
    pos = [i for i, idx in enumerate(test_df.index) if idx in cam_idx]
    y_cam   = y_test[pos]
    mu_cam  = mu_test[pos]
    sig_cam = sigma_test[pos]
    res_cam = y_cam - mu_cam
    mae     = np.abs(res_cam).mean()
    base    = np.abs(y_cam - y_cam.mean()).mean()
    improv  = (1 - mae / base) * 100
    n_train = train_cam_counts.get(cam, 0)
    print(f"  Cam{int(cam):<3} {len(y_cam):>5} {n_train:>8} {mae:>8.4f} {base:>10.4f} {improv:>11.1f}% {sig_cam.mean():>11.4f}")

torch.save({"model_state": best_state, "means": means, "stds": stds,
            "continuous_cols": CONTINUOUS,
            "cam_cols": list(cam_dummies.columns),
            "ccd_cols": list(ccd_dummies.columns)},
           "stitch_v1.pt")
print("\nSaved → stitch_v1.pt")
