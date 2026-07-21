"""
Learning curve: scatter reduction vs training set size.
Trains the NSF at 10/25/50/75/100% of available data (by star count),
evaluates scatter reduction on a fixed held-out set, plots the curve.
"""

import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────────────────────

FRACTIONS   = [0.10, 0.25, 0.50, 0.75, 1.00]
N_REPEATS   = 3       # repeat each fraction to estimate variance
MIN_SECTORS = 5       # eval: stars with ≥ this many sectors
TRANSFORMS  = 8
HIDDEN      = [128, 128]
BINS        = 8
MAX_EPOCHS  = 200
PATIENCE    = 20
BATCH_SIZE  = 512
LR          = 3e-4

CONTINUOUS = ["col", "row", "delta_sub_col", "delta_sub_row",
              "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms"]

# ── Load & clean data ─────────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["col", "row", "flux_offset"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
for col in CONTINUOUS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

cam_dummies = pd.get_dummies(df["cam"].astype(int), prefix="cam")
ccd_dummies = pd.get_dummies(df["ccd"].astype(int), prefix="ccd")

# Fixed eval set: stars with ≥ MIN_SECTORS sectors, held out from all runs
sec_counts = df.groupby("tic_id")["sector"].count()
eval_tics  = sec_counts[sec_counts >= MIN_SECTORS].index
# Use 20% of those as fixed eval
eval_tics, _ = train_test_split(eval_tics, test_size=0.80, random_state=99)
eval_df  = df[df["tic_id"].isin(eval_tics)].copy()
train_pool = df[~df["tic_id"].isin(eval_tics)]
all_train_tics = train_pool["tic_id"].unique()

print(f"Total records: {len(df):,} | Eval stars: {len(eval_tics)} | Train pool stars: {len(all_train_tics)}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_context(split_df, means, stds):
    cont   = (split_df[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(split_df["cam"].astype(int), prefix="cam").reindex(
                 columns=cam_dummies.columns, fill_value=0)
    ccd_oh = pd.get_dummies(split_df["ccd"].astype(int), prefix="ccd").reindex(
                 columns=ccd_dummies.columns, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)


def scatter_reduction(eval_df, flow, means, stds, y_mean, y_std):
    C = make_context(eval_df, means, stds)
    with torch.no_grad():
        samples = flow(torch.tensor(C)).sample((200,)).squeeze(-1)
        pred_z  = samples.mean(0).numpy()
    pred_offset = pred_z * y_std + y_mean
    eval_df = eval_df.copy()
    eval_df["pred_offset"] = pred_offset

    imprs = []
    for tic, g in eval_df.groupby("tic_id"):
        g = g.sort_values("sector")
        raw  = g["sector_median"].values
        pred = g["pred_offset"].values
        raw_norm    = raw / raw.mean()
        stitch_norm = (raw / pred); stitch_norm /= stitch_norm.mean()
        before = raw_norm.std()
        after  = stitch_norm.std()
        if before > 0:
            imprs.append((before - after) / before)
    return np.median(imprs) * 100


def train_nsf(train_df):
    means  = train_df[CONTINUOUS].mean()
    stds   = train_df[CONTINUOUS].std().replace(0, 1)
    y_mean = float(train_df["flux_offset"].mean())
    y_std  = float(train_df["flux_offset"].std())

    # val split (10% of train stars)
    tics = train_df["tic_id"].unique()
    val_tics = np.random.choice(tics, size=max(1, len(tics)//10), replace=False)
    val_df   = train_df[train_df["tic_id"].isin(val_tics)]
    tr_df    = train_df[~train_df["tic_id"].isin(val_tics)]

    C_tr = make_context(tr_df, means, stds)
    y_tr = ((tr_df["flux_offset"].values - y_mean) / y_std).astype(np.float32)
    C_va = make_context(val_df, means, stds)
    y_va = ((val_df["flux_offset"].values - y_mean) / y_std).astype(np.float32)

    context_dim = C_tr.shape[1]
    flow = zuko.flows.NSF(features=1, context=context_dim,
                          transforms=TRANSFORMS, hidden_features=HIDDEN, bins=BINS)
    opt  = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    loader = DataLoader(
        TensorDataset(torch.tensor(C_tr), torch.tensor(y_tr).unsqueeze(-1)),
        batch_size=BATCH_SIZE, shuffle=True)
    C_va_t = torch.tensor(C_va)
    y_va_t = torch.tensor(y_va).unsqueeze(-1)

    best_val, best_state, patience_count = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        flow.train()
        for cb, yb in loader:
            nll = -flow(cb).log_prob(yb).mean()
            opt.zero_grad(); nll.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        flow.eval()
        with torch.no_grad():
            val_nll = -flow(C_va_t).log_prob(y_va_t).mean().item()
        sched.step(val_nll)
        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break
    flow.load_state_dict(best_state)
    flow.eval()
    return flow, means, stds, y_mean, y_std


# ── Learning curve ────────────────────────────────────────────────────────────

results = {f: [] for f in FRACTIONS}

for frac in FRACTIONS:
    n_stars = max(10, int(len(all_train_tics) * frac))
    print(f"\nFraction {frac:.0%}  ({n_stars} stars)")
    for rep in range(N_REPEATS):
        chosen = np.random.choice(all_train_tics, size=n_stars, replace=False)
        train_df = train_pool[train_pool["tic_id"].isin(chosen)]
        flow, means, stds, y_mean, y_std = train_nsf(train_df)
        sr = scatter_reduction(eval_df, flow, means, stds, y_mean, y_std)
        results[frac].append(sr)
        print(f"  rep {rep+1}: {sr:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────

frac_arr  = np.array(FRACTIONS)
n_stars_arr = np.array([int(len(all_train_tics) * f) for f in FRACTIONS])
medians   = np.array([np.median(results[f]) for f in FRACTIONS])
lo        = np.array([np.min(results[f])    for f in FRACTIONS])
hi        = np.array([np.max(results[f])    for f in FRACTIONS])

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(n_stars_arr, lo, hi, alpha=0.2, color="#2166ac")
ax.plot(n_stars_arr, medians, "o-", color="#2166ac", lw=2, ms=7)
for f, n, m in zip(FRACTIONS, n_stars_arr, medians):
    ax.annotate(f"{m:.1f}%", (n, m), textcoords="offset points",
                xytext=(4, 6), fontsize=9)

ax.set_xlabel("Training stars", fontsize=11)
ax.set_ylabel("Median scatter reduction (%)", fontsize=11)
ax.set_title("STITCH NSF — Learning Curve\n(fixed held-out eval set, shaded = min/max across repeats)",
             fontsize=11)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150, bbox_inches="tight")
print("\nSaved → learning_curve.png")
for f, n in zip(FRACTIONS, n_stars_arr):
    print(f"  {f:.0%} ({n:4d} stars): {np.median(results[f]):.1f}% "
          f"[{np.min(results[f]):.1f}–{np.max(results[f]):.1f}]")
