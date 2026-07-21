"""
STITCH — one NSF model per cam/CCD (16 models total).

Each model sees only its own cam/CCD subset, so cam/ccd one-hot features
are dropped (they're constant). This lets each model specialize to the
specific PRF pattern of its detector cell.

Architecture is adaptive: smaller models for data-sparse cells (Cam1)
to avoid overfitting, larger for data-rich cells (Cam4).

Output: stitch_nsf_per_ccd.pt — dict keyed by "camX_ccdY", each entry
containing model_state, means, stds, y_mean, y_std, continuous_cols,
flow_config. Compatible with updated infer_star.py and evaluate_stitching.py.
"""

import numpy as np
import pandas as pd
import torch
import zuko
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ── 1. Load and clean data ────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
print(f"Loaded {len(df):,} records from {df['tic_id'].nunique():,} stars")

df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
df["n_sectors_total"] = df.groupby("tic_id")["sector"].transform("count")
df = df[df["n_sectors_total"] >= 5]
print(f"After cleaning (n_sectors >= 5): {len(df):,} records\n")

CONTINUOUS = ["col", "row", "delta_sub_col", "delta_sub_row",
              "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
              "log_sector_median", "n_sectors_total"]

for col in CONTINUOUS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

BATCH_SIZE = 256
LR         = 3e-4
MAX_EPOCHS = 300
PATIENCE   = 20

def arch_for(n_train):
    if n_train < 800:
        return dict(transforms=4, hidden_features=[64, 64], bins=6)
    elif n_train < 3000:
        return dict(transforms=6, hidden_features=[96, 96], bins=8)
    else:
        return dict(transforms=8, hidden_features=[128, 128], bins=8)


def train_one(cam, ccd, sub_df):
    label = f"Cam{cam}/CCD{ccd}"

    # Star-level train/val/test split
    star_ids = sub_df["tic_id"].unique()
    if len(star_ids) < 10:
        print(f"  {label}: too few stars ({len(star_ids)}) — skipping")
        return None

    train_tics, temp = train_test_split(star_ids, test_size=0.2, random_state=42)
    val_tics,  test_tics = train_test_split(temp, test_size=0.5, random_state=42)

    train_df = sub_df[sub_df["tic_id"].isin(train_tics)]
    val_df   = sub_df[sub_df["tic_id"].isin(val_tics)]
    test_df  = sub_df[sub_df["tic_id"].isin(test_tics)]

    n_train = len(train_df)
    arch    = arch_for(n_train)
    print(f"  {label}: {len(star_ids):,} stars | train={n_train} "
          f"| arch={arch['transforms']}T/{arch['hidden_features'][0]}H")

    # Normalisation fit on train only
    means  = train_df[CONTINUOUS].mean()
    stds   = train_df[CONTINUOUS].std().replace(0, 1)
    y_mean = float(train_df["flux_offset"].mean())
    y_std  = float(train_df["flux_offset"].std())

    def make_ctx(split_df):
        return ((split_df[CONTINUOUS] - means) / stds).values.astype(np.float32)

    def make_tgt(split_df):
        return ((split_df["flux_offset"].values - y_mean) / y_std).astype(np.float32)

    C_train = make_ctx(train_df);  y_train = make_tgt(train_df)
    C_val   = make_ctx(val_df);    y_val   = make_tgt(val_df)
    C_test  = make_ctx(test_df);   y_test  = make_tgt(test_df)

    flow = zuko.flows.NSF(
        features=1, context=len(CONTINUOUS),
        **arch,
    )
    opt   = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    loader = DataLoader(
        TensorDataset(torch.tensor(C_train), torch.tensor(y_train).unsqueeze(-1)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    C_val_t  = torch.tensor(C_val)
    y_val_t  = torch.tensor(y_val).unsqueeze(-1)
    C_test_t = torch.tensor(C_test)

    best_val, best_state, patience_ct = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        flow.train()
        for cb, yb in loader:
            nll = -flow(cb).log_prob(yb).mean()
            opt.zero_grad(); nll.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        flow.eval()
        with torch.no_grad():
            val_nll = -flow(C_val_t).log_prob(y_val_t).mean().item()
        sched.step(val_nll)
        if val_nll < best_val:
            best_val, best_state, patience_ct = val_nll, {k: v.clone() for k, v in flow.state_dict().items()}, 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                break

    flow.load_state_dict(best_state)
    flow.eval()
    with torch.no_grad():
        samples   = flow(C_test_t).sample((300,)).squeeze(-1)
        mu_test   = samples.mean(0).numpy()

    y_fo  = y_test  * y_std + y_mean
    mu_fo = mu_test * y_std + y_mean
    mae      = np.abs(y_fo - mu_fo).mean()
    baseline = np.abs(y_fo - y_fo.mean()).mean()
    improv   = (1 - mae / baseline) * 100

    print(f"    val_NLL={best_val:.4f}  MAE={mae:.4f}  "
          f"baseline={baseline:.4f}  improvement={improv:.1f}%  (epoch {epoch})")

    return {
        "model_state":     best_state,
        "means":           means,
        "stds":            stds,
        "y_mean":          y_mean,
        "y_std":           y_std,
        "continuous_cols": CONTINUOUS,
        "flow_config": {
            "features":        1,
            "context":         len(CONTINUOUS),
            "transforms":      arch["transforms"],
            "hidden_features": arch["hidden_features"],
            "bins":            arch["bins"],
        },
        "n_train_stars": int(len(train_tics)),
        "cam": int(cam),
        "ccd": int(ccd),
    }


# ── 2. Train all 16 models ────────────────────────────────────────────────────

checkpoint = {}
print("Training per-cam/CCD models...\n")
for (cam, ccd), g in df.groupby(["cam", "ccd"]):
    key = f"cam{int(cam)}_ccd{int(ccd)}"
    result = train_one(int(cam), int(ccd), g)
    if result is not None:
        checkpoint[key] = result
        # Save incrementally so a crash doesn't lose everything
        torch.save(checkpoint, "stitch_nsf_per_ccd.pt")

print(f"\nAll models saved → stitch_nsf_per_ccd.pt  ({len(checkpoint)} cam/CCD models)")

# ── 3. Summary ────────────────────────────────────────────────────────────────

print("\n=== Summary ===")
print(f"{'Cell':<14} {'Stars':>7} {'Arch':>12} {'val_NLL':>9} {'Improve':>9}")
print(f"  {'─'*55}")
for key, v in sorted(checkpoint.items()):
    cfg = v["flow_config"]
    arch_str = f"{cfg['transforms']}T/{cfg['hidden_features'][0]}H"
    # val NLL not stored — print what we have
    print(f"  {key:<12} {v['n_train_stars']:>7,}   {arch_str:>10}")
