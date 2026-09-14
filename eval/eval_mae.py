"""
Quick MAE evaluation for any STITCH model on a filtered test set.

Usage:
  python3 eval/eval_mae.py stitch_nsf_v3.pt --tmag=12
  python3 eval/eval_mae.py stitch_nsf_tmag12.pt --tmag=12
  python3 eval/eval_mae.py stitch_nsf_v3.pt          # full dataset (tmag<=13)
"""

import sys, numpy as np, pandas as pd, torch, zuko
from sklearn.model_selection import train_test_split

MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")), "stitch_nsf_v3.pt")
PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup.parquet")
_tmag   = next((s for s in sys.argv[1:] if s.startswith("--tmag=")), None)
TMAG_MAX = float(_tmag.split("=")[1]) if _tmag else 13.0

print(f"Model: {MODEL}  |  Parquet: {PARQUET}  |  Tmag <= {TMAG_MAX}")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(
    features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"],
    bins=cfg["bins"],
)
flow.load_state_dict(ckpt["model_state"])
flow.eval()

means      = ckpt["means"]
stds       = ckpt["stds"]
y_mean     = ckpt["y_mean"]
y_std      = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS   = ckpt["cam_cols"]
CCD_COLS   = ckpt["ccd_cols"]

# ── Load & clean data — same filters as training script ───────────────────────
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[df["n_sectors_total"] >= 4]
df = df[df["tmag"] <= TMAG_MAX]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
print(f"Records after filter: {len(df):,}  stars: {df['tic_id'].nunique():,}")

# ── Reproduce SAME stratified split ───────────────────────────────────────────
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))
tr_tics, tmp = train_test_split(
    star_cam["tic_id"], test_size=0.2,
    stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_, te_tics = train_test_split(
    tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)

test_df = df[df["tic_id"].isin(te_tics)].copy()
print(f"Test set: {len(test_df):,} records  {test_df['tic_id'].nunique():,} stars")

# ── Context matrix ────────────────────────────────────────────────────────────
def make_context(split_df):
    cont   = (split_df[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(split_df["cam"].astype(int), prefix="cam").reindex(
                 columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(split_df["ccd"].astype(int), prefix="ccd").reindex(
                 columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype("float32")

C_test = make_context(test_df)
y_test = test_df["flux_offset"].values.astype("float32")

# ── Inference ─────────────────────────────────────────────────────────────────
device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
flow = flow.to(device)
C_t  = torch.tensor(C_test).to(device)

with torch.no_grad():
    samples  = flow(C_t).sample((200,)).squeeze(-1)   # (200, N)
    mu_norm  = samples.mean(0).cpu().numpy()

mu_fo = mu_norm * y_std + y_mean   # back to flux_offset units

mae      = np.abs(y_test - mu_fo).mean()
baseline = np.abs(y_test - y_test.mean()).mean()
improv   = (1 - mae / baseline) * 100

print(f"\n── Results ──────────────────────────────────────────")
print(f"  Model MAE:    {mae:.4f}")
print(f"  Baseline MAE: {baseline:.4f}  (predict global mean)")
print(f"  Improvement:  {improv:.1f}%")

print(f"\n── Per-camera breakdown ─────────────────────────────")
print(f"  {'Cam':<5} {'n':>6} {'MAE':>8} {'baseline':>10} {'improv':>9}")
print(f"  {'─'*45}")
for cam, g in test_df.groupby("cam"):
    pos    = [i for i, idx in enumerate(test_df.index) if idx in g.index]
    y_c    = y_test[pos]
    mu_c   = mu_fo[pos]
    m      = np.abs(y_c - mu_c).mean()
    b      = np.abs(y_c - y_c.mean()).mean()
    print(f"  Cam{int(cam):<2} {len(y_c):>6,} {m:>8.4f} {b:>10.4f} {(1-m/b)*100:>8.1f}%")
