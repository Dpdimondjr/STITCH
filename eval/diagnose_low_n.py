"""
Diagnostic: does training on n=2 stars hurt STITCH on better-observed stars?

Checks:
 1. Per-star scatter improvement broken down by n_sectors of the TEST star
 2. Where the model *hurts* (scatter increases) — do those cluster in cam/ccd?
 3. Are the harmed test stars near n=2 training stars in (cam, ccd) space?
"""

import numpy as np
import pandas as pd
import torch, zuko
from sklearn.model_selection import train_test_split

# ── Load model ─────────────────────────────────────────────────────────────────
import sys as _sys
_model_pt = next((s for s in _sys.argv[1:] if s.endswith(".pt")), "stitch_nsf_v3.pt")
_parquet  = next((s for s in _sys.argv[1:] if s.endswith(".parquet")), "training_data_topup.parquet")
print(f"Model: {_model_pt}  Parquet: {_parquet}")
ckpt = torch.load(_model_pt, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(
    features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"],
    bins=cfg["bins"],
)
flow.load_state_dict(ckpt["model_state"])
flow.eval()
means      = ckpt["means"];  stds  = ckpt["stds"]
y_mean     = ckpt["y_mean"]; y_std = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS   = ckpt["cam_cols"]; CCD_COLS = ckpt["ccd_cols"]

# ── Load data + split ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(_parquet)
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0]).reset_index()
              .rename(columns={"cam": "dominant_cam"}))
tr, tmp = train_test_split(star_cam["tic_id"], test_size=0.2,
                           stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_, te   = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)

train_tics = set(tr); test_tics = set(te)

# n_sectors per training star
train_n = (df[df["tic_id"].isin(train_tics)]
           .groupby("tic_id")["sector"].nunique().reset_index()
           .rename(columns={"sector": "n"}))

test_df = df[df["tic_id"].isin(test_tics)].copy()
print(f"  Test records: {len(test_df):,}  TICs: {test_df['tic_id'].nunique():,}")

# ── Run inference on ALL test stars (no MIN_SECTORS filter) ───────────────────
print("Running inference on all test stars...")
def make_context(d):
    cont   = (d[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(
                 columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(
                 columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

C = make_context(test_df)
with torch.no_grad():
    samples     = flow(torch.tensor(C)).sample((500,)).squeeze(-1)
    pred_z      = samples.mean(0).numpy()
    pred_z_std  = samples.std(0).numpy()

pred_raw = pred_z * y_std + y_mean
pred_std = pred_z_std * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)
test_df  = test_df.copy()
test_df["pred_offset"] = 1.0 * (1 - weight) + pred_raw * weight

# ── Per-star scatter ───────────────────────────────────────────────────────────
print("Computing per-star scatter...")
rows = []
for tic, g in test_df.groupby("tic_id"):
    g = g.sort_values("sector")
    raw  = g["sector_median"].values
    pred = g["pred_offset"].values
    ref  = raw.mean()
    sc_before = (raw / ref).std()
    sc_after  = (raw / pred / (raw / pred).mean()).std()
    rows.append({
        "tic_id":     tic,
        "n":          len(g),
        "cam":        int(g["cam"].mode()[0]),
        "ccd":        int(g["ccd"].mode()[0]),
        "sc_before":  sc_before * 100,
        "sc_after":   sc_after  * 100,
        "improv":     (sc_before - sc_after) / sc_before * 100,
        "harmed":     sc_after > sc_before,
    })
res = pd.DataFrame(rows)

# ── 1. Improvement by n_sectors bin ───────────────────────────────────────────
print("\n=== Scatter improvement by n_sectors of TEST star ===")
bins = [(1,2),(3,5),(6,8),(9,12),(13,20),(21,35),(36,60)]
print(f"{'n_sectors':>12} {'stars':>7} {'median sc_before%':>18} "
      f"{'median improv%':>15} {'% harmed':>10}")
for lo, hi in bins:
    sub = res[(res["n"] >= lo) & (res["n"] <= hi)]
    if len(sub) == 0: continue
    print(f"  {lo if lo==hi else f'{lo}–{hi}':>10} "
          f"{len(sub):>7,} "
          f"{sub['sc_before'].median():>18.2f} "
          f"{sub['improv'].median():>15.2f} "
          f"{sub['harmed'].mean()*100:>9.1f}%")

# ── 2. Where does the model hurt? (cam/ccd breakdown for harmed stars) ─────────
print("\n=== Stars where STITCH worsens scatter (harmed), by cam/ccd (n≥5 only) ===")
well_obs = res[res["n"] >= 5]
print(f"n≥5 test stars: {len(well_obs):,}  "
      f"harmed: {well_obs['harmed'].sum():,} ({well_obs['harmed'].mean()*100:.1f}%)")
print()
harmed = well_obs[well_obs["harmed"]]
pivot = (well_obs.groupby(["cam","ccd"])
         .agg(total=("tic_id","count"), harmed=("harmed","sum"))
         .assign(harm_pct=lambda d: d["harmed"]/d["total"]*100)
         .reset_index().sort_values("harm_pct", ascending=False))
print(pivot.to_string(index=False))

# ── 3. Do harmed n≥5 test stars share cam/ccd with n=2 training stars? ────────
print("\n=== n=2 training star density vs harm rate per cam/ccd ===")
n2_train = df[df["tic_id"].isin(
    set(train_n[train_n["n"] <= 2]["tic_id"])
)]
n2_density = (n2_train.groupby(["cam","ccd"])
              .agg(n2_records=("tic_id","count"))
              .reset_index())
n_all_train = (df[df["tic_id"].isin(train_tics)]
               .groupby(["cam","ccd"])
               .agg(total_records=("tic_id","count"))
               .reset_index())
density = n2_density.merge(n_all_train, on=["cam","ccd"])
density["n2_frac"] = density["n2_records"] / density["total_records"]
combined = pivot.merge(density, on=["cam","ccd"], how="left")
print(combined[["cam","ccd","total","harmed","harm_pct","n2_frac"]].to_string(index=False))

# ── 4. Worst harmed stars (n≥5) ────────────────────────────────────────────────
print("\n=== 10 most harmed test stars (n≥5) ===")
worst = well_obs.nsmallest(10, "improv")[
    ["tic_id","n","cam","ccd","sc_before","sc_after","improv"]
].round(3)
print(worst.to_string(index=False))
