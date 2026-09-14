"""
Compare stitch_nsf.pt (LOO labels) vs stitch_nsf_gaia.pt (Gaia RP labels)
on the same held-out test set, scored against both label types.

Usage:
  python3 eval/compare_models.py [parquet]
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import zuko
from sklearn.model_selection import train_test_split

PARQUET   = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
                 "training_data_topup_pdc.parquet")
MODELS    = {
    "LOO":              "stitch_nsf.pt",
    "Gaia":             "stitch_nsf_gaia.pt",
    "LOO+GaiaFeat":     "stitch_nsf_gaia_feat.pt",
    "LOO+GaiaFeat+Nbr": "stitch_nsf_nbr.pt",
    "LOO+GaiaFeat+KNN": "stitch_nsf_knn.pt",
}
MIN_SECTORS = 4
TMAG_MAX    = 13.0
N_SAMPLES   = 200   # samples for mean estimate

# ── Reproduce the exact train/val/test split ──────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "flux_offset_loo"])
df = df[(df["flux_offset"]     > 0.85) & (df["flux_offset"]     < 1.15)]
df = df[(df["flux_offset_loo"] > 0.85) & (df["flux_offset_loo"] < 1.15)]
df = df[df["n_sectors_total"] >= MIN_SECTORS]
df = df[df["tmag"] <= TMAG_MAX]

CONTINUOUS = ["col", "row", "delta_sub_col", "delta_sub_row",
              "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
              "pdc_noi", "pr_wght2"]

# Compute Gaia-derived features if available
if "gaiarp" in df.columns and "flux_offset" in df.columns:
    df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")

# Sector-CCD leave-one-out mean (needed by NBR/KNN models)
df = df.reset_index(drop=True)
_grp = df.groupby(["sector", "cam", "ccd"])["flux_offset_loo"]
df["sector_ccd_mean_loo"] = (_grp.transform("sum") - df["flux_offset_loo"]) / \
                             (_grp.transform("count") - 1).clip(lower=1)

# Spatial KNN leave-one-out mean (needed by KNN model)
from scipy.spatial import cKDTree as _cKDTree
_K = 20
_knn_vals = np.full(len(df), np.nan, dtype=np.float32)
for (_sec, _cam, _ccd), _g in df.groupby(["sector", "cam", "ccd"]):
    _n = len(_g)
    if _n < 2: continue
    _coords = _g[["col", "row"]].values.astype(np.float32)
    _vals   = _g["flux_offset_loo"].values
    _k      = min(_K, _n - 1)
    _, _nn  = _cKDTree(_coords).query(_coords, k=_k + 1)
    _knn_vals[_g.index.values] = _vals[_nn[:, 1:]].mean(axis=1)
df["spatial_knn_mean_loo"] = _knn_vals
df["spatial_knn_mean_loo"] = df["spatial_knn_mean_loo"].fillna(df["sector_ccd_mean_loo"])
print(f"KNN feature: r={df['spatial_knn_mean_loo'].corr(df['flux_offset_loo']):.3f}")

for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

cam_dummies = pd.get_dummies(df["cam"].astype(int), prefix="cam")
ccd_dummies = pd.get_dummies(df["ccd"].astype(int), prefix="ccd")

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

test_df = df[df["tic_id"].isin(test_tics)].copy()
train_df = df[df["tic_id"].isin(train_tics)].copy()
print(f"Test set: {len(test_df):,} rows, {test_df['tic_id'].nunique():,} stars")

# ── Context builder (uses each model's own saved feature list and stats) ──────
def make_context(split_df, means, stds, continuous_cols, cam_cols, ccd_cols):
    cont   = (split_df[continuous_cols] - means) / stds
    cam_oh = pd.get_dummies(split_df["cam"].astype(int), prefix="cam").reindex(
                 columns=cam_cols, fill_value=0)
    ccd_oh = pd.get_dummies(split_df["ccd"].astype(int), prefix="ccd").reindex(
                 columns=ccd_cols, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

# ── Predict with one model ─────────────────────────────────────────────────────
device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

def predict(ckpt_path, test_df):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["flow_config"]
    flow = zuko.flows.NSF(**cfg)
    flow.load_state_dict(ck["model_state"])
    flow = flow.to(device)
    flow.eval()

    C = make_context(test_df, ck["means"], ck["stds"],
                     ck["continuous_cols"], ck["cam_cols"], ck["ccd_cols"])
    C_t = torch.tensor(C).to(device)
    with torch.no_grad():
        samples = flow(C_t).sample((N_SAMPLES,)).squeeze(-1)
        mu_norm = samples.mean(0).cpu().numpy()

    return mu_norm * ck["y_std"] + ck["y_mean"]   # back to flux_offset units

# ── Run both models ───────────────────────────────────────────────────────────
import os
predictions = {}
for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"Skipping {name} ({path} not found)")
        continue
    print(f"Running {name} model ({path}) …")
    predictions[name] = predict(path, test_df)

# ── Score against both label types ───────────────────────────────────────────
labels = {
    "Gaia": test_df["flux_offset"].values,
    "LOO":  test_df["flux_offset_loo"].values,
}

print()
print("=" * 62)
print(f"{'':20s} {'vs Gaia label':>18}   {'vs LOO label':>16}")
print(f"{'Model':20s} {'MAE':>8} {'improv%':>8}   {'MAE':>8} {'improv%':>8}")
print("=" * 62)

for model_name, preds in predictions.items():
    row = f"{model_name:<20}"
    for label_name, y_true in labels.items():
        baseline = np.abs(y_true - y_true.mean()).mean()
        mae      = np.abs(y_true - preds).mean()
        improv   = (1 - mae / baseline) * 100
        row += f"  {mae:.4f}  {improv:>6.1f}%"
    print(row)

# Baseline row
row = f"{'Baseline (mean)':20}"
for label_name, y_true in labels.items():
    baseline = np.abs(y_true - y_true.mean()).mean()
    row += f"  {baseline:.4f}  {'0.0%':>7}"
print("-" * 62)
print(row)
print("=" * 62)

# ── Per-camera breakdown ──────────────────────────────────────────────────────
model_names = list(predictions.keys())
col_w = 11

for label_name, y_true_all in labels.items():
    print()
    header = f"{'Cam':<6} {'n':>6}" + "".join(f"  {n:>{col_w}}" for n in model_names)
    print(f"Per-camera MAE (vs {label_name} label):")
    print(header)
    print("-" * len(header))
    for cam in sorted(test_df["cam"].dropna().unique()):
        mask = test_df["cam"].values == cam
        y    = y_true_all[mask]
        n    = mask.sum()
        row  = f"Cam{int(cam):<3} {n:>6}"
        maes = [np.abs(y - predictions[m][mask]).mean() for m in model_names]
        best = min(maes)
        for mae in maes:
            flag = " *" if mae == best else "  "
            row += f"  {mae:>{col_w}.4f}{flag}"[: col_w + 2]
            row += f"  {mae:>.4f}{'*' if mae == best else ' '}"
        # redo cleanly
        row = f"Cam{int(cam):<3} {n:>6}"
        for i, mae in enumerate(maes):
            marker = "*" if mae == best else " "
            row += f"  {mae:.4f}{marker}"
        print(row)
    print()

# ── Within-star scatter reduction ─────────────────────────────────────────────
# For each star: std of (corrected flux) across sectors.
# corrected = sector_median / prediction  → should be closer to constant if good.
print()
print("Within-star scatter (std of sector_median/prediction across sectors):")
print(f"  Lower = more consistent across sectors (the actual goal)")
print()

test_df = test_df.copy()
for model_name, preds in predictions.items():
    test_df[f"pred_{model_name}"]      = preds
    test_df[f"corrected_{model_name}"] = test_df["sector_median"] / preds
test_df["uncorrected"] = test_df["sector_median"]

# Only stars with >= 4 sectors in the test set
star_counts = test_df.groupby("tic_id")["sector"].count()
multi = star_counts[star_counts >= 4].index
sub = test_df[test_df["tic_id"].isin(multi)]

def norm_std(col):
    return sub.groupby("tic_id")[col].apply(lambda x: x.std() / x.mean())

cv_raw = norm_std("uncorrected")
std_raw = cv_raw.median()
print(f"  {'Uncorrected':24s}  median CV = {std_raw:.4f}")
for model_name in model_names:
    col = f"corrected_{model_name}"
    cv_model = norm_std(col)
    std = cv_model.median()
    pct_improved = (cv_model < cv_raw).mean() * 100
    print(f"  {model_name:<24s}  median CV = {std:.4f}  ({(1-std/std_raw)*100:.1f}% reduction)  "
          f"{pct_improved:.1f}% of stars improved")
print(f"  (n={len(multi):,} stars with ≥4 sectors in test set)")

# Per-decile breakdown for best model
print()
best_model = model_names[-1]
cv_best = norm_std(f"corrected_{best_model}")
decile = pd.qcut(cv_raw, 10, labels=False)
print(f"Per-decile breakdown (worst → best raw CV), model={best_model}:")
for d in range(10):
    mask = decile == d
    r = cv_raw[mask].median(); m = cv_best[mask].median()
    pct = (cv_best[mask] < cv_raw[mask]).mean() * 100
    print(f"  decile {d+1:2d} (raw CV {cv_raw[mask].min():.4f}–{cv_raw[mask].max():.4f}): "
          f"{r:.4f}→{m:.4f}  {(1-m/r)*100:.1f}% better  ({pct:.0f}% stars improved)")
