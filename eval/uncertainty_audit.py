"""
Uncertainty audit — do STITCH's prediction weights reflect star type?

For a set of known long-period variables (from ASAS-SN / AAVSO catalogs, or
any CSV with tic_id column), checks whether STITCH assigns higher pred_std
(lower weights) to them vs the quiet TARS baseline.

If the shrinkage mechanism is working correctly, variable stars should have
notably higher uncertainty than quiet stars, because their sector-to-sector
flux changes are not predictable from spatial neighbours.

Usage:
  python3 eval/uncertainty_audit.py stitch_nsf_knn.pt training_data_topup_pdc.parquet
  python3 eval/uncertainty_audit.py stitch_nsf_knn.pt training_data_topup_pdc.parquet --variables=my_variables.csv

Output:
  eval/uncertainty_audit.png  — weight distributions: quiet vs variable stars
  prints per-group weight summary
"""

import sys
import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

MODEL    = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_knn.pt")
PARQUET  = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")
VAR_CSV  = next((s.split("=")[1] for s in sys.argv[1:] if s.startswith("--variables=")), None)
print(f"Model: {MODEL}  |  Parquet: {PARQUET}")

# ── Load model ─────────────────────────────────────────────────────────────────
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

# ── Load & clean data ──────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[df["sector_median"] > 0]
df = df[df["n_sectors_total"] >= 4]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Reproduce train/test split to get test stars ───────────────────────────────
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
print(f"Test stars: {test_df['tic_id'].nunique():,}  records: {len(test_df):,}")

# ── Predict on all test records ────────────────────────────────────────────────
def make_context(d):
    cont   = (d[CONTINUOUS].reindex(columns=CONTINUOUS, fill_value=0) - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(
                 columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(
                 columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype("float32")

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
flow = flow.to(device)
test_df = test_df.reset_index(drop=True)
C = torch.tensor(make_context(test_df)).to(device)

with torch.no_grad():
    samples = flow(C).sample((300,)).squeeze(-1)
    mu_z    = samples.mean(0).cpu().numpy()
    sig_z   = samples.std(0).cpu().numpy()

pred_raw = mu_z * y_std + y_mean
pred_std = sig_z * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)

test_df["pred_raw"] = pred_raw
test_df["pred_std"] = pred_std
test_df["weight"]   = weight

# ── Per-star summary ───────────────────────────────────────────────────────────
star_summary = test_df.groupby("tic_id").agg(
    mean_weight   = ("weight",   "mean"),
    mean_pred_std = ("pred_std", "mean"),
    n_sectors     = ("sector",   "count"),
    tmag          = ("tmag",     "first"),
    raw_scatter   = ("sector_median", lambda x: (x / x.mean()).std()),
).reset_index()

# ── Load known variables if provided ──────────────────────────────────────────
if VAR_CSV:
    var_df = pd.read_csv(VAR_CSV)
    var_tics = set(var_df["tic_id"].astype(int).values)
    star_summary["is_variable"] = star_summary["tic_id"].isin(var_tics)
    print(f"Known variables in test set: {star_summary['is_variable'].sum()}")
else:
    # Proxy: stars with high raw scatter are likely variable
    # (top 10% of scatter among test stars with >= 5 sectors)
    eligible = star_summary[star_summary["n_sectors"] >= 5]
    scatter_thresh = eligible["raw_scatter"].quantile(0.90)
    star_summary["is_variable"] = (
        (star_summary["n_sectors"] >= 5) &
        (star_summary["raw_scatter"] >= scatter_thresh)
    )
    print(f"Using high-scatter proxy for variables (top 10%, thresh={scatter_thresh:.4f})")
    print(f"  'Variable' stars (proxy): {star_summary['is_variable'].sum()}")

quiet = star_summary[~star_summary["is_variable"]]
vari  = star_summary[star_summary["is_variable"]]

print(f"\n{'─'*55}")
print(f"{'Group':<20}  {'N':>5}  {'Mean weight':>12}  {'Mean pred_std':>14}")
print(f"{'─'*55}")
print(f"{'Quiet (baseline)':<20}  {len(quiet):>5}  {quiet['mean_weight'].mean():>12.4f}  {quiet['mean_pred_std'].mean():>14.5f}")
print(f"{'Variable / high-σ':<20}  {len(vari):>5}  {vari['mean_weight'].mean():>12.4f}  {vari['mean_pred_std'].mean():>14.5f}")
print(f"{'─'*55}")
print(f"\nWeight ratio (quiet / variable): {quiet['mean_weight'].mean() / vari['mean_weight'].mean():.2f}x")
print("  > 1 means variables correctly get less correction (good)")
print("  ≈ 1 means STITCH is not distinguishing variable from quiet (concerning)")

# ── Figure ─────────────────────────────────────────────────────────────────────
BG, SURF, TEXT, MUT, DIM = "#0b0f1a", "#0f1826", "#c4d4ee", "#7a9cc4", "#1c2d45"
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.patch.set_facecolor(BG)

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF)
    ax.set_xlabel(xlabel, color=MUT, fontsize=10)
    ax.set_ylabel(ylabel, color=MUT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11, pad=8)
    ax.tick_params(colors=MUT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True, color=DIM, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

# Panel 1: weight distributions
ax = axes[0]
bins = np.linspace(0, 1, 50)
ax.hist(quiet["mean_weight"], bins=bins, alpha=0.7, color="#4f8ef7",
        label=f"Quiet  (n={len(quiet):,})", density=True)
ax.hist(vari["mean_weight"],  bins=bins, alpha=0.7, color="#ef4444",
        label=f"Variable / high-σ  (n={len(vari):,})", density=True)
ax.axvline(quiet["mean_weight"].median(), color="#4f8ef7", lw=1.5, ls="--")
ax.axvline(vari["mean_weight"].median(),  color="#ef4444", lw=1.5, ls="--")
ax.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
style(ax, "Mean uncertainty weight per star", "Density",
      "Weight distribution: quiet vs variable\n(lower weight = STITCH corrects less)")

# Panel 2: scatter before vs weight
ax = axes[1]
sc = ax.scatter(star_summary[~star_summary["is_variable"]]["raw_scatter"] * 100,
                star_summary[~star_summary["is_variable"]]["mean_weight"],
                c="#4f8ef7", s=3, alpha=0.3, label="Quiet", rasterized=True)
sc2 = ax.scatter(star_summary[star_summary["is_variable"]]["raw_scatter"] * 100,
                 star_summary[star_summary["is_variable"]]["mean_weight"],
                 c="#ef4444", s=8, alpha=0.6, label="Variable / high-σ")
ax.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
style(ax, "Raw scatter before STITCH (%)", "Mean weight (0=no correction, 1=full)",
      "Weight vs raw scatter per star\n(should trend downward for high-scatter stars)")

plt.tight_layout()
out = "eval/uncertainty_audit.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
