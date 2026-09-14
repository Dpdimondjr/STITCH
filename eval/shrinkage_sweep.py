"""
Sweep the shrinkage coefficient k in weight = 1/(1 + k*pred_std).

Inference runs once; only the weighting changes per k value.
Reports per-k: median improvement, % hurt, % improved, and improvement
broken out by quiet vs noisy stars.

Usage:
  python3 eval/shrinkage_sweep.py [model.pt] [parquet]
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),
               "stitch_nsf_pdc_full.pt")
K_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
print(f"Model:   {MODEL}")
print(f"Parquet: {PARQUET}")

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
means, stds   = ckpt["means"], ckpt["stds"]
y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
CONTINUOUS, CAM_COLS, CCD_COLS = ckpt["continuous_cols"], ckpt["cam_cols"], ckpt["ccd_cols"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading parquet …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
n_sec = df.groupby("tic_id")["sector"].nunique()
df = df[df["tic_id"].isin(n_sec[n_sec >= 4].index)].copy().reset_index(drop=True)

# ── Inference (once) ──────────────────────────────────────────────────────────
def make_context(d):
    cont   = (d[CONTINUOUS] - means) / stds
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
print(f"Running inference on {len(df):,} records ({device}) …")

BATCH = 8192
mu_z_list, sig_z_list = [], []
for i in range(0, len(df), BATCH):
    chunk = df.iloc[i:i+BATCH]
    C = torch.tensor(make_context(chunk)).to(device)
    with torch.no_grad():
        samp = flow(C).sample((200,)).squeeze(-1)
        mu_z_list.append(samp.mean(0).cpu().numpy())
        sig_z_list.append(samp.std(0).cpu().numpy())

pred_raw = np.concatenate(mu_z_list) * y_std + y_mean
pred_std = np.concatenate(sig_z_list) * y_std
df["pred_raw"] = pred_raw
df["pred_std"] = pred_std

# Quiet/noisy split by cdpp1_0
cdpp_33 = df.groupby("tic_id")["cdpp1_0"].median().quantile(0.33)
cdpp_67 = df.groupby("tic_id")["cdpp1_0"].median().quantile(0.67)
quiet_tids = df.groupby("tic_id")["cdpp1_0"].median()
quiet_tids = quiet_tids[quiet_tids <= cdpp_33].index
noisy_tids = df.groupby("tic_id")["cdpp1_0"].median()
noisy_tids = noisy_tids[noisy_tids >= cdpp_67].index
print(f"  Quiet stars (low cdpp): {len(quiet_tids):,}  "
      f"Noisy stars (high cdpp): {len(noisy_tids):,}")

# ── Scatter eval helper ───────────────────────────────────────────────────────
def sector_std(x): return x.std(ddof=1) if len(x) >= 3 else np.nan

raw_std = df.groupby("tic_id")["flux_offset"].agg(sector_std)

def eval_k(k):
    w = 1.0 / (1.0 + k * df["pred_std"]) if k > 0 else pd.Series(1.0, index=df.index)
    pred  = w * df["pred_raw"] + (1 - w) * 1.0
    corr  = df["flux_offset"] / pred
    cor_std = corr.groupby(df["tic_id"]).agg(sector_std)
    imp = 1.0 - cor_std / raw_std
    imp = imp.dropna()
    return {
        "k":           k,
        "med_imp":     imp.median(),
        "pct_hurt":    (imp < 0).mean() * 100,
        "pct_imp":     (imp > 0).mean() * 100,
        "med_quiet":   imp.reindex(quiet_tids).dropna().median(),
        "med_noisy":   imp.reindex(noisy_tids).dropna().median(),
        "mean_weight": float(w.mean()),
    }

print(f"\nSweeping k = {K_VALUES} …")
print(f"{'k':>6}  {'med_imp':>8}  {'pct_hurt':>9}  {'quiet':>7}  {'noisy':>7}  {'mean_w':>7}")
results = []
for k in K_VALUES:
    r = eval_k(k)
    results.append(r)
    print(f"{k:>6.1f}  {r['med_imp']*100:>7.2f}%  {r['pct_hurt']:>8.1f}%  "
          f"{r['med_quiet']*100:>6.2f}%  {r['med_noisy']*100:>6.2f}%  "
          f"{r['mean_weight']:>7.3f}")

res = pd.DataFrame(results)
best = res.loc[res["med_imp"].idxmax()]
print(f"\nBest k = {best['k']:.1f}  →  {best['med_imp']*100:.2f}% median improvement  "
      f"({best['pct_hurt']:.1f}% hurt)")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG, AX, TEXT, MUTE = "#0b0f1a", "#060a12", "#c4d4ee", "#7a9cc4"
BLUE, RED, GRN, GOLD = "#3b82f6", "#ef4444", "#22c55e", "#f59e0b"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG)

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(AX)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=10, pad=5)
    ax.set_xlabel(xlabel, color=MUTE, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTE, fontsize=9)

# Panel 1: median improvement vs k (overall + quiet + noisy)
ax = axes[0]
style(ax, "Median scatter improvement vs k", "Shrinkage k", "Median improvement (%)")
ax.plot(res["k"], res["med_imp"]*100,   color=BLUE, lw=2.5, marker="o", ms=6, label="All stars")
ax.plot(res["k"], res["med_quiet"]*100, color=GRN,  lw=2,   marker="s", ms=5, label="Quiet (low CDPP)")
ax.plot(res["k"], res["med_noisy"]*100, color=GOLD, lw=2,   marker="^", ms=5, label="Noisy (high CDPP)")
ax.axvline(5.0, color="white", lw=1, ls="--", alpha=0.4, label="Current k=5")
ax.axvline(best["k"], color=BLUE, lw=1.5, ls=":", alpha=0.7, label=f"Best k={best['k']:.1f}")
ax.axhline(0, color=MUTE, lw=0.5, alpha=0.3)
ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor="#0d1a30", edgecolor="#1c2d45")

# Panel 2: % hurt vs k
ax = axes[1]
style(ax, "% hurt stars vs k", "Shrinkage k", "Stars hurt (%)")
ax.plot(res["k"], res["pct_hurt"], color=RED, lw=2.5, marker="o", ms=6)
ax.axvline(5.0, color="white", lw=1, ls="--", alpha=0.4, label="Current k=5")
ax.axvline(best["k"], color=BLUE, lw=1.5, ls=":", alpha=0.7, label=f"Best k={best['k']:.1f}")
ax.legend(fontsize=7.5, labelcolor=TEXT, facecolor="#0d1a30", edgecolor="#1c2d45")

# Panel 3: improvement vs % hurt trade-off
ax = axes[2]
style(ax, "Improvement vs hurt trade-off", "Stars hurt (%)", "Median improvement (%)")
sc = ax.scatter(res["pct_hurt"], res["med_imp"]*100,
                c=res["k"], cmap="plasma", s=80, zorder=5)
for _, row in res.iterrows():
    ax.annotate(f"k={row['k']:.1f}",
                (row["pct_hurt"], row["med_imp"]*100),
                textcoords="offset points", xytext=(6, 3),
                fontsize=7, color=MUTE)
cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("k", color=TEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(labelcolor=MUTE, labelsize=7)

fig.suptitle(
    f"Shrinkage coefficient sweep — {MODEL}\n"
    f"weight = 1/(1 + k·σ)   |   Best: k={best['k']:.1f} → "
    f"{best['med_imp']*100:.2f}% improvement, {best['pct_hurt']:.1f}% hurt",
    color=TEXT, fontsize=11, y=1.02
)
plt.tight_layout()
out = "eval/shrinkage_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
