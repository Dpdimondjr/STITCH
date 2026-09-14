"""
Scatter reduction evaluation — physically grounded, LOO-independent.

For each test star, divides every sector's median flux by the model's predicted
flux_offset, then measures whether the sector medians become flatter (lower std).
The model never sees the star's flux during training, so this is a clean test.

Usage:
  python3 eval/scatter_eval.py stitch_nsf_pdc_full.pt training_data_topup_pdc.parquet
  python3 eval/scatter_eval.py stitch_nsf_v3.pt training_data_topup.parquet
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

MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_pdc_full.pt")
PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")
print(f"Model: {MODEL}  |  Parquet: {PARQUET}")

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

# ── Load & clean data ─────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[df["sector_median"] > 0]
df = df[df["n_sectors_total"] >= 4]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Reproduce train/val/test split ────────────────────────────────────────────
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

# Keep test stars with ≥5 sectors for a meaningful scatter measurement
test_df = df[df["tic_id"].isin(te_tics)].copy()
sector_counts = test_df.groupby("tic_id")["sector"].count()
good_tics = sector_counts[sector_counts >= 5].index
test_df = test_df[test_df["tic_id"].isin(good_tics)].copy()
print(f"Test stars (≥5 sectors): {test_df['tic_id'].nunique():,}  records: {len(test_df):,}")

# ── Predict ───────────────────────────────────────────────────────────────────
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
C = torch.tensor(make_context(test_df)).to(device)

with torch.no_grad():
    samples = flow(C).sample((300,)).squeeze(-1)
    mu_z    = samples.mean(0).cpu().numpy()
    sig_z   = samples.std(0).cpu().numpy()

pred_raw = mu_z * y_std + y_mean
pred_std = sig_z * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)
pred     = weight * pred_raw + (1 - weight) * 1.0

test_df = test_df.reset_index(drop=True)
test_df["pred_offset"] = pred
test_df["pred_std"]    = pred_std
test_df["weight"]      = weight

# ── Per-star scatter ──────────────────────────────────────────────────────────
results = []
for tic, g in test_df.groupby("tic_id"):
    g = g.sort_values("sector")
    raw  = g["sector_median"].values
    po   = g["pred_offset"].values
    ref  = raw.mean()

    raw_norm    = raw / ref
    stitch_norm = (raw / po) / (raw / po).mean()   # relative scatter only

    sc_before = raw_norm.std()
    sc_after  = stitch_norm.std()
    improv    = (sc_before - sc_after) / sc_before

    results.append({
        "tic_id":        tic,
        "n_sectors":     len(g),
        "cam":           int(g["cam"].mode()[0]),
        "tmag":          g["tmag"].iloc[0],
        "sc_before":     sc_before,
        "sc_after":      sc_after,
        "improv":        improv,
        "sectors":       g["sector"].values,
        "raw_norm":      raw_norm,
        "stitch_norm":   stitch_norm,
        "mean_weight":   g["weight"].mean(),
    })

res = pd.DataFrame([{k: v for k, v in r.items()
                     if k not in ("sectors","raw_norm","stitch_norm")}
                    for r in results])

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"SCATTER REDUCTION  (n={len(res):,} test stars, ≥5 sectors)")
print(f"{'─'*60}")
print(f"  Scatter before:  {res['sc_before'].median():.4f}  (median across stars)")
print(f"  Scatter after:   {res['sc_after'].median():.4f}")
print(f"  Improvement:     {res['improv'].median()*100:.1f}%  (median)")
print(f"  Stars improved:  {(res['improv']>0).mean()*100:.1f}%")
print(f"  Mean weight:     {res['mean_weight'].mean():.3f}  (1.0 = full correction)")

print(f"\n  By n_sectors:")
for label, lo, hi in [("5-6",5,7),("7-10",7,11),("11-20",11,21),("21+",21,999)]:
    sub = res[(res["n_sectors"]>=lo)&(res["n_sectors"]<hi)]
    if len(sub) == 0: continue
    print(f"    {label:<6}  n={len(sub):>4}  "
          f"improv={sub['improv'].median()*100:>5.1f}%  "
          f"better={( sub['improv']>0).mean()*100:>4.0f}%")

print(f"\n  By camera:")
for cam, g in res.groupby("cam"):
    print(f"    Cam{int(cam)}  n={len(g):>4}  "
          f"improv={g['improv'].median()*100:>5.1f}%  "
          f"better={(g['improv']>0).mean()*100:>4.0f}%")

print(f"\n  By Tmag:")
for label, lo, hi in [("<9",0,9),("9-10",9,10),("10-11",10,11),("11-12",11,12),("12+",12,99)]:
    sub = res[(res["tmag"]>=lo)&(res["tmag"]<hi)]
    if len(sub) == 0: continue
    print(f"    {label:<6}  n={len(sub):>4}  "
          f"improv={sub['improv'].median()*100:>5.1f}%  "
          f"better={(sub['improv']>0).mean()*100:>4.0f}%")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0b0f1a")

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor("#060a12")
    ax.set_xlabel(xlabel, color="#7a9cc4", fontsize=10)
    ax.set_ylabel(ylabel, color="#7a9cc4", fontsize=10)
    ax.set_title(title, color="#c4d4ee", fontsize=11, pad=7)
    ax.tick_params(colors="#3a5070", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")

# Panel 1: improvement distribution
ax = axes[0]
bins = np.linspace(-60, 80, 60)
ax.hist(res["improv"]*100, bins=bins, color="#4f8ef7", alpha=0.8, edgecolor="none")
ax.axvline(0, color="white", lw=1, ls="--", alpha=0.5)
ax.axvline(res["improv"].median()*100, color="#34c580", lw=2,
           label=f"Median = {res['improv'].median()*100:.1f}%")
ax.legend(fontsize=9, framealpha=0.2, labelcolor="#c4d4ee",
          facecolor="#0f1c30", edgecolor="#1c2d45")
style(ax, "Scatter improvement (%)", "Stars",
      f"Distribution of scatter reduction\n{(res['improv']>0).mean()*100:.0f}% of stars improved")

# Panel 2: scatter before vs after
ax = axes[1]
ax.scatter(res["sc_before"]*100, res["sc_after"]*100,
           c=res["tmag"], cmap="plasma", s=3, alpha=0.4, rasterized=True,
           vmin=6, vmax=13)
lim = res[["sc_before","sc_after"]].max().max()*100*1.05
ax.plot([0,lim],[0,lim], color="white", lw=0.8, ls="--", alpha=0.4)
sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(6,13))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Tmag", color="#c4d4ee", fontsize=9)
cbar.ax.yaxis.set_tick_params(labelcolor="#7a9cc4", labelsize=8)
style(ax, "Scatter before (%)", "Scatter after (%)",
      "Per-star scatter: before vs after\n(below line = improved, colour = Tmag)")

# Panel 3: improvement by Tmag bin
ax = axes[2]
tmag_bins  = [0, 9, 10, 11, 12, 99]
tmag_labels = ["<9","9-10","10-11","11-12","12+"]
res["tmag_bin"] = pd.cut(res["tmag"], bins=tmag_bins, labels=tmag_labels)
g = res.groupby("tmag_bin", observed=True)["improv"]
med  = g.median() * 100
p25  = g.quantile(0.25) * 100
p75  = g.quantile(0.75) * 100
xs   = range(len(med))
colors_bar = ["#34c580" if v > 0 else "#ef4444" for v in med]
ax.bar(xs, med, color=colors_bar, alpha=0.75)
ax.errorbar(xs, med, yerr=[med-p25, p75-med],
            fmt="none", color="#c4d4ee", lw=1.5, capsize=4)
ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.4)
ax.set_xticks(xs); ax.set_xticklabels(tmag_labels, fontsize=9, color="#7a9cc4")
for x, (_, row) in zip(xs, res.groupby("tmag_bin", observed=True).size().items()):
    ax.text(x, med.iloc[x] + 0.5, f"n={row}", ha="center", va="bottom",
            fontsize=7, color="#3a5070")
style(ax, "Tmag", "Median scatter improvement (%)",
      "Improvement by brightness\n(does the Tmag cut matter here?)")

plt.tight_layout()
out = "eval/scatter_eval.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"\nSaved → {out}")
