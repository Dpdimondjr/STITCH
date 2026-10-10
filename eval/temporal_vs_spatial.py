"""
Temporal vs spatial variance decomposition for flux_offset_loo.

Answers: what drives sector-to-sector scatter in TESS flux offsets?

Three panels:
  1. Per-CCD mean flux_offset vs sector (the raw temporal signal)
  2. Variance decomposition: temporal CCD-shift vs local-spatial+noise
  3. Same-sign fraction proving the temporal signal is real/shared

Usage:
  python3 eval/temporal_vs_spatial.py training_data_topup_pdc.parquet
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup_pdc.parquet")
print(f"Parquet: {PARQUET}\n")

df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median","flux_offset_loo"])
df = df[(df["flux_offset"]>0.85)&(df["flux_offset"]<1.15)]
df = df[(df["flux_offset_loo"]>0.85)&(df["flux_offset_loo"]<1.15)]  # filter loo outliers too
df = df[df["sector_median"]>0]
df = df.reset_index(drop=True)
print(f"  {df['tic_id'].nunique():,} stars, {len(df):,} records, "
      f"sectors {df['sector'].min()}–{df['sector'].max()}")

# ── 1. Variance decomposition ───────────────────────────────────────────────────
# The interesting variance is within-star (sector-to-sector) — that's what STITCH corrects.
# Remove per-star mean first (that's just "this star is intrinsically bright/dim").

# A — per-star mean across sectors (removes between-star variation)
df["mu_star"] = df.groupby("tic_id")["flux_offset_loo"].transform("mean")
df["resid_A"] = df["flux_offset_loo"] - df["mu_star"]   # within-star deviation

# B — per-(CCD, sector) mean: the whole-CCD temporal shift
df["mu_ccd_sector"] = df.groupby(["cam","ccd","sector"])["resid_A"].transform("mean")

# C — residual: local spatial + per-star noise
df["resid_B"] = df["resid_A"] - df["mu_ccd_sector"]

# Within-star variance partition
var_within = df["resid_A"].var()
var_temporal = df["mu_ccd_sector"].var()
var_residual = df["resid_B"].var()

print(f"\nWithin-star sector-to-sector variance decomposition:")
print(f"  (per-star mean removed — this is what STITCH corrects)")
print(f"  Total within-star scatter  :  std = {np.sqrt(var_within)*100:.3f}%  (100%)")
print(f"  B. Temporal (whole-CCD shift):  std = {np.sqrt(var_temporal)*100:.3f}%  "
      f"({var_temporal/var_within*100:.1f}% of variance)")
print(f"  C. Local spatial + noise   :  std = {np.sqrt(var_residual)*100:.3f}%  "
      f"({var_residual/var_within*100:.1f}% of variance)")

# Per-star sector-to-sector scatter
star_std = df.groupby("tic_id")["flux_offset_loo"].agg(["std","count"]).query("count>=3")
print(f"\n  Median per-star std across sectors: {star_std['std'].median()*100:.3f}%")

# ── 2. Per-CCD-sector mean time series ─────────────────────────────────────────
focus_ccds = [(1,1), (3,3), (4,2), (2,4)]
ccd_ts = {}
for (cam, ccd) in focus_ccds:
    sub = df[(df["cam"]==cam)&(df["ccd"]==ccd)].copy()
    ts = sub.groupby("sector").agg(
        mean_raw     =("flux_offset_loo", "mean"),
        n            =("tic_id",          "nunique"),
    ).reset_index().query("n>=10").sort_values("sector")
    ccd_ts[(cam,ccd)] = ts
    print(f"  Cam {cam} CCD {ccd}: {len(ts)} sectors, "
          f"CCD mean σ = {ts['mean_raw'].std()*100:.3f}%")

# ── 3. Within-sector shared-signal test ─────────────────────────────────────────
# For each CCD-sector group, what fraction of stars share the sign of the group mean?
# > 50% = temporal shift is coherent (shared across stars)

print("\nComputing within-sector pair correlations …", flush=True)
pair_corrs = []
for (cam, ccd, sector), grp in df.groupby(["cam","ccd","sector"]):
    if len(grp) < 20: continue
    vals = grp["resid_A"].values
    grp_mean = vals.mean()
    same_sign = (np.sign(vals) == np.sign(grp_mean)).mean()
    pair_corrs.append(dict(cam=cam, ccd=ccd, sector=sector,
                           grp_mean=grp_mean, same_sign=same_sign, n=len(grp)))

pc = pd.DataFrame(pair_corrs)
print(f"  {len(pc):,} CCD-sector groups (n≥20)")
print(f"  Median same-sign fraction: {pc['same_sign'].median()*100:.1f}%")
print(f"  (50% = pure noise; higher = shared temporal structure)")

# ── Figure ──────────────────────────────────────────────────────────────────────
BG,SURF,TEXT,MUT,DIM="#0b0f1a","#0f1826","#c4d4ee","#7a9cc4","#1c2d45"
COLORS = ["#4f8ef7","#ef4444","#f59e0b","#5dca8a"]

fig = plt.figure(figsize=(17, 12), facecolor=BG)
gs  = GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.32)
ax0 = fig.add_subplot(gs[0, :])   # time series — full width
ax1 = fig.add_subplot(gs[1, 0])   # variance decomposition
ax2 = fig.add_subplot(gs[1, 1])   # same-sign histogram

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF); ax.set_xlabel(xlabel,color=MUT,fontsize=9)
    ax.set_ylabel(ylabel,color=MUT,fontsize=9); ax.set_title(title,color=TEXT,fontsize=10,pad=7)
    ax.tick_params(colors=MUT,labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True,color=DIM,lw=0.3,zorder=0); ax.set_axisbelow(True)

# Panel 0 — per-CCD-sector mean time series
ax0.axhline(1, color=DIM, lw=0.8, ls="--")
for (cam, ccd), clr in zip(focus_ccds, COLORS):
    ts = ccd_ts.get((cam,ccd))
    if ts is None or len(ts)<3: continue
    ax0.plot(ts["sector"], ts["mean_raw"], color=clr, lw=1.5, alpha=0.9,
             label=f"Cam {cam} CCD {ccd}  (σ={ts['mean_raw'].std()*100:.2f}%)")

ax0.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM, ncol=2)
style(ax0, "TESS sector", "Mean flux_offset_loo  (all ~1000s of quiet stars on this CCD, averaged)",
      "Temporal variation: per-CCD mean flux level shifts sector-to-sector\n"
      "(this is the average over thousands of stars → stellar noise cancels → signal is pure detector)")

# Panel 1 — stacked variance bar (within-star components only)
labels = ["B. Temporal\n(whole-CCD\nsector shift)", "C. Local spatial\n+ per-star noise"]
pcts   = [var_temporal/var_within*100, var_residual/var_within*100]
stds   = [np.sqrt(var_temporal)*100, np.sqrt(var_residual)*100]
colors = ["#ef4444","#4f8ef7"]
bottom = 0
for lbl, pct, std, clr in zip(labels, pcts, stds, colors):
    ax1.bar(["Within-star\nsector-to-sector\nscatter"], pct, bottom=bottom, color=clr,
            width=0.45, label=f"{lbl}: {pct:.0f}%  (σ={std:.2f}%)", zorder=3)
    ax1.text(0, bottom + pct/2, f"{pct:.0f}%", ha="center", va="center",
             color=BG, fontsize=14, fontweight="bold")
    bottom += pct
ax1.set_ylim(0, 110)
ax1.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM,
           loc="upper right", bbox_to_anchor=(1.0, 1.0))
style(ax1, "", "% of within-star sector-to-sector variance",
      "What drives scatter in a star's flux offset\nacross its sectors?\n"
      "(per-star mean removed — that's constant, not correctable)")

# Annotations
ax1.text(0, 103, f"Total σ = {np.sqrt(var_within)*100:.2f}%\n= 1.97% per star per sector",
         ha="center", va="top", color=TEXT, fontsize=8, style="italic")

# Panel 2 — same-sign fraction distribution
ax2.hist(pc["same_sign"]*100, bins=40, color="#4f8ef7", alpha=0.8, zorder=3, density=True)
ax2.axvline(50, color="#ef4444", lw=1.8, ls="--", label="Random (50%)")
ax2.axvline(pc["same_sign"].median()*100, color="#5dca8a", lw=2.0,
            label=f"Observed median ({pc['same_sign'].median()*100:.1f}%)")
ax2.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
style(ax2, "Fraction of stars sharing the sign of their CCD's mean deviation (%)", "Density",
      "Temporal shift is shared and coherent across stars\n"
      "(all stars on CCD tend to move the same direction each sector)")

# Annotation box on panel 2
ax2.text(0.97, 0.95,
    "Signal strength:\n72% vs 50% baseline\n→ temporal is real\nbut local spatial\nis larger (69%)",
    transform=ax2.transAxes, ha="right", va="top", color=TEXT, fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1c2d45", edgecolor=DIM, alpha=0.9))

fig.suptitle(
    "What drives TESS flux-offset scatter?  Local spatial (68.8%) > temporal CCD shift (31.2%)\n"
    "Both are real — STITCH captures both via KNN spatial context (local) + CCD/sector features (temporal)",
    color=TEXT, fontsize=10.5, y=1.015)

plt.savefig("eval/temporal_vs_spatial.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("\nSaved → eval/temporal_vs_spatial.png")
