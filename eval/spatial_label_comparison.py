"""
Spatial consensus label vs LOO label comparison.

LOO label:
  flux_offset(star, s) = sector_median(star, s) / mean(sector_median(star, j≠s))
  Uses the SAME STAR's other sectors as reference. Noisy for few-sector stars.
  Blind to spatial structure — two neighbours can have very different LOO labels
  even if they share the same detector systematic.

Spatial consensus label:
  spatial_label(star, s) = median( loo_label(neighbour, s) for k nearest neighbours in sector s )
  Pools LOO labels from nearby stars observed in the same sector.
  Directly estimates the detector systematic at this sky position in this sector.
  Less noisy because it averages k stars instead of one star's history.

Key question: does spatial consensus track spatial structure better and disagree
with LOO most strongly for few-sector stars (where LOO is noisiest)?

Outputs:
  eval/spatial_label_comparison.png  — four-panel comparison figure
  eval/spatial_labels.parquet        — parquet with both labels for downstream use
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_topup.parquet")
K_NEIGHBORS = 15    # nearest sky neighbors in same sector
MIN_NBRS    = 5     # minimum neighbors required to compute a valid label

print(f"Loading {PARQUET}...")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["ra", "dec", "sector_median", "flux_offset"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[(df["sector_median"] > 0)]
print(f"  {len(df):,} records, {df['tic_id'].nunique():,} stars")

# ── Build spatial consensus label sector by sector ────────────────────────────
# For each (star, sector): median of k nearest neighbours' LOO labels in that sector.
print(f"Computing spatial consensus labels (k={K_NEIGHBORS} neighbors per sector)...")

df = df.reset_index(drop=True)
spatial_labels = np.full(len(df), np.nan)

for sector, grp in df.groupby("sector"):
    idx = grp.index.values
    ra  = np.deg2rad(grp["ra"].values)
    dec = np.deg2rad(grp["dec"].values)

    # 3D unit vectors — avoids ra wraparound near 0/360°
    xyz = np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])

    loo = grp["flux_offset"].values
    k   = min(K_NEIGHBORS, len(grp) - 1)
    if k < MIN_NBRS:
        continue

    _, ii = cKDTree(xyz).query(xyz, k=k + 1)
    nbr_loo    = loo[ii[:, 1:]]                         # (n, k) neighbour LOO labels
    n_valid    = np.isfinite(nbr_loo).sum(axis=1)
    has_enough = n_valid >= MIN_NBRS

    spatial_labels[idx[has_enough]] = np.median(nbr_loo[has_enough], axis=1)

df["spatial_label"] = spatial_labels
df["loo_label"]     = df["flux_offset"]

valid = df["spatial_label"].notna() & df["loo_label"].notna()
print(f"  Valid spatial labels: {valid.sum():,} / {len(df):,} ({valid.mean()*100:.1f}%)")

d = df[valid].copy()

# ── Figure: four panels ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.patch.set_facecolor("#0b0f1a")
fig.suptitle("Spatial label vs LOO label", color="#c4d4ee", fontsize=14, y=0.995)

BLUE  = "#4f8ef7"
GREEN = "#34c580"
AMBER = "#f59e0b"

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor("#060a12")
    ax.set_xlabel(xlabel, color="#7a9cc4", fontsize=10)
    ax.set_ylabel(ylabel, color="#7a9cc4", fontsize=10)
    ax.set_title(title, color="#c4d4ee", fontsize=11, pad=7)
    ax.tick_params(colors="#3a5070", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")

# ── Panel 1: Distribution comparison ─────────────────────────────────────────
ax = axes[0][0]
bins = np.linspace(0.93, 1.07, 80)
ax.hist(d["loo_label"],     bins=bins, color=BLUE,  alpha=0.55, label="LOO label",     density=True)
ax.hist(d["spatial_label"], bins=bins, color=GREEN, alpha=0.55, label="Spatial label", density=True)
ax.axvline(1.0, color="white", lw=0.8, alpha=0.4, ls="--")
ax.legend(fontsize=9, framealpha=0.2, labelcolor="#c4d4ee",
          facecolor="#0f1c30", edgecolor="#1c2d45")
loo_std  = d["loo_label"].std()
sp_std   = d["spatial_label"].std()
ax.text(0.97, 0.95,
        f"LOO   σ={loo_std:.4f}\nSpatial σ={sp_std:.4f}",
        transform=ax.transAxes, ha="right", va="top",
        color="#c4d4ee", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#0f1c30", ec="#1c2d45"))
style(ax, "flux_offset", "density", "Distribution: LOO vs Spatial label")

# ── Panel 2: LOO vs Spatial scatter ──────────────────────────────────────────
ax = axes[0][1]
# Sample for speed
samp = d.sample(min(30000, len(d)), random_state=42)
ax.scatter(samp["loo_label"], samp["spatial_label"],
           s=0.5, alpha=0.3, color=BLUE, rasterized=True)
lo, hi = 0.93, 1.07
ax.plot([lo, hi], [lo, hi], color="white", lw=0.8, alpha=0.4, ls="--")
r = float(np.corrcoef(d["loo_label"], d["spatial_label"])[0, 1])
mae = float((d["loo_label"] - d["spatial_label"]).abs().mean())
ax.text(0.04, 0.95,
        f"Pearson r = {r:.3f}\nMAE = {mae:.4f}",
        transform=ax.transAxes, va="top",
        color="#c4d4ee", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#0f1c30", ec="#1c2d45"))
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
style(ax, "LOO label", "Spatial label", "Label agreement  (30k sample)")

# ── Panel 3: Disagreement vs n_sectors ───────────────────────────────────────
# Do few-sector stars show larger LOO–spatial discrepancy?
ax = axes[1][0]
d["delta"] = (d["loo_label"] - d["spatial_label"]).abs()
ns_bins  = [1, 2, 3, 4, 5, 7, 10, 15, 25, 100]
ns_labels = ["1","2","3","4","5","6-7","8-10","11-15","16-25","26+"]
ns_cut = pd.cut(d["n_sectors_total"], bins=ns_bins, labels=ns_labels[1:])
ns_grp = d.groupby(ns_cut, observed=True)["delta"].agg(["median","mean","count"])
xs = range(len(ns_grp))
ax.bar(xs, ns_grp["median"], color=BLUE, alpha=0.7, label="|LOO − Spatial| median")
ax.plot(xs, ns_grp["mean"],  color=AMBER, lw=1.5, marker="o", ms=4, label="mean")
ax.set_xticks(xs)
ax.set_xticklabels(ns_grp.index, fontsize=8, color="#7a9cc4")
ax.legend(fontsize=9, framealpha=0.2, labelcolor="#c4d4ee",
          facecolor="#0f1c30", edgecolor="#1c2d45")
for x, row in zip(xs, ns_grp.itertuples()):
    ax.text(x, row.median + 0.0001, f"n={int(row.count):,}",
            ha="center", va="bottom", fontsize=6, color="#3a5070")
style(ax, "n_sectors_total", "|LOO − Spatial label|",
      "Label disagreement vs star sector count\n(few-sector stars have noisier LOO)")

# ── Panel 4: Spatial label on FFI (sector 29, cam 4) ─────────────────────────
ax = axes[1][1]
sec29 = d[(d["sector"] == 29) & (d["cam"] == 4)].copy()

def gnomonic(ra_deg, dec_deg):
    ra0  = np.deg2rad(np.median(ra_deg))
    dec0 = np.deg2rad(np.median(dec_deg))
    ra   = np.deg2rad(ra_deg)
    dec  = np.deg2rad(dec_deg)
    cos_c = np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0)
    x = np.rad2deg(np.cos(dec)*np.sin(ra - ra0) / cos_c)
    y = np.rad2deg((np.cos(dec0)*np.sin(dec) - np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0)) / cos_c)
    return x, y

x29, y29 = gnomonic(sec29["ra"].values, sec29["dec"].values)

lo29, hi29 = np.percentile(sec29["spatial_label"], [1, 99])
ext = max(abs(lo29-1), abs(hi29-1), 0.008)
norm29 = TwoSlopeNorm(vcenter=1.0, vmin=1-ext, vmax=1+ext)

ax.scatter(x29, y29, c=sec29["spatial_label"].values,
           cmap="RdBu_r", norm=norm29, s=1.5, alpha=0.8,
           linewidths=0, rasterized=True)
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm29)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
cbar.set_label("spatial_label", color="#c4d4ee", fontsize=8)
cbar.ax.yaxis.set_tick_params(labelcolor="#7a9cc4", labelsize=7)
cbar.ax.axhline(y=1.0, color="white", lw=0.8, alpha=0.5)

# Compute spatial smoothness: neighbour r for spatial vs LOO
coords29 = np.column_stack([x29, y29])
tree29   = cKDTree(coords29)
_, ii29  = tree29.query(coords29, k=16)
sp_nbr   = sec29["spatial_label"].values[ii29[:, 1:]].mean(axis=1)
loo_nbr  = sec29["loo_label"].values[ii29[:, 1:]].mean(axis=1)
r_sp  = float(np.corrcoef(sec29["spatial_label"].values, sp_nbr)[0, 1])
r_loo = float(np.corrcoef(sec29["loo_label"].values, loo_nbr)[0, 1])

ax.text(0.03, 0.04,
        f"nbr r (spatial) = {r_sp:.3f}\nnbr r (LOO)     = {r_loo:.3f}",
        transform=ax.transAxes, va="bottom",
        color="#c4d4ee", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#0f1c30", ec="#1c2d45"))
style(ax, "ξ (deg)", "η (deg)", "Sector 29, Camera 4 — Spatial label on sky\n(compare to LOO in earlier plot)")

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ── Save parquet ──────────────────────────────────────────────────────────────
out_cols = ["tic_id", "sector", "cam", "ccd", "col", "row",
            "ra", "dec", "n_sectors_total", "loo_label", "spatial_label"]
out_df = df[[c for c in out_cols if c in df.columns]].copy()
out_df.to_parquet("eval/spatial_labels.parquet", index=False)
print(f"Saved labels → eval/spatial_labels.parquet ({len(out_df):,} rows)")

out = "eval/spatial_label_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"Saved figure → {out}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\n── Summary ──")
print(f"LOO std:     {d['loo_label'].std():.4f}")
print(f"Spatial std: {d['spatial_label'].std():.4f}")
print(f"Correlation: r={r:.3f}")
print(f"Mean |diff|: {mae:.4f}")
print(f"\nSector 29, Cam 4 — neighbour r:")
print(f"  LOO label:     {r_loo:.3f}")
print(f"  Spatial label: {r_sp:.3f}")
print(f"  (higher = spatially smoother = better grounded)")
