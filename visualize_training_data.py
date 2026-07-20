"""
Visualizations for STITCH training_data.parquet.

Plots produced:
  1. flux_offset distribution (overall + per cam/CCD)
  2. Spatial heatmap of mean flux_offset on (col, row) grid per CCD
  3. flux_offset vs sector (temporal drift per cam)
  4. Feature correlation with flux_offset
  5. flux_offset vs Tmag
  6. Records per cam/CCD (coverage bar chart)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

df = pd.read_parquet("training_data.parquet")

# Drop extreme outliers (flux_offset < 0.5 or > 1.5 are almost certainly bad)
n_before = len(df)
df = df[df["flux_offset"].between(0.5, 1.5)].copy()
print(f"Dropped {n_before - len(df)} outlier records (flux_offset outside 0.5–1.5)")
print(f"Working with {len(df)} records from {df['tic_id'].nunique()} stars\n")

CAM_COLORS = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3"}
CCD_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}

fig = plt.figure(figsize=(20, 24))
fig.suptitle("STITCH Training Data — flux_offset Characterization\n"
             f"({len(df)} records, {df['tic_id'].nunique()} stars, "
             f"sectors {int(df['sector'].min())}–{int(df['sector'].max())})",
             fontsize=14, y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)


# ── 1. flux_offset distribution ───────────────────────────────────────────────

ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df["flux_offset"], bins=80, color="#555", alpha=0.7, edgecolor="none")
ax1.axvline(1.0, color="red", lw=1.5, ls="--", label="1.0 reference")
ax1.axvline(df["flux_offset"].mean(), color="orange", lw=1.5, ls="-",
            label=f"mean={df['flux_offset'].mean():.4f}")
ax1.set_xlabel("flux_offset")
ax1.set_ylabel("count")
ax1.set_title("1. flux_offset distribution (overall)")
ax1.legend(fontsize=9)

# Per cam/CCD overlay
ax1b = fig.add_subplot(gs[0, 2])
for cam in sorted(df["cam"].unique()):
    sub = df[df["cam"] == cam]["flux_offset"]
    ax1b.hist(sub, bins=40, alpha=0.5, label=f"Cam{int(cam)} (n={len(sub)})",
              color=CAM_COLORS.get(int(cam), "gray"), edgecolor="none")
ax1b.axvline(1.0, color="red", lw=1.2, ls="--")
ax1b.set_xlabel("flux_offset")
ax1b.set_ylabel("count")
ax1b.set_title("1b. Per-camera distribution")
ax1b.legend(fontsize=8)


# ── 2. Spatial heatmap per CCD (Cam4 only — has enough data) ─────────────────

cam4 = df[df["cam"] == 4].copy()
ccds = sorted(cam4["ccd"].unique())
n_bins = 12

for i, ccd in enumerate(ccds):
    ax = fig.add_subplot(gs[1, i if i < 3 else 2])
    sub = cam4[cam4["ccd"] == ccd]

    col_bins = np.linspace(sub["col"].min(), sub["col"].max(), n_bins + 1)
    row_bins = np.linspace(sub["row"].min(), sub["row"].max(), n_bins + 1)

    grid = np.full((n_bins, n_bins), np.nan)
    for ci in range(n_bins):
        for ri in range(n_bins):
            mask = (
                (sub["col"] >= col_bins[ci]) & (sub["col"] < col_bins[ci+1]) &
                (sub["row"] >= row_bins[ri]) & (sub["row"] < row_bins[ri+1])
            )
            if mask.sum() >= 2:
                grid[ri, ci] = sub.loc[mask, "flux_offset"].mean()

    vmin, vmax = np.nanpercentile(grid[~np.isnan(grid)], [5, 95]) if not np.all(np.isnan(grid)) else (0.95, 1.05)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm,
                   extent=[col_bins[0], col_bins[-1], row_bins[0], row_bins[-1]])
    plt.colorbar(im, ax=ax, shrink=0.8, label="mean flux_offset")
    ax.set_xlabel("col (px)")
    ax.set_ylabel("row (px)")
    ax.set_title(f"2. Cam4/CCD{ccd} spatial\n(n={len(sub)} records)")


# ── 3. flux_offset vs sector (temporal) ──────────────────────────────────────

ax3 = fig.add_subplot(gs[2, :2])
for cam in sorted(df["cam"].unique()):
    sub = df[df["cam"] == cam].groupby("sector")["flux_offset"].agg(["mean", "std"]).reset_index()
    ax3.errorbar(sub["sector"], sub["mean"], yerr=sub["std"],
                 fmt="o-", ms=3, lw=1, alpha=0.7,
                 color=CAM_COLORS.get(int(cam), "gray"),
                 label=f"Cam{int(cam)}")
ax3.axhline(1.0, color="red", lw=1, ls="--")
ax3.set_xlabel("sector")
ax3.set_ylabel("mean flux_offset ± std")
ax3.set_title("3. Temporal variation: flux_offset vs sector")
ax3.legend(fontsize=9)


# ── 4. Feature correlations ───────────────────────────────────────────────────

ax4 = fig.add_subplot(gs[2, 2])
features = ["col", "row", "tmag", "crowdsap", "cdpp1_0", "pdcvar",
            "jitter_rms", "sector", "n_sectors_total"]
features = [f for f in features if f in df.columns]
corrs = df[features + ["flux_offset"]].corr()["flux_offset"].drop("flux_offset")
colors = ["#d73027" if v > 0 else "#4575b4" for v in corrs.values]
bars = ax4.barh(corrs.index, corrs.values, color=colors, edgecolor="none")
ax4.axvline(0, color="black", lw=0.8)
ax4.set_xlabel("Pearson r with flux_offset")
ax4.set_title("4. Feature correlations")
for bar, v in zip(bars, corrs.values):
    ax4.text(v + (0.003 if v >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
             f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)


# ── 5. flux_offset vs Tmag ────────────────────────────────────────────────────

ax5 = fig.add_subplot(gs[3, 0])
for cam in sorted(df["cam"].unique()):
    sub = df[df["cam"] == cam]
    ax5.scatter(sub["tmag"], sub["flux_offset"], s=8, alpha=0.4,
                color=CAM_COLORS.get(int(cam), "gray"), label=f"Cam{int(cam)}")
ax5.axhline(1.0, color="red", lw=1, ls="--")
ax5.set_xlabel("Tmag")
ax5.set_ylabel("flux_offset")
ax5.set_title("5. flux_offset vs Tmag")
ax5.legend(fontsize=8, markerscale=2)


# ── 6. Records per cam/CCD ────────────────────────────────────────────────────

ax6 = fig.add_subplot(gs[3, 1])
coverage = df.groupby(["cam", "ccd"]).size().reset_index(name="n")
labels = [f"Cam{int(r.cam)}/CCD{int(r.ccd)}" for _, r in coverage.iterrows()]
bar_colors = [CAM_COLORS.get(int(r.cam), "gray") for _, r in coverage.iterrows()]
ax6.bar(labels, coverage["n"], color=bar_colors, edgecolor="none")
ax6.set_xlabel("cam/CCD")
ax6.set_ylabel("records")
ax6.set_title("6. Records per cam/CCD")
ax6.tick_params(axis="x", rotation=45)
for bar, n in zip(ax6.patches, coverage["n"]):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(n), ha="center", va="bottom", fontsize=7)


# ── 7. flux_offset std per cam/CCD (signal strength) ─────────────────────────

ax7 = fig.add_subplot(gs[3, 2])
spread = df.groupby(["cam", "ccd"])["flux_offset"].std().reset_index(name="std")
labels7 = [f"Cam{int(r.cam)}/CCD{int(r.ccd)}" for _, r in spread.iterrows()]
bar_colors7 = [CAM_COLORS.get(int(r.cam), "gray") for _, r in spread.iterrows()]
ax7.bar(labels7, spread["std"], color=bar_colors7, edgecolor="none")
ax7.set_xlabel("cam/CCD")
ax7.set_ylabel("flux_offset std")
ax7.set_title("7. Signal strength per cam/CCD")
ax7.tick_params(axis="x", rotation=45)
for bar, v in zip(ax7.patches, spread["std"]):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f"{v:.3f}", ha="center", va="bottom", fontsize=7)


plt.savefig("training_data_overview.png", dpi=150, bbox_inches="tight")
print("Saved → training_data_overview.png")

# Print summary stats
print("\n=== Key numbers ===")
print(f"flux_offset std overall:  {df['flux_offset'].std():.4f}  ({df['flux_offset'].std()*100:.1f}%)")
print(f"Strongest correlation:    {corrs.abs().idxmax()} (r={corrs[corrs.abs().idxmax()]:.3f})")
print(f"Cam4 share of records:    {(df['cam']==4).mean()*100:.0f}%")
print(f"\nPer-cam flux_offset std:")
for cam, g in df.groupby("cam"):
    print(f"  Cam{int(cam)}: {g['flux_offset'].std():.4f}")
