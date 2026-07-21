"""
Feature space diagnostics for STITCH training data.

Produces:
  stitch_features_marginals.png  — per-feature histograms split by cam
  stitch_features_vs_target.png  — each feature vs flux_offset (hex density)
  stitch_features_correlation.png — feature correlation heatmap
  stitch_features_coverage.png   — 2D position coverage per cam/CCD
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
df["n_sectors_total"] = df.groupby("tic_id")["sector"].transform("count")
for col in ["crowdsap", "cdpp1_0", "pdcvar", "jitter_rms"]:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))

print(f"Loaded {len(df):,} records, {df['tic_id'].nunique():,} stars")

CONTINUOUS = ["col", "row", "delta_sub_col", "delta_sub_row",
              "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
              "log_sector_median", "n_sectors_total"]

LABELS = {
    "col":              "Column (px)",
    "row":              "Row (px)",
    "delta_sub_col":    "Sub-pixel col offset",
    "delta_sub_row":    "Sub-pixel row offset",
    "sector":           "TESS Sector",
    "tmag":             "TESS magnitude",
    "crowdsap":         "Crowding (CROWDSAP)",
    "cdpp1_0":          "CDPP 1hr (ppm)",
    "pdcvar":           "PDC variability",
    "jitter_rms":       "Jitter RMS (px)",
    "log_sector_median":"log(sector median flux)",
    "n_sectors_total":  "N sectors observed",
}

CAM_COLORS = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3"}

# ── 1. Marginal distributions by cam ─────────────────────────────────────────

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.ravel()

for i, feat in enumerate(CONTINUOUS):
    ax = axes[i]
    for cam in [1, 2, 3, 4]:
        sub = df[df["cam"] == cam][feat].dropna()
        ax.hist(sub, bins=60, alpha=0.45, color=CAM_COLORS[cam],
                label=f"Cam{cam} (n={len(sub):,})", density=True, histtype="stepfilled")
    ax.set_xlabel(LABELS[feat], fontsize=9)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(feat, fontsize=9, fontweight="bold")
    if i == 0:
        ax.legend(fontsize=7, ncol=2)

fig.suptitle("STITCH — Feature Marginal Distributions by Camera", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("stitch_features_marginals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → stitch_features_marginals.png")

# ── 2. Feature vs flux_offset (hex density) ──────────────────────────────────

fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.ravel()

for i, feat in enumerate(CONTINUOUS):
    ax = axes[i]
    x = df[feat].dropna()
    y = df.loc[x.index, "flux_offset"]
    hb = ax.hexbin(x, y, gridsize=60, cmap="YlOrRd", mincnt=1,
                   norm=LogNorm(), linewidths=0.2)
    ax.axhline(1.0, color="steelblue", lw=1, ls="--", alpha=0.7)
    ax.set_xlabel(LABELS[feat], fontsize=9)
    ax.set_ylabel("flux_offset", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(feat, fontsize=9, fontweight="bold")
    plt.colorbar(hb, ax=ax, label="count", pad=0.01)

    # Overlay running median
    try:
        sorted_x = x.sort_values()
        sorted_y = y.loc[sorted_x.index]
        bins = np.percentile(sorted_x, np.linspace(2, 98, 40))
        bin_idx = np.digitize(sorted_x, bins)
        medians = [sorted_y[bin_idx == b].median() for b in range(1, len(bins))]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        valid = [m for m in medians if not np.isnan(m)]
        if len(valid) == len(bin_centers):
            ax.plot(bin_centers, medians, color="steelblue", lw=1.5, alpha=0.85)
    except Exception:
        pass

fig.suptitle("STITCH — Features vs flux_offset (log density + running median)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("stitch_features_vs_target.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → stitch_features_vs_target.png")

# ── 3. Correlation heatmap ────────────────────────────────────────────────────

corr_df = df[CONTINUOUS + ["flux_offset"]].corr()
short_labels = [l.replace(" (px)", "").replace(" (ppm)", "").replace(" (CROWDSAP)", "")
                for l in [LABELS.get(c, c) for c in CONTINUOUS]] + ["flux_offset"]

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.03)
ax.set_xticks(range(len(corr_df))); ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr_df))); ax.set_yticklabels(short_labels, fontsize=8)
for i in range(len(corr_df)):
    for j in range(len(corr_df)):
        v = corr_df.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=6, color="white" if abs(v) > 0.5 else "black")
ax.set_title("STITCH — Feature Correlation Matrix", fontsize=12)
plt.tight_layout()
plt.savefig("stitch_features_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → stitch_features_correlation.png")

# ── 4. 2D position coverage per cam/CCD ──────────────────────────────────────

fig, axes = plt.subplots(4, 4, figsize=(14, 13), sharex=True, sharey=True,
                          gridspec_kw={"hspace": 0.08, "wspace": 0.08})

for i, cam in enumerate([1, 2, 3, 4]):
    for j, ccd in enumerate([1, 2, 3, 4]):
        ax = axes[i][j]
        sub = df[(df["cam"] == cam) & (df["ccd"] == ccd)]
        if len(sub) > 0:
            hb = ax.hexbin(sub["col"], sub["row"], gridsize=35,
                           cmap="YlOrRd", mincnt=1, norm=LogNorm(), linewidths=0.1)
            ax.text(0.04, 0.96, f"Cam{cam}/CCD{ccd}\nn={sub['tic_id'].nunique():,}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        if i == 3: ax.set_xlabel("Column", fontsize=8)
        if j == 0: ax.set_ylabel("Row", fontsize=8)
        ax.tick_params(labelsize=6)

fig.suptitle("STITCH — Training Star Coverage (col/row, log density)", fontsize=12, y=1.005)
plt.savefig("stitch_features_coverage.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → stitch_features_coverage.png")

# ── 5. Print key stats ────────────────────────────────────────────────────────

print("\n=== Feature summary ===")
print(f"{'Feature':<22} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}  corr w/ flux_offset")
print("  " + "─"*75)
for feat in CONTINUOUS:
    s = df[feat].dropna()
    r = df[[feat, "flux_offset"]].dropna().corr().iloc[0, 1]
    print(f"  {feat:<20} {s.mean():>9.3f} {s.std():>9.3f} {s.min():>9.3f} {s.max():>9.3f}  {r:+.3f}")
