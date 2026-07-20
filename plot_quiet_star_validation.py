"""
Validate that TARS-selected quiet stars are genuinely photometrically stable.

For each star: normalize every sector median by that star's own global mean.
A truly quiet star should sit at 1.0 every sector — any spread is instrumental.

Produces: stitch_quiet_star_validation.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Palette (from reference) ──────────────────────────────────────────────────
SURFACE      = "#fcfcfb"
INK_PRIMARY  = "#0b0b0b"
INK_MUTED    = "#898781"
GRIDLINE     = "#e1e0d9"
BLUE_MED     = "#2a78d6"   # series-1, median line
BLUE_LIGHT   = "#86b6ef"   # step 250, IQR band
BLUE_XLIGHT  = "#cde2fb"   # step 100, 10-90 band
GRAY_TRACE   = "#c3c2b7"   # muted, individual star traces

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["sector_median", "sector", "tic_id"])
df = df[df["sector_median"] > 0]

# Normalize each sector to that star's global mean (not LOO — true global)
global_means = df.groupby("tic_id")["sector_median"].transform("mean")
df["norm_median"] = df["sector_median"] / global_means

# Clip extreme outliers for display only
df = df[df["norm_median"].between(0.88, 1.12)]

n_sectors_per_star = df.groupby("tic_id")["sector"].count()
print(f"Total stars: {df['tic_id'].nunique():,}")
print(f"Total records: {len(df):,}")

# ── Summary stats ──────────────────────────────────────────────────────────────

within_1pct  = (df["norm_median"].between(0.99, 1.01)).mean() * 100
within_2pct  = (df["norm_median"].between(0.98, 1.02)).mean() * 100
overall_std  = df["norm_median"].std()
print(f"\nQuiet star validation:")
print(f"  Within 1% of global: {within_1pct:.1f}% of sector-observations")
print(f"  Within 2% of global: {within_2pct:.1f}% of sector-observations")
print(f"  Overall std of normalized medians: {overall_std:.4f}")

# ── Panel 1: spaghetti of individual star traces ──────────────────────────────

# Sample stars with >= 6 sectors for cleaner traces; one per cam for diversity
rng = np.random.default_rng(42)
sample_tics = []
for cam in [1, 2, 3, 4]:
    pool = n_sectors_per_star[
        n_sectors_per_star.index.isin(df[df["cam"] == cam]["tic_id"])
        & (n_sectors_per_star >= 6)
    ].index.tolist()
    n = min(60, len(pool))
    sample_tics.extend(rng.choice(pool, n, replace=False).tolist())

spaghetti = df[df["tic_id"].isin(sample_tics)].copy()

# Per-sector percentiles across ALL stars (not just sample)
sector_stats = (df.groupby("sector")["norm_median"]
                  .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                  .unstack()
                  .reset_index()
                  .sort_values("sector"))
sector_stats.columns = ["sector", "p10", "p25", "p50", "p75", "p90"]

# ── Panel 2: per-sector box data ──────────────────────────────────────────────

# Use sector bins to avoid crowding (group nearby sectors)
df["sector_bin"] = (df["sector"] // 5) * 5 + 2   # centres: 2, 7, 12, ...
bin_stats = (df.groupby("sector_bin")["norm_median"]
               .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
               .unstack()
               .reset_index())
bin_stats.columns = ["sector_bin", "p10", "p25", "p50", "p75", "p90"]

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                          facecolor=SURFACE,
                          gridspec_kw={"hspace": 0.45})

for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)

# ── Panel 1: spaghetti ────────────────────────────────────────────────────────

ax = axes[0]

# Individual star traces — thin, muted
for tic, g in spaghetti.groupby("tic_id"):
    g = g.sort_values("sector")
    ax.plot(g["sector"], g["norm_median"],
            color=GRAY_TRACE, lw=0.6, alpha=0.35, zorder=1)

# Percentile bands
s  = sector_stats["sector"].values
ax.fill_between(s, sector_stats["p10"], sector_stats["p90"],
                color=BLUE_XLIGHT, alpha=0.55, zorder=2, label="10th–90th pct")
ax.fill_between(s, sector_stats["p25"], sector_stats["p75"],
                color=BLUE_LIGHT, alpha=0.75, zorder=3, label="25th–75th pct")
ax.plot(s, sector_stats["p50"],
        color=BLUE_MED, lw=2.0, zorder=4, label="Median")

# Reference line
ax.axhline(1.0, color=INK_PRIMARY, lw=1.0, ls="--", alpha=0.5, zorder=5)

ax.set_ylabel("Sector median / star global mean", fontsize=10, color=INK_PRIMARY)
ax.set_xlabel("TESS Sector", fontsize=10, color=INK_PRIMARY)
ax.set_ylim(0.92, 1.08)
ax.set_title(
    f"TARS quiet stars — raw sector medians normalised to each star's global mean\n"
    f"Showing {len(sample_tics)} sampled stars (thin lines) + population percentiles  ·  "
    f"{within_1pct:.0f}% of sector-observations within ±1%",
    fontsize=10, color=INK_PRIMARY, loc="left"
)

legend_handles = [
    mpatches.Patch(color=BLUE_XLIGHT, label="10th–90th pct"),
    mpatches.Patch(color=BLUE_LIGHT,  label="25th–75th pct"),
    plt.Line2D([0], [0], color=BLUE_MED, lw=2, label="Median"),
    plt.Line2D([0], [0], color=GRAY_TRACE, lw=1.2, alpha=0.6, label="Individual star"),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right",
          framealpha=0.85, edgecolor=GRIDLINE)

# ── Panel 2: per-sector distribution (binned) ────────────────────────────────

ax = axes[1]

bins  = bin_stats["sector_bin"].values
width = 3.5

ax.fill_between(bins, bin_stats["p10"], bin_stats["p90"],
                step="mid", color=BLUE_XLIGHT, alpha=0.55, label="10th–90th pct")
ax.fill_between(bins, bin_stats["p25"], bin_stats["p75"],
                step="mid", color=BLUE_LIGHT, alpha=0.75, label="25th–75th pct")
ax.plot(bins, bin_stats["p50"],
        color=BLUE_MED, lw=2.0, ds="steps-mid", label="Median")
ax.axhline(1.0, color=INK_PRIMARY, lw=1.0, ls="--", alpha=0.5)

# Annotate std of the median line
median_std = bin_stats["p50"].std()
ax.text(0.02, 0.06,
        f"Std of sector medians (population median): {median_std:.4f}\n"
        f"Overall std (all star-sector pairs): {overall_std:.4f}",
        transform=ax.transAxes, fontsize=8.5, color=INK_MUTED,
        va="bottom", bbox=dict(fc=SURFACE, ec=GRIDLINE, pad=4, lw=0.8))

ax.set_ylabel("Normalised sector median", fontsize=10, color=INK_PRIMARY)
ax.set_xlabel("TESS Sector", fontsize=10, color=INK_PRIMARY)
ax.set_ylim(0.97, 1.03)
ax.set_title(
    "Population distribution per sector  ·  spread = residual instrumental offsets STITCH must correct",
    fontsize=10, color=INK_PRIMARY, loc="left"
)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85, edgecolor=GRIDLINE)

fig.suptitle("TARS Quiet Star Validation — are the selected stars actually flat?",
             fontsize=13, color=INK_PRIMARY, y=1.01, fontweight="bold")

out = "stitch_quiet_star_validation.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\nSaved → {out}")
