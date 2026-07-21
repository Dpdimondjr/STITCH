"""
Side-by-side comparison: TARS quiet stars (sys_score > 0.95) vs rejected stars
(sys_score < 0.3) to validate that Andrew's selection criterion is actually
separating quiet from variable stars.

For each group, computes the std of normalised sector medians per star
(= how flat the star is across sectors) and plots the distributions.
"""

import numpy as np
import pandas as pd
import lightkurve as lk
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
BLUE        = "#2a78d6"    # series-1: quiet stars
BLUE_LIGHT  = "#86b6ef"
RED         = "#e34948"    # series-6: non-quiet stars
RED_LIGHT   = "#f5a09f"

# ── 1. Sample non-quiet stars from TARS table 4 ───────────────────────────────

print("Loading TARS table 4 via URL column projection...")
import fsspec, pyarrow.feather as feather
ZENODO_URL = "https://zenodo.org/api/records/19917941/files/tars_table_4.feather/content"
COLS = ["TICID", "systematic_score", "camera", "ccd", "sector", "Tmag"]
fs = fsspec.filesystem("http")
with fs.open(ZENODO_URL, "rb") as f:
    t4 = feather.read_table(f, columns=COLS).to_pandas()
print(f"  {len(t4):,} TIC-sector pairs")

# Non-quiet: sys_score < 0.3, Tmag < 13 (same brightness cut as quiet group)
nonquiet_pairs = t4[(t4["systematic_score"] < 0.3) & (t4["Tmag"] < 13.0)]
nonquiet_agg = (nonquiet_pairs.groupby("TICID")
                .agg(n_sectors=("sector", "count"),
                     mean_sys=("systematic_score", "mean"))
                .reset_index())
nonquiet_agg = nonquiet_agg[nonquiet_agg["n_sectors"] >= 5]
print(f"  Non-quiet TICs (sys<0.3, ≥5 sectors, Tmag<13): {len(nonquiet_agg):,}")

# Sample 100 evenly spread across cameras
rng = np.random.default_rng(42)
sample_nonquiet = nonquiet_agg.sample(min(120, len(nonquiet_agg)), random_state=42)["TICID"].tolist()

# ── 2. Download non-quiet star light curves and compute scatter ───────────────

def star_sector_scatter(tic_id, cache_dir="./tess_cache"):
    """Download SPOC LCs for a star, return std of normalised sector medians."""
    try:
        sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
        if len(sr) == 0:
            return None
        # 200s cadence only
        try:
            sr = sr[sr.exptime.value >= 100]
        except Exception:
            pass
        if len(sr) < 3:
            return None
        # Cap at 12 sectors for speed
        if len(sr) > 12:
            sr = sr[-12:]
        lcs = sr.download_all(download_dir=cache_dir)
        meds = []
        for lc in lcs:
            try:
                sec = lc.meta.get("SECTOR")
                if sec is None or not (1 <= int(sec) <= 200):
                    continue
                med = float(np.nanmedian(lc.flux.value))
                if np.isfinite(med) and med > 0:
                    meds.append(med)
            except Exception:
                continue
        if len(meds) < 3:
            return None
        norm = np.array(meds) / np.mean(meds)
        return float(norm.std())
    except Exception:
        return None

print(f"\nDownloading {len(sample_nonquiet)} non-quiet stars...")
nonquiet_stds = []
for i, tic in enumerate(sample_nonquiet):
    s = star_sector_scatter(tic)
    if s is not None:
        nonquiet_stds.append(s)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(sample_nonquiet)}  collected {len(nonquiet_stds)} valid")

print(f"  Non-quiet stars with valid data: {len(nonquiet_stds)}")

# ── 3. Quiet stars from existing training data ────────────────────────────────

print("\nComputing quiet star scatter from training_data.parquet...")
df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["sector_median", "sector", "tic_id"])
df = df[(df["sector_median"] > 0) & (df["sector"] >= 1) & (df["sector"] <= 200)]

global_means = df.groupby("tic_id")["sector_median"].transform("mean")
df["norm_median"] = df["sector_median"] / global_means

# Only stars with ≥5 sectors (same cut as non-quiet)
sec_counts = df.groupby("tic_id")["sector"].count()
good_tics  = sec_counts[sec_counts >= 5].index
df_good    = df[df["tic_id"].isin(good_tics)]
quiet_stds = df_good.groupby("tic_id")["norm_median"].std().values
print(f"  Quiet stars: {len(quiet_stds):,}")

nonquiet_stds = np.array(nonquiet_stds)
quiet_stds    = np.array(quiet_stds)

print(f"\n=== Summary ===")
print(f"  Quiet   (sys>0.95): median std = {np.median(quiet_stds):.4f},  mean = {np.mean(quiet_stds):.4f}")
print(f"  NonQuiet(sys<0.30): median std = {np.median(nonquiet_stds):.4f},  mean = {np.mean(nonquiet_stds):.4f}")
print(f"  Non-quiet is {np.median(nonquiet_stds)/np.median(quiet_stds):.1f}x more variable (median std)")

# ── 4. Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE,
                          gridspec_kw={"wspace": 0.35})

BINS = np.linspace(0, 0.12, 55)

for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)

# Panel 1: overlapping histograms
ax = axes[0]
ax.hist(nonquiet_stds, bins=BINS, color=RED,  alpha=0.55, label=f"Non-quiet  (sys<0.30, n={len(nonquiet_stds)})",
        edgecolor=SURFACE, linewidth=0.4, zorder=2)
ax.hist(quiet_stds,    bins=BINS, color=BLUE, alpha=0.65, label=f"Quiet  (sys>0.95, n={len(quiet_stds):,})",
        edgecolor=SURFACE, linewidth=0.4, zorder=3)

ax.axvline(np.median(quiet_stds),    color=BLUE, lw=2.0, ls="--", zorder=4)
ax.axvline(np.median(nonquiet_stds), color=RED,  lw=2.0, ls="--", zorder=4)

ax.set_xlabel("Std of normalised sector medians per star", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Number of stars", fontsize=10, color=INK_PRIMARY)
ax.set_title("Distribution of per-star sector scatter", fontsize=10, color=INK_PRIMARY, loc="left")
ax.legend(fontsize=8.5, framealpha=0.85, edgecolor=GRIDLINE)
ax.text(np.median(quiet_stds) + 0.001,    ax.get_ylim()[1]*0.92,
        f"med={np.median(quiet_stds):.4f}", color=BLUE, fontsize=8)
ax.text(np.median(nonquiet_stds) + 0.001, ax.get_ylim()[1]*0.75,
        f"med={np.median(nonquiet_stds):.4f}", color=RED, fontsize=8)

# Panel 2: CDF
ax = axes[1]
for stds, color, label in [
    (quiet_stds,    BLUE, f"Quiet  (sys>0.95)"),
    (nonquiet_stds, RED,  f"Non-quiet  (sys<0.30)"),
]:
    xs = np.sort(stds)
    ys = np.arange(1, len(xs)+1) / len(xs)
    ax.plot(xs, ys, color=color, lw=2.2, label=label)

ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=":", alpha=0.7)
ax.axvline(np.median(quiet_stds),    color=BLUE, lw=1.5, ls="--", alpha=0.8)
ax.axvline(np.median(nonquiet_stds), color=RED,  lw=1.5, ls="--", alpha=0.8)

# Annotate fraction of quiet stars within tight scatter
frac_tight = (quiet_stds < 0.01).mean() * 100
ax.text(0.62, 0.22, f"{frac_tight:.0f}% of quiet stars\nhave std < 0.01",
        transform=ax.transAxes, fontsize=8.5, color=BLUE,
        bbox=dict(fc=SURFACE, ec=GRIDLINE, pad=4, lw=0.8))

ax.set_xlabel("Std of normalised sector medians per star", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Cumulative fraction of stars", fontsize=10, color=INK_PRIMARY)
ax.set_title("Cumulative distribution", fontsize=10, color=INK_PRIMARY, loc="left")
ax.set_xlim(0, 0.12)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8.5, framealpha=0.85, edgecolor=GRIDLINE)

fig.suptitle(
    "TARS selection validation — do quiet stars (sys_score > 0.95) actually have less inter-sector scatter?\n"
    f"Quiet median std = {np.median(quiet_stds):.4f}  ·  "
    f"Non-quiet median std = {np.median(nonquiet_stds):.4f}  ·  "
    f"Ratio = {np.median(nonquiet_stds)/np.median(quiet_stds):.1f}×",
    fontsize=11, color=INK_PRIMARY, y=1.02
)

out = "stitch_tars_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\nSaved → {out}")
