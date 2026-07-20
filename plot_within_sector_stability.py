"""
Compare within-sector photometric stability: TARS quiet (sys>0.95) vs non-quiet (sys<0.3).

Metric: CDPP1_0 (1-hour combined differential photometric precision, ppm),
computed by SPOC and stored in the FITS header. This is the standard measure
of within-sector scatter — what TARS sys_score is actually classifying.

Since CDPP scales with magnitude, we compare at matched Tmag ranges.
"""

import numpy as np
import pandas as pd
import lightkurve as lk
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
BLUE        = "#2a78d6"
BLUE_LIGHT  = "#86b6ef"
RED         = "#e34948"
RED_LIGHT   = "#f5a09f"

# ── 1. Quiet stars from parquet ───────────────────────────────────────────────

print("Loading quiet stars from parquet...")
df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["cdpp1_0", "tmag"])
df = df[(df["cdpp1_0"] > 0) & (df["cdpp1_0"] < 50000)]

# One record per star (median CDPP across their sectors)
quiet = (df.groupby("tic_id")
           .agg(cdpp=("cdpp1_0", "median"), tmag=("tmag", "median"))
           .reset_index())
print(f"  Quiet stars: {len(quiet):,}  (median CDPP per star)")

# ── 2. Non-quiet stars: download and extract CDPP from FITS header ───────────

print("\nLoading non-quiet TICs from TARS...")
import fsspec, pyarrow.feather as feather
ZENODO_URL = "https://zenodo.org/api/records/19917941/files/tars_table_4.feather/content"
COLS = ["TICID", "systematic_score", "Tmag", "sector"]
fs = fsspec.filesystem("http")
with fs.open(ZENODO_URL, "rb") as f:
    t4 = feather.read_table(f, columns=COLS).to_pandas()

nonquiet_pairs = t4[(t4["systematic_score"] < 0.3) & (t4["Tmag"] < 13.0)]
nonquiet_agg = (nonquiet_pairs.groupby("TICID")
                .agg(n_sectors=("sector", "count"), tmag=("Tmag", "mean"))
                .reset_index())
nonquiet_agg = nonquiet_agg[nonquiet_agg["n_sectors"] >= 3]
sample = nonquiet_agg.sample(200, random_state=42)["TICID"].tolist()
print(f"  Sampling {len(sample)} non-quiet TICs...")

def get_cdpp(tic_id, cache_dir="./tess_cache"):
    """Download one sector for a star and return (tmag, cdpp1_0)."""
    try:
        sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
        if len(sr) == 0:
            return None
        try:
            sr = sr[sr.exptime.value >= 100]
        except Exception:
            pass
        if len(sr) == 0:
            return None
        # Just download one sector — CDPP is per-sector from header
        lc = sr[-1].download(download_dir=cache_dir)
        if lc is None:
            return None
        cdpp = lc.meta.get("CDPP1_0")
        tmag = lc.meta.get("TESSMAG")
        if cdpp is None or tmag is None:
            return None
        cdpp = float(cdpp)
        tmag = float(tmag)
        if cdpp <= 0 or cdpp > 50000:
            return None
        return (tmag, cdpp)
    except Exception:
        return None

nonquiet_records = []
for i, tic in enumerate(sample):
    r = get_cdpp(tic)
    if r is not None:
        nonquiet_records.append({"tmag": r[0], "cdpp": r[1]})
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(sample)}  valid: {len(nonquiet_records)}")

nonquiet = pd.DataFrame(nonquiet_records)
print(f"  Non-quiet stars with valid CDPP: {len(nonquiet)}")

# ── 3. Matched Tmag ranges ────────────────────────────────────────────────────

TMAG_BINS = [(7, 9), (9, 11), (11, 13)]
BIN_LABELS = ["Tmag 7–9", "Tmag 9–11", "Tmag 11–13"]

print(f"\n=== Within-sector CDPP1_0 (ppm) by Tmag bin ===")
print(f"{'Tmag bin':<14} {'Quiet median':>14} {'Non-quiet median':>18} {'Ratio':>8}")
print("  " + "─" * 58)
for (lo, hi), label in zip(TMAG_BINS, BIN_LABELS):
    q = quiet[(quiet["tmag"] >= lo) & (quiet["tmag"] < hi)]["cdpp"]
    nq = nonquiet[(nonquiet["tmag"] >= lo) & (nonquiet["tmag"] < hi)]["cdpp"]
    if len(q) > 5 and len(nq) > 2:
        ratio = np.median(nq) / np.median(q)
        print(f"  {label:<12} {np.median(q):>14.0f} {np.median(nq):>18.0f} {ratio:>8.1f}×")

# ── 4. Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=SURFACE,
                          gridspec_kw={"wspace": 0.38})

for ax, (lo, hi), label in zip(axes, TMAG_BINS, BIN_LABELS):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, zorder=0)

    q  = quiet[(quiet["tmag"] >= lo) & (quiet["tmag"] < hi)]["cdpp"].values
    nq = nonquiet[(nonquiet["tmag"] >= lo) & (nonquiet["tmag"] < hi)]["cdpp"].values

    if len(q) < 3 or len(nq) < 2:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", color=INK_MUTED)
        ax.set_title(label, fontsize=10, color=INK_PRIMARY, loc="left")
        continue

    # Log-spaced bins (CDPP is log-distributed)
    lo_b = min(q.min(), nq.min()) * 0.8
    hi_b = max(np.percentile(q, 99), np.percentile(nq, 99)) * 1.2
    bins = np.logspace(np.log10(max(lo_b, 1)), np.log10(hi_b), 45)

    ax.hist(nq, bins=bins, color=RED,  alpha=0.55,
            label=f"Non-quiet (sys<0.30)\nn={len(nq)}", edgecolor=SURFACE, lw=0.3, zorder=2)
    ax.hist(q,  bins=bins, color=BLUE, alpha=0.65,
            label=f"Quiet (sys>0.95)\nn={len(q):,}", edgecolor=SURFACE, lw=0.3, zorder=3)

    ymax = ax.get_ylim()[1]
    ax.axvline(np.median(q),  color=BLUE, lw=2.0, ls="--", zorder=4)
    ax.axvline(np.median(nq), color=RED,  lw=2.0, ls="--", zorder=4)

    ax.text(np.median(q)  * 1.05, ymax * 0.88,
            f"{np.median(q):.0f}", color=BLUE, fontsize=8.5, va="top")
    ax.text(np.median(nq) * 1.05, ymax * 0.72,
            f"{np.median(nq):.0f}", color=RED,  fontsize=8.5, va="top")

    ratio = np.median(nq) / np.median(q) if np.median(q) > 0 else float("nan")
    ax.set_xscale("log")
    ax.set_xlabel("CDPP 1-hour (ppm)", fontsize=10, color=INK_PRIMARY)
    ax.set_ylabel("Number of stars", fontsize=10, color=INK_PRIMARY)
    ax.set_title(f"{label}  ·  non-quiet is {ratio:.1f}× noisier",
                 fontsize=10, color=INK_PRIMARY, loc="left")
    ax.legend(fontsize=8, framealpha=0.85, edgecolor=GRIDLINE)

fig.suptitle(
    "Within-sector photometric stability: TARS quiet (sys>0.95) vs non-quiet (sys<0.30)\n"
    "CDPP1_0 = SPOC 1-hour scatter — the quantity TARS sys_score directly classifies",
    fontsize=11, color=INK_PRIMARY, y=1.02
)

out = "stitch_within_sector_stability.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\nSaved → {out}")
