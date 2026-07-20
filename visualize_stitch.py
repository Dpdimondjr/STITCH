"""
STITCH Visualization Suite
Three figures explaining the sector stitching problem for a general audience.

  fig_problem.png          — multi-sector PDCSAP light curve showing flux jumps
  fig_detector_heatmap.png — CCD heatmap: gradient model background + measured stars
  fig_gradient_variation.png — how the spatial gradient changes sector to sector

Run as-is: fig_problem and fig_detector_heatmap work immediately.
fig_gradient_variation requires multi_sector_gradient.py to have run first.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import lightkurve as lk

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False})

CAMERA     = 4
CCD        = 1
CACHE      = "./tess_cache"
GRAD_CSV   = f"cam{CAMERA}_ccd{CCD}_gradients.csv"
STARS_CSV  = os.path.join(CACHE, f"all_sectors_cam{CAMERA}_ccd{CCD}.csv")

os.makedirs(CACHE, exist_ok=True)


# ── Figure 1: The Problem ────────────────────────────────────────────────────

def fig_problem(tic_id="TIC 264221449", outfile="fig_problem.png"):
    """
    Multi-sector PDCSAP light curve showing that flux jumps at sector boundaries
    survive the SPOC pipeline. Per-sector median lines make the discontinuities
    explicit for a non-expert reader.
    """
    print(f"[fig_problem] downloading {tic_id}...")
    sr = lk.search_lightcurve(tic_id, mission="TESS", author="SPOC")
    if len(sr) == 0:
        print("  No SPOC light curves found — skipping.")
        return

    lc_col = sr.download_all(download_dir=CACHE)
    pdcsap_lcs = []
    for lc in lc_col:
        try:
            pdcsap_lcs.append(lc.select_flux("pdcsap_flux"))
        except Exception:
            pass

    if not pdcsap_lcs:
        print("  No PDCSAP flux found — skipping.")
        return

    # Single global normalisation — deliberately NOT per-sector so jumps show
    all_flux  = np.concatenate([lc.flux.value for lc in pdcsap_lcs])
    glob_med  = np.nanmedian(all_flux)

    fig, ax = plt.subplots(figsize=(17, 4))

    n_sectors = len(pdcsap_lcs)
    # Label every sector if ≤20, otherwise every 5th
    label_step = 1 if n_sectors <= 20 else 5

    for idx, lc in enumerate(pdcsap_lcs):
        t = lc.time.value
        f = lc.flux.value / glob_med
        sec = lc.meta.get("SECTOR", idx)

        # Alternating sector shading
        if len(t) > 1:
            ax.axvspan(t[0], t[-1], alpha=0.04 if idx % 2 == 0 else 0,
                       color="steelblue", zorder=0)

        ax.plot(t, f, "k.", markersize=0.4, alpha=0.35, rasterized=True, zorder=1)

        # Sector boundary line — thinner and more subtle when there are many
        lw = 0.9 if n_sectors <= 20 else 0.5
        ax.axvline(t[0], color="crimson", alpha=0.35, linewidth=lw, zorder=2)

        # Per-sector median — the horizontal lines that make jumps obvious
        sec_med = np.nanmedian(f)
        ax.hlines(sec_med, t[0], t[-1],
                  colors="steelblue", linewidths=1.8, alpha=0.75, zorder=3)

        # Sector number label — sparse when crowded
        if idx % label_step == 0:
            ax.text(np.nanmean(t), sec_med + 0.0012,
                    f"S{sec}", ha="center", va="bottom",
                    fontsize=6.5, color="0.35", zorder=4)

    ax.set_xlabel("Time (BTJD)", fontsize=11)
    ax.set_ylabel("Normalized PDCSAP Flux", fontsize=11)
    ax.set_title(
        f"{tic_id} · PDCSAP flux across {len(pdcsap_lcs)} TESS sectors\n"
        "Blue lines = per-sector median. Vertical red lines = sector boundaries. "
        "Jumps are instrumental, not astrophysical.",
        fontsize=10
    )

    # Annotate the largest jump for visual emphasis
    meds = [np.nanmedian(lc.flux.value / glob_med) for lc in pdcsap_lcs]
    if len(meds) > 1:
        jumps = [abs(meds[i+1] - meds[i]) for i in range(len(meds) - 1)]
        worst = int(np.argmax(jumps))
        t_boundary = pdcsap_lcs[worst + 1].time.value[0]
        jump_pct   = jumps[worst] * 100
        ax.annotate(
            f"  {jump_pct:.1f}% jump",
            xy=(t_boundary, (meds[worst] + meds[worst + 1]) / 2),
            xytext=(t_boundary + 2, (meds[worst] + meds[worst + 1]) / 2 + 0.007),
            fontsize=9, color="crimson",
            arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.0),
        )

    y_lo = min(meds) - 0.015
    y_hi = max(meds) + 0.020
    ax.set_ylim(y_lo, y_hi)

    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {outfile}")


# ── Figure 2: Detector Heatmap ───────────────────────────────────────────────

def fig_detector_heatmap(outfile="fig_detector_heatmap.png"):
    """
    For every sector in the gradient CSV, draw the CCD as a 2D heatmap.

    Background  — smooth surface from the linear gradient model
                  flux_offset(col, row) = a·col + b·row + intercept
    Foreground  — actual measured stars as labelled circles (if cache exists)
    Arrow       — gradient direction (steepest increase in flux offset)

    The key message: the offset is spatially structured, not random noise.
    """
    if not os.path.exists(GRAD_CSV):
        print(f"[fig_detector_heatmap] {GRAD_CSV} not found — skipping.")
        return

    grad_df = pd.read_csv(GRAD_CSV).sort_values("sector").reset_index(drop=True)
    if grad_df.empty:
        print("[fig_detector_heatmap] gradient CSV is empty — skipping.")
        return

    print(f"[fig_detector_heatmap] {len(grad_df)} sector(s) in gradient CSV.")

    # Load actual star positions from cache if available
    stars_df = _load_stars_cache(grad_df["sector"].tolist())

    # Fine grid across CCD face
    col_lin = np.linspace(44, 2092, 400)
    row_lin = np.linspace(0, 2048, 400)
    COL, ROW = np.meshgrid(col_lin, row_lin)

    # Colour scale: symmetric around 1.0
    # Use actual data spread if we have it; fall back to gradient prediction range
    if stars_df is not None and "flux_offset" in stars_df:
        deviations = (stars_df["flux_offset"] - 1.0).abs()
        spread = max(float(deviations.quantile(0.95)), 0.005)
    else:
        # Predict range from gradient across the full CCD
        spreads = []
        for _, r in grad_df.iterrows():
            Z = r["grad_col"] * COL + r["grad_row"] * ROW + r["intercept"]
            spreads.append(max(abs(Z.max() - 1.0), abs(Z.min() - 1.0)))
        spread = max(max(spreads), 0.005)

    norm = mcolors.TwoSlopeNorm(vmin=1 - spread, vcenter=1.0, vmax=1 + spread)

    n_sec = len(grad_df)
    ncols = min(4, n_sec)
    nrows = int(np.ceil(n_sec / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.8 * ncols, 5.0 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, g in grad_df.iterrows():
        ax     = axes[i]
        sector = int(g["sector"])

        # --- Background: gradient model ---
        Z  = g["grad_col"] * COL + g["grad_row"] * ROW + g["intercept"]
        ax.pcolormesh(col_lin, row_lin, Z,
                      cmap="RdBu_r", norm=norm,
                      shading="auto", rasterized=True, zorder=1, alpha=0.85)

        # --- Gradient direction arrow (centred on CCD) ---
        gc, gr = float(g["grad_col"]), float(g["grad_row"])
        mag = np.hypot(gc, gr)
        if mag > 0:
            scale = 400 / mag          # arrow length in pixels
            cx, cy = 1068.0, 1024.0    # CCD centre
            ax.annotate(
                "", xy=(cx + gc * scale, cy + gr * scale), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.5, mutation_scale=12),
                zorder=5,
            )
            ax.plot(cx, cy, "k.", markersize=4, zorder=5)

        # --- Foreground: actual measured stars ---
        if stars_df is not None:
            sub = stars_df[stars_df["sector"] == sector]
            if not sub.empty:
                sz = sub["tmag"].apply(lambda m: max(50, 320 - m * 22)) \
                     if "tmag" in sub else 80
                ax.scatter(sub["col"], sub["row"],
                           c=sub["flux_offset"], cmap="RdBu_r", norm=norm,
                           s=sz, edgecolors="k", linewidths=0.7, zorder=3)
                for _, r in sub.iterrows():
                    ax.annotate(
                        f"{r['flux_offset']:.3f}",
                        (r["col"], r["row"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=5, color="0.15", zorder=4,
                    )

        ax.set_xlim(44, 2092)
        ax.set_ylim(0, 2048)
        n_label = f"  n={int(g['n_stars'])} stars" if "n_stars" in g else ""
        ax.set_title(f"Sector {sector}{n_label}", fontsize=10)
        ax.set_xlabel("Column (px)", fontsize=8)
        ax.set_ylabel("Row (px)", fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.12, linewidth=0.3)

    for j in range(n_sec, len(axes)):
        axes[j].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n_sec], shrink=0.55, pad=0.02)
    cbar.set_label("Flux offset  (sector median / star global median)", fontsize=10)

    star_note = "Circles: measured stable stars  ·  " if stars_df is not None else ""
    fig.suptitle(
        f"Flux offset spatial structure — Camera {CAMERA}, CCD {CCD}\n"
        f"Background: linear gradient model  ·  {star_note}Arrow: gradient direction\n"
        "Blue = below star's global median,  Red = above",
        fontsize=11,
    )

    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {outfile}")


# ── Figure 3: Gradient Variation Across Sectors ──────────────────────────────

def fig_gradient_variation(outfile="fig_gradient_variation.png"):
    """
    Show that the linear gradient fit changes sector to sector —
    evidence that a single static correction is insufficient and a
    per-sector learned model is needed.

    Requires gradient data from at least 2 sectors.
    """
    if not os.path.exists(GRAD_CSV):
        print(f"[fig_gradient_variation] {GRAD_CSV} not found — skipping.")
        return

    grad_df = pd.read_csv(GRAD_CSV).sort_values("sector").reset_index(drop=True)
    if len(grad_df) < 2:
        print(f"[fig_gradient_variation] only {len(grad_df)} sector(s) — "
              "need ≥2 to show variation. Run multi_sector_gradient.py first.")
        return

    sectors = grad_df["sector"].values.astype(int)
    gc      = grad_df["grad_col"].values
    gr      = grad_df["grad_row"].values
    ic      = grad_df["intercept"].values

    # Scale gradients to % change across full CCD width/height
    gc_pct = gc * (2092 - 44)   * 100   # % over full column span
    gr_pct = gr * 2048          * 100   # % over full row span
    mag    = np.hypot(gc_pct, gr_pct)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    bar_kw = dict(edgecolor="k", linewidth=0.5, width=0.7)

    # Panel A: total gradient magnitude
    axes[0].bar(sectors, mag, color="steelblue", **bar_kw)
    axes[0].set_xlabel("Sector", fontsize=11)
    axes[0].set_ylabel("Total gradient magnitude\n(% flux change across CCD)", fontsize=10)
    axes[0].set_title("How strong is the\nspatial pattern?", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)

    # Panel B: component breakdown (col vs row)
    x   = np.arange(len(sectors))
    w   = 0.35
    axes[1].bar(x - w/2, gc_pct, w, label="col gradient", color="steelblue", **bar_kw)
    axes[1].bar(x + w/2, gr_pct, w, label="row gradient", color="tomato",    **bar_kw)
    axes[1].axhline(0, color="k", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sectors, rotation=45, fontsize=8)
    axes[1].set_xlabel("Sector", fontsize=11)
    axes[1].set_ylabel("Gradient (% / CCD span)", fontsize=10)
    axes[1].set_title("Which direction does\nthe offset run?", fontsize=11)
    axes[1].legend(fontsize=9)

    # Panel C: overall flux level per sector (intercept at CCD origin)
    axes[2].plot(sectors, (ic - 1.0) * 100, "o-",
                 color="darkorange", markersize=7, linewidth=1.8)
    axes[2].axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[2].fill_between(sectors, (ic - 1.0) * 100, 0,
                         alpha=0.15, color="darkorange")
    axes[2].set_xlabel("Sector", fontsize=11)
    axes[2].set_ylabel("Overall flux offset (%)\n  relative to global median", fontsize=10)
    axes[2].set_title("Does the whole CCD shift\nup or down each sector?", fontsize=11)
    axes[2].tick_params(axis="x", rotation=45)

    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Gradient fit parameters change sector-to-sector — Camera {CAMERA}, CCD {CCD}\n"
        "A single static correction cannot account for this variation: "
        "a per-sector learned model is necessary.",
        fontsize=11,
    )

    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {outfile}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_stars_cache(sectors):
    """
    Try to find actual (star, sector, col, row, flux_offset) data.
    Returns a DataFrame or None if nothing is cached yet.
    """
    # Option 1: combined file from multi_sector_gradient.py
    if os.path.exists(STARS_CSV):
        df = pd.read_csv(STARS_CSV)
        print(f"  Loaded {len(df)} (star, sector) pairs from {STARS_CSV}")
        return df

    # Option 2: assemble from per-sector cache files
    dfs = []
    for s in sectors:
        p = os.path.join(CACHE, f"sector_result_s{int(s):04d}_cam{CAMERA}_ccd{CCD}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Assembled {len(df)} pairs from per-sector cache files.")
        return df

    print("  No star position cache found — showing gradient model only.")
    return None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== STITCH Visualization Suite ===\n")

    fig_problem()
    print()

    fig_detector_heatmap()
    print()

    fig_gradient_variation()

    print("\nDone.")
